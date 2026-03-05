"""Chain operations: context managers, iteration, filtering, gathering."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from inspect import isawaitable
from typing import Any

from ._core import (
  Link,
  Null,
  QuentException,
  _Break,
  _ControlFlowSignal,
  _evaluate_value,
  _handle_break_exc,
  _Return,
  _set_link_temp_args,
)


def _make_with(link: Link, ignore_result: bool) -> Callable[[Any], Any]:
  """Create a context manager operation for use in a chain."""

  async def _to_async(current_value: Any, body_result: Any, outer_value: Any) -> Any:
    try:
      body_result = await body_result
    except BaseException as exc:
      try:
        suppress = current_value.__exit__(type(exc), exc, exc.__traceback__)
        if isawaitable(suppress):
          suppress = await suppress
      except BaseException as exit_exc:
        raise exit_exc from exc
      if not suppress:
        raise
      return outer_value if ignore_result else None
    else:
      exit_result = current_value.__exit__(None, None, None)
      if isawaitable(exit_result):
        await exit_result
      if ignore_result:
        return outer_value
      return body_result

  async def _full_async(current_value: Any) -> Any:
    outer_value = current_value
    result = Null
    async with current_value as ctx:
      result = _evaluate_value(link, ctx)
      if isawaitable(result):
        result = await result
    if result is Null:
      return outer_value if ignore_result else None
    if ignore_result:
      return outer_value
    return result

  def _with_op(current_value: Any) -> Any:
    # Capture the context manager itself before __enter__ may rebind it. When
    # ignore_result=True, the chain returns this original value, not the __enter__ result.
    outer_value = current_value
    if hasattr(current_value, '__aenter__'):
      return _full_async(current_value)
    ctx = current_value.__enter__()
    try:
      result = _evaluate_value(link, ctx)
      if isawaitable(result):
        return _to_async(current_value, result, outer_value)
    except BaseException as exc:
      try:
        suppress = current_value.__exit__(type(exc), exc, exc.__traceback__)
      except BaseException as exit_exc:
        raise exit_exc from exc
      if not suppress:
        raise
      return outer_value if ignore_result else None
    else:
      current_value.__exit__(None, None, None)
      if ignore_result:
        return outer_value
      return result

  _with_op._quent_op = 'with'  # type: ignore[attr-defined]
  _with_op._ignore_result = ignore_result  # type: ignore[attr-defined]
  return _with_op


# Defined as module-level functions (not methods) because Python generator functions
# create generator objects at call time — a method would bind `self` unnecessarily
# and complicate the generator's closure. Keeping them external is cleaner.
def _sync_generator(
  chain_run: Callable[..., Any], run_args: tuple[Any, ...], fn: Callable[[Any], Any] | None, ignore_result: bool
) -> Iterator[Any]:
  """Synchronous generator over chain output."""
  try:
    for item in chain_run(*run_args):
      if fn is None:
        yield item
      else:
        result = fn(item)
        if ignore_result:
          yield item
        else:
          yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using .return_() inside an iterator is not allowed.') from None


async def _aiter_wrap(sync_iter: Iterator[Any]) -> AsyncIterator[Any]:
  """Wrap a synchronous iterator as an async iterator."""
  for item in sync_iter:
    yield item


async def _async_generator(
  chain_run: Callable[..., Any], run_args: tuple[Any, ...], fn: Callable[[Any], Any] | None, ignore_result: bool
) -> AsyncIterator[Any]:
  """Asynchronous generator over chain output."""
  iterator = chain_run(*run_args)
  if isawaitable(iterator):
    iterator = await iterator
  if not hasattr(iterator, '__aiter__'):
    iterator = _aiter_wrap(iterator)
  try:
    async for item in iterator:
      if fn is None:
        yield item
      else:
        result = fn(item)
        if isawaitable(result):
          result = await result
        if ignore_result:
          yield item
        else:
          yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using .return_() inside an iterator is not allowed.') from None


class _Generator:
  """Wraps chain output as a sync/async iterable.

  Created by Chain.iterate(). Supports both __iter__ and __aiter__,
  choosing the appropriate generator at iteration time.
  """

  __slots__ = ('_chain_run', '_fn', '_ignore_result', '_run_args')

  def __init__(self, chain_run: Callable[..., Any], fn: Callable[[Any], Any] | None, ignore_result: bool) -> None:
    self._chain_run = chain_run
    self._fn = fn
    self._ignore_result = ignore_result
    self._run_args: tuple[Any, tuple[Any, ...], dict[str, Any]] = (Null, (), {})

  def __call__(self, v: Any = Null, *args: Any, **kwargs: Any) -> _Generator:
    g = _Generator(self._chain_run, self._fn, self._ignore_result)
    g._run_args = (v, args, kwargs)
    return g

  def __iter__(self) -> Iterator[Any]:
    return _sync_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __aiter__(self) -> AsyncIterator[Any]:
    return _async_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __repr__(self) -> str:
    return '<Quent._Generator>'


def _make_foreach(link: Link, ignore_result: bool) -> Callable[[Any], Any]:
  """Create a foreach iteration operation for use in a chain."""
  fn: Callable[[Any], Any] = link.v

  # Three-tier sync/async pattern used by foreach, filter, and with_:
  #   _foreach_op: Pure sync fast path. Uses manual `while True` + `next()` instead
  #     of `for` loop so the iterator can be handed off to _to_async mid-iteration.
  #   _to_async: Sync-to-async handoff. Called when a sync iteration discovers that
  #     `fn` returned a coroutine. Continues the SAME iterator from where sync left off.
  #   _full_async: Pure async path for async iterables (`async for`).

  # Picks up a sync iteration that hit an awaitable result. Receives the live
  # iterator, the current item, and the pending awaitable to continue from.
  async def _to_async(iterator: Iterator[Any], item: Any, result: Any, lst: list[Any]) -> list[Any]:
    try:
      while True:
        if isawaitable(result):
          result = await result
        if ignore_result:
          lst.append(item)
        else:
          lst.append(result)
        item = next(iterator)
        result = fn(item)
    except _Break as exc:
      result = _handle_break_exc(exc, lst)
      if isawaitable(result):
        return await result  # type: ignore[no-any-return]
      return result  # type: ignore[no-any-return]
    except StopIteration:
      return lst
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item)
      raise

  async def _full_async(current_value: Any) -> list[Any]:
    lst = []
    item = Null
    try:
      async for item in current_value:
        result = fn(item)
        if isawaitable(result):
          result = await result
        if ignore_result:
          lst.append(item)
        else:
          lst.append(result)
      return lst
    except _Break as exc:
      result = _handle_break_exc(exc, lst)
      if isawaitable(result):
        return await result  # type: ignore[no-any-return]
      return result  # type: ignore[no-any-return]
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item)
      raise

  def _foreach_op(current_value: Any) -> Any:
    if hasattr(current_value, '__aiter__'):
      return _full_async(current_value)
    lst: list[Any] = []
    it = iter(current_value)
    item = Null
    try:
      while True:
        item = next(it)
        result = fn(item)
        if isawaitable(result):
          return _to_async(it, item, result, lst)
        if ignore_result:
          lst.append(item)
        else:
          lst.append(result)
    except _Break as exc:
      return _handle_break_exc(exc, lst)
    except StopIteration:
      return lst
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item)
      raise

  # Attach metadata as function attributes — a Python hack that lets the traceback
  # formatter identify the operation type without needing a class wrapper.
  _foreach_op._quent_op = 'foreach'  # type: ignore[attr-defined]
  _foreach_op._ignore_result = ignore_result  # type: ignore[attr-defined]
  return _foreach_op


def _make_filter(link: Link) -> Callable[[Any], Any]:
  """Create a filter operation for use in a chain."""
  fn: Callable[[Any], Any] = link.v

  async def _to_async(iterator: Iterator[Any], item: Any, result: Any, lst: list[Any]) -> list[Any]:
    try:
      while True:
        if isawaitable(result):
          result = await result
        if result:
          lst.append(item)
        item = next(iterator)
        result = fn(item)
    except StopIteration:
      return lst
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item)
      raise

  async def _full_async(current_value: Any) -> list[Any]:
    lst = []
    item = Null
    try:
      async for item in current_value:
        result = fn(item)
        if isawaitable(result):
          result = await result
        if result:
          lst.append(item)
      return lst
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item)
      raise

  def _filter_op(current_value: Any) -> Any:
    if hasattr(current_value, '__aiter__'):
      return _full_async(current_value)
    lst: list[Any] = []
    it = iter(current_value)
    item = Null
    try:
      while True:
        item = next(it)
        result = fn(item)
        if isawaitable(result):
          return _to_async(it, item, result, lst)
        if result:
          lst.append(item)
    except StopIteration:
      return lst
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item)
      raise

  _filter_op._quent_op = 'filter'  # type: ignore[attr-defined]
  return _filter_op


def _make_gather(fns: tuple[Callable[[Any], Any], ...]) -> Callable[[Any], Any]:
  """Create a gather operation for use in a chain."""

  async def _to_async(results: list[Any]) -> list[Any]:
    coros = []
    indices = []
    for i, r in enumerate(results):
      if isawaitable(r):
        coros.append(r)
        indices.append(i)
    resolved = await asyncio.gather(*coros)
    for idx, val in zip(indices, resolved):
      results[idx] = val
    return results

  def _gather_op(current_value: Any) -> Any:
    results = []
    has_coro = False
    try:
      for fn in fns:
        result = fn(current_value)
        results.append(result)
        if isawaitable(result):
          has_coro = True
    except BaseException:
      # If a fn raises during setup, close any already-created coroutines to avoid
      # "coroutine was never awaited" RuntimeWarning.
      for r in results:
        if hasattr(r, 'close'):
          r.close()
      raise
    if has_coro:
      return _to_async(results)
    return results

  _gather_op._quent_op = 'gather'  # type: ignore[attr-defined]
  return _gather_op
