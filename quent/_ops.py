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
from ._traceback import _modify_traceback


def _make_with(link: Link, ignore_result: bool) -> Callable[[Any], Any]:
  """Create a context manager operation for use in a chain."""
  # L8: Extract callable/args/kwargs from link at factory time (build-time capture).
  # Link objects are immutable after creation, so this is equivalent to runtime access.
  fn = link.v
  fn_args = link.args
  fn_kwargs = link.kwargs

  def _evaluate_body(ctx: Any) -> Any:
    """Evaluate the body callable with extracted link values."""
    v, args, kwargs = fn, fn_args, fn_kwargs

    if link.is_chain:
      if args and args[0] is ...:
        return v._run(Null, None, None)
      if args or kwargs:
        return v._run(args[0] if args else ctx, args[1:] if args else None, kwargs or {})
      return v._run(ctx, None, None)

    if args and args[0] is ...:
      return v()
    if args or kwargs:
      return v(*(args or ()), **(kwargs or {}))
    if callable(v):
      return v(ctx) if ctx is not Null else v()
    return v

  async def _to_async(current_value: Any, body_result: Any, outer_value: Any, ctx: Any) -> Any:
    try:
      body_result = await body_result
    except _ControlFlowSignal:
      exit_result = current_value.__exit__(None, None, None)
      if isawaitable(exit_result):
        await exit_result
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, ctx=ctx)
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
    signal = None
    async with current_value as ctx:
      try:
        result = _evaluate_body(ctx)
        if isawaitable(result):
          result = await result
      except _ControlFlowSignal as s:
        signal = s
      except BaseException as exc:
        result = Null  # Reset so suppressed exceptions don't return stale values
        _set_link_temp_args(exc, link, ctx=ctx)
        raise
    if signal is not None:
      raise signal
    if result is Null:
      return outer_value if ignore_result else None
    if ignore_result:
      return outer_value
    return result

  async def _await_exit_suppress(suppress: Any, exc: BaseException, outer_value: Any) -> Any:
    try:
      if await suppress:
        return outer_value if ignore_result else None
    except BaseException as exit_exc:
      raise exit_exc from exc
    raise exc

  async def _await_exit_success(exit_result: Any, outer_value: Any, result: Any) -> Any:
    await exit_result
    if ignore_result:
      return outer_value
    return result

  def _with_op(current_value: Any) -> Any:
    # Capture the context manager itself before __enter__ may rebind it. When
    # ignore_result=True, the chain returns this original value, not the __enter__ result.
    outer_value = current_value
    # L10: Prefer __enter__ over __aenter__ — start sync, transition to async only when forced.
    if hasattr(current_value, '__enter__'):
      ctx = current_value.__enter__()
      try:
        result = _evaluate_body(ctx)
        if isawaitable(result):
          return _to_async(current_value, result, outer_value, ctx)
      except _ControlFlowSignal:
        current_value.__exit__(None, None, None)
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, link, ctx=ctx)
        try:
          suppress = current_value.__exit__(type(exc), exc, exc.__traceback__)
        except BaseException as exit_exc:
          raise exit_exc from exc
        if isawaitable(suppress):
          return _await_exit_suppress(suppress, exc, outer_value)
        if not suppress:
          raise
        return outer_value if ignore_result else None
      else:
        exit_result = current_value.__exit__(None, None, None)
        if isawaitable(exit_result):
          return _await_exit_success(exit_result, outer_value, result)
        if ignore_result:
          return outer_value
        return result
    elif hasattr(current_value, '__aenter__'):
      return _full_async(current_value)
    else:
      raise TypeError(f'{type(current_value).__name__!r} object does not support the context manager protocol')

  _with_op._quent_op = 'with'  # type: ignore[attr-defined]
  _with_op._ignore_result = ignore_result  # type: ignore[attr-defined]
  return _with_op


# Defined as module-level functions (not methods) because Python generator functions
# create generator objects at call time — a method would bind `self` unnecessarily
# and complicate the generator's closure. Keeping them external is cleaner.
def _sync_generator(
  chain_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  chain: Any = None,
  link: Any = None,
) -> Iterator[Any]:
  """Synchronous generator over chain output."""
  # M4: Check if chain_run returns a coroutine and close it to prevent ResourceWarning.
  result = chain_run(*run_args)
  if isawaitable(result):
    if hasattr(result, 'close'):
      result.close()
    raise TypeError("Cannot use sync iteration on an async chain; use 'async for' instead")
  idx = 0
  try:
    for item in result:
      if fn is None:
        yield item
      else:
        try:
          fn_result = fn(item)
          if isawaitable(fn_result):
            if hasattr(fn_result, 'close'):
              fn_result.close()
            raise TypeError(
              f'iterate() callback {fn!r} returned a coroutine. '
              f'Use "async for" with __aiter__ instead of "for" with __iter__.'
            )
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          if link is not None:
            _set_link_temp_args(exc, link, item=item, index=idx)
            method = 'iterate_do' if ignore_result else 'iterate'
            _modify_traceback(exc, chain, link, chain.root_link if chain else None, extra_links=[(link, method)])
          raise
        if ignore_result:
          yield item
        else:
          yield fn_result
      idx += 1
  except _Break:
    return
  except _Return:
    raise QuentException('Using .return_() inside an iterator is not allowed.') from None


async def _aiter_wrap(sync_iter: Iterator[Any]) -> AsyncIterator[Any]:
  """Wrap a synchronous iterator as an async iterator."""
  for item in sync_iter:
    yield item


async def _async_generator(
  chain_run: Callable[..., Any],
  run_args: tuple[Any, ...],
  fn: Callable[[Any], Any] | None,
  ignore_result: bool,
  chain: Any = None,
  link: Any = None,
) -> AsyncIterator[Any]:
  """Asynchronous generator over chain output."""
  iterator = chain_run(*run_args)
  if isawaitable(iterator):
    iterator = await iterator
  if not hasattr(iterator, '__aiter__'):
    iterator = _aiter_wrap(iterator)
  idx = 0
  try:
    async for item in iterator:
      if fn is None:
        yield item
      else:
        try:
          result = fn(item)
          if isawaitable(result):
            result = await result
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          if link is not None:
            _set_link_temp_args(exc, link, item=item, index=idx)
            method = 'iterate_do' if ignore_result else 'iterate'
            _modify_traceback(exc, chain, link, chain.root_link if chain else None, extra_links=[(link, method)])
          raise
        if ignore_result:
          yield item
        else:
          yield result
      idx += 1
  except _Break:
    return
  except _Return:
    raise QuentException('Using .return_() inside an iterator is not allowed.') from None


class _Generator:
  """Wraps chain output as a sync/async iterable.

  Created by Chain.iterate(). Supports both __iter__ and __aiter__,
  choosing the appropriate generator at iteration time.
  """

  __slots__ = ('_chain', '_chain_run', '_fn', '_ignore_result', '_link', '_run_args')

  def __init__(
    self,
    chain_run: Callable[..., Any],
    fn: Callable[[Any], Any] | None,
    ignore_result: bool,
    chain: Any = None,
    link: Any = None,
  ) -> None:
    self._chain_run = chain_run
    self._fn = fn
    self._ignore_result = ignore_result
    self._chain = chain
    self._link = link
    self._run_args: tuple[Any, tuple[Any, ...], dict[str, Any]] = (Null, (), {})

  def __call__(self, v: Any = Null, *args: Any, **kwargs: Any) -> _Generator:
    g = _Generator(self._chain_run, self._fn, self._ignore_result, self._chain, self._link)
    g._run_args = (v, args, kwargs)
    return g

  def __iter__(self) -> Iterator[Any]:
    return _sync_generator(self._chain_run, self._run_args, self._fn, self._ignore_result, self._chain, self._link)

  def __aiter__(self) -> AsyncIterator[Any]:
    return _async_generator(self._chain_run, self._run_args, self._fn, self._ignore_result, self._chain, self._link)

  def __repr__(self) -> str:
    return '<Quent._Generator>'


def _make_iter_op(link: Link, mode: str) -> Callable[[Any], Any]:
  """Shared implementation for map, foreach, and filter operations.

  Three-tier sync/async pattern:
    _iter_op: Pure sync fast path. Uses manual `while True` + `next()` instead
      of `for` loop so the iterator can be handed off to _to_async mid-iteration.
    _to_async: Sync-to-async handoff. Called when a sync iteration discovers that
      `fn` returned a coroutine. Continues the SAME iterator from where sync left off.
    _full_async: Pure async path for async iterables (`async for`).

  Args:
    link: The Link containing the callable to apply to each element.
    mode: One of 'map', 'foreach', or 'filter'.
      - 'map': collect fn(item) return values
      - 'foreach': collect original items, discard fn results
      - 'filter': keep items where fn(item) is truthy
  """
  fn: Callable[[Any], Any] = link.v

  def _collect(lst: list[Any], item: Any, result: Any) -> None:
    """Append to results based on the operation mode."""
    if mode == 'map':
      lst.append(result)
    elif mode == 'foreach':
      lst.append(item)
    else:  # filter
      if result:
        lst.append(item)

  # Picks up a sync iteration that hit an awaitable result. Receives the live
  # iterator, the current item, and the pending awaitable to continue from.
  async def _to_async(iterator: Iterator[Any], item: Any, result: Any, lst: list[Any], idx: int) -> list[Any]:
    try:
      while True:
        if isawaitable(result):
          result = await result
        _collect(lst, item, result)
        idx += 1
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
      _set_link_temp_args(exc, link, item=item, index=idx)
      raise

  async def _full_async(current_value: Any) -> list[Any]:
    lst: list[Any] = []
    item = Null
    idx = 0
    try:
      async for item in current_value:
        result = fn(item)
        if isawaitable(result):
          result = await result
        _collect(lst, item, result)
        idx += 1
      return lst
    except _Break as exc:
      result = _handle_break_exc(exc, lst)
      if isawaitable(result):
        return await result  # type: ignore[no-any-return]
      return result  # type: ignore[no-any-return]
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item=item, index=idx)
      raise

  def _iter_op(current_value: Any) -> Any:
    if hasattr(current_value, '__aiter__'):
      return _full_async(current_value)
    lst: list[Any] = []
    it = iter(current_value)
    item = Null
    idx = 0
    try:
      while True:
        # H1: Separate next(it) from fn(item) so that StopIteration raised by fn
        # propagates as an error rather than being silently swallowed.
        try:
          item = next(it)
        except StopIteration:
          break
        result = fn(item)
        if isawaitable(result):
          return _to_async(it, item, result, lst, idx)
        _collect(lst, item, result)
        idx += 1
    except _Break as exc:
      return _handle_break_exc(exc, lst)
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, link, item=item, index=idx)
      raise
    return lst

  return _iter_op


def _make_foreach(link: Link, ignore_result: bool) -> Callable[[Any], Any]:
  """Create a foreach/map iteration operation for use in a chain."""
  m = 'foreach' if ignore_result else 'map'
  op = _make_iter_op(link, m)
  # Attach metadata as function attributes — a Python hack that lets the traceback
  # formatter identify the operation type without needing a class wrapper.
  op._quent_op = 'map'  # type: ignore[attr-defined]
  op._ignore_result = ignore_result  # type: ignore[attr-defined]
  return op


def _make_filter(link: Link) -> Callable[[Any], Any]:
  """Create a filter operation for use in a chain."""
  op = _make_iter_op(link, 'filter')
  op._quent_op = 'filter'  # type: ignore[attr-defined]
  return op


def _make_gather(fns: tuple[Callable[[Any], Any], ...]) -> Callable[[Any], Any]:
  """Create a gather operation for use in a chain."""

  async def _to_async(results: list[Any]) -> list[Any]:
    # H3: Create tasks from coroutines, then handle cancellation on failure.
    tasks: list[asyncio.Task[Any]] = []
    task_indices: list[int] = []
    for i, r in enumerate(results):
      if isawaitable(r):
        tasks.append(asyncio.ensure_future(r))
        task_indices.append(i)
    try:
      gathered = await asyncio.gather(*tasks)
      for idx, val in zip(task_indices, gathered):
        results[idx] = val
      return results
    except BaseException:
      for t in tasks:
        if not t.done():
          t.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)
      raise

  def _gather_op(current_value: Any) -> Any:
    results = []
    has_coro = False
    try:
      for idx, fn in enumerate(fns):  # noqa: B007
        result = fn(current_value)
        results.append(result)
        if isawaitable(result):
          has_coro = True
    except BaseException as exc:
      # If a fn raises during setup, close any already-created coroutines to avoid
      # "coroutine was never awaited" RuntimeWarning.
      for r in results:
        if isawaitable(r) and hasattr(r, 'close'):
          r.close()
      if not hasattr(exc, '__quent_gather_index__'):
        exc.__quent_gather_index__ = idx  # type: ignore[attr-defined]
        exc.__quent_gather_fn__ = fn  # type: ignore[attr-defined]
      raise
    if has_coro:
      return _to_async(results)
    return results

  _gather_op._quent_op = 'gather'  # type: ignore[attr-defined]
  _gather_op._fns = fns  # type: ignore[attr-defined]
  return _gather_op


def _make_if(predicate_link: Link | None, v_link: Link) -> Callable[[Any], Any]:
  """Create a conditional branching operation for use in a chain."""
  # L8: Extract callable from predicate link at factory time (build-time capture).
  predicate = predicate_link.v if predicate_link is not None else None

  async def _to_async_pred(pred_result: Any, current_value: Any) -> Any:
    pred_result = await pred_result
    if pred_result:
      result = _evaluate_value(v_link, current_value)
      if isawaitable(result):
        return await result
      return result
    elif _if_op._else_link is not None:  # type: ignore[attr-defined]
      result = _evaluate_value(_if_op._else_link, current_value)  # type: ignore[attr-defined]
      if isawaitable(result):
        return await result
      return result
    return current_value

  def _if_op(current_value: Any) -> Any:
    if predicate is not None:
      pred_result = predicate(current_value) if current_value is not Null else predicate()
      if isawaitable(pred_result):
        return _to_async_pred(pred_result, current_value)
    else:
      pred_result = current_value
    if pred_result:
      return _evaluate_value(v_link, current_value)
    elif _if_op._else_link is not None:  # type: ignore[attr-defined]
      return _evaluate_value(_if_op._else_link, current_value)  # type: ignore[attr-defined]
    return current_value

  _if_op._quent_op = 'if'  # type: ignore[attr-defined]
  _if_op._else_link = None  # type: ignore[attr-defined]
  _if_op._predicate_link = predicate_link  # type: ignore[attr-defined]
  return _if_op
