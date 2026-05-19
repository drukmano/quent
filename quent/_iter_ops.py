# SPDX-License-Identifier: MIT
"""Iteration operations (foreach/foreach_do) and concurrent variants."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from collections.abc import Iterator
  from concurrent.futures import Executor, Future
  from typing import Literal

from ._concurrency import (
  _HAS_TASK_GROUP,
  _cancel_pending_tasks,
  _create_tasks_py310,
  _make_dispatch,
  _run_taskgroup,
  _run_threadpool_sync,
)
from ._eval import _handle_break_exc, _isawaitable, _should_use_async_protocol
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import _UNPROCESSED, ExceptionGroup, Null, QuentException, _Break, _ControlFlowSignal, _Exit, _Return

_log = logging.getLogger('quent')


def _foreach_identity(v: Any) -> Any:
  return v


# Empty name so foreach() without fn renders as `.foreach()` in tracebacks.
_foreach_identity.__name__ = ''
_foreach_identity.__qualname__ = ''


class _IterTriageResult:
  __slots__ = ('action', 'break_exc', 'break_idx', 'exc')

  action: Literal['return', 'exc', 'break', 'reraise']
  exc: BaseException | None
  break_exc: _Break | None
  break_idx: int

  def __init__(
    self,
    action: Literal['return', 'exc', 'break', 'reraise'],
    exc: BaseException | None = None,
    break_exc: _Break | None = None,
    break_idx: int = 0,
  ) -> None:
    self.action = action
    self.exc = exc
    self.break_exc = break_exc
    self.break_idx = break_idx


def _triage_iter_exceptions(exceptions: list[BaseException], n: int, op: str) -> _IterTriageResult:
  """Classify concurrent iteration exceptions by priority.

  Priority: _Return > earliest-index _Break > BaseException > regulars.
  Multiple regulars → ExceptionGroup. ``op`` ('foreach'/'foreach_do') is used
  in the EG message. Each exception is expected to carry _quent_idx (set by
  the worker); ``n`` is the fallback (sorts last).
  """
  # Full scan before priority — _Return wins unconditionally but we must
  # collect co-occurring regulars first to log them as discarded.
  return_exc: _Return | None = None
  first_break: _Break | None = None
  first_break_idx = n
  regular: list[Exception] = []
  first_base_exc: BaseException | None = None
  first_base_idx = n
  for exc in exceptions:
    idx = getattr(exc, '_quent_idx', n)
    if isinstance(exc, _Return):
      if return_exc is None:
        return_exc = exc
      continue
    if isinstance(exc, _Break):
      if idx < first_break_idx:
        first_break = exc
        first_break_idx = idx
    elif isinstance(exc, _Exit):
      # Q.exit_() bypasses the iter carve-out per §7.5 — propagate to outermost run().
      raise exc from None
    elif isinstance(exc, _ControlFlowSignal):
      raise QuentException(f'Unknown control flow signal: {type(exc).__name__}') from exc
    else:
      if not isinstance(exc, Exception):
        if first_base_exc is None or idx < first_base_idx:
          first_base_exc = exc
          first_base_idx = idx
        continue
      regular.append(exc)
  if return_exc is not None:
    if regular:
      _log.warning(
        'concurrent %s: _Return encountered; %d regular exception(s) discarded: %r',
        op,
        len(regular),
        regular,
      )
    return _IterTriageResult('return', exc=return_exc)
  if first_break is not None:
    return _IterTriageResult('break', break_exc=first_break, break_idx=first_break_idx)
  if first_base_exc is not None:
    return _IterTriageResult('exc', exc=first_base_exc)
  if regular:
    if len(regular) == 1:
      return _IterTriageResult('exc', exc=regular[0])
    eg = ExceptionGroup(
      f'{op}() encountered {len(regular)} exceptions',
      regular,
    )
    return _IterTriageResult('exc', exc=eg)
  return _IterTriageResult('reraise')  # pragma: no cover


# Three-tier sync/async pattern shared by all iteration ops:
#   1. Sync fast path (__call__): manual `while True` / `next()` — a `for` loop
#      would silently consume an awaitable from fn(), preventing the handoff to
#      async. Separate next() also isolates StopIteration from fn().
#   2. Mid-op transition (_to_async): on first awaitable, hand off live iterator
#      and partial results to the async continuation. No work repeated.
#   3. Full async (_full_async): input is an async iterable from the start.


async def _async_handle_break(exc: _Break, lst: list[Any]) -> Any:
  __tracebackhide__ = True
  result = _handle_break_exc(exc, lst)
  if _isawaitable(result):
    return await result
  return result


class _IterOp:
  """Sequential map (foreach) or foreach_do over an iterable."""

  __slots__ = ('_collect', '_fn', '_link', '_link_name')

  _collect: Callable[[list[Any], Any, Any], None]
  _fn: Callable[..., Any]
  _link: Link
  _link_name: str

  def __init__(self, link: Link, mode: Literal['foreach', 'foreach_do']) -> None:
    if mode not in ('foreach', 'foreach_do'):  # pragma: no cover
      raise ValueError(f'invalid _IterOp mode: {mode!r}')  # pragma: no cover
    self._fn = link.v
    self._link = link
    if mode == 'foreach_do':
      self._link_name = 'foreach_do'
      self._collect = _IterOp._collect_foreach_do
    else:
      self._link_name = 'foreach'
      self._collect = _IterOp._collect_foreach

  # Specialized at factory time — saves ~11ns/item vs string dispatch.

  @staticmethod
  def _collect_foreach(lst: list[Any], item: Any, result: Any) -> None:
    lst.append(result)

  @staticmethod
  def _collect_foreach_do(lst: list[Any], item: Any, result: Any) -> None:
    lst.append(item)

  async def _to_async(self, iterator: Iterator[Any], item: Any, result: Any, lst: list[Any], idx: int) -> list[Any]:
    __tracebackhide__ = True
    try:
      while True:
        if _isawaitable(result):
          result = await result
        self._collect(lst, item, result)
        idx += 1
        # next() separate from fn() — see three-tier pattern note above.
        try:
          item = next(iterator)
        except StopIteration:
          return lst
        result = self._fn(item)
    except _Break as exc:
      return await _async_handle_break(exc, lst)  # type: ignore[no-any-return]
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, item=item, index=idx)
      raise

  async def _full_async(self, current_value: Any) -> list[Any]:
    __tracebackhide__ = True
    lst: list[Any] = []
    item = Null
    idx = 0
    try:
      async for item in current_value:
        result = self._fn(item)
        if _isawaitable(result):
          result = await result
        self._collect(lst, item, result)
        idx += 1
      return lst
    except _Break as exc:
      return await _async_handle_break(exc, lst)  # type: ignore[no-any-return]
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, item=item, index=idx)
      raise

  def __call__(self, current_value: Any) -> Any:
    __tracebackhide__ = True
    # Dual-protocol iterable: prefer async when a loop is running.
    if _should_use_async_protocol(current_value, '__iter__', '__aiter__') is True:
      return self._full_async(current_value)
    lst: list[Any] = []
    it = iter(current_value)
    item = Null
    idx = 0
    try:
      while True:
        # next() separate from fn() — StopIteration from fn() must be a real error.
        try:
          item = next(it)
        except StopIteration:
          break
        result = self._fn(item)
        if _isawaitable(result):
          return self._to_async(it, item, result, lst, idx)
        self._collect(lst, item, result)
        idx += 1
    except _Break as exc:
      return _handle_break_exc(exc, lst)
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, item=item, index=idx)
      raise
    return lst


def _batch_collect_foreach(items: list[Any], results: list[Any], count: int) -> list[Any]:
  """Collect fn results (concurrent path), skipping unprocessed sentinels."""
  return [r for r in results[:count] if r is not _UNPROCESSED]


def _batch_collect_foreach_do(items: list[Any], results: list[Any], count: int) -> list[Any]:
  """Collect original items (concurrent foreach_do), skipping unprocessed sentinels."""
  return [item for item, r in zip(items[:count], results[:count]) if r is not _UNPROCESSED]


_BATCH_COLLECT_DISPATCH: dict[str, Callable[..., list[Any]]] = {
  'foreach': _batch_collect_foreach,
  'foreach_do': _batch_collect_foreach_do,
}


class _ConcurrentIterOp:
  """Concurrent foreach/foreach_do with bounded parallelism.

  ThreadPoolExecutor (sync) / Semaphore + TaskGroup (3.11+) or asyncio.gather
  (3.10) (async). Sync/async detected by probing the first item. A new pool
  is created per sync invocation and shut down after (deterministic cleanup).
  """

  __slots__ = ('_batch_collect', '_concurrency', '_executor', '_fn', '_link', '_link_name', '_mode')

  _batch_collect: Callable[[list[Any], list[Any], int], list[Any]]
  _concurrency: int
  _executor: Executor | None
  _fn: Callable[..., Any]
  _link: Link
  _link_name: str
  _mode: Literal['foreach', 'foreach_do']

  def __init__(
    self, link: Link, mode: Literal['foreach', 'foreach_do'], concurrency: int, executor: Executor | None = None
  ) -> None:
    if mode not in ('foreach', 'foreach_do'):  # pragma: no cover
      raise ValueError(f'invalid _ConcurrentIterOp mode: {mode!r}')  # pragma: no cover
    self._fn = link.v
    self._link = link
    self._mode = mode
    self._batch_collect = _BATCH_COLLECT_DISPATCH[mode]
    self._concurrency = concurrency
    self._executor = executor
    self._link_name = mode

  async def _from_aiter(self, current_value: Any) -> list[Any]:
    """Materialize an async iterable into a list, then run concurrently.

    Entire iterable is collected upfront — required for concurrent dispatch.
    For large/unbounded inputs, use the non-concurrent variant.
    """
    __tracebackhide__ = True
    items: list[Any] = []
    async for item in current_value:
      items.append(item)
    return await self._async_concurrent(items)

  async def _async_concurrent(self, items: list[Any], first_result: Any = Null) -> list[Any]:
    __tracebackhide__ = True
    n = len(items)
    if n == 0:
      return []
    results: list[Any] = [_UNPROCESSED] * n
    effective_concurrency = n if self._concurrency == -1 else self._concurrency
    fn = self._fn
    link = self._link
    batch_collect = self._batch_collect

    async def _worker(idx: int) -> None:
      __tracebackhide__ = True
      try:
        if idx == 0 and first_result is not Null:
          r = first_result
        else:
          r = fn(items[idx])
        if _isawaitable(r):
          r = await r
        results[idx] = r
      except _Break as exc:
        # _quent_idx → earliest-index winner in triage.
        exc._quent_idx = idx  # type: ignore[attr-defined]
        raise
      except _ControlFlowSignal:
        raise
      except BaseException as exc:
        exc._quent_idx = idx  # type: ignore[attr-defined]
        _set_link_temp_args(exc, link, item=items[idx], index=idx)
        raise

    _dispatch = _make_dispatch(_worker, effective_concurrency, n)

    # Mypy suppressions on concurrent paths:
    # [attr-defined] asyncio.TaskGroup — not on 3.10
    # [misc] raise triage.exc — narrowed by triage.action
    # [arg-type] triage.break_exc — narrowed by action=='break'
    # [no-any-return] break handler return is Any

    if _HAS_TASK_GROUP:
      sub_excs = await _run_taskgroup(n, _dispatch)
      if sub_excs is not None:
        triage = _triage_iter_exceptions(sub_excs, n, self._mode)
        if triage.action == 'return':
          raise triage.exc from None  # type: ignore[misc]
        if triage.action == 'exc':
          raise triage.exc  # type: ignore[misc]
        if triage.action == 'break':
          return await _async_handle_break(triage.break_exc, batch_collect(items, results, triage.break_idx))  # type: ignore[arg-type, no-any-return]
        raise sub_excs[0]
    else:
      # 3.10 asyncio.gather fallback
      tasks = await _create_tasks_py310(n, _dispatch)
      try:
        await asyncio.gather(*tasks)
      except BaseException:
        await _cancel_pending_tasks(tasks)
        task_exceptions: list[BaseException] = []
        for t in tasks:
          if t.done() and not t.cancelled():
            exc = t.exception()
            if exc is not None:
              task_exceptions.append(exc)
        triage = _triage_iter_exceptions(task_exceptions, n, self._mode)
        if triage.action == 'return':
          raise triage.exc from None  # type: ignore[misc]
        if triage.action == 'exc':
          raise triage.exc  # type: ignore[misc]  # noqa: B904
        if triage.action == 'break':
          return await _async_handle_break(  # type: ignore[no-any-return]
            triage.break_exc,  # type: ignore[arg-type]
            batch_collect(items, results, triage.break_idx),
          )
        raise

    return batch_collect(items, results, n)

  def __call__(self, current_value: Any) -> Any:
    __tracebackhide__ = True
    if _should_use_async_protocol(current_value, '__iter__', '__aiter__') is True:
      return self._from_aiter(current_value)
    # Materialize upfront — required for concurrent dispatch. Use the non-concurrent
    # variant for large/unbounded inputs.
    items = list(current_value)
    n = len(items)
    if n == 0:
      return []
    results: list[Any] = [_UNPROCESSED] * n
    fn = self._fn
    link = self._link
    batch_collect = self._batch_collect
    probe_exc: BaseException | None = None
    try:
      results[0] = fn(items[0])
    except _ControlFlowSignal as exc:
      if n == 1:
        if isinstance(exc, _Break):
          return _handle_break_exc(exc, [])
        raise
      # _Return at idx 0 with n>1: propagate; no point submitting more.
      if isinstance(exc, _Return):
        raise
      # _Break at idx 0 with n>1: nothing collected before idx 0.
      if isinstance(exc, _Break):
        return _handle_break_exc(exc, [])
      exc._quent_idx = 0  # type: ignore[attr-defined, unused-ignore]
      probe_exc = exc
    except BaseException as exc:
      if n == 1:
        _set_link_temp_args(exc, link, item=items[0], index=0)
        raise
      # KeyboardInterrupt/SystemExit propagate immediately, not after thread pool.
      if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        _set_link_temp_args(exc, link, item=items[0], index=0)
        raise
      exc._quent_idx = 0  # type: ignore[attr-defined, unused-ignore]
      _set_link_temp_args(exc, link, item=items[0], index=0)
      probe_exc = exc
    else:
      if _isawaitable(results[0]):
        # Safety: returned coroutine is always consumed by _run_async() in the engine.
        return self._async_concurrent(items, results[0])
      if n == 1:
        return batch_collect(items, results, 1)

    effective_concurrency = n if self._concurrency == -1 else self._concurrency

    def _submit(executor: Executor, idx: int) -> Future[Any]:
      return executor.submit(copy_context().run, fn, items[idx])

    def _on_exc(exc: BaseException, idx: int) -> None:
      exc._quent_idx = idx  # type: ignore[attr-defined, unused-ignore]
      # Control flow signals skip temp args — handled by triage separately.
      if not isinstance(exc, _ControlFlowSignal):
        _set_link_temp_args(exc, link, item=items[idx], index=idx)

    exceptions, awaitable_err = _run_threadpool_sync(
      n,
      effective_concurrency,
      results,
      submit=_submit,
      on_exc=_on_exc,
      awaitable_msg=lambda idx: (
        f'Concurrent foreach/foreach_do: item at index {idx}: {fn!r} returned an awaitable in a '
        f'sync worker thread. The first item was sync, so ThreadPoolExecutor '
        f'was used. Ensure the callable is consistently sync or async.'
      ),
      executor=self._executor,
    )
    if probe_exc is not None:
      exceptions.insert(0, probe_exc)
    # Real exceptions take precedence over the mixed sync/async TypeError.
    if exceptions:
      triage = _triage_iter_exceptions(exceptions, n, self._mode)
      if triage.action == 'return':
        raise triage.exc from None  # type: ignore[misc]
      if triage.action == 'exc':
        raise triage.exc  # type: ignore[misc]
      if triage.action == 'break':
        return _handle_break_exc(triage.break_exc, batch_collect(items, results, triage.break_idx))  # type: ignore[arg-type]
      raise exceptions[0]  # pragma: no cover
    if awaitable_err is not None:
      raise awaitable_err
    return batch_collect(items, results, n)


def _make_iter_op(
  link: Link, mode: Literal['foreach', 'foreach_do'], concurrency: int | None = None, executor: Executor | None = None
) -> Callable[[Any], Any]:
  """concurrency: None=sequential, -1=unbounded, positive int=bounded."""
  if concurrency is not None:
    return _ConcurrentIterOp(link, mode, concurrency, executor)
  return _IterOp(link, mode)
