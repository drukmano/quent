# SPDX-License-Identifier: MIT
"""Iteration operations (foreach/foreach_do) and concurrent variants."""

from __future__ import annotations

import asyncio
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
from ._types import _UNPROCESSED, ExceptionGroup, Null, QuentException, _Break, _ControlFlowSignal, _Return

# ---- Exception triage helpers for concurrent iteration ----
#
# Concurrent foreach/foreach_do triage exceptions from multiple workers.
# The classification logic is shared across the TaskGroup, asyncio.gather,
# and ThreadPoolExecutor paths within this module, so it is extracted into
# _triage_iter_exceptions / _IterTriageResult rather than duplicated three times.


class _IterTriageResult:
  """Result of triaging exceptions from concurrent iteration."""

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

  Priority: _Return > first-index _Break > regular exceptions > re-raise.

  Break signals always take priority over regular exceptions regardless of
  index.  When multiple break signals occur, the one from the earliest index
  wins.  When multiple regular exceptions occur, they are wrapped in an
  ``ExceptionGroup``.

  Each exception in ``exceptions`` is expected to carry a ``_quent_idx``
  integer attribute set by the worker that raised it (see
  ``_ConcurrentIterOp._async_concurrent`` and ``_ConcurrentIterOp.__call__``).
  ``_quent_idx`` records the 0-based index of the input item being processed
  when the exception occurred.  ``n`` is used as the fallback value when the
  attribute is absent (i.e. the exception sorts last).

  ``op`` identifies the operation name ('foreach' or 'foreach_do') and is
  used in the ``ExceptionGroup`` message when multiple exceptions are wrapped.
  """
  first_break: _Break | None = None
  first_break_idx = n
  regular: list[Exception] = []
  first_base_exc: BaseException | None = None
  first_base_idx = n
  for exc in exceptions:
    idx = getattr(exc, '_quent_idx', n)
    if isinstance(exc, _Return):
      return _IterTriageResult('return', exc=exc)
    if isinstance(exc, _Break):
      if idx < first_break_idx:
        first_break = exc
        first_break_idx = idx
    elif isinstance(exc, _ControlFlowSignal):
      raise QuentException(f'Unknown control flow signal: {type(exc).__name__}') from exc
    else:
      if not isinstance(exc, Exception):
        if first_base_exc is None or getattr(exc, '_quent_idx', n) < first_base_idx:
          first_base_exc = exc
          first_base_idx = getattr(exc, '_quent_idx', n)
        continue
      regular.append(exc)
  # Break signals always take priority over regular exceptions (per spec).
  if first_break is not None:
    return _IterTriageResult('break', break_exc=first_break, break_idx=first_break_idx)
  # BaseException subclasses (KeyboardInterrupt, SystemExit) take priority over
  # regular exceptions but never end up in ExceptionGroup (which requires Exception).
  if first_base_exc is not None:
    return _IterTriageResult('exc', exc=first_base_exc)
  if regular:
    if len(regular) == 1:
      return _IterTriageResult('exc', exc=regular[0])
    # Multiple regular exceptions: wrap in ExceptionGroup.
    eg = ExceptionGroup(
      f'{op}() encountered {len(regular)} exceptions',
      regular,
    )
    return _IterTriageResult('exc', exc=eg)
  return _IterTriageResult('reraise')  # pragma: no cover  # defensive: unreachable when exceptions is non-empty


# ---- Iteration (map / foreach_do) ----
#
# Every iteration operation follows a three-tier sync/async pattern:
#
#   1. Sync fast path (__call__): Pure synchronous execution. Uses a manual
#      `while True` / `next()` loop instead of `for` — a `for` loop would
#      silently consume an awaitable returned by fn(), preventing the handoff
#      to async. The separate `next()` call also isolates StopIteration from
#      fn(): if fn() raises StopIteration, it must propagate as an error, not
#      terminate the loop.
#
#   2. Mid-operation async transition (_to_async): When sync iteration
#      discovers that fn() returned a coroutine partway through, it hands off
#      the *live iterator* and partial results to this async continuation,
#      which picks up exactly where sync left off. No work is repeated.
#
#   3. Full async path (_full_async): The input is an async iterable from the
#      start (has __aiter__ but not __iter__). Entire operation runs async.


async def _async_handle_break(exc: _Break, lst: list[Any]) -> Any:
  """Resolve a _Break signal, awaiting its value if necessary."""
  __tracebackhide__ = True
  result = _handle_break_exc(exc, lst)
  if _isawaitable(result):
    return await result
  return result


class _IterOp:
  """Iteration operation: map or foreach_do over an iterable.

  The three tiers (sync fast path, mid-operation async transition, full async
  path) are documented in the section header above.

  ``mode`` controls how results are collected:
    - ``'foreach'``: collect ``fn(item)`` return values
    - ``'foreach_do'``: collect original items, discard ``fn`` results
  """

  __slots__ = ('_collect', '_fn', '_link', '_link_name')

  _collect: Callable[[list[Any], Any, Any], None]
  _fn: Callable[..., Any]
  _link: Link
  _link_name: str

  def __init__(self, link: Link, mode: Literal['foreach', 'foreach_do']) -> None:
    if mode not in ('foreach', 'foreach_do'):  # pragma: no cover
      raise ValueError(f'invalid _IterOp mode: {mode!r}')  # pragma: no cover
    if __debug__:
      assert link.is_callable, f'_IterOp received a non-callable link: {link.v!r}'
    self._fn = link.v
    self._link = link
    if mode == 'foreach_do':
      self._link_name = 'foreach_do'
      self._collect = _IterOp._collect_foreach_do
    else:
      self._link_name = 'foreach'
      self._collect = _IterOp._collect_foreach

  # Specialize the collection strategy at factory time to eliminate
  # per-item string dispatch (~11ns savings per iteration item).

  @staticmethod
  def _collect_foreach(lst: list[Any], item: Any, result: Any) -> None:
    lst.append(result)

  @staticmethod
  def _collect_foreach_do(lst: list[Any], item: Any, result: Any) -> None:
    lst.append(item)

  async def _to_async(self, iterator: Iterator[Any], item: Any, result: Any, lst: list[Any], idx: int) -> list[Any]:
    """Continue iteration in async mode from where the sync path left off."""
    __tracebackhide__ = True
    try:
      while True:
        if _isawaitable(result):
          result = await result
        self._collect(lst, item, result)
        idx += 1
        # Separate next() from fn() — see the three-tier pattern note above.
        try:
          item = next(iterator)
        except StopIteration:
          return lst
        result = self._fn(item)
    except _Break as exc:
      return await _async_handle_break(exc, lst)  # type: ignore[no-any-return]  # break handler return type is Any
    # Control flow signals (_Return, _Break) must propagate immediately and must
    # never be caught by the broader except clause below.
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, item=item, index=idx)
      raise

  async def _full_async(self, current_value: Any) -> list[Any]:
    """Iterate an async iterable, awaiting fn() results as needed."""
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
      return await _async_handle_break(exc, lst)  # type: ignore[no-any-return]  # break handler return type is Any
    except _ControlFlowSignal:  # Must propagate — not a regular exception.
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, item=item, index=idx)
      raise

  def __call__(self, current_value: Any) -> Any:
    """Sync fast path: iterate current_value, handing off to async on first awaitable."""
    __tracebackhide__ = True
    # Dual-protocol iterable: prefer async when a loop is running; async-only always uses _full_async.
    if _should_use_async_protocol(current_value, '__iter__', '__aiter__') is True:
      return self._full_async(current_value)
    lst: list[Any] = []
    it = iter(current_value)
    item = Null
    idx = 0
    try:
      while True:
        # next() is deliberately separate from fn() so that a StopIteration
        # raised inside fn() propagates as a real error instead of silently
        # ending the loop.
        try:
          item = next(it)
        except StopIteration:
          break
        result = self._fn(item)
        # Awaitable result? Hand off the live iterator to async continuation.
        if _isawaitable(result):
          return self._to_async(it, item, result, lst, idx)
        self._collect(lst, item, result)
        idx += 1
    except _Break as exc:
      return _handle_break_exc(exc, lst)
    except _ControlFlowSignal:  # Must propagate — not a regular exception.
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, item=item, index=idx)
      raise
    return lst


def _batch_collect_foreach(items: list[Any], results: list[Any], count: int) -> list[Any]:
  """Collect fn results (map mode), skipping unprocessed sentinels.

  Batch collector for concurrent results — distinct from ``_IterOp._collect_foreach``
  which is a per-item collector for sequential iteration.
  """
  return [r for r in results[:count] if r is not _UNPROCESSED]


def _batch_collect_foreach_do(items: list[Any], results: list[Any], count: int) -> list[Any]:
  """Collect original items (foreach_do mode), skipping unprocessed sentinels.

  Batch collector for concurrent results — distinct from ``_IterOp._collect_foreach_do``
  which is a per-item collector for sequential iteration.
  """
  return [item for item, r in zip(items[:count], results[:count]) if r is not _UNPROCESSED]


# Pre-resolved dispatch table for concurrent result collection — eliminates
# per-item string comparisons in the collection loop.
_BATCH_COLLECT_DISPATCH: dict[str, Callable[..., list[Any]]] = {
  'foreach': _batch_collect_foreach,
  'foreach_do': _batch_collect_foreach_do,
}


class _ConcurrentIterOp:
  """Concurrent iteration operation: map or foreach_do with limited parallelism.

  Uses ThreadPoolExecutor for sync callables and asyncio.Semaphore with
  TaskGroup (3.11+) or asyncio.gather (3.10) for async callables.

  Note: this class follows a parallel structure with ``_ConcurrentGatherOp``
  in ``_gather_ops.py`` (probe-first-item, dispatch-to-threadpool-or-async,
  triage-exceptions).  Shared low-level utilities live in ``_concurrency.py``.
  Sync/async is detected by probing the first item.

  **Executor lifecycle:** A new ``ThreadPoolExecutor`` is created per sync
  invocation and shut down immediately after.  This is intentional: it
  guarantees deterministic thread cleanup and avoids shared-state
  complexity.  For high-frequency chains, consider caching results or
  reducing invocation frequency rather than sharing executors.
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

    Note: the entire async iterable is collected into memory before
    processing.  This is required for concurrent execution (all items
    must be available upfront for dispatch to workers).  For large or
    unbounded async iterables, use the non-concurrent variant instead.
    """
    __tracebackhide__ = True
    items: list[Any] = []
    async for item in current_value:
      items.append(item)
    return await self._async_concurrent(items)

  async def _async_concurrent(self, items: list[Any], first_result: Any = Null) -> list[Any]:
    """Run fn over items with semaphore-limited async concurrency."""
    __tracebackhide__ = True
    n = len(items)
    if n == 0:
      return []
    results: list[Any] = [_UNPROCESSED] * n
    # Resolve -1 (unbounded) to len(items) at runtime.
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
        # _quent_idx: records which input item this worker was processing.
        # _triage_iter_exceptions reads it to select the earliest-index
        # failure when multiple concurrent workers raise simultaneously.
        exc._quent_idx = idx  # type: ignore[attr-defined]  # dynamically attaching index for triage ordering
        raise
      except _ControlFlowSignal:
        raise
      except BaseException as exc:
        exc._quent_idx = idx  # type: ignore[attr-defined]  # dynamically attaching index for triage ordering (see _Break handler above)
        _set_link_temp_args(exc, link, item=items[idx], index=idx)
        raise

    _dispatch = _make_dispatch(_worker, effective_concurrency, n)

    # Mypy suppression notes for concurrent paths:
    # [attr-defined] on asyncio.TaskGroup — not available on Python 3.10
    # [misc] on raise triage.exc — narrowed by triage.action check; guaranteed non-None
    # [arg-type] on triage.break_exc — narrowed by action=='break' check; guaranteed non-None
    # [no-any-return] — break handler return type is Any

    # -- Path 1: Python 3.11+ TaskGroup --
    if _HAS_TASK_GROUP:
      sub_excs = await _run_taskgroup(n, _dispatch)
      if sub_excs is not None:
        triage = _triage_iter_exceptions(sub_excs, n, self._mode)
        if triage.action == 'return':
          raise triage.exc from None  # type: ignore[misc]  # see block comment
        if triage.action == 'exc':
          raise triage.exc  # type: ignore[misc]  # see block comment
        if triage.action == 'break':
          return await _async_handle_break(triage.break_exc, batch_collect(items, results, triage.break_idx))  # type: ignore[arg-type, no-any-return]  # see block comment
        raise sub_excs[0]
    else:
      # -- Path 2: Python 3.10 asyncio.gather fallback --
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
          raise triage.exc from None  # type: ignore[misc]  # see block comment
        if triage.action == 'exc':
          raise triage.exc  # type: ignore[misc]  # see block comment  # noqa: B904
        if triage.action == 'break':
          return await _async_handle_break(  # type: ignore[no-any-return]  # see block comment
            triage.break_exc,  # type: ignore[arg-type]  # see block comment
            batch_collect(items, results, triage.break_idx),
          )
        raise

    return batch_collect(items, results, n)

  def __call__(self, current_value: Any) -> Any:
    """Sync entry point: probe the first item to detect sync vs async, then dispatch to
    ThreadPoolExecutor (sync) or semaphore-limited async tasks (async).
    """
    __tracebackhide__ = True
    # Dual-protocol iterable: prefer async when a loop is running; async-only always uses _from_aiter.
    if _should_use_async_protocol(current_value, '__iter__', '__aiter__') is True:
      return self._from_aiter(current_value)
    # Note: the entire iterable is materialized into memory here.  This is
    # required for concurrent execution (all items must be available upfront
    # for dispatch to workers).  For large or unbounded iterables, use the
    # non-concurrent variant instead.
    items = list(current_value)
    n = len(items)
    if n == 0:
      return []
    results: list[Any] = [_UNPROCESSED] * n
    fn = self._fn
    link = self._link
    batch_collect = self._batch_collect
    # Probe first item to detect sync vs async.
    probe_exc: BaseException | None = None
    try:
      results[0] = fn(items[0])
    except _ControlFlowSignal as exc:
      # For n==1, control flow signals propagate immediately (no other workers).
      if n == 1:
        if isinstance(exc, _Break):
          return _handle_break_exc(exc, [])
        raise
      # _Return at probe index 0 with n>1: propagate immediately — no need
      # to submit remaining items to the thread pool.
      if isinstance(exc, _Return):
        raise
      # _Break at probe index 0 with n>1: handle immediately — no results
      # exist before index 0, so there is nothing to collect.
      if isinstance(exc, _Break):
        return _handle_break_exc(exc, [])
      # Other control flow signals: capture for triage alongside threadpool exceptions.
      exc._quent_idx = 0  # type: ignore[attr-defined, unused-ignore]  # dynamically attaching index for triage ordering
      probe_exc = exc
    except BaseException as exc:
      # For n==1, no other workers to run — raise immediately.
      if n == 1:
        _set_link_temp_args(exc, link, item=items[0], index=0)
        raise
      # Fast-path system signals: KeyboardInterrupt/SystemExit should propagate
      # immediately rather than being deferred until after thread pool completion.
      if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        _set_link_temp_args(exc, link, item=items[0], index=0)
        raise
      # For n>1, capture for triage alongside threadpool exceptions.
      exc._quent_idx = 0  # type: ignore[attr-defined, unused-ignore]  # dynamically attaching index for triage ordering
      _set_link_temp_args(exc, link, item=items[0], index=0)
      probe_exc = exc
    else:
      if _isawaitable(results[0]):
        # Safety: the returned coroutine is always consumed by _run_async() in the
        # engine — it is never discarded.  The awaitable at results[0] is captured
        # inside _async_concurrent and will be awaited there, so no leak occurs.
        return self._async_concurrent(items, results[0])
      if n == 1:
        return batch_collect(items, results, 1)

    # Sync path: ThreadPoolExecutor.
    # Resolve -1 (unbounded) to len(items) at runtime.
    effective_concurrency = n if self._concurrency == -1 else self._concurrency

    def _submit(executor: Executor, idx: int) -> Future[Any]:
      return executor.submit(copy_context().run, fn, items[idx])

    def _on_exc(exc: BaseException, idx: int) -> None:
      exc._quent_idx = idx  # type: ignore[attr-defined, unused-ignore]  # dynamically attaching index for triage ordering
      # _ControlFlowSignal exceptions (_Return, _Break) skip _set_link_temp_args
      # because they are control flow, not errors — the triage function handles
      # them separately.  Regular exceptions get temp args for traceback display.
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
    # Merge index-0 probe exception (if any) with threadpool exceptions.
    if probe_exc is not None:
      exceptions.insert(0, probe_exc)
    # Prioritize real exceptions over the mixed sync/async TypeError.
    # BaseException subclasses and _ControlFlowSignal take precedence.
    if exceptions:
      triage = _triage_iter_exceptions(exceptions, n, self._mode)
      if triage.action == 'return':
        raise triage.exc from None  # type: ignore[misc]  # narrowed by triage.action check
      if triage.action == 'exc':
        raise triage.exc  # type: ignore[misc]  # narrowed by triage.action check
      if triage.action == 'break':
        return _handle_break_exc(triage.break_exc, batch_collect(items, results, triage.break_idx))  # type: ignore[arg-type]  # narrowed by action=='break' check
      raise exceptions[0]  # pragma: no cover  # unreachable: no _ControlFlowSignal subclass triggers 'reraise'
    if awaitable_err is not None:
      raise awaitable_err
    return batch_collect(items, results, n)


def _make_iter_op(
  link: Link, mode: Literal['foreach', 'foreach_do'], concurrency: int | None = None, executor: Executor | None = None
) -> Callable[[Any], Any]:
  """Create an iteration operation (map or foreach_do).

  ``mode`` selects the collection strategy:
  - ``'foreach'``: collect fn(item) return values
  - ``'foreach_do'``: discard fn results, keep original items

  ``concurrency``:
  - ``None``: sequential execution (_IterOp)
  - ``-1``: unbounded concurrent execution (_ConcurrentIterOp, resolves to len(items) at runtime)
  - positive int: bounded concurrent execution (_ConcurrentIterOp)
  """
  if concurrency is not None:
    return _ConcurrentIterOp(link, mode, concurrency, executor)
  return _IterOp(link, mode)
