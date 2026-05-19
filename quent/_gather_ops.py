# SPDX-License-Identifier: MIT
"""Gather operations and concurrent variants."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextvars import copy_context
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from concurrent.futures import Executor
  from typing import Literal

from ._concurrency import (
  _HAS_TASK_GROUP,
  _cancel_pending_tasks,
  _create_tasks_py310,
  _make_dispatch,
  _run_taskgroup,
  _run_threadpool_sync,
)
from ._eval import _handle_return_exc, _isawaitable
from ._exc_meta import _set_gather_meta
from ._types import _UNPROCESSED, ExceptionGroup, QuentException, _Break, _ControlFlowSignal, _Exit, _Return

_log = logging.getLogger('quent')


class _GatherTriageResult:
  __slots__ = ('action', 'exc', 'exceptions')

  action: Literal['base_exc', 'exc_group', 'single_exc', 'reraise']
  exc: BaseException | None
  exceptions: list[Exception]

  def __init__(
    self,
    action: Literal['base_exc', 'exc_group', 'single_exc', 'reraise'],
    exc: BaseException | None = None,
    exceptions: list[Exception] | None = None,
  ) -> None:
    self.action = action
    self.exc = exc
    self.exceptions = exceptions or []


def _triage_gather_exceptions(raw_exceptions: list[BaseException]) -> _GatherTriageResult:
  """Classify concurrent gather exceptions by priority.

  Priority: _Return > _Break (invalid in gather) > BaseException > ExceptionGroup > single.
  Unlike iter triage, gather wraps all regular failures in an ExceptionGroup, so
  first-by-index ordering doesn't matter for regulars. BaseException sorts by
  _quent_idx (set by workers) to preserve earliest-fn-index priority.
  """
  # Full scan — _Return wins unconditionally but might be co-occurring with
  # _Break at a later list position, so we don't short-circuit on _Break.
  return_exc: _Return | None = None
  break_present = False
  break_origin: _Break | None = None
  regular: list[Exception] = []
  first_base_exc: BaseException | None = None
  first_base_idx: int = -1
  for exc in raw_exceptions:
    if isinstance(exc, _Return):
      if return_exc is None:
        return_exc = exc
      continue
    if isinstance(exc, _Break):
      if not break_present:
        break_present = True
        break_origin = exc
      continue
    if isinstance(exc, _Exit):
      # Q.exit_() bypasses gather carve-out — propagate to outermost run().
      raise exc from None
    if isinstance(exc, _ControlFlowSignal):
      raise QuentException(f'Unknown control flow signal: {type(exc).__name__}') from exc
    if not isinstance(exc, Exception):
      idx = getattr(exc, '_quent_idx', -1)
      if first_base_exc is None or (idx != -1 and (first_base_idx == -1 or idx < first_base_idx)):
        first_base_exc = exc
        first_base_idx = idx
      continue
    regular.append(exc)

  if return_exc is not None:
    if regular:
      _log.warning(
        'concurrent gather: _Return encountered; %d regular exception(s) discarded: %r',
        len(regular),
        regular,
      )
    raise return_exc from None
  if break_present:
    raise QuentException('break_() signals are not allowed in gather operations.') from break_origin
  if first_base_exc is not None:
    return _GatherTriageResult('base_exc', exc=first_base_exc, exceptions=regular)
  if len(regular) > 1:
    return _GatherTriageResult('exc_group', exceptions=regular)
  if len(regular) == 1:
    return _GatherTriageResult('single_exc', exc=regular[0], exceptions=regular)
  return _GatherTriageResult('reraise')


def _dispatch_gather_triage(triage: _GatherTriageResult) -> None:
  """Raise for base_exc/exc_group/single_exc; return without raising for 'reraise'."""
  if triage.action == 'base_exc':
    raise triage.exc  # type: ignore[misc]
  if triage.action == 'exc_group':
    eg = ExceptionGroup(
      f'gather() encountered {len(triage.exceptions)} exceptions',
      triage.exceptions,
    )
    _set_gather_meta(eg, -1)
    raise eg from None
  if triage.action == 'single_exc':
    raise triage.exc  # type: ignore[misc]


def _make_gather(
  fns: tuple[Callable[[Any], Any], ...], concurrency: int = -1, executor: Executor | None = None
) -> Callable[[Any], Any]:
  if not fns:
    raise QuentException('gather() requires at least one function.')
  return _ConcurrentGatherOp(fns, concurrency, executor)


class _ConcurrentGatherOp:
  """Concurrent gather: ThreadPoolExecutor (sync) or Semaphore+TaskGroup/gather (async).

  Sync gather is always concurrent — eliminates bridge asymmetry. New executor
  per sync invocation, shut down after. concurrency=-1 resolves to len(fns) at runtime.
  """

  __slots__ = ('_concurrency', '_executor', '_fns', '_link_name')

  _concurrency: int
  _executor: Executor | None
  _fns: tuple[Callable[[Any], Any], ...]
  _link_name: str

  def __init__(self, fns: tuple[Callable[[Any], Any], ...], concurrency: int, executor: Executor | None = None) -> None:
    self._fns = fns
    self._concurrency = concurrency
    self._executor = executor
    self._link_name = 'gather'

  async def _async_concurrent(self, results: list[Any], current_value: Any) -> tuple[Any, ...]:
    __tracebackhide__ = True
    fns = self._fns
    n = len(fns)
    effective_concurrency = n if self._concurrency == -1 else self._concurrency

    async def _worker(idx: int) -> None:
      __tracebackhide__ = True
      try:
        try:
          if results[idx] is not _UNPROCESSED:
            r = results[idx]
          else:
            r = fns[idx](current_value)
          if _isawaitable(r):
            r = await r
        except _Return as ret_exc:
          # Q.return_() inside a gather worker returns from the worker — value
          # becomes that gather position's tuple element.
          r = _handle_return_exc(ret_exc)
          if _isawaitable(r):
            r = await r
        results[idx] = r
      except _ControlFlowSignal:
        raise
      except BaseException as exc:
        # _quent_idx: read by 3.10 fallback to recover fn index when task
        # completion order no longer matches input order.
        exc._quent_idx = idx  # type: ignore[attr-defined]
        _set_gather_meta(exc, idx, fns[idx])
        raise

    _dispatch = _make_dispatch(_worker, effective_concurrency, n)

    if _HAS_TASK_GROUP:
      sub_excs = await _run_taskgroup(n, _dispatch)
      if sub_excs is not None:
        triage = _triage_gather_exceptions(sub_excs)
        _dispatch_gather_triage(triage)
        raise sub_excs[0]
    else:
      tasks = await _create_tasks_py310(n, _dispatch)
      try:
        await asyncio.gather(*tasks)
      except BaseException:
        await _cancel_pending_tasks(tasks)
        # Worker already calls _set_gather_meta (first-write-wins); this is a
        # safety net for exceptions created by asyncio internals.
        raw_exceptions: list[BaseException] = []
        for t in tasks:
          if t.done() and not t.cancelled():
            exc = t.exception()
            if exc is not None:
              if isinstance(exc, Exception) and not isinstance(exc, _ControlFlowSignal):
                idx_val = getattr(exc, '_quent_idx', -1)
                _set_gather_meta(exc, idx_val, fns[idx_val] if isinstance(idx_val, int) and 0 <= idx_val < n else None)
              raw_exceptions.append(exc)
        triage = _triage_gather_exceptions(raw_exceptions)
        _dispatch_gather_triage(triage)
        raise

    return tuple(results)

  def __call__(self, current_value: Any) -> Any:
    """Probe first fn to detect sync vs async; dispatch to ThreadPool or async tasks."""
    __tracebackhide__ = True
    fns = self._fns
    n = len(fns)
    results: list[Any] = [_UNPROCESSED] * n
    try:
      try:
        results[0] = fns[0](current_value)
      except _Return as ret_exc:
        r = _handle_return_exc(ret_exc)
        if _isawaitable(r):
          # Awaitable — store for _async_concurrent to await.
          results[0] = r
        else:
          results[0] = r
    except _Break as exc:
      raise QuentException('break_() signals are not allowed in gather operations.') from exc
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_gather_meta(exc, 0, fns[0])
      raise
    if _isawaitable(results[0]):
      # Safety: returned coroutine is always consumed by _run_async() in the engine.
      return self._async_concurrent(results, current_value)
    if n == 1:
      return (results[0],)

    effective_concurrency = n if self._concurrency == -1 else self._concurrency

    def _on_exc(exc: BaseException, idx: int) -> None:
      if isinstance(exc, Exception) and not isinstance(exc, _ControlFlowSignal):
        _set_gather_meta(exc, idx, fns[idx])

    def _worker_sync(idx: int) -> Any:
      __tracebackhide__ = True
      try:
        return fns[idx](current_value)
      except _Return as ret_exc:
        r = _handle_return_exc(ret_exc)
        if _isawaitable(r):
          # Lazy fn returned awaitable in sync gather worker — no loop on thread.
          if hasattr(r, 'close'):
            r.close()
          raise TypeError(
            f'gather worker at index {idx}: Q.return_() lazy value resolved to an awaitable '
            'in sync execution; use an async fn or avoid awaitable lazy values in gather workers.'
          ) from ret_exc
        return r

    exceptions, awaitable_err = _run_threadpool_sync(
      n,
      effective_concurrency,
      results,
      submit=lambda pool, idx: pool.submit(copy_context().run, _worker_sync, idx),
      on_exc=_on_exc,
      awaitable_msg=lambda idx: (
        f'Concurrent gather: function at index {idx} ({fns[idx]!r}) returned an awaitable in a '
        f'sync worker thread. The first fn was sync, so ThreadPoolExecutor '
        f'was used. Ensure callables are consistently sync or async.'
      ),
      executor=self._executor,
    )
    # Real exceptions and control flow signals take precedence over the mixed sync/async TypeError.
    if exceptions:
      triage = _triage_gather_exceptions(exceptions)
      _dispatch_gather_triage(triage)
      raise exceptions[0]  # pragma: no cover
    if awaitable_err is not None:
      raise awaitable_err
    return tuple(results)
