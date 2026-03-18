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
from ._eval import _isawaitable
from ._exc_meta import _set_gather_meta
from ._types import _UNPROCESSED, ExceptionGroup, QuentException, _Break, _ControlFlowSignal

_log = logging.getLogger('quent')


# ---- Exception triage helpers for concurrent gather ----
#
# Concurrent gather triages exceptions from multiple workers.
# The classification logic is shared across the TaskGroup, asyncio.gather,
# and ThreadPoolExecutor paths within this module, so it is extracted into
# _dispatch_gather_triage / _GatherTriageResult rather than duplicated three times.


class _GatherTriageResult:
  """Result of triaging exceptions from concurrent gather."""

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

  Priority: _ControlFlowSignal (immediate raise) > BaseException > ExceptionGroup > single > re-raise.

  Unlike ``_triage_iter_exceptions``, this function does not use ``_quent_idx``
  for ordering — gather collects *all* regular failures into an
  ``ExceptionGroup`` so none are lost, making first-by-index ordering
  unnecessary.  However, exceptions in ``raw_exceptions`` that were raised by
  ``_ConcurrentGatherOp._async_concurrent`` workers do carry ``_quent_idx``
  (the 0-based function index), which is read in the Python 3.10 fallback path
  of ``_async_concurrent`` to recover the function index for
  ``_set_gather_meta`` when the task's index can no longer be inferred from
  iteration order alone.
  """
  regular: list[Exception] = []
  first_base_exc: BaseException | None = None
  first_base_idx: int = -1
  for exc in raw_exceptions:
    if isinstance(exc, _Break):
      # Per spec §5.6: break_() signals are not allowed in gather operations.
      # Catch and wrap in QuentException.
      raise QuentException('break_() signals are not allowed in gather operations.') from exc
    if isinstance(exc, _ControlFlowSignal):
      # _Return takes absolute priority; any regular exceptions
      # already collected are intentionally discarded.
      if regular:
        _log.warning(
          'concurrent gather: _ControlFlowSignal encountered; %d regular exception(s) discarded: %r',
          len(regular),
          regular,
        )
      raise exc from None
    if not isinstance(exc, Exception):
      # Per spec §5.5: "the one from the earliest position in fns takes priority."
      # Use _quent_idx to select the earliest-index BaseException.
      idx = getattr(exc, '_quent_idx', -1)
      if first_base_exc is None or (idx != -1 and (first_base_idx == -1 or idx < first_base_idx)):
        first_base_exc = exc
        first_base_idx = idx
      continue
    regular.append(exc)
  if first_base_exc is not None:
    if regular:
      _log.warning(
        'concurrent gather: BaseException encountered; %d regular exception(s) discarded: %r',
        len(regular),
        regular,
      )
    return _GatherTriageResult('base_exc', exc=first_base_exc, exceptions=regular)
  if len(regular) > 1:
    return _GatherTriageResult('exc_group', exceptions=regular)
  if len(regular) == 1:
    return _GatherTriageResult('single_exc', exc=regular[0], exceptions=regular)
  return _GatherTriageResult('reraise')


def _dispatch_gather_triage(triage: _GatherTriageResult) -> None:
  """Raise for base_exc, exc_group, or single_exc triage results.

  Returns without raising for ``'reraise'`` — the caller handles
  re-raising the appropriate exception in its own context.
  """
  if triage.action == 'base_exc':
    raise triage.exc  # type: ignore[misc]  # narrowed by triage.action check
  if triage.action == 'exc_group':
    eg = ExceptionGroup(
      f'gather() encountered {len(triage.exceptions)} exceptions',
      triage.exceptions,
    )
    _set_gather_meta(eg, -1)
    raise eg from None
  if triage.action == 'single_exc':
    raise triage.exc  # type: ignore[misc]  # narrowed by triage.action check


# ---- Concurrency (gather) ----


def _make_gather(
  fns: tuple[Callable[[Any], Any], ...], concurrency: int = -1, executor: Executor | None = None
) -> Callable[[Any], Any]:
  """Create a gather operation that runs multiple functions concurrently.

  Sync fns always run concurrently via ThreadPoolExecutor (with
  ``max_workers=concurrency`` or ``len(fns)`` when unbounded).  This
  eliminates the bridge asymmetry: sync and async gather both execute
  concurrently, both produce ExceptionGroup on multiple failures.

  When ``concurrency`` is a positive int, limits the number of simultaneous
  executions.  When ``-1`` (default), all fns run concurrently with no limit
  (effective concurrency equals ``len(fns)`` at runtime).

  Raises:
    QuentException: If *fns* is empty (zero functions).
  """
  if not fns:
    raise QuentException('gather() requires at least one function.')
  return _ConcurrentGatherOp(fns, concurrency, executor)


class _ConcurrentGatherOp:
  """Concurrent gather with semaphore-limited parallelism.

  Uses ThreadPoolExecutor for sync and asyncio.Semaphore with
  TaskGroup (3.11+) or asyncio.gather (3.10) for async.

  Note: this class follows a parallel structure with ``_ConcurrentIterOp``
  in ``_iter_ops.py`` (probe-first-item, dispatch-to-threadpool-or-async,
  triage-exceptions).  Shared low-level utilities live in ``_concurrency.py``.

  **Executor lifecycle:** A new ``ThreadPoolExecutor`` is created per sync
  invocation and shut down immediately after.  This is intentional: it
  guarantees deterministic thread cleanup and avoids shared-state
  complexity.

  **Unbounded concurrency:** When ``concurrency`` is ``-1``, the effective
  concurrency is resolved to ``len(fns)`` at runtime.
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
        if results[idx] is not _UNPROCESSED:
          r = results[idx]
        else:
          r = fns[idx](current_value)
        if _isawaitable(r):
          r = await r
        results[idx] = r
      except _ControlFlowSignal:
        raise
      except BaseException as exc:
        # _quent_idx: records which gather function this worker was running.
        # The Python 3.10 asyncio.gather fallback path reads it to recover
        # the function index for _set_gather_meta when task completion order
        # no longer matches input order.
        exc._quent_idx = idx  # type: ignore[attr-defined]  # dynamically attaching index for triage ordering
        _set_gather_meta(exc, idx, fns[idx])
        raise

    _dispatch = _make_dispatch(_worker, effective_concurrency, n)

    # Mypy suppression notes for concurrent paths:
    # [attr-defined] on asyncio.TaskGroup — not available on Python 3.10
    # [misc] on raise triage.exc — narrowed by triage.action check; guaranteed non-None

    # -- Path 1: Python 3.11+ TaskGroup --
    if _HAS_TASK_GROUP:
      sub_excs = await _run_taskgroup(n, _dispatch)
      if sub_excs is not None:
        triage = _triage_gather_exceptions(sub_excs)
        _dispatch_gather_triage(triage)
        raise sub_excs[0]
    else:
      # -- Path 2: Python 3.10 asyncio.gather fallback --
      tasks = await _create_tasks_py310(n, _dispatch)
      try:
        await asyncio.gather(*tasks)
      except BaseException:
        await _cancel_pending_tasks(tasks)
        # Pre-attach gather metadata for exceptions that need it.
        # Note: _worker already calls _set_gather_meta when raising, but
        # _set_gather_meta uses first-write-wins so this is a no-op for
        # worker-raised exceptions.  This serves as a safety net for any
        # exceptions created by asyncio internals that bypass the worker's
        # except handler.
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
    """Sync entry point: probe the first fn to detect sync vs async, then dispatch to
    ThreadPoolExecutor (sync) or semaphore-limited async tasks (async).
    """
    __tracebackhide__ = True
    fns = self._fns
    n = len(fns)
    results: list[Any] = [_UNPROCESSED] * n
    # Probe first fn to detect sync vs async.
    try:
      results[0] = fns[0](current_value)
    except _Break as exc:
      raise QuentException('break_() signals are not allowed in gather operations.') from exc
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_gather_meta(exc, 0, fns[0])
      raise
    if _isawaitable(results[0]):
      # Safety: the returned coroutine is always consumed by _run_async() in the
      # engine — it is never discarded.  The awaitable at results[0] is captured
      # inside _async_concurrent and will be awaited there, so no leak occurs.
      return self._async_concurrent(results, current_value)
    if n == 1:
      return (results[0],)

    # Sync path: ThreadPoolExecutor.
    # Resolve -1 (unbounded) to len(fns) at runtime.
    effective_concurrency = n if self._concurrency == -1 else self._concurrency

    def _on_exc(exc: BaseException, idx: int) -> None:
      if isinstance(exc, Exception) and not isinstance(exc, _ControlFlowSignal):
        _set_gather_meta(exc, idx, fns[idx])

    exceptions, awaitable_err = _run_threadpool_sync(
      n,
      effective_concurrency,
      results,
      submit=lambda pool, idx: pool.submit(copy_context().run, fns[idx], current_value),
      on_exc=_on_exc,
      awaitable_msg=lambda idx: (
        f'Concurrent gather: function at index {idx} ({fns[idx]!r}) returned an awaitable in a '
        f'sync worker thread. The first fn was sync, so ThreadPoolExecutor '
        f'was used. Ensure callables are consistently sync or async.'
      ),
      executor=self._executor,
    )
    # Prioritize real exceptions over the mixed sync/async TypeError.
    # BaseException subclasses and _ControlFlowSignal take precedence.
    if exceptions:
      triage = _triage_gather_exceptions(exceptions)
      _dispatch_gather_triage(triage)
      raise exceptions[0]  # pragma: no cover  # unreachable: triage always raises
    if awaitable_err is not None:
      raise awaitable_err
    return tuple(results)
