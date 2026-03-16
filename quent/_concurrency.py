# SPDX-License-Identifier: MIT
"""Shared concurrency helpers for iteration and gather operations.

Extracts the TaskGroup wrapper, task cancellation patterns, and
ThreadPoolExecutor lifecycle that are common to both ``_iter_ops``
and ``_gather_ops``, eliminating code duplication while keeping
domain-specific triage logic in each module.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import sys
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any

from ._eval import _isawaitable

_HAS_TASK_GROUP = sys.version_info >= (3, 11)

# Shared task creation utility.  Used by _iter_ops.py and _gather_ops.py
# for structured concurrency (NOT registered in any global registry;
# these tasks are awaited via asyncio.gather or TaskGroup).
#
# Python 3.14 added eager_start to create_task, which avoids a round trip
# through the event loop scheduler.
if sys.version_info >= (3, 14):
  _create_task_fn = functools.partial(asyncio.create_task, eager_start=True)
else:
  _create_task_fn = asyncio.create_task


async def _run_taskgroup(n: int, worker_fn: Callable[[int], Coroutine[Any, Any, None]]) -> list[BaseException] | None:
  """Run *n* workers via ``asyncio.TaskGroup``.

  Returns ``None`` on success, or a list of sub-exceptions extracted from
  the ``ExceptionGroup`` on failure.  Non-group exceptions (e.g. a bare
  ``BaseException`` without ``.exceptions``) are re-raised immediately.
  """
  __tracebackhide__ = True
  try:
    async with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]  # not available on Python 3.10
      for idx in range(n):
        tg.create_task(worker_fn(idx))
    return None
  except BaseException as eg:
    # On Python 3.11+, TaskGroup wraps task failures in ExceptionGroup.
    # _run_taskgroup is only called when _HAS_TASK_GROUP is True (Python 3.11+),
    # and TaskGroup always raises ExceptionGroup/BaseExceptionGroup.
    # Use duck-typing (hasattr 'exceptions') for robust detection that handles
    # subclasses and avoids F821 on Python 3.10.
    sub_exceptions = getattr(eg, 'exceptions', None)
    if sub_exceptions is not None:
      return list(sub_exceptions)
    raise


async def _cancel_pending_tasks(tasks: Sequence[asyncio.Future[Any]], *, timeout: float | None = None) -> None:
  """Cancel unfinished tasks and wait for all to complete.

  Args:
    timeout: Maximum seconds to wait for tasks to finish after cancellation.
      ``None`` (default) means wait indefinitely.  When used internally
      by gather/iteration operations, there is no timeout since tasks
      must settle before results can be returned.
  """
  __tracebackhide__ = True
  for t in tasks:
    if not t.done():
      t.cancel()
  if tasks:
    await asyncio.wait(tasks, timeout=timeout)


def _run_threadpool_sync(
  n: int,
  concurrency: int,
  results: list[Any],
  submit: Callable[[Executor, int], Future[Any]],
  on_exc: Callable[[BaseException, int], None],
  awaitable_msg: Callable[[int], str],
  executor: Executor | None = None,
) -> tuple[list[BaseException], TypeError | None]:
  """Run indices ``1..n-1`` in a ``ThreadPoolExecutor`` and collect results.

  This is the shared sync-concurrent lifecycle used by both
  ``_ConcurrentIterOp`` and ``_ConcurrentGatherOp``.  Index 0 is
  assumed to have been probed by the caller already.

  .. note:: **contextvars propagation**

    This helper does **not** propagate ``contextvars`` automatically.
    Callers are responsible for wrapping submissions with
    ``copy_context().run(...)`` to ensure context variables propagate to
    worker threads.  See usage in ``_ConcurrentIterOp.__call__`` and
    ``_ConcurrentGatherOp.__call__``.

  Args:
    n: Total number of items/fns (caller handles index 0).
    concurrency: ``max_workers`` for the executor.
    results: Pre-allocated list; successful results are stored at
      ``results[idx]``.
    submit: Caller-provided function that calls
      ``executor.submit(...)`` and returns the ``Future``.
    on_exc: Caller-provided function that stamps exception metadata.
    awaitable_msg: Returns the ``TypeError`` message string for
      awaitable-in-sync detection at a given index.
    executor: Optional user-provided executor. When provided, it is used
      instead of creating a new ``ThreadPoolExecutor`` and is NOT shut
      down after use — lifecycle is the caller's responsibility. When
      ``None`` (default), a new ``ThreadPoolExecutor`` is created and
      shut down automatically.

  Returns:
    ``(exceptions, awaitable_type_error)`` — the caller handles
    triage and result collection.
  """
  __tracebackhide__ = True
  if n <= 1:
    return [], None
  extracted_exceptions: list[BaseException] = []
  cm = nullcontext(executor) if executor is not None else ThreadPoolExecutor(max_workers=min(concurrency, n - 1))
  with cm as pool:
    futures: list[concurrent.futures.Future[Any]] = []
    try:
      for idx in range(1, n):
        futures.append(submit(pool, idx))
    except BaseException as submit_exc:
      # Partial submission failure: cancel already-submitted futures to
      # avoid blocking in the `with` block's shutdown(wait=True).
      for f in futures:
        f.cancel()
      if hasattr(submit_exc, 'add_note'):
        submit_exc.add_note(
          f'quent: submission failed at index {idx}; {len(futures)} of {n - 1} futures submitted before failure'
        )
      raise
    # Memory ordering: wait() blocks until all futures reach FINISHED state.
    # Internally, Future uses a threading.Condition (Future._condition) whose
    # lock acquire/release establishes a happens-before edge between each
    # worker's writes (including results[idx]) and this thread's subsequent
    # reads.  This is safe under both GIL and free-threaded (PEP 703) Python.
    concurrent.futures.wait(futures)
    awaitable_type_error: TypeError | None = None
    for i, future in enumerate(futures):
      idx = i + 1
      exc = future.exception()
      if exc is not None:
        on_exc(exc, idx)
        extracted_exceptions.append(exc)
      else:
        result = future.result()
        if _isawaitable(result):
          if hasattr(result, 'close'):
            result.close()
          if awaitable_type_error is None:
            awaitable_type_error = TypeError(awaitable_msg(idx))
        else:
          results[idx] = result
  return extracted_exceptions, awaitable_type_error
