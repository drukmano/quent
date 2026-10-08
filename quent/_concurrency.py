# SPDX-License-Identifier: MIT
"""Shared concurrency helpers for iteration and gather operations."""

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

# Python 3.14 added eager_start to create_task — avoids a scheduler round trip.
if sys.version_info >= (3, 14):
  _create_task_fn = functools.partial(asyncio.create_task, eager_start=True)
else:
  _create_task_fn = asyncio.create_task


class _CancelledWorker(Exception):
  """Make independent child cancellation abort TaskGroup and drain siblings."""

  def __init__(self, exc: asyncio.CancelledError, independent: bool) -> None:
    self.exc = exc
    self.independent = independent


async def _run_taskgroup(n: int, worker_fn: Callable[[int], Coroutine[Any, Any, None]]) -> list[BaseException] | None:
  """Run *n* workers via asyncio.TaskGroup.

  Returns None on success, or a list of sub-exceptions from the ExceptionGroup.
  Non-group exceptions re-raise immediately.
  """
  __tracebackhide__ = True
  parent = asyncio.current_task()
  initial_cancellations = parent.cancelling() if parent is not None else 0  # type: ignore[attr-defined]

  async def _worker(idx: int) -> None:
    __tracebackhide__ = True
    try:
      await worker_fn(idx)
      task = asyncio.current_task()
      if task is not None and task.cancelling():  # type: ignore[attr-defined]  # TaskGroup path is 3.11+
        # A callback can cancel its own task and return without another await.
        # Deliver any pending cancellation before TaskGroup can hide it.
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
      # TaskGroup treats a cancelled child as successful completion. Converting
      # it temporarily to a failure also prevents waiting forever on siblings.
      task = asyncio.current_task()
      # An explicit raise or cancelled awaited Future has no task-cancel request.
      # Requests racing sibling cleanup have no public sender identity; retain
      # them when cancellation is the only outcome, not over another failure.
      independent = task is None or task.cancelling() == 0  # type: ignore[attr-defined]
      if not independent and parent is not None and parent.cancelling() > initial_cancellations:  # type: ignore[attr-defined]
        # Let TaskGroup preserve a parent's cancellation, including its message.
        # Its own abort also cancels the parent; those sibling cancellations
        # must not replace the initiating error with our temporary marker.
        raise
      raise _CancelledWorker(exc, independent) from None

  try:
    async with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]  # not available on Python 3.10
      for idx in range(n):
        tg.create_task(_worker(idx))
    return None
  except BaseException as eg:
    # Worker failures are grouped; cancellation/system signals may be ungrouped.
    # Duck-type groups to avoid referencing unavailable builtins on Python 3.10.
    sub_exceptions = getattr(eg, 'exceptions', None)
    if sub_exceptions is not None:
      failures = [
        exc.exc if isinstance(exc, _CancelledWorker) else exc
        for exc in sub_exceptions
        if not isinstance(exc, _CancelledWorker) or exc.independent
      ]
      if failures:
        # Explicit cancellation retains BaseException priority; requests from
        # sibling cleanup do not replace the error/signal that caused cleanup.
        return failures
      raise sub_exceptions[0].exc from None
    raise


def _make_dispatch(
  worker_fn: Callable[[int], Coroutine[Any, Any, None]],
  concurrency: int,
  n: int,
) -> Callable[[int], Coroutine[Any, Any, None]]:
  """Wrap *worker_fn* with a Semaphore when concurrency < n; otherwise return as-is."""
  if concurrency >= n:
    return worker_fn

  sem = asyncio.Semaphore(concurrency)

  async def _bounded_worker(idx: int) -> None:
    __tracebackhide__ = True
    async with sem:
      await worker_fn(idx)

  return _bounded_worker


async def _create_tasks_py310(
  n: int,
  dispatch: Callable[[int], Coroutine[Any, Any, None]],
) -> list[asyncio.Task[None]]:
  """Create *n* tasks (3.10 fallback). On partial creation failure, cancel and await."""
  __tracebackhide__ = True
  tasks: list[asyncio.Task[None]] = []
  try:
    for idx in range(n):
      coro = dispatch(idx)
      try:
        tasks.append(_create_task_fn(coro))
      except BaseException:
        coro.close()
        raise
  except BaseException:
    # Avoid "Task was destroyed but it is pending" warnings.
    if tasks:
      for t in tasks:
        t.cancel()
      await asyncio.wait(tasks)
    raise
  return tasks


async def _cancel_pending_tasks(tasks: Sequence[asyncio.Future[Any]], *, timeout: float | None = None) -> None:
  """Cancel unfinished tasks and wait for completion."""
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
  """Run indices 1..n-1 in a ThreadPoolExecutor; index 0 is probed by the caller.

  Caller-provided executor: NOT shut down after use. Default: new pool, auto shutdown.
  Callers must wrap submissions with copy_context().run(...) to propagate contextvars.
  """
  __tracebackhide__ = True
  if n <= 1:
    return [], None
  extracted_exceptions: list[BaseException] = []
  cm = nullcontext(executor) if executor is not None else ThreadPoolExecutor(max_workers=min(concurrency, n - 1))
  with cm as pool:
    futures: list[concurrent.futures.Future[Any]] = []
    pending: set[concurrent.futures.Future[Any]] = set()
    try:
      for idx in range(1, n):
        # Shared pools can exceed this operation's limit. Bound submissions
        # rather than occupying their extra threads with semaphore waiters.
        if len(pending) >= concurrency:
          _, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
        future = submit(pool, idx)
        futures.append(future)
        pending.add(future)
    except BaseException as submit_exc:
      # Cancel queued work; owned-pool shutdown still waits for running calls.
      for f in futures:
        f.cancel()
      if hasattr(submit_exc, 'add_note'):
        submit_exc.add_note(
          f'quent: submission failed at index {idx}; {len(futures)} of {n - 1} futures submitted before failure'
        )
      raise
    # wait() blocks until all futures FINISHED. Future's internal Condition lock
    # establishes happens-before between worker writes (results[idx]) and our reads —
    # safe under GIL and free-threaded (PEP 703) Python.
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
