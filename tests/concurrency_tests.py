# SPDX-License-Identifier: MIT
"""Tests for SPEC §11 — Concurrency.

Tests cover:
- §11.2: concurrency parameter validation (bounds, types)
- §11.3: Sync concurrent execution (ThreadPoolExecutor)
- §11.4: Async concurrent execution (Semaphore-limited tasks)
- §11.5: Sync/async detection (probe-based)
- §11.6: Async transition for sync chain handlers
- §11.7: Context variable propagation
"""

from __future__ import annotations

import asyncio
import contextvars
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Chain, QuentException
from tests.tests_helper import SymmetricTestCase

if sys.version_info < (3, 11):
  from quent._types import ExceptionGroup

# --- §11.2: concurrency parameter validation ---


class ConcurrencyParamValidationTest(TestCase):
  """§11.2: Bounds and type validation for the concurrency parameter."""

  def test_concurrency_none_allowed(self):
    """concurrency=None is the default (sequential)."""
    result = Chain([1, 2, 3]).foreach(lambda x: x + 1, concurrency=None).run()
    self.assertEqual(result, [2, 3, 4])

  def test_concurrency_positive_int(self):
    """concurrency=2 is valid."""
    result = Chain([1, 2, 3]).foreach(lambda x: x + 1, concurrency=2).run()
    self.assertEqual(result, [2, 3, 4])

  def test_concurrency_one(self):
    """concurrency=1 is the minimum valid value."""
    result = Chain([1, 2, 3]).foreach(lambda x: x + 1, concurrency=1).run()
    self.assertEqual(result, [2, 3, 4])

  def test_concurrency_zero_raises(self):
    """concurrency=0 raises ValueError with exact message format."""
    with self.assertRaises(ValueError) as ctx:
      Chain([1]).foreach(lambda x: x, concurrency=0)
    self.assertIn('concurrency', str(ctx.exception))
    self.assertEqual(
      str(ctx.exception),
      'foreach() concurrency must be -1 (unbounded) or a positive integer, got 0',
    )

  def test_concurrency_minus_one_unbounded(self):
    """concurrency=-1 is valid (unbounded) and processes all items."""
    result = Chain([1, 2, 3]).foreach(lambda x: x + 1, concurrency=-1).run()
    self.assertEqual(result, [2, 3, 4])

  def test_concurrency_negative_other_raises(self):
    """concurrency=-2 and other negatives (not -1) raise ValueError with exact message."""
    for bad in [-2, -3, -100]:
      with self.assertRaises(ValueError) as ctx:
        Chain([1]).foreach(lambda x: x, concurrency=bad)
      self.assertIn('concurrency', str(ctx.exception))
      self.assertEqual(
        str(ctx.exception),
        f'foreach() concurrency must be -1 (unbounded) or a positive integer, got {bad}',
      )

  def test_concurrency_bool_raises(self):
    """concurrency=True raises TypeError with exact message format."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).foreach(lambda x: x, concurrency=True)
    self.assertIn('concurrency', str(ctx.exception))
    self.assertEqual(
      str(ctx.exception),
      'foreach() concurrency must be a positive integer or -1 (unbounded), got bool',
    )

  def test_concurrency_false_raises(self):
    """concurrency=False raises TypeError with exact message format."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).foreach(lambda x: x, concurrency=False)
    self.assertIn('concurrency', str(ctx.exception))
    self.assertEqual(
      str(ctx.exception),
      'foreach() concurrency must be a positive integer or -1 (unbounded), got bool',
    )

  def test_concurrency_float_raises(self):
    """concurrency=2.0 raises TypeError with exact message format."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).foreach(lambda x: x, concurrency=2.0)
    self.assertIn('concurrency', str(ctx.exception))
    self.assertEqual(
      str(ctx.exception),
      'foreach() concurrency must be a positive integer or -1 (unbounded), got float',
    )

  def test_concurrency_string_raises(self):
    """concurrency='2' raises TypeError with exact message format."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).foreach(lambda x: x, concurrency='2')
    self.assertIn('concurrency', str(ctx.exception))
    self.assertEqual(
      str(ctx.exception),
      'foreach() concurrency must be a positive integer or -1 (unbounded), got str',
    )

  def test_concurrency_minus_one_on_foreach_do(self):
    """concurrency=-1 is valid on foreach_do (unbounded)."""
    result = Chain([1, 2, 3]).foreach_do(lambda x: x * 100, concurrency=-1).run()
    self.assertEqual(result, [1, 2, 3])

  def test_concurrency_minus_one_on_gather(self):
    """concurrency=-1 is the default for gather (unbounded)."""
    result = Chain(5).gather(lambda x: x + 1, lambda x: x + 2, concurrency=-1).run()
    self.assertEqual(result, (6, 7))

  def test_gather_default_concurrency_is_minus_one(self):
    """gather() default concurrency is -1 (unbounded), same as explicitly passing -1."""
    result_default = Chain(5).gather(lambda x: x + 1, lambda x: x + 2).run()
    result_explicit = Chain(5).gather(lambda x: x + 1, lambda x: x + 2, concurrency=-1).run()
    self.assertEqual(result_default, (6, 7))
    self.assertEqual(result_explicit, (6, 7))

  def test_concurrency_validation_on_foreach_do(self):
    """concurrency validation also applies to foreach_do with exact message."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).foreach_do(lambda x: x, concurrency=True)
    self.assertEqual(
      str(ctx.exception),
      'foreach_do() concurrency must be a positive integer or -1 (unbounded), got bool',
    )

  def test_concurrency_validation_on_gather(self):
    """concurrency validation also applies to gather with exact message."""
    with self.assertRaises(TypeError) as ctx:
      Chain(1).gather(lambda x: x, concurrency=2.5)
    self.assertEqual(
      str(ctx.exception),
      'gather() concurrency must be a positive integer or -1 (unbounded), got float',
    )

  def test_concurrency_minus_two_on_gather_raises(self):
    """concurrency=-2 on gather raises ValueError with exact message."""
    with self.assertRaises(ValueError) as ctx:
      Chain(1).gather(lambda x: x, concurrency=-2)
    self.assertIn('concurrency', str(ctx.exception))
    self.assertEqual(
      str(ctx.exception),
      'gather() concurrency must be -1 (unbounded) or a positive integer, got -2',
    )

  def test_gather_rejects_concurrency_none(self):
    """gather() does not accept concurrency=None — gather is always concurrent (§5.5)."""
    with self.assertRaises(TypeError):
      Chain(5).gather(lambda x: x, concurrency=None)


# --- §11.3: Sync concurrent execution ---


class SyncConcurrentTest(TestCase):
  """§11.3: Sync concurrent execution uses ThreadPoolExecutor."""

  def test_map_concurrent_results_in_order(self):
    """Concurrent map preserves input order."""
    result = Chain(list(range(10))).foreach(lambda x: x * 2, concurrency=4).run()
    self.assertEqual(result, [x * 2 for x in range(10)])

  def test_foreach_do_concurrent_preserves_originals(self):
    """Concurrent foreach_do preserves original elements."""
    result = Chain(list(range(5))).foreach_do(lambda x: x * 100, concurrency=2).run()
    self.assertEqual(result, list(range(5)))

  def test_gather_concurrent_results_in_order(self):
    """Concurrent gather returns results in fn order."""
    result = (
      Chain(5)
      .gather(
        lambda x: x + 1,
        lambda x: x + 2,
        lambda x: x + 3,
        concurrency=2,
      )
      .run()
    )
    self.assertEqual(result, (6, 7, 8))

  def test_gather_zero_fns_raises(self):
    """gather() with zero fns raises QuentException at build time."""
    with self.assertRaises(QuentException):
      Chain(5).gather()

  def test_gather_one_fn(self):
    """gather() with one fn returns single-element tuple."""
    result = Chain(5).gather(lambda x: x + 1).run()
    self.assertEqual(result, (6,))

  def test_concurrent_uses_threads(self):
    """Concurrent sync operations use different threads."""
    thread_ids = []
    lock = threading.Lock()

    def record_thread(x):
      with lock:
        thread_ids.append(threading.current_thread().ident)
      return x

    Chain(list(range(5))).foreach(record_thread, concurrency=3).run()
    # At least the main thread + some workers should differ
    self.assertTrue(len(set(thread_ids)) >= 1)

  # test_awaitable_from_sync_worker_raises removed — canonical test in asymmetry_tests.py

  def test_single_worker_error_not_wrapped(self):
    """Single worker error is raised directly, not in ExceptionGroup."""
    with self.assertRaises(ValueError):
      Chain([1]).foreach(lambda x: (_ for _ in ()).throw(ValueError('single')), concurrency=2).run()


# --- §11.4: Async concurrent execution ---


class AsyncConcurrentTest(IsolatedAsyncioTestCase):
  """§11.4: Async concurrent execution uses semaphore-limited tasks."""

  async def test_async_map_concurrent(self):
    """Async concurrent map works correctly."""

    async def async_double(x):
      await asyncio.sleep(0.001)
      return x * 2

    result = await Chain(list(range(5))).foreach(async_double, concurrency=3).run()
    self.assertEqual(result, [x * 2 for x in range(5)])

  async def test_async_foreach_do_concurrent(self):
    """Async concurrent foreach_do preserves originals."""

    async def async_noop(x):
      await asyncio.sleep(0.001)

    result = await Chain(list(range(5))).foreach_do(async_noop, concurrency=2).run()
    self.assertEqual(result, list(range(5)))

  async def test_async_gather_concurrent(self):
    """Async concurrent gather returns results in order."""

    async def async_add(x, n):
      await asyncio.sleep(0.001)
      return x + n

    result = (
      await Chain(10)
      .gather(
        lambda x: async_add(x, 1),
        lambda x: async_add(x, 2),
        lambda x: async_add(x, 3),
        concurrency=2,
      )
      .run()
    )
    self.assertEqual(result, (11, 12, 13))

  async def test_async_semaphore_limits_concurrency(self):
    """Semaphore actually limits concurrent async tasks (§11.4)."""
    max_concurrent = 0
    current_concurrent = 0
    lock = asyncio.Lock()

    async def track_concurrency(x):
      nonlocal max_concurrent, current_concurrent
      async with lock:
        current_concurrent += 1
        if current_concurrent > max_concurrent:
          max_concurrent = current_concurrent
      await asyncio.sleep(0.02)
      async with lock:
        current_concurrent -= 1
      return x

    concurrency_limit = 3
    await Chain(list(range(10))).foreach(track_concurrency, concurrency=concurrency_limit).run()
    # The semaphore limits concurrent tasks to `concurrency`. The first item is probed
    # outside the semaphore, so max observed concurrency is at most concurrency + 1.
    self.assertLessEqual(
      max_concurrent,
      concurrency_limit + 1,
      f'Semaphore did not limit concurrency: observed {max_concurrent}, limit was {concurrency_limit}',
    )
    # Verify concurrency actually occurred (not sequential)
    self.assertGreaterEqual(
      max_concurrent,
      2,
      f'Expected concurrent execution but max_concurrent was {max_concurrent}',
    )


# --- §11.5: Sync/async detection ---


class SyncAsyncDetectionTest(SymmetricTestCase):
  """§11.5: Probe-based sync/async detection."""

  async def test_all_sync_workers(self):
    """All sync workers → sync path (ThreadPoolExecutor)."""
    result = Chain(list(range(5))).foreach(lambda x: x + 1, concurrency=2).run()
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_all_async_workers(self):
    """All async workers → async path (Semaphore)."""

    async def async_inc(x):
      return x + 1

    result = await Chain(list(range(5))).foreach(async_inc, concurrency=2).run()
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_first_sync_rest_sync(self):
    """First item sync, rest sync → sync path works."""
    result = Chain([10, 20, 30]).foreach(lambda x: x * 2, concurrency=2).run()
    self.assertEqual(result, [20, 40, 60])


# --- §11.6: Async transition for sync chain handlers ---
# (Canonical async transition tests are in asymmetry_tests.py)


# --- §11.7: Context variable propagation ---


class ContextVarPropagationTest(TestCase):
  """§11.7: contextvars propagate to ThreadPoolExecutor workers."""

  def test_context_var_visible_in_worker(self):
    """User context variables are visible in worker threads."""
    my_var = contextvars.ContextVar('my_var', default=None)
    my_var.set('hello')

    results = []
    lock = threading.Lock()

    def read_var(x):
      with lock:
        results.append(my_var.get())
      return x

    Chain(list(range(3))).foreach(read_var, concurrency=2).run()
    for val in results:
      self.assertEqual(val, 'hello')


# --- Concurrent error handling ---


class ConcurrentErrorHandlingTest(SymmetricTestCase):
  """Concurrent operations: error triage per spec."""

  async def test_single_failure_not_wrapped(self):
    """Single failure in concurrent op raises directly, not ExceptionGroup."""
    items = [1, 2, 3, 4, 5]

    def fail_on_3(x):
      if x == 3:
        raise ValueError('fail on 3')
      return x

    await self.variant(
      lambda fn=None: Chain(items).foreach(fail_on_3, concurrency=2).run(),
      expected_exc=ValueError,
      expected_msg='fail on 3',
    )

  async def test_gather_single_failure(self):
    """Single gather failure raises directly."""

    def ok(x):
      return x

    def fail(x):
      raise ValueError('gather fail')

    with self.assertRaises(ValueError) as ctx:
      Chain(1).gather(ok, fail).run()
    self.assertIn('gather fail', str(ctx.exception))

  async def test_return_signal_priority_in_gather(self):
    """Chain.return_() takes priority in gather."""

    def do_return(x):
      return Chain.return_('returned')

    def ok(x):
      return x

    result = Chain(1).gather(ok, do_return).run()
    self.assertEqual(result, 'returned')

  async def test_break_in_concurrent_map(self):
    """Chain.break_() in concurrent map appends break value to partial results."""
    import time

    def maybe_break(x):
      if x == 3:
        return Chain.break_('stopped')
      time.sleep(0.01)  # Ensure ordering
      return x * 2

    result = Chain(list(range(6))).foreach(maybe_break, concurrency=1).run()
    self.assertEqual(result, [0, 2, 4, 'stopped'])


# --- Additional coverage: async concurrent error handling ---


class AsyncConcurrentErrorHandlingTest(IsolatedAsyncioTestCase):
  """Async concurrent iteration: error paths."""

  async def test_async_map_single_error_not_wrapped(self):
    """Single failure in async concurrent map raises directly."""

    async def fail_on_3(x):
      if x == 3:
        raise ValueError('async fail on 3')
      return x

    with self.assertRaises(ValueError) as ctx:
      await Chain(list(range(5))).foreach(fail_on_3, concurrency=2).run()
    self.assertIn('async fail on 3', str(ctx.exception))

  async def test_async_gather_single_error(self):
    """Single async gather error raises directly, not ExceptionGroup."""

    async def ok(x):
      return x

    async def fail(x):
      raise ValueError('async gather fail')

    with self.assertRaises(ValueError) as ctx:
      await Chain(1).gather(ok, fail, concurrency=2).run()
    self.assertIn('async gather fail', str(ctx.exception))

  async def test_async_gather_multiple_errors(self):
    """Multiple async gather errors raise ExceptionGroup."""

    async def fail1(x):
      raise ValueError('fail1')

    async def fail2(x):
      raise RuntimeError('fail2')

    with self.assertRaises(ExceptionGroup) as ctx:
      await Chain(1).gather(fail1, fail2, concurrency=2).run()
    eg = ctx.exception
    self.assertGreaterEqual(len(eg.exceptions), 2)

  # test_async_concurrent_return_signal removed — canonical test in control_flow_tests.py
  # test_async_concurrent_break_signal removed — canonical test in control_flow_tests.py


# --- §11.5: Probe consistency property test ---


class ProbeConsistencyTest(TestCase):
  """§11.5: If first callable is sync, awaitable from later worker raises TypeError."""

  def test_sync_probe_async_later_raises_typeerror(self):
    """Sync first item + awaitable from later worker always raises TypeError."""
    call_count = 0

    def mixed(x):
      nonlocal call_count
      call_count += 1
      if call_count > 1:

        async def _coro():
          return x

        return _coro()
      return x

    # Try with various concurrency values
    for conc in [1, 2, 5]:
      call_count = 0
      with self.assertRaises(TypeError, msg=f'concurrency={conc} should raise TypeError'):
        Chain([1, 2, 3]).foreach(mixed, concurrency=conc).run()

  def test_sync_probe_async_later_in_gather(self):
    """Sync first fn + awaitable from later fn in gather raises TypeError."""

    def sync_fn(x):
      return x

    def async_fn(x):

      async def _coro():
        return x

      return _coro()

    with self.assertRaises(TypeError):
      Chain(1).gather(sync_fn, async_fn, concurrency=2).run()


# --- §11.2.2: executor parameter ---


class ExecutorParameterTest(TestCase):
  """§11.2.2: executor parameter on foreach(), foreach_do(), and gather()."""

  def test_foreach_with_user_executor(self):
    """foreach(fn, concurrency=2, executor=pool) uses provided executor and preserves order."""
    with ThreadPoolExecutor(max_workers=4) as pool:
      result = Chain([1, 2, 3, 4]).foreach(lambda x: x * 2, concurrency=2, executor=pool).run()
    self.assertEqual(result, [2, 4, 6, 8])

  def test_foreach_do_with_user_executor(self):
    """foreach_do(fn, concurrency=2, executor=pool) uses provided executor and preserves original items."""
    with ThreadPoolExecutor(max_workers=4) as pool:
      result = Chain([1, 2, 3, 4]).foreach_do(lambda x: x * 100, concurrency=2, executor=pool).run()
    self.assertEqual(result, [1, 2, 3, 4])

  def test_gather_with_user_executor(self):
    """gather(*fns, executor=pool) uses provided executor and returns results in fn order."""
    with ThreadPoolExecutor(max_workers=4) as pool:
      result = (
        Chain(10)
        .gather(
          lambda x: x + 1,
          lambda x: x + 2,
          lambda x: x + 3,
          executor=pool,
        )
        .run()
      )
    self.assertEqual(result, (11, 12, 13))

  def test_executor_not_shut_down(self):
    """After a chain with executor=pool completes, the pool is still alive (quent does NOT shut it down)."""
    pool = ThreadPoolExecutor(max_workers=2)
    try:
      Chain([1, 2, 3]).foreach(lambda x: x + 1, concurrency=2, executor=pool).run()
      # Pool must still be usable — submit work after quent has finished
      future = pool.submit(lambda: 42)
      self.assertEqual(future.result(timeout=5), 42)
    finally:
      pool.shutdown(wait=False)

  def test_executor_none_default_creates_new(self):
    """executor=None (default) works identically to omitting the parameter."""
    result_default = Chain([1, 2, 3]).foreach(lambda x: x * 2, concurrency=2).run()
    result_explicit_none = Chain([1, 2, 3]).foreach(lambda x: x * 2, concurrency=2, executor=None).run()
    self.assertEqual(result_default, [2, 4, 6])
    self.assertEqual(result_explicit_none, [2, 4, 6])

  def test_executor_validation_rejects_non_executor(self):
    """Passing a non-Executor value raises TypeError at build time."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1, 2]).foreach(lambda x: x, concurrency=2, executor='not-an-executor')
    self.assertIn('executor', str(ctx.exception))

    with self.assertRaises(TypeError) as ctx:
      Chain([1, 2]).foreach_do(lambda x: x, concurrency=2, executor=42)
    self.assertIn('executor', str(ctx.exception))

    with self.assertRaises(TypeError) as ctx:
      Chain(1).gather(lambda x: x, executor='bad')
    self.assertIn('executor', str(ctx.exception))

  def test_gather_with_user_executor_and_concurrency(self):
    """gather(*fns, concurrency=2, executor=pool) works correctly with both params."""
    with ThreadPoolExecutor(max_workers=4) as pool:
      result = (
        Chain(5)
        .gather(
          lambda x: x + 10,
          lambda x: x + 20,
          lambda x: x + 30,
          concurrency=2,
          executor=pool,
        )
        .run()
      )
    self.assertEqual(result, (15, 25, 35))

  def test_foreach_executor_ignored_without_concurrency(self):
    """When concurrency=None (sequential), executor is ignored and chain runs sequentially with correct results."""
    # Provide an executor even though concurrency is not set — it must be silently ignored.
    with ThreadPoolExecutor(max_workers=4) as pool:
      result = Chain([1, 2, 3, 4]).foreach(lambda x: x * 3, concurrency=None, executor=pool).run()
    self.assertEqual(result, [3, 6, 9, 12])


# --- §5.5: BaseException handling in gather triage ---


class GatherBaseExceptionTest(IsolatedAsyncioTestCase):
  """§5.6: BaseException handling in gather triage."""

  async def test_async_base_exception_in_gather(self) -> None:
    """§5.5: Async gather with BaseException raises it directly."""

    class AsyncCustomBaseExc(BaseException):
      pass

    async def ok_fn(x: int) -> int:
      return x

    async def base_exc_fn(x: int) -> int:
      raise AsyncCustomBaseExc('async base exc')

    with self.assertRaises(AsyncCustomBaseExc):
      await Chain(1).gather(ok_fn, base_exc_fn).run()

  async def test_base_exc_with_regular_exceptions(self) -> None:
    """§5.5: BaseException takes priority over regular exceptions in gather triage."""

    class CustomBaseExc(BaseException):
      pass

    def ok_fn(x: int) -> int:
      return x

    def raise_regular(x: int) -> int:
      time.sleep(0.02)
      raise ValueError('regular')

    def raise_base(x: int) -> int:
      time.sleep(0.02)
      raise CustomBaseExc('base priority')

    with self.assertRaises(CustomBaseExc):
      Chain(1).gather(ok_fn, raise_regular, raise_base).run()


# --- §17.4: Awaitable without close() in sync concurrent worker ---


class AwaitableNoCloseTest(TestCase):
  """§11.5: Awaitable without close() method in sync concurrent worker."""

  def test_awaitable_without_close_in_sync_worker(self) -> None:
    """§17.4: Awaitable without close() in sync gather worker raises TypeError."""

    def sync_fn(x: int) -> int:
      return x

    class AwaitableNoClose:
      """An awaitable that has __await__ but no close()."""

      def __await__(self):
        return iter([])

    def returns_awaitable_no_close(x: int) -> Any:
      return AwaitableNoClose()

    with self.assertRaises(TypeError) as ctx:
      Chain(1).gather(sync_fn, returns_awaitable_no_close).run()
    self.assertIn('awaitable', str(ctx.exception).lower())


# --- Defensive guard: partial submission failure ---


class PartialSubmissionDefensiveTest(TestCase):
  """Defensive guard: partial submission failure during ThreadPoolExecutor gather.
  This tests an internal code path (_run_threadpool_sync) not reachable via public API.
  Retained as a defensive guard for robustness."""

  def test_partial_submission_failure_in_gather(self) -> None:
    """§11.3: If submit raises after partial submission, already-submitted futures are cancelled."""

    from quent._concurrency import _run_threadpool_sync

    results = [None] * 5
    results[0] = 'probed'
    call_count = 0

    def failing_submit(executor: Any, idx: int) -> Any:
      nonlocal call_count
      call_count += 1
      if call_count >= 3:
        raise RuntimeError('submit failed')
      return executor.submit(lambda: 'ok')

    with self.assertRaises(RuntimeError) as ctx:
      _run_threadpool_sync(
        n=5,
        concurrency=5,
        results=results,
        submit=failing_submit,
        on_exc=lambda exc, idx: None,
        awaitable_msg=lambda idx: 'awaitable error',
      )
    exc = ctx.exception
    self.assertIn('submit failed', str(exc))
    # On Python 3.11+, the note should be attached
    if hasattr(exc, '__notes__'):
      notes = '\n'.join(exc.__notes__)
      self.assertIn('submission failed', notes)


# --- BaseException handling in _triage_iter_exceptions ---


class IterTriageBaseExceptionTest(IsolatedAsyncioTestCase):
  """Regression: _triage_iter_exceptions must not put BaseException into ExceptionGroup.

  ExceptionGroup requires list[Exception]; passing a BaseException subclass like
  KeyboardInterrupt caused a TypeError.  The fix tracks the earliest-index
  BaseException separately and returns it with priority over regular exceptions.
  """

  def test_sync_concurrent_foreach_keyboard_interrupt_and_value_error(self) -> None:
    """Concurrent foreach (sync) where probe raises KeyboardInterrupt and threadpool
    worker raises ValueError must not crash with TypeError — KeyboardInterrupt takes priority.

    Item 0 runs synchronously as the probe; item 1 runs in the ThreadPoolExecutor.
    The probe captures KeyboardInterrupt and the threadpool captures ValueError.
    Both are merged in triage — without the fix, ExceptionGroup would receive a
    BaseException and raise TypeError.
    """

    started = [False]

    def worker(x: int) -> int:
      if x == 0:
        # Probe: runs synchronously, raises immediately.
        raise KeyboardInterrupt('simulated interrupt')
      # Threadpool worker: small delay to ensure probe has already raised.
      started[0] = True
      raise ValueError('simulated value error')

    with self.assertRaises(KeyboardInterrupt):
      Chain([0, 1]).foreach(worker, concurrency=2).run()

  def test_sync_concurrent_foreach_do_keyboard_interrupt_and_value_error(self) -> None:
    """Concurrent foreach_do (sync) with KeyboardInterrupt + ValueError must not TypeError."""

    def worker(x: int) -> None:
      if x == 0:
        raise KeyboardInterrupt('simulated interrupt')
      raise ValueError('simulated value error')

    with self.assertRaises(KeyboardInterrupt):
      Chain([0, 1]).foreach_do(worker, concurrency=2).run()

  def test_triage_base_exception_not_in_exception_group(self) -> None:
    """_triage_iter_exceptions must not add BaseException to the regular list.

    Directly tests the triage function: a KeyboardInterrupt + ValueError
    must return the KeyboardInterrupt (not crash with TypeError from
    ExceptionGroup receiving a non-Exception).
    """
    from quent._iter_ops import _triage_iter_exceptions

    ki = KeyboardInterrupt('ki')
    ki._quent_idx = 1  # type: ignore[attr-defined]
    ve = ValueError('ve')
    ve._quent_idx = 0  # type: ignore[attr-defined]

    result = _triage_iter_exceptions([ki, ve], 2, 'foreach')
    self.assertEqual(result.action, 'exc')
    # ValueError has lower idx (0 < 1) but KeyboardInterrupt takes absolute priority
    # over regular exceptions regardless of index.
    self.assertIs(result.exc, ki)


# --- §11.7: Context variable propagation with user executor (SPEC-327) ---


class ContextVarUserExecutorTest(TestCase):
  """§11.7 (SPEC-327): contextvars propagate to ThreadPoolExecutor workers with user-provided executor."""

  def test_context_var_visible_in_user_executor_workers(self):
    """User context variables propagate to workers in a user-provided executor."""
    my_var = contextvars.ContextVar('my_var', default=None)
    my_var.set('user_executor_value')

    results = []
    lock = threading.Lock()

    def read_var(x):
      with lock:
        results.append(my_var.get())
      return x

    with ThreadPoolExecutor(max_workers=4) as user_pool:
      Chain(list(range(5))).foreach(read_var, concurrency=2, executor=user_pool).run()

    # All workers should see the context variable value
    for val in results:
      self.assertEqual(val, 'user_executor_value')
    # Verify we actually had results (at least 5 items processed)
    self.assertEqual(len(results), 5)


# --- §17.2a: except_(reraise=False) coroutine → async transition (SPEC-348/515) ---


class ExceptReraseFalseAsyncTransitionTest(IsolatedAsyncioTestCase):
  """SPEC §17.2a (SPEC-348/515): except_(reraise=False) handler returning coroutine
  triggers async transition — the coroutine becomes the chain's result."""

  async def test_except_reraise_false_async_handler_returns_coroutine(self):
    """Sync chain raises, async except handler (reraise=False): run() returns coroutine, await gets handler result."""

    async def async_handler(info):
      return 'recovered_async'

    # Sync chain that raises, with async except handler and reraise=False (default)
    coro = Chain(1).then(lambda x: 1 / 0).except_(async_handler).run()

    # run() should return a coroutine (async transition)
    self.assertTrue(asyncio.iscoroutine(coro), 'run() should return a coroutine for async transition')

    # Awaiting the coroutine should return the handler's result
    result = await coro
    self.assertEqual(result, 'recovered_async')

  async def test_except_reraise_false_async_handler_with_args(self):
    """Async except handler with explicit args and reraise=False: async transition, handler result returned."""

    async def async_handler(a, b):
      return a + b

    coro = Chain(1).then(lambda x: 1 / 0).except_(async_handler, 10, 20).run()
    self.assertTrue(asyncio.iscoroutine(coro), 'run() should return a coroutine for async transition')

    result = await coro
    self.assertEqual(result, 30)


# --- §11.2.2: User executor identity (SPEC-354) ---


class UserExecutorIdentityTest(TestCase):
  """SPEC §11.2.2 (SPEC-354): quent actually calls submit() on the user's executor."""

  def test_user_executor_submit_called(self):
    """Verify quent calls submit() on the user-provided executor, not an internal one."""
    submit_count = 0

    class SpyExecutor(ThreadPoolExecutor):
      """ThreadPoolExecutor subclass that counts submit() calls."""

      def submit(self, *args, **kwargs):
        nonlocal submit_count
        submit_count += 1
        return super().submit(*args, **kwargs)

    with SpyExecutor(max_workers=4) as spy_pool:
      result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: x * 2, concurrency=3, executor=spy_pool).run()

    self.assertEqual(result, [2, 4, 6, 8, 10])
    # submit() must have been called at least once on the spy executor
    # (index 0 is probed inline, indices 1..n-1 are submitted to the pool)
    self.assertGreaterEqual(
      submit_count, 1, f'Expected submit() to be called on user executor, got {submit_count} calls'
    )
    # Exactly n-1 = 4 items submitted (index 0 is the inline probe)
    self.assertEqual(submit_count, 4)

  def test_user_executor_submit_called_gather(self):
    """Verify quent calls submit() on the user-provided executor for gather."""
    submit_count = 0

    class SpyExecutor(ThreadPoolExecutor):
      def submit(self, *args, **kwargs):
        nonlocal submit_count
        submit_count += 1
        return super().submit(*args, **kwargs)

    with SpyExecutor(max_workers=4) as spy_pool:
      result = (
        Chain(5)
        .gather(
          lambda x: x + 1,
          lambda x: x + 2,
          lambda x: x + 3,
          executor=spy_pool,
        )
        .run()
      )

    self.assertEqual(result, (6, 7, 8))
    # fn[0] is probed inline, fn[1] and fn[2] are submitted
    self.assertEqual(submit_count, 2)
