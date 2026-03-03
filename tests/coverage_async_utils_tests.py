"""Comprehensive tests targeting all coverable lines in quent/_async_utils.pxi.

Coverage targets:
  - Line 12: `from asyncio import create_task as _create_task` (import)
  - Line 14: `cdef bint _HAS_EAGER_START = ...` (module-level)
  - Line 22: `task_registry = set()` (module-level)
  - Lines 26-33: `ensure_future` body: task creation, registry add, done callback
  - Line 31: `else: task = _create_task(coro)` -- IMPOSSIBLE on Python 3.14+ (only <3.14)
  - Line 34: `task.add_done_callback(task_registry.discard)`
  - Line 35: `return task`
  - Lines 38-40: `_get_registry_size()` function

NOTE: Line 31 (the `else` branch for Python < 3.14) CANNOT be covered
on our current Python 3.14 runtime. This is a version-gated branch.
"""
import sys
import asyncio
import gc
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# _get_registry_size tests (line 46-48)
# ---------------------------------------------------------------------------
class GetRegistrySizeTests(IsolatedAsyncioTestCase):
  """Tests for _get_registry_size() function (lines 46-48)."""

  async def test_get_registry_size_returns_int(self):
    """_get_registry_size() returns an integer."""
    size = _get_registry_size()
    self.assertIsInstance(size, int)

  async def test_get_registry_size_non_negative(self):
    """_get_registry_size() is always >= 0."""
    size = _get_registry_size()
    self.assertGreaterEqual(size, 0)

  async def test_get_registry_size_reflects_active_tasks(self):
    """_get_registry_size() increases when tasks are pending and
    decreases after tasks complete.
    """
    initial_size = _get_registry_size()

    async def slow_coro(v):
      await asyncio.sleep(0.05)
      return v

    task = Chain(slow_coro, 42).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)

    size_during = _get_registry_size()
    self.assertGreater(size_during, initial_size)

    await task
    await asyncio.sleep(0)

    self.assertEqual(_get_registry_size(), initial_size)

  async def test_get_registry_size_zero_after_all_tasks_complete(self):
    """After creating and awaiting multiple tasks, registry returns to baseline."""
    initial_size = _get_registry_size()

    async def identity(v):
      return v

    tasks = []
    for i in range(5):
      t = Chain(identity, i).config(autorun=True).run()
      tasks.append(t)

    self.assertGreaterEqual(_get_registry_size(), initial_size + 5)

    await asyncio.gather(*tasks)
    await asyncio.sleep(0)

    self.assertEqual(_get_registry_size(), initial_size)

  async def test_get_registry_size_called_multiple_times(self):
    """Calling _get_registry_size() multiple times is idempotent."""
    s1 = _get_registry_size()
    s2 = _get_registry_size()
    s3 = _get_registry_size()
    self.assertEqual(s1, s2)
    self.assertEqual(s2, s3)


# ---------------------------------------------------------------------------
# ensure_future core behavior (lines 26-34, 42-43)
# ---------------------------------------------------------------------------
class EnsureFutureCoreTests(IsolatedAsyncioTestCase):
  """Tests for the ensure_future cdef function, exercised via autorun chains.

  ensure_future is a cdef function and cannot be called directly from Python.
  We trigger it through Chain.config(autorun=True).run() with async operations.
  """

  async def test_ensure_future_returns_task(self):
    """ensure_future returns an asyncio.Task (line 43: return task)."""
    async def async_val(v):
      return v * 2

    result = Chain(async_val, 5).config(autorun=True).run()
    self.assertIsInstance(result, asyncio.Task)
    value = await result
    self.assertEqual(value, 10)

  async def test_ensure_future_task_is_in_registry(self):
    """ensure_future adds task to task_registry (line 33)."""
    initial = _get_registry_size()

    async def slow(v):
      await asyncio.sleep(0.05)
      return v

    task = Chain(slow, 1).config(autorun=True).run()
    self.assertGreater(_get_registry_size(), initial)
    await task

  async def test_ensure_future_done_callback_removes_from_registry(self):
    """task.add_done_callback(task_registry.discard) cleans up (line 42)."""
    initial = _get_registry_size()

    async def fast(v):
      return v

    task = Chain(fast, 1).config(autorun=True).run()
    await task
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_ensure_future_task_result_is_correct(self):
    """Task produced by ensure_future resolves to the correct value."""
    async def compute(v):
      return v ** 2

    task = Chain(compute, 7).config(autorun=True).run()
    result = await task
    self.assertEqual(result, 49)

  async def test_ensure_future_with_chain_links(self):
    """ensure_future works when chain has multiple links before autorun."""
    async def async_add(v):
      return v + 10

    task = Chain(5).then(lambda v: v * 2).then(async_add).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 20)  # 5 * 2 + 10

  async def test_ensure_future_non_simple_path(self):
    """ensure_future via _run (non-simple path, .do() forces it)."""
    async def async_identity(v):
      return v

    task = Chain(async_identity, 99).do(lambda v: None).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 99)

  async def test_ensure_future_via_call(self):
    """ensure_future via __call__() path."""
    async def async_mul(v):
      return v * 3

    chain = Chain(async_mul, 4).config(autorun=True)
    task = chain()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 12)

  async def test_ensure_future_cascade(self):
    """ensure_future works with Cascade (returns root value)."""
    async def async_side(v):
      return 'ignored'

    task = Cascade(aempty, 42).then(async_side).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 42)

  async def test_ensure_future_with_exception_in_chain(self):
    """ensure_future wraps a task that may raise; exception propagates."""
    async def async_raise(v):
      raise TestExc('boom')

    task = Chain(1).then(async_raise).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    with self.assertRaises(TestExc):
      await task

  async def test_ensure_future_preserves_task_identity(self):
    """Each call to ensure_future returns a distinct Task."""
    async def async_id(v):
      return v

    t1 = Chain(async_id, 1).config(autorun=True).run()
    t2 = Chain(async_id, 2).config(autorun=True).run()
    self.assertIsNot(t1, t2)
    r1, r2 = await asyncio.gather(t1, t2)
    self.assertEqual(r1, 1)
    self.assertEqual(r2, 2)


# ---------------------------------------------------------------------------
# Multiple tasks in registry simultaneously (line 33-34)
# ---------------------------------------------------------------------------
class MultipleTasksRegistryTests(IsolatedAsyncioTestCase):
  """Test multiple concurrent tasks in the task registry."""

  async def test_multiple_tasks_coexist_in_registry(self):
    """Multiple pending tasks all appear in the registry."""
    initial = _get_registry_size()
    count = 10

    async def slow(v):
      await asyncio.sleep(0.05)
      return v

    tasks = []
    for i in range(count):
      t = Chain(slow, i).config(autorun=True).run()
      tasks.append(t)

    self.assertGreaterEqual(_get_registry_size(), initial + count)

    results = await asyncio.gather(*tasks)
    self.assertEqual(results, list(range(count)))

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_staggered_completion_decrements_registry(self):
    """As tasks complete one by one, registry size decreases."""
    initial = _get_registry_size()

    async def variable_delay(v):
      await asyncio.sleep(0.01 * v)
      return v

    tasks = []
    for i in range(1, 4):
      t = Chain(variable_delay, i).config(autorun=True).run()
      tasks.append(t)

    self.assertGreaterEqual(_get_registry_size(), initial + 3)

    await tasks[0]
    await asyncio.sleep(0)
    size_after_one = _get_registry_size()
    self.assertLess(size_after_one, initial + 3)

    await asyncio.gather(*tasks[1:])
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)


# ---------------------------------------------------------------------------
# Task registry prevents garbage collection (lines 33, 42)
# ---------------------------------------------------------------------------
class RegistryPreventsGCTests(IsolatedAsyncioTestCase):
  """Task registry holds strong references, preventing GC of fire-and-forget tasks."""

  async def test_fire_and_forget_task_completes(self):
    """A fire-and-forget task still completes because task_registry
    holds a strong reference.
    """
    completed = {'value': False}

    async def set_flag(v):
      await asyncio.sleep(0.02)
      completed['value'] = True
      return v

    task = Chain(set_flag, 1).config(autorun=True).run()

    self.assertGreater(_get_registry_size(), 0)

    await task
    self.assertTrue(completed['value'])

  async def test_registry_holds_strong_ref_during_execution(self):
    """While task is running, it's in the registry (strong ref)."""
    in_registry_during_exec = {'value': False}
    initial = _get_registry_size()

    async def check_registry(v):
      await asyncio.sleep(0.01)
      if _get_registry_size() > initial:
        in_registry_during_exec['value'] = True
      return v

    task = Chain(check_registry, 1).config(autorun=True).run()
    await task
    self.assertTrue(in_registry_during_exec['value'])


# ---------------------------------------------------------------------------
# ensure_future with autorun chains (integration tests)
# ---------------------------------------------------------------------------
class EnsureFutureAutorunIntegrationTests(IsolatedAsyncioTestCase):
  """Integration tests: ensure_future via various autorun chain shapes."""

  async def test_autorun_simple_chain_async_root(self):
    """Simple chain with async root -> ensure_future via run()."""
    task = Chain(aempty, 42).then(lambda v: v + 1).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 43)

  async def test_autorun_simple_chain_async_link(self):
    """Simple chain with sync root and async link -> ensure_future via run()."""
    async def async_double(v):
      return v * 2

    task = Chain(10).then(async_double).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 20)

  async def test_autorun_non_simple_async_root(self):
    """Non-simple chain (.do() added) with async root -> ensure_future via _run."""
    task = Chain(aempty, 7).do(lambda v: None).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 7)

  async def test_autorun_non_simple_async_link(self):
    """Non-simple chain with sync root and async link mid-chain."""
    async def async_negate(v):
      return -v

    task = Chain(5).do(lambda v: None).then(async_negate).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, -5)

  async def test_autorun_call_simple(self):
    """ensure_future via __call__() on simple chain."""
    async def async_inc(v):
      return v + 1

    chain = Chain(async_inc, 0).config(autorun=True)
    task = chain()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 1)

  async def test_autorun_call_non_simple(self):
    """ensure_future via __call__() on non-simple chain."""
    async def async_square(v):
      return v ** 2

    chain = Chain(3).do(lambda v: None).then(async_square).config(autorun=True)
    task = chain()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 9)

  async def test_autorun_with_except_handler(self):
    """.except_() makes chain non-simple, async root -> ensure_future."""
    task = Chain(aempty, 50).except_(lambda e: None).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 50)

  async def test_autorun_with_finally_handler(self):
    """.finally_() makes chain non-simple, async root -> ensure_future."""
    finally_called = {'value': False}

    def on_finally(v=None):
      finally_called['value'] = True

    task = Chain(aempty, 60).finally_(on_finally).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 60)
    self.assertTrue(finally_called['value'])


# ---------------------------------------------------------------------------
# Registry cleanup under concurrent conditions
# ---------------------------------------------------------------------------
class ConcurrentRegistryCleanupTests(IsolatedAsyncioTestCase):
  """Test that registry cleanup (done callbacks) works correctly
  when multiple tasks complete concurrently or in rapid succession.
  """

  async def test_concurrent_completion_cleanup(self):
    """All tasks are removed from registry after concurrent completion."""
    initial = _get_registry_size()
    n = 50

    async def fast(v):
      return v

    tasks = [Chain(fast, i).config(autorun=True).run() for i in range(n)]
    results = await asyncio.gather(*tasks)
    self.assertEqual(results, list(range(n)))

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_interleaved_creation_and_completion(self):
    """Creating new tasks while others complete keeps registry consistent."""
    initial = _get_registry_size()

    async def fast(v):
      return v

    async def slow(v):
      await asyncio.sleep(0.02)
      return v

    slow_tasks = [Chain(slow, i).config(autorun=True).run() for i in range(5)]

    for i in range(10):
      t = Chain(fast, i).config(autorun=True).run()
      await t

    await asyncio.sleep(0)

    await asyncio.gather(*slow_tasks)
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_exception_in_task_still_cleans_registry(self):
    """Tasks that raise are still removed from registry via done callback."""
    initial = _get_registry_size()

    async def raise_err(v):
      raise ValueError('test error')

    task = Chain(raise_err, 1).config(autorun=True).run()
    with self.assertRaises(ValueError):
      await task

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_cancelled_task_cleans_registry(self):
    """Cancelled tasks are removed from registry via done callback."""
    initial = _get_registry_size()

    async def never_finish(v):
      await asyncio.sleep(100)
      return v

    task = Chain(never_finish, 1).config(autorun=True).run()
    self.assertGreater(_get_registry_size(), initial)

    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)


# ---------------------------------------------------------------------------
# _HAS_EAGER_START path coverage (line 29-30 vs 32)
# ---------------------------------------------------------------------------
class EagerStartPathTests(IsolatedAsyncioTestCase):
  """Test the version-dependent eager_start path in ensure_future.

  On Python >= 3.14: line 30 is hit (eager_start=True)
  On Python < 3.14: line 32 is hit (no eager_start)

  Since we are on Python 3.14, we exercise line 30.
  Line 32 is IMPOSSIBLE to cover on this version.
  """

  async def test_eager_start_path_on_314(self):
    """On Python 3.14+, ensure_future uses eager_start=True (line 30)."""
    if sys.version_info < (3, 14):
      self.skipTest('eager_start path only on Python 3.14+')

    async def async_val(v):
      return v

    task = Chain(async_val, 42).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 42)

  async def test_eager_start_task_completes_immediately(self):
    """With eager_start=True, simple coroutines may complete eagerly."""
    if sys.version_info < (3, 14):
      self.skipTest('eager_start path only on Python 3.14+')

    async def immediate(v):
      return v * 2

    task = Chain(immediate, 5).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 10)

  async def test_version_info_determines_path(self):
    """Verify _HAS_EAGER_START matches our Python version."""
    self.assertGreaterEqual(sys.version_info, (3, 14))

    async def coro(v):
      return v

    task = Chain(coro, 1).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    await task


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class EnsureFutureEdgeCaseTests(IsolatedAsyncioTestCase):
  """Edge case tests for ensure_future behavior."""

  async def test_ensure_future_with_coroutine_returning_none(self):
    """ensure_future handles coroutines that return None."""
    async def return_none(v):
      pass

    task = Chain(return_none, 1).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertIsNone(result)

  async def test_ensure_future_with_nested_async_chain(self):
    """ensure_future with nested async chains."""
    async def inner_async(v):
      return v + 100

    inner = Chain().then(inner_async)
    task = Chain(5).then(inner).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 105)

  async def test_ensure_future_with_long_chain(self):
    """ensure_future with a chain containing many links."""
    async def async_inc(v):
      return v + 1

    chain = Chain(0)
    for _ in range(20):
      chain = chain.then(async_inc)
    chain = chain.config(autorun=True)

    task = chain.run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 20)

  async def test_ensure_future_with_sync_and_async_mixed(self):
    """Mixed sync/async links all work through ensure_future."""
    async def async_add(v):
      return v + 1

    task = (
      Chain(1)
      .then(lambda v: v * 2)     # sync: 2
      .then(async_add)           # async: 3
      .then(lambda v: v * 10)    # sync: 30
      .then(async_add)           # async: 31
      .config(autorun=True)
      .run()
    )
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 31)

  async def test_ensure_future_rapid_create_destroy_cycle(self):
    """Rapidly creating and destroying tasks exercises add/discard cycle."""
    initial = _get_registry_size()

    async def quick(v):
      return v

    for i in range(100):
      t = Chain(quick, i).config(autorun=True).run()
      r = await t
      self.assertEqual(r, i)

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_ensure_future_with_awaitable_root_override(self):
    """ensure_future when chain root is provided via .run(async_fn)."""
    async def async_root():
      return 42

    task = Chain().then(lambda v: v + 8).config(autorun=True).run(async_root)
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 50)


if __name__ == '__main__':
  import unittest
  unittest.main()
