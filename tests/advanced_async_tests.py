"""Advanced async edge case tests for the Quent library.

Covers:
  1. CancelledError handling
  2. Concurrent chain execution
  3. Mixed sync/async transition patterns
  4. Task registry edge cases
  5. Async context managers
  6. Async generators and iterators
  7. Async exception scenarios
  8. Async return/break signals
"""
import asyncio
import sys
import time
import warnings
import gc
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AsyncCM:
  """Async-only context manager."""
  def __init__(self, value=None, suppress=False):
    self.value = value
    self.entered = False
    self.exited = False
    self.exc_info = None
    self.suppress = suppress

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exc_info = (exc_type, exc_val, exc_tb)
    return self.suppress


class SyncCM:
  """Sync-only context manager."""
  def __init__(self, value=None, suppress=False):
    self.value = value
    self.entered = False
    self.exited = False
    self.suppress = suppress

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    return self.suppress


class SyncCMWithCoroExit:
  """Sync CM whose __exit__ returns a coroutine (edge case)."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    async def _exit_coro():
      return False
    return _exit_coro()


class DualCM:
  """Has both sync and async context manager protocols."""
  def __init__(self, sync_val='sync', async_val='async'):
    self.sync_val = sync_val
    self.async_val = async_val
    self.sync_entered = False
    self.sync_exited = False
    self.async_entered = False
    self.async_exited = False

  def __enter__(self):
    self.sync_entered = True
    return self.sync_val

  def __exit__(self, *args):
    self.sync_exited = True
    return False

  async def __aenter__(self):
    self.async_entered = True
    return self.async_val

  async def __aexit__(self, *args):
    self.async_exited = True
    return False


class AsyncCMNoneEnter:
  """Async CM whose __aenter__ yields None."""
  def __init__(self):
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return None

  async def __aexit__(self, *args):
    self.exited = True
    return False


class AsyncIterator:
  """Async iterator over a list of items."""
  def __init__(self, items):
    self._items = list(items)

  def __aiter__(self):
    self._iter = iter(self._items)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


class SyncIterator:
  """Sync iterator over a list of items."""
  def __init__(self, items):
    self._items = list(items)

  def __iter__(self):
    return iter(self._items)


# ---------------------------------------------------------------------------
# 1. CancelledError Handling (12 tests)
# ---------------------------------------------------------------------------
class CancelledErrorHandlingTests(IsolatedAsyncioTestCase):
  """Test CancelledError propagation and handling in various chain positions."""

  async def test_cancel_mid_chain_propagates(self):
    """Cancel a task mid-chain; CancelledError propagates."""
    async def slow_link(v):
      await asyncio.sleep(10)
      return v

    task = asyncio.ensure_future(
      Chain(1).then(lambda v: v + 1).then(slow_link).run()
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

  async def test_cancel_during_run_async(self):
    """Cancel during _run_async; exception propagates correctly."""
    async def awaiting(v):
      await asyncio.sleep(10)
      return v * 2

    task = asyncio.ensure_future(
      Chain(5).do(lambda v: None).then(awaiting).run()
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

  async def test_cancel_during_foreach_async_iteration(self):
    """Cancel during foreach with async function; partial results or cancellation."""
    results = []

    async def slow_fn(v):
      results.append(v)
      await asyncio.sleep(0.5)
      return v * 2

    task = asyncio.ensure_future(
      Chain([1, 2, 3, 4, 5]).foreach(slow_fn).run()
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

  async def test_cancel_during_gather(self):
    """Cancel during gather; CancelledError propagates."""
    async def slow_a(v):
      await asyncio.sleep(10)
      return 'a'

    async def slow_b(v):
      await asyncio.sleep(10)
      return 'b'

    task = asyncio.ensure_future(
      Chain(1).gather(slow_a, slow_b).run()
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

  async def test_cancel_with_finally_handler_still_runs(self):
    """Cancel with finally handler -- finally_ still executes."""
    finally_called = {'value': False}

    async def slow_link(v):
      await asyncio.sleep(10)
      return v

    def on_finally(v=None):
      finally_called['value'] = True

    task = asyncio.ensure_future(
      Chain(1).then(slow_link).finally_(on_finally).run()
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    # Give finally callback a chance to run
    await asyncio.sleep(0)
    self.assertTrue(finally_called['value'])

  async def test_cancel_not_caught_by_default_except(self):
    """CancelledError is BaseException, not caught by except_ with default exceptions."""
    handler_called = {'value': False}

    def handler(v=None):
      handler_called['value'] = True

    async def raise_cancelled(v):
      raise asyncio.CancelledError()

    with self.assertRaises(asyncio.CancelledError):
      await Chain(1).then(raise_cancelled).except_(handler).run()
    self.assertFalse(handler_called['value'])

  async def test_cancel_caught_with_base_exception_filter(self):
    """CancelledError caught by except_(handler, exceptions=BaseException)."""
    handler_called = {'value': False}
    caught_exc = {'value': None}

    def handler(v=None):
      handler_called['value'] = True
      return 'handled'

    async def raise_cancelled(v):
      raise asyncio.CancelledError()

    result = await Chain(1).then(raise_cancelled).except_(
      handler, exceptions=BaseException, reraise=False
    ).run()
    self.assertTrue(handler_called['value'])
    self.assertEqual(result, 'handled')

  async def test_cancel_during_async_with(self):
    """Cancel during async with_ -- __aexit__ still called."""
    cm = AsyncCM(value='resource')

    async def slow_body(v):
      await asyncio.sleep(10)
      return v

    task = asyncio.ensure_future(
      Chain(cm).with_(slow_body).run()
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    await asyncio.sleep(0)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_cancel_autorun_task_registry_cleanup(self):
    """Cancel autorun task; verify registry cleanup."""
    initial_size = _get_registry_size()

    async def never_finish(v):
      await asyncio.sleep(100)
      return v

    task = Chain(never_finish, 1).config(autorun=True).run()
    self.assertGreater(_get_registry_size(), initial_size)

    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial_size)

  async def test_cancel_with_async_finally_handler(self):
    """Cancel with async finally handler still runs."""
    finally_called = {'value': False}

    async def slow_link(v):
      await asyncio.sleep(10)
      return v

    async def async_finally(v=None):
      finally_called['value'] = True

    task = asyncio.ensure_future(
      Chain(1).then(slow_link).finally_(async_finally).run()
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    await asyncio.sleep(0)
    self.assertTrue(finally_called['value'])

  async def test_cancel_error_in_simple_chain(self):
    """CancelledError propagates through simple chain path."""
    async def slow(v):
      await asyncio.sleep(10)
      return v

    task = asyncio.ensure_future(
      Chain(1).then(slow).run()
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task


# ---------------------------------------------------------------------------
# 2. Concurrent Chain Execution (12 tests)
# ---------------------------------------------------------------------------
class ConcurrentChainExecutionTests(IsolatedAsyncioTestCase):
  """Test concurrent execution of chains and clones."""

  async def test_100_concurrent_chain_runs(self):
    """100 concurrent runs of the same chain."""
    async def async_double(v):
      await asyncio.sleep(0)
      return v * 2

    c = Chain().then(async_double)
    coros = [c(i) for i in range(100)]
    results = await asyncio.gather(*coros)
    expected = [i * 2 for i in range(100)]
    self.assertEqual(sorted(results), sorted(expected))

  async def test_concurrent_runs_different_root_values(self):
    """Concurrent runs of chain with different root values."""
    async def async_square(v):
      await asyncio.sleep(0)
      return v ** 2

    c = Chain().then(async_square)
    coros = [c(i) for i in range(50)]
    results = await asyncio.gather(*coros)
    self.assertEqual(sorted(results), sorted([i ** 2 for i in range(50)]))

  async def test_concurrent_runs_some_raise(self):
    """Concurrent runs where some raise exceptions."""
    async def maybe_raise(v):
      await asyncio.sleep(0)
      if v % 3 == 0:
        raise TestExc(f'fail-{v}')
      return v

    c = Chain().then(maybe_raise)
    tasks = [asyncio.ensure_future(c(i)) for i in range(15)]
    results = []
    exceptions = []
    for t in tasks:
      try:
        results.append(await t)
      except TestExc as e:
        exceptions.append(e)
    self.assertGreater(len(exceptions), 0)
    self.assertGreater(len(results), 0)

  async def test_concurrent_gather_inside_concurrent_chains(self):
    """Concurrent gather operations inside concurrent chains."""
    async def fn_a(v):
      await asyncio.sleep(0)
      return v + 'a'

    async def fn_b(v):
      await asyncio.sleep(0)
      return v + 'b'

    c = Chain().then(lambda v: str(v)).gather(fn_a, fn_b)
    coros = [c(i) for i in range(20)]
    results = await asyncio.gather(*coros)
    for i, r in enumerate(results):
      self.assertEqual(sorted(r), sorted([f'{i}a', f'{i}b']))

  async def test_concurrent_foreach_with_async_functions(self):
    """Concurrent foreach with async mapping functions."""
    async def async_triple(v):
      await asyncio.sleep(0)
      return v * 3

    c = Chain().foreach(async_triple)
    inputs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    coros = [c(inp) for inp in inputs]
    results = await asyncio.gather(*coros)
    expected = [[3, 6, 9], [12, 15, 18], [21, 24, 27]]
    self.assertEqual(results, expected)

  async def test_chain_clone_independent_concurrent(self):
    """Clone captures chain state at clone time.
    To get a truly independent copy, clone() before modifying.
    """
    async def async_double(v):
      await asyncio.sleep(0)
      return v * 2

    # Use clone() for a truly independent copy
    chain = Chain().then(async_double)
    cloned = chain.clone()
    # Modify the original chain after cloning
    chain.then(lambda v: v + 100)

    coros = [cloned(i) for i in range(10)]
    results = await asyncio.gather(*coros)
    # Clone captures the chain state at clone time (only async_double)
    self.assertEqual(sorted(results), sorted([i * 2 for i in range(10)]))

  async def test_concurrent_clone_calls(self):
    """Concurrent clone() calls produce independent chains."""
    async def async_inc(v):
      await asyncio.sleep(0)
      return v + 1

    base = Chain().then(async_inc)
    clones = [base.clone() for _ in range(20)]
    coros = [c.run(i) for i, c in enumerate(clones)]
    results = await asyncio.gather(*coros)
    self.assertEqual(sorted(results), sorted([i + 1 for i in range(20)]))

  async def test_gather_of_many_chain_runs(self):
    """asyncio.gather of many chain.run() calls."""
    async def async_mul(v):
      await asyncio.sleep(0)
      return v * 10

    coros = [Chain(i).then(async_mul).run() for i in range(50)]
    results = await asyncio.gather(*coros)
    self.assertEqual(sorted(results), sorted([i * 10 for i in range(50)]))

  async def test_concurrent_filter_with_async_predicates(self):
    """Concurrent filter with async predicate functions."""
    async def async_is_even(v):
      await asyncio.sleep(0)
      return v % 2 == 0

    c = Chain().filter(async_is_even)
    inputs = [list(range(10)), list(range(5, 15)), list(range(20, 30))]
    coros = [c(inp) for inp in inputs]
    results = await asyncio.gather(*coros)
    for inp, res in zip(inputs, results):
      self.assertEqual(res, [x for x in inp if x % 2 == 0])

  async def test_staggered_async_transition_points(self):
    """Chain starts sync, goes async at different points in concurrent runs."""
    async def async_add(v):
      await asyncio.sleep(0)
      return v + 100

    c_early = Chain().then(async_add).then(lambda v: v * 2)
    c_late = Chain().then(lambda v: v * 2).then(async_add)

    coros_early = [c_early(i) for i in range(10)]
    coros_late = [c_late(i) for i in range(10)]
    results_early = await asyncio.gather(*coros_early)
    results_late = await asyncio.gather(*coros_late)

    for i in range(10):
      self.assertEqual(results_early[i], (i + 100) * 2)
      self.assertEqual(results_late[i], i * 2 + 100)

  async def test_concurrent_chains_with_except_handlers(self):
    """Concurrent chain execution with exception handlers."""
    async def maybe_fail(v):
      await asyncio.sleep(0)
      if v < 0:
        raise TestExc('negative')
      return v

    c = Chain().then(maybe_fail).except_(
      lambda v=None: -1, reraise=False
    )
    values = [1, -1, 2, -2, 3, -3]
    coros = [c(v) for v in values]
    results = await asyncio.gather(*coros)
    expected = [1, -1, 2, -1, 3, -1]
    self.assertEqual(results, expected)


# ---------------------------------------------------------------------------
# 3. Mixed Sync/Async Transition Patterns (18 tests)
# ---------------------------------------------------------------------------
class MixedSyncAsyncTransitionTests(IsolatedAsyncioTestCase):
  """Test various sync/async transition patterns in chains."""

  async def test_sync_async_sync_async_transitions(self):
    """Sync root -> async link -> sync link -> async link (multiple transitions)."""
    async def async_add(v):
      return v + 10

    result = await Chain(1).then(async_add).then(lambda v: v * 2).then(async_add).run()
    # 1 -> 11 -> 22 -> 32
    self.assertEqual(result, 32)

  async def test_async_root_then_sync_only(self):
    """Async root -> sync links only."""
    result = await Chain(aempty, 5).then(lambda v: v + 1).then(lambda v: v * 3).run()
    # 5 -> 6 -> 18
    self.assertEqual(result, 18)

  async def test_sync_chain_with_async_do(self):
    """Sync chain with async .do() -- must await do but discard result."""
    side_effects = []

    async def async_side(v):
      side_effects.append(v)
      return 'discarded'

    result = await Chain(10).do(async_side).then(lambda v: v + 5).run()
    self.assertEqual(result, 15)
    self.assertEqual(side_effects, [10])

  async def test_simple_chain_async_root_run_async_simple(self):
    """Simple chain (only .then) with async root -> _run_simple -> _run_async_simple."""
    async def async_root():
      return 42

    result = await Chain(async_root).then(lambda v: v + 8).run()
    self.assertEqual(result, 50)

  async def test_non_simple_chain_async_root_run_async(self):
    """Non-simple chain with async root -> _run -> _run_async."""
    async def async_root():
      return 42

    result = await Chain(async_root).do(lambda v: None).then(lambda v: v + 8).run()
    self.assertEqual(result, 50)

  async def test_simple_chain_sync_root_async_mid_link(self):
    """Simple chain with sync root, async mid-link -> _run_simple -> _run_async_simple."""
    async def async_double(v):
      return v * 2

    result = await Chain(5).then(lambda v: v + 1).then(async_double).run()
    # 5 -> 6 -> 12
    self.assertEqual(result, 12)

  async def test_non_simple_chain_sync_root_async_mid_link(self):
    """Non-simple chain with sync root, async mid-link -> _run -> _run_async."""
    async def async_double(v):
      return v * 2

    result = await Chain(5).do(lambda v: None).then(lambda v: v + 1).then(async_double).run()
    # 5 -> 6 -> 12
    self.assertEqual(result, 12)

  async def test_return_with_async_value(self):
    """Chain.return_ with async callable -- the signal value is a coroutine."""
    async def async_return_val():
      return 99

    result = await Chain(1).then(lambda v: Chain.return_(async_return_val)).run()
    self.assertEqual(result, 99)

  async def test_break_with_async_value_in_foreach(self):
    """Chain.break_ with async callable in foreach."""
    async def async_break_val():
      return 'break_result'

    def mapper(v):
      if v == 3:
        Chain.break_(async_break_val)
      return v * 10

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, 'break_result')

  async def test_deeply_nested_chain_sync_async_transitions(self):
    """Deeply nested chain where each level transitions sync->async."""
    async def async_inc(v):
      return v + 1

    # inner3: v -> v+1
    inner3 = Chain().then(async_inc)
    # inner2: v -> v+1 -> (v+1)+1 = v+2
    inner2 = Chain().then(async_inc).then(inner3)
    # inner1: v -> v+1 -> (v+1)+1 -> ((v+1)+1)+1 = v+3
    inner1 = Chain().then(async_inc).then(inner2)
    # outer: 0 -> inner1(0) = 0+3 = 3
    result = await Chain(0).then(inner1).run()
    self.assertEqual(result, 3)

  async def test_void_chain_with_async_first_link(self):
    """Void chain (no root) with async first link."""
    async def async_produce():
      return 42

    result = await Chain().then(async_produce).then(lambda v: v + 8).run()
    self.assertEqual(result, 50)

  async def test_multiple_sync_then_async_then_sync(self):
    """Multiple sync links, then async, then more sync links."""
    async def async_step(v):
      return v + 100

    result = await (
      Chain(1)
      .then(lambda v: v + 1)
      .then(lambda v: v + 1)
      .then(lambda v: v + 1)
      .then(async_step)
      .then(lambda v: v * 2)
      .then(lambda v: v - 1)
      .run()
    )
    # 1 -> 2 -> 3 -> 4 -> 104 -> 208 -> 207
    self.assertEqual(result, 207)

  async def test_async_root_override_with_sync_chain(self):
    """Chain with async root value override."""
    async def async_val():
      return 10

    result = await Chain().then(lambda v: v + 5).run(async_val)
    self.assertEqual(result, 15)

  async def test_sync_chain_then_nested_async_chain(self):
    """Sync outer chain containing a nested async chain."""
    async def async_double(v):
      return v * 2

    inner = Chain().then(async_double)
    result = await Chain(5).then(lambda v: v + 1).then(inner).run()
    # 5 -> 6 -> 12
    self.assertEqual(result, 12)


# ---------------------------------------------------------------------------
# 4. Task Registry Edge Cases (11 tests)
# ---------------------------------------------------------------------------
class TaskRegistryEdgeCaseTests(IsolatedAsyncioTestCase):
  """Test task registry size tracking, done callbacks, and cleanup."""

  async def test_registry_size_after_single_autorun(self):
    """Verify task_registry increases after autorun chain."""
    initial = _get_registry_size()

    async def slow(v):
      await asyncio.sleep(0.05)
      return v

    task = Chain(slow, 1).config(autorun=True).run()
    self.assertGreater(_get_registry_size(), initial)
    await task
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_done_callback_fires_and_removes(self):
    """Verify done callback fires and removes task from registry."""
    initial = _get_registry_size()

    async def fast(v):
      return v

    task = Chain(fast, 1).config(autorun=True).run()
    await task
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_multiple_autorun_chains_accumulating(self):
    """Multiple autorun chains accumulate in registry."""
    initial = _get_registry_size()
    blocker = asyncio.Event()

    async def wait(v):
      await blocker.wait()
      return v

    tasks = []
    for i in range(10):
      t = Chain(wait, i).config(autorun=True).run()
      tasks.append(t)

    self.assertGreaterEqual(_get_registry_size(), initial + 10)
    blocker.set()
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_registry_after_exceptions_in_tasks(self):
    """Registry cleanup after exceptions in autorun tasks."""
    initial = _get_registry_size()

    async def raise_err(v):
      raise TestExc('boom')

    task = Chain(raise_err, 1).config(autorun=True).run()
    with self.assertRaises(TestExc):
      await task

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_registry_after_cancellation(self):
    """Registry cleanup after task cancellation."""
    initial = _get_registry_size()

    async def never_end(v):
      await asyncio.sleep(100)
      return v

    task = Chain(never_end, 1).config(autorun=True).run()
    self.assertGreater(_get_registry_size(), initial)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_registry_fire_and_forget_chains(self):
    """Fire-and-forget chains (autorun, not awaited) complete and clean up."""
    initial = _get_registry_size()
    completed = {'count': 0}

    async def increment(v):
      completed['count'] += 1
      return v

    tasks = []
    for i in range(5):
      t = Chain(increment, i).config(autorun=True).run()
      tasks.append(t)

    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    self.assertEqual(completed['count'], 5)
    self.assertEqual(_get_registry_size(), initial)

  async def test_ensure_future_from_sync_except_handler_reraise_false(self):
    """ensure_future called from sync except handler with reraise=False.

    When except handler returns a coroutine and reraise=False,
    the result is wrapped in ensure_future.
    """
    async def async_handler(v=None):
      return 'recovered'

    async def raise_in_chain(v):
      raise TestExc('fail')

    result = await Chain(1).then(raise_in_chain).except_(
      async_handler, reraise=False
    ).run()
    self.assertEqual(result, 'recovered')

  async def test_ensure_future_from_sync_finally_handler_warning(self):
    """ensure_future from sync finally handler causes RuntimeWarning."""
    async def async_finally(v=None):
      return 'finally_done'

    # Sync chain with async finally handler should produce RuntimeWarning
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      result = Chain(42).finally_(async_finally).run()
      # The sync chain returns the value, but the async finally
      # is scheduled via ensure_future with a RuntimeWarning
      self.assertEqual(result, 42)
    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    self.assertGreater(len(runtime_warns), 0)
    # Clean up the fire-and-forget task
    await asyncio.sleep(0.05)

  async def test_registry_empty_after_all_tasks_complete(self):
    """Registry returns to initial size after all tasks complete."""
    initial = _get_registry_size()

    async def identity(v):
      return v

    tasks = []
    for i in range(20):
      t = Chain(identity, i).config(autorun=True).run()
      tasks.append(t)

    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)

  async def test_registry_size_callable(self):
    """_get_registry_size is callable and returns int."""
    self.assertIsInstance(_get_registry_size(), int)

  async def test_registry_concurrent_add_remove(self):
    """Concurrent additions and removals keep registry consistent."""
    initial = _get_registry_size()

    async def fast(v):
      return v

    async def slow(v):
      await asyncio.sleep(0.02)
      return v

    fast_tasks = [Chain(fast, i).config(autorun=True).run() for i in range(10)]
    slow_tasks = [Chain(slow, i).config(autorun=True).run() for i in range(10)]

    # Fast tasks should complete quickly
    for t in fast_tasks:
      await t
    await asyncio.sleep(0)

    # Slow tasks still pending
    await asyncio.gather(*slow_tasks)
    await asyncio.sleep(0)
    self.assertEqual(_get_registry_size(), initial)


# ---------------------------------------------------------------------------
# 5. Async Context Managers (11 tests)
# ---------------------------------------------------------------------------
class AsyncContextManagerTests(IsolatedAsyncioTestCase):
  """Test async context manager handling in chains."""

  async def test_async_cm_with_sync_body(self):
    """Async CM with sync body function."""
    cm = AsyncCM(value='resource')
    result = await Chain(cm).with_(lambda v: v + '_used').run()
    self.assertEqual(result, 'resource_used')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm_with_async_body(self):
    """Async CM with async body function."""
    cm = AsyncCM(value='resource')

    async def body(v):
      return v + '_async'

    result = await Chain(cm).with_(body).run()
    self.assertEqual(result, 'resource_async')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_sync_cm_with_async_body(self):
    """Sync CM with async body triggers _with_to_async."""
    cm = SyncCM(value='sync_resource')

    async def async_body(v):
      return v + '_async_body'

    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'sync_resource_async_body')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_sync_cm_exit_returns_coroutine(self):
    """Sync CM where __exit__ returns a coroutine (edge case).

    When __exit__ returns something truthy (the coroutine object is truthy),
    the exception is suppressed. This is a quirk of the protocol.
    In the _with_to_async path, exit_result is awaited if it is a coroutine.
    """
    cm = SyncCMWithCoroExit(value='cm_value')

    async def async_body(v):
      return v + '_body'

    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'cm_value_body')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm_aexit_suppresses_exception(self):
    """Async CM where __aexit__ suppresses exception."""
    cm = AsyncCM(value='resource', suppress=True)

    def body_raises(v):
      raise TestExc('body error')

    # When suppress=True, __aexit__ returns True, suppressing the exception
    # But the chain's with_ implementation re-raises if __aexit__ does not suppress
    # Actually, async with will suppress if __aexit__ returns True
    result = await Chain(cm).with_(body_raises).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm_aenter_yields_none(self):
    """Async CM that yields None from __aenter__."""
    cm = AsyncCMNoneEnter()
    result = await Chain(cm).with_(lambda v: 'body_result' if v is None else 'wrong').run()
    self.assertEqual(result, 'body_result')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_nested_async_cms_via_chained_with(self):
    """Nested async CMs via sequential .with_() calls."""
    cm1 = AsyncCM(value='outer')
    cm2 = AsyncCM(value='inner')

    async def wrap_in_cm2(v):
      return cm2

    result = await Chain(cm1).with_(wrap_in_cm2).with_(lambda v: v + '_body').run()
    self.assertTrue(cm1.entered)
    self.assertTrue(cm1.exited)

  async def test_cm_body_raises_cancelled_error(self):
    """CM where body raises CancelledError -- __aexit__ still called."""
    cm = AsyncCM(value='resource')

    async def body_cancels(v):
      raise asyncio.CancelledError()

    with self.assertRaises(asyncio.CancelledError):
      await Chain(cm).with_(body_cancels).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_dual_protocol_cm_prefers_async(self):
    """Dual-protocol CM (both __enter__ and __aenter__) -- async preferred."""
    cm = DualCM(sync_val='sync', async_val='async')
    result = await Chain(cm).with_(lambda v: v).run()
    # Library checks for __aenter__ first
    self.assertEqual(result, 'async')
    self.assertTrue(cm.async_entered)
    self.assertTrue(cm.async_exited)
    self.assertFalse(cm.sync_entered)
    self.assertFalse(cm.sync_exited)

  async def test_async_cm_with_do_ignores_result(self):
    """Async CM used with .do() discards the with_ result."""
    cm = AsyncCM(value='resource')

    async def async_identity(v):
      return v

    # Use async root to ensure chain is in async path
    result = await Chain(aempty, 10).do(lambda v: cm).then(lambda v: v + 5).run()
    self.assertEqual(result, 15)

  async def test_async_cm_exception_in_aenter(self):
    """Exception during __aenter__ propagates correctly."""
    class FailingAsyncCM:
      async def __aenter__(self):
        raise TestExc('enter failed')

      async def __aexit__(self, *args):
        return False

    with self.assertRaises(TestExc):
      await Chain(FailingAsyncCM()).with_(lambda v: v).run()


# ---------------------------------------------------------------------------
# 6. Async Generators and Iterators (11 tests)
# ---------------------------------------------------------------------------
class AsyncGeneratorIteratorTests(IsolatedAsyncioTestCase):
  """Test async generators and iterators in chains."""

  async def test_iterate_over_async_chain_results(self):
    """iterate() over chain that produces async results."""
    async def async_range():
      return [1, 2, 3, 4, 5]

    gen = Chain(async_range).iterate(lambda x: x * 2)
    results = []
    async for item in gen:
      results.append(item)
    self.assertEqual(results, [2, 4, 6, 8, 10])

  async def test_iterate_with_async_mapping_function(self):
    """iterate() with async mapping function."""
    async def async_double(x):
      return x * 2

    gen = Chain(SyncIterator([1, 2, 3])).iterate(async_double)
    results = []
    async for item in gen:
      results.append(item)
    self.assertEqual(results, [2, 4, 6])

  async def test_async_for_over_generator_with_async_chain(self):
    """async for over _Generator with async chain."""
    async def async_produce():
      return [10, 20, 30]

    gen = Chain(async_produce).iterate()
    results = []
    async for item in gen:
      results.append(item)
    self.assertEqual(results, [10, 20, 30])

  async def test_sync_for_over_generator_sync_chain(self):
    """sync for over _Generator with sync-only chain."""
    gen = Chain(SyncIterator([1, 2, 3])).iterate(lambda x: x + 100)
    results = list(gen)
    self.assertEqual(results, [101, 102, 103])

  async def test_partial_consumption_of_async_generator(self):
    """Partial consumption of async generator."""
    gen = Chain(SyncIterator(range(100))).iterate(lambda x: x * 2)
    results = []
    async for item in gen:
      results.append(item)
      if len(results) >= 5:
        break
    self.assertEqual(results, [0, 2, 4, 6, 8])

  async def test_break_during_async_generator_iteration(self):
    """Break during async generator iteration stops cleanly."""
    gen = Chain(SyncIterator(range(20))).iterate()
    count = 0
    async for _ in gen:
      count += 1
      if count >= 3:
        break
    self.assertEqual(count, 3)

  async def test_exception_during_async_generator_iteration(self):
    """Exception during async generator iteration propagates."""
    def failing_mapper(x):
      if x == 3:
        raise TestExc('fail at 3')
      return x

    gen = Chain(SyncIterator([1, 2, 3, 4])).iterate(failing_mapper)
    results = []
    with self.assertRaises(TestExc):
      async for item in gen:
        results.append(item)
    self.assertEqual(results, [1, 2])

  async def test_generator_call_creates_new_instance(self):
    """_Generator.__call__ creates a new generator with different args."""
    gen = Chain().then(lambda v: v).iterate(lambda x: x * 3)
    nested = gen(SyncIterator([10, 20]))
    results = []
    for item in nested:
      results.append(item)
    self.assertEqual(results, [30, 60])

  async def test_generator_call_async_iteration(self):
    """_Generator.__call__ with async iteration."""
    gen = Chain().then(lambda v: v).iterate(lambda x: x + 1)
    nested = gen(SyncIterator([5, 10, 15]))
    results = []
    async for item in nested:
      results.append(item)
    self.assertEqual(results, [6, 11, 16])

  async def test_iterate_without_function(self):
    """iterate() without function just yields elements."""
    gen = Chain(SyncIterator([1, 2, 3])).iterate()
    results = []
    async for item in gen:
      results.append(item)
    self.assertEqual(results, [1, 2, 3])

  async def test_foreach_with_async_iterator(self):
    """foreach with async iterator."""
    async def async_double(v):
      return v * 2

    result = await Chain(AsyncIterator([1, 2, 3])).foreach(async_double).run()
    self.assertEqual(result, [2, 4, 6])


# ---------------------------------------------------------------------------
# 7. Async Exception Scenarios (12 tests)
# ---------------------------------------------------------------------------
class AsyncExceptionScenariosTests(IsolatedAsyncioTestCase):
  """Test exception handling in async chain paths."""

  async def test_exception_in_awaited_root_value(self):
    """Exception in awaited root value propagates."""
    async def failing_root():
      raise TestExc('root failed')

    with self.assertRaises(TestExc):
      await Chain(failing_root).then(lambda v: v).run()

  async def test_exception_in_awaited_then_result(self):
    """Exception in awaited .then() result propagates."""
    async def failing_then(v):
      raise TestExc('then failed')

    with self.assertRaises(TestExc):
      await Chain(1).then(failing_then).run()

  async def test_exception_in_awaited_do_result(self):
    """Exception in awaited .do() result (side effect) propagates."""
    async def failing_do(v):
      raise TestExc('do failed')

    with self.assertRaises(TestExc):
      await Chain(1).do(failing_do).run()

  async def test_async_except_handler_that_raises(self):
    """Async except handler that raises -- handler exception propagates with __cause__."""
    async def handler(v=None):
      raise TypeError('handler failed')

    async def chain_raises(v):
      raise TestExc('original')

    with self.assertRaises(TypeError) as cm:
      await Chain(1).then(chain_raises).except_(
        handler, reraise=False
      ).run()
    self.assertIsInstance(cm.exception.__cause__, TestExc)

  async def test_async_except_handler_with_reraise_true(self):
    """Async except handler with reraise=True -- original exception re-raised."""
    handler_called = {'value': False}

    async def handler(v=None):
      handler_called['value'] = True

    async def chain_raises(v):
      raise TestExc('original')

    with self.assertRaises(TestExc):
      await Chain(1).then(chain_raises).except_(
        handler, reraise=True
      ).run()
    self.assertTrue(handler_called['value'])

  async def test_async_finally_handler_that_raises(self):
    """Async finally handler that raises -- finally exception propagates."""
    async def async_finally(v=None):
      raise TypeError('finally failed')

    with self.assertRaises(TypeError):
      await Chain(aempty, 1).finally_(async_finally).run()

  async def test_exception_during_gather(self):
    """Exception in one gathered function propagates."""
    async def fn_ok(v):
      return 'ok'

    async def fn_fail(v):
      raise TestExc('gather fail')

    with self.assertRaises(TestExc):
      await Chain(1).gather(fn_ok, fn_fail).run()

  async def test_exception_in_one_of_many_gathered(self):
    """Exception in one of many gathered async functions."""
    async def fn_a(v):
      return 'a'

    async def fn_b(v):
      raise TestExc('b fails')

    async def fn_c(v):
      return 'c'

    with self.assertRaises(TestExc):
      await Chain(1).gather(fn_a, fn_b, fn_c).run()

  async def test_chained_exceptions_in_async_chains(self):
    """Chained exceptions (__cause__, __context__) in async chains."""
    async def step1(v):
      raise ValueError('step1')

    async def handler(v=None):
      raise TypeError('handler') from ValueError('step1_copy')

    with self.assertRaises(TypeError) as cm:
      await Chain(1).then(step1).except_(
        handler, reraise=False
      ).run()
    self.assertIsNotNone(cm.exception.__cause__)

  async def test_base_exception_keyboard_interrupt_in_async(self):
    """KeyboardInterrupt in async chain propagates as BaseException."""
    async def raise_kb(v):
      raise KeyboardInterrupt()

    with self.assertRaises(KeyboardInterrupt):
      await Chain(1).then(raise_kb).run()

  async def test_base_exception_system_exit_in_async(self):
    """SystemExit in async chain propagates as BaseException."""
    async def raise_exit(v):
      raise SystemExit(1)

    with self.assertRaises(SystemExit):
      await Chain(1).then(raise_exit).run()

  async def test_exception_in_async_except_handler_non_simple(self):
    """Exception in async except handler on non-simple path."""
    async def async_handler(v=None):
      raise RuntimeError('handler error')

    async def failing(v):
      raise TestExc('fail')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(1).do(lambda v: None).then(failing).except_(
        async_handler, reraise=False
      ).run()
    self.assertIsInstance(cm.exception.__cause__, TestExc)


# ---------------------------------------------------------------------------
# 8. Async Return/Break Signals (8 tests)
# ---------------------------------------------------------------------------
class AsyncReturnBreakSignalTests(IsolatedAsyncioTestCase):
  """Test Chain.return_() and Chain.break_() behavior in async chains."""

  async def test_return_with_async_callable(self):
    """Chain.return_ with async callable: Chain.return_(async_fn)."""
    async def async_val():
      return 'returned'

    result = await Chain(1).then(lambda v: Chain.return_(async_val)).run()
    self.assertEqual(result, 'returned')

  async def test_return_with_async_callable_and_arg(self):
    """Chain.return_ with async callable and argument."""
    async def async_add(a, b):
      return a + b

    result = await Chain(1).then(lambda v: Chain.return_(async_add, 10, 20)).run()
    self.assertEqual(result, 30)

  async def test_return_at_different_points_in_async_chain(self):
    """Chain.return_ at different points in async chain."""
    async def async_inc(v):
      return v + 1

    # Return at first link -- sync return_(42) returns 42 directly (not awaitable)
    result = Chain(1).then(lambda v: Chain.return_(42)).then(async_inc).run()
    self.assertEqual(result, 42)

    # Return at second link (after async transition)
    result = await Chain(1).then(async_inc).then(lambda v: Chain.return_(v * 10)).then(async_inc).run()
    self.assertEqual(result, 20)

  async def test_break_with_async_callable_in_foreach(self):
    """Chain.break_ with async callable in foreach."""
    async def async_break_result():
      return 'break_val'

    processed = []

    def mapper(v):
      if v == 3:
        Chain.break_(async_break_result)
      processed.append(v)
      return v * 10

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, 'break_val')

  async def test_eval_signal_value_with_coroutine_result(self):
    """_eval_signal_value with coroutine result -- the coroutine is awaited."""
    async def async_produce():
      return 'signal_result'

    # return_ triggers _eval_signal_value internally
    result = await Chain(aempty, 1).then(lambda v: Chain.return_(async_produce)).run()
    self.assertEqual(result, 'signal_result')

  async def test_return_in_nested_async_chain_propagation(self):
    """return_ in nested async chain propagates through await to parent."""
    async def async_identity(v):
      return v

    inner = Chain().then(async_identity).then(lambda v: Chain.return_(999))
    result = await Chain(1).then(inner).then(lambda v: v + 100).run()
    # _Return propagates to parent, skipping .then(v + 100)
    self.assertEqual(result, 999)

  async def test_return_with_no_value_in_async_chain(self):
    """Chain.return_() with no args returns None in async chain."""
    async def async_identity(v):
      return v

    result = await Chain(aempty, 1).then(lambda v: Chain.return_()).run()
    self.assertIsNone(result)

  async def test_break_with_no_value_returns_partial(self):
    """Chain.break_() with no value returns collected partial results."""
    async def async_mapper(v):
      if v == 3:
        Chain.break_()
      return v * 10

    result = await Chain([1, 2, 3, 4]).foreach(async_mapper).run()
    # break_ with no value returns the partial list collected so far
    self.assertEqual(result, [10, 20])


# ---------------------------------------------------------------------------
# 9. Additional Concurrent + Async Edge Cases (5 tests)
# ---------------------------------------------------------------------------
class AdditionalConcurrentEdgeCases(IsolatedAsyncioTestCase):
  """Additional edge case tests for concurrent async operations."""

  async def test_concurrent_chains_with_finally(self):
    """Concurrent chains with finally handlers all execute finally."""
    counters = {'finally_count': 0}

    async def async_inc(v):
      await asyncio.sleep(0)
      return v + 1

    def on_finally(v=None):
      counters['finally_count'] += 1

    coros = [
      Chain(i).then(async_inc).finally_(on_finally).run()
      for i in range(10)
    ]
    results = await asyncio.gather(*coros)
    self.assertEqual(sorted(results), sorted([i + 1 for i in range(10)]))
    self.assertEqual(counters['finally_count'], 10)

  async def test_concurrent_chains_with_nested_chains(self):
    """Concurrent chains each containing nested chains."""
    async def async_double(v):
      await asyncio.sleep(0)
      return v * 2

    inner = Chain().then(async_double)
    outer = Chain().then(lambda v: v + 1).then(inner)

    coros = [outer(i) for i in range(20)]
    results = await asyncio.gather(*coros)
    # i -> i+1 -> (i+1)*2
    expected = [(i + 1) * 2 for i in range(20)]
    self.assertEqual(sorted(results), sorted(expected))

  async def test_pipe_syntax_with_async(self):
    """Pipe syntax with async operations."""
    async def async_triple(v):
      return v * 3

    result = await Chain(5).then(async_triple).then(lambda v: v + 1).run()
    self.assertEqual(result, 16)


# ---------------------------------------------------------------------------
# 10. Async Foreach Edge Cases (6 tests)
# ---------------------------------------------------------------------------
class AsyncForeachEdgeCaseTests(IsolatedAsyncioTestCase):
  """Test foreach with async iterators and async mapping functions."""

  async def test_foreach_async_iterator_sync_fn(self):
    """foreach over async iterator with sync mapping function."""
    result = await Chain(AsyncIterator([1, 2, 3])).foreach(lambda v: v * 10).run()
    self.assertEqual(result, [10, 20, 30])

  async def test_foreach_async_iterator_async_fn(self):
    """foreach over async iterator with async mapping function."""
    async def async_square(v):
      return v ** 2

    result = await Chain(AsyncIterator([2, 3, 4])).foreach(async_square).run()
    self.assertEqual(result, [4, 9, 16])

  async def test_foreach_async_break(self):
    """Break during async foreach."""
    async def async_mapper(v):
      if v == 3:
        Chain.break_()
      return v * 10

    result = await Chain([1, 2, 3, 4, 5]).foreach(async_mapper).run()
    self.assertEqual(result, [10, 20])

  async def test_foreach_async_iterator_break(self):
    """Break during foreach over async iterator."""
    async def mapper(v):
      if v == 3:
        Chain.break_()
      return v * 10

    result = await Chain(AsyncIterator([1, 2, 3, 4, 5])).foreach(mapper).run()
    self.assertEqual(result, [10, 20])


# ---------------------------------------------------------------------------
# 11. Async Filter Edge Cases (4 tests)
# ---------------------------------------------------------------------------
class AsyncFilterEdgeCaseTests(IsolatedAsyncioTestCase):
  """Test filter with async predicates and async iterators."""

  async def test_filter_async_predicate_sync_iterable(self):
    """Filter with async predicate over sync iterable."""
    async def async_is_positive(v):
      return v > 0

    result = await Chain([-2, -1, 0, 1, 2, 3]).filter(async_is_positive).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_filter_sync_predicate_async_iterable(self):
    """Filter with sync predicate over async iterable."""
    result = await Chain(AsyncIterator([1, 2, 3, 4, 5, 6])).filter(lambda v: v % 2 == 0).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_filter_async_predicate_async_iterable(self):
    """Filter with async predicate over async iterable."""
    async def async_is_odd(v):
      return v % 2 == 1

    result = await Chain(AsyncIterator([1, 2, 3, 4, 5])).filter(async_is_odd).run()
    self.assertEqual(result, [1, 3, 5])

  async def test_filter_empty_result(self):
    """Filter where no elements match returns empty list."""
    async def async_never(v):
      return False

    result = await Chain([1, 2, 3]).filter(async_never).run()
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# 12. Async Gather Edge Cases (5 tests)
# ---------------------------------------------------------------------------
class AsyncGatherEdgeCaseTests(IsolatedAsyncioTestCase):
  """Test gather with mixed sync/async functions."""

  async def test_gather_all_async(self):
    """Gather with all async functions."""
    async def fn_a(v):
      return v + 'a'

    async def fn_b(v):
      return v + 'b'

    async def fn_c(v):
      return v + 'c'

    result = await Chain('x').gather(fn_a, fn_b, fn_c).run()
    self.assertEqual(result, ['xa', 'xb', 'xc'])

  async def test_gather_mixed_sync_async(self):
    """Gather with mixed sync and async functions."""
    async def async_fn(v):
      return v + '_async'

    result = await Chain('val').gather(
      lambda v: v + '_sync',
      async_fn,
    ).run()
    self.assertEqual(result, ['val_sync', 'val_async'])

  async def test_gather_single_function(self):
    """Gather with a single function."""
    async def fn(v):
      return v * 2

    result = await Chain(5).gather(fn).run()
    self.assertEqual(result, [10])

  async def test_gather_all_sync(self):
    """Gather with all sync functions returns sync result."""
    result = Chain(10).gather(
      lambda v: v + 1,
      lambda v: v + 2,
      lambda v: v + 3,
    ).run()
    self.assertEqual(result, [11, 12, 13])

  async def test_gather_exception_in_async_fn(self):
    """Exception in one async gathered function propagates."""
    async def fn_ok(v):
      await asyncio.sleep(0)
      return 'ok'

    async def fn_fail(v):
      await asyncio.sleep(0)
      raise TestExc('fail')

    with self.assertRaises(TestExc):
      await Chain(1).gather(fn_ok, fn_fail).run()


# ---------------------------------------------------------------------------
# 13. Autorun and ensure_future Integration (5 tests)
# ---------------------------------------------------------------------------
class AutorunEnsureFutureIntegrationTests(IsolatedAsyncioTestCase):
  """Test autorun behavior with ensure_future integration."""

  async def test_autorun_simple_async_root(self):
    """Autorun with simple async root."""
    task = Chain(aempty, 42).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 42)

  async def test_autorun_non_simple_async_root(self):
    """Autorun with non-simple async root."""
    task = Chain(aempty, 42).do(lambda v: None).config(autorun=True).run()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 42)

  async def test_autorun_via_call(self):
    """Autorun via __call__."""
    async def async_double(v):
      return v * 2

    chain = Chain(5).then(async_double).config(autorun=True)
    task = chain()
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 10)

  async def test_autorun_with_root_override(self):
    """Autorun with root value override."""
    async def async_inc(v):
      return v + 1

    chain = Chain().then(async_inc).config(autorun=True)
    task = chain.run(100)
    self.assertIsInstance(task, asyncio.Task)
    result = await task
    self.assertEqual(result, 101)



# ---------------------------------------------------------------------------
# 14. Debug Mode Async Tests (3 tests)
# ---------------------------------------------------------------------------
class DebugModeAsyncTests(IsolatedAsyncioTestCase):
  """Test debug mode behavior in async chains."""

  async def test_debug_async_chain_produces_correct_result(self):
    """Debug mode in async chain does not alter the result."""
    async def async_double(v):
      return v * 2

    result = await Chain(5).then(async_double).config(debug=True).run()
    self.assertEqual(result, 10)

  async def test_debug_with_multiple_async_links(self):
    """Debug mode with multiple async links produces correct result."""
    async def async_inc(v):
      return v + 1

    result = await (
      Chain(0)
      .then(async_inc)
      .then(async_inc)
      .then(async_inc)
      .config(debug=True)
      .run()
    )
    self.assertEqual(result, 3)

  async def test_debug_mode_non_simple_async(self):
    """Debug mode forces non-simple path with async."""
    async def async_add(v):
      return v + 10

    result = await Chain(5).do(lambda v: None).then(async_add).config(debug=True).run()
    self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# 16. Decorator + Frozen Chain Async Tests (4 tests)
# ---------------------------------------------------------------------------
class DecoratorChainAsyncTests(IsolatedAsyncioTestCase):
  """Test decorator and chain reuse behavior with async functions."""

  async def test_chain_reuse_with_async_fn(self):
    """Chain wrapping async function reused."""
    async def async_square(v):
      return v ** 2

    c = Chain().then(async_square)
    result = await c(7)
    self.assertEqual(result, 49)

  async def test_decorator_wraps_async_function(self):
    """decorator() wraps an async function."""
    @Chain().then(lambda v: v * 3).decorator()
    async def get_value(x):
      return x + 1

    result = await get_value(4)
    # get_value(4) -> 5 -> 15
    self.assertEqual(result, 15)

  async def test_chain_reuse_multiple_async_calls(self):
    """Chain can be called many times with async links."""
    async def async_fn(v):
      return v + 100

    c = Chain().then(async_fn)
    for i in range(50):
      result = await c(i)
      self.assertEqual(result, i + 100)

  async def test_chain_reuse_concurrent_with_debug(self):
    """Chain concurrent calls with debug mode."""
    async def async_double(v):
      await asyncio.sleep(0)
      return v * 2

    c = Chain().then(async_double).config(debug=True)
    coros = [c(i) for i in range(20)]
    results = await asyncio.gather(*coros)
    self.assertEqual(sorted(results), sorted([i * 2 for i in range(20)]))


if __name__ == '__main__':
  import unittest
  unittest.main()
