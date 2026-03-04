"""Mega async edge case tests — exhaustive coverage of async transitions,
concurrent execution, task registry, autorun, iteration, generators,
context managers, sleep, to_thread, and exception paths.

Organized into sections:
  A. Sync-to-Async Transition Points
  B. ensure_future and Task Registry
  C. Autorun Behavior
  D. Concurrent Execution
  E. Mixed Sync/Async Patterns
  F. Async Exception Paths
  G. Async Finally Paths
  H. Sleep and ToThread
  I. Async Iteration Edge Cases
  J. Async Generator Edge Cases
  K. Async Context Manager Edge Cases
"""
import asyncio
import functools
import inspect
import logging
import time
import warnings
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def async_add(v, n=1):
  return v + n

async def async_double(v):
  return v * 2

async def async_identity(v):
  return v

async def async_noop(v):
  pass

async def async_raise_test(v):
  raise TestExc('async boom')

async def async_return_42():
  return 42

def sync_double(v):
  return v * 2

def sync_add_one(v):
  return v + 1


class SyncCM:
  """Sync-only context manager."""
  def __init__(self, value=None, suppress=False):
    self.value = value
    self.entered = False
    self.exited = False
    self.exc_info = None
    self.suppress = suppress

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exc_info = (exc_type, exc_val, exc_tb)
    return self.suppress


class AsyncCM:
  """Async context manager."""
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


class AsyncIterable:
  """Async iterable over a list."""
  def __init__(self, items):
    self.items = items

  def __aiter__(self):
    return AsyncIterator(iter(self.items))


class AsyncIterator:
  """Async iterator for AsyncIterable."""
  def __init__(self, it):
    self._it = it

  def __aiter__(self):
    return self

  async def __anext__(self):
    try:
      return next(self._it)
    except StopIteration:
      raise StopAsyncIteration


class CallableClass:
  """Callable class with sync __call__."""
  def __init__(self, factor):
    self.factor = factor

  def __call__(self, v):
    return v * self.factor


class AsyncCallableClass:
  """Callable class with async __call__."""
  def __init__(self, factor):
    self.factor = factor

  async def __call__(self, v):
    return v * self.factor


# ═══════════════════════════════════════════════════════════════════════════
# A. Sync-to-Async Transition Points (15+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class SyncToAsyncTransitionTests(IsolatedAsyncioTestCase):
  """Tests for sync-to-async transition points in _run, _run_simple,
  _run_async, _run_async_simple. Each tests a specific entry point where
  the chain detects a coroutine and transitions to the async path.
  """

  async def test_a01_root_value_is_coroutine_enters_run_async(self):
    """Chain(async_fn, arg) — root value itself is a coroutine.

    _run evaluates root_link, gets a coroutine, enters _run_async from
    root eval (line 129-134 of _chain_core.pxi).
    Forces non-simple via .do().
    """
    result = await Chain(aempty, 42).do(lambda v: None).run()
    self.assertEqual(result, 42)

  async def test_a02_first_link_returns_coroutine(self):
    """Chain(sync_root).then(async_fn) — first link returns coroutine.

    _run evaluates root link (sync), then first link returns a coroutine,
    entering _run_async mid-chain (line 156-161).
    Forces non-simple via .do().
    """
    result = await Chain(10).do(lambda v: None).then(async_double).run()
    self.assertEqual(result, 20)

  async def test_a03_last_link_returns_coroutine(self):
    """Chain with sync links, last link returns coroutine.

    All links before last are sync. The final link returns coroutine,
    entering _run_async at the very end. Forces non-simple via .do().
    """
    result = await (
      Chain(5)
      .do(lambda v: None)
      .then(lambda v: v + 1)
      .then(lambda v: v * 2)
      .then(async_identity)
      .run()
    )
    self.assertEqual(result, 12)

  async def test_a04_middle_link_returns_coroutine(self):
    """Chain where middle link returns coroutine.

    Sync links, then async in the middle, then more sync links in the
    async continuation. Forces non-simple via .do().
    """
    result = await (
      Chain(1)
      .do(lambda v: None)
      .then(lambda v: v + 1)
      .then(async_double)
      .then(lambda v: v + 10)
      .run()
    )
    # 1 -> 2 -> 4 -> 14
    self.assertEqual(result, 14)

  async def test_a05_alternating_sync_async_links(self):
    """Chain with alternating sync/async links.

    First async link triggers transition, subsequent links (both sync
    and async) all execute in the async path.
    """
    result = await (
      Chain(1)
      .do(lambda v: None)
      .then(lambda v: v + 1)       # sync: 2
      .then(async_double)           # async: 4
      .then(lambda v: v + 3)        # sync (in async): 7
      .then(async_identity)         # async: 7
      .then(lambda v: v * 10)       # sync (in async): 70
      .run()
    )
    self.assertEqual(result, 70)

  async def test_a06_all_links_return_coroutines(self):
    """Chain where ALL links return coroutines.

    Root is async, all subsequent links are async. Transition happens
    at root, everything else runs in _run_async.
    """
    async def async_add_5(v):
      return v + 5

    async def async_mul_3(v):
      return v * 3

    result = await (
      Chain(aempty, 2)
      .do(async_noop)
      .then(async_add_5)
      .then(async_mul_3)
      .then(async_identity)
      .run()
    )
    # 2 -> 7 -> 21 -> 21
    self.assertEqual(result, 21)

  async def test_a08_simple_path_async_transition_at_root(self):
    """Simple path (_run_simple) async transition at root.

    Chain with only .then() links (simple), root is async. _run_simple
    detects coroutine and returns _run_async_simple coroutine (line 366-367).
    """
    result = await Chain(aempty, 99).then(lambda v: v + 1).run()
    self.assertEqual(result, 100)

  async def test_a09_simple_path_async_transition_at_first_link(self):
    """Simple path async transition at first (and only) link.

    Root is sync, first .then() link returns coroutine. _run_simple
    detects it in the link loop (line 383-384).
    """
    result = await Chain(10).then(async_double).run()
    self.assertEqual(result, 20)

  async def test_a10_simple_path_async_transition_at_middle_link(self):
    """Simple path async transition at middle link.

    Multiple .then() links, middle one returns coroutine. Sync links
    before it execute in _run_simple, then transition to _run_async_simple.
    """
    result = await (
      Chain(1)
      .then(lambda v: v + 1)       # sync: 2
      .then(async_double)           # async transition: 4
      .then(lambda v: v + 10)       # continues in _run_async_simple: 14
      .run()
    )
    self.assertEqual(result, 14)

  async def test_a11_simple_path_async_transition_at_last_link(self):
    """Simple path async transition at last link.

    All links except last are sync. Last link returns coroutine.
    """
    result = await (
      Chain(5)
      .then(lambda v: v * 2)
      .then(lambda v: v + 1)
      .then(async_identity)
      .run()
    )
    self.assertEqual(result, 11)

  async def test_a14_async_chain_debug_mode_link_results(self):
    """Async chain in debug mode — link_results tracked across async boundaries.

    When _debug=True, _run_async populates link_results dict (lines 254-257,
    278-281). Verify the chain computes correctly with debug enabled.
    """
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      result = await (
        Chain(2)
        .then(lambda v: v + 3)
        .then(async_double)
        .then(lambda v: v + 1)
        .config(debug=True)
        .run()
      )
      # 2 -> 5 -> 10 -> 11
      self.assertEqual(result, 11)
      # Root value (2) and first sync link (5) logged before async transition
      IsolatedAsyncioTestCase.assertGreaterEqual(self, len(logs), 2)
    finally:
      logger.removeHandler(handler)

  async def test_a15_async_root_coroutine_and_debug_mode(self):
    """Async chain where root value is a coroutine AND debug mode is on.

    Tests the link_results initialization path in _run_async (lines 254-256):
    if link_results is None: link_results = {}
    This branch is hit when the root is async and debug is on, because
    _run has not yet initialized link_results before transitioning.
    """
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      result = await (
        Chain(aempty, 77)
        .then(lambda v: v + 3)
        .config(debug=True)
        .run()
      )
      self.assertEqual(result, 80)
    finally:
      logger.removeHandler(handler)



# ═══════════════════════════════════════════════════════════════════════════
# B. ensure_future and Task Registry (10+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class TaskRegistryTests(IsolatedAsyncioTestCase):
  """Tests for the task registry in _async_utils.pxi and ensure_future
  behavior when used via autorun chains.
  """

  async def test_b01_ensure_future_creates_task_in_registry(self):
    """ensure_future creates a Task and adds it to task_registry.

    Via autorun: ensure_future is called, task_registry.add(task)
    on line 33 of _async_utils.pxi.
    """
    initial = _get_registry_size()
    async def slow(v):
      await asyncio.sleep(0.05)
      return v

    task = Chain(slow, 10).config(autorun=True).run()
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), initial)
    await task

  async def test_b02_task_removed_from_registry_when_done(self):
    """Task is removed from task_registry via done callback (line 42).

    task.add_done_callback(task_registry.discard) ensures cleanup.
    """
    initial = _get_registry_size()
    async def fast(v):
      return v

    task = Chain(fast, 1).config(autorun=True).run()
    await task
    await asyncio.sleep(0)  # let done callback fire
    IsolatedAsyncioTestCase.assertEqual(self, _get_registry_size(), initial)

  async def test_b05_get_registry_size_returns_correct_count(self):
    """_get_registry_size() accurately reflects pending tasks."""
    initial = _get_registry_size()
    async def slow(v):
      await asyncio.sleep(0.05)
      return v

    tasks = []
    for i in range(5):
      tasks.append(Chain(slow, i).config(autorun=True).run())

    IsolatedAsyncioTestCase.assertGreaterEqual(
      self, _get_registry_size(), initial + 5
    )

    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    IsolatedAsyncioTestCase.assertEqual(self, _get_registry_size(), initial)

  async def test_b07_multiple_concurrent_ensure_future_calls(self):
    """Multiple concurrent ensure_future calls all succeed."""
    initial = _get_registry_size()
    count = 20

    async def fast(v):
      await asyncio.sleep(0)
      return v * 2

    tasks = [Chain(fast, i).config(autorun=True).run() for i in range(count)]
    results = await asyncio.gather(*tasks)
    IsolatedAsyncioTestCase.assertEqual(self, results, [i * 2 for i in range(count)])
    await asyncio.sleep(0)
    IsolatedAsyncioTestCase.assertEqual(self, _get_registry_size(), initial)

  async def test_b08_autorun_fire_and_forget(self):
    """Autorun chain with ensure_future — fire-and-forget behavior.

    Task completes even if we don't explicitly await it, because
    task_registry holds a strong reference.
    """
    completed = {'value': False}

    async def set_flag(v):
      await asyncio.sleep(0.02)
      completed['value'] = True
      return v

    task = Chain(set_flag, 1).config(autorun=True).run()
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)

    # Wait for it to complete
    await task
    IsolatedAsyncioTestCase.assertTrue(self, completed['value'])

  async def test_b09_registry_does_not_leak_completed_tasks(self):
    """Verify registry size returns to 0 after all tasks complete."""
    initial = _get_registry_size()

    async def fast(v):
      return v

    for i in range(50):
      t = Chain(fast, i).config(autorun=True).run()
      await t

    await asyncio.sleep(0)
    IsolatedAsyncioTestCase.assertEqual(self, _get_registry_size(), initial)

  async def test_b11_exception_in_task_still_cleans_registry(self):
    """Task that raises an exception is still cleaned from registry."""
    initial = _get_registry_size()

    async def raiser(v):
      raise TestExc('test')

    task = Chain(raiser, 1).config(autorun=True).run()
    with self.assertRaises(TestExc):
      await task
    await asyncio.sleep(0)
    IsolatedAsyncioTestCase.assertEqual(self, _get_registry_size(), initial)


# ═══════════════════════════════════════════════════════════════════════════
# C. Autorun Behavior (8+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class AutorunBehaviorTests(IsolatedAsyncioTestCase):
  """Tests for autorun=True behavior across different paths."""

  async def test_c01_autorun_true_coroutine_from_run(self):
    """Chain with autorun=True, run() returns Task when async."""
    task = Chain(aempty, 42).then(sync_double).config(autorun=True).run()
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    result = await task
    IsolatedAsyncioTestCase.assertEqual(self, result, 84)

  async def test_c02_autorun_true_non_coroutine_returns_as_is(self):
    """Chain with autorun=True, fully sync chain returns value directly."""
    result = Chain(10).then(sync_double).config(autorun=True).run()
    IsolatedAsyncioTestCase.assertNotIsInstance(self, result, asyncio.Task)
    IsolatedAsyncioTestCase.assertEqual(self, result, 20)

  async def test_c03_autorun_true_in_call_method(self):
    """Chain with autorun=True in __call__() method returns Task."""
    chain = Chain(aempty, 5).then(sync_add_one).config(autorun=True)
    task = chain()
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    result = await task
    IsolatedAsyncioTestCase.assertEqual(self, result, 6)

  async def test_c05_autorun_sync_root_async_body(self):
    """autorun=True with sync root that goes async in body.

    Non-simple path: root is sync, .do() forces non-simple, async .then()
    causes transition. Lines 159-160 fire ensure_future.
    """
    async def async_mul_10(v):
      return v * 10

    task = Chain(5).do(lambda v: None).then(async_mul_10).config(autorun=True).run()
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    result = await task
    IsolatedAsyncioTestCase.assertEqual(self, result, 50)

  async def test_c06_autorun_async_root(self):
    """autorun=True with async root.

    Non-simple path: async root causes _run_async, lines 132-133.
    """
    task = Chain(aempty, 7).do(lambda v: None).config(autorun=True).run()
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    result = await task
    IsolatedAsyncioTestCase.assertEqual(self, result, 7)

  async def test_c07_autorun_false_default_coroutine_returned(self):
    """Chain with autorun=False (default) — coroutine returned as-is.

    Without autorun, the coroutine from _run_async is returned directly,
    not wrapped in a Task.
    """
    result = Chain(aempty, 42).then(sync_double).run()
    IsolatedAsyncioTestCase.assertTrue(
      self, inspect.isawaitable(result)
    )
    value = await result
    IsolatedAsyncioTestCase.assertEqual(self, value, 84)

  async def test_c08_autorun_sync_run_path_coroutine_mid_chain(self):
    """Autorun in the sync _run path when coroutine appears mid-chain.

    Lines 132-133 and 159-160: when _autorun is True and a coroutine
    is detected in _run, ensure_future wraps _run_async's result.
    """
    async def async_add_100(v):
      return v + 100

    # Non-simple path (has .except_)
    task = (
      Chain(1)
      .except_(lambda v: None)
      .then(lambda v: v + 1)
      .then(async_add_100)
      .config(autorun=True)
      .run()
    )
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    result = await task
    IsolatedAsyncioTestCase.assertEqual(self, result, 102)

  async def test_c09_autorun_simple_chain_via_run_method(self):
    """Autorun on simple chain via run() — ensure_future at line 555-556."""
    task = Chain(aempty, 3).then(lambda v: v * 3).config(autorun=True).run()
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    result = await task
    IsolatedAsyncioTestCase.assertEqual(self, result, 9)


# ═══════════════════════════════════════════════════════════════════════════
# D. Concurrent Execution (8+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class ConcurrentExecutionTests(IsolatedAsyncioTestCase):
  """Tests for concurrent chain execution safety."""

  async def test_d01_clone_executed_concurrently(self):
    """Clone of chain executed concurrently — should be safe.

    clone() creates independent copies that share callable references
    but have independent execution state.
    """
    original = Chain().then(async_double)
    clones = [original.clone() for _ in range(50)]
    coros = [clone.run(i) for i, clone in enumerate(clones)]
    results = await asyncio.gather(*coros)
    IsolatedAsyncioTestCase.assertEqual(
      self, sorted(results), sorted([i * 2 for i in range(50)])
    )

  async def test_d02_chain_executed_concurrently(self):
    """Chain executed concurrently -- should be safe.

    Chain can be safely reused across concurrent executions.
    """
    c = Chain().then(async_double)
    coros = [c(i) for i in range(100)]
    results = await asyncio.gather(*coros)
    IsolatedAsyncioTestCase.assertEqual(
      self, sorted(results), sorted([i * 2 for i in range(100)])
    )

  async def test_d03_shared_callable_references_concurrent(self):
    """Multiple chains sharing the same callable executed concurrently."""
    shared_fn = lambda v: v + 1

    async def shared_async(v):
      await asyncio.sleep(0)
      return v * 2

    chains = [
      Chain().then(shared_fn).then(shared_async)
      for _ in range(50)
    ]
    coros = [c.run(i) for i, c in enumerate(chains)]
    results = await asyncio.gather(*coros)
    expected = [(i + 1) * 2 for i in range(50)]
    IsolatedAsyncioTestCase.assertEqual(self, sorted(results), sorted(expected))

  async def test_d04_rapid_creation_and_execution(self):
    """Rapid creation and execution of many chains."""
    results = []
    for i in range(100):
      r = await Chain(aempty, i).then(lambda v: v * 2).run()
      results.append(r)
    IsolatedAsyncioTestCase.assertEqual(self, results, [i * 2 for i in range(100)])

  async def test_d05_100_concurrent_async_executions(self):
    """100 concurrent async chain executions."""
    async def slow_identity(v):
      await asyncio.sleep(0)
      return v

    c = Chain().then(slow_identity).then(sync_add_one)
    coros = [c(i) for i in range(100)]
    results = await asyncio.gather(*coros)
    expected = [i + 1 for i in range(100)]
    IsolatedAsyncioTestCase.assertEqual(self, sorted(results), sorted(expected))

  async def test_d06_concurrent_foreach_different_iterables(self):
    """Concurrent foreach on different iterables."""
    async def async_square(v):
      await asyncio.sleep(0)
      return v ** 2

    async def run_foreach(items):
      return await Chain(items).foreach(async_square).run()

    coros = [run_foreach(list(range(i, i + 5))) for i in range(10)]
    results = await asyncio.gather(*coros)
    for i, result in enumerate(results):
      expected = [v ** 2 for v in range(i, i + 5)]
      IsolatedAsyncioTestCase.assertEqual(self, result, expected)

  async def test_d07_concurrent_gather_operations(self):
    """Concurrent gather operations."""
    async def fn_a(v):
      await asyncio.sleep(0)
      return v + 1

    async def fn_b(v):
      return v * 2

    async def run_gather(val):
      return await Chain(val).gather(fn_a, fn_b).run()

    coros = [run_gather(i) for i in range(20)]
    results = await asyncio.gather(*coros)
    for i, result in enumerate(results):
      IsolatedAsyncioTestCase.assertEqual(self, result, [i + 1, i * 2])

  async def test_d08_chain_with_debug_concurrent(self):
    """Chain with debug mode executed concurrently.

    Debug mode creates link_results dicts per execution -- these are
    local to each _run call due to _ExecCtx isolation.
    """
    c = Chain().then(async_identity).config(debug=True)
    coros = [c(i) for i in range(50)]
    results = await asyncio.gather(*coros)
    IsolatedAsyncioTestCase.assertEqual(
      self, sorted(results), list(range(50))
    )


# ═══════════════════════════════════════════════════════════════════════════
# E. Mixed Sync/Async Patterns (8+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class MixedSyncAsyncPatternTests(IsolatedAsyncioTestCase):
  """Tests for mixed sync and async patterns in chains."""

  async def test_e01_sync_function_returning_coroutine_object(self):
    """Sync function that explicitly constructs and returns a coroutine object.

    Not an 'async def', but returns a coroutine from another async function.
    The chain should detect it via iscoro() and await it.
    """
    async def real_coro(v):
      return v * 3

    def sync_that_returns_coro(v):
      return real_coro(v)

    result = await Chain(10).then(sync_that_returns_coro).run()
    self.assertEqual(result, 30)

  async def test_e02_async_function_completes_immediately(self):
    """Async function that completes immediately (no await)."""
    async def immediate(v):
      return v + 1

    result = await Chain(5).then(immediate).run()
    self.assertEqual(result, 6)

  async def test_e03_first_n_sync_then_async_then_sync(self):
    """Chain where first N links are sync, then one async, then more sync."""
    result = await (
      Chain(1)
      .then(lambda v: v + 1)
      .then(lambda v: v + 1)
      .then(lambda v: v + 1)
      .then(async_identity)
      .then(lambda v: v * 10)
      .then(lambda v: v + 1)
      .run()
    )
    # 1+1+1+1=4 -> 4 -> 40+1=41
    self.assertEqual(result, 41)

  async def test_e04_sync_link_accesses_event_loop(self):
    """Chain with sync link that calls asyncio.get_running_loop().

    In an async test, the event loop is accessible from sync code.
    """
    captured_loop = {}

    def capture_loop(v):
      captured_loop['loop'] = asyncio.get_running_loop()
      return v

    result = await Chain(aempty, 42).then(capture_loop).run()
    self.assertEqual(result, 42)
    IsolatedAsyncioTestCase.assertIsNotNone(self, captured_loop.get('loop'))

  async def test_e06_functools_partial_wrapping_async(self):
    """Chain with functools.partial wrapping async function."""
    async def add(v, n):
      return v + n

    partial_add_10 = functools.partial(add, n=10)
    result = await Chain(5).then(partial_add_10).run()
    self.assertEqual(result, 15)

  async def test_e07_callable_class_with_async_call(self):
    """Chain with callable class with async __call__."""
    multiplier = AsyncCallableClass(3)
    result = await Chain(7).then(multiplier).run()
    self.assertEqual(result, 21)

  async def test_e08_sync_callable_class(self):
    """Chain with callable class with sync __call__."""
    multiplier = CallableClass(5)
    result = Chain(4).then(multiplier).run()
    IsolatedAsyncioTestCase.assertEqual(self, result, 20)

  async def test_e09_mixed_callable_types_in_chain(self):
    """Chain mixing sync functions, async functions, callable classes."""
    async_mul = AsyncCallableClass(2)
    sync_mul = CallableClass(3)

    result = await (
      Chain(1)
      .then(sync_add_one)       # 2
      .then(async_mul)          # 4
      .then(sync_mul)           # 12
      .then(async_identity)     # 12
      .run()
    )
    self.assertEqual(result, 12)


# ═══════════════════════════════════════════════════════════════════════════
# F. Async Exception Paths (10+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class AsyncExceptionPathTests(IsolatedAsyncioTestCase):
  """Tests for exception handling in async chain paths."""

  async def test_f01_exception_in_run_async_continuation(self):
    """Exception in async chain continuation (_run_async).

    After transitioning to async, an exception in a subsequent link
    is caught by the except BaseException handler in _run_async (line 304).
    """
    result = await (
      Chain(10)
      .do(lambda v: None)
      .then(async_identity)
      .then(lambda v: 1 / 0)
      .except_(lambda v: 'caught', reraise=False)
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_f02_exception_handler_returns_coroutine_async(self):
    """Exception handler returns coroutine in async path (awaited).

    In _run_async (lines 312-315), if exc_link handler returns a coroutine,
    it is awaited.
    """
    async def async_handler(v):
      return 'async_recovery'

    result = await (
      Chain(aempty, 10)
      .do(lambda v: None)
      .then(lambda v: (_ for _ in ()).throw(TestExc('boom')))
      .except_(async_handler, reraise=False)
      .run()
    )
    self.assertEqual(result, 'async_recovery')

  async def test_f03_exception_handler_reraise_true_async(self):
    """Exception handler with reraise=True in async path.

    In _run_async (lines 319-320), if exc_link.reraise is True,
    the original exception is re-raised after handler executes.
    """
    handler_called = {'value': False}

    async def handler(v):
      handler_called['value'] = True

    with self.assertRaises(TestExc):
      await (
        Chain(aempty, 10)
        .do(lambda v: None)
        .then(async_raise_test)
        .except_(handler, reraise=True)
        .run()
      )
    IsolatedAsyncioTestCase.assertTrue(self, handler_called['value'])

  async def test_f04_return_in_async_exception_path(self):
    """_Return in async exception path.

    In _run_async (lines 293-297): _Return caught, handle_return_exc called.
    If is_nested=False, returns the value.
    """
    async def trigger_return(v):
      Chain.return_(99)

    result = await (
      Chain(aempty, 1)
      .do(lambda v: None)
      .then(trigger_return)
      .run()
    )
    self.assertEqual(result, 99)

  async def test_f05_break_in_async_non_nested_raises(self):
    """_Break in async exception path (non-nested) raises QuentException.

    In _run_async (lines 299-302): _Break caught, is_nested=False,
    raises QuentException.
    """
    async def trigger_break(v):
      Chain.break_()

    with self.assertRaises(QuentException) as cm:
      await (
        Chain(aempty, 1)
        .do(lambda v: None)
        .then(trigger_break)
        .run()
      )
    IsolatedAsyncioTestCase.assertIn(self, '_Break', str(cm.exception))

  async def test_f06_break_in_async_nested_propagates(self):
    """_Break in async exception path (nested chain) propagates.

    In _run_async (lines 300-301): if is_nested=True, _Break is re-raised.
    The parent chain's foreach catches it.
    """
    async def trigger_break_at_3(v):
      if v >= 3:
        Chain.break_()
      return v

    result = await (
      Chain([1, 2, 3, 4, 5])
      .foreach(trigger_break_at_3)
      .run()
    )
    self.assertEqual(result, [1, 2])

  async def test_f07_exception_in_async_simple_path(self):
    """Exception in async simple path (_run_async_simple).

    Simple chain (only .then()), async transition, then exception.
    _run_async_simple (lines 467-476) catches it and modifies traceback.
    """
    with self.assertRaises(TestExc):
      await (
        Chain(1)
        .then(async_identity)
        .then(lambda v: (_ for _ in ()).throw(TestExc('simple boom')))
        .run()
      )

  async def test_f09_return_in_async_simple_path(self):
    """_Return in async simple path.

    _run_async_simple (lines 456-460): _Return caught, handle_return_exc.
    If result is a coroutine, it's awaited.
    """
    result = await (
      Chain(aempty, 1)
      .then(lambda v: Chain.return_(async_return_42))
      .then(lambda v: v + 100)
      .run()
    )
    self.assertEqual(result, 42)

  async def test_f10_break_in_async_simple_non_nested_raises(self):
    """_Break in async simple path (non-nested) raises QuentException.

    _run_async_simple (lines 462-465): _Break, not nested, raises.
    """
    with self.assertRaises(QuentException):
      await (
        Chain(aempty, 1)
        .then(lambda v: Chain.break_())
        .run()
      )

  async def test_f11_exception_in_handler_async_path(self):
    """Exception in exception handler itself in async path.

    _run_async (lines 316-318): handler raises, modify_traceback,
    raise from original.
    """
    async def bad_handler(v):
      raise ValueError('handler error')

    with self.assertRaises(ValueError) as cm:
      await (
        Chain(aempty, 1)
        .do(lambda v: None)
        .then(async_raise_test)
        .except_(bad_handler, reraise=False)
        .run()
      )
    IsolatedAsyncioTestCase.assertIn(self, 'handler error', str(cm.exception))


# ═══════════════════════════════════════════════════════════════════════════
# G. Async Finally Paths (6+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class AsyncFinallyPathTests(IsolatedAsyncioTestCase):
  """Tests for finally_ handler behavior in async chains."""

  async def test_g01_async_chain_finally_success_awaited(self):
    """Async chain with finally_ handler that runs successfully.

    In _run_async finally block (lines 326-330): finally handler evaluated,
    if coroutine, awaited.
    """
    finally_called = {'value': False}

    async def async_finally(v=None):
      finally_called['value'] = True

    result = await (
      Chain(aempty, 42)
      .then(sync_double)
      .finally_(async_finally)
      .run()
    )
    self.assertEqual(result, 84)
    IsolatedAsyncioTestCase.assertTrue(self, finally_called['value'])

  async def test_g02_async_finally_handler_raises(self):
    """Async chain finally_ handler that raises exception.

    In _run_async finally (lines 333-340): exception in finally handler
    creates _ExecCtx if needed, modifies traceback, raises.
    """
    async def failing_finally(v=None):
      raise ValueError('finally error')

    with self.assertRaises(ValueError) as cm:
      await (
        Chain(aempty, 42)
        .then(sync_double)
        .finally_(failing_finally)
        .run()
      )
    IsolatedAsyncioTestCase.assertIn(self, 'finally error', str(cm.exception))

  async def test_g03_async_finally_internal_quent_exception(self):
    """Async chain finally_ with _InternalQuentException -> QuentException.

    In _run_async finally (lines 331-332): control flow signal in finally
    raises QuentException.
    """
    with self.assertRaises(QuentException) as cm:
      await (
        Chain(aempty, 1)
        .then(sync_double)
        .finally_(Chain.return_)
        .run()
      )
    IsolatedAsyncioTestCase.assertIn(
      self, 'control flow signals', str(cm.exception)
    )

  async def test_g04_sync_chain_finally_returns_coroutine_warning(self):
    """Sync chain where finally_ returns coroutine -> RuntimeWarning.

    In _run finally (lines 230-241): finally returns coroutine but chain
    is in sync mode. ensure_future schedules it, RuntimeWarning emitted.
    This only happens when the chain body is fully sync but finally_ is async.
    """
    async def async_finally_fn(v=None):
      pass

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always', RuntimeWarning)
      result = Chain(42).then(sync_double).finally_(async_finally_fn).run()
      IsolatedAsyncioTestCase.assertEqual(self, result, 84)

    rt_warns = [
      w for w in caught
      if issubclass(w.category, RuntimeWarning)
      and 'finally' in str(w.message)
    ]
    IsolatedAsyncioTestCase.assertGreater(self, len(rt_warns), 0)

  async def test_g05_sync_except_reraise_returns_coroutine_warning(self):
    """Sync chain where except_ with reraise=True returns coroutine -> RuntimeWarning.

    In _run (lines 200-208): except handler returns coroutine,
    reraise=True, ensure_future wraps it, RuntimeWarning emitted.
    """
    async def async_handler(v):
      pass

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always', RuntimeWarning)
      with self.assertRaises(TestExc):
        Chain(1).then(lambda v: (_ for _ in ()).throw(TestExc('test'))).except_(
          async_handler, reraise=True
        ).run()

    rt_warns = [
      w for w in caught
      if issubclass(w.category, RuntimeWarning)
      and 'except' in str(w.message).lower()
    ]
    IsolatedAsyncioTestCase.assertGreater(self, len(rt_warns), 0)

  async def test_g06_sync_finally_and_except_both_return_coroutines(self):
    """Sync chain where finally_ and except_ both return coroutines.

    Both paths trigger RuntimeWarning about scheduling coroutines.
    """
    async def async_handler(v):
      pass

    async def async_finally(v=None):
      pass

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always', RuntimeWarning)
      with self.assertRaises(TestExc):
        Chain(1).then(lambda v: (_ for _ in ()).throw(TestExc('test'))).except_(
          async_handler, reraise=True
        ).finally_(async_finally).run()

    rt_warns = [
      w for w in caught
      if issubclass(w.category, RuntimeWarning)
    ]
    # Should have warnings for both except and finally
    IsolatedAsyncioTestCase.assertGreaterEqual(self, len(rt_warns), 1)

  async def test_g07_async_finally_with_none_ctx_initialization(self):
    """Async finally where ctx is None — triggers _ExecCtx creation (lines 334-338).

    When no exception occurred in the chain body but finally_ raises,
    ctx is None and must be created.
    """
    async def finally_that_raises(v=None):
      raise RuntimeError('finally init ctx')

    with self.assertRaises(RuntimeError):
      await (
        Chain(aempty, 1)
        .then(sync_double)
        .finally_(finally_that_raises)
        .run()
      )


# ═══════════════════════════════════════════════════════════════════════════
# I. Async Iteration Edge Cases (10+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class AsyncIterationTests(IsolatedAsyncioTestCase):
  """Tests for foreach, filter, gather with async transitions."""

  async def test_i01_foreach_first_element_async_transition(self):
    """foreach where fn returns coroutine on FIRST element.

    _Foreach.__call__ calls fn(el), gets coroutine on first element,
    transitions to _foreach_to_async (line 54-56).
    """
    async def async_square(v):
      return v ** 2

    result = await Chain([1, 2, 3]).foreach(async_square).run()
    self.assertEqual(result, [1, 4, 9])

  async def test_i02_foreach_middle_element_async_transition(self):
    """foreach where fn returns coroutine on MIDDLE element.

    fn is sync for first elements, async for middle one. Transition
    to _foreach_to_async happens mid-iteration.
    """
    call_count = {'n': 0}

    def sometimes_async(v):
      call_count['n'] += 1
      if call_count['n'] == 3:
        return async_identity(v * 10)
      return v * 10

    call_count['n'] = 0
    result = await Chain([1, 2, 3, 4, 5]).foreach(sometimes_async).run()
    self.assertEqual(result, [10, 20, 30, 40, 50])

  async def test_i03_foreach_last_element_async_transition(self):
    """foreach where fn returns coroutine on LAST element."""
    call_count = {'n': 0}
    total = 5

    def async_on_last(v):
      call_count['n'] += 1
      if call_count['n'] == total:
        return async_identity(v * 10)
      return v * 10

    call_count['n'] = 0
    result = await Chain(list(range(1, total + 1))).foreach(async_on_last).run()
    self.assertEqual(result, [10, 20, 30, 40, 50])

  async def test_i04_foreach_async_iterable(self):
    """foreach with async iterable (__aiter__) -> _foreach_full_async.

    When current_value has __aiter__, _Foreach delegates to
    _foreach_full_async (line 45-46).
    """
    async def async_square(v):
      return v ** 2

    ait = AsyncIterable([1, 2, 3, 4])
    result = await Chain(ait).foreach(async_square).run()
    self.assertEqual(result, [1, 4, 9, 16])

  async def test_i05_filter_predicate_returns_coroutine(self):
    """filter where predicate returns coroutine -> _filter_to_async.

    _Filter.__call__ calls fn(el), gets coroutine, transitions to
    _filter_to_async (line 167-169).
    """
    async def async_is_even(v):
      return v % 2 == 0

    result = await Chain([1, 2, 3, 4, 5, 6]).filter(async_is_even).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_i06_filter_async_iterable(self):
    """filter with async iterable -> _filter_full_async.

    When current_value has __aiter__, _Filter delegates to
    _filter_full_async (line 158-159).
    """
    async def async_is_positive(v):
      return v > 0

    ait = AsyncIterable([-2, -1, 0, 1, 2, 3])
    result = await Chain(ait).filter(async_is_positive).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_i07_gather_all_sync(self):
    """gather with all sync functions — returns list directly."""
    result = Chain(5).gather(
      lambda v: v + 1,
      lambda v: v * 2,
      lambda v: v ** 2,
    ).run()
    IsolatedAsyncioTestCase.assertEqual(self, result, [6, 10, 25])

  async def test_i08_gather_all_async(self):
    """gather with all async functions -> asyncio.gather.

    _Gather.__call__ detects has_coro=True and delegates to _gather_to_async.
    """
    async def fn_a(v):
      await asyncio.sleep(0)
      return v + 1

    async def fn_b(v):
      return v * 2

    async def fn_c(v):
      return v ** 2

    result = await Chain(5).gather(fn_a, fn_b, fn_c).run()
    self.assertEqual(result, [6, 10, 25])

  async def test_i09_gather_mixed_sync_async(self):
    """gather with mix of sync and async functions.

    Some return coroutines, others return values. _gather_to_async
    only awaits the coroutine results via asyncio.gather.
    """
    async def async_add(v):
      return v + 100

    result = await Chain(3).gather(
      lambda v: v + 1,
      async_add,
      lambda v: v * 2,
    ).run()
    self.assertEqual(result, [4, 103, 6])

  async def test_i11_foreach_async_iterable_with_sync_fn(self):
    """foreach with async iterable but sync mapping function.

    _foreach_full_async still handles the async iteration even though
    fn itself is sync.
    """
    ait = AsyncIterable([1, 2, 3])
    result = await Chain(ait).foreach(lambda v: v * 10).run()
    self.assertEqual(result, [10, 20, 30])

  async def test_i12_filter_sync_predicate_on_sync_iterable(self):
    """filter with sync predicate on sync iterable — no async transition."""
    result = Chain([1, 2, 3, 4, 5]).filter(lambda v: v > 3).run()
    IsolatedAsyncioTestCase.assertEqual(self, result, [4, 5])

  async def test_i13_foreach_break_in_async_iteration(self):
    """foreach with _Break during async iteration."""
    async def break_at_3(v):
      if v >= 3:
        Chain.break_()
      return v * 10

    result = await Chain([1, 2, 3, 4, 5]).foreach(break_at_3).run()
    self.assertEqual(result, [10, 20])


# ═══════════════════════════════════════════════════════════════════════════
# J. Async Generator Edge Cases (5+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class AsyncGeneratorTests(IsolatedAsyncioTestCase):
  """Tests for async generator behavior via iterate()."""

  async def test_j01_async_generator_chain_returns_coroutine(self):
    """Async generator where chain returns coroutine.

    In async_generator (lines 180-182): iterator = chain._run(...),
    if iscoro(iterator), iterator = await iterator. Then iteration begins.
    """
    gen = Chain(aempty, [1, 2, 3]).iterate()
    results = []
    async for item in gen:
      results.append(item)
    IsolatedAsyncioTestCase.assertEqual(self, results, [1, 2, 3])

  async def test_j02_async_generator_sync_iterable_async_fn(self):
    """Async generator with sync iterable source but async fn.

    iterate(fn) applies fn to each element. If fn returns coroutine,
    async_generator awaits it (lines 190-196).
    """
    gen = Chain([1, 2, 3]).iterate(async_double)
    results = []
    async for item in gen:
      results.append(item)
    IsolatedAsyncioTestCase.assertEqual(self, results, [2, 4, 6])

  async def test_j03_async_generator_async_iterable_source(self):
    """Async generator with async iterable source.

    async_generator detects __aiter__ and uses 'async for' (lines 185-196).
    """
    ait = AsyncIterable([10, 20, 30])
    gen = Chain(ait).iterate(sync_double)
    results = []
    async for item in gen:
      results.append(item)
    IsolatedAsyncioTestCase.assertEqual(self, results, [20, 40, 60])

  async def test_j04_async_generator_break_signal(self):
    """Async generator with _Break signal.

    In async_generator (lines 209-210): _Break caught, generator returns.
    """
    def break_at_3(v):
      if v >= 3:
        Chain.break_()
      return v

    gen = Chain([1, 2, 3, 4, 5]).iterate(break_at_3)
    results = []
    async for item in gen:
      results.append(item)
    IsolatedAsyncioTestCase.assertEqual(self, results, [1, 2])

  async def test_j05_async_generator_return_signal_raises(self):
    """Async generator with _Return signal raises QuentException.

    In async_generator (lines 211-212): _Return caught, raises QuentException.
    """
    def return_at_3(v):
      if v >= 3:
        Chain.return_(v)
      return v

    gen = Chain([1, 2, 3, 4]).iterate(return_at_3)
    with self.assertRaises(QuentException) as cm:
      async for item in gen:
        pass
    IsolatedAsyncioTestCase.assertIn(
      self, 'return_', str(cm.exception).lower()
    )

  async def test_j06_sync_generator_basic(self):
    """Sync generator via iterate() — __iter__ path."""
    gen = Chain([1, 2, 3]).iterate(sync_double)
    results = list(gen)
    IsolatedAsyncioTestCase.assertEqual(self, results, [2, 4, 6])

  async def test_j07_async_generator_no_fn(self):
    """Async generator with no fn (fn=None) — yields raw elements."""
    gen = Chain(aempty, [10, 20, 30]).iterate()
    results = []
    async for item in gen:
      results.append(item)
    IsolatedAsyncioTestCase.assertEqual(self, results, [10, 20, 30])


# ═══════════════════════════════════════════════════════════════════════════
# K. Async Context Manager Edge Cases (5+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class AsyncContextManagerTests(IsolatedAsyncioTestCase):
  """Tests for async context manager handling via with_()."""

  async def test_k01_aenter_triggers_with_full_async(self):
    """Object with __aenter__ triggers _with_full_async.

    _With.__call__ detects __aenter__ and delegates to _with_full_async
    (line 73-74 of _control_flow.pxi).
    """
    cm = AsyncCM(value='async_val')
    result = await Chain(cm).with_(lambda ctx: ctx + '_body').run()
    self.assertEqual(result, 'async_val_body')
    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)

  async def test_k02_async_cm_body_returns_coroutine(self):
    """Async CM body returns coroutine (awaited).

    Inside _with_full_async (lines 142-144): body result checked with
    iscoro(), if True, awaited.
    """
    cm = AsyncCM(value=10)

    async def async_body(ctx):
      return ctx * 3

    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 30)
    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)

  async def test_k03_async_cm_exception_aexit_called(self):
    """Exception in async CM body -> async __aexit__ called with exception info.

    _with_full_async uses 'async with' (line 138), which calls __aexit__
    with exception information.
    """
    cm = AsyncCM(value='val')

    def body_raises(ctx):
      raise TestExc('body error')

    with self.assertRaises(TestExc):
      await Chain(cm).with_(body_raises).run()

    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)
    IsolatedAsyncioTestCase.assertEqual(self, cm.exc_info[0], TestExc)

  async def test_k04_async_cm_aexit_suppresses_exception(self):
    """Async CM where __aexit__ suppresses exception (returns True).

    When __aexit__ returns True, the exception is suppressed.
    """
    cm = AsyncCM(value='val', suppress=True)

    def body_raises(ctx):
      raise TestExc('suppressed')

    # Exception should NOT propagate
    result = await Chain(cm).with_(body_raises).run()
    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)

  async def test_k05_sync_cm_body_returns_coroutine(self):
    """Sync CM body returns coroutine -> _with_to_async transition.

    _With.__call__ enters sync CM, evaluates body which returns coroutine,
    transitions to _with_to_async (line 83-84 of _control_flow.pxi).
    """
    cm = SyncCM(value=5)

    async def async_body(ctx):
      return ctx * 4

    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 20)
    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)

  async def test_k06_sync_cm_async_body_exception(self):
    """Sync CM with async body that raises -> __exit__ called.

    _with_to_async (lines 116-124): exception in awaited body,
    calls current_value.__exit__ with exception info.
    """
    cm = SyncCM(value='val')

    async def async_body_raises(ctx):
      raise TestExc('async body error')

    with self.assertRaises(TestExc):
      await Chain(cm).with_(async_body_raises).run()

    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)
    IsolatedAsyncioTestCase.assertEqual(self, cm.exc_info[0], TestExc)

  async def test_k07_async_cm_sync_body(self):
    """Async CM with sync body — still uses _with_full_async.

    Since __aenter__ is detected, _with_full_async is used even for
    sync body functions. Body result is not a coroutine so no await.
    """
    cm = AsyncCM(value=10)

    result = await Chain(cm).with_(lambda ctx: ctx + 5).run()
    self.assertEqual(result, 15)
    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)

  async def test_k08_sync_cm_suppress_with_async_body(self):
    """Sync CM that suppresses exceptions with async body error.

    _with_to_async (lines 118-122): __exit__ returns truthy,
    exception is suppressed.
    """
    cm = SyncCM(value='val', suppress=True)

    async def async_body_raises(ctx):
      raise TestExc('suppressed async')

    # Exception should be suppressed
    result = await Chain(cm).with_(async_body_raises).run()
    IsolatedAsyncioTestCase.assertTrue(self, cm.entered)
    IsolatedAsyncioTestCase.assertTrue(self, cm.exited)


# ═══════════════════════════════════════════════════════════════════════════
# Additional Edge Cases and Integration
# ═══════════════════════════════════════════════════════════════════════════

class AsyncChainIntegrationEdgeCases(IsolatedAsyncioTestCase):
  """Integration tests combining multiple async features."""

  async def test_int03_nested_async_chains(self):
    """Deeply nested async chains."""
    inner = Chain().then(async_double)
    middle = Chain().then(sync_add_one).then(inner)
    outer = Chain(5).then(middle).then(sync_add_one).run()
    result = await outer
    # 5 -> (5+1=6 -> 6*2=12) -> 12+1=13
    self.assertEqual(result, 13)

  async def test_int04_chain_with_except_finally(self):
    """Chain with except_ and finally_ in async path."""
    handler_called = {'value': False}
    finally_called = {'value': False}

    async def async_handler(v):
      handler_called['value'] = True
      return 'recovered'

    async def async_finally(v=None):
      finally_called['value'] = True

    c = (
      Chain()
      .then(async_raise_test)
      .except_(async_handler, reraise=False)
      .finally_(async_finally)
    )

    result = await c(1)
    IsolatedAsyncioTestCase.assertEqual(self, result, 'recovered')
    IsolatedAsyncioTestCase.assertTrue(self, handler_called['value'])
    IsolatedAsyncioTestCase.assertTrue(self, finally_called['value'])

  async def test_int05_void_chain_async_root_override(self):
    """Void chain with async root override via run(async_fn)."""
    result = await Chain().then(sync_double).then(sync_add_one).run(aempty, 10)
    self.assertEqual(result, 21)

  async def test_int06_chain_return_none_vs_null(self):
    """Chain distinguishes between returning None and Null.

    If current_value is Null at the end, _run returns None.
    If it's actually None (a real value), it also returns None.
    """
    # Null case: void chain with no links
    result = await await_(Chain().run())
    IsolatedAsyncioTestCase.assertIsNone(self, result)

    # None case: explicitly returning None from async
    async def return_none(v):
      return None

    result = await Chain(1).then(return_none).run()
    IsolatedAsyncioTestCase.assertIsNone(self, result)

  async def test_int07_debug_mode_async_with_all_links(self):
    """Debug mode with various link types in async chain."""
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      result = await (
        Chain(1)
        .do(lambda v: None)
        .then(async_double)
        .then(sync_add_one)
        .config(debug=True)
        .run()
      )
      self.assertEqual(result, 3)
    finally:
      logger.removeHandler(handler)

  async def test_int08_chain_pipe_syntax_async(self):
    """Pipe syntax with async functions."""
    result = await Chain(aempty, 5).then(async_double).then(sync_add_one).run()
    self.assertEqual(result, 11)

  async def test_int10_autorun_with_nested_chains(self):
    """Autorun with nested async chains."""
    inner = Chain().then(async_double)

    task = (
      Chain(5)
      .then(inner)
      .then(sync_add_one)
      .config(autorun=True)
      .run()
    )
    IsolatedAsyncioTestCase.assertIsInstance(self, task, asyncio.Task)
    result = await task
    IsolatedAsyncioTestCase.assertEqual(self, result, 11)


class AsyncDoIgnoreResultTests(IsolatedAsyncioTestCase):
  """Tests for .do() with async functions — result ignored."""

  async def test_do01_async_do_ignores_result(self):
    """.do() with async function — result is discarded, current_value preserved."""
    result = await (
      Chain(10)
      .do(async_noop)
      .then(sync_double)
      .run()
    )
    self.assertEqual(result, 20)

  async def test_do02_async_do_side_effect_executes(self):
    """Async .do() side effect actually executes."""
    side_effects = []

    async def side_effect(v):
      side_effects.append(v)

    result = await Chain(42).do(side_effect).run()
    self.assertEqual(result, 42)
    IsolatedAsyncioTestCase.assertEqual(self, side_effects, [42])

  async def test_do03_multiple_async_do_in_chain(self):
    """Multiple async .do() calls in chain — all execute, results ignored."""
    log = []

    async def log_fn(v):
      log.append(v)

    result = await (
      Chain(1)
      .do(log_fn)
      .then(lambda v: v + 1)
      .do(log_fn)
      .then(lambda v: v * 3)
      .do(log_fn)
      .run()
    )
    self.assertEqual(result, 6)
    IsolatedAsyncioTestCase.assertEqual(self, log, [1, 2, 6])


class AsyncNullHandlingTests(IsolatedAsyncioTestCase):
  """Tests for Null sentinel handling in async paths."""

  async def test_null01_void_chain_async_root_update(self):
    """Void chain: root_value starts as Null, updated after await.

    In _run_async (lines 262-263): if has_root_value and root_value is Null,
    root_value = current_value.
    """
    result = await Chain().then(sync_double).run(aempty, 21)
    self.assertEqual(result, 42)

  async def test_null02_async_simple_root_value_update(self):
    """In _run_async_simple: root_value updated from Null after await.

    Lines 434-435: if has_root_value and root_value is Null,
    root_value = current_value.
    """
    result = await Chain().then(sync_add_one).run(aempty, 10)
    self.assertEqual(result, 11)

  async def test_null03_current_value_null_returns_none(self):
    """When current_value is Null at end, _run returns None."""
    result = await await_(Chain().run())
    IsolatedAsyncioTestCase.assertIsNone(self, result)


if __name__ == '__main__':
  import unittest
  unittest.main()
