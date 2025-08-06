"""Tests for Phase 6: Async Edge Cases.

Covers sync-to-async transitions, task registry behavior, CancelledError handling,
concurrent execution, _await_run paths, sleep behavior, and async return/debug paths.
"""
import asyncio
import logging
import inspect
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException, run
from quent.quent import _get_registry_size


class MyTestCase(IsolatedAsyncioTestCase):
  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)

  async def assertTrue(self, expr, msg=None):
    return super().assertTrue(await await_(expr), msg)

  async def assertFalse(self, expr, msg=None):
    return super().assertFalse(await await_(expr), msg)

  async def assertEqual(self, first, second, msg=None):
    return super().assertEqual(await await_(first), second, msg)

  async def assertIsNone(self, obj, msg=None):
    return super().assertIsNone(await await_(obj), msg)

  async def assertIs(self, expr1, expr2, msg=None):
    return super().assertIs(await await_(expr1), expr2, msg)

  async def assertIsNot(self, expr1, expr2, msg=None):
    return super().assertIsNot(await await_(expr1), expr2, msg)


class SyncToAsyncTransitionTests(MyTestCase):
  """Test sync-to-async transitions in chain execution.

  The chain starts in _run/_run_simple (sync). When any link returns a
  coroutine, execution transfers to _run_async/_run_async_simple where
  the coroutine is awaited and all subsequent links are checked for
  coroutines.
  """

  async def test_sync_root_async_continuation(self):
    """Sync root, first .then() returns coroutine -> transitions to _run_async.

    Uses .do() to force _is_simple=False, ensuring the non-simple path
    (_run -> _run_async) is taken.
    """
    async def async_double(v):
      return v * 2

    result = await Chain(10).do(lambda v: None).then(async_double).run()
    await self.assertEqual(result, 20)

    # Also with except_ to force non-simple path
    result = await Chain(10).except_(lambda v: None).then(async_double).run()
    await self.assertEqual(result, 20)

  async def test_sync_root_async_simple_continuation(self):
    """Same on simple path (only .then() links) -> _run_async_simple.

    When _is_simple=True and no debug/finally/except, _run_simple is used.
    An async .then() link triggers _run_async_simple.
    """
    async def async_add(v):
      return v + 5

    result = await Chain(10).then(async_add).run()
    await self.assertEqual(result, 15)

    # Multiple .then()-only links keep simple path
    result = await Chain(1).then(lambda v: v + 1).then(async_add).run()
    await self.assertEqual(result, 7)

  async def test_sync_links_then_async_link(self):
    """Multiple sync .then() links, then async link mid-chain.

    The chain evaluates sync links in _run_simple, then transitions to
    _run_async_simple when it hits the async link. After transition,
    remaining links continue in the async path.
    """
    async def async_multiply(v):
      return v * 3

    result = await Chain(1).then(lambda v: v + 1).then(lambda v: v + 2).then(async_multiply).then(lambda v: v - 1).run()
    # 1 -> 2 -> 4 -> 12 -> 11
    await self.assertEqual(result, 11)

  async def test_async_root_sync_continuation(self):
    """Async root (aempty), all subsequent links sync.

    The root value is an async function, so _run transitions to _run_async
    immediately. All subsequent sync links are evaluated in the async path
    (they just don't need awaiting).
    """
    obj = object()
    result = await Chain(aempty, obj).then(lambda v: v).run()
    await self.assertIs(result, obj)

  async def test_multiple_async_transitions(self):
    """Chain alternates between sync and async .then() links.

    The first async link triggers transition to the async path.
    All subsequent links (sync or async) are evaluated there.
    """
    async def async_add_one(v):
      return v + 1

    result = await (
      Chain(0)
      .then(lambda v: v + 1)       # sync: 1
      .then(async_add_one)          # async transition: 2
      .then(lambda v: v * 10)       # sync (in async path): 20
      .then(async_add_one)          # async: 21
      .then(lambda v: v + 100)      # sync: 121
      .run()
    )
    await self.assertEqual(result, 121)

  async def test_cascade_async_simple_path(self):
    """Cascade with async root -> _run_async_simple with is_cascade.

    In cascade mode with simple path, _run_async_simple returns root_value
    instead of the last link's result.
    """
    async def async_identity(v):
      return v

    sentinel = object()
    result = await Cascade(async_identity, sentinel).then(lambda v: 'ignored').run()
    await self.assertIs(result, sentinel)

  async def test_void_chain_async_root_value_update(self):
    """Void chain with async root override -> root_value updates from Null.

    In _run_async (line ~664): if has_root_value and root_value is Null,
    root_value = current_value. This updates root_value when a void chain
    gets its first value from an async root.
    """
    async def async_return_42():
      return 42

    # Void chain (no root), running with async root override
    # The root_value starts as Null, gets updated after await
    result = await Chain().then(lambda v: v + 8).run(async_return_42)
    await self.assertEqual(result, 50)

    # With .root() to access the updated root_value
    result = await Chain().then(lambda v: v * 2).root(lambda v: v + 1).run(async_return_42)
    # async_return_42 -> 42, then(v*2) -> 84, root(v+1) -> 42+1=43
    await self.assertEqual(result, 43)

  async def test_non_void_chain_root_not_overwritten(self):
    """Chain with existing root, async link -> root_value stays original.

    When the chain has a root value, root_value is set from the root link.
    Even if a later link is async, root_value does not change.
    """
    async def async_double(v):
      return v * 2

    # root=5, then async_double -> 10, root(lambda v: v) accesses root which is 5
    result = await Chain(5).then(async_double).root(lambda v: v).run()
    await self.assertEqual(result, 5)


class TaskRegistryTests(MyTestCase):
  """Test task registry behavior for autorun chains.

  ensure_future() creates a task, adds it to task_registry, and attaches a
  done callback to remove it. _get_registry_size() returns len(task_registry).
  """

  async def test_ensure_future_adds_to_registry(self):
    """After autorun chain, _get_registry_size() increases."""
    initial_size = _get_registry_size()

    async def slow_fn(v):
      await asyncio.sleep(0.1)
      return v

    task = Chain(10).then(slow_fn).autorun().run()
    # Task should be in the registry now
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), initial_size)
    # Clean up: await the task
    await task

  async def test_task_removed_after_done(self):
    """After task completes, registry size decreases."""
    initial_size = _get_registry_size()

    async def fast_fn(v):
      return v

    task = Chain(10).then(fast_fn).autorun().run()
    # Wait for the task to complete
    await task
    # Give the event loop a tick for the done callback to fire
    await asyncio.sleep(0)
    IsolatedAsyncioTestCase.assertEqual(self, _get_registry_size(), initial_size)

  async def test_autorun_adds_to_registry(self):
    """chain.autorun().run() adds task to registry."""
    initial_size = _get_registry_size()

    async def wait_fn(v):
      await asyncio.sleep(0.05)
      return v * 2

    task = Chain(5).then(wait_fn).autorun().run()
    current_size = _get_registry_size()
    IsolatedAsyncioTestCase.assertGreater(self, current_size, initial_size)
    result = await task
    await self.assertEqual(result, 10)

  async def test_registry_prevents_gc(self):
    """Task is not garbage collected while in registry (verify task completes).

    The task_registry holds strong references to tasks, preventing the event
    loop from dropping them. We verify the task actually completes by checking
    its result.
    """
    completed = {"value": False}

    async def set_completed(v):
      await asyncio.sleep(0.02)
      completed["value"] = True
      return v

    task = Chain(1).then(set_completed).autorun().run()
    # The task is in the registry holding a strong reference
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), 0)
    await task
    await self.assertTrue(completed["value"])


class CancelledErrorTests(MyTestCase):
  """Test CancelledError handling in chain execution.

  CancelledError is a BaseException (Python 3.9+). except_(handler)
  defaults to exceptions=(Exception,), so it does NOT catch CancelledError.
  finally_ still runs because it's in the finally: block.
  """

  async def test_cancelled_error_not_caught_by_except(self):
    """CancelledError is NOT caught by except_(handler) with default exceptions.

    except_ defaults to exceptions=(Exception,). Since CancelledError inherits
    from BaseException (not Exception), it passes through the except handler.
    """
    handler_called = {"value": False}

    def handler(v=None):
      handler_called["value"] = True

    async def raise_cancelled(v):
      raise asyncio.CancelledError()

    with self.assertRaises(asyncio.CancelledError):
      await Chain(1).then(raise_cancelled).except_(handler).run()

    await self.assertFalse(handler_called["value"])

  async def test_finally_runs_on_cancellation(self):
    """Finally callback still runs when CancelledError occurs in the chain.

    The finally_ handler is in the finally: block of _run_async, so it
    executes regardless of exception type.
    """
    finally_called = {"value": False}

    def on_finally(v=None):
      finally_called["value"] = True

    async def raise_cancelled(v):
      raise asyncio.CancelledError()

    with self.assertRaises(asyncio.CancelledError):
      await Chain(1).then(raise_cancelled).finally_(on_finally).run()

    await self.assertTrue(finally_called["value"])


class SleepBehaviorTests(MyTestCase):
  """Test sleep behavior in chains.

  _Sleep checks for a running event loop. If present, it returns asyncio.sleep
  (a coroutine), triggering async transition. The sleep link has
  ignore_result=True, so the value passes through unchanged.
  """

  async def test_sleep_in_async_context(self):
    """Chain(1).sleep(0.01).run() uses asyncio.sleep in event loop.

    In an async test, there is a running event loop. _Sleep detects this
    and returns asyncio.sleep(delay), causing a sync-to-async transition.
    """
    import time
    start = time.monotonic()
    result = await Chain(42).sleep(0.01).run()
    elapsed = time.monotonic() - start
    await self.assertEqual(result, 42)
    # Verify we actually slept (asyncio.sleep, not time.sleep which would block)
    IsolatedAsyncioTestCase.assertGreaterEqual(self, elapsed, 0.005)

  async def test_sleep_preserves_value(self):
    """Sleep link has ignore_result=True, value passes through unchanged."""
    sentinel = object()
    result = await Chain(sentinel).sleep(0.001).run()
    await self.assertIs(result, sentinel)

  async def test_sleep_in_chain_sequence(self):
    """Chain(1).then(fn).sleep(0.01).then(fn2).run() -- value flows correctly.

    The sleep link sits between two .then() links. Since sleep has
    ignore_result=True, the value from fn passes through the sleep to fn2.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(5).then(fn).sleep(0.001).then(lambda v: v * 3).run()
        )
        await self.assertEqual(result, 15)


class ReturnAsyncTests(MyTestCase):
  """Test Chain.return_() behavior in async chains.

  _Return is caught in _run_async: result = handle_return_exc(exc, self.is_nested).
  If result is a coroutine (iscoro(result)), it is awaited.
  """

  async def test_return_with_awaitable_value(self):
    """Chain.return_(async_fn) where async_fn returns coroutine.

    handle_return_exc evaluates the value, which produces a coroutine.
    _run_async_simple detects iscoro(result) and awaits it.
    """
    async def async_return_99():
      return 99

    result = await Chain(1).then(lambda v: Chain.return_(async_return_99)).run()
    await self.assertEqual(result, 99)

  async def test_return_with_null_value(self):
    """Chain.return_() with no args -> returns None (exc._v is Null).

    handle_return_exc checks if exc._v is Null and returns None.
    """
    result = await Chain(aempty, 1).then(lambda v: Chain.return_()).run()
    await self.assertIsNone(result)

  async def test_return_in_nested_async_chain(self):
    """_Return in nested async chain propagates to parent.

    In a nested chain, handle_return_exc with propagate=True re-raises
    the _Return. The parent chain catches it. Since _Return propagates
    all the way up, the parent chain also returns the _Return value,
    skipping any subsequent links.
    """
    async def async_identity(v):
      return v

    inner = Chain().then(async_identity).then(lambda v: Chain.return_(42))
    result = await Chain(10).then(inner).then(lambda v: v + 100).run()
    # The inner chain raises _Return(42), which propagates to the parent.
    # The parent catches it and returns 42, skipping .then(v + 100).
    await self.assertEqual(result, 42)


class AsyncDebugTests(MyTestCase):
  """Test debug mode in async chain execution.

  When _debug=True, _run_async logs values via _logger.debug() and
  populates link_results dict with per-link values.
  """

  async def test_debug_async_logs_values(self):
    """Debug mode in async chain logs values (check via logging capture).

    The sync _run path logs the root value before transitioning to
    _run_async. The async path populates link_results but does not
    emit additional _logger.debug calls. So we verify the root value
    is logged in the sync phase and the final result is correct.
    """
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      async def async_double(v):
        return v * 2

      result = await Chain(5).then(async_double).config(debug=True).run()
      await self.assertEqual(result, 10)
      # The root value (5) is logged in the sync _run path before
      # transitioning to _run_async
      IsolatedAsyncioTestCase.assertGreaterEqual(self, len(logs), 1)
      IsolatedAsyncioTestCase.assertTrue(self, any('5' in log for log in logs))
    finally:
      logger.removeHandler(handler)

  async def test_debug_async_link_results_stored(self):
    """link_results dict populated in async path.

    In _run_async, when _debug=True, link_results is populated with
    {id(link): value} for each evaluated link. The sync _run path logs
    the root value before transition. In the async continuation, link_results
    is populated (used for exception tracebacks) but _logger.debug is not
    called. We verify the chain computes correctly with debug enabled and
    the root value is logged.
    """
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      async def async_add_one(v):
        return v + 1

      result = await (
        Chain(1)
        .then(async_add_one)      # async: 2
        .then(lambda v: v * 10)   # sync (in async path): 20
        .config(debug=True)
        .run()
      )
      await self.assertEqual(result, 20)
      # Root value (1) is logged in the sync _run path
      IsolatedAsyncioTestCase.assertGreaterEqual(self, len(logs), 1)
      IsolatedAsyncioTestCase.assertTrue(self, any('1' in log for log in logs))

      # Verify that debug mode with multiple sync links before async
      # logs each sync link before the transition point
      logs.clear()
      result = await (
        Chain(2)
        .then(lambda v: v + 3)    # sync: 5 (logged in _run)
        .then(async_add_one)      # async: 6 (transitions to _run_async)
        .config(debug=True)
        .run()
      )
      await self.assertEqual(result, 6)
      # Root (2) and first sync link (5) are logged before async transition
      IsolatedAsyncioTestCase.assertGreaterEqual(self, len(logs), 2)
      IsolatedAsyncioTestCase.assertTrue(self, any('2' in log for log in logs))
      IsolatedAsyncioTestCase.assertTrue(self, any('5' in log for log in logs))
    finally:
      logger.removeHandler(handler)


if __name__ == '__main__':
  import unittest
  unittest.main()
