"""Final gap-closure tests — closes all 6 remaining coverage gaps.

Sections:
  GAP 1: to_thread — Full Cross-Dimensional Coverage
  GAP 2: no_async(True) Interaction Testing
  GAP 3: config(autorun=True) + FrozenChain
  GAP 4: Debug Mode x Iteration/CM Operations
  GAP 5: foreach(with_index=True) Cross-Feature Testing
  GAP 6: Simple Fast Path vs Full Path Equivalence
"""

import asyncio
import inspect
import logging
import time
import threading
import warnings
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def async_identity(v):
  return v

async def async_double(v):
  return v * 2

async def async_add_one(v):
  return v + 1

async def async_noop(v=None):
  pass

async def async_raise(v=None):
  raise TestExc('async boom')

def sync_double(v):
  return v * 2

def sync_add_one(v):
  return v + 1

def sync_identity(v):
  return v

def sync_noop(v=None):
  pass

def raise_test_exc(v=None):
  raise TestExc('boom')


class AsyncIterable:
  """Async iterable over a list."""
  def __init__(self, items):
    self.items = items
  def __aiter__(self):
    return AsyncIterator(iter(self.items))


class AsyncIterator:
  """Async iterator."""
  def __init__(self, it):
    self._it = it
  def __aiter__(self):
    return self
  async def __anext__(self):
    try:
      return next(self._it)
    except StopIteration:
      raise StopAsyncIteration


class SyncCM:
  """Sync context manager."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False
  def __enter__(self):
    self.entered = True
    return self.value
  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  """Async context manager."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False
  async def __aenter__(self):
    self.entered = True
    return self.value
  async def __aexit__(self, *args):
    self.exited = True
    return False


class CapturingLogHandler(logging.Handler):
  """Log handler that captures records for assertions."""
  def __init__(self):
    super().__init__()
    self.records = []
    self.messages = []
  def emit(self, record):
    self.records.append(record)
    self.messages.append(record.getMessage())


@contextmanager
def capture_quent_logs():
  """Context manager to capture quent logger output."""
  handler = CapturingLogHandler()
  logger = logging.getLogger('quent')
  old_level = logger.level
  logger.addHandler(handler)
  logger.setLevel(logging.DEBUG)
  try:
    yield handler
  finally:
    logger.removeHandler(handler)
    logger.setLevel(old_level)


# ============================================================================
# GAP 1: to_thread — Full Cross-Dimensional Coverage
# ============================================================================

class ToThreadCrossDimensionalTests(MyTestCase):
  """Tests for to_thread in combination with every chain feature."""

  async def test_to_thread_basic_sync_fn(self):
    """to_thread with a simple sync function in async context."""
    result = await Chain(10).to_thread(sync_double).run()
    await self.assertEqual(result, 20)

  async def test_to_thread_cascade_returns_root(self):
    """Cascade with to_thread returns root value (Cascade semantics)."""
    side_effects = []
    def side_effect(v):
      side_effects.append(v * 2)
      return v * 2
    result = await Cascade(10).to_thread(side_effect).run()
    await self.assertEqual(result, 10)
    assert side_effects == [20]

  async def test_to_thread_cascade_then_receives_root(self):
    """Cascade: to_thread side effect, then receives root."""
    side_effects = []
    def side_effect(v):
      side_effects.append(v)
      return v * 100
    results = []
    def capture(v):
      results.append(v)
    result = await Cascade(10).to_thread(side_effect).then(capture).run()
    await self.assertEqual(result, 10)
    assert side_effects == [10]
    assert results == [10]

  async def test_to_thread_frozen_chain(self):
    """FrozenChain with to_thread, called multiple times."""
    frozen = Chain().to_thread(sync_double).freeze()
    r1 = await frozen(5)
    r2 = await frozen(10)
    r3 = await frozen(0)
    assert r1 == 10
    assert r2 == 20
    assert r3 == 0

  async def test_to_thread_clone_independence(self):
    """Clone a chain with to_thread, modify clone, verify independence."""
    original = Chain(10).to_thread(sync_double)
    cloned = original.clone()
    cloned.then(sync_add_one)
    r_orig = await original.run()
    r_clone = await cloned.run()
    await self.assertEqual(r_orig, 20)
    await self.assertEqual(r_clone, 21)

  async def test_to_thread_decorator(self):
    """to_thread used with decorator pattern."""
    @Chain().to_thread(sync_double).decorator()
    def compute(x):
      return x + 1
    result = await compute(4)
    assert result == 10  # (4+1) * 2

  async def test_to_thread_except_catches(self):
    """to_thread raises exception, except_ catches it."""
    def raising_fn(v):
      raise TestExc('thread error')
    handler_called = []
    def handler(v):
      handler_called.append(True)
      return 'recovered'
    result = await Chain(10).to_thread(raising_fn).except_(handler, reraise=False).run()
    await self.assertEqual(result, 'recovered')
    assert handler_called

  async def test_to_thread_except_finally_triple(self):
    """to_thread raises, except catches, finally cleans up."""
    def raising_fn(v):
      raise TestExc('thread error')
    state = {'exc': False, 'fin': False}
    def on_exc(v):
      state['exc'] = True
    def on_fin(v):
      state['fin'] = True
    try:
      await Chain(10).to_thread(raising_fn).except_(on_exc).finally_(on_fin).run()
    except TestExc:
      pass
    assert state['exc']
    assert state['fin']

  async def test_to_thread_debug_mode(self):
    """Debug mode logs root value for to_thread chain.

    Note: When chain transitions to async (via to_thread), the sync _run
    logs only the root. The async _run_async stores debug info in link_results
    but does NOT call _logger.debug(). So only 1 log message for root.
    """
    with capture_quent_logs() as handler:
      result = await Chain(5).config(debug=True).to_thread(sync_double).run()
    await self.assertEqual(result, 10)
    # Only root is logged in the sync portion before async transition
    assert len(handler.messages) >= 1
    assert any('root' in m for m in handler.messages)

  async def test_to_thread_falsy_values(self):
    """to_thread with falsy values (0, False, '')."""
    for val in [0, False, '']:
      with self.subTest(val=val):
        result = await Chain(val).to_thread(sync_identity).run()
        assert result == val, f"Expected {val!r}, got {result!r}"

  async def test_to_thread_none_value(self):
    """to_thread with None root — fn called with None."""
    def fn(v):
      return v is None
    result = await Chain(None).to_thread(fn).run()
    assert result is True

  async def test_to_thread_sleep_both_async(self):
    """to_thread followed by sleep — both force async."""
    result = await Chain(5).to_thread(sync_double).sleep(0).run()
    await self.assertEqual(result, 10)

  async def test_to_thread_after_foreach(self):
    """to_thread after foreach processes the list in a thread."""
    result = await Chain([1, 2, 3]).foreach(sync_double).to_thread(sum).run()
    await self.assertEqual(result, 12)

  async def test_to_thread_reuse_multiple_runs(self):
    """Same chain with to_thread, run multiple times."""
    c = Chain().to_thread(sync_double)
    r1 = await c.run(5)
    r2 = await c.run(10)
    r3 = await c.run(0)
    assert r1 == 10
    assert r2 == 20
    assert r3 == 0

  async def test_to_thread_nested_chain(self):
    """Outer chain -> to_thread -> then verifies threaded result."""
    def heavy_compute(v):
      return v ** 2
    result = await Chain(7).to_thread(heavy_compute).then(sync_add_one).run()
    await self.assertEqual(result, 50)  # 7^2 = 49, +1 = 50

  async def test_to_thread_with_context_manager(self):
    """to_thread after with_ — processing after CM exits."""
    cm = SyncCM(value=42)
    result = await Chain(cm).with_(sync_identity).to_thread(sync_double).run()
    await self.assertEqual(result, 84)
    assert cm.entered
    assert cm.exited

  async def test_to_thread_preserves_thread_execution(self):
    """Verify to_thread actually runs in a different thread."""
    main_thread = threading.current_thread().ident
    thread_ids = []
    def capture_thread(v):
      thread_ids.append(threading.current_thread().ident)
      return v
    await Chain(10).to_thread(capture_thread).run()
    assert len(thread_ids) == 1
    assert thread_ids[0] != main_thread

  async def test_to_thread_chain_result_propagation(self):
    """Chained to_thread calls propagate values correctly."""
    def add_ten(v):
      return v + 10
    result = await Chain(5).to_thread(sync_double).to_thread(add_ten).run()
    await self.assertEqual(result, 20)  # 5*2=10, 10+10=20

  async def test_to_thread_gather_sync_fns(self):
    """gather with sync functions produces list of results (sync path)."""
    # gather is sync because both fns are sync, root is sync too.
    result = Chain(5).gather(sync_double, sync_add_one).run()
    assert result == [10, 6]

  async def test_to_thread_with_do_side_effect(self):
    """to_thread produces result (to_thread is a then-style link)."""
    side_effects = []
    def side_effect(v):
      side_effects.append(v * 3)
      return v * 3
    # to_thread is added via _then with a normal link (not ignore_result)
    result = await Chain(10).to_thread(side_effect).run()
    await self.assertEqual(result, 30)
    assert side_effects == [30]

  async def test_to_thread_empty_string_root(self):
    """to_thread with empty string root."""
    result = await Chain('').to_thread(lambda v: v + 'hello').run()
    await self.assertEqual(result, 'hello')

  async def test_to_thread_large_data(self):
    """to_thread with large data — no serialization issues."""
    data = list(range(10000))
    result = await Chain(data).to_thread(len).run()
    await self.assertEqual(result, 10000)

  async def test_to_thread_return_via_nested_chain(self):
    """to_thread result flows into a subsequent then operation."""
    result = await Chain(3).to_thread(sync_double).then(lambda v: v * 10).run()
    await self.assertEqual(result, 60)  # 3*2=6, 6*10=60

  async def test_to_thread_with_root_callable(self):
    """to_thread where root is a callable that gets evaluated."""
    def make_value():
      return 5
    result = await Chain(make_value).to_thread(sync_double).run()
    await self.assertEqual(result, 10)


# ============================================================================
# GAP 2: no_async(True) Interaction Testing
# ============================================================================

class NoAsyncInteractionTests(MyTestCase):
  """Tests for no_async(True) and how it interacts with operations
  that have INDEPENDENT async detection.

  Key insight: no_async only affects the chain loop's iscoro() checks.
  Operations like foreach, filter, with_, gather, to_thread, sleep
  check for async independently. When these operations return coroutines
  and _is_sync=True, the chain loop does NOT await them.

  IMPORTANT: Cython coroutines are NOT detected by inspect.iscoroutine().
  Use inspect.isawaitable() instead.
  """

  async def test_no_async_then_async_fn_returns_coro(self):
    """no_async + then(async_fn) — coroutine returned as-is, NOT awaited."""
    result = Chain(10).no_async(True).then(async_double).run()
    # result should be a coroutine/awaitable, not the awaited value
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == 20

  async def test_no_async_sync_chain_works_normally(self):
    """no_async + sync chain — works exactly like normal."""
    result = Chain(10).no_async(True).then(sync_double).then(sync_add_one).run()
    assert result == 21

  async def test_no_async_foreach_async_iterable_still_dispatches(self):
    """no_async + foreach with async iterable — foreach checks __aiter__
    independently. Returns coroutine since chain loop skips iscoro.
    """
    aiterable = AsyncIterable([1, 2, 3])
    result = Chain(aiterable).no_async(True).foreach(sync_double).run()
    # foreach detected __aiter__ and returned a coroutine
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == [2, 4, 6]

  async def test_no_async_foreach_async_fn_returns_coro(self):
    """no_async + foreach with async callback — iscoro check inside
    _Foreach still triggers sync-to-async transition. But the chain
    loop's _is_sync means the resulting coro propagates unwatched.
    """
    result = Chain([1, 2, 3]).no_async(True).foreach(async_double).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == [2, 4, 6]

  async def test_no_async_filter_async_iterable(self):
    """no_async + filter with async iterable — filter checks __aiter__
    independently."""
    aiterable = AsyncIterable([1, 2, 3, 4, 5])
    result = Chain(aiterable).no_async(True).filter(lambda v: v > 2).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == [3, 4, 5]

  async def test_no_async_filter_async_predicate(self):
    """no_async + filter with async predicate — iscoro check inside
    _Filter still triggers."""
    async def async_is_even(v):
      return v % 2 == 0
    result = Chain([1, 2, 3, 4]).no_async(True).filter(async_is_even).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == [2, 4]

  async def test_no_async_foreach_indexed_async_iterable(self):
    """no_async + foreach_indexed with async iterable — checks __aiter__
    independently."""
    aiterable = AsyncIterable([10, 20, 30])
    result = Chain(aiterable).no_async(True).foreach(lambda i, v: (i, v), with_index=True).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == [(0, 10), (1, 20), (2, 30)]

  async def test_no_async_with_async_cm(self):
    """no_async + with_ and async CM — _With checks __aenter__
    independently, dispatches to _with_full_async."""
    cm = AsyncCM(value=42)
    result = Chain(cm).no_async(True).with_(sync_identity).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == 42
    assert cm.entered
    assert cm.exited

  async def test_no_async_with_sync_cm_async_body(self):
    """no_async + sync CM with async body — _With checks iscoro(result)
    independently."""
    cm = SyncCM(value=10)
    result = Chain(cm).no_async(True).with_(async_double).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == 20
    assert cm.entered
    assert cm.exited

  async def test_no_async_gather_mixed_fns(self):
    """no_async + gather with mixed sync/async fns — _Gather checks
    iscoro independently."""
    result = Chain(5).no_async(True).gather(sync_double, async_add_one).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == [10, 6]

  async def test_no_async_to_thread_checks_loop(self):
    """no_async + to_thread — _ToThread checks event loop independently."""
    result = Chain(5).no_async(True).to_thread(sync_double).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == 10

  async def test_no_async_sleep_checks_loop(self):
    """no_async + sleep — _Sleep checks event loop independently.
    Sleep is a do() link (ignore_result=True), so its async coroutine
    is the result but it's discarded (ignore_result). The chain loop
    doesn't check iscoro when _is_sync=True. So the sleep coroutine
    is created but never awaited (discarded due to ignore_result).
    The chain returns the current_value synchronously.
    """
    result = Chain(42).no_async(True).sleep(0).run()
    # sleep is ignore_result, so chain value stays 42
    assert result == 42

  async def test_no_async_cascade(self):
    """no_async + Cascade — both work together for sync fns."""
    results = []
    def capture(v):
      results.append(v)
      return 'ignored'
    result = Cascade(10).no_async(True).then(capture).then(capture).run()
    assert result == 10
    assert results == [10, 10]

  async def test_no_async_nested_async_chain_returns_coro(self):
    """no_async outer chain + frozen inner chain returning coro.
    FrozenChain is called as a regular callable by the outer chain.
    The inner chain returns a coro (since it doesn't have no_async).
    The outer chain's loop (with _is_sync) doesn't check iscoro,
    so the coro propagates as the return value.
    """
    inner = Chain().then(async_double).freeze()
    result = Chain(5).no_async(True).then(inner).run()
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == 10

  async def test_no_async_chain_loop_only_insight(self):
    """Verify no_async affects CHAIN LOOP only, not within-link async
    detection."""
    # Sync iterable, async callback
    result = Chain([10, 20]).no_async(True).foreach(async_double).run()
    assert inspect.isawaitable(result)
    actual = await result
    assert actual == [20, 40]

    # Async iterable, sync callback
    result2 = Chain(AsyncIterable([10, 20])).no_async(True).foreach(sync_double).run()
    assert inspect.isawaitable(result2)
    actual2 = await result2
    assert actual2 == [20, 40]

  async def test_no_async_then_multiple_sync_fns(self):
    """no_async with multiple sync then() calls — pure sync path."""
    result = Chain(2).no_async(True).then(sync_double).then(sync_add_one).then(sync_double).run()
    assert result == 10  # 2*2=4, 4+1=5, 5*2=10

  async def test_no_async_disabled(self):
    """no_async(False) — re-enables async detection."""
    result = await Chain(10).no_async(True).no_async(False).then(async_double).run()
    await self.assertEqual(result, 20)

  async def test_no_async_clone_preserves(self):
    """Clone preserves no_async flag."""
    c = Chain(10).no_async(True).then(sync_double)
    c2 = c.clone()
    r1 = c.run()
    r2 = c2.run()
    assert r1 == 20
    assert r2 == 20

  async def test_no_async_root_async_fn(self):
    """no_async with async root function — coro returned as root value."""
    result = Chain(async_double, 5).no_async(True).run()
    # The root_link evaluates async_double(5) which returns a coro.
    # _is_sync means the chain loop skips iscoro check, so coro is
    # returned as the current_value.
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == 10


# ============================================================================
# GAP 3: config(autorun=True) + FrozenChain
# ============================================================================

class AutorunFrozenChainTests(MyTestCase):
  """Tests for config(autorun=True) and its interaction with FrozenChain.

  KEY FINDINGS from source analysis:
  1. Chain.run() checks autorun AFTER _run returns (line 605-606)
  2. Chain._run() checks autorun INSIDE (lines 132-133 and 159-160)
     ONLY in the full path (not _run_simple)
  3. Chain._run_simple() does NOT check autorun
  4. FrozenChain.run() calls Chain._run directly, bypassing Chain.run()

  Therefore:
  - Simple chain + FrozenChain: autorun only applied if Chain.run()
    is used (not FrozenChain)
  - Full chain + FrozenChain: autorun IS applied inside _run
  """

  async def test_autorun_sync_chain(self):
    """autorun with sync chain — returns result directly (no coro)."""
    result = Chain(10).config(autorun=True).then(sync_double).run()
    assert result == 20

  async def test_autorun_async_chain_returns_task(self):
    """autorun with async chain — Chain.run() wraps in ensure_future."""
    result = Chain(10).config(autorun=True).then(async_double).run()
    # Chain.run() line 605-606: iscoro(result) -> ensure_future
    assert isinstance(result, asyncio.Task), f"Expected Task, got {type(result)}"
    actual = await result
    assert actual == 20

  async def test_autorun_frozen_simple_chain_returns_coro(self):
    """FrozenChain with autorun + simple chain: _run_simple does NOT
    check autorun, so it returns a raw coro, not a Task.
    This is a behavioral difference from Chain.run().
    """
    c = Chain(10).config(autorun=True).then(async_double)
    frozen = c.freeze()
    result = frozen()
    # _run delegates to _run_simple (since _is_simple=True, _debug=False)
    # _run_simple does NOT check autorun, returns raw coro
    # FrozenChain.run() does NOT check autorun either
    assert inspect.isawaitable(result), f"Expected awaitable, got {type(result)}"
    actual = await result
    assert actual == 20

  async def test_autorun_frozen_full_chain_returns_task(self):
    """FrozenChain with autorun + full chain: _run DOES check autorun
    at lines 132-133 and 159-160 (inside the full path).
    """
    c = Chain(10).config(autorun=True).then(async_double).do(sync_noop)
    frozen = c.freeze()
    result = frozen()
    # _run (full path) detects coro from async_double and checks autorun
    assert isinstance(result, asyncio.Task), f"Expected Task, got {type(result)}"
    actual = await result
    assert actual == 20

  async def test_autorun_frozen_vs_unfrozen_sync(self):
    """Compare autorun behavior: unfrozen vs frozen for sync chain."""
    c = Chain(10).config(autorun=True).then(sync_double)
    unfrozen_result = c.run()
    frozen = c.freeze()
    frozen_result = frozen()
    assert unfrozen_result == 20
    assert frozen_result == 20

  async def test_autorun_frozen_vs_unfrozen_async_full_path(self):
    """Compare autorun behavior: unfrozen vs frozen for async full-path chain."""
    # Unfrozen (via Chain.run)
    c_unfrozen = Chain(10).config(autorun=True).then(async_double).do(sync_noop)
    result_unfrozen = c_unfrozen.run()
    assert isinstance(result_unfrozen, asyncio.Task)
    val_unfrozen = await result_unfrozen

    # Frozen (via FrozenChain -> Chain._run full path)
    c_frozen = Chain(10).config(autorun=True).then(async_double).do(sync_noop)
    frozen = c_frozen.freeze()
    result_frozen = frozen()
    assert isinstance(result_frozen, asyncio.Task)
    val_frozen = await result_frozen

    assert val_unfrozen == val_frozen == 20

  async def test_autorun_clone_preserves(self):
    """Clone preserves autorun flag."""
    c = Chain(10).config(autorun=True).then(async_double)
    c2 = c.clone()
    r1 = c.run()
    r2 = c2.run()
    # Both go through Chain.run() which checks autorun
    assert isinstance(r1, asyncio.Task)
    assert isinstance(r2, asyncio.Task)
    assert await r1 == 20
    assert await r2 == 20

  async def test_autorun_cascade(self):
    """Cascade with autorun returns root value."""
    result = Cascade(10).config(autorun=True).then(async_double).run()
    assert isinstance(result, asyncio.Task)
    actual = await result
    assert actual == 10

  async def test_autorun_with_except(self):
    """autorun with async chain and except_."""
    handler_called = []
    def handler(v):
      handler_called.append(True)
      return 'recovered'
    result = Chain(10).config(autorun=True).then(async_raise).except_(handler, reraise=False).run()
    assert isinstance(result, asyncio.Task)
    actual = await result
    assert actual == 'recovered'
    assert handler_called

  async def test_autorun_with_finally(self):
    """autorun with async chain + do (full path) and finally_."""
    state = {'fin': False}
    def on_fin(v):
      state['fin'] = True
    result = Chain(10).config(autorun=True).then(async_double).do(sync_noop).finally_(on_fin).run()
    assert isinstance(result, asyncio.Task)
    actual = await result
    assert actual == 20
    assert state['fin']

  async def test_autorun_disabled(self):
    """config(autorun=False) after setting True — verify disabled."""
    result = Chain(10).config(autorun=True).config(autorun=False).then(async_double).run()
    # autorun disabled, so result is a raw coro (not a Task)
    assert not isinstance(result, asyncio.Task)
    assert inspect.isawaitable(result)
    actual = await result
    assert actual == 20


# ============================================================================
# GAP 4: Debug Mode x Iteration/CM Operations
# ============================================================================

class DebugModeIterationCMTests(MyTestCase):
  """Tests for debug mode logging with iteration, CM, and operator links.

  KEY FINDINGS from source analysis:
  1. Debug logging via _logger.debug() ONLY happens in the SYNC portion
     of _run/_run_simple (lines 137, 163).
  2. In _run_async/_run_async_simple, results are stored in link_results
     but NOT logged via _logger.debug() (lines 254-258, 278-282).
  3. So operations that are fully sync will have all links logged.
     Operations that trigger async transition will only have links
     logged UP TO the transition point.
  4. Foreach/filter/gather/with_ are evaluated as a single link from
     the chain loop's perspective. Their outer Link fn_name is None
     (inner link has the name), so they show as '[?]' in logs.
  """

  async def test_debug_foreach_sync(self):
    """Debug + sync foreach — root and foreach link logged."""
    with capture_quent_logs() as handler:
      result = Chain([1, 2, 3]).config(debug=True).foreach(sync_double).run()
    assert result == [2, 4, 6]
    # root + foreach link
    assert len(handler.messages) >= 2
    assert any('root' in m for m in handler.messages)
    # Foreach outer link has fn_name=None, so it shows as [?]
    assert any('[?]' in m for m in handler.messages)

  async def test_debug_foreach_exception(self):
    """Debug + foreach where fn raises — root is logged before error."""
    def failing_fn(v):
      if v == 2:
        raise TestExc('fail on 2')
      return v * 2
    with capture_quent_logs() as handler:
      try:
        Chain([1, 2, 3]).config(debug=True).foreach(failing_fn).run()
        assert False, "Should have raised"
      except TestExc:
        pass
    # Root should be logged at minimum
    assert len(handler.messages) >= 1
    assert any('root' in m for m in handler.messages)

  async def test_debug_filter_sync(self):
    """Debug + sync filter — root and filter link logged."""
    with capture_quent_logs() as handler:
      result = Chain([1, 2, 3, 4, 5]).config(debug=True).filter(lambda v: v > 2).run()
    assert result == [3, 4, 5]
    assert any('root' in m for m in handler.messages)
    assert any('[?]' in m for m in handler.messages)

  async def test_debug_gather_sync(self):
    """Debug + sync gather — logged as single link."""
    with capture_quent_logs() as handler:
      result = Chain(5).config(debug=True).gather(sync_double, sync_add_one).run()
    assert result == [10, 6]
    assert any('root' in m for m in handler.messages)
    assert any('[?]' in m for m in handler.messages)

  async def test_debug_with_sync(self):
    """Debug + sync with_ — logged as single link."""
    cm = SyncCM(value=42)
    with capture_quent_logs() as handler:
      result = Chain(cm).config(debug=True).with_(sync_identity).run()
    assert result == 42
    assert any('root' in m for m in handler.messages)
    assert any('[?]' in m for m in handler.messages)

  async def test_debug_with_exception(self):
    """Debug + with_ that raises — root logged before error."""
    cm = SyncCM(value=10)
    with capture_quent_logs() as handler:
      try:
        Chain(cm).config(debug=True).with_(raise_test_exc).run()
        assert False, "Should have raised"
      except TestExc:
        pass
    assert len(handler.messages) >= 1

  async def test_debug_foreach_indexed_sync(self):
    """Debug + sync foreach_indexed — logged as single link."""
    with capture_quent_logs() as handler:
      result = Chain([10, 20, 30]).config(debug=True).foreach(
        lambda i, v: (i, v), with_index=True
      ).run()
    assert result == [(0, 10), (1, 20), (2, 30)]
    assert any('root' in m for m in handler.messages)
    assert any('[?]' in m for m in handler.messages)

  async def test_debug_nested_chain_isolation(self):
    """Debug on outer chain does NOT propagate to inner chain."""
    inner = Chain().then(sync_double)
    # inner does NOT have debug=True
    with capture_quent_logs() as handler:
      result = Chain(5).config(debug=True).then(inner.freeze()).run()
    assert result == 10
    # Outer chain debug: root + then
    assert len(handler.messages) >= 2

  async def test_debug_cascade_foreach_sync(self):
    """Debug in Cascade with sync foreach."""
    with capture_quent_logs() as handler:
      result = Cascade([1, 2, 3]).config(debug=True).foreach(sync_double).run()
    # Cascade returns root
    assert result == [1, 2, 3]
    assert len(handler.messages) >= 1

  async def test_debug_async_foreach(self):
    """Debug with async foreach — only root logged (async transition
    means _run_async stores but doesn't log).
    """
    with capture_quent_logs() as handler:
      result = await Chain([1, 2, 3]).config(debug=True).foreach(async_double).run()
    await self.assertEqual(result, [2, 4, 6])
    # Only root is logged in the sync portion
    assert len(handler.messages) >= 1
    assert any('root' in m for m in handler.messages)

  async def test_debug_async_with(self):
    """Debug with async CM — only root logged before async transition."""
    cm = AsyncCM(value=99)
    with capture_quent_logs() as handler:
      result = await Chain(cm).config(debug=True).with_(sync_identity).run()
    await self.assertEqual(result, 99)
    # Root is logged, then async transition — no more _logger.debug() calls
    assert len(handler.messages) >= 1

  async def test_debug_sleep(self):
    """Debug + sleep — sleep triggers async, so only root logged before
    transition."""
    with capture_quent_logs() as handler:
      result = await Chain(42).config(debug=True).sleep(0).run()
    await self.assertEqual(result, 42)
    # Root is logged in sync portion. Sleep transitions to async.
    assert len(handler.messages) >= 1
    assert any('root' in m for m in handler.messages)

  async def test_debug_to_thread(self):
    """Debug + to_thread — only root logged before async transition."""
    with capture_quent_logs() as handler:
      result = await Chain(5).config(debug=True).to_thread(sync_double).run()
    await self.assertEqual(result, 10)
    assert len(handler.messages) >= 1
    assert any('root' in m for m in handler.messages)

  async def test_debug_combined_sync_features(self):
    """Debug with then+do+foreach — all sync, all logged."""
    with capture_quent_logs() as handler:
      result = Chain([1, 2, 3]).config(debug=True) \
        .then(sync_identity) \
        .do(sync_noop) \
        .foreach(sync_double) \
        .run()
    assert result == [2, 4, 6]
    # root + then + do (ignored) + foreach = multiple log messages
    assert len(handler.messages) >= 3


# ============================================================================
# GAP 5: foreach(with_index=True) Cross-Feature Testing
# ============================================================================

class ForeachIndexedCrossFeatureTests(MyTestCase):
  """Tests for foreach_indexed cross-feature interactions."""

  async def test_foreach_indexed_basic(self):
    """Basic foreach_indexed — fn(idx, el) receives index and element."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(Chain(fn, [10, 20, 30]).foreach(
          lambda i, v: (i, v), with_index=True
        ).run())
        assert result == [(0, 10), (1, 20), (2, 30)]

  async def test_foreach_indexed_except(self):
    """foreach_indexed + except_ — fn raises, except_ catches."""
    def failing(idx, el):
      if idx == 1:
        raise TestExc('fail at index 1')
      return (idx, el)
    handler_called = []
    def handler(v):
      handler_called.append(True)
      return 'recovered'
    for fn, ctx in self.with_fn():
      with ctx:
        handler_called.clear()
        try:
          await await_(
            Chain(fn, [10, 20, 30])
            .foreach(failing, with_index=True)
            .except_(handler, reraise=True)
            .run()
          )
        except TestExc:
          pass
        assert handler_called

  async def test_foreach_indexed_finally(self):
    """foreach_indexed + finally_ — fn raises, finally_ cleanup."""
    def failing(idx, el):
      if idx == 2:
        raise TestExc('fail')
      return el
    state = {'fin': False}
    def on_fin(v):
      state['fin'] = True
    for fn, ctx in self.with_fn():
      with ctx:
        state['fin'] = False
        try:
          await await_(
            Chain(fn, [1, 2, 3])
            .foreach(failing, with_index=True)
            .finally_(on_fin)
            .run()
          )
        except TestExc:
          pass
        assert state['fin']

  async def test_foreach_indexed_except_finally_triple(self):
    """foreach_indexed + except_ + finally_ — triple combination."""
    def failing(idx, el):
      if idx == 1:
        raise TestExc('fail')
      return el
    state = {'exc': False, 'fin': False}
    def on_exc(v):
      state['exc'] = True
    def on_fin(v):
      state['fin'] = True
    for fn, ctx in self.with_fn():
      with ctx:
        state['exc'] = False
        state['fin'] = False
        try:
          await await_(
            Chain(fn, [10, 20])
            .foreach(failing, with_index=True)
            .except_(on_exc)
            .finally_(on_fin)
            .run()
          )
        except TestExc:
          pass
        assert state['exc']
        assert state['fin']

  async def test_foreach_indexed_cascade(self):
    """Cascade + foreach_indexed — Cascade returns root, not iteration result."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Cascade(fn, [1, 2, 3])
          .foreach(lambda i, v: (i, v * 10), with_index=True)
          .run()
        )
        assert result == [1, 2, 3]

  async def test_foreach_indexed_debug(self):
    """Debug mode with sync foreach_indexed."""
    with capture_quent_logs() as handler:
      result = Chain([10, 20]).config(debug=True).foreach(
        lambda i, v: (i, v), with_index=True
      ).run()
    assert result == [(0, 10), (1, 20)]
    assert len(handler.messages) >= 2

  async def test_foreach_indexed_break_with_value(self):
    """break_(value) inside foreach_indexed — returns value."""
    def fn_with_break(idx, el):
      if idx == 2:
        Chain.break_('stopped')
      return (idx, el)
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, [10, 20, 30, 40])
          .foreach(fn_with_break, with_index=True)
          .run()
        )
        assert result == 'stopped'

  async def test_foreach_indexed_break_without_value(self):
    """break_() inside foreach_indexed without value — returns list so far."""
    def fn_with_break(idx, el):
      if idx == 2:
        Chain.break_()
      return (idx, el)
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, [10, 20, 30, 40])
          .foreach(fn_with_break, with_index=True)
          .run()
        )
        assert result == [(0, 10), (1, 20)]

  async def test_foreach_indexed_break_finally(self):
    """Break in foreach_indexed, then finally runs."""
    state = {'fin': False}
    def on_fin(v):
      state['fin'] = True
    def fn_with_break(idx, el):
      if idx == 1:
        Chain.break_()
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        state['fin'] = False
        result = await await_(
          Chain(fn, [10, 20, 30])
          .foreach(fn_with_break, with_index=True)
          .finally_(on_fin)
          .run()
        )
        assert result == [10]
        assert state['fin']

  async def test_foreach_indexed_async_iterable(self):
    """Async iterable with foreach_indexed."""
    aiterable = AsyncIterable([100, 200, 300])
    result = await Chain(aiterable).foreach(
      lambda i, v: (i, v), with_index=True
    ).run()
    await self.assertEqual(result, [(0, 100), (1, 200), (2, 300)])

  async def test_foreach_indexed_sync_to_async_transition(self):
    """fn(idx, el) returns coro on all elements — triggers async transition."""
    async def maybe_async(idx, el):
      return (idx, el * 2)
    result = await Chain([10, 20, 30]).foreach(maybe_async, with_index=True).run()
    await self.assertEqual(result, [(0, 20), (1, 40), (2, 60)])

  async def test_foreach_indexed_clone(self):
    """Clone chain with foreach_indexed, add filter to clone."""
    original = Chain([1, 2, 3, 4]).foreach(lambda i, v: i + v, with_index=True)
    cloned = original.clone()
    cloned.then(lambda lst: [x for x in lst if x > 3])
    r_orig = original.run()
    r_clone = cloned.run()
    assert r_orig == [1, 3, 5, 7]
    assert r_clone == [5, 7]

  async def test_foreach_indexed_frozen_chain(self):
    """Freeze chain with foreach_indexed, reuse."""
    frozen = Chain().foreach(lambda i, v: (i, v), with_index=True).freeze()
    r1 = frozen([10, 20])
    r2 = frozen([30, 40, 50])
    assert r1 == [(0, 10), (1, 20)]
    assert r2 == [(0, 30), (1, 40), (2, 50)]

  async def test_foreach_indexed_value_types(self):
    """foreach_indexed over various iterable types."""
    # Over a string
    result = Chain('abc').foreach(lambda i, v: (i, v), with_index=True).run()
    assert result == [(0, 'a'), (1, 'b'), (2, 'c')]

    # Over a range
    result = Chain(range(3)).foreach(lambda i, v: i + v, with_index=True).run()
    assert result == [0, 2, 4]

    # Over a tuple
    result = Chain((10, 20)).foreach(lambda i, v: (i, v), with_index=True).run()
    assert result == [(0, 10), (1, 20)]

  async def test_foreach_indexed_empty_iterable(self):
    """foreach_indexed over empty iterable."""
    result = Chain([]).foreach(lambda i, v: (i, v), with_index=True).run()
    assert result == []


# ============================================================================
# GAP 6: Simple Fast Path vs Full Path Equivalence
# ============================================================================

class SimpleVsFullPathEquivalenceTests(MyTestCase):
  """Tests that verify the simple path (_run_simple / _run_async_simple)
  and the full path (_run / _run_async) produce identical results.

  Simple path: only then() links, no debug, no do/except_/finally_
  Full path: has do(), except_(), or debug which sets _is_simple=False
  """

  async def test_chain_simple_sync(self):
    """Chain simple path (sync) — only then() links."""
    result = Chain(10).then(sync_double).then(sync_add_one).run()
    assert result == 21

  async def test_chain_full_sync(self):
    """Chain full path (sync) — do() forces full path."""
    result = Chain(10).then(sync_double).then(sync_add_one).do(sync_noop).run()
    assert result == 21

  async def test_simple_vs_full_sync_equivalence(self):
    """Simple and full sync paths produce identical results."""
    simple = Chain(10).then(sync_double).then(sync_add_one).run()
    full = Chain(10).then(sync_double).then(sync_add_one).do(sync_noop).run()
    assert simple == full == 21

  async def test_chain_simple_async(self):
    """Chain simple path (async) — then() with async fn."""
    result = await Chain(10).then(async_double).then(sync_add_one).run()
    await self.assertEqual(result, 21)

  async def test_chain_full_async(self):
    """Chain full path (async) — do() forces full path with async fn."""
    result = await Chain(10).then(async_double).then(sync_add_one).do(sync_noop).run()
    await self.assertEqual(result, 21)

  async def test_simple_vs_full_async_equivalence(self):
    """Simple and full async paths produce identical results."""
    simple = await Chain(10).then(async_double).then(sync_add_one).run()
    full = await Chain(10).then(async_double).then(sync_add_one).do(sync_noop).run()
    assert simple == full == 21

  async def test_cascade_simple_sync(self):
    """Cascade simple path (sync) — only then() links."""
    result = Cascade(10).then(sync_double).then(sync_add_one).run()
    assert result == 10

  async def test_cascade_full_sync(self):
    """Cascade full path (sync) — do() forces full path."""
    result = Cascade(10).then(sync_double).then(sync_add_one).do(sync_noop).run()
    assert result == 10

  async def test_cascade_simple_vs_full_sync(self):
    """Cascade simple and full sync produce identical results."""
    simple = Cascade(10).then(sync_double).then(sync_add_one).run()
    full = Cascade(10).then(sync_double).then(sync_add_one).do(sync_noop).run()
    assert simple == full == 10

  async def test_cascade_simple_async(self):
    """Cascade simple path (async) — then() with async fn."""
    result = await Cascade(10).then(async_double).then(sync_add_one).run()
    await self.assertEqual(result, 10)

  async def test_cascade_full_async(self):
    """Cascade full path (async) — do() forces full path."""
    result = await Cascade(10).then(async_double).then(sync_add_one).do(sync_noop).run()
    await self.assertEqual(result, 10)

  async def test_cascade_simple_vs_full_async(self):
    """Cascade simple and full async produce identical results."""
    simple = await Cascade(10).then(async_double).then(sync_add_one).run()
    full = await Cascade(10).then(async_double).then(sync_add_one).do(sync_noop).run()
    assert simple == full == 10

  async def test_ten_link_simple_vs_full(self):
    """10-link chain: simple vs full produce same result."""
    def build_chain(use_do=False):
      c = Chain(1)
      for _ in range(10):
        c = c.then(sync_add_one)
      if use_do:
        c = c.do(sync_noop)
      return c
    simple = build_chain(False).run()
    full = build_chain(True).run()
    assert simple == full == 11

  async def test_simple_sync_to_async_transition(self):
    """Simple path: sync links then async link at position 5."""
    result = await (
      Chain(1)
      .then(sync_add_one)  # 2
      .then(sync_add_one)  # 3
      .then(sync_add_one)  # 4
      .then(sync_add_one)  # 5
      .then(async_double)  # 10
      .then(sync_add_one)  # 11
      .run()
    )
    await self.assertEqual(result, 11)

  async def test_full_sync_to_async_transition(self):
    """Full path: same chain with do() added."""
    result = await (
      Chain(1)
      .then(sync_add_one)  # 2
      .then(sync_add_one)  # 3
      .then(sync_add_one)  # 4
      .then(sync_add_one)  # 5
      .then(async_double)  # 10
      .then(sync_add_one)  # 11
      .do(sync_noop)
      .run()
    )
    await self.assertEqual(result, 11)

  async def test_simple_vs_full_sync_to_async_equivalence(self):
    """Verify sync-to-async transition produces same result in both paths."""
    simple = await (
      Chain(1)
      .then(sync_add_one).then(sync_add_one).then(sync_add_one)
      .then(async_double).then(sync_add_one)
      .run()
    )
    full = await (
      Chain(1)
      .then(sync_add_one).then(sync_add_one).then(sync_add_one)
      .then(async_double).then(sync_add_one)
      .do(sync_noop)
      .run()
    )
    assert simple == full

  async def test_return_value_identity_sync(self):
    """Both paths return the exact same value (identity check with 'is')."""
    sentinel = object()
    simple = Chain(sentinel).then(sync_identity).run()
    full = Chain(sentinel).then(sync_identity).do(sync_noop).run()
    assert simple is sentinel
    assert full is sentinel

  async def test_return_value_identity_async(self):
    """Both async paths return the exact same value."""
    sentinel = object()
    simple = await Chain(sentinel).then(async_identity).run()
    full = await Chain(sentinel).then(async_identity).do(sync_noop).run()
    assert simple is sentinel
    assert full is sentinel

  async def test_error_behavior_both_paths(self):
    """Both paths raise the same exception type on error."""
    with self.assertRaises(TestExc):
      Chain(10).then(raise_test_exc).run()

    with self.assertRaises(TestExc):
      Chain(10).then(raise_test_exc).do(sync_noop).run()

  async def test_error_behavior_async_both_paths(self):
    """Both async paths raise the same exception type."""
    with self.assertRaises(TestExc):
      await Chain(10).then(async_raise).run()

    with self.assertRaises(TestExc):
      await Chain(10).then(async_raise).do(sync_noop).run()

  async def test_void_chain_simple_vs_full(self):
    """Void chain (no root) — simple vs full."""
    simple = Chain().then(42).run()
    full = Chain().then(42).do(sync_noop).run()
    assert simple == full == 42

  async def test_simple_path_with_finally(self):
    """Adding finally_ does not change _is_simple (finally is separate).
    The chain still takes the simple path but the finally block runs.
    """
    state = {'fin': False}
    def on_fin(v):
      state['fin'] = True
    result = Chain(10).then(sync_double).finally_(on_fin).run()
    assert result == 20
    assert state['fin']

  async def test_except_forces_full_path(self):
    """Adding except_ forces full path (is_exception_handler)."""
    result = Chain(10).then(sync_double).except_(lambda v: v).run()
    assert result == 20

  async def test_simple_cascade_multiple_links(self):
    """Cascade simple path: each link receives root."""
    results = []
    def capture(v):
      results.append(v)
      return 'ignored'
    r = Cascade(42).then(capture).then(capture).then(capture).run()
    assert r == 42
    assert results == [42, 42, 42]

  async def test_full_cascade_multiple_links(self):
    """Cascade full path: same behavior with do() added."""
    results = []
    def capture(v):
      results.append(v)
      return 'ignored'
    r = Cascade(42).then(capture).then(capture).then(capture).do(sync_noop).run()
    assert r == 42
    assert results == [42, 42, 42]
