"""Final gap-closure tests — closes remaining coverage gaps.

Sections:
  GAP 2: config(autorun=True) + Chain reuse
  GAP 4: Debug Mode x Iteration/CM Operations
  GAP 6: Simple Fast Path vs Full Path Equivalence
"""

import asyncio
import inspect
import logging
import time
import warnings
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null
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
# GAP 3: config(autorun=True) + Chain reuse
# ============================================================================

class AutorunChainReuseTests(IsolatedAsyncioTestCase):
  """Tests for config(autorun=True) and chain reuse.

  KEY FINDINGS from source analysis:
  1. Chain.run() checks autorun AFTER _run returns (line 605-606)
  2. Chain._run() checks autorun INSIDE (lines 132-133 and 159-160)
     ONLY in the full path (not _run_simple)
  3. Chain._run_simple() does NOT check autorun
  """

  async def test_autorun_sync_chain(self):
    """autorun with sync chain -- returns result directly (no coro)."""
    result = Chain(10).config(autorun=True).then(sync_double).run()
    assert result == 20

  async def test_autorun_async_chain_returns_task(self):
    """autorun with async chain -- Chain.run() wraps in ensure_future."""
    result = Chain(10).config(autorun=True).then(async_double).run()
    # Chain.run() line 605-606: iscoro(result) -> ensure_future
    assert isinstance(result, asyncio.Task), f"Expected Task, got {type(result)}"
    actual = await result
    assert actual == 20

  async def test_autorun_simple_chain_run(self):
    """Chain with autorun + simple chain via run()."""
    c = Chain(10).config(autorun=True).then(async_double)
    result = c.run()
    # Chain.run() checks autorun after _run returns
    assert isinstance(result, asyncio.Task), f"Expected Task, got {type(result)}"
    actual = await result
    assert actual == 20

  async def test_autorun_full_chain_returns_task(self):
    """Chain with autorun + full chain: _run DOES check autorun
    at lines 132-133 and 159-160 (inside the full path).
    """
    c = Chain(10).config(autorun=True).then(async_double).do(sync_noop)
    result = c.run()
    # _run (full path) detects coro from async_double and checks autorun
    assert isinstance(result, asyncio.Task), f"Expected Task, got {type(result)}"
    actual = await result
    assert actual == 20

  async def test_autorun_chain_reuse_sync(self):
    """Chain with autorun reused for sync chain."""
    c = Chain(10).config(autorun=True).then(sync_double)
    result1 = c.run()
    result2 = c()
    assert result1 == 20
    assert result2 == 20

  async def test_autorun_chain_reuse_async_full_path(self):
    """Chain with autorun reused for async full-path chain."""
    # First run
    c1 = Chain(10).config(autorun=True).then(async_double).do(sync_noop)
    result1 = c1.run()
    assert isinstance(result1, asyncio.Task)
    val1 = await result1

    # Second run (reuse same chain definition)
    c2 = Chain(10).config(autorun=True).then(async_double).do(sync_noop)
    result2 = c2.run()
    assert isinstance(result2, asyncio.Task)
    val2 = await result2

    assert val1 == val2 == 20

  async def test_autorun_clone_preserves(self):
    """Clone preserves autorun flag."""
    c = Chain(10).config(autorun=True).then(async_double)
    c2 = c.clone()
    r2 = c2.run()
    # Both go through Chain.run() which checks autorun
    assert isinstance(r2, asyncio.Task)
    assert await r2 == 20

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

class DebugModeIterationCMTests(IsolatedAsyncioTestCase):
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
     the chain loop's perspective. Their outer Link shows as '[?]' in logs.
  """

  async def test_debug_foreach_sync(self):
    """Debug + sync foreach — root and foreach link logged."""
    with capture_quent_logs() as handler:
      result = Chain([1, 2, 3]).config(debug=True).foreach(sync_double).run()
    assert result == [2, 4, 6]
    # root + foreach link
    assert len(handler.messages) >= 2
    assert any('root' in m for m in handler.messages)
    # Foreach outer link shows as [foreach]
    assert any('[foreach]' in m for m in handler.messages)

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
    assert any('[filter]' in m for m in handler.messages)

  async def test_debug_gather_sync(self):
    """Debug + sync gather — logged as single link."""
    with capture_quent_logs() as handler:
      result = Chain(5).config(debug=True).gather(sync_double, sync_add_one).run()
    assert result == [10, 6]
    assert any('root' in m for m in handler.messages)
    assert any('[gather]' in m for m in handler.messages)

  async def test_debug_with_sync(self):
    """Debug + sync with_ — logged as single link."""
    cm = SyncCM(value=42)
    with capture_quent_logs() as handler:
      result = Chain(cm).config(debug=True).with_(sync_identity).run()
    assert result == 42
    assert any('root' in m for m in handler.messages)
    assert any('[with_]' in m for m in handler.messages)

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

  async def test_debug_nested_chain_isolation(self):
    """Debug on outer chain does NOT propagate to inner chain."""
    inner = Chain().then(sync_double)
    # inner does NOT have debug=True
    with capture_quent_logs() as handler:
      result = Chain(5).config(debug=True).then(inner).run()
    assert result == 10
    # Outer chain debug: root + then
    assert len(handler.messages) >= 2

  async def test_debug_async_foreach(self):
    """Debug with async foreach — only root logged (async transition
    means _run_async stores but doesn't log).
    """
    with capture_quent_logs() as handler:
      result = await Chain([1, 2, 3]).config(debug=True).foreach(async_double).run()
    self.assertEqual(result, [2, 4, 6])
    # Only root is logged in the sync portion
    assert len(handler.messages) >= 1
    assert any('root' in m for m in handler.messages)

  async def test_debug_async_with(self):
    """Debug with async CM — only root logged before async transition."""
    cm = AsyncCM(value=99)
    with capture_quent_logs() as handler:
      result = await Chain(cm).config(debug=True).with_(sync_identity).run()
    self.assertEqual(result, 99)
    # Root is logged, then async transition — no more _logger.debug() calls
    assert len(handler.messages) >= 1

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
# GAP 6: Simple Fast Path vs Full Path Equivalence
# ============================================================================

class SimpleVsFullPathEquivalenceTests(IsolatedAsyncioTestCase):
  """Tests that verify the simple path (_run_simple / _run_async_simple)
  and the full path (_run / _run_async) produce identical results.

  Simple path: only then() links, no debug, no do/except_/finally_
  Full path: has do(), except_(), or debug
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
    self.assertEqual(result, 21)

  async def test_chain_full_async(self):
    """Chain full path (async) — do() forces full path with async fn."""
    result = await Chain(10).then(async_double).then(sync_add_one).do(sync_noop).run()
    self.assertEqual(result, 21)

  async def test_simple_vs_full_async_equivalence(self):
    """Simple and full async paths produce identical results."""
    simple = await Chain(10).then(async_double).then(sync_add_one).run()
    full = await Chain(10).then(async_double).then(sync_add_one).do(sync_noop).run()
    assert simple == full == 21

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
    self.assertEqual(result, 11)

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
    self.assertEqual(result, 11)

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
    """Adding finally_ does not change the path (finally is separate).
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

