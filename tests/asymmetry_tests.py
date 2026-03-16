# SPDX-License-Identifier: MIT
"""Tests for SPEC §16 — Known Asymmetries.

Tests cover:
- §16.1: Sync iterate TypeError on async chain
- §16.2: Async transition for sync handler returning coroutine
- §16.2a: except_(reraise=True) async transition
- §16.3: Sync with_() __exit__ returning coroutine during control flow (return_ and break_)
- §16.3: Warning content for with_ context manager signal
- §16.4: Concurrent sync workers detecting awaitable results
- §16.4: Awaitable closure in concurrent sync workers
- §16.5: break_(value) semantics — uniform append behavior in foreach() and iterate()
- §16.6: return_(value) semantics differ between chain and iterate()
"""

from __future__ import annotations

import asyncio
import warnings
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Chain


class SyncCMWithAsyncExit:
  """Sync CM whose __exit__ returns a coroutine (for §16.3 tests)."""

  def __enter__(self):
    return 'ctx'

  def __exit__(self, *args):
    async def _async_exit():
      pass

    return _async_exit()


# --- §16.1: Sync iterate TypeError ---


class SyncIterateTypeErrorTest(TestCase):
  """§16.1: sync iteration on async chain raises TypeError."""

  def test_sync_iterate_async_chain(self):
    """for-loop on chain with async steps raises TypeError."""

    async def async_fn(x):
      return x

    it = Chain([1, 2, 3]).then(async_fn).iterate()
    with self.assertRaises(TypeError) as ctx:
      for _item in it:
        pass
    self.assertIn('async', str(ctx.exception).lower())

  def test_sync_iterate_async_callback(self):
    """for-loop with async iterate callback raises TypeError."""

    async def async_fn(x):
      return x * 2

    it = Chain([1, 2, 3]).iterate(async_fn)
    with self.assertRaises(TypeError) as ctx:
      for _item in it:
        pass
    self.assertIn('async', str(ctx.exception).lower())


class AsyncIterateWorksTest(IsolatedAsyncioTestCase):
  """§16.1: async for works on async chains (the correct alternative)."""

  async def test_async_iterate(self):
    """async for-loop on async chain works."""

    async def async_double(x):
      return x * 2

    it = Chain([1, 2, 3]).iterate(async_double)
    results = []
    async for item in it:
      results.append(item)
    self.assertEqual(results, [2, 4, 6])


# --- §16.2: Async transition for sync handlers ---


class AsyncFinallyTransitionTest(IsolatedAsyncioTestCase):
  """§16.2: sync chain finally returning coroutine → async transition."""

  async def test_finally_coroutine_async_transition(self):
    """Sync chain's finally_ returning coroutine triggers async transition."""
    cleanup_ran = asyncio.Event()

    async def async_cleanup(rv):
      cleanup_ran.set()

    result = Chain(5).then(lambda x: x + 1).finally_(async_cleanup).run()

    # Should be a coroutine (async transition), not a plain value
    self.assertTrue(asyncio.iscoroutine(result), 'run() should return a coroutine for async transition')

    # Await the coroutine to get the chain result
    final = await result
    self.assertEqual(final, 6)

    # Verify cleanup ran
    self.assertTrue(cleanup_ran.is_set())


class AsyncFinallyTransitionNoLoopTest(TestCase):
  """§16.2: sync chain with async finally returns a coroutine (not QuentException)."""

  def test_finally_coroutine_returns_coroutine(self):
    """Sync chain's finally_ returning coroutine: run() returns a coroutine."""

    async def async_cleanup(rv):
      pass

    result = Chain(5).finally_(async_cleanup).run()
    # Under the async transition model, run() returns a coroutine
    # even without a running event loop — it's the caller's responsibility
    # to await it.
    self.assertTrue(asyncio.iscoroutine(result))
    # Clean up the coroutine to avoid ResourceWarning
    result.close()


# --- §16.2a: except_(reraise=True) async transition ---


class ExceptRaiseTrueAsyncTransitionTest(IsolatedAsyncioTestCase):
  """§16.2a: except_(reraise=True) with async handler → async transition."""

  async def test_except_raise_true_coroutine_async_transition(self):
    """except_(reraise=True) with async handler causes async transition; await re-raises."""
    handler_ran = asyncio.Event()

    async def async_handler(info):
      handler_ran.set()

    # In an async context, run() returns a coroutine (async transition).
    # Awaiting it re-raises the original exception.
    with self.assertRaises(ZeroDivisionError):
      await Chain(1).then(lambda x: 1 / 0).except_(async_handler, reraise=True).run()

    self.assertTrue(handler_ran.is_set())

  async def test_except_raise_true_async_handler_returns_coroutine(self):
    """§16.2a: except_(reraise=True) with async handler causes async transition."""
    handler_ran = asyncio.Event()

    async def async_handler(info):
      handler_ran.set()

    # run() returns a coroutine (async transition); awaiting it re-raises the original exception
    coro = Chain(1).then(lambda x: 1 / 0).except_(async_handler, reraise=True).run()
    # The result should be a coroutine (async transition occurred)
    self.assertTrue(asyncio.iscoroutine(coro), 'run() should return a coroutine for async transition')
    with self.assertRaises(ZeroDivisionError):
      await coro
    self.assertTrue(handler_ran.is_set())


# --- §16.3: Sync with_() __exit__ returning coroutine during control flow ---


class WithExitCoroutineDuringControlFlowTest(IsolatedAsyncioTestCase):
  """§16.3: with_() __exit__ returning coroutine during control flow signal."""

  async def test_with_exit_coroutine_on_return_signal(self):
    """Control flow signal with __exit__ returning coroutine: coroutine closed, signal propagates."""

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      result = Chain(SyncCMWithAsyncExit()).with_(lambda ctx: Chain.return_('early')).run()

    self.assertEqual(result, 'early')
    # RuntimeWarning about coroutine being closed/skipped
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    self.assertTrue(
      len(runtime_warnings) > 0,
      f'Expected RuntimeWarning about __exit__ coroutine, got: {[str(x.message) for x in w]}',
    )

  def test_with_exit_coroutine_on_break_signal(self):
    """§16.3: break_() inside with_() where __exit__ returns coroutine triggers §16.3 behavior.

    break_() is only valid inside foreach/iterate context. Outside that context,
    the engine wraps the _Break signal in QuentException. However, the §16.3
    behavior (coroutine closed, RuntimeWarning emitted) still triggers at the
    with_() level before the signal propagates to the engine. This test verifies
    that the warning is emitted with '_Break' in the message.
    """
    from quent import QuentException

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(QuentException):
        Chain(SyncCMWithAsyncExit()).with_(lambda ctx: Chain.break_('stop')).run()

    # §16.3: RuntimeWarning about __exit__ coroutine being closed during _Break signal
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    warning_messages = [str(x.message) for x in runtime_warnings]
    matching = [m for m in warning_messages if '_Break' in m and 'async cleanup was skipped' in m]
    self.assertTrue(
      len(matching) > 0,
      f'Expected RuntimeWarning mentioning _Break and async cleanup, got: {warning_messages}',
    )


class WithContextManagerSignalWarningTest(IsolatedAsyncioTestCase):
  """§16.3: Warning content for with_ context manager signal."""

  async def test_with_exit_coroutine_return_signal_warning_content(self):
    """§16.3: Warning message contains signal type name for return_()."""

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      result = Chain(SyncCMWithAsyncExit()).with_(lambda ctx: Chain.return_('early')).run()

    self.assertEqual(result, 'early')
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    warning_messages = [str(x.message) for x in runtime_warnings]
    # Warning should mention '_Return' (the signal type name) and 'async cleanup was skipped'
    matching = [m for m in warning_messages if '_Return' in m and 'async cleanup was skipped' in m]
    self.assertTrue(
      len(matching) > 0,
      f'Expected warning with _Return and async cleanup, got: {warning_messages}',
    )

  def test_with_exit_coroutine_break_signal_warning(self):
    """§16.3: break_() inside with_ with coroutine __exit__ emits warning about _Break."""

    # break_() inside with_ triggers §16.3: __exit__ returns a coroutine
    # during a control flow signal. The coroutine is closed and a warning emitted.
    # break_() outside iteration is caught by _run and wrapped in QuentException.
    from quent import QuentException

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(QuentException):
        Chain(SyncCMWithAsyncExit()).with_(lambda ctx: Chain.break_('stop')).run()

    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    warning_messages = [str(x.message) for x in runtime_warnings]
    # Warning should mention '_Break' (the signal type name) and 'async cleanup was skipped'
    matching = [m for m in warning_messages if '_Break' in m and 'async cleanup was skipped' in m]
    self.assertTrue(
      len(matching) > 0,
      f'Expected warning with _Break and async cleanup, got: {warning_messages}',
    )


# --- §16.4: Concurrent sync workers detecting awaitable ---


class ConcurrentSyncAwaitableDetectionTest(TestCase):
  """§16.4: sync workers returning awaitable → TypeError."""

  def test_map_sync_then_async_worker(self):
    """Concurrent map: first sync, later async → TypeError."""
    call_count = 0

    def mixed_fn(x):
      nonlocal call_count
      call_count += 1
      if call_count > 1:

        async def _coro():
          return x

        return _coro()
      return x

    with self.assertRaises(TypeError) as ctx:
      Chain([1, 2, 3, 4]).foreach(mixed_fn, concurrency=2).run()
    self.assertIn('sync', str(ctx.exception).lower())

  def test_gather_sync_then_async(self):
    """Concurrent gather: first sync, later async → TypeError."""
    calls = []

    def sync_fn(x):
      calls.append('sync')
      return x

    async def async_fn(x):
      calls.append('async')
      return x

    with self.assertRaises(TypeError) as ctx:
      Chain(1).gather(sync_fn, async_fn).run()
    self.assertIn('sync', str(ctx.exception).lower())


class AwaitableClosureTest(TestCase):
  """§16.4: Awaitable coroutines are closed when detected in sync concurrent workers."""

  def test_sync_worker_awaitable_closed(self):
    """§16.4: Awaitable result from sync worker is closed to prevent ResourceWarning."""
    closed = []
    call_count = 0

    class TrackableCoroutine:
      """A coroutine-like object that tracks whether .close() was called."""

      def __await__(self):
        yield

      def close(self):
        closed.append(True)

    def mixed_fn(x):
      nonlocal call_count
      call_count += 1
      if call_count > 1:
        return TrackableCoroutine()
      return x

    with self.assertRaises(TypeError):
      Chain([1, 2, 3, 4]).foreach(mixed_fn, concurrency=2).run()
    # The awaitable should have been closed
    self.assertTrue(len(closed) > 0, 'Awaitable should have been closed')


# --- §16.5: break_(value) semantics — uniform append behavior ---


class BreakValueMapVsIterateTest(TestCase):
  """§16.5: break_(value) appends value to partial results in both foreach() and iterate()."""

  def test_map_break_with_value_appends_to_partial_results(self):
    """§16.5: In map(), break_(value) appends value to partial results."""
    result = Chain([1, 2, 3, 4]).foreach(lambda x: Chain.return_(42) if x == 3 else x).run()
    # break_ in map appends to partial results.
    # Let's use break_ properly:
    result = Chain([1, 2, 3, 4]).foreach(lambda x: Chain.break_(42) if x == 3 else x).run()
    # break_(42) appends 42 to partial results [1, 2]
    self.assertEqual(result, [1, 2, 42])

  def test_map_break_without_value_returns_partial(self):
    """§16.5: In map(), break_() returns partial results as-is."""
    result = Chain([1, 2, 3, 4]).foreach(lambda x: Chain.break_() if x == 3 else x * 10).run()
    # break_() at x==3 returns partial results [10, 20]
    self.assertEqual(result, [10, 20])

  def test_iterate_break_with_value_yields_additional_item(self):
    """§16.5: In iterate(), break_(value) yields value as one additional item."""
    items = []
    for item in Chain([1, 2, 3, 4]).iterate(lambda x: Chain.break_(99) if x == 3 else x * 10):
      items.append(item)
    # Items 1,2 yield 10,20. At x==3, break_(99) yields 99 then stops.
    self.assertEqual(items, [10, 20, 99])

  def test_iterate_break_without_value_stops_immediately(self):
    """§16.5: In iterate(), break_() stops immediately with no additional item."""
    items = []
    for item in Chain([1, 2, 3, 4]).iterate(lambda x: Chain.break_() if x == 3 else x * 10):
      items.append(item)
    # Items 1,2 yield 10,20. At x==3, break_() stops. No additional item.
    self.assertEqual(items, [10, 20])

  def test_uniform_break_value_map_and_iterate(self):
    """§16.5: break_(value) appends uniformly in both foreach() and iterate()."""

    def fn(x):
      if x == 3:
        return Chain.break_('STOP')
      return x * 10

    # map: break_('STOP') appends to partial results
    map_result = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(map_result, [10, 20, 'STOP'])

    # iterate: break_('STOP') is yielded as additional item (same behavior)
    iterate_result = list(Chain([1, 2, 3, 4]).iterate(fn))
    self.assertEqual(iterate_result, [10, 20, 'STOP'])

  def test_contrast_break_no_value_map_vs_iterate(self):
    """§16.5: Side-by-side contrast of break_() (no value) in map vs iterate."""

    def fn(x):
      if x == 3:
        return Chain.break_()
      return x * 10

    # map: break_() returns partial results as-is
    map_result = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(map_result, [10, 20])

    # iterate: break_() stops immediately, no additional item
    iterate_result = list(Chain([1, 2, 3, 4]).iterate(fn))
    self.assertEqual(iterate_result, [10, 20])


class BreakValueMapVsIterateAsyncTest(IsolatedAsyncioTestCase):
  """§16.5: Async variants of break_(value) uniform append behavior."""

  async def test_async_map_break_with_value_appends(self):
    """§16.5: async map() break_(value) appends value to partial results."""

    async def fn(x):
      if x == 3:
        return Chain.break_('replaced')
      return x * 10

    result = await Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(result, [10, 20, 'replaced'])

  async def test_async_iterate_break_with_value_yields(self):
    """§16.5: async iterate() break_(value) yields as additional item."""

    async def fn(x):
      if x == 3:
        return Chain.break_('extra')
      return x * 10

    items = []
    async for item in Chain([1, 2, 3, 4]).iterate(fn):
      items.append(item)
    self.assertEqual(items, [10, 20, 'extra'])


# --- §6.3.3: Async finally raising during await → __context__ ---


class AsyncFinallyRaisesDuringAwaitTest(IsolatedAsyncioTestCase):
  """§6.3.3 / §6.3.5: Fully async chain with async finally that raises.

  When the finally handler raises while an exception is already active,
  the finally handler's exception replaces the original exception
  (matching Python's try/finally behavior). The original exception is
  preserved as __context__ on the finally exception.
  """

  async def test_async_finally_raise_replaces_active_exception_with_context(self):
    """§6.3.3: Async finally raising replaces active exception; __context__ is the original."""

    async def async_step(x):
      raise ValueError('original async error')

    async def async_finally_handler(rv):
      raise RuntimeError('finally handler boom')

    c = Chain(1).then(async_step).finally_(async_finally_handler)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertEqual(str(ctx.exception), 'finally handler boom')
    self.assertIsInstance(ctx.exception.__context__, ValueError)
    self.assertEqual(str(ctx.exception.__context__), 'original async error')


# --- §16.6: return_(value) semantics differ between chain and iterate() ---


class ReturnValueChainVsIterateTest(TestCase):
  """§16.6: return_(value) replaces result in chain, yields final item in iterate."""

  def test_chain_return_replaces_entire_result(self):
    """§16.6: In chain execution, return_(value) replaces the entire result."""
    result = Chain(1).then(lambda x: x + 1).then(lambda x: Chain.return_('early')).then(lambda x: x * 100).run()
    self.assertEqual(result, 'early')

  def test_iterate_return_yields_final_item(self):
    """§16.6: In iterate(), return_(value) yields value as final item."""
    items = []
    for item in Chain([1, 2, 3, 4]).iterate(lambda x: Chain.return_('done') if x == 3 else x * 10):
      items.append(item)
    # Items 1,2 yield 10,20. At x==3, return_('done') yields 'done' then stops.
    self.assertEqual(items, [10, 20, 'done'])

  def test_contrast_return_value_chain_vs_iterate(self):
    """§16.6: Side-by-side contrast of return_(value) in chain vs iterate."""

    def fn(x):
      if x == 3:
        Chain.return_('final')
      return x * 10

    # Chain: return_('final') replaces entire chain result
    chain_result = Chain([1, 2, 3, 4]).foreach(fn).run()
    # return_ propagates out of map entirely, becomes the chain result
    self.assertEqual(chain_result, 'final')

    # iterate: return_('final') yields as last item
    iterate_result = list(Chain([1, 2, 3, 4]).iterate(fn))
    self.assertEqual(iterate_result, [10, 20, 'final'])

  def test_iterate_return_no_value_stops(self):
    """§16.6: In iterate(), return_() with no value stops without extra yield."""
    items = []
    for item in Chain([1, 2, 3, 4]).iterate(lambda x: Chain.return_() if x == 3 else x * 10):
      items.append(item)
    # return_() stops iteration, no additional item
    self.assertEqual(items, [10, 20])


class ReturnValueChainVsIterateAsyncTest(IsolatedAsyncioTestCase):
  """§16.6: Async variants of return_(value) asymmetry."""

  async def test_async_chain_return_replaces_result(self):
    """§16.6: async chain return_(value) replaces entire result."""

    async def fn(x):
      return x + 1

    result = await Chain(1).then(fn).then(lambda x: Chain.return_('early')).then(fn).run()
    self.assertEqual(result, 'early')

  async def test_async_iterate_return_yields_final_item(self):
    """§16.6: async iterate() return_(value) yields as final item."""

    async def fn(x):
      if x == 3:
        Chain.return_('async_done')
      return x * 10

    items = []
    async for item in Chain([1, 2, 3, 4]).iterate(fn):
      items.append(item)
    self.assertEqual(items, [10, 20, 'async_done'])
