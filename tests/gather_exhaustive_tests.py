"""Exhaustive tests for _make_gather: _gather_op and _to_async, coroutine cleanup, edge cases."""
from __future__ import annotations

import asyncio
import unittest
import warnings
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import async_fn, sync_fn, async_raise_fn


# ---------------------------------------------------------------------------
# TestGatherSyncTier — _gather_op sync fast path
# ---------------------------------------------------------------------------

class TestGatherSyncTier(unittest.TestCase):

  def test_zero_fns(self):
    """Empty gather returns empty list."""
    result = Chain(5).gather().run()
    self.assertEqual(result, [])

  def test_single_fn(self):
    """Single sync fn."""
    result = Chain(5).gather(lambda x: x * 3).run()
    self.assertEqual(result, [15])

  def test_multiple_fns_order_preserved(self):
    """Results are in the same order as fns."""
    result = Chain(10).gather(
      lambda x: x + 1,
      lambda x: x + 2,
      lambda x: x + 3,
    ).run()
    self.assertEqual(result, [11, 12, 13])

  def test_fn_receives_current_value(self):
    """Each fn receives the current chain value."""
    received = []
    def capture(x):
      received.append(x)
      return x
    Chain(42).gather(capture, capture, capture).run()
    self.assertEqual(received, [42, 42, 42])

  def test_fn_raises_first(self):
    """Exception in first fn propagates, subsequent fns never called."""
    called = []
    def second(x):
      called.append(True)
      return x
    with self.assertRaises(ZeroDivisionError):
      Chain(5).gather(lambda x: 1 / 0, second).run()
    self.assertEqual(called, [])

  def test_fn_raises_middle(self):
    """Exception in middle fn: prior results exist, later fns skipped."""
    called = []
    def first(x):
      called.append('first')
      return x
    def third(x):
      called.append('third')
      return x
    with self.assertRaises(ValueError):
      Chain(5).gather(
        first,
        lambda x: (_ for _ in ()).throw(ValueError('mid')),
        third,
      ).run()
    self.assertEqual(called, ['first'])

  def test_fn_raises_last(self):
    """Exception in last fn: all prior fns executed."""
    called = []
    def track(x):
      called.append(x)
      return x
    with self.assertRaises(ZeroDivisionError):
      Chain(5).gather(track, track, lambda x: 1 / 0).run()
    self.assertEqual(called, [5, 5])

  def test_fn_returns_none(self):
    """fn returning None keeps None in results."""
    result = Chain(5).gather(lambda x: None, lambda x: None).run()
    self.assertEqual(result, [None, None])

  def test_fn_returns_callable(self):
    """Returned callable is stored as a value, not invoked."""
    result = Chain(5).gather(lambda x: lambda: x).run()
    self.assertEqual(len(result), 1)
    self.assertTrue(callable(result[0]))
    self.assertEqual(result[0](), 5)

  def test_many_fns_20(self):
    """20 sync fns all execute and return in order."""
    fns = [lambda x, i=i: x + i for i in range(20)]
    result = Chain(0).gather(*fns).run()
    self.assertEqual(result, list(range(20)))


# ---------------------------------------------------------------------------
# TestGatherAsyncTier — _to_async path
# ---------------------------------------------------------------------------

class TestGatherAsyncTier(IsolatedAsyncioTestCase):

  async def test_all_async_fns(self):
    """All fns are async -> _to_async path, asyncio.gather."""
    async def add_one(x):
      return x + 1
    async def add_two(x):
      return x + 2
    result = await Chain(5).gather(add_one, add_two).run()
    self.assertEqual(result, [6, 7])

  async def test_mixed_sync_async(self):
    """Mix of sync and async fns -> _to_async, sync results kept, async gathered."""
    result = await Chain(5).gather(
      lambda x: x + 1,  # sync
      async_fn,          # async (x + 1)
      lambda x: x * 2,  # sync
    ).run()
    self.assertEqual(result, [6, 6, 10])

  async def test_single_async(self):
    """Single async fn triggers _to_async."""
    async def double(x):
      return x * 2
    result = await Chain(5).gather(double).run()
    self.assertEqual(result, [10])

  async def test_preserves_order_with_async(self):
    """Order preserved even when async fns complete at different times."""
    async def slow(x):
      await asyncio.sleep(0.02)
      return 'slow'
    async def fast(x):
      return 'fast'
    result = await Chain(1).gather(slow, fast).run()
    self.assertEqual(result, ['slow', 'fast'])

  async def test_async_fn_raises(self):
    """Exception in async fn propagates via asyncio.gather."""
    with self.assertRaises(ValueError):
      await Chain(5).gather(async_raise_fn).run()

  async def test_coroutine_cleanup_on_exception(self):
    """When a fn raises during setup, already-created coroutines are closed."""
    closed = []

    class TrackableAwaitable:
      """Wrapper that delegates awaiting but tracks .close() calls."""
      def __init__(self, coro):
        self._coro = coro
      def __await__(self):
        return self._coro.__await__()
      def close(self):
        closed.append(True)
        return self._coro.close()

    async def coro_fn(x):
      return x + 1

    def make_tracking_coro(x):
      return TrackableAwaitable(coro_fn(x))

    def raising_fn(x):
      raise RuntimeError('setup error')

    with self.assertRaises(RuntimeError):
      Chain(5).gather(make_tracking_coro, raising_fn).run()
    # The awaitable from make_tracking_coro should have been closed
    self.assertEqual(closed, [True])

  async def test_no_resource_warning(self):
    """No RuntimeWarning about unawaited coroutines when fn raises during setup."""
    async def coro_fn(x):
      return x

    def raising_fn(x):
      raise RuntimeError('boom')

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(RuntimeError):
        Chain(5).gather(coro_fn, raising_fn).run()
      # Filter for RuntimeWarning about coroutines
      coro_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      self.assertEqual(coro_warnings, [])

  async def test_to_async_index_tracking(self):
    """_to_async correctly tracks indices of awaitable results."""
    async def async_double(x):
      return x * 2

    # Mix: sync at 0, async at 1, sync at 2, async at 3
    result = await Chain(5).gather(
      lambda x: x,        # 0: sync -> 5
      async_double,        # 1: async -> 10
      lambda x: x + 1,    # 2: sync -> 6
      async_double,        # 3: async -> 10
    ).run()
    self.assertEqual(result, [5, 10, 6, 10])

  async def test_many_async_fns_20(self):
    """20 async fns gathered concurrently."""
    async def add_i(x, i=0):
      return x + i
    fns = [lambda x, i=i: add_i(x, i) for i in range(20)]
    result = await Chain(0).gather(*fns).run()
    self.assertEqual(result, list(range(20)))


# ---------------------------------------------------------------------------
# BEYOND SPEC — Advanced edge cases for gather
# ---------------------------------------------------------------------------

class TestGatherWithReturn(IsolatedAsyncioTestCase):
  """gather where one fn uses Chain.return_()."""

  def test_return_in_sync_fn_propagates(self):
    """Chain.return_() inside a gather fn exits the whole chain."""
    result = Chain(5).gather(
      lambda x: Chain.return_(99),
      lambda x: x + 1,  # never reached
    ).run()
    self.assertEqual(result, 99)

  async def test_return_in_async_gather(self):
    """Chain.return_() in a sync fn inside gather with async fns."""
    async def async_fn_(x):
      return x + 1
    # return_ raises _Return which propagates out of gather
    result = Chain(5).gather(
      lambda x: Chain.return_(88),
      async_fn_,
    ).run()
    self.assertEqual(result, 88)


class TestGatherEmptyAgain(unittest.TestCase):
  """Additional empty gather scenarios."""

  def test_zero_fns_different_value(self):
    result = Chain('hello').gather().run()
    self.assertEqual(result, [])

  def test_zero_fns_none_value(self):
    result = Chain(None).gather().run()
    self.assertEqual(result, [])

  def test_zero_fns_null_value(self):
    """Gather with Null as initial value (chain with no root)."""
    result = Chain().then(lambda: 5).gather().run()
    self.assertEqual(result, [])


class TestGatherWithMap(IsolatedAsyncioTestCase):
  """Gather results used as input to map."""

  def test_gather_then_map(self):
    """Gather produces a list, map iterates it."""
    result = (
      Chain(10)
      .gather(lambda x: x + 1, lambda x: x + 2, lambda x: x + 3)
      .map(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, [22, 24, 26])

  async def test_async_gather_then_map(self):
    """Async gather -> map."""
    async def add_one(x):
      return x + 1
    result = await (
      Chain(10)
      .gather(add_one, lambda x: x + 2)
      .map(lambda x: x * 10)
      .run()
    )
    self.assertEqual(result, [110, 120])


class TestGatherDuplicateFunctions(unittest.TestCase):
  """Gather with the same function multiple times."""

  def test_same_fn_twice(self):
    fn = lambda x: x + 1
    result = Chain(5).gather(fn, fn).run()
    self.assertEqual(result, [6, 6])

  def test_same_fn_five_times(self):
    fn = lambda x: x * 2
    result = Chain(3).gather(fn, fn, fn, fn, fn).run()
    self.assertEqual(result, [6, 6, 6, 6, 6])


class TestGatherExceptionCleanup(IsolatedAsyncioTestCase):
  """Exception during _gather_op setup with coroutine cleanup."""

  async def test_multiple_coros_closed_on_exception(self):
    """When fn at index 2 raises, coroutines at 0 and 1 are closed."""
    close_count = []

    class TrackableAwaitable:
      def __init__(self, coro):
        self._coro = coro
      def __await__(self):
        return self._coro.__await__()
      def close(self):
        close_count.append(True)
        return self._coro.close()

    async def coro_fn(x):
      return x

    def make_tracked_coro(x):
      return TrackableAwaitable(coro_fn(x))

    def raiser(x):
      raise RuntimeError('boom')

    with self.assertRaises(RuntimeError):
      Chain(5).gather(make_tracked_coro, make_tracked_coro, raiser).run()
    self.assertEqual(len(close_count), 2)

  def test_sync_results_not_closed(self):
    """Sync results (no .close method) don't cause issues during cleanup."""
    def raiser(x):
      raise RuntimeError('boom')
    with self.assertRaises(RuntimeError):
      Chain(5).gather(lambda x: x, lambda x: x, raiser).run()
    # No error from cleanup — sync results silently skipped


class TestGatherWithChainOps(IsolatedAsyncioTestCase):
  """Gather combined with other chain operations."""

  def test_gather_after_then(self):
    """Gather receives value from a preceding .then() step."""
    result = (
      Chain(5)
      .then(lambda x: x * 2)
      .gather(lambda x: x + 1, lambda x: x - 1)
      .run()
    )
    self.assertEqual(result, [11, 9])

  def test_gather_with_except(self):
    """Exception in gather caught by except_."""
    result = (
      Chain(5)
      .gather(lambda x: 1 / 0)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_gather_with_except_async(self):
    """Async exception in gather caught by except_."""
    result = await (
      Chain(5)
      .gather(async_raise_fn)
      .except_(lambda rv, exc: 'async_caught')
      .run()
    )
    self.assertEqual(result, 'async_caught')

  def test_gather_then_then(self):
    """Gather result processed by subsequent .then()."""
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .then(lambda lst: sum(lst))
      .run()
    )
    self.assertEqual(result, 13)


class TestGatherReturnTypes(unittest.TestCase):
  """Gather fn return value types."""

  def test_fn_returns_list(self):
    result = Chain(5).gather(lambda x: [x, x]).run()
    self.assertEqual(result, [[5, 5]])

  def test_fn_returns_dict(self):
    result = Chain(5).gather(lambda x: {'val': x}).run()
    self.assertEqual(result, [{'val': 5}])

  def test_fn_returns_tuple(self):
    result = Chain(5).gather(lambda x: (x, x + 1)).run()
    self.assertEqual(result, [(5, 6)])

  def test_fn_returns_zero(self):
    result = Chain(5).gather(lambda x: 0).run()
    self.assertEqual(result, [0])

  def test_fn_returns_false(self):
    result = Chain(5).gather(lambda x: False).run()
    self.assertEqual(result, [False])

  def test_fn_returns_empty_string(self):
    result = Chain(5).gather(lambda x: '').run()
    self.assertEqual(result, [''])


class TestGatherConcurrencyBehavior(IsolatedAsyncioTestCase):
  """Verify async fns run concurrently via asyncio.gather."""

  async def test_async_fns_run_concurrently(self):
    """Multiple async fns should run concurrently, not sequentially."""
    import time
    start = time.monotonic()

    async def slow(x):
      await asyncio.sleep(0.05)
      return x

    result = await Chain(1).gather(slow, slow, slow).run()
    elapsed = time.monotonic() - start
    self.assertEqual(result, [1, 1, 1])
    # If sequential, would take ~0.15s. Concurrent should be ~0.05s.
    self.assertLess(elapsed, 0.12)


class TestGatherMetadata(unittest.TestCase):
  """Verify _quent_op metadata on gather operation."""

  def test_gather_op_has_metadata(self):
    from quent._ops import _make_gather
    op = _make_gather((lambda x: x,))
    self.assertEqual(op._quent_op, 'gather')
    self.assertEqual(len(op._fns), 1)


class TestGatherWithNullInput(unittest.TestCase):
  """Gather where the current value is special."""

  def test_current_value_is_none(self):
    """Fns receive None as current_value."""
    result = Chain(None).gather(lambda x: x is None).run()
    self.assertEqual(result, [True])

  def test_current_value_is_string(self):
    result = Chain('hello').gather(lambda x: x.upper(), lambda x: len(x)).run()
    self.assertEqual(result, ['HELLO', 5])


class TestGatherSingleAsyncAmongSync(IsolatedAsyncioTestCase):
  """Single async fn among many sync triggers _to_async."""

  async def test_one_async_among_four_sync(self):
    async def async_double(x):
      return x * 2
    result = await Chain(5).gather(
      lambda x: x + 1,
      lambda x: x + 2,
      async_double,
      lambda x: x + 4,
    ).run()
    self.assertEqual(result, [6, 7, 10, 9])

  async def test_last_fn_async(self):
    """Only the last fn is async."""
    async def async_last(x):
      return x * 10
    result = await Chain(3).gather(
      lambda x: x + 1,
      lambda x: x + 2,
      async_last,
    ).run()
    self.assertEqual(result, [4, 5, 30])


if __name__ == '__main__':
  unittest.main()
