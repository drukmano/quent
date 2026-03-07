"""Exhaustive sync-to-async transition point tests across every chain method combination.

Tests every possible point where _run() detects an awaitable and delegates to
_run_async(), plus the three-tier sync/async handoff patterns within operations
(map, filter, with_, if_/else_, gather).

Covers:
  PART 1: Transition point per method (then, do, map, filter, gather, with_, if_)
  PART 2: Multi-method async transition chains
  PART 3: Exception during async transition
  PART 4: Nested chain async transitions
  PART 5: iterate async transitions
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  AsyncCM,
  AsyncRange,
  SyncCM,
  SyncCMWithAwaitableExit,
  make_tracker,
  make_async_tracker,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def sync_add1(x):
  return x + 1

async def async_add1(x):
  return x + 1

def sync_double(x):
  return x * 2

async def async_double(x):
  return x * 2

def sync_identity(x):
  return x

async def async_identity(x):
  return x

async def async_const(val):
  """Return a factory that always returns an async-resolved constant."""
  return val

def sync_const(val):
  return val

def sync_noop(x):
  pass

async def async_noop(x):
  pass

async def async_truthy(x):
  return True

async def async_falsy(x):
  return False

def sync_truthy(x):
  return True

def sync_falsy(x):
  return False

async def async_is_even(x):
  return x % 2 == 0

def sync_is_even(x):
  return x % 2 == 0

async def async_raise_value_error(x=None):
  raise ValueError('async value error')

async def async_raise_runtime(x=None):
  raise RuntimeError('async runtime error')

def sync_raise_value_error(x=None):
  raise ValueError('sync value error')


class AsyncRangeLocal:
  """Local async iterable for tests that need custom ranges."""
  def __init__(self, n):
    self.n = n
  def __aiter__(self):
    self._i = 0
    return self
  async def __anext__(self):
    if self._i >= self.n:
      raise StopAsyncIteration
    val = self._i
    self._i += 1
    return val


class SyncCMLocal:
  """Simple sync CM returning a fixed value."""
  def __init__(self, val='ctx'):
    self.val = val
    self.entered = False
    self.exited = False
  def __enter__(self):
    self.entered = True
    return self.val
  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCMLocal:
  """Simple async CM returning a fixed value."""
  def __init__(self, val='async_ctx'):
    self.val = val
    self.entered = False
    self.exited = False
  async def __aenter__(self):
    self.entered = True
    return self.val
  async def __aexit__(self, *args):
    self.exited = True
    return False


class SyncCMAsyncExit:
  """Sync CM whose __exit__ returns a coroutine (awaitable exit)."""
  def __init__(self):
    self.entered = False
    self.exited = False
  def __enter__(self):
    self.entered = True
    return 'sync_ctx_async_exit'
  def __exit__(self, *args):
    self.exited = True
    async def _aexit():
      return False
    return _aexit()


# ===========================================================================
# PART 1: Transition point per method
# ===========================================================================

# ---------------------------------------------------------------------------
# then transitions (6 tests)
# ---------------------------------------------------------------------------

class TestThenTransition(IsolatedAsyncioTestCase):

  async def test_all_sync_before_async_at_middle_sync_after(self):
    """Sync steps, then async at this then, then sync steps after."""
    result = await Chain(10).then(sync_add1).then(async_add1).then(sync_double).run()
    # 10 -> 11 -> 12 -> 24
    self.assertEqual(result, 24)

  async def test_async_at_first_then(self):
    """Async transition at the very first then step."""
    result = await Chain(5).then(async_double).run()
    self.assertEqual(result, 10)

  async def test_async_at_last_then(self):
    """Async transition at the last then in the chain."""
    result = await Chain(1).then(sync_add1).then(sync_add1).then(async_add1).run()
    # 1 -> 2 -> 3 -> 4
    self.assertEqual(result, 4)

  async def test_async_at_middle_in_5_step_chain(self):
    """5-step chain with async at step 3."""
    result = await (
      Chain(1)
      .then(sync_add1)   # 2
      .then(sync_add1)   # 3
      .then(async_add1)  # 4 (transition)
      .then(sync_add1)   # 5
      .then(sync_add1)   # 6
      .run()
    )
    self.assertEqual(result, 6)

  async def test_two_async_thens_in_a_row(self):
    """Two consecutive async thens."""
    result = await Chain(1).then(async_add1).then(async_double).run()
    # 1 -> 2 -> 4
    self.assertEqual(result, 4)

  async def test_alternating_sync_async_thens(self):
    """Alternating sync and async thens."""
    result = await (
      Chain(0)
      .then(sync_add1)   # 1
      .then(async_add1)  # 2
      .then(sync_add1)   # 3
      .then(async_add1)  # 4
      .then(sync_add1)   # 5
      .then(async_add1)  # 6
      .run()
    )
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# do transitions (4 tests)
# ---------------------------------------------------------------------------

class TestDoTransition(IsolatedAsyncioTestCase):

  async def test_async_do_result_discarded(self):
    """Async do fires but its result is discarded; pipeline value preserved."""
    tracker = []
    async def track(x):
      tracker.append(x)
      return 'ignored'
    result = await Chain(42).do(track).run()
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [42])

  async def test_sync_before_async_do_sync_after(self):
    """Sync -> async do -> sync: pipeline value passes through do unchanged."""
    tracker = []
    async def track(x):
      tracker.append(x)
    result = await Chain(10).then(sync_add1).do(track).then(sync_double).run()
    # 10 -> 11 -> (do track 11) -> 22
    self.assertEqual(result, 22)
    self.assertEqual(tracker, [11])

  async def test_multiple_async_dos(self):
    """Multiple async dos in sequence, all results discarded."""
    t1, t2, t3 = [], [], []
    async def track1(x): t1.append(x)
    async def track2(x): t2.append(x)
    async def track3(x): t3.append(x)
    result = await Chain(7).do(track1).do(track2).do(track3).run()
    self.assertEqual(result, 7)
    self.assertEqual(t1, [7])
    self.assertEqual(t2, [7])
    self.assertEqual(t3, [7])

  async def test_sync_do_then_async_then(self):
    """Sync do doesn't trigger transition, but next async then does."""
    tracker = []
    def sync_track(x):
      tracker.append(x)
    result = await Chain(5).do(sync_track).then(async_double).run()
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [5])


# ---------------------------------------------------------------------------
# map transitions (6 tests)
# ---------------------------------------------------------------------------

class TestMapTransition(IsolatedAsyncioTestCase):

  async def test_sync_iterable_fn_returns_awaitable_on_first_item(self):
    """map with sync iterable and fn that always returns awaitable."""
    result = await Chain([1, 2, 3]).map(async_double).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_sync_iterable_fn_returns_awaitable_on_nth_item(self):
    """Sync for first N items, then fn returns awaitable mid-iteration."""
    call_count = 0
    async def async_after_2(x):
      return x * 10

    def mixed_fn(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 3:
        return async_after_2(x)
      return x * 10

    result = await Chain([1, 2, 3, 4, 5]).map(mixed_fn).run()
    self.assertEqual(result, [10, 20, 30, 40, 50])

  async def test_async_iterable(self):
    """map over an async iterable (__aiter__)."""
    result = await Chain(AsyncRangeLocal(4)).map(sync_double).run()
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_sync_iterable_async_fn_continues_with_sync_steps(self):
    """map async fn -> continues with sync steps after."""
    result = await Chain([1, 2]).map(async_add1).then(len).run()
    # [1,2] -> [2,3] -> len=2
    self.assertEqual(result, 2)

  async def test_foreach_with_async_fn(self):
    """foreach with async fn: original items preserved."""
    tracker = []
    async def track(x):
      tracker.append(x * 10)
    result = await Chain([1, 2, 3]).foreach(track).run()
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [10, 20, 30])

  async def test_async_iterable_with_async_fn(self):
    """Async iterable + async fn: full async path."""
    result = await Chain(AsyncRangeLocal(3)).map(async_double).run()
    self.assertEqual(result, [0, 2, 4])


# ---------------------------------------------------------------------------
# filter transitions (6 tests)
# ---------------------------------------------------------------------------

class TestFilterTransition(IsolatedAsyncioTestCase):

  async def test_sync_iterable_filter_returns_awaitable_on_first_item(self):
    """Filter fn returns awaitable on the very first item."""
    result = await Chain([1, 2, 3, 4]).filter(async_is_even).run()
    self.assertEqual(result, [2, 4])

  async def test_sync_iterable_filter_returns_awaitable_on_nth_item(self):
    """Filter fn returns awaitable starting from Nth item (mid-iteration handoff)."""
    call_count = 0

    def mixed_filter(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 3:
        return async_is_even(x)
      return x % 2 == 0

    result = await Chain([1, 2, 3, 4, 5, 6]).filter(mixed_filter).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_async_iterable_filter(self):
    """Filter over an async iterable."""
    result = await Chain(AsyncRangeLocal(6)).filter(sync_is_even).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_filter_continues_with_sync_steps_after(self):
    """Filter with async fn -> continues with sync steps after transition."""
    result = await Chain([1, 2, 3, 4]).filter(async_is_even).then(len).run()
    self.assertEqual(result, 2)

  async def test_async_iterable_async_filter_fn(self):
    """Async iterable + async filter fn: full async path."""
    result = await Chain(AsyncRangeLocal(5)).filter(async_is_even).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_filter_sync_all_pass(self):
    """Sync filter where all items pass (stays sync, no transition)."""
    result = Chain([2, 4, 6]).filter(sync_is_even).run()
    self.assertEqual(result, [2, 4, 6])


# ---------------------------------------------------------------------------
# gather transitions (4 tests)
# ---------------------------------------------------------------------------

class TestGatherTransition(IsolatedAsyncioTestCase):

  async def test_one_fn_returns_awaitable_others_sync(self):
    """One fn async, others sync -> triggers transition."""
    result = await Chain(5).gather(sync_add1, async_double, sync_identity).run()
    self.assertEqual(result, [6, 10, 5])

  async def test_all_fns_return_awaitables(self):
    """All fns are async."""
    result = await Chain(3).gather(async_add1, async_double).run()
    self.assertEqual(result, [4, 6])

  def test_no_fns_return_awaitables_stays_sync(self):
    """All sync fns -> stays sync, no transition."""
    result = Chain(10).gather(sync_add1, sync_double).run()
    self.assertEqual(result, [11, 20])

  async def test_gather_continues_with_sync_steps_after(self):
    """Gather with async fn -> continues with sync steps after."""
    result = await Chain(2).gather(async_add1, async_double).then(sum).run()
    # [3, 4] -> sum = 7
    self.assertEqual(result, 7)


# ---------------------------------------------------------------------------
# with_ transitions (6 tests)
# ---------------------------------------------------------------------------

class TestWithTransition(IsolatedAsyncioTestCase):

  async def test_sync_cm_async_body_fn(self):
    """Sync CM, async body fn -> transition inside with_."""
    cm = SyncCMLocal('hello')
    result = await Chain(cm).with_(async_identity).run()
    self.assertEqual(result, 'hello')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm(self):
    """Async CM (__aenter__/__aexit__) -> full async path."""
    cm = AsyncCMLocal('async_val')
    result = await Chain(cm).with_(sync_identity).run()
    self.assertEqual(result, 'async_val')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_sync_cm_sync_body_async_exit(self):
    """Sync CM whose __exit__ returns an awaitable."""
    cm = SyncCMAsyncExit()
    result = await Chain(cm).with_(sync_identity).run()
    self.assertEqual(result, 'sync_ctx_async_exit')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_continues_with_sync_steps_after(self):
    """with_ async body -> more sync steps after."""
    cm = SyncCMLocal(10)
    result = await Chain(cm).with_(async_identity).then(sync_double).run()
    self.assertEqual(result, 20)

  async def test_with_do_async_body(self):
    """with_do with async body: result discarded, pipeline value preserved."""
    cm = SyncCMLocal('ctx_val')
    tracker = []
    async def track(ctx):
      tracker.append(ctx)
      return 'discarded'
    result = await Chain(cm).with_do(track).run()
    self.assertEqual(result, cm)
    self.assertEqual(tracker, ['ctx_val'])

  async def test_async_cm_async_body(self):
    """Async CM + async body: full async path."""
    cm = AsyncCMLocal(99)
    result = await Chain(cm).with_(async_double).run()
    self.assertEqual(result, 198)


# ---------------------------------------------------------------------------
# if_ transitions (8 tests)
# ---------------------------------------------------------------------------

class TestIfTransition(IsolatedAsyncioTestCase):

  async def test_async_predicate_sync_fn(self):
    """Async predicate, sync fn."""
    result = await Chain(5).if_(async_truthy, then=sync_double).run()
    self.assertEqual(result, 10)

  async def test_sync_predicate_async_fn(self):
    """Sync predicate, async fn."""
    result = await Chain(5).if_(sync_truthy, then=async_double).run()
    self.assertEqual(result, 10)

  async def test_async_predicate_async_fn(self):
    """Both predicate and fn are async."""
    result = await Chain(5).if_(async_truthy, then=async_double).run()
    self.assertEqual(result, 10)

  async def test_async_predicate_falsy_passthrough(self):
    """Async predicate returns falsy -> no fn evaluation, value passes through."""
    result = await Chain(5).if_(async_falsy, then=sync_double).run()
    self.assertEqual(result, 5)

  async def test_async_else_fn(self):
    """Sync predicate falsy -> async else_ fn."""
    result = await Chain(5).if_(sync_falsy, then=sync_double).else_(async_add1).run()
    self.assertEqual(result, 6)

  async def test_sync_pred_sync_fn_then_async_next_step(self):
    """Sync if_ followed by async next step."""
    result = await Chain(5).if_(sync_truthy, then=sync_double).then(async_add1).run()
    # 5 -> 10 -> 11
    self.assertEqual(result, 11)

  async def test_if_is_last_step_returns_awaitable(self):
    """if_ is the last step and its fn returns an awaitable."""
    result = await Chain(3).if_(sync_truthy, then=async_double).run()
    self.assertEqual(result, 6)

  async def test_if_is_first_step_returns_awaitable(self):
    """if_ is the first step (no root) and returns an awaitable."""
    result = await Chain(7).if_(sync_truthy, then=async_add1).run()
    self.assertEqual(result, 8)


# ===========================================================================
# PART 2: Multi-method async transition chains (20+ tests)
# ===========================================================================

class TestMultiMethodTransitions(IsolatedAsyncioTestCase):

  async def test_then_sync_map_async_then_sync(self):
    """then(sync) -> map(async fn) -> then(sync): transition mid-map."""
    result = await (
      Chain([1, 2, 3])
      .then(sync_identity)
      .map(async_double)
      .then(sum)
      .run()
    )
    # [1,2,3] -> [2,4,6] -> 12
    self.assertEqual(result, 12)

  async def test_then_async_map_sync_filter_sync(self):
    """then(async) -> map(sync) -> filter(sync): transition at first step."""
    async def make_list(x):
      return [1, 2, 3, 4, 5]
    result = await (
      Chain(None)
      .then(make_list)
      .map(sync_double)
      .then(lambda lst: [x for x in lst if x > 4])
      .run()
    )
    self.assertEqual(result, [6, 8, 10])

  async def test_filter_sync_if_async_pred_then_sync(self):
    """filter(sync) -> if_(async pred) -> then(sync)."""
    result = await (
      Chain([1, 2, 3, 4])
      .filter(sync_is_even)
      .if_(async_truthy, then=sync_identity)
      .then(len)
      .run()
    )
    # [1,2,3,4] -> [2,4] -> if truthy -> [2,4] -> 2
    self.assertEqual(result, 2)

  async def test_gather_mixed_map_sync_then_sync(self):
    """gather(mixed) -> map(sync) -> then(sync)."""
    result = await (
      Chain(5)
      .gather(sync_add1, async_double)
      .map(sync_double)
      .then(sum)
      .run()
    )
    # gather: [6, 10] -> map double: [12, 20] -> sum: 32
    self.assertEqual(result, 32)

  async def test_with_sync_cm_async_body_then_sync_filter_sync(self):
    """with_(sync cm, async body) -> then(sync) -> filter(sync)."""
    cm = SyncCMLocal([1, 2, 3, 4])
    result = await (
      Chain(cm)
      .with_(async_identity)
      .filter(sync_is_even)
      .run()
    )
    self.assertEqual(result, [2, 4])

  async def test_chain_6_steps_transition_at_step_3(self):
    """6-step chain: sync, sync, async at step 3, then 3 sync."""
    result = await (
      Chain(1)
      .then(sync_add1)   # 2
      .then(sync_add1)   # 3
      .then(async_add1)  # 4 (transition)
      .then(sync_add1)   # 5
      .then(sync_add1)   # 6
      .then(sync_add1)   # 7
      .run()
    )
    self.assertEqual(result, 7)

  async def test_10_step_chain_transition_at_step_5(self):
    """10-step chain with transition at step 5."""
    result = await (
      Chain(0)
      .then(sync_add1)   # 1
      .then(sync_add1)   # 2
      .then(sync_add1)   # 3
      .then(sync_add1)   # 4
      .then(async_add1)  # 5 (transition)
      .then(sync_add1)   # 6
      .then(sync_add1)   # 7
      .then(sync_add1)   # 8
      .then(sync_add1)   # 9
      .then(sync_add1)   # 10
      .run()
    )
    self.assertEqual(result, 10)

  async def test_10_step_chain_transition_at_step_1(self):
    """10-step chain with transition at step 1."""
    result = await (
      Chain(0)
      .then(async_add1)  # 1 (transition)
      .then(sync_add1)   # 2
      .then(sync_add1)   # 3
      .then(sync_add1)   # 4
      .then(sync_add1)   # 5
      .then(sync_add1)   # 6
      .then(sync_add1)   # 7
      .then(sync_add1)   # 8
      .then(sync_add1)   # 9
      .then(sync_add1)   # 10
      .run()
    )
    self.assertEqual(result, 10)

  async def test_10_step_chain_transition_at_step_10(self):
    """10-step chain with transition at step 10."""
    result = await (
      Chain(0)
      .then(sync_add1)   # 1
      .then(sync_add1)   # 2
      .then(sync_add1)   # 3
      .then(sync_add1)   # 4
      .then(sync_add1)   # 5
      .then(sync_add1)   # 6
      .then(sync_add1)   # 7
      .then(sync_add1)   # 8
      .then(sync_add1)   # 9
      .then(async_add1)  # 10 (transition)
      .run()
    )
    self.assertEqual(result, 10)

  async def test_then_sync_then_sync_then_async_map_sync_if_sync(self):
    """then(sync) -> then(sync) -> then(async) -> map(sync) -> if_(sync)."""
    async def make_range(x):
      return list(range(x))
    result = await (
      Chain(4)
      .then(sync_add1)   # 5
      .then(sync_identity)  # 5
      .then(make_range)  # [0,1,2,3,4]
      .map(sync_double)  # [0,2,4,6,8]
      .if_(sync_truthy, then=sync_identity)
      .run()
    )
    self.assertEqual(result, [0, 2, 4, 6, 8])

  async def test_do_async_then_map_async_filter_sync(self):
    """do(async) -> then(sync) -> map(async) -> filter(sync)."""
    tracker = []
    async def track(x):
      tracker.append(x)

    result = await (
      Chain([1, 2, 3, 4])
      .do(track)
      .then(sync_identity)
      .map(async_double)
      .then(lambda lst: [x for x in lst if x > 4])
      .run()
    )
    self.assertEqual(result, [6, 8])
    self.assertEqual(tracker, [[1, 2, 3, 4]])

  async def test_if_async_pred_then_gather_mixed(self):
    """if_(async pred) -> gather(mixed sync/async)."""
    result = await (
      Chain(5)
      .if_(async_truthy, then=sync_identity)
      .gather(sync_add1, async_double)
      .run()
    )
    self.assertEqual(result, [6, 10])

  async def test_gather_then_map_then_filter(self):
    """gather -> map -> filter: multi-op async pipeline."""
    result = await (
      Chain(3)
      .gather(sync_add1, async_double, sync_identity)
      .map(sync_double)
      .then(lambda lst: [x for x in lst if x > 7])
      .run()
    )
    # gather: [4, 6, 3] -> map double: [8, 12, 6] -> filter >7: [8, 12]
    self.assertEqual(result, [8, 12])

  async def test_with_async_cm_then_if_then_map(self):
    """with_(async cm) -> if_(sync) -> map(sync)."""
    cm = AsyncCMLocal([1, 2, 3])
    result = await (
      Chain(cm)
      .with_(sync_identity)
      .if_(sync_truthy, then=sync_identity)
      .map(sync_double)
      .run()
    )
    self.assertEqual(result, [2, 4, 6])

  async def test_alternating_sync_async_across_methods(self):
    """Alternating sync/async across different method types."""
    cm = SyncCMLocal(10)
    result = await (
      Chain(cm)
      .with_(async_identity)    # 10 (async body)
      .then(sync_double)        # 20
      .then(async_add1)         # 21
      .then(sync_add1)          # 22
      .run()
    )
    self.assertEqual(result, 22)

  async def test_map_async_then_gather_sync(self):
    """map(async) -> then -> gather(all sync)."""
    result = await (
      Chain([1, 2, 3])
      .map(async_double)
      .then(sum)
      .gather(sync_add1, sync_double)
      .run()
    )
    # [1,2,3] -> [2,4,6] -> sum=12 -> [13, 24]
    self.assertEqual(result, [13, 24])

  async def test_filter_async_then_map_sync_then_async(self):
    """filter(async) -> map(sync) -> then(async)."""
    result = await (
      Chain([1, 2, 3, 4, 5, 6])
      .filter(async_is_even)
      .map(sync_double)
      .then(async_identity)
      .run()
    )
    # [2,4,6] -> [4,8,12]
    self.assertEqual(result, [4, 8, 12])

  async def test_many_dos_then_async_then(self):
    """Multiple sync dos -> async then: transition only at async then."""
    t1, t2 = [], []
    result = await (
      Chain(100)
      .do(lambda x: t1.append(x))
      .do(lambda x: t2.append(x))
      .then(async_add1)
      .run()
    )
    self.assertEqual(result, 101)
    self.assertEqual(t1, [100])
    self.assertEqual(t2, [100])

  async def test_if_else_async_else_fn_then_steps(self):
    """if_(falsy) -> else_(async) -> then(sync) -> then(sync)."""
    result = await (
      Chain(5)
      .if_(sync_falsy, then=sync_double).else_(async_add1)
      .then(sync_double)
      .then(sync_add1)
      .run()
    )
    # 5 -> else async_add1 -> 6 -> 12 -> 13
    self.assertEqual(result, 13)

  async def test_with_then_if_else_gather(self):
    """with -> then -> if/else -> gather: complex multi-method pipeline."""
    cm = SyncCMLocal(10)
    result = await (
      Chain(cm)
      .with_(async_identity)
      .then(sync_double)       # 20
      .if_(sync_truthy, then=sync_identity)
      .gather(sync_add1, async_double)
      .run()
    )
    # 10 -> 20 -> if true -> 20 -> gather [21, 40]
    self.assertEqual(result, [21, 40])


# ===========================================================================
# PART 3: Exception during async transition (15+ tests)
# ===========================================================================

class TestExceptionDuringTransition(IsolatedAsyncioTestCase):

  async def test_async_fn_raises_during_transition(self):
    """Async fn raises during the transition point (first awaitable)."""
    with self.assertRaises(ValueError):
      await Chain(1).then(async_raise_value_error).run()

  async def test_async_fn_raises_after_transition(self):
    """Async fn raises AFTER transition (in _run_async, not the first awaitable)."""
    with self.assertRaises(ValueError):
      await Chain(1).then(async_add1).then(async_raise_value_error).run()

  async def test_except_handler_for_exception_during_transition(self):
    """except_ handler catches exception raised during async transition."""
    result = await (
      Chain(1)
      .then(async_raise_value_error)
      .except_(lambda rv, e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_except_handler_after_sync_async_transition(self):
    """except_ handler works when exception is in _run_async after transition."""
    result = await (
      Chain(1)
      .then(async_add1)
      .then(sync_raise_value_error)
      .except_(lambda rv, e: 'caught_after')
      .run()
    )
    self.assertEqual(result, 'caught_after')

  async def test_finally_handler_runs_after_async_transition(self):
    """finally_ handler runs after async transition (success path)."""
    tracker = []
    result = await (
      Chain(10)
      .then(async_double)
      .finally_(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 20)
    self.assertEqual(tracker, [10])

  async def test_finally_handler_runs_after_async_exception(self):
    """finally_ handler runs even when async chain raises."""
    tracker = []
    with self.assertRaises(ValueError):
      await (
        Chain(10)
        .then(async_raise_value_error)
        .finally_(lambda x: tracker.append('finally_ran'))
        .run()
      )
    self.assertEqual(tracker, ['finally_ran'])

  async def test_return_during_async_chain(self):
    """return_() during async chain execution exits early."""
    result = await (
      Chain(5)
      .then(async_add1)
      .then(lambda x: Chain.return_(x * 100))
      .then(sync_add1)  # should not execute
      .run()
    )
    self.assertEqual(result, 600)

  async def test_break_during_async_map(self):
    """break_() during async map iteration."""
    counter = 0
    async def count_and_maybe_break(x):
      nonlocal counter
      counter += 1
      if x >= 3:
        Chain.break_()
      return x * 10

    result = await Chain([1, 2, 3, 4, 5]).map(count_and_maybe_break).run()
    self.assertEqual(result, [10, 20])
    self.assertEqual(counter, 3)

  async def test_exception_in_async_predicate_of_if(self):
    """Exception raised by async predicate of if_."""
    async def bad_pred(x):
      raise RuntimeError('pred error')
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(5).if_(bad_pred, then=sync_double).run()
    self.assertIn('pred error', str(ctx.exception))

  async def test_exception_in_async_else_fn(self):
    """Exception raised by async else_ fn."""
    async def bad_else(x):
      raise RuntimeError('else error')
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(5).if_(sync_falsy, then=sync_double).else_(bad_else).run()
    self.assertIn('else error', str(ctx.exception))

  async def test_exception_in_async_map_fn(self):
    """Exception raised by async map fn."""
    async def bad_fn(x):
      if x == 2:
        raise RuntimeError('map error')
      return x

    with self.assertRaises(RuntimeError) as ctx:
      await Chain([1, 2, 3]).map(bad_fn).run()
    self.assertIn('map error', str(ctx.exception))

  async def test_exception_in_async_filter_fn(self):
    """Exception raised by async filter fn."""
    async def bad_filter(x):
      if x == 3:
        raise RuntimeError('filter error')
      return True

    with self.assertRaises(RuntimeError) as ctx:
      await Chain([1, 2, 3]).filter(bad_filter).run()
    self.assertIn('filter error', str(ctx.exception))

  async def test_exception_in_async_with_body(self):
    """Exception in async body of with_."""
    cm = SyncCMLocal('ctx')
    async def bad_body(ctx):
      raise RuntimeError('body error')
    with self.assertRaises(RuntimeError):
      await Chain(cm).with_(bad_body).run()
    self.assertTrue(cm.exited)

  async def test_except_with_exception_type_filter(self):
    """except_ with specific exception type during async transition."""
    result = await (
      Chain(1)
      .then(async_raise_value_error)
      .except_(lambda rv, e: 'caught_value', exceptions=ValueError)
      .run()
    )
    self.assertEqual(result, 'caught_value')

  async def test_except_does_not_catch_wrong_type(self):
    """except_ configured for ValueError doesn't catch RuntimeError."""
    with self.assertRaises(RuntimeError):
      await (
        Chain(1)
        .then(async_raise_runtime)
        .except_(lambda e: 'caught', exceptions=ValueError)
        .run()
      )

  async def test_except_and_finally_both_fire_on_async_exception(self):
    """Both except_ and finally_ fire when async chain raises."""
    tracker = []
    result = await (
      Chain(1)
      .then(async_raise_value_error)
      .except_(lambda rv, e: (tracker.append('except'), 'recovered')[1])
      .finally_(lambda x: tracker.append('finally'))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertIn('except', tracker)
    self.assertIn('finally', tracker)

  async def test_return_before_async_step(self):
    """return_() in sync step before any async step -- stays sync."""
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(x * 2))
      .then(async_add1)  # should not execute
      .run()
    )
    self.assertEqual(result, 10)


# ===========================================================================
# PART 4: Nested chain async transitions (10+ tests)
# ===========================================================================

class TestNestedChainTransitions(IsolatedAsyncioTestCase):

  async def test_outer_sync_inner_chain_has_async_step(self):
    """Outer chain sync, inner chain has async step."""
    inner = Chain().then(async_double)
    result = await Chain(5).then(inner).run()
    self.assertEqual(result, 10)

  async def test_outer_async_inner_chain_all_sync(self):
    """Outer chain async (transition at first step), inner chain all sync."""
    inner = Chain().then(sync_double)
    result = await Chain(5).then(async_add1).then(inner).run()
    # 5 -> 6 (async) -> 12 (sync inner)
    self.assertEqual(result, 12)

  async def test_both_outer_and_inner_have_transitions(self):
    """Both outer and inner chains have async transitions."""
    inner = Chain().then(async_add1)
    result = await Chain(1).then(async_add1).then(inner).then(sync_double).run()
    # 1 -> 2 (async) -> 3 (inner async) -> 6
    self.assertEqual(result, 6)

  async def test_frozen_inner_chain_with_async_step(self):
    """Frozen inner chain with async step."""
    inner = Chain().then(async_double).freeze()
    result = await Chain(5).then(inner).run()
    self.assertEqual(result, 10)

  async def test_nested_chain_in_if_fn_with_async_step(self):
    """Nested chain used as if_ fn with async step."""
    inner = Chain().then(async_double)
    result = await Chain(5).if_(sync_truthy, then=inner).run()
    self.assertEqual(result, 10)

  async def test_nested_chain_in_map_fn_with_async_step(self):
    """Nested chain as map fn with async step."""
    inner = Chain().then(async_double)
    result = await Chain([1, 2, 3]).map(inner).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_deeply_nested_chains(self):
    """Three levels of nesting, innermost has async step."""
    inner_most = Chain().then(async_add1)
    middle = Chain().then(inner_most)
    result = await Chain(1).then(middle).run()
    # 1 -> middle -> inner_most -> async_add1 -> 2
    self.assertEqual(result, 2)

  async def test_frozen_chain_nested_in_sync_outer(self):
    """Frozen chain with async step nested in otherwise sync outer."""
    frozen = Chain().then(async_double).freeze()
    result = await Chain(7).then(sync_add1).then(frozen).then(sync_add1).run()
    # 7 -> 8 -> frozen(8) = 16 -> 17
    self.assertEqual(result, 17)

  async def test_nested_chain_with_except_handler(self):
    """Nested chain that raises, with except_ on outer chain."""
    inner = Chain().then(async_raise_value_error)
    result = await (
      Chain(1)
      .then(inner)
      .except_(lambda rv, e: 'outer_caught')
      .run()
    )
    self.assertEqual(result, 'outer_caught')

  async def test_nested_chain_in_else_branch(self):
    """Nested chain used in else_ branch with async step."""
    inner = Chain().then(async_double)
    result = await Chain(5).if_(sync_falsy, then=sync_identity).else_(inner).run()
    self.assertEqual(result, 10)

  async def test_outer_sync_inner_frozen_async_then_more_sync(self):
    """Outer sync -> frozen inner (async) -> more sync steps."""
    frozen = Chain().then(async_add1).freeze()
    result = await (
      Chain(10)
      .then(sync_add1)    # 11
      .then(frozen)       # 12 (async transition in frozen)
      .then(sync_double)  # 24
      .then(sync_add1)    # 25
      .run()
    )
    self.assertEqual(result, 25)


# ===========================================================================
# PART 5: iterate async transitions (8+ tests)
# ===========================================================================

class TestIterateAsyncTransitions(IsolatedAsyncioTestCase):

  async def test_sync_chain_iterate_consumed_with_async_for(self):
    """Sync chain.iterate() consumed with async for."""
    gen = Chain([1, 2, 3]).iterate()
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_chain_transitions_to_async_then_iterate(self):
    """Chain that transitions to async -> iterate()."""
    async def make_list(x):
      return [10, 20, 30]
    gen = Chain(None).then(make_list).iterate()
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [10, 20, 30])

  async def test_iterate_with_async_fn_via_aiter(self):
    """iterate(async_fn) consumed via __aiter__."""
    gen = Chain([1, 2, 3]).iterate(async_double)
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [2, 4, 6])

  async def test_iterate_do_with_async_fn_via_aiter(self):
    """iterate_do with async fn via __aiter__: yields original items."""
    tracker = []
    async def track(x):
      tracker.append(x * 10)
      return 'discarded'
    gen = Chain([1, 2, 3]).iterate_do(track)
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [10, 20, 30])

  async def test_iterate_sync_fn_via_sync_iter(self):
    """iterate() with sync fn consumed via __iter__ stays sync."""
    gen = Chain([1, 2, 3]).iterate(sync_double)
    result = list(gen)
    self.assertEqual(result, [2, 4, 6])

  async def test_iterate_with_run_value(self):
    """iterate() called with a run value via __call__."""
    gen = Chain().then(lambda x: list(range(x))).iterate()
    result = []
    async for item in gen(5):
      result.append(item)
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_iterate_async_chain_async_fn(self):
    """Chain with async step -> iterate with async fn."""
    async def make_list(x):
      return [10, 20, 30]
    gen = Chain(None).then(make_list).iterate(async_double)
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [20, 40, 60])

  async def test_iterate_do_sync_chain_async_fn(self):
    """iterate_do with sync chain and async fn via __aiter__."""
    tracker = []
    async def track(x):
      tracker.append(x)
    gen = Chain([10, 20]).iterate_do(track)
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [10, 20])
    self.assertEqual(tracker, [10, 20])


# ===========================================================================
# Additional edge case tests to reach 100+ total
# ===========================================================================

class TestEdgeCaseTransitions(IsolatedAsyncioTestCase):

  async def test_chain_root_is_async(self):
    """Root value of chain is an async callable."""
    async def root():
      return 42
    result = await Chain(root, ...).then(sync_double).run()
    self.assertEqual(result, 84)

  async def test_run_with_async_value(self):
    """run() called with async fn as root value."""
    result = await Chain().then(async_double).run(5)
    self.assertEqual(result, 10)

  async def test_chain_call_with_async_transition(self):
    """Chain.__call__() triggers async transition."""
    c = Chain().then(async_double)
    result = await c(5)
    self.assertEqual(result, 10)

  async def test_frozen_chain_call_with_async(self):
    """Frozen chain __call__() with async step."""
    c = Chain().then(async_double).freeze()
    result = await c(5)
    self.assertEqual(result, 10)

  async def test_gather_single_async_fn(self):
    """Gather with a single async fn."""
    result = await Chain(5).gather(async_double).run()
    self.assertEqual(result, [10])

  async def test_gather_many_fns_mixed(self):
    """Gather with many fns, mixed sync/async."""
    result = await (
      Chain(2)
      .gather(sync_add1, async_double, sync_double, async_add1, sync_identity)
      .run()
    )
    self.assertEqual(result, [3, 4, 4, 3, 2])

  async def test_if_async_pred_truthy_then_sync_fn(self):
    """if_ with async predicate that returns truthy, sync fn body."""
    async def is_positive(x):
      return x > 0
    result = await Chain(5).if_(is_positive, then=sync_double).run()
    self.assertEqual(result, 10)

  async def test_if_async_pred_falsy_with_else(self):
    """if_ with async predicate falsy -> else_ (sync)."""
    async def is_negative(x):
      return x < 0
    result = await Chain(5).if_(is_negative, then=sync_double).else_(sync_add1).run()
    self.assertEqual(result, 6)

  async def test_if_async_pred_falsy_async_else(self):
    """if_ with async predicate falsy -> async else_."""
    async def is_negative(x):
      return x < 0
    result = await Chain(5).if_(is_negative, then=sync_double).else_(async_add1).run()
    self.assertEqual(result, 6)

  async def test_map_break_with_value(self):
    """map with break_(value) during async iteration."""
    async def fn(x):
      if x == 3:
        Chain.break_('early_exit')
      return x * 10

    result = await Chain([1, 2, 3, 4]).map(fn).run()
    self.assertEqual(result, 'early_exit')

  async def test_return_with_value_from_async_chain(self):
    """return_(value) from async chain returns that value."""
    result = await (
      Chain(5)
      .then(async_double)
      .then(lambda x: Chain.return_(x + 100))
      .then(sync_add1)  # should not execute
      .run()
    )
    self.assertEqual(result, 110)

  async def test_map_empty_list_async_fn(self):
    """map with empty list and async fn -> no transition, returns []."""
    result = Chain([]).map(async_double).run()
    # empty list means no iteration, stays sync
    self.assertEqual(result, [])

  async def test_filter_empty_list_async_fn(self):
    """filter with empty list and async fn -> no transition, returns []."""
    result = Chain([]).filter(async_is_even).run()
    self.assertEqual(result, [])

  async def test_with_async_cm_raises_in_body(self):
    """Async CM with exception in body: __aexit__ still called."""
    cm = AsyncCMLocal('val')
    with self.assertRaises(RuntimeError):
      await Chain(cm).with_(lambda ctx: (_ for _ in ()).throw(RuntimeError('body fail'))).run()
    self.assertTrue(cm.exited)

  async def test_multiple_async_transitions_in_run_async(self):
    """Multiple awaitables encountered inside _run_async."""
    result = await (
      Chain(1)
      .then(async_add1)   # 2 (transition to _run_async)
      .then(async_add1)   # 3 (already in _run_async)
      .then(async_add1)   # 4
      .then(async_add1)   # 5
      .run()
    )
    self.assertEqual(result, 5)

  async def test_sync_chain_no_transition(self):
    """Purely sync chain stays sync (control test)."""
    result = Chain(1).then(sync_add1).then(sync_double).run()
    # 1 -> 2 -> 4
    self.assertEqual(result, 4)

  async def test_async_do_does_not_change_value(self):
    """Async do between thens doesn't affect pipeline value."""
    result = await (
      Chain(10)
      .then(sync_double)  # 20
      .do(async_noop)     # transition, but value stays 20
      .then(sync_add1)    # 21
      .run()
    )
    self.assertEqual(result, 21)

  async def test_if_async_fn_body_result_replaces_value(self):
    """if_ with async fn body: result replaces current value."""
    result = await Chain(5).if_(sync_truthy, then=async_double).then(sync_add1).run()
    # 5 -> async_double -> 10 -> 11
    self.assertEqual(result, 11)

  async def test_chain_with_only_root_async(self):
    """Chain with only an async root callable."""
    async def root():
      return 99
    result = await Chain(root, ...).run()
    self.assertEqual(result, 99)

  async def test_chain_with_none_root_async_then(self):
    """Chain(None) -> async then."""
    result = await Chain(None).then(async_identity).run()
    self.assertIsNone(result)


class TestTransitionValuePreservation(IsolatedAsyncioTestCase):
  """Verify that values are correctly preserved across transitions."""

  async def test_none_preserved_across_async_transition(self):
    """None value is preserved (not confused with Null) across transition."""
    result = await Chain(None).then(async_identity).run()
    self.assertIsNone(result)

  async def test_false_preserved_across_async_transition(self):
    """False value preserved across transition."""
    result = await Chain(False).then(async_identity).run()
    self.assertIs(result, False)

  async def test_zero_preserved_across_async_transition(self):
    """Zero value preserved across transition."""
    result = await Chain(0).then(async_identity).run()
    self.assertEqual(result, 0)

  async def test_empty_string_preserved_across_transition(self):
    """Empty string preserved across transition."""
    result = await Chain('').then(async_identity).run()
    self.assertEqual(result, '')

  async def test_empty_list_preserved_across_transition(self):
    """Empty list preserved across transition."""
    result = await Chain([]).then(async_identity).run()
    self.assertEqual(result, [])

  async def test_complex_object_preserved_across_transition(self):
    """Complex object identity preserved across transition."""
    obj = {'key': [1, 2, {'nested': True}]}
    result = await Chain(obj).then(async_identity).run()
    self.assertIs(result, obj)


class TestTransitionWithDecorator(IsolatedAsyncioTestCase):
  """Test async transitions when chain is used as decorator."""

  async def test_decorator_with_async_step(self):
    """Decorated fn triggers async transition."""
    @Chain().then(async_double).decorator()
    def my_fn(x):
      return x

    result = await my_fn(5)
    self.assertEqual(result, 10)

  async def test_decorator_with_async_step_and_sync_after(self):
    """Decorated fn: async transition then sync step."""
    @Chain().then(async_add1).then(sync_double).decorator()
    def my_fn(x):
      return x

    result = await my_fn(5)
    # my_fn(5)=5 -> async_add1=6 -> sync_double=12
    self.assertEqual(result, 12)


class TestTransitionWithFreeze(IsolatedAsyncioTestCase):
  """Test async transitions with frozen chains."""

  async def test_frozen_chain_with_async_step_reused(self):
    """Frozen chain with async step can be reused."""
    frozen = Chain().then(async_double).freeze()
    r1 = await frozen(5)
    r2 = await frozen(10)
    self.assertEqual(r1, 10)
    self.assertEqual(r2, 20)

  async def test_frozen_chain_complex_pipeline(self):
    """Frozen chain with multiple async/sync steps."""
    frozen = (
      Chain()
      .then(sync_add1)
      .then(async_double)
      .then(sync_add1)
      .freeze()
    )
    result = await frozen(5)
    # 5 -> 6 -> 12 -> 13
    self.assertEqual(result, 13)

  async def test_frozen_chain_with_map_async(self):
    """Frozen chain containing async map."""
    frozen = (
      Chain()
      .then(lambda x: list(range(x)))
      .map(async_double)
      .freeze()
    )
    result = await frozen(4)
    self.assertEqual(result, [0, 2, 4, 6])


class TestTransitionWithExceptAndFinally(IsolatedAsyncioTestCase):
  """More tests for except/finally during transitions."""

  async def test_async_except_handler(self):
    """Async except_ handler fn."""
    async def async_handler(rv, e):
      return 'async_caught'
    result = await (
      Chain(1)
      .then(async_raise_value_error)
      .except_(async_handler)
      .run()
    )
    self.assertEqual(result, 'async_caught')

  async def test_async_finally_handler(self):
    """Async finally_ handler fn."""
    tracker = []
    async def async_cleanup(root_val):
      tracker.append(f'cleaned_{root_val}')
    result = await (
      Chain(10)
      .then(async_double)
      .finally_(async_cleanup)
      .run()
    )
    self.assertEqual(result, 20)
    self.assertEqual(tracker, ['cleaned_10'])

  async def test_sync_except_catches_error_from_async_step(self):
    """Sync except_ handler catches error from async step."""
    result = await (
      Chain(1)
      .then(sync_add1)
      .then(async_raise_value_error)
      .except_(lambda rv, e: f'caught: {e}')
      .run()
    )
    self.assertEqual(result, 'caught: async value error')

  async def test_finally_runs_on_success_with_correct_root_value(self):
    """finally_ receives the root value on success path."""
    tracker = []
    result = await (
      Chain(5)
      .then(async_double)   # 10
      .then(sync_add1)      # 11
      .finally_(lambda root: tracker.append(root))
      .run()
    )
    self.assertEqual(result, 11)
    # Root value is the result of evaluating the root link (5)
    self.assertEqual(tracker, [5])


if __name__ == '__main__':
  unittest.main()
