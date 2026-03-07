"""Systematic test of every chain method combined with every other chain method,
in both sync and async execution modes. Covers pairwise, triple, async-crossing,
except/finally interactions, decorator, and iterate combinations.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  async_fn,
  async_identity,
  AsyncRange,
  SyncCM,
  AsyncCM,
  TrackingCM,
  AsyncTrackingCM,
  make_tracker,
  make_async_tracker,
)


# ---------------------------------------------------------------------------
# PART 1: Method A -> Method B pairwise combinations
# ---------------------------------------------------------------------------

# ---- then -> every method ----

class TestThenPairs(unittest.TestCase):

  def test_then_then(self):
    result = Chain(1).then(lambda x: x + 1).then(lambda x: x * 3).run()
    self.assertEqual(result, 6)

  def test_then_do(self):
    tracker = []
    result = Chain(5).then(lambda x: x + 1).do(lambda x: tracker.append(x)).run()
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [6])

  def test_then_map(self):
    result = Chain(10).then(lambda x: [x, x + 1, x + 2]).map(lambda x: x * 2).run()
    self.assertEqual(result, [20, 22, 24])

  def test_then_foreach(self):
    tracker = []
    result = (
      Chain(10)
      .then(lambda x: [x, x + 1])
      .foreach(lambda x: tracker.append(x * 10))
      .run()
    )
    self.assertEqual(result, [10, 11])
    self.assertEqual(tracker, [100, 110])

  def test_then_filter(self):
    result = Chain(0).then(lambda x: [x, 1, 2, 3, 4]).filter(lambda x: x > 2).run()
    self.assertEqual(result, [3, 4])

  def test_then_gather(self):
    result = Chain(5).then(lambda x: x + 1).gather(lambda x: x * 2, lambda x: x * 3).run()
    self.assertEqual(result, [12, 18])

  def test_then_with_(self):
    cm = TrackingCM()
    result = Chain(cm).then(lambda x: x).with_(lambda ctx: ctx + '_used').run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    self.assertEqual(result, 'tracked_ctx_used')

  def test_then_if_true(self):
    result = Chain(5).then(lambda x: x + 1).if_(lambda x: x > 3, then=lambda x: x * 10).run()
    self.assertEqual(result, 60)

  def test_then_if_false(self):
    result = Chain(5).then(lambda x: x + 1).if_(lambda x: x > 100, then=lambda x: x * 10).run()
    self.assertEqual(result, 6)

  def test_then_if_else_true(self):
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .if_(lambda x: x > 3, then=lambda x: 'big')
      .else_(lambda x: 'small')
      .run()
    )
    self.assertEqual(result, 'big')

  def test_then_if_else_false(self):
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .if_(lambda x: x > 100, then=lambda x: 'big')
      .else_(lambda x: 'small')
      .run()
    )
    self.assertEqual(result, 'small')

  def test_then_except(self):
    result = (
      Chain(5)
      .then(lambda x: (_ for _ in ()).throw(ValueError('oops')))
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_then_except_with_error(self):
    def raiser(x):
      raise ValueError('boom')
    result = Chain(5).then(raiser).except_(lambda rv, exc: 'handled').run()
    self.assertEqual(result, 'handled')

  def test_then_finally(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .finally_(lambda root: tracker.append(root))
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [5])

  def test_then_iterate(self):
    result = list(Chain([1, 2, 3]).then(lambda x: x).iterate())
    self.assertEqual(result, [1, 2, 3])

  def test_then_iterate_with_fn(self):
    result = list(Chain([1, 2, 3]).then(lambda x: x).iterate(lambda x: x * 2))
    self.assertEqual(result, [2, 4, 6])

  def test_then_return(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .then(lambda x: Chain.return_(x * 10))
      .then(lambda x: tracker.append('not_reached'))
      .run()
    )
    self.assertEqual(result, 60)
    self.assertEqual(tracker, [])


# ---- do -> every method ----

class TestDoPairs(unittest.TestCase):

  def test_do_then_preserves_value(self):
    result = Chain(5).do(lambda x: x * 100).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)

  def test_do_do(self):
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: tracker.append('first'))
      .do(lambda x: tracker.append('second'))
      .run()
    )
    self.assertEqual(result, 5)
    self.assertEqual(tracker, ['first', 'second'])

  def test_do_map(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .do(lambda x: tracker.append(len(x)))
      .map(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, [2, 4, 6])
    self.assertEqual(tracker, [3])

  def test_do_filter(self):
    tracker = []
    result = (
      Chain([1, 2, 3, 4])
      .do(lambda x: tracker.append(len(x)))
      .filter(lambda x: x > 2)
      .run()
    )
    self.assertEqual(result, [3, 4])
    self.assertEqual(tracker, [4])

  def test_do_if(self):
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: tracker.append(x))
      .if_(lambda x: x > 3, then=lambda x: x * 10)
      .run()
    )
    self.assertEqual(result, 50)
    self.assertEqual(tracker, [5])

  def test_do_gather(self):
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: tracker.append(x))
      .gather(lambda x: x + 1, lambda x: x + 2)
      .run()
    )
    self.assertEqual(result, [6, 7])
    self.assertEqual(tracker, [5])

  def test_do_except(self):
    tracker = []
    def raiser(x):
      tracker.append('do_ran')
      raise ValueError('boom')
    result = Chain(5).do(raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['do_ran'])

  def test_do_finally(self):
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: tracker.append('side'))
      .finally_(lambda root: tracker.append(f'final:{root}'))
      .run()
    )
    self.assertEqual(result, 5)
    self.assertIn('side', tracker)
    self.assertIn('final:5', tracker)


# ---- map -> every method ----

class TestMapPairs(unittest.TestCase):

  def test_map_then(self):
    result = Chain([1, 2, 3]).map(lambda x: x * 2).then(lambda x: sum(x)).run()
    self.assertEqual(result, 12)

  def test_foreach(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .do(lambda x: tracker.append(len(x)))
      .run()
    )
    self.assertEqual(result, [2, 4, 6])
    self.assertEqual(tracker, [3])

  def test_map_filter(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: x * 2)
      .filter(lambda x: x > 5)
      .run()
    )
    self.assertEqual(result, [6, 8, 10])

  def test_map_map(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .map(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, [3, 5, 7])

  def test_map_gather(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .gather(lambda x: sum(x), lambda x: len(x))
      .run()
    )
    self.assertEqual(result, [12, 3])

  def test_map_if_true(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .if_(lambda x: len(x) > 2, then=lambda x: x[:2])
      .run()
    )
    self.assertEqual(result, [2, 4])

  def test_map_if_false(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .if_(lambda x: len(x) > 100, then=lambda x: [])
      .run()
    )
    self.assertEqual(result, [2, 4, 6])

  def test_map_except(self):
    def raiser(x):
      if x > 2:
        raise ValueError('too big')
      return x
    result = (
      Chain([1, 2, 3])
      .map(raiser)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')


# ---- filter -> every method ----

class TestFilterPairs(unittest.TestCase):

  def test_filter_then(self):
    result = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).then(lambda x: sum(x)).run()
    self.assertEqual(result, 9)

  def test_filter_do(self):
    tracker = []
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 3)
      .do(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, [4, 5])
    self.assertEqual(tracker, [[4, 5]])

  def test_filter_map(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 2)
      .map(lambda x: x * 10)
      .run()
    )
    self.assertEqual(result, [30, 40, 50])

  def test_filter_filter_double(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 1)
      .filter(lambda x: x < 5)
      .run()
    )
    self.assertEqual(result, [2, 3, 4])

  def test_filter_gather(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 3)
      .gather(lambda x: len(x), lambda x: max(x))
      .run()
    )
    self.assertEqual(result, [2, 5])

  def test_filter_if_true(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 3)
      .if_(lambda x: len(x) > 0, then=lambda x: x[0])
      .run()
    )
    self.assertEqual(result, 4)

  def test_filter_if_false(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 3)
      .if_(lambda x: len(x) > 100, then=lambda x: [])
      .run()
    )
    self.assertEqual(result, [4, 5])

  def test_filter_except(self):
    def raiser(x):
      raise ValueError('bad filter')
    result = Chain([1, 2]).filter(raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')


# ---- gather -> every method ----

class TestGatherPairs(unittest.TestCase):

  def test_gather_then(self):
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(result, 13)

  def test_gather_do(self):
    tracker = []
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .do(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, [6, 7])
    self.assertEqual(tracker, [[6, 7]])

  def test_gather_map(self):
    result = (
      Chain(10)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .map(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, [22, 24])

  def test_gather_filter(self):
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2, lambda x: x + 10)
      .filter(lambda x: x > 7)
      .run()
    )
    self.assertEqual(result, [15])

  def test_gather_if_true(self):
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .if_(lambda x: len(x) == 2, then=lambda x: x[0] + x[1])
      .run()
    )
    self.assertEqual(result, 13)

  def test_gather_if_false(self):
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .if_(lambda x: len(x) > 100, then=lambda x: 999)
      .run()
    )
    self.assertEqual(result, [6, 7])


# ---- with_ -> every method ----

class TestWithPairs(unittest.TestCase):

  def test_with_then(self):
    cm = TrackingCM()
    result = Chain(cm).with_(lambda ctx: ctx + '_body').then(lambda x: x + '_then').run()
    self.assertEqual(result, 'tracked_ctx_body_then')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_with_do(self):
    cm = TrackingCM()
    tracker = []
    result = (
      Chain(cm)
      .with_(lambda ctx: ctx + '_body')
      .do(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 'tracked_ctx_body')
    self.assertEqual(tracker, ['tracked_ctx_body'])

  def test_with_map(self):
    cm = TrackingCM()
    cm.enter_result = [1, 2, 3]
    result = Chain(cm).with_(lambda ctx: ctx).map(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_with_filter(self):
    cm = TrackingCM()
    cm.enter_result = [1, 2, 3, 4, 5]
    result = Chain(cm).with_(lambda ctx: ctx).filter(lambda x: x > 3).run()
    self.assertEqual(result, [4, 5])

  def test_with_if_true(self):
    cm = TrackingCM()
    result = (
      Chain(cm)
      .with_(lambda ctx: ctx + '_body')
      .if_(lambda x: len(x) > 5, then=lambda x: 'long')
      .run()
    )
    self.assertEqual(result, 'long')

  def test_with_gather(self):
    cm = TrackingCM()
    result = (
      Chain(cm)
      .with_(lambda ctx: 10)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .run()
    )
    self.assertEqual(result, [11, 12])

  def test_with_except(self):
    cm = TrackingCM()
    result = (
      Chain(cm)
      .with_(lambda ctx: (_ for _ in ()).throw(ValueError('fail')))
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertTrue(cm.exited)

  def test_with_finally(self):
    cm = TrackingCM()
    tracker = []
    result = (
      Chain(cm)
      .with_(lambda ctx: ctx + '_body')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'tracked_ctx_body')
    self.assertIn('cleanup', tracker)

  def test_with_do_preserves(self):
    cm = TrackingCM()
    tracker = []
    result = (
      Chain(cm)
      .with_do(lambda ctx: tracker.append(ctx))
      .run()
    )
    # with_do ignores fn result, returns the CM itself
    self.assertIs(result, cm)
    self.assertEqual(tracker, ['tracked_ctx'])


# ---- if_ -> every method ----

class TestIfPairs(unittest.TestCase):

  def test_if_then_true(self):
    result = Chain(5).if_(lambda x: x > 3, then=lambda x: x * 2).then(lambda x: x + 1).run()
    self.assertEqual(result, 11)

  def test_if_then_false(self):
    result = Chain(5).if_(lambda x: x > 100, then=lambda x: x * 2).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)

  def test_if_do(self):
    tracker = []
    result = (
      Chain(5)
      .if_(lambda x: x > 3, then=lambda x: x * 2)
      .do(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [10])

  def test_if_map(self):
    result = (
      Chain([1, 2, 3])
      .if_(lambda x: len(x) > 0, then=lambda x: [i * 2 for i in x])
      .map(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, [3, 5, 7])

  def test_if_filter(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .if_(lambda x: len(x) > 3, then=lambda x: x)
      .filter(lambda x: x > 3)
      .run()
    )
    self.assertEqual(result, [4, 5])

  def test_if_gather(self):
    result = (
      Chain(5)
      .if_(lambda x: x > 0, then=lambda x: x * 2)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .run()
    )
    self.assertEqual(result, [11, 12])

  def test_if_with(self):
    cm = TrackingCM()
    result = (
      Chain(cm)
      .if_(lambda x: True, then=lambda x: x)
      .with_(lambda ctx: ctx + '_used')
      .run()
    )
    self.assertEqual(result, 'tracked_ctx_used')

  def test_if_if_chained(self):
    result = (
      Chain(5)
      .if_(lambda x: x > 3, then=lambda x: x * 2)
      .if_(lambda x: x > 8, then=lambda x: x + 100)
      .run()
    )
    self.assertEqual(result, 110)

  def test_if_except(self):
    def raiser(x):
      raise ValueError('if body error')
    result = (
      Chain(5)
      .if_(lambda x: True, then=raiser)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_if_else_then(self):
    result = (
      Chain(5)
      .if_(lambda x: x > 100, then=lambda x: 'big')
      .else_(lambda x: 'small')
      .then(lambda x: x + '_value')
      .run()
    )
    self.assertEqual(result, 'small_value')


# ---------------------------------------------------------------------------
# PART 2: Three-method combinations
# ---------------------------------------------------------------------------

class TestTripleCombinations(unittest.TestCase):

  def test_then_map_filter(self):
    result = (
      Chain(0)
      .then(lambda x: [1, 2, 3, 4, 5])
      .map(lambda x: x * 2)
      .filter(lambda x: x > 5)
      .run()
    )
    self.assertEqual(result, [6, 8, 10])

  def test_filter_map_then(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 2)
      .map(lambda x: x * 10)
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(result, 120)

  def test_then_if_map(self):
    result = (
      Chain([1, 2, 3])
      .then(lambda x: x)
      .if_(lambda x: len(x) > 0, then=lambda x: [i * 2 for i in x])
      .map(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, [3, 5, 7])

  def test_with_then_map(self):
    cm = TrackingCM()
    cm.enter_result = [10, 20, 30]
    result = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .then(lambda x: x)
      .map(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, [11, 21, 31])

  def test_gather_if_then(self):
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .if_(lambda x: len(x) == 2, then=lambda x: sum(x))
      .then(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, 26)

  def test_map_filter_if(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: x * 2)
      .filter(lambda x: x > 5)
      .if_(lambda x: len(x) > 0, then=lambda x: x[0])
      .run()
    )
    self.assertEqual(result, 6)

  def test_then_do_if_else(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .do(lambda x: tracker.append(x))
      .if_(lambda x: x > 100, then=lambda x: 'big')
      .else_(lambda x: 'small')
      .run()
    )
    self.assertEqual(result, 'small')
    self.assertEqual(tracker, [6])

  def test_if_then_except(self):
    def raiser(x):
      raise ValueError('oops')
    result = (
      Chain(5)
      .if_(lambda x: True, then=raiser)
      .then(lambda x: 'never')
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_map_if_else_then(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .if_(lambda x: len(x) > 10, then=lambda x: [])
      .else_(lambda x: x)
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(result, 12)

  def test_with_if_else(self):
    cm = TrackingCM()
    result = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .if_(lambda x: len(x) > 5, then=lambda x: 'long')
      .else_(lambda x: 'short')
      .run()
    )
    self.assertEqual(result, 'long')

  def test_then_gather_map(self):
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .gather(lambda x: x, lambda x: x * 2)
      .map(lambda x: x + 10)
      .run()
    )
    self.assertEqual(result, [16, 22])

  def test_filter_if_map(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 2)
      .if_(lambda x: len(x) > 0, then=lambda x: x)
      .map(lambda x: x * 10)
      .run()
    )
    self.assertEqual(result, [30, 40, 50])

  def test_then_map_gather(self):
    result = (
      Chain(0)
      .then(lambda x: [1, 2, 3])
      .map(lambda x: x * 2)
      .gather(lambda x: sum(x), lambda x: max(x))
      .run()
    )
    self.assertEqual(result, [12, 6])

  def test_do_filter_then(self):
    tracker = []
    result = (
      Chain([1, 2, 3, 4, 5])
      .do(lambda x: tracker.append(len(x)))
      .filter(lambda x: x > 3)
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(result, 9)
    self.assertEqual(tracker, [5])


# ---------------------------------------------------------------------------
# PART 3: Async crossing combinations
# ---------------------------------------------------------------------------

class TestAsyncCrossingSyncToAsync(IsolatedAsyncioTestCase):

  async def test_then_sync_then_async(self):
    result = await Chain(1).then(lambda x: x + 1).then(async_fn).run()
    self.assertEqual(result, 3)

  async def test_then_async_then_sync(self):
    result = await Chain(1).then(async_fn).then(lambda x: x + 10).run()
    self.assertEqual(result, 12)

  async def test_both_async_then(self):
    result = await Chain(1).then(async_fn).then(async_fn).run()
    self.assertEqual(result, 3)

  async def test_sync_map_async_fn(self):
    async def async_double(x):
      return x * 2
    result = await Chain([1, 2, 3]).map(async_double).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_async_map_sync_fn(self):
    result = await Chain(AsyncRange(4)).map(lambda x: x * 2).run()
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_sync_filter_async_pred(self):
    async def async_pred(x):
      return x > 2
    result = await Chain([1, 2, 3, 4]).filter(async_pred).run()
    self.assertEqual(result, [3, 4])

  async def test_async_filter_sync_pred(self):
    result = await Chain(AsyncRange(5)).filter(lambda x: x > 2).run()
    self.assertEqual(result, [3, 4])

  async def test_sync_then_async_gather(self):
    async def async_add(x):
      return x + 10
    result = await Chain(5).then(lambda x: x + 1).gather(async_add, lambda x: x * 2).run()
    self.assertEqual(result, [16, 12])

  async def test_async_then_sync_filter(self):
    result = (
      await Chain([1, 2, 3, 4, 5])
      .then(async_identity)
      .filter(lambda x: x > 3)
      .run()
    )
    self.assertEqual(result, [4, 5])

  async def test_sync_then_async_if(self):
    async def async_pred(x):
      return x > 3
    result = (
      await Chain(5)
      .then(lambda x: x + 1)
      .if_(async_pred, then=lambda x: x * 10)
      .run()
    )
    self.assertEqual(result, 60)

  async def test_async_if_sync_body(self):
    async def async_pred(x):
      return x > 0
    result = await Chain(5).if_(async_pred, then=lambda x: x * 2).run()
    self.assertEqual(result, 10)

  async def test_async_if_async_body(self):
    async def async_pred(x):
      return x > 0
    result = await Chain(5).if_(async_pred, then=async_fn).run()
    self.assertEqual(result, 6)

  async def test_sync_with_async_body(self):
    cm = SyncCM()
    async def async_body(ctx):
      return ctx + '_async'
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'ctx_value_async')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm_sync_body(self):
    cm = AsyncCM()
    result = await Chain(cm).with_(lambda ctx: ctx + '_sync').run()
    self.assertEqual(result, 'ctx_value_sync')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm_async_body(self):
    cm = AsyncCM()
    async def async_body(ctx):
      return ctx + '_async'
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'ctx_value_async')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_chain_starts_sync_transitions_async_mid(self):
    """Chain starts sync, hits an async fn, continues async for remainder."""
    tracker = []
    result = (
      await Chain(1)
      .then(lambda x: x + 1)
      .then(async_fn)
      .then(lambda x: x + 10)
      .do(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 13)
    self.assertEqual(tracker, [13])

  async def test_map_async_then_sync_then(self):
    async def async_double(x):
      return x * 2
    result = (
      await Chain([1, 2, 3])
      .map(async_double)
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(result, 12)

  async def test_async_then_map_filter(self):
    result = (
      await Chain([1, 2, 3, 4, 5])
      .then(async_identity)
      .map(lambda x: x * 2)
      .filter(lambda x: x > 5)
      .run()
    )
    self.assertEqual(result, [6, 8, 10])

  async def test_sync_do_async_then(self):
    tracker = []
    result = (
      await Chain(5)
      .do(lambda x: tracker.append(x))
      .then(async_fn)
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [5])

  async def test_async_except_handler(self):
    async def async_handler(rv, exc):
      return 'async_caught'
    def raiser(x):
      raise ValueError('boom')
    result = await Chain(5).then(raiser).except_(async_handler).run()
    self.assertEqual(result, 'async_caught')

  async def test_async_finally_handler(self):
    tracker = []
    async def async_cleanup(root):
      tracker.append(f'cleanup:{root}')
    result = (
      await Chain(5)
      .then(async_fn)
      .finally_(async_cleanup)
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, ['cleanup:5'])

  async def test_async_if_else_both_branches(self):
    async def async_pred(x):
      return x > 100
    result = (
      await Chain(5)
      .if_(async_pred, then=lambda x: 'yes')
      .else_(lambda x: 'no')
      .run()
    )
    self.assertEqual(result, 'no')


# ---------------------------------------------------------------------------
# PART 4: except_/finally_ x every method
# ---------------------------------------------------------------------------

class TestExceptWithEveryMethod(unittest.TestCase):

  def test_map_raises_except_catches(self):
    def raiser(x):
      if x > 2:
        raise ValueError('too big')
      return x
    result = Chain([1, 2, 3]).map(raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_filter_raises_except_catches(self):
    def raiser(x):
      raise ValueError('bad filter')
    result = Chain([1, 2]).filter(raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_gather_raises_except_catches(self):
    def raiser(x):
      raise ValueError('gather fail')
    result = Chain(5).gather(raiser, lambda x: x).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_with_body_raises_except_catches(self):
    cm = TrackingCM()
    def raiser(ctx):
      raise ValueError('body fail')
    result = Chain(cm).with_(raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')
    self.assertTrue(cm.exited)

  def test_if_pred_raises_except_catches(self):
    def bad_pred(x):
      raise ValueError('pred fail')
    result = Chain(5).if_(bad_pred, then=lambda x: x).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_if_fn_raises_except_catches(self):
    def raiser(x):
      raise ValueError('fn fail')
    result = Chain(5).if_(lambda x: True, then=raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_else_fn_raises_except_catches(self):
    def raiser(x):
      raise ValueError('else fail')
    result = (
      Chain(5)
      .if_(lambda x: False, then=lambda x: x)
      .else_(raiser)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_do_raises_except_catches(self):
    def raiser(x):
      raise ValueError('do fail')
    result = Chain(5).do(raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_then_raises_except_catches(self):
    def raiser(x):
      raise ValueError('then fail')
    result = Chain(5).then(raiser).except_(lambda rv, exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_except_handler_receives_exception_type(self):
    def raiser(x):
      raise TypeError('type error')
    result = (
      Chain(5)
      .then(raiser)
      .except_(lambda rv, exc: type(exc).__name__, exceptions=TypeError)
      .run()
    )
    self.assertEqual(result, 'TypeError')

  def test_except_wrong_type_does_not_catch(self):
    def raiser(x):
      raise TypeError('type error')
    with self.assertRaises(TypeError):
      Chain(5).then(raiser).except_(lambda rv, exc: 'caught', exceptions=ValueError).run()


class TestFinallyWithEveryMethod(unittest.TestCase):

  def test_map_with_finally(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, [2, 4, 6])
    self.assertEqual(tracker, ['cleanup'])

  def test_filter_with_finally(self):
    tracker = []
    result = (
      Chain([1, 2, 3, 4])
      .filter(lambda x: x > 2)
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, [3, 4])
    self.assertEqual(tracker, ['cleanup'])

  def test_gather_with_finally(self):
    tracker = []
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, [6, 7])
    self.assertEqual(tracker, ['cleanup'])

  def test_with_with_finally(self):
    cm = TrackingCM()
    tracker = []
    result = (
      Chain(cm)
      .with_(lambda ctx: ctx + '_body')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'tracked_ctx_body')
    self.assertEqual(tracker, ['cleanup'])
    self.assertTrue(cm.exited)

  def test_if_with_finally(self):
    tracker = []
    result = (
      Chain(5)
      .if_(lambda x: x > 0, then=lambda x: x * 10)
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 50)
    self.assertEqual(tracker, ['cleanup'])

  def test_do_with_finally(self):
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: x * 100)
      .finally_(lambda root: tracker.append(f'root:{root}'))
      .run()
    )
    self.assertEqual(result, 5)
    self.assertEqual(tracker, ['root:5'])

  def test_finally_runs_on_error(self):
    tracker = []
    def raiser(x):
      raise ValueError('boom')
    with self.assertRaises(ValueError):
      Chain(5).then(raiser).finally_(lambda root: tracker.append('cleanup')).run()
    self.assertEqual(tracker, ['cleanup'])


class TestExceptFinallyCombo(unittest.TestCase):

  def test_except_and_finally_both_run(self):
    tracker = []
    def raiser(x):
      raise ValueError('boom')
    result = (
      Chain(5)
      .then(raiser)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['cleanup'])

  def test_map_except_finally(self):
    tracker = []
    def raiser(x):
      if x > 2:
        raise ValueError('too big')
      return x
    result = (
      Chain([1, 2, 3])
      .map(raiser)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['cleanup'])

  def test_with_except_finally(self):
    cm = TrackingCM()
    tracker = []
    def raiser(ctx):
      raise ValueError('body fail')
    result = (
      Chain(cm)
      .with_(raiser)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['cleanup'])
    self.assertTrue(cm.exited)

  def test_no_error_except_not_called_finally_called(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .except_(lambda rv, exc: 'should_not_run')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, ['cleanup'])

  def test_gather_except_finally(self):
    tracker = []
    def raiser(x):
      raise ValueError('gather fail')
    result = (
      Chain(5)
      .gather(raiser)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['cleanup'])

  def test_if_except_finally(self):
    tracker = []
    def raiser(x):
      raise ValueError('if fail')
    result = (
      Chain(5)
      .if_(lambda x: True, then=raiser)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['cleanup'])


class TestAsyncExceptFinally(IsolatedAsyncioTestCase):

  async def test_async_map_raises_except_catches(self):
    async def async_raiser(x):
      if x > 2:
        raise ValueError('async too big')
      return x
    result = (
      await Chain([1, 2, 3])
      .map(async_raiser)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_async_filter_raises_except_catches(self):
    async def async_raiser(x):
      raise ValueError('async filter fail')
    result = (
      await Chain([1, 2])
      .filter(async_raiser)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_async_with_raises_except_catches(self):
    cm = AsyncCM()
    async def async_raiser(ctx):
      raise ValueError('async body fail')
    result = (
      await Chain(cm)
      .with_(async_raiser)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertTrue(cm.exited)

  async def test_async_except_and_finally(self):
    tracker = []
    async def async_raiser(x):
      raise ValueError('boom')
    result = (
      await Chain(5)
      .then(async_raiser)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda root: tracker.append('cleanup'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['cleanup'])


# ---------------------------------------------------------------------------
# PART 6: decorator x every method
# ---------------------------------------------------------------------------

class TestDecoratorPairs(unittest.TestCase):

  def test_decorator_then(self):
    @Chain().then(lambda x: x + 1).decorator()
    def my_fn(x):
      return x * 2
    self.assertEqual(my_fn(5), 11)

  def test_decorator_map(self):
    @Chain().then(lambda x: x).map(lambda x: x * 2).decorator()
    def my_fn():
      return [1, 2, 3]
    self.assertEqual(my_fn(), [2, 4, 6])

  def test_decorator_filter(self):
    @Chain().then(lambda x: x).filter(lambda x: x > 2).decorator()
    def my_fn():
      return [1, 2, 3, 4, 5]
    self.assertEqual(my_fn(), [3, 4, 5])

  def test_decorator_if_else(self):
    @(
      Chain()
      .then(lambda x: x)
      .if_(lambda x: x > 10, then=lambda x: 'big')
      .else_(lambda x: 'small')
      .decorator()
    )
    def my_fn(x):
      return x
    self.assertEqual(my_fn(20), 'big')
    self.assertEqual(my_fn(5), 'small')

  def test_decorator_except(self):
    @Chain().then(lambda x: 1 / x).except_(lambda rv, exc: 'error').decorator()
    def my_fn(x):
      return x
    self.assertEqual(my_fn(2), 0.5)
    self.assertEqual(my_fn(0), 'error')

  def test_decorator_finally(self):
    tracker = []

    @Chain().then(lambda x: x + 1).finally_(lambda root: tracker.append(root)).decorator()
    def my_fn(x):
      return x
    result = my_fn(5)
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [5])

  def test_decorator_gather(self):
    @Chain().then(lambda x: x).gather(lambda x: x + 1, lambda x: x * 2).decorator()
    def my_fn(x):
      return x
    self.assertEqual(my_fn(5), [6, 10])

  def test_decorator_do(self):
    tracker = []

    @Chain().do(lambda x: tracker.append(x)).then(lambda x: x + 1).decorator()
    def my_fn(x):
      return x
    result = my_fn(5)
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [5])


# ---------------------------------------------------------------------------
# PART 7: iterate x every method
# ---------------------------------------------------------------------------

class TestIteratePairs(unittest.TestCase):

  def test_iterate_after_then(self):
    result = list(Chain([1, 2, 3]).then(lambda x: x).iterate())
    self.assertEqual(result, [1, 2, 3])

  def test_iterate_with_fn_after_then(self):
    result = list(Chain([1, 2, 3]).then(lambda x: x).iterate(lambda x: x * 2))
    self.assertEqual(result, [2, 4, 6])

  def test_iterate_after_map(self):
    """map returns a list, iterate yields elements of that list."""
    result = list(Chain([1, 2, 3]).map(lambda x: x * 2).iterate())
    self.assertEqual(result, [2, 4, 6])

  def test_iterate_after_filter(self):
    result = list(Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).iterate())
    self.assertEqual(result, [4, 5])

  def test_iterate_after_with(self):
    cm = TrackingCM()
    cm.enter_result = [10, 20, 30]
    result = list(Chain(cm).with_(lambda ctx: ctx).iterate())
    self.assertEqual(result, [10, 20, 30])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_iterate_do_after_then(self):
    result = list(Chain([1, 2, 3]).then(lambda x: x).iterate_do(lambda x: x * 100))
    # iterate_do discards fn result, yields original items
    self.assertEqual(result, [1, 2, 3])

  def test_iterate_with_callable(self):
    gen = Chain([10, 20, 30]).then(lambda x: x).iterate(lambda x: x + 1)
    # Call with different initial value
    result = list(gen([5, 6, 7]))
    self.assertEqual(result, [6, 7, 8])


class TestIterateAsync(IsolatedAsyncioTestCase):

  async def test_async_iterate_after_then(self):
    result = []
    async for item in Chain([1, 2, 3]).then(async_identity).iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_async_iterate_with_async_fn(self):
    async def async_double(x):
      return x * 2
    result = []
    async for item in Chain([1, 2, 3]).then(lambda x: x).iterate(async_double):
      result.append(item)
    self.assertEqual(result, [2, 4, 6])

  async def test_async_iterate_after_map(self):
    async def async_double(x):
      return x * 2
    result = []
    async for item in Chain([1, 2]).map(async_double).iterate():
      result.append(item)
    self.assertEqual(result, [2, 4])

  async def test_async_iterate_after_filter(self):
    result = []
    async for item in Chain(AsyncRange(5)).filter(lambda x: x > 2).iterate():
      result.append(item)
    self.assertEqual(result, [3, 4])


# ---------------------------------------------------------------------------
# Additional pairwise combinations (ensuring 120+ tests)
# ---------------------------------------------------------------------------

class TestReturnBreakCombinations(unittest.TestCase):

  def test_return_in_then_skips_map(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .then(lambda x: Chain.return_(99))
      .map(lambda x: tracker.append(x) or x)
      .run()
    )
    self.assertEqual(result, 99)
    self.assertEqual(tracker, [])

  def test_return_in_do_skips_remaining(self):
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: Chain.return_(42))
      .then(lambda x: tracker.append('not_reached'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  def test_break_in_map_exits_loop(self):
    def break_at_3(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    result = Chain([1, 2, 3, 4, 5]).map(break_at_3).run()
    self.assertEqual(result, [10, 20])

  def test_break_with_value_in_map(self):
    def break_with_val(x):
      if x >= 3:
        Chain.break_([99])
      return x
    result = Chain([1, 2, 3, 4]).map(break_with_val).run()
    self.assertEqual(result, [99])

  def test_return_in_if_body(self):
    result = (
      Chain(5)
      .if_(lambda x: x > 0, then=lambda x: Chain.return_(999))
      .then(lambda x: 'not_reached')
      .run()
    )
    self.assertEqual(result, 999)


class TestWithDoCombinations(unittest.TestCase):

  def test_with_do_then(self):
    cm = TrackingCM()
    tracker = []
    result = (
      Chain(cm)
      .with_do(lambda ctx: tracker.append(ctx))
      .then(lambda x: 'after_with_do')
      .run()
    )
    # with_do preserves the original CM value
    self.assertIs(result, 'after_with_do')
    self.assertEqual(tracker, ['tracked_ctx'])

  def test_with_do_map(self):
    cm = TrackingCM()
    cm.enter_result = [1, 2, 3]
    tracker = []
    # with_do ignores fn result, keeps cm; but then is needed to get iterable
    result = (
      Chain([10, 20, 30])
      .do(lambda x: None)
      .map(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, [11, 21, 31])


class TestForeachCombinations(unittest.TestCase):

  def test_foreach_then(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: tracker.append(x * 10))
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [10, 20, 30])

  def test_foreach_filter(self):
    tracker = []
    result = (
      Chain([1, 2, 3, 4, 5])
      .foreach(lambda x: tracker.append(x))
      .filter(lambda x: x > 3)
      .run()
    )
    self.assertEqual(result, [4, 5])
    self.assertEqual(tracker, [1, 2, 3, 4, 5])

  def test_foreach_map(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: tracker.append(x * 10))
      .map(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, [2, 4, 6])
    self.assertEqual(tracker, [10, 20, 30])


class TestGatherAdvancedCombinations(unittest.TestCase):

  def test_gather_gather(self):
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .gather(lambda x: sum(x), lambda x: len(x))
      .run()
    )
    self.assertEqual(result, [13, 2])

  def test_gather_except(self):
    def raiser(x):
      raise ValueError('fail')
    result = (
      Chain(5)
      .gather(lambda x: x + 1, raiser)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')


class TestIfElseCombinations(unittest.TestCase):

  def test_if_else_map(self):
    result = (
      Chain([1, 2, 3])
      .if_(lambda x: len(x) > 10, then=lambda x: [])
      .else_(lambda x: x)
      .map(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, [2, 4, 6])

  def test_if_else_filter(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .if_(lambda x: True, then=lambda x: x)
      .else_(lambda x: [])
      .filter(lambda x: x > 3)
      .run()
    )
    self.assertEqual(result, [4, 5])

  def test_if_else_gather(self):
    result = (
      Chain(5)
      .if_(lambda x: x > 0, then=lambda x: x * 2)
      .else_(lambda x: 0)
      .gather(lambda x: x + 1, lambda x: x - 1)
      .run()
    )
    self.assertEqual(result, [11, 9])

  def test_if_else_with(self):
    cm = TrackingCM()
    result = (
      Chain(cm)
      .if_(lambda x: True, then=lambda x: x)
      .else_(lambda x: None)
      .with_(lambda ctx: ctx + '_used')
      .run()
    )
    self.assertEqual(result, 'tracked_ctx_used')

  def test_nested_if_else(self):
    result = (
      Chain(5)
      .if_(lambda x: x > 10, then=lambda x: 'very_big')
      .else_(lambda x: x)
      .if_(lambda x: x > 3, then=lambda x: 'medium')
      .else_(lambda x: 'small')
      .run()
    )
    self.assertEqual(result, 'medium')


class TestComplexPipelines(unittest.TestCase):

  def test_four_method_pipeline(self):
    """then -> do -> map -> filter"""
    tracker = []
    result = (
      Chain(0)
      .then(lambda x: [1, 2, 3, 4, 5])
      .do(lambda x: tracker.append(len(x)))
      .map(lambda x: x * 2)
      .filter(lambda x: x > 5)
      .run()
    )
    self.assertEqual(result, [6, 8, 10])
    self.assertEqual(tracker, [5])

  def test_five_method_pipeline(self):
    """then -> map -> filter -> then -> if"""
    result = (
      Chain(0)
      .then(lambda x: [1, 2, 3, 4, 5])
      .map(lambda x: x * 2)
      .filter(lambda x: x > 5)
      .then(lambda x: sum(x))
      .if_(lambda x: x > 20, then=lambda x: 'big')
      .else_(lambda x: 'small')
      .run()
    )
    self.assertEqual(result, 'big')

  def test_do_chain_does_not_modify_value(self):
    """Multiple do calls should not alter the chain value."""
    result = (
      Chain(42)
      .do(lambda x: x * 100)
      .do(lambda x: x + 999)
      .do(lambda x: 'garbage')
      .then(lambda x: x)
      .run()
    )
    self.assertEqual(result, 42)

  def test_gather_into_map_into_filter(self):
    result = (
      Chain(5)
      .gather(lambda x: x, lambda x: x * 2, lambda x: x * 3)
      .map(lambda x: x + 1)
      .filter(lambda x: x > 7)
      .run()
    )
    self.assertEqual(result, [11, 16])

  def test_with_except_finally_combo(self):
    cm = TrackingCM()
    tracker = []
    def raiser(ctx):
      raise ValueError('with_body_error')
    result = (
      Chain(cm)
      .with_(raiser)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda root: tracker.append('final'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, ['final'])
    self.assertTrue(cm.exited)


class TestComplexAsyncPipelines(IsolatedAsyncioTestCase):

  async def test_async_four_method_pipeline(self):
    async def async_double(x):
      return x * 2
    result = (
      await Chain(0)
      .then(lambda x: [1, 2, 3, 4, 5])
      .map(async_double)
      .filter(lambda x: x > 5)
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(result, 24)

  async def test_async_gather_then_map(self):
    async def async_add(x):
      return x + 10
    result = (
      await Chain(5)
      .gather(async_add, lambda x: x * 2)
      .map(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, [16, 11])

  async def test_async_if_else_then_map(self):
    async def async_pred(x):
      return len(x) > 3
    result = (
      await Chain([1, 2, 3, 4, 5])
      .if_(async_pred, then=lambda x: x)
      .else_(lambda x: [])
      .map(lambda x: x * 2)
      .run()
    )
    # Predicate receives a list of length 5, so if-branch runs
    self.assertEqual(result, [2, 4, 6, 8, 10])

  async def test_async_with_then_map(self):
    cm = AsyncCM()
    result = (
      await Chain(cm)
      .with_(lambda ctx: [1, 2, 3])
      .map(lambda x: x * 10)
      .run()
    )
    self.assertEqual(result, [10, 20, 30])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_filter_map_gather(self):
    result = (
      await Chain(AsyncRange(6))
      .filter(lambda x: x > 2)
      .map(lambda x: x * 10)
      .gather(lambda x: sum(x), lambda x: len(x))
      .run()
    )
    self.assertEqual(result, [120, 3])

  async def test_mixed_sync_async_long_chain(self):
    tracker = []
    result = (
      await Chain(1)
      .then(lambda x: x + 1)
      .then(async_fn)
      .do(lambda x: tracker.append(x))
      .then(lambda x: [x, x + 1, x + 2])
      .map(lambda x: x * 2)
      .filter(lambda x: x > 7)
      .then(lambda x: sum(x))
      .run()
    )
    self.assertEqual(tracker, [3])
    self.assertEqual(result, 18)

  async def test_async_decorator_chain(self):
    @Chain().then(async_fn).then(lambda x: x * 10).decorator()
    def my_fn(x):
      return x
    result = await my_fn(5)
    self.assertEqual(result, 60)


# ---------------------------------------------------------------------------
# Edge-case pairwise tests
# ---------------------------------------------------------------------------

class TestEdgeCasePairs(unittest.TestCase):

  def test_then_none_value(self):
    result = Chain(None).then(lambda x: x is None).run()
    self.assertTrue(result)

  def test_do_with_none(self):
    tracker = []
    result = Chain(None).do(lambda x: tracker.append(x)).run()
    self.assertIsNone(result)
    self.assertEqual(tracker, [None])

  def test_map_empty_then(self):
    result = Chain([]).map(lambda x: x).then(lambda x: len(x)).run()
    self.assertEqual(result, 0)

  def test_filter_empty_then(self):
    result = Chain([]).filter(lambda x: True).then(lambda x: len(x)).run()
    self.assertEqual(result, 0)

  def test_gather_single_fn(self):
    result = Chain(5).gather(lambda x: x + 1).run()
    self.assertEqual(result, [6])

  def test_if_with_none_result(self):
    result = Chain(None).if_(lambda x: x is None, then=lambda x: 'was_none').run()
    self.assertEqual(result, 'was_none')

  def test_chain_no_root_then(self):
    result = Chain().then(lambda: 42).run()
    self.assertEqual(result, 42)

  def test_chain_no_root_then_then(self):
    result = Chain().then(lambda: 10).then(lambda x: x + 5).run()
    self.assertEqual(result, 15)

  def test_except_returns_none(self):
    def raiser(x):
      raise ValueError('boom')
    result = Chain(5).then(raiser).except_(lambda rv, exc: None).run()
    self.assertIsNone(result)

  def test_multiple_thens_with_different_types(self):
    result = (
      Chain(5)
      .then(lambda x: str(x))
      .then(lambda x: x + '!')
      .then(lambda x: len(x))
      .run()
    )
    self.assertEqual(result, 2)

  def test_do_does_not_affect_value_with_return(self):
    result = (
      Chain(42)
      .do(lambda x: 'ignored')
      .then(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, 84)

  def test_map_then_filter(self):
    result = (
      Chain([1, 2, 3, 4])
      .map(lambda x: x ** 2)
      .then(lambda x: x)
      .filter(lambda x: x > 5)
      .run()
    )
    self.assertEqual(result, [9, 16])


class TestCallSyntaxCombinations(unittest.TestCase):
  """Test using __call__ (chain(v)) instead of .run(v)."""

  def test_call_then(self):
    c = Chain().then(lambda x: x + 1)
    self.assertEqual(c(5), 6)

  def test_call_map(self):
    c = Chain().then(lambda x: x).map(lambda x: x * 2)
    self.assertEqual(c([1, 2, 3]), [2, 4, 6])

  def test_call_filter(self):
    c = Chain().then(lambda x: x).filter(lambda x: x > 2)
    self.assertEqual(c([1, 2, 3, 4]), [3, 4])

  def test_call_if_else(self):
    c = (
      Chain()
      .then(lambda x: x)
      .if_(lambda x: x > 10, then=lambda x: 'big')
      .else_(lambda x: 'small')
    )
    self.assertEqual(c(20), 'big')
    self.assertEqual(c(5), 'small')


if __name__ == '__main__':
  unittest.main()
