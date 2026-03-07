"""Tests for if_()/else_() interacting with every other chain method,
in both sync and async modes. Covers the full cross-feature matrix:
map, filter, with_, gather, iterate, except_/finally_, decorator,
freeze, do, return_/break_, multiple if_ chaining, and nested chains.
"""
from __future__ import annotations

import asyncio
import threading
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, QuentException
from helpers import (
  AsyncCM,
  AsyncRange,
  SyncCM,
  SyncCMSuppresses,
  make_async_tracker,
  make_tracker,
)


# ---------------------------------------------------------------------------
# 1. if_ x map (sync + async)
# ---------------------------------------------------------------------------

class TestIfMap(unittest.TestCase):

  def test_if_after_map_truthy(self):
    """if_ receives the map result (a list) and applies fn when truthy."""
    result = Chain([1, 2, 3]).map(lambda x: x * 2).if_(lambda v: len(v) > 0, lambda v: sum(v)).run()
    self.assertEqual(result, 12)  # sum([2, 4, 6])

  def test_if_after_map_falsy(self):
    """if_ predicate falsy: map result passes through unchanged."""
    result = Chain([1, 2, 3]).map(lambda x: x * 2).if_(lambda v: len(v) > 100, lambda v: 'nope').run()
    self.assertEqual(result, [2, 4, 6])

  def test_if_before_map_truthy(self):
    """if_ transforms the value before map consumes it."""
    result = Chain([1, 2, 3]).if_(lambda v: True, lambda v: [x + 10 for x in v]).map(lambda x: x * 2).run()
    self.assertEqual(result, [22, 24, 26])

  def test_if_before_map_falsy_passthrough(self):
    """if_ falsy: original value passes through to map."""
    result = Chain([1, 2, 3]).if_(lambda v: False, lambda v: [99]).map(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_if_inside_map_fn_nested_chain(self):
    """if_ nested inside the map callback via a sub-chain."""
    inner = Chain().if_(lambda x: x > 1, lambda x: x * 100)
    result = Chain([1, 2, 3]).map(inner).run()
    self.assertEqual(result, [1, 200, 300])

  def test_if_else_inside_map_fn_nested_chain(self):
    """if_/else_ nested inside map callback."""
    inner = Chain().if_(lambda x: x % 2 == 0, lambda x: 'even').else_(lambda x: 'odd')
    result = Chain([1, 2, 3, 4]).map(inner).run()
    self.assertEqual(result, ['odd', 'even', 'odd', 'even'])

  def test_foreach_with_if(self):
    """foreach preserves original items; if_ applied after."""
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 100)  # side effect, original items preserved
      .if_(lambda v: len(v) == 3, lambda v: v + [4])
      .run()
    )
    self.assertEqual(result, [1, 2, 3, 4])

  def test_if_with_foreach(self):
    """if_ before foreach: transforms value then foreach keeps originals."""
    result = (
      Chain([10, 20, 30])
      .if_(lambda v: True, lambda v: [x - 5 for x in v])
      .foreach(lambda x: None)
      .run()
    )
    self.assertEqual(result, [5, 15, 25])


class TestIfMapAsync(IsolatedAsyncioTestCase):

  async def test_if_after_map_async_pred(self):
    """Async predicate in if_ after sync map."""
    async def pred(v):
      return len(v) > 0

    result = await Chain([1, 2]).map(lambda x: x + 1).if_(pred, lambda v: sum(v)).run()
    self.assertEqual(result, 5)  # sum([2, 3])

  async def test_if_after_map_async_fn(self):
    """Async fn in if_ after sync map."""
    async def transform(v):
      return sorted(v, reverse=True)

    result = await Chain([3, 1, 2]).map(lambda x: x * 2).if_(lambda v: True, transform).run()
    self.assertEqual(result, [6, 4, 2])

  async def test_async_map_then_if(self):
    """Async iterable map followed by sync if_."""
    result = await Chain(AsyncRange(4)).map(lambda x: x * 10).if_(lambda v: True, lambda v: sum(v)).run()
    self.assertEqual(result, 60)  # 0+10+20+30

  async def test_if_inside_map_async_nested(self):
    """Async if_ pred inside map via nested chain."""
    async def pred(x):
      return x > 1

    inner = Chain().if_(pred, lambda x: x * 100)
    result = await Chain([1, 2, 3]).map(inner).run()
    self.assertEqual(result, [1, 200, 300])


# ---------------------------------------------------------------------------
# 2. if_ x filter (sync + async)
# ---------------------------------------------------------------------------

class TestIfFilter(unittest.TestCase):

  def test_if_after_filter_truthy(self):
    result = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 2).if_(lambda v: len(v) > 0, lambda v: sum(v)).run()
    self.assertEqual(result, 12)  # 3+4+5

  def test_if_after_filter_falsy(self):
    result = Chain([1, 2, 3]).filter(lambda x: x > 1).if_(lambda v: len(v) > 10, lambda v: 'nope').run()
    self.assertEqual(result, [2, 3])

  def test_if_before_filter_truthy(self):
    result = Chain([1, 2, 3, 4]).if_(lambda v: True, lambda v: [x * 2 for x in v]).filter(lambda x: x > 4).run()
    self.assertEqual(result, [6, 8])

  def test_if_before_filter_falsy(self):
    result = Chain([1, 2, 3, 4]).if_(lambda v: False, lambda v: [99]).filter(lambda x: x > 2).run()
    self.assertEqual(result, [3, 4])

  def test_if_inside_filter_predicate_nested(self):
    """Use nested chain with if_ as the filter predicate."""
    # The filter predicate must return truthy/falsy.
    # Use a chain: if x > 2, return True, else return False.
    inner = Chain().if_(lambda x: x > 2, lambda x: True).else_(lambda x: False)
    result = Chain([1, 2, 3, 4]).filter(inner).run()
    self.assertEqual(result, [3, 4])

  def test_if_else_after_filter_empty(self):
    """Filter produces empty list; if_ falsy -> else_ takes over."""
    result = (
      Chain([1, 2, 3])
      .filter(lambda x: x > 100)
      .if_(lambda v: len(v) > 0, lambda v: 'has_items')
      .else_(lambda v: 'empty')
      .run()
    )
    self.assertEqual(result, 'empty')

  def test_filter_then_if_then_filter(self):
    """Chaining filter -> if_ -> filter."""
    result = (
      Chain(range(10))
      .filter(lambda x: x % 2 == 0)
      .if_(lambda v: True, lambda v: [x + 1 for x in v])
      .filter(lambda x: x > 5)
      .run()
    )
    self.assertEqual(result, [7, 9])


class TestIfFilterAsync(IsolatedAsyncioTestCase):

  async def test_async_filter_then_if(self):
    async def pred(x):
      return x > 1

    result = await Chain([1, 2, 3, 4]).filter(pred).if_(lambda v: True, lambda v: sum(v)).run()
    self.assertEqual(result, 9)  # 2+3+4

  async def test_if_async_pred_then_filter(self):
    async def is_truthy(v):
      return True

    result = await Chain([1, 2, 3]).if_(is_truthy, lambda v: [x * 3 for x in v]).filter(lambda x: x > 5).run()
    self.assertEqual(result, [6, 9])

  async def test_async_iterable_filter_then_if(self):
    result = await Chain(AsyncRange(6)).filter(lambda x: x > 3).if_(lambda v: True, lambda v: sum(v)).run()
    self.assertEqual(result, 9)  # 4+5


# ---------------------------------------------------------------------------
# 3. if_ x with_/with_do (sync + async)
# ---------------------------------------------------------------------------

class TestIfWith(unittest.TestCase):

  def test_if_after_with_truthy(self):
    cm = SyncCM()
    result = Chain(cm).with_(lambda ctx: ctx.upper()).if_(lambda v: True, lambda v: v + '!').run()
    self.assertEqual(result, 'CTX_VALUE!')
    self.assertTrue(cm.exited)

  def test_if_after_with_falsy(self):
    cm = SyncCM()
    result = Chain(cm).with_(lambda ctx: ctx.upper()).if_(lambda v: False, lambda v: 'nope').run()
    self.assertEqual(result, 'CTX_VALUE')
    self.assertTrue(cm.exited)

  def test_if_producing_cm_for_with(self):
    """if_ returns a CM which is then consumed by with_."""
    cm = SyncCM()
    result = (
      Chain(None)
      .if_(lambda v: True, lambda v: cm)
      .with_(lambda ctx: ctx + '_used')
      .run()
    )
    self.assertEqual(result, 'ctx_value_used')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_if_inside_with_body_nested_chain(self):
    """Nested chain with if_ used as with_ body."""
    cm = SyncCM()
    inner = Chain().if_(lambda ctx: ctx == 'ctx_value', lambda ctx: 'matched').else_(lambda ctx: 'unmatched')
    result = Chain(cm).with_(inner).run()
    self.assertEqual(result, 'matched')
    self.assertTrue(cm.exited)

  def test_with_do_then_if(self):
    """with_do discards result; if_ sees the original CM object."""
    cm = SyncCM()
    result = (
      Chain(cm)
      .with_do(lambda ctx: 'discarded')
      .if_(lambda v: hasattr(v, 'exited'), lambda v: 'is_cm')
      .else_(lambda v: 'not_cm')
      .run()
    )
    self.assertEqual(result, 'is_cm')
    self.assertTrue(cm.exited)

  def test_cm_suppresses_with_if(self):
    """CM that suppresses exceptions + if_ that raises in body."""
    cm = SyncCMSuppresses()
    result = (
      Chain(cm)
      .with_(lambda ctx: 'body_result')
      .if_(lambda v: True, lambda v: v + '_transformed')
      .run()
    )
    self.assertEqual(result, 'body_result_transformed')


class TestIfWithAsync(IsolatedAsyncioTestCase):

  async def test_async_cm_then_if(self):
    cm = AsyncCM()
    result = await Chain(cm).with_(lambda ctx: ctx.upper()).if_(lambda v: True, lambda v: v + '!').run()
    self.assertEqual(result, 'CTX_VALUE!')
    self.assertTrue(cm.exited)

  async def test_async_cm_with_async_if_pred(self):
    async def pred(v):
      return v == 'CTX_VALUE'

    cm = AsyncCM()
    result = await Chain(cm).with_(lambda ctx: ctx.upper()).if_(pred, lambda v: 'yes').else_(lambda v: 'no').run()
    self.assertEqual(result, 'yes')

  async def test_if_async_fn_inside_with_body(self):
    """Async fn in nested chain if_ used as with_ body."""
    async def transform(ctx):
      return ctx + '_async'

    cm = SyncCM()
    inner = Chain().if_(lambda ctx: True, transform)
    result = await Chain(cm).with_(inner).run()
    self.assertEqual(result, 'ctx_value_async')
    self.assertTrue(cm.exited)

  async def test_async_cm_with_do_then_if(self):
    cm = AsyncCM()
    result = await (
      Chain(cm)
      .with_do(lambda ctx: 'discarded')
      .if_(lambda v: True, lambda v: 'after_with_do')
      .run()
    )
    self.assertEqual(result, 'after_with_do')
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# 4. if_ x gather (sync + async)
# ---------------------------------------------------------------------------

class TestIfGather(unittest.TestCase):

  def test_if_after_gather_truthy(self):
    result = Chain(10).gather(lambda x: x + 1, lambda x: x + 2).if_(lambda v: True, lambda v: sum(v)).run()
    self.assertEqual(result, 23)  # 11 + 12

  def test_if_after_gather_falsy(self):
    result = Chain(10).gather(lambda x: x + 1, lambda x: x + 2).if_(lambda v: False, lambda v: 'nope').run()
    self.assertEqual(result, [11, 12])

  def test_if_before_gather_truthy(self):
    result = Chain(5).if_(lambda v: True, lambda v: v * 2).gather(lambda x: x + 1, lambda x: x - 1).run()
    self.assertEqual(result, [11, 9])

  def test_if_before_gather_falsy(self):
    result = Chain(5).if_(lambda v: False, lambda v: 99).gather(lambda x: x + 1, lambda x: x - 1).run()
    self.assertEqual(result, [6, 4])

  def test_if_inside_gather_fn_nested(self):
    """One of the gather fns is a nested chain with if_."""
    inner = Chain().if_(lambda x: x > 5, lambda x: x * 10).else_(lambda x: x)
    result = Chain(3).gather(inner, lambda x: x + 1).run()
    self.assertEqual(result, [3, 4])  # 3 <= 5, so passthrough

  def test_if_inside_gather_fn_truthy(self):
    inner = Chain().if_(lambda x: x > 5, lambda x: x * 10).else_(lambda x: x)
    result = Chain(10).gather(inner, lambda x: x + 1).run()
    self.assertEqual(result, [100, 11])


class TestIfGatherAsync(IsolatedAsyncioTestCase):

  async def test_async_gather_then_if(self):
    async def fn1(x):
      return x + 1

    async def fn2(x):
      return x + 2

    result = await Chain(10).gather(fn1, fn2).if_(lambda v: True, lambda v: sum(v)).run()
    self.assertEqual(result, 23)

  async def test_if_async_pred_then_gather(self):
    async def pred(v):
      return v > 3

    result = await Chain(5).if_(pred, lambda v: v * 2).gather(lambda x: x + 1, lambda x: x - 1).run()
    self.assertEqual(result, [11, 9])

  async def test_if_inside_async_gather_fn(self):
    async def transform(x):
      return x * 100

    inner = Chain().if_(lambda x: x > 5, transform).else_(lambda x: 0)
    result = await Chain(10).gather(inner, lambda x: x).run()
    self.assertEqual(result, [1000, 10])


# ---------------------------------------------------------------------------
# 5. if_ x iterate/iterate_do (sync + async)
# ---------------------------------------------------------------------------

class TestIfIterate(unittest.TestCase):

  def test_if_before_iterate(self):
    """if_ transforms value before iterate consumes it."""
    result = list(Chain([1, 2, 3]).if_(lambda v: True, lambda v: [x * 2 for x in v]).iterate())
    self.assertEqual(result, [2, 4, 6])

  def test_if_before_iterate_falsy(self):
    result = list(Chain([1, 2, 3]).if_(lambda v: False, lambda v: [99]).iterate())
    self.assertEqual(result, [1, 2, 3])

  def test_if_inside_iterate_fn_nested(self):
    """iterate fn is a nested chain with if_."""
    inner = Chain().if_(lambda x: x > 1, lambda x: x * 10)
    result = list(Chain([1, 2, 3]).iterate(inner))
    self.assertEqual(result, [1, 20, 30])

  def test_iterate_do_with_if_before(self):
    """if_ before iterate_do."""
    tracker = []
    result = list(
      Chain([1, 2, 3])
      .if_(lambda v: True, lambda v: [x + 10 for x in v])
      .iterate_do(lambda x: tracker.append(x))
    )
    self.assertEqual(result, [11, 12, 13])
    self.assertEqual(tracker, [11, 12, 13])

  def test_if_else_inside_iterate_fn(self):
    inner = Chain().if_(lambda x: x % 2 == 0, lambda x: 'even').else_(lambda x: 'odd')
    result = list(Chain([0, 1, 2, 3]).iterate(inner))
    self.assertEqual(result, ['even', 'odd', 'even', 'odd'])


class TestIfIterateAsync(IsolatedAsyncioTestCase):

  async def test_async_if_before_iterate(self):
    async def pred(v):
      return True

    gen = Chain([1, 2, 3]).if_(pred, lambda v: [x * 5 for x in v]).iterate()
    result = [item async for item in gen]
    self.assertEqual(result, [5, 10, 15])

  async def test_async_if_inside_iterate_fn(self):
    async def transform(x):
      return x * 100

    inner = Chain().if_(lambda x: x > 1, transform)
    gen = Chain([1, 2, 3]).iterate(inner)
    result = [item async for item in gen]
    self.assertEqual(result, [1, 200, 300])

  async def test_async_iterate_do_with_if(self):
    tracker = []

    async def track(x):
      tracker.append(x)

    gen = Chain([10, 20]).if_(lambda v: True, lambda v: v).iterate_do(track)
    result = [item async for item in gen]
    self.assertEqual(result, [10, 20])
    self.assertEqual(tracker, [10, 20])


# ---------------------------------------------------------------------------
# 6. if_ x except_/finally_ (sync + async)
# ---------------------------------------------------------------------------

class TestIfExceptFinally(unittest.TestCase):

  def test_if_pred_raises_caught_by_except(self):
    def bad_pred(v):
      raise ValueError('pred error')

    result = Chain(5).if_(bad_pred, lambda v: v).except_(lambda exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_if_fn_raises_caught_by_except(self):
    def bad_fn(v):
      raise ValueError('fn error')

    result = Chain(5).if_(lambda v: True, bad_fn).except_(lambda exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_else_fn_raises_caught_by_except(self):
    def bad_else(v):
      raise ValueError('else error')

    result = Chain(5).if_(lambda v: False, lambda v: v).else_(bad_else).except_(lambda exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_if_with_finally_truthy_path(self):
    tracker = []
    result = (
      Chain(10)
      .if_(lambda v: True, lambda v: v * 2)
      .finally_(lambda root: tracker.append(root))
      .run()
    )
    self.assertEqual(result, 20)
    self.assertEqual(tracker, [10])

  def test_if_with_finally_falsy_path(self):
    tracker = []
    result = (
      Chain(10)
      .if_(lambda v: False, lambda v: v * 2)
      .finally_(lambda root: tracker.append(root))
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [10])

  def test_if_with_except_and_finally(self):
    tracker = []

    def bad_fn(v):
      raise ValueError('error')

    result = (
      Chain(5)
      .if_(lambda v: True, bad_fn)
      .except_(lambda exc: 'recovered')
      .finally_(lambda root: tracker.append('finally'))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(tracker, ['finally'])

  def test_except_filters_exception_type(self):
    def bad_fn(v):
      raise TypeError('type error')

    with self.assertRaises(TypeError):
      Chain(5).if_(lambda v: True, bad_fn).except_(lambda exc: 'caught', exceptions=[ValueError]).run()

  def test_if_fn_raises_except_receives_exc(self):
    def bad_fn(v):
      raise ValueError('specific')

    result = Chain(5).if_(lambda v: True, bad_fn).except_(lambda exc: str(exc)).run()
    self.assertEqual(result, 'specific')

  def test_finally_receives_root_when_if_changes_value(self):
    """finally_ always receives root value, not if_-modified value."""
    root_tracker = []
    result = (
      Chain(42)
      .if_(lambda v: True, lambda v: v * 100)
      .finally_(lambda root: root_tracker.append(root))
      .run()
    )
    self.assertEqual(result, 4200)
    self.assertEqual(root_tracker, [42])

  def test_if_no_error_except_not_called(self):
    tracker = make_tracker()
    result = Chain(5).if_(lambda v: True, lambda v: v + 1).except_(tracker).run()
    self.assertEqual(result, 6)
    self.assertEqual(tracker.calls, [])

  def test_nested_if_raises_in_second(self):
    """Second if_ raises, caught by except_."""
    result = (
      Chain(5)
      .if_(lambda v: True, lambda v: v + 1)
      .if_(lambda v: True, lambda v: (_ for _ in ()).throw(ValueError('second')))
      .except_(lambda exc: 'caught_second')
      .run()
    )
    self.assertEqual(result, 'caught_second')

  def test_else_raises_with_finally(self):
    tracker = []

    def bad_else(v):
      raise ValueError('else boom')

    result = (
      Chain(5)
      .if_(lambda v: False, lambda v: v)
      .else_(bad_else)
      .except_(lambda exc: 'recovered')
      .finally_(lambda root: tracker.append(root))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(tracker, [5])


class TestIfExceptFinallyAsync(IsolatedAsyncioTestCase):

  async def test_async_pred_raises_caught_by_except(self):
    async def bad_pred(v):
      raise ValueError('async pred error')

    result = await Chain(5).if_(bad_pred, lambda v: v).except_(lambda exc: 'caught').run()
    self.assertEqual(result, 'caught')

  async def test_async_fn_raises_caught_by_except(self):
    async def bad_fn(v):
      raise ValueError('async fn error')

    result = await Chain(5).if_(lambda v: True, bad_fn).except_(lambda exc: 'caught').run()
    self.assertEqual(result, 'caught')

  async def test_async_if_fn_raises_async_except_handler(self):
    async def bad_fn(v):
      raise ValueError('async error')

    async def handler(exc):
      return 'async_caught'

    result = await Chain(5).if_(lambda v: True, bad_fn).except_(handler).run()
    self.assertEqual(result, 'async_caught')

  async def test_if_with_finally_async_pred(self):
    tracker = []

    async def pred(v):
      return True

    result = await (
      Chain(10)
      .if_(pred, lambda v: v * 3)
      .finally_(lambda root: tracker.append(root))
      .run()
    )
    self.assertEqual(result, 30)
    self.assertEqual(tracker, [10])

  async def test_async_else_fn_raises_caught(self):
    async def bad_else(v):
      raise ValueError('async else error')

    result = await Chain(5).if_(lambda v: False, lambda v: v).else_(bad_else).except_(lambda e: 'caught').run()
    self.assertEqual(result, 'caught')


# ---------------------------------------------------------------------------
# 7. if_ x decorator (sync + async)
# ---------------------------------------------------------------------------

class TestIfDecorator(unittest.TestCase):

  def test_decorated_fn_with_if_truthy(self):
    chain = Chain().if_(lambda v: v > 0, lambda v: v * 10)

    @chain.decorator()
    def my_fn(x):
      return x

    result = my_fn(5)
    self.assertEqual(result, 50)

  def test_decorated_fn_with_if_falsy(self):
    chain = Chain().if_(lambda v: v > 100, lambda v: v * 10)

    @chain.decorator()
    def my_fn(x):
      return x

    result = my_fn(5)
    self.assertEqual(result, 5)

  def test_decorated_fn_with_if_else(self):
    chain = Chain().if_(lambda v: v > 0, lambda v: 'positive').else_(lambda v: 'non_positive')

    @chain.decorator()
    def my_fn(x):
      return x

    self.assertEqual(my_fn(5), 'positive')
    self.assertEqual(my_fn(-3), 'non_positive')
    self.assertEqual(my_fn(0), 'non_positive')

  def test_decorator_with_frozen_chain_if(self):
    frozen = Chain().if_(lambda v: v > 0, lambda v: v + 100).freeze()
    chain = Chain().then(frozen)

    @chain.decorator()
    def my_fn(x):
      return x

    self.assertEqual(my_fn(5), 105)
    self.assertEqual(my_fn(-5), -5)

  def test_decorator_preserves_fn_name(self):
    chain = Chain().if_(lambda v: True, lambda v: v)

    @chain.decorator()
    def named_fn(x):
      return x

    self.assertEqual(named_fn.__name__, 'named_fn')


class TestIfDecoratorAsync(IsolatedAsyncioTestCase):

  async def test_async_if_in_decorated_chain(self):
    async def pred(v):
      return v > 0

    chain = Chain().if_(pred, lambda v: v * 10)

    @chain.decorator()
    def my_fn(x):
      return x

    result = await my_fn(5)
    self.assertEqual(result, 50)

  async def test_async_if_else_in_decorated_chain(self):
    async def pred(v):
      return v > 0

    chain = Chain().if_(pred, lambda v: 'pos').else_(lambda v: 'neg')

    @chain.decorator()
    def my_fn(x):
      return x

    self.assertEqual(await my_fn(5), 'pos')
    self.assertEqual(await my_fn(-1), 'neg')


# ---------------------------------------------------------------------------
# 8. if_ x freeze (sync + async)
# ---------------------------------------------------------------------------

class TestIfFreeze(unittest.TestCase):

  def test_frozen_if_reused_truthy(self):
    frozen = Chain().if_(lambda v: v > 0, lambda v: v * 2).else_(lambda v: v * -1).freeze()
    self.assertEqual(frozen(5), 10)
    self.assertEqual(frozen(-3), 3)
    self.assertEqual(frozen(0), 0)

  def test_frozen_if_multiple_calls(self):
    """Frozen chain is safe for repeated calls."""
    frozen = Chain().if_(lambda v: v % 2 == 0, lambda v: 'even').else_(lambda v: 'odd').freeze()
    results = [frozen(i) for i in range(6)]
    self.assertEqual(results, ['even', 'odd', 'even', 'odd', 'even', 'odd'])

  def test_frozen_chain_thread_safety(self):
    """Frozen chain with if_ can be called concurrently from threads."""
    frozen = Chain().if_(lambda v: v > 0, lambda v: v * 2).else_(lambda v: 0).freeze()
    results = [None] * 100
    errors = []

    def worker(idx, val):
      try:
        results[idx] = frozen(val)
      except Exception as e:
        errors.append(e)

    threads = [threading.Thread(target=worker, args=(i, i - 50)) for i in range(100)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(errors, [])
    for i in range(100):
      val = i - 50
      expected = val * 2 if val > 0 else 0
      self.assertEqual(results[i], expected)

  def test_frozen_if_nested_inside_unfrozen(self):
    frozen = Chain().if_(lambda v: v > 10, lambda v: v * 100).freeze()
    result = Chain(15).then(frozen).run()
    self.assertEqual(result, 1500)

  def test_frozen_if_nested_inside_unfrozen_falsy(self):
    frozen = Chain().if_(lambda v: v > 10, lambda v: v * 100).freeze()
    result = Chain(5).then(frozen).run()
    self.assertEqual(result, 5)

  def test_return_inside_frozen_if_fn(self):
    """return_ inside frozen if_ fn does NOT propagate to outer chain."""
    frozen = Chain().if_(lambda v: True, lambda v: Chain.return_(999)).freeze()
    result = Chain(1).then(frozen).then(lambda v: v + 1).run()
    # Frozen chain catches _Return, returns 999.
    # Outer chain's .then(lambda v: v + 1) runs on 999.
    self.assertEqual(result, 1000)


class TestIfFreezeAsync(IsolatedAsyncioTestCase):

  async def test_frozen_if_async_pred(self):
    async def pred(v):
      return v > 0

    frozen = Chain().if_(pred, lambda v: 'positive').else_(lambda v: 'non_positive').freeze()
    self.assertEqual(await frozen(5), 'positive')
    self.assertEqual(await frozen(-1), 'non_positive')


# ---------------------------------------------------------------------------
# 9. if_ x do (sync + async)
# ---------------------------------------------------------------------------

class TestIfDo(unittest.TestCase):

  def test_if_then_do_side_effect(self):
    """do after if_: side effect runs on if_-transformed value."""
    tracker = []
    result = (
      Chain(10)
      .if_(lambda v: True, lambda v: v * 2)
      .do(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(result, 20)
    self.assertEqual(tracker, [20])

  def test_do_then_if(self):
    """do before if_: side effect does not change the value for if_."""
    tracker = []
    result = (
      Chain(10)
      .do(lambda v: tracker.append(v))
      .if_(lambda v: v > 5, lambda v: v * 3)
      .run()
    )
    self.assertEqual(result, 30)
    self.assertEqual(tracker, [10])

  def test_if_inside_do_nested_result_discarded(self):
    """if_ inside a do step: the if_ result is discarded."""
    inner = Chain().if_(lambda v: True, lambda v: v * 999)
    tracker = []
    result = (
      Chain(7)
      .do(inner)
      .then(lambda v: v + 1)
      .run()
    )
    self.assertEqual(result, 8)  # do discards inner's result

  def test_do_after_if_falsy(self):
    tracker = []
    result = (
      Chain(10)
      .if_(lambda v: False, lambda v: 99)
      .do(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [10])


class TestIfDoAsync(IsolatedAsyncioTestCase):

  async def test_async_if_then_do(self):
    async def pred(v):
      return True

    tracker = []
    result = await (
      Chain(10)
      .if_(pred, lambda v: v * 2)
      .do(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(result, 20)
    self.assertEqual(tracker, [20])

  async def test_async_do_then_if(self):
    async def side(v):
      pass

    result = await (
      Chain(10)
      .do(side)
      .if_(lambda v: v > 5, lambda v: v * 3)
      .run()
    )
    self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# 10. if_ x return_/break_ (sync + async)
# ---------------------------------------------------------------------------

class TestIfReturnBreak(unittest.TestCase):

  def test_return_inside_if_fn_exits_chain(self):
    tracker = []
    result = (
      Chain(5)
      .if_(lambda v: True, lambda v: Chain.return_(42))
      .then(lambda v: tracker.append('should_not_run'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  def test_return_inside_else_fn_exits_chain(self):
    tracker = []
    result = (
      Chain(5)
      .if_(lambda v: False, lambda v: v)
      .else_(lambda v: Chain.return_(99))
      .then(lambda v: tracker.append('should_not_run'))
      .run()
    )
    self.assertEqual(result, 99)
    self.assertEqual(tracker, [])

  def test_break_inside_if_fn_within_map(self):
    """break_ called conditionally inside map lambda."""
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: Chain.break_() if x == 3 else x)
      .run()
    )
    self.assertEqual(result, [1, 2])

  def test_break_inside_else_fn_within_map(self):
    """break_ from else-like branch inside map lambda."""
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: x * 10 if x != 3 else Chain.break_())
      .run()
    )
    self.assertEqual(result, [10, 20])

  def test_return_no_value_from_if(self):
    result = (
      Chain(5)
      .if_(lambda v: True, lambda v: Chain.return_())
      .then(lambda v: 999)
      .run()
    )
    self.assertIsNone(result)

  def test_return_with_callable_from_if(self):
    result = (
      Chain(5)
      .if_(lambda v: True, lambda v: Chain.return_(lambda: 'early'))
      .then(lambda v: 999)
      .run()
    )
    self.assertEqual(result, 'early')


class TestIfReturnBreakAsync(IsolatedAsyncioTestCase):

  async def test_async_return_inside_if_fn(self):
    async def pred(v):
      return True

    tracker = []
    result = await (
      Chain(5)
      .if_(pred, lambda v: Chain.return_(42))
      .then(lambda v: tracker.append('no'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  async def test_async_break_inside_if_in_map(self):
    """break_ from if_-like condition in async map."""
    result = await (
      Chain(AsyncRange(6))
      .map(lambda x: Chain.break_() if x == 3 else x)
      .run()
    )
    self.assertEqual(result, [0, 1, 2])


# ---------------------------------------------------------------------------
# 11. if_ x multiple if_ chaining (sync + async)
# ---------------------------------------------------------------------------

class TestMultipleIfChaining(unittest.TestCase):

  def test_cascading_all_true(self):
    result = (
      Chain(1)
      .if_(lambda v: True, lambda v: v + 1)
      .if_(lambda v: True, lambda v: v + 1)
      .if_(lambda v: True, lambda v: v + 1)
      .run()
    )
    self.assertEqual(result, 4)

  def test_cascading_all_false(self):
    result = (
      Chain(1)
      .if_(lambda v: False, lambda v: v + 100)
      .if_(lambda v: False, lambda v: v + 100)
      .if_(lambda v: False, lambda v: v + 100)
      .run()
    )
    self.assertEqual(result, 1)

  def test_cascading_mixed(self):
    result = (
      Chain(1)
      .if_(lambda v: True, lambda v: v + 10)   # -> 11
      .if_(lambda v: False, lambda v: v + 100)  # -> 11 (passthrough)
      .if_(lambda v: True, lambda v: v * 2)     # -> 22
      .run()
    )
    self.assertEqual(result, 22)

  def test_alternating_if_else_first_true(self):
    result = (
      Chain(1)
      .if_(lambda v: True, lambda v: 'a').else_(lambda v: 'b')
      .if_(lambda v: v == 'a', lambda v: 'c').else_(lambda v: 'd')
      .run()
    )
    self.assertEqual(result, 'c')

  def test_alternating_if_else_first_false(self):
    result = (
      Chain(1)
      .if_(lambda v: False, lambda v: 'a').else_(lambda v: 'b')
      .if_(lambda v: v == 'a', lambda v: 'c').else_(lambda v: 'd')
      .run()
    )
    self.assertEqual(result, 'd')

  def test_six_chained_if_progressive(self):
    """Each if_ checks the current value and increments if condition met."""
    result = (
      Chain(0)
      .if_(lambda v: v == 0, lambda v: v + 1)    # -> 1
      .if_(lambda v: v == 1, lambda v: v + 1)    # -> 2
      .if_(lambda v: v == 2, lambda v: v + 1)    # -> 3
      .if_(lambda v: v == 3, lambda v: v + 1)    # -> 4
      .if_(lambda v: v == 4, lambda v: v + 1)    # -> 5
      .if_(lambda v: v == 5, lambda v: v + 1)    # -> 6
      .run()
    )
    self.assertEqual(result, 6)

  def test_if_else_branches_produce_different_types(self):
    """Cascading if_ with type changes."""
    result = (
      Chain(5)
      .if_(lambda v: v > 3, lambda v: str(v)).else_(lambda v: v)
      .if_(lambda v: isinstance(v, str), lambda v: v + '!').else_(lambda v: v * 2)
      .run()
    )
    self.assertEqual(result, '5!')


class TestMultipleIfChainingAsync(IsolatedAsyncioTestCase):

  async def test_cascading_all_async_pred(self):
    async def pred(v):
      return True

    result = await (
      Chain(1)
      .if_(pred, lambda v: v + 1)
      .if_(pred, lambda v: v + 1)
      .if_(pred, lambda v: v + 1)
      .run()
    )
    self.assertEqual(result, 4)

  async def test_mixed_sync_async_predicates(self):
    async def async_pred(v):
      return v > 5

    result = await (
      Chain(1)
      .if_(lambda v: True, lambda v: v + 10)        # sync pred -> 11
      .if_(async_pred, lambda v: v * 2)               # async pred -> 22
      .if_(lambda v: v > 20, lambda v: v + 100)       # sync pred -> 122
      .run()
    )
    self.assertEqual(result, 122)


# ---------------------------------------------------------------------------
# 12. if_ x nested chains (sync + async)
# ---------------------------------------------------------------------------

class TestIfNestedChains(unittest.TestCase):

  def test_if_fn_is_chain_with_own_if(self):
    """if_ fn is a Chain that itself has if_/else_."""
    inner = Chain().if_(lambda v: v > 10, lambda v: 'big').else_(lambda v: 'small')
    result = Chain(15).if_(lambda v: True, inner).run()
    self.assertEqual(result, 'big')

  def test_if_fn_is_chain_with_own_if_falsy_outer(self):
    inner = Chain().if_(lambda v: v > 10, lambda v: 'big').else_(lambda v: 'small')
    result = Chain(15).if_(lambda v: False, inner).run()
    self.assertEqual(result, 15)  # if_ falsy, passthrough

  def test_if_fn_is_chain_with_map(self):
    inner = Chain().then(lambda v: [v, v * 2, v * 3]).map(lambda x: x + 1)
    result = Chain(5).if_(lambda v: True, inner).run()
    self.assertEqual(result, [6, 11, 16])

  def test_if_fn_is_chain_with_except(self):
    inner = Chain().then(lambda v: 1 / 0).except_(lambda exc: 'caught_inner')
    result = Chain(5).if_(lambda v: True, inner).run()
    self.assertEqual(result, 'caught_inner')

  def test_else_fn_is_chain(self):
    inner = Chain().then(lambda v: v * 100)
    result = Chain(5).if_(lambda v: False, lambda v: v).else_(inner).run()
    self.assertEqual(result, 500)

  def test_nested_chain_with_finally(self):
    tracker = []
    inner = Chain().then(lambda v: v * 2).finally_(lambda root: tracker.append(root))
    result = Chain(5).if_(lambda v: True, inner).run()
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [5])

  def test_doubly_nested_if(self):
    """Chain inside if_ fn, which itself has if_ with another chain."""
    innermost = Chain().then(lambda v: v + 1000)
    inner = Chain().if_(lambda v: v > 10, innermost).else_(lambda v: v - 1)
    result = Chain(20).if_(lambda v: True, inner).run()
    self.assertEqual(result, 1020)

  def test_if_fn_chain_with_filter(self):
    inner = Chain().then(lambda v: list(range(v))).filter(lambda x: x % 2 == 0)
    result = Chain(8).if_(lambda v: v > 5, inner).run()
    self.assertEqual(result, [0, 2, 4, 6])


class TestIfNestedChainsAsync(IsolatedAsyncioTestCase):

  async def test_async_if_fn_chain_with_own_if(self):
    async def pred(v):
      return v > 10

    inner = Chain().if_(pred, lambda v: 'big').else_(lambda v: 'small')
    result = await Chain(15).if_(lambda v: True, inner).run()
    self.assertEqual(result, 'big')

  async def test_async_else_fn_is_chain(self):
    async def transform(v):
      return v * 100

    inner = Chain().then(transform)
    result = await Chain(5).if_(lambda v: False, lambda v: v).else_(inner).run()
    self.assertEqual(result, 500)

  async def test_async_nested_chain_with_map(self):
    async def double(x):
      return x * 2

    inner = Chain().then(lambda v: [v, v + 1, v + 2]).map(double)
    result = await Chain(10).if_(lambda v: True, inner).run()
    self.assertEqual(result, [20, 22, 24])

  async def test_async_nested_chain_with_except(self):
    async def bad_fn(v):
      raise ValueError('async nested error')

    async def handler(exc):
      return 'async_caught'

    inner = Chain().then(bad_fn).except_(handler)
    result = await Chain(5).if_(lambda v: True, inner).run()
    self.assertEqual(result, 'async_caught')


# ---------------------------------------------------------------------------
# Additional cross-feature edge cases
# ---------------------------------------------------------------------------

class TestIfEdgeCases(unittest.TestCase):

  def test_if_with_none_value(self):
    """if_ works when current value is None."""
    result = Chain(None).if_(lambda v: v is None, lambda v: 'was_none').run()
    self.assertEqual(result, 'was_none')

  def test_if_with_zero_value(self):
    """if_ with 0 as current value -- pred receives 0."""
    result = Chain(0).if_(lambda v: v == 0, lambda v: 'zero').else_(lambda v: 'nonzero').run()
    self.assertEqual(result, 'zero')

  def test_if_with_empty_string(self):
    result = Chain('').if_(lambda v: v == '', lambda v: 'empty').else_(lambda v: 'nonempty').run()
    self.assertEqual(result, 'empty')

  def test_if_with_falsy_pred_result_zero(self):
    """Predicate returns 0 (falsy) -> else branch taken."""
    result = Chain(5).if_(lambda v: 0, lambda v: 'yes').else_(lambda v: 'no').run()
    self.assertEqual(result, 'no')

  def test_if_with_truthy_pred_result_nonempty_list(self):
    """Predicate returns non-empty list (truthy) -> if branch taken."""
    result = Chain(5).if_(lambda v: [1], lambda v: 'yes').else_(lambda v: 'no').run()
    self.assertEqual(result, 'yes')

  def test_if_fn_returns_none(self):
    """if_ fn returns None, which becomes the new current value."""
    result = Chain(5).if_(lambda v: True, lambda v: None).then(lambda v: v is None).run()
    self.assertTrue(result)

  def test_if_else_with_ellipsis_args(self):
    """if_ fn called with ellipsis to suppress passing current value."""
    result = Chain(5).if_(lambda v: True, lambda: 'no_arg', ...).run()
    self.assertEqual(result, 'no_arg')

  def test_if_fn_with_extra_args(self):
    """if_ fn receives extra positional args."""
    result = Chain(5).if_(lambda v: True, lambda a, b: a + b, 10, 20).run()
    self.assertEqual(result, 30)

  def test_if_pred_returns_awaitable_sync_context(self):
    """When if_ pred returns a coroutine in sync context, chain transitions to async."""
    async def pred(v):
      return True

    # This should return a coroutine that needs to be awaited.
    result = Chain(5).if_(pred, lambda v: v * 2).run()
    # The chain should return a coroutine; verify by running it.
    self.assertTrue(asyncio.iscoroutine(result))
    actual = asyncio.run(result)
    self.assertEqual(actual, 10)


class TestIfWithGatherAndMap(unittest.TestCase):
  """Cross combinations of if_ with multiple operations."""

  def test_if_then_map_then_gather(self):
    result = (
      Chain([1, 2, 3])
      .if_(lambda v: True, lambda v: [x + 10 for x in v])
      .map(lambda x: x * 2)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    self.assertEqual(result, [72, 3])  # sum([22,24,26])=72, len=3

  def test_gather_then_if_then_map(self):
    result = (
      Chain(5)
      .gather(lambda x: x + 1, lambda x: x + 2)
      .if_(lambda v: len(v) > 0, lambda v: v)
      .map(lambda x: x * 10)
      .run()
    )
    self.assertEqual(result, [60, 70])

  def test_map_then_filter_then_if(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: x * 2)
      .filter(lambda x: x > 4)
      .if_(lambda v: len(v) > 0, lambda v: sum(v))
      .run()
    )
    self.assertEqual(result, 24)  # 6+8+10

  def test_if_with_do_and_map(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .if_(lambda v: True, lambda v: [x * 10 for x in v])
      .do(lambda v: tracker.extend(v))
      .map(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, [11, 21, 31])
    self.assertEqual(tracker, [10, 20, 30])


class TestIfWithGatherAndMapAsync(IsolatedAsyncioTestCase):

  async def test_async_if_map_gather(self):
    async def pred(v):
      return True

    result = await (
      Chain([1, 2, 3])
      .if_(pred, lambda v: [x + 10 for x in v])
      .map(lambda x: x * 2)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    self.assertEqual(result, [72, 3])

  async def test_async_filter_then_if_then_gather(self):
    async def pred(x):
      return x > 2

    result = await (
      Chain([1, 2, 3, 4])
      .filter(pred)
      .if_(lambda v: True, lambda v: v)
      .gather(lambda v: sum(v), lambda v: max(v))
      .run()
    )
    self.assertEqual(result, [7, 4])


class TestIfWithWithAndExcept(unittest.TestCase):

  def test_if_after_with_then_except(self):
    """with_ -> if_ -> except_: if_ fn raises, except_ catches."""
    cm = SyncCM()

    def bad_fn(v):
      raise ValueError('boom')

    result = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .if_(lambda v: True, bad_fn)
      .except_(lambda exc: 'recovered')
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertTrue(cm.exited)

  def test_if_with_cm_and_finally(self):
    """Full pipeline: with_ -> if_ -> finally_."""
    cm = SyncCM()
    tracker = []
    result = (
      Chain(cm)
      .with_(lambda ctx: ctx.upper())
      .if_(lambda v: True, lambda v: v + '!')
      .finally_(lambda root: tracker.append('done'))
      .run()
    )
    self.assertEqual(result, 'CTX_VALUE!')
    self.assertTrue(cm.exited)
    self.assertEqual(tracker, ['done'])


class TestIfWithWithAndExceptAsync(IsolatedAsyncioTestCase):

  async def test_async_if_after_with_then_except(self):
    cm = AsyncCM()

    async def bad_fn(v):
      raise ValueError('async boom')

    result = await (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .if_(lambda v: True, bad_fn)
      .except_(lambda exc: 'recovered')
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertTrue(cm.exited)


class TestIfIterateWithBreak(unittest.TestCase):

  def test_if_with_break_inside_iterate_fn(self):
    """break_ inside if_ fn within iterate stops iteration."""
    def check(x):
      if x >= 3:
        Chain.break_()
      return x

    result = list(Chain(range(10)).iterate(check))
    self.assertEqual(result, [0, 1, 2])

  def test_if_else_with_break_inside_iterate(self):
    """break_ inside else-like branch within iterate."""
    def check(x):
      if x < 3:
        return x * 10
      Chain.break_()

    result = list(Chain(range(10)).iterate(check))
    self.assertEqual(result, [0, 10, 20])


class TestIfIterateWithBreakAsync(IsolatedAsyncioTestCase):

  async def test_async_if_break_in_iterate(self):
    """break_ inside iterate fn stops async iteration."""
    def check(x):
      if x >= 3:
        Chain.break_()
      return x

    gen = Chain(range(10)).iterate(check)
    result = [item async for item in gen]
    self.assertEqual(result, [0, 1, 2])


class TestIfMapWithBreakAndReturn(unittest.TestCase):

  def test_break_with_value_from_if(self):
    """break_ with a value from inside if_-like condition in map."""
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: Chain.break_('stopped') if x == 3 else x)
      .run()
    )
    self.assertEqual(result, 'stopped')

  def test_return_from_if_inside_map_exits_outer(self):
    """return_ from if_-like condition inside map exits the outer chain."""
    result = (
      Chain([1, 2, 3, 4])
      .map(lambda x: Chain.return_('found') if x == 3 else x)
      .then(lambda v: 'should_not_reach')
      .run()
    )
    self.assertEqual(result, 'found')


class TestIfMapWithBreakAndReturnAsync(IsolatedAsyncioTestCase):

  async def test_async_return_from_if_inside_map(self):
    """return_ from if_-like condition in async map."""
    result = await (
      Chain(AsyncRange(5))
      .map(lambda x: Chain.return_('found') if x == 3 else x)
      .then(lambda v: 'should_not_reach')
      .run()
    )
    self.assertEqual(result, 'found')


class TestIfWithFreezeAndForEach(unittest.TestCase):

  def test_frozen_if_used_in_map(self):
    """Frozen chain with if_ used as map fn."""
    frozen = Chain().if_(lambda x: x > 2, lambda x: x * 10).else_(lambda x: x).freeze()
    result = Chain([1, 2, 3, 4]).map(frozen).run()
    self.assertEqual(result, [1, 2, 30, 40])

  def test_frozen_if_else_used_in_filter(self):
    """Frozen chain with if_/else_ used as filter predicate."""
    frozen = Chain().if_(lambda x: x > 2, lambda x: True).else_(lambda x: False).freeze()
    result = Chain([1, 2, 3, 4]).filter(frozen).run()
    self.assertEqual(result, [3, 4])


class TestIfWithFreezeAndForEachAsync(IsolatedAsyncioTestCase):

  async def test_async_frozen_if_in_map(self):
    async def pred(x):
      return x > 2

    frozen = Chain().if_(pred, lambda x: x * 10).else_(lambda x: x).freeze()
    result = await Chain([1, 2, 3, 4]).map(frozen).run()
    self.assertEqual(result, [1, 2, 30, 40])


if __name__ == '__main__':
  unittest.main()
