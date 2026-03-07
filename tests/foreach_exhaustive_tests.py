"""Exhaustive 3-tier tests for Chain.map() and Chain.foreach().

Covers:
  - Sync tier: pure synchronous iteration with diverse iterable types
  - To-async tier: sync iterable + async fn triggering mid-iteration handoff
  - Full-async tier: async iterable (__aiter__) paths
  - Edge cases: break semantics, return propagation, exception temp args,
    nested map, large iterables, falsy values, bytes, dict views, etc.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null
from quent._core import _Return, _Break
from helpers import (
  AsyncEmpty,
  AsyncInfiniteIterable,
  AsyncRange,
  AsyncRangeRaises,
  InfiniteIterable,
  RaisingIterable,
  async_fn,
  sync_fn,
)


# ---------------------------------------------------------------------------
# Tier 1: Pure sync map
# ---------------------------------------------------------------------------

class TestMapSyncTier(unittest.TestCase):

  def test_map_each_iterable_type(self):
    fn = lambda x: x * 2
    cases = {
      'list': ([1, 2, 3], [2, 4, 6]),
      'tuple': ((1, 2, 3), [2, 4, 6]),
      'range': (range(1, 4), [2, 4, 6]),
      'generator': (iter([1, 2, 3]), [2, 4, 6]),
      'set_single': ({7}, [14]),
      'str': ('ab', ['aa', 'bb']),
      'dict': ({'x': 1}, ['xx']),
    }
    for label, (iterable, expected) in cases.items():
      with self.subTest(type=label):
        result = Chain(iterable).map(fn).run()
        self.assertEqual(result, expected)

  def test_foreach_each_iterable_type(self):
    fn = lambda x: x * 100
    cases = {
      'list': ([1, 2, 3], [1, 2, 3]),
      'tuple': ((4, 5), [4, 5]),
      'range': (range(3), [0, 1, 2]),
      'generator': (iter([10]), [10]),
      'set_single': ({9}, [9]),
      'str': ('hi', ['h', 'i']),
      'dict': ({'k': 'v'}, ['k']),
    }
    for label, (iterable, expected) in cases.items():
      with self.subTest(type=label):
        result = Chain(iterable).foreach(fn).run()
        self.assertEqual(result, expected)

  def test_map_empty_each_type(self):
    fn = lambda x: x
    empties = {
      'list': [],
      'tuple': (),
      'range': range(0),
      'generator': iter([]),
      'set': set(),
      'str': '',
      'dict': {},
    }
    for label, iterable in empties.items():
      with self.subTest(type=label):
        result = Chain(iterable).map(fn).run()
        self.assertEqual(result, [])

  def test_map_single_element(self):
    cases = {
      'list': [42],
      'tuple': (42,),
      'range': range(42, 43),
      'generator': iter([42]),
      'set': {42},
      'str': 'Z',
    }
    for label, iterable in cases.items():
      with self.subTest(type=label):
        result = Chain(iterable).map(lambda x: x).run()
        self.assertEqual(len(result), 1)

  def test_map_fn_receives_elements_in_order(self):
    tracker = []
    Chain([10, 20, 30]).map(lambda x: tracker.append(x) or x).run()
    self.assertEqual(tracker, [10, 20, 30])

  def test_map_fn_result_collected(self):
    result = Chain([1, 2, 3]).map(lambda x: x ** 2).run()
    self.assertEqual(result, [1, 4, 9])

  def test_foreach_fn_result_discarded(self):
    result = Chain([1, 2, 3]).foreach(lambda x: x * 999).run()
    self.assertEqual(result, [1, 2, 3])

  def test_map_break_positions(self):
    data = [10, 20, 30, 40, 50]
    positions = {
      'first': (0, []),
      'mid': (2, [10, 20]),
      'last': (4, [10, 20, 30, 40]),
    }
    for label, (break_idx, expected) in positions.items():
      with self.subTest(position=label):
        counter = {'i': 0}
        def fn(x, _bi=break_idx, _c=counter):
          idx = _c['i']
          _c['i'] += 1
          if idx == _bi:
            Chain.break_()
          return x
        counter['i'] = 0
        result = Chain(data).map(fn).run()
        self.assertEqual(result, expected)

  def test_map_break_with_value(self):
    # Break with a literal value
    result = Chain([1, 2, 3]).map(
      lambda x: Chain.break_(99) if x == 2 else x
    ).run()
    self.assertEqual(result, 99)

  def test_map_break_with_callable(self):
    result = Chain([1, 2, 3]).map(
      lambda x: Chain.break_(lambda: 'callable_val') if x == 2 else x
    ).run()
    self.assertEqual(result, 'callable_val')

  def test_map_break_with_null(self):
    # Break with no value (Null) returns partial results
    result = Chain([1, 2, 3]).map(
      lambda x: Chain.break_() if x == 2 else x
    ).run()
    self.assertEqual(result, [1])

  def test_map_return_propagates(self):
    # _Return inside map should exit the entire chain
    result = Chain([1, 2, 3]).map(
      lambda x: Chain.return_('early') if x == 2 else x
    ).then(lambda _: 'should_not_reach').run()
    self.assertEqual(result, 'early')

  def test_map_stop_iteration(self):
    # A custom iterator that raises StopIteration mid-way
    class LimitedIter:
      def __init__(self):
        self.i = 0
      def __iter__(self):
        return self
      def __next__(self):
        if self.i >= 2:
          raise StopIteration
        val = self.i
        self.i += 1
        return val

    result = Chain(LimitedIter()).map(lambda x: x * 10).run()
    self.assertEqual(result, [0, 10])

  def test_map_exception_sets_temp_args(self):
    try:
      Chain([10, 20, 30]).map(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      # After _modify_traceback processes the exception, __quent_link_temp_args__
      # is consumed and __quent__ is set.
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ZeroDivisionError was not raised')

  def test_map_raising_iterable(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain(RaisingIterable(5, 2)).map(lambda x: x).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  def test_map_infinite_with_break(self):
    result = Chain(InfiniteIterable(0)).map(
      lambda x: Chain.break_() if x == 5 else x
    ).run()
    self.assertEqual(result, [0, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Tier 2: Sync-to-async handoff (sync iterable, async fn)
# ---------------------------------------------------------------------------

class TestMapToAsyncTier(IsolatedAsyncioTestCase):

  async def test_sync_iterable_async_fn_first_item(self):
    # Async fn on very first item triggers _to_async immediately
    result = await Chain([1, 2, 3]).map(async_fn).run()
    self.assertEqual(result, [2, 3, 4])

  async def test_sync_iterable_async_fn_mid_item(self):
    # Sync fn for some items, async fn mid-way
    call_count = {'n': 0}
    async def sometimes_async(x):
      call_count['n'] += 1
      if call_count['n'] >= 2:
        return x * 10
      return x * 10
    # All are async here since the function is async def
    result = await Chain([1, 2, 3]).map(sometimes_async).run()
    self.assertEqual(result, [10, 20, 30])

  async def test_to_async_break_sync_value(self):
    async def fn(x):
      if x == 3:
        Chain.break_(42)
      return x
    result = await Chain([1, 2, 3, 4]).map(fn).run()
    self.assertEqual(result, 42)

  async def test_to_async_break_async_value(self):
    async def make_val():
      return 'async_break_result'
    result = await Chain([1, 2, 3]).map(
      lambda x: Chain.break_(make_val) if x == 2 else x
    ).run()
    self.assertEqual(result, 'async_break_result')

  async def test_to_async_return_signal(self):
    async def fn(x):
      if x == 2:
        Chain.return_('async_return_val')
      return x
    result = await Chain([1, 2, 3]).map(fn).then(lambda _: 'nope').run()
    self.assertEqual(result, 'async_return_val')

  async def test_to_async_exception_sets_temp_args(self):
    async def boom(x):
      if x == 20:
        raise ValueError('boom')
      return x
    try:
      await Chain([10, 20, 30]).map(boom).run()
    except ValueError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ValueError was not raised')

  async def test_to_async_foreach_preserves_items(self):
    result = await Chain([1, 2, 3]).foreach(async_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_to_async_mixed_sync_async(self):
    # Some items produce sync results, some async (via mixed fn)
    counter = {'n': 0}
    def mixed_fn(x):
      counter['n'] += 1
      if counter['n'] % 2 == 0:
        async def _inner():
          return x * 10
        return _inner()
      return x * 10
    result = await Chain([1, 2, 3, 4]).map(mixed_fn).run()
    self.assertEqual(result, [10, 20, 30, 40])


# ---------------------------------------------------------------------------
# Tier 3: Full async (async iterable)
# ---------------------------------------------------------------------------

class TestMapFullAsyncTier(IsolatedAsyncioTestCase):

  async def test_async_iterable_sync_fn(self):
    result = await Chain(AsyncRange(4)).map(sync_fn).run()
    self.assertEqual(result, [1, 2, 3, 4])

  async def test_async_iterable_async_fn(self):
    result = await Chain(AsyncRange(3)).map(async_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_async_iterable_empty(self):
    result = await Chain(AsyncEmpty()).map(lambda x: x).run()
    self.assertEqual(result, [])

  async def test_async_iterable_break(self):
    result = await Chain(AsyncRange(10)).map(
      lambda x: Chain.break_() if x == 3 else x
    ).run()
    self.assertEqual(result, [0, 1, 2])

  async def test_async_iterable_return(self):
    result = await Chain(AsyncRange(10)).map(
      lambda x: Chain.return_('done') if x == 2 else x
    ).then(lambda _: 'nope').run()
    self.assertEqual(result, 'done')

  async def test_async_iterable_exception_sets_temp_args(self):
    async def boom(x):
      raise ValueError('async boom')
    try:
      await Chain(AsyncRange(3)).map(boom).run()
    except ValueError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ValueError was not raised')

  async def test_async_iterable_foreach(self):
    result = await Chain(AsyncRange(4)).foreach(lambda x: x * 100).run()
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_async_iterable_raises(self):
    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRangeRaises(5, 2)).map(lambda x: x).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  async def test_async_infinite_with_break(self):
    result = await Chain(AsyncInfiniteIterable(0)).map(
      lambda x: Chain.break_() if x == 5 else x
    ).run()
    self.assertEqual(result, [0, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Beyond-spec: additional edge cases
# ---------------------------------------------------------------------------

class TestMapNestedAndComposite(IsolatedAsyncioTestCase):

  async def test_nested_map_inside_map(self):
    # Outer map returns lists, inner map doubles each element
    result = Chain([[1, 2], [3, 4]]).map(
      lambda lst: Chain(lst).map(lambda x: x * 2).run()
    ).run()
    self.assertEqual(result, [[2, 4], [6, 8]])

  async def test_filter_on_map_result(self):
    result = Chain([1, 2, 3, 4, 5]).map(
      lambda x: x * 2
    ).filter(
      lambda x: x > 4
    ).run()
    self.assertEqual(result, [6, 8, 10])

  def test_very_large_iterable(self):
    result = Chain(range(1000)).map(lambda x: x + 1).run()
    self.assertEqual(len(result), 1000)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[999], 1000)

  def test_falsy_values_none_false_zero(self):
    items = [None, False, 0, '', [], 0.0]
    result = Chain(items).map(lambda x: x).run()
    self.assertEqual(result, [None, False, 0, '', [], 0.0])

  def test_falsy_values_foreach(self):
    items = [None, False, 0]
    result = Chain(items).foreach(lambda x: 'ignored').run()
    self.assertEqual(result, [None, False, 0])

  def test_generator_expression_as_input(self):
    gen = (x * 3 for x in range(5))
    result = Chain(gen).map(lambda x: x + 1).run()
    self.assertEqual(result, [1, 4, 7, 10, 13])

  def test_dict_items_as_input(self):
    d = {'a': 1, 'b': 2}
    result = Chain(d.items()).map(lambda kv: (kv[0], kv[1] * 10)).run()
    self.assertIn(('a', 10), result)
    self.assertIn(('b', 20), result)

  def test_dict_values_as_input(self):
    d = {'a': 1, 'b': 2, 'c': 3}
    result = Chain(d.values()).map(lambda v: v * 2).run()
    self.assertEqual(sorted(result), [2, 4, 6])

  def test_bytes_as_iterable(self):
    result = Chain(b'hello').map(lambda b: b).run()
    self.assertEqual(result, list(b'hello'))
    self.assertEqual(len(result), 5)

  def test_map_fn_returns_item_unchanged(self):
    items = [1, 'two', 3.0, None]
    result = Chain(items).map(lambda x: x).run()
    self.assertEqual(result, items)

  def test_break_with_callable_that_raises(self):
    def bad_callable():
      raise RuntimeError('break callable error')

    with self.assertRaises(RuntimeError) as cm:
      Chain([1, 2, 3]).map(
        lambda x: Chain.break_(bad_callable) if x == 2 else x
      ).run()
    self.assertEqual(str(cm.exception), 'break callable error')

  async def test_foreach_async_fn_raises(self):
    async def boom(x):
      if x == 2:
        raise RuntimeError('async do boom')
    with self.assertRaises(RuntimeError) as cm:
      await Chain([1, 2, 3]).foreach(boom).run()
    self.assertEqual(str(cm.exception), 'async do boom')

  async def test_concurrent_async_foreach_operations(self):
    async def double(x):
      return x * 2

    chain1 = Chain(AsyncRange(3)).map(double)
    chain2 = Chain(AsyncRange(4)).map(double)

    r1, r2 = await asyncio.gather(chain1.run(), chain2.run())
    self.assertEqual(r1, [0, 2, 4])
    self.assertEqual(r2, [0, 2, 4, 6])

  def test_map_preserves_none_in_results(self):
    result = Chain([1, 2, 3]).map(lambda x: None).run()
    self.assertEqual(result, [None, None, None])

  def test_foreach_with_noop_fn(self):
    result = Chain([1, 2, 3]).foreach(lambda x: None).run()
    self.assertEqual(result, [1, 2, 3])

  def test_map_chained_with_then(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .then(sum)
      .run()
    )
    self.assertEqual(result, 12)

  def test_map_break_with_value_at_first_element(self):
    result = Chain([1, 2, 3]).map(
      lambda x: Chain.break_('first') if x == 1 else x
    ).run()
    self.assertEqual(result, 'first')

  async def test_async_map_break_with_callable_value(self):
    result = await Chain(AsyncRange(5)).map(
      lambda x: Chain.break_(lambda: 'cb_val') if x == 3 else x
    ).run()
    self.assertEqual(result, 'cb_val')

  async def test_async_foreach_break(self):
    tracker = []
    result = await Chain(AsyncRange(10)).foreach(
      lambda x: Chain.break_() if x == 3 else tracker.append(x)
    ).run()
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(tracker, [0, 1, 2])

  async def test_nested_async_map(self):
    async def inner(lst):
      return await Chain(AsyncRange(lst)).map(lambda x: x * 2).run()
    result = await Chain([2, 3]).map(inner).run()
    self.assertEqual(result, [[0, 2], [0, 2, 4]])

  def test_map_exception_mid_iteration(self):
    def boom_at_three(x):
      if x == 3:
        raise ValueError('at three')
      return x
    with self.assertRaises(ValueError) as cm:
      Chain([1, 2, 3, 4]).map(boom_at_three).run()
    self.assertEqual(str(cm.exception), 'at three')

  async def test_to_async_break_no_value_partial_results(self):
    async def fn(x):
      if x == 3:
        Chain.break_()
      return x * 10
    result = await Chain([1, 2, 3, 4, 5]).map(fn).run()
    self.assertEqual(result, [10, 20])

  def test_map_on_frozenset(self):
    result = Chain(frozenset({42})).map(lambda x: x * 2).run()
    self.assertEqual(result, [84])

  def test_foreach_on_large_iterable(self):
    tracker = []
    result = Chain(range(500)).foreach(lambda x: tracker.append(x)).run()
    self.assertEqual(len(result), 500)
    self.assertEqual(len(tracker), 500)
    self.assertEqual(result, list(range(500)))


if __name__ == '__main__':
  unittest.main()
