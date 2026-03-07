"""Exhaustive 3-tier tests for Chain.filter().

Covers:
  - Sync tier: pure synchronous filtering with diverse iterable types
  - To-async tier: sync iterable + async predicate triggering mid-iteration handoff
  - Full-async tier: async iterable (__aiter__) paths
  - Edge cases: control flow propagation, exception temp args, nested filters,
    large iterables, falsy values, bytes, dict views, stateful predicates, etc.
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
# Tier 1: Pure sync filter
# ---------------------------------------------------------------------------

class TestFilterSyncTier(unittest.TestCase):

  def test_filter_each_iterable_type(self):
    pred = lambda x: x > 1 if isinstance(x, int) else True
    cases = {
      'list': ([0, 1, 2, 3], [2, 3]),
      'tuple': ((0, 1, 2, 3), [2, 3]),
      'range': (range(4), [2, 3]),
      'generator': (iter([0, 1, 2, 3]), [2, 3]),
      'set_single': ({5}, [5]),
    }
    for label, (iterable, expected) in cases.items():
      with self.subTest(type=label):
        result = Chain(iterable).filter(pred).run()
        self.assertEqual(result, expected)

  def test_filter_empty_each_type(self):
    pred = lambda x: True
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
        result = Chain(iterable).filter(pred).run()
        self.assertEqual(result, [])

  def test_filter_all_pass(self):
    result = Chain([1, 2, 3, 4, 5]).filter(lambda x: True).run()
    self.assertEqual(result, [1, 2, 3, 4, 5])

  def test_filter_none_pass(self):
    result = Chain([1, 2, 3]).filter(lambda x: False).run()
    self.assertEqual(result, [])

  def test_filter_mixed_truthiness(self):
    items = [0, 1, None, '', 'a', [], [1]]
    result = Chain(items).filter(bool).run()
    self.assertEqual(result, [1, 'a', [1]])

  def test_filter_preserves_order(self):
    result = Chain([5, 3, 1, 4, 2]).filter(lambda x: x > 2).run()
    self.assertEqual(result, [5, 3, 4])

  def test_filter_exception_sets_temp_args(self):
    try:
      Chain([10, 20, 30]).filter(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ZeroDivisionError was not raised')

  def test_filter_control_flow_propagates(self):
    # _Return inside a filter fn should exit the chain
    result = Chain([1, 2, 3]).filter(
      lambda x: Chain.return_('early') if x == 2 else True
    ).then(lambda _: 'nope').run()
    self.assertEqual(result, 'early')

  def test_filter_raising_iterable(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain(RaisingIterable(5, 2)).filter(lambda x: True).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  def test_filter_infinite_with_return(self):
    # InfiniteIterable + _Return in predicate to exit
    result = Chain(InfiniteIterable(0)).filter(
      lambda x: Chain.return_('stopped') if x == 5 else True
    ).run()
    self.assertEqual(result, 'stopped')


# ---------------------------------------------------------------------------
# Tier 2: Sync-to-async handoff (sync iterable, async predicate)
# ---------------------------------------------------------------------------

class TestFilterToAsyncTier(IsolatedAsyncioTestCase):

  async def test_sync_iterable_async_predicate(self):
    async def pred(x):
      return x > 2
    result = await Chain([1, 2, 3, 4, 5]).filter(pred).run()
    self.assertEqual(result, [3, 4, 5])

  async def test_to_async_exception_sets_temp_args(self):
    async def boom(x):
      if x == 20:
        raise ValueError('async filter boom')
      return True
    try:
      await Chain([10, 20, 30]).filter(boom).run()
    except ValueError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ValueError was not raised')

  async def test_to_async_control_flow(self):
    async def pred(x):
      if x == 3:
        Chain.return_('async_early')
      return x > 1
    result = await Chain([1, 2, 3, 4]).filter(pred).then(lambda _: 'nope').run()
    self.assertEqual(result, 'async_early')

  async def test_to_async_mixed_predicates(self):
    # Predicate that is sometimes sync, sometimes async
    counter = {'n': 0}
    def mixed_pred(x):
      counter['n'] += 1
      if counter['n'] % 2 == 0:
        async def _async_check():
          return x > 2
        return _async_check()
      return x > 2
    result = await Chain([1, 2, 3, 4, 5]).filter(mixed_pred).run()
    self.assertEqual(result, [3, 4, 5])


# ---------------------------------------------------------------------------
# Tier 3: Full async (async iterable)
# ---------------------------------------------------------------------------

class TestFilterFullAsyncTier(IsolatedAsyncioTestCase):

  async def test_async_iterable_sync_predicate(self):
    result = await Chain(AsyncRange(6)).filter(lambda x: x % 2 == 0).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_async_iterable_async_predicate(self):
    async def pred(x):
      return x > 2
    result = await Chain(AsyncRange(5)).filter(pred).run()
    self.assertEqual(result, [3, 4])

  async def test_async_iterable_empty(self):
    result = await Chain(AsyncEmpty()).filter(lambda x: True).run()
    self.assertEqual(result, [])

  async def test_async_iterable_exception(self):
    async def boom(x):
      raise ValueError('async filter exc')
    try:
      await Chain(AsyncRange(3)).filter(boom).run()
    except ValueError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ValueError was not raised')

  async def test_async_iterable_control_flow(self):
    result = await Chain(AsyncRange(10)).filter(
      lambda x: Chain.return_('async_done') if x == 4 else True
    ).then(lambda _: 'nope').run()
    self.assertEqual(result, 'async_done')


# ---------------------------------------------------------------------------
# Beyond-spec: additional edge cases
# ---------------------------------------------------------------------------

class TestFilterAdditional(IsolatedAsyncioTestCase):

  async def test_filter_on_map_result(self):
    result = Chain([1, 2, 3, 4, 5]).map(
      lambda x: x * 2
    ).filter(
      lambda x: x > 4
    ).run()
    self.assertEqual(result, [6, 8, 10])

  def test_filter_very_large_iterable(self):
    result = Chain(range(1000)).filter(lambda x: x % 100 == 0).run()
    self.assertEqual(result, [0, 100, 200, 300, 400, 500, 600, 700, 800, 900])

  def test_filter_falsy_values_preserved(self):
    # All items pass: falsy items should still be in result if predicate is True
    items = [0, None, False, '', []]
    result = Chain(items).filter(lambda x: True).run()
    self.assertEqual(result, [0, None, False, '', []])

  def test_filter_falsy_values_removed_by_bool(self):
    items = [0, 1, None, '', 'a', False, True, [], [1], 0.0]
    result = Chain(items).filter(bool).run()
    self.assertEqual(result, [1, 'a', True, [1]])

  def test_filter_generator_expression(self):
    gen = (x for x in range(10))
    result = Chain(gen).filter(lambda x: x > 6).run()
    self.assertEqual(result, [7, 8, 9])

  def test_filter_dict_items(self):
    d = {'a': 1, 'b': 2, 'c': 3}
    result = Chain(d.items()).filter(lambda kv: kv[1] > 1).run()
    self.assertIn(('b', 2), result)
    self.assertIn(('c', 3), result)
    self.assertEqual(len(result), 2)

  def test_filter_dict_values(self):
    d = {'a': 10, 'b': 5, 'c': 20}
    result = Chain(d.values()).filter(lambda v: v >= 10).run()
    self.assertEqual(sorted(result), [10, 20])

  def test_filter_bytes_as_iterable(self):
    # bytes iterates as integers
    result = Chain(b'\x00\x01\x02\x03').filter(lambda b: b > 1).run()
    self.assertEqual(result, [2, 3])

  def test_filter_predicate_modifies_external_state(self):
    seen = []
    def stateful_pred(x):
      seen.append(x)
      return x > 2
    result = Chain([1, 2, 3, 4, 5]).filter(stateful_pred).run()
    self.assertEqual(result, [3, 4, 5])
    self.assertEqual(seen, [1, 2, 3, 4, 5])

  def test_filter_with_identity_function(self):
    items = [0, 1, '', 'a', None, True, False]
    # identity: truthy items pass
    result = Chain(items).filter(lambda x: x).run()
    self.assertEqual(result, [1, 'a', True])

  def test_filter_string_chars(self):
    result = Chain('Hello World').filter(lambda c: c.isupper()).run()
    self.assertEqual(result, ['H', 'W'])

  def test_filter_chained_with_then(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 2)
      .then(sum)
      .run()
    )
    self.assertEqual(result, 12)

  async def test_filter_async_predicate_all_pass(self):
    async def pred(x):
      return True
    result = await Chain(AsyncRange(5)).filter(pred).run()
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_filter_async_predicate_none_pass(self):
    async def pred(x):
      return False
    result = await Chain(AsyncRange(5)).filter(pred).run()
    self.assertEqual(result, [])

  async def test_concurrent_async_filter_operations(self):
    async def pred(x):
      return x > 1

    chain1 = Chain(AsyncRange(4)).filter(pred)
    chain2 = Chain(AsyncRange(6)).filter(pred)

    r1, r2 = await asyncio.gather(chain1.run(), chain2.run())
    self.assertEqual(r1, [2, 3])
    self.assertEqual(r2, [2, 3, 4, 5])

  async def test_filter_async_iterable_raises(self):
    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRangeRaises(5, 2)).filter(lambda x: True).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  def test_filter_on_frozenset(self):
    result = Chain(frozenset({1, 2, 3, 4, 5})).filter(lambda x: x > 3).run()
    self.assertEqual(sorted(result), [4, 5])

  def test_filter_preserves_duplicates(self):
    result = Chain([1, 2, 2, 3, 3, 3]).filter(lambda x: x > 1).run()
    self.assertEqual(result, [2, 2, 3, 3, 3])

  async def test_filter_after_async_map(self):
    async def double(x):
      return x * 2
    result = await Chain(AsyncRange(5)).map(double).filter(lambda x: x > 4).run()
    self.assertEqual(result, [6, 8])

  def test_filter_exception_mid_iteration(self):
    def pred(x):
      if x == 3:
        raise ValueError('at three')
      return x > 0
    with self.assertRaises(ValueError) as cm:
      Chain([1, 2, 3, 4]).filter(pred).run()
    self.assertEqual(str(cm.exception), 'at three')

  async def test_async_filter_with_sync_raising_predicate(self):
    def pred(x):
      if x == 2:
        raise RuntimeError('sync pred error')
      return True
    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRange(5)).filter(pred).run()
    self.assertEqual(str(cm.exception), 'sync pred error')

  def test_filter_large_iterable_all_pass(self):
    result = Chain(range(1000)).filter(lambda x: True).run()
    self.assertEqual(len(result), 1000)
    self.assertEqual(result, list(range(1000)))

  def test_filter_large_iterable_none_pass(self):
    result = Chain(range(1000)).filter(lambda x: False).run()
    self.assertEqual(result, [])


if __name__ == '__main__':
  unittest.main()
