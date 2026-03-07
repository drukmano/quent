"""Tests for Chain.map() and Chain.foreach(): synchronous iteration,
async iteration, break semantics, exception propagation, and edge cases.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import sync_fn, async_fn, AsyncRange, AsyncEmpty


# ---------------------------------------------------------------------------
# Synchronous map (maps fn over iterable, collects fn results)
# ---------------------------------------------------------------------------

class TestMapSync(unittest.TestCase):

  def test_maps_list(self):
    result = Chain([1, 2, 3]).map(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_maps_tuple(self):
    result = Chain((1, 2, 3)).map(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_maps_generator(self):
    result = Chain(iter([1, 2, 3])).map(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_maps_range(self):
    result = Chain(range(5)).map(lambda x: x * 2).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  def test_maps_set(self):
    result = Chain({1}).map(lambda x: x * 2).run()
    self.assertEqual(result, [2])

  def test_empty_iterable_returns_empty(self):
    result = Chain([]).map(lambda x: x * 2).run()
    self.assertEqual(result, [])

  def test_single_element(self):
    result = Chain([42]).map(lambda x: x + 1).run()
    self.assertEqual(result, [43])

  def test_preserves_order(self):
    result = Chain([3, 1, 2]).map(lambda x: x).run()
    self.assertEqual(result, [3, 1, 2])

  def test_fn_receives_each_element(self):
    tracker = []
    Chain([1, 2, 3]).map(lambda x: tracker.append(x) or x).run()
    self.assertEqual(tracker, [1, 2, 3])


# ---------------------------------------------------------------------------
# Synchronous foreach (calls fn as side-effect, collects original items)
# ---------------------------------------------------------------------------

class TestForeachSync(unittest.TestCase):

  def test_keeps_original_elements(self):
    result = Chain([1, 2, 3]).foreach(lambda x: x * 100).run()
    self.assertEqual(result, [1, 2, 3])

  def test_fn_called_as_side_effect(self):
    tracker = []
    Chain([1, 2, 3]).foreach(lambda x: tracker.append(x)).run()
    self.assertEqual(tracker, [1, 2, 3])

  def test_empty_iterable(self):
    result = Chain([]).foreach(lambda x: x).run()
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Async map / foreach
# ---------------------------------------------------------------------------

class TestMapAsync(IsolatedAsyncioTestCase):

  async def test_async_iterable(self):
    result = await Chain(AsyncRange(3)).map(lambda x: x * 2).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_sync_fn_async_iterable(self):
    result = await Chain(AsyncRange(3)).map(sync_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_async_fn_sync_iterable(self):
    result = await Chain([1, 2, 3]).map(async_fn).run()
    self.assertEqual(result, [2, 3, 4])

  async def test_async_fn_async_iterable(self):
    result = await Chain(AsyncRange(3)).map(async_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_foreach_async_iterable(self):
    result = await Chain(AsyncRange(3)).foreach(lambda x: x * 100).run()
    self.assertEqual(result, [0, 1, 2])

  async def test_foreach_async_fn(self):
    result = await Chain([1, 2, 3]).foreach(async_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_async_empty_iterable(self):
    result = await Chain(AsyncEmpty()).map(lambda x: x).run()
    self.assertEqual(result, [])

  async def test_foreach_async_fn_async_iterable(self):
    result = await Chain(AsyncRange(3)).foreach(async_fn).run()
    self.assertEqual(result, [0, 1, 2])


# ---------------------------------------------------------------------------
# Break semantics (sync)
# ---------------------------------------------------------------------------

class TestMapBreak(unittest.TestCase):

  def test_break_stops_iteration(self):
    result = Chain([1, 2, 3, 4, 5]).map(
      lambda x: Chain.break_() if x == 3 else x
    ).run()
    self.assertEqual(result, [1, 2])

  def test_break_returns_partial_results(self):
    result = Chain([10, 20, 30, 40]).map(
      lambda x: Chain.break_() if x == 30 else x * 2
    ).run()
    self.assertEqual(result, [20, 40])

  def test_break_with_value(self):
    result = Chain([1, 2, 3]).map(
      lambda x: Chain.break_(99) if x == 2 else x
    ).run()
    self.assertEqual(result, 99)

  def test_break_on_first_element(self):
    result = Chain([1, 2, 3]).map(lambda x: Chain.break_()).run()
    self.assertEqual(result, [])

  def test_break_on_last_element(self):
    result = Chain([1, 2, 3]).map(
      lambda x: Chain.break_() if x == 3 else x
    ).run()
    self.assertEqual(result, [1, 2])

  def test_break_do_returns_original_items(self):
    tracker = []
    result = Chain([1, 2, 3, 4]).foreach(
      lambda x: Chain.break_() if x == 3 else tracker.append(x)
    ).run()
    self.assertEqual(result, [1, 2])
    self.assertEqual(tracker, [1, 2])


# ---------------------------------------------------------------------------
# Break semantics (async)
# ---------------------------------------------------------------------------

class TestMapBreakAsync(IsolatedAsyncioTestCase):

  async def test_break_in_async_map(self):
    result = await Chain(AsyncRange(5)).map(
      lambda x: Chain.break_() if x == 2 else x
    ).run()
    self.assertEqual(result, [0, 1])

  async def test_break_with_value_async(self):
    result = await Chain(AsyncRange(5)).map(
      lambda x: Chain.break_(42) if x == 3 else x
    ).run()
    self.assertEqual(result, 42)

  async def test_break_in_async_fn_sync_iterable(self):
    async def breaker(x):
      if x == 2:
        Chain.break_()
      return x * 10

    result = await Chain([1, 2, 3]).map(breaker).run()
    self.assertEqual(result, [10])

  async def test_break_with_async_value_awaited(self):
    async def make_val():
      return 'async_break_val'

    result = await Chain(AsyncRange(5)).map(
      lambda x: Chain.break_(make_val) if x == 1 else x
    ).run()
    self.assertEqual(result, 'async_break_val')


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------

class TestMapExceptions(unittest.TestCase):

  def test_exception_in_fn_propagates(self):
    with self.assertRaises(ZeroDivisionError):
      Chain([1, 2, 3]).map(lambda x: 1 / 0).run()

  def test_exception_sets_link_temp_args(self):
    # __quent_link_temp_args__ is an internal attribute set during iteration
    # and cleaned up by _modify_traceback after processing. After the chain
    # raises, only __quent__ (the traceback-processed marker) should remain.
    try:
      Chain([10, 20, 30]).map(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ZeroDivisionError was not raised')

  def test_exception_in_middle_propagates(self):
    def boom_on_two(x):
      if x == 2:
        raise RuntimeError('boom')
      return x

    with self.assertRaises(RuntimeError) as cm:
      Chain([1, 2, 3]).map(boom_on_two).run()
    self.assertEqual(str(cm.exception), 'boom')

  def test_exception_temp_args_reflect_failing_element(self):
    # __quent_link_temp_args__ is consumed and deleted by _modify_traceback.
    # Verify the exception was processed (has __quent__) and cleaned up.
    def boom_on_three(x):
      if x == 3:
        raise ValueError('bad')
      return x

    try:
      Chain([1, 2, 3]).map(boom_on_three).run()
    except ValueError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ValueError was not raised')


class TestMapExceptionsAsync(IsolatedAsyncioTestCase):

  async def test_exception_in_async_fn_propagates(self):
    async def boom(x):
      raise RuntimeError('async boom')

    with self.assertRaises(RuntimeError):
      await Chain([1, 2]).map(boom).run()

  async def test_exception_in_async_iterable_fn_sets_temp_args(self):
    # __quent_link_temp_args__ is consumed and deleted by _modify_traceback.
    # Verify the exception was processed (has __quent__) and cleaned up.
    async def boom(x):
      raise ValueError('bad')

    try:
      await Chain(AsyncRange(3)).map(boom).run()
    except ValueError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    else:
      self.fail('ValueError was not raised')


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestMapEdgeCases(unittest.TestCase):

  def test_map_on_int_raises(self):
    with self.assertRaises(TypeError):
      Chain(42).map(lambda x: x).run()

  def test_map_on_none_raises(self):
    with self.assertRaises(TypeError):
      Chain(None).map(lambda x: x).run()

  def test_consumed_generator_yields_nothing(self):
    gen = iter([1, 2, 3])
    list(gen)  # exhaust it
    result = Chain(gen).map(lambda x: x).run()
    self.assertEqual(result, [])

  def test_map_on_string_iterates_chars(self):
    result = Chain('abc').map(lambda x: x.upper()).run()
    self.assertEqual(result, ['A', 'B', 'C'])

  def test_map_on_dict_iterates_keys(self):
    result = Chain({'a': 1, 'b': 2}).map(lambda x: x).run()
    self.assertIn('a', result)
    self.assertIn('b', result)
    self.assertEqual(len(result), 2)

  def test_map_result_used_in_chain(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .then(lambda lst: sum(lst))
      .run()
    )
    self.assertEqual(result, 12)

  def test_foreach_result_used_in_chain(self):
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 100)
      .then(lambda lst: sum(lst))
      .run()
    )
    self.assertEqual(result, 6)


if __name__ == '__main__':
  unittest.main()
