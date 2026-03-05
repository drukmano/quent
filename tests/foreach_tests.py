"""Tests for Chain.foreach() and Chain.foreach_do(): synchronous iteration,
async iteration, break semantics, exception propagation, and edge cases.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import sync_fn, async_fn, AsyncRange, AsyncEmpty


# ---------------------------------------------------------------------------
# Synchronous foreach (maps fn over iterable, collects fn results)
# ---------------------------------------------------------------------------

class TestForeachSync(unittest.TestCase):

  def test_maps_list(self):
    result = Chain([1, 2, 3]).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_maps_tuple(self):
    result = Chain((1, 2, 3)).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_maps_generator(self):
    result = Chain(iter([1, 2, 3])).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_maps_range(self):
    result = Chain(range(5)).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  def test_maps_set(self):
    result = Chain({1}).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [2])

  def test_empty_iterable_returns_empty(self):
    result = Chain([]).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [])

  def test_single_element(self):
    result = Chain([42]).foreach(lambda x: x + 1).run()
    self.assertEqual(result, [43])

  def test_preserves_order(self):
    result = Chain([3, 1, 2]).foreach(lambda x: x).run()
    self.assertEqual(result, [3, 1, 2])

  def test_fn_receives_each_element(self):
    tracker = []
    Chain([1, 2, 3]).foreach(lambda x: tracker.append(x) or x).run()
    self.assertEqual(tracker, [1, 2, 3])


# ---------------------------------------------------------------------------
# Synchronous foreach_do (calls fn as side-effect, collects original items)
# ---------------------------------------------------------------------------

class TestForeachDoSync(unittest.TestCase):

  def test_keeps_original_elements(self):
    result = Chain([1, 2, 3]).foreach_do(lambda x: x * 100).run()
    self.assertEqual(result, [1, 2, 3])

  def test_fn_called_as_side_effect(self):
    tracker = []
    Chain([1, 2, 3]).foreach_do(lambda x: tracker.append(x)).run()
    self.assertEqual(tracker, [1, 2, 3])

  def test_empty_iterable(self):
    result = Chain([]).foreach_do(lambda x: x).run()
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Async foreach / foreach_do
# ---------------------------------------------------------------------------

class TestForeachAsync(IsolatedAsyncioTestCase):

  async def test_async_iterable(self):
    result = await Chain(AsyncRange(3)).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_sync_fn_async_iterable(self):
    result = await Chain(AsyncRange(3)).foreach(sync_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_async_fn_sync_iterable(self):
    result = await Chain([1, 2, 3]).foreach(async_fn).run()
    self.assertEqual(result, [2, 3, 4])

  async def test_async_fn_async_iterable(self):
    result = await Chain(AsyncRange(3)).foreach(async_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_foreach_do_async_iterable(self):
    result = await Chain(AsyncRange(3)).foreach_do(lambda x: x * 100).run()
    self.assertEqual(result, [0, 1, 2])

  async def test_foreach_do_async_fn(self):
    result = await Chain([1, 2, 3]).foreach_do(async_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_async_empty_iterable(self):
    result = await Chain(AsyncEmpty()).foreach(lambda x: x).run()
    self.assertEqual(result, [])

  async def test_foreach_do_async_fn_async_iterable(self):
    result = await Chain(AsyncRange(3)).foreach_do(async_fn).run()
    self.assertEqual(result, [0, 1, 2])


# ---------------------------------------------------------------------------
# Break semantics (sync)
# ---------------------------------------------------------------------------

class TestForeachBreak(unittest.TestCase):

  def test_break_stops_iteration(self):
    result = Chain([1, 2, 3, 4, 5]).foreach(
      lambda x: Chain.break_() if x == 3 else x
    ).run()
    self.assertEqual(result, [1, 2])

  def test_break_returns_partial_results(self):
    result = Chain([10, 20, 30, 40]).foreach(
      lambda x: Chain.break_() if x == 30 else x * 2
    ).run()
    self.assertEqual(result, [20, 40])

  def test_break_with_value(self):
    result = Chain([1, 2, 3]).foreach(
      lambda x: Chain.break_(99) if x == 2 else x
    ).run()
    self.assertEqual(result, 99)

  def test_break_on_first_element(self):
    result = Chain([1, 2, 3]).foreach(lambda x: Chain.break_()).run()
    self.assertEqual(result, [])

  def test_break_on_last_element(self):
    result = Chain([1, 2, 3]).foreach(
      lambda x: Chain.break_() if x == 3 else x
    ).run()
    self.assertEqual(result, [1, 2])

  def test_break_do_returns_original_items(self):
    tracker = []
    result = Chain([1, 2, 3, 4]).foreach_do(
      lambda x: Chain.break_() if x == 3 else tracker.append(x)
    ).run()
    self.assertEqual(result, [1, 2])
    self.assertEqual(tracker, [1, 2])


# ---------------------------------------------------------------------------
# Break semantics (async)
# ---------------------------------------------------------------------------

class TestForeachBreakAsync(IsolatedAsyncioTestCase):

  async def test_break_in_async_foreach(self):
    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.break_() if x == 2 else x
    ).run()
    self.assertEqual(result, [0, 1])

  async def test_break_with_value_async(self):
    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.break_(42) if x == 3 else x
    ).run()
    self.assertEqual(result, 42)

  async def test_break_in_async_fn_sync_iterable(self):
    async def breaker(x):
      if x == 2:
        Chain.break_()
      return x * 10

    result = await Chain([1, 2, 3]).foreach(breaker).run()
    self.assertEqual(result, [10])

  async def test_break_with_async_value_awaited(self):
    async def make_val():
      return 'async_break_val'

    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.break_(make_val) if x == 1 else x
    ).run()
    self.assertEqual(result, 'async_break_val')


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------

class TestForeachExceptions(unittest.TestCase):

  def test_exception_in_fn_propagates(self):
    with self.assertRaises(ZeroDivisionError):
      Chain([1, 2, 3]).foreach(lambda x: 1 / 0).run()

  def test_exception_sets_link_temp_args(self):
    try:
      Chain([10, 20, 30]).foreach(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      # The temp args dict should contain an entry whose value is (item,)
      # where item is the element that was being processed.
      values = list(exc.__quent_link_temp_args__.values())
      self.assertTrue(len(values) > 0)
      # The first item processed is 10, so temp args should be (10,).
      self.assertEqual(values[0], (10,))
    else:
      self.fail('ZeroDivisionError was not raised')

  def test_exception_in_middle_propagates(self):
    def boom_on_two(x):
      if x == 2:
        raise RuntimeError('boom')
      return x

    with self.assertRaises(RuntimeError) as cm:
      Chain([1, 2, 3]).foreach(boom_on_two).run()
    self.assertEqual(str(cm.exception), 'boom')

  def test_exception_temp_args_reflect_failing_element(self):
    def boom_on_three(x):
      if x == 3:
        raise ValueError('bad')
      return x

    try:
      Chain([1, 2, 3]).foreach(boom_on_three).run()
    except ValueError as exc:
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (3,))
    else:
      self.fail('ValueError was not raised')


class TestForeachExceptionsAsync(IsolatedAsyncioTestCase):

  async def test_exception_in_async_fn_propagates(self):
    async def boom(x):
      raise RuntimeError('async boom')

    with self.assertRaises(RuntimeError):
      await Chain([1, 2]).foreach(boom).run()

  async def test_exception_in_async_iterable_fn_sets_temp_args(self):
    async def boom(x):
      raise ValueError('bad')

    try:
      await Chain(AsyncRange(3)).foreach(boom).run()
    except ValueError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (0,))
    else:
      self.fail('ValueError was not raised')


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestForeachEdgeCases(unittest.TestCase):

  def test_foreach_on_int_raises(self):
    with self.assertRaises(TypeError):
      Chain(42).foreach(lambda x: x).run()

  def test_foreach_on_none_raises(self):
    with self.assertRaises(TypeError):
      Chain(None).foreach(lambda x: x).run()

  def test_consumed_generator_yields_nothing(self):
    gen = iter([1, 2, 3])
    list(gen)  # exhaust it
    result = Chain(gen).foreach(lambda x: x).run()
    self.assertEqual(result, [])

  def test_foreach_on_string_iterates_chars(self):
    result = Chain('abc').foreach(lambda x: x.upper()).run()
    self.assertEqual(result, ['A', 'B', 'C'])

  def test_foreach_on_dict_iterates_keys(self):
    result = Chain({'a': 1, 'b': 2}).foreach(lambda x: x).run()
    self.assertIn('a', result)
    self.assertIn('b', result)
    self.assertEqual(len(result), 2)

  def test_foreach_result_used_in_chain(self):
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .then(lambda lst: sum(lst))
      .run()
    )
    self.assertEqual(result, 12)

  def test_foreach_do_result_used_in_chain(self):
    result = (
      Chain([1, 2, 3])
      .foreach_do(lambda x: x * 100)
      .then(lambda lst: sum(lst))
      .run()
    )
    self.assertEqual(result, 6)


if __name__ == '__main__':
  unittest.main()
