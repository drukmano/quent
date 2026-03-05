"""Tests for Chain.filter(): sync filtering, async filtering, and edge cases."""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from helpers import AsyncRange, AsyncEmpty


class TestFilterSync(unittest.TestCase):

  def test_keeps_truthy(self):
    result = Chain([0, 1, 2, 3]).filter(lambda x: x > 1).run()
    self.assertEqual(result, [2, 3])

  def test_with_predicate_gt(self):
    result = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).run()
    self.assertEqual(result, [4, 5])

  def test_empty_iterable(self):
    result = Chain([]).filter(lambda x: True).run()
    self.assertEqual(result, [])

  def test_all_pass(self):
    result = Chain([1, 2, 3]).filter(lambda x: True).run()
    self.assertEqual(result, [1, 2, 3])

  def test_none_pass(self):
    result = Chain([1, 2, 3]).filter(lambda x: False).run()
    self.assertEqual(result, [])

  def test_preserves_order(self):
    result = Chain([3, 1, 4, 1, 5]).filter(lambda x: x > 2).run()
    self.assertEqual(result, [3, 4, 5])


class TestFilterAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_iterable(self):
    result = await Chain(AsyncRange(5)).filter(lambda x: x > 2).run()
    self.assertEqual(result, [3, 4])

  async def test_async_predicate(self):
    async def async_pred(x):
      return x > 2

    result = await Chain([1, 2, 3, 4]).filter(async_pred).run()
    self.assertEqual(result, [3, 4])

  async def test_async_both(self):
    async def async_pred(x):
      return x % 2 == 0

    result = await Chain(AsyncRange(6)).filter(async_pred).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_sync_iterable_async_predicate(self):
    async def async_pred(x):
      return x >= 2

    result = await Chain([1, 2, 3]).filter(async_pred).run()
    self.assertEqual(result, [2, 3])

  async def test_async_empty_iterable(self):
    result = await Chain(AsyncEmpty()).filter(lambda x: True).run()
    self.assertEqual(result, [])

  async def test_async_predicate_none_pass(self):
    async def async_pred(x):
      return False

    result = await Chain(AsyncRange(4)).filter(async_pred).run()
    self.assertEqual(result, [])

  async def test_async_predicate_all_pass(self):
    async def async_pred(x):
      return True

    result = await Chain(AsyncRange(3)).filter(async_pred).run()
    self.assertEqual(result, [0, 1, 2])


class TestFilterEdgeCases(unittest.TestCase):

  def test_on_int_raises(self):
    with self.assertRaises(TypeError):
      Chain(42).filter(lambda x: True).run()

  def test_on_none_raises(self):
    with self.assertRaises(TypeError):
      Chain(None).filter(lambda x: True).run()

  def test_consumed_generator(self):
    gen = iter([1, 2, 3])
    # Exhaust the generator.
    for _ in gen:
      pass
    result = Chain(gen).filter(lambda x: True).run()
    self.assertEqual(result, [])

  def test_predicate_exception_propagates(self):
    with self.assertRaises(ZeroDivisionError):
      Chain([1, 2, 3]).filter(lambda x: 1 / 0).run()

  def test_exception_sets_link_temp_args(self):
    try:
      Chain([10, 20, 30]).filter(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      # The dict should contain at least one entry whose value is a dict
      # holding the item that was being processed when the error occurred.
      temp_args = exc.__quent_link_temp_args__
      self.assertIsInstance(temp_args, dict)
      self.assertTrue(len(temp_args) > 0)
      # The first item in the iterable (10) is where the predicate blows up.
      values = list(temp_args.values())
      self.assertEqual(values[0], {'item': 10, 'index': 0})
    else:
      self.fail('ZeroDivisionError was not raised')

  def test_filter_on_string(self):
    result = Chain('hello').filter(lambda c: c in 'aeiou').run()
    self.assertEqual(result, ['e', 'o'])

  def test_falsy_values_filtered_correctly(self):
    result = Chain([0, None, '', False, 1, 'a']).filter(bool).run()
    self.assertEqual(result, [1, 'a'])

  def test_filter_with_identity(self):
    result = Chain([0, 1, 2, None, 3]).filter(lambda x: x).run()
    self.assertEqual(result, [1, 2, 3])


if __name__ == '__main__':
  unittest.main()
