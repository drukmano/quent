"""Tests for do() callable validation."""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain


class DoValidationTests(unittest.TestCase):
  """Verify that do() rejects non-callable arguments eagerly."""

  def test_regular_function(self):
    called = []
    def side_effect(v):
      called.append(v)
    result = Chain(10).do(side_effect).run()
    self.assertEqual(result, 10)
    self.assertEqual(called, [10])

  def test_lambda(self):
    results = []
    result = Chain(5).do(lambda v: results.append(v)).run()
    self.assertEqual(result, 5)
    self.assertEqual(results, [5])

  def test_callable_instance(self):
    class MyCallable:
      def __init__(self):
        self.called_with = []
      def __call__(self, v):
        self.called_with.append(v)

    obj = MyCallable()
    result = Chain(42).do(obj).run()
    self.assertEqual(result, 42)
    self.assertEqual(obj.called_with, [42])

  def test_rejects_int(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(1).do(123)
    self.assertEqual(str(ctx.exception), 'do() requires a callable, got int')

  def test_rejects_string(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(1).do('not_callable')
    self.assertEqual(str(ctx.exception), 'do() requires a callable, got str')

  def test_rejects_none(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(1).do(None)
    self.assertEqual(str(ctx.exception), 'do() requires a callable, got NoneType')

  def test_rejects_list(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(1).do([1, 2, 3])
    self.assertEqual(str(ctx.exception), 'do() requires a callable, got list')

  def test_error_message_format(self):
    """Verify the exact error message pattern."""
    with self.assertRaises(TypeError) as ctx:
      Chain(1).do(3.14)
    self.assertIn('do() requires a callable, got float', str(ctx.exception))


class DoValidationAsyncTests(unittest.IsolatedAsyncioTestCase):
  """Verify that do() works with async callables."""

  async def test_async_callable(self):
    called = []
    async def async_side_effect(v):
      called.append(v)

    result = await Chain(7).do(async_side_effect).run()
    self.assertEqual(result, 7)
    self.assertEqual(called, [7])


if __name__ == '__main__':
  unittest.main()
