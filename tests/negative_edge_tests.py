"""Negative tests and edge-case behavior verification for the Chain API."""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain, Null, QuentException
from quent._core import _Return, _Break


class NegativeEdgeTests(unittest.TestCase):

  def test_then_after_run(self):
    """Calling .then() after .run() should work -- chain is mutable, run doesn't freeze it."""
    c = Chain(1).then(lambda x: x + 1)
    result1 = c.run()
    self.assertEqual(result1, 2)
    # Append more links after run
    c.then(lambda x: x * 10)
    result2 = c.run()
    self.assertEqual(result2, 20)

  def test_run_with_excessive_positional_args(self):
    """run(1, 2, 3) passes extra args to the root link."""
    def multi_arg(a, b, c):
      return a + b + c
    c = Chain().then(lambda x: x)
    result = c.run(multi_arg, 1, 2, 3)
    # run(v=multi_arg, args=(1,2,3)) creates a temp Link(multi_arg, (1,2,3), {})
    # _evaluate_value: v=multi_arg, args=(1,2,3) -> multi_arg(1,2,3) = 6
    # then lambda x: x gets 6 -> returns 6
    self.assertEqual(result, 6)

  def test_double_run(self):
    """Calling chain.run() twice should produce identical results."""
    c = Chain(5).then(lambda x: x * 3)
    result1 = c.run()
    result2 = c.run()
    self.assertEqual(result1, 15)
    self.assertEqual(result2, 15)

  def test_empty_gather(self):
    """gather() with no functions should return empty list."""
    result = Chain(5).gather().run()
    self.assertEqual(result, [])

  def test_single_function_gather(self):
    """gather(fn) with a single function should return list with one element."""
    result = Chain(5).gather(lambda x: x * 2).run()
    self.assertEqual(result, [10])

  def test_map_with_non_iterable(self):
    """map() on a non-iterable should raise TypeError."""
    c = Chain(42).map(lambda x: x)
    with self.assertRaises(TypeError):
      c.run()

  def test_filter_with_non_iterable(self):
    """filter() on a non-iterable should raise TypeError."""
    c = Chain(42).filter(lambda x: x)
    with self.assertRaises(TypeError):
      c.run()

  def test_break_at_top_level(self):
    """break_() at top level (not in iteration) should raise QuentException."""
    c = Chain().then(lambda: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('map/foreach iteration', str(ctx.exception))

  def test_return_with_value(self):
    """return_(42) should cause the chain to return 42 early."""
    result = Chain(1).then(lambda x: Chain.return_(42)).then(lambda x: x + 100).run()
    self.assertEqual(result, 42)

  def test_return_without_value(self):
    """return_() without a value should return None."""
    result = Chain(1).then(lambda x: Chain.return_()).then(lambda x: x + 100).run()
    self.assertIsNone(result)

  def test_map_validation_error_message(self):
    """Verify exact error message from map() when passed non-callable."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).map(42)
    self.assertEqual(str(ctx.exception), 'map() requires a callable, got int')

  def test_foreach_validation_error_message(self):
    """Verify exact error message from foreach() when passed non-callable."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).foreach('hello')
    self.assertEqual(str(ctx.exception), 'foreach() requires a callable, got str')

  def test_filter_validation_error_message(self):
    """Verify exact error message from filter() when passed non-callable."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1]).filter(None)
    self.assertEqual(str(ctx.exception), 'filter() requires a callable, got NoneType')

  def test_gather_validation_error_message(self):
    """Verify exact error message from gather() when passed non-callable."""
    with self.assertRaises(TypeError) as ctx:
      Chain(5).gather(lambda x: x, 42)
    self.assertEqual(str(ctx.exception), 'gather() requires all arguments to be callable, got int')

  def test_gather_validation_first_arg_non_callable(self):
    """gather() with first arg non-callable should raise TypeError."""
    with self.assertRaises(TypeError) as ctx:
      Chain(5).gather(42, lambda x: x)
    self.assertEqual(str(ctx.exception), 'gather() requires all arguments to be callable, got int')

  def test_gather_validation_all_non_callable(self):
    """gather() with all non-callable args should raise TypeError on the first."""
    with self.assertRaises(TypeError) as ctx:
      Chain(5).gather('a', 'b', 'c')
    self.assertEqual(str(ctx.exception), 'gather() requires all arguments to be callable, got str')

  def test_chain_run_returns_none_when_empty(self):
    """Empty chain with no root should return None."""
    result = Chain().run()
    self.assertIsNone(result)

  def test_chain_root_none_value(self):
    """Chain(None).run() should return None (None is not callable, passes through)."""
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_double_except_raises(self):
    """Registering two except_ handlers should raise QuentException."""
    c = Chain(1).except_(lambda rv, e: None)
    with self.assertRaises(QuentException):
      c.except_(lambda rv, e: None)

  def test_double_finally_raises(self):
    """Registering two finally_ handlers should raise QuentException."""
    c = Chain(1).finally_(lambda v: None)
    with self.assertRaises(QuentException):
      c.finally_(lambda v: None)

  def test_break_error_message_exact(self):
    """Verify the exact break_() error message mentions map/foreach iteration."""
    c = Chain().then(lambda: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertEqual(
      str(ctx.exception),
      'Chain.break_() cannot be used outside of a map/foreach iteration.'
    )


class NegativeAsyncTests(unittest.IsolatedAsyncioTestCase):

  async def test_break_at_top_level_async(self):
    """break_() at top level in async path should raise QuentException."""
    async def async_break(x):
      Chain.break_()

    c = Chain(1).then(async_break)
    with self.assertRaises(QuentException) as ctx:
      await c.run()
    self.assertIn('map/foreach iteration', str(ctx.exception))

  async def test_return_with_value_async(self):
    """return_(42) in async path should cause early return."""
    async def async_return(x):
      Chain.return_(x * 10)

    result = await Chain(5).then(async_return).then(lambda x: x + 100).run()
    self.assertEqual(result, 50)

  async def test_map_with_non_iterable_async(self):
    """map() on a non-iterable should raise TypeError in async path."""
    async def async_val(x):
      return 42

    c = Chain(0).then(async_val).map(lambda x: x)
    with self.assertRaises(TypeError):
      await c.run()

  async def test_double_run_async(self):
    """Calling chain.run() twice in async should produce identical results."""
    async def async_add(x):
      return x + 10

    c = Chain(5).then(async_add)
    result1 = await c.run()
    result2 = await c.run()
    self.assertEqual(result1, 15)
    self.assertEqual(result2, 15)


if __name__ == '__main__':
  unittest.main()
