"""Tests for Chain.iterate() and Chain.iterate_do(): sync iteration,
async iteration, _Generator object behavior, break/return semantics,
and run-args injection.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import async_fn, async_identity, AsyncRange


# ---------------------------------------------------------------------------
# Synchronous iterate (yields fn results or raw items)
# ---------------------------------------------------------------------------

class TestIterateSync(unittest.TestCase):

  def test_sync_list(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(list(g), [1, 2, 3])

  def test_with_fn(self):
    g = Chain([1, 2, 3]).iterate(lambda x: x * 2)
    self.assertEqual(list(g), [2, 4, 6])

  def test_do_fn_discarded(self):
    g = Chain([1, 2, 3]).iterate_do(lambda x: x * 100)
    self.assertEqual(list(g), [1, 2, 3])

  def test_no_fn_yields_unchanged(self):
    g = Chain([10, 20, 30]).iterate()
    self.assertEqual(list(g), [10, 20, 30])

  def test_empty_iterable(self):
    g = Chain([]).iterate()
    self.assertEqual(list(g), [])

  def test_iterate_range(self):
    g = Chain(range(4)).iterate()
    self.assertEqual(list(g), [0, 1, 2, 3])

  def test_iterate_with_fn_and_range(self):
    g = Chain(range(5)).iterate(lambda x: x ** 2)
    self.assertEqual(list(g), [0, 1, 4, 9, 16])

  def test_iterate_do_fn_side_effect(self):
    tracker = []
    g = Chain([1, 2, 3]).iterate_do(lambda x: tracker.append(x))
    result = list(g)
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [1, 2, 3])

  def test_iterate_string_yields_chars(self):
    g = Chain('abc').iterate()
    self.assertEqual(list(g), ['a', 'b', 'c'])


# ---------------------------------------------------------------------------
# Async iterate
# ---------------------------------------------------------------------------

class TestIterateAsync(IsolatedAsyncioTestCase):

  async def test_async_iterate(self):
    g = Chain([1, 2, 3]).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [1, 2, 3])

  async def test_async_with_fn(self):
    g = Chain([1, 2, 3]).iterate(lambda x: x * 2)
    result = [item async for item in g]
    self.assertEqual(result, [2, 4, 6])

  async def test_async_fn_in_iterate(self):
    async def double(x):
      return x * 2
    g = Chain([1, 2, 3]).iterate(double)
    result = [item async for item in g]
    self.assertEqual(result, [2, 4, 6])

  async def test_do_async(self):
    tracker = []
    async def track(x):
      tracker.append(x)
      return x * 100
    g = Chain([1, 2, 3]).iterate_do(track)
    result = [item async for item in g]
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [1, 2, 3])

  async def test_async_chain_iter(self):
    async def make_list():
      return [10, 20, 30]
    g = Chain().then(make_list).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [10, 20, 30])

  async def test_async_iterate_empty(self):
    g = Chain([]).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [])

  async def test_async_iterate_do_no_fn(self):
    g = Chain([4, 5, 6]).iterate_do()
    result = [item async for item in g]
    self.assertEqual(result, [4, 5, 6])

  async def test_async_fn_identity(self):
    g = Chain([1, 2, 3]).iterate(async_identity)
    result = [item async for item in g]
    self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# _Generator object behavior
# ---------------------------------------------------------------------------

class TestGeneratorObject(unittest.TestCase):

  def test_call_returns_new_generator(self):
    g = Chain([1, 2, 3]).iterate()
    g2 = g(42)
    self.assertIsNot(g, g2)

  def test_call_sets_run_args(self):
    g = Chain().then(lambda x: [x, x + 1, x + 2]).iterate()
    g2 = g(10)
    self.assertEqual(g2._run_args[0], 10)

  def test_reusable_multiple_iterations(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(list(g), [1, 2, 3])
    self.assertEqual(list(g), [1, 2, 3])

  def test_repr(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(repr(g), '<Quent._Generator>')

  def test_call_does_not_mutate_original(self):
    g = Chain().then(lambda x: [x]).iterate()
    g2 = g(99)
    self.assertEqual(g._run_args, (Null, (), {}))
    self.assertEqual(g2._run_args[0], 99)

  def test_default_run_args(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(g._run_args, (Null, (), {}))


# ---------------------------------------------------------------------------
# Break semantics (sync)
# ---------------------------------------------------------------------------

class TestIterateBreak(unittest.TestCase):

  def test_break_stops_sync(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.break_() if x == 2 else x)
    self.assertEqual(list(g), [1])

  def test_break_on_first_element(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.break_())
    self.assertEqual(list(g), [])

  def test_break_on_last_element(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.break_() if x == 3 else x)
    self.assertEqual(list(g), [1, 2])

  def test_return_raises_quent_exception(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.return_(x))
    with self.assertRaises(QuentException) as cm:
      list(g)
    self.assertIn('return_', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# Break semantics (async)
# ---------------------------------------------------------------------------

class TestIterateBreakAsync(IsolatedAsyncioTestCase):

  async def test_break_stops_async(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.break_() if x == 2 else x)
    result = [item async for item in g]
    self.assertEqual(result, [1])

  async def test_return_raises_async(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.return_(x))
    with self.assertRaises(QuentException) as cm:
      _ = [item async for item in g]
    self.assertIn('return_', str(cm.exception).lower())

  async def test_break_with_async_fn(self):
    async def breaker(x):
      if x == 2:
        Chain.break_()
      return x * 10

    g = Chain([1, 2, 3]).iterate(breaker)
    result = [item async for item in g]
    self.assertEqual(result, [10])

  async def test_break_on_first_async(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.break_())
    result = [item async for item in g]
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Run-args injection
# ---------------------------------------------------------------------------

class TestIterateWithRunArgs(unittest.TestCase):

  def test_iterate_with_value_injection(self):
    g = Chain().then(lambda x: [x, x + 1, x + 2]).iterate()
    self.assertEqual(list(g(10)), [10, 11, 12])

  def test_iterate_with_fn_and_value_injection(self):
    g = Chain().then(lambda x: [x, x + 1, x + 2]).iterate(lambda v: v * 2)
    self.assertEqual(list(g(10)), [20, 22, 24])

  def test_iterate_do_with_value_injection(self):
    tracker = []
    g = Chain().then(lambda x: [x, x + 1]).iterate_do(lambda v: tracker.append(v))
    result = list(g(5))
    self.assertEqual(result, [5, 6])
    self.assertEqual(tracker, [5, 6])

  def test_different_run_args_produce_different_results(self):
    g = Chain().then(lambda x: [x, x * 2]).iterate()
    self.assertEqual(list(g(3)), [3, 6])
    self.assertEqual(list(g(7)), [7, 14])


if __name__ == '__main__':
  unittest.main()
