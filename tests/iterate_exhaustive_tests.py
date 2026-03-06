"""Exhaustive tests for Chain.iterate() / Chain.iterate_do() and the _Generator object.

Covers _Generator construction, __call__, __iter__, __aiter__, __repr__,
_sync_generator, _async_generator, _aiter_wrap, break/return semantics,
exception temp-arg attachment, traceback modification, and edge-case
iteration targets (dict, bytes, string, empty, long, nested generators).
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from quent._ops import _Generator, _sync_generator, _async_generator, _aiter_wrap
from quent._core import _Break, _Return
from helpers import async_fn, async_identity, AsyncRange, AsyncEmpty


# ---------------------------------------------------------------------------
# _Generator object behavior
# ---------------------------------------------------------------------------

class TestGeneratorObject(unittest.TestCase):

  def test_call_creates_new_generator(self):
    g = Chain([1, 2, 3]).iterate()
    g2 = g(42)
    self.assertIsInstance(g2, _Generator)
    self.assertIsNot(g, g2)

  def test_call_preserves_fn(self):
    fn = lambda x: x * 2
    g = Chain([1, 2, 3]).iterate(fn)
    g2 = g(42)
    self.assertIs(g2._fn, fn)

  def test_call_sets_run_args(self):
    g = Chain().then(lambda x: [x]).iterate()
    g2 = g(42, 1, k=2)
    self.assertEqual(g2._run_args[0], 42)
    self.assertEqual(g2._run_args[1], (1,))
    self.assertEqual(g2._run_args[2], {'k': 2})

  def test_call_does_not_mutate_original(self):
    g = Chain().then(lambda x: [x]).iterate()
    g2 = g(99)
    self.assertEqual(g._run_args, (Null, (), {}))
    self.assertEqual(g2._run_args[0], 99)

  def test_default_run_args(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(g._run_args, (Null, (), {}))

  def test_repr(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(repr(g), '<Quent._Generator>')

  def test_has_iter_and_aiter(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertTrue(hasattr(g, '__iter__'))
    self.assertTrue(hasattr(g, '__aiter__'))

  def test_reusable_sync_iteration(self):
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(list(g), [1, 2, 3])
    self.assertEqual(list(g), [1, 2, 3])
    self.assertEqual(list(g), [1, 2, 3])

  def test_call_with_no_args(self):
    g = Chain([10, 20]).iterate()
    g2 = g()
    self.assertIsNot(g, g2)
    self.assertIsInstance(g2, _Generator)
    self.assertIs(g2._run_args[0], Null)
    self.assertEqual(g2._run_args[1], ())
    self.assertEqual(g2._run_args[2], {})

  def test_call_with_kwargs_only(self):
    g = Chain().then(lambda x=5: [x, x + 1]).iterate()
    g2 = g(k=42)
    self.assertIs(g2._run_args[0], Null)
    self.assertEqual(g2._run_args[1], ())
    self.assertEqual(g2._run_args[2], {'k': 42})


# ---------------------------------------------------------------------------
# Synchronous generator (_sync_generator)
# ---------------------------------------------------------------------------

class TestSyncGenerator(unittest.TestCase):

  def test_no_fn_yields_items(self):
    g = Chain([10, 20, 30]).iterate()
    self.assertEqual(list(g), [10, 20, 30])

  def test_fn_transforms_items(self):
    g = Chain([1, 2, 3]).iterate(lambda x: x ** 2)
    self.assertEqual(list(g), [1, 4, 9])

  def test_iterate_do_discards(self):
    tracker = []
    g = Chain([1, 2, 3]).iterate_do(lambda x: tracker.append(x * 10))
    result = list(g)
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [10, 20, 30])

  def test_iterate_do_no_fn(self):
    g = Chain([4, 5, 6]).iterate_do()
    self.assertEqual(list(g), [4, 5, 6])

  def test_break_stops(self):
    g = Chain([1, 2, 3, 4, 5]).iterate(
      lambda x: Chain.break_() if x == 3 else x * 10
    )
    self.assertEqual(list(g), [10, 20])

  def test_return_raises(self):
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.return_(x))
    with self.assertRaises(QuentException) as cm:
      list(g)
    self.assertIn('return_', str(cm.exception).lower())

  def test_exception_sets_temp_args(self):
    def bad_fn(x):
      if x == 2:
        raise ValueError('boom')
      return x

    g = Chain([1, 2, 3]).iterate(bad_fn)
    try:
      list(g)
    except ValueError as exc:
      # _set_link_temp_args attaches __quent_link_temp_args__ before
      # _modify_traceback consumes them and sets __quent__.
      self.assertTrue(
        getattr(exc, '__quent__', False)
        or hasattr(exc, '__quent_link_temp_args__')
      )
    else:
      self.fail('ValueError was not raised')

  def test_exception_modifies_traceback(self):
    def explode(x):
      raise RuntimeError('explode')

    g = Chain([1]).iterate(explode)
    try:
      list(g)
    except RuntimeError as exc:
      # _modify_traceback sets __quent__ on the exception
      self.assertTrue(getattr(exc, '__quent__', False))
    else:
      self.fail('RuntimeError was not raised')

  def test_empty_iterable(self):
    g = Chain([]).iterate(lambda x: x * 2)
    self.assertEqual(list(g), [])

  def test_string_yields_chars(self):
    g = Chain('hello').iterate()
    self.assertEqual(list(g), ['h', 'e', 'l', 'l', 'o'])


# ---------------------------------------------------------------------------
# Async generator (_async_generator, _aiter_wrap)
# ---------------------------------------------------------------------------

class TestAsyncGenerator(IsolatedAsyncioTestCase):

  async def test_sync_chain_output_wrapped(self):
    """When chain output is a sync iterable, _aiter_wrap wraps it."""
    g = Chain([1, 2, 3]).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [1, 2, 3])

  async def test_async_chain_output_direct(self):
    """When chain output is an async iterable, it's used directly."""
    g = Chain(AsyncRange(4)).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_chain_returns_coroutine(self):
    """When chain_run returns a coroutine, it's awaited then iterated."""
    async def make_list():
      return [10, 20, 30]

    g = Chain().then(make_list).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [10, 20, 30])

  async def test_fn_transforms_async(self):
    g = Chain([1, 2, 3]).iterate(lambda x: x + 100)
    result = [item async for item in g]
    self.assertEqual(result, [101, 102, 103])

  async def test_fn_awaitable_result(self):
    async def double(x):
      return x * 2

    g = Chain([5, 10]).iterate(double)
    result = [item async for item in g]
    self.assertEqual(result, [10, 20])

  async def test_iterate_do_async_discards(self):
    tracker = []

    async def track(x):
      tracker.append(x)
      return x * 100

    g = Chain([1, 2, 3]).iterate_do(track)
    result = [item async for item in g]
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [1, 2, 3])

  async def test_break_stops_async(self):
    g = Chain([1, 2, 3, 4]).iterate(
      lambda x: Chain.break_() if x == 3 else x
    )
    result = [item async for item in g]
    self.assertEqual(result, [1, 2])

  async def test_return_raises_async(self):
    g = Chain([1, 2]).iterate(lambda x: Chain.return_(x))
    with self.assertRaises(QuentException) as cm:
      _ = [item async for item in g]
    self.assertIn('return_', str(cm.exception).lower())

  async def test_exception_sets_temp_args_async(self):
    def bad_fn(x):
      if x == 2:
        raise ValueError('async boom')
      return x

    g = Chain([1, 2, 3]).iterate(bad_fn)
    try:
      _ = [item async for item in g]
    except ValueError as exc:
      self.assertTrue(
        getattr(exc, '__quent__', False)
        or hasattr(exc, '__quent_link_temp_args__')
      )
    else:
      self.fail('ValueError was not raised')

  async def test_empty_async_iterable(self):
    g = Chain(AsyncEmpty()).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Beyond-spec: additional edge cases
# ---------------------------------------------------------------------------

class TestGeneratorChainReturnTypes(unittest.TestCase):

  def test_generator_with_chain_returns_different_types(self):
    """_Generator with a chain that returns different types on each call."""
    call_count = 0

    def make_iterable():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return [1, 2, 3]
      elif call_count == 2:
        return (10, 20)
      else:
        return 'xy'

    g = Chain().then(make_iterable).iterate()
    self.assertEqual(list(g), [1, 2, 3])
    self.assertEqual(list(g), [10, 20])
    self.assertEqual(list(g), ['x', 'y'])


class TestIterateNestedGenerators(unittest.TestCase):

  def test_iterate_with_fn_returning_generator(self):
    """fn that returns a generator object (nested iteration).
    Each item is replaced by the generator itself (not expanded)."""
    def gen_fn(x):
      return list(range(x))

    g = Chain([2, 3]).iterate(gen_fn)
    result = list(g)
    self.assertEqual(result, [[0, 1], [0, 1, 2]])

  def test_iterate_do_with_side_effects(self):
    """iterate_do with fn that has side effects on an external structure."""
    accumulator = {}

    def record(x):
      accumulator[x] = x ** 2
      return 'discarded'

    g = Chain([3, 4, 5]).iterate_do(record)
    result = list(g)
    self.assertEqual(result, [3, 4, 5])
    self.assertEqual(accumulator, {3: 9, 4: 16, 5: 25})


class TestGeneratorAiterWithSyncChain(IsolatedAsyncioTestCase):

  async def test_aiter_with_sync_chain_returning_list(self):
    """__aiter__ with a sync chain that returns a list uses _aiter_wrap."""
    g = Chain([100, 200]).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [100, 200])


class TestMultipleIterationsOverSameGenerator(unittest.TestCase):

  def test_multiple_iterations_same_instance(self):
    """Multiple iterations over the same _Generator instance yield same results."""
    g = Chain([7, 8, 9]).iterate(lambda x: x - 1)
    r1 = list(g)
    r2 = list(g)
    r3 = list(g)
    self.assertEqual(r1, [6, 7, 8])
    self.assertEqual(r2, [6, 7, 8])
    self.assertEqual(r3, [6, 7, 8])


class TestGeneratorComplexArgs(unittest.TestCase):

  def test_call_with_complex_args_and_kwargs(self):
    """_Generator called with complex args and kwargs."""
    g = Chain().then(lambda x, y, z=0: [x + y + z]).iterate()
    g2 = g(10, 20, z=5)
    self.assertEqual(g2._run_args[0], 10)
    self.assertEqual(g2._run_args[1], (20,))
    self.assertEqual(g2._run_args[2], {'z': 5})


class TestAsyncIterateMixedFns(IsolatedAsyncioTestCase):

  async def test_async_iterate_fn_sometimes_sync_sometimes_async(self):
    """Async iterate where fn sometimes returns sync, sometimes async."""
    call_idx = 0

    def mixed_fn(x):
      nonlocal call_idx
      call_idx += 1
      if call_idx % 2 == 0:
        import asyncio

        async def _async():
          return x * 100

        return _async()
      return x * 10

    g = Chain([1, 2, 3, 4]).iterate(mixed_fn)
    result = [item async for item in g]
    self.assertEqual(result, [10, 200, 30, 400])


class TestIterateOverDict(unittest.TestCase):

  def test_iterate_over_dict_yields_keys(self):
    """Iterating over a dict yields its keys."""
    g = Chain({'a': 1, 'b': 2, 'c': 3}).iterate()
    result = list(g)
    self.assertEqual(sorted(result), ['a', 'b', 'c'])


class TestIterateOverBytes(unittest.TestCase):

  def test_iterate_over_bytes(self):
    """Iterating over bytes yields integer byte values."""
    g = Chain(b'\x01\x02\x03').iterate()
    result = list(g)
    self.assertEqual(result, [1, 2, 3])


class TestChainRaisesBeforeIterable(unittest.TestCase):

  def test_chain_raises_before_producing_iterable(self):
    """Chain raises before producing an iterable."""
    def boom():
      raise RuntimeError('chain broke')

    g = Chain().then(boom).iterate()
    with self.assertRaises(RuntimeError) as cm:
      list(g)
    self.assertEqual(str(cm.exception), 'chain broke')


class TestBreakAtIndexZero(unittest.TestCase):

  def test_break_at_index_zero(self):
    """Break at index 0 in iterate yields empty list."""
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.break_())
    self.assertEqual(list(g), [])


class TestVeryLongIteration(unittest.TestCase):

  def test_very_long_iteration_1000_items(self):
    """Iterate over 1000 items."""
    g = Chain(list(range(1000))).iterate(lambda x: x + 1)
    result = list(g)
    self.assertEqual(len(result), 1000)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[999], 1000)


class TestIterateWithFnCallingBreak(unittest.TestCase):

  def test_iterate_with_fn_that_calls_chain_break(self):
    """fn that calls Chain.break_() stops iteration."""
    def stopping_fn(x):
      if x > 3:
        Chain.break_()
      return x * 10

    g = Chain([1, 2, 3, 4, 5]).iterate(stopping_fn)
    self.assertEqual(list(g), [10, 20, 30])


class TestAsyncBreakAtIndexZero(IsolatedAsyncioTestCase):

  async def test_break_at_index_zero_async(self):
    """Break at index 0 in async iterate yields empty list."""
    g = Chain([1, 2, 3]).iterate(lambda x: Chain.break_())
    result = [item async for item in g]
    self.assertEqual(result, [])


class TestAsyncVeryLongIteration(IsolatedAsyncioTestCase):

  async def test_async_very_long_iteration(self):
    """Async iterate over 500 items."""
    g = Chain(list(range(500))).iterate(lambda x: x * 2)
    result = [item async for item in g]
    self.assertEqual(len(result), 500)
    self.assertEqual(result[0], 0)
    self.assertEqual(result[499], 998)


class TestIterateDoPreservesItems(IsolatedAsyncioTestCase):

  async def test_iterate_do_async_preserves_items(self):
    """iterate_do with no fn in async mode yields original items."""
    g = Chain([10, 20, 30]).iterate_do()
    result = [item async for item in g]
    self.assertEqual(result, [10, 20, 30])


class TestIterateWithAsyncIterable(IsolatedAsyncioTestCase):

  async def test_iterate_async_range_with_fn(self):
    """Iterate over AsyncRange with a sync fn."""
    g = Chain(AsyncRange(5)).iterate(lambda x: x * 3)
    result = [item async for item in g]
    self.assertEqual(result, [0, 3, 6, 9, 12])

  async def test_iterate_do_async_range(self):
    """iterate_do over AsyncRange."""
    tracker = []
    g = Chain(AsyncRange(3)).iterate_do(lambda x: tracker.append(x))
    result = [item async for item in g]
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(tracker, [0, 1, 2])


class TestChainRaisesBeforeIterableAsync(IsolatedAsyncioTestCase):

  async def test_chain_raises_before_producing_iterable_async(self):
    """Chain raises before producing an iterable (async path)."""
    async def boom():
      raise RuntimeError('async chain broke')

    g = Chain().then(boom).iterate()
    with self.assertRaises(RuntimeError) as cm:
      _ = [item async for item in g]
    self.assertEqual(str(cm.exception), 'async chain broke')


class TestIterateOverTuple(unittest.TestCase):

  def test_iterate_over_tuple(self):
    g = Chain((10, 20, 30)).iterate()
    self.assertEqual(list(g), [10, 20, 30])


class TestIterateOverSet(unittest.TestCase):

  def test_iterate_over_set(self):
    g = Chain({1, 2, 3}).iterate()
    self.assertEqual(sorted(list(g)), [1, 2, 3])


class TestIterateRunArgsInjection(unittest.TestCase):

  def test_run_args_injection_with_fn(self):
    """_Generator.__call__ passes run_args through to chain execution."""
    g = Chain().then(lambda start: range(start, start + 3)).iterate(lambda x: x * 2)
    result = list(g(5))
    self.assertEqual(result, [10, 12, 14])

  def test_run_args_injection_no_fn(self):
    g = Chain().then(lambda n: list(range(n))).iterate()
    result = list(g(4))
    self.assertEqual(result, [0, 1, 2, 3])


if __name__ == '__main__':
  unittest.main()
