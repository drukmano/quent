"""Tests for StopIteration semantics across chain operations.

Covers PEP 479 compliance in generator-based iterate, StopIteration behavior
in foreach's while/next loop, StopIteration in filter predicates, and
StopAsyncIteration behavior in async paths.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, QuentException
from helpers import AsyncRange


# ---------------------------------------------------------------------------
# StopIteration in foreach (while True / next() loop)
# ---------------------------------------------------------------------------

class TestStopIterationInForeach(unittest.TestCase):

  def test_stop_iteration_in_fn_ends_early(self):
    """StopIteration raised by fn is caught by the `except StopIteration`
    in _foreach_op's while/next loop, ending iteration early.
    Only items processed before the raise are included.
    """
    def fn(x):
      if x == 3:
        raise StopIteration
      return x * 10

    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(result, [10, 20])

  def test_stop_iteration_on_first_item(self):
    """StopIteration on the very first item: nothing appended yet -> []."""
    def fn(x):
      raise StopIteration

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, [])

  def test_stop_iteration_on_last_item(self):
    """StopIteration on the last item: all previous items included."""
    def fn(x):
      if x == 3:
        raise StopIteration
      return x * 10

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, [10, 20])

  def test_stop_iteration_with_message(self):
    """StopIteration('msg') -- the value attribute is ignored; the
    accumulated list up to that point is returned."""
    def fn(x):
      if x == 2:
        raise StopIteration('should be ignored')
      return x * 10

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, [10])


# ---------------------------------------------------------------------------
# StopIteration in filter (while True / next() loop)
# ---------------------------------------------------------------------------

class TestStopIterationInFilter(unittest.TestCase):

  def test_stop_iteration_in_predicate(self):
    """StopIteration raised by filter predicate is caught by the
    `except StopIteration` in _filter_op's while/next loop."""
    def pred(x):
      if x == 3:
        raise StopIteration
      return x % 2 == 1

    result = Chain([1, 2, 3, 4, 5]).filter(pred).run()
    self.assertEqual(result, [1])

  def test_stop_iteration_on_first_item_filter(self):
    """StopIteration on the very first item of a filter: returns []."""
    def pred(x):
      raise StopIteration

    result = Chain([1, 2, 3]).filter(pred).run()
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# StopIteration in iterate (generator function -- PEP 479 applies)
# ---------------------------------------------------------------------------

class TestStopIterationInIterate(unittest.TestCase):

  def test_stop_iteration_in_sync_generator_fn(self):
    """PEP 479: StopIteration raised inside a generator function
    is converted to RuntimeError by the Python runtime.

    _sync_generator is a generator function. When fn raises StopIteration,
    it propagates through _sync_generator's frame, and Python converts it
    to RuntimeError per PEP 479.
    """
    def fn(x):
      raise StopIteration('from fn')

    g = Chain([1, 2, 3]).iterate(fn)
    with self.assertRaises(RuntimeError) as cm:
      list(g)
    self.assertIn('StopIteration', str(cm.exception))

  def test_stop_iteration_catches(self):
    """Verify that PEP 479 RuntimeError is actually raised, not silently swallowed."""
    def fn(x):
      if x == 2:
        raise StopIteration
      return x

    g = Chain([1, 2, 3]).iterate(fn)
    with self.assertRaises(RuntimeError):
      list(g)


# ---------------------------------------------------------------------------
# StopIteration in async iterate
# ---------------------------------------------------------------------------

class TestStopIterationInIterateAsync(IsolatedAsyncioTestCase):

  async def test_stop_iteration_in_async_fn(self):
    """StopIteration raised inside fn during async iteration.
    In _async_generator (an async generator), PEP 479 applies:
    StopIteration becomes RuntimeError.
    """
    def fn(x):
      if x == 2:
        raise StopIteration
      return x

    g = Chain([1, 2, 3]).iterate(fn)
    with self.assertRaises(RuntimeError):
      _ = [item async for item in g]


# ---------------------------------------------------------------------------
# StopAsyncIteration behavior
# ---------------------------------------------------------------------------

class TestStopAsyncIteration(IsolatedAsyncioTestCase):

  async def test_stop_async_iteration_behavior(self):
    """StopAsyncIteration from an async iterable terminates iteration normally."""

    class ShortAsyncIter:
      def __init__(self):
        self._i = 0

      def __aiter__(self):
        return self

      async def __anext__(self):
        if self._i >= 2:
          raise StopAsyncIteration
        val = self._i
        self._i += 1
        return val

    g = Chain(ShortAsyncIter()).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [0, 1])

  async def test_stop_async_iteration_empty(self):
    """StopAsyncIteration on first __anext__ yields empty."""

    class EmptyAsyncIter:
      def __aiter__(self):
        return self

      async def __anext__(self):
        raise StopAsyncIteration

    g = Chain(EmptyAsyncIter()).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Beyond-spec: additional StopIteration edge cases
# ---------------------------------------------------------------------------

class TestStopIterationInForeachDo(unittest.TestCase):

  def test_stop_iteration_in_foreach_do_ends_early(self):
    """StopIteration in foreach_do fn ends iteration early,
    returning only items processed before the raise."""
    def fn(x):
      if x == 2:
        raise StopIteration
      return x * 10  # discarded by foreach_do

    result = Chain([1, 2, 3]).foreach_do(fn).run()
    self.assertEqual(result, [1])

  def test_stop_iteration_in_foreach_do_first_item(self):
    """StopIteration on first item in foreach_do -> []."""
    def fn(x):
      raise StopIteration

    result = Chain([1, 2, 3]).foreach_do(fn).run()
    self.assertEqual(result, [])


class TestStopIterationAsyncForeach(IsolatedAsyncioTestCase):

  async def test_stop_iteration_in_async_foreach_fn(self):
    """In async foreach (_full_async), StopIteration from fn
    is NOT caught by except StopIteration (there's no while/next
    in the async path). It propagates as-is or becomes RuntimeError."""
    def fn(x):
      if x == 1:
        raise StopIteration
      return x

    # The async path uses `async for`, so StopIteration from fn
    # is not specially caught -- it propagates
    with self.assertRaises((StopIteration, RuntimeError)):
      await Chain(AsyncRange(3)).foreach(fn).run()


class TestStopIterationFilterAsync(IsolatedAsyncioTestCase):

  async def test_stop_iteration_in_filter_async_predicate(self):
    """StopIteration in async filter predicate propagates."""
    def pred(x):
      if x == 1:
        raise StopIteration
      return True

    with self.assertRaises((StopIteration, RuntimeError)):
      await Chain(AsyncRange(3)).filter(pred).run()


class TestStopIterationFromExhaustedInternalIterator(unittest.TestCase):

  def test_custom_iterator_exhaustion(self):
    """A custom iterator that raises StopIteration naturally
    terminates foreach normally."""

    class CountToTwo:
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

    result = Chain(CountToTwo()).foreach(lambda x: x * 10).run()
    self.assertEqual(result, [0, 10])


class TestStopIterationFromNestedNext(unittest.TestCase):

  def test_fn_calls_next_on_exhausted_iterator(self):
    """fn that calls next() on an exhausted iterator raises StopIteration,
    which is caught by _foreach_op's except StopIteration handler."""
    exhausted = iter([])

    def fn(x):
      next(exhausted)  # raises StopIteration
      return x

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, [])

  def test_fn_calls_next_midway(self):
    """fn calls next() on an iterator that exhausts mid-way."""
    inner_iter = iter([100, 200])

    def fn(x):
      return next(inner_iter)

    result = Chain([1, 2, 3]).foreach(fn).run()
    # fn(1)=100, fn(2)=200, fn(3) raises StopIteration -> caught
    self.assertEqual(result, [100, 200])


class TestStopIterationDoesNotAffectBreak(unittest.TestCase):

  def test_break_still_works_with_stop_iteration_awareness(self):
    """_Break is handled before StopIteration -- no interference."""
    result = Chain([1, 2, 3, 4]).foreach(
      lambda x: Chain.break_() if x == 3 else x
    ).run()
    self.assertEqual(result, [1, 2])


class TestPEP479InIterateWithNoLink(unittest.TestCase):

  def test_no_fn_iterate_natural_exhaustion(self):
    """iterate() without fn: the for-loop in _sync_generator exhausts
    the iterable naturally (no StopIteration leak)."""
    g = Chain([1, 2]).iterate()
    self.assertEqual(list(g), [1, 2])


class TestStopIterationMultipleOccurrences(unittest.TestCase):

  def test_stop_iteration_only_first_matters(self):
    """Only the first StopIteration matters -- subsequent items are never reached."""
    call_count = 0

    def fn(x):
      nonlocal call_count
      call_count += 1
      if x >= 2:
        raise StopIteration
      return x * 10

    result = Chain([1, 2, 3, 4, 5]).foreach(fn).run()
    self.assertEqual(result, [10])
    self.assertEqual(call_count, 2)  # fn called for 1 and 2 only


if __name__ == '__main__':
  unittest.main()
