"""Full cross-product tests for iterable type handling across chain operations.

Tests every iterable type (Axis H) against map, foreach, filter,
iterate, iterate_do with sync and async fn types. Covers empty iterables,
single-element iterables, raising iterables, and async iterables.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain
from helpers import (
  AsyncEmpty,
  AsyncInfiniteIterable,
  AsyncRange,
  AsyncRangeRaises,
  InfiniteIterable,
  RaisingIterable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _double(x):
  return x * 2


async def _async_double(x):
  return x * 2


def _is_even(x):
  return x % 2 == 0


async def _async_is_even(x):
  return x % 2 == 0


def _identity(x):
  return x


async def _async_identity(x):
  return x


def _is_truthy(x):
  return bool(x)


async def _async_is_truthy(x):
  return bool(x)


# ---------------------------------------------------------------------------
# Iterable factories: each returns (iterable, expected_items_as_list)
# ---------------------------------------------------------------------------

def _make_iterables():
  """Return dict of {label: (iterable, expected_list)} for standard cases."""
  return {
    'list': ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
    'tuple': ((0, 1, 2, 3, 4), [0, 1, 2, 3, 4]),
    'range': (range(5), [0, 1, 2, 3, 4]),
    'generator_iter': (iter([0, 1, 2, 3, 4]), [0, 1, 2, 3, 4]),
    'set_single': ({42}, [42]),
    'str': ('abcde', ['a', 'b', 'c', 'd', 'e']),
    'dict': ({'a': 1, 'b': 2}, ['a', 'b']),
    'bytes': (b'\x00\x01\x02', [0, 1, 2]),
    'generator_expr': ((x for x in range(5)), [0, 1, 2, 3, 4]),
    'dict_items': ({'a': 1, 'b': 2}.items(), [('a', 1), ('b', 2)]),
    'dict_values': ({'a': 10, 'b': 20}.values(), [10, 20]),
    'frozenset_single': (frozenset({99}), [99]),
  }


def _make_empty_iterables():
  """Return dict of {label: empty_iterable} for empty cases."""
  return {
    'list': [],
    'tuple': (),
    'range': range(0),
    'generator_iter': iter([]),
    'set': set(),
    'str': '',
    'dict': {},
    'bytes': b'',
    'generator_expr': (x for x in []),
    'dict_items': {}.items(),
    'dict_values': {}.values(),
    'frozenset': frozenset(),
  }


def _make_single_iterables():
  """Return dict of {label: (iterable, expected_single_item)} for single-element."""
  return {
    'list': ([7], 7),
    'tuple': ((7,), 7),
    'range': (range(7, 8), 7),
    'generator_iter': (iter([7]), 7),
    'set': ({7}, 7),
    'str': ('z', 'z'),
    'dict': ({'k': 'v'}, 'k'),
    'bytes': (b'\x07', 7),
    'generator_expr': ((x for x in [7]), 7),
    'frozenset': (frozenset({7}), 7),
  }


# ---------------------------------------------------------------------------
# Class 1: TestIterableTypeMapMatrix
# ---------------------------------------------------------------------------

class TestIterableTypeMapMatrix(unittest.TestCase):
  """Full cross-product: iterable types x map/foreach x sync fn."""

  def test_map_sync_fn(self):
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label, fn='sync', is_do=False):
        result = Chain(iterable).map(_double).run()
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(len(result), len(expected))
          for item in result:
            self.assertIn(item // 2, [e for e in expected])
        elif label == 'dict_items':
          self.assertEqual(len(result), len(expected))
        elif label == 'str':
          self.assertEqual(result, ['aa', 'bb', 'cc', 'dd', 'ee'])
        elif label == 'dict':
          self.assertEqual(result, ['aa', 'bb'])
        elif label == 'bytes':
          self.assertEqual(result, [0, 2, 4])
        else:
          self.assertEqual(result, [x * 2 for x in expected])

  def test_foreach_sync_fn(self):
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label, fn='sync', is_do=True):
        result = Chain(iterable).foreach(_double).run()
        # foreach keeps original items
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(len(result), len(expected))
        elif label == 'dict_items':
          self.assertEqual(len(result), len(expected))
        else:
          self.assertEqual(result, expected)

  def test_map_ordering_preserved(self):
    """Ordered iterables preserve element order."""
    for label in ('list', 'tuple', 'range', 'generator_iter', 'str', 'bytes'):
      iterables = _make_iterables()
      if label not in iterables:
        continue
      iterable, expected = iterables[label]
      with self.subTest(iterable_type=label):
        result = Chain(iterable).map(_identity).run()
        self.assertEqual(result, expected)

  def test_map_identity_fn(self):
    """map with identity returns items unchanged."""
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label):
        result = Chain(iterable).map(_identity).run()
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(set(result), set(expected))
        else:
          self.assertEqual(result, expected)


# ---------------------------------------------------------------------------
# Class 2: TestIterableTypeFilterMatrix
# ---------------------------------------------------------------------------

class TestIterableTypeFilterMatrix(unittest.TestCase):
  """Full cross-product: iterable types x filter x sync/async predicate."""

  def test_filter_numeric_even(self):
    """Filter numeric iterables for even values."""
    cases = {
      'list': ([0, 1, 2, 3, 4], [0, 2, 4]),
      'tuple': ((0, 1, 2, 3, 4), [0, 2, 4]),
      'range': (range(5), [0, 2, 4]),
      'generator_iter': (iter([0, 1, 2, 3, 4]), [0, 2, 4]),
      'bytes': (b'\x00\x01\x02\x03\x04', [0, 2, 4]),
    }
    for label, (iterable, expected) in cases.items():
      with self.subTest(iterable_type=label, predicate='is_even'):
        result = Chain(iterable).filter(_is_even).run()
        self.assertEqual(result, expected)

  def test_filter_truthy(self):
    """Filter for truthy values."""
    cases = {
      'list': ([0, 1, '', 'a', None, True, False], [1, 'a', True]),
      'tuple': ((0, 1, 2), [1, 2]),
      'str': ('abc', ['a', 'b', 'c']),  # all chars are truthy
      'dict': ({'a': 1, '': 2}, ['a']),  # filter on keys, '' is falsy
    }
    for label, (iterable, expected) in cases.items():
      with self.subTest(iterable_type=label, predicate='truthy'):
        result = Chain(iterable).filter(_is_truthy).run()
        self.assertEqual(result, expected)

  def test_filter_all_pass(self):
    """When all items pass, result equals input items."""
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label):
        result = Chain(iterable).filter(lambda x: True).run()
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(set(result), set(expected))
        else:
          self.assertEqual(result, expected)

  def test_filter_none_pass(self):
    """When no items pass, result is empty list."""
    for label, (iterable, _) in _make_iterables().items():
      with self.subTest(iterable_type=label):
        result = Chain(iterable).filter(lambda x: False).run()
        self.assertEqual(result, [])

  def test_filter_dict_items(self):
    """Filter dict.items() tuples."""
    items = {'a': 1, 'b': 2, 'c': 3}.items()
    result = Chain(items).filter(lambda kv: kv[1] > 1).run()
    self.assertEqual(sorted(result, key=lambda x: x[0]), [('b', 2), ('c', 3)])

  def test_filter_dict_values(self):
    """Filter dict.values()."""
    vals = {'a': 1, 'b': 2, 'c': 3}.values()
    result = Chain(vals).filter(lambda v: v > 1).run()
    self.assertEqual(sorted(result), [2, 3])

  def test_filter_single_element_pass(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label):
        result = Chain(iterable).filter(lambda x: True).run()
        self.assertEqual(result, [item])

  def test_filter_single_element_fail(self):
    for label, (iterable, _) in _make_single_iterables().items():
      with self.subTest(iterable_type=label):
        result = Chain(iterable).filter(lambda x: False).run()
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Class 3: TestIterableTypeIterateMatrix
# ---------------------------------------------------------------------------

class TestIterableTypeIterateMatrix(unittest.TestCase):
  """Full cross-product: iterable types x iterate/iterate_do x sync fn."""

  def test_iterate_no_fn(self):
    """iterate() with no fn yields items unchanged."""
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label, fn=None, is_do=False):
        g = Chain(iterable).iterate()
        result = list(g)
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(set(result), set(expected))
        else:
          self.assertEqual(result, expected)

  def test_iterate_with_fn(self):
    """iterate() with fn yields fn(item) for each item."""
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label, fn='double', is_do=False):
        g = Chain(iterable).iterate(_double)
        result = list(g)
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(len(result), len(expected))
        elif label == 'str':
          self.assertEqual(result, [c * 2 for c in expected])
        elif label == 'dict':
          self.assertEqual(result, [k * 2 for k in expected])
        elif label == 'dict_items':
          self.assertEqual(len(result), len(expected))
        elif label == 'bytes':
          self.assertEqual(result, [x * 2 for x in expected])
        else:
          self.assertEqual(result, [x * 2 for x in expected])

  def test_iterate_do_with_fn(self):
    """iterate_do() yields original items, discarding fn result."""
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label, fn='double', is_do=True):
        g = Chain(iterable).iterate_do(_double)
        result = list(g)
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(set(result), set(expected))
        else:
          self.assertEqual(result, expected)

  def test_iterate_do_no_fn(self):
    """iterate_do() with no fn yields items unchanged."""
    for label, (iterable, expected) in _make_iterables().items():
      with self.subTest(iterable_type=label, fn=None, is_do=True):
        g = Chain(iterable).iterate_do()
        result = list(g)
        if label in ('set_single', 'frozenset_single'):
          self.assertEqual(set(result), set(expected))
        else:
          self.assertEqual(result, expected)

  def test_iterate_preserves_order(self):
    """Ordered iterables preserve order through iterate."""
    ordered = {
      'list': [10, 20, 30],
      'tuple': (10, 20, 30),
      'range': range(3),
      'str': 'abc',
    }
    for label, iterable in ordered.items():
      with self.subTest(iterable_type=label):
        result = list(Chain(iterable).iterate())
        self.assertEqual(result, list(iterable))


# ---------------------------------------------------------------------------
# Class 4: TestAsyncIterableMatrix
# ---------------------------------------------------------------------------

class TestAsyncIterableMatrix(IsolatedAsyncioTestCase):
  """AsyncRange, AsyncEmpty, AsyncRangeRaises, AsyncInfiniteIterable
  x map/foreach/filter/iterate/iterate_do
  x sync fn/async fn."""

  # --- map ---

  async def test_async_range_map_sync(self):
    result = await Chain(AsyncRange(5)).map(_double).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  async def test_async_range_map_async(self):
    result = await Chain(AsyncRange(5)).map(_async_double).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  async def test_async_range_foreach_sync(self):
    result = await Chain(AsyncRange(5)).foreach(_double).run()
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_async_range_foreach_async(self):
    result = await Chain(AsyncRange(5)).foreach(_async_double).run()
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_async_empty_map_sync(self):
    result = await Chain(AsyncEmpty()).map(_double).run()
    self.assertEqual(result, [])

  async def test_async_empty_foreach_sync(self):
    result = await Chain(AsyncEmpty()).foreach(_double).run()
    self.assertEqual(result, [])

  async def test_async_empty_map_async(self):
    result = await Chain(AsyncEmpty()).map(_async_double).run()
    self.assertEqual(result, [])

  # --- filter ---

  async def test_async_range_filter_sync(self):
    result = await Chain(AsyncRange(6)).filter(_is_even).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_async_range_filter_async(self):
    result = await Chain(AsyncRange(6)).filter(_async_is_even).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_async_empty_filter_sync(self):
    result = await Chain(AsyncEmpty()).filter(_is_even).run()
    self.assertEqual(result, [])

  async def test_async_empty_filter_async(self):
    result = await Chain(AsyncEmpty()).filter(_async_is_even).run()
    self.assertEqual(result, [])

  # --- iterate ---

  async def test_async_range_iterate_no_fn(self):
    g = Chain(AsyncRange(4)).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_async_range_iterate_sync_fn(self):
    g = Chain(AsyncRange(4)).iterate(_double)
    result = [item async for item in g]
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_async_range_iterate_async_fn(self):
    g = Chain(AsyncRange(4)).iterate(_async_double)
    result = [item async for item in g]
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_async_range_iterate_do_sync_fn(self):
    g = Chain(AsyncRange(4)).iterate_do(_double)
    result = [item async for item in g]
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_async_range_iterate_do_async_fn(self):
    g = Chain(AsyncRange(4)).iterate_do(_async_double)
    result = [item async for item in g]
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_async_empty_iterate(self):
    g = Chain(AsyncEmpty()).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [])

  async def test_async_empty_iterate_do(self):
    g = Chain(AsyncEmpty()).iterate_do(_double)
    result = [item async for item in g]
    self.assertEqual(result, [])

  # --- AsyncRangeRaises ---

  async def test_async_range_raises_map_sync(self):
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(AsyncRangeRaises(5, 3)).map(_identity).run()
    self.assertIn('iteration error', str(ctx.exception))

  async def test_async_range_raises_map_async(self):
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(AsyncRangeRaises(5, 3)).map(_async_identity).run()
    self.assertIn('iteration error', str(ctx.exception))

  async def test_async_range_raises_filter_sync(self):
    with self.assertRaises(RuntimeError):
      await Chain(AsyncRangeRaises(5, 2)).filter(_is_even).run()

  async def test_async_range_raises_filter_async(self):
    with self.assertRaises(RuntimeError):
      await Chain(AsyncRangeRaises(5, 2)).filter(_async_is_even).run()

  async def test_async_range_raises_iterate(self):
    g = Chain(AsyncRangeRaises(5, 3)).iterate()
    collected = []
    with self.assertRaises(RuntimeError):
      async for item in g:
        collected.append(item)
    self.assertEqual(collected, [0, 1, 2])

  # --- AsyncInfiniteIterable with break ---

  async def test_async_infinite_map_break(self):
    count = 0
    def fn(x):
      nonlocal count
      count += 1
      if x >= 5:
        Chain.break_()
      return x
    result = await Chain(AsyncInfiniteIterable()).map(fn).run()
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_async_infinite_foreach_break(self):
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    result = await Chain(AsyncInfiniteIterable()).foreach(fn).run()
    self.assertEqual(result, [0, 1, 2])

  # --- Async map/filter with sync iterables (should work via _aiter_wrap) ---

  async def test_sync_iterable_map_async_fn(self):
    """Sync iterable + async fn triggers _to_async mid-iteration."""
    result = await Chain([0, 1, 2, 3]).map(_async_double).run()
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_sync_iterable_foreach_async_fn(self):
    result = await Chain([0, 1, 2, 3]).foreach(_async_double).run()
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_sync_iterable_filter_async_fn(self):
    result = await Chain([0, 1, 2, 3, 4]).filter(_async_is_even).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_sync_iterable_iterate_async_fn(self):
    g = Chain([0, 1, 2]).iterate(_async_double)
    result = [item async for item in g]
    self.assertEqual(result, [0, 2, 4])


# ---------------------------------------------------------------------------
# Class 5: TestEmptyIterableMatrix
# ---------------------------------------------------------------------------

class TestEmptyIterableMatrix(IsolatedAsyncioTestCase):
  """Empty versions of each iterable type x all operations."""

  def test_map_empty_sync(self):
    for label, iterable in _make_empty_iterables().items():
      with self.subTest(iterable_type=label, op='map'):
        result = Chain(iterable).map(_identity).run()
        self.assertEqual(result, [])

  def test_foreach_empty_sync(self):
    for label, iterable in _make_empty_iterables().items():
      with self.subTest(iterable_type=label, op='foreach'):
        result = Chain(iterable).foreach(_identity).run()
        self.assertEqual(result, [])

  def test_filter_empty_sync(self):
    for label, iterable in _make_empty_iterables().items():
      with self.subTest(iterable_type=label, op='filter'):
        result = Chain(iterable).filter(lambda x: True).run()
        self.assertEqual(result, [])

  def test_iterate_empty_sync(self):
    for label, iterable in _make_empty_iterables().items():
      with self.subTest(iterable_type=label, op='iterate'):
        result = list(Chain(iterable).iterate())
        self.assertEqual(result, [])

  def test_iterate_do_empty_sync(self):
    for label, iterable in _make_empty_iterables().items():
      with self.subTest(iterable_type=label, op='iterate_do'):
        result = list(Chain(iterable).iterate_do(_identity))
        self.assertEqual(result, [])

  async def test_async_empty_map(self):
    result = await Chain(AsyncEmpty()).map(_identity).run()
    self.assertEqual(result, [])

  async def test_async_empty_foreach(self):
    result = await Chain(AsyncEmpty()).foreach(_identity).run()
    self.assertEqual(result, [])

  async def test_async_empty_filter(self):
    result = await Chain(AsyncEmpty()).filter(lambda x: True).run()
    self.assertEqual(result, [])

  async def test_async_empty_iterate(self):
    g = Chain(AsyncEmpty()).iterate()
    result = [item async for item in g]
    self.assertEqual(result, [])

  async def test_async_empty_iterate_with_fn(self):
    g = Chain(AsyncEmpty()).iterate(_double)
    result = [item async for item in g]
    self.assertEqual(result, [])

  async def test_async_empty_iterate_do(self):
    g = Chain(AsyncEmpty()).iterate_do(_double)
    result = [item async for item in g]
    self.assertEqual(result, [])

  def test_empty_consumed_generator(self):
    """A consumed generator yields nothing."""
    gen = iter([1, 2, 3])
    list(gen)  # consume
    for op_label, op in (
      ('map', lambda it: Chain(it).map(_identity).run()),
      ('foreach', lambda it: Chain(it).foreach(_identity).run()),
      ('filter', lambda it: Chain(it).filter(lambda x: True).run()),
      ('iterate', lambda it: list(Chain(it).iterate())),
    ):
      with self.subTest(op=op_label):
        consumed = iter([])  # already empty
        result = op(consumed)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Class 6: TestSingleElementMatrix
# ---------------------------------------------------------------------------

class TestSingleElementMatrix(unittest.TestCase):
  """Single-element versions of each iterable type x all operations."""

  def test_map_single(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='map'):
        result = Chain(iterable).map(_identity).run()
        self.assertEqual(result, [item])

  def test_foreach_single(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='foreach'):
        result = Chain(iterable).foreach(_double).run()
        self.assertEqual(result, [item])

  def test_filter_single_passes(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='filter_pass'):
        result = Chain(iterable).filter(lambda x: True).run()
        self.assertEqual(result, [item])

  def test_filter_single_fails(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='filter_fail'):
        result = Chain(iterable).filter(lambda x: False).run()
        self.assertEqual(result, [])

  def test_iterate_single(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='iterate'):
        result = list(Chain(iterable).iterate())
        self.assertEqual(result, [item])

  def test_iterate_do_single(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='iterate_do'):
        result = list(Chain(iterable).iterate_do(_double))
        self.assertEqual(result, [item])

  def test_map_with_fn_single(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='map_fn'):
        result = Chain(iterable).map(_double).run()
        self.assertEqual(result, [item * 2])

  def test_iterate_with_fn_single(self):
    for label, (iterable, item) in _make_single_iterables().items():
      with self.subTest(iterable_type=label, op='iterate_fn'):
        result = list(Chain(iterable).iterate(_double))
        self.assertEqual(result, [item * 2])


# ---------------------------------------------------------------------------
# Class 7: TestIterableWithExceptionMatrix
# ---------------------------------------------------------------------------

class TestIterableWithExceptionMatrix(IsolatedAsyncioTestCase):
  """RaisingIterable(n, raise_at) with different raise_at positions x operations."""

  def test_map_raising_at_start(self):
    with self.assertRaises(RuntimeError) as ctx:
      Chain(RaisingIterable(5, 0)).map(_identity).run()
    self.assertIn('iteration error', str(ctx.exception))

  def test_map_raising_at_middle(self):
    with self.assertRaises(RuntimeError):
      Chain(RaisingIterable(5, 2)).map(_identity).run()

  def test_map_raising_at_end(self):
    with self.assertRaises(RuntimeError):
      Chain(RaisingIterable(5, 4)).map(_identity).run()

  def test_foreach_raising(self):
    for raise_at in (0, 2, 4):
      with self.subTest(raise_at=raise_at):
        with self.assertRaises(RuntimeError):
          Chain(RaisingIterable(5, raise_at)).foreach(_identity).run()

  def test_filter_raising(self):
    for raise_at in (0, 2, 4):
      with self.subTest(raise_at=raise_at):
        with self.assertRaises(RuntimeError):
          Chain(RaisingIterable(5, raise_at)).filter(lambda x: True).run()

  def test_iterate_raising(self):
    """iterate stops and propagates the exception."""
    for raise_at in (0, 2, 4):
      with self.subTest(raise_at=raise_at):
        g = Chain(RaisingIterable(5, raise_at)).iterate()
        collected = []
        with self.assertRaises(RuntimeError):
          for item in g:
            collected.append(item)
        self.assertEqual(collected, list(range(raise_at)))

  def test_iterate_do_raising(self):
    for raise_at in (0, 2, 4):
      with self.subTest(raise_at=raise_at):
        g = Chain(RaisingIterable(5, raise_at)).iterate_do(_identity)
        collected = []
        with self.assertRaises(RuntimeError):
          for item in g:
            collected.append(item)
        self.assertEqual(collected, list(range(raise_at)))

  async def test_async_range_raises_map_at_positions(self):
    for raise_at in (0, 2, 4):
      with self.subTest(raise_at=raise_at):
        with self.assertRaises(RuntimeError):
          await Chain(AsyncRangeRaises(5, raise_at)).map(_identity).run()

  async def test_async_range_raises_filter(self):
    for raise_at in (0, 2, 4):
      with self.subTest(raise_at=raise_at):
        with self.assertRaises(RuntimeError):
          await Chain(AsyncRangeRaises(5, raise_at)).filter(lambda x: True).run()

  async def test_async_range_raises_iterate(self):
    for raise_at in (0, 2, 4):
      with self.subTest(raise_at=raise_at):
        g = Chain(AsyncRangeRaises(5, raise_at)).iterate()
        collected = []
        with self.assertRaises(RuntimeError):
          async for item in g:
            collected.append(item)
        self.assertEqual(collected, list(range(raise_at)))

  def test_fn_raises_in_map(self):
    """When the fn itself raises (not the iterable)."""
    def bad_fn(x):
      if x == 2:
        raise ValueError('fn error')
      return x

    with self.assertRaises(ValueError) as ctx:
      Chain([0, 1, 2, 3]).map(bad_fn).run()
    self.assertIn('fn error', str(ctx.exception))

  def test_fn_raises_in_filter(self):
    def bad_pred(x):
      if x == 2:
        raise ValueError('pred error')
      return True

    with self.assertRaises(ValueError) as ctx:
      Chain([0, 1, 2, 3]).filter(bad_pred).run()
    self.assertIn('pred error', str(ctx.exception))

  def test_fn_raises_in_iterate(self):
    def bad_fn(x):
      if x == 2:
        raise ValueError('fn error')
      return x

    g = Chain([0, 1, 2, 3]).iterate(bad_fn)
    collected = []
    with self.assertRaises(ValueError):
      for item in g:
        collected.append(item)
    self.assertEqual(collected, [0, 1])

  async def test_async_fn_raises_in_map(self):
    async def bad_fn(x):
      if x == 2:
        raise ValueError('async fn error')
      return x

    with self.assertRaises(ValueError):
      await Chain(AsyncRange(5)).map(bad_fn).run()

  async def test_async_fn_raises_in_filter(self):
    async def bad_pred(x):
      if x == 2:
        raise ValueError('async pred error')
      return True

    with self.assertRaises(ValueError):
      await Chain(AsyncRange(5)).filter(bad_pred).run()

  async def test_async_fn_raises_in_iterate(self):
    async def bad_fn(x):
      if x == 2:
        raise ValueError('async fn error')
      return x

    g = Chain(AsyncRange(5)).iterate(bad_fn)
    collected = []
    with self.assertRaises(ValueError):
      async for item in g:
        collected.append(item)
    self.assertEqual(collected, [0, 1])


# ---------------------------------------------------------------------------
# Class 8: TestConsumedGeneratorMatrix
# ---------------------------------------------------------------------------

class TestConsumedGeneratorMatrix(unittest.TestCase):
  """Consumed generators yield nothing for all operations."""

  def _make_consumed(self):
    gen = iter([1, 2, 3])
    list(gen)  # consume it
    return gen

  def test_consumed_generator_map(self):
    result = Chain(self._make_consumed()).map(_identity).run()
    self.assertEqual(result, [])

  def test_consumed_generator_foreach(self):
    result = Chain(self._make_consumed()).foreach(_identity).run()
    self.assertEqual(result, [])

  def test_consumed_generator_filter(self):
    result = Chain(self._make_consumed()).filter(lambda x: True).run()
    self.assertEqual(result, [])

  def test_consumed_generator_iterate(self):
    result = list(Chain(self._make_consumed()).iterate())
    self.assertEqual(result, [])

  def test_consumed_generator_iterate_do(self):
    result = list(Chain(self._make_consumed()).iterate_do(_identity))
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Class 9: TestDictIterationMatrix
# ---------------------------------------------------------------------------

class TestDictIterationMatrix(unittest.TestCase):
  """Special dict iteration patterns: keys, values, items."""

  def test_map_dict_keys(self):
    result = Chain({'a': 1, 'b': 2, 'c': 3}).map(str.upper).run()
    self.assertEqual(sorted(result), ['A', 'B', 'C'])

  def test_map_dict_values(self):
    result = Chain({'a': 1, 'b': 2, 'c': 3}.values()).map(_double).run()
    self.assertEqual(sorted(result), [2, 4, 6])

  def test_map_dict_items(self):
    result = Chain({'a': 1, 'b': 2}.items()).map(lambda kv: f'{kv[0]}={kv[1]}').run()
    self.assertEqual(sorted(result), ['a=1', 'b=2'])

  def test_filter_dict_keys(self):
    result = Chain({'a': 1, 'bb': 2, 'ccc': 3}).filter(lambda k: len(k) > 1).run()
    self.assertEqual(sorted(result), ['bb', 'ccc'])

  def test_filter_dict_values(self):
    result = Chain({'a': 1, 'b': 2, 'c': 3}.values()).filter(lambda v: v > 1).run()
    self.assertEqual(sorted(result), [2, 3])

  def test_filter_dict_items(self):
    result = Chain({'a': 1, 'b': 2, 'c': 3}.items()).filter(lambda kv: kv[1] >= 2).run()
    self.assertEqual(sorted(result, key=lambda x: x[0]), [('b', 2), ('c', 3)])

  def test_iterate_dict_keys(self):
    result = list(Chain({'x': 1, 'y': 2}).iterate())
    self.assertEqual(sorted(result), ['x', 'y'])

  def test_iterate_dict_values(self):
    result = list(Chain({'x': 10, 'y': 20}.values()).iterate())
    self.assertEqual(sorted(result), [10, 20])

  def test_iterate_dict_items(self):
    result = list(Chain({'x': 10, 'y': 20}.items()).iterate())
    self.assertEqual(sorted(result, key=lambda x: x[0]), [('x', 10), ('y', 20)])


# ---------------------------------------------------------------------------
# Class 10: TestBytesAndStringMatrix
# ---------------------------------------------------------------------------

class TestBytesAndStringMatrix(unittest.TestCase):
  """Specific tests for bytes and string iteration edge cases."""

  def test_bytes_map(self):
    result = Chain(b'\x01\x02\x03').map(lambda b: b + 10).run()
    self.assertEqual(result, [11, 12, 13])

  def test_bytes_filter(self):
    result = Chain(b'\x00\x01\x02\x03').filter(lambda b: b > 1).run()
    self.assertEqual(result, [2, 3])

  def test_bytes_iterate(self):
    result = list(Chain(b'\x0a\x0b').iterate())
    self.assertEqual(result, [10, 11])

  def test_string_map(self):
    result = Chain('hello').map(str.upper).run()
    self.assertEqual(result, ['H', 'E', 'L', 'L', 'O'])

  def test_string_filter(self):
    result = Chain('aAbBcC').filter(str.isupper).run()
    self.assertEqual(result, ['A', 'B', 'C'])

  def test_string_iterate(self):
    result = list(Chain('xyz').iterate())
    self.assertEqual(result, ['x', 'y', 'z'])

  def test_empty_bytes_map(self):
    result = Chain(b'').map(_identity).run()
    self.assertEqual(result, [])

  def test_empty_string_map(self):
    result = Chain('').map(_identity).run()
    self.assertEqual(result, [])

  def test_single_byte_map(self):
    result = Chain(b'\xff').map(_identity).run()
    self.assertEqual(result, [255])

  def test_single_char_map(self):
    result = Chain('z').map(_identity).run()
    self.assertEqual(result, ['z'])


# ---------------------------------------------------------------------------
# Class 11: TestFrozensetAndSetMatrix
# ---------------------------------------------------------------------------

class TestFrozensetAndSetMatrix(unittest.TestCase):
  """Tests for set/frozenset iteration (order-insensitive)."""

  def test_set_map(self):
    result = Chain({1, 2, 3}).map(_double).run()
    self.assertEqual(sorted(result), [2, 4, 6])

  def test_frozenset_map(self):
    result = Chain(frozenset({1, 2, 3})).map(_double).run()
    self.assertEqual(sorted(result), [2, 4, 6])

  def test_set_filter(self):
    result = Chain({1, 2, 3, 4}).filter(_is_even).run()
    self.assertEqual(sorted(result), [2, 4])

  def test_frozenset_filter(self):
    result = Chain(frozenset({1, 2, 3, 4})).filter(_is_even).run()
    self.assertEqual(sorted(result), [2, 4])

  def test_set_iterate(self):
    result = sorted(list(Chain({3, 1, 2}).iterate()))
    self.assertEqual(result, [1, 2, 3])

  def test_frozenset_iterate(self):
    result = sorted(list(Chain(frozenset({3, 1, 2})).iterate()))
    self.assertEqual(result, [1, 2, 3])

  def test_set_foreach(self):
    result = Chain({10, 20, 30}).foreach(_double).run()
    self.assertEqual(sorted(result), [10, 20, 30])

  def test_frozenset_foreach(self):
    result = Chain(frozenset({10, 20, 30})).foreach(_double).run()
    self.assertEqual(sorted(result), [10, 20, 30])

  def test_empty_set_map(self):
    self.assertEqual(Chain(set()).map(_identity).run(), [])

  def test_empty_frozenset_map(self):
    self.assertEqual(Chain(frozenset()).map(_identity).run(), [])


if __name__ == '__main__':
  unittest.main()
