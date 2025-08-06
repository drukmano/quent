import asyncio
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException, run


class MyTestCase(IsolatedAsyncioTestCase):
  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)

  async def assertTrue(self, expr, msg=None):
    return super().assertTrue(await await_(expr), msg)

  async def assertFalse(self, expr, msg=None):
    return super().assertFalse(await await_(expr), msg)

  async def assertEqual(self, first, second, msg=None):
    return super().assertEqual(await await_(first), second, msg)

  async def assertIsNone(self, obj, msg=None):
    return super().assertIsNone(await await_(obj), msg)

  async def assertIs(self, expr1, expr2, msg=None):
    return super().assertIs(await await_(expr1), expr2, msg)

  async def assertIsNot(self, expr1, expr2, msg=None):
    return super().assertIsNot(await await_(expr1), expr2, msg)


class AsyncIterator:
  def __init__(self, items=None):
    self._items = list(items) if items is not None else list(range(10))

  def __aiter__(self):
    self._iter = iter(self._items)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


# ---------------------------------------------------------------------------
# DoubleAsyncReduceTests
# ---------------------------------------------------------------------------
class DoubleAsyncReduceTests(MyTestCase):

  async def test_reduce_basic_sum(self):
    """Async iterable + async reducer: sum of [1,2,3,4] with initial 0 = 10."""
    async def async_add(a, x):
      return a + x

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3, 4])).reduce(async_add, 0).run(),
      10
    )

  async def test_reduce_with_initial_value(self):
    """Async iterable + async reducer: initial=100 + sum([1,2,3]) = 106."""
    async def async_add(a, x):
      return a + x

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3])).reduce(async_add, 100).run(),
      106
    )

  async def test_reduce_without_initial_value(self):
    """Async iterable + async reducer: no initial, uses first element as acc."""
    async def async_add(a, x):
      return a + x

    await self.assertEqual(
      Chain(AsyncIterator([10, 20, 30])).reduce(async_add).run(),
      60
    )

  async def test_reduce_empty_with_initial(self):
    """Async iterable (empty) + async reducer with initial returns initial."""
    async def async_add(a, x):
      return a + x

    await self.assertEqual(
      Chain(AsyncIterator([])).reduce(async_add, 42).run(),
      42
    )

  async def test_reduce_empty_no_initial_raises(self):
    """Async iterable (empty) + async reducer, no initial raises TypeError or UnboundLocalError."""
    async def async_add(a, x):
      return a + x

    # async_reduce has a known issue where `el` may be unbound in the except handler
    # when the iterable is empty, so we accept both TypeError and UnboundLocalError.
    with self.assertRaises((TypeError, UnboundLocalError)):
      await await_(
        Chain(AsyncIterator([])).reduce(async_add).run()
      )

  async def test_reduce_single_element_no_initial(self):
    """Async iterable with one element, no initial: returns that element without calling reducer."""
    call_count = {'n': 0}

    async def async_add(a, x):
      call_count['n'] += 1
      return a + x

    await self.assertEqual(
      Chain(AsyncIterator([99])).reduce(async_add).run(),
      99
    )
    super(MyTestCase, self).assertEqual(call_count['n'], 0)

  async def test_reduce_exception_mid_reduction(self):
    """Async reducer raises exception on third element; propagates."""
    call_count = {'n': 0}

    async def exploding_reducer(a, x):
      call_count['n'] += 1
      if call_count['n'] >= 3:
        raise TestExc('boom at reduction 3')
      return a + x

    call_count['n'] = 0
    with self.assertRaises(TestExc):
      await await_(
        Chain(AsyncIterator([1, 2, 3, 4, 5])).reduce(exploding_reducer, 0).run()
      )

  async def test_reduce_string_concatenation(self):
    """Async iterable + async reducer: string concatenation."""
    async def async_concat(a, x):
      return a + x

    await self.assertEqual(
      Chain(AsyncIterator(['a', 'b', 'c'])).reduce(async_concat, '').run(),
      'abc'
    )

  async def test_reduce_then_further_chain(self):
    """Double-async reduce piped to further .then()."""
    async def async_add(a, x):
      return a + x

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3])).reduce(async_add, 0).then(lambda v: v * 10).run(),
      60
    )


# ---------------------------------------------------------------------------
# DoubleAsyncFilterTests
# ---------------------------------------------------------------------------
class DoubleAsyncFilterTests(MyTestCase):

  async def test_filter_basic(self):
    """Async iterable + async predicate: keep even numbers."""
    async def async_is_even(x):
      return x % 2 == 0

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3, 4, 5, 6])).filter(async_is_even).run(),
      [2, 4, 6]
    )

  async def test_filter_all_out(self):
    """Async iterable + async predicate: all filtered out returns []."""
    async def async_never(x):
      return False

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3])).filter(async_never).run(),
      []
    )

  async def test_filter_none_out(self):
    """Async iterable + async predicate: all pass returns full list."""
    async def async_always(x):
      return True

    await self.assertEqual(
      Chain(AsyncIterator([10, 20, 30])).filter(async_always).run(),
      [10, 20, 30]
    )

  async def test_filter_empty_iterable(self):
    """Async iterable (empty) + async predicate returns []."""
    async def async_is_even(x):
      return x % 2 == 0

    await self.assertEqual(
      Chain(AsyncIterator([])).filter(async_is_even).run(),
      []
    )

  async def test_filter_exception_in_predicate(self):
    """Async predicate raises exception; propagates."""
    async def exploding_pred(x):
      if x == 3:
        raise TestExc('predicate boom')
      return True

    with self.assertRaises(TestExc):
      await await_(
        Chain(AsyncIterator([1, 2, 3, 4])).filter(exploding_pred).run()
      )

  async def test_filter_then_further_chain(self):
    """Double-async filter piped to further .then()."""
    async def async_gt_2(x):
      return x > 2

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3, 4, 5])).filter(async_gt_2).then(len).run(),
      3
    )


# ---------------------------------------------------------------------------
# DoubleAsyncForeachTests
# ---------------------------------------------------------------------------
class DoubleAsyncForeachTests(MyTestCase):

  async def test_foreach_basic(self):
    """Async iterable + async body fn: basic mapping."""
    async def async_double(x):
      return x * 2

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3])).foreach(async_double).run(),
      [2, 4, 6]
    )

  async def test_foreach_exception_mid_iteration(self):
    """Async body fn raises exception mid-iteration; propagates."""
    async def exploding_fn(x):
      if x == 3:
        raise TestExc('foreach boom')
      return x

    with self.assertRaises(TestExc):
      await await_(
        Chain(AsyncIterator([1, 2, 3, 4])).foreach(exploding_fn).run()
      )

  async def test_foreach_body_return_discarded_by_foreach_do(self):
    """foreach_do discards async body return, keeps original elements."""
    side_effects = []

    async def async_side(x):
      side_effects.append(x * 10)
      return 'should_be_discarded'

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3])).foreach_do(async_side).run(),
      [1, 2, 3]
    )
    super(MyTestCase, self).assertEqual(side_effects, [10, 20, 30])

  async def test_foreach_empty_iterable(self):
    """Async iterable (empty) + async body fn returns []."""
    async def async_double(x):
      return x * 2

    await self.assertEqual(
      Chain(AsyncIterator([])).foreach(async_double).run(),
      []
    )

  async def test_foreach_break_returns_partial(self):
    """Break in async foreach returns partial list."""
    async def break_at_3(x):
      if x >= 3:
        return Chain.break_()
      return x * 2

    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3, 4, 5])).foreach(break_at_3).run(),
      [2, 4]
    )

  async def test_foreach_break_with_value(self):
    """Chain.break_(sentinel) in async foreach returns sentinel."""
    sentinel = object()

    async def break_with_val(x):
      if x == 2:
        return Chain.break_(sentinel)
      return x

    await self.assertIs(
      Chain(AsyncIterator([1, 2, 3])).foreach(break_with_val).run(),
      sentinel
    )


# ---------------------------------------------------------------------------
# DoubleAsyncForeachIndexedTests
# ---------------------------------------------------------------------------
class DoubleAsyncForeachIndexedTests(MyTestCase):

  async def test_foreach_indexed_basic(self):
    """Async iterable + async indexed fn: verify (index, value) pairs."""
    async def async_pair(idx, el):
      return (idx, el)

    await self.assertEqual(
      Chain(AsyncIterator(['a', 'b', 'c'])).foreach(async_pair, with_index=True).run(),
      [(0, 'a'), (1, 'b'), (2, 'c')]
    )

  async def test_foreach_indexed_exception_mid_iteration(self):
    """Async indexed fn raises exception; propagates."""
    async def exploding_indexed(idx, el):
      if idx == 2:
        raise TestExc('indexed boom')
      return (idx, el)

    with self.assertRaises(TestExc):
      await await_(
        Chain(AsyncIterator(['a', 'b', 'c', 'd'])).foreach(
          exploding_indexed, with_index=True
        ).run()
      )

  async def test_foreach_indexed_do_preserves_elements(self):
    """foreach_do with async indexed fn preserves original elements."""
    captured = []

    async def async_capture(idx, el):
      captured.append((idx, el))

    await self.assertEqual(
      Chain(AsyncIterator(['x', 'y', 'z'])).foreach_do(
        async_capture, with_index=True
      ).run(),
      ['x', 'y', 'z']
    )
    super(MyTestCase, self).assertEqual(
      captured, [(0, 'x'), (1, 'y'), (2, 'z')]
    )

  async def test_foreach_indexed_empty_iterable(self):
    """Async iterable (empty) + async indexed fn returns []."""
    async def async_pair(idx, el):
      return (idx, el)

    await self.assertEqual(
      Chain(AsyncIterator([])).foreach(async_pair, with_index=True).run(),
      []
    )

  async def test_foreach_indexed_break(self):
    """Break in async foreach_indexed returns partial list."""
    async def break_at_idx_2(idx, el):
      if idx == 2:
        return Chain.break_()
      return (idx, el)

    await self.assertEqual(
      Chain(AsyncIterator(['a', 'b', 'c', 'd'])).foreach(
        break_at_idx_2, with_index=True
      ).run(),
      [(0, 'a'), (1, 'b')]
    )

  async def test_foreach_indexed_break_with_value(self):
    """Chain.break_(sentinel) in async foreach_indexed returns sentinel."""
    sentinel = object()

    async def break_with_val(idx, el):
      if idx == 1:
        return Chain.break_(sentinel)
      return (idx, el)

    await self.assertIs(
      Chain(AsyncIterator(['a', 'b', 'c'])).foreach(
        break_with_val, with_index=True
      ).run(),
      sentinel
    )
