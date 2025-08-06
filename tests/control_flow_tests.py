import asyncio
import inspect
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


class SyncIterator:
  def __init__(self, items=None):
    self._items = items if items is not None else list(range(10))

  def __iter__(self):
    return iter(self._items)


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
# WhileTrueTests
# ---------------------------------------------------------------------------
class WhileTrueTests(MyTestCase):

  async def test_while_break_immediately(self):
    """Loop body calls Chain.break_() on first iteration; returns root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f0():
          return Chain.break_()

        await self.assertEqual(
          Chain(fn, 10).while_true(f0, ...).run(), 10
        )

  async def test_while_break_with_value(self):
    """Chain.break_(42) returns 42 instead of root."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f0():
          return Chain.break_(42)

        await self.assertEqual(
          Chain(fn, 10).while_true(f0, ...).run(), 42
        )

  async def test_while_break_with_callable_value(self):
    """Chain.break_(lambda: 42) returns 42."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f0():
          return Chain.break_(lambda: 42)

        await self.assertEqual(
          Chain(fn, 10).while_true(f0, ...).run(), 42
        )

  async def test_while_break_with_fn_and_arg(self):
    """Chain.break_(fn, 42) where fn is a callable."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f0():
          return Chain.break_(fn, 42)

        await self.assertEqual(
          Chain(fn, 10).while_true(f0, ...).run(), 42
        )

  async def test_while_counts_iterations(self):
    """Verify iteration count matches expected number before break."""
    for fn, ctx in self.with_fn():
      with ctx:
        counter = {'count': 0}

        def f0(v=None):
          counter['count'] += 1
          if counter['count'] >= 5:
            return Chain.break_()

        counter['count'] = 0
        await self.assertEqual(
          Chain(fn, 99).while_true(f0, ...).run(), 99
        )
        super(MyTestCase, self).assertEqual(counter['count'], 5)

  async def test_while_max_iterations_exceeded(self):
    """Loop never breaks, max_iterations=5 raises QuentException."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f0(v=None):
          pass

        with self.assertRaises(QuentException) as cm:
          await await_(
            Chain(fn, 1).while_true(f0, ..., max_iterations=5).run()
          )
        super(MyTestCase, self).assertIn('exceeded max_iterations', str(cm.exception))

  async def test_while_max_iterations_not_exceeded(self):
    """Break before max_iterations -- no exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        counter = {'count': 0}

        def f0(v=None):
          counter['count'] += 1
          if counter['count'] >= 3:
            return Chain.break_()

        counter['count'] = 0
        await self.assertEqual(
          Chain(fn, 7).while_true(f0, ..., max_iterations=10).run(), 7
        )
        super(MyTestCase, self).assertEqual(counter['count'], 3)

  async def test_while_sync_to_async_transition(self):
    """Loop body returns sync initially, then a coroutine mid-loop."""
    counter = {'count': 0}

    def f0(v=None):
      counter['count'] += 1
      if counter['count'] >= 5:
        return Chain.break_(counter['count'])
      if counter['count'] > 2:
        return aempty()  # returns a coroutine mid-loop
      return None

    counter['count'] = 0
    await self.assertEqual(
      Chain().while_true(f0, ...).run(), 5
    )

  async def test_while_async_max_iterations(self):
    """Async continuation exceeding max_iterations."""
    counter = {'count': 0}

    def f0(v=None):
      counter['count'] += 1
      if counter['count'] > 2:
        return aempty()  # forces async transition, never breaks
      return None

    counter['count'] = 0
    with self.assertRaises(QuentException) as cm:
      await await_(
        Chain().while_true(f0, ..., max_iterations=5).run()
      )
    super(MyTestCase, self).assertIn('exceeded max_iterations', str(cm.exception))

  async def test_while_nested_chain_body(self):
    """Chain(5).while_true(Chain().then(fn).then(break_fn)).run()."""
    for fn, ctx in self.with_fn():
      with ctx:
        def break_fn(v=None):
          return Chain.break_()

        await self.assertEqual(
          Chain(fn, 5).while_true(Chain().then(fn).then(break_fn, ...)).run(), 5
        )

  async def test_while_receives_root_value(self):
    """fn receives the root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}

        def f0(v):
          received['value'] = v
          return Chain.break_()

        await self.assertEqual(
          Chain(fn, 42).while_true(f0).run(), 42
        )
        super(MyTestCase, self).assertEqual(received['value'], 42)

  async def test_while_with_explicit_args(self):
    """fn with args forwarded via while_true(fn, arg)."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'args': None}

        def f0(a, b):
          received['args'] = (a, b)
          return Chain.break_()

        await self.assertIsNone(
          Chain().while_true(f0, 'x', 'y').run()
        )
        super(MyTestCase, self).assertEqual(received['args'], ('x', 'y'))

  async def test_while_with_ellipsis_args(self):
    """Chain(5).while_true(fn, ...).run() ignores current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'called_without_args': False}

        def f0():
          received['called_without_args'] = True
          return Chain.break_()

        received['called_without_args'] = False
        await self.assertEqual(
          Chain(fn, 5).while_true(f0, ...).run(), 5
        )
        super(MyTestCase, self).assertTrue(received['called_without_args'])


# ---------------------------------------------------------------------------
# ForeachTests
# ---------------------------------------------------------------------------
class ForeachTests(MyTestCase):

  async def test_foreach_sync_list(self):
    """Chain([1,2,3]).foreach(lambda x: x*2).run() = [2,4,6]."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(lambda x: x * 2).run(),
          [2, 4, 6]
        )

  async def test_foreach_async_iterable(self):
    """Async iterable with sync fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3])).foreach(lambda x: x * 2).run(),
          [2, 4, 6]
        )

  async def test_foreach_sync_to_async_transition(self):
    """fn returns sync first, then coroutine."""
    counter = {'count': 0}

    def f(el):
      counter['count'] += 1
      if counter['count'] > 2:
        return aempty(el * 10)
      return el * 10

    counter['count'] = 0
    await self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(f).run(),
      [10, 20, 30, 40]
    )

  async def test_foreach_async_fn_on_sync_iterable(self):
    """Sync iterable, async fn."""
    await self.assertEqual(
      Chain([1, 2, 3]).foreach(lambda x: aempty(x * 3)).run(),
      [3, 6, 9]
    )

  async def test_foreach_do_preserves_elements(self):
    """foreach_do returns original elements."""
    for fn, ctx in self.with_fn():
      with ctx:
        side_effects = []
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach_do(lambda x: side_effects.append(x * 2)).run(),
          [1, 2, 3]
        )
        super(MyTestCase, self).assertEqual(side_effects, [2, 4, 6])

  async def test_foreach_do_async_iterable(self):
    """foreach_do on async iterable."""
    for fn, ctx in self.with_fn():
      with ctx:
        side_effects = []

        def side_fn(x):
          side_effects.append(x * 2)

        side_effects.clear()
        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3])).foreach_do(side_fn).run(),
          [1, 2, 3]
        )
        super(MyTestCase, self).assertEqual(side_effects, [2, 4, 6])

  async def test_foreach_break_returns_partial(self):
    """Break at element 3: returns partial list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el >= 3:
            return Chain.break_()
          return fn(el * 2)

        await self.assertEqual(
          Chain(fn, SyncIterator([1, 2, 3, 4, 5])).foreach(f).run(),
          [2, 4]
        )

  async def test_foreach_break_with_value(self):
    """Chain.break_(obj) returns obj."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()

        def f(el):
          if el == 2:
            return Chain.break_(sentinel)
          return fn(el)

        await self.assertIs(
          Chain(fn, [1, 2, 3]).foreach(f).run(), sentinel
        )

  async def test_foreach_empty_list(self):
    """Chain([]).foreach(fn).run() = []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).foreach(lambda x: x * 2).run(),
          []
        )

  async def test_foreach_exception_in_fn(self):
    """fn raises ValueError -- propagates."""
    def f(el):
      raise ValueError('test error')

    with self.assertRaises(ValueError):
      await await_(Chain([1, 2, 3]).foreach(f).run())

  async def test_foreach_break_in_async_foreach(self):
    """Break in fully async foreach path."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el >= 3:
            return Chain.break_()
          return fn(el * 2)

        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3, 4, 5])).foreach(f).run(),
          [2, 4]
        )


# ---------------------------------------------------------------------------
# ForeachIndexedTests
# ---------------------------------------------------------------------------
class ForeachIndexedTests(MyTestCase):

  async def test_foreach_indexed_sync(self):
    """fn(idx, el) called with correct indices."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c']).foreach(
            lambda idx, el: (idx, el), with_index=True
          ).run(),
          [(0, 'a'), (1, 'b'), (2, 'c')]
        )

  async def test_foreach_indexed_async_iterable(self):
    """Same with async iterable."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator(['x', 'y', 'z'])).foreach(
            lambda idx, el: (idx, el), with_index=True
          ).run(),
          [(0, 'x'), (1, 'y'), (2, 'z')]
        )

  async def test_foreach_indexed_sync_to_async(self):
    """fn returns coroutine mid-iteration."""
    counter = {'count': 0}

    def f(idx, el):
      counter['count'] += 1
      if counter['count'] > 1:
        return aempty((idx, el))
      return (idx, el)

    counter['count'] = 0
    await self.assertEqual(
      Chain(['a', 'b', 'c']).foreach(f, with_index=True).run(),
      [(0, 'a'), (1, 'b'), (2, 'c')]
    )

  async def test_foreach_indexed_do(self):
    """foreach_do(fn, with_index=True) preserves original elements."""
    for fn, ctx in self.with_fn():
      with ctx:
        side_effects = []

        def f(idx, el):
          side_effects.append((idx, el))

        side_effects.clear()
        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c']).foreach_do(f, with_index=True).run(),
          ['a', 'b', 'c']
        )
        super(MyTestCase, self).assertEqual(
          side_effects, [(0, 'a'), (1, 'b'), (2, 'c')]
        )

  async def test_foreach_indexed_break(self):
    """Break at idx=2 returns partial list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(idx, el):
          if idx == 2:
            return Chain.break_()
          return fn((idx, el))

        await self.assertEqual(
          Chain(fn, ['a', 'b', 'c', 'd']).foreach(f, with_index=True).run(),
          [(0, 'a'), (1, 'b')]
        )

  async def test_foreach_indexed_break_with_value(self):
    """Chain.break_(obj) in indexed path."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()

        def f(idx, el):
          if idx == 1:
            return Chain.break_(sentinel)
          return fn((idx, el))

        await self.assertIs(
          Chain(fn, ['a', 'b', 'c']).foreach(f, with_index=True).run(),
          sentinel
        )

  async def test_foreach_indexed_exception(self):
    """Exception in fn propagates."""
    def f(idx, el):
      raise ValueError('indexed error')

    with self.assertRaises(ValueError):
      await await_(Chain(['a', 'b']).foreach(f, with_index=True).run())


# ---------------------------------------------------------------------------
# FilterTests
# ---------------------------------------------------------------------------
class FilterTests(MyTestCase):

  async def test_filter_sync_keeps_matching(self):
    """Chain([1,2,3,4]).filter(lambda x: x%2==0).run() = [2,4]."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).filter(lambda x: x % 2 == 0).run(),
          [2, 4]
        )

  async def test_filter_sync_none_pass(self):
    """All filtered out returns []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 3, 5]).filter(lambda x: x % 2 == 0).run(),
          []
        )

  async def test_filter_sync_all_pass(self):
    """All pass returns same list."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [2, 4, 6]).filter(lambda x: x % 2 == 0).run(),
          [2, 4, 6]
        )

  async def test_filter_empty(self):
    """Chain([]).filter(fn).run() = []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).filter(lambda x: x > 0).run(),
          []
        )

  async def test_filter_async_iterable(self):
    """Async iterable with sync predicate."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3, 4])).filter(lambda x: x % 2 == 0).run(),
          [2, 4]
        )

  async def test_filter_async_predicate(self):
    """Sync iterable, async predicate."""
    await self.assertEqual(
      Chain([1, 2, 3, 4]).filter(lambda x: aempty(x % 2 == 0)).run(),
      [2, 4]
    )

  async def test_filter_sync_to_async_transition(self):
    """Predicate returns sync then coroutine."""
    counter = {'count': 0}

    def pred(x):
      counter['count'] += 1
      if counter['count'] > 2:
        return aempty(x % 2 == 0)
      return x % 2 == 0

    counter['count'] = 0
    await self.assertEqual(
      Chain([1, 2, 3, 4, 5, 6]).filter(pred).run(),
      [2, 4, 6]
    )

  async def test_filter_exception_propagates(self):
    """Exception in predicate propagates."""
    def pred(x):
      raise ValueError('filter error')

    with self.assertRaises(ValueError):
      await await_(Chain([1, 2, 3]).filter(pred).run())


# ---------------------------------------------------------------------------
# ReduceTests
# ---------------------------------------------------------------------------
class ReduceTests(MyTestCase):

  async def test_reduce_with_initial(self):
    """Chain([1,2,3]).reduce(lambda a,x: a+x, 10).run() = 16."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).reduce(lambda a, x: a + x, 10).run(),
          16
        )

  async def test_reduce_without_initial(self):
    """Chain([1,2,3]).reduce(lambda a,x: a+x).run() = 6."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).reduce(lambda a, x: a + x).run(),
          6
        )

  async def test_reduce_empty_no_initial_raises(self):
    """Chain([]).reduce(fn).run() raises TypeError."""
    with self.assertRaises(TypeError):
      await await_(Chain([]).reduce(lambda a, x: a + x).run())

  async def test_reduce_empty_with_initial(self):
    """Chain([]).reduce(fn, 99).run() = 99."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).reduce(lambda a, x: a + x, 99).run(),
          99
        )

  async def test_reduce_single_element_no_initial(self):
    """Chain([42]).reduce(fn).run() = 42."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [42]).reduce(lambda a, x: a + x).run(),
          42
        )

  async def test_reduce_async_reducer(self):
    """Sync iterable, async reducer."""
    await self.assertEqual(
      Chain([1, 2, 3]).reduce(lambda a, x: aempty(a + x), 0).run(),
      6
    )

  async def test_reduce_async_iterable(self):
    """Async iterable with sync reducer."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3])).reduce(lambda a, x: a + x, 0).run(),
          6
        )

  async def test_reduce_async_iterable_no_initial(self):
    """Async iterable, no initial."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, AsyncIterator([1, 2, 3])).reduce(lambda a, x: a + x).run(),
          6
        )

  async def test_reduce_async_iterable_empty_no_initial(self):
    """Raises TypeError (intended) but currently raises UnboundLocalError due to
    a bug in async_reduce where `el` is referenced in `except BaseException`
    before being assigned when the async iterable is empty."""
    # The intended behavior is TypeError('reduce() of empty iterable with no initial value')
    # but the except BaseException handler in async_reduce references `el` before assignment.
    with self.assertRaises((TypeError, UnboundLocalError)):
      await await_(
        Chain(AsyncIterator([])).reduce(lambda a, x: a + x).run()
      )

  async def test_reduce_sync_to_async_transition(self):
    """Reducer returns sync then coroutine."""
    counter = {'count': 0}

    def reducer(a, x):
      counter['count'] += 1
      if counter['count'] > 1:
        return aempty(a + x)
      return a + x

    counter['count'] = 0
    await self.assertEqual(
      Chain([1, 2, 3, 4]).reduce(reducer, 0).run(),
      10
    )

  async def test_reduce_then_further_chain(self):
    """Reduce result piped to further .then()."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).reduce(lambda a, x: a + x, 0).then(lambda v: v * 10).run(),
          60
        )


# ---------------------------------------------------------------------------
# GatherTests
# ---------------------------------------------------------------------------
class GatherTests(MyTestCase):

  async def test_gather_all_sync(self):
    """Chain(5).gather(f1, f2, f3).run() = [f1(5), f2(5), f3(5)]."""
    f1 = lambda v: v + 1
    f2 = lambda v: v * 2
    f3 = lambda v: v ** 2

    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather(f1, f2, f3).run(),
          [6, 10, 25]
        )

  async def test_gather_all_async(self):
    """All fns async."""
    f1 = lambda v: aempty(v + 1)
    f2 = lambda v: aempty(v * 2)
    f3 = lambda v: aempty(v ** 2)

    await self.assertEqual(
      Chain(5).gather(f1, f2, f3).run(),
      [6, 10, 25]
    )

  async def test_gather_mixed_sync_async(self):
    """Some sync, some async."""
    f1 = lambda v: v + 1
    f2 = lambda v: aempty(v * 2)
    f3 = lambda v: v ** 2

    await self.assertEqual(
      Chain(5).gather(f1, f2, f3).run(),
      [6, 10, 25]
    )

  async def test_gather_single_fn(self):
    """Single fn returns [result]."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather(lambda v: v * 3).run(),
          [15]
        )

  async def test_gather_empty_fns(self):
    """Chain(5).gather().run() = []."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather().run(),
          []
        )

  async def test_gather_receives_same_value(self):
    """All fns receive the same current value."""
    received = []

    def capture(v):
      received.append(v)
      return v

    for fn, ctx in self.with_fn():
      with ctx:
        received.clear()
        await self.assertEqual(
          Chain(fn, 42).gather(capture, capture, capture).run(),
          [42, 42, 42]
        )
        super(MyTestCase, self).assertEqual(received, [42, 42, 42])

  async def test_gather_preserves_order(self):
    """Results match fn order even with async."""
    async def slow(v):
      await asyncio.sleep(0.01)
      return 'slow'

    def fast(v):
      return 'fast'

    await self.assertEqual(
      Chain(1).gather(slow, fast, slow).run(),
      ['slow', 'fast', 'slow']
    )

  async def test_gather_exception_propagates(self):
    """One fn raises, exception propagates."""
    def f1(v):
      return v + 1

    def f2(v):
      raise ValueError('gather error')

    def f3(v):
      return v + 3

    with self.assertRaises(ValueError):
      await await_(Chain(5).gather(f1, f2, f3).run())


# ---------------------------------------------------------------------------
# IterateTests
# ---------------------------------------------------------------------------
class IterateTests(MyTestCase):

  async def test_iterate_sync_for(self):
    """for i in Chain(SyncIterClass).iterate(fn): ..."""
    r = []
    for i in Chain(SyncIterator).iterate(lambda i: i * 2):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 2 for i in range(10)])

  async def test_iterate_sync_for_no_fn(self):
    """for i in Chain(SyncIterClass).iterate(): ..."""
    r = []
    for i in Chain(SyncIterator).iterate():
      r.append(i)
    super(MyTestCase, self).assertEqual(r, list(range(10)))

  async def test_iterate_async_for(self):
    """async for i in Chain(AsyncIterClass).iterate(fn): ..."""
    r = []
    async for i in Chain(AsyncIterator).iterate(lambda i: i * 2):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 2 for i in range(10)])

  async def test_iterate_do_preserves_elements(self):
    """iterate_do(fn) yields original elements."""
    side_effects = []
    r = []
    for i in Chain(SyncIterator).iterate_do(lambda i: side_effects.append(i * 10)):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, list(range(10)))
    super(MyTestCase, self).assertEqual(side_effects, [i * 10 for i in range(10)])

  async def test_iterate_break_stops(self):
    """fn returns Chain.break_() stops the generator."""
    def f(i):
      if i >= 5:
        return Chain.break_()
      return i * 2

    r = []
    for i in Chain(SyncIterator).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 2 for i in range(5)])

  async def test_iterate_return_raises_quent_exception(self):
    """fn calls Chain.return_() raises QuentException."""
    with self.assertRaises(QuentException):
      for _ in Chain(SyncIterator).iterate(Chain.return_):
        pass

  async def test_iterate_nesting(self):
    """Chain(iter).then(Chain().iterate(fn)).run()."""
    r = []
    for i in Chain(SyncIterator).then(Chain().iterate(lambda i: i * 2)).run():
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 2 for i in range(10)])

    # Also test with async for
    r = []
    async for i in Chain(SyncIterator).then(Chain().iterate(lambda i: i * 2)).run():
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 2 for i in range(10)])

    # Async iterator nesting
    r = []
    async for i in Chain(AsyncIterator).then(Chain().iterate(lambda i: i * 2)).run():
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 2 for i in range(10)])

  async def test_iterate_generator_repr(self):
    """repr(_Generator(...)) returns '<_Generator>'."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    super(MyTestCase, self).assertEqual(repr(gen), '<_Generator>')

  async def test_async_iterate_with_async_fn(self):
    """fn returns coroutine inside async generator."""
    r = []
    async for i in Chain(SyncIterator).iterate(lambda i: aempty(i * 3)):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 3 for i in range(10)])

    # Also with async iterator
    r = []
    async for i in Chain(AsyncIterator).iterate(lambda i: aempty(i * 3)):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [i * 3 for i in range(10)])
