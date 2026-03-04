import asyncio
from itertools import product
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_
from quent import Chain, QuentException


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
# ForeachTests
# ---------------------------------------------------------------------------
class ForeachTests(IsolatedAsyncioTestCase):

  async def test_foreach_sync_list(self):
    """Chain([1,2,3]).foreach(lambda x: x*2).run() = [2,4,6]."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [1, 2, 3]).foreach(lambda x: x * 2).run()),
          [2, 4, 6]
        )

  async def test_foreach_async_iterable(self):
    """Async iterable with sync fn."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, AsyncIterator([1, 2, 3])).foreach(lambda x: x * 2).run()),
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
    self.assertEqual(
      await Chain([1, 2, 3, 4]).foreach(f).run(),
      [10, 20, 30, 40]
    )

  async def test_foreach_async_fn_on_sync_iterable(self):
    """Sync iterable, async fn."""
    self.assertEqual(
      await Chain([1, 2, 3]).foreach(lambda x: aempty(x * 3)).run(),
      [3, 6, 9]
    )

  async def test_foreach_break_returns_partial(self):
    """Break at element 3: returns partial list."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el >= 3:
            return Chain.break_()
          return fn2(el * 2)

        self.assertEqual(
          await await_(Chain(fn1, SyncIterator([1, 2, 3, 4, 5])).foreach(f).run()),
          [2, 4]
        )

  async def test_foreach_break_with_value(self):
    """Chain.break_(obj) returns obj."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        sentinel = object()

        def f(el):
          if el == 2:
            return Chain.break_(sentinel)
          return fn2(el)

        self.assertIs(
          await await_(Chain(fn1, [1, 2, 3]).foreach(f).run()), sentinel
        )

  async def test_foreach_empty_list(self):
    """Chain([]).foreach(fn).run() = []."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, []).foreach(lambda x: x * 2).run()),
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
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el >= 3:
            return Chain.break_()
          return fn2(el * 2)

        self.assertEqual(
          await await_(Chain(fn1, AsyncIterator([1, 2, 3, 4, 5])).foreach(f).run()),
          [2, 4]
        )


# ---------------------------------------------------------------------------
# FilterTests
# ---------------------------------------------------------------------------
class FilterTests(IsolatedAsyncioTestCase):

  async def test_filter_sync_keeps_matching(self):
    """Chain([1,2,3,4]).filter(lambda x: x%2==0).run() = [2,4]."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [1, 2, 3, 4]).filter(lambda x: x % 2 == 0).run()),
          [2, 4]
        )

  async def test_filter_sync_none_pass(self):
    """All filtered out returns []."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [1, 3, 5]).filter(lambda x: x % 2 == 0).run()),
          []
        )

  async def test_filter_sync_all_pass(self):
    """All pass returns same list."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [2, 4, 6]).filter(lambda x: x % 2 == 0).run()),
          [2, 4, 6]
        )

  async def test_filter_empty(self):
    """Chain([]).filter(fn).run() = []."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, []).filter(lambda x: x > 0).run()),
          []
        )

  async def test_filter_async_iterable(self):
    """Async iterable with sync predicate."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, AsyncIterator([1, 2, 3, 4])).filter(lambda x: x % 2 == 0).run()),
          [2, 4]
        )

  async def test_filter_async_predicate(self):
    """Sync iterable, async predicate."""
    self.assertEqual(
      await Chain([1, 2, 3, 4]).filter(lambda x: aempty(x % 2 == 0)).run(),
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
    self.assertEqual(
      await Chain([1, 2, 3, 4, 5, 6]).filter(pred).run(),
      [2, 4, 6]
    )

  async def test_filter_exception_propagates(self):
    """Exception in predicate propagates."""
    def pred(x):
      raise ValueError('filter error')

    with self.assertRaises(ValueError):
      await await_(Chain([1, 2, 3]).filter(pred).run())


# ---------------------------------------------------------------------------
# GatherTests
# ---------------------------------------------------------------------------
class GatherTests(IsolatedAsyncioTestCase):

  async def test_gather_all_sync(self):
    """Chain(5).gather(f1, f2, f3).run() = [f1(5), f2(5), f3(5)]."""
    f1 = lambda v: v + 1
    f2 = lambda v: v * 2
    f3 = lambda v: v ** 2

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 5).gather(f1, f2, f3).run()),
          [6, 10, 25]
        )

  async def test_gather_all_async(self):
    """All fns async."""
    f1 = lambda v: aempty(v + 1)
    f2 = lambda v: aempty(v * 2)
    f3 = lambda v: aempty(v ** 2)

    self.assertEqual(
      await Chain(5).gather(f1, f2, f3).run(),
      [6, 10, 25]
    )

  async def test_gather_mixed_sync_async(self):
    """Some sync, some async."""
    f1 = lambda v: v + 1
    f2 = lambda v: aempty(v * 2)
    f3 = lambda v: v ** 2

    self.assertEqual(
      await Chain(5).gather(f1, f2, f3).run(),
      [6, 10, 25]
    )

  async def test_gather_single_fn(self):
    """Single fn returns [result]."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 5).gather(lambda v: v * 3).run()),
          [15]
        )

  async def test_gather_empty_fns(self):
    """Chain(5).gather().run() = []."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 5).gather().run()),
          []
        )

  async def test_gather_receives_same_value(self):
    """All fns receive the same current value."""
    received = []

    def capture(v):
      received.append(v)
      return v

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        received.clear()
        self.assertEqual(
          await await_(Chain(fn, 42).gather(capture, capture, capture).run()),
          [42, 42, 42]
        )
        self.assertEqual(received, [42, 42, 42])

  async def test_gather_preserves_order(self):
    """Results match fn order even with async."""
    async def slow(v):
      await asyncio.sleep(0.01)
      return 'slow'

    def fast(v):
      return 'fast'

    self.assertEqual(
      await Chain(1).gather(slow, fast, slow).run(),
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
class IterateTests(IsolatedAsyncioTestCase):

  async def test_iterate_sync_for(self):
    """for i in Chain(SyncIterClass).iterate(fn): ..."""
    r = []
    for i in Chain(SyncIterator).iterate(lambda i: i * 2):
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(10)])

  async def test_iterate_sync_for_no_fn(self):
    """for i in Chain(SyncIterClass).iterate(): ..."""
    r = []
    for i in Chain(SyncIterator).iterate():
      r.append(i)
    self.assertEqual(r, list(range(10)))

  async def test_iterate_async_for(self):
    """async for i in Chain(AsyncIterClass).iterate(fn): ..."""
    r = []
    async for i in Chain(AsyncIterator).iterate(lambda i: i * 2):
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(10)])

  async def test_iterate_break_stops(self):
    """fn returns Chain.break_() stops the generator."""
    def f(i):
      if i >= 5:
        return Chain.break_()
      return i * 2

    r = []
    for i in Chain(SyncIterator).iterate(f):
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(5)])

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
    self.assertEqual(r, [i * 2 for i in range(10)])

    # Also test with async for
    r = []
    async for i in Chain(SyncIterator).then(Chain().iterate(lambda i: i * 2)).run():
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(10)])

    # Async iterator nesting
    r = []
    async for i in Chain(AsyncIterator).then(Chain().iterate(lambda i: i * 2)).run():
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(10)])

  async def test_iterate_generator_repr(self):
    """repr(_Generator(...)) returns '<_Generator>'."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    self.assertEqual(repr(gen), '<_Generator>')

  async def test_async_iterate_with_async_fn(self):
    """fn returns coroutine inside async generator."""
    r = []
    async for i in Chain(SyncIterator).iterate(lambda i: aempty(i * 3)):
      r.append(i)
    self.assertEqual(r, [i * 3 for i in range(10)])

    # Also with async iterator
    r = []
    async for i in Chain(AsyncIterator).iterate(lambda i: aempty(i * 3)):
      r.append(i)
    self.assertEqual(r, [i * 3 for i in range(10)])
