from tests.utils import empty, aempty, await_, TestExc, MyTestCase
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
# GeneratorCallNestingTests
# ---------------------------------------------------------------------------
class GeneratorCallNestingTests(MyTestCase):

  async def test_call_creates_new_generator_instance(self):
    """Calling a _Generator returns a new _Generator, not the same object."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    nested = gen(SyncIterator([1, 2, 3]))
    super(MyTestCase, self).assertIsNot(gen, nested)

  async def test_call_preserves_repr(self):
    """Nested generator still has <_Generator> repr."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    nested = gen(SyncIterator([1, 2, 3]))
    super(MyTestCase, self).assertEqual(repr(nested), '<_Generator>')

  async def test_nested_generator_produces_correct_values(self):
    """A _Generator called with a new iterable produces values from that iterable."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i * 3)
    nested = gen(SyncIterator([10, 20, 30]))
    r = []
    for i in nested:
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [30, 60, 90])

  async def test_generator_reuse_across_calls(self):
    """The original _Generator can be iterated independently after creating a nested one."""
    gen = Chain(SyncIterator, [1, 2]).iterate(lambda i: i + 100)
    _ = gen(SyncIterator([7, 8]))
    r = []
    for i in gen:
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [101, 102])

  async def test_nested_generator_async_iteration(self):
    """A nested _Generator supports async for as well."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i * 5)
    nested = gen(SyncIterator([2, 4]))
    r = []
    async for i in nested:
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [10, 20])


# ---------------------------------------------------------------------------
# SyncGeneratorTests
# ---------------------------------------------------------------------------
class SyncGeneratorTests(MyTestCase):

  async def test_sync_generator_normal_completion(self):
    """Sync generator yields all elements and completes."""
    r = []
    for i in Chain(SyncIterator, [10, 20, 30]).iterate(lambda i: i + 1):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [11, 21, 31])

  async def test_sync_generator_no_fn(self):
    """Sync generator with no fn yields raw elements."""
    r = []
    for i in Chain(SyncIterator, [5, 6, 7]).iterate():
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [5, 6, 7])

  async def test_sync_generator_break_stops(self):
    """fn raises _Break via Chain.break_() which causes the generator to return."""
    def f(i):
      if i >= 3:
        return Chain.break_()
      return i * 10

    r = []
    for i in Chain(SyncIterator, list(range(6))).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [0, 10, 20])

  async def test_sync_generator_return_raises_quent_exception(self):
    """fn raises _Return via Chain.return_() which raises QuentException."""
    with self.assertRaises(QuentException):
      for _ in Chain(SyncIterator, [1, 2]).iterate(Chain.return_):
        pass

  async def test_sync_generator_empty_iterable(self):
    """Sync generator over empty iterable yields nothing."""
    r = []
    for i in Chain(SyncIterator, []).iterate(lambda i: i * 2):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [])

  async def test_sync_generator_with_chain_body(self):
    """iterate with a Chain as the body fn processes each element through the chain."""
    body = Chain().then(lambda v: v * 2).then(lambda v: v + 1)
    r = []
    for i in Chain(SyncIterator, [1, 2, 3]).iterate(body):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [3, 5, 7])


# ---------------------------------------------------------------------------
# AsyncGeneratorTests
# ---------------------------------------------------------------------------
class AsyncGeneratorTests(MyTestCase):

  async def test_async_generator_with_async_iterable(self):
    """async for over an async iterable input."""
    r = []
    async for i in Chain(AsyncIterator, [10, 20, 30]).iterate(lambda i: i + 1):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [11, 21, 31])

  async def test_async_generator_sync_iterable_with_async_fn(self):
    """Sync iterable with a fn that returns coroutines -- awaited inside async generator."""
    r = []
    async for i in Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: aempty(i * 100)):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [100, 200, 300])

  async def test_async_generator_async_iterable_with_async_fn(self):
    """Async iterable + async fn: full async path."""
    r = []
    async for i in Chain(AsyncIterator, [5, 6]).iterate(lambda i: aempty(i ** 2)):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [25, 36])

  async def test_async_generator_break_stops(self):
    """Break inside async generator stops iteration."""
    def f(i):
      if i >= 2:
        return Chain.break_()
      return i * 10

    r = []
    async for i in Chain(AsyncIterator, list(range(5))).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [0, 10])

  async def test_async_generator_exception_propagates(self):
    """Exception in fn propagates out of async generator."""
    def f(i):
      if i == 2:
        raise TestExc('boom')
      return i

    with self.assertRaises(TestExc):
      async for _ in Chain(AsyncIterator, [1, 2, 3]).iterate(f):
        pass


# ---------------------------------------------------------------------------
# IterateAdvancedTests
# ---------------------------------------------------------------------------
class IterateAdvancedTests(MyTestCase):

  async def test_iterate_over_range(self):
    """iterate over a range object."""
    r = []
    for i in Chain(range, 5).iterate(lambda i: i + 10):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [10, 11, 12, 13, 14])

  async def test_iterate_over_dict(self):
    """iterate over a dict yields keys."""
    r = []
    for i in Chain(dict, [('a', 1), ('b', 2)]).iterate():
      r.append(i)
    super(MyTestCase, self).assertEqual(sorted(r), ['a', 'b'])

  async def test_iterate_over_set(self):
    """iterate over a set yields elements (order not guaranteed)."""
    r = []
    for i in Chain(lambda: {10, 20, 30}).iterate():
      r.append(i)
    super(MyTestCase, self).assertEqual(sorted(r), [10, 20, 30])

  async def test_iterate_with_chain_as_body(self):
    """iterate with a Chain object as the body fn."""
    body = Chain().then(lambda v: v ** 2)
    r = []
    for i in Chain(SyncIterator, [2, 3, 4]).iterate(body):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [4, 9, 16])
