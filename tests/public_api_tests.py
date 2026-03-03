import asyncio
import inspect
from tests.utils import throw_if, empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run


class AsyncIter:
  def __init__(self, items):
    self.items = items
    self.index = 0

  def __aiter__(self):
    self.index = 0
    return self

  async def __anext__(self):
    if self.index >= len(self.items):
      raise StopAsyncIteration
    val = self.items[self.index]
    self.index += 1
    return val


# ---------------------------------------------------------------------------
# Class 1: FilterTests
# ---------------------------------------------------------------------------
class FilterTests(MyTestCase):

  async def test_filter_sync(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).run(),
          [2, 4]
        )

  async def test_filter_async_iterable(self):
    result = await await_(
      Chain(AsyncIter([1, 2, 3, 4, 5])).filter(lambda x: x > 3).run()
    )
    super(MyTestCase, self).assertEqual(result, [4, 5])

  async def test_filter_async_predicate(self):
    async def is_even(x):
      return x % 2 == 0
    result = await await_(
      Chain([10, 11, 12, 13, 14]).filter(is_even).run()
    )
    super(MyTestCase, self).assertEqual(result, [10, 12, 14])

  async def test_filter_empty(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).filter(lambda x: True).run(),
          []
        )

  async def test_filter_all_pass(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: True).run(),
          [1, 2, 3]
        )

  async def test_filter_none_pass(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: False).run(),
          []
        )


# ---------------------------------------------------------------------------
# Class 2: GatherTests
# ---------------------------------------------------------------------------
class GatherTests(MyTestCase):

  async def test_gather_all_sync(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather(
            lambda v: v + 1,
            lambda v: v * 2,
            lambda v: v - 3,
          ).run(),
          [6, 10, 2]
        )

  async def test_gather_all_async(self):
    async def add1(v):
      return v + 1
    async def mul2(v):
      return v * 2
    async def sub3(v):
      return v - 3
    result = await await_(
      Chain(5).gather(add1, mul2, sub3).run()
    )
    super(MyTestCase, self).assertEqual(result, [6, 10, 2])

  async def test_gather_mixed(self):
    async def async_mul(v):
      return v * 10
    result = await await_(
      Chain(3).gather(
        lambda v: v + 1,
        async_mul,
      ).run()
    )
    super(MyTestCase, self).assertEqual(result, [4, 30])

  async def test_gather_single_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 7).gather(lambda v: v * 3).run(),
          [21]
        )

  async def test_gather_receives_value(self):
    obj_ = object()
    received = []
    def collector(v):
      received.append(v)
      return v
    for fn, ctx in self.with_fn():
      with ctx:
        received.clear()
        await await_(
          Chain(fn, obj_).gather(collector, collector).run()
        )
        super(MyTestCase, self).assertEqual(len(received), 2)
        super(MyTestCase, self).assertIs(received[0], obj_)
        super(MyTestCase, self).assertIs(received[1], obj_)


# ---------------------------------------------------------------------------
# Class 3: ReprTests
# ---------------------------------------------------------------------------
class ReprTests(MyTestCase):

  async def test_repr_empty_chain(self):
    c = Chain()
    r = repr(c)
    super(MyTestCase, self).assertTrue(r.startswith('Chain()'))

  async def test_repr_with_root(self):
    def my_root_fn():
      pass
    c = Chain(my_root_fn)
    r = repr(c)
    super(MyTestCase, self).assertIn('my_root_fn', r)

  async def test_repr_with_operations(self):
    c = Chain(10).then(lambda v: v + 1).then(lambda v: v * 2)
    r = repr(c)
    super(MyTestCase, self).assertIn('.then(', r)

  async def test_repr_cascade(self):
    c = Cascade()
    r = repr(c)
    super(MyTestCase, self).assertTrue(r.startswith('Cascade'))


# ---------------------------------------------------------------------------
# Class 4: ForeachIndexedTests
# ---------------------------------------------------------------------------
class ForeachIndexedTests(MyTestCase):

  async def test_foreach_indexed_sync(self):
    results = []
    def collect(idx, el):
      results.append((idx, el))
      return (idx, el)
    for fn, ctx in self.with_fn():
      with ctx:
        results.clear()
        await await_(
          Chain(fn, [10, 20, 30]).foreach(collect, with_index=True).run()
        )
        super(MyTestCase, self).assertEqual(results, [(0, 10), (1, 20), (2, 30)])

  async def test_foreach_indexed_async(self):
    results = []
    def collect(idx, el):
      results.append((idx, el))
      return (idx, el)
    results.clear()
    result = await await_(
      Chain(AsyncIter([10, 20, 30])).foreach(collect, with_index=True).run()
    )
    super(MyTestCase, self).assertEqual(results, [(0, 10), (1, 20), (2, 30)])

  async def test_foreach_indexed_values_correct(self):
    indices = []
    def track_index(idx, el):
      indices.append(idx)
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        indices.clear()
        await await_(
          Chain(fn, ['a', 'b', 'c', 'd', 'e']).foreach(track_index, with_index=True).run()
        )
        super(MyTestCase, self).assertEqual(indices, [0, 1, 2, 3, 4])

  async def test_foreach_indexed_break(self):
    def stop_at_2(idx, el):
      if idx == 2:
        return Chain.break_()
      return el * 10
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).foreach(stop_at_2, with_index=True).run(),
          [10, 20]
        )

  async def test_foreach_indexed_exception(self):
    def raiser(idx, el):
      if el == 3:
        raise ValueError("bad element")
      return el
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, [1, 2, 3]).foreach(raiser, with_index=True).run()
          )
