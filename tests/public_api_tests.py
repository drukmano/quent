import asyncio
import inspect
from unittest import IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_, TestExc
from quent import Chain, QuentException


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
class FilterTests(IsolatedAsyncioTestCase):

  async def test_filter_sync(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).run()),
          [2, 4]
        )

  async def test_filter_async_iterable(self):
    result = await await_(
      Chain(AsyncIter([1, 2, 3, 4, 5])).filter(lambda x: x > 3).run()
    )
    self.assertEqual(result, [4, 5])

  async def test_filter_async_predicate(self):
    async def is_even(x):
      return x % 2 == 0
    result = await await_(
      Chain([10, 11, 12, 13, 14]).filter(is_even).run()
    )
    self.assertEqual(result, [10, 12, 14])

  async def test_filter_empty(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, []).filter(lambda x: True).run()),
          []
        )

  async def test_filter_all_pass(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [1, 2, 3]).filter(lambda x: True).run()),
          [1, 2, 3]
        )

  async def test_filter_none_pass(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [1, 2, 3]).filter(lambda x: False).run()),
          []
        )


# ---------------------------------------------------------------------------
# Class 2: GatherTests
# ---------------------------------------------------------------------------
class GatherTests(IsolatedAsyncioTestCase):

  async def test_gather_all_sync(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 5).gather(
            lambda v: v + 1,
            lambda v: v * 2,
            lambda v: v - 3,
          ).run()),
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
    self.assertEqual(result, [6, 10, 2])

  async def test_gather_mixed(self):
    async def async_mul(v):
      return v * 10
    result = await await_(
      Chain(3).gather(
        lambda v: v + 1,
        async_mul,
      ).run()
    )
    self.assertEqual(result, [4, 30])

  async def test_gather_single_fn(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 7).gather(lambda v: v * 3).run()),
          [21]
        )

  async def test_gather_receives_value(self):
    obj_ = object()
    received = []
    def collector(v):
      received.append(v)
      return v
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        received.clear()
        await await_(
          Chain(fn, obj_).gather(collector, collector).run()
        )
        self.assertEqual(len(received), 2)
        self.assertIs(received[0], obj_)
        self.assertIs(received[1], obj_)


# ---------------------------------------------------------------------------
# Class 3: ReprTests
# ---------------------------------------------------------------------------
class ReprTests(IsolatedAsyncioTestCase):

  async def test_repr_empty_chain(self):
    c = Chain()
    r = repr(c)
    self.assertTrue(r.startswith('Chain()'))

  async def test_repr_with_root(self):
    def my_root_fn():
      pass
    c = Chain(my_root_fn)
    r = repr(c)
    self.assertIn('my_root_fn', r)

  async def test_repr_with_operations(self):
    c = Chain(10).then(lambda v: v + 1).then(lambda v: v * 2)
    r = repr(c)
    self.assertIn('.then(', r)

