import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase
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


# ---------------------------------------------------------------------------
# GatherAsyncExceptionTests
# ---------------------------------------------------------------------------
class GatherAsyncExceptionTests(MyTestCase):

  async def test_single_async_fn_raises(self):
    """One async function raises; exception propagates from gather."""
    async def bad(v):
      raise TestExc('async fail')

    with self.assertRaises(TestExc):
      await await_(Chain(5).gather(bad).run())

  async def test_sync_fn_raises_before_gather_async(self):
    """A sync function raising prevents gather_async from being reached."""
    def bad_sync(v):
      raise TestExc('sync fail')

    with self.assertRaises(TestExc):
      await await_(Chain(5).gather(bad_sync, lambda v: v).run())

  async def test_multiple_async_fns_one_raises(self):
    """Multiple async fns, one raises; exception propagates."""
    async def ok(v):
      return v + 1

    async def bad(v):
      raise TestExc('one bad')

    with self.assertRaises(TestExc):
      await await_(Chain(5).gather(ok, bad, ok).run())

  async def test_async_raise_with_sync_fns_present(self):
    """Mixed sync/async where the async fn raises."""
    def ok_sync(v):
      return v * 2

    async def bad_async(v):
      raise TestExc('async raises')

    with self.assertRaises(TestExc):
      await await_(Chain(5).gather(ok_sync, bad_async).run())

  async def test_sync_raise_among_async_fns(self):
    """Sync fn raises while other fns are async; sync exception takes priority."""
    async def ok_async(v):
      return v

    def bad_sync(v):
      raise TestExc('sync among async')

    with self.assertRaises(TestExc):
      await await_(Chain(5).gather(ok_async, bad_sync).run())

  async def test_exception_type_preserved(self):
    """Verify the exact exception type and message are preserved through gather."""
    async def raiser(v):
      raise ValueError('specific message')

    with self.assertRaises(ValueError) as cm:
      await await_(Chain(10).gather(raiser).run())
    super(MyTestCase, self).assertIn('specific message', str(cm.exception))


# ---------------------------------------------------------------------------
# GatherIndicesTests
# ---------------------------------------------------------------------------
class GatherIndicesTests(MyTestCase):

  async def test_all_sync_ordering(self):
    """All sync fns: results appear in function order."""
    for fn, ctx in self.with_fn():
      with ctx:
        f1 = lambda v: 'a'
        f2 = lambda v: 'b'
        f3 = lambda v: 'c'
        await self.assertEqual(
          Chain(fn, 1).gather(f1, f2, f3).run(),
          ['a', 'b', 'c']
        )

  async def test_all_async_ordering(self):
    """All async fns: results appear in function order despite concurrency."""
    async def f1(v):
      await asyncio.sleep(0.02)
      return 'slow'

    async def f2(v):
      return 'fast'

    async def f3(v):
      await asyncio.sleep(0.01)
      return 'medium'

    await self.assertEqual(
      Chain(1).gather(f1, f2, f3).run(),
      ['slow', 'fast', 'medium']
    )

  async def test_interleaved_sync_async_ordering(self):
    """Alternating sync/async fns: index mapping preserves order."""
    def s1(v):
      return 1

    async def a2(v):
      return 2

    def s3(v):
      return 3

    async def a4(v):
      return 4

    def s5(v):
      return 5

    await self.assertEqual(
      Chain(0).gather(s1, a2, s3, a4, s5).run(),
      [1, 2, 3, 4, 5]
    )

  async def test_async_at_start_sync_at_end(self):
    """Async fns first, sync fns last: index reassignment correct."""
    async def a1(v):
      return 'async1'

    async def a2(v):
      return 'async2'

    def s3(v):
      return 'sync3'

    def s4(v):
      return 'sync4'

    await self.assertEqual(
      Chain(0).gather(a1, a2, s3, s4).run(),
      ['async1', 'async2', 'sync3', 'sync4']
    )

  async def test_sync_at_start_async_at_end(self):
    """Sync fns first, async fns last: index reassignment correct."""
    def s1(v):
      return 'sync1'

    def s2(v):
      return 'sync2'

    async def a3(v):
      return 'async3'

    async def a4(v):
      return 'async4'

    await self.assertEqual(
      Chain(0).gather(s1, s2, a3, a4).run(),
      ['sync1', 'sync2', 'async3', 'async4']
    )

  async def test_single_async_among_many_sync(self):
    """One async fn among many sync fns triggers gather_async; indices correct."""
    def s(v):
      return v

    async def a(v):
      return v * 10

    await self.assertEqual(
      Chain(1).gather(s, s, a, s, s).run(),
      [1, 1, 10, 1, 1]
    )

  async def test_gather_value_passed_to_all_fns(self):
    """All fns receive the same current chain value."""
    received = []

    def capture(v):
      received.append(v)
      return v

    async def async_capture(v):
      received.append(v)
      return v

    received.clear()
    await self.assertEqual(
      Chain(99).gather(capture, async_capture, capture).run(),
      [99, 99, 99]
    )
    super(MyTestCase, self).assertEqual(received, [99, 99, 99])


# ---------------------------------------------------------------------------
# GatherScaleTests
# ---------------------------------------------------------------------------
class GatherScaleTests(MyTestCase):

  async def test_wide_gather_50_sync(self):
    """Gather with 50 sync functions produces correct ordered results."""
    fns = [lambda v, i=i: v + i for i in range(50)]
    await self.assertEqual(
      Chain(0).gather(*fns).run(),
      list(range(50))
    )

  async def test_wide_gather_50_async(self):
    """Gather with 50 async functions produces correct ordered results."""
    fns = []
    for i in range(50):
      async def fn(v, i=i):
        return v + i
      fns.append(fn)

    await self.assertEqual(
      Chain(0).gather(*fns).run(),
      list(range(50))
    )

  async def test_wide_gather_100_mixed(self):
    """Gather with 100 mixed sync/async fns: even indices sync, odd async."""
    fns = []
    for i in range(100):
      if i % 2 == 0:
        fns.append(lambda v, i=i: i)
      else:
        async def fn(v, i=i):
          return i
        fns.append(fn)

    await self.assertEqual(
      Chain(0).gather(*fns).run(),
      list(range(100))
    )

  async def test_gather_empty_fn_list(self):
    """Chain(5).gather().run() returns empty list."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).gather().run(),
          []
        )

  async def test_gather_chained_with_then(self):
    """Gather result piped to further .then() operations."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 10).gather(
            lambda v: v + 1,
            lambda v: v + 2,
            lambda v: v + 3
          ).then(lambda lst: sum(lst)).run(),
          36
        )

  async def test_gather_async_chained_with_then(self):
    """Gather with async fns piped to further .then() operations."""
    await self.assertEqual(
      Chain(10).gather(
        lambda v: aempty(v + 1),
        lambda v: aempty(v + 2),
        lambda v: aempty(v + 3)
      ).then(lambda lst: sum(lst)).run(),
      36
    )
