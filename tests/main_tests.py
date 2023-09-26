import asyncio
from contextlib import contextmanager, asynccontextmanager
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_
from tests.try_except_tests import assertRaisesSync, assertRaisesAsync
from src.quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException, run


class MainTest(IsolatedAsyncioTestCase):
  async def test_root_is_literal_value(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(await await_(Chain(1).run()), 1)
        self.assertEqual(await await_(Chain(1).then(2).run()), 2)
        self.assertEqual(await await_(Chain(1).then(fn).run()), 1)
        self.assertEqual(await await_(Chain(1).then(2).then(fn).run()), 2)
        self.assertEqual(await await_(Chain(1).then(fn).then(2).run()), 2)
        self.assertTrue(await assertRaisesAsync(Exception, lambda: await_(Chain(1).then(throw_if).run())))
        self.assertTrue(await assertRaisesAsync(Exception, lambda: await_(Chain(1).then(fn).then(throw_if).run())))
        self.assertEqual(await await_(Chain(0).then(throw_if).then(fn).then(1).run()), 1)
        self.assertEqual(await await_(Chain(0).then(fn).then(throw_if).then(1).run()), 1)

  async def test_root_is_callable(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(await await_(Chain(fn).run()), None)
        self.assertEqual(await await_(Chain(fn).then(1).run()), 1)
        self.assertEqual(await await_(Chain(fn).then(2).run()), 2)
        self.assertEqual(await await_(Chain(fn).then(1).then(fn).run()), 1)
        self.assertEqual(await await_(Chain(fn).then(fn).then(1).run()), 1)
        self.assertTrue(await assertRaisesAsync(Exception, lambda: await_(Chain(fn).then(1).then(throw_if).run())))
        self.assertTrue(await assertRaisesAsync(Exception, lambda: await_(Chain(fn).then(1).then(throw_if).run())))
        self.assertEqual(await await_(Chain(fn).then(throw_if).then(fn).then(1).run()), 1)
        self.assertEqual(await await_(Chain(fn).then(fn).then(throw_if).then(1).run()), 1)

  async def test_pipe(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_((Chain(1) | (lambda v: v+1) | fn | (lambda v: v+5)).eq(7).run()))
        self.assertTrue(await await_(Chain(1) | (lambda v: v+1) | fn | (lambda v: v+5) | (lambda v: v == 7) | run()))

  async def test_void(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(10).then(fn).ignore(lambda v: v/10).eq(10).run()))
        self.assertTrue(await await_(Chain(10).then(fn).then(lambda v: v/10).eq(1).run()))
        self.assertTrue(await await_(Chain(10).ignore(fn, 100).eq(10).run()))
        self.assertTrue(await await_(Chain(10).then(fn, 100).eq(100).run()))

  async def test_empty_root(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain().then(1).then(fn).root().then(lambda v: v+10).eq(10).run(0)))

  async def test_root_value_async(self):
    self.assertTrue(await Chain(aempty, 5).eq(5).run())
    self.assertTrue(await Chain().eq(5).run(aempty, 5))

  async def test_run_without_root(self):
    self.assertRaises(QuentException, Chain().then(empty).run)
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertIsNone(await await_(Cascade().then(fn).run()))
        self.assertIsNone(await await_(Cascade().then(fn, 1).run()))

  async def test_override_empty_value(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain().then(fn).eq(1).run(1)))

  async def test_override_root_value(self):
      self.assertRaises(QuentException, Chain(1).then(empty).run, 1)

  async def test_misc_1(self):
    self.assertIsNone(Cascade().root().run())
    self.assertIsNone(Cascade().then(lambda: 5).then(lambda: 6).run())
    self.assertIn('6 links', str(Chain(5).then(lambda v: v).eq(5).else_(10).in_([10]).else_(False).not_()))

  async def test_foreach(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def f(i):
          nonlocal counter
          counter += i**2

        counter = 0
        await await_(Chain(range(10)).foreach(lambda v: fn(f(v))).run())
        self.assertEqual(counter, sum(i**2 for i in range(10)))

        counter = 0
        await await_(Chain(range(10)).then(fn).foreach(lambda v: fn(f(v))).run())
        self.assertEqual(counter, sum(i**2 for i in range(10)))

  async def test_foreach_async_gen(self):
    async def gen():
      yield 42

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def f(n):
          nonlocal num
          num = n

        num = 0
        await await_(Chain(gen).foreach(f).run())
        self.assertEqual(num, 42)

        num = 0
        await await_(Chain(gen).then(fn).foreach(f).run())
        self.assertEqual(num, 42)

  async def test_foreach_async_mid_loop(self):
      def f(i):
        nonlocal counter
        counter += i**2
        if 4 <= i <= 7:
          return aempty()

      counter = 0
      await await_(Chain(range(10)).foreach(f).run())
      self.assertEqual(counter, sum(i**2 for i in range(10)))
      coro = Chain(range(10)).foreach(f).run()
      self.assertTrue(asyncio.iscoroutine(coro))
      await coro

  async def test_with(self):
    @contextmanager
    def sync_ctx(v: int):
      yield v**2

    @asynccontextmanager
    async def async_ctx(v: int):
      yield v**2

    for ctx in [sync_ctx, async_ctx]:
      for fn in [empty, aempty]:
        with self.subTest(ctx=ctx, fn=fn):
          self.assertTrue(await await_(Chain(ctx, 10).then(fn).neq(100).run()))
          self.assertTrue(await await_(Chain(ctx, 10).then(fn).with_(...).eq(100).run()))
          self.assertTrue(await await_(Chain(ctx, 10).then(fn).with_(..., lambda v: v/10).eq(10).run()))
          self.assertTrue(await await_(Chain(None).then(fn).with_(ctx(10)).eq(100).run()))
          self.assertTrue(await await_(Chain(None).then(fn).with_(ctx(10), lambda v: v/10).eq(10).run()))

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(async_ctx, 10).then(fn).async_with(...).eq(100).run()))
        self.assertTrue(await await_(Chain(async_ctx, 10).then(fn).async_with(..., lambda v: v/10).eq(10).run()))
        self.assertTrue(await await_(Chain(None).then(fn).async_with(async_ctx(10)).eq(100).run()))
        self.assertTrue(await await_(Chain(None).then(fn).async_with(async_ctx(10), lambda v: v/10).eq(10).run()))
