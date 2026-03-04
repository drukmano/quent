from unittest import IsolatedAsyncioTestCase
from itertools import product
from tests.utils import empty, aempty, await_
from quent import Chain, QuentException


class SimpleChainTests(IsolatedAsyncioTestCase):

  async def test_simple_then_sync(self):
    # literal value
    self.assertEqual(Chain(1).then(10).run(), 10)

    # callable with args
    self.assertEqual(Chain(lambda a, b: a + b, 3, 4).then(lambda v: v * 2).run(), 14)

    # Ellipsis = ignore current value
    self.assertEqual(Chain(1).then(lambda: 99, ...).run(), 99)

    # multiple thens
    self.assertEqual(
      Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).then(lambda v: v - 2).run(),
      4  # (1+1)*3-2 = 4
    )

  async def test_simple_then_async(self):
    # async root
    self.assertEqual(await Chain(aempty, 42).then(lambda v: v + 1).run(), 43)

    # sync root, async then
    self.assertEqual(await Chain(1).then(aempty).then(lambda v: v + 5).run(), 6)

    # all async
    self.assertEqual(
      await Chain(aempty, 10).then(aempty).then(lambda v: v * 2).run(),
      20
    )

  async def test_simple_chain_reuse(self):
    # sync chain is reusable
    c = Chain(1).then(lambda v: v * 10)
    self.assertEqual(c.run(), 10)
    self.assertEqual(c.run(), 10)

    # async chain is reusable
    c = Chain(aempty, 1).then(lambda v: v * 10)
    self.assertEqual(await c.run(), 10)
    self.assertEqual(await c.run(), 10)

    # chain with __call__
    c = Chain(1).then(lambda v: v + 5)
    self.assertEqual(c(), 6)
    self.assertEqual(c(), 6)

  async def test_simple_clone(self):
    # clone preserves chain
    c = Chain(1).then(lambda v: v * 10)
    c2 = c.clone()
    self.assertEqual(c.run(), 10)
    self.assertEqual(c2.run(), 10)

    # cloned chains are independent
    c = Chain(1).then(lambda v: v * 10)
    c2 = c.clone()
    c.then(lambda v: v + 1)
    self.assertEqual(c.run(), 11)
    self.assertEqual(c2.run(), 10)

  async def test_simple_void_chain(self):
    # void chain returns None
    self.assertIsNone(Chain().run())

    # void chain with Ellipsis (ignore current value)
    self.assertEqual(Chain().then(lambda: 42, ...).run(), 42)

    # void chain with root override
    self.assertEqual(Chain().run(lambda v: v * 2, 5), 10)

  async def test_simple_root_value_override(self):
    # override root on void chain
    self.assertEqual(Chain().then(lambda v: v * 2).run(5), 10)

    # error on double-root
    with self.assertRaises(QuentException):
      Chain(1).run(2)

  async def test_simple_nested_chains(self):
    # basic nesting
    self.assertEqual(
      Chain(1).then(Chain().then(lambda v: v * 10)).run(),
      10
    )

    # deep nesting
    self.assertEqual(
      Chain(1).then(Chain().then(Chain().then(lambda v: v * 10))).run(),
      10
    )

    # nested with async
    self.assertEqual(
      await Chain(1).then(Chain().then(aempty).then(lambda v: v * 10)).run(),
      10
    )

  async def test_simple_mixed_sync_async(self):
    # sync root, async mid-chain, sync tail
    self.assertEqual(
      await Chain(1).then(lambda v: v * 10).then(aempty).then(lambda v: v + 5).run(),
      15
    )

    # async root, sync continuation
    self.assertEqual(
      await Chain(aempty, 1).then(lambda v: v * 10).run(),
      10
    )

    # multiple async transitions
    self.assertEqual(
      await Chain(aempty, 2).then(lambda v: v * 3).then(aempty).then(lambda v: v + 1).run(),
      7
    )

  async def test_simple_error_propagation(self):
    # sync error bubbles up
    with self.assertRaises(ZeroDivisionError):
      Chain(1).then(lambda v: 1 / 0).run()

    # async error bubbles up
    with self.assertRaises(ZeroDivisionError):
      await await_(Chain(aempty, 1).then(lambda v: 1 / 0).run())

    # error in root
    with self.assertRaises(TypeError):
      Chain(lambda: None + 1).run()

  async def test_simple_return_in_nested(self):
    # Chain.return_ exits the entire outer chain early with the return value
    self.assertEqual(
      Chain(Chain().then(Chain.return_, 42)).then(lambda v: v * 100).run(),
      42
    )

    # async return in nested also exits the outer chain early
    self.assertEqual(
      await Chain(Chain(aempty).then(Chain.return_, 42)).then(lambda v: v * 100).run(),
      42
    )

  async def test_simple_break_raises(self):
    # break outside loop raises QuentException (sync)
    with self.assertRaises(QuentException):
      Chain().then(Chain.break_).run()

    # break outside loop raises QuentException (async)
    with self.assertRaises(QuentException):
      await await_(Chain(aempty).then(Chain.break_).run())

  async def test_simple_with_fn_pattern(self):
    for fn1, fn2, fn3, fn4, fn5, fn6 in product([empty, aempty], repeat=6):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4, fn5=fn5, fn6=fn6):
        # basic chain with fn
        self.assertEqual(
          await await_(Chain(fn1, 1).then(fn2).then(lambda v: v * 10).run()),
          10
        )

        # chain with then
        self.assertEqual(
          await await_(Chain(fn3, 5).then(fn4).then(lambda v: v * 2).run()),
          10
        )

        # nested chains
        self.assertEqual(
          await await_(Chain(fn5, 3).then(Chain().then(fn6).then(lambda v: v * 10)).run()),
          30
        )
