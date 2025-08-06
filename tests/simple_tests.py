from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_
from quent import Chain, Cascade, QuentException, run


class SimpleTestCase(IsolatedAsyncioTestCase):
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


class _Obj:
  """Helper class with properties and methods for attribute tests."""
  def __init__(self, val=10):
    self.val = val

  @property
  def value(self):
    return self.val

  def double(self):
    return self.val * 2

  def add(self, n):
    return self.val + n


class SimpleChainTests(SimpleTestCase):

  async def test_simple_flag_stays_true(self):
    # .then()
    c = Chain(1).then(lambda v: v)
    await self.assertTrue(c._is_simple)

    # .attr()
    c = Chain(_Obj).attr('value')
    await self.assertTrue(c._is_simple)

    # .attr_fn()
    c = Chain(_Obj).attr_fn('double')
    await self.assertTrue(c._is_simple)

    # .or_()
    c = Chain(None).or_(42)
    await self.assertTrue(c._is_simple)

    # .raise_()
    c = Chain(1).raise_(ValueError('x'))
    await self.assertTrue(c._is_simple)

    # pipe |
    c = Chain(1) | (lambda v: v * 2)
    await self.assertTrue(c._is_simple)

    # .while_true()
    c = Chain(1).while_true(lambda v: Chain.break_())
    await self.assertTrue(c._is_simple)

    # .foreach()
    c = Chain([1, 2, 3]).foreach(lambda v: v * 2)
    await self.assertTrue(c._is_simple)

    # .foreach_do()
    c = Chain([1, 2, 3]).foreach_do(lambda v: v * 2)
    await self.assertTrue(c._is_simple)

    # .with_()
    c = Chain(1).with_(lambda v: v)
    await self.assertTrue(c._is_simple)

  async def test_simple_flag_becomes_false(self):
    # .do()
    c = Chain(1).do(lambda v: None)
    await self.assertFalse(c._is_simple)

    # .root()
    c = Chain(1).root(lambda v: v)
    await self.assertFalse(c._is_simple)

    # .root_do()
    c = Chain(1).root_do(lambda v: None)
    await self.assertFalse(c._is_simple)

    # .except_()
    c = Chain(1).except_(lambda v: None)
    await self.assertFalse(c._is_simple)

    # .sleep()
    c = Chain(1).sleep(0)
    await self.assertFalse(c._is_simple)

    # .with_do()
    c = Chain(1).with_do(lambda v: v)
    await self.assertFalse(c._is_simple)

  async def test_simple_then_sync(self):
    # literal value
    await self.assertEqual(Chain(1).then(10).run(), 10)

    # callable with args
    await self.assertEqual(Chain(lambda a, b: a + b, 3, 4).then(lambda v: v * 2).run(), 14)

    # Ellipsis = ignore current value
    await self.assertEqual(Chain(1).then(lambda: 99, ...).run(), 99)

    # multiple thens
    await self.assertEqual(
      Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).then(lambda v: v - 2).run(),
      4  # (1+1)*3-2 = 4
    )

  async def test_simple_then_async(self):
    # async root
    await self.assertEqual(Chain(aempty, 42).then(lambda v: v + 1).run(), 43)

    # sync root, async then
    await self.assertEqual(Chain(1).then(aempty).then(lambda v: v + 5).run(), 6)

    # all async
    await self.assertEqual(
      Chain(aempty, 10).then(aempty).then(lambda v: v * 2).run(),
      20
    )

  async def test_simple_cascade_sync(self):
    c = Cascade(10).then(lambda v: v * 2).then(lambda v: v + 5)
    await self.assertTrue(c._is_simple)
    await self.assertEqual(c.run(), 10)

  async def test_simple_cascade_async(self):
    c = Cascade(aempty, 10).then(lambda v: v * 2).then(lambda v: v + 5)
    await self.assertTrue(c._is_simple)
    await self.assertEqual(c.run(), 10)

  async def test_simple_attr(self):
    # .attr() for property access
    await self.assertEqual(Chain(_Obj, 25).attr('value').run(), 25)

    # .attr_fn() for method call
    await self.assertEqual(Chain(_Obj, 25).attr_fn('double').run(), 50)

    # .attr_fn() with argument
    await self.assertEqual(Chain(_Obj, 25).attr_fn('add', 5).run(), 30)

    # chained attr then then
    await self.assertEqual(
      Chain(_Obj, 7).attr('value').then(lambda v: v * 3).run(),
      21
    )

    # async root with attr
    await self.assertEqual(Chain(aempty, _Obj(15)).attr('value').run(), 15)

  async def test_simple_or(self):
    # falsy value -> fallback
    await self.assertEqual(Chain(None).or_(42).run(), 42)
    await self.assertEqual(Chain(0).or_(42).run(), 42)
    await self.assertEqual(Chain('').or_('default').run(), 'default')

    # truthy value -> keep
    await self.assertEqual(Chain(10).or_(42).run(), 10)
    await self.assertEqual(Chain('hello').or_('default').run(), 'hello')

    # async then or
    await self.assertEqual(Chain(aempty).then(aempty, None).or_(42).run(), 42)

    # async truthy
    await self.assertEqual(Chain(aempty, 10).or_(42).run(), 10)

  async def test_simple_raise(self):
    # sync raise
    with self.assertRaises(ValueError):
      Chain(1).raise_(ValueError('x')).run()

    # async raise
    with self.assertRaises(ValueError):
      await await_(Chain(aempty, 1).raise_(ValueError('x')).run())

  async def test_simple_pipe(self):
    # basic pipe
    await self.assertEqual(Chain(1) | (lambda v: v * 10) | (lambda v: v + 5) | run(), 15)

    # void chain with pipe and root override
    await self.assertEqual(Chain() | (lambda v: v * 2) | run(5), 10)

    # pipe preserves _is_simple
    c = Chain(1) | (lambda v: v * 10)
    await self.assertTrue(c._is_simple)

  async def test_simple_freeze(self):
    # sync frozen chain is reusable
    f = Chain(1).then(lambda v: v * 10).freeze()
    await self.assertEqual(f.run(), 10)
    await self.assertEqual(f.run(), 10)

    # async frozen chain is reusable
    f = Chain(aempty, 1).then(lambda v: v * 10).freeze()
    await self.assertEqual(f.run(), 10)
    await self.assertEqual(f.run(), 10)

    # frozen chain with __call__
    f = Chain(1).then(lambda v: v + 5).freeze()
    await self.assertEqual(f(), 6)
    await self.assertEqual(f(), 6)

  async def test_simple_clone(self):
    # clone preserves _is_simple = True
    c = Chain(1).then(lambda v: v * 10)
    c2 = c.clone()
    await self.assertTrue(c._is_simple)
    await self.assertTrue(c2._is_simple)
    await self.assertEqual(c.run(), 10)
    await self.assertEqual(c2.run(), 10)

    # clone preserves _is_simple = False
    c = Chain(1).do(lambda v: None)
    c2 = c.clone()
    await self.assertFalse(c._is_simple)
    await self.assertFalse(c2._is_simple)

    # cloned chains are independent
    c = Chain(1).then(lambda v: v * 10)
    c2 = c.clone()
    c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 11)
    await self.assertEqual(c2.run(), 10)

  async def test_simple_void_chain(self):
    # void chain returns None
    await self.assertIsNone(Chain().run())

    # void chain with Ellipsis (ignore current value)
    await self.assertEqual(Chain().then(lambda: 42, ...).run(), 42)

    # void chain with root override
    await self.assertEqual(Chain().run(lambda v: v * 2, 5), 10)

  async def test_simple_root_value_override(self):
    # override root on void chain
    await self.assertEqual(Chain().then(lambda v: v * 2).run(5), 10)

    # error on double-root
    with self.assertRaises(QuentException):
      Chain(1).run(2)

  async def test_simple_nested_chains(self):
    # basic nesting
    await self.assertEqual(
      Chain(1).then(Chain().then(lambda v: v * 10)).run(),
      10
    )

    # deep nesting
    await self.assertEqual(
      Chain(1).then(Chain().then(Chain().then(lambda v: v * 10))).run(),
      10
    )

    # nested with async
    await self.assertEqual(
      Chain(1).then(Chain().then(aempty).then(lambda v: v * 10)).run(),
      10
    )

    # nested chains are simple
    outer = Chain(1).then(Chain().then(lambda v: v * 10))
    await self.assertTrue(outer._is_simple)

  async def test_simple_mixed_sync_async(self):
    # sync root, async mid-chain, sync tail
    await self.assertEqual(
      Chain(1).then(lambda v: v * 10).then(aempty).then(lambda v: v + 5).run(),
      15
    )

    # async root, sync continuation
    await self.assertEqual(
      Chain(aempty, 1).then(lambda v: v * 10).run(),
      10
    )

    # multiple async transitions
    await self.assertEqual(
      Chain(aempty, 2).then(lambda v: v * 3).then(aempty).then(lambda v: v + 1).run(),
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
    await self.assertEqual(
      Chain(Chain().then(Chain.return_, 42)).then(lambda v: v * 100).run(),
      42
    )

    # async return in nested also exits the outer chain early
    await self.assertEqual(
      Chain(Chain(aempty).then(Chain.return_, 42)).then(lambda v: v * 100).run(),
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
    for fn, ctx in self.with_fn():
      with ctx:
        # basic chain with fn
        await self.assertEqual(
          Chain(fn, 1).then(fn).then(lambda v: v * 10).run(),
          10
        )

        # chain with .or_()
        await self.assertEqual(
          Chain(fn, None).then(fn).or_(42).run(),
          42
        )

        # chain with attr
        await self.assertEqual(
          Chain(fn, _Obj(5)).then(fn).attr('value').run(),
          5
        )

        # nested chains
        await self.assertEqual(
          Chain(fn, 3).then(Chain().then(fn).then(lambda v: v * 10)).run(),
          30
        )

  async def test_simple_cascade_with_fn_pattern(self):
    for fn, ctx in self.with_fn():
      with ctx:
        # cascade always returns root value
        await self.assertEqual(
          Cascade(fn, 42).then(fn).then(lambda v: v * 2).run(),
          42
        )

        # cascade with multiple ops
        await self.assertEqual(
          Cascade(fn, 99).then(lambda v: v + 1).then(lambda v: v * 100).run(),
          99
        )


class SetAsyncFalseTests(SimpleTestCase):

  async def test_sync_flag_default(self):
    await self.assertFalse(Chain()._is_sync)

  async def test_set_async_false_sets_flag(self):
    await self.assertTrue(Chain().set_async(False)._is_sync)

  async def test_config_async_false_sets_flag(self):
    await self.assertTrue(Chain().config(async_=False)._is_sync)

  async def test_sync_clone_preserves_flag(self):
    c = Chain(1).set_async(False)
    c2 = c.clone()
    await self.assertTrue(c._is_sync)
    await self.assertTrue(c2._is_sync)

  async def test_sync_simple_chain(self):
    # tightest fast path: _is_simple=True and _is_sync=True
    c = Chain(1).then(lambda v: v * 10).then(lambda v: v + 5).set_async(False)
    await self.assertTrue(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 15)

  async def test_sync_medium_chain(self):
    # 5-operation chain
    result = (
      Chain(2)
      .then(lambda v: v + 3)   # 5
      .then(lambda v: v * 2)   # 10
      .then(lambda v: v - 1)   # 9
      .then(lambda v: v ** 2)  # 81
      .then(lambda v: v + 19)  # 100
      .set_async(False)
      .run()
    )
    await self.assertEqual(result, 100)

  async def test_sync_cascade(self):
    c = Cascade(10).then(lambda v: v * 2).then(lambda v: v + 5).set_async(False)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 10)

  async def test_sync_attr(self):
    # .attr() with set_async(False)
    await self.assertEqual(
      Chain(_Obj, 25).attr('value').set_async(False).run(),
      25
    )

    # .attr_fn() with set_async(False)
    await self.assertEqual(
      Chain(_Obj, 25).attr_fn('double').set_async(False).run(),
      50
    )

    # .attr_fn() with arg and set_async(False)
    await self.assertEqual(
      Chain(_Obj, 25).attr_fn('add', 10).set_async(False).run(),
      35
    )

  async def test_sync_or_and_raise(self):
    # .or_() with set_async(False)
    await self.assertEqual(
      Chain(None).or_(42).set_async(False).run(),
      42
    )
    await self.assertEqual(
      Chain(10).or_(42).set_async(False).run(),
      10
    )

    # .raise_() with set_async(False)
    with self.assertRaises(ValueError):
      Chain(1).raise_(ValueError('x')).set_async(False).run()

  async def test_sync_pipe(self):
    # pipe with set_async(False)
    c = Chain(1).set_async(False)
    result = c | (lambda v: v * 10) | (lambda v: v + 5) | run()
    await self.assertEqual(result, 15)

  async def test_sync_freeze(self):
    # frozen chain with set_async(False) is reusable
    f = Chain(1).then(lambda v: v * 10).set_async(False).freeze()
    await self.assertEqual(f.run(), 10)
    await self.assertEqual(f.run(), 10)
    await self.assertEqual(f(), 10)

  async def test_sync_void_chain(self):
    # void chain with set_async(False)
    await self.assertIsNone(Chain().set_async(False).run())

    # void chain with Ellipsis
    await self.assertEqual(
      Chain().then(lambda: 42, ...).set_async(False).run(),
      42
    )

  async def test_sync_root_override(self):
    # root override with set_async(False)
    await self.assertEqual(
      Chain().then(lambda v: v * 2).set_async(False).run(5),
      10
    )

    # error on double-root
    with self.assertRaises(QuentException):
      Chain(1).set_async(False).run(2)

  async def test_sync_nested(self):
    # outer set_async(False), inner default
    await self.assertEqual(
      Chain(1).then(Chain().then(lambda v: v * 10)).set_async(False).run(),
      10
    )

    # both set_async(False)
    await self.assertEqual(
      Chain(1).then(
        Chain().then(lambda v: v * 10).set_async(False)
      ).set_async(False).run(),
      10
    )

    # deep nesting with set_async(False)
    await self.assertEqual(
      Chain(1).then(
        Chain().then(
          Chain().then(lambda v: v * 10).set_async(False)
        ).set_async(False)
      ).set_async(False).run(),
      10
    )

  async def test_sync_error_propagation(self):
    # sync error with set_async(False)
    with self.assertRaises(ZeroDivisionError):
      Chain(1).then(lambda v: 1 / 0).set_async(False).run()

    # error in root with set_async(False)
    with self.assertRaises(TypeError):
      Chain(lambda: None + 1).set_async(False).run()

  async def test_sync_combined_with_simple(self):
    # verify both flags are set
    c = Chain(1).then(lambda v: v * 10).set_async(False)
    await self.assertTrue(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 10)

    # longer chain, still simple and sync
    c = (
      Chain(5)
      .then(lambda v: v + 5)
      .then(lambda v: v * 2)
      .then(lambda v: v - 1)
      .set_async(False)
    )
    await self.assertTrue(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 19)

    # with .or_() still simple
    c = Chain(None).or_(42).set_async(False)
    await self.assertTrue(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 42)

    # with .attr() still simple
    c = Chain(_Obj, 7).attr('value').set_async(False)
    await self.assertTrue(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 7)

    # cascade combined
    c = Cascade(10).then(lambda v: v * 2).set_async(False)
    await self.assertTrue(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 10)

  async def test_sync_nonsimple_chain(self):
    # .do() breaks _is_simple but _is_sync still applies
    c = Chain(1).do(lambda v: None).then(lambda v: v * 10).set_async(False)
    await self.assertFalse(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 10)

    # .except_() breaks _is_simple
    c = Chain(1).except_(lambda v: None).then(lambda v: v * 10).set_async(False)
    await self.assertFalse(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 10)

    # .root() breaks _is_simple
    c = Chain(5).root(lambda v: v * 2).set_async(False)
    await self.assertFalse(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 10)

    # .root_do() breaks _is_simple
    side_effect = []
    c = Chain(5).root_do(lambda v: side_effect.append(v)).then(lambda v: v * 3).set_async(False)
    await self.assertFalse(c._is_simple)
    await self.assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 15)
    await self.assertEqual(side_effect, [5])
