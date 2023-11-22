import asyncio
import time
import asyncio
import contextlib
from tests.flex_context import FlexContext
import inspect
import time
from contextlib import contextmanager, asynccontextmanager
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_, DummySync, DummyAsync
from quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, DummySync, DummyAsync
from quent import Chain, ChainAttr, Cascade, CascadeAttr, run
from quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException, run


class ExceptFinallyCheckSync:
  def __init__(self):
    self.ran_exc = False
    self.ran_finally = False

  def on_except(self, v=None, *, register=True):
    self.ran_exc = register

  def on_finally(self, v=None, *, register=True):
    self.ran_finally = register


class ExceptFinallyCheckAsync:
  def __init__(self):
    self.ran_exc = False
    self.ran_finally = False

  async def on_except(self, v=None, *, register=True):
    self.ran_exc = register

  async def on_finally(self, v=None, *, register=True):
    self.ran_finally = register


def assertRaisesSync(exc, fn, *args, **kwargs) -> bool:
  exc_fin_check = ExceptFinallyCheckSync()
  try:
    fn(*args, **kwargs)
  except *exc:
    exc_fin_check.on_except()
  return exc_fin_check.ran_exc


async def assertRaisesAsync(exc, fn, *args, **kwargs) -> bool:
  exc_fin_check = ExceptFinallyCheckAsync()
  try:
    await fn(*args, **kwargs)
  except *exc:
    await exc_fin_check.on_except()
  return exc_fin_check.ran_exc


def get_empty_and_cls():
  yield from iter([(empty, ExceptFinallyCheckSync), (aempty, ExceptFinallyCheckAsync)])


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
        self.assertTrue(await await_(Chain(10).then(fn).do(lambda v: v/10).eq(10).run()))
        self.assertTrue(await await_(Chain(10).then(fn).then(lambda v: v/10).eq(1).run()))
        self.assertTrue(await await_(Chain(10).do(fn, 100).eq(10).run()))
        self.assertTrue(await await_(Chain(10).then(fn, 100).eq(100).run()))
        self.assertEqual(await await_(Chain(10).then(fn, 100).root_do(lambda v: v/10).run()), 100)

  async def test_empty_root(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain().then(1).then(fn).root(lambda r: r).then(lambda v: v+10).eq(10).run(0)))

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
    self.assertIsNone(Cascade().root(lambda: None).run())
    self.assertIsNone(Cascade().then(lambda: 5).then(lambda: 6).run())
    self.assertIn('3 links', str(Chain(5).then(lambda v: v).eq(5).if_not(10).in_([10]).if_(True).else_(False).not_()))

  async def test_foreach(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def f(i):
          nonlocal counter
          counter += i**2
          return i
        def f_mod(i):
          nonlocal counter
          counter += i**2
          return i**2

        counter = 0
        self.assertTrue(await await_(Chain(range(10)).foreach(lambda v: fn(f(v))).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))

        counter = 0
        self.assertTrue(await await_(Chain(range(10)).then(fn).foreach(lambda v: fn(f(v))).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))

        counter = 0
        self.assertTrue(await await_(Chain(range(10)).foreach_do(lambda v: fn(f_mod(v))).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))

        counter = 0
        self.assertTrue(await await_(Chain(range(10)).then(fn).foreach_do(lambda v: fn(f_mod(v))).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))

  async def test_foreach_async_gen(self):
    async def gen():
      yield 42

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def f(n):
          nonlocal num
          num = n
          return fn(n)

        num = 0
        self.assertTrue(await await_(Chain(gen).foreach(f).eq([42]).run()))
        self.assertEqual(num, 42)

        num = 0
        self.assertTrue(await await_(Chain(gen).then(fn).foreach(f).eq([42]).run()))
        self.assertEqual(num, 42)

        num = 0
        self.assertTrue(await await_(Chain(gen).foreach_do(f).eq([42]).run()))
        self.assertEqual(num, 42)

        num = 0
        await await_(Chain(gen).then(fn).foreach_do(f).run())
        self.assertEqual(num, 42)

  async def test_foreach_async_mid_loop(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def f(i):
          nonlocal counter
          counter += i**2
          if 4 <= i <= 7:
            return aempty(i)
          return i

        def f_mod(i):
          nonlocal counter
          counter += i**2
          if 4 <= i <= 7:
            return aempty(i)
          return i

        counter = 0
        self.assertTrue(await await_(Chain(range(10)).then(fn).foreach(f).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))
        counter = 0
        self.assertTrue(await await_(Chain(range(10)).foreach(f).then(fn).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))

        coro = Chain(range(10)).foreach(f).run()
        self.assertTrue(inspect.isawaitable(coro))
        await coro

        counter = 0
        self.assertTrue(await await_(Chain(range(10)).then(fn).foreach_do(f_mod).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))
        counter = 0
        self.assertTrue(await await_(Chain(range(10)).foreach_do(f_mod).then(fn).eq(list(range(10))).run()))
        self.assertEqual(counter, sum(i**2 for i in range(10)))

        coro = Chain(range(10)).foreach_do(f).run()
        self.assertTrue(inspect.isawaitable(coro))
        await coro

  async def test_autorun(self):
    result = [False]
    this_task = asyncio.current_task()

    def set_result(v):
      time.sleep(0.1)
      result[0] = v

    def check_this_task(same_task: bool):
      if same_task:
        self.assertTrue(this_task == asyncio.current_task())
      else:
        self.assertFalse(this_task == asyncio.current_task())

    Chain(True).then(aempty).then(set_result).then(check_this_task, same_task=False).autorun(True).run()
    self.assertFalse(result[0])
    await asyncio.sleep(0.2)
    self.assertTrue(result[0])

    chain = Chain(False).then(aempty).then(set_result).then(check_this_task, same_task=True).run()
    self.assertTrue(result[0])
    await asyncio.sleep(0.2)
    self.assertTrue(result[0])
    await chain
    self.assertFalse(result[0])

  async def test_nested_chains_ensure_future(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = [False]
        this_task = asyncio.current_task()

        def set_result(v):
          time.sleep(0.1)
          result[0] = v

        def check_this_task(same_task: bool):
          if same_task:
            self.assertTrue(this_task == asyncio.current_task())
          else:
            self.assertFalse(this_task == asyncio.current_task())

        # test that a nested chain is executed within the same task as the parent chain
        chain = (
          Chain(True).then(fn).then(
            Chain().then(aempty).then(set_result).then(check_this_task, same_task=False).then(lambda: hash(asyncio.current_task()), ...)
          ).then(lambda t: t == hash(asyncio.current_task()))
          .do(check_this_task, same_task=False).autorun(True).run()
        )
        self.assertFalse(result[0])
        self.assertTrue(await await_(chain))
        self.assertTrue(result[0])

        # test that a nested chain and the parent chain is executed within the same task as this
        chain = Chain(False).then(fn).then(Chain().then(aempty).then(set_result).then(check_this_task, same_task=True)).then(check_this_task, same_task=True).run()
        self.assertTrue(result[0])
        await asyncio.sleep(0.2)
        self.assertTrue(result[0])
        await chain
        self.assertFalse(result[0])

  async def test_with(self):
    # test that the with_(callback) callback is executed inside a sync context manager, even if callback is async
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(FlexContext, v=1).with_(Chain().then(fn).then(lambda ctx: ctx.get(v=0))).eq(1).run()))

    @contextmanager
    def sync_ctx(v: int):
      yield v**2

    @asynccontextmanager
    async def async_ctx(v: int):
      yield v**2

    sync_cls = contextlib.AbstractContextManager
    async_cls = contextlib.AbstractAsyncContextManager

    for ctx, cls in [(sync_ctx, sync_cls), (async_ctx, async_cls)]:
      for fn in [empty, aempty]:
        with self.subTest(ctx=ctx, fn=fn):
          self.assertTrue(await await_(Chain(ctx, 10).then(fn).neq(100).run()))
          self.assertTrue(await await_(Chain(ctx, 10).then(fn).with_(lambda c: c).eq(100).run()))
          self.assertTrue(await await_(Chain(ctx, 10).then(fn).with_(lambda v: v/10).eq(10).run()))
          self.assertTrue(await await_(Chain(ctx, 10).with_(fn).eq(100).run()))
          self.assertTrue(await await_(Chain(None).then(fn).then(ctx, 10).with_(lambda c: c).eq(100).run()))
          self.assertTrue(await await_(Chain(None).then(fn).then(ctx, 10).with_(lambda v: v/10).eq(10).run()))
          self.assertTrue(await await_(Chain(None).then(fn).then(ctx, 10).with_(fn).eq(100).run()))

          self.assertTrue(await await_(Chain(ctx, 10).then(fn).with_do(lambda v: 100).then(lambda v: isinstance(v, cls)).run()))
          self.assertTrue(await await_(Chain(ctx, 10).then(fn).with_do(lambda v: v/10).then(lambda v: isinstance(v, cls)).run()))
          self.assertTrue(await await_(Chain(ctx, 10).with_do(fn).then(lambda v: isinstance(v, cls)).run()))
          self.assertTrue(await await_(Chain(None).then(fn).then(ctx, 10).with_do(lambda v: 100).then(lambda v: isinstance(v, cls)).run()))
          self.assertTrue(await await_(Chain(None).then(fn).then(ctx, 10).with_do(lambda v: v/10).then(lambda v: isinstance(v, cls)).run()))
          self.assertTrue(await await_(Chain(None).then(fn).then(ctx, 10).with_do(fn).then(lambda v: isinstance(v, cls)).run()))

  async def test_many_nested_chains(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        parent_chain = chain = Chain(42).then(fn)
        for _ in range(100):
          chain.then(chain := Chain().then(fn))
        self.assertEqual(await await_(parent_chain.run()), 42)

        parent_chain = chain = Chain(42).autorun(True).then(fn)
        for _ in range(100):
          chain.then(chain := Chain().then(fn))
        self.assertEqual(await await_(parent_chain.run()), 42)

  async def test_or(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(lambda: None).or_(6).eq(6).run()))
        self.assertTrue(await await_(Chain(6).or_(5).eq(6).run()))
        self.assertTrue(await await_(Chain(0).or_(5).eq(5).run()))


class ConditionalTests(IsolatedAsyncioTestCase):
  # TODO add tests to test the return value of all possible exception formats
  # TODO improve naming

  async def yield_conditional(self, root_value, conditional_attr, truthy_value, falsy_value):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        get_seq = lambda: getattr(Chain(root_value).then(fn), conditional_attr)
        self.assertTrue(await await_(get_seq()(truthy_value).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).if_(True).else_(False).run()))
        self.assertTrue(await await_(get_seq()(falsy_value).if_(False).else_(True).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).if_not(False).run()))
        self.assertTrue(await await_(get_seq()(falsy_value).if_not(True).run()))

        # this shows that a falsy conditional is skipped (the 'if_' is not invoked) and
        # the chain continues with the current value unchanged by the conditional
        self.assertTrue(await await_(get_seq()(truthy_value).if_not(False).eq(root_value).run()))
        self.assertTrue(await await_(get_seq()(falsy_value).if_(False).eq(root_value).run()))
        self.assertTrue(await await_(get_seq()(falsy_value).if_(False).else_(falsy_value).eq(falsy_value).run()))

        self.assertTrue(await await_(get_seq()(truthy_value).if_(False).not_().run()))
        self.assertTrue(await await_(get_seq()(truthy_value).if_(True).else_(False).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).not_().if_(False).else_(True).run()))

        self.assertTrue(await await_(get_seq()(truthy_value).if_(False).not_().run()))
        self.assertTrue(await await_(get_seq()(truthy_value).if_(lambda v: True).else_(lambda v: False).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).if_(lambda: True, ...).else_(lambda: False, ...).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).not_().if_(lambda v: v, False).else_(lambda v: v, True).run()))

  async def test_not(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(False).not_().run()))
        self.assertTrue(await await_(Chain(False).then(fn).not_().run()))
        self.assertTrue(await await_(Chain(fn).then(False).not_().run()))
        self.assertTrue(await await_(Chain(True).not_().not_().run()))
        self.assertTrue(await await_(Chain(False).not_().run()))
        self.assertTrue(await await_(Chain(True).not_().if_(False).else_(True).run()))
        self.assertTrue(await await_(Chain(True).not_().if_not(True).run()))
        self.assertTrue(await await_(Chain(True).then(fn).not_().if_(False).else_(True).run()))
        self.assertTrue(await await_(Chain(fn).then(True).not_().if_(False).else_(True).run()))

  async def test_eq(self):
    with self.subTest(conditional='eq'):
      await self.yield_conditional(1, 'eq', 1, 2)

  async def test_neq(self):
    with self.subTest(conditional='neq'):
      await self.yield_conditional(1, 'neq', 2, 1)

  async def test_is_(self):
    o = object()
    o2 = object()
    with self.subTest(conditional='is_'):
      await self.yield_conditional(o, 'is_', o, o2)

  async def test_is_not(self):
    o = object()
    o2 = object()
    with self.subTest(conditional='is_not'):
      await self.yield_conditional(o, 'is_not', o2, o)

  async def test_in_(self):
    t = [1, 2]
    f = [2, 3]
    with self.subTest(conditional='in_'):
      await self.yield_conditional(1, 'in_', t, f)

  async def test_not_in(self):
    t = [1, 2]
    f = [2, 3]
    with self.subTest(conditional='not_in'):
      await self.yield_conditional(1, 'not_in', f, t)

  async def test_conditional_last(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(1).then(fn).eq(2).not_().run()))

  async def test_conditional_last_attr(self):
    for fn, cls in [(empty, DummySync), (aempty, DummyAsync)]:
      with self.subTest(fn=fn):
        o = cls()
        self.assertTrue(await await_(Chain(o).then(fn).attr('a1').is_(o).if_(o).attr_fn('b1').is_(o).run()))
        self.assertTrue(await await_(ChainAttr(o).then(fn).a1.is_(o).if_(o).else_(None).b1().is_(o).run()))

  async def test_combinations(self):
    # a few random combinations of conditionals
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(None).then(fn).is_(None).if_(lambda: False, ...).else_(True).is_(False).run()))
        self.assertTrue(await await_(Chain(5).then(lambda v: v*10).in_([50, 51]).if_(fn, 9).else_(fn, 10).eq(9).run()))
        self.assertTrue(await await_(Chain(5).then(lambda v: v*10).in_([50, 51]).if_(fn, 9).else_(fn, 10).then(lambda v: v < 10).if_(lambda v: False).is_(False).is_not(False).run()))
        self.assertTrue(await await_(Chain(lambda: 1).then(fn).then(Chain().then(lambda v: v*10).neq(10).if_(lambda: 5, ...).else_(lambda v: 6)).neq(6).if_(lambda v: False).else_(True).run()))
        self.assertTrue(await await_(Chain(lambda: 1).then(fn).then(Chain().then(lambda v: v+5)).eq(6).run()))

  async def test_raise_if(self):
    class Exc(Exception): pass
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(1).then(fn).raise_(Exc(1)).except_(exc_fin_check.on_except).run())
        except Exc: pass
        self.assertTrue(exc_fin_check.ran_exc)

        exc_fin_check = efc_cls()
        try:
          await await_(Chain(1).then(fn).eq(1).if_raise(Exc(1)).except_(exc_fin_check.on_except).run())
        except Exc: pass
        self.assertTrue(exc_fin_check.ran_exc)

        exc_fin_check = efc_cls()
        try:
          await await_(Chain(1).then(fn).neq(1).if_raise(Exc(1)).else_raise(Exception(1)).except_(exc_fin_check.on_except).run())
        except Exc: self.assertTrue(False)
        except Exception: pass
        self.assertTrue(exc_fin_check.ran_exc)

        self.assertTrue(await await_(Chain(1).then(fn).neq(1).if_raise(Exc(1)).eq(1).run()))

  async def test_custom_conditional(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(1).condition(lambda v: fn(v % 2 == 0)).if_(False).else_(lambda v: fn(v*10)).eq(10).run()))


class TryExceptTest(IsolatedAsyncioTestCase):
  async def test_raise_on_await(self):
    class A:
      def __await__(self):
        async def f():
          raise Exception
        return f().__await__()

    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(A()).then(fn).except_(exc_fin_check.on_except).run())
        except Exception: pass
        self.assertTrue(exc_fin_check.ran_exc)

        exc_fin_check = efc_cls()
        try:
          await await_(Chain(fn).then(A, ...).except_(exc_fin_check.on_except).run())
        except Exception: pass
        self.assertTrue(exc_fin_check.ran_exc)

  async def test_try_except_1(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        await await_(Chain(False).then(fn).then(throw_if).except_(exc_fin_check.on_except).finally_(exc_fin_check.on_finally).run())
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_2(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        await await_(Chain(False).then(fn).then(throw_if).except_(exc_fin_check.on_except, ...).finally_(exc_fin_check.on_finally, ...).run())
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_3(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        await await_(Chain(False).then(fn).then(throw_if).except_(exc_fin_check.on_except, register=False).finally_(exc_fin_check.on_finally, register=False).run())
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertFalse(exc_fin_check.ran_finally)

  async def test_try_except_4(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(True).then(fn).then(throw_if).except_(exc_fin_check.on_except).finally_(exc_fin_check.on_finally).run())
        except Exception: pass
        self.assertTrue(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_5(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(True).then(fn).then(throw_if).except_(exc_fin_check.on_except, ...).finally_(exc_fin_check.on_finally, ...).run())
        except Exception: pass
        self.assertTrue(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_6(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(True).then(fn).then(throw_if).except_(exc_fin_check.on_except, register=False).finally_(exc_fin_check.on_finally, register=False).run())
        except Exception: pass
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertFalse(exc_fin_check.ran_finally)

  async def test_try_except_unknown_async_at_exc(self):
    # this function shows that if an exception is raised "before" an async coroutine has been
    # run (or if there aren't any async functions at all), and the except / finally callbacks are async,
    # they will be called in a separate task.
    exc_fin_check = ExceptFinallyCheckAsync()
    try:
      await await_(
        Chain(True).then(throw_if).then(aempty)
        .except_(Chain(asyncio.sleep, 0.1).then(exc_fin_check.on_except), ...)
        .finally_(Chain(asyncio.sleep, 0.1).then(exc_fin_check.on_finally), ...)
        .run()
      )
    except Exception: pass
    self.assertFalse(exc_fin_check.ran_exc)
    self.assertFalse(exc_fin_check.ran_finally)
    await asyncio.sleep(0.3)  # allow for the tasks to finish
    self.assertTrue(exc_fin_check.ran_exc)
    self.assertTrue(exc_fin_check.ran_finally)


class A_sync:
  @property
  def a1(self):
    return self

  def f1(self):
    return self

  def f2(self, v=None):
    return self

  def f3(self):
    return 1


class B_sync:
  @property
  def b1(self):
    pass

  def f1(self):
    pass

  def f2(self, v=None):
    pass

  def f3(self):
    return 1


class C_sync:
  def f1(self):
    return 5

  def f2(self, v):
    return v*10

  def f3(self, v):
    return v+10

  def f4(self, v):
    return v*2


class A_async:
  @property
  async def a1(self):
    return self

  async def f1(self):
    return self

  def f2(self, v=None):
    return self

  async def f3(self):
    return 1


class B_async:
  @property
  async def b1(self):
    pass

  async def f1(self):
    pass

  def f2(self, v=None):
    pass

  async def f3(self):
    return 1


class C_async:
  async def f1(self):
    return 5

  async def f2(self, v):
    return v*10

  def f3(self, v):
    return v+10

  async def f4(self, v):
    return v*2


class AttributesTest(IsolatedAsyncioTestCase):
  async def test_attribute_last(self):
    for fn, (A, B, C) in [(empty, (A_sync, B_sync, C_sync)), (aempty, (A_async, B_async, C_async))]:
      with self.subTest(fn=fn):
        a = A()
        self.assertIs(await await_(ChainAttr(a).a1.run()), a)
        self.assertFalse(await await_(ChainAttr(a).a1.then(False).run()))

  async def test_attribute_on_void(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          r = await await_(Cascade().then(fn).attr('attr').run())
        except QuentException:
          await await_(exc_fin_check.on_except())
        self.assertTrue(exc_fin_check.ran_exc)

        exc_fin_check = efc_cls()
        try:
          await await_(Cascade().then(fn).attr('attr').except_(exc_fin_check.on_except).run())
        except QuentException: pass
        self.assertTrue(exc_fin_check.ran_exc)

  async def test_many_attributes(self):
    for fn, (A, B, C) in [(empty, (A_sync, B_sync, C_sync)), (aempty, (A_async, B_async, C_async))]:
      with self.subTest(fn=fn):
        a = A()
        self.assertTrue(await await_(Chain(a).attr('a1').attr_fn('f1').attr_fn('f2', 1).attr_fn('f2', ...).attr_fn('f3').eq(1).run()))
        self.assertTrue(await await_(ChainAttr(a).a1.f1().f2('123').f2(...).f2().f3().eq(1).run()))

        b = B()
        self.assertTrue(await await_(Chain(Cascade(b).attr('b1').attr_fn('f1').attr_fn('f2', 1).attr_fn('f2', ...).attr_fn('f3').attr('b1')).is_(b).run()))
        self.assertTrue(await await_(Chain(CascadeAttr(b).b1.f1().f2('123').f2(...).f2().f3().b1.f1()).is_(b).run()))

        c = C()
        self.assertTrue(await await_(Chain(c.f1).then(c.f2).then(c.f3).then(c.f4).eq(120).run()))
        self.assertTrue(await await_(Chain(c.f1).then(c.f2).root(c.f3).then(c.f4).eq(30).run()))

        self.assertTrue(await await_(Chain(c.f1()).then(c.f2).then(c.f3(0)).then(c.f4).eq(20).run()))
        self.assertTrue(await await_(Chain(c.f1).then(c.f2).then(c.f3(0)).then(c.f4).eq(20).run()))


class MegaTests(IsolatedAsyncioTestCase):
  async def test_mega_chain(self):
    class Exc1(Exception):
      pass
    def throw():
      raise Exception
    def throw_if(v):
      if v:
        raise Exception
      return v
    def throw_if_1(v):
      if v:
        raise Exc1
      return v

    class A:
      @property
      def a1(self):
        return None

      def b1(self):
        return self

    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(
            Chain(lambda v: v, 1)
            .then(fn)
            .then(0)
            .root(lambda v: v+1)
            .do(lambda v: 10)
            .do(
              Cascade()
              .then(fn)
              .then(Chain().if_(None).else_(throw))
              .then(Chain().if_(lambda v: False).else_(throw))
              .then(Chain().if_(lambda: False, ...).else_(throw))
              .then(Chain().not_().not_().if_not(throw))
              .then(Chain().eq(2).if_not(throw))
              .then(Chain().neq(1).if_not(throw))
              .then(Chain().eq(1).if_(throw))
              .then(Chain().then(type).is_(int).if_not(throw))
              .then(Chain().then(type).is_not(str).if_(None).else_(throw))
              .then(Chain().then(type).is_(str).if_(throw))
              .then(Chain().in_([1, 2]).if_not(throw))
              .then(Chain().not_in([1, 3]).if_not(throw))
              .then(Chain().in_([1, 3]).if_(throw))
            )
            .do(
              Cascade()
              .then(Chain(A()).attr('a1').root(Chain().attr_fn('b1').attr('a1')))
              .then(ChainAttr(A()).a1.root(ChainAttr().b1().a1.root(ChainAttr().a1)).run)
              .then(CascadeAttr(A()).a1.a1.a1.a1.b1().a1.a1.run)
              .then(lambda: 10)
              .then(lambda v: Chain() | 5 | run(v), 1)
              .then(print, Chain())
              .then(print, Chain(1))
              .then(Chain(Chain).then(bool))
              , ...
            )
            .do(Chain().neq(2).if_(True).else_(lambda v: 1))
            .do(Chain().neq(2).if_(True).else_(lambda: 1, ...))
            .eq(2).if_not(throw)
            .do(Chain(fn).then(1).run, ...)
            .then(throw_if_1, True)
            .except_(exc_fin_check.on_except)
            .finally_(exc_fin_check.on_finally)
            .run()
          )
        except Exc1: pass
        self.assertTrue(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)
