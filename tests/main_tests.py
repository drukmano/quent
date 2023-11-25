import asyncio
import contextlib
import inspect
from typing import *
import time
from contextlib import contextmanager, asynccontextmanager
from tests.except_tests import ExceptFinallyCheckSync
from tests.flex_context import FlexContext, AsyncFlexContext
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_
from quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException, run


class MyTestCase(IsolatedAsyncioTestCase):
  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)

  async def assertTrue(self, expr: Any, msg: Any = None) -> None:
    return super().assertTrue(await await_(expr), msg)

  async def assertFalse(self, expr: Any, msg: Any = None) -> None:
    return super().assertFalse(await await_(expr), msg)

  async def assertEqual(self, first: Any, second: Any, msg: Any = None) -> None:
    return super().assertEqual(await await_(first), second, msg)

  async def assertIsNone(self, obj: object, msg: Any = None) -> None:
    return super().assertIsNone(await await_(obj), msg)

  async def assertIs(self, expr1: object, expr2: object, msg: Any = None) -> None:
    return super().assertIs(await await_(expr1), expr2, msg)

  async def assertIsNot(self, expr1: object, expr2: object, msg: Any = None) -> None:
    return super().assertIsNot(await await_(expr1), expr2, msg)

  async def assertIsObj(self, expr1: object, msg: Any = None) -> None:
    return super().assertIs(await await_(expr1), FlexContext.get(obj=object()), msg)

  async def assertIsNotObj(self, expr1: object, msg: Any = None) -> None:
    return super().assertIsNot(await await_(expr1), FlexContext.get(obj=object()), msg)


class SyncIterator:
  def __init__(self):
    self.range = None

  def __iter__(self):
    self.range = iter(range(10))
    return self

  def __next__(self):
    return self.range.__next__()


class AsyncIterator:
  def __init__(self):
    self.range = None

  def __aiter__(self):
    self.range = iter(range(10))
    return self

  async def __anext__(self):
    try:
      return self.range.__next__()
    except StopIteration:
      raise StopAsyncIteration


class SingleTest(MyTestCase):
  async def test_quent_exceptions(self):
    # override root value
    efc = ExceptFinallyCheckSync()
    try:
      Chain(True).run(True)
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # non function value
    efc = ExceptFinallyCheckSync()
    Chain(None).then(True).run()
    try:
      Chain(None).do(True).run()
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # [sync] attr on void cascade
    efc = ExceptFinallyCheckSync()
    try:
      Cascade().attr('p1').run()
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # [async] attr on void cascade
    efc = ExceptFinallyCheckSync()
    try:
      await Cascade().then(aempty).attr('p1').run()
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # multiple finally callbacks
    efc = ExceptFinallyCheckSync()
    try:
      Chain(True).finally_(lambda v: v).finally_(lambda v: v)
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # else_ without a preceding if_
    efc = ExceptFinallyCheckSync()
    try:
      Chain(True).else_(False)
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

  async def test_cascade(self):
    await self.assertIsNone((Cascade().run()))

  async def test_base(self):
    obj_ = object()
    with FlexContext(obj=obj_):
      for fn, ctx in self.with_fn():
        with ctx:
          await self.assertIsNone((Chain().run()))
          await self.assertIsObj((Chain().then(fn, obj_).run()))

          await self.assertIsObj(Chain(fn, obj_).run())
          await self.assertIsObj(Chain().run(fn, obj_))
          await self.assertIsObj(Chain(fn, obj_)())
          await self.assertIsObj(Chain()(fn, obj_))

          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              await self.assertIsObj(Chain().then(*obj).run())
              await self.assertIsObj(Chain(*obj).run())
              await self.assertIsObj(Chain().run(*obj))

              for obj1, obj2 in [(obj, (fn,)), ((fn,), obj)]:
                with self.subTest(obj1=obj1, obj2=obj2):
                  await self.assertIsObj(Chain().then(*obj1).then(*obj2).run())

                  await self.assertIsObj(Chain(*obj1).then(*obj2).run())
                  await self.assertIsObj(Chain().then(*obj2).run(*obj1))

                  await self.assertIsObj(Chain(*obj1).then(*obj1).then(*obj2).run())
                  await self.assertIsObj(Chain().then(*obj1).then(*obj2).run(*obj1))

                  await self.assertIsObj(Chain(*obj1).then(*obj2).then(*obj1).run())
                  await self.assertIsObj(Chain().then(*obj2).then(*obj1).run(*obj1))

                  await self.assertIsNotObj(Chain(*obj1).then(*obj2).then(*obj1).then(*obj2).then(object()).run())
                  await self.assertIsNotObj(Chain().then(*obj2).then(*obj1).then(*obj2).then(object()).run(*obj1))

  async def test_do(self):
    obj_ = object()
    with FlexContext(obj=obj_):
      for fn, ctx in self.with_fn():
        with ctx:
          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              await self.assertIsObj(Chain(*obj).do(fn, None).run())
              await self.assertIsObj(Chain(*obj).do(fn, None).do(lambda: object(), ...).then(fn).run())
              await self.assertIsObj(Chain(*obj).do(lambda: object(), ...).then(fn).run())
              await self.assertIsObj(Chain(*obj).then(fn).do(lambda: object(), ...).run())
              await self.assertIsNotObj(Chain(object()).then(fn).do(lambda v: obj_).then(fn).run())

  async def test_root(self):
    obj_ = object()
    with FlexContext(obj=obj_):
      for fn, ctx in self.with_fn():
        with ctx:
          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              await self.assertIsObj(Chain(*obj).then(fn, None).root(fn).run())
              await self.assertIsNotObj(Chain(object()).then(*obj).root(fn).run())

  async def test_root_do(self):
    obj_ = object()
    with FlexContext(obj=obj_):
      for fn, ctx in self.with_fn():
        with ctx:
          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              await self.assertIsObj(Chain(object()).then(*obj).root_do(lambda v: self.assertIsNotObj(v)).run())
              await self.assertIsNotObj(Chain(*obj).then(fn, object()).root_do(lambda v: self.assertIsObj(v)).run())

  async def test_attr(self):
    obj_ = object()
    class A:
      @property
      def p1(self):
        return obj_
      def f1(self, v=obj_):
        return v

    with FlexContext(obj=obj_):
      for fn, ctx in self.with_fn():
        with ctx:
          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              await self.assertIsObj(Chain(A).then(fn).attr('p1').run())
              await self.assertIsObj(Chain(A).attr('p1').then(fn).run())
              await self.assertIsObj(Chain(A).then(fn).attr_fn('f1').run())
              await self.assertIsObj(Chain(A).attr_fn('f1').then(fn).run())
              await self.assertIsNotObj(Chain(A).then(fn).attr_fn('f1', object()).run())

  async def test_conditionals(self):
    obj_ = object()
    with FlexContext(obj=obj_):
      for fn, ctx in self.with_fn():
        with ctx:
          await self.assertEqual(Chain(fn, True).if_(fn).run(), True)
          await self.assertEqual(Chain(fn, False).if_(True).run(), False)
          await self.assertEqual(Chain(empty).then(fn, True).if_(fn).run(), True)
          await self.assertEqual(Chain(empty).then(fn, False).if_(True).run(), False)

          await self.assertEqual(Chain(fn, False).if_(True).else_(fn).run(), False)
          await self.assertEqual(Chain(fn, True).if_(fn).else_(False).run(), True)
          await self.assertEqual(Chain(empty).then(fn, False).if_(True).else_(fn).run(), False)
          await self.assertEqual(Chain(empty).then(fn, True).if_(fn).else_(False).run(), True)

          for on_true in [lambda v: (v,), lambda v: (fn, v)]:
            with self.subTest(on_true=on_true):
              await self.assertEqual(Chain(fn, True).if_(*on_true(1)).run(), 1)
              await self.assertEqual(Chain(fn, False).if_(1).run(), False)
              await self.assertEqual(Chain(empty).then(fn, True).if_(*on_true(1)).run(), 1)
              await self.assertEqual(Chain(empty).then(fn, False).if_(1).run(), False)

              await self.assertEqual(Chain(fn, False).if_(1).else_(*on_true(0)).run(), 0)
              await self.assertEqual(Chain(fn, True).if_(*on_true(1)).else_(0).run(), 1)
              await self.assertEqual(Chain(empty).then(fn, False).if_(1).else_(*on_true(0)).run(), 0)
              await self.assertEqual(Chain(empty).then(fn, True).if_(*on_true(1)).else_(0).run(), 1)

              await self.assertEqual(Chain(fn, 4).condition(lambda v: fn(v%2==0)).run(), True)
              await self.assertEqual(Chain(fn, 4).condition(lambda v: fn(v%2==0)).if_(*on_true(1)).run(), 1)
              await self.assertEqual(Chain(fn, 3).condition(lambda v: fn(v%2==0)).if_(1).run(), 3)
              await self.assertEqual(Chain(fn, 3).condition(lambda v: fn(v%2==0)).if_(1).else_(*on_true(2)).run(), 2)

              await self.assertEqual(Chain(empty).then(fn, 4).condition(lambda v: fn(v%2==0)).run(), True)
              await self.assertEqual(Chain(empty).then(fn, 4).condition(lambda v: fn(v%2==0)).if_(*on_true(1)).run(), 1)
              await self.assertEqual(Chain(empty).then(fn, 3).condition(lambda v: fn(v%2==0)).if_(1).run(), 3)
              await self.assertEqual(Chain(empty).then(fn, 3).condition(lambda v: fn(v%2==0)).if_(1).else_(*on_true(2)).run(), 2)

          await self.assertEqual(Chain(fn, False).if_not(1).run(), 1)
          await self.assertEqual(Chain(fn, True).if_not(1).else_(0).run(), 0)

          await self.assertTrue(Chain(fn, False).not_().run())
          await self.assertFalse(Chain(fn, 1).not_().run())

          await self.assertTrue(Chain(fn, 1).eq(1).run())
          await self.assertFalse(Chain(fn, 2).eq(1).run())
          await self.assertTrue(Chain(fn, 2).neq(1).run())
          await self.assertFalse(Chain(fn, 1).neq(1).run())

          await self.assertTrue(Chain(fn, obj_).is_(obj_).run())
          await self.assertFalse(Chain(fn, object()).is_(obj_).run())
          await self.assertTrue(Chain(fn, object()).is_not(obj_).run())
          await self.assertFalse(Chain(fn, obj_).is_not(obj_).run())

          await self.assertTrue(Chain(fn, obj_).in_([obj_]).run())
          await self.assertFalse(Chain(fn, object()).in_([obj_]).run())
          await self.assertTrue(Chain(fn, object()).not_in([obj_]).run())
          await self.assertFalse(Chain(fn, obj_).not_in([obj_]).run())

          await self.assertIsObj(Chain(None).or_(obj_).run())
          await self.assertIsNotObj(Chain(True).or_(obj_).run())

  async def test_autorun_config(self):
    # an object that can have dynamic attributes set
    obj = type('', (), {})()
    def f():
      obj._v = True

    for fn, ctx in self.with_fn():
      with ctx:
        obj._v = False
        coro = Chain(fn).then(asyncio.sleep, 0.1).then(f, ...).config(autorun=False).run()
        await self.assertFalse(obj._v)
        await asyncio.sleep(0.2)
        await self.assertFalse(obj._v)
        await coro

        obj._v = False
        coro = Chain(fn).then(asyncio.sleep, 0.1).then(f, ...).config(autorun=True).run()
        await self.assertFalse(obj._v)
        await asyncio.sleep(0.2)
        await self.assertTrue(obj._v)
        await coro

        obj._v = False
        coro = Chain(fn).then(asyncio.sleep, 0.1).then(f, ...).autorun(False).run()
        await self.assertFalse(obj._v)
        await asyncio.sleep(0.2)
        await self.assertFalse(obj._v)
        await coro

        obj._v = False
        coro = Chain(fn).then(asyncio.sleep, 0.1).then(f, ...).autorun().run()
        await self.assertFalse(obj._v)
        await asyncio.sleep(0.2)
        await self.assertTrue(obj._v)
        await coro

  async def test_freeze(self):
    obj = object()
    with FlexContext(obj=obj):
      for fn, ctx in self.with_fn():
        with ctx:
          #chain = Chain(obj).then(fn)
          #chain2 = chain.clone().then(object())
          #await self.assertIsObj(chain.run())
          #await self.assertIsNotObj(chain2.run())

          #chain = Chain(fn).then(obj)
          #chain2 = chain.clone().then(object())
          #await self.assertIsObj(chain.run())
          #await self.assertIsNotObj(chain2.run())

          chain = Chain(obj).then(fn).freeze()
          await self.assertIsObj(chain.run())
          await self.assertIsObj(chain())

          @Chain().then(fn).then(lambda v: v**2).decorator()
          def f1(v):
            return v*2
          await self.assertEqual(f1(2), 16)
          @Chain().then(fn).then(lambda v: v**2).decorator()
          async def f1(v):
            return v*2
          await self.assertEqual(f1(2), 16)

  async def test_while(self):
    class A:
      def __init__(self):
        self.i = 0
    obj = object()
    for fn, ctx in self.with_fn():
      with ctx:
        # test break
        def f0():
          Chain.break_()
          raise Exception
        await self.assertIs(Chain(obj).then(fn).while_true(f0, ...).run(), obj)
        await self.assertEqual(Chain().then(fn, 1).while_true(Chain().then(fn).then(f0, ...)).run(), 1)

        # test continue
        def f0():
          a.i += 1
          if a.i == 10:
            Chain.return_()
            raise Exception
          Chain.continue_()
          raise Exception

        a = A()
        await self.assertIs(Chain(obj).then(fn).while_true(f0, ...).run(), obj)
        a = A()
        await self.assertIs(Chain(obj).then(fn).while_true(Chain().then(fn).then(f0, ...)).run(), obj)

        # test return with different values
        for incr in [1, 3, 7]:
          with self.subTest(incr=incr):
            def f1(v=0):
              a.i += incr
              if a.i == 10:
                Chain.continue_()
                a.i = 500
              if a.i >= 500:
                raise Exception
              if a.i == 100:
                Chain.return_(1+v)
              elif a.i == 102:
                Chain.return_(3+v)
              elif a.i == 105:
                Chain.return_(7+v)
              elif a.i > 50:
                return fn()  # test a change to async mid-loop.

            async def f2(v=0):
              r = f1(v)
              if inspect.isawaitable(r):
                return await r
              return r

            for loop_fn in [f1, f2]:
              with self.subTest(loop_fn=loop_fn):
                a = A()
                await self.assertEqual(Chain().while_true(loop_fn).run(), incr)

                a = A()
                await self.assertEqual(Chain().while_true(Chain().then(fn, 0).then(loop_fn)).run(), incr)

                a = A()
                await self.assertEqual(Chain(None).while_true(loop_fn, ...).run(), incr)

                a = A()
                await self.assertEqual(Chain(5).while_true(loop_fn).run(), incr+5)

  async def test_iterate(self):
    A, B = SyncIterator, AsyncIterator
    rb = [i*2 for i in range(10)]

    r = []
    for i in Chain(A).iterate(lambda i: i*2):
      r.append(i)
    await self.assertEqual(r, rb)
    r = []
    for i in Chain(A).iterate():
      r.append(i*2)
    await self.assertEqual(r, rb)
    r = []
    async for i in Chain(A).iterate():
      r.append(i*2)
    await self.assertEqual(r, rb)

    r = []
    async for i in Chain(B).iterate(lambda i: i*2):
      r.append(i)
    await self.assertEqual(r, rb)
    r = []
    async for i in Chain(B).iterate():
      r.append(i*2)
    await self.assertEqual(r, rb)

    r = []
    for i in Chain(A).iterate_do(lambda i: i/2):
      r.append(i*2)
    await self.assertEqual(r, rb)
    r = []
    async for i in Chain(B).iterate_do(lambda i: i/2):
      r.append(i*2)
    await self.assertEqual(r, rb)

    for fn, ctx in self.with_fn():
      with ctx:
        r = []
        async for i in Chain(A).then(fn).iterate(lambda i: aempty(i*2)):
          r.append(i)
        await self.assertEqual(r, rb)
        r = []
        async for i in Chain(B).iterate(lambda i: fn(i*2)):
          r.append(i)
        await self.assertEqual(r, rb)
        r = []
        async for i in Chain(B).then(fn).iterate(lambda i: i*2):
          r.append(i)
        await self.assertEqual(r, rb)

    r = []
    for i in Chain().iterate(lambda i: i*2)(A):
      r.append(i)
    await self.assertEqual(r, rb)

  async def test_foreach(self):
    A, B = SyncIterator, AsyncIterator
    rb = [i*2 for i in range(10)]
    rb_do = [i for i in range(10)]

    for fn, ctx in self.with_fn():
      with ctx:
        for iterator in [A, B]:
          with self.subTest(iterator=iterator):
            await self.assertEqual(Chain(iterator).foreach(lambda i: fn(i*2)).run(), rb)
            for fn1 in [empty, aempty]:
              with self.subTest(fn1=fn1):
                await self.assertEqual(Chain(iterator).then(fn).foreach(lambda i: fn1(i*2)).run(), rb)
                await self.assertEqual(Chain(fn).then(iterator, ...).foreach(lambda i: fn1(i*2)).run(), rb)

                await self.assertEqual(Chain(iterator).then(fn).foreach_do(lambda i: fn1(i*2)).run(), rb_do)
                await self.assertEqual(Chain(fn).then(iterator, ...).foreach_do(lambda i: fn1(i*2)).run(), rb_do)

  async def test_with(self):
    for fn, ctx in self.with_fn():
      with ctx:
        for flx_ctx in [FlexContext, AsyncFlexContext]:
          with self.subTest(flx_ctx=flx_ctx):
            await self.assertEqual(Chain(flx_ctx(v=1)).then(fn).with_(flx_ctx.get, v=0).run(), 1)
            await self.assertEqual(Chain(flx_ctx(v=1)).then(fn).then(flx_ctx.get, v=0).run(), 0)
            for fn1 in [empty, aempty]:
              with self.subTest(fn1=fn1):
                await self.assertEqual(
                  Chain(flx_ctx(v=1)).then(fn)
                  .with_(Chain().then(fn1).then(flx_ctx.get, v=0))
                  .do(lambda: self.assertEqual(flx_ctx.get(v=0), 0), ...)
                  .run(), 1
                )
                await self.assertEqual(
                  Chain(flx_ctx(v=1)).then(fn)
                  .with_(lambda flx: flx)
                  .with_(Chain().then(fn1).then(lambda flx: flx.get(v=0)))
                  .run(), 1
                )
                await self.assertEqual(
                  Chain(flx_ctx(v=1)).then(fn)
                  .with_(Chain().then(fn1).then(flx_ctx, v=2))
                  .with_(Chain().then(fn1).then(lambda flx: flx.get(v=0)))
                  .run(), 2
                )

                await self.assertEqual(
                  Chain(flx_ctx(v=1)).then(fn)
                  .with_do(Chain().then(fn1).then(flx_ctx, v=2))
                  .with_(Chain().then(fn1).then(lambda flx: flx.get(v=0)))
                  .run(), 1
                )
                await self.assertEqual(
                  Chain(flx_ctx(v=1)).then(fn)
                  .with_(Chain().then(fn1).then(flx_ctx, v=2))
                  .with_do(Chain().then(fn1).then(lambda flx: flx.get(v=0)))
                  .with_(Chain().then(fn1).then(lambda flx: flx.get(v=0)))
                  .run(), 2
                )

  async def test_attr_cls(self):
    obj_ = object()
    class A:
      @property
      def p1(self):
        return obj_
      def f1(self, v=obj_):
        return v

    with FlexContext(obj=obj_):
      for fn, ctx in self.with_fn():
        with ctx:
          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              await self.assertIsObj(ChainAttr(A).then(fn).p1.run())
              await self.assertIsObj(ChainAttr(A).p1.then(fn).run())
              await self.assertIsObj(ChainAttr(A).then(fn).f1().run())
              await self.assertIsObj(ChainAttr(A).f1().then(fn).run())
              await self.assertIsNotObj(ChainAttr(A).then(fn).f1(object()).run())
              await self.assertIs(CascadeAttr(a := A()).then(fn).p1.f1().p1.f1(object()).then(None).run(), a)
