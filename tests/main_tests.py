import asyncio
from typing import *
from tests.except_tests import ExceptFinallyCheckSync, raise_, ExceptFinallyCheckAsync, Exc1
from tests.flex_context import FlexContext, AsyncFlexContext
from tests.utils import empty, aempty, await_, MyTestCase as _BaseTestCase
from quent import Chain, Cascade, QuentException, run


class MyTestCase(_BaseTestCase):
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

    # multiple finally callbacks
    efc = ExceptFinallyCheckSync()
    try:
      Chain(True).finally_(lambda v: v).finally_(lambda v: v)
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # using Chain.return_() inside .iterate()
    efc = ExceptFinallyCheckSync()
    try:
      for _ in Chain(SyncIterator).iterate(Chain.return_):
        pass
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)
    #
    efc = ExceptFinallyCheckSync()
    try:
      async for _ in Chain(AsyncIterator).iterate(Chain.return_):
        pass
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # using Chain.break_() outside of loops
    efc = ExceptFinallyCheckSync()
    try:
      Chain().then(Chain.break_).run()
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)
    #
    efc = ExceptFinallyCheckAsync()
    try:
      await Chain(aempty).then(Chain.break_).run()
    except QuentException:
      await efc.on_except()
    await self.assertTrue(efc.ran_exc)

    # directly running a nested chain
    efc = ExceptFinallyCheckSync()
    try:
      c = Chain().then(None)
      Chain().then(c)
      c.run()
    except QuentException:
      efc.on_except()
    await self.assertTrue(efc.ran_exc)

  async def test_base(self):
    await self.assertIsNone((Cascade().run()))

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

                  # nesting chains
                  await self.assertIsObj(Chain(*obj1).then(Chain().then(*obj2).then(Chain().then(*obj1))).run())
                  await self.assertIsObj(Chain(*obj1).then(Chain().then(*obj2).then(Chain().then(*obj1)), object()).run())
                  await self.assertIsObj(Chain(*obj1).then(Chain().then(*obj2).then(Chain(*obj1).then(*obj2), ...)).run())

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

  async def test_freeze(self):
    obj = object()
    with FlexContext(obj=obj):
      for fn, ctx in self.with_fn():
        with ctx:
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

  async def test_return(self):
    obj_ = object()
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(Chain(Chain(fn).then(raise_, Exc1).except_(Chain.return_, ..., exceptions=Exc1)).then(object()).run())

        for obj in [(obj_,), ((lambda: fn(obj_)),), (fn, obj_)]:
          with self.subTest(obj=obj):
            await self.assertIs(Chain(Chain().then(Chain.return_, *obj)).then(object()).run(), obj_)
            await self.assertIs(Chain(Chain(fn).then(Chain.return_, *obj)).then(object()).run(), obj_)
            await self.assertIs(Chain(Chain(fn).then(raise_, Exc1).except_(Chain.return_, *obj, exceptions=Exc1)).then(object()).run(), obj_)

            def on_except(return_=True):
              if return_:
                return Chain.return_(*obj)
            await self.assertIs(Chain(Chain(fn, object()).then(raise_, Exc1).except_(on_except, True, exceptions=Exc1)).then(object()).run(), obj_)
            await self.assertIs(Chain(Chain(fn, object()).then(raise_, Exc1).except_(on_except, False, exceptions=Exc1, reraise=False)).then(obj_).run(), obj_)
            await self.assertIsNot(Chain(Chain(fn, object()).then(raise_, Exc1).except_(on_except, False, exceptions=Exc1, reraise=False)).then(object()).run(), obj_)

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

    # test nesting
    r = []
    for i in Chain(A).then(Chain().iterate(lambda i: i*2)).run():
      r.append(i)
    await self.assertEqual(r, rb)
    r = []
    async for i in Chain(A).then(Chain().iterate(lambda i: i*2)).run():
      r.append(i)
    await self.assertEqual(r, rb)
    r = []
    async for i in Chain(B).then(Chain().iterate(lambda i: i*2)).run():
      r.append(i)
    await self.assertEqual(r, rb)

    rb = [i*2 for i in range(9)]
    # test break
    def f(i):
      if i == 9:
        return Chain.break_()
      return i*2
    r = []
    for i in Chain(A).iterate(f):
      r.append(i)
    await self.assertEqual(r, rb)

    for fn, ctx in self.with_fn():
      with ctx:
        def f(i):
          if i == 9:
            return Chain.break_()
          return fn(i*2)
        r = []
        async for i in Chain(A).iterate(f):
          r.append(i)
        await self.assertEqual(r, rb)
        r = []
        async for i in Chain(B).iterate(f):
          r.append(i)
        await self.assertEqual(r, rb)

  async def test_foreach(self):
    A, B = SyncIterator, AsyncIterator
    rb = [i*2 for i in range(10)]

    for fn, ctx in self.with_fn():
      with ctx:
        for iterator in [A, B]:
          with self.subTest(iterator=iterator):
            await self.assertEqual(Chain(iterator).foreach(lambda i: fn(i*2)).run(), rb)
            for fn1 in [empty, aempty]:
              with self.subTest(fn1=fn1):
                await self.assertEqual(Chain(iterator).then(fn).foreach(lambda i: fn1(i*2)).run(), rb)
                await self.assertEqual(Chain(fn).then(iterator, ...).foreach(lambda i: fn1(i*2)).run(), rb)

            rb_break = [i*2 for i in range(9)]
            # test break
            def f(i):
              if i == 9:
                return Chain.break_()
              return fn(i*2)
            await self.assertEqual(Chain(iterator).foreach(f).run(), rb_break)

            obj = object()
            for break_arg in [(obj,), (lambda: fn(obj),), (fn, obj)]:
              with self.subTest(break_arg=break_arg):
                def f(i):
                  if i == 9:
                    return Chain.break_(*break_arg)
                  return fn(i*2)
                await self.assertIs(Chain(iterator).foreach(f).run(), obj)

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
                  .do(lambda: super(_BaseTestCase, self).assertEqual(flx_ctx.get(v=0), 0), ...)
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

  async def test_null_repr(self):
    """Test __repr__ of _Null class to hit line 29."""
    from quent.quent import PyNull
    result = repr(PyNull)
    await self.assertEqual(result, '<Null>')

  async def test_traceback_modification(self):
    """Test traceback modification to hit lines 62, 69-71."""
    def sync_error():
      raise RuntimeError("Sync traceback test")

    async def async_error():
      raise RuntimeError("Async traceback test")

    # Test sync traceback modification (line 62)
    try:
      await await_(Chain(sync_error).run())
      await self.assertTrue(False)  # Should not reach here
    except RuntimeError:
      pass  # Expected

    # Test async traceback modification (lines 69-71)
    try:
      await await_(Chain(async_error).run())
      await self.assertTrue(False)  # Should not reach here
    except RuntimeError:
      pass  # Expected

  async def test_context_manager_finally_blocks(self):
    """Test context manager finally blocks to hit lines 303, 312."""

    class TestContextManager:
      def __init__(self):
        self.entered = False
        self.exited = False
        self.exc_info = None

      def __enter__(self):
        self.entered = True
        return "context_value"

      def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        self.exc_info = (exc_type, exc_val, exc_tb)
        return False  # Don't suppress exceptions

    # Test normal execution (line 303)
    cm1 = TestContextManager()
    result = await await_(Chain(cm1).with_(lambda x: f"processed_{x}").run())
    await self.assertEqual(result, "processed_context_value")
    await self.assertTrue(cm1.entered)
    await self.assertTrue(cm1.exited)

    # Test with exception (line 312 for async case)
    class AsyncTestContextManager:
      def __init__(self):
        self.entered = False
        self.exited = False

      async def __aenter__(self):
        self.entered = True
        return "async_context"

      async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

    async def async_error_in_context(ctx):
      raise ValueError("Async context error")

    async_cm = AsyncTestContextManager()
    try:
      await await_(Chain(async_cm).with_(async_error_in_context).run())
      await self.assertTrue(False)  # Should not reach here
    except ValueError:
      pass  # Expected

    await self.assertTrue(async_cm.entered)
    await self.assertTrue(async_cm.exited)
