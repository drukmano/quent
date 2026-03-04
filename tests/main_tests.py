import asyncio
from itertools import product
from typing import *
from unittest import IsolatedAsyncioTestCase
from tests.except_tests import ExceptFinallyCheckSync, raise_, ExceptFinallyCheckAsync, Exc1
from tests.flex_context import FlexContext, AsyncFlexContext
from tests.utils import empty, aempty, await_
from quent import Chain, QuentException


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


class SingleTest(IsolatedAsyncioTestCase):
  async def test_quent_exceptions(self):
    # override root value
    efc = ExceptFinallyCheckSync()
    try:
      Chain(True).run(True)
    except QuentException:
      efc.on_except()
    self.assertTrue(efc.ran_exc)

    # non function value
    efc = ExceptFinallyCheckSync()
    Chain(None).then(True).run()
    try:
      Chain(None).do(True).run()
    except QuentException:
      efc.on_except()
    self.assertTrue(efc.ran_exc)

    # multiple finally callbacks
    efc = ExceptFinallyCheckSync()
    try:
      Chain(True).finally_(lambda v: v).finally_(lambda v: v)
    except QuentException:
      efc.on_except()
    self.assertTrue(efc.ran_exc)

    # using Chain.return_() inside .iterate()
    efc = ExceptFinallyCheckSync()
    try:
      for _ in Chain(SyncIterator).iterate(Chain.return_):
        pass
    except QuentException:
      efc.on_except()
    self.assertTrue(efc.ran_exc)
    #
    efc = ExceptFinallyCheckSync()
    try:
      async for _ in Chain(AsyncIterator).iterate(Chain.return_):
        pass
    except QuentException:
      efc.on_except()
    self.assertTrue(efc.ran_exc)

    # using Chain.break_() outside of loops
    efc = ExceptFinallyCheckSync()
    try:
      Chain().then(Chain.break_).run()
    except QuentException:
      efc.on_except()
    self.assertTrue(efc.ran_exc)
    #
    efc = ExceptFinallyCheckAsync()
    try:
      await Chain(aempty).then(Chain.break_).run()
    except QuentException:
      await efc.on_except()
    self.assertTrue(efc.ran_exc)

    # directly running a nested chain
    efc = ExceptFinallyCheckSync()
    try:
      c = Chain().then(None)
      Chain().then(c)
      c.run()
    except QuentException:
      efc.on_except()
    self.assertTrue(efc.ran_exc)

  async def test_base(self):
    obj_ = object()
    with FlexContext(obj=obj_):
      for fn1, fn2, fn3, fn4, fn5, fn6 in product([empty, aempty], repeat=6):
        with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4, fn5=fn5, fn6=fn6):
          self.assertIsNone(Chain().run())
          self.assertIs(await await_(Chain().then(fn1, obj_).run()), obj_)

          self.assertIs(await await_(Chain(fn2, obj_).run()), obj_)
          self.assertIs(await await_(Chain().run(fn3, obj_)), obj_)
          self.assertIs(await await_(Chain(fn4, obj_)()), obj_)
          self.assertIs(await await_(Chain()(fn5, obj_)), obj_)

          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              self.assertIs(await await_(Chain().then(*obj).run()), obj_)
              self.assertIs(await await_(Chain(*obj).run()), obj_)
              self.assertIs(await await_(Chain().run(*obj)), obj_)

              for obj1, obj2 in [(obj, (fn6,)), ((fn6,), obj)]:
                with self.subTest(obj1=obj1, obj2=obj2):
                  self.assertIs(await await_(Chain().then(*obj1).then(*obj2).run()), obj_)

                  self.assertIs(await await_(Chain(*obj1).then(*obj2).run()), obj_)
                  self.assertIs(await await_(Chain().then(*obj2).run(*obj1)), obj_)

                  self.assertIs(await await_(Chain(*obj1).then(*obj1).then(*obj2).run()), obj_)
                  self.assertIs(await await_(Chain().then(*obj1).then(*obj2).run(*obj1)), obj_)

                  self.assertIs(await await_(Chain(*obj1).then(*obj2).then(*obj1).run()), obj_)
                  self.assertIs(await await_(Chain().then(*obj2).then(*obj1).run(*obj1)), obj_)

                  self.assertIsNot(await await_(Chain(*obj1).then(*obj2).then(*obj1).then(*obj2).then(object()).run()), obj_)
                  self.assertIsNot(await await_(Chain().then(*obj2).then(*obj1).then(*obj2).then(object()).run(*obj1)), obj_)

                  # nesting chains
                  self.assertIs(await await_(Chain(*obj1).then(Chain().then(*obj2).then(Chain().then(*obj1))).run()), obj_)
                  self.assertIs(await await_(Chain(*obj1).then(Chain().then(*obj2).then(Chain().then(*obj1)), object()).run()), obj_)
                  self.assertIs(await await_(Chain(*obj1).then(Chain().then(*obj2).then(Chain(*obj1).then(*obj2), ...)).run()), obj_)

  async def test_do(self):
    obj_ = object()
    with FlexContext(obj=obj_):
      for fn1, fn2, fn3, fn4, fn5, fn6, fn7 in product([empty, aempty], repeat=7):
        with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4, fn5=fn5, fn6=fn6, fn7=fn7):
          for obj in [(obj_,), (lambda v=None: obj_,), (lambda: obj_, ...)]:
            with self.subTest(obj=obj):
              self.assertIs(await await_(Chain(*obj).do(fn1, None).run()), obj_)
              self.assertIs(await await_(Chain(*obj).do(fn2, None).do(lambda: object(), ...).then(fn3).run()), obj_)
              self.assertIs(await await_(Chain(*obj).do(lambda: object(), ...).then(fn4).run()), obj_)
              self.assertIs(await await_(Chain(*obj).then(fn5).do(lambda: object(), ...).run()), obj_)
              self.assertIsNot(await await_(Chain(object()).then(fn6).do(lambda v: obj_).then(fn7).run()), obj_)

  async def test_autorun_config(self):
    # an object that can have dynamic attributes set
    obj = type('', (), {})()
    def f():
      obj._v = True

    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        obj._v = False
        coro = Chain(fn1).then(asyncio.sleep, 0.1).then(f, ...).config(autorun=False).run()
        self.assertFalse(obj._v)
        await asyncio.sleep(0.2)
        self.assertFalse(obj._v)
        await coro

        obj._v = False
        coro = Chain(fn2).then(asyncio.sleep, 0.1).then(f, ...).config(autorun=True).run()
        self.assertFalse(obj._v)
        await asyncio.sleep(0.2)
        self.assertTrue(obj._v)
        await coro

  async def test_chain_reuse_and_decorator(self):
    obj = object()
    with FlexContext(obj=obj):
      for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
        with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
          chain = Chain(obj).then(fn1)
          self.assertIs(await await_(chain.run()), obj)
          self.assertIs(await await_(chain()), obj)

          @Chain().then(fn2).then(lambda v: v**2).decorator()
          def f1(v):
            return v*2
          self.assertEqual(await await_(f1(2)), 16)
          @Chain().then(fn3).then(lambda v: v**2).decorator()
          async def f1(v):
            return v*2
          self.assertEqual(await await_(f1(2)), 16)

  async def test_return(self):
    obj_ = object()
    for fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8 in product([empty, aempty], repeat=8):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4, fn5=fn5, fn6=fn6, fn7=fn7, fn8=fn8):
        self.assertIsNone(await await_(Chain(Chain(fn1).then(raise_, Exc1).except_(Chain.return_, ..., exceptions=Exc1)).then(object()).run()))

        for obj in [(obj_,), ((lambda: fn2(obj_)),), (fn3, obj_)]:
          with self.subTest(obj=obj):
            self.assertIs(await await_(Chain(Chain().then(Chain.return_, *obj)).then(object()).run()), obj_)
            self.assertIs(await await_(Chain(Chain(fn4).then(Chain.return_, *obj)).then(object()).run()), obj_)
            self.assertIs(await await_(Chain(Chain(fn5).then(raise_, Exc1).except_(Chain.return_, *obj, exceptions=Exc1)).then(object()).run()), obj_)

            def on_except(return_=True):
              if return_:
                return Chain.return_(*obj)
            self.assertIs(await await_(Chain(Chain(fn6, object()).then(raise_, Exc1).except_(on_except, True, exceptions=Exc1)).then(object()).run()), obj_)
            self.assertIs(await await_(Chain(Chain(fn7, object()).then(raise_, Exc1).except_(on_except, False, exceptions=Exc1, reraise=False)).then(obj_).run()), obj_)
            self.assertIsNot(await await_(Chain(Chain(fn8, object()).then(raise_, Exc1).except_(on_except, False, exceptions=Exc1, reraise=False)).then(object()).run()), obj_)

  async def test_iterate(self):
    A, B = SyncIterator, AsyncIterator
    rb = [i*2 for i in range(10)]

    r = []
    for i in Chain(A).iterate(lambda i: i*2):
      r.append(i)
    self.assertEqual(r, rb)
    r = []
    for i in Chain(A).iterate():
      r.append(i*2)
    self.assertEqual(r, rb)
    r = []
    async for i in Chain(A).iterate():
      r.append(i*2)
    self.assertEqual(r, rb)

    r = []
    async for i in Chain(B).iterate(lambda i: i*2):
      r.append(i)
    self.assertEqual(r, rb)
    r = []
    async for i in Chain(B).iterate():
      r.append(i*2)
    self.assertEqual(r, rb)

    for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
        r = []
        async for i in Chain(A).then(fn1).iterate(lambda i: aempty(i*2)):
          r.append(i)
        self.assertEqual(r, rb)
        r = []
        async for i in Chain(B).iterate(lambda i: fn2(i*2)):
          r.append(i)
        self.assertEqual(r, rb)
        r = []
        async for i in Chain(B).then(fn3).iterate(lambda i: i*2):
          r.append(i)
        self.assertEqual(r, rb)

    # test nesting
    r = []
    for i in Chain(A).then(Chain().iterate(lambda i: i*2)).run():
      r.append(i)
    self.assertEqual(r, rb)
    r = []
    async for i in Chain(A).then(Chain().iterate(lambda i: i*2)).run():
      r.append(i)
    self.assertEqual(r, rb)
    r = []
    async for i in Chain(B).then(Chain().iterate(lambda i: i*2)).run():
      r.append(i)
    self.assertEqual(r, rb)

    rb = [i*2 for i in range(9)]
    # test break
    def f(i):
      if i == 9:
        return Chain.break_()
      return i*2
    r = []
    for i in Chain(A).iterate(f):
      r.append(i)
    self.assertEqual(r, rb)

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def f(i):
          if i == 9:
            return Chain.break_()
          return fn(i*2)
        r = []
        async for i in Chain(A).iterate(f):
          r.append(i)
        self.assertEqual(r, rb)
        r = []
        async for i in Chain(B).iterate(f):
          r.append(i)
        self.assertEqual(r, rb)

  async def test_foreach(self):
    A, B = SyncIterator, AsyncIterator
    rb = [i*2 for i in range(10)]

    for fn1, fn2, fn3, fn4, fn5, fn6, fn7 in product([empty, aempty], repeat=7):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4, fn5=fn5, fn6=fn6, fn7=fn7):
        for iterator in [A, B]:
          with self.subTest(iterator=iterator):
            self.assertEqual(await await_(Chain(iterator).foreach(lambda i: fn1(i*2)).run()), rb)
            for fn1 in [empty, aempty]:
              with self.subTest(fn1=fn1):
                self.assertEqual(await await_(Chain(iterator).then(fn2).foreach(lambda i: fn1(i*2)).run()), rb)
                self.assertEqual(await await_(Chain(fn3).then(iterator, ...).foreach(lambda i: fn1(i*2)).run()), rb)

            rb_break = [i*2 for i in range(9)]
            # test break
            def f(i):
              if i == 9:
                return Chain.break_()
              return fn4(i*2)
            self.assertEqual(await await_(Chain(iterator).foreach(f).run()), rb_break)

            obj = object()
            for break_arg in [(obj,), (lambda: fn5(obj),), (fn6, obj)]:
              with self.subTest(break_arg=break_arg):
                def f(i):
                  if i == 9:
                    return Chain.break_(*break_arg)
                  return fn7(i*2)
                self.assertIs(await await_(Chain(iterator).foreach(f).run()), obj)

  async def test_with(self):
    for fn1, fn2, fn3, fn4, fn5 in product([empty, aempty], repeat=5):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4, fn5=fn5):
        for flx_ctx in [FlexContext, AsyncFlexContext]:
          with self.subTest(flx_ctx=flx_ctx):
            self.assertEqual(await await_(Chain(flx_ctx(v=1)).then(fn1).with_(flx_ctx.get, v=0).run()), 1)
            self.assertEqual(await await_(Chain(flx_ctx(v=1)).then(fn2).then(flx_ctx.get, v=0).run()), 0)
            for fn1 in [empty, aempty]:
              with self.subTest(fn1=fn1):
                self.assertEqual(
                  await await_(
                    Chain(flx_ctx(v=1)).then(fn3)
                    .with_(Chain().then(fn1).then(flx_ctx.get, v=0))
                    .do(lambda: self.assertEqual(flx_ctx.get(v=0), 0), ...)
                    .run()
                  ), 1
                )
                self.assertEqual(
                  await await_(
                    Chain(flx_ctx(v=1)).then(fn4)
                    .with_(lambda flx: flx)
                    .with_(Chain().then(fn1).then(lambda flx: flx.get(v=0)))
                    .run()
                  ), 1
                )
                self.assertEqual(
                  await await_(
                    Chain(flx_ctx(v=1)).then(fn5)
                    .with_(Chain().then(fn1).then(flx_ctx, v=2))
                    .with_(Chain().then(fn1).then(lambda flx: flx.get(v=0)))
                    .run()
                  ), 2
                )

  async def test_null_repr(self):
    """Test __repr__ of _Null class to hit line 29."""
    from quent.quent import PyNull
    result = repr(PyNull)
    self.assertEqual(result, '<Null>')

  async def test_traceback_modification(self):
    """Test traceback modification to hit lines 62, 69-71."""
    def sync_error():
      raise RuntimeError("Sync traceback test")

    async def async_error():
      raise RuntimeError("Async traceback test")

    # Test sync traceback modification (line 62)
    try:
      await await_(Chain(sync_error).run())
      self.assertTrue(False)  # Should not reach here
    except RuntimeError:
      pass  # Expected

    # Test async traceback modification (lines 69-71)
    try:
      await await_(Chain(async_error).run())
      self.assertTrue(False)  # Should not reach here
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
    self.assertEqual(result, "processed_context_value")
    self.assertTrue(cm1.entered)
    self.assertTrue(cm1.exited)

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
      self.assertTrue(False)  # Should not reach here
    except ValueError:
      pass  # Expected

    self.assertTrue(async_cm.entered)
    self.assertTrue(async_cm.exited)
