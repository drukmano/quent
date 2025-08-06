import asyncio
import warnings
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SuccessExc(TestExc):
  pass


class BodyExc(TestExc):
  pass


class FinallyExc(TestExc):
  pass


# ---------------------------------------------------------------------------
# MyTestCase
# ---------------------------------------------------------------------------

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
# OnSuccessSyncExceptionTests
# ---------------------------------------------------------------------------

class OnSuccessSyncExceptionTests(MyTestCase):

  async def test_sync_on_success_raising_propagates(self):
    """Sync on_success callback that raises propagates the exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_success_cb(v):
          raise SuccessExc('on_success boom')
        with self.assertRaises(SuccessExc) as cm:
          await await_(
            Chain(fn, 42).on_success(on_success_cb).run()
          )
        super(MyTestCase, self).assertEqual(str(cm.exception), 'on_success boom')

  async def test_sync_on_success_raising_finally_still_runs(self):
    """When sync on_success raises, finally_ callback still executes."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = [False]
        def on_success_cb(v):
          raise SuccessExc('fail in success')
        def finally_cb(v=None):
          finally_called[0] = True
        with self.assertRaises(SuccessExc):
          await await_(
            Chain(fn, 42).on_success(on_success_cb).finally_(finally_cb).run()
          )
        super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_sync_on_success_raising_does_not_alter_chain_value(self):
    """on_success raising does not return any value — exception propagates instead."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_success_cb(v):
          raise SuccessExc('no value')
        result = None
        with self.assertRaises(SuccessExc):
          result = await await_(
            Chain(fn, 'hello').on_success(on_success_cb).run()
          )
        super(MyTestCase, self).assertIsNone(result)

  async def test_sync_on_success_receives_current_value(self):
    """Sync on_success receives the final chain value (current_value)."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        sentinel = object()
        def on_success_cb(v):
          received[0] = v
        await await_(
          Chain(fn, 1).then(sentinel).on_success(on_success_cb).run()
        )
        super(MyTestCase, self).assertIs(received[0], sentinel)

  async def test_sync_on_success_exception_does_not_trigger_except_handler(self):
    """Exception from on_success does not trigger an except_ handler registered on the chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        except_called = [False]
        def on_success_cb(v):
          raise SuccessExc('on_success error')
        def except_handler(v=None):
          except_called[0] = True
        with self.assertRaises(SuccessExc):
          await await_(
            Chain(fn, 42)
            .except_(except_handler, exceptions=SuccessExc)
            .on_success(on_success_cb)
            .run()
          )
        super(MyTestCase, self).assertFalse(except_called[0])


# ---------------------------------------------------------------------------
# OnSuccessAsyncOnSuccessPathTests
# ---------------------------------------------------------------------------

class OnSuccessAsyncOnSuccessPathTests(MyTestCase):

  async def test_async_on_success_succeeds(self):
    """Async on_success callback that succeeds — chain returns its value, not on_success return."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        async def on_success_cb(v):
          called[0] = True
          return 'ignored'
        sentinel = object()
        result = await await_(
          Chain(fn, sentinel).on_success(on_success_cb).run()
        )
        super(MyTestCase, self).assertTrue(called[0])
        super(MyTestCase, self).assertIs(result, sentinel)

  async def test_async_on_success_raises(self):
    """Async on_success that raises — exception propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        async def on_success_cb(v):
          raise SuccessExc('async on_success error')
        with self.assertRaises(SuccessExc) as cm:
          await await_(
            Chain(fn, 42).on_success(on_success_cb).run()
          )
        super(MyTestCase, self).assertEqual(str(cm.exception), 'async on_success error')

  async def test_async_on_success_raises_finally_still_runs(self):
    """Async on_success raises — finally_ still runs via _run_async_on_success."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = [False]
        async def on_success_cb(v):
          raise SuccessExc('async boom')
        def finally_cb(v=None):
          finally_called[0] = True
        with self.assertRaises(SuccessExc):
          await await_(
            Chain(fn, 10).on_success(on_success_cb).finally_(finally_cb).run()
          )
        super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_async_on_success_with_async_finally(self):
    """Both async on_success and async finally_ run in correct order."""
    for fn, ctx in self.with_fn():
      with ctx:
        order = []
        async def on_success_cb(v):
          order.append('success')
        async def finally_cb(v=None):
          order.append('finally')
        await await_(
          Chain(fn, 5).on_success(on_success_cb).finally_(finally_cb).run()
        )
        super(MyTestCase, self).assertEqual(order, ['success', 'finally'])
        order.clear()

  async def test_body_exception_skips_on_success(self):
    """When chain body raises, on_success is NOT called."""
    for fn, ctx in self.with_fn():
      with ctx:
        on_success_called = [False]
        async def on_success_cb(v):
          on_success_called[0] = True
        def raiser(v=None):
          raise BodyExc('body error')
        with self.assertRaises(BodyExc):
          await await_(
            Chain(fn, 1).then(raiser).on_success(on_success_cb).run()
          )
        super(MyTestCase, self).assertFalse(on_success_called[0])

  async def test_body_exception_with_except_noraise_skips_on_success(self):
    """Body exception caught by except_(reraise=False) still skips on_success."""
    for fn, ctx in self.with_fn():
      with ctx:
        on_success_called = [False]
        def on_success_cb(v):
          on_success_called[0] = True
        def raiser(v=None):
          raise BodyExc('caught body error')
        await await_(
          Chain(fn, 1)
          .then(raiser)
          .except_(lambda v: None, reraise=False)
          .on_success(on_success_cb)
          .run()
        )
        super(MyTestCase, self).assertFalse(on_success_called[0])


# ---------------------------------------------------------------------------
# OnSuccessCascadeTests
# ---------------------------------------------------------------------------

class OnSuccessCascadeTests(MyTestCase):

  async def test_cascade_on_success_receives_root_value(self):
    """Cascade: on_success receives the root value, not the last operation result."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        root_sentinel = object()
        def on_success_cb(v):
          received[0] = v
        await await_(
          Cascade(root_sentinel)
          .then(fn)
          .then(lambda v: 'other')
          .on_success(on_success_cb)
          .run()
        )
        super(MyTestCase, self).assertIs(received[0], root_sentinel)

  async def test_cascade_on_success_then_finally_ordering(self):
    """Cascade: on_success runs before finally_."""
    for fn, ctx in self.with_fn():
      with ctx:
        order = []
        sentinel = object()
        def on_success_cb(v):
          order.append('success')
        def finally_cb(v=None):
          order.append('finally')
        await await_(
          Cascade(fn, sentinel)
          .then(lambda v: 'ignored')
          .on_success(on_success_cb)
          .finally_(finally_cb)
          .run()
        )
        super(MyTestCase, self).assertEqual(order, ['success', 'finally'])
        order.clear()

  async def test_cascade_on_success_raising_finally_still_runs(self):
    """Cascade: on_success raises, finally_ still runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = [False]
        def on_success_cb(v):
          raise SuccessExc('cascade on_success error')
        def finally_cb(v=None):
          finally_called[0] = True
        with self.assertRaises(SuccessExc):
          await await_(
            Cascade(fn, 99)
            .then(lambda v: 'work')
            .on_success(on_success_cb)
            .finally_(finally_cb)
            .run()
          )
        super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_void_chain_on_success_called_without_args(self):
    """Void chain (no root value): on_success is called without arguments (current_value is Null)."""
    called = [False]
    arg_count = [None]
    def on_success_cb(*args):
      called[0] = True
      arg_count[0] = len(args)
    # Void chain: Chain() with no root value and no links
    result = Chain().on_success(on_success_cb).run()
    super(MyTestCase, self).assertTrue(called[0])
    # on_success is called with zero args because current_value is the internal Null sentinel
    super(MyTestCase, self).assertEqual(arg_count[0], 0)
    # chain returns None for void chains
    super(MyTestCase, self).assertIsNone(result)

  async def test_cascade_on_success_return_value_is_discarded(self):
    """Cascade: on_success return value is discarded; chain returns root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        result = await await_(
          Cascade(fn, sentinel)
          .then(lambda v: 'work')
          .on_success(lambda v: 'should be discarded')
          .run()
        )
        super(MyTestCase, self).assertIs(result, sentinel)
