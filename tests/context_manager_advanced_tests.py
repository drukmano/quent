import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import TestExc, empty, aempty, await_
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


# -- Helper classes ----------------------------------------------------------

class SyncCM:
  """Sync-only context manager."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  """Async-only context manager."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


class DualCM:
  """Has both sync and async context manager protocols."""
  def __init__(self, sync_val='sync', async_val='async'):
    self.sync_val = sync_val
    self.async_val = async_val
    self.sync_entered = False
    self.sync_exited = False
    self.async_entered = False
    self.async_exited = False

  def __enter__(self):
    self.sync_entered = True
    return self.sync_val

  def __exit__(self, *args):
    self.sync_exited = True
    return False

  async def __aenter__(self):
    self.async_entered = True
    return self.async_val

  async def __aexit__(self, *args):
    self.async_exited = True
    return False


class NoneEnterCM:
  """Context manager whose __enter__ returns None."""
  def __init__(self):
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return None

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncNoneEnterCM:
  """Async context manager whose __aenter__ returns None."""
  def __init__(self):
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return None

  async def __aexit__(self, *args):
    self.exited = True
    return False


class CoroExitCM:
  """Sync CM whose __exit__ returns a coroutine (unusual edge case)."""
  def __init__(self, value='coro_exit', suppress=False):
    self.value = value
    self.suppress = suppress
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    async def _exit():
      return self.suppress
    return _exit()


class RaisingBodyCM:
  """Sync CM that tracks enter/exit for exception tests."""
  def __init__(self, value='raise_cm'):
    self.value = value
    self.entered = False
    self.exited = False
    self.exit_exc_type = None

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exit_exc_type = exc_type
    return False


# ---------------------------------------------------------------------------
# DualProtocolCMTests
# ---------------------------------------------------------------------------
class DualProtocolCMTests(MyTestCase):

  async def test_dual_cm_prefers_async_protocol(self):
    """When an object has both __enter__ and __aenter__, quent prefers __aenter__."""
    cm = DualCM(sync_val='sync_v', async_val='async_v')
    result = await Chain(cm).with_(lambda ctx: ctx).run()
    await self.assertEqual(result, 'async_v')

  async def test_dual_cm_async_enter_called(self):
    """DualCM: __aenter__ is called, __enter__ is not."""
    cm = DualCM()
    await Chain(cm).with_(lambda ctx: ctx).run()
    await self.assertTrue(cm.async_entered)
    await self.assertFalse(cm.sync_entered)

  async def test_dual_cm_async_exit_called(self):
    """DualCM: __aexit__ is called, __exit__ is not."""
    cm = DualCM()
    await Chain(cm).with_(lambda ctx: ctx).run()
    await self.assertTrue(cm.async_exited)
    await self.assertFalse(cm.sync_exited)

  async def test_dual_cm_with_do_returns_cm(self):
    """DualCM with with_do: body result discarded, CM returned."""
    cm = DualCM(sync_val='s', async_val='a')
    result = await Chain(cm).with_do(lambda ctx: 'ignored').run()
    await self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# CMEnterReturnsNoneTests
# ---------------------------------------------------------------------------
class CMEnterReturnsNoneTests(MyTestCase):

  async def test_sync_cm_enter_returns_none_body_receives_none(self):
    """When sync __enter__ returns None, body receives None."""
    received = []
    def body(ctx):
      received.append(ctx)
      return 'ok'
    cm = NoneEnterCM()
    result = Chain(cm).with_(body).run()
    await self.assertEqual(result, 'ok')
    super(MyTestCase, self).assertEqual(received, [None])

  async def test_sync_cm_enter_returns_none_exit_called(self):
    """When sync __enter__ returns None, __exit__ is still called."""
    cm = NoneEnterCM()
    Chain(cm).with_(lambda ctx: 'ok').run()
    await self.assertTrue(cm.entered)
    await self.assertTrue(cm.exited)

  async def test_sync_cm_enter_returns_none_chain_continues(self):
    """Chain continues after with_ when __enter__ returns None."""
    cm = NoneEnterCM()
    result = Chain(cm).with_(lambda ctx: 42).then(lambda v: v + 1).run()
    await self.assertEqual(result, 43)

  async def test_async_cm_enter_returns_none_body_receives_none(self):
    """When async __aenter__ returns None, body receives None."""
    received = []
    def body(ctx):
      received.append(ctx)
      return 'async_ok'
    cm = AsyncNoneEnterCM()
    result = await Chain(cm).with_(body).run()
    await self.assertEqual(result, 'async_ok')
    super(MyTestCase, self).assertEqual(received, [None])

  async def test_async_cm_enter_returns_none_exit_called(self):
    """When async __aenter__ returns None, __aexit__ is still called."""
    cm = AsyncNoneEnterCM()
    await Chain(cm).with_(lambda ctx: 'ok').run()
    await self.assertTrue(cm.entered)
    await self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# CMExplicitArgsTests
# ---------------------------------------------------------------------------
class CMExplicitArgsTests(MyTestCase):

  async def test_explicit_args_override_context_value(self):
    """with_(body, 'a', 'b') passes 'a', 'b' to body, not the CM context."""
    received = []
    def body(a, b):
      received.append((a, b))
      return f'{a}+{b}'
    cm = SyncCM(value='should_not_appear')
    result = Chain(cm).with_(body, 'x', 'y').run()
    await self.assertEqual(result, 'x+y')
    super(MyTestCase, self).assertEqual(received, [('x', 'y')])

  async def test_explicit_kwargs_override_context_value(self):
    """with_(body, key='val') passes keyword args, not context."""
    received = []
    def body(key=None):
      received.append(key)
      return key
    cm = SyncCM(value='ignored')
    result = Chain(cm).with_(body, key='override').run()
    await self.assertEqual(result, 'override')
    super(MyTestCase, self).assertEqual(received, ['override'])

  async def test_explicit_args_with_async_cm(self):
    """Explicit args with async CM pass args to body, not __aenter__ value."""
    received = []
    def body(a, b):
      received.append((a, b))
      return f'{a}:{b}'
    cm = AsyncCM(value='async_ignored')
    result = await Chain(cm).with_(body, 'p', 'q').run()
    await self.assertEqual(result, 'p:q')
    super(MyTestCase, self).assertEqual(received, [('p', 'q')])

  async def test_explicit_args_with_do_returns_cm(self):
    """with_do with explicit args: body result discarded, CM returned."""
    cm = SyncCM(value='cm_val')
    result = Chain(cm).with_do(lambda a: a, 'explicit').run()
    await self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# CMNestedChainBodyTests
# ---------------------------------------------------------------------------
class CMNestedChainBodyTests(MyTestCase):

  async def test_body_chain_receives_context(self):
    """Body is a Chain (no root) that receives the CM entered value."""
    cm = SyncCM(value='nested_ctx')
    inner = Chain().then(lambda ctx: f'inner_{ctx}')
    result = Chain(cm).with_(inner).run()
    await self.assertEqual(result, 'inner_nested_ctx')

  async def test_body_chain_raises_calls_exit(self):
    """Body is a Chain that raises; __exit__ still called."""
    cm = RaisingBodyCM(value='ctx')

    def raise_exc(ctx):
      raise TestExc('chain_exc')

    inner = Chain().then(raise_exc)
    with self.assertRaises(TestExc):
      Chain(cm).with_(inner).run()
    await self.assertTrue(cm.exited)
    super(MyTestCase, self).assertEqual(cm.exit_exc_type, TestExc)

  async def test_with_do_body_chain_result_discarded(self):
    """with_do with a Chain body: body result is discarded."""
    cm = SyncCM(value='ctx')
    inner = Chain().then(lambda ctx: 'chain_result')
    result = Chain(cm).with_do(inner).run()
    await self.assertIs(result, cm)

  async def test_nested_with_calls(self):
    """Nested with_ calls: outer CM wraps inner CM."""
    outer_cm = SyncCM(value='outer')
    inner_cm = SyncCM(value='inner')
    result = (
      Chain(outer_cm)
      .with_(lambda ctx: Chain(inner_cm).with_(lambda ictx: f'{ctx}+{ictx}').run())
      .run()
    )
    await self.assertEqual(result, 'outer+inner')
    await self.assertTrue(outer_cm.exited)
    await self.assertTrue(inner_cm.exited)


# ---------------------------------------------------------------------------
# CMAsyncWithEdgeCaseTests
# ---------------------------------------------------------------------------
class CMAsyncWithEdgeCaseTests(MyTestCase):

  async def test_sync_cm_exit_returns_coroutine_success(self):
    """Sync CM whose __exit__ returns a coroutine: coroutine is awaited on success path."""
    cm = CoroExitCM(value='coro_val', suppress=False)
    async def async_body(ctx):
      return f'got_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'got_coro_val')
    await self.assertTrue(cm.exited)

  async def test_sync_cm_exit_returns_coroutine_exception(self):
    """Sync CM whose __exit__ returns a coroutine: exception path awaits it."""
    cm = CoroExitCM(value='coro_val', suppress=True)
    async def async_body_raises(ctx):
      raise TestExc('should be suppressed')
    # __exit__ returns a coroutine that resolves to True -> exception suppressed
    result = await Chain(cm).with_(async_body_raises).run()
    await self.assertIsNone(result)
    await self.assertTrue(cm.exited)

  async def test_async_cm_enter_exit_ordering(self):
    """Verify that __aenter__ is called before body and __aexit__ after."""
    order = []

    class OrderedAsyncCM:
      async def __aenter__(self):
        order.append('enter')
        return 'val'
      async def __aexit__(self, *args):
        order.append('exit')
        return False

    def body(ctx):
      order.append('body')
      return ctx

    cm = OrderedAsyncCM()
    await Chain(cm).with_(body).run()
    super(MyTestCase, self).assertEqual(order, ['enter', 'body', 'exit'])

  async def test_async_cm_body_coroutine_with_do(self):
    """Async body with with_do on async CM: result discarded, CM returned."""
    cm = AsyncCM(value='async_val')
    async def async_body(ctx):
      return 'should_be_ignored'
    result = await Chain(cm).with_do(async_body).run()
    await self.assertIs(result, cm)


if __name__ == '__main__':
  import unittest
  unittest.main()
