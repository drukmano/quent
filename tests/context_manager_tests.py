import asyncio
from unittest import TestCase
from tests.utils import TestExc, empty, aempty, await_, MyTestCase
from quent import Chain, Cascade, QuentException, run


class SimpleCM:
  """Sync context manager for testing."""
  def __init__(self, value='ctx_value', exit_return=False, enter_raise=None, exit_raise=None):
    self.value = value
    self.exit_return = exit_return
    self.enter_raise = enter_raise
    self.exit_raise = exit_raise
    self.entered = False
    self.exited = False
    self.exit_args = None

  def __enter__(self):
    if self.enter_raise:
      raise self.enter_raise
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exit_args = (exc_type, exc_val, exc_tb)
    if self.exit_raise:
      raise self.exit_raise
    return self.exit_return


class AsyncCM:
  """Async context manager for testing."""
  def __init__(self, value='async_ctx', exit_return=False):
    self.value = value
    self.exit_return = exit_return
    self.entered = False
    self.exited = False
    self.exit_args = None

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exit_args = (exc_type, exc_val, exc_tb)
    return self.exit_return


class SyncWithTests(MyTestCase):

  async def test_with_basic(self):
    """Chain(SimpleCM('hello')).with_(lambda ctx: f'got_{ctx}').run() == 'got_hello'"""
    result = Chain(SimpleCM('hello')).with_(lambda ctx: f'got_{ctx}').run()
    await self.assertEqual(result, 'got_hello')

  async def test_with_exit_called(self):
    """CM's __exit__ is called after body."""
    cm = SimpleCM('val')
    Chain(cm).with_(lambda ctx: ctx).run()
    await self.assertTrue(cm.exited)
    # __exit__ should have been called with (None, None, None) on success
    await self.assertEqual(cm.exit_args, (None, None, None))

  async def test_with_enter_called(self):
    """CM's __enter__ is called before body."""
    cm = SimpleCM('val')
    Chain(cm).with_(lambda ctx: ctx).run()
    await self.assertTrue(cm.entered)

  async def test_with_body_exception_calls_exit(self):
    """Body raises -> __exit__ receives exception info."""
    cm = SimpleCM('val')
    exc = TestExc('body error')
    def body_raises(ctx):
      raise exc
    with self.assertRaises(TestExc):
      Chain(cm).with_(body_raises).run()
    await self.assertTrue(cm.exited)
    await self.assertIs(cm.exit_args[0], TestExc)
    await self.assertIs(cm.exit_args[1], exc)

  async def test_with_exit_suppresses_exception(self):
    """__exit__ returns True -> exception suppressed, result is None."""
    cm = SimpleCM('val', exit_return=True)
    def body_raises(ctx):
      raise TestExc('suppressed')
    # When __exit__ returns True, the exception is suppressed.
    # In _With.__call__, the except block doesn't re-raise, then falls through
    # returning None implicitly (the else block with the result doesn't execute).
    result = Chain(cm).with_(body_raises).run()
    await self.assertIsNone(result)
    await self.assertTrue(cm.exited)

  async def test_with_enter_fails_no_exit(self):
    """__enter__ raises -> __exit__ NOT called."""
    exc = TestExc('enter failed')
    cm = SimpleCM('val', enter_raise=exc)
    with self.assertRaises(TestExc):
      Chain(cm).with_(lambda ctx: ctx).run()
    # __exit__ should NOT have been called because __enter__ failed
    await self.assertFalse(cm.exited)

  async def test_with_body_receives_context_value(self):
    """Body receives the value from __enter__."""
    cm = SimpleCM('context_payload')
    received = []
    def body(ctx):
      received.append(ctx)
      return ctx
    Chain(cm).with_(body).run()
    await self.assertEqual(received, ['context_payload'])

  async def test_with_explicit_args(self):
    """with_(fn, arg1, kwarg1=val) passes explicit args to fn instead of ctx."""
    cm = SimpleCM('ctx_val')
    received_args = []
    def body(a, b=None):
      received_args.append((a, b))
      return f'{a}_{b}'
    result = Chain(cm).with_(body, 'explicit_a', b='explicit_b').run()
    await self.assertEqual(result, 'explicit_a_explicit_b')
    await self.assertEqual(received_args, [('explicit_a', 'explicit_b')])

  async def test_with_no_args_sets_temp_args(self):
    """When no explicit args, body receives ctx via temp_args mechanism."""
    cm = SimpleCM('temp_ctx')
    received = []
    def body(ctx):
      received.append(ctx)
      return ctx
    Chain(cm).with_(body).run()
    # Body should have received 'temp_ctx' from __enter__ via temp_args=(ctx,)
    await self.assertEqual(received, ['temp_ctx'])


class AsyncWithTests(MyTestCase):

  async def test_async_with_basic(self):
    """Chain(AsyncCM('hello')).with_(lambda ctx: f'got_{ctx}').run() uses __aenter__/__aexit__."""
    result = await Chain(AsyncCM('hello')).with_(lambda ctx: f'got_{ctx}').run()
    await self.assertEqual(result, 'got_hello')

  async def test_async_with_body_receives_context(self):
    """Async CM body receives the value from __aenter__."""
    cm = AsyncCM('async_payload')
    received = []
    def body(ctx):
      received.append(ctx)
      return ctx
    await Chain(cm).with_(body).run()
    await self.assertEqual(received, ['async_payload'])

  async def test_async_with_exit_called(self):
    """__aexit__ is called."""
    cm = AsyncCM('val')
    await Chain(cm).with_(lambda ctx: ctx).run()
    await self.assertTrue(cm.exited)
    await self.assertEqual(cm.exit_args, (None, None, None))

  async def test_async_with_body_exception(self):
    """Body raises -> __aexit__ called with exception info."""
    cm = AsyncCM('val')
    exc = TestExc('async body error')
    def body_raises(ctx):
      raise exc
    with self.assertRaises(TestExc):
      await Chain(cm).with_(body_raises).run()
    await self.assertTrue(cm.exited)
    await self.assertIs(cm.exit_args[0], TestExc)
    await self.assertIs(cm.exit_args[1], exc)


class SyncWithAsyncBodyTests(MyTestCase):

  async def test_sync_cm_async_body(self):
    """Sync CM, body returns coroutine -> _with_async_fn path."""
    cm = SimpleCM('sync_ctx')
    async def async_body(ctx):
      return f'async_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'async_sync_ctx')
    await self.assertTrue(cm.entered)
    await self.assertTrue(cm.exited)

  async def test_sync_cm_async_body_exception(self):
    """Body coroutine raises -> __exit__ called."""
    cm = SimpleCM('val')
    exc = TestExc('async body error')
    async def async_body_raises(ctx):
      raise exc
    with self.assertRaises(TestExc):
      await Chain(cm).with_(async_body_raises).run()
    await self.assertTrue(cm.exited)
    await self.assertIs(cm.exit_args[0], TestExc)
    await self.assertIs(cm.exit_args[1], exc)

  async def test_sync_cm_async_body_exit_suppresses(self):
    """__exit__ returns True -> async body exception suppressed."""
    cm = SimpleCM('val', exit_return=True)
    async def async_body_raises(ctx):
      raise TestExc('suppressed async')
    # In _with_to_async: except block calls __exit__ which returns True (exit_result=True),
    # so `if not exit_result: raise` doesn't raise. The except block finishes,
    # the else block doesn't execute, function returns None.
    result = await Chain(cm).with_(async_body_raises).run()
    await self.assertIsNone(result)
    await self.assertTrue(cm.exited)

  async def test_sync_cm_async_body_exit_calls_with_none(self):
    """Successful async body -> __exit__(None, None, None)."""
    cm = SimpleCM('val')
    async def async_body(ctx):
      return 'ok'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'ok')
    await self.assertTrue(cm.exited)
    await self.assertEqual(cm.exit_args, (None, None, None))


if __name__ == '__main__':
  import unittest
  unittest.main()
