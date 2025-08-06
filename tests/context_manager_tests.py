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

  async def test_with_do_ignores_body_result(self):
    """with_do(fn) returns the CM object, not fn's result."""
    cm = SimpleCM('val')
    # with_do discards body result; the chain restores previous_value (the CM itself)
    result = Chain(cm).with_do(lambda ctx: 'body_result').run()
    # The outer Link has ignore_result=True, so previous_value (cm) is restored.
    # But _With.__call__ also returns cv_outer when ignore_result=True.
    # The chain's ignore_result on the outer Link means current_value = previous_value = cm.
    await self.assertIs(result, cm)

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

  async def test_async_with_do(self):
    """with_do(fn) on async CM."""
    cm = AsyncCM('val')
    # with_do ignores body result; chain restores previous_value (cm)
    result = await Chain(cm).with_do(lambda ctx: 'body_result').run()
    await self.assertIs(result, cm)


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
    # In with_async: except block calls __exit__ which returns True (exit_result=True),
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

  async def test_sync_cm_async_body_ignore_result(self):
    """with_do with async body returns outer cv."""
    cm = SimpleCM('val')
    async def async_body(ctx):
      return 'ignored'
    # with_do: outer Link has ignore_result=True, chain restores previous_value (cm)
    result = await Chain(cm).with_do(async_body).run()
    await self.assertIs(result, cm)


class WithContextTests(TestCase):

  def test_with_context_returns_self(self):
    """c.with_context(k=v) returns same chain instance."""
    c = Chain(1)
    result = c.with_context(key='value')
    self.assertIs(result, c)

  def test_context_accessible_during_execution(self):
    """Chain.get_context() returns context during run."""
    def check_context(v):
      ctx = Chain.get_context()
      return ctx.get('request_id')
    c = Chain(1).then(check_context).with_context(request_id='abc-123')
    self.assertEqual(c.run(), 'abc-123')

  def test_context_empty_without_with_context(self):
    """Chain.get_context() returns {} outside chain."""
    self.assertEqual(Chain.get_context(), {})

  def test_context_multiple_keys(self):
    """with_context(a=1, b=2) stores both."""
    def check_context(v):
      ctx = Chain.get_context()
      return (ctx.get('a'), ctx.get('b'))
    c = Chain(1).then(check_context).with_context(a=1, b=2)
    self.assertEqual(c.run(), (1, 2))

  def test_context_additive_calls(self):
    """.with_context(a=1).with_context(b=2) merges."""
    def check_context(v):
      ctx = Chain.get_context()
      return (ctx.get('a'), ctx.get('b'))
    c = Chain(1).with_context(a=1).with_context(b=2).then(check_context)
    self.assertEqual(c.run(), (1, 2))

  def test_context_reset_after_sync_run(self):
    """Context cleaned up after sync run."""
    c = Chain(1).then(lambda v: v).with_context(request_id='temp')
    c.run()
    # After run, context should be reset
    self.assertEqual(Chain.get_context(), {})

  def test_context_clone_independent(self):
    """Cloned chain has independent context dict."""
    c1 = Chain(1).with_context(key='original')
    c2 = c1.clone().with_context(key='clone')
    def get_key(v):
      return Chain.get_context().get('key')
    # Add then to c1 after cloning c2
    result1 = c1.then(get_key).run()
    self.assertEqual(result1, 'original')
    # c2's context should be independent
    result2 = c2.then(get_key).run()
    self.assertEqual(result2, 'clone')


class AsyncWithContextTests(MyTestCase):

  async def test_async_context_accessible(self):
    """Context available in async chain operations."""
    async def check_context(v):
      ctx = Chain.get_context()
      return ctx.get('request_id')
    c = Chain(1).then(check_context).with_context(request_id='async-123')
    result = await c.run()
    await self.assertEqual(result, 'async-123')

  async def test_async_context_reset_after_run(self):
    """Context cleaned up after async run."""
    async def noop(v):
      return v
    c = Chain(1).then(noop).with_context(request_id='temp')
    await c.run()
    await self.assertEqual(Chain.get_context(), {})

  async def test_concurrent_contexts_isolated(self):
    """Multiple concurrent chains have isolated contexts."""
    results = {}

    async def capture_first(v):
      ctx = Chain.get_context()
      await asyncio.sleep(0.02)
      results['first'] = ctx.get('id')
      return v

    async def capture_second(v):
      ctx = Chain.get_context()
      await asyncio.sleep(0.01)
      results['second'] = ctx.get('id')
      return v

    c1 = Chain(1).then(capture_first).with_context(id='first')
    c2 = Chain(2).then(capture_second).with_context(id='second')
    await asyncio.gather(c1.run(), c2.run())
    await self.assertEqual(results['first'], 'first')
    await self.assertEqual(results['second'], 'second')

  async def test_async_with_context_exception(self):
    """Exception in async chain with context -> context reset."""
    async def raises(v):
      raise TestExc('async error')
    c = Chain(1).then(raises).with_context(request_id='temp')
    with self.assertRaises(TestExc):
      await c.run()
    # Context should be reset even after exception
    await self.assertEqual(Chain.get_context(), {})


if __name__ == '__main__':
  import unittest
  unittest.main()
