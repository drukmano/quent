"""Tests for Chain.with_() and Chain.with_do(): sync/async context managers, edge cases."""
from __future__ import annotations

import unittest
from contextlib import contextmanager, asynccontextmanager, nullcontext

from quent import Chain, Null, QuentException
from helpers import (
  SyncCM, AsyncCM, SyncCMSuppresses, AsyncCMSuppresses,
  SyncCMRaisesOnEnter, SyncCMRaisesOnExit, DualCM,
  SyncCMWithAwaitableExit, SyncCMSuppressesAwaitable,
)


class TestWithSync(unittest.TestCase):

  def test_sync_cm_fn_gets_ctx(self):
    result = Chain(SyncCM()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'ctx_value')

  def test_fn_result_becomes_current_value(self):
    result = Chain(SyncCM()).with_(lambda ctx: 42).run()
    self.assertEqual(result, 42)

  def test_exit_called_on_success(self):
    cm = SyncCM()
    Chain(cm).with_(lambda ctx: ctx).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_exit_called_on_exception(self):
    cm = SyncCM()
    with self.assertRaises(ZeroDivisionError):
      Chain(cm).with_(lambda ctx: 1 / 0).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_exit_suppresses_exception(self):
    # SyncCMSuppresses.__exit__ returns True, so the exception is swallowed.
    # When suppressed, the fn didn't return normally so result is None.
    result = Chain(SyncCMSuppresses()).with_(lambda ctx: 1 / 0).run()
    self.assertIsNone(result)

  def test_enter_raises_never_calls_exit(self):
    cm = SyncCMRaisesOnEnter()
    with self.assertRaises(RuntimeError) as ctx:
      Chain(cm).with_(lambda ctx_: ctx_).run()
    self.assertEqual(str(ctx.exception), 'enter error')
    # __exit__ must not have been called (SyncCMRaisesOnEnter has no state
    # tracking, but the contract is that __exit__ is skipped when __enter__ raises).


class TestWithDoSync(unittest.TestCase):

  def test_result_discarded(self):
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 42).run()
    # with_do ignores fn result; returns the CM (the current_value before with_do).
    self.assertIs(result, cm)

  def test_preserves_current_value(self):
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  def test_exit_called(self):
    cm = SyncCM()
    Chain(cm).with_do(lambda ctx: ctx).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


class TestWithAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_cm(self):
    result = await Chain(AsyncCM()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'ctx_value')

  async def test_async_cm_fn_result(self):
    result = await Chain(AsyncCM()).with_(lambda ctx: 42).run()
    self.assertEqual(result, 42)

  async def test_sync_cm_async_body(self):
    # Sync CM with an async fn triggers the _to_async path.
    async def async_body(ctx):
      return ctx + '_async'

    cm = SyncCM()
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'ctx_value_async')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_exit_on_exception(self):
    cm = AsyncCM()
    with self.assertRaises(ZeroDivisionError):
      await Chain(cm).with_(lambda ctx: 1 / 0).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_exit_suppresses(self):
    result = await Chain(AsyncCMSuppresses()).with_(lambda ctx: 1 / 0).run()
    self.assertIsNone(result)


class TestWithDualProtocol(unittest.TestCase):

  def test_enter_preferred_over_aenter(self):
    # DualCM has both __enter__ and __aenter__; __enter__ is now preferred (sync path).
    result = Chain(DualCM()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'sync_ctx')

  def test_only_enter_used_when_no_aenter(self):
    # SyncCM has no __aenter__, so __enter__ is used.
    result = Chain(SyncCM()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'ctx_value')


class TestWithEdgeCases(unittest.TestCase):

  def test_exit_raises(self):
    # SyncCMRaisesOnExit.__exit__ raises RuntimeError('exit error').
    with self.assertRaises(RuntimeError) as ctx:
      Chain(SyncCMRaisesOnExit()).with_(lambda ctx_: ctx_).run()
    self.assertEqual(str(ctx.exception), 'exit error')

  def test_non_cm_raises_type_error(self):
    # 42 has no __enter__/__aenter__, so TypeError is raised.
    with self.assertRaises(TypeError) as ctx:
      Chain(42).with_(lambda ctx_: ctx_).run()
    self.assertIn('does not support the context manager protocol', str(ctx.exception))

  def test_with_fn_explicit_args(self):
    # Explicit positional args override the default ctx passing.
    result = Chain(SyncCM()).with_(lambda a: a, 'custom').run()
    self.assertEqual(result, 'custom')

  def test_with_fn_ellipsis(self):
    # Ellipsis as first arg means "call with no arguments".
    result = Chain(SyncCM()).with_(lambda: 'no_ctx', ...).run()
    self.assertEqual(result, 'no_ctx')


class TestWithDoEdgeCases(unittest.TestCase):

  def test_suppressed_exception_preserves_current_value(self):
    # with_do + suppressing CM: fn raises, suppressed, returns outer_value (the CM).
    cm = SyncCMSuppresses()
    result = Chain(cm).with_do(lambda ctx: 1 / 0).run()
    self.assertIs(result, cm)


class TestWithContextlib(unittest.IsolatedAsyncioTestCase):

  def test_contextmanager_decorator(self):
    @contextmanager
    def my_cm():
      yield 'yielded'

    # @contextmanager returns a ContextDecorator (callable), so wrap in a
    # lambda to prevent Chain from invoking it via its __call__.
    cm = my_cm()
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'yielded')

  async def test_asynccontextmanager_decorator(self):
    @asynccontextmanager
    async def my_acm():
      yield 'async_yielded'

    # Same reasoning: wrap in lambda to avoid ContextDecorator.__call__.
    cm = my_acm()
    result = await Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'async_yielded')

  def test_nullcontext(self):
    # nullcontext has both __enter__ and __aenter__; __enter__ is now preferred (sync path).
    # Wrap in lambda because nullcontext is callable.
    cm = nullcontext('val')
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'val')


class TestWithControlFlowSignalGuard(unittest.IsolatedAsyncioTestCase):
  """Bug 2: _ControlFlowSignal must not leak to __exit__/__aexit__."""

  def test_return_in_sync_with_propagates(self):
    # _Return inside with_ body must propagate past the CM, not be passed to __exit__ as exc_info.
    cm = SyncCMSuppresses()
    result = Chain(cm).with_(lambda ctx: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  async def test_return_in_async_body_sync_cm_propagates(self):
    # _to_async path: sync CM, async body raises _Return.
    cm = SyncCM()

    async def async_body(ctx):
      Chain.return_(42)

    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 42)
    self.assertTrue(cm.exited)

  async def test_return_in_full_async_cm_propagates(self):
    # _full_async path: async CM, body raises _Return.
    from helpers import AsyncCMSuppresses
    result = await Chain(AsyncCMSuppresses()).with_(lambda ctx: Chain.return_(42)).run()
    self.assertEqual(result, 42)


class TestWithAwaitableExit(unittest.IsolatedAsyncioTestCase):
  """Bug 3: sync __exit__ returning an awaitable must be handled."""

  async def test_awaitable_exit_false_does_not_suppress(self):
    # SyncCMWithAwaitableExit.__exit__ returns a coroutine that resolves to False.
    with self.assertRaises(ZeroDivisionError):
      await Chain(SyncCMWithAwaitableExit()).with_(lambda ctx: 1 / 0).run()

  async def test_awaitable_exit_true_suppresses(self):
    # SyncCMSuppressesAwaitable.__exit__ returns a coroutine that resolves to True.
    result = await Chain(SyncCMSuppressesAwaitable()).with_(lambda ctx: 1 / 0).run()
    self.assertIsNone(result)

  async def test_awaitable_exit_success_path(self):
    # Success path: __exit__(None, None, None) returns awaitable. Should be awaited.
    result = await Chain(SyncCMWithAwaitableExit()).with_(lambda ctx: 42).run()
    self.assertEqual(result, 42)


if __name__ == '__main__':
  unittest.main()
