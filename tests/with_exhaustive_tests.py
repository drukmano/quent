"""Exhaustive tests for _make_with: all 5 inner functions, 3-tier sync/async coverage."""
from __future__ import annotations

import asyncio
import unittest
from contextlib import asynccontextmanager, contextmanager, nullcontext
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from quent._core import _Return, _Break
from helpers import (
  SyncCM, AsyncCM, SyncCMSuppresses, AsyncCMSuppresses,
  SyncCMRaisesOnEnter, SyncCMRaisesOnExit, DualCM,
  SyncCMWithAwaitableExit, SyncCMSuppressesAwaitable,
  SyncCMExitRaisesOnSuccess, SyncCMExitReturnsAwaitableOnException,
  AsyncCMRaisesOnEnter, AsyncCMRaisesOnExit, TrackingCM, AsyncTrackingCM,
  SyncCMEnterReturnsNone, SyncCMEnterReturnsSelf,
)


# ---------------------------------------------------------------------------
# Additional CM fixtures specific to these tests
# ---------------------------------------------------------------------------

class SyncCMExitRaisesOnException:
  """Sync CM whose __exit__ raises when the body also raises."""
  def __init__(self):
    self.entered = False
    self.exited_called = False
  def __enter__(self):
    self.entered = True
    return 'ctx_value'
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited_called = True
    if exc_type is not None:
      raise RuntimeError('exit error during exception')
    return False


class SyncCMExitReturnsAwaitableTrue:
  """Sync CM whose __exit__ always returns an awaitable that resolves to True."""
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, *args):
    async def _exit():
      return True
    return _exit()


class SyncCMExitReturnsAwaitableOnSuccess:
  """Sync CM whose __exit__ returns an awaitable on success path."""
  def __init__(self):
    self.exited = False
  def __enter__(self):
    return 'ctx_value'
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    async def _exit():
      return None  # result is ignored on success path
    return _exit()


class AsyncCMSlow:
  """Async CM that simulates slow enter/exit."""
  def __init__(self, delay=0.01):
    self.delay = delay
    self.entered = False
    self.exited = False
  async def __aenter__(self):
    await asyncio.sleep(self.delay)
    self.entered = True
    return 'slow_ctx'
  async def __aexit__(self, *args):
    await asyncio.sleep(self.delay)
    self.exited = True
    return False


class SyncCMEnterReturnsFalsy:
  """CM whose __enter__ returns a falsy value (0)."""
  def __enter__(self):
    return 0
  def __exit__(self, *args):
    return False


class SyncCMEnterReturnsFalse:
  """CM whose __enter__ returns False."""
  def __enter__(self):
    return False
  def __exit__(self, *args):
    return False


class SyncCMEnterReturnsEmptyString:
  """CM whose __enter__ returns empty string."""
  def __enter__(self):
    return ''
  def __exit__(self, *args):
    return False


class SyncCMBodyTracker:
  """CM that tracks what __exit__ receives."""
  def __init__(self):
    self.exit_args = None
  def __enter__(self):
    return 'tracked'
  def __exit__(self, *args):
    self.exit_args = args
    return False


class AsyncCMBodyTracker:
  """Async CM that tracks what __aexit__ receives."""
  def __init__(self):
    self.exit_args = None
  async def __aenter__(self):
    return 'async_tracked'
  async def __aexit__(self, *args):
    self.exit_args = args
    return False


# ---------------------------------------------------------------------------
# TestWithOpSyncPaths — _with_op (line 87-118) sync fast path
# ---------------------------------------------------------------------------

class TestWithOpSyncPaths(unittest.TestCase):

  def test_sync_cm_sync_body_success(self):
    """_with_op: sync CM, sync body, no exception."""
    for cm_cls, expected_ctx in [(SyncCM, 'ctx_value'), (TrackingCM, 'tracked_ctx')]:
      with self.subTest(cm_cls=cm_cls.__name__):
        cm = cm_cls()
        result = Chain(cm).with_(lambda ctx: ctx + '_done').run()
        self.assertEqual(result, expected_ctx + '_done')
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)

  def test_sync_cm_sync_body_exception_not_suppressed(self):
    """_with_op: sync CM, body raises, __exit__ does not suppress."""
    for cm_cls in [SyncCM, TrackingCM]:
      with self.subTest(cm_cls=cm_cls.__name__):
        cm = cm_cls()
        with self.assertRaises(ValueError):
          Chain(cm).with_(lambda ctx: (_ for _ in ()).throw(ValueError('boom'))).run()
        self.assertTrue(cm.exited)

  def test_sync_cm_sync_body_exception_suppressed(self):
    """_with_op: body raises, __exit__ returns True -> exception swallowed."""
    result = Chain(SyncCMSuppresses()).with_(lambda ctx: 1 / 0).run()
    self.assertIsNone(result)

  def test_sync_cm_sync_body_control_flow(self):
    """_with_op: _Return in body -> __exit__(None,None,None), signal re-raised."""
    cm = SyncCMBodyTracker()
    result = Chain(cm).with_(lambda ctx: Chain.return_(99)).run()
    self.assertEqual(result, 99)
    # _ControlFlowSignal must NOT leak to __exit__ as exc_info
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_sync_cm_exit_raises_on_success(self):
    """_with_op: body succeeds, __exit__ raises -> propagated."""
    with self.assertRaises(RuntimeError) as ctx:
      Chain(SyncCMExitRaisesOnSuccess()).with_(lambda ctx_: 42).run()
    self.assertEqual(str(ctx.exception), 'exit error on success')

  def test_sync_cm_exit_raises_on_exception(self):
    """_with_op: body raises, __exit__ also raises -> exit_exc from body_exc."""
    cm = SyncCMExitRaisesOnException()
    with self.assertRaises(RuntimeError) as ctx:
      Chain(cm).with_(lambda ctx_: 1 / 0).run()
    self.assertEqual(str(ctx.exception), 'exit error during exception')
    # The body exception should be chained as __cause__
    self.assertIsInstance(ctx.exception.__cause__, ZeroDivisionError)

  def test_sync_cm_enter_raises(self):
    """_with_op: __enter__ raises -> __exit__ never called."""
    with self.assertRaises(RuntimeError) as ctx:
      Chain(SyncCMRaisesOnEnter()).with_(lambda ctx_: ctx_).run()
    self.assertEqual(str(ctx.exception), 'enter error')

  def test_with_do_preserves_outer_value(self):
    """_with_op with ignore_result=True: returns the CM (outer_value)."""
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 'ignored_result').run()
    self.assertIs(result, cm)

  def test_with_do_exception_suppressed_returns_cm(self):
    """_with_op: ignore_result + suppressed exception -> returns outer_value."""
    cm = SyncCMSuppresses()
    result = Chain(cm).with_do(lambda ctx: 1 / 0).run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# TestWithToAsyncPaths — _to_async (line 27-52), sync CM + async body
# ---------------------------------------------------------------------------

class TestWithToAsyncPaths(IsolatedAsyncioTestCase):

  async def test_sync_cm_async_body_success(self):
    """_to_async: sync CM, async body returns normally."""
    cm = SyncCM()
    async def body(ctx):
      return ctx + '_async'
    result = await Chain(cm).with_(body).run()
    self.assertEqual(result, 'ctx_value_async')
    self.assertTrue(cm.exited)

  async def test_sync_cm_async_body_exception_not_suppressed(self):
    """_to_async: async body raises, __exit__ does not suppress."""
    cm = SyncCM()
    async def body(ctx):
      raise ValueError('async boom')
    with self.assertRaises(ValueError):
      await Chain(cm).with_(body).run()
    self.assertTrue(cm.exited)

  async def test_sync_cm_async_body_exception_suppressed(self):
    """_to_async: async body raises, __exit__ returns True -> suppressed."""
    async def body(ctx):
      raise ZeroDivisionError
    result = await Chain(SyncCMSuppresses()).with_(body).run()
    self.assertIsNone(result)

  async def test_sync_cm_async_body_control_flow(self):
    """_to_async: _Return in async body -> __exit__(None,None,None), re-raised."""
    cm = SyncCMBodyTracker()
    async def body(ctx):
      Chain.return_(77)
    result = await Chain(cm).with_(body).run()
    self.assertEqual(result, 77)
    self.assertEqual(cm.exit_args, (None, None, None))

  async def test_sync_cm_async_body_exit_raises(self):
    """_to_async: body raises, __exit__ also raises."""
    cm = SyncCMExitRaisesOnException()
    async def body(ctx):
      raise ValueError('body error')
    with self.assertRaises(RuntimeError) as ctx_:
      await Chain(cm).with_(body).run()
    self.assertEqual(str(ctx_.exception), 'exit error during exception')
    self.assertIsInstance(ctx_.exception.__cause__, ValueError)

  async def test_awaitable_exit_false_on_exception(self):
    """_await_exit_suppress with False: __exit__ returns awaitable resolving False."""
    async def body(ctx):
      raise ZeroDivisionError
    with self.assertRaises(ZeroDivisionError):
      await Chain(SyncCMExitReturnsAwaitableOnException()).with_(body).run()

  async def test_awaitable_exit_true_on_exception(self):
    """_await_exit_suppress with True: __exit__ returns awaitable resolving True."""
    async def body(ctx):
      raise ZeroDivisionError
    result = await Chain(SyncCMExitReturnsAwaitableTrue()).with_(body).run()
    self.assertIsNone(result)

  async def test_awaitable_exit_on_success(self):
    """_await_exit_success path: __exit__ returns awaitable on success."""
    cm = SyncCMExitReturnsAwaitableOnSuccess()
    result = await Chain(cm).with_(lambda ctx: 42).run()
    self.assertEqual(result, 42)
    self.assertTrue(cm.exited)

  async def test_awaitable_exit_success_ignore_result(self):
    """_await_exit_success: ignore_result=True -> returns outer_value (line 84)."""
    cm = SyncCMExitReturnsAwaitableOnSuccess()
    result = await Chain(cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  async def test_with_do_to_async_preserves_outer(self):
    """_to_async with ignore_result: returns outer_value (the CM)."""
    cm = SyncCM()
    async def body(ctx):
      return 'ignored'
    result = await Chain(cm).with_do(body).run()
    self.assertIs(result, cm)

  async def test_sync_cm_exit_returns_awaitable_on_exception(self):
    """_with_op dispatches to _await_exit_suppress when __exit__ returns awaitable."""
    with self.assertRaises(ZeroDivisionError):
      await Chain(SyncCMExitReturnsAwaitableOnException()).with_(lambda ctx: 1 / 0).run()

  async def test_to_async_suppressed_exception_with_do(self):
    """_to_async: ignore_result + suppressed -> returns outer_value."""
    cm = SyncCMSuppresses()
    async def body(ctx):
      raise ZeroDivisionError
    result = await Chain(cm).with_do(body).run()
    self.assertIs(result, cm)

  async def test_to_async_exit_awaitable_on_success(self):
    """_to_async success path with awaitable __exit__."""
    result = await Chain(SyncCMWithAwaitableExit()).with_(lambda ctx: 'body_ok').run()
    self.assertEqual(result, 'body_ok')


# ---------------------------------------------------------------------------
# TestWithFullAsyncPaths — _full_async (line 54-74), async CM
# ---------------------------------------------------------------------------

class TestWithFullAsyncPaths(IsolatedAsyncioTestCase):

  async def test_async_cm_sync_body_success(self):
    """_full_async: async CM, sync body."""
    cm = AsyncCM()
    result = await Chain(cm).with_(lambda ctx: ctx + '_sync').run()
    self.assertEqual(result, 'ctx_value_sync')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm_async_body_success(self):
    """_full_async: async CM, async body."""
    cm = AsyncCM()
    async def body(ctx):
      return ctx + '_async'
    result = await Chain(cm).with_(body).run()
    self.assertEqual(result, 'ctx_value_async')

  async def test_async_cm_body_exception(self):
    """_full_async: body raises, async CM __aexit__ receives it."""
    cm = AsyncCM()
    with self.assertRaises(ValueError):
      await Chain(cm).with_(lambda ctx: (_ for _ in ()).throw(ValueError('err'))).run()
    self.assertTrue(cm.exited)

  async def test_async_cm_body_exception_suppressed(self):
    """_full_async: body raises, __aexit__ returns True -> suppressed."""
    result = await Chain(AsyncCMSuppresses()).with_(lambda ctx: 1 / 0).run()
    self.assertIsNone(result)

  async def test_async_cm_body_control_flow(self):
    """_full_async: _Return in body -> signal stored, re-raised after __aexit__."""
    cm = AsyncCMBodyTracker()
    result = await Chain(cm).with_(lambda ctx: Chain.return_(55)).run()
    self.assertEqual(result, 55)
    # __aexit__ should be called with no exception info (control flow, not real error)
    self.assertEqual(cm.exit_args, (None, None, None))

  async def test_async_cm_body_returns_null(self):
    """_full_async: when result is Null (body doesn't produce a value), returns None."""
    # A body that somehow evaluates to Null (edge case via _evaluate_value)
    # In practice, this happens when the link has a non-callable Null value.
    # We test by using ignore_result=False and a body returning None.
    result = await Chain(AsyncCM()).with_(lambda ctx: None).run()
    self.assertIsNone(result)

  async def test_async_cm_aexit_raises(self):
    """_full_async: __aexit__ raises -> propagated."""
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(AsyncCMRaisesOnExit()).with_(lambda ctx_: 'ok').run()
    self.assertEqual(str(ctx.exception), 'async exit error')

  async def test_async_cm_aenter_raises(self):
    """_full_async: __aenter__ raises -> __aexit__ never called."""
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(AsyncCMRaisesOnEnter()).with_(lambda ctx_: ctx_).run()
    self.assertEqual(str(ctx.exception), 'async enter error')

  def test_dual_protocol_prefers_enter(self):
    """_with_op: DualCM has both protocols -> __enter__ preferred (sync path)."""
    result = Chain(DualCM()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'sync_ctx')

  async def test_with_do_full_async_preserves(self):
    """_full_async with ignore_result: returns outer_value."""
    cm = AsyncCM()
    result = await Chain(cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  async def test_contextlib_contextmanager(self):
    """_with_op: @contextmanager CM (wrapped in lambda)."""
    @contextmanager
    def my_cm():
      yield 'yielded'
    cm = my_cm()
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'yielded')

  async def test_contextlib_asynccontextmanager(self):
    """_full_async: @asynccontextmanager CM."""
    @asynccontextmanager
    async def my_acm():
      yield 'async_yielded'
    cm = my_acm()
    result = await Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'async_yielded')

  def test_contextlib_nullcontext(self):
    """nullcontext has both __enter__ and __aenter__; __enter__ preferred (sync path)."""
    cm = nullcontext('val')
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'val')

  async def test_full_async_control_flow_break(self):
    """_full_async: _Break inside with_ on an async CM."""
    cm = AsyncCMBodyTracker()
    # _Break inside with_ should propagate upward (it's a _ControlFlowSignal)
    with self.assertRaises(QuentException):
      await Chain(cm).with_(lambda ctx: Chain.break_()).run()

  async def test_full_async_ignore_result_with_null_body(self):
    """_full_async: ignore_result + result is Null -> returns outer_value."""
    cm = AsyncCM()
    result = await Chain(cm).with_do(lambda ctx: None).run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# BEYOND SPEC — Additional edge cases
# ---------------------------------------------------------------------------

class TestWithNestedChainBody(IsolatedAsyncioTestCase):
  """with_ where the body is a nested Chain."""

  async def test_nested_chain_as_body_sync(self):
    """Body is a Chain that processes the ctx value."""
    inner = Chain().then(lambda ctx: ctx + '_chained')
    cm = SyncCM()
    result = Chain(cm).with_(inner).run()
    self.assertEqual(result, 'ctx_value_chained')
    self.assertTrue(cm.exited)

  async def test_nested_chain_as_body_async(self):
    """Body is a Chain with async step inside async CM."""
    async def async_step(ctx):
      return ctx + '_async_chained'
    inner = Chain().then(async_step)
    cm = AsyncCM()
    result = await Chain(cm).with_(inner).run()
    self.assertEqual(result, 'ctx_value_async_chained')

  async def test_nested_chain_return_inside_with(self):
    """Nested chain with return_ inside with_."""
    inner = Chain().then(lambda ctx: Chain.return_(ctx + '_returned'))
    cm = SyncCM()
    result = Chain(cm).with_(inner).run()
    self.assertEqual(result, 'ctx_value_returned')


class TestWithCMCreatedByChainStep(IsolatedAsyncioTestCase):
  """with_ where the CM is created by a preceding chain step."""

  def test_cm_from_then_step(self):
    """Chain produces a CM via .then(), then .with_() uses it."""
    result = Chain(lambda: SyncCM()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'ctx_value')

  async def test_cm_from_async_step(self):
    """Chain produces a CM via async step, then .with_()."""
    async def make_cm():
      return AsyncCM()
    result = await Chain(make_cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'ctx_value')


class TestWithExceptInteraction(IsolatedAsyncioTestCase):
  """with_ combined with .except_()."""

  def test_with_body_raises_except_handles(self):
    """Body raises inside with_, except_ catches it."""
    cm = SyncCM()
    result = (
      Chain(cm)
      .with_(lambda ctx: 1 / 0)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertTrue(cm.exited)

  async def test_async_with_body_raises_except_handles(self):
    """Async body raises inside with_, except_ catches it."""
    cm = AsyncCM()
    async def body(ctx):
      raise ValueError('async err')
    result = await (
      Chain(cm)
      .with_(body)
      .except_(lambda rv, exc: 'async_caught')
      .run()
    )
    self.assertEqual(result, 'async_caught')

  def test_suppressed_exception_skips_except(self):
    """CM suppresses exception -> except_ not reached."""
    tracker = []
    result = (
      Chain(SyncCMSuppresses())
      .with_(lambda ctx: 1 / 0)
      .except_(lambda rv, exc: tracker.append('should_not_run'))
      .run()
    )
    self.assertIsNone(result)
    self.assertEqual(tracker, [])


class TestWithDoReturnsNone(unittest.TestCase):
  """with_do where the body fn returns None explicitly."""

  def test_with_do_fn_returns_none(self):
    """Body returns None, with_do ignores it and returns outer_value."""
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: None).run()
    self.assertIs(result, cm)


class TestWithFalsyEnterValues(unittest.TestCase):
  """with_ where __enter__ returns falsy values."""

  def test_enter_returns_zero(self):
    result = Chain(SyncCMEnterReturnsFalsy()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 0)

  def test_enter_returns_false(self):
    result = Chain(SyncCMEnterReturnsFalse()).with_(lambda ctx: ctx).run()
    self.assertFalse(result)
    self.assertIsInstance(result, bool)

  def test_enter_returns_none(self):
    result = Chain(SyncCMEnterReturnsNone()).with_(lambda ctx: ctx).run()
    self.assertIsNone(result)

  def test_enter_returns_empty_string(self):
    result = Chain(SyncCMEnterReturnsEmptyString()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, '')


class TestWithSlowAsyncCM(IsolatedAsyncioTestCase):
  """Async CM with delays in enter/exit."""

  async def test_slow_cm_success(self):
    cm = AsyncCMSlow(delay=0.01)
    result = await Chain(cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'slow_ctx')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_slow_cm_exception(self):
    cm = AsyncCMSlow(delay=0.01)
    with self.assertRaises(ValueError):
      await Chain(cm).with_(lambda ctx: (_ for _ in ()).throw(ValueError('slow_err'))).run()
    self.assertTrue(cm.exited)


class TestWithNestedWith(IsolatedAsyncioTestCase):
  """with_ nested inside another with_."""

  def test_nested_sync_with(self):
    """Outer with_ -> inner with_ (both sync)."""
    outer_cm = SyncCM()
    inner_cm = TrackingCM()
    result = (
      Chain(outer_cm)
      .with_(lambda ctx: inner_cm)
      .with_(lambda ctx: ctx + '_inner')
      .run()
    )
    self.assertEqual(result, 'tracked_ctx_inner')
    self.assertTrue(outer_cm.exited)
    self.assertTrue(inner_cm.exited)

  async def test_nested_async_with(self):
    """Outer async CM -> inner async CM."""
    outer_cm = AsyncCM()
    inner_cm = AsyncTrackingCM()
    result = await (
      Chain(outer_cm)
      .with_(lambda ctx: inner_cm)
      .with_(lambda ctx: ctx + '_inner')
      .run()
    )
    self.assertEqual(result, 'async_tracked_ctx_inner')
    self.assertTrue(outer_cm.exited)
    self.assertTrue(inner_cm.exited)

  async def test_nested_mixed_with(self):
    """Outer sync CM -> inner async CM."""
    outer_cm = SyncCM()
    inner_cm = AsyncCM()
    result = await (
      Chain(outer_cm)
      .with_(lambda ctx: inner_cm)
      .with_(lambda ctx: ctx)
      .run()
    )
    self.assertEqual(result, 'ctx_value')
    self.assertTrue(outer_cm.exited)
    self.assertTrue(inner_cm.exited)


class TestWithPlaceholder(unittest.TestCase):
  """Placeholder for removed frozen chain tests."""
  pass


class TestWithEnterReturnsSelf(unittest.TestCase):
  """CM where __enter__ returns self."""

  def test_enter_returns_self(self):
    cm = SyncCMEnterReturnsSelf()
    result = Chain(cm).with_(lambda ctx: ctx).run()
    self.assertIs(result, cm)


class TestWithDoIgnoreResultConsistency(IsolatedAsyncioTestCase):
  """with_do ignore_result flag across all tiers."""

  def test_sync_with_do_returns_outer(self):
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 'sync_ignored').run()
    self.assertIs(result, cm)

  async def test_to_async_with_do_returns_outer(self):
    cm = SyncCM()
    async def body(ctx):
      return 'async_ignored'
    result = await Chain(cm).with_do(body).run()
    self.assertIs(result, cm)

  async def test_full_async_with_do_returns_outer(self):
    cm = AsyncCM()
    result = await Chain(cm).with_do(lambda ctx: 'full_async_ignored').run()
    self.assertIs(result, cm)


class TestWithChainedAfterWith(unittest.TestCase):
  """Chain steps after with_."""

  def test_then_after_with(self):
    """Step after with_ receives with_ result."""
    result = Chain(SyncCM()).with_(lambda ctx: 10).then(lambda x: x * 2).run()
    self.assertEqual(result, 20)

  def test_do_after_with(self):
    """do after with_ ignores do result, passes with_ result through."""
    tracker = []
    result = (
      Chain(SyncCM())
      .with_(lambda ctx: 10)
      .do(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [10])

  def test_with_do_then_further_step(self):
    """with_do preserves outer value, then is available for next step."""
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 'ignored').then(lambda x: type(x).__name__).run()
    self.assertEqual(result, 'SyncCM')


class TestWithExitReceivesCorrectExcInfo(unittest.TestCase):
  """Verify __exit__ receives correct exception info."""

  def test_exit_receives_exception_type_and_value(self):
    cm = SyncCMBodyTracker()
    try:
      Chain(cm).with_(lambda ctx: 1 / 0).run()
    except ZeroDivisionError:
      pass
    exc_type, exc_val, exc_tb = cm.exit_args
    self.assertIs(exc_type, ZeroDivisionError)
    self.assertIsInstance(exc_val, ZeroDivisionError)
    self.assertIsNotNone(exc_tb)

  def test_exit_receives_none_on_success(self):
    cm = SyncCMBodyTracker()
    Chain(cm).with_(lambda ctx: 'ok').run()
    self.assertEqual(cm.exit_args, (None, None, None))


class TestWithMultipleBodyExceptions(IsolatedAsyncioTestCase):
  """Edge cases around exception handling in body."""

  def test_body_raises_base_exception(self):
    """BaseException (e.g. KeyboardInterrupt) propagated through with_."""
    cm = SyncCM()
    with self.assertRaises(KeyboardInterrupt):
      Chain(cm).with_(lambda ctx: (_ for _ in ()).throw(KeyboardInterrupt())).run()
    self.assertTrue(cm.exited)

  async def test_async_body_raises_base_exception(self):
    """BaseException in async body propagated through _full_async."""
    cm = AsyncCM()
    async def body(ctx):
      raise KeyboardInterrupt()
    with self.assertRaises(KeyboardInterrupt):
      await Chain(cm).with_(body).run()
    self.assertTrue(cm.exited)


if __name__ == '__main__':
  unittest.main()
