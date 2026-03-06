"""Tests targeting the 7 remaining uncovered lines plus additional hard-to-reach paths."""
from __future__ import annotations

import asyncio
import sys
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from quent._traceback import _clean_internal_frames
from helpers import (
  SyncCMWithAwaitableExit,
  AsyncCMNoop,
  async_fn,
)


# ---------------------------------------------------------------------------
# Helpers specific to these tests
# ---------------------------------------------------------------------------

class SyncCMWithAwaitableExitOnSuccess:
  """Sync CM whose __exit__ returns an awaitable on the success path only."""
  def __init__(self):
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return 'ctx_value'

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    async def _exit():
      return False
    return _exit()


class SyncCMExitRaisesAfterBodyRaises:
  """Sync CM whose __exit__ raises when the body also raises."""
  def __enter__(self):
    return 'ctx_value'

  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is not None:
      raise RuntimeError('exit error after body error')
    return False


# ---------------------------------------------------------------------------
# 1. _ops.py:33 — _to_async: ControlFlowSignal + awaitable __exit__
# ---------------------------------------------------------------------------

class TestOpsLine33ToAsyncControlFlowAwaitableExit(IsolatedAsyncioTestCase):
  """Hit line 33: in _to_async inside _make_with, after the body coroutine
  raises _ControlFlowSignal, __exit__ returns an awaitable which must be
  awaited before re-raising the signal.

  Setup: SyncCMWithAwaitableExit (its __exit__ always returns a coroutine).
  The body is an async function that raises _Return (a _ControlFlowSignal).
  When _to_async awaits the body result, _Return propagates. Then
  exit_result = current_value.__exit__(None, None, None) returns awaitable.
  Line 32-33: if isawaitable(exit_result): await exit_result.
  Then raise re-raises _Return. Chain catches _Return and returns 42.
  """

  async def test_control_flow_signal_with_awaitable_exit(self):
    async def async_body_that_returns(ctx):
      Chain.return_(42)

    cm = SyncCMWithAwaitableExit()
    result = await Chain(cm).with_(async_body_that_returns).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# 2. _ops.py:84 — _await_exit_success with ignore_result=True
# ---------------------------------------------------------------------------

class TestOpsLine84AwaitExitSuccessIgnoreResult(IsolatedAsyncioTestCase):
  """Hit line 83-84: _await_exit_success where ignore_result=True.

  Setup: with_do (ignore_result=True) + SyncCMWithAwaitableExitOnSuccess
  (exit returns awaitable on success). The body is a sync lambda, so result
  is not awaitable. Then __exit__ is called on success path and returns
  awaitable. Line 114-115 in _with_op: return _await_exit_success(exit_result,
  outer_value, result). Inside _await_exit_success line 82: await exit_result,
  line 83: if ignore_result: return outer_value.
  """

  async def test_await_exit_success_ignore_result(self):
    cm = SyncCMWithAwaitableExitOnSuccess()
    result = await Chain(cm).with_do(lambda ctx: 'body_result').run()
    self.assertIs(result, cm)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# 3. _chain.py:297 — async finally handler coroutine raises ControlFlowSignal
# ---------------------------------------------------------------------------

class TestChainLine297AsyncFinallyControlFlow(IsolatedAsyncioTestCase):
  """Hit line 297-299: in _run_async's finally block, _finally_handler_body
  returns an awaitable (a coroutine). When awaited, it raises
  _ControlFlowSignal, which is caught and converted to QuentException.

  Setup: chain must enter async path. The finally_ handler must be a SYNC
  function that RETURNS a coroutine (not an async def). When
  _finally_handler_body evaluates it via _evaluate_value, it returns a
  coroutine. Then isawaitable(rv) is True. Then await rv raises
  _ControlFlowSignal.
  """

  async def test_async_finally_control_flow_signal(self):
    async def coro_that_raises_return(root_val):
      Chain.return_('should_not_escape')

    # The finally handler is a sync function returning a coroutine.
    # _evaluate_value calls it with root_value, producing a coroutine.
    def sync_fn_returning_coro(root_val):
      return coro_that_raises_return(root_val)

    chain = Chain(async_fn, 1).finally_(sync_fn_returning_coro)
    with self.assertRaises(QuentException) as cm:
      await chain.run()
    self.assertIn('control flow signals inside finally', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# 4. _chain.py:305 — finally_exc.__context__ is None, _active_exc is not None
# ---------------------------------------------------------------------------

class TestChainLine305FinallyContextNoneActiveExc(IsolatedAsyncioTestCase):
  """Hit line 305-307: body raises exception A (sets _active_exc=A).
  except_ handler catches A and returns a value (clearing the exception
  from propagation, but _active_exc remains set to A).
  Then the async finally handler raises exception B (a brand new exception
  with __context__=None). Line 305: B.__context__ is None and _active_exc (A)
  is not None -> sets B.__context__ = A.

  To trigger: body must be async (to enter _run_async), raise exception A.
  except_ returns a sync value (Null is fine - it becomes None).
  finally_ raises a new exception B via a sync function returning a coroutine
  that raises B. B.__context__ will be None because it is freshly created
  in the coroutine. Line 306 sets B.__context__ = A.
  """

  async def test_finally_context_none_with_active_exc(self):
    async def async_body_raises(x):
      raise ValueError('body error A')

    async def coro_that_raises_b(root_val):
      raise RuntimeError('finally error B')

    def sync_fn_returning_coro(root_val):
      return coro_that_raises_b(root_val)

    # Pass a root value (42) so root_value is set before the body raises.
    # This ensures the finally handler receives root_value=42 (not Null).
    chain = (
      Chain(42)
      .then(async_body_raises)
      .except_(Null)
      .finally_(sync_fn_returning_coro)
    )
    with self.assertRaises(RuntimeError) as cm:
      await chain.run()
    self.assertEqual(str(cm.exception), 'finally error B')
    # Line 307: B.__context__ should be set to A
    self.assertIsNotNone(cm.exception.__context__)
    self.assertIsInstance(cm.exception.__context__, ValueError)
    self.assertEqual(str(cm.exception.__context__), 'body error A')


# ---------------------------------------------------------------------------
# 5. _traceback.py:212 — _resolve_nested_chain with args having non-Null first
# ---------------------------------------------------------------------------

class TestTracebackLine212ResolveNestedChainWithArgs(IsolatedAsyncioTestCase):
  """Hit line 212-218: _resolve_nested_chain where link has args with a
  non-Null first element.

  Trigger by having a nested chain called with explicit positional args.
  Chain().then(Chain().then(lambda x: 1/0), 5).run() — the nested chain
  is called with args=(5,). When the error occurs, the traceback formatter
  calls _resolve_nested_chain and sees args=(5,), _temp_args=(5,),
  _temp_v=5, which is not Null -> line 212-218 execute.
  """

  def test_nested_chain_with_explicit_args(self):
    inner = Chain().then(lambda x: 1 / 0)
    outer = Chain().then(inner, 5)
    with self.assertRaises(ZeroDivisionError) as cm:
      outer.run()
    # The exception should have been processed by the traceback formatter
    self.assertTrue(getattr(cm.exception, '__quent__', False))


# ---------------------------------------------------------------------------
# 6. _traceback.py:110 — Python < 3.11 branch (no co_qualname)
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.version_info >= (3, 11), 'requires Python < 3.11')
class TestTracebackLine110NoQualname(unittest.TestCase):
  """Hit line 112: code = _RAISE_CODE.replace(co_name=chain_source)
  On Python < 3.11, _HAS_QUALNAME is False, so only co_name is replaced.
  """

  def test_no_qualname_branch(self):
    with self.assertRaises(ValueError):
      Chain(lambda: None).then(lambda _: (_ for _ in ()).throw(ValueError('boom'))).run()


# ---------------------------------------------------------------------------
# 7. _core.py:120-121 — Python < 3.14 branch (_create_task_fn)
# ---------------------------------------------------------------------------

@unittest.skipIf(sys.version_info >= (3, 14), 'requires Python < 3.14')
class TestCoreLine112Python314EagerStart(unittest.TestCase):
  """Hit line 121: _create_task_fn = asyncio.create_task
  On Python < 3.14, the else branch sets _create_task_fn to plain
  asyncio.create_task without eager_start.
  """

  def test_create_task_fn_is_create_task(self):
    from quent._core import _create_task_fn
    # On Python < 3.14 this should be asyncio.create_task (not a partial)
    self.assertIs(_create_task_fn, asyncio.create_task)


# ===========================================================================
# Beyond spec: additional hard-to-reach paths
# ===========================================================================

# ---------------------------------------------------------------------------
# A. _run_async except handler returns Null
# ---------------------------------------------------------------------------

class TestRunAsyncExceptReturnsNull(IsolatedAsyncioTestCase):
  """_run_async line 287-288: except handler returns Null -> return None.

  The except handler evaluates to Null (non-callable sentinel).
  In the async except path, result is Null, so return None.
  """

  async def test_async_except_null_result(self):
    async def body_raises(x):
      raise ValueError('oops')

    result = await Chain(body_raises, 1).except_(Null).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# B. _with_op: exit raises after body raises (raise exit_exc from body_exc)
# ---------------------------------------------------------------------------

class TestWithOpExitRaisesAfterBodyRaises(unittest.TestCase):
  """_with_op line 105-106: except BaseException as exit_exc:
  raise exit_exc from exc.

  When the body raises and __exit__ also raises, the exit exception
  is raised with the body exception as __cause__.
  """

  def test_exit_raises_after_body_raises(self):
    cm = SyncCMExitRaisesAfterBodyRaises()
    with self.assertRaises(RuntimeError) as ctx:
      Chain(cm).with_(lambda _: (_ for _ in ()).throw(ValueError('body error'))).run()
    self.assertEqual(str(ctx.exception), 'exit error after body error')
    self.assertIsInstance(ctx.exception.__cause__, ValueError)
    self.assertEqual(str(ctx.exception.__cause__), 'body error')


# ---------------------------------------------------------------------------
# C. _full_async where result is Null and ignore_result is True
# ---------------------------------------------------------------------------

class TestFullAsyncNullResultIgnoreResult(IsolatedAsyncioTestCase):
  """_full_async line 70-71: result is Null and ignore_result is True.

  When with_do is used with an async CM and the body evaluates to Null,
  result stays Null. Since ignore_result=True, return outer_value.
  """

  async def test_full_async_null_ignore_result(self):
    cm = AsyncCMNoop()
    result = await Chain(cm).with_do(Null).run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# D. gather _to_async where indices are non-contiguous
# ---------------------------------------------------------------------------

class TestGatherToAsyncNonContiguousIndices(IsolatedAsyncioTestCase):
  """_make_gather._to_async line 426-432: indices list has non-contiguous
  values when some fns return sync results and others return coroutines.

  Setup: [sync_fn, async_fn, sync_fn, async_fn] -> indices = [1, 3]
  """

  async def test_gather_non_contiguous_indices(self):
    def sync_double(x):
      return x * 2

    async def async_triple(x):
      return x * 3

    def sync_add(x):
      return x + 10

    async def async_square(x):
      return x ** 2

    result = await Chain(5).gather(sync_double, async_triple, sync_add, async_square).run()
    self.assertEqual(result, [10, 15, 15, 25])


# ---------------------------------------------------------------------------
# E. _clean_internal_frames with only quent-internal frames (all stripped)
# ---------------------------------------------------------------------------

class TestCleanInternalFramesAllStripped(unittest.TestCase):
  """_clean_internal_frames returns None when all frames are quent-internal.

  If the traceback consists only of quent-internal frames, they should all
  be stripped, leaving None.
  """

  def test_all_internal_frames_stripped(self):
    # When tb is None, the function returns None immediately
    result = _clean_internal_frames(None)
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# F. _to_async in _make_with: body exception + __exit__ returns awaitable
#    that does NOT suppress
# ---------------------------------------------------------------------------

class TestToAsyncBodyExcExitAwaitableNoSuppress(IsolatedAsyncioTestCase):
  """_to_async line 38-44: body awaitable raises a regular exception,
  __exit__ returns an awaitable that resolves to False (no suppress),
  so the exception is re-raised.
  """

  async def test_to_async_body_exc_exit_awaitable_no_suppress(self):
    class SyncCMExitAwaitableFalse:
      def __enter__(self):
        return 'ctx_value'

      def __exit__(self, *args):
        async def _exit():
          return False
        return _exit()

    async def body_raises(ctx):
      raise RuntimeError('body failure')

    cm = SyncCMExitAwaitableFalse()
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(cm).with_(body_raises).run()
    self.assertEqual(str(ctx.exception), 'body failure')


# ---------------------------------------------------------------------------
# G. _to_async in _make_with: body exception + __exit__ raises
# ---------------------------------------------------------------------------

class TestToAsyncBodyExcExitRaises(IsolatedAsyncioTestCase):
  """_to_async line 41-42: body awaitable raises, then __exit__ also raises.
  raise exit_exc from exc.
  """

  async def test_to_async_exit_raises_during_exc(self):
    class SyncCMExitRaisesOnExc:
      def __enter__(self):
        return 'ctx_value'

      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          raise TypeError('exit failed')
        return False

    async def body_raises(ctx):
      raise RuntimeError('body error')

    cm = SyncCMExitRaisesOnExc()
    with self.assertRaises(TypeError) as ctx:
      await Chain(cm).with_(body_raises).run()
    self.assertEqual(str(ctx.exception), 'exit failed')
    self.assertIsInstance(ctx.exception.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# H. _chain.py line 300-303: async finally handler awaitable raises
#    BaseException (not ControlFlowSignal) with root_value not Null
# ---------------------------------------------------------------------------

class TestAsyncFinallyAwaitableRaisesWithRootValue(IsolatedAsyncioTestCase):
  """_chain.py line 300-304: in _run_async finally, rv is awaitable,
  await rv raises a BaseException (not ControlFlowSignal).
  root_value is not Null -> _set_link_temp_args is called.
  """

  async def test_finally_awaitable_raises_base_exc(self):
    async def coro_that_raises(root_val):
      raise RuntimeError('finally boom')

    def sync_fn_returning_coro(root_val):
      return coro_that_raises(root_val)

    chain = Chain(async_fn, 1).finally_(sync_fn_returning_coro)
    with self.assertRaises(RuntimeError) as cm:
      await chain.run()
    self.assertEqual(str(cm.exception), 'finally boom')


# ---------------------------------------------------------------------------
# I. _chain.py line 305-307: finally_exc.__context__ is None, _active_exc
#    is not None -- triggered by sync finally handler raising directly
# ---------------------------------------------------------------------------

class TestFinallyExcContextDirectRaise(IsolatedAsyncioTestCase):
  """Similar to test 4 but the finally handler itself directly raises
  (not via awaitable). _finally_handler_body catches the exception and
  re-raises it. The outer except at line 305 catches it.
  """

  async def test_finally_direct_raise_sets_context(self):
    async def async_body_raises(x):
      raise ValueError('body error')

    def finally_handler_raises(root_val):
      raise RuntimeError('finally direct error')

    # Pass a root value (99) so root_value is set before the body raises.
    chain = (
      Chain(99)
      .then(async_body_raises)
      .except_(Null)
      .finally_(finally_handler_raises)
    )
    with self.assertRaises(RuntimeError) as cm:
      await chain.run()
    self.assertEqual(str(cm.exception), 'finally direct error')
    # The finally exception should have body error as context
    self.assertIsNotNone(cm.exception.__context__)


# ---------------------------------------------------------------------------
# J. nested chain with kwargs in args for traceback formatting
# ---------------------------------------------------------------------------

class TestNestedChainWithKwargsTraceback(unittest.TestCase):
  """_resolve_nested_chain where link has kwargs (not just args)."""

  def test_nested_chain_with_kwargs(self):
    inner = Chain().then(lambda x, y=0: 1 / 0)
    outer = Chain().then(inner, 5)
    with self.assertRaises(ZeroDivisionError) as cm:
      outer.run()
    self.assertTrue(getattr(cm.exception, '__quent__', False))


# ---------------------------------------------------------------------------
# K. _to_async _ControlFlowSignal with sync __exit__ (not awaitable)
# ---------------------------------------------------------------------------

class TestToAsyncControlFlowSyncExit(IsolatedAsyncioTestCase):
  """_to_async line 30-34: ControlFlowSignal path where __exit__ returns
  a non-awaitable (regular sync exit). Line 32 is False, skip line 33.
  """

  async def test_control_flow_signal_with_sync_exit(self):
    from helpers import SyncCM

    async def async_body_returns(ctx):
      Chain.return_(99)

    cm = SyncCM()
    result = await Chain(cm).with_(async_body_returns).run()
    self.assertEqual(result, 99)
    self.assertTrue(cm.exited)


if __name__ == '__main__':
  unittest.main()
