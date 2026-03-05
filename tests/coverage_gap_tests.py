"""Tests targeting uncovered lines in quent source files.

Each test class focuses on a specific coverage gap identified via analysis
of _chain.py, _ops.py, _traceback.py, and _core.py.
"""
from __future__ import annotations

import asyncio
import traceback
import unittest
import warnings
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

from quent import Chain, Null, QuentException
from quent._core import _ControlFlowSignal, _Return, _Break, Link
from quent._traceback import _get_obj_name, _get_true_source_link, _quent_excepthook
from helpers import (
  async_fn,
  async_identity,
  async_raise_fn,
  raise_fn,
  AsyncCM,
  AsyncCMNoop,
  AsyncRange,
  ObjWithBadName,
  ObjWithBadNameAndRepr,
  SyncCMSuppressesAwaitable,
  SyncCMWithAwaitableExit,
)


# ---------------------------------------------------------------------------
# _chain.py: sync except handler returns Null -> return None (lines 190-191)
# ---------------------------------------------------------------------------

class TestChainSyncExceptReturnsNull(unittest.TestCase):
  """When _evaluate_value on the except handler returns Null (a non-callable
  value), _except_handler_body returns Null, and _run converts it to None.
  """

  def test_sync_except_returns_null(self):
    result = Chain(raise_fn).except_(Null).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _chain.py: _run_async current_value stays Null -> return None (lines 236-237)
# ---------------------------------------------------------------------------

class TestChainAsyncCurrentValueNull(IsolatedAsyncioTestCase):
  """When the only link in _run_async returns the Null sentinel and uses
  ignore_result=True (.do()), current_value is initialized to Null (line 224)
  and never overwritten, so the final check (line 236-237) returns None.
  """

  async def test_async_current_value_stays_null(self):
    async def returns_null():
      return Null

    # .do() sets ignore_result=True; Ellipsis means "call with no args".
    # The awaited result is Null, so current_value = Null on line 224.
    # ignore_result=True skips line 226. No more links. Line 236 is True.
    result = await Chain().do(returns_null, ...).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _chain.py: _Break re-raised when is_nested in _run_async (lines 246-248)
# ---------------------------------------------------------------------------

class TestChainAsyncBreakNested(IsolatedAsyncioTestCase):
  """A _Break raised inside a nested chain's _run_async must re-raise
  when is_nested=True (line 247-248).

  The scenario: a chain with is_nested=True enters _run_async (because
  a link returns an awaitable), then a subsequent link raises _Break.
  The except _Break handler checks self.is_nested and re-raises.

  We call _run() directly (bypassing run() which catches _ControlFlowSignal)
  to mirror what _evaluate_value does for nested chains.
  """

  async def test_async_break_propagates_from_nested(self):
    chain = Chain(async_identity, 1).then(lambda x: Chain.break_())
    chain.is_nested = True

    # _run returns the coroutine from _run_async; await it to execute.
    # _Break should propagate out since is_nested=True.
    with self.assertRaises(_Break):
      await chain._run(Null, None, None)


# ---------------------------------------------------------------------------
# _chain.py: async except handler returns Null -> None (lines 255-256)
# ---------------------------------------------------------------------------

class TestChainAsyncExceptReturnsNull(IsolatedAsyncioTestCase):
  """Same pattern as sync but in the async except path."""

  async def test_async_except_returns_null(self):
    result = await Chain(async_raise_fn).except_(Null).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _chain.py: decorator catches _ControlFlowSignal (lines 274-276)
# ---------------------------------------------------------------------------

class TestChainDecoratorControlFlowSignal(unittest.TestCase):
  """The decorator wrapper should catch any _ControlFlowSignal that leaks
  out of _run and convert it to a QuentException.
  """

  def test_decorator_catches_control_flow_signal(self):
    from unittest.mock import patch

    chain = Chain()

    def mock_run(self_, v, args, kwargs):
      raise _ControlFlowSignal(Null, (), {})

    with patch.object(Chain, '_run', mock_run):
      decorated = chain.decorator()(lambda: None)
      with self.assertRaises(QuentException) as cm:
        decorated()
      self.assertIn('control flow signal', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# _chain.py: run() catches _ControlFlowSignal (lines 286-288)
# ---------------------------------------------------------------------------

class TestChainRunControlFlowSignal(unittest.TestCase):
  """run() should catch any _ControlFlowSignal that leaks out of _run."""

  def test_run_catches_control_flow_signal(self):
    from unittest.mock import patch

    chain = Chain()

    def mock_run(self_arg, v, args, kwargs):
      raise _ControlFlowSignal(Null, (), {})

    with patch.object(Chain, '_run', mock_run):
      with self.assertRaises(QuentException) as cm:
        chain.run()
      self.assertIn('control flow signal', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# _chain.py: _await_run exception path (lines 38-39)
# ---------------------------------------------------------------------------

class TestAwaitRunExceptionPath(IsolatedAsyncioTestCase):
  """When a sync except handler returns a coroutine that raises, the
  _await_run wrapper should modify the traceback before re-raising.
  """

  async def test_await_run_exception_modifies_traceback(self):
    async def async_except_handler(exc):
      raise RuntimeError('handler failed')

    chain = Chain(raise_fn).except_(async_except_handler)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      task = chain.run()
    # task is an asyncio.Task returned from _ensure_future
    with self.assertRaises(RuntimeError) as cm:
      await task
    self.assertEqual(str(cm.exception), 'handler failed')
    self.assertTrue(getattr(cm.exception, '__quent__', False))


# ---------------------------------------------------------------------------
# _ops.py: _make_with._to_async — all paths (lines 26-42)
# ---------------------------------------------------------------------------

class TestWithToAsyncPaths(IsolatedAsyncioTestCase):
  """Test the _to_async inner function of _make_with: sync CM + async body."""

  async def test_to_async_success_with_awaitable_exit(self):
    # SyncCMWithAwaitableExit: __exit__ returns a coroutine (awaitable False)
    cm = SyncCMWithAwaitableExit()
    result = await Chain(cm).with_(async_identity).run()
    self.assertEqual(result, 'ctx_value')

  async def test_to_async_exception_with_awaitable_exit_suppresses(self):
    # SyncCMSuppressesAwaitable: __exit__ returns awaitable True -> suppresses
    cm = SyncCMSuppressesAwaitable()
    result = await Chain(cm).with_(async_raise_fn).run()
    self.assertIsNone(result)

  async def test_to_async_exception_with_awaitable_exit_no_suppress(self):
    # SyncCMWithAwaitableExit: __exit__ returns awaitable False -> does NOT suppress
    cm = SyncCMWithAwaitableExit()
    with self.assertRaises(ValueError):
      await Chain(cm).with_(async_raise_fn).run()

  async def test_to_async_ignore_result_success(self):
    # with_do + sync CM + async body -> should return the CM (outer_value)
    cm = SyncCMWithAwaitableExit()
    result = await Chain(cm).with_do(async_identity).run()
    self.assertIs(result, cm)

  async def test_to_async_ignore_result_suppressed(self):
    # with_do + SyncCMSuppressesAwaitable + async raise -> suppressed, return outer_value
    cm = SyncCMSuppressesAwaitable()
    result = await Chain(cm).with_do(async_raise_fn).run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# _ops.py: _make_with._full_async — all paths (lines 44-55)
# ---------------------------------------------------------------------------

class TestWithFullAsyncPaths(IsolatedAsyncioTestCase):
  """Test the _full_async inner function of _make_with: async CM paths."""

  async def test_full_async_result_is_null(self):
    # When body evaluates to Null (not callable), _evaluate_value returns Null.
    # result is Null -> return None (line 51-52).
    result = await Chain(AsyncCMNoop()).with_(Null).run()
    self.assertIsNone(result)

  async def test_full_async_ignore_result_null(self):
    # with_do + AsyncCMNoop + Null body -> returns outer_value (line 52)
    cm = AsyncCMNoop()
    result = await Chain(cm).with_do(Null).run()
    self.assertIs(result, cm)

  async def test_full_async_ignore_result_with_real_body(self):
    # with_do + AsyncCM + real body -> returns outer_value (line 53-54)
    cm = AsyncCM()
    result = await Chain(cm).with_do(lambda ctx: 42).run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# _ops.py: _make_foreach._to_async — Return signal & exception paths
# (lines 200-204)
# ---------------------------------------------------------------------------

class TestForeachToAsyncPaths(IsolatedAsyncioTestCase):
  """Test _to_async paths inside _make_foreach: _Return propagation,
  BaseException with temp args, and _Break with async value.
  """

  async def test_to_async_return_signal(self):
    # Sync iterable, fn returns coroutine on first item -> enters _to_async.
    # Then fn raises _Return on second item -> _ControlFlowSignal propagates.
    call_count = 0

    async def fn_with_return(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        Chain.return_('early')
      return x

    result = await Chain([1, 2, 3]).foreach(fn_with_return).run()
    self.assertEqual(result, 'early')

  async def test_to_async_exception_with_temp_args(self):
    # Sync iterable, fn returns coroutine then raises on later item.
    call_count = 0

    async def fn_raises_later(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        raise RuntimeError('boom')
      return x

    try:
      await Chain([10, 20, 30]).foreach(fn_raises_later).run()
      self.fail('Should have raised')
    except RuntimeError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (20,))

  async def test_to_async_break_with_async_value(self):
    # Sync iterable + fn returns coroutine, then Break with async value.
    async def make_val():
      return 'break_val'

    call_count = 0

    async def fn_breaks(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        Chain.break_(make_val)
      return x

    result = await Chain([1, 2, 3]).foreach(fn_breaks).run()
    self.assertEqual(result, 'break_val')


# ---------------------------------------------------------------------------
# _ops.py: _make_foreach._full_async — Return signal & exception paths
# (lines 224-228)
# ---------------------------------------------------------------------------

class TestForeachFullAsyncPaths(IsolatedAsyncioTestCase):
  """Test _full_async paths inside _make_foreach: async iterable with
  _Return, BaseException temp args, and _Break with async value.
  """

  async def test_full_async_return_signal(self):
    call_count = 0

    def fn_returns(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        Chain.return_('early')
      return x

    result = await Chain(AsyncRange(5)).foreach(fn_returns).run()
    self.assertEqual(result, 'early')

  async def test_full_async_exception_with_temp_args(self):
    def fn_raises(x):
      if x == 1:
        raise RuntimeError('boom')
      return x

    try:
      await Chain(AsyncRange(3)).foreach(fn_raises).run()
      self.fail('Should have raised')
    except RuntimeError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (1,))

  async def test_full_async_break_with_async_value(self):
    async def make_val():
      return 'async_val'

    def fn_breaks(x):
      if x == 1:
        Chain.break_(make_val)
      return x

    result = await Chain(AsyncRange(5)).foreach(fn_breaks).run()
    self.assertEqual(result, 'async_val')


# ---------------------------------------------------------------------------
# _ops.py: _make_filter._to_async — ControlFlowSignal & exception paths
# (lines 278-282)
# ---------------------------------------------------------------------------

class TestFilterToAsyncPaths(IsolatedAsyncioTestCase):
  """Test _to_async paths inside _make_filter: _ControlFlowSignal
  propagation and BaseException with temp args.
  """

  async def test_to_async_control_flow_signal(self):
    call_count = 0

    async def pred(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        Chain.return_('early')
      return x > 0

    result = await Chain([1, 2, 3]).filter(pred).run()
    self.assertEqual(result, 'early')

  async def test_to_async_exception_with_temp_args(self):
    call_count = 0

    async def pred(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        raise RuntimeError('filter boom')
      return True

    try:
      await Chain([10, 20, 30]).filter(pred).run()
      self.fail('Should have raised')
    except RuntimeError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (20,))


# ---------------------------------------------------------------------------
# _ops.py: _make_filter._full_async — ControlFlowSignal & exception paths
# (lines 295-299)
# ---------------------------------------------------------------------------

class TestFilterFullAsyncPaths(IsolatedAsyncioTestCase):
  """Test _full_async paths inside _make_filter: async iterable with
  _ControlFlowSignal and BaseException temp args.
  """

  async def test_full_async_control_flow_signal(self):
    call_count = 0

    def pred(x):
      nonlocal call_count
      call_count += 1
      if call_count >= 2:
        Chain.return_('early')
      return True

    result = await Chain(AsyncRange(5)).filter(pred).run()
    self.assertEqual(result, 'early')

  async def test_full_async_exception_with_temp_args(self):
    def pred(x):
      if x == 1:
        raise RuntimeError('filter boom')
      return True

    try:
      await Chain(AsyncRange(3)).filter(pred).run()
      self.fail('Should have raised')
    except RuntimeError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (1,))


# ---------------------------------------------------------------------------
# _ops.py: _make_gather coroutine cleanup on exception (lines 351-357)
# ---------------------------------------------------------------------------

class TestGatherCoroutineCleanup(IsolatedAsyncioTestCase):
  """When a fn raises during gather setup, already-created coroutines
  must be closed to avoid ResourceWarning.
  """

  async def test_gather_cleans_coroutines_on_exception(self):
    async def coro_fn(x):
      return x

    def raise_fn_setup(x):
      raise RuntimeError('setup error')

    # coro_fn creates a coroutine, then raise_fn_setup raises.
    # The gather should close the already-created coroutine.
    with warnings.catch_warnings():
      warnings.simplefilter('error', RuntimeWarning)
      with self.assertRaises(RuntimeError) as cm:
        Chain(1).gather(coro_fn, raise_fn_setup).run()
      self.assertEqual(str(cm.exception), 'setup error')


# ---------------------------------------------------------------------------
# _traceback.py: _get_true_source_link (lines 122-141)
# ---------------------------------------------------------------------------

class TestTracebackGetTrueSourceLink(unittest.TestCase):
  """Test _get_true_source_link drills through chain links and falls back."""

  def test_drills_through_chain_link(self):
    # link.is_chain = True -> drills into inner chain's root_link
    inner_chain = Chain(lambda: 42)
    link = Link(inner_chain)  # link.is_chain = True
    result = _get_true_source_link(link, None)
    self.assertIs(result, inner_chain.root_link)

  def test_drills_through_original_value_chain(self):
    # link.original_value._is_chain -> drills into that chain's root_link
    inner_chain = Chain(lambda: 42)
    link = Link(lambda: None, original_value=inner_chain)
    result = _get_true_source_link(link, None)
    self.assertIs(result, inner_chain.root_link)

  def test_fallback_to_root_link(self):
    # source_link is None -> returns root_link fallback
    fallback = Link(lambda: 'fallback')
    result = _get_true_source_link(None, fallback)
    self.assertIs(result, fallback)

  def test_chain_without_root_link_stops(self):
    # Chain with no root (v=Null) -> root_link is None -> breaks out of loop
    inner_chain = Chain()  # root_link is None
    link = Link(inner_chain)
    result = _get_true_source_link(link, None)
    # Should return the link itself since it cannot drill further
    self.assertIs(result, link)


# ---------------------------------------------------------------------------
# _traceback.py: _get_obj_name exception handling (lines 174-181)
# ---------------------------------------------------------------------------

class TestTracebackGetObjName(unittest.TestCase):
  """Test _get_obj_name fallback paths when __name__/__qualname__/repr fail."""

  def test_bad_name_falls_back_to_repr(self):
    obj = ObjWithBadName()
    result = _get_obj_name(obj)
    self.assertEqual(result, '<ObjWithBadName>')

  def test_bad_name_and_repr_falls_back_to_type_name(self):
    obj = ObjWithBadNameAndRepr()
    result = _get_obj_name(obj)
    self.assertEqual(result, 'ObjWithBadNameAndRepr')


# ---------------------------------------------------------------------------
# _traceback.py: _quent_excepthook (lines 324-331)
# ---------------------------------------------------------------------------

class TestTracebackExcepthook(unittest.TestCase):
  """Test the custom excepthook cleans quent frames."""

  def test_excepthook_cleans_quent_frames(self):
    import quent._traceback as tb_mod

    mock_hook = MagicMock()
    original = tb_mod._original_excepthook
    tb_mod._original_excepthook = mock_hook
    try:
      exc = ValueError('test')
      exc.__quent__ = True
      _quent_excepthook(ValueError, exc, None)
      mock_hook.assert_called_once()
      call_args = mock_hook.call_args
      self.assertIs(call_args[0][1], exc)
    finally:
      tb_mod._original_excepthook = original

  def test_excepthook_passes_through_non_quent(self):
    import quent._traceback as tb_mod

    mock_hook = MagicMock()
    original = tb_mod._original_excepthook
    tb_mod._original_excepthook = mock_hook
    try:
      exc = ValueError('non-quent')
      _quent_excepthook(ValueError, exc, None)
      mock_hook.assert_called_once_with(ValueError, exc, None)
    finally:
      tb_mod._original_excepthook = original


# ---------------------------------------------------------------------------
# _traceback.py: _patched_te_init (lines 351-354)
# ---------------------------------------------------------------------------

class TestTracebackPatchedTeInit(unittest.TestCase):
  """Test the patched TracebackException.__init__ cleans quent frames."""

  def test_patched_te_init_cleans_quent_frames(self):
    exc = ValueError('test')
    exc.__quent__ = True
    # Construct TracebackException — should trigger patched __init__
    te = traceback.TracebackException(ValueError, exc, None)
    self.assertIsNotNone(te)

  def test_patched_te_init_passes_through_non_quent(self):
    exc = ValueError('non-quent')
    te = traceback.TracebackException(ValueError, exc, None)
    self.assertIsNotNone(te)


if __name__ == '__main__':
  unittest.main()
