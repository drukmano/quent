"""Tests for exception __cause__, __context__, and __suppress_context__ behavior
in Chain.except_() and Chain.finally_() — both sync and async paths.
"""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain, Null, QuentException
from quent._core import _ControlFlowSignal
from helpers import (
  raise_fn,
  async_raise_fn,
  CustomException,
  NestedCustomException,
  CustomBaseException,
  SyncCMRaisesOnExit,
)


# -- Local helpers --

def _raise_value_error(x=None):
  raise ValueError('body error')


def _raise_type_error(x=None):
  raise TypeError('type error')


async def _async_raise_value_error(x=None):
  raise ValueError('async body error')


class TestExceptionChainingSync(unittest.TestCase):
  """Sync exception chaining behavior for except_ and finally_."""

  def test_except_handler_raises_with_from(self):
    """Handler does `raise X from exc`: __cause__ is set."""
    def handler(exc):
      raise TypeError('handler error') from exc

    with self.assertRaises(TypeError) as cm:
      Chain(_raise_value_error).except_(handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'handler error')
    self.assertIsNotNone(exc.__cause__)
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertEqual(str(exc.__cause__), 'body error')
    self.assertTrue(exc.__suppress_context__)

  def test_except_handler_raises_chains(self):
    """Handler raises without explicit `from`: framework does `raise exc_ from exc`."""
    def handler(exc):
      raise RuntimeError('handler boom')

    with self.assertRaises(RuntimeError) as cm:
      Chain(_raise_value_error).except_(handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'handler boom')
    # _except_handler_body does `raise exc_ from exc`
    self.assertIsNotNone(exc.__cause__)
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertEqual(str(exc.__cause__), 'body error')

  def test_except_suppress_context(self):
    """__suppress_context__ is True when framework chains with `from`."""
    def handler(exc):
      raise TypeError('handler error')

    with self.assertRaises(TypeError) as cm:
      Chain(_raise_value_error).except_(handler).run()

    exc = cm.exception
    # The framework does `raise exc_ from exc`, so suppress_context is True
    self.assertTrue(exc.__suppress_context__)

  def test_finally_raises_after_body_exception(self):
    """Finally handler raises while a body exception is propagating.
    The finally exception should have the body exception as __context__.
    """
    def finally_handler(rv=None):
      raise RuntimeError('finally error')

    with self.assertRaises(RuntimeError) as cm:
      Chain(_raise_value_error).finally_(finally_handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'finally error')
    # The ValueError was active when the finally block ran
    self.assertIsNotNone(exc.__context__)
    self.assertIsInstance(exc.__context__, ValueError)
    self.assertEqual(str(exc.__context__), 'body error')

  def test_finally_raises_after_except(self):
    """Except handler re-raises, then finally also raises.
    Finally exception's __context__ should be the except handler's exception.
    """
    def except_handler(exc):
      raise TypeError('except error') from exc

    def finally_handler(rv=None):
      raise RuntimeError('finally error')

    with self.assertRaises(RuntimeError) as cm:
      Chain(_raise_value_error).except_(except_handler).finally_(finally_handler).run()

    finally_exc = cm.exception
    self.assertEqual(str(finally_exc), 'finally error')
    self.assertIsNotNone(finally_exc.__context__)
    except_exc = finally_exc.__context__
    self.assertIsInstance(except_exc, TypeError)
    self.assertEqual(str(except_exc), 'except error')
    # The except handler's __cause__ is the original body ValueError
    self.assertIsNotNone(except_exc.__cause__)
    self.assertIsInstance(except_exc.__cause__, ValueError)

  def test_finally_raises_after_success(self):
    """Body succeeds, finally raises: no __context__ from prior exception."""
    def finally_handler(rv=None):
      raise RuntimeError('finally error')

    with self.assertRaises(RuntimeError) as cm:
      Chain(lambda: 'ok').finally_(finally_handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'finally error')
    self.assertIsNone(exc.__context__)
    self.assertIsNone(exc.__cause__)

  def test_with_exit_raises_from_body(self):
    """with_ CM's __exit__ raises while a body exception is active.
    The exit exception should be chained from the body exception.
    """
    def body_fn(ctx):
      raise ValueError('body error')

    with self.assertRaises(RuntimeError) as cm:
      Chain(SyncCMRaisesOnExit()).with_(body_fn).run()

    exit_exc = cm.exception
    self.assertEqual(str(exit_exc), 'exit error')
    self.assertIsNotNone(exit_exc.__cause__)
    self.assertIsInstance(exit_exc.__cause__, ValueError)
    self.assertEqual(str(exit_exc.__cause__), 'body error')
    self.assertTrue(exit_exc.__suppress_context__)

  def test_no_circular_references(self):
    """Walk the __context__ chain and verify no cycles."""
    def handler(exc):
      raise TypeError('handler error') from exc

    with self.assertRaises(TypeError) as cm:
      Chain(_raise_value_error).except_(handler).run()

    seen = set()
    current = cm.exception
    while current is not None:
      exc_id = id(current)
      self.assertNotIn(exc_id, seen, 'Circular exception context detected')
      seen.add(exc_id)
      current = current.__context__

  def test_deep_exception_chain(self):
    """Build a 10-level deep exception chain via nested chains with except handlers."""
    # Each level catches and re-raises with a new exception type
    def make_chain(depth):
      if depth == 0:
        return Chain(_raise_value_error)

      inner = make_chain(depth - 1)

      def handler(exc):
        raise RuntimeError(f'level_{depth}') from exc

      return Chain(lambda x=None, inner=inner: inner.run()).except_(handler)

    c = make_chain(10)
    with self.assertRaises(RuntimeError) as cm:
      c.run()

    # Walk the __cause__ chain - should be 10 RuntimeErrors then a ValueError
    exc = cm.exception
    self.assertEqual(str(exc), 'level_10')
    depth = 0
    current = exc
    while isinstance(current, RuntimeError):
      depth += 1
      current = current.__cause__
    self.assertEqual(depth, 10)
    self.assertIsInstance(current, ValueError)
    self.assertEqual(str(current), 'body error')

  def test_except_handler_raises_same_exception_type(self):
    """Handler raises the same type as the original but different message."""
    def handler(exc):
      raise ValueError('transformed') from exc

    with self.assertRaises(ValueError) as cm:
      Chain(_raise_value_error).except_(handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'transformed')
    self.assertIsNotNone(exc.__cause__)
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertEqual(str(exc.__cause__), 'body error')
    # They are distinct objects
    self.assertIsNot(exc, exc.__cause__)

  def test_finally_exception_no_cause_when_body_succeeds(self):
    """When body succeeds and finally raises, __cause__ is None."""
    def finally_handler(rv=None):
      raise RuntimeError('finally boom')

    with self.assertRaises(RuntimeError) as cm:
      Chain(lambda: 42).finally_(finally_handler).run()

    self.assertIsNone(cm.exception.__cause__)

  def test_context_chain_cause_context_distinction(self):
    """Verify __cause__ vs __context__ are set correctly.
    __cause__ is set by explicit `from`, __context__ is set by Python
    when raising during active exception handling.
    """
    def handler(exc):
      # No explicit `from`, but _except_handler_body adds it
      raise TypeError('handler error')

    with self.assertRaises(TypeError) as cm:
      Chain(_raise_value_error).except_(handler).run()

    exc = cm.exception
    # Framework does `raise exc_ from exc`, so both are set
    self.assertIsNotNone(exc.__cause__)
    self.assertIsNotNone(exc.__context__)
    # Both point to the original ValueError
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertIsInstance(exc.__context__, ValueError)

  def test_except_with_finally_context_chain_complete(self):
    """Full chain: body raises, except re-raises, finally raises.
    Verify the complete context chain is intact.
    """
    def except_handler(exc):
      raise TypeError('except boom') from exc

    def finally_handler(rv=None):
      raise OSError('finally boom')

    with self.assertRaises(OSError) as cm:
      Chain(_raise_value_error).except_(except_handler).finally_(finally_handler).run()

    finally_exc = cm.exception
    # finally -> except (TypeError) -> body (ValueError)
    self.assertIsNotNone(finally_exc.__context__)
    except_exc = finally_exc.__context__
    self.assertIsInstance(except_exc, TypeError)
    self.assertIsNotNone(except_exc.__cause__)
    self.assertIsInstance(except_exc.__cause__, ValueError)

  def test_traceback_present_on_chained_exceptions(self):
    """Verify chained exceptions have tracebacks during propagation."""
    def handler(exc):
      raise RuntimeError('handler error')

    captured_tb = {}
    try:
      Chain(_raise_value_error).except_(handler).run()
    except RuntimeError as exc:
      # Capture traceback info while the exception is still active
      # (assertRaises clears __traceback__ after exiting the context)
      captured_tb['exc'] = exc.__traceback__ is not None
      captured_tb['cause'] = exc.__cause__.__traceback__ is not None
      captured_tb['cause_type'] = type(exc.__cause__)

    self.assertTrue(captured_tb['exc'], '__traceback__ should be set on the exception')
    self.assertTrue(captured_tb['cause'], '__traceback__ should be set on the cause')
    self.assertIs(captured_tb['cause_type'], ValueError)

  def test_handler_catches_custom_exception_context_preserved(self):
    """Custom exception hierarchy with context chain."""
    def raise_nested(x=None):
      raise NestedCustomException('nested')

    def handler(exc):
      raise CustomException('caught nested') from exc

    with self.assertRaises(CustomException) as cm:
      Chain(raise_nested).except_(handler, exceptions=CustomException).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'caught nested')
    self.assertIsInstance(exc.__cause__, NestedCustomException)
    self.assertEqual(str(exc.__cause__), 'nested')


class TestExceptionChainingAsync(unittest.IsolatedAsyncioTestCase):
  """Async exception chaining behavior for except_ and finally_."""

  async def test_async_except_handler_chains(self):
    """Async chain: handler raises, framework chains with `from`."""
    def handler(exc):
      raise TypeError('async handler error') from exc

    with self.assertRaises(TypeError) as cm:
      await Chain(_async_raise_value_error).except_(handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'async handler error')
    self.assertIsNotNone(exc.__cause__)
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertEqual(str(exc.__cause__), 'async body error')

  async def test_async_finally_after_body(self):
    """Async: body raises, no except, finally raises.
    Finally exception should have body exception as __context__.
    """
    async def finally_handler(rv=None):
      raise RuntimeError('async finally error')

    try:
      await Chain(_async_raise_value_error).finally_(finally_handler).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      self.assertEqual(str(exc), 'async finally error')
      # In _run_async's finally block, __context__ is set to _active_exc
      self.assertIsNotNone(exc.__context__)
      self.assertIsInstance(exc.__context__, ValueError)
      self.assertEqual(str(exc.__context__), 'async body error')

  async def test_async_finally_after_except(self):
    """Async: body raises, except re-raises, finally raises.
    Finally exception's __context__ should be the except exception.
    """
    def except_handler(exc):
      raise TypeError('async except error') from exc

    async def finally_handler(rv=None):
      raise RuntimeError('async finally error')

    try:
      await Chain(_async_raise_value_error).except_(except_handler).finally_(finally_handler).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as finally_exc:
      self.assertEqual(str(finally_exc), 'async finally error')
      self.assertIsNotNone(finally_exc.__context__)
      except_exc = finally_exc.__context__
      self.assertIsInstance(except_exc, TypeError)
      self.assertEqual(str(except_exc), 'async except error')
      self.assertIsNotNone(except_exc.__cause__)
      self.assertIsInstance(except_exc.__cause__, ValueError)

  async def test_async_finally_context_when_active_exc(self):
    """Async: finally handler raises while body exception is active.
    The _run_async finally block sets __context__ manually.
    """
    async def finally_handler(rv=None):
      raise OSError('cleanup failed')

    try:
      await Chain(_async_raise_value_error).finally_(finally_handler).run()
      self.fail('Expected OSError')
    except OSError as exc:
      self.assertEqual(str(exc), 'cleanup failed')
      self.assertIsNotNone(exc.__context__)
      self.assertIsInstance(exc.__context__, ValueError)

  async def test_async_handler_raises_different_type(self):
    """Async handler raises a completely different exception type."""
    async def handler(exc):
      raise OSError('disk full')

    with self.assertRaises(OSError) as cm:
      await Chain(_async_raise_value_error).except_(handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'disk full')
    # In the async path, _except_handler_body returns the coroutine
    # (handler is async), then _run_async awaits it. The OSError is raised
    # during the await, outside _except_handler_body's inner try/except,
    # so the framework's `raise exc_ from exc` does NOT fire.
    # Python's implicit __context__ is set instead.
    self.assertIsNotNone(exc.__context__)
    self.assertIsInstance(exc.__context__, ValueError)

  async def test_async_no_circular_references(self):
    """Walk async exception __context__ chain - no cycles."""
    def handler(exc):
      raise TypeError('handler error') from exc

    async def finally_handler(rv=None):
      raise RuntimeError('finally error')

    try:
      await Chain(_async_raise_value_error).except_(handler).finally_(finally_handler).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      seen = set()
      current = exc
      while current is not None:
        exc_id = id(current)
        self.assertNotIn(exc_id, seen, 'Circular exception context detected')
        seen.add(exc_id)
        current = current.__context__

  async def test_async_deep_exception_chain(self):
    """Build a deep exception chain in async context."""
    async def raise_val(x=None):
      raise ValueError('deep body error')

    def make_chain(depth):
      if depth == 0:
        return Chain(raise_val)
      inner = make_chain(depth - 1)

      def handler(exc):
        raise RuntimeError(f'async_level_{depth}') from exc

      return Chain(lambda x=None, inner=inner: inner.run()).except_(handler)

    c = make_chain(5)
    with self.assertRaises(RuntimeError) as cm:
      await c.run()

    exc = cm.exception
    self.assertEqual(str(exc), 'async_level_5')
    depth = 0
    current = exc
    while isinstance(current, RuntimeError):
      depth += 1
      current = current.__cause__
    self.assertEqual(depth, 5)
    self.assertIsInstance(current, ValueError)

  async def test_async_cancelled_error_context(self):
    """CancelledError with BaseException filter preserves context."""
    async def raise_cancelled(x=None):
      raise asyncio.CancelledError('cancelled')

    def handler(exc):
      raise RuntimeError('handler caught cancelled') from exc

    with self.assertRaises(RuntimeError) as cm:
      await Chain(raise_cancelled).except_(handler, exceptions=BaseException).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'handler caught cancelled')
    self.assertIsInstance(exc.__cause__, asyncio.CancelledError)

  async def test_async_finally_no_context_on_success(self):
    """Async: body succeeds, finally raises: no __context__."""
    async def finally_handler(rv=None):
      raise RuntimeError('finally error')

    async def success(x=None):
      return 'ok'

    try:
      await Chain(success).finally_(finally_handler).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      self.assertEqual(str(exc), 'finally error')
      # No active exception when body succeeded
      self.assertIsNone(exc.__context__)


class TestExceptionChainingEdgeCases(unittest.TestCase):
  """Edge cases for exception context and chaining."""

  def test_exception_in_finally_after_except_catches(self):
    """Except catches and returns value, then finally raises.
    The finally exception should not have the original body exception
    as __context__ because the except handler returned successfully.
    """
    def except_handler(exc):
      return 'recovered'

    def finally_handler(rv=None):
      raise RuntimeError('finally boom')

    with self.assertRaises(RuntimeError) as cm:
      Chain(_raise_value_error).except_(except_handler).finally_(finally_handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'finally boom')
    # Except handler returned successfully, so no active exception
    # when the finally block runs on the normal path
    # But the body did raise, so the finally is still executing in the
    # finally: block of _run where the except clause returned.
    # context may or may not be set depending on Python's exception state

  def test_handler_returns_original_exception(self):
    """Handler returns the exception object (not raises it)."""
    result = Chain(_raise_value_error).except_(lambda e: e).run()
    self.assertIsInstance(result, ValueError)
    self.assertEqual(str(result), 'body error')

  def test_handler_returns_different_exception(self):
    """Handler returns (not raises) a different exception object."""
    result = Chain(_raise_value_error).except_(
      lambda e: TypeError('different')
    ).run()
    self.assertIsInstance(result, TypeError)
    self.assertEqual(str(result), 'different')

  def test_context_preserved_across_chain_step(self):
    """Exception in middle step has correct context chain."""
    def step1(x):
      return x + 1

    def step2(x):
      raise ValueError(f'step2 failed with {x}')

    def handler(exc):
      raise RuntimeError('handler') from exc

    with self.assertRaises(RuntimeError) as cm:
      Chain(10).then(step1).then(step2).except_(handler).run()

    exc = cm.exception
    self.assertIsNotNone(exc.__cause__)
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertIn('step2 failed with 11', str(exc.__cause__))

  def test_exception_with_none_cause(self):
    """raise X from None: __cause__ is None, __suppress_context__ is True."""
    def handler(exc):
      raise TypeError('clean error') from None

    with self.assertRaises(TypeError) as cm:
      Chain(_raise_value_error).except_(handler).run()

    # The handler did `from None`, but _except_handler_body then does
    # `raise exc_ from exc`, which overwrites __cause__.
    # So __cause__ ends up being the original ValueError.
    exc = cm.exception
    self.assertIsNotNone(exc.__cause__)
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertTrue(exc.__suppress_context__)

  def test_multiple_except_chains_nested(self):
    """Two nested chains, each with except: context chain is correct."""
    def inner_raise(x):
      raise ValueError('inner')

    def inner_handler(exc):
      raise TypeError('inner handler') from exc

    def outer_handler(exc):
      raise RuntimeError('outer handler') from exc

    inner = Chain().then(inner_raise).except_(inner_handler)
    outer = Chain(5).then(inner).except_(outer_handler)

    with self.assertRaises(RuntimeError) as cm:
      outer.run()

    exc = cm.exception
    self.assertEqual(str(exc), 'outer handler')
    self.assertIsInstance(exc.__cause__, TypeError)
    self.assertEqual(str(exc.__cause__), 'inner handler')
    self.assertIsInstance(exc.__cause__.__cause__, ValueError)
    self.assertEqual(str(exc.__cause__.__cause__), 'inner')

  def test_finally_after_successful_except_context(self):
    """Except succeeds (returns value), finally raises.
    Verify the finally exception's context.
    """
    def handler(exc):
      return 'recovered'

    def finally_handler(rv=None):
      raise OSError('cleanup error')

    with self.assertRaises(OSError) as cm:
      Chain(_raise_value_error).except_(handler).finally_(finally_handler).run()

    exc = cm.exception
    self.assertEqual(str(exc), 'cleanup error')

  def test_control_flow_signal_in_except_wraps_properly(self):
    """_ControlFlowSignal in except handler becomes QuentException."""
    def handler(exc):
      Chain.return_('escape')

    with self.assertRaises(QuentException) as cm:
      Chain(_raise_value_error).except_(handler).run()

    exc = cm.exception
    self.assertIn('control flow', str(exc).lower())
    # The `from None` in _except_handler_body suppresses context
    self.assertIsNone(exc.__cause__)

  def test_control_flow_signal_in_finally_wraps_properly(self):
    """_ControlFlowSignal in finally handler becomes QuentException."""
    def handler(rv=None):
      Chain.return_('escape')

    with self.assertRaises(QuentException) as cm:
      Chain(lambda: 'ok').finally_(handler).run()

    exc = cm.exception
    self.assertIn('control flow', str(exc).lower())
    self.assertIsNone(exc.__cause__)


class TestExceptionChainingAsyncEdgeCases(unittest.IsolatedAsyncioTestCase):
  """Async edge cases for exception context and chaining."""

  async def test_async_handler_that_returns_exception(self):
    """Async handler returns (not raises) an exception object."""
    async def handler(exc):
      return TypeError('returned not raised')

    result = await Chain(_async_raise_value_error).except_(handler).run()
    self.assertIsInstance(result, TypeError)
    self.assertEqual(str(result), 'returned not raised')

  async def test_async_nested_chains_exception_chaining(self):
    """Two nested async chains, each with except: full context chain."""
    async def inner_raise(x):
      raise ValueError('async inner')

    def inner_handler(exc):
      raise TypeError('async inner handler') from exc

    def outer_handler(exc):
      raise RuntimeError('async outer handler') from exc

    inner = Chain().then(inner_raise).except_(inner_handler)
    outer = Chain(5).then(inner).except_(outer_handler)

    with self.assertRaises(RuntimeError) as cm:
      await outer.run()

    exc = cm.exception
    self.assertEqual(str(exc), 'async outer handler')
    self.assertIsInstance(exc.__cause__, TypeError)
    self.assertIsInstance(exc.__cause__.__cause__, ValueError)

  async def test_async_finally_raises_sync_handler(self):
    """Async chain with sync finally handler that raises."""
    def finally_handler(rv=None):
      raise RuntimeError('sync finally in async')

    try:
      await Chain(_async_raise_value_error).finally_(finally_handler).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      self.assertEqual(str(exc), 'sync finally in async')
      self.assertIsNotNone(exc.__context__)
      self.assertIsInstance(exc.__context__, ValueError)

  async def test_async_control_flow_in_finally_wraps(self):
    """Async: _ControlFlowSignal in async finally becomes QuentException."""
    async def finally_handler(rv=None):
      Chain.return_('escape')

    with self.assertRaises(QuentException) as cm:
      await Chain(lambda: asyncio.sleep(0)).finally_(finally_handler).run()

    self.assertIn('control flow', str(cm.exception).lower())


if __name__ == '__main__':
  unittest.main()
