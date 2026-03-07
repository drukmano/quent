"""Tests for except_() exception type validation and handler behavior."""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain, Null, QuentException
from quent._core import _ControlFlowSignal, _Return, _Break
from helpers import (
  raise_fn,
  async_raise_fn,
  CustomException,
  NestedCustomException,
  CustomBaseException,
)


# -- Local helpers --

def _catch_handler(rv, exc):
  return f'caught:{type(exc).__name__}'


def _raise_custom(x=None):
  raise CustomException('custom error')


def _raise_nested_custom(x=None):
  raise NestedCustomException('nested custom error')


def _raise_type_error(x=None):
  raise TypeError('type error')


def _raise_keyboard_interrupt(x=None):
  raise KeyboardInterrupt('kbd')


def _raise_system_exit(x=None):
  raise SystemExit(1)


class TestExceptTypeValidation(unittest.TestCase):
  """Validate exception type argument handling in except_()."""

  def test_single_valid_exception_type(self):
    c = Chain().except_(_catch_handler, exceptions=ValueError)
    self.assertEqual(c.on_except_exceptions, (ValueError,))

  def test_multiple_valid_types(self):
    c = Chain().except_(_catch_handler, exceptions=[ValueError, TypeError])
    self.assertEqual(c.on_except_exceptions, (ValueError, TypeError))

  def test_tuple_of_valid_types(self):
    c = Chain().except_(_catch_handler, exceptions=(ValueError, TypeError))
    self.assertEqual(c.on_except_exceptions, (ValueError, TypeError))

  def test_set_of_valid_types(self):
    c = Chain().except_(_catch_handler, exceptions={ValueError, TypeError})
    self.assertIsInstance(c.on_except_exceptions, tuple)
    self.assertEqual(set(c.on_except_exceptions), {ValueError, TypeError})

  def test_generator_of_valid_types(self):
    c = Chain().except_(_catch_handler, exceptions=(t for t in [ValueError]))
    self.assertEqual(c.on_except_exceptions, (ValueError,))

  def test_none_uses_default(self):
    c = Chain().except_(_catch_handler, exceptions=None)
    self.assertEqual(c.on_except_exceptions, (Exception,))

  def test_string_raises_type_error(self):
    with self.assertRaises(TypeError) as cm:
      Chain().except_(_catch_handler, exceptions='ValueError')
    self.assertIn('ValueError', str(cm.exception))
    self.assertIn('string', str(cm.exception).lower())

  def test_empty_list_raises(self):
    with self.assertRaises(QuentException) as cm:
      Chain().except_(_catch_handler, exceptions=[])
    self.assertIn('at least one', str(cm.exception).lower())

  def test_empty_tuple_raises(self):
    with self.assertRaises(QuentException) as cm:
      Chain().except_(_catch_handler, exceptions=())
    self.assertIn('at least one', str(cm.exception).lower())

  def test_int_raises_type_error(self):
    # int is not iterable and not a type subclass of BaseException
    with self.assertRaises(TypeError):
      Chain().except_(_catch_handler, exceptions=42)

  def test_non_exception_class_raises(self):
    # str is a type, but not a subclass of BaseException
    with self.assertRaises(TypeError) as cm:
      Chain().except_(_catch_handler, exceptions=str)
    self.assertIn('BaseException', str(cm.exception))

  def test_base_exception_subclass_accepted(self):
    c = Chain().except_(_catch_handler, exceptions=KeyboardInterrupt)
    self.assertEqual(c.on_except_exceptions, (KeyboardInterrupt,))

  def test_custom_exception_hierarchy(self):
    # CustomException should catch NestedCustomException (its subclass)
    result = Chain(_raise_nested_custom).except_(_catch_handler, exceptions=CustomException).run()
    self.assertEqual(result, 'caught:NestedCustomException')

  def test_mixed_valid_invalid_raises(self):
    with self.assertRaises(TypeError) as cm:
      Chain().except_(_catch_handler, exceptions=[ValueError, 42])
    self.assertIn('42', str(cm.exception))

  def test_exception_catches_only_specified(self):
    # except_(handler, exceptions=TypeError) should NOT catch ValueError
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(_catch_handler, exceptions=TypeError).run()

  def test_exception_catches_subclass(self):
    # except_(handler, exceptions=Exception) should catch ValueError (a subclass)
    result = Chain(raise_fn).except_(_catch_handler, exceptions=Exception).run()
    self.assertEqual(result, 'caught:ValueError')

  def test_double_except_raises(self):
    c = Chain().except_(_catch_handler)
    with self.assertRaises(QuentException) as cm:
      c.except_(_catch_handler)
    self.assertIn('one', str(cm.exception).lower())

  def test_except_message(self):
    with self.assertRaises(TypeError) as cm:
      Chain().except_(_catch_handler, exceptions='RuntimeError')
    msg = str(cm.exception)
    self.assertIn('RuntimeError', msg)
    self.assertIn('string', msg.lower())

  def test_base_exception_as_filter_catches_everything(self):
    # BaseException catches all exceptions including KeyboardInterrupt
    result = Chain(_raise_keyboard_interrupt).except_(_catch_handler, exceptions=BaseException).run()
    self.assertEqual(result, 'caught:KeyboardInterrupt')

  def test_generator_expression_multiple_types(self):
    types = [ValueError, TypeError, RuntimeError]
    c = Chain().except_(_catch_handler, exceptions=(t for t in types))
    self.assertEqual(c.on_except_exceptions, (ValueError, TypeError, RuntimeError))

  def test_frozenset_of_valid_types(self):
    c = Chain().except_(_catch_handler, exceptions=frozenset([ValueError, TypeError]))
    self.assertIsInstance(c.on_except_exceptions, tuple)
    self.assertEqual(set(c.on_except_exceptions), {ValueError, TypeError})

  def test_non_exception_class_in_list_raises(self):
    # list is a type but not BaseException subclass
    with self.assertRaises(TypeError) as cm:
      Chain().except_(_catch_handler, exceptions=[ValueError, list])
    self.assertIn('list', str(cm.exception))

  def test_none_value_in_list_raises(self):
    # None is not a type
    with self.assertRaises(TypeError) as cm:
      Chain().except_(_catch_handler, exceptions=[None])
    self.assertIn('None', str(cm.exception))

  def test_custom_base_exception_accepted(self):
    c = Chain().except_(_catch_handler, exceptions=CustomBaseException)
    self.assertEqual(c.on_except_exceptions, (CustomBaseException,))


class TestExceptHandlerBehavior(unittest.TestCase):
  """Validate how except handlers receive exceptions and return values."""

  def test_handler_receives_exception(self):
    received = []
    def handler(rv, exc):
      received.append((rv, exc))
      return 'ok'
    Chain(raise_fn).except_(handler).run()
    self.assertEqual(len(received), 1)
    self.assertIsNone(received[0][0])
    self.assertIsInstance(received[0][1], ValueError)
    self.assertEqual(str(received[0][1]), 'test error')

  def test_handler_return_value_becomes_result(self):
    result = Chain(raise_fn).except_(lambda rv, e: 42).run()
    self.assertEqual(result, 42)

  def test_handler_return_null_becomes_none(self):
    result = Chain(raise_fn).except_(lambda rv, e: Null).run()
    self.assertIsNone(result)

  def test_handler_raises_chains_exceptions(self):
    def handler(rv, exc):
      raise RuntimeError('handler error')
    with self.assertRaises(RuntimeError) as cm:
      Chain(raise_fn).except_(handler).run()
    # _except_handler_body does `raise exc_ from exc`
    self.assertIsInstance(cm.exception.__cause__, ValueError)
    self.assertEqual(str(cm.exception.__cause__), 'test error')

  def test_control_flow_in_handler_raises(self):
    def handler(rv, exc):
      Chain.return_('val')
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(handler).run()
    self.assertIn('control flow', str(cm.exception).lower())

  def test_base_exception_not_caught_by_default(self):
    # Default exceptions=(Exception,), KeyboardInterrupt is BaseException
    with self.assertRaises(KeyboardInterrupt):
      Chain(_raise_keyboard_interrupt).except_(_catch_handler).run()

  def test_handler_returns_none_explicitly(self):
    result = Chain(raise_fn).except_(lambda rv, e: None).run()
    self.assertIsNone(result)

  def test_handler_returns_complex_object(self):
    result = Chain(raise_fn).except_(lambda rv, e: {'error': str(e), 'type': type(e).__name__}).run()
    self.assertEqual(result, {'error': 'test error', 'type': 'ValueError'})

  def test_handler_that_returns_exception_object(self):
    # Handler returns the exception itself (does not raise it)
    result = Chain(raise_fn).except_(lambda rv, e: e).run()
    self.assertIsInstance(result, ValueError)
    self.assertEqual(str(result), 'test error')

  def test_handler_raises_completely_different_exception(self):
    def handler(rv, exc):
      raise OSError('disk full')
    with self.assertRaises(OSError) as cm:
      Chain(raise_fn).except_(handler).run()
    self.assertEqual(str(cm.exception), 'disk full')
    # Still chained to original
    self.assertIsInstance(cm.exception.__cause__, ValueError)

  def test_exception_with_custom_attributes(self):
    class RichException(Exception):
      def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code

    def raise_rich(x=None):
      raise RichException('rich error', code=42)

    received = []
    def handler(rv, exc):
      received.append((rv, exc))
      return exc.code

    result = Chain(raise_rich).except_(handler).run()
    self.assertEqual(result, 42)
    self.assertEqual(len(received), 1)
    self.assertIsNone(received[0][0])
    self.assertEqual(received[0][1].code, 42)

  def test_system_exit_not_caught_by_default(self):
    # SystemExit is BaseException, not Exception
    with self.assertRaises(SystemExit):
      Chain(_raise_system_exit).except_(_catch_handler).run()

  def test_system_exit_caught_with_base_exception(self):
    result = Chain(_raise_system_exit).except_(
      _catch_handler, exceptions=BaseException
    ).run()
    self.assertEqual(result, 'caught:SystemExit')

  def test_break_in_handler_raises_quent_exception(self):
    def handler(rv, exc):
      Chain.break_()
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(handler).run()
    self.assertIn('control flow', str(cm.exception).lower())

  def test_handler_is_async_returns_coroutine(self):
    # When handler is async and chain is sync, it becomes fire-and-forget
    # or raises QuentException if no event loop. Test the no-loop case.
    async def async_handler(rv, exc):
      return 'async result'
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(async_handler).run()
    self.assertIn('no event loop', str(cm.exception).lower())

  def test_multiple_nested_chains_each_with_except(self):
    def raise_inner(x):
      raise TypeError('inner error')

    inner = Chain().then(raise_inner).except_(lambda rv, e: 'inner_caught')
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 'inner_caught')

  def test_nested_chain_except_does_not_shadow_outer(self):
    def raise_inner(x):
      raise TypeError('inner error')

    # Inner chain catches TypeError
    inner = Chain().then(raise_inner).except_(
      lambda rv, e: 'inner_caught', exceptions=TypeError
    )
    # Outer chain catches ValueError (which inner does not raise)
    outer = Chain(5).then(inner).except_(
      lambda rv, e: 'outer_caught', exceptions=ValueError
    )
    result = outer.run()
    # Inner catches, so outer never sees an exception
    self.assertEqual(result, 'inner_caught')

  def test_nested_chain_exception_propagates_to_outer(self):
    def raise_inner(x):
      raise TypeError('inner error')

    # Inner chain only catches ValueError, not TypeError
    inner = Chain().then(raise_inner).except_(
      lambda rv, e: 'inner_caught', exceptions=ValueError
    )
    # Outer chain catches TypeError
    outer = Chain(5).then(inner).except_(
      lambda rv, e: f'outer_caught:{type(e).__name__}', exceptions=TypeError
    )
    result = outer.run()
    self.assertEqual(result, 'outer_caught:TypeError')


class TestExceptHandlerAsync(unittest.IsolatedAsyncioTestCase):
  """Async handler behavior for except_()."""

  async def test_async_handler_receives_exception(self):
    received = []
    async def handler(rv, exc):
      received.append((rv, exc))
      return 'async_ok'
    result = await Chain(async_raise_fn).except_(handler).run()
    self.assertEqual(result, 'async_ok')
    self.assertEqual(len(received), 1)
    self.assertIsNone(received[0][0])
    self.assertIsInstance(received[0][1], ValueError)

  async def test_async_handler_return_value(self):
    async def handler(rv, exc):
      return {'caught': True}
    result = await Chain(async_raise_fn).except_(handler).run()
    self.assertEqual(result, {'caught': True})

  async def test_async_cancelled_error_with_base_exception_filter(self):
    async def raise_cancelled(x=None):
      raise asyncio.CancelledError('cancelled')

    result = await Chain(raise_cancelled).except_(
      _catch_handler, exceptions=BaseException
    ).run()
    self.assertEqual(result, 'caught:CancelledError')

  async def test_async_cancelled_error_not_caught_by_default(self):
    async def raise_cancelled(x=None):
      raise asyncio.CancelledError('cancelled')

    with self.assertRaises(asyncio.CancelledError):
      await Chain(raise_cancelled).except_(_catch_handler).run()


if __name__ == '__main__':
  unittest.main()
