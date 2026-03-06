"""Audit tests for untested gaps in _core.py and _chain.py."""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from quent._core import (
  _Break,
  _Return,
  _handle_break_exc,
  _handle_return_exc,
  _resolve_value,
)


# ---------------------------------------------------------------------------
# 1. _handle_break_exc direct unit tests
# ---------------------------------------------------------------------------

class TestHandleBreakExc(unittest.TestCase):

  def test_null_value_returns_fallback_none(self):
    exc = _Break(Null, (), {})
    result = _handle_break_exc(exc, None)
    self.assertIsNone(result)

  def test_null_value_returns_custom_fallback(self):
    exc = _Break(Null, (), {})
    result = _handle_break_exc(exc, [1, 2, 3])
    self.assertEqual(result, [1, 2, 3])

  def test_plain_value_returned(self):
    exc = _Break(42, (), {})
    result = _handle_break_exc(exc, None)
    self.assertEqual(result, 42)

  def test_callable_value_invoked(self):
    exc = _Break(lambda: 99, (), {})
    result = _handle_break_exc(exc, None)
    self.assertEqual(result, 99)

  def test_callable_with_args(self):
    exc = _Break(lambda x: x + 1, (5,), {})
    result = _handle_break_exc(exc, None)
    self.assertEqual(result, 6)

  def test_null_value_returns_list_fallback(self):
    exc = _Break(Null, (), {})
    result = _handle_break_exc(exc, [1, 2, 3])
    self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# 2. _handle_return_exc direct unit tests
# ---------------------------------------------------------------------------

class TestHandleReturnExc(unittest.TestCase):

  def test_null_value_no_propagate_returns_none(self):
    exc = _Return(Null, (), {})
    result = _handle_return_exc(exc, propagate=False)
    self.assertIsNone(result)

  def test_plain_value_no_propagate(self):
    exc = _Return(42, (), {})
    result = _handle_return_exc(exc, propagate=False)
    self.assertEqual(result, 42)

  def test_callable_value_no_propagate(self):
    exc = _Return(lambda: 99, (), {})
    result = _handle_return_exc(exc, propagate=False)
    self.assertEqual(result, 99)

  def test_null_value_propagate_reraises(self):
    exc = _Return(Null, (), {})
    with self.assertRaises(_Return) as ctx:
      _handle_return_exc(exc, propagate=True)
    self.assertIs(ctx.exception, exc)

  def test_plain_value_propagate_reraises(self):
    exc = _Return(42, (), {})
    with self.assertRaises(_Return) as ctx:
      _handle_return_exc(exc, propagate=True)
    self.assertIs(ctx.exception, exc)


# ---------------------------------------------------------------------------
# 3. Chain.__bool__ always returns True
# ---------------------------------------------------------------------------

class TestChainBool(unittest.TestCase):

  def test_empty_chain(self):
    self.assertTrue(bool(Chain()))

  def test_chain_with_none(self):
    self.assertTrue(bool(Chain(None)))

  def test_chain_with_zero(self):
    self.assertTrue(bool(Chain(0)))

  def test_chain_with_false(self):
    self.assertTrue(bool(Chain(False)))

  def test_frozen_chain(self):
    self.assertTrue(bool(Chain().freeze()))


# ---------------------------------------------------------------------------
# 4. Retry validation tests
# ---------------------------------------------------------------------------

class TestRetryValidation(unittest.TestCase):

  # --- Invalid max_attempts ---

  def test_max_attempts_zero_raises(self):
    with self.assertRaises(QuentException):
      Chain().retry(max_attempts=0)

  def test_max_attempts_negative_raises(self):
    with self.assertRaises(QuentException):
      Chain().retry(max_attempts=-1)

  def test_max_attempts_float_raises(self):
    with self.assertRaises(QuentException):
      Chain().retry(max_attempts=1.5)

  def test_max_attempts_string_raises(self):
    with self.assertRaises(QuentException):
      Chain().retry(max_attempts='3')

  # --- Invalid on ---

  def test_on_string_raises_type_error(self):
    with self.assertRaises(TypeError):
      Chain().retry(on='ValueError')

  def test_on_tuple_with_non_type_raises(self):
    with self.assertRaises(TypeError):
      Chain().retry(on=(ValueError, 'not_a_type'))

  def test_on_empty_tuple_raises(self):
    with self.assertRaises(QuentException):
      Chain().retry(on=())

  def test_on_non_exception_type_raises(self):
    with self.assertRaises(TypeError):
      Chain().retry(on=int)

  # --- Invalid backoff ---

  def test_backoff_negative_raises(self):
    with self.assertRaises(QuentException):
      Chain().retry(backoff=-0.5)

  def test_backoff_string_raises(self):
    with self.assertRaises(TypeError):
      Chain().retry(backoff='fast')

  # --- Duplicate retry ---

  def test_duplicate_retry_raises(self):
    with self.assertRaises(QuentException):
      Chain().retry(1).retry(2)

  # --- Valid cases ---

  def test_valid_single_attempt(self):
    c = Chain().retry(1)
    self.assertEqual(c._retry_max_attempts, 1)

  def test_valid_with_single_exception_type(self):
    c = Chain().retry(3, on=ValueError)
    self.assertEqual(c._retry_max_attempts, 3)
    self.assertEqual(c._retry_on, (ValueError,))

  def test_valid_with_exception_tuple(self):
    c = Chain().retry(3, on=(ValueError, TypeError))
    self.assertEqual(c._retry_max_attempts, 3)
    self.assertEqual(c._retry_on, (ValueError, TypeError))

  def test_valid_with_float_backoff(self):
    c = Chain().retry(3, backoff=0.1)
    self.assertEqual(c._retry_backoff, 0.1)

  def test_valid_with_zero_backoff(self):
    c = Chain().retry(3, backoff=0)
    self.assertEqual(c._retry_backoff, 0)

  def test_valid_with_callable_backoff(self):
    fn = lambda n: n * 0.1
    c = Chain().retry(3, backoff=fn)
    self.assertIs(c._retry_backoff, fn)


# ---------------------------------------------------------------------------
# 5. _resolve_value direct tests
# ---------------------------------------------------------------------------

class TestResolveValue(unittest.TestCase):

  def test_non_callable_returns_as_is(self):
    self.assertEqual(_resolve_value(42, None, None), 42)

  def test_callable_no_args(self):
    self.assertEqual(_resolve_value(lambda: 99, None, None), 99)

  def test_callable_with_positional_args(self):
    self.assertEqual(_resolve_value(lambda x: x + 1, (5,), None), 6)

  def test_callable_with_ellipsis(self):
    self.assertEqual(_resolve_value(lambda: 99, (...,), None), 99)

  def test_callable_with_kwargs_only(self):
    self.assertEqual(_resolve_value(lambda **kw: kw, None, {'a': 1}), {'a': 1})


# ---------------------------------------------------------------------------
# 6. Sync except handler returning Null
# ---------------------------------------------------------------------------

class TestExceptHandlerNullReturn(unittest.TestCase):

  def test_except_handler_null_value_returns_none(self):
    """When except handler is Null (non-callable), chain returns None."""
    c = Chain(lambda: (_ for _ in ()).throw(ValueError('boom'))).except_(Null)
    # Null as a non-callable except handler value: _evaluate_value returns Null,
    # and the chain converts Null -> None.
    result = c.run()
    self.assertIsNone(result)

  def test_except_handler_plain_value_returned(self):
    """When except handler is a plain non-callable value, it is returned."""

    def raiser():
      raise ValueError('boom')

    c = Chain(raiser).except_(42)
    result = c.run()
    self.assertEqual(result, 42)

  def test_except_handler_none_value_returned(self):
    """When except handler is None (non-callable), chain returns None."""

    def raiser():
      raise ValueError('boom')

    c = Chain(raiser).except_(None)
    result = c.run()
    self.assertIsNone(result)


if __name__ == '__main__':
  unittest.main()
