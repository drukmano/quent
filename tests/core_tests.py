"""Tests for quent._core: Null, QuentException, Link, _resolve_value, _evaluate_value."""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from quent._core import Link, _evaluate_value, _resolve_value, _Null, _Return, _Break, _ControlFlowSignal
from helpers import sync_fn, async_fn


def _make_tracker():
  """Return a callable that records how it was called and its return value."""
  calls = []
  def fn(*args, **kwargs):
    calls.append((args, kwargs))
    return 'ok'
  fn.calls = calls
  return fn


class TestNull(unittest.TestCase):

  def test_null_is_singleton(self):
    from quent._core import Null as Null2
    self.assertIs(Null, Null2)

  def test_null_repr(self):
    self.assertEqual(repr(Null), '<Null>')

  def test_null_is_not_none(self):
    self.assertIsNot(Null, None)
    self.assertNotEqual(Null, None)

  def test_null_is_not_false(self):
    self.assertIsNot(Null, False)

  def test_null_distinct_from_all_falsy(self):
    falsy_values = [0, '', [], {}, set(), 0.0, b'', (), frozenset()]
    for val in falsy_values:
      self.assertIsNot(Null, val, f'Null should not be {val!r}')


class TestQuentException(unittest.TestCase):

  def test_is_exception_subclass(self):
    self.assertTrue(issubclass(QuentException, Exception))

  def test_can_be_raised_and_caught(self):
    with self.assertRaises(QuentException):
      raise QuentException('boom')

  def test_has_message(self):
    exc = QuentException('msg')
    self.assertEqual(exc.args[0], 'msg')


class TestResolveValue(unittest.TestCase):

  def test_callable_no_args_no_current(self):
    result = _resolve_value(lambda: 42, None, None)
    self.assertEqual(result, 42)

  def test_callable_with_explicit_args(self):
    result = _resolve_value(lambda x, y: x + y, (1, 2), None)
    self.assertEqual(result, 3)

  def test_callable_with_ellipsis(self):
    result = _resolve_value(lambda: 99, (...,), None)
    self.assertEqual(result, 99)

  def test_non_callable_returned_as_is(self):
    result = _resolve_value(42, None, None)
    self.assertEqual(result, 42)

  def test_callable_with_kwargs_only(self):
    result = _resolve_value(lambda x=5: x, None, {'x': 10})
    self.assertEqual(result, 10)


class TestLink(unittest.TestCase):

  def test_creation_basic(self):
    link = Link(42)
    self.assertEqual(link.v, 42)
    self.assertIsNone(link.next_link)
    self.assertFalse(link.ignore_result)

  def test_with_args_kwargs(self):
    fn = lambda x, y, k=0: x + y + k
    link = Link(fn, (1, 2), {'k': 3})
    self.assertIs(link.v, fn)
    self.assertEqual(link.args, (1, 2))
    self.assertEqual(link.kwargs, {'k': 3})

  def test_ignore_result_flag(self):
    link = Link(lambda x: x, ignore_result=True)
    self.assertTrue(link.ignore_result)

  def test_is_chain_detected_for_chain(self):
    chain = Chain()
    link = Link(chain)
    self.assertTrue(link.is_chain)

  def test_is_chain_false_for_function(self):
    link = Link(lambda x: x)
    self.assertFalse(link.is_chain)

  def test_sets_is_nested_on_inner_chain(self):
    chain = Chain()
    self.assertFalse(chain.is_nested)
    Link(chain)
    self.assertTrue(chain.is_nested)

  def test_next_link_default_none(self):
    link = Link(42)
    self.assertIsNone(link.next_link)

  def test_original_value_stored(self):
    fn = lambda: None
    link = Link(fn, original_value=42)
    self.assertEqual(link.original_value, 42)


class TestEvaluateValue(unittest.TestCase):

  def test_callable_receives_current_value(self):
    fn = _make_tracker()
    link = Link(fn)
    result = _evaluate_value(link, 'hello')
    self.assertEqual(len(fn.calls), 1)
    self.assertEqual(fn.calls[0], (('hello',), {}))
    self.assertEqual(result, 'ok')

  def test_callable_no_current_value_calls_no_args(self):
    fn = _make_tracker()
    link = Link(fn)
    result = _evaluate_value(link, Null)
    self.assertEqual(len(fn.calls), 1)
    self.assertEqual(fn.calls[0], ((), {}))
    self.assertEqual(result, 'ok')

  def test_explicit_args_override(self):
    fn = _make_tracker()
    link = Link(fn, (1, 2), {'k': 3})
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(len(fn.calls), 1)
    self.assertEqual(fn.calls[0], ((1, 2), {'k': 3}))
    self.assertEqual(result, 'ok')

  def test_ellipsis_calls_with_no_args(self):
    fn = _make_tracker()
    link = Link(fn, (...,))
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(len(fn.calls), 1)
    self.assertEqual(fn.calls[0], ((), {}))
    self.assertEqual(result, 'ok')

  def test_non_callable_returned_as_is(self):
    link = Link(42)
    result = _evaluate_value(link, 'anything')
    self.assertEqual(result, 42)

  def test_chain_link_calls_run_directly(self):
    inner = Chain(lambda: 10)
    link = Link(inner)
    result = _evaluate_value(link, Null)
    self.assertEqual(result, 10)

  def test_chain_link_with_ellipsis(self):
    inner = Chain(lambda: 77)
    link = Link(inner, (...,))
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(result, 77)

  def test_chain_link_with_args(self):
    inner = Chain()
    inner._then(Link(lambda x: x * 2))
    link = Link(inner, (5,))
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(result, 10)

  def test_chain_link_with_current_value(self):
    inner = Chain()
    inner._then(Link(lambda x: x + 100))
    link = Link(inner)
    result = _evaluate_value(link, 7)
    self.assertEqual(result, 107)


if __name__ == '__main__':
  unittest.main()
