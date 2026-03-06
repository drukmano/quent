"""Tests for all guard methods on _XExpr / X to prevent misuse.

Covers: __contains__, __iter__, __copy__, __deepcopy__, __reduce__,
__hash__ (implicit None via __eq__), bool/not truthiness, and
short-circuit operator behavior.
"""
from __future__ import annotations

import copy
import pickle
import unittest

from quent._x import X, _XExpr, _XAttr


# ---------------------------------------------------------------------------
# 1. __contains__ guard
# ---------------------------------------------------------------------------
class TestContainsGuard(unittest.TestCase):

  def test_int_in_X_raises(self):
    with self.assertRaises(TypeError) as cm:
      5 in X
    self.assertIn('in', str(cm.exception))

  def test_str_in_X_raises(self):
    with self.assertRaises(TypeError) as cm:
      'a' in X
    self.assertIn('in', str(cm.exception))

  def test_none_in_X_raises(self):
    with self.assertRaises(TypeError) as cm:
      None in X
    self.assertIn('in', str(cm.exception))

  def test_X_in_list_does_not_raise(self):
    """X in [1, 2] uses list.__contains__, not X.__contains__."""
    # This should not raise; list.__contains__ checks identity/equality.
    # Because _XExpr overrides __eq__ to return another _XExpr (truthy),
    # membership may return True — the key point is no TypeError.
    try:
      _ = X in [1, 2]
    except TypeError:
      self.fail('X in [1, 2] should not raise TypeError')

  def test_contains_on_expression_raises(self):
    with self.assertRaises(TypeError):
      5 in (X + 1)


# ---------------------------------------------------------------------------
# 2. __iter__ guard
# ---------------------------------------------------------------------------
class TestIterGuard(unittest.TestCase):

  def test_iter_X_raises(self):
    with self.assertRaises(TypeError) as cm:
      iter(X)
    self.assertIn('iterate', str(cm.exception).lower())

  def test_list_X_raises(self):
    with self.assertRaises(TypeError):
      list(X)

  def test_for_loop_raises(self):
    with self.assertRaises(TypeError):
      for x in X:
        pass

  def test_list_comprehension_raises(self):
    with self.assertRaises(TypeError):
      [x for x in X]

  def test_tuple_X_raises(self):
    with self.assertRaises(TypeError):
      tuple(X)

  def test_iter_on_expression_raises(self):
    with self.assertRaises(TypeError):
      iter(X + 1)


# ---------------------------------------------------------------------------
# 3. __copy__ guard
# ---------------------------------------------------------------------------
class TestCopyGuard(unittest.TestCase):

  def test_copy_X_raises(self):
    with self.assertRaises(TypeError) as cm:
      copy.copy(X)
    self.assertIn('copied', str(cm.exception).lower())

  def test_copy_expression_raises(self):
    with self.assertRaises(TypeError) as cm:
      copy.copy(X + 1)
    self.assertIn('copied', str(cm.exception).lower())

  def test_copy_attr_expression_raises(self):
    with self.assertRaises(TypeError):
      copy.copy(X.attr)


# ---------------------------------------------------------------------------
# 4. __deepcopy__ guard
# ---------------------------------------------------------------------------
class TestDeepcopyGuard(unittest.TestCase):

  def test_deepcopy_X_raises(self):
    with self.assertRaises(TypeError) as cm:
      copy.deepcopy(X)
    self.assertIn('deep-copied', str(cm.exception).lower())

  def test_deepcopy_attr_expression_raises(self):
    with self.assertRaises(TypeError) as cm:
      copy.deepcopy(X.attr)
    self.assertIn('deep-copied', str(cm.exception).lower())

  def test_deepcopy_arithmetic_expression_raises(self):
    with self.assertRaises(TypeError):
      copy.deepcopy(X + 1)


# ---------------------------------------------------------------------------
# 5. __reduce__ guard (pickle)
# ---------------------------------------------------------------------------
class TestPickleGuard(unittest.TestCase):

  def test_pickle_X_raises(self):
    with self.assertRaises(TypeError) as cm:
      pickle.dumps(X)
    self.assertIn('pickled', str(cm.exception).lower())

  def test_pickle_expression_raises(self):
    with self.assertRaises(TypeError) as cm:
      pickle.dumps(X + 1)
    self.assertIn('pickled', str(cm.exception).lower())

  def test_pickle_attr_expression_raises(self):
    with self.assertRaises(TypeError):
      pickle.dumps(X.attr)


# ---------------------------------------------------------------------------
# 6. __hash__ is None (unhashable)
# ---------------------------------------------------------------------------
class TestHashGuard(unittest.TestCase):

  def test_hash_X_raises(self):
    with self.assertRaises(TypeError):
      hash(X)

  def test_set_add_raises(self):
    with self.assertRaises(TypeError):
      {X}

  def test_dict_key_raises(self):
    with self.assertRaises(TypeError):
      {X: 1}

  def test_hash_expression_raises(self):
    with self.assertRaises(TypeError):
      hash(X + 1)

  def test_hash_is_none_on_class(self):
    """__hash__ should be None because __eq__ is overridden."""
    self.assertIsNone(_XExpr.__hash__)


# ---------------------------------------------------------------------------
# 7. bool(X) and not X behavior (default object truthiness)
# ---------------------------------------------------------------------------
class TestBoolTruthiness(unittest.TestCase):

  def test_bool_X_is_true(self):
    """bool(X) returns True — default object truthiness, __bool__ not overridden."""
    self.assertTrue(bool(X))

  def test_not_X_is_false(self):
    self.assertFalse(not X)

  def test_bool_expression_is_true(self):
    """Non-empty ops makes _XExpr truthy via default object truthiness."""
    self.assertTrue(bool(X + 1))

  def test_bool_attr_expression_is_true(self):
    self.assertTrue(bool(X.attr))

  def test_not_expression_is_false(self):
    self.assertFalse(not (X + 1))


# ---------------------------------------------------------------------------
# 8. Short-circuit operators (and / or)
# ---------------------------------------------------------------------------
class TestShortCircuitOperators(unittest.TestCase):

  def test_X_and_5_returns_5(self):
    """X is truthy, so `X and 5` evaluates and returns the right operand."""
    result = X and 5
    self.assertEqual(result, 5)

  def test_X_or_5_returns_X(self):
    """X is truthy, so `X or 5` short-circuits and returns X (the _XExpr)."""
    result = X or 5
    self.assertIsInstance(result, _XExpr)

  def test_false_and_X_returns_false(self):
    """False short-circuits `and`, X is never evaluated."""
    result = False and X
    self.assertIs(result, False)

  def test_true_or_X_returns_true(self):
    """True short-circuits `or`, X is never evaluated."""
    result = True or X
    self.assertIs(result, True)

  def test_none_and_X_returns_none(self):
    """None is falsy, short-circuits `and`."""
    result = None and X
    self.assertIsNone(result)

  def test_zero_or_X_returns_X(self):
    """0 is falsy, so `0 or X` evaluates and returns X."""
    result = 0 or X
    self.assertIsInstance(result, _XExpr)


# ---------------------------------------------------------------------------
# 9. Error message quality
# ---------------------------------------------------------------------------
class TestErrorMessages(unittest.TestCase):

  def test_contains_message_mentions_X_expressions(self):
    with self.assertRaises(TypeError) as cm:
      5 in X
    msg = str(cm.exception)
    self.assertIn('X expression', msg)

  def test_contains_message_suggests_lambda(self):
    with self.assertRaises(TypeError) as cm:
      5 in X
    msg = str(cm.exception)
    self.assertIn('lambda', msg)

  def test_iter_message_mentions_X_expressions(self):
    with self.assertRaises(TypeError) as cm:
      iter(X)
    msg = str(cm.exception)
    self.assertIn('X expression', msg)

  def test_iter_message_suggests_lambda(self):
    with self.assertRaises(TypeError) as cm:
      iter(X)
    msg = str(cm.exception)
    self.assertIn('lambda', msg)

  def test_copy_message_suggests_new_expression(self):
    with self.assertRaises(TypeError) as cm:
      copy.copy(X)
    msg = str(cm.exception)
    self.assertIn('new expression', msg.lower())

  def test_deepcopy_message_suggests_new_expression(self):
    with self.assertRaises(TypeError) as cm:
      copy.deepcopy(X)
    msg = str(cm.exception)
    self.assertIn('new expression', msg.lower())

  def test_pickle_message_suggests_new_expression(self):
    with self.assertRaises(TypeError) as cm:
      pickle.dumps(X)
    msg = str(cm.exception)
    self.assertIn('new expression', msg.lower())


if __name__ == '__main__':
  unittest.main()
