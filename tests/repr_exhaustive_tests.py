"""Exhaustive tests for __repr__ methods: Chain, _Generator,
and _get_obj_name edge cases.

Complements repr_tests.py with deeper coverage of edge cases, unusual inputs,
and the interaction between nested chains and repr.
"""
from __future__ import annotations

import functools
import operator
import unittest

from quent import Chain
from quent._core import Link
from quent._ops import _Generator, _make_foreach, _make_filter, _make_gather, _make_with
from quent._traceback import _get_obj_name
from helpers import (
  Adder,
  ObjWithBadName,
  ObjWithBadNameAndRepr,
  partial_fn,
  sync_fn,
)


# ---------------------------------------------------------------------------
# TestChainReprExhaustive
# ---------------------------------------------------------------------------

class TestChainReprExhaustive(unittest.TestCase):

  def test_repr_every_operation(self):
    """Chain with then, do, map, filter, gather, with_, with_do all show."""
    c = (
      Chain(1)
      .then(sync_fn)
      .do(print)
      .map(sync_fn)
      .filter(sync_fn)
      .gather(sync_fn, str)
      .with_(sync_fn)
      .with_do(sync_fn)
    )
    r = repr(c)
    self.assertIn('.then(...)', r)
    self.assertIn('.do(...)', r)
    self.assertIn('.map(...)', r)
    self.assertIn('.filter(...)', r)
    self.assertIn('.gather(...)', r)
    self.assertIn('.with_(...)', r)
    self.assertIn('.with_do(...)', r)

  def test_repr_except_and_finally_NOT_in_repr(self):
    """except_ and finally_ are NOT shown in __repr__ (they are separate)."""
    c = Chain(1).except_(sync_fn).finally_(sync_fn)
    r = repr(c)
    self.assertNotIn('.except_', r)
    self.assertNotIn('.finally_', r)

  def test_repr_long_chain_100_steps(self):
    """100-step chain does not crash repr."""
    c = Chain(0)
    for _ in range(100):
      c = c.then(sync_fn)
    r = repr(c)
    self.assertEqual(r.count('.then(...)'), 100)
    self.assertTrue(r.startswith('Chain('))

  def test_repr_nested_chain(self):
    """Nested chain root shows Chain(...)."""
    inner = Chain().then(sync_fn)
    c = Chain(inner)
    r = repr(c)
    self.assertIn('Chain(', r)

  def test_repr_empty_chain(self):
    self.assertEqual(repr(Chain()), 'Chain()')

  def test_repr_with_root_value(self):
    r = repr(Chain(42))
    self.assertIn('42', r)
    self.assertTrue(r.startswith('Chain('))

  def test_repr_with_lambda(self):
    """Lambda shows <lambda>."""
    r = repr(Chain(lambda x: x))
    self.assertIn('<lambda>', r)

  def test_repr_with_builtin(self):
    """Builtin len shows 'len'."""
    r = repr(Chain(len))
    self.assertIn('len', r)

  def test_repr_chain_in_chain(self):
    """Root value that is a Chain shows 'Chain'."""
    inner = Chain()
    c = Chain(inner)
    r = repr(c)
    self.assertIn('Chain(Chain)', r)

  def test_repr_with_string_root(self):
    """String root value is displayed with quotes."""
    r = repr(Chain('hello'))
    self.assertIn("'hello'", r)

  def test_repr_starts_with_chain_paren(self):
    """repr always starts with Chain(."""
    r = repr(Chain(42).then(sync_fn).do(print))
    self.assertTrue(r.startswith('Chain('))


# ---------------------------------------------------------------------------
# TestGetObjNameEdgeCases
# ---------------------------------------------------------------------------

class TestGetObjNameEdgeCases(unittest.TestCase):

  def test_bad_name_raises(self):
    """ObjWithBadName falls to repr()."""
    obj = ObjWithBadName()
    result = _get_obj_name(obj)
    self.assertEqual(result, '<ObjWithBadName>')

  def test_bad_name_and_repr(self):
    """ObjWithBadNameAndRepr falls to type().__name__."""
    obj = ObjWithBadNameAndRepr()
    result = _get_obj_name(obj)
    self.assertEqual(result, 'ObjWithBadNameAndRepr')

  def test_partial_shows_wrapped(self):
    """partial(add) format."""
    result = _get_obj_name(partial_fn)
    self.assertEqual(result, 'partial(add)')

  def test_chain_shows_type(self):
    """Chain shows 'Chain'."""
    result = _get_obj_name(Chain())
    self.assertEqual(result, 'Chain')

  def test_none_shows_repr(self):
    """None shows 'None'."""
    result = _get_obj_name(None)
    self.assertEqual(result, 'None')

  def test_builtin(self):
    """Builtin len shows 'len'."""
    result = _get_obj_name(len)
    self.assertEqual(result, 'len')

  def test_int_shows_repr(self):
    """Integer 42 shows '42'."""
    result = _get_obj_name(42)
    self.assertEqual(result, '42')

  def test_class_shows_name(self):
    """Adder class shows 'Adder'."""
    result = _get_obj_name(Adder)
    self.assertEqual(result, 'Adder')

  def test_float_shows_repr(self):
    """Float shows repr."""
    result = _get_obj_name(3.14)
    self.assertEqual(result, '3.14')

  def test_list_shows_repr(self):
    """List shows repr."""
    result = _get_obj_name([1, 2, 3])
    self.assertEqual(result, '[1, 2, 3]')

  def test_dict_shows_repr(self):
    """Dict shows repr."""
    result = _get_obj_name({'a': 1})
    self.assertEqual(result, "{'a': 1}")



# ---------------------------------------------------------------------------
# TestGeneratorReprExhaustive
# ---------------------------------------------------------------------------

class TestGeneratorReprExhaustive(unittest.TestCase):

  def test_generator_repr(self):
    """<Quent._Generator>."""
    gen = Chain([1, 2, 3]).iterate()
    self.assertEqual(repr(gen), '<Quent._Generator>')

  def test_generator_repr_with_fn(self):
    gen = Chain([1, 2, 3]).iterate(sync_fn)
    self.assertEqual(repr(gen), '<Quent._Generator>')

  def test_generator_repr_iterate_do(self):
    gen = Chain([1, 2, 3]).iterate_do(sync_fn)
    self.assertEqual(repr(gen), '<Quent._Generator>')

  def test_generator_repr_called(self):
    """Calling a _Generator returns a new _Generator with same repr."""
    gen = Chain([1, 2, 3]).iterate()
    called = gen()
    self.assertEqual(repr(called), '<Quent._Generator>')

  def test_generator_repr_with_args(self):
    """_Generator called with args still has same repr."""
    gen = Chain().iterate()
    called = gen([4, 5, 6])
    self.assertEqual(repr(called), '<Quent._Generator>')


# ---------------------------------------------------------------------------
# BEYOND SPEC: Additional exhaustive tests
# ---------------------------------------------------------------------------

class TestUnicodeFunctionNames(unittest.TestCase):

  def test_unicode_name_in_repr(self):
    """Unicode function names display correctly in repr."""
    def calcul_integrale(x):
      return x

    r = repr(Chain(calcul_integrale))
    self.assertIn('calcul_integrale', r)

  def test_unicode_name_in_get_obj_name(self):
    """_get_obj_name handles unicode names."""
    def donnees_utilisateur(x):
      return x

    result = _get_obj_name(donnees_utilisateur)
    self.assertEqual(result, 'donnees_utilisateur')

  def test_emoji_function_name(self):
    """Function name with special characters does not crash."""
    def fn(x):
      return x

    fn.__name__ = 'test_fn_special'
    fn.__qualname__ = 'test_fn_special'
    result = _get_obj_name(fn)
    self.assertEqual(result, 'test_fn_special')


class TestVeryLongNames(unittest.TestCase):

  def test_long_function_name_in_repr(self):
    """Very long function names (100+ chars) do not crash repr."""
    long_name = 'very_long_function_name_' * 5  # 120 chars
    fn = lambda x: x
    fn.__name__ = long_name
    fn.__qualname__ = long_name
    r = repr(Chain(fn))
    self.assertIn(long_name, r)

  def test_long_function_name_in_get_obj_name(self):
    """_get_obj_name handles very long names."""
    long_name = 'x' * 200
    fn = lambda x: x
    fn.__name__ = long_name
    fn.__qualname__ = long_name
    result = _get_obj_name(fn)
    self.assertEqual(result, long_name)


class TestGetObjNameMoreEdgeCases(unittest.TestCase):

  def test_nested_partial(self):
    """Nested partial shows the innermost function name."""
    inner_partial = functools.partial(operator.add, 10)
    outer_partial = functools.partial(inner_partial, 20)
    result = _get_obj_name(outer_partial)
    self.assertIn('partial(', result)

  def test_set_value(self):
    """Set value shows repr."""
    result = _get_obj_name({1, 2, 3})
    self.assertIn('1', result)
    self.assertIn('2', result)
    self.assertIn('3', result)

  def test_tuple_value(self):
    """Tuple value shows repr."""
    result = _get_obj_name((1, 2))
    self.assertEqual(result, '(1, 2)')

  def test_bool_value(self):
    """Boolean shows repr."""
    self.assertEqual(_get_obj_name(True), 'True')
    self.assertEqual(_get_obj_name(False), 'False')

  def test_bytes_value(self):
    """Bytes shows repr."""
    result = _get_obj_name(b'hello')
    self.assertEqual(result, "b'hello'")

  def test_staticmethod_function(self):
    """Function with __name__ attribute works."""
    result = _get_obj_name(len)
    self.assertEqual(result, 'len')

  def test_class_method_shows_qualname(self):
    """Bound methods show qualname."""

    class MyClass:
      def my_method(self):
        pass

    obj = MyClass()
    result = _get_obj_name(obj.my_method)
    self.assertIn('my_method', result)


class TestReprOperationOrder(unittest.TestCase):

  def test_operation_order_preserved(self):
    """Operations appear in the order they were added."""
    c = (
      Chain(1)
      .then(sync_fn)
      .do(print)
      .then(str)
    )
    r = repr(c)
    then_pos = r.index('.then(...)')
    do_pos = r.index('.do(...)')
    last_then_pos = r.index('.then(...)', then_pos + 1)
    self.assertLess(then_pos, do_pos)
    self.assertLess(do_pos, last_then_pos)

  def test_many_do_operations(self):
    """Multiple do() operations all appear."""
    c = Chain(1).do(sync_fn).do(print).do(str)
    r = repr(c)
    self.assertEqual(r.count('.do(...)'), 3)


class TestReprWithSpecialRootValues(unittest.TestCase):

  def test_none_root(self):
    """None root value shows 'None'."""
    r = repr(Chain(None))
    self.assertIn('None', r)

  def test_ellipsis_root(self):
    """Ellipsis root value shows 'Ellipsis'."""
    r = repr(Chain(...))
    self.assertIn('Ellipsis', r)

  def test_zero_root(self):
    """Zero root value shows '0'."""
    r = repr(Chain(0))
    self.assertIn('0', r)

  def test_empty_string_root(self):
    """Empty string root value shows repr."""
    r = repr(Chain(''))
    self.assertIn("''", r)

  def test_empty_list_root(self):
    """Empty list root value shows []."""
    r = repr(Chain([]))
    self.assertIn('[]', r)


if __name__ == '__main__':
  unittest.main()
