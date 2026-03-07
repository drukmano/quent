"""Exhaustive combinatorial tests for the X placeholder proxy."""
from __future__ import annotations

import unittest

from quent import Chain
from quent._x import X, _XExpr, _XAttr
from helpers import AsyncRange


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------
class TestXIdentity(unittest.TestCase):

  def test_identity_returns_value_for_various_types(self):
    """X(value) with no ops returns value unchanged."""
    for val in [42, 3.14, 'hello', None, True, False, [1, 2], {'a': 1}, (1,), set()]:
      self.assertIs(X(val), val)

  def test_x_is_callable(self):
    self.assertTrue(callable(X))


# ---------------------------------------------------------------------------
# 2. Arithmetic operators
# ---------------------------------------------------------------------------
class TestXArithmeticAdd(unittest.TestCase):

  def test_add_int(self):
    self.assertEqual((X + 3)(10), 13)

  def test_add_float(self):
    self.assertAlmostEqual((X + 1.5)(2.5), 4.0)

  def test_add_string_concat(self):
    self.assertEqual((X + ' world')('hello'), 'hello world')

  def test_radd_int(self):
    self.assertEqual((3 + X)(10), 13)

  def test_radd_float(self):
    self.assertAlmostEqual((1.5 + X)(2.5), 4.0)

  def test_radd_string(self):
    self.assertEqual(('hello ' + X)('world'), 'hello world')


class TestXArithmeticSub(unittest.TestCase):

  def test_sub_int(self):
    self.assertEqual((X - 3)(10), 7)

  def test_sub_float(self):
    self.assertAlmostEqual((X - 0.5)(2.0), 1.5)

  def test_rsub_int(self):
    self.assertEqual((100 - X)(30), 70)

  def test_rsub_float(self):
    self.assertAlmostEqual((10.0 - X)(3.5), 6.5)


class TestXArithmeticMul(unittest.TestCase):

  def test_mul_int(self):
    self.assertEqual((X * 4)(5), 20)

  def test_mul_float(self):
    self.assertAlmostEqual((X * 2.5)(4.0), 10.0)

  def test_mul_string_repeat(self):
    self.assertEqual((X * 3)('ab'), 'ababab')

  def test_rmul_int(self):
    self.assertEqual((4 * X)(5), 20)

  def test_rmul_string_repeat(self):
    self.assertEqual((3 * X)('ab'), 'ababab')


class TestXArithmeticDiv(unittest.TestCase):

  def test_truediv_int(self):
    self.assertAlmostEqual((X / 2)(10), 5.0)

  def test_truediv_float(self):
    self.assertAlmostEqual((X / 0.5)(3.0), 6.0)

  def test_rtruediv_int(self):
    self.assertAlmostEqual((100 / X)(4), 25.0)

  def test_rtruediv_float(self):
    self.assertAlmostEqual((9.0 / X)(3.0), 3.0)


class TestXArithmeticFloorDiv(unittest.TestCase):

  def test_floordiv_int(self):
    self.assertEqual((X // 3)(10), 3)

  def test_floordiv_float(self):
    self.assertEqual((X // 2.0)(7.0), 3.0)

  def test_rfloordiv_int(self):
    self.assertEqual((100 // X)(3), 33)

  def test_rfloordiv_float(self):
    self.assertEqual((10.0 // X)(3.0), 3.0)


class TestXArithmeticMod(unittest.TestCase):

  def test_mod_int(self):
    self.assertEqual((X % 3)(10), 1)

  def test_mod_float(self):
    self.assertAlmostEqual((X % 2.5)(5.5), 0.5)

  def test_rmod_int(self):
    self.assertEqual((100 % X)(7), 2)

  def test_rmod_string_format(self):
    # str.__mod__ takes precedence over X.__rmod__ since X is not a str subclass.
    # 'hello %s' % X produces a string (formatting X's repr), not an _XExpr.
    # So we test rmod with int instead.
    self.assertEqual((100 % X)(7), 2)


class TestXArithmeticPow(unittest.TestCase):

  def test_pow_int(self):
    self.assertEqual((X ** 3)(2), 8)

  def test_pow_float(self):
    self.assertAlmostEqual((X ** 0.5)(9.0), 3.0)

  def test_rpow_int(self):
    self.assertEqual((2 ** X)(10), 1024)

  def test_rpow_float(self):
    self.assertAlmostEqual((4.0 ** X)(0.5), 2.0)


# ---------------------------------------------------------------------------
# 3. Bitwise operators
# ---------------------------------------------------------------------------
class TestXBitwise(unittest.TestCase):

  def test_and(self):
    self.assertEqual((X & 0x0F)(0xFF), 0x0F)

  def test_rand(self):
    self.assertEqual((0x0F & X)(0xFF), 0x0F)

  def test_or(self):
    self.assertEqual((X | 0x0F)(0xF0), 0xFF)

  def test_ror(self):
    self.assertEqual((0x0F | X)(0xF0), 0xFF)

  def test_xor(self):
    self.assertEqual((X ^ 0xFF)(0x0F), 0xF0)

  def test_rxor(self):
    self.assertEqual((0xFF ^ X)(0x0F), 0xF0)

  def test_lshift(self):
    self.assertEqual((X << 2)(1), 4)

  def test_rlshift(self):
    self.assertEqual((1 << X)(3), 8)

  def test_rshift(self):
    self.assertEqual((X >> 2)(16), 4)

  def test_rrshift(self):
    self.assertEqual((16 >> X)(2), 4)

  def test_and_zero(self):
    self.assertEqual((X & 0)(0xFF), 0)

  def test_or_zero(self):
    self.assertEqual((X | 0)(0xFF), 0xFF)

  def test_xor_same(self):
    self.assertEqual((X ^ 42)(42), 0)

  def test_lshift_zero(self):
    self.assertEqual((X << 0)(5), 5)


# ---------------------------------------------------------------------------
# 4. Comparison operators
# ---------------------------------------------------------------------------
class TestXComparison(unittest.TestCase):

  def test_eq_true(self):
    self.assertTrue((X == 5)(5))

  def test_eq_false(self):
    self.assertFalse((X == 5)(6))

  def test_ne_true(self):
    self.assertTrue((X != 5)(6))

  def test_ne_false(self):
    self.assertFalse((X != 5)(5))

  def test_lt_true(self):
    self.assertTrue((X < 10)(5))

  def test_lt_false(self):
    self.assertFalse((X < 10)(10))

  def test_le_true_less(self):
    self.assertTrue((X <= 10)(5))

  def test_le_true_equal(self):
    self.assertTrue((X <= 10)(10))

  def test_le_false(self):
    self.assertFalse((X <= 10)(11))

  def test_gt_true(self):
    self.assertTrue((X > 3)(5))

  def test_gt_false(self):
    self.assertFalse((X > 3)(3))

  def test_ge_true_greater(self):
    self.assertTrue((X >= 3)(5))

  def test_ge_true_equal(self):
    self.assertTrue((X >= 3)(3))

  def test_ge_false(self):
    self.assertFalse((X >= 3)(2))


# ---------------------------------------------------------------------------
# 5. Unary operators
# ---------------------------------------------------------------------------
class TestXUnary(unittest.TestCase):

  def test_neg(self):
    self.assertEqual((-X)(5), -5)

  def test_neg_negative(self):
    self.assertEqual((-X)(-3), 3)

  def test_pos(self):
    self.assertEqual((+X)(5), 5)

  def test_pos_negative(self):
    self.assertEqual((+X)(-5), -5)

  def test_abs_positive(self):
    self.assertEqual(abs(X)(5), 5)

  def test_abs_negative(self):
    self.assertEqual(abs(X)(-7), 7)

  def test_invert(self):
    self.assertEqual((~X)(0), -1)

  def test_invert_nonzero(self):
    self.assertEqual((~X)(5), -6)


# ---------------------------------------------------------------------------
# 6. Attribute access
# ---------------------------------------------------------------------------
class TestXAttributeAccess(unittest.TestCase):

  def test_real_on_complex(self):
    self.assertEqual(X.real(3 + 4j), 3.0)

  def test_imag_on_complex(self):
    self.assertEqual(X.imag(3 + 4j), 4.0)

  def test_class_attr_resolves_via_type(self):
    """X.__class__ resolves via type MRO, not __getattr__, so no error."""
    self.assertIs(X.__class__, _XExpr)

  def test_underscore_attr_raises(self):
    with self.assertRaises(AttributeError):
      X._private_attr

  def test_double_underscore_raises(self):
    with self.assertRaises(AttributeError):
      X.__foo__

  def test_attr_returns_xattr(self):
    self.assertIsInstance(X.real, _XAttr)

  def test_chained_attr_then_op(self):
    # X.real on complex, then add 10
    expr = X.real + 10
    self.assertEqual(expr(3 + 4j), 13.0)


# ---------------------------------------------------------------------------
# 7. Item access
# ---------------------------------------------------------------------------
class TestXItemAccess(unittest.TestCase):

  def test_list_index(self):
    self.assertEqual(X[0]([10, 20, 30]), 10)

  def test_list_negative_index(self):
    self.assertEqual(X[-1]([10, 20, 30]), 30)

  def test_tuple_index(self):
    self.assertEqual(X[1]((10, 20, 30)), 20)

  def test_string_index(self):
    self.assertEqual(X[0]('hello'), 'h')

  def test_dict_key(self):
    self.assertEqual(X['key']({'key': 'val'}), 'val')

  def test_slice(self):
    self.assertEqual(X[1:3]([10, 20, 30, 40]), [20, 30])

  def test_nested_item_access(self):
    self.assertEqual(X[0][1]([[10, 20], [30, 40]]), 20)

  def test_dict_nested(self):
    self.assertEqual(X['a']['b']({'a': {'b': 42}}), 42)


# ---------------------------------------------------------------------------
# 8. Method calls
# ---------------------------------------------------------------------------
class TestXMethodCalls(unittest.TestCase):

  def test_strip_zero_args(self):
    expr = X.strip()
    self.assertEqual(expr('  hello  '), 'hello')

  def test_upper_zero_args(self):
    expr = X.upper()
    self.assertEqual(expr('hello'), 'HELLO')

  def test_lower_zero_args(self):
    expr = X.lower()
    self.assertEqual(expr('HELLO'), 'hello')

  def test_replace_two_args(self):
    expr = X.replace('a', 'b')
    self.assertEqual(expr('banana'), 'bbnbnb')

  def test_startswith_two_args(self):
    expr = X.startswith('hel', 0)
    self.assertTrue(expr('hello'))

  def test_split_single_arg_is_replay(self):
    """X.split(',') with single arg is ambiguous -- treated as replay.

    This means calling X.split(',') with a string value will treat the
    call as replay: X.split is evaluated as getattr(value, 'split'),
    yielding the bound method.
    """
    expr = X.split
    # When called with a single arg, _XAttr.__call__ treats it as replay
    result = expr(',')
    # The result is the bound method str.split on ','
    self.assertTrue(callable(result))

  def test_chained_methods(self):
    expr = X.strip().upper()
    self.assertEqual(expr('  hello  '), 'HELLO')

  def test_encode_kwarg(self):
    expr = X.encode(encoding='utf-8')
    self.assertEqual(expr('hello'), b'hello')

  def test_method_returns_xexpr_not_xattr(self):
    """After calling X.strip(), the result is a plain _XExpr, not _XAttr."""
    expr = X.strip()
    self.assertIsInstance(expr, _XExpr)
    self.assertNotIsInstance(expr, _XAttr)

  def test_chained_strip_replace(self):
    expr = X.strip().replace('o', '0')
    self.assertEqual(expr('  foo  '), 'f00')


# ---------------------------------------------------------------------------
# 9. Complex chaining
# ---------------------------------------------------------------------------
class TestXComplexChaining(unittest.TestCase):

  def test_arithmetic_then_comparison(self):
    expr = X + 1 > 0
    self.assertTrue(expr(0))
    self.assertFalse(expr(-2))

  def test_multiple_arithmetic(self):
    expr = X * 2 + 1
    self.assertEqual(expr(5), 11)

  def test_modulo_then_comparison(self):
    expr = X % 2 == 0
    self.assertTrue(expr(4))
    self.assertFalse(expr(3))

  def test_sub_then_abs(self):
    expr = abs(X - 10)
    self.assertEqual(expr(3), 7)
    self.assertEqual(expr(15), 5)

  def test_floor_div_then_mod(self):
    expr = X // 10 % 3
    self.assertEqual(expr(50), 2)  # 50 // 10 = 5, 5 % 3 = 2

  def test_pow_then_sub(self):
    expr = X ** 2 - 1
    self.assertEqual(expr(5), 24)

  def test_item_then_arithmetic(self):
    expr = X[0] + X[1]  # This chains: X[0] returns int, then + X[1] is X[1] is _XExpr...
    # X[0] + X[1] won't work because X[1] is an _XExpr not a plain int
    # Instead test: item access then arithmetic with constant
    expr = X[0] * 10
    self.assertEqual(expr([3, 7]), 30)

  def test_item_then_comparison(self):
    expr = X[0] > 5
    self.assertTrue(expr([10]))
    self.assertFalse(expr([3]))

  def test_neg_then_add(self):
    expr = -X + 100
    self.assertEqual(expr(30), 70)

  def test_bitwise_then_comparison(self):
    expr = X & 1 == 1
    # Note: operator precedence: (X & 1) == 1
    self.assertTrue(expr(3))
    self.assertFalse(expr(4))


# ---------------------------------------------------------------------------
# 10. Integration with Chain
# ---------------------------------------------------------------------------
class TestXChainIntegrationSync(unittest.TestCase):

  def test_filter_even(self):
    result = Chain([1, 2, 3, 4]).filter(X % 2 == 0).run()
    self.assertEqual(result, [2, 4])

  def test_filter_gt(self):
    result = Chain([1, 2, 3, 4, 5]).filter(X > 3).run()
    self.assertEqual(result, [4, 5])

  def test_map_multiply(self):
    result = Chain([1, 2, 3]).map(X * 10).run()
    self.assertEqual(result, [10, 20, 30])

  def test_map_add(self):
    result = Chain([10, 20, 30]).map(X + 5).run()
    self.assertEqual(result, [15, 25, 35])

  def test_then_upper(self):
    result = Chain('hello').then(X.upper()).run()
    self.assertEqual(result, 'HELLO')

  def test_then_strip(self):
    result = Chain('  hi  ').then(X.strip()).run()
    self.assertEqual(result, 'hi')

  def test_then_identity(self):
    result = Chain([3, 1, 2]).then(X).run()
    self.assertEqual(result, [3, 1, 2])

  def test_then_item_access(self):
    result = Chain([10, 20, 30]).then(X[1]).run()
    self.assertEqual(result, 20)

  def test_then_dict_key(self):
    result = Chain({'name': 'Alice'}).then(X['name']).run()
    self.assertEqual(result, 'Alice')

  def test_if_x_predicate_true(self):
    result = Chain(5).if_(X > 3, X * 2).run()
    self.assertEqual(result, 10)

  def test_if_x_predicate_false(self):
    result = Chain(2).if_(X > 3, X * 2).run()
    self.assertEqual(result, 2)

  def test_filter_then_map(self):
    result = Chain([1, 2, 3, 4]).filter(X > 2).map(X * 10).run()
    self.assertEqual(result, [30, 40])

  def test_map_neg(self):
    result = Chain([1, -2, 3]).map(-X).run()
    self.assertEqual(result, [-1, 2, -3])

  def test_map_abs(self):
    result = Chain([-5, 3, -1]).map(abs(X)).run()
    self.assertEqual(result, [5, 3, 1])

  def test_then_arithmetic(self):
    result = Chain(10).then(X * 2 + 1).run()
    self.assertEqual(result, 21)


class TestXChainIntegrationAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_filter_with_x(self):
    """X expression works in filter with async iterable."""
    result = await Chain(AsyncRange(6)).filter(X % 2 == 0).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_async_map_with_x(self):
    result = await Chain(AsyncRange(4)).map(X * 10).run()
    self.assertEqual(result, [0, 10, 20, 30])


# ---------------------------------------------------------------------------
# 11. repr tests
# ---------------------------------------------------------------------------
class TestXRepr(unittest.TestCase):

  def test_bare_x(self):
    self.assertEqual(repr(X), 'X')

  def test_add(self):
    self.assertEqual(repr(X + 1), '(X + 1)')

  def test_radd(self):
    self.assertEqual(repr(1 + X), '(1 + X)')

  def test_sub(self):
    self.assertEqual(repr(X - 2), '(X - 2)')

  def test_rsub(self):
    self.assertEqual(repr(10 - X), '(10 - X)')

  def test_mul(self):
    self.assertEqual(repr(X * 3), '(X * 3)')

  def test_rmul(self):
    self.assertEqual(repr(3 * X), '(3 * X)')

  def test_truediv(self):
    self.assertEqual(repr(X / 2), '(X / 2)')

  def test_rtruediv(self):
    self.assertEqual(repr(10 / X), '(10 / X)')

  def test_floordiv(self):
    self.assertEqual(repr(X // 3), '(X // 3)')

  def test_rfloordiv(self):
    self.assertEqual(repr(10 // X), '(10 // X)')

  def test_mod(self):
    self.assertEqual(repr(X % 2), '(X % 2)')

  def test_rmod(self):
    self.assertEqual(repr(10 % X), '(10 % X)')

  def test_pow(self):
    self.assertEqual(repr(X ** 2), '(X ** 2)')

  def test_rpow(self):
    self.assertEqual(repr(2 ** X), '(2 ** X)')

  def test_and(self):
    self.assertEqual(repr(X & 0xFF), '(X & 255)')

  def test_or(self):
    self.assertEqual(repr(X | 1), '(X | 1)')

  def test_xor(self):
    self.assertEqual(repr(X ^ 3), '(X ^ 3)')

  def test_lshift(self):
    self.assertEqual(repr(X << 2), '(X << 2)')

  def test_rshift(self):
    self.assertEqual(repr(X >> 1), '(X >> 1)')

  def test_eq(self):
    self.assertEqual(repr(X == 5), '(X == 5)')

  def test_ne(self):
    self.assertEqual(repr(X != 0), '(X != 0)')

  def test_lt(self):
    self.assertEqual(repr(X < 10), '(X < 10)')

  def test_le(self):
    self.assertEqual(repr(X <= 10), '(X <= 10)')

  def test_gt(self):
    self.assertEqual(repr(X > 0), '(X > 0)')

  def test_ge(self):
    self.assertEqual(repr(X >= 0), '(X >= 0)')

  def test_neg(self):
    self.assertEqual(repr(-X), '-(X)')

  def test_pos(self):
    self.assertEqual(repr(+X), '+(X)')

  def test_abs(self):
    self.assertEqual(repr(abs(X)), 'abs(X)')

  def test_invert(self):
    self.assertEqual(repr(~X), '~(X)')

  def test_compound_mod_eq(self):
    self.assertEqual(repr(X % 2 == 0), '((X % 2) == 0)')

  def test_compound_mul_add(self):
    self.assertEqual(repr(X * 2 + 1), '((X * 2) + 1)')

  def test_attr(self):
    self.assertEqual(repr(X.strip), 'X.strip')

  def test_method_zero_args(self):
    self.assertEqual(repr(X.strip()), 'X.strip()')

  def test_method_two_args(self):
    self.assertEqual(repr(X.replace('a', 'b')), "X.replace('a', 'b')")

  def test_item_int(self):
    self.assertEqual(repr(X[0]), 'X[0]')

  def test_item_str(self):
    self.assertEqual(repr(X['key']), "X['key']")

  def test_item_slice(self):
    self.assertEqual(repr(X[1:3]), 'X[slice(1, 3, None)]')

  def test_chained_method(self):
    self.assertEqual(repr(X.strip().upper()), 'X.strip().upper()')

  def test_attr_then_add(self):
    self.assertEqual(repr(X.real + 10), '(X.real + 10)')

  def test_method_with_kwargs(self):
    r = repr(X.encode(encoding='utf-8'))
    self.assertEqual(r, "X.encode(encoding='utf-8')")

  def test_nested_item(self):
    self.assertEqual(repr(X[0][1]), 'X[0][1]')

  def test_neg_then_add_repr(self):
    self.assertEqual(repr(-X + 100), '(-(X) + 100)')


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------
class TestXEdgeCases(unittest.TestCase):

  def test_none_value(self):
    self.assertIsNone(X(None))

  def test_boolean_true(self):
    self.assertIs(X(True), True)

  def test_boolean_false(self):
    self.assertIs(X(False), False)

  def test_empty_list(self):
    self.assertEqual(X([]), [])

  def test_empty_dict(self):
    self.assertEqual(X({}), {})

  def test_empty_string(self):
    self.assertEqual(X(''), '')

  def test_empty_tuple(self):
    self.assertEqual(X(()), ())

  def test_multiple_x_expressions_independent(self):
    """Two X expressions should not share state."""
    expr1 = X + 1
    expr2 = X * 2
    self.assertEqual(expr1(10), 11)
    self.assertEqual(expr2(10), 20)
    # Re-run to confirm independence
    self.assertEqual(expr1(5), 6)
    self.assertEqual(expr2(5), 10)

  def test_x_not_affected_by_previous_uses(self):
    _ = X + 1
    _ = X * 2
    # X itself still has no ops
    self.assertEqual(X(42), 42)

  def test_setattr_raises(self):
    with self.assertRaises(AttributeError) as ctx:
      X.foo = 42
    self.assertIn('Cannot set attributes', str(ctx.exception))

  def test_getattr_underscore_raises(self):
    with self.assertRaises(AttributeError):
      X._private

  def test_getattr_double_underscore_raises(self):
    with self.assertRaises(AttributeError):
      X.__dunder__

  def test_x_expr_no_ops_called_with_value(self):
    """A fresh _XExpr() with no ops returns value unchanged."""
    expr = _XExpr()
    self.assertEqual(expr(99), 99)

  def test_xattr_setattr_raises(self):
    attr = X.real
    with self.assertRaises(AttributeError):
      attr.foo = 42

  def test_x_has_slots(self):
    """_XExpr uses __slots__ — no __dict__."""
    self.assertFalse(hasattr(X, '__dict__'))

  def test_x_singleton_is_xexpr(self):
    self.assertIsInstance(X, _XExpr)

  def test_x_singleton_has_empty_ops(self):
    self.assertEqual(X._ops, ())

  def test_chain_returns_new_instance(self):
    expr1 = X + 1
    expr2 = X + 2
    self.assertIsNot(expr1, expr2)
    self.assertIsNot(expr1, X)

  def test_xattr_single_arg_call_is_replay(self):
    """_XAttr.__call__ with exactly 1 positional arg and no kwargs does replay."""
    attr_expr = X.upper  # _XAttr
    # Calling with single arg: replay mode
    result = attr_expr('hello')
    # Replay: getattr('hello', 'upper') -> bound method
    # but actually, since upper is the attr, and _XAttr replays,
    # it should get the attr from 'hello' and return it
    # Wait -- _XAttr.__call__ with 1 arg calls super().__call__(args[0])
    # which replays all ops on that value. The ops include ('attr', 'upper').
    # So: getattr('hello', 'upper') -> <bound method str.upper>
    self.assertTrue(callable(result))

  def test_xattr_zero_args_records_call(self):
    """_XAttr.__call__ with 0 args records a method call."""
    expr = X.upper()
    self.assertIsInstance(expr, _XExpr)
    self.assertNotIsInstance(expr, _XAttr)
    self.assertEqual(expr('hello'), 'HELLO')

  def test_xattr_multi_args_records_call(self):
    """_XAttr.__call__ with 2+ args records a method call."""
    expr = X.replace('a', 'b')
    self.assertIsInstance(expr, _XExpr)
    self.assertEqual(expr('banana'), 'bbnbnb')

  def test_xattr_kwargs_only_records_call(self):
    """_XAttr.__call__ with kwargs only records a method call."""
    expr = X.encode(encoding='ascii')
    self.assertIsInstance(expr, _XExpr)
    self.assertEqual(expr('hi'), b'hi')


# ---------------------------------------------------------------------------
# 13. Type preservation & correctness
# ---------------------------------------------------------------------------
class TestXTypePreservation(unittest.TestCase):

  def test_int_arithmetic_returns_int(self):
    self.assertIsInstance((X + 1)(2), int)

  def test_float_arithmetic_returns_float(self):
    self.assertIsInstance((X + 1.0)(2), float)

  def test_string_ops_return_string(self):
    self.assertIsInstance((X + 'b')('a'), str)

  def test_list_item_returns_element(self):
    result = X[0]([42])
    self.assertEqual(result, 42)

  def test_comparison_returns_bool(self):
    self.assertIsInstance((X > 0)(1), bool)


# ---------------------------------------------------------------------------
# 14. Operator composition depth
# ---------------------------------------------------------------------------
class TestXDeepComposition(unittest.TestCase):

  def test_three_ops(self):
    expr = X * 2 + 3 - 1
    self.assertEqual(expr(5), 12)  # (5*2)+3-1 = 12

  def test_four_ops(self):
    expr = X + 1 + 2 + 3
    self.assertEqual(expr(0), 6)

  def test_mixed_arithmetic_and_comparison(self):
    expr = X * 3 - 2 > 10
    self.assertTrue(expr(5))   # 5*3-2=13 > 10
    self.assertFalse(expr(3))  # 3*3-2=7 > 10 is False

  def test_item_then_method(self):
    expr = X[0].upper()
    self.assertEqual(expr(['hello', 'world']), 'HELLO')

  def test_item_then_item_then_arithmetic(self):
    expr = X[0][1] * 2
    self.assertEqual(expr([[10, 20], [30, 40]]), 40)

  def test_attr_then_method(self):
    """Access attribute then call method on it."""
    # complex.real is a float, float.is_integer() is a method
    expr = X.real  # _XAttr -- using as replay
    # We can't chain .is_integer() on _XAttr easily due to ambiguity
    # Instead test: attr access then arithmetic
    expr2 = X.real + X.imag  # This won't work: X.imag is _XAttr, not scalar
    # Use a constant instead
    expr3 = X.real * 2
    self.assertEqual(expr3(3 + 4j), 6.0)


# ---------------------------------------------------------------------------
# 15. Error propagation
# ---------------------------------------------------------------------------
class TestXErrorPropagation(unittest.TestCase):

  def test_type_error_on_invalid_add(self):
    expr = X + 1
    with self.assertRaises(TypeError):
      expr('string')

  def test_key_error_on_missing_dict_key(self):
    expr = X['missing']
    with self.assertRaises(KeyError):
      expr({'a': 1})

  def test_index_error_on_out_of_range(self):
    expr = X[10]
    with self.assertRaises(IndexError):
      expr([1, 2])

  def test_attribute_error_on_missing_attr(self):
    expr = X.nonexistent_attr
    with self.assertRaises(AttributeError):
      expr(42)

  def test_zero_division_error(self):
    expr = X / 0
    with self.assertRaises(ZeroDivisionError):
      expr(5)

  def test_zero_division_rdiv(self):
    expr = 1 / X
    with self.assertRaises(ZeroDivisionError):
      expr(0)


# ---------------------------------------------------------------------------
# 16. Hash and bool behavior
# ---------------------------------------------------------------------------
class TestXHashAndBool(unittest.TestCase):

  def test_x_is_unhashable(self):
    """__eq__ is overridden without __hash__, making _XExpr unhashable."""
    with self.assertRaises(TypeError):
      hash(X)

  def test_x_expr_is_unhashable(self):
    expr = X + 1
    with self.assertRaises(TypeError):
      hash(expr)

  def test_x_not_usable_as_dict_key(self):
    with self.assertRaises(TypeError):
      {X: 'value'}

  def test_x_not_usable_in_set(self):
    with self.assertRaises(TypeError):
      {X}


# ---------------------------------------------------------------------------
# 17. _XAttr specifics
# ---------------------------------------------------------------------------
class TestXAttrSpecifics(unittest.TestCase):

  def test_xattr_inherits_from_xexpr(self):
    self.assertTrue(issubclass(_XAttr, _XExpr))

  def test_xattr_has_slots(self):
    self.assertFalse(hasattr(X.strip, '__dict__'))

  def test_xattr_ops_include_attr(self):
    attr = X.strip
    self.assertEqual(attr._ops, (('attr', 'strip'),))

  def test_xattr_chained_ops(self):
    expr = X.strip().upper()
    expected_ops = (('attr', 'strip'), ('call', ((), {})), ('attr', 'upper'), ('call', ((), {})))
    self.assertEqual(expr._ops, expected_ops)


# ---------------------------------------------------------------------------
# 18. Reusability (frozen-like patterns)
# ---------------------------------------------------------------------------
class TestXReusability(unittest.TestCase):

  def test_same_expr_multiple_calls(self):
    expr = X * 2
    self.assertEqual(expr(3), 6)
    self.assertEqual(expr(5), 10)
    self.assertEqual(expr(0), 0)

  def test_same_expr_different_types(self):
    expr = X * 3
    self.assertEqual(expr(2), 6)
    self.assertEqual(expr('ab'), 'ababab')
    self.assertEqual(expr([1]), [1, 1, 1])

  def test_original_x_unmodified_after_building_exprs(self):
    _ = X + 1
    _ = X * 2
    _ = X.strip()
    _ = X[0]
    _ = -X
    self.assertEqual(X._ops, ())


# ---------------------------------------------------------------------------
# 19. Mixed left/right operator consistency
# ---------------------------------------------------------------------------
class TestXLeftRightConsistency(unittest.TestCase):

  def test_add_commutativity(self):
    self.assertEqual((X + 5)(3), (5 + X)(3))

  def test_mul_commutativity(self):
    self.assertEqual((X * 4)(7), (4 * X)(7))

  def test_sub_non_commutativity(self):
    self.assertNotEqual((X - 3)(10), (3 - X)(10))
    self.assertEqual((X - 3)(10), 7)
    self.assertEqual((3 - X)(10), -7)

  def test_div_non_commutativity(self):
    self.assertNotEqual((X / 2)(10), (2 / X)(10))
    self.assertAlmostEqual((X / 2)(10), 5.0)
    self.assertAlmostEqual((2 / X)(10), 0.2)

  def test_pow_non_commutativity(self):
    self.assertNotEqual((X ** 2)(3), (2 ** X)(3))
    self.assertEqual((X ** 2)(3), 9)
    self.assertEqual((2 ** X)(3), 8)


# ---------------------------------------------------------------------------
# 20. Chain integration — more patterns
# ---------------------------------------------------------------------------
class TestXChainAdvanced(unittest.TestCase):

  def test_filter_strings_by_length(self):
    # X is each string, len is not directly available via X ops
    # but we can filter by comparison on items
    result = Chain(['a', 'bb', 'ccc', 'dd']).filter(X > 'b').run()
    self.assertEqual(result, ['bb', 'ccc', 'dd'])

  def test_map_string_upper(self):
    result = Chain(['hello', 'world']).map(X.upper()).run()
    self.assertEqual(result, ['HELLO', 'WORLD'])

  def test_then_slice(self):
    result = Chain([10, 20, 30, 40, 50]).then(X[1:4]).run()
    self.assertEqual(result, [20, 30, 40])

  def test_then_negative_index(self):
    result = Chain([10, 20, 30]).then(X[-1]).run()
    self.assertEqual(result, 30)

  def test_if_x_with_arithmetic_predicate(self):
    result = Chain(10).if_(X % 2 == 0, X // 2).run()
    self.assertEqual(result, 5)

  def test_if_x_predicate_false_passthrough(self):
    result = Chain(11).if_(X % 2 == 0, X // 2).run()
    self.assertEqual(result, 11)

  def test_map_with_complex_expr(self):
    result = Chain([1, 2, 3]).map(X * 2 + 1).run()
    self.assertEqual(result, [3, 5, 7])

  def test_filter_with_bitwise(self):
    result = Chain([1, 2, 3, 4, 5, 6]).filter(X & 1 == 1).run()
    self.assertEqual(result, [1, 3, 5])

  def test_map_item_access(self):
    result = Chain([(1, 'a'), (2, 'b'), (3, 'c')]).map(X[0]).run()
    self.assertEqual(result, [1, 2, 3])

  def test_map_dict_key_access(self):
    data = [{'v': 10}, {'v': 20}, {'v': 30}]
    result = Chain(data).map(X['v']).run()
    self.assertEqual(result, [10, 20, 30])

  def test_chain_then_x_attr_passthrough(self):
    """X.attr used in .then() — _XAttr called with single arg triggers replay."""
    result = Chain('  hello  ').then(X.strip).run()
    # X.strip is _XAttr; Chain calls it with current_value -> replay
    # replay: getattr('  hello  ', 'strip') -> bound method
    # The result is the bound method, not the stripped string
    self.assertTrue(callable(result))

  def test_chain_then_x_method_call(self):
    """X.strip() (with parens) used in .then() works correctly."""
    result = Chain('  hello  ').then(X.strip()).run()
    self.assertEqual(result, 'hello')


if __name__ == '__main__':
  unittest.main()
