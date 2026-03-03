"""Exhaustive value-type edge-case tests for the Quent library.

Systematically tests every chain feature with unusual, tricky, and edge-case
value types: falsy values, special numerics, callables-as-values, custom
objects with broken dunder methods, the Null sentinel, exception objects,
generators, coroutine objects, and more.

Target: 150+ individual test methods.
"""
import asyncio
import math
import functools
import sys
import types
from contextlib import contextmanager

from tests.utils import empty, aempty, await_, MyTestCase
from quent import Chain, Cascade, QuentException, run
from quent.quent import PyNull as Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def raise_(exc_type=Exception):
  raise exc_type('test')

def identity(v):
  return v


class CallableObj:
  """Instance is callable."""
  def __init__(self, ret=42):
    self.ret = ret
  def __call__(self, v=None):
    if v is None:
      return self.ret
    return v


class FalsyBool:
  """Custom object with __bool__ returning False."""
  def __bool__(self):
    return False
  def __repr__(self):
    return '<FalsyBool>'


class BoolRaises:
  """Custom object with __bool__ that raises."""
  def __bool__(self):
    raise ValueError('bool exploded')
  def __repr__(self):
    return '<BoolRaises>'


class ReprRaises:
  """Custom object with __repr__ that raises."""
  def __repr__(self):
    raise RuntimeError('repr exploded')


class AlwaysEqualObj:
  """Object with __eq__ that always returns True."""
  def __eq__(self, other):
    return True
  def __hash__(self):
    return 0


class UnhashableObj:
  """Object with __hash__ = None (unhashable)."""
  __hash__ = None


class SyncCM:
  """Simple sync context manager."""
  def __init__(self, value='cm_val'):
    self.value = value
    self.entered = False
    self.exited = False
  def __enter__(self):
    self.entered = True
    return self.value
  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  """Simple async context manager."""
  def __init__(self, value='acm_val'):
    self.value = value
    self.entered = False
    self.exited = False
  async def __aenter__(self):
    self.entered = True
    return self.value
  async def __aexit__(self, *args):
    self.exited = True
    return False


class CustomIterable:
  """Object with __iter__."""
  def __init__(self, items):
    self._items = items
  def __iter__(self):
    return iter(self._items)


class CustomAsyncIterable:
  """Object with __aiter__."""
  def __init__(self, items):
    self._items = items
  def __aiter__(self):
    return self._async_gen()
  async def _async_gen(self):
    for item in self._items:
      yield item


async def dummy_coro():
  return 'coro_result'


def dummy_gen():
  yield 1
  yield 2
  yield 3


async def dummy_async_gen():
  yield 10
  yield 20


# Falsy values that should survive chain propagation
FALSY_VALUES = [
  (None, 'None'),
  (0, 'zero_int'),
  (0.0, 'zero_float'),
  (False, 'False'),
  ('', 'empty_str'),
  ([], 'empty_list'),
  ({}, 'empty_dict'),
  (set(), 'empty_set'),
  (b'', 'empty_bytes'),
  (0j, 'zero_complex'),
]

# Non-callable truthy scalars
TRUTHY_SCALARS = [
  (1, 'one'),
  (-1, 'neg_one'),
  (42, 'forty_two'),
  (1.5, 'one_point_five'),
  (-0.1, 'neg_point_one'),
  (float('inf'), 'pos_inf'),
  (float('-inf'), 'neg_inf'),
  (True, 'True'),
  ('hello', 'hello_str'),
  (b'bytes', 'bytes_val'),
]

# Container values (non-callable, truthy)
CONTAINER_VALUES = [
  ([1, 2, 3], 'list_123'),
  ((1, 2, 3), 'tuple_123'),
  ({1, 2, 3}, 'set_123'),
  ({'a': 1}, 'dict_a1'),
  (frozenset({1, 2}), 'frozenset_12'),
  (range(5), 'range_5'),
  (bytearray(b'hello'), 'bytearray_hello'),
]

# Non-callable special objects
SPECIAL_OBJECTS = [
  (..., 'Ellipsis'),
  (NotImplemented, 'NotImplemented'),
  (object(), 'bare_object'),
  (ValueError('test_exc'), 'exception_instance'),
  (TypeError('type_err'), 'type_error_instance'),
]


# =========================================================================
# Test Classes
# =========================================================================

class FalsyValueRootTests(MyTestCase):
  """Falsy values as root — Chain(falsy).run() must return the exact value."""

  async def test_none_as_root(self):
    # Chain(None).run() — None is not callable, so EVAL_RETURN_AS_IS.
    # Then current_value is None. Since current_value is Null check: None is not Null.
    # So returns None.
    await self.assertIsNone(Chain(None).run())

  async def test_zero_int_as_root(self):
    await self.assertEqual(Chain(0).run(), 0)

  async def test_zero_float_as_root(self):
    result = Chain(0.0).run()
    assert result == 0.0 and isinstance(result, float)

  async def test_false_as_root(self):
    await self.assertIs(Chain(False).run(), False)

  async def test_empty_string_as_root(self):
    await self.assertEqual(Chain('').run(), '')

  async def test_empty_list_as_root(self):
    await self.assertEqual(Chain([]).run(), [])

  async def test_empty_dict_as_root(self):
    await self.assertEqual(Chain({}).run(), {})

  async def test_empty_set_as_root(self):
    await self.assertEqual(Chain(set()).run(), set())

  async def test_empty_bytes_as_root(self):
    await self.assertEqual(Chain(b'').run(), b'')

  async def test_zero_complex_as_root(self):
    await self.assertEqual(Chain(0j).run(), 0j)


class FalsyValueThenTests(MyTestCase):
  """Falsy values via .then(literal) — should be returned as-is (EVAL_RETURN_AS_IS)."""

  async def test_then_none_literal(self):
    # None is not callable, so EVAL_RETURN_AS_IS. Chain returns None.
    await self.assertIsNone(Chain(1).then(None).run())

  async def test_then_zero_literal(self):
    await self.assertEqual(Chain(1).then(0).run(), 0)

  async def test_then_false_literal(self):
    await self.assertIs(Chain(1).then(False).run(), False)

  async def test_then_empty_string_literal(self):
    await self.assertEqual(Chain(1).then('').run(), '')

  async def test_then_empty_list_literal(self):
    await self.assertEqual(Chain(1).then([]).run(), [])

  async def test_then_empty_dict_literal(self):
    await self.assertEqual(Chain(1).then({}).run(), {})

  async def test_then_empty_set_literal(self):
    await self.assertEqual(Chain(1).then(set()).run(), set())

  async def test_then_empty_bytes_literal(self):
    await self.assertEqual(Chain(1).then(b'').run(), b'')

  async def test_then_zero_complex_literal(self):
    await self.assertEqual(Chain(1).then(0j).run(), 0j)

  async def test_then_zero_float_literal(self):
    result = Chain(1).then(0.0).run()
    assert result == 0.0 and isinstance(result, float)


class FalsyValueSurvivalTests(MyTestCase):
  """Falsy values must survive a chain pipeline: root -> then -> do -> then -> run."""

  async def test_zero_survives_pipeline(self):
    result = Chain(0).then(lambda v: v).do(lambda v: 'side_effect').then(lambda v: v).run()
    await self.assertEqual(result, 0)

  async def test_false_survives_pipeline(self):
    result = Chain(False).then(lambda v: v).do(lambda v: None).then(lambda v: v).run()
    await self.assertIs(result, False)

  async def test_empty_string_survives_pipeline(self):
    result = Chain('').then(lambda v: v).do(lambda v: 'x').then(lambda v: v).run()
    await self.assertEqual(result, '')

  async def test_none_survives_pipeline(self):
    # None is a special case: Chain._run returns None when current_value is Null,
    # but also returns None when current_value IS None.
    result = Chain(None).then(lambda v: v).do(lambda v: 'x').then(lambda v: v).run()
    await self.assertIsNone(result)

  async def test_empty_list_survives_pipeline(self):
    result = Chain([]).then(lambda v: v).do(lambda v: [1]).then(lambda v: v).run()
    await self.assertEqual(result, [])

  async def test_empty_dict_survives_pipeline(self):
    result = Chain({}).then(lambda v: v).do(lambda v: {'a': 1}).then(lambda v: v).run()
    await self.assertEqual(result, {})

  async def test_zero_complex_survives_pipeline(self):
    result = Chain(0j).then(lambda v: v).do(lambda v: 1j).then(lambda v: v).run()
    await self.assertEqual(result, 0j)

  async def test_zero_float_survives_pipeline(self):
    result = Chain(0.0).then(lambda v: v).do(lambda v: 1.0).then(lambda v: v).run()
    assert result == 0.0 and isinstance(result, float)


class FalsyValueAsyncSurvivalTests(MyTestCase):
  """Falsy values survive async pipeline."""

  async def test_zero_survives_async_pipeline(self):
    await self.assertEqual(
      Chain(0).then(aempty).then(lambda v: v).run(), 0
    )

  async def test_false_survives_async_pipeline(self):
    await self.assertIs(
      Chain(False).then(aempty).then(lambda v: v).run(), False
    )

  async def test_empty_string_survives_async_pipeline(self):
    await self.assertEqual(
      Chain('').then(aempty).then(lambda v: v).run(), ''
    )

  async def test_none_survives_async_pipeline(self):
    await self.assertIsNone(
      Chain(None).then(aempty).then(lambda v: v).run()
    )

  async def test_empty_list_survives_async_pipeline(self):
    await self.assertEqual(
      Chain([]).then(aempty).then(lambda v: v).run(), []
    )


class FalsyValueDoDiscardTests(MyTestCase):
  """do() must discard the result and preserve the current value, even for falsy values."""

  async def test_do_discards_with_zero_root(self):
    await self.assertEqual(Chain(0).do(lambda v: 999).run(), 0)

  async def test_do_discards_with_false_root(self):
    await self.assertIs(Chain(False).do(lambda v: True).run(), False)

  async def test_do_discards_with_empty_string_root(self):
    await self.assertEqual(Chain('').do(lambda v: 'nonempty').run(), '')

  async def test_do_discards_with_none_root(self):
    await self.assertIsNone(Chain(None).do(lambda v: 'something').run())

  async def test_do_discards_with_empty_list_root(self):
    await self.assertEqual(Chain([]).do(lambda v: [1, 2]).run(), [])


class FalsyValueExceptHandlerTests(MyTestCase):
  """except_ handler returning falsy value becomes the chain result."""

  async def test_except_returns_zero(self):
    await self.assertEqual(
      Chain(1).then(raise_).except_(lambda v: 0, reraise=False).run(), 0
    )

  async def test_except_returns_false(self):
    await self.assertIs(
      Chain(1).then(raise_).except_(lambda v: False, reraise=False).run(), False
    )

  async def test_except_returns_empty_string(self):
    await self.assertEqual(
      Chain(1).then(raise_).except_(lambda v: '', reraise=False).run(), ''
    )

  async def test_except_returns_none(self):
    await self.assertIsNone(
      Chain(1).then(raise_).except_(lambda v: None, reraise=False).run()
    )

  async def test_except_returns_empty_list(self):
    await self.assertEqual(
      Chain(1).then(raise_).except_(lambda v: [], reraise=False).run(), []
    )


class TruthyScalarTests(MyTestCase):
  """Truthy scalars through chain operations."""

  async def test_truthy_scalars_as_root(self):
    for val, name in TRUTHY_SCALARS:
      with self.subTest(name=name):
        result = Chain(val).run()
        if isinstance(val, float) and math.isnan(val):
          assert math.isnan(result), f'Expected NaN, got {result}'
        else:
          assert result == val or result is val, f'Expected {val!r}, got {result!r}'

  async def test_truthy_scalars_as_then_literal(self):
    for val, name in TRUTHY_SCALARS:
      with self.subTest(name=name):
        result = Chain(1).then(val).run()
        if isinstance(val, float) and math.isnan(val):
          assert math.isnan(result), f'Expected NaN, got {result}'
        else:
          assert result == val or result is val, f'Expected {val!r}, got {result!r}'

  async def test_truthy_scalars_as_lambda_result(self):
    for val, name in TRUTHY_SCALARS:
      with self.subTest(name=name):
        result = Chain(1).then(lambda v, _val=val: _val).run()
        if isinstance(val, float) and math.isnan(val):
          assert math.isnan(result), f'Expected NaN, got {result}'
        else:
          assert result == val or result is val, f'Expected {val!r}, got {result!r}'


class SpecialNumericTests(MyTestCase):
  """Special numeric edge cases."""

  async def test_nan_root(self):
    result = Chain(float('nan')).run()
    assert math.isnan(result)

  async def test_nan_through_then(self):
    result = Chain(float('nan')).then(lambda v: v).run()
    assert math.isnan(result)

  async def test_nan_do_preserves(self):
    result = Chain(float('nan')).do(lambda v: 42).run()
    assert math.isnan(result)

  async def test_inf_arithmetic(self):
    await self.assertEqual(
      Chain(float('inf')).then(lambda v: v + 1).run(), float('inf')
    )

  async def test_neg_inf_arithmetic(self):
    await self.assertEqual(
      Chain(float('-inf')).then(lambda v: v - 1).run(), float('-inf')
    )

  async def test_large_int(self):
    big = 10**1000
    await self.assertEqual(Chain(big).then(lambda v: v + 1).run(), big + 1)

  async def test_large_negative_int(self):
    big = -(10**500)
    await self.assertEqual(Chain(big).then(lambda v: v).run(), big)


class ContainerValueTests(MyTestCase):
  """Container values as root, then literal, and lambda result."""

  async def test_containers_as_root(self):
    for val, name in CONTAINER_VALUES:
      with self.subTest(name=name):
        result = Chain(val).run()
        assert result == val, f'{name}: expected {val!r}, got {result!r}'

  async def test_containers_as_then_literal(self):
    for val, name in CONTAINER_VALUES:
      with self.subTest(name=name):
        result = Chain(1).then(val).run()
        assert result == val, f'{name}: expected {val!r}, got {result!r}'

  async def test_containers_as_lambda_result(self):
    for val, name in CONTAINER_VALUES:
      with self.subTest(name=name):
        result = Chain(1).then(lambda v, _val=val: _val).run()
        assert result == val, f'{name}: expected {val!r}, got {result!r}'

  async def test_containers_survive_do(self):
    for val, name in CONTAINER_VALUES:
      with self.subTest(name=name):
        result = Chain(val).do(lambda v: 'discarded').run()
        assert result == val, f'{name}: expected {val!r}, got {result!r}'

  async def test_memoryview_as_root(self):
    mv = memoryview(b'hello')
    result = Chain(mv).run()
    assert result is mv


class CallableAsValueTests(MyTestCase):
  """Callables in chain positions — the tricky part.

  Key insight from _determine_eval_code:
  - callable(v) with no args/kwargs -> EVAL_CALL_WITH_CURRENT_VALUE
  - This means Chain(callable_thing) will CALL the callable as root evaluation.
  - Chain(1).then(callable_thing) will CALL it with current_value=1.
  - To pass a callable as a LITERAL, use .then(lambda v: callable_thing).
  """

  async def test_lambda_as_root_is_called(self):
    # Chain(lambda: 42) calls the lambda (no args since root has Null cv)
    await self.assertEqual(Chain(lambda: 42).run(), 42)

  async def test_lambda_as_then_is_called(self):
    # Chain(1).then(lambda v: v*2) calls lambda with current_value=1
    await self.assertEqual(Chain(1).then(lambda v: v * 2).run(), 2)

  async def test_builtin_type_as_then(self):
    # str is callable, so Chain('42').then(int) calls int('42') -> 42 (via int)
    # Actually Chain('42') has root '42' (literal). Then int is callable.
    # int receives current_value='42' -> 42
    # Wait: '42' is not callable, so root is EVAL_RETURN_AS_IS -> current_value='42'
    # Then int is callable -> int('42') = 42
    await self.assertEqual(Chain('42').then(int).run(), 42)

  async def test_builtin_len_as_then(self):
    # len([1,2,3]) = 3
    await self.assertEqual(Chain([1, 2, 3]).then(len).run(), 3)

  async def test_builtin_len_with_non_sequence_raises(self):
    # len(1) raises TypeError
    with self.assertRaises(TypeError):
      Chain(1).then(len).run()

  async def test_class_type_as_root_constructs(self):
    # int is callable, Chain(int) calls int() -> 0
    await self.assertEqual(Chain(int).run(), 0)

  async def test_class_type_as_root_with_args(self):
    # Chain(int, '42') calls int('42') -> 42
    await self.assertEqual(Chain(int, '42').run(), 42)

  async def test_callable_obj_as_root(self):
    # CallableObj(99)() returns 99
    obj = CallableObj(99)
    await self.assertEqual(Chain(obj).run(), 99)

  async def test_callable_obj_as_then(self):
    # CallableObj()(5) returns 5
    obj = CallableObj()
    await self.assertEqual(Chain(5).then(obj).run(), 5)

  async def test_partial_as_then(self):
    p = functools.partial(lambda a, b: a + b, b=10)
    await self.assertEqual(Chain(5).then(p).run(), 15)

  async def test_partial_as_root(self):
    p = functools.partial(lambda a, b: a + b, 3, 7)
    # p is callable, Chain(p) calls p() -> 10
    await self.assertEqual(Chain(p).run(), 10)

  async def test_callable_as_literal_via_lambda(self):
    # To store a callable AS a value (not call it), wrap in lambda
    fn = lambda: 42
    result = Chain(1).then(lambda v: fn).run()
    assert result is fn

  async def test_type_as_literal_via_lambda(self):
    result = Chain(1).then(lambda v: int).run()
    assert result is int

  async def test_staticmethod_callable(self):
    class Holder:
      @staticmethod
      def double(v):
        return v * 2
    await self.assertEqual(Chain(5).then(Holder.double).run(), 10)

  async def test_classmethod_callable(self):
    class Holder:
      factor = 3
      @classmethod
      def multiply(cls, v):
        return v * cls.factor
    await self.assertEqual(Chain(5).then(Holder.multiply).run(), 15)

  async def test_chain_bool_always_true(self):
    """Chain.__bool__ always returns True regardless of root value."""
    assert bool(Chain(0)) is True
    assert bool(Chain(False)) is True
    assert bool(Chain(None)) is True
    assert bool(Chain([])) is True
    assert bool(Chain('')) is True


class NullSentinelTests(MyTestCase):
  """Tests for the Null sentinel value."""

  async def test_null_as_chain_root(self):
    # Chain(Null) — Null is the sentinel for "no root value".
    # In __init__, root_value is Null, so root_link is NOT created.
    # Chain with no root and no links -> run returns None.
    await self.assertIsNone(Chain(Null).run())

  async def test_null_as_then_literal(self):
    # Null is _Null instance. callable(Null)? _Null has no __call__.
    # So Null is not callable -> EVAL_RETURN_AS_IS.
    # But wait: in _run, after chain completes, if current_value is Null, return None.
    # So Chain(1).then(Null).run() -> current_value = Null -> returns None.
    await self.assertIsNone(Chain(1).then(Null).run())

  async def test_null_as_lambda_result(self):
    # Lambda returns Null. Then current_value = Null.
    # At end of _run: if current_value is Null: return None.
    await self.assertIsNone(Chain(1).then(lambda v: Null).run())

  async def test_void_chain_returns_none(self):
    # Chain() with no root, no links -> returns None
    await self.assertIsNone(Chain().run())

  async def test_void_chain_with_then(self):
    # Chain().then(lambda: 5) -> void chain, first link gets Null cv
    # lambda is callable, EVAL_CALL_WITH_CURRENT_VALUE, cv is Null -> call with no args
    await self.assertEqual(Chain().then(lambda: 5).run(), 5)


class SpecialPythonObjectTests(MyTestCase):
  """Special Python objects as values in chain."""

  async def test_ellipsis_as_root(self):
    # ... (Ellipsis) is not callable -> EVAL_RETURN_AS_IS
    await self.assertIs(Chain(...).run(), ...)

  async def test_ellipsis_as_then_literal(self):
    # Ellipsis as then() literal
    await self.assertIs(Chain(1).then(...).run(), ...)

  async def test_not_implemented_as_root(self):
    await self.assertIs(Chain(NotImplemented).run(), NotImplemented)

  async def test_not_implemented_as_then_literal(self):
    await self.assertIs(Chain(1).then(NotImplemented).run(), NotImplemented)

  async def test_bare_object_as_root(self):
    obj = object()
    await self.assertIs(Chain(obj).run(), obj)

  async def test_bare_object_as_then_literal(self):
    obj = object()
    await self.assertIs(Chain(1).then(obj).run(), obj)

  async def test_exception_instance_as_root(self):
    exc = ValueError('hello')
    await self.assertIs(Chain(exc).run(), exc)

  async def test_exception_instance_as_then_literal(self):
    exc = TypeError('world')
    await self.assertIs(Chain(1).then(exc).run(), exc)

  async def test_exception_instance_through_pipeline(self):
    exc = RuntimeError('test')
    result = Chain(exc).then(lambda v: str(v)).run()
    await self.assertEqual(result, 'test')

  async def test_exception_class_as_then_callable(self):
    # ValueError is callable (constructor). Chain(1).then(ValueError) -> ValueError(1)
    result = Chain('oops').then(ValueError).run()
    assert isinstance(result, ValueError)
    assert str(result) == 'oops'

  async def test_type_metaclass_as_root(self):
    # type is callable. Chain(type) calls type() with no args...
    # Actually type() with no args raises TypeError.
    # Chain(type) -> root is callable, EVAL_CALL_WITH_CURRENT_VALUE, cv is Null -> type()
    # type() raises TypeError (needs 1 or 3 args).
    with self.assertRaises(TypeError):
      Chain(type).run()

  async def test_type_metaclass_as_then(self):
    # Chain(42).then(type) -> type(42) -> <class 'int'>
    await self.assertIs(Chain(42).then(type).run(), int)

  async def test_property_object_as_literal(self):
    p = property(lambda self: 42)
    # property is not callable in the normal sense (it's a descriptor)
    # Actually property IS callable: property(fget) creates a property.
    # So Chain(1).then(p) would try p(1) which would fail...
    # property objects ARE callable per CPython, they create new property objects.
    # p(1) -> property(1) ... that just sets fget=1.
    result = Chain(1).then(lambda v: p).run()
    assert result is p


class CustomObjectTests(MyTestCase):
  """Custom objects with unusual dunder methods."""

  async def test_falsy_custom_object_as_root(self):
    obj = FalsyBool()
    result = Chain(obj).run()
    assert result is obj

  async def test_falsy_custom_object_survives_then(self):
    obj = FalsyBool()
    result = Chain(obj).then(lambda v: v).run()
    assert result is obj

  async def test_falsy_custom_object_survives_do(self):
    obj = FalsyBool()
    result = Chain(obj).do(lambda v: 'discard').run()
    assert result is obj

  async def test_always_equal_as_root(self):
    obj = AlwaysEqualObj()
    result = Chain(obj).run()
    assert result is obj

  async def test_unhashable_as_root(self):
    obj = UnhashableObj()
    result = Chain(obj).run()
    assert result is obj

  async def test_unhashable_through_pipeline(self):
    obj = UnhashableObj()
    result = Chain(obj).then(lambda v: v).do(lambda v: None).run()
    assert result is obj

  async def test_repr_raises_as_root(self):
    obj = ReprRaises()
    result = Chain(obj).run()
    assert result is obj

  async def test_repr_raises_through_then(self):
    obj = ReprRaises()
    result = Chain(obj).then(lambda v: v).run()
    assert result is obj


class CoroutineObjectTests(MyTestCase):
  """Coroutine objects and async detection."""

  async def test_async_fn_as_then(self):
    # Chain(1).then(aempty) -> aempty(1) -> coroutine, chain awaits it -> 1
    await self.assertEqual(Chain(1).then(aempty).run(), 1)

  async def test_lambda_returning_coroutine(self):
    # Lambda returns a coroutine object. Chain detects iscoro and awaits.
    async def make_val(v):
      return v * 10
    await self.assertEqual(Chain(5).then(lambda v: make_val(v)).run(), 50)

  async def test_async_root(self):
    # async fn as root
    await self.assertEqual(Chain(aempty, 42).run(), 42)


class GeneratorTests(MyTestCase):
  """Generator objects and iterables."""

  async def test_generator_function_as_root(self):
    # dummy_gen is callable, Chain(dummy_gen) calls it -> generator object
    gen = Chain(dummy_gen).run()
    assert list(gen) == [1, 2, 3]

  async def test_range_as_root(self):
    # range is callable, Chain(range, 5) calls range(5) -> range(0,5)
    await self.assertEqual(Chain(range, 5).run(), range(5))

  async def test_generator_object_as_literal(self):
    # A generator object is not callable, so .then(gen_obj) -> EVAL_RETURN_AS_IS
    gen = dummy_gen()
    result = Chain(1).then(gen).run()
    # The generator object itself is returned as-is
    assert result is gen


class ForeachValueTypeTests(MyTestCase):
  """foreach with various iterable types."""

  async def test_foreach_list(self):
    await self.assertEqual(
      Chain([1, 2, 3]).foreach(lambda x: x * 2).run(), [2, 4, 6]
    )

  async def test_foreach_tuple(self):
    await self.assertEqual(
      Chain((10, 20, 30)).foreach(lambda x: x + 1).run(), [11, 21, 31]
    )

  async def test_foreach_set(self):
    result = Chain({1, 2, 3}).foreach(lambda x: x * 10).run()
    assert sorted(result) == [10, 20, 30]

  async def test_foreach_dict_iterates_keys(self):
    result = Chain({'a': 1, 'b': 2}).foreach(lambda k: k.upper()).run()
    assert sorted(result) == ['A', 'B']

  async def test_foreach_range(self):
    await self.assertEqual(
      Chain(range(4)).foreach(lambda x: x ** 2).run(), [0, 1, 4, 9]
    )

  async def test_foreach_string(self):
    await self.assertEqual(
      Chain('abc').foreach(lambda c: c.upper()).run(), ['A', 'B', 'C']
    )

  async def test_foreach_bytes(self):
    # bytes iterates as ints
    await self.assertEqual(
      Chain(b'\x01\x02\x03').foreach(lambda b: b * 2).run(), [2, 4, 6]
    )

  async def test_foreach_custom_iterable(self):
    await self.assertEqual(
      Chain(CustomIterable([10, 20])).foreach(lambda x: x + 5).run(), [15, 25]
    )

  async def test_foreach_async_iterable(self):
    await self.assertEqual(
      Chain(CustomAsyncIterable([5, 10])).foreach(lambda x: x * 3).run(), [15, 30]
    )

  async def test_foreach_frozenset(self):
    result = Chain(frozenset({4, 5})).foreach(lambda x: x + 1).run()
    assert sorted(result) == [5, 6]

  async def test_foreach_empty_list(self):
    await self.assertEqual(Chain([]).foreach(lambda x: x).run(), [])

  async def test_foreach_non_iterable_raises(self):
    with self.assertRaises(AttributeError):
      Chain(42).foreach(lambda x: x).run()

  async def test_foreach_bytearray(self):
    await self.assertEqual(
      Chain(bytearray(b'\x01\x02')).foreach(lambda b: b + 10).run(), [11, 12]
    )


class FilterValueTypeTests(MyTestCase):
  """filter with various iterable types."""

  async def test_filter_list(self):
    await self.assertEqual(
      Chain([0, 1, 2, 0, 3]).filter(bool).run(), [1, 2, 3]
    )

  async def test_filter_tuple(self):
    await self.assertEqual(
      Chain((0, '', 'a', None, 'b')).filter(bool).run(), ['a', 'b']
    )

  async def test_filter_string(self):
    await self.assertEqual(
      Chain('aAbBcC').filter(lambda c: c.isupper()).run(), ['A', 'B', 'C']
    )

  async def test_filter_range(self):
    await self.assertEqual(
      Chain(range(10)).filter(lambda x: x % 2 == 0).run(), [0, 2, 4, 6, 8]
    )

  async def test_filter_empty_list(self):
    await self.assertEqual(Chain([]).filter(bool).run(), [])

  async def test_filter_custom_iterable(self):
    await self.assertEqual(
      Chain(CustomIterable([0, 1, 0, 2])).filter(bool).run(), [1, 2]
    )

  async def test_filter_async_iterable(self):
    await self.assertEqual(
      Chain(CustomAsyncIterable([0, 1, 0, 2])).filter(bool).run(), [1, 2]
    )


class GatherValueTypeTests(MyTestCase):
  """gather with different function return types."""

  async def test_gather_falsy_returns(self):
    result = Chain(1).gather(
      lambda v: 0,
      lambda v: False,
      lambda v: '',
      lambda v: [],
      lambda v: None,
    ).run()
    await self.assertEqual(result, [0, False, '', [], None])

  async def test_gather_mixed_types(self):
    result = Chain(10).gather(
      lambda v: v * 2,
      lambda v: str(v),
      lambda v: [v],
      lambda v: {'val': v},
    ).run()
    await self.assertEqual(result, [20, '10', [10], {'val': 10}])

  async def test_gather_async_and_sync(self):
    result = Chain(5).gather(
      lambda v: v + 1,
      lambda v: aempty(v + 2),
    ).run()
    await self.assertEqual(result, [6, 7])


class WithContextManagerTests(MyTestCase):
  """with_ feature with various context manager types."""

  async def test_sync_cm(self):
    cm = SyncCM('test_val')
    result = Chain(cm).with_(lambda ctx: ctx + '_processed').run()
    await self.assertEqual(result, 'test_val_processed')
    assert cm.entered
    assert cm.exited

  async def test_async_cm(self):
    cm = AsyncCM('async_val')
    result = Chain(cm).with_(lambda ctx: ctx + '_done').run()
    await self.assertEqual(result, 'async_val_done')
    assert cm.entered
    assert cm.exited

  async def test_non_cm_raises(self):
    # 42 is not a context manager
    with self.assertRaises(AttributeError):
      Chain(42).with_(lambda ctx: ctx).run()


class NestedChainValueTests(MyTestCase):
  """Nested chains with various value types."""

  async def test_nested_chain_with_falsy_result(self):
    await self.assertEqual(
      Chain(1).then(Chain().then(lambda v: 0)).run(), 0
    )

  async def test_nested_chain_with_none_result(self):
    await self.assertIsNone(
      Chain(1).then(Chain().then(lambda v: None)).run()
    )

  async def test_nested_chain_with_false_result(self):
    await self.assertIs(
      Chain(1).then(Chain().then(lambda v: False)).run(), False
    )

  async def test_nested_chain_with_empty_string_result(self):
    await self.assertEqual(
      Chain(1).then(Chain().then(lambda v: '')).run(), ''
    )

  async def test_nested_chain_with_container_result(self):
    await self.assertEqual(
      Chain(1).then(Chain().then(lambda v: [v, v + 1])).run(), [1, 2]
    )

  async def test_nested_chain_with_exception_obj_result(self):
    exc = ValueError('nested')
    result = Chain(1).then(Chain().then(lambda v: exc)).run()
    assert result is exc

  async def test_triple_nested_with_falsy(self):
    await self.assertEqual(
      Chain(1).then(Chain().then(Chain().then(lambda v: 0))).run(), 0
    )


class CascadeValueTests(MyTestCase):
  """Cascade with various value types — root value is always passed to each link."""

  async def test_cascade_with_falsy_root(self):
    calls = []
    result = Cascade(0).then(lambda v: calls.append(v)).then(lambda v: calls.append(v)).run()
    await self.assertEqual(result, 0)
    assert calls == [0, 0]

  async def test_cascade_with_none_root(self):
    calls = []
    result = Cascade(None).then(lambda v: calls.append(v)).run()
    await self.assertIsNone(result)
    assert calls == [None]

  async def test_cascade_with_false_root(self):
    calls = []
    result = Cascade(False).then(lambda v: calls.append(v)).run()
    await self.assertIs(result, False)
    assert calls == [False]

  async def test_cascade_with_empty_string_root(self):
    calls = []
    result = Cascade('').then(lambda v: calls.append(v)).run()
    await self.assertEqual(result, '')
    assert calls == ['']

  async def test_cascade_with_container_root(self):
    lst = [1, 2, 3]
    calls = []
    result = Cascade(lst).then(lambda v: calls.append(len(v))).run()
    assert result is lst
    assert calls == [3]

  async def test_cascade_with_bare_object_root(self):
    obj = object()
    result = Cascade(obj).then(lambda v: None).run()
    assert result is obj


class FrozenChainValueTests(MyTestCase):
  """Frozen chains with various value types."""

  async def test_frozen_chain_with_falsy_root(self):
    frozen = Chain(0).freeze()
    await self.assertEqual(frozen(), 0)

  async def test_frozen_chain_with_false_root(self):
    frozen = Chain(False).freeze()
    await self.assertIs(frozen(), False)

  async def test_frozen_chain_with_none_root(self):
    frozen = Chain(None).freeze()
    await self.assertIsNone(frozen())

  async def test_frozen_chain_with_empty_string_root(self):
    frozen = Chain('').freeze()
    await self.assertEqual(frozen(), '')

  async def test_frozen_chain_with_object_root(self):
    obj = object()
    frozen = Chain(obj).freeze()
    assert frozen() is obj

  async def test_frozen_chain_callable_preserves_values(self):
    frozen = Chain(0).then(lambda v: v).freeze()
    await self.assertEqual(frozen(), 0)

  async def test_frozen_chain_reusable_with_falsy(self):
    frozen = Chain(0).then(lambda v: v + 1).freeze()
    await self.assertEqual(frozen(), 1)
    await self.assertEqual(frozen(), 1)


class ReturnValueTests(MyTestCase):
  """Chain.return_() with various value types."""

  async def test_return_falsy_zero(self):
    result = Chain(Chain().then(lambda: Chain.return_(0))).then(lambda v: 999).run()
    await self.assertEqual(result, 0)

  async def test_return_false(self):
    result = Chain(Chain().then(lambda: Chain.return_(False))).then(lambda v: 999).run()
    await self.assertIs(result, False)

  async def test_return_empty_string(self):
    result = Chain(Chain().then(lambda: Chain.return_(''))).then(lambda v: 999).run()
    await self.assertEqual(result, '')

  async def test_return_none(self):
    # Chain.return_() with no args -> _Return with Null value -> handle_return_exc returns None
    result = Chain(Chain().then(lambda: Chain.return_())).then(lambda v: 999).run()
    await self.assertIsNone(result)

  async def test_return_container(self):
    lst = [1, 2, 3]
    result = Chain(Chain().then(lambda: Chain.return_(lst))).then(lambda v: 999).run()
    assert result is lst

  async def test_return_exception_object(self):
    exc = ValueError('returned')
    result = Chain(Chain().then(lambda: Chain.return_(exc))).then(lambda v: 999).run()
    assert result is exc


class BreakValueTests(MyTestCase):
  """Chain.break_() with various value types in foreach context."""

  async def test_break_with_zero(self):
    def fn(x):
      if x == 2:
        Chain.break_(0)
      return x
    result = Chain([1, 2, 3]).foreach(fn).run()
    await self.assertEqual(result, 0)

  async def test_break_with_false(self):
    def fn(x):
      if x == 2:
        Chain.break_(False)
      return x
    result = Chain([1, 2, 3]).foreach(fn).run()
    await self.assertIs(result, False)

  async def test_break_with_empty_string(self):
    def fn(x):
      if x == 2:
        Chain.break_('')
      return x
    result = Chain([1, 2, 3]).foreach(fn).run()
    await self.assertEqual(result, '')

  async def test_break_with_none_fallback(self):
    # break_() with no args -> Null -> fallback to collected list
    def fn(x):
      if x == 2:
        Chain.break_()
      return x * 10
    result = Chain([1, 2, 3]).foreach(fn).run()
    await self.assertEqual(result, [10])

  async def test_break_with_container(self):
    sentinel = {'done': True}
    def fn(x):
      if x == 2:
        Chain.break_(sentinel)
      return x
    result = Chain([1, 2, 3]).foreach(fn).run()
    assert result is sentinel


class PipeOperatorTests(MyTestCase):
  """Pipe operator (|) with various value types."""

  async def test_pipe_falsy_literal(self):
    result = Chain(1) | 0 | run()
    await self.assertEqual(result, 0)

  async def test_pipe_false_literal(self):
    result = Chain(1) | False | run()
    await self.assertIs(result, False)

  async def test_pipe_empty_string(self):
    result = Chain(1) | '' | run()
    await self.assertEqual(result, '')

  async def test_pipe_none_literal(self):
    result = Chain(1) | None | run()
    await self.assertIsNone(result)

  async def test_pipe_callable(self):
    result = Chain(5) | (lambda v: v * 2) | run()
    await self.assertEqual(result, 10)

  async def test_pipe_container(self):
    lst = [1, 2, 3]
    result = Chain(1) | lst | run()
    assert result is lst


class ExceptionObjectAsValueTests(MyTestCase):
  """Exception instances used as normal values, not raised."""

  async def test_value_error_as_root(self):
    exc = ValueError('root_exc')
    result = Chain(exc).then(lambda v: type(v).__name__).run()
    await self.assertEqual(result, 'ValueError')

  async def test_type_error_through_pipeline(self):
    exc = TypeError('pipeline')
    result = Chain(exc).then(lambda v: v).do(lambda v: None).then(lambda v: str(v)).run()
    await self.assertEqual(result, 'pipeline')

  async def test_exception_as_except_return(self):
    sentinel_exc = RuntimeError('sentinel')
    result = Chain(1).then(raise_).except_(lambda v: sentinel_exc, reraise=False).run()
    assert result is sentinel_exc

  async def test_exception_in_gather(self):
    exc = ValueError('gather_exc')
    result = Chain(1).gather(lambda v: exc, lambda v: v + 1).run()
    assert result[0] is exc
    await self.assertEqual(result[1], 2)


class CloneValueTests(MyTestCase):
  """Clone preserves behavior with various value types."""

  async def test_clone_preserves_falsy_root(self):
    c = Chain(0).then(lambda v: v + 1)
    c2 = c.clone()
    await self.assertEqual(c.run(), 1)
    await self.assertEqual(c2.run(), 1)

  async def test_clone_preserves_false_root(self):
    c = Chain(False).then(lambda v: not v)
    c2 = c.clone()
    await self.assertIs(c.run(), True)
    await self.assertIs(c2.run(), True)

  async def test_clone_preserves_container_root(self):
    c = Chain([1, 2]).then(lambda v: v + [3])
    c2 = c.clone()
    await self.assertEqual(c.run(), [1, 2, 3])
    await self.assertEqual(c2.run(), [1, 2, 3])


class ForeachWithIndexValueTests(MyTestCase):
  """foreach with_index and various types."""

  async def test_foreach_indexed_list(self):
    await self.assertEqual(
      Chain([10, 20, 30]).foreach(lambda i, el: (i, el), with_index=True).run(),
      [(0, 10), (1, 20), (2, 30)]
    )

  async def test_foreach_indexed_string(self):
    await self.assertEqual(
      Chain('abc').foreach(lambda i, c: f'{i}:{c}', with_index=True).run(),
      ['0:a', '1:b', '2:c']
    )

  async def test_foreach_indexed_empty(self):
    await self.assertEqual(
      Chain([]).foreach(lambda i, el: (i, el), with_index=True).run(), []
    )


class RunOverrideValueTests(MyTestCase):
  """Chain().run(value) with various value types as root override."""

  async def test_run_override_with_falsy_zero(self):
    await self.assertEqual(Chain().then(lambda v: v + 1).run(0), 1)

  async def test_run_override_with_false(self):
    await self.assertIs(Chain().then(lambda v: v).run(False), False)

  async def test_run_override_with_empty_string(self):
    await self.assertEqual(Chain().then(lambda v: v).run(''), '')

  async def test_run_override_with_none(self):
    await self.assertIsNone(Chain().then(lambda v: v).run(None))

  async def test_run_override_with_callable(self):
    # Chain().run(int, '42') -> int is callable, called with args ('42',) -> 42
    await self.assertEqual(Chain().then(lambda v: v).run(int, '42'), 42)


class DoAsyncFalsyTests(MyTestCase):
  """do() with async functions preserves falsy current value."""

  async def test_do_async_preserves_zero(self):
    await self.assertEqual(
      Chain(0).do(aempty).run(), 0
    )

  async def test_do_async_preserves_false(self):
    await self.assertIs(
      Chain(False).do(aempty).run(), False
    )

  async def test_do_async_preserves_empty_string(self):
    await self.assertEqual(
      Chain('').do(aempty).run(), ''
    )

  async def test_do_async_preserves_none(self):
    await self.assertIsNone(
      Chain(None).do(aempty).run()
    )


class FalsyInExceptAsyncTests(MyTestCase):
  """Async except_ handlers returning falsy values."""

  async def test_async_except_returns_zero(self):
    async def handler(v):
      return 0
    await self.assertEqual(
      Chain(1).then(raise_).except_(handler, reraise=False).run(), 0
    )

  async def test_async_except_returns_false(self):
    async def handler(v):
      return False
    await self.assertIs(
      Chain(1).then(raise_).except_(handler, reraise=False).run(), False
    )

  async def test_async_except_returns_none(self):
    async def handler(v):
      return None
    await self.assertIsNone(
      Chain(1).then(raise_).except_(handler, reraise=False).run()
    )


class FalsyInFinallyTests(MyTestCase):
  """finally_ with falsy values still executes properly."""

  async def test_finally_runs_with_falsy_root(self):
    ran = []
    result = Chain(0).then(lambda v: v).finally_(lambda v: ran.append(v)).run()
    await self.assertEqual(result, 0)
    assert ran == [0]

  async def test_finally_runs_with_false_root(self):
    ran = []
    result = Chain(False).then(lambda v: v).finally_(lambda v: ran.append(v)).run()
    await self.assertIs(result, False)
    assert ran == [False]

  async def test_finally_runs_with_none_root(self):
    ran = []
    result = Chain(None).then(lambda v: v).finally_(lambda v: ran.append(v)).run()
    await self.assertIsNone(result)
    assert ran == [None]


class NoAsyncSyncOnlyFalsyTests(MyTestCase):
  """no_async mode preserves falsy values."""

  async def test_no_async_with_zero(self):
    await self.assertEqual(
      Chain(0).no_async(True).then(lambda v: v).run(), 0
    )

  async def test_no_async_with_false(self):
    await self.assertIs(
      Chain(False).no_async(True).then(lambda v: v).run(), False
    )

  async def test_no_async_with_empty_string(self):
    await self.assertEqual(
      Chain('').no_async(True).then(lambda v: v).run(), ''
    )

  async def test_no_async_with_none(self):
    await self.assertIsNone(
      Chain(None).no_async(True).then(lambda v: v).run()
    )


class FalsyValueNestedCascadeTests(MyTestCase):
  """Cascade inside Chain with falsy roots."""

  async def test_cascade_nested_falsy_zero(self):
    result = Chain(0).then(Cascade().then(lambda v: v + 1)).run()
    await self.assertEqual(result, 0)

  async def test_cascade_nested_false(self):
    result = Chain(False).then(Cascade().then(lambda v: not v)).run()
    await self.assertIs(result, False)


class IterateValueTypeTests(MyTestCase):
  """iterate() with various types."""

  async def test_iterate_list(self):
    result = []
    for item in Chain([10, 20, 30]).iterate():
      result.append(item)
    assert result == [10, 20, 30]

  async def test_iterate_with_transform(self):
    result = []
    for item in Chain([1, 2, 3]).iterate(lambda x: x * 10):
      result.append(item)
    assert result == [10, 20, 30]

  async def test_iterate_tuple(self):
    result = []
    for item in Chain(('a', 'b')).iterate():
      result.append(item)
    assert result == ['a', 'b']

  async def test_iterate_range(self):
    result = []
    for item in Chain(range(3)).iterate():
      result.append(item)
    assert result == [0, 1, 2]

  async def test_iterate_string(self):
    result = []
    for item in Chain('hi').iterate():
      result.append(item)
    assert result == ['h', 'i']

  async def test_async_iterate_list(self):
    result = []
    async for item in Chain([1, 2]).iterate():
      result.append(item)
    assert result == [1, 2]


class BoolCoercionTests(MyTestCase):
  """Chain.__bool__ always returns True."""

  async def test_chain_bool_always_true_for_zero(self):
    assert bool(Chain(0)) is True

  async def test_chain_bool_always_true_for_none(self):
    assert bool(Chain(None)) is True

  async def test_chain_bool_always_true_for_false(self):
    assert bool(Chain(False)) is True

  async def test_chain_bool_always_true_for_empty_list(self):
    assert bool(Chain([])) is True

  async def test_chain_bool_always_true_for_void(self):
    assert bool(Chain()) is True


class MixedPipelineFalsyTests(MyTestCase):
  """Complex pipelines mixing multiple falsy values."""

  async def test_zero_to_false_to_none_to_empty_string(self):
    result = (
      Chain(0)
      .then(lambda v: False if v == 0 else v)
      .then(lambda v: None if v is False else v)
      .then(lambda v: '' if v is None else v)
      .run()
    )
    await self.assertEqual(result, '')

  async def test_falsy_cascade_through_do(self):
    side_effects = []
    result = (
      Chain(0)
      .do(lambda v: side_effects.append(v))
      .then(lambda v: '')
      .do(lambda v: side_effects.append(v))
      .then(lambda v: False)
      .do(lambda v: side_effects.append(v))
      .run()
    )
    await self.assertIs(result, False)
    assert side_effects == [0, '', False]


class GatherFalsyTests(MyTestCase):
  """gather where all functions return falsy values."""

  async def test_gather_all_falsy(self):
    result = Chain(1).gather(
      lambda v: 0,
      lambda v: False,
      lambda v: '',
      lambda v: None,
      lambda v: [],
    ).run()
    await self.assertEqual(result, [0, False, '', None, []])

  async def test_gather_with_async_falsy(self):
    async def return_zero(v):
      return 0
    async def return_false(v):
      return False
    result = Chain(1).gather(return_zero, return_false, lambda v: '').run()
    await self.assertEqual(result, [0, False, ''])


class ComplexNestedFalsyTests(MyTestCase):
  """Deeply nested chains with falsy values propagating."""

  async def test_deep_nesting_falsy(self):
    inner = Chain().then(lambda v: 0)
    middle = Chain().then(inner)
    result = Chain(1).then(middle).run()
    await self.assertEqual(result, 0)

  async def test_deep_nesting_false(self):
    inner = Chain().then(lambda v: False)
    middle = Chain().then(inner)
    result = Chain(1).then(middle).run()
    await self.assertIs(result, False)

  async def test_deep_nesting_empty_string(self):
    inner = Chain().then(lambda v: '')
    middle = Chain().then(inner)
    result = Chain(1).then(middle).run()
    await self.assertEqual(result, '')

  async def test_deep_nesting_none(self):
    inner = Chain().then(lambda v: None)
    middle = Chain().then(inner)
    result = Chain(1).then(middle).run()
    await self.assertIsNone(result)


class FalsyReturnInExceptTests(MyTestCase):
  """return_() with falsy values inside except_ handlers."""

  async def test_return_zero_from_except(self):
    result = Chain(
      Chain(1).then(raise_).except_(lambda v: Chain.return_(0))
    ).then(lambda v: 999).run()
    await self.assertEqual(result, 0)

  async def test_return_false_from_except(self):
    result = Chain(
      Chain(1).then(raise_).except_(lambda v: Chain.return_(False))
    ).then(lambda v: 999).run()
    await self.assertIs(result, False)

  async def test_return_empty_string_from_except(self):
    result = Chain(
      Chain(1).then(raise_).except_(lambda v: Chain.return_(''))
    ).then(lambda v: 999).run()
    await self.assertEqual(result, '')


class SpecialObjectThroughPipelineTests(MyTestCase):
  """Special Python objects surviving full pipelines."""

  async def test_ellipsis_through_full_pipeline(self):
    result = Chain(...).then(lambda v: v).do(lambda v: None).run()
    assert result is ...

  async def test_not_implemented_through_pipeline(self):
    result = Chain(NotImplemented).then(lambda v: v).do(lambda v: None).run()
    assert result is NotImplemented

  async def test_bare_object_through_pipeline(self):
    obj = object()
    result = Chain(obj).then(lambda v: v).do(lambda v: None).run()
    assert result is obj

  async def test_exception_instance_full_pipeline(self):
    exc = RuntimeError('full_pipeline')
    result = Chain(exc).then(lambda v: v).do(lambda v: None).then(lambda v: v).run()
    assert result is exc

  async def test_frozenset_as_root(self):
    fs = frozenset({1, 2, 3})
    result = Chain(fs).then(lambda v: v).run()
    assert result is fs


class CallableEdgeCaseTests(MyTestCase):
  """Additional edge cases for callable handling."""

  async def test_print_is_callable_as_do(self):
    # print is callable; .do(print) calls print(current_value) and discards result
    # Just verify no crash (print returns None, which is discarded by do)
    result = Chain(42).do(print).run()
    await self.assertEqual(result, 42)

  async def test_sorted_as_then(self):
    await self.assertEqual(
      Chain([3, 1, 2]).then(sorted).run(), [1, 2, 3]
    )

  async def test_list_constructor_as_then(self):
    await self.assertEqual(
      Chain(range(3)).then(list).run(), [0, 1, 2]
    )

  async def test_tuple_constructor_as_then(self):
    await self.assertEqual(
      Chain([1, 2, 3]).then(tuple).run(), (1, 2, 3)
    )

  async def test_set_constructor_as_then(self):
    await self.assertEqual(
      Chain([1, 2, 2, 3]).then(set).run(), {1, 2, 3}
    )

  async def test_dict_constructor_as_then(self):
    await self.assertEqual(
      Chain([('a', 1), ('b', 2)]).then(dict).run(), {'a': 1, 'b': 2}
    )

  async def test_str_constructor_as_root(self):
    # str() with no args returns ''
    await self.assertEqual(Chain(str).run(), '')

  async def test_bool_constructor_as_root(self):
    # bool() -> False
    await self.assertIs(Chain(bool).run(), False)

  async def test_complex_constructor_as_root(self):
    # complex() -> 0j
    await self.assertEqual(Chain(complex).run(), 0j)

  async def test_bytes_constructor_as_root(self):
    # bytes() -> b''
    # Actually bytes() with no args: bytes(0) works, bytes() works in Python 3
    await self.assertEqual(Chain(bytes).run(), b'')


class FalsyInAsyncExceptTests(MyTestCase):
  """Async chains that encounter exceptions and return falsy from handler."""

  async def test_async_chain_except_returns_zero(self):
    async def raiser(v):
      raise Exception('boom')
    await self.assertEqual(
      Chain(1).then(aempty).then(raiser).except_(lambda v: 0, reraise=False).run(), 0
    )

  async def test_async_chain_except_returns_false(self):
    async def raiser(v):
      raise Exception('boom')
    await self.assertIs(
      Chain(1).then(aempty).then(raiser).except_(lambda v: False, reraise=False).run(), False
    )


class ConfigDebugValueTests(MyTestCase):
  """config(debug=True) with various value types."""

  async def test_debug_with_falsy_zero(self):
    result = Chain(0).config(debug=True).then(lambda v: v + 1).run()
    await self.assertEqual(result, 1)

  async def test_debug_with_false(self):
    result = Chain(False).config(debug=True).then(lambda v: not v).run()
    await self.assertIs(result, True)

  async def test_debug_with_none(self):
    result = Chain(None).config(debug=True).then(lambda v: v).run()
    await self.assertIsNone(result)

  async def test_debug_with_object(self):
    obj = object()
    result = Chain(obj).config(debug=True).then(lambda v: v).run()
    assert result is obj


class ForeachBreakFalsyTests(MyTestCase):
  """foreach break with falsy break values."""

  async def test_foreach_break_with_zero_value(self):
    def fn(x):
      if x == 3:
        Chain.break_(0)
      return x
    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(result, 0)

  async def test_foreach_break_with_false_value(self):
    def fn(x):
      if x == 3:
        Chain.break_(False)
      return x
    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertIs(result, False)

  async def test_foreach_break_with_empty_list_value(self):
    def fn(x):
      if x == 3:
        Chain.break_([])
      return x
    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(result, [])

  async def test_foreach_break_with_none_fallback(self):
    """break_() with no value uses fallback (accumulated list)."""
    def fn(x):
      if x == 3:
        Chain.break_()
      return x * 10
    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(result, [10, 20])


class SleepValuePreservationTests(MyTestCase):
  """sleep() preserves current value (is a do-like operation)."""

  async def test_sleep_preserves_zero(self):
    await self.assertEqual(
      Chain(0).sleep(0.001).run(), 0
    )

  async def test_sleep_preserves_false(self):
    await self.assertIs(
      Chain(False).sleep(0.001).run(), False
    )

  async def test_sleep_preserves_object(self):
    obj = object()
    result = await await_(Chain(obj).sleep(0.001).run())
    assert result is obj


class ToThreadValueTests(MyTestCase):
  """to_thread with various value types."""

  async def test_to_thread_with_identity(self):
    result = await await_(Chain(42).to_thread(lambda v: v).run())
    await self.assertEqual(result, 42)

  async def test_to_thread_with_falsy_zero(self):
    result = await await_(Chain(0).to_thread(lambda v: v).run())
    await self.assertEqual(result, 0)

  async def test_to_thread_with_false(self):
    result = await await_(Chain(False).to_thread(lambda v: v).run())
    assert result is False


class FalsyValueReturnFromChainRunTests(MyTestCase):
  """Verify that the _run method's final 'if current_value is Null: return None'
  does NOT incorrectly convert falsy values to None."""

  async def test_zero_is_not_none(self):
    result = Chain(0).run()
    assert result is not None
    assert result == 0

  async def test_false_is_not_none(self):
    result = Chain(False).run()
    assert result is not None
    assert result is False

  async def test_empty_string_is_not_none(self):
    result = Chain('').run()
    assert result is not None
    assert result == ''

  async def test_empty_list_is_not_none(self):
    result = Chain([]).run()
    assert result is not None
    assert result == []

  async def test_empty_dict_is_not_none(self):
    result = Chain({}).run()
    assert result is not None
    assert result == {}

  async def test_zero_complex_is_not_none(self):
    result = Chain(0j).run()
    assert result is not None
    assert result == 0j

  async def test_zero_float_is_not_none(self):
    result = Chain(0.0).run()
    assert result is not None
    assert result == 0.0
