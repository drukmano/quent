"""Exhaustive edge-case value/type tests for the Quent library.

Tests every conceivable edge-case value type through Chain and Cascade,
verifying that falsy values, special numerics, exotic types, containers,
callables, strings, bytes, the Null sentinel, bool protocol, and complex
value combinations all behave correctly.
"""
import asyncio
import math
import operator
import functools
import sys
from collections import OrderedDict, defaultdict, Counter, namedtuple
from dataclasses import dataclass, field
from unittest import TestCase, IsolatedAsyncioTestCase

from tests.utils import empty, aempty, await_, MyTestCase
from quent import Chain, Cascade, QuentException, run
from quent.quent import PyNull as Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CallableClass:
  """A class whose instances are callable."""
  def __init__(self, multiplier=2):
    self.multiplier = multiplier

  def __call__(self, v=None):
    if v is None:
      return self.multiplier
    return v * self.multiplier


class BoundMethodHolder:
  """Provides bound methods for testing."""
  def __init__(self, value):
    self.value = value

  def get(self):
    return self.value

  def transform(self, v):
    return v + self.value

  @staticmethod
  def static_double(v):
    return v * 2

  @classmethod
  def class_name(cls, v=None):
    return cls.__name__


class CustomBoolFalsy:
  """Object that is falsy via __bool__."""
  def __bool__(self):
    return False


class CustomBoolTruthy:
  """Object that is truthy via __bool__."""
  def __bool__(self):
    return True


class CustomRepr:
  """Object with custom __repr__."""
  def __repr__(self):
    return '<CustomRepr>'


class CustomLen:
  """Object that supports len()."""
  def __init__(self, n):
    self._n = n

  def __len__(self):
    return self._n


class CustomIter:
  """Object that is iterable with custom __iter__."""
  def __init__(self, items):
    self._items = items

  def __iter__(self):
    return iter(self._items)


class CustomContextManager:
  """A simple sync context manager."""
  def __init__(self, value):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCustomContextManager:
  """A simple async context manager."""
  def __init__(self, value):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


Point = namedtuple('Point', ['x', 'y'])


@dataclass
class DataPoint:
  x: float = 0.0
  y: float = 0.0
  label: str = ''


async def async_identity(v):
  return v


# ===================================================================
# 1. FALSY VALUES AS ROOT
# ===================================================================

class FalsyValuesAsRootTests(TestCase):
  """Falsy values used as root must pass through the chain correctly,
  never confused with the internal Null sentinel."""

  def test_none_as_root(self):
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_false_as_root(self):
    result = Chain(False).run()
    self.assertIs(result, False)

  def test_zero_int_as_root(self):
    result = Chain(0).run()
    self.assertEqual(result, 0)
    self.assertIsInstance(result, int)

  def test_zero_float_as_root(self):
    result = Chain(0.0).run()
    self.assertEqual(result, 0.0)
    self.assertIsInstance(result, float)

  def test_zero_complex_as_root(self):
    result = Chain(0j).run()
    self.assertEqual(result, 0j)
    self.assertIsInstance(result, complex)

  def test_empty_string_as_root(self):
    result = Chain('').run()
    self.assertEqual(result, '')

  def test_empty_bytes_as_root(self):
    result = Chain(b'').run()
    self.assertEqual(result, b'')

  def test_empty_list_as_root(self):
    result = Chain([]).run()
    self.assertEqual(result, [])

  def test_empty_tuple_as_root(self):
    result = Chain(()).run()
    self.assertEqual(result, ())

  def test_empty_dict_as_root(self):
    result = Chain({}).run()
    self.assertEqual(result, {})

  def test_empty_set_as_root(self):
    result = Chain(set()).run()
    self.assertEqual(result, set())

  def test_empty_frozenset_as_root(self):
    result = Chain(frozenset()).run()
    self.assertEqual(result, frozenset())

  def test_false_then_identity_returns_false(self):
    result = Chain(False).then(lambda v: v).run()
    self.assertIs(result, False)

  def test_zero_then_identity_returns_zero(self):
    result = Chain(0).then(lambda v: v).run()
    self.assertEqual(result, 0)

  def test_empty_string_then_equality_check(self):
    result = Chain('').then(lambda v: v == '').run()
    self.assertTrue(result)

  def test_none_then_is_none_check(self):
    result = Chain(None).then(lambda v: v is None).run()
    self.assertTrue(result)

  def test_custom_falsy_object_as_root(self):
    obj = CustomBoolFalsy()
    result = Chain(obj).run()
    self.assertIs(result, obj)
    self.assertFalse(bool(result))

  def test_negative_zero_as_root(self):
    result = Chain(-0.0).run()
    self.assertEqual(result, -0.0)
    self.assertIsInstance(result, float)

  def test_empty_bytearray_as_root(self):
    result = Chain(bytearray()).run()
    self.assertEqual(result, bytearray())


# ===================================================================
# 2. FALSY VALUES PROPAGATING THROUGH CHAINS
# ===================================================================

class FalsyValuesPropagatingTests(TestCase):
  """Falsy values returned by .then() must propagate to the next link."""

  def test_none_propagates_through_then(self):
    result = Chain(1).then(lambda v: None).then(lambda v: v is None).run()
    self.assertTrue(result)

  def test_none_received_as_none_not_null(self):
    result = Chain(1).then(lambda v: None).then(lambda v: type(v)).run()
    self.assertIs(result, type(None))

  def test_zero_propagates_through_then(self):
    result = Chain(1).then(lambda v: 0).then(lambda v: v).run()
    self.assertEqual(result, 0)

  def test_zero_propagates_type_check(self):
    result = Chain(1).then(lambda v: 0).then(lambda v: isinstance(v, int)).run()
    self.assertTrue(result)

  def test_false_propagates_through_then(self):
    result = Chain(1).then(lambda v: False).then(lambda v: v).run()
    self.assertIs(result, False)

  def test_false_propagates_type_check(self):
    result = Chain(1).then(lambda v: False).then(lambda v: isinstance(v, bool)).run()
    self.assertTrue(result)

  def test_empty_string_propagates_through_then(self):
    result = Chain(1).then(lambda v: '').then(lambda v: v).run()
    self.assertEqual(result, '')

  def test_empty_list_propagates_through_then(self):
    result = Chain(1).then(lambda v: []).then(lambda v: v).run()
    self.assertEqual(result, [])

  def test_empty_dict_propagates_through_then(self):
    result = Chain(1).then(lambda v: {}).then(lambda v: len(v)).run()
    self.assertEqual(result, 0)

  def test_empty_tuple_propagates_through_then(self):
    result = Chain(1).then(lambda v: ()).then(lambda v: v).run()
    self.assertEqual(result, ())

  def test_cascade_with_none_root(self):
    """Cascade with None root — all links receive None."""
    received = []
    def capture(v):
      received.append(v)
    Cascade(None).then(capture).then(capture).run()
    self.assertEqual(received, [None, None])

  def test_cascade_with_false_root(self):
    """Cascade with False root — all links receive False, returns False."""
    received = []
    def capture(v):
      received.append(v)
    result = Cascade(False).then(capture).then(capture).run()
    self.assertIs(result, False)
    self.assertEqual(received, [False, False])

  def test_cascade_with_zero_root(self):
    received = []
    def capture(v):
      received.append(v)
    result = Cascade(0).then(capture).then(capture).run()
    self.assertEqual(result, 0)
    self.assertEqual(received, [0, 0])

  def test_cascade_with_empty_string_root(self):
    received = []
    def capture(v):
      received.append(v)
    result = Cascade('').then(capture).then(capture).run()
    self.assertEqual(result, '')
    self.assertEqual(received, ['', ''])

  def test_falsy_propagates_through_multiple_links(self):
    result = (
      Chain(42)
      .then(lambda v: 0)
      .then(lambda v: v)
      .then(lambda v: v + 10)
      .run()
    )
    self.assertEqual(result, 10)

  def test_none_propagates_through_do(self):
    """do() discards its result, so the previous value (None) survives."""
    result = (
      Chain(1)
      .then(lambda v: None)
      .do(lambda v: 999)
      .run()
    )
    self.assertIsNone(result)

  def test_false_propagates_through_do(self):
    result = (
      Chain(1)
      .then(lambda v: False)
      .do(lambda v: 999)
      .run()
    )
    self.assertIs(result, False)


# ===================================================================
# 3. SPECIAL NUMERIC VALUES
# ===================================================================

class SpecialNumericValuesTests(TestCase):
  """Special numeric edge cases: inf, nan, complex, very large/small."""

  def test_positive_infinity_as_root(self):
    result = Chain(float('inf')).run()
    self.assertEqual(result, float('inf'))
    self.assertTrue(math.isinf(result))

  def test_negative_infinity_as_root(self):
    result = Chain(float('-inf')).run()
    self.assertEqual(result, float('-inf'))
    self.assertTrue(math.isinf(result))

  def test_nan_as_root(self):
    result = Chain(float('nan')).run()
    self.assertTrue(math.isnan(result))

  def test_nan_propagates_through_chain(self):
    result = Chain(float('nan')).then(lambda v: v).run()
    self.assertTrue(math.isnan(result))

  def test_nan_identity_check(self):
    """NaN != NaN, but the object should be preserved."""
    result = Chain(float('nan')).then(lambda v: math.isnan(v)).run()
    self.assertTrue(result)

  def test_complex_number_as_root(self):
    result = Chain(1+2j).run()
    self.assertEqual(result, 1+2j)

  def test_complex_zero_as_root(self):
    result = Chain(0j).run()
    self.assertEqual(result, 0j)

  def test_complex_propagates_through_chain(self):
    result = Chain(3+4j).then(lambda v: abs(v)).run()
    self.assertEqual(result, 5.0)

  def test_very_large_int_as_root(self):
    big = 10**100
    result = Chain(big).run()
    self.assertEqual(result, big)

  def test_very_large_int_propagates(self):
    big = 10**100
    result = Chain(big).then(lambda v: v + 1).run()
    self.assertEqual(result, big + 1)

  def test_very_small_float_as_root(self):
    tiny = 1e-300
    result = Chain(tiny).run()
    self.assertEqual(result, tiny)

  def test_very_small_float_propagates(self):
    tiny = 1e-300
    result = Chain(tiny).then(lambda v: v * 2).run()
    self.assertEqual(result, tiny * 2)

  def test_negative_zero_as_root(self):
    result = Chain(-0.0).run()
    self.assertEqual(result, -0.0)
    # Verify it's truly -0.0 via copysign
    self.assertEqual(math.copysign(1, result), -1.0)

  def test_negative_zero_propagates(self):
    result = Chain(-0.0).then(lambda v: v).run()
    self.assertEqual(math.copysign(1, result), -1.0)

  def test_inf_arithmetic_in_chain(self):
    result = Chain(float('inf')).then(lambda v: v + 1).run()
    self.assertEqual(result, float('inf'))

  def test_complex_in_cascade(self):
    result = Cascade(2+3j).then(lambda v: abs(v)).run()
    self.assertEqual(result, 2+3j)

  def test_sys_maxsize_as_root(self):
    result = Chain(sys.maxsize).run()
    self.assertEqual(result, sys.maxsize)

  def test_negative_sys_maxsize_as_root(self):
    result = Chain(-sys.maxsize - 1).run()
    self.assertEqual(result, -sys.maxsize - 1)


# ===================================================================
# 4. SPECIAL TYPES AS ROOT/OPERANDS
# ===================================================================

class SpecialTypesAsRootTests(TestCase):
  """Exotic Python types used as root values or link operands."""

  def test_bytes_as_root(self):
    result = Chain(b'hello').run()
    self.assertEqual(result, b'hello')

  def test_bytearray_as_root(self):
    ba = bytearray(b'hello')
    result = Chain(ba).run()
    self.assertEqual(result, bytearray(b'hello'))

  def test_memoryview_as_root(self):
    mv = memoryview(b'hello')
    result = Chain(mv).run()
    self.assertIs(result, mv)

  def test_range_as_root(self):
    """range is callable, so Chain(range) creates a chain with range as root callable."""
    result = Chain(range, 5).run()
    self.assertEqual(list(result), [0, 1, 2, 3, 4])

  def test_range_object_as_root(self):
    """A range *instance* is a non-callable literal."""
    r = range(5)
    result = Chain(r).run()
    self.assertIs(result, r)

  def test_slice_as_root(self):
    s = slice(1, 5, 2)
    result = Chain(s).run()
    self.assertIs(result, s)

  def test_ellipsis_as_root(self):
    result = Chain(...).run()
    self.assertIs(result, ...)

  def test_type_as_root_callable(self):
    """The `type` builtin is callable, so Chain(type, 42) returns type(42)."""
    result = Chain(type, 42).run()
    self.assertIs(result, int)

  def test_type_itself_as_literal(self):
    """Passing `type` with no args makes it a callable that needs current value."""
    result = Chain(42).then(type).run()
    self.assertIs(result, int)

  def test_not_implemented_as_root(self):
    result = Chain(NotImplemented).run()
    self.assertIs(result, NotImplemented)

  def test_exception_instance_as_value(self):
    """An Exception instance passed as a value — not raised."""
    exc = ValueError('test')
    result = Chain(exc).run()
    self.assertIs(result, exc)

  def test_exception_instance_propagates(self):
    exc = RuntimeError('passthrough')
    result = Chain(exc).then(lambda v: type(v).__name__).run()
    self.assertEqual(result, 'RuntimeError')

  def test_lambda_as_root_callable(self):
    result = Chain(lambda: 42).run()
    self.assertEqual(result, 42)

  def test_partial_as_root_callable(self):
    fn = functools.partial(operator.add, 10)
    result = Chain(fn, 5).run()
    self.assertEqual(result, 15)

  def test_builtin_len_as_callable(self):
    result = Chain([1, 2, 3]).then(len).run()
    self.assertEqual(result, 3)

  def test_builtin_str_as_callable(self):
    result = Chain(42).then(str).run()
    self.assertEqual(result, '42')

  def test_builtin_int_as_callable(self):
    result = Chain('123').then(int).run()
    self.assertEqual(result, 123)

  def test_builtin_sorted_as_callable(self):
    result = Chain([3, 1, 2]).then(sorted).run()
    self.assertEqual(result, [1, 2, 3])

  def test_builtin_type_as_callable(self):
    result = Chain(42).then(type).run()
    self.assertIs(result, int)

  def test_callable_class_instance_as_root(self):
    cc = CallableClass(3)
    result = Chain(cc, 5).run()
    self.assertEqual(result, 15)

  def test_callable_class_instance_in_then(self):
    cc = CallableClass(10)
    result = Chain(5).then(cc).run()
    self.assertEqual(result, 50)

  def test_bound_method_in_then(self):
    holder = BoundMethodHolder(100)
    result = Chain(5).then(holder.transform).run()
    self.assertEqual(result, 105)

  def test_bound_method_as_root(self):
    holder = BoundMethodHolder(42)
    result = Chain(holder.get, ...).run()
    self.assertEqual(result, 42)

  def test_static_method_in_then(self):
    result = Chain(5).then(BoundMethodHolder.static_double).run()
    self.assertEqual(result, 10)

  def test_classmethod_in_then(self):
    result = Chain(5).then(BoundMethodHolder.class_name).run()
    self.assertEqual(result, 'BoundMethodHolder')

  def test_property_object_as_root_value(self):
    """A property object is not callable in the normal sense — stored as literal."""
    prop = property(lambda self: 42)
    result = Chain(prop).run()
    self.assertIs(result, prop)

  def test_staticmethod_object_as_root_value(self):
    """staticmethod is callable in Python 3.10+, so Chain calls it."""
    sm = staticmethod(lambda: 99)
    result = Chain(sm, ...).run()
    # staticmethod IS callable in Python 3.10+, so it gets called
    self.assertEqual(result, 99)

  def test_chain_as_nested_value_in_then(self):
    """A chain without root used in then() — receives the current value."""
    inner = Chain().then(lambda v: v * 2)
    result = Chain(10).then(inner).run()
    self.assertEqual(result, 20)

  def test_chain_with_root_as_standalone(self):
    """A chain with a root value can be run standalone."""
    c = Chain(42)
    result = c.run()
    self.assertEqual(result, 42)

  def test_class_constructor_as_callable(self):
    result = Chain(dict, a=1, b=2).run()
    self.assertEqual(result, {'a': 1, 'b': 2})

  def test_class_constructor_in_then(self):
    result = Chain([('a', 1), ('b', 2)]).then(dict).run()
    self.assertEqual(result, {'a': 1, 'b': 2})


class GeneratorAndCoroutineTests(IsolatedAsyncioTestCase):
  """Generator objects, async generators, and coroutine objects as values."""

  def test_generator_object_as_root(self):
    def gen():
      yield 1
      yield 2
      yield 3
    g = gen()
    result = Chain(g).run()
    # generator is not callable, so it's stored as-is
    self.assertIs(result, g)

  async def test_unawaited_coroutine_as_root(self):
    """A coroutine object used as root — the chain should await it."""
    async def async_fn():
      return 42
    coro = async_fn()
    result = await Chain(coro).run()
    self.assertEqual(result, 42)

  async def test_async_generator_as_root_literal(self):
    """An async generator function called in the chain."""
    async def agen():
      yield 1
      yield 2
      yield 3
    # agen is callable; Chain(agen, ...) calls agen() which returns an async generator
    result = Chain(agen, ...).run()
    # async generator is not awaitable — it's returned directly
    items = []
    async for item in result:
      items.append(item)
    self.assertEqual(items, [1, 2, 3])


# ===================================================================
# 5. CONTAINER TYPES
# ===================================================================

class ContainerTypesTests(TestCase):
  """Lists, tuples, dicts, sets, frozensets, and specialized containers."""

  def test_list_as_root(self):
    result = Chain([1, 2, 3]).run()
    self.assertEqual(result, [1, 2, 3])

  def test_tuple_as_root(self):
    result = Chain((1, 2, 3)).run()
    self.assertEqual(result, (1, 2, 3))

  def test_dict_as_root(self):
    result = Chain({'a': 1}).run()
    self.assertEqual(result, {'a': 1})

  def test_set_as_root(self):
    result = Chain({1, 2, 3}).run()
    self.assertEqual(result, {1, 2, 3})

  def test_frozenset_as_root(self):
    result = Chain(frozenset({1, 2, 3})).run()
    self.assertEqual(result, frozenset({1, 2, 3}))

  def test_nested_list_as_root(self):
    nested = [[1, 2], [3, 4]]
    result = Chain(nested).run()
    self.assertEqual(result, [[1, 2], [3, 4]])

  def test_nested_dict_as_root(self):
    nested = {'a': {'b': 1}}
    result = Chain(nested).run()
    self.assertEqual(result, {'a': {'b': 1}})

  def test_ordered_dict_as_root(self):
    od = OrderedDict([('a', 1), ('b', 2)])
    result = Chain(od).run()
    self.assertEqual(result, od)
    self.assertIsInstance(result, OrderedDict)

  def test_defaultdict_as_root(self):
    dd = defaultdict(int, a=1, b=2)
    result = Chain(dd).run()
    self.assertEqual(result['a'], 1)
    self.assertIsInstance(result, defaultdict)

  def test_counter_as_root(self):
    c = Counter('abracadabra')
    result = Chain(c).run()
    self.assertEqual(result, c)
    self.assertIsInstance(result, Counter)

  def test_named_tuple_as_root(self):
    pt = Point(1, 2)
    result = Chain(pt).run()
    self.assertEqual(result, pt)
    self.assertEqual(result.x, 1)
    self.assertEqual(result.y, 2)

  def test_dataclass_as_root(self):
    dp = DataPoint(1.0, 2.0, 'test')
    result = Chain(dp).run()
    self.assertIs(result, dp)
    self.assertEqual(result.label, 'test')

  def test_list_propagates_through_chain(self):
    result = Chain([1, 2, 3]).then(lambda v: v + [4]).run()
    self.assertEqual(result, [1, 2, 3, 4])

  def test_dict_propagates_through_chain(self):
    result = Chain({'a': 1}).then(lambda v: {**v, 'b': 2}).run()
    self.assertEqual(result, {'a': 1, 'b': 2})

  def test_tuple_propagates_through_chain(self):
    result = Chain((1, 2)).then(lambda v: v + (3,)).run()
    self.assertEqual(result, (1, 2, 3))

  def test_named_tuple_constructor_as_callable(self):
    result = Chain(Point, 3, 4).run()
    self.assertEqual(result, Point(3, 4))

  def test_dataclass_constructor_as_callable(self):
    result = Chain(DataPoint, 1.0, 2.0, label='origin').run()
    self.assertEqual(result.label, 'origin')

  def test_nested_containers_deep(self):
    deep = {'a': [1, {'b': (2, 3, frozenset({4, 5}))}]}
    result = Chain(deep).run()
    self.assertEqual(result, deep)


# ===================================================================
# 6. CALLABLE EDGE CASES
# ===================================================================

class CallableEdgeCasesTests(TestCase):
  """functools.partial, operator module, constructors, bound methods, etc."""

  def test_partial_with_positional_args(self):
    fn = functools.partial(operator.add, 10)
    result = Chain(5).then(fn).run()
    self.assertEqual(result, 15)

  def test_partial_with_kwargs(self):
    def greet(name, greeting='Hello'):
      return f'{greeting}, {name}!'
    fn = functools.partial(greet, greeting='Hi')
    result = Chain('World').then(fn).run()
    self.assertEqual(result, 'Hi, World!')

  def test_partial_as_root(self):
    fn = functools.partial(operator.mul, 7)
    result = Chain(fn, 6).run()
    self.assertEqual(result, 42)

  def test_operator_add(self):
    result = Chain(10).then(lambda v: operator.add(v, 5)).run()
    self.assertEqual(result, 15)

  def test_operator_mul(self):
    result = Chain(6).then(lambda v: operator.mul(v, 7)).run()
    self.assertEqual(result, 42)

  def test_operator_itemgetter(self):
    getter = operator.itemgetter('key')
    result = Chain({'key': 'value'}).then(getter).run()
    self.assertEqual(result, 'value')

  def test_operator_attrgetter(self):
    pt = Point(10, 20)
    getter = operator.attrgetter('x')
    result = Chain(pt).then(getter).run()
    self.assertEqual(result, 10)

  def test_class_constructor_dict(self):
    result = Chain(dict).run()
    self.assertEqual(result, {})

  def test_class_constructor_list(self):
    result = Chain(list, 'abc').run()
    self.assertEqual(result, ['a', 'b', 'c'])

  def test_class_constructor_set(self):
    result = Chain(set, [1, 2, 2, 3]).run()
    self.assertEqual(result, {1, 2, 3})

  def test_callable_instance_as_root(self):
    cc = CallableClass(5)
    result = Chain(cc, ...).run()
    self.assertEqual(result, 5)

  def test_callable_instance_in_chain(self):
    cc = CallableClass(3)
    result = Chain(4).then(cc).run()
    self.assertEqual(result, 12)

  def test_bound_method_as_callable(self):
    h = BoundMethodHolder(10)
    result = Chain(5).then(h.transform).run()
    self.assertEqual(result, 15)

  def test_staticmethod_callable_from_class(self):
    result = Chain(7).then(BoundMethodHolder.static_double).run()
    self.assertEqual(result, 14)

  def test_classmethod_callable(self):
    result = Chain(None).then(BoundMethodHolder.class_name).run()
    self.assertEqual(result, 'BoundMethodHolder')

  def test_builtin_abs_as_callable(self):
    result = Chain(-42).then(abs).run()
    self.assertEqual(result, 42)

  def test_builtin_bool_as_callable(self):
    result = Chain(0).then(bool).run()
    self.assertIs(result, False)

  def test_builtin_list_as_callable(self):
    result = Chain('abc').then(list).run()
    self.assertEqual(result, ['a', 'b', 'c'])

  def test_builtin_tuple_as_callable(self):
    result = Chain([1, 2, 3]).then(tuple).run()
    self.assertEqual(result, (1, 2, 3))

  def test_builtin_set_as_callable(self):
    result = Chain([1, 2, 2, 3]).then(set).run()
    self.assertEqual(result, {1, 2, 3})

  def test_builtin_dict_with_list_of_pairs(self):
    result = Chain([('a', 1), ('b', 2)]).then(dict).run()
    self.assertEqual(result, {'a': 1, 'b': 2})

  def test_builtin_repr_as_callable(self):
    result = Chain(42).then(repr).run()
    self.assertEqual(result, '42')

  def test_builtin_id_as_callable(self):
    obj = object()
    result = Chain(obj).then(id).run()
    self.assertEqual(result, id(obj))

  def test_builtin_hex_as_callable(self):
    result = Chain(255).then(hex).run()
    self.assertEqual(result, '0xff')

  def test_functools_reduce_in_chain(self):
    result = Chain([1, 2, 3, 4]).then(lambda v: functools.reduce(operator.add, v)).run()
    self.assertEqual(result, 10)

  def test_map_in_chain(self):
    result = Chain([1, 2, 3]).then(lambda v: list(map(lambda x: x * 2, v))).run()
    self.assertEqual(result, [2, 4, 6])

  def test_filter_builtin_in_chain(self):
    result = Chain([1, 2, 3, 4, 5]).then(lambda v: list(filter(lambda x: x > 3, v))).run()
    self.assertEqual(result, [4, 5])


# ===================================================================
# 7. STRING AND BYTES OPERATIONS
# ===================================================================

class StringAndBytesTests(TestCase):
  """Unicode, long strings, empty vs None, bytes through chains."""

  def test_unicode_emoji_as_root(self):
    result = Chain('hello \U0001f30d').run()
    self.assertEqual(result, 'hello \U0001f30d')

  def test_unicode_cjk_as_root(self):
    result = Chain('\u3053\u3093\u306b\u3061\u306f').run()
    self.assertEqual(result, '\u3053\u3093\u306b\u3061\u306f')

  def test_unicode_propagates_through_chain(self):
    result = Chain('\u3053\u3093\u306b\u3061\u306f').then(len).run()
    self.assertEqual(result, 5)

  def test_very_long_string_as_root(self):
    long_str = 'x' * 10000
    result = Chain(long_str).run()
    self.assertEqual(result, long_str)
    self.assertEqual(len(result), 10000)

  def test_very_long_string_propagates(self):
    long_str = 'x' * 10000
    result = Chain(long_str).then(len).run()
    self.assertEqual(result, 10000)

  def test_empty_vs_none_string(self):
    """Empty string and None are distinct."""
    result_empty = Chain('').then(lambda v: v is None).run()
    result_none = Chain(None).then(lambda v: v is None).run()
    self.assertFalse(result_empty)
    self.assertTrue(result_none)

  def test_bytes_through_chain(self):
    result = Chain(b'hello').then(lambda v: v.upper()).run()
    self.assertEqual(result, b'HELLO')

  def test_bytes_to_string_in_chain(self):
    result = Chain(b'hello').then(lambda v: v.decode('utf-8')).run()
    self.assertEqual(result, 'hello')

  def test_string_to_bytes_in_chain(self):
    result = Chain('hello').then(lambda v: v.encode('utf-8')).run()
    self.assertEqual(result, b'hello')

  def test_bytearray_through_chain(self):
    ba = bytearray(b'hello')
    result = Chain(ba).then(lambda v: v.upper()).run()
    self.assertEqual(result, bytearray(b'HELLO'))

  def test_multiline_string_as_root(self):
    s = 'line1\nline2\nline3'
    result = Chain(s).then(lambda v: v.split('\n')).run()
    self.assertEqual(result, ['line1', 'line2', 'line3'])

  def test_string_with_null_bytes(self):
    s = 'hello\x00world'
    result = Chain(s).run()
    self.assertEqual(result, 'hello\x00world')
    self.assertEqual(len(result), 11)


# ===================================================================
# 8. THE NULL SENTINEL
# ===================================================================

class NullSentinelTests(TestCase):
  """Tests around the Null sentinel object from quent."""

  def test_null_repr(self):
    result = repr(Null)
    self.assertEqual(result, '<Null>')

  def test_null_identity(self):
    self.assertIs(Null, Null)

  def test_null_is_not_none(self):
    self.assertIsNot(Null, None)

  def test_chain_with_null_as_void(self):
    """Passing Null as root creates a void chain — run() returns None."""
    result = Chain(Null).run()
    self.assertIsNone(result)

  def test_null_passed_as_run_override(self):
    """run(Null) means no override — void chain returns None."""
    result = Chain().run(Null)
    self.assertIsNone(result)

  def test_null_is_singleton(self):
    """The Null sentinel is a specific singleton object."""
    from quent.quent import PyNull
    self.assertIs(Null, PyNull)


# ===================================================================
# 9. BOOL PROTOCOL
# ===================================================================

class BoolProtocolTests(TestCase):
  """Chains are always truthy, regardless of content."""

  def test_bool_empty_chain_is_true(self):
    self.assertTrue(bool(Chain()))

  def test_bool_chain_with_false_root_is_true(self):
    self.assertTrue(bool(Chain(False)))

  def test_bool_chain_with_none_root_is_true(self):
    self.assertTrue(bool(Chain(None)))

  def test_bool_chain_with_zero_root_is_true(self):
    self.assertTrue(bool(Chain(0)))

  def test_bool_cascade_is_true(self):
    self.assertTrue(bool(Cascade()))

  def test_bool_cascade_with_false_is_true(self):
    self.assertTrue(bool(Cascade(False)))

  def test_chain_in_if_always_enters(self):
    entered = False
    if Chain():
      entered = True
    self.assertTrue(entered)

  def test_chain_in_if_with_none_always_enters(self):
    entered = False
    if Chain(None):
      entered = True
    self.assertTrue(entered)

  def test_chain_and_short_circuit(self):
    """Chain() is truthy, so `Chain() and X` returns X."""
    result = Chain() and 42
    self.assertEqual(result, 42)

  def test_chain_or_short_circuit(self):
    """Chain() is truthy, so `Chain() or X` returns Chain()."""
    c = Chain()
    result = c or 42
    self.assertIs(result, c)

  def test_not_chain_is_false(self):
    self.assertFalse(not Chain())

  def test_not_chain_with_falsy_root_is_false(self):
    self.assertFalse(not Chain(False))


# ===================================================================
# 10. EDGE VALUE COMBINATIONS
# ===================================================================

class EdgeValueCombinationsTests(TestCase):
  """Complex combinations of values through chains."""

  def test_none_type_name(self):
    result = Chain(None).then(lambda v: type(v).__name__).run()
    self.assertEqual(result, 'NoneType')

  def test_cascade_none_returns_none(self):
    """Cascade always returns root, so even .then(lambda v: v is None) is discarded."""
    result = Cascade(None).then(lambda v: v is None).run()
    self.assertIsNone(result)

  def test_chain_empty_list_foreach(self):
    result = Chain([]).foreach(lambda x: x).run()
    self.assertEqual(result, [])

  def test_chain_empty_dict_len(self):
    result = Chain({}).then(lambda v: len(v)).run()
    self.assertEqual(result, 0)

  def test_false_to_int_conversion(self):
    result = Chain(False).then(lambda v: int(v)).run()
    self.assertEqual(result, 0)

  def test_true_to_int_conversion(self):
    result = Chain(True).then(lambda v: int(v)).run()
    self.assertEqual(result, 1)

  def test_none_to_str_conversion(self):
    result = Chain(None).then(lambda v: str(v)).run()
    self.assertEqual(result, 'None')

  def test_chain_of_type_checks(self):
    result = (
      Chain(42)
      .then(lambda v: isinstance(v, int))
      .then(lambda v: v is True)
      .run()
    )
    self.assertTrue(result)

  def test_chain_with_list_mutation(self):
    """Mutation inside a chain link should affect the original list."""
    lst = [1, 2, 3]
    Chain(lst).then(lambda v: v.append(4)).run()
    self.assertEqual(lst, [1, 2, 3, 4])

  def test_chain_with_dict_mutation(self):
    d = {'a': 1}
    Chain(d).then(lambda v: v.update({'b': 2})).run()
    self.assertEqual(d, {'a': 1, 'b': 2})

  def test_multiple_type_transitions(self):
    """Chain through multiple type changes: int -> str -> list -> len -> bool."""
    result = (
      Chain(42)
      .then(str)
      .then(list)
      .then(len)
      .then(bool)
      .run()
    )
    self.assertTrue(result)

  def test_chain_preserves_object_identity(self):
    """Same object in, same object out."""
    sentinel = object()
    result = Chain(sentinel).run()
    self.assertIs(result, sentinel)

  def test_chain_preserves_identity_through_then(self):
    sentinel = object()
    result = Chain(sentinel).then(lambda v: v).run()
    self.assertIs(result, sentinel)

  def test_cascade_preserves_root_identity(self):
    sentinel = object()
    result = Cascade(sentinel).then(lambda v: object()).run()
    self.assertIs(result, sentinel)


# ===================================================================
# 11. PIPE OPERATOR WITH EDGE VALUES
# ===================================================================

class PipeOperatorEdgeTests(TestCase):
  """Pipe (|) operator with edge-case values."""

  def test_pipe_with_falsy_root(self):
    result = Chain(0) | (lambda v: v + 1) | run()
    self.assertEqual(result, 1)

  def test_pipe_with_none_root(self):
    result = Chain(None) | (lambda v: v is None) | run()
    self.assertTrue(result)

  def test_pipe_with_false_root(self):
    result = Chain(False) | (lambda v: not v) | run()
    self.assertTrue(result)

  def test_pipe_with_empty_string(self):
    result = Chain('') | (lambda v: len(v)) | run()
    self.assertEqual(result, 0)

  def test_pipe_chain_then_run_override(self):
    result = Chain() | (lambda v: v * 2) | run(5)
    self.assertEqual(result, 10)

  def test_pipe_with_builtin(self):
    result = Chain(42) | str | run()
    self.assertEqual(result, '42')

  def test_pipe_multiple_steps(self):
    result = Chain(-5) | abs | str | len | run()
    self.assertEqual(result, 1)


# ===================================================================
# 12. CLONE AND FREEZE WITH EDGE VALUES
# ===================================================================

class CloneFreezeEdgeTests(TestCase):
  """clone() and freeze() with edge-case values."""

  def test_clone_with_falsy_root(self):
    c = Chain(0).then(lambda v: v + 1)
    c2 = c.clone()
    self.assertEqual(c.run(), 1)
    self.assertEqual(c2.run(), 1)

  def test_clone_with_none_root(self):
    c = Chain(None).then(lambda v: v is None)
    c2 = c.clone()
    self.assertTrue(c.run())
    self.assertTrue(c2.run())

  def test_clone_with_false_root(self):
    c = Chain(False).then(lambda v: not v)
    c2 = c.clone()
    self.assertTrue(c.run())
    self.assertTrue(c2.run())

  def test_freeze_with_falsy_root(self):
    frozen = Chain(0).then(lambda v: v + 1).freeze()
    self.assertEqual(frozen.run(), 1)
    self.assertEqual(frozen.run(), 1)  # can run multiple times

  def test_freeze_with_none_root(self):
    frozen = Chain(None).then(lambda v: 'was_none' if v is None else 'not_none').freeze()
    self.assertEqual(frozen.run(), 'was_none')

  def test_clone_cascade_with_falsy_root(self):
    c = Cascade(0).then(lambda v: v + 999)
    c2 = c.clone()
    self.assertEqual(c.run(), 0)
    self.assertEqual(c2.run(), 0)


# ===================================================================
# 13. EXCEPT AND FINALLY WITH EDGE VALUES
# ===================================================================

class ExceptFinallyEdgeTests(TestCase):
  """except_ and finally_ handlers with edge-case values."""

  def test_except_with_none_root(self):
    def fail(v):
      raise ValueError('test')
    result = Chain(None).then(fail).except_(lambda: 'caught', ..., reraise=False).run()
    self.assertEqual(result, 'caught')

  def test_except_with_false_root(self):
    def fail(v):
      raise ValueError('test')
    result = Chain(False).then(fail).except_(lambda: 'caught', ..., reraise=False).run()
    self.assertEqual(result, 'caught')

  def test_finally_with_none_root(self):
    called = {'value': False}
    def cleanup(v=None):
      called['value'] = True
    Chain(None).finally_(cleanup).run()
    self.assertTrue(called['value'])

  def test_finally_with_false_root(self):
    called = {'value': False}
    def cleanup(v=None):
      called['value'] = True
    result = Chain(False).finally_(cleanup).run()
    self.assertIs(result, False)
    self.assertTrue(called['value'])

  def test_finally_receives_root_value(self):
    received = {'value': 'sentinel'}
    def cleanup(v):
      received['value'] = v
    Chain(42).finally_(cleanup).run()
    self.assertEqual(received['value'], 42)

  def test_except_handler_receives_root_with_falsy_root(self):
    received = {'value': 'sentinel'}
    def handler(v):
      received['value'] = v
    Chain(0).then(lambda v: 1/0).except_(handler, reraise=False).run()
    self.assertEqual(received['value'], 0)


# ===================================================================
# 14. DO WITH EDGE VALUES
# ===================================================================

class DoWithEdgeValuesTests(TestCase):
  """do() discards its result — verify that falsy current values survive."""

  def test_do_preserves_none(self):
    result = Chain(None).do(lambda v: 999).run()
    self.assertIsNone(result)

  def test_do_preserves_false(self):
    result = Chain(False).do(lambda v: 999).run()
    self.assertIs(result, False)

  def test_do_preserves_zero(self):
    result = Chain(0).do(lambda v: 999).run()
    self.assertEqual(result, 0)

  def test_do_preserves_empty_string(self):
    result = Chain('').do(lambda v: 999).run()
    self.assertEqual(result, '')

  def test_do_preserves_empty_list(self):
    result = Chain([]).do(lambda v: 999).run()
    self.assertEqual(result, [])

  def test_do_after_then_returning_none(self):
    result = Chain(42).then(lambda v: None).do(lambda v: 'side_effect').run()
    self.assertIsNone(result)

  def test_do_after_then_returning_false(self):
    result = Chain(42).then(lambda v: False).do(lambda v: 'side_effect').run()
    self.assertIs(result, False)


# ===================================================================
# 15. FOREACH AND FILTER WITH EDGE VALUES
# ===================================================================

class ForeachFilterEdgeTests(TestCase):
  """foreach and filter with edge-case iterables and values."""

  def test_foreach_empty_list(self):
    result = Chain([]).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [])

  def test_foreach_list_of_nones(self):
    result = Chain([None, None, None]).foreach(lambda x: x is None).run()
    self.assertEqual(result, [True, True, True])

  def test_foreach_list_of_falsy(self):
    result = Chain([0, '', False, None]).foreach(lambda x: type(x).__name__).run()
    self.assertEqual(result, ['int', 'str', 'bool', 'NoneType'])

  def test_foreach_preserves_types(self):
    result = Chain([1, 'two', 3.0, True]).foreach(lambda x: x).run()
    self.assertEqual(result, [1, 'two', 3.0, True])

  def test_filter_all_falsy(self):
    result = Chain([0, '', False, None]).filter(lambda x: x).run()
    self.assertEqual(result, [])

  def test_filter_all_truthy(self):
    result = Chain([1, 'a', True, [1]]).filter(lambda x: x).run()
    self.assertEqual(result, [1, 'a', True, [1]])

  def test_filter_mixed(self):
    result = Chain([0, 1, 2, 3]).filter(lambda x: x > 1).run()
    self.assertEqual(result, [2, 3])

  def test_foreach_on_string(self):
    """Strings are iterable — foreach iterates characters."""
    result = Chain('abc').foreach(lambda c: c.upper()).run()
    self.assertEqual(result, ['A', 'B', 'C'])

  def test_foreach_on_dict_iterates_keys(self):
    result = Chain({'a': 1, 'b': 2}).foreach(lambda k: k.upper()).run()
    self.assertEqual(sorted(result), ['A', 'B'])

  def test_foreach_on_range(self):
    result = Chain(range(5)).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  def test_foreach_on_set(self):
    result = Chain({1, 2, 3}).foreach(lambda x: x * 2).run()
    self.assertEqual(sorted(result), [2, 4, 6])


# ===================================================================
# 16. GATHER WITH EDGE VALUES
# ===================================================================

class GatherEdgeTests(TestCase):
  """gather() with edge-case values and functions."""

  def test_gather_with_identity_functions(self):
    result = Chain(5).gather(
      lambda v: v,
      lambda v: v * 2,
      lambda v: v * 3,
    ).run()
    self.assertEqual(result, [5, 10, 15])

  def test_gather_with_type_functions(self):
    result = Chain(42).gather(
      str,
      float,
      bool,
    ).run()
    self.assertEqual(result, ['42', 42.0, True])

  def test_gather_with_falsy_root(self):
    result = Chain(0).gather(
      lambda v: v,
      lambda v: v + 1,
    ).run()
    self.assertEqual(result, [0, 1])

  def test_gather_with_none_root(self):
    result = Chain(None).gather(
      lambda v: v is None,
      lambda v: type(v).__name__,
    ).run()
    self.assertEqual(result, [True, 'NoneType'])


# ===================================================================
# 17. WITH_ (CONTEXT MANAGER) WITH EDGE VALUES
# ===================================================================

class WithContextManagerEdgeTests(TestCase):
  """with_() using context managers with edge-case values."""

  def test_with_returns_context_value(self):
    cm = CustomContextManager(42)
    result = Chain(cm).with_(lambda v: v * 2).run()
    self.assertEqual(result, 84)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_with_none_context_value(self):
    cm = CustomContextManager(None)
    result = Chain(cm).with_(lambda v: v is None).run()
    self.assertTrue(result)

  def test_with_false_context_value(self):
    cm = CustomContextManager(False)
    result = Chain(cm).with_(lambda v: not v).run()
    self.assertTrue(result)

  def test_with_preserves_entry_exit(self):
    cm = CustomContextManager('value')
    Chain(cm).with_(lambda v: v).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ===================================================================
# 18. ASYNC EDGE VALUES
# ===================================================================

class AsyncEdgeValuesTests(IsolatedAsyncioTestCase):
  """Async handling with edge-case values."""

  async def test_async_none_root(self):
    result = await Chain(aempty, None).then(lambda v: v is None).run()
    self.assertTrue(result)

  async def test_async_false_root(self):
    result = await Chain(aempty, False).then(lambda v: v).run()
    self.assertIs(result, False)

  async def test_async_zero_root(self):
    result = await Chain(aempty, 0).then(lambda v: v + 1).run()
    self.assertEqual(result, 1)

  async def test_async_empty_string_root(self):
    result = await Chain(aempty, '').then(lambda v: len(v)).run()
    self.assertEqual(result, 0)

  async def test_async_cascade_with_none_root(self):
    received = []
    async def capture(v):
      received.append(v)
    result = await Cascade(aempty, None).then(capture).then(capture).run()
    self.assertIsNone(result)
    self.assertEqual(received, [None, None])

  async def test_async_cascade_with_false_root(self):
    received = []
    async def capture(v):
      received.append(v)
    result = await Cascade(aempty, False).then(capture).then(capture).run()
    self.assertIs(result, False)
    self.assertEqual(received, [False, False])

  async def test_async_then_returning_none(self):
    result = await Chain(aempty, 42).then(lambda v: None).then(lambda v: v is None).run()
    self.assertTrue(result)

  async def test_async_then_returning_false(self):
    result = await Chain(aempty, 42).then(lambda v: False).then(lambda v: v).run()
    self.assertIs(result, False)

  async def test_async_do_preserves_falsy(self):
    result = await Chain(aempty, 0).do(lambda v: 999).run()
    self.assertEqual(result, 0)

  async def test_async_foreach_empty_list(self):
    result = await Chain(aempty, []).foreach(lambda x: x).run()
    self.assertEqual(result, [])

  async def test_async_chain_with_large_int(self):
    big = 10**100
    result = await Chain(aempty, big).then(lambda v: v + 1).run()
    self.assertEqual(result, big + 1)

  async def test_async_chain_with_complex(self):
    result = await Chain(aempty, 3+4j).then(lambda v: abs(v)).run()
    self.assertEqual(result, 5.0)


# ===================================================================
# 19. CONFIG AND NO_ASYNC WITH EDGE VALUES
# ===================================================================

class ConfigNoAsyncEdgeTests(TestCase):
  """config() and no_async() with edge-case values."""

  def test_no_async_with_falsy_root(self):
    result = Chain(0).then(lambda v: v + 1).no_async(True).run()
    self.assertEqual(result, 1)

  def test_no_async_with_none_root(self):
    result = Chain(None).then(lambda v: v is None).no_async(True).run()
    self.assertTrue(result)

  def test_config_debug_with_edge_values(self):
    result = Chain(False).then(lambda v: not v).config(debug=True).run()
    self.assertTrue(result)

  def test_config_returns_self(self):
    c = Chain(1)
    result = c.config(autorun=False)
    self.assertIs(result, c)

  def test_no_async_returns_self(self):
    c = Chain(1)
    result = c.no_async(True)
    self.assertIs(result, c)


# ===================================================================
# 20. RETURN AND BREAK WITH EDGE VALUES
# ===================================================================

class ReturnBreakEdgeTests(TestCase):
  """Chain.return_() and Chain.break_() with edge-case values."""

  def test_return_with_none(self):
    result = Chain(Chain().then(Chain.return_)).run()
    self.assertIsNone(result)

  def test_return_with_false(self):
    result = Chain(Chain().then(Chain.return_, False)).then(object()).run()
    self.assertIs(result, False)

  def test_return_with_zero(self):
    result = Chain(Chain().then(Chain.return_, 0)).then(object()).run()
    self.assertEqual(result, 0)

  def test_return_with_empty_string(self):
    result = Chain(Chain().then(Chain.return_, '')).then(object()).run()
    self.assertEqual(result, '')

  def test_break_with_none_in_foreach(self):
    def f(i):
      if i == 2:
        return Chain.break_(None)
      return i
    result = Chain([0, 1, 2, 3]).foreach(f).run()
    self.assertIsNone(result)

  def test_break_with_false_in_foreach(self):
    def f(i):
      if i == 2:
        return Chain.break_(False)
      return i
    result = Chain([0, 1, 2, 3]).foreach(f).run()
    self.assertIs(result, False)

  def test_break_with_zero_in_foreach(self):
    def f(i):
      if i == 2:
        return Chain.break_(0)
      return i
    result = Chain([0, 1, 2, 3]).foreach(f).run()
    self.assertEqual(result, 0)

  def test_break_without_value_returns_partial(self):
    def f(i):
      if i == 2:
        return Chain.break_()
      return i * 10
    result = Chain([0, 1, 2, 3]).foreach(f).run()
    self.assertEqual(result, [0, 10])


# ===================================================================
# 21. SLEEP WITH EDGE VALUES
# ===================================================================

class SleepEdgeTests(TestCase):
  """sleep() preserves the current value."""

  def test_sleep_preserves_falsy_value(self):
    result = Chain(0).sleep(0.001).run()
    self.assertEqual(result, 0)

  def test_sleep_preserves_none(self):
    result = Chain(None).sleep(0.001).run()
    self.assertIsNone(result)

  def test_sleep_preserves_false(self):
    result = Chain(False).sleep(0.001).run()
    self.assertIs(result, False)

  def test_sleep_preserves_string(self):
    result = Chain('hello').sleep(0.001).run()
    self.assertEqual(result, 'hello')


# ===================================================================
# 22. REPR WITH EDGE VALUES
# ===================================================================

class ReprEdgeTests(TestCase):
  """Chain __repr__ with edge-case values."""

  def test_repr_empty_chain(self):
    r = repr(Chain())
    self.assertIsInstance(r, str)

  def test_repr_chain_with_root(self):
    r = repr(Chain(42))
    self.assertIsInstance(r, str)

  def test_repr_chain_with_none_root(self):
    r = repr(Chain(None))
    self.assertIsInstance(r, str)

  def test_repr_chain_with_lambda(self):
    r = repr(Chain(lambda: 42))
    self.assertIsInstance(r, str)

  def test_repr_cascade(self):
    r = repr(Cascade(42))
    self.assertIsInstance(r, str)


# ===================================================================
# 23. NESTED CHAINS WITH EDGE VALUES
# ===================================================================

class NestedChainEdgeTests(TestCase):
  """Nested chains with edge-case values."""

  def test_nested_chain_with_none(self):
    inner = Chain().then(lambda v: None)
    result = Chain(42).then(inner).run()
    self.assertIsNone(result)

  def test_nested_chain_with_false(self):
    inner = Chain().then(lambda v: False)
    result = Chain(42).then(inner).run()
    self.assertIs(result, False)

  def test_nested_chain_with_zero(self):
    inner = Chain().then(lambda v: 0)
    result = Chain(42).then(inner).run()
    self.assertEqual(result, 0)

  def test_nested_chain_preserves_root(self):
    inner = Chain().then(lambda v: v * 2)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 10)

  def test_nested_cascade_returns_outer_root_propagated(self):
    """A nested Cascade returns its root, which is the value passed from outer chain."""
    inner = Cascade().then(lambda v: v * 100)
    result = Chain(7).then(inner).run()
    self.assertEqual(result, 7)

  def test_deeply_nested_chain_with_falsy(self):
    inner2 = Chain().then(lambda v: 0)
    inner1 = Chain().then(inner2)
    result = Chain(42).then(inner1).run()
    self.assertEqual(result, 0)


# ===================================================================
# 24. CALL PROTOCOL (__call__) WITH EDGE VALUES
# ===================================================================

class CallProtocolEdgeTests(TestCase):
  """Chain() as a callable (shorthand for run)."""

  def test_call_empty_chain(self):
    result = Chain()()
    self.assertIsNone(result)

  def test_call_with_root(self):
    result = Chain(42)()
    self.assertEqual(result, 42)

  def test_call_with_override(self):
    result = Chain()(99)
    self.assertEqual(result, 99)

  def test_call_with_falsy_root(self):
    result = Chain(0)()
    self.assertEqual(result, 0)

  def test_call_with_none_root(self):
    result = Chain(None)()
    self.assertIsNone(result)

  def test_call_with_false_root(self):
    result = Chain(False)()
    self.assertIs(result, False)


# ===================================================================
# 25. FROZEN CHAIN DECORATOR WITH EDGE VALUES
# ===================================================================

class FrozenChainDecoratorEdgeTests(TestCase):
  """freeze().decorator() with edge-case values."""

  def test_decorator_with_identity(self):
    @Chain().then(lambda v: v * 2).decorator()
    def double(v):
      return v
    self.assertEqual(double(5), 10)

  def test_decorator_preserves_falsy_return(self):
    @Chain().then(lambda v: v).decorator()
    def identity(v):
      return v
    self.assertEqual(identity(0), 0)
    self.assertIs(identity(False), False)
    self.assertIsNone(identity(None))

  def test_decorator_with_string(self):
    @Chain().then(lambda v: v.upper()).decorator()
    def process(v):
      return v
    self.assertEqual(process('hello'), 'HELLO')


# ===================================================================
# 26. CUSTOM ITERABLE AND ITERATOR EDGE CASES
# ===================================================================

class CustomIterableEdgeTests(TestCase):
  """Custom iterables through foreach."""

  def test_foreach_custom_iterable(self):
    ci = CustomIter([10, 20, 30])
    result = Chain(ci).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [20, 40, 60])

  def test_foreach_custom_iterable_empty(self):
    ci = CustomIter([])
    result = Chain(ci).foreach(lambda x: x).run()
    self.assertEqual(result, [])

  def test_foreach_generator_expression(self):
    result = Chain(range(5)).foreach(lambda x: x ** 2).run()
    self.assertEqual(result, [0, 1, 4, 9, 16])


# ===================================================================
# 27. THEN WITH NON-CALLABLE VALUES
# ===================================================================

class ThenNonCallableValuesTests(TestCase):
  """then() with non-callable values (literal replacement)."""

  def test_then_with_literal_none(self):
    result = Chain(42).then(None).run()
    self.assertIsNone(result)

  def test_then_with_literal_false(self):
    result = Chain(42).then(False).run()
    self.assertIs(result, False)

  def test_then_with_literal_true(self):
    result = Chain(42).then(True).run()
    self.assertIs(result, True)

  def test_then_with_literal_zero(self):
    result = Chain(42).then(0).run()
    self.assertEqual(result, 0)

  def test_then_with_literal_string(self):
    result = Chain(42).then('hello').run()
    self.assertEqual(result, 'hello')

  def test_then_with_literal_list(self):
    result = Chain(42).then([1, 2, 3]).run()
    self.assertEqual(result, [1, 2, 3])

  def test_then_with_literal_dict(self):
    result = Chain(42).then({'a': 1}).run()
    self.assertEqual(result, {'a': 1})

  def test_then_with_literal_tuple(self):
    result = Chain(42).then((1, 2)).run()
    self.assertEqual(result, (1, 2))

  def test_then_with_literal_set(self):
    result = Chain(42).then({1, 2, 3}).run()
    self.assertEqual(result, {1, 2, 3})

  def test_then_with_literal_bytes(self):
    result = Chain(42).then(b'abc').run()
    self.assertEqual(result, b'abc')

  def test_then_with_literal_frozenset(self):
    result = Chain(42).then(frozenset({1, 2})).run()
    self.assertEqual(result, frozenset({1, 2}))

  def test_then_with_literal_float(self):
    result = Chain(42).then(3.14).run()
    self.assertEqual(result, 3.14)

  def test_then_with_literal_complex(self):
    result = Chain(42).then(1+2j).run()
    self.assertEqual(result, 1+2j)

  def test_then_with_literal_ellipsis(self):
    result = Chain(42).then(...).run()
    self.assertIs(result, ...)


# ===================================================================
# 28. CHAINING MULTIPLE FALSY VALUES
# ===================================================================

class ChainingMultipleFalsyTests(TestCase):
  """Chains where multiple consecutive operations produce falsy values."""

  def test_chain_none_to_zero_to_false(self):
    result = (
      Chain(None)
      .then(lambda v: 0 if v is None else 1)
      .then(lambda v: v == 0)
      .run()
    )
    self.assertTrue(result)

  def test_chain_false_to_zero_to_empty_string(self):
    result = (
      Chain(False)
      .then(lambda v: int(v))
      .then(lambda v: str(v) if v == 0 else 'nope')
      .run()
    )
    self.assertEqual(result, '0')

  def test_chain_empty_list_to_len_to_bool(self):
    result = (
      Chain([])
      .then(len)
      .then(bool)
      .run()
    )
    self.assertIs(result, False)

  def test_chain_all_falsy_through(self):
    """Every link produces a different falsy value."""
    result = (
      Chain(None)
      .then(lambda v: 0)
      .then(lambda v: '')
      .then(lambda v: [])
      .then(lambda v: {})
      .then(lambda v: False)
      .then(lambda v: type(v).__name__)
      .run()
    )
    self.assertEqual(result, 'bool')


# ===================================================================
# 29. SPECIAL OBJECT METHODS
# ===================================================================

class SpecialObjectMethodTests(TestCase):
  """Objects with special dunder methods."""

  def test_custom_repr_object(self):
    obj = CustomRepr()
    result = Chain(obj).then(repr).run()
    self.assertEqual(result, '<CustomRepr>')

  def test_custom_len_object(self):
    obj = CustomLen(42)
    result = Chain(obj).then(len).run()
    self.assertEqual(result, 42)

  def test_custom_bool_falsy_object_through_chain(self):
    obj = CustomBoolFalsy()
    result = Chain(obj).then(lambda v: bool(v)).run()
    self.assertFalse(result)

  def test_custom_bool_truthy_object_through_chain(self):
    obj = CustomBoolTruthy()
    result = Chain(obj).then(lambda v: bool(v)).run()
    self.assertTrue(result)


# ===================================================================
# 30. VOID CHAIN EDGE CASES
# ===================================================================

class VoidChainEdgeTests(TestCase):
  """Void chains (no root) with various operations."""

  def test_void_chain_run_returns_none(self):
    result = Chain().run()
    self.assertIsNone(result)

  def test_void_chain_with_then_returns_then_value(self):
    result = Chain().then(42).run()
    self.assertEqual(result, 42)

  def test_void_chain_then_callable_no_args(self):
    result = Chain().then(lambda: 'hello', ...).run()
    self.assertEqual(result, 'hello')

  def test_void_chain_override_with_none(self):
    result = Chain().then(lambda v: v is None).run(None)
    self.assertTrue(result)

  def test_void_chain_override_with_false(self):
    result = Chain().then(lambda v: v).run(False)
    self.assertIs(result, False)

  def test_void_chain_override_with_zero(self):
    result = Chain().then(lambda v: v + 1).run(0)
    self.assertEqual(result, 1)

  def test_void_cascade_run_returns_none(self):
    result = Cascade().run()
    self.assertIsNone(result)

  def test_void_cascade_override(self):
    result = Cascade().then(lambda v: v * 2).run(5)
    self.assertEqual(result, 5)


# ===================================================================
# 31. IDENTITY AND EQUALITY EDGE CASES
# ===================================================================

class IdentityEqualityEdgeTests(TestCase):
  """Verifying identity preservation through chains."""

  def test_same_list_object_preserved(self):
    lst = [1, 2, 3]
    result = Chain(lst).then(lambda v: v).run()
    self.assertIs(result, lst)

  def test_same_dict_object_preserved(self):
    d = {'a': 1}
    result = Chain(d).then(lambda v: v).run()
    self.assertIs(result, d)

  def test_same_custom_object_preserved(self):
    obj = CustomRepr()
    result = Chain(obj).then(lambda v: v).run()
    self.assertIs(result, obj)

  def test_singleton_none_preserved(self):
    result = Chain(None).then(lambda v: v).run()
    self.assertIs(result, None)

  def test_singleton_true_preserved(self):
    result = Chain(True).then(lambda v: v).run()
    self.assertIs(result, True)

  def test_singleton_false_preserved(self):
    result = Chain(False).then(lambda v: v).run()
    self.assertIs(result, False)

  def test_singleton_ellipsis_preserved(self):
    result = Chain(...).then(lambda v: v).run()
    self.assertIs(result, ...)

  def test_singleton_not_implemented_preserved(self):
    result = Chain(NotImplemented).then(lambda v: v).run()
    self.assertIs(result, NotImplemented)
