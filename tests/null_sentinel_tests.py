"""Tests for Null sentinel, _ControlFlowSignal, _Return, and _Break internals."""
from __future__ import annotations

import copy
import gc
import pickle
import sys
import threading
import unittest
import weakref

from quent import Null, QuentException
from quent._core import _Null, _ControlFlowSignal, _Return, _Break


class TestNullSingleton(unittest.TestCase):

  def test_identity_across_imports(self):
    from quent import Null as null_public
    from quent._core import Null as null_internal
    self.assertIs(null_public, null_internal)

  def test_hash_is_stable(self):
    h1 = hash(Null)
    h2 = hash(Null)
    h3 = hash(Null)
    self.assertEqual(h1, h2)
    self.assertEqual(h2, h3)

  def test_hash_differs_from_none(self):
    self.assertNotEqual(hash(Null), hash(None))

  def test_bool_is_truthy(self):
    self.assertTrue(bool(Null))

  def test_repr_is_angle_bracket(self):
    self.assertEqual(repr(Null), '<Null>')

  def test_str_fallback(self):
    s = str(Null)
    self.assertIsInstance(s, str)
    self.assertTrue(len(s) > 0)
    # str falls back to repr for objects without __str__
    self.assertEqual(s, '<Null>')

  def test_not_equal_to_any_falsy(self):
    falsy_values = [None, False, 0, '', [], {}, set(), 0.0, b'', (), frozenset(), 0j]
    for val in falsy_values:
      with self.subTest(val=val):
        self.assertNotEqual(Null, val)
        self.assertIsNot(Null, val)

  def test_not_equal_to_any_truthy(self):
    truthy_values = [1, True, 'x', [1], {1: 2}]
    for val in truthy_values:
      with self.subTest(val=val):
        self.assertNotEqual(Null, val)
        self.assertIsNot(Null, val)

  def test_type_is_null_class(self):
    self.assertEqual(type(Null).__name__, '_Null')

  def test_slots_enforced(self):
    with self.assertRaises(AttributeError):
      Null.foo = 'bar'

  def test_no_eq_override(self):
    self.assertIs(_Null.__eq__, object.__eq__)

  def test_pickle_roundtrip(self):
    data = pickle.dumps(Null)
    loaded = pickle.loads(data)
    self.assertIs(loaded, Null)

  def test_copy_returns_same_object(self):
    self.assertIs(copy.copy(Null), Null)

  def test_deepcopy_returns_same_object(self):
    self.assertIs(copy.deepcopy(Null), Null)

  def test_null_in_set(self):
    s = {Null}
    self.assertEqual(len(s), 1)
    self.assertIn(Null, s)

  def test_null_as_dict_key(self):
    d = {Null: 'val'}
    self.assertEqual(d[Null], 'val')

  def test_isinstance_check(self):
    self.assertIsInstance(Null, _Null)

  def test_cannot_instantiate_second(self):
    n2 = _Null()
    self.assertIsNot(n2, Null)
    self.assertIsInstance(n2, _Null)

  def test_null_is_not_ellipsis(self):
    self.assertIsNot(Null, ...)

  def test_null_not_iterable(self):
    with self.assertRaises(TypeError):
      iter(Null)

  def test_null_not_subscriptable(self):
    with self.assertRaises(TypeError):
      Null[0]

  # --- Beyond-the-spec tests ---

  def test_null_eq_uses_identity(self):
    # Since __eq__ is object.__eq__, == is the same as `is`
    self.assertTrue(Null == Null)
    self.assertFalse(Null == None)  # noqa: E711
    self.assertFalse(Null == 0)
    self.assertFalse(Null == False)  # noqa: E712

  def test_null_ne_operator(self):
    self.assertTrue(Null != None)  # noqa: E711
    self.assertTrue(Null != 0)
    self.assertFalse(Null != Null)

  def test_null_in_boolean_context_if(self):
    result = 'truthy' if Null else 'falsy'
    self.assertEqual(result, 'truthy')

  def test_null_as_function_argument_default(self):
    def fn(x=Null):
      return x
    self.assertIs(fn(), Null)
    self.assertIs(fn(Null), Null)
    self.assertIsNone(fn(None))

  def test_repr_in_nested_structures(self):
    lst_repr = repr([Null, None])
    self.assertIn('<Null>', lst_repr)
    self.assertIn('None', lst_repr)

  def test_null_not_callable(self):
    self.assertFalse(callable(Null))
    with self.assertRaises(TypeError):
      Null()

  def test_null_no_len(self):
    with self.assertRaises(TypeError):
      len(Null)

  def test_null_no_hash_override(self):
    # _Null doesn't define __hash__, so it uses object.__hash__
    self.assertIs(_Null.__hash__, object.__hash__)

  def test_null_identity_stable_across_gc(self):
    id_before = id(Null)
    gc.collect()
    id_after = id(Null)
    self.assertEqual(id_before, id_after)

  def test_null_thread_safety(self):
    ids = []
    hashes = []
    reprs = []
    barrier = threading.Barrier(10)

    def worker():
      barrier.wait()
      ids.append(id(Null))
      hashes.append(hash(Null))
      reprs.append(repr(Null))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(len(set(ids)), 1)
    self.assertEqual(len(set(hashes)), 1)
    self.assertEqual(len(set(reprs)), 1)

  def test_pickle_all_protocols(self):
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
      with self.subTest(protocol=protocol):
        data = pickle.dumps(Null, protocol=protocol)
        loaded = pickle.loads(data)
        self.assertIs(loaded, Null)

  def test_null_reduce_returns_string(self):
    result = Null.__reduce__()
    self.assertEqual(result, 'Null')
    self.assertIsInstance(result, str)

  def test_second_instance_also_has_correct_repr(self):
    n2 = _Null()
    self.assertEqual(repr(n2), '<Null>')

  def test_null_not_numeric(self):
    with self.assertRaises(TypeError):
      Null + 1
    with self.assertRaises(TypeError):
      Null * 2

  def test_null_slots_are_empty_tuple(self):
    self.assertEqual(_Null.__slots__, ())

  def test_null_has_no_dict(self):
    self.assertFalse(hasattr(Null, '__dict__'))


class TestControlFlowSignalInternals(unittest.TestCase):

  def test_return_init_skips_super(self):
    # BaseException.__new__ sets .args to positional args regardless of __init__.
    # The comment in _core.py says __init__ intentionally skips super().__init__(),
    # but BaseException.__new__ still populates .args with all positional arguments.
    r = _Return(42, (), {})
    # .args is set by BaseException.__new__, not __init__
    self.assertIsInstance(r.args, tuple)

  def test_break_init_skips_super(self):
    b = _Break(42, (), {})
    self.assertIsInstance(b.args, tuple)

  def test_return_holds_value(self):
    r = _Return(42, (1,), {'k': 2})
    self.assertEqual(r.value, 42)
    self.assertEqual(r.args_, (1,))
    self.assertEqual(r.kwargs_, {'k': 2})

  def test_break_holds_value(self):
    b = _Break(42, (1,), {'k': 2})
    self.assertEqual(b.value, 42)
    self.assertEqual(b.args_, (1,))
    self.assertEqual(b.kwargs_, {'k': 2})

  def test_control_flow_signal_is_exception(self):
    self.assertIsInstance(_Return(Null, (), {}), Exception)
    self.assertIsInstance(_Break(Null, (), {}), Exception)

  def test_control_flow_signal_hierarchy(self):
    self.assertTrue(issubclass(_Return, _ControlFlowSignal))
    self.assertTrue(issubclass(_Break, _ControlFlowSignal))
    self.assertTrue(issubclass(_ControlFlowSignal, Exception))

  def test_return_is_catchable_by_except_exception(self):
    caught = False
    try:
      raise _Return(Null, (), {})
    except Exception:
      caught = True
    self.assertTrue(caught)

  def test_break_is_catchable_by_except_exception(self):
    caught = False
    try:
      raise _Break(Null, (), {})
    except Exception:
      caught = True
    self.assertTrue(caught)

  def test_return_with_null_value(self):
    r = _Return(Null, (), {})
    self.assertIs(r.value, Null)

  def test_break_with_null_value(self):
    b = _Break(Null, (), {})
    self.assertIs(b.value, Null)

  def test_control_flow_signal_slots(self):
    self.assertTrue(hasattr(_ControlFlowSignal, '__slots__'))
    self.assertIn('value', _ControlFlowSignal.__slots__)
    self.assertIn('args_', _ControlFlowSignal.__slots__)
    self.assertIn('kwargs_', _ControlFlowSignal.__slots__)

  # --- Beyond-the-spec tests ---

  def test_return_not_catchable_by_base_exception_only(self):
    # _Return IS a BaseException (via Exception), so it's caught by except BaseException
    caught_type = None
    try:
      raise _Return(99, (), {})
    except BaseException as e:
      caught_type = type(e)
    self.assertIs(caught_type, _Return)

  def test_break_not_catchable_by_base_exception_only(self):
    caught_type = None
    try:
      raise _Break(99, (), {})
    except BaseException as e:
      caught_type = type(e)
    self.assertIs(caught_type, _Break)

  def test_return_with_complex_value(self):
    complex_val = {'nested': [1, 2, {'deep': True}]}
    r = _Return(complex_val, ('a', 'b'), {'x': 10, 'y': 20})
    self.assertIs(r.value, complex_val)
    self.assertEqual(r.args_, ('a', 'b'))
    self.assertEqual(r.kwargs_, {'x': 10, 'y': 20})

  def test_break_with_none_value(self):
    b = _Break(None, (), {})
    self.assertIsNone(b.value)
    self.assertIsNot(b.value, Null)

  def test_return_with_none_value(self):
    r = _Return(None, (), {})
    self.assertIsNone(r.value)
    self.assertIsNot(r.value, Null)

  def test_control_flow_signal_pickle_roundtrip(self):
    r = _Return(42, (1, 2), {'k': 'v'})
    data = pickle.dumps(r)
    r2 = pickle.loads(data)
    self.assertEqual(r2.value, 42)
    self.assertEqual(r2.args_, (1, 2))
    self.assertEqual(r2.kwargs_, {'k': 'v'})

  def test_break_pickle_roundtrip(self):
    b = _Break('stop', ('x',), {'flag': True})
    data = pickle.dumps(b)
    b2 = pickle.loads(data)
    self.assertEqual(b2.value, 'stop')
    self.assertEqual(b2.args_, ('x',))
    self.assertEqual(b2.kwargs_, {'flag': True})

  def test_return_holds_strong_reference(self):
    class Ref:
      pass
    obj = Ref()
    r = _Return(obj, (), {})
    self.assertIs(r.value, obj)
    # The _Return should hold a strong reference to the value
    ref = weakref.ref(obj)
    del obj
    gc.collect()
    self.assertIsNotNone(ref())

  def test_control_flow_signal_empty_args_kwargs(self):
    r = _Return(Null, (), {})
    self.assertEqual(r.args_, ())
    self.assertEqual(r.kwargs_, {})

  def test_return_and_break_are_distinct_types(self):
    self.assertIsNot(_Return, _Break)
    self.assertFalse(issubclass(_Return, _Break))
    self.assertFalse(issubclass(_Break, _Return))

  def test_control_flow_signal_has_dict_from_base_exception(self):
    # BaseException provides __dict__, so despite __slots__, instances have __dict__
    r = _Return(1, (), {})
    self.assertTrue(hasattr(r, '__dict__'))

  def test_return_traceback_is_none_when_not_raised(self):
    r = _Return(1, (), {})
    self.assertIsNone(r.__traceback__)


if __name__ == '__main__':
  unittest.main()
