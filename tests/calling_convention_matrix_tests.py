"""Tests for the full cross-product of callable types x calling conventions x chain methods.

Generates a dense matrix using self.subTest() to cover every combination of:
  - 11 callable types (regular fn, async fn, lambda, builtins, partial,
    class constructor, callable obj, async callable obj, bound method,
    nested Chain, frozen Chain)
  - 6 calling conventions (no args, positional args, kwargs only,
    args+kwargs, ellipsis, ellipsis+trailing)
  - Chain methods (then, do, run, Chain(root), except_, finally_)

Also tests current_value state interactions with calling conventions.
"""
from __future__ import annotations

import functools
import operator
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null
from helpers import Adder


# ---------------------------------------------------------------------------
# Tracker callables: record (args, kwargs) for each call
# ---------------------------------------------------------------------------

def _make_tracker():
  """Return a callable that records (args, kwargs) for each call."""
  calls = []
  def fn(*args, **kwargs):
    calls.append((args, kwargs))
    if args:
      return args[0]
    return 'no_args_result'
  fn.calls = calls
  return fn


def _make_async_tracker():
  """Return an async callable that records (args, kwargs) for each call."""
  calls = []
  async def fn(*args, **kwargs):
    calls.append((args, kwargs))
    if args:
      return args[0]
    return 'no_args_result'
  fn.calls = calls
  return fn


class _BoundMethodTracker:
  """Object with a bound method that tracks calls.

  We store .calls on the object itself rather than on the bound method
  (Python 3.14 forbids setting arbitrary attrs on bound methods).
  """
  def __init__(self):
    self.calls = []

  def method(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    if args:
      return args[0]
    return 'no_args_result'


def _make_bound_method_tracker():
  """Return (bound_method, calls_list)."""
  obj = _BoundMethodTracker()
  return obj.method, obj.calls


class _AsyncCallableObjTracker:
  """Async callable object that tracks calls."""
  def __init__(self):
    self.calls = []

  async def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    if args:
      return args[0]
    return 'no_args_result'


# ---------------------------------------------------------------------------
# Convention definitions
# ---------------------------------------------------------------------------

CONVENTIONS = [
  ('no_args', (), {}),
  ('positional', (10, 20), {}),
  ('kwargs_only', (), {'x': 10, 'y': 20}),
  ('args_kwargs', (10,), {'b': 20}),
  ('ellipsis', (...,), {}),
  ('ellipsis_trailing', (..., 'extra'), {}),
]


# ---------------------------------------------------------------------------
# TestThenCallableConventionMatrix
# ---------------------------------------------------------------------------

class TestThenCallableConventionMatrix(unittest.TestCase):
  """Cross-product: 11 callable types x 6 calling conventions via then()."""

  def _run_tracker_assertions(self, tracker_calls, result, conv_name, input_value):
    """Common assertions for tracker-based callables."""
    if conv_name == 'no_args':
      self.assertEqual(tracker_calls[-1], ((input_value,), {}))
      self.assertEqual(result, input_value)
    elif conv_name == 'positional':
      self.assertEqual(tracker_calls[-1], ((10, 20), {}))
      self.assertEqual(result, 10)
    elif conv_name == 'kwargs_only':
      self.assertEqual(tracker_calls[-1], ((), {'x': 10, 'y': 20}))
      self.assertEqual(result, 'no_args_result')
    elif conv_name == 'args_kwargs':
      self.assertEqual(tracker_calls[-1], ((10,), {'b': 20}))
      self.assertEqual(result, 10)
    elif conv_name in ('ellipsis', 'ellipsis_trailing'):
      self.assertEqual(tracker_calls[-1], ((), {}))
      self.assertEqual(result, 'no_args_result')

  def test_regular_fn(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(input_value).then(fn, *conv_args, **conv_kwargs).run()
        self._run_tracker_assertions(fn.calls, result, conv_name, input_value)

  def test_lambda(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(input_value).then(fn, *conv_args, **conv_kwargs).run()
        self._run_tracker_assertions(fn.calls, result, conv_name, input_value)

  def test_builtin_str(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          self.assertEqual(Chain(42).then(str).run(), '42')
        elif conv_name == 'positional':
          # str(10, 20) -> TypeError
          with self.assertRaises(TypeError):
            Chain(42).then(str, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          # str() -> '' (no error)
          result = Chain(42).then(str, *conv_args).run()
          self.assertEqual(result, '')
        else:
          # kwargs_only, args_kwargs -> TypeError
          with self.assertRaises(TypeError):
            Chain(42).then(str, *conv_args, **conv_kwargs).run()

  def test_builtin_int(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          self.assertEqual(Chain('42').then(int).run(), 42)
        elif conv_name == 'positional':
          # int(10, 20) -> TypeError
          with self.assertRaises(TypeError):
            Chain('42').then(int, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(Chain(42).then(int, *conv_args).run(), 0)
        else:
          with self.assertRaises(TypeError):
            Chain(42).then(int, *conv_args, **conv_kwargs).run()

  def test_builtin_abs(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          self.assertEqual(Chain(-5).then(abs).run(), 5)
        else:
          with self.assertRaises(TypeError):
            Chain(-5).then(abs, *conv_args, **conv_kwargs).run()

  def test_partial(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = functools.partial(operator.add, 10)
        if conv_name == 'no_args':
          self.assertEqual(Chain(5).then(fn).run(), 15)
        elif conv_name == 'positional':
          # partial(add, 10)(10, 20) -> TypeError (add takes 2, already has 1 + 2 more = 3)
          with self.assertRaises(TypeError):
            Chain(5).then(fn, 10, 20).run()
        else:
          with self.assertRaises(TypeError):
            Chain(5).then(fn, *conv_args, **conv_kwargs).run()

  def test_class_constructor(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          result = Chain(5).then(Adder).run()
          self.assertIsInstance(result, Adder)
          self.assertEqual(result.x, 5)
        elif conv_name == 'positional':
          # Adder(10, 20) -> TypeError (only takes 1 arg x)
          with self.assertRaises(TypeError):
            Chain(5).then(Adder, 10, 20).run()
        else:
          with self.assertRaises(TypeError):
            Chain(5).then(Adder, *conv_args, **conv_kwargs).run()

  def test_callable_obj(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(input_value).then(fn, *conv_args, **conv_kwargs).run()
        self._run_tracker_assertions(fn.calls, result, conv_name, input_value)

  def test_bound_method(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        method, calls = _make_bound_method_tracker()
        result = Chain(input_value).then(method, *conv_args, **conv_kwargs).run()
        self._run_tracker_assertions(calls, result, conv_name, input_value)

  def test_nested_chain(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          inner = Chain(lambda: 'no_args_result')
          result = Chain(input_value).then(inner, *conv_args).run()
          self.assertEqual(result, 'no_args_result')
        elif conv_name == 'no_args':
          inner = Chain().then(lambda x: x)
          result = Chain(input_value).then(inner).run()
          self.assertEqual(result, input_value)
        elif conv_name == 'positional':
          # _evaluate_value for chain: v._run(args[0]=10, args[1:]=(20,), kwargs={})
          # -> Link(10, (20,), {}) created as root
          # -> _evaluate_value: args truthy, 10 is not callable -> 10(20) -> TypeError
          inner = Chain().then(lambda x: x)
          with self.assertRaises(TypeError):
            Chain(input_value).then(inner, 10, 20).run()
        elif conv_name == 'kwargs_only':
          # chain: no args but kwargs -> v._run(current_value=42, None, {x:10, y:20})
          # -> Link(42, None, {x:10, y:20}) as root
          # -> _evaluate_value: kwargs truthy -> 42(x=10, y=20) -> TypeError
          inner = Chain().then(lambda x: x)
          with self.assertRaises(TypeError):
            Chain(input_value).then(inner, **conv_kwargs).run()
        elif conv_name == 'args_kwargs':
          # chain: v._run(10, (), {b:20}) -> Link(10, (), {b:20})
          # -> _evaluate_value: kwargs truthy -> 10(b=20) -> TypeError
          inner = Chain().then(lambda x: x)
          with self.assertRaises(TypeError):
            Chain(input_value).then(inner, 10, b=20).run()

  def test_frozen_chain(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          # Frozen chain has no _is_chain, treated as regular callable
          # fn() -> frozen() -> frozen.run(Null) -> chain executes with no run_value
          frozen = Chain(lambda: 'no_args_result').freeze()
          result = Chain(input_value).then(frozen, *conv_args).run()
          self.assertEqual(result, 'no_args_result')
        elif conv_name == 'no_args':
          # Frozen treated as callable: fn(current_value) -> frozen(42)
          # -> frozen.run(42) -> chain._run(42, ...) -> 42
          frozen = Chain().then(lambda x: x).freeze()
          result = Chain(input_value).then(frozen).run()
          self.assertEqual(result, input_value)
        elif conv_name == 'positional':
          # fn(10, 20) -> frozen(10, 20) -> frozen.run(10, 20)
          # -> chain._run(10, (20,), {}) -> Link(10, (20,), {})
          # -> _evaluate_value: args truthy, 10 not callable -> 10(20) -> TypeError
          frozen = Chain().then(lambda x: x).freeze()
          with self.assertRaises(TypeError):
            Chain(input_value).then(frozen, 10, 20).run()
        elif conv_name == 'kwargs_only':
          # fn(x=10, y=20) -> frozen(x=10, y=20) -> frozen.run(x=10, y=20)
          # run signature: run(v=Null, /, *args, **kwargs)
          # -> chain._run(Null, (), {x:10, y:20})
          # has_run_value=False, no root -> first_link gets Null
          # callable with Null -> lambda() -> TypeError (needs x)
          frozen = Chain().then(lambda x: x).freeze()
          with self.assertRaises(TypeError):
            Chain(input_value).then(frozen, **conv_kwargs).run()
        elif conv_name == 'args_kwargs':
          # fn(10, b=20) -> frozen(10, b=20) -> frozen.run(10, b=20)
          # -> chain._run(10, (), {b:20}) -> Link(10, (), {b:20})
          # -> _evaluate_value: kwargs truthy -> 10(b=20) -> TypeError
          frozen = Chain().then(lambda x: x).freeze()
          with self.assertRaises(TypeError):
            Chain(input_value).then(frozen, 10, b=20).run()


# ---------------------------------------------------------------------------
# TestDoCallableConventionMatrix
# ---------------------------------------------------------------------------

class TestDoCallableConventionMatrix(unittest.TestCase):
  """Cross-product: callable types x calling conventions via do().
  Verify current_value is PRESERVED (do discards fn result).
  """

  def test_regular_fn(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(input_value).do(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, input_value)

  def test_lambda(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(input_value).do(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, input_value)

  def test_builtin_str(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          result = Chain(input_value).do(str).run()
          self.assertEqual(result, input_value)
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          # str() -> '' (no error), do() preserves 42
          result = Chain(input_value).do(str, *conv_args).run()
          self.assertEqual(result, input_value)
        elif conv_name == 'positional':
          # str(10, 20) -> TypeError
          with self.assertRaises(TypeError):
            Chain(input_value).do(str, *conv_args, **conv_kwargs).run()
        else:
          with self.assertRaises(TypeError):
            Chain(input_value).do(str, *conv_args, **conv_kwargs).run()

  def test_builtin_int(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('no_args', 'ellipsis', 'ellipsis_trailing'):
          result = Chain(input_value).do(int, *conv_args, **conv_kwargs).run()
          self.assertEqual(result, input_value)
        else:
          with self.assertRaises(TypeError):
            Chain(input_value).do(int, *conv_args, **conv_kwargs).run()

  def test_builtin_abs(self):
    input_value = -5
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          result = Chain(input_value).do(abs).run()
          self.assertEqual(result, input_value)
        else:
          with self.assertRaises(TypeError):
            Chain(input_value).do(abs, *conv_args, **conv_kwargs).run()

  def test_partial(self):
    input_value = 5
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = functools.partial(operator.add, 10)
        if conv_name == 'no_args':
          result = Chain(input_value).do(fn).run()
          self.assertEqual(result, input_value)
        else:
          with self.assertRaises(TypeError):
            Chain(input_value).do(fn, *conv_args, **conv_kwargs).run()

  def test_class_constructor(self):
    input_value = 5
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          result = Chain(input_value).do(Adder).run()
          self.assertEqual(result, input_value)
        else:
          with self.assertRaises(TypeError):
            Chain(input_value).do(Adder, *conv_args, **conv_kwargs).run()

  def test_callable_obj(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(input_value).do(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, input_value)

  def test_bound_method(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        method, calls = _make_bound_method_tracker()
        result = Chain(input_value).do(method, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, input_value)

  def test_nested_chain(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          inner = Chain(lambda: 'discarded')
          result = Chain(input_value).do(inner, *conv_args).run()
          self.assertEqual(result, input_value)
        elif conv_name == 'no_args':
          inner = Chain().then(lambda x: 'discarded')
          result = Chain(input_value).do(inner).run()
          self.assertEqual(result, input_value)
        elif conv_name == 'positional':
          # v._run(10, (20,), {}) -> Link(10, (20,), {}) -> 10(20) -> TypeError
          inner = Chain().then(lambda x: 'discarded')
          with self.assertRaises(TypeError):
            Chain(input_value).do(inner, *conv_args, **conv_kwargs).run()
        elif conv_name == 'kwargs_only':
          # v._run(42, None, {x:10, y:20}) -> Link(42, None, {x:10, y:20})
          # -> 42(x=10, y=20) -> TypeError
          inner = Chain().then(lambda x: 'discarded')
          with self.assertRaises(TypeError):
            Chain(input_value).do(inner, **conv_kwargs).run()
        elif conv_name == 'args_kwargs':
          # v._run(10, (), {b:20}) -> Link(10, (), {b:20}) -> 10(b=20) -> TypeError
          inner = Chain().then(lambda x: 'discarded')
          with self.assertRaises(TypeError):
            Chain(input_value).do(inner, *conv_args, **conv_kwargs).run()

  def test_frozen_chain(self):
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          frozen = Chain(lambda: 'discarded').freeze()
          result = Chain(input_value).do(frozen, *conv_args).run()
          self.assertEqual(result, input_value)
        elif conv_name == 'no_args':
          frozen = Chain().then(lambda x: 'discarded').freeze()
          result = Chain(input_value).do(frozen).run()
          self.assertEqual(result, input_value)
        elif conv_name == 'positional':
          # frozen(10, 20) -> frozen.run(10, 20) -> _run(10, (20,), {})
          # -> Link(10, (20,), {}) -> 10(20) -> TypeError
          frozen = Chain().then(lambda x: 'discarded').freeze()
          with self.assertRaises(TypeError):
            Chain(input_value).do(frozen, 10, 20).run()
        elif conv_name == 'kwargs_only':
          # frozen(x=10, y=20) -> frozen.run(x=10, y=20)
          # run(v=Null, /, *args, **kwargs) -> _run(Null, (), {x:10, y:20})
          # has_run_value=False -> no root -> first_link gets Null
          # callable with Null -> lambda() -> TypeError
          frozen = Chain().then(lambda x: 'discarded').freeze()
          with self.assertRaises(TypeError):
            Chain(input_value).do(frozen, **conv_kwargs).run()
        elif conv_name == 'args_kwargs':
          # frozen(10, b=20) -> frozen.run(10, b=20) -> _run(10, (), {b:20})
          # -> Link(10, (), {b:20}) -> 10(b=20) -> TypeError
          frozen = Chain().then(lambda x: 'discarded').freeze()
          with self.assertRaises(TypeError):
            Chain(input_value).do(frozen, 10, b=20).run()


# ---------------------------------------------------------------------------
# TestRootCallableConventionMatrix
# ---------------------------------------------------------------------------

class TestRootCallableConventionMatrix(unittest.TestCase):
  """Cross-product for Chain(callable, *args, **kwargs) as root.

  Root evaluation follows _evaluate_value with current_value=Null.
  """

  def test_regular_fn(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(fn, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          self.assertEqual(result, 'no_args_result')
          self.assertEqual(fn.calls[-1], ((), {}))
        elif conv_name == 'positional':
          self.assertEqual(result, 10)
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(result, 'no_args_result')
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(result, 10)
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(result, 'no_args_result')
          self.assertEqual(fn.calls[-1], ((), {}))

  def test_lambda(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(fn, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          self.assertEqual(fn.calls[-1], ((), {}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(fn.calls[-1], ((), {}))

  def test_builtin_str(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          self.assertEqual(Chain(str).run(), '')
        elif conv_name == 'positional':
          # str(10, 20) -> TypeError
          with self.assertRaises(TypeError):
            Chain(str, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(Chain(str, *conv_args).run(), '')
        else:
          with self.assertRaises(TypeError):
            Chain(str, *conv_args, **conv_kwargs).run()

  def test_builtin_int(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          self.assertEqual(Chain(int).run(), 0)
        elif conv_name == 'positional':
          with self.assertRaises(TypeError):
            Chain(int, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(Chain(int, *conv_args).run(), 0)
        else:
          with self.assertRaises(TypeError):
            Chain(int, *conv_args, **conv_kwargs).run()

  def test_builtin_abs(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          # abs() with no args -> TypeError (needs 1)
          with self.assertRaises(TypeError):
            Chain(abs).run()
        elif conv_name == 'positional':
          with self.assertRaises(TypeError):
            Chain(abs, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          with self.assertRaises(TypeError):
            Chain(abs, *conv_args).run()
        else:
          with self.assertRaises(TypeError):
            Chain(abs, *conv_args, **conv_kwargs).run()

  def test_partial(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = functools.partial(operator.add, 10)
        if conv_name == 'no_args':
          # partial(add, 10)() -> missing second arg
          with self.assertRaises(TypeError):
            Chain(fn).run()
        elif conv_name == 'positional':
          # partial(add, 10)(10, 20) -> TypeError
          with self.assertRaises(TypeError):
            Chain(fn, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          with self.assertRaises(TypeError):
            Chain(fn, *conv_args).run()
        else:
          with self.assertRaises(TypeError):
            Chain(fn, *conv_args, **conv_kwargs).run()

  def test_class_constructor(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          with self.assertRaises(TypeError):
            Chain(Adder).run()
        elif conv_name == 'positional':
          # Adder(10, 20) -> TypeError
          with self.assertRaises(TypeError):
            Chain(Adder, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          with self.assertRaises(TypeError):
            Chain(Adder, *conv_args).run()
        elif conv_name == 'kwargs_only':
          result = Chain(Adder, x=10).run()
          self.assertIsInstance(result, Adder)
          self.assertEqual(result.x, 10)
        elif conv_name == 'args_kwargs':
          with self.assertRaises(TypeError):
            Chain(Adder, *conv_args, **conv_kwargs).run()

  def test_callable_obj(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(fn, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          self.assertEqual(fn.calls[-1], ((), {}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(fn.calls[-1], ((), {}))

  def test_bound_method(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        method, calls = _make_bound_method_tracker()
        result = Chain(method, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          self.assertEqual(calls[-1], ((), {}))
          self.assertEqual(result, 'no_args_result')
        elif conv_name == 'positional':
          self.assertEqual(calls[-1], ((10, 20), {}))
          self.assertEqual(result, 10)
        elif conv_name == 'kwargs_only':
          self.assertEqual(calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(calls[-1], ((), {}))


# ---------------------------------------------------------------------------
# TestRunValueConventionMatrix
# ---------------------------------------------------------------------------

class TestRunValueConventionMatrix(unittest.TestCase):
  """Test how .run(v, *args, **kwargs) interacts with calling conventions."""

  def test_run_callable_conventions(self):
    """run(callable, *args, **kwargs) follows same convention as root."""
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        tracker = _make_tracker()
        identity_chain = Chain().then(lambda x: x)

        result = identity_chain.run(tracker, *conv_args, **conv_kwargs)

        if conv_name == 'no_args':
          self.assertEqual(tracker.calls[-1], ((), {}))
          self.assertEqual(result, 'no_args_result')
        elif conv_name == 'positional':
          self.assertEqual(tracker.calls[-1], ((10, 20), {}))
          self.assertEqual(result, 10)
        elif conv_name == 'kwargs_only':
          self.assertEqual(tracker.calls[-1], ((), {'x': 10, 'y': 20}))
          self.assertEqual(result, 'no_args_result')
        elif conv_name == 'args_kwargs':
          self.assertEqual(tracker.calls[-1], ((10,), {'b': 20}))
          self.assertEqual(result, 10)
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(tracker.calls[-1], ((), {}))
          self.assertEqual(result, 'no_args_result')

  def test_run_plain_value(self):
    """run(plain_value) sets it as root -> first link receives it."""
    chain = Chain().then(lambda x: x + 1)
    self.assertEqual(chain.run(10), 11)

  def test_run_plain_value_with_args_type_error(self):
    """run(non_callable, args) -> TypeError since non-callable can't be called."""
    with self.assertRaises(TypeError):
      Chain().then(lambda x: x).run(42, 10)

  def test_run_with_different_callable_types(self):
    """run() with various callable types as the run-value."""
    callable_types = [
      ('tracker', lambda: _make_tracker(), True),
      ('builtin_str', lambda: str, False),
      ('builtin_int', lambda: int, False),
      ('partial', lambda: functools.partial(operator.add, 10), False),
      ('class_constructor', lambda: Adder, False),
    ]

    for type_name, factory, is_tracker in callable_types:
      with self.subTest(callable_type=type_name):
        fn = factory()
        identity_chain = Chain().then(lambda x: x)

        if type_name == 'tracker':
          result = identity_chain.run(fn)
          self.assertEqual(result, 'no_args_result')
        elif type_name == 'builtin_str':
          result = identity_chain.run(str)
          self.assertEqual(result, '')
        elif type_name == 'builtin_int':
          result = identity_chain.run(int)
          self.assertEqual(result, 0)
        elif type_name == 'partial':
          with self.assertRaises(TypeError):
            identity_chain.run(fn)
        elif type_name == 'class_constructor':
          with self.assertRaises(TypeError):
            identity_chain.run(Adder)


# ---------------------------------------------------------------------------
# TestExceptCallableConventionMatrix
# ---------------------------------------------------------------------------

class TestExceptCallableConventionMatrix(unittest.TestCase):
  """Cross-product: callable types x calling conventions via except_().

  When no explicit args are provided, the except handler receives
  (root_value, exception) as arguments. root_value is None when the
  chain has no root value (Null).
  """

  def _make_raiser(self):
    def _raiser():
      raise ValueError('boom')
    return _raiser

  def test_regular_fn(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(self._make_raiser()).except_(fn, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          # Handler receives (root_value, exc); root_value is None (Null root)
          self.assertEqual(len(fn.calls[-1][0]), 2)
          self.assertIsNone(fn.calls[-1][0][0])
          self.assertIsInstance(fn.calls[-1][0][1], ValueError)
        elif conv_name == 'positional':
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(fn.calls[-1], ((), {}))

  def test_lambda(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(self._make_raiser()).except_(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(len(fn.calls), 1)

  def test_callable_obj(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(self._make_raiser()).except_(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(len(fn.calls), 1)

  def test_bound_method(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        method, calls = _make_bound_method_tracker()
        result = Chain(self._make_raiser()).except_(method, *conv_args, **conv_kwargs).run()
        self.assertEqual(len(calls), 1)
        if conv_name == 'no_args':
          # Handler receives (root_value, exc); root_value is None (Null root)
          self.assertEqual(len(calls[-1][0]), 2)
          self.assertIsNone(calls[-1][0][0])
          self.assertIsInstance(calls[-1][0][1], ValueError)
        elif conv_name == 'positional':
          self.assertEqual(calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(calls[-1], ((), {}))

  def test_partial(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          # Handler receives (root_value, exc); partial binds prefix
          handler = functools.partial(
            lambda prefix, rv, exc: f'{prefix}:{type(exc).__name__}', 'pfx'
          )
          result = Chain(self._make_raiser()).except_(handler).run()
          self.assertEqual(result, 'pfx:ValueError')
        elif conv_name == 'positional':
          handler = functools.partial(lambda prefix, a, b: f'{prefix}:{a}', 'pfx')
          result = Chain(self._make_raiser()).except_(handler, 10, 20).run()
          self.assertEqual(result, 'pfx:10')
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          handler = functools.partial(lambda prefix: f'{prefix}:handled', 'pfx')
          result = Chain(self._make_raiser()).except_(handler, *conv_args).run()
          self.assertEqual(result, 'pfx:handled')
        else:
          # kwargs combos with partial -- may or may not work
          try:
            handler = functools.partial(lambda prefix, **kw: f'{prefix}', 'pfx')
            Chain(self._make_raiser()).except_(handler, *conv_args, **conv_kwargs).run()
          except TypeError:
            pass

  def test_class_constructor(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          # Adder(rv, exc) -> TypeError (Adder only takes 1 arg)
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(Adder).run()
        elif conv_name == 'positional':
          # Adder(10, 20) -> TypeError
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(Adder, 10, 20).run()
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          # Adder() -> TypeError
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(Adder, *conv_args).run()
        elif conv_name == 'kwargs_only':
          result = Chain(self._make_raiser()).except_(Adder, x='custom').run()
          self.assertIsInstance(result, Adder)
          self.assertEqual(result.x, 'custom')
        elif conv_name == 'args_kwargs':
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(Adder, *conv_args, **conv_kwargs).run()

  def test_nested_chain(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          inner = Chain(lambda: 'chain_handled')
          result = Chain(self._make_raiser()).except_(inner, *conv_args).run()
          self.assertEqual(result, 'chain_handled')
        elif conv_name == 'no_args':
          # Nested chain receives rv (=None since root_value is Null)
          inner = Chain().then(lambda rv: f'chain:{type(rv).__name__}')
          result = Chain(self._make_raiser()).except_(inner).run()
          self.assertEqual(result, 'chain:NoneType')
        elif conv_name == 'positional':
          # v._run(10, (20,), {}) -> Link(10, (20,), {}) -> 10(20) -> TypeError
          inner = Chain().then(lambda rv: f'chain:{type(rv).__name__}')
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(inner, 10, 20).run()
        elif conv_name == 'kwargs_only':
          # v._run(rv, None, {x:10, y:20}) -> Link(rv, None, {x:10, y:20})
          # rv is None, not callable -> None(x=10, y=20) -> TypeError
          inner = Chain().then(lambda rv: f'chain:{type(rv).__name__}')
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(inner, **conv_kwargs).run()
        elif conv_name == 'args_kwargs':
          # v._run(10, (), {b:20}) -> Link(10, (), {b:20}) -> 10(b=20) -> TypeError
          inner = Chain().then(lambda rv: f'chain:{type(rv).__name__}')
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(inner, 10, b=20).run()

  def test_frozen_chain(self):
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          frozen = Chain(lambda: 'frozen_handled').freeze()
          result = Chain(self._make_raiser()).except_(frozen, *conv_args).run()
          self.assertEqual(result, 'frozen_handled')
        elif conv_name == 'no_args':
          # frozen(rv, exc) -> frozen.run(None, ValueError('boom'))
          # -> _run(None, (ValueError('boom'),), {})
          # has_run_value=True -> Link(None, (ValueError('boom'),), {})
          # args truthy -> None(exc) -> TypeError (None not callable)
          frozen = Chain().then(lambda exc: f'frozen:{type(exc).__name__}').freeze()
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(frozen).run()
        elif conv_name == 'positional':
          # frozen(10, 20) -> frozen.run(10, 20) -> _run(10, (20,), {})
          # -> Link(10, (20,), {}) -> 10(20) -> TypeError
          frozen = Chain().then(lambda x: f'frozen:{x}').freeze()
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(frozen, 10, 20).run()
        elif conv_name == 'kwargs_only':
          # frozen(x=10, y=20) -> frozen.run(x=10, y=20) -> _run(Null, (), {x:10, y:20})
          # has_run_value=False -> no root -> first_link with Null
          # lambda(**kw) can accept Null? No: callable with Null -> lambda() but
          # lambda(**kw) needs no positional -> works
          frozen = Chain().then(lambda **kw: 'frozen:handled').freeze()
          result = Chain(self._make_raiser()).except_(frozen, **conv_kwargs).run()
          self.assertEqual(result, 'frozen:handled')
        elif conv_name == 'args_kwargs':
          # frozen(10, b=20) -> frozen.run(10, b=20) -> _run(10, (), {b:20})
          # -> Link(10, (), {b:20}) -> kwargs truthy -> 10(b=20) -> TypeError
          frozen = Chain().then(lambda x, **kw: 'frozen:handled').freeze()
          with self.assertRaises(TypeError):
            Chain(self._make_raiser()).except_(frozen, 10, b=20).run()


# ---------------------------------------------------------------------------
# TestFinallyCallableConventionMatrix
# ---------------------------------------------------------------------------

class TestFinallyCallableConventionMatrix(unittest.TestCase):
  """Cross-product: callable types x calling conventions via finally_().

  The finally handler receives root_value as current_value.
  """

  def test_regular_fn(self):
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(root_value).finally_(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, root_value)
        self.assertEqual(len(fn.calls), 1)
        if conv_name == 'no_args':
          self.assertEqual(fn.calls[-1][0], (root_value,))
        elif conv_name == 'positional':
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(fn.calls[-1], ((), {}))

  def test_lambda(self):
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(root_value).finally_(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, root_value)

  def test_callable_obj(self):
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_tracker()
        result = Chain(root_value).finally_(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, root_value)

  def test_bound_method(self):
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        method, calls = _make_bound_method_tracker()
        result = Chain(root_value).finally_(method, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, root_value)
        self.assertEqual(len(calls), 1)
        if conv_name == 'no_args':
          self.assertEqual(calls[-1][0], (root_value,))
        elif conv_name == 'positional':
          self.assertEqual(calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(calls[-1], ((), {}))

  def test_partial(self):
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name == 'no_args':
          handler = functools.partial(lambda prefix, rv: f'{prefix}:{rv}', 'fin')
          result = Chain(root_value).finally_(handler).run()
          self.assertEqual(result, root_value)
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          handler = functools.partial(lambda prefix: f'{prefix}:done', 'fin')
          result = Chain(root_value).finally_(handler, *conv_args).run()
          self.assertEqual(result, root_value)
        else:
          handler = functools.partial(lambda prefix, *a, **kw: None, 'fin')
          result = Chain(root_value).finally_(handler, *conv_args, **conv_kwargs).run()
          self.assertEqual(result, root_value)

  def test_nested_chain(self):
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          inner = Chain(lambda: 'fin_result')
          result = Chain(root_value).finally_(inner, *conv_args).run()
          self.assertEqual(result, root_value)
        elif conv_name == 'no_args':
          inner = Chain().then(lambda rv: f'fin:{rv}')
          result = Chain(root_value).finally_(inner).run()
          self.assertEqual(result, root_value)
        elif conv_name == 'positional':
          # v._run(10, (20,), {}) -> Link(10, (20,), {}) -> 10(20) -> TypeError
          inner = Chain().then(lambda rv: f'fin:{rv}')
          with self.assertRaises(TypeError):
            Chain(root_value).finally_(inner, 10, 20).run()
        elif conv_name == 'kwargs_only':
          # v._run(42, None, {x:10, y:20}) -> Link(42, None, {x:10, y:20})
          # -> 42(x=10, y=20) -> TypeError
          inner = Chain().then(lambda rv: f'fin:{rv}')
          with self.assertRaises(TypeError):
            Chain(root_value).finally_(inner, **conv_kwargs).run()
        elif conv_name == 'args_kwargs':
          # v._run(10, (), {b:20}) -> Link(10, (), {b:20}) -> 10(b=20) -> TypeError
          inner = Chain().then(lambda rv: f'fin:{rv}')
          with self.assertRaises(TypeError):
            Chain(root_value).finally_(inner, 10, b=20).run()

  def test_frozen_chain(self):
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        if conv_name in ('ellipsis', 'ellipsis_trailing'):
          frozen = Chain(lambda: 'fin_result').freeze()
          result = Chain(root_value).finally_(frozen, *conv_args).run()
          self.assertEqual(result, root_value)
        elif conv_name == 'no_args':
          frozen = Chain().then(lambda rv: f'fin:{rv}').freeze()
          result = Chain(root_value).finally_(frozen).run()
          self.assertEqual(result, root_value)
        elif conv_name == 'positional':
          # frozen(10, 20) -> frozen.run(10, 20) -> _run(10, (20,), {})
          # -> Link(10, (20,), {}) -> 10(20) -> TypeError
          frozen = Chain().then(lambda x: f'fin:{x}').freeze()
          with self.assertRaises(TypeError):
            Chain(root_value).finally_(frozen, 10, 20).run()
        elif conv_name == 'kwargs_only':
          # frozen(x=10, y=20) -> frozen.run(Null, x=10, y=20)
          # has_run_value=False -> first_link with Null -> lambda() -> TypeError
          frozen = Chain().then(lambda **kw: 'fin_result').freeze()
          result = Chain(root_value).finally_(frozen, **conv_kwargs).run()
          self.assertEqual(result, root_value)
        elif conv_name == 'args_kwargs':
          # frozen(10, b=20) -> frozen.run(10, b=20) -> _run(10, (), {b:20})
          # -> Link(10, (), {b:20}) -> 10(b=20) -> TypeError
          frozen = Chain().then(lambda x, **kw: f'fin:{x}').freeze()
          with self.assertRaises(TypeError):
            Chain(root_value).finally_(frozen, 10, b=20).run()


# ---------------------------------------------------------------------------
# TestCurrentValueStateMatrix
# ---------------------------------------------------------------------------

class TestCurrentValueStateMatrix(unittest.TestCase):
  """Cross current_value states with calling conventions via then() and do()."""

  STATES = [
    ('Null', None, True),
    ('None', None, False),
    ('zero', 0, False),
    ('False', False, False),
    ('empty_str', '', False),
    ('empty_list', [], False),
    ('int_42', 42, False),
    ('str_hello', 'hello', False),
  ]

  def test_then_matrix(self):
    """then() with each current_value state x each convention."""
    for state_name, state_value, is_null in self.STATES:
      for conv_name, conv_args, conv_kwargs in CONVENTIONS:
        with self.subTest(current_value=state_name, convention=conv_name):
          tracker = _make_tracker()

          if is_null:
            chain = Chain().then(tracker, *conv_args, **conv_kwargs)
          else:
            chain = Chain(state_value).then(tracker, *conv_args, **conv_kwargs)

          result = chain.run()

          if conv_name == 'no_args':
            if is_null:
              self.assertEqual(tracker.calls[-1], ((), {}))
              self.assertEqual(result, 'no_args_result')
            else:
              self.assertEqual(tracker.calls[-1][0], (state_value,))
              self.assertEqual(result, state_value)
          elif conv_name == 'positional':
            self.assertEqual(tracker.calls[-1], ((10, 20), {}))
            self.assertEqual(result, 10)
          elif conv_name == 'kwargs_only':
            self.assertEqual(tracker.calls[-1], ((), {'x': 10, 'y': 20}))
            self.assertEqual(result, 'no_args_result')
          elif conv_name == 'args_kwargs':
            self.assertEqual(tracker.calls[-1], ((10,), {'b': 20}))
            self.assertEqual(result, 10)
          elif conv_name in ('ellipsis', 'ellipsis_trailing'):
            self.assertEqual(tracker.calls[-1], ((), {}))
            self.assertEqual(result, 'no_args_result')

  def test_do_matrix(self):
    """do() with each current_value state x each convention -- preserves current_value."""
    for state_name, state_value, is_null in self.STATES:
      for conv_name, conv_args, conv_kwargs in CONVENTIONS:
        with self.subTest(current_value=state_name, convention=conv_name):
          tracker = _make_tracker()

          if is_null:
            chain = Chain().do(tracker, *conv_args, **conv_kwargs)
          else:
            chain = Chain(state_value).do(tracker, *conv_args, **conv_kwargs)

          result = chain.run()

          # do() discards fn result -> current_value is preserved
          if is_null:
            # No root value -> current_value stays Null -> returns None
            self.assertIsNone(result)
          else:
            self.assertEqual(result, state_value)

          # Verify tracker was called with correct args
          self.assertEqual(len(tracker.calls), 1)
          if conv_name == 'no_args':
            if is_null:
              self.assertEqual(tracker.calls[-1], ((), {}))
            else:
              self.assertEqual(tracker.calls[-1][0], (state_value,))
          elif conv_name == 'positional':
            self.assertEqual(tracker.calls[-1], ((10, 20), {}))
          elif conv_name == 'kwargs_only':
            self.assertEqual(tracker.calls[-1], ((), {'x': 10, 'y': 20}))
          elif conv_name == 'args_kwargs':
            self.assertEqual(tracker.calls[-1], ((10,), {'b': 20}))
          elif conv_name in ('ellipsis', 'ellipsis_trailing'):
            self.assertEqual(tracker.calls[-1], ((), {}))


# ---------------------------------------------------------------------------
# TestAsyncCallableConventionMatrix
# ---------------------------------------------------------------------------

class TestAsyncCallableConventionMatrix(IsolatedAsyncioTestCase):
  """Async callable types x all conventions for then(), do(), except_, finally_."""

  async def test_then_async_fn(self):
    """Async function via then() with all conventions."""
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_async_tracker()
        result = await Chain(input_value).then(fn, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          self.assertEqual(result, input_value)
          self.assertEqual(fn.calls[-1], ((input_value,), {}))
        elif conv_name == 'positional':
          self.assertEqual(result, 10)
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(result, 'no_args_result')
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(result, 10)
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(result, 'no_args_result')
          self.assertEqual(fn.calls[-1], ((), {}))

  async def test_then_async_callable_obj(self):
    """Async callable object via then() with all conventions."""
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _AsyncCallableObjTracker()
        result = await Chain(input_value).then(fn, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          self.assertEqual(result, input_value)
          self.assertEqual(fn.calls[-1], ((input_value,), {}))
        elif conv_name == 'positional':
          self.assertEqual(result, 10)
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(result, 'no_args_result')
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(result, 10)
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(result, 'no_args_result')
          self.assertEqual(fn.calls[-1], ((), {}))

  async def test_do_async_fn(self):
    """Async function via do() with all conventions -- current_value preserved."""
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_async_tracker()
        result = await Chain(input_value).do(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, input_value)
        if conv_name == 'no_args':
          self.assertEqual(fn.calls[-1], ((input_value,), {}))
        elif conv_name == 'positional':
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(fn.calls[-1], ((), {}))

  async def test_do_async_callable_obj(self):
    """Async callable object via do() with all conventions."""
    input_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _AsyncCallableObjTracker()
        result = await Chain(input_value).do(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(result, input_value)

  async def test_except_async_fn(self):
    """Async function via except_() with all conventions."""
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_async_tracker()

        async def _async_raiser():
          raise ValueError('boom')

        result = await Chain(_async_raiser).except_(fn, *conv_args, **conv_kwargs).run()
        if conv_name == 'no_args':
          # Handler receives (root_value, exc); root_value is None (Null root)
          self.assertEqual(len(fn.calls[-1][0]), 2)
          self.assertIsNone(fn.calls[-1][0][0])
          self.assertIsInstance(fn.calls[-1][0][1], ValueError)
        elif conv_name == 'positional':
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(fn.calls[-1], ((), {}))

  async def test_except_async_callable_obj(self):
    """Async callable object via except_() with all conventions."""
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _AsyncCallableObjTracker()

        async def _async_raiser():
          raise ValueError('boom')

        result = await Chain(_async_raiser).except_(fn, *conv_args, **conv_kwargs).run()
        self.assertEqual(len(fn.calls), 1)

  async def test_finally_async_fn(self):
    """Async function via finally_() with all conventions."""
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _make_async_tracker()

        async def _async_identity(x):
          return x

        result = await (
          Chain(root_value)
          .then(_async_identity)
          .finally_(fn, *conv_args, **conv_kwargs)
          .run()
        )
        self.assertEqual(result, root_value)
        if conv_name == 'no_args':
          self.assertEqual(fn.calls[-1][0], (root_value,))
        elif conv_name == 'positional':
          self.assertEqual(fn.calls[-1], ((10, 20), {}))
        elif conv_name == 'kwargs_only':
          self.assertEqual(fn.calls[-1], ((), {'x': 10, 'y': 20}))
        elif conv_name == 'args_kwargs':
          self.assertEqual(fn.calls[-1], ((10,), {'b': 20}))
        elif conv_name in ('ellipsis', 'ellipsis_trailing'):
          self.assertEqual(fn.calls[-1], ((), {}))

  async def test_finally_async_callable_obj(self):
    """Async callable object via finally_() with all conventions."""
    root_value = 42
    for conv_name, conv_args, conv_kwargs in CONVENTIONS:
      with self.subTest(convention=conv_name):
        fn = _AsyncCallableObjTracker()

        async def _async_identity(x):
          return x

        result = await (
          Chain(root_value)
          .then(_async_identity)
          .finally_(fn, *conv_args, **conv_kwargs)
          .run()
        )
        self.assertEqual(result, root_value)

  async def test_current_value_state_async(self):
    """Async then() with various current_value states."""
    states = [
      ('Null', None, True),
      ('None', None, False),
      ('zero', 0, False),
      ('False', False, False),
      ('empty_str', '', False),
      ('int_42', 42, False),
      ('str_hello', 'hello', False),
    ]

    for state_name, state_value, is_null in states:
      for conv_name, conv_args, conv_kwargs in CONVENTIONS:
        with self.subTest(current_value=state_name, convention=conv_name):
          tracker = _make_async_tracker()

          if is_null:
            chain = Chain().then(tracker, *conv_args, **conv_kwargs)
          else:
            chain = Chain(state_value).then(tracker, *conv_args, **conv_kwargs)

          result = await chain.run()

          if conv_name == 'no_args':
            if is_null:
              self.assertEqual(tracker.calls[-1], ((), {}))
              self.assertEqual(result, 'no_args_result')
            else:
              self.assertEqual(tracker.calls[-1][0], (state_value,))
              self.assertEqual(result, state_value)
          elif conv_name == 'positional':
            self.assertEqual(tracker.calls[-1], ((10, 20), {}))
            self.assertEqual(result, 10)
          elif conv_name == 'kwargs_only':
            self.assertEqual(tracker.calls[-1], ((), {'x': 10, 'y': 20}))
            self.assertEqual(result, 'no_args_result')
          elif conv_name == 'args_kwargs':
            self.assertEqual(tracker.calls[-1], ((10,), {'b': 20}))
            self.assertEqual(result, 10)
          elif conv_name in ('ellipsis', 'ellipsis_trailing'):
            self.assertEqual(tracker.calls[-1], ((), {}))
            self.assertEqual(result, 'no_args_result')


if __name__ == '__main__':
  unittest.main()
