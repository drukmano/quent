"""Exhaustive tests for _evaluate_value and _resolve_value from quent._core."""
from __future__ import annotations

import asyncio
import functools
import inspect
import operator
import unittest
from typing import Any

from quent import Chain, Null
from quent._core import Link, _evaluate_value, _resolve_value
from helpers import (
  Adder,
  AsyncCallableObj,
  BoundMethodHolder,
  CallableObj,
  async_fn,
  async_identity,
  make_async_tracker,
  make_tracker,
  partial_fn,
  sync_fn,
  sync_identity,
)


# ---------------------------------------------------------------------------
# Helpers local to this module
# ---------------------------------------------------------------------------

def _run_maybe_async(result):
  """Await a coroutine via asyncio.run(); return sync values unchanged."""
  if inspect.isawaitable(result):
    return asyncio.run(result)
  return result


class _CallableObjMulti:
  """Callable object accepting *args, **kwargs."""

  def __call__(self, *args, **kwargs):
    return (args, kwargs)


class _AsyncCallableObjMulti:
  """Async callable object accepting *args, **kwargs."""

  async def __call__(self, *args, **kwargs):
    return (args, kwargs)


class _BoundMethodHolderMulti:
  def method(self, *args, **kwargs):
    return (args, kwargs)


# ---------------------------------------------------------------------------
# TestEvaluateValueCallableMatrix
# ---------------------------------------------------------------------------

# Each entry: name -> factory returning (callable, is_async).
# All generic callables accept (*args, **kwargs) and return (args, kwargs).
_CALLABLE_TYPES: dict[str, tuple[Any, bool]] = {}


def _build_callable_types():
  """Build the callable type registry. Called once at import time."""
  return {
    'sync_fn': (_CallableObjMulti(), False),
    'async_fn': (_AsyncCallableObjMulti(), True),
    'lambda': (lambda *args, **kwargs: (args, kwargs), False),
    'builtin_len': (len, False),
    'partial': (functools.partial(_CallableObjMulti(), 'partial_arg'), False),
    'class_constructor': (Adder, False),
    'callable_object': (_CallableObjMulti(), False),
    'async_callable_object': (_AsyncCallableObjMulti(), True),
    'bound_method': (_BoundMethodHolderMulti().method, False),
  }


# Types that accept generic *args/**kwargs and return (args, kwargs)
_GENERIC_TYPES = frozenset({
  'sync_fn', 'async_fn', 'lambda', 'callable_object',
  'async_callable_object', 'bound_method',
})


class TestEvaluateValueCallableMatrix(unittest.TestCase):
  """Cross-product: callable type x calling convention via _evaluate_value."""

  def setUp(self):
    self.callables = _build_callable_types()

  # -- Convention: no args --

  def test_callable_type_x_convention_no_args(self):
    """With no args/kwargs on the link, callable receives current_value."""
    for name, (fn, _is_async) in self.callables.items():
      with self.subTest(callable_type=name):
        link = Link(fn)
        if name == 'builtin_len':
          # len('cv') = 2
          result = _run_maybe_async(_evaluate_value(link, 'cv'))
          self.assertEqual(result, 2)
        elif name == 'class_constructor':
          result = _run_maybe_async(_evaluate_value(link, 99))
          self.assertIsInstance(result, Adder)
          self.assertEqual(result.x, 99)
        elif name == 'partial':
          # partial prepends 'partial_arg': fn('cv') -> (('partial_arg', 'cv'), {})
          result = _run_maybe_async(_evaluate_value(link, 'cv'))
          self.assertEqual(result, (('partial_arg', 'cv'), {}))
        else:
          # Generic: fn(current_value) → (('cv',), {})
          result = _run_maybe_async(_evaluate_value(link, 'cv'))
          self.assertEqual(result, (('cv',), {}))

  # -- Convention: pos args --

  def test_callable_type_x_convention_pos_args(self):
    """Explicit positional args override current_value."""
    for name, (fn, _is_async) in self.callables.items():
      with self.subTest(callable_type=name):
        if name == 'builtin_len':
          link = Link(fn, ([1, 2, 3],))
          result = _run_maybe_async(_evaluate_value(link, 'ignored'))
          self.assertEqual(result, 3)
        elif name == 'class_constructor':
          link = Link(fn, (42,))
          result = _run_maybe_async(_evaluate_value(link, 'ignored'))
          self.assertIsInstance(result, Adder)
          self.assertEqual(result.x, 42)
        elif name == 'partial':
          link = Link(fn, (10, 20))
          result = _run_maybe_async(_evaluate_value(link, 'ignored'))
          self.assertEqual(result, (('partial_arg', 10, 20), {}))
        else:
          link = Link(fn, (10, 20))
          result = _run_maybe_async(_evaluate_value(link, 'ignored'))
          self.assertEqual(result, ((10, 20), {}))

  # -- Convention: kwargs only --

  def test_callable_type_x_convention_kwargs(self):
    """Only kwargs provided -- triggers `if args or kwargs` branch."""
    for name, (fn, _is_async) in self.callables.items():
      with self.subTest(callable_type=name):
        if name in ('builtin_len', 'class_constructor'):
          continue  # builtins/constructors don't accept arbitrary **kwargs
        link = Link(fn, None, {'k': 99})
        result = _run_maybe_async(_evaluate_value(link, 'ignored'))
        if name == 'partial':
          self.assertEqual(result, (('partial_arg',), {'k': 99}))
        else:
          self.assertEqual(result, ((), {'k': 99}))

  # -- Convention: args + kwargs --

  def test_callable_type_x_convention_args_kwargs(self):
    """Both args and kwargs provided."""
    for name, (fn, _is_async) in self.callables.items():
      with self.subTest(callable_type=name):
        if name in ('builtin_len', 'class_constructor'):
          continue
        link = Link(fn, (10,), {'k': 99})
        result = _run_maybe_async(_evaluate_value(link, 'ignored'))
        if name == 'partial':
          self.assertEqual(result, (('partial_arg', 10), {'k': 99}))
        else:
          self.assertEqual(result, ((10,), {'k': 99}))

  # -- Convention: ellipsis --

  def test_callable_type_x_convention_ellipsis(self):
    """Ellipsis as sole arg -> v() with no arguments."""
    for name, (fn, _is_async) in self.callables.items():
      with self.subTest(callable_type=name):
        if name == 'builtin_len':
          # len() raises TypeError
          link = Link(fn, (...,))
          with self.assertRaises(TypeError):
            _evaluate_value(link, 'ignored')
          continue
        if name == 'class_constructor':
          # Adder() raises TypeError (missing x)
          link = Link(fn, (...,))
          with self.assertRaises(TypeError):
            _evaluate_value(link, 'ignored')
          continue
        link = Link(fn, (...,))
        result = _run_maybe_async(_evaluate_value(link, 'ignored'))
        if name == 'partial':
          # partial prepends 'partial_arg': fn() -> (('partial_arg',), {})
          self.assertEqual(result, (('partial_arg',), {}))
        else:
          self.assertEqual(result, ((), {}))



# ---------------------------------------------------------------------------
# TestEvaluateValueChainNested
# ---------------------------------------------------------------------------

class TestEvaluateValueChainNested(unittest.TestCase):
  """Tests for _evaluate_value when link.is_chain is True (nested Chain)."""

  def test_nested_chain_no_args_passes_current_value(self):
    """No args/kwargs -> v._run(current_value, None, None)."""
    inner = Chain()
    inner._then(Link(lambda x: x * 3))
    link = Link(inner)
    result = _evaluate_value(link, 7)
    self.assertEqual(result, 21)

  def test_nested_chain_ellipsis_passes_null(self):
    """Ellipsis -> v._run(Null, None, None); inner chain starts with Null."""
    inner = Chain(lambda: 42)
    link = Link(inner, (...,))
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(result, 42)

  def test_nested_chain_with_args_first_is_value(self):
    """args=(5,) -> v._run(5, (), {}).

    The first element of args becomes the run value passed to _run.
    """
    inner = Chain()
    inner._then(Link(lambda x: x + 100))
    link = Link(inner, (5,))
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(result, 105)

  def test_nested_chain_with_multiple_args(self):
    """args=(callable, arg1) -> v._run(callable, (arg1,), {}).

    The first element becomes the run value; remaining elements become
    its positional args. This only works when the first element is callable.
    """
    fn = lambda x: x * 10
    inner = Chain()
    inner._then(Link(lambda x: x + 1))
    link = Link(inner, (fn, 7))
    result = _evaluate_value(link, 'ignored')
    # _run(fn, (7,), {}) -> Link(fn, (7,), {}) -> fn(7) = 70
    # then lambda(70) = 71
    self.assertEqual(result, 71)

  def test_nested_chain_with_args_and_kwargs(self):
    """args=(callable,), kwargs={'k': 1} -> v._run(callable, (), {'k': 1}).

    The run value must be callable to receive kwargs. Inside _run,
    Link(callable, (), {'k': 1}) is created and _evaluate_value calls
    callable(**{'k': 1}).
    """
    fn = lambda k=0: k + 10
    inner = Chain()
    inner._then(Link(lambda x: x + 1))
    link = Link(inner, (fn,), {'k': 5})
    result = _evaluate_value(link, 'ignored')
    # _run(fn, (), {'k': 5}) -> Link(fn, (), {'k': 5})
    # _evaluate_value: args=(), kwargs={'k': 5}
    # () or {'k': 5} -> {'k': 5} truthy
    # fn(**{'k': 5}) -> fn(k=5) -> 15
    # then lambda(15) -> 16
    self.assertEqual(result, 16)

  def test_nested_chain_with_args_empty_and_kwargs_BUG1(self):
    """args=(), kwargs={'k': 1}.

    Walkthrough of _evaluate_value (chain branch):
    - `if args or kwargs`: `() or {'k': 1}` -> `{'k': 1}` (truthy).
    - `args[0] if args else current_value` -> current_value (args is falsy).
    - `args[1:] if args else None` -> None.
    - `kwargs or {}` -> {'k': 1}.
    - Call: v._run(current_value, None, {'k': 1}).

    Inside _run: v=current_value, args=None, kwargs={'k': 1}.
    Creates Link(current_value, None, {'k': 1}).
    _evaluate_value on that link: args=None (falsy), kwargs={'k': 1}.
    `if args or kwargs`: `None or {'k': 1}` -> truthy.
    `v(*(args or ()), **(kwargs or {}))` -> current_value(**{'k': 1}).
    This only works if current_value is callable.

    So kwargs ARE preserved through _evaluate_value's chain branch,
    but they become kwargs to the run value itself inside _run.
    """
    fn = lambda k=0: k + 100
    inner = Chain()
    inner._then(Link(lambda x: x * 2))
    link = Link(inner, (), {'k': 5})
    # current_value is fn (a callable)
    result = _evaluate_value(link, fn)
    # _run(fn, None, {'k': 5}) -> Link(fn, None, {'k': 5})
    # _evaluate_value: fn(k=5) -> 105
    # then lambda(105) -> 210
    self.assertEqual(result, 210)

  def test_nested_chain_with_args_none_and_kwargs(self):
    """args=None, kwargs={'k': 1}.

    _evaluate_value (chain branch):
    - `if args or kwargs`: `None or {'k': 1}` -> `{'k': 1}` (truthy).
    - `args[0] if args else current_value` -> current_value.
    - `args[1:] if args else None` -> None.
    - `kwargs or {}` -> {'k': 1}.
    - Call: v._run(current_value, None, {'k': 1}).

    kwargs are passed as the run value's kwargs inside _run,
    so the current_value (run value) must be callable.
    """
    fn = lambda k=0: k + 10
    inner = Chain()
    inner._then(Link(lambda x: x + 5))
    link = Link(inner, None, {'k': 3})
    result = _evaluate_value(link, fn)
    # _run(fn, None, {'k': 3}) -> Link(fn, None, {'k': 3})
    # _evaluate_value: fn(k=3) -> 13
    # then lambda(13) -> 18
    self.assertEqual(result, 18)

  def test_nested_chain_with_kwargs_only(self):
    """kwargs only (args=None) -> current_value used as run value with kwargs.

    Since kwargs are passed to the run value, the current_value must be
    callable if we want it to receive those kwargs.
    """
    fn = lambda unused=False: 50 if not unused else 99
    inner = Chain()
    inner._then(Link(lambda x: x - 1))
    link = Link(inner, None, {'unused': True})
    result = _evaluate_value(link, fn)
    # _run(fn, None, {'unused': True}) -> Link(fn, None, {'unused': True})
    # _evaluate_value: fn(unused=True) -> 99
    # then lambda(99) -> 98
    self.assertEqual(result, 98)


# ---------------------------------------------------------------------------
# TestEvaluateValueNonCallable
# ---------------------------------------------------------------------------

class TestEvaluateValueNonCallable(unittest.TestCase):
  """Tests for _evaluate_value when v is a non-callable literal."""

  def test_literal_int_returned(self):
    link = Link(42)
    self.assertEqual(_evaluate_value(link, 'anything'), 42)

  def test_literal_none_returned(self):
    link = Link(None)
    self.assertIsNone(_evaluate_value(link, 'anything'))

  def test_literal_false_returned(self):
    link = Link(False)
    self.assertIs(_evaluate_value(link, 'anything'), False)

  def test_literal_empty_string_returned(self):
    link = Link('')
    self.assertEqual(_evaluate_value(link, 'anything'), '')

  def test_literal_list_returned(self):
    lst = [1, 2, 3]
    link = Link(lst)
    self.assertIs(_evaluate_value(link, 'anything'), lst)

  def test_literal_null_returned(self):
    """Null is not callable -> returned as-is."""
    link = Link(Null)
    result = _evaluate_value(link, 'anything')
    self.assertIs(result, Null)

  def test_non_callable_with_args_raises(self):
    """42(1) -> TypeError because int is not callable."""
    link = Link(42, (1,))
    with self.assertRaises(TypeError):
      _evaluate_value(link, 'anything')

  def test_literal_zero_returned(self):
    link = Link(0)
    self.assertEqual(_evaluate_value(link, 'cv'), 0)

  def test_literal_empty_dict_returned(self):
    d = {}
    link = Link(d)
    self.assertIs(_evaluate_value(link, 'cv'), d)

  def test_literal_tuple_returned(self):
    t = (1, 2)
    link = Link(t)
    self.assertEqual(_evaluate_value(link, 'cv'), t)

  def test_literal_frozenset_returned(self):
    fs = frozenset([1, 2, 3])
    link = Link(fs)
    self.assertIs(_evaluate_value(link, 'cv'), fs)

  def test_literal_bytes_returned(self):
    b = b'hello'
    link = Link(b)
    self.assertEqual(_evaluate_value(link, 'cv'), b)


# ---------------------------------------------------------------------------
# TestResolveValueDense
# ---------------------------------------------------------------------------

class TestResolveValueDense(unittest.TestCase):
  """Dense coverage of _resolve_value."""

  def test_callable_no_args_no_kwargs(self):
    result = _resolve_value(lambda: 'ok', None, None)
    self.assertEqual(result, 'ok')

  def test_callable_with_args(self):
    result = _resolve_value(lambda a, b: a + b, (3, 4), None)
    self.assertEqual(result, 7)

  def test_callable_with_kwargs(self):
    result = _resolve_value(lambda k=0: k * 2, None, {'k': 5})
    self.assertEqual(result, 10)

  def test_callable_with_both(self):
    result = _resolve_value(lambda a, k=0: a + k, (1,), {'k': 9})
    self.assertEqual(result, 10)

  def test_callable_with_ellipsis(self):
    result = _resolve_value(lambda: 'ell', (...,), None)
    self.assertEqual(result, 'ell')

  def test_callable_with_ellipsis_and_kwargs_raises(self):
    """Ellipsis cannot be combined with other arguments (including kwargs)."""
    from quent import QuentException
    with self.assertRaises(QuentException):
      _resolve_value(lambda: 'ell', (...,), {'k': 99})

  def test_non_callable_no_args(self):
    result = _resolve_value(42, None, None)
    self.assertEqual(result, 42)

  def test_non_callable_with_args(self):
    """Non-callable with args -> args gets set to the tuple, v(*args) raises."""
    with self.assertRaises(TypeError):
      _resolve_value(42, (1,), None)

  def test_empty_args_tuple_treated_as_no_args(self):
    """args=() -> `args or ()` -> (), `args and ...` is False.

    Then `args or kwargs` -> `() or {}` -> {} which is falsy.
    Falls to `v() if callable(v) else v`.
    """
    result = _resolve_value(lambda: 'no_arg', (), None)
    self.assertEqual(result, 'no_arg')

  def test_empty_kwargs_dict_treated_as_no_kwargs(self):
    """kwargs={} -> `kwargs or {}` -> {}.

    `args or kwargs` -> `() or {}` -> {} -> falsy.
    Falls to `v() if callable(v) else v`.
    """
    result = _resolve_value(lambda: 'no_kwarg', None, {})
    self.assertEqual(result, 'no_kwarg')

  def test_empty_args_and_empty_kwargs(self):
    """Both empty -> treated as no args."""
    result = _resolve_value(lambda: 'empty', (), {})
    self.assertEqual(result, 'empty')

  def test_non_callable_none_returned(self):
    result = _resolve_value(None, None, None)
    self.assertIsNone(result)

  def test_non_callable_false_returned(self):
    result = _resolve_value(False, None, None)
    self.assertIs(result, False)

  def test_args_none_kwargs_present(self):
    """args=None, kwargs={'x': 1} -> args becomes (), kwargs stays.

    `args or ()` -> (). `args and args[0] is ...` -> False.
    `args or kwargs` -> `() or {'x': 1}` -> `{'x': 1}` -> truthy.
    Call: v(*(), **{'x': 1}) -> v(x=1).
    """
    result = _resolve_value(lambda x=0: x, None, {'x': 7})
    self.assertEqual(result, 7)


# ---------------------------------------------------------------------------
# TestEvaluateValueAsync
# ---------------------------------------------------------------------------

class TestEvaluateValueAsync(unittest.IsolatedAsyncioTestCase):
  """Async-specific _evaluate_value tests."""

  async def test_async_fn_returns_coroutine(self):
    """_evaluate_value with an async fn returns an awaitable."""
    async def afn(x):
      return x + 10

    link = Link(afn)
    result = _evaluate_value(link, 5)
    self.assertTrue(inspect.isawaitable(result))
    self.assertEqual(await result, 15)

  async def test_async_callable_obj_returns_coroutine(self):
    obj = AsyncCallableObj()
    link = Link(obj)
    result = _evaluate_value(link, 5)
    self.assertTrue(inspect.isawaitable(result))
    self.assertEqual(await result, 6)

  async def test_async_fn_with_ellipsis(self):
    async def afn():
      return 'async_ell'

    link = Link(afn, (...,))
    result = _evaluate_value(link, 'ignored')
    self.assertTrue(inspect.isawaitable(result))
    self.assertEqual(await result, 'async_ell')

  async def test_async_fn_with_explicit_args(self):
    async def afn(a, b):
      return a * b

    link = Link(afn, (3, 7))
    result = _evaluate_value(link, 'ignored')
    self.assertTrue(inspect.isawaitable(result))
    self.assertEqual(await result, 21)

  async def test_async_fn_with_kwargs(self):
    async def afn(k=0):
      return k + 1

    link = Link(afn, None, {'k': 41})
    result = _evaluate_value(link, 'ignored')
    self.assertTrue(inspect.isawaitable(result))
    self.assertEqual(await result, 42)

  async def test_async_fn_null_current_value(self):
    """current_value=Null -> async fn called with no args."""
    async def afn():
      return 'null_cv'

    link = Link(afn)
    result = _evaluate_value(link, Null)
    self.assertTrue(inspect.isawaitable(result))
    self.assertEqual(await result, 'null_cv')


# ---------------------------------------------------------------------------
# Beyond-spec: additional edge-case tests
# ---------------------------------------------------------------------------

class TestEvaluateValueEdgeCases(unittest.TestCase):
  """Additional edge cases beyond the spec."""

  def test_v_is_none_non_callable(self):
    """v=None with no args -> None is not callable, returned as-is."""
    link = Link(None)
    result = _evaluate_value(link, 'cv')
    self.assertIsNone(result)

  def test_v_is_none_with_args_raises(self):
    """v=None with args -> None(1) raises TypeError."""
    link = Link(None, (1,))
    with self.assertRaises(TypeError):
      _evaluate_value(link, 'cv')

  def test_args_tuple_with_none_element(self):
    """args=(None,) is a valid non-empty tuple -- different from args=None."""
    tracker = make_tracker()
    link = Link(tracker, (None,))
    _evaluate_value(link, 'cv')
    self.assertEqual(tracker.calls, [((None,), {})])

  def test_args_none_vs_args_empty_tuple(self):
    """args=None and args=() both lead to 'no explicit args' path for non-chain."""
    fn = lambda: 'called'
    link_none = Link(fn, None)
    link_empty = Link(fn, ())
    self.assertEqual(_evaluate_value(link_none, Null), 'called')
    self.assertEqual(_evaluate_value(link_empty, Null), 'called')

  def test_mutable_args_not_modified(self):
    """Ensure _evaluate_value does not mutate the args tuple."""
    args = (1, 2, 3)
    original = args
    tracker = make_tracker()
    link = Link(tracker, args)
    _evaluate_value(link, 'cv')
    self.assertIs(link.args, original)
    self.assertEqual(link.args, (1, 2, 3))

  def test_mutable_kwargs_not_modified(self):
    """Ensure _evaluate_value does not mutate the kwargs dict."""
    kwargs = {'a': 1, 'b': 2}
    original_copy = dict(kwargs)
    tracker = make_tracker()
    link = Link(tracker, None, kwargs)
    _evaluate_value(link, 'cv')
    self.assertEqual(kwargs, original_copy)

  def test_large_args_tuple(self):
    """100+ element args tuple."""
    big_args = tuple(range(150))
    fn = lambda *args: sum(args)
    link = Link(fn, big_args)
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(result, sum(range(150)))

  def test_kwargs_with_special_keys(self):
    """kwargs with keys like 'self', 'cls', 'args', 'kwargs'."""
    fn = lambda **kw: kw
    link = Link(fn, None, {'self': 1, 'cls': 2, 'args': 3, 'kwargs': 4})
    result = _evaluate_value(link, 'ignored')
    self.assertEqual(result, {'self': 1, 'cls': 2, 'args': 3, 'kwargs': 4})

  def test_nested_chain_within_nested_chain(self):
    """Chain containing a chain containing a chain.

    When _evaluate_value(outer_link, 'cv') is called:
    - middle._run('cv', None, None): has_run_value=True,
      creates Link('cv') spliced before first_link=Link(innermost).
      Link('cv') -> 'cv', then Link(innermost) -> innermost._run('cv', None, None).
    - innermost._run('cv', None, None): has_run_value=True,
      creates Link('cv') spliced before first_link=None.
      root_link was Link(lambda: 10) but it's bypassed by the run value.
      Link('cv') -> 'cv'. Done.
    - Result: 'cv'.

    To make the innermost chain produce 10 regardless, use ellipsis.
    """
    innermost = Chain(lambda: 10)
    middle = Chain()
    middle._then(Link(innermost, (...,)))
    outer_link = Link(middle)
    result = _evaluate_value(outer_link, 'cv')
    # middle._run('cv'): Link('cv') -> 'cv',
    # then Link(innermost, (...,)) -> innermost._run(Null, None, None)
    # -> evaluates root Link(lambda: 10) -> 10
    self.assertEqual(result, 10)

  def test_evaluate_value_current_value_null_callable(self):
    """current_value=Null, callable fn -> fn() with no args."""
    fn = lambda: 'null_path'
    link = Link(fn)
    result = _evaluate_value(link, Null)
    self.assertEqual(result, 'null_path')

  def test_evaluate_value_current_value_null_non_callable(self):
    """current_value=Null, non-callable -> returned as-is."""
    link = Link(42)
    result = _evaluate_value(link, Null)
    self.assertEqual(result, 42)

  def test_evaluate_value_current_value_none_vs_null(self):
    """current_value=None is not Null -- callable receives None."""
    tracker = make_tracker()
    link = Link(tracker)
    _evaluate_value(link, None)
    self.assertEqual(tracker.calls, [((None,), {})])

  def test_callable_with_default_receives_current_value(self):
    """When current_value is provided, it overrides default parameters."""
    fn = lambda x='default': x
    link = Link(fn)
    result = _evaluate_value(link, 'override')
    self.assertEqual(result, 'override')

  def test_callable_object_with_call_none(self):
    """Object where __call__ is set to None.

    On Python 3.14, callable() returns True for such objects (the slot is
    filled by the type machinery even though the value is None). So
    _evaluate_value enters the callable branch and v(current_value) raises
    TypeError because NoneType is not callable.
    """

    class BadCallable:
      __call__ = None

    obj = BadCallable()
    link = Link(obj)
    if callable(obj):
      # Python 3.14+: callable() returns True, calling raises TypeError
      with self.assertRaises(TypeError):
        _evaluate_value(link, 'cv')
    else:
      # Older Pythons: callable() returns False, returned as-is
      result = _evaluate_value(link, 'cv')
      self.assertIs(result, obj)

  def test_ellipsis_with_non_callable_raises(self):
    """Ellipsis on a non-callable -> 42() raises TypeError."""
    link = Link(42, (...,))
    with self.assertRaises(TypeError):
      _evaluate_value(link, 'cv')

  def test_nested_chain_ellipsis_ignores_current_value(self):
    """Nested chain with ellipsis passes Null, not current_value."""
    tracker = []

    def capture_fn(x=Null):
      tracker.append(x)
      return x

    inner = Chain(capture_fn, ...)
    link = Link(inner, (...,))
    _evaluate_value(link, 'should_be_ignored')
    # Inner chain's root is capture_fn(...) -> capture_fn() -> x defaults to Null
    self.assertIs(tracker[0], Null)

  def test_class_constructor_as_callable(self):
    """Class constructors are callable -- Adder(current_value) works."""
    link = Link(Adder)
    result = _evaluate_value(link, 99)
    self.assertIsInstance(result, Adder)
    self.assertEqual(result.x, 99)

  def test_partial_function_with_current_value(self):
    """functools.partial is callable."""
    p = functools.partial(operator.add, 100)
    link = Link(p)
    result = _evaluate_value(link, 5)
    self.assertEqual(result, 105)

  def test_bound_method_with_current_value(self):
    holder = BoundMethodHolder()
    link = Link(holder.method)
    result = _evaluate_value(link, 10)
    self.assertEqual(result, 11)

  def test_staticmethod_fn(self):
    """Static methods are plain functions at runtime."""

    class Cls:
      @staticmethod
      def sfn(x):
        return x * 2

    link = Link(Cls.sfn)
    result = _evaluate_value(link, 5)
    self.assertEqual(result, 10)

  def test_classmethod_fn(self):
    """Bound classmethods are callable."""

    class Cls:
      val = 100

      @classmethod
      def cfn(cls, x):
        return cls.val + x

    link = Link(Cls.cfn)
    result = _evaluate_value(link, 5)
    self.assertEqual(result, 105)

  def test_generator_function_callable(self):
    """Generator functions are callable -- returns a generator object."""

    def gen(x):
      yield x

    link = Link(gen)
    result = _evaluate_value(link, 3)
    self.assertEqual(list(result), [3])

  def test_non_callable_with_kwargs_raises(self):
    """Non-callable with kwargs -> v(**kwargs) raises TypeError."""
    link = Link(42, None, {'k': 1})
    with self.assertRaises(TypeError):
      _evaluate_value(link, 'cv')

  def test_non_callable_with_args_and_kwargs_raises(self):
    """Non-callable with args+kwargs -> v(*args, **kwargs) raises TypeError."""
    link = Link('hello', (1,), {'k': 2})
    with self.assertRaises(TypeError):
      _evaluate_value(link, 'cv')

  def test_evaluate_value_default_current_value_is_null(self):
    """_evaluate_value with no current_value arg defaults to Null."""
    fn = lambda: 'default_null'
    link = Link(fn)
    # Calling without current_value should use default=Null
    result = _evaluate_value(link)
    self.assertEqual(result, 'default_null')

  def test_evaluate_value_non_callable_default_current_value(self):
    """Non-callable with default Null current_value returns v."""
    link = Link(99)
    result = _evaluate_value(link)
    self.assertEqual(result, 99)

  def test_nested_chain_current_value_propagates(self):
    """When no args on nested chain, outer current_value flows in."""
    values = []

    def capture(x):
      values.append(x)
      return x

    inner = Chain()
    inner._then(Link(capture))
    link = Link(inner)
    _evaluate_value(link, 'propagated')
    self.assertEqual(values, ['propagated'])

  def test_chain_with_multiple_args_slicing(self):
    """args=(fn, 6, 7) -> v._run(fn, (6, 7), {}).

    The first arg becomes the run value. Remaining args become its positional
    args. So the run value must be callable.
    """
    values = []

    def capture(*args, **kwargs):
      values.append(args)
      return sum(args)

    inner = Chain()
    inner._then(Link(lambda x: x + 1))
    link = Link(inner, (capture, 6, 7))
    result = _evaluate_value(link, 'ignored')
    # _run(capture, (6, 7), {}) -> Link(capture, (6, 7), {})
    # _evaluate_value: capture(6, 7) -> 13
    # then lambda(13) -> 14
    self.assertEqual(result, 14)
    self.assertEqual(values, [(6, 7)])


class TestResolveValueEdgeCases(unittest.TestCase):
  """Additional edge cases for _resolve_value."""

  def test_ellipsis_not_first_is_regular_arg(self):
    """Ellipsis not at position 0 is passed as a regular argument."""
    fn = lambda a, b: (a, b)
    result = _resolve_value(fn, (1, ...), None)
    self.assertEqual(result, (1, ...))

  def test_callable_with_ellipsis_and_kwargs_raises(self):
    """_resolve_value with ellipsis in args and kwargs raises QuentException."""
    from quent import QuentException

    def fn(*args, **kwargs):
      return 'done'

    with self.assertRaises(QuentException):
      _resolve_value(fn, (...,), {'should': 'be_ignored'})

  def test_non_callable_with_empty_args_and_kwargs(self):
    """Non-callable with () and {} -> treated as no args -> returned as-is."""
    result = _resolve_value('literal', (), {})
    self.assertEqual(result, 'literal')

  def test_resolve_value_with_null_as_value(self):
    """Null is not callable -> returned as-is."""
    result = _resolve_value(Null, None, None)
    self.assertIs(result, Null)

  def test_resolve_value_callable_with_empty_tuple_arg(self):
    """args=((),) means one argument: the empty tuple."""
    fn = lambda x: x
    result = _resolve_value(fn, ((),), None)
    self.assertEqual(result, ())


if __name__ == '__main__':
  unittest.main()
