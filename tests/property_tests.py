# SPDX-License-Identifier: MIT
"""Property-based tests using Hypothesis for quent pipeline behavior.

Tests the calling convention rules, control flow signals, pipeline value flow,
error handling, iteration, gather, conditionals, cloning, context managers,
decorator pattern, and bridge invariants using randomly generated inputs.

All assertions are derived from SPEC.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase

from hypothesis import given, settings
from hypothesis import strategies as st

from quent import Chain, QuentException
from tests.tests_helper import (
  async_double,
)


def load_tests(loader, tests, pattern):
  if not os.environ.get('QUENT_SLOW'):
    return unittest.TestSuite()
  return tests


# --- Shared strategies ---

values = st.one_of(
  st.integers(min_value=-1000, max_value=1000),
  st.none(),
  st.text(max_size=10),
  st.floats(allow_nan=False, allow_infinity=False),
)

small_ints = st.integers(min_value=-100, max_value=100)

positive_ints = st.integers(min_value=1, max_value=100)

non_zero_ints = st.integers(min_value=1, max_value=100)

int_lists = st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=30)

non_empty_int_lists = st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=30)

short_int_lists = st.lists(st.integers(min_value=-50, max_value=50), min_size=0, max_size=10)

# Strategy for executor parameter: None (quent creates its own) or 'user' (shared instance).
executor_strategy = st.sampled_from([None, 'user'])


# --- Helpers ---


class _ExecutorMixin:
  """Mixin that manages a shared ThreadPoolExecutor for property tests.

  Tests using the executor_strategy should inherit from this mixin.
  When the drawn value is 'user', ``resolve_executor`` returns the shared
  instance; when ``None``, it returns ``None`` (quent creates its own).
  """

  def setUp(self):
    super().setUp()
    self._user_executor = ThreadPoolExecutor(max_workers=4)

  def tearDown(self):
    self._user_executor.shutdown(wait=True)
    super().tearDown()

  def resolve_executor(self, executor_choice):
    """Map strategy value to an actual executor or None."""
    if executor_choice == 'user':
      return self._user_executor
    return None


def _run_async(coro):
  """Run a coroutine in a new event loop."""
  return asyncio.run(coro)


class _TrackedSideEffect:
  """Side-effect tracker for verifying do() discards results."""

  def __init__(self):
    self.calls = []

  def __call__(self, x):
    self.calls.append(x)
    return 'should_be_discarded'


# ============================================================
# 1. Calling Convention Dispatch
# ============================================================


class CallingConventionDispatchTest(TestCase):
  """Property-based tests for SPEC section 4 calling convention rules."""

  @given(value=values)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule3_default_callable_with_current_value(self, value):
    """SPEC section 4 Rule 3: fn(current_value) when callable has no explicit args and cv exists."""
    result = Chain(value).then(lambda x: x).run()
    self.assertEqual(result, value)

  @given(value=values)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule3_default_callable_no_current_value(self, value):
    """SPEC section 4 Rule 3: fn() when callable has no explicit args and no cv."""
    result = Chain().then(lambda: value).run()
    self.assertEqual(result, value)

  @given(old=values, new=values)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule3_default_non_callable_replaces_value(self, old, new):
    """SPEC section 4 Rule 3: non-callable value replaces current value as-is."""
    result = Chain(old).then(new).run()
    self.assertEqual(result, new)

  @given(value=values, arg=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule2_explicit_args_suppress_current_value(self, value, arg):
    """SPEC section 4 Rule 2: explicit args suppress current value, fn(*args)."""
    result = Chain(value).then(lambda a: a, arg).run()
    self.assertEqual(result, arg)

  @given(a=small_ints, b=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule2_explicit_kwargs_suppress_current_value(self, a, b):
    """SPEC section 4 Rule 2: explicit kwargs suppress current value."""
    result = Chain(a).then(lambda x=0: x, x=b).run()
    self.assertEqual(result, b)

  @given(a=small_ints, b=small_ints, c=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule2_explicit_args_and_kwargs(self, a, b, c):
    """SPEC section 4 Rule 2: explicit args + kwargs, cv NOT passed."""
    result = Chain(a).then(lambda x, y=0: x + y, b, y=c).run()
    self.assertEqual(result, b + c)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule1_nested_chain_receives_current_value(self, value):
    """SPEC section 4 Rule 1: nested Chain runs with current_value as input."""
    inner = Chain().then(lambda x: x + 1)
    result = Chain(value).then(inner).run()
    self.assertEqual(result, value + 1)

  @given(value=small_ints, arg=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule1_nested_chain_with_explicit_args(self, value, arg):
    """SPEC section 4 Rule 1: nested chain with explicit args; first arg becomes input."""
    inner = Chain().then(lambda x: x * 2)
    result = Chain(value).then(inner, arg).run()
    self.assertEqual(result, arg * 2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule1_priority_over_rule3(self, value):
    """SPEC section 4: Rule 1 (nested chain) takes priority over Rule 3 (default)."""
    inner = Chain().then(lambda x: x + 10)
    result = Chain(value).then(inner).run()
    self.assertEqual(result, value + 10)

  @given(value=small_ints, arg=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_rule2_priority_over_rule3(self, value, arg):
    """SPEC section 4: Rule 2 (explicit args) takes priority over Rule 3 (default)."""
    result = Chain(value).then(lambda x: x * 3, arg).run()
    self.assertEqual(result, arg * 3)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_do_requires_callable(self, value):
    """SPEC section 3: do() requires callable, raises TypeError for non-callable."""
    with self.assertRaises(TypeError):
      Chain(value).do(42)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_do_discards_result(self, value):
    """SPEC section 3: do() discards fn return value, preserves current value."""
    result = Chain(value).do(lambda x: x * 999).run()
    self.assertEqual(result, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_non_callable_with_args_raises_type_error(self, value):
    """SPEC section 4 Rule 2: non-callable with explicit args raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(value).then(42, 'some_arg')

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_nested_chain_with_positional_and_kwargs(self, value):
    """SPEC section 4 Rule 1: nested chain with positional arg; first arg becomes input."""
    inner = Chain().then(lambda x: x + 10)
    result = Chain(value).then(inner, value * 2).run()
    self.assertEqual(result, value * 2 + 10)


# ============================================================
# 2. Except Handler Calling Convention
# ============================================================


class ExceptHandlerConventionTest(TestCase):
  """Property-based tests for SPEC section 6.2.2 except handler calling convention."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_default_receives_exc_info(self, value):
    """SPEC section 6.2.2: default handler(exc_info) receives exception info."""
    captured = {}

    def handler(exc_info):
      captured['exc_info'] = exc_info
      return 'handled'

    result = Chain(value).then(lambda x: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertIn('exc_info', captured)
    self.assertIsInstance(captured['exc_info'].exc, ZeroDivisionError)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_explicit_args_suppress_exc(self, value):
    """SPEC section 6.2.2 Rule 2: explicit args suppress exc, handler(*args)."""
    result = Chain(value).then(lambda x: 1 / 0).except_(lambda a, b: a + b, 10, 20).run()
    self.assertEqual(result, 30)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_explicit_kwargs_suppress_exc(self, value):
    """SPEC section 6.2.2 Rule 2: explicit kwargs suppress exc."""
    result = Chain(value).then(lambda x: 1 / 0).except_(lambda x=0: x, x=42).run()
    self.assertEqual(result, 42)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_nested_chain_receives_exc_info(self, value):
    """SPEC section 6.2.2 Rule 1: nested chain handler runs with exc_info as input."""
    handler_chain = Chain().then(lambda exc_info: f'caught:{type(exc_info.exc).__name__}')
    result = Chain(value).then(lambda x: 1 / 0).except_(handler_chain).run()
    self.assertEqual(result, 'caught:ZeroDivisionError')


# ============================================================
# 3. Finally Handler Calling Convention
# ============================================================


class FinallyHandlerConventionTest(TestCase):
  """Property-based tests for SPEC section 6.3 finally handler."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_receives_root_value(self, value):
    """SPEC section 6.3.2: finally handler receives root value as current value."""
    captured = {}

    def cleanup(rv):
      captured['root'] = rv

    Chain(value).then(lambda x: x * 2).finally_(cleanup).run()
    self.assertEqual(captured['root'], value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_root_value_normalized_to_none(self, value):
    """SPEC section 6.3.2: root value normalized to None when chain has no root."""
    captured = {}

    def cleanup(rv):
      captured['root'] = rv

    Chain().then(lambda: value).finally_(cleanup).run()
    self.assertIsNone(captured['root'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_return_value_discarded(self, value):
    """SPEC section 6.3.2: finally handler return value is always discarded."""
    result = Chain(value).then(lambda x: x * 2).finally_(lambda rv: 'discard_me').run()
    self.assertEqual(result, value * 2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_with_explicit_args(self, value):
    """SPEC section 6.3: finally with explicit args; root value NOT passed."""
    captured = {}

    def cleanup(a, b):
      captured['args'] = (a, b)

    Chain(value).finally_(cleanup, 'x', 'y').run()
    self.assertEqual(captured['args'], ('x', 'y'))

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_always_runs_on_success(self, value):
    """SPEC section 6.3.2: finally always runs on success path."""
    ran = {'flag': False}

    def cleanup(rv):
      ran['flag'] = True

    Chain(value).then(lambda x: x + 1).finally_(cleanup).run()
    self.assertTrue(ran['flag'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_always_runs_on_failure(self, value):
    """SPEC section 6.3.2: finally always runs on failure path."""
    ran = {'flag': False}

    def cleanup(rv):
      ran['flag'] = True

    with self.assertRaises(ZeroDivisionError):
      Chain(value).then(lambda x: 1 / 0).finally_(cleanup).run()
    self.assertTrue(ran['flag'])


# ============================================================
# 4. If/Else Predicate Calling Convention
# ============================================================


class IfPredicateConventionTest(TestCase):
  """Property-based tests for SPEC section 5.8 if_() operation."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_none_predicate_uses_truthiness(self, value):
    """SPEC section 5.8: None predicate uses truthiness of current value."""
    result = Chain(value).if_().then(lambda x: x * 2).run()
    if value:
      self.assertEqual(result, value * 2)
    else:
      self.assertEqual(result, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_callable_predicate(self, value):
    """SPEC section 5.8: callable predicate invoked per calling convention."""
    result = Chain(value).if_(lambda x: x > 0).then(lambda x: x * 10).run()
    if value > 0:
      self.assertEqual(result, value * 10)
    else:
      self.assertEqual(result, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_literal_predicate_truthiness(self, value):
    """SPEC section 5.8: literal predicate uses its own truthiness directly."""
    result = Chain(value).if_(True).then(42).run()
    self.assertEqual(result, 42)

    result2 = Chain(value).if_(False).then(42).run()
    self.assertEqual(result2, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_then_else_truthy(self, value):
    """SPEC section 5.8/5.9: truthy predicate takes then branch."""
    result = Chain(value).if_(lambda x: True).then(lambda x: x + 1).else_(lambda x: x - 1).run()
    self.assertEqual(result, value + 1)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_then_else_falsy(self, value):
    """SPEC section 5.8/5.9: falsy predicate takes else branch."""
    result = Chain(value).if_(lambda x: False).then(lambda x: x + 1).else_(lambda x: x - 1).run()
    self.assertEqual(result, value - 1)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_falsy_no_else_passes_through(self, value):
    """SPEC section 5.8: falsy predicate with no else passes current value through."""
    result = Chain(value).if_(lambda x: False).then(999).run()
    self.assertEqual(result, value)

  @given(value=small_ints, flag_val=st.booleans())
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_predicate_with_explicit_args(self, value, flag_val):
    """SPEC section 5.8: predicate with explicit args; cv NOT passed."""
    result = Chain(value).if_(lambda f: f, flag_val).then(lambda x: x * 2).run()
    if flag_val:
      self.assertEqual(result, value * 2)
    else:
      self.assertEqual(result, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_if_do_branch_discards_result(self, value):
    """SPEC section 5.8: do() branch discards result, passes cv through."""
    side_effects = []
    result = Chain(value).if_(lambda x: x > 0).do(lambda x: side_effects.append(x)).run()
    self.assertEqual(result, value)
    if value > 0:
      self.assertEqual(side_effects, [value])
    else:
      self.assertEqual(side_effects, [])


# ============================================================
# 5. Control Flow Signals
# ============================================================


class ControlFlowSignalsTest(TestCase):
  """Property-based tests for SPEC section 7 control flow signals."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_return_exits_chain_skips_remaining_steps(self, value):
    """SPEC section 7.2: return_() exits chain, skips remaining steps."""
    result = Chain(value).then(lambda x: Chain.return_(x * 10)).then(str).run()
    self.assertEqual(result, value * 10)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_return_with_non_callable_value(self, value):
    """SPEC section 7.2.1: return_(v) with non-callable returns v as-is."""
    result = Chain(value).then(lambda x: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_return_with_callable_value(self, value):
    """SPEC section 7.2.1: return_(fn) callable invoked when signal caught."""
    result = Chain(value).then(lambda x: Chain.return_(lambda: x * 3)).run()
    self.assertEqual(result, value * 3)

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(value=small_ints)
  def test_return_no_value_returns_none(self, value):
    """SPEC section 7.2.1: return_() with no value returns None."""
    result = Chain(value).then(lambda x: Chain.return_()).run()
    self.assertIsNone(result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_return_propagates_through_nested_chains(self, value):
    """SPEC section 7.2.2: return_() propagates from nested to outermost chain."""
    inner = Chain().then(lambda x: Chain.return_('early'))
    result = Chain(value).then(inner).then(lambda x: 'never_reached').run()
    self.assertEqual(result, 'early')

  @given(n=st.integers(min_value=2, max_value=20))
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_break_stops_iteration(self, n):
    """SPEC section 7.3: break_() stops foreach iteration."""
    # Use a range to guarantee unique values and predictable break point
    items = list(range(n))
    break_at = 1  # Break at index 1
    result = Chain(items).foreach(lambda x, _ba=break_at: Chain.break_() if x == _ba else x * 2).run()
    # Break with no value keeps partial results collected so far
    self.assertEqual(result, [0])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_break_with_value_appends_to_partial_results(self, value):
    """SPEC section 7.3.1: break_(value) appends to results collected so far."""
    result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_(value) if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4, value])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_break_without_value_keeps_partial_results(self, value):
    """SPEC section 7.3.1: break_() without value keeps partial results."""
    result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_() if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_break_outside_iteration_raises(self, value):
    """SPEC section 7.3.2: break_() outside iteration raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).then(lambda x: Chain.break_()).run()

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_return_in_except_handler_raises(self, value):
    """SPEC section 7.2.3: return_() in except handler raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).then(lambda x: 1 / 0).except_(lambda ei: Chain.return_('bad')).run()

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_break_in_except_handler_raises(self, value):
    """SPEC section 7.3.3: break_() in except handler raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).then(lambda x: 1 / 0).except_(lambda ei: Chain.break_()).run()

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_return_in_finally_handler_raises(self, value):
    """SPEC section 6.3.4: return_() in finally handler raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).finally_(lambda rv: Chain.return_('bad')).run()

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_break_in_finally_handler_raises(self, value):
    """SPEC section 6.3.4: break_() in finally handler raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).finally_(lambda rv: Chain.break_()).run()


# ============================================================
# 6. Error Handler Composition
# ============================================================


class ErrorHandlerCompositionTest(TestCase):
  """Property-based tests for SPEC section 6 error handling."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_consumes_exception_reraise_false(self, value):
    """SPEC section 6.2.3: reraise=False, handler result becomes output."""
    result = Chain(value).then(lambda x: 1 / 0).except_(lambda ei: 'recovered').run()
    self.assertEqual(result, 'recovered')

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_reraises_with_reraise_true(self, value):
    """SPEC section 6.2.3: reraise=True, handler runs for side-effects, original re-raised."""
    side_effects = []

    def handler(ei):
      side_effects.append('called')

    with self.assertRaises(ZeroDivisionError):
      Chain(value).then(lambda x: 1 / 0).except_(handler, reraise=True).run()
    self.assertEqual(side_effects, ['called'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_non_matching_exception_propagates(self, value):
    """SPEC section 6.2.1: non-matching exception type propagates."""
    with self.assertRaises(ZeroDivisionError):
      Chain(value).then(lambda x: 1 / 0).except_(lambda ei: 'handled', exceptions=ValueError).run()

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_handler_failure_chains_exception(self, value):
    """SPEC section 6.2.5: handler failure with reraise=False chains exceptions."""

    def bad_handler(ei):
      raise RuntimeError('handler_failed')

    with self.assertRaises(RuntimeError) as ctx:
      Chain(value).then(lambda x: 1 / 0).except_(bad_handler).run()
    self.assertIsInstance(ctx.exception.__cause__, ZeroDivisionError)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_reraise_true_handler_failure_discarded(self, value):
    """SPEC section 6.2.4: reraise=True + handler Exception, handler's exc discarded."""
    import warnings

    def bad_handler(ei):
      raise RuntimeError('handler_boom')

    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      with self.assertRaises(ZeroDivisionError):
        Chain(value).then(lambda x: 1 / 0).except_(bad_handler, reraise=True).run()

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_failure_replaces_active_exception(self, value):
    """SPEC section 6.3.3: finally exception replaces active exception."""

    def bad_finally(rv):
      raise RuntimeError('finally_boom')

    with self.assertRaises(RuntimeError) as ctx:
      Chain(value).then(lambda x: 1 / 0).finally_(bad_finally).run()
    # Original exception preserved in __context__
    self.assertIsInstance(ctx.exception.__context__, ZeroDivisionError)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_and_finally_interaction_success(self, value):
    """SPEC section 6.4: except + finally interaction on success path."""
    finally_ran = {'flag': False}

    result = (
      Chain(value)
      .then(lambda x: x * 2)
      .except_(lambda ei: 'should_not_run')
      .finally_(lambda rv: finally_ran.__setitem__('flag', True))
      .run()
    )
    self.assertEqual(result, value * 2)
    self.assertTrue(finally_ran['flag'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_and_finally_on_failure_path(self, value):
    """SPEC section 6.4: except + finally on failure, handler consumes exception."""
    finally_ran = {'flag': False}

    result = (
      Chain(value)
      .then(lambda x: 1 / 0)
      .except_(lambda ei: 'recovered')
      .finally_(lambda rv: finally_ran.__setitem__('flag', True))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertTrue(finally_ran['flag'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_single_except_per_chain(self, value):
    """SPEC section 6.2: at most one except_() per chain."""
    with self.assertRaises(QuentException):
      Chain(value).except_(lambda ei: None).except_(lambda ei: None)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_single_finally_per_chain(self, value):
    """SPEC section 6.3: at most one finally_() per chain."""
    with self.assertRaises(QuentException):
      Chain(value).finally_(lambda rv: None).finally_(lambda rv: None)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_with_various_exception_types(self, value):
    """SPEC section 6.2.1: exceptions parameter filters by type."""

    def raise_value_error(x):
      raise ValueError('test')

    def raise_type_error(x):
      raise TypeError('test')

    # Should catch ValueError
    result = Chain(value).then(raise_value_error).except_(lambda ei: 'caught_value_error', exceptions=ValueError).run()
    self.assertEqual(result, 'caught_value_error')

    # Should NOT catch TypeError when expecting ValueError
    with self.assertRaises(TypeError):
      Chain(value).then(raise_type_error).except_(lambda ei: 'caught', exceptions=ValueError).run()


# ============================================================
# 7. Clone Invariant
# ============================================================


class CloneInvariantTest(TestCase):
  """Property-based tests for SPEC section 10.1 clone()."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_clone_produces_same_result(self, value):
    """SPEC section 10.1: clone produces same result as original."""
    original = Chain(value).then(lambda x: x * 2).then(lambda x: x + 1)
    clone = original.clone()
    self.assertEqual(original.run(), clone.run())

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_clone_independence_extend_clone(self, value):
    """SPEC section 10.1: extending clone doesn't affect original."""
    original = Chain(value).then(lambda x: x * 2)
    clone = original.clone()
    clone.then(lambda x: x + 100)
    # Original should be unchanged
    self.assertEqual(original.run(), value * 2)
    self.assertEqual(clone.run(), value * 2 + 100)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_clone_independence_extend_original(self, value):
    """SPEC section 10.1: extending original doesn't affect clone."""
    original = Chain(value).then(lambda x: x * 2)
    clone = original.clone()
    original.then(lambda x: x + 100)
    self.assertEqual(clone.run(), value * 2)
    self.assertEqual(original.run(), value * 2 + 100)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_multiple_clones_independent(self, value):
    """SPEC section 10.1: multiple clones are independent of each other."""
    base = Chain(value).then(lambda x: x * 2)
    clone1 = base.clone()
    clone2 = base.clone()
    clone1.then(lambda x: x + 1)
    clone2.then(lambda x: x + 2)
    self.assertEqual(clone1.run(), value * 2 + 1)
    self.assertEqual(clone2.run(), value * 2 + 2)
    self.assertEqual(base.run(), value * 2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_clone_with_nested_chains(self, value):
    """SPEC section 10.1: clone with nested chains are recursively cloned."""
    inner = Chain().then(lambda x: x + 5)
    original = Chain(value).then(inner)
    clone = original.clone()
    self.assertEqual(original.run(), clone.run())
    self.assertEqual(clone.run(), value + 5)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_clone_with_except_handler(self, value):
    """SPEC section 10.1: clone with except handler is independent."""
    original = Chain(value).then(lambda x: 1 / 0).except_(lambda ei: 'err_original')
    clone = original.clone()
    self.assertEqual(original.run(), clone.run())
    self.assertEqual(clone.run(), 'err_original')

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_clone_with_finally_handler(self, value):
    """SPEC section 10.1: clone with finally handler is independent."""
    original_ran = {'flag': False}

    original = Chain(value).then(lambda x: x * 2).finally_(lambda rv: original_ran.__setitem__('flag', True))
    _clone = original.clone()  # clone exists to verify independence
    # Clone's finally will call the same function (shared by reference)
    self.assertEqual(original.run(), value * 2)
    self.assertTrue(original_ran['flag'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_clone_with_if_else(self, value):
    """SPEC section 10.1: clone with if/else is independent."""
    original = Chain(value).if_(lambda x: x > 0).then(lambda x: x * 2).else_(lambda x: x * -1)
    clone = original.clone()
    self.assertEqual(original.run(), clone.run())


# ============================================================
# 8. Iteration Operations
# ============================================================


class IterationOperationsTest(_ExecutorMixin, TestCase):
  """Property-based tests for SPEC section 5.3-5.4 iteration operations."""

  @given(items=int_lists)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_preserves_length(self, items):
    """SPEC section 5.3: foreach output has same length as input."""
    result = Chain(items).foreach(lambda x: x + 1).run()
    self.assertEqual(len(result), len(items))

  @given(items=int_lists)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_applies_function(self, items):
    """SPEC section 5.3: foreach applies fn to every element."""
    result = Chain(items).foreach(lambda x: x * 2).run()
    expected = [x * 2 for x in items]
    self.assertEqual(result, expected)

  @given(items=int_lists)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_preserves_order(self, items):
    """SPEC section 5.3: foreach preserves input order."""
    result = Chain(items).foreach(lambda x: x).run()
    self.assertEqual(result, items)

  @given(items=int_lists)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_do_preserves_originals(self, items):
    """SPEC section 5.4: foreach_do returns original elements, not fn results."""
    result = Chain(items).foreach_do(lambda x: x * 999).run()
    self.assertEqual(result, items)

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_foreach_empty_list(self, data):
    """SPEC section 5.3: foreach on empty list returns []."""
    result = Chain([]).foreach(lambda x: x + 1).run()
    self.assertEqual(result, [])

  @given(items=short_int_lists, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_with_concurrency_preserves_order(self, items, executor_choice):
    """SPEC section 5.3: foreach with concurrency preserves order."""
    if not items:
      return
    executor = self.resolve_executor(executor_choice)
    result = Chain(items).foreach(lambda x: x * 2, concurrency=3, executor=executor).run()
    expected = [x * 2 for x in items]
    self.assertEqual(result, expected)

  @given(items=short_int_lists, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_concurrent_same_as_sequential(self, items, executor_choice):
    """SPEC section 5.3: concurrent foreach produces same result as sequential."""
    if not items:
      return
    executor = self.resolve_executor(executor_choice)
    sequential = Chain(items).foreach(lambda x: x * 2).run()
    concurrent = Chain(items).foreach(lambda x: x * 2, concurrency=4, executor=executor).run()
    self.assertEqual(sequential, concurrent)

  @given(items=short_int_lists, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_do_with_concurrency(self, items, executor_choice):
    """SPEC section 5.4: foreach_do with concurrency preserves originals."""
    if not items:
      return
    executor = self.resolve_executor(executor_choice)
    result = Chain(items).foreach_do(lambda x: x * 999, concurrency=3, executor=executor).run()
    self.assertEqual(result, items)

  @given(items=int_lists)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_result_is_list(self, items):
    """SPEC section 5.3: foreach result is always a list."""
    result = Chain(items).foreach(lambda x: x).run()
    self.assertIsInstance(result, list)

  @given(items=int_lists)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_do_result_is_list(self, items):
    """SPEC section 5.4: foreach_do result is always a list."""
    result = Chain(items).foreach_do(lambda x: x).run()
    self.assertIsInstance(result, list)


# ============================================================
# 9. Gather Operations
# ============================================================


class GatherOperationsTest(_ExecutorMixin, TestCase):
  """Property-based tests for SPEC section 5.5 gather."""

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_returns_tuple_in_fn_order(self, value, executor_choice):
    """SPEC section 5.5: gather returns results in same order as fns."""
    executor = self.resolve_executor(executor_choice)
    result = (
      Chain(value)
      .gather(
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x - 1,
        executor=executor,
      )
      .run()
    )
    self.assertEqual(result, (value + 1, value * 2, value - 1))

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_single_fn_returns_1_tuple(self, value, executor_choice):
    """SPEC section 5.5: gather with single fn returns 1-tuple."""
    executor = self.resolve_executor(executor_choice)
    result = Chain(value).gather(lambda x: x + 1, executor=executor).run()
    self.assertEqual(result, (value + 1,))

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_zero_fns_raises(self, value):
    """SPEC section 5.5: gather with zero fns raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).gather()

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_arity_matches_fn_count(self, value, executor_choice):
    """SPEC section 5.5: gather result tuple has one element per fn."""
    executor = self.resolve_executor(executor_choice)
    for n_fns in range(1, 5):
      fns = [lambda x, i=i: x + i for i in range(n_fns)]
      result = Chain(value).gather(*fns, executor=executor).run()
      self.assertIsInstance(result, tuple)
      self.assertEqual(len(result), n_fns)

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_result_is_tuple(self, value, executor_choice):
    """SPEC section 5.5: gather always returns tuple type."""
    executor = self.resolve_executor(executor_choice)
    result = Chain(value).gather(lambda x: x, executor=executor).run()
    self.assertIsInstance(result, tuple)

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_with_concurrency_limit(self, value, executor_choice):
    """SPEC section 5.5: gather with concurrency limit produces same result."""
    executor = self.resolve_executor(executor_choice)
    result_unlimited = (
      Chain(value)
      .gather(
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x - 1,
        executor=executor,
      )
      .run()
    )
    result_limited = (
      Chain(value)
      .gather(
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x - 1,
        concurrency=2,
        executor=executor,
      )
      .run()
    )
    self.assertEqual(result_unlimited, result_limited)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_requires_callable_fns(self, value):
    """SPEC section 5.5: gather fns must be callable."""
    with self.assertRaises(TypeError):
      Chain(value).gather(42)

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_correct_values(self, value, executor_choice):
    """SPEC section 5.5: each fn receives current pipeline value."""
    executor = self.resolve_executor(executor_choice)
    result = (
      Chain(value)
      .gather(
        lambda x: x,
        lambda x: x,
        executor=executor,
      )
      .run()
    )
    self.assertEqual(result, (value, value))


# ============================================================
# 10. Pipeline Value Flow
# ============================================================


class PipelineValueFlowTest(TestCase):
  """Property-based tests for SPEC section 3 pipeline value flow."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_multiple_do_steps_dont_alter_value(self, value):
    """SPEC section 3: multiple do() steps don't alter pipeline value."""
    result = Chain(value).do(lambda x: x * 100).do(lambda x: x + 200).do(lambda x: x - 300).run()
    self.assertEqual(result, value)

  @given(old=small_ints, new=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_then_non_callable_replaces_value(self, old, new):
    """SPEC section 3: then(non-callable) replaces value regardless of state."""
    result = Chain(old).then(lambda x: x * 2).then(new).run()
    self.assertEqual(result, new)

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_empty_chain_returns_none(self, data):
    """SPEC section 3: empty chain with no root, no steps returns None."""
    result = Chain().run()
    self.assertIsNone(result)

  @given(value=values)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_v_run_for_non_callable(self, value):
    """SPEC section 3: Chain(v).run() for non-callable returns v."""
    result = Chain(value).run()
    self.assertEqual(result, value)

  @given(a=small_ints, b=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_run_value_overrides_root(self, a, b):
    """SPEC section 8.1: run value overrides root: Chain(A).run(B) uses B."""
    result = Chain(a).run(b)
    self.assertEqual(result, b)

  @given(value=values)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_is_always_truthy(self, value):
    """SPEC section 3: bool(Chain(...)) is always True."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(value)))
    self.assertTrue(bool(Chain(None)))
    self.assertTrue(bool(Chain(0)))
    self.assertTrue(bool(Chain(False)))

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_none_returns_none(self, value):
    """SPEC section 3: Chain(None).run() returns None."""
    result = Chain(None).run()
    self.assertIsNone(result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_only_do_steps_returns_none(self, value):
    """SPEC section 3: chain with only do() steps and no root returns None."""
    result = Chain().do(lambda: 42).run()
    self.assertIsNone(result)


# ============================================================
# 11. Arbitrary Pipeline Shapes + Bridge Invariant
# ============================================================


class ArbitraryPipelineShapesTest(TestCase):
  """Property-based tests for bridge invariant with random pipeline shapes."""

  @given(
    value=small_ints,
    steps=st.lists(
      st.sampled_from(['add1', 'mul2', 'sub3', 'identity']),
      min_size=1,
      max_size=6,
    ),
  )
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_sync_pipeline_deterministic(self, value, steps):
    """SPEC section 2: same pipeline + same input = same result."""
    ops = {
      'add1': lambda x: x + 1,
      'mul2': lambda x: x * 2,
      'sub3': lambda x: x - 3,
      'identity': lambda x: x,
    }
    chain1 = Chain(value)
    chain2 = Chain(value)
    for step in steps:
      chain1.then(ops[step])
      chain2.then(ops[step])
    self.assertEqual(chain1.run(), chain2.run())

  @given(
    value=small_ints,
    steps=st.lists(
      st.sampled_from(['then', 'do']),
      min_size=1,
      max_size=6,
    ),
  )
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_mixed_then_do_pipeline(self, value, steps):
    """SPEC sections 3, 5.1, 5.2: mixed then/do pipeline value flow."""
    chain = Chain(value)
    expected = value
    for step in steps:
      if step == 'then':
        chain.then(lambda x: x + 1)
        expected += 1
      else:
        chain.do(lambda x: x * 999)
        # do discards result, expected unchanged
    self.assertEqual(chain.run(), expected)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_bridge_invariant_sync_vs_async_simple(self, value):
    """SPEC section 2: sync callable and async equivalent produce same result."""
    sync_result = Chain(value).then(lambda x: x + 1).then(lambda x: x * 2).run()

    async def async_add1(x):
      return x + 1

    async_result = _run_async(Chain(value).then(async_add1).then(lambda x: x * 2).run())
    self.assertEqual(sync_result, async_result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_bridge_invariant_all_async(self, value):
    """SPEC section 2: all-async pipeline produces same result as all-sync."""
    sync_result = Chain(value).then(lambda x: x + 1).then(lambda x: x * 2).run()

    async def async_add1(x):
      return x + 1

    async def async_mul2(x):
      return x * 2

    async_result = _run_async(Chain(value).then(async_add1).then(async_mul2).run())
    self.assertEqual(sync_result, async_result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_bridge_invariant_async_at_start(self, value):
    """SPEC section 2: async at first step, sync remainder."""

    async def async_root():
      return value

    sync_result = Chain(value).then(lambda x: x * 3).run()
    async_result = _run_async(Chain(async_root).then(lambda x: x * 3).run())
    self.assertEqual(sync_result, async_result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_bridge_invariant_async_at_end(self, value):
    """SPEC section 2: sync pipeline, async at last step."""

    async def async_mul3(x):
      return x * 3

    sync_result = Chain(value).then(lambda x: x + 1).then(lambda x: x * 3).run()
    async_result = _run_async(Chain(value).then(lambda x: x + 1).then(async_mul3).run())
    self.assertEqual(sync_result, async_result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_pipeline_with_nested_chains(self, value):
    """SPEC section 4 Rule 1: nested chains compose equivalently to inline steps."""
    inner = Chain().then(lambda x: x + 1)
    nested_result = Chain(value).then(inner).then(lambda x: x * 2).run()
    flat_result = Chain(value).then(lambda x: x + 1).then(lambda x: x * 2).run()
    self.assertEqual(nested_result, flat_result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_pipeline_with_if_else_branches(self, value):
    """SPEC section 5.8: pipelines with conditional branches deterministic."""
    result1 = Chain(value).if_(lambda x: x > 0).then(lambda x: x * 2).else_(lambda x: abs(x)).run()
    result2 = Chain(value).if_(lambda x: x > 0).then(lambda x: x * 2).else_(lambda x: abs(x)).run()
    self.assertEqual(result1, result2)

  @given(value=small_ints, run_value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_run_value_overrides_root_with_steps(self, value, run_value):
    """SPEC section 8.1: run(v) overrides root, entire pipeline uses run_value."""
    result = Chain(value).then(lambda x: x + 1).run(run_value)
    self.assertEqual(result, run_value + 1)


# ============================================================
# 12. Decorator Pattern
# ============================================================


class DecoratorPatternTest(TestCase):
  """Property-based tests for SPEC section 10.2 decorator()."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_decorator_produces_consistent_results(self, value):
    """SPEC section 10.2: decorated function produces consistent results."""

    @Chain().then(lambda x: x * 2).decorator()
    def double(x):
      return x

    self.assertEqual(double(value), value * 2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_decorator_clones_chain_original_unaffected(self, value):
    """SPEC section 10.2: decorator clones the chain, original unaffected."""
    chain = Chain().then(lambda x: x * 2)

    @chain.decorator()
    def transform(x):
      return x

    # Extend original after decorator creation
    chain.then(lambda x: x + 100)

    # Decorator should NOT be affected by original's modification
    self.assertEqual(transform(value), value * 2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_decorator_multiple_calls_correct(self, value):
    """SPEC section 10.2: multiple calls to decorated function produce correct results."""

    @Chain().then(lambda x: x + 5).decorator()
    def add5(x):
      return x

    self.assertEqual(add5(value), value + 5)
    self.assertEqual(add5(value + 1), value + 6)
    self.assertEqual(add5(0), 5)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_decorator_preserves_function_name(self, value):
    """SPEC section 10.2: decorated function preserves original name via functools.wraps."""

    @Chain().then(lambda x: x).decorator()
    def my_func(x):
      return x

    self.assertEqual(my_func.__name__, 'my_func')


# ============================================================
# 13. Context Manager Operations
# ============================================================


class ContextManagerOperationsTest(TestCase):
  """Property-based tests for SPEC section 5.6-5.7 context manager operations."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_with_enters_and_exits_context(self, value):
    """SPEC section 5.6: with_() enters and exits context manager."""
    entered = {'flag': False}
    exited = {'flag': False}

    @contextlib.contextmanager
    def make_ctx():
      entered['flag'] = True
      yield value
      exited['flag'] = True

    # Use .then() to feed the context manager as a pipeline step result
    result = Chain(lambda: make_ctx()).with_(lambda x: x * 2).run()
    self.assertEqual(result, value * 2)
    self.assertTrue(entered['flag'])
    self.assertTrue(exited['flag'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_with_do_discards_result(self, value):
    """SPEC section 5.7: with_do() discards result, preserves pipeline value."""

    @contextlib.contextmanager
    def make_ctx():
      yield 'context_value'

    # The context manager itself is the pipeline value. with_do discards fn result.
    result = Chain(lambda: make_ctx()).with_do(lambda x: 'discarded').run()
    # with_do preserves the original pipeline value (the ctx manager object)
    # which is the _GeneratorContextManager instance
    self.assertIsNotNone(result)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_with_exception_cleanup(self, value):
    """SPEC section 5.6: context manager __exit__ is called even on exception."""
    exit_called = {'flag': False}

    class TrackingCtx:
      def __enter__(self):
        return value

      def __exit__(self, *args):
        exit_called['flag'] = True
        return False  # Don't suppress

    with self.assertRaises(ZeroDivisionError):
      Chain(TrackingCtx()).with_(lambda x: 1 / 0).run()
    self.assertTrue(exit_called['flag'])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_with_fn_result_replaces_value(self, value):
    """SPEC section 5.6: fn's return value replaces current pipeline value."""

    @contextlib.contextmanager
    def make_ctx():
      yield value

    result = Chain(lambda: make_ctx()).with_(lambda x: x + 100).then(lambda x: x * 2).run()
    self.assertEqual(result, (value + 100) * 2)


# ============================================================
# 14. Pipeline Idempotence
# ============================================================


class PipelineIdempotenceTest(TestCase):
  """Property-based tests for pipeline execution idempotence."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_same_chain_same_input_same_result(self, value):
    """SPEC section 8.1: same chain, same input produces same result."""
    chain = Chain().then(lambda x: x * 2).then(lambda x: x + 1)
    result1 = chain.run(value)
    result2 = chain.run(value)
    self.assertEqual(result1, result2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_with_run_value_idempotent(self, value):
    """SPEC section 8.1: chain with run value is idempotent across calls."""
    chain = Chain(value).then(lambda x: x * 3)
    results = [chain.run() for _ in range(5)]
    self.assertTrue(all(r == value * 3 for r in results))

  @given(value=small_ints, run_val=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_different_run_values_different_results(self, value, run_val):
    """SPEC section 8.1: different run values produce appropriate results."""
    chain = Chain(value).then(lambda x: x * 2)
    with_root = chain.run()
    with_run = chain.run(run_val)
    self.assertEqual(with_root, value * 2)
    self.assertEqual(with_run, run_val * 2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_concurrent_execution_produces_same_results(self, value):
    """SPEC section 3: fully constructed chain safe for concurrent execution."""
    chain = Chain().then(lambda x: x * 2).then(lambda x: x + 1)
    results = []
    errors = []

    def run_chain():
      try:
        results.append(chain.run(value))
      except Exception as e:
        errors.append(e)

    threads = [threading.Thread(target=run_chain) for _ in range(4)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(len(errors), 0, f'Errors: {errors}')
    self.assertTrue(all(r == value * 2 + 1 for r in results))


# ============================================================
# 15. Edge Cases
# ============================================================


class EdgeCasesTest(TestCase):
  """Property-based tests for edge cases and boundary conditions."""

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_chain_no_root_no_steps_returns_none(self, data):
    """SPEC section 3: Chain() with no root, no steps returns None."""
    self.assertIsNone(Chain().run())

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_chain_none_root_returns_none(self, data):
    """SPEC section 3/12: Chain(None).run() returns None."""
    self.assertIsNone(Chain(None).run())

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_deeply_nested_chains(self, value):
    """SPEC section 4: deeply nested chains (chain within chain within chain)."""
    inner3 = Chain().then(lambda x: x + 1)
    inner2 = Chain().then(inner3).then(lambda x: x + 1)
    inner1 = Chain().then(inner2).then(lambda x: x + 1)
    result = Chain(value).then(inner1).run()
    self.assertEqual(result, value + 3)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_run_with_callable_root(self, value):
    """SPEC section 3: Chain(callable).run() calls callable."""
    result = Chain(lambda: value).run()
    self.assertEqual(result, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_run_with_callable_root_and_args(self, value):
    """SPEC section 3: Chain(callable, arg).run() calls callable(arg)."""
    result = Chain(lambda x: x * 2, value).run()
    self.assertEqual(result, value * 2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_call_is_alias_for_run(self, value):
    """SPEC section 8.2: chain(v) is equivalent to chain.run(v)."""
    chain = Chain().then(lambda x: x * 2)
    self.assertEqual(chain.run(value), chain(value))

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_then_chain_composition_associativity(self, value):
    """SPEC section 4: chaining then() is associative."""
    r1 = Chain(value).then(lambda x: x + 1).then(lambda x: x * 2).run()
    inner = Chain().then(lambda x: x + 1)
    r2 = Chain(value).then(inner).then(lambda x: x * 2).run()
    self.assertEqual(r1, r2)

  @given(value=values)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_run_value_not_callable_with_args_raises(self, value):
    """SPEC section 8.1: run(non_callable, args) raises TypeError."""
    if callable(value):
      return  # Skip callable values
    with self.assertRaises(TypeError):
      Chain().run(value, 'extra_arg')

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_pending_if_without_then_raises_on_run(self, value):
    """SPEC section 5.8: pending if_() without then() raises QuentException on run()."""
    with self.assertRaises(QuentException):
      Chain(value).if_(lambda x: True).run()

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_else_without_if_raises(self, value):
    """SPEC section 5.9: else_() without preceding if_().then() raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).then(lambda x: x).else_(42)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_double_if_raises(self, value):
    """SPEC section 5.8: double if_() without then() in between raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).if_(lambda x: True).if_(lambda x: False)


# ============================================================
# 17. Null Sentinel Behavior
# ============================================================


class NullSentinelTest(TestCase):
  """Property-based tests for SPEC section 12 Null sentinel."""

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_null_never_exposed_to_user(self, data):
    """SPEC section 12.2: Null is never exposed to user code."""
    from quent._types import Null

    result = Chain().run()
    self.assertIsNone(result)
    self.assertIsNot(result, Null)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_no_value_callable_called_with_no_args(self, value):
    """SPEC section 12.3: Chain().then(fn).run() calls fn() with no args."""
    result = Chain().then(lambda: value).run()
    self.assertEqual(result, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_none_callable_called_with_none(self, value):
    """SPEC section 12.3: Chain(None).then(fn).run() calls fn(None)."""
    captured = {}

    def capture(x):
      captured['arg'] = x
      return x

    Chain(None).then(capture).run()
    self.assertIsNone(captured['arg'])

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_null_is_singleton(self, data):
    """SPEC section 12.4: Null is a singleton."""
    import copy

    from quent._types import Null

    self.assertIs(copy.copy(Null), Null)
    self.assertIs(copy.deepcopy(Null), Null)

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_null_repr(self, data):
    """SPEC section 12.4: repr(Null) returns '<Null>'."""
    from quent._types import Null

    self.assertEqual(repr(Null), '<Null>')

  @settings(max_examples=500, deadline=None, derandomize=True)
  @given(st.data())
  def test_null_unpicklable(self, data):
    """SPEC section 12.4: Null cannot be pickled."""
    import pickle

    from quent._types import Null

    with self.assertRaises(TypeError):
      pickle.dumps(Null)


# ============================================================
# 18. Unpickling Prevention
# ============================================================


class UnpicklePreventionTest(TestCase):
  """Property-based tests for CWE-502 prevention."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_unpicklable(self, value):
    """SPEC section 12: Chain objects cannot be pickled."""
    import pickle

    chain = Chain(value).then(lambda x: x)
    with self.assertRaises(TypeError):
      pickle.dumps(chain)


# ============================================================
# 19. Root Value Capture
# ============================================================


class RootValueCaptureTest(TestCase):
  """Property-based tests for SPEC section 8.1 root value capture."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_receives_root_not_current_value(self, value):
    """SPEC section 6.3.2: finally receives root value, NOT current pipeline value."""
    captured = {}

    def cleanup(rv):
      captured['root'] = rv

    Chain(value).then(lambda x: x * 100).then(lambda x: x + 999).finally_(cleanup).run()
    self.assertEqual(captured['root'], value)

  @given(value=small_ints, run_val=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_receives_run_value_when_overridden(self, value, run_val):
    """SPEC section 8.1: finally receives run value (which overrides root)."""
    captured = {}

    def cleanup(rv):
      captured['root'] = rv

    Chain(value).then(lambda x: x * 2).finally_(cleanup).run(run_val)
    self.assertEqual(captured['root'], run_val)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_root_callable_failure_finally_receives_none(self, value):
    """SPEC section 8.1: root callable failure, finally receives None (no root produced)."""
    captured = {}

    def cleanup(rv):
      captured['root'] = rv

    def bad_root():
      raise ValueError('root failed')

    with self.assertRaises(ValueError):
      Chain(bad_root).finally_(cleanup).run()
    self.assertIsNone(captured['root'])


# ============================================================
# 20. Concurrency Validation
# ============================================================


class ConcurrencyValidationTest(_ExecutorMixin, TestCase):
  """Property-based tests for SPEC section 11 concurrency parameter validation."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_concurrency_zero_raises_value_error(self, value):
    """SPEC section 11.2.1: concurrency < 1 raises ValueError."""
    with self.assertRaises(ValueError):
      Chain([1, 2, 3]).foreach(lambda x: x, concurrency=0)

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_concurrency_minus_one_is_valid_unbounded(self, value, executor_choice):
    """SPEC section 11.2.1: concurrency=-1 is valid (unbounded) — processes all items."""
    executor = self.resolve_executor(executor_choice)
    result = Chain([1, 2, 3]).foreach(lambda x: x, concurrency=-1, executor=executor).run()
    self.assertEqual(result, [1, 2, 3])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_concurrency_negative_other_raises_value_error(self, value):
    """SPEC section 11.2.1: concurrency < -1 (e.g. -2, -3) raises ValueError."""
    with self.assertRaises(ValueError):
      Chain([1, 2, 3]).foreach(lambda x: x, concurrency=-2)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_concurrency_boolean_raises_type_error(self, value):
    """SPEC section 11.2.1: boolean concurrency raises TypeError."""
    with self.assertRaises(TypeError):
      Chain([1, 2, 3]).foreach(lambda x: x, concurrency=True)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_concurrency_string_raises_type_error(self, value):
    """SPEC section 11.2.1: non-integer concurrency raises TypeError."""
    with self.assertRaises(TypeError):
      Chain([1, 2, 3]).foreach(lambda x: x, concurrency='5')

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_requires_callable(self, value):
    """SPEC section 5.3: foreach fn must be callable."""
    with self.assertRaises(TypeError):
      Chain([1, 2, 3]).foreach(42)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_do_requires_callable(self, value):
    """SPEC section 5.4: foreach_do fn must be callable."""
    with self.assertRaises(TypeError):
      Chain([1, 2, 3]).foreach_do(42)


# ============================================================
# 21. Chain Constructor
# ============================================================


class ChainConstructorTest(TestCase):
  """Property-based tests for SPEC section 3 chain construction."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_callable_root_called(self, value):
    """SPEC section 3: Chain(callable) calls callable at run time."""
    result = Chain(lambda: value * 2).run()
    self.assertEqual(result, value * 2)

  @given(value=small_ints, arg=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_callable_root_with_args(self, value, arg):
    """SPEC section 3: Chain(callable, *args) passes args at run time."""
    result = Chain(lambda x: x + 1, arg).run()
    self.assertEqual(result, arg + 1)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_non_callable_root_with_args_raises(self, value):
    """SPEC section 3: Chain(non_callable, args) raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(value, 'extra_arg')

  @given(value=values)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_non_callable_root_used_as_is(self, value):
    """SPEC section 3: Chain(non_callable) uses value as-is."""
    result = Chain(value).run()
    self.assertEqual(result, value)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_chain_callable_root_with_kwargs(self, value):
    """SPEC section 3: Chain(callable, **kwargs) passes kwargs at run time."""
    result = Chain(lambda x=0: x + 1, x=value).run()
    self.assertEqual(result, value + 1)


# ============================================================
# 22. Async Bridge for Iteration and Gather
# ============================================================


class AsyncBridgeIterationGatherTest(_ExecutorMixin, TestCase):
  """Property-based tests for bridge invariant in iteration and gather."""

  @given(items=short_int_lists, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_foreach_sync_vs_async_bridge(self, items, executor_choice):
    """SPEC section 2: foreach sync/async bridge invariant."""
    if not items:
      return
    executor = self.resolve_executor(executor_choice)
    sync_result = Chain(items).foreach(lambda x: x * 2, concurrency=-1, executor=executor).run()

    async_result = _run_async(Chain(items).foreach(async_double, concurrency=-1).run())
    self.assertEqual(sync_result, async_result)

  @given(value=small_ints, executor_choice=executor_strategy)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_gather_sync_vs_async_bridge(self, value, executor_choice):
    """SPEC section 2: gather sync/async bridge invariant."""
    executor = self.resolve_executor(executor_choice)
    sync_result = (
      Chain(value)
      .gather(
        lambda x: x + 1,
        lambda x: x * 2,
        executor=executor,
      )
      .run()
    )

    async def async_add1(x):
      return x + 1

    async_result = _run_async(
      Chain(value)
      .gather(
        async_add1,
        lambda x: x * 2,
      )
      .run()
    )
    self.assertEqual(sync_result, async_result)


# ============================================================
# 23. Exception Type Validation
# ============================================================


class ExceptionTypeValidationTest(TestCase):
  """Property-based tests for SPEC section 6.2.1 exception type validation."""

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_string_exception_type_raises_type_error(self, value):
    """SPEC section 6.2.1: string exception type raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(value).except_(lambda ei: None, exceptions='ValueError')

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_non_exception_type_raises_type_error(self, value):
    """SPEC section 6.2.1: non-BaseException subclass raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(value).except_(lambda ei: None, exceptions=int)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_empty_iterable_raises(self, value):
    """SPEC section 6.2.1: empty iterable of exceptions raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(value).except_(lambda ei: None, exceptions=[])

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_requires_callable(self, value):
    """SPEC section 6.2: except_ requires callable handler."""
    with self.assertRaises(TypeError):
      Chain(value).except_(42)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_finally_requires_callable(self, value):
    """SPEC section 6.3: finally_ requires callable handler."""
    with self.assertRaises(TypeError):
      Chain(value).finally_(42)

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_tuple_of_exception_types(self, value):
    """SPEC section 6.2.1: tuple of exception types works."""

    def raise_value_error(x):
      raise ValueError('test')

    result = Chain(value).then(raise_value_error).except_(lambda ei: 'caught', exceptions=(ValueError, TypeError)).run()
    self.assertEqual(result, 'caught')

  @given(value=small_ints)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_except_base_exception_warning(self, value):
    """SPEC section 6.2.1: catching BaseException (non-Exception) emits warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(value).except_(lambda ei: None, exceptions=BaseException)
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      self.assertTrue(len(runtime_warnings) > 0)
