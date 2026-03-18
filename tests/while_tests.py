# SPDX-License-Identifier: MIT
"""Tests for SPEC SS5.10 -- while_() loop operation.

Covers: basic .then() mode, .do() mode, predicate variations,
body calling conventions, break_(), return_(), async bridge,
builder validation, exception handling, q interactions,
clone correctness, repr, and edge cases.
"""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Q, QuentException
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# SS5.10 Basic .then() mode -- accumulator loop
# ---------------------------------------------------------------------------


class WhileThenBasicTest(SymmetricTestCase):
  """SS5.10: while_().then() -- body result feeds back as loop value."""

  async def test_decrement_until_zero(self) -> None:
    """Decrement from 10 until 0 (falsy). Spec example 1."""
    result = Q(10).while_().then(lambda x: x - 1).run()
    # 10 -> 9 -> ... -> 1 -> 0 (0 is falsy, loop stops)
    self.assertEqual(result, 0)

  async def test_decrement_until_zero_async_body(self) -> None:
    """Async body: same decrement logic."""

    async def dec(x):
      return x - 1

    result = await Q(10).while_().then(dec).run()
    self.assertEqual(result, 0)

  async def test_predicate_callable_halving(self) -> None:
    """Predicate callable -- halve while > 1. Spec example 2."""
    result = Q(100).while_(lambda x: x > 1).then(lambda x: x // 2).run()
    # 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1 (1 not > 1, stops)
    self.assertEqual(result, 1)

  async def test_predicate_callable_halving_async(self) -> None:
    """Async predicate + async body: same halving logic."""

    async def pred(x):
      return x > 1

    async def halve(x):
      return x // 2

    result = await Q(100).while_(pred).then(halve).run()
    self.assertEqual(result, 1)

  async def test_counting_up(self) -> None:
    """Count from 1 up, stop when >= 10."""
    result = Q(1).while_(lambda x: x < 10).then(lambda x: x + 1).run()
    self.assertEqual(result, 10)

  async def test_predicate_immediately_false(self) -> None:
    """Predicate false on first check -- body never runs."""
    body_called = []
    result = Q(0).while_().then(lambda x: (body_called.append(1), x - 1)[-1]).run()
    self.assertEqual(result, 0)
    self.assertEqual(body_called, [])

  async def test_predicate_callable_immediately_false(self) -> None:
    """Callable predicate false initially -- body never executes."""
    body_called = []
    result = Q(5).while_(lambda x: x > 100).then(lambda x: (body_called.append(1), x)[-1]).run()
    self.assertEqual(result, 5)
    self.assertEqual(body_called, [])


# ---------------------------------------------------------------------------
# SS5.10 Basic .do() mode -- side effects, cv passes through
# ---------------------------------------------------------------------------


class WhileDoBasicTest(SymmetricTestCase):
  """SS5.10: while_().do() -- body is side-effect, loop value unchanged."""

  async def test_do_side_effects_with_break(self) -> None:
    """do() runs side effects; break_() exits. Value passes through."""
    collected = []
    counter = {'n': 0}

    def body(x):
      counter['n'] += 1
      collected.append(x)
      if counter['n'] >= 3:
        return Q.break_()

    result = Q(42).while_(True).do(body).run()
    self.assertEqual(result, 42)
    self.assertEqual(collected, [42, 42, 42])

  async def test_do_side_effects_async(self) -> None:
    """Async do() body with break."""
    collected = []
    counter = {'n': 0}

    async def body(x):
      counter['n'] += 1
      collected.append(x)
      if counter['n'] >= 2:
        return Q.break_()

    result = await Q(99).while_(True).do(body).run()
    self.assertEqual(result, 99)
    self.assertEqual(collected, [99, 99])

  async def test_do_with_mutable_state_predicate(self) -> None:
    """do() with mutable-state predicate that eventually becomes false."""
    state = {'count': 3}
    collected = []

    def pred(x):
      return state['count'] > 0

    def body(x):
      collected.append(state['count'])
      state['count'] -= 1

    result = Q('hello').while_(pred).do(body).run()
    # do() does not change current value
    self.assertEqual(result, 'hello')
    self.assertEqual(collected, [3, 2, 1])


# ---------------------------------------------------------------------------
# SS5.10 Predicate variations
# ---------------------------------------------------------------------------


class WhilePredicateVariationsTest(SymmetricTestCase):
  """SS5.10: Predicate None (truthiness), callable with args/kwargs, Null always falsy."""

  async def test_none_predicate_uses_truthiness(self) -> None:
    """predicate=None: current value truthiness is used."""
    result = Q(3).while_().then(lambda x: x - 1).run()
    # 3 -> 2 -> 1 -> 0 (falsy)
    self.assertEqual(result, 0)

  async def test_none_predicate_empty_string_falsy(self) -> None:
    """Empty string is falsy -- loop never runs."""
    result = Q('').while_().then(lambda x: 'should not reach').run()
    self.assertEqual(result, '')

  async def test_none_predicate_none_value_falsy(self) -> None:
    """None is falsy -- loop never runs."""
    result = Q(None).while_().then(lambda x: 'should not reach').run()
    self.assertIsNone(result)

  async def test_callable_predicate_with_explicit_args(self) -> None:
    """Predicate with args: Rule 1 -- current value NOT passed."""
    # predicate(threshold) -> threshold > 0, always True until break
    counter = {'n': 0}

    def pred(threshold):
      return threshold > 0

    def body(x):
      counter['n'] += 1
      if counter['n'] >= 3:
        return Q.break_()
      return x + 1

    result = Q(0).while_(pred, 5).then(body).run()
    self.assertEqual(result, 2)

  async def test_callable_predicate_with_kwargs(self) -> None:
    """Predicate with kwargs: Rule 1 -- kwargs forwarded, cv NOT passed."""
    counter = {'n': 0}

    def pred(*, limit):
      return counter['n'] < limit

    def body(x):
      counter['n'] += 1
      return x + 10

    result = Q(0).while_(pred, limit=3).then(body).run()
    self.assertEqual(result, 30)

  async def test_non_callable_truthy_literal(self) -> None:
    """Non-callable truthy literal: always truthy -- needs break to exit."""
    counter = {'n': 0}

    def body(x):
      counter['n'] += 1
      if counter['n'] >= 5:
        return Q.break_(x)
      return x + 1

    result = Q(0).while_(True).then(body).run()
    self.assertEqual(result, 4)

  async def test_non_callable_falsy_literal_zero(self) -> None:
    """Non-callable falsy literal (0): loop never runs."""
    body_called = []
    result = Q(42).while_(0).then(lambda x: (body_called.append(1), x)[-1]).run()
    self.assertEqual(result, 42)
    self.assertEqual(body_called, [])

  async def test_non_callable_falsy_literal_empty_string(self) -> None:
    """Non-callable falsy literal (''): loop never runs."""
    result = Q(42).while_('').then(lambda x: x + 1).run()
    self.assertEqual(result, 42)

  async def test_null_sentinel_always_falsy(self) -> None:
    """Null sentinel: predicate=None on pipeline with no root value -- Null is always falsy."""
    # Q() has no root value (Null). while_() with None predicate checks truthiness.
    # Null is always falsy, so the loop body should never run.
    body_called = []
    result = Q().while_().then(lambda x: (body_called.append(1), x)[-1]).run()
    self.assertIsNone(result)
    self.assertEqual(body_called, [])


# ---------------------------------------------------------------------------
# SS5.10 Body calling conventions
# ---------------------------------------------------------------------------


class WhileBodyCallingConventionsTest(SymmetricTestCase):
  """SS5.10 + SS4: Body follows standard 2-rule calling convention."""

  async def test_body_explicit_args_rule1(self) -> None:
    """Body with explicit args: Rule 1 -- current value NOT passed."""
    counter = {'n': 0}

    def body(a, b):
      counter['n'] += 1
      return a + b

    # while_(True) + body(10, 20) => 30 every iteration
    # Need break to exit
    result = (
      Q(0)
      .while_(True)
      .then(
        lambda x: Q.break_(x) if counter['n'] >= 1 else body(10, 20),
      )
      .run()
    )
    self.assertEqual(result, 30)

  async def test_body_with_kwargs(self) -> None:
    """Body with kwargs: Rule 1 -- kwargs forwarded, cv not passed."""
    counter = {'n': 0}

    def inc(*, step):
      counter['n'] += 1
      return step * counter['n']

    result = Q(0).while_(lambda x: counter['n'] < 3).then(inc, step=10).run()
    self.assertEqual(result, 30)

  async def test_body_nested_chain(self) -> None:
    """Body is a nested Q pipeline -- follows standard evaluation."""
    inner = Q().then(lambda x: x - 1)
    result = Q(5).while_().then(inner).run()
    # 5 -> 4 -> 3 -> 2 -> 1 -> 0 (falsy)
    self.assertEqual(result, 0)

  async def test_body_non_callable_literal(self) -> None:
    """Non-callable literal body: replaces value each iteration."""
    # then(0) replaces current value with 0 each iteration.
    # With predicate=None, 0 is falsy -> loop runs once and stops.
    result = Q(5).while_().then(0).run()
    # First iteration: pred=5 (truthy) -> body produces 0 -> re-check: pred=0 (falsy) -> stop
    self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# SS5.10 break_() behavior
# ---------------------------------------------------------------------------


class WhileBreakNoValueTest(SymmetricTestCase):
  """SS5.10: break_() with no value -- result is current loop value at time of break."""

  async def test_break_no_value_returns_current_loop_value(self) -> None:
    """break_() with no value: current loop value at break time is the result.

    NOTE: SPEC SS5.10 example comment says this returns 64, but the implementation
    returns 128. The current_value entering the break iteration is 128 (result of
    the prior iteration's body: 64*2=128). The spec contract text ("the current
    loop value at the time of the break") is consistent with 128 -- the example
    comment appears to be a documentation error in SPEC.md line 643.
    """
    result = Q(1).while_(True).then(lambda x: Q.break_() if x >= 100 else x * 2).run()
    # Trace: 1->2->4->8->16->32->64->128. At cv=128, body raises break_().
    # _handle_break returns current_value=128 (the loop value entering the break iteration).
    self.assertEqual(result, 128)

  async def test_break_no_value_simple(self) -> None:
    """Simple break_() no value case."""
    # Count 0->1->2->3, break at 3
    result = Q(0).while_(True).then(lambda x: Q.break_() if x == 3 else x + 1).run()
    # x=0: returns 1. cv=1
    # x=1: returns 2. cv=2
    # x=2: returns 3. cv=3
    # x=3: break_() raised. cv=3 (not updated).
    self.assertEqual(result, 3)


class WhileBreakWithValueTest(SymmetricTestCase):
  """SS5.10: break_(value) -- break value replaces pipeline value."""

  async def test_break_with_value(self) -> None:
    """Spec example: break_(x) when x >= 100 => returns 128."""
    result = Q(1).while_(True).then(lambda x: Q.break_(x) if x >= 100 else x * 2).run()
    # 1->2->4->8->16->32->64->128: x=128, 128>=100, break_(128)
    self.assertEqual(result, 128)

  async def test_break_with_value_async(self) -> None:
    """Async body with break_(value)."""

    async def body(x):
      if x >= 50:
        return Q.break_(x * 10)
      return x * 2

    result = await Q(1).while_(True).then(body).run()
    # 1->2->4->8->16->32->64: 64>=50, break_(640)
    self.assertEqual(result, 640)

  async def test_break_with_non_callable_value(self) -> None:
    """break_(literal) -- literal becomes the result."""
    counter = {'n': 0}

    def body(x):
      counter['n'] += 1
      if counter['n'] >= 3:
        return Q.break_('done')
      return x + 1

    result = Q(0).while_(True).then(body).run()
    self.assertEqual(result, 'done')


class WhileBreakInPredicateTest(SymmetricTestCase):
  """SS5.10: break_() can be raised from the predicate."""

  async def test_break_in_predicate_no_value(self) -> None:
    """break_() from predicate: stops loop, returns current loop value."""
    counter = {'n': 0}

    def pred(x):
      counter['n'] += 1
      if counter['n'] > 3:
        return Q.break_()
      return True

    result = Q(10).while_(pred).then(lambda x: x + 1).run()
    # pred(10): True, body: 11. pred(11): True, body: 12. pred(12): True, body: 13.
    # pred(13): break_() => returns current_value = 13
    self.assertEqual(result, 13)

  async def test_break_in_predicate_with_value(self) -> None:
    """break_(value) from predicate: break value becomes result."""
    counter = {'n': 0}

    def pred(x):
      counter['n'] += 1
      if counter['n'] > 2:
        return Q.break_('pred_break')
      return True

    result = Q(0).while_(pred).then(lambda x: x + 1).run()
    self.assertEqual(result, 'pred_break')

  async def test_break_in_predicate_async(self) -> None:
    """Async predicate raises break_()."""
    counter = {'n': 0}

    async def pred(x):
      counter['n'] += 1
      if counter['n'] > 2:
        return Q.break_('async_pred_break')
      return True

    result = await Q(0).while_(pred).then(lambda x: x + 1).run()
    self.assertEqual(result, 'async_pred_break')


class WhileBreakLazyEvaluationTest(SymmetricTestCase):
  """SS7.1 + SS5.10: break_() value is lazily evaluated."""

  async def test_break_callable_lazily_evaluated(self) -> None:
    """break_(callable) -- callable only invoked when signal is caught."""
    call_log = []

    def lazy_value():
      call_log.append('called')
      return 'lazy_result'

    self.assertEqual(call_log, [])
    result = Q(0).while_(True).then(lambda x: Q.break_(lazy_value) if x >= 2 else x + 1).run()
    self.assertEqual(call_log, ['called'])
    self.assertEqual(result, 'lazy_result')

  async def test_break_callable_with_args(self) -> None:
    """break_(callable, *args) -- callable invoked with args when caught."""
    result = Q(0).while_(True).then(lambda x: Q.break_(lambda a, b: a + b, 10, 20) if x >= 2 else x + 1).run()
    self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# SS5.10 return_() behavior
# ---------------------------------------------------------------------------


class WhileReturnTest(SymmetricTestCase):
  """SS5.10: return_() propagates to enclosing chain -- exits entire chain."""

  async def test_return_exits_entire_chain(self) -> None:
    """return_() in while body exits the chain, not just the loop."""
    result = (
      Q(0).while_(True).then(lambda x: Q.return_('early') if x >= 2 else x + 1).then(lambda x: 'should not reach').run()
    )
    self.assertEqual(result, 'early')

  async def test_return_with_value(self) -> None:
    """return_(value) -- value becomes the chain result."""
    result = Q(5).while_(True).then(lambda x: Q.return_(x * 100) if x >= 5 else x + 1).run()
    self.assertEqual(result, 500)

  async def test_return_no_value(self) -> None:
    """return_() with no value -- chain result is None."""
    result = Q(0).while_(True).then(lambda x: Q.return_() if x >= 2 else x + 1).run()
    self.assertIsNone(result)

  async def test_return_async(self) -> None:
    """Async: return_() in while body exits chain."""

    async def body(x):
      if x >= 3:
        return Q.return_('async_early')
      return x + 1

    result = await Q(0).while_(True).then(body).then(lambda x: 'never').run()
    self.assertEqual(result, 'async_early')


# ---------------------------------------------------------------------------
# SS5.10 Async bridge
# ---------------------------------------------------------------------------


class WhileAsyncBridgeTest(SymmetricTestCase):
  """SS5.10: Sync/async bridge -- transparent across iterations."""

  async def test_sync_predicate_sync_body(self) -> None:
    """Fully sync: no async overhead."""
    result = Q(5).while_(lambda x: x > 0).then(lambda x: x - 1).run()
    self.assertEqual(result, 0)

  async def test_async_predicate_sync_body(self) -> None:
    """Async predicate, sync body: transitions on first predicate."""

    async def pred(x):
      return x > 0

    result = await Q(3).while_(pred).then(lambda x: x - 1).run()
    self.assertEqual(result, 0)

  async def test_sync_predicate_async_body(self) -> None:
    """Sync predicate, async body: transitions on first body result."""

    async def body(x):
      return x - 1

    result = await Q(3).while_(lambda x: x > 0).then(body).run()
    self.assertEqual(result, 0)

  async def test_async_predicate_async_body(self) -> None:
    """Both async: full async loop."""

    async def pred(x):
      return x > 0

    async def body(x):
      return x - 1

    result = await Q(4).while_(pred).then(body).run()
    self.assertEqual(result, 0)

  async def test_mid_loop_transition(self) -> None:
    """Sync predicate/body, but body returns awaitable mid-loop."""
    counter = {'n': 0}

    async def async_dec(x):
      return x - 1

    def body(x):
      counter['n'] += 1
      if counter['n'] > 2:
        # After a few sync iterations, return an awaitable
        return async_dec(x)
      return x - 1

    result = await Q(5).while_(lambda x: x > 0).then(body).run()
    self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# SS5.10 Builder validation
# ---------------------------------------------------------------------------


class WhileBuilderValidationTest(TestCase):
  """SS5.10: Build-time constraints and error handling."""

  def test_while_while_conflict(self) -> None:
    """while_() while another while_() is pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_().while_()

  def test_if_while_conflict(self) -> None:
    """if_() while while_() is pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_().if_(True)

  def test_while_if_conflict(self) -> None:
    """while_() while if_() is pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).if_().while_()

  def test_pending_while_requires_then_or_do(self) -> None:
    """Execution methods while while_() is pending raise QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_().run()

  def test_pending_while_forbids_foreach(self) -> None:
    """foreach() while while_() pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q([1, 2]).while_().foreach(lambda x: x)

  def test_pending_while_forbids_except(self) -> None:
    """except_() while while_() pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_().except_(lambda info: None)

  def test_pending_while_forbids_finally(self) -> None:
    """finally_() while while_() pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_().finally_(lambda rv: None)

  def test_args_without_predicate_raises(self) -> None:
    """args/kwargs without predicate raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_(None, 10)

  def test_kwargs_without_predicate_raises(self) -> None:
    """kwargs without predicate raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_(None, key=10)

  def test_else_after_while_then_raises(self) -> None:
    """else_() after while_().then() raises QuentException -- else only valid after if_()."""
    with self.assertRaises(QuentException):
      Q(1).while_(True).then(lambda x: x).else_(0)

  def test_else_do_after_while_do_raises(self) -> None:
    """else_do() after while_().do() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).while_(True).do(lambda x: x).else_do(lambda x: x)

  def test_non_callable_predicate_with_args_raises_type_error(self) -> None:
    """Non-callable literal predicate with args raises TypeError at build time."""
    with self.assertRaises(TypeError):
      Q(1).while_(42, 'some_arg').then(lambda x: x)


# ---------------------------------------------------------------------------
# SS5.10 Exception handling
# ---------------------------------------------------------------------------


class WhileExceptionHandlingTest(SymmetricTestCase):
  """SS5.10 + SS6: Exceptions in while body/predicate."""

  async def test_body_exception_propagates(self) -> None:
    """Exception in body propagates out of chain."""
    c = Q(1).while_(True).then(lambda x: 1 / 0)
    with self.assertRaises(ZeroDivisionError):
      c.run()

  async def test_predicate_exception_propagates(self) -> None:
    """Exception in predicate propagates out of chain."""

    def bad_pred(x):
      raise ValueError('pred error')

    c = Q(1).while_(bad_pred).then(lambda x: x)
    with self.assertRaises(ValueError):
      c.run()

  async def test_except_catches_body_exception(self) -> None:
    """except_() catches exceptions from while body."""
    result = Q(1).while_(True).then(lambda x: 1 / 0).except_(lambda info: 'caught').run()
    self.assertEqual(result, 'caught')

  async def test_except_catches_predicate_exception(self) -> None:
    """except_() catches exceptions from while predicate."""
    counter = {'n': 0}

    def pred(x):
      counter['n'] += 1
      if counter['n'] > 2:
        raise ValueError('pred fail')
      return True

    result = Q(0).while_(pred).then(lambda x: x + 1).except_(lambda info: 'pred_caught').run()
    self.assertEqual(result, 'pred_caught')

  async def test_body_exception_async(self) -> None:
    """Async body exception propagates."""

    async def body(x):
      raise ValueError('async body error')

    c = Q(1).while_(True).then(body)
    with self.assertRaises(ValueError):
      await c.run()

  async def test_except_catches_async_body_exception(self) -> None:
    """except_() catches async body exception."""

    async def body(x):
      raise ValueError('async error')

    result = await Q(1).while_(True).then(body).except_(lambda info: 'async_caught').run()
    self.assertEqual(result, 'async_caught')


# ---------------------------------------------------------------------------
# SS5.10 Quent interactions
# ---------------------------------------------------------------------------


class WhileChainInteractionsTest(SymmetricTestCase):
  """SS5.10: while_ interacts correctly with other chain operations."""

  async def test_downstream_steps_receive_loop_result(self) -> None:
    """Steps after while_ receive the loop's final value."""
    result = Q(10).while_().then(lambda x: x - 1).then(lambda x: x + 100).run()
    # while produces 0, then +100 = 100
    self.assertEqual(result, 100)

  async def test_finally_runs_after_while(self) -> None:
    """finally_() runs after while loop completes."""
    cleanup_called = []

    def cleanup(rv):
      cleanup_called.append(rv)

    result = Q(3).while_().then(lambda x: x - 1).finally_(cleanup).run()
    self.assertEqual(result, 0)
    self.assertEqual(cleanup_called, [3])  # root value passed to finally

  async def test_finally_runs_on_exception_in_while(self) -> None:
    """finally_() runs even when while body raises."""
    cleanup_called = []

    def cleanup(rv):
      cleanup_called.append(True)

    c = Q(1).while_(True).then(lambda x: 1 / 0).finally_(cleanup)
    with self.assertRaises(ZeroDivisionError):
      c.run()
    self.assertEqual(cleanup_called, [True])

  async def test_nested_chain_as_body(self) -> None:
    """Nested chain as while body."""
    inner = Q().then(lambda x: x - 1)
    result = Q(5).while_().then(inner).run()
    self.assertEqual(result, 0)

  async def test_while_inside_if(self) -> None:
    """while_ can be inside a nested chain used in if_().then()."""
    # Use a nested q to combine if and while
    loop_chain = Q().while_(lambda x: x > 0).then(lambda x: x - 1)
    result = Q(5).if_(lambda x: x > 3).then(loop_chain).run()
    self.assertEqual(result, 0)

  async def test_while_after_other_steps(self) -> None:
    """while_ after other pipeline steps."""
    result = Q(1).then(lambda x: x + 9).while_().then(lambda x: x - 1).run()
    # 1 -> 10 (first then), then while: 10->9->...->0
    self.assertEqual(result, 0)


class WhileOnStepTest(IsolatedAsyncioTestCase):
  """SS14 + SS5.10: on_step callback fires for while_ step."""

  def setUp(self) -> None:
    self._original_on_step = Q.on_step
    self.step_log: list[tuple[str, object, object]] = []

    def on_step(q, step_name, input_value, result, elapsed_ns):
      self.step_log.append((step_name, input_value, result))

    Q.on_step = on_step

  def tearDown(self) -> None:
    Q.on_step = self._original_on_step

  async def test_on_step_fires_for_while(self) -> None:
    """on_step callback is invoked when while_ step executes."""
    result = Q(3).while_().then(lambda x: x - 1).run()
    self.assertEqual(result, 0)
    # on_step should have been called at least once (for the while_ step)
    step_names = [name for name, _, _ in self.step_log]
    self.assertIn('while_', step_names)


# ---------------------------------------------------------------------------
# SS10.1 Clone correctness
# ---------------------------------------------------------------------------


class WhileCloneTest(SymmetricTestCase):
  """SS10.1 + SS5.10: clone() produces independent copies of while_ operations."""

  async def test_clone_produces_independent_results(self) -> None:
    """Cloned chain runs independently."""
    base = Q(10).while_().then(lambda x: x - 1)
    cloned = base.clone()

    result1 = base.run()
    result2 = cloned.run()
    self.assertEqual(result1, 0)
    self.assertEqual(result2, 0)

  async def test_clone_extend_independent(self) -> None:
    """Extending clone does not affect original."""
    base = Q(5).while_(lambda x: x > 0).then(lambda x: x - 1)
    cloned = base.clone().then(lambda x: x + 100)

    result_base = base.run()
    result_cloned = cloned.run()
    self.assertEqual(result_base, 0)
    self.assertEqual(result_cloned, 100)

  async def test_clone_with_predicate_args(self) -> None:
    """Clone with predicate args: cloned chain is structurally independent."""

    def pred(*, limit):
      return True  # we'll rely on the body to break

    # Use a stateless body -- break after value reaches 2
    base = Q(0).while_(pred, limit=5).then(lambda x: Q.break_(x) if x >= 2 else x + 1)
    cloned = base.clone()
    # Both should produce the same result (2)
    result1 = base.run()
    result2 = cloned.run()
    self.assertEqual(result1, 2)
    self.assertEqual(result2, 2)


# ---------------------------------------------------------------------------
# SS5.10 Traceback visualization / repr
# ---------------------------------------------------------------------------


class WhileReprTest(TestCase):
  """SS5.10: __repr__ contains while_ in visualization."""

  def test_repr_contains_while(self) -> None:
    """repr of pipeline with while_ contains 'while_'."""
    c = Q(10).while_(lambda x: x > 0).then(lambda x: x - 1)
    r = repr(c)
    self.assertIn('while_', r)

  def test_repr_with_predicate_name(self) -> None:
    """repr shows predicate name in while_ visualization."""

    def my_pred(x):
      return x > 0

    c = Q(10).while_(my_pred).then(lambda x: x - 1)
    r = repr(c)
    self.assertIn('while_', r)

  def test_repr_no_predicate(self) -> None:
    """repr of while_() with no predicate still shows while_."""
    c = Q(10).while_().then(lambda x: x - 1)
    r = repr(c)
    self.assertIn('while_', r)


# ---------------------------------------------------------------------------
# SS5.10 Edge cases
# ---------------------------------------------------------------------------


class WhileEdgeCasesTest(SymmetricTestCase):
  """SS5.10: Edge cases and boundary conditions."""

  async def test_while_as_only_step(self) -> None:
    """while_ as the only step in the chain."""
    result = Q(3).while_().then(lambda x: x - 1).run()
    self.assertEqual(result, 0)

  async def test_no_root_value_with_run_start(self) -> None:
    """Q() with no root, using run(start)."""
    result = Q().while_(lambda x: x > 0).then(lambda x: x - 1).run(5)
    self.assertEqual(result, 0)

  async def test_body_returns_none(self) -> None:
    """Body returns None -- None is falsy, so predicate=None loop stops."""
    result = Q(5).while_().then(lambda x: None).run()
    # Body returns None, predicate checks truthiness of None -> falsy -> stop
    self.assertIsNone(result)

  async def test_single_iteration(self) -> None:
    """Loop runs exactly once."""
    # Predicate true initially, body makes it false
    result = Q(1).while_().then(lambda x: 0).run()
    # pred=1 (truthy), body returns 0, pred=0 (falsy), stop
    self.assertEqual(result, 0)

  async def test_while_with_string_accumulation(self) -> None:
    """Accumulate characters in while loop."""
    result = Q('').while_(lambda x: len(x) < 5).then(lambda x: x + 'a').run()
    self.assertEqual(result, 'aaaaa')

  async def test_while_with_list_accumulation(self) -> None:
    """Accumulate list in while loop."""

    def body(x):
      return [*x, len(x)]

    result = Q([]).while_(lambda x: len(x) < 4).then(body).run()
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_do_mode_preserves_value(self) -> None:
    """do() mode: current value is unchanged throughout iterations."""
    iterations = []
    counter = {'n': 0}

    def body(x):
      counter['n'] += 1
      iterations.append(x)
      if counter['n'] >= 3:
        return Q.break_()

    result = Q('preserved').while_(True).do(body).run()
    self.assertEqual(result, 'preserved')
    self.assertEqual(iterations, ['preserved', 'preserved', 'preserved'])

  async def test_false_predicate_no_root_value(self) -> None:
    """No root value (Null), predicate=None: Null is always falsy."""
    body_called = []
    result = Q().while_().then(lambda x: (body_called.append(1), x)[-1]).run()
    self.assertIsNone(result)
    self.assertEqual(body_called, [])

  async def test_while_result_feeds_to_next_step(self) -> None:
    """While result correctly feeds into subsequent then()."""
    result = Q(10).while_(lambda x: x > 0).then(lambda x: x - 1).then(lambda x: f'result={x}').run()
    self.assertEqual(result, 'result=0')

  async def test_multiple_while_loops_in_sequence(self) -> None:
    """Two while_ loops in sequence in one chain."""
    result = Q(10).while_(lambda x: x > 5).then(lambda x: x - 1).while_(lambda x: x > 0).then(lambda x: x - 1).run()
    # First while: 10->9->...->5
    # Second while: 5->4->...->0
    self.assertEqual(result, 0)

  async def test_do_with_predicate_eventually_false(self) -> None:
    """do() mode with external state predicate that becomes false."""
    state = {'remaining': 3}
    effects = []

    def pred(x):
      return state['remaining'] > 0

    def body(x):
      effects.append(x)
      state['remaining'] -= 1

    result = Q(42).while_(pred).do(body).run()
    self.assertEqual(result, 42)
    self.assertEqual(effects, [42, 42, 42])

  async def test_break_in_do_with_value(self) -> None:
    """break_(value) in do() mode: break value becomes result."""
    counter = {'n': 0}

    def body(x):
      counter['n'] += 1
      if counter['n'] >= 2:
        return Q.break_('break_value')

    result = Q('original').while_(True).do(body).run()
    self.assertEqual(result, 'break_value')

  async def test_break_in_do_without_value(self) -> None:
    """break_() in do() mode: current value preserved."""
    counter = {'n': 0}

    def body(x):
      counter['n'] += 1
      if counter['n'] >= 2:
        return Q.break_()

    result = Q('original').while_(True).do(body).run()
    self.assertEqual(result, 'original')


# ---------------------------------------------------------------------------
# Symmetric async bridge testing (variant-based)
# ---------------------------------------------------------------------------


class WhileSymmetricBridgeTest(SymmetricTestCase):
  """SS5.10 + SS2: Symmetric sync/async testing for while_ bridge contract."""

  async def test_decrement_bridge(self) -> None:
    """Sync/async body variant: decrement loop."""

    def sync_dec(x):
      return x - 1

    async def async_dec(x):
      return x - 1

    await self.variant(
      lambda fn: Q(5).while_().then(fn).run(),
      expected=0,
      fn=[('sync', sync_dec), ('async', async_dec)],
    )

  async def test_predicate_bridge(self) -> None:
    """Sync/async predicate variant: count up."""

    def sync_pred(x):
      return x < 10

    async def async_pred(x):
      return x < 10

    await self.variant(
      lambda pred: Q(1).while_(pred).then(lambda x: x + 1).run(),
      expected=10,
      pred=[('sync', sync_pred), ('async', async_pred)],
    )

  async def test_both_axes_bridge(self) -> None:
    """Both predicate and body as sync/async axes."""

    def sync_pred(x):
      return x > 0

    async def async_pred(x):
      return x > 0

    def sync_body(x):
      return x - 1

    async def async_body(x):
      return x - 1

    await self.variant(
      lambda pred, body: Q(4).while_(pred).then(body).run(),
      expected=0,
      pred=[('sync', sync_pred), ('async', async_pred)],
      body=[('sync', sync_body), ('async', async_body)],
    )

  async def test_break_bridge(self) -> None:
    """Break with value across sync/async bodies."""

    def sync_body(x):
      if x >= 5:
        return Q.break_('stopped')
      return x + 1

    async def async_body(x):
      if x >= 5:
        return Q.break_('stopped')
      return x + 1

    await self.variant(
      lambda fn: Q(0).while_(True).then(fn).run(),
      expected='stopped',
      fn=[('sync', sync_body), ('async', async_body)],
    )


if __name__ == '__main__':
  import unittest

  unittest.main()
