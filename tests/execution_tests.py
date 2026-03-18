# SPDX-License-Identifier: MIT
"""Tests for SPEC §3 and §8 — Pipeline Model and Execution.

Covers: q construction, root value semantics, value flow,
run() behavior, run value vs root value, __call__ alias,
sync/async execution model.
"""

from __future__ import annotations

import asyncio
from unittest import TestCase

from quent import Q, QuentException, QuentExcInfo, __version__
from tests.fixtures import (
  async_double,
  async_fn,
  sync_double,
  sync_fn,
)
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §3: Quent construction
# ---------------------------------------------------------------------------


class ChainConstructionTest(SymmetricTestCase):
  """SPEC §3: Constructor signature and root value."""

  async def test_chain_no_args(self) -> None:
    """Q() — no root value."""
    result = Q().run()
    self.assertIsNone(result)

  async def test_chain_with_none(self) -> None:
    """Q(None) — root value is None, distinct from Q()."""
    result = Q(None).run()
    self.assertIsNone(result)

  async def test_chain_with_value(self) -> None:
    """Q(v) — non-callable value is used as-is."""
    result = Q(42).run()
    self.assertEqual(result, 42)

  async def test_chain_with_callable(self) -> None:
    """Q(callable) — callable is invoked with no args at run time."""
    await self.variant(
      lambda fn: Q(fn).run(),
      fn=[('sync', lambda: 99), ('async', self._async_99)],
      expected=99,
    )

  async def test_chain_with_callable_and_args(self) -> None:
    """Q(callable, *args, **kwargs) — callable invoked with provided args."""
    result = Q(lambda a, b: a + b, 3, 4).run()
    self.assertEqual(result, 7)

  async def test_chain_with_callable_and_kwargs(self) -> None:
    """Q(callable, **kwargs)."""
    result = Q(lambda *, key: key, key='hello').run()
    self.assertEqual(result, 'hello')

  async def test_chain_non_callable_with_args_raises_typeerror(self) -> None:
    """Q(non_callable, *args) → TypeError at build time."""
    with self.assertRaises(TypeError):
      Q(42, 'extra')

  async def test_chain_non_callable_with_kwargs_raises_typeerror(self) -> None:
    """Q(non_callable, **kwargs) → TypeError at build time."""
    with self.assertRaises(TypeError):
      Q(42, key='val')

  async def test_chain_kwargs_without_root_raises_typeerror(self) -> None:
    """SPEC §3: kwargs require a root value — Q(key=val) raises TypeError."""
    with self.assertRaises(TypeError):
      Q(key='val')

  def test_version_is_string(self):
    """SPEC §19: __version__ is a public API string."""
    self.assertIsInstance(__version__, str)
    self.assertNotEqual(__version__, '')

  @staticmethod
  async def _async_99():
    return 99


# ---------------------------------------------------------------------------
# §3: Root value semantics
# ---------------------------------------------------------------------------


class RootValueTest(SymmetricTestCase):
  """SPEC §3: Root value semantics — unique tests not covered elsewhere."""

  async def test_run_time_root_only(self) -> None:
    """chain.run(v) when no build-time root."""
    result = Q().then(sync_fn).run(5)
    self.assertEqual(result, 6)

  async def test_root_callable_evaluated_at_run_time(self) -> None:
    """Root callable is evaluated when the chain runs, not at construction."""
    call_count = 0

    def counter():
      nonlocal call_count
      call_count += 1
      return call_count

    c = Q(counter)
    r1 = c.run()
    r2 = c.run()
    self.assertEqual(r1, 1)
    self.assertEqual(r2, 2)


# ---------------------------------------------------------------------------
# §3: .do() side-effect validation
# ---------------------------------------------------------------------------


class DoStepValidationTest(SymmetricTestCase):
  """SPEC §3: .do() build-time validation and side-effect behavior."""

  async def test_do_side_effect_executes(self) -> None:
    """do() actually executes the callable for side effects."""
    results = []

    def track(x):
      results.append(x)

    Q(42).do(track).run()
    self.assertEqual(results, [42])

  async def test_do_requires_callable(self) -> None:
    """do(non_callable) → TypeError at build time."""
    with self.assertRaises(TypeError):
      Q(5).do(42)

  async def test_do_requires_callable_none(self) -> None:
    """do(None) → TypeError at build time."""
    with self.assertRaises(TypeError):
      Q(5).do(None)

  async def test_do_with_args(self) -> None:
    """do(fn, *args) — fn called with explicit args, result discarded."""
    results = []

    def track(a, b):
      results.append((a, b))

    Q(99).do(track, 1, 2).run()
    self.assertEqual(results, [(1, 2)])


# ---------------------------------------------------------------------------
# §3: Value flow
# ---------------------------------------------------------------------------


class ValueFlowTest(SymmetricTestCase):
  """SPEC §3: Value threading through the pipeline."""

  async def test_basic_value_flow(self) -> None:
    """root → then(f) → do(g) → then(h) → result."""
    results = []

    def log(x):
      results.append(x)

    result = Q(5).then(sync_fn).do(log).then(sync_double).run()
    # root=5, sync_fn(5)=6, do(log(6)) discarded, sync_double(6)=12
    self.assertEqual(result, 12)
    self.assertEqual(results, [6])

  async def test_pipeline_with_no_value_returns_none(self) -> None:
    """When pipeline completes with no value ever produced, returns None."""
    result = Q().do(lambda: None).run()
    self.assertIsNone(result)

  async def test_none_is_valid_pipeline_value(self) -> None:
    """None is a legitimate value, distinct from 'no value'."""
    result = Q(None).then(lambda x: x is None).run()
    self.assertTrue(result)

  async def test_value_replacement(self) -> None:
    """then(non_callable) replaces the current value directly."""
    result = Q(5).then(sync_fn).then('replaced').run()
    self.assertEqual(result, 'replaced')


# ---------------------------------------------------------------------------
# §3: Root value to error handlers — unique tests
# ---------------------------------------------------------------------------


class RootValueToHandlersTest(SymmetricTestCase):
  """SPEC §3/§6: Unique handler root value tests not covered by RootValueCaptureTest."""

  async def test_except_receives_root_on_first_step_failure(self) -> None:
    """When first step after root fails, except_ still receives the root value."""
    received_rv = []

    def handler(info: QuentExcInfo):
      received_rv.append(info.root_value)
      return 'recovered'

    result = Q(42).then(lambda x: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'recovered')
    self.assertEqual(received_rv, [42])


# ---------------------------------------------------------------------------
# §3: Pipeline structure
# ---------------------------------------------------------------------------


class PipelineStructureTest(TestCase):
  """SPEC §3: Append-only linked list execution model."""

  def test_building_appends_steps(self) -> None:
    """Building adds steps without modifying previous structure."""
    c = Q(5)
    c.then(sync_fn)
    c.then(sync_double)
    result = c.run()
    # sync_fn(5)=6, sync_double(6)=12
    self.assertEqual(result, 12)

  def test_fluent_api(self) -> None:
    """Builder methods return self for fluent chaining."""
    c = Q(5)
    ret = c.then(sync_fn)
    self.assertIs(ret, c)

  def test_chain_reuse(self) -> None:
    """A fully constructed chain can be executed multiple times."""
    c = Q(5).then(sync_fn)
    self.assertEqual(c.run(), 6)
    self.assertEqual(c.run(), 6)
    self.assertEqual(c.run(10), 11)

  def test_chain_bool_always_true(self) -> None:
    """SPEC §8.3: Q.__bool__ always returns True regardless of contents."""
    self.assertTrue(bool(Q()))
    self.assertTrue(bool(Q(None)))
    self.assertTrue(bool(Q(5)))
    self.assertTrue(bool(Q(0)))
    self.assertTrue(bool(Q().then(lambda: None)))


# ---------------------------------------------------------------------------
# §3: do-only q returns None
# ---------------------------------------------------------------------------


class DoOnlyChainReturnsNoneTest(TestCase):
  """SPEC §3: pipeline with only .do() steps and no initial value -> None."""

  def test_do_only_chain_returns_none(self) -> None:
    """Q().do(fn).run() returns None.

    Per SPEC §3: 'When the pipeline completes with no value ever having been
    produced (e.g., Q().do(print).run()), the result is None. The internal
    "no value" sentinel is never exposed to users.'
    """
    side_effects = []
    result = Q().do(lambda: side_effects.append('called')).run()
    self.assertIsNone(result)
    self.assertEqual(side_effects, ['called'])


# ---------------------------------------------------------------------------
# §8.1 run()
# ---------------------------------------------------------------------------


class RunBasicTest(SymmetricTestCase):
  """SPEC §8.1: run() — execute pipeline, return final value."""

  async def test_run_returns_final_value(self) -> None:
    """run() returns the final pipeline value."""
    result = Q(5).then(lambda x: x * 2).run()
    self.assertEqual(result, 10)

  async def test_run_no_steps_returns_root(self) -> None:
    """run() with no steps returns root value."""
    result = Q(42).run()
    self.assertEqual(result, 42)

  async def test_run_empty_chain_returns_none(self) -> None:
    """run() on empty chain returns None."""
    result = Q().run()
    self.assertIsNone(result)

  async def test_run_multiple_steps(self) -> None:
    """run() threads value through multiple steps."""
    result = Q(1).then(lambda x: x + 1).then(lambda x: x * 3).then(lambda x: x - 1).run()
    # 1 → 2 → 6 → 5
    self.assertEqual(result, 5)


# ---------------------------------------------------------------------------
# §8.1: Run value replaces root
# ---------------------------------------------------------------------------


class RunValueReplacesRootTest(SymmetricTestCase):
  """SPEC §8.1: run value replaces root value."""

  async def test_run_value_replaces_root(self) -> None:
    """Q(A).then(B).run(C) ≡ Q(C).then(B).run()."""
    c = Q(100).then(lambda x: x * 2)
    result = c.run(5)
    self.assertEqual(result, 10)  # 5*2=10, root 100 is ignored

  async def test_run_value_equivalence(self) -> None:
    """Verify equivalence: Q(A).then(B).run(C) == Q(C).then(B).run()."""
    fn = lambda x: x + 10
    r1 = Q('ignored').then(fn).run(5)
    r2 = Q(5).then(fn).run()
    self.assertEqual(r1, r2)
    self.assertEqual(r1, 15)


# ---------------------------------------------------------------------------
# §8.1: Run value is callable
# ---------------------------------------------------------------------------


class RunCallableValueTest(SymmetricTestCase):
  """SPEC §8.1: run value that is callable is evaluated first."""

  async def test_run_callable_value_evaluated(self) -> None:
    """When run value is callable, it's evaluated first."""
    result = Q().then(lambda x: x * 2).run(lambda: 5)
    self.assertEqual(result, 10)

  async def test_run_callable_with_args(self) -> None:
    """Run callable with args: fn(*args, **kwargs)."""
    result = Q().then(lambda x: x + 1).run(lambda a, b: a + b, 3, 4)
    # root = 3+4=7, then 7+1=8
    self.assertEqual(result, 8)


# ---------------------------------------------------------------------------
# §8.1: Neither root nor run value
# ---------------------------------------------------------------------------


class NoRootNoRunTest(SymmetricTestCase):
  """SPEC §8.1: Neither root nor run value."""

  async def test_no_value_first_callable_no_args(self) -> None:
    """No root, no run value: first callable gets no args."""
    result = Q().then(lambda: 42).run()
    self.assertEqual(result, 42)

  async def test_no_value_chain_of_no_arg_callables(self) -> None:
    """Multiple no-arg steps after initial value injection."""
    result = Q().then(lambda: 10).then(lambda x: x + 5).run()
    self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# §8.1: Root value capture for error handlers
# ---------------------------------------------------------------------------


class RootValueCaptureTest(SymmetricTestCase):
  """SPEC §8.1: root value capture for error handlers."""

  async def test_root_value_passed_to_except(self) -> None:
    """except_ handler receives root value, not current pipeline value."""
    received = {}

    def handler(info: QuentExcInfo):
      received['rv'] = info.root_value
      return 'handled'

    result = Q(42).then(lambda x: x + 1).then(lambda x: x + 1).then(lambda x: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(received['rv'], 42)  # root value, not 44

  async def test_root_value_passed_to_finally(self) -> None:
    """finally_ handler receives root value, not current pipeline value."""
    received = {}

    def cleanup(rv):
      received['rv'] = rv

    result = Q(42).then(lambda x: x + 1).then(lambda x: x + 1).finally_(cleanup).run()
    self.assertEqual(result, 44)
    self.assertEqual(received['rv'], 42)

  async def test_run_value_replaces_root_for_handlers(self) -> None:
    """Run value replaces root for error handlers too."""
    received = {}

    def handler(info: QuentExcInfo):
      received['rv'] = info.root_value
      return 'handled'

    result = Q(100).then(lambda x: 1 / 0).except_(handler).run(5)
    self.assertEqual(result, 'handled')
    self.assertEqual(received['rv'], 5)  # run value, not 100


# ---------------------------------------------------------------------------
# §8.1: Root callable failure
# ---------------------------------------------------------------------------


class RootCallableFailureTest(SymmetricTestCase):
  """SPEC §8.1: Root callable failure → error handling applies."""

  async def test_root_callable_failure_triggers_except(self) -> None:
    """Root callable failure: except_ handler invoked."""

    def bad_root():
      raise ValueError('root boom')

    received = {}

    def handler(info: QuentExcInfo):
      received['exc'] = info.exc
      received['rv'] = info.root_value
      return 'handled'

    result = Q(bad_root).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertIsInstance(received['exc'], ValueError)
    self.assertIsNone(received['rv'])  # root_value normalized to None

  async def test_root_callable_failure_triggers_finally(self) -> None:
    """Root callable failure: finally_ handler runs with None root."""
    cleanup = []

    def bad_root():
      raise ValueError('root boom')

    c = Q(bad_root).finally_(lambda rv: cleanup.append(rv))
    with self.assertRaises(ValueError):
      c.run()
    self.assertEqual(cleanup, [None])  # normalized to None


# ---------------------------------------------------------------------------
# §8.1: Escaped control flow signal
# ---------------------------------------------------------------------------


class EscapedControlFlowTest(TestCase):
  """SPEC §8.1: Escaped control flow signal → QuentException."""

  def test_escaped_break_raises_quent_exception(self) -> None:
    """break_() escaping run() → QuentException."""
    c = Q(1).then(lambda x: Q.break_())
    with self.assertRaises(QuentException):
      c.run()

  def test_return_at_top_level_produces_value(self) -> None:
    """return_() at top level: run() catches it and returns the value."""
    result = Q(5).then(lambda x: Q.return_(x * 2)).run()
    self.assertEqual(result, 10)


# ---------------------------------------------------------------------------
# §8.2 __call__
# ---------------------------------------------------------------------------


class CallAliasTest(SymmetricTestCase):
  """SPEC §8.2: __call__ is alias for run()."""

  async def test_call_equals_run(self) -> None:
    """chain(v) == chain.run(v)."""
    c = Q(0).then(lambda x: x + 1)
    self.assertEqual(c(5), c.run(5))
    self.assertEqual(c(5), 6)

  async def test_call_no_args(self) -> None:
    """chain() == chain.run()."""
    c = Q(5).then(lambda x: x * 2)
    self.assertEqual(c(), c.run())
    self.assertEqual(c(), 10)

  async def test_call_forwards_args_and_kwargs(self) -> None:
    """chain(callable, *args, **kwargs) forwards to run().

    Per SPEC §8.2: __call__ is alias for run(). Per _chain.py:295-297,
    __call__ forwards (v, *args, **kwargs) to run().
    """
    received = {}

    def root_fn(a, *, key):
      received['a'] = a
      received['key'] = key
      return a

    c = Q().then(lambda x: x + 1)
    result = c(root_fn, 10, key='hello')
    self.assertEqual(result, 11)  # root_fn(10, key='hello') = 10, then +1 = 11
    self.assertEqual(received, {'a': 10, 'key': 'hello'})


# ---------------------------------------------------------------------------
# §8.3 Sync/Async Execution Model
# ---------------------------------------------------------------------------


class SyncAsyncModelTest(SymmetricTestCase):
  """SPEC §8.3: Sync/async execution model."""

  async def test_fully_sync_returns_plain_value(self) -> None:
    """Fully sync pipeline returns plain value (not coroutine)."""
    result = Q(5).then(lambda x: x + 1).run()
    self.assertFalse(asyncio.iscoroutine(result))
    self.assertEqual(result, 6)

  async def test_any_async_step_returns_coroutine(self) -> None:
    """Pipeline with async step returns coroutine from run()."""

    result = Q(5).then(async_fn).run()
    self.assertTrue(asyncio.iscoroutine(result))
    value = await result
    self.assertEqual(value, 6)

  async def test_transition_is_one_way(self) -> None:
    """Once async, stays async — no back to sync."""

    # After async_fn, the sync lambda still runs in async mode
    result = await Q(5).then(async_fn).then(lambda x: x * 2).run()
    self.assertEqual(result, 12)  # 5→6→12

  async def test_transition_at_any_position(self) -> None:
    """Async transition can happen at any position."""

    # Middle
    async def async_mid(x):
      return x * 2

    result = await Q(5).then(lambda x: x + 1).then(async_mid).then(lambda x: x - 1).run()
    self.assertEqual(result, 11)  # 5→6→12→11

  async def test_fully_sync_no_event_loop(self) -> None:
    """Fully sync pipeline has zero async overhead."""
    # If no async step, run() returns a plain value, never a coroutine
    result = Q(1).then(lambda x: x + 1).then(lambda x: x + 1).run()
    self.assertNotIsInstance(result, asyncio.Future)
    self.assertFalse(asyncio.iscoroutine(result))
    self.assertEqual(result, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class RunEdgeCasesTest(SymmetricTestCase):
  """Edge cases for run() / execution."""

  async def test_none_root_vs_no_root(self) -> None:
    """Q(None) vs Q(): different behaviors."""
    # Q(None) has root value None
    r1 = Q(None).run()
    self.assertIsNone(r1)
    # Q() has no root value — result is also None
    r2 = Q().run()
    self.assertIsNone(r2)

  async def test_chain_none_root_passed_to_step(self) -> None:
    """Q(None) passes None to first step."""
    result = Q(None).then(lambda x: x is None).run()
    self.assertTrue(result)

  async def test_do_step_discards_result(self) -> None:
    """do() step discards its result."""
    result = Q(5).do(lambda x: x * 100).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)  # 5 passed through do, then +1

  async def test_non_callable_then_replaces_value(self) -> None:
    """Non-callable in then() replaces current value."""
    result = Q(1).then(2).then(3).run()
    self.assertEqual(result, 3)

  async def test_run_with_root_callable_and_args(self) -> None:
    """Q(fn, *args) evaluates fn(*args) as root."""
    result = Q(lambda a, b: a + b, 3, 7).then(lambda x: x * 2).run()
    self.assertEqual(result, 20)  # 3+7=10, 10*2=20

  async def test_run_value_callable_is_evaluated(self) -> None:
    """run(fn) evaluates fn() as the run value."""
    result = Q(100).then(lambda x: x + 1).run(lambda: 5)
    self.assertEqual(result, 6)  # run value 5, +1=6, root 100 ignored


class AsyncRunTest(SymmetricTestCase):
  """Async-specific run() tests."""

  async def test_async_run_with_multiple_async_steps(self) -> None:
    """Multiple async steps all execute correctly."""

    async def step1(x):
      return x + 1

    async def step2(x):
      return x * 2

    async def step3(x):
      return x - 3

    result = await Q(5).then(step1).then(step2).then(step3).run()
    # 5→6→12→9
    self.assertEqual(result, 9)

  async def test_async_run_mixed_sync_async(self) -> None:
    """Mixed sync and async steps execute correctly."""

    result = await Q(1).then(lambda x: x + 1).then(async_double).then(lambda x: x + 3).run()
    # 1→2→4→7
    self.assertEqual(result, 7)

  async def test_async_run_value_replaces_root(self) -> None:
    """Async: run value replaces root."""

    result = await Q(100).then(async_double).run(5)
    self.assertEqual(result, 10)

  async def test_async_root_callable_failure(self) -> None:
    """Async: root callable failure triggers error handling."""
    received = {}

    async def bad_root():
      raise ValueError('async root boom')

    def handler(info: QuentExcInfo):
      received['exc'] = info.exc
      received['rv'] = info.root_value
      return 'handled'

    result = await Q(bad_root).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertIsInstance(received['exc'], ValueError)
    self.assertIsNone(received['rv'])


# ---------------------------------------------------------------------------
# §8.1: run() non-callable + args/kwargs → TypeError
# ---------------------------------------------------------------------------


class RunNonCallableWithArgsTest(TestCase):
  """SPEC §8.1: run() non-callable v + args/kwargs raises TypeError."""

  def test_run_non_callable_with_args(self) -> None:
    """run(42, 'extra') → TypeError."""
    c = Q().then(lambda x: x)
    with self.assertRaises(TypeError):
      c.run(42, 'extra')

  def test_run_non_callable_with_kwargs(self) -> None:
    """run(42, key='val') → TypeError."""
    c = Q().then(lambda x: x)
    with self.assertRaises(TypeError):
      c.run(42, key='val')

  def test_run_non_callable_with_both(self) -> None:
    """run(42, 'extra', key='val') → TypeError."""
    c = Q().then(lambda x: x)
    with self.assertRaises(TypeError):
      c.run(42, 'extra', key='val')

  def test_run_kwargs_without_root_raises_typeerror(self) -> None:
    """SPEC §8.1: run(key=val) without positional root raises TypeError."""
    c = Q().then(lambda x: x)
    with self.assertRaises(TypeError):
      c.run(key='val')

  def test_run_callable_with_args_ok(self) -> None:
    """run(fn, arg) succeeds when v is callable."""
    result = Q().then(lambda x: x + 1).run(lambda a, b: a + b, 3, 4)
    self.assertEqual(result, 8)  # 3+4=7, 7+1=8


# ---------------------------------------------------------------------------
# Async execution paths
# ---------------------------------------------------------------------------


class AsyncTransitionDuringStepTest(SymmetricTestCase):
  """Async transition during step evaluation."""

  async def test_async_root_callable_with_args(self) -> None:
    """Async root callable with args evaluates correctly."""

    async def root(a, b):
      return a + b

    result = await Q(root, 3, 7).then(lambda x: x * 2).run()
    self.assertEqual(result, 20)  # 3+7=10, 10*2=20

  async def test_async_transition_at_first_step(self) -> None:
    """Async transition at the very first step."""

    async def first(x):
      return x * 2

    result = await Q(5).then(first).then(lambda x: x + 1).run()
    self.assertEqual(result, 11)  # 5*2=10, 10+1=11

  async def test_async_transition_at_last_step(self) -> None:
    """Async transition at the very last step."""

    async def last(x):
      return x * 3

    result = await Q(5).then(lambda x: x + 1).then(last).run()
    self.assertEqual(result, 18)  # 5+1=6, 6*3=18

  async def test_multiple_async_steps_interleaved(self) -> None:
    """Multiple async steps interleaved with sync steps."""

    async def async_add(x):
      return x + 10

    async def async_mul(x):
      return x * 2

    result = await (
      Q(1)
      .then(lambda x: x + 1)  # sync: 2
      .then(async_add)  # async: 12
      .then(lambda x: x - 2)  # sync: 10
      .then(async_mul)  # async: 20
      .then(lambda x: x + 5)  # sync: 25
      .run()
    )
    self.assertEqual(result, 25)


class AsyncControlFlowTest(SymmetricTestCase):
  """Async control flow edge cases."""

  async def test_async_return_skips_remaining_steps(self) -> None:
    """Async return_() skips all remaining steps."""
    visited = []

    async def step1(x):
      visited.append(1)
      return Q.return_(x * 2)

    result = await (
      Q(5).then(step1).then(lambda x: (visited.append(2), x)[-1]).then(lambda x: (visited.append(3), x)[-1]).run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(visited, [1])

  async def test_async_nested_chain_propagation(self) -> None:
    """Async nested chain value flows to outer chain."""

    async def inner_step(x):
      return x * 3

    inner = Q().then(inner_step)
    result = await Q(5).then(inner).then(lambda x: x + 1).run()
    self.assertEqual(result, 16)  # 5*3=15, 15+1=16

  async def test_async_return_with_callable(self) -> None:
    """Async: return_() with callable value."""

    async def step(x):
      return Q.return_(lambda: x * 10)

    result = await Q(5).then(step).then(lambda x: 'never').run()
    self.assertEqual(result, 50)

  async def test_async_run_value_replaces_root_with_callable(self) -> None:
    """Async: run(callable) evaluates callable as run value."""

    async def step(x):
      return x * 2

    result = await Q(100).then(step).run(lambda: 5)
    self.assertEqual(result, 10)  # run value 5, 5*2=10

  async def test_do_with_async_step(self) -> None:
    """do() with async step discards result, passes through value."""

    async def side_effect(x):
      return x * 100  # result discarded

    result = await Q(5).do(side_effect).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)  # 5 passed through do, then +1


# ---------------------------------------------------------------------------
# Audit §9 — Additional spec gap tests
# ---------------------------------------------------------------------------


class RunCallableWithArgsTest(TestCase):
  """SPEC §8.1: run(callable, *args, **kwargs) forwards args to callable."""

  def test_run_callable_with_positional_args(self) -> None:
    """run(fn, a, b) → fn(a, b) used as initial value."""
    result = Q().then(lambda x: x * 2).run(lambda a, b: a + b, 3, 7)
    self.assertEqual(result, 20)  # (3+7)*2=20

  def test_run_callable_with_kwargs(self) -> None:
    """run(fn, key=val) → fn(key=val) used as initial value."""
    result = Q().then(lambda x: x + 1).run(lambda key=0: key * 3, key=5)
    self.assertEqual(result, 16)  # (5*3)+1=16

  def test_run_callable_with_mixed_args(self) -> None:
    """run(fn, a, key=val) → fn(a, key=val)."""
    result = Q().then(lambda x: x).run(lambda a, b=10: a + b, 5, b=20)
    self.assertEqual(result, 25)  # 5+20=25
