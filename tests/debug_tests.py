# SPDX-License-Identifier: MIT
"""Tests for Q.debug() — execution tracing."""

from __future__ import annotations

import asyncio
import io
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Q, QuentException
from quent._debug import DebugResult, StepRecord
from tests.fixtures import (
  V_DOUBLE,
  V_FN,
  V_HANDLER,
  V_IDENTITY,
  V_NOOP,
  V_RAISE,
  V_TRIPLE,
)
from tests.symmetric import SymmetricTestCase


class DebugBasicTest(TestCase):
  """debug() returns a DebugResult with the correct pipeline value."""

  def test_debug_returns_debug_result(self):
    """debug() returns a DebugResult instance."""
    dr = Q(42).debug()
    self.assertIsInstance(dr, DebugResult)

  def test_debug_value_simple(self):
    """debug().value matches the normal run() result."""
    dr = Q(5).then(lambda x: x * 2).debug()
    self.assertEqual(dr.value, 10)

  def test_debug_value_matches_run(self):
    """debug().value is identical to run() for the same pipeline."""
    c = Q(3).then(lambda x: x + 1).then(str)
    self.assertEqual(c.debug().value, c.run())

  def test_debug_with_run_value(self):
    """debug(v) passes a run value like run(v)."""
    c = Q().then(lambda x: x * 3)
    dr = c.debug(7)
    self.assertEqual(dr.value, 21)

  def test_debug_with_callable_run_value(self):
    """debug(fn, *args) works like run(fn, *args)."""
    c = Q().then(lambda x: x + 10)
    dr = c.debug(lambda a, b: a + b, 2, 3)
    self.assertEqual(dr.value, 15)


class DebugStepCaptureTest(TestCase):
  """debug() captures step records for each pipeline step."""

  def test_steps_captured(self):
    """Steps list is populated with StepRecord instances."""
    dr = Q(10).then(lambda x: x + 1).debug()
    self.assertTrue(len(dr.steps) > 0)
    for s in dr.steps:
      self.assertIsInstance(s, StepRecord)

  def test_root_step_present(self):
    """Root value appears as step_name='root'."""
    dr = Q(42).debug()
    root_steps = [s for s in dr.steps if s.step_name == 'root']
    self.assertEqual(len(root_steps), 1)
    self.assertEqual(root_steps[0].result, 42)

  def test_then_step_present(self):
    """then() step appears with step_name='then'."""
    dr = Q(10).then(lambda x: x * 2).debug()
    then_steps = [s for s in dr.steps if s.step_name == 'then']
    self.assertEqual(len(then_steps), 1)
    self.assertEqual(then_steps[0].input_value, 10)
    self.assertEqual(then_steps[0].result, 20)

  def test_do_step_present(self):
    """do() step appears with step_name='do'."""
    side_effects = []
    dr = Q(5).do(lambda x: side_effects.append(x)).debug()
    do_steps = [s for s in dr.steps if s.step_name == 'do']
    self.assertEqual(len(do_steps), 1)

  def test_multiple_steps(self):
    """Multiple pipeline steps are all captured in order."""
    dr = Q(1).then(lambda x: x + 1).then(lambda x: x * 3).then(str).debug()
    step_names = [s.step_name for s in dr.steps]
    self.assertEqual(step_names, ['root', 'then', 'then', 'then'])

  def test_step_ok_property(self):
    """StepRecord.ok is True for successful steps."""
    dr = Q(1).then(lambda x: x + 1).debug()
    for s in dr.steps:
      self.assertTrue(s.ok)


class DebugElapsedTest(TestCase):
  """debug() populates elapsed_ns timing."""

  def test_elapsed_ns_positive(self):
    """Total elapsed_ns is a positive integer."""
    dr = Q(1).then(lambda x: x + 1).debug()
    self.assertIsInstance(dr.elapsed_ns, int)
    self.assertGreater(dr.elapsed_ns, 0)

  def test_step_elapsed_ns_populated(self):
    """Each step's elapsed_ns is a non-negative integer."""
    dr = Q(1).then(lambda x: x + 1).debug()
    for s in dr.steps:
      self.assertIsInstance(s.elapsed_ns, int)
      self.assertGreaterEqual(s.elapsed_ns, 0)


class DebugImmutabilityTest(TestCase):
  """debug() does not modify the original pipeline."""

  def test_original_unchanged(self):
    """Original pipeline produces the same result after debug()."""
    c = Q(5).then(lambda x: x * 2)
    c.debug()  # Should not affect c
    self.assertEqual(c.run(), 10)

  def test_original_class_unchanged(self):
    """Original pipeline's class is still Q, not _DebugChain."""
    c = Q(5).then(lambda x: x + 1)
    c.debug()
    self.assertIs(type(c), Q)

  def test_original_on_step_unaffected(self):
    """Q.on_step remains None after debug()."""
    c = Q(5).then(lambda x: x + 1)
    c.debug()
    self.assertIsNone(Q.on_step)


class DebugSucceededFailedTest(TestCase):
  """succeeded/failed properties reflect step outcomes."""

  def test_succeeded_all_ok(self):
    """succeeded is True when all steps complete without error."""
    dr = Q(1).then(lambda x: x + 1).debug()
    self.assertTrue(dr.succeeded)
    self.assertFalse(dr.failed)

  def test_failed_with_except(self):
    """failed is True when a step raises and except_ catches it."""
    dr = Q(1).then(lambda x: 1 / 0).except_(lambda exc_info: 'recovered').debug()
    self.assertTrue(dr.failed)
    self.assertFalse(dr.succeeded)
    # The pipeline recovers, so value should be 'recovered'
    self.assertEqual(dr.value, 'recovered')


class DebugFailureTest(TestCase):
  """debug() with failing steps captures exception info."""

  def test_exception_step_captured(self):
    """A step that raises has exception set in its StepRecord."""
    dr = Q(1).then(lambda x: 1 / 0).except_(lambda exc_info: 'caught').debug()
    failing = [s for s in dr.steps if not s.ok]
    self.assertTrue(len(failing) >= 1)
    self.assertIsInstance(failing[0].exception, ZeroDivisionError)

  def test_exception_propagates(self):
    """debug() re-raises if the pipeline has no except_ handler."""
    with self.assertRaises(ZeroDivisionError):
      Q(1).then(lambda x: 1 / 0).debug()


class DebugPrintTraceTest(TestCase):
  """print_trace() produces formatted output."""

  def test_print_trace_to_custom_file(self):
    """print_trace(file=...) writes to the specified stream."""
    dr = Q(5).then(lambda x: x * 2).debug()
    buf = io.StringIO()
    dr.print_trace(file=buf)
    output = buf.getvalue()
    self.assertIn('root', output)
    self.assertIn('then', output)
    self.assertIn('Total:', output)

  def test_print_trace_contains_table_structure(self):
    """print_trace() output contains table delimiters."""
    dr = Q(1).then(lambda x: x + 1).debug()
    buf = io.StringIO()
    dr.print_trace(file=buf)
    output = buf.getvalue()
    # Should contain header row with column names
    self.assertIn('Step', output)
    self.assertIn('Status', output)
    self.assertIn('Elapsed', output)
    # Should contain table borders
    self.assertIn('+', output)
    self.assertIn('|', output)

  def test_print_trace_shows_ok_status(self):
    """Successful steps show OK status."""
    dr = Q(1).then(lambda x: x + 1).debug()
    buf = io.StringIO()
    dr.print_trace(file=buf)
    self.assertIn('OK', buf.getvalue())

  def test_print_trace_shows_fail_status(self):
    """Failed steps show FAIL status."""
    dr = Q(1).then(lambda x: 1 / 0).except_(lambda exc_info: 'caught').debug()
    buf = io.StringIO()
    dr.print_trace(file=buf)
    self.assertIn('FAIL', buf.getvalue())


class DebugAsyncTest(IsolatedAsyncioTestCase):
  """debug() handles async pipelines transparently."""

  async def test_async_debug_returns_debug_result(self):
    """debug() with async steps returns a DebugResult after await."""

    async def async_double(x: int) -> int:
      return x * 2

    dr = await Q(5).then(async_double).debug()
    self.assertIsInstance(dr, DebugResult)
    self.assertEqual(dr.value, 10)

  async def test_async_steps_captured(self):
    """Steps are captured for async pipelines."""

    async def async_add(x: int) -> int:
      return x + 1

    dr = await Q(1).then(async_add).then(lambda x: x * 3).debug()
    step_names = [s.step_name for s in dr.steps]
    self.assertIn('root', step_names)
    self.assertIn('then', step_names)

  async def test_async_elapsed_ns(self):
    """elapsed_ns is populated for async pipelines."""

    async def async_identity(x: int) -> int:
      return x

    dr = await Q(1).then(async_identity).debug()
    self.assertIsInstance(dr.elapsed_ns, int)
    self.assertGreater(dr.elapsed_ns, 0)

  async def test_async_with_run_value(self):
    """debug(v) works for async pipelines."""

    async def async_triple(x: int) -> int:
      return x * 3

    c = Q().then(async_triple)
    dr = await c.debug(4)
    self.assertEqual(dr.value, 12)


# ---------------------------------------------------------------------------
# Symmetric tests — sync/async bridge verification for debug()
# ---------------------------------------------------------------------------


def _comparable_dr(dr: DebugResult) -> tuple:
  """Extract comparable fields from DebugResult, excluding timing."""
  return (
    dr.value,
    [(s.step_name, s.input_value, s.result, s.ok) for s in dr.steps],
    dr.succeeded,
    dr.failed,
  )


async def _debug_q(q: Q) -> DebugResult:
  """Run debug() on a pipeline, handling both sync and async returns."""
  result = q.debug()
  if asyncio.iscoroutine(result):
    result = await result
  return result


async def _debug_q_with_rv(q: Q, v, *args, **kwargs) -> DebugResult:
  """Run debug(v, ...) on a pipeline, handling both sync and async returns."""
  result = q.debug(v, *args, **kwargs)
  if asyncio.iscoroutine(result):
    result = await result
  return result


class DebugValueSymmetryTest(SymmetricTestCase):
  """Symmetric: debug().value matches across sync/async fn variants."""

  async def test_basic_value_symmetry(self) -> None:
    """debug().value is the same whether the step is sync or async."""

    async def builder(fn):
      dr = await _debug_q(Q(5).then(fn))
      return dr.value

    await self.variant(
      builder,
      fn=V_DOUBLE,
      expected=10,
    )

  async def test_multistep_value_symmetry(self) -> None:
    """debug().value matches across sync/async for multi-step pipelines."""

    async def builder(fn1, fn2):
      dr = await _debug_q(Q(5).then(fn1).then(fn2))
      return dr.value

    await self.variant(
      builder,
      fn1=V_FN,  # +1 -> 6
      fn2=V_DOUBLE,  # *2 -> 12
      expected=12,
    )

  async def test_identity_value_symmetry(self) -> None:
    """debug().value with identity fn matches across sync/async."""

    async def builder(fn):
      dr = await _debug_q(Q(42).then(fn))
      return dr.value

    await self.variant(
      builder,
      fn=V_IDENTITY,
      expected=42,
    )


class DebugStepSymmetryTest(SymmetricTestCase):
  """Symmetric: debug step metadata matches across sync/async."""

  async def test_step_count_symmetry(self) -> None:
    """len(debug().steps) is the same across sync/async variants."""

    async def builder(fn):
      dr = await _debug_q(Q(5).then(fn))
      return len(dr.steps)

    await self.variant(
      builder,
      fn=V_DOUBLE,
      expected=2,  # root + then
    )

  async def test_step_names_symmetry(self) -> None:
    """Step names list is the same across sync/async variants."""

    async def builder(fn1, fn2):
      dr = await _debug_q(Q(1).then(fn1).then(fn2))
      return [s.step_name for s in dr.steps]

    await self.variant(
      builder,
      fn1=V_FN,
      fn2=V_DOUBLE,
      expected=['root', 'then', 'then'],
    )

  async def test_step_values_symmetry(self) -> None:
    """Step input/result values are the same across sync/async variants."""

    async def builder(fn1, fn2):
      dr = await _debug_q(Q(5).then(fn1).then(fn2))
      return [(s.step_name, s.input_value, s.result) for s in dr.steps]

    await self.variant(
      builder,
      fn1=V_FN,  # +1: input=5, result=6
      fn2=V_DOUBLE,  # *2: input=6, result=12
      expected=[('root', None, 5), ('then', 5, 6), ('then', 6, 12)],
    )

  async def test_step_ok_symmetry(self) -> None:
    """step.ok flags are the same across sync/async variants."""

    async def builder(fn):
      dr = await _debug_q(Q(5).then(fn))
      return [s.ok for s in dr.steps]

    await self.variant(
      builder,
      fn=V_DOUBLE,
      expected=[True, True],
    )


class DebugSucceededFailedSymmetryTest(SymmetricTestCase):
  """Symmetric: succeeded/failed properties match across sync/async."""

  async def test_succeeded_symmetry(self) -> None:
    """succeeded is True across all sync/async variants for a successful pipeline."""

    async def builder(fn):
      dr = await _debug_q(Q(5).then(fn))
      return (dr.succeeded, dr.failed)

    await self.variant(
      builder,
      fn=V_DOUBLE,
      expected=(True, False),
    )

  async def test_failed_with_except_symmetry(self) -> None:
    """failed is True across all sync/async variants when except_ catches error."""

    async def builder(raiser, handler):
      dr = await _debug_q(Q(1).then(raiser).except_(handler))
      return (dr.value, dr.succeeded, dr.failed)

    await self.variant(
      builder,
      raiser=V_RAISE,
      handler=V_HANDLER,
      expected=('handled', False, True),
    )


class DebugRunValueSymmetryTest(SymmetricTestCase):
  """Symmetric: debug(v) run value override matches across sync/async."""

  async def test_debug_run_value_symmetry(self) -> None:
    """debug(v) produces the same value across sync/async fn variants."""

    async def builder(fn):
      dr = await _debug_q_with_rv(Q().then(fn), 7)
      return dr.value

    await self.variant(
      builder,
      fn=V_TRIPLE,
      expected=21,
    )

  async def test_debug_callable_run_value_symmetry(self) -> None:
    """debug(callable, *args) produces the same value across sync/async."""

    async def builder(fn):
      dr = await _debug_q_with_rv(Q().then(fn), lambda a, b: a + b, 2, 3)
      return dr.value

    await self.variant(
      builder,
      fn=V_FN,  # +1: (2+3)=5, +1=6
      expected=6,
    )

  async def test_debug_run_value_replaces_root_symmetry(self) -> None:
    """debug(v) replaces root value, consistent across sync/async."""

    async def builder(fn):
      dr = await _debug_q_with_rv(Q(100).then(fn), 5)
      return dr.value

    await self.variant(
      builder,
      fn=V_DOUBLE,
      expected=10,  # run value 5, *2=10, root 100 ignored
    )


class DebugDoStepSymmetryTest(SymmetricTestCase):
  """Symmetric: debug with do() steps matches across sync/async."""

  async def test_do_step_value_passthrough_symmetry(self) -> None:
    """do() step does not affect final value, consistent across sync/async."""

    async def builder(fn):
      dr = await _debug_q(Q(5).do(fn).then(lambda x: x + 1))
      return dr.value

    await self.variant(
      builder,
      fn=V_NOOP,
      expected=6,  # 5 passes through do, then +1=6
    )

  async def test_do_step_name_symmetry(self) -> None:
    """do() step appears with step_name='do' across sync/async."""

    async def builder(fn):
      dr = await _debug_q(Q(5).do(fn))
      return [s.step_name for s in dr.steps]

    await self.variant(
      builder,
      fn=V_NOOP,
      expected=['root', 'do'],
    )


class DebugFullResultSymmetryTest(SymmetricTestCase):
  """Symmetric: full comparable DebugResult matches across sync/async."""

  async def test_full_comparable_symmetry(self) -> None:
    """Full _comparable_dr output is the same across sync/async variants."""

    async def builder(fn1, fn2):
      dr = await _debug_q(Q(5).then(fn1).then(fn2))
      return _comparable_dr(dr)

    await self.variant(
      builder,
      fn1=V_FN,  # +1: 5->6
      fn2=V_DOUBLE,  # *2: 6->12
      expected=(
        12,
        [
          ('root', None, 5, True),
          ('then', 5, 6, True),
          ('then', 6, 12, True),
        ],
        True,
        False,
      ),
    )

  async def test_full_comparable_with_error_symmetry(self) -> None:
    """Full comparable result with error recovery matches across sync/async."""

    async def builder(raiser, handler):
      dr = await _debug_q(Q(1).then(raiser).except_(handler))
      # Extract only value, step names, succeeded/failed -- skip step values
      # since except_ handler input differs (QuentExcInfo contains the exc object)
      return (
        dr.value,
        [s.step_name for s in dr.steps],
        dr.succeeded,
        dr.failed,
      )

    await self.variant(
      builder,
      raiser=V_RAISE,
      handler=V_HANDLER,
      expected=(
        'handled',
        ['root', 'then', 'except_'],
        False,
        True,
      ),
    )

  async def test_three_step_pipeline_symmetry(self) -> None:
    """Three-step pipeline debug matches across all sync/async permutations."""

    async def builder(fn1, fn2, fn3):
      dr = await _debug_q(Q(2).then(fn1).then(fn2).then(fn3))
      return _comparable_dr(dr)

    await self.variant(
      builder,
      fn1=V_FN,  # +1: 2->3
      fn2=V_DOUBLE,  # *2: 3->6
      fn3=V_TRIPLE,  # *3: 6->18
      expected=(
        18,
        [
          ('root', None, 2, True),
          ('then', 2, 3, True),
          ('then', 3, 6, True),
          ('then', 6, 18, True),
        ],
        True,
        False,
      ),
    )


# ---------------------------------------------------------------------------
# §14.6: debug() build-time constraints — pending if_/while_ rejection
# ---------------------------------------------------------------------------


class DebugPendingIfWhileTest(TestCase):
  """§14.6: debug() while if_() or while_() is pending raises QuentException."""

  def test_debug_pending_if_raises(self):
    """§14.6: debug() while if_() is pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: x > 0).debug()

  def test_debug_pending_while_raises(self):
    """§14.6: debug() while while_() is pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).while_(lambda x: x > 0).debug()


# ---------------------------------------------------------------------------
# §14.6: debug() captures step records for while_ and drive_gen
# ---------------------------------------------------------------------------


class DebugWhileStepCaptureTest(TestCase):
  """§14.6: debug() captures while_ step records correctly."""

  def test_debug_while_step_present(self):
    """while_() step appears with step_name='while_' in debug output."""
    dr = Q(10).while_().then(lambda x: x - 1).debug()
    self.assertIsInstance(dr, DebugResult)
    self.assertEqual(dr.value, 0)
    step_names = [s.step_name for s in dr.steps]
    self.assertIn('root', step_names)
    self.assertIn('while_', step_names)
    while_steps = [s for s in dr.steps if s.step_name == 'while_']
    self.assertEqual(len(while_steps), 1)
    self.assertEqual(while_steps[0].input_value, 10)
    self.assertEqual(while_steps[0].result, 0)
    self.assertTrue(while_steps[0].ok)

  def test_debug_while_with_predicate(self):
    """while_(predicate) step captured correctly in debug output."""
    dr = Q(100).while_(lambda x: x > 1).then(lambda x: x // 2).debug()
    self.assertIsInstance(dr, DebugResult)
    self.assertEqual(dr.value, 1)
    while_steps = [s for s in dr.steps if s.step_name == 'while_']
    self.assertEqual(len(while_steps), 1)
    self.assertEqual(while_steps[0].result, 1)

  def test_debug_while_succeeded(self):
    """debug() with while_: succeeded=True, failed=False."""
    dr = Q(3).while_().then(lambda x: x - 1).debug()
    self.assertTrue(dr.succeeded)
    self.assertFalse(dr.failed)


class DebugDriveGenStepCaptureTest(TestCase):
  """§14.6: debug() captures drive_gen step records correctly."""

  def test_debug_drive_gen_step_present(self):
    """drive_gen() step appears with step_name='drive_gen' in debug output."""

    def gen():
      x = yield 1
      yield x + 1

    dr = Q(gen).drive_gen(lambda x: x * 2).debug()
    self.assertIsInstance(dr, DebugResult)
    step_names = [s.step_name for s in dr.steps]
    self.assertIn('drive_gen', step_names)
    dg_steps = [s for s in dr.steps if s.step_name == 'drive_gen']
    self.assertEqual(len(dg_steps), 1)
    self.assertTrue(dg_steps[0].ok)

  def test_debug_drive_gen_value_correct(self):
    """drive_gen() debug value matches normal run() result."""

    def gen():
      x = yield 42
      yield x

    c = Q(gen).drive_gen(lambda x: x * 2)
    dr = c.debug()
    self.assertEqual(dr.value, c.run())

  def test_debug_drive_gen_succeeded(self):
    """debug() with drive_gen: succeeded=True, failed=False."""

    def gen():
      yield 5

    dr = Q(gen).drive_gen(lambda x: x + 1).debug()
    self.assertTrue(dr.succeeded)
    self.assertFalse(dr.failed)


class DebugWhileAsyncTest(IsolatedAsyncioTestCase):
  """§14.6: debug() captures while_ step records in async pipelines."""

  async def test_async_debug_while_step(self):
    """Async while_ step captured correctly in debug output."""

    async def dec(x):
      return x - 1

    dr = await Q(5).while_().then(dec).debug()
    self.assertIsInstance(dr, DebugResult)
    self.assertEqual(dr.value, 0)
    step_names = [s.step_name for s in dr.steps]
    self.assertIn('while_', step_names)


class DebugDriveGenAsyncTest(IsolatedAsyncioTestCase):
  """§14.6: debug() captures drive_gen step records in async pipelines."""

  async def test_async_debug_drive_gen_step(self):
    """Async drive_gen step captured correctly in debug output."""

    def gen():
      x = yield 10
      yield x + 1

    async def async_double(x):
      return x * 2

    dr = await Q(gen).drive_gen(async_double).debug()
    self.assertIsInstance(dr, DebugResult)
    step_names = [s.step_name for s in dr.steps]
    self.assertIn('drive_gen', step_names)
