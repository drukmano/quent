# SPDX-License-Identifier: MIT
"""Tests for Chain.debug() — execution tracing."""

from __future__ import annotations

import io
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Chain
from quent._debug import DebugResult, StepRecord


class DebugBasicTest(TestCase):
  """debug() returns a DebugResult with the correct pipeline value."""

  def test_debug_returns_debug_result(self):
    """debug() returns a DebugResult instance."""
    dr = Chain(42).debug()
    self.assertIsInstance(dr, DebugResult)

  def test_debug_value_simple(self):
    """debug().value matches the normal run() result."""
    dr = Chain(5).then(lambda x: x * 2).debug()
    self.assertEqual(dr.value, 10)

  def test_debug_value_matches_run(self):
    """debug().value is identical to run() for the same chain."""
    c = Chain(3).then(lambda x: x + 1).then(str)
    self.assertEqual(c.debug().value, c.run())

  def test_debug_with_run_value(self):
    """debug(v) passes a run value like run(v)."""
    c = Chain().then(lambda x: x * 3)
    dr = c.debug(7)
    self.assertEqual(dr.value, 21)

  def test_debug_with_callable_run_value(self):
    """debug(fn, *args) works like run(fn, *args)."""
    c = Chain().then(lambda x: x + 10)
    dr = c.debug(lambda a, b: a + b, 2, 3)
    self.assertEqual(dr.value, 15)


class DebugStepCaptureTest(TestCase):
  """debug() captures step records for each pipeline step."""

  def test_steps_captured(self):
    """Steps list is populated with StepRecord instances."""
    dr = Chain(10).then(lambda x: x + 1).debug()
    self.assertTrue(len(dr.steps) > 0)
    for s in dr.steps:
      self.assertIsInstance(s, StepRecord)

  def test_root_step_present(self):
    """Root value appears as step_name='root'."""
    dr = Chain(42).debug()
    root_steps = [s for s in dr.steps if s.step_name == 'root']
    self.assertEqual(len(root_steps), 1)
    self.assertEqual(root_steps[0].result, 42)

  def test_then_step_present(self):
    """then() step appears with step_name='then'."""
    dr = Chain(10).then(lambda x: x * 2).debug()
    then_steps = [s for s in dr.steps if s.step_name == 'then']
    self.assertEqual(len(then_steps), 1)
    self.assertEqual(then_steps[0].input_value, 10)
    self.assertEqual(then_steps[0].result, 20)

  def test_do_step_present(self):
    """do() step appears with step_name='do'."""
    side_effects = []
    dr = Chain(5).do(lambda x: side_effects.append(x)).debug()
    do_steps = [s for s in dr.steps if s.step_name == 'do']
    self.assertEqual(len(do_steps), 1)

  def test_multiple_steps(self):
    """Multiple pipeline steps are all captured in order."""
    dr = Chain(1).then(lambda x: x + 1).then(lambda x: x * 3).then(str).debug()
    step_names = [s.step_name for s in dr.steps]
    self.assertEqual(step_names, ['root', 'then', 'then', 'then'])

  def test_step_ok_property(self):
    """StepRecord.ok is True for successful steps."""
    dr = Chain(1).then(lambda x: x + 1).debug()
    for s in dr.steps:
      self.assertTrue(s.ok)


class DebugElapsedTest(TestCase):
  """debug() populates elapsed_ns timing."""

  def test_elapsed_ns_positive(self):
    """Total elapsed_ns is a positive integer."""
    dr = Chain(1).then(lambda x: x + 1).debug()
    self.assertIsInstance(dr.elapsed_ns, int)
    self.assertGreater(dr.elapsed_ns, 0)

  def test_step_elapsed_ns_populated(self):
    """Each step's elapsed_ns is a non-negative integer."""
    dr = Chain(1).then(lambda x: x + 1).debug()
    for s in dr.steps:
      self.assertIsInstance(s.elapsed_ns, int)
      self.assertGreaterEqual(s.elapsed_ns, 0)


class DebugImmutabilityTest(TestCase):
  """debug() does not modify the original chain."""

  def test_original_unchanged(self):
    """Original chain produces the same result after debug()."""
    c = Chain(5).then(lambda x: x * 2)
    c.debug()  # Should not affect c
    self.assertEqual(c.run(), 10)

  def test_original_class_unchanged(self):
    """Original chain's class is still Chain, not _DebugChain."""
    c = Chain(5).then(lambda x: x + 1)
    c.debug()
    self.assertIs(type(c), Chain)

  def test_original_on_step_unaffected(self):
    """Chain.on_step remains None after debug()."""
    c = Chain(5).then(lambda x: x + 1)
    c.debug()
    self.assertIsNone(Chain.on_step)


class DebugSucceededFailedTest(TestCase):
  """succeeded/failed properties reflect step outcomes."""

  def test_succeeded_all_ok(self):
    """succeeded is True when all steps complete without error."""
    dr = Chain(1).then(lambda x: x + 1).debug()
    self.assertTrue(dr.succeeded)
    self.assertFalse(dr.failed)

  def test_failed_with_except(self):
    """failed is True when a step raises and except_ catches it."""
    dr = Chain(1).then(lambda x: 1 / 0).except_(lambda exc_info: 'recovered').debug()
    self.assertTrue(dr.failed)
    self.assertFalse(dr.succeeded)
    # The chain recovers, so value should be 'recovered'
    self.assertEqual(dr.value, 'recovered')


class DebugFailureTest(TestCase):
  """debug() with failing steps captures exception info."""

  def test_exception_step_captured(self):
    """A step that raises has exception set in its StepRecord."""
    dr = Chain(1).then(lambda x: 1 / 0).except_(lambda exc_info: 'caught').debug()
    failing = [s for s in dr.steps if not s.ok]
    self.assertTrue(len(failing) >= 1)
    self.assertIsInstance(failing[0].exception, ZeroDivisionError)

  def test_exception_propagates(self):
    """debug() re-raises if the chain has no except_ handler."""
    with self.assertRaises(ZeroDivisionError):
      Chain(1).then(lambda x: 1 / 0).debug()


class DebugPrintTraceTest(TestCase):
  """print_trace() produces formatted output."""

  def test_print_trace_to_custom_file(self):
    """print_trace(file=...) writes to the specified stream."""
    dr = Chain(5).then(lambda x: x * 2).debug()
    buf = io.StringIO()
    dr.print_trace(file=buf)
    output = buf.getvalue()
    self.assertIn('root', output)
    self.assertIn('then', output)
    self.assertIn('Total:', output)

  def test_print_trace_contains_table_structure(self):
    """print_trace() output contains table delimiters."""
    dr = Chain(1).then(lambda x: x + 1).debug()
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
    dr = Chain(1).then(lambda x: x + 1).debug()
    buf = io.StringIO()
    dr.print_trace(file=buf)
    self.assertIn('OK', buf.getvalue())

  def test_print_trace_shows_fail_status(self):
    """Failed steps show FAIL status."""
    dr = Chain(1).then(lambda x: 1 / 0).except_(lambda exc_info: 'caught').debug()
    buf = io.StringIO()
    dr.print_trace(file=buf)
    self.assertIn('FAIL', buf.getvalue())


class DebugAsyncTest(IsolatedAsyncioTestCase):
  """debug() handles async pipelines transparently."""

  async def test_async_debug_returns_debug_result(self):
    """debug() with async steps returns a DebugResult after await."""

    async def async_double(x: int) -> int:
      return x * 2

    dr = await Chain(5).then(async_double).debug()
    self.assertIsInstance(dr, DebugResult)
    self.assertEqual(dr.value, 10)

  async def test_async_steps_captured(self):
    """Steps are captured for async pipelines."""

    async def async_add(x: int) -> int:
      return x + 1

    dr = await Chain(1).then(async_add).then(lambda x: x * 3).debug()
    step_names = [s.step_name for s in dr.steps]
    self.assertIn('root', step_names)
    self.assertIn('then', step_names)

  async def test_async_elapsed_ns(self):
    """elapsed_ns is populated for async pipelines."""

    async def async_identity(x: int) -> int:
      return x

    dr = await Chain(1).then(async_identity).debug()
    self.assertIsInstance(dr.elapsed_ns, int)
    self.assertGreater(dr.elapsed_ns, 0)

  async def test_async_with_run_value(self):
    """debug(v) works for async pipelines."""

    async def async_triple(x: int) -> int:
      return x * 3

    c = Chain().then(async_triple)
    dr = await c.debug(4)
    self.assertEqual(dr.value, 12)
