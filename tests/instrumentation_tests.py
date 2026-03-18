# SPDX-License-Identifier: MIT
"""Tests for SPEC §14 — Instrumentation (on_step callback).

Tests cover:
- §14.1: Class-level on_step callback
- §14.1: input_value assertions
- §14.2: Zero overhead when disabled
- §14.3: Error handling (callback raises → logged, chain continues)
- §14.4: Thread safety (subclass overrides via type(chain).on_step)
- §14.5: Debug logging
- §14.5: Async transition debug log
- §14.5: QUENT_TRACEBACK_VALUES=0 debug logging
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import warnings
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Chain, ChainExcInfo
from tests.fixtures import async_double, async_fn, async_identity


class SimpleCM:
  """Simple sync context manager for instrumentation tests."""

  def __enter__(self):
    return 'ctx'

  def __exit__(self, *args):
    return False


class OnStepBasicTest(TestCase):
  """§14.1: on_step callback receives correct arguments after each step."""

  def setUp(self):
    self.calls = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns):
    self.calls.append(
      {
        'chain': chain,
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
      }
    )

  def test_root_step_recorded(self):
    """Root value evaluation calls on_step with step_name='root'."""
    c = Chain(42)
    c.run()
    self.assertTrue(len(self.calls) >= 1)
    root_call = self.calls[0]
    self.assertEqual(root_call['step_name'], 'root')
    self.assertIsNone(root_call['input_value'])
    self.assertEqual(root_call['result'], 42)
    self.assertIsInstance(root_call['elapsed_ns'], int)
    self.assertGreaterEqual(root_call['elapsed_ns'], 0)

  def test_then_step_recorded(self):
    """then() step calls on_step with step_name='then'."""
    Chain(10).then(lambda x: x + 1).run()
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 1)
    self.assertEqual(then_calls[0]['input_value'], 10)
    self.assertEqual(then_calls[0]['result'], 11)

  def test_do_step_recorded(self):
    """do() step calls on_step with step_name='do'."""
    Chain(10).do(lambda x: x * 100).run()
    do_calls = [c for c in self.calls if c['step_name'] == 'do']
    self.assertEqual(len(do_calls), 1)
    self.assertEqual(do_calls[0]['input_value'], 10)

  def test_chain_instance_passed(self):
    """on_step receives the actual Chain instance being executed."""
    c = Chain(5)
    c.run()
    self.assertIs(self.calls[0]['chain'], c)

  def test_elapsed_ns_is_int(self):
    """elapsed_ns is an integer (nanoseconds)."""
    Chain(1).then(lambda x: x).run()
    for call in self.calls:
      self.assertIsInstance(call['elapsed_ns'], int)
      self.assertGreaterEqual(call['elapsed_ns'], 0)

  def test_multiple_steps_all_recorded(self):
    """All steps in a multi-step chain are recorded."""
    Chain(1).then(lambda x: x + 1).then(lambda x: x + 1).do(lambda x: None).run()
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)
    self.assertEqual(step_names.count('then'), 2)
    self.assertEqual(step_names.count('do'), 1)
    # Verify input_value threading: root=None, then(1)->2, then(2)->3, do(3)
    self.assertIsNone(self.calls[0]['input_value'])  # root
    self.assertEqual(self.calls[1]['input_value'], 1)  # first then
    self.assertEqual(self.calls[2]['input_value'], 2)  # second then
    self.assertEqual(self.calls[3]['input_value'], 3)  # do

  def test_map_step_recorded(self):
    """map() step calls on_step with step_name='foreach'."""
    Chain([1, 2, 3]).foreach(lambda x: x * 2).run()
    map_calls = [c for c in self.calls if c['step_name'] == 'foreach']
    self.assertEqual(len(map_calls), 1)
    self.assertEqual(map_calls[0]['input_value'], [1, 2, 3])
    self.assertEqual(map_calls[0]['result'], [2, 4, 6])

  def test_gather_step_recorded(self):
    """gather() step calls on_step with step_name='gather'."""
    Chain(5).gather(lambda x: x + 1, lambda x: x * 2).run()
    gather_calls = [c for c in self.calls if c['step_name'] == 'gather']
    self.assertEqual(len(gather_calls), 1)
    self.assertEqual(gather_calls[0]['input_value'], 5)
    self.assertEqual(gather_calls[0]['result'], (6, 10))

  def test_if_step_recorded(self):
    """if_() step calls on_step with step_name='if_'."""
    Chain(5).if_(lambda x: x > 0).then(lambda x: x * 10).run()
    if_calls = [c for c in self.calls if c['step_name'] == 'if_']
    self.assertEqual(len(if_calls), 1)
    self.assertEqual(if_calls[0]['input_value'], 5)
    self.assertEqual(if_calls[0]['result'], 50)

  def test_else_branch_fires_on_step_with_if_name(self):
    """§14.1: on_step fires with step_name='if_' when the else branch executes."""
    result = Chain(-3).if_(lambda x: x > 0).then(lambda x: x * 10).else_(lambda x: abs(x)).run()
    self.assertEqual(result, 3)
    if_calls = [c for c in self.calls if c['step_name'] == 'if_']
    self.assertEqual(len(if_calls), 1, 'on_step must fire exactly once for the if_/else_ operation')
    self.assertEqual(if_calls[0]['input_value'], -3)
    self.assertEqual(if_calls[0]['result'], 3)

  def test_else_do_branch_fires_on_step_with_if_name(self):
    """§14.1: on_step fires with step_name='if_' when the else_do branch executes."""
    side = []
    result = Chain(-5).if_(lambda x: x > 0).then(lambda x: x * 10).else_do(lambda x: side.append(x)).run()
    self.assertEqual(result, -5)
    self.assertEqual(side, [-5])
    if_calls = [c for c in self.calls if c['step_name'] == 'if_']
    self.assertEqual(len(if_calls), 1, 'on_step must fire exactly once for the if_/else_do operation')
    self.assertEqual(if_calls[0]['input_value'], -5)
    self.assertEqual(if_calls[0]['result'], -5)

  def test_except_handler_does_not_break_on_step(self):
    """on_step still works when except_ handler is present."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda _: 'handled').run()
    self.assertEqual(result, 'handled')
    # Verify root step was recorded at minimum
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)

  def test_finally_handler_does_not_break_on_step(self):
    """on_step still works when finally_ handler is present."""
    result = Chain(1).then(lambda x: x + 1).finally_(lambda x: None).run()
    self.assertEqual(result, 2)
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)
    self.assertIn('then', step_names)

  def test_except_step_name_in_on_step(self):
    """§14.1: on_step fires for except_ handler with step_name='except_'."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda _: 'handled').run()
    self.assertEqual(result, 'handled')
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('except_', step_names)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertEqual(len(except_calls), 1)
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)
    self.assertIsInstance(except_calls[0]['input_value'].exc, ZeroDivisionError)
    self.assertEqual(except_calls[0]['result'], 'handled')
    self.assertIsInstance(except_calls[0]['elapsed_ns'], int)
    self.assertGreaterEqual(except_calls[0]['elapsed_ns'], 0)

  def test_finally_step_name_in_on_step(self):
    """§14.1: on_step fires for finally_ handler with step_name='finally_'."""
    result = Chain(1).then(lambda x: x + 1).finally_(lambda x: None).run()
    self.assertEqual(result, 2)
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('finally_', step_names)
    finally_calls = [c for c in self.calls if c['step_name'] == 'finally_']
    self.assertEqual(len(finally_calls), 1)
    self.assertEqual(finally_calls[0]['input_value'], 1)
    self.assertIsInstance(finally_calls[0]['elapsed_ns'], int)
    self.assertGreaterEqual(finally_calls[0]['elapsed_ns'], 0)

  def test_except_and_finally_both_in_on_step(self):
    """§14.1: on_step fires for both except_ and finally_ in same chain."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda _: 'handled').finally_(lambda x: None).run()
    self.assertEqual(result, 'handled')
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('except_', step_names)
    self.assertIn('finally_', step_names)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)
    finally_calls = [c for c in self.calls if c['step_name'] == 'finally_']
    self.assertEqual(finally_calls[0]['input_value'], 1)


class OnStepDisabledTest(TestCase):
  """§14.2: Zero overhead when on_step is None."""

  def setUp(self):
    Chain.on_step = None

  def tearDown(self):
    Chain.on_step = None

  def test_none_default(self):
    """on_step is None by default."""
    self.assertIsNone(Chain.on_step)

  def test_no_callback_invoked(self):
    """No callback is invoked when on_step is None."""
    called = []
    # Ensure on_step is None
    Chain.on_step = None
    # Run a chain — nothing should break
    result = Chain(5).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)
    self.assertEqual(called, [])


class OnStepErrorHandlingTest(TestCase):
  """§14.3: on_step callback errors don't break chain execution."""

  def setUp(self):
    self.callback_error = RuntimeError('callback broke')
    Chain.on_step = self._failing_callback

  def tearDown(self):
    Chain.on_step = None

  def _failing_callback(self, chain, step_name, input_value, result, elapsed_ns):
    raise self.callback_error

  def test_chain_continues_on_callback_error(self):
    """Chain execution continues when on_step raises."""
    with warnings.catch_warnings(record=True):
      warnings.simplefilter('always')
      result = Chain(5).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)

  def test_runtime_warning_emitted(self):
    """RuntimeWarning is emitted when on_step raises."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(5).then(lambda x: x + 1).run()
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    on_step_warnings = [x for x in runtime_warnings if 'on_step' in str(x.message)]
    self.assertTrue(len(on_step_warnings) > 0, f'Expected on_step warning, got: {[str(x.message) for x in w]}')

  def test_warning_logged(self):
    """on_step error is logged at WARNING level."""
    with self.assertLogs('quent', level='WARNING') as cm, warnings.catch_warnings(record=True):
      warnings.simplefilter('always')
      Chain(5).run()
    log_messages = '\n'.join(cm.output)
    self.assertIn('on_step', log_messages)


class OnStepSubclassTest(TestCase):
  """§14.4: Subclass overrides of on_step are respected."""

  def setUp(self):
    Chain.on_step = None

  def tearDown(self):
    Chain.on_step = None

  def test_subclass_override(self):
    """Subclass can define its own on_step without affecting parent."""
    parent_calls = []
    child_calls = []

    class MyChain(Chain):
      on_step = staticmethod(lambda chain, step_name, input_value, result, elapsed_ns: child_calls.append(step_name))

    Chain.on_step = lambda chain, step_name, input_value, result, elapsed_ns: parent_calls.append(step_name)

    # Parent Chain uses parent callback
    Chain(1).run()
    self.assertIn('root', parent_calls)
    self.assertEqual(child_calls, [])

    parent_calls.clear()

    # Subclass uses its own callback
    MyChain(1).run()
    self.assertIn('root', child_calls)
    self.assertEqual(parent_calls, [])


class OnStepAsyncTest(IsolatedAsyncioTestCase):
  """§14.1: on_step works in async chains too."""

  def setUp(self):
    self.calls = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns):
    self.calls.append(
      {
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
      }
    )

  async def test_async_steps_recorded(self):
    """on_step is called for async steps too."""

    async def async_add(x):
      return x + 1

    result = await Chain(10).then(async_add).run()
    self.assertEqual(result, 11)
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)
    self.assertIn('then', step_names)
    root_call = self.calls[0]
    self.assertIsNone(root_call['input_value'])
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(then_calls[0]['input_value'], 10)

  async def test_async_elapsed_ns_positive(self):
    """elapsed_ns is a non-negative integer for async steps."""

    async def async_fn(x):
      await asyncio.sleep(0.001)
      return x

    await Chain(1).then(async_fn).run()
    for call in self.calls:
      self.assertIsInstance(call['elapsed_ns'], int)
      self.assertGreaterEqual(call['elapsed_ns'], 0)

  async def test_async_step_names_match_builder_methods(self):
    """§14.1: Async step names match the builder method that registered them."""

    await Chain(10).then(async_double).do(async_identity).run()
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)
    self.assertIn('then', step_names)
    self.assertIn('do', step_names)
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(then_calls[0]['input_value'], 10)
    do_calls = [c for c in self.calls if c['step_name'] == 'do']
    self.assertEqual(do_calls[0]['input_value'], 20)

  async def test_async_map_step_name(self):
    """§14.1: map() with async fn reports step_name='foreach'."""

    await Chain([1, 2]).foreach(async_fn).run()
    map_calls = [c for c in self.calls if c['step_name'] == 'foreach']
    self.assertEqual(len(map_calls), 1)
    self.assertEqual(map_calls[0]['input_value'], [1, 2])
    self.assertEqual(map_calls[0]['result'], [2, 3])

  async def test_async_gather_step_name(self):
    """§14.1: gather() with async fns reports step_name='gather'."""

    async def fn_a(x):
      return x + 1

    async def fn_b(x):
      return x * 2

    await Chain(5).gather(fn_a, fn_b).run()
    gather_calls = [c for c in self.calls if c['step_name'] == 'gather']
    self.assertEqual(len(gather_calls), 1)
    self.assertEqual(gather_calls[0]['input_value'], 5)
    self.assertEqual(gather_calls[0]['result'], (6, 10))

  async def test_async_timing_reflects_actual_work(self):
    """§14.1: elapsed_ns for async steps includes await time."""

    async def slow_fn(x):
      await asyncio.sleep(0.05)
      return x

    await Chain(1).then(slow_fn).run()
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 1)
    # 50ms = 50_000_000 ns. Allow generous margin for Windows CI timer jitter.
    self.assertGreater(then_calls[0]['elapsed_ns'], 10_000_000)

  async def test_async_if_step_name(self):
    """§14.1: if_() with async predicate reports step_name='if_'."""

    async def async_pred(x):
      return x > 0

    await Chain(5).if_(async_pred).then(lambda x: x * 10).run()
    if_calls = [c for c in self.calls if c['step_name'] == 'if_']
    self.assertEqual(len(if_calls), 1)
    self.assertEqual(if_calls[0]['input_value'], 5)
    self.assertEqual(if_calls[0]['result'], 50)


class OnStepDebugLoggingTest(TestCase):
  """§14.5: Debug logging at key execution points."""

  def setUp(self):
    Chain.on_step = None

  def tearDown(self):
    Chain.on_step = None

  def test_debug_log_run_started(self):
    """Debug log emits 'run started' message."""
    with self.assertLogs('quent', level='DEBUG') as cm:
      Chain(5).then(lambda x: x + 1).run()
    log_messages = '\n'.join(cm.output)
    self.assertIn('run started', log_messages)

  def test_debug_log_step_completion(self):
    """Debug log shows step completion with result."""
    with self.assertLogs('quent', level='DEBUG') as cm:
      Chain(5).then(lambda x: x + 1).run()
    log_messages = '\n'.join(cm.output)
    # Should contain step name and result info
    self.assertIn('root', log_messages)
    self.assertIn('then', log_messages)

  def test_debug_log_completed(self):
    """Debug log shows chain completion."""
    with self.assertLogs('quent', level='DEBUG') as cm:
      Chain(5).run()
    log_messages = '\n'.join(cm.output)
    self.assertIn('completed', log_messages)

  def test_debug_log_failed(self):
    """Debug log shows chain failure."""
    with self.assertLogs('quent', level='DEBUG') as cm:
      try:
        Chain(1).then(lambda x: 1 / 0).run()
      except ZeroDivisionError:
        pass
    log_messages = '\n'.join(cm.output)
    self.assertIn('failed', log_messages)


class OnStepOperationNamesTest(TestCase):
  """§14.1: on_step reports step_name for foreach_do, with_, with_do."""

  def setUp(self):
    self.calls = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns):
    self.calls.append(step_name)

  def test_foreach_do_step_name(self):
    """§14.1: foreach_do() step calls on_step with step_name='foreach_do'."""
    Chain([1, 2, 3]).foreach_do(lambda x: x).run()
    self.assertIn('foreach_do', self.calls)

  def test_with_step_name(self):
    """§14.1: with_() step calls on_step with step_name='with_'."""

    Chain(SimpleCM()).with_(lambda ctx: ctx).run()
    self.assertIn('with_', self.calls)

  def test_with_do_step_name(self):
    """§14.1: with_do() step calls on_step with step_name='with_do'."""

    Chain(SimpleCM()).with_do(lambda ctx: None).run()
    self.assertIn('with_do', self.calls)


class AsyncTransitionDebugLogTest(IsolatedAsyncioTestCase):
  """§14.5: Debug log emits 'async transition at' message."""

  def setUp(self):
    Chain.on_step = None

  def tearDown(self):
    Chain.on_step = None

  async def test_async_transition_debug_log(self):
    """§14.5: Debug log emits 'async transition at <step_name>' on first awaitable."""

    with self.assertLogs('quent', level='DEBUG') as cm:
      result = await Chain(10).then(async_fn).run()
    self.assertEqual(result, 11)
    log_messages = '\n'.join(cm.output)
    self.assertIn('async transition at', log_messages)

  async def test_async_transition_at_root(self):
    """§14.5: Async transition at root shows 'root' in log."""

    async def async_root():
      return 42

    with self.assertLogs('quent', level='DEBUG') as cm:
      result = await Chain(async_root).run()
    self.assertEqual(result, 42)
    log_messages = '\n'.join(cm.output)
    self.assertIn('async transition at root', log_messages)

  async def test_debug_log_async_failed_at_root(self):
    """§14.5: Debug log shows 'failed at root' when failure occurs at root step in async path."""

    async def bad_root():
      raise ValueError('root boom')

    with self.assertLogs('quent', level='DEBUG') as cm:
      try:
        await Chain(bad_root).run()
      except ValueError:
        pass
    log_messages = '\n'.join(cm.output)
    self.assertIn('failed at root', log_messages)


class DebugLoggingTracbackValuesTest(TestCase):
  """§14.5: QUENT_TRACEBACK_VALUES=0 uses type-name placeholders in debug logs."""

  def setUp(self):
    Chain.on_step = None

  def tearDown(self):
    Chain.on_step = None

  def test_quent_traceback_values_zero_debug_log(self):
    """§14.5: When QUENT_TRACEBACK_VALUES=0, debug log uses type-name placeholders."""
    code = """
import logging
logging.basicConfig(level=logging.DEBUG)
from quent import Chain
Chain(42).then(lambda x: x + 1).run()
"""
    env = os.environ.copy()
    env['QUENT_TRACEBACK_VALUES'] = '0'
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    # With QUENT_TRACEBACK_VALUES=0, debug log should use type-name placeholders
    # like <int> instead of actual values like 42 or 43
    self.assertIn('<int>', result.stderr)


class OnStepAsyncExceptTest(IsolatedAsyncioTestCase):
  """§14.1: on_step callback for except_ handlers in async chains."""

  def setUp(self):
    self.calls: list[dict] = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns):
    self.calls.append(
      {
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
      }
    )

  async def test_async_except_handler_raise_true_on_step(self):
    """§14.1: on_step records except_ for async handler with reraise=True."""

    async def async_handler(info):
      pass  # side-effect only

    c = Chain(1).then(lambda x: 1 / 0).except_(async_handler, reraise=True)
    with self.assertRaises(ZeroDivisionError):
      await c.run()
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('except_', step_names)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)

  async def test_async_chain_except_on_step_consume(self):
    """§14.1: on_step records except_ in async chain (reraise=False)."""

    async def async_fail(x):
      raise ValueError('boom')

    def sync_handler(info):
      return 'recovered'

    result = await Chain(1).then(async_fail).except_(sync_handler).run()
    self.assertEqual(result, 'recovered')
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('except_', step_names)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)
    self.assertIsInstance(except_calls[0]['input_value'].exc, ValueError)

  async def test_async_chain_except_on_step_reraise(self):
    """§14.1: on_step records except_ in async chain (reraise=True)."""

    async def async_fail(x):
      raise ValueError('boom')

    def sync_handler(info):
      pass

    c = Chain(1).then(async_fail).except_(sync_handler, reraise=True)
    with self.assertRaises(ValueError):
      await c.run()
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('except_', step_names)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)

  async def test_async_chain_async_except_on_step_consume(self):
    """§14.1: on_step records except_ for async handler in async chain."""

    async def async_fail(x):
      raise ValueError('boom')

    async def async_handler(info):
      return 'async-recovered'

    result = await Chain(1).then(async_fail).except_(async_handler).run()
    self.assertEqual(result, 'async-recovered')
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('except_', step_names)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)
    self.assertIsInstance(except_calls[0]['input_value'].exc, ValueError)

  def test_sync_except_raise_true_on_step_recorded(self):
    """§14.1: on_step records except_ for sync except handler with reraise=True."""
    side_effects = []

    def handler(info):
      side_effects.append('handler-called')

    c = Chain(1).then(lambda x: 1 / 0).except_(handler, reraise=True)
    with self.assertRaises(ZeroDivisionError):
      c.run()
    self.assertEqual(side_effects, ['handler-called'])
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('except_', step_names)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)
    self.assertIsInstance(except_calls[0]['input_value'].exc, ZeroDivisionError)


class OnStepAsyncFinallyTest(IsolatedAsyncioTestCase):
  """§14.1: on_step callback for finally_ handlers in async chains."""

  def setUp(self):
    self.calls: list[dict] = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns):
    self.calls.append(
      {
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
      }
    )

  async def test_async_finally_on_step_timing_recorded(self):
    """§14.1: on_step records finally_ with timing for sync cleanup in async chain."""

    def sync_cleanup(rv):
      pass

    result = await Chain(5).then(async_fn).finally_(sync_cleanup).run()
    self.assertEqual(result, 6)
    finally_calls = [c for c in self.calls if c['step_name'] == 'finally_']
    self.assertEqual(len(finally_calls), 1)
    self.assertEqual(finally_calls[0]['input_value'], 5)
    self.assertIsInstance(finally_calls[0]['elapsed_ns'], int)
    self.assertGreaterEqual(finally_calls[0]['elapsed_ns'], 0)

  async def test_async_finally_async_handler_on_step_timing(self):
    """§14.1: on_step records finally_ for async cleanup in async chain."""

    async def async_cleanup(rv):
      pass

    result = await Chain(5).then(async_fn).finally_(async_cleanup).run()
    self.assertEqual(result, 6)
    finally_calls = [c for c in self.calls if c['step_name'] == 'finally_']
    self.assertEqual(len(finally_calls), 1)
    self.assertEqual(finally_calls[0]['input_value'], 5)


class OnStepInputValueTest(TestCase):
  """§14.1: Focused tests for input_value (3rd argument to on_step)."""

  def setUp(self):
    self.calls = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns):
    self.calls.append(
      {
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
      }
    )

  def test_root_input_value_is_run_value(self):
    """§14.1: Root step's input_value is the run value when provided."""
    Chain(lambda x: x * 2).run(7)
    root_call = self.calls[0]
    self.assertEqual(root_call['step_name'], 'root')
    self.assertEqual(root_call['input_value'], 7)

  def test_root_input_value_none_when_no_run_value(self):
    """§14.1: Root step's input_value is None (not Null) when no run value provided."""
    Chain(42).run()
    root_call = self.calls[0]
    self.assertEqual(root_call['step_name'], 'root')
    self.assertIsNone(root_call['input_value'])

  def test_then_input_value_is_previous_result(self):
    """§14.1: then step's input_value is the current pipeline value (previous result)."""
    Chain(5).then(lambda x: x + 10).run()
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 1)
    self.assertEqual(then_calls[0]['input_value'], 5)
    self.assertEqual(then_calls[0]['result'], 15)

  def test_do_input_value_is_current_value(self):
    """§14.1: do step's input_value is the current pipeline value."""
    Chain(5).do(lambda x: x * 100).run()
    do_calls = [c for c in self.calls if c['step_name'] == 'do']
    self.assertEqual(len(do_calls), 1)
    self.assertEqual(do_calls[0]['input_value'], 5)

  def test_except_input_value_is_chain_exc_info(self):
    """§14.1: except_ step's input_value is a ChainExcInfo instance."""
    Chain(1).then(lambda x: 1 / 0).except_(lambda info: 'handled').run()
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertEqual(len(except_calls), 1)
    self.assertIsInstance(except_calls[0]['input_value'], ChainExcInfo)
    self.assertIsInstance(except_calls[0]['input_value'].exc, ZeroDivisionError)
    self.assertEqual(except_calls[0]['input_value'].root_value, 1)

  def test_finally_input_value_is_root_value(self):
    """§14.1: finally_ step's input_value is the root value."""
    Chain(5).then(lambda x: x + 10).finally_(lambda rv: None).run()
    finally_calls = [c for c in self.calls if c['step_name'] == 'finally_']
    self.assertEqual(len(finally_calls), 1)
    self.assertEqual(finally_calls[0]['input_value'], 5)

  def test_finally_input_value_none_when_no_root(self):
    """§14.1: finally_ step's input_value is None when chain has no root value."""
    Chain().then(lambda: 99).finally_(lambda rv: None).run()
    finally_calls = [c for c in self.calls if c['step_name'] == 'finally_']
    self.assertEqual(len(finally_calls), 1)
    self.assertIsNone(finally_calls[0]['input_value'])


class OnStepExceptionParameterTest(TestCase):
  """§14.1: on_step 6-arg form receives exception parameter."""

  def setUp(self):
    self.calls = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns, exception):
    self.calls.append(
      {
        'chain': chain,
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
        'exception': exception,
      }
    )

  def test_success_exception_is_none(self):
    """§14.1: on_step fires with exception=None on success."""
    Chain(42).run()
    self.assertTrue(len(self.calls) >= 1)
    for call in self.calls:
      self.assertIsNone(call['exception'])

  def test_success_multi_step_exception_is_none(self):
    """§14.1: All successful steps have exception=None."""
    Chain(1).then(lambda x: x + 1).then(lambda x: x * 2).run()
    for call in self.calls:
      self.assertIsNone(call['exception'])

  def test_failure_exception_set(self):
    """§14.1: on_step fires with the exception when a step fails."""
    with self.assertRaises(ZeroDivisionError):
      Chain(1).then(lambda x: 1 / 0).run()
    # The failing 'then' step should have fired with the exception.
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 1)
    self.assertIsInstance(then_calls[0]['exception'], ZeroDivisionError)
    self.assertIsNone(then_calls[0]['result'])
    self.assertIsInstance(then_calls[0]['elapsed_ns'], int)
    self.assertGreaterEqual(then_calls[0]['elapsed_ns'], 0)

  def test_failure_at_root(self):
    """§14.1: on_step fires with exception when root step fails."""

    def bad_root():
      raise ValueError('root boom')

    with self.assertRaises(ValueError):
      Chain(bad_root).run()
    root_calls = [c for c in self.calls if c['step_name'] == 'root']
    self.assertEqual(len(root_calls), 1)
    self.assertIsInstance(root_calls[0]['exception'], ValueError)
    self.assertIsNone(root_calls[0]['result'])

  def test_failure_with_except_handler(self):
    """§14.1: on_step fires for failing step AND for except_ handler."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda _: 'handled').run()
    self.assertEqual(result, 'handled')
    # The failing 'then' step fires with exception set.
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 1)
    self.assertIsInstance(then_calls[0]['exception'], ZeroDivisionError)
    # The except_ handler fires as a success step (exception=None on the handler step).
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertEqual(len(except_calls), 1)
    self.assertIsNone(except_calls[0]['exception'])

  def test_success_steps_before_failure(self):
    """§14.1: Steps before the failing step have exception=None."""
    with self.assertRaises(ZeroDivisionError):
      Chain(1).then(lambda x: x + 1).then(lambda x: 1 / 0).run()
    # root and first then should succeed
    root_calls = [c for c in self.calls if c['step_name'] == 'root']
    self.assertEqual(len(root_calls), 1)
    self.assertIsNone(root_calls[0]['exception'])
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    # One success, one failure
    success_thens = [c for c in then_calls if c['exception'] is None]
    failure_thens = [c for c in then_calls if c['exception'] is not None]
    self.assertEqual(len(success_thens), 1)
    self.assertEqual(len(failure_thens), 1)
    self.assertIsInstance(failure_thens[0]['exception'], ZeroDivisionError)


class OnStepBackwardCompatTest(TestCase):
  """§14.1: 5-arg callbacks still work (backward compatibility)."""

  def setUp(self):
    self.calls = []
    Chain.on_step = self._recorder_5arg

  def tearDown(self):
    Chain.on_step = None

  def _recorder_5arg(self, chain, step_name, input_value, result, elapsed_ns):
    """Classic 5-arg callback — no exception parameter."""
    self.calls.append(
      {
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
      }
    )

  def test_5arg_callback_success(self):
    """§14.1: 5-arg on_step callback works for successful steps."""
    Chain(42).then(lambda x: x + 1).run()
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)
    self.assertIn('then', step_names)
    root_call = self.calls[0]
    self.assertEqual(root_call['result'], 42)

  def test_5arg_callback_not_called_on_failure(self):
    """§14.1: 5-arg on_step callback is NOT called for failing steps."""
    with self.assertRaises(ZeroDivisionError):
      Chain(1).then(lambda x: 1 / 0).run()
    # Root should be recorded, but failing then step should not
    # (5-arg callbacks don't receive the exception parameter).
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)
    # The failing then step should NOT be in calls for a 5-arg callback
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 0)

  def test_5arg_callback_with_except_handler(self):
    """§14.1: 5-arg on_step works with except_ handler present."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda _: 'handled').run()
    self.assertEqual(result, 'handled')
    step_names = [c['step_name'] for c in self.calls]
    self.assertIn('root', step_names)
    # except_ handler itself fires on_step (it's a success step)
    self.assertIn('except_', step_names)

  def test_5arg_callback_with_varargs(self):
    """§14.1: Callback with *args is treated as 6-arg compatible."""
    vararg_calls = []

    def vararg_recorder(chain, step_name, *args):
      vararg_calls.append({'step_name': step_name, 'args': args})

    Chain.on_step = vararg_recorder
    Chain(42).run()
    self.assertTrue(len(vararg_calls) >= 1)
    # *args should capture (input_value, result, elapsed_ns, exception)
    root_call = vararg_calls[0]
    self.assertEqual(root_call['step_name'], 'root')
    self.assertEqual(len(root_call['args']), 4)  # input_value, result, elapsed_ns, exception=None
    self.assertIsNone(root_call['args'][3])  # exception is None on success


class OnStepExceptionAsyncTest(IsolatedAsyncioTestCase):
  """§14.1: on_step exception parameter in async chains."""

  def setUp(self):
    self.calls = []
    Chain.on_step = self._recorder

  def tearDown(self):
    Chain.on_step = None

  def _recorder(self, chain, step_name, input_value, result, elapsed_ns, exception):
    self.calls.append(
      {
        'step_name': step_name,
        'input_value': input_value,
        'result': result,
        'elapsed_ns': elapsed_ns,
        'exception': exception,
      }
    )

  async def test_async_success_exception_none(self):
    """§14.1: Async steps have exception=None on success."""

    async def async_add(x):
      return x + 1

    result = await Chain(10).then(async_add).run()
    self.assertEqual(result, 11)
    for call in self.calls:
      self.assertIsNone(call['exception'])

  async def test_async_failure_exception_set(self):
    """§14.1: Async failing step fires on_step with exception set."""

    async def async_fail(x):
      raise ValueError('async boom')

    with self.assertRaises(ValueError):
      await Chain(1).then(async_fail).run()
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 1)
    self.assertIsInstance(then_calls[0]['exception'], ValueError)
    self.assertIsNone(then_calls[0]['result'])

  async def test_async_failure_with_except_handler(self):
    """§14.1: Async chain with except_ handler — on_step fires for both failing step and handler."""

    async def async_fail(x):
      raise ValueError('boom')

    result = await Chain(1).then(async_fail).except_(lambda _: 'recovered').run()
    self.assertEqual(result, 'recovered')
    then_calls = [c for c in self.calls if c['step_name'] == 'then']
    self.assertEqual(len(then_calls), 1)
    self.assertIsInstance(then_calls[0]['exception'], ValueError)
    except_calls = [c for c in self.calls if c['step_name'] == 'except_']
    self.assertEqual(len(except_calls), 1)
    self.assertIsNone(except_calls[0]['exception'])
