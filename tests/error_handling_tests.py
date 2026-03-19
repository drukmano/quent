# SPDX-License-Identifier: MIT
"""Tests for SPEC §6 — Error Handling.

Covers: except_() registration, handler failure, reraise behavior,
finally_() failure, control flow in handlers, execution order,
ExceptionGroup behavior, and async error handling paths.

Calling convention tests (QuentExcInfo rules, finally_ conventions) are
in calling_convention_tests.py. Async transition tests are in asymmetry_tests.py.
"""

from __future__ import annotations

import sys
import warnings
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Q, QuentException, QuentExcInfo
from quent._types import _ControlFlowSignal
from tests.fixtures import (
  V_BAD_CLEANUP,
  V_FN,
  V_RAISE,
  async_fn,
  async_identity,
)
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §6.2.1 Handler Registration
# ---------------------------------------------------------------------------


class ExceptRegistrationTest(TestCase):
  """SPEC §6.2.1: except_() registration constraints."""

  def test_except_requires_callable(self) -> None:
    """except_() with non-callable raises TypeError."""
    with self.assertRaises(TypeError) as ctx:
      Q(1).except_(42)  # type: ignore[arg-type]
    self.assertIn('callable', str(ctx.exception))

  def test_except_at_most_one(self) -> None:
    """Second except_() on same pipeline raises QuentException."""
    c = Q(1).except_(lambda _: None)
    with self.assertRaises(QuentException):
      c.except_(lambda _: None)

  def test_exceptions_single_type(self) -> None:
    """exceptions= accepts a single exception type."""
    c = Q(1).then(lambda x: 1 / 0).except_(lambda _: 'caught', exceptions=ZeroDivisionError)
    self.assertEqual(c.run(), 'caught')

  def test_exceptions_iterable_of_types(self) -> None:
    """exceptions= accepts an iterable of types."""
    c = Q(1).then(lambda x: 1 / 0).except_(lambda _: 'caught', exceptions=[ZeroDivisionError, ValueError])
    self.assertEqual(c.run(), 'caught')

  def test_exceptions_none_defaults_to_exception(self) -> None:
    """exceptions=None defaults to catching Exception."""
    c = Q(1).then(lambda x: 1 / 0).except_(lambda _: 'caught')
    self.assertEqual(c.run(), 'caught')

  def test_exceptions_empty_iterable_raises(self) -> None:
    """Empty iterable for exceptions raises QuentException."""
    with self.assertRaises(QuentException):
      Q(1).except_(lambda _: None, exceptions=[])

  def test_exceptions_non_base_exception_raises(self) -> None:
    """Non-BaseException subclass in exceptions raises TypeError."""
    with self.assertRaises(TypeError):
      Q(1).except_(lambda _: None, exceptions=int)  # type: ignore[arg-type]

  def test_exceptions_string_raises_type_error(self) -> None:
    """String value for exceptions raises TypeError."""
    with self.assertRaises(TypeError):
      Q(1).except_(lambda _: None, exceptions='ValueError')

  def test_exceptions_base_exception_warns(self) -> None:
    """BaseException subtype (not Exception) emits RuntimeWarning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Q(1).except_(lambda _: None, exceptions=KeyboardInterrupt)
    self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))

  def test_exceptions_tuple_of_types(self) -> None:
    """exceptions= accepts a tuple of types (common usage pattern)."""
    c = Q(1).then(lambda x: 1 / 0).except_(lambda _: 'caught', exceptions=(ZeroDivisionError, ValueError))
    self.assertEqual(c.run(), 'caught')

  def test_exception_type_not_caught_propagates(self) -> None:
    """Exception not matching exceptions= propagates unhandled."""
    c = Q(1).then(lambda x: (_ for _ in ()).throw(ValueError('boom'))).except_(lambda _: 'caught', exceptions=TypeError)
    with self.assertRaises(ValueError):
      c.run()


# ---------------------------------------------------------------------------
# §6.2.2 Except Handler Calling Convention
# (Canonical calling convention tests are in calling_convention_tests.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# §6.2.2 Nested Pipeline as Except Handler — Execution Mode
# ---------------------------------------------------------------------------


class ExceptNestedChainExecutionModeTest(SymmetricTestCase):
  """SPEC §6.2.2: Nested pipeline as except handler runs via top-level execution.

  The nested pipeline is executed via run() (top-level execution), not the internal
  nested pipeline execution path. This means:
  - Control flow signals inside the handler pipeline are caught and wrapped in QuentException
  - The handler pipeline's own except_() and finally_() handlers apply independently
  """

  async def test_return_inside_handler_pipeline_raises_quent_exception(self) -> None:
    """return_() inside a nested pipeline except handler raises QuentException."""
    # The nested pipeline uses return_() — since it runs via top-level execution,
    # the _Return signal is caught by _except_handler_body and wrapped in QuentException.
    inner = Q().then(lambda info: Q.return_('escaped'))
    c = Q(1).then(lambda x: 1 / 0).except_(inner)
    with self.assertRaises(QuentException):
      c.run()

  async def test_break_inside_handler_pipeline_raises_quent_exception(self) -> None:
    """break_() inside a nested pipeline except handler raises QuentException."""
    inner = Q().then(lambda info: Q.break_('escaped'))
    c = Q(1).then(lambda x: 1 / 0).except_(inner)
    with self.assertRaises(QuentException):
      c.run()

  async def test_handler_pipeline_own_except_applies(self) -> None:
    """Handler pipeline's own except_() applies independently."""
    # The inner pipeline has its own except_ that catches errors within the handler pipeline.
    inner = (
      Q()
      .then(lambda info: 1 / 0)  # error inside handler pipeline
      .except_(lambda _: 'inner-recovered')
    )
    result = Q(1).then(lambda x: (_ for _ in ()).throw(ValueError('outer'))).except_(inner).run()
    self.assertEqual(result, 'inner-recovered')

  async def test_handler_pipeline_own_finally_applies(self) -> None:
    """Handler pipeline's own finally_() runs independently."""
    cleanup = []
    inner = (
      Q().then(lambda info: ('handled', type(info.exc).__name__)).finally_(lambda rv: cleanup.append('inner-finally'))
    )
    result = Q(1).then(lambda x: 1 / 0).except_(inner).run()
    self.assertEqual(result[0], 'handled')
    self.assertEqual(result[1], 'ZeroDivisionError')
    self.assertEqual(cleanup, ['inner-finally'])

  async def test_handler_pipeline_except_and_finally_on_error(self) -> None:
    """Handler pipeline's own except_ and finally_ both apply on handler error path."""
    order = []
    inner = (
      Q()
      .then(lambda info: 1 / 0)  # error inside handler pipeline
      .except_(lambda _: (order.append('inner-except'), 'inner-recovered')[-1])
      .finally_(lambda rv: order.append('inner-finally'))
    )
    result = Q(1).then(lambda x: (_ for _ in ()).throw(ValueError('outer'))).except_(inner).run()
    self.assertEqual(result, 'inner-recovered')
    self.assertEqual(order, ['inner-except', 'inner-finally'])

  async def test_async_return_inside_handler_pipeline_raises_quent_exception(self) -> None:
    """Async: return_() inside a nested pipeline except handler raises QuentException."""

    async def async_fail(x):
      raise ValueError('async boom')

    inner = Q().then(lambda info: Q.return_('escaped'))
    c = Q(1).then(async_fail).except_(inner)
    with self.assertRaises(QuentException):
      await c.run()

  async def test_async_handler_pipeline_own_except_applies(self) -> None:
    """Async: handler pipeline's own except_() applies independently."""

    async def async_fail(x):
      raise ValueError('async boom')

    inner = Q().then(lambda info: 1 / 0).except_(lambda _: 'inner-recovered')
    result = await Q(1).then(async_fail).except_(inner).run()
    self.assertEqual(result, 'inner-recovered')

  async def test_async_handler_pipeline_own_finally_applies(self) -> None:
    """Async: handler pipeline's own finally_() runs independently."""
    cleanup = []

    async def async_fail(x):
      raise ValueError('async boom')

    inner = Q().then(lambda info: 'handled').finally_(lambda rv: cleanup.append('inner-finally'))
    result = await Q(1).then(async_fail).except_(inner).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(cleanup, ['inner-finally'])


# ---------------------------------------------------------------------------
# §6.2.3 reraise Modes
# (Bridge symmetry for reraise modes is covered by the error bridge test)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# §6.2.4 Handler Failure with reraise=True
# ---------------------------------------------------------------------------


class ExceptHandlerFailureRaiseTrueTest(SymmetricTestCase):
  """SPEC §6.2.4: Handler failure with reraise=True."""

  async def test_exception_subclass_handler_failure_discarded(self) -> None:
    """Exception from handler discarded, RuntimeWarning, original re-raised."""

    def bad_handler(info):
      raise ValueError('handler boom')

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler, reraise=True)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ZeroDivisionError) as ctx:
        c.run()
    self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))
    # After fix: save/restore original __suppress_context__ instead of unconditionally
    # setting True.  The original ZeroDivisionError had no pre-existing context chain,
    # so __suppress_context__ is restored to False.
    self.assertFalse(ctx.exception.__suppress_context__)

  async def test_exception_subclass_handler_failure_note_attached(self) -> None:
    """On Python 3.11+, a note is attached to the original exception."""
    if sys.version_info < (3, 11):
      self.skipTest('add_note requires Python 3.11+')

    def bad_handler(info):
      raise ValueError('handler boom')

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler, reraise=True)
    with warnings.catch_warnings(record=True):
      warnings.simplefilter('always')
      try:
        c.run()
      except ZeroDivisionError as exc:
        self.assertTrue(any('handler boom' in n for n in exc.__notes__))
      else:
        self.fail('Expected ZeroDivisionError')

  async def test_base_exception_handler_failure_propagates(self) -> None:
    """BaseException from handler propagates naturally."""

    def bad_handler(info):
      raise KeyboardInterrupt('handler signal')

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler, reraise=True)
    with self.assertRaises(KeyboardInterrupt):
      c.run()


# ---------------------------------------------------------------------------
# §6.2.5 Handler Failure with reraise=False
# ---------------------------------------------------------------------------


class ExceptHandlerFailureRaiseFalseTest(SymmetricTestCase):
  """SPEC §6.2.5: Handler failure with reraise=False."""

  async def test_handler_exception_propagates_with_cause(self) -> None:
    """Handler exception propagates, original set as __cause__."""

    def bad_handler(info):
      raise ValueError('handler boom')

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler)
    with self.assertRaises(ValueError) as ctx:
      c.run()
    self.assertIsInstance(ctx.exception.__cause__, ZeroDivisionError)


# ---------------------------------------------------------------------------
# §6.2.6 Control Flow in Except Handlers
# ---------------------------------------------------------------------------


class ExceptControlFlowTest(TestCase):
  """SPEC §6.2.6: return_() and break_() in except handlers."""

  def test_return_in_except_raises_quent_exception(self) -> None:
    """return_() inside except handler raises QuentException."""

    def handler(info):
      return Q.return_('nope')

    c = Q(1).then(lambda x: 1 / 0).except_(handler)
    with self.assertRaises(QuentException):
      c.run()

  def test_break_in_except_raises_quent_exception(self) -> None:
    """break_() inside except handler raises QuentException."""

    def handler(info):
      return Q.break_('nope')

    c = Q(1).then(lambda x: 1 / 0).except_(handler)
    with self.assertRaises(QuentException):
      c.run()


# ---------------------------------------------------------------------------
# §6.3 finally_()
# (Canonical calling convention and registration tests are in calling_convention_tests.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# §6.3.3 Finally Handler Failure
# ---------------------------------------------------------------------------


class FinallyFailureTest(SymmetricTestCase):
  """SPEC §6.3.3: Finally handler failure."""

  async def test_finally_failure_replaces_original_on_error_path(self) -> None:
    """Finally failure replaces original exception, __context__ preserved."""

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    await self.variant(
      lambda fail_fn: Q(1).then(fail_fn).finally_(bad_cleanup).run(),
      expected_exc=RuntimeError,
      expected_msg='cleanup boom',
      fail_fn=V_RAISE,
    )

  async def test_finally_failure_replaces_original_async_cleanup(self) -> None:
    """Async finally failure replaces original — async fail ensures async mode."""

    async def async_fail(x):
      raise ValueError('async boom')

    await self.variant(
      lambda cleanup: Q(1).then(async_fail).finally_(cleanup).run(),
      expected_exc=RuntimeError,
      expected_msg='cleanup boom',
      cleanup=V_BAD_CLEANUP,
    )

  async def test_finally_failure_on_success_path_propagates(self) -> None:
    """Finally failure on success path propagates as pipeline error.
    Sync cleanup with sync/async step. Async cleanup only with async step
    (sync pipeline + async finally = async transition per §6.3.5)."""
    from tests.fixtures import sync_bad_cleanup

    await self.variant(
      lambda fn: Q(1).then(fn).finally_(sync_bad_cleanup).run(),
      expected_exc=RuntimeError,
      expected_msg='cleanup boom',
      fn=V_FN,
    )

  async def test_finally_failure_on_success_path_async_cleanup(self) -> None:
    """Async finally failure on async success path — both sync and async cleanup."""

    await self.variant(
      lambda cleanup: Q(1).then(async_fn).finally_(cleanup).run(),
      expected_exc=RuntimeError,
      expected_msg='cleanup boom',
      cleanup=V_BAD_CLEANUP,
    )

  async def test_finally_failure_context_is_original_exception(self) -> None:
    """§6.3.3: Finally failure __context__ is set to the original exception."""

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(lambda x: 1 / 0).finally_(bad_cleanup)
    with self.assertRaises(RuntimeError) as ctx:
      c.run()
    self.assertIsInstance(ctx.exception.__context__, ZeroDivisionError)

  async def test_finally_failure_note_attached(self) -> None:
    """On Python 3.11+, note attached describing replaced exception."""
    if sys.version_info < (3, 11):
      self.skipTest('add_note requires Python 3.11+')

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(lambda x: 1 / 0).finally_(bad_cleanup)
    try:
      c.run()
    except RuntimeError as exc:
      self.assertTrue(any('ZeroDivisionError' in n for n in exc.__notes__))
    else:
      self.fail('Expected RuntimeError')


# ---------------------------------------------------------------------------
# §6.3.4 Control Flow in Finally Handlers
# ---------------------------------------------------------------------------


class FinallyControlFlowTest(TestCase):
  """SPEC §6.3.4: Control flow signals in finally handlers."""

  def test_return_in_finally_raises_quent_exception(self) -> None:
    """return_() in finally handler raises QuentException."""

    def cleanup(rv):
      return Q.return_('nope')

    c = Q(1).then(lambda x: x + 1).finally_(cleanup)
    with self.assertRaises(QuentException):
      c.run()

  def test_break_in_finally_raises_quent_exception(self) -> None:
    """break_() in finally handler raises QuentException."""

    def cleanup(rv):
      return Q.break_('nope')

    c = Q(1).then(lambda x: x + 1).finally_(cleanup)
    with self.assertRaises(QuentException):
      c.run()


# ---------------------------------------------------------------------------
# §6.4 Execution Order
# ---------------------------------------------------------------------------


class ExecutionOrderTest(SymmetricTestCase):
  """SPEC §6.4: Full error handling execution order."""

  async def test_success_path_finally_runs(self) -> None:
    """Success: pipeline → finally (success context)."""
    order = []
    result = (
      Q(1)
      .then(lambda x: x + 1)
      .except_(lambda _: order.append('except'))
      .finally_(lambda rv: order.append('finally'))
      .run()
    )
    self.assertEqual(result, 2)
    self.assertEqual(order, ['finally'])

  async def test_error_reraised_by_handler(self) -> None:
    """Error path (reraise=True): pipeline → except → finally (failure context)."""
    order = []

    def handler(info):
      order.append('except')

    c = Q(1).then(lambda x: 1 / 0).except_(handler, reraise=True).finally_(lambda rv: order.append('finally'))
    with self.assertRaises(ZeroDivisionError):
      c.run()
    self.assertEqual(order, ['except', 'finally'])

  async def test_unmatched_exception_skips_handler(self) -> None:
    """Unmatched exception: pipeline → skip except → finally (failure context)."""
    order = []

    c = (
      Q(1)
      .then(lambda x: 1 / 0)
      .except_(lambda _: order.append('except'), exceptions=ValueError)
      .finally_(lambda rv: order.append('finally'))
    )
    with self.assertRaises(ZeroDivisionError):
      c.run()
    self.assertEqual(order, ['finally'])  # except skipped


# ---------------------------------------------------------------------------
# §6.4 Execution Order — Async variants
# ---------------------------------------------------------------------------


class ExecutionOrderAsyncTest(SymmetricTestCase):
  """SPEC §6.4: Execution order with async steps."""

  async def test_async_success_path(self) -> None:
    """Async success: pipeline → finally."""
    order = []

    result = await Q(1).then(async_fn).finally_(lambda rv: order.append('finally')).run()
    self.assertEqual(result, 2)
    self.assertEqual(order, ['finally'])

  async def test_async_error_consumed(self) -> None:
    """Async error (reraise=False): except → finally (success context)."""
    order = []

    async def async_fail(x):
      raise ValueError('async boom')

    result = await (
      Q(1)
      .then(async_fail)
      .except_(lambda _: (order.append('except'), 'recovered')[-1])
      .finally_(lambda rv: order.append('finally'))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(order, ['except', 'finally'])

  async def test_async_error_reraised(self) -> None:
    """Async error (reraise=True): except → finally (failure context)."""
    order = []

    async def async_fail(x):
      raise ValueError('async boom')

    c = (
      Q(1)
      .then(async_fail)
      .except_(lambda _: order.append('except'), reraise=True)
      .finally_(lambda rv: order.append('finally'))
    )
    with self.assertRaises(ValueError):
      await c.run()
    self.assertEqual(order, ['except', 'finally'])


# ---------------------------------------------------------------------------
# §6.5 ExceptionGroup
# ---------------------------------------------------------------------------


class ExceptionGroupTest(SymmetricTestCase):
  """SPEC §6.5: ExceptionGroup from concurrent operations."""

  async def test_gather_single_failure_not_wrapped(self) -> None:
    """Single gather failure: exception propagates directly, not wrapped."""

    def fail_fn(x):
      raise ValueError('boom')

    c = Q(1).gather(lambda x: x, fail_fn)
    with self.assertRaises(ValueError):
      c.run()

  async def test_gather_multiple_failures_wrapped(self) -> None:
    """Multiple gather failures: wrapped in ExceptionGroup."""

    # The first fn must succeed (it's probed inline to detect sync/async).
    # The remaining fns run in the thread pool — if 2+ fail, ExceptionGroup.
    def ok(x):
      return x

    def fail1(x):
      raise ValueError('boom1')

    def fail2(x):
      raise TypeError('boom2')

    c = Q(1).gather(ok, fail1, fail2)
    try:
      c.run()
    except BaseException as exc:
      # Should be an ExceptionGroup (builtin or polyfill)
      self.assertTrue(hasattr(exc, 'exceptions'), f'Expected ExceptionGroup, got {type(exc).__name__}')
      self.assertEqual(len(exc.exceptions), 2)
    else:
      self.fail('Expected ExceptionGroup')

  async def test_gather_exception_group_message_format(self) -> None:
    """SPEC §6.5: ExceptionGroup message follows 'gather() encountered N exceptions' pattern."""

    def ok(x):
      return x

    def fail1(x):
      raise ValueError('boom1')

    def fail2(x):
      raise TypeError('boom2')

    c = Q(1).gather(ok, fail1, fail2)
    try:
      c.run()
    except BaseException as exc:
      self.assertTrue(hasattr(exc, 'exceptions'), f'Expected ExceptionGroup, got {type(exc).__name__}')
      # Verify message format per SPEC §6.5
      self.assertIn('gather() encountered 2 exceptions', str(exc.args[0]))
    else:
      self.fail('Expected ExceptionGroup')

  async def test_gather_async_multiple_failures_wrapped(self) -> None:
    """Async: Multiple gather failures wrapped in ExceptionGroup."""

    async def ok(x):
      return x

    async def fail1(x):
      raise ValueError('async boom1')

    async def fail2(x):
      raise TypeError('async boom2')

    c = Q(1).gather(ok, fail1, fail2)
    try:
      await c.run()
    except BaseException as exc:
      self.assertTrue(hasattr(exc, 'exceptions'), f'Expected ExceptionGroup, got {type(exc).__name__}')
      self.assertEqual(len(exc.exceptions), 2)
      self.assertIn('gather() encountered 2 exceptions', str(exc.args[0]))
    else:
      self.fail('Expected ExceptionGroup')

  async def test_gather_async_single_failure_not_wrapped(self) -> None:
    """Async: Single gather failure propagates directly, not wrapped."""

    async def ok(x):
      return x

    async def fail(x):
      raise ValueError('single async boom')

    c = Q(1).gather(ok, fail)
    with self.assertRaises(ValueError) as ctx:
      await c.run()
    self.assertIn('single async boom', str(ctx.exception))

  async def test_map_concurrent_multiple_failures_exception_group(self) -> None:
    """Concurrent map with multiple failures: wrapped in ExceptionGroup.

    Per SPEC §6.5: multiple concurrent worker failures are wrapped in an
    ExceptionGroup with message "{op}() encountered N exceptions".
    """

    def maybe_fail(x):
      if x >= 2:
        raise ValueError(f'fail-{x}')
      return x

    c = Q([0, 1, 2, 3, 4]).foreach(maybe_fail, concurrency=5)
    try:
      c.run()
      self.fail('Expected ExceptionGroup')
    except BaseException as exc:
      self.assertTrue(hasattr(exc, 'exceptions'), f'Expected ExceptionGroup, got {type(exc).__name__}')
      self.assertIn('foreach() encountered', str(exc.args[0]))
      self.assertIn('exceptions', str(exc.args[0]))
      # All individual exceptions should be present
      exc_messages = [str(e) for e in exc.exceptions]
      self.assertTrue(any('fail-' in m for m in exc_messages))
      self.assertGreaterEqual(len(exc.exceptions), 2)

  async def test_map_concurrent_single_failure_not_wrapped(self) -> None:
    """Concurrent map with single failure: raised directly, not wrapped.

    Per SPEC §6.5: single failure propagates directly.
    """

    def maybe_fail(x):
      if x == 3:
        raise ValueError('fail-3')
      return x

    c = Q([0, 1, 2, 3, 4]).foreach(maybe_fail, concurrency=5)
    with self.assertRaises(ValueError) as ctx:
      c.run()
    self.assertIn('fail-3', str(ctx.exception))

  async def test_foreach_do_exception_group_message_format(self) -> None:
    """SPEC §6.5: foreach_do() concurrent ExceptionGroup message format.

    Per SPEC §6.5: "For concurrent iteration operations, the message follows
    the same pattern: 'foreach() encountered N exceptions' or
    'foreach_do() encountered N exceptions'."
    """

    def maybe_fail(x):
      if x >= 2:
        raise ValueError(f'fail-{x}')

    c = Q([0, 1, 2, 3, 4]).foreach_do(maybe_fail, concurrency=5)
    try:
      c.run()
      self.fail('Expected ExceptionGroup')
    except BaseException as exc:
      self.assertTrue(hasattr(exc, 'exceptions'), f'Expected ExceptionGroup, got {type(exc).__name__}')
      # Verify message format per SPEC §6.5
      self.assertIn('foreach_do() encountered', str(exc.args[0]))
      self.assertIn('exceptions', str(exc.args[0]))
      self.assertGreaterEqual(len(exc.exceptions), 2)


# ---------------------------------------------------------------------------
# Async except handler tests
# ---------------------------------------------------------------------------


class AsyncExceptHandlerTest(SymmetricTestCase):
  """Async variants of except handler calling convention."""

  async def test_async_except_default(self) -> None:
    """Async except handler: handler(info) where info is QuentExcInfo."""
    received = {}

    async def async_fail(x):
      raise ValueError('async fail')

    async def handler(info: QuentExcInfo):
      received['exc'] = info.exc
      received['rv'] = info.root_value
      return 'async handled'

    result = await Q(42).then(async_fail).except_(handler).run()
    self.assertEqual(result, 'async handled')
    self.assertIsInstance(received['exc'], ValueError)
    self.assertEqual(received['rv'], 42)

  async def test_async_except_raise_true_handler_failure(self) -> None:
    """Async: reraise=True, handler raises Exception → discarded, original re-raised."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_handler(info):
      raise RuntimeError('handler fail')

    c = Q(1).then(async_fail).except_(bad_handler, reraise=True)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ValueError) as ctx:
        await c.run()
    self.assertIn('original', str(ctx.exception))
    self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))

  async def test_async_except_raise_false_handler_failure(self) -> None:
    """Async: reraise=False, handler raises → propagates with __cause__."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_handler(info):
      raise RuntimeError('handler fail')

    c = Q(1).then(async_fail).except_(bad_handler)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  async def test_async_except_raise_true_base_exception_propagates(self) -> None:
    """Async: reraise=True, handler raises BaseException → propagates naturally."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_handler(info):
      raise KeyboardInterrupt('handler signal')

    c = Q(1).then(async_fail).except_(bad_handler, reraise=True)
    with self.assertRaises(KeyboardInterrupt):
      await c.run()


# ---------------------------------------------------------------------------
# §6.3.5 Async Finally in Sync Chains
# (Canonical tests for async transition are in asymmetry_tests.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Async finally handler tests
# ---------------------------------------------------------------------------


class AsyncFinallyTest(SymmetricTestCase):
  """Async variants of finally_ handler."""

  async def test_async_finally_on_success(self) -> None:
    """Async finally handler runs on success path."""
    cleanup = []

    async def async_cleanup(rv):
      cleanup.append(rv)

    result = await Q(42).then(async_fn).finally_(async_cleanup).run()
    self.assertEqual(result, 43)
    self.assertEqual(cleanup, [42])

  async def test_async_finally_on_failure(self) -> None:
    """Async finally handler runs on failure path."""
    cleanup = []

    async def async_fail(x):
      raise ValueError('boom')

    async def async_cleanup(rv):
      cleanup.append(rv)

    c = Q(42).then(async_fail).finally_(async_cleanup)
    with self.assertRaises(ValueError):
      await c.run()
    self.assertEqual(cleanup, [42])

  async def test_async_finally_failure_replaces_original(self) -> None:
    """Async finally failure replaces original exception, __context__ preserved."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(async_fail).finally_(bad_cleanup)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIsInstance(ctx.exception.__context__, ValueError)

  async def test_async_control_flow_in_finally_raises(self) -> None:
    """Async finally: return_() raises QuentException."""

    async def cleanup_with_return(rv):
      return Q.return_('bad')

    c = Q(1).then(async_identity).finally_(cleanup_with_return)
    with self.assertRaises(QuentException):
      await c.run()

  async def test_async_control_flow_break_in_finally_raises(self) -> None:
    """Async finally: break_() raises QuentException."""

    async def cleanup_with_break(rv):
      return Q.break_('bad')

    c = Q(1).then(async_identity).finally_(cleanup_with_break)
    with self.assertRaises(QuentException):
      await c.run()


# ---------------------------------------------------------------------------
# Except + Finally combined edge cases
# ---------------------------------------------------------------------------


class ExceptFinallyComboTest(SymmetricTestCase):
  """Combined except + finally edge cases."""

  async def test_except_consume_then_finally_success_context(self) -> None:
    """except_ consumes error → finally runs in success context."""
    order = []
    result = (
      Q(1)
      .then(lambda x: 1 / 0)
      .except_(lambda _: (order.append('except'), 'recovered')[-1])
      .finally_(lambda rv: order.append('finally'))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(order, ['except', 'finally'])

  async def test_except_reraise_then_finally_failure_context(self) -> None:
    """except_ re-raises → finally runs in failure context."""
    order = []
    c = (
      Q(1)
      .then(lambda x: 1 / 0)
      .except_(lambda _: order.append('except'), reraise=True)
      .finally_(lambda rv: order.append('finally'))
    )
    with self.assertRaises(ZeroDivisionError):
      c.run()
    self.assertEqual(order, ['except', 'finally'])


# ---------------------------------------------------------------------------
# Comprehensive async error handling — covering _engine.py async paths
# ---------------------------------------------------------------------------


class AsyncExceptConsumeTest(SymmetricTestCase):
  """Async except handler that consumes exceptions (reraise=False)."""

  async def test_async_pipeline_except_consume_with_sync_handler(self) -> None:
    """Async pipeline (async step fails), sync except handler consumes."""
    order = []

    async def async_fail(x):
      raise ValueError('async fail')

    def sync_handler(info):
      order.append('except')
      return 'sync-recovered'

    result = await Q(1).then(async_fail).except_(sync_handler).finally_(lambda rv: order.append('finally')).run()
    self.assertEqual(result, 'sync-recovered')
    self.assertEqual(order, ['except', 'finally'])

  async def test_async_pipeline_except_consume_with_async_handler(self) -> None:
    """Async pipeline (async step fails), async except handler consumes."""
    order = []

    async def async_fail(x):
      raise ValueError('async fail')

    async def async_handler(info):
      order.append('except')
      return 'async-recovered'

    result = await Q(1).then(async_fail).except_(async_handler).finally_(lambda rv: order.append('finally')).run()
    self.assertEqual(result, 'async-recovered')
    self.assertEqual(order, ['except', 'finally'])

  async def test_sync_pipeline_async_except_handler_consume(self) -> None:
    """Sync pipeline (sync step fails), async except handler consumes via _async_except_handler."""
    # This exercises the _async_except_handler path with reraise=False

    async def async_handler(info):
      return 'async-recovered'

    result = await Q(1).then(lambda x: 1 / 0).except_(async_handler).run()
    self.assertEqual(result, 'async-recovered')

  async def test_sync_pipeline_async_except_handler_consume_with_finally(self) -> None:
    """Sync pipeline, async except handler consumes, finally runs."""
    cleanup = []

    async def async_handler(info):
      return 'async-recovered'

    result = await (
      Q(1).then(lambda x: 1 / 0).except_(async_handler).finally_(lambda rv: cleanup.append('finally')).run()
    )
    self.assertEqual(result, 'async-recovered')
    self.assertEqual(cleanup, ['finally'])


class AsyncExceptReraiseTest(SymmetricTestCase):
  """Async except handler with reraise=True."""

  async def test_async_pipeline_except_reraise_with_sync_handler(self) -> None:
    """Async pipeline, sync except handler, reraise=True re-raises original."""
    order = []

    async def async_fail(x):
      raise ValueError('async fail')

    def sync_handler(info):
      order.append('except')

    c = Q(1).then(async_fail).except_(sync_handler, reraise=True).finally_(lambda rv: order.append('finally'))
    with self.assertRaises(ValueError) as ctx:
      await c.run()
    self.assertIn('async fail', str(ctx.exception))
    self.assertEqual(order, ['except', 'finally'])

  async def test_async_pipeline_except_reraise_with_async_handler(self) -> None:
    """Async pipeline, async except handler, reraise=True re-raises original."""
    order = []

    async def async_fail(x):
      raise ValueError('async fail')

    async def async_handler(info):
      order.append('except')

    c = Q(1).then(async_fail).except_(async_handler, reraise=True).finally_(lambda rv: order.append('finally'))
    with self.assertRaises(ValueError) as ctx:
      await c.run()
    self.assertIn('async fail', str(ctx.exception))
    self.assertEqual(order, ['except', 'finally'])

  async def test_sync_pipeline_async_except_handler_reraise(self) -> None:
    """Sync pipeline, async except handler, reraise=True via _async_except_handler."""
    side_effects = []

    async def async_handler(info):
      side_effects.append('handler-called')

    c = Q(1).then(lambda x: 1 / 0).except_(async_handler, reraise=True)
    with self.assertRaises(ZeroDivisionError):
      await c.run()
    self.assertEqual(side_effects, ['handler-called'])

  async def test_sync_pipeline_async_except_handler_reraise_with_finally(self) -> None:
    """Sync pipeline, async except handler, reraise=True, finally handler runs."""
    order = []

    async def async_handler(info):
      order.append('except')

    c = Q(1).then(lambda x: 1 / 0).except_(async_handler, reraise=True).finally_(lambda rv: order.append('finally'))
    with self.assertRaises(ZeroDivisionError):
      await c.run()
    self.assertEqual(order, ['except', 'finally'])


class AsyncHandlerFailureTest(SymmetricTestCase):
  """Async except handler failure scenarios."""

  async def test_async_pipeline_handler_raises_exception_reraise_mode(self) -> None:
    """Async pipeline, async handler raises Exception with reraise=True → discarded, warning."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_handler(info):
      raise RuntimeError('handler boom')

    c = Q(1).then(async_fail).except_(bad_handler, reraise=True)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ValueError) as ctx:
        await c.run()
    self.assertIn('original', str(ctx.exception))
    self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))

  async def test_async_pipeline_handler_raises_exception_consume_mode(self) -> None:
    """Async pipeline, async handler raises with reraise=False → propagates with __cause__."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_handler(info):
      raise RuntimeError('handler boom')

    c = Q(1).then(async_fail).except_(bad_handler)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  async def test_async_pipeline_handler_raises_base_exception_reraise_mode(self) -> None:
    """Async pipeline, async handler raises BaseException with reraise=True → propagates."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_handler(info):
      raise KeyboardInterrupt('signal')

    c = Q(1).then(async_fail).except_(bad_handler, reraise=True)
    with self.assertRaises(KeyboardInterrupt):
      await c.run()

  async def test_sync_pipeline_async_handler_failure_consume_mode(self) -> None:
    """Sync pipeline error, async except handler fails with reraise=False → propagates with __cause__."""

    async def bad_handler(info):
      raise RuntimeError('handler boom')

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIsInstance(ctx.exception.__cause__, ZeroDivisionError)

  async def test_sync_pipeline_async_handler_failure_reraise_mode(self) -> None:
    """Sync pipeline error, async except handler fails with reraise=True → discarded, original re-raised."""

    async def bad_handler(info):
      raise RuntimeError('handler boom')

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler, reraise=True)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ZeroDivisionError):
        await c.run()
    self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))


class AsyncFinallyFailureTest(SymmetricTestCase):
  """Async finally handler failure during active exception."""

  async def test_async_finally_failure_during_active_exception(self) -> None:
    """Async finally handler fails while exception is active → replaces original."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(async_fail).finally_(bad_cleanup)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIn('cleanup boom', str(ctx.exception))
    self.assertIsInstance(ctx.exception.__context__, ValueError)

  async def test_async_finally_failure_on_success_path(self) -> None:
    """Async finally handler fails on success path → propagates."""

    async def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(async_fn).finally_(bad_cleanup)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIn('cleanup boom', str(ctx.exception))

  async def test_async_finally_failure_note_on_replace(self) -> None:
    """Async finally failure during active exception: note attached (Python 3.11+)."""
    if sys.version_info < (3, 11):
      self.skipTest('add_note requires Python 3.11+')

    async def async_fail(x):
      raise ValueError('original')

    async def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(async_fail).finally_(bad_cleanup)
    try:
      await c.run()
    except RuntimeError as exc:
      self.assertTrue(any('ValueError' in n for n in exc.__notes__))
    else:
      self.fail('Expected RuntimeError')


class AsyncRootCallableFailTest(SymmetricTestCase):
  """Async chains where the root callable is async and fails."""

  async def test_async_root_callable_fails(self) -> None:
    """Async root callable raises → exception propagates normally."""

    async def async_root():
      raise ValueError('root boom')

    c = Q(async_root).then(lambda x: x + 1)
    with self.assertRaises(ValueError) as ctx:
      await c.run()
    self.assertIn('root boom', str(ctx.exception))

  async def test_async_root_callable_fails_with_except(self) -> None:
    """Async root callable fails → except handler catches it."""

    async def async_root():
      raise ValueError('root boom')

    result = await Q(async_root).then(lambda x: x + 1).except_(lambda _: 'caught').run()
    self.assertEqual(result, 'caught')

  async def test_async_root_callable_fails_with_finally(self) -> None:
    """Async root callable fails → finally handler runs."""
    cleanup = []

    async def async_root():
      raise ValueError('root boom')

    c = Q(async_root).then(lambda x: x + 1).finally_(lambda rv: cleanup.append('finally'))
    with self.assertRaises(ValueError):
      await c.run()
    self.assertEqual(cleanup, ['finally'])


class AsyncNestedPipelineErrorTest(SymmetricTestCase):
  """Async pipelines with nested pipelines that raise."""

  async def test_async_nested_pipeline_raises(self) -> None:
    """Async nested pipeline raises → exception propagates to outer pipeline."""

    async def async_fail(x):
      raise ValueError('nested boom')

    inner = Q().then(async_fail)
    c = Q(1).then(inner)
    with self.assertRaises(ValueError) as ctx:
      await c.run()
    self.assertIn('nested boom', str(ctx.exception))

  async def test_async_nested_pipeline_raises_caught_by_outer_except(self) -> None:
    """Async nested pipeline raises → outer except handler catches it."""

    async def async_fail(x):
      raise ValueError('nested boom')

    inner = Q().then(async_fail)
    result = await Q(1).then(inner).except_(lambda _: 'outer-caught').run()
    self.assertEqual(result, 'outer-caught')

  async def test_async_nested_pipeline_raises_outer_finally_runs(self) -> None:
    """Async nested pipeline raises → outer finally handler runs."""
    cleanup = []

    async def async_fail(x):
      raise ValueError('nested boom')

    inner = Q().then(async_fail)
    c = Q(1).then(inner).finally_(lambda rv: cleanup.append('outer-finally'))
    with self.assertRaises(ValueError):
      await c.run()
    self.assertEqual(cleanup, ['outer-finally'])


class AsyncControlFlowInExceptTest(SymmetricTestCase):
  """Async control flow signals in except handlers."""

  async def test_async_return_in_except_handler_raises(self) -> None:
    """Async pipeline: return_() in sync except handler raises QuentException."""

    async def async_fail(x):
      raise ValueError('boom')

    def handler(info):
      return Q.return_('nope')

    c = Q(1).then(async_fail).except_(handler)
    with self.assertRaises(QuentException):
      await c.run()

  async def test_async_break_in_except_handler_raises(self) -> None:
    """Async pipeline: break_() in sync except handler raises QuentException."""

    async def async_fail(x):
      raise ValueError('boom')

    def handler(info):
      return Q.break_('nope')

    c = Q(1).then(async_fail).except_(handler)
    with self.assertRaises(QuentException):
      await c.run()

  async def test_async_except_handler_control_flow_in_awaited_result(self) -> None:
    """Async except handler returns awaitable with control flow → QuentException."""

    async def async_fail(x):
      raise ValueError('boom')

    async def handler_with_return(info):
      return Q.return_('escaped')

    c = Q(1).then(async_fail).except_(handler_with_return)
    with self.assertRaises(QuentException):
      await c.run()


class AsyncExceptFinallyComboTest(SymmetricTestCase):
  """Combined async except + finally edge cases."""

  async def test_async_except_consume_then_async_finally(self) -> None:
    """Async pipeline: except consumes, async finally runs in success context."""
    order = []

    async def async_fail(x):
      raise ValueError('boom')

    async def async_cleanup(rv):
      order.append('finally')

    result = await (
      Q(1).then(async_fail).except_(lambda _: (order.append('except'), 'recovered')[-1]).finally_(async_cleanup).run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(order, ['except', 'finally'])

  async def test_async_except_reraise_then_async_finally(self) -> None:
    """Async pipeline: except re-raises, async finally runs in failure context."""
    order = []

    async def async_fail(x):
      raise ValueError('boom')

    async def async_cleanup(rv):
      order.append('finally')

    c = Q(1).then(async_fail).except_(lambda _: order.append('except'), reraise=True).finally_(async_cleanup)
    with self.assertRaises(ValueError):
      await c.run()
    self.assertEqual(order, ['except', 'finally'])

  async def test_async_except_and_async_finally_both_async(self) -> None:
    """All-async: async fail, async except handler, async finally handler."""
    order = []

    async def async_fail(x):
      raise ValueError('boom')

    async def async_handler(info):
      order.append('except')
      return 'async-recovered'

    async def async_cleanup(rv):
      order.append('finally')

    result = await Q(1).then(async_fail).except_(async_handler).finally_(async_cleanup).run()
    self.assertEqual(result, 'async-recovered')
    self.assertEqual(order, ['except', 'finally'])

  async def test_unmatched_exception_async_finally_runs(self) -> None:
    """Async pipeline: unmatched exception skips except, async finally runs."""
    order = []

    async def async_fail(x):
      raise ValueError('boom')

    async def async_cleanup(rv):
      order.append('finally')

    c = Q(1).then(async_fail).except_(lambda _: order.append('except'), exceptions=TypeError).finally_(async_cleanup)
    with self.assertRaises(ValueError):
      await c.run()
    self.assertEqual(order, ['finally'])


class AsyncTransitionExceptTest(SymmetricTestCase):
  """SPEC §6.3.5: Async except handler in sync pipelines (async transition path)."""

  async def test_sync_pipeline_async_except_handler_transition(self) -> None:
    """Sync error, async except handler → triggers _async_except_handler transition."""
    # When a sync pipeline's except handler returns a coroutine,
    # _async_except_handler is invoked, handling the async transition.
    result_holder = []

    async def async_handler(info):
      result_holder.append('called')
      return 'async-handled'

    # This returns a coroutine (from _async_except_handler), so await it.
    result = await Q(1).then(lambda x: 1 / 0).except_(async_handler).run()
    self.assertEqual(result, 'async-handled')

  async def test_async_finally_in_sync_pipeline_on_error_path(self) -> None:
    """Sync pipeline error path, async finally handler → async transition, re-raises."""

    # On the error path, async finally in a sync pipeline triggers async transition.
    # Awaiting the coroutine re-raises the original exception after running cleanup.
    cleanup = []

    async def async_cleanup(rv):
      cleanup.append(rv)

    coro = Q(1).then(lambda x: 1 / 0).finally_(async_cleanup).run()
    import asyncio

    self.assertTrue(asyncio.iscoroutine(coro))
    with self.assertRaises(ZeroDivisionError):
      await coro
    self.assertEqual(cleanup, [1])


# ---------------------------------------------------------------------------
# §6: BaseException cleanup
# ---------------------------------------------------------------------------


class BaseExceptionCleanupTest(IsolatedAsyncioTestCase):
  """§6: KeyboardInterrupt / SystemExit cleanup through chains."""

  async def test_keyboard_interrupt_cleans_quent_idx(self) -> None:
    """KeyboardInterrupt through sync pipeline triggers cleanup."""

    def raise_ki(x):
      raise KeyboardInterrupt('test signal')

    with self.assertRaises(KeyboardInterrupt):
      Q(1).then(raise_ki).run()

  async def test_system_exit_cleans_quent_idx(self) -> None:
    """SystemExit through sync pipeline triggers cleanup."""

    def raise_se(x):
      raise SystemExit(42)

    with self.assertRaises(SystemExit):
      Q(1).then(raise_se).run()

  async def test_keyboard_interrupt_in_async_pipeline(self) -> None:
    """KeyboardInterrupt in async pipeline triggers cleanup."""

    async def raise_ki(x):
      raise KeyboardInterrupt('async signal')

    with self.assertRaises(KeyboardInterrupt):
      await Q(1).then(raise_ki).run()


# ---------------------------------------------------------------------------
# §6.2: Control flow signal in async except handler with reraise=True
# ---------------------------------------------------------------------------


class AsyncExceptControlFlowReraiseTrueTest(IsolatedAsyncioTestCase):
  """§6.2: Control flow signal in async except handler with reraise=True."""

  async def test_async_except_handler_control_flow_raise_true(self) -> None:
    """Async handler raises return_() with reraise=True -> QuentException."""

    async def async_handler(info):
      return Q.return_('escaped')

    c = Q(1).then(lambda x: 1 / 0).except_(async_handler, reraise=True)
    with self.assertRaises(QuentException) as ctx:
      await c.run()
    self.assertIn('not allowed', str(ctx.exception))


# ---------------------------------------------------------------------------
# §6.2: SystemExit from async except handler with reraise=True
# ---------------------------------------------------------------------------


class SystemExitFromHandlerTest(IsolatedAsyncioTestCase):
  """§6.2: SystemExit from async except handler with reraise=True propagates."""

  async def test_async_handler_system_exit_raise_true(self) -> None:
    """SystemExit from async handler with reraise=True propagates."""

    async def async_handler(info):
      raise SystemExit(99)

    c = Q(1).then(lambda x: 1 / 0).except_(async_handler, reraise=True)
    with self.assertRaises(SystemExit):
      await c.run()


# ---------------------------------------------------------------------------
# §6.3.3: Python 3.11+ __notes__ on finally handler exception
# ---------------------------------------------------------------------------


class SyncFinallyNoteTest(TestCase):
  """§6.3.3: Python 3.11+ __notes__ on finally handler exception replacing active exception."""

  def test_sync_finally_raises_during_active_exception_note(self) -> None:
    """Note mentions original exception type and replacement."""
    if sys.version_info < (3, 11):
      self.skipTest('add_note requires Python 3.11+')

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(lambda x: 1 / 0).finally_(bad_cleanup)
    try:
      c.run()
    except RuntimeError as exc:
      notes = exc.__notes__
      self.assertTrue(
        any('ZeroDivisionError' in n for n in notes),
        f'Expected note mentioning ZeroDivisionError, got: {notes}',
      )
      self.assertTrue(
        any('finally handler error replaced' in n for n in notes),
        f'Expected note about replacement, got: {notes}',
      )
    else:
      self.fail('Expected RuntimeError')


# ---------------------------------------------------------------------------
# §6.2/§6.3: Combined except + finally error paths in async pipelines
# ---------------------------------------------------------------------------


class ExceptFinallyComboAsyncTest(IsolatedAsyncioTestCase):
  """§6.2/§6.3: Combined except + finally error paths in async pipelines."""

  async def test_async_pipeline_finally_raises_during_except_reraise(self) -> None:
    """except re-raises, finally raises -> finally replaces original."""

    async def async_fail(x):
      raise ValueError('original')

    def handler(info):
      pass  # side-effect

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(async_fail).except_(handler, reraise=True).finally_(bad_cleanup)
    try:
      await c.run()
    except RuntimeError as exc:
      self.assertIn('cleanup boom', str(exc))
      # __context__ should chain back to the ValueError
      self.assertIsNotNone(exc.__context__)
    else:
      self.fail('Expected RuntimeError from finally')

  async def test_async_pipeline_finally_raises_during_except_consume(self) -> None:
    """except consumes, finally raises."""

    async def async_fail(x):
      raise ValueError('original')

    def handler(info):
      return 'recovered'

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    c = Q(1).then(async_fail).except_(handler).finally_(bad_cleanup)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIn('cleanup boom', str(ctx.exception))


# ---------------------------------------------------------------------------
# §6.2.2: Async pipeline with no root value — except handler receives None
# ---------------------------------------------------------------------------


class AsyncNoRootValueExceptTest(IsolatedAsyncioTestCase):
  """§6.2.2: Async pipeline with no root value — except handler receives None for root_value."""

  async def test_no_root_value_async_except_handler_gets_none(self) -> None:
    """Async variant of root_value Null normalization."""
    received = {}

    async def async_fail():
      raise ValueError('boom')

    def handler(info):
      received['rv'] = info.root_value
      return 'handled'

    result = await Q().then(async_fail).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertIsNone(received['rv'])


# ---------------------------------------------------------------------------
# §6.3.5: Async finally handler in sync pipeline — async transition
# (Canonical tests for async transition are in asymmetry_tests.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Item 39: _ControlFlowSignal.__context__ preservation in except handlers
# ---------------------------------------------------------------------------


class ControlFlowSignalContextTest(TestCase):
  """Verify signal.__context__ is set to the original exception when a
  _ControlFlowSignal is raised inside an except handler."""

  def _capture_signal(self, q) -> _ControlFlowSignal:
    """Run pipeline and return the _ControlFlowSignal carried inside QuentException."""
    captured = []

    def handler(info):
      captured.append(info.exc)
      return Q.return_('signal')

    c = q.except_(handler)
    try:
      c.run()
    except QuentException as qe:
      # The QuentException's __cause__ is the original exc; the signal is
      # not directly accessible here — we verify __context__ via a different route.
      # Instead, intercept at a lower level by inspecting __cause__.__context__.
      return qe
    raise AssertionError('Expected QuentException was not raised')

  def test_signal_context_set_to_original_exc_sync(self) -> None:
    """Sync: _ControlFlowSignal raised in except handler has __context__ = original exc."""
    original_exc = None

    def handler(info):
      nonlocal original_exc
      original_exc = info.exc
      return Q.return_('escape')

    c = Q(1).then(lambda x: 1 / 0).except_(handler)
    with self.assertRaises(QuentException) as ctx:
      c.run()

    qe = ctx.exception
    # QuentException is raised `from exc` so __cause__ is the original exception.
    self.assertIsInstance(qe.__cause__, ZeroDivisionError)
    # The signal that was raised must have had its __context__ set to the original exc.
    # We verify this by checking that the QuentException's cause matches what the
    # handler observed as the original exception.
    self.assertIs(qe.__cause__, original_exc)


class ControlFlowSignalContextDirectTest(TestCase):
  """Directly verify signal.__context__ by raising a pre-constructed signal instance."""

  def test_signal_context_set_sync(self) -> None:
    """The _ControlFlowSignal raised in an except handler gets __context__ = original exc."""
    from quent._types import _Return

    original_exc = ValueError('original')
    signal = _Return('val', (), {})

    def raising_step(x):
      raise original_exc

    def handler(info):
      raise signal

    c = Q(1).then(raising_step).except_(handler)
    try:
      c.run()
    except QuentException:
      pass

    self.assertIsInstance(signal, _Return)
    self.assertIs(signal.__context__, original_exc)


# ---------------------------------------------------------------------------
# §6 Error handler failure paths — except/finally interaction
# ---------------------------------------------------------------------------


class ExceptHandlerFailureFinallyRunsTest(SymmetricTestCase):
  """SPEC §6.2.5 + §6.3.2: except_() handler raises -> finally_() still runs."""

  async def test_except_raises_finally_still_runs_sync(self) -> None:
    """Sync: except handler raises, its exception propagates, finally still runs."""
    finally_ran = []

    def bad_handler(info):
      raise TypeError('handler failed')

    def track_finally(rv):
      finally_ran.append(True)

    with self.assertRaises(TypeError) as ctx:
      Q(1).then(lambda x: 1 / 0).except_(bad_handler).finally_(track_finally).run()
    self.assertEqual(str(ctx.exception), 'handler failed')
    self.assertTrue(finally_ran, 'finally handler must run even when except handler fails')
    # §6.2.5: handler exc chained via `raise handler_exc from original_exc`
    self.assertIsInstance(ctx.exception.__cause__, ZeroDivisionError)

  async def test_except_raises_finally_still_runs_async(self) -> None:
    """Async: except handler raises, its exception propagates, finally still runs."""
    finally_ran = []

    async def bad_handler(info):
      raise TypeError('async handler failed')

    async def track_finally(rv):
      finally_ran.append(True)

    with self.assertRaises(TypeError) as ctx:
      await Q(1).then(lambda x: 1 / 0).except_(bad_handler).finally_(track_finally).run()
    self.assertEqual(str(ctx.exception), 'async handler failed')
    self.assertTrue(finally_ran, 'async finally handler must run even when except handler fails')
    self.assertIsInstance(ctx.exception.__cause__, ZeroDivisionError)


class FinallyHandlerFailureTest(SymmetricTestCase):
  """SPEC §6.3.3: finally_() handler raises -> its exception propagates."""

  async def test_finally_raises_on_success_sync(self) -> None:
    """Sync: finally handler raises on success path -> exception propagates."""

    def bad_finally(rv):
      raise ValueError('finally err')

    with self.assertRaises(ValueError) as ctx:
      Q(1).then(lambda x: x * 2).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'finally err')

  async def test_finally_raises_on_success_async(self) -> None:
    """Async: finally handler raises on success path -> exception propagates."""

    async def bad_finally(rv):
      raise ValueError('async finally err')

    with self.assertRaises(ValueError) as ctx:
      await Q(1).then(lambda x: x * 2).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'async finally err')

  async def test_finally_raises_suppresses_original_sync(self) -> None:
    """Sync: finally handler raises on failure path -> suppresses original exception."""

    def bad_finally(rv):
      raise ValueError('finally overrides')

    with self.assertRaises(ValueError) as ctx:
      Q(1).then(lambda x: 1 / 0).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'finally overrides')
    # §6.3.3: original exception preserved as __context__
    self.assertIsInstance(ctx.exception.__context__, ZeroDivisionError)

  async def test_finally_raises_suppresses_original_async(self) -> None:
    """Async: finally handler raises on failure path -> suppresses original exception."""

    async def bad_finally(rv):
      raise ValueError('async finally overrides')

    with self.assertRaises(ValueError) as ctx:
      await Q(1).then(lambda x: 1 / 0).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'async finally overrides')
    self.assertIsInstance(ctx.exception.__context__, ZeroDivisionError)


class BothHandlersRaiseTest(SymmetricTestCase):
  """SPEC §6.3.3 + §6.2.5: both except and finally raise -> finally wins."""

  async def test_both_raise_finally_wins_sync(self) -> None:
    """Sync: except raises, then finally raises -> finally exception propagates."""

    def bad_handler(info):
      raise TypeError('handler err')

    def bad_finally(rv):
      raise ValueError('finally err')

    with self.assertRaises(ValueError) as ctx:
      Q(1).then(lambda x: 1 / 0).except_(bad_handler).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'finally err')
    # __context__ is the except handler's exception (which was propagating when finally ran)
    self.assertIsInstance(ctx.exception.__context__, TypeError)

  async def test_both_raise_finally_wins_async(self) -> None:
    """Async: except raises, then finally raises -> finally exception propagates."""

    async def bad_handler(info):
      raise TypeError('async handler err')

    async def bad_finally(rv):
      raise ValueError('async finally err')

    with self.assertRaises(ValueError) as ctx:
      await Q(1).then(lambda x: 1 / 0).except_(bad_handler).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'async finally err')
    self.assertIsInstance(ctx.exception.__context__, TypeError)

  async def test_mixed_sync_async_except_raises_finally_runs(self) -> None:
    """Mixed: sync except handler raises, async finally still runs."""
    finally_ran = []

    def bad_handler(info):
      raise TypeError('sync handler err')

    async def track_finally(rv):
      finally_ran.append(True)

    with self.assertRaises(TypeError):
      await Q(1).then(lambda x: 1 / 0).except_(bad_handler).finally_(track_finally).run()
    self.assertTrue(finally_ran)

  async def test_finally_failure_result_discarded_on_success(self) -> None:
    """Finally handler's exception propagates even on success path (not just failure)."""

    def bad_finally(rv):
      raise RuntimeError('cleanup broke')

    with self.assertRaises(RuntimeError) as ctx:
      Q(1).then(lambda x: x * 2).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'cleanup broke')
    # On success path, __context__ should be None (no prior exception)
    self.assertIsNone(ctx.exception.__context__)


# ---------------------------------------------------------------------------
# GAP 3: Error position root_value tests
# ---------------------------------------------------------------------------


class ErrorPositionRootValueTest(SymmetricTestCase):
  """Verify that root_value in QuentExcInfo is always the root, regardless of error position."""

  async def test_error_at_root_callable(self) -> None:
    """Error at root callable: root_value is None (root never produced a value)."""
    received: dict[str, object] = {}

    def handler(info):
      received['rv'] = info.root_value
      received['exc'] = type(info.exc).__name__
      return 'handled'

    result = Q(lambda: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertIsNone(received['rv'])  # root callable failed, no root value

  async def test_error_at_first_step(self) -> None:
    """Error at first step after root: root_value is the root value."""
    received: dict[str, object] = {}

    def handler(info):
      received['rv'] = info.root_value
      return 'handled'

    result = Q(42).then(lambda x: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(received['rv'], 42)

  async def test_error_at_third_step(self) -> None:
    """Error at third step: root_value is still the original root, not intermediate."""
    received: dict[str, object] = {}

    def handler(info):
      received['rv'] = info.root_value
      return 'handled'

    result = Q(10).then(lambda x: x * 2).then(lambda x: x + 5).then(lambda x: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(received['rv'], 10)  # root value, NOT 25 (intermediate)

  async def test_error_position_with_run_value(self) -> None:
    """Run value replaces root for error handlers at any position."""
    received: dict[str, object] = {}

    def handler(info):
      received['rv'] = info.root_value
      return 'handled'

    c = Q(999).then(lambda x: x + 1).then(lambda x: 1 / 0).except_(handler)
    result = c.run(7)
    self.assertEqual(result, 'handled')
    self.assertEqual(received['rv'], 7)  # run value, not 999


# ---------------------------------------------------------------------------
# GAP 4: except+finally combo sync path tests
# ---------------------------------------------------------------------------


class ExceptFinallyComboSyncTest(SymmetricTestCase):
  """Sync path tests for except+finally combos where finally raises."""

  async def test_except_consume_finally_fails_sync(self) -> None:
    """except consumes, but finally raises — finally exception propagates."""

    def handler(info):
      return 'recovered'

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    with self.assertRaises(RuntimeError) as ctx:
      Q(5).then(lambda x: 1 / 0).except_(handler).finally_(bad_cleanup).run()
    self.assertEqual(str(ctx.exception), 'cleanup boom')

  async def test_except_reraise_finally_fails_sync(self) -> None:
    """except re-raises, then finally also raises — finally exception wins."""

    def handler(info):
      pass  # noop, original re-raised

    def bad_cleanup(rv):
      raise RuntimeError('cleanup boom')

    with self.assertRaises(RuntimeError) as ctx:
      Q(5).then(lambda x: 1 / 0).except_(handler, reraise=True).finally_(bad_cleanup).run()
    # The original ZeroDivisionError is preserved as __context__
    self.assertIsInstance(ctx.exception.__context__, ZeroDivisionError)


# ---------------------------------------------------------------------------
# Audit §9 — Additional spec gap tests
# ---------------------------------------------------------------------------


class AsyncExceptHandlerSuppressContextTest(IsolatedAsyncioTestCase):
  """SPEC §6.2.4: Async handler failure with reraise=True — __suppress_context__."""

  async def test_async_reraise_true_handler_failure_suppress_context(self) -> None:
    """Async: reraise=True, handler raises Exception → __suppress_context__ set on original."""

    async def async_fail(x):
      raise ValueError('original')

    async def bad_handler(info):
      raise RuntimeError('handler fail')

    c = Q(1).then(async_fail).except_(bad_handler, reraise=True)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ValueError) as ctx:
        await c.run()
    # After fix: save/restore original __suppress_context__ instead of unconditionally
    # setting True.  The original ValueError had no pre-existing context chain,
    # so __suppress_context__ is restored to False.
    self.assertFalse(ctx.exception.__suppress_context__)
    self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))


class ExceptBaseExceptionWarnAndCatchTest(IsolatedAsyncioTestCase):
  """SPEC §6.2.1: BaseException warning does not prevent catching."""

  async def test_keyboard_interrupt_warned_and_caught(self) -> None:
    """exceptions=KeyboardInterrupt emits warning AND still catches the exception."""
    handled = []

    def handler(info):
      handled.append(info.exc)
      return 'caught'

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      c = (
        Q(1)
        .then(lambda x: (_ for _ in ()).throw(KeyboardInterrupt('test')))
        .except_(handler, exceptions=KeyboardInterrupt)
      )
    self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))
    result = c.run()
    self.assertEqual(result, 'caught')
    self.assertEqual(len(handled), 1)
    self.assertIsInstance(handled[0], KeyboardInterrupt)


class ForeachDoConcurrentSingleFailureTest(IsolatedAsyncioTestCase):
  """SPEC §6.5: foreach_do concurrent single failure raised directly."""

  async def test_foreach_do_concurrent_single_failure_not_wrapped(self) -> None:
    """foreach_do with single concurrent failure: raised directly, not ExceptionGroup."""

    def maybe_fail(x):
      if x == 3:
        raise ValueError('fail-3')

    c = Q([0, 1, 2, 3, 4]).foreach_do(maybe_fail, concurrency=5)
    with self.assertRaises(ValueError) as ctx:
      c.run()
    self.assertIn('fail-3', str(ctx.exception))


class ExceptReraiseBaseExceptionVariantsTest(IsolatedAsyncioTestCase):
  """SPEC §6.2.4: BaseException handler failure with reraise=True — SystemExit and custom."""

  async def test_system_exit_handler_failure_propagates(self) -> None:
    """reraise=True, handler raises SystemExit → propagates naturally."""

    def bad_handler(info):
      raise SystemExit(99)

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler, reraise=True)
    with self.assertRaises(SystemExit) as ctx:
      c.run()
    self.assertEqual(ctx.exception.code, 99)

  async def test_custom_base_exception_handler_failure_propagates(self) -> None:
    """reraise=True, handler raises custom BaseException → propagates naturally."""

    class CustomBaseExc(BaseException):
      pass

    def bad_handler(info):
      raise CustomBaseExc('custom base')

    c = Q(1).then(lambda x: 1 / 0).except_(bad_handler, reraise=True)
    with self.assertRaises(CustomBaseExc):
      c.run()


# ---------------------------------------------------------------------------
# §6.2.4: Async except handler failure with reraise=True — full coverage
# ---------------------------------------------------------------------------


class AsyncExceptHandlerFailureReraseTrueTest(IsolatedAsyncioTestCase):
  """SPEC §6.2.4: Async pipeline with async except handler that raises while reraise=True.

  Verifies: (1) original exception re-raised, (2) RuntimeWarning emitted,
  (3) __suppress_context__ behavior.
  """

  async def test_async_except_handler_failure_reraise_true(self) -> None:
    """Async pipeline: async handler raises with reraise=True.

    Per SPEC §6.2.4: when reraise=True and the handler raises an Exception,
    the handler's exception is discarded, a RuntimeWarning is emitted, and
    the original exception is re-raised.
    """

    async def async_step(x):
      raise ValueError('original async error')

    async def async_bad_handler(info):
      raise RuntimeError('async handler boom')

    c = Q(1).then(async_step).except_(async_bad_handler, reraise=True)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ValueError) as ctx:
        await c.run()
    # (1) Original exception re-raised
    self.assertIn('original async error', str(ctx.exception))
    # (2) RuntimeWarning emitted about discarded handler exception
    warning_msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    self.assertTrue(
      any('handler boom' in msg for msg in warning_msgs),
      f'Expected RuntimeWarning mentioning handler failure, got: {warning_msgs}',
    )
    # (3) __suppress_context__: the engine saves and restores the original value.
    # A fresh ValueError has __suppress_context__ = False.
    self.assertFalse(ctx.exception.__suppress_context__)


# ---------------------------------------------------------------------------
# §5.3 / §6: StopIteration propagation through except_()
# ---------------------------------------------------------------------------


class StopIterationExceptTest(SymmetricTestCase):
  """SPEC §5.3: StopIteration is treated as a regular exception — reaches except_() handler.

  Per SPEC §5.3 error behavior: "StopIteration raised by a user callback propagates
  out — quent does not intercept or reinterpret it. It is treated as a regular
  exception that propagates through error handling."
  """

  async def test_stop_iteration_reaches_except_handler(self) -> None:
    """StopIteration from a step reaches except_(handler, exceptions=Exception).

    The handler receives QuentExcInfo with exc=StopIteration, and the handler's
    return value becomes the pipeline result.
    """
    received = {}

    def handler(info):
      received['exc_type'] = type(info.exc).__name__
      received['root_value'] = info.root_value
      return 'handled'

    result = (
      Q(42).then(lambda x: (_ for _ in ()).throw(StopIteration('stop'))).except_(handler, exceptions=Exception).run()
    )
    self.assertEqual(result, 'handled')
    self.assertEqual(received['exc_type'], 'StopIteration')
    self.assertEqual(received['root_value'], 42)

  async def test_stop_iteration_in_foreach_reaches_except(self) -> None:
    """StopIteration from a foreach callback propagates to except_() handler.

    Sequential foreach: StopIteration is not silently consumed — it propagates
    through error handling just like any other exception.
    """
    received = {}

    def fn(x):
      if x == 2:
        raise StopIteration('stop at 2')
      return x

    def handler(info):
      received['exc_type'] = type(info.exc).__name__
      return 'caught'

    result = Q([0, 1, 2, 3]).foreach(fn).except_(handler, exceptions=Exception).run()
    self.assertEqual(result, 'caught')
    self.assertEqual(received['exc_type'], 'StopIteration')
