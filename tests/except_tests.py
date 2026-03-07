"""Tests for Chain.except_(): registration, execution, async paths, and edge cases."""
from __future__ import annotations

import asyncio
import unittest
import warnings

from quent import Chain, Null, QuentException
from quent._core import _Return, _Break
from helpers import raise_fn, async_raise_fn, sync_fn, async_fn, sync_identity, async_identity


# -- Local helpers --

def _catch_handler(rv, exc):
  return f'caught:{type(exc).__name__}'


async def _async_catch_handler(rv, exc):
  return f'async_caught:{type(exc).__name__}'


def _reraise_handler(rv, exc):
  raise RuntimeError('handler error') from exc


def _raise_type_error(x=None):
  raise TypeError('type error')


def _raise_key_error(x=None):
  raise KeyError('key error')


async def _async_raise_fn_value_error(x=None):
  raise ValueError('async error')


class TestExceptRegistration(unittest.TestCase):

  def test_register_handler(self):
    c = Chain().except_(_catch_handler)
    self.assertIsNotNone(c.on_except_link)
    self.assertIs(c.on_except_link.v, _catch_handler)

  def test_default_catches_exception(self):
    c = Chain().except_(_catch_handler)
    self.assertEqual(c.on_except_exceptions, (Exception,))

  def test_specific_exception_type(self):
    c = Chain().except_(_catch_handler, exceptions=ValueError)
    self.assertEqual(c.on_except_exceptions, (ValueError,))

  def test_multiple_exception_types_list(self):
    c = Chain().except_(_catch_handler, exceptions=[ValueError, TypeError])
    self.assertEqual(c.on_except_exceptions, (ValueError, TypeError))

  def test_tuple_of_exceptions(self):
    c = Chain().except_(_catch_handler, exceptions=(ValueError, TypeError))
    self.assertEqual(c.on_except_exceptions, (ValueError, TypeError))

  def test_double_except_raises_quent_exception(self):
    c = Chain().except_(_catch_handler)
    with self.assertRaises(QuentException) as cm:
      c.except_(_catch_handler)
    self.assertIn('one', str(cm.exception).lower())

  def test_string_as_exceptions_raises_type_error(self):
    with self.assertRaises(TypeError) as cm:
      Chain().except_(_catch_handler, exceptions='ValueError')
    self.assertIn('ValueError', str(cm.exception))

  def test_empty_iterable_raises_quent_exception(self):
    with self.assertRaises(QuentException) as cm:
      Chain().except_(_catch_handler, exceptions=[])
    self.assertIn('at least one', str(cm.exception).lower())

  def test_single_type_not_in_list(self):
    c = Chain().except_(_catch_handler, exceptions=ValueError)
    # Should be wrapped in a tuple for isinstance usage
    self.assertIsInstance(c.on_except_exceptions, tuple)
    self.assertEqual(c.on_except_exceptions, (ValueError,))

  def test_exception_inheritance_matching(self):
    # exceptions=Exception should catch ValueError (a subclass)
    result = Chain(raise_fn).except_(_catch_handler, exceptions=Exception).run()
    self.assertEqual(result, 'caught:ValueError')

  def test_except_returns_chain_for_fluency(self):
    c = Chain()
    result = c.except_(_catch_handler)
    self.assertIs(result, c)

  def test_generator_as_exceptions_iterable(self):
    # Any iterable (not just list/tuple) should work
    def exc_gen():
      yield ValueError
      yield TypeError
    c = Chain().except_(_catch_handler, exceptions=exc_gen())
    self.assertEqual(c.on_except_exceptions, (ValueError, TypeError))

  def test_set_of_exceptions(self):
    c = Chain().except_(_catch_handler, exceptions={ValueError, TypeError})
    self.assertIsInstance(c.on_except_exceptions, tuple)
    self.assertEqual(set(c.on_except_exceptions), {ValueError, TypeError})


class TestExceptExecution(unittest.TestCase):

  def test_handler_called_on_exception(self):
    called = []
    def handler(rv, exc):
      called.append((rv, exc))
      return 'handled'
    result = Chain(raise_fn).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(called), 1)
    self.assertIsNone(called[0][0])
    self.assertIsInstance(called[0][1], ValueError)

  def test_handler_return_becomes_result(self):
    result = Chain(raise_fn).except_(lambda rv, e: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_handler_raises_new_exception(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain(raise_fn).except_(_reraise_handler).run()
    self.assertEqual(str(cm.exception), 'handler error')
    # The original ValueError should be chained as __cause__
    self.assertIsInstance(cm.exception.__cause__, ValueError)

  def test_no_exception_handler_not_called(self):
    called = []
    def handler(rv, exc):
      called.append((rv, exc))
      return 'handled'
    result = Chain(42).except_(handler).run()
    self.assertEqual(result, 42)
    self.assertEqual(len(called), 0)

  def test_exception_not_matching_reraises(self):
    # Handler registered for TypeError, but ValueError is raised
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(_catch_handler, exceptions=TypeError).run()

  def test_base_exception_not_caught(self):
    # KeyboardInterrupt is a BaseException, not Exception
    def raise_keyboard_interrupt(x=None):
      raise KeyboardInterrupt()
    with self.assertRaises(KeyboardInterrupt):
      Chain(raise_keyboard_interrupt).except_(_catch_handler).run()

  def test_control_flow_in_handler_raises_quent_exception(self):
    def handler(rv, exc):
      Chain.return_('val')
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(handler).run()
    self.assertIn('control flow', str(cm.exception).lower())

  def test_break_in_handler_raises_quent_exception(self):
    def handler(rv, exc):
      Chain.break_('val')
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(handler).run()
    self.assertIn('control flow', str(cm.exception).lower())

  def test_handler_receives_root_value_and_exception(self):
    received = []
    def handler(rv, exc):
      received.append((rv, exc))
      return 'ok'
    Chain(raise_fn).except_(handler).run()
    self.assertEqual(len(received), 1)
    rv, exc = received[0]
    self.assertIsNone(rv)
    self.assertIsInstance(exc, ValueError)
    self.assertEqual(str(exc), 'test error')

  def test_handler_receives_actual_root_value(self):
    received = []
    def handler(rv, exc):
      received.append((rv, exc))
      return 'ok'
    Chain(42).then(raise_fn).except_(handler).run()
    self.assertEqual(len(received), 1)
    rv, exc = received[0]
    self.assertEqual(rv, 42)
    self.assertIsInstance(exc, ValueError)

  def test_handler_with_extra_args(self):
    # except_(handler, 'extra1', 'extra2') passes those as positional args
    received = []
    def handler(a, b):
      received.append((a, b))
      return 'ok'
    result = Chain(raise_fn).except_(handler, 'arg1', 'arg2').run()
    self.assertEqual(result, 'ok')
    self.assertEqual(received, [('arg1', 'arg2')])

  def test_handler_with_kwargs(self):
    received = []
    def handler(k=None):
      received.append(k)
      return 'ok'
    result = Chain(raise_fn).except_(handler, k='val').run()
    self.assertEqual(result, 'ok')
    self.assertEqual(received, ['val'])

  def test_exception_in_second_step_caught(self):
    result = (
      Chain(10)
      .then(raise_fn)
      .except_(lambda rv, e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_exception_in_middle_step_caught(self):
    result = (
      Chain(1)
      .then(sync_fn)       # 2
      .then(raise_fn)      # raises
      .then(sync_fn)       # never reached
      .except_(lambda rv, e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_handler_returning_none(self):
    result = Chain(raise_fn).except_(lambda rv, e: None).run()
    self.assertIsNone(result)

  def test_no_except_exception_propagates(self):
    with self.assertRaises(ValueError):
      Chain(raise_fn).run()

  def test_handler_for_type_error(self):
    result = (
      Chain(_raise_type_error)
      .except_(_catch_handler, exceptions=TypeError)
      .run()
    )
    self.assertEqual(result, 'caught:TypeError')

  def test_multiple_exception_types_first_matches(self):
    result = (
      Chain(raise_fn)
      .except_(_catch_handler, exceptions=[ValueError, TypeError])
      .run()
    )
    self.assertEqual(result, 'caught:ValueError')

  def test_multiple_exception_types_second_matches(self):
    result = (
      Chain(_raise_type_error)
      .except_(_catch_handler, exceptions=[ValueError, TypeError])
      .run()
    )
    self.assertEqual(result, 'caught:TypeError')


class TestExceptAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_handler_awaited(self):
    result = await (
      Chain(raise_fn)
      .then(async_fn)   # makes chain async before except runs
      .except_(_async_catch_handler)
      .run()
    )
    self.assertEqual(result, 'async_caught:ValueError')

  async def test_async_handler_return_value(self):
    async def handler(rv, exc):
      return 42
    result = await (
      Chain(raise_fn)
      .then(async_fn)
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 42)

  async def test_sync_handler_in_async_chain(self):
    result = await (
      Chain(async_raise_fn)
      .except_(lambda rv, e: 'sync_caught')
      .run()
    )
    self.assertEqual(result, 'sync_caught')

  async def test_exception_in_async_step_caught(self):
    result = await (
      Chain(10)
      .then(async_fn)          # 11
      .then(async_raise_fn)    # raises
      .except_(lambda rv, e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_async_handler_raises_propagates(self):
    async def handler(rv, exc):
      raise RuntimeError('async handler error') from exc
    with self.assertRaises(RuntimeError) as cm:
      await (
        Chain(async_raise_fn)
        .except_(handler)
        .run()
      )
    self.assertEqual(str(cm.exception), 'async handler error')

  async def test_async_exception_not_matching_reraises(self):
    with self.assertRaises(ValueError):
      await (
        Chain(async_raise_fn)
        .except_(_catch_handler, exceptions=TypeError)
        .run()
      )

  async def test_async_no_exception_handler_not_called(self):
    called = []
    async def handler(rv, exc):
      called.append((rv, exc))
    result = await Chain(10).then(async_fn).except_(handler).run()
    self.assertEqual(result, 11)
    self.assertEqual(len(called), 0)

  async def test_async_handler_receives_root_value_and_exception(self):
    received = []
    async def handler(rv, exc):
      received.append((rv, exc))
      return 'ok'
    await Chain(async_raise_fn).except_(handler).run()
    self.assertEqual(len(received), 1)
    rv, exc = received[0]
    self.assertIsNone(rv)
    self.assertIsInstance(exc, ValueError)
    self.assertEqual(str(exc), 'test error')

  async def test_control_flow_in_async_handler_raises_quent_exception(self):
    # When an async except handler raises a _Return, _run_async's defensive
    # guard catches _ControlFlowSignal and wraps it in QuentException,
    # preventing internal signals from leaking to user code.
    async def handler(rv, exc):
      Chain.return_('val')
    with self.assertRaises(QuentException):
      await Chain(async_raise_fn).except_(handler).run()


class TestExceptSyncReturnsCoroutine(unittest.TestCase):

  def test_fire_and_forget_warning_emitted(self):
    # A purely sync chain (no async steps before the exception) that raises,
    # with an async except handler, should emit a RuntimeWarning about
    # fire-and-forget scheduling.
    #
    # We need a running event loop for create_task to succeed.
    # IsolatedAsyncioTestCase provides one implicitly, but we want this to be
    # a sync test exercising the sync _run path. We use asyncio.run to provide
    # a loop around the sync call.
    async def _inner():
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = Chain(raise_fn).except_(_async_catch_handler).run()
        # The result should be a Task (fire-and-forget)
        self.assertIsInstance(result, asyncio.Task)
        # Await it to let it complete and avoid warnings
        await result
      # Check that a RuntimeWarning was emitted
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      self.assertGreaterEqual(len(runtime_warnings), 1)
      msg = str(runtime_warnings[0].message)
      self.assertIn('fire-and-forget', msg.lower())

    asyncio.run(_inner())

  def test_no_event_loop_raises_quent_exception(self):
    # When there is no running event loop and the sync chain's except handler
    # returns a coroutine, _ensure_future -> create_task raises RuntimeError,
    # which is caught and converted to QuentException.
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(_async_catch_handler).run()
    self.assertIn('no event loop', str(cm.exception).lower())


class TestExceptRaise(unittest.TestCase):

  def test_basic_raise_true_handler_runs_and_exception_propagates(self):
    """raise_=True: handler executes, then the original exception re-raises."""
    called = []
    def handler(rv, exc):
      called.append(exc)
      return 'should_be_discarded'
    with self.assertRaises(ValueError) as cm:
      Chain(raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(len(called), 1)
    self.assertIsInstance(called[0], ValueError)
    self.assertEqual(str(cm.exception), 'test error')

  def test_raise_false_default_swallows_exception(self):
    """Default raise_=False: handler runs, exception swallowed, handler result returned."""
    called = []
    def handler(rv, exc):
      called.append(exc)
      return 'handled'
    result = Chain(raise_fn).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(called), 1)

  def test_raise_false_explicit_swallows_exception(self):
    """Explicit raise_=False: same as default — handler result returned."""
    result = Chain(raise_fn).except_(lambda rv, e: 'caught', raise_=False).run()
    self.assertEqual(result, 'caught')

  def test_handler_side_effects_execute_before_reraise(self):
    """With raise_=True, verify the handler's side-effects run before re-raise."""
    tracker = []
    def handler(rv, exc):
      tracker.append(('handler_called', type(exc).__name__))
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(tracker, [('handler_called', 'ValueError')])

  def test_handler_return_value_discarded_with_raise_true(self):
    """With raise_=True, the handler's return value never becomes the chain result."""
    def handler(rv, exc):
      return 42
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, raise_=True).run()
    # If raise_=True didn't work, we'd get 42 back instead of the exception.

  def test_exception_instance_preserved(self):
    """The re-raised exception is the exact same instance as the original."""
    original_exc = ValueError('unique error 12345')
    def raiser(x=None):
      raise original_exc
    caught_in_handler = []
    def handler(rv, exc):
      caught_in_handler.append(exc)
      return 'discarded'
    with self.assertRaises(ValueError) as cm:
      Chain(raiser).except_(handler, raise_=True).run()
    self.assertIs(cm.exception, original_exc)
    self.assertIs(caught_in_handler[0], original_exc)

  def test_with_root_value(self):
    """Chain(42).then(raise_fn).except_(handler, raise_=True) — handler receives (42, exc)."""
    received = []
    def handler(rv, exc):
      received.append((rv, exc))
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(42).then(raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(len(received), 1)
    self.assertEqual(received[0][0], 42)
    self.assertIsInstance(received[0][1], ValueError)

  def test_without_root_value(self):
    """Chain().then(raise_fn).except_(handler, raise_=True) — handler receives (None, exc)."""
    received = []
    def handler(rv, exc):
      received.append((rv, exc))
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain().then(raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(len(received), 1)
    self.assertIsNone(received[0][0])
    self.assertIsInstance(received[0][1], ValueError)

  def test_with_run_value_as_root(self):
    """Chain().then(raise_fn).except_(handler, raise_=True).run(99) — handler receives (99, exc)."""
    received = []
    def handler(rv, exc):
      received.append((rv, exc))
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain().then(raise_fn).except_(handler, raise_=True).run(99)
    self.assertEqual(len(received), 1)
    self.assertEqual(received[0][0], 99)
    self.assertIsInstance(received[0][1], ValueError)

  def test_exception_type_filtering_matching(self):
    """exceptions=ValueError + raise_=True: matching exception triggers handler, then re-raises."""
    called = []
    def handler(rv, exc):
      called.append(exc)
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, exceptions=ValueError, raise_=True).run()
    self.assertEqual(len(called), 1)
    self.assertIsInstance(called[0], ValueError)

  def test_exception_type_filtering_non_matching(self):
    """exceptions=TypeError + raise_=True: non-matching ValueError — handler NOT called, exception propagates."""
    called = []
    def handler(rv, exc):
      called.append(exc)
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, exceptions=TypeError, raise_=True).run()
    self.assertEqual(len(called), 0)

  def test_exception_type_filtering_multiple_types(self):
    """exceptions=[ValueError, TypeError] + raise_=True: matching exception triggers handler."""
    called = []
    def handler(rv, exc):
      called.append(type(exc).__name__)
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, exceptions=[ValueError, TypeError], raise_=True).run()
    self.assertEqual(called, ['ValueError'])

  def test_interaction_with_finally(self):
    """raise_=True + finally_(): finally handler still runs after the re-raise."""
    finally_tracker = []
    handler_tracker = []
    def handler(rv, exc):
      handler_tracker.append('handler')
      return 'discarded'
    def cleanup(rv):
      finally_tracker.append('finally')
    with self.assertRaises(ValueError):
      Chain(42).then(raise_fn).except_(handler, raise_=True).finally_(cleanup).run()
    self.assertEqual(handler_tracker, ['handler'])
    self.assertEqual(finally_tracker, ['finally'])

  def test_finally_receives_root_value_with_raise_true(self):
    """raise_=True + finally_(): finally handler receives root value."""
    finally_received = []
    def handler(rv, exc):
      return 'discarded'
    def cleanup(rv):
      finally_received.append(rv)
    with self.assertRaises(ValueError):
      Chain(42).then(raise_fn).except_(handler, raise_=True).finally_(cleanup).run()
    self.assertEqual(finally_received, [42])

  def test_handler_with_explicit_args_and_raise_true(self):
    """except_(handler, 'a', 'b', raise_=True) — handler called with explicit args, then re-raises."""
    received = []
    def handler(a, b):
      received.append((a, b))
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, 'a', 'b', raise_=True).run()
    self.assertEqual(received, [('a', 'b')])

  def test_handler_with_ellipsis_and_raise_true(self):
    """except_(handler, ..., raise_=True) — handler called with no args, then re-raises."""
    received = []
    def handler():
      received.append('called_no_args')
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, ..., raise_=True).run()
    self.assertEqual(received, ['called_no_args'])

  def test_handler_with_kwargs_and_raise_true(self):
    """except_(handler, key='val', raise_=True) — handler called with kwargs, then re-raises."""
    received = []
    def handler(key=None):
      received.append(key)
      return 'discarded'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(handler, key='val', raise_=True).run()
    self.assertEqual(received, ['val'])

  def test_on_except_raise_attribute_set(self):
    """Verify on_except_raise attribute is correctly set."""
    c1 = Chain().except_(lambda rv, e: None, raise_=True)
    self.assertTrue(c1.on_except_raise)
    c2 = Chain().except_(lambda rv, e: None, raise_=False)
    self.assertFalse(c2.on_except_raise)
    c3 = Chain().except_(lambda rv, e: None)
    self.assertFalse(c3.on_except_raise)

  def test_exception_in_middle_step_with_raise_true(self):
    """Exception in middle of chain + raise_=True: handler runs, exception propagates."""
    called = []
    def handler(rv, exc):
      called.append(type(exc).__name__)
      return 'discarded'
    with self.assertRaises(ValueError):
      (
        Chain(1)
        .then(sync_fn)       # 2
        .then(raise_fn)      # raises
        .then(sync_fn)       # never reached
        .except_(handler, raise_=True)
        .run()
      )
    self.assertEqual(called, ['ValueError'])

  def test_sync_handler_returning_coroutine_with_raise_true(self):
    """Sync path: handler returns a coroutine + raise_=True. Coroutine is closed, exception re-raises."""
    async def async_handler(rv, exc):
      return 'async_result'
    with self.assertRaises(ValueError):
      Chain(raise_fn).except_(async_handler, raise_=True).run()


class TestExceptRaiseAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_basic_raise_true(self):
    """Async path: raise_=True — handler runs, exception re-raises."""
    called = []
    def handler(rv, exc):
      called.append(exc)
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(len(called), 1)
    self.assertIsInstance(called[0], ValueError)

  async def test_async_raise_false_default(self):
    """Async path: default raise_=False — handler result returned."""
    result = await Chain(async_raise_fn).except_(lambda rv, e: 'caught').run()
    self.assertEqual(result, 'caught')

  async def test_async_handler_with_raise_true(self):
    """Async handler + raise_=True: handler awaited, then exception re-raises."""
    called = []
    async def handler(rv, exc):
      called.append(('async_handler', type(exc).__name__))
      return 'async_discarded'
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(called, [('async_handler', 'ValueError')])

  async def test_async_handler_side_effects_complete_before_reraise(self):
    """Async handler side-effects complete before re-raise (result is awaited)."""
    tracker = []
    async def handler(rv, exc):
      tracker.append('step1')
      await asyncio.sleep(0)
      tracker.append('step2')
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, raise_=True).run()
    # Both steps must have completed since the handler is awaited before re-raise.
    self.assertEqual(tracker, ['step1', 'step2'])

  async def test_async_handler_return_value_discarded(self):
    """Async path + raise_=True: handler return value is discarded."""
    async def handler(rv, exc):
      return 42
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, raise_=True).run()

  async def test_async_exception_instance_preserved(self):
    """Async path: the re-raised exception is the exact same instance."""
    original_exc = ValueError('unique async error 67890')
    async def raiser(x=None):
      raise original_exc
    caught_in_handler = []
    def handler(rv, exc):
      caught_in_handler.append(exc)
      return 'discarded'
    with self.assertRaises(ValueError) as cm:
      await Chain(raiser).except_(handler, raise_=True).run()
    self.assertIs(cm.exception, original_exc)
    self.assertIs(caught_in_handler[0], original_exc)

  async def test_async_with_root_value(self):
    """Async path + root value: handler receives (root_value, exc)."""
    received = []
    async def handler(rv, exc):
      received.append((rv, exc))
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain(42).then(async_raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(len(received), 1)
    self.assertEqual(received[0][0], 42)
    self.assertIsInstance(received[0][1], ValueError)

  async def test_async_without_root_value(self):
    """Async path + no root value: handler receives (None, exc)."""
    received = []
    async def handler(rv, exc):
      received.append((rv, exc))
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain().then(async_raise_fn).except_(handler, raise_=True).run()
    self.assertEqual(len(received), 1)
    self.assertIsNone(received[0][0])
    self.assertIsInstance(received[0][1], ValueError)

  async def test_async_exception_type_filtering_matching(self):
    """Async path: matching exception type + raise_=True."""
    called = []
    async def handler(rv, exc):
      called.append(exc)
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, exceptions=ValueError, raise_=True).run()
    self.assertEqual(len(called), 1)

  async def test_async_exception_type_filtering_non_matching(self):
    """Async path: non-matching exception type — handler NOT called."""
    called = []
    async def handler(rv, exc):
      called.append(exc)
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, exceptions=TypeError, raise_=True).run()
    self.assertEqual(len(called), 0)

  async def test_async_interaction_with_finally(self):
    """Async path: raise_=True + finally_() — finally still runs after re-raise."""
    finally_tracker = []
    handler_tracker = []
    async def handler(rv, exc):
      handler_tracker.append('handler')
      return 'discarded'
    async def cleanup(rv):
      finally_tracker.append('finally')
    with self.assertRaises(ValueError):
      await Chain(42).then(async_raise_fn).except_(handler, raise_=True).finally_(cleanup).run()
    self.assertEqual(handler_tracker, ['handler'])
    self.assertEqual(finally_tracker, ['finally'])

  async def test_async_handler_with_explicit_args_and_raise_true(self):
    """Async path: except_(handler, 'a', 'b', raise_=True) — handler called with args."""
    received = []
    async def handler(a, b):
      received.append((a, b))
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, 'a', 'b', raise_=True).run()
    self.assertEqual(received, [('a', 'b')])

  async def test_async_handler_with_ellipsis_and_raise_true(self):
    """Async path: except_(handler, ..., raise_=True) — handler called with no args."""
    received = []
    async def handler():
      received.append('called_no_args')
      return 'discarded'
    with self.assertRaises(ValueError):
      await Chain(async_raise_fn).except_(handler, ..., raise_=True).run()
    self.assertEqual(received, ['called_no_args'])

  async def test_async_sync_transition_raise_true(self):
    """Chain starts sync, transitions to async, then raise_=True works in async path."""
    called = []
    def handler(rv, exc):
      called.append(type(exc).__name__)
      return 'discarded'
    with self.assertRaises(ValueError):
      await (
        Chain(10)
        .then(async_fn)          # 11, triggers async transition
        .then(async_raise_fn)    # raises
        .except_(handler, raise_=True)
        .run()
      )
    self.assertEqual(called, ['ValueError'])

  async def test_async_finally_receives_root_value_with_raise_true(self):
    """Async path: raise_=True + finally_() — finally receives root value."""
    finally_received = []
    async def handler(rv, exc):
      return 'discarded'
    async def cleanup(rv):
      finally_received.append(rv)
    with self.assertRaises(ValueError):
      await Chain(42).then(async_raise_fn).except_(handler, raise_=True).finally_(cleanup).run()
    self.assertEqual(finally_received, [42])


if __name__ == '__main__':
  unittest.main()
