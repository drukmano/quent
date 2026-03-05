"""Tests for Chain.except_(): registration, execution, async paths, and edge cases."""
from __future__ import annotations

import asyncio
import unittest
import warnings

from quent import Chain, Null, QuentException
from quent._core import _Return, _Break
from helpers import raise_fn, async_raise_fn, sync_fn, async_fn, sync_identity, async_identity


# -- Local helpers --

def _catch_handler(exc):
  return f'caught:{type(exc).__name__}'


async def _async_catch_handler(exc):
  return f'async_caught:{type(exc).__name__}'


def _reraise_handler(exc):
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
    def handler(exc):
      called.append(exc)
      return 'handled'
    result = Chain(raise_fn).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(called), 1)
    self.assertIsInstance(called[0], ValueError)

  def test_handler_return_becomes_result(self):
    result = Chain(raise_fn).except_(lambda e: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_handler_raises_new_exception(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain(raise_fn).except_(_reraise_handler).run()
    self.assertEqual(str(cm.exception), 'handler error')
    # The original ValueError should be chained as __cause__
    self.assertIsInstance(cm.exception.__cause__, ValueError)

  def test_no_exception_handler_not_called(self):
    called = []
    def handler(exc):
      called.append(exc)
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
    def handler(exc):
      Chain.return_('val')
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(handler).run()
    self.assertIn('control flow', str(cm.exception).lower())

  def test_break_in_handler_raises_quent_exception(self):
    def handler(exc):
      Chain.break_('val')
    with self.assertRaises(QuentException) as cm:
      Chain(raise_fn).except_(handler).run()
    self.assertIn('control flow', str(cm.exception).lower())

  def test_handler_receives_exception_as_arg(self):
    received = []
    def handler(exc):
      received.append(exc)
      return 'ok'
    Chain(raise_fn).except_(handler).run()
    self.assertEqual(len(received), 1)
    exc = received[0]
    self.assertIsInstance(exc, ValueError)
    self.assertEqual(str(exc), 'test error')

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
      .except_(lambda e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_exception_in_middle_step_caught(self):
    result = (
      Chain(1)
      .then(sync_fn)       # 2
      .then(raise_fn)      # raises
      .then(sync_fn)       # never reached
      .except_(lambda e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_handler_returning_none(self):
    result = Chain(raise_fn).except_(lambda e: None).run()
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
    async def handler(exc):
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
      .except_(lambda e: 'sync_caught')
      .run()
    )
    self.assertEqual(result, 'sync_caught')

  async def test_exception_in_async_step_caught(self):
    result = await (
      Chain(10)
      .then(async_fn)          # 11
      .then(async_raise_fn)    # raises
      .except_(lambda e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_async_handler_raises_propagates(self):
    async def handler(exc):
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
    async def handler(exc):
      called.append(exc)
    result = await Chain(10).then(async_fn).except_(handler).run()
    self.assertEqual(result, 11)
    self.assertEqual(len(called), 0)

  async def test_async_handler_receives_exception_object(self):
    received = []
    async def handler(exc):
      received.append(exc)
      return 'ok'
    await Chain(async_raise_fn).except_(handler).run()
    self.assertEqual(len(received), 1)
    self.assertIsInstance(received[0], ValueError)
    self.assertEqual(str(received[0]), 'test error')

  async def test_control_flow_in_async_handler_raises_internal_signal(self):
    # When an async except handler raises a _Return, it escapes _run_async's
    # except BaseException block (since the _Return is raised during the await
    # of the handler result, inside the except BaseException clause). The
    # _Return propagates out of _run_async as a raw _ControlFlowSignal.
    # In the sync path, run() would catch it and wrap it in QuentException,
    # but here the coroutine is already returned before the _Return is raised,
    # so run()'s try/except never sees it. The raw _Return escapes.
    async def handler(exc):
      Chain.return_('val')
    with self.assertRaises(_Return):
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


if __name__ == '__main__':
  unittest.main()
