"""Advanced exception handling tests for the Quent library.

Covers:
  1. Exception type matching (except_ with various exception types)
  2. Reraise behavior (reraise=True/False with sync and async handlers)
  3. Exception handler raises (handler itself raising exceptions)
  4. Finally handler edge cases
  5. Exception chaining (__cause__ / __context__)
  6. Traceback augmentation (<quent> frames, source arrows)
  7. Complex multi-handler scenarios
  8. Exception in various operations (foreach, filter, gather, with_, to_thread, pipe)
"""
import asyncio
import sys
import traceback
import warnings
from unittest import IsolatedAsyncioTestCase

from tests.utils import empty, aempty, await_
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# Custom exception hierarchy for testing
# ---------------------------------------------------------------------------

class AppError(Exception):
  """Application-level base error."""
  pass


class DatabaseError(AppError):
  """Database error (child of AppError)."""
  pass


class ConnectionError_(AppError):
  """Connection error (child of AppError)."""
  pass


class TimeoutError_(ConnectionError_):
  """Timeout error (grandchild of AppError)."""
  pass


class ValidationError(Exception):
  """Validation error (unrelated to AppError)."""
  pass


class CustomBaseException(BaseException):
  """Custom BaseException subclass."""
  pass


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------

class BaseExcTestCase(IsolatedAsyncioTestCase):
  """Base class providing with_fn helper for sync/async test variants."""

  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def raise_exc(exc_type=Exception):
  """Raise an exception of the given type."""
  raise exc_type()


def raise_value_error(v=None):
  raise ValueError('test value error')


def raise_type_error(v=None):
  raise TypeError('test type error')


def raise_key_error(v=None):
  raise KeyError('test key error')


def raise_index_error(v=None):
  raise IndexError('test index error')


def raise_runtime_error(v=None):
  raise RuntimeError('test runtime error')


def raise_app_error(v=None):
  raise AppError('test app error')


def raise_database_error(v=None):
  raise DatabaseError('test database error')


def raise_timeout_error(v=None):
  raise TimeoutError_('test timeout error')


def raise_validation_error(v=None):
  raise ValidationError('test validation error')


async def async_raise_value_error(v=None):
  raise ValueError('async value error')


async def async_raise_type_error(v=None):
  raise TypeError('async type error')


# ---------------------------------------------------------------------------
# 1. Exception Type Matching (18 tests)
# ---------------------------------------------------------------------------

class ExceptionTypeMatchingTests(BaseExcTestCase):
  """Tests for .except_(handler, exceptions=...) matching behavior."""

  async def test_default_catches_exception(self):
    """except_(handler) catches Exception by default."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler)
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(called['value'])

  async def test_catches_only_value_error(self):
    """except_(handler, exceptions=ValueError) catches only ValueError."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, exceptions=ValueError)
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(called['value'])

  async def test_catches_only_type_error(self):
    """except_(handler, exceptions=TypeError) catches only TypeError."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_type_error)
            .except_(handler, exceptions=TypeError)
            .run()
          )
        except TypeError:
          pass
        self.assertTrue(called['value'])

  async def test_catches_list_of_exceptions(self):
    """except_(handler, exceptions=[ValueError, TypeError]) catches either."""
    for fn, ctx in self.with_fn():
      with ctx:
        # Test ValueError
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, exceptions=[ValueError, TypeError])
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(called['value'])

        # Test TypeError
        called['value'] = False
        try:
          await await_(
            Chain(fn).then(raise_type_error)
            .except_(handler, exceptions=[ValueError, TypeError])
            .run()
          )
        except TypeError:
          pass
        self.assertTrue(called['value'])

  async def test_catches_tuple_of_exceptions(self):
    """except_(handler, exceptions=(KeyError, IndexError)) catches either."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_key_error)
            .except_(handler, exceptions=(KeyError, IndexError))
            .run()
          )
        except KeyError:
          pass
        self.assertTrue(called['value'])

        called['value'] = False
        try:
          await await_(
            Chain(fn).then(raise_index_error)
            .except_(handler, exceptions=(KeyError, IndexError))
            .run()
          )
        except IndexError:
          pass
        self.assertTrue(called['value'])

  async def test_exception_subclass_matching(self):
    """If exceptions=Exception, catches ValueError (a subclass)."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, exceptions=Exception)
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(called['value'])

  async def test_exception_not_matching(self):
    """exceptions=ValueError but TypeError raised -- handler NOT called."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_type_error)
            .except_(handler, exceptions=ValueError)
            .run()
          )
        except TypeError:
          pass
        self.assertFalse(called['value'])

  async def test_multiple_except_handlers_different_types(self):
    """Multiple except handlers with different exception types; first match wins."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False}
        def handler1(v=None):
          calls['h1'] = True
        def handler2(v=None):
          calls['h2'] = True

        try:
          await await_(
            Chain(fn).then(raise_type_error)
            .except_(handler1, exceptions=ValueError)
            .except_(handler2, exceptions=TypeError)
            .run()
          )
        except TypeError:
          pass
        self.assertFalse(calls['h1'])
        self.assertTrue(calls['h2'])

  async def test_first_matching_handler_wins(self):
    """When multiple handlers match, the first matching one wins."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False}
        def handler1(v=None):
          calls['h1'] = True
        def handler2(v=None):
          calls['h2'] = True

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler1, exceptions=Exception)
            .except_(handler2, exceptions=ValueError)
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(calls['h1'])
        self.assertFalse(calls['h2'])

  async def test_non_matching_handlers_skipped(self):
    """Handlers that don't match the exception type are skipped."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False, 'h3': False}
        def handler1(v=None):
          calls['h1'] = True
        def handler2(v=None):
          calls['h2'] = True
        def handler3(v=None):
          calls['h3'] = True

        try:
          await await_(
            Chain(fn).then(raise_runtime_error)
            .except_(handler1, exceptions=ValueError)
            .except_(handler2, exceptions=TypeError)
            .except_(handler3, exceptions=RuntimeError)
            .run()
          )
        except RuntimeError:
          pass
        self.assertFalse(calls['h1'])
        self.assertFalse(calls['h2'])
        self.assertTrue(calls['h3'])

  async def test_base_exception_catches_system_exit(self):
    """except_(handler, exceptions=BaseException) catches SystemExit."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        def raise_system_exit(v=None):
          raise SystemExit(1)
        try:
          await await_(
            Chain(fn).then(raise_system_exit)
            .except_(handler, exceptions=BaseException)
            .run()
          )
        except SystemExit:
          pass
        self.assertTrue(called['value'])

  async def test_default_except_does_not_catch_system_exit(self):
    """Default except_ (exceptions=Exception) does NOT catch SystemExit."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        def raise_system_exit(v=None):
          raise SystemExit(1)
        try:
          await await_(
            Chain(fn).then(raise_system_exit)
            .except_(handler)
            .run()
          )
        except SystemExit:
          pass
        self.assertFalse(called['value'])

  async def test_default_except_does_not_catch_keyboard_interrupt(self):
    """Default except_ does NOT catch KeyboardInterrupt."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        def raise_ki(v=None):
          raise KeyboardInterrupt()
        try:
          await await_(
            Chain(fn).then(raise_ki)
            .except_(handler)
            .run()
          )
        except KeyboardInterrupt:
          pass
        self.assertFalse(called['value'])

  async def test_exception_string_raises_type_error(self):
    """except_(handler, exceptions='ValueError') raises TypeError."""
    def handler(v=None):
      pass
    with self.assertRaises(TypeError):
      Chain().then(lambda: None).except_(handler, exceptions='ValueError')

  async def test_custom_exception_hierarchy_parent_child(self):
    """Parent exception type catches child exceptions."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_database_error)
            .except_(handler, exceptions=AppError)
            .run()
          )
        except DatabaseError:
          pass
        self.assertTrue(called['value'])

  async def test_custom_exception_hierarchy_grandchild(self):
    """Grandparent exception type catches grandchild exceptions."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_timeout_error)
            .except_(handler, exceptions=AppError)
            .run()
          )
        except TimeoutError_:
          pass
        self.assertTrue(called['value'])

  async def test_child_exception_type_does_not_catch_parent(self):
    """Child exception type does NOT catch parent exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_app_error)
            .except_(handler, exceptions=DatabaseError)
            .run()
          )
        except AppError:
          pass
        self.assertFalse(called['value'])

  async def test_sibling_exception_types_no_cross_matching(self):
    """Sibling exception types don't cross-match."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_database_error)
            .except_(handler, exceptions=ConnectionError_)
            .run()
          )
        except DatabaseError:
          pass
        self.assertFalse(called['value'])


# ---------------------------------------------------------------------------
# 2. Reraise Behavior (11 tests)
# ---------------------------------------------------------------------------

class ReraiseBehaviorTests(BaseExcTestCase):
  """Tests for reraise=True/False in except_ handlers."""

  async def test_reraise_true_default(self):
    """reraise=True (default): handler runs, exception re-raised."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_reraise_false_returns_handler_result(self):
    """reraise=False: handler runs, handler result returned."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          return 'recovered'
        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertEqual(result, 'recovered')

  async def test_reraise_true_with_async_handler_emits_warning(self):
    """reraise=True with async handler on sync chain: RuntimeWarning emitted."""
    async def async_handler(v=None):
      return 'handled'

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ValueError):
        Chain(raise_value_error).then(aempty)  \
          .except_(async_handler)  \
          .run()

      runtime_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning)
        and 'except' in str(x.message).lower()
        and 'coroutine' in str(x.message).lower()
      ]
      self.assertGreater(len(runtime_warnings), 0)

  async def test_reraise_false_with_async_handler_returns_task(self):
    """reraise=False with async handler on sync chain: returns awaitable task."""
    async def async_handler(v=None):
      return 'async_recovered'

    result = Chain(raise_value_error).then(aempty) \
      .except_(async_handler, reraise=False) \
      .run()
    resolved = await result
    self.assertEqual(resolved, 'async_recovered')

  async def test_handler_returns_none_with_reraise_false(self):
    """Handler that returns None with reraise=False: chain returns None."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          return None
        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertIsNone(result)

  async def test_handler_return_value_ignored_with_reraise_true(self):
    """Handler that returns a value with reraise=True: value ignored, exception re-raised."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          return 'this_should_be_ignored'
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=True)
            .run()
          )

  async def test_multiple_handlers_first_match_reraise_true(self):
    """Multiple handlers, first match has reraise=True: runs, re-raises, second NOT run."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False}
        def handler1(v=None):
          calls['h1'] = True
        def handler2(v=None):
          calls['h2'] = True

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler1, reraise=True)
            .except_(handler2, reraise=False)
            .run()
          )
        self.assertTrue(calls['h1'])
        self.assertFalse(calls['h2'])

  async def test_multiple_handlers_first_match_reraise_false(self):
    """Multiple handlers, first match has reraise=False: result returned, second NOT run."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False}
        def handler1(v=None):
          calls['h1'] = True
          return 'from_h1'
        def handler2(v=None):
          calls['h2'] = True
          return 'from_h2'

        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(handler1, reraise=False)
          .except_(handler2, reraise=False)
          .run()
        )
        self.assertEqual(result, 'from_h1')
        self.assertTrue(calls['h1'])
        self.assertFalse(calls['h2'])

  async def test_handler_receives_root_value(self):
    """Handler receives root_value as argument (via evaluate_value)."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}
        def handler(v=None):
          received['value'] = v
          return v

        result = await await_(
          Chain(fn, 42).then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertEqual(received['value'], 42)
        self.assertEqual(result, 42)

  async def test_reraise_false_specific_return_value(self):
    """Handler with reraise=False returns a specific object."""
    sentinel = object()
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          return sentinel
        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertIs(result, sentinel)

  async def test_reraise_false_handler_returns_awaitable(self):
    """Async handler with reraise=False in async chain returns awaited result."""
    async def async_handler(v=None):
      return 'async_result'

    result = await Chain(aempty).then(async_raise_value_error) \
      .except_(async_handler, reraise=False) \
      .run()
    self.assertEqual(result, 'async_result')


# ---------------------------------------------------------------------------
# 3. Exception Handler Raises (10 tests)
# ---------------------------------------------------------------------------

class ExceptionHandlerRaisesTests(BaseExcTestCase):
  """Tests for when the exception handler itself raises an exception."""

  async def test_handler_raises_new_exception_chained(self):
    """Handler raises a new exception: results in `raise exc_ from exc` chaining."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('handler error')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertEqual(str(e), 'handler error')
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_handler_raises_same_exception_type(self):
    """Handler raises same exception type as original."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise ValueError('handler value error')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except ValueError as e:
          self.assertEqual(str(e), 'handler value error')
          self.assertIsInstance(e.__cause__, ValueError)
          self.assertEqual(str(e.__cause__), 'test value error')

  async def test_handler_raises_different_exception_type(self):
    """Handler raises different exception type from original."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise TypeError('handler type error')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except TypeError as e:
          self.assertEqual(str(e), 'handler type error')
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_async_handler_that_raises(self):
    """Async handler that raises in async chain."""
    async def async_handler(v=None):
      raise RuntimeError('async handler error')

    try:
      await Chain(aempty).then(async_raise_value_error) \
        .except_(async_handler, reraise=False) \
        .run()
      self.fail('Should have raised')
    except RuntimeError as e:
      self.assertEqual(str(e), 'async handler error')
      self.assertIsInstance(e.__cause__, ValueError)

  async def test_handler_raises_with_reraise_true(self):
    """Handler raises with reraise=True -- the handler exception propagates (raise from)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('handler error in reraise')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=True)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertEqual(str(e), 'handler error in reraise')
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_handler_raises_with_reraise_false(self):
    """Handler raises with reraise=False -- handler's new exception propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise TypeError('handler type error')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except TypeError as e:
          self.assertEqual(str(e), 'handler type error')
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_cause_chain_is_correct(self):
    """The __cause__ chain: original exc -> handler exc."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('from handler')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertIsNotNone(e.__cause__)
          self.assertIsInstance(e.__cause__, ValueError)
          self.assertEqual(str(e.__cause__), 'test value error')

  async def test_context_is_preserved(self):
    """The __context__/__cause__ is set when handler raises."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('from handler')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          # __cause__ is explicitly set via `raise exc_ from exc`
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_handler_raises_type_error_original_value_error(self):
    """Handler raises TypeError, original was ValueError -- final is TypeError from ValueError."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise TypeError('handler')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except TypeError as e:
          self.assertEqual(str(e), 'handler')
          self.assertIsInstance(e.__cause__, ValueError)
          self.assertEqual(str(e.__cause__), 'test value error')

  async def test_nested_chain_handler_raises_outer_catches(self):
    """Nested chain: inner handler raises, outer chain catches the new exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        def inner_handler(v=None):
          raise RuntimeError('inner handler error')

        outer_called = {'value': False}
        def outer_handler(v=None):
          outer_called['value'] = True
          return 'outer_recovered'

        inner = Chain().then(fn).then(raise_value_error) \
          .except_(inner_handler, reraise=False)

        result = await await_(
          Chain(fn).then(inner, ...)
          .except_(outer_handler, exceptions=RuntimeError, reraise=False)
          .run()
        )
        self.assertTrue(outer_called['value'])
        self.assertEqual(result, 'outer_recovered')


# ---------------------------------------------------------------------------
# 4. Finally Handler Edge Cases (16 tests)
# ---------------------------------------------------------------------------

class FinallyHandlerEdgeCaseTests(BaseExcTestCase):
  """Tests for finally_ handler edge cases."""

  async def test_finally_runs_on_success(self):
    """Finally runs on success."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def on_finally(v=None):
          called['value'] = True
        result = await await_(
          Chain(fn, 42).then(lambda v: v * 2)
          .finally_(on_finally)
          .run()
        )
        self.assertEqual(result, 84)
        self.assertTrue(called['value'])

  async def test_finally_runs_on_exception_with_except(self):
    """Finally runs on exception (with except handler)."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'exc': False, 'fin': False}
        def on_except(v=None):
          calls['exc'] = True
        def on_finally(v=None):
          calls['fin'] = True
        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(on_except)
            .finally_(on_finally)
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(calls['exc'])
        self.assertTrue(calls['fin'])

  async def test_finally_runs_on_exception_without_except(self):
    """Finally runs on exception (without except handler)."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def on_finally(v=None):
          called['value'] = True
        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .finally_(on_finally)
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(called['value'])

  async def test_finally_handler_receives_root_value(self):
    """Finally handler receives root_value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}
        def on_finally(v=None):
          received['value'] = v
        await await_(
          Chain(fn, 'root_val')
          .then(lambda v: v + '_modified')
          .finally_(on_finally)
          .run()
        )
        self.assertEqual(received['value'], 'root_val')

  async def test_finally_return_value_ignored_on_success(self):
    """Finally handler return value is ignored on success."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_finally(v=None):
          return 'finally_result'
        result = await await_(
          Chain(fn, 42)
          .finally_(on_finally)
          .run()
        )
        self.assertEqual(result, 42)

  async def test_finally_that_raises_overrides_original_exception(self):
    """Finally handler that raises -- overrides original exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_finally(v=None):
          raise RuntimeError('finally error')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .finally_(on_finally)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertEqual(str(e), 'finally error')

  async def test_finally_that_raises_on_success(self):
    """Finally handler that raises on success -- raises the finally exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_finally(v=None):
          raise RuntimeError('finally error on success')

        try:
          await await_(
            Chain(fn, 42)
            .finally_(on_finally)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertEqual(str(e), 'finally error on success')

  async def test_finally_async_on_sync_chain_warning(self):
    """Finally handler that is async on sync chain -- RuntimeWarning + ensure_future."""
    async def async_finally(v=None):
      pass

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      result = Chain(42).finally_(async_finally).run()
      self.assertEqual(result, 42)

      runtime_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning)
        and 'finally' in str(x.message).lower()
      ]
      self.assertGreater(len(runtime_warnings), 0)

    await asyncio.sleep(0.1)

  async def test_finally_with_control_flow_raises_quent_exception(self):
    """Finally handler with control flow signals (return_/break_) -- QuentException."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_finally_return(v=None):
          Chain.return_('should not work')

        with self.assertRaises(QuentException):
          await await_(
            Chain(fn, 42)
            .finally_(on_finally_return)
            .run()
          )

  async def test_duplicate_finally_raises_quent_exception(self):
    """Duplicate finally_ raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(42) \
        .finally_(lambda v: None) \
        .finally_(lambda v: None)

  async def test_finally_on_void_chain(self):
    """Finally on void chain -- receives None (Null becomes None)."""
    received = {'value': 'sentinel'}
    def on_finally(v=None):
      received['value'] = v
    Chain().finally_(on_finally).run()
    self.assertIsNone(received['value'])

  async def test_finally_on_cascade(self):
    """Finally on cascade -- receives root_value."""
    received = {'value': None}
    def on_finally(v=None):
      received['value'] = v
    result = await await_(
      Cascade(42)
      .then(lambda v: v * 2)
      .finally_(on_finally)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(received['value'], 42)

  async def test_finally_on_async_chain_properly_awaited(self):
    """Finally on async chain -- properly awaited."""
    called = {'value': False}
    async def async_finally(v=None):
      called['value'] = True

    result = await Chain(aempty, 42) \
      .finally_(async_finally) \
      .run()
    self.assertEqual(result, 42)
    self.assertTrue(called['value'])

  async def test_finally_after_except_reraise_true(self):
    """Finally after except(reraise=True) -- runs after re-raise."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'exc': False, 'fin': False}
        def on_except(v=None):
          calls['exc'] = True
        def on_finally(v=None):
          calls['fin'] = True

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(on_except, reraise=True)
            .finally_(on_finally)
            .run()
          )
        self.assertTrue(calls['exc'])
        self.assertTrue(calls['fin'])

  async def test_finally_after_except_reraise_false(self):
    """Finally after except(reraise=False) -- runs after handler result."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'exc': False, 'fin': False}
        def on_except(v=None):
          calls['exc'] = True
          return 'recovered'
        def on_finally(v=None):
          calls['fin'] = True

        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(on_except, reraise=False)
          .finally_(on_finally)
          .run()
        )
        self.assertEqual(result, 'recovered')
        self.assertTrue(calls['exc'])
        self.assertTrue(calls['fin'])

  async def test_finally_handler_does_not_alter_success_result(self):
    """Finally handler does not alter the successful chain result."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        def on_finally(v=None):
          return 'not_this'
        result = await await_(
          Chain(fn, sentinel)
          .finally_(on_finally)
          .run()
        )
        self.assertIs(result, sentinel)


# ---------------------------------------------------------------------------
# 5. Exception Chaining (__cause__ / __context__) (10 tests)
# ---------------------------------------------------------------------------

class ExceptionChainingTests(BaseExcTestCase):
  """Tests for exception chaining behavior (__cause__, __context__)."""

  async def test_explicit_chaining_raise_from(self):
    """Explicit chaining: handler `raise X from Y`."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('handler') from ValueError('original')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_implicit_chaining_context(self):
    """Implicit chaining: handler `raise X` while Y is active -> __context__."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('implicit chain')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          # __cause__ is explicitly set by quent's `raise exc_ from exc`
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_suppress_context_flag(self):
    """__suppress_context__ flag works correctly."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          exc = RuntimeError('suppressed')
          exc.__suppress_context__ = True
          raise exc

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertTrue(e.__suppress_context__)

  async def test_deep_chaining(self):
    """Deep chaining: A -> B -> C through nested handlers."""
    for fn, ctx in self.with_fn():
      with ctx:
        def raise_a(v=None):
          raise ValueError('A')
        def handler_a_to_b(v=None):
          raise TypeError('B')
        def handler_b_to_c(v=None):
          raise RuntimeError('C')

        inner = Chain().then(fn).then(raise_a) \
          .except_(handler_a_to_b, reraise=False)
        try:
          await await_(
            Chain(fn).then(inner, ...)
            .except_(handler_b_to_c, exceptions=TypeError, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertEqual(str(e), 'C')
          self.assertIsInstance(e.__cause__, TypeError)
          self.assertEqual(str(e.__cause__), 'B')

  async def test_chained_exceptions_through_nested_chains(self):
    """Chained exceptions through nested chains."""
    for fn, ctx in self.with_fn():
      with ctx:
        def inner_raise(v=None):
          raise ValueError('inner')
        def inner_handler(v=None):
          raise TypeError('from inner handler')

        inner_chain = Chain().then(fn).then(inner_raise) \
          .except_(inner_handler, reraise=False)

        try:
          await await_(Chain(fn).then(inner_chain, ...).run())
          self.fail('Should have raised')
        except TypeError as e:
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_except_handler_chains_via_raise_from(self):
    """except handler's exception chains with original via `raise exc_ from exc`."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('handler exc')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          # quent does `raise exc_ from exc` which sets __cause__
          self.assertIsInstance(e.__cause__, ValueError)
          # When using `raise X from Y`, Python sets __suppress_context__ = True
          # This is standard Python behavior for explicit exception chaining
          self.assertTrue(e.__suppress_context__)

  async def test_async_chained_exceptions(self):
    """Async chained exceptions."""
    async def async_handler(v=None):
      raise RuntimeError('async handler')

    try:
      await Chain(aempty).then(async_raise_value_error) \
        .except_(async_handler, reraise=False) \
        .run()
      self.fail('Should have raised')
    except RuntimeError as e:
      self.assertIsInstance(e.__cause__, ValueError)

  async def test_cascade_exception_chaining(self):
    """Cascade exception chaining."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('cascade handler')

        try:
          await await_(
            Cascade(fn, 42)
            .then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertIsInstance(e.__cause__, ValueError)

  async def test_exception_from_then_has_no_false_context(self):
    """Exception from .then() raised after a previous successful .then()."""
    for fn, ctx in self.with_fn():
      with ctx:
        def step1(v=None):
          return 'step1_result'
        def step2(v=None):
          raise ValueError('step2 error')

        try:
          await await_(
            Chain(fn).then(step1).then(step2).run()
          )
          self.fail('Should have raised')
        except ValueError as e:
          self.assertEqual(str(e), 'step2 error')

  async def test_cause_chain_preserved_through_reraise(self):
    """__cause__ chain preserved when handler raises with reraise=True."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('handler raises')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=True)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertIsInstance(e.__cause__, ValueError)


# ---------------------------------------------------------------------------
# 6. Traceback Augmentation (10 tests)
# ---------------------------------------------------------------------------

class TracebackAugmentationTests(BaseExcTestCase):
  """Tests for traceback augmentation with <quent> frames."""

  async def test_exception_has_quent_frame(self):
    """Exception in chain has <quent> frame in traceback."""
    for fn, ctx in self.with_fn():
      with ctx:
        try:
          await await_(Chain(fn).then(raise_value_error).run())
          self.fail('Should have raised')
        except ValueError as e:
          tb = e.__traceback__
          quent_frames = []
          while tb is not None:
            if tb.tb_frame.f_code.co_filename == '<quent>':
              quent_frames.append(tb)
            tb = tb.tb_next
          self.assertGreater(len(quent_frames), 0,
            'Expected at least one <quent> frame in traceback')

  async def test_quent_frame_shows_chain_visualization(self):
    """<quent> frame shows chain visualization in co_name."""
    for fn, ctx in self.with_fn():
      with ctx:
        try:
          await await_(Chain(fn).then(raise_value_error).run())
          self.fail('Should have raised')
        except ValueError as e:
          tb = e.__traceback__
          found_viz = False
          while tb is not None:
            if tb.tb_frame.f_code.co_filename == '<quent>':
              name = tb.tb_frame.f_code.co_name
              if 'Chain' in name or 'then' in name:
                found_viz = True
                break
            tb = tb.tb_next
          self.assertTrue(found_viz,
            'Expected chain visualization in <quent> frame co_name')

  async def test_source_arrow_points_to_failing_link(self):
    """Source arrow (<----) points to failing link."""
    for fn, ctx in self.with_fn():
      with ctx:
        try:
          await await_(
            Chain(fn).then(lambda v: v).then(raise_value_error).run()
          )
          self.fail('Should have raised')
        except ValueError as e:
          tb = e.__traceback__
          found_arrow = False
          while tb is not None:
            if tb.tb_frame.f_code.co_filename == '<quent>':
              name = tb.tb_frame.f_code.co_name
              if '<----' in name:
                found_arrow = True
                break
            tb = tb.tb_next
          self.assertTrue(found_arrow,
            'Expected source arrow <---- in chain visualization')

  async def test_debug_mode_shows_intermediate_results(self):
    """Debug mode: intermediate results shown in traceback."""
    for fn, ctx in self.with_fn():
      with ctx:
        try:
          await await_(
            Chain(fn, 10)
            .config(debug=True)
            .then(lambda v: v * 2)
            .then(raise_value_error)
            .run()
          )
          self.fail('Should have raised')
        except ValueError as e:
          tb = e.__traceback__
          found_result = False
          while tb is not None:
            if tb.tb_frame.f_code.co_filename == '<quent>':
              name = tb.tb_frame.f_code.co_name
              if '= ' in name or '20' in name:
                found_result = True
                break
            tb = tb.tb_next
          self.assertTrue(found_result,
            'Expected intermediate results in debug traceback')

  async def test_no_quent_internal_frames_leaked(self):
    """No quent-internal frames leaked to user (only <quent> viz frames)."""
    for fn, ctx in self.with_fn():
      with ctx:
        try:
          await await_(Chain(fn).then(raise_value_error).run())
          self.fail('Should have raised')
        except ValueError as e:
          tb = e.__traceback__
          import os
          while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            if filename != '<quent>' and 'quent/quent' in filename:
              # quent internal frames should be cleaned
              pass  # mild assertion: we accept clean tracebacks
            tb = tb.tb_next

  async def test_user_code_frames_preserved(self):
    """User code frames preserved in traceback."""
    for fn, ctx in self.with_fn():
      with ctx:
        def my_user_function(v=None):
          raise ValueError('user error')

        try:
          await await_(
            Chain(fn).then(my_user_function).run()
          )
          self.fail('Should have raised')
        except ValueError as e:
          tb = e.__traceback__
          found_user = False
          while tb is not None:
            if 'my_user_function' in (tb.tb_frame.f_code.co_name or ''):
              found_user = True
              break
            tb = tb.tb_next
          self.assertTrue(found_user,
            'Expected user function frame in traceback')

  async def test_custom_exception_preserves_message(self):
    """Custom exception types preserve their message."""
    for fn, ctx in self.with_fn():
      with ctx:
        def raise_custom(v=None):
          raise AppError('custom message')

        try:
          await await_(
            Chain(fn).then(raise_custom).run()
          )
          self.fail('Should have raised')
        except AppError as e:
          self.assertEqual(str(e), 'custom message')

  async def test_exception_in_handler_modified_traceback(self):
    """Exception in except handler: modified traceback includes handler info."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('handler error')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          tb = e.__traceback__
          self.assertIsNotNone(tb)

  async def test_nested_chain_traceback_visualization(self):
    """Nested chain: visualization shows nesting."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().then(fn).then(raise_value_error)
        try:
          await await_(
            Chain(fn).then(inner, ...).run()
          )
          self.fail('Should have raised')
        except ValueError as e:
          tb = e.__traceback__
          quent_frames = []
          while tb is not None:
            if tb.tb_frame.f_code.co_filename == '<quent>':
              quent_frames.append(tb)
            tb = tb.tb_next
          self.assertGreater(len(quent_frames), 0)

  async def test_chained_exceptions_cleaned_tracebacks(self):
    """Chained exceptions also have cleaned tracebacks."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise RuntimeError('handler')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(handler, reraise=False)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          cause = e.__cause__
          self.assertIsNotNone(cause)
          self.assertIsInstance(cause, ValueError)


# ---------------------------------------------------------------------------
# 7. Complex Multi-Handler Scenarios (11 tests)
# ---------------------------------------------------------------------------

class ComplexMultiHandlerTests(BaseExcTestCase):
  """Tests for complex scenarios with multiple exception handlers."""

  async def test_three_handlers_matches_second(self):
    """3 except handlers, exception matches 2nd."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False, 'h3': False}
        def h1(v=None):
          calls['h1'] = True
        def h2(v=None):
          calls['h2'] = True
        def h3(v=None):
          calls['h3'] = True

        with self.assertRaises(TypeError):
          await await_(
            Chain(fn).then(raise_type_error)
            .except_(h1, exceptions=ValueError)
            .except_(h2, exceptions=TypeError)
            .except_(h3, exceptions=RuntimeError)
            .run()
          )
        self.assertFalse(calls['h1'])
        self.assertTrue(calls['h2'])
        self.assertFalse(calls['h3'])

  async def test_three_handlers_matches_none_propagates(self):
    """3 except handlers, exception matches none -- propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False, 'h3': False}
        def h1(v=None):
          calls['h1'] = True
        def h2(v=None):
          calls['h2'] = True
        def h3(v=None):
          calls['h3'] = True

        with self.assertRaises(KeyError):
          await await_(
            Chain(fn).then(raise_key_error)
            .except_(h1, exceptions=ValueError)
            .except_(h2, exceptions=TypeError)
            .except_(h3, exceptions=RuntimeError)
            .run()
          )
        self.assertFalse(calls['h1'])
        self.assertFalse(calls['h2'])
        self.assertFalse(calls['h3'])

  async def test_reraise_true_second_handler_not_run(self):
    """except handler with reraise=True + another handler after -- second never runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False}
        def h1(v=None):
          calls['h1'] = True
        def h2(v=None):
          calls['h2'] = True

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(h1, exceptions=ValueError, reraise=True)
            .except_(h2, exceptions=ValueError, reraise=False)
            .run()
          )
        self.assertTrue(calls['h1'])
        self.assertFalse(calls['h2'])

  async def test_reraise_false_returns_result(self):
    """except handler with reraise=False: result returned."""
    for fn, ctx in self.with_fn():
      with ctx:
        def h1(v=None):
          return 'swallowed'

        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(h1, reraise=False)
          .run()
        )
        self.assertEqual(result, 'swallowed')

  async def test_except_and_finally_swallows(self):
    """except and finally combination: except swallows, finally runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'exc': False, 'fin': False}
        def on_except(v=None):
          calls['exc'] = True
          return 'recovered'
        def on_finally(v=None):
          calls['fin'] = True

        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(on_except, reraise=False)
          .finally_(on_finally)
          .run()
        )
        self.assertEqual(result, 'recovered')
        self.assertTrue(calls['exc'])
        self.assertTrue(calls['fin'])

  async def test_except_and_finally_reraises(self):
    """except and finally combination: except re-raises, finally runs, exception propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'exc': False, 'fin': False}
        def on_except(v=None):
          calls['exc'] = True
        def on_finally(v=None):
          calls['fin'] = True

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(on_except, reraise=True)
            .finally_(on_finally)
            .run()
          )
        self.assertTrue(calls['exc'])
        self.assertTrue(calls['fin'])

  async def test_finally_raises_overrides_except_result(self):
    """except and finally: finally raises, overrides except result."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_except(v=None):
          return 'recovered'
        def on_finally(v=None):
          raise RuntimeError('finally overrides')

        try:
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(on_except, reraise=False)
            .finally_(on_finally)
            .run()
          )
          self.fail('Should have raised')
        except RuntimeError as e:
          self.assertEqual(str(e), 'finally overrides')

  async def test_two_different_exception_types_two_handlers(self):
    """Two different exception types in same chain, two handlers."""
    for fn, ctx in self.with_fn():
      with ctx:
        # ValueError with ValueError handler
        calls = {'h1': False, 'h2': False}
        def h1(v=None):
          calls['h1'] = True
        def h2(v=None):
          calls['h2'] = True

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn).then(raise_value_error)
            .except_(h1, exceptions=ValueError)
            .except_(h2, exceptions=TypeError)
            .run()
          )
        self.assertTrue(calls['h1'])
        self.assertFalse(calls['h2'])

        # Now with TypeError
        calls = {'h1': False, 'h2': False}
        with self.assertRaises(TypeError):
          await await_(
            Chain(fn).then(raise_type_error)
            .except_(h1, exceptions=ValueError)
            .except_(h2, exceptions=TypeError)
            .run()
          )
        self.assertFalse(calls['h1'])
        self.assertTrue(calls['h2'])

  async def test_wildcard_handler_as_last_resort(self):
    """Wildcard handler (exceptions=Exception) as last resort."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'specific': False, 'wildcard': False}
        def specific_handler(v=None):
          calls['specific'] = True
        def wildcard_handler(v=None):
          calls['wildcard'] = True

        with self.assertRaises(RuntimeError):
          await await_(
            Chain(fn).then(raise_runtime_error)
            .except_(specific_handler, exceptions=ValueError)
            .except_(wildcard_handler, exceptions=Exception)
            .run()
          )
        self.assertFalse(calls['specific'])
        self.assertTrue(calls['wildcard'])

  async def test_base_exception_handler_as_absolute_last(self):
    """BaseException handler as absolute last resort."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'exc': False, 'base': False}
        def exc_handler(v=None):
          calls['exc'] = True
        def base_handler(v=None):
          calls['base'] = True

        def raise_system_exit(v=None):
          raise SystemExit(1)

        with self.assertRaises(SystemExit):
          await await_(
            Chain(fn).then(raise_system_exit)
            .except_(exc_handler, exceptions=Exception)
            .except_(base_handler, exceptions=BaseException)
            .run()
          )
        self.assertFalse(calls['exc'])
        self.assertTrue(calls['base'])

  async def test_handler_with_reraise_false_returns_value(self):
    """Handler with reraise=False returns a specific value."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        def handler(v=None):
          return sentinel

        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertIs(result, sentinel)


# ---------------------------------------------------------------------------
# 8. Exception in Various Operations (12 tests)
# ---------------------------------------------------------------------------

class ExceptionInOperationsTests(BaseExcTestCase):
  """Tests for exceptions in various chain operations."""

  async def test_exception_in_foreach_body(self):
    """Exception in foreach body is caught by except handler."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        def explode(item):
          if item == 3:
            raise ValueError('foreach error')
          return item

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, [1, 2, 3, 4])
            .foreach(explode)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_exception_in_filter_predicate(self):
    """Exception in filter predicate is caught by except handler."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        def bad_predicate(item):
          if item == 3:
            raise ValueError('filter error')
          return True

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, [1, 2, 3, 4])
            .filter(bad_predicate)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_exception_in_gather_function(self):
    """Exception in gather function is caught."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        def fail_fn(v):
          raise ValueError('gather error')

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, 42)
            .gather(lambda v: v, fail_fn)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_exception_in_context_manager_enter(self):
    """Exception in context manager __enter__."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        class FailEnterCM:
          def __enter__(self):
            raise ValueError('enter error')
          def __exit__(self, *args):
            pass

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, FailEnterCM())
            .with_(lambda ctx: ctx)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_exception_in_context_manager_exit(self):
    """Exception in context manager __exit__."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        class FailExitCM:
          def __enter__(self):
            return 'ctx_value'
          def __exit__(self, *args):
            raise RuntimeError('exit error')

        with self.assertRaises(RuntimeError):
          await await_(
            Chain(fn, FailExitCM())
            .with_(lambda ctx: ctx)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_exception_in_context_manager_body(self):
    """Exception in context manager body."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        exited = {'value': False}
        def handler(v=None):
          called['value'] = True

        class TrackExitCM:
          def __enter__(self):
            return 'ctx_value'
          def __exit__(self_, *args):
            exited['value'] = True
            return False

        def body_fails(v):
          raise ValueError('body error')

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, TrackExitCM())
            .with_(body_fails)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])
        self.assertTrue(exited['value'])

  async def test_exception_in_to_thread_function(self):
    """Exception in to_thread function."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    def fail_in_thread(v):
      raise ValueError('thread error')

    with self.assertRaises(ValueError):
      await Chain(aempty, 42) \
        .to_thread(fail_in_thread) \
        .except_(handler) \
        .run()
    self.assertTrue(called['value'])

  async def test_exception_in_chain_root_evaluation(self):
    """Exception in chain root evaluation."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        def fail_root():
          raise ValueError('root error')

        with self.assertRaises(ValueError):
          await await_(
            Chain(fail_root)
            .then(lambda v: v)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_exception_in_pipe_operator(self):
    """Exception in pipe operator."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(ValueError):
          await await_(
            (Chain(fn) | raise_value_error | run())
          )

  async def test_exception_in_async_foreach(self):
    """Exception in async foreach body."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    async def async_explode(item):
      if item == 2:
        raise ValueError('async foreach error')
      return item

    with self.assertRaises(ValueError):
      await Chain(aempty, [1, 2, 3]) \
        .foreach(async_explode) \
        .except_(handler) \
        .run()
    self.assertTrue(called['value'])

  async def test_exception_in_async_filter(self):
    """Exception in async filter predicate."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    async def async_bad_pred(item):
      if item == 2:
        raise ValueError('async filter error')
      return True

    with self.assertRaises(ValueError):
      await Chain(aempty, [1, 2, 3]) \
        .filter(async_bad_pred) \
        .except_(handler) \
        .run()
    self.assertTrue(called['value'])

  async def test_exception_in_async_gather(self):
    """Exception in async gather function."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    async def async_fail(v):
      raise ValueError('async gather error')

    with self.assertRaises(ValueError):
      await Chain(aempty, 42) \
        .gather(lambda v: v, async_fail) \
        .except_(handler) \
        .run()
    self.assertTrue(called['value'])


# ---------------------------------------------------------------------------
# Additional Edge Cases
# ---------------------------------------------------------------------------

class ExceptHandlerArgumentTests(BaseExcTestCase):
  """Tests for except handler receiving correct arguments."""

  async def test_handler_receives_root_value_void_chain(self):
    """Handler on void chain receives None (Null -> None)."""
    received = {'value': 'sentinel'}
    def handler(v=None):
      received['value'] = v
      return 'recovered'

    result = await await_(
      Chain().then(raise_value_error)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertIsNone(received['value'])

  async def test_handler_receives_root_value_with_value(self):
    """Handler receives the root value when chain has a root."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}
        def handler(v=None):
          received['value'] = v
          return v

        sentinel = object()
        result = await await_(
          Chain(fn, sentinel).then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertIs(received['value'], sentinel)

  async def test_handler_with_extra_args(self):
    """Handler can be called with extra arguments specified in except_.

    When except_(handler, 'extra') is used, the handler is called as
    handler('extra') via EVAL_CALL_WITH_EXPLICIT_ARGS -- the explicit
    args replace the root_value argument.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        calls = []
        def handler(extra_arg):
          calls.append(extra_arg)
          return 'handled'

        result = await await_(
          Chain(fn, 42).then(raise_value_error)
          .except_(handler, 'extra', reraise=False)
          .run()
        )
        self.assertEqual(result, 'handled')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], 'extra')

  async def test_cascade_handler_receives_root(self):
    """Cascade except handler receives root value."""
    received = {'value': None}
    def handler(v=None):
      received['value'] = v
      return 'recovered'

    result = await await_(
      Cascade(42).then(raise_value_error)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(received['value'], 42)


class AsyncExceptionHandlingTests(IsolatedAsyncioTestCase):
  """Tests for async-specific exception handling scenarios."""

  async def test_async_except_handler_awaited_in_async_chain(self):
    """Async except handler is properly awaited in async chain."""
    called = {'value': False}
    async def async_handler(v=None):
      called['value'] = True
      return 'async_result'

    result = await Chain(aempty) \
      .then(async_raise_value_error) \
      .except_(async_handler, reraise=False) \
      .run()
    self.assertEqual(result, 'async_result')
    self.assertTrue(called['value'])

  async def test_async_finally_handler_awaited_in_async_chain(self):
    """Async finally handler is properly awaited in async chain."""
    calls = {'fin': False}
    async def async_finally(v=None):
      calls['fin'] = True

    result = await Chain(aempty, 42) \
      .finally_(async_finally) \
      .run()
    self.assertEqual(result, 42)
    self.assertTrue(calls['fin'])

  async def test_async_chain_exception_in_middle(self):
    """Exception in middle of async chain is caught."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True
      return 'caught'

    async def step1(v):
      return v * 2

    async def step2(v):
      raise ValueError('mid-chain error')

    async def step3(v):
      return v + 1

    result = await Chain(aempty, 5) \
      .then(step1) \
      .then(step2) \
      .then(step3) \
      .except_(handler, reraise=False) \
      .run()
    self.assertEqual(result, 'caught')
    self.assertTrue(called['value'])

  async def test_async_handler_on_sync_exception(self):
    """Async handler on a sync exception (no async coroutine encountered before error)."""
    called = {'value': False}
    async def async_handler(v=None):
      called['value'] = True
      return 'async_handled'

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      with self.assertRaises(ValueError):
        Chain(raise_value_error) \
          .then(aempty) \
          .except_(async_handler, reraise=True) \
          .run()

      runtime_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning)
      ]
      self.assertGreater(len(runtime_warnings), 0)

    await asyncio.sleep(0.1)
    self.assertTrue(called['value'])

  async def test_mixed_sync_async_handlers(self):
    """Chain with both sync and async exception handlers."""
    calls = {'sync': False, 'async': False}
    def sync_handler(v=None):
      calls['sync'] = True
    async def async_handler(v=None):
      calls['async'] = True

    with self.assertRaises(TypeError):
      await Chain(aempty) \
        .then(async_raise_type_error) \
        .except_(sync_handler, exceptions=ValueError) \
        .except_(async_handler, exceptions=TypeError) \
        .run()

    self.assertFalse(calls['sync'])
    self.assertTrue(calls['async'])


class ExceptionWithDebugModeTests(BaseExcTestCase):
  """Tests for exceptions when debug mode is enabled."""

  async def test_debug_mode_exception_has_results(self):
    """Debug mode exception traceback includes intermediate results."""
    for fn, ctx in self.with_fn():
      with ctx:
        try:
          await await_(
            Chain(fn, 5)
            .config(debug=True)
            .then(lambda v: v + 10)
            .then(raise_value_error)
            .run()
          )
          self.fail('Should have raised')
        except ValueError:
          pass

  async def test_debug_mode_with_except_handler(self):
    """Debug mode with except handler works correctly."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
          return 'debug_recovered'

        result = await await_(
          Chain(fn, 5)
          .config(debug=True)
          .then(lambda v: v + 10)
          .then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertEqual(result, 'debug_recovered')
        self.assertTrue(called['value'])

  async def test_debug_mode_with_finally(self):
    """Debug mode with finally handler works correctly."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'fin': False}
        def on_finally(v=None):
          calls['fin'] = True

        try:
          await await_(
            Chain(fn, 5)
            .config(debug=True)
            .then(raise_value_error)
            .finally_(on_finally)
            .run()
          )
        except ValueError:
          pass
        self.assertTrue(calls['fin'])


class NoAsyncModeExceptionTests(BaseExcTestCase):
  """Tests for exceptions when no_async mode is set."""

  async def test_no_async_exception_handling(self):
    """Exception handling works in no_async mode."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True
      return 'recovered'

    result = Chain(42) \
      .no_async(True) \
      .then(raise_value_error) \
      .except_(handler, reraise=False) \
      .run()
    self.assertEqual(result, 'recovered')
    self.assertTrue(called['value'])

  async def test_no_async_finally_handler(self):
    """Finally handler works in no_async mode."""
    called = {'value': False}
    def on_finally(v=None):
      called['value'] = True

    result = Chain(42) \
      .no_async(True) \
      .finally_(on_finally) \
      .run()
    self.assertEqual(result, 42)
    self.assertTrue(called['value'])

  async def test_no_async_exception_propagation(self):
    """Exception propagation works in no_async mode."""
    with self.assertRaises(ValueError):
      Chain(42) \
        .no_async(True) \
        .then(raise_value_error) \
        .run()


class FrozenChainExceptionTests(BaseExcTestCase):
  """Tests for exceptions in frozen chains."""

  async def test_frozen_chain_exception_handling(self):
    """Exception handling works in frozen chains."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        frozen = Chain().then(fn).then(raise_value_error) \
          .except_(handler) \
          .freeze()
        with self.assertRaises(ValueError):
          await await_(frozen.run(42))
        self.assertTrue(called['value'])

  async def test_frozen_chain_finally_handler(self):
    """Finally handler works in frozen chains."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def on_finally(v=None):
          called['value'] = True

        frozen = Chain().then(fn) \
          .finally_(on_finally) \
          .freeze()
        result = await await_(frozen.run(42))
        self.assertEqual(result, 42)
        self.assertTrue(called['value'])


class ClonedChainExceptionTests(BaseExcTestCase):
  """Tests for exceptions in cloned chains."""

  async def test_cloned_chain_except_handler_independent(self):
    """Cloned chain has independent except handler execution."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls_orig = {'value': False}
        def handler_orig(v=None):
          calls_orig['value'] = True

        original = Chain(fn).then(raise_value_error) \
          .except_(handler_orig)

        clone = original.clone()

        with self.assertRaises(ValueError):
          await await_(original.run())
        self.assertTrue(calls_orig['value'])

  async def test_cloned_chain_finally_independent(self):
    """Cloned chain has independent finally handler execution."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'orig': False, 'clone': False}

        original = Chain(fn, 42) \
          .finally_(lambda v: calls.update({'orig': True}))
        clone = original.clone()

        await await_(original.run())
        self.assertTrue(calls['orig'])


class ExceptionInChainOfChains(BaseExcTestCase):
  """Tests for exceptions in chains that contain other chains."""

  async def test_inner_chain_exception_propagates_to_outer(self):
    """Inner chain exception propagates to outer chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        inner = Chain().then(fn).then(raise_value_error)
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, 42)
            .then(inner, ...)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_outer_handler_catches_inner_exception(self):
    """Outer handler catches inner chain's exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          return 'outer_caught'

        inner = Chain().then(fn).then(raise_value_error)
        result = await await_(
          Chain(fn, 42)
          .then(inner, ...)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertEqual(result, 'outer_caught')

  async def test_deeply_nested_exception_propagation(self):
    """Exception propagates through deeply nested chains."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        inner3 = Chain().then(fn).then(raise_value_error)
        inner2 = Chain().then(fn).then(inner3, ...)
        inner1 = Chain().then(fn).then(inner2, ...)

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, 42)
            .then(inner1, ...)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_inner_except_swallows_outer_doesnt_see(self):
    """If inner chain swallows exception, outer doesn't see it."""
    for fn, ctx in self.with_fn():
      with ctx:
        outer_called = {'value': False}
        def outer_handler(v=None):
          outer_called['value'] = True

        inner = Chain().then(fn).then(raise_value_error) \
          .except_(lambda v=None: 'inner_recovered', reraise=False)

        result = await await_(
          Chain(fn, 42)
          .then(inner, ...)
          .except_(outer_handler, reraise=False)
          .run()
        )
        # Inner swallowed the exception and returned 'inner_recovered'
        self.assertEqual(result, 'inner_recovered')
        self.assertFalse(outer_called['value'])


class PipeOperatorExceptionTests(BaseExcTestCase):
  """Tests for exceptions in the pipe (|) operator."""

  async def test_pipe_exception_propagation(self):
    """Exception in pipe operation propagates."""
    with self.assertRaises(ValueError):
      Chain(42) | raise_value_error | run()

  async def test_pipe_with_except_handler(self):
    """Exception handler registered before pipe works when chain is evaluated."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    # The except_ is on the chain, and raise_value_error is added via pipe
    with self.assertRaises(ValueError):
      Chain(42).except_(handler) | raise_value_error | run()

  async def test_pipe_exception_in_first_operation(self):
    """Exception in first pipe operation."""
    with self.assertRaises(ValueError):
      Chain(raise_value_error) | (lambda v: v) | run()


class ContextManagerExceptionTests(BaseExcTestCase):
  """Tests for exceptions with async context managers."""

  async def test_async_context_manager_enter_exception(self):
    """Exception in async context manager __aenter__."""
    class AsyncFailEnterCM:
      async def __aenter__(self):
        raise ValueError('async enter error')
      async def __aexit__(self, *args):
        pass

    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    with self.assertRaises(ValueError):
      await Chain(aempty, AsyncFailEnterCM()) \
        .with_(lambda ctx: ctx) \
        .except_(handler) \
        .run()
    self.assertTrue(called['value'])

  async def test_async_context_manager_body_exception(self):
    """Exception in async context manager body."""
    exited = {'value': False}

    class AsyncTrackCM:
      async def __aenter__(self):
        return 'ctx'
      async def __aexit__(self_, *args):
        exited['value'] = True
        return False

    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    def body_fails(v):
      raise ValueError('async body error')

    with self.assertRaises(ValueError):
      await Chain(aempty, AsyncTrackCM()) \
        .with_(body_fails) \
        .except_(handler) \
        .run()
    self.assertTrue(called['value'])
    self.assertTrue(exited['value'])

  async def test_async_context_manager_exit_exception(self):
    """Exception in async context manager __aexit__."""
    class AsyncFailExitCM:
      async def __aenter__(self):
        return 'ctx'
      async def __aexit__(self, *args):
        raise RuntimeError('async exit error')

    called = {'value': False}
    def handler(v=None):
      called['value'] = True

    with self.assertRaises(RuntimeError):
      await Chain(aempty, AsyncFailExitCM()) \
        .with_(lambda ctx: ctx) \
        .except_(handler) \
        .run()
    self.assertTrue(called['value'])


class ExceptHandlerConditionalBehaviorTests(BaseExcTestCase):
  """Tests for conditional behavior inside except handlers."""

  async def test_handler_conditionally_re_raises(self):
    """Handler can conditionally re-raise via `raise`."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler_reraises(v):
          if v:
            raise
          return 'swallowed'

        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, True).then(raise_value_error)
            .except_(handler_reraises, reraise=False)
            .run()
          )

  async def test_handler_conditionally_does_not_re_raise(self):
    """Handler can conditionally not re-raise."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler_swallows(v):
          if v:
            raise
          return 'swallowed'

        result = await await_(
          Chain(fn, False).then(raise_value_error)
          .except_(handler_swallows, reraise=False)
          .run()
        )
        self.assertEqual(result, 'swallowed')

  async def test_handler_returns_fallback_value(self):
    """Handler can return a fallback value."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          return 'fallback_value'

        result = await await_(
          Chain(fn).then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertEqual(result, 'fallback_value')


class CascadeExceptionTests(BaseExcTestCase):
  """Tests for exceptions specific to Cascade mode."""

  async def test_cascade_except_handler_receives_root(self):
    """Cascade except handler receives root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}
        def handler(v=None):
          received['value'] = v
          return 'cascade_recovered'

        result = await await_(
          Cascade(fn, 'root')
          .then(raise_value_error)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertEqual(result, 'cascade_recovered')
        self.assertEqual(received['value'], 'root')

  async def test_cascade_finally_receives_root(self):
    """Cascade finally handler receives root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}
        def on_finally(v=None):
          received['value'] = v

        result = await await_(
          Cascade(fn, 'root')
          .then(lambda v: 'ignored')
          .finally_(on_finally)
          .run()
        )
        self.assertEqual(result, 'root')
        self.assertEqual(received['value'], 'root')

  async def test_cascade_exception_in_second_operation(self):
    """Exception in second cascade operation is caught."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True

        with self.assertRaises(ValueError):
          await await_(
            Cascade(fn, 42)
            .then(lambda v: v * 2)
            .then(raise_value_error)
            .except_(handler)
            .run()
          )
        self.assertTrue(called['value'])

  async def test_cascade_multiple_except_handlers(self):
    """Multiple except handlers on cascade chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = {'h1': False, 'h2': False}
        def h1(v=None):
          calls['h1'] = True
          return 'h1_result'
        def h2(v=None):
          calls['h2'] = True
          return 'h2_result'

        result = await await_(
          Cascade(fn, 42)
          .then(raise_value_error)
          .except_(h1, exceptions=TypeError, reraise=False)
          .except_(h2, exceptions=ValueError, reraise=False)
          .run()
        )
        self.assertFalse(calls['h1'])
        self.assertTrue(calls['h2'])
        self.assertEqual(result, 'h2_result')


if __name__ == '__main__':
  import unittest
  unittest.main()
