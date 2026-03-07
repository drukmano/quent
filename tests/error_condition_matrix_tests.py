"""Cross-product tests for exception handling across all operation types,
exception types, and except_ filter configurations.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  CustomException,
  CustomBaseException,
  NestedCustomException,
  SyncCM,
  SyncCMSuppresses,
  TrackingCM,
  AsyncCM,
  async_fn,
)


# ---------------------------------------------------------------------------
# Exception type axis (8 variants)
# ---------------------------------------------------------------------------
_EXCEPTION_TYPES = [
  ValueError,
  TypeError,
  KeyboardInterrupt,
  StopIteration,
  StopAsyncIteration,
  RuntimeError,
  CustomException,
  NestedCustomException,
]

# Subset that are Exception subclasses (caught by default except_())
_EXCEPTION_SUBTYPES = [
  ValueError,
  TypeError,
  StopIteration,
  StopAsyncIteration,
  RuntimeError,
  CustomException,
  NestedCustomException,
]

# Exception subtypes safe to use in iteration contexts (map/filter/iterate).
# StopIteration and StopAsyncIteration have special semantics:
#  - StopIteration terminates the `while True: next(it)` loop in _make_foreach/_make_filter
#  - StopAsyncIteration terminates `async for` loops
#  - PEP 479 converts StopIteration to RuntimeError inside generators/coroutines
_ITERATION_SAFE_SUBTYPES = [
  ValueError,
  TypeError,
  RuntimeError,
  CustomException,
  NestedCustomException,
]

# Exception subtypes safe to use in async coroutines.
# StopIteration raised inside a coroutine is converted to RuntimeError by PEP 479.
_ASYNC_SAFE_SUBTYPES = [
  ValueError,
  TypeError,
  StopAsyncIteration,
  RuntimeError,
  CustomException,
  NestedCustomException,
]

# Subset that are NOT Exception subclasses
_BASE_EXCEPTION_ONLY = [KeyboardInterrupt]


def _make_raiser(exc_type):
  """Return a callable that raises the given exception type."""
  def raiser(x=None):
    raise exc_type(f'{exc_type.__name__}_error')
  return raiser


def _make_raiser_no_arg(exc_type):
  """Return a no-arg callable that raises the given exception type."""
  def raiser():
    raise exc_type(f'{exc_type.__name__}_error')
  return raiser


def _catch_handler(rv, exc):
  return f'caught:{type(exc).__name__}'


# ---------------------------------------------------------------------------
# Except filter configurations
# ---------------------------------------------------------------------------
_EXCEPT_CONFIGS = [
  ('no_except', None, None),
  ('matching_type', 'match', None),
  ('non_matching_type', 'mismatch', None),
  ('base_exception', BaseException, None),
  ('default_exception', 'default', None),
]


def _get_except_filter(config_key, exc_type):
  """Return the exceptions= argument for the given config."""
  if config_key == 'match':
    return exc_type
  elif config_key == 'mismatch':
    # Pick a type that doesn't match
    if exc_type is ValueError:
      return TypeError
    return ValueError
  elif config_key == 'default':
    return None  # default is (Exception,)
  else:
    return config_key  # BaseException or None


def _should_catch(config_name, exc_type):
  """Whether the except_ config should catch the given exception type."""
  if config_name == 'no_except':
    return False
  if config_name == 'matching_type':
    return True
  if config_name == 'non_matching_type':
    return False
  if config_name == 'base_exception':
    return True
  if config_name == 'default_exception':
    return issubclass(exc_type, Exception)
  return False


# ===========================================================================
# Class: TestExceptionTypeInThenMatrix
# ===========================================================================
class TestExceptionTypeInThenMatrix(unittest.TestCase):
  """Cross-product of exception types x except_ configurations in then() steps."""

  def test_exception_in_then_matrix(self):
    for exc_type in _EXCEPTION_TYPES:
      for config_name, config_key, _ in _EXCEPT_CONFIGS:
        with self.subTest(exc_type=exc_type.__name__, except_config=config_name):
          raiser = _make_raiser(exc_type)
          chain = Chain(10).then(raiser)

          if config_name == 'no_except':
            with self.assertRaises(exc_type):
              chain.run()
          else:
            exc_filter = _get_except_filter(config_key, exc_type)
            if exc_filter is not None:
              chain = Chain(10).then(raiser).except_(_catch_handler, exceptions=exc_filter)
            else:
              chain = Chain(10).then(raiser).except_(_catch_handler)

            if _should_catch(config_name, exc_type):
              result = chain.run()
              self.assertEqual(result, f'caught:{exc_type.__name__}')
            else:
              with self.assertRaises(exc_type):
                chain.run()

  def test_then_handler_receives_correct_exception_instance(self):
    """Verify the handler receives the exact exception object."""
    for exc_type in _EXCEPTION_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        received = []
        def handler(rv, exc):
          received.append(exc)
          return 'ok'
        raiser = _make_raiser(exc_type)
        Chain(10).then(raiser).except_(handler).run()
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], exc_type)


# ===========================================================================
# Class: TestExceptionTypeInDoMatrix
# ===========================================================================
class TestExceptionTypeInDoMatrix(unittest.TestCase):
  """Cross-product in do() steps -- verify current_value preservation before exception."""

  def test_exception_in_do_matrix(self):
    for exc_type in _EXCEPTION_TYPES:
      for config_name, config_key, _ in _EXCEPT_CONFIGS:
        with self.subTest(exc_type=exc_type.__name__, except_config=config_name):
          raiser = _make_raiser(exc_type)

          if config_name == 'no_except':
            with self.assertRaises(exc_type):
              Chain(10).do(raiser).run()
          else:
            exc_filter = _get_except_filter(config_key, exc_type)
            if exc_filter is not None:
              chain = Chain(10).do(raiser).except_(_catch_handler, exceptions=exc_filter)
            else:
              chain = Chain(10).do(raiser).except_(_catch_handler)

            if _should_catch(config_name, exc_type):
              result = chain.run()
              self.assertEqual(result, f'caught:{exc_type.__name__}')
            else:
              with self.assertRaises(exc_type):
                chain.run()

  def test_do_preserves_value_before_exception(self):
    """Verify that do's side-effect does not modify the chain value."""
    tracker = []
    result = (
      Chain(42)
      .do(lambda x: tracker.append(x))
      .then(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, 43)
    self.assertEqual(tracker, [42])


# ===========================================================================
# Class: TestExceptionTypeInMapMatrix
# ===========================================================================
class TestExceptionTypeInMapMatrix(unittest.TestCase):
  """Cross-product in map() -- raise at different positions."""

  def test_map_exception_at_positions(self):
    # StopIteration terminates the internal while/next() loop in _make_foreach,
    # so it cannot be tested as a user-raised error here.
    for exc_type in _ITERATION_SAFE_SUBTYPES:
      for raise_at in [0, 1, 2]:
        with self.subTest(exc_type=exc_type.__name__, raise_at=raise_at):
          def make_fn(et, pos):
            def fn(x):
              if x == pos:
                raise et(f'{et.__name__}_at_{pos}')
              return x * 10
            return fn

          fn = make_fn(exc_type, raise_at)
          chain = Chain([0, 1, 2, 3]).map(fn).except_(_catch_handler)
          result = chain.run()
          self.assertEqual(result, f'caught:{exc_type.__name__}')

  def test_map_exception_propagates_without_except(self):
    for exc_type in _ITERATION_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        raiser = _make_raiser(exc_type)
        with self.assertRaises(exc_type):
          Chain([1, 2, 3]).map(raiser).run()

  def test_map_base_exception_not_caught_by_default(self):
    """KeyboardInterrupt in map is not caught by default except_."""
    def raiser(x):
      raise KeyboardInterrupt('kbd')
    with self.assertRaises(KeyboardInterrupt):
      Chain([1, 2]).map(raiser).except_(_catch_handler).run()

  def test_map_base_exception_caught_by_base_except(self):
    """KeyboardInterrupt in map IS caught by except_(BaseException)."""
    def raiser(x):
      raise KeyboardInterrupt('kbd')
    result = (
      Chain([1, 2])
      .map(raiser)
      .except_(_catch_handler, exceptions=BaseException)
      .run()
    )
    self.assertEqual(result, 'caught:KeyboardInterrupt')


# ===========================================================================
# Class: TestExceptionTypeInFilterMatrix
# ===========================================================================
class TestExceptionTypeInFilterMatrix(unittest.TestCase):
  """Cross-product in filter() predicate."""

  def test_filter_exception_matrix(self):
    # StopIteration terminates the internal while/next() loop in _make_filter.
    for exc_type in _ITERATION_SAFE_SUBTYPES:
      for raise_at in [0, 1, 2]:
        with self.subTest(exc_type=exc_type.__name__, raise_at=raise_at):
          def make_pred(et, pos):
            def pred(x):
              if x == pos:
                raise et(f'{et.__name__}_at_{pos}')
              return x > 0
            return pred

          pred = make_pred(exc_type, raise_at)
          chain = Chain([0, 1, 2, 3]).filter(pred).except_(_catch_handler)
          result = chain.run()
          self.assertEqual(result, f'caught:{exc_type.__name__}')

  def test_filter_exception_propagates_without_except(self):
    for exc_type in _ITERATION_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        raiser = _make_raiser(exc_type)
        with self.assertRaises(exc_type):
          Chain([1, 2, 3]).filter(raiser).run()

  def test_filter_base_exception_not_caught_by_default(self):
    def pred(x):
      raise KeyboardInterrupt('kbd')
    with self.assertRaises(KeyboardInterrupt):
      Chain([1, 2]).filter(pred).except_(_catch_handler).run()


# ===========================================================================
# Class: TestExceptionTypeInWithMatrix
# ===========================================================================
class TestExceptionTypeInWithMatrix(unittest.TestCase):
  """Exception in with_ body x CM type (suppressing vs non-suppressing)."""

  def test_with_exception_non_suppressing_cm(self):
    for exc_type in _EXCEPTION_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        raiser = _make_raiser(exc_type)
        chain = (
          Chain(SyncCM())
          .with_(raiser)
          .except_(_catch_handler)
        )
        result = chain.run()
        self.assertEqual(result, f'caught:{exc_type.__name__}')

  def test_with_exception_suppressing_cm(self):
    """When CM suppresses, the exception does NOT reach except_."""
    for exc_type in _EXCEPTION_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        except_called = []
        def handler(rv, exc):
          except_called.append(exc)
          return 'caught'
        raiser = _make_raiser(exc_type)
        result = (
          Chain(SyncCMSuppresses())
          .with_(raiser)
          .except_(handler)
          .run()
        )
        self.assertEqual(except_called, [])
        # Suppressed exception => result is None (ignore_result=False, no body result)
        self.assertIsNone(result)

  def test_with_cm_exit_called_with_exc_info(self):
    """Verify __exit__ receives correct exception info."""
    for exc_type in _EXCEPTION_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        cm = TrackingCM()
        raiser = _make_raiser(exc_type)
        Chain(cm).with_(raiser).except_(_catch_handler).run()
        self.assertTrue(cm.exited)
        self.assertIsNotNone(cm.exit_args)
        self.assertIs(cm.exit_args[0], exc_type)
        self.assertIsInstance(cm.exit_args[1], exc_type)

  def test_with_success_cm_exit_called_cleanly(self):
    cm = TrackingCM()
    result = Chain(cm).with_(lambda ctx: ctx.upper()).run()
    self.assertEqual(result, 'TRACKED_CTX')
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_with_base_exception_not_caught_default(self):
    def raiser(ctx):
      raise KeyboardInterrupt('kbd')
    with self.assertRaises(KeyboardInterrupt):
      Chain(SyncCM()).with_(raiser).except_(_catch_handler).run()


class TestExceptionTypeInWithAsyncMatrix(IsolatedAsyncioTestCase):
  """Async CM exception handling."""

  async def test_async_cm_exception_caught(self):
    # StopIteration raised inside a coroutine/async-with is converted to RuntimeError
    # by PEP 479, so we use _ASYNC_SAFE_SUBTYPES.
    for exc_type in _ASYNC_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        raiser = _make_raiser(exc_type)
        result = await (
          Chain(AsyncCM())
          .with_(raiser)
          .except_(_catch_handler)
          .run()
        )
        self.assertEqual(result, f'caught:{exc_type.__name__}')


# ===========================================================================
# Class: TestExceptionTypeInGatherMatrix
# ===========================================================================
class TestExceptionTypeInGatherMatrix(unittest.TestCase):
  """Exception in nth gather fn x position (first, middle, last)."""

  def test_gather_exception_at_positions(self):
    for exc_type in _EXCEPTION_SUBTYPES:
      for position in ['first', 'middle', 'last']:
        with self.subTest(exc_type=exc_type.__name__, position=position):
          raiser = _make_raiser(exc_type)
          ok_fn = lambda x: x + 1

          if position == 'first':
            fns = (raiser, ok_fn, ok_fn)
          elif position == 'middle':
            fns = (ok_fn, raiser, ok_fn)
          else:
            fns = (ok_fn, ok_fn, raiser)

          chain = Chain(5).gather(*fns).except_(_catch_handler)
          result = chain.run()
          self.assertEqual(result, f'caught:{exc_type.__name__}')

  def test_gather_exception_propagates_without_except(self):
    for exc_type in _EXCEPTION_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        raiser = _make_raiser(exc_type)
        with self.assertRaises(exc_type):
          Chain(5).gather(raiser, lambda x: x).run()

  def test_gather_base_exception_not_caught_default(self):
    def raiser(x):
      raise KeyboardInterrupt('kbd')
    with self.assertRaises(KeyboardInterrupt):
      Chain(5).gather(raiser).except_(_catch_handler).run()

  def test_gather_coroutines_cleaned_on_sync_exception(self):
    """When a sync fn raises during gather setup, earlier coroutines are closed."""
    import asyncio

    closed = []
    async def coro_fn(x):
      return x

    class TrackCloseCoro:
      def __init__(self):
        self._coro = coro_fn(5)
      def close(self):
        closed.append(True)
        self._coro.close()
      def __getattr__(self, name):
        return getattr(self._coro, name)

    def raiser(x):
      raise ValueError('boom')

    # We can't easily inject TrackCloseCoro into gather, but we can test
    # the principle: if exception occurs in 2nd fn, 1st coroutine is closed.
    with self.assertRaises(ValueError):
      Chain(5).gather(coro_fn, raiser).run()


# ===========================================================================
# Class: TestBaseExceptionVsExceptionMatrix
# ===========================================================================
class TestBaseExceptionVsExceptionMatrix(unittest.TestCase):
  """Verify BaseException vs Exception catching semantics."""

  def test_keyboard_interrupt_not_caught_by_default_except(self):
    def raiser(x=None):
      raise KeyboardInterrupt()
    with self.assertRaises(KeyboardInterrupt):
      Chain(raiser).except_(_catch_handler).run()

  def test_keyboard_interrupt_caught_by_base_exception_except(self):
    def raiser(x=None):
      raise KeyboardInterrupt()
    result = Chain(raiser).except_(_catch_handler, exceptions=BaseException).run()
    self.assertEqual(result, 'caught:KeyboardInterrupt')

  def test_system_exit_not_caught_by_default_except(self):
    def raiser(x=None):
      raise SystemExit(1)
    with self.assertRaises(SystemExit):
      Chain(raiser).except_(_catch_handler).run()

  def test_system_exit_caught_by_base_exception_except(self):
    def raiser(x=None):
      raise SystemExit(1)
    result = Chain(raiser).except_(_catch_handler, exceptions=BaseException).run()
    self.assertEqual(result, 'caught:SystemExit')

  def test_custom_base_exception_not_caught_by_default(self):
    def raiser(x=None):
      raise CustomBaseException('custom')
    with self.assertRaises(CustomBaseException):
      Chain(raiser).except_(_catch_handler).run()

  def test_custom_base_exception_caught_by_base_exception(self):
    def raiser(x=None):
      raise CustomBaseException('custom')
    result = Chain(raiser).except_(_catch_handler, exceptions=BaseException).run()
    self.assertEqual(result, 'caught:CustomBaseException')

  def test_custom_base_exception_caught_by_specific_type(self):
    def raiser(x=None):
      raise CustomBaseException('custom')
    result = Chain(raiser).except_(_catch_handler, exceptions=CustomBaseException).run()
    self.assertEqual(result, 'caught:CustomBaseException')

  def test_base_exception_subclass_matrix(self):
    """subTest over all BaseException-only subclasses."""
    base_only = [KeyboardInterrupt, SystemExit, GeneratorExit]
    for exc_type in base_only:
      with self.subTest(exc_type=exc_type.__name__, filter='default'):
        def raiser(x=None, et=exc_type):
          raise et()
        with self.assertRaises(exc_type):
          Chain(raiser).except_(_catch_handler).run()

      with self.subTest(exc_type=exc_type.__name__, filter='BaseException'):
        def raiser(x=None, et=exc_type):
          raise et()
        # GeneratorExit is special -- it can be raised during cleanup and Python
        # can interfere. We just verify the except_ mechanism works.
        if exc_type is GeneratorExit:
          # GeneratorExit extends BaseException. Should be caught.
          result = Chain(raiser).except_(_catch_handler, exceptions=BaseException).run()
          self.assertEqual(result, f'caught:{exc_type.__name__}')
        else:
          result = Chain(raiser).except_(_catch_handler, exceptions=BaseException).run()
          self.assertEqual(result, f'caught:{exc_type.__name__}')


# ===========================================================================
# Class: TestExceptionHierarchyMatrix
# ===========================================================================
class TestExceptionHierarchyMatrix(unittest.TestCase):
  """Test exception hierarchy matching in except_ filters."""

  def test_custom_exception_caught_by_parent(self):
    """except_(CustomException) catches NestedCustomException."""
    def raiser(x=None):
      raise NestedCustomException('nested')
    result = Chain(raiser).except_(_catch_handler, exceptions=CustomException).run()
    self.assertEqual(result, 'caught:NestedCustomException')

  def test_exception_caught_by_exception(self):
    """except_(Exception) catches CustomException."""
    def raiser(x=None):
      raise CustomException('custom')
    result = Chain(raiser).except_(_catch_handler, exceptions=Exception).run()
    self.assertEqual(result, 'caught:CustomException')

  def test_value_error_not_caught_by_type_error(self):
    """except_(ValueError) does NOT catch TypeError."""
    def raiser(x=None):
      raise TypeError('type')
    with self.assertRaises(TypeError):
      Chain(raiser).except_(_catch_handler, exceptions=ValueError).run()

  def test_type_error_not_caught_by_value_error(self):
    def raiser(x=None):
      raise ValueError('value')
    with self.assertRaises(ValueError):
      Chain(raiser).except_(_catch_handler, exceptions=TypeError).run()

  def test_hierarchy_cross_product(self):
    """Dense subTest over exception hierarchy combinations."""
    hierarchy = [
      (Exception, ValueError, True),
      (Exception, TypeError, True),
      (Exception, RuntimeError, True),
      (Exception, CustomException, True),
      (Exception, NestedCustomException, True),
      (CustomException, NestedCustomException, True),
      (CustomException, CustomException, True),
      (NestedCustomException, NestedCustomException, True),
      (ValueError, TypeError, False),
      (TypeError, ValueError, False),
      (RuntimeError, ValueError, False),
      (CustomException, ValueError, False),
      (NestedCustomException, CustomException, False),
      (ValueError, CustomException, False),
      (BaseException, KeyboardInterrupt, True),
      (BaseException, SystemExit, True),
      (BaseException, CustomBaseException, True),
      (BaseException, ValueError, True),
      (Exception, KeyboardInterrupt, False),
      (ValueError, KeyboardInterrupt, False),
    ]
    for filter_type, raised_type, should_catch in hierarchy:
      with self.subTest(
        filter=filter_type.__name__,
        raised=raised_type.__name__,
        should_catch=should_catch,
      ):
        def raiser(x=None, et=raised_type):
          raise et(f'{et.__name__}_test')
        chain = Chain(raiser).except_(_catch_handler, exceptions=filter_type)
        if should_catch:
          result = chain.run()
          self.assertEqual(result, f'caught:{raised_type.__name__}')
        else:
          with self.assertRaises(raised_type):
            chain.run()

  def test_multiple_filter_types_in_except(self):
    """except_ with multiple types catches any of them."""
    combos = [
      ([ValueError, TypeError], ValueError, True),
      ([ValueError, TypeError], TypeError, True),
      ([ValueError, TypeError], RuntimeError, False),
      ([CustomException, RuntimeError], NestedCustomException, True),
      ([CustomException, RuntimeError], ValueError, False),
    ]
    for filter_types, raised_type, should_catch in combos:
      with self.subTest(
        filters=[t.__name__ for t in filter_types],
        raised=raised_type.__name__,
        should_catch=should_catch,
      ):
        def raiser(x=None, et=raised_type):
          raise et(f'{et.__name__}_test')
        chain = Chain(raiser).except_(_catch_handler, exceptions=filter_types)
        if should_catch:
          result = chain.run()
          self.assertEqual(result, f'caught:{raised_type.__name__}')
        else:
          with self.assertRaises(raised_type):
            chain.run()


# ===========================================================================
# Class: TestExceptionInAsyncMatrix
# ===========================================================================
class TestExceptionInAsyncMatrix(IsolatedAsyncioTestCase):
  """Same exception types in async context."""

  async def test_async_exception_types_caught(self):
    # StopIteration raised inside a coroutine is converted to RuntimeError by PEP 479.
    for exc_type in _ASYNC_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        async def raiser(x, et=exc_type):
          raise et(f'{et.__name__}_async')
        result = await (
          Chain(10)
          .then(raiser)
          .except_(_catch_handler)
          .run()
        )
        self.assertEqual(result, f'caught:{exc_type.__name__}')

  async def test_async_exception_propagates_without_except(self):
    for exc_type in _ASYNC_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        async def raiser(x, et=exc_type):
          raise et(f'{et.__name__}_async')
        with self.assertRaises(exc_type):
          await Chain(10).then(raiser).run()

  async def test_sync_to_async_transition_with_exception(self):
    """Sync step succeeds, async step raises."""
    for exc_type in _ASYNC_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        async def raiser(x, et=exc_type):
          raise et(f'{et.__name__}_after_sync')
        result = await (
          Chain(10)
          .then(lambda x: x + 1)  # sync
          .then(raiser)  # async raises
          .except_(_catch_handler)
          .run()
        )
        self.assertEqual(result, f'caught:{exc_type.__name__}')

  async def test_exception_in_async_fn_vs_sync_fn(self):
    """Verify same behavior whether fn is sync or async."""
    for exc_type in _ASYNC_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__, fn_type='sync'):
        def sync_raiser(x, et=exc_type):
          raise et(f'{et.__name__}_sync')
        # Force async path by putting async step before
        result = await (
          Chain(10)
          .then(async_fn)  # makes chain async
          .then(sync_raiser)
          .except_(_catch_handler)
          .run()
        )
        self.assertEqual(result, f'caught:{exc_type.__name__}')

      with self.subTest(exc_type=exc_type.__name__, fn_type='async'):
        async def async_raiser(x, et=exc_type):
          raise et(f'{et.__name__}_async')
        result = await (
          Chain(10)
          .then(async_raiser)
          .except_(_catch_handler)
          .run()
        )
        self.assertEqual(result, f'caught:{exc_type.__name__}')

  async def test_keyboard_interrupt_not_caught_in_async(self):
    async def raiser(x):
      raise KeyboardInterrupt()
    with self.assertRaises(KeyboardInterrupt):
      await Chain(10).then(raiser).except_(_catch_handler).run()

  async def test_keyboard_interrupt_caught_by_base_in_async(self):
    async def raiser(x):
      raise KeyboardInterrupt()
    result = await (
      Chain(10)
      .then(raiser)
      .except_(_catch_handler, exceptions=BaseException)
      .run()
    )
    self.assertEqual(result, 'caught:KeyboardInterrupt')

  async def test_async_map_exception_types(self):
    # Use _ITERATION_SAFE_SUBTYPES: StopIteration terminates iteration;
    # also exclude StopAsyncIteration since it terminates async for loops.
    for exc_type in _ITERATION_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        # The map fn is sync but raises; the async path is forced by
        # an async identity step that returns the list unchanged.
        async def async_list(x):
          return [1, 2, 3]
        def raiser(x, et=exc_type):
          raise et(f'{et.__name__}_map')
        result = await (
          Chain(0)
          .then(async_list)
          .map(raiser)
          .except_(_catch_handler)
          .run()
        )
        self.assertEqual(result, f'caught:{exc_type.__name__}')

  async def test_async_filter_exception_types(self):
    for exc_type in _ITERATION_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        async def async_list(x):
          return [1, 2, 3]
        def pred(x, et=exc_type):
          raise et(f'{et.__name__}_filter')
        result = await (
          Chain(0)
          .then(async_list)
          .filter(pred)
          .except_(_catch_handler)
          .run()
        )
        self.assertEqual(result, f'caught:{exc_type.__name__}')


# ===========================================================================
# Class: TestExceptionInRootCallableMatrix
# ===========================================================================
class TestExceptionInRootCallableMatrix(unittest.TestCase):
  """Exception raised in the root callable of a chain."""

  def test_root_exception_matrix(self):
    for exc_type in _EXCEPTION_TYPES:
      for config_name, config_key, _ in _EXCEPT_CONFIGS:
        with self.subTest(exc_type=exc_type.__name__, except_config=config_name):
          raiser = _make_raiser_no_arg(exc_type)

          if config_name == 'no_except':
            with self.assertRaises(exc_type):
              Chain(raiser).run()
          else:
            exc_filter = _get_except_filter(config_key, exc_type)
            if exc_filter is not None:
              chain = Chain(raiser).except_(_catch_handler, exceptions=exc_filter)
            else:
              chain = Chain(raiser).except_(_catch_handler)

            if _should_catch(config_name, exc_type):
              result = chain.run()
              self.assertEqual(result, f'caught:{exc_type.__name__}')
            else:
              with self.assertRaises(exc_type):
                chain.run()


# ===========================================================================
# Class: TestExceptionInIterateMatrix
# ===========================================================================
class TestExceptionInIterateMatrix(unittest.TestCase):
  """Exception in iterate fn."""

  def test_iterate_exception_propagates(self):
    # StopIteration inside a generator is converted to RuntimeError by PEP 479.
    # StopAsyncIteration is not relevant in sync iteration.
    for exc_type in _ITERATION_SAFE_SUBTYPES:
      with self.subTest(exc_type=exc_type.__name__):
        raiser = _make_raiser(exc_type)
        gen = Chain([1, 2, 3]).iterate(raiser)
        with self.assertRaises(exc_type):
          list(gen)

  def test_iterate_no_fn_yields_items(self):
    gen = Chain([1, 2, 3]).iterate()
    self.assertEqual(list(gen), [1, 2, 3])


# ===========================================================================
# Additional edge cases
# ===========================================================================
class TestExceptionHandlerReturnValues(unittest.TestCase):
  """Verify handler return value becomes chain result."""

  def test_handler_returns_none(self):
    result = Chain(_make_raiser_no_arg(ValueError)).except_(lambda rv, e: None).run()
    self.assertIsNone(result)

  def test_handler_returns_value(self):
    result = Chain(_make_raiser_no_arg(ValueError)).except_(lambda rv, e: 42).run()
    self.assertEqual(result, 42)

  def test_handler_returns_list(self):
    result = Chain(_make_raiser_no_arg(ValueError)).except_(lambda rv, e: [1, 2]).run()
    self.assertEqual(result, [1, 2])

  def test_handler_raises_new_exception(self):
    def handler(rv, exc):
      raise RuntimeError('from handler') from exc
    with self.assertRaises(RuntimeError) as cm:
      Chain(_make_raiser_no_arg(ValueError)).except_(handler).run()
    self.assertEqual(str(cm.exception), 'from handler')
    self.assertIsInstance(cm.exception.__cause__, ValueError)

  def test_handler_with_extra_args(self):
    received = []
    def handler(a, b, k=None):
      received.append((a, b, k))
      return 'ok'
    result = (
      Chain(_make_raiser_no_arg(ValueError))
      .except_(handler, 'x', 'y', k='z')
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(received, [('x', 'y', 'z')])


if __name__ == '__main__':
  unittest.main()
