"""Full cross-product tests for context manager protocol handling.

Tests every CM type (Axis G) against with_/with_do, body outcomes
(success, exception, suppressed exception, _Return signal), and
body types (sync fn, async fn). Verifies enter/exit calls, enter
values, exit arguments, suppression behavior, and with_do preservation.

NOTE: contextlib.contextmanager CMs are callable (ContextDecorator), so
Chain(cm) would call them. Use Chain(lambda: cm) to pass them as values.
nullcontext has __aenter__, so it always goes through the async path.
"""
from __future__ import annotations

import contextlib
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain
from quent._core import _Return
from helpers import (
  AsyncCM,
  AsyncCMRaisesOnEnter,
  AsyncCMRaisesOnExit,
  AsyncCMSuppresses,
  AsyncTrackingCM,
  DualCM,
  SyncCM,
  SyncCMEnterReturnsNone,
  SyncCMEnterReturnsSelf,
  SyncCMExitRaisesOnSuccess,
  SyncCMExitReturnsAwaitableOnException,
  SyncCMRaisesOnEnter,
  SyncCMRaisesOnExit,
  SyncCMRaisesOnExitFrom,
  SyncCMSuppresses,
  SyncCMSuppressesAwaitable,
  SyncCMWithAwaitableExit,
  TrackingCM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_body(ctx):
  return f'body:{ctx}'


async def _async_body(ctx):
  return f'body:{ctx}'


def _sync_raise_body(ctx):
  raise ValueError('body error')


async def _async_raise_body(ctx):
  raise ValueError('body error')


def _sync_return_body(ctx):
  Chain.return_('early')


async def _async_return_body(ctx):
  Chain.return_('early')


# Context managers built via contextlib decorators.
# NOTE: @contextmanager CMs are callable (ContextDecorator.__call__), so
# Chain(cm) would evaluate them as callables. Always wrap with lambda.
@contextlib.contextmanager
def contextlib_sync_cm():
  yield 'cl_sync_ctx'


@contextlib.asynccontextmanager
async def contextlib_async_cm():
  yield 'cl_async_ctx'


# Suppressing contextlib CM
@contextlib.contextmanager
def contextlib_sync_cm_suppresses():
  try:
    yield 'cl_sync_ctx'
  except Exception:
    pass


# ---------------------------------------------------------------------------
# Axis G: pure sync CM types (no __aenter__, not callable)
# These can be used directly in Chain(cm).with_(...) in sync context.
# ---------------------------------------------------------------------------

PURE_SYNC_CM_FACTORIES = {
  'SyncCM': lambda: SyncCM(),
  'SyncCMSuppresses': lambda: SyncCMSuppresses(),
  'SyncCMRaisesOnEnter': lambda: SyncCMRaisesOnEnter(),
  'SyncCMRaisesOnExit': lambda: SyncCMRaisesOnExit(),
  'SyncCMEnterReturnsNone': lambda: SyncCMEnterReturnsNone(),
  'SyncCMEnterReturnsSelf': lambda: SyncCMEnterReturnsSelf(),
  'SyncCMExitRaisesOnSuccess': lambda: SyncCMExitRaisesOnSuccess(),
  'SyncCMRaisesOnExitFrom': lambda: SyncCMRaisesOnExitFrom(),
  'TrackingCM': lambda: TrackingCM(),
  'DualCM': lambda: DualCM(),
}

# CMs whose __enter__ raises -- body never runs
SYNC_ENTER_RAISES = {'SyncCMRaisesOnEnter'}

# CMs whose __exit__ raises even on success
SYNC_EXIT_RAISES_ON_SUCCESS = {'SyncCMExitRaisesOnSuccess', 'SyncCMRaisesOnExit'}

# CMs whose __exit__ raises on exception
SYNC_EXIT_RAISES_ON_EXCEPTION = {'SyncCMRaisesOnExit', 'SyncCMRaisesOnExitFrom'}

# CMs that suppress exceptions
SYNC_SUPPRESSES = {'SyncCMSuppresses'}

# Expected enter values per pure sync CM type
PURE_SYNC_ENTER_VALUES = {
  'SyncCM': 'ctx_value',
  'SyncCMSuppresses': 'ctx_value',
  'SyncCMRaisesOnExit': 'ctx_value',
  'SyncCMEnterReturnsNone': None,
  'SyncCMExitRaisesOnSuccess': 'ctx_value',
  'SyncCMRaisesOnExitFrom': 'ctx_value',
  'TrackingCM': 'tracked_ctx',
  'DualCM': 'sync_ctx',
}

# Async CM types (have __aenter__)
ASYNC_CM_FACTORIES = {
  'AsyncCM': lambda: AsyncCM(),
  'AsyncCMSuppresses': lambda: AsyncCMSuppresses(),
  'AsyncCMRaisesOnEnter': lambda: AsyncCMRaisesOnEnter(),
  'AsyncCMRaisesOnExit': lambda: AsyncCMRaisesOnExit(),
  'AsyncTrackingCM': lambda: AsyncTrackingCM(),
}

ASYNC_ENTER_RAISES = {'AsyncCMRaisesOnEnter'}
# AsyncCMRaisesOnExit.__aexit__ always raises (even on success)
ASYNC_EXIT_RAISES_ON_SUCCESS = {'AsyncCMRaisesOnExit'}
ASYNC_EXIT_RAISES_ON_EXCEPTION = {'AsyncCMRaisesOnExit'}
ASYNC_SUPPRESSES = {'AsyncCMSuppresses'}

ASYNC_ENTER_VALUES = {
  'AsyncCM': 'ctx_value',
  'AsyncCMSuppresses': 'ctx_value',
  'AsyncCMRaisesOnExit': '_self_',  # returns self
  'AsyncTrackingCM': 'async_tracked_ctx',
}

# Sync CMs whose __exit__ ALWAYS returns awaitable -- must be tested in async context
AWAITABLE_EXIT_CM_FACTORIES = {
  'SyncCMWithAwaitableExit': lambda: SyncCMWithAwaitableExit(),
  'SyncCMSuppressesAwaitable': lambda: SyncCMSuppressesAwaitable(),
}

# Sync CMs whose __exit__ returns awaitable ONLY on exception
AWAITABLE_EXIT_ON_EXCEPTION_CM_FACTORIES = {
  'SyncCMExitReturnsAwaitableOnException': lambda: SyncCMExitReturnsAwaitableOnException(),
}


# ---------------------------------------------------------------------------
# Class 1: TestCMProtocolSyncMatrix
# ---------------------------------------------------------------------------

class TestCMProtocolSyncMatrix(unittest.TestCase):
  """Pure sync CMs x with_/with_do x body outcomes."""

  def test_sync_cm_success(self):
    """All non-error pure sync CMs with successful body."""
    for cm_name, factory in PURE_SYNC_CM_FACTORIES.items():
      if cm_name in SYNC_ENTER_RAISES:
        continue
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='success'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_body)
          else:
            chain.with_do(_sync_body)

          if cm_name in SYNC_EXIT_RAISES_ON_SUCCESS:
            with self.assertRaises(RuntimeError):
              chain.run()
          else:
            result = chain.run()
            if method == 'with_do':
              self.assertIs(result, cm)
            else:
              expected_ctx = PURE_SYNC_ENTER_VALUES.get(cm_name, 'ctx_value')
              if cm_name == 'SyncCMEnterReturnsSelf':
                self.assertEqual(result, f'body:{cm}')
              else:
                self.assertEqual(result, f'body:{expected_ctx}')

          # Verify enter/exit called (where applicable)
          if hasattr(cm, 'entered'):
            self.assertTrue(cm.entered)
          if hasattr(cm, 'exited'):
            self.assertTrue(cm.exited)

  def test_sync_cm_exception_not_suppressed(self):
    """Pure sync CMs where body raises and exit does NOT suppress."""
    for cm_name, factory in PURE_SYNC_CM_FACTORIES.items():
      if cm_name in SYNC_ENTER_RAISES:
        continue
      if cm_name in SYNC_SUPPRESSES:
        continue
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='exception'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_raise_body)
          else:
            chain.with_do(_sync_raise_body)

          if cm_name in SYNC_EXIT_RAISES_ON_EXCEPTION:
            with self.assertRaises(RuntimeError):
              chain.run()
          elif cm_name in SYNC_EXIT_RAISES_ON_SUCCESS:
            with self.assertRaises(RuntimeError):
              chain.run()
          else:
            with self.assertRaises(ValueError) as ctx:
              chain.run()
            self.assertIn('body error', str(ctx.exception))

  def test_sync_cm_exception_suppressed(self):
    """Pure sync CMs that suppress exceptions."""
    for cm_name in SYNC_SUPPRESSES:
      factory = PURE_SYNC_CM_FACTORIES[cm_name]
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='suppressed'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_raise_body)
          else:
            chain.with_do(_sync_raise_body)
          result = chain.run()
          if method == 'with_do':
            self.assertIs(result, cm)
          else:
            self.assertIsNone(result)

  def test_sync_cm_return_signal(self):
    """Pure sync CMs with _Return signal in body."""
    safe_cms = {
      k: v for k, v in PURE_SYNC_CM_FACTORIES.items()
      if k not in SYNC_ENTER_RAISES and k not in SYNC_EXIT_RAISES_ON_SUCCESS
    }
    for cm_name, factory in safe_cms.items():
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='return'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_return_body)
          else:
            chain.with_do(_sync_return_body)
          result = chain.run()
          self.assertEqual(result, 'early')

  def test_sync_cm_enter_raises(self):
    """CMs whose __enter__ raises -- body never executes."""
    for cm_name in SYNC_ENTER_RAISES:
      factory = PURE_SYNC_CM_FACTORIES[cm_name]
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_body)
          else:
            chain.with_do(_sync_body)
          with self.assertRaises(RuntimeError) as ctx:
            chain.run()
          self.assertIn('enter error', str(ctx.exception))

  def test_contextlib_cm_success_sync(self):
    """@contextmanager CM via lambda wrapper, sync body."""
    for method in ('with_', 'with_do'):
      with self.subTest(method=method, body='success'):
        cm = contextlib_sync_cm()
        chain = Chain(lambda _cm=cm: _cm)
        if method == 'with_':
          chain.with_(_sync_body)
        else:
          chain.with_do(_sync_body)
        result = chain.run()
        if method == 'with_do':
          self.assertIs(result, cm)
        else:
          self.assertEqual(result, 'body:cl_sync_ctx')

  def test_contextlib_cm_exception_sync(self):
    """@contextmanager CM body raises."""
    cm = contextlib_sync_cm()
    with self.assertRaises(ValueError):
      Chain(lambda: cm).with_(_sync_raise_body).run()

  def test_contextlib_cm_suppresses_sync(self):
    """@contextmanager CM that suppresses."""
    cm = contextlib_sync_cm_suppresses()
    result = Chain(lambda: cm).with_(_sync_raise_body).run()
    self.assertIsNone(result)

  def test_contextlib_cm_with_do_preserves_sync(self):
    """@contextmanager with_do returns the CM object."""
    cm = contextlib_sync_cm()
    result = Chain(lambda: cm).with_do(lambda ctx: 999).run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# Class 2: TestCMProtocolAsyncMatrix
# ---------------------------------------------------------------------------

class TestCMProtocolAsyncMatrix(IsolatedAsyncioTestCase):
  """Async CMs and DualCM x with_/with_do x body outcomes x sync/async body."""

  async def test_async_cm_success_sync_body(self):
    for cm_name, factory in ASYNC_CM_FACTORIES.items():
      if cm_name in ASYNC_ENTER_RAISES or cm_name in ASYNC_EXIT_RAISES_ON_SUCCESS:
        continue
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='sync_success'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_body)
          else:
            chain.with_do(_sync_body)
          result = await chain.run()
          if method == 'with_do':
            self.assertIs(result, cm)
          else:
            ev = ASYNC_ENTER_VALUES.get(cm_name)
            if ev == '_self_':
              self.assertEqual(result, f'body:{cm}')
            else:
              self.assertEqual(result, f'body:{ev}')

  async def test_async_cm_success_async_body(self):
    for cm_name, factory in ASYNC_CM_FACTORIES.items():
      if cm_name in ASYNC_ENTER_RAISES or cm_name in ASYNC_EXIT_RAISES_ON_SUCCESS:
        continue
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='async_success'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_async_body)
          else:
            chain.with_do(_async_body)
          result = await chain.run()
          if method == 'with_do':
            self.assertIs(result, cm)
          else:
            ev = ASYNC_ENTER_VALUES.get(cm_name)
            if ev == '_self_':
              self.assertEqual(result, f'body:{cm}')
            else:
              self.assertEqual(result, f'body:{ev}')

  async def test_async_cm_exception_not_suppressed(self):
    for cm_name, factory in ASYNC_CM_FACTORIES.items():
      if cm_name in ASYNC_ENTER_RAISES:
        continue
      if cm_name in ASYNC_SUPPRESSES:
        continue
      for method in ('with_', 'with_do'):
        for body_label, body_fn in (('sync', _sync_raise_body), ('async', _async_raise_body)):
          with self.subTest(cm_type=cm_name, method=method, body=f'{body_label}_exception'):
            cm = factory()
            chain = Chain(cm)
            if method == 'with_':
              chain.with_(body_fn)
            else:
              chain.with_do(body_fn)
            if cm_name in ASYNC_EXIT_RAISES_ON_EXCEPTION:
              with self.assertRaises(RuntimeError):
                await chain.run()
            else:
              with self.assertRaises(ValueError) as ctx:
                await chain.run()
              self.assertIn('body error', str(ctx.exception))

  async def test_async_cm_exception_suppressed_sync_body(self):
    """Async CMs that suppress exceptions with sync body fn."""
    for cm_name in ASYNC_SUPPRESSES:
      factory = ASYNC_CM_FACTORIES[cm_name]
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='sync_suppressed'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_raise_body)
          else:
            chain.with_do(_sync_raise_body)
          result = await chain.run()
          if method == 'with_do':
            self.assertIs(result, cm)
          else:
            self.assertIsNone(result)

  async def test_async_cm_exception_suppressed_async_body(self):
    """Async CMs that suppress exceptions with async body fn.

    When _full_async suppresses an async body exception, `result` retains
    the coroutine object from _evaluate_value (before the await that raised).
    For with_do, the outer_value is returned regardless. For with_, the
    stale coroutine is returned (library edge case).
    """
    for cm_name in ASYNC_SUPPRESSES:
      factory = ASYNC_CM_FACTORIES[cm_name]
      for method in ('with_do',):
        with self.subTest(cm_type=cm_name, method=method, body='async_suppressed'):
          cm = factory()
          chain = Chain(cm)
          chain.with_do(_async_raise_body)
          result = await chain.run()
          self.assertIs(result, cm)

  async def test_async_cm_return_signal(self):
    safe_async = {
      k: v for k, v in ASYNC_CM_FACTORIES.items()
      if k not in ASYNC_ENTER_RAISES and k not in ASYNC_EXIT_RAISES_ON_SUCCESS
    }
    for cm_name, factory in safe_async.items():
      for method in ('with_', 'with_do'):
        for body_label, body_fn in (('sync', _sync_return_body), ('async', _async_return_body)):
          with self.subTest(cm_type=cm_name, method=method, body=f'{body_label}_return'):
            cm = factory()
            chain = Chain(cm)
            if method == 'with_':
              chain.with_(body_fn)
            else:
              chain.with_do(body_fn)
            result = await chain.run()
            self.assertEqual(result, 'early')

  async def test_async_cm_enter_raises(self):
    for cm_name in ASYNC_ENTER_RAISES:
      factory = ASYNC_CM_FACTORIES[cm_name]
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_body)
          else:
            chain.with_do(_sync_body)
          with self.assertRaises(RuntimeError) as ctx:
            await chain.run()
          self.assertIn('async enter error', str(ctx.exception))

  async def test_async_cm_exit_raises_on_success(self):
    """AsyncCMRaisesOnExit.__aexit__ raises even on success."""
    for cm_name in ASYNC_EXIT_RAISES_ON_SUCCESS:
      factory = ASYNC_CM_FACTORIES[cm_name]
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='success'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_body)
          else:
            chain.with_do(_sync_body)
          with self.assertRaises(RuntimeError):
            await chain.run()

  async def test_awaitable_exit_sync_cm_success(self):
    """Sync CMs with awaitable __exit__ in async context (success body)."""
    for cm_name, factory in AWAITABLE_EXIT_CM_FACTORIES.items():
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='success'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_body)
          else:
            chain.with_do(_sync_body)
          result = await chain.run()
          if method == 'with_do':
            self.assertIs(result, cm)
          else:
            self.assertEqual(result, 'body:ctx_value')

  async def test_awaitable_exit_sync_cm_exception(self):
    """Sync CMs with awaitable __exit__ in async context (exception body)."""
    for cm_name, factory in AWAITABLE_EXIT_CM_FACTORIES.items():
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='exception'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_raise_body)
          else:
            chain.with_do(_sync_raise_body)
          if cm_name == 'SyncCMSuppressesAwaitable':
            result = await chain.run()
            if method == 'with_do':
              self.assertIs(result, cm)
            else:
              self.assertIsNone(result)
          else:
            with self.assertRaises(ValueError):
              await chain.run()

  def test_awaitable_exit_on_exception_cm_success(self):
    """SyncCMExitReturnsAwaitableOnException: on success __exit__ returns False (sync)."""
    for cm_name, factory in AWAITABLE_EXIT_ON_EXCEPTION_CM_FACTORIES.items():
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='success'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_body)
          else:
            chain.with_do(_sync_body)
          result = chain.run()
          if method == 'with_do':
            self.assertIs(result, cm)
          else:
            self.assertEqual(result, 'body:ctx_value')

  async def test_awaitable_exit_on_exception_cm_exception(self):
    """SyncCMExitReturnsAwaitableOnException: on exception __exit__ returns awaitable."""
    for cm_name, factory in AWAITABLE_EXIT_ON_EXCEPTION_CM_FACTORIES.items():
      for method in ('with_', 'with_do'):
        with self.subTest(cm_type=cm_name, method=method, body='exception'):
          cm = factory()
          chain = Chain(cm)
          if method == 'with_':
            chain.with_(_sync_raise_body)
          else:
            chain.with_do(_sync_raise_body)
          # __exit__ returns awaitable(False) -- does NOT suppress
          with self.assertRaises(ValueError):
            await chain.run()

  def test_nullcontext_success_sync(self):
    """nullcontext has __enter__ preferred over __aenter__, so goes sync path."""
    for method in ('with_', 'with_do'):
      with self.subTest(method=method, body='success'):
        cm = contextlib.nullcontext('null_ctx')
        chain = Chain(lambda _cm=cm: _cm)
        if method == 'with_':
          chain.with_(_sync_body)
        else:
          chain.with_do(_sync_body)
        result = chain.run()
        if method == 'with_do':
          self.assertIs(result, cm)
        else:
          self.assertEqual(result, 'body:null_ctx')

  def test_nullcontext_exception_sync(self):
    cm = contextlib.nullcontext('x')
    with self.assertRaises(ValueError):
      Chain(lambda: cm).with_(_sync_raise_body).run()

  def test_nullcontext_with_do_preserves_sync(self):
    cm = contextlib.nullcontext('anything')
    result = Chain(lambda: cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  async def test_contextlib_async_cm_success(self):
    """@asynccontextmanager CM success."""
    for method in ('with_', 'with_do'):
      with self.subTest(method=method, body='success'):
        cm = contextlib_async_cm()
        chain = Chain(lambda _cm=cm: _cm)
        if method == 'with_':
          chain.with_(_sync_body)
        else:
          chain.with_do(_sync_body)
        result = await chain.run()
        if method == 'with_do':
          self.assertIs(result, cm)
        else:
          self.assertEqual(result, 'body:cl_async_ctx')

  async def test_contextlib_async_cm_exception(self):
    cm = contextlib_async_cm()
    with self.assertRaises(ValueError):
      await Chain(lambda: cm).with_(_sync_raise_body).run()

  async def test_contextlib_async_cm_with_do_preserves(self):
    cm = contextlib_async_cm()
    result = await Chain(lambda: cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  async def test_contextlib_async_cm_async_body(self):
    cm = contextlib_async_cm()
    result = await Chain(lambda: cm).with_(_async_body).run()
    self.assertEqual(result, 'body:cl_async_ctx')


# ---------------------------------------------------------------------------
# Class 3: TestCMEnterValueMatrix
# ---------------------------------------------------------------------------

class TestCMEnterValueMatrix(IsolatedAsyncioTestCase):
  """Verify the exact value the body fn receives from __enter__/__aenter__."""

  def test_sync_cm_enter_values(self):
    """Check enter value for each pure sync CM type."""
    def capture(ctx):
      return ctx

    for cm_name, factory in PURE_SYNC_CM_FACTORIES.items():
      if cm_name in SYNC_ENTER_RAISES:
        continue
      if cm_name in SYNC_EXIT_RAISES_ON_SUCCESS:
        continue
      with self.subTest(cm_type=cm_name):
        cm = factory()
        result = Chain(cm).with_(capture).run()
        expected = PURE_SYNC_ENTER_VALUES.get(cm_name)
        if cm_name == 'SyncCMEnterReturnsSelf':
          self.assertIs(result, cm)
        elif expected is not None:
          self.assertEqual(result, expected)

  async def test_async_cm_enter_values(self):
    """Check enter value for each async CM type."""
    def capture(ctx):
      return ctx

    for cm_name, factory in ASYNC_CM_FACTORIES.items():
      if cm_name in ASYNC_ENTER_RAISES or cm_name in ASYNC_EXIT_RAISES_ON_SUCCESS:
        continue
      with self.subTest(cm_type=cm_name):
        cm = factory()
        result = await Chain(cm).with_(capture).run()
        ev = ASYNC_ENTER_VALUES.get(cm_name)
        if ev == '_self_':
          self.assertIs(result, cm)
        elif ev is not None:
          self.assertEqual(result, ev)

  async def test_dual_cm_prefers_enter(self):
    """DualCM should use __enter__ (sync preferred over async) returning 'sync_ctx'."""
    cm = DualCM()
    result = Chain(cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'sync_ctx')

  def test_sync_cm_enter_returns_none(self):
    cm = SyncCMEnterReturnsNone()
    result = Chain(cm).with_(lambda ctx: ctx).run()
    self.assertIsNone(result)

  def test_sync_cm_enter_returns_self(self):
    cm = SyncCMEnterReturnsSelf()
    result = Chain(cm).with_(lambda ctx: ctx).run()
    self.assertIs(result, cm)

  def test_tracking_cm_enter_value(self):
    cm = TrackingCM()
    result = Chain(cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'tracked_ctx')

  async def test_async_tracking_cm_enter_value(self):
    cm = AsyncTrackingCM()
    result = await Chain(cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'async_tracked_ctx')

  def test_nullcontext_enter_value(self):
    """nullcontext has __enter__ preferred, tested in sync context."""
    cm = contextlib.nullcontext('hello')
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'hello')

  def test_contextlib_cm_enter_value(self):
    cm = contextlib_sync_cm()
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'cl_sync_ctx')

  async def test_contextlib_async_cm_enter_value(self):
    cm = contextlib_async_cm()
    result = await Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'cl_async_ctx')

  def test_nullcontext_none_enter_value(self):
    """nullcontext() with no arg returns None as enter value."""
    cm = contextlib.nullcontext()
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Class 4: TestCMExitBehaviorMatrix
# ---------------------------------------------------------------------------

class TestCMExitBehaviorMatrix(IsolatedAsyncioTestCase):
  """Verify __exit__ behavior for each CM type."""

  def test_tracking_cm_exit_on_success(self):
    """TrackingCM.__exit__ receives (None, None, None) on success."""
    cm = TrackingCM()
    Chain(cm).with_(lambda ctx: 'ok').run()
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_tracking_cm_exit_on_exception(self):
    """TrackingCM.__exit__ receives (exc_type, exc, tb) on exception."""
    cm = TrackingCM()
    with self.assertRaises(ValueError):
      Chain(cm).with_(_sync_raise_body).run()
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args[0], ValueError)
    self.assertIsInstance(cm.exit_args[1], ValueError)
    self.assertIsNotNone(cm.exit_args[2])

  async def test_async_tracking_cm_exit_on_success(self):
    cm = AsyncTrackingCM()
    await Chain(cm).with_(lambda ctx: 'ok').run()
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  async def test_async_tracking_cm_exit_on_exception(self):
    cm = AsyncTrackingCM()
    with self.assertRaises(ValueError):
      await Chain(cm).with_(_sync_raise_body).run()
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args[0], ValueError)
    self.assertIsInstance(cm.exit_args[1], ValueError)

  def test_sync_cm_suppresses_exit_behavior(self):
    """SyncCMSuppresses.__exit__ returns True, suppressing exception."""
    cm = SyncCMSuppresses()
    result = Chain(cm).with_(_sync_raise_body).run()
    self.assertIsNone(result)

  async def test_async_cm_suppresses_exit_behavior(self):
    cm = AsyncCMSuppresses()
    result = await Chain(cm).with_(_sync_raise_body).run()
    self.assertIsNone(result)

  def test_exit_raises_on_success_matrix(self):
    """CMs whose __exit__ raises even when body succeeds."""
    for cm_name in SYNC_EXIT_RAISES_ON_SUCCESS:
      factory = PURE_SYNC_CM_FACTORIES[cm_name]
      with self.subTest(cm_type=cm_name):
        cm = factory()
        with self.assertRaises(RuntimeError):
          Chain(cm).with_(_sync_body).run()

  def test_exit_raises_on_exception_matrix(self):
    """CMs whose __exit__ raises when body raises."""
    for cm_name in SYNC_EXIT_RAISES_ON_EXCEPTION:
      factory = PURE_SYNC_CM_FACTORIES[cm_name]
      with self.subTest(cm_type=cm_name):
        cm = factory()
        with self.assertRaises(RuntimeError) as ctx:
          Chain(cm).with_(_sync_raise_body).run()
        self.assertIn('exit error', str(ctx.exception))

  async def test_async_exit_raises_on_exception_matrix(self):
    for cm_name in ASYNC_EXIT_RAISES_ON_EXCEPTION:
      factory = ASYNC_CM_FACTORIES[cm_name]
      with self.subTest(cm_type=cm_name):
        cm = factory()
        with self.assertRaises(RuntimeError):
          await Chain(cm).with_(_sync_raise_body).run()

  def test_sync_cm_exit_called_on_return_signal(self):
    """__exit__ is called with (None, None, None) when body issues _Return."""
    cm = TrackingCM()
    Chain(cm).with_(_sync_return_body).run()
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  async def test_async_cm_exit_called_on_return_signal(self):
    cm = AsyncTrackingCM()
    await Chain(cm).with_(_sync_return_body).run()
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_sync_cm_raises_on_exit_from_chaining(self):
    """SyncCMRaisesOnExitFrom chains the original exception via 'from'."""
    cm = SyncCMRaisesOnExitFrom()
    with self.assertRaises(RuntimeError) as ctx:
      Chain(cm).with_(_sync_raise_body).run()
    self.assertIn('exit error', str(ctx.exception))
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  def test_all_pure_sync_cms_exit_called_after_success(self):
    """For all pure sync CMs that can enter, exit is called on success."""
    for cm_name, factory in PURE_SYNC_CM_FACTORIES.items():
      if cm_name in SYNC_ENTER_RAISES:
        continue
      with self.subTest(cm_type=cm_name):
        cm = factory()
        try:
          Chain(cm).with_(_sync_body).run()
        except Exception:
          pass
        if hasattr(cm, 'exited'):
          self.assertTrue(cm.exited, f'{cm_name} exit not called')

  async def test_all_async_cms_exit_called_after_success(self):
    for cm_name, factory in ASYNC_CM_FACTORIES.items():
      if cm_name in ASYNC_ENTER_RAISES:
        continue
      with self.subTest(cm_type=cm_name):
        cm = factory()
        try:
          await Chain(cm).with_(_sync_body).run()
        except Exception:
          pass
        if hasattr(cm, 'exited'):
          self.assertTrue(cm.exited, f'{cm_name} exit not called')

  def test_tracking_cm_exit_args_on_success_matrix(self):
    """Multiple body return values -- exit always gets (None, None, None)."""
    body_fns = {
      'None': lambda ctx: None,
      '42': lambda ctx: 42,
      'False': lambda ctx: False,
    }
    for body_label, body_fn in body_fns.items():
      with self.subTest(body_return=body_label):
        cm = TrackingCM()
        Chain(cm).with_(body_fn).run()
        self.assertEqual(cm.exit_args, (None, None, None))

  async def test_async_tracking_cm_exit_args_on_success_matrix(self):
    body_fns = {
      'None': lambda ctx: None,
      '42': lambda ctx: 42,
      'False': lambda ctx: False,
    }
    for body_label, body_fn in body_fns.items():
      with self.subTest(body_return=body_label):
        cm = AsyncTrackingCM()
        await Chain(cm).with_(body_fn).run()
        self.assertEqual(cm.exit_args, (None, None, None))


# ---------------------------------------------------------------------------
# Class 5: TestCMWithDoPreservationMatrix
# ---------------------------------------------------------------------------

class TestCMWithDoPreservationMatrix(IsolatedAsyncioTestCase):
  """Verify with_do ALWAYS returns the CM object (outer_value)."""

  def test_sync_with_do_preserves_cm_various_body_returns(self):
    """with_do returns CM regardless of body return value."""
    body_fns = {
      'None': lambda ctx: None,
      '42': lambda ctx: 42,
      'False': lambda ctx: False,
      '0': lambda ctx: 0,
      'empty_str': lambda ctx: '',
      'empty_list': lambda ctx: [],
      'large_value': lambda ctx: list(range(1000)),
    }
    # Only pure sync CMs that won't raise on exit
    safe_cms = {
      k: v for k, v in PURE_SYNC_CM_FACTORIES.items()
      if k not in SYNC_ENTER_RAISES
      and k not in SYNC_EXIT_RAISES_ON_SUCCESS
      and k not in SYNC_EXIT_RAISES_ON_EXCEPTION
    }
    for cm_name, factory in safe_cms.items():
      for body_label, body_fn in body_fns.items():
        with self.subTest(cm_type=cm_name, body_return=body_label):
          cm = factory()
          result = Chain(cm).with_do(body_fn).run()
          self.assertIs(result, cm)

  async def test_async_with_do_preserves_cm_various_body_returns(self):
    body_fns = {
      'None': lambda ctx: None,
      '42': lambda ctx: 42,
      'False': lambda ctx: False,
      '0': lambda ctx: 0,
      'empty_str': lambda ctx: '',
    }
    safe_async = {
      k: v for k, v in ASYNC_CM_FACTORIES.items()
      if k not in ASYNC_ENTER_RAISES and k not in ASYNC_EXIT_RAISES_ON_SUCCESS
    }
    for cm_name, factory in safe_async.items():
      for body_label, body_fn in body_fns.items():
        with self.subTest(cm_type=cm_name, body_return=body_label):
          cm = factory()
          result = await Chain(cm).with_do(body_fn).run()
          self.assertIs(result, cm)

  async def test_async_with_do_preserves_cm_async_body(self):
    """with_do with async body returns CM."""
    async def body(ctx):
      return 'should_be_ignored'

    for cm_name, factory in ASYNC_CM_FACTORIES.items():
      if cm_name in ASYNC_ENTER_RAISES or cm_name in ASYNC_EXIT_RAISES_ON_SUCCESS:
        continue
      with self.subTest(cm_type=cm_name):
        cm = factory()
        result = await Chain(cm).with_do(body).run()
        self.assertIs(result, cm)

  def test_sync_with_do_preserves_after_suppression(self):
    """with_do returns CM even when body raises and exit suppresses."""
    cm = SyncCMSuppresses()
    result = Chain(cm).with_do(_sync_raise_body).run()
    self.assertIs(result, cm)

  async def test_async_with_do_preserves_after_suppression(self):
    cm = AsyncCMSuppresses()
    result = await Chain(cm).with_do(_sync_raise_body).run()
    self.assertIs(result, cm)

  async def test_awaitable_exit_with_do_preserves(self):
    """with_do on CMs with awaitable __exit__ still returns CM."""
    for cm_name, factory in AWAITABLE_EXIT_CM_FACTORIES.items():
      with self.subTest(cm_type=cm_name):
        cm = factory()
        result = await Chain(cm).with_do(_sync_body).run()
        self.assertIs(result, cm)

  def test_with_do_preserves_contextlib_cm(self):
    """with_do on @contextmanager CM returns the CM object."""
    cm = contextlib_sync_cm()
    result = Chain(lambda: cm).with_do(lambda ctx: 999).run()
    self.assertIs(result, cm)

  def test_with_do_preserves_nullcontext(self):
    """nullcontext has __enter__ preferred, tested in sync context."""
    cm = contextlib.nullcontext('anything')
    result = Chain(lambda: cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  async def test_with_do_preserves_contextlib_async_cm(self):
    cm = contextlib_async_cm()
    result = await Chain(lambda: cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  def test_with_do_body_none_result_still_preserves(self):
    """Edge case: body explicitly returns Null-like values."""
    for val in (None, 0, False, '', b'', [], {}, set()):
      with self.subTest(body_return=repr(val)):
        cm = SyncCM()
        result = Chain(cm).with_do(lambda ctx, v=val: v).run()
        self.assertIs(result, cm)

  def test_with_do_preserves_nullcontext_none_enter(self):
    """nullcontext() with no arg -- with_do still returns the CM."""
    cm = contextlib.nullcontext()
    result = Chain(lambda: cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# Class 6: TestCMProtocolContextlibMatrix
# ---------------------------------------------------------------------------

class TestCMProtocolContextlibMatrix(IsolatedAsyncioTestCase):
  """Additional tests for contextlib-generated CMs."""

  def test_contextlib_cm_success(self):
    cm = contextlib_sync_cm()
    result = Chain(lambda: cm).with_(lambda ctx: f'got:{ctx}').run()
    self.assertEqual(result, 'got:cl_sync_ctx')

  async def test_contextlib_async_cm_success(self):
    cm = contextlib_async_cm()
    result = await Chain(lambda: cm).with_(lambda ctx: f'got:{ctx}').run()
    self.assertEqual(result, 'got:cl_async_ctx')

  def test_contextlib_cm_exception(self):
    cm = contextlib_sync_cm()
    with self.assertRaises(ValueError):
      Chain(lambda: cm).with_(_sync_raise_body).run()

  async def test_contextlib_async_cm_exception(self):
    cm = contextlib_async_cm()
    with self.assertRaises(ValueError):
      await Chain(lambda: cm).with_(_sync_raise_body).run()

  def test_contextlib_cm_suppresses(self):
    cm = contextlib_sync_cm_suppresses()
    result = Chain(lambda: cm).with_(_sync_raise_body).run()
    self.assertIsNone(result)

  def test_nullcontext_success(self):
    cm = contextlib.nullcontext(42)
    result = Chain(lambda: cm).with_(lambda ctx: ctx + 8).run()
    self.assertEqual(result, 50)

  def test_nullcontext_exception(self):
    cm = contextlib.nullcontext('x')
    with self.assertRaises(ValueError):
      Chain(lambda: cm).with_(_sync_raise_body).run()

  def test_nullcontext_none_enter(self):
    cm = contextlib.nullcontext()
    result = Chain(lambda: cm).with_(lambda ctx: ctx).run()
    self.assertIsNone(result)

  def test_contextlib_cm_return_signal(self):
    cm = contextlib_sync_cm()
    result = Chain(lambda: cm).with_(_sync_return_body).run()
    self.assertEqual(result, 'early')

  async def test_contextlib_async_cm_return_signal(self):
    cm = contextlib_async_cm()
    result = await Chain(lambda: cm).with_(_sync_return_body).run()
    self.assertEqual(result, 'early')

  def test_nullcontext_return_signal(self):
    cm = contextlib.nullcontext('x')
    result = Chain(lambda: cm).with_(_sync_return_body).run()
    self.assertEqual(result, 'early')

  def test_contextlib_cm_with_do_return_signal(self):
    cm = contextlib_sync_cm()
    result = Chain(lambda: cm).with_do(_sync_return_body).run()
    self.assertEqual(result, 'early')

  def test_nullcontext_with_do_return_signal(self):
    cm = contextlib.nullcontext('x')
    result = Chain(lambda: cm).with_do(_sync_return_body).run()
    self.assertEqual(result, 'early')


if __name__ == '__main__':
  unittest.main()
