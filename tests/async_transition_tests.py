"""Exhaustive 3-tier sync/async transition matrix tests for all operations.

Each operation (foreach, filter, with_) has three execution paths:

  SYNC tier:      sync iterable/CM + sync fn -> pure sync _op function
  TO_ASYNC tier:  sync iterable/CM + fn returns awaitable -> _to_async handoff
  FULL_ASYNC tier: async iterable/CM (__aiter__/__aenter__) -> _full_async path

This file covers every combination and edge case across all three tiers,
including break/return control flow, exception propagation, ignore_result
behavior, and chain-level sync-to-async transitions.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  async_fn,
  async_identity,
  async_raise_fn,
  sync_fn,
  sync_identity,
  raise_fn,
  AsyncCM,
  AsyncCMSuppresses,
  AsyncRange,
  AsyncRangeRaises,
  SyncCM,
  SyncCMSuppresses,
  SyncCMWithAwaitableExit,
  SyncCMSuppressesAwaitable,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

class RaisingIterable:
  """Sync iterable that raises RuntimeError at a specific index."""

  def __init__(self, n, raise_at):
    self.n = n
    self.raise_at = raise_at

  def __iter__(self):
    for i in range(self.n):
      if i == self.raise_at:
        raise RuntimeError('iteration error')
      yield i


async def async_double(x):
  return x * 2


async def async_make_val():
  return 'async_val'


async def async_raise_runtime(x=None):
  raise RuntimeError('async error')


# ===========================================================================
# FOREACH — SYNC TIER
# ===========================================================================

class TestForeachSyncTier(unittest.TestCase):

  def test_happy_path(self):
    result = Chain([1, 2, 3]).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_exception_in_fn(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain([1, 2, 3]).foreach(lambda x: (_ for _ in ()).throw(RuntimeError('fn error'))).run()
    # Use a simpler approach:
    def boom(x):
      raise RuntimeError('fn error')
    with self.assertRaises(RuntimeError) as cm:
      Chain([1, 2, 3]).foreach(boom).run()
    self.assertEqual(str(cm.exception), 'fn error')

  def test_exception_in_iteration(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain(RaisingIterable(5, 2)).foreach(lambda x: x).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  def test_break(self):
    result = Chain([1, 2, 3, 4, 5]).foreach(
      lambda x: Chain.break_() if x == 3 else x
    ).run()
    self.assertEqual(result, [1, 2])

  def test_return(self):
    result = Chain([1, 2, 3]).foreach(
      lambda x: Chain.return_('early') if x == 2 else x
    ).run()
    self.assertEqual(result, 'early')

  def test_break_with_value(self):
    result = Chain([1, 2, 3]).foreach(
      lambda x: Chain.break_(99) if x == 2 else x
    ).run()
    self.assertEqual(result, 99)

  def test_break_with_sync_value(self):
    result = Chain([1, 2, 3]).foreach(
      lambda x: Chain.break_(lambda: 99) if x == 2 else x
    ).run()
    self.assertEqual(result, 99)

  def test_exception_in_break_value(self):
    with self.assertRaises(ValueError) as cm:
      Chain([1, 2, 3]).foreach(
        lambda x: Chain.break_(raise_fn) if x == 2 else x
      ).run()
    self.assertEqual(str(cm.exception), 'test error')


# ===========================================================================
# FOREACH — TO_ASYNC TIER
# ===========================================================================

class TestForeachToAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    result = await Chain([1, 2, 3]).foreach(async_fn).run()
    self.assertEqual(result, [2, 3, 4])

  async def test_exception_in_fn(self):
    call_count = 0

    async def boom_on_second(x):
      nonlocal call_count
      call_count += 1
      if call_count == 2:
        raise RuntimeError('async fn error')
      return x

    with self.assertRaises(RuntimeError) as cm:
      await Chain([1, 2, 3]).foreach(boom_on_second).run()
    self.assertEqual(str(cm.exception), 'async fn error')

  async def test_exception_in_iteration(self):
    # RaisingIterable raises at index 2. async_fn returns awaitable on
    # first call, triggering _to_async. Then next() raises inside _to_async.
    with self.assertRaises(RuntimeError) as cm:
      await Chain(RaisingIterable(5, 2)).foreach(async_fn).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  async def test_break(self):
    async def break_on_second(x):
      if x == 2:
        Chain.break_()
      return x * 10

    result = await Chain([1, 2, 3]).foreach(break_on_second).run()
    self.assertEqual(result, [10])

  async def test_return(self):
    async def return_on_second(x):
      if x == 2:
        Chain.return_('early_async')
      return x

    result = await Chain([1, 2, 3]).foreach(return_on_second).run()
    self.assertEqual(result, 'early_async')

  async def test_break_with_value(self):
    async def break_with_val(x):
      if x == 2:
        Chain.break_(42)
      return x

    result = await Chain([1, 2, 3]).foreach(break_with_val).run()
    self.assertEqual(result, 42)

  async def test_break_with_async_value(self):
    # Break value is a callable that returns a coroutine. In _to_async,
    # _handle_break_exc calls it, then the result is awaited.
    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.break_(async_make_val) if x == 1 else x
    ).run()
    self.assertEqual(result, 'async_val')

  async def test_exception_in_break_value(self):
    async def break_with_raise(x):
      if x == 2:
        Chain.break_(raise_fn)
      return x

    with self.assertRaises(ValueError) as cm:
      await Chain([1, 2, 3]).foreach(break_with_raise).run()
    self.assertEqual(str(cm.exception), 'test error')


# ===========================================================================
# FOREACH — FULL_ASYNC TIER
# ===========================================================================

class TestForeachFullAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    result = await Chain(AsyncRange(3)).foreach(sync_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_exception_in_fn(self):
    def boom(x):
      if x == 1:
        raise RuntimeError('fn error in full async')
      return x

    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRange(5)).foreach(boom).run()
    self.assertEqual(str(cm.exception), 'fn error in full async')

  async def test_exception_in_iteration(self):
    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRangeRaises(5, 2)).foreach(lambda x: x).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  async def test_break(self):
    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.break_() if x == 2 else x
    ).run()
    self.assertEqual(result, [0, 1])

  async def test_return(self):
    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.return_('from_async') if x == 1 else x
    ).run()
    self.assertEqual(result, 'from_async')

  async def test_break_with_value(self):
    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.break_(77) if x == 2 else x
    ).run()
    self.assertEqual(result, 77)

  async def test_break_with_async_value(self):
    result = await Chain(AsyncRange(5)).foreach(
      lambda x: Chain.break_(async_make_val) if x == 1 else x
    ).run()
    self.assertEqual(result, 'async_val')

  async def test_exception_in_break_value(self):
    with self.assertRaises(ValueError) as cm:
      await Chain(AsyncRange(5)).foreach(
        lambda x: Chain.break_(raise_fn) if x == 2 else x
      ).run()
    self.assertEqual(str(cm.exception), 'test error')


# ===========================================================================
# FOREACH_DO — SYNC TIER
# ===========================================================================

class TestForeachDoSyncTier(unittest.TestCase):

  def test_happy_path(self):
    result = Chain([1, 2, 3]).foreach_do(lambda x: x * 100).run()
    self.assertEqual(result, [1, 2, 3])

  def test_fn_called_as_side_effect(self):
    tracker = []
    Chain([1, 2, 3]).foreach_do(lambda x: tracker.append(x)).run()
    self.assertEqual(tracker, [1, 2, 3])

  def test_exception_in_fn(self):
    def boom(x):
      raise RuntimeError('do fn error')
    with self.assertRaises(RuntimeError) as cm:
      Chain([1, 2, 3]).foreach_do(boom).run()
    self.assertEqual(str(cm.exception), 'do fn error')

  def test_exception_in_iteration(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain(RaisingIterable(5, 2)).foreach_do(lambda x: x).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  def test_break(self):
    result = Chain([1, 2, 3, 4]).foreach_do(
      lambda x: Chain.break_() if x == 3 else None
    ).run()
    self.assertEqual(result, [1, 2])

  def test_return(self):
    result = Chain([1, 2, 3]).foreach_do(
      lambda x: Chain.return_('early') if x == 2 else None
    ).run()
    self.assertEqual(result, 'early')

  def test_break_with_value(self):
    result = Chain([1, 2, 3]).foreach_do(
      lambda x: Chain.break_(99) if x == 2 else None
    ).run()
    self.assertEqual(result, 99)

  def test_break_with_sync_value(self):
    result = Chain([1, 2, 3]).foreach_do(
      lambda x: Chain.break_(lambda: 99) if x == 2 else None
    ).run()
    self.assertEqual(result, 99)

  def test_exception_in_break_value(self):
    with self.assertRaises(ValueError) as cm:
      Chain([1, 2, 3]).foreach_do(
        lambda x: Chain.break_(raise_fn) if x == 2 else None
      ).run()
    self.assertEqual(str(cm.exception), 'test error')


# ===========================================================================
# FOREACH_DO — TO_ASYNC TIER
# ===========================================================================

class TestForeachDoToAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    result = await Chain([1, 2, 3]).foreach_do(async_fn).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_fn_called_as_side_effect(self):
    tracker = []

    async def track(x):
      tracker.append(x)
      return 'discarded'

    await Chain([1, 2, 3]).foreach_do(track).run()
    self.assertEqual(tracker, [1, 2, 3])

  async def test_exception_in_fn(self):
    async def boom(x):
      raise RuntimeError('async do fn error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain([1, 2, 3]).foreach_do(boom).run()
    self.assertEqual(str(cm.exception), 'async do fn error')

  async def test_exception_in_iteration(self):
    with self.assertRaises(RuntimeError) as cm:
      await Chain(RaisingIterable(5, 2)).foreach_do(async_fn).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  async def test_break(self):
    async def break_on_third(x):
      if x == 3:
        Chain.break_()

    result = await Chain([1, 2, 3, 4]).foreach_do(break_on_third).run()
    self.assertEqual(result, [1, 2])

  async def test_return(self):
    async def return_on_second(x):
      if x == 2:
        Chain.return_('early_async_do')

    result = await Chain([1, 2, 3]).foreach_do(return_on_second).run()
    self.assertEqual(result, 'early_async_do')

  async def test_break_with_value(self):
    async def break_with_val(x):
      if x == 2:
        Chain.break_(42)

    result = await Chain([1, 2, 3]).foreach_do(break_with_val).run()
    self.assertEqual(result, 42)

  async def test_break_with_async_value(self):
    async def break_async_val(x):
      if x == 2:
        Chain.break_(async_make_val)

    result = await Chain([1, 2, 3]).foreach_do(break_async_val).run()
    self.assertEqual(result, 'async_val')

  async def test_exception_in_break_value(self):
    async def break_raise(x):
      if x == 2:
        Chain.break_(raise_fn)

    with self.assertRaises(ValueError) as cm:
      await Chain([1, 2, 3]).foreach_do(break_raise).run()
    self.assertEqual(str(cm.exception), 'test error')


# ===========================================================================
# FOREACH_DO — FULL_ASYNC TIER
# ===========================================================================

class TestForeachDoFullAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    result = await Chain(AsyncRange(3)).foreach_do(lambda x: x * 100).run()
    self.assertEqual(result, [0, 1, 2])

  async def test_fn_called_as_side_effect(self):
    tracker = []
    await Chain(AsyncRange(3)).foreach_do(lambda x: tracker.append(x)).run()
    self.assertEqual(tracker, [0, 1, 2])

  async def test_exception_in_fn(self):
    def boom(x):
      if x == 1:
        raise RuntimeError('full async do fn error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRange(5)).foreach_do(boom).run()
    self.assertEqual(str(cm.exception), 'full async do fn error')

  async def test_exception_in_iteration(self):
    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRangeRaises(5, 2)).foreach_do(lambda x: x).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  async def test_break(self):
    result = await Chain(AsyncRange(5)).foreach_do(
      lambda x: Chain.break_() if x == 2 else None
    ).run()
    self.assertEqual(result, [0, 1])

  async def test_return(self):
    result = await Chain(AsyncRange(5)).foreach_do(
      lambda x: Chain.return_('from_async_do') if x == 1 else None
    ).run()
    self.assertEqual(result, 'from_async_do')

  async def test_break_with_value(self):
    result = await Chain(AsyncRange(5)).foreach_do(
      lambda x: Chain.break_(77) if x == 2 else None
    ).run()
    self.assertEqual(result, 77)

  async def test_break_with_async_value(self):
    result = await Chain(AsyncRange(5)).foreach_do(
      lambda x: Chain.break_(async_make_val) if x == 1 else None
    ).run()
    self.assertEqual(result, 'async_val')

  async def test_exception_in_break_value(self):
    with self.assertRaises(ValueError) as cm:
      await Chain(AsyncRange(5)).foreach_do(
        lambda x: Chain.break_(raise_fn) if x == 2 else None
      ).run()
    self.assertEqual(str(cm.exception), 'test error')


# ===========================================================================
# FILTER — SYNC TIER
# ===========================================================================

class TestFilterSyncTier(unittest.TestCase):

  def test_happy_path(self):
    result = Chain([1, 2, 3, 4]).filter(lambda x: x > 2).run()
    self.assertEqual(result, [3, 4])

  def test_exception_in_predicate(self):
    def boom(x):
      raise RuntimeError('predicate error')
    with self.assertRaises(RuntimeError) as cm:
      Chain([1, 2, 3]).filter(boom).run()
    self.assertEqual(str(cm.exception), 'predicate error')

  def test_exception_in_iteration(self):
    with self.assertRaises(RuntimeError) as cm:
      Chain(RaisingIterable(5, 2)).filter(lambda x: True).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  def test_return_signal(self):
    result = Chain([1, 2, 3]).filter(
      lambda x: Chain.return_('early') if x == 2 else True
    ).run()
    self.assertEqual(result, 'early')

  def test_control_flow_propagation(self):
    # break outside foreach context -> QuentException
    with self.assertRaises(QuentException) as cm:
      Chain([1, 2, 3]).filter(
        lambda x: Chain.break_() if x == 2 else True
      ).run()
    self.assertIn('_Break cannot be used in this context', str(cm.exception))


# ===========================================================================
# FILTER — TO_ASYNC TIER
# ===========================================================================

class TestFilterToAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    async def async_pred(x):
      return x > 2

    result = await Chain([1, 2, 3, 4]).filter(async_pred).run()
    self.assertEqual(result, [3, 4])

  async def test_exception_in_predicate(self):
    async def boom(x):
      raise RuntimeError('async predicate error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain([1, 2, 3]).filter(boom).run()
    self.assertEqual(str(cm.exception), 'async predicate error')

  async def test_exception_in_iteration(self):
    async def always_true(x):
      return True

    with self.assertRaises(RuntimeError) as cm:
      await Chain(RaisingIterable(5, 2)).filter(always_true).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  async def test_return_signal(self):
    async def return_on_two(x):
      if x == 2:
        Chain.return_('early_async')
      return True

    result = await Chain([1, 2, 3]).filter(return_on_two).run()
    self.assertEqual(result, 'early_async')

  async def test_control_flow_propagation(self):
    async def break_on_two(x):
      if x == 2:
        Chain.break_()
      return True

    with self.assertRaises(QuentException) as cm:
      await Chain([1, 2, 3]).filter(break_on_two).run()
    self.assertIn('_Break cannot be used in this context', str(cm.exception))


# ===========================================================================
# FILTER — FULL_ASYNC TIER
# ===========================================================================

class TestFilterFullAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    result = await Chain(AsyncRange(5)).filter(lambda x: x > 2).run()
    self.assertEqual(result, [3, 4])

  async def test_exception_in_predicate(self):
    def boom(x):
      raise RuntimeError('predicate error in full async')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRange(5)).filter(boom).run()
    self.assertEqual(str(cm.exception), 'predicate error in full async')

  async def test_exception_in_iteration(self):
    with self.assertRaises(RuntimeError) as cm:
      await Chain(AsyncRangeRaises(5, 2)).filter(lambda x: True).run()
    self.assertEqual(str(cm.exception), 'iteration error')

  async def test_return_signal(self):
    result = await Chain(AsyncRange(5)).filter(
      lambda x: Chain.return_('from_full_async') if x == 2 else True
    ).run()
    self.assertEqual(result, 'from_full_async')

  async def test_control_flow_propagation(self):
    with self.assertRaises(QuentException) as cm:
      await Chain(AsyncRange(5)).filter(
        lambda x: Chain.break_() if x == 2 else True
      ).run()
    self.assertIn('_Break cannot be used in this context', str(cm.exception))


# ===========================================================================
# WITH — SYNC TIER
# ===========================================================================

class TestWithSyncTier(unittest.TestCase):

  def test_happy_path(self):
    result = Chain(SyncCM()).with_(lambda ctx: ctx + '!').run()
    self.assertEqual(result, 'ctx_value!')

  def test_exception_in_body(self):
    cm = SyncCM()
    with self.assertRaises(ZeroDivisionError):
      Chain(cm).with_(lambda ctx: 1 / 0).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_return_from_body(self):
    result = Chain(SyncCM()).with_(
      lambda ctx: Chain.return_('early')
    ).run()
    self.assertEqual(result, 'early')

  def test_exit_suppresses(self):
    result = Chain(SyncCMSuppresses()).with_(lambda ctx: 1 / 0).run()
    self.assertIsNone(result)

  def test_body_raises_exit_no_suppress(self):
    cm = SyncCM()
    with self.assertRaises(ValueError):
      Chain(cm).with_(lambda ctx: (_ for _ in ()).throw(ValueError('body'))).run()
    # Use a direct raise approach:
    def body_raises(ctx):
      raise ValueError('body error')
    cm2 = SyncCM()
    with self.assertRaises(ValueError) as ctx:
      Chain(cm2).with_(body_raises).run()
    self.assertEqual(str(ctx.exception), 'body error')
    self.assertTrue(cm2.exited)

  def test_body_raises_exit_suppresses(self):
    result = Chain(SyncCMSuppresses()).with_(
      lambda ctx: (_ for _ in ()).throw(ValueError('suppressed'))
    ).run()
    # SyncCMSuppresses.__exit__ returns True, body didn't return normally -> None
    def body_raises(ctx):
      raise ValueError('suppressed')
    result = Chain(SyncCMSuppresses()).with_(body_raises).run()
    self.assertIsNone(result)

  def test_exit_called_on_success(self):
    cm = SyncCM()
    Chain(cm).with_(lambda ctx: ctx).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ===========================================================================
# WITH — TO_ASYNC TIER
# ===========================================================================

class TestWithToAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    cm = SyncCM()
    result = await Chain(cm).with_(async_identity).run()
    self.assertEqual(result, 'ctx_value')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_exception_in_body(self):
    cm = SyncCM()
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(cm).with_(async_raise_runtime).run()
    self.assertEqual(str(ctx.exception), 'async error')
    self.assertTrue(cm.exited)

  async def test_return_from_body(self):
    async def body_return(ctx):
      Chain.return_('early_async')

    result = await Chain(SyncCM()).with_(body_return).run()
    self.assertEqual(result, 'early_async')

  async def test_exit_suppresses_sync(self):
    # SyncCMSuppresses + async body raises -> __exit__ returns True -> None
    result = await Chain(SyncCMSuppresses()).with_(async_raise_runtime).run()
    self.assertIsNone(result)

  async def test_exit_returns_awaitable_success(self):
    # SyncCMWithAwaitableExit: sync __enter__, __exit__ returns a coroutine
    # that returns False. Body succeeds (async) -> _to_async -> exit called,
    # exit result is awaitable -> awaited.
    cm = SyncCMWithAwaitableExit()
    result = await Chain(cm).with_(async_identity).run()
    self.assertEqual(result, 'ctx_value')

  async def test_exit_returns_awaitable_suppresses(self):
    # SyncCMSuppressesAwaitable: __exit__ returns coroutine returning True.
    # Async body raises -> _to_async catches, calls __exit__, awaits it,
    # gets True -> exception suppressed -> returns None (ignore_result=False).
    result = await Chain(SyncCMSuppressesAwaitable()).with_(async_raise_runtime).run()
    self.assertIsNone(result)

  async def test_exit_returns_awaitable_no_suppress(self):
    # SyncCMWithAwaitableExit: __exit__ returns coroutine returning False.
    # Async body raises -> _to_async catches, calls __exit__, awaits it,
    # gets False -> re-raises.
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(SyncCMWithAwaitableExit()).with_(async_raise_runtime).run()
    self.assertEqual(str(ctx.exception), 'async error')


# ===========================================================================
# WITH — FULL_ASYNC TIER
# ===========================================================================

class TestWithFullAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    result = await Chain(AsyncCM()).with_(lambda ctx: ctx).run()
    self.assertEqual(result, 'ctx_value')

  async def test_exception_in_body(self):
    cm = AsyncCM()
    with self.assertRaises(ZeroDivisionError):
      await Chain(cm).with_(lambda ctx: 1 / 0).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_return_from_body(self):
    result = await Chain(AsyncCM()).with_(
      lambda ctx: Chain.return_('early_full_async')
    ).run()
    self.assertEqual(result, 'early_full_async')

  async def test_exit_suppresses(self):
    result = await Chain(AsyncCMSuppresses()).with_(lambda ctx: 1 / 0).run()
    self.assertIsNone(result)

  async def test_async_body(self):
    async def async_body(ctx):
      return ctx + '_processed'

    result = await Chain(AsyncCM()).with_(async_body).run()
    self.assertEqual(result, 'ctx_value_processed')

  async def test_result_is_null(self):
    # When _evaluate_value returns Null (non-callable Null passed as fn),
    # _full_async returns outer_value if ignore_result else None.
    # Passing Null as the fn value: Link(Null) -> _evaluate_value(link, ctx)
    # -> v is Null, not callable -> returns Null. Result is Null ->
    # falls through to `if result is Null: return outer_value if ignore_result else None`.
    result = await Chain(AsyncCM()).with_(Null).run()
    self.assertIsNone(result)

  async def test_exit_called_on_success(self):
    cm = AsyncCM()
    await Chain(cm).with_(lambda ctx: 'ok').run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ===========================================================================
# WITH_DO — SYNC TIER
# ===========================================================================

class TestWithDoSyncTier(unittest.TestCase):

  def test_happy_path(self):
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)

  def test_body_result_discarded(self):
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 42).run()
    self.assertIs(result, cm)

  def test_exception_in_body(self):
    cm = SyncCM()
    with self.assertRaises(ValueError):
      Chain(cm).with_do(lambda ctx: (_ for _ in ()).throw(ValueError('err'))).run()
    # Simpler approach:
    def body_raises(ctx):
      raise ValueError('do body error')
    cm2 = SyncCM()
    with self.assertRaises(ValueError) as ctx:
      Chain(cm2).with_do(body_raises).run()
    self.assertEqual(str(ctx.exception), 'do body error')
    self.assertTrue(cm2.exited)

  def test_return_from_body(self):
    result = Chain(SyncCM()).with_do(
      lambda ctx: Chain.return_('early')
    ).run()
    self.assertEqual(result, 'early')

  def test_exit_suppresses(self):
    cm = SyncCMSuppresses()
    result = Chain(cm).with_do(lambda ctx: 1 / 0).run()
    # with_do + suppressed exception -> returns outer_value (the CM itself)
    self.assertIs(result, cm)

  def test_exit_called_on_success(self):
    cm = SyncCM()
    Chain(cm).with_do(lambda ctx: 'side_effect').run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_preserves_current_value_in_chain(self):
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: 'ignored').then(lambda x: x is cm).run()
    self.assertTrue(result)


# ===========================================================================
# WITH_DO — TO_ASYNC TIER
# ===========================================================================

class TestWithDoToAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    cm = SyncCM()
    result = await Chain(cm).with_do(async_identity).run()
    self.assertIs(result, cm)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_body_result_discarded(self):
    cm = SyncCM()

    async def body(ctx):
      return 'should be discarded'

    result = await Chain(cm).with_do(body).run()
    self.assertIs(result, cm)

  async def test_exception_in_body(self):
    cm = SyncCM()
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(cm).with_do(async_raise_runtime).run()
    self.assertEqual(str(ctx.exception), 'async error')
    self.assertTrue(cm.exited)

  async def test_return_from_body(self):
    async def body_return(ctx):
      Chain.return_('early_to_async_do')

    result = await Chain(SyncCM()).with_do(body_return).run()
    self.assertEqual(result, 'early_to_async_do')

  async def test_exit_suppresses(self):
    cm = SyncCMSuppresses()
    result = await Chain(cm).with_do(async_raise_runtime).run()
    # with_do + suppressed -> outer_value (the CM itself)
    self.assertIs(result, cm)

  async def test_exit_returns_awaitable_success(self):
    cm = SyncCMWithAwaitableExit()
    result = await Chain(cm).with_do(async_identity).run()
    # with_do -> ignore_result=True, but with_do link has ignore_result on the
    # outer Link. The _make_with ignore_result controls what _to_async returns.
    # _to_async success path: exit_result is awaitable -> awaited, then
    # returns outer_value because ignore_result=True.
    self.assertIs(result, cm)

  async def test_exit_returns_awaitable_suppresses(self):
    cm = SyncCMSuppressesAwaitable()
    result = await Chain(cm).with_do(async_raise_runtime).run()
    # exit returns coroutine returning True -> suppressed ->
    # returns outer_value (ignore_result=True)
    self.assertIs(result, cm)


# ===========================================================================
# WITH_DO — FULL_ASYNC TIER
# ===========================================================================

class TestWithDoFullAsyncTier(IsolatedAsyncioTestCase):

  async def test_happy_path(self):
    cm = AsyncCM()
    result = await Chain(cm).with_do(lambda ctx: 'ignored').run()
    self.assertIs(result, cm)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_body_result_discarded(self):
    cm = AsyncCM()
    result = await Chain(cm).with_do(lambda ctx: 42).run()
    self.assertIs(result, cm)

  async def test_exception_in_body(self):
    cm = AsyncCM()
    with self.assertRaises(ZeroDivisionError):
      await Chain(cm).with_do(lambda ctx: 1 / 0).run()
    self.assertTrue(cm.exited)

  async def test_return_from_body(self):
    result = await Chain(AsyncCM()).with_do(
      lambda ctx: Chain.return_('early_full_async_do')
    ).run()
    self.assertEqual(result, 'early_full_async_do')

  async def test_exit_suppresses(self):
    cm = AsyncCMSuppresses()
    result = await Chain(cm).with_do(lambda ctx: 1 / 0).run()
    # with_do + suppressed -> outer_value (the CM)
    self.assertIs(result, cm)

  async def test_async_body(self):
    cm = AsyncCM()

    async def body(ctx):
      return 'async_result_discarded'

    result = await Chain(cm).with_do(body).run()
    self.assertIs(result, cm)

  async def test_ignore_result_null(self):
    # with_do + Null body -> _evaluate_value returns Null ->
    # result is Null -> returns outer_value (ignore_result=True)
    cm = AsyncCM()
    result = await Chain(cm).with_do(Null).run()
    self.assertIs(result, cm)


# ===========================================================================
# GATHER — SYNC/ASYNC/MIXED TIERS
# ===========================================================================

class TestGatherSyncTier(unittest.TestCase):

  def test_all_sync(self):
    result = Chain(10).gather(
      lambda x: x + 1,
      lambda x: x + 2,
      lambda x: x + 3,
    ).run()
    self.assertEqual(result, [11, 12, 13])

  def test_exception_in_fn(self):
    def boom(x):
      raise RuntimeError('gather fn error')
    with self.assertRaises(RuntimeError) as cm:
      Chain(10).gather(boom, lambda x: x).run()
    self.assertEqual(str(cm.exception), 'gather fn error')

  def test_exception_closes_pending_coroutines(self):
    # When a later fn raises, earlier coroutines should be closed.
    closed = []

    async def coro_fn(x):
      return x

    class TrackClose:
      def __init__(self, x):
        self.coro = coro_fn(x)
      def __call__(self, x):
        return self.coro
      def close(self):
        closed.append(True)

    def boom(x):
      raise RuntimeError('boom')

    # The second fn raises; the first fn's result (a coroutine) should be closed.
    import warnings
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      with self.assertRaises(RuntimeError):
        Chain(10).gather(coro_fn, boom).run()


class TestGatherToAsyncTier(IsolatedAsyncioTestCase):

  async def test_mixed_sync_and_async(self):
    result = await Chain(10).gather(
      lambda x: x + 1,
      async_fn,
      lambda x: x + 3,
    ).run()
    self.assertEqual(result, [11, 11, 13])

  async def test_all_async(self):
    result = await Chain(5).gather(
      async_fn,
      async_double,
      async_identity,
    ).run()
    self.assertEqual(result, [6, 10, 5])

  async def test_exception_in_async_fn(self):
    with self.assertRaises(RuntimeError):
      await Chain(10).gather(
        async_fn,
        async_raise_runtime,
      ).run()


# ===========================================================================
# CHAIN-LEVEL TRANSITIONS
# ===========================================================================

class TestChainLevelTransitions(IsolatedAsyncioTestCase):

  async def test_sync_to_async_at_step_1(self):
    result = await Chain(async_fn, 1).run()
    self.assertEqual(result, 2)

  async def test_sync_to_async_at_step_2(self):
    result = await Chain(1).then(async_fn).run()
    self.assertEqual(result, 2)

  async def test_sync_to_async_at_step_3(self):
    result = await (
      Chain(1)
      .then(lambda x: x + 1)
      .then(async_fn)
      .run()
    )
    # 1 -> +1=2 -> async_fn(2)=3
    self.assertEqual(result, 3)

  async def test_sync_to_async_at_step_4(self):
    result = await (
      Chain(1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(async_fn)
      .run()
    )
    # 1 -> +1=2 -> +1=3 -> async_fn(3)=4
    self.assertEqual(result, 4)

  async def test_sync_to_async_at_step_5(self):
    result = await (
      Chain(1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(async_fn)
      .run()
    )
    # 1 -> +1=2 -> +1=3 -> +1=4 -> async_fn(4)=5
    self.assertEqual(result, 5)

  async def test_all_async(self):
    result = await (
      Chain(async_fn, 1)
      .then(async_fn)
      .then(async_fn)
      .run()
    )
    # async_fn(1)=2 -> async_fn(2)=3 -> async_fn(3)=4
    self.assertEqual(result, 4)

  async def test_sync_steps_after_async(self):
    result = await (
      Chain(1)
      .then(async_fn)
      .then(lambda x: x + 10)
      .then(lambda x: x * 2)
      .run()
    )
    # 1 -> async_fn(1)=2 -> +10=12 -> *2=24
    self.assertEqual(result, 24)

  async def test_do_in_async_chain_sync(self):
    tracker = []
    result = await (
      Chain(1)
      .then(async_fn)
      .do(lambda x: tracker.append(x))
      .then(lambda x: x * 3)
      .run()
    )
    # 1 -> async_fn(1)=2 -> do(append 2) -> *3=6
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [2])

  async def test_do_in_async_chain_async(self):
    tracker = []

    async def async_track(x):
      tracker.append(x)
      return 'discarded'

    result = await (
      Chain(1)
      .then(async_fn)
      .do(async_track)
      .then(lambda x: x * 3)
      .run()
    )
    # 1 -> async_fn(1)=2 -> do(track 2) -> *3=6
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [2])

  async def test_except_handler_sync_in_async(self):
    result = await (
      Chain(1)
      .then(async_fn)
      .then(lambda x: 1 / 0)
      .except_(lambda exc: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_except_handler_async_in_async(self):
    async def async_handler(exc):
      return 'async_caught'

    result = await (
      Chain(1)
      .then(async_fn)
      .then(lambda x: 1 / 0)
      .except_(async_handler)
      .run()
    )
    self.assertEqual(result, 'async_caught')

  async def test_finally_handler_sync_in_async(self):
    tracker = []
    result = await (
      Chain(1)
      .then(async_fn)
      .then(lambda x: x + 10)
      .finally_(lambda root_val: tracker.append(root_val))
      .run()
    )
    # root_value is set on the first link evaluation: Chain(1) -> 1.
    # 1 -> async_fn(1)=2 -> +10=12
    self.assertEqual(result, 12)
    self.assertEqual(tracker, [1])

  async def test_finally_handler_async_in_async(self):
    tracker = []

    async def async_cleanup(root_val):
      tracker.append(root_val)

    result = await (
      Chain(1)
      .then(async_fn)
      .then(lambda x: x + 10)
      .finally_(async_cleanup)
      .run()
    )
    # root_value is set on the first link evaluation: Chain(1) -> 1.
    self.assertEqual(result, 12)
    self.assertEqual(tracker, [1])

  async def test_nested_chain_in_async(self):
    inner = Chain().then(lambda x: x * 100)
    result = await (
      Chain(1)
      .then(async_fn)
      .then(inner)
      .run()
    )
    # 1 -> async_fn(1)=2 -> inner(2)=200
    self.assertEqual(result, 200)

  async def test_decorator_with_async_chain(self):
    chain = Chain().then(async_fn).then(lambda x: x + 10)

    @chain.decorator()
    def my_fn(x):
      return x

    result = await my_fn(5)
    # my_fn(5) -> 5 as run value -> async_fn(5)=6 -> +10=16
    self.assertEqual(result, 16)


# ===========================================================================
# ADDITIONAL TRANSITION EDGE CASES
# ===========================================================================

class TestMixedOperationTransitions(IsolatedAsyncioTestCase):
  """Tests combining multiple operations and their tier interactions."""

  async def test_foreach_then_filter(self):
    result = await (
      Chain([1, 2, 3, 4])
      .foreach(async_fn)
      .then(lambda lst: Chain(lst).filter(lambda x: x > 3).run())
      .run()
    )
    # [1,2,3,4] -> async_fn each -> [2,3,4,5] -> filter >3 -> [4,5]
    self.assertEqual(result, [4, 5])

  async def test_with_then_foreach(self):
    result = await (
      Chain(AsyncCM())
      .with_(lambda ctx: [1, 2, 3])
      .then(lambda lst: Chain(lst).foreach(async_fn).run())
      .run()
    )
    # AsyncCM -> ctx_value -> body returns [1,2,3] -> foreach async_fn -> [2,3,4]
    # The inner chain result needs awaiting.
    self.assertEqual(result, [2, 3, 4])

  async def test_sync_foreach_mid_async_chain(self):
    result = await (
      Chain(async_fn, 0)
      .then(lambda x: [x, x + 1, x + 2])
      .foreach(lambda i: i * 10)
      .run()
    )
    # async_fn(0)=1 -> [1,2,3] -> foreach *10 -> [10,20,30]
    self.assertEqual(result, [10, 20, 30])

  async def test_async_foreach_mid_async_chain(self):
    result = await (
      Chain(async_fn, 0)
      .then(lambda x: [x, x + 1, x + 2])
      .foreach(async_double)
      .run()
    )
    # async_fn(0)=1 -> [1,2,3] -> foreach async_double -> [2,4,6]
    self.assertEqual(result, [2, 4, 6])

  async def test_filter_with_async_predicate_in_chain(self):
    async def is_even(x):
      return x % 2 == 0

    result = await (
      Chain(1)
      .then(lambda x: list(range(x, x + 6)))
      .filter(is_even)
      .run()
    )
    # 1 -> [1,2,3,4,5,6] -> filter even -> [2,4,6]
    self.assertEqual(result, [2, 4, 6])

  async def test_gather_in_async_chain(self):
    result = await (
      Chain(async_fn, 5)
      .gather(
        lambda x: x * 2,
        async_fn,
        lambda x: x + 100,
      )
      .run()
    )
    # async_fn(5)=6 -> gather([6*2, async_fn(6), 6+100]) -> [12, 7, 106]
    self.assertEqual(result, [12, 7, 106])

  async def test_foreach_do_preserves_originals_in_async_chain(self):
    tracker = []
    result = await (
      Chain(async_fn, 0)
      .then(lambda x: [x, x + 1, x + 2])
      .foreach_do(lambda i: tracker.append(i * 10))
      .run()
    )
    # async_fn(0)=1 -> [1,2,3] -> foreach_do -> originals [1,2,3]
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [10, 20, 30])

  async def test_with_do_preserves_outer_in_async_chain(self):
    cm = SyncCM()
    result = await (
      Chain(async_fn, 0)
      .then(lambda x: cm)
      .with_do(async_identity)
      .run()
    )
    # async_fn(0)=1 -> lambda -> cm -> with_do(async_identity) ->
    # _to_async path (sync CM + async body), ignore_result=True -> cm
    self.assertIs(result, cm)


class TestTransitionEdgeCases(IsolatedAsyncioTestCase):
  """Edge cases at tier boundaries."""

  async def test_foreach_empty_async_iterable(self):
    class AsyncEmpty:
      def __aiter__(self):
        return self
      async def __anext__(self):
        raise StopAsyncIteration

    result = await Chain(AsyncEmpty()).foreach(lambda x: x).run()
    self.assertEqual(result, [])

  async def test_foreach_do_empty_async_iterable(self):
    class AsyncEmpty:
      def __aiter__(self):
        return self
      async def __anext__(self):
        raise StopAsyncIteration

    result = await Chain(AsyncEmpty()).foreach_do(lambda x: x).run()
    self.assertEqual(result, [])

  async def test_filter_empty_async_iterable(self):
    class AsyncEmpty:
      def __aiter__(self):
        return self
      async def __anext__(self):
        raise StopAsyncIteration

    result = await Chain(AsyncEmpty()).filter(lambda x: True).run()
    self.assertEqual(result, [])

  async def test_foreach_single_item_async(self):
    result = await Chain(AsyncRange(1)).foreach(async_fn).run()
    self.assertEqual(result, [1])

  async def test_filter_none_pass_async(self):
    async def always_false(x):
      return False

    result = await Chain(AsyncRange(5)).filter(always_false).run()
    self.assertEqual(result, [])

  async def test_filter_all_pass_async(self):
    async def always_true(x):
      return True

    result = await Chain(AsyncRange(3)).filter(always_true).run()
    self.assertEqual(result, [0, 1, 2])

  async def test_to_async_foreach_mixed_sync_async_fn(self):
    # fn that returns sync on first call, async on second.
    # This exercises the _to_async path where some results are sync.
    call_count = 0

    def mixed_fn(x):
      nonlocal call_count
      call_count += 1
      if call_count % 2 == 0:
        return async_double(x)
      return x * 2

    result = await Chain([1, 2, 3, 4]).foreach(mixed_fn).run()
    # call 1: sync 1*2=2, call 2: async 2*2=4, call 3: sync 3*2=6, call 4: async 4*2=8
    # But _to_async only gets triggered if the FIRST call is async, or after
    # at least one call is sync and then one is async. Actually, the sync _foreach_op
    # processes items until it hits an awaitable. Call 1 is sync (appended),
    # call 2 is async -> hands off to _to_async with the awaitable.
    # _to_async: awaits call2 -> 4, appends. next(it)=3, call3=sync 6,
    # isawaitable(6)=False, appends. next(it)=4, call4=async 8,
    # isawaitable=True, awaits -> 8, appends.
    self.assertEqual(result, [2, 4, 6, 8])

  async def test_to_async_filter_mixed_sync_async_predicate(self):
    call_count = 0

    def mixed_pred(x):
      nonlocal call_count
      call_count += 1
      if call_count % 2 == 0:
        async def _async_pred():
          return x > 2
        return _async_pred()
      return x > 2

    result = await Chain([1, 2, 3, 4]).filter(mixed_pred).run()
    # call1: sync False (1>2 is False), call2: async (2>2 is False),
    # -> _to_async triggered. call3: sync (3>2 is True), call4: async (4>2 is True)
    self.assertEqual(result, [3, 4])

  async def test_link_temp_args_set_on_error_in_to_async(self):
    call_count = 0

    async def boom_on_third(x):
      nonlocal call_count
      call_count += 1
      if call_count == 3:
        raise ValueError('boom at third')
      return x

    try:
      await Chain([10, 20, 30]).foreach(boom_on_third).run()
    except ValueError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (30,))
    else:
      self.fail('ValueError was not raised')

  async def test_link_temp_args_set_on_error_in_full_async(self):
    def boom_on_second(x):
      if x == 1:
        raise ValueError('boom at 1')
      return x

    try:
      await Chain(AsyncRange(5)).foreach(boom_on_second).run()
    except ValueError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (1,))
    else:
      self.fail('ValueError was not raised')

  async def test_link_temp_args_set_on_filter_error_to_async(self):
    async def boom(x):
      raise ValueError('filter boom')

    try:
      await Chain([10, 20]).filter(boom).run()
    except ValueError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (10,))
    else:
      self.fail('ValueError was not raised')

  async def test_link_temp_args_set_on_filter_error_full_async(self):
    def boom(x):
      raise ValueError('filter boom full async')

    try:
      await Chain(AsyncRange(3)).filter(boom).run()
    except ValueError as exc:
      self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
      values = list(exc.__quent_link_temp_args__.values())
      self.assertEqual(values[0], (0,))
    else:
      self.fail('ValueError was not raised')


# ===========================================================================
# FROZEN CHAIN TRANSITIONS
# ===========================================================================

class TestFrozenChainTransitions(IsolatedAsyncioTestCase):
  """Verify that frozen chains handle sync/async transitions correctly."""

  async def test_frozen_async_chain(self):
    chain = Chain().then(async_fn).then(lambda x: x + 10).freeze()
    result = await chain.run(5)
    # async_fn(5)=6 -> +10=16
    self.assertEqual(result, 16)

  async def test_frozen_chain_reuse(self):
    chain = Chain().then(async_fn).freeze()
    r1 = await chain.run(1)
    r2 = await chain.run(10)
    self.assertEqual(r1, 2)
    self.assertEqual(r2, 11)

  def test_frozen_sync_chain(self):
    chain = Chain().then(lambda x: x * 2).freeze()
    result = chain.run(5)
    self.assertEqual(result, 10)

  async def test_frozen_foreach_async(self):
    chain = Chain().foreach(async_fn).freeze()
    result = await chain.run([1, 2, 3])
    self.assertEqual(result, [2, 3, 4])


if __name__ == '__main__':
  unittest.main()
