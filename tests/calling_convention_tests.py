"""Tests for quent calling conventions across all chain methods.

Verifies the calling convention rules implemented in _evaluate_value:
  1. Ellipsis (...) as first arg: call v() with no arguments
  2. Explicit args/kwargs: call v(*args, **kwargs) -- current_value NOT passed
  3. Callable with no explicit args: call v(current_value) -- unless Null, then v()
  4. Non-callable: return v as literal

Also verifies:
  - except_ handlers receive the exception as current_value
  - finally_ handlers receive root_value as current_value
  - do() discards results
  - with_/with_do inner fn receives context manager's __enter__ value
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  sync_fn, async_fn, sync_identity, async_identity,
  raise_fn, async_raise_fn, SyncCM, AsyncCM,
)


class TestThenCallingConvention(unittest.TestCase):

  def test_fn_receives_current_value(self):
    result = Chain(5).then(lambda x: x * 2).run()
    self.assertEqual(result, 10)

  def test_fn_no_current_value_calls_no_args(self):
    result = Chain().then(lambda: 42).run()
    self.assertEqual(result, 42)

  def test_explicit_args_override(self):
    result = Chain(5).then(lambda a, b: a + b, 10, 20).run()
    self.assertEqual(result, 30)

  def test_explicit_kwargs(self):
    result = Chain(5).then(lambda x=0: x, x=99).run()
    self.assertEqual(result, 99)

  def test_mixed_args_kwargs(self):
    result = Chain(5).then(lambda a, b=0: a + b, 1, b=2).run()
    self.assertEqual(result, 3)

  def test_ellipsis_calls_no_args(self):
    result = Chain(5).then(lambda: 'no_args', ...).run()
    self.assertEqual(result, 'no_args')

  def test_literal_value_replaces(self):
    result = Chain(5).then(42).run()
    self.assertEqual(result, 42)

  def test_none_as_literal(self):
    result = Chain(5).then(None).run()
    self.assertIsNone(result)


class TestDoCallingConvention(unittest.TestCase):

  def test_fn_receives_current_value(self):
    result = Chain(5).do(lambda x: x * 2).run()
    self.assertEqual(result, 5)

  def test_fn_with_explicit_args(self):
    tracker = []
    Chain(5).do(lambda a: tracker.append(a), 99).run()
    self.assertEqual(tracker, [99])

  def test_fn_with_ellipsis(self):
    tracker = []
    Chain(5).do(lambda: tracker.append('called'), ...).run()
    self.assertEqual(tracker, ['called'])

  def test_result_discarded(self):
    result = Chain(5).do(lambda x: x * 100).then(lambda x: x).run()
    self.assertEqual(result, 5)


class TestExceptCallingConvention(unittest.TestCase):

  def test_handler_receives_exception(self):
    result = (
      Chain()
      .then(raise_fn)
      .except_(lambda e: type(e).__name__)
      .run()
    )
    self.assertEqual(result, 'ValueError')

  def test_handler_with_explicit_args(self):
    result = (
      Chain()
      .then(raise_fn)
      .except_(lambda a: a, 'custom')
      .run()
    )
    self.assertEqual(result, 'custom')

  def test_handler_with_ellipsis(self):
    result = (
      Chain()
      .then(raise_fn)
      .except_(lambda: 'handled', ...)
      .run()
    )
    self.assertEqual(result, 'handled')


class TestFinallyCallingConvention(unittest.TestCase):

  def test_handler_receives_root_value(self):
    tracker = []
    Chain(42).then(lambda x: x + 1).finally_(lambda rv: tracker.append(rv)).run()
    self.assertEqual(tracker, [42])

  def test_handler_with_explicit_args(self):
    tracker = []
    Chain(42).finally_(lambda a: tracker.append(a), 99).run()
    self.assertEqual(tracker, [99])

  def test_handler_with_ellipsis(self):
    tracker = []
    Chain(42).finally_(lambda: tracker.append('called'), ...).run()
    self.assertEqual(tracker, ['called'])


class TestEllipsisInEveryMethod(unittest.TestCase):

  def test_then_ellipsis(self):
    result = Chain(5).then(lambda: 'ignored_cv', ...).run()
    self.assertEqual(result, 'ignored_cv')

  def test_do_ellipsis(self):
    tracker = []
    result = Chain(5).do(lambda: tracker.append('side'), ...).run()
    self.assertEqual(result, 5)
    self.assertEqual(tracker, ['side'])

  def test_except_ellipsis(self):
    result = (
      Chain()
      .then(raise_fn)
      .except_(lambda: 'recovered', ...)
      .run()
    )
    self.assertEqual(result, 'recovered')

  def test_finally_ellipsis(self):
    tracker = []
    Chain(42).finally_(lambda: tracker.append('fin'), ...).run()
    self.assertEqual(tracker, ['fin'])

  def test_with_ellipsis(self):
    result = Chain(SyncCM()).with_(lambda: 'no_ctx', ...).run()
    self.assertEqual(result, 'no_ctx')

  def test_with_do_ellipsis(self):
    tracker = []
    cm = SyncCM()
    result = Chain(cm).with_do(lambda: tracker.append('called'), ...).run()
    self.assertEqual(tracker, ['called'])
    self.assertIs(result, cm)


class TestAsyncCallingConventions(IsolatedAsyncioTestCase):

  async def test_then_async_fn_with_current_value(self):
    result = await Chain(5).then(async_fn).run()
    self.assertEqual(result, 6)

  async def test_then_async_fn_with_explicit_args(self):
    result = await Chain(5).then(async_fn, 10).run()
    self.assertEqual(result, 11)

  async def test_then_async_fn_with_ellipsis(self):
    async def no_arg():
      return 99
    result = await Chain(5).then(no_arg, ...).run()
    self.assertEqual(result, 99)

  async def test_do_async_fn(self):
    result = await Chain(5).do(async_fn).run()
    self.assertEqual(result, 5)

  async def test_except_async_fn(self):
    result = await (
      Chain()
      .then(raise_fn)
      .except_(async_identity)
      .run()
    )
    self.assertIsInstance(result, ValueError)

  async def test_finally_async_fn(self):
    tracker = []

    async def async_tracker(rv):
      tracker.append(rv)

    # Chain must enter async path (via an async .then()) so that
    # the finally handler's coroutine is awaited instead of being
    # scheduled as a fire-and-forget task.
    result = await Chain(42).then(async_identity).finally_(async_tracker).run()
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [42])


if __name__ == '__main__':
  unittest.main()
