"""Tests for interactions between Chain operations: verifying that pairs and
sequences of operations compose correctly (then/do, except/finally, foreach
with except, filter with except, gather with except, with_ with except,
nested chains, decorators, freeze, long mixed chains, return_ in various
contexts, and async mixed interactions).
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  sync_fn, async_fn, SyncCM, AsyncCM, SyncCMSuppresses, raise_fn,
)


# ---------------------------------------------------------------------------
# then / do interaction
# ---------------------------------------------------------------------------

class TestThenThenInteraction(unittest.TestCase):

  def test_then_then_value_flows(self):
    result = Chain(1).then(lambda x: x + 1).then(lambda x: x * 2).run()
    self.assertEqual(result, 4)

  def test_then_do_value_preserved(self):
    # do discards its result; chain value after do is still the value before do.
    result = Chain(1).then(lambda x: x + 1).do(lambda x: x * 100).run()
    self.assertEqual(result, 2)

  def test_do_then_value_from_before_do(self):
    # do on root value 1 discards result; then receives 1.
    result = Chain(1).do(lambda x: x * 100).then(lambda x: x + 1).run()
    self.assertEqual(result, 2)


# ---------------------------------------------------------------------------
# except / finally interaction
# ---------------------------------------------------------------------------

class TestExceptFinallyInteraction(unittest.TestCase):

  def test_except_then_finally_both_run(self):
    tracker = []
    result = (
      Chain(10)
      .then(raise_fn)
      .except_(lambda e: (tracker.append('except'), 'caught')[1])
      .finally_(lambda rv: tracker.append(('finally', rv)))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertIn('except', tracker)
    # finally_ receives root_value (10)
    self.assertEqual(tracker[-1], ('finally', 10))

  def test_exception_in_except_still_runs_finally(self):
    tracker = []

    def bad_handler(exc):
      tracker.append('except')
      raise RuntimeError('handler boom')

    with self.assertRaises(RuntimeError) as cm:
      (
        Chain(10)
        .then(raise_fn)
        .except_(bad_handler)
        .finally_(lambda rv: tracker.append('finally'))
        .run()
      )
    self.assertEqual(str(cm.exception), 'handler boom')
    self.assertIn('except', tracker)
    self.assertIn('finally', tracker)

  def test_finally_after_except_success(self):
    tracker = []
    result = (
      Chain(10)
      .then(raise_fn)
      .except_(lambda e: 'recovered')
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 'recovered')
    # finally_ receives root_value = 10 (the successfully evaluated root).
    self.assertEqual(tracker, [10])


# ---------------------------------------------------------------------------
# foreach with except
# ---------------------------------------------------------------------------

class TestForeachWithExcept(unittest.TestCase):

  def test_exception_in_foreach_fn_caught_by_except(self):
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: 1 / 0)
      .except_(lambda e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_break_in_foreach_not_caught_by_except(self):
    # _Break is a _ControlFlowSignal, caught by _make_foreach before it
    # reaches the chain's except handler. Chain.break_() with no value
    # returns the accumulated list so far (fallback).
    except_called = []
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: Chain.break_() if x == 2 else x)
      .except_(lambda e: except_called.append('called') or 'caught')
      .run()
    )
    # break_ fires before item 2 is appended. Item 1 was already processed.
    self.assertEqual(result, [1])
    self.assertEqual(except_called, [])


# ---------------------------------------------------------------------------
# filter with except
# ---------------------------------------------------------------------------

class TestFilterWithExcept(unittest.TestCase):

  def test_exception_in_filter_caught_by_except(self):
    result = (
      Chain([1, 2, 3])
      .filter(lambda x: 1 / 0)
      .except_(lambda e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')


# ---------------------------------------------------------------------------
# gather with except
# ---------------------------------------------------------------------------

class TestGatherWithExcept(unittest.TestCase):

  def test_exception_in_gather_caught_by_except(self):
    result = (
      Chain(5)
      .gather(lambda x: 1 / 0)
      .except_(lambda e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')


# ---------------------------------------------------------------------------
# with_ with except
# ---------------------------------------------------------------------------

class TestWithWithExcept(unittest.TestCase):

  def test_exception_in_with_body_caught_by_except(self):
    result = (
      Chain(SyncCM())
      .with_(lambda ctx: 1 / 0)
      .except_(lambda e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  def test_cm_exit_suppresses_then_except_not_called(self):
    # SyncCMSuppresses.__exit__ returns True, suppressing the exception.
    # _make_with's except branch: if __exit__ returns truthy, the exception
    # is suppressed and the result is None (ignore_result=False).
    except_called = []
    result = (
      Chain(SyncCMSuppresses())
      .with_(lambda ctx: 1 / 0)
      .except_(lambda e: except_called.append('called') or 'caught')
      .run()
    )
    # Exception was suppressed by the CM, so except_ is never invoked.
    self.assertEqual(except_called, [])
    # When the exception is suppressed and ignore_result is False, _make_with
    # returns None.
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# nested chain with except / finally
# ---------------------------------------------------------------------------

class TestNestedWithExceptFinally(unittest.TestCase):

  def test_inner_except_outer_finally(self):
    outer_tracker = []
    inner = Chain().then(raise_fn).except_(lambda e: 'inner_caught')
    result = (
      Chain(5)
      .then(inner)
      .finally_(lambda rv: outer_tracker.append('outer_finally'))
      .run()
    )
    self.assertEqual(result, 'inner_caught')
    self.assertIn('outer_finally', outer_tracker)

  def test_inner_finally_outer_except(self):
    inner_tracker = []

    def inner_raise(x):
      raise ValueError('inner error')

    inner = Chain().then(inner_raise).finally_(lambda rv: inner_tracker.append('inner_finally'))
    result = (
      Chain(5)
      .then(inner)
      .except_(lambda e: 'outer_caught')
      .run()
    )
    self.assertEqual(result, 'outer_caught')
    self.assertIn('inner_finally', inner_tracker)


# ---------------------------------------------------------------------------
# decorator with various ops
# ---------------------------------------------------------------------------

class TestDecoratorWithAllOps(unittest.TestCase):

  def test_decorator_with_foreach(self):
    @Chain().then(lambda x: [x, x + 1]).foreach(lambda i: i * 2).decorator()
    def fn(n):
      return n
    result = fn(5)
    self.assertEqual(result, [10, 12])

  def test_decorator_with_filter(self):
    @Chain().then(lambda x: list(range(x))).filter(lambda i: i > 2).decorator()
    def fn(n):
      return n
    result = fn(5)
    self.assertEqual(result, [3, 4])

  def test_decorator_with_gather(self):
    @Chain().gather(lambda x: x + 1, lambda x: x * 2).decorator()
    def fn(n):
      return n
    result = fn(5)
    self.assertEqual(result, [6, 10])

  def test_decorator_with_with(self):
    @Chain().then(lambda x: SyncCM()).with_(lambda ctx: ctx).decorator()
    def fn(n):
      return n
    result = fn(5)
    # SyncCM.__enter__ returns 'ctx_value'
    self.assertEqual(result, 'ctx_value')


# ---------------------------------------------------------------------------
# freeze with various ops
# ---------------------------------------------------------------------------

class TestFreezeWithAllOps(unittest.TestCase):

  def test_frozen_with_except(self):
    frozen = (
      Chain()
      .then(raise_fn)
      .except_(lambda e: 'frozen_caught')
      .freeze()
    )
    result = frozen.run(5)
    self.assertEqual(result, 'frozen_caught')

  def test_frozen_with_finally(self):
    tracker = []
    frozen = (
      Chain()
      .then(lambda x: x * 2)
      .finally_(lambda rv: tracker.append(rv))
      .freeze()
    )
    result = frozen.run(5)
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [5])

  def test_frozen_with_foreach(self):
    frozen = (
      Chain()
      .then(lambda x: list(range(x)))
      .foreach(lambda i: i * 3)
      .freeze()
    )
    result = frozen.run(4)
    self.assertEqual(result, [0, 3, 6, 9])

  def test_frozen_with_gather(self):
    frozen = (
      Chain()
      .gather(lambda x: x + 1, lambda x: x - 1)
      .freeze()
    )
    result = frozen.run(10)
    self.assertEqual(result, [11, 9])


# ---------------------------------------------------------------------------
# long chain with mixed ops
# ---------------------------------------------------------------------------

class TestLongChainWithMixedOps(unittest.TestCase):

  def test_10_mixed_then_do_steps(self):
    tracker = []
    result = (
      Chain(1)
      .then(lambda x: x + 1)     # 2
      .do(lambda x: tracker.append(x))  # side-effect, value stays 2
      .then(lambda x: x * 3)     # 6
      .do(lambda x: tracker.append(x))  # side-effect, value stays 6
      .then(lambda x: x + 4)     # 10
      .do(lambda x: tracker.append(x))  # side-effect, value stays 10
      .then(lambda x: x * 2)     # 20
      .do(lambda x: tracker.append(x))  # side-effect, value stays 20
      .then(lambda x: x - 5)     # 15
      .do(lambda x: tracker.append(x))  # side-effect, value stays 15
      .run()
    )
    self.assertEqual(result, 15)
    self.assertEqual(tracker, [2, 6, 10, 20, 15])

  def test_foreach_then_filter(self):
    # foreach maps [1,2,3,4] -> [2,4,6,8], filter keeps > 4.
    result = (
      Chain([1, 2, 3, 4])
      .foreach(lambda x: x * 2)
      .filter(lambda x: x > 4)
      .run()
    )
    self.assertEqual(result, [6, 8])


# ---------------------------------------------------------------------------
# Chain.return_() in various contexts
# ---------------------------------------------------------------------------

class TestReturnInVariousContexts(unittest.TestCase):

  def test_return_in_with_body(self):
    result = Chain(SyncCM()).with_(lambda ctx: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  def test_return_in_foreach_fn(self):
    # _Return propagates out of foreach's _ControlFlowSignal handler (re-raised)
    # and is caught by the chain's _Return handler.
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: Chain.return_(99) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 99)

  def test_return_in_gather_fn(self):
    # _Return inside a gather fn propagates out of gather, caught by chain's
    # _Return handler.
    result = Chain(5).gather(lambda x: Chain.return_(42)).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# async mixed interactions
# ---------------------------------------------------------------------------

class TestAsyncMixedInteractions(IsolatedAsyncioTestCase):

  async def test_sync_then_async_foreach(self):
    result = await (
      Chain(5)
      .then(lambda x: list(range(x)))
      .foreach(async_fn)
      .run()
    )
    # range(5) -> [0,1,2,3,4], async_fn adds 1 -> [1,2,3,4,5]
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_async_with_sync_gather(self):
    result = await (
      Chain(5)
      .then(async_fn)  # makes chain async, result 6
      .gather(lambda x: x + 1, lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, [7, 12])

  async def test_async_except_sync_finally(self):
    tracker = []

    async def async_raiser(x):
      raise ValueError('async boom')

    result = await (
      Chain(5)
      .then(async_raiser)
      .except_(lambda e: 'caught')
      .finally_(lambda rv: tracker.append('finally'))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertIn('finally', tracker)


if __name__ == '__main__':
  unittest.main()
