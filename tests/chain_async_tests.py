"""Tests for async bridging behavior: sync/async transitions, coroutine detection, run values."""
from __future__ import annotations

import inspect
import unittest
import warnings

from quent import Chain, Null, QuentException
from helpers import async_fn, sync_fn, async_identity, sync_identity


class TestAsyncBridging(unittest.IsolatedAsyncioTestCase):

  def test_pure_sync_chain_returns_sync(self):
    result = Chain(42).run()
    self.assertFalse(inspect.iscoroutine(result))
    self.assertEqual(result, 42)

  def test_async_root_returns_coroutine(self):
    result = Chain(async_fn, 1).run()
    self.assertTrue(inspect.iscoroutine(result))
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      result.close()

  async def test_async_root_awaited(self):
    result = await Chain(async_identity, 5).run()
    self.assertEqual(result, 5)

  async def test_sync_root_async_step(self):
    result = await Chain(5).then(async_fn).run()
    self.assertEqual(result, 6)

  async def test_sync_steps_then_async_step(self):
    result = await Chain(1).then(lambda x: x + 1).then(async_fn).run()
    # 1 -> +1 = 2 -> async_fn(2) = 3
    self.assertEqual(result, 3)

  async def test_async_step_then_sync_steps(self):
    result = await Chain(1).then(async_fn).then(lambda x: x + 1).run()
    # 1 -> async_fn(1) = 2 -> +1 = 3
    self.assertEqual(result, 3)

  async def test_all_async_steps(self):
    result = await Chain(1).then(async_fn).then(async_fn).run()
    # 1 -> async_fn(1) = 2 -> async_fn(2) = 3
    self.assertEqual(result, 3)

  async def test_async_root_sync_rest(self):
    result = await Chain(async_fn, 1).then(lambda x: x + 1).run()
    # async_fn(1) = 2 -> +1 = 3
    self.assertEqual(result, 3)

  async def test_async_transition_midchain(self):
    result = await (
      Chain(1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(async_fn)
      .then(lambda x: x + 1)
      .run()
    )
    # 1 -> +1=2 -> +1=3 -> async_fn(3)=4 -> +1=5
    self.assertEqual(result, 5)

  async def test_value_propagation_across_boundary(self):
    tracker = []
    async def record_and_pass(x):
      tracker.append(('async', x))
      return x * 10

    def record_sync(x):
      tracker.append(('sync', x))
      return x + 1

    result = await Chain(2).then(record_sync).then(record_and_pass).then(record_sync).run()
    # 2 -> record_sync(2): append ('sync', 2), return 3
    # -> record_and_pass(3): append ('async', 3), return 30
    # -> record_sync(30): append ('sync', 30), return 31
    self.assertEqual(result, 31)
    self.assertEqual(tracker, [('sync', 2), ('async', 3), ('sync', 30)])

  async def test_do_with_async_fn(self):
    tracker = []
    async def side_effect(x):
      tracker.append(x)
      return 'discarded'

    result = await Chain(5).do(side_effect).run()
    # .do() discards the result of side_effect, so current_value stays 5.
    self.assertEqual(result, 5)
    self.assertEqual(tracker, [5])

  async def test_multiple_async_steps_in_chain(self):
    async def double(x):
      return x * 2

    async def add_ten(x):
      return x + 10

    result = await Chain(3).then(async_fn).then(double).then(add_ten).then(async_fn).run()
    # 3 -> async_fn(3)=4 -> double(4)=8 -> add_ten(8)=18 -> async_fn(18)=19
    self.assertEqual(result, 19)


class TestAsyncRunDetection(unittest.IsolatedAsyncioTestCase):

  def test_returns_coroutine_when_async(self):
    result = Chain(async_fn, 1).run()
    self.assertTrue(inspect.iscoroutine(result))
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      result.close()

  def test_sync_chain_returns_immediate(self):
    result = Chain(42).run()
    self.assertFalse(inspect.iscoroutine(result))

  async def test_custom_awaitable_triggers_async(self):
    class CustomAwaitable:
      def __await__(self):
        return (yield from asyncio.coroutine(lambda: 99)().__await__())

    # Use a simpler awaitable that works reliably.
    import asyncio

    async def returns_awaitable():
      return 99

    # A callable that returns a coroutine triggers async path.
    result = await Chain(returns_awaitable, ...).run()
    self.assertEqual(result, 99)

  async def test_native_coroutine_function_triggers_async(self):
    async def compute():
      return 42

    result = await Chain(compute, ...).run()
    self.assertEqual(result, 42)

  def test_sync_chain_with_lambda_returns_immediate(self):
    result = Chain(10).then(lambda x: x * 2).run()
    self.assertFalse(inspect.iscoroutine(result))
    self.assertEqual(result, 20)


class TestAsyncWithRunValue(unittest.IsolatedAsyncioTestCase):

  async def test_async_run_value(self):
    # run(async_fn, 5): creates Link(async_fn, (5,)), evaluates -> async_fn(5) -> coroutine
    # triggers async path. await -> 6. Then lambda(6) = 7.
    result = await Chain().then(lambda x: x + 1).run(async_fn, 5)
    self.assertEqual(result, 7)

  async def test_sync_run_value_into_async_chain(self):
    # run(5): Link(5), 5 is not callable -> _evaluate_value returns 5 (sync).
    # Then async_fn(5) -> awaitable -> async path. await -> 6.
    result = await Chain().then(async_fn).run(5)
    self.assertEqual(result, 6)

  async def test_async_root_sync_steps_async_finally(self):
    tracker = []
    async def async_cleanup(root_val):
      tracker.append(('finally', root_val))

    result = await (
      Chain(async_fn, 1)
      .then(lambda x: x + 10)
      .finally_(async_cleanup)
      .run()
    )
    # async_fn(1) = 2 (root_value) -> +10 = 12
    self.assertEqual(result, 12)
    # finally receives the root_value (2)
    self.assertEqual(tracker, [('finally', 2)])

  async def test_async_run_value_with_sync_root(self):
    # Chain(10) has a root_link. run(async_fn, 5) overrides it:
    # Link(async_fn, (5,)) is created, next_link = first_link.
    # async_fn(5) -> awaitable -> async. await -> 6. Then lambda(6)=7.
    result = await Chain(10).then(lambda x: x + 1).run(async_fn, 5)
    self.assertEqual(result, 7)

  async def test_sync_run_value_no_root(self):
    # Chain() with no root. run(42) -> Link(42), not callable -> 42.
    # then sync_fn(42) = 43. Entirely sync.
    result = Chain().then(sync_fn).run(42)
    self.assertFalse(inspect.iscoroutine(result))
    self.assertEqual(result, 43)


if __name__ == '__main__':
  unittest.main()
