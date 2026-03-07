"""Tests for Chain.finally_(): registration, execution, async behavior,
and edge cases around sync-to-async transitions.
"""
from __future__ import annotations

import asyncio
import inspect
import unittest
import warnings

from quent import Chain, Null, QuentException
from helpers import async_fn, sync_fn, raise_fn, async_raise_fn


class TestFinallyRegistration(unittest.TestCase):

  def test_register_handler(self):
    chain = Chain(1)
    self.assertIsNone(chain.on_finally_link)
    chain.finally_(lambda rv: rv)
    self.assertIsNotNone(chain.on_finally_link)

  def test_double_finally_raises_quent_exception(self):
    chain = Chain(1).finally_(lambda rv: rv)
    with self.assertRaises(QuentException) as ctx:
      chain.finally_(lambda rv: rv)
    self.assertIn('finally', str(ctx.exception).lower())


class TestFinallyExecution(unittest.TestCase):

  def test_handler_called_on_success(self):
    tracker = []
    result = Chain(42).finally_(lambda rv: tracker.append(rv)).run()
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [42])

  def test_handler_called_on_exception(self):
    tracker = []
    # Use except_ to swallow the error so the chain does not propagate.
    # finally_ should still run.
    result = (
      Chain(10)
      .then(raise_fn)
      .except_(lambda rv, exc: 'caught')
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, [10])

  def test_handler_called_on_unhandled_exception(self):
    # Even when there is no except_ and the exception propagates,
    # the finally handler still runs (it is in a `finally:` block).
    tracker = []
    with self.assertRaises(ValueError):
      Chain(10).then(raise_fn).finally_(lambda rv: tracker.append(rv)).run()
    self.assertEqual(tracker, [10])

  def test_handler_receives_root_value(self):
    tracker = []
    result = (
      Chain(42)
      .then(lambda x: x + 1)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 43)
    self.assertEqual(tracker, [42])

  def test_handler_result_does_not_affect_chain_result(self):
    result = Chain(42).finally_(lambda rv: 999).run()
    self.assertEqual(result, 42)

  def test_handler_exception_propagates(self):
    def bad_handler(rv):
      raise RuntimeError('handler boom')

    with self.assertRaises(RuntimeError) as ctx:
      Chain(42).finally_(bad_handler).run()
    self.assertEqual(str(ctx.exception), 'handler boom')

  def test_control_flow_in_handler_raises(self):
    def handler_with_return(rv):
      Chain.return_(rv)

    with self.assertRaises(QuentException) as ctx:
      Chain(42).finally_(handler_with_return).run()
    self.assertIn('control flow', str(ctx.exception).lower())

  def test_finally_runs_after_except(self):
    order = []
    result = (
      Chain(7)
      .then(raise_fn)
      .except_(lambda rv, exc: (order.append('except'), 'recovered')[1])
      .finally_(lambda rv: order.append(('finally', rv)))
      .run()
    )
    self.assertEqual(result, 'recovered')
    # except_ runs first, then finally_
    self.assertEqual(order, ['except', ('finally', 7)])

  def test_root_value_with_run_arg(self):
    tracker = []
    chain = (
      Chain()
      .then(lambda x: x + 1)
      .finally_(lambda rv: tracker.append(rv))
    )
    result = chain.run(10)
    self.assertEqual(result, 11)
    # root_value is the result of evaluating Link(10): 10 is not callable, so 10.
    self.assertEqual(tracker, [10])

  def test_root_value_is_first_evaluated_value(self):
    tracker = []
    result = (
      Chain(lambda: 5)
      .then(lambda x: x + 1)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 6)
    # root_value = first evaluated result = 5 (result of calling lambda: 5)
    self.assertEqual(tracker, [5])

  def test_handler_with_extra_args(self):
    # When the link has args, _evaluate_value calls v(*args, **kwargs)
    # and does NOT pass root_value. So the handler receives only the
    # explicit args.
    tracker = []
    result = (
      Chain(42)
      .finally_(lambda tag: tracker.append(tag), 'cleanup')
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, ['cleanup'])

  def test_handler_with_kwargs(self):
    # Same as above: kwargs makes _evaluate_value call v(**kwargs)
    # without root_value.
    tracker = []
    def handler(label=None):
      tracker.append(label)
    result = (
      Chain(42)
      .finally_(handler, label='done')
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, ['done'])


class TestFinallyAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_handler_awaited(self):
    tracker = []
    async def async_handler(rv):
      tracker.append(rv)

    result = await Chain(async_fn, 1).finally_(async_handler).run()
    # async_fn(1) = 2
    self.assertEqual(result, 2)
    self.assertEqual(tracker, [2])

  async def test_async_handler_on_success(self):
    tracker = []
    async def async_handler(rv):
      tracker.append(('finally', rv))

    result = await (
      Chain(5)
      .then(async_fn)
      .then(lambda x: x + 10)
      .finally_(async_handler)
      .run()
    )
    # 5 -> async_fn(5)=6 -> +10=16
    self.assertEqual(result, 16)
    # root_value = 5 (first evaluated result from root_link)
    self.assertEqual(tracker, [('finally', 5)])

  async def test_async_handler_on_exception(self):
    tracker = []
    async def async_handler(rv):
      tracker.append(('finally', rv))

    result = await (
      Chain(5)
      .then(async_raise_fn)
      .except_(lambda rv, exc: 'caught')
      .finally_(async_handler)
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, [('finally', 5)])

  async def test_handler_receives_root_value_async(self):
    tracker = []
    async def async_handler(rv):
      tracker.append(rv)

    result = await (
      Chain(42)
      .then(async_fn)
      .then(lambda x: x * 2)
      .finally_(async_handler)
      .run()
    )
    # 42 -> async_fn(42)=43 -> *2=86
    self.assertEqual(result, 86)
    # root_value = 42
    self.assertEqual(tracker, [42])

  async def test_async_handler_exception_propagates(self):
    async def bad_handler(rv):
      raise RuntimeError('async handler boom')

    with self.assertRaises(RuntimeError) as ctx:
      await Chain(async_fn, 1).finally_(bad_handler).run()
    self.assertEqual(str(ctx.exception), 'async handler boom')

  async def test_async_handler_result_does_not_affect_chain_result(self):
    async def handler_returns_value(rv):
      return 9999

    result = await Chain(async_fn, 1).finally_(handler_returns_value).run()
    # async_fn(1) = 2; handler result discarded
    self.assertEqual(result, 2)


class TestFinallySyncReturnsCoroutine(unittest.IsolatedAsyncioTestCase):

  async def test_fire_and_forget_warning(self):
    # A purely sync chain (no awaitable results) with an async finally_
    # handler: the returned coroutine is scheduled fire-and-forget and
    # a RuntimeWarning is emitted. Needs a running event loop so
    # _ensure_future can create the task.
    tracker = []
    async def async_handler(rv):
      tracker.append(rv)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      result = Chain(42).finally_(async_handler).run()
      self.assertEqual(result, 42)
      # Let the fire-and-forget task execute.
      await asyncio.sleep(0.05)
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      self.assertTrue(len(runtime_warnings) >= 1)
      self.assertIn('finally', str(runtime_warnings[0].message).lower())
      # The handler was scheduled and ran.
      self.assertEqual(tracker, [42])


class TestFinallyNoEventLoop(unittest.TestCase):

  def test_no_event_loop_raises(self):
    # When no event loop is running, an async finally handler causes
    # _ensure_future to raise RuntimeError, which is caught and
    # converted to QuentException.
    async def async_handler(rv):
      pass

    with self.assertRaises(QuentException) as ctx:
      Chain(42).finally_(async_handler).run()
    self.assertIn('finally', str(ctx.exception).lower())


class TestFinallyIgnoredOnAsyncTransition(unittest.IsolatedAsyncioTestCase):

  async def test_sync_finally_skipped_when_async_transition(self):
    # When a chain hits an async step, _run sets ignore_finally=True and
    # returns the coroutine from _run_async. The sync finally block is
    # skipped. However, _run_async has its own finally block that calls
    # the handler. So the handler IS called, just on the async path.
    tracker = []
    result = await (
      Chain(5)
      .then(async_fn)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    # 5 -> async_fn(5)=6
    self.assertEqual(result, 6)
    # Handler runs in the async path with root_value=5
    self.assertEqual(tracker, [5])

  async def test_async_path_finally_runs(self):
    # Verify the async path's finally block runs the handler even when
    # the async transition happens mid-chain.
    tracker = []
    result = await (
      Chain(1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(async_fn)
      .then(lambda x: x + 1)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    # 1 -> +1=2 -> +1=3 -> async_fn(3)=4 -> +1=5
    self.assertEqual(result, 5)
    # root_value = 1 (first evaluated result, set before async transition)
    self.assertEqual(tracker, [1])

  async def test_async_handler_in_async_transition(self):
    # Async handler + async transition: handler is awaited in _run_async's
    # finally block.
    tracker = []
    async def async_handler(rv):
      tracker.append(('async_handler', rv))

    result = await (
      Chain(10)
      .then(async_fn)
      .finally_(async_handler)
      .run()
    )
    # 10 -> async_fn(10)=11
    self.assertEqual(result, 11)
    self.assertEqual(tracker, [('async_handler', 10)])

  async def test_async_path_finally_runs_on_exception(self):
    # The async finally block runs even when an exception occurs in the
    # async path.
    tracker = []
    result = await (
      Chain(5)
      .then(async_raise_fn)
      .except_(lambda rv, exc: 'recovered')
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(tracker, [5])


if __name__ == '__main__':
  unittest.main()
