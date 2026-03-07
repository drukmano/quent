"""Tests for Chain.decorator(): wrapping functions with chain pipelines."""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from helpers import sync_fn, async_fn, sync_identity, async_identity


class TestDecoratorSync(unittest.TestCase):

  def test_wraps_function(self):
    @Chain().then(lambda x: x * 2).decorator()
    def fn(n):
      return n
    self.assertEqual(fn(5), 10)

  def test_fn_becomes_root_value(self):
    # fn's return value is the root value that chain steps operate on.
    tracker = []

    @Chain().then(lambda x: x + 1).finally_(lambda rv: tracker.append(rv)).decorator()
    def fn(n):
      return n

    result = fn(7)
    self.assertEqual(result, 8)
    # root_value should be fn's return value (7), not the chain result.
    self.assertEqual(tracker, [7])

  def test_preserves_name(self):
    @Chain().then(lambda x: x).decorator()
    def my_function(n):
      """My docstring."""
      return n
    self.assertEqual(my_function.__name__, 'my_function')
    self.assertEqual(my_function.__doc__, 'My docstring.')

  def test_receives_args_and_kwargs(self):
    @Chain().then(lambda x: x).decorator()
    def fn(a, b, c=0):
      return a + b + c
    self.assertEqual(fn(3, 4), 7)
    self.assertEqual(fn(3, 4, c=10), 17)

  def test_chain_steps_execute_on_fn_result(self):
    @Chain().then(lambda x: x + 1).then(lambda x: x * 2).decorator()
    def fn():
      return 5
    # (5 + 1) * 2 = 12
    self.assertEqual(fn(), 12)


class TestDecoratorAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_function(self):
    # decorator() wraps fn with a sync _wrapper. When fn is async,
    # fn(*args, **kwargs) returns a coroutine, which is awaitable,
    # so _run transitions to _run_async. The _wrapper returns the
    # coroutine from _run_async, which must be awaited.
    @Chain().then(lambda x: x * 3).decorator()
    async def fn(n):
      return n

    result = await fn(4)
    self.assertEqual(result, 12)

  async def test_async_chain_steps(self):
    # Sync fn but async chain steps.
    @Chain().then(async_fn).then(lambda x: x * 2).decorator()
    def fn(n):
      return n

    result = await fn(5)
    # fn(5) = 5, async_fn(5) = 6, 6 * 2 = 12
    self.assertEqual(result, 12)

  async def test_async_fn_and_async_steps(self):
    @Chain().then(async_fn).decorator()
    async def fn(n):
      return n + 10

    result = await fn(5)
    # fn(5) = 15, async_fn(15) = 16
    self.assertEqual(result, 16)


class TestDecoratorControlFlow(unittest.TestCase):

  def test_control_flow_escape_raises(self):
    # If a _ControlFlowSignal somehow escapes the chain, the decorator
    # wrapper catches it and raises QuentException.
    # Chain.return_ inside a then step normally gets caught by _run's
    # _Return handler. But if Chain.return_ is used in a context where
    # _run does not catch it (e.g., a nested chain scenario that leaks),
    # the decorator's wrapper catches _ControlFlowSignal and raises
    # QuentException. We can trigger this by using _run's _Break path:
    # _Break outside a map raises QuentException (via _run),
    # but let's verify the decorator wrapper's own catch too.
    #
    # Actually, the decorator's try/except _ControlFlowSignal wraps
    # chain._run. Since _run already handles _Return and _Break (converting
    # _Break to QuentException when not nested), the decorator's catch
    # is a safety net. We verify the chain raises QuentException for
    # an escaped _Break (which _run converts to QuentException).
    @Chain().then(lambda x: Chain.break_()).decorator()
    def fn():
      return 1

    with self.assertRaises(QuentException) as ctx:
      fn()
    # The message could come from _run's _Break handler or the decorator's
    # _ControlFlowSignal handler -- either way it should be QuentException.
    self.assertIsInstance(ctx.exception, QuentException)


class TestDecoratorReuse(unittest.TestCase):

  def test_decorator_applied_to_multiple_fns(self):
    dec = Chain().then(lambda x: x * 10).decorator()

    @dec
    def fn_a(n):
      return n

    @dec
    def fn_b(n):
      return n + 1

    self.assertEqual(fn_a(3), 30)
    self.assertEqual(fn_b(3), 40)
    # Names are preserved individually.
    self.assertEqual(fn_a.__name__, 'fn_a')
    self.assertEqual(fn_b.__name__, 'fn_b')

  def test_decorated_fn_called_multiple_times(self):
    call_count = 0

    @Chain().then(lambda x: x + 1).decorator()
    def fn(n):
      nonlocal call_count
      call_count += 1
      return n

    results = [fn(i) for i in range(5)]
    self.assertEqual(results, [1, 2, 3, 4, 5])
    self.assertEqual(call_count, 5)


class TestDecoratorWithExceptFinally(unittest.TestCase):

  def test_except_handler_in_decorator(self):
    @Chain().then(lambda x: 1 / 0).except_(lambda e: 'caught').decorator()
    def fn(n):
      return n

    self.assertEqual(fn(42), 'caught')

  def test_finally_handler_in_decorator(self):
    tracker = []

    @Chain().finally_(lambda rv: tracker.append(rv)).decorator()
    def fn(n):
      return n * 2

    result = fn(5)
    self.assertEqual(result, 10)
    # finally_ receives root_value, which is fn's return value.
    self.assertEqual(tracker, [10])

  def test_except_and_finally_together(self):
    tracker = []

    @(
      Chain()
      .then(lambda x: 1 / 0)
      .except_(lambda e: 'recovered')
      .finally_(lambda rv: tracker.append(rv))
      .decorator()
    )
    def fn(n):
      return n

    result = fn(7)
    self.assertEqual(result, 'recovered')
    # finally_ receives root_value = fn's return (7).
    self.assertEqual(tracker, [7])

  def test_finally_runs_even_on_unhandled_exception(self):
    tracker = []

    @Chain().then(lambda x: 1 / 0).finally_(lambda rv: tracker.append(rv)).decorator()
    def fn(n):
      return n

    with self.assertRaises(ZeroDivisionError):
      fn(3)
    # finally_ still ran with root_value = fn's return (3).
    self.assertEqual(tracker, [3])


if __name__ == '__main__':
  unittest.main()
