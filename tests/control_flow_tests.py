"""Tests for Chain.return_() and Chain.break_() control flow signals:
early return, map break, nesting propagation, async variants,
and error cases.
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from helpers import async_fn, async_identity, AsyncRange


class TestReturnSignal(unittest.TestCase):

  def test_return_no_value_returns_none(self):
    result = Chain(5).then(lambda x: Chain.return_()).run()
    self.assertIsNone(result)

  def test_return_with_value(self):
    result = Chain(5).then(lambda x: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  def test_return_with_callable_value(self):
    result = Chain(5).then(lambda x: Chain.return_(lambda: 99)).run()
    self.assertEqual(result, 99)

  def test_return_with_callable_and_args(self):
    result = Chain(5).then(lambda x: Chain.return_(lambda a, b: a + b, 3, 4)).run()
    self.assertEqual(result, 7)

  def test_return_with_ellipsis(self):
    result = Chain(5).then(lambda x: Chain.return_(lambda: 'ell', ...)).run()
    self.assertEqual(result, 'ell')

  def test_return_exits_chain_early(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .then(lambda x: tracker.append('should_not_run'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  def test_return_from_then_step(self):
    tracker = []
    result = (
      Chain(1)
      .then(lambda x: x + 1)
      .then(lambda x: Chain.return_(x * 10))
      .then(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 20)
    self.assertEqual(tracker, [])

  def test_return_from_do_step(self):
    tracker = []
    result = (
      Chain(1)
      .then(lambda x: x + 1)
      .do(lambda x: Chain.return_(x * 10))
      .then(lambda x: tracker.append(x))
      .run()
    )
    # return_ in a do step still exits the chain early with the return value
    self.assertEqual(result, 20)
    self.assertEqual(tracker, [])


class TestReturnNested(unittest.TestCase):

  def test_return_propagates_through_nested(self):
    inner = Chain().then(lambda x: Chain.return_(99))
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 99)

  def test_return_in_deeply_nested_3_levels(self):
    level3 = Chain().then(lambda x: Chain.return_(999))
    level2 = Chain().then(level3)
    result = Chain(5).then(level2).run()
    self.assertEqual(result, 999)


class TestReturnAsync(unittest.IsolatedAsyncioTestCase):

  async def test_return_in_async_chain(self):
    result = await Chain(5).then(async_fn).then(lambda x: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  async def test_return_with_async_value_awaited(self):
    async def make_value():
      return 77

    result = await Chain(5).then(async_fn).then(lambda x: Chain.return_(make_value)).run()
    self.assertEqual(result, 77)

  async def test_return_propagates_async_nested(self):
    inner = Chain().then(async_identity).then(lambda x: Chain.return_(88))
    result = await Chain(5).then(async_fn).then(inner).run()
    self.assertEqual(result, 88)


class TestBreakSignal(unittest.TestCase):

  def test_break_in_map_stops_iteration(self):
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: Chain.break_() if x == 3 else x)
      .run()
    )
    self.assertEqual(result, [1, 2])

  def test_break_no_value_returns_partial_results(self):
    result = (
      Chain([10, 20, 30, 40])
      .map(lambda x: Chain.break_() if x == 30 else x * 2)
      .run()
    )
    # break with no value returns the list accumulated so far
    self.assertEqual(result, [20, 40])

  def test_break_with_value_returns_value(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(99) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 99)

  def test_break_with_callable_value(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(lambda: 'done') if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 'done')

  def test_break_outside_map_raises_quent_exception(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: Chain.break_()).run()
    self.assertIn('Chain.break_() cannot be used outside of a map/foreach iteration', str(ctx.exception))

  def test_break_in_filter_stops(self):
    # filter catches _ControlFlowSignal and re-raises it, so _Break propagates
    # to the chain's _run handler where it raises QuentException (not in map context)
    with self.assertRaises(QuentException) as ctx:
      Chain([1, 2, 3]).filter(lambda x: Chain.break_() if x == 2 else True).run()
    self.assertIn('Chain.break_() cannot be used outside of a map/foreach iteration', str(ctx.exception))


class TestBreakAsync(unittest.IsolatedAsyncioTestCase):

  async def test_break_in_async_map(self):
    result = await (
      Chain(AsyncRange(6))
      .map(lambda x: Chain.break_() if x == 3 else x)
      .run()
    )
    self.assertEqual(result, [0, 1, 2])

  async def test_break_outside_map_async_raises(self):
    with self.assertRaises(QuentException) as ctx:
      await Chain(5).then(async_fn).then(lambda x: Chain.break_()).run()
    self.assertIn('Chain.break_() cannot be used outside of a map/foreach iteration', str(ctx.exception))

  async def test_break_with_async_value(self):
    async def make_break_val():
      return 'async_break_val'

    result = await (
      Chain(AsyncRange(5))
      .map(lambda x: Chain.break_(make_break_val) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 'async_break_val')


class TestControlFlowSignalEscape(unittest.TestCase):

  def test_return_escape_via_run_raises(self):
    # run() wraps _run() and catches any escaped _ControlFlowSignal as a safety net.
    # Normally _Return is caught inside _run, so we patch _run at the class level
    # to simulate an escape.
    from unittest.mock import patch
    from quent._core import _Return

    chain = Chain(5)
    with patch.object(Chain, '_run', side_effect=_Return(42, (), {})):
      with self.assertRaises(QuentException) as ctx:
        chain.run()
      self.assertIn('control flow signal escaped', str(ctx.exception).lower())

  def test_break_escape_via_run_raises(self):
    # Same safety net test for _Break via run().
    from unittest.mock import patch
    from quent._core import _Break, Null as _Null

    chain = Chain(5)
    with patch.object(Chain, '_run', side_effect=_Break(_Null, (), {})):
      with self.assertRaises(QuentException) as ctx:
        chain.run()
      self.assertIn('control flow signal escaped', str(ctx.exception).lower())


class TestReturnBreakInteractionWithExcept(unittest.TestCase):

  def test_return_not_caught_by_except_handler(self):
    # _Return is caught BEFORE the BaseException handler in _run.
    # Verify return_ works even with except_ registered.
    handler_called = []
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .except_(lambda exc: handler_called.append(exc))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(handler_called, [])

  def test_break_not_caught_by_except_handler(self):
    # In map context, _Break is caught inside _make_foreach before
    # the chain-level except handler ever sees it.
    handler_called = []
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: Chain.break_() if x == 3 else x)
      .except_(lambda exc: handler_called.append(exc))
      .run()
    )
    self.assertEqual(result, [1, 2])
    self.assertEqual(handler_called, [])

  def test_return_in_except_handler_raises_quent_exception(self):
    # Control flow signals in except_ handlers raise QuentException
    result_chain = (
      Chain(5)
      .then(lambda x: 1 / 0)
      .except_(lambda exc: Chain.return_(99))
    )
    with self.assertRaises(QuentException) as ctx:
      result_chain.run()
    self.assertIn('control flow signals inside except handlers is not allowed', str(ctx.exception).lower())

  def test_return_in_finally_handler_raises_quent_exception(self):
    # Control flow signals in finally_ handlers raise QuentException
    result_chain = (
      Chain(5)
      .then(lambda x: x)
      .finally_(lambda x: Chain.return_(99))
    )
    with self.assertRaises(QuentException) as ctx:
      result_chain.run()
    self.assertIn('control flow signals inside finally handlers is not allowed', str(ctx.exception).lower())


class TestControlFlowInAsyncFinally(unittest.IsolatedAsyncioTestCase):

  async def test_return_in_async_finally_raises_quent_exception(self):
    # Mirrors test_return_in_finally_handler_raises_quent_exception (sync)
    # but exercises the _run_async finally path.
    async def async_step(x):
      return x

    result_chain = (
      Chain(5)
      .then(async_step)
      .finally_(lambda x: Chain.return_(99))
    )
    with self.assertRaises(QuentException) as ctx:
      await result_chain.run()
    self.assertIn('control flow signals inside finally handlers is not allowed', str(ctx.exception).lower())

  async def test_break_in_async_finally_raises_quent_exception(self):
    async def async_step(x):
      return x

    result_chain = (
      Chain(5)
      .then(async_step)
      .finally_(lambda x: Chain.break_())
    )
    with self.assertRaises(QuentException) as ctx:
      await result_chain.run()
    self.assertIn('control flow signals inside finally handlers is not allowed', str(ctx.exception).lower())


if __name__ == '__main__':
  unittest.main()
