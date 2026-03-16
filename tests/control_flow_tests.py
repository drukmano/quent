# SPDX-License-Identifier: MIT
"""Tests for SPEC §7 — Control Flow.

Covers: Chain.return_() value semantics, nested chain propagation,
restrictions; Chain.break_() value semantics, outside iteration,
concurrent iteration break, priority.
"""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Chain, QuentException
from tests.tests_helper import (
  SymmetricTestCase,
)

# ---------------------------------------------------------------------------
# §7.2 Early Return — Chain.return_()
# ---------------------------------------------------------------------------


class ReturnNoValueTest(SymmetricTestCase):
  """SPEC §7.2.1: return_() with no value → None."""

  async def test_return_no_value(self) -> None:
    """return_() with no value produces None."""
    result = Chain(5).then(lambda x: Chain.return_()).then(lambda x: x * 100).run()
    self.assertIsNone(result)

  async def test_return_no_value_async(self) -> None:
    """Async: return_() with no value produces None."""

    async def step(x):
      return Chain.return_()

    result = await Chain(5).then(step).then(lambda x: x * 100).run()
    self.assertIsNone(result)


class ReturnWithValueTest(SymmetricTestCase):
  """SPEC §7.2.1: return_() with values."""

  async def test_return_non_callable(self) -> None:
    """return_() with non-callable value returns as-is."""
    result = Chain(5).then(lambda x: Chain.return_(42)).then(lambda x: 'never').run()
    self.assertEqual(result, 42)

  async def test_return_callable(self) -> None:
    """return_() with callable: called when signal caught, return value becomes result."""
    result = Chain(5).then(lambda x: Chain.return_(lambda: 'from_fn')).then(lambda x: 'never').run()
    self.assertEqual(result, 'from_fn')

  async def test_return_callable_with_args(self) -> None:
    """return_() with callable + args follows calling conventions."""
    result = Chain(5).then(lambda x: Chain.return_(lambda a, b: a + b, 10, 20)).then(lambda x: 'never').run()
    self.assertEqual(result, 30)

  async def test_return_callable_with_ellipsis_as_arg(self) -> None:
    """return_() with callable + Ellipsis: Ellipsis passed as explicit arg."""
    result = Chain(5).then(lambda x: Chain.return_(lambda a: f'got {a}', ...)).then(lambda x: 'never').run()
    self.assertEqual(result, f'got {Ellipsis}')

  async def test_return_skips_remaining_steps(self) -> None:
    """Steps after return_() are skipped."""
    visited = []
    result = (
      Chain(5)
      .then(lambda x: (visited.append(1), Chain.return_(x * 2))[-1])
      .then(lambda x: (visited.append(2), x * 3)[-1])
      .then(lambda x: (visited.append(3), x * 4)[-1])
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(visited, [1])


# ---------------------------------------------------------------------------
# §7.2.2 Nested Chain Propagation
# ---------------------------------------------------------------------------


class ReturnNestedPropagationTest(SymmetricTestCase):
  """SPEC §7.2.2: return_() propagates to outermost chain."""

  async def test_return_exits_outermost_chain(self) -> None:
    """return_() in nested chain exits the outermost chain."""
    inner = Chain().then(lambda x: Chain.return_('early') if x > 3 else x)
    result = Chain(5).then(inner).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'early')

  async def test_return_in_deep_nesting(self) -> None:
    """return_() propagates through multiple nesting levels."""
    inner2 = Chain().then(lambda x: Chain.return_('deep'))
    inner1 = Chain().then(inner2)
    result = Chain(1).then(inner1).then(lambda x: 'unreachable').run()
    self.assertEqual(result, 'deep')

  async def test_return_in_nested_no_match_passes_through(self) -> None:
    """When return_() condition not met, value passes through normally."""
    inner = Chain().then(lambda x: Chain.return_('early') if x > 10 else x * 2)
    result = Chain(3).then(inner).then(lambda x: x + 1).run()
    self.assertEqual(result, 7)  # 3*2=6, 6+1=7


# ---------------------------------------------------------------------------
# §7.2.3 Restrictions
# ---------------------------------------------------------------------------


class ReturnRestrictionsTest(TestCase):
  """SPEC §7.2.3: return_() restrictions."""

  def test_return_in_except_raises_quent_exception(self) -> None:
    """return_() in except handler raises QuentException."""

    def handler(info):
      return Chain.return_('bad')

    c = Chain(1).then(lambda x: 1 / 0).except_(handler)
    with self.assertRaises(QuentException):
      c.run()

  def test_return_in_finally_raises_quent_exception(self) -> None:
    """return_() in finally handler raises QuentException."""

    def cleanup(rv):
      return Chain.return_('bad')

    c = Chain(1).finally_(cleanup)
    with self.assertRaises(QuentException):
      c.run()

  def test_return_in_top_level_chain_extracts_value(self) -> None:
    """return_() in top-level chain: run() catches signal and extracts value."""
    c = Chain(1).then(lambda x: Chain.return_(x))
    result = c.run()
    # run() catches _Return and returns the carried value.
    self.assertEqual(result, 1)


# ---------------------------------------------------------------------------
# §7.3 Break — Chain.break_()
# ---------------------------------------------------------------------------


class BreakNoValueTest(SymmetricTestCase):
  """SPEC §7.3.1: break_() with no value → partial results preserved."""

  async def test_break_no_value_preserves_partial(self) -> None:
    """break_() with no value: results collected so far are preserved."""
    result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_() if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4])  # items before break

  async def test_break_no_value_async(self) -> None:
    """Async break_() with no value: partial results preserved."""

    async def mapper(x):
      if x == 3:
        return Chain.break_()
      return x * 2

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4])


class BreakWithValueTest(SymmetricTestCase):
  """SPEC §7.3.1: break_() with value appends to partial results."""

  async def test_break_with_value_appends_to_results(self) -> None:
    """break_() with value: appends to partial results."""
    result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_(x * 10) if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4, 30])

  async def test_break_with_value_appends_async(self) -> None:
    """Async break_() with value appends to partial results."""

    async def mapper(x):
      if x == 3:
        return Chain.break_(x * 10)
      return x * 2

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4, 30])


class BreakCallableValueTest(SymmetricTestCase):
  """SPEC §7.3.1: break_() with callable values."""

  async def test_break_callable_called_when_caught(self) -> None:
    """break_() with callable: called when signal is caught, appended to partial."""
    result = Chain([1, 2, 3]).foreach(lambda x: Chain.break_(lambda: 'stop') if x == 2 else x).run()
    self.assertEqual(result, [1, 'stop'])

  async def test_break_callable_with_args(self) -> None:
    """break_() with callable + args follows calling conventions."""
    result = Chain([1, 2, 3]).foreach(lambda x: Chain.break_(lambda a, b: a + b, 10, 20) if x == 2 else x).run()
    self.assertEqual(result, [1, 30])


# ---------------------------------------------------------------------------
# §7.3.2 Outside Iteration
# ---------------------------------------------------------------------------


class BreakOutsideIterationTest(TestCase):
  """SPEC §7.3.2: break_() outside iteration."""

  def test_break_outside_iteration_raises(self) -> None:
    """break_() outside foreach/foreach_do raises QuentException."""
    c = Chain(1).then(lambda x: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('cannot be used outside', str(ctx.exception))

  def test_break_outside_iteration_specific_message(self) -> None:
    """break_() outside iteration produces the exact specified error message."""
    c = Chain(1).then(lambda x: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertEqual(
      str(ctx.exception),
      'Chain.break_() cannot be used outside of a foreach/foreach_do iteration.',
    )


# ---------------------------------------------------------------------------
# §7.3.3 In Except/Finally Handlers
# ---------------------------------------------------------------------------


class BreakInHandlersTest(TestCase):
  """SPEC §7.3.3: break_() in except/finally handlers."""

  def test_break_in_except_raises_quent_exception(self) -> None:
    """break_() in except handler raises QuentException."""

    def handler(info):
      return Chain.break_()

    c = Chain(1).then(lambda x: 1 / 0).except_(handler)
    with self.assertRaises(QuentException):
      c.run()

  def test_break_in_finally_raises_quent_exception(self) -> None:
    """break_() in finally handler raises QuentException."""

    def cleanup(rv):
      return Chain.break_()

    c = Chain(1).finally_(cleanup)
    with self.assertRaises(QuentException):
      c.run()


# ---------------------------------------------------------------------------
# §7.3.4 Concurrent Iteration Break
# ---------------------------------------------------------------------------


class ConcurrentBreakTest(SymmetricTestCase):
  """SPEC §7.3.4: Concurrent iteration break behavior."""

  async def test_concurrent_break_truncates_results(self) -> None:
    """Concurrent break: results truncated to elements before break index."""
    # With concurrency, break at index 2 (x==3) should truncate to indices 0,1
    result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_() if x == 3 else x * 2, concurrency=2).run()
    self.assertEqual(result, [2, 4])

  async def test_concurrent_break_with_value(self) -> None:
    """Concurrent break with value: appends to truncated results."""
    result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_('stopped') if x == 3 else x * 2, concurrency=2).run()
    self.assertEqual(result, [2, 4, 'stopped'])

  async def test_concurrent_break_earliest_index_wins(self) -> None:
    """When multiple workers break, earliest index wins."""
    # All elements >= 2 break. Element at index 1 (value 2) is earliest.
    result = Chain([1, 2, 3, 4, 5]).foreach(lambda x: Chain.break_() if x >= 2 else x * 2, concurrency=5).run()
    self.assertEqual(result, [2])  # only index 0 before earliest break


# ---------------------------------------------------------------------------
# §7.3.5 Priority in Concurrent Iteration
# ---------------------------------------------------------------------------


class ConcurrentPriorityTest(SymmetricTestCase):
  """SPEC §7.3.5: Priority — return > break > regular exceptions."""

  async def test_return_has_highest_priority(self) -> None:
    """return_() takes priority over break_() and regular exceptions."""

    def mixed(x):
      if x == 1:
        return Chain.return_('return wins')
      if x == 2:
        return Chain.break_('break')
      if x == 3:
        raise ValueError('error')
      return x

    result = Chain([1, 2, 3, 4]).foreach(mixed, concurrency=4).run()
    self.assertEqual(result, 'return wins')

  async def test_break_over_regular_exception(self) -> None:
    """break_() takes priority over regular exceptions."""

    def mixed(x):
      if x == 1:
        return Chain.break_('break wins')
      if x == 3:
        raise ValueError('error')
      return x

    result = Chain([0, 1, 2, 3]).foreach(mixed, concurrency=4).run()
    self.assertEqual(result, [0, 'break wins'])

  async def test_break_over_exception_even_when_exception_index_earlier(self) -> None:
    """break_() takes priority over regular exceptions regardless of index order.

    Per SPEC §7.3.5: break signals always take priority over regular exceptions
    regardless of index. Item at index 0 is the inline probe and must succeed.
    Exception at index 1 (earlier) and break at index 3 (later): break still wins.
    """

    def mixed(x):
      if x == 1:
        raise ValueError('error at index 1')
      if x == 3:
        return Chain.break_('break wins')
      return x

    result = Chain([0, 1, 2, 3, 4]).foreach(mixed, concurrency=5).run()
    self.assertEqual(result, [0, 2, 'break wins'])


# ---------------------------------------------------------------------------
# Break in foreach_do
# ---------------------------------------------------------------------------


class BreakInForeachTest(SymmetricTestCase):
  """break_() in foreach_do operations."""

  async def test_break_in_foreach_do_no_value(self) -> None:
    """break_() in foreach_do: partial original elements preserved."""
    side_effects = []

    def effect(x):
      if x == 3:
        return Chain.break_()
      side_effects.append(x)

    result = Chain([1, 2, 3, 4, 5]).foreach_do(effect).run()
    self.assertEqual(result, [1, 2])
    self.assertEqual(side_effects, [1, 2])

  async def test_break_in_foreach_do_with_value(self) -> None:
    """break_() in foreach_do with value: appends to partial items."""
    result = Chain([1, 2, 3, 4, 5]).foreach_do(lambda x: Chain.break_('done') if x == 3 else None).run()
    self.assertEqual(result, [1, 2, 'done'])


# ---------------------------------------------------------------------------
# Async return in nested chains
# ---------------------------------------------------------------------------


class AsyncReturnTest(SymmetricTestCase):
  """Async return_() behavior."""

  async def test_async_return_in_then(self) -> None:
    """Async step with return_() exits chain."""

    async def step(x):
      return Chain.return_(x * 10)

    result = await Chain(5).then(step).then(lambda x: 'never').run()
    self.assertEqual(result, 50)

  async def test_async_return_propagates_through_nesting(self) -> None:
    """Async return_() propagates through nested chains."""

    async def inner_step(x):
      return Chain.return_('async early')

    inner = Chain().then(inner_step)
    result = await Chain(1).then(inner).then(lambda x: 'unreachable').run()
    self.assertEqual(result, 'async early')

  async def test_async_break_outside_iteration(self) -> None:
    """Async break_() outside iteration raises QuentException."""

    async def step(x):
      return Chain.break_()

    c = Chain(1).then(step)
    with self.assertRaises(QuentException):
      await c.run()


# ---------------------------------------------------------------------------
# §5.6 break_() in gather → QuentException
# ---------------------------------------------------------------------------


class BreakInGatherTest(SymmetricTestCase):
  """SPEC §5.6: break_() signals are not allowed in gather operations."""

  async def test_break_in_gather_raises_quent_exception(self) -> None:
    """break_() in gather raises QuentException."""
    c = Chain(5).gather(lambda x: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('break', str(ctx.exception).lower())
    self.assertIn('gather', str(ctx.exception).lower())

  async def test_break_in_gather_async(self) -> None:
    """Async break_() in gather raises QuentException."""

    async def fn(x):
      return Chain.break_()

    c = Chain(5).gather(fn)
    with self.assertRaises(QuentException) as ctx:
      await c.run()
    self.assertIn('break', str(ctx.exception).lower())
    self.assertIn('gather', str(ctx.exception).lower())

  async def test_break_in_gather_multiple_fns(self) -> None:
    """break_() in one of multiple gather fns raises QuentException."""
    c = Chain(5).gather(lambda x: x * 2, lambda x: Chain.break_(), lambda x: x + 1)
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('break', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Async control flow: return_() from async step in various positions
# ---------------------------------------------------------------------------


class AsyncReturnFromNestedChainTest(SymmetricTestCase):
  """Async return_() through nested chain propagation."""

  async def test_async_return_from_deeply_nested_chain(self) -> None:
    """Async return_() propagates through deeply nested chains."""

    async def deep_step(x):
      return Chain.return_('deep_async')

    inner2 = Chain().then(deep_step)
    inner1 = Chain().then(inner2)
    result = await Chain(1).then(inner1).then(lambda x: 'unreachable').run()
    self.assertEqual(result, 'deep_async')


class AsyncBreakInMapTest(SymmetricTestCase):
  """Async break_() in map operations."""

  async def test_async_break_in_map_no_value(self) -> None:
    """Async break_() in map: partial results preserved."""

    async def mapper(x):
      if x == 3:
        return Chain.break_()
      return x * 2

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4])

  async def test_async_break_in_map_with_value(self) -> None:
    """Async break_() in map with value: appends to partial results."""

    async def mapper(x):
      if x == 3:
        return Chain.break_('stopped')
      return x * 2

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4, 'stopped'])

  async def test_async_break_in_foreach_do(self) -> None:
    """Async break_() in foreach_do: partial original elements preserved."""
    side_effects = []

    async def effect(x):
      if x == 3:
        return Chain.break_()
      side_effects.append(x)

    result = await Chain([1, 2, 3, 4, 5]).foreach_do(effect).run()
    self.assertEqual(result, [1, 2])
    self.assertEqual(side_effects, [1, 2])


# ---------------------------------------------------------------------------
# §7 Iteration control flow signal tests
# ---------------------------------------------------------------------------


class ReturnInForeachTest(SymmetricTestCase):
  """return_() inside foreach callback exits loop and chain."""

  async def test_return_in_foreach_exits_chain_sync(self) -> None:
    """Sync: return_() in foreach callback exits loop and returns value from chain."""

    def mapper(x):
      if x == 3:
        return Chain.return_('early_exit')
      return x * 2

    result = Chain([1, 2, 3, 4, 5]).foreach(mapper).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'early_exit')

  async def test_return_in_foreach_exits_chain_async(self) -> None:
    """Async: return_() in foreach callback exits loop and returns value from chain."""

    async def mapper(x):
      if x == 3:
        return Chain.return_('async_early_exit')
      return x * 2

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper).then(lambda x: 'nope').run()
    self.assertEqual(result, 'async_early_exit')

  async def test_return_in_foreach_do_exits_chain_sync(self) -> None:
    """Sync: return_() in foreach_do callback exits loop and chain."""

    def effect(x):
      if x == 3:
        return Chain.return_('early from foreach_do')

    result = Chain([1, 2, 3, 4, 5]).foreach_do(effect).then(lambda x: 'nope').run()
    self.assertEqual(result, 'early from foreach_do')

  async def test_return_in_foreach_do_exits_chain_async(self) -> None:
    """Async: return_() in foreach_do callback exits loop and chain."""

    async def effect(x):
      if x == 3:
        return Chain.return_('async_early_foreach_do')

    result = await Chain([1, 2, 3, 4, 5]).foreach_do(effect).then(lambda x: 'nope').run()
    self.assertEqual(result, 'async_early_foreach_do')


class ReturnInNestedChainWithinForeachTest(SymmetricTestCase):
  """return_() in nested chain used as foreach callback.

  When a chain is used as a foreach callback, _IterOp calls chain(item)
  which goes through chain.run(). The nested chain's own _run() catches
  _Return and extracts the value, so return_() does NOT propagate to the
  outer chain. Instead, the return value becomes the result for that
  element in the foreach.
  """

  async def test_return_in_nested_chain_within_foreach_sync(self) -> None:
    """Sync: return_() in nested chain becomes element result, does not exit outer chain."""
    inner = Chain().then(lambda x: Chain.return_('nested_return') if x == 3 else x * 2)
    result = Chain([1, 2, 3, 4, 5]).foreach(inner).run()
    self.assertEqual(result, [2, 4, 'nested_return', 8, 10])

  async def test_return_in_nested_chain_within_foreach_async(self) -> None:
    """Async: return_() in nested chain becomes element result, does not exit outer chain."""

    async def inner_step(x):
      if x == 3:
        return Chain.return_('async_nested_return')
      return x * 2

    inner = Chain().then(inner_step)
    result = await Chain([1, 2, 3, 4, 5]).foreach(inner).run()
    self.assertEqual(result, [2, 4, 'async_nested_return', 8, 10])


class ControlFlowInConcurrentForeachTest(SymmetricTestCase):
  """Control flow signals in concurrent foreach."""

  async def test_concurrent_return_exits_chain(self) -> None:
    """Concurrent foreach: return_() signal exits chain entirely."""

    def mapper(x):
      if x == 3:
        return Chain.return_('concurrent_return')
      return x * 2

    result = Chain([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, 'concurrent_return')

  async def test_concurrent_break_truncates(self) -> None:
    """Concurrent foreach: break_() truncates results to items before break index."""

    def mapper(x):
      if x == 3:
        return Chain.break_()
      return x * 2

    result = Chain([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, [2, 4])

  async def test_concurrent_break_with_value(self) -> None:
    """Concurrent foreach: break_(value) appends value to truncated results."""

    def mapper(x):
      if x == 3:
        return Chain.break_('stopped')
      return x * 2

    result = Chain([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, [2, 4, 'stopped'])

  async def test_concurrent_async_return_exits_chain(self) -> None:
    """Async concurrent foreach: return_() exits chain."""

    async def mapper(x):
      if x == 3:
        return Chain.return_('async_concurrent_return')
      return x * 2

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, 'async_concurrent_return')

  async def test_concurrent_async_break_truncates(self) -> None:
    """Async concurrent foreach: break_() truncates results."""

    async def mapper(x):
      if x == 3:
        return Chain.break_()
      return x * 2

    result = await Chain([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, [2, 4])


# ---------------------------------------------------------------------------
# §7.3: break_() in deeply nested async chain
# ---------------------------------------------------------------------------


class AsyncNestedBreakPropagationTest(IsolatedAsyncioTestCase):
  """§7.3: break_() in deeply nested async chain propagates through _run_async."""

  async def test_async_break_in_nested_chain_propagates_through_run_async(self) -> None:
    """Deeply nested async break propagation."""

    async def async_step(x):
      return x

    chain_b = Chain().then(async_step).then(lambda x: Chain.break_() if x >= 2 else x)
    chain_a = Chain().then(chain_b)
    chain_wrapper = Chain().then(chain_a)

    # foreach_do catches the QuentException from the outermost chain
    with self.assertRaises(QuentException) as ctx:
      await Chain([1, 2, 3]).foreach_do(chain_wrapper).run()
    self.assertIn('cannot be used outside', str(ctx.exception))


# ---------------------------------------------------------------------------
# §7.2: return_() with async callable — awaitable result is awaited
# ---------------------------------------------------------------------------


class AsyncReturnAwaitableValueTest(IsolatedAsyncioTestCase):
  """§7.2: return_() with async callable — awaitable result is awaited."""

  async def test_async_return_with_async_callable_value(self) -> None:
    """return_(async_callable) awaitable is awaited."""

    async def compute():
      return 99

    async def step(x):
      return Chain.return_(compute)

    result = await Chain(5).then(step).then(lambda x: 'never').run()
    self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# §7.1 Lazy Evaluation of Signal Values (SPEC-201)
# ---------------------------------------------------------------------------


class ReturnLazyEvaluationTest(SymmetricTestCase):
  """SPEC §7.1: Signal values are lazily evaluated — callable only invoked when caught."""

  async def test_return_callable_not_called_during_propagation(self) -> None:
    """return_(callable) in nested chain: callable NOT called during propagation, only when caught."""
    call_log: list[str] = []

    def tracked_callable():
      call_log.append('evaluated')
      return 'lazy_result'

    # Build 3-level nesting: inner2 raises return_, inner1 wraps it, outer catches it.
    inner2 = Chain().then(lambda x: Chain.return_(tracked_callable))
    inner1 = Chain().then(inner2)

    # Before running, callable not called
    self.assertEqual(call_log, [])

    result = Chain(1).then(inner1).then(lambda x: 'unreachable').run()

    # The callable should have been called exactly once — when the outermost chain caught the signal
    self.assertEqual(call_log, ['evaluated'])
    # The callable's return value should be the chain's result
    self.assertEqual(result, 'lazy_result')

  async def test_break_callable_not_called_during_propagation(self) -> None:
    """break_(callable) in foreach: callable only called when caught, not when signal raised."""
    call_log: list[str] = []

    def tracked_callable():
      call_log.append('evaluated')
      return 'break_value'

    def mapper(x):
      if x == 2:
        return Chain.break_(tracked_callable)
      return x * 10

    self.assertEqual(call_log, [])

    result = Chain([1, 2, 3]).foreach(mapper).run()

    # Callable called exactly once when foreach caught the break signal
    self.assertEqual(call_log, ['evaluated'])
    # break_() appends evaluated value to partial results
    self.assertEqual(result, [10, 'break_value'])


# ---------------------------------------------------------------------------
# §16 — Unawaited coroutine: finally skipped (SPEC-261)
# ---------------------------------------------------------------------------


class UnawaitedCoroutineFinallySkippedTest(TestCase):
  """SPEC-261: When a sync chain produces a coroutine but it is never awaited,
  the finally handler is NOT executed because the async continuation never runs."""

  def test_unawaited_coroutine_finally_not_called(self) -> None:
    """Sync chain with async step + finally: not awaiting skips finally."""
    finally_called: list[bool] = []

    async def async_step(x):
      return x * 2

    def cleanup(rv):
      finally_called.append(True)

    chain = Chain(5).then(async_step).finally_(cleanup)
    result = chain.run()

    # run() returns a coroutine (async transition occurred)
    import asyncio

    self.assertTrue(asyncio.iscoroutine(result))

    # The finally handler has NOT been called — the async continuation was never entered
    self.assertEqual(finally_called, [])

    # Close the coroutine to avoid ResourceWarning
    result.close()

    # Confirm the finally handler still hasn't run
    self.assertEqual(finally_called, [])
