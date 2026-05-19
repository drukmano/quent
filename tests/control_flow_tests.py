# SPDX-License-Identifier: MIT
"""Tests for SPEC §7 — Control Flow.

Covers:
- §7.1 Q.return_() — exits the **current** Q only (like Python `return`).
- §7.2 Q.break_() — propagates to nearest iteration scope (like labeled break).
- §7.5 Q.exit_() — propagates through ALL nesting/carve-outs to outermost run() (like sys.exit()).
- §7.3/7.4 priority + restrictions.
"""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Q, QuentException
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §7.2 Early Return — Q.return_()
# ---------------------------------------------------------------------------


class ReturnNoValueTest(SymmetricTestCase):
  """SPEC §7.2.1: return_() with no value → None."""

  async def test_return_no_value(self) -> None:
    """return_() with no value produces None."""
    result = Q(5).then(lambda x: Q.return_()).then(lambda x: x * 100).run()
    self.assertIsNone(result)

  async def test_return_no_value_async(self) -> None:
    """Async: return_() with no value produces None."""

    async def step(x):
      return Q.return_()

    result = await Q(5).then(step).then(lambda x: x * 100).run()
    self.assertIsNone(result)


class ReturnWithValueTest(SymmetricTestCase):
  """SPEC §7.2.1: return_() with values."""

  async def test_return_non_callable(self) -> None:
    """return_() with non-callable value returns as-is."""
    result = Q(5).then(lambda x: Q.return_(42)).then(lambda x: 'never').run()
    self.assertEqual(result, 42)

  async def test_return_callable(self) -> None:
    """return_() with callable: called when signal caught, return value becomes result."""
    result = Q(5).then(lambda x: Q.return_(lambda: 'from_fn')).then(lambda x: 'never').run()
    self.assertEqual(result, 'from_fn')

  async def test_return_callable_with_args(self) -> None:
    """return_() with callable + args follows calling conventions."""
    result = Q(5).then(lambda x: Q.return_(lambda a, b: a + b, 10, 20)).then(lambda x: 'never').run()
    self.assertEqual(result, 30)

  async def test_return_callable_with_ellipsis_as_arg(self) -> None:
    """return_() with callable + Ellipsis: Ellipsis passed as explicit arg."""
    result = Q(5).then(lambda x: Q.return_(lambda a: f'got {a}', ...)).then(lambda x: 'never').run()
    self.assertEqual(result, f'got {Ellipsis}')

  async def test_return_skips_remaining_steps(self) -> None:
    """Steps after return_() are skipped."""
    visited = []
    result = (
      Q(5)
      .then(lambda x: (visited.append(1), Q.return_(x * 2))[-1])
      .then(lambda x: (visited.append(2), x * 3)[-1])
      .then(lambda x: (visited.append(3), x * 4)[-1])
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(visited, [1])


# ---------------------------------------------------------------------------
# §7.1 Nested Q — return_() is absorbed at each Q boundary (NEW SEMANTIC)
# ---------------------------------------------------------------------------


class ReturnNestedLocalTest(SymmetricTestCase):
  """SPEC §7.1: return_() returns from the current Q only.

  Each Q boundary absorbs its own Q.return_(). The nested Q's return value
  flows to the outer pipeline as the nested step's result; the outer chain
  continues. (Like Python's `return` exits the current function, not its caller.)
  """

  async def test_return_in_nested_q_stays_local(self) -> None:
    """return_() in nested Q exits the nested Q only; value flows to outer chain which continues."""
    inner = Q().then(lambda x: Q.return_('early') if x > 3 else x)
    # Outer chain continues with 'early' as the nested step's result.
    result = Q(5).then(inner).then(lambda x: f'outer_saw:{x}').run()
    self.assertEqual(result, 'outer_saw:early')

  async def test_return_in_deep_nesting_each_boundary_absorbs(self) -> None:
    """In deep nesting, return_() only exits the innermost Q.

    inner2 raises return_; inner2 absorbs it (its result becomes 'deep');
    inner1 receives 'deep' as inner2's step result and passes it through;
    outer receives 'deep' and the .then(lambda) continues.
    """
    inner2 = Q().then(lambda x: Q.return_('deep'))
    inner1 = Q().then(inner2)
    result = Q(1).then(inner1).then(lambda x: f'outer:{x}').run()
    self.assertEqual(result, 'outer:deep')

  async def test_return_in_nested_no_match_passes_through(self) -> None:
    """When return_() condition not met, value passes through normally."""
    inner = Q().then(lambda x: Q.return_('early') if x > 10 else x * 2)
    result = Q(3).then(inner).then(lambda x: x + 1).run()
    self.assertEqual(result, 7)  # 3*2=6, 6+1=7

  async def test_return_in_nested_chain_value_consumed_by_outer(self) -> None:
    """return_() in nested Q with a value: outer step receives the returned value as its input."""
    inner = Q().then(lambda x: x * 10).then(lambda x: Q.return_(x + 1))
    # inner: 5 * 10 = 50, return_(51).  inner.run() = 51.  Outer receives 51 → 51 * 2 = 102.
    result = Q(5).then(inner).then(lambda x: x * 2).run()
    self.assertEqual(result, 102)


# ---------------------------------------------------------------------------
# §7.5 Q.exit_() — hard exit through ALL nesting and carve-outs
# ---------------------------------------------------------------------------


class ExitFromNestingTest(SymmetricTestCase):
  """SPEC §7.5: Q.exit_() propagates through every Q boundary; absorbed only at outermost run()."""

  async def test_exit_exits_outermost_chain(self) -> None:
    """exit_() in nested chain exits the outermost chain."""
    inner = Q().then(lambda x: Q.exit_('early') if x > 3 else x)
    result = Q(5).then(inner).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'early')

  async def test_exit_in_deep_nesting(self) -> None:
    """exit_() propagates through multiple nesting levels straight to outermost run()."""
    inner2 = Q().then(lambda x: Q.exit_('deep'))
    inner1 = Q().then(inner2)
    result = Q(1).then(inner1).then(lambda x: 'unreachable').run()
    self.assertEqual(result, 'deep')

  async def test_exit_no_value(self) -> None:
    """exit_() with no value produces None at the outermost run()."""
    inner = Q().then(lambda x: Q.exit_())
    result = Q(5).then(inner).then(lambda x: 'unreachable').run()
    self.assertIsNone(result)

  async def test_exit_non_callable_value(self) -> None:
    """exit_(42) — non-callable value returned as-is."""
    inner = Q().then(lambda x: Q.exit_(42))
    result = Q(5).then(inner).then(lambda x: 'never').run()
    self.assertEqual(result, 42)

  async def test_exit_callable_value(self) -> None:
    """exit_(callable) — callable invoked at outermost run() catch frame; return value becomes result."""
    inner = Q().then(lambda x: Q.exit_(lambda: 'lazy_exit'))
    result = Q(5).then(inner).then(lambda x: 'never').run()
    self.assertEqual(result, 'lazy_exit')

  async def test_exit_callable_with_args(self) -> None:
    """exit_(callable, *args) — args passed explicitly (calling convention)."""
    inner = Q().then(lambda x: Q.exit_(lambda a, b: a + b, 10, 20))
    result = Q(5).then(inner).then(lambda x: 'never').run()
    self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# §7.2.3 Restrictions
# ---------------------------------------------------------------------------


class ReturnRestrictionsTest(TestCase):
  """SPEC §7.2.3: return_() restrictions."""

  def test_return_in_except_raises_quent_exception(self) -> None:
    """return_() in except handler raises QuentException."""

    def handler(info):
      return Q.return_('bad')

    c = Q(1).then(lambda x: 1 / 0).except_(handler)
    with self.assertRaises(QuentException):
      c.run()

  def test_return_in_finally_raises_quent_exception(self) -> None:
    """return_() in finally handler raises QuentException."""

    def cleanup(rv):
      return Q.return_('bad')

    c = Q(1).finally_(cleanup)
    with self.assertRaises(QuentException):
      c.run()

  def test_return_in_top_level_chain_extracts_value(self) -> None:
    """return_() in top-level chain: run() catches signal and extracts value."""
    c = Q(1).then(lambda x: Q.return_(x))
    result = c.run()
    # run() catches _Return and returns the carried value.
    self.assertEqual(result, 1)


# ---------------------------------------------------------------------------
# §7.3 Break — Q.break_()
# ---------------------------------------------------------------------------


class BreakNoValueTest(SymmetricTestCase):
  """SPEC §7.3.1: break_() with no value → partial results preserved."""

  async def test_break_no_value_preserves_partial(self) -> None:
    """break_() with no value: results collected so far are preserved."""
    result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_() if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4])  # items before break

  async def test_break_no_value_async(self) -> None:
    """Async break_() with no value: partial results preserved."""

    async def mapper(x):
      if x == 3:
        return Q.break_()
      return x * 2

    result = await Q([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4])


class BreakWithValueTest(SymmetricTestCase):
  """SPEC §7.3.1: break_() with value appends to partial results."""

  async def test_break_with_value_appends_to_results(self) -> None:
    """break_() with value: appends to partial results."""
    result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_(x * 10) if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4, 30])

  async def test_break_with_value_appends_async(self) -> None:
    """Async break_() with value appends to partial results."""

    async def mapper(x):
      if x == 3:
        return Q.break_(x * 10)
      return x * 2

    result = await Q([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4, 30])


class BreakCallableValueTest(SymmetricTestCase):
  """SPEC §7.3.1: break_() with callable values."""

  async def test_break_callable_called_when_caught(self) -> None:
    """break_() with callable: called when signal is caught, appended to partial."""
    result = Q([1, 2, 3]).foreach(lambda x: Q.break_(lambda: 'stop') if x == 2 else x).run()
    self.assertEqual(result, [1, 'stop'])

  async def test_break_callable_with_args(self) -> None:
    """break_() with callable + args follows calling conventions."""
    result = Q([1, 2, 3]).foreach(lambda x: Q.break_(lambda a, b: a + b, 10, 20) if x == 2 else x).run()
    self.assertEqual(result, [1, 30])


# ---------------------------------------------------------------------------
# §7.3.2 Outside Iteration
# ---------------------------------------------------------------------------


class BreakOutsideIterationTest(TestCase):
  """SPEC §7.3.2: break_() outside iteration."""

  def test_break_outside_iteration_raises(self) -> None:
    """break_() outside foreach/foreach_do raises QuentException."""
    c = Q(1).then(lambda x: Q.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('cannot be used outside', str(ctx.exception))

  def test_break_outside_iteration_specific_message(self) -> None:
    """break_() outside iteration produces the exact specified error message."""
    c = Q(1).then(lambda x: Q.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    expected = (
      'Q.break_() cannot be used outside of a loop or iteration context'
      ' (foreach, foreach_do, iterate, iterate_do, flat_iterate, flat_iterate_do, while_).'
    )
    self.assertEqual(str(ctx.exception), expected)


# ---------------------------------------------------------------------------
# §7.3.3 In Except/Finally Handlers
# ---------------------------------------------------------------------------


class BreakInHandlersTest(TestCase):
  """SPEC §7.3.3: break_() in except/finally handlers."""

  def test_break_in_except_raises_quent_exception(self) -> None:
    """break_() in except handler raises QuentException."""

    def handler(info):
      return Q.break_()

    c = Q(1).then(lambda x: 1 / 0).except_(handler)
    with self.assertRaises(QuentException):
      c.run()

  def test_break_in_finally_raises_quent_exception(self) -> None:
    """break_() in finally handler raises QuentException."""

    def cleanup(rv):
      return Q.break_()

    c = Q(1).finally_(cleanup)
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
    result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_() if x == 3 else x * 2, concurrency=2).run()
    self.assertEqual(result, [2, 4])

  async def test_concurrent_break_with_value(self) -> None:
    """Concurrent break with value: appends to truncated results."""
    result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_('stopped') if x == 3 else x * 2, concurrency=2).run()
    self.assertEqual(result, [2, 4, 'stopped'])

  async def test_concurrent_break_earliest_index_wins(self) -> None:
    """When multiple workers break, earliest index wins."""
    # All elements >= 2 break. Element at index 1 (value 2) is earliest.
    result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_() if x >= 2 else x * 2, concurrency=5).run()
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
        return Q.return_('return wins')
      if x == 2:
        return Q.break_('break')
      if x == 3:
        raise ValueError('error')
      return x

    result = Q([1, 2, 3, 4]).foreach(mixed, concurrency=4).run()
    self.assertEqual(result, 'return wins')

  async def test_break_over_regular_exception(self) -> None:
    """break_() takes priority over regular exceptions."""

    def mixed(x):
      if x == 1:
        return Q.break_('break wins')
      if x == 3:
        raise ValueError('error')
      return x

    result = Q([0, 1, 2, 3]).foreach(mixed, concurrency=4).run()
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
        return Q.break_('break wins')
      return x

    result = Q([0, 1, 2, 3, 4]).foreach(mixed, concurrency=5).run()
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
        return Q.break_()
      side_effects.append(x)

    result = Q([1, 2, 3, 4, 5]).foreach_do(effect).run()
    self.assertEqual(result, [1, 2])
    self.assertEqual(side_effects, [1, 2])

  async def test_break_in_foreach_do_with_value(self) -> None:
    """break_() in foreach_do with value: appends to partial items."""
    result = Q([1, 2, 3, 4, 5]).foreach_do(lambda x: Q.break_('done') if x == 3 else None).run()
    self.assertEqual(result, [1, 2, 'done'])


# ---------------------------------------------------------------------------
# Async return in nested chains
# ---------------------------------------------------------------------------


class AsyncReturnTest(SymmetricTestCase):
  """Async return_() behavior."""

  async def test_async_return_in_then(self) -> None:
    """Async step with return_() exits the (top-level) chain — current Q is the only Q."""

    async def step(x):
      return Q.return_(x * 10)

    result = await Q(5).then(step).then(lambda x: 'never').run()
    self.assertEqual(result, 50)

  async def test_async_return_in_nested_q_stays_local(self) -> None:
    """Async return_() in nested Q exits the nested Q only; outer chain continues."""

    async def inner_step(x):
      return Q.return_('async early')

    inner = Q().then(inner_step)
    # Outer chain receives 'async early' as nested step's result and continues.
    result = await Q(1).then(inner).then(lambda x: f'outer:{x}').run()
    self.assertEqual(result, 'outer:async early')

  async def test_async_break_outside_iteration(self) -> None:
    """Async break_() outside iteration raises QuentException."""

    async def step(x):
      return Q.break_()

    c = Q(1).then(step)
    with self.assertRaises(QuentException):
      await c.run()


# ---------------------------------------------------------------------------
# §5.5 break_() in gather → QuentException
# ---------------------------------------------------------------------------


class BreakInGatherTest(SymmetricTestCase):
  """SPEC §5.5: break_() signals are not allowed in gather operations."""

  async def test_break_in_gather_raises_quent_exception(self) -> None:
    """break_() in gather raises QuentException with exact message (§5.5)."""
    c = Q(5).gather(lambda x: Q.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('break_() signals are not allowed in gather operations', str(ctx.exception))

  async def test_break_in_gather_async(self) -> None:
    """Async break_() in gather raises QuentException with exact message (§5.5)."""

    async def fn(x):
      return Q.break_()

    c = Q(5).gather(fn)
    with self.assertRaises(QuentException) as ctx:
      await c.run()
    self.assertIn('break_() signals are not allowed in gather operations', str(ctx.exception))

  async def test_break_in_gather_multiple_fns(self) -> None:
    """break_() in one of multiple gather fns raises QuentException."""
    c = Q(5).gather(lambda x: x * 2, lambda x: Q.break_(), lambda x: x + 1)
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('break_() signals are not allowed in gather operations', str(ctx.exception))


# ---------------------------------------------------------------------------
# Async control flow: return_() from async step in various positions
# ---------------------------------------------------------------------------


class AsyncReturnFromNestedChainTest(SymmetricTestCase):
  """Async return_() in deeply nested chains — each Q absorbs its own return_() (§7.1)."""

  async def test_async_return_from_deeply_nested_chain_stays_local(self) -> None:
    """Async return_() in innermost Q only ends inner; the value flows outward step-by-step.

    Per §7.1: each Q absorbs its own _Return.  inner2.run() returns 'deep_async';
    inner1 receives 'deep_async' as inner2's step result and passes it through;
    outer receives 'deep_async' and continues to the .then(lambda).
    """

    async def deep_step(x):
      return Q.return_('deep_async')

    inner2 = Q().then(deep_step)
    inner1 = Q().then(inner2)
    result = await Q(1).then(inner1).then(lambda x: f'outer:{x}').run()
    self.assertEqual(result, 'outer:deep_async')


class AsyncExitFromNestedChainTest(IsolatedAsyncioTestCase):
  """SPEC §7.5: Async Q.exit_() in deeply nested chains propagates to outermost run()."""

  async def test_async_exit_from_deeply_nested_chain(self) -> None:
    """Async Q.exit_() propagates through deeply nested chains to outermost run()."""

    async def deep_step(x):
      return Q.exit_('deep_async')

    inner2 = Q().then(deep_step)
    inner1 = Q().then(inner2)
    result = await Q(1).then(inner1).then(lambda x: 'unreachable').run()
    self.assertEqual(result, 'deep_async')


class AsyncBreakInMapTest(SymmetricTestCase):
  """Async break_() in map operations."""

  async def test_async_break_in_map_no_value(self) -> None:
    """Async break_() in map: partial results preserved."""

    async def mapper(x):
      if x == 3:
        return Q.break_()
      return x * 2

    result = await Q([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4])

  async def test_async_break_in_map_with_value(self) -> None:
    """Async break_() in map with value: appends to partial results."""

    async def mapper(x):
      if x == 3:
        return Q.break_('stopped')
      return x * 2

    result = await Q([1, 2, 3, 4, 5]).foreach(mapper).run()
    self.assertEqual(result, [2, 4, 'stopped'])

  async def test_async_break_in_foreach_do(self) -> None:
    """Async break_() in foreach_do: partial original elements preserved."""
    side_effects = []

    async def effect(x):
      if x == 3:
        return Q.break_()
      side_effects.append(x)

    result = await Q([1, 2, 3, 4, 5]).foreach_do(effect).run()
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
        return Q.return_('early_exit')
      return x * 2

    result = Q([1, 2, 3, 4, 5]).foreach(mapper).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'early_exit')

  async def test_return_in_foreach_exits_chain_async(self) -> None:
    """Async: return_() in foreach callback exits loop and returns value from chain."""

    async def mapper(x):
      if x == 3:
        return Q.return_('async_early_exit')
      return x * 2

    result = await Q([1, 2, 3, 4, 5]).foreach(mapper).then(lambda x: 'nope').run()
    self.assertEqual(result, 'async_early_exit')

  async def test_return_in_foreach_do_exits_chain_sync(self) -> None:
    """Sync: return_() in foreach_do callback exits loop and chain."""

    def effect(x):
      if x == 3:
        return Q.return_('early from foreach_do')

    result = Q([1, 2, 3, 4, 5]).foreach_do(effect).then(lambda x: 'nope').run()
    self.assertEqual(result, 'early from foreach_do')

  async def test_return_in_foreach_do_exits_chain_async(self) -> None:
    """Async: return_() in foreach_do callback exits loop and chain."""

    async def effect(x):
      if x == 3:
        return Q.return_('async_early_foreach_do')

    result = await Q([1, 2, 3, 4, 5]).foreach_do(effect).then(lambda x: 'nope').run()
    self.assertEqual(result, 'async_early_foreach_do')


class ReturnInNestedChainWithinForeachTest(SymmetricTestCase):
  """SPEC §7.1: return_() in nested Q used as foreach callback stays local.

  Each Q boundary absorbs its own return_().  The nested q.run() catches
  _Return and yields the value as the element's foreach result; the outer
  foreach continues iterating.
  """

  async def test_return_in_nested_chain_within_foreach_sync(self) -> None:
    """Sync: return_() in nested chain becomes element result, does not exit outer chain."""
    inner = Q().then(lambda x: Q.return_('nested_return') if x == 3 else x * 2)
    result = Q([1, 2, 3, 4, 5]).foreach(inner).run()
    self.assertEqual(result, [2, 4, 'nested_return', 8, 10])

  async def test_return_in_nested_chain_within_foreach_async(self) -> None:
    """Async: return_() in nested chain becomes element result, does not exit outer chain."""

    async def inner_step(x):
      if x == 3:
        return Q.return_('async_nested_return')
      return x * 2

    inner = Q().then(inner_step)
    result = await Q([1, 2, 3, 4, 5]).foreach(inner).run()
    self.assertEqual(result, [2, 4, 'async_nested_return', 8, 10])


class ExitInForeachTest(SymmetricTestCase):
  """SPEC §7.5: Q.exit_() inside foreach bypasses iteration scope and exits outermost run()."""

  async def test_exit_in_foreach_mapper_exits_outermost(self) -> None:
    """Sync: exit_() in foreach mapper exits the outermost run() — not just the loop."""

    def mapper(x):
      if x == 3:
        return Q.exit_('full_exit')
      return x * 2

    # .then() after foreach is NEVER reached — exit_ bypasses everything.
    result = Q([1, 2, 3, 4, 5]).foreach(mapper).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'full_exit')

  async def test_exit_in_plain_fn_inside_foreach_exits_outermost(self) -> None:
    """A plain (non-Q) callback raising Q.exit_() propagates all the way to outermost run()."""

    def mapper(x):
      return Q.exit_('full_exit') if x == 3 else x * 2

    result = Q([1, 2, 3, 4, 5]).foreach(mapper).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'full_exit')

  async def test_exit_in_direct_q_foreach_callback_absorbed_at_inner_run(self) -> None:
    """Direct-Q callback acts like §4.2 lambda-wrapping case.

    When foreach receives a Q directly, it invokes ``inner(item)`` which is
    ``inner.run(item)`` — the outermost run() from inner's perspective.
    ``inner.run()`` absorbs Q.exit_() locally. Use a plain function or
    `.then(inner)` registration to propagate Q.exit_() across nesting.
    """
    inner = Q().then(lambda x: Q.exit_('local_exit') if x == 3 else x * 2)
    result = Q([1, 2, 3, 4, 5]).foreach(inner).run()
    # foreach gets ['local_exit'] returned by inner.run() at x=3.
    self.assertEqual(result, [2, 4, 'local_exit', 8, 10])


class ExitWithCarveOutsTest(SymmetricTestCase):
  """SPEC §7.5/7.4: Q.exit_() bypasses except_/finally_ traps and gather carve-out."""

  async def test_exit_runs_finally_during_propagation(self) -> None:
    """exit_() triggers all finally_ handlers as it propagates outward."""
    log: list[str] = []

    def finally_inner(_rv):
      log.append('inner_finally')

    def finally_outer(_rv):
      log.append('outer_finally')

    inner = Q().then(lambda x: Q.exit_('done')).finally_(finally_inner)
    result = Q(1).then(inner).then(lambda x: 'never').finally_(finally_outer).run()
    self.assertEqual(result, 'done')
    # Both finally handlers ran — inner first (closer to signal site), then outer.
    self.assertEqual(log, ['inner_finally', 'outer_finally'])

  async def test_exit_bypasses_except_(self) -> None:
    """exit_() is NOT trapped by an outer except_() — it propagates through."""

    def handler(_info):
      # Should NEVER be called for exit_ — it bypasses except.
      return 'handler_ran'

    result = Q(1).then(lambda x: Q.exit_('exit_value')).except_(handler).then(lambda x: 'never').run()
    self.assertEqual(result, 'exit_value')


class ControlFlowInConcurrentForeachTest(SymmetricTestCase):
  """Control flow signals in concurrent foreach."""

  async def test_concurrent_return_exits_chain(self) -> None:
    """Concurrent foreach: return_() signal exits chain entirely."""

    def mapper(x):
      if x == 3:
        return Q.return_('concurrent_return')
      return x * 2

    result = Q([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, 'concurrent_return')

  async def test_concurrent_break_truncates(self) -> None:
    """Concurrent foreach: break_() truncates results to items before break index."""

    def mapper(x):
      if x == 3:
        return Q.break_()
      return x * 2

    result = Q([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, [2, 4])

  async def test_concurrent_break_with_value(self) -> None:
    """Concurrent foreach: break_(value) appends value to truncated results."""

    def mapper(x):
      if x == 3:
        return Q.break_('stopped')
      return x * 2

    result = Q([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, [2, 4, 'stopped'])

  async def test_concurrent_async_return_exits_chain(self) -> None:
    """Async concurrent foreach: return_() exits chain."""

    async def mapper(x):
      if x == 3:
        return Q.return_('async_concurrent_return')
      return x * 2

    result = await Q([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, 'async_concurrent_return')

  async def test_concurrent_async_break_truncates(self) -> None:
    """Async concurrent foreach: break_() truncates results."""

    async def mapper(x):
      if x == 3:
        return Q.break_()
      return x * 2

    result = await Q([1, 2, 3, 4, 5]).foreach(mapper, concurrency=3).run()
    self.assertEqual(result, [2, 4])


# ---------------------------------------------------------------------------
# §7.3: break_() in deeply nested async q
# ---------------------------------------------------------------------------


class AsyncNestedBreakPropagationTest(IsolatedAsyncioTestCase):
  """§7.3: break_() in deeply nested async pipeline propagates through _run_async."""

  async def test_async_break_in_nested_chain_propagates_through_run_async(self) -> None:
    """Deeply nested async break propagation."""

    async def async_step(x):
      return x

    chain_b = Q().then(async_step).then(lambda x: Q.break_() if x >= 2 else x)
    chain_a = Q().then(chain_b)
    chain_wrapper = Q().then(chain_a)

    # foreach_do catches the QuentException from the outermost q
    with self.assertRaises(QuentException) as ctx:
      await Q([1, 2, 3]).foreach_do(chain_wrapper).run()
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
      return Q.return_(compute)

    result = await Q(5).then(step).then(lambda x: 'never').run()
    self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# §7.1 Lazy Evaluation of Signal Values (SPEC-201)
# ---------------------------------------------------------------------------


class ReturnLazyEvaluationTest(SymmetricTestCase):
  """SPEC §7.1: Signal values are lazily evaluated — callable only invoked when caught."""

  async def test_return_callable_evaluated_at_current_q_boundary(self) -> None:
    """return_(callable) in nested Q: callable invoked at the **inner Q's** catch frame.

    Under new semantic (§7.1), return_() is absorbed at the **innermost** Q boundary.
    The lazy callable is evaluated once, there, and the resulting value flows to the
    outer pipeline as the nested step's result.
    """
    call_log: list[str] = []

    def tracked_callable():
      call_log.append('evaluated')
      return 'lazy_result'

    inner = Q().then(lambda x: Q.return_(tracked_callable))

    # Before running, callable not called
    self.assertEqual(call_log, [])

    result = Q(1).then(inner).then(lambda x: f'outer:{x}').run()

    # Callable invoked exactly once at the inner Q's _Return catch frame.
    self.assertEqual(call_log, ['evaluated'])
    # The value flows to the outer pipeline as inner's step result; outer continues.
    self.assertEqual(result, 'outer:lazy_result')

  async def test_exit_callable_not_called_during_propagation(self) -> None:
    """exit_(callable) in nested chain: callable NOT called during propagation, only when caught.

    Per §7.5, Q.exit_() propagates through every Q boundary to the outermost run()'s
    catch frame, where the lazy callable is evaluated exactly once.
    """
    call_log: list[str] = []

    def tracked_callable():
      call_log.append('evaluated')
      return 'lazy_result'

    # Build 3-level nesting: inner2 raises exit_, inner1 wraps it, outer catches it.
    inner2 = Q().then(lambda x: Q.exit_(tracked_callable))
    inner1 = Q().then(inner2)

    # Before running, callable not called
    self.assertEqual(call_log, [])

    result = Q(1).then(inner1).then(lambda x: 'unreachable').run()

    # The callable should have been called exactly once — when the outermost q caught the signal
    self.assertEqual(call_log, ['evaluated'])
    # The callable's return value should be the pipeline's result
    self.assertEqual(result, 'lazy_result')

  async def test_break_callable_not_called_during_propagation(self) -> None:
    """break_(callable) in foreach: callable only called when caught, not when signal raised."""
    call_log: list[str] = []

    def tracked_callable():
      call_log.append('evaluated')
      return 'break_value'

    def mapper(x):
      if x == 2:
        return Q.break_(tracked_callable)
      return x * 10

    self.assertEqual(call_log, [])

    result = Q([1, 2, 3]).foreach(mapper).run()

    # Callable called exactly once when foreach caught the break signal
    self.assertEqual(call_log, ['evaluated'])
    # break_() appends evaluated value to partial results
    self.assertEqual(result, [10, 'break_value'])


# ---------------------------------------------------------------------------
# §17 — Unawaited coroutine: finally skipped (SPEC-261)
# ---------------------------------------------------------------------------


class UnawaitedCoroutineFinallySkippedTest(TestCase):
  """SPEC-261: When a sync pipeline produces a coroutine but it is never awaited,
  the finally handler is NOT executed because the async continuation never runs."""

  def test_unawaited_coroutine_finally_not_called(self) -> None:
    """Sync pipeline with async step + finally: not awaiting skips finally."""
    finally_called: list[bool] = []

    async def async_step(x):
      return x * 2

    def cleanup(rv):
      finally_called.append(True)

    q = Q(5).then(async_step).finally_(cleanup)
    result = q.run()

    # run() returns a coroutine (async transition occurred)
    import asyncio

    self.assertTrue(asyncio.iscoroutine(result))

    # The finally handler has NOT been called — the async continuation was never entered
    self.assertEqual(finally_called, [])

    # Close the coroutine to avoid ResourceWarning
    result.close()

    # Confirm the finally handler still hasn't run
    self.assertEqual(finally_called, [])


class ConcurrentExitRegressionTest(SymmetricTestCase):
  """SPEC §7.5 + §7.4: Q.exit_() bypasses every iteration carve-out, including concurrent foreach/foreach_do.

  Regression for previously-unimplemented behavior: ``_triage_iter_exceptions`` was
  wrapping unknown ``_ControlFlowSignal`` subclasses as ``QuentException``, which
  swallowed ``_Exit``.  The spec mandates propagation through every level.
  """

  async def test_concurrent_foreach_exit_propagates_to_outermost(self) -> None:
    def mapper(x):
      return Q.exit_('exit_from_concurrent') if x == 3 else x * 2

    result = Q([1, 2, 3, 4, 5]).foreach(mapper, concurrency=2).then(lambda x: 'unreached').run()
    self.assertEqual(result, 'exit_from_concurrent')

  async def test_concurrent_foreach_do_exit_propagates_to_outermost(self) -> None:
    def effect(x):
      if x == 3:
        return Q.exit_('exit_from_do')
      return None

    result = Q([1, 2, 3, 4, 5]).foreach_do(effect, concurrency=2).then(lambda x: 'unreached').run()
    self.assertEqual(result, 'exit_from_do')


class SyncPipelineAsyncFinallyAbsorbedReturnRegressionTest(TestCase):
  """SPEC §6.2 + §7.1: sync pipeline + async finally_ + absorbed Q.return_().

  Regression: ``_run_sync_finally_dispatch`` was treating ``_active_exc != None``
  as "re-raise after finally", which incorrectly re-raised an *absorbed* _Return.
  Fix: dispatch now distinguishes "pipeline_result set ⇒ chain-only" from
  "pipeline_result is Null ⇒ re-raise" (the §6.2 finally-context-chain semantic).
  """

  def test_sync_return_with_async_finally_returns_absorbed_value(self) -> None:
    import asyncio

    log: list[str] = []

    async def async_finally(_rv: object) -> None:
      log.append('finally_ran')

    def early(_x: object) -> object:
      return Q.return_('absorbed')

    coro = Q(1).then(early).then(lambda x: 'unreached').finally_(async_finally).run()
    self.assertTrue(asyncio.iscoroutine(coro))
    result = asyncio.run(coro)
    self.assertEqual(result, 'absorbed')
    self.assertEqual(log, ['finally_ran'])


class LazyValueSignalMisuseTest(SymmetricTestCase):
  """SPEC §7.1 / §7.2 / §7.5: control-flow signals raised inside a lazy value are misuse → QuentException.

  Regression: ``_handle_*_exc`` helpers were not catching ``_ControlFlowSignal`` from
  the lazy callable, so a raw ``_Return``/``_Break``/``_Exit`` could leak past the catch frame.
  """

  async def test_return_lazy_callable_raising_signal_wraps_as_quent_exception(self) -> None:
    def evil() -> object:
      return Q.return_('evil')

    with self.assertRaises(QuentException) as ctx:
      Q(1).then(lambda x: Q.return_(evil)).run()
    self.assertIn('lazy value raised', str(ctx.exception))

  async def test_break_lazy_callable_raising_signal_wraps_as_quent_exception(self) -> None:
    def evil() -> object:
      return Q.return_('evil')

    with self.assertRaises(QuentException) as ctx:
      Q([1, 2, 3]).foreach(lambda x: Q.break_(evil) if x == 2 else x).run()
    self.assertIn('lazy value raised', str(ctx.exception))

  async def test_exit_lazy_callable_raising_signal_wraps_as_quent_exception(self) -> None:
    def evil() -> object:
      return Q.return_('evil')

    with self.assertRaises(QuentException) as ctx:
      Q(1).then(lambda x: Q.exit_(evil)).run()
    self.assertIn('lazy value raised', str(ctx.exception))


class ExitInIterateRegressionTest(SymmetricTestCase):
  """SPEC §17.3: Q.exit_() during deferred iteration yields the value as one final item, then stops.

  Same semantic as Q.return_() in iterate (§17.3) — there's no outermost run()
  during deferred iteration to absorb the exit.  The user-friendly interpretation
  is "yield and stop", consistent with how Q.return_() is handled.

  Regression: ``_generator.py`` previously caught only ``(_Break, _Return)``
  and let raw ``_Exit`` leak out of ``__iter__``/``__aiter__``.
  """

  async def test_exit_in_iterate_yields_value_and_stops(self) -> None:
    def step(x):
      return Q.exit_('end_value') if x == 3 else x * 10

    result = []
    async for item in Q([1, 2, 3, 4, 5]).iterate(step):
      result.append(item)
    self.assertEqual(result, [10, 20, 'end_value'])

  async def test_exit_no_value_in_iterate_stops_without_yield(self) -> None:
    def step(x):
      if x == 2:
        return Q.exit_()
      return x

    result = []
    async for item in Q([1, 2, 3, 4]).iterate(step):
      result.append(item)
    self.assertEqual(result, [1])

  async def test_exit_in_iterate_do_yields_value_and_stops(self) -> None:
    seen: list[int] = []

    def effect(x):
      seen.append(x)
      if x == 3:
        return Q.exit_('do_end')
      return None

    result = []
    async for item in Q([1, 2, 3, 4, 5]).iterate_do(effect):
      result.append(item)
    # iterate_do yields original items; exit value is the final yielded element.
    self.assertEqual(result, [1, 2, 'do_end'])
    self.assertEqual(seen, [1, 2, 3])
