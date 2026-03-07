"""Tests for nested chain detection, value propagation, return/break
propagation through nesting levels, async behavior, and except/finally
interactions with nested chains.
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from quent._core import Link
from helpers import async_fn, async_identity


# ---------------------------------------------------------------------------
# Detection: Link.__init__ sets is_chain and is_nested
# ---------------------------------------------------------------------------

class TestNestedDetection(unittest.TestCase):

  def test_chain_as_step_detected(self):
    """When a Chain is passed to .then(), the wrapping Link.is_chain is True."""
    inner = Chain()
    outer = Chain().then(inner)
    self.assertTrue(outer.first_link.is_chain)

  def test_inner_is_nested_set(self):
    """Link.__init__ sets inner.is_nested = True on the Chain it wraps."""
    inner = Chain()
    self.assertFalse(inner.is_nested)
    Chain().then(inner)
    self.assertTrue(inner.is_nested)

  def test_frozen_not_detected_as_nested(self):
    """_FrozenChain does NOT have _is_chain, so Link.is_chain is False."""
    frozen = Chain().freeze()
    link = Link(frozen)
    self.assertFalse(link.is_chain)
    # Also verify _FrozenChain lacks the attribute entirely.
    self.assertFalse(hasattr(frozen, '_is_chain'))

  def test_deeply_nested_3_levels(self):
    """Three levels of nesting: all inner chains are detected."""
    level3 = Chain()
    level2 = Chain().then(level3)
    level1 = Chain().then(level2)

    # level1's first_link wraps level2 -- is_chain True
    self.assertTrue(level1.first_link.is_chain)
    # level2's first_link wraps level3 -- is_chain True
    self.assertTrue(level2.first_link.is_chain)
    # All inner chains are marked as nested
    self.assertTrue(level2.is_nested)
    self.assertTrue(level3.is_nested)


# ---------------------------------------------------------------------------
# Value propagation through nested chains
# ---------------------------------------------------------------------------

class TestNestedValuePropagation(unittest.TestCase):

  def test_inner_result_flows_out(self):
    """Inner chain's final value becomes the outer chain step's result."""
    result = (
      Chain(5)
      .then(Chain().then(lambda x: x * 2))
      .run()
    )
    self.assertEqual(result, 10)

  def test_inner_receives_current_value(self):
    """Nested chain receives the outer chain's current_value via _run(current_value, None, None)."""
    result = (
      Chain(5)
      .then(Chain().then(lambda x: x + 1))
      .run()
    )
    self.assertEqual(result, 6)

  def test_inner_with_ellipsis(self):
    """With ellipsis, inner chain receives Null: _run(Null, None, None)."""
    result = (
      Chain(5)
      .then(Chain().then(lambda: 99), ...)
      .run()
    )
    self.assertEqual(result, 99)

  def test_inner_with_explicit_args(self):
    """With explicit args, inner chain receives args[0]: _run(args[0], args[1:], kwargs)."""
    result = (
      Chain(5)
      .then(Chain().then(lambda x: x + 1), 10)
      .run()
    )
    self.assertEqual(result, 11)

  def test_inner_chain_with_root_link(self):
    """Inner chain with its own root_link evaluates that root first."""
    # Chain(100) has root_link=100, so inner._run(5, None, None) creates a
    # Link(5) whose next_link is the inner's first_link. The root_link (100)
    # is the chain's root_link, but when has_run_value is True, root_link is
    # overridden by the injected Link(v=5). So lambda receives 5.
    result = (
      Chain(5)
      .then(Chain(100).then(lambda x: x + 1))
      .run()
    )
    # The outer passes current_value=5 to inner._run(5, None, None).
    # In _run, has_run_value is True, so Link(5) is created. Its next_link
    # is set to self.first_link (lambda x: x+1). So: evaluate Link(5) -> 5,
    # then evaluate lambda(5) -> 6.
    self.assertEqual(result, 6)

  def test_inner_chain_result_replaces_outer_current_value(self):
    """The outer chain continues with the inner chain's result."""
    result = (
      Chain(5)
      .then(Chain().then(lambda x: x * 3))
      .then(lambda x: x + 1)
      .run()
    )
    # Inner: 5 * 3 = 15. Outer continues: 15 + 1 = 16.
    self.assertEqual(result, 16)


# ---------------------------------------------------------------------------
# Return propagation through nested chains
# ---------------------------------------------------------------------------

class TestNestedReturnPropagation(unittest.TestCase):

  def test_return_in_inner_exits_outer(self):
    """_Return raised in inner chain propagates to outer, skipping remaining steps."""
    result = (
      Chain(5)
      .then(Chain().then(lambda x: Chain.return_(42)))
      .then(lambda x: x + 100)
      .run()
    )
    self.assertEqual(result, 42)

  def test_return_with_value(self):
    """_Return with a specific value propagates correctly."""
    result = (
      Chain('hello')
      .then(Chain().then(lambda x: Chain.return_(x + ' world')))
      .then(lambda x: 'should not reach')
      .run()
    )
    self.assertEqual(result, 'hello world')

  def test_return_deeply_nested_3_levels(self):
    """_Return in the innermost of 3 levels exits all the way out."""
    level3 = Chain().then(lambda x: Chain.return_(999))
    level2 = Chain().then(level3)
    result = (
      Chain(5)
      .then(level2)
      .then(lambda x: x + 100)
      .run()
    )
    self.assertEqual(result, 999)

  def test_return_no_value_in_nested(self):
    """_Return with no value in a nested chain returns None from outer."""
    result = (
      Chain(5)
      .then(Chain().then(lambda x: Chain.return_()))
      .then(lambda x: 'should not reach')
      .run()
    )
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Break propagation through nested chains (within map)
# ---------------------------------------------------------------------------

class TestNestedBreakPropagation(unittest.TestCase):

  def test_break_in_nested_chain_as_then_step(self):
    """_Break in a nested chain (used as a .then() step, not map fn)
    propagates through is_nested re-raise to the outer chain's handler."""
    # When a Chain is passed to .then(), it's detected as nested.
    # _Break in the nested chain's _run re-raises because is_nested=True.
    # The outer chain's _run catches _Break but since the outer chain
    # is also not in a map context, it raises QuentException.
    inner = Chain().then(lambda x: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(inner).run()
    self.assertIn('Chain.break_() cannot be used outside of a map/foreach iteration', str(ctx.exception))

  def test_break_in_map_with_plain_fn(self):
    """Verify map with a plain lambda handles break correctly (baseline)."""
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: Chain.break_() if x == 3 else x)
      .run()
    )
    self.assertEqual(result, [1, 2])

  def test_break_in_chain_used_as_map_fn_raises(self):
    """A Chain passed to map() is called via run(), which traps _Break
    as a control flow signal escape -- raising QuentException."""
    # map(fn) extracts fn from Link(fn).v and calls fn(item).
    # When fn is a Chain, fn(item) = Chain.run(item), and run() catches
    # _ControlFlowSignal with QuentException.
    inner = Chain().then(lambda x: Chain.break_() if x == 3 else x)
    with self.assertRaises(QuentException) as ctx:
      Chain([1, 2, 3, 4, 5]).map(inner).run()
    self.assertIn('control flow signal escaped', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Async nested chains
# ---------------------------------------------------------------------------

class TestNestedAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_inner_chain(self):
    """Nested chain with an async step makes the whole chain async."""
    result = await (
      Chain(5)
      .then(Chain().then(async_fn))
      .run()
    )
    # async_fn(5) -> 6
    self.assertEqual(result, 6)

  async def test_sync_outer_async_inner(self):
    """Sync outer chain, async inner chain -- properly awaited."""
    result = await (
      Chain(10)
      .then(Chain().then(async_fn).then(lambda x: x * 2))
      .run()
    )
    # async_fn(10) -> 11, then 11 * 2 = 22
    self.assertEqual(result, 22)

  async def test_return_in_async_nested(self):
    """_Return in async nested chain propagates to outer chain."""
    result = await (
      Chain(5)
      .then(
        Chain()
        .then(async_identity)
        .then(lambda x: Chain.return_(x * 100))
      )
      .then(lambda x: 'should not reach')
      .run()
    )
    self.assertEqual(result, 500)

  async def test_async_nested_value_propagation(self):
    """Async nested chain passes its result back to the outer chain."""
    result = await (
      Chain(3)
      .then(Chain().then(async_fn))
      .then(lambda x: x + 10)
      .run()
    )
    # async_fn(3) -> 4, then 4 + 10 = 14
    self.assertEqual(result, 14)


# ---------------------------------------------------------------------------
# Except / finally interactions with nested chains
# ---------------------------------------------------------------------------

class TestNestedExceptFinally(unittest.TestCase):

  def test_inner_except_handles_first(self):
    """Inner chain's except_ catches its own error before the outer sees it."""
    outer_handler_called = []
    result = (
      Chain(5)
      .then(
        Chain()
        .then(lambda x: 1 / 0)
        .except_(lambda rv, exc: 'inner_caught')
      )
      .except_(lambda rv, exc: (outer_handler_called.append(True), 'outer_caught')[1])
      .run()
    )
    # Inner except_ catches the ZeroDivisionError and returns 'inner_caught'.
    # Outer chain never sees an exception.
    self.assertEqual(result, 'inner_caught')
    self.assertEqual(outer_handler_called, [])

  def test_inner_exception_propagates_to_outer_except(self):
    """Inner chain raises (no except_), outer except_ catches it."""
    result = (
      Chain(5)
      .then(
        Chain().then(lambda x: 1 / 0)
      )
      .except_(lambda rv, exc: f'outer_caught:{type(exc).__name__}')
      .run()
    )
    self.assertEqual(result, 'outer_caught:ZeroDivisionError')

  def test_inner_exception_type_preserved(self):
    """The specific exception type from the inner chain reaches the outer handler."""
    received = []

    def handler(rv, exc):
      received.append(exc)
      return 'handled'

    def raise_type_error(x):
      raise TypeError('inner type error')

    result = (
      Chain(5)
      .then(Chain().then(raise_type_error))
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 'handled')
    self.assertEqual(len(received), 1)
    self.assertIsInstance(received[0], TypeError)
    self.assertEqual(str(received[0]), 'inner type error')

  def test_nested_chain_does_not_trigger_outer_finally_early(self):
    """Inner chain execution does not trigger the outer chain's finally_ prematurely."""
    finally_tracker = []
    result = (
      Chain(5)
      .then(Chain().then(lambda x: x + 1))
      .then(lambda x: x + 10)
      .finally_(lambda rv: finally_tracker.append(rv))
      .run()
    )
    # Inner: 5+1=6. Outer continues: 6+10=16.
    self.assertEqual(result, 16)
    # finally_ is called exactly once, with root_value=5 (the initial value).
    self.assertEqual(len(finally_tracker), 1)
    self.assertEqual(finally_tracker[0], 5)


if __name__ == '__main__':
  unittest.main()
