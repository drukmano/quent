"""Exhaustive async combinatorial tests for Chain.if_() and Chain.else_().

Tests every combination of:
  Predicate (4): sync-truthy, sync-falsy, async-truthy, async-falsy
  fn (4): sync callable, async callable, plain value, nested Chain
  else_ (3): none, sync callable, async callable

Total matrix: 4 x 4 x 3 = 48 combinations, plus 12+ async-specific edge cases.
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------

def pred_sync_true(v):
  return v > 5

def pred_sync_false(v):
  return v > 100

async def pred_async_true(v):
  return v > 5

async def pred_async_false(v):
  return v > 100


# ---------------------------------------------------------------------------
# fn helpers
# ---------------------------------------------------------------------------

def fn_sync(v):
  return v * 2

async def fn_async(v):
  return v * 2

FN_VALUE = 42


# ---------------------------------------------------------------------------
# else helpers
# ---------------------------------------------------------------------------

def else_sync(v):
  return v + 100

async def else_async(v):
  return v + 100


# ---------------------------------------------------------------------------
# Test value
# ---------------------------------------------------------------------------

V = 10

# Expected results:
#   truthy pred + F_SYNC  -> 20
#   truthy pred + F_ASYNC -> 20
#   truthy pred + F_VALUE -> 42
#   truthy pred + F_CHAIN -> 11
#   falsy  pred + E_NONE  -> 10  (passthrough)
#   falsy  pred + E_SYNC  -> 110
#   falsy  pred + E_ASYNC -> 110


class IfElseAsyncMatrixTests(unittest.IsolatedAsyncioTestCase):
  """48-combination matrix: predicate x fn x else_ across sync/async."""

  # =========================================================================
  # Sync predicate, truthy (P_SYNC_T)
  # =========================================================================

  # --- fn: sync ---

  async def test_sync_pred_truthy__sync_fn__no_else(self):
    r = Chain(V).if_(pred_sync_true, then=fn_sync).run()
    self.assertEqual(r, 20)

  async def test_sync_pred_truthy__sync_fn__sync_else(self):
    r = Chain(V).if_(pred_sync_true, then=fn_sync).else_(else_sync).run()
    self.assertEqual(r, 20)

  async def test_sync_pred_truthy__sync_fn__async_else(self):
    r = Chain(V).if_(pred_sync_true, then=fn_sync).else_(else_async).run()
    self.assertEqual(r, 20)

  # --- fn: async ---

  async def test_sync_pred_truthy__async_fn__no_else(self):
    r = await Chain(V).if_(pred_sync_true, then=fn_async).run()
    self.assertEqual(r, 20)

  async def test_sync_pred_truthy__async_fn__sync_else(self):
    r = await Chain(V).if_(pred_sync_true, then=fn_async).else_(else_sync).run()
    self.assertEqual(r, 20)

  async def test_sync_pred_truthy__async_fn__async_else(self):
    r = await Chain(V).if_(pred_sync_true, then=fn_async).else_(else_async).run()
    self.assertEqual(r, 20)

  # --- fn: plain value ---

  async def test_sync_pred_truthy__value_fn__no_else(self):
    r = Chain(V).if_(pred_sync_true, then=FN_VALUE).run()
    self.assertEqual(r, 42)

  async def test_sync_pred_truthy__value_fn__sync_else(self):
    r = Chain(V).if_(pred_sync_true, then=FN_VALUE).else_(else_sync).run()
    self.assertEqual(r, 42)

  async def test_sync_pred_truthy__value_fn__async_else(self):
    r = Chain(V).if_(pred_sync_true, then=FN_VALUE).else_(else_async).run()
    self.assertEqual(r, 42)

  # --- fn: nested Chain ---

  async def test_sync_pred_truthy__chain_fn__no_else(self):
    r = Chain(V).if_(pred_sync_true, then=Chain().then(lambda v: v + 1)).run()
    self.assertEqual(r, 11)

  async def test_sync_pred_truthy__chain_fn__sync_else(self):
    r = Chain(V).if_(pred_sync_true, then=Chain().then(lambda v: v + 1)).else_(else_sync).run()
    self.assertEqual(r, 11)

  async def test_sync_pred_truthy__chain_fn__async_else(self):
    r = Chain(V).if_(pred_sync_true, then=Chain().then(lambda v: v + 1)).else_(else_async).run()
    self.assertEqual(r, 11)

  # =========================================================================
  # Sync predicate, falsy (P_SYNC_F)
  # =========================================================================

  # --- fn: sync ---

  async def test_sync_pred_falsy__sync_fn__no_else(self):
    r = Chain(V).if_(pred_sync_false, then=fn_sync).run()
    self.assertEqual(r, 10)

  async def test_sync_pred_falsy__sync_fn__sync_else(self):
    r = Chain(V).if_(pred_sync_false, then=fn_sync).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_sync_pred_falsy__sync_fn__async_else(self):
    r = await Chain(V).if_(pred_sync_false, then=fn_sync).else_(else_async).run()
    self.assertEqual(r, 110)

  # --- fn: async ---

  async def test_sync_pred_falsy__async_fn__no_else(self):
    r = Chain(V).if_(pred_sync_false, then=fn_async).run()
    self.assertEqual(r, 10)

  async def test_sync_pred_falsy__async_fn__sync_else(self):
    r = Chain(V).if_(pred_sync_false, then=fn_async).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_sync_pred_falsy__async_fn__async_else(self):
    r = await Chain(V).if_(pred_sync_false, then=fn_async).else_(else_async).run()
    self.assertEqual(r, 110)

  # --- fn: plain value ---

  async def test_sync_pred_falsy__value_fn__no_else(self):
    r = Chain(V).if_(pred_sync_false, then=FN_VALUE).run()
    self.assertEqual(r, 10)

  async def test_sync_pred_falsy__value_fn__sync_else(self):
    r = Chain(V).if_(pred_sync_false, then=FN_VALUE).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_sync_pred_falsy__value_fn__async_else(self):
    r = await Chain(V).if_(pred_sync_false, then=FN_VALUE).else_(else_async).run()
    self.assertEqual(r, 110)

  # --- fn: nested Chain ---

  async def test_sync_pred_falsy__chain_fn__no_else(self):
    r = Chain(V).if_(pred_sync_false, then=Chain().then(lambda v: v + 1)).run()
    self.assertEqual(r, 10)

  async def test_sync_pred_falsy__chain_fn__sync_else(self):
    r = Chain(V).if_(pred_sync_false, then=Chain().then(lambda v: v + 1)).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_sync_pred_falsy__chain_fn__async_else(self):
    r = await Chain(V).if_(pred_sync_false, then=Chain().then(lambda v: v + 1)).else_(else_async).run()
    self.assertEqual(r, 110)

  # =========================================================================
  # Async predicate, truthy (P_ASYNC_T)
  # =========================================================================

  # --- fn: sync ---

  async def test_async_pred_truthy__sync_fn__no_else(self):
    r = await Chain(V).if_(pred_async_true, then=fn_sync).run()
    self.assertEqual(r, 20)

  async def test_async_pred_truthy__sync_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_true, then=fn_sync).else_(else_sync).run()
    self.assertEqual(r, 20)

  async def test_async_pred_truthy__sync_fn__async_else(self):
    r = await Chain(V).if_(pred_async_true, then=fn_sync).else_(else_async).run()
    self.assertEqual(r, 20)

  # --- fn: async ---

  async def test_async_pred_truthy__async_fn__no_else(self):
    r = await Chain(V).if_(pred_async_true, then=fn_async).run()
    self.assertEqual(r, 20)

  async def test_async_pred_truthy__async_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_true, then=fn_async).else_(else_sync).run()
    self.assertEqual(r, 20)

  async def test_async_pred_truthy__async_fn__async_else(self):
    r = await Chain(V).if_(pred_async_true, then=fn_async).else_(else_async).run()
    self.assertEqual(r, 20)

  # --- fn: plain value ---

  async def test_async_pred_truthy__value_fn__no_else(self):
    r = await Chain(V).if_(pred_async_true, then=FN_VALUE).run()
    self.assertEqual(r, 42)

  async def test_async_pred_truthy__value_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_true, then=FN_VALUE).else_(else_sync).run()
    self.assertEqual(r, 42)

  async def test_async_pred_truthy__value_fn__async_else(self):
    r = await Chain(V).if_(pred_async_true, then=FN_VALUE).else_(else_async).run()
    self.assertEqual(r, 42)

  # --- fn: nested Chain ---

  async def test_async_pred_truthy__chain_fn__no_else(self):
    r = await Chain(V).if_(pred_async_true, then=Chain().then(lambda v: v + 1)).run()
    self.assertEqual(r, 11)

  async def test_async_pred_truthy__chain_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_true, then=Chain().then(lambda v: v + 1)).else_(else_sync).run()
    self.assertEqual(r, 11)

  async def test_async_pred_truthy__chain_fn__async_else(self):
    r = await Chain(V).if_(pred_async_true, then=Chain().then(lambda v: v + 1)).else_(else_async).run()
    self.assertEqual(r, 11)

  # =========================================================================
  # Async predicate, falsy (P_ASYNC_F)
  # =========================================================================

  # --- fn: sync ---

  async def test_async_pred_falsy__sync_fn__no_else(self):
    r = await Chain(V).if_(pred_async_false, then=fn_sync).run()
    self.assertEqual(r, 10)

  async def test_async_pred_falsy__sync_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_false, then=fn_sync).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_async_pred_falsy__sync_fn__async_else(self):
    r = await Chain(V).if_(pred_async_false, then=fn_sync).else_(else_async).run()
    self.assertEqual(r, 110)

  # --- fn: async ---

  async def test_async_pred_falsy__async_fn__no_else(self):
    r = await Chain(V).if_(pred_async_false, then=fn_async).run()
    self.assertEqual(r, 10)

  async def test_async_pred_falsy__async_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_false, then=fn_async).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_async_pred_falsy__async_fn__async_else(self):
    r = await Chain(V).if_(pred_async_false, then=fn_async).else_(else_async).run()
    self.assertEqual(r, 110)

  # --- fn: plain value ---

  async def test_async_pred_falsy__value_fn__no_else(self):
    r = await Chain(V).if_(pred_async_false, then=FN_VALUE).run()
    self.assertEqual(r, 10)

  async def test_async_pred_falsy__value_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_false, then=FN_VALUE).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_async_pred_falsy__value_fn__async_else(self):
    r = await Chain(V).if_(pred_async_false, then=FN_VALUE).else_(else_async).run()
    self.assertEqual(r, 110)

  # --- fn: nested Chain ---

  async def test_async_pred_falsy__chain_fn__no_else(self):
    r = await Chain(V).if_(pred_async_false, then=Chain().then(lambda v: v + 1)).run()
    self.assertEqual(r, 10)

  async def test_async_pred_falsy__chain_fn__sync_else(self):
    r = await Chain(V).if_(pred_async_false, then=Chain().then(lambda v: v + 1)).else_(else_sync).run()
    self.assertEqual(r, 110)

  async def test_async_pred_falsy__chain_fn__async_else(self):
    r = await Chain(V).if_(pred_async_false, then=Chain().then(lambda v: v + 1)).else_(else_async).run()
    self.assertEqual(r, 110)


class IfElseAsyncEdgeCaseTests(unittest.IsolatedAsyncioTestCase):
  """Async-specific edge case tests for if_/else_."""

  # 1. Async predicate that raises
  async def test_async_predicate_raises(self):
    async def bad_pred(v):
      raise RuntimeError('predicate boom')

    with self.assertRaises(RuntimeError) as ctx:
      await Chain(V).if_(bad_pred, then=fn_sync).run()
    self.assertIn('predicate boom', str(ctx.exception))

  # 2. Async fn that raises (with truthy predicate)
  async def test_async_fn_raises_with_truthy_pred(self):
    async def bad_fn(v):
      raise ValueError('fn boom')

    with self.assertRaises(ValueError) as ctx:
      await Chain(V).if_(pred_sync_true, then=bad_fn).run()
    self.assertIn('fn boom', str(ctx.exception))

  # 3. Async else_fn that raises (with falsy predicate)
  async def test_async_else_raises_with_falsy_pred(self):
    async def bad_else(v):
      raise TypeError('else boom')

    with self.assertRaises(TypeError) as ctx:
      await Chain(V).if_(pred_sync_false, then=fn_sync).else_(bad_else).run()
    self.assertIn('else boom', str(ctx.exception))

  # 4. if_ in chain where earlier step is async (chain already in async mode)
  async def test_if_after_async_step(self):
    async def async_add_one(v):
      return v + 1

    r = await Chain(V).then(async_add_one).if_(pred_sync_true, then=fn_sync).run()
    # After async step: 11, then if_(11 > 5 => True) -> 11 * 2 = 22
    self.assertEqual(r, 22)

  # 5. if_ in chain where later step is async
  async def test_if_before_async_step(self):
    async def async_add_one(v):
      return v + 1

    r = await Chain(V).if_(pred_sync_true, then=fn_sync).then(async_add_one).run()
    # if_(10 > 5 => True) -> 20, then async step: 21
    self.assertEqual(r, 21)

  # 6. Multiple if_/else_ in sequence, mixing sync and async
  async def test_multiple_if_else_mixed(self):
    async def async_mul_3(v):
      return v * 3

    r = await (
      Chain(V)
      .if_(pred_sync_true, then=fn_sync)       # 10 > 5 => True -> 20
      .if_(pred_async_false, then=fn_async)     # 20 > 100 => False -> passthrough 20
      .else_(else_async)                   # falsy -> 20 + 100 = 120
      .if_(pred_async_true, then=async_mul_3)   # 120 > 5 => True -> 360
      .run()
    )
    self.assertEqual(r, 360)

  # 7. if_ with async predicate inside map
  async def test_if_async_pred_inside_map(self):
    async def async_check(v):
      return v > 2

    r = await (
      Chain([1, 2, 3, 4, 5])
      .map(lambda item: Chain(item).if_(async_check, then=lambda v: v * 10).run())
      .run()
    )
    # item 1: 1 > 2 False => passthrough 1
    # item 2: 2 > 2 False => passthrough 2
    # item 3: 3 > 2 True => 30
    # item 4: 4 > 2 True => 40
    # item 5: 5 > 2 True => 50
    self.assertEqual(r, [1, 2, 30, 40, 50])

  # 8. if_ with frozen chain containing async operations
  async def test_if_with_frozen_chain_async(self):
    async def async_double(v):
      return v * 2

    frozen = Chain().then(async_double).freeze()
    r = await Chain(V).if_(pred_sync_true, then=frozen).run()
    self.assertEqual(r, 20)

  # 9. Chain with except_ handler + if_ that has async predicate raising
  async def test_except_handler_with_async_pred_raising(self):
    async def bad_pred(v):
      raise RuntimeError('pred error')

    r = await (
      Chain(V)
      .if_(bad_pred, then=fn_sync)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(r, 'caught')

  # 10. Async predicate returning truthy 0-like awaitable (edge case)
  async def test_async_pred_returns_zero_like_truthy(self):
    # An async pred that returns a non-empty list (truthy)
    async def pred_returns_list(v):
      return [0]  # truthy (non-empty list) despite containing 0

    r = await Chain(V).if_(pred_returns_list, then=fn_sync).run()
    self.assertEqual(r, 20)

  async def test_async_pred_returns_zero(self):
    # An async pred that returns 0 (falsy)
    async def pred_returns_zero(v):
      return 0

    r = await Chain(V).if_(pred_returns_zero, then=fn_sync).run()
    self.assertEqual(r, 10)  # passthrough

  async def test_async_pred_returns_empty_string(self):
    # An async pred that returns '' (falsy)
    async def pred_returns_empty(v):
      return ''

    r = await Chain(V).if_(pred_returns_empty, then=fn_sync).else_(else_sync).run()
    self.assertEqual(r, 110)

  # 11. if_/else_ with Chain.return_() inside fn
  async def test_return_inside_if_fn(self):
    async def fn_with_return(v):
      Chain.return_(999)

    r = await Chain(V).if_(pred_async_true, then=fn_with_return).then(lambda v: v + 1).run()
    self.assertEqual(r, 999)

  async def test_return_inside_else_fn(self):
    async def else_with_return(v):
      Chain.return_(888)

    r = await Chain(V).if_(pred_async_false, then=fn_sync).else_(else_with_return).then(lambda v: v + 1).run()
    self.assertEqual(r, 888)

  # 12. if_/else_ with Chain.break_() inside fn (within map)
  async def test_break_inside_if_fn_in_map(self):
    """break_() in the if_ fn propagates to the outer map via nested chain."""
    async def always_true(v):
      return True

    async def fn_with_break(v):
      if v >= 3:
        Chain.break_()
      return v * 10

    def make_inner(item):
      c = Chain(item).if_(always_true, then=fn_with_break)
      c.is_nested = True
      return c._run(Null, None, None)

    r = await (
      Chain([1, 2, 3, 4, 5])
      .map(make_inner)
      .run()
    )
    # item 1: True -> 10
    # item 2: True -> 20
    # item 3: True -> break_() => stops iteration
    self.assertEqual(r, [10, 20])

  async def test_break_inside_else_fn_in_map(self):
    """break_() in the else_ fn propagates to the outer map via nested chain."""
    async def always_false(v):
      return False

    async def else_with_break(v):
      if v >= 2:
        Chain.break_()
      return v + 100

    def make_inner(item):
      c = Chain(item).if_(always_false, then=fn_sync).else_(else_with_break)
      c.is_nested = True
      return c._run(Null, None, None)

    r = await (
      Chain([1, 2, 3])
      .map(make_inner)
      .run()
    )
    # item 1: False -> else -> 101
    # item 2: False -> else -> break_()
    self.assertEqual(r, [101])

  # --- Additional async crossing tests ---

  async def test_async_pred_truthy_sync_fn_chain_continues_sync(self):
    """Async pred makes chain async; verify subsequent sync step works."""
    r = await (
      Chain(V)
      .if_(pred_async_true, then=fn_sync)   # async pred -> True -> sync fn -> 20
      .then(lambda v: v + 5)           # sync step -> 25
      .run()
    )
    self.assertEqual(r, 25)

  async def test_async_pred_falsy_async_else_chain_continues(self):
    """Async pred + async else, then more steps."""
    r = await (
      Chain(V)
      .if_(pred_async_false, then=fn_sync)  # async pred -> False
      .else_(else_async)               # async else -> 110
      .then(lambda v: v * 2)           # 220
      .run()
    )
    self.assertEqual(r, 220)

  async def test_nested_chain_fn_with_async_steps(self):
    """fn is a nested Chain that itself has async steps."""
    async def async_triple(v):
      return v * 3

    inner = Chain().then(async_triple).then(lambda v: v + 1)
    r = await Chain(V).if_(pred_sync_true, then=inner).run()
    # 10 > 5 True -> inner chain: 10 * 3 = 30, then + 1 = 31
    self.assertEqual(r, 31)

  async def test_else_with_nested_chain_async(self):
    """else_ branch is a nested Chain with async steps."""
    async def async_negate(v):
      return -v

    inner_else = Chain().then(async_negate)
    r = await Chain(V).if_(pred_sync_false, then=fn_sync).else_(inner_else).run()
    # 10 > 100 False -> else chain: -10
    self.assertEqual(r, -10)

  async def test_if_with_do_preserves_value(self):
    """if_ result used in .do() is discarded, original value preserved."""
    tracker = []

    async def async_side_effect(v):
      tracker.append(v)
      return 'discarded'

    r = await (
      Chain(V)
      .do(lambda v: Chain(v).if_(pred_async_true, then=async_side_effect).run())
      .run()
    )
    self.assertEqual(r, 10)
    self.assertEqual(tracker, [10])

  async def test_sync_pred_falsy_no_else_value_is_none_root(self):
    """When root value is Null and pred is falsy with no else, current_value passes through."""
    r = await Chain().then(lambda: V).if_(pred_async_false, then=fn_sync).run()
    self.assertEqual(r, 10)

  async def test_multiple_sequential_if_all_async(self):
    """Multiple if_ in sequence, all with async predicates."""
    async def add_1(v):
      return v + 1

    async def mul_2(v):
      return v * 2

    async def sub_5(v):
      return v - 5

    r = await (
      Chain(V)
      .if_(pred_async_true, then=add_1)   # 10 > 5 True -> 11
      .if_(pred_async_true, then=mul_2)   # 11 > 5 True -> 22
      .if_(pred_async_false, then=sub_5)  # 22 > 100 False -> passthrough 22
      .run()
    )
    self.assertEqual(r, 22)

  async def test_async_pred_with_except_on_fn_raise(self):
    """Async pred truthy, fn raises, except_ catches it."""
    async def bad_fn(v):
      raise ValueError('fn error')

    r = await (
      Chain(V)
      .if_(pred_async_true, then=bad_fn)
      .except_(lambda rv, exc: f'handled: {exc}')
      .run()
    )
    self.assertEqual(r, 'handled: fn error')

  async def test_async_else_with_except_on_else_raise(self):
    """Async pred falsy, async else raises, except_ catches it."""
    async def bad_else(v):
      raise ValueError('else error')

    r = await (
      Chain(V)
      .if_(pred_async_false, then=fn_sync)
      .else_(bad_else)
      .except_(lambda rv, exc: f'handled: {exc}')
      .run()
    )
    self.assertEqual(r, 'handled: else error')

  async def test_if_with_finally_async_pred(self):
    """finally_ runs after if_ with async predicate."""
    tracker = []

    r = await (
      Chain(V)
      .if_(pred_async_true, then=fn_async)
      .finally_(lambda v: tracker.append('done'))
      .run()
    )
    self.assertEqual(r, 20)
    self.assertEqual(tracker, ['done'])


if __name__ == '__main__':
  unittest.main()
