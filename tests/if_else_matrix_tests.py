"""Exhaustive combinatorial tests for Chain.if_() and Chain.else_().

Matrix dimensions:
  - Predicate type (sync truthy, sync falsy, async truthy, async falsy, boundary, raising)
  - fn type (sync, async, plain value, nested Chain, extra args, raising, returns None, Ellipsis)
  - else_ variation (none, sync, async, plain value)
  - Current value state (truthy, falsy, None, no root)
  - Chain context (single if_, after then, before then, multiple if_/else_)
  - Error cases (else_ without if_, else_ after then, predicate/fn/else_fn raises)
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def async_truthy_pred(v):
  return v > 5

async def async_falsy_pred(v):
  return v > 100

async def async_double(v):
  return v * 2

async def async_triple(v):
  return v * 3

async def async_add_one(v):
  return v + 1

async def async_return_none(v):
  return None

async def async_raise(v):
  raise RuntimeError('async fn error')

async def async_pred_raises(v):
  raise RuntimeError('async predicate error')

async def async_else_fn(v):
  return v * 100

def no_arg_fn():
  return 'no_arg_result'

async def async_no_arg_fn():
  return 'async_no_arg_result'

async def async_boundary_pred(v):
  return v == 0


# ============================================================================
# 1. Sync predicate + sync fn combinations
# ============================================================================

class TestIfSyncPredicateSyncFn(unittest.TestCase):

  def test_truthy_predicate_applies_sync_fn(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: v * 2).run()
    self.assertEqual(r, 20)

  def test_falsy_predicate_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).run()
    self.assertEqual(r, 3)

  def test_boundary_predicate_zero_equals_zero(self):
    r = Chain(0).if_(lambda v: v == 0, then=lambda v: 'was_zero').run()
    self.assertEqual(r, 'was_zero')

  def test_boundary_predicate_nonzero_passthrough(self):
    r = Chain(5).if_(lambda v: v == 0, then=lambda v: 'was_zero').run()
    self.assertEqual(r, 5)

  def test_truthy_predicate_fn_returns_none(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: None).run()
    self.assertIsNone(r)

  def test_truthy_predicate_fn_with_extra_args(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda a, b: a + b, args=(3, 7,)).run()
    self.assertEqual(r, 10)

  def test_truthy_predicate_fn_with_kwargs(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda x=0, y=0: x + y, kwargs=dict(x=3, y=7)).run()
    self.assertEqual(r, 10)

  def test_truthy_predicate_plain_value_fn(self):
    r = Chain(10).if_(lambda v: v > 5, then=42).run()
    self.assertEqual(r, 42)

  def test_falsy_predicate_plain_value_fn_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=42).run()
    self.assertEqual(r, 3)

  def test_truthy_predicate_ellipsis_fn(self):
    r = Chain(10).if_(lambda v: v > 5, then=no_arg_fn, args=(...,)).run()
    self.assertEqual(r, 'no_arg_result')

  def test_falsy_predicate_ellipsis_fn_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=no_arg_fn, args=(...,)).run()
    self.assertEqual(r, 3)

  def test_truthy_predicate_nested_chain_fn(self):
    inner = Chain().then(lambda v: v + 100)
    r = Chain(10).if_(lambda v: v > 5, then=inner).run()
    self.assertEqual(r, 110)

  def test_falsy_predicate_nested_chain_fn_passthrough(self):
    inner = Chain().then(lambda v: v + 100)
    r = Chain(3).if_(lambda v: v > 5, then=inner).run()
    self.assertEqual(r, 3)


# ============================================================================
# 2. Sync predicate + async fn
# ============================================================================

class TestIfSyncPredicateAsyncFn(unittest.IsolatedAsyncioTestCase):

  async def test_truthy_predicate_async_fn(self):
    r = await Chain(10).if_(lambda v: v > 5, then=async_double).run()
    self.assertEqual(r, 20)

  async def test_falsy_predicate_async_fn_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=async_double).run()
    self.assertEqual(r, 3)

  async def test_truthy_predicate_async_fn_returns_none(self):
    r = await Chain(10).if_(lambda v: v > 5, then=async_return_none).run()
    self.assertIsNone(r)


# ============================================================================
# 3. Async predicate + sync fn
# ============================================================================

class TestIfAsyncPredicateSyncFn(unittest.IsolatedAsyncioTestCase):

  async def test_async_truthy_predicate_sync_fn(self):
    r = await Chain(10).if_(async_truthy_pred, then=lambda v: v * 2).run()
    self.assertEqual(r, 20)

  async def test_async_falsy_predicate_sync_fn_passthrough(self):
    r = await Chain(3).if_(async_truthy_pred, then=lambda v: v * 2).run()
    self.assertEqual(r, 3)

  async def test_async_truthy_predicate_plain_value(self):
    r = await Chain(10).if_(async_truthy_pred, then='replaced').run()
    self.assertEqual(r, 'replaced')

  async def test_async_falsy_predicate_plain_value_passthrough(self):
    r = await Chain(3).if_(async_truthy_pred, then='replaced').run()
    self.assertEqual(r, 3)

  async def test_async_truthy_predicate_nested_chain(self):
    inner = Chain().then(lambda v: v + 1000)
    r = await Chain(10).if_(async_truthy_pred, then=inner).run()
    self.assertEqual(r, 1010)

  async def test_async_truthy_predicate_ellipsis_fn(self):
    r = await Chain(10).if_(async_truthy_pred, then=no_arg_fn, args=(...,)).run()
    self.assertEqual(r, 'no_arg_result')


# ============================================================================
# 4. Async predicate + async fn
# ============================================================================

class TestIfAsyncPredicateAsyncFn(unittest.IsolatedAsyncioTestCase):

  async def test_async_truthy_predicate_async_fn(self):
    r = await Chain(10).if_(async_truthy_pred, then=async_double).run()
    self.assertEqual(r, 20)

  async def test_async_falsy_predicate_async_fn_passthrough(self):
    r = await Chain(3).if_(async_truthy_pred, then=async_double).run()
    self.assertEqual(r, 3)

  async def test_async_boundary_predicate_async_fn(self):
    r = await Chain(0).if_(async_boundary_pred, then=async_double).run()
    self.assertEqual(r, 0)  # 0 * 2 = 0

  async def test_async_boundary_predicate_nonzero_passthrough(self):
    r = await Chain(5).if_(async_boundary_pred, then=async_double).run()
    self.assertEqual(r, 5)

  async def test_async_truthy_predicate_async_ellipsis_fn(self):
    r = await Chain(10).if_(async_truthy_pred, then=async_no_arg_fn, args=(...,)).run()
    self.assertEqual(r, 'async_no_arg_result')


# ============================================================================
# 5. else_ variations
# ============================================================================

class TestIfElseSyncSync(unittest.TestCase):

  def test_truthy_takes_if_branch_not_else(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: v * 2).else_(lambda v: v * 3).run()
    self.assertEqual(r, 20)

  def test_falsy_takes_else_branch(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).else_(lambda v: v * 3).run()
    self.assertEqual(r, 9)

  def test_falsy_else_plain_value(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).else_('fallback').run()
    self.assertEqual(r, 'fallback')

  def test_truthy_ignores_else_plain_value(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: v * 2).else_('fallback').run()
    self.assertEqual(r, 20)

  def test_falsy_else_with_extra_args(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).else_(lambda a, b: a - b, 100, 50).run()
    self.assertEqual(r, 50)


class TestIfElseAsyncVariants(unittest.IsolatedAsyncioTestCase):

  async def test_falsy_async_else_fn(self):
    r = await Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).else_(async_else_fn).run()
    self.assertEqual(r, 300)

  async def test_truthy_ignores_async_else_fn(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: v * 2).else_(async_else_fn).run()
    self.assertEqual(r, 20)

  async def test_async_pred_falsy_sync_else(self):
    r = await Chain(3).if_(async_truthy_pred, then=lambda v: v * 2).else_(lambda v: v + 1000).run()
    self.assertEqual(r, 1003)

  async def test_async_pred_falsy_async_else(self):
    r = await Chain(3).if_(async_truthy_pred, then=async_double).else_(async_else_fn).run()
    self.assertEqual(r, 300)

  async def test_async_pred_truthy_ignores_else(self):
    r = await Chain(10).if_(async_truthy_pred, then=async_double).else_(async_else_fn).run()
    self.assertEqual(r, 20)

  async def test_async_pred_falsy_else_plain_value(self):
    r = await Chain(3).if_(async_truthy_pred, then=async_double).else_('default').run()
    self.assertEqual(r, 'default')


# ============================================================================
# 6. Current value states
# ============================================================================

class TestIfCurrentValueStates(unittest.TestCase):

  def test_truthy_value(self):
    r = Chain(10).if_(lambda v: v > 0, then=lambda v: v + 1).run()
    self.assertEqual(r, 11)

  def test_falsy_value_zero(self):
    r = Chain(0).if_(lambda v: v, then=lambda v: 'should_not').run()
    self.assertEqual(r, 0)

  def test_falsy_value_empty_string(self):
    r = Chain('').if_(lambda v: v, then=lambda v: 'should_not').run()
    self.assertEqual(r, '')

  def test_falsy_value_empty_list(self):
    r = Chain([]).if_(lambda v: v, then=lambda v: 'should_not').run()
    self.assertEqual(r, [])

  def test_none_as_current_value_passthrough(self):
    r = Chain(None).if_(lambda v: v is not None, then=lambda v: v + 1).run()
    self.assertIsNone(r)

  def test_none_as_current_value_truthy_pred(self):
    r = Chain(None).if_(lambda v: v is None, then=lambda v: 'was_none').run()
    self.assertEqual(r, 'was_none')

  def test_no_root_value_run_provides_value_truthy(self):
    # Chain() with no root, but run(value) provides it
    r = Chain().if_(lambda v: v > 5, then=lambda v: v * 2).run(10)
    self.assertEqual(r, 20)

  def test_no_root_value_run_provides_value_falsy(self):
    r = Chain().if_(lambda v: v > 5, then=lambda v: v * 2).run(3)
    self.assertEqual(r, 3)

  def test_no_root_value_run_provides_value_with_else(self):
    r = Chain().if_(lambda v: v > 5, then=lambda v: v * 2).else_(lambda v: v * 3).run(3)
    self.assertEqual(r, 9)

  def test_falsy_value_zero_with_else(self):
    r = Chain(0).if_(lambda v: v, then=lambda v: 'truthy').else_(lambda v: 'falsy').run()
    self.assertEqual(r, 'falsy')

  def test_none_value_with_else(self):
    r = Chain(None).if_(lambda v: v is not None, then=lambda v: 'not_none').else_(lambda v: 'is_none').run()
    self.assertEqual(r, 'is_none')


# ============================================================================
# 7. Chain context: position of if_ in chain
# ============================================================================

class TestIfChainContext(unittest.TestCase):

  def test_single_if_step(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: v * 2).run()
    self.assertEqual(r, 20)

  def test_if_after_then_steps(self):
    r = Chain(2).then(lambda v: v + 3).then(lambda v: v * 2).if_(lambda v: v > 5, then=lambda v: v + 100).run()
    # 2 -> 5 -> 10 -> 10 > 5 is True -> 110
    self.assertEqual(r, 110)

  def test_if_before_then_steps_truthy(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: v * 2).then(lambda v: v + 1).run()
    # 10 -> 20 -> 21
    self.assertEqual(r, 21)

  def test_if_before_then_steps_falsy_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).then(lambda v: v + 1).run()
    # 3 passthrough -> 4
    self.assertEqual(r, 4)

  def test_multiple_if_in_sequence(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=lambda v: v * 2)      # 10 > 5 -> 20
      .if_(lambda v: v > 15, then=lambda v: v + 100)    # 20 > 15 -> 120
      .run()
    )
    self.assertEqual(r, 120)

  def test_multiple_if_some_truthy_some_falsy(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=lambda v: v * 2)       # 10 > 5 -> 20
      .if_(lambda v: v > 50, then=lambda v: v + 1000)   # 20 > 50 -> passthrough 20
      .run()
    )
    self.assertEqual(r, 20)

  def test_multiple_if_else_in_sequence(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=lambda v: 'big').else_(lambda v: 'small')    # 'small'
      .if_(lambda v: v == 'small', then=lambda v: 'confirmed').else_(lambda v: 'unexpected')
      .run()
    )
    self.assertEqual(r, 'confirmed')

  def test_if_with_do_before(self):
    tracker = []
    r = (
      Chain(10)
      .do(lambda v: tracker.append(v))
      .if_(lambda v: v > 5, then=lambda v: v * 2)
      .run()
    )
    self.assertEqual(r, 20)
    self.assertEqual(tracker, [10])

  def test_if_with_do_after(self):
    tracker = []
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=lambda v: v * 2)
      .do(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(r, 20)
    self.assertEqual(tracker, [20])


# ============================================================================
# 8. Error cases
# ============================================================================

class TestIfElseErrors(unittest.TestCase):

  def test_else_without_if_raises(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(10).then(lambda v: v).else_(lambda v: v)
    self.assertIn('else_() must be called immediately after if_() with no operations in between', str(ctx.exception))

  def test_else_after_then_after_if_raises(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(10).if_(lambda v: True, then=lambda v: v).then(lambda v: v).else_(lambda v: v)
    self.assertIn('else_() must be called immediately after if_() with no operations in between', str(ctx.exception))

  def test_else_on_empty_chain_raises(self):
    with self.assertRaises(QuentException) as ctx:
      Chain().else_(lambda v: v)
    self.assertIn('else_() must be called immediately after if_() with no operations in between', str(ctx.exception))

  def test_else_after_do_raises(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(10).do(lambda v: v).else_(lambda v: v)
    self.assertIn('else_() must be called immediately after if_() with no operations in between', str(ctx.exception))

  def test_else_after_map_raises(self):
    with self.assertRaises(QuentException) as ctx:
      Chain([1, 2]).map(lambda v: v).else_(lambda v: v)
    self.assertIn('else_() must be called immediately after if_() with no operations in between', str(ctx.exception))

  def test_predicate_raises_propagates(self):
    def bad_pred(v):
      raise ValueError('pred boom')
    with self.assertRaises(ValueError) as ctx:
      Chain(10).if_(bad_pred, then=lambda v: v * 2).run()
    self.assertIn('pred boom', str(ctx.exception))

  def test_fn_raises_propagates(self):
    def bad_fn(v):
      raise RuntimeError('fn boom')
    with self.assertRaises(RuntimeError) as ctx:
      Chain(10).if_(lambda v: True, then=bad_fn).run()
    self.assertIn('fn boom', str(ctx.exception))

  def test_else_fn_raises_propagates(self):
    def bad_else(v):
      raise TypeError('else boom')
    with self.assertRaises(TypeError) as ctx:
      Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).else_(bad_else).run()
    self.assertIn('else boom', str(ctx.exception))


class TestIfElseErrorsAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_predicate_raises(self):
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(10).if_(async_pred_raises, then=lambda v: v * 2).run()
    self.assertIn('async predicate error', str(ctx.exception))

  async def test_async_fn_raises(self):
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(10).if_(lambda v: True, then=async_raise).run()
    self.assertIn('async fn error', str(ctx.exception))

  async def test_async_else_fn_raises(self):
    async def bad_async_else(v):
      raise TypeError('async else boom')
    with self.assertRaises(TypeError) as ctx:
      await Chain(3).if_(lambda v: v > 5, then=lambda v: v).else_(bad_async_else).run()
    self.assertIn('async else boom', str(ctx.exception))

  async def test_async_pred_raises_with_else(self):
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(10).if_(async_pred_raises, then=lambda v: v).else_(lambda v: 'else').run()
    self.assertIn('async predicate error', str(ctx.exception))

  async def test_fn_raises_after_async_truthy_pred(self):
    def bad_fn(v):
      raise RuntimeError('sync fn boom after async pred')
    with self.assertRaises(RuntimeError) as ctx:
      await Chain(10).if_(async_truthy_pred, then=bad_fn).run()
    self.assertIn('sync fn boom after async pred', str(ctx.exception))


# ============================================================================
# 9. if_ with except_ and finally_ handlers
# ============================================================================

class TestIfWithExceptFinally(unittest.TestCase):

  def test_if_fn_raises_caught_by_except(self):
    def bad_fn(v):
      raise ValueError('if fn error')
    r = (
      Chain(10)
      .if_(lambda v: True, then=bad_fn)
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(r, 'caught')

  def test_if_predicate_raises_caught_by_except(self):
    def bad_pred(v):
      raise ValueError('pred error')
    r = (
      Chain(10)
      .if_(bad_pred, then=lambda v: v)
      .except_(lambda rv, exc: 'caught_pred')
      .run()
    )
    self.assertEqual(r, 'caught_pred')

  def test_else_fn_raises_caught_by_except(self):
    def bad_else(v):
      raise ValueError('else error')
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=lambda v: v)
      .else_(bad_else)
      .except_(lambda rv, exc: 'caught_else')
      .run()
    )
    self.assertEqual(r, 'caught_else')

  def test_if_truthy_with_finally(self):
    tracker = []
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=lambda v: v * 2)
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(r, 20)
    self.assertEqual(tracker, [10])  # finally receives root value

  def test_if_falsy_with_finally(self):
    tracker = []
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=lambda v: v * 2)
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(r, 3)
    self.assertEqual(tracker, [3])


# ============================================================================
# 10. Nested Chain as fn with if_
# ============================================================================

class TestIfNestedChainFn(unittest.TestCase):

  def test_nested_chain_receives_current_value(self):
    inner = Chain().then(lambda v: v * 10)
    r = Chain(7).if_(lambda v: v > 5, then=inner).run()
    self.assertEqual(r, 70)

  def test_nested_chain_falsy_passthrough(self):
    inner = Chain().then(lambda v: v * 10)
    r = Chain(3).if_(lambda v: v > 5, then=inner).run()
    self.assertEqual(r, 3)

  def test_nested_chain_with_multiple_steps(self):
    inner = Chain().then(lambda v: v + 1).then(lambda v: v * 2)
    r = Chain(10).if_(lambda v: v > 5, then=inner).run()
    # 10 -> inner: 10+1=11, 11*2=22
    self.assertEqual(r, 22)

  def test_nested_chain_in_else(self):
    inner_if = Chain().then(lambda v: 'if_branch')
    inner_else = Chain().then(lambda v: 'else_branch')
    r = Chain(3).if_(lambda v: v > 5, then=inner_if).else_(inner_else).run()
    self.assertEqual(r, 'else_branch')


class TestIfNestedChainAsync(unittest.IsolatedAsyncioTestCase):

  async def test_nested_chain_with_async_step(self):
    inner = Chain().then(async_double)
    r = await Chain(10).if_(lambda v: v > 5, then=inner).run()
    self.assertEqual(r, 20)

  async def test_async_pred_nested_chain_with_async(self):
    inner = Chain().then(async_add_one)
    r = await Chain(10).if_(async_truthy_pred, then=inner).run()
    self.assertEqual(r, 11)


# ============================================================================
# 11. if_ with run() providing value
# ============================================================================

class TestIfWithRunValue(unittest.TestCase):

  def test_if_on_valueless_chain_run_provides_value_truthy(self):
    c = Chain().if_(lambda v: v > 5, then=lambda v: v * 2)
    r = c.run(10)
    self.assertEqual(r, 20)

  def test_if_on_valueless_chain_run_provides_value_falsy(self):
    c = Chain().if_(lambda v: v > 5, then=lambda v: v * 2)
    r = c.run(3)
    self.assertEqual(r, 3)

  def test_if_else_on_valueless_chain_run_provides_value(self):
    c = Chain().if_(lambda v: v > 5, then=lambda v: v * 2).else_(lambda v: v * 3)
    self.assertEqual(c.run(10), 20)
    self.assertEqual(c.run(3), 9)

  def test_if_on_valueless_chain_run_provides_none(self):
    c = Chain().if_(lambda v: v is None, then=lambda v: 'was_none')
    r = c.run(None)
    self.assertEqual(r, 'was_none')

  def test_if_on_valueless_chain_run_provides_zero(self):
    c = Chain().if_(lambda v: v == 0, then=lambda v: 'was_zero')
    r = c.run(0)
    self.assertEqual(r, 'was_zero')


# ============================================================================
# 12. Complex combinations
# ============================================================================

class TestIfComplexCombinations(unittest.TestCase):

  def test_if_then_if_chained(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=lambda v: v * 2)       # 20
      .then(lambda v: v + 1)                        # 21
      .if_(lambda v: v > 20, then=lambda v: v * 10)      # 210
      .run()
    )
    self.assertEqual(r, 210)

  def test_if_else_then_if_else_chained(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=lambda v: 'big').else_(lambda v: 'small')
      .then(lambda v: v.upper())
      .if_(lambda v: v == 'SMALL', then=lambda v: 'confirmed_small').else_(lambda v: 'unexpected')
      .run()
    )
    self.assertEqual(r, 'confirmed_small')

  def test_multiple_if_all_falsy_passthrough(self):
    r = (
      Chain(1)
      .if_(lambda v: v > 10, then=lambda v: 'a')
      .if_(lambda v: v > 20, then=lambda v: 'b')
      .if_(lambda v: v > 30, then=lambda v: 'c')
      .run()
    )
    self.assertEqual(r, 1)

  def test_multiple_if_all_truthy_last_wins(self):
    r = (
      Chain(100)
      .if_(lambda v: v > 1, then=lambda v: v + 1)    # 101
      .if_(lambda v: v > 1, then=lambda v: v + 2)    # 103
      .if_(lambda v: v > 1, then=lambda v: v + 3)    # 106
      .run()
    )
    self.assertEqual(r, 106)

  def test_if_with_string_value(self):
    r = Chain('hello').if_(lambda v: len(v) > 3, then=lambda v: v.upper()).run()
    self.assertEqual(r, 'HELLO')

  def test_if_with_list_value(self):
    r = Chain([1, 2, 3]).if_(lambda v: len(v) > 2, then=lambda v: sum(v)).run()
    self.assertEqual(r, 6)

  def test_if_with_dict_value(self):
    r = Chain({'a': 1}).if_(lambda v: 'a' in v, then=lambda v: v['a']).run()
    self.assertEqual(r, 1)


class TestIfComplexAsync(unittest.IsolatedAsyncioTestCase):

  async def test_then_async_then_if_sync(self):
    r = await Chain(5).then(async_add_one).if_(lambda v: v > 5, then=lambda v: v * 10).run()
    # 5 -> async 6 -> 6 > 5 -> 60
    self.assertEqual(r, 60)

  async def test_if_async_fn_then_sync_then(self):
    r = await Chain(10).if_(lambda v: v > 5, then=async_double).then(lambda v: v + 1).run()
    # 10 > 5 -> async 20 -> 21
    self.assertEqual(r, 21)

  async def test_if_falsy_async_fn_then_sync_then(self):
    r = await Chain(3).if_(lambda v: v > 5, then=async_double).then(async_add_one).run()
    # 3 passthrough -> async 4
    self.assertEqual(r, 4)

  async def test_chain_with_both_async_pred_and_async_fn_and_else(self):
    r = await Chain(10).if_(async_truthy_pred, then=async_double).else_(async_triple).run()
    self.assertEqual(r, 20)

    r2 = await Chain(3).if_(async_truthy_pred, then=async_double).else_(async_triple).run()
    self.assertEqual(r2, 9)

  async def test_multiple_if_async_in_sequence(self):
    r = await (
      Chain(10)
      .if_(async_truthy_pred, then=async_double)            # 10 > 5 -> 20
      .if_(lambda v: v > 15, then=async_add_one)            # 20 > 15 -> 21
      .run()
    )
    self.assertEqual(r, 21)


# ============================================================================
# 13. Edge cases
# ============================================================================

class TestIfEdgeCases(unittest.TestCase):

  def test_predicate_returns_truthy_nonbool(self):
    # Truthy non-bool (non-zero int)
    r = Chain(10).if_(lambda v: 1, then=lambda v: 'yes').run()
    self.assertEqual(r, 'yes')

  def test_predicate_returns_falsy_nonbool_zero(self):
    r = Chain(10).if_(lambda v: 0, then=lambda v: 'yes').run()
    self.assertEqual(r, 10)

  def test_predicate_returns_falsy_nonbool_none(self):
    r = Chain(10).if_(lambda v: None, then=lambda v: 'yes').run()
    self.assertEqual(r, 10)

  def test_predicate_returns_falsy_nonbool_empty_string(self):
    r = Chain(10).if_(lambda v: '', then=lambda v: 'yes').run()
    self.assertEqual(r, 10)

  def test_predicate_returns_truthy_nonbool_string(self):
    r = Chain(10).if_(lambda v: 'nonempty', then=lambda v: 'yes').run()
    self.assertEqual(r, 'yes')

  def test_predicate_returns_truthy_nonbool_list(self):
    r = Chain(10).if_(lambda v: [1], then=lambda v: 'yes').run()
    self.assertEqual(r, 'yes')

  def test_predicate_returns_falsy_nonbool_empty_list(self):
    r = Chain(10).if_(lambda v: [], then=lambda v: 'yes').run()
    self.assertEqual(r, 10)

  def test_fn_replaces_value_with_false(self):
    r = Chain(10).if_(lambda v: True, then=lambda v: False).run()
    self.assertIs(r, False)

  def test_fn_replaces_value_with_zero(self):
    r = Chain(10).if_(lambda v: True, then=lambda v: 0).run()
    self.assertEqual(r, 0)

  def test_fn_replaces_value_with_empty_string(self):
    r = Chain(10).if_(lambda v: True, then=lambda v: '').run()
    self.assertEqual(r, '')

  def test_fn_plain_value_none(self):
    r = Chain(10).if_(lambda v: True, then=None).run()
    self.assertIsNone(r)

  def test_fn_plain_value_zero(self):
    r = Chain(10).if_(lambda v: True, then=0).run()
    self.assertEqual(r, 0)

  def test_fn_plain_value_false(self):
    r = Chain(10).if_(lambda v: True, then=False).run()
    self.assertIs(r, False)

  def test_else_fn_returns_none(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v).else_(lambda v: None).run()
    self.assertIsNone(r)

  def test_else_plain_value_none(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v).else_(None).run()
    self.assertIsNone(r)

  def test_else_plain_value_zero(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v).else_(0).run()
    self.assertEqual(r, 0)

  def test_else_plain_value_false(self):
    r = Chain(3).if_(lambda v: v > 5, then=lambda v: v).else_(False).run()
    self.assertIs(r, False)

  def test_if_true_preserves_type(self):
    r = Chain(10).if_(lambda v: True, then=lambda v: v).run()
    self.assertIsInstance(r, int)
    self.assertEqual(r, 10)

  def test_if_false_preserves_type(self):
    r = Chain(10).if_(lambda v: False, then=lambda v: 'changed').run()
    self.assertIsInstance(r, int)
    self.assertEqual(r, 10)


# ============================================================================
# 14. if_ interaction with __call__ (alias for run)
# ============================================================================

class TestIfWithCallAlias(unittest.TestCase):

  def test_chain_call_truthy(self):
    r = Chain(10).if_(lambda v: v > 5, then=lambda v: v * 2)()
    self.assertEqual(r, 20)

  def test_chain_call_with_value(self):
    c = Chain().if_(lambda v: v > 5, then=lambda v: v * 2)
    self.assertEqual(c(10), 20)
    self.assertEqual(c(3), 3)


# ============================================================================
# 15. if_ with Null passthrough behavior
# ============================================================================

class TestIfNullPassthrough(unittest.TestCase):

  def test_no_root_run_value_falsy_pred_returns_passthrough(self):
    # Chain() with no root, run(val) provides value, predicate falsy -> passthrough
    r = Chain().if_(lambda v: v > 100, then=lambda v: 'yes').run(5)
    self.assertEqual(r, 5)

  def test_no_root_run_value_truthy_pred_returns_fn_result(self):
    r = Chain().if_(lambda v: v > 0, then=lambda v: 'yes').run(5)
    self.assertEqual(r, 'yes')

  def test_no_root_run_value_if_else_takes_if(self):
    r = Chain().if_(lambda v: v > 0, then=lambda v: 'if_val').else_(lambda v: 'else_val').run(5)
    self.assertEqual(r, 'if_val')

  def test_no_root_run_value_if_else_takes_else(self):
    r = Chain().if_(lambda v: v > 100, then=lambda v: 'if_val').else_(lambda v: 'else_val').run(5)
    self.assertEqual(r, 'else_val')


# ============================================================================
# 16. Async edge: sync pred returns truthy, fn is async, else is async
# ============================================================================

class TestIfMixedSyncAsyncPaths(unittest.IsolatedAsyncioTestCase):

  async def test_sync_pred_truthy_async_fn_no_else(self):
    r = await Chain(10).if_(lambda v: v > 5, then=async_double).run()
    self.assertEqual(r, 20)

  async def test_sync_pred_falsy_async_else(self):
    r = await Chain(3).if_(lambda v: v > 5, then=lambda v: v * 2).else_(async_triple).run()
    self.assertEqual(r, 9)

  async def test_async_pred_truthy_sync_fn_sync_else(self):
    r = await Chain(10).if_(async_truthy_pred, then=lambda v: v * 2).else_(lambda v: v * 3).run()
    self.assertEqual(r, 20)

  async def test_async_pred_falsy_sync_fn_sync_else(self):
    r = await Chain(3).if_(async_truthy_pred, then=lambda v: v * 2).else_(lambda v: v * 3).run()
    self.assertEqual(r, 9)

  async def test_async_pred_truthy_async_fn_async_else(self):
    r = await Chain(10).if_(async_truthy_pred, then=async_double).else_(async_triple).run()
    self.assertEqual(r, 20)

  async def test_async_pred_falsy_async_fn_async_else(self):
    r = await Chain(3).if_(async_truthy_pred, then=async_double).else_(async_triple).run()
    self.assertEqual(r, 9)


# ============================================================================
# 17. if_ with return_ and break_ interactions
# ============================================================================

class TestIfWithControlFlow(unittest.TestCase):

  def test_return_inside_if_fn(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=lambda v: Chain.return_(999))
      .then(lambda v: 'should_not_reach')
      .run()
    )
    self.assertEqual(r, 999)

  def test_return_inside_else_fn(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=lambda v: v)
      .else_(lambda v: Chain.return_(888))
      .then(lambda v: 'should_not_reach')
      .run()
    )
    self.assertEqual(r, 888)

  def test_if_truthy_does_not_trigger_return_in_else(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=lambda v: v * 2)
      .else_(lambda v: Chain.return_(888))
      .then(lambda v: v + 1)
      .run()
    )
    self.assertEqual(r, 21)


class TestIfWithControlFlowAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_return_inside_if_fn(self):
    r = await (
      Chain(10)
      .if_(async_truthy_pred, then=lambda v: Chain.return_(999))
      .then(lambda v: 'should_not_reach')
      .run()
    )
    self.assertEqual(r, 999)

  async def test_async_return_inside_else_fn(self):
    r = await (
      Chain(3)
      .if_(async_truthy_pred, then=lambda v: v)
      .else_(lambda v: Chain.return_(888))
      .then(lambda v: 'should_not_reach')
      .run()
    )
    self.assertEqual(r, 888)


if __name__ == '__main__':
  unittest.main()
