"""Cross-feature tests: X placeholder expressions x every Chain method, sync and async."""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain, Null
from quent._x import X, _XExpr, _XAttr
from helpers import (
  AsyncRange,
  SyncCM,
  AsyncCM,
  TrackingCM,
  AsyncTrackingCM,
  make_tracker,
  make_async_tracker,
)


# ---------------------------------------------------------------------------
# 1. X x then (sync)
# ---------------------------------------------------------------------------
class TestXThenSync(unittest.TestCase):

  def test_then_arithmetic_add(self):
    self.assertEqual(Chain(5).then(X + 1).run(), 6)

  def test_then_method_call_upper(self):
    self.assertEqual(Chain('hello').then(X.upper()).run(), 'HELLO')

  def test_then_item_access_list(self):
    self.assertEqual(Chain([10, 20, 30]).then(X[0]).run(), 10)

  def test_then_dict_access(self):
    self.assertEqual(Chain({'a': 1, 'b': 2}).then(X['a']).run(), 1)

  def test_then_compound_expression(self):
    self.assertEqual(Chain(5).then(X * 2 + 1).run(), 11)

  def test_then_unary_abs(self):
    self.assertEqual(Chain(-5).then(abs(X)).run(), 5)

  def test_then_identity_passthrough(self):
    result = Chain(5).then(X).run()
    self.assertEqual(result, 5)

  def test_then_multiple_x_steps(self):
    result = Chain(5).then(X + 1).then(X * 2).run()
    self.assertEqual(result, 12)

  def test_then_string_strip_then_lower(self):
    result = Chain('  HELLO  ').then(X.strip()).then(X.lower()).run()
    self.assertEqual(result, 'hello')

  def test_then_subtraction(self):
    self.assertEqual(Chain(10).then(X - 3).run(), 7)

  def test_then_floor_div(self):
    self.assertEqual(Chain(17).then(X // 5).run(), 3)

  def test_then_modulo(self):
    self.assertEqual(Chain(17).then(X % 5).run(), 2)

  def test_then_power(self):
    self.assertEqual(Chain(3).then(X ** 2).run(), 9)

  def test_then_negation(self):
    self.assertEqual(Chain(5).then(-X).run(), -5)

  def test_then_list_slice(self):
    self.assertEqual(Chain([1, 2, 3, 4, 5]).then(X[1:4]).run(), [2, 3, 4])

  def test_then_negative_index(self):
    self.assertEqual(Chain([10, 20, 30]).then(X[-1]).run(), 30)

  def test_then_nested_item_access(self):
    self.assertEqual(Chain([[1, 2], [3, 4]]).then(X[1][0]).run(), 3)

  def test_then_nested_dict_access(self):
    data = {'a': {'b': 42}}
    self.assertEqual(Chain(data).then(X['a']['b']).run(), 42)

  def test_then_radd(self):
    self.assertEqual(Chain(5).then(10 + X).run(), 15)

  def test_then_rsub(self):
    self.assertEqual(Chain(3).then(10 - X).run(), 7)

  def test_then_rmul(self):
    self.assertEqual(Chain(4).then(3 * X).run(), 12)

  def test_then_rtruediv(self):
    self.assertAlmostEqual(Chain(4).then(20 / X).run(), 5.0)

  def test_then_rpow(self):
    self.assertEqual(Chain(3).then(2 ** X).run(), 8)

  def test_then_string_concat(self):
    self.assertEqual(Chain('hello').then(X + ' world').run(), 'hello world')

  def test_then_string_repeat(self):
    self.assertEqual(Chain('ab').then(X * 3).run(), 'ababab')

  def test_then_chained_strip_replace(self):
    result = Chain('  foo  ').then(X.strip().replace('o', '0')).run()
    self.assertEqual(result, 'f00')


# ---------------------------------------------------------------------------
# 2. X x then (async)
# ---------------------------------------------------------------------------
class TestXThenAsync(unittest.IsolatedAsyncioTestCase):

  async def test_then_x_add_after_async_root(self):
    async def async_val():
      return 5
    result = await Chain(async_val).then(X + 10).run()
    self.assertEqual(result, 15)

  async def test_then_x_upper_after_async(self):
    async def async_val():
      return 'hello'
    result = await Chain(async_val).then(X.upper()).run()
    self.assertEqual(result, 'HELLO')

  async def test_then_x_item_after_async(self):
    async def async_val():
      return [10, 20, 30]
    result = await Chain(async_val).then(X[1]).run()
    self.assertEqual(result, 20)

  async def test_then_multiple_x_steps_async(self):
    async def async_val():
      return 3
    result = await Chain(async_val).then(X * 2).then(X + 1).run()
    self.assertEqual(result, 7)


# ---------------------------------------------------------------------------
# 3. X x do (sync)
# ---------------------------------------------------------------------------
class TestXDoSync(unittest.TestCase):

  def test_do_x_discards_result(self):
    """do() with X expression should not change pipeline value."""
    result = Chain(5).do(X + 100).run()
    self.assertEqual(result, 5)

  def test_do_x_pipeline_value_preserved(self):
    result = Chain('hello').do(X.upper()).run()
    self.assertEqual(result, 'hello')

  def test_do_x_between_then_steps(self):
    result = Chain(5).then(X + 1).do(X * 100).then(X * 2).run()
    # 5 -> 6 -> do(discarded) -> 6*2 = 12
    self.assertEqual(result, 12)

  def test_do_x_identity_preserves(self):
    result = Chain(42).do(X).run()
    self.assertEqual(result, 42)

  def test_do_x_item_access_discarded(self):
    result = Chain([1, 2, 3]).do(X[0]).run()
    self.assertEqual(result, [1, 2, 3])

  def test_do_x_arithmetic_discarded(self):
    result = Chain(10).do(X ** 2).then(X + 1).run()
    self.assertEqual(result, 11)


# ---------------------------------------------------------------------------
# 4. X x do (async)
# ---------------------------------------------------------------------------
class TestXDoAsync(unittest.IsolatedAsyncioTestCase):

  async def test_do_x_discards_in_async_chain(self):
    async def async_val():
      return 5
    result = await Chain(async_val).do(X + 100).run()
    self.assertEqual(result, 5)


# ---------------------------------------------------------------------------
# 5. X x foreach (sync)
# ---------------------------------------------------------------------------
class TestXForeachSync(unittest.TestCase):

  def test_foreach_multiply(self):
    result = Chain([1, 2, 3]).foreach(X * 10).run()
    self.assertEqual(result, [10, 20, 30])

  def test_foreach_upper(self):
    result = Chain(['a', 'b', 'c']).foreach(X.upper()).run()
    self.assertEqual(result, ['A', 'B', 'C'])

  def test_foreach_item_access(self):
    result = Chain([[1, 2], [3, 4], [5, 6]]).foreach(X[0]).run()
    self.assertEqual(result, [1, 3, 5])

  def test_foreach_power(self):
    result = Chain([1, 2, 3, 4]).foreach(X ** 2).run()
    self.assertEqual(result, [1, 4, 9, 16])

  def test_foreach_abs(self):
    result = Chain([1, -2, 3, -4]).foreach(abs(X)).run()
    self.assertEqual(result, [1, 2, 3, 4])

  def test_foreach_compound_expr(self):
    result = Chain([1, 2, 3]).foreach(X * 2 + 1).run()
    self.assertEqual(result, [3, 5, 7])

  def test_foreach_negation(self):
    result = Chain([1, -2, 3]).foreach(-X).run()
    self.assertEqual(result, [-1, 2, -3])

  def test_foreach_string_strip(self):
    result = Chain(['  a ', ' b  ', '  c  ']).foreach(X.strip()).run()
    self.assertEqual(result, ['a', 'b', 'c'])

  def test_foreach_dict_key_access(self):
    data = [{'v': 10}, {'v': 20}, {'v': 30}]
    result = Chain(data).foreach(X['v']).run()
    self.assertEqual(result, [10, 20, 30])

  def test_foreach_tuple_index(self):
    result = Chain([(1, 'a'), (2, 'b'), (3, 'c')]).foreach(X[1]).run()
    self.assertEqual(result, ['a', 'b', 'c'])

  def test_foreach_add(self):
    result = Chain([10, 20, 30]).foreach(X + 5).run()
    self.assertEqual(result, [15, 25, 35])

  def test_foreach_radd(self):
    result = Chain([1, 2, 3]).foreach(100 + X).run()
    self.assertEqual(result, [101, 102, 103])


# ---------------------------------------------------------------------------
# 6. X x foreach_do (sync)
# ---------------------------------------------------------------------------
class TestXForeachDoSync(unittest.TestCase):

  def test_foreach_do_preserves_original_elements(self):
    result = Chain([1, 2, 3]).foreach_do(X * 10).run()
    self.assertEqual(result, [1, 2, 3])

  def test_foreach_do_with_method_call(self):
    result = Chain(['a', 'b']).foreach_do(X.upper()).run()
    self.assertEqual(result, ['a', 'b'])


# ---------------------------------------------------------------------------
# 7. X x foreach (async)
# ---------------------------------------------------------------------------
class TestXForeachAsync(unittest.IsolatedAsyncioTestCase):

  async def test_foreach_x_with_async_iterable(self):
    result = await Chain(AsyncRange(4)).foreach(X * 10).run()
    self.assertEqual(result, [0, 10, 20, 30])

  async def test_foreach_x_power_async_iterable(self):
    result = await Chain(AsyncRange(5)).foreach(X ** 2).run()
    self.assertEqual(result, [0, 1, 4, 9, 16])

  async def test_foreach_x_after_async_root(self):
    async def async_val():
      return [1, 2, 3]
    result = await Chain(async_val).foreach(X + 100).run()
    self.assertEqual(result, [101, 102, 103])


# ---------------------------------------------------------------------------
# 8. X x filter (sync)
# ---------------------------------------------------------------------------
class TestXFilterSync(unittest.TestCase):

  def test_filter_even(self):
    result = Chain([1, 2, 3, 4]).filter(X % 2 == 0).run()
    self.assertEqual(result, [2, 4])

  def test_filter_gt(self):
    result = Chain([1, 2, 3, 4, 5]).filter(X > 3).run()
    self.assertEqual(result, [4, 5])

  def test_filter_ne_zero(self):
    result = Chain([0, 1, 2, 0, 3]).filter(X != 0).run()
    self.assertEqual(result, [1, 2, 3])

  def test_filter_gte(self):
    result = Chain([-3, -1, 0, 1, 3]).filter(X >= 0).run()
    self.assertEqual(result, [0, 1, 3])

  def test_filter_lt(self):
    result = Chain([1, 5, 10, 15, 20]).filter(X < 10).run()
    self.assertEqual(result, [1, 5])

  def test_filter_le(self):
    result = Chain([1, 5, 10, 15]).filter(X <= 10).run()
    self.assertEqual(result, [1, 5, 10])

  def test_filter_eq(self):
    result = Chain([1, 2, 3, 2, 1]).filter(X == 2).run()
    self.assertEqual(result, [2, 2])

  def test_filter_bitwise_odd(self):
    result = Chain([1, 2, 3, 4, 5, 6]).filter(X & 1 == 1).run()
    self.assertEqual(result, [1, 3, 5])

  def test_filter_compound_expr(self):
    """Filter where (X * 2) > 5."""
    result = Chain([1, 2, 3, 4]).filter(X * 2 > 5).run()
    self.assertEqual(result, [3, 4])

  def test_filter_then_foreach(self):
    result = Chain([1, 2, 3, 4]).filter(X > 2).foreach(X * 10).run()
    self.assertEqual(result, [30, 40])

  def test_filter_empty_result(self):
    result = Chain([1, 2, 3]).filter(X > 100).run()
    self.assertEqual(result, [])

  def test_filter_all_pass(self):
    result = Chain([1, 2, 3]).filter(X > 0).run()
    self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# 9. X x filter (async)
# ---------------------------------------------------------------------------
class TestXFilterAsync(unittest.IsolatedAsyncioTestCase):

  async def test_filter_x_async_iterable(self):
    result = await Chain(AsyncRange(6)).filter(X % 2 == 0).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_filter_x_gt_async(self):
    result = await Chain(AsyncRange(6)).filter(X > 3).run()
    self.assertEqual(result, [4, 5])

  async def test_filter_then_foreach_async(self):
    result = await Chain(AsyncRange(6)).filter(X > 2).foreach(X * 10).run()
    self.assertEqual(result, [30, 40, 50])


# ---------------------------------------------------------------------------
# 10. X x if_/else_ (sync)
# ---------------------------------------------------------------------------
class TestXIfElseSync(unittest.TestCase):

  def test_if_x_gt_true_path(self):
    result = Chain(5).if_(X > 3, X * 2).run()
    self.assertEqual(result, 10)

  def test_if_x_gt_false_passthrough(self):
    result = Chain(1).if_(X > 3, X * 2).run()
    self.assertEqual(result, 1)

  def test_if_x_else_true_path(self):
    result = Chain(5).if_(X > 3, X * 2).else_(X + 100).run()
    self.assertEqual(result, 10)

  def test_if_x_else_false_path(self):
    result = Chain(1).if_(X > 3, X * 2).else_(X + 100).run()
    self.assertEqual(result, 101)

  def test_if_x_truthy_list(self):
    result = Chain([1, 2, 3]).if_(X, X).run()
    self.assertEqual(result, [1, 2, 3])

  def test_if_x_falsy_empty_list(self):
    result = Chain([]).if_(X, X).else_(lambda v: 'empty').run()
    self.assertEqual(result, 'empty')

  def test_if_x_eq_predicate_true(self):
    result = Chain(0).if_(X == 0, X + 999).run()
    self.assertEqual(result, 999)

  def test_if_x_eq_predicate_false(self):
    result = Chain(5).if_(X == 0, X + 999).run()
    self.assertEqual(result, 5)

  def test_if_x_ne_predicate(self):
    result = Chain(5).if_(X != 0, X * 3).run()
    self.assertEqual(result, 15)

  def test_if_x_lt_predicate(self):
    result = Chain(2).if_(X < 10, X + 1).run()
    self.assertEqual(result, 3)

  def test_if_x_modulo_predicate(self):
    result = Chain(10).if_(X % 2 == 0, X // 2).run()
    self.assertEqual(result, 5)

  def test_if_x_modulo_predicate_false(self):
    result = Chain(11).if_(X % 2 == 0, X // 2).run()
    self.assertEqual(result, 11)

  def test_if_with_x_predicate_lambda_fn(self):
    result = Chain(5).if_(X > 3, lambda v: v * 100).run()
    self.assertEqual(result, 500)

  def test_if_with_lambda_predicate_x_fn(self):
    result = Chain(5).if_(lambda v: v > 3, X * 100).run()
    self.assertEqual(result, 500)

  def test_if_x_le_with_else(self):
    result = Chain(15).if_(X <= 10, X * 2).else_(X - 5).run()
    self.assertEqual(result, 10)

  def test_if_x_ge_with_else(self):
    result = Chain(5).if_(X >= 10, X * 2).else_(X + 10).run()
    self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# 11. X x if_/else_ (async)
# ---------------------------------------------------------------------------
class TestXIfElseAsync(unittest.IsolatedAsyncioTestCase):

  async def test_if_x_after_async_root(self):
    async def async_val():
      return 5
    result = await Chain(async_val).if_(X > 3, X * 2).run()
    self.assertEqual(result, 10)

  async def test_if_else_x_after_async_root_false(self):
    async def async_val():
      return 1
    result = await Chain(async_val).if_(X > 3, X * 2).else_(X + 100).run()
    self.assertEqual(result, 101)


# ---------------------------------------------------------------------------
# 12. X x gather (sync)
# ---------------------------------------------------------------------------
class TestXGatherSync(unittest.TestCase):

  def test_gather_multiple_x_exprs(self):
    result = Chain(5).gather(X + 1, X * 2, X ** 2).run()
    self.assertEqual(result, [6, 10, 25])

  def test_gather_x_sub_and_add(self):
    result = Chain(10).gather(X - 1, X + 1).run()
    self.assertEqual(result, [9, 11])

  def test_gather_mixed_x_and_lambda(self):
    result = Chain(5).gather(X + 1, lambda v: v * 3, X ** 2).run()
    self.assertEqual(result, [6, 15, 25])

  def test_gather_single_x_expr(self):
    result = Chain(7).gather(X * 2).run()
    self.assertEqual(result, [14])

  def test_gather_x_identity(self):
    result = Chain(42).gather(X, X, X).run()
    self.assertEqual(result, [42, 42, 42])

  def test_gather_x_negation_and_abs(self):
    result = Chain(-5).gather(-X, abs(X), X + 10).run()
    self.assertEqual(result, [5, 5, 5])


# ---------------------------------------------------------------------------
# 13. X x gather (async)
# ---------------------------------------------------------------------------
class TestXGatherAsync(unittest.IsolatedAsyncioTestCase):

  async def test_gather_x_after_async_root(self):
    async def async_val():
      return 5
    result = await Chain(async_val).gather(X + 1, X * 2).run()
    self.assertEqual(result, [6, 10])

  async def test_gather_mixed_x_and_async(self):
    async def double(v):
      return v * 2
    result = await Chain(5).gather(X + 1, double).run()
    self.assertEqual(result, [6, 10])


# ---------------------------------------------------------------------------
# 14. X x with_ (sync)
# ---------------------------------------------------------------------------
class TestXWithSync(unittest.TestCase):

  def test_with_x_identity_on_ctx(self):
    cm = SyncCM()
    result = Chain(cm).with_(X).run()
    self.assertEqual(result, 'ctx_value')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_with_x_add_on_ctx(self):
    """X + '_suffix' applied to ctx value (string)."""
    cm = TrackingCM()
    result = Chain(cm).with_(X + '_suffix').run()
    self.assertEqual(result, 'tracked_ctx_suffix')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_with_fn_then_x(self):
    cm = SyncCM()
    result = Chain(cm).with_(lambda ctx: ctx).then(X.upper()).run()
    self.assertEqual(result, 'CTX_VALUE')

  def test_with_x_upper_on_ctx(self):
    cm = SyncCM()
    result = Chain(cm).with_(X.upper()).run()
    self.assertEqual(result, 'CTX_VALUE')


# ---------------------------------------------------------------------------
# 15. X x with_ (async)
# ---------------------------------------------------------------------------
class TestXWithAsync(unittest.IsolatedAsyncioTestCase):

  async def test_with_x_identity_async_cm(self):
    cm = AsyncCM()
    result = await Chain(cm).with_(X).run()
    self.assertEqual(result, 'ctx_value')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_fn_then_x_async_cm(self):
    cm = AsyncCM()
    result = await Chain(cm).with_(lambda ctx: ctx).then(X.upper()).run()
    self.assertEqual(result, 'CTX_VALUE')


# ---------------------------------------------------------------------------
# 16. X x iterate (sync)
# ---------------------------------------------------------------------------
class TestXIterateSync(unittest.TestCase):

  def test_iterate_x_multiply(self):
    gen = Chain([1, 2, 3]).iterate(X * 10)
    result = list(gen)
    self.assertEqual(result, [10, 20, 30])

  def test_iterate_do_x_preserves_items(self):
    gen = Chain([1, 2, 3]).iterate_do(X * 10)
    result = list(gen)
    self.assertEqual(result, [1, 2, 3])

  def test_iterate_x_upper(self):
    gen = Chain(['a', 'b', 'c']).iterate(X.upper())
    result = list(gen)
    self.assertEqual(result, ['A', 'B', 'C'])

  def test_iterate_x_add(self):
    gen = Chain([10, 20, 30]).iterate(X + 5)
    result = list(gen)
    self.assertEqual(result, [15, 25, 35])


# ---------------------------------------------------------------------------
# 17. X x iterate (async)
# ---------------------------------------------------------------------------
class TestXIterateAsync(unittest.IsolatedAsyncioTestCase):

  async def test_iterate_x_async(self):
    gen = Chain(AsyncRange(4)).iterate(X * 10)
    result = [item async for item in gen]
    self.assertEqual(result, [0, 10, 20, 30])

  async def test_iterate_do_x_async(self):
    gen = Chain(AsyncRange(3)).iterate_do(X * 100)
    result = [item async for item in gen]
    self.assertEqual(result, [0, 1, 2])


# ---------------------------------------------------------------------------
# 18. X x except_ (sync)
# ---------------------------------------------------------------------------
class TestXExceptSync(unittest.TestCase):

  def test_except_x_identity_returns_exc(self):
    """X as identity handler: returns the caught exception itself."""
    def bad_fn(v):
      raise ValueError('test_err')
    result = Chain(5).then(bad_fn).except_(X).run()
    self.assertIsInstance(result, ValueError)
    self.assertIn('test_err', str(result))

  def test_except_x_expr_on_exc(self):
    """X expression applied to the exception object."""
    def bad_fn(v):
      raise ValueError('something went wrong')
    # X.args returns _XAttr; when called with single arg it replays
    # so X.args on ValueError -> ('something went wrong',)
    result = Chain(5).then(bad_fn).except_(X.args).run()
    self.assertEqual(result, ('something went wrong',))

  def test_except_x_with_lambda_mix(self):
    def bad_fn(v):
      raise ValueError('err')
    result = Chain(5).then(bad_fn).except_(lambda exc: str(exc)).run()
    self.assertIn('err', result)

  def test_except_does_not_fire_without_error(self):
    result = Chain(5).then(X + 1).except_(X).run()
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# 19. X x except_ (async)
# ---------------------------------------------------------------------------
class TestXExceptAsync(unittest.IsolatedAsyncioTestCase):

  async def test_except_x_identity_async(self):
    async def bad_fn(v):
      raise ValueError('async_err')
    result = await Chain(5).then(bad_fn).except_(X).run()
    self.assertIsInstance(result, ValueError)
    self.assertIn('async_err', str(result))


# ---------------------------------------------------------------------------
# 20. X x finally_ (sync)
# ---------------------------------------------------------------------------
class TestXFinallySync(unittest.TestCase):

  def test_finally_x_identity_runs(self):
    """X in finally_: it receives the root value and runs."""
    tracker = []
    def track(v):
      tracker.append(v)
    result = Chain(5).then(X + 1).finally_(track).run()
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [5])

  def test_finally_x_does_not_alter_result(self):
    """finally_ with X expression doesn't change the return value."""
    result = Chain(5).then(X + 1).finally_(X * 100).run()
    self.assertEqual(result, 6)

  def test_finally_runs_on_exception_too(self):
    tracker = []
    def bad_fn(v):
      raise ValueError('err')
    def track(v):
      tracker.append(v)
    with self.assertRaises(ValueError):
      Chain(5).then(bad_fn).finally_(track).run()
    self.assertEqual(tracker, [5])


# ---------------------------------------------------------------------------
# 21. X replay error during chain execution
# ---------------------------------------------------------------------------
class TestXReplayErrorsInChain(unittest.TestCase):

  def test_x_item_access_on_non_subscriptable(self):
    """X[0] on an int raises TypeError during chain execution."""
    with self.assertRaises(TypeError):
      Chain(42).then(X[0]).run()

  def test_x_upper_on_non_string(self):
    with self.assertRaises(AttributeError):
      Chain(42).then(X.upper()).run()

  def test_x_missing_dict_key_in_chain(self):
    with self.assertRaises(KeyError):
      Chain({'a': 1}).then(X['b']).run()


# ---------------------------------------------------------------------------
# 22. X x freeze (sync)
# ---------------------------------------------------------------------------
class TestXFreezeSync(unittest.TestCase):

  def test_frozen_chain_with_x(self):
    frozen = Chain().then(X + 1).freeze()
    self.assertEqual(frozen(5), 6)
    self.assertEqual(frozen(10), 11)

  def test_frozen_chain_reuse_with_different_values(self):
    frozen = Chain().then(X * 2).then(X + 1).freeze()
    self.assertEqual(frozen(3), 7)
    self.assertEqual(frozen(10), 21)
    self.assertEqual(frozen(0), 1)

  def test_frozen_filter_with_x(self):
    frozen = Chain().filter(X > 2).freeze()
    self.assertEqual(frozen([1, 2, 3, 4, 5]), [3, 4, 5])
    self.assertEqual(frozen([10, 0, 1]), [10])

  def test_frozen_foreach_with_x(self):
    frozen = Chain().foreach(X * 10).freeze()
    self.assertEqual(frozen([1, 2, 3]), [10, 20, 30])
    self.assertEqual(frozen([5, 6]), [50, 60])

  def test_frozen_if_with_x(self):
    frozen = Chain().if_(X > 5, X * 2).else_(X + 100).freeze()
    self.assertEqual(frozen(10), 20)
    self.assertEqual(frozen(3), 103)

  def test_frozen_gather_with_x(self):
    frozen = Chain().gather(X + 1, X * 2).freeze()
    self.assertEqual(frozen(5), [6, 10])
    self.assertEqual(frozen(10), [11, 20])


# ---------------------------------------------------------------------------
# 23. X x freeze concurrent use
# ---------------------------------------------------------------------------
class TestXFreezeConcurrent(unittest.IsolatedAsyncioTestCase):

  async def test_frozen_concurrent_x(self):
    frozen = Chain().then(X * 2).then(X + 1).freeze()

    async def run_frozen(val):
      return frozen(val)

    results = await asyncio.gather(
      run_frozen(1), run_frozen(2), run_frozen(3),
      run_frozen(4), run_frozen(5),
    )
    self.assertEqual(results, [3, 5, 7, 9, 11])


# ---------------------------------------------------------------------------
# 24. X x decorator
# ---------------------------------------------------------------------------
class TestXDecorator(unittest.TestCase):

  def test_decorator_with_x_then(self):
    @Chain().then(X + 1).decorator()
    def add_one(v):
      return v
    self.assertEqual(add_one(5), 6)
    self.assertEqual(add_one(10), 11)

  def test_decorator_with_x_multiply(self):
    @Chain().then(X * 3).decorator()
    def triple(v):
      return v
    self.assertEqual(triple(4), 12)

  def test_decorator_with_x_filter(self):
    @Chain().filter(X > 2).decorator()
    def filter_fn(lst):
      return lst
    self.assertEqual(filter_fn([1, 2, 3, 4, 5]), [3, 4, 5])

  def test_decorator_with_x_foreach(self):
    @Chain().foreach(X * 10).decorator()
    def map_fn(lst):
      return lst
    self.assertEqual(map_fn([1, 2, 3]), [10, 20, 30])


# ---------------------------------------------------------------------------
# 25. X x complex pipelines (sync)
# ---------------------------------------------------------------------------
class TestXComplexPipelinesSync(unittest.TestCase):

  def test_filter_then_foreach_then_then(self):
    result = Chain([1, 2, 3, 4, 5]).filter(X % 2 == 0).foreach(X ** 2).run()
    self.assertEqual(result, [4, 16])

  def test_chain_x_ops_item_then_add(self):
    result = Chain([10, 20, 30]).then(X[0]).then(X + 5).run()
    self.assertEqual(result, 15)

  def test_dict_pipeline_x(self):
    result = Chain({'name': 'Alice', 'age': 30}).then(X['name']).then(X.upper()).run()
    self.assertEqual(result, 'ALICE')

  def test_filter_then_identity_passthrough(self):
    result = Chain([3, 1, 4, 1, 5]).filter(X > 2).then(X).run()
    self.assertEqual(result, [3, 4, 5])

  def test_filter_gt_then_foreach_mul(self):
    result = Chain([1, 2, 3, 4, 5]).filter(X >= 3).foreach(X * 100).run()
    self.assertEqual(result, [300, 400, 500])

  def test_multiple_then_x_chained(self):
    result = Chain(2).then(X + 1).then(X * 2).then(X - 1).then(X ** 2).run()
    # 2 -> 3 -> 6 -> 5 -> 25
    self.assertEqual(result, 25)

  def test_foreach_x_then_filter_x(self):
    result = Chain([1, 2, 3, 4]).foreach(X * 3).then(lambda v: Chain(v).filter(X > 5).run()).run()
    # [3, 6, 9, 12] -> filter > 5 -> [6, 9, 12]
    self.assertEqual(result, [6, 9, 12])

  def test_then_x_item_then_x_item(self):
    data = {'users': [{'name': 'Alice'}, {'name': 'Bob'}]}
    result = Chain(data).then(X['users']).then(X[0]).then(X['name']).run()
    self.assertEqual(result, 'Alice')

  def test_if_then_foreach(self):
    """if_ predicate checks truthy list, then foreach maps it."""
    result = Chain([1, 2, 3]).if_(X, lambda v: Chain(v).foreach(X * 2).run()).run()
    self.assertEqual(result, [2, 4, 6])

  def test_gather_then_filter(self):
    """Gather produces a list, then filter that list."""
    result = Chain(5).gather(X + 1, X - 1, X * 2, X * 3).then(lambda v: Chain(v).filter(X > 5).run()).run()
    # [6, 4, 10, 15] -> filter > 5 -> [6, 10, 15]
    self.assertEqual(result, [6, 10, 15])


# ---------------------------------------------------------------------------
# 26. X x complex pipelines (async)
# ---------------------------------------------------------------------------
class TestXComplexPipelinesAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_filter_then_foreach(self):
    result = await Chain(AsyncRange(8)).filter(X % 2 == 0).foreach(X + 100).run()
    self.assertEqual(result, [100, 102, 104, 106])

  async def test_async_chain_with_x_then_steps(self):
    async def async_val():
      return 10
    result = await Chain(async_val).then(X * 2).then(X + 5).run()
    self.assertEqual(result, 25)

  async def test_async_gather_with_x(self):
    async def async_val():
      return 7
    result = await Chain(async_val).gather(X + 1, X * 2, X - 1).run()
    self.assertEqual(result, [8, 14, 6])


# ---------------------------------------------------------------------------
# 27. X as XAttr behavior in chain methods
# ---------------------------------------------------------------------------
class TestXAttrInChain(unittest.TestCase):

  def test_xattr_in_then_is_replay(self):
    """X.strip (no parens) in then: _XAttr called with 1 arg -> replay."""
    result = Chain('  hello  ').then(X.strip).run()
    # Replay: getattr('  hello  ', 'strip') -> bound method
    self.assertTrue(callable(result))

  def test_xattr_method_zero_args_in_then(self):
    """X.strip() (parens) in then: proper method call, returns stripped."""
    result = Chain('  hello  ').then(X.strip()).run()
    self.assertEqual(result, 'hello')

  def test_xattr_method_multi_args_in_then(self):
    """X.replace('a', 'b') with 2 args works fine."""
    result = Chain('banana').then(X.replace('a', 'b')).run()
    self.assertEqual(result, 'bbnbnb')

  def test_xattr_kwargs_in_then(self):
    result = Chain('hello').then(X.encode(encoding='ascii')).run()
    self.assertEqual(result, b'hello')

  def test_xattr_chained_methods(self):
    result = Chain('  Hello World  ').then(X.strip().lower()).run()
    self.assertEqual(result, 'hello world')


# ---------------------------------------------------------------------------
# 28. X x comparison operators in filter (exhaustive)
# ---------------------------------------------------------------------------
class TestXComparisonInFilter(unittest.TestCase):

  def test_filter_x_lt(self):
    self.assertEqual(Chain([1, 5, 10]).filter(X < 5).run(), [1])

  def test_filter_x_le(self):
    self.assertEqual(Chain([1, 5, 10]).filter(X <= 5).run(), [1, 5])

  def test_filter_x_gt(self):
    self.assertEqual(Chain([1, 5, 10]).filter(X > 5).run(), [10])

  def test_filter_x_ge(self):
    self.assertEqual(Chain([1, 5, 10]).filter(X >= 5).run(), [5, 10])

  def test_filter_x_eq(self):
    self.assertEqual(Chain([1, 2, 3, 2, 1]).filter(X == 2).run(), [2, 2])

  def test_filter_x_ne(self):
    self.assertEqual(Chain([1, 2, 3, 2, 1]).filter(X != 2).run(), [1, 3, 1])


# ---------------------------------------------------------------------------
# 29. X x reverse operators in chain
# ---------------------------------------------------------------------------
class TestXReverseOpsInChain(unittest.TestCase):

  def test_then_radd_in_chain(self):
    self.assertEqual(Chain(5).then(10 + X).run(), 15)

  def test_then_rsub_in_chain(self):
    self.assertEqual(Chain(3).then(10 - X).run(), 7)

  def test_then_rmul_in_chain(self):
    self.assertEqual(Chain(4).then(3 * X).run(), 12)

  def test_then_rfloordiv_in_chain(self):
    self.assertEqual(Chain(3).then(10 // X).run(), 3)

  def test_then_rmod_in_chain(self):
    self.assertEqual(Chain(7).then(100 % X).run(), 2)

  def test_then_rpow_in_chain(self):
    self.assertEqual(Chain(3).then(2 ** X).run(), 8)


# ---------------------------------------------------------------------------
# 30. X identity edge cases in chain
# ---------------------------------------------------------------------------
class TestXIdentityInChain(unittest.TestCase):

  def test_x_identity_none(self):
    result = Chain(None).then(X).run()
    self.assertIsNone(result)

  def test_x_identity_false(self):
    result = Chain(False).then(X).run()
    self.assertIs(result, False)

  def test_x_identity_zero(self):
    result = Chain(0).then(X).run()
    self.assertEqual(result, 0)

  def test_x_identity_empty_string(self):
    result = Chain('').then(X).run()
    self.assertEqual(result, '')

  def test_x_identity_empty_list(self):
    result = Chain([]).then(X).run()
    self.assertEqual(result, [])

  def test_x_identity_empty_dict(self):
    result = Chain({}).then(X).run()
    self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# 31. X x bitwise ops in chain
# ---------------------------------------------------------------------------
class TestXBitwiseInChain(unittest.TestCase):

  def test_then_x_and(self):
    self.assertEqual(Chain(0xFF).then(X & 0x0F).run(), 0x0F)

  def test_then_x_or(self):
    self.assertEqual(Chain(0xF0).then(X | 0x0F).run(), 0xFF)

  def test_then_x_xor(self):
    self.assertEqual(Chain(0x0F).then(X ^ 0xFF).run(), 0xF0)

  def test_then_x_lshift(self):
    self.assertEqual(Chain(1).then(X << 4).run(), 16)

  def test_then_x_rshift(self):
    self.assertEqual(Chain(16).then(X >> 4).run(), 1)

  def test_then_x_invert(self):
    self.assertEqual(Chain(0).then(~X).run(), -1)


# ---------------------------------------------------------------------------
# 32. X x unary ops in foreach
# ---------------------------------------------------------------------------
class TestXUnaryInForeach(unittest.TestCase):

  def test_foreach_negation(self):
    self.assertEqual(Chain([1, -2, 3]).foreach(-X).run(), [-1, 2, -3])

  def test_foreach_abs(self):
    self.assertEqual(Chain([-5, 3, -1]).foreach(abs(X)).run(), [5, 3, 1])

  def test_foreach_pos(self):
    self.assertEqual(Chain([1, -2, 3]).foreach(+X).run(), [1, -2, 3])

  def test_foreach_invert(self):
    self.assertEqual(Chain([0, 1, 2]).foreach(~X).run(), [-1, -2, -3])


# ---------------------------------------------------------------------------
# 33. X expressions reuse across calls (statelessness)
# ---------------------------------------------------------------------------
class TestXExprReuse(unittest.TestCase):

  def test_same_x_expr_in_multiple_chains(self):
    expr = X + 1
    self.assertEqual(Chain(5).then(expr).run(), 6)
    self.assertEqual(Chain(10).then(expr).run(), 11)

  def test_same_filter_expr_reused(self):
    pred = X > 3
    self.assertEqual(Chain([1, 2, 3, 4, 5]).filter(pred).run(), [4, 5])
    self.assertEqual(Chain([0, 10, 3, 7]).filter(pred).run(), [10, 7])

  def test_x_expr_independence(self):
    """Two different X expressions don't share state."""
    e1 = X + 1
    e2 = X * 2
    self.assertEqual(e1(10), 11)
    self.assertEqual(e2(10), 20)
    self.assertEqual(e1(10), 11)


# ---------------------------------------------------------------------------
# 34. X x with_do (sync)
# ---------------------------------------------------------------------------
class TestXWithDoSync(unittest.TestCase):

  def test_with_do_x_preserves_cm(self):
    """with_do discards the body result, returns the CM itself."""
    cm = SyncCM()
    result = Chain(cm).with_do(X).run()
    # with_do returns the outer value (the CM)
    self.assertIs(result, cm)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# 35. X x iterate with run args
# ---------------------------------------------------------------------------
class TestXIterateWithRunArgs(unittest.TestCase):

  def test_iterate_with_run_value(self):
    gen = Chain().iterate(X * 2)
    result = list(gen([1, 2, 3]))
    self.assertEqual(result, [2, 4, 6])

  def test_iterate_do_with_run_value(self):
    gen = Chain().iterate_do(X * 2)
    result = list(gen([1, 2, 3]))
    self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# 36. X x chain called directly (no .run())
# ---------------------------------------------------------------------------
class TestXChainCallable(unittest.TestCase):

  def test_chain_call_with_x(self):
    c = Chain().then(X + 10)
    self.assertEqual(c(5), 15)
    self.assertEqual(c(0), 10)

  def test_chain_call_filter_x(self):
    c = Chain().filter(X > 2)
    self.assertEqual(c([1, 2, 3, 4]), [3, 4])


# ---------------------------------------------------------------------------
# 37. X x multiple if_/else_ in pipeline
# ---------------------------------------------------------------------------
class TestXMultipleIfElse(unittest.TestCase):

  def test_chained_if_else_blocks(self):
    """Two if_ blocks in a row."""
    result = (
      Chain(5)
        .if_(X > 3, X + 10)
        .if_(X > 10, X * 2)
        .run()
    )
    # 5 -> if > 3: 5+10=15 -> if > 10: 15*2=30
    self.assertEqual(result, 30)

  def test_chained_if_else_second_false(self):
    result = (
      Chain(5)
        .if_(X > 3, X + 1)
        .if_(X > 10, X * 2)
        .run()
    )
    # 5 -> if > 3: 5+1=6 -> if > 10: 6 (passthrough)
    self.assertEqual(result, 6)

  def test_chained_if_else_first_false(self):
    result = (
      Chain(2)
        .if_(X > 3, X + 10).else_(X * 100)
        .if_(X > 100, X - 50)
        .run()
    )
    # 2 -> if > 3: else 2*100=200 -> if > 100: 200-50=150
    self.assertEqual(result, 150)


# ---------------------------------------------------------------------------
# 38. X x run() with value
# ---------------------------------------------------------------------------
class TestXRunWithValue(unittest.TestCase):

  def test_run_with_value_x_add(self):
    c = Chain().then(X + 1)
    self.assertEqual(c.run(5), 6)
    self.assertEqual(c.run(10), 11)

  def test_run_with_value_overrides_root(self):
    c = Chain(999).then(X + 1)
    # run(5) provides the value, overriding root
    self.assertEqual(c.run(5), 6)


# ---------------------------------------------------------------------------
# 39. X in deeply nested data extraction
# ---------------------------------------------------------------------------
class TestXDeepDataExtraction(unittest.TestCase):

  def test_nested_dict_list_dict(self):
    data = {'results': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]}
    result = Chain(data).then(X['results']).then(X[1]).then(X['name']).run()
    self.assertEqual(result, 'Bob')

  def test_nested_list_list(self):
    data = [[1, 2], [3, 4], [5, 6]]
    result = Chain(data).then(X[2]).then(X[1]).run()
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# 40. X x empty chain edge cases
# ---------------------------------------------------------------------------
class TestXEmptyChainEdgeCases(unittest.TestCase):

  def test_chain_only_root_x_identity(self):
    """Chain(5) with just .then(X) returns 5."""
    self.assertEqual(Chain(5).then(X).run(), 5)

  def test_chain_no_root_then_x(self):
    """Chain().then(X) called with run(5) passes through."""
    self.assertEqual(Chain().then(X).run(5), 5)


if __name__ == '__main__':
  unittest.main()
