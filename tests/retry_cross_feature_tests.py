"""Exhaustive tests for retry's interaction with foreach, filter, gather,
with_, iterate, freeze, decorator, nested chains, X expressions, and edge cases.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException, X
from helpers import AsyncRange, AsyncEmpty


# ---------------------------------------------------------------------------
# Helpers local to this test file
# ---------------------------------------------------------------------------

def _make_flaky(fail_count, exc_type=ValueError, exc_msg='flaky'):
  """Return a callable that fails `fail_count` times then succeeds.

  Uses a list as a mutable counter so it works across retry attempts.
  Returns (fn, attempts_list).
  """
  attempts = []

  def fn(x):
    attempts.append(x)
    if len(attempts) <= fail_count:
      raise exc_type(exc_msg)
    return x

  return fn, attempts


def _make_flaky_no_arg(fail_count, exc_type=ValueError, exc_msg='flaky'):
  """Like _make_flaky but takes no arguments."""
  attempts = []

  def fn():
    attempts.append(1)
    if len(attempts) <= fail_count:
      raise exc_type(exc_msg)
    return 'ok'

  return fn, attempts


def _make_async_flaky(fail_count, exc_type=ValueError, exc_msg='flaky'):
  """Async version of _make_flaky."""
  attempts = []

  async def fn(x):
    attempts.append(x)
    if len(attempts) <= fail_count:
      raise exc_type(exc_msg)
    return x

  return fn, attempts


def _make_async_flaky_no_arg(fail_count, exc_type=ValueError, exc_msg='flaky'):
  """Async version of _make_flaky_no_arg."""
  attempts = []

  async def fn():
    attempts.append(1)
    if len(attempts) <= fail_count:
      raise exc_type(exc_msg)
    return 'ok'

  return fn, attempts


# ---------------------------------------------------------------------------
# Category 1: retry x foreach()
# ---------------------------------------------------------------------------

class TestRetryForeach(unittest.TestCase):

  def test_foreach_raises_then_succeeds_on_retry(self):
    """foreach callback raises on some attempts, succeeds after retry."""
    attempts = []

    def flaky_map(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('retry')
      return x * 2

    result = Chain([1, 2, 3]).foreach(flaky_map).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [2, 4, 6])
    # First two attempts fail on item 1, third attempt succeeds for all items.
    self.assertEqual(attempts, [1, 1, 1, 2, 3])

  def test_foreach_iterable_is_reiterated_on_retry(self):
    """Verify the iterable is re-iterated from scratch on each retry attempt."""
    iteration_starts = []

    class TrackingIterable:
      def __iter__(self):
        iteration_starts.append(1)
        return iter([10, 20, 30])

    attempts = []

    def flaky_map(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('retry')
      return x

    result = Chain(TrackingIterable()).foreach(flaky_map).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [10, 20, 30])
    # The iterable was iterated 3 times (2 failures + 1 success).
    self.assertEqual(len(iteration_starts), 3)

  def test_foreach_break_is_not_retried(self):
    """break_() is a _ControlFlowSignal and must NOT be retried."""
    attempts = []

    def fn_with_break(x):
      attempts.append(x)
      if x == 2:
        Chain.break_()
      return x * 10

    result = Chain([1, 2, 3]).foreach(fn_with_break).retry(5, on=(Exception,)).run()
    self.assertEqual(result, [10])
    # break_ triggers on first encounter; no retry.
    self.assertEqual(attempts, [1, 2])

  def test_foreach_do_side_effects_happen_each_attempt(self):
    """foreach_do with retry: side effects happen on every attempt."""
    side_effects = []

    def flaky_side_effect(x):
      side_effects.append(x)
      if len(side_effects) < 4:
        raise ValueError('retry')

    result = Chain([1, 2, 3]).foreach_do(flaky_side_effect).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [1, 2, 3])
    # Side effects from failed attempts plus the successful one.
    self.assertEqual(side_effects, [1, 1, 1, 1, 2, 3])

  def test_foreach_raises_on_specific_item_full_reexecute(self):
    """foreach raises on a specific item -- full chain re-executes from scratch."""
    attempts = []

    def fail_on_second(x):
      attempts.append(x)
      if x == 2 and len(attempts) < 4:
        raise ValueError('item 2 fails')
      return x * 10

    result = Chain([1, 2, 3]).foreach(fail_on_second).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [10, 20, 30])
    # Attempt 1: 1 ok (len=1), 2 fails (len=2, <4). Attempt 2: 1 ok (len=3), 2 ok (len=4, not <4), 3 ok.
    self.assertEqual(attempts, [1, 2, 1, 2, 3])


class TestRetryForeachAsync(IsolatedAsyncioTestCase):

  async def test_foreach_async_callback_retry(self):
    """foreach with async callback + retry."""
    attempts = []

    async def async_flaky_map(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('retry')
      return x * 2

    result = await Chain([1, 2, 3]).foreach(async_flaky_map).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [2, 4, 6])
    self.assertEqual(attempts, [1, 1, 1, 2, 3])

  async def test_foreach_async_iterable_retry(self):
    """foreach with async iterable + retry."""
    attempts = []

    def flaky_map(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 3

    result = await Chain(AsyncRange(3)).foreach(flaky_map).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [0, 3, 6])

  async def test_foreach_do_async_retry(self):
    """foreach_do with async callback + retry."""
    side_effects = []

    async def async_side(x):
      side_effects.append(x)
      if len(side_effects) < 2:
        raise ValueError('retry')

    result = await Chain([10, 20]).foreach_do(async_side).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [10, 20])
    # Attempt 1: item 10 fails. Attempt 2: item 10 ok, item 20 ok.
    self.assertEqual(side_effects, [10, 10, 20])

  async def test_foreach_break_not_retried_async(self):
    """break_() is not retried even in async path."""
    attempts = []

    async def fn_with_break(x):
      attempts.append(x)
      if x == 1:
        Chain.break_()
      return x * 10

    result = await Chain(AsyncRange(5)).foreach(fn_with_break).retry(5, on=(Exception,)).run()
    self.assertEqual(result, [0])
    self.assertEqual(attempts, [0, 1])


# ---------------------------------------------------------------------------
# Category 2: retry x filter()
# ---------------------------------------------------------------------------

class TestRetryFilter(unittest.TestCase):

  def test_filter_raises_then_succeeds_on_retry(self):
    """filter predicate raises on some attempts, succeeds on retry."""
    attempts = []

    def flaky_pred(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x > 2

    result = Chain([1, 2, 3, 4]).filter(flaky_pred).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [3, 4])

  def test_filter_refilters_from_scratch_on_retry(self):
    """filter re-filters from scratch after retry."""
    attempts = []

    def flaky_pred(x):
      attempts.append(x)
      if x == 3 and len(attempts) < 5:
        raise ValueError('retry on 3')
      return x % 2 == 0

    result = Chain([1, 2, 3, 4]).filter(flaky_pred).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [2, 4])
    # Attempt 1: 1(len=1), 2(len=2), 3(len=3, <5 -> fail).
    # Attempt 2: 1(len=4), 2(len=5), 3(len=6, not <5 -> ok), 4(len=7).
    self.assertEqual(attempts, [1, 2, 3, 1, 2, 3, 4])

  def test_filter_correct_output_after_retry(self):
    """Verify filter produces correct output list after retry succeeds."""
    fn, attempts = _make_flaky(1, ValueError)
    result = Chain([10, 20, 30]).filter(lambda x: x > 15).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, [20, 30])


class TestRetryFilterAsync(IsolatedAsyncioTestCase):

  async def test_filter_async_predicate_retry(self):
    """filter with async predicate + retry."""
    attempts = []

    async def async_pred(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('retry')
      return x > 2

    result = await Chain([1, 2, 3, 4]).filter(async_pred).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [3, 4])

  async def test_filter_async_iterable_retry(self):
    """filter on async iterable + retry."""
    attempts = []

    def flaky_pred(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x >= 2

    result = await Chain(AsyncRange(5)).filter(flaky_pred).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [2, 3, 4])


# ---------------------------------------------------------------------------
# Category 3: retry x gather()
# ---------------------------------------------------------------------------

class TestRetryGather(unittest.TestCase):

  def test_gather_one_function_fails_retry_reruns_all(self):
    """gather with one failing function: retry re-runs all functions."""
    attempts = []

    def fn1(x):
      attempts.append(('fn1', x))
      if len(attempts) < 3:
        raise ValueError('retry')
      return x + 1

    def fn2(x):
      attempts.append(('fn2', x))
      return x + 2

    result = Chain(10).gather(fn1, fn2).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [11, 12])
    # fn1 is called first. On failure, fn2 is not called. Re-run: fn1, fn1+fn2 succeed.
    self.assertEqual(attempts, [('fn1', 10), ('fn1', 10), ('fn1', 10), ('fn2', 10)])

  def test_gather_different_functions_fail_different_attempts(self):
    """gather where different functions fail on different attempts."""
    attempts = []

    def fn1(x):
      attempts.append(('fn1', len(attempts)))
      # fn1 fails on first call
      if sum(1 for a in attempts if a[0] == 'fn1') < 2:
        raise ValueError('fn1 fail')
      return x + 1

    def fn2(x):
      attempts.append(('fn2', len(attempts)))
      return x + 2

    result = Chain(10).gather(fn1, fn2).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [11, 12])


class TestRetryGatherAsync(IsolatedAsyncioTestCase):

  async def test_gather_async_functions_retry(self):
    """gather with async functions + retry."""
    attempts = []

    async def fn1(x):
      attempts.append(('fn1', x))
      if sum(1 for a in attempts if a[0] == 'fn1') < 2:
        raise ValueError('retry')
      return x + 1

    async def fn2(x):
      attempts.append(('fn2', x))
      return x + 2

    result = await Chain(10).gather(fn1, fn2).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [11, 12])

  async def test_gather_mixed_async_sync_retry(self):
    """gather with mix of sync and async + retry."""
    attempts = []

    def sync_fn(x):
      attempts.append(('sync', x))
      if sum(1 for a in attempts if a[0] == 'sync') < 2:
        raise ValueError('sync fail')
      return x * 2

    async def async_fn(x):
      return x * 3

    result = await Chain(5).gather(sync_fn, async_fn).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [10, 15])

  async def test_gather_all_results_correct_after_retry(self):
    """Verify all gathered results are correct after retry."""
    call_count = [0]

    async def fn1(x):
      call_count[0] += 1
      if call_count[0] < 3:
        raise ValueError('retry')
      return x + 10

    async def fn2(x):
      return x + 20

    async def fn3(x):
      return x + 30

    result = await Chain(100).gather(fn1, fn2, fn3).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, [110, 120, 130])


# ---------------------------------------------------------------------------
# Category 4: retry x with_() (context managers)
# ---------------------------------------------------------------------------

class TestRetryWith(unittest.TestCase):

  def test_with_body_raises_retry_reenters_cm(self):
    """with_ where body raises: retry re-enters context manager."""
    enter_count = [0]
    exit_count = [0]
    exit_args_list = []

    class TrackCM:
      def __enter__(self):
        enter_count[0] += 1
        return f'ctx_{enter_count[0]}'

      def __exit__(self, *args):
        exit_count[0] += 1
        exit_args_list.append(args)
        return False

    cm = TrackCM()
    attempts = []

    def body(ctx):
      attempts.append(ctx)
      if len(attempts) < 3:
        raise ValueError('retry body')
      return f'result_{ctx}'

    result = Chain(cm).with_(body).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, 'result_ctx_3')
    self.assertEqual(enter_count[0], 3)
    self.assertEqual(exit_count[0], 3)
    # First two exits should have exception info, last one should have None.
    self.assertIsNotNone(exit_args_list[0][1])
    self.assertIsNotNone(exit_args_list[1][1])
    self.assertIsNone(exit_args_list[2][1])

  def test_with_do_retry(self):
    """with_do + retry: side effect, original value preserved."""
    enter_count = [0]

    class SimpleCM:
      def __enter__(self):
        enter_count[0] += 1
        return f'ctx_{enter_count[0]}'

      def __exit__(self, *args):
        return False

    cm = SimpleCM()
    side_effects = []

    def body(ctx):
      side_effects.append(ctx)
      if len(side_effects) < 2:
        raise ValueError('retry')
      return 'ignored_result'

    result = Chain(cm).with_do(body).retry(5, on=(ValueError,)).run()
    # with_do discards body result, returns original CM value.
    self.assertIs(result, cm)
    self.assertEqual(enter_count[0], 2)
    self.assertEqual(side_effects, ['ctx_1', 'ctx_2'])

  def test_with_exit_receives_exc_on_failed_attempts(self):
    """Verify __exit__ receives exception info on failed attempts."""
    exit_exc_types = []

    class RecordingCM:
      def __enter__(self):
        return 'ctx'

      def __exit__(self, exc_type, exc_val, exc_tb):
        exit_exc_types.append(exc_type)
        return False

    cm = RecordingCM()
    attempts = []

    def body(ctx):
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('fail')
      return 'ok'

    Chain(cm).with_(body).retry(5, on=(ValueError,)).run()
    # 2 failures + 1 success.
    self.assertEqual(exit_exc_types, [ValueError, ValueError, None])


class TestRetryWithAsync(IsolatedAsyncioTestCase):

  async def test_with_async_cm_retry(self):
    """with_ with async context manager + retry."""
    enter_count = [0]
    exit_count = [0]

    class AsyncTrackCM:
      async def __aenter__(self):
        enter_count[0] += 1
        return f'actx_{enter_count[0]}'

      async def __aexit__(self, *args):
        exit_count[0] += 1
        return False

    cm = AsyncTrackCM()
    attempts = []

    async def body(ctx):
      attempts.append(ctx)
      if len(attempts) < 2:
        raise ValueError('retry')
      return f'result_{ctx}'

    result = await Chain(cm).with_(body).retry(5, on=(ValueError,)).run()
    self.assertEqual(result, 'result_actx_2')
    self.assertEqual(enter_count[0], 2)
    self.assertEqual(exit_count[0], 2)

  async def test_with_do_async_retry(self):
    """with_do with async body + retry."""
    enter_count = [0]

    class AsyncSimpleCM:
      async def __aenter__(self):
        enter_count[0] += 1
        return f'actx_{enter_count[0]}'

      async def __aexit__(self, *args):
        return False

    cm = AsyncSimpleCM()
    side_effects = []

    async def body(ctx):
      side_effects.append(ctx)
      if len(side_effects) < 2:
        raise ValueError('retry')

    result = await Chain(cm).with_do(body).retry(5, on=(ValueError,)).run()
    self.assertIs(result, cm)
    self.assertEqual(side_effects, ['actx_1', 'actx_2'])


# ---------------------------------------------------------------------------
# Category 5: retry x iterate()
# ---------------------------------------------------------------------------

class TestRetryIterate(unittest.TestCase):

  def test_iterate_source_chain_retries(self):
    """iterate() on a chain with retry: source chain retries on failure."""
    fn, attempts = _make_flaky_no_arg(2)

    def source():
      return fn()

    attempts2 = []

    def actual_source():
      attempts2.append(1)
      if len(attempts2) < 3:
        raise ValueError('retry')
      return [1, 2, 3]

    gen = Chain(actual_source, ...).retry(5, on=(ValueError,)).iterate()
    result = list(gen)
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(len(attempts2), 3)

  def test_iterate_with_fn_and_retry(self):
    """iterate(fn) with retry: source chain retries, fn applied per item."""
    attempts = []

    def flaky_source():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return [10, 20]

    gen = Chain(flaky_source, ...).retry(3, on=(ValueError,)).iterate(lambda x: x * 2)
    result = list(gen)
    self.assertEqual(result, [20, 40])


class TestRetryIterateAsync(IsolatedAsyncioTestCase):

  async def test_iterate_async_source_retry(self):
    """async iterate on chain with retry."""
    attempts = []

    async def flaky_source():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return [10, 20, 30]

    gen = Chain(flaky_source, ...).retry(3, on=(ValueError,)).iterate()
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(len(attempts), 2)

  async def test_iterate_async_fn_retry(self):
    """async iterate with async fn applied per item, source retries."""
    attempts = []

    async def flaky_source():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return [1, 2]

    gen = Chain(flaky_source, ...).retry(3, on=(ValueError,)).iterate(lambda x: x * 10)
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [10, 20])


# ---------------------------------------------------------------------------
# Category 6: retry x freeze()
# ---------------------------------------------------------------------------

class TestRetryFreeze(unittest.TestCase):

  def test_freeze_preserves_retry_config(self):
    """Frozen chain preserves retry config and retries correctly."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('retry')
      return x * 2

    frozen = Chain().then(flaky).retry(5, on=(ValueError,)).freeze()
    result = frozen(10)
    self.assertEqual(result, 20)
    self.assertEqual(attempts, [10, 10, 10])

  def test_freeze_after_retry_config_preserved(self):
    """freeze() after retry() preserves the retry config."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x + 100

    c = Chain().then(flaky)
    c.retry(3, on=(ValueError,))
    frozen = c.freeze()
    result = frozen(5)
    self.assertEqual(result, 105)
    self.assertEqual(len(attempts), 2)

  def test_frozen_chain_independent_per_invocation(self):
    """Each invocation of a frozen chain with retry has independent retry state."""
    call_count = [0]

    def flaky(x):
      call_count[0] += 1
      if call_count[0] % 2 == 1:
        raise ValueError('odd attempt')
      return x * 2

    frozen = Chain().then(flaky).retry(3, on=(ValueError,)).freeze()
    # Call 1: attempt 1 fails (odd=1), attempt 2 succeeds (even=2).
    r1 = frozen(5)
    self.assertEqual(r1, 10)
    # Call 2: attempt 3 fails (odd=3), attempt 4 succeeds (even=4).
    r2 = frozen(10)
    self.assertEqual(r2, 20)

  def test_frozen_chain_used_as_nested_with_retry(self):
    """Frozen chain used as nested chain in outer chain, frozen chain has its own retry."""
    inner_attempts = []

    def inner_fn(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 2:
        raise ConnectionError('inner fail')
      return x * 2

    frozen_inner = Chain().then(inner_fn).retry(3, on=(ConnectionError,)).freeze()
    result = Chain(5).then(frozen_inner).run()
    self.assertEqual(result, 10)
    self.assertEqual(inner_attempts, [5, 5])


class TestRetryFreezeAsync(IsolatedAsyncioTestCase):

  async def test_frozen_async_retry(self):
    """Frozen chain with async step + retry."""
    attempts = []

    async def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 3

    frozen = Chain().then(flaky).retry(3, on=(ValueError,)).freeze()
    result = await frozen(7)
    self.assertEqual(result, 21)
    self.assertEqual(attempts, [7, 7])


# ---------------------------------------------------------------------------
# Category 7: retry x decorator()
# ---------------------------------------------------------------------------

class TestRetryDecorator(unittest.TestCase):

  def test_decorator_retry_applies(self):
    """Chain with retry used as decorator: retry applies to decorated function."""
    attempts = []

    @Chain().then(lambda x: x + 1).retry(5, on=(ValueError,)).decorator()
    def add_two(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('retry')
      return x + 2

    result = add_two(10)
    self.assertEqual(result, 13)
    self.assertEqual(attempts, [10, 10, 10])

  def test_decorator_exhausts_retries(self):
    """Decorated function exhausts retries, exception propagates."""
    @Chain().retry(3, on=(ValueError,)).decorator()
    def always_fail(x):
      raise ValueError('always')

    with self.assertRaises(ValueError) as ctx:
      always_fail(1)
    self.assertEqual(str(ctx.exception), 'always')

  def test_decorator_with_chain_steps(self):
    """Decorator with chain steps after the decorated function + retry."""
    attempts = []

    @Chain().then(lambda x: x * 2).then(lambda x: x + 100).retry(3, on=(ValueError,)).decorator()
    def compute(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x + 10

    result = compute(5)
    # decorated fn(5)=15, then 15*2=30, then 30+100=130.
    self.assertEqual(result, 130)


class TestRetryDecoratorAsync(IsolatedAsyncioTestCase):

  async def test_async_decorator_retry(self):
    """Async decorated function with retry."""
    attempts = []

    @Chain().then(lambda x: x + 1).retry(5, on=(ValueError,)).decorator()
    async def async_add(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('retry')
      return x + 2

    result = await async_add(10)
    self.assertEqual(result, 13)
    self.assertEqual(attempts, [10, 10, 10])


# ---------------------------------------------------------------------------
# Category 8: retry x nested chains (per-link retry pattern)
# ---------------------------------------------------------------------------

class TestRetryNestedChains(unittest.TestCase):

  def test_inner_chain_retries_independently(self):
    """Inner chain retries independently, outer chain succeeds."""
    inner_attempts = []

    def inner_flaky(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 2:
        raise ConnectionError('inner')
      return x * 2

    inner = Chain().then(inner_flaky).retry(3, on=(ConnectionError,))
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 10)
    self.assertEqual(inner_attempts, [5, 5])

  def test_inner_exhausts_retries_exception_propagates(self):
    """Inner chain exhausts retries -> inner exception propagates to outer chain."""
    inner_attempts = []

    def always_fail(x):
      inner_attempts.append(x)
      raise ConnectionError('always')

    inner = Chain().then(always_fail).retry(3, on=(ConnectionError,))
    with self.assertRaises(ConnectionError):
      Chain(5).then(inner).run()
    self.assertEqual(len(inner_attempts), 3)

  def test_inner_exhausts_retries_outer_except_catches(self):
    """Inner chain exhausts retries, outer except_ catches the exception."""
    def always_fail(x):
      raise ConnectionError('always')

    inner = Chain().then(always_fail).retry(2, on=(ConnectionError,))
    result = (
      Chain(5)
      .then(inner)
      .except_(lambda exc: f'caught:{type(exc).__name__}')
      .run()
    )
    self.assertEqual(result, 'caught:ConnectionError')

  def test_outer_and_inner_independent_retry_loops(self):
    """Outer chain has retry, inner chain has retry -- independent retry loops."""
    inner_attempts = []
    outer_attempts = []

    def inner_fn(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 2:
        raise ConnectionError('inner')
      return x * 2

    def outer_fn(x):
      outer_attempts.append(x)
      if len(outer_attempts) < 2:
        raise ValueError('outer')
      return x + 100

    inner = Chain().then(inner_fn).retry(3, on=(ConnectionError,))
    result = Chain(5).then(inner).then(outer_fn).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 110)

  def test_nested_different_retry_configs(self):
    """Nested chains with different retry configs (different max_attempts, different `on` types)."""
    inner_attempts = []
    outer_attempts = []

    def inner_fn(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 3:
        raise ConnectionError('inner')
      return x * 2

    def outer_fn(x):
      outer_attempts.append(x)
      return x + 1

    inner = Chain().then(inner_fn).retry(5, on=(ConnectionError,))
    result = Chain(5).then(inner).then(outer_fn).retry(2, on=(TypeError,)).run()
    self.assertEqual(result, 11)

  def test_triple_nesting_with_retry(self):
    """Triple nesting: outer -> middle -> inner, each with retry."""
    inner_attempts = []
    middle_attempts = []
    outer_attempts = []

    def inner_fn(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 2:
        raise ConnectionError('inner')
      return x * 2

    def middle_fn(x):
      middle_attempts.append(x)
      if len(middle_attempts) < 2:
        raise TimeoutError('middle')
      return x + 100

    def outer_fn(x):
      outer_attempts.append(x)
      return x + 1000

    inner = Chain().then(inner_fn).retry(3, on=(ConnectionError,))
    middle = Chain().then(inner).then(middle_fn).retry(3, on=(TimeoutError,))
    result = Chain(5).then(middle).then(outer_fn).run()
    self.assertEqual(result, 1110)

  def test_inner_retry_succeeds_outer_continues(self):
    """Inner retry succeeds, outer chain continues normally."""
    inner_attempts = []

    def inner_fn(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 2:
        raise ConnectionError('inner')
      return x * 2

    inner = Chain().then(inner_fn).retry(3, on=(ConnectionError,))
    result = (
      Chain(5)
      .then(inner)
      .then(lambda x: x + 100)
      .then(lambda x: x * 3)
      .run()
    )
    # inner: 5*2=10 (after 1 retry), then 10+100=110, then 110*3=330.
    self.assertEqual(result, 330)

  def test_multiple_nested_chains_each_with_retry(self):
    """Multiple nested chains in same outer chain, each with own retry."""
    attempts_a = []
    attempts_b = []

    def fn_a(x):
      attempts_a.append(x)
      if len(attempts_a) < 2:
        raise ConnectionError('a fail')
      return x + 10

    def fn_b(x):
      attempts_b.append(x)
      if len(attempts_b) < 3:
        raise TimeoutError('b fail')
      return x + 20

    chain_a = Chain().then(fn_a).retry(3, on=(ConnectionError,))
    chain_b = Chain().then(fn_b).retry(5, on=(TimeoutError,))

    result = Chain(5).then(chain_a).then(chain_b).run()
    # chain_a: 5+10=15 (2 attempts). chain_b: 15+20=35 (3 attempts).
    self.assertEqual(result, 35)
    self.assertEqual(len(attempts_a), 2)
    self.assertEqual(len(attempts_b), 3)


class TestRetryNestedChainsAsync(IsolatedAsyncioTestCase):

  async def test_inner_async_chain_retries(self):
    """Inner async chain retries independently."""
    inner_attempts = []

    async def inner_fn(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 2:
        raise ConnectionError('inner')
      return x * 2

    inner = Chain().then(inner_fn).retry(3, on=(ConnectionError,))
    result = await Chain(5).then(inner).run()
    self.assertEqual(result, 10)
    self.assertEqual(inner_attempts, [5, 5])

  async def test_outer_async_retry_with_inner_retry(self):
    """Outer async chain retries with inner async chain also retrying."""
    inner_attempts = []
    outer_attempts = []

    async def inner_fn(x):
      inner_attempts.append(x)
      if len(inner_attempts) < 2:
        raise ConnectionError('inner')
      return x * 2

    async def outer_fn(x):
      outer_attempts.append(x)
      if len(outer_attempts) < 2:
        raise ValueError('outer')
      return x + 100

    inner = Chain().then(inner_fn).retry(3, on=(ConnectionError,))
    result = await Chain(5).then(inner).then(outer_fn).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 110)


# ---------------------------------------------------------------------------
# Category 9: retry x X placeholder
# ---------------------------------------------------------------------------

class TestRetryXPlaceholder(unittest.TestCase):

  def test_x_expression_with_retry(self):
    """X expression + retry: verify X replay works correctly across retries."""
    fn, attempts = _make_flaky(1, ValueError)
    result = Chain(10).then(fn).then(X * 2).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 20)

  def test_x_attr_with_retry(self):
    """X.attr + retry."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    result = Chain('hello').then(flaky).then(X.upper()).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 'HELLO')

  def test_x_item_with_retry(self):
    """X[item] + retry."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    result = Chain([10, 20, 30]).then(flaky).then(X[1]).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 20)

  def test_x_operator_with_retry(self):
    """X + operator + retry."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    result = Chain(10).then(flaky).then(X + 5).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 15)

  def test_x_chained_ops_with_retry(self):
    """X with chained operations + retry."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    result = Chain('hello world').then(flaky).then(X.split()).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, ['hello', 'world'])


class TestRetryXPlaceholderAsync(IsolatedAsyncioTestCase):

  async def test_x_expression_async_retry(self):
    """X expression + async retry."""
    attempts = []

    async def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    result = await Chain(10).then(flaky).then(X * 3).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# Category 10: Edge cases
# ---------------------------------------------------------------------------

class TestRetryEdgeCases(unittest.TestCase):

  def test_empty_chain_with_retry(self):
    """Empty chain with retry returns None."""
    result = Chain().retry(3).run()
    self.assertIsNone(result)

  def test_chain_only_root_value_with_retry(self):
    """Chain(42).retry(3).run() returns 42 -- no failure possible."""
    result = Chain(42).retry(3).run()
    self.assertEqual(result, 42)

  def test_retry_with_null_value(self):
    """Chain().retry(3).run() returns None (Null -> None)."""
    result = Chain().retry(3).run()
    self.assertIsNone(result)

  def test_retry_with_ellipsis_convention(self):
    """Chain(fn, ...).retry(3).run() -> fn called with no args."""
    attempts = []

    def fn():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return 'no_args'

    result = Chain(fn, ...).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 'no_args')
    self.assertEqual(len(attempts), 2)

  def test_different_exception_types_each_attempt(self):
    """Chain raises different exception type on each attempt, all matched by on=()."""
    attempts = []

    def fn():
      attempts.append(1)
      n = len(attempts)
      if n == 1:
        raise ValueError('v')
      if n == 2:
        raise TypeError('t')
      return 'ok'

    result = Chain(fn, ...).retry(5, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 3)

  def test_matching_then_nonmatching_exception(self):
    """Retry matches first exceptions but a different one on a later attempt."""
    attempts = []

    def fn():
      attempts.append(1)
      n = len(attempts)
      if n < 3:
        raise ValueError('retryable')
      raise RuntimeError('not retryable')

    with self.assertRaises(RuntimeError) as ctx:
      Chain(fn, ...).retry(5, on=(ValueError,)).run()
    self.assertEqual(str(ctx.exception), 'not retryable')
    self.assertEqual(len(attempts), 3)

  def test_backoff_callable_raises(self):
    """If the backoff callable itself raises, the error propagates immediately."""
    attempts = []

    def fn():
      attempts.append(1)
      raise ValueError('fail')

    def bad_backoff(attempt):
      raise RuntimeError('backoff broken')

    with self.assertRaises(RuntimeError) as ctx:
      Chain(fn, ...).retry(3, on=(ValueError,), backoff=bad_backoff).run()
    self.assertEqual(str(ctx.exception), 'backoff broken')
    self.assertEqual(len(attempts), 1)

  def test_max_attempts_zero_still_executes_once(self):
    """max_attempts=0 -> treated as 1 due to `self._retry_max_attempts or 1`."""
    attempts = []

    def fn():
      attempts.append(1)
      return 'ok'

    result = Chain(fn, ...).retry(0, on=(ValueError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 1)

  def test_max_attempts_zero_with_failure(self):
    """max_attempts=0 with failure: treated as 1, exception propagates immediately."""
    with self.assertRaises(ValueError):
      Chain(lambda: (_ for _ in ()).throw(ValueError('fail'))).retry(0, on=(ValueError,)).run()

  def test_max_attempts_one_no_retry(self):
    """max_attempts=1 means execute once, no retry."""
    attempts = []

    def fn():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('fail')
      return 'ok'

    with self.assertRaises(ValueError):
      Chain(fn, ...).retry(1, on=(ValueError,)).run()
    self.assertEqual(len(attempts), 1)

  def test_long_chain_20_links_with_retry(self):
    """Very long chain (20+ links) with retry: all links re-execute."""
    attempts = []

    def flaky_first(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    c = Chain(1).then(flaky_first)
    for i in range(20):
      c = c.then(lambda x, i=i: x + 1)
    c = c.retry(3, on=(ValueError,))
    result = c.run()
    self.assertEqual(result, 21)
    self.assertEqual(len(attempts), 2)

  def test_combined_link_types_with_retry(self):
    """Chain with foreach + then + do combined, plus retry."""
    attempts = []

    def flaky_step(x):
      attempts.append(('step', x))
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    side_effects = []

    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .then(lambda lst: sum(lst))
      .then(flaky_step)
      .do(lambda x: side_effects.append(x))
      .then(lambda x: x + 100)
      .retry(3, on=(ValueError,))
      .run()
    )
    # [1,2,3] -> foreach *2 -> [2,4,6] -> sum -> 12 -> flaky(12) -> 12 -> +100 -> 112
    self.assertEqual(result, 112)
    # Side effects only happen on successful pass.
    self.assertEqual(side_effects, [12])

  def test_retry_config_independent_per_chain_instance(self):
    """Two chains with different retry configs operate independently."""
    attempts1 = []
    attempts2 = []

    def fn1():
      attempts1.append(1)
      if len(attempts1) < 2:
        raise ValueError('r')
      return 'chain1'

    def fn2():
      attempts2.append(1)
      if len(attempts2) < 4:
        raise TypeError('r')
      return 'chain2'

    c1 = Chain(fn1, ...).retry(3, on=(ValueError,))
    c2 = Chain(fn2, ...).retry(5, on=(TypeError,))
    self.assertEqual(c1.run(), 'chain1')
    self.assertEqual(c2.run(), 'chain2')
    self.assertEqual(len(attempts1), 2)
    self.assertEqual(len(attempts2), 4)

  def test_retry_with_flat_backoff(self):
    """Retry with flat float backoff delay."""
    import time

    attempts = []

    def fn():
      attempts.append(time.monotonic())
      if len(attempts) < 3:
        raise ValueError('retry')
      return 'ok'

    result = Chain(fn, ...).retry(3, on=(ValueError,), backoff=0.05).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 3)
    # Verify some delay happened between attempts.
    for i in range(1, len(attempts)):
      self.assertGreater(attempts[i] - attempts[i - 1], 0.03)

  def test_retry_with_callable_backoff(self):
    """Retry with callable backoff: backoff(attempt_index)."""
    backoff_calls = []

    def backoff(attempt):
      backoff_calls.append(attempt)
      return 0.01

    attempts = []

    def fn():
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('retry')
      return 'ok'

    result = Chain(fn, ...).retry(3, on=(ValueError,), backoff=backoff).run()
    self.assertEqual(result, 'ok')
    # backoff called with attempt indices 0 and 1 (before retries 1 and 2).
    self.assertEqual(backoff_calls, [0, 1])


class TestRetryEdgeCasesAsync(IsolatedAsyncioTestCase):

  async def test_async_different_exception_types(self):
    """Async chain raises different exception types, all matched."""
    attempts = []

    async def fn():
      attempts.append(1)
      n = len(attempts)
      if n == 1:
        raise ValueError('v')
      if n == 2:
        raise TypeError('t')
      return 'ok'

    result = await Chain(fn, ...).retry(5, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 3)

  async def test_async_backoff_uses_asyncio_sleep(self):
    """Async retry uses asyncio.sleep for backoff, not time.sleep."""
    import time

    attempts = []

    async def fn():
      attempts.append(time.monotonic())
      if len(attempts) < 2:
        raise ValueError('retry')
      return 'ok'

    result = await Chain(fn, ...).retry(3, on=(ValueError,), backoff=0.05).run()
    self.assertEqual(result, 'ok')
    self.assertGreater(attempts[1] - attempts[0], 0.03)

  async def test_async_long_chain_retry(self):
    """Async long chain with retry."""
    attempts = []

    async def flaky_first(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    c = Chain(1).then(flaky_first)
    for i in range(10):
      c = c.then(lambda x, i=i: x + 1)
    c = c.retry(3, on=(ValueError,))
    result = await c.run()
    self.assertEqual(result, 11)


# ---------------------------------------------------------------------------
# Category 11: retry preserves chain identity / fluent API
# ---------------------------------------------------------------------------

class TestRetryFluentAPI(unittest.TestCase):

  def test_retry_returns_self(self):
    """retry() returns self for fluent chaining."""
    c = Chain()
    r = c.retry(3)
    self.assertIs(r, c)

  def test_retry_then_more_methods(self):
    """After retry(), can chain more methods: .retry(3).then(fn).run()."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x

    result = Chain(10).retry(3, on=(ValueError,)).then(flaky).then(lambda x: x + 5).run()
    self.assertEqual(result, 15)

  def test_retry_called_before_other_methods(self):
    """retry() can be called before then()."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 2

    result = Chain(5).retry(3, on=(ValueError,)).then(flaky).run()
    self.assertEqual(result, 10)

  def test_retry_called_after_other_methods(self):
    """retry() can be called after then()."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 2

    result = Chain(5).then(flaky).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 10)

  def test_retry_multiple_times_last_wins(self):
    """Calling retry() multiple times: last one wins."""
    attempts = []

    def flaky():
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('retry')
      return 'ok'

    # First retry(2) would not be enough (only 2 attempts), but retry(5) overrides.
    c = Chain(flaky, ...).retry(2, on=(ValueError,)).retry(5, on=(ValueError,))
    result = c.run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 3)

  def test_retry_single_exception_type_not_tuple(self):
    """retry(on=ValueError) with a single type (not wrapped in tuple)."""
    attempts = []

    def flaky():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return 'ok'

    result = Chain(flaky, ...).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')

  def test_retry_default_on_is_exception(self):
    """retry() with default on=(Exception,) catches any Exception subclass."""
    attempts = []

    def flaky():
      attempts.append(1)
      if len(attempts) < 2:
        raise RuntimeError('retry')
      return 'ok'

    result = Chain(flaky, ...).retry(3).run()
    self.assertEqual(result, 'ok')

  def test_retry_does_not_match_base_exception_by_default(self):
    """retry() default on=(Exception,) does NOT catch BaseException subclasses like KeyboardInterrupt."""
    # We cannot actually raise KeyboardInterrupt easily in test, but we can
    # verify that a custom BaseException subclass is not caught.
    class MyBaseExc(BaseException):
      pass

    def fn():
      raise MyBaseExc('stop')

    with self.assertRaises(MyBaseExc):
      Chain(fn, ...).retry(3).run()

  def test_retry_with_base_exception_in_on(self):
    """retry(on=(BaseException,)) catches BaseException subclasses."""
    attempts = []

    class MyBaseExc(BaseException):
      pass

    def fn():
      attempts.append(1)
      if len(attempts) < 2:
        raise MyBaseExc('stop')
      return 'ok'

    result = Chain(fn, ...).retry(3, on=(MyBaseExc,)).run()
    self.assertEqual(result, 'ok')


# ---------------------------------------------------------------------------
# Category: retry x return_() interaction
# ---------------------------------------------------------------------------

class TestRetryReturn(unittest.TestCase):

  def test_return_is_not_retried(self):
    """Chain.return_() is a _ControlFlowSignal, so it must NOT be retried."""
    attempts = []

    def fn(x):
      attempts.append(x)
      Chain.return_(42)

    result = Chain(10).then(fn).retry(5, on=(Exception,)).run()
    self.assertEqual(result, 42)
    # return_ triggers immediately, no retry.
    self.assertEqual(attempts, [10])


# ---------------------------------------------------------------------------
# Category: retry x if_/else_
# ---------------------------------------------------------------------------

class TestRetryConditional(unittest.TestCase):

  def test_retry_with_if_predicate_fails(self):
    """if_() where predicate raises -> chain retries from scratch."""
    attempts = []

    def flaky_pred(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x > 5

    result = (
      Chain(10)
      .if_(flaky_pred, lambda x: x * 2)
      .retry(3, on=(ValueError,))
      .run()
    )
    self.assertEqual(result, 20)

  def test_retry_with_if_body_fails(self):
    """if_() where the body raises -> chain retries from scratch."""
    attempts = []

    def flaky_body(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 3

    result = (
      Chain(10)
      .if_(lambda x: True, flaky_body)
      .retry(3, on=(ValueError,))
      .run()
    )
    self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# Category: retry x except_/finally_ interaction
# ---------------------------------------------------------------------------

class TestRetryExceptFinally(unittest.TestCase):

  def test_retry_exhausted_then_except_handles(self):
    """When retry is exhausted, except_ handler catches the final exception."""
    attempts = []

    def always_fail():
      attempts.append(1)
      raise ValueError('always')

    result = (
      Chain(always_fail, ...)
      .retry(3, on=(ValueError,))
      .except_(lambda exc: f'caught after {len(attempts)} attempts')
      .run()
    )
    self.assertEqual(result, 'caught after 3 attempts')

  def test_retry_succeeds_finally_called_once(self):
    """When retry succeeds, finally_ is called exactly once."""
    finally_tracker = []
    attempts = []

    def flaky():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return 'ok'

    result = (
      Chain(flaky, ...)
      .retry(3, on=(ValueError,))
      .finally_(lambda rv: finally_tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(len(finally_tracker), 1)

  def test_retry_exhausted_finally_called_once(self):
    """When retry is exhausted, finally_ is still called exactly once.

    Because the chain uses (fn, ...) with Ellipsis, the root_value stays Null
    when all attempts fail (no successful evaluation). The finally_ handler
    is then called with no arguments (callable with Null current_value).
    """
    finally_tracker = []

    def always_fail():
      raise ValueError('always')

    with self.assertRaises(ValueError):
      (
        Chain(always_fail, ...)
        .retry(3, on=(ValueError,))
        .finally_(lambda: finally_tracker.append('called'))
        .run()
      )
    self.assertEqual(len(finally_tracker), 1)

  def test_except_not_called_during_retries(self):
    """except_ handler is NOT called during intermediate retry attempts."""
    except_calls = []
    attempts = []

    def flaky():
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('retry')
      return 'ok'

    result = (
      Chain(flaky, ...)
      .retry(5, on=(ValueError,))
      .except_(lambda exc: except_calls.append(exc) or 'handled')
      .run()
    )
    self.assertEqual(result, 'ok')
    # except_ should NOT have been called -- retry succeeded.
    self.assertEqual(except_calls, [])


class TestRetryExceptFinallyAsync(IsolatedAsyncioTestCase):

  async def test_async_retry_exhausted_except_handles(self):
    """Async: when retry exhausted, except_ catches."""
    attempts = []

    async def always_fail():
      attempts.append(1)
      raise ValueError('always')

    result = await (
      Chain(always_fail, ...)
      .retry(3, on=(ValueError,))
      .except_(lambda exc: f'caught after {len(attempts)}')
      .run()
    )
    self.assertEqual(result, 'caught after 3')

  async def test_async_retry_succeeds_finally_once(self):
    """Async: when retry succeeds, finally_ called once."""
    finally_tracker = []
    attempts = []

    async def flaky():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return 'ok'

    result = await (
      Chain(flaky, ...)
      .retry(3, on=(ValueError,))
      .finally_(lambda rv: finally_tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(len(finally_tracker), 1)


# ---------------------------------------------------------------------------
# Category: retry x run(value)
# ---------------------------------------------------------------------------

class TestRetryRunValue(unittest.TestCase):

  def test_retry_with_run_value(self):
    """Chain with retry, invoked via run(value)."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 2

    c = Chain().then(flaky).retry(3, on=(ValueError,))
    result = c.run(10)
    self.assertEqual(result, 20)
    self.assertEqual(attempts, [10, 10])

  def test_retry_with_call_value(self):
    """Chain with retry, invoked via __call__(value)."""
    attempts = []

    def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 2

    c = Chain().then(flaky).retry(3, on=(ValueError,))
    result = c(10)
    self.assertEqual(result, 20)


class TestRetryRunValueAsync(IsolatedAsyncioTestCase):

  async def test_async_retry_with_run_value(self):
    """Async chain with retry, invoked via run(value)."""
    attempts = []

    async def flaky(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('retry')
      return x * 2

    c = Chain().then(flaky).retry(3, on=(ValueError,))
    result = await c.run(10)
    self.assertEqual(result, 20)


# ---------------------------------------------------------------------------
# Category: Combined operation chains + kwargs + frozen-iterate with retry
# ---------------------------------------------------------------------------

class TestRetryCombinedOps(unittest.TestCase):

  def test_gather_then_do_retry(self):
    """gather() + do() both re-execute on each retry attempt."""
    attempts = []
    side_effects = []

    def fn1(v):
      return v * 2

    def fn2(v):
      return v * 3

    def flaky_side_effect(v):
      side_effects.append(list(v))
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('fail')

    # Chain: 5 -> gather(fn1, fn2) -> [10, 15] -> do(flaky_side_effect) -> [10, 15]
    # do() is a side-effect step (result discarded), so the chain value stays [10, 15].
    # When do raises, retry restarts the whole chain including gather.
    result = (
      Chain(5)
      .gather(fn1, fn2)
      .do(flaky_side_effect)
      .retry(3, on=(ValueError,))
      .run()
    )
    self.assertEqual(result, [10, 15])
    self.assertEqual(len(attempts), 3)
    # side_effects captured the gather result on each attempt
    for se in side_effects:
      self.assertEqual(se, [10, 15])

  def test_filter_then_gather_retry(self):
    """filter() followed by gather() with retry: both re-execute on each attempt."""
    attempts = []

    def is_even(x):
      return x % 2 == 0

    def flaky_sum(v):
      attempts.append(v)
      if len(attempts) < 3:
        raise ValueError('fail')
      return sum(v)

    def flaky_len(v):
      return len(v)

    # Chain: [1,2,3,4,5] -> filter(is_even) -> [2,4] -> gather(flaky_sum, flaky_len)
    # On first two attempts, flaky_sum raises inside gather, so retry restarts.
    result = (
      Chain([1, 2, 3, 4, 5])
      .filter(is_even)
      .gather(flaky_sum, flaky_len)
      .retry(5, on=(ValueError,))
      .run()
    )
    self.assertEqual(result, [6, 2])
    self.assertEqual(len(attempts), 3)
    # Each attempt received the same filtered list
    for att in attempts:
      self.assertEqual(att, [2, 4])

  def test_run_kwargs_preserved_across_retries(self):
    """run() kwargs are preserved across retry attempts.

    run(callable, **kwargs) makes the callable the root value; it is called
    with the supplied kwargs on every retry attempt.
    """
    attempts = []

    def fn(timeout=None, retries=None):
      attempts.append({'timeout': timeout, 'retries': retries})
      if len(attempts) < 3:
        raise ValueError('fail')
      return f'{timeout}-{retries}'

    result = Chain().retry(3, on=(ValueError,)).run(fn, timeout=5, retries=2)
    self.assertEqual(result, '5-2')
    self.assertEqual(len(attempts), 3)
    # Verify kwargs were identical each time
    for att in attempts:
      self.assertEqual(att, {'timeout': 5, 'retries': 2})

  def test_run_positional_and_kwargs_preserved(self):
    """run() with both positional args and kwargs preserved across retries.

    run(callable, pos_arg, key=val) calls callable(pos_arg, key=val) on
    each retry attempt with identical arguments.
    """
    attempts = []

    def fn(x, multiplier=1):
      attempts.append({'x': x, 'multiplier': multiplier})
      if len(attempts) < 2:
        raise ValueError('fail')
      return x * multiplier

    result = Chain().retry(3, on=(ValueError,)).run(fn, 10, multiplier=3)
    self.assertEqual(result, 30)
    self.assertEqual(len(attempts), 2)
    for att in attempts:
      self.assertEqual(att, {'x': 10, 'multiplier': 3})

  def test_frozen_chain_with_retry_inside_iterate(self):
    """Frozen chain with retry used as the transform fn in iterate()."""
    call_count = [0]

    def flaky_double(x):
      call_count[0] += 1
      # Fail on every other call to exercise retry within each iteration
      if call_count[0] % 2 == 1:
        raise ValueError('retry')
      return x * 2

    frozen = Chain().then(flaky_double).retry(3, on=(ValueError,)).freeze()
    # Use frozen chain as a step applied to each element via foreach
    result = Chain([1, 2, 3]).foreach(frozen).run()
    # Each element: 1*2=2, 2*2=4, 3*2=6 (each required 1 retry)
    self.assertEqual(result, [2, 4, 6])
    # 3 elements * 2 calls each (1 fail + 1 success) = 6
    self.assertEqual(call_count[0], 6)

  def test_iterate_over_chain_with_retry(self):
    """iterate() on a chain that has retry: source retries, then yields elements."""
    attempts = []

    def flaky_source():
      attempts.append(1)
      if len(attempts) < 2:
        raise ValueError('retry')
      return [10, 20, 30]

    frozen = Chain(flaky_source, ...).retry(3, on=(ValueError,)).freeze()
    gen = Chain(frozen, ...).iterate()
    result = list(gen)
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(len(attempts), 2)


class TestRetryCombinedOpsAsync(IsolatedAsyncioTestCase):

  async def test_gather_then_do_retry_async(self):
    """Async: gather() + do() both re-execute on each retry attempt."""
    attempts = []
    side_effects = []

    async def fn1(v):
      return v * 2

    async def fn2(v):
      return v * 3

    async def flaky_side_effect(v):
      side_effects.append(list(v))
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('fail')

    result = await (
      Chain(5)
      .gather(fn1, fn2)
      .do(flaky_side_effect)
      .retry(3, on=(ValueError,))
      .run()
    )
    self.assertEqual(result, [10, 15])
    self.assertEqual(len(attempts), 3)
    for se in side_effects:
      self.assertEqual(se, [10, 15])

  async def test_run_kwargs_preserved_across_retries_async(self):
    """Async: run() kwargs are preserved across retry attempts."""
    attempts = []

    async def fn(timeout=None, retries=None):
      attempts.append({'timeout': timeout, 'retries': retries})
      if len(attempts) < 3:
        raise ValueError('fail')
      return f'{timeout}-{retries}'

    result = await Chain().retry(3, on=(ValueError,)).run(fn, timeout=7, retries=4)
    self.assertEqual(result, '7-4')
    self.assertEqual(len(attempts), 3)
    for att in attempts:
      self.assertEqual(att, {'timeout': 7, 'retries': 4})

  async def test_frozen_chain_with_retry_inside_foreach_async(self):
    """Async: Frozen chain with retry used inside foreach."""
    call_count = [0]

    async def flaky_double(x):
      call_count[0] += 1
      if call_count[0] % 2 == 1:
        raise ValueError('retry')
      return x * 2

    frozen = Chain().then(flaky_double).retry(3, on=(ValueError,)).freeze()
    result = await Chain([1, 2, 3]).foreach(frozen).run()
    self.assertEqual(result, [2, 4, 6])
    self.assertEqual(call_count[0], 6)


if __name__ == '__main__':
  unittest.main()
