"""Exhaustive sync/async bridging tests for Chain.retry().

Tests the retry mechanism across all execution paths:
  - Pure sync retry (stays in _run, uses time.sleep)
  - Pure async retry (stays in _run_async, uses asyncio.sleep)
  - Sync-to-async transition during retry (the critical path)
  - Various awaitable patterns during retry
  - Backoff verification (sync vs async sleep)
  - Run arguments preservation across retries
"""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from quent import Chain, Null, QuentException
from helpers import (
  async_fn,
  async_identity,
  sync_fn,
  sync_identity,
  AsyncRange,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

class FailNTimes:
  """Sync callable that raises on the first N calls, then succeeds."""

  def __init__(self, n, exc_type=ValueError, success_value='ok'):
    self.n = n
    self.exc_type = exc_type
    self.success_value = success_value
    self.call_count = 0

  def __call__(self, *args, **kwargs):
    self.call_count += 1
    if self.call_count <= self.n:
      raise self.exc_type(f'fail #{self.call_count}')
    return self.success_value


class AsyncFailNTimes:
  """Async callable that raises on the first N calls, then succeeds."""

  def __init__(self, n, exc_type=ValueError, success_value='ok'):
    self.n = n
    self.exc_type = exc_type
    self.success_value = success_value
    self.call_count = 0

  async def __call__(self, *args, **kwargs):
    self.call_count += 1
    if self.call_count <= self.n:
      raise self.exc_type(f'fail #{self.call_count}')
    return self.success_value


class TrackingCallable:
  """Sync callable that records all calls."""

  def __init__(self, fn=None):
    self.calls = []
    self._fn = fn

  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    if self._fn:
      return self._fn(*args, **kwargs)
    return args[0] if args else None


class AsyncTrackingCallable:
  """Async callable that records all calls."""

  def __init__(self, fn=None):
    self.calls = []
    self._fn = fn

  async def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    if self._fn:
      result = self._fn(*args, **kwargs)
      if asyncio.iscoroutine(result):
        return await result
      return result
    return args[0] if args else None


class SyncThenAsyncCallable:
  """Returns sync on first N calls, then returns a coroutine."""

  def __init__(self, n_sync, sync_value='sync', async_value='async'):
    self.n_sync = n_sync
    self.sync_value = sync_value
    self.async_value = async_value
    self.call_count = 0

  def __call__(self, *args, **kwargs):
    self.call_count += 1
    if self.call_count <= self.n_sync:
      return self.sync_value
    return self._async_result()

  async def _async_result(self):
    return self.async_value


class AsyncThenSyncCallable:
  """Returns a coroutine on first N calls, then returns sync."""

  def __init__(self, n_async, sync_value='sync', async_value='async'):
    self.n_async = n_async
    self.sync_value = sync_value
    self.async_value = async_value
    self.call_count = 0

  def __call__(self, *args, **kwargs):
    self.call_count += 1
    if self.call_count <= self.n_async:
      return self._async_result()
    return self.sync_value

  async def _async_result(self):
    return self.async_value


class FailSyncThenAsync:
  """Fails sync on first call, fails async on second, succeeds on third."""

  def __init__(self, success_value='ok'):
    self.call_count = 0
    self.success_value = success_value

  def __call__(self, *args, **kwargs):
    self.call_count += 1
    if self.call_count == 1:
      raise ValueError('sync fail')
    if self.call_count == 2:
      return self._async_fail()
    return self.success_value

  async def _async_fail(self):
    raise ValueError('async fail')


class FailAsyncThenSync:
  """Fails async on first call, fails sync on second, succeeds on third."""

  def __init__(self, success_value='ok'):
    self.call_count = 0
    self.success_value = success_value

  def __call__(self, *args, **kwargs):
    self.call_count += 1
    if self.call_count == 1:
      return self._async_fail()
    if self.call_count == 2:
      raise ValueError('sync fail')
    return self.success_value

  async def _async_fail(self):
    raise ValueError('async fail')


# ===========================================================================
# CATEGORY 1: PURE SYNC RETRY
# ===========================================================================

class TestPureSyncRetry(unittest.TestCase):
  """All links sync, all failures sync -> retry loop stays in _run()."""

  def test_single_link_succeeds_first_try(self):
    tracker = TrackingCallable(fn=lambda x: x + 1)
    result = Chain(tracker, 10).retry(3, on=ValueError).run()
    self.assertEqual(result, 11)
    self.assertEqual(len(tracker.calls), 1)

  def test_single_link_fails_once_then_succeeds(self):
    fn = FailNTimes(1)
    result = Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn.call_count, 2)

  def test_single_link_fails_twice_then_succeeds(self):
    fn = FailNTimes(2)
    result = Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn.call_count, 3)

  def test_single_link_exhausts_all_attempts(self):
    fn = FailNTimes(3)
    with self.assertRaises(ValueError) as cm:
      Chain(fn).retry(3, on=ValueError).run()
    self.assertIn('fail #3', str(cm.exception))
    self.assertEqual(fn.call_count, 3)

  def test_two_links_failure_in_first(self):
    fn1 = FailNTimes(1)
    tracker2 = TrackingCallable(fn=lambda x: x.upper())
    result = Chain(fn1).then(tracker2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'OK')
    self.assertEqual(fn1.call_count, 2)
    # tracker2 only runs on the successful attempt
    self.assertEqual(len(tracker2.calls), 1)

  def test_two_links_failure_in_second(self):
    tracker1 = TrackingCallable(fn=lambda: 'hello')
    fn2 = FailNTimes(1)
    result = Chain(tracker1, ...).then(fn2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    # tracker1 is called for both attempts (chain restarts from scratch)
    self.assertEqual(len(tracker1.calls), 2)
    self.assertEqual(fn2.call_count, 2)

  def test_three_links_failure_in_middle(self):
    fn1 = TrackingCallable(fn=lambda: 1)
    fn2 = FailNTimes(1, success_value=2)
    fn3 = TrackingCallable(fn=lambda x: x + 10)
    result = Chain(fn1, ...).then(fn2).then(fn3).retry(3, on=ValueError).run()
    self.assertEqual(result, 12)
    self.assertEqual(len(fn1.calls), 2)
    self.assertEqual(fn2.call_count, 2)
    self.assertEqual(len(fn3.calls), 1)

  def test_non_retryable_exception_not_retried(self):
    fn = FailNTimes(1, exc_type=RuntimeError)
    with self.assertRaises(RuntimeError):
      Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(fn.call_count, 1)

  def test_sync_backoff_uses_time_sleep(self):
    fn = FailNTimes(1)
    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(3, on=ValueError, backoff=0.1).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_called_once_with(0.1)

  def test_sync_backoff_callable(self):
    fn = FailNTimes(2)
    delays = []

    def backoff_fn(attempt):
      delay = (attempt + 1) * 0.01
      delays.append((attempt, delay))
      return delay

    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(3, on=ValueError, backoff=backoff_fn).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(delays), 2)
    self.assertEqual(delays[0][0], 0)
    self.assertEqual(delays[1][0], 1)

  def test_no_backoff_means_no_sleep(self):
    fn = FailNTimes(1)
    with patch('time.sleep') as mock_sleep:
      Chain(fn).retry(3, on=ValueError).run()
    mock_sleep.assert_not_called()

  def test_retry_with_except_handler_all_fail(self):
    """Except handler catches the error when all retries are exhausted."""
    fn = FailNTimes(3)
    handler_calls = []
    # except_ catches the ValueError after retries are exhausted;
    # the handler returns a value, so the chain does not re-raise.
    result = Chain(fn).retry(3, on=ValueError).except_(
      lambda rv, exc: handler_calls.append(str(exc)) or 'handled'
    ).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(handler_calls), 1)
    self.assertIn('fail #3', handler_calls[0])

  def test_retry_succeeds_except_not_called(self):
    fn = FailNTimes(1)
    handler_calls = []
    result = Chain(fn).retry(3, on=ValueError).except_(
      lambda rv, exc: handler_calls.append(str(exc))
    ).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(handler_calls, [])

  def test_retry_with_finally_handler(self):
    fn = FailNTimes(1)
    finally_calls = []
    result = Chain(fn).retry(3, on=ValueError).finally_(
      lambda rv: finally_calls.append(rv)
    ).run()
    self.assertEqual(result, 'ok')
    # Finally runs once, after all retries complete
    self.assertEqual(len(finally_calls), 1)

  def test_retry_with_run_value(self):
    call_count = 0

    def processor(x):
      nonlocal call_count
      call_count += 1
      if call_count <= 1:
        raise ValueError('fail')
      return x * 2

    result = Chain().then(processor).retry(3, on=ValueError).run(5)
    self.assertEqual(result, 10)
    self.assertEqual(call_count, 2)

  def test_retry_max_attempts_one(self):
    fn = FailNTimes(1)
    with self.assertRaises(ValueError):
      Chain(fn).retry(1, on=ValueError).run()
    self.assertEqual(fn.call_count, 1)

  def test_retry_on_tuple_of_exceptions(self):
    call_count = 0

    def alternating():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ValueError('v')
      if call_count == 2:
        raise TypeError('t')
      return 'ok'

    result = Chain(alternating).retry(3, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(call_count, 3)


# ===========================================================================
# CATEGORY 2: PURE ASYNC RETRY
# ===========================================================================

class TestPureAsyncRetry(IsolatedAsyncioTestCase):
  """All links async, all failures async -> _run_async handles all retries."""

  async def test_single_async_link_succeeds_first_try(self):
    fn = AsyncTrackingCallable(fn=lambda x: x + 1)
    result = await Chain(fn, 10).retry(3, on=ValueError).run()
    self.assertEqual(result, 11)
    self.assertEqual(len(fn.calls), 1)

  async def test_single_async_link_fails_once_then_succeeds(self):
    fn = AsyncFailNTimes(1)
    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn.call_count, 2)

  async def test_single_async_link_fails_twice_then_succeeds(self):
    fn = AsyncFailNTimes(2)
    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn.call_count, 3)

  async def test_single_async_link_exhausts_attempts(self):
    fn = AsyncFailNTimes(3)
    with self.assertRaises(ValueError) as cm:
      await Chain(fn).retry(3, on=ValueError).run()
    self.assertIn('fail #3', str(cm.exception))
    self.assertEqual(fn.call_count, 3)

  async def test_two_async_links_failure_in_first(self):
    fn1 = AsyncFailNTimes(1, success_value='hello')
    tracker2 = AsyncTrackingCallable(fn=lambda x: x.upper())
    result = await Chain(fn1).then(tracker2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'HELLO')
    self.assertEqual(fn1.call_count, 2)
    self.assertEqual(len(tracker2.calls), 1)

  async def test_two_async_links_failure_in_second(self):
    tracker1 = AsyncTrackingCallable(fn=lambda: 'hello')
    fn2 = AsyncFailNTimes(1)
    result = await Chain(tracker1, ...).then(fn2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(tracker1.calls), 2)
    self.assertEqual(fn2.call_count, 2)

  async def test_three_async_links_failure_at_each_position(self):
    """Failure at each position across three async links."""
    for fail_pos in range(3):
      fns = []
      for i in range(3):
        if i == fail_pos:
          if i == 0:
            fns.append(AsyncFailNTimes(1, success_value=i * 10))
          else:
            fns.append(AsyncFailNTimes(1, success_value=i * 10))
        else:
          if i == 0:
            # First link gets no current_value, so use no-arg version
            fns.append(AsyncTrackingCallable(fn=lambda *a, _i=i: _i * 10))
          else:
            fns.append(AsyncTrackingCallable(fn=lambda *a, _i=i: _i * 10))
      c = Chain(fns[0]).then(fns[1]).then(fns[2]).retry(3, on=ValueError)
      result = await c.run()
      self.assertEqual(result, 20, f'fail_pos={fail_pos}')

  async def test_async_non_retryable_exception(self):
    fn = AsyncFailNTimes(1, exc_type=RuntimeError)
    with self.assertRaises(RuntimeError):
      await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(fn.call_count, 1)

  async def test_async_backoff_uses_asyncio_sleep(self):
    fn = AsyncFailNTimes(1)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(3, on=ValueError, backoff=0.05).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_called_once_with(0.05)

  async def test_async_backoff_callable(self):
    fn = AsyncFailNTimes(2)
    recorded_attempts = []

    def backoff_fn(attempt):
      recorded_attempts.append(attempt)
      return 0.001

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(3, on=ValueError, backoff=backoff_fn).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(recorded_attempts, [0, 1])
    self.assertEqual(mock_sleep.call_count, 2)

  async def test_async_no_backoff_no_sleep(self):
    fn = AsyncFailNTimes(1)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      await Chain(fn).retry(3, on=ValueError).run()
    mock_sleep.assert_not_called()

  async def test_async_retry_with_except_handler_all_fail(self):
    """Except handler catches error after all retries exhausted."""
    fn = AsyncFailNTimes(3)
    handler_calls = []
    result = await (
      Chain(fn)
      .retry(3, on=ValueError)
      .except_(lambda rv, exc: handler_calls.append(str(exc)) or 'handled')
      .run()
    )
    self.assertEqual(result, 'handled')
    self.assertEqual(len(handler_calls), 1)

  async def test_async_retry_with_finally_handler(self):
    fn = AsyncFailNTimes(1)
    finally_calls = []
    result = await Chain(fn).retry(3, on=ValueError).finally_(
      lambda rv: finally_calls.append('ran')
    ).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(finally_calls), 1)

  async def test_async_retry_with_run_value(self):
    call_count = 0

    async def processor(x):
      nonlocal call_count
      call_count += 1
      if call_count <= 1:
        raise ValueError('fail')
      return x * 2

    result = await Chain().then(processor).retry(3, on=ValueError).run(5)
    self.assertEqual(result, 10)
    self.assertEqual(call_count, 2)

  async def test_async_retry_on_tuple_of_exceptions(self):
    call_count = 0

    async def alternating():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ValueError('v')
      if call_count == 2:
        raise TypeError('t')
      return 'ok'

    result = await Chain(alternating).retry(3, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(call_count, 3)


# ===========================================================================
# CATEGORY 3: SYNC-TO-ASYNC TRANSITION DURING RETRY (CRITICAL PATH)
# ===========================================================================

class TestSyncToAsyncTransitionRetry(IsolatedAsyncioTestCase):
  """The core bridging tests: _run() starts sync, encounters awaitable,
  hands off to _run_async() which then handles retries.
  """

  async def test_sync_link_then_async_link_failure_in_async(self):
    """sync link 1 -> async link 2 fails -> _run_async retries from scratch."""
    tracker = TrackingCallable(fn=lambda: 'hello')
    fn2 = AsyncFailNTimes(1, success_value='HELLO')
    result = await Chain(tracker, ...).then(fn2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'HELLO')
    # tracker called for both attempts (chain restarts from scratch)
    self.assertEqual(len(tracker.calls), 2)
    self.assertEqual(fn2.call_count, 2)

  async def test_async_first_link_fails_on_first_await(self):
    """Async first link -> _run() immediately hands off to _run_async."""
    fn = AsyncFailNTimes(1)
    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn.call_count, 2)

  async def test_chain_sync_on_attempt1_async_on_attempt2(self):
    """Callable returns sync on attempt 1 (fails), returns async on attempt 2 (succeeds)."""
    call_count = 0

    def morphing():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ValueError('sync fail')
      # Attempt 2: return a coroutine
      return _async_ok()

    async def _async_ok():
      return 'async_success'

    result = await Chain(morphing).retry(3, on=ValueError).run()
    self.assertEqual(result, 'async_success')
    self.assertEqual(call_count, 2)

  async def test_chain_async_on_attempt1_sync_on_attempt2(self):
    """Callable returns coroutine on attempt 1 (fails async), sync on attempt 2 (succeeds)."""
    call_count = 0

    def morphing():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return _async_fail()
      return 'sync_success'

    async def _async_fail():
      raise ValueError('async fail')

    result = await Chain(morphing).retry(3, on=ValueError).run()
    self.assertEqual(result, 'sync_success')
    self.assertEqual(call_count, 2)

  async def test_failure_in_sync_part_never_reaches_async(self):
    """Sync link 1 fails -> retry stays in _run() (never reaches async link 2)."""
    fn1 = FailNTimes(1, success_value='hello')
    async_tracker = AsyncTrackingCallable(fn=lambda x: x.upper())

    # Since fn1 is sync and fails before we reach async_tracker,
    # the retry in _run() handles it. When fn1 succeeds on attempt 2,
    # the chain then hits the async link and hands off to _run_async.
    result = await Chain(fn1).then(async_tracker).retry(3, on=ValueError).run()
    self.assertEqual(result, 'HELLO')
    self.assertEqual(fn1.call_count, 2)
    self.assertEqual(len(async_tracker.calls), 1)

  async def test_mixed_sync_async_sync_failure_at_position_0(self):
    """sync(fail) -> async -> sync, failure at link 0."""
    fn0 = FailNTimes(1, success_value='a')

    async def fn1(x):
      return x + 'b'

    fn2 = TrackingCallable(fn=lambda x: x + 'c')
    result = await Chain(fn0).then(fn1).then(fn2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'abc')
    self.assertEqual(fn0.call_count, 2)

  async def test_mixed_sync_async_sync_failure_at_position_1(self):
    """sync -> async(fail) -> sync, failure at link 1."""
    fn0 = TrackingCallable(fn=lambda: 'a')
    fn1 = AsyncFailNTimes(1, success_value='ab')
    fn2 = TrackingCallable(fn=lambda x: x + 'c')
    result = await Chain(fn0, ...).then(fn1).then(fn2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'abc')
    self.assertEqual(len(fn0.calls), 2)  # restarted from scratch
    self.assertEqual(fn1.call_count, 2)

  async def test_mixed_sync_async_sync_failure_at_position_2(self):
    """sync -> async -> sync(fail), failure at link 2."""
    fn0 = TrackingCallable(fn=lambda: 'a')

    async def fn1(x):
      return x + 'b'

    fn2 = FailNTimes(1, success_value='abc')
    result = await Chain(fn0, ...).then(fn1).then(fn2).retry(3, on=ValueError).run()
    self.assertEqual(result, 'abc')
    self.assertEqual(len(fn0.calls), 2)

  async def test_sync_to_async_handoff_preserves_retry_attempt(self):
    """When _run() hands off to _run_async, the retry_attempt is passed.

    Flow: _run() starts sync, hits awaitable, calls _run_async(retry_attempt=0).
    _run_async completes attempt 0 (which fails), then retries from scratch for
    attempt 1 (which also fails), then retries for attempt 2 (which succeeds).

    The key insight: _run() hands off at attempt 0 and _run_async continues
    from there. So root_fn is only called once in _run() (attempt 0), and
    on retries _run_async re-evaluates root_fn.
    """
    attempt_tracker = []

    def root_fn():
      attempt_tracker.append('root')
      return 'val'

    call_count = 0

    async def async_fn_fails(x):
      nonlocal call_count
      call_count += 1
      attempt_tracker.append('async')
      if call_count <= 2:
        raise ValueError('fail')
      return x

    result = await Chain(root_fn).then(async_fn_fails).retry(4, on=ValueError).run()
    self.assertEqual(result, 'val')
    # Attempt 0: _run() calls root_fn (sync), then hits async_fn_fails -> _run_async
    #   _run_async awaits it, fails. Retries:
    # Attempt 1: _run_async restarts chain: root_fn (evaluated), async_fn_fails. Fails.
    # Attempt 2: _run_async restarts chain: root_fn, async_fn_fails. Succeeds.
    # root_fn is called once in _run() (attempt 0) + twice in _run_async retries
    self.assertEqual(attempt_tracker.count('root'), 3)
    self.assertEqual(attempt_tracker.count('async'), 3)

  async def test_transition_with_multiple_sync_links_before_async(self):
    """Multiple sync links execute before reaching async link."""
    counters = {'a': 0, 'b': 0}

    def link_a():
      counters['a'] += 1
      return 1

    def link_b(x):
      counters['b'] += 1
      return x + 1

    async_fail = AsyncFailNTimes(1, success_value=99)

    result = await Chain(link_a).then(link_b).then(async_fail).retry(3, on=ValueError).run()
    self.assertEqual(result, 99)
    self.assertEqual(counters['a'], 2)
    self.assertEqual(counters['b'], 2)
    self.assertEqual(async_fail.call_count, 2)

  async def test_transition_preserves_except_handler(self):
    """Except handler fires after all retries exhausted in async path."""
    fn = AsyncFailNTimes(5)
    handler_result = []

    def sync_root():
      return 'start'

    result = await (
      Chain(sync_root)
      .then(fn)
      .retry(3, on=ValueError)
      .except_(lambda rv, exc: handler_result.append(str(exc)) or 'handled')
      .run()
    )
    self.assertEqual(result, 'handled')
    self.assertEqual(len(handler_result), 1)

  async def test_transition_preserves_finally_handler(self):
    """Finally handler fires after retries complete in async path."""
    fn = AsyncFailNTimes(1, success_value='done')
    finally_calls = []

    def sync_root():
      return 'start'

    result = await (
      Chain(sync_root)
      .then(fn)
      .retry(3, on=ValueError)
      .finally_(lambda rv: finally_calls.append(rv))
      .run()
    )
    self.assertEqual(result, 'done')
    self.assertEqual(len(finally_calls), 1)

  async def test_sync_fail_then_async_fail_then_success(self):
    """FailSyncThenAsync: sync error -> async error -> sync success."""
    fn = FailSyncThenAsync(success_value='ok')
    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn.call_count, 3)

  async def test_async_fail_then_sync_fail_then_success(self):
    """FailAsyncThenSync: async error -> sync error -> sync success."""
    fn = FailAsyncThenSync(success_value='ok')
    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn.call_count, 3)


# ===========================================================================
# CATEGORY 4: ASYNC RETRY WITH VARIOUS AWAITABLE PATTERNS
# ===========================================================================

class TestAsyncRetryAwaitablePatterns(IsolatedAsyncioTestCase):
  """Retry with coroutine functions, plain coroutine returns, and nested chains."""

  async def test_coroutine_function(self):
    fn = AsyncFailNTimes(1)
    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')

  async def test_regular_function_returning_coroutine(self):
    """A plain function that returns a coroutine object."""
    call_count = 0

    def returns_coro():
      nonlocal call_count
      call_count += 1

      async def coro():
        if call_count <= 1:
          raise ValueError('coro fail')
        return 'coro_ok'

      return coro()

    result = await Chain(returns_coro).retry(3, on=ValueError).run()
    self.assertEqual(result, 'coro_ok')
    self.assertEqual(call_count, 2)

  async def test_function_returns_awaitable_on_some_calls(self):
    """Function that returns sync failure on first call, awaitable on second."""
    call_count = 0

    def alternating():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ValueError('sync fail')
      return _async_ok()

    async def _async_ok():
      return 'ok'

    result = await Chain(alternating).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(call_count, 2)

  async def test_nested_chain_where_inner_is_async(self):
    inner_fn = AsyncFailNTimes(1, success_value='inner_ok')
    inner = Chain().then(inner_fn)
    result = await Chain(5).then(inner).retry(3, on=ValueError).run()
    self.assertEqual(result, 'inner_ok')

  async def test_chain_with_async_map_and_retry(self):
    """map with async fn that fails on first chain attempt, succeeds on second."""
    attempt_count = 0

    async def transform(x):
      nonlocal attempt_count
      attempt_count += 1
      # Fail on the very first item of the first attempt
      if attempt_count == 1:
        raise ValueError('map fail')
      return x * 2

    result = await Chain([1, 2, 3]).map(transform).retry(3, on=ValueError).run()
    self.assertEqual(result, [2, 4, 6])
    # attempt 1: item 1 -> fail (attempt_count=1)
    # attempt 2: items 1,2,3 all succeed (attempt_count=2,3,4)
    self.assertEqual(attempt_count, 4)

  async def test_chain_with_async_filter_and_retry(self):
    """filter with async fn that fails on first chain attempt, succeeds on second."""
    attempt_count = 0

    async def pred(x):
      nonlocal attempt_count
      attempt_count += 1
      if attempt_count == 1:
        raise ValueError('filter fail')
      return x > 1

    result = await Chain([1, 2, 3]).filter(pred).retry(3, on=ValueError).run()
    self.assertEqual(result, [2, 3])

  async def test_chain_with_async_gather_and_retry(self):
    """gather with async fns where first attempt fails."""
    call_count = 0

    async def fn_a(x):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ValueError('gather fail')
      return x + 1

    async def fn_b(x):
      return x + 2

    result = await Chain(10).gather(fn_a, fn_b).retry(3, on=ValueError).run()
    self.assertEqual(result, [11, 12])


# ===========================================================================
# CATEGORY 5: ASYNC BACKOFF VERIFICATION
# ===========================================================================

class TestAsyncBackoffVerification(IsolatedAsyncioTestCase):
  """Verify correct sleep function is used in each execution path."""

  async def test_async_path_uses_asyncio_sleep_not_time_sleep(self):
    fn = AsyncFailNTimes(1)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_async, \
         patch('time.sleep') as mock_sync:
      result = await Chain(fn).retry(3, on=ValueError, backoff=0.01).run()
    self.assertEqual(result, 'ok')
    mock_async.assert_called_once_with(0.01)
    mock_sync.assert_not_called()

  async def test_sync_path_uses_time_sleep_not_asyncio_sleep(self):
    """When failure is in sync link and retry stays in _run()."""
    fn = FailNTimes(1)
    # Ensure the chain stays fully sync (no async links).
    with patch('time.sleep') as mock_sync:
      result = Chain(fn).retry(3, on=ValueError, backoff=0.01).run()
    self.assertEqual(result, 'ok')
    mock_sync.assert_called_once_with(0.01)

  async def test_backoff_callable_receives_correct_attempt_indices(self):
    fn = AsyncFailNTimes(3)
    received_attempts = []

    def backoff_fn(attempt):
      received_attempts.append(attempt)
      return 0.001

    with patch('asyncio.sleep', new_callable=AsyncMock):
      with self.assertRaises(ValueError):
        await Chain(fn).retry(3, on=ValueError, backoff=backoff_fn).run()
    # max_attempts=3 means attempts 0,1,2.
    # After attempt 0 fails -> backoff(0)
    # After attempt 1 fails -> backoff(1)
    # Attempt 2 fails -> no backoff (last attempt), raises
    self.assertEqual(received_attempts, [0, 1])

  async def test_no_delay_before_first_attempt(self):
    """The first attempt should never have a delay."""
    fn = AsyncFailNTimes(1)
    delays = []

    def backoff_fn(attempt):
      delays.append(attempt)
      return 0.001

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      await Chain(fn).retry(3, on=ValueError, backoff=backoff_fn).run()
    # Backoff only after attempt 0 (the first failure)
    self.assertEqual(delays, [0])
    mock_sleep.assert_called_once()

  async def test_custom_backoff_with_async_chain_timing(self):
    """Custom backoff callable with various return values."""
    fn = AsyncFailNTimes(2)
    delays = []

    def exp_backoff(attempt):
      delay = 0.001 * (2 ** attempt)
      delays.append(delay)
      return delay

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(3, on=ValueError, backoff=exp_backoff).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(delays), 2)
    self.assertAlmostEqual(delays[0], 0.001)
    self.assertAlmostEqual(delays[1], 0.002)

  async def test_flat_backoff_value(self):
    fn = AsyncFailNTimes(2)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(3, on=ValueError, backoff=0.5).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(mock_sleep.call_count, 2)
    for call in mock_sleep.call_args_list:
      self.assertEqual(call.args[0], 0.5)

  async def test_zero_backoff_skips_sleep(self):
    fn = AsyncFailNTimes(1)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(3, on=ValueError, backoff=0.0).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_not_called()

  async def test_backoff_callable_returns_zero_skips_sleep(self):
    fn = AsyncFailNTimes(1)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(3, on=ValueError, backoff=lambda a: 0.0).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_not_called()

  async def test_transition_backoff_goes_async(self):
    """Sync link succeeds, async link fails -> backoff uses asyncio.sleep."""
    fn2 = AsyncFailNTimes(1)

    def sync_root():
      return 'start'

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_async, \
         patch('time.sleep') as mock_sync:
      result = await Chain(sync_root).then(fn2).retry(3, on=ValueError, backoff=0.01).run()
    self.assertEqual(result, 'ok')
    mock_async.assert_called_once_with(0.01)
    mock_sync.assert_not_called()


# ===========================================================================
# CATEGORY 7: RETRY WITH RUN ARGUMENTS IN ASYNC CONTEXT
# ===========================================================================

class TestRetryWithRunArguments(IsolatedAsyncioTestCase):
  """Verify run() arguments are correctly passed on each retry attempt."""

  async def test_run_value_preserved_across_retries_async(self):
    received_values = []

    async def fn(x):
      received_values.append(x)
      if len(received_values) <= 1:
        raise ValueError('fail')
      return x * 2

    result = await Chain().then(fn).retry(3, on=ValueError).run(42)
    self.assertEqual(result, 84)
    self.assertEqual(received_values, [42, 42])

  async def test_run_with_callable_and_args_preserved(self):
    """run(callable, arg1, arg2) - callable is evaluated with args on each retry."""
    received = []

    def root_fn(x, y, z):
      received.append((x, y, z))
      if len(received) <= 1:
        raise ValueError('fail')
      return x + y + z

    result = Chain().retry(3, on=ValueError).run(root_fn, 1, 2, 3)
    self.assertEqual(result, 6)
    self.assertEqual(received, [(1, 2, 3), (1, 2, 3)])

  async def test_run_value_with_sync_root_and_async_then(self):
    root_calls = []

    def sync_root(x):
      root_calls.append(x)
      return x + 10

    fail_fn = AsyncFailNTimes(1, success_value='done')

    result = await Chain().then(sync_root).then(fail_fn).retry(3, on=ValueError).run(5)
    self.assertEqual(result, 'done')
    self.assertEqual(root_calls, [5, 5])

  async def test_run_value_none_preserved(self):
    """None as run value is distinct from Null (no value)."""
    received = []

    async def fn(x):
      received.append(x)
      if len(received) <= 1:
        raise ValueError('fail')
      return 'got_none' if x is None else 'got_something'

    result = await Chain().then(fn).retry(3, on=ValueError).run(None)
    self.assertEqual(result, 'got_none')
    self.assertEqual(received, [None, None])

  async def test_chain_with_root_value_and_retry(self):
    """Chain(root_value).then(fn).retry() - root value is re-evaluated on retry."""
    fn = AsyncFailNTimes(1, success_value='result')
    result = await Chain(100).then(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'result')
    self.assertEqual(fn.call_count, 2)

  async def test_empty_chain_with_run_value_and_retry(self):
    """Chain().retry().run(value) with no links - value passes through."""
    result = Chain().retry(3, on=ValueError).run(42)
    self.assertEqual(result, 42)

  async def test_do_link_preserved_across_retries(self):
    """Side-effect (do) links are re-executed on retry."""
    side_effects = []
    call_count = 0

    async def main_fn():
      nonlocal call_count
      call_count += 1
      if call_count <= 2:
        raise ValueError('fail')
      return 'done'

    result = await (
      Chain(main_fn)
      .do(lambda x: side_effects.append(x))
      .retry(3, on=ValueError)
      .run()
    )
    # On attempts 1 and 2, main_fn raises before do runs.
    # On attempt 3, main_fn succeeds, do runs.
    self.assertEqual(result, 'done')
    self.assertEqual(len(side_effects), 1)

  async def test_run_value_with_async_root(self):
    """Chain with async root receiving run value."""
    received = []

    async def root(x):
      received.append(x)
      if len(received) <= 1:
        raise ValueError('fail')
      return x * 3

    result = await Chain().then(root).retry(3, on=ValueError).run(7)
    self.assertEqual(result, 21)
    self.assertEqual(received, [7, 7])


# ===========================================================================
# EDGE CASES & INTERACTION TESTS
# ===========================================================================

class TestRetryEdgeCases(IsolatedAsyncioTestCase):
  """Additional edge cases for retry + async bridging."""

  async def test_control_flow_signal_never_retried(self):
    """_Return and _Break should NOT be retried."""
    tracker = []

    def fn():
      tracker.append(1)
      Chain.return_('early')

    result = Chain(fn).retry(3, on=Exception).run()
    self.assertEqual(result, 'early')
    self.assertEqual(len(tracker), 1)

  async def test_control_flow_return_in_async_not_retried(self):
    """_Return in async context should NOT be retried."""
    tracker = []

    async def fn():
      tracker.append(1)
      Chain.return_('early_async')

    result = await Chain(fn).retry(3, on=Exception).run()
    self.assertEqual(result, 'early_async')
    self.assertEqual(len(tracker), 1)

  async def test_retry_with_nested_async_chain(self):
    """Outer chain retries, inner chain is async."""
    inner_calls = []

    async def inner_fn(x):
      inner_calls.append(x)
      if len(inner_calls) <= 1:
        raise ValueError('inner fail')
      return x * 2

    inner = Chain().then(inner_fn)
    result = await Chain(5).then(inner).retry(3, on=ValueError).run()
    self.assertEqual(result, 10)
    self.assertEqual(inner_calls, [5, 5])

  async def test_retry_chained_with_if(self):
    """Retry works with if_ conditional links."""
    call_count = 0

    async def check(x):
      nonlocal call_count
      call_count += 1
      if call_count <= 1:
        raise ValueError('fail')
      return x > 0

    result = await (
      Chain(5)
      .if_(check, then=lambda x: x * 10)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 50)

  async def test_retry_preserves_ignore_result_semantics(self):
    """do() links still ignore results after retry."""
    fn = AsyncFailNTimes(1, success_value='side_effect')

    result = await (
      Chain(42)
      .do(fn)
      .retry(3, on=ValueError)
      .run()
    )
    # do() discards fn's return value, so chain value stays 42
    self.assertEqual(result, 42)
    self.assertEqual(fn.call_count, 2)

  async def test_retry_with_ellipsis_calling_convention(self):
    """Ellipsis (...) calling convention preserved across retries."""
    fn = AsyncFailNTimes(1, success_value='no_args')
    result = await Chain(fn, ...).retry(3, on=ValueError).run()
    self.assertEqual(result, 'no_args')

  async def test_retry_zero_backoff_no_delay(self):
    """backoff=None means no delay at all."""
    fn = AsyncFailNTimes(1)
    start = time.monotonic()
    result = await Chain(fn).retry(3, on=ValueError, backoff=None).run()
    elapsed = time.monotonic() - start
    self.assertEqual(result, 'ok')
    self.assertLess(elapsed, 1.0)

  async def test_retry_exception_types_inheritance(self):
    """Retry matches on exception subclasses."""
    class CustomError(ValueError):
      pass

    fn_calls = 0

    async def fn():
      nonlocal fn_calls
      fn_calls += 1
      if fn_calls <= 1:
        raise CustomError('custom')
      return 'ok'

    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn_calls, 2)

  async def test_retry_base_exception_not_caught_by_default(self):
    """BaseException subclasses (not Exception) are not retried by default."""
    class MyBaseExc(BaseException):
      pass

    fn_calls = 0

    async def fn():
      nonlocal fn_calls
      fn_calls += 1
      raise MyBaseExc('base')

    with self.assertRaises(MyBaseExc):
      await Chain(fn).retry(3, on=(Exception,)).run()
    self.assertEqual(fn_calls, 1)

  async def test_retry_with_base_exception_in_on(self):
    """Retry can catch BaseException subclasses when specified."""
    class MyBaseExc(BaseException):
      pass

    fn_calls = 0

    async def fn():
      nonlocal fn_calls
      fn_calls += 1
      if fn_calls <= 1:
        raise MyBaseExc('base')
      return 'ok'

    result = await Chain(fn).retry(3, on=MyBaseExc).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(fn_calls, 2)

  async def test_retry_with_sync_chain_containing_only_plain_values(self):
    """Chain with plain values (no callables) should work with retry."""
    result = Chain(42).then(100).retry(3, on=ValueError).run()
    self.assertEqual(result, 100)

  async def test_retry_multiple_exception_types_async_transition(self):
    """Different exception types across attempts with sync-to-async transition."""
    call_count = 0

    def root():
      return 'start'

    async def fn(x):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise ValueError('v')
      if call_count == 2:
        raise TypeError('t')
      return 'ok'

    result = await (
      Chain(root)
      .then(fn)
      .retry(3, on=(ValueError, TypeError))
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(call_count, 3)

  async def test_retry_with_chain_returning_none(self):
    """Chain that returns None on success after retry."""
    fn_calls = 0

    async def fn():
      nonlocal fn_calls
      fn_calls += 1
      if fn_calls <= 1:
        raise ValueError('fail')
      return None

    result = await Chain(fn).retry(3, on=ValueError).run()
    self.assertIsNone(result)
    self.assertEqual(fn_calls, 2)

  async def test_retry_preserves_current_value_on_success(self):
    """On successful retry, intermediate values are correctly threaded."""
    fn_calls = 0

    async def step1(x):
      nonlocal fn_calls
      fn_calls += 1
      if fn_calls <= 1:
        raise ValueError('fail')
      return x + 10

    async def step2(x):
      return x * 2

    result = await Chain(5).then(step1).then(step2).retry(3, on=ValueError).run()
    # step1(5) -> 15, step2(15) -> 30
    self.assertEqual(result, 30)

  async def test_async_retry_no_root_no_run_value(self):
    """Cover _chain.py line 323: async retry restart with no root/run value."""
    attempts = []

    async def async_flaky():
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('fail')
      return 'ok'

    result = await Chain().then(async_flaky).retry(3, on=(ValueError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 3)


class TestRetryWithContextManagers(IsolatedAsyncioTestCase):
  """Retry with context manager (with_) operations."""

  async def test_retry_with_sync_cm(self):
    """Retry when a with_ link fails."""
    call_count = 0

    class RetryCM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, *args):
        return False

    def body(ctx):
      nonlocal call_count
      call_count += 1
      if call_count <= 1:
        raise ValueError('cm body fail')
      return f'{ctx}_done'

    cm = RetryCM()
    result = Chain(cm).with_(body).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ctx_done')
    self.assertEqual(call_count, 2)

  async def test_retry_with_async_cm(self):
    call_count = 0

    class AsyncRetryCM:
      async def __aenter__(self):
        return 'actx'
      async def __aexit__(self, *args):
        return False

    async def body(ctx):
      nonlocal call_count
      call_count += 1
      if call_count <= 1:
        raise ValueError('acm body fail')
      return f'{ctx}_done'

    cm = AsyncRetryCM()
    result = await Chain(cm).with_(body).retry(3, on=ValueError).run()
    self.assertEqual(result, 'actx_done')
    self.assertEqual(call_count, 2)

  async def test_retry_with_cm_enter_raises(self):
    """CM __enter__ raises, retry re-enters on next attempt."""
    enter_count = 0

    class FlakyEnterCM:
      def __enter__(self):
        nonlocal enter_count
        enter_count += 1
        if enter_count <= 1:
          raise ValueError('enter fail')
        return 'ctx'
      def __exit__(self, *args):
        return False

    result = Chain(FlakyEnterCM()).with_(lambda ctx: ctx + '_done').retry(3, on=ValueError).run()
    self.assertEqual(result, 'ctx_done')
    self.assertEqual(enter_count, 2)


class TestRetryAPIConfiguration(unittest.TestCase):
  """Tests for the retry() API configuration itself."""

  def test_retry_returns_self_for_fluent_chaining(self):
    c = Chain(lambda: 1)
    result = c.retry(3)
    self.assertIs(result, c)

  def test_retry_default_on_is_exception(self):
    c = Chain(lambda: 1)
    c.retry(3)
    self.assertEqual(c._retry_on, (Exception,))

  def test_retry_single_exception_type_wrapped_in_tuple(self):
    c = Chain(lambda: 1)
    c.retry(3, on=ValueError)
    self.assertEqual(c._retry_on, (ValueError,))

  def test_retry_tuple_of_exceptions_preserved(self):
    c = Chain(lambda: 1)
    c.retry(3, on=(ValueError, TypeError))
    self.assertEqual(c._retry_on, (ValueError, TypeError))

  def test_retry_max_attempts_stored(self):
    c = Chain(lambda: 1)
    c.retry(5)
    self.assertEqual(c._retry_max_attempts, 5)

  def test_retry_backoff_none(self):
    c = Chain(lambda: 1)
    c.retry(3, backoff=None)
    self.assertIsNone(c._retry_backoff)

  def test_retry_backoff_float(self):
    c = Chain(lambda: 1)
    c.retry(3, backoff=1.5)
    self.assertEqual(c._retry_backoff, 1.5)

  def test_retry_backoff_callable(self):
    fn = lambda attempt: attempt * 0.1
    c = Chain(lambda: 1)
    c.retry(3, backoff=fn)
    self.assertIs(c._retry_backoff, fn)

  def test_get_retry_delay_none(self):
    c = Chain(lambda: 1)
    c.retry(3, backoff=None)
    self.assertEqual(c._get_retry_delay(0), 0.0)

  def test_get_retry_delay_float(self):
    c = Chain(lambda: 1)
    c.retry(3, backoff=0.5)
    self.assertEqual(c._get_retry_delay(0), 0.5)
    self.assertEqual(c._get_retry_delay(1), 0.5)

  def test_get_retry_delay_callable(self):
    c = Chain(lambda: 1)
    c.retry(3, backoff=lambda a: a * 0.1)
    self.assertAlmostEqual(c._get_retry_delay(0), 0.0)
    self.assertAlmostEqual(c._get_retry_delay(1), 0.1)
    self.assertAlmostEqual(c._get_retry_delay(2), 0.2)


class TestRetryWithNoRetryConfig(unittest.TestCase):
  """Chain without retry() should behave normally (max_attempts=1)."""

  def test_no_retry_config_no_retry(self):
    fn = FailNTimes(1)
    with self.assertRaises(ValueError):
      Chain(fn).run()
    self.assertEqual(fn.call_count, 1)

  def test_no_retry_config_success(self):
    result = Chain(lambda: 42).run()
    self.assertEqual(result, 42)


if __name__ == '__main__':
  unittest.main()
