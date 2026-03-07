"""Exhaustive tests for retry's interaction with except_, finally_, return_,
break_, if_, else_, and do().

Key invariants:
  - except_() fires ONLY after ALL retry attempts are exhausted.
  - finally_() fires ONCE, after all retries (success or failure).
  - _ControlFlowSignal (return_, break_) is NEVER retried.
  - Retry restarts the entire chain from scratch on each attempt.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import AsyncRange


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

class Counter:
  """Thread-unsafe call counter for tracking attempt/handler invocations."""

  def __init__(self):
    self.calls = []

  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    return len(self.calls)

  @property
  def count(self):
    return len(self.calls)


class FailNTimes:
  """Callable that raises on the first N calls, then succeeds."""

  def __init__(self, n, exc_type=ValueError, success_value='ok'):
    self.n = n
    self.exc_type = exc_type
    self.success_value = success_value
    self.attempts = []

  def __call__(self, *args, **kwargs):
    self.attempts.append((args, kwargs))
    if len(self.attempts) <= self.n:
      raise self.exc_type(f'fail #{len(self.attempts)}')
    return self.success_value


class AsyncFailNTimes:
  """Async version of FailNTimes."""

  def __init__(self, n, exc_type=ValueError, success_value='ok'):
    self.n = n
    self.exc_type = exc_type
    self.success_value = success_value
    self.attempts = []

  async def __call__(self, *args, **kwargs):
    self.attempts.append((args, kwargs))
    if len(self.attempts) <= self.n:
      raise self.exc_type(f'fail #{len(self.attempts)}')
    return self.success_value


def always_fail(exc_type=ValueError, msg='always'):
  """Return a callable that always raises the given exception type."""
  attempts = []

  def fn(*args, **kwargs):
    attempts.append((args, kwargs))
    raise exc_type(msg)

  fn.attempts = attempts
  return fn


def async_always_fail(exc_type=ValueError, msg='always'):
  """Return an async callable that always raises."""
  attempts = []

  async def fn(*args, **kwargs):
    attempts.append((args, kwargs))
    raise exc_type(msg)

  fn.attempts = attempts
  return fn


# ===========================================================================
# Category 1: retry x except_()
# ===========================================================================

class TestRetryExceptAllFail(unittest.TestCase):
  """All attempts fail with a matching exception -> except_ fires once."""

  def test_all_attempts_fail_except_fires_once(self):
    fn = always_fail(ValueError)
    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(handler)
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(handler.count, 1)
    # handler received the exception from the LAST attempt
    exc = handler.calls[0][0][0]
    self.assertIsInstance(exc, ValueError)

  def test_except_returns_value_after_exhaustion(self):
    fn = always_fail(ValueError)
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(lambda exc: 'recovered')
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(result, 'recovered')

  def test_except_returns_none(self):
    fn = always_fail(ValueError)
    result = (
      Chain(fn, ...)
      .retry(2, on=ValueError)
      .except_(lambda exc: None)
      .run()
    )
    self.assertEqual(len(fn.attempts), 2)
    self.assertIsNone(result)

  def test_except_reraises(self):
    fn = always_fail(ValueError)

    def reraise(exc):
      raise exc

    with self.assertRaises(ValueError):
      Chain(fn, ...).retry(3, on=ValueError).except_(reraise).run()
    self.assertEqual(len(fn.attempts), 3)

  def test_except_raises_different_exception(self):
    fn = always_fail(ValueError)

    def raise_runtime(exc):
      raise RuntimeError('handler error')

    with self.assertRaises(RuntimeError) as ctx:
      Chain(fn, ...).retry(3, on=ValueError).except_(raise_runtime).run()
    self.assertEqual(str(ctx.exception), 'handler error')
    self.assertEqual(len(fn.attempts), 3)


class TestRetryExceptLastSucceeds(unittest.TestCase):
  """Some attempts fail, last succeeds -> except_ never fires."""

  def test_success_on_last_attempt_no_except(self):
    fn = FailNTimes(2, ValueError, 'ok')
    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(handler.count, 0)

  def test_success_on_second_attempt_no_except(self):
    fn = FailNTimes(1, ValueError, 'second')
    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 'second')
    self.assertEqual(len(fn.attempts), 2)
    self.assertEqual(handler.count, 0)


class TestRetryExceptFirstSucceeds(unittest.TestCase):
  """First attempt succeeds -> except_ never fires."""

  def test_first_attempt_succeeds(self):
    handler = Counter()
    result = (
      Chain(lambda: 'first', ...)
      .retry(3, on=ValueError)
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 'first')
    self.assertEqual(handler.count, 0)


class TestRetryExceptTypeMatching(unittest.TestCase):
  """except_ exception type vs retry on type interactions."""

  def test_except_catches_same_type_as_retry_on(self):
    fn = always_fail(ValueError)
    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(2, on=ValueError)
      .except_(handler, exceptions=ValueError)
      .run()
    )
    self.assertEqual(len(fn.attempts), 2)
    self.assertEqual(handler.count, 1)

  def test_except_catches_broader_type(self):
    """except_ catches Exception while retry is on (ValueError,)."""
    fn = always_fail(ValueError)
    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(handler, exceptions=Exception)
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(handler.count, 1)

  def test_except_catches_different_type_non_retryable_fires_immediately(self):
    """If a non-retryable exception occurs, except_ fires immediately (no retry)."""
    fn = always_fail(RuntimeError)
    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(5, on=ValueError)
      .except_(handler, exceptions=RuntimeError)
      .run()
    )
    # RuntimeError is not in retry on=(ValueError,), so no retry occurs.
    self.assertEqual(len(fn.attempts), 1)
    self.assertEqual(handler.count, 1)

  def test_except_different_type_not_caught(self):
    """except_ type doesn't match the exception -> propagates."""
    fn = always_fail(RuntimeError)
    with self.assertRaises(RuntimeError):
      Chain(fn, ...).retry(5, on=ValueError).except_(lambda e: None, exceptions=ValueError).run()
    # Only 1 attempt: RuntimeError not retryable by on=ValueError
    self.assertEqual(len(fn.attempts), 1)

  def test_except_narrower_type_than_retry_on(self):
    """Retry on (Exception,) but except_ only catches TypeError -> ValueError propagates."""
    fn = always_fail(ValueError)
    with self.assertRaises(ValueError):
      Chain(fn, ...).retry(2, on=Exception).except_(lambda e: None, exceptions=TypeError).run()
    self.assertEqual(len(fn.attempts), 2)


class TestRetryExceptExceptionFromLastAttempt(unittest.TestCase):
  """Verify the exception passed to except_ is from the LAST attempt."""

  def test_last_attempt_exception_message(self):
    attempts = []

    def failing(x=None):
      attempts.append(1)
      raise ValueError(f'attempt {len(attempts)}')

    caught = []
    Chain(failing, ...).retry(3, on=ValueError).except_(lambda e: caught.append(str(e))).run()
    self.assertEqual(len(attempts), 3)
    self.assertEqual(caught, ['attempt 3'])


class TestRetryExceptAsync(IsolatedAsyncioTestCase):
  """Async except_ handlers with retry."""

  async def test_async_except_handler_fires_after_exhaustion(self):
    fn = async_always_fail(ValueError)
    caught = []

    async def handler(exc):
      caught.append(str(exc))
      return 'async_recovered'

    result = await (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(handler)
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(len(caught), 1)
    self.assertEqual(result, 'async_recovered')

  async def test_async_except_handler_not_fired_on_success(self):
    fn = AsyncFailNTimes(1, ValueError, 'async_ok')
    caught = []

    async def handler(exc):
      caught.append(exc)

    result = await (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 'async_ok')
    self.assertEqual(caught, [])

  async def test_sync_chain_becoming_async_retry_except(self):
    """Chain starts sync, transitions to async mid-way, retries, except_ fires."""
    attempts = []

    async def flaky(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('not yet')
      return 'done'

    caught = []
    result = await (
      Chain(10)
      .then(flaky)
      .retry(3, on=ValueError)
      .except_(lambda e: caught.append(e))
      .run()
    )
    self.assertEqual(result, 'done')
    self.assertEqual(len(attempts), 3)
    self.assertEqual(caught, [])


# ===========================================================================
# Category 2: retry x finally_()
# ===========================================================================

class TestRetryFinallySuccess(unittest.TestCase):
  """Retry succeeds -> finally_ fires once."""

  def test_finally_fires_once_on_success(self):
    finally_calls = []
    result = (
      Chain(lambda: 'success', ...)
      .retry(3, on=ValueError)
      .finally_(lambda rv: finally_calls.append(rv))
      .run()
    )
    self.assertEqual(result, 'success')
    self.assertEqual(len(finally_calls), 1)

  def test_finally_fires_once_after_retry_success(self):
    fn = FailNTimes(2, ValueError, 'recovered')
    finally_calls = []
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .finally_(lambda rv: finally_calls.append(rv))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(len(finally_calls), 1)


class TestRetryFinallyExhaustion(unittest.TestCase):
  """Retry exhaustion -> finally_ fires once."""

  def test_finally_fires_once_on_exhaustion(self):
    fn = always_fail(ValueError)
    finally_calls = []
    # When all retries fail, root_value stays Null, so finally_ handler
    # is called with no args (callable with no current_value).
    with self.assertRaises(ValueError):
      Chain(fn, ...).retry(3, on=ValueError).finally_(lambda: finally_calls.append('called')).run()
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(len(finally_calls), 1)


class TestRetryFinallyHandlerRaises(unittest.TestCase):
  """finally_ handler that raises -> propagates."""

  def test_finally_handler_raises_propagates(self):
    fn = always_fail(ValueError)

    def bad_finally():
      raise RuntimeError('finally error')

    # except_ handles the ValueError, but finally_ raises RuntimeError.
    with self.assertRaises(RuntimeError) as ctx:
      Chain(fn, ...).retry(2, on=ValueError).except_(lambda e: 'handled').finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'finally error')

  def test_finally_handler_raises_on_success(self):
    def bad_finally(rv):
      raise RuntimeError('finally exploded')

    with self.assertRaises(RuntimeError) as ctx:
      Chain(lambda: 'ok', ...).retry(3, on=ValueError).finally_(bad_finally).run()
    self.assertEqual(str(ctx.exception), 'finally exploded')


class TestRetryFinallyAsync(IsolatedAsyncioTestCase):
  """Async finally_ handler with retry."""

  async def test_async_finally_fires_once_on_success(self):
    fn = AsyncFailNTimes(1, ValueError, 'async_ok')
    finally_calls = []

    async def afin(rv):
      finally_calls.append(rv)

    result = await (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .finally_(afin)
      .run()
    )
    self.assertEqual(result, 'async_ok')
    self.assertEqual(len(finally_calls), 1)

  async def test_async_finally_fires_once_on_exhaustion(self):
    fn = async_always_fail(ValueError)
    finally_calls = []

    # When all retries fail, root_value is Null -> finally_ called with no args
    async def afin():
      finally_calls.append('called')

    with self.assertRaises(ValueError):
      await (
        Chain(fn, ...)
        .retry(2, on=ValueError)
        .finally_(afin)
        .run()
      )
    self.assertEqual(len(fn.attempts), 2)
    self.assertEqual(len(finally_calls), 1)


class TestRetryExceptAndFinally(unittest.TestCase):
  """Retry with both except_ AND finally_ -> verify order and single-fire."""

  def test_exhaustion_except_then_finally(self):
    fn = always_fail(ValueError)
    order = []

    def exc_handler(e):
      order.append('except')
      return 'handled'

    # After except_ handles the error, finally_ is called.
    # When root fn always fails, root_value is Null -> finally_ receives no args.
    def fin_handler():
      order.append('finally')

    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(exc_handler)
      .finally_(fin_handler)
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(result, 'handled')
    # except_ fires first, then finally_
    self.assertEqual(order, ['except', 'finally'])

  def test_success_no_except_then_finally(self):
    fn = FailNTimes(1, ValueError, 'ok')
    order = []

    def exc_handler(e):
      order.append('except')

    def fin_handler(rv):
      order.append('finally')

    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(exc_handler)
      .finally_(fin_handler)
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(order, ['finally'])

  def test_except_returns_value_finally_still_fires(self):
    fn = always_fail(ValueError)
    order = []

    # root_value is Null (all retries fail) -> finally_ called with no args
    result = (
      Chain(fn, ...)
      .retry(2, on=ValueError)
      .except_(lambda e: (order.append('except'), 'val')[1])
      .finally_(lambda: order.append('finally'))
      .run()
    )
    self.assertEqual(result, 'val')
    self.assertEqual(order, ['except', 'finally'])


class TestRetryExceptAndFinallyAsync(IsolatedAsyncioTestCase):
  """Async handlers for all three: retry + except_ + finally_."""

  async def test_async_all_three_exhaustion(self):
    fn = async_always_fail(ValueError)
    order = []

    async def exc_handler(e):
      order.append('except')
      return 'async_handled'

    # root_value Null on exhaustion -> no args
    async def fin_handler():
      order.append('finally')

    result = await (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(exc_handler)
      .finally_(fin_handler)
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(result, 'async_handled')
    self.assertEqual(order, ['except', 'finally'])

  async def test_async_all_three_success(self):
    fn = AsyncFailNTimes(1, ValueError, 'async_ok')
    order = []

    async def exc_handler(e):
      order.append('except')

    async def fin_handler(rv):
      order.append('finally')

    result = await (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(exc_handler)
      .finally_(fin_handler)
      .run()
    )
    self.assertEqual(result, 'async_ok')
    self.assertEqual(order, ['finally'])


# ===========================================================================
# Category 3: retry x return_()
# ===========================================================================

class TestRetryReturn(unittest.TestCase):
  """_Return is a _ControlFlowSignal -> NEVER retried."""

  def test_return_not_retried(self):
    attempts = []

    def step(x):
      attempts.append(x)
      return Chain.return_(42)

    result = Chain(10).then(step).retry(5, on=Exception).run()
    self.assertEqual(result, 42)
    # return_ fired on first attempt, no retries
    self.assertEqual(len(attempts), 1)

  def test_return_in_first_link_with_retry(self):
    attempts = []

    def root():
      attempts.append(1)
      return Chain.return_('early')

    result = Chain(root, ...).retry(3, on=Exception).run()
    self.assertEqual(result, 'early')
    self.assertEqual(len(attempts), 1)

  def test_return_in_middle_link_with_retry(self):
    attempts = []

    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .then(lambda x: (attempts.append(x), Chain.return_(x * 10))[1])
      .then(lambda x: x + 999)  # should never run
      .retry(5, on=Exception)
      .run()
    )
    self.assertEqual(result, 60)
    self.assertEqual(len(attempts), 1)

  def test_return_with_value_plus_retry(self):
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(lambda: 'val', ...))
      .retry(3, on=Exception)
      .run()
    )
    self.assertEqual(result, 'val')

  def test_return_not_caught_by_retry_even_if_on_exception(self):
    """retry(on=Exception) must NOT retry _ControlFlowSignal."""
    attempts = []

    def step(x):
      attempts.append(1)
      return Chain.return_('done')

    result = Chain(1).then(step).retry(10, on=Exception).run()
    self.assertEqual(result, 'done')
    self.assertEqual(len(attempts), 1)


class TestRetryReturnNested(unittest.TestCase):
  """Nested chain with return_() + outer retry."""

  def test_nested_return_propagates_no_outer_retry(self):
    outer_attempts = []

    inner = Chain().then(lambda x: Chain.return_(99))

    def outer_step(x):
      outer_attempts.append(x)
      return inner.run(x)

    # The inner chain's return_ should propagate up and NOT be retried
    # because _Return escapes from inner.run() via _run -> _handle_return_exc
    # and the outer chain sees the returned value, not an exception
    result = Chain(5).then(outer_step).retry(3, on=Exception).run()
    self.assertEqual(result, 99)
    self.assertEqual(len(outer_attempts), 1)


class TestRetryReturnAsync(IsolatedAsyncioTestCase):
  """Async variants of return_ + retry."""

  async def test_async_return_not_retried(self):
    attempts = []

    async def step(x):
      attempts.append(x)
      return Chain.return_('async_early')

    result = await Chain(10).then(step).retry(5, on=Exception).run()
    self.assertEqual(result, 'async_early')
    self.assertEqual(len(attempts), 1)


# ===========================================================================
# Category 4: retry x break_()
# ===========================================================================

class TestRetryBreak(unittest.TestCase):
  """_Break is a _ControlFlowSignal -> NEVER retried."""

  def test_break_in_map_not_retried(self):
    attempts = []

    def body(x):
      attempts.append(x)
      if x == 2:
        return Chain.break_()
      return x * 10

    result = (
      Chain([1, 2, 3, 4])
      .map(body)
      .retry(5, on=Exception)
      .run()
    )
    self.assertEqual(result, [10])
    # break_ fires, map completes with partial results, no retry
    self.assertEqual(attempts, [1, 2])

  def test_break_exits_map_chain_continues_no_retry(self):
    attempts = []

    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: (attempts.append(x), Chain.break_(99) if x == 3 else x)[1])
      .then(lambda x: f'result={x}')
      .retry(3, on=Exception)
      .run()
    )
    self.assertEqual(result, 'result=99')
    self.assertEqual(attempts, [1, 2, 3])

  def test_break_outside_map_with_retry_raises_quent_exception(self):
    """break_ outside map raises QuentException, which is NOT retried
    because _ControlFlowSignal is caught before the retry except clause."""
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: Chain.break_()).retry(5, on=Exception).run()
    self.assertIn('break_() cannot be used outside of a map/foreach iteration', str(ctx.exception))


class TestRetryBreakAsync(IsolatedAsyncioTestCase):
  """Async break_ + retry."""

  async def test_async_break_in_map_not_retried(self):
    attempts = []

    async def body(x):
      attempts.append(x)
      if x == 1:
        return Chain.break_()
      return x

    result = await (
      Chain(AsyncRange(5))
      .map(body)
      .retry(5, on=Exception)
      .run()
    )
    self.assertEqual(result, [0])
    self.assertEqual(attempts, [0, 1])


# ===========================================================================
# Category 5: retry x if_() / else_()
# ===========================================================================

class TestRetryIf(unittest.TestCase):
  """if_/else_ with retry."""

  def test_if_true_fn_raises_retried(self):
    """if_ branch fn raises -> retried."""
    attempts = []

    def pred(x):
      return True

    def fn(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('not yet')
      return 'ok'

    result = (
      Chain(10)
      .if_(pred, fn)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 3)

  def test_if_false_else_fn_raises_retried(self):
    """else_ branch fn raises -> retried."""
    attempts = []

    def fn(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('not yet')
      return 'else_ok'

    result = (
      Chain(10)
      .if_(lambda x: False, lambda x: 'should_not')
      .else_(fn)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'else_ok')
    self.assertEqual(len(attempts), 2)

  def test_predicate_raises_retried(self):
    """Predicate itself raises -> retried (if matching exception)."""
    pred_calls = []

    def pred(x):
      pred_calls.append(x)
      if len(pred_calls) < 3:
        raise ValueError('pred fail')
      return True

    result = (
      Chain(10)
      .if_(pred, lambda x: 'branch_hit')
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'branch_hit')
    self.assertEqual(len(pred_calls), 3)

  def test_predicate_changes_between_retries(self):
    """Predicate result changes across attempts -> different branches taken."""
    pred_calls = []

    def pred(x):
      pred_calls.append(x)
      # Falsy on first two calls, truthy on third
      return len(pred_calls) >= 3

    fn_calls = []
    else_calls = []

    def if_fn(x):
      fn_calls.append(x)
      return 'if_branch'

    def else_fn(x):
      else_calls.append(x)
      # Fail to trigger retry
      raise ValueError('else fail')

    result = (
      Chain(10)
      .if_(pred, if_fn)
      .else_(else_fn)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'if_branch')
    self.assertEqual(len(pred_calls), 3)
    self.assertEqual(len(fn_calls), 1)  # Only on the third attempt
    self.assertEqual(len(else_calls), 2)  # First two attempts

  def test_if_with_retry_first_success(self):
    """if_ succeeds on first attempt, no retry."""
    pred_calls = []
    result = (
      Chain(5)
      .if_(lambda x: (pred_calls.append(1), True)[1], lambda x: x * 2)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(len(pred_calls), 1)


class TestRetryIfAsync(IsolatedAsyncioTestCase):
  """Async predicate + retry."""

  async def test_async_predicate_raises_retried(self):
    pred_calls = []

    async def pred(x):
      pred_calls.append(x)
      if len(pred_calls) < 2:
        raise ValueError('async pred fail')
      return True

    result = await (
      Chain(10)
      .if_(pred, lambda x: 'hit')
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'hit')
    self.assertEqual(len(pred_calls), 2)

  async def test_async_if_fn_raises_retried(self):
    attempts = []

    async def fn(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('async fn fail')
      return 'async_ok'

    result = await (
      Chain(10)
      .if_(lambda x: True, fn)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'async_ok')
    self.assertEqual(len(attempts), 2)


# ===========================================================================
# Category 6: retry x except_ x finally_ combined (triple interaction)
# ===========================================================================

class TestRetryTripleInteraction(unittest.TestCase):
  """All three set: retry + except_ + finally_."""

  def test_all_three_exhaustion_order(self):
    fn = always_fail(ValueError)
    order = []
    # root_value stays Null when root always fails -> finally_ gets no args
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(lambda e: (order.append('except'), 'recovered')[1])
      .finally_(lambda: order.append('finally'))
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(result, 'recovered')
    self.assertEqual(order, ['except', 'finally'])

  def test_all_three_success(self):
    fn = FailNTimes(1, ValueError, 'ok')
    order = []
    result = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(lambda e: order.append('except'))
      .finally_(lambda rv: order.append('finally'))
      .run()
    )
    self.assertEqual(result, 'ok')
    self.assertEqual(order, ['finally'])

  def test_except_returns_value_finally_fires(self):
    fn = always_fail(ValueError)
    order = []
    # root_value Null -> finally_ no args
    result = (
      Chain(fn, ...)
      .retry(2, on=ValueError)
      .except_(lambda e: (order.append('except'), 42)[1])
      .finally_(lambda: order.append('finally'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(order, ['except', 'finally'])


class TestRetryTripleInteractionAsync(IsolatedAsyncioTestCase):
  """Async handlers for all three."""

  async def test_async_triple_exhaustion(self):
    fn = async_always_fail(ValueError)
    order = []

    async def exc_handler(e):
      order.append('except')
      return 'async_recovered'

    # root_value Null on exhaustion -> no args
    async def fin_handler():
      order.append('finally')

    result = await (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(exc_handler)
      .finally_(fin_handler)
      .run()
    )
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(result, 'async_recovered')
    self.assertEqual(order, ['except', 'finally'])

  async def test_async_triple_success(self):
    fn = AsyncFailNTimes(1, ValueError, 'async_ok')
    order = []

    async def exc_handler(e):
      order.append('except')

    async def fin_handler(rv):
      order.append('finally')

    result = await (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(exc_handler)
      .finally_(fin_handler)
      .run()
    )
    self.assertEqual(result, 'async_ok')
    self.assertEqual(order, ['finally'])


# ===========================================================================
# Category 7: retry with do() (side effects)
# ===========================================================================

class TestRetryDo(unittest.TestCase):
  """do() side effects happen on EACH retry attempt."""

  def test_do_side_effect_runs_each_attempt(self):
    side_effects = []
    fn = FailNTimes(2, ValueError, 'ok')
    result = (
      Chain(fn, ...)
      .do(lambda x: side_effects.append(f'do:{x}'))
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'ok')
    # do() ran on each successful attempt -- only the 3rd attempt gets past fn
    # On attempts 1 and 2, fn raises before do() runs
    # On attempt 3, fn returns 'ok', then do() runs with 'ok'
    self.assertEqual(side_effects, ['do:ok'])

  def test_do_before_failing_step_runs_each_attempt(self):
    """do() placed before the failing step runs on every retry."""
    side_effects = []
    attempts = []

    def failing(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('fail')
      return 'done'

    result = (
      Chain(5)
      .do(lambda x: side_effects.append(x))
      .then(failing)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'done')
    self.assertEqual(len(side_effects), 3)
    self.assertEqual(len(attempts), 3)

  def test_do_result_discarded_between_retries(self):
    """do() result is discarded; current_value unchanged across retry."""
    fn = FailNTimes(1, ValueError, 42)
    result = (
      Chain(fn, ...)
      .do(lambda x: 'ignored_side_effect')
      .then(lambda x: x * 2)
      .retry(2, on=ValueError)
      .run()
    )
    self.assertEqual(result, 84)

  def test_do_that_raises_triggers_retry(self):
    """do() that raises -> triggers retry."""
    attempts = []

    def bad_side_effect(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('do fail')

    result = (
      Chain(lambda: 10, ...)
      .do(bad_side_effect)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(len(attempts), 3)


class TestRetryDoAsync(IsolatedAsyncioTestCase):
  """Async do() + retry."""

  async def test_async_do_side_effect_runs_each_attempt(self):
    side_effects = []

    async def side_effect(x):
      side_effects.append(x)

    fn = AsyncFailNTimes(1, ValueError, 'async_ok')
    # On attempt 1: fn raises, side_effect never reached
    # On attempt 2: fn succeeds with 'async_ok', side_effect runs
    result = await (
      Chain(fn, ...)
      .do(side_effect)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'async_ok')
    self.assertEqual(side_effects, ['async_ok'])


# ===========================================================================
# Category 8: retry attempt count accuracy
# ===========================================================================

class TestRetryAttemptCount(unittest.TestCase):
  """Verify exact attempt counts in various scenarios."""

  def test_max_attempts_1_no_retry(self):
    fn = always_fail(ValueError)
    with self.assertRaises(ValueError):
      Chain(fn, ...).retry(1, on=ValueError).run()
    self.assertEqual(len(fn.attempts), 1)

  def test_max_attempts_equals_failures_plus_one(self):
    fn = FailNTimes(4, ValueError, 'ok')
    result = Chain(fn, ...).retry(5, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.attempts), 5)

  def test_max_attempts_exceeded(self):
    fn = FailNTimes(5, ValueError, 'ok')
    with self.assertRaises(ValueError):
      Chain(fn, ...).retry(5, on=ValueError).run()
    self.assertEqual(len(fn.attempts), 5)

  def test_retry_with_run_value(self):
    """retry works when chain is invoked with run(value)."""
    attempts = []

    def step(x):
      attempts.append(x)
      if len(attempts) < 3:
        raise ValueError('fail')
      return x * 2

    c = Chain().then(step).retry(3, on=ValueError)
    result = c.run(5)
    self.assertEqual(result, 10)
    self.assertEqual(len(attempts), 3)
    self.assertEqual(attempts, [5, 5, 5])


class TestRetryAttemptCountAsync(IsolatedAsyncioTestCase):
  """Async attempt count verification."""

  async def test_async_exact_attempt_count(self):
    fn = AsyncFailNTimes(2, ValueError, 'ok')
    result = await Chain(fn, ...).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.attempts), 3)


# ===========================================================================
# Category 9: retry backoff interaction
# ===========================================================================

class TestRetryBackoffWithHandlers(unittest.TestCase):
  """Backoff interacts correctly with handlers."""

  def test_backoff_zero_with_except_and_finally(self):
    fn = always_fail(ValueError)
    order = []
    # root_value is Null when all retries fail -> finally_ gets no args
    result = (
      Chain(fn, ...)
      .retry(2, on=ValueError, backoff=0.0)
      .except_(lambda e: (order.append('except'), 'handled')[1])
      .finally_(lambda: order.append('finally'))
      .run()
    )
    self.assertEqual(len(fn.attempts), 2)
    self.assertEqual(result, 'handled')
    self.assertEqual(order, ['except', 'finally'])

  def test_callable_backoff_invoked_per_retry(self):
    fn = always_fail(ValueError)
    backoff_calls = []

    def my_backoff(attempt):
      backoff_calls.append(attempt)
      return 0.0  # no actual delay

    with self.assertRaises(ValueError):
      Chain(fn, ...).retry(4, on=ValueError, backoff=my_backoff).run()
    self.assertEqual(len(fn.attempts), 4)
    # backoff called for attempts 0, 1, 2 (before retries 1, 2, 3)
    self.assertEqual(backoff_calls, [0, 1, 2])


# ===========================================================================
# Category 10: retry with multi-step chains
# ===========================================================================

class TestRetryMultiStep(unittest.TestCase):
  """Retry restarts the entire chain from scratch."""

  def test_all_steps_rerun_on_retry(self):
    step1_calls = []
    step2_calls = []
    step3_calls = []

    def step1(x):
      step1_calls.append(x)
      return x + 1

    def step2(x):
      step2_calls.append(x)
      return x * 2

    def step3(x):
      step3_calls.append(x)
      if len(step3_calls) < 3:
        raise ValueError('fail')
      return x + 100

    result = (
      Chain(5)
      .then(step1)
      .then(step2)
      .then(step3)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 112)  # (5+1)*2 + 100
    self.assertEqual(len(step1_calls), 3)
    self.assertEqual(len(step2_calls), 3)
    self.assertEqual(len(step3_calls), 3)

  def test_chain_state_reset_between_retries(self):
    """Verify each retry starts fresh (not from the failed link)."""
    calls = []

    def step(x):
      calls.append(('step', x))
      if len(calls) < 3:
        raise ValueError('fail')
      return 'final'

    result = (
      Chain(lambda: 'start', ...)
      .then(step)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'final')
    # The root lambda is re-evaluated each time
    self.assertEqual(len(calls), 3)
    for tag, val in calls:
      self.assertEqual(val, 'start')


class TestRetryMultiStepAsync(IsolatedAsyncioTestCase):
  """Async multi-step chain with retry."""

  async def test_async_all_steps_rerun(self):
    step1_calls = []
    step2_calls = []

    async def step1(x):
      step1_calls.append(x)
      return x + 1

    async def step2(x):
      step2_calls.append(x)
      if len(step2_calls) < 2:
        raise ValueError('fail')
      return x * 10

    result = await (
      Chain(5)
      .then(step1)
      .then(step2)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 60)
    self.assertEqual(len(step1_calls), 2)
    self.assertEqual(len(step2_calls), 2)


# ===========================================================================
# Category 11: retry with frozen chains
# ===========================================================================

class TestRetryFrozen(unittest.TestCase):
  """Frozen chains with retry."""

  def test_frozen_chain_retry(self):
    fn = FailNTimes(2, ValueError, 'ok')
    frozen = Chain(fn, ...).retry(3, on=ValueError).freeze()
    result = frozen.run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.attempts), 3)

  def test_frozen_chain_retry_with_except(self):
    fn = always_fail(ValueError)
    handler = Counter()
    frozen = (
      Chain(fn, ...)
      .retry(3, on=ValueError)
      .except_(handler)
      .freeze()
    )
    result = frozen.run()
    self.assertEqual(len(fn.attempts), 3)
    self.assertEqual(handler.count, 1)


# ===========================================================================
# Category 12: retry with non-retryable exceptions
# ===========================================================================

class TestRetryNonRetryable(unittest.TestCase):
  """Exceptions not in retry on=() are not retried."""

  def test_non_matching_exception_not_retried(self):
    fn = always_fail(RuntimeError)
    with self.assertRaises(RuntimeError):
      Chain(fn, ...).retry(5, on=ValueError).run()
    self.assertEqual(len(fn.attempts), 1)

  def test_non_matching_with_except_handler(self):
    fn = always_fail(RuntimeError)
    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(5, on=ValueError)
      .except_(handler, exceptions=RuntimeError)
      .run()
    )
    self.assertEqual(len(fn.attempts), 1)
    self.assertEqual(handler.count, 1)

  def test_keyboard_interrupt_not_retried(self):
    """BaseException subclasses (non-Exception) not retried."""
    attempts = []

    def fn():
      attempts.append(1)
      raise KeyboardInterrupt()

    with self.assertRaises(KeyboardInterrupt):
      Chain(fn, ...).retry(5, on=Exception).run()
    self.assertEqual(len(attempts), 1)


# ===========================================================================
# Category 13: retry on= tuple with multiple exception types
# ===========================================================================

class TestRetryMultipleExceptionTypes(unittest.TestCase):
  """retry(on=(ValueError, TypeError))."""

  def test_multiple_exception_types_retried(self):
    attempts = []

    def fn():
      attempts.append(1)
      if len(attempts) == 1:
        raise ValueError('v')
      if len(attempts) == 2:
        raise TypeError('t')
      return 'ok'

    result = Chain(fn, ...).retry(3, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(attempts), 3)

  def test_except_catches_after_mixed_retries(self):
    attempts = []

    def fn():
      attempts.append(1)
      if len(attempts) % 2 == 1:
        raise ValueError('v')
      raise TypeError('t')

    handler = Counter()
    result = (
      Chain(fn, ...)
      .retry(4, on=(ValueError, TypeError))
      .except_(handler)
      .run()
    )
    self.assertEqual(len(attempts), 4)
    self.assertEqual(handler.count, 1)
    # Last exception is from attempt 4 (even -> TypeError)
    exc = handler.calls[0][0][0]
    self.assertIsInstance(exc, TypeError)


# ===========================================================================
# Category 14: retry with single exception type (not tuple)
# ===========================================================================

class TestRetrySingleExceptionType(unittest.TestCase):
  """retry(on=ValueError) -- single type, not tuple."""

  def test_single_type_retried(self):
    fn = FailNTimes(2, ValueError, 'ok')
    result = Chain(fn, ...).retry(3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.attempts), 3)


# ===========================================================================
# Category 15: retry with map (exception in map body)
# ===========================================================================

class TestRetryMap(unittest.TestCase):
  """Retry with map: exception in map body -> entire chain retried."""

  def test_map_body_raises_retried(self):
    attempts = []

    def body(x):
      attempts.append(x)
      if len(attempts) <= 3:
        raise ValueError(f'fail {x}')
      return x * 10

    # [1, 2] has 2 elements; the body fails on first 3 calls total.
    # Attempt 1: body(1) fails -> retry
    # Attempt 2: body(1) fails -> retry
    # Attempt 3: body(1) fails -> retry
    # Attempt 4: body(1) ok -> body(2) ok -> success
    result = (
      Chain([1, 2])
      .map(body)
      .retry(4, on=ValueError)
      .run()
    )
    self.assertEqual(result, [10, 20])
    self.assertEqual(len(attempts), 5)  # 3 failures + 2 successes

  def test_map_with_except_after_retry_exhaustion(self):
    fn = always_fail(ValueError)
    handler = Counter()
    result = (
      Chain([1, 2, 3])
      .map(fn)
      .retry(2, on=ValueError)
      .except_(handler)
      .run()
    )
    # fn always fails, called once per retry attempt (fails on first element each time)
    self.assertEqual(len(fn.attempts), 2)
    self.assertEqual(handler.count, 1)


# ===========================================================================
# Category 16: edge cases
# ===========================================================================

class TestRetryEdgeCases(unittest.TestCase):
  """Miscellaneous edge cases."""

  def test_retry_with_no_links(self):
    """Empty chain with retry."""
    result = Chain().retry(3, on=ValueError).run()
    self.assertIsNone(result)

  def test_retry_default_on_exception(self):
    """Default on=(Exception,) catches ValueError."""
    fn = FailNTimes(1, ValueError, 'ok')
    result = Chain(fn, ...).retry(3).run()
    self.assertEqual(result, 'ok')

  def test_retry_preserves_chain_value_on_success(self):
    """After retry succeeds, the value from successful attempt is returned."""
    attempts = []

    def fn():
      attempts.append(1)
      if len(attempts) < 3:
        raise ValueError('fail')
      return len(attempts) * 100

    result = Chain(fn, ...).retry(5, on=ValueError).run()
    self.assertEqual(result, 300)

  def test_retry_with_root_value_and_steps(self):
    """Chain(initial_value).then(step).retry(...)"""
    attempts = []

    def step(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('fail')
      return x + 100

    result = Chain(5).then(step).retry(3, on=ValueError).run()
    self.assertEqual(result, 105)
    self.assertEqual(attempts, [5, 5])

  def test_except_handler_with_retry_single_attempt(self):
    """retry(1) means 1 attempt total, so except_ fires on first failure."""
    fn = always_fail(ValueError)
    handler = Counter()
    result = Chain(fn, ...).retry(1, on=ValueError).except_(handler).run()
    self.assertEqual(len(fn.attempts), 1)
    self.assertEqual(handler.count, 1)


# ===========================================================================
# Category 17: retry with return_ + except_/finally_ combined
# ===========================================================================

class TestRetryReturnWithHandlers(unittest.TestCase):
  """return_ + retry + except_/finally_ interactions."""

  def test_return_with_except_handler_except_not_called(self):
    """return_ fires, except_ should NOT be called."""
    handler = Counter()
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .retry(3, on=Exception)
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(handler.count, 0)

  def test_return_with_finally_handler_finally_fires(self):
    """return_ fires, finally_ should still fire once."""
    finally_calls = []
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .retry(3, on=Exception)
      .finally_(lambda rv: finally_calls.append(rv))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(len(finally_calls), 1)


# ===========================================================================
# Category 18: retry with break_ + except_/finally_ combined
# ===========================================================================

class TestRetryBreakWithHandlers(unittest.TestCase):
  """break_ + retry + except_/finally_ interactions."""

  def test_break_in_map_with_except_not_called(self):
    handler = Counter()
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(99) if x == 2 else x)
      .retry(3, on=Exception)
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 99)
    self.assertEqual(handler.count, 0)

  def test_break_in_map_with_finally_fires(self):
    finally_calls = []
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(99) if x == 2 else x)
      .retry(3, on=Exception)
      .finally_(lambda rv: finally_calls.append(rv))
      .run()
    )
    self.assertEqual(result, 99)
    self.assertEqual(len(finally_calls), 1)


# ===========================================================================
# Category 19: retry with filter
# ===========================================================================

class TestRetryFilter(unittest.TestCase):
  """Retry with filter: exception in filter fn -> entire chain retried."""

  def test_filter_fn_raises_retried(self):
    attempts = []

    def pred(x):
      attempts.append(x)
      if len(attempts) <= 2:
        raise ValueError('filter fail')
      return x > 1

    result = (
      Chain([1, 2, 3])
      .filter(pred)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, [2, 3])
    # First 2 attempts fail on first element, third attempt processes all
    self.assertEqual(len(attempts), 5)  # 2 failures + 3 successes


# ===========================================================================
# Category 20: retry with gather
# ===========================================================================

class TestRetryGatherAsync(IsolatedAsyncioTestCase):
  """Retry with gather: exception in one fn -> entire chain retried."""

  async def test_gather_fn_raises_retried(self):
    attempts = []

    async def fn1(x):
      return x + 1

    async def fn2(x):
      attempts.append(x)
      if len(attempts) < 2:
        raise ValueError('gather fail')
      return x + 2

    result = await (
      Chain(10)
      .gather(fn1, fn2)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, [11, 12])
    self.assertEqual(len(attempts), 2)


# ===========================================================================
# Category 21: retry with nested (sub-)chains
# ===========================================================================

class TestRetryNestedChain(unittest.TestCase):
  """Retry behavior with nested Chain objects."""

  def test_inner_chain_exception_retried_by_outer(self):
    attempts = []
    inner = Chain().then(lambda x: (attempts.append(x), x + 1)[1])

    def maybe_fail(x):
      if len(attempts) < 3:
        raise ValueError('inner fail')
      return x

    result = (
      Chain(5)
      .then(inner)
      .then(maybe_fail)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(len(attempts), 3)

  def test_outer_retry_does_not_affect_inner_retry(self):
    """Inner chain has its own retry config, independent of outer."""
    inner_attempts = []
    outer_attempts = []

    inner_fn = FailNTimes(1, ValueError, 'inner_ok')
    inner = Chain(inner_fn, ...).retry(2, on=ValueError)

    def outer_step(x):
      outer_attempts.append(x)
      return x

    result = (
      Chain(inner)
      .then(outer_step)
      .retry(3, on=ValueError)
      .run()
    )
    self.assertEqual(result, 'inner_ok')
    self.assertEqual(len(inner_fn.attempts), 2)
    self.assertEqual(len(outer_attempts), 1)


if __name__ == '__main__':
  unittest.main()
