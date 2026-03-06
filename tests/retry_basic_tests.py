"""Exhaustive combinatorial tests for Chain.retry(): basic behavior,
exception filtering, backoff, state reset, and max_attempts edge cases.
"""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from quent import Chain, Null, QuentException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fail_then_succeed(fail_count, exc_type=ValueError, success_value='ok'):
  """Return a callable that raises exc_type `fail_count` times, then returns success_value."""
  counter = []

  def fn(*args, **kwargs):
    counter.append(1)
    if len(counter) <= fail_count:
      raise exc_type(f'attempt {len(counter)}')
    return success_value

  fn.counter = counter
  return fn


def make_async_fail_then_succeed(fail_count, exc_type=ValueError, success_value='ok'):
  """Async version of make_fail_then_succeed."""
  counter = []

  async def fn(*args, **kwargs):
    counter.append(1)
    if len(counter) <= fail_count:
      raise exc_type(f'attempt {len(counter)}')
    return success_value

  fn.counter = counter
  return fn


def make_always_fail(exc_type=ValueError, msg='always'):
  """Return a callable that always raises."""
  counter = []

  def fn(*args, **kwargs):
    counter.append(1)
    raise exc_type(msg)

  fn.counter = counter
  return fn


def make_async_always_fail(exc_type=ValueError, msg='always'):
  """Async version of make_always_fail."""
  counter = []

  async def fn(*args, **kwargs):
    counter.append(1)
    raise exc_type(msg)

  fn.counter = counter
  return fn


def make_counter_fn(return_value='counted'):
  """Return a callable that counts invocations and returns a fixed value."""
  counter = []

  def fn(*args, **kwargs):
    counter.append((args, kwargs))
    return return_value

  fn.counter = counter
  return fn


def make_async_counter_fn(return_value='counted'):
  """Async version of make_counter_fn."""
  counter = []

  async def fn(*args, **kwargs):
    counter.append((args, kwargs))
    return return_value

  fn.counter = counter
  return fn


# ---------------------------------------------------------------------------
# Category 1: Basic retry behavior (matrix)
# ---------------------------------------------------------------------------

class TestRetryBasicSucceedOnFirst(unittest.TestCase):
  """max_attempts varies but the callable always succeeds on the first attempt."""

  def test_max1_succeed_first_single_link(self):
    fn = make_fail_then_succeed(0)
    result = Chain(fn).retry(max_attempts=1).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 1)

  def test_max2_succeed_first_single_link(self):
    fn = make_fail_then_succeed(0)
    result = Chain(fn).retry(max_attempts=2).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 1)

  def test_max3_succeed_first_single_link(self):
    fn = make_fail_then_succeed(0)
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 1)

  def test_max5_succeed_first_single_link(self):
    fn = make_fail_then_succeed(0)
    result = Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 1)

  def test_max10_succeed_first_single_link(self):
    fn = make_fail_then_succeed(0)
    result = Chain(fn).retry(max_attempts=10).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 1)


class TestRetryBasicSucceedOnSecond(unittest.TestCase):
  """Callable fails once then succeeds — needs max_attempts >= 2."""

  def test_max1_fail_on_first_propagates(self):
    fn = make_fail_then_succeed(1)
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=1).run()
    self.assertEqual(len(fn.counter), 1)

  def test_max2_succeed_on_second(self):
    fn = make_fail_then_succeed(1)
    result = Chain(fn).retry(max_attempts=2).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_max3_succeed_on_second(self):
    fn = make_fail_then_succeed(1)
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_max5_succeed_on_second(self):
    fn = make_fail_then_succeed(1)
    result = Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_max10_succeed_on_second(self):
    fn = make_fail_then_succeed(1)
    result = Chain(fn).retry(max_attempts=10).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)


class TestRetryBasicSucceedOnLast(unittest.TestCase):
  """Callable fails (max_attempts - 1) times then succeeds on the final attempt."""

  def test_max2_succeed_on_last(self):
    fn = make_fail_then_succeed(1)
    result = Chain(fn).retry(max_attempts=2).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_max3_succeed_on_last(self):
    fn = make_fail_then_succeed(2)
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 3)

  def test_max5_succeed_on_last(self):
    fn = make_fail_then_succeed(4)
    result = Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 5)

  def test_max10_succeed_on_last(self):
    fn = make_fail_then_succeed(9)
    result = Chain(fn).retry(max_attempts=10).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 10)


class TestRetryBasicFailAll(unittest.TestCase):
  """Callable always fails — verify all attempts are exhausted."""

  def test_max1_fail_all(self):
    fn = make_always_fail()
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=1).run()
    self.assertEqual(len(fn.counter), 1)

  def test_max2_fail_all(self):
    fn = make_always_fail()
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=2).run()
    self.assertEqual(len(fn.counter), 2)

  def test_max3_fail_all(self):
    fn = make_always_fail()
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(len(fn.counter), 3)

  def test_max5_fail_all(self):
    fn = make_always_fail()
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(len(fn.counter), 5)

  def test_max10_fail_all(self):
    fn = make_always_fail()
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=10).run()
    self.assertEqual(len(fn.counter), 10)


class TestRetrySucceedOnNth(unittest.TestCase):
  """Succeed on various N values within max_attempts window."""

  def test_succeed_on_3rd_of_5(self):
    fn = make_fail_then_succeed(2, success_value='third')
    result = Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(result, 'third')
    self.assertEqual(len(fn.counter), 3)

  def test_succeed_on_4th_of_5(self):
    fn = make_fail_then_succeed(3, success_value='fourth')
    result = Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(result, 'fourth')
    self.assertEqual(len(fn.counter), 4)

  def test_succeed_on_7th_of_10(self):
    fn = make_fail_then_succeed(6, success_value='seventh')
    result = Chain(fn).retry(max_attempts=10).run()
    self.assertEqual(result, 'seventh')
    self.assertEqual(len(fn.counter), 7)

  def test_succeed_on_2nd_of_3(self):
    fn = make_fail_then_succeed(1, success_value='second')
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'second')
    self.assertEqual(len(fn.counter), 2)


class TestRetryChainShapes(unittest.TestCase):
  """Retry with various chain shapes: single link, multi-link, root value + links."""

  def test_single_link_chain(self):
    fn = make_fail_then_succeed(1)
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_two_link_chain_failure_in_second(self):
    """First link always succeeds, second fails then succeeds."""
    first_counter = []

    def first(x=None):
      first_counter.append(1)
      return 10

    call_count = []

    def second(x):
      call_count.append(1)
      if len(call_count) <= 1:
        raise ValueError('fail')
      return x + 5

    result = Chain(first).then(second).retry(max_attempts=3).run()
    self.assertEqual(result, 15)
    # First link re-executes on each attempt
    self.assertEqual(len(first_counter), 2)
    self.assertEqual(len(call_count), 2)

  def test_three_link_chain_failure_in_third(self):
    counters = [[], [], []]

    def link1(x=None):
      counters[0].append(1)
      return 1

    def link2(x):
      counters[1].append(1)
      return x + 1

    def link3(x):
      counters[2].append(1)
      if len(counters[2]) <= 2:
        raise ValueError('fail')
      return x + 1

    result = Chain(link1).then(link2).then(link3).retry(max_attempts=5).run()
    self.assertEqual(result, 3)
    # All three links re-execute on each attempt
    self.assertEqual(len(counters[0]), 3)
    self.assertEqual(len(counters[1]), 3)
    self.assertEqual(len(counters[2]), 3)

  def test_root_value_plus_links(self):
    counter = []

    def step(x):
      counter.append(1)
      if len(counter) <= 1:
        raise ValueError('fail')
      return x * 2

    result = Chain(10).then(step).retry(max_attempts=3).run()
    self.assertEqual(result, 20)
    self.assertEqual(len(counter), 2)

  def test_root_value_multiple_links(self):
    c1, c2 = [], []

    def step1(x):
      c1.append(1)
      return x + 1

    def step2(x):
      c2.append(1)
      if len(c2) <= 2:
        raise ValueError('fail')
      return x + 1

    result = Chain(0).then(step1).then(step2).retry(max_attempts=5).run()
    self.assertEqual(result, 2)
    self.assertEqual(len(c1), 3)
    self.assertEqual(len(c2), 3)


class TestRetryWithRunArguments(unittest.TestCase):
  """Retry with .run(), .run(value), .run(v, arg1, arg2)."""

  def test_run_no_args(self):
    fn = make_fail_then_succeed(1, success_value=42)
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 42)
    self.assertEqual(len(fn.counter), 2)

  def test_run_with_value(self):
    counter = []

    def fn(x):
      counter.append(x)
      if len(counter) <= 1:
        raise ValueError('fail')
      return x * 3

    result = Chain().then(fn).retry(max_attempts=3).run(7)
    self.assertEqual(result, 21)
    # Value 7 is preserved across retries
    self.assertEqual(counter, [7, 7])

  def test_run_with_callable_root_and_extra_args(self):
    """run(callable, arg1, arg2) — root is a callable invoked with extra args."""
    counter = []

    def root(a, b):
      counter.append((a, b))
      if len(counter) <= 1:
        raise ValueError('fail')
      return a + b

    result = Chain().retry(max_attempts=3).run(root, 2, 3)
    self.assertEqual(result, 5)
    # All args preserved across retries
    self.assertEqual(counter, [(2, 3), (2, 3)])

  def test_run_value_preserved_across_many_retries(self):
    counter = []

    def fn(x):
      counter.append(x)
      if len(counter) <= 4:
        raise ValueError('fail')
      return x

    result = Chain().then(fn).retry(max_attempts=10).run(99)
    self.assertEqual(result, 99)
    self.assertEqual(len(counter), 5)
    self.assertTrue(all(v == 99 for v in counter))


class TestRetryAsyncBasic(unittest.IsolatedAsyncioTestCase):
  """Async versions of basic retry behavior."""

  async def test_async_succeed_on_first(self):
    fn = make_async_fail_then_succeed(0)
    result = await Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 1)

  async def test_async_succeed_on_second(self):
    fn = make_async_fail_then_succeed(1)
    result = await Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  async def test_async_succeed_on_last(self):
    fn = make_async_fail_then_succeed(2)
    result = await Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 3)

  async def test_async_fail_all(self):
    fn = make_async_always_fail()
    with self.assertRaises(ValueError):
      await Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(len(fn.counter), 3)

  async def test_async_succeed_on_4th_of_5(self):
    fn = make_async_fail_then_succeed(3, success_value='fourth')
    result = await Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(result, 'fourth')
    self.assertEqual(len(fn.counter), 4)

  async def test_async_run_with_value(self):
    counter = []

    async def fn(x):
      counter.append(x)
      if len(counter) <= 1:
        raise ValueError('fail')
      return x * 2

    result = await Chain().then(fn).retry(max_attempts=3).run(5)
    self.assertEqual(result, 10)
    self.assertEqual(counter, [5, 5])

  async def test_async_multi_link_retry(self):
    c1, c2 = [], []

    async def step1(x=None):
      c1.append(1)
      return 10

    async def step2(x):
      c2.append(1)
      if len(c2) <= 1:
        raise ValueError('fail')
      return x + 5

    result = await Chain(step1).then(step2).retry(max_attempts=3).run()
    self.assertEqual(result, 15)
    self.assertEqual(len(c1), 2)
    self.assertEqual(len(c2), 2)

  async def test_async_sync_transition_on_retry(self):
    """Chain starts sync, transitions to async mid-chain, retries from scratch."""
    c_sync, c_async = [], []

    def sync_step(x=None):
      c_sync.append(1)
      return 10

    async def async_step(x):
      c_async.append(1)
      if len(c_async) <= 1:
        raise ValueError('fail')
      return x + 1

    result = await Chain(sync_step).then(async_step).retry(max_attempts=3).run()
    self.assertEqual(result, 11)
    self.assertEqual(len(c_sync), 2)
    self.assertEqual(len(c_async), 2)


# ---------------------------------------------------------------------------
# Category 2: Exception filtering (matrix)
# ---------------------------------------------------------------------------

class TestRetryExceptionFilterExactMatch(unittest.TestCase):
  """on=(ExactType,) and ExactType is raised."""

  def test_exact_match_valueerror(self):
    fn = make_fail_then_succeed(1, exc_type=ValueError)
    result = Chain(fn).retry(max_attempts=3, on=(ValueError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_exact_match_typeerror(self):
    fn = make_fail_then_succeed(1, exc_type=TypeError)
    result = Chain(fn).retry(max_attempts=3, on=(TypeError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_exact_match_runtime_error(self):
    fn = make_fail_then_succeed(2, exc_type=RuntimeError)
    result = Chain(fn).retry(max_attempts=5, on=(RuntimeError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 3)


class TestRetryExceptionFilterSubclass(unittest.TestCase):
  """on=(BaseType,) and a subclass is raised."""

  def test_exception_catches_valueerror(self):
    fn = make_fail_then_succeed(1, exc_type=ValueError)
    result = Chain(fn).retry(max_attempts=3, on=(Exception,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_exception_catches_typeerror(self):
    fn = make_fail_then_succeed(1, exc_type=TypeError)
    result = Chain(fn).retry(max_attempts=3, on=(Exception,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_lookuperror_catches_keyerror(self):
    fn = make_fail_then_succeed(1, exc_type=KeyError)
    result = Chain(fn).retry(max_attempts=3, on=(LookupError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_lookuperror_catches_indexerror(self):
    fn = make_fail_then_succeed(1, exc_type=IndexError)
    result = Chain(fn).retry(max_attempts=3, on=(LookupError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)


class TestRetryExceptionFilterNoMatch(unittest.TestCase):
  """on=(TypeA,) but TypeB raised — no retry, immediate propagation."""

  def test_no_match_valueerror_vs_typeerror(self):
    fn = make_fail_then_succeed(1, exc_type=TypeError)
    with self.assertRaises(TypeError):
      Chain(fn).retry(max_attempts=5, on=(ValueError,)).run()
    # Only one attempt — no retry on non-matching exception
    self.assertEqual(len(fn.counter), 1)

  def test_no_match_runtime_vs_keyerror(self):
    fn = make_fail_then_succeed(1, exc_type=KeyError)
    with self.assertRaises(KeyError):
      Chain(fn).retry(max_attempts=5, on=(RuntimeError,)).run()
    self.assertEqual(len(fn.counter), 1)

  def test_no_match_ioerror_vs_valueerror(self):
    fn = make_fail_then_succeed(1, exc_type=ValueError)
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=5, on=(IOError,)).run()
    self.assertEqual(len(fn.counter), 1)


class TestRetryExceptionFilterMultipleTypes(unittest.TestCase):
  """on=(TypeA, TypeB) — retry on either."""

  def test_retry_on_first_type(self):
    fn = make_fail_then_succeed(1, exc_type=ValueError)
    result = Chain(fn).retry(max_attempts=3, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_retry_on_second_type(self):
    fn = make_fail_then_succeed(1, exc_type=TypeError)
    result = Chain(fn).retry(max_attempts=3, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_no_match_with_multiple_types(self):
    fn = make_fail_then_succeed(1, exc_type=RuntimeError)
    with self.assertRaises(RuntimeError):
      Chain(fn).retry(max_attempts=5, on=(ValueError, TypeError)).run()
    self.assertEqual(len(fn.counter), 1)

  def test_three_exception_types(self):
    fn = make_fail_then_succeed(1, exc_type=KeyError)
    result = Chain(fn).retry(max_attempts=3, on=(ValueError, TypeError, KeyError)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)


class TestRetryExceptionFilterBaseException(unittest.TestCase):
  """on=(KeyboardInterrupt,) — retry on BaseException subclass."""

  def test_keyboard_interrupt_retried(self):
    fn = make_fail_then_succeed(1, exc_type=KeyboardInterrupt)
    result = Chain(fn).retry(max_attempts=3, on=(KeyboardInterrupt,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_system_exit_retried(self):
    fn = make_fail_then_succeed(1, exc_type=SystemExit)
    result = Chain(fn).retry(max_attempts=3, on=(SystemExit,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_base_exception_default_does_not_catch_keyboard_interrupt(self):
    """Default on=(Exception,) does NOT catch KeyboardInterrupt (a BaseException)."""
    fn = make_fail_then_succeed(1, exc_type=KeyboardInterrupt)
    with self.assertRaises(KeyboardInterrupt):
      Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(len(fn.counter), 1)


class TestRetryExceptionFilterDifferentPerAttempt(unittest.TestCase):
  """Different exception types on different attempts."""

  def test_different_exceptions_both_matched(self):
    counter = []

    def fn(x=None):
      counter.append(1)
      if len(counter) == 1:
        raise ValueError('first')
      if len(counter) == 2:
        raise TypeError('second')
      return 'success'

    result = Chain(fn).retry(max_attempts=5, on=(ValueError, TypeError)).run()
    self.assertEqual(result, 'success')
    self.assertEqual(len(counter), 3)

  def test_different_exceptions_second_not_matched(self):
    counter = []

    def fn(x=None):
      counter.append(1)
      if len(counter) == 1:
        raise ValueError('first')
      if len(counter) == 2:
        raise RuntimeError('second — not matched')
      return 'success'

    with self.assertRaises(RuntimeError):
      Chain(fn).retry(max_attempts=5, on=(ValueError,)).run()
    self.assertEqual(len(counter), 2)


class TestRetryExceptionFilterSingleType(unittest.TestCase):
  """on=ValueError (single type, not tuple) — auto-wrapped to tuple."""

  def test_single_type_auto_wrapped(self):
    fn = make_fail_then_succeed(1, exc_type=ValueError)
    result = Chain(fn).retry(max_attempts=3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_single_type_no_match(self):
    fn = make_fail_then_succeed(1, exc_type=TypeError)
    with self.assertRaises(TypeError):
      Chain(fn).retry(max_attempts=5, on=ValueError).run()
    self.assertEqual(len(fn.counter), 1)

  def test_single_type_subclass_match(self):
    fn = make_fail_then_succeed(1, exc_type=KeyError)
    result = Chain(fn).retry(max_attempts=3, on=LookupError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)


class TestRetryExceptionFilterDefaultOn(unittest.TestCase):
  """Default on=(Exception,) behavior."""

  def test_default_catches_valueerror(self):
    fn = make_fail_then_succeed(1, exc_type=ValueError)
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_default_catches_runtime_error(self):
    fn = make_fail_then_succeed(1, exc_type=RuntimeError)
    result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  def test_default_does_not_catch_base_exception(self):
    fn = make_fail_then_succeed(1, exc_type=KeyboardInterrupt)
    with self.assertRaises(KeyboardInterrupt):
      Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(len(fn.counter), 1)


class TestRetryControlFlowSignalNeverRetried(unittest.TestCase):
  """_ControlFlowSignal (_Return, _Break) is NEVER retried regardless of `on`."""

  def test_return_not_retried(self):
    counter = []

    def fn(x=None):
      counter.append(1)
      Chain.return_(42)

    result = Chain(fn).retry(max_attempts=5, on=(Exception,)).run()
    self.assertEqual(result, 42)
    self.assertEqual(len(counter), 1)

  def test_break_in_foreach_not_retried(self):
    counter = []

    def fn(x):
      counter.append(x)
      if x == 2:
        Chain.break_()
      return x

    result = Chain([1, 2, 3, 4]).foreach(fn).retry(max_attempts=5).run()
    self.assertEqual(result, [1])
    # foreach was called for items 1 and 2, break at 2
    self.assertEqual(counter, [1, 2])

  def test_return_not_retried_even_with_broad_on(self):
    counter = []

    def fn(x=None):
      counter.append(1)
      Chain.return_('early')

    result = Chain(fn).retry(max_attempts=10, on=(BaseException,)).run()
    self.assertEqual(result, 'early')
    self.assertEqual(len(counter), 1)


class TestRetryExceptionFilterAsync(unittest.IsolatedAsyncioTestCase):
  """Async versions of exception filtering tests."""

  async def test_async_exact_match(self):
    fn = make_async_fail_then_succeed(1, exc_type=ValueError)
    result = await Chain(fn).retry(max_attempts=3, on=(ValueError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  async def test_async_no_match(self):
    fn = make_async_fail_then_succeed(1, exc_type=TypeError)
    with self.assertRaises(TypeError):
      await Chain(fn).retry(max_attempts=5, on=(ValueError,)).run()
    self.assertEqual(len(fn.counter), 1)

  async def test_async_subclass_match(self):
    fn = make_async_fail_then_succeed(1, exc_type=KeyError)
    result = await Chain(fn).retry(max_attempts=3, on=(LookupError,)).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  async def test_async_single_type_auto_wrapped(self):
    fn = make_async_fail_then_succeed(1, exc_type=ValueError)
    result = await Chain(fn).retry(max_attempts=3, on=ValueError).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(fn.counter), 2)

  async def test_async_control_flow_not_retried(self):
    counter = []

    async def fn(x=None):
      counter.append(1)
      Chain.return_(42)

    result = await Chain(fn).retry(max_attempts=5).run()
    self.assertEqual(result, 42)
    self.assertEqual(len(counter), 1)


# ---------------------------------------------------------------------------
# Category 3: Backoff parameter (matrix)
# ---------------------------------------------------------------------------

class TestRetryBackoffNone(unittest.TestCase):
  """backoff=None — no delay between retries."""

  def test_no_backoff_default(self):
    fn = make_fail_then_succeed(2)
    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(max_attempts=3).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_not_called()

  def test_no_backoff_explicit_none(self):
    fn = make_fail_then_succeed(2)
    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(max_attempts=3, backoff=None).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_not_called()


class TestRetryBackoffFlat(unittest.TestCase):
  """backoff=float — flat delay between retries."""

  def test_flat_backoff_called(self):
    fn = make_fail_then_succeed(2)
    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(max_attempts=3, backoff=0.01).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(mock_sleep.call_count, 2)
    for call in mock_sleep.call_args_list:
      self.assertEqual(call[0][0], 0.01)

  def test_flat_backoff_single_retry(self):
    fn = make_fail_then_succeed(1)
    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(max_attempts=2, backoff=0.05).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(mock_sleep.call_count, 1)
    self.assertEqual(mock_sleep.call_args[0][0], 0.05)


class TestRetryBackoffCallable(unittest.TestCase):
  """backoff=callable — callable receives attempt index."""

  def test_increasing_backoff(self):
    fn = make_fail_then_succeed(3)
    recorded = []

    def backoff_fn(n):
      recorded.append(n)
      return 0.01 * (n + 1)

    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(max_attempts=5, backoff=backoff_fn).run()
    self.assertEqual(result, 'ok')
    # Attempts 0, 1, 2 are retry indices (first retry=0, second=1, third=2)
    self.assertEqual(recorded, [0, 1, 2])
    self.assertEqual(mock_sleep.call_count, 3)
    self.assertAlmostEqual(mock_sleep.call_args_list[0][0][0], 0.01)
    self.assertAlmostEqual(mock_sleep.call_args_list[1][0][0], 0.02)
    self.assertAlmostEqual(mock_sleep.call_args_list[2][0][0], 0.03)

  def test_callable_returning_zero(self):
    fn = make_fail_then_succeed(2)
    recorded = []

    def backoff_fn(n):
      recorded.append(n)
      return 0

    with patch('time.sleep') as mock_sleep:
      result = Chain(fn).retry(max_attempts=3, backoff=backoff_fn).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(recorded, [0, 1])
    # sleep is not called when delay <= 0
    mock_sleep.assert_not_called()

  def test_backoff_callable_call_count(self):
    fn = make_fail_then_succeed(4)
    recorded = []

    def backoff_fn(n):
      recorded.append(n)
      return 0.001

    with patch('time.sleep'):
      Chain(fn).retry(max_attempts=5, backoff=backoff_fn).run()
    self.assertEqual(len(recorded), 4)
    self.assertEqual(recorded, [0, 1, 2, 3])


class TestRetryBackoffAsync(unittest.IsolatedAsyncioTestCase):
  """Async path uses asyncio.sleep instead of time.sleep."""

  async def test_async_flat_backoff(self):
    fn = make_async_fail_then_succeed(2)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(max_attempts=3, backoff=0.01).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(mock_sleep.call_count, 2)
    for call in mock_sleep.call_args_list:
      self.assertEqual(call[0][0], 0.01)

  async def test_async_callable_backoff(self):
    fn = make_async_fail_then_succeed(2)
    recorded = []

    def backoff_fn(n):
      recorded.append(n)
      return 0.01 * (n + 1)

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(max_attempts=3, backoff=backoff_fn).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(recorded, [0, 1])
    self.assertEqual(mock_sleep.call_count, 2)
    self.assertAlmostEqual(mock_sleep.call_args_list[0][0][0], 0.01)
    self.assertAlmostEqual(mock_sleep.call_args_list[1][0][0], 0.02)

  async def test_async_no_backoff(self):
    fn = make_async_fail_then_succeed(1)
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(max_attempts=3, backoff=None).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_not_called()

  async def test_async_callable_returning_zero(self):
    fn = make_async_fail_then_succeed(2)

    def backoff_fn(n):
      return 0

    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
      result = await Chain(fn).retry(max_attempts=3, backoff=backoff_fn).run()
    self.assertEqual(result, 'ok')
    mock_sleep.assert_not_called()


class TestRetryBackoffSyncUsesTimeSleep(unittest.TestCase):
  """Verify sync path calls time.sleep, not asyncio.sleep."""

  def test_sync_path_uses_time_sleep(self):
    fn = make_fail_then_succeed(1)
    with patch('time.sleep') as mock_time_sleep:
      Chain(fn).retry(max_attempts=3, backoff=0.01).run()
    self.assertEqual(mock_time_sleep.call_count, 1)


class TestRetryBackoffMaxAttempts1(unittest.TestCase):
  """max_attempts=1 with backoff set — backoff never called."""

  def test_backoff_never_called_max1(self):
    fn = make_always_fail()
    recorded = []

    def backoff_fn(n):
      recorded.append(n)
      return 0.01

    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=1, backoff=backoff_fn).run()
    self.assertEqual(recorded, [])


# ---------------------------------------------------------------------------
# Category 4: State reset on retry
# ---------------------------------------------------------------------------

class TestRetryStateReset(unittest.TestCase):
  """Verify chain state is fully reset between attempts."""

  def test_current_value_starts_fresh(self):
    """Each attempt starts with current_value=Null, not the value from the previous attempt."""
    values_seen = []

    def root_fn(x=None):
      return 'root_val'

    counter = []

    def step(x):
      values_seen.append(x)
      counter.append(1)
      if len(counter) <= 1:
        raise ValueError('fail')
      return x + '_done'

    result = Chain(root_fn).then(step).retry(max_attempts=3).run()
    self.assertEqual(result, 'root_val_done')
    # Both attempts see 'root_val' as current_value (not leftover from prior attempt)
    self.assertEqual(values_seen, ['root_val', 'root_val'])

  def test_root_value_recalculated_each_attempt(self):
    """Root callable is re-invoked on each retry."""
    root_counter = []

    def root_fn(x=None):
      root_counter.append(1)
      return len(root_counter) * 10

    step_counter = []

    def step(x):
      step_counter.append(x)
      if len(step_counter) <= 2:
        raise ValueError('fail')
      return x

    result = Chain(root_fn).then(step).retry(max_attempts=5).run()
    # Third attempt succeeds, root_fn called 3 times, returning 30 on 3rd call
    self.assertEqual(result, 30)
    self.assertEqual(len(root_counter), 3)
    self.assertEqual(step_counter, [10, 20, 30])

  def test_do_side_effects_happen_each_attempt(self):
    """do() steps re-execute on every retry."""
    side_effects = []

    def do_fn(x):
      side_effects.append(x)

    counter = []

    def fail_fn(x):
      counter.append(1)
      if len(counter) <= 2:
        raise ValueError('fail')
      return x

    result = Chain(10).do(do_fn).then(fail_fn).retry(max_attempts=5).run()
    self.assertEqual(result, 10)
    self.assertEqual(side_effects, [10, 10, 10])

  def test_all_links_reexecuted_from_start(self):
    """Every link in the chain re-executes on each attempt."""
    c1, c2, c3 = [], [], []

    def link1(x=None):
      c1.append(1)
      return 'a'

    def link2(x):
      c2.append(1)
      return x + 'b'

    def link3(x):
      c3.append(1)
      if len(c3) <= 1:
        raise ValueError('fail')
      return x + 'c'

    result = Chain(link1).then(link2).then(link3).retry(max_attempts=3).run()
    self.assertEqual(result, 'abc')
    self.assertEqual(len(c1), 2)
    self.assertEqual(len(c2), 2)
    self.assertEqual(len(c3), 2)

  def test_run_callable_root_args_preserved_across_retries(self):
    """run(callable, arg1, arg2) passes the same args on each retry."""
    seen_args = []

    def root(a, b):
      seen_args.append((a, b))
      if len(seen_args) <= 2:
        raise ValueError('fail')
      return a + b

    result = Chain().retry(max_attempts=5).run(root, 20, 30)
    self.assertEqual(result, 50)
    self.assertEqual(seen_args, [(20, 30)] * 3)


class TestRetryStateResetAsync(unittest.IsolatedAsyncioTestCase):
  """Async versions of state reset tests."""

  async def test_async_current_value_starts_fresh(self):
    values_seen = []

    async def root_fn(x=None):
      return 'root_val'

    counter = []

    async def step(x):
      values_seen.append(x)
      counter.append(1)
      if len(counter) <= 1:
        raise ValueError('fail')
      return x + '_done'

    result = await Chain(root_fn).then(step).retry(max_attempts=3).run()
    self.assertEqual(result, 'root_val_done')
    self.assertEqual(values_seen, ['root_val', 'root_val'])

  async def test_async_root_value_recalculated(self):
    root_counter = []

    async def root_fn(x=None):
      root_counter.append(1)
      return len(root_counter) * 10

    step_counter = []

    async def step(x):
      step_counter.append(x)
      if len(step_counter) <= 2:
        raise ValueError('fail')
      return x

    result = await Chain(root_fn).then(step).retry(max_attempts=5).run()
    self.assertEqual(result, 30)
    self.assertEqual(len(root_counter), 3)
    self.assertEqual(step_counter, [10, 20, 30])

  async def test_async_do_side_effects_each_attempt(self):
    side_effects = []

    async def do_fn(x):
      side_effects.append(x)

    counter = []

    async def fail_fn(x):
      counter.append(1)
      if len(counter) <= 2:
        raise ValueError('fail')
      return x

    result = await Chain(10).do(do_fn).then(fail_fn).retry(max_attempts=5).run()
    self.assertEqual(result, 10)
    self.assertEqual(side_effects, [10, 10, 10])

  async def test_async_run_callable_root_args_preserved(self):
    seen_args = []

    async def root(a, b):
      seen_args.append((a, b))
      if len(seen_args) <= 1:
        raise ValueError('fail')
      return a + b

    result = await Chain().retry(max_attempts=5).run(root, 2, 3)
    self.assertEqual(result, 5)
    self.assertEqual(seen_args, [(2, 3), (2, 3)])


# ---------------------------------------------------------------------------
# Category 5: max_attempts edge cases
# ---------------------------------------------------------------------------

class TestRetryMaxAttemptsEdgeCases(unittest.TestCase):
  """Edge cases for max_attempts values."""

  def test_max_attempts_1_no_retry(self):
    """max_attempts=1 means no retry — exception propagates immediately."""
    fn = make_always_fail()
    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=1).run()
    self.assertEqual(len(fn.counter), 1)

  def test_max_attempts_0_treated_as_1(self):
    """max_attempts=0 — falsy value defaults to 1 via `or 1`, so chain runs once."""
    fn = make_counter_fn()
    result = Chain(fn).retry(max_attempts=0).run()
    # 0 is falsy, so `self._retry_max_attempts or 1` evaluates to 1
    self.assertEqual(result, 'counted')
    self.assertEqual(len(fn.counter), 1)

  def test_max_attempts_negative_treated_as_1(self):
    """max_attempts=-1 — negative is truthy, but range(-1) is empty, so no execution."""
    fn = make_counter_fn()
    result = Chain(fn).retry(max_attempts=-1).run()
    # -1 is truthy, so max_attempts=-1 is used directly. range(-1) is empty.
    self.assertIsNone(result)
    self.assertEqual(len(fn.counter), 0)

  def test_large_max_attempts_with_early_success(self):
    """max_attempts=100 but succeed on attempt 3."""
    fn = make_fail_then_succeed(2, success_value='early')
    result = Chain(fn).retry(max_attempts=100).run()
    self.assertEqual(result, 'early')
    self.assertEqual(len(fn.counter), 3)

  def test_max_attempts_1_with_backoff_never_called(self):
    """max_attempts=1 with backoff set — backoff is never invoked."""
    fn = make_always_fail()
    backoff_calls = []

    def backoff_fn(n):
      backoff_calls.append(n)
      return 0.01

    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=1, backoff=backoff_fn).run()
    self.assertEqual(backoff_calls, [])

  def test_max_attempts_1_succeeds_normally(self):
    fn = make_fail_then_succeed(0, success_value='ok1')
    result = Chain(fn).retry(max_attempts=1).run()
    self.assertEqual(result, 'ok1')
    self.assertEqual(len(fn.counter), 1)


class TestRetryMaxAttemptsEdgeCasesAsync(unittest.IsolatedAsyncioTestCase):
  """Async versions of max_attempts edge cases."""

  async def test_async_max_attempts_1_no_retry(self):
    fn = make_async_always_fail()
    with self.assertRaises(ValueError):
      await Chain(fn).retry(max_attempts=1).run()
    self.assertEqual(len(fn.counter), 1)

  async def test_async_max_attempts_0_treated_as_1(self):
    fn = make_async_counter_fn()
    result = await Chain(fn).retry(max_attempts=0).run()
    # 0 is falsy, so `self._retry_max_attempts or 1` evaluates to 1
    self.assertEqual(result, 'counted')
    self.assertEqual(len(fn.counter), 1)

  async def test_async_large_max_attempts_early_success(self):
    fn = make_async_fail_then_succeed(2, success_value='early_async')
    result = await Chain(fn).retry(max_attempts=100).run()
    self.assertEqual(result, 'early_async')
    self.assertEqual(len(fn.counter), 3)


# ---------------------------------------------------------------------------
# Additional: except_/finally_ interaction with retry
# ---------------------------------------------------------------------------

class TestRetryExceptFinallyInteraction(unittest.TestCase):
  """except_/finally_ fire only after all retries exhausted."""

  def test_except_fires_after_retries_exhausted(self):
    fn = make_always_fail()
    except_calls = []

    def handler(exc):
      except_calls.append(str(exc))
      return 'handled'

    result = Chain(fn).retry(max_attempts=3).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(fn.counter), 3)
    self.assertEqual(len(except_calls), 1)
    self.assertEqual(except_calls[0], 'always')

  def test_except_not_called_on_successful_retry(self):
    fn = make_fail_then_succeed(2)
    except_calls = []

    def handler(exc):
      except_calls.append(1)
      return 'handled'

    result = Chain(fn).retry(max_attempts=3).except_(handler).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(except_calls), 0)

  def test_finally_fires_after_retries(self):
    fn = make_fail_then_succeed(1)
    finally_calls = []

    def cleanup(x):
      finally_calls.append(x)

    result = Chain(fn).retry(max_attempts=3).finally_(cleanup).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(finally_calls), 1)

  def test_finally_fires_on_failure_after_retries(self):
    fn = make_always_fail()
    finally_calls = []

    def cleanup(x=None):
      finally_calls.append('cleaned')

    with self.assertRaises(ValueError):
      Chain(fn).retry(max_attempts=3).finally_(cleanup).run()
    self.assertEqual(len(fn.counter), 3)
    self.assertEqual(len(finally_calls), 1)


class TestRetryExceptFinallyInteractionAsync(unittest.IsolatedAsyncioTestCase):
  """Async versions of except_/finally_ interaction."""

  async def test_async_except_fires_after_retries_exhausted(self):
    fn = make_async_always_fail()
    except_calls = []

    def handler(exc):
      except_calls.append(str(exc))
      return 'handled'

    result = await Chain(fn).retry(max_attempts=3).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(fn.counter), 3)
    self.assertEqual(len(except_calls), 1)

  async def test_async_except_not_called_on_success(self):
    fn = make_async_fail_then_succeed(1)
    except_calls = []

    def handler(exc):
      except_calls.append(1)
      return 'handled'

    result = await Chain(fn).retry(max_attempts=3).except_(handler).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(except_calls), 0)

  async def test_async_finally_fires_after_retries(self):
    fn = make_async_fail_then_succeed(1)
    finally_calls = []

    def cleanup(x):
      finally_calls.append(x)

    result = await Chain(fn).retry(max_attempts=3).finally_(cleanup).run()
    self.assertEqual(result, 'ok')
    self.assertEqual(len(finally_calls), 1)


# ---------------------------------------------------------------------------
# Additional: retry with freeze
# ---------------------------------------------------------------------------

class TestRetryWithFreeze(unittest.TestCase):
  """Retry works through frozen chains."""

  def test_frozen_chain_retries(self):
    fn = make_fail_then_succeed(1, success_value='frozen_ok')
    frozen = Chain(fn).retry(max_attempts=3).freeze()
    result = frozen.run()
    self.assertEqual(result, 'frozen_ok')
    self.assertEqual(len(fn.counter), 2)

  def test_frozen_chain_reusable_with_retry(self):
    """A frozen chain with retry can be called multiple times."""
    call_count = []

    def fn(x=None):
      call_count.append(1)
      total = len(call_count)
      # First two calls of each batch fail
      if total % 3 != 0:
        raise ValueError('fail')
      return total

    frozen = Chain(fn).retry(max_attempts=3).freeze()
    r1 = frozen.run()
    self.assertEqual(r1, 3)
    r2 = frozen.run()
    self.assertEqual(r2, 6)


# ---------------------------------------------------------------------------
# Additional: retry with __call__
# ---------------------------------------------------------------------------

class TestRetryViaCall(unittest.TestCase):
  """Chain.__call__ is an alias for .run() — verify retry works through it."""

  def test_call_triggers_retry(self):
    fn = make_fail_then_succeed(1, success_value='call_ok')
    chain = Chain(fn).retry(max_attempts=3)
    result = chain()
    self.assertEqual(result, 'call_ok')
    self.assertEqual(len(fn.counter), 2)

  def test_call_with_callable_root_triggers_retry(self):
    counter = []

    def root(y):
      counter.append(y)
      if len(counter) <= 1:
        raise ValueError('fail')
      return y * 2

    chain = Chain().retry(max_attempts=3)
    result = chain(root, 4)
    self.assertEqual(result, 8)
    self.assertEqual(counter, [4, 4])


# ---------------------------------------------------------------------------
# Additional: retry with no retry config (baseline)
# ---------------------------------------------------------------------------

class TestNoRetryBaseline(unittest.TestCase):
  """Chains without .retry() should behave exactly as before."""

  def test_no_retry_single_failure_propagates(self):
    fn = make_always_fail()
    with self.assertRaises(ValueError):
      Chain(fn).run()
    self.assertEqual(len(fn.counter), 1)

  def test_no_retry_success(self):
    fn = make_fail_then_succeed(0, success_value='baseline')
    result = Chain(fn).run()
    self.assertEqual(result, 'baseline')
    self.assertEqual(len(fn.counter), 1)


if __name__ == '__main__':
  unittest.main()
