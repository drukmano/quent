"""Property/invariant tests: structural properties verified with subTest() over dense value sets."""
from __future__ import annotations

import asyncio
import unittest
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  SyncCM,
  SyncCMEnterReturnsNone,
  SyncCMEnterReturnsSelf,
  TrackingCM,
  sync_identity,
  async_identity,
)


# ---------------------------------------------------------------------------
# TestDoNeverChangesValue
# ---------------------------------------------------------------------------

class TestDoNeverChangesValue(unittest.TestCase):
  """Invariant: do() NEVER changes current_value, regardless of what fn returns."""

  def test_do_preserves_value_dense(self):
    fn_returns = [None, 0, False, '', [], {}, 42, 'hello', lambda: 99, Exception('x')]
    input_vals = [1, 'test', [1, 2]]
    for input_val in input_vals:
      for fn_ret in fn_returns:
        with self.subTest(input_val=input_val, fn_ret=fn_ret):
          result = Chain(input_val).do(lambda x, r=fn_ret: r).run()
          self.assertEqual(result, input_val)

  def test_do_preserves_with_exception(self):
    """If do()'s fn raises, current_value was not changed before the raise."""
    tracker = []

    def failing_fn(x):
      tracker.append(x)
      raise ValueError('boom')

    with self.assertRaises(ValueError):
      Chain(42).do(failing_fn).run()
    # The fn received the correct current_value before raising
    self.assertEqual(tracker, [42])

  def test_multiple_do_preserves(self):
    f1 = lambda x: x * 100
    f2 = lambda x: 'replaced'
    f3 = lambda x: None
    result = Chain(5).do(f1).do(f2).do(f3).run()
    self.assertEqual(result, 5)


# ---------------------------------------------------------------------------
# TestThenAlwaysReplaces
# ---------------------------------------------------------------------------

class TestThenAlwaysReplaces(unittest.TestCase):
  """Invariant: then() ALWAYS replaces current_value with fn's return value."""

  def test_then_replaces_dense(self):
    replacements = [42, 'hello', [1, 2, 3], {'k': 'v'}, (1,), 3.14, True, b'bytes']
    for replacement in replacements:
      with self.subTest(replacement=replacement):
        result = Chain(999).then(lambda x, r=replacement: r).run()
        self.assertEqual(result, replacement)

  def test_then_replaces_with_none(self):
    result = Chain(42).then(lambda x: None).run()
    self.assertIsNone(result)

  def test_then_replaces_with_false(self):
    result = Chain(42).then(lambda x: False).run()
    self.assertIs(result, False)

  def test_then_replaces_with_callable(self):
    """A callable returned by then() is stored as the value, not invoked."""
    fn = lambda: 'should not be called'
    result = Chain(42).then(lambda x: fn).run()
    self.assertIs(result, fn)
    self.assertTrue(callable(result))

  def test_then_replaces_with_zero(self):
    result = Chain(42).then(lambda x: 0).run()
    self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# TestForeachPreservesItems
# ---------------------------------------------------------------------------

class TestForeachPreservesItems(unittest.TestCase):
  """Invariant: foreach() ALWAYS returns original items, ignoring fn results."""

  def test_foreach_preserves_dense(self):
    fn_returns = [None, 0, False, '', [], {}, 42, 'replaced', lambda: 99]
    for fn_ret in fn_returns:
      with self.subTest(fn_ret=fn_ret):
        result = Chain([10, 20, 30]).foreach(lambda x, r=fn_ret: r).run()
        self.assertEqual(result, [10, 20, 30])


# ---------------------------------------------------------------------------
# TestFinallyDoesNotAffectResult
# ---------------------------------------------------------------------------

class TestFinallyDoesNotAffectResult(unittest.TestCase):
  """Invariant: finally_() result NEVER affects chain return value."""

  def test_finally_result_ignored_dense(self):
    handler_returns = [None, 0, False, '', [], {}, 42, 'replaced', 999]
    for handler_ret in handler_returns:
      with self.subTest(handler_ret=handler_ret):
        result = Chain(5).then(lambda x: x * 2).finally_(lambda x, r=handler_ret: r).run()
        self.assertEqual(result, 10)

  def test_finally_result_ignored_sync(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .finally_(lambda x: tracker.append('cleanup') or 'ignored')
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, ['cleanup'])

  def test_finally_result_ignored_async(self):
    """Async variant: finally_ result is ignored even in async chains."""

    async def async_step(x):
      return x + 1

    async def run_it():
      return await (
        Chain(5)
        .then(async_step)
        .finally_(lambda x: 'ignored_return')
        .run()
      )

    result = asyncio.run(run_it())
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# TestMonadLawsDense
# ---------------------------------------------------------------------------

class TestMonadLawsDense(unittest.TestCase):
  """Monad laws verified over a dense set of edge-case values."""

  def test_left_identity_dense(self):
    """Chain(a).then(f).run() == f(a)"""
    f = lambda x: (x, 'transformed')
    values = [0, None, False, '', [], {}, 42, 'hello']
    for a in values:
      with self.subTest(a=a):
        result = Chain(a).then(f).run()
        expected = f(a)
        self.assertEqual(result, expected)

  def test_right_identity_dense(self):
    """Chain(a).then(identity).run() == a"""
    values = [0, None, False, '', [], {}, 42, 'hello']
    for a in values:
      with self.subTest(a=a):
        result = Chain(a).then(sync_identity).run()
        self.assertEqual(result, a)

  def test_associativity_dense(self):
    """Chain(a).then(f).then(g).run() == Chain(a).then(lambda x: g(f(x))).run()"""
    f = lambda x: (x, 'f')
    g = lambda x: (x, 'g')
    values = [0, None, False, '', [], {}, 42, 'hello']
    for a in values:
      with self.subTest(a=a):
        lhs = Chain(a).then(f).then(g).run()
        rhs = Chain(a).then(lambda x: g(f(x))).run()
        self.assertEqual(lhs, rhs)


# ---------------------------------------------------------------------------
# TestChainBoolAlwaysTrue
# ---------------------------------------------------------------------------

class TestChainBoolAlwaysTrue(unittest.TestCase):
  """Invariant: bool(chain) is ALWAYS True regardless of contents."""

  def test_bool_dense(self):
    subjects = [
      None, False, 0, '', [], Chain(), Chain(0), Chain(False),
    ]
    for subject in subjects:
      with self.subTest(subject=subject):
        if isinstance(subject, Chain):
          self.assertTrue(bool(subject))
        else:
          self.assertTrue(bool(Chain(subject)))



# ---------------------------------------------------------------------------
# TestWithDoPreservesOuter
# ---------------------------------------------------------------------------

class TestWithDoPreservesOuter(unittest.TestCase):
  """Invariant: with_do() ALWAYS returns the CM value (the outer current_value)."""

  def test_with_do_preserves(self):
    body_returns = [None, 0, False, 42, 'replaced', [], {}, lambda: 99]
    for body_ret in body_returns:
      with self.subTest(body_ret=body_ret):
        cm = SyncCM()
        result = Chain(cm).with_do(lambda ctx, r=body_ret: r).run()
        self.assertIs(result, cm)

  def test_with_do_preserves_tracking_cm(self):
    cm = TrackingCM()
    result = Chain(cm).with_do(lambda ctx: 'body_result').run()
    self.assertIs(result, cm)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# Additional property tests (BEYOND SPEC)
# ---------------------------------------------------------------------------

class TestDoWithNonCallable(unittest.TestCase):
  """do() with a non-callable: raises TypeError eagerly."""

  def test_do_rejects_int(self):
    """do(42): raises TypeError because do() requires a callable."""
    with self.assertRaises(TypeError) as cm:
      Chain(5).do(42)
    self.assertEqual(str(cm.exception), 'do() requires a callable, got int')

  def test_do_rejects_none(self):
    with self.assertRaises(TypeError) as cm:
      Chain(5).do(None)
    self.assertEqual(str(cm.exception), 'do() requires a callable, got NoneType')


class TestChainAllStepsReturnNone(unittest.TestCase):
  """Chain where every step returns None."""

  def test_all_none_then(self):
    result = Chain(None).then(lambda x: None).then(lambda x: None).run()
    self.assertIsNone(result)

  def test_all_none_do(self):
    """do() preserves current value, so None is preserved."""
    result = Chain(None).do(lambda x: None).do(lambda x: None).run()
    self.assertIsNone(result)


class TestChainAllStepsReturnNull(unittest.TestCase):
  """Chain where every step returns Null sentinel."""

  def test_all_null_then(self):
    """Null is not callable, so .then(Null) sets current_value to Null.
    At the end: current_value is Null -> returns None."""
    result = Chain(5).then(Null).run()
    self.assertIsNone(result)


class TestRunWithNullExplicit(unittest.TestCase):
  """run(Null) is equivalent to run() -- Null means 'no value'."""

  def test_run_null_uses_root(self):
    result = Chain(5).run(Null)
    self.assertEqual(result, 5)

  def test_run_null_with_steps(self):
    result = Chain(5).then(lambda x: x * 2).run(Null)
    self.assertEqual(result, 10)


class TestThenWithEllipsisAsValue(unittest.TestCase):
  """Then with Ellipsis as the value (not as arg sentinel)."""

  def test_ellipsis_value_stored(self):
    """When Ellipsis is passed as the value to then(), it is not callable,
    so it is returned as-is and replaces the current_value."""
    result = Chain(5).then(...).run()
    self.assertIs(result, ...)


class TestChainReuse(unittest.TestCase):
  """Running the same chain 100 times produces consistent results."""

  def test_reuse_100_times(self):
    c = Chain().then(lambda x: x * 2)
    for i in range(100):
      with self.subTest(i=i):
        result = c.run(i)
        self.assertEqual(result, i * 2)



class TestDeeplyNestedChains(unittest.TestCase):
  """Deeply nested chains (20+ levels)."""

  def test_20_levels_nested(self):
    c = Chain().then(lambda x: x)
    for _ in range(19):
      c = Chain().then(c)
    result = c.run(42)
    self.assertEqual(result, 42)

  def test_25_levels_nested_with_transform(self):
    c = Chain().then(lambda x: x + 1)
    for _ in range(24):
      c = Chain().then(c)
    result = c.run(0)
    self.assertEqual(result, 1)


class TestDecoratorOnAlreadyDecorated(unittest.TestCase):
  """decorator() on a function that's already decorated."""

  def test_double_decoration(self):
    import functools

    def my_decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        return fn(*args, **kwargs) + 100
      return wrapper

    chain = Chain().then(lambda x: x * 2)

    @chain.decorator()
    @my_decorator
    def my_fn(x):
      return x

    result = my_fn(5)
    # chain receives my_fn (wrapped by my_decorator).
    # chain.run(my_fn) -> my_fn is callable -> my_fn(Null) ? No:
    # decorator creates: chain._run(fn, args, kwargs)
    # fn = the wrapped function, args = (5,), kwargs = {}
    # So root link is Link(fn, (5,), {}), fn(5) -> my_decorator wrapper -> 5 + 100 = 105
    # Then: then(lambda x: x * 2) -> 105 * 2 = 210
    self.assertEqual(result, 210)


class TestZeroArgLambdaAsRootWithRunValue(unittest.TestCase):
  """Chain with 0-arg lambda as root + run(value)."""

  def test_run_value_overrides_root(self):
    """run(value) creates a temporary Link that becomes the root, bypassing
    the original root_link."""
    c = Chain(lambda: 'root_result')
    result = c.run(99)
    # run(99) -> Link(99, (), {}) spliced before first_link
    # 99 is not callable, returned as-is -> current_value = 99
    # No first_link steps -> returns 99
    self.assertEqual(result, 99)

  def test_run_no_value_uses_root_lambda(self):
    c = Chain(lambda: 'root_result')
    result = c.run()
    self.assertEqual(result, 'root_result')


class TestDoPreservesInChainContext(unittest.TestCase):
  """do() preserves value in multi-step chains."""

  def test_do_between_thens(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x + 1)
      .do(lambda x: tracker.append(x))
      .then(lambda x: x * 2)
      .run()
    )
    self.assertEqual(result, 12)
    self.assertEqual(tracker, [6])


class TestFinallyReceivedRootValue(unittest.TestCase):
  """finally_ receives the root value, not the current pipeline value."""

  def test_finally_gets_root(self):
    received = []
    Chain(5).then(lambda x: x * 100).finally_(lambda x: received.append(x)).run()
    self.assertEqual(received, [5])


class TestThenReplacesChainingSequence(unittest.TestCase):
  """then() replacements chain correctly."""

  def test_then_chain_of_replacements(self):
    result = (
      Chain(1)
      .then(lambda x: x + 1)  # 2
      .then(lambda x: x * 3)  # 6
      .then(lambda x: x - 1)  # 5
      .then(lambda x: str(x))  # '5'
      .run()
    )
    self.assertEqual(result, '5')


class TestForeachSideEffectsExecuted(unittest.TestCase):
  """foreach() executes side effects even though it preserves items."""

  def test_side_effects_run(self):
    tracker = []
    result = Chain([10, 20, 30]).foreach(lambda x: tracker.append(x * 2)).run()
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(tracker, [20, 40, 60])


if __name__ == '__main__':
  unittest.main()
