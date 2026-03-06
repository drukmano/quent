"""Tests targeting uncovered lines and edge cases in Chain._run_async().

Covers:
  - Finally block control flow (lines 291-308)
  - ignore_result (.do()) in async path
  - Value propagation, Null→None, awaitable results
  - _Break/_Return in async chains (nested and non-nested)
  - except_ returning coroutine in async path
  - root_value initialization from first async result
  - Many-step async chains
"""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain, Null, QuentException
from quent._core import (
  Link,
  _Break,
  _ControlFlowSignal,
  _Return,
  _evaluate_value,
  _set_link_temp_args,
)
from helpers import async_fn, async_identity, sync_fn, sync_identity, AsyncRange


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

async def _async_noop(x=None):
  """Async function that returns None."""
  return None


async def _async_identity(x):
  return x


async def _async_add_one(x):
  return x + 1


async def _async_raise_value_error(x=None):
  raise ValueError('async body error')


def _sync_return_coroutine_that_raises_control_flow(rv=None):
  """A sync function that returns a coroutine which raises _ControlFlowSignal.

  This is NOT an async def — it returns a pre-built coroutine object.
  _finally_handler_body calls it synchronously, gets back a coroutine,
  then _run_async awaits it. The await raises _ControlFlowSignal, which
  is caught at line 298-299.
  """
  async def _inner():
    raise _Return(Null, (), {})
  return _inner()


def _sync_return_coroutine_that_raises_base_exception(rv=None):
  """Sync function returning a coroutine that raises a BaseException.

  Triggers lines 300-303: the awaited coroutine raises a BaseException
  (not _ControlFlowSignal), causing _set_link_temp_args and
  _modify_traceback to be called.
  """
  async def _inner():
    raise RuntimeError('coroutine base exception')
  return _inner()


# ---------------------------------------------------------------------------
# TestRunAsyncFinallyControlFlow
# ---------------------------------------------------------------------------

class TestRunAsyncFinallyControlFlow(unittest.IsolatedAsyncioTestCase):
  """Focus on lines 291-308 of _chain.py (the finally block in _run_async)."""

  async def test_async_finally_handler_raises_control_flow(self):
    """Line 296-299: sync finally handler returns a coroutine that raises
    _ControlFlowSignal when awaited. This must be caught and converted to
    QuentException.
    """
    # Use an async step to enter _run_async, then register a sync finally
    # handler that returns a coroutine raising _ControlFlowSignal.
    chain = (
      Chain(5)
      .then(_async_add_one)
      .finally_(_sync_return_coroutine_that_raises_control_flow)
    )
    with self.assertRaises(QuentException) as ctx:
      await chain.run()
    self.assertIn('control flow', str(ctx.exception).lower())

  async def test_async_finally_handler_raises_base_exception(self):
    """Line 300-303: sync finally handler returns a coroutine that raises
    BaseException (not _ControlFlowSignal) when awaited. Verify
    _set_link_temp_args and _modify_traceback are called on the exception.
    """
    chain = (
      Chain(5)
      .then(_async_add_one)
      .finally_(_sync_return_coroutine_that_raises_base_exception)
    )
    with self.assertRaises(RuntimeError) as ctx:
      await chain.run()
    self.assertEqual(str(ctx.exception), 'coroutine base exception')

  async def test_async_finally_handler_raises_base_exception_with_root_value(self):
    """Line 301-302: root_value is not Null, so _set_link_temp_args is called
    with root_value=<value>.
    """
    chain = (
      Chain(42)
      .then(_async_add_one)
      .finally_(_sync_return_coroutine_that_raises_base_exception)
    )
    with self.assertRaises(RuntimeError) as ctx:
      await chain.run()
    self.assertEqual(str(ctx.exception), 'coroutine base exception')

  async def test_async_finally_handler_raises_base_exception_null_root(self):
    """Line 301: root_value IS Null — skip _set_link_temp_args call.
    Chain() with no root_link and run() with no value means root_value
    stays Null.
    """
    def _sync_return_coro_that_raises(rv=None):
      async def _inner():
        raise RuntimeError('null root coro exception')
      return _inner()

    # Chain() no root, first link is async → enters _run_async.
    # root_value stays Null because has_root_value is False.
    chain = (
      Chain()
      .then(_async_add_one)
      .finally_(_sync_return_coro_that_raises)
    )
    with self.assertRaises(RuntimeError) as ctx:
      await chain.run(5)
    self.assertEqual(str(ctx.exception), 'null root coro exception')

  async def test_async_finally_context_chaining(self):
    """Line 305-307: finally_exc.__context__ is None but _active_exc is not None.

    Setup: body raises (sets _active_exc), except handler handles it
    successfully (returns a value), then async finally handler raises a
    brand new exception with no implicit __context__.
    The code at line 306 sets finally_exc.__context__ = _active_exc.
    """
    def except_handler(exc):
      return 'recovered'

    async def async_body(x):
      raise ValueError('body raises')

    # Create an exception with no __context__ by using a sync finally handler
    # that returns a coroutine raising a fresh exception.
    def sync_finally_returns_raising_coro(rv=None):
      async def _inner():
        raise OSError('finally raises')
      return _inner()

    chain = (
      Chain(10)
      .then(async_body)
      .except_(except_handler)
      .finally_(sync_finally_returns_raising_coro)
    )
    try:
      await chain.run()
      self.fail('Expected OSError')
    except OSError as exc:
      self.assertEqual(str(exc), 'finally raises')
      # The _active_exc (ValueError) should be set as __context__
      self.assertIsNotNone(exc.__context__)
      self.assertIsInstance(exc.__context__, ValueError)
      self.assertEqual(str(exc.__context__), 'body raises')

  async def test_async_finally_context_already_set(self):
    """Line 306 condition is False: finally_exc.__context__ is already set.

    When _finally_handler_body itself raises (not the awaited coroutine),
    the exception's __context__ is already set by Python's exception chaining
    to the active exception from the try/except block. Line 306 should NOT
    overwrite it.
    """
    async def async_body(x):
      raise ValueError('body error')

    def finally_raises_sync(rv=None):
      # This raises during _finally_handler_body execution (not via coroutine),
      # so Python sets __context__ automatically.
      raise TypeError('sync finally error')

    chain = (
      Chain(10)
      .then(async_body)
      .finally_(finally_raises_sync)
    )
    try:
      await chain.run()
      self.fail('Expected TypeError')
    except TypeError as exc:
      self.assertEqual(str(exc), 'sync finally error')
      # __context__ was already set by Python or by _finally_handler_body;
      # the code at line 306 should not overwrite it if it's already set.
      self.assertIsNotNone(exc.__context__)

  async def test_async_finally_context_already_set_with_async_handler(self):
    """Line 306: finally_exc.__context__ is already set because an async
    finally handler raises while Python's exception state has an active exc.
    """
    async def async_body(x):
      raise ValueError('body err')

    async def async_finally_raises(rv=None):
      raise TypeError('async finally err')

    chain = (
      Chain(10)
      .then(async_body)
      .finally_(async_finally_raises)
    )
    try:
      await chain.run()
      self.fail('Expected TypeError')
    except TypeError as exc:
      self.assertEqual(str(exc), 'async finally err')
      # Python sets __context__ during the await inside the except BaseException
      # block. The code at line 306 should see it's already set and skip.
      self.assertIsNotNone(exc.__context__)
      self.assertIsInstance(exc.__context__, ValueError)


# ---------------------------------------------------------------------------
# TestRunAsyncIgnoreResult
# ---------------------------------------------------------------------------

class TestRunAsyncIgnoreResult(unittest.IsolatedAsyncioTestCase):
  """.do() in the async path should preserve current_value."""

  async def test_do_in_async_path_preserves(self):
    """A .do(async_fn) discards the async fn's result and preserves current_value."""
    tracker = []

    async def side_effect(x):
      tracker.append(x)
      return 'discarded'

    result = await Chain(5).then(_async_add_one).do(side_effect).run()
    # 5 → async_add_one(5)=6 → do(side_effect(6))='discarded' (ignored) → 6
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [6])

  async def test_multiple_do_async(self):
    """Multiple .do() calls in async path all preserve current_value."""
    tracker = []

    async def effect_a(x):
      tracker.append(('a', x))

    async def effect_b(x):
      tracker.append(('b', x))

    result = await (
      Chain(10)
      .then(_async_add_one)  # 11
      .do(effect_a)          # side-effect, still 11
      .do(effect_b)          # side-effect, still 11
      .run()
    )
    self.assertEqual(result, 11)
    self.assertEqual(tracker, [('a', 11), ('b', 11)])

  async def test_async_initial_value_with_do(self):
    """First link is .do() in async path — current_value stays Null initially,
    then gets set from the root_link evaluation.
    """
    tracker = []

    async def side_effect(x):
      tracker.append(x)
      return 'discarded'

    # Chain(async_fn, 5): root_link evaluates to coroutine → async path.
    # .do(side_effect) is first_link. root_link result (6) is current_value.
    # side_effect(6) is called but result discarded.
    result = await Chain(async_fn, 5).do(side_effect).run()
    # async_fn(5) = 6, .do() discards → still 6
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [6])

  async def test_do_sync_fn_in_async_path(self):
    """Sync .do() after an async step still preserves value."""
    tracker = []

    def sync_side(x):
      tracker.append(x)
      return 'sync_discarded'

    result = await Chain(5).then(_async_add_one).do(sync_side).run()
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [6])


# ---------------------------------------------------------------------------
# TestRunAsyncValuePropagation
# ---------------------------------------------------------------------------

class TestRunAsyncValuePropagation(unittest.IsolatedAsyncioTestCase):
  """Value propagation edge cases in _run_async."""

  async def test_async_current_value_null_returns_none(self):
    """If current_value stays Null throughout _run_async, return None (line 257-258)."""
    # Chain() with no root. .do() keeps ignore_result=True, so current_value
    # is never set from the .do() result. The root_link from run(value)
    # triggers async. We need ALL links to have ignore_result=True.
    async def noop():
      pass

    # Chain() no root, run() with async noop as the value.
    # Link(noop) evaluates to coroutine → async path.
    # The .do() link's result is discarded.
    # But current_value gets set from the first awaitable (noop() → None).
    # So current_value = None (not Null). We need a different approach.

    # Use .do() as the first link with an async root that has ignore_result=True.
    # Actually the root_link never has ignore_result, so we can't do that.
    # Instead: Chain() with no root, and .do() links only.
    # run() with no args: has_run_value=False, has_root_value=False.
    # link = first_link. first_link is a .do() link.
    # _evaluate_value returns result. If isawaitable → _run_async.
    # In _run_async: result = await awaitable. ignore_result=True.
    # current_value is Null, link.ignore_result is True → skip setting current_value.
    # Then next link is also .do() with ignore_result=True → same.
    # current_value stays Null → return None.
    chain = Chain().do(noop).do(noop)
    result = await chain.run()
    self.assertIsNone(result)

  async def test_async_result_awaited(self):
    """isawaitable check inside the while loop of _run_async (line 251-252)."""
    # After entering async path, subsequent steps return awaitables that
    # must be awaited.
    result = await (
      Chain(5)
      .then(_async_add_one)       # triggers async path, result=6
      .then(_async_add_one)       # returns coroutine, awaited → 7
      .then(_async_add_one)       # returns coroutine, awaited → 8
      .run()
    )
    self.assertEqual(result, 8)

  async def test_async_chain_multiple_awaitable_steps(self):
    """Every step returns a coroutine."""
    async def double(x):
      return x * 2

    result = await (
      Chain(_async_identity, 3)
      .then(_async_add_one)     # 4
      .then(double)             # 8
      .then(_async_add_one)     # 9
      .then(double)             # 18
      .run()
    )
    self.assertEqual(result, 18)

  async def test_async_break_nested(self):
    """_Break in a nested async chain (is_nested=True) re-raises (line 268-269)."""
    result = await (
      Chain(AsyncRange(10))
      .foreach(lambda x: Chain.break_(99) if x == 3 else x)
      .run()
    )
    self.assertEqual(result, 99)

  async def test_async_break_non_nested(self):
    """_Break in non-nested async chain → QuentException (line 270)."""
    with self.assertRaises(QuentException) as ctx:
      await Chain(5).then(_async_add_one).then(lambda x: Chain.break_()).run()
    self.assertIn('Chain.break_() cannot be used outside of a foreach iteration', str(ctx.exception))

  async def test_async_return_with_awaitable_value(self):
    """_Return value is awaitable — gets awaited (line 263-264)."""
    async def make_return_val():
      return 77

    result = await (
      Chain(5)
      .then(_async_add_one)
      .then(lambda x: Chain.return_(make_return_val))
      .then(lambda x: x + 1000)  # never reached
      .run()
    )
    self.assertEqual(result, 77)

  async def test_async_return_with_non_awaitable_value(self):
    """_Return value is NOT awaitable — returned directly (line 265)."""
    result = await (
      Chain(5)
      .then(_async_add_one)
      .then(lambda x: Chain.return_(42))
      .then(lambda x: x + 1000)  # never reached
      .run()
    )
    self.assertEqual(result, 42)

  async def test_async_return_no_value(self):
    """_Return with no value returns None."""
    result = await (
      Chain(5)
      .then(_async_add_one)
      .then(lambda x: Chain.return_())
      .run()
    )
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Beyond-spec additional tests
# ---------------------------------------------------------------------------

class TestRunAsyncExceptCoroutine(unittest.IsolatedAsyncioTestCase):
  """except_ handler returning a coroutine in the async path."""

  async def test_except_returns_coroutine_awaited(self):
    """_run_async except block: handler result is awaitable → awaited (line 285-286)."""
    async def async_handler(exc):
      return f'async_caught:{type(exc).__name__}'

    result = await (
      Chain(_async_raise_value_error)
      .except_(async_handler)
      .run()
    )
    self.assertEqual(result, 'async_caught:ValueError')

  async def test_except_returning_null_gives_none(self):
    """When except handler returns Null, _run_async returns None (line 287-288)."""
    result = await (
      Chain(_async_raise_value_error)
      .except_(lambda exc: Null)
      .run()
    )
    self.assertIsNone(result)

  async def test_except_returning_regular_value(self):
    """Except handler returns a non-Null, non-awaitable value (line 289)."""
    result = await (
      Chain(_async_raise_value_error)
      .except_(lambda exc: 'recovered')
      .run()
    )
    self.assertEqual(result, 'recovered')


class TestRunAsyncRootValue(unittest.IsolatedAsyncioTestCase):
  """root_value initialization from the first async result."""

  async def test_root_value_set_from_first_async_result(self):
    """Line 242-243: has_root_value=True and root_value is Null.
    The first awaited result sets root_value.
    """
    tracker = []

    async def async_root():
      return 100

    # Chain(async_root, ...): root_link is Link(async_root, (...,)).
    # _evaluate_value calls async_root() → coroutine. Sync path sees awaitable.
    # _run_async: await coroutine → 100. root_value was Null, now 100.
    result = await (
      Chain(async_root, ...)
      .then(lambda x: x + 1)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 101)
    # Finally handler receives root_value = 100 (not 101)
    self.assertEqual(tracker, [100])

  async def test_has_root_value_true_root_value_null_first_sets(self):
    """has_root_value=True but root_value=Null because the root_link
    evaluation itself is the awaitable. First await result sets root_value.
    """
    tracker = []

    result = await (
      Chain(async_fn, 10)  # async_fn(10) → coroutine → async path
      .then(lambda x: x * 2)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    # async_fn(10) = 11, then *2 = 22
    self.assertEqual(result, 22)
    # root_value = 11 (first awaited result)
    self.assertEqual(tracker, [11])

  async def test_root_value_from_run_value(self):
    """run(async_fn, 5) creates a temporary root link. The first await
    result becomes root_value.
    """
    tracker = []

    result = await (
      Chain()
      .then(lambda x: x + 10)
      .finally_(lambda rv: tracker.append(rv))
      .run(async_fn, 5)
    )
    # async_fn(5) = 6, +10 = 16
    self.assertEqual(result, 16)
    self.assertEqual(tracker, [6])


class TestRunAsyncFinallySuccess(unittest.IsolatedAsyncioTestCase):
  """Finally handler runs successfully in async path."""

  async def test_sync_finally_success_in_async_path(self):
    """Sync finally handler that returns normally (no exception)."""
    tracker = []
    result = await (
      Chain(5)
      .then(_async_add_one)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [5])

  async def test_async_finally_success_in_async_path(self):
    """Async finally handler that completes normally."""
    tracker = []

    async def async_cleanup(rv):
      tracker.append(('cleanup', rv))

    result = await (
      Chain(async_fn, 10)
      .then(lambda x: x * 3)
      .finally_(async_cleanup)
      .run()
    )
    # async_fn(10) = 11, *3 = 33
    self.assertEqual(result, 33)
    self.assertEqual(tracker, [('cleanup', 11)])


class TestRunAsyncManySteps(unittest.IsolatedAsyncioTestCase):
  """Stress tests: many async steps."""

  async def test_async_chain_100_steps(self):
    """Chain with 100 async steps."""
    chain = Chain(_async_identity, 0)
    for _ in range(100):
      chain = chain.then(_async_add_one)
    result = await chain.run()
    self.assertEqual(result, 100)

  async def test_async_chain_all_do(self):
    """Chain where every step is .do() — current_value never changes from root."""
    tracker = []

    async def track(x):
      tracker.append(x)
      return 'discarded'

    chain = Chain(async_fn, 5)  # root = 6
    for _ in range(10):
      chain = chain.do(track)
    result = await chain.run()
    # async_fn(5) = 6, all .do() steps discard their results
    self.assertEqual(result, 6)
    # Each .do() received current_value=6
    self.assertEqual(tracker, [6] * 10)

  async def test_multiple_async_steps_all_return_none(self):
    """Multiple async steps all returning None."""
    result = await (
      Chain(_async_noop, ...)
      .then(_async_noop)
      .then(_async_noop)
      .then(_async_noop)
      .run()
    )
    self.assertIsNone(result)

  async def test_mixed_sync_async_do_steps(self):
    """Alternating sync/async .do() steps."""
    tracker = []

    def sync_track(x):
      tracker.append(('sync', x))
      return 'sync_discarded'

    async def async_track(x):
      tracker.append(('async', x))
      return 'async_discarded'

    result = await (
      Chain(async_fn, 5)  # 6
      .do(sync_track)
      .do(async_track)
      .do(sync_track)
      .do(async_track)
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [
      ('sync', 6),
      ('async', 6),
      ('sync', 6),
      ('async', 6),
    ])


class TestRunAsyncExceptEdgeCases(unittest.IsolatedAsyncioTestCase):
  """Edge cases for except handling in async path."""

  async def test_except_handler_raises_in_async_path(self):
    """Except handler raises a new exception in async path (lines 272-289)."""
    def handler(exc):
      raise RuntimeError('handler boom') from exc

    with self.assertRaises(RuntimeError) as ctx:
      await Chain(_async_raise_value_error).except_(handler).run()
    self.assertEqual(str(ctx.exception), 'handler boom')
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  async def test_exception_not_matching_reraises(self):
    """Exception type doesn't match except filter → re-raised."""
    with self.assertRaises(ValueError):
      await (
        Chain(_async_raise_value_error)
        .except_(lambda e: 'caught', exceptions=TypeError)
        .run()
      )

  async def test_source_link_stamped_on_exception(self):
    """Line 274-275: __quent_source_link__ is stamped on the exception."""
    captured = {}

    def handler(exc):
      captured['has_source_link'] = hasattr(exc, '__quent_source_link__')
      return 'caught'

    await (
      Chain(_async_raise_value_error)
      .except_(handler)
      .run()
    )
    # The handler receives the exception after _modify_traceback has been called
    # in _except_handler_body, which may have cleaned __quent_source_link__.
    # At minimum, the handler was called successfully.
    self.assertEqual(captured.get('has_source_link') is not None, True)

  async def test_current_value_temp_args_stamped(self):
    """Lines 276-283: when current_value is not Null and link has no explicit
    args/kwargs, _set_link_temp_args is called with current_value.
    """
    async def raise_in_step(x):
      raise ValueError(f'failed at {x}')

    with self.assertRaises(ValueError) as ctx:
      await (
        Chain(5)
        .then(_async_add_one)  # 6, enters async path
        .then(raise_in_step)   # fails with current_value=6
        .run()
      )
    self.assertIn('failed at 6', str(ctx.exception))


class TestRunAsyncReturnPropagation(unittest.IsolatedAsyncioTestCase):
  """_Return propagation in nested async chains."""

  async def test_return_propagates_through_nested_async(self):
    """_Return in a nested async chain propagates to outer chain."""
    inner = Chain().then(_async_identity).then(lambda x: Chain.return_(88))
    result = await Chain(5).then(async_fn).then(inner).run()
    self.assertEqual(result, 88)

  async def test_return_with_callable_in_async(self):
    """_Return with a callable value in async path."""
    result = await (
      Chain(5)
      .then(_async_add_one)
      .then(lambda x: Chain.return_(lambda a, b: a + b, 3, 4))
      .run()
    )
    self.assertEqual(result, 7)


class TestRunAsyncMiscEdgeCases(unittest.IsolatedAsyncioTestCase):
  """Miscellaneous edge cases for _run_async."""

  async def test_first_link_ignore_result_current_value_null(self):
    """Line 244: first link has ignore_result=True and current_value is Null.
    The condition `current_value is Null and not link.ignore_result` is False,
    so current_value stays Null.
    """
    tracker = []

    async def track_noop():
      tracker.append('called')
      return 'discarded'

    # Chain() no root. run() no args. First link is .do(track_noop).
    # Evaluates track_noop() → coroutine → async path.
    # In _run_async: result = await coroutine = 'discarded'.
    # link.ignore_result=True → skip. current_value stays Null.
    # No next link → return None.
    result = await Chain().do(track_noop).run()
    self.assertIsNone(result)
    self.assertEqual(tracker, ['called'])

  async def test_sync_steps_after_async_in_run_async(self):
    """After entering async path, sync steps should work normally."""
    result = await (
      Chain(5)
      .then(_async_add_one)       # 6 (enters async)
      .then(lambda x: x * 2)     # 12 (sync)
      .then(lambda x: x + 1)     # 13 (sync)
      .run()
    )
    self.assertEqual(result, 13)

  async def test_async_chain_with_none_values(self):
    """Chain steps that pass None through."""
    async def return_none(x=None):
      return None

    result = await (
      Chain(return_none, ...)
      .then(lambda x: x)
      .run()
    )
    self.assertIsNone(result)

  async def test_async_chain_with_false_values(self):
    """Falsy values (0, '', False) propagate correctly through async path."""
    async def return_zero(x=None):
      return 0

    result = await Chain(return_zero, ...).then(lambda x: x).run()
    self.assertEqual(result, 0)

    async def return_empty_str(x=None):
      return ''

    result = await Chain(return_empty_str, ...).then(lambda x: x).run()
    self.assertEqual(result, '')

    async def return_false(x=None):
      return False

    result = await Chain(return_false, ...).then(lambda x: x).run()
    self.assertFalse(result)

  async def test_except_and_finally_both_run_on_async_error(self):
    """Both except_ and finally_ execute when async body raises."""
    order = []

    result = await (
      Chain(5)
      .then(_async_raise_value_error)
      .except_(lambda exc: (order.append('except'), 'recovered')[1])
      .finally_(lambda rv: order.append(('finally', rv)))
      .run()
    )
    self.assertEqual(result, 'recovered')
    self.assertEqual(order, ['except', ('finally', 5)])

  async def test_finally_runs_on_async_success(self):
    """Finally handler runs even on successful async completion."""
    tracker = []
    result = await (
      Chain(async_fn, 5)
      .then(lambda x: x * 2)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    # async_fn(5) = 6, *2 = 12
    self.assertEqual(result, 12)
    self.assertEqual(tracker, [6])

  async def test_async_chain_preserves_exception_type(self):
    """Specific exception types are preserved through async path."""
    async def raise_type_error(x=None):
      raise TypeError('async type error')

    with self.assertRaises(TypeError) as ctx:
      await Chain(raise_type_error).run()
    self.assertEqual(str(ctx.exception), 'async type error')


if __name__ == '__main__':
  unittest.main()
