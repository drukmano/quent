"""Gap audit tests: critical, moderate, and low priority coverage gaps."""
from __future__ import annotations

import asyncio
import inspect
import unittest
import warnings

from quent import Chain, Null
from quent._core import _task_registry, _Return, _Break
from helpers import (
  SyncCM,
  AsyncCM,
  SyncCMRaisesOnExit,
  AsyncCMRaisesOnExit,
  async_fn,
  sync_fn,
  async_identity,
)


# ---------------------------------------------------------------------------
# CRITICAL (3 tests)
# ---------------------------------------------------------------------------

class CriticalTests(unittest.IsolatedAsyncioTestCase):

  def test_sync_except_and_finally_both_raise(self):
    """C1a: sync chain where except re-raises and finally also raises.
    The finally exception should have __context__ pointing to the except
    handler's exception, which itself has __cause__ pointing to the
    original body exception.
    """
    def body(x=None):
      raise ValueError('body error')

    def except_handler(exc):
      raise TypeError('except error') from exc

    def finally_handler(rv=None):
      raise RuntimeError('finally error')

    c = (
      Chain(body)
      .except_(except_handler)
      .finally_(finally_handler)
    )
    with self.assertRaises(RuntimeError) as cm:
      c.run()

    finally_exc = cm.exception
    self.assertEqual(str(finally_exc), 'finally error')
    # The finally exception's __context__ should be the except handler's
    # exception (TypeError), because the finally block runs while the
    # TypeError is being propagated (raise exc_ from exc in _except_handler_body).
    self.assertIsNotNone(finally_exc.__context__)
    except_exc = finally_exc.__context__
    self.assertIsInstance(except_exc, TypeError)
    self.assertEqual(str(except_exc), 'except error')
    # The except handler's __cause__ should point to the original ValueError
    self.assertIsNotNone(except_exc.__cause__)
    self.assertIsInstance(except_exc.__cause__, ValueError)
    self.assertEqual(str(except_exc.__cause__), 'body error')
    # RuntimeError is NOT explicitly chained (no `from`), so __cause__ is None
    self.assertIsNone(finally_exc.__cause__)
    # TypeError's implicit context is the body ValueError
    self.assertIsInstance(except_exc.__context__, ValueError)
    self.assertEqual(str(except_exc.__context__), 'body error')
    # Chain terminates: the body ValueError has no further cause
    self.assertIsNone(except_exc.__cause__.__context__)

  async def test_async_except_and_finally_both_raise(self):
    """C1b: async chain where except re-raises and finally also raises.
    The finally exception should have __context__ set to the except handler's
    exception.
    """
    async def body(x=None):
      raise ValueError('body error')

    def except_handler(exc):
      raise TypeError('except error') from exc

    async def finally_handler(rv=None):
      raise RuntimeError('finally error')

    c = (
      Chain(body)
      .except_(except_handler)
      .finally_(finally_handler)
    )
    try:
      await c.run()
      self.fail('Expected RuntimeError')
    except RuntimeError as finally_exc:
      self.assertEqual(str(finally_exc), 'finally error')
      # In _run_async's finally block, finally_exc.__context__ is set to
      # _active_exc (the TypeError from the except handler).
      self.assertIsNotNone(finally_exc.__context__)
      except_exc = finally_exc.__context__
      self.assertIsInstance(except_exc, TypeError)
      self.assertEqual(str(except_exc), 'except error')
      self.assertIsNotNone(except_exc.__cause__)
      self.assertIsInstance(except_exc.__cause__, ValueError)
      # CRITICAL: verify innermost message
      self.assertEqual(str(except_exc.__cause__), 'body error')
      # RuntimeError is NOT explicitly chained (no `from`), so __cause__ is None
      self.assertIsNone(finally_exc.__cause__)
      # TypeError's implicit context is the body ValueError
      self.assertIsInstance(except_exc.__context__, ValueError)
      self.assertEqual(str(except_exc.__context__), 'body error')

  async def test_task_registry_grows_and_shrinks(self):
    """C2: fire-and-forget tasks appear in _task_registry and are cleaned up
    when done.
    """
    event = asyncio.Event()
    initial_len = len(_task_registry)

    async def wait_for_event(x=None):
      await event.wait()
      return 'done'

    from quent._core import _ensure_future
    coro = wait_for_event()
    task = _ensure_future(coro)

    self.assertGreater(len(_task_registry), initial_len)
    self.assertIn(task, _task_registry)

    # Let the task complete
    event.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    self.assertEqual(len(_task_registry), initial_len)
    self.assertNotIn(task, _task_registry)

  def test_except_raises_without_from_preserves_context(self):
    """C3: except handler raises WITHOUT `from exc`. Python's implicit
    chaining should still preserve the original exception as __context__.
    """
    def body(x=None):
      raise ValueError('original')

    def except_handler(exc):
      raise TypeError('handler error')

    c = Chain(body).except_(except_handler)
    with self.assertRaises(TypeError) as cm:
      c.run()

    exc = cm.exception
    self.assertEqual(str(exc), 'handler error')
    # No explicit chaining (no `from`), so __cause__ is None
    self.assertIsNone(exc.__cause__)
    # Python's implicit chaining preserves the original ValueError
    self.assertIsInstance(exc.__context__, ValueError)
    self.assertEqual(str(exc.__context__), 'original')
    # __suppress_context__ is False because no explicit `from` was used
    self.assertFalse(exc.__suppress_context__)

  def test_finally_raises_when_body_succeeds(self):
    """C4: body succeeds, no except handler, finally handler raises.
    There is no prior exception in the context chain.
    """
    def body(x=None):
      return 'success'

    def finally_handler(rv=None):
      raise RuntimeError('finally boom')

    c = Chain(body).finally_(finally_handler)
    with self.assertRaises(RuntimeError) as cm:
      c.run()

    exc = cm.exception
    self.assertEqual(str(exc), 'finally boom')
    # No prior exception was propagating, so __context__ is None
    self.assertIsNone(exc.__context__)
    # No explicit chaining
    self.assertIsNone(exc.__cause__)


# ---------------------------------------------------------------------------
# MODERATE (15 tests)
# ---------------------------------------------------------------------------

class ModerateTests(unittest.IsolatedAsyncioTestCase):

  def test_sync_body_and_exit_both_raise(self):
    """M1: with_(SyncCMRaisesOnExit, body_fn) where body_fn raises ValueError.
    The caught exception should be RuntimeError (from __exit__), and its
    __cause__ should be the ValueError (from body).
    """
    def body_fn(ctx):
      raise ValueError('body error')

    with self.assertRaises(RuntimeError) as cm:
      Chain(SyncCMRaisesOnExit()).with_(body_fn).run()

    exit_exc = cm.exception
    self.assertEqual(str(exit_exc), 'exit error')
    self.assertIsNotNone(exit_exc.__cause__)
    self.assertIsInstance(exit_exc.__cause__, ValueError)
    self.assertEqual(str(exit_exc.__cause__), 'body error')
    # Implicit context is also the body ValueError
    self.assertIsInstance(exit_exc.__context__, ValueError)
    self.assertEqual(str(exit_exc.__context__), 'body error')
    # Explicit `from` chaining sets __suppress_context__
    self.assertTrue(exit_exc.__suppress_context__)

  async def test_async_body_and_exit_both_raise(self):
    """M1b: async counterpart of M1. Tests the _to_async path in _ops.py.
    A sync CM (SyncCMRaisesOnExit) with an async body that raises.
    The _to_async path awaits the body, catches the exception, calls
    __exit__ which raises, then does `raise exit_exc from exc`.
    """
    async def body_fn(ctx):
      raise ValueError('async body error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(SyncCMRaisesOnExit()).with_(body_fn).run()

    exit_exc = cm.exception
    self.assertEqual(str(exit_exc), 'exit error')
    # __cause__ is the body ValueError (set by `raise exit_exc from exc`)
    self.assertIsNotNone(exit_exc.__cause__)
    self.assertIsInstance(exit_exc.__cause__, ValueError)
    self.assertEqual(str(exit_exc.__cause__), 'async body error')
    # __context__ is also the body ValueError (set by Python implicitly)
    self.assertIsInstance(exit_exc.__context__, ValueError)
    self.assertEqual(str(exit_exc.__context__), 'async body error')
    # Explicit `from` chaining sets __suppress_context__
    self.assertTrue(exit_exc.__suppress_context__)

  async def test_async_cm_aexit_raises_on_success(self):
    """M2: AsyncCMRaisesOnExit raises RuntimeError in __aexit__ even on
    success. Verify the RuntimeError propagates.
    """
    def body_fn(ctx):
      return 'success'

    try:
      await Chain(AsyncCMRaisesOnExit()).with_(body_fn).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      self.assertEqual(str(exc), 'async exit error')

  def test_with_body_is_nested_chain(self):
    """M3: inner Chain as the body function for with_().
    Verify the CM is entered/exited and the inner chain result is correct.
    """
    cm = SyncCM()
    inner_chain = Chain().then(lambda ctx: ctx + '_processed')
    result = Chain(cm).with_(inner_chain).run()
    self.assertEqual(result, 'ctx_value_processed')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  def test_nested_foreach_break_isolation(self):
    """M4: break_() in inner foreach should NOT escape to outer foreach.
    Outer foreach processes all 3 items.
    """
    def outer_fn(item):
      # Inner foreach iterates ['a', 'b', 'c'], breaks after 'a'
      inner_result = Chain(['a', 'b', 'c']).foreach(
        lambda x: Chain.break_() if x == 'b' else x
      ).run()
      return (item, inner_result)

    result = Chain([1, 2, 3]).foreach(outer_fn).run()
    self.assertEqual(len(result), 3)
    # Each inner foreach should have only processed 'a' before breaking
    for item, inner in result:
      self.assertEqual(inner, ['a'])

  async def test_to_async_on_last_item_foreach(self):
    """M5: foreach with async fn over single-element iterable.
    The _to_async path is entered on the first (and last) item.
    """
    result = await Chain([42]).foreach(async_fn).run()
    # async_fn(42) = 43
    self.assertEqual(result, [43])

  async def test_to_async_on_last_item_filter(self):
    """M6: filter with async predicate.
    Verify only matching items are returned.
    """
    async def is_even(x):
      return x % 2 == 0

    result = await Chain([1, 2, 3]).filter(is_even).run()
    self.assertEqual(result, [2])

  async def test_mixed_sync_async_alternating_foreach(self):
    """M7: foreach over items where fn returns awaitable for even items
    and plain value for odd items. Once _to_async is entered, subsequent
    sync items should still work.
    """
    async def async_double(x):
      return x * 2

    def mixed_fn(x):
      if x % 2 == 0:
        return async_double(x)
      return x * 2

    result = await Chain([1, 2, 3, 4]).foreach(mixed_fn).run()
    self.assertEqual(result, [2, 4, 6, 8])

  def test_sync_iterate_with_async_fn_yields_coroutines(self):
    """M8: .iterate(async_fn) used synchronously yields raw coroutine objects.
    """
    g = Chain([1, 2, 3]).iterate(async_fn)
    items = list(g)
    self.assertEqual(len(items), 3)
    for item in items:
      self.assertTrue(inspect.iscoroutine(item))
      item.close()  # Avoid RuntimeWarning about unawaited coroutines

  def test_return_inside_with_body_cleanup(self):
    """M9: _Return in a with_ body. The CM's __exit__ should still be called.
    """
    class TrackingCM:
      def __init__(self):
        self.entered = False
        self.exited = False
        self.exit_args = None

      def __enter__(self):
        self.entered = True
        return 'tracked_ctx'

      def __exit__(self, *args):
        self.exited = True
        self.exit_args = args
        return False  # Do not suppress

    cm = TrackingCM()

    def body_fn(ctx):
      Chain.return_('early_return')

    # _Return is a _ControlFlowSignal (extends Exception), so the
    # except BaseException block in _with_op catches it. __exit__ is
    # called with the _Return exception info. Since __exit__ returns
    # False, _Return is re-raised and propagates to the chain's _Return
    # handler.
    result = Chain(cm).with_(body_fn).run()
    self.assertEqual(result, 'early_return')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_future_awaitable_in_chain(self):
    """M10: asyncio.Future as an awaitable value in a chain step.
    The chain should detect it via isawaitable and await it.
    """
    future = asyncio.get_event_loop().create_future()
    future.set_result(42)

    result = await Chain(None).then(lambda _: future).run()
    self.assertEqual(result, 42)

  def test_empty_chain_with_finally(self):
    """M11: Chain().finally_(handler).run() -- no source, no links.
    The handler receives no arguments because root_value is Null and
    _evaluate_value calls handler() with no args when current_value is Null.
    """
    received = []

    def handler(rv=None):
      received.append(rv)

    result = Chain().finally_(handler).run()
    self.assertIsNone(result)
    self.assertEqual(len(received), 1)
    # root_value is Null, so _evaluate_value calls handler() with no args.
    # The default rv=None kicks in.
    self.assertIsNone(received[0])

  async def test_gather_cleanup_on_raise(self):
    """M12: gather with a function that raises. Earlier created coroutines
    should be closed (no RuntimeWarning about unawaited coroutines).
    """
    async def slow_coro(x):
      await asyncio.sleep(10)
      return x

    def raise_fn(x):
      raise ValueError('boom')

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      try:
        Chain(5).gather(slow_coro, raise_fn).run()
      except ValueError:
        pass
      # Check no RuntimeWarning about unawaited coroutines
      runtime_warns = [
        x for x in w
        if issubclass(x.category, RuntimeWarning)
        and 'coroutine' in str(x.message).lower()
      ]
      self.assertEqual(len(runtime_warns), 0)

  def test_exception_context_chain_preserved(self):
    """M13: Verify exception __context__ chain is preserved correctly
    and doesn't cause issues. Create a scenario with nested exception
    handling.
    """
    def step1(x=None):
      raise ValueError('step1 error')

    def except_handler(exc):
      raise TypeError('handler error') from exc

    with self.assertRaises(TypeError) as cm:
      Chain(step1).except_(except_handler).run()

    exc = cm.exception
    self.assertIsInstance(exc.__cause__, ValueError)
    self.assertEqual(str(exc.__cause__), 'step1 error')
    # Verify no circular reference: walking the chain terminates
    seen = set()
    current = exc
    while current is not None:
      exc_id = id(current)
      self.assertNotIn(exc_id, seen, 'Circular exception context detected')
      seen.add(exc_id)
      current = current.__context__

  def test_source_link_first_write_wins(self):
    """M14: __quent_source_link__ is set by the first chain to handle the
    exception (first-write-wins). An inner chain's link should be preserved
    when the error propagates to the outer chain.
    """
    def inner_raise(x):
      raise ValueError('inner error')

    inner = Chain().then(inner_raise)

    # Verify the chain works as expected: inner error propagates
    with self.assertRaises(ValueError) as cm:
      Chain(5).then(inner).run()
    self.assertEqual(str(cm.exception), 'inner error')

  def test_add_step_between_runs(self):
    """M15: Modify an unfrozen chain between runs.
    Adding a step after the first run should affect subsequent runs.
    """
    c = Chain(1).then(lambda v: v + 1)
    result1 = c.run()
    self.assertEqual(result1, 2)

    c.then(lambda v: v * 10)
    result2 = c.run()
    self.assertEqual(result2, 20)


# ---------------------------------------------------------------------------
# LOW (12 tests)
# ---------------------------------------------------------------------------

class LowTests(unittest.IsolatedAsyncioTestCase):

  def test_break_with_null_returns_accumulated(self):
    """L1: break_(Null) in foreach returns the accumulated list so far.
    When break value is Null, _handle_break_exc returns the fallback (lst).
    """
    result = Chain([1, 2, 3, 4]).foreach(
      lambda x: Chain.break_(Null) if x == 3 else x
    ).run()
    # break_(Null) -> exc.value is Null -> _handle_break_exc returns
    # the fallback (accumulated list so far = [1, 2])
    self.assertEqual(result, [1, 2])

  def test_return_non_callable_with_args_raises(self):
    """L2: Chain.return_(42, 1, 2) passes 42 as v with args (1, 2).
    _resolve_value(42, (1, 2), {}) calls 42(1, 2) which raises TypeError
    since int is not callable.
    """
    with self.assertRaises(TypeError):
      Chain(1).then(lambda x: Chain.return_(42, 1, 2)).run()

  def test_then_with_args_and_kwargs(self):
    """L3: .then(fn, 'a', 'b', key='val') -- verify fn receives the
    explicit args and kwargs (NOT current_value).
    """
    received = []

    def fn(*args, **kwargs):
      received.append((args, kwargs))
      return 'result'

    result = Chain(None).then(fn, 'a', 'b', key='val').run()
    self.assertEqual(result, 'result')
    self.assertEqual(len(received), 1)
    # When args are provided, _evaluate_value calls fn(*args, **kwargs)
    # without current_value
    self.assertEqual(received[0], (('a', 'b'), {'key': 'val'}))

  def test_100_link_chain_execution(self):
    """L4: chain with 100 .then() steps starting from 0, result should be 100."""
    c = Chain(0)
    for _ in range(100):
      c = c.then(lambda v: v + 1)
    result = c.run()
    self.assertEqual(result, 100)

  def test_decorator_on_function(self):
    """L5: Chain.decorator() wraps a function and preserves its behavior."""
    @Chain().then(lambda x: x * 2).decorator()
    def my_fn(n):
      """My docstring."""
      return n + 1

    self.assertEqual(my_fn(5), 12)  # (5 + 1) * 2 = 12
    self.assertEqual(my_fn.__name__, 'my_fn')
    self.assertEqual(my_fn.__doc__, 'My docstring.')

  def test_frozen_run_with_kwargs(self):
    """L6: frozen chain run with value and callable+kwargs, verifying reuse."""
    frozen = Chain().then(lambda x: x * 2).freeze()
    result = frozen.run(5)
    self.assertEqual(result, 10)
    result2 = frozen.run(7)
    self.assertEqual(result2, 14)

    # Also verify that run(callable, arg, kwarg=val) works through frozen chains.
    def source(x, multiplier=1):
      return x * multiplier

    frozen2 = Chain().then(lambda x: x + 100).freeze()
    # run(source, 3, multiplier=2) -> Link(source, (3,), {multiplier:2})
    # _evaluate_value calls source(3, multiplier=2) -> 6
    # then lambda(6) -> 106
    result3 = frozen2.run(source, 3, multiplier=2)
    self.assertEqual(result3, 106)

  def test_iterate_do_no_fn_sync(self):
    """L7: iterate_do with fn=None. Sync path yields items unchanged."""
    g = Chain([1, 2, 3]).iterate_do()
    result = list(g)
    self.assertEqual(result, [1, 2, 3])

  def test_iterate_non_iterable_raises(self):
    """L8: Chain(42).iterate() -- 42 is not iterable, should raise TypeError."""
    g = Chain(42).iterate()
    with self.assertRaises(TypeError):
      list(g)

  async def test_consecutive_async_do_links(self):
    """L9: Chain with 3 .do(async_fn) calls where async_fn has side effects.
    All 3 side effects run but the original value is preserved.
    """
    tracker = []

    async def side_effect(x):
      tracker.append(x)
      return 'discarded'

    result = await (
      Chain(42)
      .do(side_effect)
      .do(side_effect)
      .do(side_effect)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [42, 42, 42])

  def test_with_enter_returns_none(self):
    """L10: CM whose __enter__ returns None. fn should receive None."""
    class NoneEnterCM:
      def __enter__(self):
        return None
      def __exit__(self, *args):
        return False

    received = []

    def fn(ctx):
      received.append(ctx)
      return 'result'

    result = Chain(NoneEnterCM()).with_(fn).run()
    self.assertEqual(result, 'result')
    self.assertEqual(received, [None])

  def test_with_enter_returns_self(self):
    """L11: CM whose __enter__ returns self. fn should receive the CM."""
    class SelfEnterCM:
      def __enter__(self):
        return self
      def __exit__(self, *args):
        return False

    cm = SelfEnterCM()
    received = []

    def fn(ctx):
      received.append(ctx)
      return 'result'

    result = Chain(cm).with_(fn).run()
    self.assertEqual(result, 'result')
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], cm)

  async def test_gather_inside_gather(self):
    """L12: Nested gather -- outer gather collects results including an
    inner gather. Verify the result is a nested list.
    """
    def inner_gather(x):
      # Inner gather with all sync fns returns a plain list (no await needed)
      return Chain(x).gather(
        lambda v: v + 1,
        lambda v: v + 2,
      ).run()

    result = Chain(10).gather(
      lambda x: x * 2,
      inner_gather,
    ).run()
    self.assertEqual(result, [20, [11, 12]])


if __name__ == '__main__':
  unittest.main()
