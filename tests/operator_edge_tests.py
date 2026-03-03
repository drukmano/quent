import time
import asyncio
from unittest import TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null
from quent.quent import _ToThread, _Sleep


# ---------------------------------------------------------------------------
# SleepSyncFallbackTests
# ---------------------------------------------------------------------------
class SleepSyncFallbackTests(TestCase):
  """Tests for _Sleep.time.sleep fallback (line 39 of _operators.pxi).

  By using unittest.TestCase (not IsolatedAsyncioTestCase), there is no
  running event loop, so _asyncio_get_running_loop_internal() returns None
  and the code falls through to time.sleep(self.delay).
  """

  def test_sleep_sync_preserves_value(self):
    """Chain(42).sleep(0.01).run() in sync context returns 42 via time.sleep path."""
    result = Chain(42).sleep(0.01).run()
    self.assertEqual(result, 42)

  def test_sleep_sync_elapsed_time(self):
    """Verify time.sleep is actually called by checking elapsed time."""
    delay = 0.05
    start = time.monotonic()
    Chain(1).sleep(delay).run()
    elapsed = time.monotonic() - start
    self.assertGreaterEqual(elapsed, delay * 0.9)

  def test_sleep_sync_value_flows_through_chain(self):
    """Chain(10).then(fn).sleep(0.01).then(fn2).run() -- value flows correctly."""
    result = (
      Chain(10)
      .then(lambda v: v + 5)
      .sleep(0.001)
      .then(lambda v: v * 2)
      .run()
    )
    self.assertEqual(result, 30)  # (10+5)*2

  def test_sleep_direct_call_sync(self):
    """_Sleep(delay)(current_value) returns None in sync context (time.sleep path)."""
    s = _Sleep(0.001)
    result = s(99)
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# ToThreadSyncFallbackTests
# ---------------------------------------------------------------------------
class ToThreadSyncFallbackTests(TestCase):
  """Tests for _ToThread sync fallback (lines 70-72 of _operators.pxi).

  By using unittest.TestCase (not IsolatedAsyncioTestCase), there is no
  running event loop, so _asyncio_get_running_loop_internal() returns None
  and the code falls through to direct fn() / fn(current_value) calls.
  """

  def test_to_thread_sync_with_value(self):
    """Chain(5).to_thread(fn).run() in sync context calls fn(5) directly (line 72)."""
    result = Chain(5).to_thread(lambda v: v * 2).run()
    self.assertEqual(result, 10)

  def test_to_thread_sync_value_propagates(self):
    """to_thread result propagates as new current value in sync context."""
    result = (
      Chain(3)
      .to_thread(lambda v: v + 7)
      .then(lambda v: v * 2)
      .run()
    )
    self.assertEqual(result, 20)  # (3+7)*2

  def test_to_thread_sync_multiple_in_sequence(self):
    """Multiple to_thread calls in sequence in sync context."""
    result = (
      Chain(2)
      .to_thread(lambda v: v * 3)
      .to_thread(lambda v: v + 4)
      .run()
    )
    self.assertEqual(result, 10)  # 2*3+4

  def test_to_thread_sync_direct_call_with_value(self):
    """_ToThread(fn)(current_value) directly -- line 72."""
    t = _ToThread(lambda v: v * 3)
    result = t(7)
    self.assertEqual(result, 21)

  def test_to_thread_sync_direct_call_with_null(self):
    """_ToThread(fn)(Null) directly -- lines 70-71.

    When current_value is the Null sentinel, _ToThread calls self.fn()
    without any arguments.
    """
    t = _ToThread(lambda: 42)
    result = t(Null)
    self.assertEqual(result, 42)

  def test_to_thread_sync_direct_null_complex_fn(self):
    """_ToThread(fn)(Null) with a function that returns a computed value."""
    state = {'called': False}

    def compute():
      state['called'] = True
      return 100

    t = _ToThread(compute)
    result = t(Null)
    self.assertEqual(result, 100)
    self.assertTrue(state['called'])

  def test_to_thread_sync_fn_returns_none(self):
    """_ToThread sync path where fn returns None."""
    t = _ToThread(lambda v: None)
    result = t(5)
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# ToThreadAsyncNullTests
# ---------------------------------------------------------------------------
class ToThreadAsyncNullTests(MyTestCase):
  """Tests for _ToThread with Null in async context (line 67-68 of _operators.pxi).

  When the event loop IS running and current_value is Null, _ToThread returns
  asyncio.to_thread(self.fn) without any extra argument.
  """

  async def test_to_thread_async_null_direct(self):
    """_ToThread(fn)(Null) in async context returns asyncio.to_thread(fn) (line 68)."""
    t = _ToThread(lambda: 77)
    result = t(Null)
    # In async context, this returns a coroutine
    awaited = await result
    super(MyTestCase, self).assertEqual(awaited, 77)

  async def test_to_thread_async_null_complex_fn(self):
    """_ToThread(fn)(Null) in async context with a function that has side effects."""
    state = {'called': False}

    def compute():
      state['called'] = True
      return 'async_null_result'

    t = _ToThread(compute)
    result = await t(Null)
    super(MyTestCase, self).assertEqual(result, 'async_null_result')
    super(MyTestCase, self).assertTrue(state['called'])

  async def test_to_thread_async_with_value_direct(self):
    """_ToThread(fn)(value) in async context returns asyncio.to_thread(fn, value) (line 69)."""
    t = _ToThread(lambda v: v * 5)
    result = await t(10)
    super(MyTestCase, self).assertEqual(result, 50)


# ---------------------------------------------------------------------------
# EvalSignalValueTests
# ---------------------------------------------------------------------------
class EvalSignalValueTests(MyTestCase):
  """Tests for all _eval_signal_value branches (lines 81-101 of _operators.pxi).

  _eval_signal_value is called from handle_return_exc and handle_break_exc.
  We exercise each branch via Chain.return_() and Chain.break_() within chains.
  """

  # --- return_ paths ---

  async def test_return_literal_value(self):
    """_eval_signal_value line 101: not callable, no args/kwargs -> return v."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, 99).run(), 99
        )

  async def test_return_callable_no_args(self):
    """_eval_signal_value line 98: callable, no args/kwargs -> v()."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, lambda: 77).run(), 77
        )

  async def test_return_callable_with_ellipsis(self):
    """_eval_signal_value line 84: args[0] is ... -> v()."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, lambda: 55, ...).run(), 55
        )

  async def test_return_callable_with_explicit_args(self):
    """_eval_signal_value line 91: args truthy, args[0] is not ...,
    kwargs is {} (not None, not EMPTY_DICT) -> v(*args, **kwargs)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def add(a, b):
          return a + b

        await self.assertEqual(
          Chain(fn).then(Chain.return_, add, 10, 20).run(), 30
        )

  async def test_return_callable_with_args_and_kwargs(self):
    """_eval_signal_value line 91: args truthy + kwargs truthy -> v(*args, **kwargs)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda a, key=None: a + key, 1, key=cv)

        await self.assertEqual(
          Chain(fn, 42).then(trigger).run(), 43
        )

  async def test_return_callable_with_kwargs_only(self):
    """_eval_signal_value line 92-95: no args, kwargs truthy -> v(*args, **kwargs)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda **kw: kw['key'], key=cv)

        await self.assertEqual(
          Chain(fn, 42).then(trigger).run(), 42
        )

  # --- break_ paths (same _eval_signal_value via handle_break_exc) ---

  async def test_break_literal_value(self):
    """_eval_signal_value line 101 via break_: non-callable literal."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x == 2:
            Chain.break_('stopped')
          return fn(x)

        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          'stopped'
        )

  async def test_break_callable_no_args(self):
    """_eval_signal_value line 98 via break_: callable, no args/kwargs -> v()."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x == 2:
            Chain.break_(lambda: 'break_result')
          return fn(x)

        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          'break_result'
        )

  async def test_break_callable_with_ellipsis(self):
    """_eval_signal_value line 84 via break_: args[0] is ... -> v()."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x == 2:
            Chain.break_(lambda: 'ellipsis_break', ...)
          return fn(x)

        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          'ellipsis_break'
        )

  async def test_break_callable_with_explicit_args(self):
    """_eval_signal_value line 91 via break_: args truthy, args[0] is not ..."""
    for fn, ctx in self.with_fn():
      with ctx:
        def add(a, b):
          return a + b

        def f(x):
          if x == 2:
            Chain.break_(add, 10, 20)
          return fn(x)

        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          30
        )

  async def test_break_callable_with_args_and_kwargs(self):
    """_eval_signal_value line 91 via break_: args + kwargs."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x == 2:
            Chain.break_(lambda a, key=None: a + key, 1, key=100)
          return fn(x)

        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          101
        )

  async def test_break_callable_with_kwargs_only(self):
    """_eval_signal_value line 92-95 via break_: kwargs only."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x == 2:
            Chain.break_(lambda **kw: kw['val'], val=77)
          return fn(x)

        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          77
        )

  async def test_break_no_value_returns_fallback(self):
    """handle_break_exc: value is Null -> returns fallback (partial list)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x >= 3:
            Chain.break_()
          return fn(x * 2)

        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(f).run(),
          [2, 4]
        )

  async def test_return_no_value_returns_none(self):
    """handle_return_exc: value is Null -> returns None.

    Chain.return_() with no value raises _Return(Null, (), {}). Since
    value is Null, handle_return_exc returns None.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_()

        await self.assertIsNone(
          Chain(fn, 10).then(trigger).run()
        )

  async def test_return_async_callable(self):
    """_eval_signal_value line 98 with async callable: v() returns a coroutine."""
    async def async_fn():
      return 88

    await self.assertEqual(
      Chain().then(Chain.return_, async_fn).run(), 88
    )

  async def test_return_falsy_literals(self):
    """_eval_signal_value line 101: falsy non-callable values returned as-is."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, 0).run(), 0
        )
        await self.assertEqual(
          Chain(fn).then(Chain.return_, '').run(), ''
        )
        await self.assertEqual(
          Chain(fn).then(Chain.return_, False).run(), False
        )


if __name__ == '__main__':
  import unittest
  unittest.main()
