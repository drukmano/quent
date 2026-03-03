"""Tests for autorun=True + ensure_future() code paths and _InternalQuentException wrapping.

Covers:
  1. autorun + async root in _run (non-simple) -> ensure_future (line 132-133)
  2. autorun + async link mid-chain in _run (non-simple) -> ensure_future (line 159-160)
  3. autorun via run() on simple chain -> ensure_future (line 549-550)
  4. autorun via __call__() on simple chain -> ensure_future (line 654-655)
  5. _InternalQuentException wrapping in run() and __call__() (lines 547-548, 652-653)
"""
import asyncio
import inspect
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# AutorunAsyncRootTests
# ---------------------------------------------------------------------------
class AutorunAsyncRootTests(MyTestCase):
  """autorun=True + async root value in _run (non-simple path).

  When the chain is non-simple (_is_simple=False, forced by .do() or .except_()),
  _run() evaluates the root link. If it returns a coroutine, _run_async() is
  called, and the autorun check at lines 132-133 wraps the result in a Task
  via ensure_future().
  """

  async def test_autorun_async_root_returns_task(self):
    """Chain(async_fn).do(side_effect).config(autorun=True).run() returns a Task.

    .do() makes _is_simple=False, so _run's main body is used.
    The async root produces a coro, triggering _run_async, then ensure_future.
    """
    side_effects = {'called': False}

    def side_effect(v):
      side_effects['called'] = True

    result = Chain(aempty, 42).do(side_effect).config(autorun=True).run()
    # Must be a Task, not a bare coroutine
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)
    super(MyTestCase, self).assertTrue(side_effects['called'])

  async def test_autorun_async_root_adds_to_registry(self):
    """ensure_future called from _run line 133 adds the task to task_registry."""
    initial_size = _get_registry_size()

    async def slow_root(v):
      await asyncio.sleep(0.05)
      return v

    task = Chain(slow_root, 10).do(lambda v: None).config(autorun=True).run()
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), initial_size)
    value = await task
    super(MyTestCase, self).assertEqual(value, 10)

  async def test_autorun_async_root_with_except_forces_non_simple(self):
    """.except_() also forces _is_simple=False, same ensure_future path."""
    result = Chain(aempty, 99).except_(lambda v: None).config(autorun=True).run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 99)


# ---------------------------------------------------------------------------
# AutorunAsyncLinkTests
# ---------------------------------------------------------------------------
class AutorunAsyncLinkTests(MyTestCase):
  """autorun=True + async link mid-chain in _run (non-simple path).

  When the chain is non-simple, the root is sync, and a subsequent link returns
  a coroutine, _run transitions to _run_async. The autorun check at lines 159-160
  wraps the result in a Task via ensure_future().
  """

  async def test_autorun_async_link_returns_task(self):
    """Chain(1).do(async_fn).config(autorun=True).run() returns a Task.

    .do() makes _is_simple=False. The sync root (1) evaluates fine,
    then async_fn returns a coro, triggering _run_async + ensure_future.
    """
    async def async_side_effect(v):
      pass

    result = Chain(1).do(async_side_effect).config(autorun=True).run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    # .do() discards its result; the chain's current_value remains 1
    super(MyTestCase, self).assertEqual(value, 1)

  async def test_autorun_async_then_after_sync_root_non_simple(self):
    """Chain(10).do(sync_fn).then(async_fn).config(autorun=True).run() -> Task.

    The sync root and sync .do() evaluate in _run. The async .then() link
    triggers _run_async. autorun wraps with ensure_future at line 159-160.
    """
    async def async_double(v):
      return v * 2

    result = (
      Chain(10)
      .do(lambda v: None)
      .then(async_double)
      .config(autorun=True)
      .run()
    )
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 20)

  async def test_autorun_async_link_adds_to_registry(self):
    """ensure_future from _run line 160 adds the task to task_registry."""
    initial_size = _get_registry_size()

    async def slow_link(v):
      await asyncio.sleep(0.05)
      return v + 5

    task = Chain(10).do(lambda v: None).then(slow_link).config(autorun=True).run()
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), initial_size)
    value = await task
    super(MyTestCase, self).assertEqual(value, 15)


# ---------------------------------------------------------------------------
# AutorunRunTests
# ---------------------------------------------------------------------------
class AutorunRunTests(MyTestCase):
  """autorun=True via run() on simple chain -> ensure_future (lines 549-550).

  For simple chains (_is_simple=True, no .do()/.except_()/.finally_()/debug),
  _run delegates to _run_simple. _run_simple does NOT contain autorun logic.
  When _run_simple encounters an async root or link, it returns a coroutine
  from _run_async_simple. This coroutine flows back through _run (which just
  returns it) to run(). The check at line 549 catches it and wraps with
  ensure_future.
  """

  async def test_autorun_simple_async_root_via_run(self):
    """Chain(aempty, 42).then(sync_fn).config(autorun=True).run() -> Task.

    The chain is simple (only .then() links). _run_simple gets a coro from
    the async root, returns _run_async_simple coroutine. run() wraps it.
    """
    result = Chain(aempty, 42).then(lambda v: v * 2).config(autorun=True).run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 84)

  async def test_autorun_simple_async_link_via_run(self):
    """Chain(10).then(async_fn).config(autorun=True).run() -> Task.

    Sync root, async .then() link on a simple chain. _run_simple encounters
    the coro in the link loop and returns _run_async_simple. run() wraps it.
    """
    async def async_add(v):
      return v + 5

    result = Chain(10).then(async_add).config(autorun=True).run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 15)

  async def test_autorun_simple_chain_task_in_registry(self):
    """run() ensure_future at line 550 adds to task_registry."""
    initial_size = _get_registry_size()

    async def slow_fn(v):
      await asyncio.sleep(0.05)
      return v

    task = Chain(slow_fn, 7).config(autorun=True).run()
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), initial_size)
    value = await task
    super(MyTestCase, self).assertEqual(value, 7)

  async def test_autorun_sync_chain_returns_value_directly(self):
    """When the chain is fully sync, autorun has no effect — result returned directly."""
    result = Chain(10).then(lambda v: v + 1).config(autorun=True).run()
    # Sync result: not a Task, not a coroutine
    super(MyTestCase, self).assertNotIsInstance(result, asyncio.Task)
    super(MyTestCase, self).assertFalse(inspect.isawaitable(result))
    super(MyTestCase, self).assertEqual(result, 11)


# ---------------------------------------------------------------------------
# AutorunCallTests
# ---------------------------------------------------------------------------
class AutorunCallTests(MyTestCase):
  """autorun=True via __call__() -> ensure_future (lines 654-655).

  Same logic as run(), but triggered via chain() instead of chain.run().
  __call__ delegates to _run, and the autorun check at line 654 wraps
  a returned coroutine with ensure_future.
  """

  async def test_autorun_call_simple_async_root(self):
    """chain() on a simple async chain wraps result via __call__ ensure_future."""
    chain = Chain(aempty, 42).then(lambda v: v * 3).config(autorun=True)
    result = chain()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 126)

  async def test_autorun_call_simple_async_link(self):
    """chain() with sync root and async link on a simple chain."""
    async def async_negate(v):
      return -v

    chain = Chain(5).then(async_negate).config(autorun=True)
    result = chain()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, -5)

  async def test_autorun_call_non_simple_async_root(self):
    """chain() on a non-simple chain with async root (hits _run lines 132-133 via __call__)."""
    chain = Chain(aempty, 10).do(lambda v: None).config(autorun=True)
    result = chain()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 10)

  async def test_autorun_call_non_simple_async_link(self):
    """chain() on a non-simple chain with async link (hits _run lines 159-160 via __call__)."""
    async def async_double(v):
      return v * 2

    chain = Chain(7).do(lambda v: None).then(async_double).config(autorun=True)
    result = chain()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 14)

  async def test_autorun_call_sync_returns_directly(self):
    """Fully sync chain via __call__ with autorun — no wrapping, direct return."""
    chain = Chain(3).then(lambda v: v ** 2).config(autorun=True)
    result = chain()
    super(MyTestCase, self).assertNotIsInstance(result, asyncio.Task)
    super(MyTestCase, self).assertFalse(inspect.isawaitable(result))
    super(MyTestCase, self).assertEqual(result, 9)

  async def test_autorun_call_adds_to_registry(self):
    """__call__ ensure_future at line 655 adds to task_registry."""
    initial_size = _get_registry_size()

    async def slow_fn(v):
      await asyncio.sleep(0.05)
      return v

    chain = Chain(slow_fn, 3).config(autorun=True)
    task = chain()
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), initial_size)
    value = await task
    super(MyTestCase, self).assertEqual(value, 3)


# ---------------------------------------------------------------------------
# InternalExceptionWrappingTests
# ---------------------------------------------------------------------------
class InternalExceptionWrappingTests(MyTestCase):
  """Tests for _InternalQuentException wrapping in run() and __call__().

  run() and __call__() both catch _InternalQuentException (base of _Return
  and _Break) and wrap it as QuentException. These guards are defensive:
  in normal execution, _Return and _Break are caught by _run/_run_simple.
  However, there are edge cases where they can escape, for example
  Chain.break_() at top level produces a QuentException through _run_simple.

  We also verify Chain.return_() and Chain.break_() as classmethods raise
  the appropriate control flow exceptions when called directly.
  """

  async def test_break_in_non_nested_chain_raises_quent_exception_via_run(self):
    """Chain().then(Chain.break_).run() -> QuentException from _run_simple.

    Chain.break_() raises _Break. _run_simple catches it, sees is_nested=False,
    raises QuentException('_Break cannot be used in this context.').
    run() does NOT need to catch _InternalQuentException here because _run_simple
    already converted it. We verify the QuentException is raised.
    """
    with self.assertRaises(QuentException) as cm:
      Chain().then(Chain.break_).run()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_in_non_nested_chain_raises_quent_exception_via_call(self):
    """Same as above but via __call__()."""
    with self.assertRaises(QuentException) as cm:
      Chain().then(Chain.break_)()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_return_at_top_level_returns_value_via_run(self):
    """Chain().then(Chain.return_, 42).run() -> 42.

    Chain.return_(42) raises _Return. _run_simple catches it,
    handle_return_exc with propagate=False returns the value.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, 42).run(),
          42
        )

  async def test_return_at_top_level_returns_value_via_call(self):
    """Chain().then(Chain.return_, 42)() -> 42."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, 42)(),
          42
        )

  async def test_return_none_at_top_level(self):
    """Chain.return_() with no value -> None."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(
          Chain(fn).then(Chain.return_).run()
        )

  async def test_break_direct_call_raises(self):
    """Calling Chain.break_() directly raises an exception (it's a control flow signal)."""
    # Chain.break_() raises _Break, which inherits from _InternalQuentException
    # which inherits from Exception. We catch Exception to verify it raises.
    with self.assertRaises(Exception):
      Chain.break_()

  async def test_return_direct_call_raises(self):
    """Calling Chain.return_() directly raises an exception (it's a control flow signal)."""
    with self.assertRaises(Exception):
      Chain.return_()

  async def test_break_in_non_simple_chain_via_run(self):
    """Chain(1).do(Chain.break_).run() -> QuentException.

    .do() forces non-simple path. _run's except _Break catches it,
    is_nested=False -> raises QuentException.
    """
    with self.assertRaises(QuentException) as cm:
      Chain(1).do(Chain.break_).run()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_break_in_non_simple_chain_via_call(self):
    """Same as above but via __call__()."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).do(Chain.break_)()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_return_in_non_simple_chain_returns_value(self):
    """Chain(1).do(lambda v: Chain.return_(99)).run() -> 99.

    .do() forces non-simple path. _run's except _Return catches it,
    handle_return_exc with propagate=False returns the value.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).do(lambda v: Chain.return_(99)).run(),
          99
        )

  async def test_control_flow_in_finally_raises_quent_exception(self):
    """Using return_/break_ inside finally_ raises QuentException.

    The finally block in _run catches _InternalQuentException and
    converts it to QuentException('Using control flow signals inside
    finally handlers is not allowed.').
    """
    with self.assertRaises(QuentException) as cm:
      Chain(1).finally_(Chain.return_).run()
    super(MyTestCase, self).assertIn('control flow signals', str(cm.exception))

    with self.assertRaises(QuentException) as cm:
      Chain(1).finally_(Chain.break_).run()
    super(MyTestCase, self).assertIn('control flow signals', str(cm.exception))
