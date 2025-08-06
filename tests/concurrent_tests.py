"""Tests for concurrent chain execution and chain reuse behavior.

Issue 53: Missing Test — Concurrent Execution of Frozen Chains
Issue 54: Missing Test — Non-Frozen Chain Reuse Behavior

With _ExecCtx per-execution context (Issue N4), each execution gets
isolated state. Frozen chains are safe for concurrent async use,
and non-frozen chain reuse works correctly.
"""
import asyncio
import inspect
from unittest import TestCase, IsolatedAsyncioTestCase
from quent import Chain, Cascade, QuentException


class FrozenChainSequentialReuseTests(TestCase):
  """Issue 53: Verify frozen chains work when called sequentially (no race)."""

  def test_frozen_chain_sequential_sync_reuse(self):
    """Frozen chain (void) with sync function, called with different values."""
    frozen = Chain().then(lambda v: v * 2).freeze()
    results = [frozen(i) for i in range(10)]
    self.assertEqual(results, [i * 2 for i in range(10)])

  def test_frozen_chain_sequential_sync_reuse_with_root(self):
    """Frozen chain with a root value, called without arguments repeatedly."""
    frozen = Chain(5).then(lambda v: v * 2).freeze()
    for _ in range(10):
      self.assertEqual(frozen(), 10)

  def test_frozen_chain_sequential_sync_multi_step(self):
    """Frozen chain with multiple sync steps, reused sequentially."""
    frozen = Chain().then(lambda v: v + 1).then(lambda v: v * 3).freeze()
    results = [frozen(i) for i in range(10)]
    self.assertEqual(results, [(i + 1) * 3 for i in range(10)])

  def test_frozen_chain_sequential_sync_stability(self):
    """Frozen chain returns consistent results over many sequential calls."""
    frozen = Chain().then(lambda v: v ** 2).then(lambda v: v + 1).freeze()
    for _ in range(100):
      self.assertEqual(frozen(5), 26)
      self.assertEqual(frozen(0), 1)
      self.assertEqual(frozen(3), 10)


class FrozenChainConcurrentAsyncTests(IsolatedAsyncioTestCase):
  """Issue 53: Test concurrent async execution of the same frozen chain.

  With _ExecCtx per-execution context (Issue N4), each concurrent
  execution gets isolated state, making frozen chains safe for
  concurrent async use.
  """

  async def test_frozen_chain_concurrent_sync_fn(self):
    """Concurrent async calls to a frozen chain with sync-only functions.

    With sync-only functions, _run completes without yielding to the
    event loop, so no actual interleaving occurs even under gather.
    """
    frozen = Chain().then(lambda v: v * 2).freeze()
    coros = [self._call_frozen_async(frozen, i) for i in range(100)]
    results = await asyncio.gather(*coros)
    expected = [i * 2 for i in range(100)]
    self.assertEqual(sorted(results), sorted(expected))

  async def test_frozen_chain_with_async_fn_concurrent(self):
    """Concurrent async calls to a frozen chain containing an async function.

    With _ExecCtx per-execution context, each concurrent call gets isolated
    state, so frozen chains are safe for concurrent async execution.
    """
    async def async_double(v):
      await asyncio.sleep(0)  # yield to event loop
      return v * 2

    frozen = Chain().then(async_double).freeze()
    n = 100
    coros = [frozen(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    expected = [i * 2 for i in range(n)]
    self.assertEqual(sorted(results), sorted(expected))

  async def test_frozen_chain_with_async_multi_step_concurrent(self):
    """Concurrent calls to frozen chain with multiple async steps.

    With _ExecCtx per-execution context, each concurrent call gets isolated
    state, so multiple async steps work correctly under concurrency.
    """
    async def add_one(v):
      await asyncio.sleep(0)
      return v + 1

    async def multiply_three(v):
      await asyncio.sleep(0)
      return v * 3

    frozen = Chain().then(add_one).then(multiply_three).freeze()
    n = 50
    coros = [frozen(i) for i in range(n)]
    results = await asyncio.gather(*coros)
    expected = [(i + 1) * 3 for i in range(n)]
    self.assertEqual(sorted(results), sorted(expected))

  async def test_frozen_chain_with_root_concurrent(self):
    """Concurrent calls to frozen chain that has a root value (no args).

    Since all calls pass no arguments, there's no temp_root_link
    contention — but link.result is still shared.
    """
    async def async_double(v):
      await asyncio.sleep(0)
      return v * 2

    frozen = Chain(10).then(async_double).freeze()
    n = 50
    coros = [frozen() for _ in range(n)]
    results = await asyncio.gather(*coros)
    # All should be 20
    self.assertEqual(results, [20] * n)

  async def test_frozen_chain_sequential_async_reuse(self):
    """Frozen chain with async function works when called sequentially."""
    async def async_double(v):
      return v * 2

    frozen = Chain().then(async_double).freeze()
    results = []
    for i in range(20):
      result = await frozen(i)
      results.append(result)
    self.assertEqual(results, [i * 2 for i in range(20)])

  async def _call_frozen_async(self, frozen, value):
    """Helper: call a frozen chain, awaiting if the result is a coroutine."""
    result = frozen(value)
    if inspect.iscoroutine(result) or asyncio.isfuture(result):
      return await result
    return result


class NonFrozenChainSequentialReuseTests(TestCase):
  """Issue 54: Test non-frozen chain reuse behavior.

  With _ExecCtx per-execution context, execution state is isolated
  per call. Non-frozen chain reuse works correctly.
  """

  def test_non_frozen_chain_reuse_with_root_sync(self):
    """Run the same chain (with root value) multiple times — sync."""
    c = Chain(1).then(lambda v: v + 1)
    r1 = c.run()
    r2 = c.run()
    r3 = c.run()
    self.assertEqual(r1, 2)
    self.assertEqual(r2, 2)
    self.assertEqual(r3, 2)

  def test_non_frozen_chain_reuse_void_with_different_values(self):
    """Void chain called with different values via .run(v) each time."""
    c = Chain().then(lambda v: v * 10)
    self.assertEqual(c.run(1), 10)
    self.assertEqual(c.run(2), 20)
    self.assertEqual(c.run(3), 30)

  def test_non_frozen_chain_reuse_with_stateful_function(self):
    """Chain with a stateful function, reused multiple times."""
    counter = {"value": 0}
    def increment(v):
      counter["value"] += 1
      return v + counter["value"]

    c = Chain().then(increment)
    r1 = c.run(0)
    r2 = c.run(0)
    r3 = c.run(0)
    # Each call increments counter: 1, 2, 3
    self.assertEqual(r1, 1)
    self.assertEqual(r2, 2)
    self.assertEqual(r3, 3)

  def test_non_frozen_chain_with_root_cannot_override(self):
    """A chain with a root value raises QuentException if called with a value."""
    c = Chain(1).then(lambda v: v + 1)
    self.assertEqual(c.run(), 2)
    with self.assertRaises(QuentException):
      c.run(5)

  def test_non_frozen_chain_multi_step_sync_reuse(self):
    """Void chain with multiple steps, reused with different values."""
    c = Chain().then(lambda v: v + 1).then(lambda v: v * 2).then(lambda v: v - 3)
    self.assertEqual(c.run(0), -1)   # (0+1)*2-3 = -1
    self.assertEqual(c.run(5), 9)    # (5+1)*2-3 = 9
    self.assertEqual(c.run(10), 19)  # (10+1)*2-3 = 19


class NonFrozenChainAsyncReuseTests(IsolatedAsyncioTestCase):
  """Issue 54: Test non-frozen chain reuse with async functions.

  With _ExecCtx per-execution context, execution state is isolated
  per call. Sequential async reuse works correctly.
  """

  async def test_non_frozen_chain_async_sequential_reuse(self):
    """Void chain with async function, called with different values."""
    async def async_add_ten(v):
      return v + 10

    c = Chain().then(async_add_ten)
    r1 = await c.run(1)
    r2 = await c.run(2)
    r3 = await c.run(3)
    self.assertEqual(r1, 11)
    self.assertEqual(r2, 12)
    self.assertEqual(r3, 13)

  async def test_non_frozen_chain_async_with_root_reuse(self):
    """Chain with root value and async function, called repeatedly."""
    async def async_double(v):
      return v * 2

    c = Chain(5).then(async_double)
    r1 = await c.run()
    r2 = await c.run()
    r3 = await c.run()
    self.assertEqual(r1, 10)
    self.assertEqual(r2, 10)
    self.assertEqual(r3, 10)

  async def test_non_frozen_chain_async_multi_step_reuse(self):
    """Void chain with mixed sync/async steps, reused sequentially."""
    async def async_double(v):
      return v * 2

    c = Chain().then(async_double).then(lambda v: v + 1)
    r1 = await c.run(5)
    r2 = await c.run(10)
    r3 = await c.run(0)
    self.assertEqual(r1, 11)   # 5*2+1
    self.assertEqual(r2, 21)   # 10*2+1
    self.assertEqual(r3, 1)    # 0*2+1

  async def test_non_frozen_chain_void_async_reuse(self):
    """Void chain with async-only function, reused sequentially."""
    async def async_square(v):
      return v ** 2

    c = Chain().then(async_square)
    r1 = await c.run(4)
    r2 = await c.run(7)
    r3 = await c.run(3)
    self.assertEqual(r1, 16)
    self.assertEqual(r2, 49)
    self.assertEqual(r3, 9)
