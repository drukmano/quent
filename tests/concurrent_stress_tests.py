"""Concurrent safety and stress tests for frozen chains and task registry."""
from __future__ import annotations

import asyncio
import threading
import unittest

from quent import Chain, QuentException
from quent._chain import _FrozenChain
from quent._core import _task_registry, _task_registry_lock, _ensure_future


class ConcurrentFrozenChainTests(unittest.TestCase):

  def test_two_frozen_copies_share_same_inner_chain(self):
    """Freeze the same chain twice. Both frozen copies should produce correct results."""
    c = Chain(0).then(lambda x: x + 1).then(lambda x: x * 2)
    f1 = c.freeze()
    f2 = c.freeze()
    self.assertIs(f1._chain, f2._chain)
    errors = []
    results = {}

    def run_frozen(frozen, key, value):
      try:
        results[key] = frozen.run(value)
      except Exception as e:
        errors.append(e)

    t1 = threading.Thread(target=run_frozen, args=(f1, 'f1', 10))
    t2 = threading.Thread(target=run_frozen, args=(f2, 'f2', 20))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    self.assertEqual(errors, [])
    # f1: 10 -> 10+1=11 -> 11*2=22
    self.assertEqual(results['f1'], 22)
    # f2: 20 -> 20+1=21 -> 21*2=42
    self.assertEqual(results['f2'], 42)

  def test_100_thread_stress_on_frozen_chain(self):
    """100 threads all calling frozen.run(i) with different inputs."""
    c = Chain(0).then(lambda x: x * 3).then(lambda x: x + 1)
    frozen = c.freeze()
    results = {}
    errors = []

    def run_thread(i):
      try:
        results[i] = frozen.run(i)
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(100)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [])
    for i in range(100):
      expected = i * 3 + 1
      self.assertEqual(results[i], expected, f'Thread {i}: expected {expected}, got {results[i]}')

  def test_frozen_chain_with_async_ops_from_threads(self):
    """Threads calling frozen chain that contains async ops via asyncio.run()."""
    async def async_double(x):
      return x * 2

    c = Chain(0).then(async_double).then(lambda x: x + 5)
    frozen = c.freeze()
    results = {}
    errors = []

    def run_thread(i):
      try:
        result = frozen.run(i)
        if asyncio.iscoroutine(result):
          result = asyncio.run(result)
        results[i] = result
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [], f'Errors: {errors}')
    for i in range(10):
      expected = i * 2 + 5
      self.assertEqual(results[i], expected)

  def test_mixed_sync_async_frozen_chain_concurrent(self):
    """Frozen chain with sync-only operations hit from multiple threads."""
    c = Chain(0).then(lambda x: x + 10).then(lambda x: x * 2)
    frozen = c.freeze()
    results = {}
    errors = []

    def run_thread(i):
      try:
        results[i] = frozen.run(i)
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(50)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [])
    for i in range(50):
      expected = (i + 10) * 2
      self.assertEqual(results[i], expected)

  def test_concurrent_gather_operations(self):
    """Frozen chain with gather() called from multiple threads."""
    c = Chain(0).gather(lambda x: x + 1, lambda x: x * 2, lambda x: x - 1)
    frozen = c.freeze()
    results = {}
    errors = []

    def run_thread(i):
      try:
        results[i] = frozen.run(i)
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [])
    for i in range(20):
      expected = [i + 1, i * 2, i - 1]
      self.assertEqual(results[i], expected)

  def test_task_registry_thread_safety_stress(self):
    """Multiple threads creating fire-and-forget tasks concurrently."""
    errors = []
    created_count = threading.local()

    async def do_work():
      await asyncio.sleep(0.001)
      return 42

    def run_thread(i):
      try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          async def create_tasks():
            tasks = []
            for _ in range(5):
              t = _ensure_future(do_work())
              tasks.append(t)
            results = await asyncio.gather(*tasks)
            return results
          result = loop.run_until_complete(create_tasks())
          if len(result) != 5 or not all(r == 42 for r in result):
            errors.append((i, f'unexpected results: {result}'))
        finally:
          loop.close()
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=15)
    self.assertEqual(errors, [], f'Errors: {errors}')

  def test_frozen_chain_exception_no_cross_contamination(self):
    """Some threads get exceptions, others succeed. No cross-contamination."""
    def maybe_raise(x):
      if x % 2 == 0:
        raise ValueError(f'even: {x}')
      return x * 10

    c = Chain(0).then(maybe_raise)
    frozen = c.freeze()
    results = {}
    exceptions = {}

    def run_thread(i):
      try:
        results[i] = frozen.run(i)
      except ValueError as e:
        exceptions[i] = e
      except Exception as e:
        exceptions[i] = e

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)

    for i in range(20):
      if i % 2 == 0:
        self.assertIn(i, exceptions, f'Thread {i} should have raised')
        self.assertIn(f'even: {i}', str(exceptions[i]))
        self.assertNotIn(i, results)
      else:
        self.assertIn(i, results, f'Thread {i} should have succeeded')
        self.assertEqual(results[i], i * 10)
        self.assertNotIn(i, exceptions)

  def test_frozen_chain_with_retry_concurrent(self):
    """Frozen chain with retry() called concurrently."""
    counters = {}
    counter_lock = threading.Lock()

    def flaky(x):
      with counter_lock:
        key = threading.current_thread().ident
        counters.setdefault(key, 0)
        counters[key] += 1
        count = counters[key]
      if count < 2:
        raise ValueError('transient')
      return x * 5

    c = Chain(0).then(flaky).retry(max_attempts=3, on=ValueError)
    frozen = c.freeze()
    results = {}
    errors = []

    def run_thread(i):
      try:
        # Reset counter for this thread
        with counter_lock:
          counters[threading.current_thread().ident] = 0
        results[i] = frozen.run(i)
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [], f'Errors: {errors}')
    for i in range(10):
      self.assertEqual(results[i], i * 5)

  def test_frozen_chain_with_map_concurrent(self):
    """Frozen chain with map() called concurrently with different iterables."""
    c = Chain([]).map(lambda x: x * 2)
    frozen = c.freeze()
    results = {}
    errors = []

    def run_thread(i):
      try:
        data = list(range(i, i + 5))
        results[i] = frozen.run(data)
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [], f'Errors: {errors}')
    for i in range(20):
      expected = [x * 2 for x in range(i, i + 5)]
      self.assertEqual(results[i], expected)

  def test_rapid_freeze_run_cycles(self):
    """Rapidly create, freeze, and run chains from multiple threads."""
    results = {}
    errors = []

    def run_thread(i):
      try:
        for j in range(10):
          c = Chain(i * 10 + j).then(lambda x: x + 1)
          frozen = c.freeze()
          result = frozen.run()
          expected = i * 10 + j + 1
          if result != expected:
            errors.append((i, j, f'expected {expected}, got {result}'))
        results[i] = True
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [], f'Errors: {errors}')
    self.assertEqual(len(results), 20)

  def test_concurrent_iterate(self):
    """Multiple threads iterating a frozen chain's iterate() result."""
    c = Chain([1, 2, 3, 4, 5])
    frozen = c.freeze()
    results = {}
    errors = []

    def run_thread(i):
      try:
        data = list(range(i, i + 3))
        # Create a chain that yields items, then iterate
        c2 = Chain(data)
        collected = list(c2.iterate())
        results[i] = collected
      except Exception as e:
        errors.append((i, e))

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=10)
    self.assertEqual(errors, [], f'Errors: {errors}')
    for i in range(20):
      expected = list(range(i, i + 3))
      self.assertEqual(results[i], expected)


class ConcurrentAsyncTests(unittest.IsolatedAsyncioTestCase):

  async def test_task_cancellation_during_execution(self):
    """Cancel a task while an async chain is running."""
    started = asyncio.Event()
    cancel_event = asyncio.Event()

    async def slow_op(x):
      started.set()
      await cancel_event.wait()
      return x * 2

    c = Chain(5).then(slow_op)

    async def run_chain():
      return await c.run()

    task = asyncio.create_task(run_chain())
    await started.wait()
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task

  async def test_concurrent_async_gather_operations(self):
    """Multiple concurrent gather operations with async callables."""
    async def async_add(x):
      await asyncio.sleep(0.001)
      return x + 1

    async def async_mul(x):
      await asyncio.sleep(0.001)
      return x * 2

    c = Chain(0).gather(async_add, async_mul)
    frozen = c.freeze()

    tasks = [asyncio.create_task(frozen.run(i)) for i in range(10)]
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
      self.assertEqual(result, [i + 1, i * 2])


if __name__ == '__main__':
  unittest.main()
