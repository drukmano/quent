"""Concurrent safety and stress tests for task registry."""
from __future__ import annotations

import asyncio
import threading
import unittest

from quent import Chain, QuentException
from quent._core import _task_registry, _task_registry_lock, _ensure_future


class ConcurrentChainTests(unittest.TestCase):

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

    tasks = [asyncio.create_task(c.run(i)) for i in range(10)]
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
      self.assertEqual(result, [i + 1, i * 2])


if __name__ == '__main__':
  unittest.main()
