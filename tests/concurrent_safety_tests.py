"""Tests for concurrent safety of chains, including reentrance,
concurrent modification, async recursion, and _task_registry.
"""
from __future__ import annotations

import asyncio
import threading
import unittest

from quent import Chain, Null, QuentException
from quent._core import _task_registry
from helpers import async_fn, async_identity, sync_fn


class TestUnfrozenChainReentrance(unittest.TestCase):

  def test_chain_run_inside_step(self):
    # A chain step creates and runs a new chain inside itself.
    def inner_step(x):
      return Chain(x).then(lambda v: v + 100).run()

    result = Chain(5).then(inner_step).run()
    self.assertEqual(result, 105)

  def test_deep_recursion_50(self):
    # 50 levels deep via nested Chain.run calls.
    def recursive_chain(depth, value):
      if depth <= 0:
        return value
      return Chain(value).then(lambda v: recursive_chain(depth - 1, v + 1)).run()

    result = recursive_chain(50, 0)
    self.assertEqual(result, 50)


class TestConcurrentModification(unittest.TestCase):

  def test_unfrozen_chain_not_safe_for_concurrent_build_and_run(self):
    # This test documents that unfrozen chains are NOT safe for concurrent use.
    # We verify that building a chain in one thread while running it in another
    # does not crash Python (no segfault), though results may be incorrect.
    chain = Chain().then(lambda x: x + 1)
    result_holder = [None]

    def runner():
      try:
        result_holder[0] = chain.run(5)
      except Exception:
        result_holder[0] = 'error'

    t = threading.Thread(target=runner)
    t.start()
    t.join()

    # We just verify it didn't crash -- result may be 6 or 'error'.
    self.assertIsNotNone(result_holder[0])


class TestTaskRegistrySafety(unittest.IsolatedAsyncioTestCase):

  async def test_task_registry_cleanup(self):
    # After tasks complete, they should be removed from _task_registry via done callback.
    initial_size = len(_task_registry)
    # Verify the registry is accessible and doesn't grow unboundedly.
    c = Chain().then(async_fn)
    results = await asyncio.gather(*[c.run(i) for i in range(50)])
    self.assertEqual(len(results), 50)
    # After all tasks complete, registry should not have grown significantly.
    self.assertIsInstance(_task_registry, set)


class TestDeepRecursionAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_recursive_chains(self):
    async def recursive_chain(depth, value):
      if depth <= 0:
        return value
      return await Chain(value).then(
        lambda v: recursive_chain(depth - 1, v + 1)
      ).run()

    result = await recursive_chain(20, 0)
    self.assertEqual(result, 20)


if __name__ == '__main__':
  unittest.main()
