"""Tests for Chain.freeze() and _FrozenChain: freezing semantics,
delegation, reuse, nested-chain detection bypass, and async behavior.
"""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain, Null, QuentException
from quent._chain import _FrozenChain
from helpers import async_fn, sync_fn


class TestFreezeBasic(unittest.TestCase):

  def test_returns_frozen_chain(self):
    frozen = Chain(42).freeze()
    self.assertIs(type(frozen), _FrozenChain)

  def test_frozen_run_delegates(self):
    result = Chain(42).freeze().run()
    self.assertEqual(result, 42)

  def test_frozen_callable(self):
    result = Chain(42).freeze()()
    self.assertEqual(result, 42)

  def test_frozen_bool_always_true(self):
    self.assertIs(bool(Chain().freeze()), True)

  def test_frozen_repr(self):
    frozen = Chain(42).freeze()
    r = repr(frozen)
    self.assertIn('Frozen(', r)


class TestFrozenNotNested(unittest.TestCase):

  def test_no_is_chain_attribute(self):
    frozen = Chain().freeze()
    self.assertFalse(hasattr(frozen, '_is_chain'))

  def test_as_step_not_detected_as_nested(self):
    # When a frozen chain is added as a step via .then(), the Link
    # constructor checks getattr(v, '_is_chain', False). Since
    # _FrozenChain lacks _is_chain, the link's is_chain must be False.
    from quent._core import Link
    frozen = Chain(42).freeze()
    link = Link(frozen)
    self.assertFalse(link.is_chain)

  def test_as_step_treated_as_callable(self):
    # frozen.__call__(5) -> frozen.run(5) -> inner chain runs with run value 5.
    # run(v) bypasses root_link and starts from first_link, so the inner chain
    # must have a .then() step (not just a root callable) to process the value.
    frozen = Chain().then(lambda x: x + 1).freeze()
    result = Chain(5).then(frozen).run()
    self.assertEqual(result, 6)


class TestFrozenReuse(unittest.TestCase):

  def test_multiple_runs(self):
    frozen = Chain(42).freeze()
    for _ in range(3):
      self.assertEqual(frozen.run(), 42)

  def test_different_values(self):
    frozen = Chain().then(lambda x: x * 2).freeze()
    self.assertEqual(frozen.run(1), 2)
    self.assertEqual(frozen.run(2), 4)

  def test_independent_results(self):
    frozen = Chain().then(lambda x: x + 1).freeze()
    results = [frozen.run(i) for i in range(5)]
    self.assertEqual(results, [1, 2, 3, 4, 5])


class TestFrozenAsync(unittest.IsolatedAsyncioTestCase):

  async def test_frozen_async_chain(self):
    frozen = Chain(5).then(async_fn).freeze()
    result = await frozen.run()
    self.assertEqual(result, 6)

  async def test_frozen_run_with_async_value(self):
    frozen = Chain().then(async_fn).then(async_fn).freeze()
    result = await frozen.run(10)
    self.assertEqual(result, 12)


class TestFrozenConcurrent(unittest.IsolatedAsyncioTestCase):

  async def test_concurrent_frozen_runs(self):
    frozen = Chain().then(async_fn).freeze()
    results = await asyncio.gather(
      frozen.run(1),
      frozen.run(2),
      frozen.run(3),
    )
    self.assertEqual(sorted(results), [2, 3, 4])


if __name__ == '__main__':
  unittest.main()
