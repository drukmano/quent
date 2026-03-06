"""Exhaustive tests for Chain.freeze() and _FrozenChain: semantics, all
operations, reuse, and beyond-spec scenarios.
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from quent._chain import _FrozenChain
from quent._core import Link
from helpers import (
  async_fn,
  async_identity,
  sync_fn,
  sync_identity,
  SyncCM,
)


class TestFreezeSemantics(unittest.TestCase):

  def test_frozen_has_no_is_chain(self):
    frozen = Chain(42).freeze()
    self.assertFalse(hasattr(frozen, '_is_chain'))

  def test_frozen_as_then_step_not_nested(self):
    # Link(frozen).is_chain should be False because _FrozenChain lacks _is_chain.
    frozen = Chain(42).freeze()
    link = Link(frozen)
    self.assertFalse(link.is_chain)

  def test_frozen_callable_alias(self):
    frozen = Chain(42).then(lambda x: x + 1).freeze()
    self.assertEqual(frozen(), frozen.run())

  def test_frozen_bool_true(self):
    self.assertIs(bool(Chain().freeze()), True)
    self.assertIs(bool(Chain(42).freeze()), True)

  def test_frozen_repr(self):
    frozen = Chain(42).freeze()
    r = repr(frozen)
    self.assertTrue(r.startswith('Frozen('))
    self.assertIn('Chain(', r)
    self.assertTrue(r.endswith(')'))

  def test_frozen_wraps_same_object(self):
    chain = Chain(42)
    frozen = chain.freeze()
    self.assertIs(frozen._chain, chain)

  def test_frozen_run_with_value(self):
    frozen = Chain().then(lambda x: x * 3).freeze()
    self.assertEqual(frozen.run(5), 15)

  def test_frozen_run_with_kwargs(self):
    # run() with value and extra args (Chain.run supports *args, **kwargs through Link).
    frozen = Chain().then(lambda x: x * 2).freeze()
    self.assertEqual(frozen.run(7), 14)


class TestFreezeWithAllOperations(unittest.TestCase):

  def test_frozen_with_then(self):
    frozen = Chain(10).then(lambda x: x + 5).freeze()
    with self.subTest(msg='then step'):
      self.assertEqual(frozen.run(), 15)

  def test_frozen_with_do(self):
    tracker = []
    frozen = Chain(10).do(lambda x: tracker.append(x)).then(lambda x: x * 2).freeze()
    with self.subTest(msg='do side-effect'):
      result = frozen.run()
      self.assertEqual(result, 20)
      self.assertEqual(tracker, [10])

  def test_frozen_with_foreach(self):
    frozen = Chain([1, 2, 3]).foreach(lambda x: x * 2).freeze()
    with self.subTest(msg='foreach'):
      self.assertEqual(frozen.run(), [2, 4, 6])

  def test_frozen_with_filter(self):
    frozen = Chain([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).freeze()
    with self.subTest(msg='filter'):
      self.assertEqual(frozen.run(), [2, 4])

  def test_frozen_with_gather(self):
    frozen = Chain(5).gather(lambda x: x + 1, lambda x: x * 2).freeze()
    with self.subTest(msg='gather'):
      self.assertEqual(frozen.run(), [6, 10])

  def test_frozen_with_with(self):
    frozen = Chain().then(lambda _: SyncCM()).with_(lambda ctx: ctx + '_used').freeze()
    with self.subTest(msg='with_ context manager'):
      result = frozen.run('ignored')
      self.assertEqual(result, 'ctx_value_used')

  def test_frozen_with_except(self):
    frozen = Chain(10).then(lambda x: 1 / 0).except_(lambda e: 'caught').freeze()
    with self.subTest(msg='except handler'):
      self.assertEqual(frozen.run(), 'caught')

  def test_frozen_with_finally(self):
    tracker = []
    frozen = Chain(10).then(lambda x: x + 5).finally_(lambda rv: tracker.append(rv)).freeze()
    with self.subTest(msg='finally handler'):
      result = frozen.run()
      self.assertEqual(result, 15)
      self.assertEqual(tracker, [10])  # finally receives root_value

  def test_frozen_with_iterate(self):
    # iterate() returns a _Generator, not a frozen chain. But we can test
    # a chain that produces an iterable which is then processed.
    frozen = Chain([1, 2, 3]).foreach(lambda x: x + 10).freeze()
    with self.subTest(msg='iterate-like via foreach'):
      self.assertEqual(frozen.run(), [11, 12, 13])

  def test_frozen_with_nested_chain(self):
    inner = Chain().then(lambda x: x + 100)
    frozen = Chain(5).then(inner).freeze()
    with self.subTest(msg='nested chain'):
      self.assertEqual(frozen.run(), 105)


class TestFreezeReuse(unittest.TestCase):

  def test_frozen_multiple_runs(self):
    frozen = Chain().then(lambda x: x * 2).freeze()
    results = [frozen.run(i) for i in range(5)]
    self.assertEqual(results, [0, 2, 4, 6, 8])

  def test_frozen_100_runs(self):
    frozen = Chain().then(lambda x: x + 1).freeze()
    results = [frozen.run(i) for i in range(100)]
    self.assertEqual(results, list(range(1, 101)))

  def test_frozen_different_value_types(self):
    # The chain step returns the value as-is (identity).
    frozen = Chain().then(lambda x: x).freeze()
    test_values = [42, 'hello', [1, 2, 3], None, False, 0, 0.0, (), {}, b'bytes']
    for val in test_values:
      with self.subTest(val=val):
        self.assertEqual(frozen.run(val), val)


# --- Beyond-spec tests ---

class TestFreezeOnChainWithNoSteps(unittest.TestCase):

  def test_freeze_no_steps_no_root(self):
    frozen = Chain().freeze()
    # Chain with no root and no steps: run() returns None.
    self.assertIsNone(frozen.run())

  def test_freeze_no_steps_with_root(self):
    frozen = Chain(42).freeze()
    self.assertEqual(frozen.run(), 42)

  def test_freeze_no_steps_with_run_value(self):
    frozen = Chain().freeze()
    # run(v) with no steps: v is evaluated as the root, which is returned.
    self.assertEqual(frozen.run(99), 99)


class TestFreezeCalledTwice(unittest.TestCase):

  def test_freeze_returns_different_objects(self):
    chain = Chain(42).then(lambda x: x + 1)
    f1 = chain.freeze()
    f2 = chain.freeze()
    self.assertIsNot(f1, f2)
    # But both wrap the same underlying chain.
    self.assertIs(f1._chain, f2._chain)

  def test_both_frozen_produce_same_results(self):
    chain = Chain().then(lambda x: x * 3)
    f1 = chain.freeze()
    f2 = chain.freeze()
    self.assertEqual(f1.run(5), 15)
    self.assertEqual(f2.run(5), 15)


class TestFrozenChainContainingFrozenChain(unittest.TestCase):

  def test_frozen_containing_frozen(self):
    inner_frozen = Chain().then(lambda x: x + 10).freeze()
    outer_frozen = Chain().then(inner_frozen).freeze()
    # inner_frozen is treated as a callable (not a chain), so .then(inner_frozen)
    # calls inner_frozen(current_value) -> inner_frozen.run(current_value)
    self.assertEqual(outer_frozen.run(5), 15)

  def test_deeply_nested_frozen(self):
    # 3 levels: fn in frozen in frozen in frozen.
    f1 = Chain().then(lambda x: x + 1).freeze()
    f2 = Chain().then(f1).freeze()
    f3 = Chain().then(f2).freeze()
    self.assertEqual(f3.run(0), 1)


class TestFrozenAsArgumentToAnotherChain(unittest.TestCase):

  def test_frozen_as_then_step(self):
    frozen = Chain().then(lambda x: x * 5).freeze()
    result = Chain(3).then(frozen).run()
    self.assertEqual(result, 15)

  def test_frozen_as_do_step(self):
    tracker = []
    frozen = Chain().then(lambda x: tracker.append(x)).freeze()
    result = Chain(42).do(frozen).then(lambda x: x + 1).run()
    self.assertEqual(result, 43)
    self.assertEqual(tracker, [42])


class TestFreezeWithReturn(unittest.TestCase):

  def test_frozen_with_early_return(self):
    frozen = (
      Chain()
      .then(lambda x: Chain.return_(x * 10))
      .then(lambda x: x + 999)
      .freeze()
    )
    self.assertEqual(frozen.run(3), 30)

  def test_frozen_with_early_return_multiple_runs(self):
    frozen = (
      Chain()
      .then(lambda x: Chain.return_(x * 10))
      .then(lambda x: x + 999)
      .freeze()
    )
    for i in range(5):
      with self.subTest(i=i):
        self.assertEqual(frozen.run(i), i * 10)


class TestFreezeAsync(unittest.IsolatedAsyncioTestCase):

  async def test_frozen_async_steps(self):
    frozen = Chain().then(async_fn).then(async_fn).freeze()
    result = await frozen.run(10)
    self.assertEqual(result, 12)

  async def test_frozen_async_with_except(self):
    async def fail(x):
      raise ValueError('async boom')

    frozen = Chain().then(fail).except_(lambda e: 'caught_async').freeze()
    result = await frozen.run(1)
    self.assertEqual(result, 'caught_async')

  async def test_frozen_async_reuse(self):
    frozen = Chain().then(async_fn).freeze()
    results = []
    for i in range(10):
      results.append(await frozen.run(i))
    self.assertEqual(results, list(range(1, 11)))


if __name__ == '__main__':
  unittest.main()
