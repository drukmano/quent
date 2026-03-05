"""Tests for monad laws adapted to Chain semantics.

The three monad laws, expressed in Chain terms:
  Left identity:  Chain(a).then(f).run() == f(a)
  Right identity: Chain(a).then(identity).run() == a
  Associativity:  Chain(a).then(f).then(g).run() == Chain(a).then(lambda x: g(f(x))).run()
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from helpers import sync_fn, async_fn, sync_identity, async_identity


class TestMonadLaws(unittest.TestCase):

  def test_left_identity(self):
    # Chain(a).then(f).run() == f(a)
    a = 5
    result = Chain(a).then(sync_fn).run()
    expected = sync_fn(a)
    self.assertEqual(result, expected)
    self.assertEqual(result, 6)

  def test_right_identity(self):
    # Chain(a).then(identity).run() == a
    a = 5
    result = Chain(a).then(sync_identity).run()
    self.assertEqual(result, a)

  def test_associativity(self):
    # Chain(a).then(f).then(g).run() == g(f(a))
    a = 5
    f = lambda x: x + 1
    g = lambda x: x * 2
    lhs = Chain(a).then(f).then(g).run()
    rhs = Chain(a).then(lambda x: g(f(x))).run()
    self.assertEqual(lhs, rhs)
    self.assertEqual(lhs, 12)

  def test_left_identity_multiple_values(self):
    # Left identity must hold for falsy and edge-case values.
    # 0
    self.assertEqual(
      Chain(0).then(sync_fn).run(),
      sync_fn(0),
    )
    self.assertEqual(Chain(0).then(sync_fn).run(), 1)

    # None
    check_none = lambda x: x is None
    self.assertEqual(
      Chain(None).then(check_none).run(),
      check_none(None),
    )
    self.assertTrue(Chain(None).then(check_none).run())

    # False
    negate = lambda x: not x
    self.assertEqual(
      Chain(False).then(negate).run(),
      negate(False),
    )
    self.assertTrue(Chain(False).then(negate).run())

    # Empty string
    append_a = lambda x: x + 'a'
    self.assertEqual(
      Chain('').then(append_a).run(),
      append_a(''),
    )
    self.assertEqual(Chain('').then(append_a).run(), 'a')

    # Empty list
    append_one = lambda x: x + [1]
    self.assertEqual(
      Chain([]).then(append_one).run(),
      append_one([]),
    )
    self.assertEqual(Chain([]).then(append_one).run(), [1])


class TestMonadLawsAsync(unittest.IsolatedAsyncioTestCase):

  async def test_left_identity_async(self):
    # Chain(a).then(async_f).run() == await async_f(a)
    a = 5
    result = await Chain(a).then(async_fn).run()
    expected = await async_fn(a)
    self.assertEqual(result, expected)
    self.assertEqual(result, 6)

  async def test_right_identity_async(self):
    # Chain(a).then(async_identity).run() == a
    a = 5
    result = await Chain(a).then(async_identity).run()
    self.assertEqual(result, a)

  async def test_associativity_async(self):
    # Chain(a).then(f).then(g).run() == g(await f(a))
    a = 5

    async def f(x):
      return x + 1

    async def g(x):
      return x * 2

    lhs = await Chain(a).then(f).then(g).run()

    async def composed(x):
      return await g(await f(x))

    rhs = await Chain(a).then(composed).run()
    self.assertEqual(lhs, rhs)
    self.assertEqual(lhs, 12)

  async def test_identity_with_falsy_values_async(self):
    # Right identity must hold for falsy and edge-case values in async chains.
    # 0
    self.assertEqual(
      await Chain(0).then(async_identity).run(),
      0,
    )

    # None
    self.assertIsNone(
      await Chain(None).then(async_identity).run(),
    )

    # False
    self.assertFalse(
      await Chain(False).then(async_identity).run(),
    )

    # Empty string
    self.assertEqual(
      await Chain('').then(async_identity).run(),
      '',
    )

    # Empty list
    self.assertEqual(
      await Chain([]).then(async_identity).run(),
      [],
    )


if __name__ == '__main__':
  unittest.main()
