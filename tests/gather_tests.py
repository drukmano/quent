"""Tests for Chain.gather(): parallel execution of multiple functions on
the current chain value, with sync, async, mixed, exception, and edge-case
coverage.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import async_fn, sync_fn, async_raise_fn


# ---------------------------------------------------------------------------
# Sync gather
# ---------------------------------------------------------------------------

class TestGatherSync(unittest.TestCase):

  def test_multiple_sync_fns(self):
    result = Chain(5).gather(lambda x: x + 1, lambda x: x + 2, lambda x: x + 3).run()
    self.assertEqual(result, [6, 7, 8])

  def test_single_fn(self):
    result = Chain(5).gather(lambda x: x * 2).run()
    self.assertEqual(result, [10])

  def test_zero_fns_returns_empty(self):
    result = Chain(5).gather().run()
    self.assertEqual(result, [])

  def test_preserves_order(self):
    fns = [lambda x, i=i: (x, i) for i in range(5)]
    result = Chain(99).gather(*fns).run()
    self.assertEqual(result, [(99, 0), (99, 1), (99, 2), (99, 3), (99, 4)])

  def test_fns_receive_current_value(self):
    received = []
    def capture(x):
      received.append(x)
      return x
    Chain(5).gather(capture, capture, capture).run()
    self.assertEqual(received, [5, 5, 5])


# ---------------------------------------------------------------------------
# Async gather
# ---------------------------------------------------------------------------

class TestGatherAsync(IsolatedAsyncioTestCase):

  async def test_all_async_fns(self):
    result = await Chain(5).gather(async_fn, async_fn).run()
    self.assertEqual(result, [6, 6])

  async def test_mixed_sync_async(self):
    result = await Chain(5).gather(lambda x: x + 1, async_fn).run()
    self.assertEqual(result, [6, 6])

  async def test_single_async_fn(self):
    result = await Chain(5).gather(async_fn).run()
    self.assertEqual(result, [6])

  async def test_preserves_order_async(self):
    async def slow(x):
      return x + 10
    async def fast(x):
      return x + 20
    result = await Chain(1).gather(slow, fast).run()
    self.assertEqual(result, [11, 21])


# ---------------------------------------------------------------------------
# Exception handling (sync)
# ---------------------------------------------------------------------------

class TestGatherExceptions(unittest.TestCase):

  def test_sync_exception_propagates(self):
    with self.assertRaises(ZeroDivisionError):
      Chain(5).gather(lambda x: 1 / 0, lambda x: x).run()

  def test_exception_in_first_fn(self):
    called = []
    def second(x):
      called.append(True)
      return x
    with self.assertRaises(ZeroDivisionError):
      Chain(5).gather(lambda x: 1 / 0, second).run()
    # Sequential evaluation: exception in first fn prevents second from running.
    self.assertEqual(called, [])


# ---------------------------------------------------------------------------
# Exception handling (async)
# ---------------------------------------------------------------------------

class TestGatherExceptionsAsync(IsolatedAsyncioTestCase):

  async def test_async_exception_propagates(self):
    with self.assertRaises(ValueError):
      await Chain(5).gather(async_raise_fn).run()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestGatherEdgeCases(unittest.TestCase):

  def test_fn_returns_none(self):
    result = Chain(5).gather(lambda x: None).run()
    self.assertEqual(result, [None])

  def test_fn_returns_another_callable(self):
    result = Chain(5).gather(lambda x: lambda: x).run()
    self.assertIsInstance(result, list)
    self.assertEqual(len(result), 1)
    # The returned lambda is NOT called; it is kept as a value.
    self.assertTrue(callable(result[0]))
    self.assertEqual(result[0](), 5)

  def test_many_fns(self):
    fns = [lambda x, i=i: x + i for i in range(20)]
    result = Chain(0).gather(*fns).run()
    self.assertEqual(result, list(range(20)))

  def test_gather_with_lambdas(self):
    result = Chain(10).gather(
      lambda x: x * 1,
      lambda x: x * 2,
      lambda x: x * 3,
      lambda x: x * 4,
    ).run()
    self.assertEqual(result, [10, 20, 30, 40])


if __name__ == '__main__':
  unittest.main()
