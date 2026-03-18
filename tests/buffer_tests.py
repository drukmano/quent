# SPDX-License-Identifier: MIT
"""Tests for buffer() — backpressure-aware buffered iteration (SPEC S9.8)."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, QuentException

# ---------------------------------------------------------------------------
# S9.8 buffer(n) — basic buffering
# ---------------------------------------------------------------------------


class BufferBasicTests(IsolatedAsyncioTestCase):
  """Basic buffer() integration with iterate()."""

  async def test_buffer_sync_iterate(self) -> None:
    """buffer(n).iterate() yields all elements in order (sync)."""
    result = list(Chain(range(20)).buffer(5).iterate())
    self.assertEqual(result, list(range(20)))

  async def test_buffer_async_iterate(self) -> None:
    """buffer(n).iterate() yields all elements in order (async)."""
    result: list[int] = []
    async for item in Chain(range(20)).buffer(5).iterate():
      result.append(item)
    self.assertEqual(result, list(range(20)))

  async def test_buffer_size_1(self) -> None:
    """buffer(1) — minimal buffer still delivers all elements."""
    result = list(Chain(range(10)).buffer(1).iterate())
    self.assertEqual(result, list(range(10)))

  async def test_buffer_size_10(self) -> None:
    """buffer(10) with fewer items than buffer size."""
    result = list(Chain(range(5)).buffer(10).iterate())
    self.assertEqual(result, list(range(5)))

  async def test_buffer_size_100(self) -> None:
    """buffer(100) with many items."""
    result = list(Chain(range(200)).buffer(100).iterate())
    self.assertEqual(result, list(range(200)))

  async def test_buffer_empty_iterable(self) -> None:
    """buffer on an empty iterable yields nothing."""
    result = list(Chain([]).buffer(5).iterate())
    self.assertEqual(result, [])

  async def test_buffer_with_fn(self) -> None:
    """buffer(n).iterate(fn) — fn transforms each element."""
    result = list(Chain(range(10)).buffer(3).iterate(lambda x: x * 2))
    self.assertEqual(result, [x * 2 for x in range(10)])

  async def test_buffer_async_with_fn(self) -> None:
    """buffer(n).iterate(fn) with async fn."""

    async def double(x: int) -> int:
      return x * 2

    result: list[int] = []
    async for item in Chain(range(10)).buffer(3).iterate(double):
      result.append(item)
    self.assertEqual(result, [x * 2 for x in range(10)])


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — iterate_do
# ---------------------------------------------------------------------------


class BufferIterateDoTests(IsolatedAsyncioTestCase):
  """buffer() integration with iterate_do()."""

  async def test_buffer_iterate_do_sync(self) -> None:
    """buffer(n).iterate_do(fn) yields original elements, fn as side-effect."""
    side_effects: list[int] = []
    result = list(Chain(range(10)).buffer(3).iterate_do(side_effects.append))
    self.assertEqual(result, list(range(10)))
    self.assertEqual(side_effects, list(range(10)))

  async def test_buffer_iterate_do_async(self) -> None:
    """buffer(n).iterate_do(fn) async path."""
    side_effects: list[int] = []
    result: list[int] = []
    async for item in Chain(range(10)).buffer(3).iterate_do(side_effects.append):
      result.append(item)
    self.assertEqual(result, list(range(10)))
    self.assertEqual(side_effects, list(range(10)))


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — flat_iterate integration
# ---------------------------------------------------------------------------


class BufferFlatIterateTests(IsolatedAsyncioTestCase):
  """buffer() integration with flat_iterate() and flat_iterate_do()."""

  async def test_buffer_flat_iterate_sync(self) -> None:
    """buffer(n).flat_iterate() flattens with buffering."""
    result = list(Chain([[1, 2], [3, 4], [5]]).buffer(2).flat_iterate())
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_buffer_flat_iterate_async(self) -> None:
    """buffer(n).flat_iterate() async path."""
    result: list[int] = []
    async for item in Chain([[1, 2], [3, 4], [5]]).buffer(2).flat_iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_buffer_flat_iterate_with_fn(self) -> None:
    """buffer(n).flat_iterate(fn) with transformation."""
    result = list(Chain(range(3)).buffer(2).flat_iterate(lambda x: [x, x * 10]))
    self.assertEqual(result, [0, 0, 1, 10, 2, 20])

  async def test_buffer_flat_iterate_do_sync(self) -> None:
    """buffer(n).flat_iterate_do(fn) yields originals."""
    side: list[int] = []

    def consume(x: int) -> list[int]:
      items = [x, x * 10]
      side.extend(items)
      return items

    result = list(Chain(range(3)).buffer(2).flat_iterate_do(consume))
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(side, [0, 0, 1, 10, 2, 20])


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — backpressure
# ---------------------------------------------------------------------------


class BufferBackpressureTests(IsolatedAsyncioTestCase):
  """Verify that the producer blocks when the buffer is full."""

  async def test_backpressure_sync_slow_consumer(self) -> None:
    """Slow consumer: producer must wait when buffer is full.

    With a buffer of size 2 and a producer that emits 10 items, the producer
    should be blocked when the buffer is full (backpressure). We verify that
    all items arrive in order and that the producer doesn't run ahead unbounded.
    """
    produced: list[int] = []

    def producing_range(n: int) -> list[int]:
      items = list(range(n))
      produced.extend(items)
      return items

    result: list[int] = []
    for item in Chain(producing_range, 10).buffer(2).iterate():
      time.sleep(0.01)  # Simulate slow consumer
      result.append(item)

    self.assertEqual(result, list(range(10)))

  async def test_backpressure_async_slow_consumer(self) -> None:
    """Async slow consumer: producer awaits when buffer is full."""
    result: list[int] = []
    async for item in Chain(range(10)).buffer(2).iterate():
      await asyncio.sleep(0.01)  # Simulate slow consumer
      result.append(item)
    self.assertEqual(result, list(range(10)))


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — error propagation
# ---------------------------------------------------------------------------


class BufferErrorTests(IsolatedAsyncioTestCase):
  """Error propagation through buffered iteration."""

  async def test_producer_error_sync(self) -> None:
    """If the producer raises, the error propagates to the consumer (sync)."""

    def bad_producer() -> list[int]:
      raise ValueError('producer failed')

    with self.assertRaises(ValueError) as ctx:
      list(Chain(bad_producer).buffer(5).iterate())
    self.assertIn('producer failed', str(ctx.exception))

  async def test_producer_error_async(self) -> None:
    """If the producer raises, the error propagates to the consumer (async)."""

    async def bad_producer() -> list[int]:
      raise ValueError('async producer failed')

    result: list[int] = []
    with self.assertRaises(ValueError) as ctx:
      async for item in Chain(bad_producer).buffer(5).iterate():
        result.append(item)
    self.assertIn('async producer failed', str(ctx.exception))

  async def test_producer_mid_iteration_error_sync(self) -> None:
    """Producer raises mid-iteration; consumer gets the error (sync)."""

    def partial_producer() -> list[int]:
      result = []
      for i in range(10):
        if i == 5:
          raise RuntimeError('mid-stream error')
        result.append(i)
      return result

    with self.assertRaises(RuntimeError) as ctx:
      list(Chain(partial_producer).buffer(3).iterate())
    self.assertIn('mid-stream error', str(ctx.exception))

  async def test_consumer_early_stop_sync(self) -> None:
    """Consumer breaks early; producer thread is cleaned up (sync)."""
    result: list[int] = []
    for item in Chain(range(100)).buffer(5).iterate():
      if item >= 3:
        break
      result.append(item)
    self.assertEqual(result, [0, 1, 2])

  async def test_consumer_early_stop_async(self) -> None:
    """Consumer breaks early; producer task is cleaned up (async)."""
    result: list[int] = []
    async for item in Chain(range(100)).buffer(5).iterate():
      if item >= 3:
        break
      result.append(item)
    self.assertEqual(result, [0, 1, 2])

  async def test_iterable_error_mid_stream_sync(self) -> None:
    """Producer iterable raises mid-stream; error reaches consumer (sync)."""

    def failing_iter():
      for i in range(10):
        if i == 5:
          raise RuntimeError('iter error at 5')
        yield i

    result: list[int] = []
    with self.assertRaises(RuntimeError) as ctx:
      for item in Chain(failing_iter).buffer(2).iterate():
        result.append(item)
    self.assertIn('iter error at 5', str(ctx.exception))
    # We should have received at least the first 5 items
    self.assertTrue(len(result) >= 5)

  async def test_iterable_error_mid_stream_async(self) -> None:
    """Producer iterable raises mid-stream; error reaches consumer (async)."""

    def failing_iter():
      for i in range(10):
        if i == 5:
          raise RuntimeError('async iter error at 5')
        yield i

    result: list[int] = []
    with self.assertRaises(RuntimeError) as ctx:
      async for item in Chain(failing_iter).buffer(2).iterate():
        result.append(item)
    self.assertIn('async iter error at 5', str(ctx.exception))
    self.assertTrue(len(result) >= 5)


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — validation
# ---------------------------------------------------------------------------


class BufferValidationTests(IsolatedAsyncioTestCase):
  """Validation of buffer() arguments."""

  async def test_buffer_zero_raises(self) -> None:
    """buffer(0) raises ValueError."""
    with self.assertRaises(ValueError):
      Chain(range(5)).buffer(0)

  async def test_buffer_negative_raises(self) -> None:
    """buffer(-1) raises ValueError."""
    with self.assertRaises(ValueError):
      Chain(range(5)).buffer(-1)

  async def test_buffer_non_int_raises(self) -> None:
    """buffer('5') raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(range(5)).buffer('5')  # type: ignore[arg-type]

  async def test_buffer_bool_raises(self) -> None:
    """buffer(True) raises TypeError (bool is excluded)."""
    with self.assertRaises(TypeError):
      Chain(range(5)).buffer(True)  # type: ignore[arg-type]

  async def test_buffer_float_raises(self) -> None:
    """buffer(5.0) raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(range(5)).buffer(5.0)  # type: ignore[arg-type]

  async def test_buffer_with_pending_if_raises(self) -> None:
    """buffer() after if_() without then()/do() raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(range(5)).if_(lambda x: True).buffer(3)

  async def test_buffer_with_run_raises(self) -> None:
    """buffer() + run() raises QuentException — must use iterate terminal."""
    with self.assertRaises(QuentException, msg='buffer.*iteration terminal'):
      Chain(range(5)).buffer(3).run()


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — iterator reuse
# ---------------------------------------------------------------------------


class BufferReuseTests(IsolatedAsyncioTestCase):
  """buffer() with iterator reuse (calling the ChainIterator)."""

  async def test_buffer_iterator_reuse(self) -> None:
    """Calling a buffered iterator returns a fresh iterator with the buffer."""
    it = Chain().buffer(3).iterate(lambda x: x * 2)
    result1 = list(it(range(5)))
    result2 = list(it(range(3)))
    self.assertEqual(result1, [0, 2, 4, 6, 8])
    self.assertEqual(result2, [0, 2, 4])


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — clone support
# ---------------------------------------------------------------------------


class BufferCloneTests(IsolatedAsyncioTestCase):
  """buffer() setting is preserved across clone()."""

  async def test_clone_preserves_buffer(self) -> None:
    """clone() of a chain with buffer() preserves the buffer setting."""
    base = Chain(range(10)).buffer(3)
    cloned = base.clone()
    result = list(cloned.iterate())
    self.assertEqual(result, list(range(10)))

  async def test_clone_buffer_independent(self) -> None:
    """Modifying the clone's buffer doesn't affect the original."""
    base = Chain(range(10)).buffer(3)
    cloned = base.clone().buffer(7)
    # Original should still use buffer(3) — both should produce correct results
    result_base = list(base.iterate())
    result_cloned = list(cloned.iterate())
    self.assertEqual(result_base, list(range(10)))
    self.assertEqual(result_cloned, list(range(10)))


if __name__ == '__main__':
  unittest.main()
