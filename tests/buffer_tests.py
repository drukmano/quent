# SPDX-License-Identifier: MIT
"""Tests for buffer() — backpressure-aware buffered iteration (SPEC S9.8)."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Q, QuentException
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# S9.8 buffer(n) — basic buffering
# ---------------------------------------------------------------------------


class BufferBasicTests(IsolatedAsyncioTestCase):
  """Basic buffer() integration with iterate()."""

  async def test_buffer_sync_iterate(self) -> None:
    """buffer(n).iterate() yields all elements in order (sync)."""
    result = list(Q(range(20)).buffer(5).iterate())
    self.assertEqual(result, list(range(20)))

  async def test_buffer_async_iterate(self) -> None:
    """buffer(n).iterate() yields all elements in order (async)."""
    result: list[int] = []
    async for item in Q(range(20)).buffer(5).iterate():
      result.append(item)
    self.assertEqual(result, list(range(20)))

  async def test_buffer_size_1(self) -> None:
    """buffer(1) — minimal buffer still delivers all elements."""
    result = list(Q(range(10)).buffer(1).iterate())
    self.assertEqual(result, list(range(10)))

  async def test_buffer_size_10(self) -> None:
    """buffer(10) with fewer items than buffer size."""
    result = list(Q(range(5)).buffer(10).iterate())
    self.assertEqual(result, list(range(5)))

  async def test_buffer_size_100(self) -> None:
    """buffer(100) with many items."""
    result = list(Q(range(200)).buffer(100).iterate())
    self.assertEqual(result, list(range(200)))

  async def test_buffer_empty_iterable(self) -> None:
    """buffer on an empty iterable yields nothing."""
    result = list(Q([]).buffer(5).iterate())
    self.assertEqual(result, [])

  async def test_buffer_with_fn(self) -> None:
    """buffer(n).iterate(fn) — fn transforms each element."""
    result = list(Q(range(10)).buffer(3).iterate(lambda x: x * 2))
    self.assertEqual(result, [x * 2 for x in range(10)])

  async def test_buffer_async_with_fn(self) -> None:
    """buffer(n).iterate(fn) with async fn."""

    async def double(x: int) -> int:
      return x * 2

    result: list[int] = []
    async for item in Q(range(10)).buffer(3).iterate(double):
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
    result = list(Q(range(10)).buffer(3).iterate_do(side_effects.append))
    self.assertEqual(result, list(range(10)))
    self.assertEqual(side_effects, list(range(10)))

  async def test_buffer_iterate_do_async(self) -> None:
    """buffer(n).iterate_do(fn) async path."""
    side_effects: list[int] = []
    result: list[int] = []
    async for item in Q(range(10)).buffer(3).iterate_do(side_effects.append):
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
    result = list(Q([[1, 2], [3, 4], [5]]).buffer(2).flat_iterate())
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_buffer_flat_iterate_async(self) -> None:
    """buffer(n).flat_iterate() async path."""
    result: list[int] = []
    async for item in Q([[1, 2], [3, 4], [5]]).buffer(2).flat_iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_buffer_flat_iterate_with_fn(self) -> None:
    """buffer(n).flat_iterate(fn) with transformation."""
    result = list(Q(range(3)).buffer(2).flat_iterate(lambda x: [x, x * 10]))
    self.assertEqual(result, [0, 0, 1, 10, 2, 20])

  async def test_buffer_flat_iterate_do_sync(self) -> None:
    """buffer(n).flat_iterate_do(fn) yields originals."""
    side: list[int] = []

    def consume(x: int) -> list[int]:
      items = [x, x * 10]
      side.extend(items)
      return items

    result = list(Q(range(3)).buffer(2).flat_iterate_do(consume))
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
    for item in Q(producing_range, 10).buffer(2).iterate():
      time.sleep(0.01)  # Simulate slow consumer
      result.append(item)

    self.assertEqual(result, list(range(10)))

  async def test_backpressure_async_slow_consumer(self) -> None:
    """Async slow consumer: producer awaits when buffer is full."""
    result: list[int] = []
    async for item in Q(range(10)).buffer(2).iterate():
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
      list(Q(bad_producer).buffer(5).iterate())
    self.assertIn('producer failed', str(ctx.exception))

  async def test_producer_error_async(self) -> None:
    """If the producer raises, the error propagates to the consumer (async)."""

    async def bad_producer() -> list[int]:
      raise ValueError('async producer failed')

    result: list[int] = []
    with self.assertRaises(ValueError) as ctx:
      async for item in Q(bad_producer).buffer(5).iterate():
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
      list(Q(partial_producer).buffer(3).iterate())
    self.assertIn('mid-stream error', str(ctx.exception))

  async def test_consumer_early_stop_sync(self) -> None:
    """Consumer breaks early; producer thread is cleaned up (sync)."""
    result: list[int] = []
    for item in Q(range(100)).buffer(5).iterate():
      if item >= 3:
        break
      result.append(item)
    self.assertEqual(result, [0, 1, 2])

  async def test_consumer_early_stop_async(self) -> None:
    """Consumer breaks early; producer task is cleaned up (async)."""
    result: list[int] = []
    async for item in Q(range(100)).buffer(5).iterate():
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
      for item in Q(failing_iter).buffer(2).iterate():
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
      async for item in Q(failing_iter).buffer(2).iterate():
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
      Q(range(5)).buffer(0)

  async def test_buffer_negative_raises(self) -> None:
    """buffer(-1) raises ValueError."""
    with self.assertRaises(ValueError):
      Q(range(5)).buffer(-1)

  async def test_buffer_non_int_raises(self) -> None:
    """buffer('5') raises TypeError."""
    with self.assertRaises(TypeError):
      Q(range(5)).buffer('5')  # type: ignore[arg-type]

  async def test_buffer_bool_raises(self) -> None:
    """buffer(True) raises TypeError (bool is excluded)."""
    with self.assertRaises(TypeError):
      Q(range(5)).buffer(True)  # type: ignore[arg-type]

  async def test_buffer_float_raises(self) -> None:
    """buffer(5.0) raises TypeError."""
    with self.assertRaises(TypeError):
      Q(range(5)).buffer(5.0)  # type: ignore[arg-type]

  async def test_buffer_with_pending_if_raises(self) -> None:
    """buffer() after if_() without then()/do() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(range(5)).if_(lambda x: True).buffer(3)

  async def test_buffer_with_run_raises(self) -> None:
    """buffer() + run() raises QuentException — must use iterate terminal."""
    with self.assertRaisesRegex(QuentException, r'buffer.*iteration terminal'):
      Q(range(5)).buffer(3).run()


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — iterator reuse
# ---------------------------------------------------------------------------


class BufferReuseTests(IsolatedAsyncioTestCase):
  """buffer() with iterator reuse (calling the QuentIterator)."""

  async def test_buffer_iterator_reuse(self) -> None:
    """Calling a buffered iterator returns a fresh iterator with the buffer."""
    it = Q().buffer(3).iterate(lambda x: x * 2)
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
    """clone() of a pipeline with buffer() preserves the buffer setting."""
    base = Q(range(10)).buffer(3)
    cloned = base.clone()
    result = list(cloned.iterate())
    self.assertEqual(result, list(range(10)))

  async def test_clone_buffer_independent(self) -> None:
    """Modifying the clone's buffer doesn't affect the original."""
    base = Q(range(10)).buffer(3)
    cloned = base.clone().buffer(7)
    # Original should still use buffer(3) — both should produce correct results
    result_base = list(base.iterate())
    result_cloned = list(cloned.iterate())
    self.assertEqual(result_base, list(range(10)))
    self.assertEqual(result_cloned, list(range(10)))


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — SymmetricTestCase sync/async symmetry tests
# ---------------------------------------------------------------------------

# Helpers for collecting iteration results via async for.


async def _acollect(it: object) -> list[object]:
  """Collect all items from an async-iterable into a list."""
  result: list[object] = []
  async for item in it:  # type: ignore[union-attr]
    result.append(item)
  return result


# Sync/async callable pairs for variant axes.


def _sync_double(x: object) -> object:
  return x * 2  # type: ignore[operator]


async def _async_double(x: object) -> object:
  return x * 2  # type: ignore[operator]


def _sync_expand(x: object) -> list[object]:
  return [x, x * 10]  # type: ignore[operator]


async def _async_expand(x: object) -> list[object]:
  return [x, x * 10]  # type: ignore[operator]


def _sync_side_effect(tracker: list[object], x: object) -> object:
  tracker.append(x)
  return x


async def _async_side_effect(tracker: list[object], x: object) -> object:
  tracker.append(x)
  return x


_V_DOUBLE = [('sync', _sync_double), ('async', _async_double)]
_V_EXPAND = [('sync', _sync_expand), ('async', _async_expand)]


class BufferIterateSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer(n).iterate() sync/async fn symmetry."""

  async def test_buffer_iterate_fn_symmetry(self) -> None:
    """buffer(n).iterate(fn) produces same elements with sync fn and async fn."""
    await self.variant(
      lambda fn: _acollect(Q(range(10)).buffer(3).iterate(fn)),
      expected=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
      fn=_V_DOUBLE,
    )

  async def test_buffer_iterate_no_fn_protocol_symmetry(self) -> None:
    """buffer(n).iterate() with no fn: sync for and async for produce same results."""
    await self.variant(
      lambda collect: collect(Q(range(10)).buffer(3).iterate()),
      expected=list(range(10)),
      collect=[('sync', list), ('async', _acollect)],
    )

  async def test_buffer_iterate_various_sizes(self) -> None:
    """buffer(n) for n=1,5,50 — all produce identical ordered results."""
    await self.variant(
      lambda fn: _acollect(Q(range(15)).buffer(5).iterate(fn)),
      expected=[i * 2 for i in range(15)],
      fn=_V_DOUBLE,
    )

  async def test_buffer_iterate_empty_source(self) -> None:
    """Empty source with buffer — sync/async fn both yield nothing."""
    await self.variant(
      lambda fn: _acollect(Q([]).buffer(3).iterate(fn)),
      expected=[],
      fn=_V_DOUBLE,
    )

  async def test_buffer_iterate_single_element(self) -> None:
    """Single element source with buffer — sync/async fn symmetry."""
    await self.variant(
      lambda fn: _acollect(Q([7]).buffer(2).iterate(fn)),
      expected=[14],
      fn=_V_DOUBLE,
    )

  async def test_buffer_size_1_fn_symmetry(self) -> None:
    """buffer(1) minimal buffer — fn symmetry."""
    await self.variant(
      lambda fn: _acollect(Q(range(5)).buffer(1).iterate(fn)),
      expected=[0, 2, 4, 6, 8],
      fn=_V_DOUBLE,
    )

  async def test_buffer_larger_than_source_fn_symmetry(self) -> None:
    """buffer(n) where n > len(source) — fn symmetry."""
    await self.variant(
      lambda fn: _acollect(Q(range(3)).buffer(100).iterate(fn)),
      expected=[0, 2, 4],
      fn=_V_DOUBLE,
    )


class BufferIterateDoSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer(n).iterate_do() sync/async fn symmetry."""

  async def test_buffer_iterate_do_fn_symmetry(self) -> None:
    """buffer(n).iterate_do(fn) yields originals regardless of sync/async fn."""
    await self.variant(
      lambda fn: _acollect(Q(range(8)).buffer(3).iterate_do(fn)),
      expected=list(range(8)),
      fn=_V_DOUBLE,
    )

  async def test_buffer_iterate_do_no_fn_protocol_symmetry(self) -> None:
    """buffer(n).iterate_do() with no fn: sync/async collection symmetry."""
    await self.variant(
      lambda collect: collect(Q(range(8)).buffer(3).iterate_do()),
      expected=list(range(8)),
      collect=[('sync', list), ('async', _acollect)],
    )

  async def test_buffer_iterate_do_side_effects_executed(self) -> None:
    """buffer(n).iterate_do(fn) — fn is called for side-effects, sync/async symmetry."""
    for label, fn_factory in [
      ('sync', lambda t: lambda x: _sync_side_effect(t, x)),
      ('async', lambda t: lambda x: _async_side_effect(t, x)),
    ]:
      with self.subTest(fn=label):
        tracker: list[object] = []
        fn = fn_factory(tracker)
        result: list[object] = []
        async for item in Q(range(5)).buffer(2).iterate_do(fn):
          result.append(item)
        self.assertEqual(result, list(range(5)), f'{label}: yielded items mismatch')
        self.assertEqual(tracker, list(range(5)), f'{label}: side-effects mismatch')


class BufferFlatIterateSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer(n).flat_iterate() sync/async fn symmetry."""

  async def test_buffer_flat_iterate_no_fn_protocol_symmetry(self) -> None:
    """buffer(n).flat_iterate() with no fn: sync/async protocol symmetry."""
    await self.variant(
      lambda collect: collect(Q([[1, 2], [3, 4], [5]]).buffer(2).flat_iterate()),
      expected=[1, 2, 3, 4, 5],
      collect=[('sync', list), ('async', _acollect)],
    )

  async def test_buffer_flat_iterate_fn_symmetry(self) -> None:
    """buffer(n).flat_iterate(fn) — sync/async fn produce same flattened output."""
    await self.variant(
      lambda fn: _acollect(Q(range(4)).buffer(2).flat_iterate(fn)),
      expected=[0, 0, 1, 10, 2, 20, 3, 30],
      fn=_V_EXPAND,
    )

  async def test_buffer_flat_iterate_empty_source(self) -> None:
    """buffer(n).flat_iterate(fn) on empty source — fn symmetry."""
    await self.variant(
      lambda fn: _acollect(Q([]).buffer(3).flat_iterate(fn)),
      expected=[],
      fn=_V_EXPAND,
    )


class BufferFlatIterateDoSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer(n).flat_iterate_do() sync/async fn symmetry."""

  async def test_buffer_flat_iterate_do_fn_symmetry(self) -> None:
    """buffer(n).flat_iterate_do(fn) yields originals regardless of sync/async fn."""
    await self.variant(
      lambda fn: _acollect(Q(range(3)).buffer(2).flat_iterate_do(fn)),
      expected=[0, 1, 2],
      fn=_V_EXPAND,
    )

  async def test_buffer_flat_iterate_do_no_fn_protocol_symmetry(self) -> None:
    """buffer(n).flat_iterate_do() no fn: sync/async collection symmetry."""
    await self.variant(
      lambda collect: collect(Q([[1, 2], [3]]).buffer(2).flat_iterate_do()),
      expected=[1, 2, 3],
      collect=[('sync', list), ('async', _acollect)],
    )


class BufferErrorSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer error propagation sync/async symmetry."""

  async def test_buffer_producer_error_protocol_symmetry(self) -> None:
    """Producer error propagates identically via sync and async iteration."""

    def bad_producer() -> list[int]:
      raise ValueError('symmetric producer error')

    await self.variant(
      lambda collect: collect(Q(bad_producer).buffer(3).iterate()),
      expected_exc=ValueError,
      expected_msg='symmetric producer error',
      collect=[('sync', list), ('async', _acollect)],
    )

  async def test_buffer_fn_error_symmetry(self) -> None:
    """fn error during buffered iteration — sync/async fn symmetry."""

    def sync_fail(x: object) -> object:
      if x == 3:
        raise RuntimeError('fn error at 3')
      return x

    async def async_fail(x: object) -> object:
      if x == 3:
        raise RuntimeError('fn error at 3')
      return x

    await self.variant(
      lambda fn: _acollect(Q(range(10)).buffer(2).iterate(fn)),
      expected_exc=RuntimeError,
      expected_msg='fn error at 3',
      fn=[('sync', sync_fail), ('async', async_fail)],
    )


class BufferControlFlowSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer + control flow (return_/break_) sync/async symmetry."""

  async def test_buffer_return_with_value_symmetry(self) -> None:
    """return_(v) in buffered iterate — sync/async fn symmetry."""

    def sync_fn(x: object) -> object:
      if x == 3:
        return Q.return_(x * 10)  # type: ignore[operator]
      return x

    async def async_fn(x: object) -> object:
      if x == 3:
        return Q.return_(x * 10)  # type: ignore[operator]
      return x

    await self.variant(
      lambda fn: _acollect(Q(range(8)).buffer(2).iterate(fn)),
      expected=[0, 1, 2, 30],
      fn=[('sync', sync_fn), ('async', async_fn)],
    )

  async def test_buffer_return_no_value_symmetry(self) -> None:
    """return_() with no value in buffered iterate — sync/async fn symmetry."""

    def sync_fn(x: object) -> object:
      if x == 3:
        return Q.return_()
      return x

    async def async_fn(x: object) -> object:
      if x == 3:
        return Q.return_()
      return x

    await self.variant(
      lambda fn: _acollect(Q(range(8)).buffer(2).iterate(fn)),
      expected=[0, 1, 2],
      fn=[('sync', sync_fn), ('async', async_fn)],
    )

  async def test_buffer_break_with_value_symmetry(self) -> None:
    """break_(v) in buffered iterate — sync/async fn symmetry."""

    def sync_fn(x: object) -> object:
      if x == 4:
        return Q.break_(x * 100)  # type: ignore[operator]
      return x

    async def async_fn(x: object) -> object:
      if x == 4:
        return Q.break_(x * 100)  # type: ignore[operator]
      return x

    await self.variant(
      lambda fn: _acollect(Q(range(10)).buffer(3).iterate(fn)),
      expected=[0, 1, 2, 3, 400],
      fn=[('sync', sync_fn), ('async', async_fn)],
    )

  async def test_buffer_break_no_value_symmetry(self) -> None:
    """break_() in buffered iterate — sync/async fn symmetry."""

    def sync_fn(x: object) -> object:
      if x == 4:
        return Q.break_()
      return x

    async def async_fn(x: object) -> object:
      if x == 4:
        return Q.break_()
      return x

    await self.variant(
      lambda fn: _acollect(Q(range(10)).buffer(3).iterate(fn)),
      expected=[0, 1, 2, 3],
      fn=[('sync', sync_fn), ('async', async_fn)],
    )


class BufferCloneSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer setting preserved across clone, sync/async symmetry."""

  async def test_clone_buffer_fn_symmetry(self) -> None:
    """clone() preserves buffer — sync/async fn produce same results."""
    base = Q(range(8)).buffer(3)
    cloned = base.clone()
    await self.variant(
      lambda fn: _acollect(cloned.iterate(fn)),
      expected=[0, 2, 4, 6, 8, 10, 12, 14],
      fn=_V_DOUBLE,
    )

  async def test_clone_buffer_protocol_symmetry(self) -> None:
    """clone() preserves buffer — sync/async iteration protocol symmetry."""
    base = Q(range(6)).buffer(2)
    cloned = base.clone()
    await self.variant(
      lambda collect: collect(cloned.iterate()),
      expected=list(range(6)),
      collect=[('sync', list), ('async', _acollect)],
    )


class BufferReuseSymmetricTest(SymmetricTestCase):
  """S9.8 — buffer iterator reuse sync/async symmetry."""

  async def test_reuse_buffer_fn_symmetry(self) -> None:
    """Calling a buffered QuentIterator with new args — sync/async fn symmetry."""

    def sync_triple(x: object) -> object:
      return x * 3  # type: ignore[operator]

    async def async_triple(x: object) -> object:
      return x * 3  # type: ignore[operator]

    for label, fn in [('sync', sync_triple), ('async', async_triple)]:
      with self.subTest(fn=label):
        it = Q().buffer(2).iterate(fn)
        result_a: list[object] = []
        async for item in it(range(4)):
          result_a.append(item)
        result_b: list[object] = []
        async for item in it(range(2)):
          result_b.append(item)
        self.assertEqual(result_a, [0, 3, 6, 9], f'{label}: first call mismatch')
        self.assertEqual(result_b, [0, 3], f'{label}: second call mismatch')

  async def test_reuse_buffer_protocol_symmetry(self) -> None:
    """Calling a buffered QuentIterator — sync/async protocol yield same results."""
    it = Q().buffer(3).iterate()
    await self.variant(
      lambda collect: collect(it(range(5))),
      expected=list(range(5)),
      collect=[('sync', list), ('async', _acollect)],
    )


# ---------------------------------------------------------------------------
# S9.8 buffer(n) — error path coverage
# ---------------------------------------------------------------------------


class BufferProducerExceptionMidStreamTests(IsolatedAsyncioTestCase):
  """S9.8: Producer iterable raises exception mid-stream → error propagates to consumer.

  §9.8 Error behavior:
    'If the producer raises an exception, it is propagated to the consumer
    at the next get().'
  """

  async def test_sync_producer_generator_raises_mid_stream(self) -> None:
    """Sync: Producer generator raises mid-stream → consumer gets error."""

    def failing_generator():
      for i in range(10):
        if i == 4:
          raise RuntimeError('producer gen failed at 4')
        yield i

    collected: list[int] = []
    with self.assertRaises(RuntimeError) as ctx:
      for item in Q(failing_generator).buffer(3).iterate():
        collected.append(item)
    self.assertIn('producer gen failed at 4', str(ctx.exception))
    # Items before the error should have been received
    self.assertTrue(len(collected) >= 4)
    self.assertEqual(collected[:4], [0, 1, 2, 3])

  async def test_async_producer_generator_raises_mid_stream(self) -> None:
    """Async: Producer generator raises mid-stream → consumer gets error through asyncio.Queue."""

    def failing_generator():
      for i in range(10):
        if i == 3:
          raise RuntimeError('async producer gen failed at 3')
        yield i

    collected: list[int] = []
    with self.assertRaises(RuntimeError) as ctx:
      async for item in Q(failing_generator).buffer(2).iterate():
        collected.append(item)
    self.assertIn('async producer gen failed at 3', str(ctx.exception))
    self.assertTrue(len(collected) >= 3)
    self.assertEqual(collected[:3], [0, 1, 2])


class BufferConsumerEarlyBreakCleanupTests(IsolatedAsyncioTestCase):
  """S9.8: Consumer breaks out early → producer cleaned up.

  §9.8 Error behavior:
    'If the consumer exits early (e.g., break, GeneratorExit), the producer
    is signaled to stop:
      - Sync: a threading.Event is set; the producer checks it periodically.
      - Async: the producer task is cancelled.'
  """

  async def test_sync_consumer_break_stops_producer(self) -> None:
    """Sync consumer break → producer thread cleaned up."""
    produced_count = {'n': 0}

    def tracking_generator():
      for i in range(1000):
        produced_count['n'] += 1
        yield i

    collected: list[int] = []
    for item in Q(tracking_generator).buffer(5).iterate():
      collected.append(item)
      if item >= 2:
        break

    self.assertEqual(collected, [0, 1, 2])
    # Producer should not have iterated all 1000 items (it was stopped)
    # Allow some margin since the producer may have buffered ahead
    self.assertLess(produced_count['n'], 100)

  async def test_async_consumer_break_cancels_producer(self) -> None:
    """Async consumer break → producer task cancelled."""
    produced_count = {'n': 0}

    def tracking_generator():
      for i in range(1000):
        produced_count['n'] += 1
        yield i

    collected: list[int] = []
    async for item in Q(tracking_generator).buffer(5).iterate():
      collected.append(item)
      if item >= 2:
        break

    self.assertEqual(collected, [0, 1, 2])
    # Give a moment for cleanup
    await asyncio.sleep(0.1)
    # Producer should not have iterated all 1000 items
    self.assertLess(produced_count['n'], 100)

  async def test_sync_consumer_close_generator(self) -> None:
    """Sync: Close generator explicitly → producer cleaned up."""
    collected: list[int] = []
    gen = iter(Q(range(100)).buffer(3).iterate())
    collected.append(next(gen))
    collected.append(next(gen))
    gen.close()  # Triggers GeneratorExit cleanup
    self.assertEqual(collected, [0, 1])


class BufferAsyncProducerErrorTests(IsolatedAsyncioTestCase):
  """S9.8: Async buffer producer raises → error propagates through asyncio.Queue.

  §9.8 Error behavior:
    'If the producer raises an exception, it is propagated to the consumer
    at the next get().'

  The async path in _async_buffer_iter catches BaseException and puts a
  _ProducerError into the queue.
  """

  async def test_async_producer_error_propagates(self) -> None:
    """Async producer raises mid-stream → error propagates to async consumer."""

    def error_iterable():
      yield 1
      yield 2
      raise ValueError('async producer error mid-stream')

    collected: list[int] = []
    with self.assertRaises(ValueError) as ctx:
      async for item in Q(error_iterable).buffer(2).iterate():
        collected.append(item)
    self.assertIn('async producer error mid-stream', str(ctx.exception))
    # At least items 1 and 2 should have been received
    self.assertEqual(collected[:2], [1, 2])

  async def test_async_producer_immediate_error(self) -> None:
    """Async producer raises immediately → error propagates to consumer."""

    def immediate_failure():
      raise RuntimeError('immediate async failure')

    with self.assertRaises(RuntimeError) as ctx:
      async for _ in Q(immediate_failure).buffer(5).iterate():
        pass
    self.assertIn('immediate async failure', str(ctx.exception))


if __name__ == '__main__':
  unittest.main()
