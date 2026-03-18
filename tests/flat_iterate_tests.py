# SPDX-License-Identifier: MIT
"""Tests for flat_iterate / flat_iterate_do."""

from __future__ import annotations

import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase

from quent import Chain, QuentException
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §9 — flat_iterate core behavior
# ---------------------------------------------------------------------------


class FlatIterateBasicTests(SymmetricTestCase):
  """Core flat_iterate() tests — flatmap semantics."""

  async def test_flat_iterate_basic(self) -> None:
    """flat_iterate() with no fn — flattens nested iterables."""
    result = list(Chain([[1, 2], [3], [], [4, 5]]).flat_iterate())
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_flat_iterate_with_fn(self) -> None:
    """flat_iterate(fn) — fn returns iterable, each sub-item yielded."""
    result = list(Chain(range(3)).flat_iterate(lambda x: [x, x * 10]))
    self.assertEqual(result, [0, 0, 1, 10, 2, 20])

  async def test_flat_iterate_with_flush(self) -> None:
    """After source exhaustion, flush() output is yielded."""
    result = list(Chain([[1], [2]]).flat_iterate(flush=lambda: [98, 99]))
    self.assertEqual(result, [1, 2, 98, 99])

  async def test_flat_iterate_fn_and_flush(self) -> None:
    """Both fn and flush — fn transforms each item, then flush appends."""
    result = list(
      Chain(range(3)).flat_iterate(
        lambda x: [x, x * 10],
        flush=lambda: [99],
      )
    )
    self.assertEqual(result, [0, 0, 1, 10, 2, 20, 99])

  async def test_flat_iterate_empty_fn_result(self) -> None:
    """fn returns empty iterable — skip, no items yielded for that element."""
    result = list(Chain([1, 2, 3]).flat_iterate(lambda x: [] if x == 2 else [x]))
    self.assertEqual(result, [1, 3])

  async def test_flat_iterate_empty_flush(self) -> None:
    """flush returns empty iterable — nothing extra yielded."""
    result = list(Chain([[1], [2]]).flat_iterate(flush=lambda: []))
    self.assertEqual(result, [1, 2])


# ---------------------------------------------------------------------------
# §9 — flat_iterate_do core behavior
# ---------------------------------------------------------------------------


class FlatIterateDoBasicTests(SymmetricTestCase):
  """Core flat_iterate_do() tests — side-effect flatmap."""

  async def test_flat_iterate_do_basic(self) -> None:
    """flat_iterate_do(fn) — fn runs as side-effect, original items yielded."""
    consumed: list[Any] = []

    def capture_items(x: int) -> list[int]:
      items = [x * 10, x * 100]
      consumed.extend(items)
      return items

    result = list(Chain([1, 2, 3]).flat_iterate_do(capture_items))
    # Original items yielded (not fn results)
    self.assertEqual(result, [1, 2, 3])
    # Side-effect: fn was called and its iterable consumed
    self.assertEqual(consumed, [10, 100, 20, 200, 30, 300])

  async def test_flat_iterate_do_with_flush(self) -> None:
    """flat_iterate_do with flush — original items yielded, then flush output."""
    side_effects: list[int] = []
    result = list(
      Chain([1, 2]).flat_iterate_do(
        lambda x: side_effects.append(x) or [x * 10],
        flush=lambda: [88, 99],
      )
    )
    # Original items [1, 2] then flush output [88, 99]
    self.assertEqual(result, [1, 2, 88, 99])
    self.assertEqual(side_effects, [1, 2])

  async def test_flat_iterate_do_no_fn(self) -> None:
    """flat_iterate_do with no fn — items yielded as-is (like iterate)."""
    # When fn is None and flat=True, no fn processing happens,
    # but each item is itself iterated (flat mode without fn).
    # flat_iterate_do with no fn: flat mode means each item is iterated.
    # Since ignore_result=True and fn is None, flat mode yields sub-items.
    result = list(Chain([[1, 2], [3]]).flat_iterate_do())
    # No fn, flat mode: each item is iterated and sub-items yielded
    self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class FlatIterateAsyncTests(IsolatedAsyncioTestCase):
  """Async iteration tests for flat_iterate / flat_iterate_do."""

  async def test_flat_iterate_async(self) -> None:
    """async for with sync source — flat_iterate works in async context."""
    result: list[int] = []
    async for item in Chain([[1, 2], [3], [4, 5]]).flat_iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3, 4, 5])

  async def test_flat_iterate_async_fn(self) -> None:
    """Async fn in async flat_iterate."""

    async def async_expand(x: int) -> list[int]:
      return [x, x * 10]

    result: list[int] = []
    async for item in Chain(range(3)).flat_iterate(async_expand):
      result.append(item)
    self.assertEqual(result, [0, 0, 1, 10, 2, 20])

  async def test_flat_iterate_async_flush(self) -> None:
    """Async flush callable in async flat_iterate."""

    async def async_flush() -> list[int]:
      return [98, 99]

    result: list[int] = []
    async for item in Chain([[1], [2]]).flat_iterate(flush=async_flush):
      result.append(item)
    self.assertEqual(result, [1, 2, 98, 99])

  async def test_flat_iterate_do_async(self) -> None:
    """flat_iterate_do in async context — original items yielded."""
    consumed: list[int] = []

    def capture(x: int) -> list[int]:
      items = [x * 10]
      consumed.extend(items)
      return items

    result: list[int] = []
    async for item in Chain([1, 2, 3]).flat_iterate_do(capture):
      result.append(item)
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(consumed, [10, 20, 30])


# ---------------------------------------------------------------------------
# Integration: deferred with_ + flat_iterate
# ---------------------------------------------------------------------------


class FlatIterateWithDeferredWithTests(SymmetricTestCase):
  """flat_iterate combined with deferred with_."""

  async def test_flat_iterate_with_deferred_with(self) -> None:
    """Chain(cm).with_(fn).flat_iterate(transform, flush=flush_fn) — CM stays open."""

    class TrackingCM:
      def __init__(self, value: Any) -> None:
        self.value = value
        self.entered = False
        self.exited = False

      def __enter__(self) -> Any:
        self.entered = True
        return self.value

      def __exit__(self, *args: Any) -> bool:
        self.exited = True
        return False

    items = [1, 2, 3]
    cm = TrackingCM(items)
    result = list(
      Chain(cm)
      .with_(lambda ctx: ctx)
      .flat_iterate(
        lambda x: [x, x * 10],
        flush=lambda: [99],
      )
    )
    self.assertEqual(result, [1, 10, 2, 20, 3, 30, 99])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# Streaming codec pattern (simplified simulation)
# ---------------------------------------------------------------------------


class FlatIterateStreamingCodecPatternTests(SymmetricTestCase):
  """Streaming codec pattern: CM wraps iteration, flat_iterate decodes chunks, flush emits remaining buffered items."""

  async def test_streaming_codec_pattern(self) -> None:
    """Streaming codec pattern: CM wraps iteration, flat_iterate decodes, flush emits buffered remainder."""
    # Simulate: request_context CM that wraps the iteration
    # iter_raw returns raw chunks, decode_raw_bytes is the flatmap fn,
    # flush_decoder is the flush callable.

    class RequestContextCM:
      def __init__(self) -> None:
        self.entered = False
        self.exited = False

      def __enter__(self) -> Any:
        self.entered = True
        return self

      def __exit__(self, *args: Any) -> bool:
        self.exited = True
        return False

    raw_chunks = [b'hello ', b'world']

    def iter_raw() -> list[bytes]:
      return raw_chunks

    def decode_raw_bytes(raw: bytes) -> list[str]:
      return [raw.decode('utf-8')]

    def flush_decoder() -> list[str]:
      return ['<flush>']

    request_ctx = RequestContextCM()
    result = list(
      Chain(request_ctx)
      .with_(lambda _ctx: iter_raw())
      .flat_iterate(
        decode_raw_bytes,
        flush=flush_decoder,
      )
    )
    self.assertEqual(result, ['hello ', 'world', '<flush>'])
    self.assertTrue(request_ctx.entered)
    self.assertTrue(request_ctx.exited)

  async def test_streaming_codec_pattern_async(self) -> None:
    """Async variant of the streaming codec pattern."""

    class AsyncRequestContextCM:
      def __init__(self) -> None:
        self.entered = False
        self.exited = False

      async def __aenter__(self) -> Any:
        self.entered = True
        return self

      async def __aexit__(self, *args: Any) -> bool:
        self.exited = True
        return False

    raw_chunks = [b'async ', b'data']

    def decode_raw(raw: bytes) -> list[str]:
      return [raw.decode('utf-8')]

    def flush_fn() -> list[str]:
      return ['<end>']

    request_ctx = AsyncRequestContextCM()
    result: list[str] = []
    async for item in Chain(request_ctx).with_(lambda _ctx: raw_chunks).flat_iterate(decode_raw, flush=flush_fn):
      result.append(item)
    self.assertEqual(result, ['async ', 'data', '<end>'])
    self.assertTrue(request_ctx.entered)
    self.assertTrue(request_ctx.exited)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class FlatIterateEdgeCaseTests(SymmetricTestCase):
  """Edge cases for flat_iterate."""

  async def test_flat_iterate_fn_returns_string(self) -> None:
    """Strings are iterables of chars — verify they are flattened."""
    result = list(Chain(['ab', 'cd']).flat_iterate(lambda x: x))
    # 'ab' -> 'a', 'b'; 'cd' -> 'c', 'd'
    self.assertEqual(result, ['a', 'b', 'c', 'd'])

  async def test_flat_iterate_flush_only_no_fn(self) -> None:
    """fn=None + flush — flatten items, then yield flush output."""
    result = list(Chain([[1, 2], [3]]).flat_iterate(flush=lambda: [99]))
    self.assertEqual(result, [1, 2, 3, 99])

  async def test_flat_iterate_empty_source(self) -> None:
    """Empty source iterable — only flush output (if any)."""
    result = list(Chain([]).flat_iterate(flush=lambda: [42]))
    self.assertEqual(result, [42])

  async def test_flat_iterate_empty_source_no_flush(self) -> None:
    """Empty source, no flush — yields nothing."""
    result = list(Chain([]).flat_iterate())
    self.assertEqual(result, [])

  async def test_flat_iterate_single_element(self) -> None:
    """Single element source with fn."""
    result = list(Chain([5]).flat_iterate(lambda x: [x, x + 1]))
    self.assertEqual(result, [5, 6])

  async def test_flat_iterate_reuse(self) -> None:
    """ChainIterator from flat_iterate is reusable via __call__."""
    it = Chain(lambda: None).flat_iterate(lambda x: [x, x * 10])
    result_a = list(it([1, 2]))
    self.assertEqual(result_a, [1, 10, 2, 20])
    result_b = list(it([3]))
    self.assertEqual(result_b, [3, 30])


# ---------------------------------------------------------------------------
# §5.8 Pending if_() checks for flat_iterate variants
# ---------------------------------------------------------------------------


class FlatIteratePendingIfTest(unittest.TestCase):
  """Pending if_() before flat_iterate/flat_iterate_do raises QuentException."""

  def test_flat_iterate_with_pending_if_raises(self) -> None:
    """flat_iterate() with a pending if_() raises QuentException."""
    with self.assertRaises(QuentException) as ctx:
      Chain([[1, 2], [3]]).if_(lambda x: True).flat_iterate()
    self.assertIn('if_() must be followed by .then() or .do()', str(ctx.exception))

  def test_flat_iterate_do_with_pending_if_raises(self) -> None:
    """flat_iterate_do() with a pending if_() raises QuentException."""
    with self.assertRaises(QuentException) as ctx:
      Chain([[1, 2], [3]]).if_(lambda x: True).flat_iterate_do()
    self.assertIn('if_() must be followed by .then() or .do()', str(ctx.exception))


if __name__ == '__main__':
  unittest.main()
