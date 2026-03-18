# SPDX-License-Identifier: MIT
"""Structural parity tests for _sync_generator / _async_generator mirrors.

These two functions in quent/_generator.py are explicit mirrors that must
produce identical result sequences for the same inputs.  Each test builds
ONE q, collects results via both sync iteration (list()) and async
iteration (async for), and asserts the output sequences are identical.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase

from quent import Q

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_collect(chain_iter: Any) -> list[Any]:
  """Collect all items from a QuentIterator via async for."""
  result: list[Any] = []
  async for item in chain_iter:
    result.append(item)
  return result


def _sync_collect(chain_iter: Any) -> list[Any]:
  """Collect all items from a QuentIterator via sync for."""
  return list(chain_iter)


# ---------------------------------------------------------------------------
# §1: Basic iteration with and without fn
# ---------------------------------------------------------------------------


class BasicIterationParityTests(IsolatedAsyncioTestCase):
  """Sync and async generators produce identical results for basic iterate()."""

  async def test_iterate_no_fn(self) -> None:
    """iterate() with no fn — sync and async yield identical elements."""
    it = Q(range(6)).iterate()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 2, 3, 4, 5])

  async def test_iterate_with_sync_fn(self) -> None:
    """iterate(fn) with sync fn — sync and async yield identical transformed elements."""
    fn = lambda x: x * 3
    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 3, 6, 9, 12])

  async def test_iterate_empty_source(self) -> None:
    """iterate() with empty source — both paths yield nothing."""
    it = Q([]).iterate()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [])

  async def test_iterate_single_element(self) -> None:
    """iterate() with single element source."""
    it = Q([42]).iterate()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [42])

  async def test_iterate_with_fn_empty_source(self) -> None:
    """iterate(fn) with empty source — fn never called, both yield nothing."""
    calls: list[Any] = []

    def tracking_fn(x: Any) -> Any:
      calls.append(x)
      return x * 2

    it = Q([]).iterate(tracking_fn)
    sync_result = _sync_collect(it)
    calls_after_sync = len(calls)
    async_result = await _async_collect(it)
    calls_after_async = len(calls)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [])
    # fn called 0 times in each path
    self.assertEqual(calls_after_sync, 0)
    self.assertEqual(calls_after_async, 0)

  async def test_iterate_do_no_fn(self) -> None:
    """iterate_do() with no fn — identical to iterate() without fn."""
    it = Q(range(4)).iterate_do()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 2, 3])

  async def test_iterate_do_with_fn(self) -> None:
    """iterate_do(fn) — fn runs as side-effect, original elements yielded identically."""
    sync_side: list[int] = []
    async_side: list[int] = []

    def sync_track(x: int) -> str:
      sync_side.append(x)
      return 'discarded'

    def async_track(x: int) -> str:
      async_side.append(x)
      return 'discarded'

    it_sync = Q(range(4)).iterate_do(sync_track)
    it_async = Q(range(4)).iterate_do(async_track)
    sync_result = _sync_collect(it_sync)
    async_result = await _async_collect(it_async)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 2, 3])
    # Both paths invoked fn with the same items
    self.assertEqual(sync_side, async_side)
    self.assertEqual(sync_side, [0, 1, 2, 3])

  async def test_iterate_string_elements(self) -> None:
    """iterate() with non-numeric elements — parity holds for any element type."""
    data = ['alpha', 'beta', 'gamma']
    it = Q(data).iterate()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, data)

  async def test_iterate_with_chain_steps_before(self) -> None:
    """iterate() on a pipeline with .then() steps — run phase is identical."""
    it = Q(5).then(lambda x: list(range(x))).iterate(lambda x: x + 10)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [10, 11, 12, 13, 14])


# ---------------------------------------------------------------------------
# §2: Flat mode (flat_iterate / flat_iterate_do)
# ---------------------------------------------------------------------------


class FlatIterateParityTests(IsolatedAsyncioTestCase):
  """Sync and async generators produce identical results for flat_iterate()."""

  async def test_flat_iterate_no_fn(self) -> None:
    """flat_iterate() with no fn — flattens one level identically."""
    it = Q([[1, 2], [3], [], [4, 5]]).flat_iterate()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2, 3, 4, 5])

  async def test_flat_iterate_with_fn(self) -> None:
    """flat_iterate(fn) — fn returns sub-iterable, items yielded identically."""
    fn = lambda x: [x, x * 10]
    it = Q(range(3)).flat_iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 0, 1, 10, 2, 20])

  async def test_flat_iterate_empty_source(self) -> None:
    """flat_iterate() with empty source — both yield nothing."""
    it = Q([]).flat_iterate()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [])

  async def test_flat_iterate_fn_returns_empty(self) -> None:
    """flat_iterate(fn) where fn returns empty for some items."""
    fn = lambda x: [] if x == 2 else [x]
    it = Q([1, 2, 3]).flat_iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 3])

  async def test_flat_iterate_do_with_fn(self) -> None:
    """flat_iterate_do(fn) — fn's iterable consumed as side-effect, originals yielded."""
    sync_consumed: list[int] = []
    async_consumed: list[int] = []

    def sync_capture(x: int) -> list[int]:
      items = [x * 10, x * 100]
      sync_consumed.extend(items)
      return items

    def async_capture(x: int) -> list[int]:
      items = [x * 10, x * 100]
      async_consumed.extend(items)
      return items

    it_sync = Q([1, 2, 3]).flat_iterate_do(sync_capture)
    it_async = Q([1, 2, 3]).flat_iterate_do(async_capture)
    sync_result = _sync_collect(it_sync)
    async_result = await _async_collect(it_async)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2, 3])
    self.assertEqual(sync_consumed, async_consumed)
    self.assertEqual(sync_consumed, [10, 100, 20, 200, 30, 300])

  async def test_flat_iterate_do_no_fn(self) -> None:
    """flat_iterate_do() with no fn — flattens one level identically."""
    it = Q([[1, 2], [3]]).flat_iterate_do()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2, 3])

  async def test_flat_iterate_string_flattening(self) -> None:
    """flat_iterate(fn) where fn returns strings — chars yielded identically."""
    fn = lambda x: x  # identity — strings are iterable
    it = Q(['ab', 'cd']).flat_iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, ['a', 'b', 'c', 'd'])


# ---------------------------------------------------------------------------
# §3: _Break / _Return signals (Q.break_() / Q.return_())
# ---------------------------------------------------------------------------


class ControlFlowSignalParityTests(IsolatedAsyncioTestCase):
  """Sync and async generators handle break_/return_ signals identically."""

  async def test_break_no_value(self) -> None:
    """break_() with no value — stops iteration, no extra yield."""

    def fn(x: int) -> int:
      if x == 3:
        return Q.break_()
      return x

    it = Q(range(10)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 2])

  async def test_break_with_value(self) -> None:
    """break_(v) — yields value then stops."""

    def fn(x: int) -> int:
      if x == 2:
        return Q.break_(x * 100)
      return x

    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 200])

  async def test_return_no_value(self) -> None:
    """return_() with no value — stops iteration, no extra yield."""

    def fn(x: int) -> int:
      if x == 2:
        return Q.return_()
      return x

    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1])

  async def test_return_with_value(self) -> None:
    """return_(v) — yields value then stops."""

    def fn(x: int) -> int:
      if x == 2:
        return Q.return_(x * 10)
      return x

    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 20])

  async def test_break_with_callable_value(self) -> None:
    """break_(callable) — callable is evaluated, result yielded then stop."""

    def fn(x: int) -> int:
      if x == 1:
        return Q.break_(lambda: 999)
      return x

    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 999])

  async def test_return_with_callable_value(self) -> None:
    """return_(callable) — callable is evaluated, result yielded then stop."""

    def fn(x: int) -> int:
      if x == 1:
        return Q.return_(lambda: 888)
      return x

    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 888])

  async def test_break_in_iterate_do(self) -> None:
    """break_() in iterate_do — original elements yielded before break."""

    def fn(x: int) -> int:
      if x == 3:
        return Q.break_()
      return x

    it = Q(range(10)).iterate_do(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 2])

  async def test_return_in_iterate_do_with_value(self) -> None:
    """return_(v) in iterate_do — originals yielded, then return value."""

    def fn(x: int) -> int:
      if x == 2:
        return Q.return_(x * 10)
      return x

    it = Q(range(5)).iterate_do(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 20])

  async def test_break_at_first_element(self) -> None:
    """break_() at the very first element — nothing yielded before break."""

    def fn(x: int) -> int:
      return Q.break_()

    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [])

  async def test_break_at_first_with_value(self) -> None:
    """break_(v) at the very first element — only break value yielded."""

    def fn(x: int) -> int:
      return Q.break_(77)

    it = Q(range(5)).iterate(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [77])


# ---------------------------------------------------------------------------
# §4: Deferred with_(fn) context manager
# ---------------------------------------------------------------------------


class DeferredWithParityTests(IsolatedAsyncioTestCase):
  """Sync and async generators handle deferred with_() identically."""

  async def test_deferred_with_iterate(self) -> None:
    """with_(fn).iterate() — CM stays open, fn result iterated identically."""

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

    # Sync path
    cm_sync = TrackingCM([10, 20, 30])
    it_sync = Q(cm_sync).with_(lambda ctx: ctx).iterate()
    sync_result = _sync_collect(it_sync)

    # Async path — fresh CM since the previous one was already used
    cm_async = TrackingCM([10, 20, 30])
    it_async = Q(cm_async).with_(lambda ctx: ctx).iterate()
    async_result = await _async_collect(it_async)

    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [10, 20, 30])
    # Both CMs were properly entered and exited
    self.assertTrue(cm_sync.entered)
    self.assertTrue(cm_sync.exited)
    self.assertTrue(cm_async.entered)
    self.assertTrue(cm_async.exited)

  async def test_deferred_with_iterate_with_fn(self) -> None:
    """with_(fn).iterate(transform) — CM open, fn transforms elements identically."""

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

    transform = lambda x: x * 2

    cm_sync = TrackingCM([1, 2, 3])
    it_sync = Q(cm_sync).with_(lambda ctx: ctx).iterate(transform)
    sync_result = _sync_collect(it_sync)

    cm_async = TrackingCM([1, 2, 3])
    it_async = Q(cm_async).with_(lambda ctx: ctx).iterate(transform)
    async_result = await _async_collect(it_async)

    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [2, 4, 6])

  async def test_deferred_with_flat_iterate(self) -> None:
    """with_(fn).flat_iterate(transform, flush=flush_fn) — CM + flat mode parity."""

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
    transform = lambda x: [x, x * 10]
    flush_fn = lambda: [99]

    cm_sync = TrackingCM(items)
    it_sync = Q(cm_sync).with_(lambda ctx: ctx).flat_iterate(transform, flush=flush_fn)
    sync_result = _sync_collect(it_sync)

    cm_async = TrackingCM(items)
    it_async = Q(cm_async).with_(lambda ctx: ctx).flat_iterate(transform, flush=flush_fn)
    async_result = await _async_collect(it_async)

    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 10, 2, 20, 3, 30, 99])
    self.assertTrue(cm_sync.entered and cm_sync.exited)
    self.assertTrue(cm_async.entered and cm_async.exited)

  async def test_deferred_with_do_iterate(self) -> None:
    """with_do(fn).iterate() — CM is itself iterated (with_do ignores fn result)."""

    class IterableCM:
      """CM that is itself iterable — for with_do where the CM becomes the iterable."""

      def __init__(self) -> None:
        self.items = [100, 200, 300]
        self.entered = False
        self.exited = False

      def __enter__(self) -> IterableCM:
        self.entered = True
        return self

      def __exit__(self, *args: Any) -> bool:
        self.exited = True
        return False

      def __iter__(self) -> Any:
        return iter(self.items)

    cm_sync = IterableCM()
    it_sync = Q(cm_sync).with_do(lambda ctx: 'ignored').iterate()
    sync_result = _sync_collect(it_sync)

    cm_async = IterableCM()
    it_async = Q(cm_async).with_do(lambda ctx: 'ignored').iterate()
    async_result = await _async_collect(it_async)

    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [100, 200, 300])
    self.assertTrue(cm_sync.entered and cm_sync.exited)
    self.assertTrue(cm_async.entered and cm_async.exited)


# ---------------------------------------------------------------------------
# §5: flush (on_exhaust) behavior
# ---------------------------------------------------------------------------


class FlushParityTests(IsolatedAsyncioTestCase):
  """Sync and async generators handle flush identically."""

  async def test_flat_iterate_with_flush(self) -> None:
    """flat_iterate(flush=fn) — flush output appended identically."""
    flush_fn = lambda: [98, 99]
    it = Q([[1], [2]]).flat_iterate(flush=flush_fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2, 98, 99])

  async def test_flat_iterate_fn_and_flush(self) -> None:
    """flat_iterate(fn, flush=fn) — both fn and flush, identical results."""
    transform = lambda x: [x, x * 10]
    flush_fn = lambda: [99]
    it = Q(range(3)).flat_iterate(transform, flush=flush_fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 0, 1, 10, 2, 20, 99])

  async def test_flat_iterate_empty_flush(self) -> None:
    """flush returns empty iterable — nothing extra yielded in either path."""
    it = Q([[1], [2]]).flat_iterate(flush=lambda: [])
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2])

  async def test_flat_iterate_empty_source_with_flush(self) -> None:
    """Empty source + flush — only flush output in both paths."""
    it = Q([]).flat_iterate(flush=lambda: [42])
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [42])

  async def test_flat_iterate_do_with_flush(self) -> None:
    """flat_iterate_do with flush — originals + flush output, identical."""
    sync_side: list[int] = []
    async_side: list[int] = []

    def sync_capture(x: int) -> list[int]:
      sync_side.append(x)
      return [x * 10]

    def async_capture(x: int) -> list[int]:
      async_side.append(x)
      return [x * 10]

    flush_fn = lambda: [88, 99]

    it_sync = Q([1, 2]).flat_iterate_do(sync_capture, flush=flush_fn)
    it_async = Q([1, 2]).flat_iterate_do(async_capture, flush=flush_fn)
    sync_result = _sync_collect(it_sync)
    async_result = await _async_collect(it_async)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2, 88, 99])
    self.assertEqual(sync_side, async_side)
    self.assertEqual(sync_side, [1, 2])

  async def test_flush_multiple_items(self) -> None:
    """flush returns many items — all yielded identically."""
    flush_fn = lambda: list(range(10, 20))
    it = Q([[1]]).flat_iterate(flush=flush_fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, *list(range(10, 20))])


# ---------------------------------------------------------------------------
# §6: ignore_result mode (iterate_do / flat_iterate_do)
# ---------------------------------------------------------------------------


class IgnoreResultParityTests(IsolatedAsyncioTestCase):
  """Sync and async generators handle ignore_result mode identically."""

  async def test_iterate_do_discards_fn_result(self) -> None:
    """iterate_do(fn) — fn result discarded, originals yielded identically."""
    fn = lambda x: x * 1000  # result should be discarded

    it = Q(range(5)).iterate_do(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 2, 3, 4])

  async def test_iterate_do_side_effect_parity(self) -> None:
    """iterate_do(fn) — side effects happen identically in both paths."""
    sync_effects: list[int] = []
    async_effects: list[int] = []

    def sync_effect(x: int) -> str:
      sync_effects.append(x * 2)
      return 'discarded'

    def async_effect(x: int) -> str:
      async_effects.append(x * 2)
      return 'discarded'

    it_sync = Q([10, 20, 30]).iterate_do(sync_effect)
    it_async = Q([10, 20, 30]).iterate_do(async_effect)
    sync_result = _sync_collect(it_sync)
    async_result = await _async_collect(it_async)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [10, 20, 30])
    self.assertEqual(sync_effects, async_effects)
    self.assertEqual(sync_effects, [20, 40, 60])

  async def test_flat_iterate_do_discards_sub_items(self) -> None:
    """flat_iterate_do(fn) — fn's sub-items consumed but not yielded."""
    fn = lambda x: [x * 10, x * 100]

    it_sync = Q([1, 2, 3]).flat_iterate_do(fn)
    it_async = Q([1, 2, 3]).flat_iterate_do(fn)
    sync_result = _sync_collect(it_sync)
    async_result = await _async_collect(it_async)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2, 3])

  async def test_iterate_do_with_break(self) -> None:
    """iterate_do + break_() — originals yielded before break, identical paths."""

    def fn(x: int) -> int:
      if x == 2:
        return Q.break_(x * 100)
      return x

    it = Q(range(5)).iterate_do(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    # Items 0, 1 are yielded as originals, then break value 200
    self.assertEqual(sync_result, [0, 1, 200])

  async def test_iterate_do_with_none_fn(self) -> None:
    """iterate_do(fn=None) — no fn, elements yielded as-is in both paths."""
    it = Q([7, 8, 9]).iterate_do()
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [7, 8, 9])

  async def test_flat_iterate_do_with_break(self) -> None:
    """flat_iterate_do + break_() — original elements yielded, break stops."""

    def fn(x: int) -> list[int]:
      if x == 2:
        return Q.break_()
      return [x * 10]

    it = Q([1, 2, 3]).flat_iterate_do(fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1])


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------


class CombinedParityTests(IsolatedAsyncioTestCase):
  """Combined scenarios exercising multiple features together."""

  async def test_deferred_with_and_break(self) -> None:
    """with_(fn).iterate(fn) with break_() — CM cleanup + control flow parity."""

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

    def fn(x: int) -> int:
      if x == 3:
        return Q.break_(x * 100)
      return x

    cm_sync = TrackingCM(range(10))
    it_sync = Q(cm_sync).with_(lambda ctx: ctx).iterate(fn)
    sync_result = _sync_collect(it_sync)

    cm_async = TrackingCM(range(10))
    it_async = Q(cm_async).with_(lambda ctx: ctx).iterate(fn)
    async_result = await _async_collect(it_async)

    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [0, 1, 2, 300])
    # CM properly cleaned up in both paths
    self.assertTrue(cm_sync.entered and cm_sync.exited)
    self.assertTrue(cm_async.entered and cm_async.exited)

  async def test_deferred_with_and_finally(self) -> None:
    """with_(fn).iterate() + finally_() — both deferred, cleanup parity."""
    sync_cleanup: list[str] = []
    async_cleanup: list[str] = []

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

    cm_sync = TrackingCM([1, 2, 3])
    it_sync = Q(cm_sync).with_(lambda ctx: ctx).finally_(lambda rv: sync_cleanup.append('done')).iterate()
    sync_result = _sync_collect(it_sync)

    cm_async = TrackingCM([1, 2, 3])
    it_async = Q(cm_async).with_(lambda ctx: ctx).finally_(lambda rv: async_cleanup.append('done')).iterate()
    async_result = await _async_collect(it_async)

    self.assertEqual(sync_result, async_result)
    self.assertEqual(sync_result, [1, 2, 3])
    # Both cleanup paths ran
    self.assertEqual(sync_cleanup, ['done'])
    self.assertEqual(async_cleanup, ['done'])
    self.assertTrue(cm_sync.exited and cm_async.exited)

  async def test_flat_iterate_with_flush_and_break(self) -> None:
    """flat_iterate(fn, flush=fn) + break_() — break before flush, flush skipped."""

    def fn(x: int) -> list[int]:
      if x == 2:
        return Q.break_(999)
      return [x, x * 10]

    flush_fn = lambda: [77, 88]  # Should NOT be reached due to break
    it = Q(range(5)).flat_iterate(fn, flush=flush_fn)
    sync_result = _sync_collect(it)
    async_result = await _async_collect(it)
    self.assertEqual(sync_result, async_result)
    # break_(999) halts during fn processing of x=2
    # x=0 yields [0, 0], x=1 yields [1, 10], x=2 triggers break with 999
    self.assertEqual(sync_result, [0, 0, 1, 10, 999])

  async def test_reuse_parity(self) -> None:
    """Iterator reuse via __call__ — both paths produce identical sequences."""
    it = Q().then(lambda x: x).iterate(lambda x: x * 2)
    # Use it with different root values
    sync_a = _sync_collect(it([1, 2, 3]))
    async_a = await _async_collect(it([1, 2, 3]))
    self.assertEqual(sync_a, async_a)
    self.assertEqual(sync_a, [2, 4, 6])

    sync_b = _sync_collect(it([10, 20]))
    async_b = await _async_collect(it([10, 20]))
    self.assertEqual(sync_b, async_b)
    self.assertEqual(sync_b, [20, 40])


if __name__ == '__main__':
  unittest.main()
