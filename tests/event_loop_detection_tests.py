# SPDX-License-Identifier: MIT
"""Tests for event loop detection — §16.10 Dual-Protocol Objects Prefer Async.

SPEC §16.10 states:
  Loop detection covers asyncio, trio, and curio without importing them —
  detection uses ``sys.modules`` lookups (~50ns dict get when a library is
  not loaded), so there is zero overhead when those libraries are absent.

These tests verify that ``_has_running_loop()`` correctly detects asyncio,
trio, and curio event loops, and that ``_should_use_async_protocol()``
returns the correct protocol choice under each runtime.
"""

from __future__ import annotations

import unittest

from quent._eval import _has_running_loop, _should_use_async_protocol

# ---------------------------------------------------------------------------
# Optional imports — tests are skipped when the library is not installed.
# ---------------------------------------------------------------------------

try:
  import trio

  _has_trio = True
except ImportError:
  _has_trio = False

try:
  import curio

  _has_curio = True
except (ImportError, AttributeError):
  # AttributeError: curio < 1.7 fails on Python 3.14+ due to removed
  # multiprocessing.connection.CHALLENGE constant.
  _has_curio = False


# ---------------------------------------------------------------------------
# Dual-protocol test helpers
# ---------------------------------------------------------------------------


class DualProtocolCM:
  """Context manager supporting both sync and async protocols."""

  def __enter__(self) -> str:
    return 'sync'

  def __exit__(self, *args: object) -> bool:
    return False

  async def __aenter__(self) -> str:
    return 'async'

  async def __aexit__(self, *args: object) -> bool:
    return False


class SyncOnlyCM:
  """Context manager supporting only the sync protocol."""

  def __enter__(self) -> str:
    return 'sync'

  def __exit__(self, *args: object) -> bool:
    return False


class AsyncOnlyCM:
  """Context manager supporting only the async protocol."""

  async def __aenter__(self) -> str:
    return 'async'

  async def __aexit__(self, *args: object) -> bool:
    return False


class DualProtocolIterable:
  """Iterable supporting both sync and async protocols."""

  def __iter__(self) -> DualProtocolIterable:
    return self

  def __next__(self) -> int:
    raise StopIteration

  def __aiter__(self) -> DualProtocolIterable:
    return self

  async def __anext__(self) -> int:
    raise StopAsyncIteration


# ---------------------------------------------------------------------------
# §16.10 — _has_running_loop() detection
# ---------------------------------------------------------------------------


class NoLoopDetectionTest(unittest.TestCase):
  """_has_running_loop() returns False when no event loop is running."""

  def test_no_loop_returns_false(self) -> None:
    self.assertFalse(_has_running_loop())


class AsyncioLoopDetectionTest(unittest.IsolatedAsyncioTestCase):
  """_has_running_loop() returns True inside an asyncio event loop."""

  async def test_asyncio_loop_detected(self) -> None:
    self.assertTrue(_has_running_loop())


@unittest.skipUnless(_has_trio, 'trio not installed')
class TrioLoopDetectionTest(unittest.TestCase):
  """_has_running_loop() returns True inside a trio event loop."""

  def test_trio_loop_detected(self) -> None:
    async def check() -> None:
      self.assertTrue(_has_running_loop())

    trio.run(check)

  def test_trio_loop_not_detected_outside(self) -> None:
    """After trio.run() completes, the loop is no longer running."""

    async def noop() -> None:
      pass

    trio.run(noop)
    self.assertFalse(_has_running_loop())


@unittest.skipUnless(_has_curio, 'curio not installed')
class CurioLoopDetectionTest(unittest.TestCase):
  """_has_running_loop() returns True inside a curio event loop."""

  def test_curio_loop_detected(self) -> None:
    async def check() -> None:
      self.assertTrue(_has_running_loop())

    curio.run(check)

  def test_curio_loop_not_detected_outside(self) -> None:
    """After curio.run() completes, the loop is no longer running."""

    async def noop() -> None:
      pass

    curio.run(noop)
    self.assertFalse(_has_running_loop())


# ---------------------------------------------------------------------------
# §16.10 — _should_use_async_protocol() under each runtime
# ---------------------------------------------------------------------------


class ShouldUseAsyncProtocolNoLoopTest(unittest.TestCase):
  """_should_use_async_protocol() behavior when no event loop is running."""

  def test_dual_protocol_cm_returns_false(self) -> None:
    """Both protocols present, no loop → sync (False)."""
    result = _should_use_async_protocol(DualProtocolCM(), '__enter__', '__aenter__')
    self.assertFalse(result)

  def test_sync_only_cm_returns_false(self) -> None:
    """Only sync protocol → False."""
    result = _should_use_async_protocol(SyncOnlyCM(), '__enter__', '__aenter__')
    self.assertFalse(result)

  def test_async_only_cm_returns_true(self) -> None:
    """Only async protocol → True (regardless of loop)."""
    result = _should_use_async_protocol(AsyncOnlyCM(), '__enter__', '__aenter__')
    self.assertTrue(result)

  def test_neither_protocol_returns_none(self) -> None:
    """Neither protocol → None."""
    result = _should_use_async_protocol(object(), '__enter__', '__aenter__')
    self.assertIsNone(result)

  def test_dual_protocol_iterable_returns_false(self) -> None:
    """Both iter protocols present, no loop → sync (False)."""
    result = _should_use_async_protocol(DualProtocolIterable(), '__iter__', '__aiter__')
    self.assertFalse(result)


class ShouldUseAsyncProtocolAsyncioTest(unittest.IsolatedAsyncioTestCase):
  """_should_use_async_protocol() behavior inside an asyncio event loop."""

  async def test_dual_protocol_cm_returns_true(self) -> None:
    """Both protocols present, asyncio loop running → async (True)."""
    result = _should_use_async_protocol(DualProtocolCM(), '__enter__', '__aenter__')
    self.assertTrue(result)

  async def test_sync_only_cm_returns_false(self) -> None:
    """Only sync protocol → False (regardless of loop)."""
    result = _should_use_async_protocol(SyncOnlyCM(), '__enter__', '__aenter__')
    self.assertFalse(result)

  async def test_async_only_cm_returns_true(self) -> None:
    """Only async protocol → True."""
    result = _should_use_async_protocol(AsyncOnlyCM(), '__enter__', '__aenter__')
    self.assertTrue(result)

  async def test_neither_protocol_returns_none(self) -> None:
    """Neither protocol → None."""
    result = _should_use_async_protocol(object(), '__enter__', '__aenter__')
    self.assertIsNone(result)

  async def test_dual_protocol_iterable_returns_true(self) -> None:
    """Both iter protocols present, asyncio loop running → async (True)."""
    result = _should_use_async_protocol(DualProtocolIterable(), '__iter__', '__aiter__')
    self.assertTrue(result)


@unittest.skipUnless(_has_trio, 'trio not installed')
class ShouldUseAsyncProtocolTrioTest(unittest.TestCase):
  """_should_use_async_protocol() behavior inside a trio event loop."""

  def test_dual_protocol_cm_returns_true(self) -> None:
    """Both protocols present, trio loop running → async (True)."""

    async def check() -> None:
      result = _should_use_async_protocol(DualProtocolCM(), '__enter__', '__aenter__')
      self.assertTrue(result)

    trio.run(check)

  def test_dual_protocol_iterable_returns_true(self) -> None:
    """Both iter protocols present, trio loop running → async (True)."""

    async def check() -> None:
      result = _should_use_async_protocol(DualProtocolIterable(), '__iter__', '__aiter__')
      self.assertTrue(result)

    trio.run(check)

  def test_sync_only_cm_returns_false(self) -> None:
    """Only sync protocol → False (regardless of loop)."""

    async def check() -> None:
      result = _should_use_async_protocol(SyncOnlyCM(), '__enter__', '__aenter__')
      self.assertFalse(result)

    trio.run(check)

  def test_async_only_cm_returns_true(self) -> None:
    """Only async protocol → True."""

    async def check() -> None:
      result = _should_use_async_protocol(AsyncOnlyCM(), '__enter__', '__aenter__')
      self.assertTrue(result)

    trio.run(check)


@unittest.skipUnless(_has_curio, 'curio not installed')
class ShouldUseAsyncProtocolCurioTest(unittest.TestCase):
  """_should_use_async_protocol() behavior inside a curio event loop."""

  def test_dual_protocol_cm_returns_true(self) -> None:
    """Both protocols present, curio loop running → async (True)."""

    async def check() -> None:
      result = _should_use_async_protocol(DualProtocolCM(), '__enter__', '__aenter__')
      self.assertTrue(result)

    curio.run(check)

  def test_dual_protocol_iterable_returns_true(self) -> None:
    """Both iter protocols present, curio loop running → async (True)."""

    async def check() -> None:
      result = _should_use_async_protocol(DualProtocolIterable(), '__iter__', '__aiter__')
      self.assertTrue(result)

    curio.run(check)

  def test_sync_only_cm_returns_false(self) -> None:
    """Only sync protocol → False (regardless of loop)."""

    async def check() -> None:
      result = _should_use_async_protocol(SyncOnlyCM(), '__enter__', '__aenter__')
      self.assertFalse(result)

    curio.run(check)

  def test_async_only_cm_returns_true(self) -> None:
    """Only async protocol → True."""

    async def check() -> None:
      result = _should_use_async_protocol(AsyncOnlyCM(), '__enter__', '__aenter__')
      self.assertTrue(result)

    curio.run(check)


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


class TrackingDualCM:
  """Dual-protocol context manager that records which protocol was entered."""

  def __init__(self, value: object) -> None:
    self.value = value
    self.used_async = False
    self.used_sync = False

  def __enter__(self) -> object:
    self.used_sync = True
    return self.value

  def __exit__(self, *args: object) -> bool:
    return False

  async def __aenter__(self) -> object:
    self.used_async = True
    return self.value

  async def __aexit__(self, *args: object) -> bool:
    return False


class TrackingDualIterable:
  """Dual-protocol iterable that records which protocol was used."""

  def __init__(self, items: list[int]) -> None:
    self._items = items
    self.used_async = False
    self.used_sync = False

  def __iter__(self) -> TrackingDualIterable:
    self.used_sync = True
    return self

  def __next__(self) -> int:
    raise StopIteration

  def __aiter__(self) -> TrackingDualIterable:
    self.used_async = True
    return self

  async def __anext__(self) -> int:
    raise StopAsyncIteration


# ---------------------------------------------------------------------------
# §16.10 — Full pipeline integration tests under trio
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_trio, 'trio not installed')
class TrioPipelineTests(unittest.TestCase):
  """Full pipeline execution under trio verifies dual-protocol objects use async."""

  def test_dual_cm_uses_async_protocol_under_trio(self) -> None:
    """Chain(dual_cm).with_(fn).run() inside trio.run() uses __aenter__, not __enter__."""
    from quent import Chain

    cm = TrackingDualCM('hello')
    result_holder: list[object] = []

    async def check() -> None:
      result = await Chain(cm).with_(lambda v: v).run()
      result_holder.append(result)

    trio.run(check)
    self.assertTrue(cm.used_async, 'Expected __aenter__ to be called under trio')
    self.assertFalse(cm.used_sync, 'Expected __enter__ NOT to be called under trio')
    self.assertEqual(result_holder, ['hello'])

  def test_basic_async_chain_under_trio(self) -> None:
    """Chain(5).then(async_fn).run() works inside trio.run() and returns correct result."""
    from quent import Chain

    result_holder: list[object] = []

    async def async_double(x: int) -> int:
      return x * 2

    async def check() -> None:
      result = await Chain(5).then(async_double).run()
      result_holder.append(result)

    trio.run(check)
    self.assertEqual(result_holder, [10])

  def test_dual_iterable_uses_async_protocol_under_trio(self) -> None:
    """Chain(dual_iterable).foreach(fn).run() inside trio.run() uses __aiter__."""
    from quent import Chain

    iterable = TrackingDualIterable([])
    result_holder: list[object] = []

    async def check() -> None:
      result = await Chain(iterable).foreach(lambda x: x).run()
      result_holder.append(result)

    trio.run(check)
    self.assertTrue(iterable.used_async, 'Expected __aiter__ to be called under trio')
    self.assertFalse(iterable.used_sync, 'Expected __iter__ NOT to be called under trio')
    self.assertEqual(result_holder, [[]])


# ---------------------------------------------------------------------------
# §16.10 — Full pipeline integration tests under curio
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_curio, 'curio not installed')
class CurioPipelineTests(unittest.TestCase):
  """Full pipeline execution under curio verifies dual-protocol objects use async."""

  def test_dual_cm_uses_async_protocol_under_curio(self) -> None:
    """Chain(dual_cm).with_(fn).run() inside curio.run() uses __aenter__, not __enter__."""
    from quent import Chain

    cm = TrackingDualCM('hello')
    result_holder: list[object] = []

    async def check() -> None:
      result = await Chain(cm).with_(lambda v: v).run()
      result_holder.append(result)

    curio.run(check)
    self.assertTrue(cm.used_async, 'Expected __aenter__ to be called under curio')
    self.assertFalse(cm.used_sync, 'Expected __enter__ NOT to be called under curio')
    self.assertEqual(result_holder, ['hello'])

  def test_basic_async_chain_under_curio(self) -> None:
    """Chain(5).then(async_fn).run() works inside curio.run() and returns correct result."""
    from quent import Chain

    result_holder: list[object] = []

    async def async_double(x: int) -> int:
      return x * 2

    async def check() -> None:
      result = await Chain(5).then(async_double).run()
      result_holder.append(result)

    curio.run(check)
    self.assertEqual(result_holder, [10])

  def test_dual_iterable_uses_async_protocol_under_curio(self) -> None:
    """Chain(dual_iterable).foreach(fn).run() inside curio.run() uses __aiter__."""
    from quent import Chain

    iterable = TrackingDualIterable([])
    result_holder: list[object] = []

    async def check() -> None:
      result = await Chain(iterable).foreach(lambda x: x).run()
      result_holder.append(result)

    curio.run(check)
    self.assertTrue(iterable.used_async, 'Expected __aiter__ to be called under curio')
    self.assertFalse(iterable.used_sync, 'Expected __iter__ NOT to be called under curio')
    self.assertEqual(result_holder, [[]])


if __name__ == '__main__':
  unittest.main()
