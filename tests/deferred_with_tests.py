# SPDX-License-Identifier: MIT
"""Tests for deferred with_ in iterate / iterate_do / flat_iterate / flat_iterate_do."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase

from quent import Q
from tests.fixtures import SyncCM, async_double, sync_double
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# Tracking context managers for lifecycle verification
# ---------------------------------------------------------------------------


class TrackingCM:
  """Sync CM that records enter/exit lifecycle events."""

  def __init__(self, value: Any) -> None:
    self.value = value
    self.entered = False
    self.exited = False
    self.exit_args: tuple[Any, ...] | None = None

  def __enter__(self) -> Any:
    self.entered = True
    return self.value

  def __exit__(self, *args: Any) -> bool:
    self.exited = True
    self.exit_args = args
    return False


class AsyncTrackingCM:
  """Async CM that records enter/exit lifecycle events."""

  def __init__(self, value: Any) -> None:
    self.value = value
    self.entered = False
    self.exited = False
    self.exit_args: tuple[Any, ...] | None = None

  async def __aenter__(self) -> Any:
    self.entered = True
    return self.value

  async def __aexit__(self, *args: Any) -> bool:
    self.exited = True
    self.exit_args = args
    return False


class TrackingSuppressCM:
  """Sync tracking CM that suppresses exceptions."""

  def __init__(self, value: Any) -> None:
    self.value = value
    self.entered = False
    self.exited = False
    self.exit_args: tuple[Any, ...] | None = None

  def __enter__(self) -> Any:
    self.entered = True
    return self.value

  def __exit__(self, *args: Any) -> bool:
    self.exited = True
    self.exit_args = args
    return True  # suppress


class AsyncTrackingSuppressCM:
  """Async tracking CM that suppresses exceptions."""

  def __init__(self, value: Any) -> None:
    self.value = value
    self.entered = False
    self.exited = False
    self.exit_args: tuple[Any, ...] | None = None

  async def __aenter__(self) -> Any:
    self.entered = True
    return self.value

  async def __aexit__(self, *args: Any) -> bool:
    self.exited = True
    self.exit_args = args
    return True  # suppress


class DualProtocolTrackingCM:
  """Dual-protocol CM that tracks which protocol was used."""

  def __init__(self, value: Any) -> None:
    self.value = value
    self.sync_entered = False
    self.sync_exited = False
    self.async_entered = False
    self.async_exited = False

  def __enter__(self) -> Any:
    self.sync_entered = True
    return self.value

  def __exit__(self, *args: Any) -> bool:
    self.sync_exited = True
    return False

  async def __aenter__(self) -> Any:
    self.async_entered = True
    return self.value

  async def __aexit__(self, *args: Any) -> bool:
    self.async_exited = True
    return False


# ---------------------------------------------------------------------------
# §9 — Deferred with_ core behavior
# ---------------------------------------------------------------------------


class DeferredWithIterateTests(SymmetricTestCase):
  """Deferred with_ + iterate: CM stays open during iteration."""

  async def test_with_iterate_sync(self) -> None:
    """Q(SyncCM(items)).with_(lambda ctx: ctx).iterate(fn) — CM wraps iteration."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).iterate(sync_double))
    self.assertEqual(result, [2, 4, 6])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    # Clean exit — no exception info passed
    self.assertEqual(cm.exit_args, (None, None, None))

  async def test_with_iterate_async(self) -> None:
    """Async CM + async for — CM wraps iteration."""
    items = [1, 2, 3]
    cm = AsyncTrackingCM(items)
    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate(async_double):
      result.append(item)
    self.assertEqual(result, [2, 4, 6])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  async def test_with_do_iterate_sync(self) -> None:
    """with_do: CM is the iterable (not inner fn result), fn is side-effect."""
    # with_do means: inner fn runs as side-effect, result is the CM itself
    # So the CM must be iterable.
    side_effects: list[Any] = []

    class IterableCM:
      def __init__(self, items: list[int]) -> None:
        self.items = items
        self.entered = False
        self.exited = False

      def __enter__(self) -> list[int]:
        self.entered = True
        return self.items

      def __exit__(self, *args: Any) -> bool:
        self.exited = True
        return False

      def __iter__(self) -> Any:
        return iter(self.items)

    cm = IterableCM([10, 20, 30])
    result = list(Q(cm).with_do(lambda ctx: side_effects.append(ctx)).iterate(sync_double))
    # with_do: ignore_result=True in _WithOp => the iterable is cm (original value)
    # The CM object itself is iterated, so items are [10, 20, 30]
    self.assertEqual(result, [20, 40, 60])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_iterate_no_fn(self) -> None:
    """with_ + iterate() with no fn — yields items directly."""
    items = [5, 10, 15]
    cm = TrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).iterate())
    self.assertEqual(result, [5, 10, 15])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_iterate_do(self) -> None:
    """with_ + iterate_do — fn runs as side-effect, yields original items."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    side_effects: list[int] = []
    result = list(Q(cm).with_(lambda ctx: ctx).iterate_do(lambda x: side_effects.append(x * 10)))
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(side_effects, [10, 20, 30])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_flat_iterate(self) -> None:
    """with_ + flat_iterate — deferred with_ + flatmap."""
    items = [[1, 2], [3], [], [4, 5]]
    cm = TrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).flat_iterate())
    self.assertEqual(result, [1, 2, 3, 4, 5])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_flat_iterate_do(self) -> None:
    """with_ + flat_iterate_do — fn runs as side-effect, yields original items."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    consumed: list[Any] = []
    result = list(Q(cm).with_(lambda ctx: ctx).flat_iterate_do(lambda x: consumed.append(x) or [x * 10, x * 100]))
    # flat_iterate_do: fn result iterable is consumed (side-effect) but original items yielded
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(consumed, [1, 2, 3])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# CM lifecycle verification
# ---------------------------------------------------------------------------


class DeferredWithLifecycleTests(SymmetricTestCase):
  """Verify CM enter/exit timing and exception handling."""

  async def test_cm_enters_at_iteration_start(self) -> None:
    """CM enter happens when iteration begins, not when iterate() is called."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    it = Q(cm).with_(lambda ctx: ctx).iterate()
    # iterate() returns QuentIterator — CM not entered yet
    self.assertFalse(cm.entered)
    result = list(it)
    # Now CM should have been entered and exited
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    self.assertEqual(result, [1, 2, 3])

  async def test_cm_exits_after_iteration_complete(self) -> None:
    """CM exit happens after all items consumed."""
    items = [10, 20]
    cm = TrackingCM(items)
    gen = iter(Q(cm).with_(lambda ctx: ctx).iterate())
    # First next — starts iteration, enters CM
    self.assertEqual(next(gen), 10)
    self.assertTrue(cm.entered)
    self.assertFalse(cm.exited)
    # Second next — still iterating
    self.assertEqual(next(gen), 20)
    self.assertFalse(cm.exited)
    # Exhaust — StopIteration triggers CM exit
    with self.assertRaises(StopIteration):
      next(gen)
    self.assertTrue(cm.exited)

  async def test_cm_exits_on_exception(self) -> None:
    """Exception during fn(item) — CM.__exit__ receives exception info."""
    items = [1, 2, 3]
    cm = TrackingCM(items)

    def boom(x: int) -> int:
      if x == 2:
        raise ValueError('boom at 2')
      return x

    with self.assertRaises(ValueError):
      list(Q(cm).with_(lambda ctx: ctx).iterate(boom))
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    # __exit__ should receive the exception info
    assert cm.exit_args is not None
    self.assertIs(cm.exit_args[0], ValueError)
    self.assertIsInstance(cm.exit_args[1], ValueError)

  async def test_cm_suppresses_exception(self) -> None:
    """CM.__exit__ returns True — generator stops cleanly, no exception."""
    items = [1, 2, 3]
    cm = TrackingSuppressCM(items)

    def boom(x: int) -> int:
      if x == 2:
        raise ValueError('boom')
      return x

    # The suppressing CM catches the exception — generator stops cleanly
    result = list(Q(cm).with_(lambda ctx: ctx).iterate(boom))
    # Items yielded before the exception: [1] (item 2 raised)
    self.assertEqual(result, [1])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_cm_exits_on_break(self) -> None:
    """_Break during iteration — CM exits cleanly."""
    items = [1, 2, 3, 4, 5]
    cm = TrackingCM(items)
    result: list[int] = []
    for item in Q(cm).with_(lambda ctx: ctx).iterate():
      if item == 3:
        break
      result.append(item)
    self.assertEqual(result, [1, 2])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_cm_exits_on_close(self) -> None:
    """Generator closed (.close()) — CM exits cleanly."""
    items = [1, 2, 3, 4, 5]
    cm = TrackingCM(items)
    gen = iter(Q(cm).with_(lambda ctx: ctx).iterate())
    self.assertEqual(next(gen), 1)
    self.assertTrue(cm.entered)
    gen.close()
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# Async lifecycle tests
# ---------------------------------------------------------------------------


class DeferredWithAsyncLifecycleTests(IsolatedAsyncioTestCase):
  """Async-specific lifecycle tests for deferred with_."""

  async def test_async_cm_exits_on_exception(self) -> None:
    """Async CM receives exception info on __aexit__."""
    items = [1, 2, 3]
    cm = AsyncTrackingCM(items)

    def boom(x: int) -> int:
      if x == 2:
        raise ValueError('async boom')
      return x

    with self.assertRaises(ValueError):
      async for _ in Q(cm).with_(lambda ctx: ctx).iterate(boom):
        pass
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    assert cm.exit_args is not None
    self.assertIs(cm.exit_args[0], ValueError)

  async def test_async_cm_suppresses_exception(self) -> None:
    """Async CM that suppresses — generator stops cleanly."""
    items = [1, 2, 3]
    cm = AsyncTrackingSuppressCM(items)

    def boom(x: int) -> int:
      if x == 2:
        raise ValueError('suppressed')
      return x

    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate(boom):
      result.append(item)
    self.assertEqual(result, [1])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_cm_exits_on_break(self) -> None:
    """Async for with break — CM exits cleanly."""
    items = [1, 2, 3, 4]
    cm = AsyncTrackingCM(items)
    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate():
      if item == 3:
        break
      result.append(item)
    self.assertEqual(result, [1, 2])
    self.assertTrue(cm.entered)
    # The async generator's aclose() is triggered by break, but the cleanup
    # coroutine (which awaits __aexit__) runs on a subsequent event loop tick.
    # Multiple yields may be needed for Python's async generator finalization.
    for _ in range(5):
      await asyncio.sleep(0)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class DeferredWithEdgeCaseTests(SymmetricTestCase):
  """Edge cases for deferred with_ detection and behavior."""

  async def test_with_not_last_link(self) -> None:
    """with_ NOT the last link — normal behavior (not deferred)."""
    # When with_ is not the last link, it executes during chain_run normally.
    cm = SyncCM([1, 2, 3])
    # .then(list) after with_ means with_ is NOT the last link
    result = Q(cm).with_(lambda ctx: ctx).then(list).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_with_iterate_reuse(self) -> None:
    """QuentIterator.__call__ returns new iterator with same deferred_with."""
    items_a = [1, 2, 3]
    items_b = [10, 20]

    it = Q(lambda: None).with_(lambda ctx: ctx).iterate(sync_double)
    # Reuse with different input
    result_a = list(it(TrackingCM(items_a)))
    self.assertEqual(result_a, [2, 4, 6])

    result_b = list(it(TrackingCM(items_b)))
    self.assertEqual(result_b, [20, 40])

  async def test_finally_and_with_both_deferred(self) -> None:
    """finally_() + with_ + iterate — both deferred, CM exits first then finally runs."""
    items = [1, 2, 3]
    TrackingCM(items)
    order: list[str] = []

    def cleanup(rv: Any) -> None:
      order.append('finally')

    # Wrap cm in a tracking CM that records order
    class OrderTrackingCM:
      def __init__(self, value: Any) -> None:
        self.value = value
        self.entered = False
        self.exited = False

      def __enter__(self) -> Any:
        self.entered = True
        return self.value

      def __exit__(self, *args: Any) -> bool:
        self.exited = True
        order.append('cm_exit')
        return False

    order_cm = OrderTrackingCM(items)
    result = list(Q(order_cm).finally_(cleanup).with_(lambda ctx: ctx).iterate())
    self.assertEqual(result, [1, 2, 3])
    self.assertTrue(order_cm.entered)
    self.assertTrue(order_cm.exited)
    # CM exits first, then deferred finally runs
    self.assertEqual(order, ['cm_exit', 'finally'])


# ---------------------------------------------------------------------------
# Async CM protocol detection
# ---------------------------------------------------------------------------


class DeferredWithProtocolTests(IsolatedAsyncioTestCase):
  """Test async CM protocol selection in deferred with_."""

  async def test_async_cm_in_async_iterate(self) -> None:
    """Async CM uses __aenter__/__aexit__ in async for."""
    items = [10, 20, 30]
    cm = AsyncTrackingCM(items)
    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate():
      result.append(item)
    self.assertEqual(result, [10, 20, 30])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_dual_protocol_cm_async(self) -> None:
    """Dual-protocol CM uses async protocol in async for."""
    items = [1, 2, 3]
    cm = DualProtocolTrackingCM(items)
    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3])
    # In async context, the async protocol should be preferred
    self.assertTrue(cm.async_entered)
    self.assertTrue(cm.async_exited)
    self.assertFalse(cm.sync_entered)
    self.assertFalse(cm.sync_exited)

  async def test_dual_protocol_cm_sync(self) -> None:
    """Dual-protocol CM uses sync protocol in sync for."""
    items = [1, 2, 3]
    cm = DualProtocolTrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).iterate())
    self.assertEqual(result, [1, 2, 3])
    # In sync context, the sync protocol should be used
    self.assertTrue(cm.sync_entered)
    self.assertTrue(cm.sync_exited)
    self.assertFalse(cm.async_entered)
    self.assertFalse(cm.async_exited)

  async def test_async_only_cm_sync_iteration_raises_typeerror(self) -> None:
    """§9.7: Async-only CM with sync iteration raises TypeError.

    Sync iteration (`for`): only `__enter__`/`__exit__` is used.
    If the CM only supports async protocol, a TypeError is raised
    directing the user to use `async for`.
    """

    class AsyncOnlyCM:
      """CM that only supports async protocol — no __enter__/__exit__."""

      def __init__(self, value: Any) -> None:
        self.value = value

      async def __aenter__(self) -> Any:
        return self.value

      async def __aexit__(self, *args: Any) -> bool:
        return False

    cm = AsyncOnlyCM([1, 2, 3])
    with self.assertRaises(TypeError) as ctx:
      list(Q(cm).with_(lambda val: val).iterate())
    self.assertIn('async for', str(ctx.exception))


# ---------------------------------------------------------------------------
# Integration: deferred with_ + flat_iterate
# ---------------------------------------------------------------------------


class DeferredWithFlatIterateTests(SymmetricTestCase):
  """Deferred with_ combined with flat_iterate variants."""

  async def test_with_flat_iterate_with_fn(self) -> None:
    """Deferred with_ + flat_iterate(fn) — CM wraps flatmap iteration."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).flat_iterate(lambda x: [x, x * 10]))
    self.assertEqual(result, [1, 10, 2, 20, 3, 30])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_flat_iterate_with_flush(self) -> None:
    """Deferred with_ + flat_iterate with flush — CM stays open through flush."""
    items = [1, 2]
    cm = TrackingCM(items)
    result = list(
      Q(cm)
      .with_(lambda ctx: ctx)
      .flat_iterate(
        lambda x: [x * 2],
        flush=lambda: [99, 100],
      )
    )
    self.assertEqual(result, [2, 4, 99, 100])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# Lifecycle contrast: .with_().then() vs .with_().iterate()
# ---------------------------------------------------------------------------


class WithLifecycleContrastTests(SymmetricTestCase):
  """Verify CM lifecycle differs between non-deferred and deferred paths."""

  async def test_with_then_cm_exits_before_then(self) -> None:
    """For .with_(fn).then(f), CM is already __exit__'d when f runs."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    exited_during_then: bool | None = None

    def check_exit(val: Any) -> Any:
      nonlocal exited_during_then
      exited_during_then = cm.exited
      return val

    Q(cm).with_(lambda ctx: ctx).then(check_exit).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    # CM was already exited when then() ran
    self.assertTrue(exited_during_then)

  async def test_with_iterate_cm_open_during_iteration(self) -> None:
    """For .with_(fn).iterate(f), CM is still open while f runs on each item."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    cm_open_during_iteration: list[bool] = []

    def check_open(item: int) -> int:
      cm_open_during_iteration.append(cm.entered and not cm.exited)
      return item * 2

    result = list(Q(cm).with_(lambda ctx: ctx).iterate(check_open))
    self.assertEqual(result, [2, 4, 6])
    # CM was open during ALL iterations
    self.assertTrue(all(cm_open_during_iteration))
    self.assertEqual(len(cm_open_during_iteration), 3)
    # CM exited after iteration completed
    self.assertTrue(cm.exited)

  async def test_with_iterate_cm_open_during_async_iteration(self) -> None:
    """For async .with_(fn).iterate(f), CM is still open while f runs."""
    items = [10, 20]
    cm = AsyncTrackingCM(items)
    cm_open: list[bool] = []

    async def check_open(item: int) -> int:
      cm_open.append(cm.entered and not cm.exited)
      return item

    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate(check_open):
      result.append(item)
    self.assertEqual(result, [10, 20])
    self.assertTrue(all(cm_open))


# ---------------------------------------------------------------------------
# with_(identity fn) tests — explicit identity lambda as the callable
# ---------------------------------------------------------------------------


class WithIdentityFnTests(SymmetricTestCase):
  """Tests for with_(lambda ctx: ctx) — the explicit identity-function form."""

  async def test_with_identity_iterate(self) -> None:
    """Q(cm).with_(lambda ctx: ctx).iterate() — ctx is the iterable."""
    items = [1, 2, 3]
    cm = TrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).iterate(sync_double))
    self.assertEqual(result, [2, 4, 6])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_identity_iterate_no_fn(self) -> None:
    """Q(cm).with_(lambda ctx: ctx).iterate() — yields ctx items directly."""
    items = [5, 10, 15]
    cm = TrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).iterate())
    self.assertEqual(result, [5, 10, 15])

  async def test_with_identity_flat_iterate(self) -> None:
    """Q(cm).with_(lambda ctx: ctx).flat_iterate() — ctx is flattened."""
    items = [[1, 2], [3]]
    cm = TrackingCM(items)
    result = list(Q(cm).with_(lambda ctx: ctx).flat_iterate())
    self.assertEqual(result, [1, 2, 3])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_identity_async_iterate(self) -> None:
    """Async: Q(cm).with_(lambda ctx: ctx).iterate() — async CM."""
    items = [10, 20]
    cm = AsyncTrackingCM(items)
    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate():
      result.append(item)
    self.assertEqual(result, [10, 20])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# §9.7 — CM __exit__ suppresses exception during deferred with_
# ---------------------------------------------------------------------------


class DeferredWithExitSuppressesTest(SymmetricTestCase):
  """§9.7: CM.__exit__ returns truthy during deferred with_ — exception suppressed, iteration stops cleanly."""

  async def test_sync_exit_suppresses_exception(self) -> None:
    """Sync CM.__exit__ returns True → exception suppressed, iteration stops cleanly.

    §9.7 CM exit semantics:
      'Exception during iteration: __exit__(*sys.exc_info()) — the CM receives
      the exception. If __exit__ returns truthy, the exception is suppressed and
      the generator stops cleanly. If falsy, the exception propagates.'
    """
    items = [1, 2, 3, 4]
    cm = TrackingSuppressCM(items)

    def boom(x: int) -> int:
      if x == 3:
        raise ValueError('boom at 3')
      return x

    # CM suppresses the exception — no error raised, iteration stops cleanly
    result = list(Q(cm).with_(lambda ctx: ctx).iterate(boom))
    # Items yielded before the exception: [1, 2] (item 3 raised)
    self.assertEqual(result, [1, 2])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    # __exit__ received the exception info
    assert cm.exit_args is not None
    self.assertIs(cm.exit_args[0], ValueError)

  async def test_async_exit_suppresses_exception(self) -> None:
    """Async CM.__aexit__ returns True → exception suppressed in async iteration.

    §9.7 CM exit semantics apply equally to async CMs.
    """
    items = [10, 20, 30, 40]
    cm = AsyncTrackingSuppressCM(items)

    def boom(x: int) -> int:
      if x == 30:
        raise RuntimeError('boom at 30')
      return x

    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate(boom):
      result.append(item)
    # Items yielded before the exception: [10, 20]
    self.assertEqual(result, [10, 20])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    assert cm.exit_args is not None
    self.assertIs(cm.exit_args[0], RuntimeError)


# ---------------------------------------------------------------------------
# §9.7 — CM __exit__ raises during deferred with_
# ---------------------------------------------------------------------------


class DeferredWithExitRaisesTest(SymmetricTestCase):
  """§9.7: CM.__exit__ raises during deferred with_ — new exception replaces original."""

  async def test_sync_exit_raises_replaces_original(self) -> None:
    """Sync CM.__exit__ raises → new exception replaces the original.

    §9.7 CM exit semantics:
      '__exit__ itself raises: The new exception replaces the original (if any),
      matching Python's native with statement behavior.'
    """
    items = [1, 2, 3]

    class ExitRaisesCM:
      def __init__(self, value: Any) -> None:
        self.value = value
        self.entered = False

      def __enter__(self) -> Any:
        self.entered = True
        return self.value

      def __exit__(self, *args: Any) -> bool:
        raise RuntimeError('exit exploded')

    cm = ExitRaisesCM(items)

    def boom(x: int) -> int:
      if x == 2:
        raise ValueError('original error')
      return x

    # The CM's __exit__ raises RuntimeError, which replaces the original ValueError
    with self.assertRaises(RuntimeError) as ctx:
      list(Q(cm).with_(lambda ctx_val: ctx_val).iterate(boom))
    self.assertIn('exit exploded', str(ctx.exception))
    self.assertTrue(cm.entered)

  async def test_async_exit_raises_replaces_original(self) -> None:
    """Async CM.__aexit__ raises → new exception replaces the original.

    Same contract as sync: '__exit__ itself raises: The new exception replaces
    the original'.
    """
    items = [10, 20, 30]

    class AsyncExitRaisesCM:
      def __init__(self, value: Any) -> None:
        self.value = value
        self.entered = False

      async def __aenter__(self) -> Any:
        self.entered = True
        return self.value

      async def __aexit__(self, *args: Any) -> bool:
        raise RuntimeError('async exit exploded')

    cm = AsyncExitRaisesCM(items)

    def boom(x: int) -> int:
      if x == 20:
        raise ValueError('async original error')
      return x

    with self.assertRaises(RuntimeError) as ctx:
      async for _ in Q(cm).with_(lambda ctx_val: ctx_val).iterate(boom):
        pass
    self.assertIn('async exit exploded', str(ctx.exception))
    self.assertTrue(cm.entered)


# ---------------------------------------------------------------------------
# §9.7 — Dual-protocol CM prefers async in async iteration
# ---------------------------------------------------------------------------


class DeferredWithDualProtocolAsyncIterationTest(IsolatedAsyncioTestCase):
  """§9.7: Dual-protocol CM uses __aenter__/__aexit__ in async for.

  §9.7 Protocol selection:
    'For dual-protocol CMs (supporting both __enter__/__exit__ and
    __aenter__/__aexit__), the async protocol is preferred when an
    async event loop is running (asyncio, trio, or curio), matching §16.10.'
  """

  async def test_dual_protocol_cm_uses_async_in_async_iteration(self) -> None:
    """Dual-protocol CM in deferred with_ + async for uses __aenter__/__aexit__."""
    items = [1, 2, 3]
    cm = DualProtocolTrackingCM(items)
    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3])
    # In async context (asyncio event loop running), the async protocol must be preferred
    self.assertTrue(cm.async_entered)
    self.assertTrue(cm.async_exited)
    self.assertFalse(cm.sync_entered)
    self.assertFalse(cm.sync_exited)

  async def test_dual_protocol_cm_uses_async_with_fn(self) -> None:
    """Dual-protocol CM + deferred with_ + iterate(fn) in async for uses async protocol."""
    items = [10, 20]
    cm = DualProtocolTrackingCM(items)
    result: list[int] = []
    async for item in Q(cm).with_(lambda ctx: ctx).iterate(lambda x: x * 2):
      result.append(item)
    self.assertEqual(result, [20, 40])
    self.assertTrue(cm.async_entered)
    self.assertTrue(cm.async_exited)
    self.assertFalse(cm.sync_entered)
    self.assertFalse(cm.sync_exited)


if __name__ == '__main__':
  unittest.main()
