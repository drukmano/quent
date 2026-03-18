# SPDX-License-Identifier: MIT
"""Tests for SPEC §5.11 — drive_gen(fn).

Covers: bidirectional generator driving via send protocol, all 4 sync/async
bridge modes, empty/single/multi-yield generators, callable-producing-generator
resolution, error propagation, control flow signals, generator cleanup
lifecycle, and q composition (then, do, except_, finally_, clone).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase

from quent import Q, QuentException, QuentExcInfo
from tests.fixtures import (
  V_DOUBLE,
  V_IDENTITY,
  async_double,
  async_identity,
  sync_double,
  sync_identity,
)
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# Generator fixtures
# ---------------------------------------------------------------------------


def sync_gen_multi():
  """Yields 1, then (sent+1), then (sent+1). Three yields total."""
  x = yield 1
  x = yield x + 1
  x = yield x + 1


async def async_gen_multi():
  """Async equivalent of sync_gen_multi."""
  x = yield 1
  x = yield x + 1
  x = yield x + 1


def sync_gen_single():
  """Yields once."""
  yield 42


async def async_gen_single():
  """Async: yields once."""
  yield 42


def sync_gen_empty():
  """Yields nothing."""
  return
  yield  # make it a generator


async def async_gen_empty():
  """Async: yields nothing."""
  return
  yield


def sync_gen_with_return():
  """Yields 1, returns 'ignored'."""
  yield 1
  return 'ignored'


async def async_gen_with_return():
  """Async: yields 1. (Async generators cannot use return with a value.)"""
  yield 1


def sync_gen_accumulator():
  """Yields running total: starts at 0, adds sent value each time. 5 yields."""
  total = 0
  for _ in range(5):
    x = yield total
    total += x


async def async_gen_accumulator():
  """Async version of accumulator. 5 yields."""
  total = 0
  for _ in range(5):
    x = yield total
    total += x


def sync_gen_large(n: int = 150):
  """Yields incrementing values n times."""
  for i in range(n):
    _ = yield i


async def async_gen_large(n: int = 150):
  """Async: yields incrementing values n times."""
  for i in range(n):
    _ = yield i


# Generators with finally blocks for cleanup tracking

_cleanup_log: list[str] = []


def sync_gen_tracked():
  """Sync gen that records cleanup to _cleanup_log."""
  try:
    x = yield 1
    x = yield x + 1
    x = yield x + 1
  finally:
    _cleanup_log.append('sync_closed')


async def async_gen_tracked():
  """Async gen that records cleanup to _cleanup_log."""
  try:
    x = yield 1
    x = yield x + 1
    x = yield x + 1
  finally:
    _cleanup_log.append('async_closed')


def sync_gen_tracked_short():
  """Sync gen with single yield and cleanup tracking."""
  try:
    yield 1
  finally:
    _cleanup_log.append('sync_short_closed')


async def async_gen_tracked_short():
  """Async gen with single yield and cleanup tracking."""
  try:
    yield 1
  finally:
    _cleanup_log.append('async_short_closed')


# Generator that raises on send


def sync_gen_raise_on_send():
  """Yields 1, then raises RuntimeError when value is sent."""
  yield 1
  raise RuntimeError('gen send error')


async def async_gen_raise_on_send():
  """Async: yields 1, then raises RuntimeError when value is sent."""
  yield 1
  raise RuntimeError('gen send error')


# Variant axes for generators

V_GEN_MULTI = [('sync', sync_gen_multi), ('async', async_gen_multi)]
V_GEN_SINGLE = [('sync', sync_gen_single), ('async', async_gen_single)]
V_GEN_EMPTY = [('sync', sync_gen_empty), ('async', async_gen_empty)]
V_GEN_ACCUM = [('sync', sync_gen_accumulator), ('async', async_gen_accumulator)]
V_GEN_RAISE_ON_SEND = [('sync', sync_gen_raise_on_send), ('async', async_gen_raise_on_send)]


# ---------------------------------------------------------------------------
# Helper step functions
# ---------------------------------------------------------------------------


def sync_add_10(x: Any) -> Any:
  return x + 10


async def async_add_10(x: Any) -> Any:
  return x + 10


V_ADD_10 = [('sync', sync_add_10), ('async', async_add_10)]


def sync_failing_step(x: Any) -> Any:
  raise ValueError('step_fn error')


async def async_failing_step(x: Any) -> Any:
  raise ValueError('step_fn error')


V_FAILING_STEP = [('sync', sync_failing_step), ('async', async_failing_step)]


# ---------------------------------------------------------------------------
# §5.11 — drive_gen basic bridge tests
# ---------------------------------------------------------------------------


class DriveGenBasicTests(SymmetricTestCase):
  """§5.11 — Core bridge tests using variant() for all 4 sync/async modes."""

  async def test_multi_yield_double(self) -> None:
    """Multi-yield generator driven by double: last fn result is pipeline value.

    Flow: yield 1 -> fn(1)=2 -> send 2 -> yield 3 -> fn(3)=6 -> send 6
          -> yield 7 -> fn(7)=14 -> StopIteration
    Expected: 14
    """
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=14,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
    )

  async def test_single_yield_identity(self) -> None:
    """Single-yield generator: fn called once, its result is pipeline value.

    Generator yields 42, step_fn=identity -> fn(42)=42 -> StopIteration
    Expected: 42
    """
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=42,
      gen=V_GEN_SINGLE,
      fn=V_IDENTITY,
    )

  async def test_single_yield_double(self) -> None:
    """Single-yield generator with double: fn(42)=84.

    Expected: 84
    """
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=84,
      gen=V_GEN_SINGLE,
      fn=V_DOUBLE,
    )

  async def test_accumulator_add_10(self) -> None:
    """Accumulator generator: each send adds to running total.

    step_fn: add 10
    Yields: 0 -> fn(0)=10 -> send 10 -> yield 10 -> fn(10)=20 -> send 20
         -> yield 30 -> fn(30)=40 -> send 40 -> yield 70 -> fn(70)=80
         -> send 80 -> yield 150 -> fn(150)=160 -> StopIteration
    Expected: 160
    """
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=160,
      gen=V_GEN_ACCUM,
      fn=V_ADD_10,
    )

  async def test_multi_yield_identity(self) -> None:
    """Multi-yield with identity fn: each yield gets sent value unchanged.

    Flow: yield 1 -> fn(1)=1 -> send 1 -> yield 2 -> fn(2)=2 -> send 2
          -> yield 3 -> fn(3)=3 -> StopIteration
    Expected: 3
    """
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=3,
      gen=V_GEN_MULTI,
      fn=V_IDENTITY,
    )


# ---------------------------------------------------------------------------
# §5.11 — drive_gen edge cases
# ---------------------------------------------------------------------------


class DriveGenEdgeCaseTests(SymmetricTestCase):
  """§5.11 — Edge cases: empty gen, callable resolution, type errors, return value."""

  async def test_empty_generator(self) -> None:
    """Empty generator (yields nothing): pipeline value is None.

    fn is never called. StopIteration on first next() -> result is None.
    """
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=None,
      gen=V_GEN_EMPTY,
      fn=V_DOUBLE,
    )

  async def test_callable_produces_generator(self) -> None:
    """Callable that produces a generator is resolved: callable invoked to get gen.

    Q(gen_factory).drive_gen(fn).run() where gen_factory is a callable.
    """
    await self.variant(
      lambda gen, fn: Q(gen).drive_gen(fn).run(),
      expected=14,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
    )

  async def test_non_generator_type_error(self) -> None:
    """Non-generator pipeline value raises TypeError at runtime."""
    await self.variant(
      lambda fn: Q(42).drive_gen(fn).run(),
      expected_exc=TypeError,
      expected_msg='not a generator',
      fn=V_DOUBLE,
    )

  async def test_non_callable_fn_type_error(self) -> None:
    """Non-callable fn raises TypeError at build time."""
    with self.assertRaises(TypeError):
      Q(sync_gen_multi()).drive_gen(42)  # type: ignore[arg-type]

  async def test_generator_return_value_ignored(self) -> None:
    """Generator's return value (StopIteration.value) does NOT become pipeline value.

    sync_gen_with_return yields 1 then returns 'ignored'.
    step_fn=identity -> fn(1)=1 -> send 1 -> StopIteration(value='ignored')
    Pipeline value should be 1 (last fn result), not 'ignored'.
    """
    # Sync gen with return value
    result = Q(sync_gen_with_return()).drive_gen(sync_identity).run()
    self.assertEqual(result, 1)

  async def test_async_generator_return_value_ignored(self) -> None:
    """Async generator's yield-once: pipeline value is last fn result.

    async_gen_with_return yields 1 (async generators cannot return a value).
    step_fn=identity -> fn(1)=1 -> StopAsyncIteration
    Expected: 1
    """
    result = await Q(async_gen_with_return()).drive_gen(async_identity).run()
    self.assertEqual(result, 1)

  async def test_large_iteration_count(self) -> None:
    """Generator with 150 yields: no stack overflow or recursion issues."""
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=149,  # identity of last yielded value (149)
      gen=[('sync', lambda: sync_gen_large(150)), ('async', lambda: async_gen_large(150))],
      fn=V_IDENTITY,
    )

  async def test_string_pipeline_value_type_error(self) -> None:
    """String pipeline value raises TypeError (not a generator or callable)."""
    await self.variant(
      lambda fn: Q('not a gen').drive_gen(fn).run(),
      expected_exc=TypeError,
      expected_msg='not a generator',
      fn=V_DOUBLE,
    )

  async def test_list_pipeline_value_type_error(self) -> None:
    """List pipeline value raises TypeError."""
    await self.variant(
      lambda fn: Q([1, 2, 3]).drive_gen(fn).run(),
      expected_exc=TypeError,
      expected_msg='not a generator',
      fn=V_DOUBLE,
    )


# ---------------------------------------------------------------------------
# §5.11 — drive_gen cleanup verification
# ---------------------------------------------------------------------------


class DriveGenCleanupTests(SymmetricTestCase):
  """§5.11 — Generator cleanup: gen.close() / gen.aclose() always called."""

  def setUp(self) -> None:
    super().setUp()
    _cleanup_log.clear()

  async def test_sync_gen_closed_on_normal_completion(self) -> None:
    """Sync generator's finally block runs on normal completion."""
    _cleanup_log.clear()
    result = Q(sync_gen_tracked()).drive_gen(sync_double).run()
    self.assertEqual(result, 14)
    self.assertIn('sync_closed', _cleanup_log)

  async def test_async_gen_closed_on_normal_completion(self) -> None:
    """Async generator's finally block runs on normal completion."""
    _cleanup_log.clear()
    result = await Q(async_gen_tracked()).drive_gen(async_double).run()
    self.assertEqual(result, 14)
    self.assertIn('async_closed', _cleanup_log)

  async def test_sync_gen_closed_on_step_fn_error(self) -> None:
    """Sync generator's finally block runs when step_fn raises."""
    _cleanup_log.clear()
    try:
      Q(sync_gen_tracked_short()).drive_gen(sync_failing_step).run()
    except ValueError:
      pass
    self.assertIn('sync_short_closed', _cleanup_log)

  async def test_async_gen_closed_on_step_fn_error(self) -> None:
    """Async generator's finally block runs when step_fn raises."""
    _cleanup_log.clear()
    try:
      await Q(async_gen_tracked_short()).drive_gen(async_failing_step).run()
    except ValueError:
      pass
    self.assertIn('async_short_closed', _cleanup_log)

  async def test_sync_gen_closed_on_gen_send_error(self) -> None:
    """Sync generator's finally block runs when gen raises on send."""

    closed = []

    def gen_raises_on_send():
      try:
        yield 1
        raise RuntimeError('boom on send')
      finally:
        closed.append('closed')

    try:
      Q(gen_raises_on_send()).drive_gen(sync_identity).run()
    except RuntimeError:
      pass
    self.assertIn('closed', closed)

  async def test_async_gen_closed_on_gen_send_error(self) -> None:
    """Async generator's finally block runs when async gen raises on send."""

    closed = []

    async def gen_raises_on_send():
      try:
        yield 1
        raise RuntimeError('boom on send')
      finally:
        closed.append('closed')

    try:
      await Q(gen_raises_on_send()).drive_gen(async_identity).run()
    except RuntimeError:
      pass
    self.assertIn('closed', closed)

  async def test_sync_gen_closed_on_mid_transition_error(self) -> None:
    """Sync gen + async failing step_fn: generator cleaned up on error."""

    closed = []

    def gen_tracked():
      try:
        yield 1
        yield 2
      finally:
        closed.append('closed')

    try:
      await Q(gen_tracked()).drive_gen(async_failing_step).run()
    except ValueError:
      pass
    self.assertIn('closed', closed)

  async def test_empty_sync_gen_closed(self) -> None:
    """Empty sync generator is still closed after StopIteration on first next()."""

    closed = []

    def empty_gen():
      try:
        return
        yield
      finally:
        closed.append('closed')

    result = Q(empty_gen()).drive_gen(sync_identity).run()
    self.assertIsNone(result)
    self.assertIn('closed', closed)

  async def test_empty_async_gen_closed(self) -> None:
    """Empty async generator is still closed after StopAsyncIteration on first __anext__."""

    closed = []

    async def empty_gen():
      try:
        return
        yield
      finally:
        closed.append('closed')

    result = await Q(empty_gen()).drive_gen(async_identity).run()
    self.assertIsNone(result)
    self.assertIn('closed', closed)


# ---------------------------------------------------------------------------
# §5.11 — drive_gen error handling
# ---------------------------------------------------------------------------


class DriveGenErrorTests(SymmetricTestCase):
  """§5.11 — Error semantics: fn errors propagate, gen errors propagate."""

  async def test_step_fn_exception_propagates(self) -> None:
    """Exception from step_fn propagates out of drive_gen."""
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected_exc=ValueError,
      expected_msg='step_fn error',
      gen=V_GEN_MULTI,
      fn=V_FAILING_STEP,
    )

  async def test_gen_send_exception_propagates(self) -> None:
    """Exception from gen.send() (non-Stop) propagates out of drive_gen."""
    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected_exc=RuntimeError,
      expected_msg='gen send error',
      gen=V_GEN_RAISE_ON_SEND,
      fn=V_IDENTITY,
    )

  async def test_compose_with_except(self) -> None:
    """drive_gen error caught by chain's except_().

    step_fn raises ValueError -> except_ handler receives QuentExcInfo.
    """

    def sync_err_handler(info: QuentExcInfo) -> str:
      return 'handled'

    async def async_err_handler(info: QuentExcInfo) -> str:
      return 'handled'

    await self.variant(
      lambda gen, fn, handler: Q(gen()).drive_gen(fn).except_(handler).run(),
      expected='handled',
      gen=V_GEN_MULTI,
      fn=V_FAILING_STEP,
      handler=[('sync', sync_err_handler), ('async', async_err_handler)],
    )

  async def test_compose_with_finally(self) -> None:
    """drive_gen with finally_(): cleanup handler runs on success.

    finally_() receives the root value (the generator object per SPEC §6.3),
    not the pipeline result.
    """
    cleanup_ran = []

    def sync_cleanup(rv: Any) -> None:
      cleanup_ran.append('ran')

    result = Q(sync_gen_single()).drive_gen(sync_double).finally_(sync_cleanup).run()
    self.assertEqual(result, 84)
    self.assertEqual(len(cleanup_ran), 1)

  async def test_compose_with_finally_async(self) -> None:
    """Async drive_gen with finally_(): cleanup handler runs on success.

    finally_() receives the root value (the generator object per SPEC §6.3),
    not the pipeline result.
    """
    cleanup_ran = []

    async def async_cleanup(rv: Any) -> None:
      cleanup_ran.append('ran')

    result = await Q(async_gen_single()).drive_gen(async_double).finally_(async_cleanup).run()
    self.assertEqual(result, 84)
    self.assertEqual(len(cleanup_ran), 1)

  async def test_compose_with_except_and_finally(self) -> None:
    """Both except_ and finally_ run correctly on drive_gen error."""
    order: list[str] = []

    def handler(info: QuentExcInfo) -> str:
      order.append('except')
      return 'recovered'

    def cleanup(rv: Any) -> None:
      order.append('finally')

    result = Q(sync_gen_multi()).drive_gen(sync_failing_step).except_(handler).finally_(cleanup).run()
    self.assertEqual(result, 'recovered')
    self.assertIn('except', order)
    self.assertIn('finally', order)

  async def test_compose_with_except_and_finally_async(self) -> None:
    """Async: both except_ and finally_ run correctly on drive_gen error."""
    order: list[str] = []

    async def handler(info: QuentExcInfo) -> str:
      order.append('except')
      return 'recovered'

    async def cleanup(rv: Any) -> None:
      order.append('finally')

    result = await Q(async_gen_multi()).drive_gen(async_failing_step).except_(handler).finally_(cleanup).run()
    self.assertEqual(result, 'recovered')
    self.assertIn('except', order)
    self.assertIn('finally', order)

  async def test_finally_runs_on_success(self) -> None:
    """finally_() runs on successful drive_gen completion."""
    cleanup_ran = []

    def cleanup(rv: Any) -> None:
      cleanup_ran.append(rv)

    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).finally_(cleanup).run(),
      expected=14,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
    )
    # cleanup should have been called at least once (once per variant combo)
    self.assertTrue(len(cleanup_ran) > 0)

  async def test_gen_send_error_caught_by_except(self) -> None:
    """gen.send() RuntimeError caught by except_()."""

    def handler(info: QuentExcInfo) -> str:
      return 'gen_error_handled'

    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).except_(handler).run(),
      expected='gen_error_handled',
      gen=V_GEN_RAISE_ON_SEND,
      fn=V_IDENTITY,
    )


# ---------------------------------------------------------------------------
# §5.11 — drive_gen q composition
# ---------------------------------------------------------------------------


class DriveGenIntegrationTests(SymmetricTestCase):
  """§5.11 — Quent composition: then, do, root value, clone."""

  async def test_compose_with_then(self) -> None:
    """then() after drive_gen receives drive_gen's result.

    drive_gen produces 14, then(double) -> 28.
    """
    await self.variant(
      lambda gen, fn, then_fn: Q(gen()).drive_gen(fn).then(then_fn).run(),
      expected=28,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
      then_fn=V_DOUBLE,
    )

  async def test_compose_with_do(self) -> None:
    """do() after drive_gen runs side effect, pipeline value unchanged.

    drive_gen produces 14, do(side_effect) -> 14 still.
    """
    side_effects: list[Any] = []

    def sync_side(x: Any) -> None:
      side_effects.append(x)

    async def async_side(x: Any) -> None:
      side_effects.append(x)

    await self.variant(
      lambda gen, fn, do_fn: Q(gen()).drive_gen(fn).do(do_fn).run(),
      expected=14,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
      do_fn=[('sync', sync_side), ('async', async_side)],
    )
    # side effect should have received 14 at least once
    self.assertIn(14, side_effects)

  async def test_root_value_feeds_drive_gen(self) -> None:
    """Root callable -> then(create_gen) -> drive_gen(fn): pipeline threading.

    Q(lambda: 'root').then(lambda _: gen()).drive_gen(double).run()
    """
    await self.variant(
      lambda gen, fn: Q(lambda: 'root').then(lambda _: gen()).drive_gen(fn).run(),
      expected=14,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
    )

  async def test_clone_preserves_drive_gen(self) -> None:
    """clone() preserves drive_gen: cloned pipeline produces same result."""
    original = Q(sync_gen_multi).drive_gen(sync_double)
    cloned = original.clone()
    r1 = original.run()
    r2 = cloned.run()
    self.assertEqual(r1, 14)
    self.assertEqual(r2, 14)

  async def test_clone_preserves_drive_gen_async(self) -> None:
    """Async: clone() preserves drive_gen."""
    original = Q(async_gen_multi).drive_gen(async_double)
    cloned = original.clone()
    r1 = await original.run()
    r2 = await cloned.run()
    self.assertEqual(r1, 14)
    self.assertEqual(r2, 14)

  async def test_multiple_then_after_drive_gen(self) -> None:
    """Multiple then() steps after drive_gen chain correctly.

    drive_gen -> 14, then(double) -> 28, then(add_10) -> 38.
    """
    result = Q(sync_gen_multi()).drive_gen(sync_double).then(sync_double).then(sync_add_10).run()
    self.assertEqual(result, 38)

  async def test_drive_gen_after_then(self) -> None:
    """then() produces a generator, followed by drive_gen.

    Q(None).then(lambda _: gen()).drive_gen(double) -> 14.
    """
    await self.variant(
      lambda gen, fn: Q(None).then(lambda _: gen()).drive_gen(fn).run(),
      expected=14,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
    )

  async def test_callable_factory_with_then(self) -> None:
    """Callable factory as root: Q(factory).drive_gen(fn).then(transform).

    The factory is invoked to produce the generator (callable resolution).
    """
    await self.variant(
      lambda gen, fn, then_fn: Q(gen).drive_gen(fn).then(then_fn).run(),
      expected=28,
      gen=V_GEN_MULTI,
      fn=V_DOUBLE,
      then_fn=V_DOUBLE,
    )


# ---------------------------------------------------------------------------
# §5.11 + §7 — drive_gen control flow signals
# ---------------------------------------------------------------------------


class DriveGenControlFlowTests(IsolatedAsyncioTestCase):
  """§5.11 + §7 — Control flow signals: return_(), break_() through drive_gen."""

  async def test_return_from_step_fn_sync(self) -> None:
    """return_() in sync step_fn: chain returns early, generator cleaned up."""
    closed = []

    def gen():
      try:
        yield 1
        yield 2
        yield 3
      finally:
        closed.append('closed')

    def step(x):
      if x == 1:
        return Q.return_('early')
      return x

    result = Q(gen()).drive_gen(step).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'early')
    self.assertIn('closed', closed)

  async def test_return_from_step_fn_async(self) -> None:
    """return_() in async step_fn: chain returns early, generator cleaned up."""
    closed = []

    async def gen():
      try:
        yield 1
        yield 2
        yield 3
      finally:
        closed.append('closed')

    async def step(x):
      if x == 1:
        return Q.return_('early')
      return x

    result = await Q(gen()).drive_gen(step).then(lambda x: 'should not reach').run()
    self.assertEqual(result, 'early')
    self.assertIn('closed', closed)

  async def test_return_from_step_fn_mid_transition(self) -> None:
    """return_() in async step_fn with sync generator: mid-transition mode."""
    closed = []

    def gen():
      try:
        yield 1
        yield 2
      finally:
        closed.append('closed')

    async def step(x):
      return Q.return_('mid_early')

    result = await Q(gen()).drive_gen(step).then(lambda x: 'unreachable').run()
    self.assertEqual(result, 'mid_early')
    self.assertIn('closed', closed)

  async def test_break_from_step_fn_standalone_raises(self) -> None:
    """break_() in step_fn of a standalone chain raises QuentException.

    break_() can only be used inside an iteration context. A standalone
    Q().drive_gen().run() is NOT nested in an iteration, so break_()
    raises QuentException per SPEC §7.
    """
    closed = []

    def gen():
      try:
        yield 'stop_signal'
      finally:
        closed.append('closed')

    def step(x):
      return Q.break_()

    with self.assertRaises(QuentException):
      Q(gen()).drive_gen(step).run()
    # Generator is still cleaned up
    self.assertIn('closed', closed)

  async def test_break_from_step_fn_standalone_async_raises(self) -> None:
    """Async: break_() in step_fn of a standalone async pipeline raises QuentException."""
    closed = []

    async def gen():
      try:
        yield 'stop_signal'
      finally:
        closed.append('closed')

    async def step(x):
      return Q.break_()

    with self.assertRaises(QuentException):
      await Q(gen()).drive_gen(step).run()
    self.assertIn('closed', closed)

  async def test_control_flow_does_not_inject_into_generator(self) -> None:
    """Control flow signals are NOT injected into the generator (no gen.throw()).

    The generator should see a clean close(), not an exception throw.
    """
    gen_saw_exception = []

    def gen():
      try:
        yield 1
        yield 2
      except BaseException as e:
        gen_saw_exception.append(type(e).__name__)
        raise
      finally:
        pass  # close() triggers GeneratorExit which is normal

    def step(x):
      return Q.return_('early')

    result = Q(gen()).drive_gen(step).run()
    self.assertEqual(result, 'early')
    # Generator should NOT have seen any exception from step_fn
    # (GeneratorExit from close() is expected and normal, but ValueError etc. should not appear)
    for exc_name in gen_saw_exception:
      self.assertNotIn('Return', exc_name)
      self.assertNotIn('ValueError', exc_name)


# ---------------------------------------------------------------------------
# §5.11 — drive_gen sync/async bridging specific tests
# ---------------------------------------------------------------------------


class DriveGenBridgingTests(IsolatedAsyncioTestCase):
  """§5.11 — Test all 4 sync/async bridging modes explicitly."""

  async def test_fully_sync(self) -> None:
    """Sync gen + sync step_fn: fully sync, no coroutine returned from run()."""
    result = Q(sync_gen_multi()).drive_gen(sync_double).run()
    # Should not be a coroutine -- fully sync path
    self.assertNotIsInstance(result, type(asyncio.sleep(0)))
    self.assertEqual(result, 14)

  async def test_mid_transition(self) -> None:
    """Sync gen + async step_fn: mid-transition, run() returns coroutine.

    Flow: yield 1 -> fn(1)=awaitable(2) -> await -> send 2 -> yield 3
          -> fn(3)=awaitable(6) -> await -> send 6 -> yield 7
          -> fn(7)=awaitable(14) -> await -> StopIteration
    Expected: 14
    """
    result = await Q(sync_gen_multi()).drive_gen(async_double).run()
    self.assertEqual(result, 14)

  async def test_fully_async_sync_fn(self) -> None:
    """Async gen + sync step_fn: fully async."""
    result = await Q(async_gen_multi()).drive_gen(sync_double).run()
    self.assertEqual(result, 14)

  async def test_fully_async_async_fn(self) -> None:
    """Async gen + async step_fn: fully async."""
    result = await Q(async_gen_multi()).drive_gen(async_double).run()
    self.assertEqual(result, 14)

  async def test_mid_transition_empty_gen(self) -> None:
    """Sync gen (empty) + async step_fn: fn never called, result is None."""
    result = Q(sync_gen_empty()).drive_gen(async_double).run()
    # Empty sync gen -> StopIteration on first next() -> returns None
    # No awaitable encountered, so should be sync (None returned directly)
    self.assertIsNone(result)

  async def test_fully_async_empty_gen(self) -> None:
    """Async gen (empty) + async step_fn: fn never called, result is None."""
    result = await Q(async_gen_empty()).drive_gen(async_double).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# §5.11 — drive_gen calling convention
# ---------------------------------------------------------------------------


class DriveGenCallingConventionTests(SymmetricTestCase):
  """§5.11 — fn is always called as fn(yielded_value), no args/kwargs dispatch."""

  async def test_fn_receives_yielded_value(self) -> None:
    """fn always receives the yielded value as its single argument."""
    received: list[Any] = []

    def sync_capture(x: Any) -> Any:
      received.append(x)
      return x * 2

    async def async_capture(x: Any) -> Any:
      received.append(x)
      return x * 2

    await self.variant(
      lambda gen, fn: Q(gen()).drive_gen(fn).run(),
      expected=14,
      gen=V_GEN_MULTI,
      fn=[('sync', sync_capture), ('async', async_capture)],
    )
    # For the multi-gen: yields 1, 3, 7 across all variants.
    # Each variant combo records its own yields, so we check that
    # each run saw the right sequence.
    # The sync variant sees [1, 3, 7], async variant sees [1, 3, 7], etc.
    # With 4 combos (2 gen x 2 fn), we get at least 12 entries.
    # Just verify the pattern is correct by checking multiples of 3.
    self.assertEqual(len(received) % 3, 0)
    for i in range(0, len(received), 3):
      self.assertEqual(received[i], 1)
      self.assertEqual(received[i + 1], 3)
      self.assertEqual(received[i + 2], 7)

  async def test_fn_return_sent_back_to_generator(self) -> None:
    """fn's return value is sent back into the generator via gen.send()."""
    sent_values: list[Any] = []

    def gen():
      x = yield 'first'
      sent_values.append(x)
      x = yield 'second'
      sent_values.append(x)

    result = Q(gen()).drive_gen(lambda x: f'processed_{x}').run()
    self.assertEqual(result, 'processed_second')
    self.assertEqual(sent_values, ['processed_first', 'processed_second'])

  async def test_fn_return_sent_back_async(self) -> None:
    """Async: fn's return value is sent back into the async generator."""
    sent_values: list[Any] = []

    async def gen():
      x = yield 'first'
      sent_values.append(x)
      x = yield 'second'
      sent_values.append(x)

    result = await Q(gen()).drive_gen(lambda x: f'processed_{x}').run()
    self.assertEqual(result, 'processed_second')
    self.assertEqual(sent_values, ['processed_first', 'processed_second'])


# ---------------------------------------------------------------------------
# §5.11 — drive_gen with various pipeline values
# ---------------------------------------------------------------------------


class DriveGenPipelineValueTests(SymmetricTestCase):
  """§5.11 — Pipeline value resolution: generators, callables, non-generators."""

  async def test_direct_sync_generator_instance(self) -> None:
    """Direct sync generator instance as pipeline value."""
    result = Q(sync_gen_multi()).drive_gen(sync_double).run()
    self.assertEqual(result, 14)

  async def test_direct_async_generator_instance(self) -> None:
    """Direct async generator instance as pipeline value."""
    result = await Q(async_gen_multi()).drive_gen(async_double).run()
    self.assertEqual(result, 14)

  async def test_sync_callable_producing_sync_gen(self) -> None:
    """Callable (not generator) invoked to get sync generator."""
    result = Q(sync_gen_multi).drive_gen(sync_double).run()
    self.assertEqual(result, 14)

  async def test_async_callable_producing_async_gen(self) -> None:
    """Callable (not generator) invoked to get async generator."""
    result = await Q(async_gen_multi).drive_gen(async_double).run()
    self.assertEqual(result, 14)

  async def test_lambda_producing_sync_gen(self) -> None:
    """Lambda producing a sync generator."""
    result = Q(lambda: sync_gen_multi()).drive_gen(sync_double).run()
    self.assertEqual(result, 14)

  async def test_lambda_producing_async_gen(self) -> None:
    """Lambda producing an async generator."""
    result = await Q(lambda: async_gen_multi()).drive_gen(async_double).run()
    self.assertEqual(result, 14)

  async def test_none_pipeline_value_type_error(self) -> None:
    """None pipeline value raises TypeError."""
    with self.assertRaises(TypeError):
      Q(None).drive_gen(sync_identity).run()

  async def test_integer_pipeline_value_type_error(self) -> None:
    """Integer pipeline value raises TypeError."""
    with self.assertRaises(TypeError):
      Q(42).drive_gen(sync_identity).run()


# ---------------------------------------------------------------------------
# §5.11 — drive_gen fluent API
# ---------------------------------------------------------------------------


class DriveGenFluentTests(IsolatedAsyncioTestCase):
  """§5.11 — drive_gen returns self for fluent chaining."""

  async def test_returns_self(self) -> None:
    """drive_gen() returns the chain instance for fluent chaining."""
    q = Q(sync_gen_multi())
    result = q.drive_gen(sync_double)
    self.assertIs(result, q)

  async def test_fluent_chaining(self) -> None:
    """Fluent: Q(gen()).drive_gen(fn).then(transform).run() works."""
    result = Q(sync_gen_multi()).drive_gen(sync_double).then(lambda x: x + 1).run()
    self.assertEqual(result, 15)
