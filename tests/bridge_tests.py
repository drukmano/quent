# SPDX-License-Identifier: MIT
"""Tests for SPEC §2 — The Bridge Contract, and §6 — Error Handling.

The fundamental guarantee: replacing any step's sync callable with a
functionally equivalent async callable (or vice versa) produces the
same observable result.

Also covers error-handling axes: error injection (ValueError, BaseException,
return_() signal, break_() signal), handler configurations, handler sync/async,
nesting depth, and concurrency x error.
"""

from __future__ import annotations

import asyncio
from unittest import TestCase

from quent import Chain
from tests.tests_helper import (
  ALL_BRICKS,
  ERROR_INJECTION_TYPES,
  HANDLER_CONFIGS,
  V_DOUBLE,
  V_FN,
  SymmetricTestCase,
  async_fn,
  async_raise,
  async_raise_base,
  run_bridge,
  sync_double,
  sync_fn,
  sync_identity,
  sync_raise,
  sync_raise_base,
  verify_all_brick_oracles,
)

# ---------------------------------------------------------------------------
# Reduced brick sets for bridge testing
# ---------------------------------------------------------------------------
# 24 bricks covering every distinct engine code path for sync↔async bridge
# testing. The remaining 72 bricks in ALL_BRICKS exercise the same engine
# paths with different calling conventions or constructor variants.
#
# Redundancy analysis (why each removed brick is covered):
#   kwargs variants  → same Rule 1 dispatch as args variants
#   nested_chain_args → combination of nested_chain + args (no new branch)
#   from_steps/clone/named/decorator → same as nested_chain (Chain callable)
#   *_conc/*_unbounded_conc → bridge runner's concurrency axis covers this
#   gather_single/bounded/multi → same gather path, different arity
#   with_do_* CM variants → orthogonal to with_do flag (covered separately)
#   if_ calling convention variants → calling convention covered by then_*
#   set_get variants → same _ctx_get/_ctx_set engine path
#   error handler variants → happy path, handler not invoked; same bridge test

_BRIDGE_BRICK_NAMES = frozenset(
  {
    # then — 4 distinct dispatch paths
    'then_default',
    'then_args',
    'then_nested_chain',
    'then_literal',
    # do — ignore_result path
    'do_default',
    # foreach — sync vs async iteration
    'map_default',
    'map_async_iter',
    # foreach_do — sync vs async iteration
    'foreach_do_default',
    'foreach_do_async_iter',
    # gather — concurrent execution
    'gather_default',
    # with_ — 4 distinct CM protocol paths
    'with_default',
    'with_async_cm',
    'with_dual_cm',
    'with_sync_async_exit',
    # with_do
    'with_do_default',
    # if_ — all distinct branch/predicate paths
    'if_true_default',
    'if_false_else',
    'if_do_branch',
    'if_pred_fn',
    'if_false_passthrough',
    'else_do',
    # context API
    'set_get',
    # error handlers (happy path)
    'except_nested',
    'finally_nested',
  }
)

# 10 representative bricks for error handling bridge — one per distinct
# error propagation path. Calling convention doesn't affect error
# propagation (exception type and handler dispatch are the same).
_BRIDGE_REPR_NAMES = frozenset(
  {
    'then_default',  # standard step
    'then_args',  # Rule 1 dispatch
    'then_nested_chain',  # nested chain propagation
    'do_default',  # side-effect step
    'map_default',  # iteration error (supports_concurrency)
    'foreach_do_default',  # iteration do error (supports_concurrency)
    'gather_default',  # concurrent error (supports_concurrency)
    'with_default',  # CM error path (sync __exit__ with exc)
    'with_async_cm',  # async CM error (__aexit__ with exc)
    'if_true_default',  # conditional branch error
  }
)

_BRIDGE_BRICKS = [b for b in ALL_BRICKS if b.name in _BRIDGE_BRICK_NAMES]
_BRIDGE_REPR = [b for b in ALL_BRICKS if b.name in _BRIDGE_REPR_NAMES]

# ===========================================================================
# Exhaustive bridge testing
# ===========================================================================


class BridgeExhaustiveTest(SymmetricTestCase):
  """Exhaustive bridge testing — all axes, all permutations."""

  async def test_exhaustive_bridge(self) -> None:
    """24 bridge bricks, lengths 1-2, error paths, concurrency. No handler configs."""
    stats = await run_bridge(
      self,
      bricks=_BRIDGE_BRICKS,
      max_length=2,
      include_error_path=True,
      exclude_gather_error=True,
      include_concurrency=True,
    )
    self.assertEqual(stats.failures, [])

  async def test_error_handling_bridge(self) -> None:
    """10 representative bricks x all error types x all handler configs x nesting."""
    stats = await run_bridge(
      self,
      bricks=_BRIDGE_REPR,
      max_length=2,
      handler_configs=HANDLER_CONFIGS,
      error_types=ERROR_INJECTION_TYPES,
      nesting_depths=[0, 1],
      include_handler_async=True,
      include_concurrency=False,
      exclude_gather_error=True,
    )
    self.assertEqual(stats.failures, [])

  async def test_concurrency_error_bridge(self) -> None:
    """Concurrency-capable bricks x error x handlers x nesting."""
    conc_bricks = [b for b in _BRIDGE_REPR if b.supports_concurrency]
    stats = await run_bridge(
      self,
      bricks=conc_bricks,
      max_length=2,
      handler_configs=HANDLER_CONFIGS,
      error_types=[('exception', sync_raise, async_raise)],
      nesting_depths=[0, 1],
      include_concurrency=True,
      include_handler_async=True,
      exclude_gather_error=True,
    )
    self.assertEqual(stats.failures, [])

  async def test_gather_error_bridge(self) -> None:
    """Gather bricks x error types x handlers, asymmetry-aware comparison."""
    gather_bricks = [b for b in _BRIDGE_BRICKS if b.op == 'gather']
    core = [b for b in _BRIDGE_REPR if b.name in ('then_default', 'do_default')]
    stats = await run_bridge(
      self,
      bricks=gather_bricks + core,
      max_length=2,
      handler_configs=HANDLER_CONFIGS,
      error_types=[
        ('exception', sync_raise, async_raise),
        ('base_exception', sync_raise_base, async_raise_base),
      ],
      include_handler_async=True,
      include_concurrency=False,
      gather_aware=True,
    )
    self.assertEqual(stats.failures, [])

  async def test_brick_oracles_correctness(self) -> None:
    """Verify every brick's oracle matches actual chain execution."""
    await verify_all_brick_oracles(self)


# ===========================================================================
# Two-tier model and zero async overhead
# ===========================================================================


class BridgeTwoTierModelTest(SymmetricTestCase):
  """SPEC §2: The two-tier execution model."""

  async def test_fully_sync_pipeline_returns_plain_value(self) -> None:
    """A pipeline where every step is synchronous returns a plain value,
    not a coroutine. No event loop is created."""
    result = Chain(5).then(sync_fn).then(sync_double).run()
    # sync_fn(5)=6, sync_double(6)=12
    self.assertEqual(result, 12)
    # Must NOT be a coroutine
    self.assertFalse(asyncio.iscoroutine(result))

  async def test_pipeline_with_async_step_returns_coroutine(self) -> None:
    """A pipeline with any async step returns a coroutine from .run()."""
    result = Chain(5).then(async_fn).run()
    self.assertTrue(asyncio.iscoroutine(result))
    value = await result
    self.assertEqual(value, 6)

  async def test_transition_at_first_step(self) -> None:
    """Async transition at the first step."""
    await self.variant(
      lambda fn: Chain(5).then(fn).then(sync_double).run(),
      fn=V_FN,
      expected=12,  # fn(5)=6, sync_double(6)=12
    )

  async def test_transition_at_middle_step(self) -> None:
    """Async transition at a middle step."""
    await self.variant(
      lambda fn: Chain(5).then(sync_fn).then(fn).then(sync_fn).run(),
      fn=V_DOUBLE,
      expected=13,  # sync_fn(5)=6, double(6)=12, sync_fn(12)=13
    )

  async def test_transition_at_last_step(self) -> None:
    """Async transition at the last step."""
    await self.variant(
      lambda fn: Chain(5).then(sync_fn).then(fn).run(),
      fn=V_DOUBLE,
      expected=12,  # sync_fn(5)=6, double(6)=12
    )

  async def test_once_async_stays_async(self) -> None:
    """Once the engine transitions to async, it stays async.
    There is no 'transition back to sync'."""
    # After async_fn, sync_fn and sync_double still execute correctly
    result = await Chain(5).then(async_fn).then(sync_fn).then(sync_double).run()
    # async_fn(5)=6, sync_fn(6)=7, sync_double(7)=14
    self.assertEqual(result, 14)


class BridgeZeroAsyncOverheadTest(TestCase):
  """SPEC §2: Fully sync pipelines have zero async overhead."""

  def test_sync_pipeline_no_coroutine(self) -> None:
    """A fully sync pipeline returns a plain value, not a coroutine."""
    result = Chain(5).then(sync_fn).then(sync_double).then(sync_identity).run()
    self.assertNotIsInstance(
      result, asyncio.coroutines._CoroutineMeta if hasattr(asyncio.coroutines, '_CoroutineMeta') else type(None)
    )
    self.assertFalse(asyncio.iscoroutine(result))
    # sync_fn(5)=6, sync_double(6)=12, sync_identity(12)=12
    self.assertEqual(result, 12)

  def test_sync_pipeline_does_not_require_event_loop(self) -> None:
    """A fully sync pipeline can be executed without any event loop."""
    result = Chain(10).then(sync_fn).then(sync_double).run()
    # sync_fn(10)=11, sync_double(11)=22
    self.assertEqual(result, 22)
