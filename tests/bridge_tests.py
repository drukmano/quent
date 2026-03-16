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
import os
import unittest
from unittest import TestCase

from quent import Chain
from tests.tests_helper import (
  ALL_BRICKS,
  ERROR_INJECTION_TYPES,
  HANDLER_CONFIGS,
  REPRESENTATIVE_BRICKS,
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


def load_tests(loader, tests, pattern):
  if not os.environ.get('QUENT_SLOW'):
    return unittest.TestSuite()
  return tests


# ===========================================================================
# Exhaustive bridge testing
# ===========================================================================


class BridgeExhaustiveTest(SymmetricTestCase):
  """Exhaustive bridge testing — all axes, all permutations."""

  async def test_exhaustive_bridge(self) -> None:
    """All 31 bricks, lengths 1-3, error paths, concurrency. No handler configs."""
    stats = await run_bridge(
      self,
      bricks=ALL_BRICKS,
      max_length=3,
      include_error_path=True,
      exclude_gather_error=True,
      include_concurrency=True,
    )
    self.assertEqual(stats.failures, [])

  async def test_error_handling_bridge(self) -> None:
    """15 representative bricks x all error types x all handler configs x nesting."""
    stats = await run_bridge(
      self,
      bricks=REPRESENTATIVE_BRICKS,
      max_length=2,
      handler_configs=HANDLER_CONFIGS,
      error_types=ERROR_INJECTION_TYPES,
      nesting_depths=[0, 1, 2],
      include_handler_async=True,
      include_concurrency=False,
      exclude_gather_error=True,
    )
    self.assertEqual(stats.failures, [])

  async def test_concurrency_error_bridge(self) -> None:
    """Concurrency-capable bricks x error x handlers x nesting."""
    conc_bricks = [b for b in REPRESENTATIVE_BRICKS if b.supports_concurrency]
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
    gather_bricks = [b for b in ALL_BRICKS if b.op == 'gather']
    core = [b for b in REPRESENTATIVE_BRICKS if b.name in ('then_default', 'do_default')]
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
