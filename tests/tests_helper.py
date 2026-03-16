"""Exhaustive sync/async bridge test infrastructure.

The fundamental invariant: quent is a transparent sync/async bridge.
Swapping ANY sync callable for its async equivalent -- at ANY position,
in ANY operation, in ANY order, with ANY calling convention -- must
produce the SAME result.

This module provides two testing layers:

1. ``SymmetricTestCase.variant()`` -- cartesian product of named axes.
   For hand-written tests with specific expected values.

2. ``run_bridge()`` -- unified bridge test runner covering all axes:
   operation type, chain length, operation order (permutations),
   sync/async per position, error path, concurrency, error handlers,
   nesting depth, and gather-aware asymmetry comparison.

   The bridge contract is tested WITHOUT computing expected values:
   for any fixed chain configuration, ALL sync/async permutations
   must produce the same result.  If any permutation differs, the
   bridge is broken.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest import IsolatedAsyncioTestCase

from quent import Chain

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

_UNSET = object()


# ---------------------------------------------------------------------------
# Result capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Result:
  """Captured chain execution result."""

  success: bool
  value: Any = None
  exc_type: type | None = None
  exc_message: str | None = None
  sub_exc_types: frozenset[type] | None = None


def _extract_sub_exc_types(exc: BaseException) -> frozenset[type] | None:
  """Extract sub-exception types from an ExceptionGroup, or None."""
  if hasattr(exc, 'exceptions') and exc.exceptions:
    return frozenset(type(e) for e in exc.exceptions)
  return None


async def capture(fn: Any) -> Result:
  """Execute fn(), await if coroutine, capture result or exception."""
  try:
    result = fn()
    if asyncio.iscoroutine(result):
      result = await result
    return Result(success=True, value=result)
  except BaseException as exc:
    return Result(
      success=False,
      exc_type=type(exc),
      exc_message=str(exc),
      sub_exc_types=_extract_sub_exc_types(exc),
    )


# ---------------------------------------------------------------------------
# SymmetricTestCase — manual variant testing
# ---------------------------------------------------------------------------

VariantAxis = list[tuple[str, Any]]


class SymmetricTestCase(IsolatedAsyncioTestCase):
  """Base class for cartesian-product sync/async bridge testing.

  ``variant()`` computes the cartesian product of all axes, runs the
  builder with each combination, and asserts:
  1. ALL combinations produce the same result.
  2. The result matches the expected value or exception.
  """

  async def variant(
    self,
    builder: Any,
    *,
    expected: Any = _UNSET,
    expected_exc: Any = _UNSET,
    expected_msg: str | None = None,
    **axes: Any,
  ) -> list[Result]:
    axis_names = list(axes.keys())
    axis_values = list(axes.values())
    combos = list(itertools.product(*axis_values))

    results: list[Result] = []
    labels: list[str] = []

    for combo in combos:
      label_parts = []
      kwargs: dict[str, Any] = {}
      for name, (label, value) in zip(axis_names, combo):
        label_parts.append(f'{name}={label}')
        kwargs[name] = value
      label = ', '.join(label_parts) if label_parts else 'default'
      labels.append(label)

      kw = dict(kwargs)
      result = await capture(lambda _kw=kw: builder(**_kw))
      results.append(result)

    # Assert all combinations produce the same outcome
    first = results[0]
    for i in range(1, len(results)):
      with self.subTest(symmetry=f'{labels[0]} vs {labels[i]}'):
        self.assertEqual(
          first.success,
          results[i].success,
          f'{labels[0]} {"succeeded" if first.success else "failed"} but '
          f'{labels[i]} {"succeeded" if results[i].success else "failed"}',
        )
        if first.success:
          self.assertEqual(first.value, results[i].value, f'value mismatch: {labels[i]}')
        else:
          self.assertEqual(first.exc_type, results[i].exc_type, f'exc type mismatch: {labels[i]}')

    # Assert expected value/exception
    for result, label in zip(results, labels):
      with self.subTest(expected=label):
        if expected is not _UNSET:
          self.assertTrue(
            result.success,
            f'{label}: expected success but got '
            f'{result.exc_type.__name__ if result.exc_type else "?"}: {result.exc_message}',
          )
          self.assertEqual(result.value, expected, f'{label}: value mismatch')
        if expected_exc is not _UNSET:
          self.assertFalse(
            result.success,
            f'{label}: expected {expected_exc.__name__} but succeeded with {result.value!r}',
          )
          self.assertEqual(result.exc_type, expected_exc, f'{label}: exception type mismatch')
          if expected_msg is not None:
            self.assertIn(expected_msg, result.exc_message or '', f'{label}: message mismatch')

    return results


# ---------------------------------------------------------------------------
# Callable fixtures — sync/async pairs
# ---------------------------------------------------------------------------


def sync_fn(x: Any) -> Any:
  return x + 1


async def async_fn(x: Any) -> Any:
  return x + 1


def sync_identity(x: Any) -> Any:
  return x


async def async_identity(x: Any) -> Any:
  return x


def sync_double(x: Any) -> Any:
  return x * 2


async def async_double(x: Any) -> Any:
  return x * 2


def sync_is_even(x: Any) -> bool:
  return x % 2 == 0


async def async_is_even(x: Any) -> bool:
  return x % 2 == 0


def sync_is_truthy(x: Any) -> bool:
  return bool(x)


async def async_is_truthy(x: Any) -> bool:
  return bool(x)


def sync_raise(x: Any) -> Any:
  raise ValueError('test error')


async def async_raise(x: Any) -> Any:
  raise ValueError('test error')


def sync_noop(x: Any) -> None:
  return None


async def async_noop(x: Any) -> None:
  return None


def sync_always_true(x: Any) -> bool:
  return True


async def async_always_true(x: Any) -> bool:
  return True


def sync_always_false(x: Any) -> bool:
  return False


async def async_always_false(x: Any) -> bool:
  return False


# Multi-arg fixtures for calling convention testing
def sync_add(a: Any, b: Any) -> Any:
  return a + b


async def async_add(a: Any, b: Any) -> Any:
  return a + b


def sync_kw(*, key: Any) -> Any:
  return key


async def async_kw(*, key: Any) -> Any:
  return key


def sync_triple(x: Any) -> Any:
  return x * 3


async def async_triple(x: Any) -> Any:
  return x * 3


def sync_gt0(x: Any) -> bool:
  return x > 0


async def async_gt0(x: Any) -> bool:
  return x > 0


# Error handler fixtures (for except_() axis testing)
def sync_handler(info: Any) -> str:
  return 'handled'


async def async_handler(info: Any) -> str:
  return 'handled'


# Cleanup handler that raises (for finally_() failure testing)
def sync_bad_cleanup(rv: Any) -> None:
  raise RuntimeError('cleanup boom')


async def async_bad_cleanup(rv: Any) -> None:
  raise RuntimeError('cleanup boom')


# ---------------------------------------------------------------------------
# Context manager fixtures
# ---------------------------------------------------------------------------


class SyncCM:
  """Sync CM returning a numeric value for pipeline compatibility."""

  def __init__(self, value: Any = 10) -> None:
    self._value = value

  def __enter__(self) -> Any:
    return self._value

  def __exit__(self, *args: Any) -> bool:
    return False


class AsyncCM:
  """Async CM returning a numeric value for pipeline compatibility."""

  def __init__(self, value: Any = 10) -> None:
    self._value = value

  async def __aenter__(self) -> Any:
    return self._value

  async def __aexit__(self, *args: Any) -> bool:
    return False


class SyncCMSuppresses:
  def __enter__(self) -> Any:
    return 10

  def __exit__(self, *args: Any) -> bool:
    return True


class AsyncCMSuppresses:
  async def __aenter__(self) -> Any:
    return 10

  async def __aexit__(self, *args: Any) -> bool:
    return True


class DualProtocolCM:
  """Dual-protocol CM: sync __enter__/__exit__ + async __aenter__/__aexit__.

  The bridge runner exercises both protocols naturally:
  - Pure sync permutations use __enter__/__exit__
  - Async permutations (with running event loop) prefer __aenter__/__aexit__
  Both return the same numeric value for pipeline compatibility.
  """

  def __init__(self, value: Any = 10) -> None:
    self._value = value

  def __enter__(self) -> Any:
    return self._value

  def __exit__(self, *args: Any) -> bool:
    return False

  async def __aenter__(self) -> Any:
    return self._value

  async def __aexit__(self, *args: Any) -> bool:
    return False


# ---------------------------------------------------------------------------
# Iterable fixtures
# ---------------------------------------------------------------------------


class AsyncRange:
  """Async iterable over range(n)."""

  def __init__(self, n: int) -> None:
    self._n = n

  def __aiter__(self) -> Any:
    return self._gen()

  async def _gen(self) -> Any:
    for i in range(self._n):
      yield i


# ---------------------------------------------------------------------------
# Exception fixtures
# ---------------------------------------------------------------------------


class CustomError(Exception):
  """Custom exception for type-specific testing."""


class CustomBaseError(BaseException):
  """Custom BaseException for BaseException filtering tests."""


# ---------------------------------------------------------------------------
# Variant axes — sync/async dimensions
# ---------------------------------------------------------------------------

V_FN: VariantAxis = [('sync', sync_fn), ('async', async_fn)]
V_IDENTITY: VariantAxis = [('sync', sync_identity), ('async', async_identity)]
V_DOUBLE: VariantAxis = [('sync', sync_double), ('async', async_double)]
V_IS_EVEN: VariantAxis = [('sync', sync_is_even), ('async', async_is_even)]
V_IS_TRUTHY: VariantAxis = [('sync', sync_is_truthy), ('async', async_is_truthy)]
V_RAISE: VariantAxis = [('sync', sync_raise), ('async', async_raise)]
V_NOOP: VariantAxis = [('sync', sync_noop), ('async', async_noop)]
V_TRUE: VariantAxis = [('sync', sync_always_true), ('async', async_always_true)]
V_FALSE: VariantAxis = [('sync', sync_always_false), ('async', async_always_false)]
V_CM: VariantAxis = [('sync', SyncCM), ('async', AsyncCM)]
V_CM_SUPPRESSES: VariantAxis = [('sync', SyncCMSuppresses), ('async', AsyncCMSuppresses)]
V_ITER: VariantAxis = [('list', list(range(5))), ('async', AsyncRange(5))]
V_ADD: VariantAxis = [('sync', sync_add), ('async', async_add)]
V_KW: VariantAxis = [('sync', sync_kw), ('async', async_kw)]
V_TRIPLE: VariantAxis = [('sync', sync_triple), ('async', async_triple)]
V_GT0: VariantAxis = [('sync', sync_gt0), ('async', async_gt0)]
V_HANDLER: VariantAxis = [('sync', sync_handler), ('async', async_handler)]
V_BAD_CLEANUP: VariantAxis = [('sync', sync_bad_cleanup), ('async', async_bad_cleanup)]


# ===========================================================================
# EXHAUSTIVE BRIDGE TESTING
# ===========================================================================
#
# The 7 axes:
#   1. Operation type     — then, do, map, foreach_do, gather, with_, if_
#   2. Chain length       — 1 through 4
#   3. Operation order    — all permutations of length L from the operation set
#   4. Sync/async         — each callable position independently sync or async
#   5. Calling convention — default, explicit args
#   6. Error path         — which position (if any) raises an exception
#   7. Concurrency        — None or N (for ops that support it)
#
# Axes 1-3 collapse into "brick permutations."
# Axes 4-7 are per-position variants on each brick.
#
# A "brick" is a self-contained, type-normalizing chain segment:
#   input: number → does its operation → output: number
# This ensures all permutations are type-compatible.


# ---------------------------------------------------------------------------
# Brick definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Brick:
  """A self-contained, type-normalizing chain operation.

  Each brick takes a numeric pipeline value, applies its operation,
  and returns a numeric value.  This ensures any permutation of
  bricks is type-compatible.

  Attributes:
    name: Human-readable name (e.g. 'then', 'map_default').
    op: The operation type (e.g. 'then', 'map', 'gather').
    apply: Callable(chain, fn) that appends the operation to the chain.
        ``fn`` is the sync/async variant to use.
    oracle: Callable(value, fn) that computes the expected result
        using plain Python (no quent).  Used to verify correctness,
        not just symmetry.
    supports_concurrency: Whether this op accepts a concurrency param.
    calling_convention: Which calling convention this brick exercises.
    fn_input: Callable(pipeline_value) -> value that fn actually receives.
        For default convention this is the pipeline value itself.
        For args convention this is the explicit arg (e.g. 42).
        Used to compute return_signal values in oracle checks.
        None means "use pipeline value" (same as lambda v: v).
  """

  name: str
  op: str
  apply: Callable[..., Any]
  oracle: Callable[..., Any]
  supports_concurrency: bool = False
  calling_convention: str = 'default'
  fn_input: Callable[..., Any] | None = None


def _make_bricks() -> list[Brick]:
  """Build the complete set of bricks covering all operations x calling conventions."""
  bricks: list[Brick] = []

  # ---- then ----
  # default: fn(current_value)
  bricks.append(
    Brick(
      name='then_default',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(fn),
      oracle=lambda v, fn: fn(v),
    )
  )
  # explicit args: fn(42) — current value NOT passed
  bricks.append(
    Brick(
      name='then_args',
      op='then',
      calling_convention='args',
      apply=lambda c, fn: c.then(fn, 42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- do ----
  # default: fn(current_value), result discarded
  bricks.append(
    Brick(
      name='do_default',
      op='do',
      calling_convention='default',
      apply=lambda c, fn: c.do(fn),
      oracle=lambda v, fn: v,  # value passes through unchanged
    )
  )
  # explicit args
  bricks.append(
    Brick(
      name='do_args',
      op='do',
      calling_convention='args',
      apply=lambda c, fn: c.do(fn, 42),
      oracle=lambda v, fn: v,
      fn_input=lambda v: 42,
    )
  )

  # ---- map ----
  # default: map each element, collect to list, then sum to normalize to number
  bricks.append(
    Brick(
      name='map_default',
      op='map',
      calling_convention='default',
      supports_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach(fn).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do ----
  # default: side-effect, originals kept, sum to normalize
  bricks.append(
    Brick(
      name='foreach_do_default',
      op='foreach_do',
      calling_convention='default',
      supports_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach_do(fn).then(sum),
      oracle=lambda v, fn: v + (v + 1),  # originals preserved, summed
    )
  )

  # ---- gather ----
  # default: run fns, get tuple, take first element to normalize
  bricks.append(
    Brick(
      name='gather_default',
      op='gather',
      calling_convention='default',
      supports_concurrency=True,
      apply=lambda c, fn: c.gather(fn, fn).then(lambda t: t[0]),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_ ----
  # default: enter CM, fn(ctx), result replaces value
  bricks.append(
    Brick(
      name='with_default',
      op='with_',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: SyncCM(x)).with_(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_do ----
  # default: enter CM, fn(ctx) as side-effect, value passes through
  bricks.append(
    Brick(
      name='with_do_default',
      op='with_do',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: SyncCM(x)).with_do(fn).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,  # CM instance passes through, we extract _value
    )
  )

  # ---- if_ (truthy path) ----
  # default: predicate always true, then=fn, result replaces value
  bricks.append(
    Brick(
      name='if_true_default',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: True).then(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- if_ (falsy path with else) ----
  bricks.append(
    Brick(
      name='if_false_else',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- if_ (no predicate, uses value truthiness) ----
  bricks.append(
    Brick(
      name='if_no_pred',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_().then(fn),
      oracle=lambda v, fn: fn(v) if v else v,  # truthy → apply fn
    )
  )

  # ---- then (nested chain — standard rules apply) ----
  # Nested chain receives current_value, applies fn
  bricks.append(
    Brick(
      name='then_nested_chain',
      op='then',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(Chain().then(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )
  # Nested chain with explicit args (explicit args → Rule 1)
  bricks.append(
    Brick(
      name='then_nested_chain_args',
      op='then',
      calling_convention='nested_chain_args',
      apply=lambda c, fn: c.then(Chain().then(fn), 42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )
  # Non-callable literal value (Rule 2 non-callable path)
  bricks.append(
    Brick(
      name='then_literal',
      op='then',
      calling_convention='literal',
      apply=lambda c, fn: c.then(fn).then(42).then(fn),
      oracle=lambda v, fn: fn(42),
    )
  )

  # ---- do (nested chain — standard rules apply) ----
  bricks.append(
    Brick(
      name='do_nested_chain',
      op='do',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.do(Chain().then(fn)),
      oracle=lambda v, fn: v,
    )
  )

  # ---- with_ (explicit args, Rule 1) ----
  bricks.append(
    Brick(
      name='with_args',
      op='with_',
      calling_convention='args',
      apply=lambda c, fn: c.then(lambda x: SyncCM(x)).with_(fn, 42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- with_do (explicit args, Rule 1) ----
  bricks.append(
    Brick(
      name='with_do_args',
      op='with_do',
      calling_convention='args',
      apply=lambda c, fn: c.then(lambda x: SyncCM(x)).with_do(fn, 42).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
      fn_input=lambda v: 42,
    )
  )

  # ---- with_ (dual-protocol CM — exercises _full_async in async perms) ----
  bricks.append(
    Brick(
      name='with_dual_cm',
      op='with_',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_ (nested chain body — standard rules, chain runs with ctx value) ----
  bricks.append(
    Brick(
      name='with_nested_chain',
      op='with_',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_(Chain().then(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_do (nested chain body — standard rules, chain runs, result discarded) ----
  bricks.append(
    Brick(
      name='with_do_nested_chain',
      op='with_do',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_do(Chain().then(fn)).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
    )
  )

  # ---- if_ (do branch — ignore_result path) ----
  bricks.append(
    Brick(
      name='if_do_branch',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: True).do(fn),
      oracle=lambda v, fn: v,
    )
  )

  # ---- if_ (fn as predicate — async predicate dispatch) ----
  # fn is always truthy for positive inputs (x+1 > 0)
  bricks.append(
    Brick(
      name='if_pred_fn',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(fn).then(lambda x: x * 2),
      oracle=lambda v, fn: v * 2,
    )
  )

  # ---- if_ (predicate with explicit args, Rule 1) ----
  # fn(42) is always truthy (42+1=43 > 0)
  bricks.append(
    Brick(
      name='if_pred_args',
      op='if_',
      calling_convention='pred_args',
      apply=lambda c, fn: c.if_(fn, 42).then(lambda x: x * 2),
      oracle=lambda v, fn: v * 2,
      fn_input=lambda v: 42,
    )
  )

  # ---- if_ (predicate is nested chain — standard rules apply) ----
  # Nested chain returns fn(current_value); fn(10)=11 is truthy
  bricks.append(
    Brick(
      name='if_pred_nested_chain',
      op='if_',
      calling_convention='pred_nested_chain',
      apply=lambda c, fn: c.if_(Chain().then(fn)).then(lambda x: x * 2),
      oracle=lambda v, fn: v * 2,
    )
  )

  # ---- if_ (predicate is non-callable literal, Rule 2 non-callable) ----
  # Literal True is always truthy
  bricks.append(
    Brick(
      name='if_pred_literal',
      op='if_',
      calling_convention='pred_literal',
      apply=lambda c, fn: c.if_(True).then(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- else_ (nested chain — standard rules apply) ----
  bricks.append(
    Brick(
      name='else_nested_chain',
      op='if_',
      calling_convention='else_nested_chain',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_(Chain().then(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- else_ (explicit args, Rule 1) ----
  bricks.append(
    Brick(
      name='else_args',
      op='if_',
      calling_convention='else_args',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_(fn, 42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- else_ (non-callable literal, Rule 2 non-callable) ----
  bricks.append(
    Brick(
      name='else_literal',
      op='if_',
      calling_convention='else_literal',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_(42).then(fn),
      oracle=lambda v, fn: fn(42),
    )
  )

  # ---- else_do (side-effect, discards fn result) ----
  bricks.append(
    Brick(
      name='else_do',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_do(fn),
      oracle=lambda v, fn: v,  # else_do discards fn's return, preserves pipeline value
    )
  )

  return bricks


ALL_BRICKS = _make_bricks()

# Index bricks by operation type for targeted testing
BRICKS_BY_OP: dict[str, list[Brick]] = {}
for _b in ALL_BRICKS:
  BRICKS_BY_OP.setdefault(_b.op, []).append(_b)


# ---------------------------------------------------------------------------
# Unified bridge runner
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BridgeStats:
  """Accumulated statistics from a bridge test run."""

  total_orderings: int = 0
  total_combinations: int = 0
  total_configs: int = 0
  failures: list[str] = field(default_factory=list)


async def run_bridge(
  test: IsolatedAsyncioTestCase,
  *,
  bricks: list[Brick] | None = None,
  max_length: int = 3,
  min_length: int = 1,
  include_error_path: bool = True,
  exclude_gather_error: bool = True,
  include_concurrency: bool = True,
  handler_configs: list[HandlerConfig] | None = None,
  error_types: list[tuple[str, Any, Any]] | None = None,
  nesting_depths: list[int] | None = None,
  include_handler_async: bool = False,
  gather_aware: bool = False,
) -> BridgeStats:
  """Unified bridge test runner.

  Modes (determined by parameters):
    - handler_configs=None: exhaustive bridge -- all bricks, all orderings,
      sync/async permutations, optional error path + concurrency. No handler axis.
    - handler_configs set, gather_aware=False: error bridge -- error types,
      handler configs, handler sync/async, nesting, concurrency.
    - gather_aware=True: gather error bridge -- error injection at gather
      positions only, asymmetry-aware comparison (ExceptionGroup wrapping).

  For every combination, runs ALL sync/async step permutations and asserts
  they produce the same result. Oracle correctness checks validate expected
  values where applicable.

  Args:
    test: The test case instance (for assertions and subTest).
    bricks: Brick pool to permute. Defaults to ALL_BRICKS.
    max_length: Maximum chain length.
    min_length: Minimum chain length.
    include_error_path: Include error position axis (positions 0..length-1).
    exclude_gather_error: Skip error injection at gather brick positions.
    include_concurrency: Include concurrency axis for supporting bricks.
    handler_configs: Handler configurations. None = skip handler axis entirely.
    error_types: Error injection types. Defaults to ERROR_INJECTION_TYPES
        when handler_configs is set.
    nesting_depths: Nesting depths to test. Defaults to [0].
    include_handler_async: Include handler sync/async axis.
    gather_aware: Use asymmetry-aware comparison for gather error paths.

  Returns:
    BridgeStats with totals and any failures.
  """
  if bricks is None:
    bricks = ALL_BRICKS

  has_handlers = handler_configs is not None

  if nesting_depths is None:
    nesting_depths = [0]
  if not has_handlers:
    nesting_depths = [0]
  if has_handlers and error_types is None:
    error_types = ERROR_INJECTION_TYPES

  handler_async_variants: list[tuple[str, bool]] = [('sync_h', False)]
  if include_handler_async and has_handlers:
    handler_async_variants.append(('async_h', True))

  handler_configs_iter: list[HandlerConfig | None] = list(handler_configs) if has_handlers else [None]

  stats = BridgeStats()
  _log = lambda msg: print(msg, flush=True)

  n = len(bricks)
  t0 = time.monotonic()
  last_log_time = t0

  total_orderings_est = sum(math.perm(n, k) for k in range(min_length, max_length + 1))
  _log(
    f'\n[bridge] starting: {n} bricks, lengths {min_length}-{max_length}, '
    f'~{total_orderings_est} orderings, '
    f'handlers={"yes" if has_handlers else "no"}, '
    f'gather_aware={gather_aware}, '
    f'error_path={include_error_path}, concurrency={include_concurrency}'
  )

  # Capture warnings instead of ignoring them.  Bridge tests legitimately trigger
  # RuntimeWarning from quent (e.g., when an except_ handler itself fails — SPEC
  # 6.3.5).  We record all warnings and validate after the run that only
  # expected quent-internal RuntimeWarnings were emitted.  Any other warning
  # category or non-quent RuntimeWarning will cause a test failure, ensuring
  # spec-violating warnings are never silently swallowed.
  with warnings.catch_warnings(record=True) as _bridge_warnings:
    warnings.simplefilter('always')

    for length in range(min_length, max_length + 1):
      length_t0 = time.monotonic()
      length_orderings = math.perm(n, length)
      _log(f'[bridge] length={length}: {length_orderings} orderings')

      for perm in itertools.permutations(bricks, length):
        stats.total_orderings += 1
        ordering_name = ' -> '.join(b.name for b in perm)

        # Determine error positions
        if gather_aware:
          error_positions = [i for i, b in enumerate(perm) if b.op == 'gather']
          if not error_positions:
            continue
        elif include_error_path:
          error_positions = list(range(-1, length))
        else:
          error_positions = [-1]

        # Concurrency variants per brick
        conc_axes: list[list[tuple[str, int | None]]] = []
        for b in perm:
          if include_concurrency and b.supports_concurrency:
            conc_axes.append([('seq', None), ('conc', 2)])
          else:
            conc_axes.append([('seq', None)])

        for error_pos in error_positions:
          # Skip error injection at gather positions if requested (non-gather-aware mode)
          if exclude_gather_error and not gather_aware and error_pos >= 0 and perm[error_pos].op == 'gather':
            continue

          # Determine active error types for this position
          if error_pos < 0:
            active_error_types: list[tuple[str, Any, Any]] = [('none', sync_fn, async_fn)]
          elif has_handlers:
            active_error_types = list(error_types)
          else:
            active_error_types = [('exception', sync_raise, async_raise)]

          for err_type_name, err_sync, err_async in active_error_types:
            for hconfig in handler_configs_iter:
              for ha_label, is_handler_async in handler_async_variants:
                # Skip async handler variant for configs with finally_ handlers
                # (SPEC 6.3.5: documented asymmetry, not a bridge violation)
                if hconfig is not None and is_handler_async and hconfig.has_finally:
                  continue
                for nesting_depth in nesting_depths:
                  for conc_combo in itertools.product(*conc_axes):
                    # Build sync/async axis: 2^length combinations
                    sa_axes: list[list[tuple[str, Any]]] = []
                    for i in range(length):
                      if error_pos == i:
                        sa_axes.append([('sync_err', err_sync), ('async_err', err_async)])
                      else:
                        sa_axes.append([('sync', sync_fn), ('async', async_fn)])

                    # Run all sync/async combos and assert symmetry
                    results: list[Result] = []
                    combo_labels: list[str] = []

                    for sa_combo in itertools.product(*sa_axes):
                      conc_label = ','.join(c[0] for c in conc_combo)
                      sa_label = ','.join(s[0] for s in sa_combo)
                      err_label = f'err@{error_pos}({err_type_name})' if error_pos >= 0 else 'ok'
                      h_label = hconfig.name if hconfig is not None else 'no_handler'
                      label = (
                        f'{ordering_name} [{sa_label}] [{conc_label}] '
                        f'[{err_label}] [{h_label}] [{ha_label}] '
                        f'[nest={nesting_depth}]'
                      )
                      combo_labels.append(label)

                      async def _run_chain(
                        _perm=perm,
                        _sa=sa_combo,
                        _conc=conc_combo,
                        _hconfig=hconfig,
                        _is_ha=is_handler_async,
                        _ndepth=nesting_depth,
                      ) -> Result:
                        def _build() -> Any:
                          c = Chain(10)
                          for _i, (brick, (_, fn), (_, conc_val)) in enumerate(zip(_perm, _sa, _conc)):
                            _apply_brick_with_options(c, brick, fn, conc_val, _ndepth)
                          if _hconfig is not None:
                            _hconfig.apply(c, _is_ha)
                          return c.run()

                        return await capture(_build)

                      result = await _run_chain()
                      results.append(result)
                      stats.total_combinations += 1

                    stats.total_configs += 1

                    # Assert symmetry
                    symmetry_ok = True
                    if results:
                      first = results[0]
                      for i in range(1, len(results)):
                        if gather_aware:
                          equiv, reason = gather_results_equivalent(first, results[i])
                          if not equiv:
                            symmetry_ok = False
                            failure_msg = (
                              f'GATHER BRIDGE BROKEN: {combo_labels[0]} != {combo_labels[i]}\n'
                              f'  first:  {first}\n  other:  {results[i]}\n'
                              f'  reason: {reason}'
                            )
                            stats.failures.append(failure_msg)
                            test.fail(failure_msg)
                        else:
                          if (
                            first.success != results[i].success
                            or (first.success and first.value != results[i].value)
                            or (not first.success and first.exc_type != results[i].exc_type)
                          ):
                            symmetry_ok = False
                            failure_msg = (
                              f'BRIDGE BROKEN: {combo_labels[0]} != {combo_labels[i]}\n'
                              f'  first:  {first}\n  other:  {results[i]}'
                            )
                            stats.failures.append(failure_msg)
                            test.fail(failure_msg)

                      # Oracle correctness check (skip for gather_aware and when error brick has concurrency)
                      error_has_conc = error_pos >= 0 and conc_combo[error_pos][1] is not None
                      if symmetry_ok and not error_has_conc and not gather_aware:
                        if hconfig is not None:
                          # Error bridge oracle (only at nesting_depth 0)
                          if nesting_depth == 0 and hconfig.oracle is not None:
                            happy_value = compose_oracles(perm, 10)
                            if error_pos >= 0:
                              pipeline_at_error = compose_oracles(perm[:error_pos], 10)
                              error_brick = perm[error_pos]
                              if error_brick.fn_input is not None:
                                partial_value = error_brick.fn_input(pipeline_at_error)
                              else:
                                partial_value = pipeline_at_error
                            else:
                              partial_value = happy_value
                            expected = hconfig.oracle(err_type_name, happy_value, partial_value)
                            if expected is not None:
                              ref = results[0]
                              if (
                                ref.success != expected.success
                                or (ref.success and ref.value != expected.value)
                                or (not ref.success and ref.exc_type != expected.exc_type)
                              ):
                                failure_msg = (
                                  f'ORACLE MISMATCH: {combo_labels[0]}\n  actual:   {ref}\n  expected: {expected}'
                                )
                                stats.failures.append(failure_msg)
                                test.fail(failure_msg)
                        else:
                          # Exhaustive bridge oracle
                          ref = results[0]
                          if error_pos < 0:
                            expected_value = compose_oracles(perm, 10, sync_fn)
                            expected = Result(success=True, value=expected_value)
                          else:
                            expected = Result(success=False, exc_type=ValueError)
                          if (
                            ref.success != expected.success
                            or (ref.success and ref.value != expected.value)
                            or (not ref.success and ref.exc_type != expected.exc_type)
                          ):
                            failure_msg = (
                              f'ORACLE MISMATCH: {combo_labels[0]}\n  actual:   {ref}\n  expected: {expected}'
                            )
                            stats.failures.append(failure_msg)
                            test.fail(failure_msg)

                    # Progress logging every 10s
                    now = time.monotonic()
                    if now - last_log_time >= 10.0:
                      elapsed = now - t0
                      _log(
                        f'[bridge] {stats.total_combinations} combos, '
                        f'{stats.total_configs} configs, '
                        f'{elapsed:.0f}s elapsed, '
                        f'{len(stats.failures)} failures'
                      )
                      last_log_time = now

      _log(f'[bridge] length={length} done in {time.monotonic() - length_t0:.1f}s')

    # Validate captured warnings: only quent-internal RuntimeWarnings are expected.
    # Any other warning type (or a RuntimeWarning not from quent) indicates a
    # spec violation that must not be silently swallowed.
    unexpected = [
      w
      for w in _bridge_warnings
      if not (issubclass(w.category, RuntimeWarning) and str(w.message).startswith('quent: '))
    ]
    if unexpected:
      msgs = '\n'.join(f'  {w.filename}:{w.lineno}: [{w.category.__name__}] {w.message}' for w in unexpected)
      test.fail(f'[bridge] unexpected warnings emitted during bridge run:\n{msgs}')

  elapsed = time.monotonic() - t0
  _log(
    f'[bridge] done: {stats.total_orderings} orderings, '
    f'{stats.total_configs} configs, '
    f'{stats.total_combinations} combos, {elapsed:.1f}s, '
    f'{len(stats.failures)} failures'
  )
  return stats


def _apply_with_conc(c: Chain[Any], brick: Brick, fn: Any, conc: int) -> None:
  """Apply a brick with concurrency parameter.

  Rebuilds the operation call to include ``concurrency=conc``.
  Only called for bricks where ``supports_concurrency=True``.
  """
  op = brick.op
  if op == 'map':
    c.then(lambda x: [x, x + 1]).foreach(fn, concurrency=conc).then(sum)
  elif op == 'foreach_do':
    c.then(lambda x: [x, x + 1]).foreach_do(fn, concurrency=conc).then(sum)
  elif op == 'gather':
    c.gather(fn, fn, concurrency=conc).then(lambda t: t[0])


# ---------------------------------------------------------------------------
# Oracle-based correctness verification
# ---------------------------------------------------------------------------


def compose_oracles(bricks: list[Brick] | tuple[Brick, ...], input_value: Any, fn: Any = None) -> Any:
  """Chain all brick oracles sequentially to compute the expected value.

  Args:
    bricks: Sequence of bricks whose oracles to compose.
    input_value: The initial pipeline value.
    fn: The callable to pass to each oracle. Defaults to sync_fn.

  Returns:
    The expected value after all bricks are applied.
  """
  if fn is None:
    fn = sync_fn
  value = input_value
  for brick in bricks:
    value = brick.oracle(value, fn)
  return value


async def verify_brick_oracle(
  test: IsolatedAsyncioTestCase,
  brick: Brick,
  input_value: Any = 10,
) -> None:
  """Verify a single brick produces the oracle-predicted result.

  Runs the brick with sync fn and compares against the oracle
  computed with the same fn.  This validates the brick definition
  itself is correct.
  """
  c = Chain(input_value)
  brick.apply(c, sync_fn)
  result = await capture(c.run)
  expected = brick.oracle(input_value, sync_fn)
  test.assertTrue(result.success, f'{brick.name}: unexpected error: {result.exc_message}')
  test.assertEqual(
    result.value,
    expected,
    f'{brick.name}: oracle={expected}, got={result.value}',
  )


async def verify_all_brick_oracles(
  test: IsolatedAsyncioTestCase,
  input_value: Any = 10,
) -> None:
  """Verify every brick's oracle is correct."""
  for brick in ALL_BRICKS:
    with test.subTest(brick=brick.name):
      await verify_brick_oracle(test, brick, input_value)


# ---------------------------------------------------------------------------
# Except handler fixtures (1-arg convention: current_value = exc)
# ---------------------------------------------------------------------------


def sync_except_consume(exc: Any) -> str:
  return 'recovered'


async def async_except_consume(exc: Any) -> str:
  return 'recovered'


def sync_except_noop(exc: Any) -> None:
  pass


async def async_except_noop(exc: Any) -> None:
  pass


def sync_except_fails(exc: Any) -> Any:
  raise RuntimeError('handler boom')


async def async_except_fails(exc: Any) -> Any:
  raise RuntimeError('handler boom')


# ---------------------------------------------------------------------------
# Finally handler fixtures
# ---------------------------------------------------------------------------


def sync_finally_ok(rv: Any) -> None:
  pass


async def async_finally_ok(rv: Any) -> None:
  pass


# sync_bad_cleanup / async_bad_cleanup already exist above


# ---------------------------------------------------------------------------
# Error injection fixtures
# ---------------------------------------------------------------------------


def sync_raise_base(x: Any) -> Any:
  raise CustomBaseError('base error')


async def async_raise_base(x: Any) -> Any:
  raise CustomBaseError('base error')


def sync_return_signal(x: Any) -> Any:
  return Chain.return_(x * 100)


async def async_return_signal(x: Any) -> Any:
  return Chain.return_(x * 100)


def sync_break_signal(x: Any) -> Any:
  Chain.break_(x * 100)


async def async_break_signal(x: Any) -> Any:
  Chain.break_(x * 100)


# ---------------------------------------------------------------------------
# HandlerConfig and handler setup functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandlerConfig:
  """Configuration for attaching error handlers to a chain.

  The ``oracle`` callable computes the expected Result for a given
  (error_type, happy_value, partial_value) combination:
    - error_type: 'none', 'exception', 'base_exception', 'return_signal', 'break_signal'
    - happy_value: the composed oracle value when all bricks succeed
    - partial_value: the composed oracle value up to error_pos (for return_signal)
  Returns None to skip oracle checking (e.g. break_signal).
  """

  name: str
  apply: Callable[..., Any]  # (chain, is_handler_async: bool) -> None
  has_finally: bool = False  # True if this config attaches a finally_ handler
  oracle: Callable[..., Result | None] | None = None


def _apply_no_handler(chain: Any, is_handler_async: bool) -> None:
  pass


def _apply_except_consume(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_consume if is_handler_async else sync_except_consume
  chain.except_(handler)


def _apply_except_reraise(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_noop if is_handler_async else sync_except_noop
  chain.except_(handler, reraise=True)


def _apply_except_fails(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_fails if is_handler_async else sync_except_fails
  chain.except_(handler)


def _apply_finally_ok(chain: Any, is_handler_async: bool) -> None:
  handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.finally_(handler)


def _apply_finally_fails(chain: Any, is_handler_async: bool) -> None:
  handler = async_bad_cleanup if is_handler_async else sync_bad_cleanup
  chain.finally_(handler)


def _apply_except_consume_finally_ok(chain: Any, is_handler_async: bool) -> None:
  exc_handler = async_except_consume if is_handler_async else sync_except_consume
  fin_handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.except_(exc_handler)
  chain.finally_(fin_handler)


def _apply_except_reraise_finally_ok(chain: Any, is_handler_async: bool) -> None:
  exc_handler = async_except_noop if is_handler_async else sync_except_noop
  fin_handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.except_(exc_handler, reraise=True)
  chain.finally_(fin_handler)


def _apply_except_consume_finally_fails(chain: Any, is_handler_async: bool) -> None:
  exc_handler = async_except_consume if is_handler_async else sync_except_consume
  fin_handler = async_bad_cleanup if is_handler_async else sync_bad_cleanup
  chain.except_(exc_handler)
  chain.finally_(fin_handler)


def _apply_except_with_args(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_consume if is_handler_async else sync_except_consume
  chain.except_(handler, 'injected_arg')


def _apply_except_nested_chain(chain: Any, is_handler_async: bool) -> None:
  handler = async_except_consume if is_handler_async else sync_except_consume
  chain.except_(Chain().then(handler))


def _apply_finally_with_args(chain: Any, is_handler_async: bool) -> None:
  handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.finally_(handler, 'injected_arg')


def _apply_finally_nested_chain(chain: Any, is_handler_async: bool) -> None:
  handler = async_finally_ok if is_handler_async else sync_finally_ok
  chain.finally_(Chain().then(handler))


# ---------------------------------------------------------------------------
# Handler config oracle functions
# ---------------------------------------------------------------------------
#
# Each oracle: (error_type, happy_value, partial_value) -> Result | None
#   error_type: 'none', 'exception', 'base_exception', 'return_signal', 'break_signal'
#   happy_value: composed oracle value (all bricks succeed)
#   partial_value: composed oracle value up to error_pos (for return_signal)
#   Returns None to skip oracle checking (e.g. break_signal).


def _oracle_no_handler(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_consume(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    # BaseException not caught by except_ (which uses Exception)
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    # _Return caught by engine before except_
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_reraise(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=RuntimeError)
  if error_type == 'base_exception':
    # BaseException not caught by except_
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_ok: no except handler, finally runs but doesn't affect result."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_fails: finally raises RuntimeError, always overrides outcome."""
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError regardless of error_type
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_consume_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_reraise_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_consume_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_reraise_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_fails_finally_ok(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=RuntimeError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_fails_finally_fails(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  if error_type == 'break_signal':
    return None
  # finally_fails always raises RuntimeError
  return Result(success=False, exc_type=RuntimeError)


def _oracle_except_with_args(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """except_with_args: handler('injected_arg') -> 'recovered' (Rule 1: args override cv)."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_except_nested_chain(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """except_nested_chain: Chain().then(consume) -> 'recovered'."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=True, value='recovered')
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_with_args(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_with_args: no except, finally runs (ok) but doesn't affect result."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


def _oracle_finally_nested_chain(error_type: str, happy_value: Any, partial_value: Any) -> Result | None:
  """finally_nested_chain: no except, finally runs (ok) but doesn't affect result."""
  if error_type == 'break_signal':
    return None
  if error_type == 'none':
    return Result(success=True, value=happy_value)
  if error_type == 'exception':
    return Result(success=False, exc_type=ValueError)
  if error_type == 'base_exception':
    return Result(success=False, exc_type=CustomBaseError)
  if error_type == 'return_signal':
    return Result(success=True, value=partial_value * 100)
  return None


HANDLER_CONFIGS: list[HandlerConfig] = [
  HandlerConfig(name='no_handler', apply=_apply_no_handler, oracle=_oracle_no_handler),
  HandlerConfig(name='except_consume', apply=_apply_except_consume, oracle=_oracle_except_consume),
  HandlerConfig(name='except_reraise', apply=_apply_except_reraise, oracle=_oracle_except_reraise),
  HandlerConfig(name='except_fails', apply=_apply_except_fails, oracle=_oracle_except_fails),
  HandlerConfig(name='finally_ok', apply=_apply_finally_ok, has_finally=True, oracle=_oracle_finally_ok),
  HandlerConfig(name='finally_fails', apply=_apply_finally_fails, has_finally=True, oracle=_oracle_finally_fails),
  HandlerConfig(
    name='except_consume+finally_ok',
    apply=_apply_except_consume_finally_ok,
    has_finally=True,
    oracle=_oracle_except_consume_finally_ok,
  ),
  HandlerConfig(
    name='except_reraise+finally_ok',
    apply=_apply_except_reraise_finally_ok,
    has_finally=True,
    oracle=_oracle_except_reraise_finally_ok,
  ),
  HandlerConfig(
    name='except_consume+finally_fails',
    apply=_apply_except_consume_finally_fails,
    has_finally=True,
    oracle=_oracle_except_consume_finally_fails,
  ),
  HandlerConfig(
    name='except_reraise+finally_fails',
    apply=lambda chain, is_async: chain.except_(
      async_except_noop if is_async else sync_except_noop, reraise=True
    ).finally_(async_bad_cleanup if is_async else sync_bad_cleanup),
    has_finally=True,
    oracle=_oracle_except_reraise_finally_fails,
  ),
  HandlerConfig(
    name='except_fails+finally_ok',
    apply=lambda chain, is_async: chain.except_(async_except_fails if is_async else sync_except_fails).finally_(
      async_finally_ok if is_async else sync_finally_ok
    ),
    has_finally=True,
    oracle=_oracle_except_fails_finally_ok,
  ),
  HandlerConfig(
    name='except_fails+finally_fails',
    apply=lambda chain, is_async: chain.except_(async_except_fails if is_async else sync_except_fails).finally_(
      async_bad_cleanup if is_async else sync_bad_cleanup
    ),
    has_finally=True,
    oracle=_oracle_except_fails_finally_fails,
  ),
  HandlerConfig(name='except_with_args', apply=_apply_except_with_args, oracle=_oracle_except_with_args),
  HandlerConfig(name='except_nested_chain', apply=_apply_except_nested_chain, oracle=_oracle_except_nested_chain),
  HandlerConfig(
    name='finally_with_args', apply=_apply_finally_with_args, has_finally=True, oracle=_oracle_finally_with_args
  ),
  HandlerConfig(
    name='finally_nested_chain',
    apply=_apply_finally_nested_chain,
    has_finally=True,
    oracle=_oracle_finally_nested_chain,
  ),
]


# ---------------------------------------------------------------------------
# Representative bricks (working subset — no ellipsis, no unpack)
# ---------------------------------------------------------------------------


REPRESENTATIVE_BRICKS: list[Brick] = [
  b
  for b in ALL_BRICKS
  if b.name
  in (
    'then_default',
    'then_args',
    'then_nested_chain',
    'do_default',
    'map_default',
    'foreach_do_default',
    'gather_default',
    'with_default',
    'with_dual_cm',
    'if_true_default',
    'if_do_branch',
    'if_pred_fn',
    'if_pred_nested_chain',
    'else_nested_chain',
    'else_do',
  )
]


# ---------------------------------------------------------------------------
# Error injection types
# ---------------------------------------------------------------------------


ERROR_INJECTION_TYPES: list[tuple[str, Any, Any]] = [
  ('exception', sync_raise, async_raise),
  ('base_exception', sync_raise_base, async_raise_base),
  ('return_signal', sync_return_signal, async_return_signal),
  ('break_signal', sync_break_signal, async_break_signal),
]


# ---------------------------------------------------------------------------
# Nesting and brick application helpers
# ---------------------------------------------------------------------------


def _apply_brick_nested(chain: Any, brick: Brick, fn: Any, depth: int) -> None:
  """Apply a brick at a given nesting depth (0=flat, 1=one level, 2=two levels)."""
  if depth <= 0:
    brick.apply(chain, fn)
    return
  inner = Chain()
  _apply_brick_nested(inner, brick, fn, depth - 1)
  chain.then(inner)


def _apply_brick_with_options(chain: Any, brick: Brick, fn: Any, conc_val: int | None, nesting_depth: int) -> None:
  """Apply a brick with optional concurrency and nesting depth."""
  if nesting_depth <= 0:
    if conc_val is not None and brick.supports_concurrency:
      _apply_with_conc(chain, brick, fn, conc_val)
    else:
      brick.apply(chain, fn)
  else:
    inner = Chain()
    _apply_brick_with_options(inner, brick, fn, conc_val, nesting_depth - 1)
    chain.then(inner)


# ---------------------------------------------------------------------------
# Gather error path comparison
# ---------------------------------------------------------------------------
#
# Gather error paths are excluded from the main exhaustive bridge test
# because sync/async gather produce different exception wrapping
# (ExceptionGroup vs direct) — a documented asymmetry (SPEC §16).
#
# This module provides asymmetry-aware bridge testing that verifies
# semantic equivalence without requiring identical exception structure:
#   - Same success/failure outcome
#   - Same exception types within ExceptionGroup (order may differ)
#   - Same result values on success paths
#   - Exception wrapping structure CAN differ


def _is_exception_group(exc_type: type | None) -> bool:
  """Check if an exception type is an ExceptionGroup (builtin or polyfill)."""
  if exc_type is None:
    return False
  return exc_type.__name__ == 'ExceptionGroup' or (hasattr(exc_type, 'exceptions') and issubclass(exc_type, Exception))


def gather_results_equivalent(a: Result, b: Result) -> tuple[bool, str]:
  """Compare two Results with asymmetry-aware semantics.

  Returns (equivalent, reason). If not equivalent, reason describes
  the mismatch.

  Asymmetry-aware rules:
  1. success/failure must match
  2. On success: values must match
  3. On failure with same exc_type: equivalent
  4. On failure with different exc_type but both are ExceptionGroup:
     sub-exception types must match (order-independent)
  5. On failure where one is ExceptionGroup with 1 sub-exception and
     the other is that sub-exception type directly: equivalent
     (single-failure wrapping asymmetry)
  """
  # Rule 1: success/failure must match
  if a.success != b.success:
    return False, f'success mismatch: {a.success} vs {b.success}'

  # Rule 2: on success, values must match
  if a.success:
    if a.value != b.value:
      return False, f'value mismatch: {a.value!r} vs {b.value!r}'
    return True, ''

  # Both failed — compare exception semantics
  # Rule 3: same exc_type
  if a.exc_type == b.exc_type:
    # If both are ExceptionGroups, also verify sub-exception types match
    if (
      _is_exception_group(a.exc_type)
      and a.sub_exc_types is not None
      and b.sub_exc_types is not None
      and a.sub_exc_types != b.sub_exc_types
    ):
      return False, (
        f'ExceptionGroup sub-exception types differ: '
        f'{sorted(t.__name__ for t in a.sub_exc_types)} vs '
        f'{sorted(t.__name__ for t in b.sub_exc_types)}'
      )
    return True, ''

  # Rule 4: both ExceptionGroups with same sub-exception types
  a_is_eg = _is_exception_group(a.exc_type)
  b_is_eg = _is_exception_group(b.exc_type)
  if a_is_eg and b_is_eg:
    if a.sub_exc_types is not None and b.sub_exc_types is not None:
      if a.sub_exc_types == b.sub_exc_types:
        return True, ''
      return False, (
        f'ExceptionGroup sub-exception types differ: '
        f'{sorted(t.__name__ for t in a.sub_exc_types)} vs '
        f'{sorted(t.__name__ for t in b.sub_exc_types)}'
      )
    return True, ''  # Both EG but can't inspect sub-types — accept

  # Rule 5: single-failure wrapping asymmetry
  # One is ExceptionGroup with 1 sub-exc, the other is that sub-exc directly
  if a_is_eg and not b_is_eg and a.sub_exc_types is not None and len(a.sub_exc_types) == 1:
    (sole_type,) = a.sub_exc_types
    if sole_type == b.exc_type:
      return True, ''
  if b_is_eg and not a_is_eg and b.sub_exc_types is not None and len(b.sub_exc_types) == 1:
    (sole_type,) = b.sub_exc_types
    if sole_type == a.exc_type:
      return True, ''

  return False, f'exception type mismatch: {a.exc_type} vs {b.exc_type}'
