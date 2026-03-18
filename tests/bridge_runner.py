"""Unified bridge test runner and oracle verification infrastructure.

Provides ``run_bridge()`` -- the unified bridge test runner covering all axes:
operation type, chain length, operation order (permutations), sync/async per
position, error path, concurrency, error handlers, nesting depth, and
gather-aware asymmetry comparison.

The bridge contract is tested WITHOUT computing expected values: for any fixed
chain configuration, ALL sync/async permutations must produce the same result.
If any permutation differs, the bridge is broken.
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import Any
from unittest import IsolatedAsyncioTestCase

from quent import Chain
from tests.bricks import ALL_BRICKS, Brick
from tests.fixtures import (
  Result,
  async_fn,
  async_raise,
  capture,
  sync_fn,
  sync_raise,
)
from tests.handler_configs import ERROR_INJECTION_TYPES, HandlerConfig

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

  _log(
    f'\n[bridge] starting: {n} bricks, lengths {min_length}-{max_length}, '
    f'handlers={"yes" if has_handlers else "no"}, '
    f'gather_aware={gather_aware}, '
    f'error_path={include_error_path}, concurrency={include_concurrency}'
    f', sequential'
  )

  # Capture warnings instead of ignoring them.  Bridge tests legitimately trigger
  # RuntimeWarning from quent (e.g., when an except_ handler itself fails -- SPEC
  # 6.3.5).  We record all warnings and validate after the run that only
  # expected quent-internal RuntimeWarnings were emitted.  Any other warning
  # category or non-quent RuntimeWarning will cause a test failure, ensuring
  # spec-violating warnings are never silently swallowed.
  with warnings.catch_warnings(record=True) as _bridge_warnings:
    warnings.simplefilter('always')

    for length in range(min_length, max_length + 1):
      length_t0 = time.monotonic()
      length_orderings = n**length
      length_orderings_done = 0
      _log(f'[bridge] length={length}: {length_orderings} orderings')

      for perm in itertools.product(bricks, repeat=length):
        stats.total_orderings += 1
        length_orderings_done += 1
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

                      # Oracle correctness check (skip for gather_aware and when error brick has concurrency)
                      error_has_conc = error_pos >= 0 and (
                        conc_combo[error_pos][1] is not None or perm[error_pos].has_builtin_concurrency
                      )
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

                              # Check if brick absorbs this error type
                              eo = error_brick.error_oracle
                              eo_result = eo(partial_value, err_type_name) if eo is not None else None
                              if eo_result is not None and eo_result.success:
                                # Brick absorbed the error. Pipeline continues as if no error.
                                try:
                                  remaining = compose_oracles(perm[error_pos + 1 :], eo_result.value)
                                except Exception:
                                  expected = None  # Oracle computation itself failed; skip check
                                else:
                                  expected = hconfig.oracle('none', remaining, remaining)
                              else:
                                expected = hconfig.oracle(err_type_name, happy_value, partial_value)
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
                        else:
                          # Exhaustive bridge oracle
                          ref = results[0]
                          if error_pos < 0:
                            expected_value = compose_oracles(perm, 10, sync_fn)
                            expected = Result(success=True, value=expected_value)
                          else:
                            error_brick = perm[error_pos]
                            eo = error_brick.error_oracle
                            if eo is not None:
                              pipeline_at_error = compose_oracles(perm[:error_pos], 10, sync_fn)
                              eo_input = (
                                error_brick.fn_input(pipeline_at_error)
                                if error_brick.fn_input is not None
                                else pipeline_at_error
                              )
                              eo_result = eo(eo_input, 'exception')
                              if eo_result is not None and eo_result.success:
                                # Skip oracle check: absorbed value may violate
                                # downstream brick oracle assumptions (e.g. an oracle
                                # that assumes truthy pipeline values).  Symmetry
                                # checking still validates the bridge contract.
                                expected = None
                              else:
                                expected = Result(success=False, exc_type=ValueError)
                            else:
                              expected = Result(success=False, exc_type=ValueError)
                          if expected is not None and (
                            ref.success != expected.success
                            or (ref.success and ref.value != expected.value)
                            or (not ref.success and ref.exc_type != expected.exc_type)
                          ):
                            failure_msg = (
                              f'ORACLE MISMATCH: {combo_labels[0]}\n  actual:   {ref}\n  expected: {expected}'
                            )
                            stats.failures.append(failure_msg)

                    # Progress logging every 10s
                    now = time.monotonic()
                    if now - last_log_time >= 10.0:
                      elapsed = now - t0
                      length_elapsed = now - length_t0
                      pct = length_orderings_done / length_orderings * 100 if length_orderings else 0
                      if length_orderings_done > 0:
                        rate = length_elapsed / length_orderings_done
                        remaining = (length_orderings - length_orderings_done) * rate
                        eta_str = f'ETA {remaining:.0f}s'
                      else:
                        eta_str = 'ETA ?'
                      _log(
                        f'[bridge] len={length}: {length_orderings_done}/{length_orderings} '
                        f'({pct:.0f}%) | {stats.total_combinations} combos, '
                        f'{elapsed:.0f}s elapsed, {eta_str}, '
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
      stats.failures.append(f'[bridge] unexpected warnings emitted during bridge run:\n{msgs}')

  elapsed = time.monotonic() - t0
  _log(
    f'[bridge] done: {stats.total_orderings} orderings, '
    f'{stats.total_configs} configs, '
    f'{stats.total_combinations} combos, {elapsed:.1f}s, '
    f'{len(stats.failures)} failures'
  )

  # Single assertion point for all failures (from both sequential and parallel paths)
  if stats.failures:
    test.fail(f'{len(stats.failures)} bridge failure(s):\n' + '\n'.join(stats.failures[:20]))

  return stats


def _apply_with_conc(c: Chain[Any], brick: Brick, fn: Any, conc: int) -> None:
  """Apply a brick with concurrency parameter.

  Rebuilds the operation call to include ``concurrency=conc``.
  Only called for bricks where ``supports_concurrency=True``.
  """
  op = brick.op
  if op == 'foreach':
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
# (ExceptionGroup vs direct) -- a documented asymmetry (SPEC S17).
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

  # Both failed -- compare exception semantics
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
    return True, ''  # Both EG but can't inspect sub-types -- accept

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
