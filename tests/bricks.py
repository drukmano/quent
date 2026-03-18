"""Brick definitions -- self-contained, type-normalizing pipeline operations for bridge testing.

A "brick" is a self-contained, type-normalizing pipeline segment:
  input: number -> does its operation -> output: number
This ensures all permutations are type-compatible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quent import Q
from tests.fixtures import (
  AsyncCM,
  AsyncPair,
  DualProtocolCM,
  Result,
  SyncCM,
  SyncCMAsyncExit,
  SyncCMSuppresses,
)


@dataclass(frozen=True, slots=True)
class Brick:
  """A self-contained, type-normalizing pipeline operation.

  Each brick takes a numeric pipeline value, applies its operation,
  and returns a numeric value.  This ensures any permutation of
  bricks is type-compatible.

  Attributes:
    name: Human-readable name (e.g. 'then', 'map_default').
    op: The operation type (e.g. 'then', 'map', 'gather').
    apply: Callable(q, fn) that appends the operation to the pipeline.
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
    is_terminal: Whether this brick produces a ChainIterator (terminal) rather
        than appending to the pipeline. Terminal bricks are collected via
        iteration rather than chain.run().
  """

  name: str
  op: str
  apply: Callable[..., Any]
  oracle: Callable[..., Any]
  supports_concurrency: bool = False
  has_builtin_concurrency: bool = False
  error_oracle: Callable[[Any, str], Any] | None = None
  calling_convention: str = 'default'
  fn_input: Callable[..., Any] | None = None
  is_terminal: bool = False


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
  # explicit args: fn(42) -- current value NOT passed
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
      op='foreach',
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

  # ---- map (async iterable -- exercises __aiter__ path in foreach) ----
  bricks.append(
    Brick(
      name='map_async_iter',
      op='foreach',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: AsyncPair(x)).foreach(fn).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do (async iterable -- exercises __aiter__ path in foreach_do) ----
  bricks.append(
    Brick(
      name='foreach_do_async_iter',
      op='foreach_do',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: AsyncPair(x)).foreach_do(fn).then(sum),
      oracle=lambda v, fn: v + (v + 1),  # originals preserved, summed
    )
  )

  # ---- map (async iterable + concurrency -- exercises _from_aiter concurrent path) ----
  bricks.append(
    Brick(
      name='map_async_iter_conc',
      op='foreach',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: AsyncPair(x)).foreach(fn, concurrency=2).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do (async iterable + concurrency -- exercises _from_aiter concurrent path) ----
  bricks.append(
    Brick(
      name='foreach_do_async_iter_conc',
      op='foreach_do',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: AsyncPair(x)).foreach_do(fn, concurrency=2).then(sum),
      oracle=lambda v, fn: v + (v + 1),
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

  # ---- gather (nested pipeline as fn -- pipeline is callable) ----
  bricks.append(
    Brick(
      name='gather_nested_chain',
      op='gather',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.gather(Q().then(fn), fn).then(lambda t: t[0]),
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
      oracle=lambda v, fn: fn(v) if v else v,  # truthy -> apply fn
    )
  )

  # ---- then (nested pipeline -- standard rules apply) ----
  # Nested pipeline receives current_value, applies fn
  bricks.append(
    Brick(
      name='then_nested_chain',
      op='then',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(Q().then(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )
  # Nested pipeline with explicit args (explicit args -> Rule 1)
  bricks.append(
    Brick(
      name='then_nested_chain_args',
      op='then',
      calling_convention='nested_chain_args',
      apply=lambda c, fn: c.then(Q().then(fn), 42),
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

  # ---- do (nested pipeline -- standard rules apply) ----
  bricks.append(
    Brick(
      name='do_nested_chain',
      op='do',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.do(Q().then(fn)),
      oracle=lambda v, fn: v,
    )
  )

  # ---- map (nested pipeline as iteration body) ----
  bricks.append(
    Brick(
      name='map_nested_chain',
      op='foreach',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach(Q().then(fn)).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do (nested pipeline as iteration body) ----
  bricks.append(
    Brick(
      name='foreach_do_nested_chain',
      op='foreach_do',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach_do(Q().then(fn)).then(sum),
      oracle=lambda v, fn: v + (v + 1),
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

  # ---- with_ (dual-protocol CM -- exercises _full_async in async perms) ----
  bricks.append(
    Brick(
      name='with_dual_cm',
      op='with_',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_ (async-only CM -- always triggers _full_async path) ----
  bricks.append(
    Brick(
      name='with_async_cm',
      op='with_',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: AsyncCM(x)).with_(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_do (async-only CM -- always triggers _full_async, result discarded) ----
  bricks.append(
    Brick(
      name='with_do_async_cm',
      op='with_do',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: AsyncCM(x)).with_do(fn).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
    )
  )

  # ---- with_ (sync CM, async __exit__ -- triggers _await_exit_success path) ----
  bricks.append(
    Brick(
      name='with_sync_async_exit',
      op='with_',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: SyncCMAsyncExit(x)).with_(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_do (sync CM, async __exit__ -- triggers _await_exit_success path, result discarded) ----
  bricks.append(
    Brick(
      name='with_do_sync_async_exit',
      op='with_do',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: SyncCMAsyncExit(x)).with_do(fn).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
    )
  )

  # ---- with_do (dual-protocol CM -- exercises _full_async in async perms, result discarded) ----
  bricks.append(
    Brick(
      name='with_do_dual_cm',
      op='with_do',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_do(fn).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
    )
  )

  # ---- with_ (nested pipeline body -- standard rules, pipeline runs with ctx value) ----
  bricks.append(
    Brick(
      name='with_nested_chain',
      op='with_',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_(Q().then(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_do (nested pipeline body -- standard rules, pipeline runs, result discarded) ----
  bricks.append(
    Brick(
      name='with_do_nested_chain',
      op='with_do',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_do(Q().then(fn)).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
    )
  )

  # ---- if_ (do branch -- ignore_result path) ----
  bricks.append(
    Brick(
      name='if_do_branch',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: True).do(fn),
      oracle=lambda v, fn: v,
    )
  )

  # ---- if_ (fn as predicate -- async predicate dispatch) ----
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

  # ---- if_ (predicate is nested pipeline -- standard rules apply) ----
  # Nested pipeline returns fn(current_value); fn(10)=11 is truthy
  bricks.append(
    Brick(
      name='if_pred_nested_chain',
      op='if_',
      calling_convention='pred_nested_chain',
      apply=lambda c, fn: c.if_(Q().then(fn)).then(lambda x: x * 2),
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

  # ---- else_ (nested pipeline -- standard rules apply) ----
  bricks.append(
    Brick(
      name='else_nested_chain',
      op='if_',
      calling_convention='else_nested_chain',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_(Q().then(fn)),
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

  # ---- else_do (explicit args, Rule 1) ----
  bricks.append(
    Brick(
      name='else_do_args',
      op='if_',
      calling_convention='else_do_args',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_do(fn, 42),
      oracle=lambda v, fn: v,
      fn_input=lambda v: 42,
    )
  )

  # ---- else_do (nested pipeline body) ----
  bricks.append(
    Brick(
      name='else_do_nested_chain',
      op='if_',
      calling_convention='else_do_nested_chain',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_do(Q().then(fn)),
      oracle=lambda v, fn: v,
    )
  )

  # ---- set/get (context round-trip through sync/async fn) ----
  bricks.append(
    Brick(
      name='set_get',
      op='set_get',
      calling_convention='default',
      apply=lambda c, fn: c.set('_bk').then(fn).then(Q.get, '_bk'),
      oracle=lambda v, fn: v,  # set stores v, fn transforms it, get retrieves original v
    )
  )

  # ---- set/get round-trip (context API survives async transition) ----
  bricks.append(
    Brick(
      name='set_get_roundtrip',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.set('_bridge_k').then(fn).then(Q.get, '_bridge_k'),
      oracle=lambda v, fn: v,
    )
  )

  # ---- set/get descriptor (uses q.get('key') instance method instead of Q.get) ----
  bricks.append(
    Brick(
      name='set_get_descriptor',
      op='set_get',
      calling_convention='default',
      apply=lambda c, fn: c.set('_bk2').then(fn).get('_bk2'),
      oracle=lambda v, fn: v,
    )
  )

  # ---- set tail (set as final operation -- tests set->X transitions at brick boundaries) ----
  bricks.append(
    Brick(
      name='set_tail',
      op='set_get',
      calling_convention='default',
      apply=lambda c, fn: c.then(fn).set('_st'),
      oracle=lambda v, fn: fn(v),  # set preserves CV
    )
  )

  # ---- then with kwargs (Rule 1: kwargs trigger explicit-args path) ----
  bricks.append(
    Brick(
      name='then_kwargs',
      op='then',
      calling_convention='args',
      apply=lambda c, fn: c.then(fn, x=42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- do with kwargs ----
  bricks.append(
    Brick(
      name='do_kwargs',
      op='do',
      calling_convention='args',
      apply=lambda c, fn: c.do(fn, x=42),
      oracle=lambda v, fn: v,
      fn_input=lambda v: 42,
    )
  )

  # ---- then with from_steps (constructor variant) ----
  bricks.append(
    Brick(
      name='then_from_steps',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q.from_steps(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- set with explicit value + get (two-arg set form) ----
  bricks.append(
    Brick(
      name='set_explicit_value',
      op='set_get',
      calling_convention='default',
      apply=lambda c, fn: c.set('_bk3', 99).then(fn).then(Q.get, '_bk3'),
      oracle=lambda v, fn: 99,
    )
  )

  # ---- get with default (key does not exist -- returns default) ----
  bricks.append(
    Brick(
      name='get_with_default',
      op='set_get',
      calling_convention='default',
      apply=lambda c, fn: c.then(fn).get('_nonexistent_bridge', 42),
      oracle=lambda v, fn: 42,
    )
  )

  # ---- nested pipeline with except_ (happy path -- handler not invoked) ----
  bricks.append(
    Brick(
      name='except_nested',
      op='then',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=-1) if err == 'exception' else None,
      apply=lambda c, fn: c.then(Q().then(fn).except_(lambda ei: -1)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- nested pipeline with finally_ (handler fires, return discarded) ----
  bricks.append(
    Brick(
      name='finally_nested',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q().then(fn).finally_(lambda rv: None)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- gather with single function (returns 1-element tuple) ----
  bricks.append(
    Brick(
      name='gather_single',
      op='gather',
      calling_convention='default',
      apply=lambda c, fn: c.gather(fn).then(lambda t: t[0]),
      oracle=lambda v, fn: fn(v),
      supports_concurrency=True,
    )
  )

  # ---- gather with bounded concurrency ----
  bricks.append(
    Brick(
      name='gather_bounded_conc',
      op='gather',
      calling_convention='default',
      apply=lambda c, fn: c.gather(fn, fn, concurrency=2).then(lambda t: t[0]),
      oracle=lambda v, fn: fn(v),
      supports_concurrency=True,
    )
  )

  # ---- foreach with concurrency on sync iterable (ThreadPoolExecutor path) ----
  bricks.append(
    Brick(
      name='map_sync_conc',
      op='foreach',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach(fn, concurrency=2).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do with concurrency on sync iterable ----
  bricks.append(
    Brick(
      name='foreach_do_sync_conc',
      op='foreach_do',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach_do(fn, concurrency=2).then(sum),
      oracle=lambda v, fn: v + (v + 1),
    )
  )

  # ---- bare set (no fn -- tests set as isolated step at brick boundaries) ----
  bricks.append(
    Brick(
      name='set_bare',
      op='set_get',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=v),
      apply=lambda c, fn: c.set('_bare'),
      oracle=lambda v, fn: v,
    )
  )

  # ---- bare set with explicit value (no fn -- two-arg form as isolated step) ----
  bricks.append(
    Brick(
      name='set_explicit_bare',
      op='set_get',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=v),
      apply=lambda c, fn: c.set('_bare_e', 99),
      oracle=lambda v, fn: v,
    )
  )

  # ---- if_ (truthy branch with explicit args, Rule 1) ----
  bricks.append(
    Brick(
      name='if_true_args',
      op='if_',
      calling_convention='args',
      apply=lambda c, fn: c.if_(lambda x: True).then(fn, 42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- if_ (truthy branch with nested pipeline body) ----
  bricks.append(
    Brick(
      name='if_true_nested_chain',
      op='if_',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.if_(lambda x: True).then(Q().then(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- if_ (falsy path, no else -- value passes through unchanged) ----
  bricks.append(
    Brick(
      name='if_false_passthrough',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).then(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- if_ (do branch with explicit args, Rule 1) ----
  bricks.append(
    Brick(
      name='if_do_args',
      op='if_',
      calling_convention='args',
      apply=lambda c, fn: c.if_(lambda x: True).do(fn, 42),
      oracle=lambda v, fn: v,
      fn_input=lambda v: 42,
    )
  )

  # ---- if_ (predicate with kwargs, Rule 1) ----
  bricks.append(
    Brick(
      name='if_pred_kwargs',
      op='if_',
      calling_convention='pred_kwargs',
      apply=lambda c, fn: c.if_(fn, x=42).then(lambda x: x * 2),
      oracle=lambda v, fn: v * 2,
      fn_input=lambda v: 42,
    )
  )

  # ---- if_ (falsy predicate + do branch, no else -- passthrough) ----
  bricks.append(
    Brick(
      name='if_false_do_passthrough',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: False).do(fn).then(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- if_ (falsy with do-style truthy branch + else_) ----
  bricks.append(
    Brick(
      name='if_false_do_else',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: False).do(lambda x: None).else_(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- if_ (both branches do-style, predicate false) ----
  bricks.append(
    Brick(
      name='if_false_do_else_do',
      op='if_',
      calling_convention='default',
      apply=lambda c, fn: c.if_(lambda x: False).do(lambda x: None).else_do(fn),
      oracle=lambda v, fn: v,
    )
  )

  # ---- if_ (do branch with nested pipeline body) ----
  bricks.append(
    Brick(
      name='if_do_nested_chain',
      op='if_',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.if_(lambda x: True).do(Q().then(fn)),
      oracle=lambda v, fn: v,
    )
  )

  # ---- with_ (kwargs, Rule 1) ----
  bricks.append(
    Brick(
      name='with_kwargs',
      op='with_',
      calling_convention='args',
      apply=lambda c, fn: c.then(lambda x: SyncCM(x)).with_(fn, x=42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- with_do (kwargs, Rule 1) ----
  bricks.append(
    Brick(
      name='with_do_kwargs',
      op='with_do',
      calling_convention='args',
      apply=lambda c, fn: c.then(lambda x: SyncCM(x)).with_do(fn, x=42).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
      fn_input=lambda v: 42,
    )
  )

  # ---- with_ (nested pipeline body with explicit args) ----
  bricks.append(
    Brick(
      name='with_nested_chain_args',
      op='with_',
      calling_convention='nested_chain_args',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_(Q().then(fn), 42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- with_do (nested pipeline body with explicit args) ----
  bricks.append(
    Brick(
      name='with_do_nested_chain_args',
      op='with_do',
      calling_convention='nested_chain_args',
      apply=lambda c, fn: c.then(lambda x: DualProtocolCM(x)).with_do(Q().then(fn), 42).then(lambda cm: cm._value),
      oracle=lambda v, fn: v,
      fn_input=lambda v: 42,
    )
  )

  # ---- else_ (nested pipeline with explicit args) ----
  bricks.append(
    Brick(
      name='else_nested_chain_args',
      op='if_',
      calling_convention='else_nested_chain_args',
      apply=lambda c, fn: c.if_(lambda x: False).then(lambda x: -999).else_(Q().then(fn), 42),
      oracle=lambda v, fn: fn(42),
      fn_input=lambda v: 42,
    )
  )

  # ---- foreach (sync iterable, unbounded concurrency) ----
  bricks.append(
    Brick(
      name='map_sync_unbounded_conc',
      op='foreach',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach(fn, concurrency=-1).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do (sync iterable, unbounded concurrency) ----
  bricks.append(
    Brick(
      name='foreach_do_sync_unbounded_conc',
      op='foreach_do',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach_do(fn, concurrency=-1).then(sum),
      oracle=lambda v, fn: v + (v + 1),
    )
  )

  # ---- foreach (async iterable, unbounded concurrency) ----
  bricks.append(
    Brick(
      name='map_async_iter_unbounded_conc',
      op='foreach',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: AsyncPair(x)).foreach(fn, concurrency=-1).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do (async iterable, unbounded concurrency) ----
  bricks.append(
    Brick(
      name='foreach_do_async_iter_unbounded_conc',
      op='foreach_do',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: AsyncPair(x)).foreach_do(fn, concurrency=-1).then(sum),
      oracle=lambda v, fn: v + (v + 1),
    )
  )

  # ---- foreach (nested pipeline body + concurrency) ----
  bricks.append(
    Brick(
      name='map_nested_chain_conc',
      op='foreach',
      calling_convention='nested_chain',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach(Q().then(fn), concurrency=2).then(sum),
      oracle=lambda v, fn: fn(v) + fn(v + 1),
    )
  )

  # ---- foreach_do (nested pipeline body + concurrency) ----
  bricks.append(
    Brick(
      name='foreach_do_nested_chain_conc',
      op='foreach_do',
      calling_convention='nested_chain',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).foreach_do(Q().then(fn), concurrency=2).then(sum),
      oracle=lambda v, fn: v + (v + 1),
    )
  )

  # ---- gather (all nested chains) ----
  bricks.append(
    Brick(
      name='gather_multi_nested_chain',
      op='gather',
      calling_convention='nested_chain',
      supports_concurrency=True,
      apply=lambda c, fn: c.gather(Q().then(fn), Q().then(fn)).then(lambda t: t[0]),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- gather (3 functions, unbounded concurrency) ----
  bricks.append(
    Brick(
      name='gather_unbounded_3fns',
      op='gather',
      calling_convention='default',
      has_builtin_concurrency=True,
      apply=lambda c, fn: c.gather(fn, fn, fn).then(lambda t: t[0]),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- set explicit value + descriptor get ----
  bricks.append(
    Brick(
      name='set_explicit_value_descriptor',
      op='set_get',
      calling_convention='default',
      apply=lambda c, fn: c.set('_bk4', 99).then(fn).get('_bk4'),
      oracle=lambda v, fn: 99,
    )
  )

  # ---- nested pipeline with except_ + explicit args on handler ----
  bricks.append(
    Brick(
      name='except_args_nested',
      op='then',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=-1) if err == 'exception' else None,
      apply=lambda c, fn: c.then(Q().then(fn).except_(lambda ei: -1, 'unused_arg')),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- nested pipeline with except_ + reraise=True (happy path) ----
  bricks.append(
    Brick(
      name='except_reraise_nested',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q().then(fn).except_(lambda ei: None, reraise=True)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- nested pipeline with except_ where handler is a nested pipeline ----
  bricks.append(
    Brick(
      name='except_nested_chain_handler',
      op='then',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=-1) if err == 'exception' else None,
      apply=lambda c, fn: c.then(Q().then(fn).except_(Q().then(lambda ei: -1))),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- nested pipeline with finally_ + explicit args on handler ----
  bricks.append(
    Brick(
      name='finally_args_nested',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q().then(fn).finally_(lambda rv: None, 'unused_arg')),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- nested pipeline with finally_ where handler is a nested pipeline ----
  bricks.append(
    Brick(
      name='finally_nested_chain_handler',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q().then(fn).finally_(Q().then(lambda rv: None))),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- nested pipeline with both except_ and finally_ (happy path) ----
  bricks.append(
    Brick(
      name='except_finally_nested',
      op='then',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=-1) if err == 'exception' else None,
      apply=lambda c, fn: c.then(Q().then(fn).except_(lambda ei: -1).finally_(lambda rv: None)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- clone() used as nested pipeline ----
  bricks.append(
    Brick(
      name='then_clone_chain',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q().then(fn).clone()),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- from_steps with multiple steps ----
  bricks.append(
    Brick(
      name='then_from_steps_multi',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q.from_steps(fn, lambda x: x - 1, fn)),
      oracle=lambda v, fn: fn(fn(v) - 1),
    )
  )

  # ---- named pipeline (name() is cosmetic) ----
  bricks.append(
    Brick(
      name='then_named_chain',
      op='then',
      calling_convention='default',
      apply=lambda c, fn: c.then(Q().name('test').then(fn)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- as_decorator()-wrapped function as pipeline step ----
  bricks.append(
    Brick(
      name='then_decorator_chain',
      op='then',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=v * 100) if err == 'return_signal' else None,
      apply=lambda c, fn: c.then(Q().then(fn).as_decorator()(lambda x: x)),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- with_ using suppressing CM (happy path -- no error, normal execution) ----
  bricks.append(
    Brick(
      name='with_suppress',
      op='with_',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=None) if err in ('exception', 'base_exception') else None,
      fn_input=lambda v: 10,
      apply=lambda c, fn: c.then(lambda x: SyncCMSuppresses()).with_(fn),
      oracle=lambda v, fn: fn(10),  # SyncCMSuppresses.__enter__ returns 10
    )
  )

  # ---- with_do using suppressing CM ----
  bricks.append(
    Brick(
      name='with_do_suppress',
      op='with_do',
      calling_convention='default',
      error_oracle=lambda v, err: Result(success=True, value=10) if err in ('exception', 'base_exception') else None,
      fn_input=lambda v: 10,
      apply=lambda c, fn: c.then(lambda x: SyncCMSuppresses()).with_do(fn).then(lambda _: 10),
      oracle=lambda v, fn: 10,
    )
  )

  # ===========================================================================
  # while_ bricks
  # ===========================================================================

  # ---- while_ then (truthiness predicate, fn is body) ----
  # Start at -2, loop fn (x+1) until 0 (falsy).
  # -2 -> fn(-2)=-1 -> fn(-1)=0. Loop result: 0. Then fn(0) = 1.
  bricks.append(
    Brick(
      name='while_then_default',
      op='while_',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: -2).while_().then(fn).then(fn),
      oracle=lambda v, fn: fn(0),  # loop converges to 0, then fn(0)
    )
  )

  # ---- while_ then (callable predicate, fn is body) ----
  # Start at -2, loop while < 0, body is fn (x+1). Converges to 0.
  # Then fn(0) runs as the post-loop step.
  bricks.append(
    Brick(
      name='while_then_pred',
      op='while_',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda x: -2).while_(lambda x: x < 0).then(fn).then(fn),
      oracle=lambda v, fn: fn(0),
    )
  )

  # ---- while_ do (fn as side-effect body, value preserved) ----
  # Use a single-iteration loop: predicate checks a counter embedded in
  # a mutable wrapper. The do() body runs fn as a side effect, then
  # decrements the counter to terminate the loop.
  # Design: value is a [number, counter] list. Predicate checks counter > 0.
  # do() body calls fn on number (side effect), then decrements counter.
  # After loop, extract number.  But this is too complex -- use a simpler
  # approach: pipeline with while_(pred_that_runs_once).do(fn).
  # Simplest: use a non-callable truthy literal predicate (True) with
  # a body that issues break_() after calling fn. But do() discards
  # fn's return including break. So use a wrapper that calls fn then breaks.
  # Actually simpler: use value truthiness with then(0) to stop after one
  # iteration, but do() doesn't change the value.
  #
  # Cleanest approach: use a factory that creates a countdown predicate,
  # embedded in the apply lambda.  The predicate runs exactly once.
  bricks.append(
    Brick(
      name='while_do_default',
      op='while_',
      calling_convention='default',
      apply=lambda c, fn: c.while_(lambda x, _s={'n': 1}: (_s.__setitem__('n', _s['n'] - 1), _s['n'] >= 0)[1]).do(fn),
      oracle=lambda v, fn: v,  # do() preserves pipeline value
    )
  )

  # ---- while_ then with break (fn applied, break exits loop) ----
  # Loop from v with True predicate. Body: if x >= v+1, break_(fn(x)),
  # else fn(x). So fn runs twice: fn(v), then break_(fn(fn(v))).
  # Result: fn(fn(v)) = fn(v+1) = v+2.
  bricks.append(
    Brick(
      name='while_then_break',
      op='while_',
      calling_convention='default',
      apply=lambda c, fn: c.while_(True).then(lambda x, _fn=fn: Q.break_(_fn(x)) if x > 10 else _fn(x)),
      oracle=lambda v, fn: fn(fn(v)),  # fn(10)=11, 11>10 so break_(fn(11))=12
    )
  )

  # ---- while_ then with nested pipeline body ----
  # Body is a nested pipeline that applies fn. Tests pipeline_callable dispatch
  # in while_ body context. Start at -2 for fast convergence.
  bricks.append(
    Brick(
      name='while_then_nested_chain',
      op='while_',
      calling_convention='nested_chain',
      apply=lambda c, fn: c.then(lambda x: -2).while_().then(Q().then(fn)).then(fn),
      oracle=lambda v, fn: fn(0),
    )
  )

  # ---- while_ then with explicit args on body (Rule 1) ----
  # Body has explicit args: fn(42). Loop runs once (predicate runs exactly
  # once via mutable counter). Result: fn(42) = 43.
  # Then fn(43) = 44 as post-loop step.
  bricks.append(
    Brick(
      name='while_then_args',
      op='while_',
      calling_convention='args',
      apply=lambda c, fn: (
        c.while_(lambda x, _s={'n': 1}: (_s.__setitem__('n', _s['n'] - 1), _s['n'] >= 0)[1]).then(fn, 42).then(fn)
      ),
      oracle=lambda v, fn: fn(fn(42)),  # body: fn(42)=43, post-loop: fn(43)=44
      fn_input=lambda v: 42,
    )
  )

  # ---- while_ then with predicate args (Rule 1 on predicate) ----
  # Predicate with explicit args: pred(limit) where limit=0, so pred
  # returns False immediately. Body never runs. Value passes through.
  bricks.append(
    Brick(
      name='while_pred_args',
      op='while_',
      calling_convention='default',
      apply=lambda c, fn: c.while_(lambda limit: limit > 100, 0).then(fn).then(fn),
      oracle=lambda v, fn: fn(v),  # pred(0) is False, body never runs, then fn(v) runs
    )
  )

  # ---- while_ then with non-callable literal predicate (falsy: 0) ----
  # Literal predicate 0 is falsy, loop never runs. Value passes through.
  bricks.append(
    Brick(
      name='while_pred_literal_falsy',
      op='while_',
      calling_convention='default',
      apply=lambda c, fn: c.while_(0).then(lambda x: x * 100).then(fn),
      oracle=lambda v, fn: fn(v),  # loop never runs, fn(v) as post-step
    )
  )

  # ===========================================================================
  # drive_gen bricks
  # ===========================================================================

  # ---- drive_gen (single-yield sync gen, fn processes yielded value) ----
  # Generator yields the pipeline value once. fn processes it. Result: fn(v).
  bricks.append(
    Brick(
      name='drive_gen_single',
      op='drive_gen',
      calling_convention='default',
      apply=lambda c, fn: c.then(_gen_passthrough).drive_gen(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- drive_gen (multi-yield sync gen, fn drives 3 yields) ----
  # Fixed sync generator: yield 1, yield sent+1, yield sent+1.
  # With fn(x)=x+1: yield 1->fn(1)=2->send 2->yield 3->fn(3)=4
  #   ->send 4->yield 5->fn(5)=6->stop. Result: 6.
  bricks.append(
    Brick(
      name='drive_gen_multi',
      op='drive_gen',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda _: _gen_multi_sync()).drive_gen(fn),
      oracle=lambda v, fn: fn(fn(fn(1) + 1) + 1),
    )
  )

  # ---- drive_gen (async gen, single yield) ----
  # Async generator yields pipeline value once. Forces full-async path.
  bricks.append(
    Brick(
      name='drive_gen_async_gen',
      op='drive_gen',
      calling_convention='default',
      apply=lambda c, fn: c.then(_async_gen_passthrough).drive_gen(fn),
      oracle=lambda v, fn: fn(v),
    )
  )

  # ---- drive_gen (callable factory, fn drives multi-yield gen) ----
  # Pipeline value is a callable (gen factory), resolved by drive_gen.
  # Tests callable resolution path.
  bricks.append(
    Brick(
      name='drive_gen_factory',
      op='drive_gen',
      calling_convention='default',
      apply=lambda c, fn: c.then(lambda _: _gen_multi_sync).drive_gen(fn),
      oracle=lambda v, fn: fn(fn(fn(1) + 1) + 1),
    )
  )

  # ---- drive_gen (empty sync gen, fn never called) ----
  # Empty generator: fn is never called. Result: None.
  # Post-step fn normalizes None -> fn(None). But fn(None) would be
  # None+1 which raises TypeError! So we use then(lambda x: 0 if x is None else x)
  # to normalize, then fn.
  bricks.append(
    Brick(
      name='drive_gen_empty',
      op='drive_gen',
      calling_convention='default',
      apply=lambda c, fn: (
        c.then(lambda _: _gen_empty_sync()).drive_gen(fn).then(lambda x: 0 if x is None else x).then(fn)
      ),
      oracle=lambda v, fn: fn(0),  # empty gen -> None -> 0 -> fn(0)
    )
  )

  # ---- drive_gen + then (compose with downstream step) ----
  # drive_gen result feeds into then(fn). Tests pipeline threading.
  bricks.append(
    Brick(
      name='drive_gen_then',
      op='drive_gen',
      calling_convention='default',
      apply=lambda c, fn: c.then(_gen_passthrough).drive_gen(fn).then(fn),
      oracle=lambda v, fn: fn(fn(v)),  # drive_gen: fn(v), then: fn(fn(v))
    )
  )

  # ---- iterate (terminal) ----
  # Returns ChainIterator — collected via for/async for, not chain.run()
  bricks.append(
    Brick(
      name='iterate_default',
      op='iterate',
      calling_convention='default',
      is_terminal=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).iterate(fn),
      oracle=lambda v, fn: [fn(v), fn(v + 1)],
    )
  )

  # ---- iterate_do (terminal) ----
  bricks.append(
    Brick(
      name='iterate_do_default',
      op='iterate_do',
      calling_convention='default',
      is_terminal=True,
      apply=lambda c, fn: c.then(lambda x: [x, x + 1]).iterate_do(fn),
      oracle=lambda v, fn: [v, v + 1],
    )
  )

  # ---- flat_iterate (terminal) ----
  bricks.append(
    Brick(
      name='flat_iterate_default',
      op='flat_iterate',
      calling_convention='default',
      is_terminal=True,
      apply=lambda c, fn: c.then(lambda x: [[x, x + 1]]).flat_iterate(),
      oracle=lambda v, fn: [v, v + 1],
    )
  )

  # ---- flat_iterate_do (terminal) ----
  bricks.append(
    Brick(
      name='flat_iterate_do_default',
      op='flat_iterate_do',
      calling_convention='default',
      is_terminal=True,
      apply=lambda c, fn: c.then(lambda x: [[x, x + 1]]).flat_iterate_do(),
      oracle=lambda v, fn: [v, v + 1],
    )
  )

  return bricks


# ---------------------------------------------------------------------------
# Generator helpers for drive_gen bricks
# ---------------------------------------------------------------------------


def _gen_passthrough(x):
  """Sync generator that yields x once."""
  yield x


async def _async_gen_passthrough(x):
  """Async generator that yields x once."""
  yield x


def _gen_multi_sync():
  """Sync generator: yields 1, then (sent+1), then (sent+1). Three yields."""
  x = yield 1
  x = yield x + 1
  x = yield x + 1


def _gen_empty_sync():
  """Sync generator that yields nothing."""
  return
  yield  # make it a generator


ALL_BRICKS = _make_bricks()

# Filtered views: standard bricks (for run_bridge) vs terminal bricks (for run_terminal_bridge)
STANDARD_BRICKS = [b for b in ALL_BRICKS if not b.is_terminal]
TERMINAL_BRICKS = [b for b in ALL_BRICKS if b.is_terminal]

# Index bricks by operation type for targeted testing
BRICKS_BY_OP: dict[str, list[Brick]] = {}
for _b in ALL_BRICKS:
  BRICKS_BY_OP.setdefault(_b.op, []).append(_b)


# ---------------------------------------------------------------------------
# Representative bricks (working subset -- no ellipsis, no unpack)
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
    'map_async_iter',
    'foreach_do_default',
    'foreach_do_async_iter',
    'gather_default',
    'with_default',
    'with_dual_cm',
    'with_async_cm',
    'with_sync_async_exit',
    'with_do_default',
    'with_do_dual_cm',
    'with_do_async_cm',
    'if_true_default',
    'if_do_branch',
    'if_pred_fn',
    'if_pred_nested_chain',
    'else_nested_chain',
    'else_do',
    'set_get',
    'set_get_roundtrip',
    'set_get_descriptor',
    # New representative bricks covering distinct code paths
    'if_false_passthrough',  # falsy path with no else (passthrough)
    'if_pred_kwargs',  # predicate with kwargs (Rule 1 on pred)
    'if_false_do_else',  # do-style truthy branch + else_ combination
    'with_kwargs',  # with_ kwargs path (Rule 1)
    'with_nested_chain_args',  # with_ nested pipeline + explicit args
    'else_nested_chain_args',  # else_ nested pipeline + explicit args
    'map_sync_unbounded_conc',  # unbounded concurrency (concurrency=-1)
    'gather_multi_nested_chain',  # gather with all nested pipelines
    'except_args_nested',  # except handler with explicit args
    'except_nested_chain_handler',  # except handler is a pipeline
    'finally_nested_chain_handler',  # finally handler is a pipeline
    'then_clone_chain',  # clone() reuse path
    'then_decorator_chain',  # as_decorator() reuse path
    'set_explicit_value_descriptor',  # two-arg set + descriptor get combo
  )
]
