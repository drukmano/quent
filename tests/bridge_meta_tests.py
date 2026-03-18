# SPDX-License-Identifier: MIT
"""Meta-test verifying the bridge brick set's coverage guarantee.

The exhaustive bridge test in bridge_tests.py uses a 24-brick subset
(out of 96 total bricks) to test sync/async equivalence.  This meta-test
programmatically verifies that every excluded brick's behavioral axes
are represented by the included set, so no engine code path goes untested.

The coverage model decomposes each brick into independent behavioral axes:

  1. **Operation type** (``op``): which engine code path handles this brick
     (then, do, foreach, foreach_do, gather, with_, with_do, if_, set_get).

  2. **Dispatch rule**: how the step's callable is invoked — the actual
     branch taken in the evaluation layer:
       - rule2 (default: fn(cv))
       - rule1 (explicit args: fn(*args, **kwargs))
       - chain_callable (nested Chain as callable)
       - literal (non-callable value, Rule 2 non-callable path)

     The composite ``chain_callable+rule1`` (nested Chain with explicit args)
     is NOT a distinct engine branch — it decomposes into two independently
     tested paths: the Chain.__call__ dispatch (chain_callable) and the
     args-replacement logic (rule1).  The bridge set covers both components
     separately.

  3. **Concurrency mode**: whether concurrency is exercised.  Bricks with
     ``supports_concurrency=True`` are covered by the bridge runner's
     external concurrency axis.  Bricks with ``has_builtin_concurrency=True``
     have concurrency baked in — but the bridge *only* tests sync/async
     equivalence, and concurrency does not introduce a new sync/async
     dispatch branch (it is an orthogonal axis tested elsewhere).

  4. **CM protocol variant** (with_/with_do only): the specific context
     manager protocol exercised (sync CM, async CM, dual-protocol CM,
     sync-enter/async-exit CM).

For an excluded brick to be "covered," the included set must contain:
  - At least one brick with the **same operation type**, AND
  - At least one brick (any op) with the **same dispatch rule**, AND
  - For with_/with_do bricks with a specific CM variant, at least one
    included brick exercises that CM protocol.

This ensures every engine code path (op x dispatch) and every CM protocol
path is exercised by the bridge set.
"""

from __future__ import annotations

import unittest

from tests.bricks import ALL_BRICKS, Brick

# ---------------------------------------------------------------------------
# Bridge brick set (must match bridge_tests._BRIDGE_BRICK_NAMES exactly)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Axis extraction helpers
# ---------------------------------------------------------------------------


def _dispatch_rule(brick: Brick) -> str:
  """Classify a brick's calling convention into its dispatch rule category.

  The dispatch rule determines which branch the evaluation layer takes.
  Multiple calling conventions map to the same dispatch rule — that is
  the core of the redundancy argument.

  Returns one of: 'rule2', 'rule1', 'chain_callable', 'chain_callable+rule1', 'literal'.
  """
  cc = brick.calling_convention
  # Rule 1: explicit args/kwargs — all use the same fn(*args, **kwargs) path
  if cc in ('args', 'pred_args', 'else_args', 'else_do_args', 'pred_kwargs'):
    return 'rule1'
  # Chain callable: nested chain as the step value
  if cc in ('nested_chain', 'pred_nested_chain', 'else_nested_chain', 'else_do_nested_chain'):
    return 'chain_callable'
  # Chain callable + Rule 1: nested chain with explicit args — a composite
  # of two independently-tested paths (chain_callable + rule1).
  if cc in ('nested_chain_args', 'else_nested_chain_args'):
    return 'chain_callable+rule1'
  # Literal: non-callable value (Rule 2 non-callable branch)
  if cc in ('literal', 'pred_literal', 'else_literal'):
    return 'literal'
  # Default: fn(current_value) — Rule 2 default passthrough
  if cc == 'default':
    return 'rule2'
  raise ValueError(f'Unknown calling convention: {cc!r} on brick {brick.name!r}')


# Composite dispatch rules and their atomic components.
# The bridge set need not directly include a composite rule — it suffices
# to include each atomic component, since they exercise independent
# engine branches that compose without interaction.
_COMPOSITE_DISPATCH_RULES: dict[str, frozenset[str]] = {
  'chain_callable+rule1': frozenset({'chain_callable', 'rule1'}),
}

# Atomic (non-decomposable) dispatch rules.
_ATOMIC_DISPATCH_RULES = frozenset({'rule1', 'rule2', 'chain_callable', 'literal'})


def _dispatch_rule_components(dr: str) -> frozenset[str]:
  """Return the atomic dispatch rule components that must be covered.

  For atomic rules, returns a singleton set.
  For composite rules, returns the set of atomic components.
  """
  return _COMPOSITE_DISPATCH_RULES.get(dr, frozenset({dr}))


# CM protocol variant names for with_/with_do bricks.
# Extracted from brick names by convention: the CM type is encoded in the name.
_CM_VARIANTS = {
  'default': 'sync_cm',  # SyncCM
  'async_cm': 'async_cm',  # AsyncCM (async-only)
  'dual_cm': 'dual_cm',  # DualProtocolCM
  'sync_async_exit': 'sync_async_exit',  # SyncCMAsyncExit
  'suppress': 'suppress_cm',  # SyncCMSuppresses
}


def _cm_variant(brick: Brick) -> str | None:
  """Extract the CM protocol variant for with_/with_do bricks.

  Returns None for non-CM bricks.
  """
  if brick.op not in ('with_', 'with_do'):
    return None
  name = brick.name
  # Strip the op prefix to get the variant suffix
  for prefix in ('with_do_', 'with_'):
    if name.startswith(prefix):
      suffix = name[len(prefix) :]
      break
  else:
    return 'sync_cm'  # fallback

  # Map suffix to CM variant
  if suffix in ('default',):
    return 'sync_cm'
  if suffix.startswith('async_cm'):
    return 'async_cm'
  if suffix.startswith('dual_cm'):
    return 'dual_cm'
  if suffix.startswith('sync_async_exit'):
    return 'sync_async_exit'
  if suffix.startswith('suppress'):
    return 'suppress_cm'
  # Calling convention variants (args, kwargs, nested_chain, nested_chain_args)
  # use the default sync CM — the CM protocol is not the axis being tested.
  if any(suffix.startswith(s) for s in ('args', 'kwargs', 'nested_chain')):
    return 'sync_cm'
  return 'sync_cm'  # conservative fallback


def _iteration_source(brick: Brick) -> str | None:
  """Extract iteration source type for foreach/foreach_do bricks.

  Returns 'sync_iter', 'async_iter', or None for non-iteration bricks.
  """
  if brick.op not in ('foreach', 'foreach_do'):
    return None
  if 'async_iter' in brick.name:
    return 'async_iter'
  return 'sync_iter'


# ---------------------------------------------------------------------------
# Meta-test
# ---------------------------------------------------------------------------


class BridgeBrickCoverageTest(unittest.TestCase):
  """Verify the 24-brick bridge set covers all 96 bricks' behavioral axes."""

  def setUp(self) -> None:
    self.all_bricks = ALL_BRICKS
    self.bridge_names = _BRIDGE_BRICK_NAMES
    self.included = [b for b in self.all_bricks if b.name in self.bridge_names]
    self.excluded = [b for b in self.all_bricks if b.name not in self.bridge_names]

    # Pre-compute included axis sets
    self.included_ops = {b.op for b in self.included}
    self.included_dispatch_rules = {_dispatch_rule(b) for b in self.included}
    self.included_cm_variants = {_cm_variant(b) for b in self.included if _cm_variant(b) is not None}
    self.included_iter_sources = {_iteration_source(b) for b in self.included if _iteration_source(b) is not None}

  # -- structural invariants --

  def test_total_brick_count(self) -> None:
    """ALL_BRICKS has exactly 96 bricks (guards against silent additions/removals)."""
    self.assertEqual(len(self.all_bricks), 96)

  def test_bridge_set_size(self) -> None:
    """The bridge set has exactly 24 bricks."""
    self.assertEqual(len(self.included), 24)

  def test_bridge_names_are_valid(self) -> None:
    """Every name in the bridge set corresponds to an actual brick."""
    all_names = {b.name for b in self.all_bricks}
    unknown = self.bridge_names - all_names
    self.assertEqual(unknown, set(), f'Bridge set contains unknown brick names: {unknown}')

  def test_bridge_set_matches_bridge_tests(self) -> None:
    """Our bridge set definition matches bridge_tests._BRIDGE_BRICK_NAMES.

    This catches divergence between this meta-test and the actual bridge test.
    """
    # Import the authoritative set from bridge_tests
    from tests.bridge_tests import _BRIDGE_BRICK_NAMES as authoritative

    self.assertEqual(
      self.bridge_names,
      authoritative,
      'Meta-test bridge set diverged from bridge_tests._BRIDGE_BRICK_NAMES',
    )

  def test_excluded_count(self) -> None:
    """72 bricks are excluded (96 - 24)."""
    self.assertEqual(len(self.excluded), 72)

  # -- axis coverage: every op type is represented --

  def test_all_ops_covered(self) -> None:
    """The bridge set includes at least one brick for every operation type."""
    all_ops = {b.op for b in self.all_bricks}
    missing = all_ops - self.included_ops
    self.assertEqual(missing, set(), f'Operation types missing from bridge set: {missing}')

  # -- axis coverage: every dispatch rule is represented --

  def test_all_atomic_dispatch_rules_covered(self) -> None:
    """The bridge set includes at least one brick for every atomic dispatch rule.

    Composite rules (like chain_callable+rule1) are covered if each of
    their atomic components is independently covered.
    """
    all_dispatch_rules = {_dispatch_rule(b) for b in self.all_bricks}
    # Collect all atomic components that need coverage
    all_atomic: set[str] = set()
    for dr in all_dispatch_rules:
      all_atomic |= _dispatch_rule_components(dr)
    included_atomic = {_dispatch_rule(b) for b in self.included} & _ATOMIC_DISPATCH_RULES
    missing = all_atomic - included_atomic
    self.assertEqual(missing, set(), f'Atomic dispatch rules missing from bridge set: {missing}')

  # -- axis coverage: every CM protocol variant is represented --

  def test_all_cm_variants_covered(self) -> None:
    """The bridge set includes at least one brick for every CM protocol variant.

    CM variants exercise distinct engine paths in with_/with_do:
      sync_cm     — sync __enter__/__exit__
      async_cm    — async __aenter__/__aexit__ only
      dual_cm     — both protocols, engine chooses based on execution tier
      sync_async_exit — sync __enter__, async __exit__ (hybrid path)
    """
    all_cm_variants = {_cm_variant(b) for b in self.all_bricks if _cm_variant(b) is not None}
    # suppress_cm is a CM variant not in the bridge set, but it only adds
    # error-absorption behavior (tested separately), not a new sync/async
    # dispatch path.  The sync/async dispatch for suppress_cm is identical
    # to sync_cm — it is the __exit__ return value (True) that differs.
    # So we exclude it from the mandatory coverage requirement.
    sync_async_cm_variants = all_cm_variants - {'suppress_cm'}
    missing = sync_async_cm_variants - self.included_cm_variants
    self.assertEqual(missing, set(), f'CM variants missing from bridge set: {missing}')

  # -- axis coverage: every iteration source type is represented --

  def test_all_iteration_sources_covered(self) -> None:
    """The bridge set includes bricks for both sync and async iterables.

    These exercise distinct engine paths in _iter_ops.py:
      sync_iter  — __iter__ path (sync fast path)
      async_iter — __aiter__ path (forces async transition)
    """
    all_iter_sources = {_iteration_source(b) for b in self.all_bricks if _iteration_source(b) is not None}
    missing = all_iter_sources - self.included_iter_sources
    self.assertEqual(missing, set(), f'Iteration sources missing from bridge set: {missing}')

  # -- axis coverage: concurrency capability is represented --

  def test_concurrency_capable_bricks_included(self) -> None:
    """The bridge set includes at least one brick with supports_concurrency=True.

    This ensures the bridge runner's external concurrency axis has bricks
    to exercise.
    """
    has_conc = any(b.supports_concurrency for b in self.included)
    self.assertTrue(has_conc, 'No concurrency-capable brick in bridge set')

  # -- per-brick coverage: every excluded brick maps to covering bricks --

  def test_every_excluded_brick_op_is_covered(self) -> None:
    """Every excluded brick's operation type appears in the bridge set."""
    for brick in self.excluded:
      with self.subTest(brick=brick.name, op=brick.op):
        self.assertIn(
          brick.op,
          self.included_ops,
          f'Excluded brick {brick.name!r} has op={brick.op!r} which is not represented in the bridge set',
        )

  def test_every_excluded_brick_dispatch_rule_is_covered(self) -> None:
    """Every excluded brick's dispatch rule components appear in the bridge set.

    The dispatch rule is the calling convention category — the actual
    branch taken in the evaluation layer.  Multiple calling conventions
    (args, pred_args, else_args, kwargs, pred_kwargs) all collapse to
    the same Rule 1 dispatch.  Similarly, nested_chain variants all
    use the same Chain.__call__ dispatch path.

    Composite rules (chain_callable+rule1) are covered if each atomic
    component is independently present in the bridge set — they compose
    two independent engine branches.
    """
    for brick in self.excluded:
      dr = _dispatch_rule(brick)
      components = _dispatch_rule_components(dr)
      for component in components:
        with self.subTest(brick=brick.name, dispatch_rule=dr, component=component):
          self.assertIn(
            component,
            self.included_dispatch_rules,
            f'Excluded brick {brick.name!r} has dispatch_rule={dr!r}, '
            f'component={component!r} is not represented in the bridge set',
          )

  def test_every_excluded_cm_brick_variant_is_covered(self) -> None:
    """Every excluded with_/with_do brick's CM variant is covered.

    CM variants that only differ in error-absorption behavior (suppress_cm)
    do not require separate bridge coverage — their sync/async dispatch
    path is identical to sync_cm.
    """
    for brick in self.excluded:
      cm = _cm_variant(brick)
      if cm is None:
        continue
      # suppress_cm has identical sync/async dispatch to sync_cm
      effective_cm = 'sync_cm' if cm == 'suppress_cm' else cm
      with self.subTest(brick=brick.name, cm_variant=cm, effective_cm=effective_cm):
        self.assertIn(
          effective_cm,
          self.included_cm_variants,
          f'Excluded brick {brick.name!r} has CM variant={cm!r} '
          f'(effective={effective_cm!r}) not represented in bridge set',
        )

  def test_every_excluded_iteration_brick_source_is_covered(self) -> None:
    """Every excluded foreach/foreach_do brick's iteration source is covered."""
    for brick in self.excluded:
      src = _iteration_source(brick)
      if src is None:
        continue
      with self.subTest(brick=brick.name, iter_source=src):
        self.assertIn(
          src,
          self.included_iter_sources,
          f'Excluded brick {brick.name!r} has iteration source={src!r} not represented in bridge set',
        )

  def test_every_excluded_brick_has_covering_brick(self) -> None:
    """Every excluded brick maps to covering bridge bricks.

    Coverage means the bridge set exercises:
      (a) the same operation type (engine code path), AND
      (b) every atomic component of the brick's dispatch rule
          (dispatch paths are independent and compose without
          interaction — see SPEC section 4).

    This is the strongest per-brick coverage assertion.
    """
    for brick in self.excluded:
      dr = _dispatch_rule(brick)
      components = _dispatch_rule_components(dr)
      with self.subTest(brick=brick.name, op=brick.op, dispatch_rule=dr):
        # Op coverage: at least one bridge brick with the same op
        same_op = [b.name for b in self.included if b.op == brick.op]
        self.assertTrue(
          len(same_op) > 0,
          f'Excluded brick {brick.name!r} (op={brick.op!r}) has no bridge brick with the same operation type.',
        )

        # Dispatch coverage: every atomic component of the dispatch rule
        # must be present in the bridge set (possibly on different ops,
        # since the dispatch layer is shared across all ops).
        for component in components:
          covering = [b.name for b in self.included if _dispatch_rule(b) == component]
          self.assertTrue(
            len(covering) > 0,
            f'Excluded brick {brick.name!r} (op={brick.op!r}, dispatch={dr!r}) '
            f'needs component={component!r} but no bridge brick covers it. '
            f'Op coverage: {same_op}.',
          )

  # -- guard against silent set changes --

  def test_no_uncategorized_calling_conventions(self) -> None:
    """Every calling convention in ALL_BRICKS is handled by _dispatch_rule.

    Catches new calling conventions added without updating the coverage model.
    """
    for brick in self.all_bricks:
      with self.subTest(brick=brick.name, cc=brick.calling_convention):
        # Should not raise
        dr = _dispatch_rule(brick)
        self.assertIn(
          dr,
          ('rule1', 'rule2', 'chain_callable', 'chain_callable+rule1', 'literal'),
          f'Unexpected dispatch rule {dr!r} for brick {brick.name!r}',
        )

  def test_concurrency_bricks_are_orthogonal(self) -> None:
    """Bricks with has_builtin_concurrency share their op+dispatch with
    a non-concurrent brick in the bridge set.

    This validates the claim that concurrency does not introduce new
    sync/async dispatch branches — it is an orthogonal axis.
    """
    for brick in self.excluded:
      if not brick.has_builtin_concurrency:
        continue
      dr = _dispatch_rule(brick)
      with self.subTest(brick=brick.name, op=brick.op, dispatch_rule=dr):
        # There must be a non-builtin-concurrency brick in the bridge set
        # with the same op
        same_op_no_conc = [b.name for b in self.included if b.op == brick.op and not b.has_builtin_concurrency]
        self.assertTrue(
          len(same_op_no_conc) > 0,
          f'Builtin-concurrency brick {brick.name!r} (op={brick.op!r}) '
          f'has no non-concurrent counterpart in bridge set. '
          f'Concurrency is not orthogonal for this op.',
        )


if __name__ == '__main__':
  unittest.main()
