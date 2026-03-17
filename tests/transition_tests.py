# SPDX-License-Identifier: MIT
"""Method Transition Matrix test — chains every triplet of atomic operations and verifies correctness,
across all sync/async variant combinations. Currently 26 atomic ops."""

from __future__ import annotations

import asyncio
import itertools
from unittest import IsolatedAsyncioTestCase

from quent import Chain
from quent._context import _ctx_store
from tests.tests_helper import SyncCM


def _reset_context() -> None:
  try:
    _ctx_store.get()
    _ctx_store.set({})
  except LookupError:
    pass


# ---------------------------------------------------------------------------
# Async helper functions (module-level to avoid closure issues in lambdas)
# ---------------------------------------------------------------------------


async def _async_add1(x):
  return x + 1


async def _async_add2(x):
  return x + 2


async def _async_noop(x):
  return None


async def _async_true(x):
  return True


async def _async_false(x):
  return False


async def _async_neg1(x):
  return -1


async def _async_kw_add1(*, x):
  return x + 1


async def _async_add3(x):
  return x + 3


# ---------------------------------------------------------------------------
# Atomic operations table: (name, apply_factory, oracle_fn, swappable)
#
# Each op is defined as a 4-tuple:
#   - name:          string label for subTest reporting
#   - apply_factory: callable(chain, is_async) -> None  (mutates chain in-place)
#   - oracle_fn:     callable(v) -> expected  (given input v, what should the output be?)
#   - swappable:     bool — whether this op has a callable that can be swapped sync/async
#
# Context keys use unique names per-op to avoid collisions when two ops are
# chained (e.g. 'set_cv' + 'get' must not share a key).
# ---------------------------------------------------------------------------

ATOMIC_OPS = [
  # 1. then: transforms CV by adding 1
  (
    'then',
    lambda c, a, _af=_async_add1: c.then(_af if a else lambda x: x + 1),
    lambda v: v + 1,
    True,
  ),
  # 2. do: side-effect only, CV unchanged
  (
    'do',
    lambda c, a, _af=_async_noop: c.do(_af if a else lambda x: None),
    lambda v: v,
    True,
  ),
  # 3. set_cv: stores current value under '_sc', CV unchanged
  (
    'set_cv',
    lambda c, a: c.set('_sc'),
    lambda v: v,
    False,
  ),
  # 4. set_explicit: stores explicit value 99 under '_se', CV unchanged
  (
    'set_explicit',
    lambda c, a: c.set('_se', 99),
    lambda v: v,
    False,
  ),
  # 5. get: round-trip — stores CV, adds 1 (swappable), then retrieves original; output == input v
  (
    'get',
    lambda c, a, _af=_async_add1: c.set('_g').then(_af if a else lambda x: x + 1).get('_g'),
    lambda v: v,
    True,
  ),
  # 6. foreach: [v, v+1] -> apply x+1 to each -> [v+1, v+2] -> sum
  (
    'foreach',
    lambda c, a, _af=_async_add1: c.then(lambda x: [x, x + 1]).foreach(_af if a else lambda x: x + 1).then(sum),
    lambda v: (v + 1) + (v + 2),
    True,
  ),
  # 7. foreach_do: [v, v+1] -> side-effect (unchanged) -> sum = v + (v+1)
  (
    'foreach_do',
    lambda c, a, _af=_async_noop: c.then(lambda x: [x, x + 1]).foreach_do(_af if a else lambda x: None).then(sum),
    lambda v: v + (v + 1),
    True,
  ),
  # 8. gather: produces tuple (v+1, v+2), extracts first element
  (
    'gather',
    lambda c, a, _af1=_async_add1, _af2=_async_add2: c.gather(
      _af1 if a else lambda x: x + 1,
      _af2 if a else lambda x: x + 2,
    ).then(lambda t: t[0]),
    lambda v: v + 1,
    True,
  ),
  # 9. with_: SyncCM(v).__enter__ returns v; fn(v) = v+1
  (
    'with_',
    lambda c, a, _af=_async_add1: c.then(SyncCM).with_(_af if a else lambda x: x + 1),
    lambda v: v + 1,
    True,
  ),
  # 10. with_do: SyncCM(v) entered, side-effect, outer_value (CM instance) passes through,
  #     then extract ._value to recover v
  (
    'with_do',
    lambda c, a, _af=_async_noop: c.then(SyncCM).with_do(_af if a else lambda x: None).then(lambda cm: cm._value),
    lambda v: v,
    True,
  ),
  # 11. if_then: condition always True, applies +1
  (
    'if_then',
    lambda c, a, _afp=_async_true, _afb=_async_add1: c.if_(_afp if a else lambda x: True).then(
      _afb if a else lambda x: x + 1
    ),
    lambda v: v + 1,
    True,
  ),
  # 12. if_do: condition always True, side-effect only, CV unchanged
  (
    'if_do',
    lambda c, a, _afp=_async_true, _afb=_async_noop: c.if_(_afp if a else lambda x: True).do(
      _afb if a else lambda x: None
    ),
    lambda v: v,
    True,
  ),
  # 13. if_else: condition always False, else branch adds 1
  (
    'if_else',
    lambda c, a, _afp=_async_false, _aft=_async_neg1, _afe=_async_add1: (
      c.if_(_afp if a else lambda x: False).then(_aft if a else lambda x: -1).else_(_afe if a else lambda x: x + 1)
    ),
    lambda v: v + 1,
    True,
  ),
  # 14. then_literal: non-callable value replaces CV
  (
    'then_literal',
    lambda c, a: c.then(42),
    lambda v: 42,
    False,
  ),
  # 15. nested_chain: composition via nested Chain
  (
    'nested_chain',
    lambda c, a, _af=_async_add1: c.then(Chain().then(_af if a else lambda x: x + 1)),
    lambda v: v + 1,
    True,
  ),
  # 16. except_nested: nested chain with except_ (happy path — handler not invoked)
  (
    'except_nested',
    lambda c, a, _af=_async_add1: c.then(Chain().then(_af if a else lambda x: x + 1).except_(lambda ei: -1)),
    lambda v: v + 1,
    True,
  ),
  # 17. finally_nested: nested chain with finally_ (handler fires, return discarded)
  (
    'finally_nested',
    lambda c, a, _af=_async_add1: c.then(Chain().then(_af if a else lambda x: x + 1).finally_(lambda rv: None)),
    lambda v: v + 1,
    True,
  ),
  # 18. gather_single: single-function gather, extract element
  (
    'gather_single',
    lambda c, a, _af=_async_add1: c.gather(_af if a else lambda x: x + 1).then(lambda t: t[0]),
    lambda v: v + 1,
    True,
  ),
  # 19. else_do: falsy predicate, side-effect else branch, CV unchanged
  (
    'else_do',
    lambda c, a, _afp=_async_false, _aft=_async_neg1, _afe=_async_noop: (
      c.if_(_afp if a else lambda x: False).then(_aft if a else lambda x: -1).else_do(_afe if a else lambda x: None)
    ),
    lambda v: v,
    True,
  ),
  # 20. then_kwargs: kwargs-only calling convention (Rule 1)
  (
    'then_kwargs',
    lambda c, a, _af=_async_kw_add1: c.then(_af if a else lambda *, x: x + 1, x=42),
    lambda v: 43,
    True,
  ),
  # 21. if_no_pred: no predicate — CV itself tested for truthiness; 10 is truthy → then branch
  (
    'if_no_pred',
    lambda c, a, _af=_async_add1: c.if_().then(_af if a else lambda x: x + 1),
    lambda v: v + 1,  # v=10 is truthy, so then-branch fires
    True,
  ),
  # 22. else_literal: falsy pred, else branch is a non-callable literal 42
  (
    'else_literal',
    lambda c, a, _afp=_async_false: c.if_(_afp if a else lambda x: False).then(lambda x: -1).else_(42),
    lambda v: 42,
    True,
  ),
  # 23. with_args: with_ body receives explicit arg 42 instead of entered value
  (
    'with_args',
    lambda c, a, _af=_async_add1: c.then(SyncCM).with_(_af if a else lambda x: x + 1, 42),
    lambda v: 43,  # fn(42) = 43, not fn(entered_value)
    True,
  ),
  # 24. do_nested: nested chain as side-effect, CV unchanged
  (
    'do_nested',
    lambda c, a, _af=_async_add1: c.do(Chain().then(_af if a else lambda x: x + 1)),
    lambda v: v,
    True,
  ),
  # 25. from_steps: nested chain via from_steps constructor
  (
    'from_steps',
    lambda c, a, _af=_async_add1: c.then(Chain.from_steps(_af if a else lambda x: x + 1)),
    lambda v: v + 1,
    True,
  ),
  # 26. gather_3fns: gather with 3 functions, extract first
  (
    'gather_3fns',
    lambda c, a, _a1=_async_add1, _a2=_async_add2, _a3=_async_add3: c.gather(
      _a1 if a else lambda x: x + 1,
      _a2 if a else lambda x: x + 2,
      _a3 if a else lambda x: x + 3,
    ).then(lambda t: t[0]),
    lambda v: v + 1,
    True,
  ),
]


class MethodTransitionMatrixTest(IsolatedAsyncioTestCase):
  """Empirical proof that every method-triplet transition produces correct results,
  across all sync/async variant combinations."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  async def test_all_transitions(self) -> None:
    for a_name, a_apply, a_oracle, a_swap in ATOMIC_OPS:
      for b_name, b_apply, b_oracle, b_swap in ATOMIC_OPS:
        for c_name, c_apply, c_oracle, c_swap in ATOMIC_OPS:
          a_variants = [False, True] if a_swap else [False]
          b_variants = [False, True] if b_swap else [False]
          c_variants = [False, True] if c_swap else [False]
          for a_async, b_async, c_async in itertools.product(a_variants, b_variants, c_variants):
            with self.subTest(a=a_name, b=b_name, c=c_name, aa=a_async, ba=b_async, ca=c_async):
              _reset_context()
              chain = Chain(10)
              a_apply(chain, a_async)
              b_apply(chain, b_async)
              c_apply(chain, c_async)
              result = chain.run()
              if asyncio.iscoroutine(result):
                result = await result
              expected = c_oracle(b_oracle(a_oracle(10)))
              self.assertEqual(result, expected)


if __name__ == '__main__':
  import unittest

  unittest.main()
