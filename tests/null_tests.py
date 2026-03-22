# SPDX-License-Identifier: MIT
"""Tests for SPEC §12 — Null Sentinel.

Covers Null vs None distinction, external boundary normalization,
effect on calling conventions, and singleton properties.
"""

from __future__ import annotations

import copy
from unittest import TestCase

from quent import Q
from quent._types import Null
from quent._types import _Null as NullType
from tests.fixtures import sync_fn
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §12.1: Null vs None distinction
# ---------------------------------------------------------------------------


class NullVsNoneTest(SymmetricTestCase):
  """SPEC §12.1: Null distinguishes 'no value' from None."""

  async def test_q_no_arg_has_no_root_value(self) -> None:
    """Q() — no root value (internal Null)."""
    # Verify by checking that fn gets called with zero args
    result = Q().then(lambda: 'no_root').run()
    self.assertEqual(result, 'no_root')

  async def test_q_none_has_root_value_none(self) -> None:
    """Q(None) — root value is None."""
    # Verify by checking that fn gets called with None
    result = Q(None).then(lambda x: x is None).run()
    self.assertTrue(result)

  async def test_q_none_passes_none_to_then(self) -> None:
    """Q(None).then(fn) → fn(None), not fn()."""
    received = []

    def capture_args(*args):
      received.append(args)
      return args

    Q(None).then(capture_args).run()
    self.assertEqual(received, [(None,)])

  async def test_q_empty_calls_then_with_no_args(self) -> None:
    """Q().then(fn) → fn(), not fn(None)."""
    received = []

    def capture_args(*args):
      received.append(args)
      return args

    Q().then(capture_args).run()
    self.assertEqual(received, [()])


# ---------------------------------------------------------------------------
# §12.2: External boundary — Null is never exposed
# ---------------------------------------------------------------------------


class NullNeverExposedTest(SymmetricTestCase):
  """SPEC §12.2: Null is never exposed to user code."""

  async def test_run_returns_none_not_null(self) -> None:
    """run() returns None when no value, not Null."""
    result = Q().run()
    self.assertIsNone(result)
    self.assertIsNot(result, Null)

  async def test_run_with_do_only_returns_none(self) -> None:
    """Pipeline with only .do() steps returns None."""
    result = Q().do(lambda: None).run()
    self.assertIsNone(result)

  async def test_except_handler_receives_none_not_null(self) -> None:
    """Except handler receives None when pipeline has no root value."""
    received = []

    def handler(info):
      received.append(info.root_value)
      self.assertIsNone(info.root_value)
      self.assertIsNot(info.root_value, Null)
      return 'handled'

    result = Q().then(lambda: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(received), 1)

  async def test_finally_handler_receives_none_not_null(self) -> None:
    """Finally handler receives None when pipeline has no root value."""
    received = []

    def cleanup(rv):
      received.append(rv)
      self.assertIsNone(rv)
      self.assertIsNot(rv, Null)

    Q().then(lambda: 42).finally_(cleanup).run()
    self.assertEqual(len(received), 1)


# ---------------------------------------------------------------------------
# §12.3: Effect on calling conventions
# ---------------------------------------------------------------------------


class NullCallingConventionTest(SymmetricTestCase):
  """SPEC §12.3: Null affects how pipeline steps are called."""

  async def test_null_current_value_calls_with_zero_args(self) -> None:
    """When current value is Null, callable gets zero args: fn()."""
    result = Q().then(lambda: 'zero_args').run()
    self.assertEqual(result, 'zero_args')

  async def test_none_current_value_calls_with_none(self) -> None:
    """When current value is None, callable gets fn(None)."""
    result = Q(None).then(lambda x: f'got_{x}').run()
    self.assertEqual(result, 'got_None')

  async def test_explicit_args_override_regardless(self) -> None:
    """Explicit args override regardless of Null/non-Null status."""
    # With Null (no root)
    result1 = Q().then(sync_fn, 5).run()
    self.assertEqual(result1, 6)
    # With value
    result2 = Q(99).then(sync_fn, 5).run()
    self.assertEqual(result2, 6)

  async def test_q_empty_then_fn_bridge(self) -> None:
    """Q().then(fn) with sync/async fn — fn() called with zero args per §12.3."""

    async def async_zero():
      return 42

    # Test with Null -> zero-arg calling convention (no Ellipsis needed)
    await self.variant(
      lambda fn: Q().then(fn).run(),
      fn=[('sync', lambda: 42), ('async', async_zero)],
      expected=42,
    )

  async def test_do_step_with_null_value(self) -> None:
    """do(fn) with Null current value — fn() called with zero args per §12.3."""
    received_args = []

    def side_effect(*args):
      received_args.append(args)

    # Q() has no root value (Null), so do(fn) should call fn() with zero args
    Q().do(side_effect).run()
    self.assertEqual(received_args, [()], 'Null means no value — fn should receive zero args')

  async def test_multiple_steps_null_propagation(self) -> None:
    """When then() returns None, subsequent step gets fn(None)."""
    result = Q(5).then(lambda x: None).then(lambda x: x is None).run()
    self.assertTrue(result)

  async def test_then_non_callable_after_null(self) -> None:
    """Non-callable value replaces Null in the pipeline."""
    result = Q().then(42).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# §12.4: Singleton properties
# ---------------------------------------------------------------------------


class NullSingletonTest(TestCase):
  """SPEC §12.4: Null singleton properties."""

  def test_null_is_singleton(self) -> None:
    """Exactly one Null instance exists."""
    self.assertIs(Null, Null)

  def test_copy_returns_same_instance(self) -> None:
    """copy.copy(Null) is Null."""
    self.assertIs(copy.copy(Null), Null)

  def test_deepcopy_returns_same_instance(self) -> None:
    """copy.deepcopy(Null) is Null."""
    self.assertIs(copy.deepcopy(Null), Null)

  def test_repr(self) -> None:
    """repr(Null) → '<Null>'."""
    self.assertEqual(repr(Null), '<Null>')

  def test_null_type(self) -> None:
    """Null is an instance of NullType (_Null)."""
    self.assertIsInstance(Null, NullType)

  def test_null_is_not_none(self) -> None:
    """Null is not None."""
    self.assertIsNot(Null, None)
    self.assertNotEqual(Null, None)

  def test_null_is_truthy_by_default_object_behavior(self) -> None:
    """Null is truthy — default Python object truthiness (no __bool__ override)."""
    # Objects are truthy by default in Python
    self.assertTrue(bool(Null))

  def test_nulltype_instantiation_raises_type_error(self) -> None:
    """SPEC §12.4: NullType() raises TypeError — singleton is enforced, not just conventional."""
    with self.assertRaises(TypeError) as ctx:
      NullType()
    self.assertIn('_Null is a singleton', str(ctx.exception))
    self.assertIn('quent.Null', str(ctx.exception))


class QReprTest(TestCase):
  """Coverage for Q.__repr__ edge cases."""

  def test_repr_with_root_link(self) -> None:
    """repr includes root link name. Covers _q.py line 312."""
    c = Q(int)
    r = repr(c)
    self.assertIn('Q(', r)
    self.assertIn('int', r)

  def test_repr_truncates_long_pipelines(self) -> None:
    """repr truncates pipelines exceeding the per-level link limit (100). SPEC section 13.11."""
    c = Q(5)
    for _ in range(110):
      c = c.then(lambda x: x)
    r = repr(c)
    self.assertIn('... and 10 more steps', r)


# ---------------------------------------------------------------------------
# §12: Integration — Null boundary across pipeline operations
# ---------------------------------------------------------------------------


class NullBoundaryIntegrationTest(SymmetricTestCase):
  """Integration tests for Null boundary behavior across operations."""

  async def test_except_handler_no_root_nested_pipeline(self) -> None:
    """Nested pipeline except handler receives None as root value."""
    inner = Q().then(lambda info: info.root_value)
    result = Q().then(lambda: 1 / 0).except_(inner).run()
    # Root value normalized to None
    self.assertIsNone(result)

  async def test_run_value_replaces_null_root(self) -> None:
    """run(v) replaces Null root — fn receives v, not nothing."""
    result = Q().then(sync_fn).run(5)
    self.assertEqual(result, 6)

  async def test_q_none_vs_q_empty_distinct(self) -> None:
    """Q(None) and Q() are observably distinct."""
    received_none = []
    received_empty = []

    def capture_none(*args):
      received_none.append(args)
      return args

    def capture_empty(*args):
      received_empty.append(args)
      return args

    Q(None).then(capture_none).run()
    Q().then(capture_empty).run()

    # Q(None) passes None; Q() passes nothing
    self.assertEqual(received_none, [(None,)])
    self.assertEqual(received_empty, [()])

  async def test_gather_with_null_root(self) -> None:
    """gather() with no root value — each fn gets no args.
    Need to provide a value since gather always passes current value."""
    result = Q(0).gather(lambda x: 1, lambda x: 2).run()
    self.assertEqual(result, (1, 2))

  async def test_map_with_none_value(self) -> None:
    """map() works when pipeline value is a list containing None."""
    result = Q([None, None]).foreach(lambda x: x is None).run()
    self.assertEqual(result, [True, True])

  async def test_if_null_predicate_is_falsy(self) -> None:
    """SPEC §5.8: When predicate is None and pipeline has no current value
    (Null sentinel), the predicate evaluates to falsy. The then branch
    should NOT execute."""
    called = []

    def then_fn(x):
      called.append(True)
      return 'then_result'

    # Q() has no root value (Null). if_(then=fn) uses current value as
    # predicate. Null is always falsy for predicate purposes.
    result = Q().if_().then(then_fn).run()
    self.assertIsNone(result)  # Null normalized to None on output
    self.assertEqual(called, [])  # then branch NOT called

  async def test_if_null_predicate_with_else(self) -> None:
    """SPEC §5.8: Null predicate triggers else branch when registered."""
    result = Q().if_().then(lambda x: 'then').else_(lambda: 'else').run()
    self.assertEqual(result, 'else')
