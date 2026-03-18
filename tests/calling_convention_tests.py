# SPDX-License-Identifier: MIT
"""Tests for SPEC §4 — Calling Conventions.

Covers the 2 standard rules (Rule 1: Explicit Args/Kwargs, Rule 2: Default Passthrough),
nested q behavior (callable — follows standard rules), except handler calling convention (QuentExcInfo),
finally handler calling convention, if_() predicate calling convention,
and do() calling convention.
"""

from __future__ import annotations

from quent import Q, QuentException
from tests.fixtures import (
  V_GT0,
  V_KW,
  capture,
  sync_add,
  sync_double,
  sync_fn,
)
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §4: Nested Q pipeline (standard rules apply)
# ---------------------------------------------------------------------------


class NestedChainTest(SymmetricTestCase):
  """SPEC §4: Nested chain is callable — follows standard rules."""

  async def test_nested_chain_signal_propagation(self) -> None:
    """Control flow signals propagate from nested chain to outer chain."""
    inner = Q().then(lambda x: Q.return_('early'))
    result = Q(5).then(inner).then(lambda x: 'not reached').run()
    self.assertEqual(result, 'early')


# ---------------------------------------------------------------------------
# §4: Rule 1 — Explicit Args/Kwargs
# ---------------------------------------------------------------------------


class ExplicitArgsTest(SymmetricTestCase):
  """SPEC §4 Rule 1: Explicit args suppress current value."""

  async def test_explicit_kwargs(self) -> None:
    """fn(**kwargs) — current value is NOT passed."""
    await self.variant(
      lambda fn: Q(5).then(fn, key='hello').run(),
      fn=V_KW,
      expected='hello',
    )

  async def test_explicit_args_and_kwargs(self) -> None:
    """fn(*args, **kwargs)."""

    def fn(a, *, key):
      return f'{a}-{key}'

    result = Q(999).then(fn, 'X', key='Y').run()
    self.assertEqual(result, 'X-Y')

  async def test_non_callable_with_args_raises(self) -> None:
    """Providing args to a non-callable raises TypeError at build time."""
    # Non-callables cannot receive args at build time (Link constructor check)
    with self.assertRaises(TypeError):
      Q(5).then(42, 'arg')


# ---------------------------------------------------------------------------
# §4: Rule 2 — Default Passthrough
# ---------------------------------------------------------------------------


class DefaultPassthroughTest(SymmetricTestCase):
  """SPEC §4 Rule 2: Default calling behavior."""

  async def test_callable_no_current_value(self) -> None:
    """fn() — callable with no current value gets zero args."""
    result = Q().then(lambda: 42).run()
    self.assertEqual(result, 42)

  async def test_non_callable_returned_as_is(self) -> None:
    """Non-callable is returned as-is as the new current value."""
    result = Q(5).then(42).run()
    self.assertEqual(result, 42)

  async def test_none_is_passed_as_value(self) -> None:
    """None is a valid current value — fn(None)."""
    result = Q(None).then(lambda x: x is None).run()
    self.assertTrue(result)


# ---------------------------------------------------------------------------
# §4: Except Handler Calling Convention (QuentExcInfo)
# ---------------------------------------------------------------------------


class ExceptHandlerConventionTest(SymmetricTestCase):
  """SPEC §4: Except handler calling convention — QuentExcInfo pattern."""

  async def test_default_handler(self) -> None:
    """Default: handler(info) where info is QuentExcInfo."""
    received = []

    def handler(info):
      received.append((type(info.exc).__name__, info.root_value))
      return 'handled'

    result = Q(42).then(lambda x: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(received, [('ZeroDivisionError', 42)])

  async def test_explicit_args(self) -> None:
    """Explicit args: handler(*args, **kwargs) — QuentExcInfo NOT passed."""
    received = []

    def handler(a, b):
      received.append((a, b))
      return 'handled'

    result = Q(42).then(lambda x: 1 / 0).except_(handler, 'x', 'y').run()
    self.assertEqual(result, 'handled')
    self.assertEqual(received, [('x', 'y')])

  async def test_nested_chain_default(self) -> None:
    """Nested chain default: inner_chain runs with QuentExcInfo as input."""
    inner = Q().then(lambda info: (type(info.exc).__name__, info.root_value))
    result = Q(42).then(lambda x: 1 / 0).except_(inner).run()
    self.assertEqual(result, ('ZeroDivisionError', 42))

  async def test_nested_chain_explicit_args(self) -> None:
    """Nested chain explicit args: inner_chain runs with explicit args."""
    inner = Q().then(sync_double)
    result = Q(42).then(lambda x: 1 / 0).except_(inner, 7).run()
    # inner receives 7 as input, sync_double(7) = 14
    self.assertEqual(result, 14)

  async def test_root_value_normalized_to_none(self) -> None:
    """Root value is None when chain has no root value."""
    received = []

    def handler(info):
      received.append(info.root_value)
      return 'handled'

    result = Q().then(lambda: 1 / 0).except_(handler).run()
    self.assertEqual(result, 'handled')
    self.assertEqual(received, [None])  # Null normalized to None

  async def test_except_requires_callable(self) -> None:
    """except_() requires a callable handler."""
    with self.assertRaises(TypeError):
      Q(5).except_(42)

  async def test_duplicate_except_raises(self) -> None:
    """Only one except_ per chain."""
    with self.assertRaises(QuentException):
      Q(5).except_(lambda info: None).except_(lambda info: None)


# ---------------------------------------------------------------------------
# §4: Finally Handler Calling Convention
# ---------------------------------------------------------------------------


class FinallyHandlerConventionTest(SymmetricTestCase):
  """SPEC §4: Finally handler uses standard calling conventions."""

  async def test_finally_receives_root_value(self) -> None:
    """Finally handler receives root value as current value."""
    received = []

    def cleanup(rv):
      received.append(rv)

    Q(42).then(sync_fn).finally_(cleanup).run()
    self.assertEqual(received, [42])

  async def test_finally_root_value_normalized_to_none(self) -> None:
    """Finally receives None when chain has no root value."""
    received = []

    def cleanup(rv):
      received.append(rv)

    Q().then(lambda: 99).finally_(cleanup).run()
    self.assertEqual(received, [None])

  async def test_finally_with_explicit_args(self) -> None:
    """Explicit args in finally suppress root value — standard convention."""
    received = []

    def cleanup(a, b):
      received.append((a, b))

    Q(42).then(sync_fn).finally_(cleanup, 'x', 'y').run()
    self.assertEqual(received, [('x', 'y')])

  async def test_finally_return_value_discarded(self) -> None:
    """Finally handler's return value is always discarded."""
    result = Q(42).then(sync_fn).finally_(lambda rv: 999).run()
    # sync_fn(42) = 43, finally returns 999 but it's discarded
    self.assertEqual(result, 43)

  async def test_finally_requires_callable(self) -> None:
    """finally_() requires a callable handler."""
    with self.assertRaises(TypeError):
      Q(5).finally_(42)

  async def test_duplicate_finally_raises(self) -> None:
    """Only one finally_ per chain."""
    with self.assertRaises(QuentException):
      Q(5).finally_(lambda rv: None).finally_(lambda rv: None)

  async def test_finally_runs_on_success(self) -> None:
    """Finally handler runs on success path."""
    called = []

    def cleanup(rv):
      called.append(True)

    result = Q(5).then(sync_fn).finally_(cleanup).run()
    self.assertEqual(result, 6)
    self.assertTrue(called)

  async def test_finally_runs_on_failure(self) -> None:
    """Finally handler runs on failure path."""
    called = []

    def cleanup(rv):
      called.append(True)

    result = await capture(lambda: Q(5).then(lambda x: 1 / 0).finally_(cleanup).run())
    self.assertFalse(result.success)
    self.assertTrue(called)

  async def test_finally_nested_chain_receives_root_value(self) -> None:
    """Nested chain as finally handler receives root value as input."""
    received = []

    def record(rv):
      received.append(rv)

    async def async_record(rv):
      received.append(rv)

    await self.variant(
      lambda fn: Q(5).then(lambda x: x * 2).finally_(Q().then(fn)).run(),
      fn=[('sync', record), ('async', async_record)],
      expected=10,
    )
    # The inner q receives root value (5) as its input.
    # fn(5) is called for each variant; return value is discarded.
    self.assertEqual(received, [5, 5])


# ---------------------------------------------------------------------------
# §4: Awaitable detection edge cases (_isawaitable coverage)
# ---------------------------------------------------------------------------


class AwaitableDetectionTest(SymmetricTestCase):
  """Tests for edge cases in awaitable detection (_isawaitable)."""

  async def test_object_with_getattr_raising_non_attribute_error(self) -> None:
    """Object whose __getattr__ raises non-AttributeError is handled gracefully.

    Covers _eval.py lines 65-66: the except Exception branch.
    """

    class BadGetattr:
      def __getattr__(self, name):
        raise RuntimeError('unexpected error from __getattr__')

    # Should not raise — _isawaitable catches the RuntimeError and returns False
    result = Q(5).then(lambda x: BadGetattr()).then(lambda x: 'ok').run()
    self.assertEqual(result, 'ok')

  async def test_custom_awaitable_with_dunder_await(self) -> None:
    """Object with __await__ method is recognized as awaitable.

    Covers _eval.py line 64: the getattr __await__ path.
    """
    import asyncio

    class CustomAwaitable:
      def __init__(self, value):
        self._value = value

      def __await__(self):
        return asyncio.coroutine(lambda: self._value)().__await__()

    async def step_returning_awaitable(x):
      return x + 1

    result = await Q(5).then(step_returning_awaitable).run()
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# §4: Except handler with explicit args + kwargs
# ---------------------------------------------------------------------------


class ExceptRule5CombinedArgsKwargsTest(SymmetricTestCase):
  """Except handler with explicit args+kwargs: handler(*args, **kwargs), QuentExcInfo NOT passed."""

  async def test_except_explicit_args_and_kwargs(self) -> None:
    """except_(handler, 'x', key='val') → handler('x', key='val').

    Per the 2-rule convention: explicit args suppress QuentExcInfo.
    """
    received = []

    def handler(a, *, key):
      received.append((a, key))
      return 'handled'

    result = Q(42).then(lambda x: 1 / 0).except_(handler, 'x', key='val').run()
    self.assertEqual(result, 'handled')
    self.assertEqual(received, [('x', 'val')])


# ---------------------------------------------------------------------------
# §4: Multi-arg nested q
# ---------------------------------------------------------------------------


class MultiArgNestedChainTest(SymmetricTestCase):
  """SPEC §4: .then(inner_chain, arg1, arg2) — extra args flow through."""

  async def test_multi_arg_nested_chain_callable_root(self) -> None:
    """then(inner_chain, callable, arg1, arg2) — extra args flow to root evaluation.

    Per SPEC §4 and _eval.py:164-167:
    .then(inner_chain, fn, a, b) → run_value=fn, run_args=(a, b).
    Inner q runs with fn(*run_args) as root value.
    """
    inner = Q().then(sync_double)
    result = Q(999).then(inner, sync_add, 3, 4).run()
    # _eval: run_value=sync_add, run_args=(3, 4) → inner root = sync_add(3, 4) = 7
    # then sync_double(7) = 14
    self.assertEqual(result, 14)

  async def test_multi_arg_nested_chain_with_kwargs(self) -> None:
    """then(inner_chain, arg1, key=val) — kwargs flow to inner chain root.

    Per SPEC §4: kwargs flow through to inner q's root evaluation.
    Via _eval.py:164-167: run_value=args[0], kwargs passed through.
    """

    def root_fn(a, *, key):
      return f'{a}-{key}'

    inner = Q().then(lambda x: x.upper())
    result = Q(999).then(inner, root_fn, 'hello', key='world').run()
    # _eval: run_value=root_fn, run_args=('hello',), kwargs={'key': 'world'}
    # inner root = root_fn('hello', key='world') = 'hello-world'
    # then upper() = 'HELLO-WORLD'
    self.assertEqual(result, 'HELLO-WORLD')

  async def test_kwargs_only_nested_chain_does_not_pass_current_value(self) -> None:
    """then(inner_chain, key=val) — kwargs only, current_value is NOT passed.

    Per SPEC §4: when only kwargs are provided (no positional args),
    the inner q runs with no run value (Null). current_value must NOT
    leak through.
    """

    def root_fn(*, key):
      return f'got-{key}'

    inner = Q(root_fn).then(lambda x: x.upper())
    result = Q(999).then(inner, key='hello').run()
    # _eval: run_value=Null (no positional args), kwargs={'key': 'hello'}
    # inner root = root_fn(key='hello') = 'got-hello'
    # then upper() = 'GOT-HELLO'
    self.assertEqual(result, 'GOT-HELLO')

  async def test_nested_chain_kwargs_replace_not_merge(self) -> None:
    """SPEC §4: caller kwargs replace inner chain's build-time kwargs entirely.

    When .then(inner_chain, key=val) invokes an inner q that has build-time
    kwargs on its root, the caller's kwargs replace them — no merging.
    """

    def root_fn(*, key):
      return f'got-{key}'

    inner = Q(root_fn, key='build_time').then(lambda x: x.upper())
    result = Q(999).then(inner, key='run_time').run()
    # Caller kwargs replace build-time: root_fn(key='run_time'), not root_fn(key='build_time', ...)
    self.assertEqual(result, 'GOT-RUN_TIME')


# ---------------------------------------------------------------------------
# §4: No-root-value except handler nested q
# ---------------------------------------------------------------------------


class NoRootValueExceptNestedChainTest(SymmetricTestCase):
  """Nested chain except handler receives QuentExcInfo where root_value is None."""

  async def test_no_root_except_nested_chain(self) -> None:
    """No root value: nested chain receives QuentExcInfo with root_value=None.

    Per SPEC §4 root_value normalization: when the root value is the internal
    'no value' sentinel, it is normalized to None before being passed to the handler.
    """
    inner = Q().then(lambda info: (type(info.exc).__name__, info.root_value))
    result = Q().then(lambda: 1 / 0).except_(inner).run()
    self.assertEqual(result[0], 'ZeroDivisionError')
    self.assertIsNone(result[1])  # Null normalized to None


# ---------------------------------------------------------------------------
# §4: if_() Predicate Calling Convention
# ---------------------------------------------------------------------------


class IfPredicateConventionTest(SymmetricTestCase):
  """SPEC §4 / §5.8: if_() predicate follows the standard calling convention.

  The predicate's "current value" is the pipeline's current value.
  Rule 1: Explicit args — predicate(*args, **kwargs), cv NOT passed.
  Rule 2: Default — predicate(cv), or cv truthiness if None, or literal truthiness.
  Nested Q predicates are callable and follow the default rule.
  """

  async def test_nested_chain_predicate_receives_cv(self) -> None:
    """Nested chain (default rule): predicate chain runs with cv as input, result tested for truthiness."""
    # Inner q receives cv=5, sync_gt0(5)=True → truthy → then branch runs
    await self.variant(
      lambda fn: Q(5).if_(Q().then(fn)).then(sync_double).run(),
      fn=V_GT0,
      expected=10,
    )

  async def test_nested_chain_predicate_falsy(self) -> None:
    """Nested chain (default rule): predicate chain returns falsy — then branch skipped."""
    # Inner q receives cv=-1, sync_gt0(-1)=False → falsy → then branch skipped
    await self.variant(
      lambda fn: Q(-1).if_(Q().then(fn)).then(sync_double).run(),
      fn=V_GT0,
      expected=-1,
    )

  async def test_predicate_with_explicit_args(self) -> None:
    """Rule 1 (explicit args): predicate invoked with explicit args, cv NOT passed."""
    # predicate(10) → sync_gt0(10)=True, cv=5 is NOT passed to predicate
    await self.variant(
      lambda fn: Q(5).if_(fn, 10).then(sync_double).run(),
      fn=V_GT0,
      expected=10,
    )

  async def test_predicate_explicit_args_falsy(self) -> None:
    """Rule 1 (explicit args): predicate with explicit args returns falsy."""
    # predicate(-1) → sync_gt0(-1)=False, then branch skipped
    await self.variant(
      lambda fn: Q(5).if_(fn, -1).then(sync_double).run(),
      fn=V_GT0,
      expected=5,
    )

  async def test_default_callable_predicate(self) -> None:
    """Rule 2 (default): callable predicate receives cv."""
    # predicate(5) → sync_gt0(5)=True → then branch runs
    await self.variant(
      lambda fn: Q(5).if_(fn).then(sync_double).run(),
      fn=V_GT0,
      expected=10,
    )

  async def test_default_callable_predicate_falsy(self) -> None:
    """Rule 2 (default): callable predicate returns falsy — then branch skipped."""
    await self.variant(
      lambda fn: Q(-1).if_(fn).then(sync_double).run(),
      fn=V_GT0,
      expected=-1,
    )

  async def test_none_predicate_uses_cv_truthiness(self) -> None:
    """None predicate: uses cv truthiness directly."""
    # cv=5 is truthy → then branch runs
    result = Q(5).if_().then(sync_double).run()
    self.assertEqual(result, 10)

  async def test_none_predicate_falsy_cv(self) -> None:
    """None predicate: falsy cv → then branch skipped."""
    result = Q(0).if_().then(sync_double).run()
    self.assertEqual(result, 0)

  async def test_literal_predicate_truthy(self) -> None:
    """Literal predicate: uses its own truthiness, not cv."""
    # predicate=True (literal, non-callable) → truthy → then branch runs
    result = Q(5).if_(True).then(sync_double).run()
    self.assertEqual(result, 10)

  async def test_literal_predicate_falsy(self) -> None:
    """Literal predicate: falsy literal → then branch skipped."""
    # predicate=0 (literal falsy) → then branch skipped
    result = Q(5).if_(0).then(sync_double).run()
    self.assertEqual(result, 5)


# ---------------------------------------------------------------------------
# §4 / §5.2: do() Calling Convention
# ---------------------------------------------------------------------------


class DoCallingConventionTest(SymmetricTestCase):
  """SPEC §4 / §5.2: do() follows the standard calling convention.

  do(fn) invokes fn as a side-effect. The return value of fn is discarded.
  The current pipeline value passes through unchanged.
  """

  async def test_explicit_args_in_do(self) -> None:
    """Rule 1 (explicit args): fn(*args), cv NOT passed, result discarded."""
    received = []

    def record(a, b):
      received.append((a, b))

    result = Q(5).do(record, 'x', 'y').run()
    self.assertEqual(result, 5)
    self.assertEqual(received, [('x', 'y')])

  async def test_default_do_receives_cv(self) -> None:
    """Rule 2 (default): fn(cv), result discarded, cv passes through."""
    received = []

    def record(x):
      received.append(x)

    result = Q(5).do(record).run()
    self.assertEqual(result, 5)
    self.assertEqual(received, [5])

  async def test_do_requires_callable(self) -> None:
    """do() requires a callable — non-callable raises TypeError at build time."""
    with self.assertRaises(TypeError):
      Q(5).do(42)
