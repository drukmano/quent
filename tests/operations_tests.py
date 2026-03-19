# SPDX-License-Identifier: MIT
"""Tests for SPEC §5 — Operations."""

from __future__ import annotations

import sys
import unittest
from typing import Any
from unittest import IsolatedAsyncioTestCase

from quent import Q, QuentException
from tests.fixtures import (
  V_CM,
  V_CM_SUPPRESSES,
  V_DOUBLE,
  V_FN,
  V_RAISE,
  V_TRIPLE,
  AsyncCM,
  AsyncCMSuppresses,
  AsyncRange,
  SyncCM,
  async_double,
  capture,
  sync_double,
  sync_fn,
)
from tests.symmetric import SymmetricTestCase

if sys.version_info < (3, 11):
  from quent._types import ExceptionGroup

# ---------------------------------------------------------------------------
# §5.1 then()
# ---------------------------------------------------------------------------


class ThenTests(SymmetricTestCase):
  """§5.1 — then() appends a step whose result replaces current value."""

  async def test_callable_with_kwargs(self) -> None:
    """fn(**kwargs) — explicit kwargs suppress current value."""

    def sync_kw(*, key: int) -> int:
      return key + 1

    async def async_kw(*, key: int) -> int:
      return key + 1

    await self.variant(
      lambda fn: Q(10).then(fn, key=42).run(),
      expected=43,
      fn=[('sync', sync_kw), ('async', async_kw)],
    )

  async def test_non_callable_literal(self) -> None:
    """Non-callable replaces current value as-is."""
    result = Q(10).then(42).run()
    self.assertEqual(result, 42)

  async def test_non_callable_pipeline(self) -> None:
    """Q(1).then(2).then(3).run() produces 3."""
    result = Q(1).then(2).then(3).run()
    self.assertEqual(result, 3)

  async def test_non_callable_with_args_raises_at_build_time(self) -> None:
    """§5.1/§4 Rule 4: then(non_callable, arg) raises TypeError at build time.

    Non-callable values cannot receive args/kwargs. This is enforced at
    build time (during .then()), not at run time.
    """
    with self.assertRaises(TypeError) as ctx:
      Q(5).then(42, 'extra_arg')  # error before .run()
    self.assertIn('not callable', str(ctx.exception))

  async def test_non_callable_with_kwargs_raises_at_build_time(self) -> None:
    """§5.1/§4 Rule 4: then(non_callable, key=val) raises TypeError at build time."""
    with self.assertRaises(TypeError) as ctx:
      Q(5).then(42, key='val')  # error before .run()
    self.assertIn('not callable', str(ctx.exception))

  async def test_nested_chain(self) -> None:
    """Nested pipeline receives current value as input."""
    inner = Q().then(lambda x: x * 2)
    await self.variant(
      lambda fn: Q(5).then(inner).then(fn).run(),
      expected=11,
      fn=V_FN,
    )

  async def test_result_replaces_current_value(self) -> None:
    """Each then() result becomes the new current value."""
    await self.variant(
      lambda fn: Q(1).then(fn).then(fn).then(fn).run(),
      expected=4,
      fn=V_FN,
    )


# ---------------------------------------------------------------------------
# §5.2 do()
# ---------------------------------------------------------------------------


class DoTests(SymmetricTestCase):
  """§5.2 — do() is a side-effect step; result discarded, value passes through."""

  async def test_side_effect_executed(self) -> None:
    """do() actually invokes fn."""
    calls: list[int] = []

    def sync_track(x: Any) -> None:
      calls.append(x)

    async def async_track(x: Any) -> None:
      calls.append(x)

    await self.variant(
      lambda fn: Q(10).do(fn).run(),
      expected=10,
      fn=[('sync', sync_track), ('async', async_track)],
    )
    self.assertTrue(len(calls) >= 1)

  async def test_requires_callable(self) -> None:
    """do() raises TypeError at build time if not callable."""
    with self.assertRaises(TypeError):
      Q(10).do(42)  # type: ignore[arg-type]

  async def test_awaitable_return_awaited_but_discarded(self) -> None:
    """If fn returns awaitable, it is awaited but result discarded."""
    awaited = []

    async def async_side(x: Any) -> str:
      awaited.append(x)
      return 'should be discarded'

    result = await Q(10).do(async_side).run()
    self.assertEqual(result, 10)
    self.assertEqual(awaited, [10])

  async def test_with_explicit_args(self) -> None:
    """do(fn, arg) — explicit args suppress current value."""
    calls: list[Any] = []

    def track(x: Any) -> None:
      calls.append(x)

    result = Q(10).do(track, 42).run()
    self.assertEqual(result, 10)
    self.assertEqual(calls, [42])


# ---------------------------------------------------------------------------
# §5.3 foreach()
# ---------------------------------------------------------------------------


class ForeachTests(SymmetricTestCase):
  """§5.3 — foreach(fn) applies fn to each element, result is list."""

  async def test_basic_map(self) -> None:
    """map applies fn to each element, returns list."""
    await self.variant(
      lambda fn, iterable: Q(iterable).foreach(fn).run(),
      expected=[1, 2, 3, 4, 5],
      fn=V_FN,
      iterable=[('list', list(range(5))), ('async', AsyncRange(5))],
    )

  async def test_map_preserves_order(self) -> None:
    """Results in same order as input."""
    await self.variant(
      lambda fn, iterable: Q(iterable).foreach(fn).run(),
      expected=[0, 2, 4, 6, 8],
      fn=V_DOUBLE,
      iterable=[('list', list(range(5))), ('async', AsyncRange(5))],
    )

  async def test_map_empty_iterable(self) -> None:
    """map over empty list returns empty list."""
    result = Q([]).foreach(sync_fn).run()
    self.assertEqual(result, [])

  async def test_map_async_iterable(self) -> None:
    """map works with async iterables."""
    result = await Q(AsyncRange(4)).foreach(sync_double).run()
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_map_requires_callable(self) -> None:
    """map raises TypeError if fn not callable."""
    with self.assertRaises(TypeError):
      Q([1]).foreach(42)  # type: ignore[arg-type]

  async def test_map_concurrent(self) -> None:
    """map with concurrency processes in parallel."""
    await self.variant(
      lambda fn: Q([1, 2, 3]).foreach(fn, concurrency=2).run(),
      expected=[2, 3, 4],
      fn=V_FN,
    )

  async def test_map_concurrent_preserves_order(self) -> None:
    """Concurrent map preserves input order."""
    await self.variant(
      lambda fn: Q([10, 20, 30]).foreach(fn, concurrency=2).run(),
      expected=[20, 40, 60],
      fn=V_DOUBLE,
    )

  async def test_map_break_no_value(self) -> None:
    """break_() stops iteration, returns partial results."""
    result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_() if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4])

  async def test_map_break_with_value(self) -> None:
    """break_(value) appends to partial results."""
    result = Q([1, 2, 3, 4, 5]).foreach(lambda x: Q.break_(x * 10) if x == 3 else x * 2).run()
    self.assertEqual(result, [2, 4, 30])

  async def test_identity_basic(self) -> None:
    """foreach() without fn collects elements as-is."""
    result = Q([1, 2, 3]).foreach().run()
    self.assertEqual(result, [1, 2, 3])

  async def test_identity_empty(self) -> None:
    """foreach() without fn on empty iterable returns []."""
    result = Q([]).foreach().run()
    self.assertEqual(result, [])

  async def test_identity_async_iterable(self) -> None:
    """foreach() without fn works with async iterables."""
    result = await Q(AsyncRange(4)).foreach().run()
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_identity_preserves_types(self) -> None:
    """foreach() without fn passes elements through unchanged, preserving type."""
    result = Q([(1, 'a'), (2, 'b')]).foreach().run()
    self.assertEqual(result, [(1, 'a'), (2, 'b')])

  async def test_identity_concurrent(self) -> None:
    """foreach() without fn with concurrency collects elements as-is."""
    result = Q([1, 2, 3]).foreach(concurrency=2).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_identity_none_elements(self) -> None:
    """foreach() without fn preserves None elements (not confused with no-fn sentinel)."""
    result = Q([None, None, None]).foreach().run()
    self.assertEqual(result, [None, None, None])

  async def test_identity_preserves_object_identity(self) -> None:
    """foreach() without fn preserves object identity, not just equality.

    SPEC §5.3: "elements are collected as-is" — mutable objects must be
    the exact same objects (assertIs), not copies.
    """
    items = [{'a': 1}, {'b': 2}, {'c': 3}]
    result = Q(items).foreach().run()
    self.assertEqual(len(result), 3)
    for i, item in enumerate(items):
      self.assertIs(result[i], item, f'Element {i} should be the same object, not a copy')


# ---------------------------------------------------------------------------
# §5.4 foreach_do()
# ---------------------------------------------------------------------------


class ForeachDoTests(SymmetricTestCase):
  """§5.4 — foreach_do(fn) collects original elements, discards fn return."""

  async def test_basic_foreach_do(self) -> None:
    """foreach_do returns original elements."""
    await self.variant(
      lambda fn, iterable: Q(iterable).foreach_do(fn).run(),
      expected=[0, 1, 2, 3, 4],
      fn=V_DOUBLE,
      iterable=[('list', list(range(5))), ('async', AsyncRange(5))],
    )

  async def test_foreach_do_fn_executed(self) -> None:
    """foreach_do actually invokes fn for side-effects."""
    calls: list[int] = []

    def track(x: int) -> None:
      calls.append(x)

    Q([1, 2, 3]).foreach_do(track).run()
    self.assertEqual(calls, [1, 2, 3])

  async def test_foreach_do_fn_return_discarded(self) -> None:
    """fn return values are not in the result."""
    result = Q([1, 2, 3]).foreach_do(lambda x: x * 100).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_foreach_do_concurrent(self) -> None:
    """foreach_do with concurrency processes in parallel."""
    await self.variant(
      lambda fn: Q([1, 2, 3]).foreach_do(fn, concurrency=2).run(),
      expected=[1, 2, 3],
      fn=V_DOUBLE,
    )

  async def test_foreach_do_requires_callable(self) -> None:
    """foreach_do raises TypeError if fn not callable."""
    with self.assertRaises(TypeError):
      Q([1]).foreach_do(42)  # type: ignore[arg-type]

  async def test_foreach_do_break_no_value(self) -> None:
    """§5.4: break_() stops iteration, partial originals returned."""

    def sync_maybe_break(x: int) -> None:
      if x == 3:
        raise Q.break_()

    async def async_maybe_break(x: int) -> None:
      if x == 3:
        raise Q.break_()

    await self.variant(
      lambda fn: Q([1, 2, 3, 4, 5]).foreach_do(fn).run(),
      expected=[1, 2],
      fn=[('sync', sync_maybe_break), ('async', async_maybe_break)],
    )

  async def test_foreach_do_break_with_value(self) -> None:
    """§5.4: break_(value) appends to partial originals."""

    def sync_maybe_break(x: int) -> None:
      if x == 3:
        raise Q.break_(x * 10)

    async def async_maybe_break(x: int) -> None:
      if x == 3:
        raise Q.break_(x * 10)

    await self.variant(
      lambda fn: Q([1, 2, 3, 4, 5]).foreach_do(fn).run(),
      expected=[1, 2, 30],
      fn=[('sync', sync_maybe_break), ('async', async_maybe_break)],
    )


# ---------------------------------------------------------------------------
# §5.3/§5.4 Sequential error propagation
# ---------------------------------------------------------------------------


class ForeachErrorPropagationTests(SymmetricTestCase):
  """§5.3/§5.4 — Exceptions propagate immediately, stopping iteration."""

  async def test_foreach_sequential_error_propagation(self) -> None:
    """§5.3: Exception stops iteration at failing element."""

    def sync_fail_on_3(x: int) -> int:
      if x == 3:
        raise ValueError('fail at 3')
      return x * 2

    async def async_fail_on_3(x: int) -> int:
      if x == 3:
        raise ValueError('fail at 3')
      return x * 2

    await self.variant(
      lambda fn: Q([1, 2, 3, 4, 5]).foreach(fn).run(),
      expected_exc=ValueError,
      expected_msg='fail at 3',
      fn=[('sync', sync_fail_on_3), ('async', async_fail_on_3)],
    )

  async def test_foreach_do_sequential_error_propagation(self) -> None:
    """§5.4: Exception stops iteration at failing element."""

    def sync_fail_on_3(x: int) -> None:
      if x == 3:
        raise ValueError('fail at 3')

    async def async_fail_on_3(x: int) -> None:
      if x == 3:
        raise ValueError('fail at 3')

    await self.variant(
      lambda fn: Q([1, 2, 3, 4, 5]).foreach_do(fn).run(),
      expected_exc=ValueError,
      expected_msg='fail at 3',
      fn=[('sync', sync_fail_on_3), ('async', async_fail_on_3)],
    )


# ---------------------------------------------------------------------------
# §5.5 gather()
# ---------------------------------------------------------------------------


class GatherTests(SymmetricTestCase):
  """§5.5 — gather(*fns) runs functions concurrently, returns tuple."""

  async def test_basic_gather(self) -> None:
    """gather returns tuple of results."""
    await self.variant(
      lambda fn: Q(5).gather(fn, fn).run(),
      expected=(6, 6),
      fn=V_FN,
    )

  async def test_gather_zero_fns(self) -> None:
    """Zero fns raises QuentException at build time (§5.5)."""
    with self.assertRaises(QuentException):
      Q(5).gather()  # error at build time, before .run()

  async def test_gather_one_fn(self) -> None:
    """One fn returns single-element tuple."""
    await self.variant(
      lambda fn: Q(5).gather(fn).run(),
      expected=(6,),
      fn=V_FN,
    )

  async def test_gather_preserves_order(self) -> None:
    """Results are in the same positional order as fns."""
    result = (
      Q(5)
      .gather(
        lambda x: x * 2,
        lambda x: x * 3,
        lambda x: x + 1,
      )
      .run()
    )
    # gather is always concurrent, so result is a tuple
    # but we still check result type
    self.assertIsInstance(result, tuple)
    self.assertEqual(result, (10, 15, 6))

  async def test_gather_single_error(self) -> None:
    """Single error propagates directly (not ExceptionGroup)."""
    await self.variant(
      lambda fn: Q(5).gather(fn).run(),
      expected_exc=ValueError,
      fn=V_RAISE,
    )

  async def test_gather_multiple_errors(self) -> None:
    """Multiple errors wrapped in ExceptionGroup (§5.5).

    Per spec: "When multiple functions fail: all regular exceptions
    (Exception subclasses) are wrapped in an ExceptionGroup."
    The first fn is the probe (runs on calling thread). If it fails,
    remaining fns never execute — so to test multiple failures, fn[0]
    must succeed while fn[1] and fn[2] fail.
    """

    def ok(x: Any) -> Any:
      return x

    def err1(x: Any) -> Any:
      raise ValueError('e1')

    def err2(x: Any) -> Any:
      raise ValueError('e2')

    with self.assertRaises(ExceptionGroup) as ctx:
      Q(5).gather(ok, err1, err2).run()
    eg = ctx.exception
    self.assertIn('gather()', str(eg))
    self.assertGreaterEqual(len(eg.exceptions), 2)

  async def test_gather_requires_callable(self) -> None:
    """gather raises TypeError if any fn not callable."""
    with self.assertRaises(TypeError):
      Q(5).gather(42)  # type: ignore[arg-type]

  async def test_gather_always_concurrent(self) -> None:
    """gather is always concurrent — uses ThreadPoolExecutor (sync) or TaskGroup (async)."""
    import threading

    threads: list[str] = []

    def track_thread(x: int) -> int:
      threads.append(threading.current_thread().name)
      return x

    # With 2 fns, sync path uses ThreadPoolExecutor
    result = Q(5).gather(track_thread, track_thread).run()
    self.assertEqual(result, (5, 5))
    # The first fn is probed in the calling thread; remaining go to thread pool
    self.assertTrue(len(threads) >= 2)

  async def test_gather_break_raises_quent_exception(self) -> None:
    """break_() in gather raises QuentException (§5.5)."""
    result = await capture(lambda: Q(5).gather(lambda x: Q.break_()).run())
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, QuentException)
    self.assertIn('break_() signals are not allowed in gather', result.exc_message or '')

  async def test_gather_async_fns(self) -> None:
    """gather with async functions runs concurrently via TaskGroup/asyncio.gather."""

    async def async_triple(x: int) -> int:
      return x * 3

    result = await Q(5).gather(async_double, async_triple).run()
    self.assertIsInstance(result, tuple)
    self.assertEqual(result, (10, 15))

  async def test_gather_async_single_fn(self) -> None:
    """gather with a single async fn returns single-element tuple."""

    async def async_inc(x: int) -> int:
      return x + 1

    result = await Q(5).gather(async_inc).run()
    self.assertEqual(result, (6,))

  async def test_gather_async_error(self) -> None:
    """gather with async fn that raises propagates the error."""

    async def async_err(x: int) -> int:
      raise ValueError('async gather error')

    result = await capture(lambda: Q(5).gather(async_err).run())
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, ValueError)

  async def test_gather_bounded_concurrency(self) -> None:
    """§5.5: gather with concurrency=2 limits simultaneous executions."""
    await self.variant(
      lambda fn: Q(5).gather(fn, fn, fn, concurrency=2).run(),
      expected=(6, 6, 6),
      fn=V_FN,
    )


# ---------------------------------------------------------------------------
# §5.6 with_()
# ---------------------------------------------------------------------------


class WithTests(SymmetricTestCase):
  """§5.6 — with_(fn) enters CM, fn gets context value, result replaces current value."""

  async def test_with_fn_result_replaces_value(self) -> None:
    """fn return value becomes new current value."""
    await self.variant(
      lambda cm, fn: Q(cm(5)).with_(fn).run(),
      expected=15,
      cm=V_CM,
      fn=V_TRIPLE,
    )

  async def test_with_exception_suppression(self) -> None:
    """If CM suppresses exception, pipeline continues with None (§5.6)."""
    await self.variant(
      lambda cm: Q(cm()).with_(lambda x: (_ for _ in ()).throw(ValueError('oops'))).run(),
      expected=None,
      cm=V_CM_SUPPRESSES,
    )

  async def test_with_exception_suppression_continues_pipeline(self) -> None:
    """After suppression, pipeline continues with None as current value (§5.6)."""
    await self.variant(
      lambda cm: (
        Q(cm()).with_(lambda x: (_ for _ in ()).throw(ValueError('oops'))).then(lambda x: (x, 'continued')).run()
      ),
      expected=(None, 'continued'),
      cm=V_CM_SUPPRESSES,
    )

  async def test_with_exception_propagation(self) -> None:
    """If CM does not suppress, exception propagates."""
    await self.variant(
      lambda cm, fn: Q(cm()).with_(fn).run(),
      expected_exc=ValueError,
      cm=V_CM,
      fn=V_RAISE,
    )

  async def test_with_requires_callable(self) -> None:
    """with_ raises TypeError if fn not callable."""
    with self.assertRaises(TypeError):
      Q(SyncCM()).with_(42)  # type: ignore[arg-type]

  async def test_with_not_cm_raises_typeerror(self) -> None:
    """with_ raises TypeError if current value is not a context manager."""
    result = await capture(lambda: Q(42).with_(lambda x: x).run())
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, TypeError)

  async def test_with_control_flow_signal(self) -> None:
    """Control flow signals propagate through with_ (exit called cleanly).
    Per spec §5.6: return_() causes __exit__(None, None, None) and signal propagates."""
    await self.variant(
      lambda cm: Q(10).then(lambda x: cm(x)).with_(lambda x: Q.return_(x * 2)).then(lambda x: x + 999).run(),
      expected=20,
      cm=V_CM,
    )


# ---------------------------------------------------------------------------
# §5.7 with_do()
# ---------------------------------------------------------------------------


class WithDoTests(SymmetricTestCase):
  """§5.7 — with_do(fn) is like with_ but fn result discarded, CM object passes through."""

  async def test_basic_with_do(self) -> None:
    """with_do discards fn result, CM object passes through."""
    # with_do returns the CM object — verify via _value attribute.
    for cm_label, cm_cls in V_CM:
      for fn_label, fn in V_DOUBLE:
        with self.subTest(cm=cm_label, fn=fn_label):
          result = await capture(lambda _cm=cm_cls, _fn=fn: Q(_cm(10)).with_do(_fn).run())
          self.assertTrue(result.success, f'{result.exc_message}')
          self.assertEqual(result.value._value, 10)

  async def test_with_do_exception_suppression_keeps_original(self) -> None:
    """§5.7: If exception suppressed, result is the original CM object (not None)."""
    for cm_label, cm_cls in V_CM_SUPPRESSES:
      for fn_label, fn in V_RAISE:
        with self.subTest(cm=cm_label, fn=fn_label):
          cm_instance = cm_cls()
          result = await capture(lambda _cm=cm_instance, _fn=fn: Q(_cm).with_do(_fn).run())
          self.assertTrue(result.success, f'{result.exc_message}')
          self.assertIs(result.value, cm_instance)

  async def test_with_do_requires_callable(self) -> None:
    """with_do raises TypeError if fn not callable."""
    with self.assertRaises(TypeError):
      Q(SyncCM()).with_do(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §5.6/§5.7 Async context manager tests (coverage for _with_ops.py async paths)
# ---------------------------------------------------------------------------


class AsyncWithTests(IsolatedAsyncioTestCase):
  """Async context manager tests for with_() and with_do()."""

  async def test_with_async_cm_basic(self) -> None:
    """with_ enters async CM, fn receives context value, result replaces current value."""
    result = await Q(AsyncCM(10)).with_(lambda x: x + 1).run()
    self.assertEqual(result, 11)

  async def test_with_async_cm_fn_result(self) -> None:
    """with_ with async CM — fn return replaces current value."""
    result = await Q(AsyncCM(5)).with_(lambda x: x * 3).run()
    self.assertEqual(result, 15)

  async def test_with_async_cm_async_body(self) -> None:
    """with_ with async CM and async body fn."""

    async def async_body(x: int) -> int:
      return x * 2

    result = await Q(AsyncCM(7)).with_(async_body).run()
    self.assertEqual(result, 14)

  async def test_with_async_cm_exception_suppression(self) -> None:
    """with_ with async CM that suppresses exceptions returns None."""
    result = await Q(AsyncCMSuppresses()).with_(lambda x: (_ for _ in ()).throw(ValueError('oops'))).run()
    self.assertIsNone(result)

  async def test_with_async_cm_exception_propagation(self) -> None:
    """with_ with async CM — non-suppressing CM propagates exception."""

    async def raise_fn(x: Any) -> Any:
      raise ValueError('async with error')

    result = await capture(lambda: Q(AsyncCM(5)).with_(raise_fn).run())
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, ValueError)

  async def test_with_async_cm_control_flow(self) -> None:
    """with_ with async CM — control flow signals propagate, __aexit__ called cleanly."""
    result = await Q(AsyncCM(10)).with_(lambda x: Q.return_(x * 2)).then(lambda x: x + 999).run()
    self.assertEqual(result, 20)

  async def test_with_do_async_cm(self) -> None:
    """with_do with async CM — fn result discarded, CM object passes through."""
    cm = AsyncCM(10)
    result = await Q(cm).with_do(lambda x: x * 100).run()
    self.assertIs(result, cm)

  async def test_with_do_async_cm_suppression(self) -> None:
    """with_do with async CM suppressing exception — original value passes through."""

    def raise_in_body(x: Any) -> Any:
      raise ValueError('oops')

    cm = AsyncCMSuppresses()
    result = await Q(cm).with_do(raise_in_body).run()
    self.assertIs(result, cm)

  async def test_with_cm_async_body_transition(self) -> None:
    """with_ with CM and async body — triggers async paths."""

    async def async_body(x: int) -> int:
      return x + 5

    for cm_label, cm_cls in V_CM:
      with self.subTest(cm=cm_label):
        result = await Q(cm_cls(10)).with_(async_body).run()
        self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# §5.8 if_()
# ---------------------------------------------------------------------------


class IfTests(SymmetricTestCase):
  """§5.8 — if_(predicate).then(...) conditional step."""

  async def test_predicate_async(self) -> None:
    """Async predicates are awaited."""

    async def async_pred(x: int) -> bool:
      return x > 0

    result = await Q(5).if_(async_pred).then(lambda x: x * 2).run()
    self.assertEqual(result, 10)

  async def test_predicate_literal(self) -> None:
    """Literal predicate — truthiness used directly."""
    result_truthy = Q(5).if_(True).then(lambda x: x * 2).run()
    self.assertEqual(result_truthy, 10)

    result_falsy = Q(5).if_(False).then(lambda x: x * 2).run()
    self.assertEqual(result_falsy, 5)

    result_int = Q(5).if_(42).then(lambda x: x * 2).run()
    self.assertEqual(result_int, 10)

    result_zero = Q(5).if_(0).then(lambda x: x * 2).run()
    self.assertEqual(result_zero, 5)

  async def test_then_callable_with_args(self) -> None:
    """if_().then(fn, *args): callable called with explicit args."""

    def add(a: int, b: int) -> int:
      return a + b

    result = Q(5).if_(lambda x: True).then(add, 10, 20).run()
    self.assertEqual(result, 30)

  async def test_then_callable_with_kwargs(self) -> None:
    """if_().then(fn, **kwargs): callable called with keyword args."""

    def kw_fn(*, key: int) -> int:
      return key * 2

    result = Q(5).if_(lambda x: True).then(kw_fn, key=7).run()
    self.assertEqual(result, 14)

  async def test_else_with_args(self) -> None:
    """else_() with explicit args: callable called with provided args."""

    def add(a: int, b: int) -> int:
      return a + b

    result = Q(5).if_(lambda x: x > 10).then(lambda x: x).else_(add, 10, 20).run()
    self.assertEqual(result, 30)

  async def test_else_with_kwargs(self) -> None:
    """else_() with explicit kwargs: callable called with keyword args."""

    def kw_fn(*, key: int) -> int:
      return key * 2

    result = Q(5).if_(lambda x: x > 10).then(lambda x: x).else_(kw_fn, key=7).run()
    self.assertEqual(result, 14)

  async def test_nested_pipeline_predicate_propagates_control_flow(self) -> None:
    """Nested pipeline predicates run via the internal execution path,
    so return_() signals propagate through to the outermost pipeline.
    Per spec §7.2.2: return_() in nested pipeline propagates to outermost pipeline."""
    pred_q = Q().then(lambda x: Q.return_('escaped'))
    # The predicate pipeline runs via internal execution, return_() propagates
    # through the outer pipeline, becoming the final result.
    result = Q(5).if_(pred_q).then(lambda x: x * 2).run()
    self.assertEqual(result, 'escaped')

  async def test_then_non_callable_literal(self) -> None:
    """if_().then() accepts non-callable literal value (§5.8)."""
    result = Q(5).if_(lambda x: True).then(42).run()
    self.assertEqual(result, 42)

  async def test_then_non_callable_falsy(self) -> None:
    """if_() with non-callable then — falsy predicate passes through."""
    result = Q(5).if_(lambda x: False).then(42).run()
    self.assertEqual(result, 5)

  async def test_null_predicate_no_value(self) -> None:
    """if_() with None predicate and no current value — predicate is falsy (§5.8)."""
    calls: list[bool] = []

    def track(x: Any) -> Any:
      calls.append(True)
      return x

    result = Q().if_().then(track).run()
    self.assertIsNone(result)
    self.assertEqual(calls, [])  # then branch NOT executed

  async def test_if_async_then_branch(self) -> None:
    """if_() with async then branch."""

    result = await Q(5).if_(lambda x: x > 0).then(async_double).run()
    self.assertEqual(result, 10)

  async def test_if_async_else_branch(self) -> None:
    """if_() with async else branch."""

    async def async_triple(x: int) -> int:
      return x * 3

    result = await Q(5).if_(lambda x: x > 10).then(lambda x: x * 2).else_(async_triple).run()
    self.assertEqual(result, 15)

  async def test_predicate_with_explicit_args(self) -> None:
    """§5.8: if_(check_flag, 'feature_x') — explicit args suppress current value."""

    def sync_check_flag(flag: str) -> bool:
      return flag == 'feature_x'

    async def async_check_flag(flag: str) -> bool:
      return flag == 'feature_x'

    await self.variant(
      lambda pred: Q(5).if_(pred, 'feature_x').then(lambda x: x * 2).run(),
      expected=10,
      pred=[('sync', sync_check_flag), ('async', async_check_flag)],
    )

  async def test_predicate_with_explicit_args_falsy(self) -> None:
    """§5.8: if_(check_flag, 'other') — predicate with args returns falsy, passthrough."""

    def sync_check_flag(flag: str) -> bool:
      return flag == 'feature_x'

    async def async_check_flag(flag: str) -> bool:
      return flag == 'feature_x'

    await self.variant(
      lambda pred: Q(5).if_(pred, 'other').then(lambda x: x * 2).run(),
      expected=5,
      pred=[('sync', sync_check_flag), ('async', async_check_flag)],
    )

  async def test_pending_if_run_raises(self) -> None:
    """run() with pending if_() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: x > 0).run()

  async def test_do_as_conditional_branch(self) -> None:
    """if_().do(fn) creates a side-effect conditional branch."""
    calls: list[int] = []
    result = Q(5).if_(lambda x: x > 0).do(lambda x: calls.append(x)).run()
    self.assertEqual(result, 5)  # value passes through (do discards result)
    self.assertEqual(calls, [5])


# ---------------------------------------------------------------------------
# §5.9 else_()
# ---------------------------------------------------------------------------


class ElseTests(SymmetricTestCase):
  """§5.9 — else_() registers alternative branch for preceding if_()."""

  async def test_else_not_evaluated_when_truthy(self) -> None:
    """else branch not run when predicate truthy."""
    result = Q(5).if_(lambda x: x > 0).then(lambda x: x * 2).else_(lambda x: x * 100).run()
    self.assertEqual(result, 10)

  async def test_else_must_follow_if(self) -> None:
    """else_() without preceding if_() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).then(lambda x: x).else_(lambda x: x)

  async def test_else_on_empty_pipeline(self) -> None:
    """else_() on empty pipeline raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).else_(lambda x: x)

  async def test_only_one_else_per_if(self) -> None:
    """Second else_() on same if_() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: True).then(lambda x: x).else_(lambda x: x).else_(lambda x: x)

  async def test_else_with_non_callable(self) -> None:
    """else_ with non-callable value."""
    result = Q(5).if_(lambda x: x > 10).then(lambda x: x * 2).else_(99).run()
    self.assertEqual(result, 99)

  async def test_else_non_callable_with_args_raises(self) -> None:
    """else_(non_callable, arg) raises TypeError (§5.9: args require callable)."""
    with self.assertRaises(TypeError) as ctx:
      Q(5).if_(lambda x: x > 10).then(lambda x: x).else_(99, 'extra_arg')
    self.assertIn('not callable', str(ctx.exception))

  async def test_else_on_pending_if_raises(self) -> None:
    """else_() while if_() is still pending (no then/do yet) raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: x > 0).else_(lambda x: x)


# ---------------------------------------------------------------------------
# §5.9.1 else_do()
# ---------------------------------------------------------------------------


class ElseDoTests(SymmetricTestCase):
  """§5.9.1 — else_do() side-effect else branch."""

  async def test_else_do_basic_side_effect(self) -> None:
    """else_do() runs side-effect and passes current value through."""
    calls: list[Any] = []
    result = Q(-5).if_(lambda x: x > 0).then(str).else_do(calls.append).run()
    self.assertEqual(result, -5)
    self.assertEqual(calls, [-5])

  async def test_else_do_not_called_when_truthy(self) -> None:
    """else_do() fn is not invoked when predicate is truthy."""
    calls: list[Any] = []
    result = Q(5).if_(lambda x: x > 0).then(str).else_do(calls.append).run()
    self.assertEqual(result, '5')
    self.assertEqual(calls, [])

  async def test_else_do_async_side_effect(self) -> None:
    """else_do() works with async side-effect fn; value still passes through."""
    calls: list[Any] = []

    async def async_append(x: Any) -> None:
      calls.append(x)

    result = await Q(-5).if_(lambda x: x > 0).then(str).else_do(async_append).run()
    self.assertEqual(result, -5)
    self.assertEqual(calls, [-5])

  async def test_else_do_with_args(self) -> None:
    """else_do() with explicit args: fn called with provided args, result discarded."""
    calls: list[Any] = []

    def record(label: str) -> str:
      calls.append(label)
      return 'discarded'

    result = Q(-5).if_(lambda x: x > 0).then(str).else_do(record, 'neg').run()
    self.assertEqual(result, -5)
    self.assertEqual(calls, ['neg'])  # explicit args: fn('neg'), current_value not passed

  async def test_else_do_non_callable_raises(self) -> None:
    """else_do() with non-callable raises TypeError."""
    with self.assertRaises(TypeError) as ctx:
      Q(5).if_(lambda x: x > 0).then(str).else_do(42)  # type: ignore[arg-type]
    self.assertIn('else_do', str(ctx.exception))

  async def test_else_do_requires_preceding_if(self) -> None:
    """else_do() without preceding if_() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).then(lambda x: x).else_do(lambda x: x)

  async def test_else_do_on_empty_pipeline_raises(self) -> None:
    """else_do() on empty pipeline raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).else_do(lambda x: x)

  async def test_else_do_on_pending_if_raises(self) -> None:
    """else_do() while if_() is still pending raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: x > 0).else_do(lambda x: x)

  async def test_else_do_duplicate_raises(self) -> None:
    """Second else branch on same if_() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: False).then(str).else_do(lambda x: x).else_do(lambda x: x)

  async def test_else_do_then_else_do_duplicate_raises(self) -> None:
    """else_() followed by else_do() on same if_() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: False).then(str).else_(abs).else_do(lambda x: x)

  async def test_else_do_sync_pred_passthrough(self) -> None:
    """else_do() with sync predicate: current value passes through."""
    calls: list[Any] = []
    result = Q(5).if_(lambda x: x > 10).then(lambda x: x * 2).else_do(calls.append).run()
    self.assertEqual(result, 5)
    self.assertEqual(calls, [5])

  async def test_else_do_async_pred_passthrough(self) -> None:
    """else_do() with async predicate: current value still passes through."""
    calls: list[Any] = []

    async def async_gt10(x: Any) -> bool:
      return x > 10

    result = await Q(5).if_(async_gt10).then(lambda x: x * 2).else_do(calls.append).run()
    self.assertEqual(result, 5)
    self.assertEqual(calls, [5])


class IfPredicateErrorTest(unittest.TestCase):
  """Test error propagation when if_() predicate raises."""

  def test_if_predicate_raises_exception(self) -> None:
    """Exception from predicate evaluation propagates normally."""

    def bad_predicate(x: int) -> bool:
      raise ValueError('predicate failed')

    with self.assertRaises(ValueError) as ctx:
      Q(5).if_(bad_predicate).then(lambda x: x * 2).run()
    self.assertIn('predicate failed', str(ctx.exception))

  def test_if_predicate_raises_base_exception(self) -> None:
    """BaseException from predicate evaluation propagates normally."""

    class CustomBaseErr(BaseException):
      pass

    def bad_predicate(x: int) -> bool:
      raise CustomBaseErr('stop')

    with self.assertRaises(CustomBaseErr):
      Q(5).if_(bad_predicate).then(lambda x: x * 2).run()


class IfValidationTest(unittest.TestCase):
  """Test if_() builder-time validation."""

  def test_if_none_with_positional_args(self) -> None:
    """if_(None, arg) raises QuentException."""
    with self.assertRaises(QuentException) as ctx:
      Q(5).if_(None, 1, 2)
    self.assertIn('args/kwargs but no predicate', str(ctx.exception))

  def test_if_none_with_kwargs(self) -> None:
    """if_(None, key=val) raises QuentException."""
    with self.assertRaises(QuentException) as ctx:
      Q(5).if_(None, key=42)
    self.assertIn('args/kwargs but no predicate', str(ctx.exception))


class IfAsyncSideEffectTest(IsolatedAsyncioTestCase):
  """Test async side-effect branches in if_().do()."""

  async def test_if_do_async_truthy_branch(self) -> None:
    """if_().do(async_fn) awaits the side-effect, passes through current_value."""
    side_effects: list[int] = []

    async def async_effect(x: int) -> int:
      side_effects.append(x)
      return x * 100  # result discarded by .do()

    result = await Q(5).if_(lambda x: x > 0).do(async_effect).run()
    self.assertEqual(result, 5)
    self.assertEqual(side_effects, [5])

  async def test_if_do_async_with_async_predicate(self) -> None:
    """if_(async_pred).do(async_fn) awaits both predicate and side-effect."""
    side_effects: list[int] = []

    async def async_pred(x: int) -> bool:
      return x > 0

    async def async_effect(x: int) -> int:
      side_effects.append(x)
      return x * 100

    result = await Q(5).if_(async_pred).do(async_effect).run()
    self.assertEqual(result, 5)
    self.assertEqual(side_effects, [5])


# ---------------------------------------------------------------------------
# §5.8 if_() predicate control flow signals
# ---------------------------------------------------------------------------


class IfPredicateReturnPropagationTest(SymmetricTestCase):
  """§5.8: return_() in if_() predicate pipeline propagates to outer pipeline."""

  async def test_sync_return_in_predicate_pipeline_propagates(self) -> None:
    """Sync: return_() in predicate pipeline exits outer pipeline with the return value."""
    pred_q = Q().then(lambda x: Q.return_('escaped'))
    result = Q(5).if_(pred_q).then(lambda x: x * 2).run()
    self.assertEqual(result, 'escaped')

  async def test_async_return_in_predicate_pipeline_propagates(self) -> None:
    """Async: return_() in async predicate pipeline exits outer pipeline."""

    async def async_step(x):
      return Q.return_('async_escaped')

    pred_q = Q().then(async_step)
    result = await Q(5).if_(pred_q).then(lambda x: x * 2).run()
    self.assertEqual(result, 'async_escaped')

  async def test_return_in_predicate_skips_then_and_else(self) -> None:
    """return_() in predicate bypasses both then and else branches."""
    then_called = []
    else_called = []
    pred_q = Q().then(lambda x: Q.return_('early'))
    result = (
      Q(5)
      .if_(pred_q)
      .then(lambda x: then_called.append(True) or x)
      .else_(lambda x: else_called.append(True) or x)
      .run()
    )
    self.assertEqual(result, 'early')
    self.assertEqual(then_called, [])
    self.assertEqual(else_called, [])


class IfPredicateBreakTrappedTest(SymmetricTestCase):
  """§5.8: break_() in if_() predicate pipeline raises QuentException."""

  async def test_sync_break_in_predicate_pipeline_raises(self) -> None:
    """Sync: break_() in predicate pipeline raises QuentException."""
    pred_q = Q().then(lambda x: Q.break_())
    with self.assertRaises(QuentException) as ctx:
      Q(5).if_(pred_q).then(lambda x: x * 2).run()
    self.assertIn('break_() cannot be used inside an if_() predicate', str(ctx.exception))

  async def test_async_break_in_predicate_pipeline_raises(self) -> None:
    """Async: break_() in async predicate chain raises QuentException."""

    async def async_step(x):
      return Q.break_()

    pred_q = Q().then(async_step)
    with self.assertRaises(QuentException) as ctx:
      await Q(5).if_(pred_q).then(lambda x: x * 2).run()
    self.assertIn('break_() cannot be used inside an if_() predicate', str(ctx.exception))

  async def test_break_with_value_in_predicate_pipeline_raises(self) -> None:
    """break_(value) in predicate pipeline also raises QuentException."""
    pred_q = Q().then(lambda x: Q.break_('some_value'))
    with self.assertRaises(QuentException) as ctx:
      Q(5).if_(pred_q).then(lambda x: x * 2).run()
    self.assertIn('break_() cannot be used inside an if_() predicate', str(ctx.exception))

  async def test_break_in_deep_nested_predicate_raises(self) -> None:
    """break_() in deeply nested predicate pipeline raises QuentException."""
    inner = Q().then(lambda x: Q.break_())
    pred_q = Q().then(inner)
    with self.assertRaises(QuentException) as ctx:
      Q(5).if_(pred_q).then(lambda x: x * 2).run()
    self.assertIn('break_() cannot be used inside an if_() predicate', str(ctx.exception))

  async def test_break_cause_preserved(self) -> None:
    """The original _Break is preserved as __cause__ of the QuentException."""
    pred_q = Q().then(lambda x: Q.break_())
    try:
      Q(5).if_(pred_q).then(lambda x: x * 2).run()
      self.fail('Expected QuentException')
    except QuentException as exc:
      # The __cause__ should be the original _Break signal
      self.assertIsNotNone(exc.__cause__)
      from quent._types import _Break

      self.assertIsInstance(exc.__cause__, _Break)


# ---------------------------------------------------------------------------
# Async iteration coverage tests
# ---------------------------------------------------------------------------


class AsyncIterationTests(IsolatedAsyncioTestCase):
  """Tests for async iteration paths in _iter_ops.py."""

  async def test_map_async_iterable_with_async_fn(self) -> None:
    """map with async iterable and async fn — full async path."""

    result = await Q(AsyncRange(4)).foreach(async_double).run()
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_foreach_do_async_iterable(self) -> None:
    """foreach_do with async iterable — full async path."""
    calls: list[int] = []

    def track(x: int) -> None:
      calls.append(x)

    result = await Q(AsyncRange(3)).foreach_do(track).run()
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(calls, [0, 1, 2])

  async def test_map_sync_iterable_async_fn_transition(self) -> None:
    """map with sync iterable but async fn — mid-operation async transition."""

    async def async_inc(x: int) -> int:
      return x + 1

    result = await Q([1, 2, 3]).foreach(async_inc).run()
    self.assertEqual(result, [2, 3, 4])

  async def test_foreach_do_sync_iterable_async_fn_transition(self) -> None:
    """foreach_do with sync iterable but async fn — mid-operation async transition."""
    calls: list[int] = []

    async def async_track(x: int) -> None:
      calls.append(x)

    result = await Q([10, 20, 30]).foreach_do(async_track).run()
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(calls, [10, 20, 30])

  async def test_concurrent_map_async_fn(self) -> None:
    """map with concurrency and async fn — async concurrent path."""

    result = await Q([1, 2, 3]).foreach(async_double, concurrency=2).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_concurrent_foreach_do_async_fn(self) -> None:
    """foreach_do with concurrency and async fn — async concurrent path."""
    calls: list[int] = []

    async def async_track(x: int) -> None:
      calls.append(x)

    result = await Q([1, 2, 3]).foreach_do(async_track, concurrency=2).run()
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(sorted(calls), [1, 2, 3])

  async def test_concurrent_map_async_iterable(self) -> None:
    """map with concurrency and async iterable — _from_aiter path."""

    result = await Q(AsyncRange(4)).foreach(async_double, concurrency=2).run()
    self.assertEqual(result, [0, 2, 4, 6])

  async def test_map_async_iterable_break(self) -> None:
    """map with async iterable and break_() — full async break path."""
    result = await Q(AsyncRange(10)).foreach(lambda x: Q.break_() if x == 3 else x * 2).run()
    self.assertEqual(result, [0, 2, 4])

  async def test_map_async_iterable_break_with_value(self) -> None:
    """map with async iterable and break_(value) — appends to partial results."""
    result = await Q(AsyncRange(10)).foreach(lambda x: Q.break_(x * 10) if x == 3 else x * 2).run()
    self.assertEqual(result, [0, 2, 4, 30])


# ---------------------------------------------------------------------------
# Concurrency validation tests
# ---------------------------------------------------------------------------


class ConcurrencyValidationTests(IsolatedAsyncioTestCase):
  """Test concurrency parameter validation for foreach/foreach_do/gather."""

  def test_concurrency_boolean_rejected(self) -> None:
    """Booleans are not valid concurrency values."""
    with self.assertRaises(TypeError):
      Q([1]).foreach(sync_fn, concurrency=True)  # type: ignore[arg-type]

  def test_concurrency_zero_rejected(self) -> None:
    """concurrency < 1 raises ValueError."""
    with self.assertRaises(ValueError):
      Q([1]).foreach(sync_fn, concurrency=0)

  def test_concurrency_minus_one_accepted(self) -> None:
    """concurrency=-1 is valid (unbounded) — processes all items."""
    result = Q([1, 2, 3]).foreach(sync_fn, concurrency=-1).run()
    self.assertEqual(result, [2, 3, 4])

  def test_concurrency_negative_other_rejected(self) -> None:
    """concurrency=-2 and other negatives (not -1) raise ValueError."""
    with self.assertRaises(ValueError):
      Q([1]).foreach(sync_fn, concurrency=-2)

  def test_concurrency_float_rejected(self) -> None:
    """Float concurrency raises TypeError."""
    with self.assertRaises(TypeError):
      Q([1]).foreach(sync_fn, concurrency=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §5.6/§5.7 Async edge cases for with_() and with_do()
# ---------------------------------------------------------------------------


class WithOpsAsyncEdgeTests(IsolatedAsyncioTestCase):
  """§5.6/§5.7: Async edge cases for with_() and with_do() operations."""

  async def test_to_async_control_flow_signal(self) -> None:
    """§5.6: Sync CM + async body + return_() — __exit__ called clean."""
    exit_args: list[tuple[Any, ...]] = []

    class TrackingCM:
      def __enter__(self) -> int:
        return 42

      def __exit__(self, *args: Any) -> bool:
        exit_args.append(args)
        return False

    async def async_body(x: Any) -> Any:
      return Q.return_(x * 2)

    result = await Q(TrackingCM()).with_(async_body).run()
    self.assertEqual(result, 84)
    # __exit__ called with clean exit args (signal path)
    self.assertEqual(len(exit_args), 1)
    self.assertEqual(exit_args[0], (None, None, None))

  async def test_full_async_aenter_failure(self) -> None:
    """§5.6: Async CM __aenter__ raises."""

    class FailingAenterCM:
      async def __aenter__(self) -> Any:
        raise RuntimeError('aenter boom')

      async def __aexit__(self, *args: Any) -> bool:
        return False

    result = await capture(lambda: Q(FailingAenterCM()).with_(lambda x: x).run())
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, RuntimeError)
    self.assertIn('aenter boom', result.exc_message or '')

  async def test_full_async_signal_with_aexit_failure(self) -> None:
    """§5.6: Async CM + return_() + __aexit__ raises."""

    class AexitFailsCM:
      async def __aenter__(self) -> int:
        return 10

      async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        raise RuntimeError('aexit failed')

    result = await capture(lambda: Q(AexitFailsCM()).with_(lambda x: Q.return_(x * 2)).run())
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, RuntimeError)
    self.assertIn('aexit failed', result.exc_message or '')

  async def test_full_async_exception_with_aexit_failure(self) -> None:
    """§5.6: Async CM + body raises + __aexit__ raises."""

    class AexitFailsOnExcCM:
      async def __aenter__(self) -> int:
        return 10

      async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        raise RuntimeError('aexit error')

    result = await capture(
      lambda: Q(AexitFailsOnExcCM()).with_(lambda x: (_ for _ in ()).throw(ValueError('body error'))).run()
    )
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, RuntimeError)
    self.assertIn('aexit error', result.exc_message or '')

  async def test_full_async_with_unset_result(self) -> None:
    """§5.7: with_do + async CM suppression preserves original."""

    class AsyncSuppressingCM:
      async def __aenter__(self) -> int:
        return 10

      async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return True  # suppress

    result = await Q(AsyncSuppressingCM()).with_do(lambda x: (_ for _ in ()).throw(ValueError('suppressed'))).run()
    # with_do returns the original CM object
    self.assertIsInstance(result, AsyncSuppressingCM)

  async def test_await_exit_success_ignore_result(self) -> None:
    """§5.7: with_do + sync CM + async __exit__ success — original passes."""

    class CMWithAsyncExitSuccess:
      def __enter__(self) -> int:
        return 10

      def __exit__(self, *args: Any) -> Any:
        async def _exit() -> bool:
          return False

        return _exit()

    cm = CMWithAsyncExitSuccess()
    result = await Q(cm).with_do(lambda x: x * 100).run()
    # with_do returns the original CM object
    self.assertIs(result, cm)

  async def test_dual_protocol_prefers_async_when_loop_running(self) -> None:
    """§16.10: Dual-protocol CM prefers async when loop running."""

    class DualProtocolCM:
      def __init__(self) -> None:
        self.protocol_used: str = ''

      def __enter__(self) -> int:
        self.protocol_used = 'sync'
        return 10

      def __exit__(self, *args: Any) -> bool:
        return False

      async def __aenter__(self) -> int:
        self.protocol_used = 'async'
        return 20

      async def __aexit__(self, *args: Any) -> bool:
        return False

    cm = DualProtocolCM()
    result = await Q(cm).with_(lambda x: x + 1).run()
    # async protocol should be used, so context value is 20
    self.assertEqual(result, 21)
    self.assertEqual(cm.protocol_used, 'async')

  def test_dual_protocol_no_event_loop_uses_sync(self) -> None:
    """§16.10: Dual-protocol CM falls back to sync when no loop."""

    class DualProtocolCMSync:
      def __init__(self) -> None:
        self.protocol_used: str = ''

      def __enter__(self) -> int:
        self.protocol_used = 'sync'
        return 10

      def __exit__(self, *args: Any) -> bool:
        return False

      async def __aenter__(self) -> int:
        self.protocol_used = 'async'
        return 20

      async def __aexit__(self, *args: Any) -> bool:
        return False

    cm = DualProtocolCMSync()
    result = Q(cm).with_(lambda x: x + 1).run()
    self.assertEqual(result, 11)
    self.assertEqual(cm.protocol_used, 'sync')

  async def test_to_async_control_flow_signal_with_exit_raises(self) -> None:
    """§5.6: Sync CM + async body + return_() + __exit__ raises."""
    exit_raised = False

    class CMExitRaises:
      def __enter__(self) -> int:
        return 42

      def __exit__(self, *args: Any) -> bool:
        nonlocal exit_raised
        exit_raised = True
        raise RuntimeError('exit raises on signal')

    async def async_body(x: Any) -> Any:
      return Q.return_(x * 2)

    result = await capture(lambda: Q(CMExitRaises()).with_(async_body).run())
    self.assertFalse(result.success)
    self.assertEqual(result.exc_type, RuntimeError)
    self.assertIn('exit raises on signal', result.exc_message or '')
    self.assertTrue(exit_raised)

  async def test_to_async_control_flow_signal_with_async_exit(self) -> None:
    """§5.6: Sync CM + async body + return_() + async __exit__ awaited."""
    exit_awaited = False

    class CMAsyncExitOnSignal:
      def __enter__(self) -> int:
        return 42

      def __exit__(self, *args: Any) -> Any:
        async def _exit() -> bool:
          nonlocal exit_awaited
          exit_awaited = True
          return False

        return _exit()

    async def async_body(x: Any) -> Any:
      return Q.return_(x * 2)

    result = await Q(CMAsyncExitOnSignal()).with_(async_body).run()
    self.assertEqual(result, 84)
    self.assertTrue(exit_awaited)

  async def test_to_async_exception_with_async_exit_suppress(self) -> None:
    """§5.6: Sync CM + async __exit__ suppression during exception."""

    class CMWithAsyncExitSuppresses:
      def __enter__(self) -> int:
        return 10

      def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if exc_type is not None:

          async def _exit() -> bool:
            return True  # suppress

          return _exit()
        return False

    async def async_body(x: Any) -> Any:
      raise ValueError('should be suppressed')

    result = await Q(CMWithAsyncExitSuppresses()).with_(async_body).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# GAP 2: Missing brick fast tests — do(nested_pipeline)
# ---------------------------------------------------------------------------


class DoNestedChainTest(SymmetricTestCase):
  """do(Q) — nested pipeline as side-effect, result discarded."""

  async def test_do_with_nested_pipeline(self) -> None:
    """do(Q().then(fn)) — nested pipeline as side-effect, result discarded."""
    side_effects: list[Any] = []
    inner = Q().then(lambda x: side_effects.append(x))
    result = Q(5).do(inner).run()
    self.assertEqual(result, 5)  # do discards
    self.assertEqual(side_effects, [5])  # inner pipeline ran with cv=5


# ---------------------------------------------------------------------------
# GAP 2: Missing brick fast tests — with_ explicit args, nested pipeline
# ---------------------------------------------------------------------------


class WithExplicitArgsTest(SymmetricTestCase):
  """with_(fn, explicit_args) — explicit args suppress CM value."""

  async def test_with_explicit_args(self) -> None:
    """with_(fn, 42) — explicit args suppress CM value."""
    await self.variant(
      lambda fn: Q(5).then(lambda x: SyncCM(x)).with_(fn, 42).run(),
      fn=V_FN,
      expected=43,  # fn(42)=43, CM value (5) NOT passed
    )

  async def test_with_nested_pipeline_body(self) -> None:
    """with_(Q().then(fn)) — nested pipeline as with_ body."""
    await self.variant(
      lambda fn: Q(5).then(lambda x: SyncCM(x)).with_(Q().then(fn)).run(),
      fn=V_FN,
      expected=6,  # SyncCM(5).__enter__=5, inner pipeline fn(5)=6
    )


# ---------------------------------------------------------------------------
# GAP 2: Missing brick fast tests — with_do explicit args, nested pipeline
# ---------------------------------------------------------------------------


class WithDoExplicitArgsTest(SymmetricTestCase):
  """with_do(fn, explicit_args) and with_do(nested_chain)."""

  async def test_with_do_explicit_args(self) -> None:
    """with_do(fn, 42) — explicit args, result discarded."""
    received: list[Any] = []

    def track(a: Any) -> None:
      received.append(a)

    cm = SyncCM(5)
    result = Q(cm).with_do(track, 42).run()
    self.assertIs(result, cm)  # with_do passes CM through
    self.assertEqual(received, [42])  # track received explicit arg, not CM value

  async def test_with_do_nested_pipeline_body(self) -> None:
    """with_do(Q().then(fn)) — nested pipeline as with_do body, result discarded."""
    cm = SyncCM(99)
    result = Q(cm).with_do(Q().then(lambda x: x * 2)).run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# GAP 2: Missing brick fast test — else_(nested_pipeline)
# ---------------------------------------------------------------------------


class ElseNestedChainTest(SymmetricTestCase):
  """else_(Q) — nested pipeline as else branch."""

  async def test_else_with_nested_pipeline(self) -> None:
    """else_(Q().then(fn)) — nested pipeline as else branch."""
    await self.variant(
      lambda fn: Q(5).if_(lambda x: False).then(lambda x: -999).else_(Q().then(fn)).run(),
      fn=V_FN,
      expected=6,  # predicate false, else branch: inner pipeline fn(5)=6
    )


# ---------------------------------------------------------------------------
# GAP 1: Multi-operation composition smoke tests
# ---------------------------------------------------------------------------


class MultiOperationCompositionTest(SymmetricTestCase):
  """Fast regression tests for multi-operation composition.

  The exhaustive bridge tests all permutations but takes ~500s.
  These smoke tests catch composition regressions in <1s.
  """

  async def test_then_foreach_then(self) -> None:
    """then -> foreach -> then: value threading across 3 op types."""
    await self.variant(
      lambda fn: Q(5).then(fn).then(lambda x: [x, x + 1]).foreach(fn).then(sum).run(),
      fn=V_FN,
      expected=15,  # fn(5)=6, [6,7], foreach fn: [7,8], sum=15
    )

  async def test_then_if_else(self) -> None:
    """then -> if_ -> else_: conditional after transform."""
    await self.variant(
      lambda fn: Q(5).then(fn).if_(lambda x: x > 10).then(lambda x: x * 2).else_(fn).run(),
      fn=V_FN,
      expected=7,  # fn(5)=6, 6>10 false, else fn(6)=7
    )

  async def test_then_with_then(self) -> None:
    """then -> with_ -> then: context manager in the middle."""
    await self.variant(
      lambda fn: Q(5).then(fn).then(lambda x: SyncCM(x)).with_(fn).then(fn).run(),
      fn=V_FN,
      expected=8,  # fn(5)=6, SyncCM(6).__enter__=6, with_ fn(6)=7, then fn(7)=8
    )

  async def test_then_gather_then(self) -> None:
    """then -> gather -> then: gather in the middle."""
    await self.variant(
      lambda fn: Q(5).then(fn).gather(fn, fn).then(lambda t: t[0] + t[1]).run(),
      fn=V_FN,
      expected=14,  # fn(5)=6, gather(fn(6), fn(6))=(7,7), 7+7=14
    )

  async def test_foreach_do_gather_then(self) -> None:
    """foreach_do -> then -> gather: three different ops."""
    await self.variant(
      lambda fn: Q([1, 2]).foreach_do(fn).then(sum).gather(fn, fn).then(lambda t: t[0] + t[1]).run(),
      fn=V_FN,
      expected=8,  # foreach_do keeps [1,2], sum=3, gather(fn(3),fn(3))=(4,4), 4+4=8
    )


# ---------------------------------------------------------------------------
# Audit §9 — Additional spec gap tests
# ---------------------------------------------------------------------------


class AsyncAexitFailureContextTest(IsolatedAsyncioTestCase):
  """SPEC §5.6: Async __aexit__ failure preserves body exception as __context__."""

  async def test_async_aexit_failure_preserves_context(self) -> None:
    """When async __aexit__ raises, __context__ is set to the original body exception."""

    class AexitFailsCM:
      async def __aenter__(self) -> int:
        return 10

      async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        raise RuntimeError('aexit boom')

    c = Q(AexitFailsCM()).with_(lambda x: (_ for _ in ()).throw(ValueError('body error')))
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIn('aexit boom', str(ctx.exception))
    self.assertIsInstance(ctx.exception.__context__, ValueError)


class AsyncGatherSingleFnTupleTest(IsolatedAsyncioTestCase):
  """SPEC §5.5: Single-fn gather returns single-element tuple in async path."""

  async def test_async_gather_single_fn_returns_tuple(self) -> None:
    """Async gather(fn) returns (result,) tuple."""

    async def async_fn(x):
      return x + 1

    result = await Q(5).gather(async_fn).run()
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 1)
    self.assertEqual(result, (6,))


class WithDoNestedChainReturnDiscardedTest(IsolatedAsyncioTestCase):
  """SPEC §5.7: with_do fn return value is discarded — explicit nested pipeline test."""

  async def test_with_do_nested_pipeline_return_discarded(self) -> None:
    """with_do(Q().then(fn)) discards nested pipeline result, CM object passes through."""
    cm = SyncCM(42)
    result = Q(cm).with_do(Q().then(lambda x: 'should_be_discarded')).run()
    self.assertIs(result, cm)


class ReturnInsideWithCleanExitTest(IsolatedAsyncioTestCase):
  """SPEC §5.6: return_() inside with_() calls __exit__ cleanly, signal propagates."""

  async def test_return_inside_with_clean_exit_tracking(self) -> None:
    """return_() in with_ body → __exit__ called with no exception info, signal propagates."""
    exit_args: list[Any] = []

    class TrackingCM:
      def __enter__(self) -> int:
        return 10

      def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        exit_args.append((exc_type, exc_val, exc_tb))
        return False

    result = Q(TrackingCM()).with_(lambda v: Q.return_('early')).then(lambda x: 'never').run()
    self.assertEqual(result, 'early')
    # __exit__ was called with no exception info (clean exit)
    self.assertEqual(len(exit_args), 1)
    self.assertEqual(exit_args[0], (None, None, None))


if __name__ == '__main__':
  unittest.main()
