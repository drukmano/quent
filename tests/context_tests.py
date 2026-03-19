# SPDX-License-Identifier: MIT
"""Tests for SPEC §15 — Context API.

Covers:
- §15.1: set(key) / set(key, value) instance method (pipeline step, preserves current value)
- §15.2: Q.set(key, value) class-level call (immediate store)
- §15.3: q.get(key) / Q.get(key) dual dispatch (descriptor pipeline step / immediate retrieval)
- §15.4: Storage scoping (copy-on-write, concurrent isolation, persistence)
- §15.5: Dual dispatch via descriptor (both set and get)
- Operations between set and get (context survives through pipeline ops)
- Conditional set via if_()
- q.get('key') as pipeline step (descriptor, replaces CV)
- q.set('key', value) with explicit value (instance method, preserves CV)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Q, QuentException
from quent._context import _ctx_store
from tests.symmetric import SymmetricTestCase


def _reset_context() -> None:
  """Reset the quent context store to a clean state.

  Uses the ContextVar Token mechanism to restore the variable
  to its initial (unset) state for test isolation.
  """
  try:
    _ctx_store.get()
    # ContextVar has no public 'delete' -- set to empty dict
    # so subsequent get() calls don't find stale keys.
    _ctx_store.set({})
  except LookupError:
    pass


# ---------------------------------------------------------------------------
# §15.1 set(key) — Instance Method (Pipeline Step)
# ---------------------------------------------------------------------------


class SetGetBasicTest(TestCase):
  """§15.1/§15.3: Basic set/get round-trip."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_set_get_basic(self) -> None:
    """Q(42).set('k').then(Q.get, 'k').run() == 42.

    §15.1: set stores current value under key.
    §15.3: Q.get retrieves it via explicit args (Rule 1).
    """
    result = Q(42).set('k').then(Q.get, 'k').run()
    self.assertEqual(result, 42)

  def test_set_preserves_cv(self) -> None:
    """§15.1: .set() does NOT change the current value — like .do().

    The pipeline value flows through unchanged after set.
    """
    result = Q(42).set('k').then(lambda x: x + 1).run()
    self.assertEqual(result, 43)

  def test_multiple_keys(self) -> None:
    """§15.1: Multiple keys can be stored and retrieved independently."""
    result = Q(10).set('a').then(lambda x: x * 2).set('b').then(lambda x: (Q.get('a'), Q.get('b'), x)).run()
    self.assertEqual(result, (10, 20, 20))

  def test_key_overwrite(self) -> None:
    """§15.4: Second .set() with same key overwrites the first value.

    Copy-on-write creates a new dict with the updated key.
    """
    result = Q(10).set('k').then(lambda x: x * 3).set('k').then(Q.get, 'k').run()
    self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# §15.2 Q.set(key, value) — Class-Level Call
# ---------------------------------------------------------------------------


class StaticSetTest(TestCase):
  """§15.2: Q.set(key, value) stores value immediately (not a pipeline step)."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_static_set(self) -> None:
    """§15.2: Q.set('k', 99) stores value, Q.get('k') retrieves it."""
    Q.set('k', 99)
    self.assertEqual(Q.get('k'), 99)

  def test_static_set_returns_none(self) -> None:
    """§15.2: Q.set(key, value) returns None."""
    result = Q.set('k', 42)
    self.assertIsNone(result)

  def test_static_set_used_in_pipeline(self) -> None:
    """§15.2: Pre-populated context is visible inside a pipeline."""
    Q.set('config', 'prod')
    result = Q(1).then(lambda x: Q.get('config')).run()
    self.assertEqual(result, 'prod')


# ---------------------------------------------------------------------------
# §15.3 Q.get(key, default) — Static Method
# ---------------------------------------------------------------------------


class GetTest(TestCase):
  """§15.3: Q.get retrieval semantics."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_get_missing_key_raises(self) -> None:
    """§15.3: Q.get('nonexistent') raises KeyError when key not found."""
    with self.assertRaises(KeyError):
      Q.get('nonexistent')

  def test_get_with_default(self) -> None:
    """§15.3: Q.get('nonexistent', 'fallback') returns 'fallback'."""
    result = Q.get('nonexistent', 'fallback')
    self.assertEqual(result, 'fallback')

  def test_get_default_none(self) -> None:
    """§15.3: Q.get('nonexistent', None) returns None (not KeyError).

    Providing None as the default is distinct from not providing a default.
    """
    result = Q.get('nonexistent', None)
    self.assertIsNone(result)

  def test_get_existing_key_ignores_default(self) -> None:
    """§15.3: When key exists, the default is ignored."""
    Q.set('k', 42)
    result = Q.get('k', 'fallback')
    self.assertEqual(result, 42)

  def test_get_missing_key_no_context_raises(self) -> None:
    """§15.3: KeyError raised even when context store has never been initialized."""
    _reset_context()
    with self.assertRaises(KeyError):
      Q.get('never_set')

  def test_get_missing_key_no_context_with_default(self) -> None:
    """§15.3: Default returned even when context store has never been initialized."""
    _reset_context()
    result = Q.get('never_set', 'safe')
    self.assertEqual(result, 'safe')


# ---------------------------------------------------------------------------
# Operations between set and get (CRITICAL — context survives through pipeline ops)
# ---------------------------------------------------------------------------


class SetThenOpsThenGetTest(TestCase):
  """§15.1/§15.4: Context survives through various pipeline operations.

  These tests verify that .set('k') stores a value that remains
  accessible via Q.get('k') after arbitrary intermediate pipeline
  operations transform the current value.
  """

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_set_then_ops_then_get(self) -> None:
    """Context survives through multiple .then() steps."""
    result = Q(10).set('k').then(lambda x: x + 1).then(lambda x: x * 2).then(lambda x: x + 100).then(Q.get, 'k').run()
    self.assertEqual(result, 10)

  def test_set_do_then_get(self) -> None:
    """Context survives through .do() (side-effect, discards result)."""
    side_effects: list[Any] = []
    result = Q(10).set('k').do(lambda x: side_effects.append(x)).then(Q.get, 'k').run()
    self.assertEqual(result, 10)
    self.assertEqual(side_effects, [10])

  def test_set_foreach_then_get(self) -> None:
    """Context survives through .foreach() (iteration)."""
    result = Q(5).set('k').then(lambda x: [x, x + 1, x + 2]).foreach(lambda x: x * 10).then(lambda _: Q.get('k')).run()
    self.assertEqual(result, 5)

  def test_set_gather_then_get(self) -> None:
    """Context survives through .gather() (concurrent execution)."""
    result = Q(7).set('k').gather(lambda x: x + 1, lambda x: x + 2).then(lambda _: Q.get('k')).run()
    self.assertEqual(result, 7)

  def test_set_with_then_get(self) -> None:
    """Context survives through .with_() (context manager)."""

    @contextmanager
    def make_cm():
      yield 'inside'

    result = Q(42).set('k').then(lambda _: make_cm()).with_(lambda ctx: ctx.upper()).then(lambda _: Q.get('k')).run()
    self.assertEqual(result, 42)

  def test_set_if_then_get(self) -> None:
    """Context survives through .if_() (conditional execution)."""
    result = Q(10).set('k').if_(lambda x: x > 5).then(lambda x: x * 100).then(Q.get, 'k').run()
    self.assertEqual(result, 10)

  def test_set_nested_chain_then_get(self) -> None:
    """Context survives through nested pipeline execution."""
    inner = Q().then(lambda x: x * 2)
    result = Q(15).set('k').then(inner).then(Q.get, 'k').run()
    self.assertEqual(result, 15)

  def test_set_many_ops_then_get(self) -> None:
    """Context survives through a long pipeline with mixed operations."""
    side: list[Any] = []
    result = (
      Q(8).set('k').then(lambda x: x + 1).do(lambda x: side.append(x)).then(lambda x: x * 2).then(Q.get, 'k').run()
    )
    self.assertEqual(result, 8)
    self.assertEqual(side, [9])


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class SetGetAsyncTest(SymmetricTestCase):
  """§15.1/§15.3: set/get works across sync/async pipeline steps."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  async def test_set_get_async(self) -> None:
    """Pipeline with async steps: set/get still works."""

    async def async_double(x: Any) -> Any:
      return x * 2

    def sync_double(x: Any) -> Any:
      return x * 2

    await self.variant(
      lambda fn: Q(10).set('k').then(fn).then(Q.get, 'k').run(),
      expected=10,
      fn=[('sync', sync_double), ('async', async_double)],
    )

  async def test_set_get_async_multiple_steps(self) -> None:
    """Async pipeline: context survives through multiple async steps."""

    async def async_add_one(x: Any) -> Any:
      return x + 1

    def sync_add_one(x: Any) -> Any:
      return x + 1

    await self.variant(
      lambda fn: Q(5).set('k').then(fn).then(fn).then(fn).then(Q.get, 'k').run(),
      expected=5,
      fn=[('sync', sync_add_one), ('async', async_add_one)],
    )


# ---------------------------------------------------------------------------
# §15.1 if_() support — Conditional set
# ---------------------------------------------------------------------------


class SetAfterIfRaisesTest(TestCase):
  """§5.8: set() after if_() raises QuentException — only then()/do() consume if_()."""

  def test_if_set_raises(self) -> None:
    """chain.if_(pred).set('k') raises QuentException."""
    with self.assertRaises(QuentException):
      Q(42).if_(lambda x: x > 0).set('k')

  def test_if_set_explicit_raises(self) -> None:
    """chain.if_(pred).set('k', value) raises QuentException."""
    with self.assertRaises(QuentException):
      Q(42).if_(lambda x: x > 0).set('k', 'val')


# ---------------------------------------------------------------------------
# §15.4 Concurrent context — isolation
# ---------------------------------------------------------------------------


class ConcurrentContextTest(TestCase):
  """§15.4: Concurrent workers inherit context snapshot; writes are isolated."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_concurrent_foreach_sees_context(self) -> None:
    """§15.4: Workers in concurrent foreach can read context set before dispatch."""
    results: list[Any] = []
    lock = threading.Lock()

    def read_context(x: Any) -> Any:
      val = Q.get('shared')
      with lock:
        results.append(val)
      return x

    Q(99).set('shared').then(lambda _: [1, 2, 3]).foreach(read_context, concurrency=2).run()
    self.assertEqual(len(results), 3)
    for val in results:
      self.assertEqual(val, 99)

  def test_concurrent_gather_sees_context(self) -> None:
    """§15.4: Workers in concurrent gather can read context set before dispatch."""

    def read_context_a(x: Any) -> Any:
      return Q.get('shared')

    def read_context_b(x: Any) -> Any:
      return Q.get('shared')

    result = Q(77).set('shared').gather(read_context_a, read_context_b, concurrency=2).run()
    self.assertEqual(result, (77, 77))

  def test_concurrent_worker_set_isolation(self) -> None:
    """§15.4: A worker calling Q.set() does NOT affect parent or siblings.

    Copy-on-write dict semantics + copy_context().run() ensure isolation.

    Note: §11.3/§11.5 — the first item is probed in the calling thread,
    so only items dispatched to the thread pool (items after the first)
    are isolated via copy_context().run(). This test uses gather() where
    all functions are dispatched to pool workers after the first probe.
    """
    Q.set('parent_key', 'original')

    def worker_a(x: Any) -> Any:
      Q.set('parent_key', 'from_a')
      Q.set('a_only', 'a')
      return Q.get('parent_key')

    def worker_b(x: Any) -> Any:
      Q.set('parent_key', 'from_b')
      Q.set('b_only', 'b')
      return Q.get('parent_key')

    # gather: first fn is probed, second is dispatched to pool.
    # The pool worker (worker_b) runs in a copied context.
    result = Q(1).gather(worker_a, worker_b, concurrency=2).run()

    # Each worker sees its own writes
    self.assertEqual(result[0], 'from_a')
    self.assertEqual(result[1], 'from_b')

    # worker_b's writes (pool thread) must NOT be visible in parent
    with self.assertRaises(KeyError):
      Q.get('b_only')


class ConcurrentContextAsyncTest(IsolatedAsyncioTestCase):
  """§15.4: Async concurrent workers inherit and isolate context."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  async def test_async_concurrent_foreach_sees_context(self) -> None:
    """§15.4: Async concurrent foreach workers read parent context."""
    results: list[Any] = []

    async def read_context(x: Any) -> Any:
      val = Q.get('shared')
      results.append(val)
      return x

    await Q(55).set('shared').then(lambda _: [1, 2, 3]).foreach(read_context, concurrency=2).run()
    self.assertEqual(len(results), 3)
    for val in results:
      self.assertEqual(val, 55)

  async def test_async_concurrent_gather_sees_context(self) -> None:
    """§15.4: Async concurrent gather workers read parent context."""

    async def read_a(x: Any) -> Any:
      return Q.get('shared')

    async def read_b(x: Any) -> Any:
      return Q.get('shared')

    result = await Q(33).set('shared').gather(read_a, read_b, concurrency=2).run()
    self.assertEqual(result, (33, 33))


# ---------------------------------------------------------------------------
# §15.4 Context scoping — persistence across runs
# ---------------------------------------------------------------------------


class ContextPersistenceTest(TestCase):
  """§15.4: Values set in one .run() are visible to subsequent .run() in same context."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_context_persists_across_runs(self) -> None:
    """§15.4: contextvars semantics — values persist in same thread context.

    'Main execution path (non-concurrent): Context persists in the caller's
    thread. Values set during one pipeline execution are visible to
    subsequent executions in the same thread context.'
    """
    # First run: store a value
    Q(100).set('persistent').run()

    # Second run: retrieve the value stored by the first run
    result = Q(0).then(lambda _: Q.get('persistent')).run()
    self.assertEqual(result, 100)

  def test_static_set_persists_to_pipeline(self) -> None:
    """§15.2/§15.4: Static set before run is visible inside pipeline."""
    Q.set('pre', 'before_run')
    result = Q(0).then(lambda _: Q.get('pre')).run()
    self.assertEqual(result, 'before_run')

  def test_pipeline_set_visible_after_run(self) -> None:
    """§15.4: Pipeline .set() values are visible after .run() completes."""
    Q(200).set('post').run()
    self.assertEqual(Q.get('post'), 200)


# ---------------------------------------------------------------------------
# §15.1 Null sentinel normalization
# ---------------------------------------------------------------------------


class NullNormalizationTest(TestCase):
  """§15.1: Null sentinel is normalized to None before storage."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_no_root_value_set_stores_none(self) -> None:
    """§15.1: When no current value exists, set stores None.

    'When no current value exists, it is called with no arguments.
    The Null sentinel is normalized to None before storage —
    context values never contain Null.'
    """
    # Q() with no root value — current value is Null internally
    result = Q().set('k').then(lambda: Q.get('k')).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# §15.5 Dual dispatch — descriptor behavior
# ---------------------------------------------------------------------------


class DualDispatchTest(TestCase):
  """§15.5: Q.set and Q.get as descriptors enable dual dispatch."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_instance_set_returns_chain(self) -> None:
    """§15.5: Instance access chain.set('key') returns the chain (fluent)."""
    c = Q(1)
    result = c.set('k')
    self.assertIs(result, c)

  def test_class_set_returns_none(self) -> None:
    """§15.5: Class access Q.set('key', value) returns None."""
    result = Q.set('k', 42)
    self.assertIsNone(result)

  def test_class_get_retrieves_immediately(self) -> None:
    """§15.3/§15.5: Class access Q.get('k') retrieves value immediately."""
    Q.set('k', 'hello')
    result = Q.get('k')
    self.assertEqual(result, 'hello')

  def test_class_get_as_pipeline_callable(self) -> None:
    """§15.3: Q.get can be passed as a callable to .then() (Rule 1)."""
    Q.set('k', 'hello')
    result = Q(0).then(Q.get, 'k').run()
    self.assertEqual(result, 'hello')

  def test_instance_get_returns_chain(self) -> None:
    """§15.5: Instance access chain.get('key') returns the chain (fluent)."""
    c = Q(1).set('k')
    result = c.get('k')
    self.assertIs(result, c)

  def test_set_get_dual_dispatch_symmetry(self) -> None:
    """§15.5: Both set and get support instance (fluent) and class (immediate) access.

    Instance: q.set('k') / q.get('k') — pipeline steps, return q.
    Class: Q.set('k', v) / Q.get('k') — immediate, return None/value.
    """
    # Class access (immediate)
    Q.set('class_k', 'class_v')
    self.assertEqual(Q.get('class_k'), 'class_v')

    # Instance access (pipeline steps)
    result = Q(99).set('inst_k').get('inst_k').run()
    self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# §15.3 q.get('key') — Instance Method (Descriptor Pipeline Step)
# ---------------------------------------------------------------------------


class GetDescriptorBasicTest(TestCase):
  """§15.3: chain.get('key') appends a pipeline step that retrieves
  the stored context value and replaces the current value (like .then()).
  """

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_get_descriptor_basic(self) -> None:
    """§15.3: Q(42).set('k').get('k').run() == 42.

    Instance .get('k') retrieves value from context, replaces CV.
    """
    result = Q(42).set('k').get('k').run()
    self.assertEqual(result, 42)

  def test_get_descriptor_replaces_cv(self) -> None:
    """§15.3: chain.get('k') replaces current value with stored value.

    Unlike .set() which preserves CV, .get() replaces it —
    behaves like .then(), not .do().
    """
    result = (
      Q(10)
      .set('k')
      .then(lambda x: x * 100)  # CV is now 1000
      .get('k')  # CV should be replaced by stored value (10)
      .run()
    )
    self.assertEqual(result, 10)

  def test_get_descriptor_with_default(self) -> None:
    """§15.3: chain.get('nonexistent', 'fb').run() returns 'fb'.

    When key is missing and default is provided, the default becomes CV.
    """
    result = Q(42).get('nonexistent', 'fb').run()
    self.assertEqual(result, 'fb')

  def test_get_descriptor_missing_raises(self) -> None:
    """§15.3: chain.get('nonexistent').run() raises KeyError.

    When key is missing and no default, KeyError is raised at execution time.
    """
    with self.assertRaises(KeyError):
      Q(42).get('nonexistent').run()

  def test_get_descriptor_returns_chain(self) -> None:
    """§15.5: Instance access chain.get('key') returns the chain (fluent)."""
    c = Q(1).set('k')
    result = c.get('k')
    self.assertIs(result, c)

  def test_get_descriptor_default_none(self) -> None:
    """§15.3: chain.get('nonexistent', None).run() returns None (not KeyError).

    Providing None as the default is distinct from not providing a default.
    """
    result = Q(42).get('nonexistent', None).run()
    self.assertIsNone(result)

  def test_get_descriptor_existing_key_ignores_default(self) -> None:
    """§15.3: When key exists in context, the default is ignored."""
    result = Q(42).set('k').get('k', 'fallback').run()
    self.assertEqual(result, 42)


class GetDescriptorAfterOpsTest(TestCase):
  """§15.3/§15.4: chain.get('k') retrieves the original value after
  intermediate pipeline operations transform the current value.
  """

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_get_descriptor_after_then(self) -> None:
    """Context value retrieved via .get() after .then() operations."""
    result = Q(10).set('k').then(lambda x: x + 1).then(lambda x: x * 2).then(lambda x: x + 100).get('k').run()
    self.assertEqual(result, 10)

  def test_get_descriptor_after_do(self) -> None:
    """Context value retrieved via .get() after .do() (side-effect)."""
    side_effects: list[Any] = []
    result = Q(10).set('k').do(lambda x: side_effects.append(x)).get('k').run()
    self.assertEqual(result, 10)
    self.assertEqual(side_effects, [10])

  def test_get_descriptor_after_foreach(self) -> None:
    """Context value retrieved via .get() after .foreach() (iteration)."""
    result = (
      Q(5)
      .set('k')
      .then(lambda x: [x, x + 1, x + 2])
      .foreach(lambda x: x * 10)
      .then(lambda _: 'discarded')
      .get('k')
      .run()
    )
    self.assertEqual(result, 5)

  def test_get_descriptor_after_gather(self) -> None:
    """Context value retrieved via .get() after .gather() (concurrent)."""
    result = Q(7).set('k').gather(lambda x: x + 1, lambda x: x + 2).then(lambda _: 'discarded').get('k').run()
    self.assertEqual(result, 7)

  def test_get_descriptor_after_with(self) -> None:
    """Context value retrieved via .get() after .with_() (context manager)."""

    @contextmanager
    def make_cm():
      yield 'inside'

    result = Q(42).set('k').then(lambda _: make_cm()).with_(lambda ctx: ctx.upper()).get('k').run()
    self.assertEqual(result, 42)

  def test_get_descriptor_after_if(self) -> None:
    """Context value retrieved via .get() after .if_() (conditional)."""
    result = Q(10).set('k').if_(lambda x: x > 5).then(lambda x: x * 100).get('k').run()
    self.assertEqual(result, 10)

  def test_get_descriptor_after_many_ops(self) -> None:
    """Context value retrieved via .get() after a long mixed pipeline."""
    side: list[Any] = []
    result = Q(8).set('k').then(lambda x: x + 1).do(lambda x: side.append(x)).then(lambda x: x * 2).get('k').run()
    self.assertEqual(result, 8)
    self.assertEqual(side, [9])


class GetAfterIfRaisesTest(TestCase):
  """§5.8: get() after if_() raises QuentException — only then()/do() consume if_()."""

  def test_if_get_raises(self) -> None:
    """chain.if_(pred).get('k') raises QuentException."""
    with self.assertRaises(QuentException):
      Q(42).set('k').if_(lambda x: x > 0).get('k')

  def test_if_get_with_default_raises(self) -> None:
    """chain.if_(pred).get('k', default) raises QuentException."""
    with self.assertRaises(QuentException):
      Q(42).if_(lambda x: x > 0).get('k', 'fb')


class GetDescriptorNestedChainTest(TestCase):
  """§15.3/§15.4: Context visible inside nested chains via .get()."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_get_descriptor_nested_chain(self) -> None:
    """A nested chain can use .get() to access context from parent."""
    inner = Q().then(lambda x: x * 2).get('outer_k')
    result = Q(15).set('outer_k').then(lambda x: x + 5).then(inner).run()
    # inner receives 20 (15+5), doubles to 40, then .get('outer_k') → 15
    self.assertEqual(result, 15)


class GetDescriptorAsyncTest(SymmetricTestCase):
  """§15.3: chain.get('key') works correctly with async pipeline steps."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  async def test_get_descriptor_async(self) -> None:
    """Pipeline with async steps: chain.get() still works."""

    async def async_double(x: Any) -> Any:
      return x * 2

    def sync_double(x: Any) -> Any:
      return x * 2

    await self.variant(
      lambda fn: Q(10).set('k').then(fn).get('k').run(),
      expected=10,
      fn=[('sync', sync_double), ('async', async_double)],
    )

  async def test_get_descriptor_async_multiple_steps(self) -> None:
    """Async pipeline: .get() retrieves context after multiple async steps."""

    async def async_add_one(x: Any) -> Any:
      return x + 1

    def sync_add_one(x: Any) -> Any:
      return x + 1

    await self.variant(
      lambda fn: Q(5).set('k').then(fn).then(fn).then(fn).get('k').run(),
      expected=5,
      fn=[('sync', sync_add_one), ('async', async_add_one)],
    )


class GetDescriptorChainingTest(TestCase):
  """§15.3/§15.1: Complex chaining of set/get descriptor forms."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_get_descriptor_chaining(self) -> None:
    """Multiple .set() and .get() in a chain: values tracked independently.

    q.set('a').set('b').get('a').set('c').get('b'):
    - set('a') stores CV under 'a'
    - set('b') stores CV under 'b' (same CV, set preserves it)
    - get('a') replaces CV with stored 'a'
    - set('c') stores current CV (which is 'a' value) under 'c'
    - get('b') replaces CV with stored 'b'
    """
    result = (
      Q(42)
      .set('a')  # context: a=42, CV=42
      .then(lambda x: x * 2)  # CV=84
      .set('b')  # context: a=42, b=84, CV=84
      .get('a')  # CV=42
      .set('c')  # context: a=42, b=84, c=42, CV=42
      .get('b')  # CV=84
      .run()
    )
    self.assertEqual(result, 84)
    # Verify all context values are correct
    self.assertEqual(Q.get('a'), 42)
    self.assertEqual(Q.get('b'), 84)
    self.assertEqual(Q.get('c'), 42)

  def test_get_descriptor_round_trip(self) -> None:
    """set then get then continue pipeline: get result feeds next step."""
    result = (
      Q(10)
      .set('k')
      .then(lambda x: x * 100)  # CV=1000
      .get('k')  # CV=10
      .then(lambda x: x + 5)  # CV=15
      .run()
    )
    self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# §15.1 q.set('key', value) — Instance Method with Explicit Value
# ---------------------------------------------------------------------------


class SetExplicitBasicTest(TestCase):
  """§15.1: chain.set('key', value) stores an explicit value under key.
  The current value is unchanged (like .do()).
  """

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_set_explicit_basic(self) -> None:
    """§15.1: Q(10).set('s', 'active').run() == 10.

    Explicit value is stored; CV passes through unchanged.
    Q.get('s') returns the explicit value.
    """
    result = Q(10).set('s', 'active').run()
    self.assertEqual(result, 10)
    self.assertEqual(Q.get('s'), 'active')

  def test_set_explicit_preserves_cv(self) -> None:
    """§15.1: Explicit set doesn't change current value.

    'The current value is not changed — the step's result is discarded, like .do().'
    """
    result = (
      Q(10)
      .set('s', 'hello')
      .then(lambda x: x + 1)  # x should be 10 (not 'hello')
      .run()
    )
    self.assertEqual(result, 11)

  def test_set_explicit_then_get(self) -> None:
    """§15.1/§15.3: Q(10).set('s', 'hello').get('s').run() == 'hello'.

    Explicit set stores value, then .get() retrieves it.
    """
    result = Q(10).set('s', 'hello').get('s').run()
    self.assertEqual(result, 'hello')

  def test_set_explicit_overwrites(self) -> None:
    """§15.1/§15.4: Explicit set overwrites a previous set on same key."""
    result = Q(10).set('k').set('k', 'override').get('k').run()
    # .set('k') stores 10, then .set('k', 'override') overwrites with 'override'
    self.assertEqual(result, 'override')

  def test_set_explicit_overwrites_by_cv_set(self) -> None:
    """§15.1/§15.4: CV-based set overwrites a previous explicit set."""
    result = (
      Q(10)
      .set('k', 'explicit')
      .then(lambda x: x * 5)  # CV=50
      .set('k')  # stores CV (50), overwriting 'explicit'
      .get('k')
      .run()
    )
    self.assertEqual(result, 50)

  def test_set_explicit_returns_chain(self) -> None:
    """§15.5: Instance access chain.set('key', value) returns the chain (fluent)."""
    c = Q(1)
    result = c.set('k', 42)
    self.assertIs(result, c)


class SetExplicitWithOpsTest(TestCase):
  """§15.1/§15.4: Explicit set value survives through pipeline operations."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_set_explicit_with_ops_between(self) -> None:
    """Explicit set value retrievable after multiple pipeline operations."""
    result = (
      Q(10).set('config', 'prod').then(lambda x: x + 1).then(lambda x: x * 2).do(lambda x: None).get('config').run()
    )
    self.assertEqual(result, 'prod')

  def test_set_explicit_cv_unchanged_through_pipeline(self) -> None:
    """Explicit set doesn't interfere with CV flowing through pipeline."""
    result = Q(10).set('meta', {'source': 'test'}).then(lambda x: x + 5).then(lambda x: x * 2).run()
    self.assertEqual(result, 30)
    self.assertEqual(Q.get('meta'), {'source': 'test'})


class SetExplicitAfterIfRaisesTest(TestCase):
  """§5.8: set(key, value) after if_() raises QuentException."""

  def test_if_set_explicit_raises(self) -> None:
    """chain.if_(pred).set('k', value) raises QuentException."""
    with self.assertRaises(QuentException):
      Q(42).if_(lambda x: x > 0).set('k', 'stored')


class SetExplicitVsCvTest(TestCase):
  """§15.1: Verify that set('k') stores CV while set('k', value) stores explicit value."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_set_explicit_vs_cv(self) -> None:
    """chain.set('k') stores CV; chain.set('k', value) stores explicit value.

    Even when CV differs from the explicit value, each form stores
    the correct thing.
    """
    result = (
      Q(10)
      .set('cv_store')  # stores CV (10)
      .set('explicit_store', 'custom')  # stores 'custom'
      .then(lambda x: x * 3)  # CV=30
      .set('cv_store_2')  # stores CV (30)
      .set('explicit_store_2', 'custom2')  # stores 'custom2'
      .run()
    )
    self.assertEqual(result, 30)
    self.assertEqual(Q.get('cv_store'), 10)
    self.assertEqual(Q.get('explicit_store'), 'custom')
    self.assertEqual(Q.get('cv_store_2'), 30)
    self.assertEqual(Q.get('explicit_store_2'), 'custom2')

  def test_set_explicit_ignores_cv(self) -> None:
    """§15.1: Two-arg form: 'the explicit value is captured in a closure
    and the current value is ignored.'
    """
    # Even though CV is 999, set('k', 'fixed') stores 'fixed'
    result = Q(999).set('k', 'fixed').get('k').run()
    self.assertEqual(result, 'fixed')

  def test_set_explicit_multiple_keys_mixed(self) -> None:
    """Mix of CV-based and explicit sets on different keys."""
    result = (
      Q(10)
      .set('a')  # a=10 (CV)
      .set('b', 'explicit_b')  # b='explicit_b'
      .then(lambda x: x + 5)  # CV=15
      .set('c')  # c=15 (CV)
      .set('d', 'explicit_d')  # d='explicit_d'
      .run()
    )
    self.assertEqual(result, 15)
    self.assertEqual(Q.get('a'), 10)
    self.assertEqual(Q.get('b'), 'explicit_b')
    self.assertEqual(Q.get('c'), 15)
    self.assertEqual(Q.get('d'), 'explicit_d')


class SetExplicitAsyncTest(SymmetricTestCase):
  """§15.1: chain.set('key', value) works with async pipeline steps."""

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  async def test_set_explicit_async(self) -> None:
    """Explicit set works correctly when async steps are in the pipeline."""

    async def async_double(x: Any) -> Any:
      return x * 2

    def sync_double(x: Any) -> Any:
      return x * 2

    await self.variant(
      lambda fn: Q(10).set('k', 'explicit').then(fn).get('k').run(),
      expected='explicit',
      fn=[('sync', sync_double), ('async', async_double)],
    )


# ---------------------------------------------------------------------------
# §15.4 — Async concurrent worker context isolation
# ---------------------------------------------------------------------------


class AsyncConcurrentWorkerContextIsolationTest(IsolatedAsyncioTestCase):
  """§15.4: Async concurrent workers inherit context snapshot; set() does not propagate back.

  §15.4:
    'Async concurrent tasks: Async tasks inherit context naturally through
    Python's asyncio task creation mechanism, which copies the current context
    into the new task. The same isolation guarantees apply — a task's set()
    does not affect the parent or siblings.'
  """

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  async def test_async_foreach_workers_inherit_but_writes_isolated(self) -> None:
    """Async workers via foreach(async_fn, concurrency=N) inherit context snapshot.

    Workers can read parent context values, but set() from a worker does NOT
    propagate back to the parent.
    """
    Q.set('parent_val', 'visible')

    async def async_worker(x: Any) -> Any:
      # Worker can read parent context
      parent = Q.get('parent_val')
      # Worker sets its own key — should NOT propagate to parent
      Q.set(f'worker_{x}', f'set_by_{x}')
      return (x, parent)

    result = await Q([1, 2, 3]).foreach(async_worker, concurrency=2).run()
    # All workers see the parent's context value
    for item in result:
      self.assertEqual(item[1], 'visible')

    # Parent still has its original value
    self.assertEqual(Q.get('parent_val'), 'visible')

    # Worker writes should NOT be visible in parent context
    for i in [1, 2, 3]:
      self.assertEqual(Q.get(f'worker_{i}', 'missing'), 'missing')

  async def test_async_gather_workers_inherit_but_writes_isolated(self) -> None:
    """Async gather workers inherit context but writes are isolated."""
    Q.set('shared_key', 'original')

    async def worker_a(x: Any) -> Any:
      Q.set('shared_key', 'from_worker_a')
      Q.set('a_only', 'a_val')
      return Q.get('shared_key')

    async def worker_b(x: Any) -> Any:
      Q.set('shared_key', 'from_worker_b')
      Q.set('b_only', 'b_val')
      return Q.get('shared_key')

    result = await Q(1).gather(worker_a, worker_b, concurrency=2).run()
    # Each worker sees its own writes
    self.assertEqual(result[0], 'from_worker_a')
    self.assertEqual(result[1], 'from_worker_b')

    # Parent context is unchanged
    self.assertEqual(Q.get('shared_key'), 'original')
    # Worker-only keys are not visible to parent
    self.assertEqual(Q.get('a_only', 'missing'), 'missing')
    self.assertEqual(Q.get('b_only', 'missing'), 'missing')


# ---------------------------------------------------------------------------
# §15.1/§15.3 — Conditional workaround patterns
# ---------------------------------------------------------------------------


class ConditionalContextPatternsTest(TestCase):
  """§15.1/§15.3: Spec-suggested conditional workaround patterns.

  §15.1: '.set() does not consume a pending if_() or while_(). For conditional
  storage, use .if_(pred).do(lambda cv: Q.set("key", cv)).'
  §15.3: 'For conditional retrieval, use .if_(pred).then(Q.get, "key").'
  """

  def setUp(self) -> None:
    _reset_context()

  def tearDown(self) -> None:
    _reset_context()

  def test_conditional_get_pattern(self) -> None:
    """if_(pred).then(Q.get, 'key') — conditional retrieval.

    §15.3: 'For conditional retrieval, use .if_(pred).then(Q.get, "key").'
    When the predicate is true, Q.get('key') is called with Rule 1
    (explicit args: key is passed, current value is NOT passed).
    """
    Q.set('fallback_key', 'fallback_value')
    # When predicate is true, the get replaces current value
    result = Q(42).if_(lambda x: x > 10).then(Q.get, 'fallback_key').run()
    self.assertEqual(result, 'fallback_value')

  def test_conditional_get_pattern_false(self) -> None:
    """if_(pred).then(Q.get, 'key') when predicate is false — current value passes through."""
    Q.set('fallback_key', 'fallback_value')
    # When predicate is false, the then branch is skipped; CV passes through
    result = Q(42).if_(lambda x: x < 10).then(Q.get, 'fallback_key').run()
    self.assertEqual(result, 42)

  def test_conditional_set_pattern(self) -> None:
    """if_(pred).do(lambda cv: Q.set('key', cv)) — conditional storage.

    §15.1: 'For conditional storage, use .if_(pred).do(lambda cv: Q.set("key", cv)).'
    The lambda receives CV as its argument (Rule 2) and calls Q.set class-level
    method to store it immediately.
    """
    result = Q(99).if_(lambda x: x > 50).do(lambda cv: Q.set('stored', cv)).run()
    self.assertEqual(result, 99)  # .do() preserves CV
    self.assertEqual(Q.get('stored'), 99)

  def test_conditional_set_pattern_false(self) -> None:
    """if_(pred).do(lambda cv: Q.set('key', cv)) when predicate is false — no store."""
    result = Q(5).if_(lambda x: x > 50).do(lambda cv: Q.set('stored2', cv)).run()
    self.assertEqual(result, 5)
    # Key was not stored because predicate was false
    with self.assertRaises(KeyError):
      Q.get('stored2')


if __name__ == '__main__':
  import unittest

  unittest.main()
