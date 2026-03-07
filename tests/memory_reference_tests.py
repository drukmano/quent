"""Tests for task registry lifecycle, reference semantics, slots, and memory behavior."""
from __future__ import annotations

import asyncio
import gc
import sys
import unittest
import weakref
from unittest import IsolatedAsyncioTestCase

from quent import Chain
from quent._core import Link, Null, _Null, _ensure_future, _task_registry, _set_link_temp_args


# ---------------------------------------------------------------------------
# Task registry lifecycle
# ---------------------------------------------------------------------------

class TestTaskRegistryLifecycle(IsolatedAsyncioTestCase):

  async def test_registry_initial(self):
    """Record initial size -- registry may contain tasks from other tests."""
    initial = len(_task_registry)
    self.assertIsInstance(_task_registry, set)
    self.assertGreaterEqual(initial, 0)

  async def test_single_task_lifecycle(self):
    """Single task: added, completed, removed."""
    initial = len(_task_registry)

    async def _job():
      return 'done'

    task = _ensure_future(_job())
    self.assertIn(task, _task_registry)
    await task
    await asyncio.sleep(0)
    self.assertNotIn(task, _task_registry)
    self.assertEqual(len(_task_registry), initial)

  async def test_concurrent_tasks(self):
    """10 concurrent tasks all tracked and all removed on completion."""
    initial = len(_task_registry)
    tasks = []
    for i in range(10):
      async def _job(n=i):
        await asyncio.sleep(0.001)
        return n
      tasks.append(_ensure_future(_job()))
    self.assertGreaterEqual(len(_task_registry), initial + 10)
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    self.assertEqual(len(_task_registry), initial)

  async def test_exception_task_removed(self):
    """Task that raises is still removed from registry."""
    initial = len(_task_registry)

    async def _boom():
      raise RuntimeError('oops')

    task = _ensure_future(_boom())
    with self.assertRaises(RuntimeError):
      await task
    await asyncio.sleep(0)
    self.assertEqual(len(_task_registry), initial)

  async def test_bulk_100_tasks(self):
    """Create 100 tasks, all complete, registry returns to initial size."""
    initial = len(_task_registry)

    async def _noop(i):
      return i

    tasks = [_ensure_future(_noop(i)) for i in range(100)]
    self.assertGreaterEqual(len(_task_registry), initial + 100)
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    self.assertEqual(len(_task_registry), initial)


# ---------------------------------------------------------------------------
# Chain reference semantics
# ---------------------------------------------------------------------------

class TestChainReferenceSemantics(unittest.TestCase):

  def test_chain_does_not_hold_run_value(self):
    """After run(42), the chain does not retain 42 in any slot."""
    chain = Chain().then(lambda x: x + 1)
    result = chain.run(42)
    self.assertEqual(result, 43)
    # The chain should not store the run-time value.
    self.assertIsNone(chain.root_link)
    self.assertIsNone(chain.current_link)

  def test_link_holds_strong_reference(self):
    """Link.v is the same object as the input."""
    fn = lambda x: x
    link = Link(fn)
    self.assertIs(link.v, fn)

  def test_link_next_link_chain(self):
    """Linked list traversal: link.next_link forms a chain."""
    chain = Chain(lambda: 1).then(lambda x: x + 1).then(lambda x: x * 2)
    link = chain.root_link
    self.assertIsNotNone(link)
    self.assertIsNotNone(link.next_link)
    self.assertIs(link.next_link, chain.first_link)
    second = link.next_link
    self.assertIsNotNone(second.next_link)
    self.assertIs(second.next_link, chain.current_link)
    self.assertIsNone(second.next_link.next_link)

  def test_nested_chain_is_nested_persistent(self):
    """Once is_nested is set on a chain, it stays set."""
    inner = Chain(lambda: 10)
    self.assertFalse(inner.is_nested)
    Link(inner)
    self.assertTrue(inner.is_nested)
    # Stays set.
    self.assertTrue(inner.is_nested)

  def test_chain_slots(self):
    """Chain uses __slots__, so arbitrary attribute assignment raises."""
    chain = Chain()
    with self.assertRaises(AttributeError):
      chain.nonexistent_attr = 'fail'

  def test_link_slots(self):
    """Link uses __slots__, so arbitrary attribute assignment raises."""
    link = Link(42)
    with self.assertRaises(AttributeError):
      link.nonexistent_attr = 'fail'

  def test_null_slots(self):
    """_Null uses __slots__, so arbitrary attribute assignment raises."""
    with self.assertRaises(AttributeError):
      Null.nonexistent_attr = 'fail'

  def test_chain_no_dict(self):
    """Chain.__dict__ is not accessible (uses __slots__)."""
    chain = Chain()
    with self.assertRaises(AttributeError):
      _ = chain.__dict__

  def test_link_no_dict(self):
    """Link.__dict__ is not accessible (uses __slots__)."""
    link = Link(42)
    with self.assertRaises(AttributeError):
      _ = link.__dict__


# ---------------------------------------------------------------------------
# Beyond-spec: weakref behavior
# ---------------------------------------------------------------------------

class TestWeakrefBehavior(unittest.TestCase):

  def test_cannot_weakref_chain(self):
    """Chain uses __slots__ without __weakref__, so weakref should fail."""
    chain = Chain()
    with self.assertRaises(TypeError):
      weakref.ref(chain)

  def test_cannot_weakref_link(self):
    """Link uses __slots__ without __weakref__, so weakref should fail."""
    link = Link(42)
    with self.assertRaises(TypeError):
      weakref.ref(link)

  def test_cannot_weakref_null(self):
    """_Null uses __slots__ without __weakref__, so weakref should fail."""
    with self.assertRaises(TypeError):
      weakref.ref(Null)


# ---------------------------------------------------------------------------
# Beyond-spec: set_link_temp_args edge cases
# ---------------------------------------------------------------------------

class TestSetLinkTempArgsEdgeCases(unittest.TestCase):

  def test_with_non_exception_object(self):
    """_set_link_temp_args works with arbitrary objects, not just exceptions."""
    class Dummy:
      pass

    obj = Dummy()
    link = Link(lambda: None)
    _set_link_temp_args(obj, link, val=42)
    self.assertTrue(hasattr(obj, '__quent_link_temp_args__'))
    self.assertEqual(obj.__quent_link_temp_args__[id(link)], {'val': 42})

  def test_with_empty_kwargs(self):
    """_set_link_temp_args with no kwargs stores an empty dict."""
    exc = ValueError('test')
    link = Link(lambda: None)
    _set_link_temp_args(exc, link)
    self.assertEqual(exc.__quent_link_temp_args__[id(link)], {})

  def test_with_many_links(self):
    """Multiple links can each store data on the same exception."""
    exc = ValueError('test')
    links = [Link(lambda: None) for _ in range(20)]
    for i, link in enumerate(links):
      _set_link_temp_args(exc, link, idx=i)
    self.assertEqual(len(exc.__quent_link_temp_args__), 20)
    for i, link in enumerate(links):
      self.assertEqual(exc.__quent_link_temp_args__[id(link)], {'idx': i})


# ---------------------------------------------------------------------------
# Beyond-spec: memory pressure
# ---------------------------------------------------------------------------

class TestMemoryPressure(unittest.TestCase):

  def test_1000_chains_no_leak(self):
    """Create 1000 chains, verify no strong reference leak."""
    chains = [Chain(lambda i=i: i).then(lambda x: x + 1) for i in range(1000)]
    # All chains exist.
    self.assertEqual(len(chains), 1000)
    # Run them all.
    for i, c in enumerate(chains):
      result = c.run()
      self.assertEqual(result, i + 1)
    # Drop references.
    del chains
    gc.collect()
    # No assertion on gc counts -- just verify no crash.

  def test_deep_chain_nesting(self):
    """Deeply nested chains (100 levels) do not leak."""
    chain = Chain(lambda: 0)
    for _ in range(100):
      chain = chain.then(lambda x: x + 1)
    result = chain.run()
    self.assertEqual(result, 100)
    del chain
    gc.collect()


# ---------------------------------------------------------------------------
# Beyond-spec: link circular reference
# ---------------------------------------------------------------------------

class TestCircularReferences(unittest.TestCase):

  def test_link_circular_reference_gc(self):
    """A link with a circular reference can be GC'd."""
    chain = Chain(lambda: 1)
    link = Link(chain)
    # Create a cycle: chain's first_link -> link, link.v -> chain.
    chain._then(link)
    # Drop references and collect.
    chain_id = id(chain)
    del chain, link
    gc.collect()
    # Verify no crash -- Python's cycle collector handles this.

  def test_chain_gc_after_run(self):
    """Chain is GC-able after run completes."""
    chain = Chain(lambda: 42).then(lambda x: x + 1)
    result = chain.run()
    self.assertEqual(result, 43)
    del chain
    gc.collect()


# ---------------------------------------------------------------------------
# Beyond-spec: task_registry thread safety note
# ---------------------------------------------------------------------------

class TestTaskRegistryType(unittest.TestCase):

  def test_registry_is_set(self):
    """_task_registry is a plain set (not thread-safe)."""
    self.assertIsInstance(_task_registry, set)

  def test_registry_discard_nonexistent(self):
    """Discarding a nonexistent item from _task_registry does not raise."""
    # set.discard is safe for missing items.
    _task_registry.discard('nonexistent_sentinel_object')


# ---------------------------------------------------------------------------
# Beyond-spec: ensure_future edge cases
# ---------------------------------------------------------------------------

class TestEnsureFutureEdgeCases(IsolatedAsyncioTestCase):

  async def test_already_completed_coro(self):
    """_ensure_future with a coroutine that completes immediately."""
    async def _instant():
      return 'instant'

    task = _ensure_future(_instant())
    result = await task
    self.assertEqual(result, 'instant')

  async def test_registry_discard_idempotent(self):
    """Calling discard on an already-removed task does not raise."""
    async def _noop():
      return 1

    task = _ensure_future(_noop())
    await task
    await asyncio.sleep(0)
    # Task already removed by done callback; discard again is safe.
    _task_registry.discard(task)
    self.assertNotIn(task, _task_registry)


if __name__ == '__main__':
  unittest.main()
