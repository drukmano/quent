"""Tests for _ensure_future, _task_registry, and fire-and-forget task lifecycle."""
from __future__ import annotations

import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase

from quent._core import _ensure_future, _task_registry


class TestEnsureFuture(IsolatedAsyncioTestCase):

  async def test_task_added_to_registry(self):
    """After _ensure_future(coro), the task appears in _task_registry."""
    event = asyncio.Event()

    async def _wait():
      await event.wait()

    task = _ensure_future(_wait())
    self.assertIn(task, _task_registry)
    event.set()
    await task

  async def test_task_removed_on_completion(self):
    """After the task completes normally, it is removed from _task_registry."""
    async def _noop():
      return 42

    task = _ensure_future(_noop())
    self.assertIn(task, _task_registry)
    await task
    # Allow the done callback to fire.
    await asyncio.sleep(0)
    self.assertNotIn(task, _task_registry)

  async def test_task_removed_on_exception(self):
    """A task that raises is still removed from _task_registry."""
    async def _boom():
      raise ValueError('test')

    task = _ensure_future(_boom())
    self.assertIn(task, _task_registry)
    with self.assertRaises(ValueError):
      await task
    await asyncio.sleep(0)
    self.assertNotIn(task, _task_registry)

  async def test_multiple_tasks_tracked(self):
    """Multiple concurrent tasks are all tracked in _task_registry."""
    events = [asyncio.Event() for _ in range(5)]
    tasks = []
    for ev in events:
      async def _wait(e=ev):
        await e.wait()
      tasks.append(_ensure_future(_wait()))
    for t in tasks:
      self.assertIn(t, _task_registry)
    for ev in events:
      ev.set()
    await asyncio.gather(*tasks)

  async def test_registry_size_after_bulk(self):
    """Create 50 tasks, all complete, registry returns to initial size."""
    initial_size = len(_task_registry)

    async def _noop(i):
      return i

    tasks = [_ensure_future(_noop(i)) for i in range(50)]
    self.assertGreaterEqual(len(_task_registry), initial_size + 50)
    await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    self.assertEqual(len(_task_registry), initial_size)

  async def test_done_callback_is_discard(self):
    """Verify the callback mechanism uses _task_registry.discard."""
    async def _noop():
      return 1

    task = _ensure_future(_noop())
    # The done callback should be registered.
    # asyncio.Task stores callbacks internally; we verify the effect.
    self.assertIn(task, _task_registry)
    await task
    await asyncio.sleep(0)
    # discard was called via the done callback.
    self.assertNotIn(task, _task_registry)

  async def test_cancelled_task_removed(self):
    """Cancelling a task removes it from _task_registry."""
    async def _forever():
      await asyncio.sleep(3600)

    task = _ensure_future(_forever())
    self.assertIn(task, _task_registry)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    await asyncio.sleep(0)
    self.assertNotIn(task, _task_registry)

  async def test_coroutine_closed_on_runtime_error(self):
    """When no event loop is available, _ensure_future closes the coro and raises."""
    # We cannot truly test "no event loop" from within an async test,
    # but we can verify the behavior via the sync test class below.
    # This test verifies that _ensure_future works correctly within
    # a running loop (positive case).
    async def _noop():
      return 99

    task = _ensure_future(_noop())
    result = await task
    self.assertEqual(result, 99)

  async def test_task_result_preserved(self):
    """The task returned by _ensure_future preserves the coro's return value."""
    async def _compute():
      return 'hello'

    task = _ensure_future(_compute())
    result = await task
    self.assertEqual(result, 'hello')

  async def test_task_exception_preserved(self):
    """The task returned by _ensure_future preserves the coro's exception."""
    async def _boom():
      raise TypeError('bad type')

    task = _ensure_future(_boom())
    with self.assertRaises(TypeError) as ctx:
      await task
    self.assertEqual(str(ctx.exception), 'bad type')

  async def test_rapid_create_and_complete(self):
    """Rapidly creating and completing tasks does not leak registry entries."""
    initial_size = len(_task_registry)
    for _ in range(20):
      async def _quick():
        return True
      task = _ensure_future(_quick())
      await task
      await asyncio.sleep(0)
    self.assertEqual(len(_task_registry), initial_size)

  async def test_task_is_asyncio_task(self):
    """_ensure_future returns an asyncio.Task instance."""
    async def _noop():
      pass

    task = _ensure_future(_noop())
    self.assertIsInstance(task, asyncio.Task)
    await task

  async def test_concurrent_completion_order(self):
    """Tasks completing in different orders all get removed."""
    initial_size = len(_task_registry)
    results = []

    async def _delayed(val, delay):
      await asyncio.sleep(delay)
      results.append(val)
      return val

    t1 = _ensure_future(_delayed('a', 0.02))
    t2 = _ensure_future(_delayed('b', 0.01))
    t3 = _ensure_future(_delayed('c', 0.005))
    await asyncio.gather(t1, t2, t3)
    await asyncio.sleep(0)
    self.assertEqual(len(_task_registry), initial_size)
    # c should finish first, then b, then a.
    self.assertEqual(results, ['c', 'b', 'a'])


class TestEnsureFutureNoLoop(unittest.TestCase):

  def test_no_event_loop_raises(self):
    """Calling _ensure_future outside an async context raises RuntimeError."""
    async def _noop():
      return 1

    coro = _noop()
    with self.assertRaises(RuntimeError):
      _ensure_future(coro)

  def test_no_event_loop_closes_coro(self):
    """When _ensure_future raises RuntimeError, the coroutine is closed."""
    async def _noop():
      return 1

    coro = _noop()
    # _ensure_future will call coro.close() then raise RuntimeError.
    with self.assertRaises(RuntimeError):
      _ensure_future(coro)
    # After close(), sending into the coro raises StopIteration (already closed).
    # Calling close() again on an already-closed coro is a no-op.
    # We verify the coro is closed by checking cr_frame is None.
    self.assertIsNone(coro.cr_frame)

  def test_no_event_loop_does_not_add_to_registry(self):
    """When _ensure_future raises, the task is not added to _task_registry."""
    initial_size = len(_task_registry)

    async def _noop():
      return 1

    coro = _noop()
    with self.assertRaises(RuntimeError):
      _ensure_future(coro)
    self.assertEqual(len(_task_registry), initial_size)


if __name__ == '__main__':
  unittest.main()
