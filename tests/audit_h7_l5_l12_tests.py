"""Tests for bug fixes H7, L5, and L12.

H7: Thread-safe _task_registry (quent/_core.py)
L5: Gather index annotation (quent/_ops.py)
L12: _get_obj_name falsy name (quent/_traceback.py)
"""
from __future__ import annotations

import asyncio
import threading
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain
from quent._core import (
  _ensure_future,
  _task_registry,
  _task_registry_discard,
  _task_registry_lock,
)
from quent._traceback import _get_obj_name


# ---------------------------------------------------------------------------
# H7: Thread-safe _task_registry
# ---------------------------------------------------------------------------


class TestTaskRegistryLock(unittest.TestCase):

  def test_task_registry_lock_exists(self):
    """_task_registry_lock is a threading.Lock instance."""
    self.assertIsInstance(_task_registry_lock, type(threading.Lock()))

  def test_task_registry_discard_function_exists(self):
    """_task_registry_discard is a callable."""
    self.assertTrue(callable(_task_registry_discard))


class TestEnsureFuture(IsolatedAsyncioTestCase):

  async def test_ensure_future_registers_task(self):
    """_ensure_future adds the task to _task_registry."""
    event = asyncio.Event()

    async def waiter():
      await event.wait()

    task = _ensure_future(waiter())
    try:
      self.assertIn(task, _task_registry)
    finally:
      event.set()
      await task

  async def test_ensure_future_cleanup_on_done(self):
    """After a task completes, it is removed from _task_registry."""
    async def quick():
      return 42

    task = _ensure_future(quick())
    await task
    # The done callback is invoked synchronously after await completes,
    # but we yield control once to ensure it has fired.
    await asyncio.sleep(0)
    self.assertNotIn(task, _task_registry)

  def test_ensure_future_no_loop_raises(self):
    """Calling _ensure_future without a running event loop raises RuntimeError
    and closes the coroutine.
    """
    async def dummy():
      pass  # pragma: no cover

    coro = dummy()
    with self.assertRaises(RuntimeError):
      _ensure_future(coro)
    # Coroutine should have been closed by the except handler.
    # Calling .close() on an already-closed coroutine is a no-op (no error),
    # but .send(None) raises RuntimeError('cannot reuse already awaited coroutine').
    with self.assertRaises(RuntimeError):
      coro.send(None)


class TestTaskRegistryDiscard(unittest.TestCase):

  def test_task_registry_discard_removes_task(self):
    """_task_registry_discard removes a mock task from the registry."""
    sentinel = object()
    _task_registry.add(sentinel)  # type: ignore[arg-type]
    self.assertIn(sentinel, _task_registry)
    _task_registry_discard(sentinel)  # type: ignore[arg-type]
    self.assertNotIn(sentinel, _task_registry)

  def test_task_registry_discard_absent_is_noop(self):
    """Discarding a task not in the registry does not raise."""
    sentinel = object()
    _task_registry_discard(sentinel)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# L5: Gather index annotation
# ---------------------------------------------------------------------------


class TestGatherIndexAnnotationSync(unittest.TestCase):

  def test_gather_error_has_index_attribute(self):
    """When a gathered fn raises, the exception has __quent_gather_index__."""
    failing = lambda x: 1 / 0  # noqa: E731
    with self.assertRaises(ZeroDivisionError) as cm:
      Chain(5).then(Chain().gather(lambda x: x, failing)).run()
    self.assertEqual(cm.exception.__quent_gather_index__, 1)

  def test_gather_error_has_fn_attribute(self):
    """When a gathered fn raises, the exception has __quent_gather_fn__."""
    failing = lambda x: 1 / 0  # noqa: E731
    with self.assertRaises(ZeroDivisionError) as cm:
      Chain(5).then(Chain().gather(lambda x: x, failing)).run()
    self.assertIs(cm.exception.__quent_gather_fn__, failing)

  def test_gather_first_fn_fails_index_zero(self):
    """When the first gathered fn raises, __quent_gather_index__ == 0."""
    failing = lambda x: 1 / 0  # noqa: E731
    with self.assertRaises(ZeroDivisionError) as cm:
      Chain(5).then(Chain().gather(failing, lambda x: x)).run()
    self.assertEqual(cm.exception.__quent_gather_index__, 0)


class TestGatherIndexAnnotationAsync(IsolatedAsyncioTestCase):

  async def test_gather_async_error_has_index(self):
    """When a sync fn raises in a gather that also contains async fns,
    the exception still has __quent_gather_index__ and __quent_gather_fn__.
    """
    async def ok(x):
      return x

    failing = lambda x: 1 / 0  # noqa: E731

    # The sync `failing` raises during the setup loop in _gather_op,
    # before _to_async is ever called.
    with self.assertRaises(ZeroDivisionError) as cm:
      await Chain(5).then(Chain().gather(ok, failing)).run()
    self.assertEqual(cm.exception.__quent_gather_index__, 1)
    self.assertIs(cm.exception.__quent_gather_fn__, failing)


# ---------------------------------------------------------------------------
# L12: _get_obj_name falsy name
# ---------------------------------------------------------------------------


class _Obj:
  """Helper: creates an object with arbitrary __name__ / __qualname__."""
  pass


class TestGetObjNameFalsy(unittest.TestCase):

  def test_get_obj_name_empty_string_name(self):
    """An object whose __name__ is '' should return '' (not fall through)."""
    obj = _Obj()
    obj.__name__ = ''  # type: ignore[attr-defined]
    self.assertEqual(_get_obj_name(obj), '')

  def test_get_obj_name_zero_name(self):
    """An object whose __name__ is 0 should return '0' (str conversion)."""
    obj = _Obj()
    obj.__name__ = 0  # type: ignore[attr-defined]
    self.assertEqual(_get_obj_name(obj), '0')

  def test_get_obj_name_false_name(self):
    """An object whose __name__ is False should return 'False'."""
    obj = _Obj()
    obj.__name__ = False  # type: ignore[attr-defined]
    self.assertEqual(_get_obj_name(obj), 'False')

  def test_get_obj_name_normal_name(self):
    """An object with __name__ = 'foo' returns 'foo'."""
    obj = _Obj()
    obj.__name__ = 'foo'  # type: ignore[attr-defined]
    self.assertEqual(_get_obj_name(obj), 'foo')

  def test_get_obj_name_no_name_uses_qualname(self):
    """An object with no __name__ but __qualname__ = 'bar' returns 'bar'."""
    obj = object.__new__(object)  # object() has no __name__ or __qualname__ by default
    # We need an object where getattr(obj, '__name__', None) returns None.
    # Use a simple namespace-like approach.
    class Bare:
      pass
    bare = Bare()
    # Remove __name__ inherited from class — Bare instances don't have instance __name__.
    # getattr(bare, '__name__', None) will check instance then class; class has __name__.
    # So use a custom descriptor-free object.
    class NoName:
      __slots__ = ('__qualname__',)
    obj2 = NoName()
    obj2.__qualname__ = 'bar'
    # NoName class itself has __name__ = 'NoName', but getattr(obj2, '__name__') would
    # find NoName.__name__ on the class. We need to be more careful.
    # Actually, __name__ is a regular attribute on classes, not on instances.
    # For an instance of NoName, getattr(obj2, '__name__', None) returns None
    # because NoName uses __slots__ which doesn't include __name__, and the
    # instance won't inherit the class's __name__ through normal attribute lookup
    # — wait, it will, because __name__ is a class attribute.
    # Let's just use a types.SimpleNamespace and delete __name__.
    import types
    ns = types.SimpleNamespace()
    ns.__qualname__ = 'bar'
    # SimpleNamespace instances don't have __name__ by default.
    self.assertIsNone(getattr(ns, '__name__', None))
    self.assertEqual(_get_obj_name(ns), 'bar')


if __name__ == '__main__':
  unittest.main()
