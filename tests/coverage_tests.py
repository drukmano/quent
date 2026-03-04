import asyncio
import inspect
from unittest import IsolatedAsyncioTestCase
from tests.utils import aempty, await_
from quent import Chain


class BoolMethodTests(IsolatedAsyncioTestCase):
  """Tests for Chain.__bool__() (quent.pyx:725-726)."""

  async def test_chain_is_truthy(self):
    """Chain instances are always truthy."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(1)))
    self.assertTrue(bool(Chain(None)))
    self.assertTrue(bool(Chain(0)))

  async def test_chain_with_operations_is_truthy(self):
    """Chain with operations is still truthy."""
    self.assertTrue(bool(Chain(1).then(lambda v: v)))

class ChainReuseTests(IsolatedAsyncioTestCase):
  """Tests for chain reuse (run/call multiple times)."""

  async def test_chain_run_multiple_times(self):
    """Chain can be run multiple times via .run()."""
    c = Chain(10).then(lambda v: v * 2)
    self.assertEqual(await await_(c.run()), 20)
    self.assertEqual(await await_(c.run()), 20)

  async def test_chain_call_multiple_times(self):
    """Chain can be called multiple times."""
    c = Chain(5).then(lambda v: v + 3)
    self.assertEqual(await await_(c()), 8)
    self.assertEqual(await await_(c()), 8)

  async def test_chain_run_with_root_override(self):
    """Chain .run() can accept a root value."""
    c = Chain().then(lambda v: v * 3)
    self.assertEqual(await await_(c.run(7)), 21)
    self.assertEqual(await await_(c.run(2)), 6)

  async def test_chain_decorator_returns_callable(self):
    """Chain .decorator() returns a callable decorator."""
    decorator = Chain().then(lambda v: v ** 2).decorator()
    self.assertTrue(callable(decorator))

  async def test_chain_reuse_with_async(self):
    """Chain works with async operations and can be reused."""
    c = Chain(aempty, 10).then(lambda v: v + 5)
    self.assertEqual(await c.run(), 15)
    self.assertEqual(await c(), 15)


class EmptyChainTests(IsolatedAsyncioTestCase):
  """Tests for empty chain behavior."""

  async def test_empty_chain_returns_none(self):
    """Chain().run() with no root and no operations returns None."""
    result = await await_(Chain().run())
    self.assertIsNone(result)

  async def test_empty_chain_with_run_value(self):
    """Chain().run(value) uses value as root."""
    result = await await_(Chain().run(42))
    self.assertEqual(result, 42)

  async def test_empty_chain_with_callable_run_value(self):
    """Chain().run(fn) calls fn as root."""
    result = await await_(Chain().run(lambda: 99))
    self.assertEqual(result, 99)

  async def test_empty_chain_bool(self):
    """Empty chain is still truthy."""
    self.assertTrue(bool(Chain()))


class AsyncChainBehaviorTests(IsolatedAsyncioTestCase):
  """Tests for async chain return behavior."""

  async def test_async_chain_returns_awaitable(self):
    """run() on an async chain returns an awaitable, not a Task."""
    async def async_op():
      return 42
    result = Chain(async_op).run()
    self.assertNotIsInstance(result, asyncio.Task)
    self.assertTrue(inspect.isawaitable(result))
    value = await result
    self.assertEqual(value, 42)
