import asyncio
import inspect
import time
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_
from quent import Chain, Cascade, QuentException, run


class RunClassTests(IsolatedAsyncioTestCase):
  """Tests for the `run` class (quent.pyx:790-802)."""

  async def test_run_basic(self):
    """run() as pipe terminator with no arguments."""
    result = await await_(Chain(lambda: 42) | run())
    self.assertEqual(result, 42)

  async def test_run_with_root_value(self):
    """run(value) provides a root value via pipe syntax."""
    result = await await_(Chain() | (lambda v: v + 1) | run(10))
    self.assertEqual(result, 11)

  async def test_run_with_callable_root(self):
    """run(fn, arg) passes callable + arg as root via pipe."""
    result = await await_(Chain() | (lambda v: v * 2) | run(lambda: 5))
    self.assertEqual(result, 10)

  async def test_run_pipe_equivalence(self):
    """Verify pipe+run is equivalent to .run()."""
    chain_result = await await_(Chain(1).then(lambda v: v + 1).run())
    pipe_result = await await_(Chain(1) | (lambda v: v + 1) | run())
    self.assertEqual(chain_result, pipe_result)

  async def test_run_with_async(self):
    """run() works with async chains."""
    result = await (Chain(aempty, 5) | (lambda v: v * 3) | run())
    self.assertEqual(result, 15)


class SleepMethodTests(IsolatedAsyncioTestCase):
  """Tests for Chain.sleep() (quent.pyx:659-662)."""

  async def test_sleep_preserves_value(self):
    """sleep() does not alter the chain value (ignore_result=True)."""
    result = await await_(Chain(42).sleep(0.01).run())
    self.assertEqual(result, 42)

  async def test_sleep_async_preserves_value(self):
    """sleep() in an async context preserves value."""
    result = await Chain(aempty, 99).sleep(0.01).run()
    self.assertEqual(result, 99)

  async def test_sleep_in_chain(self):
    """sleep() can be chained between operations."""
    result = await await_(Chain(10).then(lambda v: v * 2).sleep(0.01).then(lambda v: v + 1).run())
    self.assertEqual(result, 21)


class NullMethodTests(IsolatedAsyncioTestCase):
  """Tests for Chain.null() classmethod (quent.pyx:664-666)."""

  async def test_null_returns_sentinel(self):
    """Chain.null() returns the Null sentinel."""
    from quent import Null
    self.assertIs(Chain.null(), Null)

  async def test_null_is_consistent(self):
    """Chain.null() always returns the same object."""
    self.assertIs(Chain.null(), Chain.null())


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

  async def test_cascade_is_truthy(self):
    """Cascade instances are always truthy."""
    self.assertTrue(bool(Cascade()))
    self.assertTrue(bool(Cascade(1)))


class FrozenChainTests(IsolatedAsyncioTestCase):
  """Tests for _FrozenChain (quent.pyx:770-787), accessed via Chain.freeze()."""

  async def test_freeze_run(self):
    """Frozen chain can be run multiple times via .run()."""
    frozen = Chain(10).then(lambda v: v * 2).freeze()
    self.assertEqual(await await_(frozen.run()), 20)
    self.assertEqual(await await_(frozen.run()), 20)

  async def test_freeze_call(self):
    """Frozen chain can be called."""
    frozen = Chain(5).then(lambda v: v + 3).freeze()
    self.assertEqual(await await_(frozen()), 8)
    self.assertEqual(await await_(frozen()), 8)

  async def test_freeze_with_root_override(self):
    """Frozen chain .run() can accept a root value."""
    frozen = Chain().then(lambda v: v * 3).freeze()
    self.assertEqual(await await_(frozen.run(7)), 21)
    self.assertEqual(await await_(frozen.run(2)), 6)

  async def test_freeze_decorator_returns_callable(self):
    """Frozen chain .decorator() returns a callable decorator."""
    decorator = Chain().then(lambda v: v ** 2).decorator()
    self.assertTrue(callable(decorator))

  async def test_freeze_with_async_chain(self):
    """Frozen chain works with async operations."""
    frozen = Chain(aempty, 10).then(lambda v: v + 5).freeze()
    self.assertEqual(await frozen.run(), 15)
    self.assertEqual(await frozen(), 15)


class EmptyChainTests(IsolatedAsyncioTestCase):
  """Tests for empty chain behavior."""

  async def test_empty_chain_returns_none(self):
    """Chain().run() with no root and no operations returns None."""
    result = await await_(Chain().run())
    self.assertIsNone(result)

  async def test_empty_cascade_returns_none(self):
    """Cascade().run() with no root and no operations returns None."""
    result = Cascade().run()
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


class AutorunSimpleChainTests(IsolatedAsyncioTestCase):
  """Tests for autorun behavior on simple chains (quent.pyx __call__ and run() autorun paths)."""

  async def test_autorun_call_simple_chain(self):
    """Autorun via __call__() on a simple chain wraps the result in a Task."""
    result = [None]
    async def async_op():
      await asyncio.sleep(0.05)
      result[0] = 'done'
    Chain(async_op).autorun()()
    self.assertIsNone(result[0])
    await asyncio.sleep(0.15)
    self.assertEqual(result[0], 'done')

  async def test_autorun_run_simple_chain(self):
    """Autorun via run() on a simple chain wraps the result in a Task."""
    result = [None]
    async def async_op():
      await asyncio.sleep(0.05)
      result[0] = 'done'
    Chain(async_op).autorun().run()
    self.assertIsNone(result[0])
    await asyncio.sleep(0.15)
    self.assertEqual(result[0], 'done')

  async def test_autorun_run_with_context(self):
    """Autorun via run() with with_context() wraps the async result in a Task."""
    result = [None]
    async def async_op():
      await asyncio.sleep(0.05)
      result[0] = 'done'
    Chain(async_op).autorun().with_context(key='value').run()
    self.assertIsNone(result[0])
    await asyncio.sleep(0.15)
    self.assertEqual(result[0], 'done')

  async def test_autorun_disabled_returns_coroutine(self):
    """Without autorun, run() on an async chain returns an awaitable, not a Task."""
    async def async_op():
      return 42
    result = Chain(async_op).run()
    self.assertNotIsInstance(result, asyncio.Task)
    self.assertTrue(inspect.isawaitable(result))
    value = await result
    self.assertEqual(value, 42)
