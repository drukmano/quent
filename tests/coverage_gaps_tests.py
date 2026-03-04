from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain


class AsyncIterator:
  def __init__(self, items=None):
    self._items = list(items) if items is not None else list(range(10))

  def __aiter__(self):
    self._iter = iter(self._items)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


class SyncIterator:
  def __init__(self, items=None):
    self._items = items if items is not None else list(range(10))

  def __iter__(self):
    return iter(self._items)


class AsyncCM:
  def __init__(self, value='async_ctx'):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    return False


class SyncCMWithCoroExit:
  """Sync CM whose __exit__ returns a coroutine."""
  def __init__(self, value='ctx', suppress=False):
    self.value = value
    self.suppress = suppress
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    return aempty(self.suppress)


# ---------------------------------------------------------------------------
# foreach: break with coroutine + exception paths
# ---------------------------------------------------------------------------
class ForeachBreakCoroTests(IsolatedAsyncioTestCase):

  async def test_foreach_break_coro_sync_to_async(self):
    """Break value is coroutine in _foreach_to_async (sync iterator, transition)."""
    counter = {'count': 0}
    def f(el):
      counter['count'] += 1
      if counter['count'] == 1:
        return aempty(el * 10)
      return Chain.break_(aempty, 'done')
    counter['count'] = 0
    self.assertEqual(
      await Chain([1, 2, 3]).foreach(f).run(), 'done'
    )

  async def test_foreach_break_coro_async_iterator(self):
    """Break value is coroutine in _foreach_full_async."""
    def f(el):
      if el >= 2:
        return Chain.break_(aempty, 'done')
      return el * 10
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3])).foreach(f).run(), 'done'
    )

  async def test_foreach_exception_async_iterator(self):
    """Exception in _foreach_full_async sets link.temp_args."""
    def f(el):
      if el >= 2:
        raise ValueError('async foreach error')
      return el * 10
    with self.assertRaises(ValueError):
      await await_(
        Chain(AsyncIterator([1, 2, 3])).foreach(f).run()
      )

  async def test_foreach_exception_sync_to_async(self):
    """Exception in _foreach_to_async sets link.temp_args."""
    counter = {'count': 0}
    def f(el):
      counter['count'] += 1
      if counter['count'] == 1:
        return aempty(el * 10)
      raise ValueError('transition error')
    counter['count'] = 0
    with self.assertRaises(ValueError):
      await await_(
        Chain([1, 2, 3]).foreach(f).run()
      )


# ---------------------------------------------------------------------------
# filter: exception paths
# ---------------------------------------------------------------------------
class FilterExceptionTests(IsolatedAsyncioTestCase):

  async def test_filter_exception_async_iterator(self):
    """Exception in _filter_full_async sets link.temp_args."""
    def pred(x):
      if x >= 2:
        raise ValueError('async filter error')
      return True
    with self.assertRaises(ValueError):
      await await_(
        Chain(AsyncIterator([1, 2, 3])).filter(pred).run()
      )

  async def test_filter_exception_sync_to_async(self):
    """Exception in _filter_to_async sets link.temp_args."""
    counter = {'count': 0}
    def pred(x):
      counter['count'] += 1
      if counter['count'] == 1:
        return aempty(True)
      raise ValueError('filter transition error')
    counter['count'] = 0
    with self.assertRaises(ValueError):
      await await_(
        Chain([1, 2, 3]).filter(pred).run()
      )


# ---------------------------------------------------------------------------
# context manager edge cases
# ---------------------------------------------------------------------------
class ContextManagerEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_async_cm_with_async_body(self):
    """AsyncCM with body fn returning a coroutine."""
    cm = AsyncCM('ctx_val')
    async def async_body(ctx):
      return f'async_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'async_ctx_val')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_normal(self):
    """Sync CM whose __exit__ returns a coroutine, body succeeds."""
    cm = SyncCMWithCoroExit('val')
    async def async_body(ctx):
      return f'got_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'got_val')
    self.assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_exception(self):
    """Sync CM whose __exit__ returns a coroutine, body raises."""
    cm = SyncCMWithCoroExit('val', suppress=False)
    async def async_body_raises(ctx):
      raise TestExc('body error')
    with self.assertRaises(TestExc):
      await Chain(cm).with_(async_body_raises).run()
    self.assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_suppress(self):
    """Sync CM whose __exit__ returns a coroutine that suppresses."""
    cm = SyncCMWithCoroExit('val', suppress=True)
    async def async_body_raises(ctx):
      raise TestExc('suppressed')
    result = await Chain(cm).with_(async_body_raises).run()
    self.assertIsNone(result)
    self.assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# iterate (generator) break
# ---------------------------------------------------------------------------
class IterateBreakTests(IsolatedAsyncioTestCase):

  async def test_iterate_break_sync_generator(self):
    """Chain.break_() inside sync iterate() stops the generator."""
    def f(i):
      if i >= 3:
        return Chain.break_()
      return i * 2
    r = []
    for i in Chain(SyncIterator([0, 1, 2, 3, 4])).iterate(f):
      r.append(i)
    self.assertEqual(r, [0, 2, 4])

  async def test_iterate_break_async_generator(self):
    """Chain.break_() inside async iterate() stops the async generator."""
    def f(i):
      if i >= 3:
        return Chain.break_()
      return i * 2
    r = []
    async for i in Chain(AsyncIterator([0, 1, 2, 3, 4])).iterate(f):
      r.append(i)
    self.assertEqual(r, [0, 2, 4])


# ---------------------------------------------------------------------------
# _run_async: _Return with coroutine value
# (chain must be non-simple to reach _run_async instead of _run_async_simple)
# ---------------------------------------------------------------------------
class RunAsyncControlFlowTests(IsolatedAsyncioTestCase):

  async def test_return_coro_value_in_async(self):
    """Chain.return_(aempty, val) in _run_async path awaits the coro result."""
    def body(v):
      Chain.return_(aempty, 42)
    # .do() makes _is_simple=False so _run is used instead of _run_simple
    result = await Chain(aempty, 1).do(lambda v: None).then(body).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# chain .decorator()
# ---------------------------------------------------------------------------
class ChainDecoratorTests(IsolatedAsyncioTestCase):

  async def test_chain_decorator(self):
    """Chain().then(fn).decorator() applied to a function."""
    decorator = Chain().then(lambda v: v * 2).decorator()
    @decorator
    def my_fn(x):
      return x
    result = my_fn(5)
    self.assertEqual(result, 10)

  async def test_chain_decorator_shorthand(self):
    """Chain().then(fn).decorator() shorthand."""
    decorator = Chain().then(lambda v: v + 10).decorator()
    @decorator
    def my_fn(x):
      return x
    result = my_fn(5)
    self.assertEqual(result, 15)


if __name__ == '__main__':
  import unittest
  unittest.main()
