import asyncio
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException


class MyTestCase(IsolatedAsyncioTestCase):
  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)

  async def assertTrue(self, expr, msg=None):
    return super().assertTrue(await await_(expr), msg)

  async def assertFalse(self, expr, msg=None):
    return super().assertFalse(await await_(expr), msg)

  async def assertEqual(self, first, second, msg=None):
    return super().assertEqual(await await_(first), second, msg)

  async def assertIsNone(self, obj, msg=None):
    return super().assertIsNone(await await_(obj), msg)

  async def assertIs(self, expr1, expr2, msg=None):
    return super().assertIs(await await_(expr1), expr2, msg)

  async def assertIsNot(self, expr1, expr2, msg=None):
    return super().assertIsNot(await await_(expr1), expr2, msg)


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
# 1. while_true: break returning coroutine
# ---------------------------------------------------------------------------
class WhileTrueBreakCoroTests(MyTestCase):

  async def test_while_break_coro_value(self):
    """Break value is a coroutine in while_true_async path."""
    counter = {'count': 0}
    def f():
      counter['count'] += 1
      if counter['count'] == 1:
        return aempty()
      return Chain.break_(aempty, 42)
    counter['count'] = 0
    await self.assertEqual(
      Chain().while_true(f, ...).run(), 42
    )


# ---------------------------------------------------------------------------
# 2-5. foreach: break with coroutine + exception paths
# ---------------------------------------------------------------------------
class ForeachBreakCoroTests(MyTestCase):

  async def test_foreach_break_coro_sync_to_async(self):
    """Break value is coroutine in foreach_async (sync iterator, transition)."""
    counter = {'count': 0}
    def f(el):
      counter['count'] += 1
      if counter['count'] == 1:
        return aempty(el * 10)
      return Chain.break_(aempty, 'done')
    counter['count'] = 0
    await self.assertEqual(
      Chain([1, 2, 3]).foreach(f).run(), 'done'
    )

  async def test_foreach_break_coro_async_iterator(self):
    """Break value is coroutine in async_foreach."""
    def f(el):
      if el >= 2:
        return Chain.break_(aempty, 'done')
      return el * 10
    await self.assertEqual(
      Chain(AsyncIterator([1, 2, 3])).foreach(f).run(), 'done'
    )

  async def test_foreach_exception_async_iterator(self):
    """Exception in async_foreach sets link.temp_args."""
    def f(el):
      if el >= 2:
        raise ValueError('async foreach error')
      return el * 10
    with self.assertRaises(ValueError):
      await await_(
        Chain(AsyncIterator([1, 2, 3])).foreach(f).run()
      )

  async def test_foreach_exception_sync_to_async(self):
    """Exception in foreach_async sets link.temp_args."""
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
# 6-8. foreach_indexed: break with coroutine + exception paths
# ---------------------------------------------------------------------------
class ForeachIndexedBreakCoroTests(MyTestCase):

  async def test_foreach_indexed_break_coro_sync_to_async(self):
    """Break value is coroutine in foreach_indexed_async (transition)."""
    counter = {'count': 0}
    def f(idx, el):
      counter['count'] += 1
      if counter['count'] == 1:
        return aempty((idx, el))
      return Chain.break_(aempty, 'done')
    counter['count'] = 0
    await self.assertEqual(
      Chain(['a', 'b', 'c']).foreach(f, with_index=True).run(), 'done'
    )

  async def test_foreach_indexed_break_coro_async_iterator(self):
    """Break value is coroutine in async_foreach_indexed."""
    def f(idx, el):
      if idx >= 1:
        return Chain.break_(aempty, 'done')
      return (idx, el)
    await self.assertEqual(
      Chain(AsyncIterator(['a', 'b', 'c'])).foreach(f, with_index=True).run(), 'done'
    )

  async def test_foreach_indexed_exception_async_iterator(self):
    """Exception in async_foreach_indexed sets link.temp_args."""
    def f(idx, el):
      if idx >= 1:
        raise ValueError('indexed async error')
      return (idx, el)
    with self.assertRaises(ValueError):
      await await_(
        Chain(AsyncIterator(['a', 'b', 'c'])).foreach(f, with_index=True).run()
      )


# ---------------------------------------------------------------------------
# 9-10. filter: exception paths
# ---------------------------------------------------------------------------
class FilterExceptionTests(MyTestCase):

  async def test_filter_exception_async_iterator(self):
    """Exception in async_filter sets link.temp_args."""
    def pred(x):
      if x >= 2:
        raise ValueError('async filter error')
      return True
    with self.assertRaises(ValueError):
      await await_(
        Chain(AsyncIterator([1, 2, 3])).filter(pred).run()
      )

  async def test_filter_exception_sync_to_async(self):
    """Exception in filter_async sets link.temp_args."""
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
# 11. reduce: exception paths
# ---------------------------------------------------------------------------
class ReduceExceptionTests(MyTestCase):

  async def test_reduce_exception_async_iterator(self):
    """Exception in async_reduce sets link.temp_args."""
    def reducer(a, x):
      if x >= 2:
        raise ValueError('async reduce error')
      return a + x
    with self.assertRaises(ValueError):
      await await_(
        Chain(AsyncIterator([1, 2, 3])).reduce(reducer, 0).run()
      )

  async def test_reduce_exception_sync_to_async(self):
    """Exception in reduce_async sets link.temp_args."""
    counter = {'count': 0}
    def reducer(a, x):
      counter['count'] += 1
      if counter['count'] == 1:
        return aempty(a + x)
      raise ValueError('reduce transition error')
    counter['count'] = 0
    with self.assertRaises(ValueError):
      await await_(
        Chain([1, 2, 3, 4]).reduce(reducer, 0).run()
      )


# ---------------------------------------------------------------------------
# 12-14. context manager edge cases
# ---------------------------------------------------------------------------
class ContextManagerEdgeCaseTests(MyTestCase):

  async def test_async_cm_with_async_body(self):
    """AsyncCM with body fn returning a coroutine."""
    cm = AsyncCM('ctx_val')
    async def async_body(ctx):
      return f'async_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'async_ctx_val')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_normal(self):
    """Sync CM whose __exit__ returns a coroutine, body succeeds."""
    cm = SyncCMWithCoroExit('val')
    async def async_body(ctx):
      return f'got_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    await self.assertEqual(result, 'got_val')
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_exception(self):
    """Sync CM whose __exit__ returns a coroutine, body raises."""
    cm = SyncCMWithCoroExit('val', suppress=False)
    async def async_body_raises(ctx):
      raise TestExc('body error')
    with self.assertRaises(TestExc):
      await Chain(cm).with_(async_body_raises).run()
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_suppress(self):
    """Sync CM whose __exit__ returns a coroutine that suppresses."""
    cm = SyncCMWithCoroExit('val', suppress=True)
    async def async_body_raises(ctx):
      raise TestExc('suppressed')
    result = await Chain(cm).with_(async_body_raises).run()
    await self.assertIsNone(result)
    super(MyTestCase, self).assertTrue(cm.exited)


# ---------------------------------------------------------------------------
# 15-16. iterate (generator) break
# ---------------------------------------------------------------------------
class IterateBreakTests(MyTestCase):

  async def test_iterate_break_sync_generator(self):
    """Chain.break_() inside sync iterate() stops the generator."""
    def f(i):
      if i >= 3:
        return Chain.break_()
      return i * 2
    r = []
    for i in Chain(SyncIterator([0, 1, 2, 3, 4])).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [0, 2, 4])

  async def test_iterate_break_async_generator(self):
    """Chain.break_() inside async iterate() stops the async generator."""
    def f(i):
      if i >= 3:
        return Chain.break_()
      return i * 2
    r = []
    async for i in Chain(AsyncIterator([0, 1, 2, 3, 4])).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [0, 2, 4])


# ---------------------------------------------------------------------------
# 17-18. context variable management for async chains
# ---------------------------------------------------------------------------
class ContextVarTests(MyTestCase):

  async def test_async_chain_with_context(self):
    """Async chain with .with_context() exercises token set/reset."""
    async def check_context(v):
      ctx = Chain.get_context()
      return ctx.get('key')
    c = Chain(1).then(check_context).with_context(key='async_val')
    result = await c.run()
    await self.assertEqual(result, 'async_val')
    super(MyTestCase, self).assertEqual(Chain.get_context(), {})

  async def test_call_with_context(self):
    """chain() instead of chain.run() when _context is set."""
    def check(v):
      return Chain.get_context().get('key')
    c = Chain(1).then(check).with_context(key='call_val')
    result = c()
    await self.assertEqual(result, 'call_val')

  async def test_async_call_with_context(self):
    """Async chain via __call__ with context."""
    async def check(v):
      return Chain.get_context().get('key')
    c = Chain(1).then(check).with_context(key='async_call')
    result = await c()
    await self.assertEqual(result, 'async_call')


# ---------------------------------------------------------------------------
# 19-20. _run_async: _Return with coroutine value + _Break in nested chain
# (chains must be non-simple to reach _run_async instead of _run_async_simple)
# ---------------------------------------------------------------------------
class RunAsyncControlFlowTests(MyTestCase):

  async def test_return_coro_value_in_async(self):
    """Chain.return_(aempty, val) in _run_async path awaits the coro result."""
    def body(v):
      Chain.return_(aempty, 42)
    # .do() makes _is_simple=False so _run is used instead of _run_simple
    result = await Chain(aempty, 1).do(lambda v: None).then(body).run()
    await self.assertEqual(result, 42)

  async def test_break_in_nested_async_chain(self):
    """Chain.break_() in nested chain within while_true, _run_async path."""
    def body():
      Chain.break_()
    # .do() makes the inner chain non-simple → uses _run/_run_async
    inner = Chain().then(aempty).do(lambda v: None).then(body, ...)
    result = await Chain(99).while_true(inner, ...).run()
    await self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# 21. on_success async exception path
# ---------------------------------------------------------------------------
class OnSuccessAsyncExcTests(MyTestCase):

  async def test_on_success_async_raises(self):
    """Async on_success raises exercises _run_async_on_success exception."""
    async def failing_cb(v):
      raise ValueError('on_success async error')
    c = Chain(42).set_async().on_success(failing_cb)
    with self.assertRaises(ValueError):
      await await_(c.run())

  async def test_on_success_async_raises_with_finally(self):
    """Async on_success raises, finally handler still executes."""
    finally_called = [False]
    async def failing_cb(v):
      raise ValueError('on_success error')
    def finally_cb(v):
      finally_called[0] = True
    c = Chain(42).set_async().on_success(failing_cb).finally_(finally_cb)
    with self.assertRaises(ValueError):
      await await_(c.run())
    super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_on_success_async_raises_with_async_finally(self):
    """Async on_success raises, async finally handler awaited."""
    finally_called = [False]
    async def failing_cb(v):
      raise ValueError('on_success error')
    async def async_finally_cb(v):
      finally_called[0] = True
    c = Chain(42).set_async().on_success(failing_cb).finally_(async_finally_cb)
    with self.assertRaises(ValueError):
      await await_(c.run())
    super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_on_success_ok_async_finally_raises(self):
    """on_success succeeds, finally handler raises in _run_async_on_success."""
    async def ok_cb(v):
      pass
    def failing_finally(v):
      raise RuntimeError('finally error')
    c = Chain(42).set_async().on_success(ok_cb).finally_(failing_finally)
    with self.assertRaises(RuntimeError):
      await await_(c.run())

  async def test_on_success_ok_async_finally_control_flow(self):
    """on_success succeeds, finally uses control flow signal."""
    async def ok_cb(v):
      pass
    c = Chain(42).set_async().on_success(ok_cb).finally_(Chain.return_, 99)
    with self.assertRaises(QuentException) as cm:
      await await_(c.run())
    super(MyTestCase, self).assertIn(
      'control flow signals', str(cm.exception).lower()
    )


# ---------------------------------------------------------------------------
# 22. frozen chain .decorator()
# ---------------------------------------------------------------------------
class FrozenChainDecoratorTests(MyTestCase):

  async def test_frozen_chain_decorator(self):
    """Chain().then(fn).freeze().decorator() applied to a function."""
    decorator = Chain().then(lambda v: v * 2).freeze().decorator()
    @decorator
    def my_fn(x):
      return x
    result = my_fn(5)
    await self.assertEqual(result, 10)

  async def test_chain_decorator_shorthand(self):
    """Chain().then(fn).decorator() shorthand."""
    decorator = Chain().then(lambda v: v + 10).decorator()
    @decorator
    def my_fn(x):
      return x
    result = my_fn(5)
    await self.assertEqual(result, 15)


if __name__ == '__main__':
  import unittest
  unittest.main()
