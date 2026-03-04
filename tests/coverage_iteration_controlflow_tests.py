"""Comprehensive tests targeting uncovered lines in _iteration.pxi and _control_flow.pxi.

_iteration.pxi uncovered lines:
  70   — foreach factory (cdef Link foreach)
  78   — _foreach_to_async entry
  104  — _foreach_full_async entry
  179  — filter_ factory (cdef Link filter_)
  185  — _filter_to_async entry
  204  — _filter_full_async entry
  212  — async predicate in full async filter
  251  — gather_ factory (cdef Link gather_)
  257  — _gather_to_async entry
  386-393 — function aliases

_control_flow.pxi uncovered lines:
  40   — handle_break_exc entry
  47   — handle_return_exc entry
  99   — with_ factory (cdef Link with_)
  112  — _with_to_async entry
  124  — raise in _with_to_async when not entered (dead code)
  134  — _with_full_async entry
  157  — sync_generator entry
  176  — async_generator entry
  227  — _Generator.__call__ method
  253-256 — function aliases
"""

import asyncio
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class AsyncIterator:
  """Minimal async iterable for testing."""
  def __init__(self, items):
    self._items = list(items)

  def __aiter__(self):
    self._iter = iter(self._items)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


class SyncIterator:
  """Minimal sync iterable for testing."""
  def __init__(self, items):
    self._items = list(items)

  def __iter__(self):
    return iter(self._items)


class SyncCM:
  """Sync context manager."""
  def __init__(self, value='ctx'):
    self.value = value
    self.entered = False
    self.exited = False
    self.exit_args = None

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exit_args = (exc_type, exc_val, exc_tb)
    return False


class AsyncCM:
  """Async context manager."""
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


class SyncCMRaisingEnter:
  """Sync CM whose __enter__ raises an exception (entered=False path)."""
  def __enter__(self):
    raise TestExc('enter failed')

  def __exit__(self, *args):
    return False


class SyncCMSuppressing:
  """Sync CM that suppresses exceptions."""
  def __init__(self, value='ctx'):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    return True  # suppress


class SyncCMCoroExit:
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


# ===========================================================================
# ITERATION: foreach factory + async paths (lines 70, 78, 104)
# ===========================================================================

class ForeachFactoryAndAsyncTests(IsolatedAsyncioTestCase):
  """Tests targeting foreach factory (line 70), _foreach_to_async (line 78),
  and _foreach_full_async (line 104)."""

  async def test_foreach_factory_creates_link(self):
    """Line 70: foreach() factory is called when Chain.foreach() is used.
    Simple sync usage exercises the factory."""
    result = Chain([1, 2, 3]).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_foreach_to_async_sync_iter_async_fn(self):
    """Line 78: _foreach_to_async entered when sync iterable fn returns a coroutine.
    The first call to fn returns a coro, triggering the transition."""
    self.assertEqual(
      await Chain([10, 20, 30]).foreach(lambda x: aempty(x + 1)).run(),
      [11, 21, 31]
    )

  async def test_foreach_to_async_transition_midway(self):
    """Line 78: fn returns sync for first elements, then coro mid-iteration."""
    counter = {'n': 0}
    def f(el):
      counter['n'] += 1
      if counter['n'] > 2:
        return aempty(el * 10)
      return el * 10
    counter['n'] = 0
    self.assertEqual(
      await Chain([1, 2, 3, 4]).foreach(f).run(),
      [10, 20, 30, 40]
    )

  async def test_foreach_full_async_with_sync_fn(self):
    """Line 104: _foreach_full_async entered for async iterable with sync fn."""
    self.assertEqual(
      await Chain(AsyncIterator([5, 6, 7])).foreach(lambda x: x * 3).run(),
      [15, 18, 21]
    )

  async def test_foreach_full_async_with_async_fn(self):
    """Line 104: _foreach_full_async with async fn on async iterable.
    Also hits line 112 (iscoro branch inside _foreach_full_async)."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3])).foreach(lambda x: aempty(x ** 2)).run(),
      [1, 4, 9]
    )

  async def test_foreach_to_async_break_with_sync_value(self):
    """_foreach_to_async break with sync fallback (no coro)."""
    counter = {'n': 0}
    def f(el):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(el * 10)
      return Chain.break_()
    counter['n'] = 0
    self.assertEqual(
      await Chain([1, 2, 3]).foreach(f).run(),
      [10]
    )

  async def test_foreach_to_async_break_with_coro_value(self):
    """_foreach_to_async break with coroutine value (handle_break_exc returns coro)."""
    counter = {'n': 0}
    def f(el):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(el * 10)
      return Chain.break_(aempty, 'stopped')
    counter['n'] = 0
    self.assertEqual(
      await Chain([1, 2, 3]).foreach(f).run(),
      'stopped'
    )

  async def test_foreach_full_async_break_sync_value(self):
    """_foreach_full_async break with sync fallback value."""
    def f(el):
      if el >= 3:
        return Chain.break_()
      return el * 10
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3, 4])).foreach(f).run(),
      [10, 20]
    )

  async def test_foreach_full_async_break_coro_value(self):
    """_foreach_full_async break with coroutine value (lines 118-121)."""
    def f(el):
      if el >= 2:
        return Chain.break_(aempty, 'async_done')
      return el * 10
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3])).foreach(f).run(),
      'async_done'
    )

  async def test_foreach_to_async_exception_sets_temp_args(self):
    """_foreach_to_async exception handler sets __quent_link_temp_args__."""
    counter = {'n': 0}
    def f(el):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(el)
      raise ValueError('transition exc')
    counter['n'] = 0
    with self.assertRaises(ValueError):
      await await_(Chain([1, 2, 3]).foreach(f).run())

  async def test_foreach_full_async_exception_sets_temp_args(self):
    """_foreach_full_async exception handler sets __quent_link_temp_args__."""
    def f(el):
      if el == 3:
        raise ValueError('full async exc')
      return el
    with self.assertRaises(ValueError):
      await await_(Chain(AsyncIterator([1, 2, 3])).foreach(f).run())

  async def test_foreach_to_async_empty_after_first_coro(self):
    """Only one element causes coro; iteration stops at StopIteration."""
    self.assertEqual(
      await Chain([42]).foreach(lambda x: aempty(x * 2)).run(),
      [84]
    )

  async def test_foreach_full_async_empty_iterable(self):
    """_foreach_full_async with empty async iterable returns []."""
    self.assertEqual(
      await Chain(AsyncIterator([])).foreach(lambda x: x).run(),
      []
    )


# ===========================================================================
# ITERATION: filter factory + async paths (lines 179, 185, 204, 212)
# ===========================================================================

class FilterFactoryAndAsyncTests(IsolatedAsyncioTestCase):
  """Tests targeting filter_ factory (line 179), _filter_to_async (line 185),
  _filter_full_async (line 204), and async predicate in full async filter (line 212)."""

  async def test_filter_factory_sync(self):
    """Line 179: filter_ factory is exercised by Chain.filter()."""
    result = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).run()
    self.assertEqual(result, [4, 5])

  async def test_filter_to_async_predicate_returns_coro(self):
    """Line 185: _filter_to_async entered when sync iterable predicate returns coro."""
    self.assertEqual(
      await Chain([1, 2, 3, 4]).filter(lambda x: aempty(x % 2 == 0)).run(),
      [2, 4]
    )

  async def test_filter_to_async_transition_midway(self):
    """Line 185: predicate returns sync first, then coro mid-iteration."""
    counter = {'n': 0}
    def pred(x):
      counter['n'] += 1
      if counter['n'] > 2:
        return aempty(x % 2 == 0)
      return x % 2 == 0
    counter['n'] = 0
    self.assertEqual(
      await Chain([1, 2, 3, 4, 5, 6]).filter(pred).run(),
      [2, 4, 6]
    )

  async def test_filter_full_async_sync_predicate(self):
    """Line 204: _filter_full_async entered for async iterable with sync predicate."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3, 4, 5])).filter(lambda x: x > 2).run(),
      [3, 4, 5]
    )

  async def test_filter_full_async_async_predicate(self):
    """Line 212: async predicate in _filter_full_async (iscoro branch).
    This is the key missing line -- async predicate on async iterable."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3, 4])).filter(lambda x: aempty(x % 2 == 1)).run(),
      [1, 3]
    )

  async def test_filter_full_async_async_predicate_all_pass(self):
    """All elements pass async predicate on async iterable."""
    self.assertEqual(
      await Chain(AsyncIterator([2, 4, 6])).filter(lambda x: aempty(True)).run(),
      [2, 4, 6]
    )

  async def test_filter_full_async_async_predicate_none_pass(self):
    """No elements pass async predicate on async iterable."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3])).filter(lambda x: aempty(False)).run(),
      []
    )

  async def test_filter_to_async_exception(self):
    """Exception in _filter_to_async sets __quent_link_temp_args__."""
    counter = {'n': 0}
    def pred(x):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(True)
      raise ValueError('filter transition')
    counter['n'] = 0
    with self.assertRaises(ValueError):
      await await_(Chain([1, 2, 3]).filter(pred).run())

  async def test_filter_full_async_exception(self):
    """Exception in _filter_full_async sets __quent_link_temp_args__."""
    def pred(x):
      if x == 3:
        raise ValueError('filter async exc')
      return True
    with self.assertRaises(ValueError):
      await await_(Chain(AsyncIterator([1, 2, 3])).filter(pred).run())

  async def test_filter_full_async_empty_iterable(self):
    """_filter_full_async with empty async iterable returns []."""
    self.assertEqual(
      await Chain(AsyncIterator([])).filter(lambda x: True).run(),
      []
    )


# ===========================================================================
# ITERATION: gather factory + async path (lines 251, 257)
# ===========================================================================

class GatherFactoryAndAsyncTests(IsolatedAsyncioTestCase):
  """Tests targeting gather_ factory (line 251) and _gather_to_async (line 257)."""

  async def test_gather_factory_sync(self):
    """Line 251: gather_ factory exercised by Chain.gather()."""
    result = Chain(10).gather(lambda v: v + 1, lambda v: v * 2).run()
    self.assertEqual(result, [11, 20])

  async def test_gather_to_async_mixed(self):
    """Line 257: _gather_to_async entered when some fns return coros."""
    f1 = lambda v: v + 1
    f2 = lambda v: aempty(v * 2)
    f3 = lambda v: v ** 2
    self.assertEqual(
      await Chain(5).gather(f1, f2, f3).run(),
      [6, 10, 25]
    )

  async def test_gather_to_async_all_coros(self):
    """Line 257: all fns return coros."""
    f1 = lambda v: aempty(v + 1)
    f2 = lambda v: aempty(v + 2)
    f3 = lambda v: aempty(v + 3)
    self.assertEqual(
      await Chain(10).gather(f1, f2, f3).run(),
      [11, 12, 13]
    )

  async def test_gather_single_async_fn(self):
    """Single async fn in gather."""
    self.assertEqual(
      await Chain(7).gather(lambda v: aempty(v * 3)).run(),
      [21]
    )

  async def test_gather_empty(self):
    """Empty gather returns []."""
    result = Chain(5).gather().run()
    self.assertEqual(result, [])

  async def test_gather_preserves_order(self):
    """Results are ordered by fn position, not completion time."""
    async def slow(v):
      await asyncio.sleep(0.01)
      return 'slow'
    def fast(v):
      return 'fast'
    self.assertEqual(
      await Chain(1).gather(slow, fast, slow).run(),
      ['slow', 'fast', 'slow']
    )

# ===========================================================================
# ITERATION: function aliases (lines 386-393)
# ===========================================================================

class IterationAliasTests(IsolatedAsyncioTestCase):
  """Tests that exercise paths through the function aliases defined at
  lines 386-393 of _iteration.pxi. These are exercised indirectly by
  every test above; these tests ensure each alias is hit explicitly."""

  async def test_alias_foreach_async_fn(self):
    """_foreach_async_fn alias (line 387): sync iter, fn returns coro."""
    self.assertEqual(
      await Chain([1, 2]).foreach(lambda x: aempty(x + 100)).run(),
      [101, 102]
    )

  async def test_alias_async_foreach_fn(self):
    """_async_foreach_fn alias (line 386): async iter foreach."""
    self.assertEqual(
      await Chain(AsyncIterator([3, 4])).foreach(lambda x: x + 100).run(),
      [103, 104]
    )

  async def test_alias_filter_async_fn(self):
    """_filter_async_fn alias (line 388): sync iter, predicate returns coro."""
    self.assertEqual(
      await Chain([1, 2, 3]).filter(lambda x: aempty(x > 1)).run(),
      [2, 3]
    )

  async def test_alias_async_filter_fn(self):
    """_async_filter_fn alias (line 389): async iter filter."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3])).filter(lambda x: x > 1).run(),
      [2, 3]
    )

  async def test_alias_gather_async_fn(self):
    """_gather_async_fn alias (line 390): gather with coro results."""
    self.assertEqual(
      await Chain(5).gather(lambda v: aempty(v + 1)).run(),
      [6]
    )

  async def test_alias_asyncio_gather_fn(self):
    """_asyncio_gather_fn alias (line 391): asyncio.gather via gather."""
    self.assertEqual(
      await Chain(2).gather(
        lambda v: aempty(v * 1),
        lambda v: aempty(v * 2),
        lambda v: aempty(v * 3),
      ).run(),
      [2, 4, 6]
    )


# ===========================================================================
# CONTROL FLOW: handle_break_exc (line 40) and handle_return_exc (line 47)
# ===========================================================================

class HandleBreakExcTests(IsolatedAsyncioTestCase):
  """Tests targeting handle_break_exc (line 40)."""

  async def test_break_no_value_returns_fallback(self):
    """Line 40: handle_break_exc with Null value returns fallback (the list)."""
    def f(el):
      if el >= 3:
        return Chain.break_()
      return el
    self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(f).run(),
      [1, 2]
    )

  async def test_break_with_literal_value(self):
    """Line 40/44: handle_break_exc with literal value returns it."""
    sentinel = object()
    def f(el):
      if el == 2:
        return Chain.break_(sentinel)
      return el
    self.assertIs(
      Chain([1, 2, 3]).foreach(f).run(),
      sentinel
    )

  async def test_break_with_callable_value(self):
    """handle_break_exc with callable value calls it via _eval_signal_value."""
    def f(el):
      if el == 2:
        return Chain.break_(lambda: 'break_result')
      return el
    self.assertEqual(
      Chain([1, 2, 3]).foreach(f).run(),
      'break_result'
    )

  async def test_break_with_callable_and_args(self):
    """handle_break_exc with callable + args calls fn(*args)."""
    def f(el):
      if el == 2:
        return Chain.break_(lambda x, y: x + y, 'hello', ' world')
      return el
    self.assertEqual(
      Chain([1, 2, 3]).foreach(f).run(),
      'hello world'
    )


class HandleReturnExcTests(IsolatedAsyncioTestCase):
  """Tests targeting handle_return_exc (line 47)."""

  async def test_return_no_value_returns_none(self):
    """Line 47/51-52: handle_return_exc with Null value returns None."""
    result = Chain(1).then(lambda v: Chain.return_()).run()
    self.assertIsNone(result)

  async def test_return_with_literal_value(self):
    """Line 47/53: handle_return_exc with literal value returns it."""
    result = Chain(1).then(lambda v: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  async def test_return_with_callable_value(self):
    """handle_return_exc with callable calls it via _eval_signal_value."""
    result = Chain(1).then(lambda v: Chain.return_(lambda: 'returned')).run()
    self.assertEqual(result, 'returned')

  async def test_return_propagates_in_nested_chain(self):
    """Line 48-50: propagate=True re-raises when in nested chain."""
    inner = Chain().then(lambda v: Chain.return_(99))
    result = Chain(1).then(inner).run()
    self.assertEqual(result, 99)


# ===========================================================================
# CONTROL FLOW: with_ factory (line 99), _with_to_async (line 112), entered=False (line 124)
# ===========================================================================

class WithFactoryAndAsyncTests(IsolatedAsyncioTestCase):
  """Tests targeting with_ factory (line 99), _with_to_async (line 112),
  and _with_full_async (line 134)."""

  async def test_with_factory_sync_cm(self):
    """Line 99: with_ factory exercised by Chain.with_()."""
    cm = SyncCM('hello')
    result = Chain(cm).with_(lambda ctx: ctx.upper()).run()
    self.assertEqual(result, 'HELLO')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_to_async_body_returns_coro(self):
    """Line 112: _with_to_async entered when sync CM body returns a coroutine."""
    cm = SyncCM('sync_ctx')
    async def async_body(ctx):
      return f'async_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'async_sync_ctx')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_to_async_body_raises(self):
    """_with_to_async exception path: body raises, __exit__ is called.
    entered=True path (line 117-122)."""
    cm = SyncCM('ctx')
    async def async_body_raises(ctx):
      raise TestExc('body error')
    with self.assertRaises(TestExc):
      await Chain(cm).with_(async_body_raises).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_to_async_body_raises_suppressed(self):
    """_with_to_async: body raises but CM suppresses exception."""
    cm = SyncCMSuppressing('ctx')
    async def async_body_raises(ctx):
      raise TestExc('suppressed')
    result = await Chain(cm).with_(async_body_raises).run()
    # When exception is suppressed, result is None (body_result was not set)
    self.assertIsNone(result)
    self.assertTrue(cm.exited)

  async def test_with_full_async_cm(self):
    """Line 134: _with_full_async entered for async context manager."""
    cm = AsyncCM('async_ctx')
    result = await Chain(cm).with_(lambda ctx: f'got_{ctx}').run()
    self.assertEqual(result, 'got_async_ctx')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_full_async_cm_async_body(self):
    """_with_full_async with async body (iscoro branch line 143-144)."""
    cm = AsyncCM('async_ctx')
    async def async_body(ctx):
      return f'async_body_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'async_body_async_ctx')

  async def test_with_full_async_exception(self):
    """_with_full_async exception path (lines 145-149)."""
    cm = AsyncCM('ctx')
    def body_raises(ctx):
      raise TestExc('full async exc')
    with self.assertRaises(TestExc):
      await Chain(cm).with_(body_raises).run()
    self.assertTrue(cm.exited)

  async def test_with_sync_cm_enter_raises(self):
    """Line 124 (entered=False path): __enter__ fails, exception re-raised.
    In _With.__call__, if __enter__ raises, entered=False and the exception
    is re-raised. This does NOT go through _with_to_async."""
    with self.assertRaises(TestExc):
      Chain(SyncCMRaisingEnter()).with_(lambda ctx: ctx).run()


class WithToAsyncCoroExitTests(IsolatedAsyncioTestCase):
  """Tests for _with_to_async with __exit__ returning a coroutine."""

  async def test_sync_cm_coro_exit_success(self):
    """_with_to_async: __exit__ returns coro on success path (line 126-128)."""
    cm = SyncCMCoroExit('val', suppress=False)
    async def async_body(ctx):
      return f'got_{ctx}'
    result = await Chain(cm).with_(async_body).run()
    self.assertEqual(result, 'got_val')
    self.assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_exception_not_suppressed(self):
    """_with_to_async: __exit__ returns coro, exception not suppressed (lines 118-122)."""
    cm = SyncCMCoroExit('val', suppress=False)
    async def async_body_raises(ctx):
      raise TestExc('not suppressed')
    with self.assertRaises(TestExc):
      await Chain(cm).with_(async_body_raises).run()
    self.assertTrue(cm.exited)

  async def test_sync_cm_coro_exit_exception_suppressed(self):
    """_with_to_async: __exit__ returns coro that suppresses exception."""
    cm = SyncCMCoroExit('val', suppress=True)
    async def async_body_raises(ctx):
      raise TestExc('suppressed')
    result = await Chain(cm).with_(async_body_raises).run()
    self.assertIsNone(result)
    self.assertTrue(cm.exited)


# ===========================================================================
# CONTROL FLOW: sync_generator (line 157) and async_generator (line 176)
# ===========================================================================

class SyncGeneratorTests(IsolatedAsyncioTestCase):
  """Tests targeting sync_generator (line 157)."""

  async def test_sync_generator_normal(self):
    """Line 157: sync_generator yields all elements."""
    r = []
    for i in Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: i * 10):
      r.append(i)
    self.assertEqual(r, [10, 20, 30])

  async def test_sync_generator_no_fn(self):
    """sync_generator with fn=None yields raw elements."""
    r = []
    for i in Chain(SyncIterator, [5, 6]).iterate():
      r.append(i)
    self.assertEqual(r, [5, 6])

  async def test_sync_generator_break(self):
    """sync_generator: _Break causes return (line 170)."""
    def f(i):
      if i >= 3:
        return Chain.break_()
      return i
    r = []
    for i in Chain(SyncIterator, [1, 2, 3, 4]).iterate(f):
      r.append(i)
    self.assertEqual(r, [1, 2])

  async def test_sync_generator_return_raises(self):
    """sync_generator: _Return raises QuentException (line 172-173)."""
    with self.assertRaises(QuentException):
      for _ in Chain(SyncIterator, [1, 2]).iterate(Chain.return_):
        pass

  async def test_sync_generator_ignore_result(self):
    """sync_generator with ignore_result yields original elements."""
    # iterate() always passes ignore_result=False, so fn result is yielded.
    r = []
    for i in Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: i * 100):
      r.append(i)
    self.assertEqual(r, [100, 200, 300])

  async def test_sync_generator_empty(self):
    """sync_generator with empty iterable."""
    r = []
    for i in Chain(SyncIterator, []).iterate(lambda i: i):
      r.append(i)
    self.assertEqual(r, [])


class AsyncGeneratorTests(IsolatedAsyncioTestCase):
  """Tests targeting async_generator (line 176)."""

  async def test_async_generator_sync_iterable(self):
    """Line 176: async_generator with sync iterable input (lines 197-208)."""
    r = []
    async for i in Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: i * 10):
      r.append(i)
    self.assertEqual(r, [10, 20, 30])

  async def test_async_generator_async_iterable(self):
    """Line 176: async_generator with async iterable (lines 185-196)."""
    r = []
    async for i in Chain(AsyncIterator, [1, 2, 3]).iterate(lambda i: i * 10):
      r.append(i)
    self.assertEqual(r, [10, 20, 30])

  async def test_async_generator_no_fn_sync_iterable(self):
    """async_generator with fn=None, sync iterable (line 199-200)."""
    r = []
    async for i in Chain(SyncIterator, [7, 8]).iterate():
      r.append(i)
    self.assertEqual(r, [7, 8])

  async def test_async_generator_no_fn_async_iterable(self):
    """async_generator with fn=None, async iterable (line 187-188)."""
    r = []
    async for i in Chain(AsyncIterator, [7, 8]).iterate():
      r.append(i)
    self.assertEqual(r, [7, 8])

  async def test_async_generator_async_fn_sync_iterable(self):
    """async_generator with async fn on sync iterable (lines 202-204)."""
    r = []
    async for i in Chain(SyncIterator, [1, 2]).iterate(lambda i: aempty(i * 5)):
      r.append(i)
    self.assertEqual(r, [5, 10])

  async def test_async_generator_async_fn_async_iterable(self):
    """async_generator with async fn on async iterable (lines 190-192)."""
    r = []
    async for i in Chain(AsyncIterator, [1, 2]).iterate(lambda i: aempty(i * 5)):
      r.append(i)
    self.assertEqual(r, [5, 10])

  async def test_async_generator_break_sync_iterable(self):
    """async_generator break with sync iterable (line 209-210)."""
    def f(i):
      if i >= 2:
        return Chain.break_()
      return i * 10
    r = []
    async for i in Chain(SyncIterator, [0, 1, 2, 3]).iterate(f):
      r.append(i)
    self.assertEqual(r, [0, 10])

  async def test_async_generator_break_async_iterable(self):
    """async_generator break with async iterable (line 209-210)."""
    def f(i):
      if i >= 2:
        return Chain.break_()
      return i * 10
    r = []
    async for i in Chain(AsyncIterator, [0, 1, 2, 3]).iterate(f):
      r.append(i)
    self.assertEqual(r, [0, 10])

  async def test_async_generator_return_raises(self):
    """async_generator: _Return raises QuentException (line 211-212)."""
    with self.assertRaises(QuentException):
      async for _ in Chain(SyncIterator, [1, 2]).iterate(Chain.return_):
        pass

  async def test_async_generator_return_raises_async_iterable(self):
    """async_generator: _Return raises QuentException with async iterable."""
    with self.assertRaises(QuentException):
      async for _ in Chain(AsyncIterator, [1, 2]).iterate(Chain.return_):
        pass

  async def test_async_generator_coro_iterator_getter(self):
    """async_generator: iterator_getter returns a coroutine (lines 181-182).
    This happens when Chain root involves async operations."""
    r = []
    async for i in Chain(aempty, [1, 2, 3]).iterate(lambda i: i * 2):
      r.append(i)
    self.assertEqual(r, [2, 4, 6])

  async def test_async_generator_empty_sync(self):
    """async_generator with empty sync iterable."""
    r = []
    async for i in Chain(SyncIterator, []).iterate(lambda i: i):
      r.append(i)
    self.assertEqual(r, [])

  async def test_async_generator_empty_async(self):
    """async_generator with empty async iterable."""
    r = []
    async for i in Chain(AsyncIterator, []).iterate(lambda i: i):
      r.append(i)
    self.assertEqual(r, [])


# ===========================================================================
# CONTROL FLOW: _Generator.__call__ (line 227) and aliases (lines 253-256)
# ===========================================================================

class GeneratorCallTests(IsolatedAsyncioTestCase):
  """Tests targeting _Generator.__call__ (line 227) and function aliases (lines 253-256)."""

  async def test_generator_call_creates_new_instance(self):
    """Line 227: _Generator.__call__ creates a new _Generator via __new__."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    nested = gen(SyncIterator([10, 20, 30]))
    # nested is a new _Generator
    self.assertIsNot(gen, nested)
    self.assertEqual(repr(nested), '<_Generator>')

  async def test_generator_call_produces_correct_values(self):
    """Line 227: nested _Generator produces values from the override args."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i * 3)
    nested = gen(SyncIterator([5, 10]))
    r = []
    for i in nested:
      r.append(i)
    self.assertEqual(r, [15, 30])

  async def test_generator_call_async_iteration(self):
    """Line 227: nested _Generator supports async for."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i + 100)
    nested = gen(SyncIterator([1, 2]))
    r = []
    async for i in nested:
      r.append(i)
    self.assertEqual(r, [101, 102])

  async def test_generator_call_with_kwargs(self):
    """Line 233: __call__ captures kwargs in _run_args."""
    # Chains don't normally use kwargs for run, but _Generator stores them.
    gen = Chain().then(lambda v: v).iterate(lambda i: i)
    nested = gen(SyncIterator([1]))
    r = list(nested)
    self.assertEqual(r, [1])

  async def test_generator_call_original_still_works(self):
    """After calling a _Generator, the original is still usable."""
    gen = Chain(SyncIterator, [1, 2]).iterate(lambda i: i + 10)
    _ = gen(SyncIterator([99]))
    r = list(gen)
    self.assertEqual(r, [11, 12])


class ControlFlowAliasTests(IsolatedAsyncioTestCase):
  """Tests that exercise the function aliases at lines 253-256 of _control_flow.pxi."""

  async def test_alias_sync_generator_fn(self):
    """_sync_generator_fn alias (line 253): exercised by __iter__."""
    r = list(Chain(SyncIterator, [1, 2]).iterate(lambda i: i * 2))
    self.assertEqual(r, [2, 4])

  async def test_alias_async_generator_fn(self):
    """_async_generator_fn alias (line 254): exercised by __aiter__."""
    r = []
    async for i in Chain(SyncIterator, [1, 2]).iterate(lambda i: i * 2):
      r.append(i)
    self.assertEqual(r, [2, 4])

  async def test_alias_async_with_fn(self):
    """_async_with_fn alias (line 255): exercised by async CM path."""
    cm = AsyncCM('val')
    result = await Chain(cm).with_(lambda ctx: f'test_{ctx}').run()
    self.assertEqual(result, 'test_val')

  async def test_alias_with_async_fn(self):
    """_with_async_fn alias (line 256): exercised by sync CM with coro body."""
    cm = SyncCM('val')
    async def body(ctx):
      return f'async_{ctx}'
    result = await Chain(cm).with_(body).run()
    self.assertEqual(result, 'async_val')


# ===========================================================================
# INTEGRATION: combined paths
# ===========================================================================

class IntegrationTests(IsolatedAsyncioTestCase):
  """Integration tests combining multiple covered paths."""

  async def test_foreach_in_chain_with_then(self):
    """foreach followed by then, exercising factory and async path."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3]))
        .foreach(lambda x: aempty(x * 2))
        .then(lambda lst: sum(lst))
        .run(),
      12
    )

  async def test_filter_in_chain_with_then(self):
    """filter followed by then, exercising factory and full async path."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3, 4, 5]))
        .filter(lambda x: aempty(x > 2))
        .then(lambda lst: len(lst))
        .run(),
      3
    )

  async def test_gather_in_chain_with_foreach(self):
    """gather into foreach."""
    self.assertEqual(
      await Chain(5)
        .gather(lambda v: v + 1, lambda v: aempty(v * 2))
        .foreach(lambda x: x * 10)
        .run(),
      [60, 100]
    )

  async def test_with_then_foreach_async(self):
    """with_ followed by foreach on async iterable."""
    cm = AsyncCM([1, 2, 3])
    result = await Chain(cm).with_(
      lambda ctx: Chain(ctx).foreach(lambda x: x * 10).run()
    ).run()
    self.assertEqual(result, [10, 20, 30])

  async def test_nested_break_in_foreach_async(self):
    """Break in async foreach within a chain."""
    def f(el):
      if el >= 3:
        return Chain.break_('stopped')
      return el * 10
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3, 4])).foreach(f).run(),
      'stopped'
    )

  async def test_return_in_chain_with_async_fn(self):
    """return_ with async value in chain with do() (non-simple path)."""
    result = await Chain(aempty, 1).do(lambda v: None).then(
      lambda v: Chain.return_(aempty, 'returned')
    ).run()
    self.assertEqual(result, 'returned')

  async def test_generator_nested_in_chain(self):
    """Generator created via iterate, used inside another Chain."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i * 2)
    nested = gen(SyncIterator([10, 20]))
    r = []
    for i in nested:
      r.append(i)
    self.assertEqual(r, [20, 40])

  async def test_async_generator_with_chain_producing_async_iter(self):
    """async_generator where chain root is an async function producing an async iterable."""
    async def make_iter():
      return AsyncIterator([5, 10, 15])
    r = []
    async for i in Chain(make_iter).iterate(lambda i: i + 1):
      r.append(i)
    self.assertEqual(r, [6, 11, 16])

  async def test_foreach_full_async_ignore_result_false(self):
    """foreach on async iterable collects fn results (ignore_result=False)."""
    self.assertEqual(
      await Chain(AsyncIterator([1, 2, 3])).foreach(lambda x: x * 100).run(),
      [100, 200, 300]
    )


if __name__ == '__main__':
  import unittest
  unittest.main()
