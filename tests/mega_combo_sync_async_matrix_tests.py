"""Systematic sync/async matrix tests for every Chain feature.

Tests each feature across 4 execution modes:
  1. Pure sync — all functions are sync
  2. Pure async — all functions are async
  3. Sync-to-async transition — chain starts sync, encounters coro mid-way
  4. Mixed — interleaved sync and async functions

This exercises DIFFERENT code paths in the Cython implementation:
  - _run vs _run_async (chain_core)
  - _run_simple vs _run_async_simple (chain_core fast path)
  - Sync __call__ vs _*_to_async vs _*_full_async (iteration/control_flow)
"""

import asyncio
import inspect
from contextlib import contextmanager, asynccontextmanager
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sync_add1(v):
  return v + 1

def sync_double(v):
  return v * 2

def sync_identity(v):
  return v

async def async_add1(v):
  return v + 1

async def async_double(v):
  return v * 2

async def async_identity(v):
  return v


class AsyncIter:
  """Async iterator over a list of items."""
  def __init__(self, items):
    self._items = list(items)
    self._index = 0

  def __aiter__(self):
    return self

  async def __anext__(self):
    if self._index >= len(self._items):
      raise StopAsyncIteration
    val = self._items[self._index]
    self._index += 1
    return val


class SyncCM:
  """Sync context manager that yields a value."""
  def __init__(self, value):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  """Async context manager that yields a value."""
  def __init__(self, value):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


# ---------------------------------------------------------------------------
# A. then() — 4 modes
# ---------------------------------------------------------------------------
class ThenSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_then_pure_sync(self):
    """Pure sync: all .then() callbacks are sync."""
    self.assertEqual(Chain(1).then(sync_add1).run(), 2)
    self.assertEqual(Chain(1).then(sync_add1).then(sync_double).run(), 4)
    self.assertEqual(
      Chain(1).then(sync_add1).then(sync_double).then(sync_add1).run(), 5
    )

  async def test_then_pure_async(self):
    """Pure async: all .then() callbacks are async."""
    self.assertEqual(await Chain(1).then(async_add1).run(), 2)
    self.assertEqual(await Chain(1).then(async_add1).then(async_double).run(), 4)
    self.assertEqual(
      await Chain(1).then(async_add1).then(async_double).then(async_add1).run(), 5
    )

  async def test_then_sync_to_async_transition(self):
    """Sync root + sync then, then async then triggers _run_async."""
    self.assertEqual(
      await Chain(1).then(sync_add1).then(async_double).run(), 4
    )
    # Transition at root: async root, rest sync
    self.assertEqual(
      await Chain(async_add1, 0).then(sync_double).run(), 2
    )

  async def test_then_mixed(self):
    """Mixed: sync and async interleaved in multiple positions."""
    self.assertEqual(
      await Chain(1).then(async_add1).then(sync_double).then(async_add1).run(), 5
    )
    self.assertEqual(
      await Chain(1).then(sync_add1).then(async_double).then(sync_add1).run(), 5
    )
    # 5 links: sync -> async -> sync -> async -> sync
    self.assertEqual(
      await Chain(1)
      .then(sync_add1)      # 2
      .then(async_double)   # 4
      .then(sync_add1)      # 5
      .then(async_double)   # 10
      .then(sync_add1)      # 11
      .run(), 11
    )

  async def test_then_async_root(self):
    """Async root value triggers async from the very start."""
    self.assertEqual(await Chain(aempty, 10).then(sync_add1).run(), 11)
    self.assertEqual(await Chain(aempty, 10).then(async_add1).run(), 11)


# ---------------------------------------------------------------------------
# B. do() — 4 modes
# ---------------------------------------------------------------------------
class DoSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_do_pure_sync(self):
    """Pure sync: do() side effect runs, result is discarded."""
    log = []
    self.assertEqual(
      Chain(10).do(lambda v: log.append(v)).run(), 10
    )
    assert log == [10]

  async def test_do_pure_async(self):
    """Pure async: do() with async side effect."""
    log = []
    async def async_log(v):
      log.append(v)
    self.assertEqual(
      await Chain(10).do(async_log).run(), 10
    )
    assert log == [10]

  async def test_do_sync_to_async_transition(self):
    """Sync chain, then async do() triggers transition."""
    log = []
    async def async_log(v):
      log.append(v)
    self.assertEqual(
      await Chain(10).then(sync_add1).do(async_log).run(), 11
    )
    assert log == [11]

  async def test_do_mixed(self):
    """Mixed: async do, sync then, async do."""
    log = []
    async def async_log(v):
      log.append(('async', v))
    def sync_log(v):
      log.append(('sync', v))
    self.assertEqual(
      await Chain(1)
      .do(async_log)
      .then(sync_double)
      .do(sync_log)
      .then(async_add1)
      .do(async_log)
      .run(), 3
    )
    assert log == [('async', 1), ('sync', 2), ('async', 3)]

  async def test_do_preserves_value_across_modes(self):
    """do() result is always discarded, regardless of sync/async mode."""
    self.assertEqual(
      Chain(5).do(lambda v: 999).run(), 5
    )
    async def async_side(v):
      return 999
    self.assertEqual(
      await Chain(5).do(async_side).run(), 5
    )


# ---------------------------------------------------------------------------
# C. except_() — 4 modes
# ---------------------------------------------------------------------------
class ExceptSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_except_sync_raise_sync_handler(self):
    """Sync raise + sync handler."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    result = Chain(1).then(lambda v: (_ for _ in ()).throw(TestExc())).except_(
      handler, reraise=False
    ).run()
    # Direct approach: use a raising function
    def sync_raise(v):
      raise TestExc('sync')
    caught.clear()
    result = Chain(1).then(sync_raise).except_(handler, reraise=False).run()
    self.assertEqual(result, 'recovered')
    assert caught['ran']

  async def test_except_async_raise_async_handler(self):
    """Async raise + async handler."""
    async def async_raise(v):
      raise TestExc('async')
    caught = {}
    async def async_handler(v):
      caught['ran'] = True
      return 'async_recovered'
    result = Chain(1).then(async_raise).except_(async_handler, reraise=False).run()
    self.assertEqual(await result, 'async_recovered')
    assert caught['ran']

  async def test_except_sync_raise_async_handler(self):
    """Sync raise + async handler (handler returns coro from sync context)."""
    def sync_raise(v):
      raise TestExc('sync')
    caught = {}
    async def async_handler(v):
      caught['ran'] = True
      return 'async_handled'
    result = Chain(1).then(sync_raise).except_(async_handler, reraise=False).run()
    self.assertEqual(await result, 'async_handled')
    assert caught['ran']

  async def test_except_async_raise_sync_handler(self):
    """Async raise + sync handler."""
    async def async_raise(v):
      raise TestExc('async')
    caught = {}
    def sync_handler(v):
      caught['ran'] = True
      return 'sync_handled'
    result = Chain(1).then(async_raise).except_(sync_handler, reraise=False).run()
    self.assertEqual(await result, 'sync_handled')
    assert caught['ran']

  async def test_except_reraise_true_sync(self):
    """Sync reraise=True: handler runs, exception still propagates."""
    def sync_raise(v):
      raise TestExc
    caught = {}
    def handler(v):
      caught['ran'] = True
    try:
      Chain(1).then(sync_raise).except_(handler).run()
    except TestExc:
      pass
    assert caught['ran']

  async def test_except_reraise_true_async(self):
    """Async reraise=True: handler runs, exception still propagates."""
    async def async_raise(v):
      raise TestExc
    caught = {}
    def handler(v):
      caught['ran'] = True
    try:
      await await_(Chain(1).then(async_raise).except_(handler).run())
    except TestExc:
      pass
    assert caught['ran']

  async def test_except_exception_type_filtering_sync(self):
    """Sync: handler only runs if exception type matches."""
    def sync_raise(v):
      raise ValueError('test')
    caught_value = {}
    caught_type = {}
    def handler_value(v):
      caught_value['ran'] = True
    def handler_type(v):
      caught_type['ran'] = True
    try:
      Chain(1).then(sync_raise).except_(
        handler_value, exceptions=TypeError, reraise=True
      ).except_(
        handler_type, exceptions=ValueError, reraise=True
      ).run()
    except ValueError:
      pass
    assert 'ran' not in caught_value
    assert caught_type.get('ran')

  async def test_except_exception_type_filtering_async(self):
    """Async: handler only runs if exception type matches."""
    async def async_raise(v):
      raise ValueError('test')
    caught_value = {}
    caught_type = {}
    def handler_value(v):
      caught_value['ran'] = True
    def handler_type(v):
      caught_type['ran'] = True
    try:
      await await_(Chain(1).then(async_raise).except_(
        handler_value, exceptions=TypeError, reraise=True
      ).except_(
        handler_type, exceptions=ValueError, reraise=True
      ).run())
    except ValueError:
      pass
    assert 'ran' not in caught_value
    assert caught_type.get('ran')


# ---------------------------------------------------------------------------
# D. finally_() — 4 modes
# ---------------------------------------------------------------------------
class FinallySyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_finally_sync_chain_sync_handler(self):
    """Sync chain + sync finally."""
    log = []
    def on_finally(v):
      log.append('finally')
    result = Chain(10).then(sync_add1).finally_(on_finally).run()
    self.assertEqual(result, 11)
    assert log == ['finally']

  async def test_finally_async_chain_async_handler(self):
    """Async chain + async finally."""
    log = []
    async def on_finally(v):
      log.append('finally')
    result = Chain(10).then(async_add1).finally_(on_finally).run()
    self.assertEqual(await result, 11)
    assert log == ['finally']

  async def test_finally_sync_chain_async_handler(self):
    """Sync chain + async finally (forces async finally handling)."""
    log = []
    async def on_finally(v):
      log.append('finally')
    # In a pure sync chain, async finally is scheduled as a fire-and-forget task
    import warnings
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      result = Chain(10).then(sync_add1).finally_(on_finally).run()
    self.assertEqual(result, 11)
    await asyncio.sleep(0.1)
    assert log == ['finally']

  async def test_finally_mixed_chain_sync_handler(self):
    """Mixed chain + sync finally."""
    log = []
    def on_finally(v):
      log.append('finally')
    result = Chain(10).then(sync_add1).then(async_double).finally_(on_finally).run()
    self.assertEqual(await result, 22)
    assert log == ['finally']

  async def test_finally_runs_on_exception_sync(self):
    """Sync: finally runs even when exception is raised."""
    log = []
    def on_finally(v):
      log.append('finally')
    def sync_raise(v):
      raise TestExc
    try:
      Chain(1).then(sync_raise).finally_(on_finally).run()
    except TestExc:
      pass
    assert log == ['finally']

  async def test_finally_runs_on_exception_async(self):
    """Async: finally runs even when async exception is raised."""
    log = []
    async def on_finally(v):
      log.append('finally')
    async def async_raise(v):
      raise TestExc
    try:
      await await_(Chain(1).then(async_raise).finally_(on_finally).run())
    except TestExc:
      pass
    assert log == ['finally']


# ---------------------------------------------------------------------------
# E. foreach() — 4 modes x 2 iterable types
# ---------------------------------------------------------------------------
class ForeachSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_foreach_sync_iterable_sync_fn(self):
    """Sync iterable + sync fn: pure sync path (_Foreach.__call__)."""
    result = Chain([1, 2, 3]).foreach(sync_double).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_foreach_sync_iterable_async_fn(self):
    """Sync iterable + async fn: triggers _foreach_to_async transition."""
    result = Chain([1, 2, 3]).foreach(async_double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_foreach_async_iterable_sync_fn(self):
    """Async iterable + sync fn: triggers _foreach_full_async."""
    result = Chain(AsyncIter([1, 2, 3])).foreach(sync_double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_foreach_async_iterable_async_fn(self):
    """Async iterable + async fn: triggers _foreach_full_async with coro await."""
    result = Chain(AsyncIter([1, 2, 3])).foreach(async_double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_foreach_sync_iterable_transition_mid_iteration(self):
    """Sync iterable: fn is sync for some elements, returns coro mid-iteration."""
    call_count = [0]
    def mixed_fn(v):
      call_count[0] += 1
      if call_count[0] > 2:
        return async_double(v)
      return v * 2
    result = Chain([1, 2, 3, 4]).foreach(mixed_fn).run()
    self.assertEqual(await result, [2, 4, 6, 8])

  async def test_foreach_upstream_async_sync_iterable_sync_fn(self):
    """Async upstream -> sync iterable + sync fn: chain is already async."""
    result = Chain(aempty, [1, 2, 3]).foreach(sync_double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_foreach_upstream_async_sync_iterable_async_fn(self):
    """Async upstream -> sync iterable + async fn."""
    result = Chain(aempty, [1, 2, 3]).foreach(async_double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_foreach_empty_sync_iterable(self):
    """Empty sync iterable returns empty list."""
    self.assertEqual(Chain([]).foreach(sync_double).run(), [])

  async def test_foreach_empty_async_iterable(self):
    """Empty async iterable returns empty list."""
    self.assertEqual(await Chain(AsyncIter([])).foreach(sync_double).run(), [])

  async def test_foreach_break_sync(self):
    """Break in sync foreach."""
    def fn(v):
      if v == 3:
        Chain.break_()
      return v * 2
    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(result, [2, 4])

  async def test_foreach_break_async(self):
    """Break in async foreach."""
    async def fn(v):
      if v == 3:
        Chain.break_()
      return v * 2
    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(await result, [2, 4])


# ---------------------------------------------------------------------------
# F. filter() — 4 modes x 2 iterable types
# ---------------------------------------------------------------------------
class FilterSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_filter_sync_iterable_sync_predicate(self):
    """Sync iterable + sync predicate: pure sync path."""
    result = Chain([1, 2, 3, 4, 5]).filter(lambda v: v % 2 == 0).run()
    self.assertEqual(result, [2, 4])

  async def test_filter_sync_iterable_async_predicate(self):
    """Sync iterable + async predicate: triggers _filter_to_async."""
    async def async_even(v):
      return v % 2 == 0
    result = Chain([1, 2, 3, 4, 5]).filter(async_even).run()
    self.assertEqual(await result, [2, 4])

  async def test_filter_async_iterable_sync_predicate(self):
    """Async iterable + sync predicate: triggers _filter_full_async."""
    result = Chain(AsyncIter([1, 2, 3, 4, 5])).filter(lambda v: v % 2 == 0).run()
    self.assertEqual(await result, [2, 4])

  async def test_filter_async_iterable_async_predicate(self):
    """Async iterable + async predicate: triggers _filter_full_async with coro."""
    async def async_even(v):
      return v % 2 == 0
    result = Chain(AsyncIter([1, 2, 3, 4, 5])).filter(async_even).run()
    self.assertEqual(await result, [2, 4])

  async def test_filter_sync_iterable_transition_mid_iteration(self):
    """Sync iterable: predicate transitions to coro mid-iteration."""
    call_count = [0]
    def mixed_pred(v):
      call_count[0] += 1
      if call_count[0] > 2:
        return async_identity(v % 2 == 0)
      return v % 2 == 0
    result = Chain([1, 2, 3, 4, 5]).filter(mixed_pred).run()
    self.assertEqual(await result, [2, 4])

  async def test_filter_empty_sync(self):
    """Empty sync iterable."""
    self.assertEqual(Chain([]).filter(lambda v: True).run(), [])

  async def test_filter_empty_async(self):
    """Empty async iterable."""
    self.assertEqual(await Chain(AsyncIter([])).filter(lambda v: True).run(), [])

  async def test_filter_upstream_async(self):
    """Async upstream -> sync iterable + sync predicate."""
    result = Chain(aempty, [1, 2, 3, 4, 5]).filter(lambda v: v > 3).run()
    self.assertEqual(await result, [4, 5])

  async def test_filter_none_pass(self):
    """Filter where nothing passes."""
    self.assertEqual(
      Chain([1, 2, 3]).filter(lambda v: False).run(), []
    )

  async def test_filter_all_pass(self):
    """Filter where everything passes."""
    self.assertEqual(
      Chain([1, 2, 3]).filter(lambda v: True).run(), [1, 2, 3]
    )


# ---------------------------------------------------------------------------
# H. gather() — 4 modes
# ---------------------------------------------------------------------------
class GatherSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_gather_all_sync(self):
    """All sync fns."""
    result = Chain(5).gather(
      lambda v: v + 1,
      lambda v: v * 2,
      lambda v: v - 1
    ).run()
    self.assertEqual(result, [6, 10, 4])

  async def test_gather_all_async(self):
    """All async fns."""
    async def a1(v): return v + 1
    async def a2(v): return v * 2
    async def a3(v): return v - 1
    result = Chain(5).gather(a1, a2, a3).run()
    self.assertEqual(await result, [6, 10, 4])

  async def test_gather_mixed_sync_async(self):
    """Mixed: some sync, some async — triggers _gather_to_async."""
    async def a1(v): return v + 1
    result = Chain(5).gather(
      lambda v: v * 2,
      a1,
      lambda v: v - 1
    ).run()
    self.assertEqual(await result, [10, 6, 4])

  async def test_gather_single_coro_among_many_sync(self):
    """Single coro among many sync — still triggers _gather_to_async."""
    async def a1(v): return v + 100
    result = Chain(5).gather(
      lambda v: v + 1,
      lambda v: v + 2,
      a1,
      lambda v: v + 4
    ).run()
    self.assertEqual(await result, [6, 7, 105, 9])

  async def test_gather_upstream_async(self):
    """Async upstream -> gather with mixed fns."""
    async def a1(v): return v * 10
    result = Chain(aempty, 3).gather(
      lambda v: v + 1,
      a1
    ).run()
    self.assertEqual(await result, [4, 30])

  async def test_gather_empty(self):
    """Gather with no fns — returns empty list."""
    result = Chain(5).gather().run()
    self.assertEqual(result, [])

  async def test_gather_single_sync(self):
    """Gather with one sync fn."""
    result = Chain(5).gather(lambda v: v * 3).run()
    self.assertEqual(result, [15])

  async def test_gather_single_async(self):
    """Gather with one async fn."""
    async def a1(v): return v * 3
    result = Chain(5).gather(a1).run()
    self.assertEqual(await result, [15])


# ---------------------------------------------------------------------------
# I. with_() — 4 modes x 2 CM types
# ---------------------------------------------------------------------------
class WithSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_with_sync_cm_sync_body(self):
    """Sync CM + sync body: pure sync path."""
    cm = SyncCM(42)
    result = Chain(cm).with_(lambda ctx: ctx + 1).run()
    self.assertEqual(result, 43)
    assert cm.entered and cm.exited

  async def test_with_sync_cm_async_body(self):
    """Sync CM + async body: triggers _with_to_async."""
    cm = SyncCM(42)
    async def async_body(ctx):
      return ctx + 1
    result = Chain(cm).with_(async_body).run()
    self.assertEqual(await result, 43)
    assert cm.entered and cm.exited

  async def test_with_async_cm_sync_body(self):
    """Async CM + sync body: triggers _with_full_async."""
    cm = AsyncCM(42)
    result = Chain(cm).with_(lambda ctx: ctx + 1).run()
    self.assertEqual(await result, 43)
    assert cm.entered and cm.exited

  async def test_with_async_cm_async_body(self):
    """Async CM + async body: triggers _with_full_async with coro await."""
    cm = AsyncCM(42)
    async def async_body(ctx):
      return ctx + 1
    result = Chain(cm).with_(async_body).run()
    self.assertEqual(await result, 43)
    assert cm.entered and cm.exited

  async def test_with_sync_cm_body_returns_coro(self):
    """Sync CM where body returns coro: triggers _with_to_async."""
    cm = SyncCM(10)
    result = Chain(cm).with_(async_double).run()
    self.assertEqual(await result, 20)
    assert cm.entered and cm.exited

  async def test_with_upstream_async_sync_cm(self):
    """Async upstream -> sync CM + sync body."""
    async def make_cm(v=None):
      return SyncCM(99)
    result = Chain(make_cm).with_(lambda ctx: ctx * 2).run()
    self.assertEqual(await result, 198)

  async def test_with_upstream_async_async_cm(self):
    """Async upstream -> async CM + async body."""
    async def make_cm(v=None):
      return AsyncCM(99)
    async def async_body(ctx):
      return ctx * 2
    result = Chain(make_cm).with_(async_body).run()
    self.assertEqual(await result, 198)

  async def test_with_exception_sync_cm_sync_body(self):
    """Sync CM: exception in body still calls __exit__."""
    cm = SyncCM(1)
    def raising_body(ctx):
      raise TestExc('in body')
    try:
      Chain(cm).with_(raising_body).run()
    except TestExc:
      pass
    assert cm.entered and cm.exited

  async def test_with_exception_async_cm_async_body(self):
    """Async CM: exception in async body still calls __aexit__."""
    cm = AsyncCM(1)
    async def raising_body(ctx):
      raise TestExc('in body')
    try:
      await await_(Chain(cm).with_(raising_body).run())
    except TestExc:
      pass
    assert cm.entered and cm.exited

  async def test_with_nested_chain_body(self):
    """with_ body is a nested chain, sync CM."""
    cm = SyncCM(5)
    result = Chain(cm).with_(Chain().then(sync_double)).run()
    self.assertEqual(result, 10)

  async def test_with_nested_chain_body_async(self):
    """with_ body is a nested chain with async, async CM."""
    cm = AsyncCM(5)
    result = Chain(cm).with_(Chain().then(async_double)).run()
    self.assertEqual(await result, 10)


# ---------------------------------------------------------------------------
# J. iterate() — sync and async generators
# ---------------------------------------------------------------------------
class IterateSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_iterate_sync_generator(self):
    """Sync generator (sync_generator path)."""
    items = []
    for i in Chain([1, 2, 3]).iterate(sync_double):
      items.append(i)
    self.assertEqual(items, [2, 4, 6])

  async def test_iterate_async_generator(self):
    """Async generator (async_generator path)."""
    items = []
    async for i in Chain([1, 2, 3]).iterate(sync_double):
      items.append(i)
    self.assertEqual(items, [2, 4, 6])

  async def test_iterate_async_generator_with_async_fn(self):
    """Async generator with async transform function."""
    items = []
    async for i in Chain([1, 2, 3]).iterate(async_double):
      items.append(i)
    self.assertEqual(items, [2, 4, 6])

  async def test_iterate_sync_generator_no_fn(self):
    """Sync generator without transform."""
    items = []
    for i in Chain([10, 20, 30]).iterate():
      items.append(i)
    self.assertEqual(items, [10, 20, 30])

  async def test_iterate_async_generator_no_fn(self):
    """Async generator without transform."""
    items = []
    async for i in Chain([10, 20, 30]).iterate():
      items.append(i)
    self.assertEqual(items, [10, 20, 30])

  async def test_iterate_async_source_async_generator(self):
    """Async source iterable with async generator."""
    class AsyncList:
      def __init__(self, items):
        self._items = items
        self._idx = 0
      def __aiter__(self):
        return self
      async def __anext__(self):
        if self._idx >= len(self._items):
          raise StopAsyncIteration
        val = self._items[self._idx]
        self._idx += 1
        return val

    items = []
    async for i in Chain(lambda: AsyncList([1, 2, 3])).iterate(sync_double):
      items.append(i)
    self.assertEqual(items, [2, 4, 6])

  async def test_iterate_break_sync(self):
    """Break in sync generator."""
    def fn(v):
      if v == 3:
        Chain.break_()
      return v * 2
    items = []
    for i in Chain([1, 2, 3, 4]).iterate(fn):
      items.append(i)
    self.assertEqual(items, [2, 4])

  async def test_iterate_break_async(self):
    """Break in async generator."""
    def fn(v):
      if v == 3:
        Chain.break_()
      return v * 2
    items = []
    async for i in Chain([1, 2, 3, 4]).iterate(fn):
      items.append(i)
    self.assertEqual(items, [2, 4])


# ---------------------------------------------------------------------------
# M. Nested chains — 4 modes
# ---------------------------------------------------------------------------
class NestedChainsSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_nested_sync_outer_sync_inner(self):
    """Sync outer + sync inner."""
    inner = Chain().then(sync_double)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 10)

  async def test_nested_sync_outer_async_inner(self):
    """Sync outer + async inner: transitions outer to async."""
    inner = Chain().then(async_double)
    result = Chain(5).then(inner).run()
    self.assertEqual(await result, 10)

  async def test_nested_async_outer_sync_inner(self):
    """Async outer + sync inner."""
    inner = Chain().then(sync_double)
    result = Chain(aempty, 5).then(inner).run()
    self.assertEqual(await result, 10)

  async def test_nested_async_outer_async_inner(self):
    """Async outer + async inner."""
    inner = Chain().then(async_double)
    result = Chain(aempty, 5).then(inner).run()
    self.assertEqual(await result, 10)

  async def test_nested_deep_sync(self):
    """Deep nesting, all sync."""
    inner_inner = Chain().then(sync_double)
    inner = Chain().then(inner_inner).then(sync_add1)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 11)

  async def test_nested_deep_async(self):
    """Deep nesting, all async."""
    inner_inner = Chain().then(async_double)
    inner = Chain().then(inner_inner).then(async_add1)
    result = Chain(5).then(inner).run()
    self.assertEqual(await result, 11)

  async def test_nested_deep_mixed(self):
    """Deep nesting, mixed sync/async."""
    inner_inner = Chain().then(async_double)
    inner = Chain().then(sync_add1).then(inner_inner)
    result = Chain(5).then(inner).run()
    self.assertEqual(await result, 12)

  async def test_nested_with_explicit_args(self):
    """Nested chain passed with explicit args."""
    inner = Chain().then(lambda v: v * 3)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 15)
    # With a constant root in the nested chain
    inner2 = Chain(10).then(sync_double)
    result = Chain(5).then(inner2, ...).run()
    self.assertEqual(result, 20)

  async def test_nested_chain_preserves_value_flow(self):
    """Value flows correctly through nested chains."""
    inner = Chain().then(lambda v: v + 10)
    result = Chain(5).then(sync_double).then(inner).then(sync_add1).run()
    self.assertEqual(result, 21)  # 5*2=10, +10=20, +1=21

  async def test_nested_chain_preserves_value_flow_async(self):
    """Async value flows correctly through nested chains."""
    inner = Chain().then(async_add1)
    result = Chain(5).then(async_double).then(inner).then(sync_add1).run()
    self.assertEqual(await result, 12)  # 5*2=10, +1=11, +1=12


# ---------------------------------------------------------------------------
# N. Autorun (run class) — 2 modes
# ---------------------------------------------------------------------------
class AutorunSyncAsyncMatrixTests(IsolatedAsyncioTestCase):

  async def test_autorun_sync(self):
    """Sync autorun with config."""
    obj = type('', (), {'v': False})()
    def set_v(v=None):
      obj.v = True
    Chain(set_v, ...).config(autorun=True).run()
    assert obj.v

  async def test_autorun_async_returns_task(self):
    """Async autorun returns a Task."""
    obj = type('', (), {'v': False})()
    async def async_set_v(v=None):
      obj.v = True
    task = Chain(async_set_v, ...).config(autorun=True).run()
    assert inspect.isawaitable(task)
    await task
    assert obj.v

  async def test_pipe_run_sync(self):
    """Pipe syntax with run() — sync."""
    result = Chain(5).then(sync_double).run()
    self.assertEqual(result, 10)

  async def test_pipe_run_async(self):
    """Pipe syntax with run() — async."""
    result = Chain(5).then(async_double).run()
    self.assertEqual(await result, 10)

  async def test_pipe_run_with_root(self):
    """Pipe syntax with root override."""
    result = Chain().then(sync_double).run(5)
    self.assertEqual(result, 10)


# ---------------------------------------------------------------------------
# O. _run_simple vs _run path
# ---------------------------------------------------------------------------
class SimpleVsFullPathTests(IsolatedAsyncioTestCase):

  async def test_simple_path_sync(self):
    """Simple path: only .then() links, sync."""
    c = Chain(1).then(sync_add1).then(sync_double)
    self.assertEqual(c.run(), 4)

  async def test_simple_path_async(self):
    """Simple path: only .then() links, async transition."""
    c = Chain(1).then(sync_add1).then(async_double)
    self.assertEqual(await c.run(), 4)

  async def test_full_path_with_do_sync(self):
    """Full path: has do(), sync."""
    c = Chain(1).do(lambda v: None).then(sync_add1)
    self.assertEqual(c.run(), 2)

  async def test_full_path_with_do_async(self):
    """Full path: has do(), async."""
    c = Chain(1).do(lambda v: None).then(async_add1)
    self.assertEqual(await c.run(), 2)

  async def test_full_path_with_except_sync(self):
    """Full path: has except_(), sync."""
    c = Chain(1).except_(lambda v: None).then(sync_add1)
    self.assertEqual(c.run(), 2)

  async def test_full_path_with_except_async(self):
    """Full path: has except_(), async."""
    c = Chain(1).except_(lambda v: None).then(async_add1)
    self.assertEqual(await c.run(), 2)

  async def test_full_path_with_finally_sync(self):
    """Full path: has finally_(), sync."""
    log = []
    c = Chain(1).then(sync_add1).finally_(lambda v: log.append('f'))
    self.assertEqual(c.run(), 2)
    assert log == ['f']

  async def test_full_path_with_finally_async(self):
    """Full path: has finally_(), async."""
    log = []
    c = Chain(1).then(async_add1).finally_(lambda v: log.append('f'))
    self.assertEqual(await c.run(), 2)
    assert log == ['f']


# ---------------------------------------------------------------------------
# P. Cross-mode error handling
# ---------------------------------------------------------------------------
class CrossModeErrorHandlingTests(IsolatedAsyncioTestCase):

  async def test_sync_error_in_async_chain(self):
    """Sync error raised in an otherwise async chain."""
    def sync_raise(v):
      raise TestExc('sync error in async chain')
    with self.assertRaises(TestExc):
      await await_(Chain(aempty, 1).then(sync_raise).run())

  async def test_async_error_in_sync_started_chain(self):
    """Async error in a chain that started sync."""
    async def async_raise(v):
      raise TestExc('async error in sync chain')
    with self.assertRaises(TestExc):
      await await_(Chain(1).then(async_raise).run())

  async def test_error_during_sync_to_async_transition(self):
    """Error during the sync-to-async transition point."""
    async def async_raise(v):
      raise TestExc('at transition')
    with self.assertRaises(TestExc):
      await await_(Chain(1).then(sync_add1).then(async_raise).run())

  async def test_error_in_async_finally_of_sync_started_chain(self):
    """Error in async finally from a chain that started sync then transitioned."""
    async def async_raise_finally(v):
      raise ValueError('finally error')
    with self.assertRaises(ValueError):
      await await_(
        Chain(1).then(async_add1).finally_(async_raise_finally).run()
      )

  async def test_except_catches_across_mode_boundary_sync_to_async(self):
    """except_ catches error that occurs after sync-to-async transition."""
    async def async_raise(v):
      raise TestExc('across boundary')
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    result = Chain(1).then(sync_add1).then(async_raise).except_(
      handler, reraise=False
    ).run()
    self.assertEqual(await result, 'caught')
    assert caught['ran']

  async def test_except_catches_sync_error_with_async_handler(self):
    """Sync error caught by async handler in the full _run path."""
    def sync_raise(v):
      raise TestExc
    caught = {}
    async def async_handler(v):
      caught['ran'] = True
      return 'async_caught'
    result = Chain(1).then(sync_raise).except_(async_handler, reraise=False).run()
    self.assertEqual(await result, 'async_caught')
    assert caught['ran']

  async def test_error_propagation_through_nested_chain(self):
    """Error in nested chain propagates to outer."""
    inner = Chain().then(lambda v: 1/0)
    with self.assertRaises(ZeroDivisionError):
      Chain(5).then(inner).run()

  async def test_error_propagation_through_async_nested_chain(self):
    """Error in async nested chain propagates to outer."""
    async def async_raise(v):
      raise TestExc
    inner = Chain().then(async_raise)
    with self.assertRaises(TestExc):
      await await_(Chain(5).then(inner).run())


# ---------------------------------------------------------------------------
# Q. Value flow across mode transitions
# ---------------------------------------------------------------------------
class ValueFlowAcrossTransitionTests(IsolatedAsyncioTestCase):

  async def test_sync_to_async_value_preserved(self):
    """sync fn returns value -> async fn receives it correctly."""
    result = Chain(5).then(sync_double).then(async_add1).run()
    self.assertEqual(await result, 11)  # 5*2=10, +1=11

  async def test_async_to_sync_value_preserved(self):
    """async fn returns value -> sync fn receives it correctly."""
    result = Chain(5).then(async_double).then(sync_add1).run()
    self.assertEqual(await result, 11)  # 5*2=10, +1=11

  async def test_multiple_transitions_preserve_value(self):
    """Value preserved across multiple sync/async transitions."""
    result = (
      Chain(1)
      .then(sync_add1)      # 2 (sync)
      .then(async_double)   # 4 (async)
      .then(sync_add1)      # 5 (sync after async)
      .then(async_double)   # 10 (async)
      .then(sync_add1)      # 11 (sync after async)
      .run()
    )
    self.assertEqual(await result, 11)

  async def test_value_flow_through_do(self):
    """Value preserved through do() across mode transitions."""
    log = []
    async def async_log(v):
      log.append(v)
    result = (
      Chain(5)
      .then(sync_double)    # 10
      .do(async_log)         # logs 10, returns 10
      .then(sync_add1)      # 11
      .do(lambda v: log.append(v))  # logs 11
      .run()
    )
    self.assertEqual(await result, 11)
    assert log == [10, 11]

  async def test_value_flow_through_nested_chains(self):
    """Value flows through nested chains across transitions."""
    inner_sync = Chain().then(sync_double)
    inner_async = Chain().then(async_add1)
    result = Chain(5).then(inner_sync).then(inner_async).run()
    self.assertEqual(await result, 11)  # 5*2=10, +1=11

  async def test_value_flow_through_nested_chains_reverse(self):
    """Value flows through nested chains, async first."""
    inner_async = Chain().then(async_double)
    inner_sync = Chain().then(sync_add1)
    result = Chain(5).then(inner_async).then(inner_sync).run()
    self.assertEqual(await result, 11)  # 5*2=10, +1=11

  async def test_none_preserved_across_transitions(self):
    """None value preserved across sync/async transitions."""
    result = Chain(None).then(async_identity).then(sync_identity).run()
    self.assertIsNone(await result)

  async def test_falsy_preserved_across_transitions(self):
    """Falsy values (0, '', False, []) preserved across transitions."""
    for val in [0, '', False, []]:
      with self.subTest(val=val):
        result = Chain(val).then(async_identity).then(sync_identity).run()
        self.assertEqual(await result, val)

  async def test_complex_object_across_transitions(self):
    """Complex object preserved across sync/async transitions."""
    obj = {'key': [1, 2, 3], 'nested': {'a': 'b'}}
    result = Chain(obj).then(async_identity).then(sync_identity).run()
    self.assertEqual(await result, obj)


# ---------------------------------------------------------------------------
# R. Reuse and Clone across sync/async
# ---------------------------------------------------------------------------
class ReuseCloneSyncAsyncTests(IsolatedAsyncioTestCase):

  async def test_chain_reuse_sync(self):
    """Chain, sync, reusable."""
    c = Chain(5).then(sync_double)
    self.assertEqual(c.run(), 10)
    self.assertEqual(c.run(), 10)

  async def test_chain_reuse_async(self):
    """Chain, async, reusable."""
    c = Chain(5).then(async_double)
    self.assertEqual(await c.run(), 10)
    self.assertEqual(await c.run(), 10)

  async def test_chain_callable(self):
    """Chain __call__."""
    c = Chain(5).then(sync_double)
    self.assertEqual(c(), 10)

  async def test_clone_sync(self):
    """Clone sync chain is independent."""
    c = Chain(5).then(sync_double)
    c2 = c.clone()
    c.then(sync_add1)
    self.assertEqual(c.run(), 11)
    self.assertEqual(c2.run(), 10)

  async def test_clone_async(self):
    """Clone async chain is independent."""
    c = Chain(5).then(async_double)
    c2 = c.clone()
    c.then(async_add1)
    self.assertEqual(await c.run(), 11)
    self.assertEqual(await c2.run(), 10)

  async def test_decorator_sync(self):
    """Decorator pattern, sync."""
    @Chain().then(lambda v: v * 2).decorator()
    def my_fn(v):
      return v + 1
    self.assertEqual(my_fn(5), 12)

  async def test_decorator_async(self):
    """Decorator pattern, async."""
    @Chain().then(async_double).decorator()
    def my_fn(v):
      return v + 1
    self.assertEqual(await my_fn(5), 12)


# ---------------------------------------------------------------------------
# T. Combined features across sync/async
# ---------------------------------------------------------------------------
class CombinedFeaturesSyncAsyncTests(IsolatedAsyncioTestCase):

  async def test_foreach_then_filter_sync(self):
    """Chain: generate list, then filter — all sync."""
    result = (
      Chain([1, 2, 3, 4, 5])
      .foreach(sync_double)
      .filter(lambda v: v > 4)
      .run()
    )
    self.assertEqual(result, [6, 8, 10])

  async def test_foreach_then_filter_async(self):
    """Chain: generate list, then filter — all async."""
    async def async_even(v):
      return v > 4
    result = (
      Chain([1, 2, 3, 4, 5])
      .foreach(async_double)
      .filter(async_even)
      .run()
    )
    self.assertEqual(await result, [6, 8, 10])

  async def test_foreach_then_filter_mixed(self):
    """Chain: foreach async, filter sync."""
    result = (
      Chain([1, 2, 3, 4, 5])
      .foreach(async_double)
      .filter(lambda v: v > 4)
      .run()
    )
    self.assertEqual(await result, [6, 8, 10])

  async def test_with_then_foreach_sync(self):
    """with_ then foreach — sync."""
    cm = SyncCM([1, 2, 3])
    result = Chain(cm).with_(lambda ctx: ctx).foreach(sync_double).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_with_then_foreach_async(self):
    """with_ (async CM) then foreach (async fn)."""
    cm = AsyncCM([1, 2, 3])
    result = Chain(cm).with_(async_identity).foreach(async_double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_gather_then_foreach_sync(self):
    """gather then foreach — sync."""
    result = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v + 2, lambda v: v + 3)
      .foreach(sync_double)
      .run()
    )
    self.assertEqual(result, [12, 14, 16])

  async def test_gather_then_foreach_async(self):
    """gather (async) then foreach (async)."""
    async def a1(v): return v + 1
    async def a2(v): return v + 2
    result = (
      Chain(5)
      .gather(a1, a2)
      .foreach(async_double)
      .run()
    )
    self.assertEqual(await result, [12, 14])

  async def test_except_finally_with_error_across_transition(self):
    """Error after sync-to-async transition, caught by except, finally runs."""
    log = []
    async def async_raise(v):
      raise TestExc
    def handler(v):
      log.append('except')
      return 'recovered'
    def on_finally(v):
      log.append('finally')
    result = (
      Chain(5)
      .then(sync_add1)
      .then(async_raise)
      .except_(handler, reraise=False)
      .finally_(on_finally)
      .run()
    )
    self.assertEqual(await result, 'recovered')
    assert 'except' in log
    assert 'finally' in log

  async def test_nested_chain_with_foreach(self):
    """Nested chain producing a list, outer does foreach."""
    inner = Chain().then(lambda v: [v, v+1, v+2])
    result = Chain(5).then(inner).foreach(sync_double).run()
    self.assertEqual(result, [10, 12, 14])

  async def test_nested_chain_with_gather(self):
    """Nested chain with gather inside."""
    inner = Chain().gather(sync_double, sync_add1)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, [10, 6])


# ---------------------------------------------------------------------------
# U. Void chains across sync/async
# ---------------------------------------------------------------------------
class VoidChainSyncAsyncTests(IsolatedAsyncioTestCase):

  async def test_void_chain_returns_none(self):
    """Void chain returns None."""
    self.assertIsNone(Chain().run())

  async def test_void_chain_with_then_sync(self):
    """Void chain with then and root override, sync."""
    self.assertEqual(Chain().then(sync_double).run(5), 10)

  async def test_void_chain_with_then_async(self):
    """Void chain with then and root override, async."""
    self.assertEqual(await Chain().then(async_double).run(5), 10)

  async def test_void_chain_with_ellipsis_sync(self):
    """Void chain with ellipsis (no-arg call)."""
    self.assertEqual(Chain().then(lambda: 42, ...).run(), 42)

  async def test_void_chain_with_ellipsis_async(self):
    """Void chain with async ellipsis call."""
    async def async_42():
      return 42
    self.assertEqual(await Chain().then(async_42, ...).run(), 42)


# ---------------------------------------------------------------------------
# V. Edge cases for async iterable interactions
# ---------------------------------------------------------------------------
class AsyncIterableEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_foreach_large_async_iterable(self):
    """Large async iterable."""
    items = list(range(100))
    result = Chain(AsyncIter(items)).foreach(sync_double).run()
    self.assertEqual(await result, [i * 2 for i in range(100)])

  async def test_filter_large_async_iterable(self):
    """Large async iterable filter."""
    items = list(range(100))
    result = Chain(AsyncIter(items)).filter(lambda v: v % 10 == 0).run()
    self.assertEqual(await result, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

  async def test_foreach_async_iterable_single_element(self):
    """Async iterable with single element."""
    result = Chain(AsyncIter([42])).foreach(sync_double).run()
    self.assertEqual(await result, [84])

  async def test_filter_async_iterable_single_element(self):
    """Async iterable with single element, passes filter."""
    result = Chain(AsyncIter([42])).filter(lambda v: True).run()
    self.assertEqual(await result, [42])

  async def test_filter_async_iterable_single_element_fails(self):
    """Async iterable with single element, fails filter."""
    result = Chain(AsyncIter([42])).filter(lambda v: False).run()
    self.assertEqual(await result, [])


# ---------------------------------------------------------------------------
# X. Error handling in iteration across modes
# ---------------------------------------------------------------------------
class IterationErrorHandlingTests(IsolatedAsyncioTestCase):

  async def test_foreach_sync_error_in_fn(self):
    """Sync foreach: error in fn propagates."""
    def bad_fn(v):
      if v == 2:
        raise TestExc
      return v
    with self.assertRaises(TestExc):
      Chain([1, 2, 3]).foreach(bad_fn).run()

  async def test_foreach_async_error_in_fn(self):
    """Async foreach: error in fn propagates."""
    async def bad_fn(v):
      if v == 2:
        raise TestExc
      return v
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2, 3]).foreach(bad_fn).run())

  async def test_foreach_async_iterable_error_in_fn(self):
    """Async iterable foreach: error in fn propagates."""
    def bad_fn(v):
      if v == 2:
        raise TestExc
      return v
    with self.assertRaises(TestExc):
      await await_(Chain(AsyncIter([1, 2, 3])).foreach(bad_fn).run())

  async def test_filter_sync_error_in_predicate(self):
    """Sync filter: error in predicate propagates."""
    def bad_pred(v):
      if v == 2:
        raise TestExc
      return True
    with self.assertRaises(TestExc):
      Chain([1, 2, 3]).filter(bad_pred).run()

  async def test_filter_async_error_in_predicate(self):
    """Async filter: error in predicate propagates."""
    async def bad_pred(v):
      if v == 2:
        raise TestExc
      return True
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2, 3]).filter(bad_pred).run())

  async def test_gather_sync_error_propagates(self):
    """Sync gather: error in one fn propagates."""
    def bad_fn(v):
      raise TestExc
    with self.assertRaises(TestExc):
      Chain(5).gather(sync_double, bad_fn).run()

  async def test_gather_async_error_propagates(self):
    """Async gather: error in one fn propagates."""
    async def bad_fn(v):
      raise TestExc
    with self.assertRaises(TestExc):
      await await_(Chain(5).gather(async_double, bad_fn).run())

  async def test_foreach_error_with_except(self):
    """Foreach error caught by except_."""
    def bad_fn(v):
      if v == 2:
        raise TestExc
      return v
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    result = Chain([1, 2, 3]).foreach(bad_fn).except_(handler, reraise=False).run()
    self.assertEqual(result, 'caught')
    assert caught['ran']

  async def test_foreach_async_error_with_except(self):
    """Async foreach error caught by except_."""
    async def bad_fn(v):
      if v == 2:
        raise TestExc
      return v
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    result = Chain([1, 2, 3]).foreach(bad_fn).except_(handler, reraise=False).run()
    self.assertEqual(await result, 'caught')
    assert caught['ran']


if __name__ == '__main__':
  import unittest
  unittest.main()
