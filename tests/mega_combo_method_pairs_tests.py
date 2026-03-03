"""Comprehensive combinatorial tests for every meaningful pair of Chain methods.

For each pair (A, B), tests:
  1. A followed by B
  2. B followed by A (where order matters)
  3. Both sync and async variants where applicable
  4. Edge cases specific to that pair

Methods covered: then, do, except_, finally_, with_, foreach, filter,
gather, foreach_indexed, sleep, return_, break_, iterate, clone, to_thread.
"""

import asyncio
import threading
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SyncCM:
  """Sync context manager for testing."""
  def __init__(self, value='ctx'):
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
  """Async context manager for testing."""
  def __init__(self, value='async_ctx'):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


class Exc1(TestExc):
  pass


class Exc2(TestExc):
  pass


def raise_(v=None):
  raise TestExc()


def raise_exc1(v=None):
  raise Exc1()


def raise_exc2(v=None):
  raise Exc2()


async def async_raise(v=None):
  raise TestExc()


# ===========================================================================
# 1. then + do
# ===========================================================================
class ThenDoPairTests(MyTestCase):

  async def test_then_do_result_flow(self):
    """then's result flows forward; do's result is discarded."""
    for fn, ctx in self.with_fn():
      with ctx:
        log = []
        r = Chain(fn, 10).then(lambda v: v * 2).do(lambda v: log.append(v)).run()
        await self.assertEqual(r, 20)
        super(MyTestCase, self).assertEqual(log, [20])

  async def test_do_then_result_flow(self):
    """do discards its result; then gets the value from before do."""
    for fn, ctx in self.with_fn():
      with ctx:
        r = Chain(fn, 5).do(lambda v: v * 100).then(lambda v: v + 1).run()
        await self.assertEqual(r, 6)

  async def test_then_do_async_fn(self):
    """do with async fn still discards result."""
    log = []
    async def async_side(v):
      log.append(v)
      return 'ignored'
    r = Chain(3).then(lambda v: v + 7).do(async_side).run()
    await self.assertEqual(r, 10)
    super(MyTestCase, self).assertEqual(log, [10])

  async def test_multiple_do_between_then(self):
    """Multiple do calls between then calls."""
    log = []
    r = (
      Chain(1)
      .then(lambda v: v + 1)
      .do(lambda v: log.append(('a', v)))
      .do(lambda v: log.append(('b', v)))
      .then(lambda v: v * 3)
      .do(lambda v: log.append(('c', v)))
      .run()
    )
    await self.assertEqual(r, 6)
    super(MyTestCase, self).assertEqual(log, [('a', 2), ('b', 2), ('c', 6)])


# ===========================================================================
# 2. then + except_
# ===========================================================================
class ThenExceptPairTests(MyTestCase):

  async def test_then_except_catches(self):
    """except_ catches exception raised in then."""
    for fn, ctx in self.with_fn():
      with ctx:
        caught = {}
        def handler(v):
          caught['ran'] = True
          return 'recovered'
        r = Chain(fn, 1).then(raise_exc1).except_(handler, reraise=False).run()
        await self.assertEqual(r, 'recovered')
        super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_then_except_reraise(self):
    """except_ with reraise=True still raises after handler runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        caught = {}
        def handler(v):
          caught['ran'] = True
        with self.assertRaises(Exc1):
          await await_(Chain(fn, 1).then(raise_exc1).except_(handler).run())
        super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_except_then_order(self):
    """except_ only catches exceptions from links BEFORE it."""
    caught = {}
    def handler(v):
      caught['ran'] = True
    with self.assertRaises(Exc1):
      await await_(Chain(1).except_(handler, reraise=False).then(raise_exc1).run())
    super(MyTestCase, self).assertNotIn('ran', caught)

  async def test_then_except_type_filter(self):
    """except_ with exceptions= filters by type."""
    caught_1 = {}
    caught_2 = {}
    def h1(v): caught_1['ran'] = True
    def h2(v): caught_2['ran'] = True; return 'ok'
    r = (
      Chain(1).then(raise_exc1)
      .except_(h1, exceptions=Exc2)
      .except_(h2, exceptions=Exc1, reraise=False)
      .run()
    )
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertNotIn('ran', caught_1)
    super(MyTestCase, self).assertTrue(caught_2['ran'])

  async def test_then_async_except(self):
    """Async then raising, caught by except_."""
    caught = {}
    async def failing(v):
      raise TestExc()
    def handler(v):
      caught['ran'] = True
      return 42
    r = Chain(1).then(failing).except_(handler, reraise=False).run()
    await self.assertEqual(r, 42)
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 3. then + finally_
# ===========================================================================
class ThenFinallyPairTests(MyTestCase):

  async def test_then_finally_runs_on_success(self):
    """finally_ runs after successful then."""
    for fn, ctx in self.with_fn():
      with ctx:
        log = []
        r = Chain(fn, 5).then(lambda v: v * 2).finally_(lambda v: log.append('fin')).run()
        await self.assertEqual(r, 10)
        super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_then_finally_runs_on_exception(self):
    """finally_ runs even when then raises."""
    for fn, ctx in self.with_fn():
      with ctx:
        log = []
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 1).then(raise_).finally_(lambda v: log.append('fin')).run())
        super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_then_finally_value_not_altered(self):
    """finally_ does not alter the chain's return value."""
    r = Chain(10).then(lambda v: v + 5).finally_(lambda v: 999).run()
    await self.assertEqual(r, 15)


# ===========================================================================
# 4. then + foreach
# ===========================================================================
class ThenForeachPairTests(MyTestCase):

  async def test_then_foreach_basic(self):
    """then produces list, foreach transforms each element."""
    for fn, ctx in self.with_fn():
      with ctx:
        r = Chain(fn, [1, 2, 3]).then(lambda v: v).foreach(lambda x: x * 10).run()
        await self.assertEqual(r, [10, 20, 30])

  async def test_foreach_then(self):
    """foreach produces list, then transforms the list."""
    r = Chain([1, 2, 3]).foreach(lambda x: x + 1).then(lambda v: sum(v)).run()
    await self.assertEqual(r, 9)

  async def test_then_foreach_async(self):
    """Async fn in foreach after then."""
    r = Chain([10, 20]).then(lambda v: v).foreach(lambda x: aempty(x * 2)).run()
    await self.assertEqual(r, [20, 40])

  async def test_then_foreach_empty(self):
    """then producing empty list, foreach on empty."""
    r = Chain([]).then(lambda v: v).foreach(lambda x: x * 2).run()
    await self.assertEqual(r, [])


# ===========================================================================
# 5. then + filter
# ===========================================================================
class ThenFilterPairTests(MyTestCase):

  async def test_then_filter_basic(self):
    """then produces list, filter keeps matching elements."""
    for fn, ctx in self.with_fn():
      with ctx:
        r = Chain(fn, [1, 2, 3, 4, 5]).then(lambda v: v).filter(lambda x: x % 2 == 0).run()
        await self.assertEqual(r, [2, 4])

  async def test_filter_then(self):
    """filter produces filtered list, then transforms it."""
    r = Chain([1, 2, 3, 4]).filter(lambda x: x > 2).then(len).run()
    await self.assertEqual(r, 2)

  async def test_then_filter_async_predicate(self):
    """Async predicate in filter."""
    r = Chain([1, 2, 3, 4]).filter(lambda x: aempty(x > 2)).run()
    await self.assertEqual(r, [3, 4])

  async def test_then_filter_all_rejected(self):
    """filter rejects all elements."""
    r = Chain([1, 2, 3]).filter(lambda x: False).run()
    await self.assertEqual(r, [])

  async def test_then_filter_all_accepted(self):
    """filter accepts all elements."""
    r = Chain([1, 2, 3]).filter(lambda x: True).run()
    await self.assertEqual(r, [1, 2, 3])


# ===========================================================================
# 6. then + gather
# ===========================================================================
class ThenGatherPairTests(MyTestCase):

  async def test_then_gather_basic(self):
    """then provides value, gather fans out to multiple fns."""
    r = Chain(5).then(lambda v: v).gather(
      lambda v: v + 1, lambda v: v + 2, lambda v: v + 3
    ).run()
    await self.assertEqual(r, [6, 7, 8])

  async def test_gather_then(self):
    """gather produces list, then transforms it."""
    r = Chain(10).gather(
      lambda v: v * 1, lambda v: v * 2
    ).then(sum).run()
    await self.assertEqual(r, 30)

  async def test_then_gather_async(self):
    """gather with async fns."""
    r = Chain(3).gather(
      lambda v: aempty(v + 10), lambda v: v + 20
    ).run()
    await self.assertEqual(r, [13, 23])

  async def test_then_gather_single_fn(self):
    """gather with a single fn."""
    r = Chain(7).gather(lambda v: v * 3).run()
    await self.assertEqual(r, [21])


# ===========================================================================
# 7. then + with_
# ===========================================================================
class ThenWithPairTests(MyTestCase):

  async def test_then_with_sync(self):
    """then produces CM, with_ uses it."""
    cm = SyncCM('hello')
    r = Chain(cm).with_(lambda ctx: ctx + ' world').run()
    await self.assertEqual(r, 'hello world')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_then_with_async_cm(self):
    """then produces async CM, with_ uses it."""
    cm = AsyncCM('async_hello')
    r = Chain(cm).with_(lambda ctx: ctx + '!').run()
    await self.assertEqual(r, 'async_hello!')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_with_then_result_flows(self):
    """with_ result flows to next then."""
    cm = SyncCM(42)
    r = Chain(cm).with_(lambda ctx: ctx * 2).then(lambda v: v + 1).run()
    await self.assertEqual(r, 85)

  async def test_then_with_fn_pattern(self):
    """with_ with sync/async root via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        cm = SyncCM(10)
        r = Chain(fn, cm).with_(lambda ctx: ctx + 5).run()
        await self.assertEqual(r, 15)


# ===========================================================================
# 8. then + sleep
# ===========================================================================
class ThenSleepPairTests(MyTestCase):

  async def test_then_sleep_then(self):
    """sleep between two then calls preserves value."""
    r = Chain(5).then(lambda v: v * 2).sleep(0).then(lambda v: v + 1).run()
    await self.assertEqual(r, 11)

  async def test_sleep_then(self):
    """sleep before then."""
    r = Chain(3).sleep(0).then(lambda v: v + 7).run()
    await self.assertEqual(r, 10)

  async def test_sleep_does_not_alter_value(self):
    """sleep does not change the current value."""
    r = Chain(42).sleep(0).run()
    await self.assertEqual(r, 42)


# ===========================================================================
# 9. then + return_
# ===========================================================================
class ThenReturnPairTests(MyTestCase):

  async def test_then_return_exits_early(self):
    """return_ inside then exits chain early with value."""
    r = Chain(1).then(lambda v: Chain.return_(99)).then(lambda v: v + 1000).run()
    await self.assertEqual(r, 99)

  async def test_return_with_callable(self):
    """return_ with a callable value evaluates it."""
    r = Chain(1).then(lambda v: Chain.return_(lambda: 42)).run()
    await self.assertEqual(r, 42)

  async def test_return_none(self):
    """return_() with no value returns None."""
    r = Chain(1).then(lambda v: Chain.return_()).run()
    await self.assertIsNone(r)


# ===========================================================================
# 10. then + break_
# ===========================================================================
class ThenBreakPairTests(MyTestCase):

  async def test_break_inside_foreach(self):
    """break_ inside foreach stops iteration."""
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = Chain([1, 2, 3, 4, 5]).foreach(fn).run()
    await self.assertEqual(r, [10, 20])

  async def test_break_with_value_inside_foreach(self):
    """break_ with value replaces the collected list."""
    def fn(x):
      if x == 3:
        Chain.break_('stopped')
      return x
    r = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(r, 'stopped')

  async def test_break_outside_foreach_raises(self):
    """break_ outside foreach/iterate raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).then(lambda v: Chain.break_()).run()


# ===========================================================================
# 11. then + iterate
# ===========================================================================
class ThenIteratePairTests(MyTestCase):

  async def test_then_iterate_sync(self):
    """then produces iterable, iterate yields elements."""
    gen = Chain([10, 20, 30]).then(lambda v: v).iterate()
    r = list(gen)
    super(MyTestCase, self).assertEqual(r, [10, 20, 30])

  async def test_then_iterate_with_fn(self):
    """iterate with transform fn."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x * 5)
    r = list(gen)
    super(MyTestCase, self).assertEqual(r, [5, 10, 15])

  async def test_then_iterate_async(self):
    """iterate async iteration."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x + 100)
    r = []
    async for item in gen:
      r.append(item)
    super(MyTestCase, self).assertEqual(r, [101, 102, 103])


# ===========================================================================
# 12. then + clone
# ===========================================================================
class ThenClonePairTests(MyTestCase):

  async def test_clone_independence(self):
    """Cloned chain is independent."""
    c = Chain(5).then(lambda v: v * 2)
    c2 = c.clone()
    c2.then(lambda v: v + 100)
    await self.assertEqual(c.run(), 10)
    await self.assertEqual(c2.run(), 110)

  async def test_clone_preserves_chain(self):
    """Clone preserves existing chain operations."""
    c = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3)
    c2 = c.clone()
    await self.assertEqual(c.run(), 6)
    await self.assertEqual(c2.run(), 6)


# ===========================================================================
# 13. then + to_thread
# ===========================================================================
class ThenToThreadPairTests(MyTestCase):

  async def test_then_to_thread(self):
    """then followed by to_thread."""
    r = Chain(5).then(lambda v: v + 1).to_thread(lambda v: v * 3).run()
    await self.assertEqual(r, 18)

  async def test_to_thread_then(self):
    """to_thread followed by then."""
    r = Chain(4).to_thread(lambda v: v * 2).then(lambda v: v + 10).run()
    await self.assertEqual(r, 18)

  async def test_then_to_thread_async(self):
    """Async then followed by to_thread."""
    r = Chain(aempty, 3).then(lambda v: v + 2).to_thread(lambda v: v * 10).run()
    await self.assertEqual(r, 50)


# ===========================================================================
# 14. do + except_
# ===========================================================================
class DoExceptPairTests(MyTestCase):

  async def test_do_except_catches(self):
    """except_ catches exception raised in do."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = Chain(1).do(raise_exc1).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_do_except_do_discards_result(self):
    """do discards result; if no exception, except_ is skipped."""
    caught = {}
    def handler(v):
      caught['ran'] = True
    log = []
    r = Chain(5).do(lambda v: log.append(v)).except_(handler).run()
    await self.assertEqual(r, 5)
    super(MyTestCase, self).assertNotIn('ran', caught)
    super(MyTestCase, self).assertEqual(log, [5])


# ===========================================================================
# 15. do + finally_
# ===========================================================================
class DoFinallyPairTests(MyTestCase):

  async def test_do_finally_both_run(self):
    """Both do and finally_ execute."""
    for fn, ctx in self.with_fn():
      with ctx:
        log = []
        r = Chain(fn, 10).do(lambda v: log.append('do')).finally_(lambda v: log.append('fin')).run()
        await self.assertEqual(r, 10)
        super(MyTestCase, self).assertEqual(log, ['do', 'fin'])

  async def test_do_raises_finally_still_runs(self):
    """finally_ runs even when do raises."""
    log = []
    with self.assertRaises(TestExc):
      await await_(Chain(1).do(raise_).finally_(lambda v: log.append('fin')).run())
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 16. do + foreach
# ===========================================================================
class DoForeachPairTests(MyTestCase):

  async def test_do_foreach(self):
    """do discards result, foreach gets the value before do."""
    log = []
    r = Chain([1, 2, 3]).do(lambda v: log.append('side')).foreach(lambda x: x * 2).run()
    await self.assertEqual(r, [2, 4, 6])
    super(MyTestCase, self).assertEqual(log, ['side'])

  async def test_foreach_do(self):
    """foreach produces list, do discards its own result, value stays."""
    log = []
    r = Chain([1, 2]).foreach(lambda x: x + 10).do(lambda v: log.append(v)).run()
    await self.assertEqual(r, [11, 12])
    super(MyTestCase, self).assertEqual(log, [[11, 12]])


# ===========================================================================
# 17. do + with_
# ===========================================================================
class DoWithPairTests(MyTestCase):

  async def test_do_then_with(self):
    """do side-effect before with_."""
    log = []
    cm = SyncCM(99)
    r = Chain(cm).do(lambda v: log.append('do')).with_(lambda ctx: ctx + 1).run()
    await self.assertEqual(r, 100)
    super(MyTestCase, self).assertEqual(log, ['do'])
    super(MyTestCase, self).assertTrue(cm.entered)


# ===========================================================================
# 18. except_ + finally_
# ===========================================================================
class ExceptFinallyPairTests(MyTestCase):

  async def test_except_finally_both_run_on_error(self):
    """Both except_ and finally_ run when an exception occurs."""
    for fn, ctx in self.with_fn():
      with ctx:
        log = []
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(raise_)
            .except_(lambda v: log.append('exc'))
            .finally_(lambda v: log.append('fin'))
            .run()
          )
        super(MyTestCase, self).assertIn('exc', log)
        super(MyTestCase, self).assertIn('fin', log)

  async def test_except_noraise_finally(self):
    """except_ with reraise=False + finally_: no exception, finally_ runs."""
    log = []
    r = (
      Chain(1).then(raise_)
      .except_(lambda v: log.append('exc'), reraise=False)
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    await self.assertIsNone(r)
    super(MyTestCase, self).assertEqual(log, ['exc', 'fin'])

  async def test_finally_runs_on_success_no_except(self):
    """finally_ runs on success, except_ handler is not called."""
    log = []
    r = (
      Chain(5).then(lambda v: v + 1)
      .except_(lambda v: log.append('exc'))
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    await self.assertEqual(r, 6)
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 19. except_ + foreach
# ===========================================================================
class ExceptForeachPairTests(MyTestCase):

  async def test_foreach_raises_except_catches(self):
    """Exception in foreach fn caught by except_."""
    def fn(x):
      if x == 3:
        raise TestExc()
      return x
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = Chain([1, 2, 3, 4]).foreach(fn).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_foreach_async_raises_except_catches(self):
    """Async exception in foreach caught by except_."""
    async def fn(x):
      if x == 2:
        raise TestExc()
      return x * 10
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    r = Chain([1, 2, 3]).foreach(fn).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'caught')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 20. except_ + with_
# ===========================================================================
class ExceptWithPairTests(MyTestCase):

  async def test_with_body_raises_except_catches(self):
    """Exception in with_ body caught by except_."""
    cm = SyncCM('val')
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = Chain(cm).with_(lambda ctx: raise_()).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_async_with_body_raises_except_catches(self):
    """Exception in async with_ body caught by except_."""
    cm = AsyncCM('val')
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    r = Chain(cm).with_(lambda ctx: raise_()).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'caught')
    super(MyTestCase, self).assertTrue(caught['ran'])
    super(MyTestCase, self).assertTrue(cm.exited)


# ===========================================================================
# 21. except_ + gather
# ===========================================================================
class ExceptGatherPairTests(MyTestCase):

  async def test_gather_raises_except_catches(self):
    """Exception in gather fn caught by except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = (
      Chain(1).gather(lambda v: v, lambda v: raise_())
      .except_(handler, reraise=False).run()
    )
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_gather_async_raises_except_catches(self):
    """Async exception in gather fn caught by except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    r = (
      Chain(1).gather(lambda v: aempty(v), lambda v: async_raise(v))
      .except_(handler, reraise=False).run()
    )
    await self.assertEqual(r, 'caught')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 22. finally_ + foreach
# ===========================================================================
class FinallyForeachPairTests(MyTestCase):

  async def test_foreach_success_finally_runs(self):
    """finally_ runs after successful foreach."""
    log = []
    r = Chain([1, 2]).foreach(lambda x: x * 2).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, [2, 4])
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_foreach_raises_finally_runs(self):
    """finally_ runs even when foreach raises."""
    log = []
    def fn(x):
      if x == 2:
        raise TestExc()
      return x
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2]).foreach(fn).finally_(lambda v: log.append('fin')).run())
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 23. finally_ + with_
# ===========================================================================
class FinallyWithPairTests(MyTestCase):

  async def test_with_success_finally_runs(self):
    """finally_ runs after successful with_."""
    log = []
    cm = SyncCM(5)
    r = Chain(cm).with_(lambda ctx: ctx * 2).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 10)
    super(MyTestCase, self).assertEqual(log, ['fin'])
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_with_raises_finally_runs(self):
    """finally_ runs even when with_ body raises."""
    log = []
    cm = SyncCM(5)
    with self.assertRaises(TestExc):
      await await_(Chain(cm).with_(lambda ctx: raise_()).finally_(lambda v: log.append('fin')).run())
    super(MyTestCase, self).assertEqual(log, ['fin'])
    super(MyTestCase, self).assertTrue(cm.exited)


# ===========================================================================
# 24. finally_ + gather
# ===========================================================================
class FinallyGatherPairTests(MyTestCase):

  async def test_gather_success_finally_runs(self):
    """finally_ runs after successful gather."""
    log = []
    r = (
      Chain(5).gather(lambda v: v + 1, lambda v: v + 2)
      .finally_(lambda v: log.append('fin')).run()
    )
    await self.assertEqual(r, [6, 7])
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_gather_raises_finally_runs(self):
    """finally_ runs when gather raises."""
    log = []
    with self.assertRaises(TestExc):
      await await_(
        Chain(5).gather(lambda v: raise_())
        .finally_(lambda v: log.append('fin')).run()
      )
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 25. foreach + filter
# ===========================================================================
class ForeachFilterPairTests(MyTestCase):

  async def test_foreach_then_filter(self):
    """foreach transforms, then filter on the result."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .foreach(lambda x: x * 2)
      .then(lambda v: v)
      .filter(lambda x: x > 5)
      .run()
    )
    await self.assertEqual(r, [6, 8, 10])

  async def test_filter_then_foreach(self):
    """filter first, then foreach on filtered result."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x % 2 == 1)
      .then(lambda v: v)
      .foreach(lambda x: x * 10)
      .run()
    )
    await self.assertEqual(r, [10, 30, 50])

  async def test_filter_foreach_async(self):
    """Async filter then foreach."""
    r = (
      Chain([1, 2, 3, 4])
      .filter(lambda x: aempty(x > 2))
      .then(lambda v: v)
      .foreach(lambda x: x + 100)
      .run()
    )
    await self.assertEqual(r, [103, 104])


# ===========================================================================
# 26. foreach + with_
# ===========================================================================
class ForeachWithPairTests(MyTestCase):

  async def test_with_produces_list_foreach_transforms(self):
    """with_ body returns list, foreach transforms elements."""
    cm = SyncCM([1, 2, 3])
    r = Chain(cm).with_(lambda ctx: ctx).foreach(lambda x: x * 5).run()
    await self.assertEqual(r, [5, 10, 15])

  async def test_async_with_foreach(self):
    """Async CM with_ body returns list, foreach transforms."""
    cm = AsyncCM([10, 20])
    r = Chain(cm).with_(lambda ctx: ctx).foreach(lambda x: x + 1).run()
    await self.assertEqual(r, [11, 21])


# ===========================================================================
# 27. foreach + gather
# ===========================================================================
class ForeachGatherPairTests(MyTestCase):

  async def test_foreach_then_gather(self):
    """foreach produces list, gather fans it out."""
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    await self.assertEqual(r, [12, 3])

  async def test_gather_then_foreach(self):
    """gather produces list, foreach transforms each element."""
    r = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v + 2, lambda v: v + 3)
      .foreach(lambda x: x * 10)
      .run()
    )
    await self.assertEqual(r, [60, 70, 80])


# ===========================================================================
# 28. foreach + return_
# ===========================================================================
class ForeachReturnPairTests(MyTestCase):

  async def test_return_inside_foreach(self):
    """return_ inside foreach exits the entire chain."""
    r = Chain([1, 2, 3]).foreach(
      lambda x: Chain.return_('early') if x == 2 else x
    ).then(lambda v: 'should not reach').run()
    await self.assertEqual(r, 'early')

  async def test_return_no_value_inside_foreach(self):
    """return_() with no value inside foreach."""
    r = Chain([1, 2, 3]).foreach(
      lambda x: Chain.return_() if x == 2 else x
    ).run()
    await self.assertIsNone(r)


# ===========================================================================
# 29. foreach + break_
# ===========================================================================
class ForeachBreakPairTests(MyTestCase):

  async def test_break_stops_foreach(self):
    """break_ inside foreach stops iteration early."""
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(r, [10, 20])

  async def test_break_with_value(self):
    """break_ with value replaces result."""
    def fn(x):
      if x == 3:
        Chain.break_([999])
      return x
    r = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(r, [999])

  async def test_break_on_first_element(self):
    """break_ on first element returns empty result."""
    def fn(x):
      Chain.break_()
      return x
    r = Chain([1, 2, 3]).foreach(fn).run()
    await self.assertEqual(r, [])

  async def test_async_break(self):
    """break_ with async foreach fn."""
    async def fn(x):
      if x >= 2:
        Chain.break_()
      return x * 5
    r = Chain([1, 2, 3]).foreach(fn).run()
    await self.assertEqual(r, [5])


# ===========================================================================
# 30. filter + gather
# ===========================================================================
class FilterGatherPairTests(MyTestCase):

  async def test_filter_then_gather(self):
    """filter produces filtered list, gather fans it out."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 3)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    await self.assertEqual(r, [9, 2])

  async def test_gather_then_filter(self):
    """gather produces list, filter on it."""
    r = (
      Chain(3)
      .gather(lambda v: v, lambda v: v * 2, lambda v: v * 3)
      .filter(lambda x: x > 3)
      .run()
    )
    await self.assertEqual(r, [6, 9])


# ===========================================================================
# 31. filter + with_
# ===========================================================================
class FilterWithPairTests(MyTestCase):

  async def test_with_produces_list_filter(self):
    """with_ body returns list, filter filters it."""
    cm = SyncCM([1, 2, 3, 4, 5])
    r = Chain(cm).with_(lambda ctx: ctx).filter(lambda x: x % 2 == 0).run()
    await self.assertEqual(r, [2, 4])


# ===========================================================================
# 32. gather + with_
# ===========================================================================
class GatherWithPairTests(MyTestCase):

  async def test_with_then_gather(self):
    """with_ body returns value, gather fans it out."""
    cm = SyncCM(10)
    r = Chain(cm).with_(lambda ctx: ctx).gather(
      lambda v: v + 1, lambda v: v * 2
    ).run()
    await self.assertEqual(r, [11, 20])


# ===========================================================================
# 33. with_ + return_
# ===========================================================================
class WithReturnPairTests(MyTestCase):

  async def test_return_inside_with(self):
    """return_ inside with_ body exits chain early."""
    cm = SyncCM('val')
    r = Chain(cm).with_(lambda ctx: Chain.return_('early')).then(lambda v: 'no').run()
    await self.assertEqual(r, 'early')
    super(MyTestCase, self).assertTrue(cm.entered)

  async def test_return_inside_async_with(self):
    """return_ inside async with_ body exits chain early."""
    cm = AsyncCM('val')
    r = Chain(cm).with_(lambda ctx: Chain.return_('early_async')).then(lambda v: 'no').run()
    await self.assertEqual(r, 'early_async')
    super(MyTestCase, self).assertTrue(cm.entered)


# ===========================================================================
# 34. with_ + break_
# ===========================================================================
class WithBreakPairTests(MyTestCase):

  async def test_break_inside_with_inside_foreach(self):
    """break_ inside with_ body that is inside foreach."""
    items = [SyncCM(1), SyncCM(2), SyncCM(3)]
    def fn(cm):
      with cm:
        if cm.value >= 2:
          Chain.break_()
        return cm.value * 10
    r = Chain(items).foreach(fn).run()
    await self.assertEqual(r, [10])


# ===========================================================================
# 35. sleep + then
# ===========================================================================
class SleepThenPairTests(MyTestCase):

  async def test_sleep_between_computations(self):
    """sleep between then calls."""
    r = Chain(1).then(lambda v: v + 1).sleep(0).then(lambda v: v * 5).run()
    await self.assertEqual(r, 10)

  async def test_multiple_sleeps(self):
    """Multiple sleep calls in chain."""
    r = Chain(1).sleep(0).then(lambda v: v + 1).sleep(0).then(lambda v: v * 3).run()
    await self.assertEqual(r, 6)


# ===========================================================================
# 36. sleep + foreach
# ===========================================================================
class SleepForeachPairTests(MyTestCase):

  async def test_sleep_then_foreach(self):
    """sleep before foreach."""
    r = Chain([1, 2, 3]).sleep(0).foreach(lambda x: x * 2).run()
    await self.assertEqual(r, [2, 4, 6])


# ===========================================================================
# 37. sleep + except_
# ===========================================================================
class SleepExceptPairTests(MyTestCase):

  async def test_sleep_except(self):
    """sleep then exception, caught by except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = Chain(1).sleep(0).then(raise_).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 38. sleep + finally_
# ===========================================================================
class SleepFinallyPairTests(MyTestCase):

  async def test_sleep_finally(self):
    """finally_ runs after sleep chain."""
    log = []
    r = Chain(5).sleep(0).then(lambda v: v + 1).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 6)
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 39. sleep + do
# ===========================================================================
class SleepDoPairTests(MyTestCase):

  async def test_sleep_do(self):
    """sleep followed by do; do discards result."""
    log = []
    r = Chain(7).sleep(0).do(lambda v: log.append(v)).run()
    await self.assertEqual(r, 7)
    super(MyTestCase, self).assertEqual(log, [7])


# ===========================================================================
# 40. to_thread + except_
# ===========================================================================
class ToThreadExceptPairTests(MyTestCase):

  async def test_to_thread_raises_except_catches(self):
    """Exception in to_thread caught by except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    def failing(v):
      raise TestExc()
    r = Chain(1).to_thread(failing).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 41. to_thread + finally_
# ===========================================================================
class ToThreadFinallyPairTests(MyTestCase):

  async def test_to_thread_finally(self):
    """finally_ runs after to_thread."""
    log = []
    r = Chain(5).to_thread(lambda v: v * 2).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 10)
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_to_thread_raises_finally_still_runs(self):
    """finally_ runs even when to_thread raises."""
    log = []
    def failing(v):
      raise TestExc()
    with self.assertRaises(TestExc):
      await await_(Chain(1).to_thread(failing).finally_(lambda v: log.append('fin')).run())
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 42. to_thread + do
# ===========================================================================
class ToThreadDoPairTests(MyTestCase):

  async def test_to_thread_do(self):
    """to_thread followed by do; do discards its result."""
    log = []
    r = Chain(5).to_thread(lambda v: v * 2).do(lambda v: log.append(v)).run()
    await self.assertEqual(r, 10)
    super(MyTestCase, self).assertEqual(log, [10])

  async def test_do_to_thread(self):
    """do followed by to_thread; do does not alter value."""
    log = []
    r = Chain(3).do(lambda v: log.append(v)).to_thread(lambda v: v * 4).run()
    await self.assertEqual(r, 12)
    super(MyTestCase, self).assertEqual(log, [3])


# ===========================================================================
# 43. to_thread + foreach
# ===========================================================================
class ToThreadForeachPairTests(MyTestCase):

  async def test_to_thread_then_foreach(self):
    """to_thread produces list, foreach transforms."""
    r = Chain(3).to_thread(lambda v: list(range(v))).foreach(lambda x: x + 10).run()
    await self.assertEqual(r, [10, 11, 12])

  async def test_foreach_then_to_thread(self):
    """foreach transforms, to_thread on the result list."""
    r = Chain([1, 2, 3]).foreach(lambda x: x * 2).to_thread(lambda v: sum(v)).run()
    await self.assertEqual(r, 12)


# ===========================================================================
# 44. to_thread + filter
# ===========================================================================
class ToThreadFilterPairTests(MyTestCase):

  async def test_to_thread_filter(self):
    """to_thread produces list, filter filters."""
    r = Chain(5).to_thread(lambda v: list(range(v))).filter(lambda x: x > 2).run()
    await self.assertEqual(r, [3, 4])


# ===========================================================================
# 45. to_thread + gather
# ===========================================================================
class ToThreadGatherPairTests(MyTestCase):

  async def test_to_thread_gather(self):
    """to_thread produces value, gather fans it out."""
    r = Chain(5).to_thread(lambda v: v).gather(
      lambda v: v + 1, lambda v: v * 2
    ).run()
    await self.assertEqual(r, [6, 10])


# ===========================================================================
# 46. to_thread + with_
# ===========================================================================
class ToThreadWithPairTests(MyTestCase):

  async def test_to_thread_produces_cm_with_uses_it(self):
    """to_thread produces CM, with_ uses it."""
    cm = SyncCM(42)
    r = Chain(cm).to_thread(lambda v: v).with_(lambda ctx: ctx + 1).run()
    # to_thread receives the CM object itself, returns it; with_ uses it
    # Actually, to_thread(lambda v: v) returns the CM object, so with_ enters it
    await self.assertEqual(r, 43)
    super(MyTestCase, self).assertTrue(cm.entered)


# ===========================================================================
# 47. to_thread + sleep
# ===========================================================================
class ToThreadSleepPairTests(MyTestCase):

  async def test_to_thread_sleep(self):
    """to_thread followed by sleep."""
    r = Chain(5).to_thread(lambda v: v * 3).sleep(0).run()
    await self.assertEqual(r, 15)

  async def test_sleep_to_thread(self):
    """sleep followed by to_thread."""
    r = Chain(4).sleep(0).to_thread(lambda v: v + 6).run()
    await self.assertEqual(r, 10)


# ===========================================================================
# 48. foreach_indexed + then
# ===========================================================================
class ForeachIndexedThenPairTests(MyTestCase):

  async def test_foreach_indexed_basic(self):
    """foreach_indexed passes (index, element) to fn."""
    r = Chain([10, 20, 30]).foreach(lambda i, x: (i, x), with_index=True).run()
    await self.assertEqual(r, [(0, 10), (1, 20), (2, 30)])

  async def test_foreach_indexed_then(self):
    """foreach_indexed followed by then."""
    r = (
      Chain([10, 20, 30])
      .foreach(lambda i, x: x + i, with_index=True)
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 63)  # 10+0 + 20+1 + 30+2 = 63

  async def test_foreach_indexed_async(self):
    """foreach_indexed with async fn."""
    r = Chain([5, 10]).foreach(
      lambda i, x: aempty(x * (i + 1)), with_index=True
    ).run()
    await self.assertEqual(r, [5, 20])


# ===========================================================================
# 49. foreach_indexed + except_
# ===========================================================================
class ForeachIndexedExceptPairTests(MyTestCase):

  async def test_foreach_indexed_raises_except_catches(self):
    """Exception in foreach_indexed caught by except_."""
    def fn(i, x):
      if i == 2:
        raise TestExc()
      return x * 10
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = (
      Chain([1, 2, 3, 4])
      .foreach(fn, with_index=True)
      .except_(handler, reraise=False)
      .run()
    )
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 50. foreach_indexed + break_
# ===========================================================================
class ForeachIndexedBreakPairTests(MyTestCase):

  async def test_break_in_foreach_indexed(self):
    """break_ inside foreach_indexed stops iteration."""
    def fn(i, x):
      if i >= 2:
        Chain.break_()
      return x * 10
    r = Chain([5, 6, 7, 8]).foreach(fn, with_index=True).run()
    await self.assertEqual(r, [50, 60])


# ===========================================================================
# 51. foreach_indexed + finally_
# ===========================================================================
class ForeachIndexedFinallyPairTests(MyTestCase):

  async def test_foreach_indexed_finally(self):
    """finally_ runs after foreach_indexed."""
    log = []
    r = (
      Chain([1, 2])
      .foreach(lambda i, x: x + i, with_index=True)
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    await self.assertEqual(r, [1, 3])
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 52. iterate + foreach
# ===========================================================================
class IterateForeachPairTests(MyTestCase):

  async def test_iterate_generator_used_in_foreach(self):
    """iterate produces generator, which can be consumed by foreach in another chain."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x * 2)
    r = Chain(gen).foreach(lambda x: x + 100).run()
    await self.assertEqual(r, [102, 104, 106])

  async def test_iterate_async_foreach(self):
    """iterate produces async-capable generator for async foreach."""
    gen = Chain([10, 20]).iterate(lambda x: aempty(x + 5))
    r = []
    async for item in gen:
      r.append(item)
    super(MyTestCase, self).assertEqual(r, [15, 25])


# ===========================================================================
# 53. clone + except_
# ===========================================================================
class CloneExceptPairTests(MyTestCase):

  async def test_clone_preserves_except(self):
    """Cloned chain preserves except_ handler."""
    c = Chain(1).then(raise_).except_(lambda v: 'caught', reraise=False)
    c2 = c.clone()
    await self.assertEqual(c.run(), 'caught')
    await self.assertEqual(c2.run(), 'caught')

  async def test_clone_except_independence(self):
    """Adding except_ to clone doesn't affect original."""
    c = Chain(1).then(raise_)
    c2 = c.clone()
    c2.except_(lambda v: 'caught', reraise=False)
    with self.assertRaises(TestExc):
      c.run()
    await self.assertEqual(c2.run(), 'caught')


# ===========================================================================
# 54. clone + finally_
# ===========================================================================
class CloneFinallyPairTests(MyTestCase):

  async def test_clone_preserves_finally(self):
    """Cloned chain preserves finally_ handler."""
    log = []
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: log.append('fin'))
    c2 = c.clone()
    await self.assertEqual(c.run(), 2)
    await self.assertEqual(c2.run(), 2)
    super(MyTestCase, self).assertEqual(log, ['fin', 'fin'])


# ===========================================================================
# 55. clone + foreach
# ===========================================================================
class CloneForeachPairTests(MyTestCase):

  async def test_clone_with_foreach(self):
    """Cloned chain with foreach works independently."""
    c = Chain([1, 2, 3]).foreach(lambda x: x * 2)
    c2 = c.clone()
    await self.assertEqual(c.run(), [2, 4, 6])
    await self.assertEqual(c2.run(), [2, 4, 6])


# ===========================================================================
# 56. Nested chain (freeze) + except_
# ===========================================================================
class NestedFreezeExceptPairTests(MyTestCase):

  async def test_nested_chain_freeze_except(self):
    """Nested frozen chain's exception caught by outer except_."""
    inner = Chain().then(raise_).freeze()
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    r = Chain(1).then(inner).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'caught')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 57. Nested chain (freeze) + finally_
# ===========================================================================
class NestedFreezeFinallyPairTests(MyTestCase):

  async def test_nested_chain_freeze_finally(self):
    """finally_ runs after nested frozen chain."""
    inner = Chain().then(lambda v: v * 10).freeze()
    log = []
    r = Chain(3).then(inner).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 30)
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_nested_chain_freeze_raises_finally_runs(self):
    """finally_ runs even when nested frozen chain raises."""
    inner = Chain().then(raise_).freeze()
    log = []
    with self.assertRaises(TestExc):
      await await_(Chain(1).then(inner).finally_(lambda v: log.append('fin')).run())
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 58. Nested chain (freeze) + foreach
# ===========================================================================
class NestedFreezeForeachPairTests(MyTestCase):

  async def test_foreach_with_frozen_chain(self):
    """foreach with frozen chain as transform fn."""
    inner = Chain().then(lambda v: v * 3).freeze()
    r = Chain([1, 2, 3]).foreach(inner).run()
    await self.assertEqual(r, [3, 6, 9])

  async def test_foreach_with_frozen_chain_async(self):
    """foreach with frozen chain containing async fn."""
    inner = Chain().then(lambda v: aempty(v + 100)).freeze()
    r = Chain([1, 2]).foreach(inner).run()
    await self.assertEqual(r, [101, 102])


# ===========================================================================
# 59. Nested chain (freeze) + with_
# ===========================================================================
class NestedFreezeWithPairTests(MyTestCase):

  async def test_with_frozen_chain_body(self):
    """with_ body is a frozen chain."""
    inner = Chain().then(lambda ctx: ctx + '_processed').freeze()
    cm = SyncCM('data')
    r = Chain(cm).with_(inner).run()
    await self.assertEqual(r, 'data_processed')
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)


# ===========================================================================
# 60. Cascade-specific pair tests
# ===========================================================================
class CascadePairTests(MyTestCase):

  async def test_cascade_then_do(self):
    """Cascade: both then and do receive root value."""
    log = []
    r = Cascade(10).then(lambda v: log.append(('then', v))).do(lambda v: log.append(('do', v))).run()
    await self.assertEqual(r, 10)
    super(MyTestCase, self).assertEqual(log, [('then', 10), ('do', 10)])

  async def test_cascade_then_foreach(self):
    """Cascade: then gets root, foreach gets root."""
    log = []
    r = Cascade([1, 2, 3]).then(lambda v: log.append(len(v))).foreach(lambda x: x * 2).run()
    await self.assertEqual(r, [1, 2, 3])
    super(MyTestCase, self).assertEqual(log, [3])

  async def test_cascade_then_except(self):
    """Cascade: except_ catches from then."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      caught['val'] = v
      return 'ignored'
    r = Cascade(42).then(raise_).except_(handler, reraise=False).run()
    # except_ with reraise=False returns handler's result, not root
    await self.assertEqual(r, 'ignored')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_cascade_then_finally(self):
    """Cascade: finally_ runs and receives root."""
    log = []
    r = Cascade(99).then(lambda v: 'ignored').finally_(lambda v: log.append(v)).run()
    await self.assertEqual(r, 99)
    super(MyTestCase, self).assertEqual(log, [99])

  async def test_cascade_do_then(self):
    """Cascade: do and then both get root value."""
    log = []
    r = Cascade(7).do(lambda v: log.append(('do', v))).then(lambda v: log.append(('then', v))).run()
    await self.assertEqual(r, 7)
    super(MyTestCase, self).assertEqual(log, [('do', 7), ('then', 7)])

  async def test_cascade_with_gather(self):
    """Cascade: gather receives root value."""
    r = Cascade(5).gather(lambda v: v + 1, lambda v: v * 2).run()
    await self.assertEqual(r, 5)

  async def test_cascade_with_sleep(self):
    """Cascade: sleep doesn't alter root value."""
    r = Cascade(42).sleep(0).then(lambda v: v * 100).run()
    await self.assertEqual(r, 42)


# ===========================================================================
# 61. do + filter
# ===========================================================================
class DoFilterPairTests(MyTestCase):

  async def test_do_filter(self):
    """do discards result, filter gets value before do."""
    log = []
    r = Chain([1, 2, 3, 4]).do(lambda v: log.append(len(v))).filter(lambda x: x > 2).run()
    await self.assertEqual(r, [3, 4])
    super(MyTestCase, self).assertEqual(log, [4])


# ===========================================================================
# 62. do + gather
# ===========================================================================
class DoGatherPairTests(MyTestCase):

  async def test_do_gather(self):
    """do discards result, gather gets value before do."""
    log = []
    r = (
      Chain(10)
      .do(lambda v: log.append(v))
      .gather(lambda v: v + 1, lambda v: v * 2)
      .run()
    )
    await self.assertEqual(r, [11, 20])
    super(MyTestCase, self).assertEqual(log, [10])

  async def test_gather_do(self):
    """gather produces list, do sees it but discards own result."""
    log = []
    r = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v + 2)
      .do(lambda v: log.append(v))
      .run()
    )
    await self.assertEqual(r, [6, 7])
    super(MyTestCase, self).assertEqual(log, [[6, 7]])


# ===========================================================================
# 63. do + sleep
# ===========================================================================
class DoSleepPairTests(MyTestCase):

  async def test_do_sleep(self):
    """do then sleep; value preserved."""
    log = []
    r = Chain(8).do(lambda v: log.append(v)).sleep(0).run()
    await self.assertEqual(r, 8)
    super(MyTestCase, self).assertEqual(log, [8])


# ===========================================================================
# 64. do + to_thread
# ===========================================================================
class DoToThreadPairTests(MyTestCase):

  async def test_do_to_thread(self):
    """do then to_thread; do doesn't alter value."""
    log = []
    r = Chain(6).do(lambda v: log.append(v)).to_thread(lambda v: v * 5).run()
    await self.assertEqual(r, 30)
    super(MyTestCase, self).assertEqual(log, [6])


# ===========================================================================
# 65. except_ + sleep
# ===========================================================================
class ExceptSleepPairTests(MyTestCase):

  async def test_except_after_sleep_and_raise(self):
    """sleep then raise, except_ catches."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = Chain(1).sleep(0).then(raise_).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 66. except_ + to_thread
# ===========================================================================
class ExceptToThreadPairTests(MyTestCase):

  async def test_to_thread_raises_except_catches(self):
    """to_thread raises, except_ catches."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    def failing(v):
      raise TestExc()
    r = Chain(1).to_thread(failing).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 67. finally_ + sleep
# ===========================================================================
class FinallySleepPairTests(MyTestCase):

  async def test_sleep_finally(self):
    """finally_ runs after sleep."""
    log = []
    r = Chain(5).sleep(0).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 5)
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 68. finally_ + to_thread
# ===========================================================================
class FinallyToThreadPairTests(MyTestCase):

  async def test_to_thread_finally(self):
    """finally_ runs after to_thread."""
    log = []
    r = Chain(5).to_thread(lambda v: v * 3).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 15)
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 69. foreach + sleep
# ===========================================================================
class ForeachSleepPairTests(MyTestCase):

  async def test_foreach_sleep(self):
    """foreach followed by sleep; value preserved."""
    r = Chain([1, 2]).foreach(lambda x: x + 10).sleep(0).run()
    await self.assertEqual(r, [11, 12])


# ===========================================================================
# 70. foreach + to_thread
# ===========================================================================
class ForeachToThreadPairTests(MyTestCase):

  async def test_foreach_to_thread(self):
    """foreach followed by to_thread."""
    r = Chain([1, 2, 3]).foreach(lambda x: x * 2).to_thread(lambda v: sum(v)).run()
    await self.assertEqual(r, 12)


# ===========================================================================
# 71. filter + sleep
# ===========================================================================
class FilterSleepPairTests(MyTestCase):

  async def test_filter_sleep(self):
    """filter followed by sleep."""
    r = Chain([1, 2, 3, 4]).filter(lambda x: x > 2).sleep(0).run()
    await self.assertEqual(r, [3, 4])


# ===========================================================================
# 72. filter + except_
# ===========================================================================
class FilterExceptPairTests(MyTestCase):

  async def test_filter_raises_except_catches(self):
    """Exception in filter predicate caught by except_."""
    def pred(x):
      if x == 3:
        raise TestExc()
      return x > 1
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = Chain([1, 2, 3, 4]).filter(pred).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 73. filter + finally_
# ===========================================================================
class FilterFinallyPairTests(MyTestCase):

  async def test_filter_finally(self):
    """finally_ runs after filter."""
    log = []
    r = Chain([1, 2, 3]).filter(lambda x: x > 1).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, [2, 3])
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 74. filter + to_thread
# ===========================================================================
class FilterToThreadPairTests(MyTestCase):

  async def test_filter_to_thread(self):
    """filter followed by to_thread."""
    r = Chain([1, 2, 3, 4]).filter(lambda x: x % 2 == 0).to_thread(lambda v: sum(v)).run()
    await self.assertEqual(r, 6)


# ===========================================================================
# 75. gather + sleep
# ===========================================================================
class GatherSleepPairTests(MyTestCase):

  async def test_gather_sleep(self):
    """gather followed by sleep."""
    r = Chain(5).gather(lambda v: v + 1, lambda v: v + 2).sleep(0).run()
    await self.assertEqual(r, [6, 7])


# ===========================================================================
# 76. gather + to_thread
# ===========================================================================
class GatherToThreadPairTests(MyTestCase):

  async def test_gather_to_thread(self):
    """gather followed by to_thread."""
    r = Chain(5).gather(lambda v: v + 1, lambda v: v * 2).to_thread(lambda v: sum(v)).run()
    await self.assertEqual(r, 16)


# ===========================================================================
# 77. gather + except_
# ===========================================================================
class GatherExceptPairTests(MyTestCase):

  async def test_gather_all_sync_raises(self):
    """All sync gather fns, one raises, except_ catches."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = (
      Chain(1).gather(lambda v: v, lambda v: raise_())
      .except_(handler, reraise=False).run()
    )
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 78. gather + finally_
# ===========================================================================
class GatherFinallyPairTests(MyTestCase):

  async def test_gather_finally(self):
    """finally_ after gather."""
    log = []
    r = Chain(3).gather(lambda v: v, lambda v: v * 2).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, [3, 6])
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 79. with_ + sleep
# ===========================================================================
class WithSleepPairTests(MyTestCase):

  async def test_with_sleep(self):
    """with_ followed by sleep."""
    cm = SyncCM(10)
    r = Chain(cm).with_(lambda ctx: ctx * 2).sleep(0).run()
    await self.assertEqual(r, 20)
    super(MyTestCase, self).assertTrue(cm.exited)


# ===========================================================================
# 80. with_ + to_thread
# ===========================================================================
class WithToThreadPairTests(MyTestCase):

  async def test_with_to_thread(self):
    """with_ followed by to_thread."""
    cm = SyncCM(7)
    r = Chain(cm).with_(lambda ctx: ctx * 3).to_thread(lambda v: v + 1).run()
    await self.assertEqual(r, 22)
    super(MyTestCase, self).assertTrue(cm.exited)


# ===========================================================================
# 81. with_ + except_
# ===========================================================================
class WithExceptPairTests(MyTestCase):

  async def test_with_except_cm_exit_called(self):
    """with_ body raises, except_ catches, CM __exit__ is called."""
    cm = SyncCM('val')
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = Chain(cm).with_(lambda ctx: raise_()).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(cm.exited)
    super(MyTestCase, self).assertTrue(caught['ran'])


# ===========================================================================
# 82. with_ + finally_
# ===========================================================================
class WithFinallyPairTests(MyTestCase):

  async def test_with_finally(self):
    """with_ followed by finally_."""
    cm = SyncCM(5)
    log = []
    r = Chain(cm).with_(lambda ctx: ctx + 1).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 6)
    super(MyTestCase, self).assertTrue(cm.exited)
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 83. with_ + gather
# ===========================================================================
class WithGatherPairTests(MyTestCase):

  async def test_with_gather(self):
    """with_ produces value, gather fans it out."""
    cm = SyncCM(10)
    r = Chain(cm).with_(lambda ctx: ctx).gather(lambda v: v + 1, lambda v: v * 2).run()
    await self.assertEqual(r, [11, 20])


# ===========================================================================
# 84. with_ + filter
# ===========================================================================
class WithFilterPairTests(MyTestCase):

  async def test_with_filter(self):
    """with_ produces list, filter filters."""
    cm = SyncCM([1, 2, 3, 4])
    r = Chain(cm).with_(lambda ctx: ctx).filter(lambda x: x > 2).run()
    await self.assertEqual(r, [3, 4])


# ===========================================================================
# 85. with_ + foreach
# ===========================================================================
class WithForeachPairTests(MyTestCase):

  async def test_with_foreach(self):
    """with_ produces list, foreach transforms."""
    cm = SyncCM([10, 20])
    r = Chain(cm).with_(lambda ctx: ctx).foreach(lambda x: x + 5).run()
    await self.assertEqual(r, [15, 25])


# ===========================================================================
# 86. sleep + filter
# ===========================================================================
class SleepFilterPairTests(MyTestCase):

  async def test_sleep_filter(self):
    """sleep before filter."""
    r = Chain([1, 2, 3, 4]).sleep(0).filter(lambda x: x % 2 == 0).run()
    await self.assertEqual(r, [2, 4])


# ===========================================================================
# 87. sleep + gather
# ===========================================================================
class SleepGatherPairTests(MyTestCase):

  async def test_sleep_gather(self):
    """sleep before gather."""
    r = Chain(5).sleep(0).gather(lambda v: v + 1, lambda v: v * 2).run()
    await self.assertEqual(r, [6, 10])


# ===========================================================================
# 88. sleep + with_
# ===========================================================================
class SleepWithPairTests(MyTestCase):

  async def test_sleep_with(self):
    """sleep before with_."""
    cm = SyncCM(8)
    r = Chain(cm).sleep(0).with_(lambda ctx: ctx + 2).run()
    await self.assertEqual(r, 10)
    super(MyTestCase, self).assertTrue(cm.entered)


# ===========================================================================
# 89. sleep + sleep
# ===========================================================================
class SleepSleepPairTests(MyTestCase):

  async def test_double_sleep(self):
    """Two sleep calls in sequence."""
    r = Chain(42).sleep(0).sleep(0).run()
    await self.assertEqual(r, 42)


# ===========================================================================
# 90. then + then
# ===========================================================================
class ThenThenPairTests(MyTestCase):

  async def test_then_chain(self):
    """Multiple then calls chain results."""
    r = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).then(lambda v: v - 2).run()
    await self.assertEqual(r, 4)

  async def test_then_async_chain(self):
    """then with alternating sync/async."""
    r = Chain(1).then(lambda v: v + 1).then(lambda v: aempty(v * 2)).then(lambda v: v + 10).run()
    await self.assertEqual(r, 14)


# ===========================================================================
# 91. do + do
# ===========================================================================
class DoDoPairTests(MyTestCase):

  async def test_multiple_do(self):
    """Multiple do calls all get the same value."""
    log = []
    r = Chain(5).do(lambda v: log.append(('a', v))).do(lambda v: log.append(('b', v))).run()
    await self.assertEqual(r, 5)
    super(MyTestCase, self).assertEqual(log, [('a', 5), ('b', 5)])


# ===========================================================================
# 92. except_ + except_ (multiple handlers)
# ===========================================================================
class ExceptExceptPairTests(MyTestCase):

  async def test_multiple_except_type_dispatch(self):
    """Multiple except_ handlers with different type filters."""
    log = []
    r = (
      Chain(1).then(raise_exc1)
      .except_(lambda v: log.append('h1'), exceptions=Exc2)
      .except_(lambda v: log.append('h2'), exceptions=Exc1, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    super(MyTestCase, self).assertEqual(log, ['h2'])

  async def test_first_matching_except_wins(self):
    """First matching except_ handler is used."""
    log = []
    r = (
      Chain(1).then(raise_exc1)
      .except_(lambda v: log.append('h1'), exceptions=Exc1, reraise=False)
      .except_(lambda v: log.append('h2'), exceptions=Exc1, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    super(MyTestCase, self).assertEqual(log, ['h1'])


# ===========================================================================
# 93. gather + gather
# ===========================================================================
class GatherGatherPairTests(MyTestCase):

  async def test_gather_then_gather(self):
    """gather produces list, second gather fans it out again."""
    r = (
      Chain(3)
      .gather(lambda v: v + 1, lambda v: v + 2)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    await self.assertEqual(r, [9, 2])


# ===========================================================================
# 94. foreach + foreach
# ===========================================================================
class ForeachForeachPairTests(MyTestCase):

  async def test_foreach_foreach(self):
    """foreach produces list, second foreach transforms the list elements."""
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: [x, x * 10])
      .foreach(lambda x: sum(x))
      .run()
    )
    await self.assertEqual(r, [11, 22, 33])


# ===========================================================================
# 95. filter + filter
# ===========================================================================
class FilterFilterPairTests(MyTestCase):

  async def test_filter_filter(self):
    """Two filters in sequence."""
    r = Chain([1, 2, 3, 4, 5, 6]).filter(lambda x: x > 2).filter(lambda x: x < 5).run()
    await self.assertEqual(r, [3, 4])


# ===========================================================================
# 96. with_ + with_ (nested CMs)
# ===========================================================================
class WithWithPairTests(MyTestCase):

  async def test_nested_with(self):
    """with_ result is CM, second with_ uses it."""
    outer_cm = SyncCM(SyncCM(42))
    r = Chain(outer_cm).with_(lambda ctx: ctx).with_(lambda ctx: ctx + 1).run()
    await self.assertEqual(r, 43)
    super(MyTestCase, self).assertTrue(outer_cm.entered)


# ===========================================================================
# 97. clone + do
# ===========================================================================
class CloneDoPairTests(MyTestCase):

  async def test_clone_do_independence(self):
    """Cloned chain with do is independent."""
    log1 = []
    log2 = []
    c = Chain(5).do(lambda v: log1.append(v))
    c2 = c.clone()
    c2.do(lambda v: log2.append(v * 10))
    await self.assertEqual(c.run(), 5)
    await self.assertEqual(c2.run(), 5)
    super(MyTestCase, self).assertEqual(log1, [5, 5])  # both runs trigger log1's do
    super(MyTestCase, self).assertEqual(log2, [50])


# ===========================================================================
# 98. clone + foreach
# ===========================================================================
class CloneForeachPairTests2(MyTestCase):

  async def test_clone_foreach_independence(self):
    """Cloned chain with foreach can be extended independently."""
    c = Chain([1, 2, 3]).foreach(lambda x: x * 2)
    c2 = c.clone()
    c2.then(sum)
    await self.assertEqual(c.run(), [2, 4, 6])
    await self.assertEqual(c2.run(), 12)


# ===========================================================================
# 99. clone + with_
# ===========================================================================
class CloneWithPairTests(MyTestCase):

  async def test_clone_with(self):
    """Cloned chain with with_."""
    cm1 = SyncCM(10)
    cm2 = SyncCM(10)
    c = Chain().with_(lambda ctx: ctx + 5)
    c2 = c.clone()
    await self.assertEqual(c.run(cm1), 15)
    await self.assertEqual(c2.run(cm2), 15)
    super(MyTestCase, self).assertTrue(cm1.entered)
    super(MyTestCase, self).assertTrue(cm2.entered)


# ===========================================================================
# 100. clone + gather
# ===========================================================================
class CloneGatherPairTests(MyTestCase):

  async def test_clone_gather(self):
    """Cloned chain with gather."""
    c = Chain(5).gather(lambda v: v + 1, lambda v: v * 2)
    c2 = c.clone()
    await self.assertEqual(c.run(), [6, 10])
    await self.assertEqual(c2.run(), [6, 10])


# ===========================================================================
# 101. clone + filter
# ===========================================================================
class CloneFilterPairTests(MyTestCase):

  async def test_clone_filter(self):
    """Cloned chain with filter."""
    c = Chain([1, 2, 3, 4]).filter(lambda x: x > 2)
    c2 = c.clone()
    await self.assertEqual(c.run(), [3, 4])
    await self.assertEqual(c2.run(), [3, 4])


# ===========================================================================
# 102. clone + sleep
# ===========================================================================
class CloneSleepPairTests(MyTestCase):

  async def test_clone_sleep(self):
    """Cloned chain with sleep."""
    c = Chain(42).sleep(0)
    c2 = c.clone()
    await self.assertEqual(c.run(), 42)
    await self.assertEqual(c2.run(), 42)


# ===========================================================================
# 103. clone + to_thread
# ===========================================================================
class CloneToThreadPairTests(MyTestCase):

  async def test_clone_to_thread(self):
    """Cloned chain with to_thread."""
    c = Chain(3).to_thread(lambda v: v * 4)
    c2 = c.clone()
    c2.then(lambda v: v + 100)
    await self.assertEqual(c.run(), 12)
    await self.assertEqual(c2.run(), 112)


# ===========================================================================
# 104. Three-method combos (important triple interactions)
# ===========================================================================
class ThreeMethodComboTests(MyTestCase):

  async def test_then_except_finally(self):
    """then + except_ + finally_ all work together."""
    log = []
    with self.assertRaises(TestExc):
      await await_(
        Chain(1).then(raise_)
        .except_(lambda v: log.append('exc'))
        .finally_(lambda v: log.append('fin'))
        .run()
      )
    super(MyTestCase, self).assertEqual(log, ['exc', 'fin'])

  async def test_then_foreach_except(self):
    """then + foreach + except_."""
    def fn(x):
      if x == 3:
        raise TestExc()
      return x * 2
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    r = Chain([1, 2, 3]).then(lambda v: v).foreach(fn).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'caught')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_then_foreach_finally(self):
    """then + foreach + finally_."""
    log = []
    r = (
      Chain([1, 2])
      .then(lambda v: v)
      .foreach(lambda x: x * 10)
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    await self.assertEqual(r, [10, 20])
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_with_except_finally(self):
    """with_ + except_ + finally_."""
    cm = SyncCM('v')
    log = []
    with self.assertRaises(TestExc):
      await await_(
        Chain(cm).with_(lambda ctx: raise_())
        .except_(lambda v: log.append('exc'))
        .finally_(lambda v: log.append('fin'))
        .run()
      )
    super(MyTestCase, self).assertIn('exc', log)
    super(MyTestCase, self).assertIn('fin', log)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_do_then_except(self):
    """do + then(raise) + except_."""
    log = []
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = (
      Chain(1)
      .do(lambda v: log.append(v))
      .then(raise_)
      .except_(handler, reraise=False)
      .run()
    )
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertEqual(log, [1])
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_gather_then_foreach(self):
    """gather + then + foreach."""
    r = (
      Chain(2)
      .gather(lambda v: v, lambda v: v * 3, lambda v: v * 5)
      .then(lambda v: v)
      .foreach(lambda x: x + 100)
      .run()
    )
    await self.assertEqual(r, [102, 106, 110])

  async def test_filter_foreach_then(self):
    """filter + foreach + then."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x % 2 == 1)
      .foreach(lambda x: x * 10)
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 90)  # (10+30+50)

  async def test_sleep_then_except(self):
    """sleep + then(raise) + except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = Chain(1).sleep(0).then(raise_).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_to_thread_then_finally(self):
    """to_thread + then + finally_."""
    log = []
    r = (
      Chain(3)
      .to_thread(lambda v: v * 2)
      .then(lambda v: v + 1)
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    await self.assertEqual(r, 7)
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_clone_then_except(self):
    """clone + then + except_: cloned chain also catches."""
    c = Chain(1).then(raise_).except_(lambda v: 'caught', reraise=False)
    c2 = c.clone()
    # except_ with reraise=False short-circuits the chain; further then() is not reached
    await self.assertEqual(c.run(), 'caught')
    await self.assertEqual(c2.run(), 'caught')

  async def test_foreach_break_except(self):
    """foreach with break_ + except_: break_ is not an exception for except_."""
    log = []
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = (
      Chain([1, 2, 3, 4])
      .foreach(fn)
      .except_(lambda v: log.append('exc'), reraise=False)
      .run()
    )
    await self.assertEqual(r, [10, 20])
    super(MyTestCase, self).assertEqual(log, [])  # break_ is not caught by except_

  async def test_foreach_return_finally(self):
    """foreach with return_ + finally_: finally_ still runs."""
    log = []
    def fn(x):
      if x == 2:
        Chain.return_('early')
      return x
    r = (
      Chain([1, 2, 3])
      .foreach(fn)
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    await self.assertEqual(r, 'early')
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 105. Pipe operator pairs
# ===========================================================================
class PipeOperatorPairTests(MyTestCase):

  async def test_pipe_then_except(self):
    """Pipe operator combined with except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = (Chain(1) | raise_).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_pipe_then_finally(self):
    """Pipe operator combined with finally_."""
    log = []
    r = (Chain(5) | (lambda v: v + 1)).finally_(lambda v: log.append('fin')) | run()
    await self.assertEqual(r, 6)
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_pipe_then_foreach(self):
    """Pipe operator producing list, then foreach."""
    r = (Chain([1, 2, 3]) | (lambda v: v)).foreach(lambda x: x * 2) | run()
    await self.assertEqual(r, [2, 4, 6])


# ===========================================================================
# 106. Void chain pairs (no root value)
# ===========================================================================
class VoidChainPairTests(MyTestCase):

  async def test_void_then_then(self):
    """Void chain with run(value) and then."""
    r = Chain().then(lambda v: v + 1).then(lambda v: v * 3).run(5)
    await self.assertEqual(r, 18)

  async def test_void_then_except(self):
    """Void chain with then(raise) and except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = Chain().then(raise_).except_(handler, reraise=False).run(1)
    await self.assertEqual(r, 'ok')
    super(MyTestCase, self).assertTrue(caught['ran'])

  async def test_void_then_finally(self):
    """Void chain with then and finally_."""
    log = []
    r = Chain().then(lambda v: v + 10).finally_(lambda v: log.append('fin')).run(5)
    await self.assertEqual(r, 15)
    super(MyTestCase, self).assertEqual(log, ['fin'])

  async def test_void_then_foreach(self):
    """Void chain producing list with foreach."""
    r = Chain().then(lambda v: v).foreach(lambda x: x * 2).run([1, 2, 3])
    await self.assertEqual(r, [2, 4, 6])

  async def test_void_then_filter(self):
    """Void chain with filter."""
    r = Chain().then(lambda v: v).filter(lambda x: x > 2).run([1, 2, 3, 4])
    await self.assertEqual(r, [3, 4])

  async def test_void_then_gather(self):
    """Void chain with gather."""
    r = Chain().then(lambda v: v).gather(lambda v: v + 1, lambda v: v * 2).run(5)
    await self.assertEqual(r, [6, 10])


# ===========================================================================
# 107. Frozen chain pair tests
# ===========================================================================
class FrozenChainPairTests(MyTestCase):

  async def test_frozen_then_foreach(self):
    """Frozen chain used in foreach."""
    inner = Chain().then(lambda v: v * 3).freeze()
    r = Chain([1, 2, 3]).foreach(inner).run()
    await self.assertEqual(r, [3, 6, 9])

  async def test_frozen_then_filter(self):
    """Frozen chain used in filter."""
    inner = Chain().then(lambda v: v > 2).freeze()
    r = Chain([1, 2, 3, 4]).filter(inner).run()
    await self.assertEqual(r, [3, 4])

  async def test_frozen_reuse(self):
    """Frozen chain can be reused multiple times."""
    frozen = Chain().then(lambda v: v + 10).freeze()
    await self.assertEqual(frozen.run(1), 11)
    await self.assertEqual(frozen.run(2), 12)
    await self.assertEqual(frozen.run(3), 13)

  async def test_frozen_in_gather(self):
    """Frozen chain used as fn in gather."""
    inner = Chain().then(lambda v: v * 100).freeze()
    r = Chain(5).gather(inner, lambda v: v + 1).run()
    await self.assertEqual(r, [500, 6])


# ===========================================================================
# 108. Return_ propagation through nesting
# ===========================================================================
class ReturnPropagationTests(MyTestCase):

  async def test_return_propagates_through_nested(self):
    """return_ propagates through ALL nesting levels."""
    inner = Chain().then(lambda v: Chain.return_('from_inner'))
    r = Chain(1).then(inner).then(lambda v: 'should not reach').run()
    await self.assertEqual(r, 'from_inner')

  async def test_return_in_nested_with_outer_finally(self):
    """return_ in nested chain, outer finally_ still runs."""
    log = []
    inner = Chain().then(lambda v: Chain.return_('early'))
    r = Chain(1).then(inner).finally_(lambda v: log.append('fin')).run()
    await self.assertEqual(r, 'early')
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 109. Edge case: except_ with return value + then
# ===========================================================================
class ExceptReturnValuePairTests(MyTestCase):

  async def test_except_return_value_flows_to_nothing(self):
    """except_ with reraise=False returns its value; no further then executed."""
    r = (
      Chain(1).then(raise_)
      .except_(lambda v: 42, reraise=False)
      .run()
    )
    await self.assertEqual(r, 42)

  async def test_except_return_async(self):
    """Async except_ handler with reraise=False."""
    async def handler(v):
      return 'async_recovery'
    r = Chain(1).then(raise_).except_(handler, reraise=False).run()
    await self.assertEqual(r, 'async_recovery')


# ===========================================================================
# 110. foreach_indexed + filter (combined)
# ===========================================================================
class ForeachIndexedFilterPairTests(MyTestCase):

  async def test_foreach_indexed_then_filter(self):
    """foreach_indexed followed by filter."""
    r = (
      Chain([10, 20, 30, 40])
      .foreach(lambda i, x: x + i, with_index=True)
      .filter(lambda x: x > 21)
      .run()
    )
    await self.assertEqual(r, [32, 43])


# ===========================================================================
# 111. foreach_indexed + gather
# ===========================================================================
class ForeachIndexedGatherPairTests(MyTestCase):

  async def test_foreach_indexed_then_gather(self):
    """foreach_indexed followed by gather."""
    r = (
      Chain([1, 2, 3])
      .foreach(lambda i, x: x * (i + 1), with_index=True)
      .gather(sum, len)
      .run()
    )
    await self.assertEqual(r, [14, 3])  # [1*1, 2*2, 3*3] = [1,4,9], sum=14, len=3


# ===========================================================================
# 112. foreach_indexed + do
# ===========================================================================
class ForeachIndexedDoPairTests(MyTestCase):

  async def test_foreach_indexed_do(self):
    """foreach_indexed with do: do gets value before foreach_indexed."""
    log = []
    r = (
      Chain([10, 20])
      .do(lambda v: log.append(len(v)))
      .foreach(lambda i, x: (i, x), with_index=True)
      .run()
    )
    await self.assertEqual(r, [(0, 10), (1, 20)])
    super(MyTestCase, self).assertEqual(log, [2])


# ===========================================================================
# 113. do + iterate
# ===========================================================================
class DoIteratePairTests(MyTestCase):

  async def test_do_does_not_affect_iterate(self):
    """do before iterate; iterate still gets root value."""
    gen = Chain([1, 2, 3]).do(lambda v: None).iterate(lambda x: x * 5)
    r = list(gen)
    super(MyTestCase, self).assertEqual(r, [5, 10, 15])


# ===========================================================================
# 114. Additional edge case: nested chain error + outer except_ + finally_
# ===========================================================================
class NestedErrorOuterHandlingTests(MyTestCase):

  async def test_nested_error_outer_except_finally(self):
    """Error in nested chain caught by outer except_, outer finally_ runs."""
    inner = Chain().then(raise_)
    log = []
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = (
      Chain(1).then(inner)
      .except_(handler, reraise=False)
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    await self.assertEqual(r, 'recovered')
    super(MyTestCase, self).assertTrue(caught['ran'])
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ===========================================================================
# 115. Additional: then + foreach_indexed
# ===========================================================================
class ThenForeachIndexedPairTests(MyTestCase):

  async def test_then_foreach_indexed(self):
    """then produces list, foreach_indexed transforms with index."""
    for fn, ctx in self.with_fn():
      with ctx:
        r = Chain(fn, [5, 10, 15]).foreach(lambda i, x: x + i, with_index=True).run()
        await self.assertEqual(r, [5, 11, 17])


# ===========================================================================
# 116. Additional: clone + clone
# ===========================================================================
class CloneClonePairTests(MyTestCase):

  async def test_clone_of_clone(self):
    """Clone of a clone is independent."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    c3.then(lambda v: v * 100)
    await self.assertEqual(c.run(), 2)
    await self.assertEqual(c2.run(), 2)
    await self.assertEqual(c3.run(), 200)


# ===========================================================================
# 117. Additional: to_thread + to_thread
# ===========================================================================
class ToThreadToThreadPairTests(MyTestCase):

  async def test_double_to_thread(self):
    """Two to_thread calls in sequence."""
    r = Chain(2).to_thread(lambda v: v * 3).to_thread(lambda v: v + 4).run()
    await self.assertEqual(r, 10)


# ===========================================================================
# 118. Additional: with_ + do
# ===========================================================================
class WithDoPairTests(MyTestCase):

  async def test_with_do(self):
    """with_ followed by do."""
    cm = SyncCM(10)
    log = []
    r = Chain(cm).with_(lambda ctx: ctx * 2).do(lambda v: log.append(v)).run()
    await self.assertEqual(r, 20)
    super(MyTestCase, self).assertEqual(log, [20])


# ===========================================================================
# 119. Additional: foreach + do
# ===========================================================================
class ForeachDoDetailPairTests(MyTestCase):

  async def test_do_after_foreach(self):
    """do after foreach sees the list result."""
    log = []
    r = Chain([1, 2]).foreach(lambda x: x * 3).do(lambda v: log.append(v)).run()
    await self.assertEqual(r, [3, 6])
    super(MyTestCase, self).assertEqual(log, [[3, 6]])


# ===========================================================================
# 120. Additional: filter + do
# ===========================================================================
class FilterDoPairTests(MyTestCase):

  async def test_do_after_filter(self):
    """do after filter sees the filtered list."""
    log = []
    r = Chain([1, 2, 3, 4]).filter(lambda x: x > 2).do(lambda v: log.append(v)).run()
    await self.assertEqual(r, [3, 4])
    super(MyTestCase, self).assertEqual(log, [[3, 4]])


# ===========================================================================
# 121. Additional: gather + do
# ===========================================================================
class GatherDoPairTests(MyTestCase):

  async def test_do_after_gather(self):
    """do after gather sees the gathered list."""
    log = []
    r = Chain(5).gather(lambda v: v + 1).do(lambda v: log.append(v)).run()
    await self.assertEqual(r, [6])
    super(MyTestCase, self).assertEqual(log, [[6]])


if __name__ == '__main__':
  import unittest
  unittest.main()
