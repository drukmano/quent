"""Comprehensive combinatorial tests for every meaningful pair of Chain methods.

For each pair (A, B), tests:
  1. A followed by B
  2. B followed by A (where order matters)
  3. Both sync and async variants where applicable
  4. Edge cases specific to that pair

Methods covered: then, do, except_, finally_, with_, foreach, filter,
gather, return_, break_, iterate, clone.
"""

import asyncio
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null


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
class ThenDoPairTests(IsolatedAsyncioTestCase):

  async def test_then_do_result_flow(self):
    """then's result flows forward; do's result is discarded."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        log = []
        r = Chain(fn, 10).then(lambda v: v * 2).do(lambda v: log.append(v)).run()
        self.assertEqual(await await_(r), 20)
        self.assertEqual(log, [20])

  async def test_do_then_result_flow(self):
    """do discards its result; then gets the value from before do."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        r = Chain(fn, 5).do(lambda v: v * 100).then(lambda v: v + 1).run()
        self.assertEqual(await await_(r), 6)

  async def test_then_do_async_fn(self):
    """do with async fn still discards result."""
    log = []
    async def async_side(v):
      log.append(v)
      return 'ignored'
    r = Chain(3).then(lambda v: v + 7).do(async_side).run()
    self.assertEqual(await await_(r), 10)
    self.assertEqual(log, [10])

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
    self.assertEqual(await await_(r), 6)
    self.assertEqual(log, [('a', 2), ('b', 2), ('c', 6)])


# ===========================================================================
# 2. then + except_
# ===========================================================================
class ThenExceptPairTests(IsolatedAsyncioTestCase):

  async def test_then_except_catches(self):
    """except_ catches exception raised in then."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        caught = {}
        def handler(v):
          caught['ran'] = True
          return 'recovered'
        r = Chain(fn, 1).then(raise_exc1).except_(handler, reraise=False).run()
        self.assertEqual(await await_(r), 'recovered')
        self.assertTrue(caught['ran'])

  async def test_then_except_reraise(self):
    """except_ with reraise=True still raises after handler runs."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        caught = {}
        def handler(v):
          caught['ran'] = True
        with self.assertRaises(Exc1):
          await await_(Chain(fn, 1).then(raise_exc1).except_(handler).run())
        self.assertTrue(caught['ran'])

  async def test_except_then_order(self):
    """except_ only catches exceptions from links BEFORE it."""
    caught = {}
    def handler(v):
      caught['ran'] = True
    with self.assertRaises(Exc1):
      await await_(Chain(1).except_(handler, reraise=False).then(raise_exc1).run())
    self.assertNotIn('ran', caught)

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
    self.assertEqual(await await_(r), 'ok')
    self.assertNotIn('ran', caught_1)
    self.assertTrue(caught_2['ran'])

  async def test_then_async_except(self):
    """Async then raising, caught by except_."""
    caught = {}
    async def failing(v):
      raise TestExc()
    def handler(v):
      caught['ran'] = True
      return 42
    r = Chain(1).then(failing).except_(handler, reraise=False).run()
    self.assertEqual(await await_(r), 42)
    self.assertTrue(caught['ran'])


# ===========================================================================
# 3. then + finally_
# ===========================================================================
class ThenFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_then_finally_runs_on_success(self):
    """finally_ runs after successful then."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        log = []
        r = Chain(fn, 5).then(lambda v: v * 2).finally_(lambda v: log.append('fin')).run()
        self.assertEqual(await await_(r), 10)
        self.assertEqual(log, ['fin'])

  async def test_then_finally_runs_on_exception(self):
    """finally_ runs even when then raises."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        log = []
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 1).then(raise_).finally_(lambda v: log.append('fin')).run())
        self.assertEqual(log, ['fin'])

  async def test_then_finally_value_not_altered(self):
    """finally_ does not alter the chain's return value."""
    r = Chain(10).then(lambda v: v + 5).finally_(lambda v: 999).run()
    self.assertEqual(await await_(r), 15)


# ===========================================================================
# 4. then + foreach
# ===========================================================================
class ThenForeachPairTests(IsolatedAsyncioTestCase):

  async def test_then_foreach_basic(self):
    """then produces list, foreach transforms each element."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        r = Chain(fn, [1, 2, 3]).then(lambda v: v).foreach(lambda x: x * 10).run()
        self.assertEqual(await await_(r), [10, 20, 30])

  async def test_foreach_then(self):
    """foreach produces list, then transforms the list."""
    r = Chain([1, 2, 3]).foreach(lambda x: x + 1).then(lambda v: sum(v)).run()
    self.assertEqual(await await_(r), 9)

  async def test_then_foreach_async(self):
    """Async fn in foreach after then."""
    r = Chain([10, 20]).then(lambda v: v).foreach(lambda x: aempty(x * 2)).run()
    self.assertEqual(await r, [20, 40])

  async def test_then_foreach_empty(self):
    """then producing empty list, foreach on empty."""
    r = Chain([]).then(lambda v: v).foreach(lambda x: x * 2).run()
    self.assertEqual(await await_(r), [])


# ===========================================================================
# 5. then + filter
# ===========================================================================
class ThenFilterPairTests(IsolatedAsyncioTestCase):

  async def test_then_filter_basic(self):
    """then produces list, filter keeps matching elements."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        r = Chain(fn, [1, 2, 3, 4, 5]).then(lambda v: v).filter(lambda x: x % 2 == 0).run()
        self.assertEqual(await await_(r), [2, 4])

  async def test_filter_then(self):
    """filter produces filtered list, then transforms it."""
    r = Chain([1, 2, 3, 4]).filter(lambda x: x > 2).then(len).run()
    self.assertEqual(await await_(r), 2)

  async def test_then_filter_async_predicate(self):
    """Async predicate in filter."""
    r = Chain([1, 2, 3, 4]).filter(lambda x: aempty(x > 2)).run()
    self.assertEqual(await r, [3, 4])

  async def test_then_filter_all_rejected(self):
    """filter rejects all elements."""
    r = Chain([1, 2, 3]).filter(lambda x: False).run()
    self.assertEqual(await await_(r), [])

  async def test_then_filter_all_accepted(self):
    """filter accepts all elements."""
    r = Chain([1, 2, 3]).filter(lambda x: True).run()
    self.assertEqual(await await_(r), [1, 2, 3])


# ===========================================================================
# 6. then + gather
# ===========================================================================
class ThenGatherPairTests(IsolatedAsyncioTestCase):

  async def test_then_gather_basic(self):
    """then provides value, gather fans out to multiple fns."""
    r = Chain(5).then(lambda v: v).gather(
      lambda v: v + 1, lambda v: v + 2, lambda v: v + 3
    ).run()
    self.assertEqual(await await_(r), [6, 7, 8])

  async def test_gather_then(self):
    """gather produces list, then transforms it."""
    r = Chain(10).gather(
      lambda v: v * 1, lambda v: v * 2
    ).then(sum).run()
    self.assertEqual(await await_(r), 30)

  async def test_then_gather_async(self):
    """gather with async fns."""
    r = Chain(3).gather(
      lambda v: aempty(v + 10), lambda v: v + 20
    ).run()
    self.assertEqual(await r, [13, 23])

  async def test_then_gather_single_fn(self):
    """gather with a single fn."""
    r = Chain(7).gather(lambda v: v * 3).run()
    self.assertEqual(await await_(r), [21])


# ===========================================================================
# 7. then + with_
# ===========================================================================
class ThenWithPairTests(IsolatedAsyncioTestCase):

  async def test_then_with_sync(self):
    """then produces CM, with_ uses it."""
    cm = SyncCM('hello')
    r = Chain(cm).with_(lambda ctx: ctx + ' world').run()
    self.assertEqual(await await_(r), 'hello world')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_then_with_async_cm(self):
    """then produces async CM, with_ uses it."""
    cm = AsyncCM('async_hello')
    r = Chain(cm).with_(lambda ctx: ctx + '!').run()
    self.assertEqual(await r, 'async_hello!')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_with_then_result_flows(self):
    """with_ result flows to next then."""
    cm = SyncCM(42)
    r = Chain(cm).with_(lambda ctx: ctx * 2).then(lambda v: v + 1).run()
    self.assertEqual(await await_(r), 85)

  async def test_then_with_fn_pattern(self):
    """with_ with sync/async root via with_fn."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        cm = SyncCM(10)
        r = Chain(fn, cm).with_(lambda ctx: ctx + 5).run()
        self.assertEqual(await await_(r), 15)


# ===========================================================================
# 9. then + return_
# ===========================================================================
class ThenReturnPairTests(IsolatedAsyncioTestCase):

  async def test_then_return_exits_early(self):
    """return_ inside then exits chain early with value."""
    r = Chain(1).then(lambda v: Chain.return_(99)).then(lambda v: v + 1000).run()
    self.assertEqual(await await_(r), 99)

  async def test_return_with_callable(self):
    """return_ with a callable value evaluates it."""
    r = Chain(1).then(lambda v: Chain.return_(lambda: 42)).run()
    self.assertEqual(await await_(r), 42)

  async def test_return_none(self):
    """return_() with no value returns None."""
    r = Chain(1).then(lambda v: Chain.return_()).run()
    self.assertIsNone(await await_(r))


# ===========================================================================
# 10. then + break_
# ===========================================================================
class ThenBreakPairTests(IsolatedAsyncioTestCase):

  async def test_break_inside_foreach(self):
    """break_ inside foreach stops iteration."""
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = Chain([1, 2, 3, 4, 5]).foreach(fn).run()
    self.assertEqual(await await_(r), [10, 20])

  async def test_break_with_value_inside_foreach(self):
    """break_ with value replaces the collected list."""
    def fn(x):
      if x == 3:
        Chain.break_('stopped')
      return x
    r = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(await await_(r), 'stopped')

  async def test_break_outside_foreach_raises(self):
    """break_ outside foreach/iterate raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).then(lambda v: Chain.break_()).run()


# ===========================================================================
# 11. then + iterate
# ===========================================================================
class ThenIteratePairTests(IsolatedAsyncioTestCase):

  async def test_then_iterate_sync(self):
    """then produces iterable, iterate yields elements."""
    gen = Chain([10, 20, 30]).then(lambda v: v).iterate()
    r = list(gen)
    self.assertEqual(r, [10, 20, 30])

  async def test_then_iterate_with_fn(self):
    """iterate with transform fn."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x * 5)
    r = list(gen)
    self.assertEqual(r, [5, 10, 15])

  async def test_then_iterate_async(self):
    """iterate async iteration."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x + 100)
    r = []
    async for item in gen:
      r.append(item)
    self.assertEqual(r, [101, 102, 103])


# ===========================================================================
# 12. then + clone
# ===========================================================================
class ThenClonePairTests(IsolatedAsyncioTestCase):

  async def test_clone_independence(self):
    """Cloned chain is independent."""
    c = Chain(5).then(lambda v: v * 2)
    c2 = c.clone()
    c2.then(lambda v: v + 100)
    self.assertEqual(await await_(c.run()), 10)
    self.assertEqual(await await_(c2.run()), 110)

  async def test_clone_preserves_chain(self):
    """Clone preserves existing chain operations."""
    c = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3)
    c2 = c.clone()
    self.assertEqual(await await_(c.run()), 6)
    self.assertEqual(await await_(c2.run()), 6)


# ===========================================================================
# 14. do + except_
# ===========================================================================
class DoExceptPairTests(IsolatedAsyncioTestCase):

  async def test_do_except_catches(self):
    """except_ catches exception raised in do."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = Chain(1).do(raise_exc1).except_(handler, reraise=False).run()
    self.assertEqual(await await_(r), 'recovered')
    self.assertTrue(caught['ran'])

  async def test_do_except_do_discards_result(self):
    """do discards result; if no exception, except_ is skipped."""
    caught = {}
    def handler(v):
      caught['ran'] = True
    log = []
    r = Chain(5).do(lambda v: log.append(v)).except_(handler).run()
    self.assertEqual(await await_(r), 5)
    self.assertNotIn('ran', caught)
    self.assertEqual(log, [5])


# ===========================================================================
# 15. do + finally_
# ===========================================================================
class DoFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_do_finally_both_run(self):
    """Both do and finally_ execute."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        log = []
        r = Chain(fn, 10).do(lambda v: log.append('do')).finally_(lambda v: log.append('fin')).run()
        self.assertEqual(await await_(r), 10)
        self.assertEqual(log, ['do', 'fin'])

  async def test_do_raises_finally_still_runs(self):
    """finally_ runs even when do raises."""
    log = []
    with self.assertRaises(TestExc):
      await await_(Chain(1).do(raise_).finally_(lambda v: log.append('fin')).run())
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 16. do + foreach
# ===========================================================================
class DoForeachPairTests(IsolatedAsyncioTestCase):

  async def test_do_foreach(self):
    """do discards result, foreach gets the value before do."""
    log = []
    r = Chain([1, 2, 3]).do(lambda v: log.append('side')).foreach(lambda x: x * 2).run()
    self.assertEqual(await await_(r), [2, 4, 6])
    self.assertEqual(log, ['side'])

  async def test_foreach_do(self):
    """foreach produces list, do discards its own result, value stays."""
    log = []
    r = Chain([1, 2]).foreach(lambda x: x + 10).do(lambda v: log.append(v)).run()
    self.assertEqual(await await_(r), [11, 12])
    self.assertEqual(log, [[11, 12]])


# ===========================================================================
# 17. do + with_
# ===========================================================================
class DoWithPairTests(IsolatedAsyncioTestCase):

  async def test_do_then_with(self):
    """do side-effect before with_."""
    log = []
    cm = SyncCM(99)
    r = Chain(cm).do(lambda v: log.append('do')).with_(lambda ctx: ctx + 1).run()
    self.assertEqual(await await_(r), 100)
    self.assertEqual(log, ['do'])
    self.assertTrue(cm.entered)


# ===========================================================================
# 18. except_ + finally_
# ===========================================================================
class ExceptFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_except_finally_both_run_on_error(self):
    """Both except_ and finally_ run when an exception occurs."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        log = []
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(raise_)
            .except_(lambda v: log.append('exc'))
            .finally_(lambda v: log.append('fin'))
            .run()
          )
        self.assertIn('exc', log)
        self.assertIn('fin', log)

  async def test_except_noraise_finally(self):
    """except_ with reraise=False + finally_: no exception, finally_ runs."""
    log = []
    r = (
      Chain(1).then(raise_)
      .except_(lambda v: log.append('exc'), reraise=False)
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    self.assertIsNone(await await_(r))
    self.assertEqual(log, ['exc', 'fin'])

  async def test_finally_runs_on_success_no_except(self):
    """finally_ runs on success, except_ handler is not called."""
    log = []
    r = (
      Chain(5).then(lambda v: v + 1)
      .except_(lambda v: log.append('exc'))
      .finally_(lambda v: log.append('fin'))
      .run()
    )
    self.assertEqual(await await_(r), 6)
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 19. except_ + foreach
# ===========================================================================
class ExceptForeachPairTests(IsolatedAsyncioTestCase):

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
    self.assertEqual(await await_(r), 'recovered')
    self.assertTrue(caught['ran'])

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
    self.assertEqual(await r, 'caught')
    self.assertTrue(caught['ran'])


# ===========================================================================
# 20. except_ + with_
# ===========================================================================
class ExceptWithPairTests(IsolatedAsyncioTestCase):

  async def test_with_body_raises_except_catches(self):
    """Exception in with_ body caught by except_."""
    cm = SyncCM('val')
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'recovered'
    r = Chain(cm).with_(lambda ctx: raise_()).except_(handler, reraise=False).run()
    self.assertEqual(await await_(r), 'recovered')
    self.assertTrue(caught['ran'])
    self.assertTrue(cm.exited)

  async def test_async_with_body_raises_except_catches(self):
    """Exception in async with_ body caught by except_."""
    cm = AsyncCM('val')
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    r = Chain(cm).with_(lambda ctx: raise_()).except_(handler, reraise=False).run()
    self.assertEqual(await r, 'caught')
    self.assertTrue(caught['ran'])
    self.assertTrue(cm.exited)


# ===========================================================================
# 21. except_ + gather
# ===========================================================================
class ExceptGatherPairTests(IsolatedAsyncioTestCase):

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
    self.assertEqual(await await_(r), 'recovered')
    self.assertTrue(caught['ran'])

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
    self.assertEqual(await r, 'caught')
    self.assertTrue(caught['ran'])


# ===========================================================================
# 22. finally_ + foreach
# ===========================================================================
class FinallyForeachPairTests(IsolatedAsyncioTestCase):

  async def test_foreach_success_finally_runs(self):
    """finally_ runs after successful foreach."""
    log = []
    r = Chain([1, 2]).foreach(lambda x: x * 2).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), [2, 4])
    self.assertEqual(log, ['fin'])

  async def test_foreach_raises_finally_runs(self):
    """finally_ runs even when foreach raises."""
    log = []
    def fn(x):
      if x == 2:
        raise TestExc()
      return x
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2]).foreach(fn).finally_(lambda v: log.append('fin')).run())
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 23. finally_ + with_
# ===========================================================================
class FinallyWithPairTests(IsolatedAsyncioTestCase):

  async def test_with_success_finally_runs(self):
    """finally_ runs after successful with_."""
    log = []
    cm = SyncCM(5)
    r = Chain(cm).with_(lambda ctx: ctx * 2).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), 10)
    self.assertEqual(log, ['fin'])
    self.assertTrue(cm.exited)

  async def test_with_raises_finally_runs(self):
    """finally_ runs even when with_ body raises."""
    log = []
    cm = SyncCM(5)
    with self.assertRaises(TestExc):
      await await_(Chain(cm).with_(lambda ctx: raise_()).finally_(lambda v: log.append('fin')).run())
    self.assertEqual(log, ['fin'])
    self.assertTrue(cm.exited)


# ===========================================================================
# 24. finally_ + gather
# ===========================================================================
class FinallyGatherPairTests(IsolatedAsyncioTestCase):

  async def test_gather_success_finally_runs(self):
    """finally_ runs after successful gather."""
    log = []
    r = (
      Chain(5).gather(lambda v: v + 1, lambda v: v + 2)
      .finally_(lambda v: log.append('fin')).run()
    )
    self.assertEqual(await await_(r), [6, 7])
    self.assertEqual(log, ['fin'])

  async def test_gather_raises_finally_runs(self):
    """finally_ runs when gather raises."""
    log = []
    with self.assertRaises(TestExc):
      await await_(
        Chain(5).gather(lambda v: raise_())
        .finally_(lambda v: log.append('fin')).run()
      )
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 25. foreach + filter
# ===========================================================================
class ForeachFilterPairTests(IsolatedAsyncioTestCase):

  async def test_foreach_then_filter(self):
    """foreach transforms, then filter on the result."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .foreach(lambda x: x * 2)
      .then(lambda v: v)
      .filter(lambda x: x > 5)
      .run()
    )
    self.assertEqual(await await_(r), [6, 8, 10])

  async def test_filter_then_foreach(self):
    """filter first, then foreach on filtered result."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x % 2 == 1)
      .then(lambda v: v)
      .foreach(lambda x: x * 10)
      .run()
    )
    self.assertEqual(await await_(r), [10, 30, 50])

  async def test_filter_foreach_async(self):
    """Async filter then foreach."""
    r = (
      Chain([1, 2, 3, 4])
      .filter(lambda x: aempty(x > 2))
      .then(lambda v: v)
      .foreach(lambda x: x + 100)
      .run()
    )
    self.assertEqual(await r, [103, 104])


# ===========================================================================
# 26. foreach + with_
# ===========================================================================
class ForeachWithPairTests(IsolatedAsyncioTestCase):

  async def test_with_produces_list_foreach_transforms(self):
    """with_ body returns list, foreach transforms elements."""
    cm = SyncCM([1, 2, 3])
    r = Chain(cm).with_(lambda ctx: ctx).foreach(lambda x: x * 5).run()
    self.assertEqual(await await_(r), [5, 10, 15])

  async def test_async_with_foreach(self):
    """Async CM with_ body returns list, foreach transforms."""
    cm = AsyncCM([10, 20])
    r = Chain(cm).with_(lambda ctx: ctx).foreach(lambda x: x + 1).run()
    self.assertEqual(await r, [11, 21])


# ===========================================================================
# 27. foreach + gather
# ===========================================================================
class ForeachGatherPairTests(IsolatedAsyncioTestCase):

  async def test_foreach_then_gather(self):
    """foreach produces list, gather fans it out."""
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    self.assertEqual(await await_(r), [12, 3])

  async def test_gather_then_foreach(self):
    """gather produces list, foreach transforms each element."""
    r = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v + 2, lambda v: v + 3)
      .foreach(lambda x: x * 10)
      .run()
    )
    self.assertEqual(await await_(r), [60, 70, 80])


# ===========================================================================
# 28. foreach + return_
# ===========================================================================
class ForeachReturnPairTests(IsolatedAsyncioTestCase):

  async def test_return_inside_foreach(self):
    """return_ inside foreach exits the entire chain."""
    r = Chain([1, 2, 3]).foreach(
      lambda x: Chain.return_('early') if x == 2 else x
    ).then(lambda v: 'should not reach').run()
    self.assertEqual(await await_(r), 'early')

  async def test_return_no_value_inside_foreach(self):
    """return_() with no value inside foreach."""
    r = Chain([1, 2, 3]).foreach(
      lambda x: Chain.return_() if x == 2 else x
    ).run()
    self.assertIsNone(await await_(r))


# ===========================================================================
# 29. foreach + break_
# ===========================================================================
class ForeachBreakPairTests(IsolatedAsyncioTestCase):

  async def test_break_stops_foreach(self):
    """break_ inside foreach stops iteration early."""
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(await await_(r), [10, 20])

  async def test_break_with_value(self):
    """break_ with value replaces result."""
    def fn(x):
      if x == 3:
        Chain.break_([999])
      return x
    r = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(await await_(r), [999])

  async def test_break_on_first_element(self):
    """break_ on first element returns empty result."""
    def fn(x):
      Chain.break_()
      return x
    r = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(await await_(r), [])

  async def test_async_break(self):
    """break_ with async foreach fn."""
    async def fn(x):
      if x >= 2:
        Chain.break_()
      return x * 5
    r = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(await r, [5])


# ===========================================================================
# 30. filter + gather
# ===========================================================================
class FilterGatherPairTests(IsolatedAsyncioTestCase):

  async def test_filter_then_gather(self):
    """filter produces filtered list, gather fans it out."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 3)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    self.assertEqual(await await_(r), [9, 2])

  async def test_gather_then_filter(self):
    """gather produces list, filter on it."""
    r = (
      Chain(3)
      .gather(lambda v: v, lambda v: v * 2, lambda v: v * 3)
      .filter(lambda x: x > 3)
      .run()
    )
    self.assertEqual(await await_(r), [6, 9])


# ===========================================================================
# 31. filter + with_
# ===========================================================================
class FilterWithPairTests(IsolatedAsyncioTestCase):

  async def test_with_produces_list_filter(self):
    """with_ body returns list, filter filters it."""
    cm = SyncCM([1, 2, 3, 4, 5])
    r = Chain(cm).with_(lambda ctx: ctx).filter(lambda x: x % 2 == 0).run()
    self.assertEqual(await await_(r), [2, 4])


# ===========================================================================
# 32. gather + with_
# ===========================================================================
class GatherWithPairTests(IsolatedAsyncioTestCase):

  async def test_with_then_gather(self):
    """with_ body returns value, gather fans it out."""
    cm = SyncCM(10)
    r = Chain(cm).with_(lambda ctx: ctx).gather(
      lambda v: v + 1, lambda v: v * 2
    ).run()
    self.assertEqual(await await_(r), [11, 20])


# ===========================================================================
# 33. with_ + return_
# ===========================================================================
class WithReturnPairTests(IsolatedAsyncioTestCase):

  async def test_return_inside_with(self):
    """return_ inside with_ body exits chain early."""
    cm = SyncCM('val')
    r = Chain(cm).with_(lambda ctx: Chain.return_('early')).then(lambda v: 'no').run()
    self.assertEqual(await await_(r), 'early')
    self.assertTrue(cm.entered)

  async def test_return_inside_async_with(self):
    """return_ inside async with_ body exits chain early."""
    cm = AsyncCM('val')
    r = Chain(cm).with_(lambda ctx: Chain.return_('early_async')).then(lambda v: 'no').run()
    self.assertEqual(await r, 'early_async')
    self.assertTrue(cm.entered)


# ===========================================================================
# 34. with_ + break_
# ===========================================================================
class WithBreakPairTests(IsolatedAsyncioTestCase):

  async def test_break_inside_with_inside_foreach(self):
    """break_ inside with_ body that is inside foreach."""
    items = [SyncCM(1), SyncCM(2), SyncCM(3)]
    def fn(cm):
      with cm:
        if cm.value >= 2:
          Chain.break_()
        return cm.value * 10
    r = Chain(items).foreach(fn).run()
    self.assertEqual(await await_(r), [10])


# ===========================================================================
# 52. iterate + foreach
# ===========================================================================
class IterateForeachPairTests(IsolatedAsyncioTestCase):

  async def test_iterate_generator_used_in_foreach(self):
    """iterate produces generator, which can be consumed by foreach in another chain."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x * 2)
    r = Chain(gen).foreach(lambda x: x + 100).run()
    self.assertEqual(await await_(r), [102, 104, 106])

  async def test_iterate_async_foreach(self):
    """iterate produces async-capable generator for async foreach."""
    gen = Chain([10, 20]).iterate(lambda x: aempty(x + 5))
    r = []
    async for item in gen:
      r.append(item)
    self.assertEqual(r, [15, 25])


# ===========================================================================
# 53. clone + except_
# ===========================================================================
class CloneExceptPairTests(IsolatedAsyncioTestCase):

  async def test_clone_preserves_except(self):
    """Cloned chain preserves except_ handler."""
    c = Chain(1).then(raise_).except_(lambda v: 'caught', reraise=False)
    c2 = c.clone()
    self.assertEqual(await await_(c.run()), 'caught')
    self.assertEqual(await await_(c2.run()), 'caught')

  async def test_clone_except_independence(self):
    """Adding except_ to clone doesn't affect original."""
    c = Chain(1).then(raise_)
    c2 = c.clone()
    c2.except_(lambda v: 'caught', reraise=False)
    with self.assertRaises(TestExc):
      c.run()
    self.assertEqual(await await_(c2.run()), 'caught')


# ===========================================================================
# 54. clone + finally_
# ===========================================================================
class CloneFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_clone_preserves_finally(self):
    """Cloned chain preserves finally_ handler."""
    log = []
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: log.append('fin'))
    c2 = c.clone()
    self.assertEqual(await await_(c.run()), 2)
    self.assertEqual(await await_(c2.run()), 2)
    self.assertEqual(log, ['fin', 'fin'])


# ===========================================================================
# 55. clone + foreach
# ===========================================================================
class CloneForeachPairTests(IsolatedAsyncioTestCase):

  async def test_clone_with_foreach(self):
    """Cloned chain with foreach works independently."""
    c = Chain([1, 2, 3]).foreach(lambda x: x * 2)
    c2 = c.clone()
    self.assertEqual(await await_(c.run()), [2, 4, 6])
    self.assertEqual(await await_(c2.run()), [2, 4, 6])


# ===========================================================================
# 56. Nested chain + except_
# ===========================================================================
class NestedChainExceptPairTests(IsolatedAsyncioTestCase):

  async def test_nested_chain_except(self):
    """Nested chain's exception caught by outer except_."""
    inner = Chain().then(raise_)
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'caught'
    r = Chain(1).then(inner).except_(handler, reraise=False).run()
    self.assertEqual(await await_(r), 'caught')
    self.assertTrue(caught['ran'])


# ===========================================================================
# 57. Nested chain + finally_
# ===========================================================================
class NestedChainFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_nested_chain_finally(self):
    """finally_ runs after nested chain."""
    inner = Chain().then(lambda v: v * 10)
    log = []
    r = Chain(3).then(inner).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), 30)
    self.assertEqual(log, ['fin'])

  async def test_nested_chain_raises_finally_runs(self):
    """finally_ runs even when nested chain raises."""
    inner = Chain().then(raise_)
    log = []
    with self.assertRaises(TestExc):
      await await_(Chain(1).then(inner).finally_(lambda v: log.append('fin')).run())
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 58. Nested chain + foreach
# ===========================================================================
class NestedChainForeachPairTests(IsolatedAsyncioTestCase):

  async def test_foreach_with_chain(self):
    """foreach with chain as transform fn."""
    inner = Chain().then(lambda v: v * 3)
    r = Chain([1, 2, 3]).foreach(inner).run()
    self.assertEqual(await await_(r), [3, 6, 9])

  async def test_foreach_with_chain_async(self):
    """foreach with chain containing async fn."""
    inner = Chain().then(lambda v: aempty(v + 100))
    r = Chain([1, 2]).foreach(inner).run()
    self.assertEqual(await r, [101, 102])


# ===========================================================================
# 59. Nested chain + with_
# ===========================================================================
class NestedChainWithPairTests(IsolatedAsyncioTestCase):

  async def test_with_chain_body(self):
    """with_ body is a chain."""
    inner = Chain().then(lambda ctx: ctx + '_processed')
    cm = SyncCM('data')
    r = Chain(cm).with_(inner).run()
    self.assertEqual(await await_(r), 'data_processed')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


# ===========================================================================
# 60. do + filter
# ===========================================================================
class DoFilterPairTests(IsolatedAsyncioTestCase):

  async def test_do_filter(self):
    """do discards result, filter gets value before do."""
    log = []
    r = Chain([1, 2, 3, 4]).do(lambda v: log.append(len(v))).filter(lambda x: x > 2).run()
    self.assertEqual(await await_(r), [3, 4])
    self.assertEqual(log, [4])


# ===========================================================================
# 62. do + gather
# ===========================================================================
class DoGatherPairTests(IsolatedAsyncioTestCase):

  async def test_do_gather(self):
    """do discards result, gather gets value before do."""
    log = []
    r = (
      Chain(10)
      .do(lambda v: log.append(v))
      .gather(lambda v: v + 1, lambda v: v * 2)
      .run()
    )
    self.assertEqual(await await_(r), [11, 20])
    self.assertEqual(log, [10])

  async def test_gather_do(self):
    """gather produces list, do sees it but discards own result."""
    log = []
    r = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v + 2)
      .do(lambda v: log.append(v))
      .run()
    )
    self.assertEqual(await await_(r), [6, 7])
    self.assertEqual(log, [[6, 7]])


# ===========================================================================
# 72. filter + except_
# ===========================================================================
class FilterExceptPairTests(IsolatedAsyncioTestCase):

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
    self.assertEqual(await await_(r), 'recovered')
    self.assertTrue(caught['ran'])


# ===========================================================================
# 73. filter + finally_
# ===========================================================================
class FilterFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_filter_finally(self):
    """finally_ runs after filter."""
    log = []
    r = Chain([1, 2, 3]).filter(lambda x: x > 1).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), [2, 3])
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 77. gather + except_
# ===========================================================================
class GatherExceptPairTests(IsolatedAsyncioTestCase):

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
    self.assertEqual(await await_(r), 'ok')
    self.assertTrue(caught['ran'])


# ===========================================================================
# 78. gather + finally_
# ===========================================================================
class GatherFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_gather_finally(self):
    """finally_ after gather."""
    log = []
    r = Chain(3).gather(lambda v: v, lambda v: v * 2).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), [3, 6])
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 81. with_ + except_
# ===========================================================================
class WithExceptPairTests(IsolatedAsyncioTestCase):

  async def test_with_except_cm_exit_called(self):
    """with_ body raises, except_ catches, CM __exit__ is called."""
    cm = SyncCM('val')
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = Chain(cm).with_(lambda ctx: raise_()).except_(handler, reraise=False).run()
    self.assertEqual(await await_(r), 'ok')
    self.assertTrue(cm.exited)
    self.assertTrue(caught['ran'])


# ===========================================================================
# 82. with_ + finally_
# ===========================================================================
class WithFinallyPairTests(IsolatedAsyncioTestCase):

  async def test_with_finally(self):
    """with_ followed by finally_."""
    cm = SyncCM(5)
    log = []
    r = Chain(cm).with_(lambda ctx: ctx + 1).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), 6)
    self.assertTrue(cm.exited)
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 83. with_ + gather
# ===========================================================================
class WithGatherPairTests(IsolatedAsyncioTestCase):

  async def test_with_gather(self):
    """with_ produces value, gather fans it out."""
    cm = SyncCM(10)
    r = Chain(cm).with_(lambda ctx: ctx).gather(lambda v: v + 1, lambda v: v * 2).run()
    self.assertEqual(await await_(r), [11, 20])


# ===========================================================================
# 84. with_ + filter
# ===========================================================================
class WithFilterPairTests(IsolatedAsyncioTestCase):

  async def test_with_filter(self):
    """with_ produces list, filter filters."""
    cm = SyncCM([1, 2, 3, 4])
    r = Chain(cm).with_(lambda ctx: ctx).filter(lambda x: x > 2).run()
    self.assertEqual(await await_(r), [3, 4])


# ===========================================================================
# 85. with_ + foreach
# ===========================================================================
class WithForeachPairTests(IsolatedAsyncioTestCase):

  async def test_with_foreach(self):
    """with_ produces list, foreach transforms."""
    cm = SyncCM([10, 20])
    r = Chain(cm).with_(lambda ctx: ctx).foreach(lambda x: x + 5).run()
    self.assertEqual(await await_(r), [15, 25])


# ===========================================================================
# 90. then + then
# ===========================================================================
class ThenThenPairTests(IsolatedAsyncioTestCase):

  async def test_then_chain(self):
    """Multiple then calls chain results."""
    r = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).then(lambda v: v - 2).run()
    self.assertEqual(await await_(r), 4)

  async def test_then_async_chain(self):
    """then with alternating sync/async."""
    r = Chain(1).then(lambda v: v + 1).then(lambda v: aempty(v * 2)).then(lambda v: v + 10).run()
    self.assertEqual(await r, 14)


# ===========================================================================
# 91. do + do
# ===========================================================================
class DoDoPairTests(IsolatedAsyncioTestCase):

  async def test_multiple_do(self):
    """Multiple do calls all get the same value."""
    log = []
    r = Chain(5).do(lambda v: log.append(('a', v))).do(lambda v: log.append(('b', v))).run()
    self.assertEqual(await await_(r), 5)
    self.assertEqual(log, [('a', 5), ('b', 5)])


# ===========================================================================
# 92. except_ + except_ (multiple handlers)
# ===========================================================================
class ExceptExceptPairTests(IsolatedAsyncioTestCase):

  async def test_multiple_except_type_dispatch(self):
    """Multiple except_ handlers with different type filters."""
    log = []
    r = (
      Chain(1).then(raise_exc1)
      .except_(lambda v: log.append('h1'), exceptions=Exc2)
      .except_(lambda v: log.append('h2'), exceptions=Exc1, reraise=False)
      .run()
    )
    self.assertIsNone(await await_(r))
    self.assertEqual(log, ['h2'])

  async def test_first_matching_except_wins(self):
    """First matching except_ handler is used."""
    log = []
    r = (
      Chain(1).then(raise_exc1)
      .except_(lambda v: log.append('h1'), exceptions=Exc1, reraise=False)
      .except_(lambda v: log.append('h2'), exceptions=Exc1, reraise=False)
      .run()
    )
    self.assertIsNone(await await_(r))
    self.assertEqual(log, ['h1'])


# ===========================================================================
# 93. gather + gather
# ===========================================================================
class GatherGatherPairTests(IsolatedAsyncioTestCase):

  async def test_gather_then_gather(self):
    """gather produces list, second gather fans it out again."""
    r = (
      Chain(3)
      .gather(lambda v: v + 1, lambda v: v + 2)
      .gather(lambda v: sum(v), lambda v: len(v))
      .run()
    )
    self.assertEqual(await await_(r), [9, 2])


# ===========================================================================
# 94. foreach + foreach
# ===========================================================================
class ForeachForeachPairTests(IsolatedAsyncioTestCase):

  async def test_foreach_foreach(self):
    """foreach produces list, second foreach transforms the list elements."""
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: [x, x * 10])
      .foreach(lambda x: sum(x))
      .run()
    )
    self.assertEqual(await await_(r), [11, 22, 33])


# ===========================================================================
# 95. filter + filter
# ===========================================================================
class FilterFilterPairTests(IsolatedAsyncioTestCase):

  async def test_filter_filter(self):
    """Two filters in sequence."""
    r = Chain([1, 2, 3, 4, 5, 6]).filter(lambda x: x > 2).filter(lambda x: x < 5).run()
    self.assertEqual(await await_(r), [3, 4])


# ===========================================================================
# 96. with_ + with_ (nested CMs)
# ===========================================================================
class WithWithPairTests(IsolatedAsyncioTestCase):

  async def test_nested_with(self):
    """with_ result is CM, second with_ uses it."""
    outer_cm = SyncCM(SyncCM(42))
    r = Chain(outer_cm).with_(lambda ctx: ctx).with_(lambda ctx: ctx + 1).run()
    self.assertEqual(await await_(r), 43)
    self.assertTrue(outer_cm.entered)


# ===========================================================================
# 97. clone + do
# ===========================================================================
class CloneDoPairTests(IsolatedAsyncioTestCase):

  async def test_clone_do_independence(self):
    """Cloned chain with do is independent."""
    log1 = []
    log2 = []
    c = Chain(5).do(lambda v: log1.append(v))
    c2 = c.clone()
    c2.do(lambda v: log2.append(v * 10))
    self.assertEqual(await await_(c.run()), 5)
    self.assertEqual(await await_(c2.run()), 5)
    self.assertEqual(log1, [5, 5])  # both runs trigger log1's do
    self.assertEqual(log2, [50])


# ===========================================================================
# 98. clone + foreach
# ===========================================================================
class CloneForeachPairTests2(IsolatedAsyncioTestCase):

  async def test_clone_foreach_independence(self):
    """Cloned chain with foreach can be extended independently."""
    c = Chain([1, 2, 3]).foreach(lambda x: x * 2)
    c2 = c.clone()
    c2.then(sum)
    self.assertEqual(await await_(c.run()), [2, 4, 6])
    self.assertEqual(await await_(c2.run()), 12)


# ===========================================================================
# 99. clone + with_
# ===========================================================================
class CloneWithPairTests(IsolatedAsyncioTestCase):

  async def test_clone_with(self):
    """Cloned chain with with_."""
    cm1 = SyncCM(10)
    cm2 = SyncCM(10)
    c = Chain().with_(lambda ctx: ctx + 5)
    c2 = c.clone()
    self.assertEqual(await await_(c.run(cm1)), 15)
    self.assertEqual(await await_(c2.run(cm2)), 15)
    self.assertTrue(cm1.entered)
    self.assertTrue(cm2.entered)


# ===========================================================================
# 100. clone + gather
# ===========================================================================
class CloneGatherPairTests(IsolatedAsyncioTestCase):

  async def test_clone_gather(self):
    """Cloned chain with gather."""
    c = Chain(5).gather(lambda v: v + 1, lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(await await_(c.run()), [6, 10])
    self.assertEqual(await await_(c2.run()), [6, 10])


# ===========================================================================
# 101. clone + filter
# ===========================================================================
class CloneFilterPairTests(IsolatedAsyncioTestCase):

  async def test_clone_filter(self):
    """Cloned chain with filter."""
    c = Chain([1, 2, 3, 4]).filter(lambda x: x > 2)
    c2 = c.clone()
    self.assertEqual(await await_(c.run()), [3, 4])
    self.assertEqual(await await_(c2.run()), [3, 4])


# ===========================================================================
# 104. Three-method combos (important triple interactions)
# ===========================================================================
class ThreeMethodComboTests(IsolatedAsyncioTestCase):

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
    self.assertEqual(log, ['exc', 'fin'])

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
    self.assertEqual(await await_(r), 'caught')
    self.assertTrue(caught['ran'])

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
    self.assertEqual(await await_(r), [10, 20])
    self.assertEqual(log, ['fin'])

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
    self.assertIn('exc', log)
    self.assertIn('fin', log)
    self.assertTrue(cm.exited)

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
    self.assertEqual(await await_(r), 'ok')
    self.assertEqual(log, [1])
    self.assertTrue(caught['ran'])

  async def test_gather_then_foreach(self):
    """gather + then + foreach."""
    r = (
      Chain(2)
      .gather(lambda v: v, lambda v: v * 3, lambda v: v * 5)
      .then(lambda v: v)
      .foreach(lambda x: x + 100)
      .run()
    )
    self.assertEqual(await await_(r), [102, 106, 110])

  async def test_filter_foreach_then(self):
    """filter + foreach + then."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x % 2 == 1)
      .foreach(lambda x: x * 10)
      .then(sum)
      .run()
    )
    self.assertEqual(await await_(r), 90)  # (10+30+50)

  async def test_clone_then_except(self):
    """clone + then + except_: cloned chain also catches."""
    c = Chain(1).then(raise_).except_(lambda v: 'caught', reraise=False)
    c2 = c.clone()
    # except_ with reraise=False short-circuits the chain; further then() is not reached
    self.assertEqual(await await_(c.run()), 'caught')
    self.assertEqual(await await_(c2.run()), 'caught')

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
    self.assertEqual(await await_(r), [10, 20])
    self.assertEqual(log, [])  # break_ is not caught by except_

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
    self.assertEqual(await await_(r), 'early')
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 105. Pipe operator pairs
# ===========================================================================
class PipeOperatorPairTests(IsolatedAsyncioTestCase):

  async def test_pipe_then_except(self):
    """Pipe operator combined with except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = (Chain(1) | raise_).except_(handler, reraise=False).run()
    self.assertEqual(await await_(r), 'ok')
    self.assertTrue(caught['ran'])

  async def test_pipe_then_finally(self):
    """Pipe operator combined with finally_."""
    log = []
    r = Chain(5).then(lambda v: v + 1).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), 6)
    self.assertEqual(log, ['fin'])

  async def test_pipe_then_foreach(self):
    """Pipe operator producing list, then foreach."""
    r = Chain([1, 2, 3]).then(lambda v: v).foreach(lambda x: x * 2).run()
    self.assertEqual(await await_(r), [2, 4, 6])


# ===========================================================================
# 106. Void chain pairs (no root value)
# ===========================================================================
class VoidChainPairTests(IsolatedAsyncioTestCase):

  async def test_void_then_then(self):
    """Void chain with run(value) and then."""
    r = Chain().then(lambda v: v + 1).then(lambda v: v * 3).run(5)
    self.assertEqual(await await_(r), 18)

  async def test_void_then_except(self):
    """Void chain with then(raise) and except_."""
    caught = {}
    def handler(v):
      caught['ran'] = True
      return 'ok'
    r = Chain().then(raise_).except_(handler, reraise=False).run(1)
    self.assertEqual(await await_(r), 'ok')
    self.assertTrue(caught['ran'])

  async def test_void_then_finally(self):
    """Void chain with then and finally_."""
    log = []
    r = Chain().then(lambda v: v + 10).finally_(lambda v: log.append('fin')).run(5)
    self.assertEqual(await await_(r), 15)
    self.assertEqual(log, ['fin'])

  async def test_void_then_foreach(self):
    """Void chain producing list with foreach."""
    r = Chain().then(lambda v: v).foreach(lambda x: x * 2).run([1, 2, 3])
    self.assertEqual(await await_(r), [2, 4, 6])

  async def test_void_then_filter(self):
    """Void chain with filter."""
    r = Chain().then(lambda v: v).filter(lambda x: x > 2).run([1, 2, 3, 4])
    self.assertEqual(await await_(r), [3, 4])

  async def test_void_then_gather(self):
    """Void chain with gather."""
    r = Chain().then(lambda v: v).gather(lambda v: v + 1, lambda v: v * 2).run(5)
    self.assertEqual(await await_(r), [6, 10])


# ===========================================================================
# 107. Chain reuse pair tests
# ===========================================================================
class ChainReusePairTests(IsolatedAsyncioTestCase):

  async def test_chain_then_foreach(self):
    """Chain used in foreach."""
    inner = Chain().then(lambda v: v * 3)
    r = Chain([1, 2, 3]).foreach(inner).run()
    self.assertEqual(await await_(r), [3, 6, 9])

  async def test_chain_then_filter(self):
    """Chain used in filter."""
    inner = Chain().then(lambda v: v > 2)
    r = Chain([1, 2, 3, 4]).filter(inner).run()
    self.assertEqual(await await_(r), [3, 4])

  async def test_chain_reuse(self):
    """Chain can be reused multiple times."""
    c = Chain().then(lambda v: v + 10)
    self.assertEqual(await await_(c.run(1)), 11)
    self.assertEqual(await await_(c.run(2)), 12)
    self.assertEqual(await await_(c.run(3)), 13)

  async def test_chain_in_gather(self):
    """Chain used as fn in gather."""
    inner = Chain().then(lambda v: v * 100)
    r = Chain(5).gather(inner, lambda v: v + 1).run()
    self.assertEqual(await await_(r), [500, 6])


# ===========================================================================
# 108. Return_ propagation through nesting
# ===========================================================================
class ReturnPropagationTests(IsolatedAsyncioTestCase):

  async def test_return_propagates_through_nested(self):
    """return_ propagates through ALL nesting levels."""
    inner = Chain().then(lambda v: Chain.return_('from_inner'))
    r = Chain(1).then(inner).then(lambda v: 'should not reach').run()
    self.assertEqual(await await_(r), 'from_inner')

  async def test_return_in_nested_with_outer_finally(self):
    """return_ in nested chain, outer finally_ still runs."""
    log = []
    inner = Chain().then(lambda v: Chain.return_('early'))
    r = Chain(1).then(inner).finally_(lambda v: log.append('fin')).run()
    self.assertEqual(await await_(r), 'early')
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 109. Edge case: except_ with return value + then
# ===========================================================================
class ExceptReturnValuePairTests(IsolatedAsyncioTestCase):

  async def test_except_return_value_flows_to_nothing(self):
    """except_ with reraise=False returns its value; no further then executed."""
    r = (
      Chain(1).then(raise_)
      .except_(lambda v: 42, reraise=False)
      .run()
    )
    self.assertEqual(await await_(r), 42)

  async def test_except_return_async(self):
    """Async except_ handler with reraise=False."""
    async def handler(v):
      return 'async_recovery'
    r = Chain(1).then(raise_).except_(handler, reraise=False).run()
    self.assertEqual(await r, 'async_recovery')


# ===========================================================================
# 113. do + iterate
# ===========================================================================
class DoIteratePairTests(IsolatedAsyncioTestCase):

  async def test_do_does_not_affect_iterate(self):
    """do before iterate; iterate still gets root value."""
    gen = Chain([1, 2, 3]).do(lambda v: None).iterate(lambda x: x * 5)
    r = list(gen)
    self.assertEqual(r, [5, 10, 15])


# ===========================================================================
# 114. Additional edge case: nested chain error + outer except_ + finally_
# ===========================================================================
class NestedErrorOuterHandlingTests(IsolatedAsyncioTestCase):

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
    self.assertEqual(await await_(r), 'recovered')
    self.assertTrue(caught['ran'])
    self.assertEqual(log, ['fin'])


# ===========================================================================
# 116. Additional: clone + clone
# ===========================================================================
class CloneClonePairTests(IsolatedAsyncioTestCase):

  async def test_clone_of_clone(self):
    """Clone of a clone is independent."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    c3.then(lambda v: v * 100)
    self.assertEqual(await await_(c.run()), 2)
    self.assertEqual(await await_(c2.run()), 2)
    self.assertEqual(await await_(c3.run()), 200)


# ===========================================================================
# 118. Additional: with_ + do
# ===========================================================================
class WithDoPairTests(IsolatedAsyncioTestCase):

  async def test_with_do(self):
    """with_ followed by do."""
    cm = SyncCM(10)
    log = []
    r = Chain(cm).with_(lambda ctx: ctx * 2).do(lambda v: log.append(v)).run()
    self.assertEqual(await await_(r), 20)
    self.assertEqual(log, [20])


# ===========================================================================
# 119. Additional: foreach + do
# ===========================================================================
class ForeachDoDetailPairTests(IsolatedAsyncioTestCase):

  async def test_do_after_foreach(self):
    """do after foreach sees the list result."""
    log = []
    r = Chain([1, 2]).foreach(lambda x: x * 3).do(lambda v: log.append(v)).run()
    self.assertEqual(await await_(r), [3, 6])
    self.assertEqual(log, [[3, 6]])


# ===========================================================================
# 120. Additional: filter + do
# ===========================================================================
class FilterDoPairTests(IsolatedAsyncioTestCase):

  async def test_do_after_filter(self):
    """do after filter sees the filtered list."""
    log = []
    r = Chain([1, 2, 3, 4]).filter(lambda x: x > 2).do(lambda v: log.append(v)).run()
    self.assertEqual(await await_(r), [3, 4])
    self.assertEqual(log, [[3, 4]])


# ===========================================================================
# 121. Additional: gather + do
# ===========================================================================
class GatherDoPairTests(IsolatedAsyncioTestCase):

  async def test_do_after_gather(self):
    """do after gather sees the gathered list."""
    log = []
    r = Chain(5).gather(lambda v: v + 1).do(lambda v: log.append(v)).run()
    self.assertEqual(await await_(r), [6])
    self.assertEqual(log, [[6]])


if __name__ == '__main__':
  import unittest
  unittest.main()
