"""Comprehensive combinatorial test file for THREE-WAY feature interactions.

Tests A+B+C where three features interact together to produce emergent behavior
that would not be caught by pairwise testing alone.
"""
import asyncio
import logging
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, Null


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class SimpleCM:
  """Sync context manager for testing."""
  def __init__(self, value='ctx', exit_return=False):
    self.value = value
    self.exit_return = exit_return
    self.entered = False
    self.exited = False
    self.exit_args = None

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    self.exit_args = (exc_type, exc_val, exc_tb)
    return self.exit_return


class AsyncCM:
  """Async context manager for testing."""
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


class Tracker:
  """General-purpose call tracker."""
  def __init__(self):
    self.calls = []
    self.exc_called = False
    self.finally_called = False

  def on_except(self, v=None):
    """Exception handler that does NOT return root value (returns None)."""
    self.exc_called = True
    self.calls.append('except')

  def on_finally(self, v=None):
    self.finally_called = True
    self.calls.append('finally')


def raise_exc(v=None):
  raise TestExc('test error')


async def araise_exc(v=None):
  raise TestExc('async test error')


async def async_identity(v):
  return v


async def async_double(v):
  return v * 2


def sync_double(v):
  return v * 2


# Use unittest.TestCase base assertions for non-awaitable values
# by calling super(MyTestCase, self) -- or just use plain assert.

# =========================================================================
# Category 1: Exception handling triples
# =========================================================================

class ExceptionHandlingTriples(MyTestCase):

  # -- then + except_ + finally_ --
  async def test_then_except_finally_no_error(self):
    """then + except_ + finally_: no error, finally still runs."""
    t = Tracker()
    r = Chain(1).then(lambda v: v + 1).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 2)
    assert not t.exc_called
    assert t.finally_called

  async def test_then_except_finally_with_error(self):
    """then + except_ + finally_: error caught by except, finally runs."""
    t = Tracker()
    r = Chain(1).then(raise_exc).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    # except_ handler returns None (no explicit return), so chain result is None
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_then_except_finally_reraise(self):
    """then + except_ + finally_: error reraised, finally still runs."""
    t = Tracker()
    with self.assertRaises(TestExc):
      Chain(1).then(raise_exc).except_(t.on_except, reraise=True).finally_(t.on_finally).run()
    assert t.exc_called
    assert t.finally_called

  # -- do + except_ + finally_ --
  async def test_do_except_finally_no_error(self):
    """do discards result, except not triggered, finally runs."""
    t = Tracker()
    r = Chain(10).do(lambda v: v * 99).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 10)
    assert not t.exc_called
    assert t.finally_called

  async def test_do_except_finally_error_in_do(self):
    """Error in do's fn triggers except, finally runs."""
    t = Tracker()
    r = Chain(10).do(raise_exc).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  # -- foreach + except_ + finally_ --
  async def test_foreach_except_finally_no_error(self):
    """foreach iteration, no error, except not triggered, finally runs."""
    t = Tracker()
    r = Chain([1, 2, 3]).foreach(lambda x: x * 2).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, [2, 4, 6])
    assert not t.exc_called
    assert t.finally_called

  async def test_foreach_except_finally_error_in_iteration(self):
    """Error in foreach fn triggers except, finally runs."""
    t = Tracker()
    def fail_on_2(x):
      if x == 2:
        raise TestExc('fail on 2')
      return x
    r = Chain([1, 2, 3]).foreach(fail_on_2).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  # -- filter + except_ + finally_ --
  async def test_filter_except_finally_no_error(self):
    """filter predicate, no error, except not triggered, finally runs."""
    t = Tracker()
    r = Chain([1, 2, 3, 4]).filter(lambda x: x % 2 == 0).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, [2, 4])
    assert not t.exc_called
    assert t.finally_called

  async def test_filter_except_finally_error_in_predicate(self):
    """Error in filter predicate triggers except, finally runs."""
    t = Tracker()
    def bad_pred(x):
      if x == 3:
        raise TestExc('bad predicate')
      return x % 2 == 0
    r = Chain([1, 2, 3, 4]).filter(bad_pred).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  # -- gather + except_ + finally_ --
  async def test_gather_except_finally_no_error(self):
    """gather, no error, except not triggered, finally runs."""
    t = Tracker()
    r = Chain(5).gather(lambda v: v + 1, lambda v: v * 2).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, [6, 10])
    assert not t.exc_called
    assert t.finally_called

  async def test_gather_except_finally_error_in_one(self):
    """Error in one gather fn triggers except, finally runs."""
    t = Tracker()
    r = Chain(5).gather(lambda v: v + 1, raise_exc).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  # -- with_ + except_ + finally_ --
  async def test_with_except_finally_no_error(self):
    """CM + except + finally: no error, CM entered/exited, finally runs."""
    t = Tracker()
    cm = SimpleCM('val')
    r = Chain(cm).with_(lambda ctx: ctx.upper()).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 'VAL')
    assert cm.entered and cm.exited
    assert not t.exc_called
    assert t.finally_called

  async def test_with_except_finally_error_in_body(self):
    """Error in CM body triggers except, CM exits, finally runs."""
    t = Tracker()
    cm = SimpleCM('val')
    with self.assertRaises(TestExc):
      Chain(cm).with_(raise_exc).except_(t.on_except, reraise=True).finally_(t.on_finally).run()
    assert cm.entered and cm.exited
    assert t.exc_called
    assert t.finally_called

  # -- sleep + except_ + finally_ --
  async def test_sleep_except_finally(self):
    """sleep makes chain async, error after sleep caught, finally runs."""
    t = Tracker()
    r = await Chain(1).sleep(0.01).then(raise_exc).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_sleep_except_finally_no_error(self):
    """sleep + no error: except not triggered, finally runs."""
    t = Tracker()
    r = await Chain(42).sleep(0.01).then(lambda v: v + 1).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 43)
    assert not t.exc_called
    assert t.finally_called

  # -- to_thread + except_ + finally_ --
  async def test_to_thread_except_finally(self):
    """to_thread error caught, finally runs."""
    t = Tracker()
    r = await Chain(1).to_thread(raise_exc).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_to_thread_except_finally_no_error(self):
    """to_thread success, except not triggered, finally runs."""
    t = Tracker()
    r = await Chain(5).to_thread(sync_double).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 10)
    assert not t.exc_called
    assert t.finally_called

  # -- nested_chain + except_ + finally_ --
  async def test_nested_chain_except_finally_error_bubbles(self):
    """Error in nested chain bubbles up, caught by outer except, finally runs."""
    t = Tracker()
    inner = Chain().then(raise_exc)
    r = Chain(1).then(inner).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_nested_chain_except_finally_no_error(self):
    """Nested chain succeeds, except not triggered, finally runs."""
    t = Tracker()
    inner = Chain().then(lambda v: v * 10)
    r = Chain(3).then(inner).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 30)
    assert not t.exc_called
    assert t.finally_called


# =========================================================================
# Category 2: Iteration + control flow triples
# =========================================================================

class IterationControlFlowTriples(MyTestCase):

  # -- foreach + break_ + except_ --
  async def test_foreach_break_except_not_triggered(self):
    """Break during iteration does NOT trigger except_."""
    t = Tracker()
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 2
    r = Chain([1, 2, 3, 4, 5]).foreach(fn).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, [2, 4])
    assert not t.exc_called
    assert t.finally_called

  # -- foreach + break_ + finally_ --
  async def test_foreach_break_finally_triggered(self):
    """Break during iteration: finally_ IS triggered."""
    t = Tracker()
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 2
    r = Chain([1, 2, 3, 4]).foreach(fn).finally_(t.on_finally).run()
    await self.assertEqual(r, [2, 4])
    assert t.finally_called

  # -- foreach + return_ + except_ --
  async def test_foreach_return_except_not_triggered(self):
    """return_ during iteration exits chain, except not triggered."""
    t = Tracker()
    sentinel = object()
    def fn(x):
      if x == 2:
        Chain.return_(sentinel)
      return x
    r = Chain([1, 2, 3]).foreach(fn).except_(t.on_except, reraise=False).run()
    assert r is sentinel
    assert not t.exc_called

  # -- foreach + return_ + finally_ --
  async def test_foreach_return_finally_triggered(self):
    """return_ during iteration: finally IS triggered."""
    t = Tracker()
    sentinel = object()
    def fn(x):
      if x == 2:
        Chain.return_(sentinel)
      return x
    r = Chain([1, 2, 3]).foreach(fn).finally_(t.on_finally).run()
    assert r is sentinel
    assert t.finally_called

  # -- foreach_indexed + break_ + finally_ --
  async def test_foreach_indexed_break_finally(self):
    """Break in indexed foreach, finally runs."""
    t = Tracker()
    def fn(idx, el):
      if idx >= 2:
        Chain.break_()
      return (idx, el)
    r = Chain(['a', 'b', 'c', 'd']).foreach(fn, with_index=True).finally_(t.on_finally).run()
    await self.assertEqual(r, [(0, 'a'), (1, 'b')])
    assert t.finally_called

  # -- foreach_indexed + return_ + except_ --
  async def test_foreach_indexed_return_except(self):
    """return_ in indexed foreach, except not triggered."""
    t = Tracker()
    sentinel = object()
    def fn(idx, el):
      if idx == 1:
        Chain.return_(sentinel)
      return (idx, el)
    r = Chain(['a', 'b', 'c']).foreach(fn, with_index=True).except_(t.on_except, reraise=False).run()
    assert r is sentinel
    assert not t.exc_called

  # -- filter + break_ + finally_ --
  async def test_filter_break_raises_through_finally_still_runs(self):
    """filter does NOT handle break: it raises through. finally still runs."""
    t = Tracker()
    def pred(x):
      if x == 3:
        Chain.break_()
      return x % 2 == 0
    # _Break propagates through filter -> chain -> QuentException
    with self.assertRaises(QuentException):
      Chain([1, 2, 3, 4]).filter(pred).finally_(t.on_finally).run()
    assert t.finally_called

  # -- foreach + break_ + do --
  async def test_foreach_break_do_side_effects(self):
    """Break in foreach with do side-effect: do runs, break partial."""
    side = []
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 2
    r = Chain([1, 2, 3, 4]).foreach(fn).do(lambda v: side.append(len(v))).run()
    await self.assertEqual(r, [2, 4])
    assert side == [2]

  # -- foreach + break_ + then --
  async def test_foreach_break_then_transform(self):
    """Break in foreach: then transforms the partial result."""
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = Chain([1, 2, 3, 4]).foreach(fn).then(lambda lst: sum(lst)).run()
    await self.assertEqual(r, 30)  # 10 + 20


# =========================================================================
# Category 3: Context manager triples
# =========================================================================

class ContextManagerTriples(MyTestCase):

  # -- with_ + then + except_ --
  async def test_with_then_except_no_error(self):
    """CM body via with_, then transforms, except not triggered."""
    t = Tracker()
    cm = SimpleCM('hello')
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx.upper())
      .then(lambda v: v + '!')
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertEqual(r, 'HELLO!')
    assert not t.exc_called

  async def test_with_then_except_error_in_then(self):
    """CM body ok, error in then caught by except."""
    t = Tracker()
    cm = SimpleCM('hello')
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .then(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called
    assert cm.exited

  # -- with_ + do + finally_ --
  async def test_with_do_finally(self):
    """CM body, side-effect via do, cleanup via finally."""
    t = Tracker()
    side = []
    cm = SimpleCM('val')
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx.upper())
      .do(lambda v: side.append(v))
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 'VAL')
    assert side == ['VAL']
    assert t.finally_called

  # -- with_ + foreach + except_ --
  async def test_with_foreach_except_no_error(self):
    """Iterate result of CM body, no error."""
    t = Tracker()
    cm = SimpleCM([1, 2, 3])
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .foreach(lambda x: x * 2)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertEqual(r, [2, 4, 6])
    assert not t.exc_called

  async def test_with_foreach_except_error_in_iteration(self):
    """Error in iteration after CM body caught by except."""
    t = Tracker()
    cm = SimpleCM([1, 2, 3])
    def fail_on_2(x):
      if x == 2:
        raise TestExc()
      return x
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .foreach(fail_on_2)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- with_ + foreach + break_ --
  async def test_with_foreach_break(self):
    """Break inside foreach that follows with_."""
    cm = SimpleCM([1, 2, 3, 4, 5])
    def fn(x):
      if x >= 4:
        Chain.break_()
      return x * 10
    r = Chain(cm).with_(lambda ctx: ctx).foreach(fn).run()
    await self.assertEqual(r, [10, 20, 30])
    assert cm.exited

  # -- with_ + gather + except_ --
  async def test_with_gather_except(self):
    """Gather inside CM, error in one gather fn."""
    t = Tracker()
    cm = SimpleCM(10)
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .gather(lambda v: v + 1, raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- with_ + filter + finally_ --
  async def test_with_filter_finally(self):
    """Filter result of CM body, finally runs."""
    t = Tracker()
    cm = SimpleCM([1, 2, 3, 4, 5])
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .filter(lambda x: x > 3)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [4, 5])
    assert t.finally_called

  # -- with_ + return_ + finally_ --
  async def test_with_return_finally(self):
    """return_ from CM body exits chain early, finally still runs."""
    t = Tracker()
    sentinel = object()
    cm = SimpleCM('val')
    def body(ctx):
      Chain.return_(sentinel)
    r = Chain(cm).with_(body).then(lambda v: 'should not reach').finally_(t.on_finally).run()
    assert r is sentinel
    assert t.finally_called

  # -- with_ + nested_chain + except_ --
  async def test_with_nested_chain_except(self):
    """Nested chain inside CM body, error caught by except."""
    t = Tracker()
    inner = Chain().then(raise_exc)
    cm = SimpleCM('val')
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .then(inner)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- with_ + sleep + finally_ (async CM) --
  async def test_with_sleep_finally_async(self):
    """Async CM with sleep, finally runs."""
    t = Tracker()
    cm = AsyncCM('async_val')
    r = await (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .sleep(0.01)
      .then(lambda v: v.upper())
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 'ASYNC_VAL')
    assert t.finally_called
    assert cm.exited


# =========================================================================
# Category 4: Iteration + iteration triples
# =========================================================================

class IterationIterationTriples(MyTestCase):

  # -- foreach + foreach + then (nested foreach) --
  async def test_foreach_foreach_then(self):
    """foreach produces list of lists, flatten, then aggregate."""
    r = (
      Chain([[1, 2], [3, 4]])
      .foreach(lambda lst: [x * 10 for x in lst])
      .then(lambda lsts: [item for sublist in lsts for item in sublist])
      .run()
    )
    await self.assertEqual(r, [10, 20, 30, 40])

  # -- foreach + filter + then --
  async def test_foreach_filter_then(self):
    """foreach transforms, filter keeps some, then aggregates."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .foreach(lambda x: x * 2)
      .filter(lambda x: x > 4)
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 6 + 8 + 10)

  # -- filter + foreach + then --
  async def test_filter_foreach_then(self):
    """filter first, foreach transforms remaining, then aggregates."""
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x % 2 == 1)
      .foreach(lambda x: x * 10)
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 10 + 30 + 50)

  # -- foreach + gather + then --
  async def test_foreach_gather_then(self):
    """foreach produces list, gather applies fns to list, then uses results."""
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .gather(lambda v: sum(v), lambda v: len(v))
      .then(lambda results: results[0] / results[1])
      .run()
    )
    await self.assertEqual(r, 4.0)

  # -- gather + foreach + filter --
  async def test_gather_foreach_filter(self):
    """gather results, foreach transforms, filter keeps."""
    r = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v * 2, lambda v: v - 1)
      .foreach(lambda x: x * 10)
      .filter(lambda x: x > 50)
      .run()
    )
    await self.assertEqual(r, [60, 100])

  # -- foreach_indexed + filter + then --
  async def test_foreach_indexed_filter_then(self):
    """Indexed foreach, filter by index, then aggregate."""
    r = (
      Chain(['a', 'b', 'c', 'd'])
      .foreach(lambda idx, el: (idx, el), with_index=True)
      .filter(lambda pair: pair[0] % 2 == 0)
      .then(lambda pairs: [p[1] for p in pairs])
      .run()
    )
    await self.assertEqual(r, ['a', 'c'])

  # -- foreach + foreach_indexed + except_ --
  async def test_foreach_foreach_indexed_except(self):
    """foreach produces list, extract, foreach_indexed errors, except catches."""
    t = Tracker()
    def fail_fn(idx, el):
      if idx == 1:
        raise TestExc()
      return (idx, el)
    r = (
      Chain([10, 20, 30])
      .foreach_indexed(fail_fn)
      .except_(t.on_except, reraise=False)
      .run()
    ) if hasattr(Chain, 'foreach_indexed') else (
      Chain([10, 20, 30])
      .foreach(fail_fn, with_index=True)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called


# =========================================================================
# Category 5: Chain variant triples (Cascade interactions)
# =========================================================================

class CascadeVariantTriples(MyTestCase):

  # -- Cascade + then + do --
  async def test_cascade_then_do_both_receive_root(self):
    """In Cascade, both then and do receive the root value."""
    received_then = []
    received_do = []
    r = (
      Cascade(42)
      .then(lambda v: received_then.append(v) or v)
      .do(lambda v: received_do.append(v))
      .run()
    )
    await self.assertEqual(r, 42)
    assert received_then == [42]
    assert received_do == [42]

  # -- Cascade + foreach + except_ --
  async def test_cascade_foreach_except(self):
    """Cascade: foreach on root, error caught."""
    t = Tracker()
    def fail_on_2(x):
      if x == 2:
        raise TestExc()
      return x
    r = (
      Cascade([1, 2, 3])
      .foreach(fail_on_2)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- Cascade + with_ + finally_ --
  async def test_cascade_with_finally(self):
    """Cascade: CM on root, finally cleanup."""
    t = Tracker()
    cm = SimpleCM('cascade_val')
    r = (
      Cascade(cm)
      .with_(lambda ctx: ctx.upper())
      .finally_(t.on_finally)
      .run()
    )
    # Cascade returns root value (the CM object)
    assert r is cm
    assert t.finally_called
    assert cm.exited

  # -- Cascade + gather + then --
  async def test_cascade_gather_then(self):
    """Cascade: gather on root, then on root, root returned."""
    r = (
      Cascade(10)
      .gather(lambda v: v + 1, lambda v: v * 2)
      .then(lambda v: v + 100)
      .run()
    )
    # Cascade returns root = 10
    await self.assertEqual(r, 10)

  # -- Cascade + foreach + break_ --
  async def test_cascade_foreach_break(self):
    """Cascade: break in foreach iteration, root returned."""
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = Cascade([1, 2, 3, 4]).foreach(fn).run()
    # Cascade returns root
    await self.assertEqual(r, [1, 2, 3, 4])

  # -- Cascade + filter + finally_ --
  async def test_cascade_filter_finally(self):
    """Cascade: filter on root, finally runs, root returned."""
    t = Tracker()
    r = (
      Cascade([1, 2, 3, 4])
      .filter(lambda x: x > 2)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [1, 2, 3, 4])
    assert t.finally_called

  # -- Cascade + clone + then --
  async def test_cascade_clone_then(self):
    """Clone a Cascade, add then: cascade behavior preserved."""
    c = Cascade(100).do(lambda v: None)
    c2 = c.clone()
    c2.then(lambda v: v * 99)
    r = c2.run()
    await self.assertEqual(r, 100)

  # -- Cascade + nested_chain + except_ --
  async def test_cascade_nested_chain_except(self):
    """Cascade: nested chain error, caught by except."""
    t = Tracker()
    inner = Chain().then(raise_exc)
    r = (
      Cascade(5)
      .then(inner)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called


# =========================================================================
# Category 6: Clone/freeze triples
# =========================================================================

class CloneFreezeTriples(MyTestCase):

  # -- clone + then + except_ --
  async def test_clone_then_except(self):
    """Clone a chain, add then that errors, except catches."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(raise_exc)
    c2.except_(lambda v: None, reraise=False)
    r = c2.run()
    await self.assertIsNone(r)
    # Original unaffected
    await self.assertEqual(c.run(), 2)

  # -- clone + foreach + finally_ --
  async def test_clone_foreach_finally(self):
    """Clone chain with foreach, add finally."""
    t = Tracker()
    c = Chain([1, 2, 3]).foreach(lambda x: x * 2)
    c2 = c.clone()
    c2.finally_(t.on_finally)
    r = c2.run()
    await self.assertEqual(r, [2, 4, 6])
    assert t.finally_called

  # -- clone + with_ + except_ --
  async def test_clone_with_except(self):
    """Clone CM chain, add error handling."""
    t = Tracker()
    cm_factory = lambda: SimpleCM('val')
    c = Chain(cm_factory).with_(lambda ctx: ctx.upper())
    c2 = c.clone()
    # Make another clone that errors
    c3 = c.clone()
    c3.then(raise_exc)
    c3.except_(t.on_except, reraise=False)
    r = c3.run()
    await self.assertIsNone(r)
    assert t.exc_called
    # Original clone still works
    r2 = c2.run()
    await self.assertEqual(r2, 'VAL')

  # -- freeze + then + except_ --
  async def test_freeze_then_except(self):
    """Frozen chain as nested in then, error handling."""
    t = Tracker()
    frozen = Chain().then(lambda v: v * 10).freeze()
    r = Chain(3).then(frozen).except_(t.on_except, reraise=False).run()
    await self.assertEqual(r, 30)
    assert not t.exc_called

  async def test_freeze_then_except_error(self):
    """Frozen chain as nested, raises, except catches."""
    t = Tracker()
    frozen = Chain().then(raise_exc).freeze()
    r = Chain(3).then(frozen).except_(t.on_except, reraise=False).run()
    await self.assertIsNone(r)
    assert t.exc_called

  # -- freeze + foreach + break_ --
  async def test_freeze_foreach_break(self):
    """Frozen chain used as foreach fn, break stops early."""
    frozen = Chain().then(lambda v: v * 10).freeze()
    def fn(x):
      if x >= 3:
        Chain.break_()
      return frozen(x)
    r = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(r, [10, 20])

  # -- freeze + with_ + finally_ --
  async def test_freeze_with_finally(self):
    """Frozen chain wrapping CM usage, finally runs."""
    t = Tracker()
    frozen = Chain().then(lambda v: v.upper()).freeze()
    cm = SimpleCM('freeze_val')
    r = Chain(cm).with_(frozen).finally_(t.on_finally).run()
    await self.assertEqual(r, 'FREEZE_VAL')
    assert t.finally_called
    assert cm.exited


# =========================================================================
# Category 7: Async transition triples
# =========================================================================

class AsyncTransitionTriples(MyTestCase):

  # -- then(sync) + then(async) + except_ --
  async def test_sync_to_async_except(self):
    """Sync then, async then that errors, except catches."""
    t = Tracker()
    r = await (
      Chain(1)
      .then(lambda v: v + 1)
      .then(araise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- then(sync) + then(async) + finally_ --
  async def test_sync_to_async_finally(self):
    """Sync then, async then, finally runs."""
    t = Tracker()
    r = await (
      Chain(1)
      .then(lambda v: v + 1)
      .then(async_double)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 4)
    assert t.finally_called

  # -- foreach(async_fn) + except_ + finally_ --
  async def test_foreach_async_fn_except_finally(self):
    """Async foreach fn, error, except + finally."""
    t = Tracker()
    async def fail_on_2(x):
      if x == 2:
        raise TestExc()
      return x * 10
    r = await (
      Chain([1, 2, 3])
      .foreach(fail_on_2)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_foreach_async_fn_except_finally_no_error(self):
    """Async foreach fn, no error, except not triggered, finally runs."""
    t = Tracker()
    async def transform(x):
      return x * 10
    r = await (
      Chain([1, 2, 3])
      .foreach(transform)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [10, 20, 30])
    assert not t.exc_called
    assert t.finally_called

  # -- with_(async_cm) + then + except_ --
  async def test_async_cm_then_except(self):
    """Async CM, then transforms, error caught."""
    t = Tracker()
    cm = AsyncCM('hello')
    r = await (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .then(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called
    assert cm.exited

  async def test_async_cm_then_except_no_error(self):
    """Async CM, then transforms, no error."""
    t = Tracker()
    cm = AsyncCM('hello')
    r = await (
      Chain(cm)
      .with_(lambda ctx: ctx.upper())
      .then(lambda v: v + '!')
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertEqual(r, 'HELLO!')
    assert not t.exc_called

  # -- gather(sync, async) + then + finally_ --
  async def test_gather_mixed_then_finally(self):
    """Mixed sync/async gather, then transforms, finally runs."""
    t = Tracker()
    r = await (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: aempty(v * 2))
      .then(lambda results: sum(results))
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 16)
    assert t.finally_called

  # -- sleep + then + except_ --
  async def test_sleep_then_except(self):
    """Sleep forces async, then errors, except catches."""
    t = Tracker()
    r = await (
      Chain(1)
      .sleep(0.01)
      .then(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  async def test_sleep_then_except_no_error(self):
    """Sleep forces async, then succeeds, except not triggered."""
    t = Tracker()
    r = await (
      Chain(5)
      .sleep(0.01)
      .then(lambda v: v * 2)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertEqual(r, 10)
    assert not t.exc_called

  # -- to_thread + then + finally_ --
  async def test_to_thread_then_finally(self):
    """to_thread, then transforms, finally runs."""
    t = Tracker()
    r = await (
      Chain(5)
      .to_thread(sync_double)
      .then(lambda v: v + 1)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 11)
    assert t.finally_called

  # -- then(async) + foreach + break_ --
  async def test_async_then_foreach_break(self):
    """Async then produces list, foreach with break."""
    async def make_list(v):
      return [1, 2, 3, 4, 5]
    def fn(x):
      if x >= 4:
        Chain.break_()
      return x * 10
    r = await Chain(None).then(make_list).foreach(fn).run()
    await self.assertEqual(r, [10, 20, 30])

  # -- then(async) + filter + finally_ --
  async def test_async_then_filter_finally(self):
    """Async then produces list, filter, finally runs."""
    t = Tracker()
    async def make_list(v):
      return [1, 2, 3, 4, 5]
    r = await (
      Chain(None)
      .then(make_list)
      .filter(lambda x: x > 3)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [4, 5])
    assert t.finally_called


# =========================================================================
# Category 8: Debug mode triples
# =========================================================================

class DebugModeTriples(MyTestCase):

  def _capture_logs(self):
    """Set up log capture, return (logs_list, handler, logger)."""
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logs, handler, logger

  # -- debug + then + except_ --
  async def test_debug_then_except_error(self):
    """Debug mode with error: logs appear before except."""
    logs, handler, logger = self._capture_logs()
    try:
      t = Tracker()
      r = Chain(5).then(raise_exc).except_(t.on_except, reraise=False).config(debug=True).run()
      await self.assertIsNone(r)
      assert t.exc_called
      assert any('5' in log for log in logs)
    finally:
      logger.removeHandler(handler)

  async def test_debug_then_except_no_error(self):
    """Debug mode with no error: logs appear, except not triggered."""
    logs, handler, logger = self._capture_logs()
    try:
      t = Tracker()
      r = Chain(5).then(lambda v: v * 2).except_(t.on_except, reraise=False).config(debug=True).run()
      await self.assertEqual(r, 10)
      assert not t.exc_called
      assert any('5' in log for log in logs)
      assert any('10' in log for log in logs)
    finally:
      logger.removeHandler(handler)

  # -- debug + foreach + finally_ --
  async def test_debug_foreach_finally(self):
    """Debug mode with foreach and finally: logs appear, finally runs."""
    logs, handler, logger = self._capture_logs()
    try:
      t = Tracker()
      r = Chain([1, 2]).foreach(lambda x: x * 10).finally_(t.on_finally).config(debug=True).run()
      await self.assertEqual(r, [10, 20])
      assert t.finally_called
      assert len(logs) > 0
    finally:
      logger.removeHandler(handler)

  # -- debug + with_ + except_ --
  async def test_debug_with_except(self):
    """Debug mode with CM and error."""
    logs, handler, logger = self._capture_logs()
    try:
      t = Tracker()
      cm = SimpleCM('debug_val')
      r = (
        Chain(cm)
        .with_(lambda ctx: ctx)
        .then(raise_exc)
        .except_(t.on_except, reraise=False)
        .config(debug=True)
        .run()
      )
      await self.assertIsNone(r)
      assert t.exc_called
      assert len(logs) > 0
    finally:
      logger.removeHandler(handler)

  # -- debug + gather + finally_ --
  async def test_debug_gather_finally(self):
    """Debug mode with gather and finally."""
    logs, handler, logger = self._capture_logs()
    try:
      t = Tracker()
      r = Chain(5).gather(lambda v: v + 1, lambda v: v * 2).finally_(t.on_finally).config(debug=True).run()
      await self.assertEqual(r, [6, 10])
      assert t.finally_called
      assert len(logs) > 0
    finally:
      logger.removeHandler(handler)

  # -- debug + nested_chain + except_ --
  async def test_debug_nested_chain_except(self):
    """Debug mode with nested chain error."""
    logs, handler, logger = self._capture_logs()
    try:
      t = Tracker()
      inner = Chain().then(raise_exc)
      r = Chain(5).then(inner).except_(t.on_except, reraise=False).config(debug=True).run()
      await self.assertIsNone(r)
      assert t.exc_called
      assert any('5' in log for log in logs)
    finally:
      logger.removeHandler(handler)


# =========================================================================
# Category 9: Deep nesting triples
# =========================================================================

class DeepNestingTriples(MyTestCase):

  # -- nested(nested(chain)) + except_ + finally_ --
  async def test_double_nested_except_finally(self):
    """Double-nested chain error bubbles up to outer except, finally runs."""
    t = Tracker()
    inner2 = Chain().then(raise_exc)
    inner1 = Chain().then(inner2)
    r = Chain(1).then(inner1).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_double_nested_no_error(self):
    """Double-nested chain, no error, values flow correctly."""
    t = Tracker()
    inner2 = Chain().then(lambda v: v * 3)
    inner1 = Chain().then(inner2)
    r = Chain(2).then(inner1).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 6)
    assert not t.exc_called
    assert t.finally_called

  # -- nested + with_ + nested --
  async def test_nested_with_nested(self):
    """Chain inside CM inside chain."""
    inner = Chain().then(lambda v: v.upper())
    cm = SimpleCM('hello')
    r = Chain(cm).with_(inner).run()
    await self.assertEqual(r, 'HELLO')
    assert cm.exited

  async def test_nested_with_nested_error(self):
    """Nested chain inside CM errors, except catches."""
    t = Tracker()
    inner = Chain().then(raise_exc)
    cm = SimpleCM('val')
    with self.assertRaises(TestExc):
      Chain(cm).with_(inner).except_(t.on_except).run()
    assert t.exc_called
    assert cm.exited

  # -- foreach(nested_chain) + except_ + finally_ --
  async def test_foreach_nested_chain_except_finally(self):
    """foreach with fn that raises, except + finally."""
    t = Tracker()
    def transform(v):
      if v == 2:
        raise TestExc()
      return v * 10
    r = Chain([1, 2, 3]).foreach(transform).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_foreach_frozen_chain_except_finally_no_error(self):
    """foreach with frozen chain fn, no error."""
    t = Tracker()
    inner = Chain().then(lambda v: v * 10)
    r = Chain([1, 2, 3]).foreach(inner.freeze()).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, [10, 20, 30])
    assert not t.exc_called
    assert t.finally_called

  # -- gather(nested1, nested2) + except_ + finally_ --
  async def test_gather_nested_chains_except_finally(self):
    """Parallel nested chains, one errors, except + finally."""
    t = Tracker()
    frozen1 = Chain().then(lambda v: v * 2).freeze()
    frozen2 = Chain().then(raise_exc).freeze()
    r = Chain(5).gather(frozen1, frozen2).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  async def test_gather_nested_chains_no_error(self):
    """Parallel nested chains, no error."""
    t = Tracker()
    frozen1 = Chain().then(lambda v: v * 2).freeze()
    frozen2 = Chain().then(lambda v: v + 10).freeze()
    r = Chain(5).gather(frozen1, frozen2).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, [10, 15])
    assert not t.exc_called
    assert t.finally_called


# =========================================================================
# Category 10: Value flow triples
# =========================================================================

class ValueFlowTriples(MyTestCase):

  # -- then(a) + then(b) + then(c) --
  async def test_then_then_then_flow(self):
    """Verify a->b->c value flow."""
    r = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).then(lambda v: v - 2).run()
    await self.assertEqual(r, 4)

  async def test_then_then_then_flow_async(self):
    """a->b->c flow with async."""
    r = await Chain(1).then(async_identity).then(async_double).then(lambda v: v + 5).run()
    await self.assertEqual(r, 7)

  # -- then(a) + do(b) + then(c) --
  async def test_then_do_then_value_bypass(self):
    """Verify a flows past do to c (do discards result)."""
    side = []
    r = (
      Chain(10)
      .then(lambda v: v * 2)
      .do(lambda v: side.append(v * 100))
      .then(lambda v: v + 1)
      .run()
    )
    await self.assertEqual(r, 21)
    assert side == [2000]

  async def test_then_do_then_async(self):
    """Same with async do fn."""
    side = []
    async def async_side(v):
      side.append(v)
    r = await (
      Chain(5)
      .then(lambda v: v * 3)
      .do(async_side)
      .then(lambda v: v + 10)
      .run()
    )
    await self.assertEqual(r, 25)
    assert side == [15]

  # -- then(make_list) + foreach(transform) + then(use_list) --
  async def test_then_foreach_then_list_flow(self):
    """Verify list transformation flows."""
    r = (
      Chain(3)
      .then(lambda v: list(range(v)))
      .foreach(lambda x: x * 10)
      .then(lambda lst: sum(lst))
      .run()
    )
    await self.assertEqual(r, 30)

  # -- then(get_cm) + with_(body) + then(use_result) --
  async def test_then_with_then_cm_flow(self):
    """Verify CM result flows to next then."""
    r = (
      Chain(lambda: SimpleCM('payload'))
      .with_(lambda ctx: ctx.upper())
      .then(lambda v: v + '_done')
      .run()
    )
    await self.assertEqual(r, 'PAYLOAD_done')

  # -- then(value) + gather(f1,f2) + then(use_results) --
  async def test_then_gather_then_results_flow(self):
    """Verify gather results flow to next then."""
    r = (
      Chain(10)
      .then(lambda v: v + 5)
      .gather(lambda v: v * 2, lambda v: v - 1, lambda v: v ** 2)
      .then(lambda results: {'doubled': results[0], 'minus1': results[1], 'squared': results[2]})
      .run()
    )
    await self.assertEqual(r, {'doubled': 30, 'minus1': 14, 'squared': 225})


# =========================================================================
# Additional combinatorial edge cases
# =========================================================================

class AdditionalEdgeCases(MyTestCase):

  # -- except_ + finally_ ordering verification --
  async def test_except_finally_ordering(self):
    """except_ is called before finally_ on error."""
    order = []
    def on_exc(v=None):
      order.append('except')
    def on_fin(v=None):
      order.append('finally')
    with self.assertRaises(TestExc):
      Chain(1).then(raise_exc).except_(on_exc).finally_(on_fin).run()
    assert order == ['except', 'finally']

  # -- Cascade + do + except_ --
  async def test_cascade_do_except_error(self):
    """Cascade: do errors, except catches."""
    t = Tracker()
    r = Cascade(42).do(raise_exc).except_(t.on_except, reraise=False).run()
    await self.assertIsNone(r)
    assert t.exc_called

  # -- clone + freeze + then --
  async def test_clone_freeze_then(self):
    """Clone a chain, freeze it, use frozen in then."""
    c = Chain().then(lambda v: v * 5)
    c2 = c.clone()
    frozen = c2.freeze()
    r = Chain(3).then(frozen).run()
    await self.assertEqual(r, 15)

  # -- Multiple except_ handlers + finally_ --
  async def test_multiple_except_finally(self):
    """Multiple except_ with different exception types + finally."""
    t = Tracker()
    class CustomExc(Exception):
      pass
    caught = {'which': None}
    def handler_custom(v=None):
      caught['which'] = 'custom'
    def handler_test(v=None):
      caught['which'] = 'test'
    r = (
      Chain(1)
      .then(raise_exc)
      .except_(handler_custom, exceptions=CustomExc, reraise=False)
      .except_(handler_test, exceptions=TestExc, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertIsNone(r)
    assert caught['which'] == 'test'
    assert t.finally_called

  # -- foreach + async fn + break_ --
  async def test_foreach_async_fn_break(self):
    """Async foreach fn with break."""
    async def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = await Chain([1, 2, 3, 4, 5]).foreach(fn).run()
    await self.assertEqual(r, [10, 20])

  # -- with_ + async body + except_ --
  async def test_with_async_body_except(self):
    """Sync CM with async body that errors, except catches."""
    t = Tracker()
    cm = SimpleCM('val')
    async def async_body(ctx):
      raise TestExc('async body error')
    with self.assertRaises(TestExc):
      await Chain(cm).with_(async_body).except_(t.on_except, reraise=True).run()
    assert t.exc_called
    assert cm.exited

  # -- then + do + finally_ (value preservation) --
  async def test_then_do_finally_value_preserved(self):
    """then produces value, do discards side-effect, finally runs, value correct."""
    t = Tracker()
    r = Chain(7).then(lambda v: v * 3).do(lambda v: None).finally_(t.on_finally).run()
    await self.assertEqual(r, 21)
    assert t.finally_called

  # -- gather(async, async) + except_ + finally_ --
  async def test_gather_all_async_except_finally(self):
    """All async gather fns, one errors, except + finally."""
    t = Tracker()
    async def ok_fn(v):
      return v * 2
    r = await (
      Chain(5)
      .gather(ok_fn, araise_exc)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  # -- foreach_indexed + async fn + finally_ --
  async def test_foreach_indexed_async_finally(self):
    """Async indexed foreach, finally runs."""
    t = Tracker()
    async def fn(idx, el):
      return (idx, el * 10)
    r = await (
      Chain([1, 2, 3])
      .foreach(fn, with_index=True)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [(0, 10), (1, 20), (2, 30)])
    assert t.finally_called

  # -- Cascade + then + finally_ --
  async def test_cascade_then_finally(self):
    """Cascade: then gets root, finally runs, root returned."""
    t = Tracker()
    received = []
    r = (
      Cascade(99)
      .then(lambda v: received.append(v) or v)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 99)
    assert received == [99]
    assert t.finally_called

  # -- foreach + break_(value) + then --
  async def test_foreach_break_value_then(self):
    """Break with value in foreach, then transforms that value."""
    sentinel = 'break_result'
    def fn(x):
      if x == 3:
        Chain.break_(sentinel)
      return x
    r = Chain([1, 2, 3, 4]).foreach(fn).then(lambda v: f'got_{v}').run()
    await self.assertEqual(r, 'got_break_result')

  # -- filter + then + except_ --
  async def test_filter_then_except(self):
    """filter, then transforms, error in then caught."""
    t = Tracker()
    r = (
      Chain([1, 2, 3, 4])
      .filter(lambda x: x > 2)
      .then(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- with_ + do + except_ --
  async def test_with_do_except(self):
    """CM + do side-effect that errors + except catches."""
    t = Tracker()
    cm = SimpleCM('val')
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .do(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- clone + except_ + finally_ together --
  async def test_clone_except_finally(self):
    """Clone chain that has except_ + finally_."""
    t = Tracker()
    c = (
      Chain(1)
      .then(raise_exc)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
    )
    c2 = c.clone()
    r = c2.run()
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called

  # -- do + foreach + then --
  async def test_do_foreach_then(self):
    """do doesn't change value, foreach iterates it, then transforms."""
    side = []
    r = (
      Chain([1, 2, 3])
      .do(lambda v: side.append(len(v)))
      .foreach(lambda x: x * 10)
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 60)
    assert side == [3]

  # -- nested_chain + foreach + finally_ --
  async def test_nested_chain_foreach_finally(self):
    """Nested chain returns list, foreach iterates, finally runs."""
    t = Tracker()
    inner = Chain().then(lambda v: [v, v * 2, v * 3])
    r = (
      Chain(5)
      .then(inner)
      .foreach(lambda x: x + 100)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [105, 110, 115])
    assert t.finally_called

  # -- Cascade + sleep + finally_ --
  async def test_cascade_sleep_finally(self):
    """Cascade with sleep (async), finally runs."""
    t = Tracker()
    r = await (
      Cascade(42)
      .sleep(0.01)
      .then(lambda v: v * 99)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 42)
    assert t.finally_called

  # -- gather + then + except_ --
  async def test_gather_then_except(self):
    """gather ok, then errors, except catches."""
    t = Tracker()
    r = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v * 2)
      .then(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- with_ + then + finally_ --
  async def test_with_then_finally(self):
    """CM + then + finally: all interact correctly."""
    t = Tracker()
    cm = SimpleCM('world')
    r = (
      Chain(cm)
      .with_(lambda ctx: f'hello_{ctx}')
      .then(lambda v: v.upper())
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 'HELLO_WORLD')
    assert t.finally_called
    assert cm.exited

  # -- foreach + do + then --
  async def test_foreach_do_then(self):
    """foreach produces list, do has side effect, then transforms."""
    side = []
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .do(lambda v: side.extend(v))
      .then(lambda v: len(v))
      .run()
    )
    await self.assertEqual(r, 3)
    assert side == [2, 4, 6]

  # -- filter + foreach + except_ --
  async def test_filter_foreach_except(self):
    """filter, then foreach errors, except catches."""
    t = Tracker()
    def fail_fn(x):
      if x == 5:
        raise TestExc()
      return x * 10
    r = (
      Chain([1, 2, 3, 4, 5])
      .filter(lambda x: x > 2)
      .foreach(fail_fn)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- freeze + except_ + finally_ --
  async def test_freeze_except_finally(self):
    """Frozen chain in pipeline with except and finally."""
    t = Tracker()
    frozen = Chain().then(lambda v: v * 10).freeze()
    r = Chain(3).then(frozen).except_(t.on_except, reraise=False).finally_(t.on_finally).run()
    await self.assertEqual(r, 30)
    assert not t.exc_called
    assert t.finally_called

  # -- do + except_ + finally_ --
  async def test_do_except_finally_no_error_value_flow(self):
    """do + except + finally: verify value flows past do correctly."""
    t = Tracker()
    r = (
      Chain(10)
      .then(lambda v: v + 5)
      .do(lambda v: None)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 15)
    assert not t.exc_called
    assert t.finally_called

  # -- with_ + foreach + finally_ --
  async def test_with_foreach_finally(self):
    """CM produces list, foreach iterates, finally runs."""
    t = Tracker()
    cm = SimpleCM([10, 20, 30])
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .foreach(lambda x: x + 1)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [11, 21, 31])
    assert t.finally_called
    assert cm.exited

  # -- then(sync) + then(async) + then(sync) --
  async def test_sync_async_sync_transition(self):
    """Value flows through sync->async->sync transitions."""
    r = await (
      Chain(1)
      .then(lambda v: v + 1)
      .then(async_double)
      .then(lambda v: v + 10)
      .run()
    )
    await self.assertEqual(r, 14)

  # -- foreach(async) + break_ + finally_ --
  async def test_foreach_async_break_finally(self):
    """Async foreach with break, finally runs."""
    t = Tracker()
    async def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = await (
      Chain([1, 2, 3, 4])
      .foreach(fn)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [10, 20])
    assert t.finally_called

  # -- nested chain + then + finally_ --
  async def test_nested_then_finally(self):
    """Nested chain in then, then transforms result, finally runs."""
    t = Tracker()
    inner = Chain().then(lambda v: v * 2)
    r = (
      Chain(5)
      .then(inner)
      .then(lambda v: v + 100)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 110)
    assert t.finally_called

  # -- Cascade + do + finally_ --
  async def test_cascade_do_finally(self):
    """Cascade: do side-effects, finally runs, root returned."""
    t = Tracker()
    side = []
    r = (
      Cascade(50)
      .do(lambda v: side.append(v))
      .do(lambda v: side.append(v * 2))
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 50)
    assert side == [50, 100]
    assert t.finally_called

  # -- clone + then + finally_ --
  async def test_clone_then_finally(self):
    """Clone chain, add then and finally to clone."""
    t = Tracker()
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(lambda v: v * 10)
    c2.finally_(t.on_finally)
    r = c2.run()
    await self.assertEqual(r, 20)
    assert t.finally_called
    await self.assertEqual(c.run(), 2)

  # -- foreach + filter + except_ --
  async def test_foreach_filter_except(self):
    """foreach transforms, filter errors, except catches."""
    t = Tracker()
    def bad_pred(x):
      if x > 20:
        raise TestExc()
      return x > 5
    r = (
      Chain([1, 2, 3, 4, 5])
      .foreach(lambda x: x * 10)
      .filter(bad_pred)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- gather + filter + then --
  async def test_gather_filter_then(self):
    """gather results, filter them, then aggregate."""
    r = (
      Chain(5)
      .gather(lambda v: v + 1, lambda v: v * 2, lambda v: v - 3)
      .filter(lambda x: x > 5)
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 16)

  # -- with_ + gather + finally_ --
  async def test_with_gather_finally(self):
    """CM produces value, gather multiple fns, finally runs."""
    t = Tracker()
    cm = SimpleCM(100)
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .gather(lambda v: v + 1, lambda v: v * 2)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [101, 200])
    assert t.finally_called

  # -- then + sleep + finally_ --
  async def test_then_sleep_finally(self):
    """then + sleep (forces async) + finally."""
    t = Tracker()
    r = await (
      Chain(5)
      .then(lambda v: v * 2)
      .sleep(0.01)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 10)
    assert t.finally_called

  # -- Cascade + foreach + finally_ --
  async def test_cascade_foreach_finally(self):
    """Cascade: foreach on root, finally runs, root returned."""
    t = Tracker()
    r = (
      Cascade([1, 2, 3])
      .foreach(lambda x: x * 10)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [1, 2, 3])
    assert t.finally_called

  # -- then + then + except_ (error in middle) --
  async def test_then_then_except_error_in_middle(self):
    """Error in second then, third then not reached, except catches."""
    t = Tracker()
    reached = []
    r = (
      Chain(1)
      .then(lambda v: v + 1)
      .then(raise_exc)
      .then(lambda v: reached.append('should not reach'))
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called
    assert reached == []

  # -- except_ with return value + finally_ --
  async def test_except_return_value_finally(self):
    """except_(reraise=False) returns handler's value, finally still runs."""
    t = Tracker()
    sentinel = object()
    r = (
      Chain(1)
      .then(raise_exc)
      .except_(lambda v: sentinel, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    assert r is sentinel
    assert t.finally_called

  # -- foreach + do + except_ --
  async def test_foreach_do_except(self):
    """foreach ok, do errors, except catches."""
    t = Tracker()
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .do(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- to_thread + then + except_ --
  async def test_to_thread_then_except(self):
    """to_thread transforms, then errors, except catches."""
    t = Tracker()
    r = await (
      Chain(5)
      .to_thread(sync_double)
      .then(raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called

  # -- foreach_indexed + do + then --
  async def test_foreach_indexed_do_then(self):
    """Indexed foreach, do side-effect, then aggregate."""
    side = []
    r = (
      Chain(['x', 'y', 'z'])
      .foreach(lambda idx, el: f'{idx}:{el}', with_index=True)
      .do(lambda v: side.extend(v))
      .then(lambda v: ','.join(v))
      .run()
    )
    await self.assertEqual(r, '0:x,1:y,2:z')
    assert side == ['0:x', '1:y', '2:z']

  # -- gather + do + then --
  async def test_gather_do_then(self):
    """gather results, do side effect, then aggregate."""
    side = []
    r = (
      Chain(10)
      .gather(lambda v: v + 1, lambda v: v * 2)
      .do(lambda v: side.extend(v))
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 31)
    assert side == [11, 20]

  # -- with_ + then + finally_ (async CM) --
  async def test_async_cm_then_finally(self):
    """Async CM + then + finally."""
    t = Tracker()
    cm = AsyncCM('async_val')
    r = await (
      Chain(cm)
      .with_(lambda ctx: ctx.upper())
      .then(lambda v: v + '!')
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 'ASYNC_VAL!')
    assert t.finally_called
    assert cm.exited

  # -- freeze + do + then --
  async def test_freeze_do_then(self):
    """Frozen chain in do (side-effect), then transforms."""
    side = []
    frozen = Chain().then(lambda v: side.append(v)).freeze()
    r = Chain(5).do(frozen).then(lambda v: v * 10).run()
    await self.assertEqual(r, 50)
    assert side == [5]

  # -- clone + do + except_ --
  async def test_clone_do_except(self):
    """Clone chain with do that errors, except catches."""
    t = Tracker()
    c = Chain(1).do(raise_exc).except_(t.on_except, reraise=False)
    c2 = c.clone()
    r = c2.run()
    await self.assertIsNone(r)
    assert t.exc_called

  # -- Cascade + clone + except_ --
  async def test_cascade_clone_except(self):
    """Clone Cascade, add except."""
    t = Tracker()
    c = Cascade(5).then(lambda v: v * 2)
    c2 = c.clone()
    c2.then(raise_exc)
    c2.except_(t.on_except, reraise=False)
    r = c2.run()
    await self.assertIsNone(r)
    assert t.exc_called


# =========================================================================
# Async context manager combined triples
# =========================================================================

class AsyncCMCombinedTriples(MyTestCase):

  async def test_async_cm_foreach_except(self):
    """Async CM body returns list, foreach iterates, error caught."""
    t = Tracker()
    cm = AsyncCM([1, 2, 3])
    def fail_on_2(x):
      if x == 2:
        raise TestExc()
      return x * 10
    r = await (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .foreach(fail_on_2)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called
    assert cm.exited

  async def test_async_cm_filter_finally(self):
    """Async CM + filter + finally."""
    t = Tracker()
    cm = AsyncCM([1, 2, 3, 4, 5])
    r = await (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .filter(lambda x: x > 3)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [4, 5])
    assert t.finally_called
    assert cm.exited

  async def test_async_cm_gather_finally(self):
    """Async CM + gather + finally."""
    t = Tracker()
    cm = AsyncCM(10)
    r = await (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .gather(lambda v: v + 1, lambda v: v * 2)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [11, 20])
    assert t.finally_called
    assert cm.exited


# =========================================================================
# Return_ propagation triples
# =========================================================================

class ReturnPropagationTriples(MyTestCase):

  async def test_return_through_nested_chain_finally(self):
    """return_ in nested chain propagates to outer, finally runs."""
    t = Tracker()
    sentinel = object()
    inner = Chain().then(lambda v: Chain.return_(sentinel))
    r = Chain(1).then(inner).then(lambda v: 'not reached').finally_(t.on_finally).run()
    assert r is sentinel
    assert t.finally_called

  async def test_return_in_foreach_nested_chain(self):
    """return_ inside foreach exits the ENTIRE chain."""
    sentinel = object()
    def fn(x):
      if x == 2:
        Chain.return_(sentinel)
      return x
    r = Chain([1, 2, 3]).foreach(fn).then(lambda v: 'not reached').run()
    assert r is sentinel

  async def test_break_in_foreach_does_not_exit_chain(self):
    """break_ only exits foreach, chain continues."""
    def fn(x):
      if x >= 3:
        Chain.break_()
      return x * 10
    r = Chain([1, 2, 3, 4]).foreach(fn).then(lambda v: sum(v)).run()
    await self.assertEqual(r, 30)


# =========================================================================
# Async foreach_indexed combined triples
# =========================================================================

class AsyncForeachIndexedTriples(MyTestCase):

  async def test_foreach_indexed_async_break_finally(self):
    """Async indexed foreach with break, finally runs."""
    t = Tracker()
    async def fn(idx, el):
      if idx >= 2:
        Chain.break_()
      return (idx, el)
    r = await (
      Chain(['a', 'b', 'c', 'd'])
      .foreach(fn, with_index=True)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [(0, 'a'), (1, 'b')])
    assert t.finally_called

  async def test_foreach_indexed_async_except_finally(self):
    """Async indexed foreach error, except + finally."""
    t = Tracker()
    async def fn(idx, el):
      if idx == 1:
        raise TestExc()
      return (idx, el)
    r = await (
      Chain(['a', 'b', 'c'])
      .foreach(fn, with_index=True)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called
    assert t.finally_called


# =========================================================================
# Pipe syntax triples
# =========================================================================

class PipeSyntaxTriples(MyTestCase):

  async def test_pipe_then_except_finally(self):
    """Pipe syntax with except and finally."""
    t = Tracker()
    from quent import run as Run
    r = (
      Chain(5)
      .then(lambda v: v * 2)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      | Run()
    )
    await self.assertEqual(r, 10)
    assert not t.exc_called
    assert t.finally_called


# =========================================================================
# no_async triples
# =========================================================================

class NoAsyncTriples(MyTestCase):

  async def test_no_async_then_except_finally(self):
    """no_async chain with except and finally (pure sync)."""
    t = Tracker()
    r = (
      Chain(5)
      .no_async(True)
      .then(lambda v: v * 2)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 10)
    assert not t.exc_called
    assert t.finally_called

  async def test_no_async_foreach_then_finally(self):
    """no_async chain with foreach, then, finally."""
    t = Tracker()
    r = (
      Chain([1, 2, 3])
      .no_async(True)
      .foreach(lambda x: x * 10)
      .then(sum)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 60)
    assert t.finally_called


# =========================================================================
# Extra tests for full coverage
# =========================================================================

class ExtraTriples(MyTestCase):

  async def test_chain_with_body_returning_nested(self):
    """with_ body returns a value from nested chain."""
    inner = Chain().then(lambda v: v * 100)
    cm = SimpleCM(5)
    r = Chain(cm).with_(inner).run()
    await self.assertEqual(r, 500)
    assert cm.exited

  async def test_gather_then_filter_then(self):
    """gather -> then(flatten) -> filter -> then(sum)."""
    r = (
      Chain(3)
      .gather(lambda v: v, lambda v: v * 2, lambda v: v * 3)
      .filter(lambda x: x > 3)
      .then(sum)
      .run()
    )
    # gather: [3, 6, 9], filter: [6, 9], sum: 15
    await self.assertEqual(r, 15)

  async def test_cascade_multiple_do_finally(self):
    """Cascade with multiple do ops and finally."""
    t = Tracker()
    calls = []
    r = (
      Cascade(10)
      .do(lambda v: calls.append(f'a:{v}'))
      .do(lambda v: calls.append(f'b:{v}'))
      .do(lambda v: calls.append(f'c:{v}'))
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 10)
    assert calls == ['a:10', 'b:10', 'c:10']
    assert t.finally_called

  async def test_freeze_reuse_multiple_times(self):
    """Frozen chain can be reused multiple times with different values."""
    frozen = Chain().then(lambda v: v * 10).freeze()
    r1 = frozen(1)
    r2 = frozen(2)
    r3 = frozen(3)
    await self.assertEqual(r1, 10)
    await self.assertEqual(r2, 20)
    await self.assertEqual(r3, 30)

  async def test_debug_async_then_except(self):
    """Debug mode with async then that errors."""
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      t = Tracker()
      r = await (
        Chain(5)
        .then(async_double)
        .then(araise_exc)
        .except_(t.on_except, reraise=False)
        .config(debug=True)
        .run()
      )
      await self.assertIsNone(r)
      assert t.exc_called
      assert len(logs) > 0
    finally:
      logger.removeHandler(handler)

  async def test_foreach_filter_finally(self):
    """foreach + filter + finally: all three interact."""
    t = Tracker()
    r = (
      Chain([1, 2, 3, 4, 5, 6])
      .foreach(lambda x: x * 2)
      .filter(lambda x: x > 6)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [8, 10, 12])
    assert t.finally_called

  async def test_clone_cascade_foreach_finally(self):
    """Clone Cascade with foreach and finally."""
    t = Tracker()
    c = Cascade([1, 2, 3]).foreach(lambda x: x * 10)
    c2 = c.clone()
    c2.finally_(t.on_finally)
    r = c2.run()
    # Cascade returns root
    await self.assertEqual(r, [1, 2, 3])
    assert t.finally_called

  async def test_with_gather_except_finally(self):
    """CM + gather + except + finally: 4-way but testing core triple."""
    t = Tracker()
    cm = SimpleCM(10)
    r = (
      Chain(cm)
      .with_(lambda ctx: ctx)
      .gather(lambda v: v + 1, lambda v: v * 2)
      .except_(t.on_except, reraise=False)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [11, 20])
    assert not t.exc_called
    assert t.finally_called

  async def test_async_foreach_filter_then(self):
    """Async foreach + filter + then."""
    async def transform(x):
      return x * 2
    r = await (
      Chain([1, 2, 3, 4, 5])
      .foreach(transform)
      .filter(lambda x: x > 4)
      .then(sum)
      .run()
    )
    await self.assertEqual(r, 6 + 8 + 10)

  async def test_nested_chain_gather_finally(self):
    """Nested chain + gather + finally."""
    t = Tracker()
    inner = Chain().then(lambda v: v * 2)
    r = (
      Chain(5)
      .then(inner)
      .gather(lambda v: v + 1, lambda v: v - 1)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, [11, 9])
    assert t.finally_called

  async def test_do_then_finally(self):
    """do side-effect + then transform + finally cleanup."""
    t = Tracker()
    side = []
    r = (
      Chain(5)
      .do(lambda v: side.append(v))
      .then(lambda v: v * 3)
      .finally_(t.on_finally)
      .run()
    )
    await self.assertEqual(r, 15)
    assert side == [5]
    assert t.finally_called

  async def test_foreach_gather_except(self):
    """foreach + gather + except: error in gather fn."""
    t = Tracker()
    r = (
      Chain([1, 2, 3])
      .foreach(lambda x: x * 2)
      .gather(sum, raise_exc)
      .except_(t.on_except, reraise=False)
      .run()
    )
    await self.assertIsNone(r)
    assert t.exc_called


if __name__ == '__main__':
  import unittest
  unittest.main()
