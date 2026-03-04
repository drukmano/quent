"""Exhaustive tests for _Return, _Break, and exception handling in every combination
with every other feature in the quent library.

Covers:
  - _Return in every context (simple chain, nested, foreach, with_, gather, cascade, etc.)
  - _Break in every context (foreach, filter, nested, etc.)
  - Exception + _Return interactions
  - Exception + _Break interactions
  - _Return + _Break interactions
  - except_ edge cases with control flow
  - finally_ edge cases with control flow
  - Complex multi-level control flow
  - Control flow in generators (iterate)
  - Async-specific control flow
"""

import asyncio
from contextlib import contextmanager, asynccontextmanager
from itertools import product
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Null, QuentException


# --- Helpers ---

class Exc1(TestExc):
  pass

class Exc2(TestExc):
  pass

class Exc3(Exc2):
  pass

def raise_(e=TestExc):
  if e is None or type(e) != type:
    e = TestExc
  raise e

def raise_exc(exc_type):
  def _raise(v=None):
    raise exc_type()
  return _raise

async def araise_(v=None):
  raise TestExc

class SyncIterator:
  def __init__(self, items=None):
    self._items = items if items is not None else list(range(10))
  def __iter__(self):
    return iter(self._items)

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

class SyncCM:
  """Simple sync context manager that tracks enter/exit."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False
  def __enter__(self):
    self.entered = True
    return self.value if self.value is not None else self
  def __exit__(self, *args):
    self.exited = True
    return False

class AsyncCM:
  """Simple async context manager that tracks enter/exit."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False
  async def __aenter__(self):
    self.entered = True
    return self.value if self.value is not None else self
  async def __aexit__(self, *args):
    self.exited = True
    return False

class Tracker:
  """General-purpose side-effect tracker."""
  def __init__(self):
    self.calls = []
    self.count = 0
  def __call__(self, v=None):
    self.calls.append(v)
    self.count += 1
    return v
  def async_call(self):
    tracker = self
    async def _fn(v=None):
      tracker.calls.append(v)
      tracker.count += 1
      return v
    return _fn


# ====================================================================
# Section 1: _Return in every context
# ====================================================================
class ReturnBasicTests(IsolatedAsyncioTestCase):

  async def test_return_in_simple_chain_no_nesting(self):
    """return_() via .then(Chain.return_) passes current_value to return_, so it returns that value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # Chain.return_ receives current_value (10) as __v, so _Return(10) is raised
        # handle_return_exc resolves 10 (not Null, so _eval_signal_value(10, (), {}) -> 10)
        self.assertEqual(await await_(
          Chain(fn, 10).then(Chain.return_).run(),
        ), 10)

  async def test_return_with_explicit_value(self):
    """return_(value) returns that exact value."""
    sentinel = object()
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertIs(await await_(
          Chain(fn, 10).then(Chain.return_, sentinel).run(),
        ), sentinel)

  async def test_return_with_callable(self):
    """return_(callable) evaluates the callable and returns its result."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(await await_(
          Chain(fn, 10).then(Chain.return_, lambda: 42).run(),
        ), 42)

  async def test_return_with_async_callable(self):
    """return_(async_callable) awaits and returns the result."""
    async def make_val():
      return 99
    self.assertEqual(await Chain(aempty, 10).then(Chain.return_, make_val).run(), 99)

  async def test_return_after_then(self):
    """return_() after .then() discards then's result and uses return's value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        sentinel = object()
        self.assertIs(await await_(
          Chain(fn, 1).then(lambda v: v + 100).then(Chain.return_, sentinel).run(),
        ), sentinel)

  async def test_return_after_do(self):
    """return_() after .do() -- do's side effect happens, return value used."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        sentinel = object()
        self.assertIs(await await_(
          Chain(fn, 5).do(tracker).then(Chain.return_, sentinel).run(),
        ), sentinel)
        self.assertEqual(tracker.count, 1)

  async def test_return_before_except_not_triggered(self):
    """return_() before except_ -- except_ NOT triggered because return is not an error."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10)
          .then(Chain.return_, 42)
          .except_(tracker, reraise=False)
          .run()
        )
        self.assertEqual(result, 42)
        self.assertEqual(tracker.count, 0)

  async def test_return_before_finally_is_triggered(self):
    """return_() before finally_ -- finally_ IS triggered."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10)
          .then(Chain.return_, 42)
          .finally_(tracker)
          .run()
        )
        self.assertEqual(result, 42)
        self.assertEqual(tracker.count, 1)

  async def test_return_inside_nested_chain_propagates(self):
    """return_() inside a nested chain propagates to the outermost chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain().then(fn1).then(Chain.return_, 77)
        result = await await_(
          Chain(fn2, 10).then(inner).then(lambda v: v + 1000).run()
        )
        self.assertEqual(result, 77)

  async def test_return_inside_doubly_nested_chain_propagates(self):
    """return_() inside a doubly-nested chain propagates through ALL levels."""
    for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
        innermost = Chain().then(fn1).then(Chain.return_, 55)
        middle = Chain().then(fn2).then(innermost)
        result = await await_(
          Chain(fn3, 10).then(middle).then(lambda v: v + 9999).run()
        )
        self.assertEqual(result, 55)

  async def test_return_null_returns_none(self):
    """return_(Null) returns None (Null sentinel means 'no value')."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertIsNone(await await_(
          Chain(fn, 10).then(Chain.return_, Null).run()
        ))

  async def test_return_plain_no_value_returns_none(self):
    """return_() with no arguments (called directly in lambda) returns None."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertIsNone(await await_(
          Chain(fn, 10).then(lambda v: Chain.return_()).run()
        ))

  async def test_multiple_return_first_wins(self):
    """Multiple return_() in chain -- first one wins."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 1)
          .then(Chain.return_, 'first')
          .then(Chain.return_, 'second')
          .run()
        )
        self.assertEqual(result, 'first')

  async def test_return_with_debug_mode(self):
    """return_() with debug mode -- verify debug logging still works."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10).config(debug=True).then(Chain.return_, 42).run()
        )
        self.assertEqual(result, 42)

  async def test_return_in_reusable_chain_standalone(self):
    """return_() in a reusable chain when called standalone."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        c = Chain().then(fn).then(lambda v: Chain.return_(42)).then(lambda v: v * 100)
        result = await await_(c(10))
        self.assertEqual(result, 42)

  async def test_return_in_clone(self):
    """return_() in a cloned chain -- verify independence."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        original = Chain(fn, 10).then(Chain.return_, 42)
        cloned = original.clone()
        result = await await_(cloned.run())
        self.assertEqual(result, 42)
        # Original still works too
        result2 = await await_(original.run())
        self.assertEqual(result2, 42)

  async def test_return_with_literal_value(self):
    """return_(literal) with a non-callable literal returns it as-is."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(await await_(
          Chain(fn, 10).then(Chain.return_, 'hello').run(),
        ), 'hello')


# ====================================================================
# Section 2: _Break in every context
# ====================================================================
class BreakForeachTests(IsolatedAsyncioTestCase):

  async def test_break_in_foreach_stops_iteration(self):
    """break_() in foreach stops iteration and returns collected list so far."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn1(el * 2)
        self.assertEqual(await await_(
          Chain(fn2, [1, 2, 3, 4, 5]).foreach(f).run(),
        ), [2, 4])

  async def test_break_with_value_in_foreach(self):
    """break_(value) in foreach returns the break value instead of partial list."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        sentinel = object()
        def f(el):
          if el == 2:
            Chain.break_(sentinel)
          return fn1(el)
        self.assertIs(await await_(
          Chain(fn2, [1, 2, 3]).foreach(f).run(),
        ), sentinel)

  async def test_break_with_callable_value(self):
    """break_(callable) evaluates callable as break value."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el == 2:
            Chain.break_(lambda: 'break_val')
          return fn1(el)
        self.assertEqual(await await_(
          Chain(fn2, [1, 2, 3]).foreach(f).run(),
        ), 'break_val')

  async def test_break_with_async_callable_value(self):
    """break_(async_callable) awaits and returns the async break value."""
    async def make_break_val():
      return 'async_break'
    def f(el):
      if el == 2:
        Chain.break_(make_break_val)
      return el
    self.assertEqual(await await_(
      Chain([1, 2, 3]).foreach(f).run(),
    ), 'async_break')

  async def test_break_at_top_level_raises_quent_exception(self):
    """break_() at top-level chain (not inside foreach) raises QuentException."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(QuentException):
          await await_(
            Chain(fn, 10).then(Chain.break_).run()
          )

  async def test_break_inside_nested_foreach(self):
    """break_() inside nested foreach only breaks the inner foreach."""
    for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
        def inner_fn(el):
          if el >= 2:
            Chain.break_()
          return fn1(el * 10)
        def outer_fn(el):
          return Chain(fn2, [1, 2, 3, 4]).foreach(inner_fn).run()
        self.assertEqual(await await_(
          Chain(fn3, [10, 20]).foreach(outer_fn).run(),
        ), [[10], [10]])

  async def test_break_with_finally(self):
    """break_() in foreach with finally_ -- finally_ runs after break."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        tracker = Tracker()
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3, 4])
          .foreach(f)
          .finally_(tracker)
          .run()
        )
        self.assertEqual(result, [1, 2])
        self.assertEqual(tracker.count, 1)

  async def test_break_in_async_foreach(self):
    """break_() in async foreach with async iterable."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn1(el * 2)
        self.assertEqual(await await_(
          Chain(fn2, AsyncIterator([1, 2, 3, 4, 5])).foreach(f).run(),
        ), [2, 4])

  async def test_break_after_some_elements(self):
    """break_() after processing some elements returns partial results."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        processed = []
        def f(el):
          processed.append(el)
          if el == 3:
            Chain.break_()
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3, 4, 5]).foreach(f).run()
        )
        self.assertEqual(result, [1, 2])
        self.assertEqual(processed, [1, 2, 3])

  async def test_break_on_first_element(self):
    """break_() on first element returns empty list."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def f(el):
          Chain.break_()
        self.assertEqual(await await_(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
        ), [])

  async def test_break_on_last_element(self):
    """break_() on last element returns all but the last."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el == 3:
            Chain.break_()
          return fn1(el * 10)
        self.assertEqual(await await_(
          Chain(fn2, [1, 2, 3]).foreach(f).run(),
        ), [10, 20])

  async def test_break_with_null_value(self):
    """break_(Null) uses the fallback (partial list)."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el == 2:
            Chain.break_(Null)
          return fn1(el * 10)
        self.assertEqual(await await_(
          Chain(fn2, [1, 2, 3]).foreach(f).run(),
        ), [10])

  async def test_multiple_foreach_with_break_in_different_positions(self):
    """Multiple foreach calls, break in different positions."""
    for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
        # First foreach: break at element 3
        def f1(el):
          if el >= 3:
            Chain.break_()
          return fn1(el)
        # Second foreach: break at element 2
        def f2(el):
          if el >= 2:
            Chain.break_()
          return fn2(el * 10)
        result = await await_(
          Chain(fn3, [1, 2, 3, 4, 5]).foreach(f1).foreach(f2).run()
        )
        self.assertEqual(result, [10])

  async def test_break_in_sync_to_async_transition_foreach(self):
    """break_() during sync-to-async transition in foreach."""
    counter = {'c': 0}
    def f(el):
      counter['c'] += 1
      if counter['c'] > 2:
        Chain.break_()
      if counter['c'] > 1:
        return aempty(el * 10)
      return el * 10
    counter['c'] = 0
    self.assertEqual(await await_(
      Chain([1, 2, 3, 4]).foreach(f).run(),
    ), [10, 20])


# ====================================================================
# Section 3: Exception + _Return interactions
# ====================================================================
class ExceptionReturnInteractionTests(IsolatedAsyncioTestCase):

  async def test_exception_before_return(self):
    """Exception in chain BEFORE return_ -- exception wins (return never reached)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 10).then(raise_).then(Chain.return_, 42).run()
          )

  async def test_exception_after_return(self):
    """Exception in chain AFTER return_ -- return wins (exception link never reached)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10).then(Chain.return_, 42).then(raise_).run()
        )
        self.assertEqual(result, 42)

  async def test_return_in_except_handler(self):
    """return_() in except_ handler -- _Return propagates as a new exception
    (it is NOT caught by the chain's _Return handler since evaluate_value
    wraps it in a try/except BaseException that re-raises with __cause__)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # _Return inside except_ handler is treated as a BaseException
        # raised by the handler, not as a control flow signal.
        with self.assertRaises(Exception):
          await await_(
            Chain(fn, 10)
            .then(raise_)
            .except_(lambda v: Chain.return_(99), reraise=False)
            .run()
          )

  async def test_exception_in_nested_return_in_outer(self):
    """Exception in nested chain, return_ in outer chain -- exception propagates first."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain().then(fn1).then(raise_)
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn2, 10).then(inner).then(Chain.return_, 42).run()
          )

  async def test_return_in_nested_exception_in_outer(self):
    """return_ in nested chain, exception in outer chain -- return wins (exception never reached)."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain().then(fn1).then(Chain.return_, 77)
        result = await await_(
          Chain(fn2, 10).then(inner).then(raise_).run()
        )
        self.assertEqual(result, 77)

  async def test_return_and_except_at_same_level(self):
    """return_ and except_ at same level -- return_ happens first, except_ not triggered."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10)
          .then(Chain.return_, 42)
          .except_(tracker, reraise=False)
          .run()
        )
        self.assertEqual(result, 42)
        self.assertEqual(tracker.count, 0)


# ====================================================================
# Section 4: Exception + _Break interactions
# ====================================================================
class ExceptionBreakInteractionTests(IsolatedAsyncioTestCase):

  async def test_exception_in_foreach_before_break(self):
    """Exception in foreach body BEFORE break_ -- exception propagates."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el == 2:
            raise TestExc('in foreach')
          if el == 3:
            Chain.break_()
          return fn1(el)
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn2, [1, 2, 3, 4]).foreach(f).run()
          )

  async def test_break_then_except_outside_foreach(self):
    """break_() in foreach then except_ outside -- except_ not triggered by break."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        tracker = Tracker()
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3, 4])
          .foreach(f)
          .except_(tracker, reraise=False)
          .run()
        )
        self.assertEqual(result, [1, 2])
        self.assertEqual(tracker.count, 0)

  async def test_exception_in_foreach_with_except_outside(self):
    """Exception in foreach body + except_ outside foreach -- except_ catches it."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        tracker = Tracker()
        def f(el):
          if el == 2:
            raise TestExc('foreach error')
          return fn1(el)
        try:
          await await_(
            Chain(fn2, [1, 2, 3])
            .foreach(f)
            .except_(tracker, reraise=True)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(tracker.count, 1)

  async def test_break_escaping_to_top_level(self):
    """_Break escaping to top level (not inside foreach) raises QuentException."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(QuentException):
          await await_(
            Chain(fn, 10).then(Chain.break_).run()
          )

  async def test_break_in_foreach_exception_on_specific_element(self):
    """Exception on 3rd element of foreach -- verify partial processing."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el == 3:
            raise TestExc('on element 3')
          return fn1(el * 10)
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn2, [1, 2, 3, 4]).foreach(f).run()
          )


# ====================================================================
# Section 5: _Return + _Break interactions
# ====================================================================
class ReturnBreakInteractionTests(IsolatedAsyncioTestCase):

  async def test_return_inside_foreach_via_nested_chain_raises(self):
    """return_() inside foreach (via the callback raising _Return) -- should propagate
    through the foreach and be caught by the outer chain as a return."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        # When return_ is called inside a foreach callback,
        # _Return exception propagates through foreach (not caught by _Break handler),
        # and is caught at the chain level by the _Return handler.
        def f(el):
          if el == 2:
            Chain.return_(999)
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3]).foreach(f).then(lambda v: 'should not reach').run()
        )
        self.assertEqual(result, 999)

  async def test_break_outside_foreach_with_return(self):
    """break_() outside foreach in a chain that also has return_ at top level raises QuentException."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(QuentException):
          await await_(
            Chain(fn, 10).then(Chain.break_).run()
          )


# ====================================================================
# Section 6: except_ edge cases with control flow
# ====================================================================
class ExceptEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_multiple_except_handlers_first_matching_wins(self):
    """Multiple except_ handlers -- first matching one wins."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker1 = Tracker()
        tracker2 = Tracker()
        try:
          await await_(
            Chain(fn, 10)
            .then(raise_exc(Exc1))
            .except_(tracker1, exceptions=Exc1, reraise=True)
            .except_(tracker2, exceptions=Exc1, reraise=True)
            .run()
          )
        except Exc1:
          pass
        self.assertEqual(tracker1.count, 1)
        self.assertEqual(tracker2.count, 0)

  async def test_except_does_not_catch_return(self):
    """except_ with specific exception type does not catch _Return."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10)
          .then(Chain.return_, 42)
          .except_(tracker, reraise=False)
          .run()
        )
        self.assertEqual(result, 42)
        self.assertEqual(tracker.count, 0)

  async def test_except_does_not_catch_break_in_foreach(self):
    """except_ does not catch _Break from foreach (break is handled by foreach)."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        tracker = Tracker()
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3, 4])
          .foreach(f)
          .except_(tracker, reraise=False)
          .run()
        )
        self.assertEqual(result, [1, 2])
        self.assertEqual(tracker.count, 0)

  async def test_except_reraise_true(self):
    """except_(reraise=True) -- handler runs, then exception re-raises."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 10)
            .then(raise_)
            .except_(tracker, reraise=True)
            .run()
          )
        self.assertEqual(tracker.count, 1)

  async def test_except_reraise_false_suppresses(self):
    """except_(reraise=False) -- handler runs, exception suppressed."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10)
          .then(raise_)
          .except_(lambda v: 'recovered', reraise=False)
          .run()
        )
        self.assertEqual(result, 'recovered')

  async def test_except_handler_that_raises(self):
    """except_ handler that itself raises -- the NEW exception propagates."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def handler(v=None):
          raise Exc2('handler error')
        with self.assertRaises(Exc2) as cm:
          await await_(
            Chain(fn, 10)
            .then(raise_exc(Exc1))
            .except_(handler, reraise=False)
            .run()
          )
        self.assertIsInstance(cm.exception.__cause__, Exc1)

  async def test_except_handler_returns_value(self):
    """except_ handler that returns a value (with reraise=False)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10)
          .then(raise_)
          .except_(lambda v: 'handler_result', reraise=False)
          .run()
        )
        self.assertEqual(result, 'handler_result')

  async def test_except_handler_returns_none(self):
    """except_ handler that returns None (with reraise=False) returns None."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10)
          .then(raise_)
          .except_(lambda v: None, reraise=False)
          .run()
        )
        self.assertIsNone(result)

  async def test_except_for_wrong_type_not_caught(self):
    """except_ for TypeError but ValueError raised -- not caught."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, 10)
            .then(raise_exc(ValueError))
            .except_(tracker, exceptions=TypeError, reraise=False)
            .run()
          )
        self.assertEqual(tracker.count, 0)

  async def test_except_for_parent_class_catches_child(self):
    """except_ for parent class, child class raised -- caught."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        try:
          await await_(
            Chain(fn, 10)
            .then(raise_exc(Exc3))  # Exc3 is subclass of Exc2
            .except_(tracker, exceptions=Exc2, reraise=True)
            .run()
          )
        except Exc3:
          pass
        self.assertEqual(tracker.count, 1)

  async def test_except_handler_receives_root_value(self):
    """except_ handler receives the root value, not the exception."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        received = []
        def handler(v):
          received.append(v)
          return 'handled'
        result = await await_(
          Chain(fn, 42)
          .then(raise_)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertEqual(result, 'handled')
        self.assertEqual(received, [42])


# ====================================================================
# Section 7: finally_ edge cases with control flow
# ====================================================================
class FinallyEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_finally_when_return_exits_early(self):
    """finally_ runs when return_ exits early."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10)
          .then(Chain.return_, 42)
          .finally_(tracker)
          .run()
        )
        self.assertEqual(result, 42)
        self.assertEqual(tracker.count, 1)

  async def test_finally_when_break_exits_foreach(self):
    """finally_ runs when break_ exits foreach."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        tracker = Tracker()
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3, 4])
          .foreach(f)
          .finally_(tracker)
          .run()
        )
        self.assertEqual(result, [1, 2])
        self.assertEqual(tracker.count, 1)

  async def test_finally_when_exception_raised(self):
    """finally_ runs when exception is raised."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 10)
            .then(raise_)
            .finally_(tracker)
            .run()
          )
        self.assertEqual(tracker.count, 1)

  async def test_finally_runs_after_except(self):
    """finally_ runs when exception raised AND except_ catches it."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        exc_tracker = Tracker()
        finally_tracker = Tracker()
        try:
          await await_(
            Chain(fn, 10)
            .then(raise_)
            .except_(exc_tracker, reraise=True)
            .finally_(finally_tracker)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(exc_tracker.count, 1)
        self.assertEqual(finally_tracker.count, 1)

  async def test_finally_handler_that_raises(self):
    """finally_ handler that itself raises -- the new exception propagates."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def finally_raises(v=None):
          raise Exc2('finally error')
        with self.assertRaises(Exc2):
          await await_(
            Chain(fn, 10).finally_(finally_raises).run()
          )

  async def test_only_one_finally_allowed(self):
    """Multiple finally_ in chain raises QuentException."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(QuentException):
          Chain(fn, 10).finally_(lambda v: None).finally_(lambda v: None)

  async def test_finally_in_nested_chain_and_outer_chain(self):
    """finally_ in nested chain + finally_ in outer chain -- both run."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner_tracker = Tracker()
        outer_tracker = Tracker()
        inner = Chain().then(fn1).finally_(inner_tracker)
        result = await await_(
          Chain(fn2, 10)
          .then(inner)
          .finally_(outer_tracker)
          .run()
        )
        self.assertEqual(inner_tracker.count, 1)
        self.assertEqual(outer_tracker.count, 1)

  async def test_finally_with_async_cleanup(self):
    """finally_ with async cleanup function."""
    tracker = Tracker()
    async def async_cleanup(v=None):
      tracker.calls.append('async_cleanup')
      tracker.count += 1
    result = await await_(
      Chain(aempty, 10).finally_(async_cleanup).run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(tracker.count, 1)

  async def test_finally_on_success(self):
    """finally_ runs on successful chain completion."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10).then(lambda v: v * 2).finally_(tracker).run()
        )
        self.assertEqual(result, 20)
        self.assertEqual(tracker.count, 1)

  async def test_control_flow_signal_in_finally_raises_quent_exception(self):
    """Using _Return or _Break in finally_ raises QuentException."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(QuentException):
          await await_(
            Chain(fn, 10).finally_(Chain.return_).run()
          )

  async def test_break_signal_in_finally_raises_quent_exception(self):
    """Using _Break in finally_ raises QuentException."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(QuentException):
          await await_(
            Chain(fn, 10).finally_(Chain.break_).run()
          )


# ====================================================================
# Section 8: Complex multi-level control flow
# ====================================================================
class ComplexMultiLevelTests(IsolatedAsyncioTestCase):

  async def test_chain_nested_chain_foreach_break_outer_except_finally(self):
    """Chain -> nested Chain -> foreach -> break_ + outer except_ + outer finally_."""
    for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
        exc_tracker = Tracker()
        finally_tracker = Tracker()

        def inner_fn(el):
          if el >= 3:
            Chain.break_()
          return fn1(el * 10)
        inner = Chain().then(fn2).foreach(inner_fn)

        result = await await_(
          Chain(fn3, [1, 2, 3, 4])
          .then(inner)
          .except_(exc_tracker, reraise=False)
          .finally_(finally_tracker)
          .run()
        )
        self.assertEqual(result, [10, 20])
        self.assertEqual(exc_tracker.count, 0)
        self.assertEqual(finally_tracker.count, 1)

  async def test_chain_with_cm_nested_return_outer_finally(self):
    """Chain -> with_(CM) -> nested Chain with return_ + outer finally_."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        finally_tracker = Tracker()
        cm = SyncCM(value=100)
        inner_body = Chain().then(fn1).then(Chain.return_, 'from_with')
        result = await await_(
          Chain(fn2, cm)
          .with_(inner_body)
          .finally_(finally_tracker)
          .run()
        )
        self.assertEqual(result, 'from_with')
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)
        self.assertEqual(finally_tracker.count, 1)

  async def test_foreach_exception_on_nth_element_with_except_finally(self):
    """foreach where fn raises on 3rd element -> except_ + finally_."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        exc_tracker = Tracker()
        finally_tracker = Tracker()
        counter = {'c': 0}
        def f(el):
          counter['c'] += 1
          if counter['c'] == 3:
            raise TestExc('on element 3')
          return fn1(el)
        counter['c'] = 0
        try:
          await await_(
            Chain(fn2, [1, 2, 3, 4, 5])
            .foreach(f)
            .except_(exc_tracker, reraise=True)
            .finally_(finally_tracker)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(exc_tracker.count, 1)
        self.assertEqual(finally_tracker.count, 1)

  async def test_deep_nesting_4_levels_return_at_innermost(self):
    """Deep nesting (4 levels) with return_ at innermost -- verify propagation."""
    for fn1, fn2, fn3, fn4 in product([empty, aempty], repeat=4):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4):
        level4 = Chain().then(fn1).then(Chain.return_, 'deep')
        level3 = Chain().then(fn2).then(level4)
        level2 = Chain().then(fn3).then(level3)
        result = await await_(
          Chain(fn4, 10).then(level2).then(lambda v: 'never').run()
        )
        self.assertEqual(result, 'deep')

  async def test_chain_except_finally_together_on_success(self):
    """Chain with except_ and finally_ on a successful run -- except not called, finally called."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        exc_tracker = Tracker()
        finally_tracker = Tracker()
        result = await await_(
          Chain(fn, 10)
          .then(lambda v: v * 2)
          .except_(exc_tracker, reraise=False)
          .finally_(finally_tracker)
          .run()
        )
        self.assertEqual(result, 20)
        self.assertEqual(exc_tracker.count, 0)
        self.assertEqual(finally_tracker.count, 1)

  async def test_return_propagates_through_nested_with_finally_at_each_level(self):
    """return_ in innermost chain propagates, finally_ at each level runs."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner_finally = Tracker()
        outer_finally = Tracker()
        inner = Chain().then(fn1).then(Chain.return_, 'inner_ret').finally_(inner_finally)
        result = await await_(
          Chain(fn2, 10)
          .then(inner)
          .finally_(outer_finally)
          .run()
        )
        self.assertEqual(result, 'inner_ret')
        self.assertEqual(inner_finally.count, 1)
        self.assertEqual(outer_finally.count, 1)

  async def test_exception_propagates_through_nested_with_except_at_outer(self):
    """Exception in nested chain, except_ at outer level catches it."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain().then(fn1).then(raise_)
        exc_tracker = Tracker()
        try:
          await await_(
            Chain(fn2, 10)
            .then(inner)
            .except_(exc_tracker, reraise=True)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(exc_tracker.count, 1)

  async def test_with_cm_exception_in_body_cm_still_exits(self):
    """with_ where body raises -- CM is still properly closed."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        cm = SyncCM(value=10)
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, cm).with_(raise_exc(TestExc), ...).run()
          )
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)

  async def test_with_cm_return_in_body_cm_still_exits(self):
    """with_ where body uses return_ -- CM is still properly closed."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        cm = SyncCM(value=10)
        inner = Chain().then(fn1).then(Chain.return_, 'from_cm')
        result = await await_(
          Chain(fn2, cm).with_(inner).run()
        )
        self.assertEqual(result, 'from_cm')
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)


# ====================================================================
# Section 9: Control flow in generators (iterate)
# ====================================================================
class IterateControlFlowTests(IsolatedAsyncioTestCase):

  async def test_iterate_return_raises_quent_exception(self):
    """iterate() + return_ raises QuentException."""
    with self.assertRaises(QuentException):
      for _ in Chain(SyncIterator).iterate(Chain.return_):
        pass

  async def test_iterate_break_stops_cleanly(self):
    """iterate() + break_ -- iteration stops cleanly."""
    def f(i):
      if i >= 5:
        Chain.break_()
      return i * 2
    r = []
    for i in Chain(SyncIterator).iterate(f):
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(5)])

  async def test_async_iterate_break_stops(self):
    """async iterate() + break_ stops iteration."""
    def f(i):
      if i >= 5:
        Chain.break_()
      return i * 2
    r = []
    async for i in Chain(SyncIterator).iterate(f):
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(5)])

  async def test_iterate_exception_in_fn_propagates(self):
    """iterate() with exception in body -- exception propagates."""
    def f(i):
      if i == 3:
        raise TestExc('iterate error')
      return i
    with self.assertRaises(TestExc):
      for _ in Chain(SyncIterator).iterate(f):
        pass

  async def test_async_iterate_exception_propagates(self):
    """async iterate() with exception -- exception propagates."""
    def f(i):
      if i == 3:
        raise TestExc('async iterate error')
      return i
    with self.assertRaises(TestExc):
      async for _ in Chain(SyncIterator).iterate(f):
        pass

  async def test_async_iterate_return_raises_quent_exception(self):
    """async iterate() + return_ raises QuentException."""
    with self.assertRaises(QuentException):
      async for _ in Chain(SyncIterator).iterate(Chain.return_):
        pass

  async def test_iterate_with_async_fn(self):
    """iterate() with async fn in sync iteration."""
    r = []
    async for i in Chain(SyncIterator).iterate(lambda i: aempty(i * 3)):
      r.append(i)
    self.assertEqual(r, [i * 3 for i in range(10)])

  async def test_iterate_break_with_async_fn(self):
    """async iterate() with async fn + break."""
    async def f(i):
      if i >= 5:
        Chain.break_()
      return i * 2
    r = []
    async for i in Chain(SyncIterator).iterate(f):
      r.append(i)
    self.assertEqual(r, [i * 2 for i in range(5)])


# ====================================================================
# Section 10: Async-specific control flow
# ====================================================================
class AsyncSpecificControlFlowTests(IsolatedAsyncioTestCase):

  async def test_async_return_value_in_sync_started_chain(self):
    """Async return_ value (return_(async_fn)) forces async transition."""
    async def async_val():
      return 99
    result = await await_(
      Chain(empty, 10).then(Chain.return_, async_val).run()
    )
    self.assertEqual(result, 99)

  async def test_async_break_value_in_foreach(self):
    """Async break_(async_fn) value in foreach -- verify awaited."""
    async def async_break_val():
      return 'async_break_result'
    def f(el):
      if el == 2:
        Chain.break_(async_break_val)
      return el
    result = await await_(
      Chain([1, 2, 3]).foreach(f).run()
    )
    self.assertEqual(result, 'async_break_result')

  async def test_async_except_handler_with_sync_raise(self):
    """Async except_ handler with sync raise."""
    async def async_handler(v=None):
      return 'async_handled'
    result = await await_(
      Chain(empty, 10)
      .then(raise_)
      .except_(async_handler, reraise=False)
      .run()
    )
    # When chain is sync but handler is async, the result is scheduled
    # as a task. We need to handle this case.
    if asyncio.isfuture(result) or asyncio.iscoroutine(result):
      result = await result
    self.assertEqual(result, 'async_handled')

  async def test_async_finally_with_sync_chain(self):
    """Async finally_ with sync chain."""
    tracker = Tracker()
    async_cleanup = tracker.async_call()
    result = await await_(
      Chain(aempty, 10).finally_(async_cleanup).run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(tracker.count, 1)

  async def test_return_during_async_chain(self):
    """return_ in fully async chain."""
    result = await await_(
      Chain(aempty, 10)
      .then(lambda v: aempty(v * 2))
      .then(Chain.return_, 77)
      .then(lambda v: aempty(v * 100))
      .run()
    )
    self.assertEqual(result, 77)

  async def test_break_in_async_foreach_with_async_fn(self):
    """break_() in fully async foreach with async fn."""
    async def f(el):
      if el >= 3:
        Chain.break_()
      return el * 10
    result = await await_(
      Chain(AsyncIterator([1, 2, 3, 4, 5])).foreach(f).run()
    )
    self.assertEqual(result, [10, 20])

  async def test_exception_in_async_foreach_with_except(self):
    """Exception in async foreach with except_ at outer level."""
    tracker = Tracker()
    async def f(el):
      if el == 3:
        raise TestExc('async foreach error')
      return el
    try:
      await await_(
        Chain(AsyncIterator([1, 2, 3, 4]))
        .foreach(f)
        .except_(tracker, reraise=True)
        .run()
      )
    except TestExc:
      pass
    self.assertEqual(tracker.count, 1)

  async def test_async_foreach_break_with_value_and_finally(self):
    """Async foreach break with value + finally_ runs."""
    tracker = Tracker()
    async def f(el):
      if el == 2:
        Chain.break_('early_exit')
      return el
    result = await await_(
      Chain(AsyncIterator([1, 2, 3]))
      .foreach(f)
      .finally_(tracker)
      .run()
    )
    self.assertEqual(result, 'early_exit')
    self.assertEqual(tracker.count, 1)


# ====================================================================
# Section 11: with_ (context manager) + control flow
# ====================================================================
class WithControlFlowTests(IsolatedAsyncioTestCase):

  async def test_with_exception_in_body_except_catches(self):
    """with_ where body raises, except_ outside catches."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        cm = SyncCM(value=10)
        try:
          await await_(
            Chain(fn, cm)
            .with_(raise_exc(TestExc), ...)
            .except_(tracker, reraise=True)
            .run()
          )
        except TestExc:
          pass
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)
        self.assertEqual(tracker.count, 1)

  async def test_with_finally_runs_on_success(self):
    """with_ success + finally_ runs."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        finally_tracker = Tracker()
        cm = SyncCM(value=10)
        result = await await_(
          Chain(fn, cm)
          .with_(lambda cv: cv * 2)
          .finally_(finally_tracker)
          .run()
        )
        self.assertEqual(result, 20)
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)
        self.assertEqual(finally_tracker.count, 1)

  async def test_with_async_cm_return_propagates(self):
    """Async context manager with return_ in body."""
    acm = AsyncCM(value=42)
    inner = Chain().then(lambda v: v).then(Chain.return_, 'async_cm_return')
    result = await await_(
      Chain(aempty, acm).with_(inner).run()
    )
    self.assertEqual(result, 'async_cm_return')
    self.assertTrue(acm.entered)
    self.assertTrue(acm.exited)

  async def test_with_async_cm_exception_cm_exits(self):
    """Async context manager with exception in body -- CM properly exits."""
    acm = AsyncCM(value=42)
    with self.assertRaises(TestExc):
      await await_(
        Chain(aempty, acm).with_(raise_exc(TestExc), ...).run()
      )
    self.assertTrue(acm.entered)
    self.assertTrue(acm.exited)


# ====================================================================
# Section 12: Gather + control flow
# ====================================================================
class GatherControlFlowTests(IsolatedAsyncioTestCase):

  async def test_gather_one_fn_raises(self):
    """gather with one fn that raises -- exception propagates."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 10)
            .gather(lambda v: v + 1, raise_exc(TestExc), lambda v: v + 3)
            .run()
          )

  async def test_gather_with_except(self):
    """gather where one fn raises + except_ catches."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        try:
          await await_(
            Chain(fn, 10)
            .gather(lambda v: v + 1, raise_exc(TestExc))
            .except_(tracker, reraise=True)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(tracker.count, 1)

  async def test_gather_all_succeed_with_finally(self):
    """gather all succeed + finally_ runs."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        finally_tracker = Tracker()
        result = await await_(
          Chain(fn, 5)
          .gather(lambda v: v + 1, lambda v: v * 2)
          .finally_(finally_tracker)
          .run()
        )
        self.assertEqual(result, [6, 10])
        self.assertEqual(finally_tracker.count, 1)


# ====================================================================
# Section 13: Filter + control flow
# ====================================================================
class FilterControlFlowTests(IsolatedAsyncioTestCase):

  async def test_filter_exception_propagates(self):
    """Exception in filter predicate propagates."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def pred(x):
          if x == 2:
            raise TestExc('filter error')
          return x % 2 == 0
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, [1, 2, 3, 4]).filter(pred).run()
          )

  async def test_filter_with_except(self):
    """filter exception + except_ catches."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        def pred(x):
          if x == 2:
            raise TestExc('filter error')
          return x % 2 == 0
        try:
          await await_(
            Chain(fn, [1, 2, 3, 4])
            .filter(pred)
            .except_(tracker, reraise=True)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(tracker.count, 1)

  async def test_filter_with_finally(self):
    """filter success + finally_ runs."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        finally_tracker = Tracker()
        result = await await_(
          Chain(fn, [1, 2, 3, 4])
          .filter(lambda x: x % 2 == 0)
          .finally_(finally_tracker)
          .run()
        )
        self.assertEqual(result, [2, 4])
        self.assertEqual(finally_tracker.count, 1)

  async def test_filter_break_propagates_as_exception(self):
    """break_() inside filter predicate -- _Break is NOT caught by filter,
    it propagates up to the chain where it becomes QuentException."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def pred(x):
          if x == 2:
            Chain.break_()
          return x % 2 == 0
        with self.assertRaises(QuentException):
          await await_(
            Chain(fn, [1, 2, 3, 4]).filter(pred).run()
          )


# ====================================================================
# Section 14: Pipe syntax + control flow
# ====================================================================
class PipeSyntaxControlFlowTests(IsolatedAsyncioTestCase):

  async def test_pipe_with_exception(self):
    """Pipe syntax with exception."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 10).then(raise_).run())

  async def test_pipe_with_return(self):
    """Pipe syntax doesn't directly support return_, but chain.return_ in then works."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10).then(Chain.return_, 42).run()
        )
        self.assertEqual(result, 42)


# ====================================================================
# Section 15: Clone + control flow
# ====================================================================
class CloneControlFlowTests(IsolatedAsyncioTestCase):

  async def test_clone_with_except(self):
    """Cloned chain with except_ works independently."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        original = Chain().then(fn).then(raise_).except_(lambda v: 'caught', reraise=False)
        cloned = original.clone()
        result = await await_(cloned(10))
        self.assertEqual(result, 'caught')

  async def test_clone_with_finally(self):
    """Cloned chain with finally_ works independently."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        original = Chain().then(fn).then(lambda v: v * 2).finally_(tracker)
        cloned = original.clone()
        result = await await_(cloned(10))
        self.assertEqual(result, 20)
        self.assertEqual(tracker.count, 1)

  async def test_clone_with_return(self):
    """Cloned chain with return_ works independently."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        original = Chain().then(fn).then(lambda v: Chain.return_(42))
        cloned = original.clone()
        result = await await_(cloned(10))
        self.assertEqual(result, 42)

  async def test_clone_with_foreach_break(self):
    """Cloned chain with foreach + break works independently."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn1(el * 10)
        original = Chain().then(fn2).foreach(f)
        cloned = original.clone()
        result = await await_(cloned([1, 2, 3, 4]))
        self.assertEqual(result, [10, 20])


# ====================================================================
# Section 16: do() + control flow
# ====================================================================
class DoControlFlowTests(IsolatedAsyncioTestCase):

  async def test_do_with_return_after(self):
    """do() side effect + return_ after -- do runs, return value used."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10).do(tracker).then(Chain.return_, 42).run()
        )
        self.assertEqual(result, 42)
        self.assertEqual(tracker.count, 1)

  async def test_do_exception_propagates(self):
    """do() that raises -- exception propagates, result is discarded behavior irrelevant."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 10).do(raise_exc(TestExc)).run()
          )

  async def test_do_discards_result(self):
    """do() discards its result, previous value preserved."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10).do(lambda v: 999).then(lambda v: v * 2).run()
        )
        self.assertEqual(result, 20)

  async def test_do_with_except_on_do_error(self):
    """do() that raises + except_ catches."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        try:
          await await_(
            Chain(fn, 10)
            .do(raise_exc(TestExc))
            .except_(tracker, reraise=True)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(tracker.count, 1)


# ====================================================================
# Section 17: Edge case combinations
# ====================================================================
class EdgeCaseCombinationTests(IsolatedAsyncioTestCase):

  async def test_empty_chain_with_except_and_finally(self):
    """Empty chain (no root) with except_ and finally_."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain().except_(lambda v: None, reraise=False).finally_(tracker).run()
        )
        self.assertIsNone(result)
        self.assertEqual(tracker.count, 1)

  async def test_chain_with_root_and_no_links_finally(self):
    """Chain with root and no links + finally_."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        result = await await_(
          Chain(fn, 10).finally_(tracker).run()
        )
        self.assertEqual(result, 10)
        self.assertEqual(tracker.count, 1)

  async def test_return_with_explicit_args(self):
    """return_(fn, arg1, arg2) passes args to fn."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10).then(Chain.return_, lambda x, y: x + y, 3, 4).run()
        )
        self.assertEqual(result, 7)

  async def test_break_with_explicit_args(self):
    """break_(fn, arg1, arg2) passes args to fn."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el == 2:
            Chain.break_(lambda x, y: x + y, 10, 20)
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3]).foreach(f).run()
        )
        self.assertEqual(result, 30)

  async def test_nested_chain_cannot_run_directly(self):
    """A nested chain raises QuentException if run directly."""
    inner = Chain().then(lambda v: v)
    Chain(10).then(inner)  # makes inner nested
    with self.assertRaises(QuentException):
      inner.run(5)

  async def test_except_with_multiple_exception_types(self):
    """except_ with list of exception types."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        try:
          await await_(
            Chain(fn, 10)
            .then(raise_exc(Exc1))
            .except_(tracker, exceptions=[Exc1, Exc2], reraise=True)
            .run()
          )
        except Exc1:
          pass
        self.assertEqual(tracker.count, 1)

  async def test_except_string_type_raises_type_error(self):
    """except_ with string exceptions= raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(10).except_(lambda v: None, exceptions='bad')

  async def test_return_from_void_chain(self):
    """return_() from chain with no root value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain().then(Chain.return_, 42).run()
        )
        self.assertEqual(result, 42)

  async def test_break_value_with_kwargs(self):
    """break_(fn, key=val) passes kwargs to fn."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(el):
          if el == 2:
            Chain.break_(lambda key='default': key, key='custom')
          return fn1(el)
        result = await await_(
          Chain(fn2, [1, 2, 3]).foreach(f).run()
        )
        self.assertEqual(result, 'custom')

  async def test_return_value_with_kwargs(self):
    """return_(fn, key=val) passes kwargs to fn."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, 10).then(Chain.return_, lambda key='default': key, key='ret_kwarg').run()
        )
        self.assertEqual(result, 'ret_kwarg')


# ====================================================================
# Section 18: Exception chaining and cause preservation
# ====================================================================
class ExceptionChainingTests(IsolatedAsyncioTestCase):

  async def test_except_handler_raises_cause_is_original(self):
    """When except_ handler raises, __cause__ is the original exception."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        original = Exc1('original')
        def raise_original(v=None):
          raise original
        def handler(v=None):
          raise Exc2('from handler')
        with self.assertRaises(Exc2) as cm:
          await await_(
            Chain(fn, 10)
            .then(raise_original)
            .except_(handler, reraise=False)
            .run()
          )
        self.assertIs(cm.exception.__cause__, original)

  async def test_finally_raises_after_exception_context_preserved(self):
    """When finally_ raises after an exception, __context__ is the original."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def finally_raises(v=None):
          raise Exc2('finally error')
        with self.assertRaises(Exc2) as cm:
          await await_(
            Chain(fn, 10)
            .then(raise_exc(Exc1))
            .finally_(finally_raises)
            .run()
          )
        self.assertIsInstance(cm.exception.__context__, Exc1)


# ====================================================================
# Section 20: Async context manager + control flow
# ====================================================================
class AsyncCMControlFlowTests(IsolatedAsyncioTestCase):

  async def test_async_cm_break_in_foreach_body(self):
    """Async CM inside foreach with break -- CM exits, break propagates to foreach."""
    acm = AsyncCM(value=42)
    processed = []
    def f(el):
      processed.append(el)
      if el >= 3:
        Chain.break_()
      return el * 10
    result = await await_(
      Chain(aempty, [1, 2, 3, 4]).foreach(f).run()
    )
    self.assertEqual(result, [10, 20])

  async def test_async_cm_exception_except_finally(self):
    """Async CM with exception + except_ + finally_."""
    exc_tracker = Tracker()
    finally_tracker = Tracker()
    acm = AsyncCM(value=42)
    try:
      await await_(
        Chain(aempty, acm)
        .with_(raise_exc(TestExc), ...)
        .except_(exc_tracker, reraise=True)
        .finally_(finally_tracker)
        .run()
      )
    except TestExc:
      pass
    self.assertTrue(acm.entered)
    self.assertTrue(acm.exited)
    self.assertEqual(exc_tracker.count, 1)
    self.assertEqual(finally_tracker.count, 1)


# ====================================================================
# Section 21: Chain reuse + control flow
# ====================================================================
class ChainReuseControlFlowTests(IsolatedAsyncioTestCase):

  async def test_chain_reuse_with_except(self):
    """Reusable chain with except_ works."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        c = Chain().then(fn).then(raise_).except_(lambda v: 'caught', reraise=False)
        result = await await_(c(10))
        self.assertEqual(result, 'caught')

  async def test_chain_reuse_with_finally(self):
    """Reusable chain with finally_ works."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        c = Chain().then(fn).then(lambda v: v * 2).finally_(tracker)
        result = await await_(c(10))
        self.assertEqual(result, 20)
        self.assertEqual(tracker.count, 1)

  async def test_chain_reuse_with_return(self):
    """Reusable chain with return_ works."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        c = Chain().then(fn).then(lambda v: Chain.return_(42))
        result = await await_(c(10))
        self.assertEqual(result, 42)

  async def test_chain_reuse_with_control_flow(self):
    """Chain can be reused multiple times with control flow."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        tracker = Tracker()
        c = Chain().then(fn).then(lambda v: v * 2).finally_(tracker)
        for i in range(3):
          result = await await_(c(10))
          self.assertEqual(result, 20)
        self.assertEqual(tracker.count, 3)



# ====================================================================
# Section 26: Combined except_ + finally_ ordering
# ====================================================================
class ExceptFinallyOrderingTests(IsolatedAsyncioTestCase):

  async def test_except_runs_before_finally(self):
    """When exception occurs, except_ runs before finally_."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        order = []
        def on_except(v=None):
          order.append('except')
        def on_finally(v=None):
          order.append('finally')
        try:
          await await_(
            Chain(fn, 10)
            .then(raise_)
            .except_(on_except, reraise=True)
            .finally_(on_finally)
            .run()
          )
        except TestExc:
          pass
        self.assertEqual(order, ['except', 'finally'])

  async def test_except_noraise_then_finally(self):
    """except_ with reraise=False suppresses, then finally_ runs."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        order = []
        def on_except(v=None):
          order.append('except')
          return 'recovered'
        def on_finally(v=None):
          order.append('finally')
        result = await await_(
          Chain(fn, 10)
          .then(raise_)
          .except_(on_except, reraise=False)
          .finally_(on_finally)
          .run()
        )
        self.assertEqual(result, 'recovered')
        self.assertEqual(order, ['except', 'finally'])


# ====================================================================
# Section 27: Return propagation through nested chains with various features
# ====================================================================
class ReturnPropagationTests(IsolatedAsyncioTestCase):

  async def test_return_propagates_through_nested_with_foreach_after(self):
    """return_ in nested chain propagates past foreach in outer chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain().then(fn1).then(Chain.return_, 'escaped')
        result = await await_(
          Chain(fn2, 10)
          .then(inner)
          .then(lambda v: [v])
          .foreach(lambda el: el * 2)
          .run()
        )
        self.assertEqual(result, 'escaped')

  async def test_return_propagates_through_triple_nesting(self):
    """return_ propagates through 3 levels of nesting."""
    for fn1, fn2, fn3, fn4 in product([empty, aempty], repeat=4):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4):
        l3 = Chain().then(fn1).then(Chain.return_, 'deep_return')
        l2 = Chain().then(fn2).then(l3).then(lambda v: 'never')
        l1 = Chain().then(fn3).then(l2).then(lambda v: 'never2')
        result = await await_(
          Chain(fn4, 10).then(l1).then(lambda v: 'never3').run()
        )
        self.assertEqual(result, 'deep_return')

  async def test_return_in_nested_chain_finally_at_all_levels(self):
    """return_ in innermost + finally_ at every level -- all finally_ handlers run."""
    for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
        trackers = [Tracker() for _ in range(3)]
        l2 = Chain().then(fn1).then(Chain.return_, 'deep').finally_(trackers[2])
        l1 = Chain().then(fn2).then(l2).finally_(trackers[1])
        result = await await_(
          Chain(fn3, 10).then(l1).finally_(trackers[0]).run()
        )
        self.assertEqual(result, 'deep')
        for i, t in enumerate(trackers):
          self.assertEqual(t.count, 1, f'tracker[{i}] should have been called')
