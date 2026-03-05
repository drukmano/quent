"""Adversarial and edge-case tests for quent: reentrant runs, sentinel abuse,
consumed iterables, type confusion, control-flow interception, concurrency,
thread safety, and other boundary conditions.
"""
from __future__ import annotations

import asyncio
import threading
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from quent._core import _Return, _Break, _ControlFlowSignal
from quent._chain import _FrozenChain
from helpers import async_fn


# ---------------------------------------------------------------------------
# Reentrant runs: creating or running chains inside chain steps
# ---------------------------------------------------------------------------

class TestReentrantRun(unittest.TestCase):

  def test_run_inside_chain_step(self):
    """Creating and running a brand-new Chain inside a step works normally."""
    result = Chain(5).then(lambda x: Chain(x).then(lambda y: y + 1).run()).run()
    self.assertEqual(result, 6)

  def test_recursive_chain_run_same_chain(self):
    """Running the *same* chain object recursively works because _run uses
    local variables for link traversal -- no shared mutable iteration state.
    """
    c = Chain()
    c.then(lambda x: c.run(x - 1) if x > 0 else x)
    result = c.run(5)
    self.assertEqual(result, 0)

  def test_recursive_chain_with_accumulation(self):
    """Recursive same-chain with accumulating result."""
    c = Chain()
    c.then(lambda x: c.run(x - 1) + 1 if x > 0 else 0)
    result = c.run(5)
    self.assertEqual(result, 5)


# ---------------------------------------------------------------------------
# Null sentinel as explicit value
# ---------------------------------------------------------------------------

class TestNullAsExplicitValue(unittest.TestCase):

  def test_then_with_null_sentinel(self):
    """Null is not callable, so .then(Null) sets current_value to Null.
    At the end of the loop, `if current_value is Null: return None`.
    So the final result is None.
    """
    result = Chain(5).then(Null).run()
    self.assertIsNone(result)

  def test_run_with_null_means_no_injection(self):
    """run(Null) is treated as run() -- v is Null means has_run_value=False,
    so the root_link is used. Result is 42.
    """
    result = Chain(42).run(Null)
    self.assertEqual(result, 42)

  def test_null_not_equal_to_none(self):
    """Null and None are distinct; a chain that produces None is not Null."""
    self.assertIsNot(Null, None)
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_chain_initialized_with_null_has_no_root(self):
    """Chain(Null) is equivalent to Chain() -- root_link is None."""
    c = Chain(Null)
    self.assertIsNone(c.root_link)


# ---------------------------------------------------------------------------
# Ellipsis as data (not first-arg sentinel)
# ---------------------------------------------------------------------------

class TestEllipsisAsData(unittest.TestCase):

  def test_ellipsis_not_first_arg(self):
    """When Ellipsis is NOT the first positional arg, it passes through
    as a literal value. Link args=(1, ...) -> v(1, ...).
    """
    result = Chain(5).then(lambda x, y: (x, y), 1, ...).run()
    self.assertEqual(result, (1, Ellipsis))

  def test_ellipsis_as_only_arg_calls_with_no_args(self):
    """When Ellipsis IS the first (only) arg, the callable is invoked
    with zero arguments, ignoring the current chain value.
    """
    result = Chain(999).then(lambda: 'no_args', ...).run()
    self.assertEqual(result, 'no_args')

  def test_ellipsis_with_trailing_args(self):
    """Ellipsis as first arg with trailing args: v() is called with no args.
    The trailing args in the tuple are ignored (only first element checked).
    Actually: args=(Ellipsis, 2, 3) -> args[0] is ... -> v() called.
    """
    # _evaluate_value: args[0] is ... -> return v()
    result = Chain(5).then(lambda: 'ok', ..., 2, 3).run()
    self.assertEqual(result, 'ok')


# ---------------------------------------------------------------------------
# Callable returning callable
# ---------------------------------------------------------------------------

class TestCallableReturningCallable(unittest.TestCase):

  def test_fn_returns_fn(self):
    """A step that returns a callable does NOT auto-call it.
    The callable is just the current_value. If there is no next step,
    it is returned as-is.
    """
    result = Chain(5).then(lambda x: lambda: x * 2).run()
    self.assertTrue(callable(result))
    self.assertEqual(result(), 10)

  def test_fn_returns_fn_then_next_step_receives_it(self):
    """When a step returns a callable, the next step receives that callable
    as its argument.
    """
    result = (
      Chain(5)
      .then(lambda x: lambda: x * 2)
      .then(lambda fn: fn())
      .run()
    )
    self.assertEqual(result, 10)

  def test_fn_returns_class(self):
    """Returning a class (which is callable) from a step: it's just the value."""
    result = Chain(5).then(lambda x: int).run()
    self.assertIs(result, int)


# ---------------------------------------------------------------------------
# Coroutine object passed directly to then
# ---------------------------------------------------------------------------

class TestCoroutineObjectDirectly(IsolatedAsyncioTestCase):

  async def test_coroutine_object_to_then(self):
    """Passing an already-created coroutine object to .then() -- it is not
    callable, but it IS awaitable. _evaluate_value returns it as a non-callable
    value, then isawaitable triggers the async path, and it gets awaited.
    """
    async def coro_fn():
      return 42

    coro = coro_fn()
    result = await Chain(5).then(coro).run()
    self.assertEqual(result, 42)

  async def test_coroutine_object_as_root(self):
    """Coroutine object as root value: it's not callable, so _evaluate_value
    returns it as-is. isawaitable triggers async, and the coroutine is awaited.
    """
    async def coro_fn():
      return 77

    result = await Chain(coro_fn()).run()
    self.assertEqual(result, 77)


# ---------------------------------------------------------------------------
# Consumed iterables
# ---------------------------------------------------------------------------

class TestConsumedIterables(unittest.TestCase):

  def test_foreach_on_consumed_generator(self):
    """A fully consumed generator yields nothing; foreach returns []."""
    gen = iter([1, 2, 3])
    list(gen)  # exhaust
    result = Chain(gen).foreach(lambda x: x).run()
    self.assertEqual(result, [])

  def test_filter_on_consumed_generator(self):
    """A fully consumed generator yields nothing; filter returns []."""
    gen = iter([1, 2, 3])
    list(gen)  # exhaust
    result = Chain(gen).filter(lambda x: True).run()
    self.assertEqual(result, [])

  def test_partially_consumed_generator(self):
    """A partially consumed generator starts from where it left off."""
    gen = iter([1, 2, 3, 4, 5])
    next(gen)  # consume 1
    next(gen)  # consume 2
    result = Chain(gen).foreach(lambda x: x * 10).run()
    self.assertEqual(result, [30, 40, 50])


# ---------------------------------------------------------------------------
# TypeError triggers: non-iterable to iteration ops
# ---------------------------------------------------------------------------

class TestTypeErrors(unittest.TestCase):

  def test_foreach_on_int(self):
    with self.assertRaises(TypeError):
      Chain(42).foreach(lambda x: x).run()

  def test_filter_on_int(self):
    with self.assertRaises(TypeError):
      Chain(42).filter(lambda x: True).run()

  def test_with_on_non_cm(self):
    """Calling .with_() on a non-context-manager raises AttributeError
    because int has no __aenter__ or __enter__.
    """
    with self.assertRaises(AttributeError):
      Chain(42).with_(lambda x: x).run()

  def test_foreach_on_none(self):
    with self.assertRaises(TypeError):
      Chain(None).foreach(lambda x: x).run()

  def test_filter_on_none(self):
    with self.assertRaises(TypeError):
      Chain(None).filter(lambda x: True).run()


# ---------------------------------------------------------------------------
# StopIteration raised inside foreach fn
# ---------------------------------------------------------------------------

class TestExplicitStopIteration(unittest.TestCase):

  def test_stop_iteration_in_foreach_fn(self):
    """When fn raises StopIteration, it is caught by the `except StopIteration`
    clause in the while/next loop of _foreach_op, ending iteration early.
    Only items processed *before* the raise are included.

    Processing: item=1 -> fn(1)=10 -> appended. item=2 -> fn(2) raises
    StopIteration -> caught -> return [10].
    """
    def fn(x):
      if x == 2:
        raise StopIteration
      return x * 10

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, [10])

  def test_stop_iteration_on_first_item(self):
    """StopIteration on the very first item: nothing appended yet -> []."""
    def fn(x):
      raise StopIteration

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, [])

  def test_stop_iteration_on_last_item(self):
    """StopIteration on the last item: all previous items included."""
    def fn(x):
      if x == 3:
        raise StopIteration
      return x * 10

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, [10, 20])


# ---------------------------------------------------------------------------
# Deeply nested chains
# ---------------------------------------------------------------------------

class TestDeeplyNested(unittest.TestCase):

  def test_50_levels_nesting_via_run(self):
    """50 levels of chains calling run() in lambdas (not true nested chains).
    Works because each run() is an independent call frame.
    """
    def make_nested(depth):
      if depth == 0:
        return Chain(1)
      inner = make_nested(depth - 1)
      return Chain().then(lambda x, inner=inner: inner.run(x))

    c = make_nested(50)
    result = c.run(1)
    self.assertEqual(result, 1)

  def test_50_levels_nesting_via_chain_objects(self):
    """50 levels of Chain-within-Chain (true nested chains via _is_chain).
    _evaluate_value calls inner._run() directly. No run() wrapper overhead.
    """
    c = Chain().then(lambda x: x)
    for _ in range(49):
      c = Chain().then(c)
    result = c.run(42)
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# Chain.__bool__ always True
# ---------------------------------------------------------------------------

class TestChainBoolAlwaysTrue(unittest.TestCase):

  def test_empty_chain_true(self):
    self.assertTrue(bool(Chain()))

  def test_chain_with_value_true(self):
    """Even when the root value is falsy (0), the chain itself is truthy."""
    self.assertTrue(bool(Chain(0)))

  def test_chain_with_none_true(self):
    self.assertTrue(bool(Chain(None)))

  def test_chain_with_false_true(self):
    self.assertTrue(bool(Chain(False)))

  def test_frozen_chain_true(self):
    self.assertTrue(bool(Chain().freeze()))

  def test_frozen_chain_with_falsy_true(self):
    self.assertTrue(bool(Chain(0).freeze()))


# ---------------------------------------------------------------------------
# Concurrent async execution
# ---------------------------------------------------------------------------

class TestConcurrentExecution(IsolatedAsyncioTestCase):

  async def test_same_chain_multiple_tasks(self):
    """Running the same (async) chain concurrently via asyncio.gather."""
    async def async_double(x):
      return x * 2

    c = Chain().then(async_double)
    results = await asyncio.gather(*[c.run(i) for i in range(5)])
    self.assertEqual(sorted(results), [0, 2, 4, 6, 8])

  async def test_frozen_chain_concurrent(self):
    """Running a frozen async chain concurrently."""
    async def async_double(x):
      return x * 2

    frozen = Chain().then(async_double).freeze()
    results = await asyncio.gather(*[frozen.run(i) for i in range(5)])
    self.assertEqual(sorted(results), [0, 2, 4, 6, 8])

  async def test_concurrent_with_delay(self):
    """Concurrent tasks with actual async delay to stress interleaving."""
    async def delayed_add(x):
      await asyncio.sleep(0.01)
      return x + 100

    frozen = Chain().then(delayed_add).freeze()
    results = await asyncio.gather(*[frozen.run(i) for i in range(10)])
    self.assertEqual(sorted(results), list(range(100, 110)))


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):

  def test_chain_created_one_thread_run_another(self):
    """Create chain in main thread, run in a worker thread."""
    c = Chain(5).then(lambda x: x + 1)
    results = []
    errors = []

    def worker():
      try:
        results.append(c.run())
      except Exception as e:
        errors.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    self.assertEqual(errors, [])
    self.assertEqual(results, [6])

  def test_frozen_chain_multiple_threads(self):
    """Run a frozen chain from multiple threads concurrently."""
    frozen = Chain().then(lambda x: x * 3).freeze()
    results = []
    lock = threading.Lock()

    def worker(val):
      r = frozen.run(val)
      with lock:
        results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    self.assertEqual(sorted(results), [i * 3 for i in range(10)])


# ---------------------------------------------------------------------------
# User code intercepting control-flow signals
# ---------------------------------------------------------------------------

class TestUserCodeCatchesControlFlow(unittest.TestCase):

  def test_user_except_catches_return_signal(self):
    """_Return is a subclass of Exception. User code doing `except Exception`
    in a step can intercept it, preventing early return. The step returns
    its own value instead.
    """
    def step(x):
      try:
        Chain.return_(99)
      except Exception:
        return 'intercepted'

    result = Chain(5).then(step).run()
    self.assertEqual(result, 'intercepted')

  def test_user_except_catches_break_signal(self):
    """Same for _Break: user code can intercept it."""
    def step(x):
      try:
        Chain.break_(42)
      except Exception:
        return 'intercepted'

    result = Chain(5).then(step).run()
    self.assertEqual(result, 'intercepted')

  def test_intercepted_return_does_not_exit_chain(self):
    """When user code catches _Return, the chain continues normally."""
    tracker = []

    def interceptor(x):
      try:
        Chain.return_('early')
      except Exception:
        return x

    result = (
      Chain(5)
      .then(interceptor)
      .then(lambda x: tracker.append(x) or x + 1)
      .run()
    )
    # Chain continued past the interceptor
    self.assertEqual(tracker, [5])
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# except_ with bad types as exception filter
# ---------------------------------------------------------------------------

class TestExceptWithBadTypes(unittest.TestCase):

  def test_except_with_int_as_filter(self):
    """exceptions=42: int is not str and not Iterable, so it goes to the
    else branch: on_except_exceptions = (42,). At runtime, isinstance(exc, (42,))
    raises TypeError.
    """
    c = Chain(lambda: 1 / 0).except_(lambda e: 'caught', exceptions=42)
    with self.assertRaises(TypeError):
      c.run()

  def test_except_with_none_as_filter(self):
    """exceptions=None uses the default (Exception,)."""
    result = Chain(lambda: 1 / 0).except_(lambda e: 'caught', exceptions=None).run()
    self.assertEqual(result, 'caught')

  def test_except_with_empty_list_raises(self):
    """exceptions=[] raises QuentException (at least one type required)."""
    with self.assertRaises(QuentException):
      Chain().except_(lambda e: e, exceptions=[])

  def test_except_with_string_raises_type_error(self):
    """exceptions='ValueError' raises TypeError at registration time."""
    with self.assertRaises(TypeError):
      Chain().except_(lambda e: e, exceptions='ValueError')


# ---------------------------------------------------------------------------
# Passing non-callables to methods that need callables
# ---------------------------------------------------------------------------

class TestPassingUncallableToMethodsRequiringCallable(unittest.TestCase):

  def test_do_with_non_callable(self):
    """do(42): Link(42, ignore_result=True). _evaluate_value: 42 is not callable,
    no args -> returns 42. ignore_result=True -> current_value not updated.
    Chain returns the original root value (5).
    """
    result = Chain(5).do(42).run()
    self.assertEqual(result, 5)

  def test_foreach_with_non_callable(self):
    """foreach(42): fn=42. fn(item) -> 42(1) -> TypeError."""
    with self.assertRaises(TypeError):
      Chain([1, 2, 3]).foreach(42).run()

  def test_filter_with_non_callable(self):
    """filter(42): fn=42. fn(item) -> 42(1) -> TypeError."""
    with self.assertRaises(TypeError):
      Chain([1, 2, 3]).filter(42).run()

  def test_gather_with_non_callable(self):
    """gather(42): fn=42. fn(current_value) -> 42(5) -> TypeError."""
    with self.assertRaises(TypeError):
      Chain(5).gather(42).run()

  def test_then_with_non_callable_literal(self):
    """then(42): non-callable, returned as-is. Replaces current_value."""
    result = Chain(5).then(42).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# Object with __call__ = None
# ---------------------------------------------------------------------------

class TestObjectWithCallNone(unittest.TestCase):

  def test_object_with_call_none(self):
    """callable() returns True for objects with __call__ attribute, even if
    it is None. But actually calling it raises TypeError.
    """
    class WeirdCallable:
      __call__ = None

    obj = WeirdCallable()
    self.assertTrue(callable(obj))
    # _evaluate_value: callable(obj) is True, so it tries obj(current_value)
    # which raises TypeError since __call__ is None.
    with self.assertRaises(TypeError):
      Chain(5).then(obj).run()


# ---------------------------------------------------------------------------
# Modification after freeze
# ---------------------------------------------------------------------------

class TestModificationAfterFreeze(unittest.TestCase):

  def test_add_then_after_freeze(self):
    """freeze() wraps the chain but does not copy it. The frozen chain
    delegates to the same Chain object. Modifying the underlying chain
    after freeze is undefined behavior, but in practice the frozen chain
    sees the added step because it's the same object.
    """
    c = Chain(5)
    frozen = c.freeze()
    # Add step AFTER freezing
    c.then(lambda x: x * 2)
    # Frozen chain sees the new step
    result = frozen.run()
    self.assertEqual(result, 10)

  def test_add_except_after_freeze(self):
    """Adding an except handler after freeze: the frozen chain sees it."""
    c = Chain(lambda: 1 / 0)
    frozen = c.freeze()
    c.except_(lambda e: 'caught')
    result = frozen.run()
    self.assertEqual(result, 'caught')

  def test_frozen_wraps_same_object(self):
    """Verify _FrozenChain._chain is the same object as the original."""
    c = Chain(5)
    frozen = c.freeze()
    self.assertIs(frozen._chain, c)


# ---------------------------------------------------------------------------
# Misc adversarial edge cases
# ---------------------------------------------------------------------------

class TestMiscAdversarial(unittest.TestCase):

  def test_chain_of_chains_result(self):
    """A chain that produces another chain as its value (not nested)."""
    inner = Chain(42)
    # .then(lambda x: inner) returns the Chain object as a value
    result = Chain(5).then(lambda x: inner).run()
    # The Chain object is returned as-is (it's just the current_value)
    # It is NOT auto-run because it's returned from a lambda, not added as a step
    self.assertIs(result, inner)

  def test_none_as_root_value(self):
    """Chain(None) is a valid root. None is not Null, so it's a real value."""
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_false_as_root_value(self):
    result = Chain(False).run()
    self.assertIs(result, False)

  def test_zero_as_root_value(self):
    result = Chain(0).run()
    self.assertEqual(result, 0)

  def test_empty_chain_returns_none(self):
    """Chain with no root and no steps: current_value is Null -> returns None."""
    result = Chain().run()
    self.assertIsNone(result)

  def test_exception_in_root_callable(self):
    """Exception in the root callable propagates normally."""
    with self.assertRaises(ZeroDivisionError):
      Chain(lambda: 1 / 0).run()

  def test_except_handler_with_ellipsis_args(self):
    """except_ handler with Ellipsis as first arg: calls handler with no args."""
    result = Chain(lambda: 1 / 0).except_(lambda: 'caught', ...).run()
    self.assertEqual(result, 'caught')

  def test_foreach_preserves_none_results(self):
    """foreach fn returning None: None is a valid list element."""
    result = Chain([1, 2, 3]).foreach(lambda x: None).run()
    self.assertEqual(result, [None, None, None])

  def test_filter_all_falsy_returns_empty(self):
    """Filter where every item is falsy -> empty list."""
    result = Chain([0, '', None, False]).filter(lambda x: x).run()
    self.assertEqual(result, [])


if __name__ == '__main__':
  unittest.main()
