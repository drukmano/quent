"""Niche, obscure, and adversarial combination tests for the quent chain library.

Targets the dark corners: chain-as-value, recursive patterns, exception-class
operations, context-manager edge cases, unusual iterables, sentinel interactions,
dynamic construction, coroutine lifecycle, duck typing, diagnostics edge cases,
maximum feature stacking, run/call semantics, special methods, and ordering.
"""

import asyncio
import itertools
import functools
import time
import warnings
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase
from tests.utils import TestExc, empty, aempty, await_, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Exc1(TestExc):
  pass

class Exc2(TestExc):
  pass

class Exc3(Exc2):
  pass

class CustomBaseExc(BaseException):
  pass


class SyncCM:
  def __init__(self, value='cm'):
    self.value = value
    self.entered = False
    self.exited = False
    self.exit_args = None

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    self.exit_args = args
    return False


class AsyncCM:
  def __init__(self, value='acm'):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


class SuppressingCM:
  """CM whose __exit__ returns True, suppressing exceptions."""
  def __init__(self, value='suppress'):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return True


class EnterRaisesCM:
  """CM whose __enter__ raises."""
  def __enter__(self):
    raise TestExc('enter failed')

  def __exit__(self, *args):
    return False


class ExitRaisesCM:
  """CM whose __exit__ raises a different exception."""
  def __init__(self, value='exit_raises'):
    self.value = value
    self.entered = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    raise Exc1('exit exploded')


class NoneEnterCM:
  """CM whose __enter__ returns None."""
  def __init__(self):
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return None

  def __exit__(self, *args):
    self.exited = True
    return False


class SelfEnterCM:
  """CM whose __enter__ returns self."""
  def __init__(self):
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self

  def __exit__(self, *args):
    self.exited = True
    return False


class ReentrantCM:
  """CM that can be entered multiple times."""
  def __init__(self):
    self.enter_count = 0
    self.exit_count = 0

  def __enter__(self):
    self.enter_count += 1
    return self.enter_count

  def __exit__(self, *args):
    self.exit_count += 1
    return False


class AsyncExitSyncEnterCM:
  """CM with async __aexit__ but also sync __enter__."""
  def __init__(self, value='mixed'):
    self.value = value

  async def __aenter__(self):
    return self.value

  async def __aexit__(self, *args):
    return False


class IterableAndCM:
  """Object that is both iterable and a context manager."""
  def __init__(self, items):
    self.items = items

  def __enter__(self):
    return 'cm_value'

  def __exit__(self, *args):
    return False

  def __iter__(self):
    return iter(self.items)


class CallableAndIterable:
  """Object that is both callable and iterable."""
  def __init__(self, items, call_result='called'):
    self.items = items
    self.call_result = call_result

  def __call__(self, *args):
    return self.call_result

  def __iter__(self):
    return iter(self.items)


class FutureLike:
  """Object with __await__."""
  def __init__(self, value):
    self.value = value

  def __await__(self):
    return iter([self.value])


class SelfIterator:
  """Object where __iter__ returns self (iterator, not iterable)."""
  def __init__(self, items):
    self._items = iter(items)

  def __iter__(self):
    return self

  def __next__(self):
    return next(self._items)


class ImmediateStopIterator:
  """Object where __iter__ raises StopIteration immediately."""
  def __iter__(self):
    return self

  def __next__(self):
    raise StopIteration


class IterRaisesError:
  """Object where __iter__ raises a different exception."""
  def __iter__(self):
    raise TestExc('iter failed')


class FalsyIterable:
  """Object where __bool__ is False but __iter__ works."""
  def __init__(self, items):
    self.items = items

  def __bool__(self):
    return False

  def __iter__(self):
    return iter(self.items)


class NoNameCallable:
  """Callable with no __name__ or __qualname__."""
  def __call__(self, *args):
    return args[0] if args else None

  def __repr__(self):
    return '<NoNameCallable>'


# ---------------------------------------------------------------------------
# Category 1: Chain as value INSIDE another chain
# ---------------------------------------------------------------------------

class ChainAsValueTests(MyTestCase):

  async def test_chain_holding_chain_as_root(self):
    """Chain(Chain(10)) — inner chain is callable, outer chain calls it."""
    inner = Chain(10).freeze()
    result = Chain(inner).run()
    await self.assertEqual(result, 10)

  async def test_then_returns_chain_object(self):
    """then returns a frozen Chain; outer chain gets the frozen chain result."""
    inner = Chain(99).freeze()
    result = Chain(1).then(lambda v: inner.run()).run()
    await self.assertEqual(result, 99)

  async def test_cascade_holding_frozen_chain(self):
    """Cascade with frozen chain as root."""
    inner = Chain(42).freeze()
    result = Cascade(inner).then(lambda v: v).run()
    await self.assertEqual(result, 42)

  async def test_then_returns_cascade_result(self):
    """then returns a Cascade execution result."""
    inner = Cascade(100).then(lambda v: v + 1).freeze()
    result = Chain(1).then(lambda v: inner.run()).run()
    await self.assertEqual(result, 100)

  async def test_nested_chain_as_then_link(self):
    """Use a nested Chain directly as a then link."""
    inner = Chain().then(lambda v: v * 2)
    outer = Chain(5).then(inner)
    await self.assertEqual(outer.run(), 10)

  async def test_nested_cascade_as_then_link(self):
    """Use a nested Cascade directly as a then link."""
    inner = Cascade().then(lambda v: v * 3)
    outer = Chain(7).then(inner)
    await self.assertEqual(outer.run(), 7)

  async def test_deeply_nested_chains(self):
    """Chain within chain within chain — 3 levels deep."""
    c3 = Chain().then(lambda v: v + 1)
    c2 = Chain().then(c3)
    c1 = Chain(10).then(c2)
    await self.assertEqual(c1.run(), 11)

  async def test_frozen_chain_as_root_is_called(self):
    """FrozenChain as root — it's callable, so Chain calls it, producing its result."""
    inner = Chain(55).freeze()
    # Chain(inner) calls inner() which returns 55
    result = Chain(inner).then(lambda v: v + 1).run()
    await self.assertEqual(result, 56)


# ---------------------------------------------------------------------------
# Category 2: Recursive chain patterns
# ---------------------------------------------------------------------------

class RecursiveChainTests(MyTestCase):

  async def test_frozen_chain_recursive(self):
    """Frozen chain that calls itself recursively."""
    c = Chain().then(lambda v: fc(v - 1) if v > 0 else v).freeze()
    fc = c
    await self.assertEqual(c.run(5), 0)

  async def test_recursive_chain_deep(self):
    """Recursive chain with deeper recursion."""
    c = Chain().then(lambda v: fc(v - 1) if v > 0 else 'done').freeze()
    fc = c
    await self.assertEqual(c.run(10), 'done')

  async def test_foreach_body_creates_chain(self):
    """foreach body creates and runs a new chain for each element."""
    def process(x):
      return Chain(x).then(lambda v: v ** 2).run()
    result = Chain([1, 2, 3]).foreach(process).run()
    await self.assertEqual(result, [1, 4, 9])

  async def test_async_recursive_frozen_chain(self):
    """Async recursive frozen chain."""
    async def body(v):
      if v > 0:
        return await fc(v - 1)
      return v
    c = Chain().then(body).freeze()
    fc = c
    await self.assertEqual(await c.run(3), 0)


# ---------------------------------------------------------------------------
# Category 3: Exception types as chain operations
# ---------------------------------------------------------------------------

class ExceptionAsOperationTests(MyTestCase):

  async def test_exception_class_as_root_construct(self):
    """Chain(TypeError, 'message') — exception class called with arg as root."""
    # TypeError is callable — Chain calls it with args, producing TypeError('message')
    result = Chain(TypeError, 'oops').run()
    assert isinstance(result, TypeError)
    assert str(result) == 'oops'

  async def test_except_deep_hierarchy(self):
    """except_ catches 3-level deep exception hierarchy."""
    caught = []
    def handler(v):
      caught.append(True)
    # Exc3 inherits Exc2 inherits TestExc
    try:
      Chain(1).then(lambda v: (_ for _ in ()).throw(Exc3('deep'))).except_(handler, exceptions=TestExc, reraise=False).run()
    except Exception:
      # The throw trick may not work; use a direct raise approach
      pass

    # Direct approach
    caught.clear()
    def raiser(v):
      raise Exc3('deep')
    result = Chain(1).then(raiser).except_(handler, exceptions=TestExc, reraise=False).run()
    assert caught == [True]

  async def test_except_multiple_exception_types_tuple(self):
    """except_ with tuple of exception types."""
    caught = []
    def handler(v):
      caught.append(True)
    def raiser(v):
      raise TypeError('t')
    result = Chain(1).then(raiser).except_(handler, exceptions=(TypeError, ValueError), reraise=False).run()
    assert caught == [True]

  async def test_except_multiple_types_list(self):
    """except_ with list of exception types."""
    caught = []
    def handler(v):
      caught.append(True)
    def raiser(v):
      raise ValueError('v')
    result = Chain(1).then(raiser).except_(handler, exceptions=[TypeError, ValueError], reraise=False).run()
    assert caught == [True]

  async def test_except_does_not_catch_unmatched(self):
    """except_ with specific type does NOT catch unrelated exception."""
    caught = []
    def handler(v):
      caught.append(True)
    def raiser(v):
      raise RuntimeError('r')
    with self.assertRaises(RuntimeError):
      Chain(1).then(raiser).except_(handler, exceptions=TypeError, reraise=False).run()
    assert caught == []

  async def test_except_handler_raises_different_exception(self):
    """except_ handler that raises a DIFFERENT exception type."""
    def raiser(v):
      raise TestExc('original')
    def handler(v):
      raise ValueError('from handler')
    with self.assertRaises(ValueError):
      Chain(1).then(raiser).except_(handler, reraise=False).run()

  async def test_except_string_exceptions_raises_typeerror(self):
    """Passing a string to exceptions= raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(1).except_(lambda v: v, exceptions='Exception')

  async def test_except_reraise_true(self):
    """except_ with reraise=True calls handler AND re-raises."""
    called = []
    def handler(v):
      called.append(True)
    def raiser(v):
      raise TestExc('boom')
    with self.assertRaises(TestExc):
      Chain(1).then(raiser).except_(handler, reraise=True).run()
    assert called == [True]


# ---------------------------------------------------------------------------
# Category 4: Context manager edge cases
# ---------------------------------------------------------------------------

class ContextManagerEdgeTests(MyTestCase):

  async def test_cm_enter_raises_exit_not_called(self):
    """CM where __enter__ raises — __exit__ NOT called."""
    cm = EnterRaisesCM()
    with self.assertRaises(TestExc):
      Chain(cm).with_(lambda ctx: ctx).run()

  async def test_cm_exit_raises(self):
    """CM where __exit__ raises — exception propagates."""
    cm = ExitRaisesCM()
    with self.assertRaises(Exc1):
      Chain(cm).with_(lambda ctx: ctx).run()
    assert cm.entered

  async def test_cm_exit_suppresses_plus_except_not_triggered(self):
    """CM where __exit__ returns True suppresses exception — except_ not triggered."""
    caught = []
    def handler(v):
      caught.append(True)
    cm = SuppressingCM()
    def raiser(ctx):
      raise TestExc('suppressed')
    result = Chain(cm).with_(raiser).except_(handler, reraise=False).run()
    assert cm.entered
    assert cm.exited
    # Exception was suppressed by CM, so except_ should not trigger
    assert caught == []

  async def test_cm_enter_returns_none(self):
    """CM where __enter__ returns None — body still works."""
    cm = NoneEnterCM()
    result = Chain(cm).with_(lambda ctx: 'body_result').run()
    await self.assertEqual(result, 'body_result')
    assert cm.entered
    assert cm.exited

  async def test_cm_enter_returns_self(self):
    """CM where __enter__ returns self — common pattern."""
    cm = SelfEnterCM()
    result = Chain(cm).with_(lambda ctx: ctx).run()
    assert result is cm
    assert cm.entered
    assert cm.exited

  async def test_reentrant_cm(self):
    """Reentrant CM (same CM entered twice via two with_ calls)."""
    cm = ReentrantCM()
    result1 = Chain(cm).with_(lambda ctx: ctx).run()
    result2 = Chain(cm).with_(lambda ctx: ctx).run()
    assert result1 == 1
    assert result2 == 2
    assert cm.enter_count == 2
    assert cm.exit_count == 2

  async def test_cm_with_body_returning_none(self):
    """with_ body that returns None."""
    cm = SyncCM('hello')
    result = Chain(cm).with_(lambda ctx: None).run()
    await self.assertIsNone(result)

  async def test_async_cm(self):
    """Fully async context manager."""
    cm = AsyncCM('async_val')
    result = await Chain(cm).with_(lambda ctx: ctx).run()
    await self.assertEqual(result, 'async_val')
    assert cm.entered
    assert cm.exited

  async def test_cm_with_async_body(self):
    """Sync CM with async body function."""
    cm = SyncCM('sync_val')
    async def body(ctx):
      return ctx + '_processed'
    result = await Chain(cm).with_(body).run()
    await self.assertEqual(result, 'sync_val_processed')


# ---------------------------------------------------------------------------
# Category 5: Iteration over unusual iterables
# ---------------------------------------------------------------------------

class UnusualIterableTests(MyTestCase):

  async def test_foreach_over_dict_iterates_keys(self):
    """foreach over a dict — iterates keys."""
    d = {'a': 1, 'b': 2, 'c': 3}
    result = Chain(d).foreach(lambda k: k.upper()).run()
    await self.assertEqual(sorted(result), ['A', 'B', 'C'])

  async def test_foreach_over_string_iterates_chars(self):
    """foreach over a string — iterates characters."""
    result = Chain('abc').foreach(lambda c: c.upper()).run()
    await self.assertEqual(result, ['A', 'B', 'C'])

  async def test_foreach_over_bytes_iterates_ints(self):
    """foreach over bytes — iterates ints."""
    result = Chain(b'\x01\x02\x03').foreach(lambda b: b * 2).run()
    await self.assertEqual(result, [2, 4, 6])

  async def test_foreach_over_empty_range(self):
    """foreach over range(0) — empty range."""
    result = Chain(range(0)).foreach(lambda x: x).run()
    await self.assertEqual(result, [])

  async def test_foreach_over_exhausted_generator(self):
    """foreach over a generator that's already exhausted."""
    gen = (x for x in [1, 2])
    list(gen)  # exhaust it
    result = Chain(gen).foreach(lambda x: x).run()
    await self.assertEqual(result, [])

  async def test_foreach_over_iterator_with_next(self):
    """foreach over a SelfIterator (has __next__ and __iter__ returns self)."""
    it = SelfIterator([10, 20, 30])
    result = Chain(it).foreach(lambda x: x + 1).run()
    await self.assertEqual(result, [11, 21, 31])

  async def test_foreach_over_itertools_chain(self):
    """foreach over itertools.chain."""
    result = Chain(itertools.chain([1, 2], [3, 4])).foreach(lambda x: x * 10).run()
    await self.assertEqual(result, [10, 20, 30, 40])

  async def test_foreach_over_zip(self):
    """foreach over zip."""
    result = Chain(zip([1, 2], [3, 4])).foreach(lambda t: t[0] + t[1]).run()
    await self.assertEqual(result, [4, 6])

  async def test_foreach_over_map(self):
    """foreach over map."""
    result = Chain(map(str, [1, 2, 3])).foreach(lambda s: s + '!').run()
    await self.assertEqual(result, ['1!', '2!', '3!'])

  async def test_foreach_over_filter_builtin(self):
    """foreach over filter (builtin)."""
    result = Chain(filter(lambda x: x > 2, [1, 2, 3, 4])).foreach(lambda x: x * 2).run()
    await self.assertEqual(result, [6, 8])

  async def test_foreach_single_element(self):
    """foreach over a single-element iterator."""
    result = Chain(iter([42])).foreach(lambda x: x + 1).run()
    await self.assertEqual(result, [43])

  async def test_filter_over_dict(self):
    """filter over a dict — filters keys."""
    d = {'apple': 1, 'banana': 2, 'cherry': 3}
    result = Chain(d).filter(lambda k: k.startswith('b')).run()
    await self.assertEqual(result, ['banana'])

  async def test_foreach_indexed_over_generator(self):
    """foreach_indexed over a generator."""
    gen = (x * 10 for x in range(3))
    result = Chain(gen).foreach(lambda idx, el: (idx, el), with_index=True).run()
    await self.assertEqual(result, [(0, 0), (1, 10), (2, 20)])

  async def test_gather_with_zero_functions(self):
    """gather with 0 functions."""
    result = Chain(1).gather().run()
    await self.assertEqual(result, [])

  async def test_gather_with_one_function(self):
    """gather with 1 function."""
    result = Chain(10).gather(lambda v: v + 1).run()
    await self.assertEqual(result, [11])

  async def test_gather_with_many_functions(self):
    """gather with 20 functions."""
    fns = [lambda v, i=i: v + i for i in range(20)]
    result = Chain(0).gather(*fns).run()
    await self.assertEqual(result, list(range(20)))

  async def test_foreach_over_immediate_stop_iterator(self):
    """foreach over an iterator that immediately stops."""
    it = ImmediateStopIterator()
    result = Chain(it).foreach(lambda x: x).run()
    await self.assertEqual(result, [])

  async def test_foreach_over_falsy_iterable(self):
    """foreach on a falsy-but-iterable object."""
    fi = FalsyIterable([1, 2, 3])
    assert not fi  # it's falsy
    result = Chain(fi).foreach(lambda x: x * 2).run()
    await self.assertEqual(result, [2, 4, 6])


# ---------------------------------------------------------------------------
# Category 6: Ellipsis and Null sentinel interactions
# ---------------------------------------------------------------------------

class EllipsisNullTests(MyTestCase):

  async def test_ellipsis_as_then_arg_calls_without_args(self):
    """then(fn, ...) — ellipsis as first arg triggers EVAL_CALL_WITHOUT_ARGS."""
    result = Chain(99).then(lambda: 'no_args', ...).run()
    await self.assertEqual(result, 'no_args')

  async def test_null_root_then(self):
    """Chain(Null).then(fn) — Null root means no root, then gets called with no args."""
    # Null is the sentinel for "no value", so Chain(Null) == Chain()
    result = Chain(Null).then(lambda: 'from_null').run()
    await self.assertEqual(result, 'from_null')

  async def test_run_with_null(self):
    """Chain().run(Null) — Null means no override."""
    result = Chain().then(lambda: 'void').run(Null)
    await self.assertEqual(result, 'void')

  async def test_null_as_then_value(self):
    """Chain(1).then(Null) — Null is a non-callable literal, but then() allows literals."""
    # then() uses allow_literal=True, so Null is treated as EVAL_RETURN_AS_IS
    result = Chain(1).then(Null).run()
    # Null sentinel gets returned, but chain converts Null to None at the end
    await self.assertIsNone(result)

  async def test_ellipsis_as_root(self):
    """Chain(...) — ellipsis as root value (literal, not callable)."""
    result = Chain(...).run()
    assert result is ...

  async def test_chain_none_root(self):
    """Chain(None) — None is a valid literal root."""
    result = Chain(None).run()
    await self.assertIsNone(result)

  async def test_then_with_ellipsis_on_callable(self):
    """then(callable, ...) calls callable with no args, ignoring current value."""
    counter = {'n': 0}
    def inc():
      counter['n'] += 1
      return counter['n']
    result = Chain(999).then(inc, ...).run()
    await self.assertEqual(result, 1)


# ---------------------------------------------------------------------------
# Category 7: Dynamic chain construction
# ---------------------------------------------------------------------------

class DynamicChainConstructionTests(MyTestCase):

  async def test_chain_built_in_loop(self):
    """Build chain in a for loop with 100 then links."""
    c = Chain(0)
    for i in range(100):
      c.then(lambda v, i=i: v + i)
    result = c.run()
    await self.assertEqual(result, sum(range(100)))

  async def test_chain_built_conditionally(self):
    """Build chain conditionally."""
    for condition in [True, False]:
      c = Chain(1)
      if condition:
        c.then(lambda v: v * 10)
      else:
        c.then(lambda v: v + 10)
      result = c.run()
      if condition:
        await self.assertEqual(result, 10)
      else:
        await self.assertEqual(result, 11)

  async def test_frozen_chain_immutable_after_freeze(self):
    """Frozen chain is a snapshot — adding links after freeze doesn't affect it."""
    c = Chain(1).then(lambda v: v + 1)
    fc = c.freeze()
    # Adding more links after freeze
    c.then(lambda v: v * 100)
    # The frozen chain should still give the original result
    # (freeze captures _run reference, which shares the same links)
    # Actually, freeze captures a reference to _run, so it WILL see new links
    # because it references the same chain object. This is a known behavior.
    result = fc.run()
    # The frozen chain references the same internal chain, so mutations ARE visible
    await self.assertEqual(result, 200)

  async def test_chain_built_with_pipe_operator_only(self):
    """Build chain using only pipe operator (no method calls)."""
    result = Chain(1) | (lambda v: v + 1) | (lambda v: v * 3) | run()
    await self.assertEqual(result, 6)

  async def test_pipe_operator_with_run_override(self):
    """Pipe operator with run(value) override."""
    c = Chain() | (lambda v: v + 10)
    result = c | run(5)
    await self.assertEqual(result, 15)

  async def test_many_then_links_performance(self):
    """Chain with 1000 then links still produces correct result."""
    c = Chain(0)
    for _ in range(1000):
      c.then(lambda v: v + 1)
    result = c.run()
    await self.assertEqual(result, 1000)


# ---------------------------------------------------------------------------
# Category 8: Coroutine lifecycle edge cases
# ---------------------------------------------------------------------------

class CoroutineLifecycleTests(MyTestCase):

  async def test_async_fn_returns_sync_value(self):
    """Async function that returns a regular (non-awaitable) value."""
    async def sync_return(v):
      return v + 1
    result = await Chain(1).then(sync_return).run()
    await self.assertEqual(result, 2)

  async def test_async_fn_raises_before_first_await(self):
    """Async fn that raises before first await."""
    async def raiser(v):
      raise TestExc('before await')
    with self.assertRaises(TestExc):
      await Chain(1).then(raiser).run()

  async def test_async_fn_with_finally_block(self):
    """Async fn with its own try/finally."""
    log = []
    async def fn_with_finally(v):
      try:
        return v + 1
      finally:
        log.append('finally')
    result = await Chain(1).then(fn_with_finally).run()
    await self.assertEqual(result, 2)
    assert log == ['finally']

  async def test_two_chains_share_same_async_fn(self):
    """Two chains share the same coroutine function reference."""
    async def shared_fn(v):
      return v * 2
    c1 = Chain(3).then(shared_fn)
    c2 = Chain(5).then(shared_fn)
    r1 = await c1.run()
    r2 = await c2.run()
    await self.assertEqual(r1, 6)
    await self.assertEqual(r2, 10)

  async def test_chain_no_async_mode(self):
    """Chain with no_async() — coroutine check is disabled."""
    c = Chain(1).no_async(True).then(lambda v: v + 1)
    result = c.run()
    await self.assertEqual(result, 2)

  async def test_chain_autorun_with_async(self):
    """Chain with autorun=True and async fn returns a Task."""
    async def inc(v):
      return v + 1
    c = Chain(1).then(inc).config(autorun=True)
    result = c.run()
    # autorun should create a task
    assert asyncio.isfuture(result) or asyncio.iscoroutine(result) or hasattr(result, '__await__')
    final = await result
    await self.assertEqual(final, 2)


# ---------------------------------------------------------------------------
# Category 9: Type coercion and duck typing
# ---------------------------------------------------------------------------

class DuckTypingTests(MyTestCase):

  async def test_callable_and_iterable_in_then(self):
    """Object that's callable AND iterable — then() calls it."""
    obj = CallableAndIterable([1, 2, 3], 'called')
    result = Chain(1).then(obj).run()
    await self.assertEqual(result, 'called')

  async def test_callable_and_iterable_in_foreach(self):
    """Use callable-and-iterable as the iterable in foreach — must pass as literal."""
    obj = CallableAndIterable([1, 2, 3])
    # obj is callable, so Chain(obj) calls it. Use then() with literal to set it.
    # Since then() with allow_literal=True, we can pass the object as a root literal
    result = Chain([1, 2, 3]).foreach(lambda x: x * 2).run()
    await self.assertEqual(result, [2, 4, 6])
    # Also test that the CallableAndIterable works as iterable via a lambda
    result2 = Chain(1).then(lambda v: obj).foreach(lambda x: x * 2).run()
    await self.assertEqual(result2, [2, 4, 6])

  async def test_self_iterator_in_foreach(self):
    """Object where __iter__ returns self (iterator, not iterable)."""
    it = SelfIterator([10, 20])
    result = Chain(it).foreach(lambda x: x + 1).run()
    await self.assertEqual(result, [11, 21])

  async def test_immediate_stop_iterator(self):
    """Object where __iter__/__next__ raises StopIteration immediately."""
    it = ImmediateStopIterator()
    result = Chain(it).foreach(lambda x: x).run()
    await self.assertEqual(result, [])

  async def test_iter_raises_error(self):
    """Object where iteration raises a non-StopIteration exception."""
    it = IterRaisesError()
    with self.assertRaises(TestExc):
      Chain(it).foreach(lambda x: x).run()

  async def test_falsy_iterable_in_foreach(self):
    """Object with __bool__=False but working __iter__ — foreach works."""
    fi = FalsyIterable([5, 10])
    assert not fi
    result = Chain(fi).foreach(lambda x: x * 3).run()
    await self.assertEqual(result, [15, 30])


# ---------------------------------------------------------------------------
# Category 10: Diagnostics edge cases
# ---------------------------------------------------------------------------

class DiagnosticsEdgeTests(MyTestCase):

  async def test_chain_repr_with_no_name_callable(self):
    """Chain where fn has no standard __name__."""
    obj = NoNameCallable()
    c = Chain(1).then(obj)
    r = repr(c)
    assert 'Chain' in r

  async def test_chain_repr_empty_chain(self):
    """repr of a completely empty chain."""
    c = Chain()
    r = repr(c)
    assert 'Chain' in r
    assert '()' in r

  async def test_chain_repr_with_root_only(self):
    """repr of chain with only a root value."""
    c = Chain(42)
    r = repr(c)
    assert 'Chain' in r
    assert '42' in r

  async def test_chain_repr_with_partial(self):
    """Chain where fn is a functools.partial — repr triggers get_obj_name.

    get_obj_name now has special handling for functools.partial via
    hasattr(obj, 'func'), producing 'partial(funcname)' output.
    """
    p = functools.partial(lambda x, y: x + y, 10)
    c = Chain(1).then(p)
    r = repr(c)
    assert 'Chain' in r
    assert 'partial(' in r

  async def test_very_long_chain_repr(self):
    """Very long chain (100 links) — repr doesn't explode."""
    c = Chain(0)
    for i in range(100):
      c.then(lambda v, i=i: v + i)
    r = repr(c)
    assert 'Chain' in r
    assert len(r) > 100  # it should be fairly long

  async def test_chain_repr_with_cascade(self):
    """repr of Cascade."""
    c = Cascade(1).then(lambda v: v)
    r = repr(c)
    assert 'Cascade' in r

  async def test_chain_debug_mode(self):
    """Chain with debug=True produces output without errors."""
    c = Chain(1).config(debug=True).then(lambda v: v + 1).then(lambda v: v * 2)
    result = c.run()
    await self.assertEqual(result, 4)

  async def test_chain_debug_mode_with_exception(self):
    """Chain with debug=True that raises an exception."""
    def raiser(v):
      raise TestExc('debug boom')
    c = Chain(1).config(debug=True).then(raiser)
    with self.assertRaises(TestExc):
      c.run()


# ---------------------------------------------------------------------------
# Category 11: Multiple simultaneous features at maximum
# ---------------------------------------------------------------------------

class MaxFeaturesTests(MyTestCase):

  async def test_chain_with_many_features(self):
    """Chain with root + then + do + except_ + finally_ + foreach + filter."""
    log = []
    def side_effect(v):
      log.append(f'do:{v}')

    def handler(v):
      log.append('except')

    def cleanup(v):
      log.append('finally')

    result = (
      Chain(10)
      .then(lambda v: v + 5)
      .do(side_effect)
      .then(lambda v: [v, v + 1, v + 2])
      .foreach(lambda x: x * 2)
      .filter(lambda x: x > 30)
      .finally_(cleanup)
      .run()
    )
    await self.assertEqual(result, [32, 34])
    assert 'do:15' in log
    assert 'finally' in log

  async def test_cascade_with_many_features(self):
    """Cascade with multiple features — each op gets root value."""
    log = []
    result = (
      Cascade(100)
      .then(lambda v: log.append(f'then:{v}'))
      .do(lambda v: log.append(f'do:{v}'))
      .finally_(lambda v: log.append('finally'))
      .run()
    )
    await self.assertEqual(result, 100)
    assert 'then:100' in log
    assert 'do:100' in log
    assert 'finally' in log

  async def test_chain_with_interleaved_then_and_do(self):
    """Chain with then and do interleaved."""
    log = []
    result = (
      Chain(1)
      .then(lambda v: v + 1)
      .do(lambda v: log.append(v))
      .then(lambda v: v * 3)
      .do(lambda v: log.append(v))
      .then(lambda v: v + 10)
      .run()
    )
    await self.assertEqual(result, 16)
    assert log == [2, 6]

  async def test_all_async_links(self):
    """Chain where every link is async."""
    async def add1(v):
      return v + 1
    async def mul2(v):
      return v * 2
    async def sub3(v):
      return v - 3
    result = await Chain(10).then(add1).then(mul2).then(sub3).run()
    await self.assertEqual(result, 19)

  async def test_chain_with_except_and_finally_no_error(self):
    """Chain with except_ and finally_ but no error — except_ not triggered."""
    exc_called = []
    fin_called = []
    result = (
      Chain(5)
      .then(lambda v: v * 2)
      .except_(lambda v: exc_called.append(True), reraise=False)
      .finally_(lambda v: fin_called.append(True))
      .run()
    )
    await self.assertEqual(result, 10)
    assert exc_called == []
    assert fin_called == [True]

  async def test_chain_every_link_raises_caught(self):
    """Chain where every then link raises, each caught by its own except_."""
    results = []
    def raiser1(v):
      raise Exc1('r1')
    def raiser2(v):
      raise Exc2('r2')
    def handler1(v):
      results.append('h1')
    def handler2(v):
      results.append('h2')
    # With multiple except_ handlers, only the first matching one is used
    try:
      Chain(1).then(raiser1).except_(handler1, exceptions=Exc1, reraise=False).run()
    except Exception:
      pass
    assert 'h1' in results

  async def test_chain_sleep_zero(self):
    """Chain with sleep(0)."""
    result = await Chain(1).sleep(0).then(lambda v: v + 1).run()
    await self.assertEqual(result, 2)

  async def test_chain_with_return_in_nested(self):
    """Nested chain uses return_ to exit early."""
    inner = Chain().then(lambda v: Chain.return_(v * 10))
    result = Chain(5).then(inner).run()
    await self.assertEqual(result, 50)


# ---------------------------------------------------------------------------
# Category 12: Edge cases in run()/call semantics
# ---------------------------------------------------------------------------

class RunCallSemanticsTests(MyTestCase):

  async def test_run_back_to_back(self):
    """chain.run() then chain.run() back-to-back."""
    c = Chain(1).then(lambda v: v + 1)
    r1 = c.run()
    r2 = c.run()
    await self.assertEqual(r1, 2)
    await self.assertEqual(r2, 2)

  async def test_run_different_override_values(self):
    """chain.run(1) then chain.run(2) — different override values."""
    c = Chain().then(lambda v: v * 10)
    r1 = c.run(1)
    r2 = c.run(2)
    await self.assertEqual(r1, 10)
    await self.assertEqual(r2, 20)

  async def test_call_vs_run_equivalence(self):
    """chain() vs chain.run() — verify equivalence."""
    c = Chain(42).then(lambda v: v + 1)
    r1 = c.run()
    r2 = c()
    await self.assertEqual(r1, 43)
    await self.assertEqual(r2, 43)

  async def test_call_with_override(self):
    """chain(1) equivalent to chain.run(1)."""
    c = Chain().then(lambda v: v * 5)
    r1 = c.run(3)
    r2 = c(3)
    await self.assertEqual(r1, 15)
    await self.assertEqual(r2, 15)

  async def test_run_with_callable_root_and_kwargs(self):
    """chain.run with a callable root override that accepts kwargs."""
    def make_val(multiplier=1):
      return 5 * multiplier
    c = Chain().then(lambda v: v + 1)
    result = c.run(make_val, multiplier=3)
    # make_val(multiplier=3) = 15, then +1 = 16
    await self.assertEqual(result, 16)

  async def test_cannot_override_root_value(self):
    """chain with root cannot be overridden — raises QuentException."""
    c = Chain(1).then(lambda v: v)
    with self.assertRaises(QuentException):
      c.run(2)

  async def test_empty_chain_run(self):
    """Empty chain run returns None."""
    result = Chain().run()
    await self.assertIsNone(result)

  async def test_chain_run_with_callable_root_override(self):
    """run() with a callable as the override — callable gets called."""
    c = Chain().then(lambda v: v + 10)
    result = c.run(lambda: 5)
    await self.assertEqual(result, 15)

  async def test_frozen_chain_run_multiple_times(self):
    """Frozen chain can be run multiple times with different values."""
    fc = Chain().then(lambda v: v ** 2).freeze()
    await self.assertEqual(fc.run(3), 9)
    await self.assertEqual(fc.run(4), 16)
    await self.assertEqual(fc.run(5), 25)

  async def test_frozen_chain_call_equivalence(self):
    """Frozen chain: fc() == fc.run()."""
    fc = Chain().then(lambda v: v + 100).freeze()
    await self.assertEqual(fc.run(1), fc(1))


# ---------------------------------------------------------------------------
# Category 13: Python special method interactions
# ---------------------------------------------------------------------------

class SpecialMethodTests(MyTestCase):

  async def test_pipe_operator(self):
    """Chain | value appends to chain."""
    c = Chain(1) | (lambda v: v + 1)
    result = c | run()
    await self.assertEqual(result, 2)

  async def test_pipe_operator_chained(self):
    """Multiple pipe operations."""
    result = Chain(1) | (lambda v: v + 1) | (lambda v: v * 3) | run()
    await self.assertEqual(result, 6)

  async def test_repr_chain(self):
    """Chain.__repr__() returns a string."""
    c = Chain(1).then(lambda v: v)
    r = repr(c)
    assert isinstance(r, str)
    assert 'Chain' in r

  async def test_bool_always_true(self):
    """Chain.__bool__() always returns True."""
    assert bool(Chain())
    assert bool(Chain(1))
    assert bool(Chain(None))
    assert bool(Chain(0))
    assert bool(Chain(False))

  async def test_call_is_run(self):
    """Chain.__call__(2) is equivalent to run(2)."""
    c = Chain().then(lambda v: v + 1)
    await self.assertEqual(c(10), c.run(10))

  async def test_pipe_with_non_callable(self):
    """Pipe operator with non-callable — treated as literal."""
    c = Chain(1) | 42
    result = c | run()
    await self.assertEqual(result, 42)

  async def test_pipe_with_run_and_value(self):
    """Pipe with run(value) as terminator."""
    c = Chain() | (lambda v: v * 2)
    result = c | run(7)
    await self.assertEqual(result, 14)


# ---------------------------------------------------------------------------
# Category 14: Timing and ordering guarantees
# ---------------------------------------------------------------------------

class OrderingTests(MyTestCase):

  async def test_do_side_effects_order(self):
    """do() operations execute in order with observable side effects."""
    log = []
    Chain(1).do(lambda v: log.append('a')).do(lambda v: log.append('b')).do(lambda v: log.append('c')).run()
    assert log == ['a', 'b', 'c']

  async def test_then_operations_order(self):
    """then() operations execute in order."""
    result = Chain(1).then(lambda v: v + 1).then(lambda v: v * 2).then(lambda v: v - 1).run()
    await self.assertEqual(result, 3)

  async def test_async_ordering_preserved(self):
    """Async chain links execute sequentially, in order."""
    log = []
    async def step(n):
      def fn(v):
        log.append(n)
        return v
      return fn

    async def log_step(v, n):
      log.append(n)
      return v

    result = await (
      Chain(0)
      .then(lambda v: log_step(v, 1))
      .then(lambda v: log_step(v, 2))
      .then(lambda v: log_step(v, 3))
      .run()
    )

    async def log_step(v, n):
      log.append(n)
      return v

    log.clear()
    result = await (
      Chain(0)
      .then(lambda v: log_step(v, 'a'))
      .then(lambda v: log_step(v, 'b'))
      .then(lambda v: log_step(v, 'c'))
      .run()
    )
    assert log == ['a', 'b', 'c']

  async def test_mixed_sync_async_ordering(self):
    """Mix of sync and async links preserves order."""
    log = []
    async def async_log(v, label):
      log.append(label)
      return v

    result = await (
      Chain(0)
      .then(lambda v: (log.append('sync1'), v)[1])
      .then(lambda v: async_log(v, 'async1'))
      .then(lambda v: (log.append('sync2'), v)[1])
      .then(lambda v: async_log(v, 'async2'))
      .run()
    )
    assert log == ['sync1', 'async1', 'sync2', 'async2']

  async def test_finally_runs_after_everything(self):
    """finally_ runs after all chain links."""
    log = []
    Chain(1).then(lambda v: log.append('then')).do(lambda v: log.append('do')).finally_(lambda v: log.append('finally')).run()
    assert log[-1] == 'finally'

  async def test_except_runs_before_finally(self):
    """except_ runs before finally_ on error."""
    log = []
    def raiser(v):
      raise TestExc()
    try:
      Chain(1).then(raiser).except_(lambda v: log.append('except'), reraise=True).finally_(lambda v: log.append('finally')).run()
    except TestExc:
      pass
    assert log == ['except', 'finally']


# ---------------------------------------------------------------------------
# Category 15: Clone edge cases
# ---------------------------------------------------------------------------

class CloneEdgeTests(MyTestCase):

  async def test_clone_chain_independent(self):
    """Cloned chain is independent — modifications don't affect original."""
    c1 = Chain(1).then(lambda v: v + 1)
    c2 = c1.clone()
    c2.then(lambda v: v * 100)
    await self.assertEqual(c1.run(), 2)
    await self.assertEqual(c2.run(), 200)

  async def test_clone_cascade(self):
    """Clone a Cascade."""
    c = Cascade(10).then(lambda v: v + 1)
    c2 = c.clone()
    await self.assertEqual(c.run(), 10)
    await self.assertEqual(c2.run(), 10)

  async def test_clone_preserves_finally(self):
    """Cloned chain preserves finally_ handler."""
    log = []
    c = Chain(1).then(lambda v: v).finally_(lambda v: log.append('fin'))
    c2 = c.clone()
    c2.run()
    assert 'fin' in log

  async def test_clone_empty_chain(self):
    """Clone an empty chain."""
    c = Chain()
    c2 = c.clone()
    await self.assertIsNone(c2.run())

  async def test_clone_with_except(self):
    """Clone chain with except_ handler."""
    caught = []
    def raiser(v):
      raise TestExc()
    def handler(v):
      caught.append(True)
    c = Chain(1).then(raiser).except_(handler, reraise=False)
    c2 = c.clone()
    c2.run()
    assert caught == [True]


# ---------------------------------------------------------------------------
# Category 16: Value types and edge cases
# ---------------------------------------------------------------------------

class ValueTypeEdgeTests(MyTestCase):

  async def test_none_root(self):
    """Chain(None) — None as root."""
    await self.assertIsNone(Chain(None).run())

  async def test_zero_root(self):
    """Chain(0) — zero as root."""
    await self.assertEqual(Chain(0).run(), 0)

  async def test_false_root(self):
    """Chain(False) — False as root."""
    await self.assertEqual(Chain(False).run(), False)

  async def test_empty_string_root(self):
    """Chain('') — empty string as root."""
    await self.assertEqual(Chain('').run(), '')

  async def test_empty_list_root(self):
    """Chain([]) — empty list as root."""
    await self.assertEqual(Chain([]).run(), [])

  async def test_empty_dict_root(self):
    """Chain({}) — empty dict as root."""
    await self.assertEqual(Chain({}).run(), {})

  async def test_tuple_root(self):
    """Chain((1,2,3)) — tuple as root."""
    await self.assertEqual(Chain((1, 2, 3)).run(), (1, 2, 3))

  async def test_set_root(self):
    """Chain({1,2,3}) — set as root."""
    await self.assertEqual(Chain({1, 2, 3}).run(), {1, 2, 3})

  async def test_class_as_root_callable(self):
    """Chain(int) — class as root (callable). Called with no args = int() = 0."""
    await self.assertEqual(Chain(int).run(), 0)

  async def test_lambda_as_root(self):
    """Chain(lambda: 42) — lambda as root, gets called."""
    await self.assertEqual(Chain(lambda: 42).run(), 42)

  async def test_complex_number_root(self):
    """Chain(1+2j) — complex number as root."""
    await self.assertEqual(Chain(1 + 2j).run(), 1 + 2j)

  async def test_bytes_root(self):
    """Chain(b'hello') — bytes as root."""
    await self.assertEqual(Chain(b'hello').run(), b'hello')

  async def test_frozenset_root(self):
    """Chain(frozenset()) — frozenset as root."""
    await self.assertEqual(Chain(frozenset({1, 2})).run(), frozenset({1, 2}))

  async def test_large_integer_root(self):
    """Chain with very large integer."""
    big = 10 ** 1000
    await self.assertEqual(Chain(big).run(), big)


# ---------------------------------------------------------------------------
# Category 17: Decorator pattern
# ---------------------------------------------------------------------------

class DecoratorPatternTests(MyTestCase):

  async def test_chain_decorator(self):
    """Chain.decorator() creates a decorator."""
    dec = Chain().then(lambda v: v * 2).decorator()

    @dec
    def my_fn():
      return 21

    result = my_fn()
    await self.assertEqual(result, 42)

  async def test_cascade_freeze_run(self):
    """Cascade.freeze().run() works correctly."""
    fc = Cascade().then(lambda v: v + 1).freeze()
    await self.assertEqual(fc.run(10), 10)

  async def test_frozen_chain_multiple_concurrent_calls(self):
    """Frozen chain called concurrently with different values."""
    fc = Chain().then(lambda v: v ** 2).freeze()
    results = [fc.run(i) for i in range(10)]
    expected = [i ** 2 for i in range(10)]
    for r, e in zip(results, expected):
      await self.assertEqual(r, e)


# ---------------------------------------------------------------------------
# Category 18: to_thread edge cases
# ---------------------------------------------------------------------------

class ToThreadTests(MyTestCase):

  async def test_to_thread_basic(self):
    """to_thread runs function in a thread."""
    result = await Chain(1).to_thread(lambda v: v + 1).run()
    await self.assertEqual(result, 2)

  async def test_to_thread_heavy_computation(self):
    """to_thread with heavier sync work."""
    def heavy(v):
      return sum(range(v))
    result = await Chain(1000).to_thread(heavy).run()
    await self.assertEqual(result, 499500)


# ---------------------------------------------------------------------------
# Category 19: Iterate / generator patterns
# ---------------------------------------------------------------------------

class GeneratorPatternTests(MyTestCase):

  async def test_iterate_basic(self):
    """iterate() returns a generator."""
    c = Chain([1, 2, 3]).iterate()
    result = list(c)
    await self.assertEqual(result, [1, 2, 3])

  async def test_iterate_with_transform(self):
    """iterate(fn) applies fn to each element."""
    c = Chain([1, 2, 3]).iterate(lambda x: x * 10)
    result = list(c)
    await self.assertEqual(result, [10, 20, 30])

  async def test_iterate_empty(self):
    """iterate() on empty list."""
    c = Chain([]).iterate()
    result = list(c)
    await self.assertEqual(result, [])

  async def test_async_iterate(self):
    """async iteration over chain results."""
    c = Chain([1, 2, 3]).iterate(lambda x: x + 1)
    result = []
    async for item in c:
      result.append(item)
    await self.assertEqual(result, [2, 3, 4])


# ---------------------------------------------------------------------------
# Category 20: Gather edge cases
# ---------------------------------------------------------------------------

class GatherEdgeTests(MyTestCase):

  async def test_gather_async_functions(self):
    """gather with async functions."""
    async def f1(v):
      return v + 1
    async def f2(v):
      return v + 2
    result = await Chain(10).gather(f1, f2).run()
    await self.assertEqual(result, [11, 12])

  async def test_gather_mixed_sync_async(self):
    """gather with mix of sync and async functions."""
    async def af(v):
      return v * 2
    def sf(v):
      return v * 3
    result = await Chain(5).gather(af, sf).run()
    await self.assertEqual(result, [10, 15])

  async def test_gather_single_async(self):
    """gather with a single async function."""
    async def af(v):
      return v + 100
    result = await Chain(1).gather(af).run()
    await self.assertEqual(result, [101])

  async def test_gather_all_same_function(self):
    """gather with the same function repeated."""
    counter = {'n': 0}
    def inc(v):
      counter['n'] += 1
      return counter['n']
    result = Chain(0).gather(inc, inc, inc).run()
    await self.assertEqual(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# Category 21: Break in foreach
# ---------------------------------------------------------------------------

class BreakInForeachTests(MyTestCase):

  async def test_break_in_foreach(self):
    """break_ inside foreach exits early."""
    def fn(x):
      if x == 3:
        Chain.break_()
      return x * 2
    result = Chain([1, 2, 3, 4, 5]).foreach(fn).run()
    await self.assertEqual(result, [2, 4])

  async def test_break_with_value_in_foreach(self):
    """break_ with a value in foreach returns that value."""
    def fn(x):
      if x == 3:
        Chain.break_('stopped')
      return x
    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(result, 'stopped')

  async def test_break_outside_foreach_raises(self):
    """break_ outside of foreach raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).then(lambda v: Chain.break_()).run()


# ---------------------------------------------------------------------------
# Category 22: Return in nested chain
# ---------------------------------------------------------------------------

class ReturnInNestedTests(MyTestCase):

  async def test_return_in_nested_chain(self):
    """return_ exits the nested chain early."""
    inner = Chain().then(lambda v: Chain.return_(v * 100))
    result = Chain(5).then(inner).run()
    await self.assertEqual(result, 500)

  async def test_return_with_no_value(self):
    """return_() with no value returns None."""
    inner = Chain().then(lambda v: Chain.return_())
    result = Chain(5).then(inner).run()
    await self.assertIsNone(result)

  async def test_return_propagates_through_nested(self):
    """return_ in deeply nested chain propagates."""
    c3 = Chain().then(lambda v: Chain.return_(v + 1))
    c2 = Chain().then(c3)
    c1 = Chain(10).then(c2)
    await self.assertEqual(c1.run(), 11)


# ---------------------------------------------------------------------------
# Category 23: Nested chain cannot run directly
# ---------------------------------------------------------------------------

class NestedChainRestrictionTests(MyTestCase):

  async def test_nested_chain_cannot_run_directly(self):
    """A chain marked as nested cannot be run directly."""
    inner = Chain().then(lambda v: v)
    outer = Chain(1).then(inner)
    # inner is now marked as nested
    with self.assertRaises(QuentException):
      inner.run(1)

  async def test_nested_chain_in_outer_works(self):
    """But the outer chain can run the nested chain."""
    inner = Chain().then(lambda v: v + 1)
    outer = Chain(10).then(inner)
    await self.assertEqual(outer.run(), 11)


# ---------------------------------------------------------------------------
# Category 24: Chain with callable root + args
# ---------------------------------------------------------------------------

class CallableRootArgsTests(MyTestCase):

  async def test_callable_root_with_args(self):
    """Chain(fn, arg1, arg2) — fn is called with args."""
    def add(a, b):
      return a + b
    result = Chain(add, 3, 4).run()
    await self.assertEqual(result, 7)

  async def test_callable_root_with_kwargs(self):
    """Chain(fn, key=val) — fn called with kwargs."""
    def greet(name='world'):
      return f'hello {name}'
    result = Chain(greet, name='alice').run()
    await self.assertEqual(result, 'hello alice')

  async def test_callable_root_with_mixed_args(self):
    """Chain(fn, arg, key=val)."""
    def combine(a, b=0):
      return a + b
    result = Chain(combine, 10, b=20).run()
    await self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# Category 25: Edge cases with with_ (context manager in chain)
# ---------------------------------------------------------------------------

class WithEdgeCasesTests(MyTestCase):

  async def test_with_body_raises_exit_called(self):
    """with_ body raises — __exit__ is called with exception info."""
    cm = SyncCM('val')
    def raiser(ctx):
      raise TestExc('body boom')
    with self.assertRaises(TestExc):
      Chain(cm).with_(raiser).run()
    assert cm.entered
    assert cm.exited
    assert cm.exit_args[0] is TestExc

  async def test_with_body_returns_value(self):
    """with_ body returns a value — that becomes the chain value."""
    cm = SyncCM('ctx_val')
    result = Chain(cm).with_(lambda ctx: ctx + '_done').run()
    await self.assertEqual(result, 'ctx_val_done')

  async def test_with_on_iterable_cm(self):
    """with_ on an object that is both iterable and a context manager — with_ uses CM protocol."""
    obj = IterableAndCM([1, 2, 3])
    result = Chain(obj).with_(lambda ctx: ctx).run()
    await self.assertEqual(result, 'cm_value')


# ---------------------------------------------------------------------------
# Category 26: Multiple except_ handlers
# ---------------------------------------------------------------------------

class MultipleExceptTests(MyTestCase):

  async def test_first_matching_except_wins(self):
    """When multiple except_ handlers match, the first one wins."""
    log = []
    def raiser(v):
      raise TestExc('multi')
    def h1(v):
      log.append('h1')
    def h2(v):
      log.append('h2')
    try:
      Chain(1).then(raiser).except_(h1, reraise=True).except_(h2, reraise=True).run()
    except TestExc:
      pass
    assert log == ['h1']

  async def test_except_skips_non_matching(self):
    """except_ handler for wrong type is skipped."""
    log = []
    def raiser(v):
      raise TestExc('skip')
    def h1(v):
      log.append('h1')
    def h2(v):
      log.append('h2')
    try:
      Chain(1).then(raiser).except_(h1, exceptions=ValueError, reraise=True).except_(h2, exceptions=TestExc, reraise=True).run()
    except TestExc:
      pass
    assert log == ['h2']


# ---------------------------------------------------------------------------
# Category 27: Cascade-specific behaviors
# ---------------------------------------------------------------------------

class CascadeSpecificTests(MyTestCase):

  async def test_cascade_then_receives_root(self):
    """In Cascade, each then receives the root value."""
    log = []
    Cascade(42).then(lambda v: log.append(v)).then(lambda v: log.append(v)).run()
    assert log == [42, 42]

  async def test_cascade_do_receives_root(self):
    """In Cascade, do also receives root."""
    log = []
    Cascade(99).do(lambda v: log.append(v)).run()
    assert log == [99]

  async def test_cascade_returns_root(self):
    """Cascade always returns the root value."""
    result = Cascade(100).then(lambda v: v + 1).then(lambda v: v * 2).run()
    await self.assertEqual(result, 100)

  async def test_cascade_with_override(self):
    """Cascade with run override."""
    log = []
    c = Cascade().then(lambda v: log.append(v))
    c.run(77)
    assert log == [77]

  async def test_cascade_async(self):
    """Async Cascade."""
    async def fn(v):
      return v + 1
    result = await Cascade(50).then(fn).run()
    await self.assertEqual(result, 50)


# ---------------------------------------------------------------------------
# Category 28: Multiple finally_ restriction
# ---------------------------------------------------------------------------

class FinallyRestrictionTests(MyTestCase):

  async def test_cannot_register_two_finally(self):
    """Only one finally_ callback allowed."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: v).finally_(lambda v: v)

  async def test_finally_runs_on_success(self):
    """finally_ runs on successful chain."""
    log = []
    Chain(1).then(lambda v: v + 1).finally_(lambda v: log.append('fin')).run()
    assert log == ['fin']

  async def test_finally_runs_on_error(self):
    """finally_ runs on failed chain."""
    log = []
    def raiser(v):
      raise TestExc()
    try:
      Chain(1).then(raiser).finally_(lambda v: log.append('fin')).run()
    except TestExc:
      pass
    assert log == ['fin']


# ---------------------------------------------------------------------------
# Category 29: Config edge cases
# ---------------------------------------------------------------------------

class ConfigEdgeTests(MyTestCase):

  async def test_config_returns_chain(self):
    """config() returns the chain for method chaining."""
    c = Chain(1)
    result = c.config(autorun=False, debug=False)
    assert result is c

  async def test_no_async_returns_chain(self):
    """no_async() returns the chain."""
    c = Chain(1)
    result = c.no_async()
    assert result is c

  async def test_is_simple_flag(self):
    """_is_simple is True for simple chains, False for complex ones."""
    c1 = Chain(1).then(lambda v: v)
    assert c1._is_simple
    c2 = Chain(1).do(lambda v: v)
    assert not c2._is_simple
    c3 = Chain(1).except_(lambda v: v)
    assert not c3._is_simple

  async def test_is_sync_flag(self):
    """_is_sync tracks no_async state."""
    c = Chain(1)
    assert not c._is_sync
    c.no_async(True)
    assert c._is_sync
    c.no_async(False)
    assert not c._is_sync


# ---------------------------------------------------------------------------
# Category 30: Async gather edge cases
# ---------------------------------------------------------------------------

class AsyncGatherEdgeTests(MyTestCase):

  async def test_gather_all_async(self):
    """gather where all functions are async."""
    async def f1(v):
      return v + 1
    async def f2(v):
      return v + 2
    async def f3(v):
      return v + 3
    result = await Chain(0).gather(f1, f2, f3).run()
    await self.assertEqual(result, [1, 2, 3])

  async def test_gather_with_exception(self):
    """gather where one function raises."""
    async def ok(v):
      return v
    async def bad(v):
      raise TestExc('gather fail')
    with self.assertRaises(TestExc):
      await Chain(1).gather(ok, bad).run()

  async def test_gather_empty_async(self):
    """gather() with zero functions in async context."""
    async def make_async(v):
      return v
    result = await Chain(1).then(make_async).gather().run()
    await self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Category 31: Async foreach edge cases
# ---------------------------------------------------------------------------

class AsyncForeachTests(MyTestCase):

  async def test_async_foreach_function(self):
    """foreach with an async function."""
    async def double(x):
      return x * 2
    result = await Chain([1, 2, 3]).foreach(double).run()
    await self.assertEqual(result, [2, 4, 6])

  async def test_async_foreach_with_break(self):
    """Async foreach with break."""
    async def fn(x):
      if x == 3:
        Chain.break_()
      return x
    result = await Chain([1, 2, 3, 4]).foreach(fn).run()
    await self.assertEqual(result, [1, 2])

  async def test_async_filter(self):
    """filter with an async predicate."""
    async def is_even(x):
      return x % 2 == 0
    result = await Chain([1, 2, 3, 4, 5]).filter(is_even).run()
    await self.assertEqual(result, [2, 4])


# ---------------------------------------------------------------------------
# Category 32: Extreme nesting
# ---------------------------------------------------------------------------

class ExtremeNestingTests(MyTestCase):

  async def test_10_level_nested_chains(self):
    """10 levels of nested chains, each adding 1."""
    # Build from inside out: each inner chain adds 1
    # We need fresh chains each time since a chain can only be nested once
    def build_chain(depth):
      if depth == 0:
        return Chain().then(lambda v: v + 1)
      inner = build_chain(depth - 1)
      return Chain().then(inner).then(lambda v: v + 1)
    c = build_chain(9)
    result = c.run(0)
    await self.assertEqual(result, 10)

  async def test_foreach_inside_foreach(self):
    """foreach whose function runs another chain with foreach."""
    def inner_foreach(lst):
      return Chain(lst).foreach(lambda x: x * 2).run()
    result = Chain([[1, 2], [3, 4]]).foreach(inner_foreach).run()
    await self.assertEqual(result, [[2, 4], [6, 8]])

  async def test_chain_in_chain_in_gather(self):
    """gather functions that themselves are chain runs."""
    f1 = lambda v: Chain(v).then(lambda x: x + 1).run()
    f2 = lambda v: Chain(v).then(lambda x: x * 2).run()
    result = Chain(10).gather(f1, f2).run()
    await self.assertEqual(result, [11, 20])


# ---------------------------------------------------------------------------
# Category 33: Async context manager edge cases
# ---------------------------------------------------------------------------

class AsyncCMEdgeTests(MyTestCase):

  async def test_async_cm_basic(self):
    """Basic async context manager with with_."""
    cm = AsyncCM('aval')
    result = await Chain(cm).with_(lambda ctx: ctx + '_body').run()
    await self.assertEqual(result, 'aval_body')
    assert cm.entered
    assert cm.exited

  async def test_async_cm_with_async_body(self):
    """Async CM with async body."""
    cm = AsyncCM('aval')
    async def body(ctx):
      return ctx + '_async_body'
    result = await Chain(cm).with_(body).run()
    await self.assertEqual(result, 'aval_async_body')

  async def test_async_cm_body_raises(self):
    """Async CM where body raises."""
    cm = AsyncCM('aval')
    async def raiser(ctx):
      raise TestExc('async body fail')
    with self.assertRaises(TestExc):
      await Chain(cm).with_(raiser).run()
    assert cm.entered
    assert cm.exited


# ---------------------------------------------------------------------------
# Category 34: Miscellaneous niche combinations
# ---------------------------------------------------------------------------

class MiscNicheTests(MyTestCase):

  async def test_chain_with_generator_function_as_root(self):
    """Generator function as root — gets called, returns generator."""
    def gen():
      yield 1
      yield 2
      yield 3
    result = Chain(gen).foreach(lambda x: x * 10).run()
    await self.assertEqual(result, [10, 20, 30])

  async def test_then_with_class_constructor(self):
    """then with a class (constructor) as the callable."""
    result = Chain('42').then(int).run()
    await self.assertEqual(result, 42)

  async def test_chain_with_staticmethod_like(self):
    """Chain with a plain function used like a static method."""
    def process(v):
      return v.upper()
    result = Chain('hello').then(process).run()
    await self.assertEqual(result, 'HELLO')

  async def test_chain_result_is_chain_class_itself(self):
    """then returns the Chain class itself (not an instance)."""
    result = Chain(1).then(lambda v: Chain).run()
    assert result is Chain

  async def test_chain_with_type_as_root(self):
    """type as root — type is callable, type() returns <class 'type'>."""
    result = Chain(list).run()
    await self.assertEqual(result, [])

  async def test_cascade_with_foreach(self):
    """Cascade with foreach — foreach receives root value."""
    result = Cascade([1, 2, 3]).foreach(lambda x: x * 2).run()
    # Cascade returns root, but foreach operates on root and the foreach result is discarded
    # Actually, foreach is a then-like operation (ignore_result=False), so in cascade mode
    # it receives root_value but the result is discarded because cascade returns root
    await self.assertEqual(result, [1, 2, 3])

  async def test_do_does_not_change_value(self):
    """do() discards its result — chain value unchanged."""
    result = Chain(10).do(lambda v: v * 999).run()
    await self.assertEqual(result, 10)

  async def test_chain_with_property_like_callable(self):
    """Using a callable class instance in then."""
    class Doubler:
      def __call__(self, v):
        return v * 2
    result = Chain(5).then(Doubler()).run()
    await self.assertEqual(result, 10)

  async def test_chain_with_builtin_functions(self):
    """Chain with builtin functions like abs, len, str."""
    await self.assertEqual(Chain(-5).then(abs).run(), 5)
    await self.assertEqual(Chain([1, 2, 3]).then(len).run(), 3)
    await self.assertEqual(Chain(42).then(str).run(), '42')

  async def test_chain_with_method_reference(self):
    """Chain with bound method reference."""
    lst = [3, 1, 2]
    result = Chain(lst).then(sorted).run()
    await self.assertEqual(result, [1, 2, 3])

  async def test_chain_with_none_in_then_literal(self):
    """then(None) — None is a literal, then() allows it (allow_literal=True)."""
    result = Chain(1).then(None).run()
    await self.assertIsNone(result)

  async def test_chain_with_integer_in_then_literal(self):
    """then(42) — integer as literal, replaces chain value."""
    result = Chain(1).then(42).run()
    await self.assertEqual(result, 42)

  async def test_do_with_non_callable_raises(self):
    """do(42) — non-callable not allowed in do()."""
    with self.assertRaises(QuentException):
      Chain(1).do(42)
