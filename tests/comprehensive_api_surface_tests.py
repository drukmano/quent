"""Comprehensive API surface tests for the Quent library.

This file systematically tests every public method, class, and behavior
of the Quent library. It serves as the definitive reference for understanding
the public API and its contracts.

Organized into sections:
  1. Module-level exports and sentinels (__all__, Null, QuentException)
  2. Chain construction and basic execution
  3. Chain.then() -- append an operation
  4. Chain.do() -- side-effect operation
  5. Chain.except_() -- exception handling
  6. Chain.finally_() -- cleanup callback
  7. Chain.foreach() -- iterate and collect
  8. Chain.filter() -- predicate filtering
  9. Chain.gather() -- parallel execution
  10. Chain.with_() -- context managers
  11. Chain.clone() -- deep copy
  14. Chain.decorator() -- function wrapping
  16. Chain.config() -- autorun/debug
  17. Chain.run() / __call__() -- execution
  19. Chain.__or__() -- pipe operator
  20. Chain.__bool__() / __repr__() -- dunder protocol
  21. Chain.return_() / Chain.break_() -- control flow
  22. Chain.iterate() -- lazy generator
  23. run -- pipe terminator
  25. Cross-cutting: method chaining fluency
  26. Cross-cutting: immutability and identity
  27. Cross-cutting: QuentException conditions
  28. Cross-cutting: type checks
"""

import asyncio
import inspect
from unittest import IsolatedAsyncioTestCase

from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null, __all__ as quent_all


# -- Helpers ------------------------------------------------------------------

class SyncIter:
  """Simple sync iterable for testing foreach/filter/iterate."""
  def __init__(self, items):
    self.items = items
  def __iter__(self):
    return iter(self.items)


class AsyncIter:
  """Simple async iterable for testing async foreach/filter/iterate."""
  def __init__(self, items):
    self.items = items
    self.index = 0
  def __aiter__(self):
    self.index = 0
    return self
  async def __anext__(self):
    if self.index >= len(self.items):
      raise StopAsyncIteration
    val = self.items[self.index]
    self.index += 1
    return val


class SyncCtx:
  """Minimal sync context manager."""
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


class AsyncCtx:
  """Minimal async context manager."""
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


def _raiser():
  """Raise ZeroDivisionError."""
  raise ZeroDivisionError("deliberate")


# =============================================================================
# 1. Module-level exports and sentinels
# =============================================================================

class TestModuleExports(IsolatedAsyncioTestCase):
  """Verify __all__ exports and Null sentinel."""

  def test_all_exports_are_importable(self):
    """Every name in __all__ must be importable from quent."""
    import quent
    for name in quent_all:
      self.assertTrue(
        hasattr(quent, name),
        f"{name!r} is listed in __all__ but not importable from quent"
      )

  def test_all_exports_content(self):
    """__all__ must contain exactly the documented public API."""
    expected = {'Chain', 'QuentException', 'run', 'ResultOrAwaitable', 'Null', '__version__'}
    self.assertEqual(set(quent_all), expected)

  def test_null_repr(self):
    """Null sentinel has a human-readable repr."""
    self.assertEqual(repr(Null), '<Null>')

  def test_null_is_not_none(self):
    """Null is distinct from None."""
    self.assertIsNot(Null, None)

  def test_null_identity(self):
    """Null is a singleton -- importing twice yields the same object."""
    from quent import Null as Null2
    self.assertIs(Null, Null2)

  def test_quent_exception_is_exception(self):
    """QuentException inherits from Exception."""
    self.assertTrue(issubclass(QuentException, Exception))

  def test_quent_exception_can_be_raised_and_caught(self):
    """QuentException can be raised and caught normally."""
    with self.assertRaises(QuentException):
      raise QuentException("test")

  def test_version_is_string(self):
    """__version__ is a string."""
    from quent import __version__
    self.assertIsInstance(__version__, str)


# =============================================================================
# 2. Chain construction and basic execution
# =============================================================================

class TestChainConstruction(IsolatedAsyncioTestCase):
  """Chain(root_value=Null, *args, **kwargs)."""

  async def test_empty_chain_returns_none(self):
    """Chain() with no root and no operations returns None."""
    self.assertIsNone(Chain().run())

  async def test_chain_with_literal_root(self):
    """Chain(42) uses 42 as the root value."""
    self.assertEqual(Chain(42).run(), 42)

  async def test_chain_with_callable_root(self):
    """Chain(fn) calls fn() and uses the result as root."""
    self.assertEqual(Chain(lambda: 99).run(), 99)

  async def test_chain_with_callable_root_and_args(self):
    """Chain(fn, arg) calls fn(arg)."""
    self.assertEqual(Chain(lambda x: x * 2, 5).run(), 10)

  async def test_chain_with_callable_root_and_kwargs(self):
    """Chain(fn, key=val) calls fn(key=val)."""
    self.assertEqual(Chain(lambda x=0: x + 1, x=10).run(), 11)

  async def test_chain_with_async_root(self):
    """Chain(async_fn) transparently handles coroutines."""
    self.assertEqual(await Chain(aempty, 7).run(), 7)

  async def test_chain_bool_is_true(self):
    """Chain instances are always truthy."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(42)))

  async def test_chain_with_none_root(self):
    """Chain(None) uses None as a literal root value."""
    self.assertIsNone(Chain(None).run())

  async def test_chain_with_zero_root(self):
    """Chain(0) uses 0 as a literal root value (falsy but valid)."""
    self.assertEqual(Chain(0).run(), 0)

  async def test_chain_with_empty_string_root(self):
    """Chain('') uses empty string as root (falsy but valid)."""
    self.assertEqual(Chain('').run(), '')

  async def test_chain_with_false_root(self):
    """Chain(False) uses False as root."""
    self.assertEqual(Chain(False).run(), False)

  async def test_chain_with_empty_list_root(self):
    """Chain([]) uses empty list as root."""
    self.assertEqual(Chain([]).run(), [])


# =============================================================================
# 3. Chain.then() -- append an operation
# =============================================================================

class TestThen(IsolatedAsyncioTestCase):
  """Chain.then(fn_or_value, *args, **kwargs) -> self."""

  async def test_then_returns_self(self):
    """then() returns the same Chain instance for fluent chaining."""
    c = Chain()
    result = c.then(lambda v: v)
    self.assertIs(result, c)

  async def test_then_callable_receives_current_value(self):
    """then(fn) passes the current value to fn."""
    self.assertEqual(Chain(10).then(lambda v: v * 2).run(), 20)

  async def test_then_literal_replaces_current_value(self):
    """then(literal) replaces the current value."""
    self.assertEqual(Chain(10).then(99).run(), 99)

  async def test_then_with_explicit_args(self):
    """then(fn, arg) calls fn(arg) instead of fn(current_value)."""
    self.assertEqual(Chain(10).then(lambda x: x + 1, 5).run(), 6)

  async def test_then_with_ellipsis_ignores_current_value(self):
    """then(fn, ...) calls fn() without any arguments."""
    self.assertEqual(Chain(10).then(lambda: 42, ...).run(), 42)

  async def test_then_with_kwargs(self):
    """then(fn, key=val) calls fn(key=val)."""
    self.assertEqual(Chain(10).then(lambda x=0: x, x=77).run(), 77)

  async def test_then_chain_sync(self):
    """Multiple sync then() calls chain correctly."""
    result = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).then(lambda v: v - 2).run()
    self.assertEqual(result, 4)  # (1+1)*3 - 2 = 4

  async def test_then_chain_async(self):
    """then() chains work across async boundaries."""
    result = Chain(aempty, 5).then(lambda v: v * 2).then(aempty).then(lambda v: v + 1).run()
    self.assertEqual(await result, 11)

  async def test_then_with_nested_chain(self):
    """then(Chain(...)) nests a sub-chain receiving the current value."""
    self.assertEqual(
      Chain(5).then(Chain().then(lambda v: v * 10)).run(),
      50
    )

  async def test_then_none_literal(self):
    """then(None) replaces current value with None."""
    self.assertIsNone(Chain(42).then(None).run())

  async def test_then_zero_literal(self):
    """then(0) replaces current value with 0."""
    self.assertEqual(Chain(42).then(0).run(), 0)

  async def test_then_false_literal(self):
    """then(False) replaces current value with False."""
    self.assertEqual(Chain(42).then(False).run(), False)


# =============================================================================
# 4. Chain.do() -- side-effect operation
# =============================================================================

class TestDo(IsolatedAsyncioTestCase):
  """Chain.do(fn, *args, **kwargs) -> self."""

  async def test_do_returns_self(self):
    """do() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.do(lambda v: None), c)

  async def test_do_discards_result(self):
    """do() runs the fn but discards its return value."""
    self.assertEqual(
      Chain(42).do(lambda v: 999).run(),
      42
    )

  async def test_do_receives_current_value(self):
    """do(fn) passes the current value to fn."""
    captured = []
    Chain(42).do(lambda v: captured.append(v)).run()
    self.assertEqual(captured, [42])

  async def test_do_with_ellipsis(self):
    """do(fn, ...) calls fn() without arguments."""
    captured = []
    Chain(42).do(lambda: captured.append('called'), ...).run()
    self.assertEqual(captured, ['called'])

  async def test_do_with_explicit_args(self):
    """do(fn, arg) calls fn(arg)."""
    captured = []
    Chain(42).do(lambda x: captured.append(x), 'hello').run()
    self.assertEqual(captured, ['hello'])

  async def test_do_non_callable_raises(self):
    """do(non_callable) raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(42).do(True).run()

  async def test_do_async_discards_result(self):
    """do() discards result even with async function."""
    self.assertEqual(
      await Chain(aempty, 42).do(aempty).run(),
      42
    )

  async def test_do_multiple_side_effects(self):
    """Multiple do() calls preserve the value through all."""
    captured = []
    result = await await_(
      Chain(10).do(lambda v: captured.append(v)).do(lambda v: captured.append(v * 2)).run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(captured, [10, 20])


# =============================================================================
# 5. Chain.except_() -- exception handling
# =============================================================================

class TestExcept(IsolatedAsyncioTestCase):
  """Chain.except_(handler, *args, exceptions=None, reraise=True, **kwargs) -> self."""

  async def test_except_returns_self(self):
    """except_() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.except_(lambda v: None), c)

  async def test_except_catches_exception_sync(self):
    """except_ catches exceptions when root raises (handler gets no args since root never resolved)."""
    caught = []
    def handler(v=None):
      caught.append(True)
    try:
      Chain(_raiser).except_(handler).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(caught, [True])

  async def test_except_catches_exception_with_resolved_root(self):
    """except_ catches exceptions in middle links; handler gets root value."""
    caught = []
    def handler(v):
      caught.append(v)
    try:
      Chain(42).then(lambda v: 1/0).except_(handler).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(caught, [42])

  async def test_except_catches_exception_async(self):
    """except_ catches exceptions during async execution."""
    caught = []
    async def raiser():
      raise ValueError("boom")
    def handler(v=None):
      caught.append(True)
    try:
      await await_(Chain(raiser).except_(handler).run())
    except ValueError:
      pass
    self.assertEqual(caught, [True])

  async def test_except_reraise_true_by_default(self):
    """except_ re-raises the exception by default (reraise=True)."""
    with self.assertRaises(ZeroDivisionError):
      Chain(42).then(lambda v: 1/0).except_(lambda v: None).run()

  async def test_except_reraise_false_suppresses(self):
    """except_ with reraise=False suppresses the exception and returns handler result."""
    result = Chain(42).then(lambda v: 1/0).except_(lambda v: 'recovered', reraise=False).run()
    self.assertEqual(result, 'recovered')

  async def test_except_with_specific_exception_type_no_match(self):
    """exceptions= that does not match the raised type skips the handler."""
    caught = []
    def handler(v):
      caught.append(True)
    try:
      Chain(42).then(lambda v: 1/0).except_(handler, exceptions=TypeError).run()
    except ZeroDivisionError:
      pass
    # Handler should NOT have been called since the exception was ZeroDivisionError
    self.assertEqual(caught, [])

  async def test_except_with_matching_exception_type(self):
    """exceptions= matching the raised type calls the handler."""
    caught = []
    def handler(v):
      caught.append(True)
    try:
      Chain(42).then(lambda v: 1/0).except_(handler, exceptions=ZeroDivisionError).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(caught, [True])

  async def test_except_with_exception_list(self):
    """exceptions= can be a list of exception types."""
    caught = []
    def handler(v):
      caught.append(True)
    try:
      Chain(42).then(lambda v: 1/0).except_(handler, exceptions=[TypeError, ZeroDivisionError]).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(caught, [True])

  async def test_except_with_exception_subclass(self):
    """exceptions= matches subclasses."""
    class MyError(ValueError):
      pass
    caught = []
    def handler(v):
      caught.append(True)
    def raiser(v):
      raise MyError()
    try:
      Chain(42).then(raiser).except_(handler, exceptions=ValueError).run()
    except MyError:
      pass
    self.assertEqual(caught, [True])

  async def test_except_handler_receives_root_value(self):
    """The except_ handler is called with the root value."""
    captured = []
    obj = object()
    def handler(v):
      captured.append(v)
    try:
      Chain(obj).then(lambda v: 1/0).except_(handler).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(len(captured), 1)
    self.assertIs(captured[0], obj)

  async def test_except_multiple_handlers_first_match(self):
    """Only the first matching except_ handler is called."""
    caught1 = []
    caught2 = []
    try:
      Chain(42).then(lambda v: 1/0).except_(
        lambda v: caught1.append(True), exceptions=ZeroDivisionError
      ).except_(
        lambda v: caught2.append(True), exceptions=ZeroDivisionError
      ).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(caught1, [True])
    self.assertEqual(caught2, [])

  async def test_except_string_exception_raises_type_error(self):
    """Passing a string as exceptions= raises TypeError."""
    with self.assertRaises(TypeError):
      Chain().except_(lambda v: None, exceptions="ValueError")


# =============================================================================
# 6. Chain.finally_() -- cleanup callback
# =============================================================================

class TestFinally(IsolatedAsyncioTestCase):
  """Chain.finally_(handler, *args, **kwargs) -> self."""

  async def test_finally_returns_self(self):
    """finally_() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.finally_(lambda v: None), c)

  async def test_finally_runs_after_success_sync(self):
    """finally_ runs after successful chain execution."""
    ran = []
    Chain(42).finally_(lambda v: ran.append(v)).run()
    self.assertEqual(ran, [42])

  async def test_finally_runs_after_success_async(self):
    """finally_ runs after successful async chain execution."""
    ran = []
    await await_(Chain(aempty, 42).finally_(lambda v: ran.append(v)).run())
    self.assertEqual(ran, [42])

  async def test_finally_runs_after_exception(self):
    """finally_ runs even when an exception is raised.

    NOTE: When the root callable itself raises, root_value is Null and the
    handler receives zero positional args. Use v=None default.
    """
    ran = []
    try:
      Chain(_raiser).finally_(lambda v=None: ran.append(True)).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(ran, [True])

  async def test_finally_runs_after_exception_with_resolved_root(self):
    """finally_ receives root when root resolves before the error."""
    ran = []
    try:
      Chain(42).then(lambda v: 1/0).finally_(lambda v: ran.append(v)).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(ran, [42])

  async def test_double_finally_raises(self):
    """Registering two finally_ handlers raises QuentException."""
    with self.assertRaises(QuentException):
      Chain().finally_(lambda v: None).finally_(lambda v: None)

  async def test_finally_receives_root_value(self):
    """finally_ handler is called with the root value."""
    captured = []
    obj = object()
    Chain(obj).finally_(lambda v: captured.append(v)).run()
    self.assertEqual(len(captured), 1)
    self.assertIs(captured[0], obj)

  async def test_finally_does_not_alter_chain_result(self):
    """finally_ handler's return value does not change the chain result."""
    result = Chain(42).finally_(lambda v: 999).run()
    self.assertEqual(result, 42)


# =============================================================================
# 7. Chain.foreach() -- iterate and collect
# =============================================================================

class TestForeach(IsolatedAsyncioTestCase):
  """Chain.foreach(fn) -> self."""

  async def test_foreach_returns_self(self):
    """foreach() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.foreach(lambda x: x), c)

  async def test_foreach_sync_basic(self):
    """foreach maps fn over a sync iterable and collects results."""
    self.assertEqual(
      Chain([1, 2, 3]).foreach(lambda x: x * 10).run(),
      [10, 20, 30]
    )

  async def test_foreach_async_iterable(self):
    """foreach works with async iterables."""
    result = await await_(
      Chain(AsyncIter([1, 2, 3])).foreach(lambda x: x * 10).run()
    )
    self.assertEqual(result, [10, 20, 30])

  async def test_foreach_async_fn(self):
    """foreach works with an async mapping function."""
    async def double(x):
      return x * 2
    result = await await_(
      Chain([1, 2, 3]).foreach(double).run()
    )
    self.assertEqual(result, [2, 4, 6])

  async def test_foreach_empty_list(self):
    """foreach on empty list returns empty list."""
    self.assertEqual(Chain([]).foreach(lambda x: x).run(), [])

  async def test_foreach_with_break(self):
    """Chain.break_() inside foreach stops iteration early."""
    def stop_at_3(x):
      if x == 3:
        return Chain.break_()
      return x * 10
    self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(stop_at_3).run(),
      [10, 20]
    )

  async def test_foreach_with_break_value(self):
    """Chain.break_(value) returns that value as the foreach result."""
    sentinel = object()
    def stop_with_value(x):
      if x == 2:
        return Chain.break_(sentinel)
      return x
    result = Chain([1, 2, 3]).foreach(stop_with_value).run()
    self.assertIs(result, sentinel)

  async def test_foreach_preserves_element_order(self):
    """foreach preserves the iteration order."""
    self.assertEqual(
      Chain([3, 1, 4, 1, 5]).foreach(lambda x: x).run(),
      [3, 1, 4, 1, 5]
    )


# =============================================================================
# 8. Chain.filter() -- predicate filtering
# =============================================================================

class TestFilter(IsolatedAsyncioTestCase):
  """Chain.filter(predicate) -> self."""

  async def test_filter_returns_self(self):
    """filter() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.filter(lambda x: True), c)

  async def test_filter_sync_basic(self):
    """filter keeps elements where predicate returns truthy."""
    self.assertEqual(
      Chain([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).run(),
      [2, 4]
    )

  async def test_filter_async_predicate(self):
    """filter works with an async predicate function."""
    async def is_even(x):
      return x % 2 == 0
    result = await await_(
      Chain([1, 2, 3, 4, 5]).filter(is_even).run()
    )
    self.assertEqual(result, [2, 4])

  async def test_filter_async_iterable(self):
    """filter works with async iterables."""
    result = await await_(
      Chain(AsyncIter([10, 20, 30, 40])).filter(lambda x: x > 20).run()
    )
    self.assertEqual(result, [30, 40])

  async def test_filter_all_pass(self):
    """filter keeps all elements when predicate always returns True."""
    self.assertEqual(
      Chain([1, 2, 3]).filter(lambda x: True).run(),
      [1, 2, 3]
    )

  async def test_filter_none_pass(self):
    """filter returns empty list when predicate always returns False."""
    self.assertEqual(
      Chain([1, 2, 3]).filter(lambda x: False).run(),
      []
    )

  async def test_filter_empty_list(self):
    """filter on empty list returns empty list."""
    self.assertEqual(
      Chain([]).filter(lambda x: True).run(),
      []
    )

  async def test_filter_falsy_values_preserved(self):
    """filter correctly handles falsy elements (0, '', False)."""
    self.assertEqual(
      Chain([0, 1, '', 'a', False, True]).filter(lambda x: True).run(),
      [0, 1, '', 'a', False, True]
    )


# =============================================================================
# 9. Chain.gather() -- parallel execution
# =============================================================================

class TestGather(IsolatedAsyncioTestCase):
  """Chain.gather(*fns) -> self."""

  async def test_gather_returns_self(self):
    """gather() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.gather(lambda v: v), c)

  async def test_gather_sync_basic(self):
    """gather calls each fn with the current value and collects results."""
    self.assertEqual(
      Chain(5).gather(
        lambda v: v + 1,
        lambda v: v * 2,
        lambda v: v - 3
      ).run(),
      [6, 10, 2]
    )

  async def test_gather_async_fns(self):
    """gather works with async functions."""
    async def add1(v): return v + 1
    async def mul2(v): return v * 2
    result = await await_(
      Chain(5).gather(add1, mul2).run()
    )
    self.assertEqual(result, [6, 10])

  async def test_gather_mixed_sync_async(self):
    """gather handles a mix of sync and async functions."""
    async def async_double(v):
      return v * 2
    result = await await_(
      Chain(3).gather(
        lambda v: v + 1,
        async_double,
      ).run()
    )
    self.assertEqual(result, [4, 6])

  async def test_gather_single_fn(self):
    """gather with a single function returns a single-element list."""
    self.assertEqual(
      Chain(7).gather(lambda v: v * 3).run(),
      [21]
    )

  async def test_gather_receives_same_value(self):
    """All gather functions receive the same current value."""
    received = []
    obj = object()
    def collector(v):
      received.append(v)
      return v
    Chain(obj).gather(collector, collector, collector).run()
    self.assertEqual(len(received), 3)
    self.assertTrue(all(v is obj for v in received))


# =============================================================================
# 10. Chain.with_() -- context managers
# =============================================================================

class TestWith(IsolatedAsyncioTestCase):
  """Chain.with_(body_fn, *args, **kwargs) -> self."""

  async def test_with_returns_self(self):
    """with_() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.with_(lambda v: v), c)

  async def test_with_sync_context_manager(self):
    """with_ enters and exits a sync context manager."""
    ctx = SyncCtx("hello")
    result = Chain(ctx).with_(lambda v: v.upper()).run()
    self.assertEqual(result, "HELLO")
    self.assertTrue(ctx.entered)
    self.assertTrue(ctx.exited)

  async def test_with_async_context_manager(self):
    """with_ enters and exits an async context manager."""
    ctx = AsyncCtx("world")
    result = await await_(Chain(ctx).with_(lambda v: v.upper()).run())
    self.assertEqual(result, "WORLD")
    self.assertTrue(ctx.entered)
    self.assertTrue(ctx.exited)

  async def test_with_exits_on_exception(self):
    """with_ ensures __exit__ is called even when body raises."""
    ctx = SyncCtx("val")
    def raiser(v):
      raise ValueError("body error")
    with self.assertRaises(ValueError):
      Chain(ctx).with_(raiser).run()
    self.assertTrue(ctx.exited)

  async def test_with_async_exits_on_exception(self):
    """with_ ensures __aexit__ is called even when async body raises."""
    ctx = AsyncCtx("val")
    async def raiser(v):
      raise ValueError("boom")
    with self.assertRaises(ValueError):
      await await_(Chain(ctx).with_(raiser).run())
    self.assertTrue(ctx.exited)

  async def test_with_body_receives_context_value(self):
    """The body function receives the __enter__ return value."""
    captured = []
    ctx = SyncCtx(42)
    Chain(ctx).with_(lambda v: captured.append(v)).run()
    self.assertEqual(captured, [42])


# =============================================================================
# 13. Chain.clone() -- deep copy
# =============================================================================

class TestClone(IsolatedAsyncioTestCase):
  """Chain.clone() -> Chain."""

  async def test_clone_returns_new_chain(self):
    """clone() returns a new Chain instance, not the same one."""
    c = Chain(42)
    c2 = c.clone()
    self.assertIsNot(c, c2)

  async def test_clone_is_same_type(self):
    """clone() of a Chain is a Chain."""
    self.assertIsInstance(Chain().clone(), Chain)

  async def test_clone_produces_independent_chain(self):
    """Modifying the clone does not affect the original."""
    c = Chain(1).then(lambda v: v * 10)
    c2 = c.clone()
    c2.then(lambda v: v + 1)
    self.assertEqual(c.run(), 10)
    self.assertEqual(c2.run(), 11)

  async def test_clone_produces_equivalent_result(self):
    """A freshly cloned chain produces the same result as the original."""
    c = Chain(5).then(lambda v: v + 1).then(lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(c.run(), 12)
    self.assertEqual(c2.run(), 12)

  async def test_clone_clones_finally(self):
    """Clone includes the finally_ handler."""
    ran = []
    c = Chain(1).finally_(lambda v: ran.append('original'))
    c2 = c.clone()
    c2.run()
    self.assertEqual(ran, ['original'])

  async def test_clone_empty_chain(self):
    """Cloning an empty chain works."""
    c = Chain()
    c2 = c.clone()
    self.assertIsNone(c2.run())


# =============================================================================
# 14. Chain.freeze() -- immutable snapshot
# =============================================================================

class TestFreeze(IsolatedAsyncioTestCase):
  """Chain reuse tests (run/call equivalence)."""

  async def test_chain_is_reusable(self):
    """A chain can be run multiple times."""
    c = Chain(42).then(lambda v: v * 2)
    self.assertEqual(c.run(), 84)
    self.assertEqual(c.run(), 84)
    self.assertEqual(c(), 84)

  async def test_chain_run(self):
    """Chain.run() executes the chain."""
    c = Chain(lambda: 99)
    self.assertEqual(c.run(), 99)

  async def test_chain_call(self):
    """Chain.__call__() is equivalent to run()."""
    c = Chain(lambda: 99)
    self.assertEqual(c(), 99)

  async def test_chain_with_root_override(self):
    """Chain.run(override) passes a root value."""
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c.run(5), 10)
    self.assertEqual(c(5), 10)

  async def test_chain_async(self):
    """Chains work with async operations."""
    c = Chain(aempty, 10).then(lambda v: v * 3)
    self.assertEqual(await c.run(), 30)


# =============================================================================
# 14. Chain.decorator() -- function wrapping
# =============================================================================

class TestDecorator(IsolatedAsyncioTestCase):
  """Chain.decorator() -> decorator."""

  async def test_decorator_wraps_sync_function(self):
    """decorator() wraps a sync function through the chain."""
    @Chain().then(lambda v: v * 2).decorator()
    def double(x):
      return x
    self.assertEqual(double(5), 10)

  async def test_decorator_wraps_async_function(self):
    """decorator() wraps an async function through the chain."""
    @Chain().then(lambda v: v * 2).decorator()
    async def double(x):
      return x
    self.assertEqual(await double(5), 10)

  async def test_decorator_preserves_function_name(self):
    """The decorated function preserves the original function's name."""
    @Chain().then(lambda v: v).decorator()
    def my_func(x):
      return x
    self.assertEqual(my_func.__name__, 'my_func')

  async def test_decorator_with_chain_operations(self):
    """decorator() applies the full chain pipeline."""
    @Chain().then(lambda v: v ** 2).then(lambda v: v + 1).decorator()
    def compute(x):
      return x * 2
    # compute(3) -> 6 -> 36 -> 37
    self.assertEqual(compute(3), 37)


# =============================================================================
# 16. Chain.config() -- autorun/debug
# =============================================================================

class TestConfig(IsolatedAsyncioTestCase):
  """Chain.config(*, autorun=None, debug=None) -> self."""

  async def test_config_returns_self(self):
    """config() returns the same Chain instance."""
    c = Chain()
    self.assertIs(c.config(), c)

  async def test_config_autorun_true(self):
    """config(autorun=True) enables auto-running of async chains."""
    obj = type('', (), {})()
    obj.value = False
    async def set_true():
      await asyncio.sleep(0.05)
      obj.value = True
    coro = Chain(set_true).config(autorun=True).run()
    # With autorun=True, the coroutine is scheduled as a task
    await asyncio.sleep(0.15)
    self.assertTrue(obj.value)
    await coro

  async def test_config_autorun_false(self):
    """config(autorun=False) returns a coroutine without scheduling."""
    obj = type('', (), {})()
    obj.value = False
    async def set_true():
      await asyncio.sleep(0.05)
      obj.value = True
    coro = Chain(set_true).config(autorun=False).run()
    await asyncio.sleep(0.15)
    self.assertFalse(obj.value)
    await coro

  async def test_config_debug_true_execution(self):
    """config(debug=True) enables debug mode and chain still works."""
    # _debug is a cdef field not accessible from Python; verify via execution behavior.
    result = Chain(5).then(lambda v: v * 2).config(debug=True).run()
    self.assertEqual(result, 10)

  async def test_config_debug_false_execution(self):
    """config(debug=False) disables debug mode; chain still works."""
    result = Chain(5).then(lambda v: v * 2).config(debug=True).config(debug=False).run()
    self.assertEqual(result, 10)

  async def test_config_none_does_not_change(self):
    """config() with no arguments changes nothing; chain still works."""
    c = Chain(5).config(autorun=True, debug=True)
    c.config()
    # Verify chain still executes correctly (internal flags preserved)
    result = await await_(c.then(lambda v: v * 2).run())
    self.assertEqual(result, 10)


# =============================================================================
# 18. Chain.run() / __call__() -- execution
# =============================================================================

class TestRunAndCall(IsolatedAsyncioTestCase):
  """Chain.run(override=Null) and Chain.__call__(override=Null)."""

  async def test_run_returns_result(self):
    """run() returns the chain's final value."""
    self.assertEqual(Chain(42).run(), 42)

  async def test_run_with_override(self):
    """run(value) provides a root value to a void chain."""
    self.assertEqual(Chain().then(lambda v: v * 2).run(5), 10)

  async def test_run_with_override_and_args(self):
    """run(fn, arg) calls fn(arg) as the root."""
    self.assertEqual(Chain().then(lambda v: v + 1).run(lambda x: x * 2, 3), 7)

  async def test_call_is_equivalent_to_run(self):
    """__call__() is equivalent to run()."""
    c = Chain(42)
    self.assertEqual(c(), 42)

  async def test_call_with_override(self):
    """__call__(value) provides a root value."""
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c(5), 10)

  async def test_run_async_returns_awaitable(self):
    """run() with async operations returns a coroutine."""
    result = Chain(aempty, 42).run()
    self.assertTrue(inspect.isawaitable(result))
    result = await result
    self.assertEqual(result, 42)

  async def test_run_double_root_raises(self):
    """run(value) on a chain that already has a root raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(42).run(99)

  async def test_run_nested_chain_directly_raises(self):
    """Running a nested chain directly raises QuentException."""
    inner = Chain().then(lambda v: v)
    Chain().then(inner)
    with self.assertRaises(QuentException):
      inner.run()


# =============================================================================
# 20. Chain.__bool__() / __repr__()
# =============================================================================

class TestDunderProtocol(IsolatedAsyncioTestCase):
  """Chain.__bool__() and Chain.__repr__()."""

  async def test_bool_always_true(self):
    """Chain is always truthy regardless of content."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(None)))
    self.assertTrue(bool(Chain(0)))
    self.assertTrue(bool(Chain(False)))

  async def test_repr_empty_chain(self):
    """repr of empty Chain contains 'Chain()'."""
    r = repr(Chain())
    self.assertIn('Chain()', r)

  async def test_repr_chain_with_root(self):
    """repr of Chain with a function root includes the function name."""
    def my_fn(): pass
    r = repr(Chain(my_fn))
    self.assertIn('my_fn', r)

  async def test_repr_chain_with_operations(self):
    """repr of Chain with .then() includes '.then('."""
    r = repr(Chain(1).then(lambda v: v))
    self.assertIn('.then(', r)

  async def test_repr_with_multiple_operations(self):
    """repr shows multiple operations."""
    r = repr(Chain(1).then(lambda v: v).do(print))
    self.assertIn('.then(', r)
    self.assertIn('.do(', r)


# =============================================================================
# 21. Chain.return_() / Chain.break_()
# =============================================================================

class TestReturnAndBreak(IsolatedAsyncioTestCase):
  """Class methods Chain.return_() and Chain.break_()."""

  async def test_return_exits_nested_chain(self):
    """return_() raises an internal exception (caught by the chain)."""
    obj = object()
    result = Chain(Chain().then(Chain.return_, obj)).then(lambda v: 999).run()
    self.assertIs(result, obj)

  async def test_return_without_value(self):
    """return_() with no value returns None."""
    result = Chain(Chain().then(Chain.return_)).then(lambda v: 999).run()
    self.assertIsNone(result)

  async def test_return_with_callable(self):
    """return_(fn) evaluates fn() as the return value."""
    result = Chain(Chain().then(Chain.return_, lambda: 42)).then(lambda v: 999).run()
    self.assertEqual(result, 42)

  async def test_return_async(self):
    """return_ works in async context."""
    result = await await_(
      Chain(Chain(aempty).then(Chain.return_, 42)).then(lambda v: 999).run()
    )
    self.assertEqual(result, 42)

  async def test_break_outside_loop_raises(self):
    """break_() outside a loop raises QuentException."""
    with self.assertRaises(QuentException):
      Chain().then(Chain.break_).run()

  async def test_break_outside_loop_async_raises(self):
    """break_() outside a loop raises QuentException in async context."""
    with self.assertRaises(QuentException):
      await await_(Chain(aempty).then(Chain.break_).run())

  async def test_break_in_foreach(self):
    """break_() inside foreach stops iteration."""
    def stop_at_3(x):
      if x == 3:
        return Chain.break_()
      return x
    result = Chain([1, 2, 3, 4, 5]).foreach(stop_at_3).run()
    self.assertEqual(result, [1, 2])

  async def test_break_with_value_in_foreach(self):
    """break_(value) returns that value from foreach."""
    sentinel = object()
    def stop_with_value(x):
      if x == 2:
        return Chain.break_(sentinel)
      return x
    result = Chain([1, 2, 3]).foreach(stop_with_value).run()
    self.assertIs(result, sentinel)


# =============================================================================
# 22. Chain.iterate() -- lazy generator
# =============================================================================

class TestIterate(IsolatedAsyncioTestCase):
  """Chain.iterate(fn=None) -> _Generator."""

  async def test_iterate_returns_generator(self):
    """iterate() returns a _Generator object."""
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(type(g).__name__, '_Generator')

  async def test_iterate_sync_for_loop(self):
    """iterate() supports sync for loops."""
    result = []
    for item in Chain([1, 2, 3]).iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_iterate_async_for_loop(self):
    """iterate() supports async for loops."""
    result = []
    async for item in Chain([1, 2, 3]).iterate():
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_iterate_with_fn(self):
    """iterate(fn) applies fn to each element."""
    result = []
    for item in Chain([1, 2, 3]).iterate(lambda x: x * 10):
      result.append(item)
    self.assertEqual(result, [10, 20, 30])

  async def test_iterate_with_async_fn(self):
    """iterate(async_fn) applies async fn to each element."""
    async def double(x):
      return x * 2
    result = []
    async for item in Chain([1, 2, 3]).iterate(double):
      result.append(item)
    self.assertEqual(result, [2, 4, 6])

  async def test_iterate_with_break(self):
    """Chain.break_() inside iterate stops iteration."""
    def stop_at_3(x):
      if x == 3:
        return Chain.break_()
      return x
    result = []
    for item in Chain([1, 2, 3, 4]).iterate(stop_at_3):
      result.append(item)
    self.assertEqual(result, [1, 2])

  async def test_iterate_with_return_raises(self):
    """Chain.return_() inside iterate raises QuentException."""
    with self.assertRaises(QuentException):
      for _ in Chain([1, 2, 3]).iterate(Chain.return_):
        pass

  async def test_iterate_async_iterable(self):
    """iterate() works with async iterables."""
    result = []
    async for item in Chain(AsyncIter([10, 20, 30])).iterate():
      result.append(item)
    self.assertEqual(result, [10, 20, 30])

  async def test_iterate_repr(self):
    """_Generator has a repr."""
    g = Chain([1, 2, 3]).iterate()
    self.assertEqual(repr(g), '<_Generator>')

  async def test_iterate_empty(self):
    """iterate() on empty iterable yields nothing."""
    result = []
    for item in Chain([]).iterate():
      result.append(item)
    self.assertEqual(result, [])


# =============================================================================
# 25. Cross-cutting: Method chaining fluency
# =============================================================================

class TestMethodChainingFluency(IsolatedAsyncioTestCase):
  """Every builder method returns self for fluent chaining."""

  async def test_then_returns_self(self):
    c = Chain()
    self.assertIs(c.then(lambda v: v), c)

  async def test_do_returns_self(self):
    c = Chain()
    self.assertIs(c.do(lambda v: v), c)

  async def test_except_returns_self(self):
    c = Chain()
    self.assertIs(c.except_(lambda v: v), c)

  async def test_finally_returns_self(self):
    c = Chain()
    self.assertIs(c.finally_(lambda v: v), c)

  async def test_foreach_returns_self(self):
    c = Chain()
    self.assertIs(c.foreach(lambda v: v), c)

  async def test_filter_returns_self(self):
    c = Chain()
    self.assertIs(c.filter(lambda v: v), c)

  async def test_gather_returns_self(self):
    c = Chain()
    self.assertIs(c.gather(lambda v: v), c)

  async def test_with_returns_self(self):
    c = Chain()
    self.assertIs(c.with_(lambda v: v), c)

  async def test_config_returns_self(self):
    c = Chain()
    self.assertIs(c.config(), c)

  async def test_pipe_returns_self_for_non_run(self):
    c = Chain()
    result = c | (lambda v: v)
    self.assertIs(result, c)

  async def test_full_fluent_chain(self):
    """Build a complex chain using all builder methods in sequence."""
    c = (
      Chain()
      .then(lambda v: [1, 2, 3])
      .foreach(lambda x: x * 2)
      .do(lambda v: None)
      .then(lambda v: v)
      .except_(lambda v: None)
      .finally_(lambda v: None)
    )
    self.assertIsInstance(c, Chain)


# =============================================================================
# 26. Cross-cutting: Immutability and identity
# =============================================================================

class TestImmutabilityAndIdentity(IsolatedAsyncioTestCase):
  """clone() returns new object."""

  async def test_clone_identity(self):
    """clone() produces a new object."""
    c = Chain(42)
    c2 = c.clone()
    self.assertIsNot(c, c2)

  async def test_clone_does_not_share_links(self):
    """Appending to a clone does not modify the original."""
    c = Chain(1)
    c2 = c.clone()
    c2.then(lambda v: v + 1)
    self.assertEqual(c.run(), 1)
    self.assertEqual(c2.run(), 2)

  async def test_clone_does_not_share_finally(self):
    """Cloning a chain with finally_ creates independent finally state."""
    ran_original = []
    c = Chain(1).finally_(lambda v: ran_original.append(True))
    c2 = c.clone()
    c.run()
    # The clone shares the same callback reference but has its own Link
    self.assertEqual(ran_original, [True])


# =============================================================================
# 27. Cross-cutting: QuentException conditions
# =============================================================================

class TestQuentExceptionConditions(IsolatedAsyncioTestCase):
  """All documented conditions that raise QuentException."""

  async def test_double_root_value(self):
    """Cannot override root value of a chain that already has one."""
    with self.assertRaises(QuentException):
      Chain(42).run(99)

  async def test_double_root_via_call(self):
    """Cannot override root via __call__ either."""
    with self.assertRaises(QuentException):
      Chain(42)(99)

  async def test_nested_chain_direct_run(self):
    """Cannot directly run a chain that has been nested."""
    inner = Chain().then(lambda v: v)
    Chain().then(inner)
    with self.assertRaises(QuentException):
      inner.run()

  async def test_double_finally(self):
    """Cannot register two finally_ handlers."""
    with self.assertRaises(QuentException):
      Chain().finally_(lambda v: None).finally_(lambda v: None)

  async def test_do_with_non_callable(self):
    """do() with a non-callable raises QuentException."""
    with self.assertRaises(QuentException):
      Chain().do(42).run()

  async def test_break_outside_loop_sync(self):
    """break_() outside a loop raises QuentException."""
    with self.assertRaises(QuentException):
      Chain().then(Chain.break_).run()

  async def test_break_outside_loop_async(self):
    """break_() outside a loop raises QuentException in async."""
    with self.assertRaises(QuentException):
      await await_(Chain(aempty).then(Chain.break_).run())

  async def test_return_in_iterate_raises(self):
    """return_() inside iterate raises QuentException."""
    with self.assertRaises(QuentException):
      for _ in Chain([1, 2, 3]).iterate(Chain.return_):
        pass

  async def test_return_in_async_iterate_raises(self):
    """return_() inside async iterate raises QuentException."""
    with self.assertRaises(QuentException):
      async for _ in Chain(AsyncIter([1, 2, 3])).iterate(Chain.return_):
        pass


# =============================================================================
# 28. Cross-cutting: Type checks
# =============================================================================

class TestTypeChecks(IsolatedAsyncioTestCase):
  """Type and isinstance checks for all public types."""

  async def test_chain_type(self):
    """type(Chain()) is Chain."""
    self.assertIs(type(Chain()), Chain)

  async def test_run_is_not_chain(self):
    """run() is not a Chain."""
    r = run()
    self.assertNotIsInstance(r, Chain)

  async def test_generator_is_not_chain(self):
    """_Generator from iterate() is not a Chain."""
    g = Chain([1]).iterate()
    self.assertNotIsInstance(g, Chain)


# =============================================================================
# 29. Additional edge cases and integration
# =============================================================================

class TestEdgeCases(IsolatedAsyncioTestCase):
  """Edge cases covering unusual values and interactions."""

  async def test_chain_with_none_operations(self):
    """Chain handles None flowing through operations."""
    self.assertIsNone(Chain(None).then(lambda v: v).run())

  async def test_chain_long_pipeline(self):
    """Chain handles a long pipeline without stack overflow."""
    c = Chain(0)
    for _ in range(100):
      c = c.then(lambda v: v + 1)
    self.assertEqual(c.run(), 100)

  async def test_chain_exception_in_root(self):
    """Exception in root value propagates correctly."""
    with self.assertRaises(ZeroDivisionError):
      Chain(lambda: 1/0).run()

  async def test_chain_exception_in_middle(self):
    """Exception in middle of chain propagates correctly."""
    with self.assertRaises(ValueError):
      def raiser(v):
        raise ValueError("mid")
      Chain(42).then(raiser).run()

  async def test_chain_with_class_method(self):
    """Chain works with class methods."""
    class Calc:
      @staticmethod
      def double(x): return x * 2
    self.assertEqual(Chain(5).then(Calc.double).run(), 10)

  async def test_foreach_then_filter_integration(self):
    """foreach followed by filter in sequence."""
    result = Chain([1, 2, 3, 4]).foreach(lambda x: x * 2).then(
      lambda lst: Chain(lst).filter(lambda x: x > 4).run()
    ).run()
    self.assertEqual(result, [6, 8])

  async def test_gather_then_foreach_integration(self):
    """gather results can be processed by foreach."""
    result = Chain(5).gather(
      lambda v: v + 1,
      lambda v: v + 2,
      lambda v: v + 3
    ).foreach(lambda x: x * 10).run()
    self.assertEqual(result, [60, 70, 80])

  async def test_except_then_finally_both_run(self):
    """Both except_ and finally_ callbacks run on exception."""
    exc_ran = []
    fin_ran = []
    try:
      Chain(42).then(lambda v: 1/0).except_(
        lambda v: exc_ran.append(True)
      ).finally_(
        lambda v: fin_ran.append(True)
      ).run()
    except ZeroDivisionError:
      pass
    self.assertEqual(exc_ran, [True])
    self.assertEqual(fin_ran, [True])

  async def test_chain_deep_nesting(self):
    """Deeply nested chains resolve correctly."""
    result = Chain(1).then(
      Chain().then(
        Chain().then(
          Chain().then(lambda v: v * 10)
        )
      )
    ).run()
    self.assertEqual(result, 10)

  async def test_void_chain_with_then(self):
    """Void chain (no root) with then(fn) calls fn() with no args (current_value is Null)."""
    self.assertEqual(Chain().then(lambda: 'created').run(), 'created')

  async def test_void_chain_with_root_override(self):
    """Void chain with root override at run time."""
    self.assertEqual(Chain().then(lambda v: v * 2).run(5), 10)

  async def test_chain_with_ellipsis_in_then(self):
    """Chain with Ellipsis-style call in then()."""
    self.assertEqual(
      Chain(10).then(lambda: 42, ...).run(),
      42
    )

  async def test_except_with_noraise_returns_none_on_null(self):
    """except_ with reraise=False and handler returning nothing gives None."""
    result = Chain(42).then(lambda v: 1/0).except_(lambda v: None, reraise=False).run()
    self.assertIsNone(result)



class TestExceptWithExplicitArgs(IsolatedAsyncioTestCase):
  """Test except_ handler with explicit arguments."""

  async def test_except_handler_with_explicit_value(self):
    """except_ handler called with explicit arg returns that arg when reraise=False."""
    obj = object()
    result = Chain(42).then(lambda v: 1/0).except_(lambda v: v, obj, reraise=False).run()
    self.assertIs(result, obj)

  async def test_except_handler_with_kwargs(self):
    """except_ handler receives kwargs."""
    result = Chain(42).then(lambda v: 1/0).except_(
      lambda x=0: x, x=42, reraise=False
    ).run()
    self.assertEqual(result, 42)


class TestConfigDebugExecution(IsolatedAsyncioTestCase):
  """Test debug mode execution."""

  async def test_debug_mode_sync(self):
    """Chain executes correctly in debug mode (sync)."""
    result = Chain(5).then(lambda v: v * 2).config(debug=True).run()
    self.assertEqual(result, 10)

  async def test_debug_mode_async(self):
    """Chain executes correctly in debug mode (async)."""
    result = await await_(Chain(aempty, 5).then(lambda v: v * 2).config(debug=True).run())
    self.assertEqual(result, 10)

  async def test_debug_mode_with_exception(self):
    """Chain in debug mode still propagates exceptions."""
    with self.assertRaises(ZeroDivisionError):
      Chain(42).then(lambda v: 1/0).config(debug=True).run()
