"""Exhaustive error-path, boundary-condition, and unusual-input-type tests.

Targets untested edge cases across BaseException propagation, CancelledError,
non-iterable/non-CM/non-callable errors, generators as values, reentrant chains,
duplicate callables, unusual value types, unusual exceptions, pipe operator edges,
Null sentinel behavior, config edges, sleep edges, and deep feature combos.
"""
import sys
import time
import types
import asyncio
import warnings
import contextlib
from itertools import product
from unittest import IsolatedAsyncioTestCase

from quent import Chain, QuentException, Null
from tests.utils import empty, aempty, await_, TestExc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CustomBaseExc(BaseException):
  pass


class CustomKBInterrupt(KeyboardInterrupt):
  pass


class SyncCM:
  """Simple sync context manager for testing."""
  def __init__(self):
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  """Simple async context manager for testing."""
  def __init__(self):
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self

  async def __aexit__(self, *args):
    self.exited = True
    return False


class EnterOnlyCM:
  """Object with __enter__ but no __exit__."""
  def __enter__(self):
    return self


class AsyncEnterOnlyCM:
  """Object with __aenter__ but body is sync."""
  async def __aenter__(self):
    return self

  async def __aexit__(self, *args):
    return False


class BadReprException(Exception):
  def __repr__(self):
    raise RuntimeError("repr is broken")


# ---------------------------------------------------------------------------
# A. BaseException Propagation Tests
# ---------------------------------------------------------------------------

class BaseExceptionPropagationTests(IsolatedAsyncioTestCase):

  async def test_keyboard_interrupt_in_root_propagates(self):
    """KeyboardInterrupt in root evaluation propagates out (not caught by default except_)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        handler_called = [False]
        def handler(v=None):
          handler_called[0] = True
        def raise_kbi(v=None):
          raise KeyboardInterrupt("test")
        with self.assertRaises(KeyboardInterrupt):
          await await_(
            Chain(fn).then(raise_kbi).except_(handler).run()
          )
        self.assertFalse(handler_called[0])

  async def test_keyboard_interrupt_in_link_propagates(self):
    """KeyboardInterrupt in a link propagates out past default except_."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        handler_called = [False]
        def handler(v=None):
          handler_called[0] = True
        def raise_kbi_link(v=None):
          raise KeyboardInterrupt("link")
        with self.assertRaises(KeyboardInterrupt):
          await await_(
            Chain(fn, 42)
            .then(raise_kbi_link)
            .except_(handler)
            .run()
          )
        self.assertFalse(handler_called[0])

  async def test_system_exit_in_chain_propagates(self):
    """SystemExit in chain propagates out (not caught by default except_)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        handler_called = [False]
        def handler(v=None):
          handler_called[0] = True
        def raise_se(v=None):
          raise SystemExit(1)
        with self.assertRaises(SystemExit):
          await await_(
            Chain(fn, 1).then(raise_se).except_(handler).run()
          )
        self.assertFalse(handler_called[0])

  async def test_keyboard_interrupt_in_async_chain_propagates(self):
    """KeyboardInterrupt in async chain link propagates out."""
    async def raise_kbi(v=None):
      raise KeyboardInterrupt("async")
    with self.assertRaises(KeyboardInterrupt):
      await await_(
        Chain(aempty, 1).then(raise_kbi).except_(lambda v: None).run()
      )

  async def test_except_with_base_exception_catches_keyboard_interrupt(self):
    """except_ with exceptions=BaseException catches KeyboardInterrupt."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        handler_called = [False]
        def handler(v=None):
          handler_called[0] = True
        def raise_kbi(v=None):
          raise KeyboardInterrupt("caught")
        try:
          await await_(
            Chain(fn, 1)
            .then(raise_kbi)
            .except_(handler, exceptions=BaseException)
            .run()
          )
        except KeyboardInterrupt:
          pass
        self.assertTrue(handler_called[0])

  async def test_except_with_exception_does_not_catch_keyboard_interrupt(self):
    """except_ with exceptions=Exception does NOT catch KeyboardInterrupt."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        handler_called = [False]
        def handler(v=None):
          handler_called[0] = True
        def raise_kbi(v=None):
          raise KeyboardInterrupt("not caught")
        with self.assertRaises(KeyboardInterrupt):
          await await_(
            Chain(fn, 1)
            .then(raise_kbi)
            .except_(handler, exceptions=Exception)
            .run()
          )
        self.assertFalse(handler_called[0])

  async def test_finally_runs_when_base_exception_raised(self):
    """finally_ runs even when BaseException (non-Exception) is raised."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        finally_called = [False]
        def fin_handler(v=None):
          finally_called[0] = True
        def raise_custom_base(v=None):
          raise CustomBaseExc("test")
        with self.assertRaises(CustomBaseExc):
          await await_(
            Chain(fn, 1)
            .then(raise_custom_base)
            .finally_(fin_handler)
            .run()
          )
        self.assertTrue(finally_called[0])

  async def test_base_exception_in_foreach_propagates_with_temp_args(self):
    """BaseException in foreach propagates out."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def fn_raise(x):
          if x == 2:
            raise CustomBaseExc("in foreach")
          return fn1(x)
        with self.assertRaises(CustomBaseExc):
          await await_(
            Chain(fn2, [1, 2, 3]).foreach(fn_raise).run()
          )

  async def test_custom_base_exception_subclass_propagates(self):
    """Custom BaseException subclass propagates through chain."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def raise_it(v=None):
          raise CustomKBInterrupt("custom")
        with self.assertRaises(CustomKBInterrupt):
          await await_(
            Chain(fn, 1).then(raise_it).except_(lambda v: None).run()
          )


# ---------------------------------------------------------------------------
# B. CancelledError Propagation Tests
# ---------------------------------------------------------------------------

class CancelledErrorPropagationTests(IsolatedAsyncioTestCase):

  async def test_cancelled_error_in_async_chain_link_propagates(self):
    """asyncio.CancelledError in async chain link propagates."""
    async def raise_ce(v=None):
      raise asyncio.CancelledError()
    with self.assertRaises(asyncio.CancelledError):
      await await_(
        Chain(aempty, 1).then(raise_ce).run()
      )

  async def test_cancelled_error_in_async_foreach_propagates(self):
    """CancelledError in async foreach propagates."""
    async def raise_ce(x):
      if x == 2:
        raise asyncio.CancelledError()
      return x
    with self.assertRaises(asyncio.CancelledError):
      await await_(
        Chain(aempty, [1, 2, 3]).foreach(raise_ce).run()
      )

  async def test_cancelled_error_in_async_cm_calls_aexit(self):
    """CancelledError in async context manager body calls __aexit__."""
    cm = AsyncCM()
    async def raise_ce(v=None):
      raise asyncio.CancelledError()
    with self.assertRaises(asyncio.CancelledError):
      await await_(
        Chain(aempty, cm).with_(raise_ce).run()
      )
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_cancelled_error_in_async_gather_propagates(self):
    """CancelledError in gather propagates."""
    async def raise_ce(v):
      raise asyncio.CancelledError()
    async def ok_fn(v):
      return v
    with self.assertRaises((asyncio.CancelledError, BaseException)):
      await await_(
        Chain(aempty, 1).gather(raise_ce, ok_fn).run()
      )

  async def test_cancelled_error_not_caught_by_default_except(self):
    """CancelledError is BaseException (on 3.9+), not caught by default except_ (exceptions=Exception)."""
    handler_called = [False]
    def handler(v=None):
      handler_called[0] = True
    async def raise_ce(v=None):
      raise asyncio.CancelledError()
    # On Python 3.9+, CancelledError is BaseException, not Exception
    if issubclass(asyncio.CancelledError, BaseException) and not issubclass(asyncio.CancelledError, Exception):
      with self.assertRaises(asyncio.CancelledError):
        await await_(
          Chain(aempty, 1).then(raise_ce).except_(handler).run()
        )
      self.assertFalse(handler_called[0])
    else:
      # On older Python, CancelledError IS Exception
      try:
        await await_(
          Chain(aempty, 1).then(raise_ce).except_(handler).run()
        )
      except asyncio.CancelledError:
        pass
      self.assertTrue(handler_called[0])

  async def test_cancelled_error_with_finally_still_runs(self):
    """CancelledError during async chain with finally_ -- finally_ still runs."""
    finally_called = [False]
    async def fin_handler(v=None):
      finally_called[0] = True
    async def raise_ce(v=None):
      raise asyncio.CancelledError()
    with self.assertRaises(asyncio.CancelledError):
      await await_(
        Chain(aempty, 1).then(raise_ce).finally_(fin_handler).run()
      )
    self.assertTrue(finally_called[0])


# ---------------------------------------------------------------------------
# C. Non-Iterable in foreach/filter
# ---------------------------------------------------------------------------

class NonIterableTests(IsolatedAsyncioTestCase):

  async def test_foreach_on_integer_raises_type_error(self):
    """foreach on an integer (not iterable) raises TypeError/AttributeError."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises((TypeError, AttributeError)):
          await await_(
            Chain(fn, 42).foreach(lambda x: x).run()
          )

  async def test_foreach_on_none_raises_type_error(self):
    """foreach on None raises TypeError/AttributeError."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises((TypeError, AttributeError)):
          await await_(
            Chain(fn, None).foreach(lambda x: x).run()
          )

  async def test_filter_on_integer_raises_type_error(self):
    """filter on an integer raises TypeError/AttributeError."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises((TypeError, AttributeError)):
          await await_(
            Chain(fn, 42).filter(lambda x: True).run()
          )

  async def test_filter_on_none_raises_type_error(self):
    """filter on None raises TypeError/AttributeError."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises((TypeError, AttributeError)):
          await await_(
            Chain(fn, None).filter(lambda x: True).run()
          )


# ---------------------------------------------------------------------------
# D. Non-Context-Manager in with_
# ---------------------------------------------------------------------------

class NonContextManagerTests(IsolatedAsyncioTestCase):

  async def test_with_on_integer_raises_attribute_error(self):
    """with_ on an integer (no __enter__) raises AttributeError."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(AttributeError):
          await await_(
            Chain(fn, 42).with_(lambda v: v).run()
          )

  async def test_with_on_none_raises_attribute_error(self):
    """with_ on None raises AttributeError."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(AttributeError):
          await await_(
            Chain(fn, None).with_(lambda v: v).run()
          )

  async def test_with_on_enter_only_cm_raises(self):
    """with_ on object with __enter__ but no __exit__ raises on exit."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        cm = EnterOnlyCM()
        with self.assertRaises(AttributeError):
          await await_(
            Chain(fn, cm).with_(lambda v: v).run()
          )

  async def test_with_on_async_enter_cm_uses_async_path(self):
    """with_ on object with __aenter__ uses async path."""
    cm = AsyncEnterOnlyCM()
    result = await await_(
      Chain(aempty, cm).with_(lambda v: "body_result").run()
    )
    self.assertEqual(result, "body_result")


# ---------------------------------------------------------------------------
# E. Non-Callable Errors
# ---------------------------------------------------------------------------

class NonCallableErrorTests(IsolatedAsyncioTestCase):

  async def test_do_with_integer_raises_quent_exception(self):
    """Chain(42).do(42) raises QuentException for non-callable in .do()."""
    with self.assertRaises(QuentException):
      Chain(42).do(42)

  async def test_do_with_none_raises_quent_exception(self):
    """Chain(42).do(None) raises QuentException for non-callable in .do()."""
    with self.assertRaises(QuentException):
      Chain(42).do(None)

  async def test_do_with_string_raises_quent_exception(self):
    """Chain(42).do('hello') raises QuentException for non-callable in .do()."""
    with self.assertRaises(QuentException):
      Chain(42).do("hello")

  async def test_gather_with_non_callable_raises_type_error(self):
    """gather with non-callable in the tuple raises TypeError at call time."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(TypeError):
          await await_(
            Chain(fn, 42).gather(lambda v: v, 99).run()
          )

  async def test_foreach_with_non_callable_fn_raises(self):
    """foreach with non-callable fn raises QuentException at construction time."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        with self.assertRaises(QuentException):
          Chain(fn, [1, 2, 3]).foreach(42)


# ---------------------------------------------------------------------------
# F. Generator/Iterator Objects as Values
# ---------------------------------------------------------------------------

class GeneratorIteratorValueTests(IsolatedAsyncioTestCase):

  async def test_chain_with_generator_object_as_root(self):
    """Chain(iter([1,2,3])) -- iterator is not callable, returned as-is via then."""
    it = iter([1, 2, 3])
    result = await await_(Chain(it).run())
    # iter is not callable, so EVAL_RETURN_AS_IS -- but wait, iter obj IS the root
    # and it's not callable, so with allow_literal=True, it should be returned as-is
    self.assertIs(result, it)

  async def test_chain_with_generator_function_as_root(self):
    """Chain(my_gen) -- generator function is callable, called to produce generator."""
    def my_gen():
      yield 1
      yield 2
    result = await await_(Chain(my_gen).run())
    # my_gen is callable, so it's called with no args -> returns a generator
    self.assertTrue(hasattr(result, '__next__'))
    self.assertEqual(list(result), [1, 2])

  async def test_then_with_iterator_object_as_literal(self):
    """Chain().then(iter([1,2,3])) -- iter is not callable, returned as-is via then."""
    it = iter([1, 2, 3])
    result = await await_(Chain(42).then(it).run())
    # then uses allow_literal=True, so non-callable is returned as-is
    self.assertIs(result, it)

  async def test_foreach_on_generator_object(self):
    """foreach on generator object iterates correctly."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, [10, 20, 30]).then(iter).foreach(lambda x: x * 2).run()
        )
        self.assertEqual(result, [20, 40, 60])

  async def test_foreach_on_range_object(self):
    """foreach on range object iterates correctly."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, range(5)).foreach(lambda x: x ** 2).run()
        )
        self.assertEqual(result, [0, 1, 4, 9, 16])

  async def test_filter_on_generator_object(self):
    """filter on generator object iterates correctly."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, [1, 2, 3, 4, 5]).then(iter).filter(lambda x: x % 2 == 0).run()
        )
        self.assertEqual(result, [2, 4])


# ---------------------------------------------------------------------------
# G. Reentrant/Recursive Chain Execution
# ---------------------------------------------------------------------------

class ReentrantChainTests(IsolatedAsyncioTestCase):

  async def test_link_runs_different_chain(self):
    """Chain where a link calls run() on a DIFFERENT chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        other = Chain(fn1, 10).then(lambda v: v * 3)
        result = await await_(
          Chain(fn2, 1).then(lambda v: other.run()).run()
        )
        self.assertEqual(result, 30)

  async def test_except_handler_runs_different_chain(self):
    """except_ handler runs a different chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        recovery = Chain(fn1, 99)
        def raise_err(v=None):
          raise TestExc("fail")
        def handler(v=None):
          return recovery.run()
        result = await await_(
          Chain(fn2, 1).then(raise_err).except_(handler, reraise=False).run()
        )
        self.assertEqual(result, 99)

  async def test_finally_handler_runs_different_chain(self):
    """finally_ handler runs a different chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        side_effects = []
        cleanup_chain = Chain(fn1).then(lambda v: side_effects.append("cleaned"))
        with warnings.catch_warnings():
          warnings.simplefilter('ignore', RuntimeWarning)
          result = await await_(
            Chain(fn2, 42).finally_(lambda v: cleanup_chain.run()).run()
          )
        self.assertEqual(result, 42)
        self.assertIn("cleaned", side_effects)

  async def test_chain_within_chain_link(self):
    """Chain executed within another chain's link."""
    # Use chains WITHOUT root values so run(value) works
    inner = Chain().then(lambda v: v + 10)
    outer = Chain().then(lambda v: inner.run(v))
    result = await await_(outer.run(5))
    self.assertEqual(result, 15)

  async def test_link_creates_and_runs_new_chain(self):
    """Chain where a link creates and runs a brand new chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def create_and_run(v):
          return Chain(fn1, v).then(lambda x: x * 2).run()
        result = await await_(
          Chain(fn2, 7).then(create_and_run).run()
        )
        self.assertEqual(result, 14)


# ---------------------------------------------------------------------------
# H. Same Callable Multiple Times
# ---------------------------------------------------------------------------

class SameCallableMultipleTimesTests(IsolatedAsyncioTestCase):

  async def test_same_fn_as_root_and_then(self):
    """Same function used as root AND then: Chain(fn).then(fn).run()."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        result = await await_(Chain(fn1, 42).then(fn2).run())
        self.assertEqual(result, 42)

  async def test_same_fn_in_multiple_then(self):
    """Same function used in multiple then: Chain(42).then(fn).then(fn).then(fn)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        double = lambda v: v * 2
        result = await await_(
          Chain(fn, 1).then(double).then(double).then(double).run()
        )
        self.assertEqual(result, 8)

  async def test_same_lambda_in_multiple_positions(self):
    """Same lambda used in multiple positions."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        add_one = lambda v: v + 1
        result = await await_(
          Chain(fn, 0).then(add_one).then(add_one).then(add_one).run()
        )
        self.assertEqual(result, 3)

  async def test_same_chain_as_link_in_two_parents(self):
    """Same Chain used as link in two different parent chains."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        shared = Chain().then(lambda v: v + 100)
        r1 = await await_(Chain(fn1, 1).then(shared).run())
        r2 = await await_(Chain(fn2, 2).then(shared).run())
        self.assertEqual(r1, 101)
        self.assertEqual(r2, 102)


# ---------------------------------------------------------------------------
# I. Unusual Value Types Through Chain
# ---------------------------------------------------------------------------

class UnusualValueTypesTests(IsolatedAsyncioTestCase):

  async def test_none_flowing_through_multiple_links(self):
    """None flowing through multiple links."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, None).then(lambda v: v).then(lambda v: v).run()
        )
        self.assertIsNone(result)

  async def test_exception_object_as_data(self):
    """Exception object (not raised, passed as data) flowing through chain."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        exc = ValueError("test data")
        result = await await_(
          Chain(fn, exc).then(lambda v: v).run()
        )
        self.assertIsInstance(result, ValueError)
        self.assertEqual(str(result), "test data")

  async def test_chain_object_as_nested_chain(self):
    """Chain object passed to then() is treated as nested chain (not data)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # Inner chain WITHOUT root value: receives current value from parent
        inner = Chain().then(lambda v: v + 100)
        result = await await_(
          Chain(fn, 5).then(inner).run()
        )
        self.assertEqual(result, 105)

  async def test_chain_with_root_as_nested_raises(self):
    """Chain with root used as nested chain raises QuentException on root override."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain(fn1, 99)
        with self.assertRaises(QuentException):
          await await_(
            Chain(fn2, 5).then(inner).run()
          )

  async def test_class_type_object_through_chain(self):
    """A class (type object) flowing through chain as data via fn(dict)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # fn(dict) returns dict (the class itself), so it flows as value
        result = await await_(Chain(fn, dict).run())
        self.assertIs(result, dict)

  async def test_class_type_as_literal_through_then(self):
    """A class passed to then() is called with current value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(Chain(fn, "hello").then(list).run())
        self.assertEqual(result, ['h', 'e', 'l', 'l', 'o'])

  async def test_module_object_through_chain(self):
    """A module object flowing through chain (as literal)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # Module is not callable -> returned as-is with allow_literal=True
        result = await await_(Chain(fn, sys).then(lambda v: type(v).__name__).run())
        self.assertEqual(result, "module")

  async def test_bytes_through_chain(self):
    """bytes flowing through chain."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, b"hello").then(lambda v: v + b" world").run()
        )
        self.assertEqual(result, b"hello world")

  async def test_bytearray_through_chain(self):
    """bytearray flowing through chain."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, bytearray(b"abc")).then(lambda v: len(v)).run()
        )
        self.assertEqual(result, 3)

  async def test_memoryview_through_chain(self):
    """memoryview flowing through chain."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        data = b"test"
        result = await await_(
          Chain(fn, memoryview(data)).then(lambda v: bytes(v)).run()
        )
        self.assertEqual(result, b"test")

  async def test_slice_object_through_chain(self):
    """slice object flowing through chain."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        s = slice(1, 5, 2)
        result = await await_(
          Chain(fn, s).then(lambda v: (v.start, v.stop, v.step)).run()
        )
        self.assertEqual(result, (1, 5, 2))


# ---------------------------------------------------------------------------
# J. Callback That Raises Unusual Exceptions
# ---------------------------------------------------------------------------

class UnusualExceptionTests(IsolatedAsyncioTestCase):

  async def test_callback_raises_stop_iteration(self):
    """Callback that raises StopIteration outside iteration context.
    In sync context, StopIteration propagates. In async context, Python wraps
    StopIteration in RuntimeError ('generator raised StopIteration').
    """
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def raise_stop(v=None):
          raise StopIteration("outside iter")
        if fn is empty:
          with self.assertRaises(StopIteration):
            await await_(
              Chain(fn, 1).then(raise_stop).run()
            )
        else:
          # async: Python converts StopIteration -> RuntimeError
          with self.assertRaises(RuntimeError):
            await await_(
              Chain(fn, 1).then(raise_stop).run()
            )

  async def test_callback_raises_stop_async_iteration(self):
    """Callback that raises StopAsyncIteration."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def raise_stop(v=None):
          raise StopAsyncIteration("outside aiter")
        with self.assertRaises(StopAsyncIteration):
          await await_(
            Chain(fn, 1).then(raise_stop).run()
          )

  async def test_callback_raises_generator_exit(self):
    """Callback that raises GeneratorExit."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def raise_ge(v=None):
          raise GeneratorExit()
        # GeneratorExit is BaseException, not Exception
        with self.assertRaises(GeneratorExit):
          await await_(
            Chain(fn, 1).then(raise_ge).except_(lambda v: None).run()
          )

  async def test_callback_raises_recursion_error(self):
    """Callback that raises RecursionError from infinite recursion."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def infinite(v=None):
          return infinite(v)
        with self.assertRaises(RecursionError):
          await await_(
            Chain(fn, 1).then(infinite).run()
          )

  async def test_callback_raises_memory_error(self):
    """Callback that raises MemoryError."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def raise_mem(v=None):
          raise MemoryError("simulated")
        with self.assertRaises(MemoryError):
          await await_(
            Chain(fn, 1).then(raise_mem).run()
          )

  async def test_exception_with_bad_repr(self):
    """Exception with unprintable __repr__ -- traceback handling still works."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def raise_bad(v=None):
          raise BadReprException("bad repr")
        with self.assertRaises(BadReprException):
          await await_(
            Chain(fn, 1).then(raise_bad).run()
          )


# ---------------------------------------------------------------------------
# K. Pipe Operator Edge Cases
# ---------------------------------------------------------------------------

class PipeOperatorEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_pipe_none_literal(self):
    """Chain(42) | None -- pipes None literal."""
    c = Chain(42) | None
    result = await await_(c.run())
    self.assertIsNone(result)

  async def test_pipe_zero_literal(self):
    """Chain(42) | 0 -- pipes 0 literal."""
    c = Chain(42) | 0
    result = await await_(c.run())
    self.assertEqual(result, 0)

  async def test_pipe_false_literal(self):
    """Chain(42) | False -- pipes False."""
    c = Chain(42) | False
    result = await await_(c.run())
    self.assertFalse(result)

  async def test_pipe_empty_string(self):
    """Chain(42) | '' -- pipes empty string."""
    c = Chain(42) | ""
    result = await await_(c.run())
    self.assertEqual(result, "")

  async def test_pipe_empty_list(self):
    """Chain(42) | [] -- pipes empty list."""
    c = Chain(42) | []
    result = await await_(c.run())
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# L. Null Sentinel Interactions
# ---------------------------------------------------------------------------

class NullSentinelTests(IsolatedAsyncioTestCase):

  async def test_null_identity(self):
    """Null is Null (identity check)."""
    self.assertIs(Null, Null)

  async def test_null_is_not_none(self):
    """Null is not None."""
    self.assertIsNot(Null, None)

  async def test_null_repr(self):
    """repr(Null) == '<Null>'."""
    self.assertEqual(repr(Null), '<Null>')

  async def test_null_is_truthy(self):
    """bool(Null) returns True (no __bool__ defined, so truthy by default)."""
    self.assertTrue(bool(Null))


# ---------------------------------------------------------------------------
# M. Config Edge Cases
# ---------------------------------------------------------------------------

class ConfigEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_config_autorun_with_truthy_int(self):
    """config(autorun=1) -- non-bool truthy converts to True via bool()."""
    # Test indirectly: autorun=1 should make __call__ behave like run()
    # which wraps the result in ensure_future if it's a coroutine
    c = Chain(42).config(autorun=1)
    # config returns self for chaining
    self.assertIsInstance(c, Chain)
    # Verify it runs correctly (autorun just affects call behavior)
    result = await await_(c.run())
    self.assertEqual(result, 42)

  async def test_config_autorun_with_falsy_int(self):
    """config(autorun=0) -- non-bool falsy converts to False."""
    c = Chain(42).config(autorun=0)
    result = await await_(c.run())
    self.assertEqual(result, 42)

  async def test_config_autorun_with_truthy_string(self):
    """config(autorun='yes') -- string truthy converts to True."""
    c = Chain(42).config(autorun="yes")
    result = await await_(c.run())
    self.assertEqual(result, 42)

  async def test_config_called_multiple_times_last_wins(self):
    """config called multiple times -- last call wins."""
    c = Chain(42).config(autorun=True).config(autorun=False)
    # config returns self, so chaining works
    self.assertIsInstance(c, Chain)
    result = await await_(c.run())
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# O. Deep Feature Combinations
# ---------------------------------------------------------------------------

class DeepFeatureCombinationTests(IsolatedAsyncioTestCase):

  async def test_foreach_inside_with(self):
    """Chain with foreach inside with_: chain -> CM -> foreach inside CM body."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        cm = SyncCM() if fn2 is empty else AsyncCM()
        def body_fn(cm_val):
          return Chain(fn1, [1, 2, 3]).foreach(lambda x: x * 10).run()
        result = await await_(
          Chain(fn2, cm).with_(body_fn).run()
        )
        self.assertEqual(result, [10, 20, 30])

  async def test_gather_inside_foreach(self):
    """Chain with gather inside foreach: foreach(fn_that_uses_gather)."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def process(x):
          return Chain(fn1, x).gather(lambda v: v + 1, lambda v: v + 2).run()
        result = await await_(
          Chain(fn2, [10, 20]).foreach(process).run()
        )
        self.assertEqual(result, [[11, 12], [21, 22]])

  async def test_with_inside_foreach(self):
    """Chain with with_ inside foreach: foreach(fn_that_uses_with)."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def process_with_cm(x):
          cm = SyncCM() if fn1 is empty else AsyncCM()
          return Chain(fn1, cm).with_(lambda v: x * 2).run()
        result = await await_(
          Chain(fn2, [1, 2, 3]).foreach(process_with_cm).run()
        )
        self.assertEqual(result, [2, 4, 6])

  async def test_except_catches_error_from_nested_foreach(self):
    """except_ catches error from nested foreach."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        caught = [False]
        def handler(v=None):
          caught[0] = True
          return "recovered"
        def raise_in_foreach(x):
          if x == 2:
            raise TestExc("in foreach")
          return fn1(x)
        result = await await_(
          Chain(fn2, [1, 2, 3])
          .foreach(raise_in_foreach)
          .except_(handler, reraise=False)
          .run()
        )
        self.assertTrue(caught[0])
        self.assertEqual(result, "recovered")

  async def test_kitchen_sink_combination(self):
    """Chain with finally_ + except_ + foreach + nested chain -- the kitchen sink."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        finally_called = [False]
        except_called = [False]

        def fin_handler(v=None):
          finally_called[0] = True

        def exc_handler(v=None):
          except_called[0] = True
          return "handled"

        # inner_chain receives a list and tries list + 1 -> TypeError
        inner_chain = Chain().then(lambda v: v + 1)

        result = await await_(
          Chain(fn, [10, 20, 30])
          .foreach(lambda x: x * 2)
          .then(inner_chain)
          .except_(exc_handler, reraise=False)
          .finally_(fin_handler)
          .run()
        )

        self.assertTrue(finally_called[0])
        self.assertTrue(except_called[0])
        self.assertEqual(result, "handled")


# ---------------------------------------------------------------------------
# Additional edge tests
# ---------------------------------------------------------------------------

class AdditionalEdgeTests(IsolatedAsyncioTestCase):
  """Extra tests for completeness beyond the main categories."""

  async def test_stop_iteration_caught_by_except(self):
    """StopIteration is an Exception subclass, caught by default except_.
    In async, Python wraps StopIteration -> RuntimeError, which is also Exception.
    """
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        caught = [False]
        def handler(v=None):
          caught[0] = True
          return "caught_stop"
        def raise_stop(v=None):
          raise StopIteration("stop")
        result = await await_(
          Chain(fn, 1).then(raise_stop).except_(handler, reraise=False).run()
        )
        self.assertTrue(caught[0])
        self.assertEqual(result, "caught_stop")

  async def test_recursion_error_caught_by_except(self):
    """RecursionError is an Exception subclass, caught by default except_."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        caught = [False]
        def handler(v=None):
          caught[0] = True
          return "recovered"
        def infinite(v=None):
          return infinite(v)
        result = await await_(
          Chain(fn, 1).then(infinite).except_(handler, reraise=False).run()
        )
        self.assertTrue(caught[0])
        self.assertEqual(result, "recovered")

  async def test_generator_exit_is_base_exception(self):
    """GeneratorExit is BaseException, NOT Exception -- not caught by default except_."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        caught = [False]
        def handler(v=None):
          caught[0] = True
        def raise_ge(v=None):
          raise GeneratorExit()
        with self.assertRaises(GeneratorExit):
          await await_(
            Chain(fn, 1).then(raise_ge).except_(handler).run()
          )
        self.assertFalse(caught[0])

  async def test_except_with_tuple_of_exceptions(self):
    """except_ with exceptions=(TypeError, ValueError) catches both."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        for exc_cls in [TypeError, ValueError]:
          caught = [None]
          def handler(v=None):
            caught[0] = True
            return "handled"
          def raiser(v=None, _cls=exc_cls):
            raise _cls("test")
          result = await await_(
            Chain(fn, 1)
            .then(raiser)
            .except_(handler, exceptions=(TypeError, ValueError), reraise=False)
            .run()
          )
          self.assertTrue(caught[0])
          self.assertEqual(result, "handled")

  async def test_empty_chain_returns_none(self):
    """Empty chain with no root and no links returns None."""
    result = await await_(Chain().run())
    self.assertIsNone(result)

  async def test_chain_with_only_root_returns_root(self):
    """Chain with only root value returns that value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(Chain(fn, 42).run())
        self.assertEqual(result, 42)

  async def test_chain_multiple_executions(self):
    """Chain can be executed multiple times with different values."""
    # Chain WITHOUT root value so we can supply one via run()
    c = Chain().then(lambda v: v + 1)
    r1 = await await_(c.run(10))
    r2 = await await_(c.run(20))
    r3 = await await_(c.run(30))
    self.assertEqual(r1, 11)
    self.assertEqual(r2, 21)
    self.assertEqual(r3, 31)

  async def test_config_debug_with_truthy_values(self):
    """config(debug=1) and config(debug='yes') enable debug mode."""
    # _debug is a cdef field, so test indirectly via behavior
    # In debug mode, the chain adds extra logging; just verify it doesn't crash
    c = Chain(42).config(debug=1)
    result = await await_(c.run())
    self.assertEqual(result, 42)

    c2 = Chain(42).config(debug="yes")
    result2 = await await_(c2.run())
    self.assertEqual(result2, 42)

    c3 = Chain(42).config(debug=0)
    result3 = await await_(c3.run())
    self.assertEqual(result3, 42)

  async def test_chain_bool_is_always_true(self):
    """Chain.__bool__ always returns True."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(42)))

  async def test_except_with_string_raises_type_error(self):
    """except_ with exceptions='SomeString' raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(42).except_(lambda v: None, exceptions="ValueError")

  async def test_double_finally_raises(self):
    """Registering two finally_ handlers raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(42).finally_(lambda v: None).finally_(lambda v: None)

  async def test_nested_chain_cannot_be_run_directly(self):
    """A nested chain (is_nested=True) cannot be run() directly."""
    inner = Chain().then(lambda v: v + 1)
    # Make inner nested by using it in a parent
    _ = Chain(42).then(inner)
    with self.assertRaises(QuentException):
      inner.run()

  async def test_cannot_override_root_value(self):
    """Cannot override the root value of a Chain that already has one."""
    c = Chain(42)
    with self.assertRaises(QuentException):
      await await_(c.run(99))

  async def test_foreach_with_empty_list(self):
    """foreach on empty list returns empty list."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, []).foreach(lambda x: x * 2).run()
        )
        self.assertEqual(result, [])

  async def test_filter_with_empty_list(self):
    """filter on empty list returns empty list."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        result = await await_(
          Chain(fn, []).filter(lambda x: True).run()
        )
        self.assertEqual(result, [])


if __name__ == '__main__':
  import unittest
  unittest.main()
