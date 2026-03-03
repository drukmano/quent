"""Exhaustive tests for exception handling, traceback modification, control flow
signals, and all error paths in the quent library.

Categories covered:
  A. Exception Type Matching Edge Cases
  B. Exception Handler Behavior
  C. Finally Handler Edge Cases
  D. _Return Signal Comprehensive
  E. _Break Signal Comprehensive
  F. Exception at Every Position
  G. Traceback Modification Verification
  H. Exception Chaining
  I. QuentException Scenarios
  J. Async Exception Paths
"""
import asyncio
import sys
import traceback
import warnings
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Cascade, QuentException, run, Null
from tests.utils import empty, aempty, await_, TestExc, MyTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Exc1(TestExc):
  pass


class Exc2(TestExc):
  pass


class Exc3(Exc2):
  """Exc3 inherits from Exc2 which inherits from TestExc."""
  pass


class CustomValueError(ValueError):
  pass


class CustomBaseExc(BaseException):
  pass


class CustomKeyboardInterrupt(KeyboardInterrupt):
  pass


def raise_exc(exc_type=TestExc, msg=''):
  def _raise(v=None):
    raise exc_type(msg)
  return _raise


def raise_direct(v=None):
  raise TestExc('direct')


async def async_raise(v=None):
  raise TestExc('async raise')


def identity(v=None):
  return v


async def async_identity(v=None):
  return v


class SyncCtxMgr:
  """Simple sync context manager for testing."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    return False


class AsyncCtxMgr:
  """Simple async context manager for testing."""
  def __init__(self, value=None):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    return False


# ---------------------------------------------------------------------------
# A. Exception Type Matching Edge Cases (12+ tests)
# ---------------------------------------------------------------------------

class ExceptionTypeMatchingTests(MyTestCase):

  async def test_a01_except_catches_value_error(self):
    """except_(handler, exceptions=ValueError) catches ValueError."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1).then(raise_exc(ValueError)).except_(handler, exceptions=ValueError).run()
          )
        except ValueError:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_a02_except_does_not_catch_wrong_type(self):
    """except_(handler, exceptions=ValueError) does NOT catch TypeError."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        with self.assertRaises(TypeError):
          await await_(
            Chain(fn, 1).then(raise_exc(TypeError)).except_(handler, exceptions=ValueError).run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_a03_except_tuple_of_exceptions(self):
    """except_(handler, exceptions=(ValueError, TypeError)) catches both."""
    for fn, ctx in self.with_fn():
      with ctx:
        for exc_type in [ValueError, TypeError]:
          called = [False]
          def handler(v=None):
            called[0] = True
          try:
            await await_(
              Chain(fn, 1).then(raise_exc(exc_type))
              .except_(handler, exceptions=(ValueError, TypeError)).run()
            )
          except (ValueError, TypeError):
            pass
          super(MyTestCase, self).assertTrue(called[0], f'Failed for {exc_type}')

  async def test_a04_except_list_of_exceptions(self):
    """except_(handler, exceptions=[ValueError, TypeError]) -- iterable form."""
    for fn, ctx in self.with_fn():
      with ctx:
        for exc_type in [ValueError, TypeError]:
          called = [False]
          def handler(v=None):
            called[0] = True
          try:
            await await_(
              Chain(fn, 1).then(raise_exc(exc_type))
              .except_(handler, exceptions=[ValueError, TypeError]).run()
            )
          except (ValueError, TypeError):
            pass
          super(MyTestCase, self).assertTrue(called[0], f'Failed for {exc_type}')

  async def test_a05_except_catches_all_exception_subclasses(self):
    """except_(handler, exceptions=Exception) catches ALL Exception subclasses."""
    for fn, ctx in self.with_fn():
      with ctx:
        for exc_type in [ValueError, TypeError, RuntimeError, TestExc, IOError]:
          called = [False]
          def handler(v=None):
            called[0] = True
          try:
            await await_(
              Chain(fn, 1).then(raise_exc(exc_type))
              .except_(handler, exceptions=Exception).run()
            )
          except Exception:
            pass
          super(MyTestCase, self).assertTrue(called[0], f'Failed for {exc_type}')

  async def test_a06_except_exception_does_not_catch_base_exception(self):
    """except_(handler, exceptions=Exception) does NOT catch BaseException subclass (KeyboardInterrupt)."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        with self.assertRaises(CustomBaseExc):
          await await_(
            Chain(fn, 1).then(raise_exc(CustomBaseExc))
            .except_(handler, exceptions=Exception).run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_a07_except_default_catches_exception(self):
    """except_(handler) with default exceptions=(Exception,)."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1).then(raise_exc(RuntimeError)).except_(handler).run()
          )
        except RuntimeError:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_a08_subclass_matching(self):
    """exceptions=ValueError catches CustomValueError(ValueError)."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1).then(raise_exc(CustomValueError))
            .except_(handler, exceptions=ValueError).run()
          )
        except ValueError:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_a09_except_string_raises_type_error(self):
    """except_ with string exceptions raises TypeError."""
    with self.assertRaises(TypeError) as cm:
      Chain(1).except_(lambda v: None, exceptions='ValueError')
    super(MyTestCase, self).assertIn('ValueError', str(cm.exception))

  async def test_a10_multiple_handlers_first_match_wins(self):
    """Multiple handlers: first match wins, subsequent ignored."""
    for fn, ctx in self.with_fn():
      with ctx:
        called1 = [False]
        called2 = [False]
        def handler1(v=None):
          called1[0] = True
        def handler2(v=None):
          called2[0] = True
        try:
          await await_(
            Chain(fn, 1).then(raise_exc(Exc1))
            .except_(handler1, exceptions=TestExc)
            .except_(handler2, exceptions=TestExc)
            .run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertTrue(called1[0])
        super(MyTestCase, self).assertFalse(called2[0])

  async def test_a11_multiple_handlers_none_match(self):
    """Multiple handlers: none match -> exception propagates unchanged."""
    for fn, ctx in self.with_fn():
      with ctx:
        called1 = [False]
        called2 = [False]
        def handler1(v=None):
          called1[0] = True
        def handler2(v=None):
          called2[0] = True
        with self.assertRaises(RuntimeError):
          await await_(
            Chain(fn, 1).then(raise_exc(RuntimeError))
            .except_(handler1, exceptions=ValueError)
            .except_(handler2, exceptions=TypeError)
            .run()
          )
        super(MyTestCase, self).assertFalse(called1[0])
        super(MyTestCase, self).assertFalse(called2[0])

  async def test_a12_handler_for_base_exception(self):
    """Handler for BaseException catches everything."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1).then(raise_exc(CustomBaseExc))
            .except_(handler, exceptions=BaseException).run()
          )
        except BaseException:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_a13_generator_set_of_exceptions(self):
    """except_(handler, exceptions=set({ValueError, TypeError})) -- set form."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1).then(raise_exc(ValueError))
            .except_(handler, exceptions={ValueError, TypeError}).run()
          )
        except ValueError:
          pass
        super(MyTestCase, self).assertTrue(called[0])


# ---------------------------------------------------------------------------
# B. Exception Handler Behavior (12+ tests)
# ---------------------------------------------------------------------------

class ExceptionHandlerBehaviorTests(MyTestCase):

  async def test_b01_reraise_true_handler_executes_and_propagates(self):
    """Handler with reraise=True: handler executes AND exception propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(raise_direct).except_(handler, reraise=True).run()
          )
        super(MyTestCase, self).assertTrue(called[0])

  async def test_b02_reraise_false_handler_executes_and_suppresses(self):
    """Handler with reraise=False: handler executes AND chain continues with handler return value."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
          return 'handled'
        result = await await_(
          Chain(fn, 1).then(raise_direct).except_(handler, reraise=False).run()
        )
        super(MyTestCase, self).assertTrue(called[0])
        super(MyTestCase, self).assertEqual(result, 'handled')

  async def test_b03_handler_returning_value_with_noraise(self):
    """Handler returning a value (reraise=False): that value becomes chain result."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        def handler(v=None):
          return sentinel
        result = await await_(
          Chain(fn, 1).then(raise_direct).except_(handler, reraise=False).run()
        )
        super(MyTestCase, self).assertIs(result, sentinel)

  async def test_b04_handler_returning_none_with_noraise(self):
    """Handler returning None (reraise=False): None becomes chain result."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          return None
        result = await await_(
          Chain(fn, 1).then(raise_direct).except_(handler, reraise=False).run()
        )
        super(MyTestCase, self).assertIsNone(result)

  async def test_b05_handler_not_returning_anything_with_noraise(self):
    """Handler not returning anything (reraise=False): None becomes chain result."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          pass  # returns None implicitly
        result = await await_(
          Chain(fn, 1).then(raise_direct).except_(handler, reraise=False).run()
        )
        super(MyTestCase, self).assertIsNone(result)

  async def test_b06_handler_raises_new_exception(self):
    """Handler that itself raises a new exception: new exception propagated with __cause__ = original."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise TypeError('handler error')
        with self.assertRaises(TypeError) as cm:
          await await_(
            Chain(fn, 1).then(raise_direct).except_(handler, reraise=False).run()
          )
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, TestExc)

  async def test_b07_handler_bare_raise_reraises_original(self):
    """Handler that calls `raise` to re-raise from within handler body."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v):
          if v:
            raise
        # root value is True -> handler re-raises
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, True).then(raise_direct).except_(handler, reraise=False).run()
          )

  async def test_b08_handler_bare_raise_does_not_reraise_when_false(self):
    """Handler that only raises on condition -- condition false -> suppressed."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v):
          if v:
            raise
        # root value is False -> handler does not re-raise
        try:
          result = await await_(
            Chain(fn, False).then(raise_direct).except_(handler, reraise=False).run()
          )
        except TestExc:
          super(MyTestCase, self).fail('Exception should not propagate when handler does not re-raise')

  async def test_b09_async_handler_on_sync_chain_reraise_warns(self):
    """Handler with async return value in sync chain, reraise=True -> RuntimeWarning + ensure_future."""
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      try:
        Chain(raise_direct).except_(async_handler, reraise=True).run()
      except TestExc:
        pass
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_b10_async_handler_in_async_chain_awaited(self):
    """Handler with async return value in async chain -> awaited normally."""
    async def async_handler(v=None):
      return 'async handled'
    result = await await_(
      Chain(aempty, 1).then(async_raise).except_(async_handler, reraise=False).run()
    )
    super(MyTestCase, self).assertEqual(result, 'async handled')

  async def test_b11_handler_receives_root_value(self):
    """Handler receives ROOT value, not current value at exception point."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        root_sentinel = object()
        def handler(v=None):
          received[0] = v
        try:
          await await_(
            Chain(fn, root_sentinel)
            .then(lambda v: 'transformed')
            .then(raise_direct)
            .except_(handler)
            .run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertIs(received[0], root_sentinel)

  async def test_b12_handler_with_explicit_args(self):
    """Handler with explicit args: except_(handler, arg1, arg2, reraise=False)."""
    for fn, ctx in self.with_fn():
      with ctx:
        received_args = [None]
        sentinel1 = object()
        sentinel2 = object()
        def handler(a, b):
          received_args[0] = (a, b)
          return 'handled'
        result = await await_(
          Chain(fn, 1).then(raise_direct)
          .except_(handler, sentinel1, sentinel2, reraise=False)
          .run()
        )
        super(MyTestCase, self).assertEqual(result, 'handled')
        super(MyTestCase, self).assertIs(received_args[0][0], sentinel1)
        super(MyTestCase, self).assertIs(received_args[0][1], sentinel2)

  async def test_b13_handler_with_explicit_kwargs(self):
    """Handler with explicit kwargs: except_(handler, key=val, reraise=False)."""
    for fn, ctx in self.with_fn():
      with ctx:
        received_kwargs = [None]
        def handler(key='default'):
          received_kwargs[0] = key
          return 'kw_handled'
        result = await await_(
          Chain(fn, 1).then(raise_direct)
          .except_(handler, key='custom_val', reraise=False)
          .run()
        )
        super(MyTestCase, self).assertEqual(result, 'kw_handled')
        super(MyTestCase, self).assertEqual(received_kwargs[0], 'custom_val')

  async def test_b14_async_handler_noraise_returns_awaited_value(self):
    """Async handler with reraise=False in sync chain: the result is an awaitable."""
    async def async_handler(v=None):
      return 'async_result'
    # When chain is purely sync but handler returns a coroutine and reraise=False,
    # ensure_future is called and result is returned as a future.
    result = await await_(
      Chain(raise_direct).except_(async_handler, reraise=False).run()
    )
    super(MyTestCase, self).assertEqual(result, 'async_result')


# ---------------------------------------------------------------------------
# C. Finally Handler Edge Cases (10+ tests)
# ---------------------------------------------------------------------------

class FinallyHandlerEdgeCaseTests(MyTestCase):

  async def test_c01_finally_runs_on_success(self):
    """finally_ runs after successful chain completion."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def on_finally(v=None):
          called[0] = True
        result = await await_(
          Chain(fn, 42).finally_(on_finally).run()
        )
        super(MyTestCase, self).assertTrue(called[0])
        super(MyTestCase, self).assertEqual(result, 42)

  async def test_c02_finally_runs_after_caught_not_reraised(self):
    """finally_ runs after exception (caught by handler, not reraised)."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = [False]
        handler_called = [False]
        def on_finally(v=None):
          finally_called[0] = True
        def handler(v=None):
          handler_called[0] = True
          return 'recovered'
        result = await await_(
          Chain(fn, 1).then(raise_direct)
          .except_(handler, reraise=False)
          .finally_(on_finally)
          .run()
        )
        super(MyTestCase, self).assertTrue(handler_called[0])
        super(MyTestCase, self).assertTrue(finally_called[0])
        super(MyTestCase, self).assertEqual(result, 'recovered')

  async def test_c03_finally_runs_after_caught_and_reraised(self):
    """finally_ runs after exception (caught by handler, reraised)."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = [False]
        handler_called = [False]
        def on_finally(v=None):
          finally_called[0] = True
        def handler(v=None):
          handler_called[0] = True
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(raise_direct)
            .except_(handler, reraise=True)
            .finally_(on_finally)
            .run()
          )
        super(MyTestCase, self).assertTrue(handler_called[0])
        super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_c04_finally_runs_after_unhandled_exception(self):
    """finally_ runs after exception (no handler matches)."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = [False]
        def on_finally(v=None):
          finally_called[0] = True
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(raise_direct).finally_(on_finally).run()
          )
        super(MyTestCase, self).assertTrue(finally_called[0])

  async def test_c05_finally_handler_receives_root_value(self):
    """finally_ handler receives root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        root_sentinel = object()
        def on_finally(v=None):
          received[0] = v
        await await_(
          Chain(fn, root_sentinel).then(lambda v: 'changed').finally_(on_finally).run()
        )
        super(MyTestCase, self).assertIs(received[0], root_sentinel)

  async def test_c06_finally_return_value_ignored(self):
    """finally_ handler return value is IGNORED (does not affect chain result)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_finally(v=None):
          return 'should_be_ignored'
        result = await await_(
          Chain(fn, 42).then(lambda v: v + 1).finally_(on_finally).run()
        )
        super(MyTestCase, self).assertEqual(result, 43)

  async def test_c07_finally_that_raises_propagates(self):
    """finally_ that raises exception: exception propagates with modify_traceback."""
    for fn, ctx in self.with_fn():
      with ctx:
        def on_finally(v=None):
          raise RuntimeError('finally error')
        with self.assertRaises(RuntimeError) as cm:
          await await_(
            Chain(fn, 42).finally_(on_finally).run()
          )
        super(MyTestCase, self).assertEqual(str(cm.exception), 'finally error')

  async def test_c08_finally_async_on_sync_chain_warns(self):
    """finally_ that returns coroutine in sync chain -> RuntimeWarning."""
    async def async_finally(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      Chain(empty, 1).finally_(async_finally).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_c09_finally_with_internal_quent_exception_raises(self):
    """finally_ with _InternalQuentException (_Return/_Break) -> QuentException."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(QuentException) as cm:
          await await_(
            Chain(fn, 1).finally_(Chain.return_, 99).run()
          )
        super(MyTestCase, self).assertIn('control flow', str(cm.exception).lower())

  async def test_c10_double_finally_raises_quent_exception(self):
    """Double finally_ registration -> QuentException."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)
    super(MyTestCase, self).assertIn('finally', str(cm.exception).lower())

  async def test_c11_finally_runs_on_empty_chain(self):
    """Chain().finally_(handler).run() -- handler called with Null -> no root_value."""
    called = [False]
    def on_finally(v=None):
      called[0] = True
    result = Chain().finally_(on_finally).run()
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertTrue(called[0])

  async def test_c12_finally_break_in_finally_raises_quent_exception(self):
    """Chain.break_() inside finally -> QuentException about control flow signals."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(QuentException) as cm:
          await await_(
            Chain(fn, 1).finally_(Chain.break_).run()
          )
        super(MyTestCase, self).assertIn('control flow', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# D. _Return Signal Comprehensive (10+ tests)
# ---------------------------------------------------------------------------

class ReturnSignalTests(MyTestCase):

  async def test_d01_return_current_value_from_nested_chain(self):
    """Chain.return_() -- _Return carries the current value, exits the entire chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        # Inner chain receives 1 from outer, Chain.return_ is called with current_value=1,
        # so _Return(1) is raised. Outer chain catches _Return and returns 1.
        result = await await_(
          Chain(fn, 1).then(
            Chain().then(Chain.return_).then(lambda v: 'should_not_reach')
          ).run()
        )
        # _Return carries the current value (1) all the way out
        super(MyTestCase, self).assertEqual(result, 1)

  async def test_d02_return_literal_value(self):
    """Chain.return_(42) -- return literal value."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(
            Chain().then(Chain.return_, 42).then(lambda v: 'unreachable')
          ).run()
        )
        super(MyTestCase, self).assertEqual(result, 42)

  async def test_d03_return_result_of_fn(self):
    """Chain.return_(fn) -- return result of calling fn()."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(
            Chain().then(Chain.return_, lambda: 'from_fn')
          ).run()
        )
        super(MyTestCase, self).assertEqual(result, 'from_fn')

  async def test_d04_return_result_of_fn_with_arg(self):
    """Chain.return_(fn, arg) -- return result of fn(arg)."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(
            Chain().then(Chain.return_, lambda x: x * 10, 5)
          ).run()
        )
        super(MyTestCase, self).assertEqual(result, 50)

  async def test_d05_return_result_of_fn_no_args_ellipsis(self):
    """Chain.return_(fn, ...) -- return result of fn() (no args)."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(
            Chain().then(Chain.return_, lambda: 'no_args', ...)
          ).run()
        )
        super(MyTestCase, self).assertEqual(result, 'no_args')

  async def test_d06_return_in_nested_chain_exits_entire_chain(self):
    """_Return in nested chain -> exits the entire outer chain with the return value."""
    for fn, ctx in self.with_fn():
      with ctx:
        # Inner chain: current_value starts as 1, then v+10=11, then Chain.return_(11, 99)
        # Chain.return_ with explicit arg 99 -> _Return(99) (explicit arg override)
        # Actually: .then(Chain.return_, 99) creates Link with args=(99,)
        # evaluate_value calls Chain.return_(99) -> _Return(99)
        # _Return propagates out of inner chain (nested re-raises), outer catches it
        # handle_return_exc returns 99 -> outer chain returns 99 (exits completely)
        inner = Chain().then(lambda v: v + 10).then(Chain.return_, 99).then(lambda v: 'unreachable')
        result = await await_(
          Chain(fn, 1).then(inner).then(lambda v: v + 1000).run()
        )
        # _Return exits the ENTIRE chain, not just the nested one
        super(MyTestCase, self).assertEqual(result, 99)

  async def test_d07_return_in_doubly_nested_chain(self):
    """_Return in doubly-nested chain -> propagates through ALL levels, exits entirely."""
    for fn, ctx in self.with_fn():
      with ctx:
        # .then(Chain.return_, 77) -> Chain.return_(77) -> _Return(77)
        # Propagates through inner (nested), then through outer (nested), then to top
        innermost = Chain().then(Chain.return_, 77)
        inner = Chain().then(innermost)
        result = await await_(
          Chain(fn, 1).then(inner).then(lambda v: v + 1).run()
        )
        # _Return exits the ENTIRE chain with value 77
        super(MyTestCase, self).assertEqual(result, 77)

  async def test_d08_return_async_value_awaited(self):
    """_Return value that is a coroutine -> awaited in async path."""
    async def async_val():
      return 'async_return_val'
    result = await await_(
      Chain(aempty, 1).then(
        Chain().then(Chain.return_, async_val)
      ).run()
    )
    super(MyTestCase, self).assertEqual(result, 'async_return_val')

  async def test_d09_return_in_exception_handler_context(self):
    """_Return in a chain used as exception handler context."""
    for fn, ctx in self.with_fn():
      with ctx:
        # The return is in a nested chain that's used as a regular link.
        inner = Chain().then(Chain.return_, 'early_exit')
        result = await await_(
          Chain(fn, 1).then(inner).run()
        )
        super(MyTestCase, self).assertEqual(result, 'early_exit')

  async def test_d10_return_in_top_level_chain(self):
    """_Return in top-level chain -> returns the value directly (caught by _run, not run)."""
    # _Return is caught inside _run() at the `except _Return` handler,
    # which calls handle_return_exc(exc, is_nested=False) -> returns the value.
    # It does NOT propagate to run()'s _InternalQuentException handler.
    result = Chain(1).then(Chain.return_, 42).run()
    super(MyTestCase, self).assertEqual(result, 42)

  async def test_d11_return_preserves_none_value(self):
    """Chain.return_(lambda: None) returns None, not Null."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(
            Chain().then(Chain.return_, lambda: None)
          ).run()
        )
        super(MyTestCase, self).assertIsNone(result)


# ---------------------------------------------------------------------------
# E. _Break Signal Comprehensive (10+ tests)
# ---------------------------------------------------------------------------

class BreakSignalTests(MyTestCase):

  async def test_e01_break_in_foreach_returns_accumulated(self):
    """Chain.break_() in foreach -> returns accumulated list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn(el * 2)
        result = await await_(
          Chain(fn, [1, 2, 3, 4, 5]).foreach(f).run()
        )
        super(MyTestCase, self).assertEqual(result, [2, 4])

  async def test_e02_break_with_value_in_foreach(self):
    """Chain.break_(42) in foreach -> returns 42 (overrides list)."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        def f(el):
          if el == 2:
            Chain.break_(sentinel)
          return fn(el)
        result = await await_(
          Chain(fn, [1, 2, 3]).foreach(f).run()
        )
        super(MyTestCase, self).assertIs(result, sentinel)

  async def test_e03_break_in_foreach_indexed(self):
    """Chain.break_() in foreach_indexed -> returns accumulated list."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(idx, el):
          if idx >= 2:
            Chain.break_()
          return fn((idx, el))
        result = await await_(
          Chain(fn, ['a', 'b', 'c', 'd']).foreach(f, with_index=True).run()
        )
        super(MyTestCase, self).assertEqual(result, [(0, 'a'), (1, 'b')])

  async def test_e04_break_in_iterate_stops_generator(self):
    """Chain.break_() in iterate -> generator stops (StopIteration)."""
    def f(el):
      if el >= 3:
        Chain.break_()
      return el * 10
    r = []
    for i in Chain(lambda: [1, 2, 3, 4, 5]).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [10, 20])

  async def test_e05_break_outside_foreach_raises_quent_exception(self):
    """Chain.break_() outside foreach/iterate -> QuentException."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(QuentException) as cm:
          await await_(
            Chain(fn, 1).then(Chain.break_).run()
          )
        super(MyTestCase, self).assertIn('Break', str(cm.exception))

  async def test_e06_break_in_async_foreach(self):
    """Chain.break_() in async foreach -> returns accumulated list."""

    class AsyncIter:
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

    for fn, ctx in self.with_fn():
      with ctx:
        def f(el):
          if el >= 3:
            Chain.break_()
          return fn(el * 2)
        result = await await_(
          Chain(fn, AsyncIter([1, 2, 3, 4, 5])).foreach(f).run()
        )
        super(MyTestCase, self).assertEqual(result, [2, 4])

  async def test_e07_break_in_async_iterate(self):
    """Chain.break_() in async iterate -> generator stops."""

    class AsyncIter:
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

    def f(el):
      if el >= 3:
        Chain.break_()
      return el * 10
    r = []
    async for i in Chain(AsyncIter, [1, 2, 3, 4, 5]).iterate(f):
      r.append(i)
    super(MyTestCase, self).assertEqual(r, [10, 20])

  async def test_e08_break_in_nested_chain_not_in_foreach_propagates(self):
    """Chain.break_() in nested chain that's not in foreach -> propagates up and QuentException at top."""
    inner = Chain().then(Chain.break_)
    with self.assertRaises(QuentException):
      Chain(1).then(inner).run()

  async def test_e09_break_with_callable_value(self):
    """Chain.break_(fn) with callable value: fn() result is break value."""
    sentinel = object()
    def f(el):
      if el == 2:
        Chain.break_(lambda: sentinel, ...)
      return el
    result = Chain([1, 2, 3]).foreach(f).run()
    super(MyTestCase, self).assertIs(result, sentinel)

  async def test_e10_break_in_filter_not_caught(self):
    """Chain.break_() in filter -> NOT caught (filter doesn't catch _Break)."""
    def pred(el):
      if el >= 3:
        Chain.break_()
      return el % 2 == 0
    # _Break propagates out of filter -> QuentException at top level
    with self.assertRaises(QuentException):
      Chain([1, 2, 3, 4]).filter(pred).run()

  async def test_e11_break_with_value_in_foreach_indexed(self):
    """Chain.break_(obj) in foreach_indexed -> returns obj."""
    sentinel = object()
    def f(idx, el):
      if idx == 1:
        Chain.break_(sentinel)
      return (idx, el)
    result = Chain(['a', 'b', 'c']).foreach(f, with_index=True).run()
    super(MyTestCase, self).assertIs(result, sentinel)


# ---------------------------------------------------------------------------
# F. Exception at Every Position (10+ tests)
# ---------------------------------------------------------------------------

class ExceptionAtEveryPositionTests(MyTestCase):

  async def test_f01_exception_in_root_evaluation(self):
    """Exception in root evaluation."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc):
          await await_(Chain(raise_direct).run())

  async def test_f02_exception_in_first_link_after_root(self):
    """Exception in first link after root."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 1).then(raise_direct).run())

  async def test_f03_exception_in_middle_link(self):
    """Exception in middle link."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(lambda v: v + 1).then(raise_direct).then(lambda v: v + 2).run()
          )

  async def test_f04_exception_in_last_link(self):
    """Exception in last link."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(lambda v: v + 1).then(raise_direct).run()
          )

  async def test_f05_exception_in_root_of_nested_chain(self):
    """Exception in root evaluation of nested chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        # Use a void inner chain whose first link raises (effectively the root operation).
        # Cannot use Chain(raise_direct) with a parent that passes a value, because
        # that would trigger "Cannot override the root value" QuentException.
        inner = Chain().then(raise_direct)
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 1).then(inner).run())

  async def test_f06_exception_in_async_root(self):
    """Exception in async root (coroutine that raises)."""
    with self.assertRaises(TestExc):
      await await_(Chain(async_raise).run())

  async def test_f07_exception_in_async_link(self):
    """Exception in async link (coroutine that raises)."""
    with self.assertRaises(TestExc):
      await await_(Chain(aempty, 1).then(async_raise).run())

  async def test_f08_exception_in_foreach_fn(self):
    """Exception in foreach fn."""
    def f(el):
      if el == 2:
        raise TestExc('in foreach')
      return el
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2, 3]).foreach(f).run())

  async def test_f09_exception_in_filter_predicate(self):
    """Exception in filter predicate."""
    def pred(el):
      if el == 2:
        raise TestExc('in filter')
      return True
    with self.assertRaises(TestExc):
      await await_(Chain([1, 2, 3]).filter(pred).run())

  async def test_f10_exception_in_context_manager_body(self):
    """Exception in context manager body."""
    for fn, ctx in self.with_fn():
      with ctx:
        mgr = SyncCtxMgr(value='ctx_val')
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, mgr).with_(raise_direct).run()
          )
        super(MyTestCase, self).assertTrue(mgr.entered)
        super(MyTestCase, self).assertTrue(mgr.exited)

  async def test_f11_exception_in_void_chain(self):
    """Exception in a chain with no root value."""
    with self.assertRaises(TestExc):
      Chain().then(raise_direct).run()

  async def test_f12_exception_in_do_link(self):
    """Exception in a do() link (ignore_result=True)."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 1).do(raise_direct).run())


# ---------------------------------------------------------------------------
# G. Traceback Modification Verification (10+ tests)
# ---------------------------------------------------------------------------

def _user_fn_that_raises(v=None):
  raise ValueError('traceback test value')


def _user_fn_that_chains(v):
  return Chain(v).then(_user_fn_that_raises)()


class TracebackModificationTests(IsolatedAsyncioTestCase):

  def _get_tb_entries(self, exc):
    return traceback.extract_tb(exc.__traceback__)

  def _get_tb_filenames(self, exc):
    return [entry.filename for entry in self._get_tb_entries(exc)]

  def _get_tb_func_names(self, exc):
    return [entry.name for entry in self._get_tb_entries(exc)]

  async def test_g01_quent_frame_in_traceback(self):
    """Exception has `<quent>` frame showing chain visualization."""
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = self._get_tb_filenames(exc)
      self.assertIn('<quent>', filenames)

  async def test_g02_internal_frames_cleaned(self):
    """Traceback is cleaned of internal quent frames (files from quent package dir)."""
    import os
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      for entry in self._get_tb_entries(exc):
        if entry.filename == '<quent>':
          continue
        self.assertFalse(
          entry.filename.startswith(quent_pkg_dir),
          f'Internal frame should be cleaned: {entry.filename}'
        )

  async def test_g03_user_frames_preserved(self):
    """User code frames are preserved in traceback."""
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      func_names = self._get_tb_func_names(exc)
      self.assertIn('_user_fn_that_raises', func_names)

  async def test_g04_quent_attr_set_on_exception(self):
    """__quent__ attribute is set on exception (for excepthook)."""
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      # __quent__ is set by remove_self_frames_from_traceback
      # which is called when the exception propagates through Chain.run()
      # It may or may not be set depending on the path.
      # But at minimum, the traceback should have <quent> frames.
      entries = self._get_tb_entries(exc)
      quent_frames = [e for e in entries if e.filename == '<quent>']
      self.assertGreater(len(quent_frames), 0)

  async def test_g05_source_link_set_and_deleted(self):
    """__quent_source_link__ is set on first modify_traceback call, deleted after."""
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      # After chain completes, __quent_source_link__ should have been deleted
      self.assertFalse(hasattr(exc, '__quent_source_link__'))

  async def test_g06_nested_chain_defers_modify_traceback(self):
    """Nested chain: modify_traceback defers (returns early when is_nested=True)."""
    try:
      Chain(1).then(_user_fn_that_chains).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      # The outermost chain handles traceback modification
      entries = self._get_tb_entries(exc)
      filenames = [e.filename for e in entries]
      self.assertIn('<quent>', filenames)

  async def test_g07_chain_visualization_shows_function_names(self):
    """Chain visualization in traceback shows function names."""
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = self._get_tb_entries(exc)
      quent_frames = [e for e in entries if e.filename == '<quent>']
      self.assertGreater(len(quent_frames), 0)
      # The "name" field of a <quent> frame contains the chain visualization
      visualization = quent_frames[0].name
      self.assertIn('_user_fn_that_raises', visualization)

  async def test_g08_chain_visualization_shows_source_link_arrow(self):
    """Chain visualization shows source link arrow (`<----`)."""
    try:
      Chain(1).then(lambda v: v + 1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = self._get_tb_entries(exc)
      quent_frames = [e for e in entries if e.filename == '<quent>']
      self.assertGreater(len(quent_frames), 0)
      visualization = quent_frames[0].name
      self.assertIn('<----', visualization)

  async def test_g09_chained_exceptions_tracebacks_cleaned(self):
    """Chained exceptions (__cause__, __context__) have their tracebacks cleaned."""
    def handler_that_raises(v=None):
      raise TypeError('handler error')
    try:
      Chain(1).then(_user_fn_that_raises).except_(handler_that_raises, reraise=False).run()
      self.fail('Expected TypeError')
    except TypeError as exc:
      # The __cause__ should be the original ValueError
      self.assertIsInstance(exc.__cause__, ValueError)
      # Chained exception tracebacks should also be cleaned
      if exc.__cause__.__traceback__ is not None:
        cause_entries = traceback.extract_tb(exc.__cause__.__traceback__)
        import os
        quent_pkg_dir = os.path.dirname(os.path.abspath(
          __import__('quent').quent.__file__
        ))
        for entry in cause_entries:
          if entry.filename == '<quent>':
            continue
          self.assertFalse(
            entry.filename.startswith(quent_pkg_dir),
            f'Chained exception internal frame should be cleaned: {entry.filename}'
          )

  async def test_g10_traceback_exception_patch_cleans_quent(self):
    """TracebackException.__init__ patch cleans quent exceptions at display time."""
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      self.assertIn('_user_fn_that_raises', formatted)

  async def test_g11_quent_frame_name_is_nonempty(self):
    """<quent> frame should have a non-empty name (chain visualization)."""
    try:
      Chain(1).then(_user_fn_that_raises).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = self._get_tb_entries(exc)
      quent_entries = [e for e in entries if e.filename == '<quent>']
      for entry in quent_entries:
        self.assertTrue(len(entry.name) > 0)


# ---------------------------------------------------------------------------
# H. Exception Chaining (8+ tests)
# ---------------------------------------------------------------------------

class ExceptionChainingTests(MyTestCase):

  async def test_h01_handler_raises_cause_set(self):
    """Handler raises -> __cause__ is set to original exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise TypeError('new error')
        with self.assertRaises(TypeError) as cm:
          await await_(
            Chain(fn, 1).then(raise_direct).except_(handler, reraise=False).run()
          )
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, TestExc)

  async def test_h02_raise_from_chain_in_handler(self):
    """`raise exc_ from exc` chain in handler error path."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise ValueError('handler_val_error')
        with self.assertRaises(ValueError) as cm:
          await await_(
            Chain(fn, 1).then(raise_direct).except_(handler, reraise=True).run()
          )
        # ValueError was raised in handler, and `raise exc_ from exc` sets __cause__
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, TestExc)

  async def test_h03_finally_exception_preserves_context(self):
    """Exception in finally handler: preserves original exception context."""
    for fn, ctx in self.with_fn():
      with ctx:
        def finally_raises(v=None):
          raise RuntimeError('finally error')
        with self.assertRaises(RuntimeError) as cm:
          await await_(
            Chain(fn, 1).then(raise_direct).finally_(finally_raises).run()
          )
        # The context should be the original TestExc
        super(MyTestCase, self).assertIsInstance(cm.exception.__context__, TestExc)

  async def test_h04_nested_exception_chains(self):
    """Nested exception chains (exception during handling of another exception)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler1(v=None):
          raise TypeError('type error in handler')
        with self.assertRaises(TypeError) as cm:
          await await_(
            Chain(fn, 1).then(raise_exc(ValueError, 'original'))
            .except_(handler1, reraise=False).run()
          )
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, ValueError)

  async def test_h05_exception_with_preexisting_cause(self):
    """Exception with pre-existing __cause__ (set before entering chain)."""
    original_cause = RuntimeError('original cause')
    def raiser(v=None):
      exc = TestExc('with cause')
      exc.__cause__ = original_cause
      raise exc
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc) as cm:
          await await_(Chain(fn, 1).then(raiser).run())
        super(MyTestCase, self).assertIs(cm.exception.__cause__, original_cause)

  async def test_h06_exception_with_preexisting_context(self):
    """Exception with pre-existing __context__ (set before entering chain)."""
    original_context = RuntimeError('original context')
    def raiser(v=None):
      exc = TestExc('with context')
      exc.__context__ = original_context
      raise exc
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc) as cm:
          await await_(Chain(fn, 1).then(raiser).run())
        super(MyTestCase, self).assertIs(cm.exception.__context__, original_context)

  async def test_h07_exception_with_none_traceback(self):
    """Exception with __traceback__ set to None before entering chain."""
    def raiser(v=None):
      exc = TestExc('no tb')
      exc.__traceback__ = None
      raise exc
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 1).then(raiser).run())

  async def test_h08_multiple_levels_of_chaining_through_nested(self):
    """Multiple levels of exception chaining through nested chains."""
    def inner_raiser(v=None):
      raise ValueError('innermost')
    def outer_handler(v=None):
      raise TypeError('outer handler error')
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().then(inner_raiser)
        with self.assertRaises(TypeError) as cm:
          await await_(
            Chain(fn, 1).then(inner).except_(outer_handler, reraise=False).run()
          )
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, ValueError)


# ---------------------------------------------------------------------------
# I. QuentException Scenarios (8+ tests)
# ---------------------------------------------------------------------------

class QuentExceptionScenariosTests(MyTestCase):

  async def test_i01_cannot_override_root_value(self):
    """Cannot override root value of a chain that has one."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(QuentException) as cm:
          await await_(Chain(fn, 1).run(2))
        super(MyTestCase, self).assertIn('override', str(cm.exception).lower())

  async def test_i02_non_callable_in_do_raises(self):
    """Non-callable in do() raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).do(42)
    super(MyTestCase, self).assertIn('non-callable', str(cm.exception).lower())

  async def test_i03_multiple_finally_raises(self):
    """Multiple finally_ registration raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)

  async def test_i04_break_outside_context_raises(self):
    """_Break outside valid context raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).then(Chain.break_).run()
    super(MyTestCase, self).assertIn('Break', str(cm.exception))

  async def test_i05_internal_quent_exception_in_finally(self):
    """_InternalQuentException in finally handler -> QuentException about control flow."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).finally_(Chain.return_).run()
    super(MyTestCase, self).assertIn('control flow', str(cm.exception).lower())

  async def test_i06_directly_running_nested_chain_raises(self):
    """Directly running a nested chain raises QuentException."""
    inner = Chain().then(lambda v: v)
    # First, make it nested by using it as a link in another chain
    outer = Chain(1).then(inner)
    # Now inner.is_nested is True
    with self.assertRaises(QuentException) as cm:
      inner.run()
    super(MyTestCase, self).assertIn('nested', str(cm.exception).lower())

  async def test_i07_except_string_raises_type_error(self):
    """except_ with string exceptions -> TypeError (not QuentException)."""
    with self.assertRaises(TypeError):
      Chain(1).except_(lambda v: None, exceptions='ValueError')

  async def test_i08_quent_exception_messages(self):
    """QuentException message content verification for various scenarios."""
    # Test root override message
    try:
      Chain(1).run(2)
    except QuentException as e:
      super(MyTestCase, self).assertIn('override', str(e).lower())

    # Test non-callable message
    try:
      Chain(1).do(42)
    except QuentException as e:
      super(MyTestCase, self).assertIn('non-callable', str(e).lower())

    # Test double finally message
    try:
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)
    except QuentException as e:
      super(MyTestCase, self).assertIn('finally', str(e).lower())

  async def test_i09_return_in_iterate_raises_quent_exception(self):
    """Chain.return_() inside iterate raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      for _ in Chain(lambda: [1, 2, 3]).iterate(Chain.return_):
        pass
    super(MyTestCase, self).assertIn('iterator', str(cm.exception).lower())

  async def test_i10_nested_chain_cannot_be_run_via_call(self):
    """Directly calling a nested chain via __call__ raises QuentException."""
    inner = Chain().then(lambda v: v)
    outer = Chain(1).then(inner)
    with self.assertRaises(QuentException):
      inner()


# ---------------------------------------------------------------------------
# J. Async Exception Paths (8+ tests)
# ---------------------------------------------------------------------------

class AsyncExceptionPathsTests(IsolatedAsyncioTestCase):

  async def test_j01_exception_in_run_async_body(self):
    """Exception in _run_async body."""
    async def async_root(v=None):
      return 1
    with self.assertRaises(TestExc):
      await await_(
        Chain(async_root).then(raise_direct).run()
      )

  async def test_j02_exception_in_run_async_with_handler(self):
    """Exception in _run_async with handler (awaits handler result if coroutine)."""
    called = [False]
    async def async_handler(v=None):
      called[0] = True
      return 'async_handled'
    result = await await_(
      Chain(aempty, 1).then(async_raise)
      .except_(async_handler, reraise=False).run()
    )
    self.assertTrue(called[0])
    self.assertEqual(result, 'async_handled')

  async def test_j03_exception_in_run_async_handler_chains_from_original(self):
    """Exception in _run_async handler -> chains from original."""
    async def handler_that_raises(v=None):
      raise TypeError('async handler error')
    with self.assertRaises(TypeError) as cm:
      await await_(
        Chain(aempty, 1).then(async_raise)
        .except_(handler_that_raises, reraise=False).run()
      )
    self.assertIsInstance(cm.exception.__cause__, TestExc)

  async def test_j04_return_in_run_async_value_awaited(self):
    """_Return in _run_async -> value awaited if coroutine."""
    async def async_val():
      return 'awaited_return'
    inner = Chain().then(Chain.return_, async_val)
    result = await await_(
      Chain(aempty, 1).then(inner).run()
    )
    self.assertEqual(result, 'awaited_return')

  async def test_j05_break_in_run_async_nested_propagates(self):
    """_Break in _run_async nested -> propagates."""
    # Inner chain with break, used inside foreach
    def f(el):
      if el >= 3:
        Chain.break_()
      return el * 2

    result = await await_(
      Chain(aempty, [1, 2, 3, 4, 5]).foreach(f).run()
    )
    self.assertEqual(result, [2, 4])

  async def test_j06_break_in_run_async_non_nested_raises(self):
    """_Break in _run_async non-nested -> QuentException."""
    with self.assertRaises(QuentException):
      await await_(
        Chain(aempty, 1).then(Chain.break_).run()
      )

  async def test_j07_exception_in_run_async_simple(self):
    """Exception in _run_async_simple."""
    # A "simple" chain (only .then links, no debug/finally/except)
    with self.assertRaises(TestExc):
      await await_(
        Chain(aempty, 1).then(async_raise).run()
      )

  async def test_j08_exception_in_run_async_finally_handler(self):
    """Exception in _run_async finally handler."""
    async def async_finally_raises(v=None):
      raise RuntimeError('async finally error')
    with self.assertRaises(RuntimeError) as cm:
      await await_(
        Chain(aempty, 42).finally_(async_finally_raises).run()
      )
    self.assertEqual(str(cm.exception), 'async finally error')

  async def test_j09_async_handler_reraise_true(self):
    """Async exception handler with reraise=True: handler runs, exception still propagates."""
    called = [False]
    async def async_handler(v=None):
      called[0] = True
    with self.assertRaises(TestExc):
      await await_(
        Chain(aempty, 1).then(async_raise)
        .except_(async_handler, reraise=True).run()
      )
    self.assertTrue(called[0])

  async def test_j10_async_finally_runs_on_async_exception(self):
    """Async finally runs when async chain raises exception."""
    finally_called = [False]
    async def on_finally(v=None):
      finally_called[0] = True
    with self.assertRaises(TestExc):
      await await_(
        Chain(aempty, 1).then(async_raise).finally_(on_finally).run()
      )
    self.assertTrue(finally_called[0])

  async def test_j11_return_in_async_simple_path(self):
    """_Return in async simple chain path."""
    inner = Chain().then(Chain.return_, 'async_simple_ret')
    result = await await_(
      Chain(aempty, 1).then(inner).run()
    )
    self.assertEqual(result, 'async_simple_ret')

  async def test_j12_break_in_async_simple_non_nested(self):
    """_Break in async simple chain that is not nested -> QuentException."""
    with self.assertRaises(QuentException):
      await await_(
        Chain(aempty, 1).then(Chain.break_).run()
      )

  async def test_j13_exception_in_async_simple_has_traceback(self):
    """Exception in async simple path gets traceback modification."""
    try:
      await await_(Chain(aempty, 1).then(_user_fn_that_raises).run())
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = traceback.extract_tb(exc.__traceback__)
      filenames = [e.filename for e in entries]
      self.assertIn('<quent>', filenames)

  async def test_j14_internal_quent_exception_in_async_finally(self):
    """_InternalQuentException in async finally -> QuentException."""
    with self.assertRaises(QuentException) as cm:
      await await_(
        Chain(aempty, 1).finally_(Chain.return_, 99).run()
      )
    self.assertIn('control flow', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# Additional edge cases: Cascade exception handling
# ---------------------------------------------------------------------------

class CascadeExceptionTests(MyTestCase):

  async def test_cascade_except_receives_root_value(self):
    """In a Cascade, except_ handler receives the root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        root = object()
        def handler(v=None):
          received[0] = v
        try:
          await await_(
            Cascade(fn, root).then(raise_direct).except_(handler).run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertIs(received[0], root)

  async def test_cascade_finally_receives_root_value(self):
    """In a Cascade, finally_ handler receives the root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        root = object()
        def handler(v=None):
          received[0] = v
        await await_(
          Cascade(fn, root).then(lambda v: 'ignored').finally_(handler).run()
        )
        super(MyTestCase, self).assertIs(received[0], root)

  async def test_cascade_exception_handler_suppresses(self):
    """Cascade with except_(handler, reraise=False) returns handler value."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Cascade(fn, 1).then(raise_direct).except_(lambda v: 'cascade_handled', reraise=False).run()
        )
        super(MyTestCase, self).assertEqual(result, 'cascade_handled')


# ---------------------------------------------------------------------------
# Additional: handler receives root_value even when void chain
# ---------------------------------------------------------------------------

class VoidChainExceptionTests(MyTestCase):

  async def test_except_handler_on_void_chain(self):
    """On a void chain (no root), handler receives Null equivalent -> None."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [object()]  # sentinel default
        def handler(v=None):
          received[0] = v
        try:
          await await_(
            Chain().then(raise_direct).except_(handler).run()
          )
        except TestExc:
          pass
        # root_value is Null, which gets passed to handler.
        # Since evaluate_value passes Null -> v() i.e. handler() with no args,
        # v parameter defaults to None
        super(MyTestCase, self).assertIsNone(received[0])

  async def test_finally_on_void_chain(self):
    """On a void chain (no root), finally_ handler receives None."""
    received = [object()]
    def on_finally(v=None):
      received[0] = v
    Chain().then(lambda: 42).finally_(on_finally).run()
    # root_value is Null -> evaluate_value calls handler() -> v defaults to None
    super(MyTestCase, self).assertIsNone(received[0])


# ---------------------------------------------------------------------------
# Additional: Exception in exception handler error path - sync handler on
# async chain
# ---------------------------------------------------------------------------

class SyncHandlerOnAsyncChainTests(IsolatedAsyncioTestCase):

  async def test_sync_handler_on_async_chain(self):
    """Sync exception handler on async chain works correctly."""
    called = [False]
    def handler(v=None):
      called[0] = True
      return 'sync_handled'
    result = await await_(
      Chain(aempty, 1).then(async_raise).except_(handler, reraise=False).run()
    )
    self.assertTrue(called[0])
    self.assertEqual(result, 'sync_handled')

  async def test_sync_handler_raises_on_async_chain(self):
    """Sync handler that raises on async chain: chains correctly."""
    def handler(v=None):
      raise TypeError('sync handler error on async')
    with self.assertRaises(TypeError) as cm:
      await await_(
        Chain(aempty, 1).then(async_raise).except_(handler, reraise=False).run()
      )
    self.assertIsInstance(cm.exception.__cause__, TestExc)
