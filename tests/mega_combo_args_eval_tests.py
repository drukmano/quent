"""Comprehensive tests for the eval code system and argument passing combinations.

Tests every possible combination of:
  - _determine_eval_code matrix (callable x args x kwargs x allow_literal)
  - then() argument passing variants
  - do() argument passing variants
  - except_() argument passing
  - finally_() argument passing
  - with_() argument passing
  - foreach/filter/gather argument passing
  - return_() and break_() argument passing
  - run() entry point argument passing
  - Pipe operator argument passing
  - Chain reuse argument passing
  - Decorator argument passing
  - Edge cases
"""

import asyncio
from contextlib import contextmanager, asynccontextmanager
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_
from quent import Chain, QuentException, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def identity(v):
  return v

async def aidentity(v):
  return v

def add(a, b):
  return a + b

def add3(a, b, c):
  return a + b + c

def kw_fn(**kwargs):
  return kwargs

def mixed_fn(a, b=10):
  return (a, b)

def no_args_fn():
  return 'no_args_result'

async def async_no_args_fn():
  return 'async_no_args_result'

def capture(*args, **kwargs):
  """Capture all args and kwargs for inspection."""
  return ('args', args, 'kwargs', kwargs)

def const(v):
  """Return a zero-arg function that returns v."""
  def _inner():
    return v
  return _inner

class Tracker:
  """Tracks calls and their arguments."""
  def __init__(self):
    self.calls = []

  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    if args:
      return args[0]
    return None

  def reset(self):
    self.calls = []


class SyncCM:
  """Simple sync context manager."""
  def __init__(self, value='cm_value'):
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
  """Simple async context manager."""
  def __init__(self, value='async_cm_value'):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


class TestExc(Exception):
  pass


class TestExc2(Exception):
  pass


# ===================================================================
# Section 1: _determine_eval_code matrix
# ===================================================================

class TestDetermineEvalCodeMatrix(IsolatedAsyncioTestCase):
  """Test each cell of the eval code determination matrix."""

  # --- callable, no args, no kwargs ---
  async def test_callable_no_args_no_kwargs(self):
    """callable=Yes, args=No, kwargs=No -> EVAL_CALL_WITH_CURRENT_VALUE.
    fn receives current_value as first arg."""
    result = Chain(10).then(identity).run()
    self.assertEqual(result, 10)

  async def test_callable_no_args_no_kwargs_async(self):
    result = Chain(10).then(aidentity).run()
    self.assertEqual(await result, 10)

  # --- callable, args=Yes, kwargs=No ---
  async def test_callable_with_args_no_kwargs(self):
    """callable=Yes, args=Yes, kwargs=No -> EVAL_CALL_WITH_EXPLICIT_ARGS.
    fn receives *args, current_value ignored."""
    result = Chain(999).then(add, 1, 2).run()
    self.assertEqual(result, 3)

  # --- callable, args=No, kwargs=Yes ---
  async def test_callable_no_args_with_kwargs(self):
    """callable=Yes, args=No, kwargs=Yes -> EVAL_CALL_WITH_EXPLICIT_ARGS.
    fn receives **kwargs only."""
    result = Chain(999).then(kw_fn, x=1, y=2).run()
    self.assertEqual(result, {'x': 1, 'y': 2})

  # --- callable, args=Yes, kwargs=Yes ---
  async def test_callable_with_args_and_kwargs(self):
    """callable=Yes, args=Yes, kwargs=Yes -> EVAL_CALL_WITH_EXPLICIT_ARGS."""
    result = Chain(999).then(mixed_fn, 5, b=20).run()
    self.assertEqual(result, (5, 20))

  # --- non-callable, no args, no kwargs, allow_literal=True ---
  async def test_non_callable_no_args_allow_literal_true(self):
    """non-callable, no args, allow_literal=True -> EVAL_RETURN_AS_IS."""
    result = Chain(10).then(42).run()
    self.assertEqual(result, 42)

  async def test_non_callable_string_return_as_is(self):
    result = Chain(10).then('hello').run()
    self.assertEqual(result, 'hello')

  async def test_non_callable_none_return_as_is(self):
    result = Chain(10).then(None).run()
    self.assertIsNone(result)

  async def test_non_callable_list_return_as_is(self):
    result = Chain(10).then([1, 2, 3]).run()
    self.assertEqual(result, [1, 2, 3])

  # --- non-callable, no args, no kwargs, allow_literal=False ---
  async def test_non_callable_no_args_allow_literal_false_raises(self):
    """non-callable, no args, allow_literal=False -> RAISES QuentException.
    do() sets allow_literal=False."""
    with self.assertRaises(QuentException):
      Chain(10).do(42).run()

  async def test_non_callable_string_not_allowed_in_do(self):
    with self.assertRaises(QuentException):
      Chain(10).do('hello').run()

  async def test_non_callable_none_not_allowed_in_do(self):
    with self.assertRaises(QuentException):
      Chain(10).do(None).run()

  # --- non-callable, with args -> TypeError at runtime ---
  async def test_non_callable_with_args_raises_typeerror(self):
    """Non-callable with args: _determine_eval_code assigns EVAL_CALL_WITH_EXPLICIT_ARGS
    (it doesn't check callability when args are present), but runtime TypeError occurs."""
    with self.assertRaises(TypeError):
      Chain(10).do(42, 'arg1').run()

  async def test_non_callable_with_kwargs_raises_typeerror(self):
    """Non-callable with kwargs only: same behavior, TypeError at runtime."""
    with self.assertRaises(TypeError):
      Chain(10).do(42, key='val').run()

  # --- Ellipsis as first arg -> EVAL_CALL_WITHOUT_ARGS ---
  async def test_ellipsis_triggers_call_without_args(self):
    """args[0] is Ellipsis -> EVAL_CALL_WITHOUT_ARGS. fn() called with no args."""
    result = Chain(999).then(no_args_fn, ...).run()
    self.assertEqual(result, 'no_args_result')

  async def test_ellipsis_async(self):
    result = Chain(999).then(async_no_args_fn, ...).run()
    self.assertEqual(await result, 'async_no_args_result')

  # --- Null value as root ---
  async def test_null_root_void_chain(self):
    """Chain() with no root -> void chain, links get Null."""
    result = Chain().then(no_args_fn).run()
    self.assertEqual(result, 'no_args_result')

  async def test_null_current_value_calls_without_args(self):
    """When current_value is Null, callable is called with no args (fast path in evaluate_value)."""
    result = Chain().then(no_args_fn).run()
    self.assertEqual(result, 'no_args_result')


# ===================================================================
# Section 2: then() argument passing variants
# ===================================================================

class TestThenArgumentPassing(IsolatedAsyncioTestCase):
  """Test all then() argument passing combinations."""

  async def test_then_fn_receives_current_value(self):
    """then(fn) -> fn(current_value)."""
    result = Chain(10).then(lambda v: v * 2).run()
    self.assertEqual(result, 20)

  async def test_then_fn_with_one_arg(self):
    """then(fn, arg1) -> fn(arg1). current_value is IGNORED."""
    result = Chain(999).then(identity, 42).run()
    self.assertEqual(result, 42)

  async def test_then_fn_with_two_args(self):
    """then(fn, arg1, arg2) -> fn(arg1, arg2)."""
    result = Chain(999).then(add, 3, 7).run()
    self.assertEqual(result, 10)

  async def test_then_fn_with_three_args(self):
    """then(fn, arg1, arg2, arg3) -> fn(arg1, arg2, arg3)."""
    result = Chain(999).then(add3, 1, 2, 3).run()
    self.assertEqual(result, 6)

  async def test_then_fn_with_kwargs_only(self):
    """then(fn, key=val) -> fn(key=val). current_value ignored."""
    result = Chain(999).then(kw_fn, x=10).run()
    self.assertEqual(result, {'x': 10})

  async def test_then_fn_with_args_and_kwargs(self):
    """then(fn, arg1, key=val) -> fn(arg1, key=val)."""
    result = Chain(999).then(mixed_fn, 5, b=20).run()
    self.assertEqual(result, (5, 20))

  async def test_then_literal_value(self):
    """then(literal_value) -> returns literal_value (EVAL_RETURN_AS_IS)."""
    result = Chain(10).then(42).run()
    self.assertEqual(result, 42)

  async def test_then_literal_none(self):
    """then(None) -> returns None."""
    result = Chain(10).then(None).run()
    self.assertIsNone(result)

  async def test_then_literal_string(self):
    result = Chain(10).then('hello').run()
    self.assertEqual(result, 'hello')

  async def test_then_literal_dict(self):
    result = Chain(10).then({'a': 1}).run()
    self.assertEqual(result, {'a': 1})

  async def test_then_callable_no_params_gets_current_value(self):
    """then(callable_no_args) -> fn(current_value). This will TypeError
    if fn does not accept args, because current_value is always passed."""
    tracker = Tracker()
    Chain(10).then(tracker).run()
    assert tracker.calls == [((10,), {})]

  async def test_then_with_args_current_value_not_passed(self):
    """When args are provided to then(), current_value is NOT passed at all."""
    tracker = Tracker()
    Chain(10).then(tracker, 'a', 'b').run()
    assert tracker.calls == [(('a', 'b'), {})]

  async def test_then_chained_values(self):
    """Multiple then()s each get the previous result."""
    result = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).run()
    self.assertEqual(result, 6)

  async def test_then_with_explicit_args_chain(self):
    """Chain of then() with explicit args."""
    result = (
      Chain(999)
      .then(add, 1, 2)
      .then(lambda v: v + 10)
      .run()
    )
    self.assertEqual(result, 13)

  async def test_then_async_fn_receives_current_value(self):
    result = Chain(10).then(aidentity).run()
    self.assertEqual(await result, 10)

  async def test_then_async_fn_with_args(self):
    async def async_add(a, b):
      return a + b
    result = Chain(999).then(async_add, 3, 4).run()
    self.assertEqual(await result, 7)


# ===================================================================
# Section 3: do() argument passing variants
# ===================================================================

class TestDoArgumentPassing(IsolatedAsyncioTestCase):
  """Test do() argument passing: side-effects whose results are discarded."""

  async def test_do_fn_receives_current_value(self):
    """do(fn) -> fn(current_value), result discarded."""
    tracker = Tracker()
    result = Chain(10).do(tracker).run()
    self.assertEqual(result, 10)
    assert tracker.calls == [((10,), {})]

  async def test_do_fn_with_args(self):
    """do(fn, arg1) -> fn(arg1), result discarded."""
    tracker = Tracker()
    result = Chain(10).do(tracker, 'side').run()
    self.assertEqual(result, 10)
    assert tracker.calls == [(('side',), {})]

  async def test_do_fn_with_kwargs(self):
    """do(fn, key=val) -> fn(key=val), result discarded."""
    tracker = Tracker()
    result = Chain(10).do(tracker, x=42).run()
    self.assertEqual(result, 10)
    assert tracker.calls == [((), {'x': 42})]

  async def test_do_fn_with_args_and_kwargs(self):
    """do(fn, arg1, key=val) -> fn(arg1, key=val), result discarded."""
    tracker = Tracker()
    result = Chain(10).do(tracker, 'a', x=1).run()
    self.assertEqual(result, 10)
    assert tracker.calls == [(('a',), {'x': 1})]

  async def test_do_result_discarded(self):
    """Regardless of what do() returns, the chain continues with the previous value."""
    result = Chain(10).do(lambda v: 9999).then(identity).run()
    self.assertEqual(result, 10)

  async def test_do_non_callable_raises(self):
    """do(literal) raises QuentException because allow_literal=False."""
    with self.assertRaises(QuentException):
      Chain(10).do(42).run()

  async def test_do_none_raises(self):
    with self.assertRaises(QuentException):
      Chain(10).do(None).run()

  async def test_do_with_ellipsis(self):
    """do(fn, ...) -> fn() called with no args, result discarded."""
    tracker = Tracker()
    result = Chain(10).do(tracker, ...).run()
    self.assertEqual(result, 10)
    assert tracker.calls == [((), {})]

  async def test_do_async_fn(self):
    tracker = Tracker()
    result = Chain(10).do(aidentity).run()
    self.assertEqual(await result, 10)


# ===================================================================
# Section 4: except_() argument passing
# ===================================================================

class TestExceptArgumentPassing(IsolatedAsyncioTestCase):
  """Test except_() handler argument passing.

  Key insight from source: except_ handler is called with:
    evaluate_value(exc_link, root_value)
  So the handler receives ROOT_VALUE, not the exception.
  """

  async def test_except_handler_receives_root_value(self):
    """except_(fn) -> fn(root_value). The handler receives the chain's root value.
    When reraise=False, handler's return value becomes chain result."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(tracker, reraise=False).run()
    # Tracker returns args[0] which is root_value=42
    self.assertEqual(result, 42)
    assert tracker.calls == [((42,), {})]

  async def test_except_handler_with_explicit_args(self):
    """except_(fn, arg1) -> fn(arg1). Root value is ignored.
    Handler's return value becomes chain result."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(tracker, 'caught', reraise=False).run()
    # Tracker returns args[0] which is 'caught'
    self.assertEqual(result, 'caught')
    assert tracker.calls == [(('caught',), {})]

  async def test_except_handler_with_kwargs(self):
    """except_(fn, key=val) -> fn(key=val)."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(tracker, x='err', reraise=False).run()
    self.assertIsNone(result)
    assert tracker.calls == [((), {'x': 'err'})]

  async def test_except_catches_specific_exception(self):
    """except_(fn, exceptions=ExcType) only catches that type."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(tracker, exceptions=TestExc, reraise=False).run()
    # Tracker receives root_value=42 and returns 42
    self.assertEqual(result, 42)
    assert len(tracker.calls) == 1

  async def test_except_does_not_catch_wrong_type(self):
    """except_(fn, exceptions=WrongType) does not catch other exceptions."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    with self.assertRaises(TestExc):
      Chain(42).then(raise_fn).except_(tracker, exceptions=TestExc2, reraise=False).run()
    assert len(tracker.calls) == 0

  async def test_except_reraise_true(self):
    """except_(fn, reraise=True) calls handler and re-raises."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    with self.assertRaises(TestExc):
      Chain(42).then(raise_fn).except_(tracker, reraise=True).run()
    assert len(tracker.calls) == 1

  async def test_except_reraise_false_returns_none(self):
    """except_(fn, reraise=False) swallows exception, returns None if handler returns Null."""
    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(lambda v: None, reraise=False).run()
    self.assertIsNone(result)

  async def test_except_reraise_false_handler_return_value(self):
    """When reraise=False, the handler's return value becomes the chain result."""
    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(lambda v: 'recovered', reraise=False).run()
    self.assertEqual(result, 'recovered')

  async def test_except_with_ellipsis(self):
    """except_(fn, ...) -> fn() called with no args."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(tracker, ..., reraise=False).run()
    self.assertIsNone(result)
    assert tracker.calls == [((), {})]

  async def test_except_non_callable_raises(self):
    """except_ with non-callable raises QuentException (allow_literal=False)."""
    with self.assertRaises(QuentException):
      Chain(10).except_(42, reraise=False)

  async def test_except_void_chain_root_is_override(self):
    """In a void chain with run(override), root_value is the override.
    Handler receives root_value=10 and Tracker returns 10."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain().then(raise_fn).except_(tracker, reraise=False).run(10)
    # Tracker returns args[0] which is root_value=10
    self.assertEqual(result, 10)
    assert tracker.calls == [((10,), {})]

  async def test_except_with_multiple_exception_types(self):
    """except_(fn, exceptions=[Type1, Type2]) catches either."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc2('boom')

    result = (
      Chain(42)
      .then(raise_fn)
      .except_(tracker, exceptions=[TestExc, TestExc2], reraise=False)
      .run()
    )
    # Tracker receives root_value=42 and returns 42
    self.assertEqual(result, 42)
    assert len(tracker.calls) == 1

  async def test_except_async_handler(self):
    """Async exception handler."""
    async def handler(v):
      return 'async_recovered'

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(handler, reraise=False).run()
    self.assertEqual(await result, 'async_recovered')


# ===================================================================
# Section 5: finally_() argument passing
# ===================================================================

class TestFinallyArgumentPassing(IsolatedAsyncioTestCase):
  """Test finally_() handler argument passing.

  Key insight from source: finally_ handler is called with:
    evaluate_value(self.on_finally_link, root_value)
  So the handler receives ROOT_VALUE.
  """

  async def test_finally_handler_receives_root_value(self):
    """finally_(fn) -> fn(root_value)."""
    tracker = Tracker()
    result = Chain(42).then(identity).finally_(tracker).run()
    self.assertEqual(result, 42)
    assert tracker.calls == [((42,), {})]

  async def test_finally_handler_with_args(self):
    """finally_(fn, arg1) -> fn(arg1)."""
    tracker = Tracker()
    result = Chain(42).then(identity).finally_(tracker, 'cleanup').run()
    self.assertEqual(result, 42)
    assert tracker.calls == [(('cleanup',), {})]

  async def test_finally_handler_with_kwargs(self):
    """finally_(fn, key=val) -> fn(key=val)."""
    tracker = Tracker()
    result = Chain(42).then(identity).finally_(tracker, x='final').run()
    self.assertEqual(result, 42)
    assert tracker.calls == [((), {'x': 'final'})]

  async def test_finally_handler_with_args_and_kwargs(self):
    tracker = Tracker()
    result = Chain(42).then(identity).finally_(tracker, 'a', x=1).run()
    self.assertEqual(result, 42)
    assert tracker.calls == [(('a',), {'x': 1})]

  async def test_finally_runs_on_exception(self):
    """finally_ runs even when an exception occurs."""
    tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    with self.assertRaises(TestExc):
      Chain(42).then(raise_fn).finally_(tracker).run()
    assert len(tracker.calls) == 1

  async def test_finally_result_discarded(self):
    """finally_ handler's return value is always discarded."""
    result = Chain(42).then(identity).finally_(lambda v: 9999).run()
    self.assertEqual(result, 42)

  async def test_finally_with_ellipsis(self):
    """finally_(fn, ...) -> fn() called with no args."""
    tracker = Tracker()
    result = Chain(42).then(identity).finally_(tracker, ...).run()
    self.assertEqual(result, 42)
    assert tracker.calls == [((), {})]

  async def test_finally_non_callable_raises(self):
    """finally_ with non-callable raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(10).finally_(42)

  async def test_finally_void_chain_with_override(self):
    """Void chain with run(override) -> root is override."""
    tracker = Tracker()
    result = Chain().then(identity).finally_(tracker).run(99)
    self.assertEqual(result, 99)
    assert tracker.calls == [((99,), {})]

  async def test_only_one_finally_allowed(self):
    """Cannot register more than one finally_ callback."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(identity).finally_(identity)


# ===================================================================
# Section 6: with_() argument passing
# ===================================================================

class TestWithArgumentPassing(IsolatedAsyncioTestCase):
  """Test with_() (context manager) argument passing.

  Key insight from source:
    if not self.args and not self.kwargs:
      self.link.temp_args = (ctx,)
    result = evaluate_value(self.link, ctx)

  So:
  - with_(fn): fn receives ctx as current_value (EVAL_CALL_WITH_CURRENT_VALUE -> fn(ctx))
  - with_(fn, arg1): fn called with explicit args, ctx ignored (EVAL_CALL_WITH_EXPLICIT_ARGS -> fn(arg1))
  """

  async def test_with_fn_receives_ctx(self):
    """with_(fn) -> fn(ctx) where ctx is __enter__ result."""
    cm = SyncCM('the_ctx')
    result = Chain(cm).with_(identity).run()
    self.assertEqual(result, 'the_ctx')
    assert cm.entered
    assert cm.exited

  async def test_with_fn_with_args_ignores_ctx(self):
    """with_(fn, arg1) -> fn(arg1). ctx is NOT passed.
    The body's return value becomes the chain's current value."""
    cm = SyncCM('the_ctx')
    tracker = Tracker()
    result = Chain(cm).with_(tracker, 'explicit_arg').run()
    # Tracker returns args[0] = 'explicit_arg'
    self.assertEqual(result, 'explicit_arg')
    assert tracker.calls == [(('explicit_arg',), {})]
    assert cm.entered
    assert cm.exited

  async def test_with_fn_with_kwargs_ignores_ctx(self):
    """with_(fn, key=val) -> fn(key=val). ctx is NOT passed."""
    cm = SyncCM('the_ctx')
    result = Chain(cm).with_(kw_fn, x=42).run()
    self.assertEqual(result, {'x': 42})

  async def test_with_fn_with_args_and_kwargs(self):
    """with_(fn, arg1, key=val) -> fn(arg1, key=val)."""
    cm = SyncCM('the_ctx')
    result = Chain(cm).with_(mixed_fn, 5, b=20).run()
    self.assertEqual(result, (5, 20))

  async def test_with_async_cm(self):
    """with_() works with async context managers."""
    cm = AsyncCM('async_ctx')
    result = Chain(cm).with_(identity).run()
    self.assertEqual(await result, 'async_ctx')
    assert cm.entered
    assert cm.exited

  async def test_with_async_cm_with_args(self):
    """with_(fn, arg1) on async cm. Body return value propagates."""
    cm = AsyncCM('async_ctx')
    tracker = Tracker()
    result = Chain(cm).with_(tracker, 'explicit').run()
    # Tracker returns args[0] = 'explicit'
    self.assertEqual(await result, 'explicit')
    assert tracker.calls == [(('explicit',), {})]

  async def test_with_ellipsis(self):
    """with_(fn, ...) -> fn() called with no args."""
    cm = SyncCM('the_ctx')
    result = Chain(cm).with_(no_args_fn, ...).run()
    self.assertEqual(result, 'no_args_result')

  async def test_with_result_propagates(self):
    """with_() result becomes the new current value."""
    cm = SyncCM('the_ctx')
    result = Chain(cm).with_(lambda ctx: ctx.upper()).run()
    self.assertEqual(result, 'THE_CTX')


# ===================================================================
# Section 7: foreach/filter argument passing
# ===================================================================

class TestForeachFilterArgumentPassing(IsolatedAsyncioTestCase):
  """Test iteration operations argument passing."""

  async def test_foreach_fn_receives_element(self):
    """foreach(fn) -> fn(element) for each element."""
    result = Chain([1, 2, 3]).foreach(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  async def test_filter_fn_receives_element(self):
    """filter(fn) -> keeps elements where fn(element) is truthy."""
    result = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).run()
    self.assertEqual(result, [4, 5])

  async def test_foreach_with_identity(self):
    """foreach(identity) -> returns elements unchanged."""
    result = Chain([1, 2, 3]).foreach(identity).run()
    self.assertEqual(result, [1, 2, 3])

  async def test_foreach_async_fn(self):
    """foreach with async function."""
    async def double(x):
      return x * 2
    result = Chain([1, 2, 3]).foreach(double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_filter_async_fn(self):
    """filter with async function."""
    async def is_even(x):
      return x % 2 == 0
    result = Chain([1, 2, 3, 4]).filter(is_even).run()
    self.assertEqual(await result, [2, 4])

  async def test_foreach_empty_list(self):
    result = Chain([]).foreach(identity).run()
    self.assertEqual(result, [])

  async def test_filter_none_pass(self):
    """Filter where no elements pass."""
    result = Chain([1, 2, 3]).filter(lambda x: x > 10).run()
    self.assertEqual(result, [])

  async def test_filter_all_pass(self):
    result = Chain([1, 2, 3]).filter(lambda x: True).run()
    self.assertEqual(result, [1, 2, 3])


# ===================================================================
# Section 8: gather() argument passing
# ===================================================================

class TestGatherArgumentPassing(IsolatedAsyncioTestCase):
  """Test gather() argument passing."""

  async def test_gather_each_fn_receives_current_value(self):
    """gather(fn1, fn2) -> [fn1(current_value), fn2(current_value)]."""
    result = Chain(10).gather(
      lambda v: v + 1,
      lambda v: v * 2,
      lambda v: v - 3
    ).run()
    self.assertEqual(result, [11, 20, 7])

  async def test_gather_single_fn(self):
    result = Chain(5).gather(lambda v: v * 3).run()
    self.assertEqual(result, [15])

  async def test_gather_async_fns(self):
    async def add1(v):
      return v + 1
    async def mul2(v):
      return v * 2
    result = Chain(10).gather(add1, mul2).run()
    self.assertEqual(await result, [11, 20])

  async def test_gather_mixed_sync_async(self):
    async def async_add1(v):
      return v + 1
    result = Chain(10).gather(
      lambda v: v * 2,
      async_add1
    ).run()
    self.assertEqual(await result, [20, 11])


# ===================================================================
# Section 9: return_() and break_() argument passing
# ===================================================================

class TestReturnBreakArgumentPassing(IsolatedAsyncioTestCase):
  """Test return_() and break_() argument passing.

  From _eval_signal_value:
  - If callable and no args: v() is called
  - If callable with args: v(*args, **kwargs)
  - If not callable: returns v as-is (literal)
  """

  async def test_return_current_value(self):
    """return_() called via then(Chain.return_) receives current_value.
    Chain.return_(v) raises _Return(v, (), {}). _eval_signal_value(v, (), {})
    since v=10 is non-callable -> returns 10 as-is."""
    inner = Chain().then(Chain.return_)
    result = Chain(10).then(inner).run()
    self.assertEqual(result, 10)

  async def test_return_literal_value(self):
    """return_(literal) -> returns the literal."""
    inner = Chain().then(lambda v: Chain.return_(42))
    result = Chain(10).then(inner).run()
    self.assertEqual(result, 42)

  async def test_return_callable_value(self):
    """return_(fn) -> calls fn() (no args since it's a signal)."""
    inner = Chain().then(lambda v: Chain.return_(no_args_fn))
    result = Chain(10).then(inner).run()
    self.assertEqual(result, 'no_args_result')

  async def test_return_callable_with_args(self):
    """return_(fn, arg1) -> fn(arg1)."""
    inner = Chain().then(lambda v: Chain.return_(add, 3, 7))
    result = Chain(10).then(inner).run()
    self.assertEqual(result, 10)

  async def test_return_callable_with_kwargs(self):
    """return_(fn, key=val) -> fn(key=val)."""
    inner = Chain().then(lambda v: Chain.return_(kw_fn, x=42))
    result = Chain(10).then(inner).run()
    self.assertEqual(result, {'x': 42})

  async def test_break_no_value_in_foreach(self):
    """break_() in foreach -> returns accumulated results."""
    def maybe_break(x):
      if x == 3:
        Chain.break_()
      return x * 10

    result = Chain([1, 2, 3, 4, 5]).foreach(maybe_break).run()
    self.assertEqual(result, [10, 20])

  async def test_break_with_value_in_foreach(self):
    """break_(value) in foreach -> returns the value."""
    def maybe_break(x):
      if x == 3:
        Chain.break_('stopped')
      return x * 10

    result = Chain([1, 2, 3, 4, 5]).foreach(maybe_break).run()
    self.assertEqual(result, 'stopped')

  async def test_break_with_callable_in_foreach(self):
    """break_(fn) in foreach -> calls fn()."""
    def maybe_break(x):
      if x == 3:
        Chain.break_(no_args_fn)
      return x * 10

    result = Chain([1, 2, 3, 4, 5]).foreach(maybe_break).run()
    self.assertEqual(result, 'no_args_result')

  async def test_break_with_callable_and_args_in_foreach(self):
    """break_(fn, arg1, arg2) in foreach -> fn(arg1, arg2)."""
    def maybe_break(x):
      if x == 3:
        Chain.break_(add, 100, 200)
      return x * 10

    result = Chain([1, 2, 3, 4, 5]).foreach(maybe_break).run()
    self.assertEqual(result, 300)

  async def test_break_outside_loop_raises(self):
    """break_() outside of a loop raises QuentException."""
    with self.assertRaises(QuentException):
      Chain().then(Chain.break_).run()

  async def test_return_with_ellipsis(self):
    """return_(fn, ...) -> fn() called with no args."""
    inner = Chain().then(lambda v: Chain.return_(no_args_fn, ...))
    result = Chain(10).then(inner).run()
    self.assertEqual(result, 'no_args_result')


# ===================================================================
# Section 10: run() argument passing (entry point)
# ===================================================================

class TestRunArgumentPassing(IsolatedAsyncioTestCase):
  """Test run() entry point argument passing."""

  async def test_run_no_args_uses_root(self):
    """chain.run() with no args uses the root value."""
    result = Chain(42).run()
    self.assertEqual(result, 42)

  async def test_run_override_on_void_chain(self):
    """Chain().run(override) -> override becomes root."""
    result = Chain().then(identity).run(42)
    self.assertEqual(result, 42)

  async def test_run_override_on_rooted_chain_raises(self):
    """Chain(root).run(override) raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(10).run(20)

  async def test_run_callable_root_gets_called(self):
    """Chain(fn).run() -> fn() called with no args (Null current_value)."""
    result = Chain(no_args_fn).run()
    self.assertEqual(result, 'no_args_result')

  async def test_run_callable_root_with_args(self):
    """Chain(fn, arg1).run() -> fn(arg1)."""
    result = Chain(add, 3, 7).run()
    self.assertEqual(result, 10)

  async def test_run_callable_root_with_kwargs(self):
    """Chain(fn, key=val).run() -> fn(key=val)."""
    result = Chain(kw_fn, x=42).run()
    self.assertEqual(result, {'x': 42})

  async def test_void_chain_run_passes_override_as_current_value(self):
    """Chain().then(fn).run(10) -> fn(10)."""
    result = Chain().then(lambda v: v * 3).run(10)
    self.assertEqual(result, 30)

  async def test_void_chain_run_callable_override(self):
    """Chain().run(fn) -> fn() is called (make_temp_link, allow_literal=True)."""
    result = Chain().then(identity).run(no_args_fn)
    self.assertEqual(result, 'no_args_result')

  async def test_void_chain_run_callable_override_with_args(self):
    """Chain().run(fn, arg1, arg2) -> fn(arg1, arg2) via _make_temp_link."""
    result = Chain().then(identity).run(add, 3, 7)
    self.assertEqual(result, 10)

  async def test_void_chain_run_literal_override(self):
    """Chain().run(literal) -> literal returned as-is (_make_temp_link, allow_literal=True)."""
    result = Chain().then(identity).run(42)
    self.assertEqual(result, 42)

  async def test_run_no_root_no_links_returns_none(self):
    """Chain().run() -> None (no root, no links)."""
    result = Chain().run()
    self.assertIsNone(result)

  async def test_run_root_only_no_links(self):
    """Chain(42).run() -> 42."""
    result = Chain(42).run()
    self.assertEqual(result, 42)

  async def test_run_with_override_and_links(self):
    """Chain().then(fn1).then(fn2).run(5) -> pipeline from 5."""
    result = (
      Chain()
      .then(lambda v: v + 1)
      .then(lambda v: v * 2)
      .run(5)
    )
    self.assertEqual(result, 12)


# ===================================================================
# Section 12: Chain reuse argument passing
# ===================================================================

class TestChainReuseArgumentPassing(IsolatedAsyncioTestCase):
  """Test chain reuse argument passing."""

  async def test_chain_uses_original_root(self):
    """chain.run() uses the original root."""
    c = Chain(42).then(identity)
    result = c.run()
    self.assertEqual(result, 42)

  async def test_chain_override_on_void_chain(self):
    """chain.run(override) on void chain."""
    c = Chain().then(lambda v: v * 2)
    result = c.run(10)
    self.assertEqual(result, 20)

  async def test_chain_override_with_args(self):
    """chain.run(fn, arg1) passes args to _make_temp_link."""
    c = Chain().then(identity)
    result = c.run(add, 3, 7)
    self.assertEqual(result, 10)

  async def test_chain_call_equivalent_to_run(self):
    """chain(value) -> chain.run(value)."""
    c = Chain().then(identity)
    result = c(42)
    self.assertEqual(result, 42)

  async def test_chain_reusable(self):
    """Chain can be run multiple times."""
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c.run(5), 10)
    self.assertEqual(c.run(10), 20)
    self.assertEqual(c.run(15), 30)


# ===================================================================
# Section 13: Decorator argument passing
# ===================================================================

class TestDecoratorArgumentPassing(IsolatedAsyncioTestCase):
  """Test decorator() argument passing."""

  async def test_decorator_basic(self):
    """decorator()(wrapped_fn)(args) -> chain runs with wrapped_fn as root."""
    dec = Chain().then(lambda fn_result: fn_result + ' decorated').decorator()

    @dec
    def greet(name):
      return f'hello {name}'

    result = greet('world')
    self.assertEqual(result, 'hello world decorated')

  async def test_decorator_args_forwarded_to_wrapped_fn(self):
    """Arguments passed to the decorated function are forwarded to the wrapped fn."""
    dec = Chain().then(lambda v: v * 2).decorator()

    @dec
    def compute(a, b):
      return a + b

    result = compute(3, 4)
    self.assertEqual(result, 14)

  async def test_decorator_kwargs_forwarded(self):
    """Kwargs passed to decorated function are forwarded."""
    dec = Chain().then(identity).decorator()

    @dec
    def fn(x, y=10):
      return x + y

    result = fn(5, y=20)
    self.assertEqual(result, 25)

  async def test_decorator_with_chain_operations(self):
    """Decorator with multiple chain operations."""
    dec = (
      Chain()
      .then(lambda v: v + 10)
      .then(lambda v: v * 2)
      .decorator()
    )

    @dec
    def start(x):
      return x

    result = start(5)
    self.assertEqual(result, 30)


# ===================================================================
# Section 14: Edge cases
# ===================================================================

class TestEdgeCases(IsolatedAsyncioTestCase):
  """Test edge cases in argument passing."""

  async def test_then_fn_many_positional_args(self):
    """then(fn, *many_args)."""
    def sum_all(*args):
      return sum(args)
    result = Chain(999).then(sum_all, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10).run()
    self.assertEqual(result, 55)

  async def test_then_fn_many_kwargs(self):
    """then(fn, **many_kwargs)."""
    result = Chain(999).then(kw_fn, a=1, b=2, c=3, d=4, e=5).run()
    self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})

  async def test_then_fn_takes_no_args_gets_current_value_typeerror(self):
    """then(fn) where fn takes no args -> TypeError when current_value passed."""
    with self.assertRaises(TypeError):
      Chain(10).then(no_args_fn).run()

  async def test_none_as_explicit_arg(self):
    """then(fn, None) -> fn(None), not fn(current_value)."""
    tracker = Tracker()
    Chain(999).then(tracker, None).run()
    assert tracker.calls == [((None,), {})]

  async def test_false_as_explicit_arg(self):
    """then(fn, False) -> fn(False)."""
    tracker = Tracker()
    Chain(999).then(tracker, False).run()
    assert tracker.calls == [((False,), {})]

  async def test_zero_as_explicit_arg(self):
    """then(fn, 0) -> fn(0)."""
    tracker = Tracker()
    Chain(999).then(tracker, 0).run()
    assert tracker.calls == [((0,), {})]

  async def test_empty_string_as_explicit_arg(self):
    """then(fn, '') -> fn('')."""
    tracker = Tracker()
    Chain(999).then(tracker, '').run()
    assert tracker.calls == [(('',), {})]

  async def test_empty_tuple_as_explicit_arg(self):
    """then(fn, ()) -> fn(()). Note: () is a single tuple arg."""
    tracker = Tracker()
    Chain(999).then(tracker, ()).run()
    assert tracker.calls == [(((),), {})]

  async def test_null_as_root_value(self):
    """Chain(Null) -> same as Chain() since Null is the sentinel."""
    result = Chain(Null).then(no_args_fn).run()
    self.assertEqual(result, 'no_args_result')

  async def test_chain_callable_root_evaluated(self):
    """Chain(callable) -> callable is called at run time."""
    result = Chain(lambda: 42).run()
    self.assertEqual(result, 42)

  async def test_chain_callable_root_with_args(self):
    """Chain(fn, arg1, arg2) -> fn(arg1, arg2)."""
    result = Chain(add, 3, 7).run()
    self.assertEqual(result, 10)

  async def test_chain_non_callable_root_returned_as_is(self):
    """Chain(literal) -> literal is returned as-is."""
    result = Chain(42).run()
    self.assertEqual(result, 42)

  async def test_chain_none_root(self):
    """Chain(None) -> None is returned."""
    result = Chain(None).run()
    self.assertIsNone(result)

  async def test_chain_of_chains(self):
    """then(Chain) -> nested chain is executed."""
    inner = Chain().then(lambda v: v + 100)
    result = Chain(10).then(inner).run()
    self.assertEqual(result, 110)

  async def test_chain_of_chains_with_args(self):
    """then(Chain, arg1) -> nested chain run with arg1 as root."""
    inner = Chain().then(lambda v: v * 2)
    result = Chain(999).then(inner, 5).run()
    self.assertEqual(result, 10)

  async def test_eval_call_with_current_value_null_calls_no_args(self):
    """EVAL_CALL_WITH_CURRENT_VALUE with Null current_value -> fn()."""
    # This is the fast path in evaluate_value
    result = Chain().then(no_args_fn).run()
    self.assertEqual(result, 'no_args_result')

  async def test_multiple_do_then_mix(self):
    """Mix of do() and then() preserves correct values."""
    side_effects = []
    result = (
      Chain(1)
      .then(lambda v: v + 1)     # current = 2
      .do(lambda v: side_effects.append(v))  # side effect with 2
      .then(lambda v: v * 3)     # current = 6
      .do(lambda v: side_effects.append(v))  # side effect with 6
      .then(lambda v: v + 4)     # current = 10
      .run()
    )
    self.assertEqual(result, 10)
    assert side_effects == [2, 6]

  async def test_do_then_chain_current_value_preserved(self):
    """After do(), the current value is preserved for the next then()."""
    result = Chain(5).do(lambda v: v * 100).then(lambda v: v + 1).run()
    self.assertEqual(result, 6)

  async def test_except_string_exception_type_raises_type_error(self):
    """except_(fn, exceptions='string') raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(1).except_(identity, exceptions='ValueError')

  async def test_run_kwargs_override(self):
    """Chain().run(fn, key=val)."""
    result = Chain().then(identity).run(kw_fn, x=42)
    self.assertEqual(result, {'x': 42})

  async def test_chain_override_kwargs(self):
    """chain.run(fn, key=val)."""
    c = Chain().then(identity)
    result = c.run(kw_fn, x=42)
    self.assertEqual(result, {'x': 42})


# ===================================================================
# Section 16: Complex multi-step argument flow
# ===================================================================

class TestComplexArgumentFlow(IsolatedAsyncioTestCase):
  """Test complex argument flows through chains."""

  async def test_explicit_args_override_in_pipeline(self):
    """Verify explicit args completely replace current value in pipeline."""
    pipeline = (
      Chain(1)                         # root = 1
      .then(lambda v: v + 9)          # 10
      .then(add, 20, 30)              # add(20, 30) = 50, ignores 10
      .then(lambda v: v + 5)          # 55
      .run()
    )
    self.assertEqual(pipeline, 55)

  async def test_do_does_not_affect_pipeline(self):
    """do() operations never affect the pipeline value."""
    accumulator = []
    result = (
      Chain(10)
      .do(lambda v: accumulator.append(v))   # append 10
      .then(lambda v: v + 5)                  # 15
      .do(lambda v: accumulator.append(v))   # append 15
      .then(lambda v: v * 2)                  # 30
      .run()
    )
    self.assertEqual(result, 30)
    assert accumulator == [10, 15]

  async def test_except_does_not_affect_normal_flow(self):
    """except_ handlers are skipped during normal (non-exception) flow."""
    tracker = Tracker()
    result = (
      Chain(10)
      .then(lambda v: v + 5)
      .except_(tracker, reraise=False)
      .then(lambda v: v * 2)
      .run()
    )
    self.assertEqual(result, 30)
    assert len(tracker.calls) == 0

  async def test_full_pipeline_with_all_operations(self):
    """Full pipeline combining then, do, except, finally."""
    side_effects = []
    cleanup = []
    result = (
      Chain(5)
      .then(lambda v: v + 5)                  # 10
      .do(lambda v: side_effects.append(v))   # side: 10
      .then(lambda v: v * 2)                  # 20
      .except_(lambda v: None, reraise=False)
      .finally_(lambda v: cleanup.append(v))
      .run()
    )
    self.assertEqual(result, 20)
    assert side_effects == [10]
    assert cleanup == [5]  # finally receives root_value

  async def test_void_chain_override_flows_through(self):
    """Void chain with run(value) -> value flows as both root and current."""
    tracker_do = Tracker()
    tracker_finally = Tracker()
    result = (
      Chain()
      .then(lambda v: v + 1)       # 11
      .do(tracker_do)               # do receives 11
      .then(lambda v: v * 2)       # 22
      .finally_(tracker_finally)    # finally receives root = 10
      .run(10)
    )
    self.assertEqual(result, 22)
    assert tracker_do.calls == [((11,), {})]
    assert tracker_finally.calls == [((10,), {})]

  async def test_nested_chain_arg_isolation(self):
    """Nested chains have their own argument scope."""
    inner = Chain().then(lambda v: v + 100)
    result = (
      Chain(5)
      .then(lambda v: v * 2)     # 10
      .then(inner)                 # inner receives 10, returns 110
      .then(lambda v: v + 1)     # 111
      .run()
    )
    self.assertEqual(result, 111)

  async def test_nested_chain_with_explicit_override(self):
    """Nested chain with explicit arg override."""
    inner = Chain().then(lambda v: v + 100)
    result = (
      Chain(5)
      .then(lambda v: v * 2)     # 10
      .then(inner, 50)            # inner receives 50, returns 150
      .then(lambda v: v + 1)     # 151
      .run()
    )
    self.assertEqual(result, 151)


# ===================================================================
# Section 17: Async variants of key tests
# ===================================================================

class TestAsyncArgumentPassing(IsolatedAsyncioTestCase):
  """Test argument passing with async functions to cover async code paths."""

  async def test_async_then_receives_current_value(self):
    result = Chain(10).then(aidentity).run()
    self.assertEqual(await result, 10)

  async def test_async_then_with_explicit_args(self):
    async def async_add(a, b):
      return a + b
    result = Chain(999).then(async_add, 3, 7).run()
    self.assertEqual(await result, 10)

  async def test_async_do_receives_current_value(self):
    tracker = Tracker()
    result = Chain(10).then(aidentity).do(tracker).run()
    self.assertEqual(await result, 10)
    assert tracker.calls == [((10,), {})]

  async def test_async_root_evaluation(self):
    async def async_root():
      return 42
    result = Chain(async_root).run()
    self.assertEqual(await result, 42)

  async def test_async_root_with_args(self):
    async def async_add(a, b):
      return a + b
    result = Chain(async_add, 3, 7).run()
    self.assertEqual(await result, 10)

  async def test_async_chain_pipeline(self):
    result = (
      Chain(10)
      .then(aidentity)
      .then(lambda v: v + 5)
      .then(aidentity)
      .run()
    )
    self.assertEqual(await result, 15)

  async def test_async_void_chain_override(self):
    result = Chain().then(aidentity).run(42)
    self.assertEqual(await result, 42)

  async def test_async_except_handler(self):
    async def handler(v):
      return 'recovered'

    def raise_fn(v):
      raise TestExc('boom')

    result = Chain(42).then(raise_fn).except_(handler, reraise=False).run()
    self.assertEqual(await result, 'recovered')

  async def test_async_finally_handler(self):
    tracker = Tracker()

    async def async_cleanup(v):
      tracker((v,), {})

    result = Chain(42).then(aidentity).finally_(async_cleanup).run()
    self.assertEqual(await result, 42)

  async def test_async_foreach(self):
    async def double(x):
      return x * 2
    result = Chain([1, 2, 3]).foreach(double).run()
    self.assertEqual(await result, [2, 4, 6])

  async def test_async_filter(self):
    async def is_even(x):
      return x % 2 == 0
    result = Chain([1, 2, 3, 4]).filter(is_even).run()
    self.assertEqual(await result, [2, 4])

  async def test_async_gather(self):
    async def add1(v):
      return v + 1
    async def mul2(v):
      return v * 2
    result = Chain(10).gather(add1, mul2).run()
    self.assertEqual(await result, [11, 20])

  async def test_async_with(self):
    cm = AsyncCM('async_ctx')
    result = Chain(cm).with_(aidentity).run()
    self.assertEqual(await result, 'async_ctx')

  async def test_async_decorator(self):
    dec = Chain().then(aidentity).decorator()

    @dec
    def fn(x):
      return x * 2

    result = fn(5)
    self.assertEqual(await result, 10)

  async def test_async_break_in_foreach(self):
    async def maybe_break(x):
      if x == 3:
        Chain.break_('stopped_async')
      return x

    result = Chain([1, 2, 3, 4]).foreach(maybe_break).run()
    self.assertEqual(await result, 'stopped_async')


# ===================================================================
# Section 18: kwargs-only chain link (no args, yes kwargs)
# ===================================================================

class TestKwargsOnlyEdgeCases(IsolatedAsyncioTestCase):
  """Test the specific edge case where kwargs are provided but args are not.

  In _determine_eval_code:
    - if args is falsy and kwargs is truthy:
      - if is_chain: EVAL_CALL_WITHOUT_ARGS
      - else: EVAL_CALL_WITH_EXPLICIT_ARGS (with args normalized to EMPTY_TUPLE)
  """

  async def test_kwargs_only_non_chain(self):
    """Non-chain with kwargs only -> EVAL_CALL_WITH_EXPLICIT_ARGS."""
    result = Chain(999).then(kw_fn, x=1, y=2).run()
    self.assertEqual(result, {'x': 1, 'y': 2})

  async def test_kwargs_only_with_chain(self):
    """Chain with kwargs only -> EVAL_CALL_WITHOUT_ARGS.
    The chain is called without any args."""
    inner = Chain(42)
    # kwargs only for a chain link triggers EVAL_CALL_WITHOUT_ARGS
    result = Chain(999).then(inner, x=1).run()
    # inner chain runs with Null (no override) and returns its own root
    self.assertEqual(result, 42)

  async def test_kwargs_only_in_do(self):
    """do(fn, key=val) -> fn(key=val)."""
    tracker = Tracker()
    Chain(10).do(tracker, x=42).run()
    assert tracker.calls == [((), {'x': 42})]

  async def test_empty_kwargs_dict_treated_as_no_kwargs(self):
    """When kwargs dict is empty (from **{}), it's treated as no kwargs."""
    # Empty kwargs dict from Python -> empty dict, which is falsy
    tracker = Tracker()
    Chain(10).then(tracker).run()
    assert tracker.calls == [((10,), {})]


# ===================================================================
# Section 19: _make_temp_link via run() override
# ===================================================================

class TestMakeTempLink(IsolatedAsyncioTestCase):
  """Test _make_temp_link behavior (triggered by run() override)."""

  async def test_temp_link_callable_no_args(self):
    """run(callable) -> callable() called, EVAL_CALL_WITH_CURRENT_VALUE with Null."""
    result = Chain().then(identity).run(no_args_fn)
    self.assertEqual(result, 'no_args_result')

  async def test_temp_link_callable_with_args(self):
    """run(fn, arg1) -> fn(arg1), EVAL_CALL_WITH_EXPLICIT_ARGS."""
    result = Chain().then(identity).run(add, 3, 7)
    self.assertEqual(result, 10)

  async def test_temp_link_callable_with_kwargs(self):
    """run(fn, key=val) -> fn(key=val), EVAL_CALL_WITH_EXPLICIT_ARGS."""
    result = Chain().then(identity).run(kw_fn, x=42)
    self.assertEqual(result, {'x': 42})

  async def test_temp_link_literal(self):
    """run(literal) -> returns literal, EVAL_RETURN_AS_IS."""
    result = Chain().then(identity).run(42)
    self.assertEqual(result, 42)

  async def test_temp_link_none(self):
    """run(None) -> returns None."""
    result = Chain().then(identity).run(None)
    self.assertIsNone(result)

  async def test_temp_link_with_ellipsis(self):
    """run(fn, ...) -> fn() called with no args, EVAL_CALL_WITHOUT_ARGS."""
    result = Chain().then(identity).run(no_args_fn, ...)
    self.assertEqual(result, 'no_args_result')

  async def test_temp_link_callable_with_args_and_kwargs(self):
    """run(fn, arg, key=val)."""
    result = Chain().then(identity).run(mixed_fn, 5, b=20)
    self.assertEqual(result, (5, 20))


# ===================================================================
# Section 20: Cross-cutting concerns
# ===================================================================

class TestCrossCuttingConcerns(IsolatedAsyncioTestCase):
  """Test interactions between different argument passing mechanisms."""

  async def test_chain_is_reusable_isolation(self):
    """Multiple runs of chain are isolated."""
    c = Chain().then(lambda v: v * 2)
    r1 = c.run(5)
    r2 = c.run(10)
    r3 = c.run(15)
    self.assertEqual(r1, 10)
    self.assertEqual(r2, 20)
    self.assertEqual(r3, 30)

  async def test_clone_preserves_argument_behavior(self):
    """Cloned chain preserves all argument passing semantics."""
    original = Chain().then(lambda v: v + 1).then(add, 10, 20)
    cloned = original.clone()
    result = cloned.run(5)
    self.assertEqual(result, 30)

  async def test_chain_bool_is_true(self):
    """Chain is always truthy."""
    assert bool(Chain()) is True
    assert bool(Chain(42)) is True

  async def test_chain_with_all_eval_codes_in_sequence(self):
    """Chain using all 4 eval codes in sequence."""
    result = (
      Chain(10)                          # root = 10
      .then(lambda v: v + 5)           # EVAL_CALL_WITH_CURRENT_VALUE -> 15
      .then(add, 20, 30)               # EVAL_CALL_WITH_EXPLICIT_ARGS -> 50
      .then(no_args_fn, ...)            # EVAL_CALL_WITHOUT_ARGS -> 'no_args_result'
      .then('final_literal')            # EVAL_RETURN_AS_IS -> 'final_literal'
      .run()
    )
    self.assertEqual(result, 'final_literal')

  async def test_nested_chain_preserves_parent_value(self):
    """After nested chain, parent continues with nested chain's result."""
    inner = Chain().then(lambda v: v + 100)
    result = (
      Chain(5)
      .then(inner)            # inner(5) = 105
      .then(lambda v: v + 1)  # 106
      .run()
    )
    self.assertEqual(result, 106)

  async def test_except_handler_return_value_becomes_result(self):
    """When reraise=False, except handler's return value is the chain result."""
    def raise_fn(v):
      raise TestExc('boom')

    result = (
      Chain(42)
      .then(raise_fn)
      .except_(lambda v: v + 100, reraise=False)
      .run()
    )
    self.assertEqual(result, 142)

  async def test_except_and_finally_both_run_on_exception(self):
    """Both except_ and finally_ run when exception occurs."""
    except_tracker = Tracker()
    finally_tracker = Tracker()

    def raise_fn(v):
      raise TestExc('boom')

    result = (
      Chain(42)
      .then(raise_fn)
      .except_(except_tracker, reraise=False)
      .finally_(finally_tracker)
      .run()
    )
    # except handler returns root_value=42 via Tracker
    self.assertEqual(result, 42)
    assert len(except_tracker.calls) == 1
    assert len(finally_tracker.calls) == 1
    # Both receive root_value
    assert except_tracker.calls[0] == ((42,), {})
    assert finally_tracker.calls[0] == ((42,), {})

  async def test_pipe_run_with_kwargs(self):
    """chain.run(fn, key=val)."""
    result = Chain().then(identity).run(kw_fn, x=99)
    self.assertEqual(result, {'x': 99})
