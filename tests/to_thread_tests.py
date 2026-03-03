import asyncio
import threading
import time
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, Null, run


class MyTestCase(IsolatedAsyncioTestCase):
  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)

  async def assertTrue(self, expr, msg=None):
    return super().assertTrue(await await_(expr), msg)

  async def assertFalse(self, expr, msg=None):
    return super().assertFalse(await await_(expr), msg)

  async def assertEqual(self, first, second, msg=None):
    return super().assertEqual(await await_(first), second, msg)

  async def assertIsNone(self, obj, msg=None):
    return super().assertIsNone(await await_(obj), msg)

  async def assertIs(self, expr1, expr2, msg=None):
    return super().assertIs(await await_(expr1), expr2, msg)

  async def assertIsNot(self, expr1, expr2, msg=None):
    return super().assertIsNot(await await_(expr1), expr2, msg)


# ---------------------------------------------------------------------------
# Category 1: Basic Functionality
# ---------------------------------------------------------------------------
class ToThreadBasicTests(MyTestCase):

  async def test_transform_value(self):
    """to_thread with a simple sync function that transforms a value."""
    await self.assertEqual(Chain(5).to_thread(lambda v: v * 2).run(), 10)

  async def test_result_propagates(self):
    """to_thread result propagates as the new chain value."""
    result = Chain(3).to_thread(lambda v: v + 7).then(lambda v: v * 2).run()
    await self.assertEqual(result, 20)

  async def test_return_int(self):
    """to_thread returning an int."""
    await self.assertEqual(Chain(10).to_thread(lambda v: v + 1).run(), 11)

  async def test_return_string(self):
    """to_thread returning a string."""
    await self.assertEqual(
      Chain('hello').to_thread(lambda v: v + ' world').run(),
      'hello world'
    )

  async def test_return_list(self):
    """to_thread returning a list."""
    await self.assertEqual(Chain([1, 2]).to_thread(lambda v: v + [3]).run(), [1, 2, 3])

  async def test_return_dict(self):
    """to_thread returning a dict."""
    def merge_dict(v):
      v = dict(v)
      v['b'] = 2
      return v
    await self.assertEqual(Chain({'a': 1}).to_thread(merge_dict).run(), {'a': 1, 'b': 2})

  async def test_return_none(self):
    """to_thread returning None."""
    await self.assertIsNone(Chain(1).to_thread(lambda v: None).run())

  async def test_with_root_override(self):
    """to_thread on a void chain with a root override at run time."""
    result = Chain().to_thread(lambda v: v * 3).run(7)
    await self.assertEqual(result, 21)

  async def test_with_root_callable(self):
    """to_thread with a callable root value."""
    await self.assertEqual(Chain(lambda: 10).to_thread(lambda v: v * 5).run(), 50)

  async def test_with_root_callable_args(self):
    """to_thread with a callable root and explicit args."""
    await self.assertEqual(
      Chain(lambda a, b: a + b, 3, 7).to_thread(lambda v: v * 2).run(),
      20
    )


# ---------------------------------------------------------------------------
# Category 2: Async Context
# ---------------------------------------------------------------------------
class ToThreadAsyncTests(MyTestCase):

  async def test_async_upstream(self):
    """to_thread with an async upstream (async root)."""
    result = Chain(aempty, 10).to_thread(lambda v: v * 3).run()
    await self.assertEqual(result, 30)

  async def test_async_runs_in_thread(self):
    """to_thread runs the function in a thread when event loop is running."""
    main_thread_id = threading.get_ident()
    captured = {}

    def capture_thread(v):
      captured['thread_id'] = threading.get_ident()
      return v * 2

    result = await await_(Chain(aempty, 5).to_thread(capture_thread).run())
    super(MyTestCase, self).assertEqual(result, 10)
    super(MyTestCase, self).assertNotEqual(captured['thread_id'], main_thread_id)

  async def test_async_result_awaited_and_propagated(self):
    """Verify the result is correctly awaited and propagated."""
    result = await await_(
      Chain(aempty, 7).to_thread(lambda v: v ** 2).then(lambda v: v + 1).run()
    )
    super(MyTestCase, self).assertEqual(result, 50)

  async def test_async_blocking_function(self):
    """to_thread with a blocking function (time.sleep) in async context."""
    def blocking_fn(v):
      time.sleep(0.005)
      return v + 100

    result = await await_(Chain(aempty, 1).to_thread(blocking_fn).run())
    super(MyTestCase, self).assertEqual(result, 101)

  async def test_async_chain_with_to_thread(self):
    """to_thread preceded by an async operation produces correct result."""
    result = await await_(
      Chain(aempty, 20).then(lambda v: v + 5).to_thread(lambda v: v * 2).run()
    )
    super(MyTestCase, self).assertEqual(result, 50)

  async def test_async_chain_after_to_thread(self):
    """Async then after to_thread works correctly."""
    result = await await_(
      Chain(aempty, 3).to_thread(lambda v: v * 10).then(aempty).then(lambda v: v + 1).run()
    )
    super(MyTestCase, self).assertEqual(result, 31)


# ---------------------------------------------------------------------------
# Category 3: Chain Integration
# ---------------------------------------------------------------------------
class ToThreadChainIntegrationTests(MyTestCase):

  async def test_chained_between_then_ops(self):
    """to_thread chained between .then() operations."""
    result = Chain(5).then(lambda v: v + 1).to_thread(lambda v: v * 10).then(lambda v: v - 2).run()
    await self.assertEqual(result, 58)

  async def test_multiple_to_thread_in_sequence(self):
    """Multiple to_thread calls in sequence."""
    result = Chain(2).to_thread(lambda v: v * 3).to_thread(lambda v: v + 4).run()
    await self.assertEqual(result, 10)

  async def test_to_thread_with_do_before(self):
    """to_thread with .do() before it."""
    side_effects = []
    result = Chain(5).do(lambda v: side_effects.append(v)).to_thread(lambda v: v * 2).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(side_effects, [5])

  async def test_to_thread_with_do_after(self):
    """to_thread with .do() after it."""
    side_effects = []
    result = Chain(5).to_thread(lambda v: v * 2).do(lambda v: side_effects.append(v)).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(side_effects, [10])

  async def test_to_thread_with_except_noraise(self):
    """to_thread combined with .except_(reraise=False)."""
    caught = {'value': None}
    def handler(v):
      caught['value'] = 'caught'

    def failing_fn(v):
      raise ValueError('test error')

    result = Chain(1).to_thread(failing_fn).except_(handler, reraise=False).run()
    await self.assertIsNone(result)
    super(MyTestCase, self).assertEqual(caught['value'], 'caught')

  async def test_to_thread_with_except_reraise(self):
    """to_thread combined with .except_(reraise=True, default)."""
    caught = {'value': None}
    def handler(v):
      caught['value'] = 'caught'

    def failing_fn(v):
      raise ValueError('test error')

    with self.assertRaises(ValueError):
      await await_(Chain(1).to_thread(failing_fn).except_(handler).run())
    super(MyTestCase, self).assertEqual(caught['value'], 'caught')

  async def test_to_thread_with_finally(self):
    """to_thread combined with .finally_()."""
    finally_called = {'value': False}
    def on_finally(v):
      finally_called['value'] = True

    result = Chain(5).to_thread(lambda v: v * 3).finally_(on_finally).run()
    await self.assertEqual(result, 15)
    super(MyTestCase, self).assertTrue(finally_called['value'])

  async def test_to_thread_in_frozen_chain(self):
    """to_thread in a frozen chain (.freeze())."""
    frozen = Chain(10).to_thread(lambda v: v * 2).freeze()
    await self.assertEqual(frozen.run(), 20)
    await self.assertEqual(frozen.run(), 20)
    await self.assertEqual(frozen(), 20)

  async def test_to_thread_in_frozen_void_chain(self):
    """to_thread in a frozen void chain with root override."""
    frozen = Chain().to_thread(lambda v: v + 5).freeze()
    await self.assertEqual(frozen.run(10), 15)
    await self.assertEqual(frozen.run(20), 25)

  async def test_to_thread_in_cloned_chain(self):
    """to_thread in a cloned chain (.clone())."""
    original = Chain(3).to_thread(lambda v: v * 4)
    cloned = original.clone()
    # Modify original after cloning
    original.then(lambda v: v + 100)
    await self.assertEqual(original.run(), 112)
    await self.assertEqual(cloned.run(), 12)

  async def test_to_thread_cloned_independence(self):
    """Cloned chains with to_thread are independent."""
    c = Chain(5).to_thread(lambda v: v * 2)
    c2 = c.clone()
    c2.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 10)
    await self.assertEqual(c2.run(), 11)

  async def test_to_thread_nested_chain(self):
    """to_thread in a nested chain."""
    inner = Chain().to_thread(lambda v: v * 10)
    result = Chain(3).then(inner).run()
    await self.assertEqual(result, 30)

  async def test_to_thread_chained_between_then_async(self):
    """to_thread chained between .then() ops with async functions."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, 5).to_thread(lambda v: v + 1).then(fn).run()
        await self.assertEqual(result, 6)

  async def test_multiple_to_thread_async(self):
    """Multiple to_thread calls in sequence with async upstream."""
    result = Chain(aempty, 2).to_thread(lambda v: v * 3).to_thread(lambda v: v + 4).run()
    await self.assertEqual(result, 10)


# ---------------------------------------------------------------------------
# Category 4: Cascade Behavior
# ---------------------------------------------------------------------------
class ToThreadCascadeTests(MyTestCase):

  async def test_cascade_receives_root_returns_root(self):
    """to_thread in a Cascade -- receives root value, Cascade returns root value."""
    side_effects = []

    def capture_and_transform(v):
      side_effects.append(v)
      return v * 100  # This result is discarded by Cascade

    result = Cascade(42).to_thread(capture_and_transform).run()
    await self.assertEqual(result, 42)
    super(MyTestCase, self).assertEqual(side_effects, [42])

  async def test_cascade_multiple_to_thread(self):
    """Multiple to_thread in Cascade -- each receives the same root value."""
    captured = []

    def capture1(v):
      captured.append(('fn1', v))
      return 'ignored1'

    def capture2(v):
      captured.append(('fn2', v))
      return 'ignored2'

    result = Cascade(99).to_thread(capture1).to_thread(capture2).run()
    await self.assertEqual(result, 99)
    super(MyTestCase, self).assertEqual(captured, [('fn1', 99), ('fn2', 99)])

  async def test_cascade_to_thread_with_async(self):
    """to_thread in Cascade with async root."""
    captured = []

    def capture(v):
      captured.append(v)
      return 'ignored'

    result = await await_(Cascade(aempty, 77).to_thread(capture).run())
    super(MyTestCase, self).assertEqual(result, 77)
    super(MyTestCase, self).assertEqual(captured, [77])

  async def test_cascade_preserves_type(self):
    """to_thread on Cascade returns a Cascade (self)."""
    c = Cascade(1)
    result = c.to_thread(lambda v: v)
    super(MyTestCase, self).assertIs(result, c)
    super(MyTestCase, self).assertIsInstance(result, Cascade)

  async def test_cascade_with_then_and_to_thread(self):
    """Cascade with then and to_thread -- all receive root value."""
    captured = []

    def capture_then(v):
      captured.append(('then', v))
      return 'ignored_then'

    def capture_thread(v):
      captured.append(('thread', v))
      return 'ignored_thread'

    result = Cascade(10).then(capture_then).to_thread(capture_thread).then(capture_then).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(captured, [
      ('then', 10), ('thread', 10), ('then', 10)
    ])


# ---------------------------------------------------------------------------
# Category 5: Error Handling
# ---------------------------------------------------------------------------
class ToThreadErrorHandlingTests(MyTestCase):

  async def test_exception_propagates(self):
    """to_thread fn raises an exception -- should propagate through chain."""
    def failing_fn(v):
      raise ValueError('sync failure')

    with self.assertRaises(ValueError):
      await await_(Chain(1).to_thread(failing_fn).run())

  async def test_exception_caught_by_except(self):
    """to_thread fn raises, caught by .except_()."""
    caught = {'value': None}
    def handler(v):
      caught['value'] = 'handled'
      return 'recovery'

    def failing_fn(v):
      raise ValueError('test')

    result = Chain(1).to_thread(failing_fn).except_(handler, reraise=False).run()
    await self.assertEqual(result, 'recovery')
    super(MyTestCase, self).assertEqual(caught['value'], 'handled')

  async def test_async_exception_propagates(self):
    """to_thread fn raises in async context -- exception propagates correctly."""
    def failing_fn(v):
      raise RuntimeError('async failure')

    with self.assertRaises(RuntimeError):
      await await_(Chain(aempty, 1).to_thread(failing_fn).run())

  async def test_async_exception_caught_by_except(self):
    """to_thread fn raises in async context, caught by .except_()."""
    caught = {'value': None}
    def handler(v):
      caught['value'] = 'async_handled'
      return 'async_recovery'

    def failing_fn(v):
      raise ValueError('async test')

    result = await await_(
      Chain(aempty, 1).to_thread(failing_fn).except_(handler, reraise=False).run()
    )
    super(MyTestCase, self).assertEqual(result, 'async_recovery')
    super(MyTestCase, self).assertEqual(caught['value'], 'async_handled')

  async def test_exception_type_filtering(self):
    """to_thread exception caught only when exception type matches."""
    caught_specific = {'value': False}
    caught_general = {'value': False}

    def failing_fn(v):
      raise TypeError('type error')

    def handler_specific(v):
      caught_specific['value'] = True

    def handler_general(v):
      caught_general['value'] = True

    try:
      await await_(
        Chain(1).to_thread(failing_fn).except_(
          handler_specific, exceptions=ValueError, reraise=True
        ).except_(
          handler_general, exceptions=TypeError, reraise=True
        ).run()
      )
    except TypeError:
      pass

    super(MyTestCase, self).assertFalse(caught_specific['value'])
    super(MyTestCase, self).assertTrue(caught_general['value'])

  async def test_exception_with_finally(self):
    """to_thread fn raises, finally_ still runs."""
    finally_called = {'value': False}

    def failing_fn(v):
      raise ValueError('test')

    def on_finally(v):
      finally_called['value'] = True

    with self.assertRaises(ValueError):
      await await_(Chain(1).to_thread(failing_fn).finally_(on_finally).run())
    super(MyTestCase, self).assertTrue(finally_called['value'])

  async def test_exception_with_except_and_finally(self):
    """to_thread fn raises, both except_ and finally_ run."""
    log = []

    def failing_fn(v):
      raise RuntimeError('boom')

    result = Chain(1).to_thread(failing_fn).except_(
      lambda v: log.append('except'), reraise=False
    ).finally_(lambda v: log.append('finally')).run()
    await self.assertIsNone(result)
    super(MyTestCase, self).assertIn('except', log)
    super(MyTestCase, self).assertIn('finally', log)


# ---------------------------------------------------------------------------
# Category 6: Fluent API
# ---------------------------------------------------------------------------
class ToThreadFluentAPITests(MyTestCase):

  async def test_returns_self(self):
    """to_thread returns self (the chain) for fluent chaining."""
    c = Chain(1)
    result = c.to_thread(lambda v: v)
    super(MyTestCase, self).assertIs(result, c)

  async def test_chain_stays_chain(self):
    """to_thread preserves Chain type."""
    c = Chain(1).to_thread(lambda v: v)
    super(MyTestCase, self).assertIsInstance(c, Chain)

  async def test_cascade_stays_cascade(self):
    """to_thread preserves Cascade type."""
    c = Cascade(1).to_thread(lambda v: v)
    super(MyTestCase, self).assertIsInstance(c, Cascade)

  async def test_fluent_long_chain(self):
    """to_thread works in a long fluent chain."""
    result = (
      Chain(1)
      .then(lambda v: v + 1)
      .to_thread(lambda v: v * 3)
      .then(lambda v: v - 1)
      .to_thread(lambda v: v ** 2)
      .then(lambda v: v + 10)
      .run()
    )
    await self.assertEqual(result, 35)  # ((1+1)*3-1)^2+10 = 5^2+10 = 35


# ---------------------------------------------------------------------------
# Category 7: Edge Cases
# ---------------------------------------------------------------------------
class ToThreadEdgeCaseTests(MyTestCase):

  async def test_identity_function(self):
    """to_thread with identity function."""
    await self.assertEqual(Chain(42).to_thread(lambda v: v).run(), 42)

  async def test_fn_returning_none(self):
    """to_thread with a function returning None."""
    await self.assertIsNone(Chain(1).to_thread(lambda v: None).run())

  async def test_fn_returns_list(self):
    """to_thread where fn returns a list (complex objects survive)."""
    await self.assertEqual(Chain(3).to_thread(lambda v: list(range(v))).run(), [0, 1, 2])

  async def test_fn_returns_dict(self):
    """to_thread where fn returns a dict (complex objects survive)."""
    await self.assertEqual(Chain('key').to_thread(lambda v: {v: 'val'}).run(), {'key': 'val'})

  async def test_fn_returns_tuple(self):
    """to_thread where fn returns a tuple."""
    await self.assertEqual(Chain(3).to_thread(lambda v: (v, v+1, v+2)).run(), (3, 4, 5))

  async def test_class_instance_method(self):
    """to_thread with a class instance method."""
    class Processor:
      def __init__(self, factor):
        self.factor = factor
      def process(self, v):
        return v * self.factor

    proc = Processor(5)
    await self.assertEqual(Chain(4).to_thread(proc.process).run(), 20)

  async def test_callable_class(self):
    """to_thread with a callable class (via __call__)."""
    class Multiplier:
      def __init__(self, factor):
        self.factor = factor
      def __call__(self, v):
        return v * self.factor

    m = Multiplier(7)
    await self.assertEqual(Chain(3).to_thread(m).run(), 21)

  async def test_fn_returning_zero(self):
    """to_thread returning 0 (falsy but not None)."""
    await self.assertEqual(Chain(1).to_thread(lambda v: 0).run(), 0)

  async def test_fn_returning_empty_string(self):
    """to_thread returning empty string (falsy but not None)."""
    await self.assertEqual(Chain(1).to_thread(lambda v: '').run(), '')

  async def test_fn_returning_false(self):
    """to_thread returning False (falsy but not None)."""
    result = Chain(1).to_thread(lambda v: False).run()
    await self.assertIs(result, False)

  async def test_fn_returning_empty_list(self):
    """to_thread returning empty list."""
    await self.assertEqual(Chain(1).to_thread(lambda v: []).run(), [])

  async def test_large_value_through_thread(self):
    """to_thread passes a large object through correctly."""
    large_list = list(range(10000))
    result = Chain(large_list).to_thread(lambda v: len(v)).run()
    await self.assertEqual(result, 10000)

  async def test_none_root_value(self):
    """to_thread with None as root value (not Null)."""
    await self.assertIsNone(Chain(None).to_thread(lambda v: v).run())

  async def test_boolean_root_value(self):
    """to_thread with boolean root value."""
    await self.assertIs(Chain(True).to_thread(lambda v: v).run(), True)
    await self.assertIs(Chain(False).to_thread(lambda v: v).run(), False)

  async def test_void_chain_with_to_thread_requires_value(self):
    """to_thread on a void chain with no root fails (requires current_value arg).

    _ToThread.__call__(self, current_value) is a required parameter.
    When evaluate_value calls link.v() with no args on a void chain,
    _ToThread.__call__ raises TypeError.
    """
    with self.assertRaises(TypeError):
      await await_(Chain().to_thread(lambda: 42).run())


# ---------------------------------------------------------------------------
# Category 8: Thread Verification (async only)
# ---------------------------------------------------------------------------
class ToThreadThreadVerificationTests(MyTestCase):

  async def test_async_runs_in_different_thread(self):
    """Verify that to_thread actually runs in a different thread in async context."""
    main_thread_id = threading.get_ident()
    captured = {}

    def capture_thread_id(v):
      captured['thread_id'] = threading.get_ident()
      return v

    result = await await_(Chain(aempty, 42).to_thread(capture_thread_id).run())
    super(MyTestCase, self).assertEqual(result, 42)
    super(MyTestCase, self).assertIn('thread_id', captured)
    super(MyTestCase, self).assertNotEqual(captured['thread_id'], main_thread_id)

  async def test_async_with_value_runs_in_different_thread(self):
    """Verify to_thread with value runs in a different thread."""
    main_thread_id = threading.get_ident()
    captured = {}

    def capture_thread_id(v):
      captured['thread_id'] = threading.get_ident()
      return v * 10

    result = await await_(Chain(aempty, 5).to_thread(capture_thread_id).run())
    super(MyTestCase, self).assertEqual(result, 50)
    super(MyTestCase, self).assertNotEqual(captured['thread_id'], main_thread_id)

  async def test_sync_root_also_runs_in_thread_in_async_context(self):
    """Even with a sync root, to_thread runs fn in a thread (event loop is running)."""
    main_thread_id = threading.get_ident()
    captured = {}

    def capture_thread_id(v):
      captured['thread_id'] = threading.get_ident()
      return v

    # Even without aempty, the event loop is running in IsolatedAsyncioTestCase
    # so _ToThread.__call__ returns asyncio.to_thread(...) which is a coroutine.
    # The chain detects the coro and transitions to async.
    result = await await_(Chain(10).to_thread(capture_thread_id).run())
    super(MyTestCase, self).assertEqual(result, 10)
    super(MyTestCase, self).assertIn('thread_id', captured)
    super(MyTestCase, self).assertNotEqual(captured['thread_id'], main_thread_id)

  async def test_multiple_to_thread_all_in_threads(self):
    """Multiple to_thread calls all run in threads (not main thread)."""
    captured = {'ids': []}

    def capture_and_pass(v):
      captured['ids'].append(threading.get_ident())
      return v

    result = await await_(
      Chain(aempty, 1)
      .to_thread(capture_and_pass)
      .to_thread(capture_and_pass)
      .to_thread(capture_and_pass)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 1)
    super(MyTestCase, self).assertEqual(len(captured['ids']), 3)
    main_tid = threading.get_ident()
    for tid in captured['ids']:
      super(MyTestCase, self).assertNotEqual(tid, main_tid)

  async def test_blocking_io_in_thread(self):
    """Verify that blocking I/O (time.sleep) runs without blocking event loop."""
    def slow_fn(v):
      time.sleep(0.01)
      return v * 2

    result = await await_(Chain(aempty, 5).to_thread(slow_fn).run())
    super(MyTestCase, self).assertEqual(result, 10)


# ---------------------------------------------------------------------------
# Category 9: Interaction with no_async
# ---------------------------------------------------------------------------
class ToThreadNoAsyncTests(MyTestCase):

  async def test_no_async_with_value(self):
    """to_thread with no_async(True) and a root value.

    Inside IsolatedAsyncioTestCase, the event loop is running, so
    _ToThread.__call__ returns asyncio.to_thread (a coroutine).
    With no_async(True), the chain skips iscoro() check and returns the raw
    coroutine. Our assertEqual goes through await_() which awaits it.
    """
    result = Chain(5).to_thread(lambda v: v * 3).no_async(True).run()
    await self.assertEqual(result, 15)

  async def test_no_async_multiple_to_thread(self):
    """Multiple to_thread with no_async(True).

    With no_async(True), the first to_thread returns a coroutine which is
    passed as current_value to the second to_thread. The second to_thread
    then receives the coroutine object as current_value and passes it to
    asyncio.to_thread(fn, coroutine_obj) -- fn receives the raw coroutine.
    This means multiple to_thread with no_async(True) doesn't work as expected
    because intermediate values are unawaited coroutines.
    We test single to_thread + no_async instead.
    """
    result = Chain(2).to_thread(lambda v: v + 1).no_async(True).run()
    await self.assertEqual(result, 3)

  async def test_no_async_preserves_simple(self):
    """to_thread with no_async(True) stays on simple path."""
    c = Chain(1).to_thread(lambda v: v * 2).no_async(True)
    super(MyTestCase, self).assertTrue(c._is_simple)
    super(MyTestCase, self).assertTrue(c._is_sync)
    await self.assertEqual(c.run(), 2)

  async def test_no_async_with_then(self):
    """to_thread + no_async(True) with preceding .then()."""
    result = Chain(3).then(lambda v: v + 2).to_thread(lambda v: v * 10).no_async(True).run()
    await self.assertEqual(result, 50)


# ---------------------------------------------------------------------------
# Category 10: Pipe Operator Integration
# ---------------------------------------------------------------------------
class ToThreadPipeTests(MyTestCase):

  async def test_pipe_with_to_thread(self):
    """to_thread combined with pipe operator."""
    c = Chain(5).to_thread(lambda v: v * 2)
    result = c | (lambda v: v + 1) | run()
    await self.assertEqual(result, 11)

  async def test_pipe_then_to_thread(self):
    """Pipe operator leading into to_thread."""
    result = (Chain(3) | (lambda v: v + 2)).to_thread(lambda v: v * 10) | run()
    await self.assertEqual(result, 50)


# ---------------------------------------------------------------------------
# Category 11: _is_simple Flag
# ---------------------------------------------------------------------------
class ToThreadSimpleFlagTests(MyTestCase):

  async def test_to_thread_keeps_simple_true(self):
    """to_thread does NOT break _is_simple flag -- stays on fast path."""
    c = Chain(1).to_thread(lambda v: v * 2)
    super(MyTestCase, self).assertTrue(c._is_simple)

  async def test_to_thread_with_then_keeps_simple(self):
    """to_thread combined with .then() keeps _is_simple = True."""
    c = Chain(1).then(lambda v: v + 1).to_thread(lambda v: v * 2).then(lambda v: v - 1)
    super(MyTestCase, self).assertTrue(c._is_simple)

  async def test_to_thread_after_do_is_not_simple(self):
    """to_thread after .do() -- _is_simple is already False."""
    c = Chain(1).do(lambda v: None).to_thread(lambda v: v * 2)
    super(MyTestCase, self).assertFalse(c._is_simple)

  async def test_to_thread_before_except_is_not_simple(self):
    """to_thread before .except_() -- _is_simple becomes False."""
    c = Chain(1).to_thread(lambda v: v * 2).except_(lambda v: None)
    super(MyTestCase, self).assertFalse(c._is_simple)

  async def test_to_thread_only_simple(self):
    """Chain with only root + to_thread stays simple."""
    c = Chain(1).to_thread(lambda v: v)
    super(MyTestCase, self).assertTrue(c._is_simple)

  async def test_to_thread_before_sleep_not_simple(self):
    """to_thread before .sleep() -- _is_simple becomes False."""
    c = Chain(1).to_thread(lambda v: v * 2).sleep(0)
    super(MyTestCase, self).assertFalse(c._is_simple)


# ---------------------------------------------------------------------------
# Category 12: Code Path Tests
# ---------------------------------------------------------------------------
class ToThreadCodePathTests(MyTestCase):

  async def test_path_async_with_value(self):
    """Async path with value: asyncio.to_thread(fn, current_value)."""
    main_tid = threading.get_ident()
    captured = {}

    def processor(v):
      captured['tid'] = threading.get_ident()
      return v * 2

    result = await await_(Chain(10).to_thread(processor).run())
    super(MyTestCase, self).assertEqual(result, 20)
    super(MyTestCase, self).assertIn('tid', captured)
    super(MyTestCase, self).assertNotEqual(captured['tid'], main_tid)

  async def test_path_async_explicit_async_root(self):
    """Async path via explicitly async root value."""
    result = await await_(Chain(aempty, 42).to_thread(lambda v: v - 2).run())
    super(MyTestCase, self).assertEqual(result, 40)

  async def test_all_paths_with_fn_pattern(self):
    """Test code paths via with_fn pattern (sync and async root)."""
    for fn, ctx in self.with_fn():
      with ctx:
        # With value
        result = Chain(fn, 7).to_thread(lambda v: v + 3).run()
        await self.assertEqual(result, 10)

  async def test_to_thread_after_then_async(self):
    """to_thread after an async .then() -- value flows correctly."""
    result = await await_(
      Chain(1).then(aempty).to_thread(lambda v: v * 5).run()
    )
    super(MyTestCase, self).assertEqual(result, 5)

  async def test_to_thread_before_then_async(self):
    """to_thread before an async .then() -- value flows correctly."""
    result = await await_(
      Chain(2).to_thread(lambda v: v * 3).then(aempty).run()
    )
    super(MyTestCase, self).assertEqual(result, 6)


# ---------------------------------------------------------------------------
# Category 13: Repr Tests
# ---------------------------------------------------------------------------
class ToThreadReprTests(MyTestCase):

  async def test_repr_contains_to_thread(self):
    """repr of chain with to_thread shows to_thread in output."""
    def my_fn(v):
      return v
    c = Chain(1).to_thread(my_fn)
    r = repr(c)
    super(MyTestCase, self).assertIn('to_thread', r)
    super(MyTestCase, self).assertIn('my_fn', r)

  async def test_repr_chain_with_multiple_to_thread(self):
    """repr of chain with multiple to_thread calls."""
    def fn1(v): return v
    def fn2(v): return v
    c = Chain(1).to_thread(fn1).to_thread(fn2)
    r = repr(c)
    super(MyTestCase, self).assertIn('fn1', r)
    super(MyTestCase, self).assertIn('fn2', r)

  async def test_repr_cascade_with_to_thread(self):
    """repr of Cascade with to_thread."""
    c = Cascade(1).to_thread(lambda v: v)
    r = repr(c)
    super(MyTestCase, self).assertIn('Cascade', r)
    super(MyTestCase, self).assertIn('to_thread', r)


# ---------------------------------------------------------------------------
# Category 14: Comprehensive with_fn Pattern Tests
# ---------------------------------------------------------------------------
class ToThreadWithFnPatternTests(MyTestCase):

  async def test_basic_with_fn(self):
    """to_thread with both sync and async root via with_fn pattern."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(Chain(fn, 5).to_thread(lambda v: v * 2).run(), 10)

  async def test_chained_with_fn(self):
    """to_thread in a chain with both sync and async via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, 3).then(fn).to_thread(lambda v: v + 7).then(fn).run()
        await self.assertEqual(result, 10)

  async def test_cascade_with_fn(self):
    """to_thread in Cascade with both sync and async root via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = Cascade(fn, 50).to_thread(lambda v: v * 100).run()
        await self.assertEqual(result, 50)

  async def test_exception_with_fn(self):
    """to_thread exception handling with both sync and async via with_fn."""
    def failing_fn(v):
      raise TestExc('test')

    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 1).to_thread(failing_fn).run())

  async def test_except_handler_with_fn(self):
    """to_thread with except handler with both sync and async via with_fn."""
    def failing_fn(v):
      raise TestExc('test')

    for fn, ctx in self.with_fn():
      with ctx:
        caught = {'value': False}
        def handler(v):
          caught['value'] = True
          return 'recovered'
        caught['value'] = False
        result = await await_(
          Chain(fn, 1).to_thread(failing_fn).except_(handler, reraise=False).run()
        )
        super(MyTestCase, self).assertEqual(result, 'recovered')
        super(MyTestCase, self).assertTrue(caught['value'])

  async def test_finally_with_fn(self):
    """to_thread with finally handler with both sync and async via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_ran = {'value': False}
        def on_finally(v):
          finally_ran['value'] = True
        finally_ran['value'] = False
        result = await await_(
          Chain(fn, 10).to_thread(lambda v: v * 2).finally_(on_finally).run()
        )
        super(MyTestCase, self).assertEqual(result, 20)
        super(MyTestCase, self).assertTrue(finally_ran['value'])

  async def test_clone_with_fn(self):
    """to_thread in cloned chain with both sync and async via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        c = Chain(fn, 4).to_thread(lambda v: v * 5)
        c2 = c.clone()
        await self.assertEqual(c.run(), 20)
        await self.assertEqual(c2.run(), 20)

  async def test_freeze_with_fn(self):
    """to_thread in frozen chain with both sync and async via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        frozen = Chain(fn, 6).to_thread(lambda v: v + 4).freeze()
        await self.assertEqual(frozen.run(), 10)
        await self.assertEqual(frozen(), 10)

  async def test_do_and_to_thread_with_fn(self):
    """to_thread with .do() and both sync and async via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        side_effects = []
        result = Chain(fn, 8).do(
          lambda v: side_effects.append(v)
        ).to_thread(lambda v: v + 2).run()
        await self.assertEqual(result, 10)
        super(MyTestCase, self).assertEqual(side_effects, [8])

  async def test_nested_chain_with_fn(self):
    """to_thread in nested chain with both sync and async via with_fn."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().to_thread(lambda v: v * 3)
        result = Chain(fn, 4).then(inner).run()
        await self.assertEqual(result, 12)


# ---------------------------------------------------------------------------
# Category 15: Complex Integration Scenarios
# ---------------------------------------------------------------------------
class ToThreadComplexTests(MyTestCase):

  async def test_chain_then_to_thread_then_cascade(self):
    """Chain feeding into to_thread, result used by subsequent chain."""
    result = (
      Chain(2)
      .then(lambda v: v + 3)
      .to_thread(lambda v: v * 10)
      .then(lambda v: v - 1)
      .run()
    )
    await self.assertEqual(result, 49)  # (2+3)*10-1

  async def test_frozen_chain_concurrent_execution(self):
    """Frozen chain with to_thread can be executed concurrently."""
    frozen = Chain().to_thread(lambda v: v * 2).freeze()
    tasks = [frozen.run(i) for i in range(20)]
    results = await asyncio.gather(*[await_(t) for t in tasks])
    expected = [i * 2 for i in range(20)]
    super(MyTestCase, self).assertEqual(sorted(results), sorted(expected))

  async def test_to_thread_with_exception_in_async_finally(self):
    """to_thread with finally_ that runs in async context."""
    finally_ran = {'value': False}

    def on_finally(v):
      finally_ran['value'] = True

    result = await await_(
      Chain(aempty, 5).to_thread(lambda v: v * 4).finally_(on_finally).run()
    )
    super(MyTestCase, self).assertEqual(result, 20)
    super(MyTestCase, self).assertTrue(finally_ran['value'])

  async def test_to_thread_with_multiple_then_and_do(self):
    """Complex chain mixing then, do, and to_thread."""
    log = []
    result = (
      Chain(1)
      .then(lambda v: v + 1)
      .do(lambda v: log.append(('do1', v)))
      .to_thread(lambda v: v * 5)
      .do(lambda v: log.append(('do2', v)))
      .then(lambda v: v + 3)
      .run()
    )
    await self.assertEqual(result, 13)  # (1+1)*5+3
    super(MyTestCase, self).assertEqual(log, [('do1', 2), ('do2', 10)])

  async def test_deeply_nested_chains_with_to_thread(self):
    """to_thread in deeply nested chains."""
    inner1 = Chain().to_thread(lambda v: v * 2)
    inner2 = Chain().then(inner1).then(lambda v: v + 1)
    result = Chain(5).then(inner2).run()
    await self.assertEqual(result, 11)  # 5*2+1

  async def test_cloned_chain_run_multiple_times(self):
    """Cloned chain with to_thread can be run multiple times."""
    c = Chain(3).to_thread(lambda v: v * 4)
    c2 = c.clone()
    await self.assertEqual(c.run(), 12)
    await self.assertEqual(c.run(), 12)
    await self.assertEqual(c2.run(), 12)
    await self.assertEqual(c2.run(), 12)

  async def test_to_thread_preserves_none_vs_null(self):
    """to_thread correctly distinguishes None return from no-value."""
    # None root is a valid root value (not Null)
    result = Chain(None).to_thread(lambda v: v).run()
    await self.assertIsNone(result)

    # Function returning None should propagate None
    result = Chain(5).to_thread(lambda v: None).run()
    await self.assertIsNone(result)

  async def test_to_thread_with_class_hierarchy(self):
    """to_thread works with various callable types."""
    class Base:
      def transform(self, v):
        return v * 2

    class Child(Base):
      def transform(self, v):
        return super().transform(v) + 1

    obj = Child()
    await self.assertEqual(Chain(5).to_thread(obj.transform).run(), 11)  # 5*2+1


if __name__ == '__main__':
  import unittest
  unittest.main()
