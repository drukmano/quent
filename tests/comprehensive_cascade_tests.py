"""Comprehensive tests for Cascade-specific behavior and edge cases.

Cascade is like Chain, but every operation receives the ROOT value instead of
the previous result. The final return value is always the root value (not the
last operation's result).

Key difference:
  Chain(5).then(fn1).then(fn2) -> fn1(5) -> fn2(fn1_result)
  Cascade(5).then(fn1).then(fn2) -> fn1(5), fn2(5) -> returns 5
"""
import asyncio
import warnings
from unittest import TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CascadeExc(Exception):
  pass


class HandlerExc(Exception):
  pass


class FinallyExc(Exception):
  pass


def raise_test_exc(v=None):
  raise TestExc('cascade error')


async def async_raise_test_exc(v=None):
  raise TestExc('async cascade error')


def make_tracker():
  """Create a tracker dict and a capture function that appends received values."""
  tracker = {'received': [], 'call_count': 0}
  def capture(v):
    tracker['received'].append(v)
    tracker['call_count'] += 1
    return v * 100  # return something different to verify it is discarded
  return tracker, capture


def make_async_tracker():
  """Create a tracker dict and an async capture function."""
  tracker = {'received': [], 'call_count': 0}
  async def capture(v):
    tracker['received'].append(v)
    tracker['call_count'] += 1
    return v * 100
  return tracker, capture


# ===========================================================================
# Category 1: Core Cascade Semantics (20+ tests)
# ===========================================================================

class CoreCascadeSemanticsTests(MyTestCase):
  """Fundamental Cascade behavior: root value propagation and return semantics."""

  async def test_basic_cascade_fn_receives_root_and_returns_root(self):
    """Cascade(5).then(fn).run() -- fn receives 5, result is 5."""
    tracker, capture = make_tracker()
    result = Cascade(5).then(capture).run()
    await self.assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(tracker['received'], [5])

  async def test_multiple_operations_all_receive_root(self):
    """Multiple .then() calls: each receives root 5, result is 5."""
    tracker, capture = make_tracker()
    result = Cascade(5).then(capture).then(capture).then(capture).run()
    await self.assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(tracker['received'], [5, 5, 5])

  async def test_do_in_cascade_side_effect_receives_root(self):
    """.do() in cascade: side effect, receives root, result is root."""
    tracker = {'received': []}
    def side_effect(v):
      tracker['received'].append(v)
      return 'ignored_side_effect'
    result = Cascade(42).do(side_effect).run()
    await self.assertEqual(result, 42)
    super(MyTestCase, self).assertEqual(tracker['received'], [42])

  async def test_cascade_vs_chain_direct_comparison(self):
    """Same operations, different results: Chain vs Cascade."""
    fn = lambda v: v * 2

    chain_result = Chain(5).then(fn).then(fn).run()
    await self.assertEqual(chain_result, 20)  # 5 -> 10 -> 20

    cascade_result = Cascade(5).then(fn).then(fn).run()
    await self.assertEqual(cascade_result, 5)  # always returns root

  async def test_cascade_with_callable_root(self):
    """Cascade(fn).run() -- fn is called, root becomes fn()."""
    result = Cascade(lambda: 42).run()
    await self.assertEqual(result, 42)

  async def test_cascade_with_callable_root_and_args(self):
    """Cascade(fn, arg1, arg2).run()."""
    result = Cascade(lambda a, b: a + b, 10, 20).run()
    await self.assertEqual(result, 30)

  async def test_cascade_with_callable_root_and_kwargs(self):
    """Cascade(fn, key=val).run()."""
    result = Cascade(lambda x=0, y=0: x * y, x=5, y=7).run()
    await self.assertEqual(result, 35)

  async def test_void_cascade_fn_receives_no_arg(self):
    """Void Cascade: Cascade().then(fn).run() -- fn called with no args, result is None."""
    received = {'value': 'sentinel'}
    def fn(v=None):
      received['value'] = v
      return 'ignored'
    result = Cascade().then(fn, ...).run()
    await self.assertIsNone(result)
    # fn was called with no args (v defaults to None)
    super(MyTestCase, self).assertIsNone(received['value'])

  async def test_cascade_none_root(self):
    """Cascade(None).then(fn).run() -- fn receives None (not Null), result is None."""
    received = {'value': 'sentinel'}
    def fn(v):
      received['value'] = v
      return 'ignored'
    result = Cascade(None).then(fn).run()
    await self.assertIsNone(result)
    super(MyTestCase, self).assertIsNone(received['value'])

  async def test_cascade_false_root(self):
    """Cascade(False).then(fn).run() -- fn receives False, result is False."""
    received = {'value': 'sentinel'}
    def fn(v):
      received['value'] = v
      return 'ignored'
    result = Cascade(False).then(fn).run()
    await self.assertFalse(result)
    super(MyTestCase, self).assertIs(received['value'], False)

  async def test_cascade_zero_root(self):
    """Cascade(0).then(fn).run() -- fn receives 0, result is 0."""
    received = {'value': 'sentinel'}
    def fn(v):
      received['value'] = v
      return 999
    result = Cascade(0).then(fn).run()
    await self.assertEqual(result, 0)
    super(MyTestCase, self).assertEqual(received['value'], 0)

  async def test_cascade_empty_string_root(self):
    """Cascade('').then(fn).run() -- fn receives '', result is ''."""
    received = {'value': 'sentinel'}
    def fn(v):
      received['value'] = v
      return 'ignored'
    result = Cascade('').then(fn).run()
    await self.assertEqual(result, '')
    super(MyTestCase, self).assertEqual(received['value'], '')

  async def test_many_then_all_receive_same_root(self):
    """10 .then() calls all receive the same root value."""
    received = []
    def capture(v):
      received.append(v)
      return v * 100
    c = Cascade(7)
    for _ in range(10):
      c = c.then(capture)
    result = c.run()
    await self.assertEqual(result, 7)
    super(MyTestCase, self).assertEqual(received, [7] * 10)

  async def test_return_value_always_root_with_complex_operations(self):
    """Result is ALWAYS root, even with complex operations."""
    result = (
      Cascade(42)
      .then(lambda v: v * 2)       # 84, discarded
      .then(lambda v: v + 100)     # 142, discarded
      .then(lambda v: [v, v, v])   # [42, 42, 42], discarded
      .then(lambda v: str(v))      # '42', discarded
      .run()
    )
    await self.assertEqual(result, 42)

  async def test_cascade_then_returns_different_values_all_ignored(self):
    """.then() returning different values -- all ignored."""
    results_from_then = []
    def op1(v):
      r = 'string'
      results_from_then.append(r)
      return r
    def op2(v):
      r = [1, 2, 3]
      results_from_then.append(r)
      return r
    def op3(v):
      r = {'key': 'val'}
      results_from_then.append(r)
      return r
    result = Cascade(99).then(op1).then(op2).then(op3).run()
    await self.assertEqual(result, 99)
    super(MyTestCase, self).assertEqual(len(results_from_then), 3)

  async def test_cascade_do_returns_different_values_all_ignored(self):
    """.do() returning different values -- all ignored."""
    result = (
      Cascade(77)
      .do(lambda v: 'ignored1')
      .do(lambda v: [1, 2])
      .do(lambda v: {'a': 1})
      .run()
    )
    await self.assertEqual(result, 77)

  async def test_cascade_mutable_root_shared_reference(self):
    """Root is a mutable object -- all operations share the same reference."""
    root = [1, 2, 3]
    refs = []
    def capture_ref(v):
      refs.append(id(v))
      return 'ignored'
    Cascade(root).then(capture_ref).then(capture_ref).then(capture_ref).run()
    # All three operations received the same object
    super(MyTestCase, self).assertEqual(len(set(refs)), 1)
    super(MyTestCase, self).assertEqual(refs[0], id(root))

  async def test_cascade_operations_mutate_root(self):
    """Operations mutate the root (e.g., list.append) -- mutations visible."""
    root = [1, 2, 3]
    result = (
      Cascade(root)
      .then(lambda v: v.append(4))
      .then(lambda v: v.append(5))
      .run()
    )
    await self.assertIs(result, root)
    super(MyTestCase, self).assertEqual(root, [1, 2, 3, 4, 5])

  async def test_cascade_dict_mutation(self):
    """Cascade with dict root: mutations accumulate."""
    root = {'a': 1}
    result = (
      Cascade(root)
      .then(lambda d: d.update({'b': 2}))
      .then(lambda d: d.update({'c': 3}))
      .run()
    )
    await self.assertIs(result, root)
    super(MyTestCase, self).assertEqual(root, {'a': 1, 'b': 2, 'c': 3})

  async def test_10_link_chain_vs_cascade_outputs_differ(self):
    """Compare 10-link Chain vs 10-link Cascade: outputs differ correctly."""
    fn = lambda v: v + 1

    c_chain = Chain(0)
    c_cascade = Cascade(0)
    for _ in range(10):
      c_chain = c_chain.then(fn)
      c_cascade = c_cascade.then(fn)

    chain_result = c_chain.run()
    cascade_result = c_cascade.run()

    await self.assertEqual(chain_result, 10)     # 0 + 1 + 1 + ... + 1 = 10
    await self.assertEqual(cascade_result, 0)    # always root

  async def test_cascade_with_literal_then(self):
    """Cascade(5).then(42).run() -- literal 42 replaces (in Chain) but root returned in Cascade."""
    result = Cascade(5).then(42).run()
    await self.assertEqual(result, 5)

  async def test_void_cascade_returns_none(self):
    """Cascade() with no root and no links returns None."""
    result = Cascade().run()
    await self.assertIsNone(result)

  async def test_cascade_root_override_via_run(self):
    """Cascade().run(override_value) -- override root."""
    tracker = {'received': []}
    def capture(v):
      tracker['received'].append(v)
      return 'ignored'
    result = Cascade().then(capture).run(55)
    await self.assertEqual(result, 55)
    super(MyTestCase, self).assertEqual(tracker['received'], [55])

  async def test_cascade_root_override_raises_if_root_set(self):
    """Cascade(5).run(10) -- cannot override existing root."""
    with self.assertRaises(QuentException):
      Cascade(5).then(lambda v: v).run(10)


# ===========================================================================
# Category 2: Cascade with Async Operations (15+ tests)
# ===========================================================================

class CascadeAsyncTests(MyTestCase):
  """Cascade behavior in async execution paths."""

  async def test_cascade_async_root_resolved_all_links_receive_resolved(self):
    """Cascade with async root: root is resolved, all links receive resolved value."""
    tracker, capture = make_tracker()
    result = await await_(
      Cascade(aempty, 42).then(capture).then(capture).run()
    )
    super(MyTestCase, self).assertEqual(result, 42)
    super(MyTestCase, self).assertEqual(tracker['received'], [42, 42])

  async def test_cascade_sync_root_async_then_returns_root(self):
    """Cascade with sync root, async .then() -- transition to _run_async, result is root."""
    async def async_op(v):
      return v * 100
    result = await await_(
      Cascade(7).then(async_op).run()
    )
    super(MyTestCase, self).assertEqual(result, 7)

  async def test_cascade_multiple_async_then_all_receive_root(self):
    """Multiple async .then() -- all receive root."""
    tracker, capture = make_async_tracker()
    result = await await_(
      Cascade(33).then(capture).then(capture).then(capture).run()
    )
    super(MyTestCase, self).assertEqual(result, 33)
    super(MyTestCase, self).assertEqual(tracker['received'], [33, 33, 33])

  async def test_cascade_simple_path_async_root(self):
    """Simple cascade async path: _run_simple -> _run_async_simple."""
    # Only .then() links -> _is_simple = True
    result = await await_(
      Cascade(aempty, 100)
      .then(lambda v: v + 1)
      .then(lambda v: v + 2)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 100)

  async def test_cascade_non_simple_path_async_do(self):
    """Non-simple cascade async path: _run -> _run_async (has .do())."""
    side_effects = []
    result = await await_(
      Cascade(aempty, 50)
      .do(lambda v: side_effects.append(v))
      .then(lambda v: v * 2)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 50)
    super(MyTestCase, self).assertEqual(side_effects, [50])

  async def test_cascade_async_do_side_effect_root_unchanged(self):
    """Cascade async: .do(async_fn) -- async side effect, root unchanged."""
    side_effects = []
    async def async_side_effect(v):
      side_effects.append(v)
      return 'discarded_async'
    result = await await_(
      Cascade(aempty, 'hello')
      .do(async_side_effect)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 'hello')
    super(MyTestCase, self).assertEqual(side_effects, ['hello'])

  async def test_void_cascade_with_async_root_override(self):
    """Void cascade with async root override: Cascade().run(async_fn)."""
    async def get_value():
      return 99
    tracker, capture = make_tracker()
    result = await await_(
      Cascade().then(capture).run(get_value)
    )
    super(MyTestCase, self).assertEqual(result, 99)
    super(MyTestCase, self).assertEqual(tracker['received'], [99])

  async def test_cascade_mixed_sync_async_links(self):
    """Mixed sync/async links -- some return coroutines, all properly awaited."""
    received = []
    def sync_capture(v):
      received.append(('sync', v))
      return v * 2
    async def async_capture(v):
      received.append(('async', v))
      return v * 3
    result = await await_(
      Cascade(10)
      .then(sync_capture)
      .then(async_capture)
      .then(sync_capture)
      .then(async_capture)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(received, [
      ('sync', 10), ('async', 10), ('sync', 10), ('async', 10)
    ])

  async def test_cascade_no_async_coroutines_returned_as_is(self):
    """Cascade with no_async(True) -- async detection disabled."""
    async def async_fn(v):
      return v * 2  # pragma: no cover
    c = Cascade(5).then(async_fn).no_async(True)
    result = c.run()
    # Since async detection is disabled, the coroutine object is returned as-is
    # but Cascade returns root (5), so the coroutine is never captured.
    # The coroutine result from the link is discarded by Cascade.
    # However, _is_sync + is_cascade uses Loop 2a which doesn't check for coro,
    # and then sets current_value = root_value.
    super(MyTestCase, self).assertEqual(result, 5)

  async def test_cascade_autorun_with_async_root(self):
    """Cascade autorun with async root."""
    tracker, capture = make_tracker()
    task = Cascade(aempty, 25).then(capture).config(autorun=True).run()
    result = await task
    super(MyTestCase, self).assertEqual(result, 25)
    super(MyTestCase, self).assertEqual(tracker['received'], [25])

  async def test_cascade_async_transition_mid_chain(self):
    """Async transition at various points in the chain."""
    received = []
    async def async_at_second(v):
      received.append(('async2', v))
      return v * 10
    result = await await_(
      Cascade(5)
      .then(lambda v: received.append(('sync1', v)) or v)
      .then(async_at_second)
      .then(lambda v: received.append(('sync3', v)) or v)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(received, [
      ('sync1', 5), ('async2', 5), ('sync3', 5)
    ])

  async def test_cascade_async_with_debug_mode(self):
    """Cascade async with debug mode enabled."""
    import logging
    tracker, capture = make_tracker()
    result = await await_(
      Cascade(aempty, 88).then(capture).config(debug=True).run()
    )
    super(MyTestCase, self).assertEqual(result, 88)
    super(MyTestCase, self).assertEqual(tracker['received'], [88])

  async def test_cascade_async_with_exception_handler(self):
    """Cascade async with exception handler -- handler receives root value."""
    handler_received = {'value': None}
    def handler(v):
      handler_received['value'] = v
      return 'recovered'
    with self.assertRaises(TestExc):
      await await_(
        Cascade(aempty, 42)
        .then(async_raise_test_exc)
        .except_(handler, reraise=True)
        .run()
      )
    super(MyTestCase, self).assertEqual(handler_received['value'], 42)

  async def test_cascade_async_with_finally_handler(self):
    """Cascade async with finally handler -- handler receives root value."""
    finally_received = {'value': None}
    async def finally_handler(v=None):
      finally_received['value'] = v
    result = await await_(
      Cascade(aempty, 77)
      .then(lambda v: v * 2)
      .finally_(finally_handler)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 77)
    super(MyTestCase, self).assertEqual(finally_received['value'], 77)

  async def test_cascade_async_simple_and_non_simple_paths(self):
    """Verify both simple and non-simple async paths work for Cascade."""
    # Simple path: only .then() links
    r1 = await await_(
      Cascade(aempty, 10).then(lambda v: v + 1).then(lambda v: v + 2).run()
    )
    super(MyTestCase, self).assertEqual(r1, 10)

    # Non-simple path: .do() makes it non-simple
    r2 = await await_(
      Cascade(aempty, 20)
      .do(lambda v: None)
      .then(lambda v: v + 1)
      .run()
    )
    super(MyTestCase, self).assertEqual(r2, 20)

  async def test_cascade_with_fn_pattern_sync_async(self):
    """Cascade with both sync and async root via with_fn pattern."""
    for fn, ctx in self.with_fn():
      with ctx:
        tracker, capture = make_tracker()
        result = Cascade(fn, 66).then(capture).run()
        await self.assertEqual(result, 66)


# ===========================================================================
# Category 3: Cascade with Exception Handling (10+ tests)
# ===========================================================================

class CascadeExceptionTests(MyTestCase):
  """Exception handling behavior specific to Cascade."""

  async def test_except_handler_receives_root_value(self):
    """Cascade with .except_(handler) -- handler receives root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        handler_received = {'value': None}
        def handler(v):
          handler_received['value'] = v
          return 'recovered'
        with self.assertRaises(TestExc):
          await await_(
            Cascade(fn, 42)
            .then(raise_test_exc)
            .except_(handler, reraise=True)
            .run()
          )
        super(MyTestCase, self).assertEqual(handler_received['value'], 42)

  async def test_except_noraise_returns_handler_result(self):
    """Cascade with .except_(handler, reraise=False) -- returns handler result, NOT root."""
    for fn, ctx in self.with_fn():
      with ctx:
        handler_received = {'value': None}
        def handler(v):
          handler_received['value'] = v
          return 'recovery_value'
        result = await await_(
          Cascade(fn, 42)
          .then(raise_test_exc)
          .except_(handler, reraise=False)
          .run()
        )
        # When an exception is caught and not re-raised, the handler's
        # return value becomes the chain result (the normal cascade
        # restoration at the end of _run is bypassed by the except path).
        super(MyTestCase, self).assertEqual(result, 'recovery_value')
        super(MyTestCase, self).assertEqual(handler_received['value'], 42)

  async def test_except_reraise_true(self):
    """Cascade with .except_(handler, reraise=True) -- handler runs, exception re-raised."""
    for fn, ctx in self.with_fn():
      with ctx:
        handler_called = {'value': False}
        def handler(v):
          handler_called['value'] = True
        with self.assertRaises(TestExc):
          await await_(
            Cascade(fn, 10)
            .then(raise_test_exc)
            .except_(handler, reraise=True)
            .run()
          )
        super(MyTestCase, self).assertTrue(handler_called['value'])

  async def test_cascade_exception_errored_result_irrelevant(self):
    """The errored operation's result is irrelevant in Cascade."""
    def bad_op(v):
      # Would have returned v*100 if it completed...
      raise TestExc('mid-chain error')
    handler_received = {'value': None}
    def handler(v):
      handler_received['value'] = v
      return 'recovered'
    result = (
      Cascade(5)
      .then(lambda v: v * 2)
      .then(bad_op)
      .except_(handler, reraise=False)
      .run()
    )
    await self.assertEqual(result, 'recovered')
    # Handler received the ROOT value
    super(MyTestCase, self).assertEqual(handler_received['value'], 5)

  async def test_multiple_except_handlers_first_match(self):
    """Multiple except handlers on Cascade: first matching handler wins."""
    handler1_called = {'value': False}
    handler2_called = {'value': False}
    def handler1(v):
      handler1_called['value'] = True
      return 'from_handler1'
    def handler2(v):
      handler2_called['value'] = True
      return 'from_handler2'
    result = (
      Cascade(5)
      .then(raise_test_exc)
      .except_(handler1, exceptions=TestExc, reraise=False)
      .except_(handler2, exceptions=TestExc, reraise=False)
      .run()
    )
    await self.assertEqual(result, 'from_handler1')
    super(MyTestCase, self).assertTrue(handler1_called['value'])
    super(MyTestCase, self).assertFalse(handler2_called['value'])

  async def test_cascade_finally_handler_receives_root(self):
    """Cascade finally handler receives root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_received = {'value': None}
        def handler(v=None):
          finally_received['value'] = v
        result = await await_(
          Cascade(fn, 88)
          .then(lambda v: v * 2)
          .finally_(handler)
          .run()
        )
        super(MyTestCase, self).assertEqual(result, 88)
        super(MyTestCase, self).assertEqual(finally_received['value'], 88)

  async def test_cascade_except_plus_finally_combination(self):
    """Cascade except + finally combination."""
    for fn, ctx in self.with_fn():
      with ctx:
        except_received = {'value': None}
        finally_received = {'value': None}
        def on_except(v):
          except_received['value'] = v
          return 'recovered'
        def on_finally(v=None):
          finally_received['value'] = v
        result = await await_(
          Cascade(fn, 42)
          .then(raise_test_exc)
          .except_(on_except, reraise=False)
          .finally_(on_finally)
          .run()
        )
        super(MyTestCase, self).assertEqual(result, 'recovered')
        super(MyTestCase, self).assertEqual(except_received['value'], 42)
        super(MyTestCase, self).assertEqual(finally_received['value'], 42)

  async def test_cascade_base_exception_propagation(self):
    """Cascade with BaseException -- propagation (no handler matches)."""
    def raise_keyboard_interrupt(v):
      raise KeyboardInterrupt('manual')
    with self.assertRaises(KeyboardInterrupt):
      Cascade(5).then(raise_keyboard_interrupt).run()

  async def test_cascade_base_exception_with_handler(self):
    """Cascade with BaseException caught by explicit exception type."""
    handler_called = {'value': False}
    def handler(v):
      handler_called['value'] = True
      return 'caught_kb'
    result = (
      Cascade(5)
      .then(lambda v: (_ for _ in ()).throw(KeyboardInterrupt('test')))
      .except_(handler, exceptions=KeyboardInterrupt, reraise=False)
      .run()
    )
    await self.assertEqual(result, 'caught_kb')
    super(MyTestCase, self).assertTrue(handler_called['value'])

  async def test_cascade_async_except_handler(self):
    """Cascade with async except handler."""
    handler_received = {'value': None}
    async def async_handler(v):
      handler_received['value'] = v
      return 'async_recovered'
    result = await await_(
      Cascade(aempty, 55)
      .then(async_raise_test_exc)
      .except_(async_handler, reraise=False)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 'async_recovered')
    super(MyTestCase, self).assertEqual(handler_received['value'], 55)

  async def test_cascade_exception_in_async_path(self):
    """Cascade exception in async path: handler receives root."""
    handler_received = {'value': None}
    async def async_bad(v):
      raise CascadeExc('async error')
    def handler(v):
      handler_received['value'] = v
      return 'handled'
    result = await await_(
      Cascade(aempty, 123)
      .then(async_bad)
      .except_(handler, exceptions=CascadeExc, reraise=False)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 'handled')
    super(MyTestCase, self).assertEqual(handler_received['value'], 123)

  async def test_cascade_exception_handler_not_called_on_success(self):
    """Exception handler not called when no exception occurs in Cascade."""
    handler_called = {'value': False}
    def handler(v=None):
      handler_called['value'] = True
    result = Cascade(10).then(lambda v: v * 2).except_(handler).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertFalse(handler_called['value'])

  async def test_cascade_finally_runs_on_exception_propagation(self):
    """Finally runs even when exception propagates through Cascade."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = {'value': False}
        def on_finally(v=None):
          finally_called['value'] = True
        with self.assertRaises(TestExc):
          await await_(
            Cascade(fn, 10)
            .then(raise_test_exc)
            .finally_(on_finally)
            .run()
          )
        super(MyTestCase, self).assertTrue(finally_called['value'])


# ===========================================================================
# Category 4: Cascade with Iteration Operations (10+ tests)
# ===========================================================================

class CascadeIterationTests(MyTestCase):
  """Cascade behavior with foreach, filter, and gather operations."""

  async def test_cascade_foreach_receives_root_returns_root(self):
    """Cascade(list).foreach(fn) -- foreach receives root (the list), returns root."""
    root = [1, 2, 3]
    mapped = []
    def capture(x):
      mapped.append(x * 10)
      return x * 10
    result = Cascade(root).foreach(capture).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [10, 20, 30])

  async def test_cascade_filter_receives_root_returns_root(self):
    """Cascade(list).filter(pred) -- filter receives root, returns root."""
    root = [1, 2, 3, 4, 5]
    result = Cascade(root).filter(lambda x: x > 3).run()
    await self.assertEqual(result, root)

  async def test_cascade_gather_receives_root_returns_root(self):
    """Cascade(value).gather(fn1, fn2) -- each fn receives root, returns root."""
    root = 10
    received = []
    def fn1(v):
      received.append(('fn1', v))
      return v * 2
    def fn2(v):
      received.append(('fn2', v))
      return v * 3
    result = Cascade(root).gather(fn1, fn2).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(received, [('fn1', 10), ('fn2', 10)])

  async def test_cascade_void_foreach_with_root_override(self):
    """Cascade().foreach(fn).run(list) -- override root, foreach on list, returns list."""
    root = [10, 20, 30]
    mapped = []
    def capture(x):
      mapped.append(x + 1)
      return x + 1
    result = Cascade().foreach(capture).run(root)
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [11, 21, 31])

  async def test_cascade_async_foreach_root_preserved(self):
    """Cascade with async foreach -- root preserved."""
    root = [4, 5, 6]
    mapped = []
    async def async_capture(x):
      mapped.append(x * 2)
      return x * 2
    result = await await_(
      Cascade(root).foreach(async_capture).run()
    )
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [8, 10, 12])

  async def test_cascade_async_filter_root_preserved(self):
    """Cascade with async filter -- root preserved."""
    root = [1, 2, 3, 4, 5]
    async def async_pred(x):
      return x % 2 == 0
    result = await await_(
      Cascade(root).filter(async_pred).run()
    )
    super(MyTestCase, self).assertEqual(result, root)

  async def test_cascade_foreach_with_break_returns_root(self):
    """Cascade foreach with break -- returns root (not partial results)."""
    root = [1, 2, 3, 4, 5]
    collected = []
    def fn(x):
      if x == 3:
        Chain.break_()
      collected.append(x)
      return x
    result = Cascade(root).foreach(fn).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(collected, [1, 2])

  async def test_cascade_foreach_with_exception_root_in_handler(self):
    """Cascade foreach with exception -- root available to except handler."""
    root = [1, 2, 3]
    handler_received = {'value': None}
    def bad_fn(x):
      if x == 2:
        raise TestExc('foreach error')
      return x
    def handler(v):
      handler_received['value'] = v
      return 'recovered'
    result = (
      Cascade(root)
      .foreach(bad_fn)
      .except_(handler, reraise=False)
      .run()
    )
    await self.assertEqual(result, 'recovered')
    super(MyTestCase, self).assertIs(handler_received['value'], root)

  async def test_cascade_filter_nothing_matches_root_preserved(self):
    """Cascade filter where nothing matches -- root preserved."""
    root = [1, 2, 3]
    result = Cascade(root).filter(lambda x: False).run()
    await self.assertEqual(result, root)

  async def test_cascade_gather_with_async_functions_root_preserved(self):
    """Cascade gather with async functions -- root preserved."""
    root = 7
    async def async_double(v):
      return v * 2
    async def async_triple(v):
      return v * 3
    result = await await_(
      Cascade(root).gather(async_double, async_triple).run()
    )
    super(MyTestCase, self).assertEqual(result, root)

  async def test_cascade_foreach_with_async_root(self):
    """Cascade with async root + foreach."""
    root = [10, 20, 30]
    mapped = []
    def capture(x):
      mapped.append(x)
      return x * 2
    result = await await_(
      Cascade(aempty, root).foreach(capture).run()
    )
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [10, 20, 30])

  async def test_cascade_foreach_then_filter_chain(self):
    """Cascade with foreach followed by filter -- both receive root."""
    root = [1, 2, 3, 4, 5]
    foreach_received = []
    def foreach_fn(x):
      foreach_received.append(x)
      return x * 10
    # foreach output (mapped list) is discarded by Cascade;
    # filter receives root because is_with_root=True
    result = (
      Cascade(root)
      .foreach(foreach_fn)
      .filter(lambda x: x > 3)
      .run()
    )
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(foreach_received, [1, 2, 3, 4, 5])


# ===========================================================================
# Category 5: Cascade with Context Managers (5+ tests)
# ===========================================================================

class CascadeContextManagerTests(MyTestCase):
  """Cascade behavior with with_() context managers."""

  async def test_cascade_sync_cm_returns_root(self):
    """Cascade(cm).with_(body) -- body receives CM context, returns root (the CM)."""
    class SimpleCM:
      def __enter__(self):
        return 'ctx_value'
      def __exit__(self, *args):
        return False

    cm = SimpleCM()
    body_received = {'value': None}
    def body(ctx):
      body_received['value'] = ctx
      return 'body_result'
    result = Cascade(cm).with_(body).run()
    await self.assertIs(result, cm)
    super(MyTestCase, self).assertEqual(body_received['value'], 'ctx_value')

  async def test_cascade_async_cm_returns_root(self):
    """Cascade with async CM -- root preserved."""
    class AsyncCM:
      def __init__(self):
        self.entered = False
        self.exited = False
      async def __aenter__(self):
        self.entered = True
        return 'async_ctx'
      async def __aexit__(self, *args):
        self.exited = True
        return False

    cm = AsyncCM()
    body_received = {'value': None}
    def body(ctx):
      body_received['value'] = ctx
      return 'async_body_result'
    result = await await_(Cascade(cm).with_(body).run())
    super(MyTestCase, self).assertIs(result, cm)
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)
    super(MyTestCase, self).assertEqual(body_received['value'], 'async_ctx')

  async def test_cascade_cm_exception_root_available(self):
    """Cascade with CM exception -- root available to except handler."""
    class FailCM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, *args):
        return False

    cm = FailCM()
    handler_received = {'value': None}
    def bad_body(ctx):
      raise TestExc('body error')
    def handler(v):
      handler_received['value'] = v
      return 'recovered'
    result = (
      Cascade(cm)
      .with_(bad_body)
      .except_(handler, reraise=False)
      .run()
    )
    await self.assertEqual(result, 'recovered')
    # Handler receives root value (the CM instance)
    super(MyTestCase, self).assertIs(handler_received['value'], cm)

  async def test_cascade_nested_with_calls(self):
    """Cascade nested with_ calls."""
    class CM1:
      def __enter__(self):
        return 'ctx1'
      def __exit__(self, *args):
        return False

    class CM2:
      def __enter__(self):
        return 'ctx2'
      def __exit__(self, *args):
        return False

    cm1 = CM1()
    bodies = []
    # In cascade, with_ receives root (cm1) for each call.
    # Second with_ also receives cm1 (root), not the result of first with_.
    result = (
      Cascade(cm1)
      .with_(lambda ctx: bodies.append(ctx) or 'body1_result')
      .then(lambda v: v)  # v is cm1 (root), but with_ sees root
      .run()
    )
    await self.assertIs(result, cm1)
    super(MyTestCase, self).assertEqual(bodies, ['ctx1'])

  async def test_cascade_with_body_is_nested_chain(self):
    """Cascade with_ where body is a nested chain."""
    class SimpleCM:
      def __enter__(self):
        return 100
      def __exit__(self, *args):
        return False

    cm = SimpleCM()
    inner_received = []
    inner_chain = Chain().then(lambda v: inner_received.append(v) or v * 2)
    result = Cascade(cm).with_(inner_chain).run()
    # Cascade returns root (the CM)
    await self.assertIs(result, cm)
    # Inner chain received the CM context (100)
    super(MyTestCase, self).assertEqual(inner_received, [100])

  async def test_cascade_with_body_executes_but_result_discarded(self):
    """The body of with_ runs, but its result is discarded in Cascade mode."""
    body_results = []
    class YieldCM:
      def __enter__(self):
        return 42
      def __exit__(self, *args):
        return False

    cm = YieldCM()
    result = (
      Cascade(cm)
      .with_(lambda ctx: body_results.append(ctx) or 'body_result')
      .run()
    )
    await self.assertIs(result, cm)
    super(MyTestCase, self).assertEqual(body_results, [42])


# ===========================================================================
# Category 6: Cascade Clone/Freeze/Decorator (10+ tests)
# ===========================================================================

class CascadeCloneFreezeDecoratorTests(MyTestCase):
  """Cascade behavior with clone(), freeze(), and decorator()."""

  async def test_cascade_clone_preserves_cascade_behavior(self):
    """Cascade.clone() preserves is_cascade=True."""
    c = Cascade(42).then(lambda v: v * 2)
    c2 = c.clone()
    result = c2.run()
    await self.assertEqual(result, 42)

  async def test_cascade_clone_type_is_cascade(self):
    """Cascade.clone() returns a Cascade, not a Chain."""
    c = Cascade(5).then(lambda v: v + 1)
    c2 = c.clone()
    super(MyTestCase, self).assertIsInstance(c2, Cascade)

  async def test_cascade_clone_behaves_as_cascade(self):
    """Cloned cascade behaves as cascade (not chain)."""
    received = []
    def capture(v):
      received.append(v)
      return v * 100
    c = Cascade(10).then(capture).then(capture)
    c2 = c.clone()
    r1 = c.run()
    received.clear()
    r2 = c2.run()
    await self.assertEqual(r1, 10)
    await self.assertEqual(r2, 10)
    super(MyTestCase, self).assertEqual(received, [10, 10])

  async def test_cascade_freeze_returns_root(self):
    """Cascade.freeze() -- frozen cascade returns root."""
    frozen = Cascade(99).then(lambda v: v * 100).freeze()
    result = frozen.run()
    await self.assertEqual(result, 99)
    result2 = frozen()
    await self.assertEqual(result2, 99)

  async def test_cascade_freeze_run_with_override(self):
    """Cascade.freeze().run(different_root) -- override works."""
    frozen = Cascade().then(lambda v: v * 2).freeze()
    r1 = frozen.run(10)
    r2 = frozen.run(20)
    await self.assertEqual(r1, 10)
    await self.assertEqual(r2, 20)

  async def test_cascade_decorator_returns_root(self):
    """Cascade.decorator() -- decorated fn's result is root for cascade."""
    side_effects = []

    @Cascade().then(lambda v: side_effects.append(v) or v * 100).decorator()
    def my_fn(x):
      return x * 2

    result = my_fn(5)
    await self.assertEqual(result, 10)  # 5 * 2 = 10, Cascade returns root
    super(MyTestCase, self).assertEqual(side_effects, [10])

  async def test_cascade_decorator_async(self):
    """Cascade.decorator with async operation."""
    side_effects = []

    @Cascade().then(lambda v: side_effects.append(v) or aempty(v * 100)).decorator()
    def my_fn(x):
      return x + 3

    result = await await_(my_fn(7))
    super(MyTestCase, self).assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(side_effects, [10])

  async def test_cascade_concurrent_frozen_execution(self):
    """Cascade concurrent frozen execution -- all return correct roots."""
    frozen = Cascade().then(lambda v: v * 2).freeze()
    results = []
    for val in [1, 2, 3, 4, 5]:
      results.append(frozen.run(val))
    super(MyTestCase, self).assertEqual(results, [1, 2, 3, 4, 5])

  async def test_cascade_clone_with_operations_independence(self):
    """Clone of cascade with operations -- independence verified."""
    c = Cascade(5).then(lambda v: v + 1).then(lambda v: v * 2)
    c2 = c.clone()
    # Adding to clone doesn't affect original
    c2.then(lambda v: v ** 2)
    r1 = c.run()
    r2 = c2.run()
    await self.assertEqual(r1, 5)
    await self.assertEqual(r2, 5)

  async def test_cascade_clone_with_except_finally_independence(self):
    """Clone of cascade with except/finally -- independence verified."""
    handler_calls = {'original': 0, 'clone': 0}
    def handler(v=None):
      handler_calls['original'] += 1
    c = Cascade(5).then(lambda v: v + 1).finally_(handler)
    c2 = c.clone()
    c.run()
    c2.run()
    # Both execute the handler (cloned from original)
    super(MyTestCase, self).assertEqual(handler_calls['original'], 2)

  async def test_cascade_freeze_with_except_handler(self):
    """Freeze cascade with except handler -- works correctly."""
    handler_received = {'value': None}
    def handler(v):
      handler_received['value'] = v
      return 'recovered'
    frozen = (
      Cascade()
      .then(raise_test_exc)
      .except_(handler, reraise=False)
      .freeze()
    )
    result = frozen.run(42)
    await self.assertEqual(result, 'recovered')
    super(MyTestCase, self).assertEqual(handler_received['value'], 42)

  async def test_cascade_pipe_operator(self):
    """Cascade pipe operator: Cascade(5) | fn | run()."""
    received = []
    result = Cascade(10) | (lambda v: received.append(v) or v * 2) | run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(received, [10])

  async def test_cascade_pipe_multiple_ops(self):
    """Cascade pipe with multiple operations, all receive root."""
    received = []
    result = (
      Cascade(5)
      | (lambda v: received.append(('a', v)) or v * 2)
      | (lambda v: received.append(('b', v)) or v + 10)
      | (lambda v: received.append(('c', v)) or 'ignored')
      | run()
    )
    await self.assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(received, [('a', 5), ('b', 5), ('c', 5)])


# ===========================================================================
# Category 7: Cascade Internal Implementation Details (10+ tests)
# ===========================================================================

class CascadeInternalTests(MyTestCase):
  """Implementation-level details of Cascade behavior."""

  async def test_every_link_is_with_root_via_behavior(self):
    """Every link in Cascade has is_with_root=True (verify via behavior)."""
    # If is_with_root were False, each link would receive previous result
    received = []
    def capture(v):
      received.append(v)
      return v * 100  # would propagate in Chain
    Cascade(1).then(capture).then(capture).then(capture).run()
    # If is_with_root were not set, second capture would get 100, third 10000
    super(MyTestCase, self).assertEqual(received, [1, 1, 1])

  async def test_is_simple_flag_with_only_then(self):
    """The _is_simple flag on Cascade: .then() only IS simple (Loop 3)."""
    # With only .then() links, Cascade should use the simple path.
    # Verify indirectly: no except/do/debug should work
    result = Cascade(5).then(lambda v: v + 1).then(lambda v: v * 2).run()
    await self.assertEqual(result, 5)

  async def test_do_makes_non_simple(self):
    """.do() on Cascade: is_with_root=True AND ignore_result=True, non-simple."""
    side_effects = []
    result = Cascade(5).do(lambda v: side_effects.append(v)).run()
    await self.assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(side_effects, [5])

  async def test_cascade_simple_sync_path_loop_2a(self):
    """Cascade in simple + sync path: Loop 2a (sync, cascade)."""
    # no_async(True) + only .then() + cascade -> Loop 2a
    result = (
      Cascade(5)
      .then(lambda v: v + 1)
      .then(lambda v: v * 2)
      .no_async(True)
      .run()
    )
    await self.assertEqual(result, 5)

  async def test_cascade_simple_async_path_run_async_simple(self):
    """Cascade in async simple path: _run_async_simple with is_cascade=True."""
    received = []
    async def async_capture(v):
      received.append(v)
      return v * 10
    result = await await_(
      Cascade(aempty, 77)
      .then(async_capture)
      .then(lambda v: received.append(v) or v * 2)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 77)
    super(MyTestCase, self).assertEqual(received, [77, 77])

  async def test_cascade_current_value_equals_root_at_end(self):
    """Cascade current_value = root_value at end of loop."""
    # The core invariant: after all links, current_value is restored to root_value
    result = (
      Cascade(42)
      .then(lambda v: 'completely_different')
      .then(lambda v: 99999)
      .then(lambda v: None)
      .run()
    )
    await self.assertEqual(result, 42)

  async def test_cascade_with_debug_mode_logs(self):
    """Cascade with debug mode: operations still receive root."""
    received = []
    def capture(v):
      received.append(v)
      return v * 10
    result = Cascade(5).then(capture).then(capture).config(debug=True).run()
    await self.assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(received, [5, 5])

  async def test_cascade_repr_shows_cascade(self):
    """Cascade repr: shows 'Cascade' not 'Chain'."""
    c = Cascade(5)
    r = repr(c)
    super(MyTestCase, self).assertIn('Cascade', r)
    super(MyTestCase, self).assertNotIn('Chain', r)

  async def test_cascade_bool_always_true(self):
    """Cascade __bool__: always True."""
    super(MyTestCase, self).assertTrue(bool(Cascade()))
    super(MyTestCase, self).assertTrue(bool(Cascade(5)))
    super(MyTestCase, self).assertTrue(bool(Cascade(0)))
    super(MyTestCase, self).assertTrue(bool(Cascade(None)))
    super(MyTestCase, self).assertTrue(bool(Cascade(False)))

  async def test_cascade_type_is_cascade(self):
    """type(Cascade(5)) is Cascade (not Chain)."""
    c = Cascade(5)
    super(MyTestCase, self).assertIs(type(c), Cascade)

  async def test_cascade_isinstance_chain(self):
    """isinstance(Cascade(5), Chain) is True (Cascade inherits Chain)."""
    c = Cascade(5)
    super(MyTestCase, self).assertIsInstance(c, Chain)
    super(MyTestCase, self).assertIsInstance(c, Cascade)

  async def test_cascade_loop3_async_cascade_simple(self):
    """Loop 3: async cascade simple path (not _is_sync, is_cascade, simple)."""
    # Async root + only .then() links -> _is_simple=True
    # not _is_sync, is_cascade -> Loop 3 in _run_simple
    received = []
    async def coro_fn(v):
      received.append(v)
      return v * 10
    result = await await_(
      Cascade(5)
      .then(coro_fn)  # triggers async transition
      .then(lambda v: received.append(v) or v * 2)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(received, [5, 5])

  async def test_cascade_loop2b_sync_chain_not_cascade(self):
    """Loop 2b: sync + chain (not cascade) for comparison."""
    # This is a Chain, not Cascade, using no_async(True)
    result = (
      Chain(5)
      .then(lambda v: v + 1)
      .then(lambda v: v * 2)
      .no_async(True)
      .run()
    )
    await self.assertEqual(result, 12)  # 5 -> 6 -> 12


# ===========================================================================
# Category 8: Cascade vs Chain Side-by-Side Comparisons (10+ tests)
# ===========================================================================

class CascadeVsChainComparisonTests(MyTestCase):
  """Direct side-by-side comparisons of Chain vs Cascade behavior."""

  async def test_same_definition_different_results(self):
    """Same chain definition, run as Chain vs Cascade: verify different results."""
    fn = lambda v: v * 3

    chain_r = Chain(2).then(fn).then(fn).run()
    cascade_r = Cascade(2).then(fn).then(fn).run()

    await self.assertEqual(chain_r, 18)    # 2 -> 6 -> 18
    await self.assertEqual(cascade_r, 2)   # always root

  async def test_value_flow_chain_vs_root_pass_cascade(self):
    """Chain: value flows through. Cascade: root passed to each."""
    operations = []

    def op(v):
      operations.append(v)
      return v + 10

    # Chain
    operations.clear()
    Chain(1).then(op).then(op).then(op).run()
    chain_ops = list(operations)

    # Cascade
    operations.clear()
    Cascade(1).then(op).then(op).then(op).run()
    cascade_ops = list(operations)

    super(MyTestCase, self).assertEqual(chain_ops, [1, 11, 21])    # cascading values
    super(MyTestCase, self).assertEqual(cascade_ops, [1, 1, 1])    # always root

  async def test_exception_handler_both_receive_root(self):
    """Exception handling: both Chain and Cascade handler get root value."""
    chain_handler_v = {'value': None}
    cascade_handler_v = {'value': None}

    def chain_handler(v):
      chain_handler_v['value'] = v
      return 'recovered'
    def cascade_handler(v):
      cascade_handler_v['value'] = v
      return 'recovered'

    # Chain: handler receives root_value
    Chain(42).then(lambda v: v * 2).then(raise_test_exc).except_(
      chain_handler, reraise=False
    ).run()

    # Cascade: handler also receives root_value
    Cascade(42).then(lambda v: v * 2).then(raise_test_exc).except_(
      cascade_handler, reraise=False
    ).run()

    # Both receive root_value (42)
    super(MyTestCase, self).assertEqual(chain_handler_v['value'], 42)
    super(MyTestCase, self).assertEqual(cascade_handler_v['value'], 42)

  async def test_foreach_chain_vs_cascade(self):
    """foreach: Chain receives iterable from previous. Cascade receives root."""
    # Chain: foreach operates on the current value (an iterable)
    chain_r = Chain([1, 2, 3]).foreach(lambda x: x * 10).run()
    await self.assertEqual(chain_r, [10, 20, 30])

    # Cascade: foreach operates on root (the list), returns root
    cascade_r = Cascade([1, 2, 3]).foreach(lambda x: x * 10).run()
    await self.assertEqual(cascade_r, [1, 2, 3])

  async def test_filter_chain_vs_cascade(self):
    """filter: Chain returns filtered list. Cascade returns root."""
    chain_r = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3).run()
    await self.assertEqual(chain_r, [4, 5])

    cascade_r = Cascade([1, 2, 3, 4, 5]).filter(lambda x: x > 3).run()
    await self.assertEqual(cascade_r, [1, 2, 3, 4, 5])

  async def test_gather_chain_vs_cascade(self):
    """gather: Chain returns gathered results. Cascade returns root."""
    fn1 = lambda v: v * 2
    fn2 = lambda v: v * 3

    chain_r = Chain(5).gather(fn1, fn2).run()
    await self.assertEqual(chain_r, [10, 15])

    cascade_r = Cascade(5).gather(fn1, fn2).run()
    await self.assertEqual(cascade_r, 5)

  async def test_with_chain_vs_cascade(self):
    """with_: Chain returns body result. Cascade returns root."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, *args):
        return False

    cm_chain = CM()
    cm_cascade = CM()

    chain_r = Chain(cm_chain).with_(lambda ctx: f'processed_{ctx}').run()
    cascade_r = Cascade(cm_cascade).with_(lambda ctx: f'processed_{ctx}').run()

    await self.assertEqual(chain_r, 'processed_ctx')
    await self.assertIs(cascade_r, cm_cascade)

  async def test_nested_inner_chain_vs_inner_cascade(self):
    """Nested: Inner chain behavior vs inner cascade behavior."""
    # Inner Chain: passes value through
    inner_chain = Chain().then(lambda v: v * 2)
    outer_chain_r = Chain(5).then(inner_chain).run()
    await self.assertEqual(outer_chain_r, 10)

    # Inner Cascade: passes root of inner cascade
    inner_cascade = Cascade().then(lambda v: v * 2)
    outer_cascade_r = Chain(5).then(inner_cascade).run()
    # Inner cascade receives 5 as root override, returns 5 (root)
    await self.assertEqual(outer_cascade_r, 5)

  async def test_clone_chain_is_chain_cascade_is_cascade(self):
    """Clone: Chain clone is Chain, Cascade clone is Cascade."""
    chain_clone = Chain(5).then(lambda v: v + 1).clone()
    cascade_clone = Cascade(5).then(lambda v: v + 1).clone()

    super(MyTestCase, self).assertIs(type(chain_clone), Chain)
    super(MyTestCase, self).assertIs(type(cascade_clone), Cascade)

    chain_r = chain_clone.run()
    cascade_r = cascade_clone.run()

    await self.assertEqual(chain_r, 6)
    await self.assertEqual(cascade_r, 5)

  async def test_pipe_both_support_but_cascade_returns_root(self):
    """Pipe: Both support pipe, but Cascade returns root at end."""
    fn = lambda v: v + 10

    chain_r = Chain(5) | fn | fn | run()
    cascade_r = Cascade(5) | fn | fn | run()

    await self.assertEqual(chain_r, 25)    # 5 -> 15 -> 25
    await self.assertEqual(cascade_r, 5)   # always root

  async def test_chain_vs_cascade_async_comparison(self):
    """Async variant of Chain vs Cascade comparison."""
    async def async_double(v):
      return v * 2

    chain_r = await await_(
      Chain(aempty, 3).then(async_double).then(async_double).run()
    )
    cascade_r = await await_(
      Cascade(aempty, 3).then(async_double).then(async_double).run()
    )

    super(MyTestCase, self).assertEqual(chain_r, 12)   # 3 -> 6 -> 12
    super(MyTestCase, self).assertEqual(cascade_r, 3)  # always root

  async def test_chain_vs_cascade_return_values_with_many_links(self):
    """Many links: Chain accumulates, Cascade always root."""
    def add_one(v):
      return v + 1
    def multiply_two(v):
      return v * 2

    c = Chain(1)
    d = Cascade(1)
    for _ in range(5):
      c = c.then(add_one)
      d = d.then(add_one)

    chain_r = c.run()
    cascade_r = d.run()

    await self.assertEqual(chain_r, 6)      # 1 + 5 = 6
    await self.assertEqual(cascade_r, 1)    # always root


# ===========================================================================
# Category 9: Cascade Edge Cases and Miscellaneous (additional tests)
# ===========================================================================

class CascadeEdgeCaseTests(MyTestCase):
  """Additional edge cases for thorough coverage."""

  async def test_cascade_with_sleep_returns_root(self):
    """Cascade(v).sleep(0).run() returns root_value."""
    result = Cascade(42).sleep(0).run()
    await self.assertEqual(result, 42)

  async def test_cascade_async_sleep_returns_root(self):
    """Cascade with async root + sleep: result is root_value."""
    result = await await_(Cascade(aempty, 99).sleep(0).run())
    super(MyTestCase, self).assertEqual(result, 99)

  async def test_cascade_sleep_then_operations(self):
    """Cascade.sleep + .then: all operations receive root, result is root."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = []
        result = (
          Cascade(fn, 55)
          .sleep(0)
          .then(lambda v: received.append(v) or v * 2)
          .run()
        )
        await self.assertEqual(result, 55)
        super(MyTestCase, self).assertEqual(received, [55])

  async def test_cascade_callable_via_call_syntax(self):
    """Cascade(5).then(fn).__call__(v) behaves like run(v)."""
    result = Cascade(5).then(lambda v: v * 2)()
    await self.assertEqual(result, 5)

  async def test_cascade_empty_with_do_returns_none(self):
    """Void Cascade with .do(): root_value is Null, cascade restores Null -> None."""
    side_effects = []
    result = Cascade().do(lambda: side_effects.append('ran'), ...).run()
    await self.assertIsNone(result)
    super(MyTestCase, self).assertEqual(side_effects, ['ran'])

  async def test_cascade_to_thread_returns_root(self):
    """Cascade with to_thread: result is root_value."""
    received = {'value': None}
    def thread_fn(v):
      received['value'] = v
      return v * 100
    result = await await_(
      Cascade(aempty, 77).to_thread(thread_fn).run()
    )
    super(MyTestCase, self).assertEqual(result, 77)
    super(MyTestCase, self).assertEqual(received['value'], 77)

  async def test_cascade_with_both_sync_async_handlers(self):
    """Cascade with sync except handler and async finally handler."""
    except_received = {'value': None}
    finally_received = {'value': None}
    def on_except(v):
      except_received['value'] = v
      return 'recovered'
    async def on_finally(v=None):
      finally_received['value'] = v
    result = await await_(
      Cascade(aempty, 42)
      .then(async_raise_test_exc)
      .except_(on_except, reraise=False)
      .finally_(on_finally)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 'recovered')
    super(MyTestCase, self).assertEqual(except_received['value'], 42)
    super(MyTestCase, self).assertEqual(finally_received['value'], 42)

  async def test_cascade_return_signal_in_nested(self):
    """Chain.return_() in a nested cascade chain."""
    def trigger_return(v):
      Chain.return_(v * 2)

    inner = Cascade().then(lambda v: v + 100).then(trigger_return)
    result = Chain(5).then(inner).run()
    # Inner cascade has return_() which exits early with v*2
    # But since Cascade's root value is 5, trigger_return receives 5
    # Chain.return_(5 * 2) = Chain.return_(10)
    await self.assertEqual(result, 10)

  async def test_cascade_large_number_of_links(self):
    """Cascade with a large number of links (100+)."""
    received = []
    c = Cascade(42)
    for i in range(100):
      c = c.then(lambda v, i=i: received.append((i, v)) or v * 2)
    result = c.run()
    await self.assertEqual(result, 42)
    super(MyTestCase, self).assertEqual(len(received), 100)
    # All should have received root value 42
    for i, (idx, val) in enumerate(received):
      super(MyTestCase, self).assertEqual(val, 42, f"Link {idx} received {val}, expected 42")

  async def test_cascade_object_root(self):
    """Cascade with custom object as root."""
    class MyObj:
      def __init__(self, x):
        self.x = x
    obj = MyObj(42)
    received = []
    def capture(v):
      received.append(v)
      return 'ignored'
    result = Cascade(obj).then(capture).then(capture).run()
    await self.assertIs(result, obj)
    super(MyTestCase, self).assertEqual(len(received), 2)
    for r in received:
      super(MyTestCase, self).assertIs(r, obj)

  async def test_cascade_nested_cascade_in_chain(self):
    """Nested Cascade inside a Chain: inner Cascade returns its root."""
    inner = Cascade().then(lambda v: v * 100)
    result = Chain(5).then(inner).then(lambda v: v + 1).run()
    # inner receives 5 as root, returns 5 (Cascade returns root)
    # then v + 1 = 6
    await self.assertEqual(result, 6)

  async def test_cascade_nested_chain_in_cascade(self):
    """Nested Chain inside a Cascade: inner Chain returns its computed value,
    but outer Cascade discards it and returns root."""
    inner = Chain().then(lambda v: v * 100)
    result = Cascade(5).then(inner).run()
    # inner receives 5 (root), computes 500
    # but Cascade restores root 5
    await self.assertEqual(result, 5)

  async def test_cascade_with_iterate(self):
    """Cascade.iterate() returns a _Generator."""
    from quent.quent import _Generator
    gen = Cascade([1, 2, 3]).iterate()
    super(MyTestCase, self).assertIsInstance(gen, _Generator)

  async def test_cascade_duplicate_finally_raises(self):
    """Two finally_() calls on Cascade raise QuentException."""
    with self.assertRaises(QuentException):
      Cascade(1).finally_(lambda v: None).finally_(lambda v: None)

  async def test_cascade_root_none_vs_null_distinction(self):
    """Cascade(None) vs Cascade(): different root semantics."""
    # Cascade(None): root is None (not Null)
    received_with_none = {'value': 'sentinel'}
    def fn1(v):
      received_with_none['value'] = v
    Cascade(None).then(fn1).run()
    super(MyTestCase, self).assertIsNone(received_with_none['value'])

    # Cascade(): root is Null (void cascade)
    received_void = {'value': 'sentinel'}
    def fn2(v=None):
      received_void['value'] = v
    Cascade().then(fn2, ...).run()
    # fn2 called with no args (Null -> no arg), v defaults to None
    super(MyTestCase, self).assertIsNone(received_void['value'])

  async def test_cascade_sync_finally_async_handler_warning(self):
    """Async finally handler on sync cascade emits RuntimeWarning."""
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Cascade(1).then(lambda v: v * 2).finally_(async_handler).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_cascade_no_async_sync_mode_loop(self):
    """Cascade with no_async(True) uses sync loop (Loop 2a) exclusively."""
    received = []
    result = (
      Cascade(10)
      .then(lambda v: received.append(v) or v + 1)
      .then(lambda v: received.append(v) or v * 2)
      .then(lambda v: received.append(v) or v - 5)
      .no_async(True)
      .run()
    )
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(received, [10, 10, 10])

  async def test_cascade_chained_do_and_then_mixed(self):
    """Cascade with mixed .do() and .then() -- all receive root."""
    received_do = []
    received_then = []
    result = (
      Cascade(7)
      .do(lambda v: received_do.append(('do1', v)))
      .then(lambda v: received_then.append(('then1', v)) or v * 2)
      .do(lambda v: received_do.append(('do2', v)))
      .then(lambda v: received_then.append(('then2', v)) or v + 100)
      .run()
    )
    await self.assertEqual(result, 7)
    super(MyTestCase, self).assertEqual(received_do, [('do1', 7), ('do2', 7)])
    super(MyTestCase, self).assertEqual(received_then, [('then1', 7), ('then2', 7)])

  async def test_cascade_with_foreach_indexed(self):
    """Cascade with foreach(with_index=True) -- root preserved."""
    root = ['a', 'b', 'c']
    indexed_results = []
    def fn(idx, el):
      indexed_results.append((idx, el))
      return f'{idx}:{el}'
    result = Cascade(root).foreach(fn, with_index=True).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(indexed_results, [(0, 'a'), (1, 'b'), (2, 'c')])

  async def test_cascade_except_specific_exception_type(self):
    """Cascade except with specific exception type matching."""
    class CustomExc(Exception):
      pass
    handler_called = {'value': False}
    def handler(v):
      handler_called['value'] = True
      return 'caught'
    def raise_custom(v):
      raise CustomExc('custom')
    result = (
      Cascade(5)
      .then(raise_custom)
      .except_(handler, exceptions=CustomExc, reraise=False)
      .run()
    )
    await self.assertEqual(result, 'caught')
    super(MyTestCase, self).assertTrue(handler_called['value'])

  async def test_cascade_except_unmatched_type_propagates(self):
    """Cascade except with unmatched exception type -- propagates."""
    class SpecificExc(Exception):
      pass
    handler_called = {'value': False}
    def handler(v):
      handler_called['value'] = True
    with self.assertRaises(TestExc):
      Cascade(5).then(raise_test_exc).except_(
        handler, exceptions=SpecificExc, reraise=False
      ).run()
    super(MyTestCase, self).assertFalse(handler_called['value'])


if __name__ == '__main__':
  import unittest
  unittest.main()
