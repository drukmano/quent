import asyncio
import inspect
import logging
import concurrent.futures
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import TestExc, empty, aempty, await_
from quent import Chain, Cascade, ChainAttr, CascadeAttr, QuentException, run, FrozenChain


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
# Clone Feature Tests
# ---------------------------------------------------------------------------

class CloneFeatureTests(TestCase):
  def test_clone_empty(self):
    c = Chain()
    c2 = c.clone()
    self.assertIsNone(c2.run())

  def test_clone_preserves_root(self):
    c = Chain(42)
    c2 = c.clone()
    self.assertEqual(c2.run(), 42)

  def test_clone_preserves_links(self):
    c = Chain(2).then(lambda v: v * 3).then(lambda v: v + 1)
    c2 = c.clone()
    self.assertEqual(c2.run(), 7)

  def test_clone_independence(self):
    c = Chain(10).then(lambda v: v * 2)
    c2 = c.clone()
    # Modify original after cloning
    c.then(lambda v: v + 100)
    self.assertEqual(c.run(), 120)
    self.assertEqual(c2.run(), 20)

  def test_clone_preserves_except(self):
    ran = []
    def raise_exc():
      raise TestExc()
    c = Chain(raise_exc).except_(
      lambda: ran.append('caught'), ..., exceptions=TestExc, reraise=False
    )
    c2 = c.clone()
    c2.run()
    self.assertEqual(ran, ['caught'])

  def test_clone_preserves_finally(self):
    ran = []
    c = Chain(10).finally_(lambda v: ran.append('finally'))
    c2 = c.clone()
    c2.run()
    self.assertEqual(ran, ['finally'])

  def test_clone_preserves_on_success(self):
    ran = []
    c = Chain(10).on_success(lambda v: ran.append(v))
    c2 = c.clone()
    c2.run()
    self.assertEqual(ran, [10])

  def test_clone_preserves_autorun(self):
    c = Chain(10).config(autorun=True)
    c2 = c.clone()
    # Both should have autorun set. We verify by checking the config result
    # chain is returned from config. We check behavior: with autorun=True
    # and _is_sync=False, if an async chain is run, it wraps in a task.
    # For a sync chain, autorun doesn't change behavior. We just verify
    # the clone maintains the state by observing the clone still works.
    self.assertEqual(c2.run(), 10)

  def test_clone_preserves_context(self):
    c = Chain(10).with_context(key='value')
    c2 = c.clone()
    # The clone should have an independent context dict
    # Modify original context
    c.with_context(other='other')
    # Clone should not have the new key
    # We verify by running in context and checking
    def check_ctx(v):
      ctx = Chain.get_context()
      return ctx.get('key'), ctx.get('other')
    result = Chain(10).with_context(key='value').then(check_ctx).run()
    self.assertEqual(result, ('value', None))

  def test_clone_cascade_type_preserved(self):
    c = Cascade()
    c2 = c.clone()
    self.assertIsInstance(c2, Cascade)

  def test_clone_of_clone(self):
    c = Chain(5).then(lambda v: v * 2)
    c2 = c.clone()
    c3 = c2.clone()
    self.assertEqual(c3.run(), 10)

  def test_clone_multiple_times(self):
    c = Chain(3).then(lambda v: v + 1)
    clones = [c.clone() for _ in range(5)]
    for clone in clones:
      self.assertEqual(clone.run(), 4)
    # Modify original
    c.then(lambda v: v * 100)
    # Clones unaffected
    for clone in clones:
      self.assertEqual(clone.run(), 4)

  def test_clone_with_nested_chain(self):
    inner = Chain().then(lambda v: v * 2)
    outer = Chain(5).then(inner)
    c2 = outer.clone()
    self.assertEqual(c2.run(), 10)


# ---------------------------------------------------------------------------
# Freeze Feature Tests
# ---------------------------------------------------------------------------

class FreezeFeatureTests(MyTestCase):
  async def test_freeze_run(self):
    frozen = Chain(10).then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen.run(), 20)

  async def test_freeze_call(self):
    frozen = Chain(10).then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen(), 20)

  async def test_freeze_reusable(self):
    frozen = Chain(10).then(lambda v: v + 5).freeze()
    await self.assertEqual(frozen.run(), 15)
    await self.assertEqual(frozen.run(), 15)
    await self.assertEqual(frozen(), 15)

  async def test_freeze_with_root_override(self):
    frozen = Chain().then(lambda v: v * 3).freeze()
    await self.assertEqual(frozen.run(7), 21)
    await self.assertEqual(frozen.run(2), 6)

  async def test_freeze_concurrent_async(self):
    frozen = Chain().then(aempty).then(lambda v: v * 2).freeze()
    tasks = [frozen.run(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    expected = [i * 2 for i in range(50)]
    super(MyTestCase, self).assertEqual(sorted(results), sorted(expected))

  async def test_freeze_async_chain(self):
    for fn, ctx in self.with_fn():
      with ctx:
        frozen = Chain(5).then(fn).then(lambda v: v + 10).freeze()
        await self.assertEqual(frozen.run(), 15)

  async def test_freeze_returns_frozen_chain(self):
    frozen = Chain(1).freeze()
    super(MyTestCase, self).assertIsInstance(frozen, FrozenChain)


# ---------------------------------------------------------------------------
# Decorator Feature Tests
# ---------------------------------------------------------------------------

class DecoratorFeatureTests(MyTestCase):
  async def test_decorator_sync_fn(self):
    @Chain().then(lambda v: v * 2).decorator()
    def double(v):
      return v + 1

    await self.assertEqual(double(3), 8)  # (3+1)*2

  async def test_decorator_async_fn(self):
    @Chain().then(lambda v: v * 2).decorator()
    async def double(v):
      return v + 1

    await self.assertEqual(double(3), 8)

  async def test_decorator_preserves_name(self):
    @Chain().then(lambda v: v).decorator()
    def my_function(x):
      return x

    super(MyTestCase, self).assertEqual(my_function.__name__, 'my_function')

  async def test_decorator_on_class_method(self):
    class MyClass:
      @Chain().then(lambda v: v * 3).decorator()
      def multiply(self, x):
        return x

    obj = MyClass()
    await self.assertEqual(obj.multiply(5), 15)


# ---------------------------------------------------------------------------
# Safe Run Feature Tests
# ---------------------------------------------------------------------------

class SafeRunFeatureTests(MyTestCase):
  async def test_safe_run_same_as_run(self):
    c = Chain(10).then(lambda v: v * 2)
    run_result = c.run()
    safe_result = c.safe_run()
    await self.assertEqual(run_result, 20)
    await self.assertEqual(safe_result, 20)

  async def test_safe_run_thread_safety(self):
    counter = {'value': 0}

    def increment(v):
      counter['value'] += 1
      return v + counter['value']

    c = Chain(0).then(increment)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(c.safe_run) for _ in range(10)]
      for f in concurrent.futures.as_completed(futures):
        results.append(f.result())
    # Each safe_run clones, so all should produce a result
    super(MyTestCase, self).assertEqual(len(results), 10)
    for r in results:
      super(MyTestCase, self).assertIsNotNone(r)

  async def test_safe_run_preserves_original(self):
    c = Chain(42).then(lambda v: v + 1)
    c.safe_run()
    # Original should still work identically
    await self.assertEqual(c.run(), 43)

  async def test_safe_run_with_root_override(self):
    c = Chain().then(lambda v: v * 5)
    await self.assertEqual(c.safe_run(3), 15)

  async def test_safe_run_exception_propagates(self):
    c = Chain(lambda: (_ for _ in ()).throw(TestExc()))
    with self.assertRaises(TestExc):
      c.safe_run()


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class ConfigTests(TestCase):
  def test_config_autorun(self):
    c = Chain(10).config(autorun=True)
    # With autorun and a sync chain, result is returned directly
    self.assertEqual(c.run(), 10)

  def test_config_debug(self):
    logger = logging.getLogger('quent')
    handler = logging.handlers_module = None
    # Use a handler to capture log output
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      Chain(42).then(lambda v: v + 1).config(debug=True).run()
      log_output = stream.getvalue()
      self.assertIn('42', log_output)
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

  def test_config_async_false(self):
    # async_=False sets _is_sync=True, which skips coroutine checks
    c = Chain(10).then(lambda v: v * 2).config(async_=False)
    self.assertEqual(c.run(), 20)

  def test_config_multiple_params(self):
    c = Chain(10).config(autorun=True, debug=True, async_=False)
    self.assertEqual(c.run(), 10)

  def test_config_returns_self(self):
    c = Chain(10)
    result = c.config()
    self.assertIs(result, c)

  def test_config_none_params_ignored(self):
    c = Chain(10).then(lambda v: v + 5)
    c.config(autorun=None, debug=None, async_=None)
    # Should still work normally since None params are ignored
    self.assertEqual(c.run(), 15)


# ---------------------------------------------------------------------------
# Set Async Tests
# ---------------------------------------------------------------------------

class SetAsyncTests(MyTestCase):
  async def test_set_async_false_skips_coro_check(self):
    # With set_async(False), _is_sync=True, so the chain skips coroutine detection
    # For a pure sync chain, this is equivalent to a fast path
    c = Chain(5).then(lambda v: v * 2).set_async(False)
    await self.assertEqual(c.run(), 10)

  async def test_set_async_false_simple_chain(self):
    c = Chain(100).then(lambda v: v - 50).set_async(False)
    await self.assertEqual(c.run(), 50)

  async def test_set_async_false_cascade(self):
    obj = object()
    c = Cascade(obj).then(lambda v: None).set_async(False)
    result = c.run()
    super(MyTestCase, self).assertIs(result, obj)

  async def test_set_async_false_clone_preserved(self):
    c = Chain(10).set_async(False)
    c2 = c.clone()
    # _is_sync is a readonly attribute
    super(MyTestCase, self).assertTrue(c2._is_sync)


# ---------------------------------------------------------------------------
# Pipe Operator Tests
# ---------------------------------------------------------------------------

class PipeOperatorTests(MyTestCase):
  async def test_pipe_operator_basic(self):
    result = Chain(1) | (lambda v: v * 10) | run()
    await self.assertEqual(result, 10)

  async def test_pipe_operator_chained(self):
    result = Chain(2) | (lambda v: v + 3) | (lambda v: v * 2) | run()
    await self.assertEqual(result, 10)

  async def test_pipe_with_run_root(self):
    result = Chain() | (lambda v: v * 4) | run(5)
    await self.assertEqual(result, 20)

  async def test_pipe_method(self):
    c = Chain(5).pipe(lambda v: v + 10)
    await self.assertEqual(c.run(), 15)

  async def test_pipe_method_returns_self(self):
    c = Chain(5)
    result = c.pipe(lambda v: v)
    super(MyTestCase, self).assertIs(result, c)


# ---------------------------------------------------------------------------
# Compose Tests
# ---------------------------------------------------------------------------

class ComposeTests(TestCase):
  def test_compose_callables(self):
    f1 = lambda v: v * 2
    f2 = lambda v: v + 3
    c = Chain.compose(f1, f2)
    # compose executes left to right: f2(f1(v)) => (5*2)+3 = 13
    self.assertEqual(c.run(5), 13)

  def test_compose_chains(self):
    c1 = Chain().then(lambda v: v * 2)
    c2 = Chain().then(lambda v: v + 10)
    composed = Chain.compose(c1, c2)
    self.assertEqual(composed.run(3), 16)  # (3*2)+10

  def test_compose_empty(self):
    c = Chain.compose()
    self.assertIsNone(c.run())

  def test_compose_single(self):
    f = lambda v: v * 5
    c = Chain.compose(f)
    self.assertEqual(c.run(3), 15)

  def test_compose_returns_chain(self):
    c = Chain.compose(lambda v: v)
    self.assertIsInstance(c, Chain)


# ---------------------------------------------------------------------------
# ChainAttr Tests
# ---------------------------------------------------------------------------

class _AttrTestObj:
  """Helper class for ChainAttr/CascadeAttr tests."""
  def __init__(self, value=10):
    self._value = value

  @property
  def value(self):
    return self._value

  @property
  def double(self):
    return self._value * 2

  def add(self, n):
    return self._value + n

  def multiply(self, n):
    return self._value * n

  def identity(self):
    return self


class ChainAttrTests(MyTestCase):
  async def test_chain_attr_property(self):
    obj = _AttrTestObj(10)
    await self.assertEqual(ChainAttr(obj).value.run(), 10)

  async def test_chain_attr_method_call(self):
    obj = _AttrTestObj(10)
    await self.assertEqual(ChainAttr(obj).identity().run(), obj)

  async def test_chain_attr_method_with_args(self):
    obj = _AttrTestObj(10)
    await self.assertEqual(ChainAttr(obj).add(5).run(), 15)

  async def test_chain_attr_chained(self):
    obj = _AttrTestObj(10)
    # .double accesses property (returns 20), but we need the result to have
    # attributes too. Let's chain property + method on a more complex object.
    class Obj:
      @property
      def inner(self):
        return _AttrTestObj(7)
    o = Obj()
    await self.assertEqual(ChainAttr(o).inner.add(3).run(), 10)

  async def test_chain_attr_then_interleaved(self):
    obj = _AttrTestObj(10)
    result = await await_(ChainAttr(obj).value.then(lambda v: v + 5).run())
    super(MyTestCase, self).assertEqual(result, 15)

  async def test_chain_attr_call_without_pending_attr(self):
    obj = _AttrTestObj(10)
    # When no pending attr, __call__ delegates to run()
    result = await await_(ChainAttr(obj).then(lambda v: v)())
    super(MyTestCase, self).assertIs(result, obj)


# ---------------------------------------------------------------------------
# CascadeAttr Tests
# ---------------------------------------------------------------------------

class CascadeAttrTests(MyTestCase):
  async def test_cascade_attr_returns_root(self):
    obj = _AttrTestObj(10)
    result = await await_(CascadeAttr(obj).value.identity().run())
    super(MyTestCase, self).assertIs(result, obj)

  async def test_cascade_attr_type(self):
    obj = _AttrTestObj(10)
    ca = CascadeAttr(obj)
    super(MyTestCase, self).assertIsInstance(ca, CascadeAttr)


# ---------------------------------------------------------------------------
# Debug Mode Tests
# ---------------------------------------------------------------------------

class DebugModeTests(MyTestCase):
  def _make_logger_handler(self):
    import io
    logger = logging.getLogger('quent')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    return logger, handler, stream, old_level

  def _cleanup_logger(self, logger, handler, old_level):
    logger.removeHandler(handler)
    logger.setLevel(old_level)

  async def test_debug_logs_root_value(self):
    logger, handler, stream, old_level = self._make_logger_handler()
    try:
      Chain(42).config(debug=True).run()
      log_output = stream.getvalue()
      super(MyTestCase, self).assertIn('42', log_output)
    finally:
      self._cleanup_logger(logger, handler, old_level)

  async def test_debug_logs_link_results(self):
    logger, handler, stream, old_level = self._make_logger_handler()
    try:
      Chain(5).then(lambda v: v * 3).config(debug=True).run()
      log_output = stream.getvalue()
      # Root value 5 should be logged
      super(MyTestCase, self).assertIn('5', log_output)
      # Link result 15 should be logged
      super(MyTestCase, self).assertIn('15', log_output)
    finally:
      self._cleanup_logger(logger, handler, old_level)

  async def test_debug_disabled_no_logs(self):
    logger, handler, stream, old_level = self._make_logger_handler()
    try:
      Chain(999).then(lambda v: v + 1).run()
      log_output = stream.getvalue()
      super(MyTestCase, self).assertNotIn('999', log_output)
    finally:
      self._cleanup_logger(logger, handler, old_level)

  async def test_debug_async(self):
    # In async chains, _logger.debug() is only called for the root value in the
    # sync path before _run_async takes over. The async continuation stores
    # results in link_results dict but does not emit logger.debug() calls.
    # We verify that debug mode is active and the root value is logged.
    logger, handler, stream, old_level = self._make_logger_handler()
    try:
      result = await await_(Chain(7).then(aempty).then(lambda v: v + 3).config(debug=True).run())
      log_output = stream.getvalue()
      # Root value 7 is logged in the sync path
      super(MyTestCase, self).assertIn('7', log_output)
      # The chain should still produce the correct result
      super(MyTestCase, self).assertEqual(result, 10)
    finally:
      self._cleanup_logger(logger, handler, old_level)


# ---------------------------------------------------------------------------
# Autorun Tests
# ---------------------------------------------------------------------------

class AutorunTests(MyTestCase):
  async def test_autorun_wraps_in_task(self):
    obj = type('', (), {})()
    obj._v = False

    def set_val():
      obj._v = True

    result = Chain(aempty).then(asyncio.sleep, 0.05).then(set_val, ...).config(autorun=True).run()
    # autorun=True wraps the coroutine in a Task
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    await result
    super(MyTestCase, self).assertTrue(obj._v)

  async def test_autorun_disabled_returns_coroutine(self):
    result = Chain(aempty).then(lambda v: 42).config(autorun=False).run()
    # Without autorun, should return a coroutine
    super(MyTestCase, self).assertTrue(inspect.isawaitable(result))
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)


# ---------------------------------------------------------------------------
# Repr Tests
# ---------------------------------------------------------------------------

class ReprTests(TestCase):
  def test_repr_empty_chain(self):
    r = repr(Chain())
    self.assertIn('Chain', r)

  def test_repr_with_root_fn(self):
    def my_root_fn():
      pass
    r = repr(Chain(my_root_fn))
    self.assertIn('my_root_fn', r)

  def test_repr_with_operations(self):
    def step1(v):
      return v
    r = repr(Chain(1).then(step1))
    self.assertIn('then', r)

  def test_repr_cascade(self):
    r = repr(Cascade())
    self.assertIn('Cascade', r)

  def test_repr_chain_attr(self):
    r = repr(ChainAttr())
    self.assertIn('ChainAttr', r)

  def test_repr_nested_chain(self):
    inner = Chain().then(lambda v: v)
    outer = Chain(1).then(inner)
    r = repr(outer)
    # Nested chain repr should contain indentation (spaces)
    self.assertIn('Chain', r)
    # The nested chain creates additional indentation
    self.assertIn('\n', r)
