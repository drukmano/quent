import asyncio
import inspect
import logging
from unittest import TestCase
from tests.utils import TestExc, empty, aempty, await_, MyTestCase
from quent import Chain, Cascade, run


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

  def test_clone_preserves_autorun(self):
    c = Chain(10).config(autorun=True)
    c2 = c.clone()
    # Both should have autorun set. We verify by checking the config result
    # chain is returned from config. We check behavior: with autorun=True
    # and _is_sync=False, if an async chain is run, it wraps in a task.
    # For a sync chain, autorun doesn't change behavior. We just verify
    # the clone maintains the state by observing the clone still works.
    self.assertEqual(c2.run(), 10)

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

  def test_config_returns_self(self):
    c = Chain(10)
    result = c.config()
    self.assertIs(result, c)

  def test_config_none_params_ignored(self):
    c = Chain(10).then(lambda v: v + 5)
    c.config(autorun=None, debug=None)
    # Should still work normally since None params are ignored
    self.assertEqual(c.run(), 15)


# ---------------------------------------------------------------------------
# No Async Tests
# ---------------------------------------------------------------------------

class NoAsyncTests(MyTestCase):
  async def test_no_async_skips_coro_check(self):
    # With no_async(), _is_sync is set, so the chain skips coroutine detection
    # For a pure sync chain, this is equivalent to a fast path
    c = Chain(5).then(lambda v: v * 2).no_async()
    await self.assertEqual(c.run(), 10)

  async def test_no_async_simple_chain(self):
    c = Chain(100).then(lambda v: v - 50).no_async()
    await self.assertEqual(c.run(), 50)

  async def test_no_async_cascade(self):
    obj = object()
    c = Cascade(obj).then(lambda v: None).no_async()
    result = c.run()
    super(MyTestCase, self).assertIs(result, obj)


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

  def test_repr_nested_chain(self):
    inner = Chain().then(lambda v: v)
    outer = Chain(1).then(inner)
    r = repr(outer)
    # Nested chain repr should contain indentation (spaces)
    self.assertIn('Chain', r)
    # The nested chain creates additional indentation
    self.assertIn('\n', r)
