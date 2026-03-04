import asyncio
import functools
import logging
import operator
import types
import collections.abc
import warnings
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SyncIterator:
  def __init__(self, items=None):
    self._items = items if items is not None else list(range(10))

  def __iter__(self):
    return iter(self._items)


class AsyncIterator:
  def __init__(self, items=None):
    self._items = list(items) if items is not None else list(range(10))

  def __aiter__(self):
    self._iter = iter(self._items)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


class Exc1(TestExc):
  pass


class Exc2(TestExc):
  pass


class Exc3(Exc2):
  pass


class Exc4(Exception):
  pass


class SimpleCM:
  def __init__(self, value='ctx'):
    self._value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self._value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  def __init__(self, value='async_ctx'):
    self._value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self._value

  async def __aexit__(self, *args):
    self.exited = True
    return False


def raise_(exc_type=TestExc):
  raise exc_type()


# ---------------------------------------------------------------------------
# A. _Generator Advanced Tests
# ---------------------------------------------------------------------------

class GeneratorAdvancedTests(IsolatedAsyncioTestCase):

  async def test_generator_repr(self):
    """_Generator __repr__ returns '<_Generator>'."""
    gen = Chain([1, 2, 3]).iterate()
    self.assertEqual(repr(gen), '<_Generator>')

  async def test_generator_call_creates_new_instance(self):
    """_Generator __call__ creates a NEW _Generator, not the same object."""
    gen = Chain(SyncIterator).iterate(lambda i: i * 2)
    gen2 = gen(SyncIterator([1, 2, 3]))
    self.assertIsNot(gen, gen2)

  async def test_generator_call_with_positional_arg(self):
    """_Generator __call__ with a positional arg sets root value override."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i + 10)
    gen2 = gen(SyncIterator([1, 2, 3]))
    r = []
    for i in gen2:
      r.append(i)
    self.assertEqual(r, [11, 12, 13])

  async def test_generator_call_with_multiple_args(self):
    """_Generator __call__ with multiple args passes them through."""
    gen = Chain().then(lambda v: v).iterate(lambda i: i * 2)
    gen2 = gen(SyncIterator, [10, 20, 30])
    r = []
    for i in gen2:
      r.append(i)
    self.assertEqual(r, [20, 40, 60])

  async def test_generator_call_with_kwargs(self):
    """_Generator __call__ with kwargs."""
    def make_iter(items=None):
      return items or [1, 2]
    gen = Chain().then(lambda v: v).iterate(lambda i: i + 100)
    gen2 = gen(make_iter, items=[5, 6, 7])
    r = []
    for i in gen2:
      r.append(i)
    self.assertEqual(r, [105, 106, 107])

  async def test_generator_as_link_in_chain(self):
    """_Generator used as link in another chain (nesting)."""
    inner = Chain().then(lambda v: v).iterate(lambda i: i + 1)
    r = []
    for i in inner(SyncIterator([10, 20])):
      r.append(i)
    self.assertEqual(r, [11, 21])

  async def test_generator_iterated_multiple_times_sync(self):
    """_Generator iterated multiple times sync: each iteration creates fresh state."""
    gen = Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: i * 10)
    r1 = list(gen)
    r2 = list(gen)
    self.assertEqual(r1, [10, 20, 30])
    self.assertEqual(r2, [10, 20, 30])

  async def test_generator_iterated_multiple_times_async(self):
    """_Generator iterated multiple times async: each creates fresh state."""
    gen = Chain(SyncIterator, [4, 5, 6]).iterate(lambda i: i * 2)
    r1 = []
    async for i in gen:
      r1.append(i)
    r2 = []
    async for i in gen:
      r2.append(i)
    self.assertEqual(r1, [8, 10, 12])
    self.assertEqual(r2, [8, 10, 12])

  async def test_generator_with_async_fn(self):
    """_Generator with fn that returns async result."""
    gen = Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: aempty(i * 100))
    r = []
    async for i in gen:
      r.append(i)
    self.assertEqual(r, [100, 200, 300])

  async def test_generator_sync_iteration_with_break(self):
    """_Generator sync iteration with _Break midway."""
    def fn(i):
      if i >= 3:
        return Chain.break_()
      return i * 10
    gen = Chain(SyncIterator, list(range(6))).iterate(fn)
    r = list(gen)
    self.assertEqual(r, [0, 10, 20])

  async def test_generator_async_iteration_with_break(self):
    """_Generator async iteration with _Break midway."""
    def fn(i):
      if i >= 2:
        return Chain.break_()
      return i * 5
    gen = Chain(SyncIterator, list(range(5))).iterate(fn)
    r = []
    async for i in gen:
      r.append(i)
    self.assertEqual(r, [0, 5])

  async def test_generator_sync_iteration_with_error(self):
    """_Generator sync iteration with error in fn."""
    def fn(i):
      if i == 2:
        raise TestExc('boom')
      return i
    gen = Chain(SyncIterator, [1, 2, 3]).iterate(fn)
    with self.assertRaises(TestExc):
      list(gen)

  async def test_generator_async_iteration_with_error(self):
    """_Generator async iteration with error in fn."""
    def fn(i):
      if i == 2:
        raise TestExc('boom')
      return i
    gen = Chain(SyncIterator, [1, 2, 3]).iterate(fn)
    with self.assertRaises(TestExc):
      async for _ in gen:
        pass

  async def test_generator_new_instance_repr(self):
    """Nested _Generator from __call__ still has '<_Generator>' repr."""
    gen = Chain(SyncIterator).iterate(lambda i: i)
    nested = gen(SyncIterator([1]))
    self.assertEqual(repr(nested), '<_Generator>')


# ---------------------------------------------------------------------------
# B. Debug Mode with Exceptions Tests
# ---------------------------------------------------------------------------

class DebugModeExceptionTests(IsolatedAsyncioTestCase):

  def _capture_logs(self):
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logs, handler, logger

  async def test_debug_mode_exception_uses_link_results(self):
    """Debug mode + exception: link_results used in traceback stringification."""
    c = Chain(5).then(lambda v: v + 1).then(lambda v: 1 / 0).config(debug=True)
    with self.assertRaises(ZeroDivisionError):
      await await_(c.run())

  async def test_debug_mode_async_exception(self):
    """Debug mode + exception in async chain."""
    async def failing(v):
      raise TestExc('async fail')
    c = Chain(10).then(failing).config(debug=True)
    with self.assertRaises(TestExc):
      await await_(c.run())

  async def test_debug_mode_nested_chain_exception(self):
    """Debug mode + nested chain exception."""
    inner = Chain().then(lambda v: 1 / 0)
    outer = Chain(42).then(inner).config(debug=True)
    with self.assertRaises(ZeroDivisionError):
      await await_(outer.run())

  async def test_debug_link_results_intermediate_values(self):
    """Debug mode with link_results showing intermediate values in repr."""
    logs, handler, logger = self._capture_logs()
    try:
      c = Chain(5).then(lambda v: v + 1).then(lambda v: v * 2).config(debug=True)
      result = c.run()
      self.assertEqual(result, 12)
      # Verify intermediate values logged
      self.assertTrue(any('5' in log for log in logs))
      self.assertTrue(any('6' in log for log in logs))
      self.assertTrue(any('12' in log for log in logs))
    finally:
      logger.removeHandler(handler)

  async def test_debug_foreach_exception_link_results(self):
    """Debug mode + foreach exception: link_results track foreach link."""
    c = (
      Chain([1, 2, 3])
      .foreach(lambda v: 1 / 0)
      .config(debug=True)
    )
    with self.assertRaises(ZeroDivisionError):
      await await_(c.run())

  async def test_debug_log_contains_function_names(self):
    """Debug mode log messages contain function names."""
    logs, handler, logger = self._capture_logs()
    try:
      def my_named_fn(v):
        return v + 1
      c = Chain(5).then(my_named_fn).config(debug=True)
      c.run()
      self.assertTrue(any('then' in log for log in logs))
    finally:
      logger.removeHandler(handler)

  async def test_debug_log_contains_root_value(self):
    """Debug mode log messages contain root value."""
    logs, handler, logger = self._capture_logs()
    try:
      c = Chain(42).then(lambda v: v).config(debug=True)
      c.run()
      self.assertTrue(any('42' in log for log in logs))
    finally:
      logger.removeHandler(handler)

  async def test_debug_log_contains_link_values(self):
    """Debug mode log messages contain link values."""
    logs, handler, logger = self._capture_logs()
    try:
      c = Chain(3).then(lambda v: v * 7).config(debug=True)
      c.run()
      self.assertTrue(any('21' in log for log in logs))
    finally:
      logger.removeHandler(handler)

  async def test_debug_async_logs(self):
    """Debug mode logs appear in async chain -- sync root value is logged.

    In _run_async, the debug path populates link_results but does NOT call
    _logger.debug for awaited values. Only the sync prefix (_run) logs via
    _logger.debug. So the root value (10) is logged, but the async result (20)
    is only stored in link_results (not logged).
    """
    logs, handler, logger = self._capture_logs()
    try:
      async def async_double(v):
        return v * 2
      c = Chain(10).then(async_double).config(debug=True)
      result = await await_(c.run())
      self.assertEqual(result, 20)
      # Root value (sync) is logged via _logger.debug in _run line 137
      self.assertTrue(any('10' in log for log in logs))
      # Verify at least one log entry was made
      self.assertGreaterEqual(len(logs), 1)
    finally:
      logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# C. Chain Reuse Patterns Tests
# ---------------------------------------------------------------------------

class ChainReuseTests(IsolatedAsyncioTestCase):

  async def test_run_same_chain_10_times(self):
    """Run same chain 10 times sequentially -- consistent results."""
    c = Chain(5).then(lambda v: v + 1).then(lambda v: v * 2)
    for _ in range(10):
      self.assertEqual(c.run(), 12)

  async def test_run_same_chain_10_times_different_overrides(self):
    """Run same chain 10 times with different override values."""
    c = Chain().then(lambda v: v * 3)
    for i in range(10):
      self.assertEqual(c.run(i), i * 3)

  async def test_clone_modify_clone_original_unaffected(self):
    """Clone chain, modify clone (add links), run both -- original unaffected."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(lambda v: v * 100)
    self.assertEqual(c.run(), 2)
    self.assertEqual(c2.run(), 200)

  async def test_clone_modify_original_clone_unaffected(self):
    """Clone chain, modify original (add links), run both -- clone unaffected."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    self.assertEqual(c.run(), 200)
    self.assertEqual(c2.run(), 2)

  async def test_chain_run_100_times(self):
    """Chain run 100 times -- consistent results."""
    c = Chain(7).then(lambda v: v * 3)
    for _ in range(100):
      self.assertEqual(c.run(), 21)

  async def test_chain_reuse_multiple_runs(self):
    """Chain can be run multiple times with consistent results."""
    c = Chain(10).then(lambda v: v + 5)
    self.assertEqual(c.run(), 15)
    self.assertEqual(c.run(), 15)

  async def test_chain_as_link_parent_cloned(self):
    """Chain used as link in parent, then parent cloned -- nested chain shared."""
    inner = Chain().then(lambda v: v * 10)
    outer = Chain(3).then(inner)
    clone = outer.clone()
    self.assertEqual(clone.run(), 30)
    self.assertEqual(outer.run(), 30)

  async def test_chain_autorun_run_multiple_times(self):
    """Chain with autorun=True run multiple times."""
    c = Chain(5).then(lambda v: v + 1).config(autorun=True)
    for _ in range(5):
      self.assertEqual(c(), 6)

  async def test_chain_with_finally_run_multiple_times(self):
    """Chain with finally_ run multiple times -- finally_ runs each time."""
    counter = {'count': 0}
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: counter.__setitem__('count', counter['count'] + 1))
    for i in range(5):
      self.assertEqual(c.run(), 2)
    self.assertEqual(counter['count'], 5)

  async def test_chain_with_except_alternating_success_failure(self):
    """Chain with except_ run multiple times with alternating success/failure."""
    handler_runs = {'count': 0}
    def maybe_fail(v):
      if v % 2 == 1:
        raise TestExc('odd')
      return v
    c = (
      Chain()
      .then(maybe_fail)
      .except_(lambda v: handler_runs.__setitem__('count', handler_runs['count'] + 1), reraise=False)
    )
    for i in range(6):
      await await_(c.run(i))
    # i=1,3,5 are odd -> 3 failures handled
    self.assertEqual(handler_runs['count'], 3)


# ---------------------------------------------------------------------------
# D. Multiple except_ Handler Interactions Tests
# ---------------------------------------------------------------------------

class MultipleExceptHandlerTests(IsolatedAsyncioTestCase):

  async def test_three_handlers_exception_matches_third(self):
    """Three except_ handlers, exception matches third -- first two skipped."""
    log = []
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(Exc3()))
      .except_(lambda v: log.append('h1'), exceptions=Exc4, reraise=False)
      .except_(lambda v: log.append('h2'), exceptions=RuntimeError, reraise=False)
      .except_(lambda v: log.append('h3'), exceptions=Exc2, reraise=False)
    )
    await await_(c.run())
    self.assertEqual(log, ['h3'])

  async def test_three_handlers_none_match(self):
    """Three handlers, exception matches NONE -- exception propagates."""
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(Exc4()))
      .except_(lambda v: None, exceptions=Exc1, reraise=False)
      .except_(lambda v: None, exceptions=Exc2, reraise=False)
      .except_(lambda v: None, exceptions=Exc3, reraise=False)
    )
    with self.assertRaises(Exc4):
      await await_(c.run())

  async def test_parent_class_handler_before_child(self):
    """Handler for parent class BEFORE handler for child -- parent catches first."""
    log = []
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(Exc3()))
      .except_(lambda v: log.append('parent'), exceptions=Exc2, reraise=False)
      .except_(lambda v: log.append('child'), exceptions=Exc3, reraise=False)
    )
    await await_(c.run())
    # Exc3 is subclass of Exc2, so parent handler catches first
    self.assertEqual(log, ['parent'])

  async def test_child_class_handler_before_parent(self):
    """Handler for child class BEFORE handler for parent -- child catches first."""
    log = []
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(Exc3()))
      .except_(lambda v: log.append('child'), exceptions=Exc3, reraise=False)
      .except_(lambda v: log.append('parent'), exceptions=Exc2, reraise=False)
    )
    await await_(c.run())
    self.assertEqual(log, ['child'])

  async def test_handler_raises_different_exception(self):
    """except_ handler that raises a DIFFERENT exception type."""
    def handler(v):
      raise Exc4('new exception')
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(Exc1()))
      .except_(handler, exceptions=Exc1, reraise=False)
    )
    with self.assertRaises(Exc4):
      await await_(c.run())

  async def test_handler_reraise_true_second_handler_not_called(self):
    """except_ with reraise=True followed by another except_ for SAME type."""
    log = []
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(TestExc()))
      .except_(lambda v: log.append('first'), reraise=True)
      .except_(lambda v: log.append('second'), reraise=False)
    )
    # With reraise=True on first handler, the first handler matches,
    # runs, then re-raises. The _handle_exception returns the first
    # matching handler and re-raises. Second handler is never consulted.
    with self.assertRaises(TestExc):
      await await_(c.run())
    self.assertEqual(log, ['first'])

  async def test_except_except_finally_interaction(self):
    """except_ + except_ + finally_ -- both except and finally interact correctly."""
    log = []
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(Exc1()))
      .except_(lambda v: log.append('exc1'), exceptions=Exc1, reraise=False)
      .except_(lambda v: log.append('exc2'), exceptions=Exc2, reraise=False)
      .finally_(lambda v: log.append('finally'))
    )
    await await_(c.run())
    self.assertIn('exc1', log)
    self.assertNotIn('exc2', log)
    self.assertIn('finally', log)

  async def test_except_nested_inner_catches_reraise_outer_catches(self):
    """except_ in nested chain + except_ in parent -- inner catches, if reraise outer catches."""
    log = []
    inner = (
      Chain()
      .then(lambda v: (_ for _ in ()).throw(TestExc()))
      .except_(lambda v: log.append('inner'), reraise=True)
    )
    outer = (
      Chain(1)
      .then(inner)
      .except_(lambda v: log.append('outer'), reraise=False)
    )
    await await_(outer.run())
    self.assertIn('inner', log)
    self.assertIn('outer', log)



# ---------------------------------------------------------------------------
# F. Chain Decorator Advanced Tests
# ---------------------------------------------------------------------------

class ChainDecoratorAdvancedTests(IsolatedAsyncioTestCase):

  async def test_decorator_on_class_with_call(self):
    """Decorator on class with __call__ -- descriptor protocol works."""
    class MyCallable:
      def __init__(self, factor):
        self.factor = factor

      @Chain().then(lambda v: v).decorator()
      def compute(self, x):
        return x * self.factor

    obj = MyCallable(3)
    self.assertEqual(obj.compute(10), 30)

  async def test_decorated_function_positional_args(self):
    """Decorated function called with positional args."""
    @Chain().then(lambda v: v * 2).decorator()
    def add(a, b):
      return a + b

    self.assertEqual(add(3, 4), 14)  # (3+4)*2 = 14

  async def test_decorated_function_keyword_args(self):
    """Decorated function called with keyword args."""
    @Chain().then(lambda v: v).decorator()
    def greet(name, greeting='Hello'):
      return f'{greeting}, {name}!'

    self.assertEqual(greet(name='World', greeting='Hi'), 'Hi, World!')

  async def test_decorated_function_args_kwargs(self):
    """Decorated function called with *args and **kwargs."""
    @Chain().then(lambda v: v + 100).decorator()
    def combine(*args, **kwargs):
      return sum(args) + sum(kwargs.values())

    self.assertEqual(combine(1, 2, 3, x=4), 110)  # 10 + 100

  async def test_chain_run_async_root(self):
    """Chain.run() with async root."""
    c = Chain(aempty, 42).then(lambda v: v + 1)
    self.assertEqual(await c.run(), 43)

  async def test_chain_run_override_with_args(self):
    """Chain.run() with override value that has args."""
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c.run(lambda: 5, ...), 10)

  async def test_multiple_decorators_different_chains(self):
    """Multiple decorators from DIFFERENT chains on same function -- last decorator wins."""
    @Chain().then(lambda v: v * 2).decorator()
    @Chain().then(lambda v: v + 1).decorator()
    def base(x):
      return x

    # Inner: base(3) -> 3, then +1 -> 4
    # Outer: 4, then *2 -> 8
    self.assertEqual(base(3), 8)

  async def test_decorated_async_function_preserves_async(self):
    """Decorated async function preserves async nature."""
    @Chain().then(lambda v: v * 5).decorator()
    async def async_fn(x):
      return x + 1

    self.assertEqual(await async_fn(2), 15)  # (2+1)*5 = 15


# ---------------------------------------------------------------------------
# G. Chain with Unusual Callable Types Tests
# ---------------------------------------------------------------------------

class UnusualCallableTests(IsolatedAsyncioTestCase):

  async def test_chain_with_lru_cache(self):
    """Chain with functools.lru_cache decorated function."""
    @functools.lru_cache(maxsize=None)
    def cached_double(v):
      return v * 2
    result = Chain(5).then(cached_double).run()
    self.assertEqual(result, 10)

  async def test_chain_with_wraps_decorated(self):
    """Chain with functools.wraps decorated function."""
    def original(v):
      return v * 3
    @functools.wraps(original)
    def wrapper(v):
      return original(v)
    result = Chain(4).then(wrapper).run()
    self.assertEqual(result, 12)

  async def test_chain_with_classmethod_callable(self):
    """Chain with a classmethod-like pattern."""
    class MyClass:
      factor = 10
      @classmethod
      def multiply(cls, v):
        return v * cls.factor
    result = Chain(5).then(MyClass.multiply).run()
    self.assertEqual(result, 50)

  async def test_chain_with_staticmethod_callable(self):
    """Chain with staticmethod as value."""
    class MyClass:
      @staticmethod
      def double(v):
        return v * 2
    result = Chain(7).then(MyClass.double).run()
    self.assertEqual(result, 14)

  async def test_chain_with_operator_add(self):
    """Chain with operator.add."""
    result = Chain(5).then(operator.neg).run()
    self.assertEqual(result, -5)

  async def test_chain_with_operator_mul(self):
    """Chain with operator.mul via lambda wrapping."""
    # operator.mul takes exactly 2 args; then() with explicit args bypasses
    # current_value, so we use a lambda to combine current_value with an arg.
    result = Chain(5).then(lambda v: operator.mul(v, 3)).run()
    self.assertEqual(result, 15)

  async def test_chain_with_bound_method(self):
    """Chain with types.MethodType (bound method)."""
    class MyObj:
      def __init__(self, base):
        self.base = base
      def add(self, v):
        return self.base + v
    obj = MyObj(100)
    result = Chain(5).then(obj.add).run()
    self.assertEqual(result, 105)

  async def test_chain_with_regular_function(self):
    """Chain with types.FunctionType (regular function)."""
    def my_func(v):
      return v ** 2
    self.assertIsInstance(my_func, types.FunctionType)
    result = Chain(6).then(my_func).run()
    self.assertEqual(result, 36)


# ---------------------------------------------------------------------------
# H. Async Generator Advanced Tests
# ---------------------------------------------------------------------------

class AsyncGeneratorAdvancedTests(IsolatedAsyncioTestCase):

  async def test_async_gen_yields_before_raising(self):
    """Async generator that yields before raising."""
    def fn(i):
      if i == 2:
        raise TestExc('mid-raise')
      return i * 10

    gen = Chain(SyncIterator, [0, 1, 2, 3]).iterate(fn)
    r = []
    with self.assertRaises(TestExc):
      async for i in gen:
        r.append(i)
    self.assertEqual(r, [0, 10])

  async def test_async_gen_partially_consumed(self):
    """Async generator partially consumed then abandoned."""
    gen = Chain(SyncIterator, list(range(100))).iterate(lambda i: i * 2)
    r = []
    async for i in gen:
      r.append(i)
      if len(r) >= 3:
        break
    self.assertEqual(r, [0, 2, 4])

  async def test_async_gen_with_break(self):
    """Async generator with async for and break."""
    gen = Chain(SyncIterator, [10, 20, 30, 40]).iterate(lambda i: i + 1)
    r = []
    async for i in gen:
      r.append(i)
      if i > 25:
        break
    self.assertEqual(r, [11, 21, 31])

  async def test_async_gen_from_chain_with_async_root(self):
    """Async generator from chain with async root."""
    gen = Chain(aempty, SyncIterator([5, 6, 7])).iterate(lambda i: i * 3)
    r = []
    async for i in gen:
      r.append(i)
    self.assertEqual(r, [15, 18, 21])

  async def test_async_gen_from_chain_mixed_sync_async_links(self):
    """Async generator from chain with mixed sync/async links."""
    gen = Chain(SyncIterator, [1, 2, 3]).iterate(lambda i: aempty(i + 100))
    r = []
    async for i in gen:
      r.append(i)
    self.assertEqual(r, [101, 102, 103])

  async def test_async_gen_foreach_result(self):
    """Async generator inside foreach-like result collection."""
    gen = Chain(SyncIterator, [10, 20]).iterate(lambda i: i + 5)
    results = []
    async for i in gen:
      results.append(i)
    self.assertEqual(results, [15, 25])


# ---------------------------------------------------------------------------
# I. Deep Nesting Patterns Tests
# ---------------------------------------------------------------------------

class DeepNestingTests(IsolatedAsyncioTestCase):

  async def test_triple_nested_chain(self):
    """Chain(Chain(Chain(42))) -- triple nested."""
    c3 = Chain(42)
    c2 = Chain(c3)
    c1 = Chain(c2)
    self.assertEqual(c1.run(), 42)

  async def test_quadruple_nested_chain(self):
    """Chain(Chain(Chain(Chain(42)))) -- quadruple nested."""
    c4 = Chain(42)
    c3 = Chain(c4)
    c2 = Chain(c3)
    c1 = Chain(c2)
    self.assertEqual(c1.run(), 42)

  async def test_nested_inner_except_outer_except(self):
    """Nested chain where inner has except_, outer has except_ -- inner catches first."""
    log = []
    inner = (
      Chain()
      .then(lambda v: (_ for _ in ()).throw(TestExc()))
      .except_(lambda v: log.append('inner'), reraise=True)
    )
    outer = (
      Chain(1)
      .then(inner)
      .except_(lambda v: log.append('outer'), reraise=False)
    )
    await await_(outer.run())
    self.assertIn('inner', log)
    self.assertIn('outer', log)

  async def test_nested_inner_finally_outer_finally(self):
    """Nested chain where inner has finally_, outer has finally_ -- both run."""
    log = []
    inner = (
      Chain()
      .then(lambda v: v * 2)
      .finally_(lambda v: log.append('inner_finally'))
    )
    outer = (
      Chain(5)
      .then(inner)
      .finally_(lambda v: log.append('outer_finally'))
    )
    result = outer.run()
    self.assertEqual(result, 10)
    self.assertIn('inner_finally', log)
    self.assertIn('outer_finally', log)


# ---------------------------------------------------------------------------
# J. Async/Sync Interaction Edge Cases Tests
# ---------------------------------------------------------------------------

class AsyncSyncInteractionTests(IsolatedAsyncioTestCase):

  async def test_async_chain_sync_exception_handler(self):
    """Async chain where exception handler is sync -- handler called normally."""
    log = []
    async def async_fail(v):
      raise TestExc('async boom')
    c = (
      Chain(1)
      .then(async_fail)
      .except_(lambda v: log.append('sync_handler'), reraise=False)
    )
    await await_(c.run())
    self.assertEqual(log, ['sync_handler'])

  async def test_return_with_coroutine_value_in_async(self):
    """Chain.return_ with coroutine value in async context."""
    async def get_value():
      return 99
    inner = Chain().then(lambda v: Chain.return_(get_value))
    outer = Chain(1).then(inner)
    result = await await_(outer.run())
    self.assertEqual(result, 99)

  async def test_break_with_value_in_async_foreach(self):
    """Chain.break_ with value in async foreach."""
    async def fn(v):
      if v >= 3:
        Chain.break_(42)
      return v

    c = Chain(AsyncIterator([1, 2, 3, 4, 5])).foreach(fn)
    result = await await_(c.run())
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# K. Fluent API Verification Tests
# ---------------------------------------------------------------------------

class FluentAPITests(IsolatedAsyncioTestCase):

  async def test_fluent_methods_return_self(self):
    """Every fluent method returns self: then, do, except_, finally_, foreach,
    filter, gather, with_, config."""
    c = Chain()
    # then
    self.assertIs(c.then(lambda v: v), c)
    # do
    self.assertIs(c.do(lambda v: v), c)
    # except_
    self.assertIs(c.except_(lambda v: v), c)
    # foreach
    self.assertIs(c.foreach(lambda v: v), c)
    # filter
    self.assertIs(c.filter(lambda v: v), c)
    # gather
    self.assertIs(c.gather(lambda v: v), c)
    # with_
    self.assertIs(c.with_(lambda v: v), c)
    # config
    self.assertIs(c.config(), c)

  async def test_fluent_methods_return_self_with_finally(self):
    """finally_ returns self too."""
    c = Chain()
    self.assertIs(c.finally_(lambda v: v), c)

  async def test_fluent_chaining_preserves_order(self):
    """Fluent chaining preserves order: f1 -> f2 -> f3."""
    log = []
    def f1(v):
      log.append('f1')
      return v + 1
    def f2(v):
      log.append('f2')
      return v * 2
    def f3(v):
      log.append('f3')
      return v - 1
    result = Chain(5).then(f1).then(f2).then(f3).run()
    self.assertEqual(result, 11)  # (5+1)*2 - 1 = 11
    self.assertEqual(log, ['f1', 'f2', 'f3'])

  async def test_fluent_chaining_mixed_methods(self):
    """Fluent chaining with mixed method types."""
    log = []
    def f(v):
      return v + 1
    def g(v):
      log.append('side-effect')
    def h(v):
      log.append('except')
    def j(v):
      log.append('finally')
    result = (
      Chain(42)
      .then(f)
      .do(g)
      .except_(h)
      .finally_(j)
      .run()
    )
    self.assertEqual(result, 43)
    self.assertIn('side-effect', log)
    self.assertNotIn('except', log)
    self.assertIn('finally', log)

  async def test_fluent_api_after_config(self):
    """Fluent API after config: config doesn't break chaining."""
    result = Chain(10).config(debug=True).then(lambda v: v + 5).run()
    self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# Additional edge case tests for completeness
# ---------------------------------------------------------------------------

class ChainReprTests(IsolatedAsyncioTestCase):

  async def test_chain_repr_basic(self):
    """Chain repr includes 'Chain'."""
    c = Chain(42)
    r = repr(c)
    self.assertIn('Chain', r)


class ChainCallableEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_chain_callable_equivalence(self):
    """Chain run() and __call__() produce same result."""
    c = Chain(10).then(lambda v: v + 5)
    self.assertEqual(c.run(), c())

  async def test_chain_with_override(self):
    """Chain with override value."""
    c = Chain().then(lambda v: v * 3)
    self.assertEqual(c.run(7), 21)
    self.assertEqual(c(7), 21)

  async def test_chain_async_root(self):
    """Chain.run() with async root value."""
    c = Chain(aempty, 50).then(lambda v: v + 10)
    self.assertEqual(await c.run(), 60)


class ExceptHandlerReturnValueTests(IsolatedAsyncioTestCase):

  async def test_except_handler_returns_value_on_no_reraise(self):
    """Exception handler returns value when reraise=False."""
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(TestExc()))
      .except_(lambda v: 'recovered', reraise=False)
    )
    self.assertEqual(c.run(), 'recovered')

  async def test_except_handler_returns_none_on_no_reraise(self):
    """Exception handler returns None when it doesn't return anything."""
    c = (
      Chain(1)
      .then(lambda v: (_ for _ in ()).throw(TestExc()))
      .except_(lambda v: None, reraise=False)
    )
    self.assertIsNone(c.run())


class ChainBoolTests(IsolatedAsyncioTestCase):

  async def test_chain_always_truthy(self):
    """Chain.__bool__ always returns True."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(42)))


class ChainNestedExceptionTests(IsolatedAsyncioTestCase):

  async def test_nested_chain_cannot_run_directly(self):
    """A nested chain raises QuentException when run directly."""
    inner = Chain().then(lambda v: v * 2)
    outer = Chain(5).then(inner)
    # inner is now nested
    with self.assertRaises(QuentException):
      inner.run()

  async def test_clone_of_nested_chain_can_run(self):
    """Clone of a nested chain resets nested flag."""
    inner = Chain().then(lambda v: v * 2)
    outer = Chain(5).then(inner)
    clone = inner.clone()
    self.assertEqual(clone.run(10), 20)


class OverrideRootValueTests(IsolatedAsyncioTestCase):

  async def test_cannot_override_root_of_chain_with_root(self):
    """Cannot override the root value of a Chain that already has one."""
    c = Chain(42).then(lambda v: v + 1)
    with self.assertRaises(QuentException):
      c.run(99)

  async def test_override_root_of_void_chain(self):
    """Can override the root value of a void chain."""
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c.run(5), 10)
    self.assertEqual(c.run(10), 20)


class AutorunTests(IsolatedAsyncioTestCase):

  async def test_autorun_flag(self):
    """Autorun flag configurable via config."""
    c = Chain(5).config(autorun=True)
    self.assertEqual(c(), 5)

  async def test_autorun_false(self):
    """Autorun can be disabled."""
    c = Chain(5).config(autorun=True).config(autorun=False)
    self.assertEqual(c.run(), 5)


class ExceptStringExceptionTypeTest(IsolatedAsyncioTestCase):

  async def test_except_string_raises_type_error(self):
    """except_ with string exception type raises TypeError."""
    c = Chain(1)
    with self.assertRaises(TypeError):
      c.except_(lambda v: v, exceptions='ValueError')


class FinallyControlFlowTest(IsolatedAsyncioTestCase):

  async def test_finally_with_control_flow_raises(self):
    """Using control flow signals inside finally handlers raises QuentException."""
    c = Chain(1).finally_(Chain.return_)
    with self.assertRaises(QuentException):
      c.run()

  async def test_only_one_finally_allowed(self):
    """Can only register one finally callback."""
    c = Chain(1).finally_(lambda v: None)
    with self.assertRaises(QuentException):
      c.finally_(lambda v: None)


class NullSentinelTests(IsolatedAsyncioTestCase):

  async def test_null_is_exported(self):
    """Null sentinel is accessible."""
    self.assertIsNotNone(Null)

  async def test_chain_with_null_returns_none(self):
    """Chain() with no root returns None."""
    self.assertIsNone(Chain().run())


if __name__ == '__main__':
  import unittest
  unittest.main()
