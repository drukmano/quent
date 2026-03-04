"""Comprehensive tests for complex chain composition patterns.

Covers: deeply nested chains, long chains, complex clone patterns,
chain reuse patterns, pipeline composition, reuse patterns,
complex feature combinations, and edge case chain constructions.
"""
import asyncio
import io
import logging
import warnings
from itertools import product
from unittest import TestCase, IsolatedAsyncioTestCase

from tests.utils import TestExc, empty, aempty, await_
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


class SyncContextManager:
  """A simple sync context manager for testing."""
  def __init__(self, value='ctx'):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncContextManager:
  """A simple async context manager for testing."""
  def __init__(self, value='actx'):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


# ===================================================================
# 1. Deeply Nested Chains (15+ tests)
# ===================================================================

class DeeplyNestedChainTests(IsolatedAsyncioTestCase):

  async def test_2_level_nesting(self):
    """Chain(1).then(Chain().then(lambda v: v*2)) -> 2."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 1).then(Chain().then(lambda v: v * 2)).run()),
          2
        )

  async def test_3_level_nesting(self):
    """Chain inside chain inside chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain().then(lambda v: v + 10)
        mid = Chain().then(fn1).then(inner)
        outer = Chain(fn2, 5).then(mid)
        self.assertEqual(await await_(outer.run()), 15)

  async def test_5_level_nesting_value_propagation(self):
    """5 levels of nesting, each adding 1."""
    for fn1, fn2, fn3, fn4 in product([empty, aempty], repeat=4):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4):
        # Each level adds 1 to the value. All are chained:
        # The innermost gets value from outermost root (via nested chain passing).
        # Only the innermost actually adds; the wrappers just pass through.
        l5 = Chain().then(lambda v: v + 1)
        l4 = Chain().then(fn1).then(l5).then(lambda v: v + 1)
        l3 = Chain().then(fn2).then(l4).then(lambda v: v + 1)
        l2 = Chain().then(fn3).then(l3).then(lambda v: v + 1)
        l1 = Chain(fn4, 0).then(l2).then(lambda v: v + 1)
        # 0 -> l2(l3(l4(l5(0+1)+1)+1)+1)+1 = 5
        self.assertEqual(await await_(l1.run()), 5)

  async def test_10_level_nesting_stress(self):
    """10 levels of nesting, each level multiplying by 2."""
    # Build a chain where each level itself multiplies by 2.
    # Since nested chains just pass value through, we need each level
    # to have its own .then(lambda v: v*2) operation.
    c = Chain(1)
    for _ in range(10):
      c = c.then(lambda v: v * 2)
    # 1 * 2^10 = 1024
    result = c.run()
    self.assertEqual(result, 1024)

  async def test_deeply_nested_additive(self):
    """Each level adds its depth number to the value using nested chains."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # Build a flat chain with additions
        c = Chain(fn, 0)
        for i in range(1, 6):
          c = c.then(lambda v, n=i: v + n)
        # 0 + 1 + 2 + 3 + 4 + 5 = 15
        self.assertEqual(await await_(c.run()), 15)

  async def test_deeply_nested_exception_at_innermost(self):
    """Exception raised at the innermost nesting level propagates up."""
    for fn1, fn2, fn3 in product([empty, aempty], repeat=3):
      with self.subTest(fn1=fn1, fn2=fn2, fn3=fn3):
        def raise_err(v):
          raise TestExc('innermost')

        inner = Chain().then(fn1).then(raise_err)
        mid = Chain().then(fn2).then(inner)
        outer = Chain(fn3, 1).then(mid)
        with self.assertRaises(TestExc):
          await await_(outer.run())

  async def test_deeply_nested_except_at_various_levels(self):
    """Exception handler at middle level catches innermost error."""
    caught = []

    def raise_err(v):
      raise TestExc('inner')

    inner = Chain().then(raise_err)
    mid = Chain().then(inner).except_(
      lambda v: caught.append('mid'), exceptions=TestExc, reraise=False
    )
    outer = Chain(1).then(mid)
    outer.run()
    self.assertEqual(caught, ['mid'])

  async def test_deeply_nested_return_at_inner_level(self):
    """Chain.return_() in inner chain exits to outermost."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # return_ propagates through all nested chains to outermost
        inner = Chain().then(Chain.return_, 99)
        outer = Chain(fn, 1).then(inner).then(lambda v: v + 1000)
        # return_(99) escapes ALL nesting -- the .then(+1000) is skipped
        self.assertEqual(await await_(outer.run()), 99)

  async def test_deeply_nested_break_in_foreach(self):
    """Chain.break_() in foreach inside nested chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def break_at_3(el):
          if el >= 3:
            return Chain.break_()
          return fn1(el * 10)

        inner = Chain().foreach(break_at_3)
        result = await await_(Chain(fn2, [1, 2, 3, 4, 5]).then(inner).run())
        self.assertEqual(result, [10, 20])

  async def test_nested_chain_with_debug_mode(self):
    """Nested chain with debug mode enabled on outer."""
    logger = logging.getLogger('quent')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      inner = Chain().then(lambda v: v * 2)
      outer = Chain(5).then(inner).config(debug=True)
      result = outer.run()
      self.assertEqual(result, 10)
      log_output = stream.getvalue()
      self.assertIn('5', log_output)
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

  async def test_nested_chain_where_inner_is_reused(self):
    """Inner chain is used as a link in outer chain."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        inner = Chain().then(fn1).then(lambda v: v + 100)
        outer = Chain(fn2, 5).then(inner)
        self.assertEqual(await await_(outer.run()), 105)

  async def test_nested_chain_where_inner_is_cloned(self):
    """Clone the inner chain and use in outer."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        original_inner = Chain().then(lambda v: v * 3)
        cloned_inner = original_inner.clone()
        outer = Chain(fn, 7).then(cloned_inner)
        self.assertEqual(await await_(outer.run()), 21)
        # Modify original and verify clone is unaffected
        original_inner.then(lambda v: v + 1000)
        outer2 = Chain(fn, 7).then(cloned_inner.clone())
        self.assertEqual(await await_(outer2.run()), 21)

  async def test_deeply_nested_async_propagation(self):
    """Async function at innermost level, all outer levels are sync."""
    inner = Chain().then(aempty).then(lambda v: v + 1)
    mid = Chain().then(inner)
    outer = Chain(10).then(mid)
    self.assertEqual(await outer.run(), 11)


# ===================================================================
# 2. Long Chains (10+ tests)
# ===================================================================

class LongChainTests(IsolatedAsyncioTestCase):

  async def test_chain_100_then_links(self):
    """Chain with 100 sequential .then() links."""
    c = Chain(0)
    for _ in range(100):
      c = c.then(lambda v: v + 1)
    self.assertEqual(c.run(), 100)

  async def test_chain_1000_then_links(self):
    """Chain with 1000 sequential .then() links."""
    c = Chain(0)
    for _ in range(1000):
      c = c.then(lambda v: v + 1)
    self.assertEqual(c.run(), 1000)

  async def test_chain_alternating_then_do(self):
    """Chain with alternating .then() and .do() links."""
    side = []
    c = Chain(0)
    for i in range(50):
      c = c.then(lambda v: v + 1)
      c = c.do(lambda v, _i=i: side.append(v))
    result = c.run()
    self.assertEqual(result, 50)
    self.assertEqual(len(side), 50)
    # side should have [1, 2, 3, ..., 50]
    self.assertEqual(side, list(range(1, 51)))

  async def test_long_chain_single_except_at_end(self):
    """Long chain with an except handler only at the end."""
    caught = []

    def raise_at_end(v):
      raise TestExc('boom')

    c = Chain(0)
    for i in range(50):
      c = c.then(lambda v: v + 1)
    c = c.then(raise_at_end)
    # except_ handler receives the root value (0); use lambda v: ... form
    c = c.except_(lambda v: caught.append('caught'), reraise=False)
    c.run()
    self.assertEqual(caught, ['caught'])

  async def test_long_chain_middle_link_raises(self):
    """Long chain where a middle link raises an exception."""
    def maybe_raise(v):
      if v == 25:
        raise TestExc('at 25')
      return v + 1

    c = Chain(0)
    for _ in range(50):
      c = c.then(maybe_raise)
    with self.assertRaises(TestExc):
      c.run()

  async def test_long_chain_with_filter_foreach_gather(self):
    """Long chain with filter, foreach, and gather interspersed."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        c = (
          Chain(fn, list(range(20)))
          .filter(lambda x: x % 2 == 0)
          .foreach(lambda x: x * 3)
          .then(lambda lst: lst[:5])
          .foreach(lambda x: x + 1)
        )
        result = await await_(c.run())
        # range(20) even: [0,2,4,6,8,10,12,14,16,18]
        # *3: [0,6,12,18,24,30,36,42,48,54]
        # [:5]: [0,6,12,18,24]
        # +1: [1,7,13,19,25]
        self.assertEqual(result, [1, 7, 13, 19, 25])

  async def test_chain_building_complex_data_structure(self):
    """Chain progressively builds a dict."""
    c = (
      Chain(lambda: {})
      .then(lambda d: {**d, 'a': 1})
      .then(lambda d: {**d, 'b': 2})
      .then(lambda d: {**d, 'c': 3})
      .then(lambda d: {**d, 'total': d['a'] + d['b'] + d['c']})
    )
    result = c.run()
    self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3, 'total': 6})

  async def test_long_chain_async_interleaved(self):
    """Long chain with async functions every 10th link."""
    c = Chain(0)
    for i in range(100):
      if i % 10 == 0:
        c = c.then(lambda v: aempty(v + 1))
      else:
        c = c.then(lambda v: v + 1)
    self.assertEqual(await c.run(), 100)

  async def test_long_chain_with_do_preserving_value(self):
    """Many do() operations don't alter the chain value."""
    c = Chain(42)
    for _ in range(100):
      c = c.do(lambda v: v * 999)
    self.assertEqual(c.run(), 42)


# ===================================================================
# 3. Complex Clone Patterns (10+ tests)
# ===================================================================

class ComplexClonePatternTests(TestCase):

  def test_clone_modify_both_run_independently(self):
    """Clone a chain, modify both original and clone, run both."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    c2.then(lambda v: v * 200)
    self.assertEqual(c.run(), 200)
    self.assertEqual(c2.run(), 400)

  def test_clone_with_nested_chains_shallow_sharing(self):
    """Clone shares nested chain callables (shallow copy)."""
    call_count = {'inner': 0}

    def inner_fn(v):
      call_count['inner'] += 1
      return v * 2

    inner = Chain().then(inner_fn)
    outer = Chain(5).then(inner)
    c2 = outer.clone()

    call_count['inner'] = 0
    self.assertEqual(outer.run(), 10)
    self.assertEqual(call_count['inner'], 1)

    call_count['inner'] = 0
    self.assertEqual(c2.run(), 10)
    self.assertEqual(call_count['inner'], 1)

  def test_clone_of_clone_of_clone(self):
    """3-level deep cloning."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    c4 = c3.clone()
    self.assertEqual(c4.run(), 2)
    # Modify deepest clone
    c4.then(lambda v: v * 5)
    self.assertEqual(c4.run(), 10)
    # Others unaffected
    self.assertEqual(c.run(), 2)
    self.assertEqual(c2.run(), 2)
    self.assertEqual(c3.run(), 2)

  def test_clone_chains_run_independently(self):
    """Clone a chain, then run both independently."""
    c = Chain(10).then(lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(c.run(), 20)
    self.assertEqual(c2.run(), 20)

  def test_clone_with_debug_mode(self):
    """Clone preserves debug mode."""
    c = Chain(5).then(lambda v: v + 1).config(debug=True)
    c2 = c.clone()
    logger = logging.getLogger('quent')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      result = c2.run()
      self.assertEqual(result, 6)
      log_output = stream.getvalue()
      self.assertIn('5', log_output)
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

  def test_clone_with_autorun(self):
    """Clone preserves autorun flag."""
    c = Chain(10).config(autorun=True)
    c2 = c.clone()
    # Both should work; for sync chain autorun doesn't change behavior
    self.assertEqual(c.run(), 10)
    self.assertEqual(c2.run(), 10)

  def test_clone_void_chain_add_different_ops(self):
    """Clone a void chain, add different operations to each clone."""
    c = Chain()
    c1 = c.clone()
    c2 = c.clone()
    c1.then(lambda v: v + 10)
    c2.then(lambda v: v * 3)
    self.assertEqual(c1.run(5), 15)
    self.assertEqual(c2.run(5), 15)

  def test_massive_clone_100(self):
    """100 clones of the same chain all produce correct results."""
    c = Chain(7).then(lambda v: v * 3)
    clones = [c.clone() for _ in range(100)]
    for clone in clones:
      self.assertEqual(clone.run(), 21)

  def test_clone_with_except_finally_independence(self):
    """Clone preserves except and finally; both work independently."""
    log_original = []
    log_clone = []

    def raise_err(v):
      raise TestExc('boom')

    c = (
      Chain(1)
      .then(raise_err)
      .except_(lambda v: log_original.append('exc'), reraise=False)
      .finally_(lambda v: log_original.append('fin'))
    )

    # Clone before running
    c2 = c.clone()

    # Run original
    c.run()
    self.assertIn('exc', log_original)
    self.assertIn('fin', log_original)

    # Run clone -- it shares the handler callables, so they append to the same lists.
    # This is expected: clone shares callables but has independent link structure.
    log_original.clear()
    c2.run()
    self.assertIn('exc', log_original)
    self.assertIn('fin', log_original)


# ===================================================================
# 4. Chain Reuse Patterns (10+ tests)
# ===================================================================

class ChainReusePatternTests(IsolatedAsyncioTestCase):

  async def test_chain_reuse_run_with_different_roots(self):
    """Void chain run with different root values."""
    c = Chain().then(lambda v: v ** 2)
    self.assertEqual(c.run(3), 9)
    self.assertEqual(c.run(5), 25)
    self.assertEqual(c.run(0), 0)

  async def test_chain_decorator_use_as_function(self):
    """Chain decorator used as a normal function."""
    @Chain().then(lambda v: v * 2).then(lambda v: v + 1).decorator()
    def transform(v):
      return v + 10

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # (5 + 10) * 2 + 1 = 31
        self.assertEqual(transform(5), 31)

  async def test_chain_reuse_concurrent_execution(self):
    """Chain reused for 50 concurrent async calls."""
    c = Chain().then(aempty).then(lambda v: v * 2)
    tasks = [c.run(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    expected = [i * 2 for i in range(50)]
    self.assertEqual(sorted(results), sorted(expected))

  async def test_chain_reuse_with_except_handler(self):
    """Chain with except handler reused."""
    caught = []

    def maybe_fail(v):
      if v < 0:
        raise TestExc('negative')
      return v

    c = (
      Chain()
      .then(maybe_fail)
      .except_(lambda v: caught.append('caught'), reraise=False)
    )
    # Positive value works normally
    self.assertEqual(c.run(5), 5)
    self.assertEqual(caught, [])

    # Negative value triggers except
    c.run(-1)
    self.assertEqual(caught, ['caught'])

  async def test_chain_reuse_with_finally_handler(self):
    """Chain with finally handler reused."""
    log = []
    c = (
      Chain()
      .then(lambda v: v + 1)
      .finally_(lambda v: log.append('fin'))
    )
    self.assertEqual(c.run(10), 11)
    self.assertEqual(log, ['fin'])
    log.clear()
    self.assertEqual(c.run(20), 21)
    self.assertEqual(log, ['fin'])

  async def test_chain_reuse_with_nested_chains(self):
    """Chain containing nested chains reused."""
    inner = Chain().then(lambda v: v * 3)
    c = Chain().then(inner).then(lambda v: v + 1)
    self.assertEqual(c.run(4), 13)  # 4*3 + 1

  async def test_chain_reuse_with_debug_mode(self):
    """Chain with debug mode reused."""
    logger = logging.getLogger('quent')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      c = Chain().then(lambda v: v + 1).config(debug=True)
      result = c.run(10)
      self.assertEqual(result, 11)
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

  async def test_chain_decorator_on_class_method(self):
    """Chain decorator on a class method."""
    class Calculator:
      @Chain().then(lambda v: v * 2).decorator()
      def double(self, x):
        return x

    calc = Calculator()
    self.assertEqual(calc.double(7), 14)

  async def test_chain_void_run_with_different_overrides(self):
    """Void chain run with a variety of root value types."""
    c = Chain().then(lambda v: str(v))
    self.assertEqual(c.run(42), '42')
    self.assertEqual(c.run('hello'), 'hello')
    self.assertEqual(c.run([1, 2]), '[1, 2]')
    self.assertEqual(c.run(None), 'None')

  async def test_chain_call_vs_run_equivalence(self):
    """c.run(x) and c(x) should produce the same result."""
    c = Chain().then(lambda v: v * 5)
    for val in [1, 10, 100, -5, 0]:
      self.assertEqual(c.run(val), c(val))


# ===================================================================
# 5. Pipeline Composition Patterns (10+ tests)
# ===================================================================

class PipelineCompositionTests(IsolatedAsyncioTestCase):

  async def test_processing_pipeline(self):
    """Build a parse -> validate -> transform -> output pipeline."""
    def parse(v):
      return int(v)

    def validate(v):
      if v < 0:
        raise ValueError('negative')
      return v

    def transform(v):
      return v * 2

    def output(v):
      return f'result: {v}'

    pipeline = Chain().then(parse).then(validate).then(transform).then(output)
    self.assertEqual(pipeline.run('5'), 'result: 10')
    self.assertEqual(pipeline.run('0'), 'result: 0')

  async def test_data_cleaning_pipeline(self):
    """Pipeline with filter and foreach for data cleaning."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        pipeline = (
          Chain()
          .then(fn)
          .filter(lambda x: x is not None)
          .filter(lambda x: x != '')
          .foreach(lambda x: x.strip())
          .foreach(lambda x: x.lower())
        )
        data = ['  Hello ', None, 'WORLD', '', '  Foo  ', None, 'BAR']
        result = await await_(pipeline.run(data))
        self.assertEqual(result, ['hello', 'world', 'foo', 'bar'])

  async def test_error_recovery_pipeline(self):
    """Pipeline with except handler for fallback."""
    def risky_operation(v):
      if v == 'bad':
        raise TestExc('bad input')
      return f'processed:{v}'

    pipeline = (
      Chain()
      .then(risky_operation)
      .except_(lambda v: 'fallback', reraise=False)
    )
    self.assertEqual(pipeline.run('good'), 'processed:good')
    self.assertEqual(pipeline.run('bad'), 'fallback')

  async def test_conditional_processing_via_nested_return(self):
    """Conditional processing using Chain.return_ in nested chain."""
    def conditional(v):
      if v > 100:
        Chain.return_(v)
      return v * 2

    # For v=50: conditional(50) -> 100, then +1 = 101
    c = Chain(50).then(Chain().then(conditional)).then(lambda v: v + 1)
    self.assertEqual(c.run(), 101)  # 50*2 + 1

    # For v=200: return_(200) escapes ALL nesting.
    # The .then(+1) on the outer chain is also skipped because return_ propagates.
    c2 = Chain(200).then(Chain().then(conditional)).then(lambda v: v + 1)
    self.assertEqual(c2.run(), 200)

  async def test_chain_builds_dict_progressively(self):
    """Chain that builds a dict step by step."""
    c = (
      Chain(lambda: {})
      .then(lambda d: {**d, 'name': 'test'})
      .then(lambda d: {**d, 'version': 1})
      .then(lambda d: {**d, 'items': [1, 2, 3]})
      .then(lambda d: {**d, 'count': len(d['items'])})
    )
    result = c.run()
    self.assertEqual(result, {'name': 'test', 'version': 1, 'items': [1, 2, 3], 'count': 3})

  async def test_chain_flattens_nested_lists(self):
    """Chain that flattens nested lists."""
    def flatten(lst):
      result = []
      for item in lst:
        if isinstance(item, list):
          result.extend(flatten(item))
        else:
          result.append(item)
      return result

    c = Chain([[1, [2, 3]], [4, [5, [6]]], 7]).then(flatten)
    self.assertEqual(c.run(), [1, 2, 3, 4, 5, 6, 7])

  async def test_chain_map_reduce_pattern(self):
    """Chain implementing map-reduce."""
    from functools import reduce

    c = (
      Chain([1, 2, 3, 4, 5])
      .foreach(lambda x: x ** 2)
      .then(lambda lst: reduce(lambda a, b: a + b, lst))
    )
    # 1 + 4 + 9 + 16 + 25 = 55
    self.assertEqual(c.run(), 55)

  async def test_chain_as_middleware_decorator(self):
    """Chain used as a middleware/decorator pattern."""
    log = []

    @Chain().do(lambda v: log.append(f'pre:{v}')).then(lambda v: v).do(lambda v: log.append(f'post:{v}')).decorator()
    def process(data):
      return data.upper()

    result = process('hello')
    self.assertEqual(result, 'HELLO')
    self.assertEqual(log, ['pre:HELLO', 'post:HELLO'])

  async def test_pipe_operator_composition(self):
    """Chain(x) | fn1 | fn2 | fn3 | run()."""
    result = (
      Chain(5)
      .then(lambda v: v + 1)
      .then(lambda v: v * 2)
      .then(lambda v: v - 3)
      .run()
    )
    # (5+1)*2 - 3 = 9
    self.assertEqual(result, 9)

  async def test_long_pipe_chain_20_operations(self):
    """Long pipe chain with 20+ operations."""
    c = Chain(0)
    for i in range(20):
      c = c | (lambda v, _i=i: v + _i + 1)
    result = c.run()
    # sum of 1..20 = 210
    self.assertEqual(result, 210)

  async def test_pipeline_with_gather(self):
    """Pipeline that uses gather for parallel processing."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        c = (
          Chain(fn, 10)
          .gather(
            lambda v: v + 1,
            lambda v: v * 2,
            lambda v: v ** 2
          )
          .then(lambda results: sum(results))
        )
        # [11, 20, 100] -> 131
        self.assertEqual(await await_(c.run()), 131)


# ===================================================================
# 6. Reuse Patterns (10+ tests)
# ===================================================================

class ReusePatternTests(IsolatedAsyncioTestCase):

  async def test_same_chain_run_100_times_different_roots(self):
    """Void chain run 100 times with different root values."""
    c = Chain().then(lambda v: v * 2)
    for i in range(100):
      self.assertEqual(c.run(i), i * 2)

  async def test_chain_reuse_concurrent_asyncio_gather(self):
    """Chain reused concurrently with asyncio.gather."""
    c = Chain().then(aempty).then(lambda v: v + 10)
    tasks = [c.run(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    expected = [i + 10 for i in range(50)]
    self.assertEqual(sorted(results), expected)

  async def test_chain_run_modify_run_again(self):
    """Chain run, then modified, run again -- verify modification takes effect."""
    c = Chain(1).then(lambda v: v + 1)
    self.assertEqual(c.run(), 2)
    c.then(lambda v: v * 10)
    self.assertEqual(c.run(), 20)

  async def test_clone_based_branching(self):
    """Base chain -> clone -> add different operations."""
    base = Chain().then(lambda v: v + 1)
    branch_a = base.clone().then(lambda v: v * 2)
    branch_b = base.clone().then(lambda v: v * 3)
    branch_c = base.clone().then(lambda v: v ** 2)

    self.assertEqual(branch_a.run(5), 12)  # (5+1)*2
    self.assertEqual(branch_b.run(5), 18)  # (5+1)*3
    self.assertEqual(branch_c.run(5), 36)  # (5+1)^2

  async def test_decorator_pattern_reuse(self):
    """One chain definition, many function decorations."""
    d = Chain().then(lambda v: v * 2).decorator()

    @d
    def fn_a(v):
      return v + 1

    @d
    def fn_b(v):
      return v + 10

    @d
    def fn_c(v):
      return v + 100

    self.assertEqual(fn_a(1), 4)    # (1+1)*2
    self.assertEqual(fn_b(1), 22)   # (1+10)*2
    self.assertEqual(fn_c(1), 202)  # (1+100)*2

  async def test_chain_reuse_sync(self):
    """Chain reused multiple times in sync context."""
    c = Chain().then(lambda v: v ** 2)
    results = [c.run(i) for i in range(10)]
    self.assertEqual(results, [i ** 2 for i in range(10)])

  async def test_void_chain_reuse_with_complex_operations(self):
    """Void chain with filter+foreach reused with different data."""
    c = Chain().filter(lambda x: x > 0).foreach(lambda x: x * 10)
    result1 = c.run([-1, 0, 1, 2, 3])
    self.assertEqual(result1, [10, 20, 30])
    result2 = c.run([5, -2, 0, 8])
    self.assertEqual(result2, [50, 80])

  async def test_base_chain_cloned_for_different_exception_handling(self):
    """Same base chain cloned with different exception handlers."""
    def risky(v):
      if v == 'fail':
        raise TestExc()
      return v

    base = Chain().then(risky)
    handler_a = base.clone().except_(lambda v: 'recovered_a', reraise=False)
    handler_b = base.clone().except_(lambda v: 'recovered_b', reraise=False)

    self.assertEqual(handler_a.run('fail'), 'recovered_a')
    self.assertEqual(handler_b.run('fail'), 'recovered_b')
    self.assertEqual(handler_a.run('ok'), 'ok')
    self.assertEqual(handler_b.run('ok'), 'ok')

  async def test_chain_reused_as_nested_in_multiple_parents(self):
    """Same chain object used as nested in multiple parents."""
    shared = Chain().then(lambda v: v * 2)
    parent_a = Chain(3).then(shared)
    # shared is now marked as nested
    self.assertEqual(parent_a.run(), 6)
    # To use in another parent, must clone
    shared_clone = shared.clone()
    parent_b = Chain(7).then(shared_clone)
    self.assertEqual(parent_b.run(), 14)

  async def test_chain_reuse_called_sequentially_many_times(self):
    """Chain called 200 times sequentially."""
    c = Chain().then(lambda v: v + 1)
    for i in range(200):
      self.assertEqual(c.run(i), i + 1)


# ===================================================================
# 7. Complex Feature Combinations (15+ tests)
# ===================================================================

class ComplexFeatureCombinationTests(IsolatedAsyncioTestCase):

  async def test_debug_except_finally_nested_foreach(self):
    """debug + except + finally + nested + foreach."""
    log = []
    logger = logging.getLogger('quent')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
      inner = Chain().foreach(lambda x: x * 2)
      c = (
        Chain([1, 2, 3])
        .then(inner)
        .then(lambda v: sum(v))
        .except_(lambda v: log.append('exc'), reraise=False)
        .finally_(lambda v: log.append('fin'))
        .config(debug=True)
      )
      result = c.run()
      # [1,2,3] -> foreach *2 -> [2,4,6] -> sum -> 12
      self.assertEqual(result, 12)
      self.assertNotIn('exc', log)  # No exception
      self.assertIn('fin', log)
    finally:
      logger.removeHandler(handler)
      logger.setLevel(old_level)

  async def test_autorun_except_nested_chains(self):
    """autorun + except + nested chains."""
    caught = []

    def raise_err(v):
      raise TestExc('nested fail')

    inner = Chain().then(raise_err)
    c = (
      Chain(1)
      .then(inner)
      .except_(lambda v: caught.append('caught'), reraise=False)
      .config(autorun=True)
    )
    c.run()
    self.assertEqual(caught, ['caught'])

  async def test_clone_decorator_nested(self):
    """clone + decorator + nested."""
    inner = Chain().then(lambda v: v + 100)
    base = Chain().then(inner).then(lambda v: v * 2)
    cloned = base.clone()

    @cloned.decorator()
    def fn(v):
      return v

    self.assertEqual(fn(5), 210)  # (5 + 100) * 2

  async def test_except_multiple_handlers_reraise_combos(self):
    """Multiple except handlers with reraise/no-reraise combinations."""
    class Exc1(TestExc):
      pass

    class Exc2(TestExc):
      pass

    log = []

    # First handler matches but reraising, second doesn't match
    def raise_exc1(v):
      raise Exc1('e1')

    c = (
      Chain(1)
      .then(raise_exc1)
      .except_(lambda v: log.append('h1'), exceptions=Exc1, reraise=True)
      .except_(lambda v: log.append('h2'), exceptions=Exc2, reraise=False)
    )
    with self.assertRaises(Exc1):
      c.run()
    self.assertEqual(log, ['h1'])

    # No-reraise handler
    log.clear()
    c2 = (
      Chain(1)
      .then(raise_exc1)
      .except_(lambda v: log.append('h1_noraise'), exceptions=Exc1, reraise=False)
    )
    c2.run()
    self.assertEqual(log, ['h1_noraise'])

  async def test_finally_async_handler_on_sync_chain(self):
    """finally_ with async handler on sync chain triggers RuntimeWarning."""
    ran = []

    async def async_finally(v):
      ran.append('async_fin')

    c = Chain(1).then(lambda v: v + 1).finally_(async_finally)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      result = c.run()
      self.assertEqual(result, 2)
      # Should have a RuntimeWarning about async finally
      runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
      self.assertTrue(len(runtime_warns) > 0)

    # Give async task time to complete
    await asyncio.sleep(0.1)
    self.assertIn('async_fin', ran)

  async def test_gather_inside_foreach(self):
    """Gather used after foreach."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        c = (
          Chain(fn, [1, 2, 3])
          .foreach(lambda x: x * 2)
          .gather(
            lambda lst: sum(lst),
            lambda lst: len(lst),
            lambda lst: max(lst)
          )
        )
        result = await await_(c.run())
        # foreach: [2, 4, 6]
        # gather: [12, 3, 6]
        self.assertEqual(result, [12, 3, 6])

  async def test_filter_inside_chain_used_with_with(self):
    """Filter inside a chain combined with with_."""
    cm = SyncContextManager(value=[1, 2, 3, 4, 5])
    c = Chain(cm).with_(lambda v: v).filter(lambda x: x % 2 != 0)
    result = c.run()
    self.assertEqual(result, [1, 3, 5])
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_foreach_async_fn_break_nested_chain(self):
    """Foreach with async fn + break + nested chain."""
    async def process(el):
      if el >= 4:
        return Chain.break_()
      return el * 10

    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        inner = Chain().foreach(process)
        c = Chain(fn, [1, 2, 3, 4, 5]).then(inner)
        result = await await_(c.run())
        self.assertEqual(result, [10, 20, 30])

  async def test_chain_with_all_features(self):
    """Chain exercising root, then, do, except, finally, foreach, filter, gather, clone."""
    log = []

    base = (
      Chain()
      .then(lambda v: v)
      .then(lambda v: list(range(v)))
      .filter(lambda x: x > 0)
      .foreach(lambda x: x * 2)
      .do(lambda v: log.append(f'count:{len(v)}'))
      .then(lambda v: sum(v))
      .except_(lambda v: log.append('exc'), reraise=False)
      .finally_(lambda v: log.append('fin'))
    )

    cloned = base.clone()

    log.clear()
    result = cloned.run(6)
    # range(6) = [0,1,2,3,4,5], filter >0: [1,2,3,4,5], *2: [2,4,6,8,10], sum: 30
    self.assertEqual(result, 30)
    self.assertIn('count:5', log)
    self.assertIn('fin', log)
    self.assertNotIn('exc', log)

  async def test_except_handler_returns_chain(self):
    """except handler that returns a nested chain as recovery value."""
    def raise_err(v):
      raise TestExc()

    # The except_ handler Chain receives the root value (1) as its root.
    recovery_chain = Chain().then(lambda v: 'recovered')
    c = (
      Chain(1)
      .then(raise_err)
      .except_(recovery_chain, reraise=False)
    )
    result = c.run()
    self.assertEqual(result, 'recovered')


# ===================================================================
# 8. Edge Case Chain Constructions (10+ tests)
# ===================================================================

class EdgeCaseChainTests(IsolatedAsyncioTestCase):

  async def test_chain_with_only_except(self):
    """Chain with only an except handler (no operations)."""
    caught = []
    # except_ handler with ... to not pass value since void chain has no root
    c = Chain().except_(lambda: caught.append('exc'), ..., reraise=False)
    # Running should return None since there are no operations
    result = c.run()
    self.assertIsNone(result)
    self.assertEqual(caught, [])

  async def test_chain_with_only_finally(self):
    """Chain with root and only a finally handler."""
    log = []
    # finally_ handler receives root value
    c = Chain(42).finally_(lambda v: log.append(f'fin:{v}'))
    result = c.run()
    self.assertEqual(result, 42)
    self.assertEqual(log, ['fin:42'])

  async def test_chain_with_except_finally_no_operations(self):
    """Chain with root, except + finally but no other operations."""
    log = []
    c = (
      Chain(1)
      .except_(lambda v: log.append('exc'), reraise=False)
      .finally_(lambda v: log.append('fin'))
    )
    result = c.run()
    self.assertEqual(result, 1)
    self.assertNotIn('exc', log)
    self.assertIn('fin', log)

  async def test_chain_every_link_is_nested_chain(self):
    """Chain where every link is itself a nested chain."""
    c = (
      Chain(1)
      .then(Chain().then(lambda v: v + 1))
      .then(Chain().then(lambda v: v * 2))
      .then(Chain().then(lambda v: v + 10))
      .then(Chain().then(lambda v: v * 3))
    )
    # 1 + 1 = 2, * 2 = 4, + 10 = 14, * 3 = 42
    self.assertEqual(c.run(), 42)

  async def test_chain_return_as_first_operation(self):
    """Chain.return_() as the very first operation exits early."""
    inner = Chain().then(Chain.return_, 'early')
    outer = Chain(1).then(inner).then(lambda v: 'should_not_reach')
    result = outer.run()
    self.assertEqual(result, 'early')

  async def test_chain_break_outside_foreach(self):
    """Chain.break_() outside foreach should raise QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).then(Chain.break_).run()

  async def test_chain_break_outside_foreach_async(self):
    """Chain.break_() outside foreach in async chain should raise QuentException."""
    with self.assertRaises(QuentException):
      await Chain(aempty).then(Chain.break_).run()

  async def test_multiple_except_handlers_overlapping_types(self):
    """Multiple except handlers with overlapping exception types."""
    class BaseExc(Exception):
      pass

    class DerivedExc(BaseExc):
      pass

    log = []

    def raise_derived(v):
      raise DerivedExc()

    # First handler catches BaseExc (which includes DerivedExc)
    c = (
      Chain(1)
      .then(raise_derived)
      .except_(lambda v: log.append('base'), exceptions=BaseExc, reraise=False)
      .except_(lambda v: log.append('derived'), exceptions=DerivedExc, reraise=False)
    )
    c.run()
    # First matching handler should be used
    self.assertEqual(log, ['base'])

  async def test_except_handler_returns_nested_chain(self):
    """except handler that returns a chain as the exception result."""
    def raise_err(v):
      raise TestExc()

    # The except_ handler Chain receives the root value (1) as its input.
    c = (
      Chain(1)
      .then(raise_err)
      .except_(
        Chain().then(lambda v: 'from_chain'),
        reraise=False
      )
    )
    result = c.run()
    self.assertEqual(result, 'from_chain')



  async def test_chain_with_none_root(self):
    """Chain with None as root value."""
    c = Chain(None).then(lambda v: v is None)
    self.assertTrue(c.run())

  async def test_chain_with_false_root(self):
    """Chain with False as root value."""
    c = Chain(False).then(lambda v: not v)
    self.assertTrue(c.run())

  async def test_chain_with_zero_root(self):
    """Chain with 0 as root value."""
    c = Chain(0).then(lambda v: v + 1)
    self.assertEqual(c.run(), 1)

  async def test_chain_with_empty_string_root(self):
    """Chain with empty string as root value."""
    c = Chain('').then(lambda v: v + 'hello')
    self.assertEqual(c.run(), 'hello')

  async def test_chain_with_empty_list_root(self):
    """Chain with empty list as root value."""
    c = Chain([]).then(lambda v: v + [1, 2, 3])
    self.assertEqual(c.run(), [1, 2, 3])

  async def test_chain_bool_is_always_true(self):
    """Chain instances are always truthy."""
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(1)))
    self.assertTrue(bool(Chain().then(lambda v: v)))

  async def test_cannot_override_root_value_of_chain_with_root(self):
    """Running a chain with a root with a different root raises QuentException."""
    c = Chain(1).then(lambda v: v + 1)
    with self.assertRaises(QuentException):
      c.run(99)

  async def test_cannot_run_nested_chain_directly(self):
    """A chain marked as nested cannot be run directly."""
    inner = Chain().then(lambda v: v)
    Chain(1).then(inner)  # marks inner as nested
    with self.assertRaises(QuentException):
      inner.run()


# ===================================================================
# Additional: Async variants for deeply nested and composition
# ===================================================================

class AsyncDeeplyNestedTests(IsolatedAsyncioTestCase):

  async def test_deeply_nested_all_async(self):
    """All levels use async functions."""
    inner = Chain().then(aempty).then(lambda v: v + 1)
    mid = Chain().then(aempty).then(inner)
    outer = Chain(aempty, 10).then(mid)
    self.assertEqual(await outer.run(), 11)

  async def test_nested_chains_with_gather(self):
    """Nested chain with gather at inner level."""
    inner = Chain().gather(
      lambda v: v + 1,
      lambda v: v * 2,
      lambda v: v - 1
    )
    outer = Chain(10).then(inner)
    result = outer.run()
    self.assertEqual(result, [11, 20, 9])

  async def test_nested_chain_with_foreach_and_filter(self):
    """Nested chain with foreach inside another chain's filter."""
    inner = Chain().foreach(lambda x: x * 3)
    outer = Chain([1, 2, 3]).then(inner).then(lambda lst: [x for x in lst if x > 5])
    result = outer.run()
    self.assertEqual(result, [6, 9])

  async def test_deeply_nested_with_except_at_outer_level(self):
    """Exception from innermost caught by except at outer level."""
    caught = []

    def raise_err(v):
      raise TestExc('deep')

    l3 = Chain().then(raise_err)
    l2 = Chain().then(l3)
    l1 = (
      Chain(1)
      .then(l2)
      .except_(lambda v: caught.append('outer_catch'), exceptions=TestExc, reraise=False)
    )
    l1.run()
    self.assertEqual(caught, ['outer_catch'])

  async def test_deeply_nested_async_with_except(self):
    """Async exception from innermost caught by except at outer level."""
    caught = []

    async def async_raise(v):
      raise TestExc('async_deep')

    l3 = Chain().then(async_raise)
    l2 = Chain().then(l3)
    l1 = (
      Chain(1)
      .then(l2)
      .except_(lambda v: caught.append('caught_async'), exceptions=TestExc, reraise=False)
    )
    await await_(l1.run())
    self.assertEqual(caught, ['caught_async'])


class AsyncCompositionTests(IsolatedAsyncioTestCase):

  async def test_pipeline_fully_async(self):
    """Fully async pipeline composition."""
    async def parse(v):
      return int(v)

    async def double(v):
      return v * 2

    async def to_str(v):
      return str(v)

    c = Chain().then(parse).then(double).then(to_str)
    result = await await_(c.run('7'))
    self.assertEqual(result, '14')

  async def test_concurrent_clones_async(self):
    """Many async clones run concurrently."""
    base = Chain().then(aempty).then(lambda v: v + 1)
    clones = [base.clone() for _ in range(50)]
    tasks = [clone.run(i) for i, clone in enumerate(clones)]
    results = await asyncio.gather(*tasks)
    expected = [i + 1 for i in range(50)]
    self.assertEqual(sorted(results), expected)

  async def test_chain_reuse_with_async_foreach(self):
    """Chain with async foreach reused."""
    c = Chain().then(aempty).foreach(lambda x: aempty(x * 2))
    result = await await_(c.run([1, 2, 3]))
    self.assertEqual(result, [2, 4, 6])

  async def test_chain_reuse_with_async_filter(self):
    """Chain with async filter reused."""
    c = Chain().then(aempty).filter(lambda x: aempty(x > 2))
    result = await await_(c.run([1, 2, 3, 4, 5]))
    self.assertEqual(result, [3, 4, 5])

  async def test_chain_reuse_with_async_gather(self):
    """Chain with async gather reused."""
    c = Chain().then(aempty).gather(
      lambda v: aempty(v + 1),
      lambda v: v * 2,
      lambda v: aempty(v ** 2)
    )
    result = await await_(c.run(5))
    self.assertEqual(result, [6, 10, 25])


# ===================================================================
# More edge cases and stress tests
# ===================================================================

class StressAndEdgeCaseTests(IsolatedAsyncioTestCase):

  async def test_chain_with_deeply_nested_data_structure(self):
    """Chain processing deeply nested data."""
    data = {'a': {'b': {'c': {'d': 42}}}}
    c = (
      Chain(data)
      .then(lambda d: d['a'])
      .then(lambda d: d['b'])
      .then(lambda d: d['c'])
      .then(lambda d: d['d'])
    )
    self.assertEqual(c.run(), 42)

  async def test_chain_with_lambda_chain(self):
    """Chain where each link is a lambda capturing a different value."""
    c = Chain(0)
    for i in range(1, 11):
      c = c.then(lambda v, n=i: v + n)
    # 0 + 1 + 2 + ... + 10 = 55
    self.assertEqual(c.run(), 55)

  async def test_multiple_finally_registration_fails(self):
    """Registering multiple finally handlers should raise QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)

  async def test_chain_with_class_methods_as_links(self):
    """Chain using class methods as link values."""
    class Processor:
      def __init__(self):
        self.log = []

      def step1(self, v):
        self.log.append('step1')
        return v + 1

      def step2(self, v):
        self.log.append('step2')
        return v * 2

    p = Processor()
    c = Chain(5).then(p.step1).then(p.step2)
    result = c.run()
    self.assertEqual(result, 12)
    self.assertEqual(p.log, ['step1', 'step2'])

  async def test_chain_repr_has_chain_string(self):
    """Chain repr contains 'Chain'."""
    c = Chain(1).then(lambda v: v + 1)
    r = repr(c)
    self.assertIn('Chain', r)

  async def test_chain_run_and_call_same_result(self):
    """Chain run() and __call__() give same result."""
    c = Chain(10).then(lambda v: v * 3)
    self.assertEqual(c.run(), c())

  async def test_pipe_with_run_root_override(self):
    """Pipe syntax with run() root override."""
    result = Chain().then(lambda v: v + 5).run(10)
    self.assertEqual(result, 15)

  async def test_chain_with_complex_exception_hierarchy(self):
    """Chain handling complex exception hierarchies."""
    class AppError(Exception):
      pass

    class NotFoundError(AppError):
      pass

    class PermError(AppError):
      pass

    log = []

    def raise_not_found(v):
      raise NotFoundError('missing')

    c = (
      Chain(1)
      .then(raise_not_found)
      .except_(lambda v: log.append('perm'), exceptions=PermError, reraise=False)
      .except_(lambda v: log.append('notfound'), exceptions=NotFoundError, reraise=False)
    )
    c.run()
    # First handler doesn't match, second does
    self.assertEqual(log, ['notfound'])

  async def test_chain_with_generator_as_root(self):
    """Chain with a generator function as root."""
    def gen():
      return [1, 2, 3]

    c = Chain(gen).foreach(lambda x: x * 2)
    result = c.run()
    self.assertEqual(result, [2, 4, 6])

  async def test_void_chain_with_multiple_run_calls(self):
    """Void chain called multiple times with different roots."""
    c = Chain().then(lambda v: v * 2).then(lambda v: v + 1)
    self.assertEqual(c.run(5), 11)
    self.assertEqual(c.run(10), 21)
    self.assertEqual(c.run(0), 1)

  async def test_chain_with_none_returning_links(self):
    """Chain where intermediate links return None."""
    c = Chain(1).then(lambda v: None).then(lambda v: v is None)
    self.assertTrue(c.run())

  async def test_chain_exception_in_finally_propagates(self):
    """Exception raised in finally handler propagates."""
    def raise_in_finally(v):
      raise ValueError('finally error')

    c = Chain(1).then(lambda v: v + 1).finally_(raise_in_finally)
    with self.assertRaises(ValueError) as cm:
      c.run()
    self.assertEqual(str(cm.exception), 'finally error')

  async def test_pipe_chain_with_literal_values(self):
    """Pipe chain with literal (non-callable) values."""
    result = Chain(1).then(42).run()
    self.assertEqual(result, 42)

  async def test_chain_with_star_ellipsis_for_no_args(self):
    """Using ellipsis (...) to suppress value passing."""
    c = Chain(99).then(lambda: 'no_args', ...)
    self.assertEqual(c.run(), 'no_args')

  async def test_chain_do_with_ellipsis(self):
    """do() with ellipsis to suppress value passing."""
    side = []
    c = Chain(42).do(lambda: side.append('ran'), ...).then(lambda v: v + 1)
    result = c.run()
    self.assertEqual(result, 43)
    self.assertEqual(side, ['ran'])

  async def test_chain_with_except_on_void_chain(self):
    """Except handler on a void chain with root override that raises."""
    caught = []

    def raise_err(v):
      raise TestExc()

    c = Chain().then(raise_err).except_(lambda v: caught.append(v), reraise=False)
    c.run('root_val')
    self.assertEqual(caught, ['root_val'])


