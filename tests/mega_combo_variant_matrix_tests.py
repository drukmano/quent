"""Variant x Feature matrix tests.

Systematically tests every Chain feature across ALL chain variants:
Chain, run (pipe terminator), clone, decorator.

Sections:
  1. run (pipe terminator) with every feature
  2. Chain reuse with every feature
  3. clone() with every feature
  4. decorator() with every feature
  5. Cross-variant nesting
  6. Variant-specific edge cases
  7. Async variants for each chain type
"""

import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleCM:
  """Sync context manager that yields a predetermined value."""
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
  """Async context manager that yields a predetermined value."""
  def __init__(self, value='actx'):
    self._value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self._value

  async def __aexit__(self, *args):
    self.exited = True
    return False


def raise_test_exc(v=None):
  raise TestExc('boom')


# ===================================================================
# Section 2: Chain reuse with every feature
# ===================================================================

class ChainReuseThenTests(IsolatedAsyncioTestCase):
  """Chain reuse with then operations."""

  async def test_chain_reuse_basic(self):
    c = Chain(10).then(lambda v: v * 2)
    self.assertEqual(c(), 20)
    self.assertEqual(c.run(), 20)

  async def test_chain_reuse_with_root_override(self):
    c = Chain().then(lambda v: v ** 2)
    self.assertEqual(c(3), 9)
    self.assertEqual(c(4), 16)
    self.assertEqual(c(5), 25)

  async def test_chain_reuse_multiple_calls(self):
    c = Chain(0).then(lambda v: v + 1)
    results = [c() for _ in range(5)]
    self.assertEqual(results, [1, 1, 1, 1, 1])


class ChainReuseExceptTests(IsolatedAsyncioTestCase):
  """Chain reuse with except_."""

  async def test_chain_reuse_except_noraise(self):
    c = Chain(10).then(raise_test_exc).except_(lambda v: 'recovered', reraise=False)
    self.assertEqual(c(), 'recovered')

  async def test_chain_reuse_except_reraise(self):
    c = Chain(10).then(raise_test_exc).except_(lambda v: None)
    with self.assertRaises(TestExc):
      c()

  async def test_chain_reuse_except_reuse(self):
    c = Chain(10).then(raise_test_exc).except_(lambda v: 42, reraise=False)
    self.assertEqual(c(), 42)
    self.assertEqual(c(), 42)


class ChainReuseFinallyTests(IsolatedAsyncioTestCase):
  """Chain reuse with finally_."""

  async def test_chain_reuse_finally_runs(self):
    ran = []
    c = Chain(10).finally_(lambda v: ran.append('fin'))
    self.assertEqual(c(), 10)
    self.assertEqual(ran, ['fin'])

  async def test_chain_reuse_finally_on_error(self):
    ran = []
    c = Chain(raise_test_exc).finally_(lambda: ran.append('fin'), ...)
    try:
      c()
    except TestExc:
      pass
    self.assertEqual(ran, ['fin'])


class ChainReuseForeachTests(IsolatedAsyncioTestCase):
  """Chain reuse with foreach."""

  async def test_chain_reuse_foreach(self):
    c = Chain([1, 2, 3]).foreach(lambda x: x * 10)
    self.assertEqual(c(), [10, 20, 30])

  async def test_chain_reuse_foreach_reuse(self):
    c = Chain().foreach(lambda x: x * 2)
    self.assertEqual(c([1, 2]), [2, 4])
    self.assertEqual(c([10, 20]), [20, 40])


class ChainReuseFilterTests(IsolatedAsyncioTestCase):
  """Chain reuse with filter."""

  async def test_chain_reuse_filter(self):
    c = Chain([1, 2, 3, 4, 5]).filter(lambda x: x > 3)
    self.assertEqual(c(), [4, 5])


class ChainReuseGatherTests(IsolatedAsyncioTestCase):
  """Chain reuse with gather."""

  async def test_chain_reuse_gather(self):
    c = Chain(10).gather(lambda v: v * 2, lambda v: v + 5)
    self.assertEqual(c(), [20, 15])

  async def test_chain_reuse_gather_reuse(self):
    c = Chain().gather(lambda v: v * 2, lambda v: v + 1)
    self.assertEqual(c(5), [10, 6])
    self.assertEqual(c(3), [6, 4])


class ChainReuseWithCMTests(IsolatedAsyncioTestCase):
  """Chain reuse with with_ (context manager)."""

  async def test_chain_reuse_with_cm(self):
    c = Chain(SimpleCM(value=42)).with_(lambda ctx: ctx + 8)
    self.assertEqual(c(), 50)


class ChainReuseNestedTests(IsolatedAsyncioTestCase):
  """Chain nested inside another chain (primary use case)."""

  async def test_chain_nested_in_chain(self):
    doubler = Chain().then(lambda v: v * 2)
    result = Chain(5).then(doubler).run()
    self.assertEqual(result, 10)

  async def test_chain_nested_in_chain_multiple(self):
    add1 = Chain().then(lambda v: v + 1)
    mul3 = Chain().then(lambda v: v * 3)
    result = Chain(2).then(add1).then(mul3).run()
    self.assertEqual(result, 9)  # (2+1)*3 = 9


class ChainReuseBoolTests(TestCase):
  """Chain __bool__ behavior."""

  def test_chain_is_truthy(self):
    c = Chain(1)
    self.assertTrue(bool(c))


# ===================================================================
# Section 3: clone() with every feature
# ===================================================================

class CloneThenTests(TestCase):
  """Clone with then, verify independence."""

  def test_clone_add_more_to_clone(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(lambda v: v * 100)
    self.assertEqual(c.run(), 2)
    self.assertEqual(c2.run(), 200)

  def test_clone_add_more_to_original(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    self.assertEqual(c.run(), 200)
    self.assertEqual(c2.run(), 2)


class CloneExceptTests(TestCase):
  """Clone with except_, verify error handling works independently."""

  def test_clone_except_works(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc'), reraise=False)
    c2 = c.clone()
    ran.clear()
    c2.run()
    self.assertEqual(ran, ['exc'])

  def test_clone_except_original_unaffected(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc'), reraise=False)
    c2 = c.clone()
    ran.clear()
    c.run()
    self.assertEqual(ran, ['exc'])
    ran.clear()
    c2.run()
    self.assertEqual(ran, ['exc'])


class CloneFinallyTests(TestCase):
  """Clone with finally_, verify cleanup works independently."""

  def test_clone_finally_works(self):
    ran = []
    c = Chain(10).finally_(lambda v: ran.append('fin'))
    c2 = c.clone()
    ran.clear()
    c2.run()
    self.assertEqual(ran, ['fin'])

  def test_clone_finally_on_error(self):
    ran = []
    c = Chain(raise_test_exc).finally_(lambda: ran.append('fin'), ...)
    c2 = c.clone()
    ran.clear()
    try:
      c2.run()
    except TestExc:
      pass
    self.assertEqual(ran, ['fin'])


class CloneForeachTests(TestCase):
  """Clone with foreach."""

  def test_clone_foreach(self):
    c = Chain([1, 2, 3]).foreach(lambda x: x * 10)
    c2 = c.clone()
    self.assertEqual(c2.run(), [10, 20, 30])

  def test_clone_foreach_add_filter_to_clone(self):
    c = Chain([1, 2, 3, 4]).foreach(lambda x: x * 10)
    c2 = c.clone()
    c2.then(lambda lst: [x for x in lst if x > 20])
    self.assertEqual(c.run(), [10, 20, 30, 40])
    self.assertEqual(c2.run(), [30, 40])


class CloneWithCMTests(TestCase):
  """Clone with with_."""

  def test_clone_with_cm(self):
    c = Chain(SimpleCM(value=42)).with_(lambda ctx: ctx + 8)
    c2 = c.clone()
    self.assertEqual(c2.run(), 50)


class CloneGatherTests(TestCase):
  """Clone with gather."""

  def test_clone_gather(self):
    c = Chain(10).gather(lambda v: v * 2, lambda v: v + 5)
    c2 = c.clone()
    self.assertEqual(c2.run(), [20, 15])
class CloneConfigTests(TestCase):
  """Clone preserves config (debug, autorun)."""


  def test_clone_preserves_autorun(self):
    c = Chain(10).config(autorun=True)
    c2 = c.clone()
    self.assertEqual(c2.run(), 10)


class CloneMultipleAndDeepTests(TestCase):
  """Multiple clones and clone of clone."""

  def test_multiple_clones_same_original(self):
    c = Chain(3).then(lambda v: v + 1)
    clones = [c.clone() for _ in range(5)]
    for clone in clones:
      self.assertEqual(clone.run(), 4)
    c.then(lambda v: v * 100)
    for clone in clones:
      self.assertEqual(clone.run(), 4)

  def test_clone_of_clone(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    c2.then(lambda v: v * 10)
    c3.then(lambda v: v * 100)
    self.assertEqual(c.run(), 2)
    self.assertEqual(c2.run(), 20)
    self.assertEqual(c3.run(), 200)


# ===================================================================
# Section 4: decorator() with every feature
# ===================================================================

class DecoratorThenTests(IsolatedAsyncioTestCase):
  """decorator() with then."""

  async def test_decorator_basic(self):
    @Chain().then(lambda v: v * 2).decorator()
    def double(x):
      return x
    self.assertEqual(double(5), 10)

  async def test_decorator_multiple_then(self):
    @Chain().then(lambda v: v * 2).then(lambda v: v + 1).decorator()
    def fn(x):
      return x
    self.assertEqual(fn(3), 7)  # 3*2+1 = 7

  async def test_decorator_with_args(self):
    @Chain().then(lambda v: v).decorator()
    def add(a, b):
      return a + b
    self.assertEqual(add(3, 7), 10)


class DecoratorExceptTests(IsolatedAsyncioTestCase):
  """decorator() with except_."""

  async def test_decorator_except_noraise(self):
    @Chain().then(raise_test_exc).except_(lambda v: 'fallback', reraise=False).decorator()
    def fn(x):
      return x
    self.assertEqual(fn(5), 'fallback')

  async def test_decorator_except_reraise(self):
    @Chain().then(raise_test_exc).except_(lambda v: None).decorator()
    def fn(x):
      return x
    with self.assertRaises(TestExc):
      fn(5)


class DecoratorFinallyTests(IsolatedAsyncioTestCase):
  """decorator() with finally_."""

  async def test_decorator_finally(self):
    ran = []

    @Chain().then(lambda v: v * 2).finally_(lambda v: ran.append('fin')).decorator()
    def fn(x):
      return x
    self.assertEqual(fn(5), 10)
    self.assertEqual(ran, ['fin'])


class DecoratorForeachTests(IsolatedAsyncioTestCase):
  """decorator() with foreach (decorated fn returns list)."""

  async def test_decorator_foreach(self):
    @Chain().foreach(lambda x: x * 10).decorator()
    def get_list():
      return [1, 2, 3]
    self.assertEqual(get_list(), [10, 20, 30])
class DecoratorReuseTests(IsolatedAsyncioTestCase):
  """decorator() reuse."""

  async def test_decorator_reuse(self):
    @Chain().then(lambda v: v + 1).decorator()
    def inc(x):
      return x
    self.assertEqual(inc(1), 2)
    self.assertEqual(inc(10), 11)
    self.assertEqual(inc(100), 101)


class DecoratorAsyncTests(IsolatedAsyncioTestCase):
  """decorator() with async decorated function."""

  async def test_decorator_async_fn(self):
    @Chain().then(lambda v: v * 5).decorator()
    async def fn(x):
      return x + 1
    self.assertEqual(await await_(fn(2)), 15)  # (2+1)*5 = 15


# ===================================================================
# Section 5: Cross-variant nesting
# ===================================================================

class CrossVariantNestingTests(IsolatedAsyncioTestCase):
  """Test nesting one variant inside another."""


  async def test_nested_chain_three_levels(self):
    inner = Chain().then(lambda v: v + 1)
    middle = Chain().then(inner).then(lambda v: v * 2)
    result = Chain(3).then(middle).run()
    self.assertEqual(result, 8)  # (3+1)*2 = 8

  async def test_clone_chain_with_nested(self):
    inner = Chain().then(lambda v: v * 10)
    c = Chain(2).then(inner)
    c2 = c.clone()
    c.then(lambda v: v + 1)
    self.assertEqual(c.run(), 21)   # 2*10+1 = 21
    self.assertEqual(c2.run(), 20)  # 2*10 = 20

  async def test_decorator_wrapping_chain_with_nested(self):
    inner = Chain().then(lambda v: v + 10)

    @Chain().then(inner).then(lambda v: v * 2).decorator()
    def fn(x):
      return x
    self.assertEqual(fn(5), 30)  # (5+10)*2 = 30


# ===================================================================
# Section 6: Variant-specific edge cases
# ===================================================================

class VariantEdgeCasesTests(IsolatedAsyncioTestCase):
  """Edge cases specific to each variant."""


  async def test_run_no_links_just_root(self):
    result = Chain(42).run()
    self.assertEqual(result, 42)

  async def test_void_chain_returns_none(self):
    result = Chain().run()
    self.assertIsNone(result)


  async def test_chain_no_args_vs_with_args(self):
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c(5), 10)
    # No args on a void chain: v is Null at link time, lambda receives Null
    # Actually the chain has no root, so it calls lambda with no arg
    # lambda v: v*2 with no arg -> error. Let's test with a rooted chain.
    c2 = Chain(3).then(lambda v: v * 2)
    self.assertEqual(c2(), 6)

  async def test_chain_bool_is_true(self):
    c = Chain()
    self.assertTrue(bool(c))

  async def test_chain_call_is_run(self):
    result = Chain(42)()
    self.assertEqual(result, 42)

  async def test_clone_vs_original_independence(self):
    c = Chain(5).then(lambda v: v * 2)
    c2 = c.clone()
    # Both should produce the same result
    self.assertEqual(c(), 10)
    self.assertEqual(c2(), 10)
    # Now modify original, clone should still work
    c.then(lambda v: v + 100)
    self.assertEqual(c.run(), 110)
    # c2 is from the clone, independent
    self.assertEqual(c2(), 10)


  async def test_chain_then_val_vs_chain_val(self):
    """Chain().then(val) vs Chain(val) - both should return val."""
    c1 = Chain().then(10)
    c2 = Chain(10)
    self.assertEqual(c1(), 10)
    self.assertEqual(c2(), 10)


# ===================================================================
# Section 7: Async variants for each chain type
# ===================================================================

class AsyncChainTests(IsolatedAsyncioTestCase):
  """Async behavior with Chain."""

  async def test_async_then(self):
    async def double(v):
      return v * 2
    result = await Chain(5).then(double).run()
    self.assertEqual(result, 10)

  async def test_async_foreach(self):
    async def transform(x):
      return x * 10
    result = await Chain([1, 2, 3]).foreach(transform).run()
    self.assertEqual(result, [10, 20, 30])

  async def test_async_with_cm(self):
    cm = AsyncCM(value='hello')
    result = await Chain(cm).with_(lambda ctx: ctx.upper()).run()
    self.assertEqual(result, 'HELLO')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_gather(self):
    async def double(v):
      return v * 2
    result = await Chain(5).gather(double, lambda v: v + 1).run()
    self.assertEqual(result, [10, 6])

  async def test_sync_to_async_transition(self):
    """Sync root transitions to async in the middle."""
    async def async_add(v):
      return v + 10
    result = await Chain(5).then(lambda v: v * 2).then(async_add).run()
    self.assertEqual(result, 20)  # 5*2=10, 10+10=20
class AsyncChainReuseTests(IsolatedAsyncioTestCase):
  """Async behavior with chain reuse."""

  async def test_async_chain_reuse_then(self):
    async def double(v):
      return v * 2
    c = Chain(5).then(double)
    self.assertEqual(await await_(c()), 10)

  async def test_async_chain_reuse(self):
    async def inc(v):
      return v + 1
    c = Chain().then(inc)
    self.assertEqual(await await_(c(1)), 2)
    self.assertEqual(await await_(c(10)), 11)

  async def test_async_chain_reuse_except(self):
    async def fail(v):
      raise TestExc('async boom')
    c = Chain(10).then(fail).except_(lambda v: 'caught', reraise=False)
    self.assertEqual(await await_(c()), 'caught')

class AsyncCloneTests(IsolatedAsyncioTestCase):
  """Async behavior with clone."""

  async def test_async_clone(self):
    async def add10(v):
      return v + 10
    c = Chain(5).then(add10)
    c2 = c.clone()
    self.assertEqual(await await_(c.run()), 15)
    self.assertEqual(await await_(c2.run()), 15)

  async def test_async_clone_independence(self):
    async def add10(v):
      return v + 10
    c = Chain(5).then(add10)
    c2 = c.clone()
    c2.then(lambda v: v * 2)
    self.assertEqual(await await_(c.run()), 15)
    self.assertEqual(await await_(c2.run()), 30)


class AsyncDecoratorTests(IsolatedAsyncioTestCase):
  """Async behavior with decorator."""

  async def test_async_decorator_chain(self):
    async def add10(v):
      return v + 10

    @Chain().then(add10).decorator()
    def fn(x):
      return x * 2
    self.assertEqual(await await_(fn(3)), 16)  # 3*2=6, 6+10=16

  async def test_async_decorator_async_fn(self):
    @Chain().then(lambda v: v * 5).decorator()
    async def fn(x):
      return x + 1
    self.assertEqual(await await_(fn(2)), 15)  # (2+1)*5=15


# ===================================================================
# Additional coverage: iterate, to_thread, debug
# ===================================================================

class IterateTests(IsolatedAsyncioTestCase):
  """iterate() returns a generator."""

  async def test_chain_iterate_sync(self):
    results = []
    for i in Chain(lambda: [1, 2, 3]).iterate(lambda i: i * 2):
      results.append(i)
    self.assertEqual(results, [2, 4, 6])

  async def test_chain_iterate_no_fn(self):
    results = []
    for i in Chain(lambda: [10, 20]).iterate():
      results.append(i)
    self.assertEqual(results, [10, 20])

  async def test_chain_iterate_async(self):
    results = []
    async for i in Chain(lambda: [1, 2, 3]).iterate(lambda i: aempty(i * 2)):
      results.append(i)
    self.assertEqual(results, [2, 4, 6])
class DebugTests(IsolatedAsyncioTestCase):
  """debug mode tests."""

  async def test_debug_chain(self):
    c = Chain(10).config(debug=True).then(lambda v: v + 1)
    result = c.run()
    self.assertEqual(result, 11)


class ChainReprTests(TestCase):
  """Chain __repr__ returns a string."""

  def test_chain_repr(self):
    c = Chain(10).then(lambda v: v + 1)
    r = repr(c)
    self.assertIsInstance(r, str)
    self.assertIn('Chain', r)


class ChainExceptFinallyComboTests(IsolatedAsyncioTestCase):
  """except_ + finally_ together across variants."""

  async def test_chain_except_finally_on_error(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc')).finally_(lambda v: ran.append('fin')).run()
    except TestExc:
      pass
    self.assertIn('exc', ran)
    self.assertIn('fin', ran)


  async def test_chain_except_noraise_finally(self):
    ran = []
    result = Chain(10).then(raise_test_exc).except_(lambda v: 'recovered', reraise=False).finally_(lambda v: ran.append('fin')).run()
    self.assertEqual(result, 'recovered')
    self.assertEqual(ran, ['fin'])


class ChainReuseExceptFinallyComboTests(IsolatedAsyncioTestCase):
  """Chain reuse except_ + finally_ combo."""

  async def test_chain_reuse_except_finally_on_error(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc')).finally_(lambda v: ran.append('fin'))
    try:
      c()
    except TestExc:
      pass
    self.assertIn('exc', ran)
    self.assertIn('fin', ran)

  async def test_chain_reuse_except_noraise_finally(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: 'ok', reraise=False).finally_(lambda v: ran.append('fin'))
    self.assertEqual(c(), 'ok')
    self.assertEqual(ran, ['fin'])


class CloneExceptFinallyComboTests(TestCase):
  """Clone except_ + finally_ combo."""

  def test_clone_except_finally(self):
    ran = []
    c = Chain(10).then(raise_test_exc).except_(lambda v: ran.append('exc'), reraise=False).finally_(lambda v: ran.append('fin'))
    c2 = c.clone()
    ran.clear()
    c2.run()
    self.assertIn('exc', ran)
    self.assertIn('fin', ran)


class DecoratorExceptFinallyComboTests(IsolatedAsyncioTestCase):
  """Decorator except_ + finally_ combo."""

  async def test_decorator_except_finally(self):
    ran = []

    @Chain().then(raise_test_exc).except_(lambda v: 'recovered', reraise=False).finally_(lambda v: ran.append('fin')).decorator()
    def fn(x):
      return x
    result = fn(5)
    self.assertEqual(result, 'recovered')
    self.assertEqual(ran, ['fin'])


class AsyncExceptFinallyComboTests(IsolatedAsyncioTestCase):
  """Async except_ + finally_ combo."""

  async def test_async_chain_except_finally(self):
    ran = []

    async def fail(v):
      raise TestExc('async')

    result = await Chain(10).then(fail).except_(lambda v: 'caught', reraise=False).finally_(lambda v: ran.append('fin')).run()
    self.assertEqual(result, 'caught')
    self.assertEqual(ran, ['fin'])


class GatherAsyncTests(IsolatedAsyncioTestCase):
  """Gather with async functions."""

  async def test_chain_gather_async(self):
    async def double(v):
      return v * 2

    async def triple(v):
      return v * 3
    result = await Chain(5).gather(double, triple).run()
    self.assertEqual(result, [10, 15])


class FilterAsyncTests(IsolatedAsyncioTestCase):
  """filter with async."""

  async def test_chain_filter_async(self):
    async def pred(x):
      return x > 2
    result = await Chain([1, 2, 3, 4]).filter(pred).run()
    self.assertEqual(result, [3, 4])


class WithAsyncTests(IsolatedAsyncioTestCase):
  """with_ with async context manager."""

  async def test_chain_with_async_cm(self):
    cm = AsyncCM(value=42)
    result = await Chain(cm).with_(lambda ctx: ctx + 8).run()
    self.assertEqual(result, 50)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


class ChainCallableVsRunTests(IsolatedAsyncioTestCase):
  """Chain() vs Chain.run() equivalence."""

  async def test_call_vs_run(self):
    c = Chain(10).then(lambda v: v + 5)
    self.assertEqual(c(), c.run())

  async def test_call_vs_run_with_override(self):
    c = Chain().then(lambda v: v * 2)
    self.assertEqual(c(7), c.run(7))


class CloneNotNestedTests(TestCase):
  """Clone should not be marked as nested."""

  def test_clone_is_not_nested(self):
    inner = Chain(1).then(lambda v: v + 1)
    Chain(0).then(inner)  # marks inner as nested
    c2 = inner.clone()
    # clone should not be nested
    self.assertEqual(c2.run(), 2)


class VoidChainVariantTests(IsolatedAsyncioTestCase):
  """Void chains (no root) across variants."""

  async def test_void_chain(self):
    self.assertIsNone(Chain().run())


  async def test_void_chain_with_then(self):
    result = Chain().then(lambda: 42, ...).run()
    self.assertEqual(result, 42)


  async def test_void_chain_reuse(self):
    c = Chain()
    self.assertIsNone(c())

  async def test_void_clone(self):
    c = Chain()
    c2 = c.clone()
    self.assertIsNone(c2.run())


class ChainExceptSpecificExceptionsTests(IsolatedAsyncioTestCase):
  """except_ with specific exception types."""

  async def test_chain_except_specific_match(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).except_(lambda v: ran.append('caught'), exceptions=TestExc).run()
    except TestExc:
      pass
    self.assertEqual(ran, ['caught'])

  async def test_chain_except_specific_no_match(self):
    ran = []
    try:
      Chain(10).then(raise_test_exc).except_(lambda v: ran.append('caught'), exceptions=ValueError).run()
    except TestExc:
      pass
    self.assertEqual(ran, [])


class MultipleExceptHandlersTests(IsolatedAsyncioTestCase):
  """Multiple except_ handlers."""

  async def test_first_matching_handler_wins(self):
    class Exc1(TestExc):
      pass

    class Exc2(TestExc):
      pass

    ran = []

    def raise_exc1(v=None):
      raise Exc1()

    result = (
      Chain(10)
      .then(raise_exc1)
      .except_(lambda v: ran.append('h1') or 'h1', exceptions=Exc2, reraise=False)
      .except_(lambda v: ran.append('h2') or 'h2', exceptions=Exc1, reraise=False)
      .run()
    )
    self.assertEqual(result, 'h2')
    self.assertEqual(ran, ['h2'])
class ChainVoidWithOverrideTests(IsolatedAsyncioTestCase):
  """Void chain called with root override."""

  async def test_void_chain_with_override(self):
    c = Chain().then(lambda v: v * 3)
    self.assertEqual(c(10), 30)
    self.assertEqual(c(7), 21)

  async def test_void_chain_foreach_with_override(self):
    c = Chain().foreach(lambda x: x + 1)
    self.assertEqual(c([1, 2, 3]), [2, 3, 4])

  async def test_void_chain_gather_with_override(self):
    c = Chain().gather(lambda v: v * 2, lambda v: v + 1)
    self.assertEqual(c(5), [10, 6])


class CloneVoidChainTests(TestCase):
  """Clone void chains."""

  def test_clone_void_chain_then_override(self):
    c = Chain().then(lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(c2.run(5), 10)
    self.assertEqual(c2.run(3), 6)


if __name__ == '__main__':
  import unittest
  unittest.main()
