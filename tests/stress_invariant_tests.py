import asyncio
import functools
import gc
import weakref
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException, run, FrozenChain
from quent.quent import _get_registry_size

try:
  import hypothesis
  from hypothesis import given, strategies as st, settings as hyp_settings
  HAS_HYPOTHESIS = True
except ImportError:
  HAS_HYPOTHESIS = False

import unittest


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


class _Ref:
  """Simple wrapper to allow weakref on objects that don't support it (e.g. Cython classes)."""
  def __init__(self, obj):
    self.obj = obj


class StressTests(MyTestCase):

  async def test_very_long_chain_1000_links(self):
    c = Chain(0)
    for _ in range(1000):
      c = c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 1000)

  async def test_very_long_chain_1000_links_async(self):
    c = Chain(0)
    for i in range(1000):
      if i % 10 == 0:
        c = c.then(lambda v: aempty(v + 1))
      else:
        c = c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 1000)

  async def test_deeply_nested_chains_10_levels(self):
    inner = Chain(1)
    for _ in range(9):
      inner = Chain().then(inner)
    await self.assertEqual(inner.run(), 1)

  async def test_deeply_nested_chains_20_levels(self):
    inner = Chain(1)
    for _ in range(19):
      inner = Chain().then(inner)
    await self.assertEqual(inner.run(), 1)

  async def test_concurrent_frozen_chain_1000_calls(self):
    frozen = Chain().then(lambda v: v * 2).freeze()
    tasks = [asyncio.ensure_future(await_(frozen.run(i))) for i in range(1000)]
    results = await asyncio.gather(*tasks)
    for i in range(1000):
      super(MyTestCase, self).assertEqual(results[i], i * 2)

  async def test_concurrent_frozen_chain_with_async_ops(self):
    frozen = Chain().then(lambda v: aempty(v * 3)).freeze()
    tasks = [asyncio.ensure_future(await_(frozen.run(i))) for i in range(500)]
    results = await asyncio.gather(*tasks)
    for i in range(500):
      super(MyTestCase, self).assertEqual(results[i], i * 3)

  async def test_rapid_clone_stress(self):
    c = Chain(42).then(lambda v: v + 1).then(lambda v: v * 2)
    clones = [c.clone() for _ in range(1000)]
    for clone in clones:
      await self.assertEqual(clone.run(), 86)

  async def test_memory_cleanup_after_chain(self):
    wrapper = _Ref(Chain(42).then(lambda v: v + 1))
    ref = weakref.ref(wrapper)
    result = wrapper.obj.run()
    super(MyTestCase, self).assertEqual(result, 43)
    del wrapper
    gc.collect()
    super(MyTestCase, self).assertIsNone(ref())

  async def test_task_registry_under_load(self):
    tasks = []
    for i in range(50):
      c = Chain(aempty, i).then(lambda v: v + 1).autorun()
      result = c.run()
      tasks.append(result)
    results = await asyncio.gather(*tasks)
    for i in range(50):
      super(MyTestCase, self).assertEqual(results[i], i + 1)
    # Allow the event loop to process done callbacks
    await asyncio.sleep(0)
    size_after = _get_registry_size()
    # All tasks completed and their done callbacks should have discarded them
    super(MyTestCase, self).assertEqual(size_after, 0)


class AlgebraicPropertyTests(MyTestCase):

  async def test_identity_chain(self):
    for v in [None, 0, False, '', 42, 'hello', [], {}]:
      with self.subTest(v=v):
        await self.assertEqual(Chain(v).then(lambda v: v).run(), v)

  async def test_identity_cascade(self):
    for v in [1, 42, 'hello', [1, 2]]:
      with self.subTest(v=v):
        await self.assertEqual(Cascade(v).then(lambda v: str(v) + '_ignored').run(), v)

  async def test_composition_associativity(self):
    f = lambda v: v + 1
    g = lambda v: v * 2
    for v in [0, 1, 5, 10, -3]:
      with self.subTest(v=v):
        result1 = await await_(Chain(v).then(f).then(g).run())
        result2 = await await_(Chain(v).then(lambda v: g(f(v))).run())
        super(MyTestCase, self).assertEqual(result1, result2)

  async def test_cascade_invariant(self):
    root_val = 99
    c = Cascade(root_val).then(lambda v: v + 1).then(lambda v: v * 100).then(lambda v: 'ignored')
    await self.assertEqual(c.run(), root_val)

  async def test_clone_equivalence(self):
    chains = [
      Chain(42),
      Chain(10).then(lambda v: v + 5),
      Chain(3).then(lambda v: v * 2).then(lambda v: v - 1),
      Chain('hello').then(lambda v: v.upper()),
    ]
    for c in chains:
      with self.subTest(chain=repr(c)):
        original_result = await await_(c.clone().run())
        clone_result = await await_(c.clone().run())
        super(MyTestCase, self).assertEqual(original_result, clone_result)

  async def test_freeze_equivalence(self):
    c = Chain().then(lambda v: v * 2)
    frozen = c.freeze()
    for v in [1, 5, 10, -3, 0]:
      with self.subTest(v=v):
        frozen_result = await await_(frozen(v))
        clone_result = await await_(c.clone().run(v))
        super(MyTestCase, self).assertEqual(frozen_result, clone_result)

  async def test_safe_run_equivalence(self):
    c = Chain(42).then(lambda v: v + 8)
    safe_result = await await_(c.safe_run())
    run_result = await await_(c.clone().run())
    super(MyTestCase, self).assertEqual(safe_result, run_result)

  async def test_pipe_then_equivalence(self):
    f = lambda v: v + 10
    for v in [0, 5, -1, 100]:
      with self.subTest(v=v):
        pipe_result = await await_(Chain(v).pipe(f).run())
        then_result = await await_(Chain(v).then(f).run())
        super(MyTestCase, self).assertEqual(pipe_result, then_result)

  async def test_or_identity_truthy(self):
    fallback = 'fallback'
    for v in [1, True, 'yes', [1], {'a': 1}, 42]:
      with self.subTest(v=v):
        result = await await_(Chain(v).or_(fallback).run())
        super(MyTestCase, self).assertEqual(result, v)

  async def test_or_identity_falsy(self):
    fallback = 'fallback'
    for v in [0, False, '', None, [], {}]:
      with self.subTest(v=v):
        result = await await_(Chain(v).or_(fallback).run())
        super(MyTestCase, self).assertEqual(result, fallback)

  async def test_not_involution(self):
    for v in [True, False, 0, 1, '', 'x', None, [], [1]]:
      with self.subTest(v=v):
        result = await await_(Chain(v).not_().run())
        super(MyTestCase, self).assertEqual(result, not v)

  async def test_suppress_then_no_exception(self):
    def raiser(v):
      raise ValueError('test error')
    result = await await_(Chain(1).then(raiser).suppress().run())
    super(MyTestCase, self).assertIsNone(result)

  async def test_foreach_map_equivalence(self):
    lst = [1, 2, 3, 4, 5]
    f = lambda x: x * 2
    result = await await_(Chain(lst).foreach(f).run())
    super(MyTestCase, self).assertEqual(result, list(map(f, lst)))

  async def test_filter_equivalence(self):
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pred = lambda x: x % 2 == 0
    result = await await_(Chain(lst).filter(pred).run())
    super(MyTestCase, self).assertEqual(result, list(filter(pred, lst)))

  async def test_reduce_equivalence(self):
    lst = [1, 2, 3, 4, 5]
    fn = lambda a, b: a + b
    init = 0
    result = await await_(Chain(lst).reduce(fn, init).run())
    super(MyTestCase, self).assertEqual(result, functools.reduce(fn, lst, init))


if HAS_HYPOTHESIS:
  class PropertyBasedTests(TestCase):

    @given(st.integers())
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_chain_preserves_ints(self, v):
      self.assertEqual(Chain(v).run(), v)

    @given(st.text())
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_chain_preserves_strings(self, v):
      self.assertEqual(Chain(v).run(), v)

    def test_chain_preserves_none(self):
      self.assertIsNone(Chain(None).run())

    @given(st.booleans())
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_chain_preserves_bools(self, v):
      self.assertEqual(Chain(v).run(), v)

    @given(st.integers(), st.integers())
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_chain_then_literal(self, a, b):
      self.assertEqual(Chain(a).then(b).run(), b)

    @given(st.integers())
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_cascade_returns_root(self, v):
      self.assertEqual(Cascade(v).then(lambda v: v * 2).run(), v)

    @given(
      st.one_of(st.none(), st.integers()),
      st.integers()
    )
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_or_semantics(self, v, fallback):
      result = Chain(v).or_(fallback).run()
      expected = v or fallback
      self.assertEqual(result, expected)

    @given(st.one_of(st.none(), st.booleans(), st.integers()))
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_not_semantics(self, v):
      result = Chain(v).not_().run()
      self.assertEqual(result, not v)

    @given(st.integers(), st.integers())
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_eq_semantics(self, a, b):
      result = Chain(a).eq(b).run()
      self.assertEqual(result, a == b)

    @given(st.lists(st.integers()))
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_filter_preserves_elements(self, lst):
      result = Chain(lst).filter(lambda x: True).run()
      self.assertEqual(result, lst)

    @given(st.lists(st.integers()))
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_foreach_length(self, lst):
      result = Chain(lst).foreach(lambda x: x * 2).run()
      self.assertEqual(len(result), len(lst))

    @given(st.lists(st.integers(), min_size=1))
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_reduce_sum(self, lst):
      result = Chain(lst).reduce(lambda a, b: a + b).run()
      self.assertEqual(result, sum(lst))

    @given(st.integers())
    @hyp_settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_clone_idempotent(self, v):
      self.assertEqual(Chain(v).clone().run(), v)

else:
  @unittest.skip('hypothesis not installed')
  class PropertyBasedTests(TestCase):
    pass


class _SimpleContextManager:
  """A non-callable context manager for use in tests."""
  def __init__(self, value):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.exited = True
    return False


class EdgeCaseComboTests(MyTestCase):

  async def test_set_async_false_with_except_finally(self):
    tracker = {'exc': False, 'fin': False}
    def on_except(v=None):
      tracker['exc'] = True
    def on_finally(v=None):
      tracker['fin'] = True
    def raiser():
      raise ValueError('test')
    c = (
      Chain(raiser)
      .set_async(False)
      .except_(on_except, exceptions=ValueError, reraise=False)
      .finally_(on_finally)
    )
    result = await await_(c.run())
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertTrue(tracker['exc'])
    super(MyTestCase, self).assertTrue(tracker['fin'])

  async def test_debug_with_except_finally(self):
    tracker = {'exc': False, 'fin': False}
    def on_except(v=None):
      tracker['exc'] = True
    def on_finally(v=None):
      tracker['fin'] = True
    def raiser():
      raise ValueError('debug test')
    c = (
      Chain(raiser)
      .config(debug=True)
      .except_(on_except, exceptions=ValueError, reraise=False)
      .finally_(on_finally)
    )
    result = await await_(c.run())
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertTrue(tracker['exc'])
    super(MyTestCase, self).assertTrue(tracker['fin'])

  async def test_filter_then_foreach(self):
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = await await_(
      Chain(lst).filter(lambda x: x % 2 == 0).foreach(lambda x: x * 10).run()
    )
    super(MyTestCase, self).assertEqual(result, [20, 40, 60, 80, 100])

  async def test_reduce_then_then(self):
    lst = [1, 2, 3, 4, 5]
    result = await await_(
      Chain(lst).reduce(lambda a, b: a + b, 0).then(lambda v: v * 2).run()
    )
    super(MyTestCase, self).assertEqual(result, 30)

  async def test_with_then_except(self):
    tracker = {'exc': False}
    def on_except(v=None):
      tracker['exc'] = True

    def body(ctx_val):
      raise ValueError('in with body')

    cm = _SimpleContextManager('ctx_value')
    c = (
      Chain(cm)
      .with_(body)
      .except_(on_except, exceptions=ValueError, reraise=False)
    )
    result = await await_(c.run())
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertTrue(tracker['exc'])
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_multiple_condition_if_else_sequences(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 10)
            .condition(lambda v: fn(v > 5)).if_(lambda v: fn('big')).else_(lambda v: fn('small'))
            .condition(lambda v: fn(len(v) > 2)).if_(lambda v: fn('long')).else_(lambda v: fn('short'))
            .run()
        )
        super(MyTestCase, self).assertEqual(result, 'long')

  async def test_isinstance_if_else_with_async(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 'hello')
            .isinstance_(str)
            .if_(lambda v: fn(v.upper()))
            .else_(lambda v: fn(str(v)))
            .run()
        )
        super(MyTestCase, self).assertEqual(result, 'HELLO')

        result = await await_(
          Chain(fn, 123)
            .isinstance_(str)
            .if_(lambda v: fn(v.upper()))
            .else_(lambda v: fn(str(v)))
            .run()
        )
        super(MyTestCase, self).assertEqual(result, '123')

  async def test_safe_run_with_except_finally(self):
    tracker = {'exc': False, 'fin': False}
    def on_except(v=None):
      tracker['exc'] = True
    def on_finally(v=None):
      tracker['fin'] = True
    def raiser():
      raise ValueError('safe_run test')
    c = (
      Chain(raiser)
      .except_(on_except, exceptions=ValueError, reraise=False)
      .finally_(on_finally)
    )
    result = await await_(c.safe_run())
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertTrue(tracker['exc'])
    super(MyTestCase, self).assertTrue(tracker['fin'])

  async def test_compose_with_async(self):
    async def async_double(v):
      return v * 2
    chain1 = Chain().then(lambda v: v + 1)
    chain2 = Chain().then(lambda v: v + 10)
    composed = Chain.compose(chain1, async_double, chain2)
    result = await await_(composed.run(5))
    # 5 -> chain1(5)=6 -> async_double(6)=12 -> chain2(12)=22
    super(MyTestCase, self).assertEqual(result, 22)

  async def test_empty_gather(self):
    result = await await_(Chain(5).gather().run())
    super(MyTestCase, self).assertEqual(result, [])

  async def test_set_async_false_with_foreach_filter_reduce(self):
    lst = [1, 2, 3, 4, 5, 6]
    # foreach with set_async(False)
    result = Chain(lst).set_async(False).foreach(lambda x: x * 2).run()
    super(MyTestCase, self).assertEqual(result, [2, 4, 6, 8, 10, 12])

    # filter with set_async(False)
    result = Chain(lst).set_async(False).filter(lambda x: x > 3).run()
    super(MyTestCase, self).assertEqual(result, [4, 5, 6])

    # reduce with set_async(False)
    result = Chain(lst).set_async(False).reduce(lambda a, b: a + b, 0).run()
    super(MyTestCase, self).assertEqual(result, 21)


if __name__ == '__main__':
  unittest.main()
