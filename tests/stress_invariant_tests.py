import asyncio
import gc
import weakref
from unittest import TestCase
from tests.utils import empty, aempty, await_, MyTestCase
from quent import Chain, Cascade

try:
  import hypothesis
  from hypothesis import given, strategies as st, settings as hyp_settings
  HAS_HYPOTHESIS = True
except ImportError:
  HAS_HYPOTHESIS = False

import unittest


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

  async def test_empty_gather(self):
    result = await await_(Chain(5).gather().run())
    super(MyTestCase, self).assertEqual(result, [])


if __name__ == '__main__':
  unittest.main()
