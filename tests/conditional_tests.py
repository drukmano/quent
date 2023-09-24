from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_, DummySync, DummyAsync
from src.quent import Chain, ChainR, Cascade, CascadeR


class ConditionalTests(IsolatedAsyncioTestCase):
  async def yield_conditional(self, root_value, conditional_attr, truthy_value, falsy_value):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        get_seq = lambda: getattr(Chain(root_value).then(fn), conditional_attr)
        self.assertTrue(await await_(get_seq()(truthy_value).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).else_(False).run()))
        self.assertTrue(await await_(get_seq()(falsy_value).else_(True).run()))

        self.assertTrue(await await_(get_seq()(truthy_value, False).not_().run()))
        self.assertTrue(await await_(get_seq()(truthy_value, True).else_(False).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).not_().else_(True).run()))

        self.assertTrue(await await_(get_seq()(truthy_value, lambda: False).not_().run()))
        self.assertTrue(await await_(get_seq()(truthy_value, lambda: True).else_(lambda: False).run()))
        self.assertTrue(await await_(get_seq()(truthy_value).not_().else_(lambda: True).run()))

  async def test_not(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(False).not_().run()))
        self.assertTrue(await await_(Chain(False).then(fn).not_().run()))
        self.assertTrue(await await_(Chain(fn).then(False).not_().run()))
        self.assertTrue(await await_(Chain(True).not_().not_().run()))
        self.assertTrue(await await_(Chain(False).not_().run()))
        self.assertTrue(await await_(Chain(True).not_().else_(True).run()))
        self.assertTrue(await await_(Chain(True).then(fn).not_().else_(True).run()))
        self.assertTrue(await await_(Chain(fn).then(True).not_().else_(True).run()))

  async def test_eq(self):
    with self.subTest(conditional='eq'):
      await self.yield_conditional(1, 'eq', 1, 2)

  async def test_neq(self):
    with self.subTest(conditional='neq'):
      await self.yield_conditional(1, 'neq', 2, 1)

  async def test_is_(self):
    o = object()
    o2 = object()
    with self.subTest(conditional='is_'):
      await self.yield_conditional(o, 'is_', o, o2)

  async def test_is_not(self):
    o = object()
    o2 = object()
    with self.subTest(conditional='is_not'):
      await self.yield_conditional(o, 'is_not', o2, o)

  async def test_in_(self):
    t = [1, 2]
    f = [2, 3]
    with self.subTest(conditional='in_'):
      await self.yield_conditional(1, 'in_', t, f)

  async def test_not_in(self):
    t = [1, 2]
    f = [2, 3]
    with self.subTest(conditional='not_in'):
      await self.yield_conditional(1, 'not_in', f, t)

  async def test_conditional_last(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(1).then(fn).eq(2).not_().run()))

  async def test_conditional_last_attr(self):
    for fn, cls in [(empty, DummySync), (aempty, DummyAsync)]:
      with self.subTest(fn=fn):
        o = cls()
        self.assertTrue(await await_(Chain(o).then(fn).attr('a1').is_(o, o).attr_fn('b1').is_(o).run()))
        self.assertTrue(await await_(ChainR(o).then(fn).a1.is_(o, o).b1().is_(o).run()))

  async def test_combinations(self):
    # a few random combinations of conditionals
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain(None).then(fn).is_(None, lambda: False).else_(True).is_(False).run()))
        self.assertTrue(await await_(Chain(5).then(lambda v: v*10).in_([50, 51], fn, 9).else_(fn, 10).eq(9).run()))
        self.assertTrue(await await_(Chain(5).then(lambda v: v*10).in_([50, 51], fn, 9).else_(fn, 10).if_(lambda v: v < 10, lambda: False).is_(False).is_not(False).run()))
        self.assertTrue(await await_(Chain(lambda: 1).then(fn).then(Chain().then(lambda v: v*10).neq(10, lambda: 5).else_(lambda: 6)).neq(6).if_(..., lambda: False).else_(True).run()))
        self.assertTrue(await await_(Chain(lambda: 1).then(fn).then(Chain().then(lambda v: v+5)).eq(6).run()))
