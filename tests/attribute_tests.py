import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_
from tests.try_except_tests import get_empty_and_cls
from quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException


class A_sync:
  @property
  def a1(self):
    return self

  def f1(self):
    return self

  def f2(self, v=None):
    return self

  def f3(self):
    return 1


class B_sync:
  @property
  def b1(self):
    pass

  def f1(self):
    pass

  def f2(self, v=None):
    pass

  def f3(self):
    return 1


class C_sync:
  def f1(self):
    return 5

  def f2(self, v):
    return v*10

  def f3(self, v):
    return v+10

  def f4(self, v):
    return v*2


class A_async:
  @property
  async def a1(self):
    return self

  async def f1(self):
    return self

  def f2(self, v=None):
    return self

  async def f3(self):
    return 1


class B_async:
  @property
  async def b1(self):
    pass

  async def f1(self):
    pass

  def f2(self, v=None):
    pass

  async def f3(self):
    return 1


class C_async:
  async def f1(self):
    return 5

  async def f2(self, v):
    return v*10

  def f3(self, v):
    return v+10

  async def f4(self, v):
    return v*2


class AttributesTest(IsolatedAsyncioTestCase):
  async def test_attribute_last(self):
    for fn, (A, B, C) in [(empty, (A_sync, B_sync, C_sync)), (aempty, (A_async, B_async, C_async))]:
      with self.subTest(fn=fn):
        a = A()
        self.assertIs(await await_(ChainAttr(a).a1.run()), a)
        self.assertFalse(await await_(ChainAttr(a).a1.then(False).run()))

  async def test_attribute_on_void(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          r = await await_(Cascade().then(fn).attr('attr').run())
        except QuentException:
          await await_(exc_fin_check.on_except())
        self.assertTrue(exc_fin_check.ran_exc)

        exc_fin_check = efc_cls()
        try:
          await await_(Cascade().then(fn).attr('attr').except_(exc_fin_check.on_except).run())
        except QuentException: pass
        self.assertTrue(exc_fin_check.ran_exc)

  async def test_many_attributes(self):
    for fn, (A, B, C) in [(empty, (A_sync, B_sync, C_sync)), (aempty, (A_async, B_async, C_async))]:
      with self.subTest(fn=fn):
        a = A()
        self.assertTrue(await await_(Chain(a).attr('a1').attr_fn('f1').attr_fn('f2', 1).attr_fn('f2', ...).attr_fn('f3').eq(1).run()))
        self.assertTrue(await await_(ChainAttr(a).a1.f1().f2('123').f2(...).f2().f3().eq(1).run()))

        b = B()
        self.assertTrue(await await_(Chain(Cascade(b).attr('b1').attr_fn('f1').attr_fn('f2', 1).attr_fn('f2', ...).attr_fn('f3').attr('b1')).is_(b).run()))
        self.assertTrue(await await_(Chain(CascadeAttr(b).b1.f1().f2('123').f2(...).f2().f3().b1.f1()).is_(b).run()))

        c = C()
        self.assertTrue(await await_(Chain(c.f1).then(c.f2).then(c.f3).then(c.f4).eq(120).run()))
        self.assertTrue(await await_(Chain(c.f1).then(c.f2).root(c.f3).then(c.f4).eq(30).run()))

        self.assertTrue(await await_(Chain(c.f1()).then(c.f2).then(c.f3(0)).then(c.f4).eq(20).run()))
        self.assertTrue(await await_(Chain(c.f1).then(c.f2).then(c.f3(0)).then(c.f4).eq(20).run()))
