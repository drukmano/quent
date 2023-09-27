from tests.try_except_tests import get_empty_and_cls
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, DummySync, DummyAsync
from src.quent import Chain, ChainAttr, Cascade, CascadeAttr, run


class Exc1(Exception):
  pass



def throw():
  raise Exception


def throw_if(v):
  if v:
    raise Exception
  return v


def throw_if_1(v):
  if v:
    raise Exc1
  return v


class MegaTests(IsolatedAsyncioTestCase):
  async def test_mega_chain(self):
    class A:
      @property
      def a1(self):
        return None

      def b1(self):
        return self

    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(
            Chain(lambda v: v, 1)
            .then(fn)
            .then(0)
            .root(lambda v: v+1)
            .ignore(lambda v: 10)
            .ignore(
              Cascade()
              .then(fn)
              .then(Chain().if_(None).else_(throw))
              .then(Chain().if_(lambda v: False).else_(throw))
              .then(Chain().if_(lambda: False, ...).else_(throw))
              .then(Chain().not_().not_().if_not(throw))
              .then(Chain().eq(2).if_not(throw))
              .then(Chain().neq(1).if_not(throw))
              .then(Chain().eq(1).if_(throw))
              .then(Chain().then(type).is_(int).if_not(throw))
              .then(Chain().then(type).is_not(str).if_(None).else_(throw))
              .then(Chain().then(type).is_(str).if_(throw))
              .then(Chain().in_([1, 2]).if_not(throw))
              .then(Chain().not_in([1, 3]).if_not(throw))
              .then(Chain().in_([1, 3]).if_(throw))
            )
            .ignore(
              Cascade()
              .then(Chain(A()).attr('a1').root().attr_fn('b1').attr('a1'))
              .then(ChainAttr(A()).a1.root().b1().a1.root().a1.run)
              .then(CascadeAttr(A()).a1.a1.a1.a1.b1().a1.a1.run)
              .then(lambda: 10)
              .then(lambda v: Chain() | 5 | run(v), 1)
              .then(print, Chain())
              .then(print, Chain(1))
              .then(Chain(Chain).then(bool))
              , ...
            )
            .ignore(Chain().neq(2).if_(True).else_(lambda v: 1))
            .ignore(Chain().neq(2).if_(True).else_(lambda: 1, ...))
            .eq(2).if_not(throw)
            .ignore(Chain(fn).then(1).run, ...)
            .then(throw_if_1, True)
            .except_(exc_fin_check.on_except)
            .finally_(exc_fin_check.on_finally)
            .run()
          )
        except Exc1: pass
        self.assertTrue(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)
