import asyncio
import contextlib
import inspect
from typing import *
import time
from contextlib import contextmanager, asynccontextmanager
from tests.flex_context import FlexContext
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_, TestExc
from quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException, run


class ExceptFinallyCheckSync:
  def __init__(self):
    self.ran_exc = False
    self.ran_finally = False

  def on_except(self, v=None, *, register=True):
    self.ran_exc = register
    return v

  def on_finally(self, v=None, *, register=True):
    self.ran_finally = register


class ExceptFinallyCheckAsync:
  def __init__(self):
    self.ran_exc = False
    self.ran_finally = False

  async def on_except(self, v=None, *, register=True):
    self.ran_exc = register
    return v

  async def on_finally(self, v=None, *, register=True):
    self.ran_finally = register


class MyExcTestCase(IsolatedAsyncioTestCase):
  def with_fn_efc(self):
    for fn, efc in [(empty, ExceptFinallyCheckSync), (aempty, ExceptFinallyCheckAsync)]:
      yield fn, efc, self.subTest(fn=fn, efc=efc)


class Exc1(TestExc):
  pass


class Exc2(TestExc):
  pass


class Exc3(Exc2):
  pass


def raise_(e=TestExc):
  if e is None:
    e = TestExc
  raise e


class ExcFinallyTests(MyExcTestCase):
  async def test_except(self):
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except).run())
        except TestExc: pass
        self.assertTrue(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except, exceptions=Exc1).run())
        except TestExc: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except, exceptions=[Exc1, Exc2]).run())
        except TestExc: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_, Exc1).except_(efc.on_except, exceptions=Exc1).run())
        except TestExc: pass
        self.assertTrue(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_, Exc1).except_(efc.on_except, exceptions=Exc2).run())
        except TestExc: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_, Exc1).except_(efc.on_except, exceptions=[Exc2, Exc3]).run())
        except TestExc: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        efc2 = efc_cls()
        try:
          await await_(
            Chain(fn).then(raise_, Exc1)
            .except_(efc.on_except, exceptions=Exc2)
            .except_(efc2.on_except, exceptions=Exc1).run()
          )
        except TestExc: pass
        self.assertFalse(efc.ran_exc)
        self.assertTrue(efc2.ran_exc)

        efc = efc_cls()
        efc2 = efc_cls()
        try:
          await await_(
            Chain(fn).then(raise_, Exc3)
            .except_(efc.on_except, exceptions=[Exc1, Exc2])
            .except_(efc2.on_except, exceptions=RuntimeError).run()
          )
        except TestExc: pass
        self.assertTrue(efc.ran_exc)
        self.assertFalse(efc2.ran_exc)

        efc = efc_cls()
        efc2 = efc_cls()
        try:
          await await_(
            Chain(fn).then(raise_, Exc3)
            .except_(efc.on_except, exceptions=[Exc2])
            .except_(efc2.on_except, exceptions=[Exc1, TestExc]).run()
          )
        except TestExc: pass
        self.assertTrue(efc.ran_exc)
        self.assertFalse(efc2.ran_exc)

  async def test_except_in_with(self):
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        try:
          await await_(Chain(fn).then(FlexContext, v=1).with_(raise_, ...).except_(efc.on_except).run())
        except TestExc: pass
        self.assertTrue(efc.ran_exc)

  async def test_except_noraise(self):
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(lambda v: v, raise_=True).run())
        except TestExc:
          await await_(efc.on_except())
        self.assertTrue(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(lambda v: v, raise_=False).run())
        except TestExc:
          await await_(efc.on_except())
        self.assertFalse(efc.ran_exc)

  async def test_except_return(self):
    obj = object()
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        self.assertIs(await await_(Chain(fn).then(raise_).except_(efc.on_except, obj, return_=True).run()), obj)
        self.assertTrue(efc.ran_exc)

        efc = efc_cls()
        efc2 = efc_cls()
        self.assertIs(await await_(
          Chain(fn).then(raise_, Exc2)
          .except_(efc.on_except, object(), exceptions=Exc1, return_=True)
          .except_(efc2.on_except, obj, exceptions=Exc2, return_=True)
          .run()), obj)
        self.assertFalse(efc.ran_exc)
        self.assertTrue(efc2.ran_exc)

  async def test_multi_except(self):
    obj = object()
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc1 = efc_cls()
        efc2 = efc_cls()
        efc3 = efc_cls()
        efc4 = efc_cls()
        efc5 = efc_cls()
        try:
          await await_(
            Chain(fn)
            .except_(efc1.on_except)
            .raise_(Exc1())
            .except_(efc2.on_except, exceptions=Exc2)
            .except_(efc3.on_except, exceptions=TestExc)
            .except_(efc4.on_except)
            .run()
          )
        except TestExc:
          await await_(efc5.on_except())
        self.assertFalse(efc1.ran_exc)
        self.assertFalse(efc2.ran_exc)
        self.assertTrue(efc3.ran_exc)
        self.assertFalse(efc4.ran_exc)
        self.assertTrue(efc5.ran_exc)

        # with return_=True
        efc1 = efc_cls()
        efc2 = efc_cls()
        efc3 = efc_cls()
        efc4 = efc_cls()
        try:
          self.assertIs(await await_(
            Chain(fn)
            .except_(efc1.on_except)
            .raise_(Exc1())
            .except_(efc2.on_except, exceptions=Exc2)
            .except_(efc3.on_except, obj, exceptions=TestExc, return_=True)
            .except_(efc4.on_except)
            .run()
          ), obj)
        except TestExc:
          self.assertTrue(False)
        self.assertFalse(efc1.ran_exc)
        self.assertFalse(efc2.ran_exc)
        self.assertTrue(efc3.ran_exc)
        self.assertFalse(efc4.ran_exc)

        # with nested chain and no raise, the chain continues after exception
        efc1 = efc_cls()
        efc2 = efc_cls()
        efc3 = efc_cls()
        efc4 = efc_cls()
        try:
          self.assertIs(await await_(
            Chain(fn)
            .except_(efc1.on_except)
            .then(
              Chain(fn).raise_(Exc1())
              .except_(efc2.on_except, exceptions=Exc2)
              .except_(efc3.on_except, exceptions=TestExc, raise_=False), ...
            )
            .except_(efc4.on_except)
            .then(obj)
            .run()
          ), obj)
        except TestExc:
          self.assertTrue(False)
        self.assertFalse(efc1.ran_exc)
        self.assertFalse(efc2.ran_exc)
        self.assertTrue(efc3.ran_exc)
        self.assertFalse(efc4.ran_exc)

  async def test_raise_on_await(self):
    async def f(v=None):
      raise TestExc

    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        for fn1, fn2 in [(fn, f), (f, fn)]:
          with self.subTest(fn1=fn1, fn2=fn2):
            efc = efc_cls()
            try:
              await await_(Chain(fn1).then(fn2).except_(efc.on_except).run())
            except TestExc: pass
            self.assertTrue(efc.ran_exc)

  async def test_async_except_on_sync(self):
    # this function shows that if an exception is raised "before" an async coroutine has been
    # run (or if there aren't any async functions at all), and the except / finally callbacks are async,
    # they will be called in a separate task.
    efc = ExceptFinallyCheckAsync()
    try:
      await await_(
        Chain(raise_).then(aempty)
        .except_(Chain(asyncio.sleep, 0.1).then(efc.on_except), ...)
        .finally_(Chain(asyncio.sleep, 0.1).then(efc.on_finally), ...)
        .run()
      )
    except TestExc: pass
    self.assertFalse(efc.ran_exc)
    self.assertFalse(efc.ran_finally)
    await asyncio.sleep(0.3)  # allow for the tasks to finish
    self.assertTrue(efc.ran_exc)
    self.assertTrue(efc.ran_finally)

    # but, we can also await it if we use return_=True
    efc = ExceptFinallyCheckAsync()
    try:
      r = await await_(
        Chain(raise_).then(aempty)
        .except_(Chain(asyncio.sleep, 0.1).then(efc.on_except, 1), ..., return_=True)
        .run()
      )
      self.assertEqual(r, 1)
    except TestExc:
      self.assertTrue(False)
    self.assertTrue(efc.ran_exc)

  async def test_finally(self):
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        await await_(Chain(fn).finally_(efc.on_finally).run())
        self.assertTrue(efc.ran_finally)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).finally_(efc.on_finally).run())
        except TestExc: pass
        self.assertTrue(efc.ran_finally)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except).finally_(efc.on_finally).run())
        except TestExc: pass
        self.assertTrue(efc.ran_exc)
        self.assertTrue(efc.ran_finally)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except, exceptions=Exc1).finally_(efc.on_finally).run())
        except TestExc: pass
        self.assertFalse(efc.ran_exc)
        self.assertTrue(efc.ran_finally)
