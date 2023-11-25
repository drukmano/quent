import asyncio
import contextlib
import inspect
from typing import *
import time
from contextlib import contextmanager, asynccontextmanager
from tests.flex_context import FlexContext
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_
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


class Exc1(Exception):
  pass


class Exc2(Exception):
  pass


class Exc3(Exc2):
  pass


def raise_(e=Exception):
  raise e


class ExcFinallyTests(MyExcTestCase):
  async def test_except(self):
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except).run())
        except Exception: pass
        self.assertTrue(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except, exceptions=Exc1).run())
        except Exception: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except, exceptions=[Exc1, Exc2]).run())
        except Exception: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_, Exc1).except_(efc.on_except, exceptions=Exc1).run())
        except Exception: pass
        self.assertTrue(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_, Exc1).except_(efc.on_except, exceptions=Exc2).run())
        except Exception: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_, Exc1).except_(efc.on_except, exceptions=[Exc2, Exc3]).run())
        except Exception: pass
        self.assertFalse(efc.ran_exc)

        efc = efc_cls()
        efc2 = efc_cls()
        try:
          await await_(
            Chain(fn).then(raise_, Exc1)
            .except_(efc.on_except, exceptions=Exc2)
            .except_(efc2.on_except, exceptions=Exc1).run()
          )
        except Exception: pass
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
        except Exception: pass
        self.assertTrue(efc.ran_exc)
        self.assertFalse(efc2.ran_exc)

        efc = efc_cls()
        efc2 = efc_cls()
        try:
          await await_(
            Chain(fn).then(raise_, Exc3)
            .except_(efc.on_except, exceptions=[Exc2])
            .except_(efc2.on_except, exceptions=[Exc1, Exception]).run()
          )
        except Exception: pass
        self.assertTrue(efc.ran_exc)
        self.assertFalse(efc2.ran_exc)

  async def test_except_in_with(self):
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        try:
          await await_(Chain(fn).then(FlexContext, v=1).with_(raise_).except_(efc.on_except).run())
        except Exception: pass
        self.assertTrue(efc.ran_exc)

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

  async def test_raise_on_await(self):
    async def f():
      raise Exception

    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        for fn1, fn2 in [(fn, f), (f, fn)]:
          with self.subTest(fn1=fn1, fn2=fn2):
            efc = efc_cls()
            try:
              await await_(Chain(fn1).then(fn2).except_(efc.on_except).run())
            except Exception: pass
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
    except Exception: pass
    self.assertFalse(efc.ran_exc)
    self.assertFalse(efc.ran_finally)
    await asyncio.sleep(0.3)  # allow for the tasks to finish
    self.assertTrue(efc.ran_exc)
    self.assertTrue(efc.ran_finally)

  async def test_finally(self):
    for fn, efc_cls, ctx in self.with_fn_efc():
      with ctx:
        efc = efc_cls()
        await await_(Chain(fn).finally_(efc.on_finally).run())
        self.assertTrue(efc.ran_finally)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).finally_(efc.on_finally).run())
        except Exception: pass
        self.assertTrue(efc.ran_finally)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except).finally_(efc.on_finally).run())
        except Exception: pass
        self.assertTrue(efc.ran_exc)
        self.assertTrue(efc.ran_finally)

        efc = efc_cls()
        try:
          await await_(Chain(fn).then(raise_).except_(efc.on_except, exceptions=Exc1).finally_(efc.on_finally).run())
        except Exception: pass
        self.assertFalse(efc.ran_exc)
        self.assertTrue(efc.ran_finally)
