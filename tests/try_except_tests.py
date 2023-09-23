from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_
from src.quent import Chain, ChainR, Cascade, CascadeR


class ExceptFinallyCheckSync:
  def __init__(self):
    self.ran_exc = False
    self.ran_finally = False

  def on_except(self, v=None, *, register=True):
    self.ran_exc = register

  def on_finally(self, v=None, *, register=True):
    self.ran_finally = register


class ExceptFinallyCheckAsync:
  def __init__(self):
    self.ran_exc = False
    self.ran_finally = False

  async def on_except(self, v=None, *, register=True):
    self.ran_exc = register

  async def on_finally(self, v=None, *, register=True):
    self.ran_finally = register


def get_empty_and_cls():
  yield from iter([(empty, ExceptFinallyCheckSync), (aempty, ExceptFinallyCheckAsync)])


class TryExceptTest(IsolatedAsyncioTestCase):
  # TODO improve naming
  async def test_try_except_1(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        await await_(Chain(False).then(fn).then(throw_if).except_(exc_fin_check.on_except).finally_(exc_fin_check.on_finally).run())
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_2(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        await await_(Chain(False).then(fn).then(throw_if).except_(exc_fin_check.on_except, ...).finally_(exc_fin_check.on_finally, ...).run())
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_3(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        await await_(Chain(False).then(fn).then(throw_if).except_(exc_fin_check.on_except, register=False).finally_(exc_fin_check.on_finally, register=False).run())
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertFalse(exc_fin_check.ran_finally)

  async def test_try_except_4(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(True).then(fn).then(throw_if).except_(exc_fin_check.on_except).finally_(exc_fin_check.on_finally).run())
        except Exception: pass
        self.assertTrue(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_5(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(True).then(fn).then(throw_if).except_(exc_fin_check.on_except, ...).finally_(exc_fin_check.on_finally, ...).run())
        except Exception: pass
        self.assertTrue(exc_fin_check.ran_exc)
        self.assertTrue(exc_fin_check.ran_finally)

  async def test_try_except_6(self):
    for fn, efc_cls in get_empty_and_cls():
      with self.subTest(fn=fn):
        exc_fin_check = efc_cls()
        try:
          await await_(Chain(True).then(fn).then(throw_if).except_(exc_fin_check.on_except, register=False).finally_(exc_fin_check.on_finally, register=False).run())
        except Exception: pass
        self.assertFalse(exc_fin_check.ran_exc)
        self.assertFalse(exc_fin_check.ran_finally)

  async def test_try_except_unknown_async_at_exc(self):
    # this function shows that if an exception is raised "before" an async coroutine has been
    # run (or if there aren't any async functions at all), and the except / finally callbacks are async,
    # they will not be awaited
    exc_fin_check = ExceptFinallyCheckAsync()
    try:
      await await_(Chain(True).then(throw_if).then(aempty).except_(exc_fin_check.on_except).finally_(exc_fin_check.on_finally).run())
    except Exception: pass
    self.assertFalse(exc_fin_check.ran_exc)
    self.assertFalse(exc_fin_check.ran_finally)
