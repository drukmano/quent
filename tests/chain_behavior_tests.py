import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import await_
from quent import Chain


class ExceptSkipTests(TestCase):
  """Issue 57: Test that except_ handler is skipped when no exception occurs.

  When the chain runs without raising, exception handler links are
  skipped during evaluation (lines 283-285 of quent.pyx).
  """

  def test_handler_not_called_on_success(self):
    called = {"value": False}
    def handler(v=None):
      called["value"] = True
    result = Chain(1).then(lambda v: v + 1).except_(handler).run()
    self.assertEqual(result, 2)
    self.assertFalse(called["value"])

  def test_multiple_handlers_not_called_on_success(self):
    call_count = {"value": 0}
    def handler1(v=None):
      call_count["value"] += 1
    def handler2(v=None):
      call_count["value"] += 1
    result = (
      Chain(10)
      .then(lambda v: v * 2)
      .except_(handler1)
      .then(lambda v: v + 5)
      .except_(handler2)
      .run()
    )
    self.assertEqual(result, 25)
    self.assertEqual(call_count["value"], 0)

  def test_handler_not_called_with_void_chain(self):
    called = {"value": False}
    def handler(v=None):
      called["value"] = True
    result = Chain().then(lambda: 42).except_(handler).run()
    self.assertEqual(result, 42)
    self.assertFalse(called["value"])


class FinallyEmptyChainTests(IsolatedAsyncioTestCase):
  """Issue 58: Test finally_ on an empty chain (no evaluation links).

  Chain().finally_(handler).run() should call the handler and return None.
  """

  def test_sync_finally_on_empty_chain(self):
    called = {"value": False}
    def handler(v=None):
      called["value"] = True
    result = Chain().finally_(handler).run()
    self.assertIsNone(result)
    self.assertTrue(called["value"])

  def test_sync_finally_handler_receives_null_as_none(self):
    received = {"value": "sentinel"}
    def handler(v=None):
      received["value"] = v
    Chain().finally_(handler).run()
    # rv is Null (no root value), so evaluate_value calls handler()
    # with no positional arg, meaning v defaults to None
    self.assertIsNone(received["value"])

  async def test_async_finally_on_empty_chain(self):
    called = {"value": False}
    async def handler(v=None):
      called["value"] = True
    # An async finally handler on a sync-mode chain gets scheduled
    # as a separate Task (with a RuntimeWarning). We need to let
    # the event loop process it.
    import warnings
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)
      result = Chain().finally_(handler).run()
    self.assertIsNone(result)
    # Allow the scheduled task to execute
    await asyncio.sleep(0.1)
    self.assertTrue(called["value"])

  async def test_async_finally_on_empty_chain_with_sync_handler(self):
    called = {"value": False}
    def handler(v=None):
      called["value"] = True
    result = await await_(Chain().finally_(handler).run())
    self.assertIsNone(result)
    self.assertTrue(called["value"])
