"""Tests for the simple execution path finally_ support.

Covers:
  1. _is_simple flag behavior with finally_
  2. finally_ on success paths (sync and async)
  3. finally_ on exception paths
  4. finally_ handler itself raising exceptions
  5. Control flow signals (return_, break_) inside finally_
  6. RuntimeWarning for async finally on sync chain
  7. no_async mode with finally_
  8. finally_ after Return/Break signals from chain body
  9. Cascade + finally_
"""
import unittest
import asyncio
import warnings
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# 1. SimpleFinallyFlagTests
# ---------------------------------------------------------------------------
class SimpleFinallyFlagTests(unittest.TestCase):
  """Verify _is_simple flag behavior when finally_ is used."""

  def test_is_simple_with_finally(self):
    """Chain with .then() and .finally_() should remain simple."""
    c = Chain(42).then(lambda v: v).finally_(lambda v: None)
    self.assertTrue(c._is_simple)

  def test_is_simple_false_with_do(self):
    """Chain with .do() should not be simple."""
    c = Chain(42).do(lambda v: None)
    self.assertFalse(c._is_simple)

  def test_is_simple_false_with_except(self):
    """Chain with .except_() should not be simple."""
    c = Chain(42).except_(lambda v: None)
    self.assertFalse(c._is_simple)

  def test_is_simple_with_only_then(self):
    """Chain with only .then() should be simple."""
    c = Chain(42).then(str)
    self.assertTrue(c._is_simple)

  def test_is_simple_with_multiple_then_and_finally(self):
    """Chain with multiple .then() and .finally_() should remain simple."""
    c = Chain(42).then(str).then(len).finally_(lambda v: None)
    self.assertTrue(c._is_simple)


# ---------------------------------------------------------------------------
# 2. SimpleFinallySuccessTests
# ---------------------------------------------------------------------------
class SimpleFinallySuccessTests(MyTestCase):
  """Verify finally_ runs on happy path for simple chains."""

  async def test_finally_runs_on_success_sync(self):
    """Sync chain with .then() and .finally_() — finally runs and result is correct."""
    tracker = []
    result = Chain(42).then(lambda v: v + 1).finally_(lambda v: tracker.append(v)).run()
    await self.assertEqual(result, 43)
    unittest.TestCase.assertEqual(self, len(tracker), 1)

  async def test_finally_runs_on_success_async(self):
    """Async chain with .then() and .finally_() — finally runs and result is correct."""
    tracker = []
    result = Chain(aempty, 42).then(lambda v: v + 1).finally_(lambda v: tracker.append(v)).run()
    await self.assertEqual(result, 43)
    unittest.TestCase.assertEqual(self, len(tracker), 1)

  async def test_finally_receives_root_value(self):
    """The finally callback receives the root value, not the current value."""
    received = []
    for fn, ctx in self.with_fn():
      with ctx:
        received.clear()
        result = Chain(fn, 10).then(lambda v: v * 5).finally_(lambda v: received.append(v)).run()
        await self.assertEqual(result, 50)
        unittest.TestCase.assertEqual(self, received, [10])

  async def test_finally_does_not_alter_return(self):
    """The finally callback's return value does not alter the chain result."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, 42).then(lambda v: v + 8).finally_(lambda v: 9999).run()
        await self.assertEqual(result, 50)

  async def test_void_chain_with_finally(self):
    """Chain with no root value — .then() result is returned, finally runs."""
    tracker = []
    # Void chain: no root value, so the first .then() receives no args (Null).
    # Use a no-arg lambda for the .then() and a default-param lambda for finally_.
    result = Chain().then(lambda: 42).finally_(lambda v=None: tracker.append(v)).run()
    await self.assertEqual(result, 42)
    unittest.TestCase.assertEqual(self, len(tracker), 1)

  async def test_multiple_then_with_finally(self):
    """Chain with 3+ .then() links and .finally_() — all execute and finally runs."""
    tracker = []
    for fn, ctx in self.with_fn():
      with ctx:
        tracker.clear()
        result = (
          Chain(fn, 2)
          .then(lambda v: v * 3)
          .then(lambda v: v + 4)
          .then(lambda v: v * 10)
          .finally_(lambda v: tracker.append(v))
          .run()
        )
        await self.assertEqual(result, 100)
        unittest.TestCase.assertEqual(self, tracker, [2])

  async def test_root_override_with_finally(self):
    """Root override via .run(value) — result is correct and finally receives root."""
    tracker = []
    result = (
      Chain()
      .then(lambda v: v * 2)
      .finally_(lambda v: tracker.append(v))
      .run(10)
    )
    await self.assertEqual(result, 20)
    unittest.TestCase.assertEqual(self, tracker, [10])

  async def test_mixed_sync_async_then_with_finally(self):
    """Chain with both sync and async .then() callbacks plus finally_."""
    tracker = []
    for fn, ctx in self.with_fn():
      with ctx:
        tracker.clear()
        result = (
          Chain(fn, 5)
          .then(lambda v: v + 5)
          .then(aempty)
          .then(lambda v: v * 2)
          .finally_(lambda v: tracker.append(v))
          .run()
        )
        await self.assertEqual(result, 20)
        unittest.TestCase.assertEqual(self, tracker, [5])


# ---------------------------------------------------------------------------
# 3. SimpleFinallyExceptionTests
# ---------------------------------------------------------------------------
class SimpleFinallyExceptionTests(MyTestCase):
  """Verify finally_ runs when an exception occurs in the chain."""

  async def test_finally_runs_on_sync_exception(self):
    """Sync chain raises in .then() — finally runs and exception propagates."""
    tracker = []
    def raise_test_exc(v):
      raise TestExc('boom')
    with self.assertRaises(TestExc):
      Chain(42).then(raise_test_exc).finally_(lambda v: tracker.append(v)).run()
    unittest.TestCase.assertEqual(self, tracker, [42])

  async def test_finally_runs_on_async_exception(self):
    """Async chain raises — finally runs and exception propagates."""
    tracker = []
    async def async_raise(v):
      raise TestExc('async boom')
    with self.assertRaises(TestExc):
      await Chain(aempty, 42).then(async_raise).finally_(lambda v: tracker.append(v)).run()
    unittest.TestCase.assertEqual(self, tracker, [42])

  async def test_finally_runs_on_root_exception(self):
    """Root value callable raises — finally still runs."""
    tracker = []
    def bad_root():
      raise TestExc('root exploded')
    # When root raises, root_value is Null, so finally gets called with no args.
    # Use a default parameter to handle this.
    with self.assertRaises(TestExc):
      Chain(bad_root).then(lambda v: v).finally_(lambda v=None: tracker.append(v)).run()
    unittest.TestCase.assertEqual(self, len(tracker), 1)

  async def test_finally_runs_on_zero_division(self):
    """.then(lambda v: v / 0) — finally runs, ZeroDivisionError propagates."""
    tracker = []
    for fn, ctx in self.with_fn():
      with ctx:
        tracker.clear()
        with self.assertRaises(ZeroDivisionError):
          await await_(
            Chain(fn, 42).then(lambda v: v / 0).finally_(lambda v: tracker.append(v)).run()
          )
        unittest.TestCase.assertEqual(self, tracker, [42])


# ---------------------------------------------------------------------------
# 4. SimpleFinallyHandlerRaisesTests
# ---------------------------------------------------------------------------
class SimpleFinallyHandlerRaisesTests(MyTestCase):
  """Verify behavior when the finally_ handler itself raises."""

  async def test_handler_raises_on_success_path(self):
    """Chain succeeds but finally handler raises — that exception propagates."""
    def bad_finally(v):
      raise ValueError('finally went wrong')
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(ValueError) as cm:
          await await_(
            Chain(fn, 42).then(lambda v: v + 1).finally_(bad_finally).run()
          )
        unittest.TestCase.assertIn(self, 'finally went wrong', str(cm.exception))

  async def test_handler_raises_on_exception_path(self):
    """Chain raises TestExc, finally handler raises ValueError — ValueError propagates."""
    def raise_test(v):
      raise TestExc('chain error')
    def bad_finally(v):
      raise ValueError('finally error')
    with self.assertRaises(ValueError) as cm:
      Chain(42).then(raise_test).finally_(bad_finally).run()
    unittest.TestCase.assertIn(self, 'finally error', str(cm.exception))

  async def test_async_handler_raises(self):
    """Async finally handler raises — exception propagates."""
    async def bad_async_finally(v):
      raise ValueError('async finally error')
    with self.assertRaises(ValueError) as cm:
      await Chain(aempty, 42).then(lambda v: v).finally_(bad_async_finally).run()
    unittest.TestCase.assertIn(self, 'async finally error', str(cm.exception))

  async def test_context_preserved_when_handler_raises_over_chain_exception(self):
    """When finally handler raises over a chain exception, __context__ is the original."""
    def raise_test(v):
      raise TestExc('original')
    def bad_finally(v):
      raise ValueError('over original')
    try:
      Chain(42).then(raise_test).finally_(bad_finally).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      unittest.TestCase.assertIsInstance(self, exc.__context__, TestExc)
      unittest.TestCase.assertIn(self, 'original', str(exc.__context__))


# ---------------------------------------------------------------------------
# 5. SimpleFinallyControlFlowTests
# ---------------------------------------------------------------------------
class SimpleFinallyControlFlowTests(MyTestCase):
  """Verify control flow signals in finally_ raise QuentException."""

  async def test_return_in_finally_raises_quent_exception(self):
    """Chain.return_() in finally_ raises QuentException with control flow message."""
    with self.assertRaises(QuentException) as cm:
      Chain(42).then(lambda v: v).finally_(lambda v: Chain.return_(v)).run()
    unittest.TestCase.assertIn(self, 'control flow', str(cm.exception).lower())

  async def test_break_in_finally_raises_quent_exception(self):
    """Chain.break_() in finally_ raises QuentException with control flow message."""
    with self.assertRaises(QuentException) as cm:
      Chain(42).then(lambda v: v).finally_(lambda v: Chain.break_()).run()
    unittest.TestCase.assertIn(self, 'control flow', str(cm.exception).lower())

  async def test_return_in_finally_async(self):
    """Async variant: Chain.return_() in finally_ raises QuentException."""
    async def async_return_finally(v):
      Chain.return_(v)
    with self.assertRaises(QuentException) as cm:
      await Chain(aempty, 42).then(lambda v: v).finally_(async_return_finally).run()
    unittest.TestCase.assertIn(self, 'control flow', str(cm.exception).lower())

  async def test_break_in_finally_async(self):
    """Async variant: Chain.break_() in finally_ raises QuentException."""
    async def async_break_finally(v):
      Chain.break_()
    with self.assertRaises(QuentException) as cm:
      await Chain(aempty, 42).then(lambda v: v).finally_(async_break_finally).run()
    unittest.TestCase.assertIn(self, 'control flow', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# 6. SimpleFinallyAsyncWarningTests
# ---------------------------------------------------------------------------
class SimpleFinallyAsyncWarningTests(MyTestCase):
  """Verify RuntimeWarning for async finally on sync-only chain."""

  async def test_async_finally_on_sync_chain_warns(self):
    """no_async(True) with async finally_ issues RuntimeWarning."""
    with self.assertWarns(RuntimeWarning):
      Chain(42).no_async(True).then(lambda v: v).finally_(aempty).run()

  async def test_sync_finally_on_sync_chain_no_warning(self):
    """no_async(True) with sync finally_ emits no RuntimeWarning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(42).no_async(True).then(lambda v: v).finally_(lambda v: None).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      unittest.TestCase.assertEqual(self, len(runtime_warnings), 0)


# ---------------------------------------------------------------------------
# 7. SimpleFinallyNoAsyncTests
# ---------------------------------------------------------------------------
class SimpleFinallyNoAsyncTests(unittest.TestCase):
  """Verify no_async mode with finally_."""

  def test_no_async_with_finally(self):
    """no_async(True) with .then() and .finally_() — result correct, finally ran."""
    tracker = []
    result = (
      Chain(42)
      .no_async(True)
      .then(lambda v: v + 1)
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(result, 43)
    self.assertEqual(tracker, [42])

  def test_no_async_no_quent_warning(self):
    """no_async(True) with sync finally_ does not emit any warnings."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(42).no_async(True).then(lambda v: v).finally_(lambda v: None).run()
      self.assertEqual(len(w), 0)


# ---------------------------------------------------------------------------
# 8. SimpleFinallyReturnBreakTests
# ---------------------------------------------------------------------------
class SimpleFinallyReturnBreakTests(MyTestCase):
  """Verify finally_ runs after Return/Break signals from the chain body."""

  async def test_finally_runs_after_return(self):
    """Chain.return_() in .then() — result is the return value, finally ran."""
    tracker = []
    result = (
      Chain(42)
      .then(lambda v: Chain.return_(v * 2))
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    await self.assertEqual(result, 84)
    unittest.TestCase.assertEqual(self, tracker, [42])

  async def test_finally_runs_after_return_async(self):
    """Async variant: Chain.return_() in .then() — result is the return value, finally ran."""
    tracker = []
    result = (
      Chain(aempty, 42)
      .then(lambda v: Chain.return_(v * 2))
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    await self.assertEqual(result, 84)
    unittest.TestCase.assertEqual(self, tracker, [42])

  async def test_finally_runs_after_break_in_nested(self):
    """Chain.break_() raises QuentException outside iteration — finally still runs."""
    tracker = []
    with self.assertRaises(QuentException):
      Chain(42).then(lambda v: Chain.break_()).finally_(lambda v: tracker.append(v)).run()
    unittest.TestCase.assertEqual(self, tracker, [42])


# ---------------------------------------------------------------------------
# 9. SimpleFinallyWithCascadeTests
# ---------------------------------------------------------------------------
class SimpleFinallyWithCascadeTests(MyTestCase):
  """Verify Cascade mode + finally_ works correctly."""

  async def test_cascade_with_finally_sync(self):
    """Cascade returns root value; finally runs with root."""
    tracker = []
    result = (
      Cascade(42)
      .then(lambda v: v + 1)
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    await self.assertEqual(result, 42)
    unittest.TestCase.assertEqual(self, tracker, [42])

  async def test_cascade_with_finally_async(self):
    """Async Cascade returns root value; finally runs."""
    tracker = []
    result = (
      Cascade(aempty, 42)
      .then(lambda v: v + 1)
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    await self.assertEqual(result, 42)
    unittest.TestCase.assertEqual(self, tracker, [42])

  async def test_cascade_finally_receives_root(self):
    """The finally callback in cascade mode receives the original root value."""
    received = []
    for fn, ctx in self.with_fn():
      with ctx:
        received.clear()
        result = (
          Cascade(fn, 100)
          .then(lambda v: v + 50)
          .then(lambda v: v + 25)
          .finally_(lambda v: received.append(v))
          .run()
        )
        await self.assertEqual(result, 100)
        unittest.TestCase.assertEqual(self, received, [100])


if __name__ == '__main__':
  unittest.main()
