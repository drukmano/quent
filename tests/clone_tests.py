import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase
from quent import Chain


class CloneBasicTests(TestCase):
  """Test basic clone() functionality."""

  def test_clone_empty_chain(self):
    """Cloning a void chain (no root, no links) produces a working chain."""
    c = Chain()
    c2 = c.clone()
    self.assertIsNone(c2.run())

  def test_clone_root_only(self):
    """Cloning a chain with only a root value."""
    c = Chain(42)
    c2 = c.clone()
    self.assertEqual(c2.run(), 42)

  def test_clone_produces_same_result(self):
    """Clone produces the same result as the original."""
    c = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3)
    c2 = c.clone()
    self.assertEqual(c.run(), c2.run())
    self.assertEqual(c2.run(), 6)

  def test_clone_with_do(self):
    """Clone preserves do() operations."""
    side_effects = []
    c = Chain(10).do(lambda v: side_effects.append(v)).then(lambda v: v + 5)
    c2 = c.clone()
    side_effects.clear()
    self.assertEqual(c2.run(), 15)
    self.assertEqual(side_effects, [10])

  def test_clone_with_literal_values(self):
    """Clone works with literal (non-callable) values in then()."""
    c = Chain(1).then(99)
    c2 = c.clone()
    self.assertEqual(c2.run(), 99)

  def test_clone_with_explicit_args(self):
    """Clone preserves explicit args and kwargs on links."""
    def add(a, b, extra=0):
      return a + b + extra
    c = Chain(1).then(add, 2, 3, extra=10)
    c2 = c.clone()
    self.assertEqual(c2.run(), 15)

  def test_clone_multiple_times(self):
    """A chain can be cloned multiple times independently."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c.clone()
    self.assertEqual(c2.run(), 2)
    self.assertEqual(c3.run(), 2)

  def test_clone_of_clone(self):
    """A clone can be cloned."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    self.assertEqual(c3.run(), 2)


class CloneIndependenceTests(TestCase):
  """Test that clone is independent from the original."""

  def test_modify_original_after_clone(self):
    """Adding to the original after cloning does not affect the clone."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    self.assertEqual(c2.run(), 2)
    self.assertEqual(c.run(), 200)

  def test_modify_clone_after_clone(self):
    """Adding to the clone does not affect the original."""
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(lambda v: v * 100)
    self.assertEqual(c.run(), 2)
    self.assertEqual(c2.run(), 200)

  def test_clone_independent_execution_state(self):
    """Execution state (result, temp_args) is not shared."""
    c = Chain(1).then(lambda v: v + 1)
    self.assertEqual(c.run(), 2)
    c2 = c.clone()
    self.assertEqual(c2.run(), 2)

  def test_clone_is_not_nested(self):
    """Clone should not be marked as nested even if original is used nested."""
    inner = Chain(1).then(lambda v: v + 1)
    outer = Chain(0).then(inner)
    # inner is now marked as nested
    c2 = inner.clone()
    # clone should not be nested, so it can be run directly
    self.assertEqual(c2.run(), 2)


class CloneRootValueTests(TestCase):
  """Test cloning with various root value configurations."""

  def test_clone_run_with_override_root(self):
    """A cloned void chain can be run with a root value override."""
    c = Chain().then(lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(c2.run(5), 10)

  def test_clone_preserves_root_callable(self):
    """Clone preserves a callable root value."""
    c = Chain(lambda: 42).then(lambda v: v + 1)
    c2 = c.clone()
    self.assertEqual(c2.run(), 43)

  def test_clone_with_root_args(self):
    """Clone preserves root value args/kwargs."""
    def root_fn(a, b):
      return a + b
    c = Chain(root_fn, 3, 7).then(lambda v: v * 2)
    c2 = c.clone()
    self.assertEqual(c2.run(), 20)


class CloneExceptFinallyTests(TestCase):
  """Test cloning with exception handlers and finally callbacks."""

  def test_clone_preserves_except(self):
    """Clone preserves except_ handlers."""
    handler_called = {"original": False, "clone": False}

    def failing_fn(v):
      raise ValueError("test")

    c = Chain(1).then(failing_fn).except_(
      lambda v: handler_called.update({"original": True}),
      reraise=False
    )
    c2 = c.clone()
    # Run original
    c.run()
    self.assertTrue(handler_called["original"])
    # Run clone
    handler_called["original"] = False
    c2.run()
    # The handler callable is shared, so it updates the same dict
    self.assertTrue(handler_called["original"])

  def test_clone_preserves_except_with_specific_exceptions(self):
    """Clone preserves exception type filtering in except_."""
    caught = {"value": None}
    def handler(v):
      caught["value"] = "caught"

    c = Chain(1).then(lambda v: (_ for _ in ()).throw(ValueError("x"))).except_(
      handler, exceptions=[ValueError], reraise=False
    )
    c2 = c.clone()
    caught["value"] = None

    def raise_value_error(v):
      raise ValueError("test")

    c3 = Chain(1).then(raise_value_error).except_(
      handler, exceptions=[ValueError], reraise=False
    )
    c4 = c3.clone()
    caught["value"] = None
    c4.run()
    self.assertEqual(caught["value"], "caught")

  def test_clone_preserves_finally(self):
    """Clone preserves finally_ callback."""
    calls = []

    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: calls.append("finally"))
    c2 = c.clone()

    calls.clear()
    self.assertEqual(c2.run(), 2)
    self.assertEqual(calls, ["finally"])

  def test_clone_except_and_finally_together(self):
    """Clone preserves both except_ and finally_ callbacks."""
    log = []

    def raise_err(v):
      raise RuntimeError("boom")

    c = (
      Chain(1)
      .then(raise_err)
      .except_(lambda v: log.append("except"), reraise=False)
      .finally_(lambda v: log.append("finally"))
    )
    c2 = c.clone()
    log.clear()
    c2.run()
    self.assertIn("except", log)
    self.assertIn("finally", log)


class CloneNestedChainTests(TestCase):
  """Test cloning chains that contain nested chains."""

  def test_clone_with_nested_chain(self):
    """Clone works when the chain contains a nested Chain as a link value."""
    inner = Chain().then(lambda v: v * 10)
    outer = Chain(2).then(inner)
    c2 = outer.clone()
    # The nested chain callable (inner) is shared, which is fine
    self.assertEqual(c2.run(), 20)

  def test_clone_with_multiple_nested(self):
    """Clone with multiple nested chains."""
    step1 = Chain().then(lambda v: v + 1)
    step2 = Chain().then(lambda v: v * 3)
    c = Chain(5).then(step1).then(step2)
    c2 = c.clone()
    self.assertEqual(c2.run(), 18)


class CloneAsyncTests(IsolatedAsyncioTestCase):
  """Test clone with async operations."""

  async def test_clone_async_chain(self):
    """Clone works with async callables."""
    async def async_add(v):
      return v + 10

    c = Chain(5).then(async_add)
    c2 = c.clone()
    result = await c2.run()
    self.assertEqual(result, 15)

  async def test_clone_async_root(self):
    """Clone works with an async root callable."""
    async def async_root():
      return 42

    c = Chain(async_root).then(lambda v: v + 1)
    c2 = c.clone()
    result = await c2.run()
    self.assertEqual(result, 43)

  async def test_clone_async_independence(self):
    """Original and clone can be executed concurrently without interference."""
    async def slow_add(v):
      await asyncio.sleep(0.01)
      return v + 1

    c = Chain(1).then(slow_add).then(slow_add)
    c2 = c.clone()

    result1, result2 = await asyncio.gather(
      c.run(),
      c2.run(),
    )
    self.assertEqual(result1, 3)
    self.assertEqual(result2, 3)
