import unittest
import asyncio
from quent import Chain, Cascade


class ContextBasicTests(unittest.TestCase):
  def test_with_context_returns_self(self):
    c = Chain(1)
    self.assertIs(c.with_context(request_id="abc"), c)

  def test_context_accessible_during_execution(self):
    def check_context(v):
      ctx = Chain.get_context()
      return ctx.get('request_id')
    c = Chain(1).then(check_context).with_context(request_id="abc-123")
    self.assertEqual(c.run(), "abc-123")

  def test_context_empty_without_with_context(self):
    def check_context(v):
      ctx = Chain.get_context()
      return ctx
    c = Chain(1).then(check_context)
    self.assertEqual(c.run(), {})

  def test_context_multiple_keys(self):
    def check_context(v):
      ctx = Chain.get_context()
      return f"{ctx.get('user')}-{ctx.get('request')}"
    c = Chain(1).then(check_context).with_context(user="alice", request="req-1")
    self.assertEqual(c.run(), "alice-req-1")

  def test_context_additive(self):
    def check_context(v):
      ctx = Chain.get_context()
      return len(ctx)
    c = Chain(1).with_context(a=1).with_context(b=2).then(check_context)
    self.assertEqual(c.run(), 2)

  def test_context_clone_independent(self):
    c1 = Chain(1).with_context(key="original")
    c2 = c1.clone().with_context(key="clone")
    def get_key(v):
      return Chain.get_context().get('key')
    c1_result = c1.then(get_key).run()
    # Note: c1 was modified by .then(get_key), but c2 was cloned before that
    self.assertEqual(c1_result, "original")

  def test_context_reset_after_run(self):
    """Context should be cleaned up after sync chain execution."""
    c = Chain(1).then(lambda v: v).with_context(request_id="temp")
    c.run()
    # After run, context should be reset
    self.assertEqual(Chain.get_context(), {})


class ContextAsyncTests(unittest.IsolatedAsyncioTestCase):
  async def test_async_context_accessible(self):
    async def check_context(v):
      ctx = Chain.get_context()
      return ctx.get('request_id')
    c = Chain(1).then(check_context).with_context(request_id="async-123")
    result = await c.run()
    self.assertEqual(result, "async-123")

  async def test_async_context_reset_after_run(self):
    async def noop(v):
      return v
    c = Chain(1).then(noop).with_context(request_id="temp")
    await c.run()
    self.assertEqual(Chain.get_context(), {})

  async def test_concurrent_contexts_isolated(self):
    """Multiple concurrent chains should have isolated contexts."""
    results = []

    async def capture(v):
      ctx = Chain.get_context()
      await asyncio.sleep(0.01)
      results.append(ctx.get('id'))
      return v

    c1 = Chain(1).then(capture).with_context(id="first")
    c2 = Chain(2).then(capture).with_context(id="second")
    await asyncio.gather(c1.run(), c2.run())
    self.assertIn("first", results)
    self.assertIn("second", results)


if __name__ == '__main__':
  unittest.main()
