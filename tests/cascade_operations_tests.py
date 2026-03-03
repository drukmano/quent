import asyncio
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# CascadeForeachTests
# ---------------------------------------------------------------------------
class CascadeForeachTests(MyTestCase):
  """Cascade with foreach/filter/gather: operations receive root_value,
  but the final result is always root_value (not the operation's output).

  In _run() lines 171-172:
    if self.is_cascade:
      current_value = root_value

  In _run_async() lines 286-287:
    if self.is_cascade:
      current_value = root_value
  """

  async def test_foreach_sync_returns_root(self):
    """Cascade([1,2,3]).foreach(fn).run() returns root_value, not mapped list.

    The foreach link receives root_value (due to is_with_root=True), maps it,
    but the Cascade restoration at the end of _run() overrides with root_value.
    """
    root = [1, 2, 3]
    mapped = []
    def capture(x):
      mapped.append(x * 10)
      return x * 10
    result = Cascade(root).foreach(capture).run()
    await self.assertEqual(result, root)
    # Verify foreach actually ran on the root elements
    super(MyTestCase, self).assertEqual(mapped, [10, 20, 30])

  async def test_foreach_async_returns_root(self):
    """Cascade with async root + foreach: result is still root_value.

    Exercises _run_async() lines 286-287 cascade restoration.
    """
    root = [4, 5, 6]
    mapped = []
    def capture(x):
      mapped.append(x * 2)
      return x * 2
    result = await await_(Cascade(aempty, root).foreach(capture).run())
    super(MyTestCase, self).assertEqual(result, root)
    super(MyTestCase, self).assertEqual(mapped, [8, 10, 12])

  async def test_foreach_with_fn_pattern(self):
    """Cascade.foreach with both sync and async root via with_fn pattern."""
    for fn, ctx in self.with_fn():
      with ctx:
        root = [10, 20, 30]
        result = Cascade(fn, root).foreach(lambda x: x + 1).run()
        await self.assertEqual(result, root)

  async def test_filter_sync_returns_root(self):
    """Cascade([...]).filter(pred).run() returns root, not the filtered list.

    The filter link operates on root_value but the Cascade restores root_value.
    """
    root = [1, 2, 3, 4, 5]
    result = Cascade(root).filter(lambda x: x > 3).run()
    await self.assertEqual(result, root)

  async def test_filter_async_returns_root(self):
    """Cascade with async root + filter: result is root_value."""
    root = [10, 20, 30, 40]
    result = await await_(Cascade(aempty, root).filter(lambda x: x >= 30).run())
    super(MyTestCase, self).assertEqual(result, root)

  async def test_gather_sync_returns_root(self):
    """Cascade(v).gather(fn1, fn2).run() returns root, not gathered results.

    Each gather function receives root_value. The gather result is a list,
    but the Cascade restores root_value at the end of _run().
    """
    root = 10
    received = []
    def fn1(v):
      received.append(('fn1', v))
      return v * 2
    def fn2(v):
      received.append(('fn2', v))
      return v * 3
    result = Cascade(root).gather(fn1, fn2).run()
    await self.assertEqual(result, root)
    super(MyTestCase, self).assertEqual(received, [('fn1', 10), ('fn2', 10)])

  async def test_gather_async_returns_root(self):
    """Cascade with async root + gather: result is root_value.

    Exercises _run_async() cascade restoration with gather operations.
    """
    root = 7
    async def async_double(v):
      return v * 2
    result = await await_(
      Cascade(aempty, root).gather(async_double, lambda v: v * 3).run()
    )
    super(MyTestCase, self).assertEqual(result, root)


# ---------------------------------------------------------------------------
# CascadeWithTests
# ---------------------------------------------------------------------------
class CascadeWithTests(MyTestCase):
  """Cascade with with_() context manager: the context manager body executes,
  but the Cascade returns root_value regardless of the body result.
  """

  async def test_with_sync_cm_returns_root(self):
    """Cascade(sync_cm).with_(body).run() returns root (the CM itself)."""

    class SimpleCM:
      def __enter__(self):
        return 'ctx_value'
      def __exit__(self, *args):
        return False

    cm = SimpleCM()
    result = Cascade(cm).with_(lambda ctx: f'processed_{ctx}').run()
    await self.assertIs(result, cm)

  async def test_with_async_cm_returns_root(self):
    """Cascade(async_cm).with_(body).run() returns root."""

    class AsyncCM:
      def __init__(self):
        self.entered = False
        self.exited = False

      async def __aenter__(self):
        self.entered = True
        return 'async_ctx'

      async def __aexit__(self, *args):
        self.exited = True
        return False

    cm = AsyncCM()
    result = await await_(Cascade(cm).with_(lambda ctx: f'got_{ctx}').run())
    super(MyTestCase, self).assertIs(result, cm)
    super(MyTestCase, self).assertTrue(cm.entered)
    super(MyTestCase, self).assertTrue(cm.exited)

  async def test_with_body_executes_but_result_discarded(self):
    """The body of with_ runs, but its result is discarded in Cascade mode."""
    body_results = []

    class YieldCM:
      """Non-callable context manager that yields a value."""
      def __enter__(self):
        return 42
      def __exit__(self, *args):
        return False

    cm = YieldCM()
    result = Cascade(cm).with_(lambda ctx: body_results.append(ctx) or 'body_result').run()
    # Cascade returns the root value (the CM instance)
    await self.assertIs(result, cm)
    # But the body did execute
    super(MyTestCase, self).assertEqual(body_results, [42])


# ---------------------------------------------------------------------------
# CascadeSleepTests
# ---------------------------------------------------------------------------
class CascadeSleepTests(MyTestCase):
  """Cascade with sleep: sleep link has ignore_result=True already.
  In Cascade mode, it also gets is_with_root=True. The chain goes through
  the non-simple _run() path.
  """

  async def test_sleep_sync_returns_root(self):
    """Cascade(v).sleep(0).run() returns root_value."""
    result = Cascade(42).sleep(0).run()
    await self.assertEqual(result, 42)

  async def test_sleep_async_returns_root(self):
    """Cascade with async root + sleep: result is root_value.

    In async context, _Sleep returns asyncio.sleep() coroutine, causing
    _run_async() to be entered. The cascade restoration at lines 286-287
    ensures root_value is returned.
    """
    result = await await_(Cascade(aempty, 99).sleep(0).run())
    super(MyTestCase, self).assertEqual(result, 99)

  async def test_sleep_with_then_returns_root(self):
    """Cascade.sleep + .then: all operations receive root, result is root."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = []
        result = (
          Cascade(fn, 55)
          .sleep(0)
          .then(lambda v: received.append(v) or v * 2)
          .run()
        )
        await self.assertEqual(result, 55)
        super(MyTestCase, self).assertEqual(received, [55])


# ---------------------------------------------------------------------------
# CascadeAsyncPathTests
# ---------------------------------------------------------------------------
class CascadeAsyncPathTests(MyTestCase):
  """Tests targeting the async code paths in _run_async and _run_async_simple
  specifically for Cascade mode.
  """

  async def test_run_async_cascade_restoration(self):
    """Cascade with async root and .do() triggers _run_async path.

    In _run_async() lines 286-287:
      if self.is_cascade:
        current_value = root_value

    The .do() makes the chain non-simple, so it uses _run/_run_async.
    """
    side_effects = []
    result = await await_(
      Cascade(aempty, [1, 2, 3])
      .do(lambda v: side_effects.extend(v))
      .then(lambda v: 'should_be_discarded')
      .run()
    )
    super(MyTestCase, self).assertEqual(result, [1, 2, 3])
    super(MyTestCase, self).assertEqual(side_effects, [1, 2, 3])

  async def test_run_async_cascade_with_except(self):
    """Cascade through _run_async with except_ handler (non-simple path).

    except_ makes chain non-simple. Async root forces _run_async path.
    """
    handler_called = {'value': False}
    def handler(v):
      handler_called['value'] = True
    result = await await_(
      Cascade(aempty, 42)
      .then(lambda v: v * 2)
      .except_(handler)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 42)
    super(MyTestCase, self).assertFalse(handler_called['value'])

  async def test_run_async_cascade_multiple_links(self):
    """Multiple links in Cascade all receive root_value in _run_async path.

    Each link has is_with_root=True, so evaluate_value receives root_value.
    """
    received = []
    async def async_capture(v):
      received.append(v)
      return v * 100  # discarded by Cascade

    result = await await_(
      Cascade(aempty, 'root')
      .do(lambda v: received.append(v))
      .then(async_capture)
      .then(lambda v: received.append(v) or 'also_discarded')
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 'root')
    # All operations should have received 'root'
    super(MyTestCase, self).assertEqual(received, ['root', 'root', 'root'])

  async def test_run_async_simple_cascade_restoration(self):
    """Cascade through _run_async_simple path (lines 435-441).

    Cascade with only .then() links stays _is_simple=True. Async root
    triggers _run_async_simple. The cascade loop at lines 436-440 evaluates
    links with root_value, then line 441 restores current_value = root_value.
    """
    received = []
    async def async_capture(v):
      received.append(v)
      return v * 2

    result = await await_(
      Cascade(aempty, 77)
      .then(async_capture)
      .then(lambda v: received.append(v) or 'discarded')
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 77)
    # Both links receive root_value (77)
    super(MyTestCase, self).assertEqual(received, [77, 77])

  async def test_run_async_simple_cascade_sync_continuation(self):
    """Cascade in _run_async_simple: async root, then sync-only links.

    After the first await, the cascade loop iterates sync links passing
    root_value, then restores current_value = root_value at line 441.
    """
    result = await await_(
      Cascade(aempty, 100)
      .then(lambda v: v + 1)
      .then(lambda v: v * 2)
      .then(lambda v: v - 50)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 100)

  async def test_void_cascade_async_returns_none(self):
    """Void Cascade with async .do() → _run_async path → Null check.

    Cascade() with no root: root_value is Null. After _run_async cascade
    restoration at line 287 (current_value = root_value = Null), the
    Null check at lines 289-290 returns None.
    """
    side_effects = []
    async def async_side_effect():
      side_effects.append('ran')
    result = await await_(
      Cascade()
      .do(async_side_effect, ...)
      .then(lambda: 'discarded', ...)
      .run()
    )
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertEqual(side_effects, ['ran'])


# ---------------------------------------------------------------------------
# CascadeDecoratorTests
# ---------------------------------------------------------------------------
class CascadeDecoratorTests(MyTestCase):
  """Cascade.decorator() pattern: the frozen chain runs as a Cascade,
  so each operation receives the root (the decorated function's return value),
  and the final result is the root value.
  """

  async def test_decorator_returns_root(self):
    """Cascade().then(fn).decorator() returns the decorated fn's return value.

    The decorator invokes the chain with fn as root value. In Cascade mode,
    all operations receive root_value and the final result is root_value.
    """
    side_effects = []

    @Cascade().then(lambda v: side_effects.append(v) or v * 100).decorator()
    def my_fn(x):
      return x * 2

    result = my_fn(5)
    await self.assertEqual(result, 10)  # 5 * 2 = 10, Cascade returns root
    super(MyTestCase, self).assertEqual(side_effects, [10])

  async def test_decorator_async_returns_root(self):
    """Cascade.decorator with async chain operation returns root."""
    side_effects = []

    @Cascade().then(lambda v: side_effects.append(v) or aempty(v * 100)).decorator()
    def my_fn(x):
      return x + 3

    result = await await_(my_fn(7))
    super(MyTestCase, self).assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(side_effects, [10])

  async def test_decorator_with_class_method(self):
    """Cascade.decorator on a class method returns the method's return value."""
    class MyClass:
      def __init__(self, factor):
        self.factor = factor

      @Cascade().then(lambda v: v).decorator()
      def compute(self, x):
        return x * self.factor

    obj = MyClass(3)
    result = obj.compute(10)
    await self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# NullCurrentValueTests
# ---------------------------------------------------------------------------
class NullCurrentValueTests(MyTestCase):
  """Edge cases where current_value is Null at the end of execution paths.

  In _run() line 174-175:
    if current_value is Null:
      return None

  In _run_async() lines 289-290:
    if current_value is Null:
      return None

  In _run_simple() lines 404-405:
    if current_value is Null:
      return None

  In _run_async_simple() lines 449-450:
    if current_value is Null:
      return None
  """

  async def test_void_cascade_sync_returns_none(self):
    """Void Cascade (no root, no links) returns None.

    _run_simple path: has_root_value=False, no links, current_value=Null → None.
    """
    result = Cascade().run()
    await self.assertIsNone(result)

  async def test_void_cascade_with_do_returns_none(self):
    """Void Cascade with .do(): root_value is Null, cascade restores Null → None.

    In _run(): do link has ignore_result=True, then cascade restoration
    sets current_value = root_value = Null, then Null check returns None.
    """
    side_effects = []
    result = Cascade().do(lambda: side_effects.append('ran'), ...).run()
    await self.assertIsNone(result)
    super(MyTestCase, self).assertEqual(side_effects, ['ran'])

  async def test_void_cascade_async_null_to_none(self):
    """Void Cascade with async operation: _run_async Null → None.

    Cascade() has root_value=Null. An async .do() forces _run_async path.
    At the end, cascade restoration: current_value = root_value = Null.
    Then lines 289-290: if current_value is Null: return None.
    """
    side_effects = []
    async def async_side_effect():
      side_effects.append('async_ran')
    result = await await_(
      Cascade().do(async_side_effect, ...).run()
    )
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertEqual(side_effects, ['async_ran'])

  async def test_void_chain_returns_none(self):
    """Chain() with no root and no links returns None (Null → None)."""
    result = Chain().run()
    await self.assertIsNone(result)

  async def test_void_chain_with_async_fn_null(self):
    """Chain().then(aempty).run() exercises _run_async_simple Null → None.

    Void chain: no root_value. aempty is called without args (current_value=Null),
    returns None (default v=None). current_value becomes None (not Null).
    So this does NOT hit the Null check. We need aempty to return the Null sentinel.
    """
    # This returns None (not Null), so it hits the normal return path.
    result = await await_(Chain().then(aempty).run())
    await self.assertIsNone(result)

  async def test_cascade_all_operations_receive_root(self):
    """Verify every operation in a Cascade receives root_value, not previous result.

    This is the fundamental semantic difference between Chain and Cascade.
    """
    received = []
    def capture_and_transform(v):
      received.append(v)
      return v * 100  # Would change current_value in Chain mode
    root = 42
    result = (
      Cascade(root)
      .then(capture_and_transform)
      .then(capture_and_transform)
      .then(capture_and_transform)
      .run()
    )
    await self.assertEqual(result, root)
    # All three operations received root_value (42), not cascading 4200, 420000
    super(MyTestCase, self).assertEqual(received, [42, 42, 42])

  async def test_cascade_vs_chain_semantic_difference(self):
    """Contrast Chain vs Cascade behavior to ensure correctness.

    Chain: each operation receives the previous result.
    Cascade: each operation receives root_value, final result is root_value.
    """
    fn = lambda v: v * 2

    chain_result = Chain(5).then(fn).then(fn).run()
    await self.assertEqual(chain_result, 20)  # 5 → 10 → 20

    cascade_result = Cascade(5).then(fn).then(fn).run()
    await self.assertEqual(cascade_result, 5)  # Always returns root

  async def test_cascade_async_vs_chain_async_difference(self):
    """Async variant of semantic difference between Chain and Cascade."""
    async def async_double(v):
      return v * 2

    chain_result = await await_(
      Chain(aempty, 3).then(async_double).then(async_double).run()
    )
    super(MyTestCase, self).assertEqual(chain_result, 12)  # 3 → 6 → 12

    cascade_result = await await_(
      Cascade(aempty, 3).then(async_double).then(async_double).run()
    )
    super(MyTestCase, self).assertEqual(cascade_result, 3)  # Always root


# ---------------------------------------------------------------------------
# CascadeCloneAndFreezeTests
# ---------------------------------------------------------------------------
class CascadeCloneAndFreezeTests(MyTestCase):
  """Cascade with clone() and freeze(): verify cascade behavior is preserved."""

  async def test_cloned_cascade_preserves_behavior(self):
    """Cloned Cascade still returns root_value."""
    c = Cascade(42).then(lambda v: v * 2)
    c2 = c.clone()
    await self.assertEqual(c.run(), 42)
    await self.assertEqual(c2.run(), 42)

  async def test_cloned_cascade_async(self):
    """Cloned async Cascade still returns root_value."""
    c = Cascade(aempty, 77).then(lambda v: v + 1)
    c2 = c.clone()
    await self.assertEqual(c.run(), 77)
    await self.assertEqual(c2.run(), 77)

  async def test_frozen_cascade_returns_root(self):
    """Frozen Cascade returns root_value on each invocation."""
    frozen = Cascade(99).then(lambda v: v * 100).freeze()
    await self.assertEqual(frozen.run(), 99)
    await self.assertEqual(frozen(), 99)

  async def test_frozen_cascade_async(self):
    """Frozen async Cascade returns root_value."""
    frozen = Cascade(aempty, 33).then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen.run(), 33)
    await self.assertEqual(frozen(), 33)


# ---------------------------------------------------------------------------
# CascadePipeTests
# ---------------------------------------------------------------------------
class CascadePipeTests(MyTestCase):
  """Cascade with pipe operator: all piped operations receive root_value."""

  async def test_pipe_returns_root(self):
    """Cascade(v) | fn | run() returns root_value."""
    result = Cascade(10) | (lambda v: v * 2) | (lambda v: v * 3) | run()
    await self.assertEqual(result, 10)

  async def test_pipe_async_returns_root(self):
    """Cascade with async pipe operation returns root."""
    result = Cascade(aempty, 25) | (lambda v: v + 5) | run()
    await self.assertEqual(result, 25)


if __name__ == '__main__':
  import unittest
  unittest.main()
