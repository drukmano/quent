"""Loop parity tests for _run_simple() / _run_async_simple().

The _run_simple() method has 4 specialized inner loops that handle every
combination of is_sync x is_cascade.  There are also 2 loops in
_run_async_simple() (cascade vs chain).  The 6 loops are:

In _run_simple():
  Loop 1  (async-capable, Chain):   not is_sync and not is_cascade
  Loop 2a (sync-only,     Cascade): is_sync and is_cascade
  Loop 2b (sync-only,     Chain):   is_sync and not is_cascade
  Loop 3  (async-capable, Cascade): not is_sync and is_cascade

In _run_async_simple():
  Loop 5  (async, Cascade): is_cascade
  Loop 6  (async, Chain):   not is_cascade

These tests run identical chain logic through all applicable loops and
assert the same result.  If someone modifies one loop without updating
the others, these tests catch the discrepancy.

Loop selection:
  Loop 1    -> Chain,   no_async=False, all sync callbacks
  Loop 1->6 -> Chain,   no_async=False, at least one async callback
  Loop 2a   -> Cascade, no_async=True,  all sync callbacks
  Loop 2b   -> Chain,   no_async=True,  all sync callbacks
  Loop 3    -> Cascade, no_async=False, all sync callbacks
  Loop 3->5 -> Cascade, no_async=False, at least one async callback

Cascade chains using only .then() remain _is_simple=True because the
Link constructor sets is_with_root=False by default, and _then() checks
is_with_root BEFORE the cascade code sets it to True.
"""
import unittest
import asyncio
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_chain(cls, root, callbacks, no_async=False):
  """Build a chain of the given class with root value and callback list."""
  c = cls(root) if root is not _SENTINEL else cls()
  for cb in callbacks:
    c = c.then(cb)
  if no_async:
    c = c.no_async(True)
  return c


def build_chain_with_finally(cls, root, callbacks, finally_cb, no_async=False):
  """Build a chain with .then() callbacks and a .finally_() handler."""
  c = cls(root) if root is not _SENTINEL else cls()
  for cb in callbacks:
    c = c.then(cb)
  c = c.finally_(finally_cb)
  if no_async:
    c = c.no_async(True)
  return c


_SENTINEL = object()


# ---------------------------------------------------------------------------
# LoopParitySingleCallbackTests
# ---------------------------------------------------------------------------
class LoopParitySingleCallbackTests(MyTestCase):
  """Run a single .then() callback through all 6 loop paths."""

  async def _run_all_paths(self, root, callbacks, expected_chain, expected_cascade):
    """Run callbacks through all 6 loop paths, assert expected results."""
    # Loop 2b: sync Chain (is_sync=True, is_cascade=False)
    c = build_chain(Chain, root, callbacks, no_async=True)
    result_2b = c.run()
    unittest.TestCase.assertEqual(self, result_2b, expected_chain, 'Loop 2b (sync chain)')

    # Loop 2a: sync Cascade (is_sync=True, is_cascade=True)
    c = build_chain(Cascade, root, callbacks, no_async=True)
    result_2a = c.run()
    unittest.TestCase.assertEqual(self, result_2a, expected_cascade, 'Loop 2a (sync cascade)')

    # Loop 1: async-capable Chain, sync callbacks (stays in loop 1)
    c = build_chain(Chain, root, callbacks, no_async=False)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected_chain, 'Loop 1 (async-capable chain, sync cbs)')

    # Loop 3: async-capable Cascade, sync callbacks (stays in loop 3)
    c = build_chain(Cascade, root, callbacks, no_async=False)
    result_3 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3, expected_cascade, 'Loop 3 (async-capable cascade, sync cbs)')

    # Loop 1->6: async Chain (inject aempty to force async transition)
    c = build_chain(Chain, root, callbacks, no_async=False)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected_chain, 'Loop 1->6 (async chain)')

    # Loop 3->5: async Cascade (inject aempty to force async transition)
    c = build_chain(Cascade, root, callbacks, no_async=False)
    c = c.then(aempty)
    result_3_5 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3_5, expected_cascade, 'Loop 3->5 (async cascade)')

  async def test_identity_parity(self):
    """Chain(42).then(identity) -> 42 through all loops."""
    await self._run_all_paths(42, [lambda v: v], 42, 42)

  async def test_transform_parity(self):
    """Chain(10).then(v*3) -> 30; Cascade(10).then(v*3) -> 10."""
    await self._run_all_paths(10, [lambda v: v * 3], 30, 10)

  async def test_string_transform_parity(self):
    """Chain('hello').then(upper) -> 'HELLO'; Cascade -> 'hello'."""
    await self._run_all_paths('hello', [lambda v: v.upper()], 'HELLO', 'hello')

  async def test_negate_parity(self):
    """Chain(7).then(-v) -> -7 through all loops."""
    await self._run_all_paths(7, [lambda v: -v], -7, 7)

  async def test_list_append_parity(self):
    """Chain([1,2]).then(v+[3]) -> [1,2,3]; Cascade -> [1,2]."""
    await self._run_all_paths([1, 2], [lambda v: v + [3]], [1, 2, 3], [1, 2])

  async def test_none_root_transform_parity(self):
    """Chain(None).then(lambda v: 99) -> 99 through all loops."""
    await self._run_all_paths(None, [lambda v: 99], 99, None)


# ---------------------------------------------------------------------------
# LoopParityMultiCallbackTests
# ---------------------------------------------------------------------------
class LoopParityMultiCallbackTests(MyTestCase):
  """Run a chain of 3+ .then() callbacks through all 6 loops."""

  async def _run_all_paths(self, root, callbacks, expected_chain, expected_cascade):
    """Run callbacks through all 6 loop paths, assert expected results."""
    # Loop 2b
    c = build_chain(Chain, root, callbacks, no_async=True)
    result_2b = c.run()
    unittest.TestCase.assertEqual(self, result_2b, expected_chain, 'Loop 2b (sync chain)')

    # Loop 2a
    c = build_chain(Cascade, root, callbacks, no_async=True)
    result_2a = c.run()
    unittest.TestCase.assertEqual(self, result_2a, expected_cascade, 'Loop 2a (sync cascade)')

    # Loop 1
    c = build_chain(Chain, root, callbacks, no_async=False)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected_chain, 'Loop 1 (async-capable chain, sync cbs)')

    # Loop 3
    c = build_chain(Cascade, root, callbacks, no_async=False)
    result_3 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3, expected_cascade, 'Loop 3 (async-capable cascade, sync cbs)')

    # Loop 1->6
    c = build_chain(Chain, root, callbacks, no_async=False)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected_chain, 'Loop 1->6 (async chain)')

    # Loop 3->5
    c = build_chain(Cascade, root, callbacks, no_async=False)
    c = c.then(aempty)
    result_3_5 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3_5, expected_cascade, 'Loop 3->5 (async cascade)')

  async def test_chained_arithmetic_parity(self):
    """Chain(1).then(+1).then(*2).then(+10) -> 14; Cascade -> 1."""
    await self._run_all_paths(
      1,
      [lambda v: v + 1, lambda v: v * 2, lambda v: v + 10],
      14, 1,
    )

  async def test_chained_string_ops_parity(self):
    """Chain('a').then(+'b').then(+'c').then(upper) -> 'ABC'; Cascade -> 'a'."""
    await self._run_all_paths(
      'a',
      [lambda v: v + 'b', lambda v: v + 'c', lambda v: v.upper()],
      'ABC', 'a',
    )

  async def test_five_step_chain_parity(self):
    """5-step transformation: 2 -> ((2+3)*2-1)//2+100 = 104; Cascade -> 2."""
    await self._run_all_paths(
      2,
      [
        lambda v: v + 3,   # 5
        lambda v: v * 2,   # 10
        lambda v: v - 1,   # 9
        lambda v: v // 2,  # 4
        lambda v: v + 100, # 104
      ],
      104, 2,
    )

  async def test_four_step_string_parity(self):
    """4-step string chain: 'foo' -> 'FOO!!__X'; Cascade -> 'foo'."""
    await self._run_all_paths(
      'foo',
      [
        lambda v: v.upper(),    # 'FOO'
        lambda v: v + '!!',     # 'FOO!!'
        lambda v: v + '__',     # 'FOO!!__'
        lambda v: v + 'X',      # 'FOO!!__X'
      ],
      'FOO!!__X', 'foo',
    )

  async def test_three_step_list_parity(self):
    """3-step list chain: [] -> [1,2,3]; Cascade -> []."""
    await self._run_all_paths(
      [],
      [
        lambda v: v + [1],
        lambda v: v + [2],
        lambda v: v + [3],
      ],
      [1, 2, 3], [],
    )


# ---------------------------------------------------------------------------
# LoopParityAsyncTransitionTests
# ---------------------------------------------------------------------------
class LoopParityAsyncTransitionTests(MyTestCase):
  """Test chains that start sync then hit an async callback mid-chain.

  These specifically test the jump from loop 1->6 and loop 3->5, comparing
  results against the fully-sync versions (loops 2b, 2a).
  """

  async def test_async_mid_chain_parity(self):
    """Async callback in the middle of a chain matches sync equivalent.

    Chain(5).then(+1).then(aempty).then(*2) = 12  (loop 1->6)
    Chain(5).no_async(True).then(+1).then(empty).then(*2) = 12  (loop 2b)
    """
    # sync chain (loop 2b)
    result_sync = Chain(5).no_async(True).then(lambda v: v + 1).then(empty).then(lambda v: v * 2).run()
    unittest.TestCase.assertEqual(self, result_sync, 12, 'sync chain')

    # async chain — aempty mid-chain forces loop 1->6
    result_async = await await_(
      Chain(5).then(lambda v: v + 1).then(aempty).then(lambda v: v * 2).run()
    )
    unittest.TestCase.assertEqual(self, result_async, 12, 'async chain (loop 1->6)')

    # sync cascade (loop 2a)
    result_sync_casc = Cascade(5).no_async(True).then(lambda v: v + 1).then(empty).then(lambda v: v * 2).run()
    unittest.TestCase.assertEqual(self, result_sync_casc, 5, 'sync cascade')

    # async cascade — aempty mid-chain forces loop 3->5
    result_async_casc = await await_(
      Cascade(5).then(lambda v: v + 1).then(aempty).then(lambda v: v * 2).run()
    )
    unittest.TestCase.assertEqual(self, result_async_casc, 5, 'async cascade (loop 3->5)')

  async def test_async_first_callback_parity(self):
    """Async callback is the first .then() — should match sync."""
    # sync
    result_sync = Chain(10).no_async(True).then(empty).then(lambda v: v + 5).run()
    unittest.TestCase.assertEqual(self, result_sync, 15, 'sync chain')

    # async — aempty as first .then() triggers transition
    result_async = await await_(
      Chain(10).then(aempty).then(lambda v: v + 5).run()
    )
    unittest.TestCase.assertEqual(self, result_async, 15, 'async chain first callback')

  async def test_async_last_callback_parity(self):
    """Async callback is the last .then() — should match sync."""
    result_sync = Chain(10).no_async(True).then(lambda v: v + 5).then(empty).run()
    unittest.TestCase.assertEqual(self, result_sync, 15, 'sync chain')

    result_async = await await_(
      Chain(10).then(lambda v: v + 5).then(aempty).run()
    )
    unittest.TestCase.assertEqual(self, result_async, 15, 'async chain last callback')

  async def test_cascade_async_mid_chain_parity(self):
    """Cascade with async callback mid-chain produces same result as sync."""
    tracker_sync = []
    tracker_async = []

    # Sync cascade (loop 2a) — side effects tracked
    Cascade(99).no_async(True).then(
      lambda v: tracker_sync.append(v)
    ).then(
      lambda v: tracker_sync.append(v * 2)
    ).run()
    unittest.TestCase.assertEqual(self, tracker_sync, [99, 198], 'sync cascade side effects')

    # Async cascade (loop 3->5) — aempty forces transition
    await await_(
      Cascade(99).then(
        lambda v: tracker_async.append(v)
      ).then(aempty).then(
        lambda v: tracker_async.append(v * 2)
      ).run()
    )
    unittest.TestCase.assertEqual(self, tracker_async, [99, 198], 'async cascade side effects')

  async def test_multiple_async_callbacks_parity(self):
    """Multiple async callbacks interspersed — same result as all-sync."""
    result_sync = Chain(1).no_async(True).then(
      lambda v: v + 1
    ).then(empty).then(
      lambda v: v * 3
    ).then(empty).then(
      lambda v: v - 2
    ).run()
    unittest.TestCase.assertEqual(self, result_sync, 4, 'sync chain')

    result_async = await await_(
      Chain(1).then(
        lambda v: v + 1
      ).then(aempty).then(
        lambda v: v * 3
      ).then(aempty).then(
        lambda v: v - 2
      ).run()
    )
    unittest.TestCase.assertEqual(self, result_async, 4, 'async chain multi-async')

  async def test_async_transition_preserves_value_type(self):
    """Verify that the async transition does not alter the type of the value."""
    result_sync = Chain({'a': 1}).no_async(True).then(
      lambda v: {**v, 'b': 2}
    ).then(empty).run()

    result_async = await await_(
      Chain({'a': 1}).then(
        lambda v: {**v, 'b': 2}
      ).then(aempty).run()
    )

    unittest.TestCase.assertEqual(self, result_sync, {'a': 1, 'b': 2}, 'sync dict')
    unittest.TestCase.assertEqual(self, result_async, {'a': 1, 'b': 2}, 'async dict')
    unittest.TestCase.assertEqual(self, type(result_sync), type(result_async), 'type parity')


# ---------------------------------------------------------------------------
# LoopParityRootOverrideTests
# ---------------------------------------------------------------------------
class LoopParityRootOverrideTests(MyTestCase):
  """Test .run(override_value) through all loops."""

  async def _run_all_paths_override(self, override, callbacks, expected_chain, expected_cascade):
    """Run callbacks through all 6 loop paths with root override."""
    # Loop 2b: sync Chain
    c = build_chain(Chain, _SENTINEL, callbacks, no_async=True)
    result_2b = c.run(override)
    unittest.TestCase.assertEqual(self, result_2b, expected_chain, 'Loop 2b (sync chain override)')

    # Loop 2a: sync Cascade
    c = build_chain(Cascade, _SENTINEL, callbacks, no_async=True)
    result_2a = c.run(override)
    unittest.TestCase.assertEqual(self, result_2a, expected_cascade, 'Loop 2a (sync cascade override)')

    # Loop 1: async-capable Chain, sync callbacks
    c = build_chain(Chain, _SENTINEL, callbacks, no_async=False)
    result_1 = await await_(c.run(override))
    unittest.TestCase.assertEqual(self, result_1, expected_chain, 'Loop 1 (async-capable chain override)')

    # Loop 3: async-capable Cascade, sync callbacks
    c = build_chain(Cascade, _SENTINEL, callbacks, no_async=False)
    result_3 = await await_(c.run(override))
    unittest.TestCase.assertEqual(self, result_3, expected_cascade, 'Loop 3 (async-capable cascade override)')

    # Loop 1->6: async Chain
    c = build_chain(Chain, _SENTINEL, callbacks, no_async=False)
    c = c.then(aempty)
    result_1_6 = await await_(c.run(override))
    unittest.TestCase.assertEqual(self, result_1_6, expected_chain, 'Loop 1->6 (async chain override)')

    # Loop 3->5: async Cascade
    c = build_chain(Cascade, _SENTINEL, callbacks, no_async=False)
    c = c.then(aempty)
    result_3_5 = await await_(c.run(override))
    unittest.TestCase.assertEqual(self, result_3_5, expected_cascade, 'Loop 3->5 (async cascade override)')

  async def test_root_override_parity(self):
    """Chain().then(v*2).run(21) -> 42 through all Chain loops."""
    await self._run_all_paths_override(21, [lambda v: v * 2], 42, 21)

  async def test_root_override_with_multi_then_parity(self):
    """Multiple .then() with root override."""
    await self._run_all_paths_override(
      5,
      [lambda v: v + 1, lambda v: v * 10, lambda v: v - 3],
      57, 5,
    )

  async def test_root_override_string_parity(self):
    """String root override through all loops."""
    await self._run_all_paths_override(
      'test',
      [lambda v: v.upper(), lambda v: v + '!'],
      'TEST!', 'test',
    )

  async def test_root_override_none_parity(self):
    """None as root override through all loops."""
    await self._run_all_paths_override(
      None,
      [lambda v: 42],
      42, None,
    )

  async def test_root_override_callable_parity(self):
    """Callable root override — root is evaluated, then callbacks use result."""
    # When root is a callable, it is called with no args (Null -> calls v())
    # Loop 2b: sync Chain
    c = Chain().no_async(True).then(lambda v: v + 10)
    result_2b = c.run(lambda: 5)
    unittest.TestCase.assertEqual(self, result_2b, 15, 'Loop 2b callable override')

    # Loop 2a: sync Cascade
    c = Cascade().no_async(True).then(lambda v: v + 10)
    result_2a = c.run(lambda: 5)
    unittest.TestCase.assertEqual(self, result_2a, 5, 'Loop 2a callable override')

    # Loop 1
    c = Chain().then(lambda v: v + 10)
    result_1 = await await_(c.run(lambda: 5))
    unittest.TestCase.assertEqual(self, result_1, 15, 'Loop 1 callable override')

    # Loop 3
    c = Cascade().then(lambda v: v + 10)
    result_3 = await await_(c.run(lambda: 5))
    unittest.TestCase.assertEqual(self, result_3, 5, 'Loop 3 callable override')

    # Loop 1->6
    c = Chain().then(lambda v: v + 10).then(aempty)
    result_1_6 = await await_(c.run(lambda: 5))
    unittest.TestCase.assertEqual(self, result_1_6, 15, 'Loop 1->6 callable override')

    # Loop 3->5
    c = Cascade().then(lambda v: v + 10).then(aempty)
    result_3_5 = await await_(c.run(lambda: 5))
    unittest.TestCase.assertEqual(self, result_3_5, 5, 'Loop 3->5 callable override')


# ---------------------------------------------------------------------------
# LoopParityVoidChainTests
# ---------------------------------------------------------------------------
class LoopParityVoidChainTests(MyTestCase):
  """Test chains with no root value through all loops."""

  async def _run_all_paths_void(self, callbacks, expected_chain, expected_cascade):
    """Run void-root callbacks through all 6 loop paths."""
    # Loop 2b: sync Chain
    c = build_chain(Chain, _SENTINEL, callbacks, no_async=True)
    result_2b = c.run()
    unittest.TestCase.assertEqual(self, result_2b, expected_chain, 'Loop 2b (sync void chain)')

    # Loop 2a: sync Cascade
    c = build_chain(Cascade, _SENTINEL, callbacks, no_async=True)
    result_2a = c.run()
    unittest.TestCase.assertEqual(self, result_2a, expected_cascade, 'Loop 2a (sync void cascade)')

    # Loop 1: async-capable Chain
    c = build_chain(Chain, _SENTINEL, callbacks, no_async=False)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected_chain, 'Loop 1 (async-capable void chain)')

    # Loop 3: async-capable Cascade
    c = build_chain(Cascade, _SENTINEL, callbacks, no_async=False)
    result_3 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3, expected_cascade, 'Loop 3 (async-capable void cascade)')

    # Loop 1->6: async Chain
    c = build_chain(Chain, _SENTINEL, callbacks, no_async=False)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected_chain, 'Loop 1->6 (async void chain)')

    # Loop 3->5: async Cascade
    c = build_chain(Cascade, _SENTINEL, callbacks, no_async=False)
    c = c.then(aempty)
    result_3_5 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3_5, expected_cascade, 'Loop 3->5 (async void cascade)')

  async def test_void_chain_parity(self):
    """Chain().then(lambda: 42) -> 42 through all loops.

    For void cascades, the root is Null throughout, so the result is None.
    """
    await self._run_all_paths_void(
      [lambda: 42],
      42, None,
    )

  async def test_void_chain_multi_then_parity(self):
    """Chain().then(lambda: 10).then(lambda v: v + 5) -> 15.

    For void Cascade, all callbacks receive Null (root_value), which means
    evaluate_value calls link.v() with no args. So multi-step void Cascade
    tests must use no-arg callbacks. We test Chain and Cascade separately.
    """
    # Chain paths only (void cascade multi-step with arg-requiring callbacks
    # would fail because root_value stays Null)
    for label, no_async, inject_async in [
      ('2b', True, False),
      ('1', False, False),
      ('1->6', False, True),
    ]:
      with self.subTest(loop=label):
        c = Chain()
        c = c.then(lambda: 10).then(lambda v: v + 5)
        if inject_async:
          c = c.then(aempty)
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 15, f'void chain multi in loop {label}')

    # Cascade paths — all callbacks receive Null (no args), result is always None
    for label, no_async, inject_async in [
      ('2a', True, False),
      ('3', False, False),
      ('3->5', False, True),
    ]:
      with self.subTest(loop=label):
        tracker = []
        c = Cascade()
        c = c.then(lambda: tracker.append('a')).then(lambda: tracker.append('b'))
        if inject_async:
          c = c.then(aempty)
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertIsNone(self, result, f'void cascade returns None in loop {label}')
        unittest.TestCase.assertEqual(self, tracker, ['a', 'b'], f'void cascade side effects in loop {label}')

  async def test_void_chain_three_step_parity(self):
    """Chain().then(lambda: 1).then(+1).then(*3) -> 6.

    Same separation as above: Chain and Cascade tested independently.
    """
    # Chain paths
    for label, no_async, inject_async in [
      ('2b', True, False),
      ('1', False, False),
      ('1->6', False, True),
    ]:
      with self.subTest(loop=label):
        c = Chain()
        c = c.then(lambda: 1).then(lambda v: v + 1).then(lambda v: v * 3)
        if inject_async:
          c = c.then(aempty)
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 6, f'void chain 3-step in loop {label}')

    # Cascade paths — no-arg callbacks, result is None
    for label, no_async, inject_async in [
      ('2a', True, False),
      ('3', False, False),
      ('3->5', False, True),
    ]:
      with self.subTest(loop=label):
        tracker = []
        c = Cascade()
        c = c.then(lambda: tracker.append(1))
        c = c.then(lambda: tracker.append(2))
        c = c.then(lambda: tracker.append(3))
        if inject_async:
          c = c.then(aempty)
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertIsNone(self, result, f'void cascade returns None in loop {label}')
        unittest.TestCase.assertEqual(self, tracker, [1, 2, 3], f'void cascade side effects in loop {label}')


# ---------------------------------------------------------------------------
# LoopParityExceptionTests
# ---------------------------------------------------------------------------
class LoopParityExceptionTests(MyTestCase):
  """Verify that exceptions propagate identically through all loops."""

  async def _assert_raises_all_paths(self, root, callbacks, exc_type):
    """Assert that running callbacks raises exc_type through all 6 loops."""
    # Loop 2b: sync Chain
    c = build_chain(Chain, root, callbacks, no_async=True)
    with self.assertRaises(exc_type, msg='Loop 2b (sync chain)'):
      c.run()

    # Loop 2a: sync Cascade
    c = build_chain(Cascade, root, callbacks, no_async=True)
    with self.assertRaises(exc_type, msg='Loop 2a (sync cascade)'):
      c.run()

    # Loop 1: async-capable Chain, sync callbacks
    c = build_chain(Chain, root, callbacks, no_async=False)
    with self.assertRaises(exc_type, msg='Loop 1 (async-capable chain)'):
      await await_(c.run())

    # Loop 3: async-capable Cascade, sync callbacks
    c = build_chain(Cascade, root, callbacks, no_async=False)
    with self.assertRaises(exc_type, msg='Loop 3 (async-capable cascade)'):
      await await_(c.run())

    # Loop 1->6: async Chain (async callback before the raising one)
    async_callbacks = [aempty] + list(callbacks)
    c = build_chain(Chain, root, async_callbacks, no_async=False)
    with self.assertRaises(exc_type, msg='Loop 1->6 (async chain)'):
      await await_(c.run())

    # Loop 3->5: async Cascade (async callback before the raising one)
    c = build_chain(Cascade, root, async_callbacks, no_async=False)
    with self.assertRaises(exc_type, msg='Loop 3->5 (async cascade)'):
      await await_(c.run())

  async def test_exception_parity(self):
    """ZeroDivisionError through all loops."""
    await self._assert_raises_all_paths(
      42, [lambda v: 1 / 0], ZeroDivisionError,
    )

  async def test_exception_after_transforms_parity(self):
    """Exception after some successful transforms — same behavior everywhere."""
    await self._assert_raises_all_paths(
      1,
      [lambda v: v + 1, lambda v: v * 2, lambda v: (_ for _ in ()).throw(TestExc())],
      TestExc,
    )

  async def test_exception_in_root_callable_parity(self):
    """Exception raised by root callable."""
    def raise_root():
      raise TestExc('root error')

    # Loop 2b
    c = Chain(raise_root).no_async(True).then(lambda v: v)
    with self.assertRaises(TestExc, msg='Loop 2b'):
      c.run()

    # Loop 2a
    c = Cascade(raise_root).no_async(True).then(lambda v: v)
    with self.assertRaises(TestExc, msg='Loop 2a'):
      c.run()

    # Loop 1
    c = Chain(raise_root).then(lambda v: v)
    with self.assertRaises(TestExc, msg='Loop 1'):
      await await_(c.run())

    # Loop 3
    c = Cascade(raise_root).then(lambda v: v)
    with self.assertRaises(TestExc, msg='Loop 3'):
      await await_(c.run())

    # Loop 1->6
    c = Chain(raise_root).then(aempty)
    with self.assertRaises(TestExc, msg='Loop 1->6'):
      await await_(c.run())

    # Loop 3->5
    c = Cascade(raise_root).then(aempty)
    with self.assertRaises(TestExc, msg='Loop 3->5'):
      await await_(c.run())

  async def test_typeerror_parity(self):
    """TypeError from calling non-callable result through all loops."""
    await self._assert_raises_all_paths(
      'hello',
      [lambda v: v + 1],  # str + int -> TypeError
      TypeError,
    )

  async def test_keyerror_parity(self):
    """KeyError from dict access through all loops."""
    await self._assert_raises_all_paths(
      {'a': 1},
      [lambda v: v['missing']],
      KeyError,
    )

  async def test_exception_preserves_message_parity(self):
    """Exception message is the same across all loops."""
    msg = 'specific error 12345'

    def raise_with_msg(v):
      raise TestExc(msg)

    messages = []

    # Loop 2b
    c = Chain(1).no_async(True).then(raise_with_msg)
    try:
      c.run()
    except TestExc as e:
      messages.append(str(e))

    # Loop 2a
    c = Cascade(1).no_async(True).then(raise_with_msg)
    try:
      c.run()
    except TestExc as e:
      messages.append(str(e))

    # Loop 1
    c = Chain(1).then(raise_with_msg)
    try:
      await await_(c.run())
    except TestExc as e:
      messages.append(str(e))

    # Loop 3
    c = Cascade(1).then(raise_with_msg)
    try:
      await await_(c.run())
    except TestExc as e:
      messages.append(str(e))

    # Loop 1->6
    c = Chain(1).then(aempty).then(raise_with_msg)
    try:
      await await_(c.run())
    except TestExc as e:
      messages.append(str(e))

    # Loop 3->5
    c = Cascade(1).then(aempty).then(raise_with_msg)
    try:
      await await_(c.run())
    except TestExc as e:
      messages.append(str(e))

    unittest.TestCase.assertEqual(self, len(messages), 6, 'all 6 loops raised')
    for i, m in enumerate(messages):
      unittest.TestCase.assertEqual(self, m, msg, f'message parity at index {i}')


# ---------------------------------------------------------------------------
# LoopParityFinallyTests
# ---------------------------------------------------------------------------
class LoopParityFinallyTests(MyTestCase):
  """Verify that finally_ behavior is identical across all loops."""

  async def test_finally_runs_in_all_loops(self):
    """finally_ handler runs and result is correct through all 6 loops."""
    # We use a list per loop to track finally calls
    for label, cls, no_async, inject_async, expected in [
      ('2b', Chain, True, False, 30),
      ('2a', Cascade, True, False, 10),
      ('1', Chain, False, False, 30),
      ('3', Cascade, False, False, 10),
      ('1->6', Chain, False, True, 30),
      ('3->5', Cascade, False, True, 10),
    ]:
      with self.subTest(loop=label):
        tracker = []
        c = cls(10)
        c = c.then(lambda v: v * 3)
        if inject_async:
          c = c.then(aempty)
        c = c.finally_(lambda v: tracker.append(v))
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, expected, f'result in loop {label}')
        unittest.TestCase.assertEqual(self, len(tracker), 1, f'finally ran in loop {label}')

  async def test_finally_on_exception_in_all_loops(self):
    """finally_ runs despite exception, through all loops."""
    def raise_exc(v):
      raise TestExc('boom')

    for label, cls, no_async, inject_async in [
      ('2b', Chain, True, False),
      ('2a', Cascade, True, False),
      ('1', Chain, False, False),
      ('3', Cascade, False, False),
      ('1->6', Chain, False, True),
      ('3->5', Cascade, False, True),
    ]:
      with self.subTest(loop=label):
        tracker = []
        c = cls(10)
        if inject_async:
          c = c.then(aempty)
        c = c.then(raise_exc)
        c = c.finally_(lambda v: tracker.append('finally'))
        if no_async:
          c = c.no_async(True)
        with self.assertRaises(TestExc, msg=f'exception in loop {label}'):
          await await_(c.run())
        unittest.TestCase.assertEqual(self, len(tracker), 1, f'finally ran in loop {label}')

  async def test_finally_receives_root_in_all_loops(self):
    """Verify finally handler receives root value in all loops."""
    for label, cls, no_async, inject_async in [
      ('2b', Chain, True, False),
      ('2a', Cascade, True, False),
      ('1', Chain, False, False),
      ('3', Cascade, False, False),
      ('1->6', Chain, False, True),
      ('3->5', Cascade, False, True),
    ]:
      with self.subTest(loop=label):
        received = []
        c = cls(42)
        c = c.then(lambda v: v * 2)
        if inject_async:
          c = c.then(aempty)
        c = c.finally_(lambda v: received.append(v))
        if no_async:
          c = c.no_async(True)
        await await_(c.run())
        unittest.TestCase.assertEqual(self, len(received), 1, f'finally ran in loop {label}')
        unittest.TestCase.assertEqual(self, received[0], 42, f'finally got root in loop {label}')

  async def test_finally_does_not_alter_result_parity(self):
    """finally_ return value does not change the chain result, across all loops."""
    for label, cls, no_async, inject_async, expected in [
      ('2b', Chain, True, False, 20),
      ('2a', Cascade, True, False, 10),
      ('1', Chain, False, False, 20),
      ('3', Cascade, False, False, 10),
      ('1->6', Chain, False, True, 20),
      ('3->5', Cascade, False, True, 10),
    ]:
      with self.subTest(loop=label):
        c = cls(10)
        c = c.then(lambda v: v * 2)
        if inject_async:
          c = c.then(aempty)
        # finally_ returns 999, but this should NOT change the chain result
        c = c.finally_(lambda v: 999)
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, expected, f'result unchanged in loop {label}')

  async def test_finally_with_void_chain_parity(self):
    """finally_ on void chains — root_value is Null, so finally_ is called with no args.

    evaluate_value(on_finally_link, root_value) where root_value is Null
    calls fn() with no positional args (see evaluate_value line: if current_value is Null: return link.v()).
    So the finally callback must accept no positional arguments.
    """
    for label, cls, no_async, inject_async in [
      ('2b', Chain, True, False),
      ('2a', Cascade, True, False),
      ('1', Chain, False, False),
      ('3', Cascade, False, False),
      ('1->6', Chain, False, True),
      ('3->5', Cascade, False, True),
    ]:
      with self.subTest(loop=label):
        received = []
        c = cls()
        c = c.then(lambda: 42)
        if inject_async:
          c = c.then(aempty)
        c = c.finally_(lambda: received.append('finally'))
        if no_async:
          c = c.no_async(True)
        await await_(c.run())
        unittest.TestCase.assertEqual(self, len(received), 1, f'finally ran in loop {label}')
        unittest.TestCase.assertEqual(self, received[0], 'finally', f'finally callback executed in loop {label}')


# ---------------------------------------------------------------------------
# LoopParityCascadeSideEffectTests
# ---------------------------------------------------------------------------
class LoopParityCascadeSideEffectTests(MyTestCase):
  """Verify that Cascade always passes root to each callback, across all loops.

  Since Cascade discards callback results and always returns root,
  these tests verify that side effects see the correct (root) value.
  """

  async def test_cascade_all_callbacks_receive_root(self):
    """Every callback in a Cascade receives the root value, not previous results."""
    for label, no_async, inject_async in [
      ('2a', True, False),
      ('3', False, False),
      ('3->5', False, True),
    ]:
      with self.subTest(loop=label):
        seen = []
        c = Cascade(42)
        c = c.then(lambda v: seen.append(v) or v * 2)
        c = c.then(lambda v: seen.append(v) or v + 100)
        c = c.then(lambda v: seen.append(v) or 'ignored')
        if inject_async:
          c = c.then(aempty)
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 42, f'cascade returns root in loop {label}')
        # All callbacks should have received 42 (the root)
        unittest.TestCase.assertEqual(self, seen, [42, 42, 42], f'all got root in loop {label}')

  async def test_chain_callbacks_receive_previous_result(self):
    """Contrast: Chain callbacks receive the result of the previous callback."""
    for label, no_async, inject_async in [
      ('2b', True, False),
      ('1', False, False),
      ('1->6', False, True),
    ]:
      with self.subTest(loop=label):
        seen = []
        c = Chain(10)
        c = c.then(lambda v: (seen.append(v), v + 5)[1])
        c = c.then(lambda v: (seen.append(v), v * 2)[1])
        c = c.then(lambda v: (seen.append(v), v - 1)[1])
        if inject_async:
          c = c.then(aempty)
        if no_async:
          c = c.no_async(True)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 29, f'chain result in loop {label}')
        # Callbacks see: 10, 15, 30
        unittest.TestCase.assertEqual(self, seen, [10, 15, 30], f'chain values in loop {label}')


# ---------------------------------------------------------------------------
# LoopParityRootCallableTests
# ---------------------------------------------------------------------------
class LoopParityRootCallableTests(MyTestCase):
  """Test chains where the root value is a callable (evaluated on run)."""

  async def _run_all_paths_callable_root(self, root_fn, callbacks, expected_chain, expected_cascade):
    """Run callable-root chains through all 6 loop paths."""
    # Loop 2b
    c = build_chain(Chain, root_fn, callbacks, no_async=True)
    result_2b = c.run()
    unittest.TestCase.assertEqual(self, result_2b, expected_chain, 'Loop 2b')

    # Loop 2a
    c = build_chain(Cascade, root_fn, callbacks, no_async=True)
    result_2a = c.run()
    unittest.TestCase.assertEqual(self, result_2a, expected_cascade, 'Loop 2a')

    # Loop 1
    c = build_chain(Chain, root_fn, callbacks, no_async=False)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected_chain, 'Loop 1')

    # Loop 3
    c = build_chain(Cascade, root_fn, callbacks, no_async=False)
    result_3 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3, expected_cascade, 'Loop 3')

    # Loop 1->6
    c = build_chain(Chain, root_fn, callbacks, no_async=False)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected_chain, 'Loop 1->6')

    # Loop 3->5
    c = build_chain(Cascade, root_fn, callbacks, no_async=False)
    c = c.then(aempty)
    result_3_5 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_3_5, expected_cascade, 'Loop 3->5')

  async def test_callable_root_single_then(self):
    """Chain(lambda: 10).then(v*2) -> 20 across all loops."""
    await self._run_all_paths_callable_root(
      lambda: 10,
      [lambda v: v * 2],
      20, 10,
    )

  async def test_callable_root_multi_then(self):
    """Chain(lambda: 3).then(+1).then(*5) -> 20 across all loops."""
    await self._run_all_paths_callable_root(
      lambda: 3,
      [lambda v: v + 1, lambda v: v * 5],
      20, 3,
    )

  async def test_callable_root_returns_string(self):
    """Chain(lambda: 'abc').then(upper) -> 'ABC' across all loops."""
    await self._run_all_paths_callable_root(
      lambda: 'abc',
      [lambda v: v.upper()],
      'ABC', 'abc',
    )


# ---------------------------------------------------------------------------
# LoopParityEdgeCaseTests
# ---------------------------------------------------------------------------
class LoopParityEdgeCaseTests(MyTestCase):
  """Edge cases that might trip up one loop but not another."""

  async def _run_all_chain_paths(self, root, callbacks, expected, use_sentinel=False):
    """Run through all 6 paths for Chain only, asserting expected result."""
    r = _SENTINEL if use_sentinel else root

    # Loop 2b
    c = build_chain(Chain, r, callbacks, no_async=True)
    res = c.run() if not use_sentinel else c.run(root)
    unittest.TestCase.assertEqual(self, res, expected, 'Loop 2b')

    # Loop 1
    c = build_chain(Chain, r, callbacks, no_async=False)
    res = await await_(c.run() if not use_sentinel else c.run(root))
    unittest.TestCase.assertEqual(self, res, expected, 'Loop 1')

    # Loop 1->6
    c = build_chain(Chain, r, callbacks, no_async=False)
    c = c.then(aempty)
    res = await await_(c.run() if not use_sentinel else c.run(root))
    unittest.TestCase.assertEqual(self, res, expected, 'Loop 1->6')

  async def test_zero_value_parity(self):
    """0 is a valid value, not treated as falsy sentinel."""
    await self._run_all_chain_paths(0, [lambda v: v + 1], 1)

  async def test_false_value_parity(self):
    """False is a valid value, not treated as falsy sentinel."""
    await self._run_all_chain_paths(False, [lambda v: not v], True)

  async def test_empty_string_parity(self):
    """Empty string is a valid value."""
    await self._run_all_chain_paths('', [lambda v: v + 'x'], 'x')

  async def test_empty_list_parity(self):
    """Empty list is a valid value."""
    await self._run_all_chain_paths([], [lambda v: v + [1]], [1])

  async def test_none_propagation_parity(self):
    """None flows through the chain correctly."""
    await self._run_all_chain_paths(None, [lambda v: v is None], True)

  async def test_large_chain_parity(self):
    """20-step chain produces same result in all loops."""
    callbacks = [lambda v: v + 1 for _ in range(20)]
    await self._run_all_chain_paths(0, callbacks, 20)

  async def test_boolean_result_parity(self):
    """Boolean results flow correctly through all loops."""
    await self._run_all_chain_paths(
      10,
      [lambda v: v > 5, lambda v: v and 'yes' or 'no'],
      'yes',
    )

  async def test_tuple_result_parity(self):
    """Tuple results are preserved through all loops."""
    await self._run_all_chain_paths(
      (1, 2),
      [lambda v: v + (3,), lambda v: v + (4,)],
      (1, 2, 3, 4),
    )

  async def test_nested_data_structure_parity(self):
    """Complex nested data structures survive all loops."""
    await self._run_all_chain_paths(
      {'key': [1, 2]},
      [lambda v: {**v, 'key': v['key'] + [3]}],
      {'key': [1, 2, 3]},
    )


# ---------------------------------------------------------------------------
# LoopParityAsyncRootTests
# ---------------------------------------------------------------------------
class LoopParityAsyncRootTests(MyTestCase):
  """Test chains where the root itself is an async callable.

  When root is async, the root evaluation in _run_simple hits iscoro() and
  jumps to _run_async_simple before any loop iteration.
  """

  async def test_async_root_chain_parity(self):
    """Async root callable, then sync callbacks — chain result matches."""
    # This tests the path: root coro -> _run_async_simple -> loop 6
    result = await await_(
      Chain(aempty, 10).then(lambda v: v + 5).then(lambda v: v * 2).run()
    )
    unittest.TestCase.assertEqual(self, result, 30, 'async root chain')

    # Sync equivalent for comparison
    result_sync = Chain(empty, 10).no_async(True).then(lambda v: v + 5).then(lambda v: v * 2).run()
    unittest.TestCase.assertEqual(self, result_sync, 30, 'sync root chain')

  async def test_async_root_cascade_parity(self):
    """Async root callable in Cascade — root value preserved."""
    result = await await_(
      Cascade(aempty, 10).then(lambda v: v + 5).then(lambda v: v * 2).run()
    )
    unittest.TestCase.assertEqual(self, result, 10, 'async root cascade')

    result_sync = Cascade(empty, 10).no_async(True).then(lambda v: v + 5).then(lambda v: v * 2).run()
    unittest.TestCase.assertEqual(self, result_sync, 10, 'sync root cascade')

  async def test_async_root_with_async_callbacks_parity(self):
    """Both root and callbacks are async — result should match sync version."""
    result_async = await await_(
      Chain(aempty, 5).then(aempty).then(lambda v: v * 3).run()
    )
    result_sync = Chain(empty, 5).no_async(True).then(empty).then(lambda v: v * 3).run()
    unittest.TestCase.assertEqual(self, result_async, result_sync, 'async root + async cbs')
    unittest.TestCase.assertEqual(self, result_async, 15, 'value check')


if __name__ == '__main__':
  unittest.main()
