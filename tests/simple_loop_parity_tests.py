"""Loop parity tests for _run_simple() / _run_async_simple().

The _run_simple() method has a main loop that handles Chain execution.
There is also a loop in _run_async_simple() for async Chain execution.
The 2 relevant loops are:

In _run_simple():
  Loop 1  (async-capable, Chain):   all sync callbacks stay in loop 1

In _run_async_simple():
  Loop 6  (async, Chain):   at least one async callback

These tests run identical chain logic through both loop paths and
assert the same result.  If someone modifies one loop without updating
the other, these tests catch the discrepancy.

Loop selection:
  Loop 1    -> Chain, all sync callbacks
  Loop 1->6 -> Chain, at least one async callback
"""
import unittest
import asyncio
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_chain(root, callbacks):
  """Build a Chain with root value and callback list."""
  c = Chain(root) if root is not _SENTINEL else Chain()
  for cb in callbacks:
    c = c.then(cb)
  return c


def build_chain_with_finally(root, callbacks, finally_cb):
  """Build a chain with .then() callbacks and a .finally_() handler."""
  c = Chain(root) if root is not _SENTINEL else Chain()
  for cb in callbacks:
    c = c.then(cb)
  c = c.finally_(finally_cb)
  return c


_SENTINEL = object()


# ---------------------------------------------------------------------------
# LoopParitySingleCallbackTests
# ---------------------------------------------------------------------------
class LoopParitySingleCallbackTests(IsolatedAsyncioTestCase):
  """Run a single .then() callback through both loop paths."""

  async def _run_all_paths(self, root, callbacks, expected):
    """Run callbacks through both loop paths, assert expected results."""
    # Loop 1: async-capable Chain, sync callbacks (stays in loop 1)
    c = build_chain(root, callbacks)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected, 'Loop 1 (async-capable chain, sync cbs)')

    # Loop 1->6: async Chain (inject aempty to force async transition)
    c = build_chain(root, callbacks)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected, 'Loop 1->6 (async chain)')

  async def test_identity_parity(self):
    """Chain(42).then(identity) -> 42 through all loops."""
    await self._run_all_paths(42, [lambda v: v], 42)

  async def test_transform_parity(self):
    """Chain(10).then(v*3) -> 30."""
    await self._run_all_paths(10, [lambda v: v * 3], 30)

  async def test_string_transform_parity(self):
    """Chain('hello').then(upper) -> 'HELLO'."""
    await self._run_all_paths('hello', [lambda v: v.upper()], 'HELLO')

  async def test_negate_parity(self):
    """Chain(7).then(-v) -> -7 through all loops."""
    await self._run_all_paths(7, [lambda v: -v], -7)

  async def test_list_append_parity(self):
    """Chain([1,2]).then(v+[3]) -> [1,2,3]."""
    await self._run_all_paths([1, 2], [lambda v: v + [3]], [1, 2, 3])

  async def test_none_root_transform_parity(self):
    """Chain(None).then(lambda v: 99) -> 99 through all loops."""
    await self._run_all_paths(None, [lambda v: 99], 99)


# ---------------------------------------------------------------------------
# LoopParityMultiCallbackTests
# ---------------------------------------------------------------------------
class LoopParityMultiCallbackTests(IsolatedAsyncioTestCase):
  """Run a chain of 3+ .then() callbacks through both loops."""

  async def _run_all_paths(self, root, callbacks, expected):
    """Run callbacks through both loop paths, assert expected results."""
    # Loop 1
    c = build_chain(root, callbacks)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected, 'Loop 1 (async-capable chain, sync cbs)')

    # Loop 1->6
    c = build_chain(root, callbacks)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected, 'Loop 1->6 (async chain)')

  async def test_chained_arithmetic_parity(self):
    """Chain(1).then(+1).then(*2).then(+10) -> 14."""
    await self._run_all_paths(
      1,
      [lambda v: v + 1, lambda v: v * 2, lambda v: v + 10],
      14,
    )

  async def test_chained_string_ops_parity(self):
    """Chain('a').then(+'b').then(+'c').then(upper) -> 'ABC'."""
    await self._run_all_paths(
      'a',
      [lambda v: v + 'b', lambda v: v + 'c', lambda v: v.upper()],
      'ABC',
    )

  async def test_five_step_chain_parity(self):
    """5-step transformation: 2 -> ((2+3)*2-1)//2+100 = 104."""
    await self._run_all_paths(
      2,
      [
        lambda v: v + 3,   # 5
        lambda v: v * 2,   # 10
        lambda v: v - 1,   # 9
        lambda v: v // 2,  # 4
        lambda v: v + 100, # 104
      ],
      104,
    )

  async def test_four_step_string_parity(self):
    """4-step string chain: 'foo' -> 'FOO!!__X'."""
    await self._run_all_paths(
      'foo',
      [
        lambda v: v.upper(),    # 'FOO'
        lambda v: v + '!!',     # 'FOO!!'
        lambda v: v + '__',     # 'FOO!!__'
        lambda v: v + 'X',      # 'FOO!!__X'
      ],
      'FOO!!__X',
    )

  async def test_three_step_list_parity(self):
    """3-step list chain: [] -> [1,2,3]."""
    await self._run_all_paths(
      [],
      [
        lambda v: v + [1],
        lambda v: v + [2],
        lambda v: v + [3],
      ],
      [1, 2, 3],
    )


# ---------------------------------------------------------------------------
# LoopParityAsyncTransitionTests
# ---------------------------------------------------------------------------
class LoopParityAsyncTransitionTests(IsolatedAsyncioTestCase):
  """Test chains that start sync then hit an async callback mid-chain.

  These specifically test the jump from loop 1->6, comparing
  results against the fully-sync versions.
  """

  async def test_async_mid_chain_parity(self):
    """Async callback in the middle of a chain matches sync equivalent.

    Chain(5).then(+1).then(aempty).then(*2) = 12  (loop 1->6)
    Chain(5).then(+1).then(empty).then(*2) = 12  (loop 1)
    """
    # sync chain (loop 1)
    result_sync = await await_(Chain(5).then(lambda v: v + 1).then(empty).then(lambda v: v * 2).run())
    unittest.TestCase.assertEqual(self, result_sync, 12, 'sync chain')

    # async chain -- aempty mid-chain forces loop 1->6
    result_async = await await_(
      Chain(5).then(lambda v: v + 1).then(aempty).then(lambda v: v * 2).run()
    )
    unittest.TestCase.assertEqual(self, result_async, 12, 'async chain (loop 1->6)')

  async def test_async_first_callback_parity(self):
    """Async callback is the first .then() -- should match sync."""
    # sync
    result_sync = await await_(Chain(10).then(empty).then(lambda v: v + 5).run())
    unittest.TestCase.assertEqual(self, result_sync, 15, 'sync chain')

    # async -- aempty as first .then() triggers transition
    result_async = await await_(
      Chain(10).then(aempty).then(lambda v: v + 5).run()
    )
    unittest.TestCase.assertEqual(self, result_async, 15, 'async chain first callback')

  async def test_async_last_callback_parity(self):
    """Async callback is the last .then() -- should match sync."""
    result_sync = await await_(Chain(10).then(lambda v: v + 5).then(empty).run())
    unittest.TestCase.assertEqual(self, result_sync, 15, 'sync chain')

    result_async = await await_(
      Chain(10).then(lambda v: v + 5).then(aempty).run()
    )
    unittest.TestCase.assertEqual(self, result_async, 15, 'async chain last callback')

  async def test_multiple_async_callbacks_parity(self):
    """Multiple async callbacks interspersed -- same result as all-sync."""
    result_sync = await await_(Chain(1).then(
      lambda v: v + 1
    ).then(empty).then(
      lambda v: v * 3
    ).then(empty).then(
      lambda v: v - 2
    ).run())
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
    result_sync = await await_(Chain({'a': 1}).then(
      lambda v: {**v, 'b': 2}
    ).then(empty).run())

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
class LoopParityRootOverrideTests(IsolatedAsyncioTestCase):
  """Test .run(override_value) through both loops."""

  async def _run_all_paths_override(self, override, callbacks, expected):
    """Run callbacks through both loop paths with root override."""
    # Loop 1: async-capable Chain, sync callbacks
    c = build_chain(_SENTINEL, callbacks)
    result_1 = await await_(c.run(override))
    unittest.TestCase.assertEqual(self, result_1, expected, 'Loop 1 (async-capable chain override)')

    # Loop 1->6: async Chain
    c = build_chain(_SENTINEL, callbacks)
    c = c.then(aempty)
    result_1_6 = await await_(c.run(override))
    unittest.TestCase.assertEqual(self, result_1_6, expected, 'Loop 1->6 (async chain override)')

  async def test_root_override_parity(self):
    """Chain().then(v*2).run(21) -> 42 through all Chain loops."""
    await self._run_all_paths_override(21, [lambda v: v * 2], 42)

  async def test_root_override_with_multi_then_parity(self):
    """Multiple .then() with root override."""
    await self._run_all_paths_override(
      5,
      [lambda v: v + 1, lambda v: v * 10, lambda v: v - 3],
      57,
    )

  async def test_root_override_string_parity(self):
    """String root override through all loops."""
    await self._run_all_paths_override(
      'test',
      [lambda v: v.upper(), lambda v: v + '!'],
      'TEST!',
    )

  async def test_root_override_none_parity(self):
    """None as root override through all loops."""
    await self._run_all_paths_override(
      None,
      [lambda v: 42],
      42,
    )

  async def test_root_override_callable_parity(self):
    """Callable root override -- root is evaluated, then callbacks use result."""
    # When root is a callable, it is called with no args (Null -> calls v())
    # Loop 1
    c = Chain().then(lambda v: v + 10)
    result_1 = await await_(c.run(lambda: 5))
    unittest.TestCase.assertEqual(self, result_1, 15, 'Loop 1 callable override')

    # Loop 1->6
    c = Chain().then(lambda v: v + 10).then(aempty)
    result_1_6 = await await_(c.run(lambda: 5))
    unittest.TestCase.assertEqual(self, result_1_6, 15, 'Loop 1->6 callable override')


# ---------------------------------------------------------------------------
# LoopParityVoidChainTests
# ---------------------------------------------------------------------------
class LoopParityVoidChainTests(IsolatedAsyncioTestCase):
  """Test chains with no root value through both loops."""

  async def _run_all_paths_void(self, callbacks, expected):
    """Run void-root callbacks through both loop paths."""
    # Loop 1: async-capable Chain
    c = build_chain(_SENTINEL, callbacks)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected, 'Loop 1 (async-capable void chain)')

    # Loop 1->6: async Chain
    c = build_chain(_SENTINEL, callbacks)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected, 'Loop 1->6 (async void chain)')

  async def test_void_chain_parity(self):
    """Chain().then(lambda: 42) -> 42 through all loops."""
    await self._run_all_paths_void(
      [lambda: 42],
      42,
    )

  async def test_void_chain_multi_then_parity(self):
    """Chain().then(lambda: 10).then(lambda v: v + 5) -> 15."""
    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        c = Chain()
        c = c.then(lambda: 10).then(lambda v: v + 5)
        if inject_async:
          c = c.then(aempty)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 15, f'void chain multi in loop {label}')

  async def test_void_chain_three_step_parity(self):
    """Chain().then(lambda: 1).then(+1).then(*3) -> 6."""
    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        c = Chain()
        c = c.then(lambda: 1).then(lambda v: v + 1).then(lambda v: v * 3)
        if inject_async:
          c = c.then(aempty)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 6, f'void chain 3-step in loop {label}')


# ---------------------------------------------------------------------------
# LoopParityExceptionTests
# ---------------------------------------------------------------------------
class LoopParityExceptionTests(IsolatedAsyncioTestCase):
  """Verify that exceptions propagate identically through both loops."""

  async def _assert_raises_all_paths(self, root, callbacks, exc_type):
    """Assert that running callbacks raises exc_type through both loops."""
    # Loop 1: async-capable Chain, sync callbacks
    c = build_chain(root, callbacks)
    with self.assertRaises(exc_type, msg='Loop 1 (async-capable chain)'):
      await await_(c.run())

    # Loop 1->6: async Chain (async callback before the raising one)
    async_callbacks = [aempty] + list(callbacks)
    c = build_chain(root, async_callbacks)
    with self.assertRaises(exc_type, msg='Loop 1->6 (async chain)'):
      await await_(c.run())

  async def test_exception_parity(self):
    """ZeroDivisionError through all loops."""
    await self._assert_raises_all_paths(
      42, [lambda v: 1 / 0], ZeroDivisionError,
    )

  async def test_exception_after_transforms_parity(self):
    """Exception after some successful transforms -- same behavior everywhere."""
    await self._assert_raises_all_paths(
      1,
      [lambda v: v + 1, lambda v: v * 2, lambda v: (_ for _ in ()).throw(TestExc())],
      TestExc,
    )

  async def test_exception_in_root_callable_parity(self):
    """Exception raised by root callable."""
    def raise_root():
      raise TestExc('root error')

    # Loop 1
    c = Chain(raise_root).then(lambda v: v)
    with self.assertRaises(TestExc, msg='Loop 1'):
      await await_(c.run())

    # Loop 1->6
    c = Chain(raise_root).then(aempty)
    with self.assertRaises(TestExc, msg='Loop 1->6'):
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
    """Exception message is the same across both loops."""
    msg = 'specific error 12345'

    def raise_with_msg(v):
      raise TestExc(msg)

    messages = []

    # Loop 1
    c = Chain(1).then(raise_with_msg)
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

    unittest.TestCase.assertEqual(self, len(messages), 2, 'both loops raised')
    for i, m in enumerate(messages):
      unittest.TestCase.assertEqual(self, m, msg, f'message parity at index {i}')


# ---------------------------------------------------------------------------
# LoopParityFinallyTests
# ---------------------------------------------------------------------------
class LoopParityFinallyTests(IsolatedAsyncioTestCase):
  """Verify that finally_ behavior is identical across both loops."""

  async def test_finally_runs_in_all_loops(self):
    """finally_ handler runs and result is correct through both loops."""
    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        tracker = []
        c = Chain(10)
        c = c.then(lambda v: v * 3)
        if inject_async:
          c = c.then(aempty)
        c = c.finally_(lambda v: tracker.append(v))
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 30, f'result in loop {label}')
        unittest.TestCase.assertEqual(self, len(tracker), 1, f'finally ran in loop {label}')

  async def test_finally_on_exception_in_all_loops(self):
    """finally_ runs despite exception, through both loops."""
    def raise_exc(v):
      raise TestExc('boom')

    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        tracker = []
        c = Chain(10)
        if inject_async:
          c = c.then(aempty)
        c = c.then(raise_exc)
        c = c.finally_(lambda v: tracker.append('finally'))
        with self.assertRaises(TestExc, msg=f'exception in loop {label}'):
          await await_(c.run())
        unittest.TestCase.assertEqual(self, len(tracker), 1, f'finally ran in loop {label}')

  async def test_finally_receives_root_in_all_loops(self):
    """Verify finally handler receives root value in both loops."""
    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        received = []
        c = Chain(42)
        c = c.then(lambda v: v * 2)
        if inject_async:
          c = c.then(aempty)
        c = c.finally_(lambda v: received.append(v))
        await await_(c.run())
        unittest.TestCase.assertEqual(self, len(received), 1, f'finally ran in loop {label}')
        unittest.TestCase.assertEqual(self, received[0], 42, f'finally got root in loop {label}')

  async def test_finally_does_not_alter_result_parity(self):
    """finally_ return value does not change the chain result, across both loops."""
    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        c = Chain(10)
        c = c.then(lambda v: v * 2)
        if inject_async:
          c = c.then(aempty)
        # finally_ returns 999, but this should NOT change the chain result
        c = c.finally_(lambda v: 999)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 20, f'result unchanged in loop {label}')

  async def test_finally_with_void_chain_parity(self):
    """finally_ on void chains -- root_value is Null, so finally_ is called with no args.

    evaluate_value(on_finally_link, root_value) where root_value is Null
    calls fn() with no positional args (see evaluate_value line: if current_value is Null: return link.v()).
    So the finally callback must accept no positional arguments.
    """
    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        received = []
        c = Chain()
        c = c.then(lambda: 42)
        if inject_async:
          c = c.then(aempty)
        c = c.finally_(lambda: received.append('finally'))
        await await_(c.run())
        unittest.TestCase.assertEqual(self, len(received), 1, f'finally ran in loop {label}')
        unittest.TestCase.assertEqual(self, received[0], 'finally', f'finally callback executed in loop {label}')


# ---------------------------------------------------------------------------
# LoopParityChainValueFlowTests
# ---------------------------------------------------------------------------
class LoopParityChainValueFlowTests(IsolatedAsyncioTestCase):
  """Verify that Chain callbacks receive the result of the previous callback."""

  async def test_chain_callbacks_receive_previous_result(self):
    """Chain callbacks receive the result of the previous callback."""
    for label, inject_async in [
      ('1', False),
      ('1->6', True),
    ]:
      with self.subTest(loop=label):
        seen = []
        c = Chain(10)
        c = c.then(lambda v: (seen.append(v), v + 5)[1])
        c = c.then(lambda v: (seen.append(v), v * 2)[1])
        c = c.then(lambda v: (seen.append(v), v - 1)[1])
        if inject_async:
          c = c.then(aempty)
        result = await await_(c.run())
        unittest.TestCase.assertEqual(self, result, 29, f'chain result in loop {label}')
        # Callbacks see: 10, 15, 30
        unittest.TestCase.assertEqual(self, seen, [10, 15, 30], f'chain values in loop {label}')


# ---------------------------------------------------------------------------
# LoopParityRootCallableTests
# ---------------------------------------------------------------------------
class LoopParityRootCallableTests(IsolatedAsyncioTestCase):
  """Test chains where the root value is a callable (evaluated on run)."""

  async def _run_all_paths_callable_root(self, root_fn, callbacks, expected):
    """Run callable-root chains through both loop paths."""
    # Loop 1
    c = build_chain(root_fn, callbacks)
    result_1 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1, expected, 'Loop 1')

    # Loop 1->6
    c = build_chain(root_fn, callbacks)
    c = c.then(aempty)
    result_1_6 = await await_(c.run())
    unittest.TestCase.assertEqual(self, result_1_6, expected, 'Loop 1->6')

  async def test_callable_root_single_then(self):
    """Chain(lambda: 10).then(v*2) -> 20 across all loops."""
    await self._run_all_paths_callable_root(
      lambda: 10,
      [lambda v: v * 2],
      20,
    )

  async def test_callable_root_multi_then(self):
    """Chain(lambda: 3).then(+1).then(*5) -> 20 across all loops."""
    await self._run_all_paths_callable_root(
      lambda: 3,
      [lambda v: v + 1, lambda v: v * 5],
      20,
    )

  async def test_callable_root_returns_string(self):
    """Chain(lambda: 'abc').then(upper) -> 'ABC' across all loops."""
    await self._run_all_paths_callable_root(
      lambda: 'abc',
      [lambda v: v.upper()],
      'ABC',
    )


# ---------------------------------------------------------------------------
# LoopParityEdgeCaseTests
# ---------------------------------------------------------------------------
class LoopParityEdgeCaseTests(IsolatedAsyncioTestCase):
  """Edge cases that might trip up one loop but not another."""

  async def _run_all_chain_paths(self, root, callbacks, expected, use_sentinel=False):
    """Run through both paths for Chain only, asserting expected result."""
    r = _SENTINEL if use_sentinel else root

    # Loop 1
    c = build_chain(r, callbacks)
    res = await await_(c.run() if not use_sentinel else c.run(root))
    unittest.TestCase.assertEqual(self, res, expected, 'Loop 1')

    # Loop 1->6
    c = build_chain(r, callbacks)
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
class LoopParityAsyncRootTests(IsolatedAsyncioTestCase):
  """Test chains where the root itself is an async callable.

  When root is async, the root evaluation in _run_simple hits iscoro() and
  jumps to _run_async_simple before any loop iteration.
  """

  async def test_async_root_chain_parity(self):
    """Async root callable, then sync callbacks -- chain result matches."""
    # This tests the path: root coro -> _run_async_simple -> loop 6
    result = await await_(
      Chain(aempty, 10).then(lambda v: v + 5).then(lambda v: v * 2).run()
    )
    unittest.TestCase.assertEqual(self, result, 30, 'async root chain')

    # Sync equivalent for comparison
    result_sync = await await_(
      Chain(empty, 10).then(lambda v: v + 5).then(lambda v: v * 2).run()
    )
    unittest.TestCase.assertEqual(self, result_sync, 30, 'sync root chain')

  async def test_async_root_with_async_callbacks_parity(self):
    """Both root and callbacks are async -- result should match sync version."""
    result_async = await await_(
      Chain(aempty, 5).then(aempty).then(lambda v: v * 3).run()
    )
    result_sync = await await_(
      Chain(empty, 5).then(empty).then(lambda v: v * 3).run()
    )
    unittest.TestCase.assertEqual(self, result_async, result_sync, 'async root + async cbs')
    unittest.TestCase.assertEqual(self, result_async, 15, 'value check')


if __name__ == '__main__':
  unittest.main()
