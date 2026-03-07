"""Falsy value matrix tests: verify ALL falsy values are correctly preserved/propagated
through EVERY chain operation.

Falsy values are notoriously buggy in pipeline libraries because code often uses
`if value` instead of `if value is not None/Null`. This module exhaustively tests
that every falsy Python value survives every chain operation without corruption.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null

# All falsy values to test. Each entry is (name, value).
# Null is special: it means "no value provided" and is mapped to None at chain boundaries.
FALSY_VALUES = [
  ('None', None),
  ('False', False),
  ('zero_int', 0),
  ('zero_float', 0.0),
  ('zero_complex', 0j),
  ('empty_string', ''),
  ('empty_bytes', b''),
  ('empty_list', []),
  ('empty_tuple', ()),
  ('empty_dict', {}),
  ('empty_set', set()),
  ('empty_frozenset', frozenset()),
]

# Subset that can be used as Chain() root (excludes Null since Chain(Null) == Chain())
FALSY_ROOT_VALUES = FALSY_VALUES

# Subset that are hashable (needed for set membership checks)
FALSY_HASHABLE = [
  ('None', None),
  ('False', False),
  ('zero_int', 0),
  ('zero_float', 0.0),
  ('zero_complex', 0j),
  ('empty_string', ''),
  ('empty_bytes', b''),
  ('empty_tuple', ()),
  ('empty_frozenset', frozenset()),
]


def _assert_falsy_equal(test, result, expected, msg=''):
  """Assert that result equals expected, using identity for mutable types."""
  if isinstance(expected, (list, dict, set)):
    test.assertEqual(type(result), type(expected), msg)
    test.assertEqual(result, expected, msg)
  else:
    test.assertIs(result, expected, msg)


# ---------------------------------------------------------------------------
# TestFalsyThroughThen
# ---------------------------------------------------------------------------


class TestFalsyThroughThen(unittest.TestCase):
  """Falsy values flowing through .then() steps."""

  def test_falsy_as_root_then_identity(self):
    """Chain(falsy).then(lambda x: x).run() — falsy flows through identity then."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = Chain(val).then(lambda x: x).run()
        _assert_falsy_equal(self, result, val)

  def test_falsy_as_then_result(self):
    """Chain(1).then(lambda x: falsy).run() — then step returns falsy."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain(1).then(lambda x, v=val: v).run()
        _assert_falsy_equal(self, result, val)

  def test_falsy_as_run_arg(self):
    """Chain().then(lambda x: x).run(falsy) — falsy passed via run()."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = Chain().then(lambda x: x).run(val)
        _assert_falsy_equal(self, result, val)

  def test_falsy_root_no_then(self):
    """Chain(falsy).run() — falsy as root with no then steps."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = Chain(val).run()
        _assert_falsy_equal(self, result, val)

  def test_falsy_through_two_then_steps(self):
    """Chain(falsy).then(identity).then(identity).run() — two hops."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = Chain(val).then(lambda x: x).then(lambda x: x).run()
        _assert_falsy_equal(self, result, val)

  def test_then_receives_falsy_value(self):
    """Verify the lambda in .then() actually receives the falsy value."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        captured = []
        Chain(val).then(lambda x: captured.append(x)).run()
        self.assertEqual(len(captured), 1, f'then was not called for {name}')
        _assert_falsy_equal(self, captured[0], val)


# ---------------------------------------------------------------------------
# TestFalsyThroughDo
# ---------------------------------------------------------------------------


class TestFalsyThroughDo(unittest.TestCase):
  """Falsy values preserved across .do() side-effect steps."""

  def test_falsy_preserved_after_do(self):
    """Chain(falsy).do(lambda x: 'ignored').run() — falsy is current_value after do."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = Chain(val).do(lambda x: 'ignored').run()
        _assert_falsy_equal(self, result, val)

  def test_do_return_does_not_replace_falsy(self):
    """Chain(42).do(lambda x: falsy).run() — do return is discarded, 42 preserved."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain(42).do(lambda x, v=val: v).run()
        self.assertEqual(result, 42)

  def test_do_receives_falsy_current_value(self):
    """The do() callback receives the falsy current_value."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        captured = []
        Chain(val).do(lambda x: captured.append(x)).run()
        self.assertEqual(len(captured), 1)
        _assert_falsy_equal(self, captured[0], val)

  def test_multiple_do_preserve_falsy(self):
    """Chain(falsy).do(f1).do(f2).do(f3).run() — all do's receive falsy, result is falsy."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        calls = []
        result = (
          Chain(val)
          .do(lambda x: calls.append(('f1', x)))
          .do(lambda x: calls.append(('f2', x)))
          .do(lambda x: calls.append(('f3', x)))
          .run()
        )
        _assert_falsy_equal(self, result, val)
        self.assertEqual(len(calls), 3)


# ---------------------------------------------------------------------------
# TestFalsyThroughMap
# ---------------------------------------------------------------------------


class TestFalsyThroughMap(unittest.TestCase):
  """Falsy values inside iterables through .map()."""

  def test_map_preserves_falsy_items(self):
    """Chain([falsy, falsy]).map(lambda x: x).run() — both preserved."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain([val, val]).map(lambda x: x).run()
        self.assertEqual(len(result), 2)
        for item in result:
          _assert_falsy_equal(self, item, val)

  def test_map_mixed_falsy_iterable(self):
    """All falsy values in one list — map identity preserves all."""
    items = [0, None, False, '', []]
    result = Chain(items).map(lambda x: x).run()
    self.assertEqual(len(result), 5)
    self.assertIs(result[0], 0)
    self.assertIsNone(result[1])
    self.assertIs(result[2], False)
    self.assertEqual(result[3], '')
    self.assertEqual(result[4], [])

  def test_foreach_preserves_original_falsy_items(self):
    """foreach discards fn result, keeps original falsy items."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain([val, val]).foreach(lambda x: 'ignored').run()
        self.assertEqual(len(result), 2)
        for item in result:
          _assert_falsy_equal(self, item, val)

  def test_map_returns_falsy_values(self):
    """map fn returns falsy — result list contains falsy values."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain([1, 2, 3]).map(lambda x, v=val: v).run()
        self.assertEqual(len(result), 3)
        for item in result:
          _assert_falsy_equal(self, item, val)

  def test_map_callback_receives_falsy_items(self):
    """The map callback receives each falsy item."""
    captured = []
    items = [0, None, False, '', b'', 0.0, 0j]
    Chain(items).map(lambda x: captured.append(x) or x).run()
    self.assertEqual(len(captured), len(items))
    for i, item in enumerate(items):
      _assert_falsy_equal(self, captured[i], item)


# ---------------------------------------------------------------------------
# TestFalsyThroughFilter
# ---------------------------------------------------------------------------


class TestFalsyThroughFilter(unittest.TestCase):
  """Falsy values through .filter()."""

  def test_filter_bool_removes_all_falsy(self):
    """Filter with bool: all falsy values are filtered OUT."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain([val, 1, val, 2]).filter(bool).run()
        self.assertEqual(result, [1, 2], f'{name} was not filtered out')

  def test_filter_always_true_keeps_all_falsy(self):
    """Filter with lambda x: True — ALL falsy values pass through."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain([val, val]).filter(lambda x: True).run()
        self.assertEqual(len(result), 2)
        for item in result:
          _assert_falsy_equal(self, item, val)

  def test_filter_not_none_only_removes_none(self):
    """Filter with `lambda x: x is not None` — only None removed."""
    items = [0, None, False, '', 0.0]
    result = Chain(items).filter(lambda x: x is not None).run()
    self.assertEqual(len(result), 4)
    self.assertNotIn(None, result)
    self.assertIn(0, result)
    self.assertIn(False, result)
    self.assertIn('', result)
    self.assertIn(0.0, result)

  def test_filter_preserves_falsy_items_when_predicate_true(self):
    """All falsy values in one list with always-true predicate."""
    items = [0, None, False, '', b'', [], (), {}, set(), frozenset(), 0.0, 0j]
    result = Chain(items).filter(lambda x: True).run()
    self.assertEqual(len(result), len(items))

  def test_filter_callback_receives_falsy_items(self):
    """The filter predicate receives each falsy item correctly."""
    captured = []
    items = [0, None, False, '']
    Chain(items).filter(lambda x: captured.append(x) or True).run()
    self.assertEqual(len(captured), len(items))
    for i, item in enumerate(items):
      _assert_falsy_equal(self, captured[i], item)


# ---------------------------------------------------------------------------
# TestFalsyThroughWith
# ---------------------------------------------------------------------------


class _FalsyCM:
  """Context manager whose __enter__ returns a specific falsy value."""
  def __init__(self, enter_val):
    self._enter_val = enter_val
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self._enter_val

  def __exit__(self, *args):
    self.exited = True
    return False


class TestFalsyThroughWith(unittest.TestCase):
  """Falsy values through .with_() context manager operations."""

  def test_cm_enter_returns_falsy(self):
    """CM.__enter__ returns each falsy value — fn receives it."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        cm = _FalsyCM(val)
        captured = []
        Chain(cm).with_(lambda ctx: captured.append(ctx) or ctx).run()
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)
        self.assertEqual(len(captured), 1)
        _assert_falsy_equal(self, captured[0], val)

  def test_with_do_falsy_cm_preserves_outer_value(self):
    """with_do with falsy CM enter — outer_value (the CM) preserved."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        cm = _FalsyCM(val)
        result = Chain(cm).with_do(lambda ctx: 'ignored').run()
        self.assertIs(result, cm)
        self.assertTrue(cm.entered)
        self.assertTrue(cm.exited)

  def test_with_returns_falsy_body_result(self):
    """with_ body returns falsy — that falsy value is the chain result."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        cm = _FalsyCM('truthy_enter')
        result = Chain(cm).with_(lambda ctx, v=val: v).run()
        _assert_falsy_equal(self, result, val)


# ---------------------------------------------------------------------------
# TestFalsyThroughGather
# ---------------------------------------------------------------------------


class TestFalsyThroughGather(unittest.TestCase):
  """Falsy values through .gather()."""

  def test_gather_each_fn_returns_different_falsy(self):
    """Each fn in gather returns a different falsy value — all collected."""
    fns = [lambda x, v=val: v for _, val in FALSY_VALUES]
    result = Chain(42).gather(*fns).run()
    self.assertEqual(len(result), len(FALSY_VALUES))
    for i, (name, val) in enumerate(FALSY_VALUES):
      with self.subTest(value=name):
        _assert_falsy_equal(self, result[i], val)

  def test_gather_current_value_is_falsy(self):
    """Current value is falsy — all fns receive it."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        captured = []
        result = Chain(val).gather(
          lambda x: captured.append(x) or 'a',
          lambda x: captured.append(x) or 'b',
        ).run()
        self.assertEqual(result, ['a', 'b'])
        self.assertEqual(len(captured), 2)
        for item in captured:
          _assert_falsy_equal(self, item, val)

  def test_gather_all_return_same_falsy(self):
    """All gather fns return the same falsy value."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain(1).gather(
          lambda x, v=val: v,
          lambda x, v=val: v,
          lambda x, v=val: v,
        ).run()
        self.assertEqual(len(result), 3)
        for item in result:
          _assert_falsy_equal(self, item, val)


# ---------------------------------------------------------------------------
# TestFalsyThroughExcept
# ---------------------------------------------------------------------------


class TestFalsyThroughExcept(unittest.TestCase):
  """Falsy values returned by except handlers."""

  def test_except_handler_returns_falsy(self):
    """Except handler returns each falsy value — chain result is that value."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain(1).then(lambda x: 1 / 0).except_(lambda exc, v=val: v).run()
        _assert_falsy_equal(self, result, val)

  def test_except_handler_returns_null_becomes_none(self):
    """Except handler returns Null — chain result becomes None."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda exc: Null).run()
    self.assertIsNone(result)

  def test_except_with_falsy_root(self):
    """Exception occurs when root is falsy — except handler still runs."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = (
          Chain(val)
          .then(lambda x: (_ for _ in ()).throw(ValueError('test')))
          .except_(lambda exc: 'caught')
          .run()
        )
        self.assertEqual(result, 'caught')


# ---------------------------------------------------------------------------
# TestFalsyThroughFinally
# ---------------------------------------------------------------------------


class TestFalsyThroughFinally(unittest.TestCase):
  """Falsy values through .finally_() handlers."""

  def test_finally_receives_falsy_root(self):
    """Root value is falsy — finally handler receives it."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        captured = []
        Chain(val).finally_(lambda x: captured.append(x)).run()
        self.assertEqual(len(captured), 1, f'finally not called for {name}')
        _assert_falsy_equal(self, captured[0], val)

  def test_finally_does_not_affect_falsy_result(self):
    """Finally handler returns falsy — chain result NOT affected."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = Chain(42).finally_(lambda x, v=val: v).run()
        self.assertEqual(result, 42)

  def test_finally_handler_return_does_not_replace_falsy_chain_result(self):
    """Chain result is falsy, finally returns truthy — chain result stays falsy."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = Chain(val).finally_(lambda x: 'truthy_finally').run()
        _assert_falsy_equal(self, result, val)

  def test_finally_with_falsy_root_and_then(self):
    """Falsy root, then step, finally — all interact correctly."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        captured = []
        result = Chain(val).then(lambda x: x).finally_(lambda x: captured.append(x)).run()
        _assert_falsy_equal(self, result, val)
        self.assertEqual(len(captured), 1)
        _assert_falsy_equal(self, captured[0], val)


# ---------------------------------------------------------------------------
# TestFalsyThroughNestedChain
# ---------------------------------------------------------------------------


class TestFalsyThroughNestedChain(unittest.TestCase):
  """Falsy values passing through nested chain boundaries."""

  def test_falsy_passes_through_nested_chain(self):
    """Falsy value enters nested chain and comes back out."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        inner = Chain().then(lambda x: x)
        result = Chain(val).then(inner).run()
        _assert_falsy_equal(self, result, val)

  def test_nested_chain_returns_falsy(self):
    """Nested chain returns falsy — outer chain receives it."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        inner = Chain().then(lambda x, v=val: v)
        result = Chain(1).then(inner).run()
        _assert_falsy_equal(self, result, val)

  def test_falsy_through_double_nested_chain(self):
    """Falsy value passes through two levels of nesting."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        inner2 = Chain().then(lambda x: x)
        inner1 = Chain().then(inner2)
        result = Chain(val).then(inner1).run()
        _assert_falsy_equal(self, result, val)

  def test_nested_chain_do_preserves_falsy(self):
    """Nested chain used in do() — outer falsy value preserved."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        inner = Chain().then(lambda x: 'discarded')
        result = Chain(val).do(inner).run()
        _assert_falsy_equal(self, result, val)


# ---------------------------------------------------------------------------
# TestFalsyThroughFreeze
# ---------------------------------------------------------------------------


class TestFalsyThroughFreeze(unittest.TestCase):
  """Falsy values through frozen chains."""

  def test_frozen_chain_with_falsy_root(self):
    """Frozen chain with falsy root — correct result."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        frozen = Chain(val).then(lambda x: x).freeze()
        result = frozen.run()
        _assert_falsy_equal(self, result, val)

  def test_frozen_chain_falsy_run_value(self):
    """Frozen chain called with falsy run value."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        frozen = Chain().then(lambda x: x).freeze()
        result = frozen.run(val)
        _assert_falsy_equal(self, result, val)

  def test_frozen_chain_falsy_step_result(self):
    """Frozen chain where a step returns falsy."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        frozen = Chain(1).then(lambda x, v=val: v).freeze()
        result = frozen.run()
        _assert_falsy_equal(self, result, val)

  def test_frozen_chain_reuse_with_falsy(self):
    """Frozen chain called multiple times with different falsy values."""
    frozen = Chain().then(lambda x: x).freeze()
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = frozen.run(val)
        _assert_falsy_equal(self, result, val)


# ---------------------------------------------------------------------------
# TestFalsyThroughDecorator
# ---------------------------------------------------------------------------


class TestFalsyThroughDecorator(unittest.TestCase):
  """Falsy values through .decorator()."""

  def test_decorated_fn_returns_falsy(self):
    """Decorated fn returns falsy — chain processes it correctly."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        @Chain().then(lambda x: x).decorator()
        def fn(v=val):
          return v

        result = fn()
        _assert_falsy_equal(self, result, val)

  def test_decorated_fn_receives_falsy_args(self):
    """Decorated fn called with falsy arg — chain processes it."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        @Chain().then(lambda x: x).decorator()
        def fn(x):
          return x

        result = fn(val)
        _assert_falsy_equal(self, result, val)

  def test_decorator_do_preserves_falsy(self):
    """Decorated fn with do() — falsy return value preserved."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        @Chain().do(lambda x: 'side_effect').then(lambda x: x).decorator()
        def fn(v=val):
          return v

        result = fn()
        _assert_falsy_equal(self, result, val)


# ---------------------------------------------------------------------------
# TestFalsyMultiStep
# ---------------------------------------------------------------------------


class TestFalsyMultiStep(unittest.TestCase):
  """Falsy values through multi-step chains."""

  def test_five_then_steps_each_returns_different_falsy(self):
    """Chain with 5 steps, each returns a different falsy value.
    Final result is last then's return."""
    result = (
      Chain(42)
      .then(lambda x: 0)
      .then(lambda x: '')
      .then(lambda x: False)
      .then(lambda x: None)
      .then(lambda x: [])
      .run()
    )
    self.assertEqual(result, [])

  def test_do_steps_dont_corrupt_falsy_current_value(self):
    """do() steps interspersed with then() — falsy current_value not corrupted."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        result = (
          Chain(val)
          .do(lambda x: 'side1')
          .do(lambda x: 'side2')
          .then(lambda x: x)
          .do(lambda x: 'side3')
          .run()
        )
        _assert_falsy_equal(self, result, val)

  def test_chain_of_falsy_transformations(self):
    """Each step transforms one falsy to another — all intermediate values correct."""
    captured = []
    result = (
      Chain(0)
      .then(lambda x: (captured.append(x), None)[1])
      .then(lambda x: (captured.append(x), False)[1])
      .then(lambda x: (captured.append(x), '')[1])
      .then(lambda x: (captured.append(x), ())[1])
      .run()
    )
    self.assertEqual(result, ())
    self.assertEqual(captured, [0, None, False, ''])

  def test_falsy_through_then_and_do_interleaved(self):
    """then(falsy) -> do(truthy) -> then(identity) — falsy preserved."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        result = (
          Chain(1)
          .then(lambda x, v=val: v)
          .do(lambda x: 'ignored')
          .then(lambda x: x)
          .run()
        )
        _assert_falsy_equal(self, result, val)


# ---------------------------------------------------------------------------
# TestFalsyAsync
# ---------------------------------------------------------------------------


class TestFalsyAsync(IsolatedAsyncioTestCase):
  """Async paths with falsy values."""

  async def test_async_fn_returns_falsy(self):
    """Async function returns each falsy value — preserved."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        async def ret_falsy(x, v=val):
          return v

        result = await Chain(42).then(ret_falsy).run()
        _assert_falsy_equal(self, result, val)

  async def test_sync_to_async_transition_with_falsy_current_value(self):
    """Sync root is falsy, then async step — falsy current_value preserved."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        async def async_identity(x):
          return x

        result = await Chain(val).then(async_identity).run()
        _assert_falsy_equal(self, result, val)

  async def test_async_root_returning_falsy(self):
    """Async root returns falsy — chain receives and preserves it."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        async def async_root(v=val):
          return v

        result = await Chain(async_root).then(lambda x: x).run()
        _assert_falsy_equal(self, result, val)

  async def test_async_do_preserves_falsy(self):
    """Async do() with falsy current_value — preserved."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        async def async_side(x):
          return 'ignored'

        result = await Chain(val).do(async_side).run()
        _assert_falsy_equal(self, result, val)

  async def test_async_multi_step_falsy(self):
    """Multiple async steps returning falsy values."""
    async def to_zero(x):
      return 0

    async def to_false(x):
      return False

    async def to_none(x):
      return None

    async def to_empty(x):
      return ''

    result = await Chain(42).then(to_zero).then(to_false).then(to_none).then(to_empty).run()
    self.assertEqual(result, '')

  async def test_async_map_with_falsy_items(self):
    """Async map with falsy items in iterable."""
    async def async_identity(x):
      return x

    items = [0, None, False, '', []]
    result = await Chain(items).map(async_identity).run()
    self.assertEqual(len(result), 5)
    self.assertIs(result[0], 0)
    self.assertIsNone(result[1])
    self.assertIs(result[2], False)
    self.assertEqual(result[3], '')
    self.assertEqual(result[4], [])

  async def test_async_gather_returns_falsy(self):
    """Async gather fns returning falsy values."""
    async def ret_zero(x):
      return 0

    async def ret_none(x):
      return None

    async def ret_false(x):
      return False

    result = await Chain(42).gather(ret_zero, ret_none, ret_false).run()
    self.assertEqual(result, [0, None, False])

  async def test_async_except_returns_falsy(self):
    """Async except handler returns falsy."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        async def handler(exc, v=val):
          return v

        result = await Chain(1).then(lambda x: 1 / 0).except_(handler).run()
        _assert_falsy_equal(self, result, val)

  async def test_async_nested_chain_with_falsy(self):
    """Async nested chain with falsy value."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        async def async_identity(x):
          return x

        inner = Chain().then(async_identity)
        result = await Chain(val).then(inner).run()
        _assert_falsy_equal(self, result, val)

  async def test_async_filter_with_falsy(self):
    """Async filter predicate with falsy items."""
    async def always_true(x):
      return True

    items = [0, None, False, '', []]
    result = await Chain(items).filter(always_true).run()
    self.assertEqual(len(result), len(items))

  async def test_async_with_cm_enter_returns_falsy(self):
    """Async CM whose __aenter__ returns falsy."""
    for name, val in FALSY_VALUES:
      with self.subTest(value=name):
        class AsyncFalsyCM:
          async def __aenter__(self_cm):
            return val

          async def __aexit__(self_cm, *args):
            return False

        captured = []
        await Chain(AsyncFalsyCM()).with_(lambda ctx: captured.append(ctx) or ctx).run()
        self.assertEqual(len(captured), 1)
        _assert_falsy_equal(self, captured[0], val)

  async def test_async_finally_receives_falsy_root(self):
    """Async finally handler receives falsy root value."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        async def async_identity(x):
          return x

        captured = []
        result = await Chain(val).then(async_identity).finally_(lambda x: captured.append(x)).run()
        _assert_falsy_equal(self, result, val)
        self.assertEqual(len(captured), 1)
        _assert_falsy_equal(self, captured[0], val)


# ---------------------------------------------------------------------------
# TestNullVsNoneDistinction
# ---------------------------------------------------------------------------


class TestNullVsNoneDistinction(unittest.TestCase):
  """Null sentinel vs None: verify the distinction is maintained."""

  def test_null_as_root_returns_none(self):
    """Null as root -> run() returns None (Null means 'no value', mapped to None)."""
    result = Chain(Null).run()
    self.assertIsNone(result)

  def test_none_as_root_returns_none(self):
    """None as root -> run() returns None (actual None, not Null-to-None)."""
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_empty_chain_returns_none(self):
    """Chain().run() -> None (no root at all, current_value stays Null -> mapped to None)."""
    result = Chain().run()
    self.assertIsNone(result)

  def test_chain_none_returns_none(self):
    """Chain(None).run() -> None (explicit None root)."""
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_null_root_has_no_root_link(self):
    """Chain(Null) creates no root_link — equivalent to Chain()."""
    c1 = Chain(Null)
    c2 = Chain()
    self.assertIsNone(c1.root_link)
    self.assertIsNone(c2.root_link)

  def test_none_root_has_root_link(self):
    """Chain(None) creates a root_link — None is a valid value."""
    c = Chain(None)
    self.assertIsNotNone(c.root_link)

  def test_none_flows_through_then(self):
    """Chain(None).then(lambda x: x).run() — x receives None."""
    captured = []
    result = Chain(None).then(lambda x: captured.append(x) or x).run()
    self.assertIsNone(result)
    self.assertEqual(len(captured), 1)
    self.assertIsNone(captured[0])

  def test_null_root_then_called_with_no_args(self):
    """Chain(Null).then(fn).run() — fn called with no args (Null = no value)."""
    # Chain(Null) has no root_link, so first_link becomes root.
    # _evaluate_value with current_value=Null and no args/kwargs calls fn().
    result = Chain(Null).then(lambda: 'from_no_arg').run()
    self.assertEqual(result, 'from_no_arg')

  def test_except_returns_null_becomes_none(self):
    """Except handler returns Null -> chain result is None."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda exc: Null).run()
    self.assertIsNone(result)

  def test_except_returns_none_is_none(self):
    """Except handler returns None -> chain result is None."""
    result = Chain(1).then(lambda x: 1 / 0).except_(lambda exc: None).run()
    self.assertIsNone(result)

  def test_run_with_null_is_no_value(self):
    """run(Null) is treated as 'no value' — not the same as run(None)."""
    # run(Null) => v is Null => has_run_value=False => no run-link created
    result = Chain(42).run(Null)
    self.assertEqual(result, 42)

  def test_run_with_none_is_value(self):
    """run(None) provides None as the run value."""
    # run(None) => v is not Null => has_run_value=True => Link(None) created
    result = Chain(42).then(lambda x: x).run(None)
    self.assertIsNone(result)

  def test_null_vs_none_in_except(self):
    """Distinguish Null and None in except handler returns."""
    # Null -> None (Null-to-None mapping)
    r1 = Chain(1).then(lambda x: 1 / 0).except_(lambda e: Null).run()
    # None -> None (actual None)
    r2 = Chain(1).then(lambda x: 1 / 0).except_(lambda e: None).run()
    # Both are None, but through different paths
    self.assertIsNone(r1)
    self.assertIsNone(r2)

  def test_chain_current_value_null_returns_none(self):
    """If current_value remains Null at end of chain, result is None."""
    # Chain with only do() steps — current_value stays Null (do() ignores result)
    result = Chain().do(lambda: 'side_effect').run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TestFalsyEdgeCases
# ---------------------------------------------------------------------------


class TestFalsyEdgeCases(unittest.TestCase):
  """Edge cases with falsy values that are especially likely to cause bugs."""

  def test_zero_not_treated_as_no_value(self):
    """0 must not be treated as Null or trigger 'no value' paths."""
    result = Chain(0).then(lambda x: x + 1).run()
    self.assertEqual(result, 1)

  def test_false_not_treated_as_no_value(self):
    """False must not be treated as Null or trigger 'no value' paths."""
    result = Chain(False).then(lambda x: not x).run()
    self.assertTrue(result)

  def test_empty_string_not_treated_as_no_value(self):
    """'' must not be treated as Null."""
    result = Chain('').then(lambda x: x + 'hello').run()
    self.assertEqual(result, 'hello')

  def test_none_not_treated_as_null(self):
    """None is a valid value, not the same as Null."""
    captured = []
    Chain(None).then(lambda x: captured.append(x)).run()
    self.assertEqual(len(captured), 1)
    self.assertIsNone(captured[0])

  def test_zero_through_gather_map_filter(self):
    """0 survives gather, map, and filter in sequence."""
    result = (
      Chain(0)
      .then(lambda x: [x, x, x])
      .filter(lambda x: True)
      .map(lambda x: x)
      .run()
    )
    self.assertEqual(result, [0, 0, 0])

  def test_none_through_gather_map_filter(self):
    """None survives gather, map, and filter in sequence."""
    result = (
      Chain(None)
      .then(lambda x: [x, x])
      .filter(lambda x: True)
      .map(lambda x: x)
      .run()
    )
    self.assertEqual(result, [None, None])

  def test_false_preserved_in_chain_then_run_arg(self):
    """False as run arg -> then -> result."""
    result = Chain().then(lambda x: x).run(False)
    self.assertIs(result, False)

  def test_zero_preserved_in_frozen_reuse(self):
    """Frozen chain called multiple times with 0 — always returns 0."""
    frozen = Chain().then(lambda x: x).freeze()
    for _ in range(5):
      result = frozen.run(0)
      self.assertEqual(result, 0)

  def test_empty_containers_preserved_through_do(self):
    """All empty container types preserved through do()."""
    for name, val in [('list', []), ('dict', {}), ('set', set()), ('tuple', ())]:
      with self.subTest(value=name):
        result = Chain(val).do(lambda x: 'side').run()
        self.assertEqual(type(result), type(val))
        self.assertEqual(result, val)

  def test_all_falsy_as_root_then_finally(self):
    """Every falsy value as root with finally — both result and finally arg correct."""
    for name, val in FALSY_ROOT_VALUES:
      with self.subTest(value=name):
        finally_args = []
        result = Chain(val).finally_(lambda x: finally_args.append(x)).run()
        _assert_falsy_equal(self, result, val)
        self.assertEqual(len(finally_args), 1)
        _assert_falsy_equal(self, finally_args[0], val)


if __name__ == '__main__':
  unittest.main()
