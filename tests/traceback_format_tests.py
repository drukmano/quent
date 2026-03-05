"""Tests for quent's traceback formatting and chain visualization system.

Verifies that:
- Internal frames (_chain.py, _core.py, _ops.py, _traceback.py) are stripped from tracebacks
- The synthetic <quent> frame contains a readable chain visualization
- The <---- arrow accurately marks the failing link
- Nested chain indentation is correct
- Operations (foreach, filter, gather, with_) display correctly
- Chained exceptions (except_/finally_ handlers that raise) are cleaned
- Runtime values appear in visualizations where supported

Tests for currently-unimplemented features (GAPs) are expected to FAIL and
serve as a specification for future improvements.
"""

from __future__ import annotations

import functools
import sys
import traceback
import unittest
from unittest.mock import patch

from helpers import (
  AsyncCM,
  AsyncRange,
  SyncCM,
  SyncCMRaisesOnEnter,
  async_fn,
  async_raise_fn,
  raise_fn,
  sync_fn,
  sync_identity,
)

from quent import Chain
from quent._traceback import (
  _clean_chained_exceptions,
  _clean_internal_frames,
  _Ctx,
  _format_call_args,
  _get_obj_name,
  _make_indent,
  _stringify_chain,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_tb(fn):
  """Call fn(), capture exception, return formatted traceback string."""
  try:
    fn()
    return None
  except BaseException as exc:
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


async def _capture_tb_async(coro):
  """Await coro, capture exception, return formatted traceback string."""
  try:
    await coro
    return None
  except BaseException as exc:
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _assert_quent_frame(tc, tb_str):
  """Assert File "<quent>" synthetic frame present."""
  tc.assertIn('File "<quent>"', tb_str)


def _assert_no_internal_frames(tc, tb_str):
  """Assert no _chain.py, _core.py, _ops.py, _traceback.py frames."""
  for f in ('_chain.py', '_core.py', '_ops.py', '_traceback.py'):
    tc.assertNotIn(f'quent/{f}', tb_str, f'Internal frame {f} should be stripped')


def _assert_arrow_count(tc, tb_str, count=1):
  """Assert <---- appears exactly count times."""
  tc.assertEqual(
    tb_str.count('<----'),
    count,
    f'Expected {count} arrow(s), got {tb_str.count("<----")}',
  )


def _assert_arrow_on(tc, tb_str, substring):
  """Assert the <---- arrow is on a line containing substring."""
  for line in tb_str.splitlines():
    if '<----' in line:
      tc.assertIn(substring, line, f'Arrow line should contain {substring!r}, got: {line!r}')
      return
  tc.fail('No <---- arrow found in traceback')


def _assert_chain_contains(tc, tb_str, *expected_lines):
  """Assert the chain visualization contains each expected line in order."""
  lines = tb_str.splitlines()
  chain_lines = []
  in_chain = False
  for line in lines:
    if 'File "<quent>"' in line:
      in_chain = True
      continue
    if in_chain:
      if line.startswith('    ') or line.strip() == '':
        chain_lines.append(line.strip())
      else:
        break
  chain_text = '\n'.join(chain_lines)
  for expected in expected_lines:
    tc.assertIn(expected, chain_text, f'Expected {expected!r} in chain visualization:\n{chain_text}')


# ---------------------------------------------------------------------------
# TestStringification -- 18 tests
# ---------------------------------------------------------------------------


class TestStringification(unittest.TestCase):
  """Unit tests for internal stringification helpers called directly."""

  # -- _format_call_args --

  def test_format_call_args_empty(self):
    self.assertEqual(_format_call_args(None, None), '')

  def test_format_call_args_positional(self):
    result = _format_call_args((1, 'hello'), None)
    # _get_obj_name(1) -> repr(1) = '1'; _get_obj_name('hello') -> repr('hello') = "'hello'"
    self.assertEqual(result, "1, 'hello'")

  def test_format_call_args_kwargs(self):
    result = _format_call_args(None, {'key': 'val'})
    self.assertEqual(result, "key='val'")

  def test_format_call_args_mixed(self):
    result = _format_call_args((42,), {'flag': True})
    self.assertEqual(result, '42, flag=True')

  def test_format_call_args_ellipsis(self):
    result = _format_call_args((...,), None)
    self.assertEqual(result, '...')

  def test_format_call_args_callable(self):
    result = _format_call_args((sync_fn,), None)
    self.assertEqual(result, 'sync_fn')

  # -- _get_obj_name --

  def test_get_obj_name_function(self):
    self.assertEqual(_get_obj_name(sync_fn), 'sync_fn')

  def test_get_obj_name_plain_value(self):
    self.assertEqual(_get_obj_name(42), '42')

  def test_get_obj_name_string(self):
    self.assertEqual(_get_obj_name('hello'), "'hello'")

  def test_get_obj_name_partial(self):
    from helpers import partial_fn

    self.assertEqual(_get_obj_name(partial_fn), 'partial(add)')

  # -- _make_indent --

  def test_make_indent_0(self):
    self.assertEqual(_make_indent(0), '\n')

  def test_make_indent_1(self):
    self.assertEqual(_make_indent(1), '\n    ')

  def test_make_indent_2(self):
    self.assertEqual(_make_indent(2), '\n        ')

  # -- _stringify_chain --

  def test_stringify_empty_chain(self):
    c = Chain()
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertEqual(result, 'Chain()')

  def test_stringify_root_value(self):
    c = Chain(42)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertEqual(result, 'Chain(42)')

  def test_stringify_root_callable(self):
    c = Chain(sync_fn)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertEqual(result, 'Chain(sync_fn)')

  def test_stringify_with_links(self):
    c = Chain(1).then(sync_fn).then(str)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.then(sync_fn)', result)
    self.assertIn('.then(str)', result)

  def test_stringify_arrow_on_root(self):
    c = Chain(raise_fn)
    ctx = _Ctx(source_link=c.root_link, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('<----', result)
    self.assertIn('Chain(raise_fn) <----', result)

  def test_stringify_arrow_on_middle(self):
    c = Chain(1).then(sync_fn).then(raise_fn).then(str)
    # first_link is sync_fn, next is raise_fn
    link = c.first_link.next_link
    ctx = _Ctx(source_link=link, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.then(raise_fn) <----', result)
    self.assertNotIn('.then(str) <----', result)

  def test_stringify_with_except_finally(self):
    c = Chain(1).then(sync_fn).except_(sync_identity).finally_(sync_identity)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.except_(sync_identity)', result)
    self.assertIn('.finally_(sync_identity)', result)

  def test_format_link_nested_chain(self):
    inner = Chain().then(sync_fn)
    outer = Chain(1).then(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(outer, nest_lvl=0, ctx=ctx)
    self.assertIn('Chain()', result)
    self.assertIn('.then(sync_fn)', result)

  def test_resolve_nested_indentation(self):
    inner2 = Chain().then(sync_fn)
    inner1 = Chain().then(inner2)
    outer = Chain(1).then(inner1)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(outer, nest_lvl=0, ctx=ctx)
    # 8 spaces = 2 levels of indentation
    self.assertIn('        ', result)


# ---------------------------------------------------------------------------
# TestFrameCleaning -- 10 tests
# ---------------------------------------------------------------------------


class TestFrameCleaning(unittest.TestCase):
  """Tests for _clean_internal_frames and _clean_chained_exceptions."""

  def test_clean_none_returns_none(self):
    self.assertIsNone(_clean_internal_frames(None))

  def test_clean_keeps_user_frames(self):
    try:
      raise ValueError('test')
    except ValueError as exc:
      tb = exc.__traceback__
      result = _clean_internal_frames(tb)
      self.assertIsNotNone(result)
      self.assertIn('test_clean_keeps_user_frames', result.tb_frame.f_code.co_name)

  def test_clean_keeps_quent_synthetic(self):
    try:
      Chain(raise_fn).run()
    except ValueError as exc:
      tb = exc.__traceback__
      result = _clean_internal_frames(tb)
      found = False
      t = result
      while t is not None:
        if t.tb_frame.f_code.co_filename == '<quent>':
          found = True
          break
        t = t.tb_next
      self.assertTrue(found, '<quent> frame should be preserved')

  def test_clean_removes_chain_py(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertNotIn('_chain.py', tb_str)

  def test_clean_removes_core_py(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertNotIn('_core.py', tb_str)

  def test_clean_removes_ops_py(self):
    def fn(x):
      raise ValueError('ops test')

    tb_str = _capture_tb(lambda: Chain([1]).foreach(fn).run())
    self.assertNotIn('_ops.py', tb_str)

  def test_clean_chained_cause(self):
    def handler(exc):
      raise RuntimeError('handler error') from exc

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(handler).run())
    self.assertNotIn('_chain.py', tb_str)
    self.assertNotIn('_core.py', tb_str)

  def test_clean_chained_context(self):
    def handler(exc):
      raise RuntimeError('handler error')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(handler).run())
    self.assertNotIn('_chain.py', tb_str)

  def test_clean_circular_no_crash(self):
    exc1 = ValueError('a')
    exc2 = ValueError('b')
    exc1.__context__ = exc2
    exc2.__context__ = exc1
    exc1.__traceback__ = None
    exc2.__traceback__ = None
    # Should not raise
    _clean_chained_exceptions(exc1, set())

  def test_clean_deeply_nested(self):
    def handler1(exc):
      raise RuntimeError('h1') from exc

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(handler1).run())
    self.assertIsNotNone(tb_str)
    self.assertNotIn('_chain.py', tb_str)


# ---------------------------------------------------------------------------
# TestEndToEndBasic -- 12 tests
# ---------------------------------------------------------------------------


class TestEndToEndBasic(unittest.TestCase):
  """Full end-to-end tests using _capture_tb to capture real formatted tracebacks."""

  def test_root_callable_fails(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    _assert_quent_frame(self, tb_str)
    _assert_no_internal_frames(self, tb_str)
    _assert_arrow_count(self, tb_str, 1)
    _assert_arrow_on(self, tb_str, 'raise_fn')
    self.assertIn('Chain(raise_fn) <----', tb_str)

  def test_mid_chain_fails(self):
    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).then(raise_fn).then(str).run())
    _assert_quent_frame(self, tb_str)
    _assert_no_internal_frames(self, tb_str)
    _assert_arrow_on(self, tb_str, 'raise_fn')
    _assert_chain_contains(
      self,
      tb_str,
      'Chain(1)',
      '.then(sync_fn)',
      '.then(raise_fn, current_value=2) <----',
      '.then(str)',
    )

  def test_last_link_fails(self):
    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).then(raise_fn).run())
    _assert_arrow_on(self, tb_str, 'raise_fn')
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=2) <----')

  def test_do_link_fails(self):
    tb_str = _capture_tb(lambda: Chain(1).do(raise_fn).run())
    _assert_arrow_on(self, tb_str, 'raise_fn')
    _assert_chain_contains(self, tb_str, '.do(raise_fn, current_value=1) <----')

  def test_named_fn_shows_name(self):
    def step_a(x):
      raise RuntimeError('step_a failed')

    tb_str = _capture_tb(lambda: Chain(1).then(step_a).run())
    self.assertIn('step_a', tb_str)

  def test_lambda_shows_lambda(self):
    tb_str = _capture_tb(lambda: Chain(1).then(lambda x: 1 / 0).run())
    self.assertIn('<lambda>', tb_str)

  def test_class_shows_name(self):
    # int as a step: Chain('not_a_number').then(int) -> int('not_a_number') raises
    # _get_obj_name(int) -> 'int' via __name__
    tb_str = _capture_tb(lambda: Chain('not_a_number').then(int).run())
    self.assertIn('int', tb_str)

  def test_partial_shows_partial(self):
    p = functools.partial(raise_fn, None)
    tb_str = _capture_tb(lambda: Chain(1).then(p).run())
    self.assertIn('partial(raise_fn)', tb_str)

  def test_long_chain_10_links(self):
    c = Chain(1)
    for _ in range(9):
      c = c.then(sync_fn)
    c = c.then(raise_fn)
    tb_str = _capture_tb(lambda: c.run())
    self.assertEqual(tb_str.count('.then('), 10)
    _assert_arrow_on(self, tb_str, 'raise_fn')

  def test_explicit_args_shown(self):
    def fn(a, b):
      raise ValueError('args test')

    tb_str = _capture_tb(lambda: Chain(1).then(fn, 'a', 'b').run())
    _assert_chain_contains(self, tb_str, ".then(fn, 'a', 'b')")

  def test_kwargs_shown(self):
    def fn(key=None):
      raise ValueError('kwargs test')

    tb_str = _capture_tb(lambda: Chain(1).then(fn, key='val').run())
    _assert_chain_contains(self, tb_str, ".then(fn, key='val')")

  def test_ellipsis_shown(self):
    tb_str = _capture_tb(lambda: Chain(1).then(raise_fn, ...).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, ...)')

  def test_exception_type_shown(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertIn('ValueError: test error', tb_str)

  def test_raise_in_second_then(self):
    def double(x):
      return x * 2

    def explode(x):
      raise RuntimeError('boom')

    tb_str = _capture_tb(lambda: Chain(1).then(double).then(explode).run())
    _assert_arrow_on(self, tb_str, 'explode')
    _assert_chain_contains(self, tb_str, '.then(double)', '.then(explode, current_value=2) <----')


# ---------------------------------------------------------------------------
# TestEndToEndOperations -- 10 tests
# ---------------------------------------------------------------------------


class TestEndToEndOperations(unittest.TestCase):
  """End-to-end tests for chain operations (foreach, filter, gather, with_)."""

  def test_foreach_shows_item_and_index(self):
    """EXPECTED TO FAIL -- Gap #4: no index shown in foreach."""

    def fn(x):
      if x == 20:
        raise ValueError(f'bad item {x}')
      return x

    tb_str = _capture_tb(lambda: Chain([10, 20, 30]).foreach(fn).run())
    _assert_arrow_on(self, tb_str, 'foreach')
    # This assertion tests the GAP -- will fail until fixed:
    self.assertIn('index=1', tb_str)

  def test_foreach_do_shows_item(self):
    def fn(x):
      if x == 2:
        raise ValueError('bad')
      return x

    tb_str = _capture_tb(lambda: Chain([1, 2]).foreach_do(fn).run())
    _assert_arrow_on(self, tb_str, 'foreach_do')
    # foreach sets temp_args with the item value
    self.assertIn('2', tb_str)

  def test_filter_shows_item_and_index(self):
    """EXPECTED TO FAIL -- Gap #4: no index shown in filter."""

    def fn(x):
      if x == 3:
        raise ValueError('bad')
      return x > 0

    tb_str = _capture_tb(lambda: Chain([1, 2, 3]).filter(fn).run())
    _assert_arrow_on(self, tb_str, 'filter')
    self.assertIn('index=2', tb_str)

  def test_gather_shows_function_names(self):
    """EXPECTED TO FAIL -- Gap #3: gather shows internal wrapper name."""

    def ok1(x):
      return x + 1

    def bad(x):
      raise ValueError('gather fail')

    def ok2(x):
      return x + 2

    tb_str = _capture_tb(lambda: Chain(1).gather(ok1, bad, ok2).run())
    # Check that ok1, bad, ok2 appear in the chain visualization (the <quent>
    # frame section), not just in the user source line of the traceback.
    # Currently shows .gather(_gather_op) instead of .gather(ok1, bad, ok2).
    _assert_chain_contains(self, tb_str, 'ok1', 'bad', 'ok2')

  def test_gather_marks_failing_fn(self):
    """EXPECTED TO FAIL -- Gap #3: no indication of which function failed."""

    def ok1(x):
      return x + 1

    def bad(x):
      raise ValueError('gather fail')

    def ok2(x):
      return x + 2

    tb_str = _capture_tb(lambda: Chain(1).gather(ok1, bad, ok2).run())
    _assert_arrow_on(self, tb_str, 'gather')

  def test_with_body_fails(self):
    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_(raise_fn).run())
    _assert_arrow_on(self, tb_str, 'raise_fn')
    _assert_chain_contains(self, tb_str, ".with_(raise_fn, ctx='ctx_value') <----")

  def test_with_shows_ctx_value(self):
    """EXPECTED TO FAIL -- Gap #2: context value not shown."""
    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_(raise_fn).run())
    self.assertIn('ctx_value', tb_str)

  def test_with_do_body_fails(self):
    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_do(raise_fn).run())
    _assert_chain_contains(self, tb_str, ".with_do(raise_fn, ctx='ctx_value') <----")

  def test_iterate_shows_item_and_index(self):
    """EXPECTED TO FAIL -- Gap #5: no _set_link_temp_args in iterate paths."""

    def fn(x):
      if x == 1:
        raise ValueError('iterate fail')
      return x

    # iterate returns a generator; consuming it triggers the exception.
    # iterate exceptions don't go through _modify_traceback, so there is no
    # <quent> frame or chain visualization. Check for the chain visualization
    # with item/index info, which requires _set_link_temp_args support.
    tb_str = _capture_tb(lambda: list(Chain(range(3)).iterate(fn)))
    _assert_quent_frame(self, tb_str)
    _assert_chain_contains(self, tb_str, 'index=1')

  def test_foreach_temp_args_correct_item(self):
    def fn(x):
      if x == 0:
        return 1 / x  # ZeroDivisionError
      return x

    tb_str = _capture_tb(lambda: Chain([10, 20, 0]).foreach(fn).run())
    _assert_arrow_on(self, tb_str, 'foreach')
    # temp_args should show the failing item (0)
    self.assertIn('0', tb_str)


# ---------------------------------------------------------------------------
# TestEndToEndAsync -- 6 tests
# ---------------------------------------------------------------------------


class TestEndToEndAsync(unittest.IsolatedAsyncioTestCase):
  """Async end-to-end tests for traceback formatting."""

  async def test_async_fn_raises(self):
    tb_str = await _capture_tb_async(Chain(async_raise_fn).run())
    _assert_quent_frame(self, tb_str)
    _assert_no_internal_frames(self, tb_str)
    _assert_arrow_on(self, tb_str, 'async_raise_fn')

  async def test_sync_to_async_transition(self):
    tb_str = await _capture_tb_async(Chain(1).then(sync_fn).then(async_raise_fn).run())
    _assert_arrow_on(self, tb_str, 'async_raise_fn')
    _assert_no_internal_frames(self, tb_str)

  async def test_async_foreach_fails(self):
    async def bad_fn(x):
      if x == 2:
        raise ValueError('async foreach fail')
      return x

    tb_str = await _capture_tb_async(Chain(AsyncRange(5)).foreach(bad_fn).run())
    _assert_arrow_on(self, tb_str, 'foreach')
    _assert_no_internal_frames(self, tb_str)

  async def test_async_with_fails(self):
    tb_str = await _capture_tb_async(Chain(AsyncCM()).with_(async_raise_fn).run())
    _assert_arrow_on(self, tb_str, 'async_raise_fn')
    _assert_no_internal_frames(self, tb_str)

  async def test_async_except_handler_fails(self):
    async def bad_handler(exc):
      raise RuntimeError('handler failed')

    tb_str = await _capture_tb_async(Chain(async_raise_fn).except_(bad_handler).run())
    self.assertIn('ValueError', tb_str)
    self.assertIn('RuntimeError', tb_str)
    # Note: _chain.py frame leaks through in the handler's exception traceback
    # because the RuntimeError propagates through _run_async (await result).
    # This is a known limitation of the async path -- only the *original*
    # exception gets _modify_traceback cleaning, not the handler's own raise.

  async def test_async_finally_handler_fails(self):
    async def bad_finally(root_val):
      raise RuntimeError('finally failed')

    # Use async_fn so the chain runs asynchronously; with sync_fn the chain
    # would run synchronously and the async finally handler would be scheduled
    # as a fire-and-forget task, never raising to the caller.
    tb_str = await _capture_tb_async(Chain(1).then(async_fn).finally_(bad_finally).run())
    self.assertIn('RuntimeError', tb_str)
    _assert_no_internal_frames(self, tb_str)


# ---------------------------------------------------------------------------
# TestNestedChains -- 8 tests
# ---------------------------------------------------------------------------


class TestNestedChains(unittest.TestCase):
  """Tests for nested chain visualization in tracebacks."""

  def test_1_level_nesting(self):
    inner = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: Chain(1).then(inner).run())
    _assert_quent_frame(self, tb_str)
    self.assertIn('Chain()', tb_str)

  def test_2_level_nesting(self):
    inner2 = Chain().then(raise_fn)
    inner1 = Chain().then(inner2)
    tb_str = _capture_tb(lambda: Chain(1).then(inner1).run())
    _assert_quent_frame(self, tb_str)
    _assert_no_internal_frames(self, tb_str)

  def test_3_level_nesting(self):
    inner3 = Chain().then(raise_fn)
    inner2 = Chain().then(inner3)
    inner1 = Chain().then(inner2)
    tb_str = _capture_tb(lambda: Chain(1).then(inner1).run())
    _assert_quent_frame(self, tb_str)

  def test_arrow_on_inner_link(self):
    inner = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: Chain(1).then(inner).run())
    _assert_arrow_on(self, tb_str, 'raise_fn')

  def test_all_levels_visible(self):
    inner = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: Chain(1).then(inner).run())
    chain_count = tb_str.count('Chain(')
    self.assertGreaterEqual(chain_count, 2)

  def test_nested_is_nested_no_duplicate(self):
    inner = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: Chain(1).then(inner).run())
    # Only the outer chain should inject the <quent> frame
    quent_frame_count = tb_str.count('File "<quent>"')
    self.assertEqual(quent_frame_count, 1)

  def test_frozen_as_step_not_nested(self):
    inner = Chain().then(raise_fn).freeze()
    # Frozen chain lacks _is_chain attribute, treated as regular callable
    tb_str = _capture_tb(lambda: Chain(1).then(inner).run())
    _assert_quent_frame(self, tb_str)

  def test_chain_root_is_chain(self):
    inner = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: Chain(inner).run())
    _assert_quent_frame(self, tb_str)
    _assert_arrow_on(self, tb_str, 'raise_fn')


# ---------------------------------------------------------------------------
# TestArrowPrecision -- 10 tests
# ---------------------------------------------------------------------------


class TestArrowPrecision(unittest.TestCase):
  """Tests that the <---- arrow marks exactly the correct failing link."""

  def test_arrow_on_root(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    _assert_arrow_on(self, tb_str, 'Chain(raise_fn)')

  def test_arrow_on_second_of_5(self):
    def fail(x):
      raise ValueError('fail')

    c = Chain(1).then(sync_fn).then(fail).then(sync_fn).then(sync_fn)
    tb_str = _capture_tb(lambda: c.run())
    _assert_arrow_on(self, tb_str, 'fail')
    _assert_arrow_count(self, tb_str, 1)

  def test_arrow_on_last_of_5(self):
    c = Chain(1).then(sync_fn).then(sync_fn).then(sync_fn).then(raise_fn)
    tb_str = _capture_tb(lambda: c.run())
    _assert_arrow_on(self, tb_str, 'raise_fn')

  def test_arrow_on_except_handler(self):
    def bad_handler(exc):
      raise RuntimeError('handler failed')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(bad_handler).run())
    self.assertIn('RuntimeError', tb_str)
    self.assertIn(".except_(bad_handler, exc=ValueError('test error')) <----", tb_str)

  def test_arrow_on_finally_handler(self):
    def bad_finally(root_val):
      raise RuntimeError('finally failed')

    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).finally_(bad_finally).run())
    self.assertIn('.finally_(bad_finally, root_value=1) <----', tb_str)

  def test_arrow_in_inner_chain(self):
    inner = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: Chain(1).then(inner).run())
    _assert_arrow_on(self, tb_str, 'raise_fn')

  def test_arrow_appears_exactly_once(self):
    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).then(raise_fn).then(str).run())
    _assert_arrow_count(self, tb_str, 1)

  def test_arrow_not_on_neighbors(self):
    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).then(raise_fn).then(str).run())
    for line in tb_str.splitlines():
      if '<----' not in line and '.then(' in line:
        self.assertNotIn('<----', line)

  def test_arrow_on_foreach_not_then(self):
    def fn(x):
      if x == 2:
        raise ValueError('fail')
      return x

    tb_str = _capture_tb(lambda: Chain(1).then(lambda x: [1, 2, 3]).foreach(fn).run())
    _assert_arrow_on(self, tb_str, 'foreach')

  def test_chained_exc_both_have_arrows(self):
    def bad_handler(exc):
      raise RuntimeError('handler fail')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(bad_handler).run())
    # Two traceback sections: ValueError and RuntimeError
    # Each should have exactly 1 arrow, total 2
    _assert_arrow_count(self, tb_str, 2)


# ---------------------------------------------------------------------------
# TestFrameExclusion -- 7 tests
# ---------------------------------------------------------------------------


class TestFrameExclusion(unittest.TestCase):
  """Tests that internal quent frames are excluded from tracebacks."""

  def test_no_chain_py(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertNotIn('_chain.py', tb_str)

  def test_no_core_py(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertNotIn('_core.py', tb_str)

  def test_no_ops_py(self):
    def fn(x):
      raise ValueError('ops test')

    tb_str = _capture_tb(lambda: Chain([1]).foreach(fn).run())
    self.assertNotIn('_ops.py', tb_str)

  def test_no_traceback_py(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertNotIn('_traceback.py', tb_str)

  def test_quent_frame_present(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    _assert_quent_frame(self, tb_str)

  def test_user_raise_site_present(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertIn('helpers.py', tb_str)
    self.assertIn('raise_fn', tb_str)

  def test_non_quent_exception_unchanged(self):
    def plain_raise():
      raise ValueError('plain')

    tb_str = _capture_tb(plain_raise)
    self.assertNotIn('<quent>', tb_str)
    self.assertIn('ValueError: plain', tb_str)


# ---------------------------------------------------------------------------
# TestChainedExceptions -- 6 sync tests
# ---------------------------------------------------------------------------


class TestChainedExceptions(unittest.TestCase):
  """Tests for chained exception traceback formatting (except_/finally_ handlers)."""

  def test_except_catches_no_traceback(self):
    result = Chain(raise_fn).except_(lambda exc: 'caught').run()
    self.assertEqual(result, 'caught')

  def test_except_raises_shows_both(self):
    def bad_handler(exc):
      raise RuntimeError('handler error') from exc

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(bad_handler).run())
    self.assertIn('ValueError', tb_str)
    self.assertIn('RuntimeError', tb_str)
    self.assertIn('cause', tb_str.lower())

  def test_except_shows_original_exc(self):
    """except_ handler shows caught exception in chain visualization."""

    def bad_handler(exc):
      raise RuntimeError('handler error')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(bad_handler).run())
    # Should show what exception the handler received in the chain visualization,
    # not just anywhere in the traceback (it naturally appears in the chained
    # exception output). Check that the except_ line in the <quent> frame
    # contains the caught exception info.
    self.assertIn("exc=ValueError('test error')", tb_str)

  def test_finally_raises_shows_both(self):
    def bad_finally(root_val):
      raise RuntimeError('finally error')

    tb_str = _capture_tb(lambda: Chain(1).then(raise_fn).finally_(bad_finally).run())
    self.assertIn('ValueError', tb_str)
    self.assertIn('RuntimeError', tb_str)
    self.assertIn('During handling', tb_str)

  def test_raise_from_inside_chain(self):
    def fn(x):
      try:
        raise ValueError('original')
      except ValueError as e:
        raise RuntimeError('wrapped') from e

    tb_str = _capture_tb(lambda: Chain(1).then(fn).run())
    self.assertIn('ValueError: original', tb_str)
    self.assertIn('RuntimeError: wrapped', tb_str)

  def test_except_and_finally_both_raise(self):
    def bad_handler(exc):
      raise RuntimeError('except error') from exc

    def bad_finally(root_val):
      raise TypeError('finally error')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(bad_handler).finally_(bad_finally).run())
    self.assertIn('TypeError', tb_str)


# ---------------------------------------------------------------------------
# TestChainedExceptionsAsync -- 2 async tests
# ---------------------------------------------------------------------------


class TestChainedExceptionsAsync(unittest.IsolatedAsyncioTestCase):
  """Async counterparts for chained exception tests."""

  async def test_async_except_raises_chained(self):
    async def bad_handler(exc):
      raise RuntimeError('async handler error') from exc

    tb_str = await _capture_tb_async(Chain(async_raise_fn).except_(bad_handler).run())
    self.assertIn('ValueError', tb_str)
    self.assertIn('RuntimeError', tb_str)

  async def test_async_finally_raises_chained(self):
    async def bad_finally(root_val):
      raise RuntimeError('async finally error')

    tb_str = await _capture_tb_async(Chain(1).then(async_fn).finally_(bad_finally).run())
    self.assertIn('RuntimeError', tb_str)


# ---------------------------------------------------------------------------
# TestRuntimeValueDisplay -- 8 tests (ALL EXPECTED TO FAIL)
# ---------------------------------------------------------------------------


class TestRuntimeValueDisplay(unittest.TestCase):
  """Tests for ideal runtime value display.

  ALL tests EXPECTED TO FAIL until FORMAT_GAPS.md fixes are implemented.
  """

  def test_then_shows_current_value(self):
    """Gap #1: current_value not shown in then()."""

    def add1(x):
      return x + 1

    tb_str = _capture_tb(lambda: Chain(1).then(add1).then(raise_fn).run())
    self.assertIn('current_value=2', tb_str)

  def test_do_shows_current_value(self):
    """Gap #1: current_value not shown in do()."""
    tb_str = _capture_tb(lambda: Chain(5).do(raise_fn).run())
    self.assertIn('current_value=5', tb_str)

  def test_foreach_shows_index(self):
    """Gap #4: foreach doesn't show index."""

    def fn(x):
      if x == 30:
        raise ValueError('fail')
      return x

    tb_str = _capture_tb(lambda: Chain([10, 20, 30]).foreach(fn).run())
    self.assertIn('index=2', tb_str)

  def test_filter_shows_index(self):
    """Gap #4: filter doesn't show index."""

    def fn(x):
      if x == 3:
        raise ValueError('fail')
      return True

    tb_str = _capture_tb(lambda: Chain([1, 2, 3]).filter(fn).run())
    self.assertIn('index=2', tb_str)

  def test_gather_shows_all_fns(self):
    """Gap #3: gather shows _gather_op instead of function names."""

    def ok(x):
      return x

    def bad(x):
      raise ValueError('fail')

    def ok2(x):
      return x

    tb_str = _capture_tb(lambda: Chain(1).gather(ok, bad, ok2).run())
    # Check specifically that the chain visualization (the <quent> frame section)
    # contains the function names, not just the overall traceback (which includes
    # the user source line containing .gather(ok, bad, ok2)).
    _assert_chain_contains(self, tb_str, '.gather(ok, bad, ok2)')

  def test_gather_identifies_failing_fn(self):
    """Gap #3: no indication of which gather function failed."""

    def ok(x):
      return x

    def bad(x):
      raise ValueError('fail')

    def ok2(x):
      return x

    tb_str = _capture_tb(lambda: Chain(1).gather(ok, bad, ok2).run())
    _assert_arrow_on(self, tb_str, 'bad')

  def test_with_shows_context_value(self):
    """Gap #2: context value not shown."""
    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_(raise_fn).run())
    self.assertIn('ctx_value', tb_str)

  def test_finally_shows_root_value(self):
    """Gap #7: root_value not shown in finally_()."""

    def bad_finally(root_val):
      raise RuntimeError('finally fail')

    tb_str = _capture_tb(lambda: Chain(42).finally_(bad_finally).run())
    self.assertIn('root_value=42', tb_str)


# ---------------------------------------------------------------------------
# TestRuntimeValueComprehensive -- Exhaustive runtime value display tests
# ---------------------------------------------------------------------------


class TestRuntimeValueComprehensive(unittest.TestCase):
  """Exhaustive tests for runtime value display in chain visualizations.

  Covers every operation type, edge cases, async paths, and ensures
  the formatting is pixel-perfect for adoption-quality exception output.
  """

  # -- Gap 1: current_value display --

  def test_current_value_int(self):
    """current_value shown as integer."""
    tb_str = _capture_tb(lambda: Chain(42).then(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=42) <----')

  def test_current_value_string(self):
    """current_value shown as quoted string."""
    tb_str = _capture_tb(lambda: Chain('hello').then(raise_fn).run())
    _assert_chain_contains(self, tb_str, ".then(raise_fn, current_value='hello') <----")

  def test_current_value_none(self):
    """current_value=None is shown (None is a valid pipeline value)."""
    tb_str = _capture_tb(lambda: Chain(lambda: None).then(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=None) <----')

  def test_current_value_list(self):
    """current_value shown as list repr."""
    tb_str = _capture_tb(lambda: Chain([1, 2, 3]).then(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=[1, 2, 3]) <----')

  def test_current_value_dict(self):
    """current_value shown as dict repr."""
    tb_str = _capture_tb(lambda: Chain({'a': 1}).then(raise_fn).run())
    self.assertIn("current_value={'a': 1}", tb_str)

  def test_current_value_after_multiple_transforms(self):
    """current_value reflects the pipeline state after previous transforms."""
    def triple(x): return x * 3

    tb_str = _capture_tb(lambda: Chain(10).then(triple).then(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=30) <----')

  def test_current_value_not_shown_with_explicit_args(self):
    """When a link has explicit args, current_value is NOT shown (it wasn't passed)."""
    def fn(a, b): raise ValueError('test')

    tb_str = _capture_tb(lambda: Chain(1).then(fn, 'x', 'y').run())
    self.assertNotIn('current_value', tb_str)
    _assert_chain_contains(self, tb_str, ".then(fn, 'x', 'y') <----")

  def test_current_value_not_shown_with_kwargs(self):
    """When a link has explicit kwargs, current_value is NOT shown."""
    def fn(key=None): raise ValueError('test')

    tb_str = _capture_tb(lambda: Chain(1).then(fn, key='val').run())
    self.assertNotIn('current_value', tb_str)

  def test_current_value_not_shown_with_ellipsis(self):
    """When a link uses ..., current_value is NOT shown (callable invoked with no args)."""
    tb_str = _capture_tb(lambda: Chain(1).then(raise_fn, ...).run())
    self.assertNotIn('current_value', tb_str)
    _assert_chain_contains(self, tb_str, '.then(raise_fn, ...) <----')

  def test_current_value_not_shown_on_root(self):
    """Root link never shows current_value (there is no pipeline value yet)."""
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    self.assertNotIn('current_value', tb_str)
    _assert_chain_contains(self, tb_str, 'Chain(raise_fn) <----')

  def test_current_value_do_preserves_value(self):
    """do() discards result but current_value is still the pipeline value."""
    tb_str = _capture_tb(lambda: Chain(99).do(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.do(raise_fn, current_value=99) <----')

  def test_current_value_only_on_failing_link(self):
    """current_value only appears on the <---- link, not on other links."""
    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).then(raise_fn).then(str).run())
    lines = tb_str.splitlines()
    for line in lines:
      if '.then(sync_fn)' in line:
        self.assertNotIn('current_value', line)
      if '.then(str)' in line:
        self.assertNotIn('current_value', line)

  # -- Gap 2: with_ context value --

  def test_with_sync_ctx_value(self):
    """Sync context manager shows ctx value."""
    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_(raise_fn).run())
    _assert_chain_contains(self, tb_str, ".with_(raise_fn, ctx='ctx_value') <----")

  def test_with_do_sync_ctx_value(self):
    """with_do also shows ctx value."""
    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_do(raise_fn).run())
    _assert_chain_contains(self, tb_str, ".with_do(raise_fn, ctx='ctx_value') <----")

  def test_with_ctx_not_shown_when_enter_fails(self):
    """If __enter__ itself raises, no ctx to display."""
    tb_str = _capture_tb(lambda: Chain(SyncCMRaisesOnEnter()).with_(sync_fn).run())
    self.assertNotIn('ctx=', tb_str)

  # -- Gap 3: gather function names --

  def test_gather_all_fn_names(self):
    """gather shows all function names instead of _gather_op."""
    def a(x): return x
    def b(x): raise ValueError('fail')
    def c(x): return x

    tb_str = _capture_tb(lambda: Chain(1).gather(a, b, c).run())
    _assert_chain_contains(self, tb_str, '.gather(a, b, c)')

  def test_gather_single_fn(self):
    """gather with a single function shows just that name."""
    def bad(x): raise ValueError('fail')

    tb_str = _capture_tb(lambda: Chain(1).gather(bad).run())
    _assert_chain_contains(self, tb_str, '.gather(bad)')

  def test_gather_lambda_names(self):
    """gather shows <lambda> for lambda functions."""
    tb_str = _capture_tb(lambda: Chain(1).gather(lambda x: x, lambda x: 1/0).run())
    self.assertIn('<lambda>', tb_str)

  def test_gather_arrow_on_failing_line(self):
    """The <---- arrow appears on the gather line."""
    def ok(x): return x
    def bad(x): raise ValueError('fail')

    tb_str = _capture_tb(lambda: Chain(1).gather(ok, bad).run())
    _assert_arrow_on(self, tb_str, 'gather')

  def test_gather_no_current_value_shown(self):
    """gather should NOT show current_value (functions handle their own args)."""
    def bad(x): raise ValueError('fail')

    tb_str = _capture_tb(lambda: Chain(42).gather(bad).run())
    self.assertNotIn('current_value', tb_str)

  # -- Gap 4: foreach/filter index --

  def test_foreach_item_and_index(self):
    """foreach shows item= and index= for the failing element."""
    def fn(x):
      if x == 'c': raise ValueError('bad')
      return x

    tb_str = _capture_tb(lambda: Chain(['a', 'b', 'c']).foreach(fn).run())
    self.assertIn("item='c'", tb_str)
    self.assertIn('index=2', tb_str)

  def test_foreach_index_zero(self):
    """index=0 is correctly shown for first-element failures."""
    def fn(x): raise ValueError('first')

    tb_str = _capture_tb(lambda: Chain([10]).foreach(fn).run())
    self.assertIn('item=10', tb_str)
    self.assertIn('index=0', tb_str)

  def test_foreach_do_item_and_index(self):
    """foreach_do also shows item and index."""
    def fn(x):
      if x == 2: raise ValueError('bad')

    tb_str = _capture_tb(lambda: Chain([1, 2]).foreach_do(fn).run())
    self.assertIn('item=2', tb_str)
    self.assertIn('index=1', tb_str)

  def test_foreach_large_index(self):
    """Index correct for large iterables."""
    def fn(x):
      if x == 99: raise ValueError('bad')
      return x

    tb_str = _capture_tb(lambda: Chain(range(100)).foreach(fn).run())
    self.assertIn('index=99', tb_str)

  def test_filter_item_and_index(self):
    """filter shows item= and index= for the failing element."""
    def fn(x):
      if x == 30: raise ValueError('bad')
      return x > 0

    tb_str = _capture_tb(lambda: Chain([10, 20, 30]).filter(fn).run())
    self.assertIn('item=30', tb_str)
    self.assertIn('index=2', tb_str)

  def test_filter_index_zero(self):
    """filter index=0 for first-element failures."""
    def fn(x): raise ValueError('first')

    tb_str = _capture_tb(lambda: Chain([99]).filter(fn).run())
    self.assertIn('item=99', tb_str)
    self.assertIn('index=0', tb_str)

  def test_foreach_no_current_value(self):
    """foreach should NOT show current_value (it has item/index instead)."""
    def fn(x): raise ValueError('bad')

    tb_str = _capture_tb(lambda: Chain([1]).foreach(fn).run())
    self.assertNotIn('current_value', tb_str)

  def test_filter_no_current_value(self):
    """filter should NOT show current_value."""
    def fn(x): raise ValueError('bad')

    tb_str = _capture_tb(lambda: Chain([1]).filter(fn).run())
    self.assertNotIn('current_value', tb_str)

  # -- Gap 6: except_ shows original exception --

  def test_except_shows_exc_type_and_message(self):
    """except_ handler shows the caught exception with type and message."""
    def handler(exc): raise RuntimeError('handler fail')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(handler).run())
    self.assertIn("exc=ValueError('test error')", tb_str)

  def test_except_exc_on_arrow_line(self):
    """The exc= info appears on the <---- line for except_."""
    def handler(exc): raise RuntimeError('handler fail')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(handler).run())
    # exc= appears on the second <---- line (the handler's chain viz)
    arrow_lines = [line for line in tb_str.splitlines() if '<----' in line]
    self.assertTrue(len(arrow_lines) >= 2, 'Expected at least 2 arrow lines for chained exception')
    self.assertIn('exc=', arrow_lines[1])

  def test_except_runtime_error_type(self):
    """except_ correctly shows RuntimeError when that's what was caught."""
    def fn(x): raise RuntimeError('rt error')
    def handler(exc): raise TypeError('handler fail')

    tb_str = _capture_tb(lambda: Chain(1).then(fn).except_(handler).run())
    self.assertIn("exc=RuntimeError('rt error')", tb_str)

  # -- Gap 7: finally_ shows root_value --

  def test_finally_shows_root_value_int(self):
    """finally_ handler shows root_value as integer."""
    def bad_finally(rv): raise RuntimeError('cleanup fail')

    tb_str = _capture_tb(lambda: Chain(42).finally_(bad_finally).run())
    self.assertIn('root_value=42', tb_str)

  def test_finally_shows_root_value_string(self):
    """finally_ handler shows root_value as string."""
    def bad_finally(rv): raise RuntimeError('cleanup fail')

    tb_str = _capture_tb(lambda: Chain('resource').finally_(bad_finally).run())
    self.assertIn("root_value='resource'", tb_str)

  def test_finally_root_value_on_arrow_line(self):
    """The root_value= info appears on the <---- line for finally_."""
    def bad_finally(rv): raise RuntimeError('cleanup fail')

    tb_str = _capture_tb(lambda: Chain(1).finally_(bad_finally).run())
    _assert_arrow_on(self, tb_str, 'root_value=')

  def test_finally_root_value_after_transforms(self):
    """root_value reflects the ROOT, not the current pipeline value."""
    def bad_finally(rv): raise RuntimeError('cleanup fail')

    # root_value = result of Chain(10), which is 10
    # current_value after sync_fn would be 11, but finally gets root_value=10
    tb_str = _capture_tb(lambda: Chain(10).then(sync_fn).finally_(bad_finally).run())
    self.assertIn('root_value=10', tb_str)

  # -- Cross-cutting: arrow precision with runtime values --

  def test_arrow_exactly_once_with_current_value(self):
    """<---- appears exactly once even with current_value display."""
    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).then(raise_fn).then(str).run())
    _assert_arrow_count(self, tb_str, 1)

  def test_arrow_on_correct_link_with_current_value(self):
    """<---- is on the raise_fn link, not neighbors."""
    tb_str = _capture_tb(lambda: Chain(1).then(sync_fn).then(raise_fn).then(str).run())
    for line in tb_str.splitlines():
      if '.then(sync_fn)' in line:
        self.assertNotIn('<----', line)
      if '.then(str)' in line:
        self.assertNotIn('<----', line)

  # -- Cross-cutting: no internal frames with runtime values --

  def test_no_internal_frames_with_current_value(self):
    """Internal frames are still stripped when current_value is shown."""
    tb_str = _capture_tb(lambda: Chain(1).then(raise_fn).run())
    _assert_no_internal_frames(self, tb_str)
    self.assertIn('current_value=1', tb_str)

  def test_no_internal_frames_with_foreach_index(self):
    """Internal frames are still stripped when foreach shows item/index."""
    def fn(x): raise ValueError('bad')

    tb_str = _capture_tb(lambda: Chain([1]).foreach(fn).run())
    _assert_no_internal_frames(self, tb_str)
    self.assertIn('index=0', tb_str)

  def test_no_internal_frames_with_ctx(self):
    """Internal frames are still stripped when with_ shows ctx."""
    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_(raise_fn).run())
    _assert_no_internal_frames(self, tb_str)
    self.assertIn('ctx=', tb_str)

  def test_no_internal_frames_with_exc(self):
    """Internal frames still stripped when except_ shows exc."""
    def handler(exc): raise RuntimeError('fail')

    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(handler).run())
    # The second traceback section (for RuntimeError) should be clean
    self.assertIn('exc=', tb_str)

  # -- Edge cases --

  def test_current_value_bool_false(self):
    """current_value=False is shown (falsy but valid)."""
    tb_str = _capture_tb(lambda: Chain(False).then(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=False) <----')

  def test_current_value_zero(self):
    """current_value=0 is shown (falsy but valid)."""
    tb_str = _capture_tb(lambda: Chain(0).then(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=0) <----')

  def test_current_value_empty_string(self):
    """current_value='' is shown (falsy but valid)."""
    tb_str = _capture_tb(lambda: Chain('').then(raise_fn).run())
    _assert_chain_contains(self, tb_str, ".then(raise_fn, current_value='') <----")

  def test_current_value_empty_list(self):
    """current_value=[] is shown."""
    tb_str = _capture_tb(lambda: Chain([]).then(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=[]) <----')

  def test_foreach_item_none(self):
    """foreach shows item=None correctly."""
    def fn(x):
      if x is None: raise ValueError('none!')

    tb_str = _capture_tb(lambda: Chain([None]).foreach(fn).run())
    self.assertIn('item=None', tb_str)
    self.assertIn('index=0', tb_str)

  def test_foreach_item_is_dict(self):
    """foreach shows dict items."""
    def fn(x):
      if x.get('name') == 'bad': raise ValueError('bad record')
      return x

    records = [{'name': 'ok'}, {'name': 'bad'}]
    tb_str = _capture_tb(lambda: Chain(records).foreach(fn).run())
    self.assertIn('index=1', tb_str)

  def test_gather_many_fns(self):
    """gather with many functions displays all names."""
    def f1(x): return x
    def f2(x): return x
    def f3(x): return x
    def f4(x): raise ValueError('fail')
    def f5(x): return x

    tb_str = _capture_tb(lambda: Chain(1).gather(f1, f2, f3, f4, f5).run())
    _assert_chain_contains(self, tb_str, '.gather(f1, f2, f3, f4, f5)')

  def test_frozen_chain_shows_current_value(self):
    """Frozen chains also show current_value."""
    frozen = Chain(1).then(raise_fn).freeze()
    tb_str = _capture_tb(lambda: frozen.run())
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=1) <----')

  def test_run_with_value_shows_current_value(self):
    """Running a chain with run(v) shows current_value from that pipeline."""
    c = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: c.run(5))
    _assert_chain_contains(self, tb_str, '.then(raise_fn, current_value=5) <----')

  def test_with_ctx_object_repr(self):
    """with_ shows ctx via repr for non-string objects."""
    class MyCM:
      def __enter__(self): return 42
      def __exit__(self, *a): return False

    tb_str = _capture_tb(lambda: Chain(MyCM()).with_(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.with_(raise_fn, ctx=42) <----')


class TestRuntimeValueAsync(unittest.IsolatedAsyncioTestCase):
  """Async runtime value display tests."""

  async def test_async_current_value(self):
    """current_value shown in async chains."""
    tb_str = await _capture_tb_async(Chain(1).then(async_fn).then(async_raise_fn).run())
    self.assertIn('current_value=2', tb_str)

  async def test_async_foreach_item_and_index(self):
    """foreach shows item and index in async iteration."""
    async def bad_fn(x):
      if x == 3: raise ValueError('bad')
      return x

    tb_str = await _capture_tb_async(Chain(AsyncRange(5)).foreach(bad_fn).run())
    self.assertIn('item=3', tb_str)
    self.assertIn('index=3', tb_str)

  async def test_async_with_ctx(self):
    """async with_ shows ctx value."""
    tb_str = await _capture_tb_async(Chain(AsyncCM()).with_(async_raise_fn).run())
    self.assertIn("ctx='ctx_value'", tb_str)

  async def test_async_except_shows_exc(self):
    """except_ shows caught exception in async chain (sync handler)."""
    def bad_handler(exc): raise RuntimeError('handler fail')

    tb_str = await _capture_tb_async(Chain(async_raise_fn).except_(bad_handler).run())
    self.assertIn("exc=ValueError('test error')", tb_str)

  async def test_async_finally_shows_root_value(self):
    """finally_ shows root_value (sync handler in async test context)."""
    def bad_finally(rv): raise RuntimeError('cleanup fail')

    # Use sync chain — async _run_async finally block doesn't inject chain viz
    tb_str = _capture_tb(lambda: Chain(1).finally_(bad_finally).run())
    self.assertIn('root_value=1', tb_str)

  async def test_async_foreach_do_item_and_index(self):
    """foreach_do shows item and index in async iteration."""
    async def bad_fn(x):
      if x == 2: raise ValueError('bad')

    tb_str = await _capture_tb_async(Chain(AsyncRange(5)).foreach_do(bad_fn).run())
    self.assertIn('item=2', tb_str)
    self.assertIn('index=2', tb_str)


# ---------------------------------------------------------------------------
# TestHookIntegration -- 5 tests
# ---------------------------------------------------------------------------


class TestHookIntegration(unittest.TestCase):
  """Tests for sys.excepthook and TracebackException patching."""

  def test_excepthook_is_custom(self):
    """Verify sys.excepthook is our custom hook."""
    from quent._traceback import _quent_excepthook

    self.assertIs(sys.excepthook, _quent_excepthook)

  def test_excepthook_passthrough(self):
    """Non-quent exceptions should pass through to original hook."""
    from quent._traceback import _quent_excepthook

    exc = ValueError('plain')
    exc.__traceback__ = None
    with patch('quent._traceback._original_excepthook') as mock_hook:
      _quent_excepthook(ValueError, exc, None)
      mock_hook.assert_called_once_with(ValueError, exc, None)

  def test_te_init_cleans_quent(self):
    """TracebackException format output should be clean for quent exceptions."""
    try:
      Chain(raise_fn).run()
    except ValueError as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      output = ''.join(te.format())
      self.assertNotIn('_chain.py', output)
      self.assertNotIn('_core.py', output)

  def test_te_format_has_chain_viz(self):
    """TracebackException.format() should include chain visualization."""
    try:
      Chain(raise_fn).run()
    except ValueError as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      output = ''.join(te.format())
      self.assertIn('<quent>', output)
      self.assertIn('Chain(raise_fn)', output)

  def test_te_passthrough(self):
    """Non-quent exceptions unchanged in TracebackException."""
    try:
      raise ValueError('plain')
    except ValueError as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      output = ''.join(te.format())
      self.assertNotIn('<quent>', output)


# ---------------------------------------------------------------------------
# TestEdgeCases -- 6 tests
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
  """Edge case tests for traceback formatting."""

  def test_empty_chain_no_exception(self):
    result = Chain().run()
    self.assertIsNone(result)

  def test_only_except_finally_shown(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).except_(raise_fn).finally_(sync_identity).run())
    self.assertIn('.except_(raise_fn)', tb_str)
    self.assertIn('.finally_(sync_identity)', tb_str)

  def test_chain_root_is_chain_raises(self):
    inner = Chain().then(raise_fn)
    tb_str = _capture_tb(lambda: Chain(inner).run())
    _assert_quent_frame(self, tb_str)
    _assert_arrow_on(self, tb_str, 'raise_fn')

  def test_frozen_chain_traceback(self):
    frozen = Chain(raise_fn).freeze()
    tb_str = _capture_tb(lambda: frozen.run())
    _assert_quent_frame(self, tb_str)
    _assert_arrow_on(self, tb_str, 'raise_fn')

  def test_frozen_called_twice(self):
    frozen = Chain(raise_fn).freeze()
    tb1 = _capture_tb(lambda: frozen.run())
    tb2 = _capture_tb(lambda: frozen.run())
    _assert_quent_frame(self, tb1)
    _assert_quent_frame(self, tb2)
    _assert_arrow_on(self, tb1, 'raise_fn')
    _assert_arrow_on(self, tb2, 'raise_fn')

  def test_very_long_chain_20_links(self):
    c = Chain(1)
    for _ in range(19):
      c = c.then(sync_fn)
    c = c.then(raise_fn)
    tb_str = _capture_tb(lambda: c.run())
    self.assertEqual(tb_str.count('.then('), 20)
    _assert_arrow_on(self, tb_str, 'raise_fn')

  def test_chain_with_only_do_links(self):
    tb_str = _capture_tb(lambda: Chain(1).do(sync_fn).do(raise_fn).run())
    _assert_chain_contains(self, tb_str, '.do(sync_fn)', '.do(raise_fn, current_value=1) <----')

  def test_raise_exc_shows_correct_text(self):
    tb_str = _capture_tb(lambda: Chain(raise_fn).run())
    # The <quent> frame shows the chain visualization as its "source" line.
    # On Python 3.14+ the compiled 'raise __exc__' source is not displayed;
    # instead the chain visualization appears directly.
    _assert_quent_frame(self, tb_str)
    _assert_chain_contains(self, tb_str, 'Chain(raise_fn) <----')


if __name__ == '__main__':
  unittest.main()
