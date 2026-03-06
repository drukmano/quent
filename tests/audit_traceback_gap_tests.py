"""Audit tests covering untested gaps in quent/_traceback.py.

Gaps covered:
  1. ObjWithBadName / ObjWithBadNameAndRepr fallback paths in _get_obj_name
  2. _modify_traceback visualization error fallback (H8 fix)
  3. disable_traceback_patching / enable_traceback_patching effect + idempotency
  4. else_() branch visibility in chain visualization (L20 fix)
  5. _get_link_name coverage for 'if' operation type
  6. kwargs-only nested chain visualization (L19 fix)
  7. _format_call_args edge cases
"""
from __future__ import annotations

import sys
import traceback
import unittest

from quent import Chain, Null, QuentException
from quent._traceback import (
  _modify_traceback,
  _Ctx,
  _stringify_chain,
  _format_call_args,
  _get_obj_name,
  _get_link_name,
  _clean_internal_frames,
  enable_traceback_patching,
  disable_traceback_patching,
  _quent_excepthook,
  _patched_te_init,
  _original_excepthook,
  _resolve_nested_chain,
  _format_link,
)
from quent._core import Link, Null
from quent._ops import _make_foreach, _make_filter, _make_gather, _make_with, _make_if
from helpers import (
  ObjWithBadName,
  ObjWithBadNameAndRepr,
  sync_fn,
  raise_fn,
  sync_identity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exc_with_tb():
  """Create a ValueError with a real traceback."""
  try:
    raise ValueError('test')
  except ValueError as exc:
    return exc


# ---------------------------------------------------------------------------
# Gap 1: ObjWithBadName / ObjWithBadNameAndRepr helpers
# ---------------------------------------------------------------------------

class TestGetObjNameFallbackPaths(unittest.TestCase):
  """_get_obj_name fallback paths exercised by helpers with broken attributes."""

  def test_obj_with_bad_name_falls_to_repr(self):
    """ObjWithBadName: __name__/__qualname__ raise RuntimeError, fallback to repr()."""
    obj = ObjWithBadName()
    result = _get_obj_name(obj)
    self.assertEqual(result, '<ObjWithBadName>')

  def test_obj_with_bad_name_and_repr_falls_to_type_name(self):
    """ObjWithBadNameAndRepr: both name and repr raise, fallback to type(obj).__name__."""
    obj = ObjWithBadNameAndRepr()
    result = _get_obj_name(obj)
    self.assertEqual(result, 'ObjWithBadNameAndRepr')

  def test_obj_with_bad_name_in_chain_visualization(self):
    """ObjWithBadName used as a link value renders via repr fallback in stringify."""
    c = Chain(ObjWithBadName())
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('<ObjWithBadName>', result)

  def test_obj_with_bad_name_and_repr_in_chain_visualization(self):
    """ObjWithBadNameAndRepr used as a link value renders via type name fallback."""
    c = Chain(ObjWithBadNameAndRepr())
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('ObjWithBadNameAndRepr', result)


# ---------------------------------------------------------------------------
# Gap 2: _modify_traceback visualization error fallback (H8 fix)
# ---------------------------------------------------------------------------

class TestModifyTracebackVisualizationFallback(unittest.TestCase):
  """When _stringify_chain raises, _modify_traceback preserves the original exception."""

  def test_broken_repr_preserves_original_exception(self):
    """A chain with a broken-repr object still yields the original exception type/message.

    The object must be well-behaved during chain execution (so _chain.py
    can call getattr(link.v, '_quent_op', None) without error), but broken
    only during visualization (when _get_obj_name calls repr/name).
    """
    call_count = 0

    class BreaksOnSecondRepr:
      """repr() works the first time (chain execution), breaks on second (visualization)."""
      def __repr__(self):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
          raise RuntimeError('broken repr on visualization')
        return '<BreaksOnSecondRepr>'

      def __call__(self, x):
        raise ValueError('original error')

    try:
      Chain(1).then(BreaksOnSecondRepr()).run()
    except ValueError as exc:
      # The original exception type and message must be preserved.
      self.assertEqual(str(exc), 'original error')
      self.assertTrue(getattr(exc, '__quent__', False))
    except Exception as exc:
      self.fail(f'Expected ValueError but got {type(exc).__name__}: {exc}')

  def test_visualization_failure_still_sets_quent_flag(self):
    """Even when visualization fails, __quent__ flag is set on the exception."""

    class ReprBreaksOnViz:
      """__name__/__qualname__ raise non-AttributeError, repr also raises."""
      _call_count = 0

      def __repr__(self):
        self._call_count += 1
        if self._call_count > 1:
          raise RuntimeError('bad repr')
        return '<ReprBreaksOnViz>'

      def __call__(self, x):
        raise TypeError('inner error')

    try:
      Chain(1).then(ReprBreaksOnViz()).run()
    except TypeError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))

  def test_visualization_failure_cleans_frames(self):
    """Even when visualization fails, internal frames are cleaned from traceback."""

    class BrokenForViz:
      _call_count = 0

      def __repr__(self):
        self._call_count += 1
        if self._call_count > 1:
          raise RuntimeError('repr broken')
        return '<BrokenForViz>'

      def __call__(self, x):
        raise RuntimeError('user error')

    try:
      Chain(1).then(BrokenForViz()).run()
    except RuntimeError as exc:
      if str(exc) == 'user error':
        tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.assertNotIn('_chain.py', tb_str)
        self.assertNotIn('_core.py', tb_str)
      else:
        self.fail(f'Got unexpected RuntimeError: {exc}')


# ---------------------------------------------------------------------------
# Gap 3: disable_traceback_patching / enable_traceback_patching
# ---------------------------------------------------------------------------

class TestTracebackPatchingToggle(unittest.TestCase):
  """Enable/disable traceback patching, including idempotency."""

  def setUp(self):
    # Ensure patching is enabled at the start of each test.
    enable_traceback_patching()

  def tearDown(self):
    # Always restore patching after each test.
    enable_traceback_patching()

  def test_disable_restores_excepthook(self):
    """disable_traceback_patching restores sys.excepthook to original."""
    self.assertIs(sys.excepthook, _quent_excepthook)
    disable_traceback_patching()
    self.assertIs(sys.excepthook, _original_excepthook)

  def test_disable_restores_te_init(self):
    """disable_traceback_patching restores TracebackException.__init__."""
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)
    disable_traceback_patching()
    self.assertIsNot(traceback.TracebackException.__init__, _patched_te_init)

  def test_enable_installs_hooks(self):
    """enable_traceback_patching installs quent's hooks."""
    disable_traceback_patching()
    self.assertIsNot(sys.excepthook, _quent_excepthook)
    enable_traceback_patching()
    self.assertIs(sys.excepthook, _quent_excepthook)
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)

  def test_enable_idempotent(self):
    """Calling enable_traceback_patching twice does not break anything."""
    enable_traceback_patching()
    enable_traceback_patching()
    self.assertIs(sys.excepthook, _quent_excepthook)
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)

  def test_disable_idempotent(self):
    """Calling disable_traceback_patching twice does not break anything."""
    disable_traceback_patching()
    disable_traceback_patching()
    self.assertIs(sys.excepthook, _original_excepthook)

  def test_toggle_roundtrip(self):
    """disable -> enable -> verify hooks are reinstalled."""
    disable_traceback_patching()
    self.assertIs(sys.excepthook, _original_excepthook)
    enable_traceback_patching()
    self.assertIs(sys.excepthook, _quent_excepthook)
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)

  def test_disable_enable_disable(self):
    """Multiple toggles do not corrupt state."""
    disable_traceback_patching()
    enable_traceback_patching()
    disable_traceback_patching()
    self.assertIs(sys.excepthook, _original_excepthook)
    self.assertIsNot(traceback.TracebackException.__init__, _patched_te_init)


# ---------------------------------------------------------------------------
# Gap 4: else_() branch visibility in chain visualization (L20 fix)
# ---------------------------------------------------------------------------

class TestElseBranchVisualization(unittest.TestCase):
  """else_() branches appear in the chain visualization string."""

  def test_else_appears_in_stringify(self):
    """_stringify_chain output contains '.else_(...)' for chains with else_()."""
    c = Chain(10).if_(lambda x: x > 5, lambda x: x * 2).else_(lambda x: x + 1)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.else_', result)

  def test_else_appears_in_traceback_on_error(self):
    """else_() is visible in the traceback visualization when the chain errors."""
    c = (
      Chain(10)
      .if_(lambda x: x > 5, lambda x: x * 2)
      .else_(lambda x: x + 1)
      .then(raise_fn)
    )
    try:
      c.run()
    except ValueError as exc:
      tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIn('.else_', tb_str)

  def test_else_and_if_both_visible(self):
    """Both .if_(...) and .else_(...) appear in the visualization."""
    c = Chain(10).if_(lambda x: x > 5, lambda x: x * 2).else_(lambda x: x + 1)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.if_', result)
    self.assertIn('.else_', result)

  def test_else_with_named_function(self):
    """else_ shows the function name in the visualization."""
    def my_else_handler(x):
      return x + 1

    c = Chain(10).if_(lambda x: x > 5, lambda x: x * 2).else_(my_else_handler)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('my_else_handler', result)

  def test_else_arrow_marking(self):
    """The <---- arrow can point to the else_ link if it is the source."""
    def my_else_fn(x):
      raise ValueError('else error')

    c = Chain(3).if_(lambda x: x > 5, sync_fn).else_(my_else_fn)
    # The else link is stored on the if op's _else_link attribute
    if_link = c.first_link
    else_link = if_link.v._else_link
    ctx = _Ctx(source_link=else_link, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.else_(my_else_fn) <----', result)


# ---------------------------------------------------------------------------
# Gap 5: _get_link_name coverage for 'if' operation type
# ---------------------------------------------------------------------------

class TestGetLinkNameIfOp(unittest.TestCase):
  """_get_link_name returns 'if_' for links wrapping _make_if operations."""

  def test_if_link_name(self):
    """A link with _quent_op='if' returns 'if_'."""
    pred_link = Link(lambda x: x > 0)
    fn_link = Link(lambda x: x * 2)
    wrapper = _make_if(pred_link, fn_link)
    link = Link(wrapper, original_value=fn_link)
    self.assertEqual(_get_link_name(link), 'if_')

  def test_if_via_chain_api(self):
    """Chain.if_() creates a link whose _get_link_name is 'if_'."""
    c = Chain(1).if_(lambda x: x > 0, sync_fn)
    link = c.first_link
    self.assertEqual(_get_link_name(link), 'if_')

  def test_if_in_stringify(self):
    """_stringify_chain shows .if_(...) for if operations."""
    c = Chain(1).if_(lambda x: x > 0, sync_fn)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.if_', result)


# ---------------------------------------------------------------------------
# Gap 6: kwargs-only nested chain visualization (L19 fix)
# ---------------------------------------------------------------------------

class TestKwargsOnlyNestedChain(unittest.TestCase):
  """_resolve_nested_chain with kwargs only and no positional args."""

  def test_kwargs_only_shown(self):
    """kwargs={'k': 1} with no positional args shows 'k=' in output."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, None, {'k': 1}, 0, ctx)
    self.assertIn('k=', result)

  def test_kwargs_only_multiple(self):
    """Multiple kwargs all appear in the output."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, None, {'a': 1, 'b': 2}, 0, ctx)
    self.assertIn('a=', result)
    self.assertIn('b=', result)

  def test_empty_args_with_kwargs(self):
    """args=() with kwargs still shows kwargs."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, (), {'key': 'val'}, 0, ctx)
    self.assertIn('key=', result)


# ---------------------------------------------------------------------------
# Gap 7: _format_call_args edge cases
# ---------------------------------------------------------------------------

class TestFormatCallArgsEdgeCases(unittest.TestCase):
  """Edge cases for _format_call_args beyond basic usage."""

  def test_ellipsis_first_arg(self):
    """Ellipsis as first arg produces '...'."""
    result = _format_call_args((...,), None)
    self.assertEqual(result, '...')

  def test_empty_args_empty_kwargs(self):
    """Empty args tuple and empty kwargs dict produce ''."""
    result = _format_call_args((), {})
    self.assertEqual(result, '')

  def test_none_args_none_kwargs(self):
    """None args and None kwargs produce ''."""
    result = _format_call_args(None, None)
    self.assertEqual(result, '')

  def test_args_and_kwargs_combined(self):
    """Both args and kwargs are comma-separated."""
    result = _format_call_args((42,), {'flag': True})
    self.assertEqual(result, '42, flag=True')

  def test_ellipsis_with_kwargs(self):
    """Ellipsis as first arg combined with kwargs."""
    result = _format_call_args((...,), {'x': 1})
    self.assertEqual(result, '..., x=1')

  def test_multiple_positional_args(self):
    """Multiple positional args are comma-separated."""
    result = _format_call_args((1, 2, 3), None)
    self.assertEqual(result, '1, 2, 3')

  def test_callable_in_kwargs(self):
    """Callable value in kwargs uses _get_obj_name."""
    result = _format_call_args(None, {'fn': sync_fn})
    self.assertEqual(result, 'fn=sync_fn')

  def test_obj_with_bad_name_in_args(self):
    """ObjWithBadName in args falls back correctly."""
    result = _format_call_args((ObjWithBadName(),), None)
    self.assertEqual(result, '<ObjWithBadName>')

  def test_obj_with_bad_name_and_repr_in_kwargs(self):
    """ObjWithBadNameAndRepr in kwargs falls back to type name."""
    result = _format_call_args(None, {'obj': ObjWithBadNameAndRepr()})
    self.assertEqual(result, 'obj=ObjWithBadNameAndRepr')


# ---------------------------------------------------------------------------
# Additional cross-gap integration tests
# ---------------------------------------------------------------------------

class TestFormatLinkWithIfElse(unittest.TestCase):
  """Integration: _format_link correctly renders if/else in context."""

  def test_format_link_if_operation(self):
    """_format_link renders an if link with its predicate name."""
    c = Chain(10).if_(lambda x: x > 5, sync_fn)
    link = c.first_link
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _format_link(link, nest_lvl=0, ctx=ctx, method_name='if_')
    self.assertIn('.if_', result)
    self.assertIn('sync_fn', result)

  def test_format_link_else_operation(self):
    """_format_link renders an else link with its callable name."""
    c = Chain(10).if_(lambda x: x > 5, sync_fn).else_(sync_identity)
    if_link = c.first_link
    else_link = if_link.v._else_link
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _format_link(else_link, nest_lvl=0, ctx=ctx, method_name='else_')
    self.assertIn('.else_', result)
    self.assertIn('sync_identity', result)


class TestModifyTracebackWithBrokenVisualizationDirect(unittest.TestCase):
  """Direct _modify_traceback call with a chain that will fail visualization."""

  def test_direct_call_with_broken_stringify(self):
    """Calling _modify_traceback directly when stringify would raise."""
    c = Chain(1).then(sync_fn)
    exc = _exc_with_tb()

    # Temporarily sabotage _get_obj_name to force visualization failure
    import quent._traceback as tb_mod
    original_get_obj_name = tb_mod._get_obj_name

    def broken_get_obj_name(obj):
      raise RuntimeError('forced visualization failure')

    tb_mod._get_obj_name = broken_get_obj_name
    try:
      result = _modify_traceback(exc, c, c.root_link, c.root_link)
      # Exception should still be returned
      self.assertIs(result, exc)
      # __quent__ should be set
      self.assertTrue(getattr(exc, '__quent__', False))
      # Internal frames should still be cleaned
      self.assertFalse(hasattr(exc, '__quent_source_link__'))
    finally:
      tb_mod._get_obj_name = original_get_obj_name


if __name__ == '__main__':
  unittest.main()
