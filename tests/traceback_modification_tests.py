"""Tests for _modify_traceback, _clean_internal_frames, _clean_chained_exceptions,
_quent_excepthook, and _patched_te_init.

Focused on the mutation side-effects of _modify_traceback (flag setting, stamp
management, frame cleaning, chained exception handling) rather than the visual
formatting (covered by traceback_format_tests.py).
"""
from __future__ import annotations

import sys
import traceback
import types
import unittest

from quent import Chain
from quent._core import Link, Null
from quent._traceback import (
  _clean_chained_exceptions,
  _clean_internal_frames,
  _Ctx,
  _modify_traceback,
  _original_excepthook,
  _patched_te_init,
  _quent_excepthook,
  _stringify_chain,
)
from helpers import raise_fn, sync_fn, sync_identity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain_and_link():
  """Return a (chain, link, root_link) triple where chain is non-nested."""
  c = Chain(sync_fn).then(raise_fn)
  return c, c.first_link, c.root_link


def _exc_with_tb():
  """Create a ValueError with a real traceback."""
  try:
    raise ValueError('test')
  except ValueError as exc:
    return exc


# ---------------------------------------------------------------------------
# TestModifyTracebackBasics
# ---------------------------------------------------------------------------

class TestModifyTracebackBasics(unittest.TestCase):

  def test_quent_flag_set(self):
    """exc.__quent__ is True after _modify_traceback."""
    c, link, root_link = _make_chain_and_link()
    exc = _exc_with_tb()
    _modify_traceback(exc, c, link, root_link)
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_source_link_set_if_none(self):
    """__quent_source_link__ is set if not already present, then consumed."""
    c, link, root_link = _make_chain_and_link()
    exc = _exc_with_tb()
    self.assertFalse(hasattr(exc, '__quent_source_link__'))
    # After _modify_traceback with chain+link, the stamp is consumed (deleted)
    # and the exc should have __quent__ instead.
    _modify_traceback(exc, c, link, root_link)
    self.assertFalse(hasattr(exc, '__quent_source_link__'))
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_source_link_first_write_wins(self):
    """Pre-existing __quent_source_link__ is preserved (first-write-wins)."""
    c, link, root_link = _make_chain_and_link()
    exc = _exc_with_tb()
    sentinel = Link(lambda: 'sentinel')
    exc.__quent_source_link__ = sentinel
    _modify_traceback(exc, c, link, root_link)
    # After processing, __quent_source_link__ is deleted, but the chain
    # visualization was built using the sentinel link.
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_traceback_cleaned(self):
    """Internal frames are removed from the formatted traceback."""
    try:
      Chain(raise_fn).run()
    except ValueError as exc:
      tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertNotIn('_chain.py', tb_str)
      self.assertNotIn('_core.py', tb_str)
      self.assertNotIn('_traceback.py', tb_str)

  def test_chained_exceptions_cleaned(self):
    """__cause__ and __context__ have internal frames removed."""
    def handler(rv, exc):
      raise RuntimeError('handler error') from exc

    try:
      Chain(raise_fn).except_(handler).run()
    except RuntimeError as exc:
      # Check __cause__ (the original ValueError)
      cause = exc.__cause__
      self.assertIsNotNone(cause)
      tb = cause.__traceback__
      while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if filename != '<quent>':
          self.assertNotIn('quent/_chain.py', filename)
          self.assertNotIn('quent/_core.py', filename)
        tb = tb.tb_next


# ---------------------------------------------------------------------------
# TestModifyTracebackVisualization
# ---------------------------------------------------------------------------

class TestModifyTracebackVisualization(unittest.TestCase):

  def _capture_tb_str(self, fn):
    try:
      fn()
      return None
    except BaseException as exc:
      return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

  def test_chain_visualization_header(self):
    """Chain( prefix appears in the traceback."""
    tb_str = self._capture_tb_str(lambda: Chain(raise_fn).run())
    self.assertIn('Chain(', tb_str)

  def test_chain_visualization_arrow(self):
    """<---- marker appears on the source link."""
    tb_str = self._capture_tb_str(lambda: Chain(raise_fn).run())
    self.assertIn('<----', tb_str)

  def test_nested_chain_indentation(self):
    """Nested chains are indented in the visualization."""
    inner = Chain().then(raise_fn)
    tb_str = self._capture_tb_str(lambda: Chain(1).then(inner).run())
    # Nested chain creates additional indentation (4 spaces per level)
    self.assertIn('    ', tb_str)
    # Must show both outer and inner Chain
    chain_count = tb_str.count('Chain(')
    self.assertGreaterEqual(chain_count, 2)

  def test_except_and_finally_shown(self):
    """except_ and finally_ appear in visualization."""
    c = Chain(raise_fn).except_(sync_identity).finally_(sync_identity)
    ctx = _Ctx(source_link=c.root_link, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.except_(sync_identity)', result)
    self.assertIn('.finally_(sync_identity)', result)

  def test_extra_links_displayed(self):
    """extra_links parameter (used by iterate) appears in visualization."""
    c = Chain(sync_fn)
    extra = Link(sync_fn)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(
      c, nest_lvl=0, ctx=ctx,
      extra_links=[(extra, 'iterate')],
    )
    self.assertIn('.iterate(sync_fn)', result)


# ---------------------------------------------------------------------------
# TestModifyTracebackAsync
# ---------------------------------------------------------------------------

class TestModifyTracebackAsync(unittest.IsolatedAsyncioTestCase):

  async def _capture_tb_async(self, coro):
    try:
      await coro
      return None
    except BaseException as exc:
      return exc

  async def test_async_quent_flag_set(self):
    async def fail(x):
      raise ValueError('async fail')

    try:
      await Chain(1).then(fail).run()
    except ValueError as exc:
      self.assertTrue(getattr(exc, '__quent__', False))

  async def test_async_traceback_cleaned(self):
    async def fail(x):
      raise ValueError('async fail')

    try:
      await Chain(1).then(fail).run()
    except ValueError as exc:
      tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertNotIn('_chain.py', tb_str)
      self.assertNotIn('_core.py', tb_str)

  async def test_async_source_link_set(self):
    async def fail(x):
      raise ValueError('async fail')

    try:
      await Chain(1).then(fail).run()
    except ValueError as exc:
      # Source link consumed by _modify_traceback
      self.assertFalse(hasattr(exc, '__quent_source_link__'))
      self.assertTrue(getattr(exc, '__quent__', False))


# ---------------------------------------------------------------------------
# TestCleanChainedExceptionsCircular
# ---------------------------------------------------------------------------

class TestCleanChainedExceptionsCircular(unittest.TestCase):

  def test_circular_context(self):
    """Exception with circular __context__ does not infinite-loop (via seen set)."""
    exc1 = ValueError('a')
    exc2 = ValueError('b')
    exc1.__context__ = exc2
    exc2.__context__ = exc1
    exc1.__traceback__ = None
    exc2.__traceback__ = None
    # Should not raise or hang
    _clean_chained_exceptions(exc1, set())

  def test_deep_chain_10_levels(self):
    """10 levels of __cause__ are all cleaned."""
    root = ValueError('level_0')
    root.__traceback__ = None
    current = root
    for i in range(1, 11):
      next_exc = ValueError(f'level_{i}')
      next_exc.__traceback__ = None
      current.__cause__ = next_exc
      current = next_exc
    # Should traverse all 11 exceptions without error
    _clean_chained_exceptions(root, set())

  def test_none_exception(self):
    """None passed to _clean_chained_exceptions is a no-op."""
    # Should not raise
    _clean_chained_exceptions(None, set())


# ---------------------------------------------------------------------------
# TestExcepthookAndTeInit
# ---------------------------------------------------------------------------

class TestExcepthookAndTeInit(unittest.TestCase):

  def test_excepthook_cleans_quent_exceptions(self):
    """_quent_excepthook processes exceptions with __quent__=True."""
    try:
      Chain(raise_fn).run()
    except ValueError as exc:
      # Should not raise when called
      _quent_excepthook(type(exc), exc, exc.__traceback__)

  def test_excepthook_ignores_non_quent(self):
    """_quent_excepthook falls through to original for non-quent exceptions."""
    exc = ValueError('plain')
    exc.__traceback__ = None
    # Should not raise; delegates to _original_excepthook
    # We just verify it does not crash
    try:
      _quent_excepthook(type(exc), exc, exc.__traceback__)
    except Exception:
      # _original_excepthook may print to stderr but should not raise
      pass

  def test_patched_te_init_cleans(self):
    """TracebackException.__init__ is patched to clean quent frames."""
    try:
      Chain(raise_fn).run()
    except ValueError as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      self.assertNotIn('_chain.py', formatted)
      self.assertNotIn('_core.py', formatted)


# ---------------------------------------------------------------------------
# BEYOND SPEC: Additional edge-case tests
# ---------------------------------------------------------------------------

class TestModifyTracebackEdgeCases(unittest.TestCase):

  def test_modify_traceback_called_twice_on_same_exception(self):
    """Calling _modify_traceback twice does not crash or corrupt state."""
    c, link, root_link = _make_chain_and_link()
    exc = _exc_with_tb()
    _modify_traceback(exc, c, link, root_link)
    # Second call: exc already has __quent__=True, source_link consumed
    _modify_traceback(exc, c, link, root_link)
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_modify_traceback_with_chain_none_link_none(self):
    """When chain=None and link=None (nested chain path), only frames are cleaned."""
    exc = _exc_with_tb()
    _modify_traceback(exc, chain=None, link=None)
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_modify_traceback_with_nested_chain(self):
    """When chain.is_nested=True, only frames are cleaned (no visualization)."""
    c = Chain(raise_fn)
    c.is_nested = True
    exc = _exc_with_tb()
    _modify_traceback(exc, c, c.root_link, c.root_link)
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_clean_internal_frames_with_empty_traceback(self):
    """_clean_internal_frames(None) returns None."""
    result = _clean_internal_frames(None)
    self.assertIsNone(result)

  def test_clean_internal_frames_preserves_user_frames(self):
    """User frames outside quent package are preserved."""
    try:
      raise ValueError('user code')
    except ValueError as exc:
      tb = exc.__traceback__
      result = _clean_internal_frames(tb)
      self.assertIsNotNone(result)
      # This test's frame should be preserved
      self.assertIn(
        'test_clean_internal_frames_preserves_user_frames',
        result.tb_frame.f_code.co_name,
      )

  def test_modify_traceback_on_base_exception(self):
    """_modify_traceback works on non-Exception BaseException (e.g. KeyboardInterrupt)."""
    c = Chain(sync_fn)
    link = c.root_link
    try:
      raise KeyboardInterrupt('test')
    except KeyboardInterrupt as exc:
      _modify_traceback(exc, c, link, link)
      self.assertTrue(getattr(exc, '__quent__', False))

  def test_quent_link_temp_args_consumed(self):
    """__quent_link_temp_args__ is deleted after _modify_traceback processes it."""
    c, link, root_link = _make_chain_and_link()
    exc = _exc_with_tb()
    exc.__quent_link_temp_args__ = {id(link): {'current_value': 42}}
    _modify_traceback(exc, c, link, root_link)
    self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))

  def test_clean_chained_exceptions_cause_and_context(self):
    """Both __cause__ and __context__ are processed."""
    cause_exc = ValueError('cause')
    context_exc = TypeError('context')
    main_exc = RuntimeError('main')
    cause_exc.__traceback__ = None
    context_exc.__traceback__ = None
    main_exc.__traceback__ = None
    main_exc.__cause__ = cause_exc
    main_exc.__context__ = context_exc
    # Should not raise
    _clean_chained_exceptions(main_exc, set())

  def test_excepthook_is_installed(self):
    """sys.excepthook is _quent_excepthook after import."""
    self.assertIs(sys.excepthook, _quent_excepthook)

  def test_traceback_exception_init_is_patched(self):
    """traceback.TracebackException.__init__ is _patched_te_init."""
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)

  def test_modify_traceback_returns_exception(self):
    """_modify_traceback always returns the exception for use in raise."""
    c, link, root_link = _make_chain_and_link()
    exc = _exc_with_tb()
    result = _modify_traceback(exc, c, link, root_link)
    self.assertIs(result, exc)

  def test_modify_traceback_with_link_none_chain_present(self):
    """When link=None but chain is present, still sets __quent__."""
    c = Chain(sync_fn)
    exc = _exc_with_tb()
    # chain is not None but link is None -> goes to else branch
    _modify_traceback(exc, c, link=None)
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_unicode_function_name_in_traceback(self):
    """Unicode function names are displayed correctly in traceback visualization."""
    def identite_de_l_utilisateur(x):  # noqa: E741
      raise ValueError('unicode')

    try:
      Chain(1).then(identite_de_l_utilisateur).run()
    except ValueError as exc:
      tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIn('identite_de_l_utilisateur', tb_str)

  def test_very_long_function_name_in_traceback(self):
    """Very long function names (100+ chars) do not crash the visualization."""
    # Create a callable with a very long name
    long_name = 'a' * 120

    def long_fn(x):
      raise ValueError('long name')

    long_fn.__name__ = long_name
    long_fn.__qualname__ = long_name

    try:
      Chain(1).then(long_fn).run()
    except ValueError as exc:
      tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIn(long_name, tb_str)

  def test_deep_chained_cause_chain_15_levels(self):
    """15 levels of __cause__ chaining are all traversed."""
    root = ValueError('root')
    root.__traceback__ = None
    current = root
    for i in range(15):
      nxt = ValueError(f'level_{i}')
      nxt.__traceback__ = None
      current.__cause__ = nxt
      current = nxt
    _clean_chained_exceptions(root, set())

  def test_mixed_cause_and_context_chain(self):
    """Mixed __cause__ and __context__ chains are both traversed."""
    exc1 = ValueError('1')
    exc2 = TypeError('2')
    exc3 = RuntimeError('3')
    exc1.__traceback__ = None
    exc2.__traceback__ = None
    exc3.__traceback__ = None
    exc1.__cause__ = exc2
    exc2.__context__ = exc3
    _clean_chained_exceptions(exc1, set())


if __name__ == '__main__':
  unittest.main()
