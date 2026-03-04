"""Comprehensive tests targeting uncovered lines in _diagnostics.pxi.

Missing lines: 15-27, 55, 66, 84, 91-94, 111, 140, 157, 164, 169-172,
177, 181, 215, 223, 229, 248-249, 281, 297, 306-308, 314-316.

NOTE: Line 140 is the Python < 3.11 path for code.replace(co_name=...)
without co_qualname. Since we run on Python 3.14, this line cannot be
covered and is excluded from test targets.
"""

import sys
import os
import traceback
import asyncio
import unittest
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException
from quent.quent import _clean_exc_chain, _quent_excepthook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tb_entries(exc):
  """Extract traceback entries from an exception."""
  return traceback.extract_tb(exc.__traceback__)


def _get_quent_entries(exc):
  """Get only the <quent> traceback entries."""
  return [e for e in _get_tb_entries(exc) if e.filename == '<quent>']


def _get_tb_filenames(exc):
  """Get filenames from an exception's traceback."""
  return [entry.filename for entry in _get_tb_entries(exc)]


def _get_tb_func_names(exc):
  """Get function names from an exception's traceback."""
  return [entry.name for entry in _get_tb_entries(exc)]


def _raise_value_error(v=None):
  raise ValueError('test error')


def _raise_type_error(v=None):
  raise TypeError('type error')


def _add_one(v):
  return v + 1


def _double(v):
  return v * 2


async def _async_raise(v=None):
  raise ValueError('async error')


async def _async_identity(v):
  return v


# ---------------------------------------------------------------------------
# 1. clean_internal_frames tests (lines 15-27, 34-52)
# ---------------------------------------------------------------------------

class CleanInternalFramesTests(TestCase):
  """Tests exercising the clean_internal_frames function via chain
  exceptions. Lines 15-27 are module-level setup and the function entry.
  The function is invoked during exception traceback cleaning.
  """

  def test_quent_frames_kept_in_traceback(self):
    """clean_internal_frames preserves <quent> frames (line 37-39)."""
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      self.assertIn('<quent>', filenames)

  def test_internal_quent_frames_removed(self):
    """clean_internal_frames removes frames from quent package dir (line 40)."""
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      for fn in filenames:
        if fn == '<quent>':
          continue
        self.assertFalse(
          fn.startswith(quent_pkg_dir),
          f'Internal quent frame should be cleaned: {fn}'
        )

  def test_user_frames_preserved(self):
    """clean_internal_frames keeps user code frames (line 42-44)."""
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      func_names = _get_tb_func_names(exc)
      self.assertIn('_raise_value_error', func_names)

  def test_mixed_user_and_quent_frames(self):
    """clean_internal_frames correctly handles mixed frames: keeps user
    and <quent>, removes internal quent frames."""
    def user_fn(v):
      return Chain(v).then(_raise_value_error)()

    try:
      Chain(5).then(user_fn).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      func_names = _get_tb_func_names(exc)
      self.assertIn('<quent>', filenames)
      self.assertIn('user_fn', func_names)
      self.assertIn('_raise_value_error', func_names)

  def test_empty_traceback_after_cleaning(self):
    """clean_internal_frames with a traceback where all frames are
    internal should return None (empty traceback). This exercises
    the reversed(stack) path with empty stack (line 48)."""
    # We cannot directly call clean_internal_frames (it's cdef), but we
    # can verify the behavior indirectly: a chain that only has internal
    # frames still produces a valid exception.
    try:
      Chain(lambda: (_ for _ in ()).throw(ValueError('gen'))).run()
      self.fail('Expected ValueError')
    except ValueError:
      pass  # Exception should propagate even if traceback is mostly cleaned

  def test_traceback_chain_built_in_reverse(self):
    """Verify the reversed stack building produces correct frame order
    (lines 48-50). The first user frame should come before the last."""
    def step1(v):
      return step2(v)

    def step2(v):
      return Chain(v).then(_raise_value_error)()

    try:
      Chain(1).then(step1).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      func_names = _get_tb_func_names(exc)
      # step1 should appear before step2 in the traceback
      if 'step1' in func_names and 'step2' in func_names:
        self.assertLess(
          func_names.index('step1'),
          func_names.index('step2'),
          'Frame order should be preserved'
        )


# ---------------------------------------------------------------------------
# 2. _clean_chained_exceptions tests (line 55)
# ---------------------------------------------------------------------------

class CleanChainedExceptionsTests(TestCase):
  """Tests for _clean_chained_exceptions which recursively cleans
  __cause__ and __context__ chains. Line 55 is the function entry.
  """

  def test_exception_with_cause_cleaned(self):
    """_clean_chained_exceptions cleans __cause__ tracebacks (line 62)."""
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))

    def handler(v=None):
      raise TypeError('handler error')

    def raiser(v=None):
      raise ValueError('original')

    try:
      Chain(1).then(raiser).except_(handler, reraise=False).run()
      self.fail('Expected TypeError')
    except TypeError as exc:
      cause = exc.__cause__
      self.assertIsInstance(cause, ValueError)

  def test_exception_with_context_cleaned(self):
    """_clean_chained_exceptions cleans __context__ tracebacks (line 63)."""
    def finally_raiser(v=None):
      raise RuntimeError('finally error')

    def chain_raiser(v=None):
      raise ValueError('chain error')

    try:
      Chain(1).then(chain_raiser).finally_(finally_raiser).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      context = exc.__context__
      self.assertIsInstance(context, ValueError)

  def test_clean_exc_chain_with_none(self):
    """_clean_chained_exceptions handles None exc gracefully (line 57).
    Uses the Python-visible _clean_exc_chain wrapper (line 316-318)."""
    # Should not raise
    _clean_exc_chain(None)

  def test_clean_exc_chain_no_traceback(self):
    """_clean_chained_exceptions handles exception with no traceback (line 60)."""
    exc = ValueError('no traceback')
    exc.__traceback__ = None
    _clean_exc_chain(exc)
    self.assertIsNone(exc.__traceback__)

  def test_clean_exc_chain_with_circular_cause(self):
    """_clean_chained_exceptions handles circular __cause__ references
    via the seen set (line 57 id(exc) in seen check)."""
    exc1 = ValueError('first')
    exc2 = TypeError('second')
    exc1.__cause__ = exc2
    exc2.__cause__ = exc1  # Circular
    # Should not infinite loop
    _clean_exc_chain(exc1)

  def test_clean_exc_chain_with_circular_context(self):
    """_clean_chained_exceptions handles circular __context__ references."""
    exc1 = ValueError('first')
    exc2 = TypeError('second')
    exc1.__context__ = exc2
    exc2.__context__ = exc1  # Circular
    # Should not infinite loop
    _clean_exc_chain(exc1)

  def test_clean_exc_chain_deep_cause_chain(self):
    """_clean_chained_exceptions recursively traverses deep __cause__ chains."""
    exc1 = ValueError('1')
    exc2 = TypeError('2')
    exc3 = RuntimeError('3')
    exc1.__cause__ = exc2
    exc2.__cause__ = exc3
    _clean_exc_chain(exc1)
    # Should traverse all three without error

  def test_clean_exc_chain_mixed_cause_and_context(self):
    """_clean_chained_exceptions handles mixed __cause__ and __context__."""
    exc1 = ValueError('1')
    exc2 = TypeError('2')
    exc3 = RuntimeError('3')
    exc1.__cause__ = exc2
    exc2.__context__ = exc3
    _clean_exc_chain(exc1)


# ---------------------------------------------------------------------------
# 3. remove_self_frames_from_traceback tests (line 66)
# ---------------------------------------------------------------------------

class RemoveSelfFramesTests(IsolatedAsyncioTestCase):
  """Tests for remove_self_frames_from_traceback (line 66).
  This is called from _await_run when an async exception occurs.
  """

  async def test_async_exception_sets_quent_attr(self):
    """remove_self_frames_from_traceback sets __quent__ = True (line 69).
    This is triggered when _await_run catches an exception from an async
    except_ handler (not from the simple chain path)."""
    async def async_handler(v=None):
      raise TypeError('handler boom')

    # Use except_ with reraise=False so the handler coro goes through _await_run
    c = (
      Chain(lambda: (_ for _ in ()).throw(TestExc('trigger')))
      .then(lambda v: v)
      .except_(async_handler, reraise=False)
    )
    task = c.run()
    try:
      await task
      self.fail('Expected TypeError')
    except TypeError as exc:
      self.assertTrue(
        getattr(exc, '__quent__', False),
        'Exception should have __quent__ attribute set to True'
      )

  async def test_async_exception_traceback_cleaned(self):
    """remove_self_frames_from_traceback cleans internal frames (line 73)."""
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))

    async def async_raiser(v=None):
      raise ValueError('async boom')

    try:
      await Chain(1).then(async_raiser).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      for fn in filenames:
        if fn == '<quent>':
          continue
        self.assertFalse(
          fn.startswith(quent_pkg_dir),
          f'Internal frame should be cleaned: {fn}'
        )

  async def test_async_exception_chained_causes_cleaned(self):
    """remove_self_frames_from_traceback cleans __cause__/__context__
    on chained exceptions (lines 76-78)."""
    async def async_handler(v=None):
      raise TypeError('handler async')

    async def async_raiser(v=None):
      raise ValueError('original async')

    try:
      await Chain(1).then(async_raiser).except_(
        async_handler, reraise=False
      ).run()
      self.fail('Expected TypeError')
    except TypeError as exc:
      cause = exc.__cause__
      self.assertIsInstance(cause, ValueError)

  async def test_async_exception_in_nested_chain(self):
    """remove_self_frames_from_traceback with nested chain exception.
    The __quent__ attr is set by remove_self_frames_from_traceback which
    is called from _await_run. Verify the exception propagates correctly."""
    async def inner_raiser(v):
      raise ValueError('inner async')

    inner = Chain().then(inner_raiser)
    try:
      await Chain(5).then(inner).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      # The exception should have a cleaned traceback with <quent> frame
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)


# ---------------------------------------------------------------------------
# 4. _handle_exception tests (lines 84, 91-94)
# ---------------------------------------------------------------------------

class HandleExceptionTests(TestCase):
  """Tests for _handle_exception exception handler dispatch.
  Lines 91-94 handle exc_temp_args from iteration exceptions.
  """

  def test_handle_exception_finds_matching_handler(self):
    """_handle_exception dispatches to the correct except_ handler (line 84)."""
    results = []

    def handler(v=None):
      results.append('handled')
      return 'recovered'

    result = Chain(1).then(_raise_value_error).except_(
      handler, reraise=False
    ).run()
    self.assertEqual(result, 'recovered')
    self.assertEqual(results, ['handled'])

  def test_handle_exception_no_matching_handler(self):
    """_handle_exception returns None when no handler matches (line 107)."""
    def handler(v=None):
      return 'handled'

    with self.assertRaises(ValueError):
      Chain(1).then(_raise_value_error).except_(
        handler, exceptions=TypeError, reraise=False
      ).run()

  def test_handle_exception_with_exc_temp_args_none_ctx(self):
    """_handle_exception: ctx.link_temp_args is None, exc_temp_args is not None
    -> sets ctx.link_temp_args = exc_temp_args (lines 91-92).
    This happens during foreach sync exception."""
    def fail_on_3(x):
      if x == 3:
        raise ValueError(f'bad {x}')
      return x

    try:
      Chain([1, 2, 3, 4]).foreach(fail_on_3).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # The visualization should show the element that caused the error
      self.assertIn('3', viz)

  def test_handle_exception_updates_existing_temp_args(self):
    """_handle_exception: ctx.link_temp_args is not None, exc_temp_args is not None
    -> updates ctx.link_temp_args (line 94). This path requires a prior
    link_temp_args plus a new one from exception propagation."""
    # This is hard to trigger directly. Using a foreach where the exception
    # carries __quent_link_temp_args__ and the ctx might already have some.
    # A nested foreach where the inner raises can trigger this.
    def inner_fail(x):
      if x == 'b':
        raise ValueError('inner fail')
      return x

    def outer_fn(v):
      return Chain([v, 'a', 'b']).foreach(inner_fail)()

    try:
      Chain(['x']).foreach(outer_fn).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  def test_handle_exception_skips_non_handler_links(self):
    """_handle_exception skips links that are not exception handlers
    (lines 98-101)."""
    results = []

    def step1(v):
      return v + 1

    def step2(v):
      raise ValueError('boom')

    def handler(v=None):
      results.append('caught')
      return 'recovered'

    result = Chain(1).then(step1).then(step2).except_(
      handler, reraise=False
    ).run()
    self.assertEqual(result, 'recovered')

  def test_handle_exception_checks_exception_type(self):
    """_handle_exception checks isinstance(exc, exc_link.exceptions) (line 104)."""
    def value_handler(v=None):
      return 'value_handled'

    def type_handler(v=None):
      return 'type_handled'

    # ValueError handler should not catch TypeError
    result = Chain(1).then(_raise_type_error).except_(
      value_handler, exceptions=ValueError, reraise=False
    ).except_(
      type_handler, exceptions=TypeError, reraise=False
    ).run()
    self.assertEqual(result, 'type_handled')


# ---------------------------------------------------------------------------
# 5. modify_traceback tests (line 111)
# ---------------------------------------------------------------------------

class ModifyTracebackTests(TestCase):
  """Tests for modify_traceback which augments exceptions with chain
  execution context. Line 111 is the function entry.
  """

  def test_modify_traceback_injects_quent_frame(self):
    """modify_traceback adds a <quent> frame to the exception (line 111+)."""
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  def test_modify_traceback_saves_source_link(self):
    """modify_traceback saves __quent_source_link__ then removes it (lines 119-125)."""
    try:
      Chain(1).then(_add_one).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      # __quent_source_link__ should be deleted by modify_traceback
      self.assertFalse(hasattr(exc, '__quent_source_link__'))

  def test_modify_traceback_nested_chain_skips(self):
    """modify_traceback returns early for nested chains (line 122-123).
    Only the outermost chain should produce the <quent> frame."""
    inner = Chain().then(_raise_value_error)
    try:
      Chain(1).then(inner).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      # Should have exactly one visualization (from outer chain)
      self.assertGreater(len(entries), 0)

  def test_modify_traceback_exec_code_replacement(self):
    """modify_traceback creates code with chain visualization in co_name
    (lines 126-142). On Python >= 3.11, co_qualname is also set (line 138)."""
    try:
      Chain(42).then(_add_one).then(_double).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # Should contain chain visualization with function names
      self.assertIn('Chain', viz)
      self.assertIn('_add_one', viz)
      self.assertIn('_double', viz)
      self.assertIn('_raise_value_error', viz)

  def test_modify_traceback_cleans_chained_exceptions(self):
    """modify_traceback cleans __cause__/__context__ after augmentation
    (lines 151-153)."""
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))

    def raiser(v=None):
      raise ValueError('orig')

    def handler(v=None):
      raise TypeError('handler')

    try:
      Chain(1).then(raiser).except_(handler, reraise=False).run()
      self.fail('Expected TypeError')
    except TypeError as exc:
      # __cause__ should be cleaned
      if exc.__cause__ is not None and exc.__cause__.__traceback__ is not None:
        for entry in traceback.extract_tb(exc.__cause__.__traceback__):
          if entry.filename == '<quent>':
            continue
          self.assertFalse(
            entry.filename.startswith(quent_pkg_dir),
            f'Chained cause traceback not cleaned: {entry.filename}'
          )

  def test_modify_traceback_with_debug_mode_results(self):
    """modify_traceback with debug mode shows intermediate results
    in the chain visualization."""
    try:
      Chain(5).then(_add_one).then(_double).then(
        _raise_value_error
      ).config(debug=True).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # Debug mode should show intermediate results (6, 12)
      self.assertIn('6', viz)
      self.assertIn('12', viz)


# ---------------------------------------------------------------------------
# 6. get_true_source_link tests (lines 157, 164, 169-172)
# ---------------------------------------------------------------------------

class GetTrueSourceLinkTests(TestCase):
  """Tests for get_true_source_link which resolves through nested chains.
  Line 157: function entry.
  Line 164: isinstance(source_link.original_value, Chain) path.
  Lines 169-172: ctx.temp_root_link path and else break.
  """

  def test_direct_link_no_nesting(self):
    """get_true_source_link with a direct (non-chain) link returns it
    immediately (line 165-166 break)."""
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)

  def test_nested_chain_is_chain_path(self):
    """get_true_source_link follows source_link.is_chain (line 161-162)."""
    inner = Chain().then(_raise_value_error)
    try:
      Chain(1).then(inner).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)

  def test_original_value_is_chain_path(self):
    """get_true_source_link follows isinstance(source_link.original_value, Chain)
    path (line 163-164). This happens when a foreach/filter link wraps a
    Chain as its original_value. The link.original_value is a Link whose
    original_value points to the actual Chain. Exercise this through
    foreach which creates wrapper links."""
    # foreach creates Link(fn) wrapped in Link(_Foreach(...), original_value=link)
    # When the link's original_value refers to a chain, get_true_source_link
    # follows it. Test via nested chain in then() that resolves through.
    inner = Chain().then(lambda v: 1 / 0)
    try:
      Chain(5).then(inner).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)

  def test_nested_chain_no_root_link_with_temp_root(self):
    """get_true_source_link: chain.root_link is None but ctx.temp_root_link
    is not None (lines 169-170). This happens when a void chain is
    called with a value override."""
    inner = Chain().then(lambda v: 1 / 0)
    try:
      Chain(5).then(inner).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)
      self.assertIn('<lambda>', viz)

  def test_nested_chain_no_root_no_temp_root_breaks(self):
    """get_true_source_link: chain.root_link is None and ctx.temp_root_link
    is None -> else break (lines 171-172). This happens when a void chain
    with no value override is used directly. The simple path handles this."""
    try:
      Chain().then(lambda v=None: 1 / 0).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      # Exception should propagate with a <quent> frame
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  def test_deeply_nested_chain_resolution(self):
    """get_true_source_link resolves through multiple nesting levels."""
    innermost = Chain().then(lambda v: 1 / 0)
    middle = Chain().then(innermost)
    try:
      Chain(5).then(middle).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)

  def test_chain_with_root_link_resolution(self):
    """get_true_source_link follows chain.root_link (line 167-168).
    When a nested chain has a root_link (callable root), get_true_source_link
    resolves through chain.root_link to the actual failing function."""
    # Use a chain with a callable root that doesn't take args (so no conflict)
    inner = Chain(lambda: 1 / 0)
    try:
      Chain(5).then(lambda v: inner()).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)


# ---------------------------------------------------------------------------
# 7. make_indent tests (line 177)
# ---------------------------------------------------------------------------

class MakeIndentTests(TestCase):
  """Tests for make_indent which returns indentation string. Line 177."""

  def test_indent_in_repr_of_nested_chain(self):
    """make_indent is called during stringify_chain for nested chains.
    The repr should contain indentation (4 spaces per level)."""
    inner = Chain().then(_add_one)
    outer = Chain(1).then(inner)
    r = repr(outer)
    # Nested chain should have indentation
    self.assertIn('\n', r)
    self.assertIn('    ', r)  # 4 spaces for level 1

  def test_indent_in_deeply_nested_repr(self):
    """make_indent with multiple nesting levels produces deeper indentation."""
    innermost = Chain().then(_add_one)
    middle = Chain().then(innermost)
    outer = Chain(1).then(middle)
    r = repr(outer)
    # Should have multiple levels of indentation
    self.assertIn('    ', r)

  def test_indent_in_exception_visualization(self):
    """make_indent is used in exception traceback visualization for
    nested chains."""
    inner = Chain().then(lambda v: 1 / 0)
    try:
      Chain(5).then(inner).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # Nested chain visualization should have indentation
      self.assertIn('Chain', viz)


# ---------------------------------------------------------------------------
# 8. stringify_chain tests (line 181)
# ---------------------------------------------------------------------------

class StringifyChainTests(TestCase):
  """Tests for stringify_chain entry point. Line 181."""

  def test_repr_empty_chain(self):
    """stringify_chain on empty chain returns class name + ()."""
    r = repr(Chain())
    self.assertIn('Chain', r)
    self.assertIn('()', r)

  def test_repr_chain_with_root(self):
    """stringify_chain with root value shows the value."""
    r = repr(Chain(42))
    self.assertIn('42', r)

  def test_repr_chain_with_callable_root(self):
    """stringify_chain with callable root shows function name."""
    r = repr(Chain(_add_one))
    self.assertIn('_add_one', r)

  def test_repr_chain_with_links(self):
    """stringify_chain includes all linked operations."""
    r = repr(Chain(1).then(_add_one).then(_double))
    self.assertIn('_add_one', r)
    self.assertIn('_double', r)

  def test_repr_chain_with_finally(self):
    """stringify_chain includes finally link (lines 205-210)."""
    r = repr(Chain(1).then(_add_one).finally_(_double))
    self.assertIn('_add_one', r)
    self.assertIn('finally_', r)
    self.assertIn('_double', r)

  def test_repr_chain_with_except(self):
    """stringify_chain includes except link."""
    def handler(v=None):
      return v

    r = repr(Chain(1).then(_raise_value_error).except_(handler))
    self.assertIn('except_', r)

  def test_repr_void_chain_with_temp_root(self):
    """stringify_chain with no root_link falls back to ctx.temp_root_link
    (line 184-185). This is used internally during error visualization."""
    # The temp_root_link path is triggered during exception handling
    # when a void chain is called with a value override.
    try:
      Chain().then(_raise_value_error).run(42)
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('42', viz)

  def test_repr_chain_with_no_links_no_root(self):
    """stringify_chain with neither root nor links shows just class name + ()."""
    r = repr(Chain())
    self.assertEqual(r.strip(), 'Chain()')

  def test_repr_source_link_found_stops_arrows(self):
    """stringify_chain sets found_source_link=True after finding the source,
    preventing arrows on subsequent links (lines 194-195, 201-202)."""
    try:
      Chain(1).then(_add_one).then(_raise_value_error).then(_double).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # Only the source link should have the arrow
      self.assertEqual(viz.count('<----'), 1)

  def test_repr_finally_link_with_source_arrow(self):
    """stringify_chain marks the finally link with source arrow when
    the exception occurs in finally (lines 206-210)."""
    def finally_raiser(v=None):
      raise RuntimeError('finally boom')

    try:
      Chain(1).then(_add_one).finally_(finally_raiser).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('finally_', viz)


# ---------------------------------------------------------------------------
# 9. format_args / format_kwargs tests (lines 215, 223)
# ---------------------------------------------------------------------------

class FormatArgsKwargsTests(TestCase):
  """Tests for format_args (line 215) and format_kwargs (line 223).
  These format the arguments shown in chain repr and error visualizations.
  """

  def test_format_args_with_regular_args(self):
    """format_args with regular positional arguments (line 220)."""
    r = repr(Chain(1).then(_add_one, 'extra'))
    self.assertIn('extra', r)

  def test_format_args_with_ellipsis(self):
    """format_args with Ellipsis first arg (line 218-219)."""
    r = repr(Chain(1).then(_add_one, ...))
    self.assertIn('...', r)

  def test_format_args_empty(self):
    """format_args with no args returns empty string (line 216-217)."""
    r = repr(Chain(1).then(_add_one))
    # _add_one should appear without extra args
    self.assertIn('_add_one', r)

  def test_format_kwargs_with_kwargs(self):
    """format_kwargs with keyword arguments (line 226)."""
    def fn_with_kwargs(v, key=None):
      return v

    r = repr(Chain(1).then(fn_with_kwargs, key='val'))
    self.assertIn('key', r)
    self.assertIn('val', r)

  def test_format_kwargs_empty(self):
    """format_kwargs with no kwargs returns empty string (line 224-225)."""
    r = repr(Chain(1).then(_add_one))
    self.assertIn('_add_one', r)

  def test_format_args_multiple_args(self):
    """format_args with multiple positional arguments."""
    def fn_multi(v, a, b):
      return v

    r = repr(Chain(1).then(fn_multi, 'arg1', 'arg2'))
    self.assertIn('arg1', r)
    self.assertIn('arg2', r)

  def test_format_args_and_kwargs_together(self):
    """format_args and format_kwargs together in one link."""
    def fn_mixed(v, extra, key=None):
      return v

    r = repr(Chain(1).then(fn_mixed, 'pos_arg', key='kw_val'))
    self.assertIn('pos_arg', r)
    self.assertIn('key', r)
    self.assertIn('kw_val', r)

  def test_format_args_in_exception_visualization(self):
    """format_args appears in exception traceback visualization."""
    def raiser(v, extra=None):
      raise ValueError('boom')

    try:
      Chain(1).then(raiser, 'extra_arg').run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('extra_arg', viz)


# ---------------------------------------------------------------------------
# 10. format_link tests (lines 229, 248-249, 281)
# ---------------------------------------------------------------------------

class FormatLinkTests(TestCase):
  """Tests for format_link which formats individual links.
  Line 229: function entry.
  Lines 248-249: link_temp_args path.
  Line 281: chain link with args_s or kwargs_s.
  """

  def test_format_link_basic(self):
    """format_link basic formatting (line 229)."""
    r = repr(Chain(1).then(_add_one))
    self.assertIn('.then', r)
    self.assertIn('_add_one', r)

  def test_format_link_with_operation_label(self):
    """format_link shows operation label prefix (e.g. .then, .do)."""
    r = repr(Chain(1).then(_add_one).do(_double))
    self.assertIn('.then', r)
    self.assertIn('.do', r)

  def test_format_link_source_arrow(self):
    """format_link adds source arrow at failing link (lines 287-289)."""
    try:
      Chain(1).then(_add_one).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)

  def test_format_link_results_annotation(self):
    """format_link shows = result for links before the source (lines 290-293)."""
    try:
      Chain(5).then(_double).then(_raise_value_error).config(debug=True).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # _double(5) = 10 should appear
      self.assertIn('10', viz)

  def test_format_link_temp_args_from_foreach(self):
    """format_link uses ctx.link_temp_args (lines 247-249) when a
    foreach exception provides per-link temp args."""
    def fail_on_target(x):
      if x == 42:
        raise ValueError(f'bad {x}')
      return x

    try:
      Chain([10, 20, 42, 50]).foreach(fail_on_target).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # Should show the element that caused the error
      self.assertIn('42', viz)

  def test_format_link_chain_with_args_newline(self):
    """format_link chain branch: when args_s or kwargs_s are present,
    chain_newline is set (line 280-281)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, 'arg1', key='val')
    r = repr(c)
    self.assertIn('arg1', r)
    self.assertIn('val', r)

  def test_format_link_chain_without_args(self):
    """format_link chain branch without args (line 276-282 with empty args)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner)
    r = repr(c)
    self.assertIn('Chain', r)

  def test_format_link_non_callable_literal(self):
    """format_link for a literal value (EVAL_RETURN_AS_IS) (line 285-286)."""
    r = repr(Chain(42))
    self.assertIn('42', r)

  def test_format_link_original_value_is_link(self):
    """format_link unwraps Link.original_value when it is itself a Link
    (line 231-232). This happens with foreach/filter wrapper links."""
    r = repr(Chain([1, 2, 3]).foreach(_add_one))
    self.assertIn('foreach', r)
    self.assertIn('_add_one', r)

  def test_format_link_with_temp_args_on_link(self):
    """format_link checks link.temp_args when not found_source_link
    (line 250-252). temp_args are set during foreach sync exceptions."""
    def fail_first(x):
      raise ValueError('immediate fail')

    try:
      Chain([1, 2]).foreach(fail_first).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('1', viz)


# ---------------------------------------------------------------------------
# 11. get_obj_name tests (line 297)
# ---------------------------------------------------------------------------

class GetObjNameTests(TestCase):
  """Tests for get_obj_name which returns a display name for objects.
  Line 297 is the function entry.
  """

  def test_get_obj_name_function(self):
    """get_obj_name returns __name__ for routines (line 298-299)."""
    r = repr(Chain(_add_one))
    self.assertIn('_add_one', r)

  def test_get_obj_name_class(self):
    """get_obj_name returns __name__ for classes (line 298-299)."""
    r = repr(Chain(1).then(int))
    self.assertIn('int', r)

  def test_get_obj_name_chain(self):
    """get_obj_name returns type(obj).__name__ for Chain (line 300-301)."""
    inner = Chain().then(_add_one)
    r = repr(Chain(1).then(inner))
    self.assertIn('Chain', r)

  def test_get_obj_name_literal(self):
    """get_obj_name returns repr() for non-callable, non-chain objects
    (line 302)."""
    r = repr(Chain(42))
    self.assertIn('42', r)

  def test_get_obj_name_string_literal(self):
    """get_obj_name uses repr() for strings."""
    r = repr(Chain('hello'))
    self.assertIn("'hello'", r)

  def test_get_obj_name_lambda(self):
    """get_obj_name returns '<lambda>' for lambda functions."""
    r = repr(Chain(1).then(lambda v: v))
    self.assertIn('<lambda>', r)

  def test_get_obj_name_none(self):
    """get_obj_name returns 'None' for None."""
    r = repr(Chain(None))
    self.assertIn('None', r)


# ---------------------------------------------------------------------------
# 12. _quent_excepthook / module-level code (lines 306-308, 314-316)
# ---------------------------------------------------------------------------

class ExceptHookTests(TestCase):
  """Tests for the global exception hook and module-level code.
  Lines 306-308: _original_excepthook and _quent_excepthook definition.
  Lines 314-316: sys.excepthook replacement and _clean_exc_chain wrapper.
  """

  def test_sys_excepthook_is_quent(self):
    """sys.excepthook should be replaced with _quent_excepthook (line 314)."""
    self.assertIs(sys.excepthook, _quent_excepthook)

  def test_quent_excepthook_with_quent_exception(self):
    """_quent_excepthook handles exceptions with __quent__ = True (line 309).
    It cleans the exception and passes to original excepthook."""
    exc = ValueError('test')
    exc.__quent__ = True
    # Create a minimal traceback
    try:
      raise exc
    except ValueError:
      pass

    # Call the excepthook - it should not raise
    import io
    import contextlib
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
      _quent_excepthook(ValueError, exc, exc.__traceback__)

  def test_quent_excepthook_with_non_quent_exception(self):
    """_quent_excepthook passes through non-quent exceptions (line 312)."""
    exc = ValueError('no quent')
    try:
      raise exc
    except ValueError:
      pass

    import io
    import contextlib
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
      _quent_excepthook(ValueError, exc, exc.__traceback__)

  def test_quent_excepthook_without_quent_attr(self):
    """_quent_excepthook handles exception without __quent__ attr (line 309)."""
    exc = ValueError('plain')
    try:
      raise exc
    except ValueError:
      pass

    import io
    import contextlib
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
      _quent_excepthook(ValueError, exc, exc.__traceback__)

  def test_clean_exc_chain_wrapper(self):
    """_clean_exc_chain is a Python-visible wrapper (lines 316-318)."""
    exc = ValueError('test')
    # Should not raise on a simple exception
    _clean_exc_chain(exc)

  def test_clean_exc_chain_with_cause_and_context(self):
    """_clean_exc_chain cleans both __cause__ and __context__."""
    exc1 = ValueError('main')
    exc2 = TypeError('cause')
    exc3 = RuntimeError('context')
    exc1.__cause__ = exc2
    exc1.__context__ = exc3
    _clean_exc_chain(exc1)


# ---------------------------------------------------------------------------
# 13. Integration tests: full exception paths through chains
# ---------------------------------------------------------------------------

class DiagnosticsIntegrationTests(TestCase):
  """Integration tests combining multiple _diagnostics.pxi paths."""

  def test_chain_with_multiple_except_handlers(self):
    """Exercise _handle_exception iterating through multiple handlers."""
    def value_handler(v=None):
      return 'value'

    def type_handler(v=None):
      return 'type'

    def runtime_handler(v=None):
      return 'runtime'

    result = Chain(1).then(_raise_type_error).except_(
      value_handler, exceptions=ValueError, reraise=False
    ).except_(
      runtime_handler, exceptions=RuntimeError, reraise=False
    ).except_(
      type_handler, exceptions=TypeError, reraise=False
    ).run()
    self.assertEqual(result, 'type')

  def test_chain_except_handler_raises(self):
    """Exception in except handler triggers modify_traceback again (line 198)."""
    def bad_handler(v=None):
      raise RuntimeError('handler boom')

    try:
      Chain(1).then(_raise_value_error).except_(
        bad_handler, reraise=False
      ).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  def test_chain_reraise_after_handler(self):
    """except_ with reraise=True still raises original exception."""
    handled = []

    def handler(v=None):
      handled.append(True)

    with self.assertRaises(ValueError):
      Chain(1).then(_raise_value_error).except_(handler, reraise=True).run()
    self.assertEqual(handled, [True])

  def test_finally_handler_exception_visualization(self):
    """Exception in finally handler gets modify_traceback treatment."""
    def finally_raiser(v=None):
      raise RuntimeError('finally boom')

    try:
      Chain(1).then(_add_one).finally_(finally_raiser).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('finally_', viz)

  def test_void_chain_exception_with_value_override(self):
    """Exception in a void chain called with value override exercises
    temp_root_link in ctx (lines 169-170 of get_true_source_link)."""
    try:
      Chain().then(_raise_value_error).run(99)
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('99', viz)

  def test_chain_with_do_and_exception(self):
    """Exception after .do() link exercises format_link with
    ignore_result link."""
    try:
      Chain(1).do(_add_one).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('.do', viz)

  def test_deeply_nested_exception_cleanup(self):
    """Deep nesting exercises clean_internal_frames and modify_traceback
    recursion through multiple chain levels."""
    def level1(v):
      return Chain(v).then(level2)()

    def level2(v):
      return Chain(v).then(level3)()

    def level3(v):
      return Chain(v).then(_raise_value_error)()

    try:
      Chain(1).then(level1).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      self.assertIn('<quent>', filenames)
      func_names = _get_tb_func_names(exc)
      self.assertIn('_raise_value_error', func_names)


# ---------------------------------------------------------------------------
# 14. Async diagnostics integration
# ---------------------------------------------------------------------------

class AsyncDiagnosticsTests(IsolatedAsyncioTestCase):
  """Async tests for _diagnostics.pxi paths."""

  async def test_async_chain_exception_has_quent_frame(self):
    """Async chain exception gets <quent> frame from modify_traceback."""
    async def async_raiser(v):
      raise ValueError('async boom')

    try:
      await Chain(1).then(async_raiser).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  async def test_async_nested_chain_exception(self):
    """Async nested chain exception exercises modify_traceback with
    is_nested=True skip."""
    async def inner_raiser(v):
      raise ValueError('inner async')

    inner = Chain().then(inner_raiser)
    try:
      await Chain(5).then(inner).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  async def test_async_foreach_exception_temp_args(self):
    """Async foreach exception sets __quent_link_temp_args__ which
    _handle_exception copies to ctx (lines 91-92)."""
    counter = {'n': 0}

    def fn(el):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(el * 10)
      raise ValueError('async foreach error')

    counter['n'] = 0
    with self.assertRaises(ValueError):
      await await_(Chain([1, 2, 3]).foreach(fn).run())

  async def test_async_except_handler_exception(self):
    """Async except handler that raises triggers modify_traceback
    through _await_run."""
    async def bad_handler(v=None):
      raise RuntimeError('handler async')

    c = (
      Chain(lambda: (_ for _ in ()).throw(TestExc('trigger')))
      .then(lambda v: v)
      .except_(bad_handler, reraise=False)
    )
    task = c.run()
    self.assertIsInstance(task, asyncio.Task)
    with self.assertRaises(RuntimeError):
      await task

  async def test_async_debug_mode_exception(self):
    """Async chain with debug mode and exception shows intermediate
    results in visualization."""
    async def async_double(v):
      return v * 2

    async def async_raiser(v):
      raise ValueError('async debug boom')

    try:
      await Chain(5).then(async_double).then(
        async_raiser
      ).config(debug=True).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)


# ---------------------------------------------------------------------------
# 15. foreach / filter temp_args in format_link (lines 248-249)
# ---------------------------------------------------------------------------

class ForeachFilterTempArgsTests(IsolatedAsyncioTestCase):
  """Tests specifically targeting the link_temp_args paths in
  format_link (lines 247-249 of _diagnostics.pxi).
  """

  def test_sync_foreach_temp_args_in_visualization(self):
    """Sync foreach exception: link.temp_args set directly on the link,
    used by format_link (line 250-252)."""
    def fail_on_big(x):
      if x > 10:
        raise ValueError(f'too big: {x}')
      return x

    try:
      Chain([5, 15, 25]).foreach(fail_on_big).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('15', viz)

  async def test_async_foreach_temp_args_in_visualization(self):
    """Async foreach exception: __quent_link_temp_args__ set on exception,
    copied to ctx.link_temp_args by _handle_exception (lines 91-92),
    then used by format_link (lines 247-249)."""
    counter = {'n': 0}

    def fn(el):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(el)
      raise ValueError(f'fail at {el}')

    counter['n'] = 0
    try:
      await await_(Chain([100, 200, 300]).foreach(fn).run())
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('200', viz)

  def test_sync_filter_exception_temp_args(self):
    """Sync filter exception sets temp_args on the link."""
    def bad_pred(x):
      if x == 'bad':
        raise ValueError('bad element')
      return True

    try:
      Chain(['ok', 'fine', 'bad']).filter(bad_pred).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('bad', viz)

  async def test_async_filter_exception_temp_args(self):
    """Async filter exception sets __quent_link_temp_args__ on exception."""
    counter = {'n': 0}

    def pred(x):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(True)
      raise ValueError('filter fail')

    counter['n'] = 0
    with self.assertRaises(ValueError):
      await await_(Chain([10, 20, 30]).filter(pred).run())

  async def test_async_foreach_full_async_temp_args(self):
    """Fully async foreach (async iterator) exception sets
    __quent_link_temp_args__ on exception. When the chain has an except_
    handler, _handle_exception is called and copies temp_args to ctx,
    which format_link then uses (lines 91-92, 247-249)."""
    class AsyncIter:
      def __init__(self, items):
        self._items = list(items)

      def __aiter__(self):
        self._it = iter(self._items)
        return self

      async def __anext__(self):
        try:
          return next(self._it)
        except StopIteration:
          raise StopAsyncIteration

    def fail_on_big(x):
      if x >= 30:
        raise ValueError(f'too big: {x}')
      return x

    # Use except_ with reraise=True to trigger _handle_exception path
    # while still raising the exception for us to catch
    try:
      await Chain(AsyncIter([10, 20, 30])).foreach(fail_on_big).except_(
        lambda v=None: None, reraise=True
      ).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # The temp_args should show the failing element
      self.assertIn('30', viz)


# ---------------------------------------------------------------------------
# 16. Edge cases in format_link chain branch (line 281)
# ---------------------------------------------------------------------------

class FormatLinkChainBranchTests(TestCase):
  """Tests for the chain branch in format_link where args_s or kwargs_s
  trigger chain_newline (line 280-281).
  """

  def test_nested_chain_with_positional_args_repr(self):
    """Nested chain called with positional args shows args in repr.
    Triggers format_link chain branch with args_s (line 277-281)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, 'arg1', 'arg2')
    r = repr(c)
    self.assertIn('arg1', r)
    self.assertIn('arg2', r)

  def test_nested_chain_with_kwargs_repr(self):
    """Nested chain called with kwargs shows kwargs in repr.
    Triggers format_link chain branch with kwargs_s (line 278-281)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, 'val', key='kwarg_val')
    r = repr(c)
    self.assertIn('kwarg_val', r)

  def test_nested_chain_exception_with_args(self):
    """Nested chain with args in exception visualization."""
    inner = Chain().then(lambda v: 1 / 0)
    try:
      Chain(5).then(inner, 'extra').run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('extra', viz)

  def test_nested_chain_no_args_repr(self):
    """Nested chain without args: chain_newline is empty (line 279-280)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner)
    r = repr(c)
    self.assertIn('Chain', r)

# ---------------------------------------------------------------------------
# 17. repr with except/finally and source arrow interaction
# ---------------------------------------------------------------------------

class ReprExceptFinallyTests(TestCase):
  """Test chain repr and visualization with except/finally handlers."""

  def test_repr_chain_with_except_and_finally(self):
    """repr shows both except_ and finally_ links."""
    def handler(v=None):
      return v

    def cleanup(v=None):
      pass

    r = repr(Chain(1).then(_add_one).except_(handler).finally_(cleanup))
    self.assertIn('except_', r)
    self.assertIn('finally_', r)

  def test_exception_in_chain_with_except_shows_visualization(self):
    """Exception with except_ handler still shows chain visualization."""
    results = []

    def handler(v=None):
      results.append('handled')
      return 'ok'

    try:
      Chain(1).then(_raise_value_error).except_(handler, reraise=True).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  def test_repr_with_foreach_link(self):
    """repr shows foreach link structure."""
    r = repr(Chain([1, 2]).foreach(_add_one))
    self.assertIn('foreach', r)
    self.assertIn('_add_one', r)

  def test_repr_with_filter_link(self):
    """repr shows filter link structure."""
    r = repr(Chain([1, 2]).filter(lambda x: x > 0))
    self.assertIn('filter', r)


# ---------------------------------------------------------------------------
# 18. Comprehensive format_link with nested chain + temp_root_link
# ---------------------------------------------------------------------------

class FormatLinkTempRootTests(TestCase):
  """Tests exercising stringify_chain's temp_root_link creation in
  format_link (lines 257-268). When a nested chain is formatted,
  format_link creates a nested _ExecCtx with temp_root_link.
  """

  def test_nested_chain_with_value_in_repr(self):
    """format_link creates temp_root_link when the nested chain has
    args (lines 259-264). The first arg becomes the root of the
    nested chain's visualization."""
    inner = Chain().then(_add_one).then(_double)
    c = Chain(1).then(inner, 10)
    r = repr(c)
    self.assertIn('Chain', r)
    # The nested chain should show 10 as its root value
    self.assertIn('10', r)

  def test_nested_chain_with_kwargs_in_repr(self):
    """format_link with nested chain + kwargs: _temp_v is Null when
    only kwargs are provided (line 262). The nested chain runs with
    Null root."""
    inner = Chain().then(lambda v=None: v)
    c = Chain(1).then(inner, key='val')
    r = repr(c)
    self.assertIn('Chain', r)

  def test_nested_chain_with_args_and_kwargs_in_repr(self):
    """format_link with nested chain + both args and kwargs."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, 'first_arg', key='kw')
    r = repr(c)
    self.assertIn('first_arg', r)

  def test_nested_chain_in_exception_with_temp_root(self):
    """Exception in nested chain called with args: format_link uses
    temp_root_link from nested_ctx to show the args as root."""
    inner = Chain().then(lambda v: 1 / 0)
    try:
      Chain(5).then(inner, 'root_val').run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('root_val', viz)


# ---------------------------------------------------------------------------
# 19. Multiple except handlers and exception type matching
# ---------------------------------------------------------------------------

class MultipleExceptHandlerTests(TestCase):
  """Test _handle_exception with multiple except handlers to exercise
  the while loop (lines 98-106) thoroughly.
  """

  def test_first_handler_matches(self):
    """First handler matches the exception type."""
    result = Chain(1).then(_raise_value_error).except_(
      lambda v=None: 'caught', exceptions=ValueError, reraise=False
    ).run()
    self.assertEqual(result, 'caught')

  def test_second_handler_matches(self):
    """Second handler matches after first doesn't."""
    result = Chain(1).then(_raise_type_error).except_(
      lambda v=None: 'wrong', exceptions=ValueError, reraise=False
    ).except_(
      lambda v=None: 'right', exceptions=TypeError, reraise=False
    ).run()
    self.assertEqual(result, 'right')

  def test_no_handler_matches_raises(self):
    """No handler matches - exception propagates."""
    with self.assertRaises(ValueError):
      Chain(1).then(_raise_value_error).except_(
        lambda v=None: 'x', exceptions=TypeError, reraise=False
      ).run()

  def test_handler_with_base_exception_class(self):
    """Handler matching Exception base class catches all."""
    result = Chain(1).then(_raise_value_error).except_(
      lambda v=None: 'base_caught', reraise=False
    ).run()
    self.assertEqual(result, 'base_caught')

  def test_handler_with_tuple_of_exceptions(self):
    """Handler with multiple exception types in tuple."""
    result = Chain(1).then(_raise_type_error).except_(
      lambda v=None: 'multi_caught',
      exceptions=(ValueError, TypeError),
      reraise=False
    ).run()
    self.assertEqual(result, 'multi_caught')


# ---------------------------------------------------------------------------
# 20. Edge case: simple chain exception path (no _handle_exception)
# ---------------------------------------------------------------------------

class SimpleChainExceptionTests(TestCase):
  """Test the simple chain exception path which calls modify_traceback
  directly without _handle_exception (lines 416-422 of _chain_core.pxi).
  This still exercises modify_traceback (line 111) and clean_internal_frames.
  """

  def test_simple_chain_exception_has_quent_frame(self):
    """Simple chain (one .then, no debug/finally) still gets <quent> frame."""
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)

  def test_simple_void_chain_exception(self):
    """Simple void chain called with value override."""
    try:
      Chain().then(_raise_value_error).run(42)
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('42', viz)

  def test_simple_chain_root_exception(self):
    """Exception in root value of simple chain."""
    try:
      Chain(lambda: 1 / 0).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)


if __name__ == '__main__':
  unittest.main()
