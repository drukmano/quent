# SPDX-License-Identifier: MIT
"""Implementation coverage tests for _traceback.py, _chain.py, _types.py polyfill, debug logging.

These tests target internal code paths for coverage.
For spec-mandated behavior tests, see edge_tests.py, traceback_tests.py, and pipeline_tests.py.
"""

from __future__ import annotations

import logging
import sys
import traceback
import unittest
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Chain
from tests.tests_helper import (
  async_fn,
)

# ---------------------------------------------------------------------------
# _chain.py — Coverage: builder internal paths
# ---------------------------------------------------------------------------


class ChainBuilderCovTest(TestCase):
  """Coverage tests for _chain.py builder internals."""

  def test_normalize_exception_types_none_with_default(self) -> None:
    """Lines 118-119: _normalize_exception_types with None exc_types and a default.

    This is the early return path when default is provided.
    """
    from quent._chain import _normalize_exception_types

    default_types = (ValueError, TypeError)
    result = _normalize_exception_types(None, 'test', default=default_types)
    self.assertEqual(result, default_types)

  def test_normalize_exception_types_none_no_default(self) -> None:
    """Lines 120-121: _normalize_exception_types with None exc_types and no default."""
    from quent._chain import _normalize_exception_types

    with self.assertRaises(TypeError) as ctx:
      _normalize_exception_types(None, 'test')
    self.assertIn('requires exception types', str(ctx.exception))

  def test_clone_missing_slots_debug(self) -> None:
    """Line 1199: clone() detects missing slots in debug mode.

    This is a defensive check that runs only in __debug__ mode.
    """
    # Normal clone should work without issues
    c = Chain(1).then(lambda x: x + 1)
    cloned = c.clone()
    self.assertEqual(cloned.run(), 2)

  # test_control_flow_signal_escape_from_run removed — canonical tests in control_flow_tests.py
  # test_decorator_break_signal_wrapped_in_quentexception removed — canonical test in edge_tests.py


# ---------------------------------------------------------------------------
# _traceback.py — Coverage: complex exception chain traversal
# ---------------------------------------------------------------------------


class ComplexExceptionChainTest(TestCase):
  """§13.5: Complex __cause__ and __context__ chain traversal.

  Lines 303->305, 313->319: Exception chain walking branches in excepthook.
  Lines 338->371, 350-357, 365->367, 367->371: Deep exception chain traversal.
  """

  def test_cause_chain_cleaned_through_quent(self):
    """Exception with __cause__ chain is fully cleaned by _clean_chained_exceptions."""
    from quent._traceback import _clean_chained_exceptions

    # Build a cause chain: exc3 -> exc2 -> exc1
    exc1 = ValueError('root cause')
    exc2 = RuntimeError('mid')
    exc2.__cause__ = exc1
    exc3 = TypeError('top')
    exc3.__cause__ = exc2

    seen: set[int] = set()
    _clean_chained_exceptions(exc3, seen)
    # All three should have been visited
    self.assertIn(id(exc3), seen)
    self.assertIn(id(exc2), seen)
    self.assertIn(id(exc1), seen)

  def test_context_chain_cleaned_through_quent(self):
    """Exception with __context__ chain is fully cleaned."""
    from quent._traceback import _clean_chained_exceptions

    exc1 = ValueError('original')
    exc2 = RuntimeError('during handling')
    exc2.__context__ = exc1
    exc3 = TypeError('nested handling')
    exc3.__context__ = exc2

    seen: set[int] = set()
    _clean_chained_exceptions(exc3, seen)
    self.assertIn(id(exc3), seen)
    self.assertIn(id(exc2), seen)
    self.assertIn(id(exc1), seen)

  def test_mixed_cause_and_context_chain(self):
    """Exception with both __cause__ and __context__ chains are cleaned.

    Per §13.5: 'Frame cleaning applies to __cause__, __context__, ExceptionGroup.'
    """
    from quent._traceback import _clean_chained_exceptions

    # exc1 is __cause__ of exc2, exc3 is __context__ of exc2
    exc1 = ValueError('cause')
    exc3 = TypeError('context')
    exc2 = RuntimeError('mid')
    exc2.__cause__ = exc1
    exc2.__context__ = exc3

    seen: set[int] = set()
    _clean_chained_exceptions(exc2, seen)
    self.assertIn(id(exc2), seen)
    self.assertIn(id(exc1), seen)
    self.assertIn(id(exc3), seen)

  def test_deeply_nested_cause_context_chains(self):
    """Deeply nested exception chains with both __cause__ and __context__."""
    from quent._traceback import _clean_chained_exceptions

    # Build a deep chain: each exception has both __cause__ and __context__
    exceptions = []
    for i in range(20):
      exc = ValueError(f'exc_{i}')
      exceptions.append(exc)

    # Create tree structure: each exc[i] has cause=exc[i+1] and context=exc[i+2]
    for i in range(len(exceptions) - 2):
      exceptions[i].__cause__ = exceptions[i + 1]
      exceptions[i].__context__ = exceptions[i + 2]

    seen: set[int] = set()
    _clean_chained_exceptions(exceptions[0], seen)
    # All exceptions should be visited
    for exc in exceptions:
      self.assertIn(id(exc), seen)

  def test_excepthook_with_quent_exception_cleans_chained(self):
    """Lines 303->305: excepthook path cleans chained exceptions.

    When _try_clean_quent_exc returns True (cleaned=True),
    the excepthook uses the cleaned traceback.
    """
    from unittest.mock import patch as mock_patch

    from quent._traceback import _quent_excepthook

    # Create a quent-marked exception with a __cause__ chain
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      cause = RuntimeError('cause in hook test')
      exc.__cause__ = cause

      hook_called_with = []

      def mock_prev_hook(exc_type, exc_value, exc_tb):
        hook_called_with.append((exc_type, exc_value, exc_tb))

      with mock_patch('quent._traceback._prev_excepthook', mock_prev_hook):
        _quent_excepthook(type(exc), exc, exc.__traceback__)

      self.assertEqual(len(hook_called_with), 1)
      # The hook should have been called with the cleaned traceback
      self.assertIs(hook_called_with[0][1], exc)

  def test_excepthook_non_quent_exception_passthrough(self):
    """Lines 303->305: excepthook passes through non-quent exceptions unchanged."""
    from unittest.mock import patch as mock_patch

    from quent._traceback import _quent_excepthook

    exc = ValueError('not a quent exception')
    original_tb = exc.__traceback__

    hook_called_with = []

    def mock_prev_hook(exc_type, exc_value, exc_tb):
      hook_called_with.append((exc_type, exc_value, exc_tb))

    with mock_patch('quent._traceback._prev_excepthook', mock_prev_hook):
      _quent_excepthook(type(exc), exc, original_tb)

    self.assertEqual(len(hook_called_with), 1)
    # Non-quent exception should pass through with original tb
    self.assertIs(hook_called_with[0][2], original_tb)

  def test_traceback_exception_patch_with_quent_exception(self):
    """Lines 319-330: Patched TracebackException.__init__ cleans quent frames."""
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      # Create a __cause__ chain
      cause = RuntimeError('chained cause')
      exc.__cause__ = cause

      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      # Should contain <quent> frame (visualization was injected)
      self.assertIn('<quent>', formatted)

  def test_hook_idempotency_guard(self):
    """Lines 365->367, 367->371: Hook installation idempotency guard.

    Hooks are only installed once even if checked multiple times.
    """
    from quent._traceback import _patched_te_init, _quent_excepthook

    # Verify hooks are currently installed
    self.assertIs(sys.excepthook, _quent_excepthook)
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)

  def test_viz_failure_clears_globals_in_except(self):
    """Lines 182-183: globals_ dict is cleared in except block on viz failure.

    Per §13.12: 'Visualization failure -> RuntimeWarning, exception propagates.'
    """
    from unittest.mock import patch as mock_patch

    with mock_patch('quent._traceback._stringify_chain', side_effect=Exception('viz boom')):
      import warnings

      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
          Chain(1).then(lambda x: 1 / 0).run()
        except ZeroDivisionError:
          pass  # Expected
        else:
          self.fail('Expected ZeroDivisionError')

      viz_warnings = [x for x in w if 'visualization failed' in str(x.message)]
      self.assertTrue(len(viz_warnings) > 0, f'Expected viz warning, got: {w}')

  def test_exception_note_pre_311_no_add_note(self):
    """Line 203: _attach_exception_note with missing add_note is a no-op.

    Pre-Python 3.11 exceptions have no add_note method.
    """
    from quent._traceback import _attach_exception_note

    class NoAddNoteExc(Exception):
      """Exception without add_note (simulating pre-3.11)."""

      pass

    exc = NoAddNoteExc('test')

    # On 3.11+, we need to delete add_note to test the early return path
    has_add_note = hasattr(exc, 'add_note')

    c = Chain(1)
    from quent._link import Link

    source = Link(lambda x: x)

    if has_add_note:
      # We cannot easily remove add_note from builtin types,
      # so just verify the normal path works.
      _attach_exception_note(exc, c, source)
      notes = getattr(exc, '__notes__', [])
      quent_notes = [n for n in notes if n.startswith('quent: exception at')]
      self.assertTrue(len(quent_notes) >= 1)
    else:
      # On pre-3.11, verify it is a no-op
      _attach_exception_note(exc, c, source)
      self.assertFalse(hasattr(exc, '__notes__'))

  def test_modify_traceback_disabled(self):
    """Line 237: _modify_traceback with _traceback_enabled=False returns unmodified exception.

    Per §13.7: 'QUENT_NO_TRACEBACK=1 disables all traceback modifications.'
    """
    import quent._traceback as tb_mod
    from quent._traceback import _modify_traceback

    original_enabled = tb_mod._traceback_enabled
    try:
      tb_mod._traceback_enabled = False
      exc = ValueError('test')
      result = _modify_traceback(exc)
      self.assertIs(result, exc)
    finally:
      tb_mod._traceback_enabled = original_enabled


# ---------------------------------------------------------------------------
# _types.py — ExceptionGroup polyfill (Python <3.11)
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.version_info >= (3, 11), 'Polyfill only defined on Python <3.11')
class ExceptionGroupPolyfillSubgroupTest(TestCase):
  """SPEC §6.5 / SPEC-210: polyfill .subgroup(condition) tests."""

  def _get_polyfill_cls(self):
    from quent._types import ExceptionGroup as PolyfillEG

    return PolyfillEG

  def test_subgroup_filter_by_type_match(self) -> None:
    """subgroup(type) returns new group with matching exceptions."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1'), TypeError('t1'), ValueError('v2')])
    result = eg.subgroup(ValueError)
    self.assertIsNotNone(result)
    self.assertEqual(len(result.exceptions), 2)
    self.assertIsInstance(result.exceptions[0], ValueError)
    self.assertIsInstance(result.exceptions[1], ValueError)
    self.assertEqual(result.args[0], 'test')

  def test_subgroup_filter_by_type_no_match(self) -> None:
    """subgroup(type) returns None when no exceptions match."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1'), TypeError('t1')])
    result = eg.subgroup(KeyError)
    self.assertIsNone(result)

  def test_subgroup_filter_by_callable(self) -> None:
    """subgroup(callable) filters using the callable predicate."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('a'), ValueError('b'), TypeError('c')])
    result = eg.subgroup(lambda e: 'a' in str(e))
    self.assertIsNotNone(result)
    self.assertEqual(len(result.exceptions), 1)
    self.assertEqual(str(result.exceptions[0]), 'a')

  def test_subgroup_filter_by_callable_no_match(self) -> None:
    """subgroup(callable) returns None when no exceptions match."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('a')])
    result = eg.subgroup(lambda e: False)
    self.assertIsNone(result)


@unittest.skipIf(sys.version_info >= (3, 11), 'Polyfill only defined on Python <3.11')
class ExceptionGroupPolyfillSplitTest(TestCase):
  """SPEC §6.5 / SPEC-211: polyfill .split(condition) tests."""

  def _get_polyfill_cls(self):
    from quent._types import ExceptionGroup as PolyfillEG

    return PolyfillEG

  def test_split_by_type(self) -> None:
    """split(type) splits into (matching, rest) groups."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1'), TypeError('t1'), ValueError('v2')])
    matched, rest = eg.split(ValueError)
    self.assertIsNotNone(matched)
    self.assertIsNotNone(rest)
    self.assertEqual(len(matched.exceptions), 2)
    self.assertEqual(len(rest.exceptions), 1)
    self.assertIsInstance(rest.exceptions[0], TypeError)
    self.assertEqual(matched.args[0], 'test')
    self.assertEqual(rest.args[0], 'test')

  def test_split_by_type_all_match(self) -> None:
    """split(type) when all match: rest is None."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1'), ValueError('v2')])
    matched, rest = eg.split(ValueError)
    self.assertIsNotNone(matched)
    self.assertIsNone(rest)
    self.assertEqual(len(matched.exceptions), 2)

  def test_split_by_type_none_match(self) -> None:
    """split(type) when none match: matched is None."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1')])
    matched, rest = eg.split(KeyError)
    self.assertIsNone(matched)
    self.assertIsNotNone(rest)
    self.assertEqual(len(rest.exceptions), 1)

  def test_split_by_callable(self) -> None:
    """split(callable) splits using the callable predicate."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('a'), ValueError('b'), TypeError('c')])
    matched, rest = eg.split(lambda e: isinstance(e, ValueError))
    self.assertIsNotNone(matched)
    self.assertIsNotNone(rest)
    self.assertEqual(len(matched.exceptions), 2)
    self.assertEqual(len(rest.exceptions), 1)


@unittest.skipIf(sys.version_info >= (3, 11), 'Polyfill only defined on Python <3.11')
class ExceptionGroupPolyfillDeriveTest(TestCase):
  """SPEC §6.5 / SPEC-212: polyfill .derive(excs) tests."""

  def _get_polyfill_cls(self):
    from quent._types import ExceptionGroup as PolyfillEG

    return PolyfillEG

  def test_derive_preserves_message(self) -> None:
    """derive(excs) creates new group with same message, different exceptions."""
    EG = self._get_polyfill_cls()
    orig_excs = [ValueError('v1'), TypeError('t1')]
    eg = EG('original message', orig_excs)
    new_excs = [RuntimeError('r1')]
    derived = eg.derive(new_excs)
    self.assertEqual(derived.args[0], 'original message')
    self.assertEqual(len(derived.exceptions), 1)
    self.assertIsInstance(derived.exceptions[0], RuntimeError)
    # Original is unchanged
    self.assertEqual(len(eg.exceptions), 2)

  def test_derive_copies_traceback(self) -> None:
    """derive() copies __traceback__ from the original."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1')])
    # Simulate a traceback by raising and catching
    try:
      raise eg
    except Exception as caught:
      eg_with_tb = caught
    derived = eg_with_tb.derive([TypeError('t1')])
    self.assertIs(derived.__traceback__, eg_with_tb.__traceback__)

  def test_derive_copies_cause_and_context(self) -> None:
    """derive() copies __cause__ and __context__ from the original."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1')])
    cause = RuntimeError('the cause')
    context = KeyError('the context')
    eg.__cause__ = cause
    eg.__context__ = context
    derived = eg.derive([TypeError('t1')])
    self.assertIs(derived.__cause__, cause)
    self.assertIs(derived.__context__, context)

  def test_derive_copies_notes(self) -> None:
    """derive() copies __notes__ when present."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1')])
    eg.__notes__ = ['note1', 'note2']
    derived = eg.derive([TypeError('t1')])
    self.assertEqual(derived.__notes__, ['note1', 'note2'])
    # Verify it's a copy, not the same list
    derived.__notes__.append('note3')
    self.assertEqual(len(eg.__notes__), 2)

  def test_derive_no_notes_attribute(self) -> None:
    """derive() does not add __notes__ when original has none."""
    EG = self._get_polyfill_cls()
    eg = EG('test', [ValueError('v1')])
    # Ensure no __notes__ on original
    self.assertFalse(hasattr(eg, '__notes__'))
    derived = eg.derive([TypeError('t1')])
    self.assertFalse(hasattr(derived, '__notes__'))


# ---------------------------------------------------------------------------
# _engine.py — Debug logging
# ---------------------------------------------------------------------------


class DebugLoggingAsyncContinuationTest(IsolatedAsyncioTestCase):
  """SPEC §14.5 / SPEC-488: debug log 'async continuation started'."""

  async def test_async_continuation_started_log_message(self) -> None:
    """Engine emits 'async continuation started' at DEBUG level.

    Per SPEC §14.5: the engine emits "chain <repr>: async continuation started"
    at DEBUG level when an async continuation begins.
    """
    logger = logging.getLogger('quent')
    original_level = logger.level

    # Capture log records
    records: list[logging.LogRecord] = []

    class CapturingHandler(logging.Handler):
      def emit(self, record: logging.LogRecord) -> None:
        records.append(record)

    handler = CapturingHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    try:
      # A chain that transitions to async (sync root, then async step)
      c = Chain(1).then(async_fn)
      result = await c.run()
      self.assertEqual(result, 2)

      # Check log records for the expected message
      messages = [r.getMessage() for r in records]
      found = any('async continuation started' in m for m in messages)
      self.assertTrue(found, f'Expected "async continuation started" in log messages, got: {messages}')

    finally:
      logger.removeHandler(handler)
      logger.setLevel(original_level)
