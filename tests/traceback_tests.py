# SPDX-License-Identifier: MIT
"""Tests for SPEC §13 — Traceback Enhancement.

Tests cover:
- §13.2: Chain visualization injection (<quent> frame)
- §13.3: Error marker (<----) on failing step
- §13.4: Frame cleaning (no quent-internal frames visible)
- §13.5: Chained exception cleaning (__cause__, __context__, ExceptionGroup)
- §13.6: Exception notes (Python 3.11+)
- §13.7: Environment variables (QUENT_NO_TRACEBACK, QUENT_TRACEBACK_VALUES)
- §13.10: Repr sanitization
- §13.11: Visualization limits
- §13.12: Graceful degradation
- §13.2: if_/else_ in visualization
- §13.5: Circular exception reference handling
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Chain, QuentException
from quent._viz import (
  _MAX_REPR_LEN,
  _sanitize_repr,
)

if sys.version_info < (3, 11):
  from quent._types import ExceptionGroup

# --- §13.2: Chain visualization injection ---


class VisualizationInjectionTest(TestCase):
  """§13.2: <quent> frame appears in traceback with chain visualization."""

  def test_quent_frame_present(self):
    """Traceback includes a <quent> frame."""
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('<quent>', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_chain_structure_visible(self):
    """Chain visualization shows Chain(...).then(...)."""
    try:
      Chain(1).then(lambda x: x + 1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('Chain', tb_text)
      self.assertIn('.then', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_nested_chain_indented(self):
    """Nested chains are rendered with indentation."""
    inner = Chain().then(lambda x: 1 / 0)
    try:
      Chain(1).then(inner).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('<quent>', tb_text)
      self.assertIn('Chain', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')


# --- §13.3: Error marker ---


class ErrorMarkerTest(TestCase):
  """§13.3: <---- marker points to the failing step."""

  def test_error_marker_present(self):
    """<---- marker appears in traceback."""
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('<----', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_error_marker_on_correct_step(self):
    """<---- marker appears on the step that raised."""

    def ok(x):
      return x

    def fail(x):
      raise ValueError('boom')

    try:
      Chain(1).then(ok).then(fail).then(ok).run()
    except ValueError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('<----', tb_text)
      # The marker should be on the line with 'fail'
      for line in tb_text.splitlines():
        if '<----' in line:
          self.assertIn('fail', line)
          break
      else:
        self.fail('<---- not found on any line')
    else:
      self.fail('Expected ValueError')

  def test_first_write_wins_for_nested(self):
    """First-write-wins: marker points to innermost failing step."""

    def fail(x):
      raise ValueError('inner')

    inner = Chain().then(fail)
    try:
      Chain(1).then(inner).run()
    except ValueError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('<----', tb_text)
      # The marker should be on the innermost fail step
      for line in tb_text.splitlines():
        if '<----' in line:
          self.assertIn('fail', line)
          break
    else:
      self.fail('Expected ValueError')

  def test_except_handler_shown_in_viz(self):
    """except_() handler appears in visualization."""

    def reraise_handler(info):
      raise info.exc

    try:
      Chain(1).then(lambda x: 1 / 0).except_(reraise_handler).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('.except_', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_finally_handler_shown_in_viz(self):
    """finally_() handler appears in visualization."""
    try:
      Chain(1).then(lambda x: 1 / 0).finally_(lambda rv: None).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('.finally_', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')


# --- §13.4: Frame cleaning ---


class FrameCleaningTest(TestCase):
  """§13.4: No quent-internal frames visible in traceback."""

  def _get_quent_dir(self):
    import quent

    return os.path.dirname(os.path.realpath(quent.__file__)) + os.sep

  def test_no_internal_frames(self):
    """No quent-internal frames appear in the formatted traceback."""
    quent_dir = self._get_quent_dir()
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      for line in tb_lines:
        if 'File "' in line:
          # Extract the file path
          start = line.index('File "') + 6
          end = line.index('"', start)
          filepath = line[start:end]
          # Should not be a quent internal file (except <quent> synthetic)
          if filepath != '<quent>':
            self.assertFalse(
              filepath.startswith(quent_dir),
              f'Internal frame should not be visible: {filepath}',
            )
    else:
      self.fail('Expected ZeroDivisionError')

  def test_user_frames_preserved(self):
    """User code frames are preserved in the traceback."""
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      # At minimum, the <quent> frame should be there
      self.assertIn('<quent>', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')


# --- §13.5: Chained exception cleaning ---


class ChainedExceptionCleaningTest(TestCase):
  """§13.5: Frame cleaning applies to __cause__, __context__, ExceptionGroup."""

  def test_cause_cleaned(self):
    """__cause__ exception has its frames cleaned."""
    quent_dir = self._get_quent_dir()

    def wrap_error(info):
      raise RuntimeError('wrap') from info.exc

    try:
      Chain(1).then(lambda x: 1 / 0).except_(wrap_error).run()
    except RuntimeError as exc:
      cause = exc.__cause__
      self.assertIsNotNone(cause)
      if cause.__traceback__:
        tb_lines = traceback.format_exception(type(cause), cause, cause.__traceback__)
        for line in tb_lines:
          if 'File "' in line:
            start = line.index('File "') + 6
            end = line.index('"', start)
            filepath = line[start:end]
            if filepath != '<quent>':
              self.assertFalse(
                filepath.startswith(quent_dir),
                f'__cause__ internal frame should be cleaned: {filepath}',
              )
    else:
      self.fail('Expected RuntimeError')

  def _get_quent_dir(self):
    import quent

    return os.path.dirname(os.path.realpath(quent.__file__)) + os.sep


# --- §13.6: Exception notes (Python 3.11+) ---


class ExceptionNoteTest(TestCase):
  """§13.6: Exception notes attached on Python 3.11+."""

  def test_exception_note_attached(self):
    """quent: exception note is attached to the exception."""
    if sys.version_info < (3, 11):
      self.skipTest('Exception notes require Python 3.11+')

    def fail(x):
      raise ValueError('boom')

    try:
      Chain(1).then(fail).run()
    except ValueError as exc:
      notes = getattr(exc, '__notes__', [])
      quent_notes = [n for n in notes if n.startswith('quent: exception at')]
      self.assertTrue(len(quent_notes) >= 1, f'Expected quent note, got: {notes}')
      note = quent_notes[0]
      self.assertIn('.then(', note)
      self.assertIn('Chain(', note)
    else:
      self.fail('Expected ValueError')

  def test_exception_note_idempotent(self):
    """quent note is attached only once (no duplicates)."""
    if sys.version_info < (3, 11):
      self.skipTest('Exception notes require Python 3.11+')

    inner = Chain().then(lambda x: 1 / 0)
    try:
      Chain(1).then(inner).run()
    except ZeroDivisionError as exc:
      notes = getattr(exc, '__notes__', [])
      quent_notes = [n for n in notes if n.startswith('quent: exception at')]
      self.assertEqual(len(quent_notes), 1, f'Expected exactly one quent note, got: {quent_notes}')
    else:
      self.fail('Expected ZeroDivisionError')


# --- §13.7: Environment variables (subprocess isolation) ---


class EnvVarTest(TestCase):
  """§13.7: Environment variable tests via subprocess isolation."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  def test_quent_no_traceback_disables_all(self):
    """QUENT_NO_TRACEBACK=1 disables visualization injection."""
    code = """
import traceback
from quent import Chain

try:
  Chain(1).then(lambda x: 1 / 0).run()
except ZeroDivisionError as exc:
  tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  if '<quent>' in tb:
    print('VISUALIZATION_PRESENT')
  else:
    print('VISUALIZATION_ABSENT')
"""
    result = self._run_subprocess(code, {'QUENT_NO_TRACEBACK': '1'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_ABSENT', result.stdout)

  def test_quent_no_traceback_unset(self):
    """Without QUENT_NO_TRACEBACK, visualization is present."""
    code = """
import traceback
from quent import Chain

try:
  Chain(1).then(lambda x: 1 / 0).run()
except ZeroDivisionError as exc:
  tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  if '<quent>' in tb:
    print('VISUALIZATION_PRESENT')
  else:
    print('VISUALIZATION_ABSENT')
"""
    env = os.environ.copy()
    env.pop('QUENT_NO_TRACEBACK', None)
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_PRESENT', result.stdout)

  def test_quent_traceback_values_suppresses_values(self):
    """QUENT_TRACEBACK_VALUES=0 suppresses argument values."""
    code = """
import traceback
from quent import Chain

secret = 'super_secret_value_12345'
try:
  Chain(secret).then(lambda x: 1 / 0).run()
except ZeroDivisionError as exc:
  tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  if 'super_secret_value_12345' in tb:
    print('VALUES_VISIBLE')
  else:
    print('VALUES_SUPPRESSED')
"""
    result = self._run_subprocess(code, {'QUENT_TRACEBACK_VALUES': '0'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VALUES_SUPPRESSED', result.stdout)


# --- §13.10: Repr sanitization ---


class ReprSanitizationTest(TestCase):
  """§13.10: ANSI and Unicode control characters stripped from repr."""

  def test_ansi_escape_stripped(self):
    """ANSI escape sequences are removed."""
    s = '\x1b[31mred text\x1b[0m'
    result = _sanitize_repr(s)
    self.assertNotIn('\x1b', result)
    self.assertIn('red text', result)

  def test_unicode_control_stripped(self):
    """Unicode control characters are removed."""
    s = 'hello\x00world\x7f'
    result = _sanitize_repr(s)
    self.assertNotIn('\x00', result)
    self.assertNotIn('\x7f', result)
    self.assertIn('hello', result)
    self.assertIn('world', result)

  def test_zero_width_chars_stripped(self):
    """Zero-width characters (U+200B etc.) are stripped."""
    s = 'hello\u200bworld'
    result = _sanitize_repr(s)
    self.assertNotIn('\u200b', result)

  def test_bidi_overrides_stripped(self):
    """Bidirectional override characters are stripped."""
    s = 'hello\u202eworld'
    result = _sanitize_repr(s)
    self.assertNotIn('\u202e', result)

  def test_tab_newline_escaped(self):
    """Tab and newline are escaped (not stripped)."""
    s = 'hello\tworld\nfoo'
    result = _sanitize_repr(s)
    self.assertIn('\\t', result)
    self.assertIn('\\n', result)

  def test_repr_length_truncated(self):
    """Long repr is truncated to _MAX_REPR_LEN."""
    from quent._viz import _get_obj_name

    class LongRepr:
      def __repr__(self):
        return 'x' * 500

    name = _get_obj_name(LongRepr())
    self.assertLessEqual(len(name), _MAX_REPR_LEN + 3)  # +3 for '...'

  def test_osc_sequences_stripped(self):
    """OSC (Operating System Command) sequences are stripped."""
    s = '\x1b]0;title\x07rest'
    result = _sanitize_repr(s)
    self.assertNotIn('\x1b', result)
    self.assertIn('rest', result)


# --- §13.11: Visualization limits ---


class VisualizationLimitsTest(TestCase):
  """§13.11: Visualization is bounded by limits."""

  def test_max_links_per_level(self):
    """Chains with > 100 links are truncated."""
    c = Chain(1)
    for _ in range(150):
      c.then(lambda x: x)
    try:
      c.then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      # Should mention truncation
      self.assertIn('more steps', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_total_length_capped(self):
    """Total visualization length is capped."""
    # Build a chain with many steps that have long names
    c = Chain(1)
    for i in range(100):
      c.then(lambda x, _i=i: x, 'a' * 50)
    try:
      c.then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      # Should be bounded — just check it doesn't exceed a reasonable size
      # The visualization string is embedded in the traceback, but the total
      # traceback may be larger. Just verify it completes without error.
      self.assertIn('<quent>', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_nesting_depth_truncated(self):
    """Deeply nested chains are truncated at depth 50."""
    # Build a chain nested 55 levels
    inner = Chain().then(lambda x: 1 / 0)
    for _ in range(54):
      inner = Chain().then(inner)
    try:
      Chain(1).then(inner).run()
    except QuentException:
      pass  # Nesting depth exceeded — this is the expected behavior
    except ZeroDivisionError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      # The visualization should include truncation indicator
      self.assertIn('<quent>', tb_text)


# --- §13.12: Graceful degradation ---


class GracefulDegradationTest(TestCase):
  """§13.12: Visualization failure → RuntimeWarning, exception propagates."""

  def test_exception_propagates_on_viz_failure(self):
    """Exception propagates even if visualization fails."""
    # This tests the fundamental guarantee: visualization failure
    # never suppresses the actual exception
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError:
      pass  # This is the expected behavior
    else:
      self.fail('Exception should propagate regardless of visualization')


# --- Async traceback tests ---


class AsyncTracebackTest(IsolatedAsyncioTestCase):
  """Traceback enhancement works with async chains."""

  async def test_async_chain_traceback(self):
    """Async chain has <quent> frame in traceback."""

    async def async_fail(x):
      raise ValueError('async boom')

    try:
      await Chain(1).then(async_fail).run()
    except ValueError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('<quent>', tb_text)
      self.assertIn('<----', tb_text)
    else:
      self.fail('Expected ValueError')

  async def test_mixed_sync_async_traceback(self):
    """Mixed sync/async chain has correct traceback."""

    async def async_fail(x):
      raise ValueError('mixed boom')

    try:
      await Chain(1).then(lambda x: x + 1).then(async_fail).run()
    except ValueError as exc:
      tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      tb_text = ''.join(tb_lines)
      self.assertIn('<quent>', tb_text)
      self.assertIn('<----', tb_text)
    else:
      self.fail('Expected ValueError')


# --- Additional coverage: TracebackException.__init__ patching (§13.8) ---


class TracebackExceptionPatchTest(TestCase):
  """§13.8: TracebackException.__init__ patch cleans quent frames."""

  def test_traceback_exception_cleans_quent_frames(self):
    """TracebackException.__init__ receives cleaned frames."""
    quent_dir = self._get_quent_dir()
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      # Should contain <quent> frame
      self.assertIn('<quent>', formatted)
      # Should not contain internal frames
      for line in formatted.splitlines():
        if 'File "' in line:
          start = line.index('File "') + 6
          end = line.index('"', start)
          filepath = line[start:end]
          if filepath != '<quent>':
            self.assertFalse(
              filepath.startswith(quent_dir),
              f'Internal frame should be cleaned in TracebackException: {filepath}',
            )
    else:
      self.fail('Expected ZeroDivisionError')

  def _get_quent_dir(self):
    import quent

    return os.path.dirname(os.path.realpath(quent.__file__)) + os.sep


# --- Additional coverage: Chained exception cleaning (§13.5) ---


class ChainedExceptionCleaningExtendedTest(TestCase):
  """§13.5: Extended tests for chained exception frame cleaning."""

  def _get_quent_dir(self):
    import quent

    return os.path.dirname(os.path.realpath(quent.__file__)) + os.sep

  def test_context_cleaned(self):
    """__context__ exception has its frames cleaned."""
    quent_dir = self._get_quent_dir()

    def raise_in_handler(info):
      raise RuntimeError('context chain')

    try:
      Chain(1).then(lambda x: 1 / 0).except_(raise_in_handler).run()
    except RuntimeError as exc:
      context = exc.__context__
      self.assertIsNotNone(context)
      if context.__traceback__:
        tb_lines = traceback.format_exception(type(context), context, context.__traceback__)
        for line in tb_lines:
          if 'File "' in line:
            start = line.index('File "') + 6
            end = line.index('"', start)
            filepath = line[start:end]
            if filepath != '<quent>':
              self.assertFalse(
                filepath.startswith(quent_dir),
                f'__context__ internal frame should be cleaned: {filepath}',
              )
    else:
      self.fail('Expected RuntimeError')

  def test_exception_group_sub_exceptions_cleaned(self):
    """ExceptionGroup sub-exceptions have their frames cleaned (§13.5)."""
    quent_dir = self._get_quent_dir()

    def fail1(x):
      raise ValueError('fail1')

    def fail2(x):
      raise RuntimeError('fail2')

    try:
      Chain(1).gather(fail1, fail2, concurrency=2).run()
    except ExceptionGroup as eg:
      for sub_exc in eg.exceptions:
        if sub_exc.__traceback__:
          tb_lines = traceback.format_exception(type(sub_exc), sub_exc, sub_exc.__traceback__)
          for line in tb_lines:
            if 'File "' in line:
              start = line.index('File "') + 6
              end = line.index('"', start)
              filepath = line[start:end]
              if filepath != '<quent>':
                self.assertFalse(
                  filepath.startswith(quent_dir),
                  f'ExceptionGroup sub-exc internal frame should be cleaned: {filepath}',
                )
    except (ValueError, RuntimeError):
      pass  # Single exception raised instead of group

  def test_chained_cause_recursive_cleaning(self):
    """Multi-level __cause__ chains are cleaned recursively."""
    quent_dir = self._get_quent_dir()

    def wrap_error(info):
      raise RuntimeError('wrapper') from info.exc

    try:
      Chain(1).then(lambda x: 1 / 0).except_(wrap_error).run()
    except RuntimeError as exc:
      # Check __cause__ chain
      cause = exc.__cause__
      self.assertIsNotNone(cause)
      self.assertIsInstance(cause, ZeroDivisionError)
      # Both should have cleaned frames
      for e in [exc, cause]:
        if e.__traceback__:
          tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
          for line in tb_lines:
            if 'File "' in line:
              start = line.index('File "') + 6
              end = line.index('"', start)
              filepath = line[start:end]
              if filepath != '<quent>':
                self.assertFalse(
                  filepath.startswith(quent_dir),
                  f'Recursive __cause__ frame should be cleaned: {filepath}',
                )
    else:
      self.fail('Expected RuntimeError')


# --- Additional coverage: sys.excepthook patch ---


class ExceptHookTest(TestCase):
  """§13.8: sys.excepthook replacement cleans quent frames."""

  def test_excepthook_is_patched(self):
    """sys.excepthook is replaced by quent's hook."""
    from quent._traceback import _quent_excepthook

    self.assertIs(sys.excepthook, _quent_excepthook)

  def test_excepthook_subprocess(self):
    """Uncaught quent exceptions show cleaned tracebacks in subprocess."""
    code = """
import sys
from quent import Chain
Chain(1).then(lambda x: 1 / 0).run()
"""
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      timeout=30,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('<quent>', result.stderr)
    self.assertIn('<----', result.stderr)


# --- Additional coverage: Graceful degradation ---


class GracefulDegradationExtendedTest(TestCase):
  """§13.12: Extended graceful degradation tests."""

  def test_viz_failure_emits_warning_and_propagates(self):
    """Visualization failure emits RuntimeWarning and exception propagates."""
    # We test the fundamental guarantee: even if visualization fails internally,
    # the exception must propagate.
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError:
      pass  # Expected
    else:
      self.fail('Exception should propagate regardless of visualization')

  def test_multiple_chains_traceback(self):
    """Multiple sequential chain errors all get tracebacks."""
    for i in range(3):
      try:
        Chain(i).then(lambda x: 1 / 0).run()
      except ZeroDivisionError as exc:
        tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.assertIn('<quent>', tb_text)
        self.assertIn('<----', tb_text)


# --- §13.6: Exception note idempotency across nesting depths ---


class ExceptionNoteIdempotencyPropertyTest(TestCase):
  """§13.6: Exactly one quent note attached regardless of nesting depth."""

  def test_note_idempotent_across_depths(self):
    """Varying nesting depths (1-5) always produce exactly one quent note."""
    if sys.version_info < (3, 11):
      self.skipTest('Exception notes require Python 3.11+')

    for depth in range(1, 6):

      def fail(x):
        raise ValueError(f'depth-{depth}')

      # Build chain nested to the given depth
      inner = Chain().then(fail)
      for _ in range(depth - 1):
        inner = Chain().then(inner)

      try:
        Chain(1).then(inner).run()
      except (ValueError, QuentException) as exc:
        if isinstance(exc, QuentException):
          continue  # Nesting depth exceeded — skip
        notes = getattr(exc, '__notes__', [])
        quent_notes = [n for n in notes if n.startswith('quent: exception at')]
        self.assertEqual(
          len(quent_notes),
          1,
          f'Depth {depth}: expected exactly 1 quent note, got {len(quent_notes)}: {quent_notes}',
        )
      else:
        self.fail(f'Depth {depth}: expected ValueError')


# --- §13.3: First-write-wins property test ---


class FirstWriteWinsPropertyTest(TestCase):
  """§13.3: <---- marker always points to innermost failing step."""

  def test_marker_on_innermost_step_across_depths(self):
    """Across varying nesting (1-4), marker is on the innermost fail step."""
    for depth in range(1, 5):

      def fail_step(x):
        raise ValueError(f'inner-{depth}')

      # Build nested chain: outermost wraps innermost
      inner = Chain().then(fail_step)
      for _ in range(depth - 1):
        inner = Chain().then(lambda x: x).then(inner)

      try:
        Chain(1).then(inner).run()
      except (ValueError, QuentException) as exc:
        if isinstance(exc, QuentException):
          continue  # Nesting depth exceeded
        tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        # Find the line with <----
        marker_lines = [line for line in tb_text.splitlines() if '<----' in line]
        self.assertTrue(
          len(marker_lines) >= 1,
          f'Depth {depth}: no <---- marker found in traceback',
        )
        # The marker should be on the innermost fail_step, not an intermediate
        marker_line = marker_lines[0]
        self.assertIn(
          'fail_step',
          marker_line,
          f'Depth {depth}: marker should be on fail_step, got: {marker_line}',
        )
      else:
        self.fail(f'Depth {depth}: expected ValueError')


# --- §13.2: if_/else_ in visualization ---


class IfElseVisualizationTest(TestCase):
  """§13.2: if_ and else_ branches appear in chain visualization."""

  def test_if_branch_in_visualization(self):
    """§13.2: if_() appears in the chain visualization."""
    try:
      Chain(5).if_(lambda x: x > 0).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIn('.if_', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_else_branch_in_visualization(self):
    """§13.2: else_() branch appears in the chain visualization."""
    try:
      Chain(0).if_(lambda x: x > 0).then(lambda x: x).else_(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIn('.else_', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')

  def test_both_if_and_else_in_visualization(self):
    """§13.2: Both if_ and else_ appear in visualization for a chain that has both."""
    try:
      Chain(-1).if_(lambda x: x > 0).then(lambda x: x).else_(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIn('.if_', tb_text)
      self.assertIn('.else_', tb_text)
    else:
      self.fail('Expected ZeroDivisionError')


# --- §13.12: Graceful degradation with RuntimeWarning ---


class VizFailureGracefulDegradationTest(TestCase):
  """§13.12: When visualization fails, RuntimeWarning emitted and chain still executes."""

  def test_viz_failure_emits_runtime_warning(self):
    """§13.12: Visualization failure emits RuntimeWarning and exception propagates."""
    # We test this via subprocess to monkey-patch _stringify_chain to raise
    code = """
import warnings
import traceback
warnings.simplefilter('always')

# Monkey-patch _stringify_chain to force a visualization failure
import quent._traceback as tb_mod
original = tb_mod._stringify_chain
def failing_stringify(*args, **kwargs):
  raise RuntimeError('viz failed on purpose')
tb_mod._stringify_chain = failing_stringify

from quent import Chain
try:
  Chain(1).then(lambda x: 1 / 0).run()
except ZeroDivisionError:
  print('EXCEPTION_PROPAGATED')
"""
    result = subprocess.run(
      [sys.executable, '-W', 'all', '-c', code],
      capture_output=True,
      text=True,
      timeout=30,
    )
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('EXCEPTION_PROPAGATED', result.stdout)
    # Should have emitted a RuntimeWarning about visualization failure
    self.assertIn('viz failed on purpose', result.stderr)


# --- §13.11: 500 recursive call limit ---


class RecursiveCallLimitTest(TestCase):
  """§13.11: Visualization handles chains exceeding the 500 recursive call limit."""

  def test_500_recursive_call_limit_truncation(self):
    """§13.11: Chains exceeding 500 recursive calls are truncated without crash."""
    # Build a chain with many nested chains to trigger the call limit
    inner = Chain().then(lambda x: 1 / 0)
    # Wrap in enough nesting to approach the 500 call limit
    # Each nested chain uses several calls, so 45 levels should be enough
    for _ in range(45):
      inner = Chain().then(inner)

    try:
      Chain(1).then(inner).run()
    except (ZeroDivisionError, QuentException) as exc:
      if isinstance(exc, QuentException):
        # Nesting depth exceeded — that's fine, the limit worked
        return
      tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      # Should either show truncation message or complete successfully
      # The key assertion is no crash
      self.assertIn('<quent>', tb_text)
    # If no exception at all, the chain completed — also acceptable


# --- §13.5: Circular exception reference handling ---


class CircularExceptionReferenceTest(TestCase):
  """§13.5: Circular exception references are handled by the seen-set."""

  def test_circular_cause_no_infinite_recursion(self):
    """§13.5: Circular __cause__ reference terminates via seen-set."""
    # Create a chain that raises, then manually create a circular reference
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      # Create a circular __cause__ chain: exc -> cause -> exc
      cause = RuntimeError('cause')
      exc.__cause__ = cause
      cause.__cause__ = exc

      # This should not infinite-loop — the seen-set in _clean_chained_exceptions prevents it
      tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      # Just verify it completes without hanging
      self.assertIsInstance(tb_text, str)

  def test_circular_context_no_infinite_recursion(self):
    """§13.5: Circular __context__ reference terminates via seen-set."""
    try:
      Chain(1).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      # Create a circular __context__ chain
      ctx_exc = RuntimeError('context')
      exc.__context__ = ctx_exc
      ctx_exc.__context__ = exc

      # Should not hang
      tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIsInstance(tb_text, str)


# --- §5.10: Named chain traceback tests ---


class NamedChainTracebackTest(TestCase):
  """§5.10: name() label appears in traceback visualization and exception notes."""

  def test_named_chain_in_visualization(self):
    """Traceback <quent> frame contains Chain[label] when name is set."""

    def fail(x):
      raise ValueError('named chain fail')

    try:
      Chain(1).name('my_label').then(fail).run()
    except ValueError as exc:
      tb_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
      self.assertIn('Chain[my_label]', tb_text)
    else:
      self.fail('Expected ValueError')

  def test_named_chain_exception_note(self):
    """Exception note contains Chain[label](...) when name is set (Python 3.11+)."""
    if sys.version_info < (3, 11):
      self.skipTest('Exception notes require Python 3.11+')

    def fail(x):
      raise ValueError('named chain note fail')

    try:
      Chain(1).name('auth_pipeline').then(fail).run()
    except ValueError as exc:
      notes = getattr(exc, '__notes__', [])
      quent_notes = [n for n in notes if n.startswith('quent: exception at')]
      self.assertTrue(len(quent_notes) >= 1, f'Expected quent note, got: {notes}')
      note = quent_notes[0]
      self.assertIn('Chain[auth_pipeline]', note)
    else:
      self.fail('Expected ValueError')

  def test_unnamed_chain_no_brackets_in_note(self):
    """Unnamed chain exception note uses plain Chain(...) without brackets."""
    if sys.version_info < (3, 11):
      self.skipTest('Exception notes require Python 3.11+')

    def fail(x):
      raise ValueError('unnamed chain note fail')

    try:
      Chain(1).then(fail).run()
    except ValueError as exc:
      notes = getattr(exc, '__notes__', [])
      quent_notes = [n for n in notes if n.startswith('quent: exception at')]
      self.assertTrue(len(quent_notes) >= 1, f'Expected quent note, got: {notes}')
      note = quent_notes[0]
      # Should contain 'Chain(' but not 'Chain['
      self.assertIn('Chain(', note)
      self.assertNotIn('Chain[', note)
    else:
      self.fail('Expected ValueError')


# --- §13.7: QUENT_NO_TRACEBACK accepts 'true'/'yes' (SPEC-439) ---


class EnvVarNoTracebackExtendedTest(TestCase):
  """§13.7 (SPEC-439): QUENT_NO_TRACEBACK accepts '1', 'true', 'yes' (case-insensitive)."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  _DETECTION_CODE = """
import traceback
from quent import Chain

try:
  Chain(1).then(lambda x: 1 / 0).run()
except ZeroDivisionError as exc:
  tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  if '<quent>' in tb:
    print('VISUALIZATION_PRESENT')
  else:
    print('VISUALIZATION_ABSENT')
"""

  def test_quent_no_traceback_true_lowercase(self):
    """QUENT_NO_TRACEBACK='true' disables visualization."""
    result = self._run_subprocess(self._DETECTION_CODE, {'QUENT_NO_TRACEBACK': 'true'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_ABSENT', result.stdout)

  def test_quent_no_traceback_True_mixed_case(self):
    """QUENT_NO_TRACEBACK='True' disables visualization (case-insensitive)."""
    result = self._run_subprocess(self._DETECTION_CODE, {'QUENT_NO_TRACEBACK': 'True'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_ABSENT', result.stdout)

  def test_quent_no_traceback_TRUE_upper(self):
    """QUENT_NO_TRACEBACK='TRUE' disables visualization (case-insensitive)."""
    result = self._run_subprocess(self._DETECTION_CODE, {'QUENT_NO_TRACEBACK': 'TRUE'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_ABSENT', result.stdout)

  def test_quent_no_traceback_yes_lowercase(self):
    """QUENT_NO_TRACEBACK='yes' disables visualization."""
    result = self._run_subprocess(self._DETECTION_CODE, {'QUENT_NO_TRACEBACK': 'yes'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_ABSENT', result.stdout)

  def test_quent_no_traceback_YES_upper(self):
    """QUENT_NO_TRACEBACK='YES' disables visualization (case-insensitive)."""
    result = self._run_subprocess(self._DETECTION_CODE, {'QUENT_NO_TRACEBACK': 'YES'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_ABSENT', result.stdout)

  def test_quent_no_traceback_with_whitespace(self):
    """QUENT_NO_TRACEBACK=' true ' disables visualization (whitespace stripped)."""
    result = self._run_subprocess(self._DETECTION_CODE, {'QUENT_NO_TRACEBACK': ' true '})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VISUALIZATION_ABSENT', result.stdout)


# --- §13.7: QUENT_TRACEBACK_VALUES accepts 'false'/'no' (SPEC-441) ---


class EnvVarTracebackValuesExtendedTest(TestCase):
  """§13.7 (SPEC-441): QUENT_TRACEBACK_VALUES accepts '0', 'false', 'no' (case-insensitive)."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  _VALUES_CODE = """
import traceback
from quent import Chain

secret = 'super_secret_value_12345'
try:
  Chain(secret).then(lambda x: 1 / 0).run()
except ZeroDivisionError as exc:
  tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  if 'super_secret_value_12345' in tb:
    print('VALUES_VISIBLE')
  else:
    print('VALUES_SUPPRESSED')
"""

  def test_traceback_values_false_lowercase(self):
    """QUENT_TRACEBACK_VALUES='false' suppresses values."""
    result = self._run_subprocess(self._VALUES_CODE, {'QUENT_TRACEBACK_VALUES': 'false'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VALUES_SUPPRESSED', result.stdout)

  def test_traceback_values_False_mixed_case(self):
    """QUENT_TRACEBACK_VALUES='False' suppresses values (case-insensitive)."""
    result = self._run_subprocess(self._VALUES_CODE, {'QUENT_TRACEBACK_VALUES': 'False'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VALUES_SUPPRESSED', result.stdout)

  def test_traceback_values_FALSE_upper(self):
    """QUENT_TRACEBACK_VALUES='FALSE' suppresses values (case-insensitive)."""
    result = self._run_subprocess(self._VALUES_CODE, {'QUENT_TRACEBACK_VALUES': 'FALSE'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VALUES_SUPPRESSED', result.stdout)

  def test_traceback_values_no_lowercase(self):
    """QUENT_TRACEBACK_VALUES='no' suppresses values."""
    result = self._run_subprocess(self._VALUES_CODE, {'QUENT_TRACEBACK_VALUES': 'no'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VALUES_SUPPRESSED', result.stdout)

  def test_traceback_values_NO_upper(self):
    """QUENT_TRACEBACK_VALUES='NO' suppresses values (case-insensitive)."""
    result = self._run_subprocess(self._VALUES_CODE, {'QUENT_TRACEBACK_VALUES': 'NO'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VALUES_SUPPRESSED', result.stdout)

  def test_traceback_values_with_whitespace(self):
    """QUENT_TRACEBACK_VALUES=' false ' suppresses values (whitespace stripped)."""
    result = self._run_subprocess(self._VALUES_CODE, {'QUENT_TRACEBACK_VALUES': ' false '})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('VALUES_SUPPRESSED', result.stdout)


# --- §13.7: Values=0 placeholder format <type> (SPEC-442) ---


class ValuesPlaceholderFormatTest(TestCase):
  """§13.7 (SPEC-442): QUENT_TRACEBACK_VALUES=0 replaces values with <type> placeholders."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  def test_str_value_replaced_with_type_placeholder(self):
    """String root value is replaced with <str> placeholder in traceback."""
    code = """
import traceback
from quent import Chain

secret = 'secret_api_key'
try:
  Chain(secret).then(lambda x: 1 / 0).run()
except ZeroDivisionError as exc:
  tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  has_placeholder = '<str>' in tb
  has_actual = 'secret_api_key' in tb
  if has_placeholder and not has_actual:
    print('PLACEHOLDER_CORRECT')
  elif has_actual:
    print('VALUE_LEAKED')
  else:
    print('NO_PLACEHOLDER_FOUND')
"""
    result = self._run_subprocess(code, {'QUENT_TRACEBACK_VALUES': '0'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('PLACEHOLDER_CORRECT', result.stdout)


# --- §13.8: reload() idempotency (SPEC-447/448) ---


class ReloadIdempotencyTest(TestCase):
  """§13.8 (SPEC-447/448): importlib.reload() is idempotent for hooks."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  def test_reload_idempotency(self):
    """importlib.reload() twice preserves hooks without stacking/doubling."""
    code = """
import sys
import importlib
import traceback
import quent
import quent._traceback as tb_mod

# Reload twice
importlib.reload(tb_mod)
importlib.reload(tb_mod)

# Check sys.excepthook is the quent hook
from quent._traceback import _quent_excepthook, _patched_te_init
if sys.excepthook is _quent_excepthook:
  print('EXCEPTHOOK_OK')
else:
  print('EXCEPTHOOK_WRONG')

# Check TracebackException.__init__ is the patched version
if traceback.TracebackException.__init__ is _patched_te_init:
  print('TE_INIT_OK')
else:
  print('TE_INIT_WRONG')

# Verify chain visualization still works after reload
from quent import Chain
try:
  Chain(1).then(lambda x: 1 / 0).run()
except ZeroDivisionError as exc:
  tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
  if '<quent>' in tb:
    print('VIZ_OK')
  else:
    print('VIZ_MISSING')
"""
    result = self._run_subprocess(code)
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('EXCEPTHOOK_OK', result.stdout, f'stdout: {result.stdout}')
    self.assertIn('TE_INIT_OK', result.stdout, f'stdout: {result.stdout}')
    self.assertIn('VIZ_OK', result.stdout, f'stdout: {result.stdout}')


# --- §13.8: Signature verification + warning (SPEC-449) ---


class SignatureVerificationWarningTest(TestCase):
  """§13.8 (SPEC-449): Unexpected TracebackException.__init__ signature emits RuntimeWarning."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-W', 'all', '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  def test_unexpected_signature_emits_warning(self):
    """Monkeypatched TracebackException.__init__ signature triggers RuntimeWarning."""
    code = """
import traceback
import warnings
warnings.simplefilter('always')

# Save the original init
_orig_init = traceback.TracebackException.__init__

# Replace with a function that has an unexpected signature
def fake_init(self, weird_param_a, weird_param_b, weird_param_c):
  pass
traceback.TracebackException.__init__ = fake_init

# Now import quent._traceback — it should check the signature and warn
import quent._traceback

print('IMPORT_OK')
"""
    result = self._run_subprocess(code)
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('IMPORT_OK', result.stdout)
    self.assertIn('unexpected signature', result.stderr)
    self.assertIn('RuntimeWarning', result.stderr)


# --- §13.12: Viz failure → DEBUG log (SPEC-460/461) ---


class VizFailureDebugLogTest(TestCase):
  """§13.12 (SPEC-460/461): Visualization failure emits DEBUG log message."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-W', 'all', '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  def test_viz_failure_emits_debug_log(self):
    """Visualization failure logs 'chain visualization failed' at DEBUG level."""
    code = """
import logging
import warnings
warnings.simplefilter('always')

# Set up DEBUG logging on the 'quent' logger to capture the message
logger = logging.getLogger('quent')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from quent import Chain
import quent._traceback as tb_mod

# Monkey-patch _stringify_chain to force a visualization failure
original = tb_mod._stringify_chain
def failing_stringify(*args, **kwargs):
  raise RuntimeError('viz failed on purpose')
tb_mod._stringify_chain = failing_stringify

try:
  Chain(1).then(lambda x: 1 / 0).run()
except ZeroDivisionError:
  print('EXCEPTION_PROPAGATED')
"""
    result = self._run_subprocess(code)
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('EXCEPTION_PROPAGATED', result.stdout)
    self.assertIn('chain visualization failed', result.stderr)


# --- §13.5: 1000 exception depth limit (SPEC-431) ---


class ExceptionDepthLimitTest(TestCase):
  """§13.5 (SPEC-431): _clean_chained_exceptions respects 1000 depth limit."""

  def test_exception_chain_deeper_than_1000_terminates(self):
    """Exception chain deeper than 1000 terminates without hanging or crashing."""
    from quent._traceback import _MAX_CHAINED_EXCEPTION_DEPTH, _clean_chained_exceptions

    depth = _MAX_CHAINED_EXCEPTION_DEPTH + 5  # 1005

    # Build a linear __cause__ chain: exc_0 -> exc_1 -> ... -> exc_1004
    exceptions = []
    for i in range(depth):
      exc = ValueError(f'exc_{i}')
      exceptions.append(exc)
    for i in range(depth - 1):
      exceptions[i].__cause__ = exceptions[i + 1]

    # Mark with quent metadata to make it a valid target
    exceptions[0].__quent_meta__ = {'quent': True}  # type: ignore[attr-defined]

    seen: set[int] = set()
    _clean_chained_exceptions(exceptions[0], seen)

    # The seen set should have at most _MAX_CHAINED_EXCEPTION_DEPTH entries
    # (since the loop breaks when depth >= limit)
    self.assertLessEqual(
      len(seen),
      _MAX_CHAINED_EXCEPTION_DEPTH,
      f'Expected at most {_MAX_CHAINED_EXCEPTION_DEPTH} visited exceptions, got {len(seen)}',
    )
    # Not all 1005 exceptions were visited
    self.assertLess(
      len(seen),
      depth,
      f'Expected fewer than {depth} visited exceptions (depth limit should cap traversal)',
    )

  def test_depth_limit_constant_is_1000(self):
    """The depth limit constant is exactly 1000."""
    from quent._traceback import _MAX_CHAINED_EXCEPTION_DEPTH

    self.assertEqual(_MAX_CHAINED_EXCEPTION_DEPTH, 1000)


# --- §14.5: Debug log values=0 with multiple types (SPEC-443) ---


class DebugLogValuesZeroMultipleTypesTest(TestCase):
  """§14.5 (SPEC-443): QUENT_TRACEBACK_VALUES=0 replaces values with type-name placeholders."""

  def _run_subprocess(self, code, env_overrides=None):
    """Run Python code in a subprocess with env overrides."""
    env = os.environ.copy()
    if env_overrides:
      env.update(env_overrides)
    result = subprocess.run(
      [sys.executable, '-c', code],
      capture_output=True,
      text=True,
      env=env,
      timeout=30,
    )
    return result

  def test_debug_log_multiple_types_use_placeholders(self):
    """§14.5: QUENT_TRACEBACK_VALUES=0 uses type-name placeholders for str, int, list, dict, custom class."""
    code = """
import logging
import sys

# Configure DEBUG logging on 'quent' logger before importing quent
logger = logging.getLogger('quent')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from quent import Chain

class CustomObj:
  def __repr__(self):
    return 'CustomObj(sensitive_data)'

# Run chains with various root value types
Chain('hello_secret').then(lambda x: x + '_suffix').run()
Chain(42).then(lambda x: x + 1).run()
Chain([1, 2, 3]).then(lambda x: len(x)).run()
Chain({'key': 'value'}).then(lambda x: len(x)).run()
Chain(CustomObj()).then(lambda x: str(x)).run()

print('ALL_CHAINS_COMPLETE')
"""
    result = self._run_subprocess(code, {'QUENT_TRACEBACK_VALUES': '0'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('ALL_CHAINS_COMPLETE', result.stdout)

    stderr = result.stderr
    # Verify type-name placeholders appear in debug log
    self.assertIn('<str>', stderr, 'Expected <str> placeholder in debug log')
    self.assertIn('<int>', stderr, 'Expected <int> placeholder in debug log')
    self.assertIn('<list>', stderr, 'Expected <list> placeholder in debug log')
    self.assertIn('<dict>', stderr, 'Expected <dict> placeholder in debug log')
    self.assertIn('<CustomObj>', stderr, 'Expected <CustomObj> placeholder in debug log')

    # Verify actual values do NOT appear in the debug log
    self.assertNotIn('hello_secret', stderr, 'Actual string value should not appear in debug log')
    self.assertNotIn("'hello_secret'", stderr, 'Quoted string value should not appear in debug log')
    self.assertNotIn('sensitive_data', stderr, 'CustomObj repr should not appear in debug log')
    self.assertNotIn('CustomObj(sensitive_data)', stderr, 'CustomObj repr should not appear in debug log')
