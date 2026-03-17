# SPDX-License-Identifier: MIT
"""Hypothesis-based fuzz tests for repr sanitization (CWE-117).

Tests the _sanitize_repr function and _get_obj_name truncation from _viz.py
using adversarial and random inputs to verify sanitization invariants.

All assertions are derived from SPEC.md section 13.10.
"""

from __future__ import annotations

from unittest import TestCase

from hypothesis import given, settings
from hypothesis import strategies as st

from quent._viz import _MAX_REPR_LEN, _get_obj_name, _sanitize_repr

# --- Adversarial strategies ---

# ANSI CSI sequences: ESC [ <params> <letter>
_csi_sequences = st.builds(
  lambda params, letter: f'\x1b[{params}{letter}',
  params=st.from_regex(r'[0-9;]{0,10}', fullmatch=True),
  letter=st.sampled_from(list('ABCDEFGHJKSTfmnsuhlr')),
)

# ANSI OSC sequences terminated by BEL
_osc_bel_sequences = st.builds(
  lambda payload: f'\x1b]{payload}\x07',
  payload=st.text(
    alphabet=st.characters(blacklist_characters='\x07\x1b'),
    min_size=0,
    max_size=30,
  ),
)

# ANSI OSC sequences terminated by ST (ESC + backslash)
_osc_st_sequences = st.builds(
  lambda payload: f'\x1b]{payload}\x1b\\',
  payload=st.text(
    alphabet=st.characters(blacklist_characters='\x07\x1b'),
    min_size=0,
    max_size=30,
  ),
)

# Simple ESC sequences: ESC + one char (not [ ] ( ))
_simple_esc_sequences = st.builds(
  lambda c: f'\x1b{c}',
  c=st.characters(blacklist_characters='[]()'),
)

# All ANSI escape sequence types
_ansi_sequences = st.one_of(
  _csi_sequences,
  _osc_bel_sequences,
  _osc_st_sequences,
  _simple_esc_sequences,
)

# C0 control characters (excluding \t=0x09, \n=0x0A, \r=0x0D, \x1b=ESC)
# ESC is excluded because it's tested via _ansi_sequences and interacts
# with adjacent characters (forming ANSI sequences that consume them).
_c0_controls = st.sampled_from(
  [chr(c) for c in range(0x00, 0x09)]  # NUL through BS
  + [chr(0x0B), chr(0x0C)]  # VT, FF
  + [chr(c) for c in range(0x0E, 0x1B)]  # SO through SUB (before ESC)
  + [chr(c) for c in range(0x1C, 0x20)]  # FS through US (after ESC)
)

# C1 control characters (DEL + 0x80-0x9F)
_c1_controls = st.sampled_from([chr(0x7F)] + [chr(c) for c in range(0x80, 0xA0)])

# Zero-width and invisible unicode characters
_zero_width_chars = st.sampled_from(
  [
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u200e',  # left-to-right mark
    '\u200f',  # right-to-left mark
  ]
)

# Bidirectional override characters
_bidi_chars = st.sampled_from(
  [
    '\u2028',  # line separator
    '\u2029',  # paragraph separator
    '\u202a',  # left-to-right embedding
    '\u202b',  # right-to-left embedding
    '\u202c',  # pop directional formatting
    '\u202d',  # left-to-right override
    '\u202e',  # right-to-left override
  ]
)

# Other invisible format characters
_format_chars = st.sampled_from(
  [
    '\u2060',  # word joiner
    '\u2061',  # function application
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\u2066',  # left-to-right isolate
    '\u2067',  # right-to-left isolate
    '\u2068',  # first strong isolate
    '\u2069',  # pop directional isolate
    '\ufeff',  # byte order mark
    '\ufff9',  # interlinear annotation anchor
    '\ufffa',  # interlinear annotation separator
    '\ufffb',  # interlinear annotation terminator
  ]
)

# All control/dangerous characters (excluding ESC)
_all_control_chars = st.one_of(
  _c0_controls,
  _c1_controls,
  _zero_width_chars,
  _bidi_chars,
  _format_chars,
)

# Safe printable ASCII text (for verifying preservation)
_safe_ascii = st.text(
  alphabet=st.characters(
    min_codepoint=0x20,
    max_codepoint=0x7E,
  ),
  min_size=0,
  max_size=20,
)

# Mixed adversarial strings: interleave safe text with dangerous content
_adversarial_strings = st.builds(
  lambda parts: ''.join(parts),
  parts=st.lists(
    st.one_of(
      _safe_ascii,
      _ansi_sequences,
      _all_control_chars,
      st.just('\t'),
      st.just('\n'),
      st.just('\r'),
      st.just('\x00'),
    ),
    min_size=1,
    max_size=15,
  ),
)

# Strings that exceed the truncation limit
_long_adversarial_strings = st.builds(
  lambda base, repeat: base * repeat,
  base=_adversarial_strings,
  repeat=st.integers(min_value=5, max_value=30),
)

# Fully random unicode text (broad fuzzing)
_random_unicode = st.text(min_size=0, max_size=300)

# Characters that the control char regex should strip
_STRIPPED_CHAR_RANGES = [
  *range(0x00, 0x09),
  0x0B,
  0x0C,
  *range(0x0E, 0x20),
  *range(0x7F, 0xA0),
  *range(0x200B, 0x2010),
  *range(0x2028, 0x202F),
  *range(0x2060, 0x206A),
  0xFEFF,
  *range(0xFFF9, 0xFFFC),
]
_STRIPPED_CHARS = frozenset(chr(c) for c in _STRIPPED_CHAR_RANGES)


class ReprSanitizationFuzzTest(TestCase):
  """Hypothesis-based fuzz tests for SPEC section 13.10 repr sanitization (CWE-117).

  Invariants tested:
  - No ANSI escape sequences remain after sanitization
  - No control characters remain (C0/C1, zero-width, bidi, BOM, etc.)
  - Tab, newline, and carriage return are converted to visible escape forms
  - Printable ASCII is never mutated
  - Sanitization is idempotent
  - Sanitization never raises (infallible)
  - _get_obj_name truncation respects _MAX_REPR_LEN
  - Log injection vectors (newlines, carriage returns) are neutralized
  """

  # --- Core invariant: no ANSI escape sequences in output ---

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_no_ansi_escapes_in_output(self, s):
    """SPEC section 13.10: ANSI escape sequences are stripped."""
    result = _sanitize_repr(s)
    self.assertNotIn('\x1b', result)

  @given(s=_ansi_sequences)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_pure_ansi_sequences_produce_no_escape(self, s):
    """SPEC section 13.10: pure ANSI sequence inputs produce no ESC character."""
    result = _sanitize_repr(s)
    self.assertNotIn('\x1b', result)

  @given(
    prefix=_safe_ascii,
    ansi=_ansi_sequences,
    suffix=_safe_ascii,
  )
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_ansi_stripped_preserves_surrounding_text(self, prefix, ansi, suffix):
    """SPEC section 13.10: ANSI stripping preserves surrounding printable text."""
    result = _sanitize_repr(prefix + ansi + suffix)
    self.assertIn(prefix, result)
    self.assertIn(suffix, result)
    self.assertNotIn('\x1b', result)

  # --- Core invariant: no control characters in output ---

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_no_control_chars_in_output(self, s):
    """SPEC section 13.10: unicode control characters are stripped."""
    result = _sanitize_repr(s)
    for ch in result:
      self.assertNotIn(
        ch,
        _STRIPPED_CHARS,
        f'Control character U+{ord(ch):04X} ({ch!r}) found in sanitized output',
      )

  @given(ctrl=_all_control_chars)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_individual_control_chars_stripped(self, ctrl):
    """SPEC section 13.10: each individual control character is stripped."""
    result = _sanitize_repr(ctrl)
    self.assertNotIn(ctrl, result)

  @given(
    prefix=_safe_ascii,
    ctrl=_all_control_chars,
    suffix=_safe_ascii,
  )
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_control_char_stripped_preserves_surrounding(self, prefix, ctrl, suffix):
    """SPEC section 13.10: control char stripping preserves surrounding text."""
    result = _sanitize_repr(prefix + ctrl + suffix)
    self.assertIn(prefix, result)
    self.assertIn(suffix, result)

  # --- ESC character specifically (dual nature: control char + ANSI start) ---

  @given(s=_safe_ascii)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_bare_esc_stripped(self, s):
    """SPEC section 13.10: bare ESC at end of string is stripped as control char."""
    result = _sanitize_repr(s + '\x1b')
    self.assertNotIn('\x1b', result)
    self.assertIn(s, result)

  @given(
    prefix=_safe_ascii,
    suffix=_safe_ascii,
  )
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_esc_between_text_stripped(self, prefix, suffix):
    """SPEC section 13.10: ESC between text forms simple ESC sequence; ESC removed."""
    # ESC + first char of suffix may form a simple ESC sequence, consuming
    # that character. This is correct behavior — the test only verifies
    # that no ESC remains and the prefix is preserved.
    result = _sanitize_repr(prefix + '\x1b' + suffix)
    self.assertNotIn('\x1b', result)
    self.assertIn(prefix, result)

  # --- Tab, newline, carriage return conversion ---

  @given(s=_safe_ascii)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_tab_converted_to_visible(self, s):
    """SPEC section 13.10: tabs are converted to visible \\t escape form."""
    result = _sanitize_repr(s + '\t' + s)
    self.assertNotIn('\t', result)
    self.assertIn('\\t', result)

  @given(s=_safe_ascii)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_newline_converted_to_visible(self, s):
    """SPEC section 13.10: newlines are converted to visible \\n escape form."""
    result = _sanitize_repr(s + '\n' + s)
    self.assertNotIn('\n', result)
    self.assertIn('\\n', result)

  @given(s=_safe_ascii)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_carriage_return_converted_to_visible(self, s):
    """SPEC section 13.10: carriage returns are converted to visible \\r escape form."""
    result = _sanitize_repr(s + '\r' + s)
    self.assertNotIn('\r', result)
    self.assertIn('\\r', result)

  @given(
    tabs=st.integers(min_value=1, max_value=10),
    newlines=st.integers(min_value=1, max_value=10),
    crs=st.integers(min_value=1, max_value=10),
  )
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_multiple_whitespace_chars_all_converted(self, tabs, newlines, crs):
    """SPEC section 13.10: all tab/newline/CR instances converted."""
    s = '\t' * tabs + '\n' * newlines + '\r' * crs
    result = _sanitize_repr(s)
    self.assertNotIn('\t', result)
    self.assertNotIn('\n', result)
    self.assertNotIn('\r', result)
    self.assertEqual(result.count('\\t'), tabs)
    self.assertEqual(result.count('\\n'), newlines)
    self.assertEqual(result.count('\\r'), crs)

  # --- Printable ASCII preservation ---

  @given(s=_safe_ascii)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_printable_ascii_preserved(self, s):
    """SPEC section 13.10: printable ASCII is never mutated by sanitization."""
    result = _sanitize_repr(s)
    self.assertEqual(result, s)

  # --- Idempotency ---

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_idempotent_adversarial(self, s):
    """SPEC section 13.10: sanitization is idempotent on adversarial input."""
    once = _sanitize_repr(s)
    twice = _sanitize_repr(once)
    self.assertEqual(once, twice)

  @given(s=_random_unicode)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_idempotent_random_unicode(self, s):
    """SPEC section 13.10: sanitization is idempotent on random unicode."""
    once = _sanitize_repr(s)
    twice = _sanitize_repr(once)
    self.assertEqual(once, twice)

  # --- Infallibility (never raises) ---

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_never_raises_adversarial(self, s):
    """SPEC section 13.10: sanitization never raises on adversarial input."""
    try:
      _sanitize_repr(s)
    except Exception as e:
      self.fail(f'_sanitize_repr raised {type(e).__name__}: {e}')

  @given(s=_random_unicode)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_never_raises_random_unicode(self, s):
    """SPEC section 13.10: sanitization never raises on random unicode."""
    try:
      _sanitize_repr(s)
    except Exception as e:
      self.fail(f'_sanitize_repr raised {type(e).__name__}: {e}')

  @given(s=st.binary(min_size=0, max_size=200))
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_never_raises_decoded_binary(self, s):
    """SPEC section 13.10: sanitization never raises on decoded binary content."""
    try:
      text = s.decode('latin-1')  # latin-1 decodes any byte sequence
      _sanitize_repr(text)
    except Exception as e:
      self.fail(f'_sanitize_repr raised {type(e).__name__}: {e}')

  # --- Truncation via _get_obj_name ---

  @given(s=_long_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_get_obj_name_truncation(self, s):
    """SPEC section 13.10: repr length truncated to _MAX_REPR_LEN."""

    class FakeObj:
      def __repr__(self_):
        return s

    name = _get_obj_name(FakeObj())
    # Truncated output is at most _MAX_REPR_LEN + len('...') = 203
    self.assertLessEqual(len(name), _MAX_REPR_LEN + 3)
    # Must not contain ANSI escapes or control chars
    self.assertNotIn('\x1b', name)
    for ch in name:
      self.assertNotIn(ch, _STRIPPED_CHARS)

  @given(n=st.integers(min_value=_MAX_REPR_LEN + 1, max_value=_MAX_REPR_LEN + 500))
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_truncation_adds_ellipsis(self, n):
    """SPEC section 13.10: repr exceeding limit gets '...' suffix."""
    long_str = 'x' * n

    class FakeObj:
      def __repr__(self_):
        return long_str

    name = _get_obj_name(FakeObj())
    self.assertTrue(name.endswith('...'))
    self.assertEqual(len(name), _MAX_REPR_LEN + 3)

  @given(n=st.integers(min_value=1, max_value=_MAX_REPR_LEN))
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_no_truncation_within_limit(self, n):
    """SPEC section 13.10: repr within limit is not truncated."""
    safe_str = 'a' * n

    class FakeObj:
      def __repr__(self_):
        return safe_str

    name = _get_obj_name(FakeObj())
    self.assertEqual(name, safe_str)
    self.assertFalse(name.endswith('...'))

  # --- _get_obj_name infallibility ---

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_get_obj_name_never_raises(self, s):
    """SPEC section 13.10: _get_obj_name never raises, even with adversarial repr."""

    class FakeObj:
      def __repr__(self_):
        return s

    try:
      _get_obj_name(FakeObj())
    except Exception as e:
      self.fail(f'_get_obj_name raised {type(e).__name__}: {e}')

  @given(st.data())
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_get_obj_name_handles_repr_exception(self, data):
    """SPEC section 13.10: _get_obj_name handles repr() that raises."""

    class BadRepr:
      def __repr__(self):
        raise RuntimeError('malicious repr')

    name = _get_obj_name(BadRepr())
    self.assertEqual(name, 'BadRepr')

  # --- Log injection vectors ---

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_no_raw_newlines_in_output(self, s):
    """SPEC section 13.10: no raw newlines (log line forging prevention)."""
    result = _sanitize_repr(s)
    self.assertNotIn('\n', result)

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_no_raw_carriage_returns_in_output(self, s):
    """SPEC section 13.10: no raw carriage returns (line overwrite prevention)."""
    result = _sanitize_repr(s)
    self.assertNotIn('\r', result)

  @given(s=_adversarial_strings)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_no_null_bytes_in_output(self, s):
    """SPEC section 13.10: no null bytes in output."""
    result = _sanitize_repr(s)
    self.assertNotIn('\x00', result)

  # --- Combined adversarial: ANSI + control + whitespace ---

  @given(
    ansi1=_ansi_sequences,
    ctrl=_all_control_chars,
    ansi2=_ansi_sequences,
    safe=_safe_ascii,
  )
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_combined_ansi_and_control_chars(self, ansi1, ctrl, ansi2, safe):
    """SPEC section 13.10: combined ANSI + control chars fully sanitized."""
    s = ansi1 + '\t' + ctrl + safe + ansi2 + '\n' + ctrl
    result = _sanitize_repr(s)
    self.assertNotIn('\x1b', result)
    self.assertNotIn('\t', result)
    self.assertNotIn('\n', result)
    for ch in result:
      self.assertNotIn(ch, _STRIPPED_CHARS)
    self.assertIn(safe, result)

  @given(s=_random_unicode)
  @settings(max_examples=500, deadline=None, derandomize=True)
  def test_output_terminal_safe_random_unicode(self, s):
    """SPEC section 13.10: output is safe for terminal display (random unicode)."""
    result = _sanitize_repr(s)
    # No ESC character
    self.assertNotIn('\x1b', result)
    # No raw whitespace control chars
    self.assertNotIn('\t', result)
    self.assertNotIn('\n', result)
    self.assertNotIn('\r', result)
    # No stripped control chars
    for ch in result:
      self.assertNotIn(ch, _STRIPPED_CHARS)
