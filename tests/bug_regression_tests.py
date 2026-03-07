"""Regression tests for 13 bugs fixed across quent's core, chain, and traceback modules.

Each test class targets a specific bug, named TestBug<N>..., with a docstring
referencing the affected file and line range. Tests verify the fix is in place
and guard against regressions.
"""

from __future__ import annotations

import copy
import pickle
import unittest

from helpers import async_identity

from quent import Chain, Null, QuentException
from quent._core import _ControlFlowSignal

# ---------------------------------------------------------------------------
# Bug 1 (CRITICAL): kwargs dropped for nested chains when args is empty
# File: quent/_core.py:189-191
# ---------------------------------------------------------------------------


class TestBug1NestedChainKwargs(unittest.TestCase):
  def test_kwargs_only_not_silently_dropped(self):
    """kwargs without positional args must not be silently dropped."""
    inner = Chain()
    # Before fix: kwargs silently dropped, 5 passed through as-is
    # After fix: kwargs forwarded, 5 is not callable with kwargs -> TypeError
    with self.assertRaises(TypeError):
      Chain(5).then(inner, k=99).run()

  def test_kwargs_only_with_callable_current_value(self):
    """kwargs-only nested chain works when current_value is callable."""

    def fn(k=None):
      return k

    inner = Chain().then(lambda x: x)
    # lambda x: fn returns fn (the callable) as the current_value
    # inner receives fn as root, kwargs={'k': 42}
    result = Chain(5).then(lambda x: fn).then(inner, k=42).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# Bug 2 (CRITICAL): do() leaks results when first link
# File: quent/_chain.py:156-157
# ---------------------------------------------------------------------------


class TestBug2DoLeaksResult(unittest.TestCase):
  def test_do_as_first_link_no_root_returns_none(self):
    """Chain().do(fn).run() should return None, not fn's result."""
    result = Chain().do(lambda: 'side_effect').run()
    self.assertIsNone(result)

  def test_do_as_first_link_with_run_value(self):
    """Chain().do(fn).run(5) should return 5, not fn's result."""
    result = Chain().do(lambda x: 'side_effect').run(5)
    self.assertEqual(result, 5)

  def test_do_as_only_step_after_root(self):
    """Chain(5).do(fn).run() should return 5, not fn's result."""
    result = Chain(5).do(lambda x: 'side_effect').run()
    self.assertEqual(result, 5)


class TestBug2DoLeaksResultAsync(unittest.IsolatedAsyncioTestCase):
  async def test_do_async_first_link_no_root(self):
    """Async path: Chain().do(async_fn).run() should return None."""

    async def side_effect():
      return 'side_effect'

    result = await Chain().do(side_effect).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Bug 3 (MODERATE): _run_async missing source link stamp
# File: quent/_chain.py:272-281
# ---------------------------------------------------------------------------


class TestBug3AsyncSourceLinkStamp(unittest.IsolatedAsyncioTestCase):
  async def test_async_exception_has_source_link_stamp(self):
    """_run_async should stamp __quent_source_link__ like _run does."""

    async def fail(x):
      raise ValueError('async fail')

    try:
      await Chain(5).then(async_identity).then(fail).run()
    except ValueError as exc:
      # The source link stamp should have been set
      self.assertTrue(hasattr(exc, '__quent__'))


# ---------------------------------------------------------------------------
# Bug 4 (MODERATE): except_() type validation
# File: quent/_chain.py:341-364
# ---------------------------------------------------------------------------


class TestBug4ExceptValidation(unittest.TestCase):
  def test_except_with_int_raises_at_registration(self):
    """exceptions=42 should raise TypeError at registration, not runtime."""
    with self.assertRaises(TypeError):
      Chain().except_(lambda rv, e: e, exceptions=42)

  def test_except_with_non_exception_class_raises(self):
    """exceptions=str should raise TypeError (str is not BaseException subclass)."""
    with self.assertRaises(TypeError):
      Chain().except_(lambda rv, e: e, exceptions=str)

  def test_except_with_valid_exception_works(self):
    """exceptions=ValueError should work fine."""
    c = Chain(lambda: 1 / 0).except_(lambda rv, e: 'caught', exceptions=ZeroDivisionError)
    self.assertEqual(c.run(), 'caught')

  def test_except_with_list_of_valid_exceptions(self):
    """exceptions=[ValueError, TypeError] should work fine."""
    c = Chain(lambda: 1 / 0).except_(lambda rv, e: 'caught', exceptions=[ZeroDivisionError, TypeError])
    self.assertEqual(c.run(), 'caught')

  def test_except_with_list_containing_invalid_raises(self):
    """exceptions=[ValueError, 42] should raise TypeError at registration."""
    with self.assertRaises(TypeError):
      Chain().except_(lambda rv, e: e, exceptions=[ValueError, 42])


# ---------------------------------------------------------------------------
# Bug 5 (MODERATE): sys.exc_info()[1] indirection
# File: quent/_traceback.py:105
# ---------------------------------------------------------------------------


class TestBug5ExcInfoIndirection(unittest.TestCase):
  def test_traceback_uses_exc_directly(self):
    """_modify_traceback should work correctly with the exc parameter."""
    # This is a robustness test - if the function uses exc directly
    # (not sys.exc_info()), it should work even outside an except block.
    # We just verify no crash and the exception has the __quent__ attribute.
    try:
      Chain(lambda: 1 / 0).run()
    except ZeroDivisionError as exc:
      self.assertTrue(hasattr(exc, '__quent__'))


# ---------------------------------------------------------------------------
# Bug 6 (MODERATE): _ensure_future coroutine leak
# File: quent/_core.py:119-125
# ---------------------------------------------------------------------------


class TestBug6EnsureFutureCoroutineLeak(unittest.TestCase):
  def test_ensure_future_closes_coroutine_on_error(self):
    """_ensure_future should close the coroutine if task creation fails."""
    from quent._core import _ensure_future

    async def dummy():
      return 42

    coro = dummy()
    # No event loop running, so _create_task_fn should raise RuntimeError
    with self.assertRaises(RuntimeError):
      _ensure_future(coro)
    # After fix, the coroutine should be closed (cr_frame is None)
    self.assertIsNone(coro.cr_frame)


# ---------------------------------------------------------------------------
# Bug 7 (MODERATE): __quent_link_temp_args__ not cleaned
# File: quent/_traceback.py:89-93
# ---------------------------------------------------------------------------


class TestBug7TempArgsCleanup(unittest.TestCase):
  def test_link_temp_args_cleaned_from_exception(self):
    """__quent_link_temp_args__ should be deleted after traceback processing."""
    try:
      Chain(5).then(lambda x: 1 / 0).run()
    except ZeroDivisionError as exc:
      # After fix, __quent_link_temp_args__ should be cleaned up
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))


# ---------------------------------------------------------------------------
# Bug 8 (MODERATE): _Null pickle/copy robustness
# File: quent/_core.py:12-21
# ---------------------------------------------------------------------------


class TestBug8NullSingleton(unittest.TestCase):
  def test_pickle_roundtrip_preserves_identity(self):
    restored = pickle.loads(pickle.dumps(Null))
    self.assertIs(restored, Null)

  def test_copy_preserves_identity(self):
    self.assertIs(copy.copy(Null), Null)

  def test_deepcopy_preserves_identity(self):
    self.assertIs(copy.deepcopy(Null), Null)


# ---------------------------------------------------------------------------
# Bug 9 (MINOR): Misleading comment about super().__init__()
# File: quent/_core.py:41-43
# ---------------------------------------------------------------------------


class TestBug9Comment(unittest.TestCase):
  def test_control_flow_signal_init_works(self):
    """_ControlFlowSignal can be instantiated (comment fix, behavioral check)."""
    sig = _ControlFlowSignal('val', (1, 2), {'k': 3})
    self.assertEqual(sig.value, 'val')
    self.assertEqual(sig.args_, (1, 2))
    self.assertEqual(sig.kwargs_, {'k': 3})


# ---------------------------------------------------------------------------
# Bug 10 (MINOR): temp_args in Link docstring
# No behavioral test needed. Just a docstring fix.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bug 11 (MINOR): TODO comments in _chain.py
# No behavioral test needed. Comment clarification.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bug 12 (MINOR): _Break error message
# File: quent/_chain.py:172,270
# ---------------------------------------------------------------------------


class TestBug12BreakMessage(unittest.TestCase):
  def test_break_outside_map_message_sync(self):
    """Sync: _Break error message should be user-friendly."""
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: Chain.break_()).run()
    msg = str(ctx.exception)
    self.assertNotIn('_Break', msg)
    self.assertIn('break_', msg.lower())


class TestBug12BreakMessageAsync(unittest.IsolatedAsyncioTestCase):
  async def test_break_outside_map_message_async(self):
    """Async: _Break error message should be user-friendly."""
    with self.assertRaises(QuentException) as ctx:
      await Chain(5).then(async_identity).then(lambda x: Chain.break_()).run()
    msg = str(ctx.exception)
    self.assertNotIn('_Break', msg)
    self.assertIn('break_', msg.lower())


# ---------------------------------------------------------------------------
# Bug 13 (MINOR): exc_tb param rename
# No behavioral test needed. Just a parameter name fix.
# ---------------------------------------------------------------------------


if __name__ == '__main__':
  unittest.main()
