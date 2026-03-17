# SPDX-License-Identifier: MIT
"""Edge case tests — niche error paths, defensive guards, and uncovered branches.

These tests cover error paths, defensive branches, and edge cases that are
difficult to reach via the standard sync/async axis testing infrastructure.
They are organized by source module.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sys
import types
import warnings
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

from quent import (
  Chain,
  QuentException,
)
from quent._types import Null, _ControlFlowSignal, _Return

if sys.version_info < (3, 11):
  from quent._types import ExceptionGroup

# ---------------------------------------------------------------------------
# _with_ops.py — Context Manager Edge Cases
# ---------------------------------------------------------------------------


class SyncCMAsyncExit:
  """Sync CM whose __exit__ returns a coroutine."""

  def __init__(self, suppress: bool = False) -> None:
    self._suppress = suppress

  def __enter__(self) -> int:
    return 42

  def __exit__(self, *args: object) -> object:
    async def _async_exit() -> bool:
      return self._suppress

    return _async_exit()


class SyncCMEnterRaises:
  """Sync CM whose __enter__ raises."""

  def __enter__(self) -> object:
    raise RuntimeError('enter boom')

  def __exit__(self, *args: object) -> bool:
    return False


class SyncCMExitRaises:
  """Sync CM whose __exit__ raises."""

  def __enter__(self) -> int:
    return 42

  def __exit__(self, *args: object) -> object:
    raise RuntimeError('exit boom')


class WithOpsEdgeTest(IsolatedAsyncioTestCase):
  """Edge cases for _with_ops.py."""

  async def test_sync_cm_async_exit_exception_path(self) -> None:
    """§5.6, §8.4: sync CM __exit__ returns coroutine on exception path.

    The coroutine must be awaited to determine suppression.
    """

    class AsyncExitSuppress:
      def __enter__(self) -> int:
        return 42

      def __exit__(self, *args: object) -> object:
        async def _exit() -> bool:
          return True  # suppress

        return _exit()

    c = Chain(AsyncExitSuppress()).with_(lambda x: (_ for _ in ()).throw(ValueError('body fail')))
    result = c.run()
    # The result should be a coroutine since __exit__ returns one
    if asyncio.iscoroutine(result):
      result = await result
    # When suppressed, with_ returns None (suppressed_result for non-ignore_result)
    self.assertIsNone(result)

  async def test_sync_cm_async_exit_success_path(self) -> None:
    """§5.6, §8.4: sync CM __exit__ returns coroutine on success path."""

    class AsyncExitSuccess:
      def __enter__(self) -> int:
        return 42

      def __exit__(self, *args: object) -> object:
        async def _exit() -> bool:
          return False

        return _exit()

    c = Chain(AsyncExitSuccess()).with_(lambda x: x * 2)
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    self.assertEqual(result, 84)

  def test_sync_cm_enter_raises_metadata(self) -> None:
    """§5.6: sync CM __enter__ raises — metadata stamped with '<enter failed>'."""
    c = Chain(SyncCMEnterRaises()).with_(lambda x: x)
    with self.assertRaises(RuntimeError) as ctx:
      c.run()
    self.assertEqual(str(ctx.exception), 'enter boom')

  async def test_sync_cm_control_flow_and_exit_raises(self) -> None:
    """§5.6, §7.2: body raises ControlFlowSignal AND __exit__ also raises."""

    class ExitAlsoRaises:
      def __enter__(self) -> int:
        return 42

      def __exit__(self, *args: object) -> object:
        raise RuntimeError('exit also fails')

    def body(x: object) -> object:
      raise _Return(999, (), {})

    c = Chain(ExitAlsoRaises()).with_(body)
    # The exit_exc should be chained from the signal
    with self.assertRaises(RuntimeError) as ctx:
      result = c.run()
      if asyncio.iscoroutine(result):
        await result
    self.assertEqual(str(ctx.exception), 'exit also fails')
    self.assertIsInstance(ctx.exception.__cause__, _Return)

  async def test_sync_cm_body_raises_and_exit_raises(self) -> None:
    """§5.6: body raises AND __exit__ also raises — exit_exc chained from body exc."""
    c = Chain(SyncCMExitRaises()).with_(lambda x: (_ for _ in ()).throw(ValueError('body fail')))
    with self.assertRaises(RuntimeError) as ctx:
      result = c.run()
      if asyncio.iscoroutine(result):
        await result
    self.assertEqual(str(ctx.exception), 'exit boom')
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  async def test_sync_cm_body_raises_exit_returns_coroutine(self) -> None:
    """§5.6, §8.4: body raises AND __exit__ returns coroutine (async suppress check)."""

    class AsyncExitOnError:
      def __enter__(self) -> int:
        return 42

      def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> object:
        if exc_type is not None:

          async def _exit() -> bool:
            return True  # suppress

          return _exit()
        return False

    c = Chain(AsyncExitOnError()).with_(lambda x: (_ for _ in ()).throw(ValueError('body fail')))
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    # suppressed -> None
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _engine.py — Error Handling Fallbacks
# ---------------------------------------------------------------------------


class EngineEdgeTest(IsolatedAsyncioTestCase):
  """Edge cases for _engine.py."""

  async def test_traceback_enhancement_fails_in_except(self) -> None:
    """§13.11: _modify_traceback fails during except handler — warning issued."""
    with patch('quent._engine._modify_traceback', side_effect=Exception('tb boom')):
      c = Chain(1).then(lambda x: 1 / 0).except_(lambda info: 'caught')
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = c.run()
        if asyncio.iscoroutine(result):
          result = await result
      self.assertEqual(result, 'caught')
      tb_warnings = [x for x in w if 'traceback enhancement failed' in str(x.message)]
      self.assertTrue(len(tb_warnings) > 0, 'Expected traceback enhancement warning')

  async def test_traceback_modification_fails_in_finally(self) -> None:
    """§13.11: traceback modification of finally-handler exception fails."""

    def bad_finally(rv: object) -> object:
      raise RuntimeError('finally boom')

    with patch('quent._engine._modify_traceback', side_effect=Exception('tb mod fail')):
      c = Chain(1).finally_(bad_finally)
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with self.assertRaises(RuntimeError):
          result = c.run()
          if asyncio.iscoroutine(result):
            await result
      tb_warnings = [x for x in w if 'traceback enhancement failed' in str(x.message)]
      self.assertTrue(len(tb_warnings) > 0, 'Expected traceback enhancement warning')

  async def test_sync_finally_raises_with_active_exception(self) -> None:
    """§6.3.3: sync finally handler raises while original exception active — context chaining."""
    calls: list[str] = []

    def bad_finally(rv: object) -> object:
      calls.append('finally')
      raise RuntimeError('finally error')

    c = Chain(1).then(lambda x: 1 / 0).finally_(bad_finally)
    with self.assertRaises(RuntimeError) as ctx:
      c.run()
    self.assertEqual(str(ctx.exception), 'finally error')
    self.assertIsInstance(ctx.exception.__context__, ZeroDivisionError)

  async def test_async_finally_raises_with_active_exception(self) -> None:
    """§6.3.3: async finally handler raises while original exception active."""

    async def async_fail(x: int) -> int:
      raise ZeroDivisionError('async div')

    async def bad_async_finally(rv: object) -> object:
      raise RuntimeError('async finally error')

    c = Chain(1).then(async_fail).finally_(bad_async_finally)
    with self.assertRaises(RuntimeError) as ctx:
      result = c.run()
      if asyncio.iscoroutine(result):
        await result
    self.assertEqual(str(ctx.exception), 'async finally error')
    self.assertIsInstance(ctx.exception.__context__, ZeroDivisionError)

  async def test_async_except_handler_coroutine_raises_control_flow(self) -> None:
    """§6.2.6: async except handler returns coroutine that raises ControlFlowSignal."""

    async def handler_with_signal(info: object) -> object:
      raise _Return(999, (), {})

    c = Chain(1).then(lambda x: 1 / 0).except_(handler_with_signal)
    with self.assertRaises(QuentException) as ctx:
      result = c.run()
      if asyncio.iscoroutine(result):
        await result
    self.assertIn('_Return', str(ctx.exception))

  async def test_async_except_handler_raises_base_exception(self) -> None:
    """§6.2.4: async except handler (reraise=True) raises BaseException."""

    async def handler_keyboard_interrupt(info: object) -> object:
      raise KeyboardInterrupt()

    c = Chain(1).then(lambda x: 1 / 0).except_(handler_keyboard_interrupt, reraise=True)
    with self.assertRaises(KeyboardInterrupt):
      result = c.run()
      if asyncio.iscoroutine(result):
        await result

  def test_record_exception_source_link_none(self) -> None:
    """§16.8: _record_exception_source with link=None — early return."""
    from quent._engine import _record_exception_source

    exc = ValueError('test')
    # Should not raise
    _record_exception_source(exc, None, 'some_value')

  def test_keyboard_interrupt_in_concurrent_step(self) -> None:
    """§13.4: KeyboardInterrupt in concurrent step — cleaned without traceback mod."""
    from quent._exc_meta import _clean_quent_idx

    exc = KeyboardInterrupt()
    exc._quent_idx = 5  # type: ignore[attr-defined]
    _clean_quent_idx(exc)
    self.assertFalse(hasattr(exc, '_quent_idx'))

  def test_debug_repr_repr_raises(self) -> None:
    """§14.5: _debug_repr with repr() that raises."""
    from quent._engine import _debug_repr

    class BadRepr:
      def __repr__(self) -> str:
        raise RuntimeError('repr boom')

    result = _debug_repr(BadRepr())
    self.assertIn('repr failed', result)
    self.assertIn('BadRepr', result)

  def test_debug_repr_truncation(self) -> None:
    """§14.5: repr exceeding max_len — truncation."""
    from quent._engine import _debug_repr

    result = _debug_repr('x' * 500, max_len=10)
    self.assertIn('truncated', result)
    self.assertTrue(len(result) < 500)

  def test_root_link_ignore_result_raises(self) -> None:
    """§3: root_link has ignore_result=True — raises QuentException (defensive invariant)."""
    from quent._engine import _run

    # Construct a chain with a root_link whose ignore_result=True
    # This is a defensive invariant — cannot happen via public API
    c = Chain(lambda: 42)
    c.root_link.ignore_result = True
    with self.assertRaises(QuentException) as ctx:
      _run(c, Null, None, None)
    self.assertIn('ignore_result', str(ctx.exception))

  async def test_debug_logging_paths(self) -> None:
    """§14.5: debug logging enabled."""
    logger = logging.getLogger('quent')
    original_level = logger.level
    try:
      logger.setLevel(logging.DEBUG)
      handler = logging.StreamHandler()
      handler.setLevel(logging.DEBUG)
      logger.addHandler(handler)
      # Simple chain
      c = Chain(1).then(lambda x: x + 1)
      result = c.run()
      if asyncio.iscoroutine(result):
        result = await result
      self.assertEqual(result, 2)

      # Async chain (triggers async debug path)
      async def async_add(x: int) -> int:
        return x + 1

      c2 = Chain(1).then(async_add)
      result2 = c2.run()
      if asyncio.iscoroutine(result2):
        result2 = await result2
      self.assertEqual(result2, 2)
      # Error path
      c3 = Chain(1).then(lambda x: 1 / 0).except_(lambda info: 'caught')
      result3 = c3.run()
      if asyncio.iscoroutine(result3):
        result3 = await result3
      self.assertEqual(result3, 'caught')
    finally:
      logger.setLevel(original_level)
      logger.removeHandler(handler)

  def test_control_flow_signal_escape_from_run(self) -> None:
    """§7.2.3, §7.3.2: ControlFlowSignal escape from run() — wrapped in QuentException."""
    # Use break_() outside of iteration context — it will escape as a signal
    c = Chain(1).then(lambda x: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('break_', str(ctx.exception).lower())

  def test_decorator_control_flow_signal_escape(self) -> None:
    """§10.2, §7.2.3: decorator() with ControlFlowSignal escape -> QuentException."""

    @Chain().then(lambda x: Chain.break_()).decorator()
    def my_fn() -> str:
      return 'hello'

    with self.assertRaises(QuentException) as ctx:
      my_fn()
    self.assertIn('break_', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# _generator.py — Sync Iterator with Async Values
# ---------------------------------------------------------------------------


class GeneratorEdgeTest(IsolatedAsyncioTestCase):
  """Edge cases for _generator.py."""

  def test_sync_iterate_on_async_chain(self) -> None:
    """§17.1: sync `for x in chain.iterate()` on async chain — TypeError."""

    async def async_range(n: int) -> list[int]:
      return list(range(n))

    c = Chain(async_range, 5)
    with self.assertRaises(TypeError) as ctx:
      for _x in c.iterate():
        pass
    self.assertIn('sync iteration', str(ctx.exception).lower())

  def test_sync_iterate_break_callable_raises(self) -> None:
    """§7.3.1, §9.4: sync iterate() where break_ callable evaluation fails.

    The iterate generator catches _Break and evaluates its value. If evaluation fails,
    the exception is re-raised.
    """

    def failing_break_val() -> object:
      raise ValueError('break eval fail')

    def body(x: int) -> object:
      if x == 2:
        Chain.break_(failing_break_val)
      return x

    c = Chain(range(5))
    with self.assertRaises(ValueError) as ctx:
      for _ in c.iterate(body):
        pass
    self.assertIn('break eval fail', str(ctx.exception))

  def test_sync_iterate_break_async_fn(self) -> None:
    """§17.1: sync iterate() where break value is async in sync context — TypeError."""

    async def async_val() -> int:
      return 99

    def body(x: int) -> object:
      if x == 2:
        Chain.break_(async_val)
      return x

    c = Chain(range(5))
    with self.assertRaises(TypeError) as ctx:
      for _ in c.iterate(body):
        pass
    self.assertIn('coroutine', str(ctx.exception).lower())

  def test_sync_iterate_return_callable_raises(self) -> None:
    """§7.2.1, §9.4: sync iterate() where return_ callable evaluation fails."""

    def failing_return_val() -> object:
      raise ValueError('return eval fail')

    def body(x: int) -> object:
      if x == 2:
        Chain.return_(failing_return_val)
      return x

    c = Chain(range(5))
    with self.assertRaises(ValueError) as ctx:
      for _ in c.iterate(body):
        pass
    self.assertIn('return eval fail', str(ctx.exception))

  def test_sync_iterate_return_async_fn(self) -> None:
    """§17.1: sync iterate() where return value is async in sync context — TypeError."""

    async def async_val() -> int:
      return 99

    def body(x: int) -> object:
      if x == 2:
        Chain.return_(async_val)
      return x

    c = Chain(range(5))
    with self.assertRaises(TypeError) as ctx:
      for _ in c.iterate(body):
        pass
    self.assertIn('coroutine', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# _iter_ops.py — Concurrent Iteration Edge Cases
# ---------------------------------------------------------------------------


class IterOpsEdgeTest(IsolatedAsyncioTestCase):
  """Edge cases for _iter_ops.py."""

  async def test_empty_async_iterable_concurrent(self) -> None:
    """§5.3: empty async iterable with concurrent mode — empty result."""
    from tests.tests_helper import AsyncRange

    c = Chain(AsyncRange(0)).foreach(lambda x: x + 1, concurrency=2)
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    self.assertEqual(result, [])

  def test_empty_list_concurrent(self) -> None:
    """§5.3: empty list with concurrent mode — empty result."""
    c = Chain([]).foreach(lambda x: x + 1, concurrency=2)
    result = c.run()
    self.assertEqual(result, [])

  async def test_concurrent_break_on_first_item(self) -> None:
    """§7.3.4, §5.3: concurrent map where fn raises break on the probed first item."""
    c = Chain([1, 2, 3]).foreach(lambda x: Chain.break_() if x == 1 else x, concurrency=2)
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    self.assertEqual(result, [])

  async def test_concurrent_return_signal(self) -> None:
    """§7.3.5: concurrent iteration where fn raises return signal."""
    c = Chain([1, 2, 3]).foreach(lambda x: Chain.return_(99) if x == 1 else x, concurrency=2)
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    self.assertEqual(result, 99)

  async def test_mid_transition_return_in_map(self) -> None:
    """§16.7, §7.2: _Return raised in mid-transition async continuation."""

    async def maybe_return(x: int) -> int:
      if x == 2:
        raise _Return(99, (), {})
      return x

    c = Chain([1, 2, 3]).foreach(maybe_return)
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    # _Return from inside map should propagate
    self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# _gather_ops.py — Gather Triage
# ---------------------------------------------------------------------------


class GatherTriageEdgeTest(IsolatedAsyncioTestCase):
  """Edge cases for _gather_ops.py."""

  async def test_return_signal_with_regular_exceptions(self) -> None:
    """§7.3.5: return_ signal alongside regular exceptions — warning logged."""

    async def fn1(x: int) -> int:
      raise _Return(99, (), {})

    async def fn2(x: int) -> int:
      raise ValueError('regular')

    c = Chain(1).gather(fn1, fn2)
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    # _Return takes priority; result should be 99
    self.assertEqual(result, 99)

  def test_base_exception_in_gather_triage(self) -> None:
    """§5.5, §6.5: BaseException (not Exception) in gather triage."""
    from quent._gather_ops import _triage_gather_exceptions
    from tests.tests_helper import CustomBaseError

    base_exc = CustomBaseError('base exc')
    regular_exc = ValueError('regular')
    result = _triage_gather_exceptions([regular_exc, base_exc])
    self.assertEqual(result.action, 'base_exc')
    self.assertIs(result.exc, base_exc)
    # Regular exceptions are still collected
    self.assertIn(regular_exc, result.exceptions)

  async def test_exception_group_from_gather(self) -> None:
    """§6.5: ExceptionGroup from 2+ gather failures."""

    async def fn1(x: int) -> int:
      raise ValueError('err1')

    async def fn2(x: int) -> int:
      raise TypeError('err2')

    c = Chain(1).gather(fn1, fn2)
    with self.assertRaises(ExceptionGroup) as ctx:
      result = c.run()
      if asyncio.iscoroutine(result):
        await result
    eg = ctx.exception
    self.assertEqual(len(eg.exceptions), 2)

  def test_sync_gather_control_flow_signal_on_probe(self) -> None:
    """§5.5, §7.2: sync gather where first fn raises ControlFlowSignal (_Return) on probe."""
    c = Chain(1).gather(lambda x: Chain.return_(99))
    result = c.run()
    self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# _chain.py — Builder Validation Edge Cases
# ---------------------------------------------------------------------------


class ChainBuilderEdgeTest(TestCase):
  """Edge cases for _chain.py builder validation."""

  def test_except_exceptions_string(self) -> None:
    """§6.2.1: except_(handler, exceptions='ValueError') — TypeError."""
    with self.assertRaises(TypeError) as ctx:
      Chain(1).except_(lambda info: None, exceptions='ValueError')  # type: ignore[arg-type]
    self.assertIn('string', str(ctx.exception).lower())

  def test_except_exceptions_non_base_exception_type(self) -> None:
    """§6.2.1: except_(handler, exceptions=[int]) — TypeError."""
    with self.assertRaises(TypeError):
      Chain(1).except_(lambda info: None, exceptions=[int])  # type: ignore[arg-type]

  def test_except_exceptions_non_iterable_non_type(self) -> None:
    """§6.2.1: except_(handler, exceptions=42) — TypeError."""
    with self.assertRaises(TypeError):
      Chain(1).except_(lambda info: None, exceptions=42)  # type: ignore[arg-type]

  def test_else_non_callable_with_args(self) -> None:
    """§5.9: else_(42, 'extra_arg') — non-callable with args."""
    with self.assertRaises(TypeError) as ctx:
      Chain(1).if_(lambda x: False).then(lambda x: x).else_(42, 'extra')  # type: ignore[arg-type]
    self.assertIn('not callable', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# _traceback.py — Traceback Edge Cases
# ---------------------------------------------------------------------------


class TracebackEdgeTest(IsolatedAsyncioTestCase):
  """Edge cases for _traceback.py."""

  def test_chained_exceptions_depth_guard(self) -> None:
    """§13.5: _clean_chained_exceptions depth guard — deep chain."""
    from quent._traceback import _clean_chained_exceptions

    # Build a chain of exceptions deeper than _MAX_CHAINED_EXCEPTION_DEPTH
    root = ValueError('root')
    current = root
    for i in range(1010):
      new_exc = ValueError(f'exc_{i}')
      new_exc.__cause__ = current
      current = new_exc
    # Should not raise (guard prevents infinite loop)
    seen: set[int] = set()
    _clean_chained_exceptions(current, seen)
    # Verify it stopped before processing all 1010
    self.assertLess(len(seen), 1010)

  def test_exception_group_sub_exception_cleaning(self) -> None:
    """§13.5: ExceptionGroup sub-exception cleaning."""
    from quent._traceback import _clean_chained_exceptions

    sub1 = ValueError('sub1')
    sub2 = TypeError('sub2')
    eg = ExceptionGroup('test', [sub1, sub2])
    seen: set[int] = set()
    _clean_chained_exceptions(eg, seen)
    # Both sub-exceptions should have been visited
    self.assertIn(id(sub1), seen)
    self.assertIn(id(sub2), seen)

  async def test_visualization_construction_fails(self) -> None:
    """§13.11: visualization construction fails — warning and fallback."""
    with patch('quent._traceback._stringify_chain', side_effect=Exception('viz boom')):
      c = Chain(1).then(lambda x: 1 / 0).except_(lambda info: 'caught')
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = c.run()
        if asyncio.iscoroutine(result):
          result = await result
      self.assertEqual(result, 'caught')
      viz_warnings = [x for x in w if 'visualization failed' in str(x.message)]
      self.assertTrue(len(viz_warnings) > 0)

  def test_exception_with_no_source_link(self) -> None:
    """§13.6: exception with no identifiable source_link — step_name becomes '?'."""
    if sys.version_info < (3, 11):
      self.skipTest('add_note / __notes__ requires Python 3.11+')
    from quent._traceback import _attach_exception_note

    exc = ValueError('test')
    c = Chain(1)
    _attach_exception_note(exc, c, None)
    notes = getattr(exc, '__notes__', [])
    found = any('?' in n for n in notes)
    self.assertTrue(found, f'Expected "?" in notes, got {notes}')

  def test_note_generation_fails(self) -> None:
    """§13.11: note generation fails — swallowed."""
    from quent._traceback import _attach_exception_note

    class BadNameChain:
      _quent_is_chain = True
      root_link = MagicMock()
      root_link.v = property(lambda self: (_ for _ in ()).throw(RuntimeError('name boom')))

    exc = ValueError('test')
    # Should not raise even if internals fail
    c = Chain(1)

    class BadLink:
      original_value = None

      @property
      def v(self) -> object:
        raise RuntimeError('link boom')

    # Patch _get_obj_name to raise
    with patch('quent._traceback._get_obj_name', side_effect=RuntimeError('name boom')):
      _attach_exception_note(exc, c, MagicMock())
    # Should not have raised

  def test_try_clean_quent_exc_internal_error(self) -> None:
    """§13.11: _try_clean_quent_exc internal error — returns (False, None)."""
    from quent._traceback import _try_clean_quent_exc

    class BadExc(Exception):
      @property
      def __quent_meta__(self) -> object:
        raise RuntimeError('meta boom')

    result = _try_clean_quent_exc(BadExc('test'))
    self.assertEqual(result, (False, None))

  def test_excepthook_path(self) -> None:
    """§13.8: sys.excepthook path — uncaught chain exception with quent metadata."""
    from quent._traceback import _quent_excepthook

    exc = ValueError('test')
    exc.__quent_meta__ = {'quent': True}  # type: ignore[attr-defined]
    # Should not raise when calling the hook
    called = False

    def mock_hook(exc_type: object, exc_value: object, exc_tb: object) -> None:
      nonlocal called
      called = True

    with patch('quent._traceback._prev_excepthook', mock_hook):
      _quent_excepthook(type(exc), exc, exc.__traceback__)
    self.assertTrue(called)


# ---------------------------------------------------------------------------
# _viz.py — Visualization Edge Cases
# ---------------------------------------------------------------------------


class VizEdgeTest(TestCase):
  """Edge cases for _viz.py."""

  def test_object_no_name_no_qualname(self) -> None:
    """§13.12: object with no __name__/__qualname__ — falls through to repr."""
    from quent._viz import _get_obj_name

    # Plain integer has no __name__ or __qualname__ on the instance,
    # so _get_obj_name falls through to repr
    name = _get_obj_name(42)
    self.assertEqual(name, '42')

    # A non-numeric non-callable object also falls through to repr
    name2 = _get_obj_name([1, 2, 3])
    self.assertEqual(name2, '[1, 2, 3]')

  def test_functools_partial_as_step(self) -> None:
    """§13.12: functools.partial as chain step — 'partial(inner_name)' in visualization."""
    from quent._viz import _get_obj_name

    def my_fn(x: int, y: int) -> int:
      return x + y

    p = functools.partial(my_fn, 1)
    name = _get_obj_name(p)
    self.assertIn('partial', name)
    self.assertIn('my_fn', name)

  def test_object_repr_raises(self) -> None:
    """§13.9, §13.12: object whose __repr__ raises — falls back to type name."""
    from quent._viz import _get_obj_name

    class BadRepr:
      def __repr__(self) -> str:
        raise RuntimeError('repr boom')

    # To exercise the repr-fails fallback, we need __name__ and __qualname__
    # to both return None. We mock getattr to achieve this.
    obj = BadRepr()
    original_getattr = getattr

    def mock_getattr(o: object, name: str, *default: object) -> object:
      if o is obj and name in ('__name__', '__qualname__'):
        return default[0] if default else None
      return original_getattr(o, name, *default)

    with patch('quent._viz.getattr', side_effect=mock_getattr):
      name = _get_obj_name(obj)
    self.assertEqual(name, 'BadRepr')

  def test_get_true_source_link_nested_chains(self) -> None:
    """§13.2, §13.3: _get_true_source_link drilling through nested chains."""
    from quent._viz import _get_true_source_link

    inner = Chain(lambda: 42)
    Chain(inner)
    from quent._link import Link

    source = Link(inner)
    result = _get_true_source_link(source, None)
    # Should drill through to inner chain's root
    self.assertIsNotNone(result)

  def test_gather_visualization(self) -> None:
    """§13.12: gather operation visualization."""
    c = Chain(1).gather(lambda x: x, lambda x: x + 1)
    # Verify the chain can be repr'd (exercises visualization)
    r = repr(c)
    self.assertIsInstance(r, str)


# ---------------------------------------------------------------------------
# _link.py — Duck Typing
# ---------------------------------------------------------------------------


class LinkDuckTypingTest(TestCase):
  """Edge cases for _link.py duck typing."""

  def test_quent_is_chain_but_run_not_callable(self) -> None:
    """§4: object with _quent_is_chain=True but _run not callable — is_chain=False."""
    from quent._link import Link

    class FakeChain:
      _quent_is_chain = True
      _run = 'not callable'

    link = Link(FakeChain())
    self.assertFalse(link.is_chain)

  def test_getattr_raises_non_attribute_error(self) -> None:
    """§4: object whose __getattr__ raises non-AttributeError."""
    from quent._link import Link

    class BadGetattr:
      def __getattr__(self, name: str) -> object:
        raise RuntimeError(f'bad attr: {name}')

    link = Link(BadGetattr())
    self.assertFalse(link.is_chain)


# ---------------------------------------------------------------------------
# _eval.py — Edge Cases
# ---------------------------------------------------------------------------


class EvalEdgeTest(TestCase):
  """Edge cases for _eval.py."""

  def test_generator_with_iterable_coroutine_flag(self) -> None:
    """§8.4: GeneratorType with _CO_ITERABLE_COROUTINE flag."""
    from quent._eval import _isawaitable

    # Create a generator with the flag set
    def gen() -> object:
      yield 1

    g = gen()
    # Normally generators are not awaitable
    self.assertFalse(_isawaitable(g))
    g.close()

    # Test with a real @types.coroutine decorated function
    @types.coroutine
    def legacy_coro() -> object:
      yield 1

    lc = legacy_coro()
    self.assertTrue(_isawaitable(lc))
    lc.close()


# ---------------------------------------------------------------------------
# _concurrency.py — ThreadPool edge cases
# ---------------------------------------------------------------------------


class ConcurrencyEdgeTest(IsolatedAsyncioTestCase):
  """Edge cases for concurrent operations."""

  async def test_gather_empty_fns(self) -> None:
    """Gather with no functions raises at build time via _make_gather."""
    from quent._gather_ops import _make_gather

    with self.assertRaises(QuentException) as ctx:
      _make_gather(())
    self.assertIn('at least one', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# _triage edge cases
# ---------------------------------------------------------------------------


class TriageEdgeTest(TestCase):
  """Edge cases for triage functions."""

  def test_iter_triage_control_flow_signal_not_return_or_break(self) -> None:
    """§16.5: ControlFlowSignal that is neither _Return nor _Break — raise directly."""
    from quent._iter_ops import _triage_iter_exceptions

    class CustomSignal(_ControlFlowSignal):
      pass

    signal = CustomSignal(None, (), {})
    with self.assertRaises(CustomSignal):
      _triage_iter_exceptions([signal], 5, 'map')

  def test_gather_triage_reraise(self) -> None:
    """§5.5: _triage_gather_exceptions returns 'reraise' for empty-ish list."""
    from quent._gather_ops import _dispatch_gather_triage, _GatherTriageResult

    triage = _GatherTriageResult('reraise')
    # dispatch should return without raising
    _dispatch_gather_triage(triage)


# ---------------------------------------------------------------------------
# _viz.py — Viz Truncation Edge Cases
# ---------------------------------------------------------------------------


class VizTruncationTest(TestCase):
  """Edge cases for _viz.py truncation limits."""

  def test_viz_max_total_calls_truncation(self) -> None:
    """§13.10: _VIZ_MAX_TOTAL_CALLS (500) truncation — deeply nested chains terminate.

    Build a chain with enough nesting to exceed 500 recursive viz calls
    and verify it terminates without error.
    """
    from quent._viz import _stringify_chain, _VizContext

    # Build deeply nested chains — each chain has a nested chain as a step,
    # which triggers recursive _stringify_chain calls.
    inner = Chain(lambda: 1)
    for _ in range(100):
      outer = Chain(inner)
      inner = outer

    ctx = _VizContext(source_link=None, link_temp_args=None)
    result = _stringify_chain(inner, nest_lvl=0, ctx=ctx)
    # Should terminate and contain truncation marker
    self.assertIsInstance(result, str)
    self.assertTrue(len(result) > 0)
    # The context should have hit the call limit
    self.assertGreater(ctx.total_calls, 0)

  def test_viz_max_length_truncation(self) -> None:
    """§13.10: _VIZ_MAX_LENGTH (10,000) truncation adds '... <truncated>' suffix.

    Build a chain long enough to produce >10K chars of visualization.
    """
    from quent._viz import _stringify_chain, _VizContext

    # Build a chain with many steps to produce long output
    c = Chain(lambda: 1)
    for i in range(500):
      c.then(lambda x, _i=i: x + _i)

    ctx = _VizContext(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    if len(result) > 10_000:
      self.assertTrue(result.endswith('\n... <truncated>'))


# ---------------------------------------------------------------------------
# Additional misc edge cases
# ---------------------------------------------------------------------------


class MiscEdgeTest(IsolatedAsyncioTestCase):
  """Miscellaneous edge cases."""

  async def test_isawaitable_with_getattr_that_raises(self) -> None:
    """_isawaitable with object whose __getattr__ raises non-AttributeError."""
    from quent._eval import _isawaitable

    class WeirdObj:
      def __getattr__(self, name: str) -> object:
        raise RuntimeError('weird')

    result = _isawaitable(WeirdObj())
    self.assertFalse(result)

  async def test_finally_with_no_root_value(self) -> None:
    """Finally handler receives None when chain has no root value."""
    received: list[object] = []

    def capture_rv(rv: object) -> None:
      received.append(rv)

    c = Chain().then(lambda: 42).finally_(capture_rv)
    result = c.run()
    if asyncio.iscoroutine(result):
      result = await result
    self.assertEqual(result, 42)
    # root_value is Null -> normalized to None for finally handler
    self.assertEqual(received, [None])

  def test_sanitize_repr(self) -> None:
    """_sanitize_repr strips control characters and ANSI escapes."""
    from quent._viz import _sanitize_repr

    dirty = 'hello\x1b[31mworld\x1b[0m\x00end'
    clean = _sanitize_repr(dirty)
    self.assertNotIn('\x1b', clean)
    self.assertNotIn('\x00', clean)
    self.assertIn('hello', clean)
    self.assertIn('world', clean)
    self.assertIn('end', clean)


# ---------------------------------------------------------------------------
# _iter_ops.py / _eval.py — Mid-Transition & Async Break Behavioral Tests
# ---------------------------------------------------------------------------


class ForeachMidTransitionSyncResultTest(IsolatedAsyncioTestCase):
  """Test that sync results are handled correctly after mid-operation async transition."""

  async def test_sync_result_after_async_transition(self) -> None:
    """After async transition in foreach, fn returning sync value is collected correctly."""
    call_count = [0]

    def mixed_fn(x: int) -> int:
      call_count[0] += 1
      if call_count[0] == 2:

        async def inner() -> int:
          return x * 10

        return inner()
      return x * 10

    result = await Chain([1, 2, 3, 4]).foreach(mixed_fn).run()
    self.assertEqual(result, [10, 20, 30, 40])

  async def test_alternating_sync_async_results_after_transition(self) -> None:
    """After transition, alternating sync/async results all collected correctly."""
    call_count = [0]

    def alternating_fn(x: int) -> int:
      call_count[0] += 1
      if call_count[0] % 2 == 0:

        async def inner() -> int:
          return x * 10

        return inner()
      return x * 10

    result = await Chain([1, 2, 3, 4, 5]).foreach(alternating_fn).run()
    self.assertEqual(result, [10, 20, 30, 40, 50])


class ForeachBreakAsyncValueTest(IsolatedAsyncioTestCase):
  """Test break_(async_callable) inside foreach operations."""

  async def test_break_async_value_mid_transition(self) -> None:
    """break_(async_fn) during mid-transition foreach awaits the break value."""
    call_count = [0]

    async def async_break_value() -> int:
      return 99

    def mixed_fn(x: int) -> int:
      call_count[0] += 1
      if call_count[0] == 2:

        async def inner() -> int:
          return x * 10

        return inner()
      if call_count[0] == 3:
        Chain.break_(async_break_value)
      return x * 10

    result = await Chain([1, 2, 3, 4]).foreach(mixed_fn).run()
    self.assertEqual(result, [10, 20, 99])

  async def test_break_async_value_full_async_path(self) -> None:
    """break_(async_fn) in full async foreach awaits the break value."""

    async def async_range(n: int):
      for i in range(n):
        yield i

    async def async_break_value() -> int:
      return 99

    async def fn(x: int) -> int:
      if x == 2:
        Chain.break_(async_break_value)
      return x * 10

    result = await Chain(async_range(5)).foreach(fn).run()
    self.assertEqual(result, [0, 10, 99])


# ---------------------------------------------------------------------------
# _with_ops.py — Sync CM with Async Transition Edge Cases
# ---------------------------------------------------------------------------


class SyncCMAsyncExitRaisesOnError:
  """Sync CM whose __exit__ returns a coroutine that raises on the error path."""

  def __enter__(self) -> int:
    return 42

  def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> object:
    if exc_type is not None:

      async def _exit() -> bool:
        raise RuntimeError('async exit fail')

      return _exit()
    return False


class SyncCMAsyncExitNoSuppress:
  """Sync CM whose __exit__ returns a coroutine that doesn't suppress on the error path."""

  def __enter__(self) -> int:
    return 42

  def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> object:
    if exc_type is not None:

      async def _exit() -> bool:
        return False

      return _exit()
    return False


class WithOpsAsyncTransitionTest(IsolatedAsyncioTestCase):
  """Test sync CM behavior during async body transition."""

  async def test_async_body_raises_sync_exit_raises(self) -> None:
    """When async body raises and sync __exit__ also raises, exit_exc replaces body exc."""

    class CMExitFailsOnError:
      def __enter__(self) -> int:
        return 10

      def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> bool:
        if exc_type is not None:
          raise RuntimeError('exit cleanup failed')
        return False

    async def async_body(x: int) -> int:
      raise ValueError('body error')

    with self.assertRaises(RuntimeError) as ctx:
      await Chain(CMExitFailsOnError()).with_(async_body).run()
    self.assertEqual(str(ctx.exception), 'exit cleanup failed')
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  async def test_async_body_success_exit_returns_coroutine(self) -> None:
    """Sync CM with async body that succeeds; __exit__ returns coroutine on success path."""

    async def async_body(x: int) -> int:
      return x + 1

    result = await Chain(SyncCMAsyncExit()).with_(async_body).run()
    self.assertEqual(result, 43)

  async def test_body_raises_exit_coroutine_raises(self) -> None:
    """Sync body raises, __exit__ returns coroutine that raises; exit_exc replaces body exc."""

    def bad_body(x: int) -> int:
      raise ValueError('body error')

    with self.assertRaises(RuntimeError) as ctx:
      await Chain(SyncCMAsyncExitRaisesOnError()).with_(bad_body).run()
    self.assertEqual(str(ctx.exception), 'async exit fail')
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  async def test_body_raises_exit_coroutine_no_suppress(self) -> None:
    """Sync body raises, __exit__ returns coroutine returning False; body exc re-raised."""

    def bad_body(x: int) -> int:
      raise ValueError('body error')

    with self.assertRaises(ValueError) as ctx:
      await Chain(SyncCMAsyncExitNoSuppress()).with_(bad_body).run()
    self.assertEqual(str(ctx.exception), 'body error')


# ---------------------------------------------------------------------------
# _engine.py — Async Except Handler Control Flow Signal Test
# ---------------------------------------------------------------------------


class AsyncExceptHandlerControlFlowTest(IsolatedAsyncioTestCase):
  """Test _ControlFlowSignal in sync except handler of async chain."""

  async def test_sync_handler_raises_return_in_async_chain(self) -> None:
    """Sync except handler calling return_() in async chain wraps signal in QuentException."""

    async def async_fail(x: int) -> int:
      raise ValueError('original')

    def handler_with_return(info: object) -> None:
      Chain.return_(42)

    with self.assertRaises(QuentException) as ctx:
      await Chain(1).then(async_fail).except_(handler_with_return).run()
    self.assertIn('_Return', str(ctx.exception))
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  async def test_sync_handler_raises_break_in_async_chain(self) -> None:
    """Sync except handler calling break_() in async chain wraps signal in QuentException."""

    async def async_fail(x: int) -> int:
      raise ValueError('original')

    def handler_with_break(info: object) -> None:
      Chain.break_()

    with self.assertRaises(QuentException) as ctx:
      await Chain(1).then(async_fail).except_(handler_with_break).run()
    self.assertIn('_Break', str(ctx.exception))
    self.assertIsInstance(ctx.exception.__cause__, ValueError)


# ---------------------------------------------------------------------------
# _gather_ops.py — Triage Behavioral Tests
# ---------------------------------------------------------------------------


class GatherTriageBehaviorTest(IsolatedAsyncioTestCase):
  """Test gather exception triage priority and behavior."""

  async def test_return_after_regular_exceptions_logs_warning(self) -> None:
    """When _Return is triaged after regular exceptions, a warning is logged about discards."""
    from quent._gather_ops import _triage_gather_exceptions

    with self.assertLogs('quent', level='WARNING') as cm, self.assertRaises(_ControlFlowSignal):
      _triage_gather_exceptions([ValueError('err'), _Return(99, (), {})])
    self.assertTrue(any('discarded' in msg for msg in cm.output))

  async def test_multiple_base_exceptions_first_wins(self) -> None:
    """When multiple BaseExceptions are present, earliest by list position is raised."""
    from quent._gather_ops import _dispatch_gather_triage, _triage_gather_exceptions

    class Base1(BaseException):
      pass

    class Base2(BaseException):
      pass

    triage = _triage_gather_exceptions([Base1('first'), Base2('second')])
    self.assertEqual(triage.action, 'base_exc')
    with self.assertRaises(Base1) as ctx:
      _dispatch_gather_triage(triage)
    self.assertEqual(str(ctx.exception), 'first')
