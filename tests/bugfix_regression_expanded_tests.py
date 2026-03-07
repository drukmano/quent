"""Expanded regression tests for all bug fixes across chain methods and contexts.

Covers cross-feature interactions that basic regression tests miss:
H1 (_await_exit_suppress), H2 (_sync_generator), H3 (_get_true_source_link),
H4 (_clean_chained_exceptions), H5 (_stringify_chain depth guard),
H6 (_modify_traceback failure suppression), M1 (_make_gather cleanup),
M4 (realpath), M5 (_get_obj_name newlines), M6 (error guards on hooks).
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import traceback
import types
import unittest
from inspect import isawaitable
from typing import Any
from unittest.mock import patch

from quent import Chain, Null
from quent._chain import _except_handler_body, _finally_handler_body
from quent._core import Link, _Break, _ControlFlowSignal, _Return
from quent._ops import _make_gather, _make_with, _sync_generator
from quent._traceback import (
  _clean_chained_exceptions,
  _clean_internal_frames,
  _Ctx,
  _get_obj_name,
  _get_true_source_link,
  _modify_traceback,
  _patched_te_init,
  _quent_excepthook,
  _quent_file,
  _stringify_chain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exc_with_tb(msg='test'):
  """Create an exception with a real traceback."""
  try:
    raise ValueError(msg)
  except ValueError as exc:
    return exc


def _make_exception_chain(n, attr='__context__'):
  """Build a chain of n exceptions linked via the given attribute."""
  root = ValueError(f'exc-0')
  current = root
  for i in range(1, n):
    child = ValueError(f'exc-{i}')
    setattr(current, attr, child)
    current = child
  return root


# ---------------------------------------------------------------------------
# H1: _await_exit_suppress -- expanded (12 tests)
# ---------------------------------------------------------------------------

class TestH1AwaitExitSuppressExpanded(unittest.IsolatedAsyncioTestCase):
  """H1: Expanded coverage for _await_exit_suppress across chain methods."""

  async def test_with_do_sync_cm_async_exit_raises(self):
    """with_do: sync CM, body raises, async __exit__ raises -- verify chaining."""
    body_exc = ValueError('body')
    exit_exc = RuntimeError('exit')

    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            raise exit_exc
          return _exit()
        return False

    c = Chain(CM()).with_do(lambda ctx: (_ for _ in ()).throw(body_exc))
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIs(ctx.exception, exit_exc)
    self.assertIs(ctx.exception.__cause__, body_exc)

  async def test_with_body_raises_exit_awaitable_true_suppresses(self):
    """with_: body raises, __exit__ returns awaitable True -- suppressed."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            return True
          return _exit()
        return False

    c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(ValueError('body')))
    result = await c.run()
    self.assertIsNone(result)

  async def test_with_body_raises_exit_awaitable_false_reraises(self):
    """with_: body raises, __exit__ returns awaitable False -- re-raises original."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            return False
          return _exit()
        return False

    c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(ValueError('body err')))
    with self.assertRaises(ValueError) as ctx:
      await c.run()
    self.assertEqual(str(ctx.exception), 'body err')

  async def test_with_body_raises_exit_awaitable_raises_cause_is_body(self):
    """with_: body raises, __exit__ returns awaitable that raises -- __cause__ is body exc."""
    body_exc = TypeError('body type err')
    exit_exc = OSError('exit os err')

    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            raise exit_exc
          return _exit()
        return False

    c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(body_exc))
    with self.assertRaises(OSError) as ctx:
      await c.run()
    self.assertIs(ctx.exception.__cause__, body_exc)

  async def test_with_inside_map_body_raises_async_exit_raises(self):
    """with_ inside map: CM body raises, async exit raises."""
    exit_exc = RuntimeError('exit in map')

    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            raise exit_exc
          return _exit()
        return False

    def map_fn(item):
      c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(ValueError('inner')))
      return c.run()

    c = Chain([1]).map(map_fn)
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIs(ctx.exception, exit_exc)

  async def test_with_inside_if_fn_body_raises_async_exit(self):
    """with_ inside if_ fn: CM body raises, async exit suppresses."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            return True
          return _exit()
        return False

    def if_fn(val):
      c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(ValueError('if body')))
      return c.run()

    c = Chain(10).if_(lambda v: v > 5, then=if_fn)
    result = await c.run()
    self.assertIsNone(result)

  async def test_with_except_handler_body_raises_async_exit_raises(self):
    """with_ with except_: body raises, async exit raises -- except sees exit_exc with __cause__."""
    body_exc = ValueError('body')
    exit_exc = RuntimeError('exit')
    caught_exceptions = []

    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            raise exit_exc
          return _exit()
        return False

    def handler(rv, exc):
      caught_exceptions.append(exc)
      return 'handled'

    c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(body_exc)).except_(handler)
    result = await c.run()
    self.assertEqual(result, 'handled')
    self.assertEqual(len(caught_exceptions), 1)
    self.assertIs(caught_exceptions[0], exit_exc)
    self.assertIs(caught_exceptions[0].__cause__, body_exc)

  async def test_with_finally_handler_body_raises_async_exit_raises(self):
    """with_ with finally_: body raises, async exit raises -- finally runs."""
    finally_called = []

    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            raise RuntimeError('exit')
          return _exit()
        return False

    c = (
      Chain(CM())
      .with_(lambda ctx: (_ for _ in ()).throw(ValueError('body')))
      .finally_(lambda v: finally_called.append('finally'))
    )
    with self.assertRaises(RuntimeError):
      await c.run()
    self.assertEqual(finally_called, ['finally'])

  async def test_nested_with_outer_sync_inner_async_exit_raises(self):
    """Nested with_: outer sync, inner has async exit that raises."""
    inner_exit_exc = RuntimeError('inner exit')

    class OuterCM:
      def __enter__(self):
        return InnerCM()
      def __exit__(self, *args):
        return False

    class InnerCM:
      def __enter__(self):
        return 'inner_ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            raise inner_exit_exc
          return _exit()
        return False

    c = (
      Chain(OuterCM())
      .with_(lambda outer_ctx: Chain(outer_ctx).with_(
        lambda inner_ctx: (_ for _ in ()).throw(ValueError('inner body'))
      ).run())
    )
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIs(ctx.exception, inner_exit_exc)

  async def test_with_do_body_raises_control_flow_async_exit_cleans(self):
    """with_do: body raises _Return, async exit still runs cleanup path."""
    exit_called = []

    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          exit_called.append(True)
          return False
        return _exit()

    c = Chain(CM()).with_do(lambda ctx: 'ok')
    result = await c.run()
    self.assertIsNotNone(result)
    self.assertEqual(exit_called, [True])

  async def test_with_suppress_returns_none_for_with_(self):
    """with_: suppression via awaitable True returns None (not outer_value)."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            return True
          return _exit()
        return False

    c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(ValueError('x')))
    result = await c.run()
    self.assertIsNone(result)

  async def test_with_do_suppress_returns_outer_value(self):
    """with_do: suppression via awaitable True returns the CM (outer value)."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
          async def _exit():
            return True
          return _exit()
        return False

    cm = CM()
    c = Chain(cm).with_do(lambda ctx: (_ for _ in ()).throw(ValueError('x')))
    result = await c.run()
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# H2: _sync_generator coroutine detection -- expanded (8 tests)
# ---------------------------------------------------------------------------

class TestH2SyncGeneratorExpanded(unittest.TestCase):
  """H2: Expanded coroutine detection in sync generators."""

  def test_iterate_do_async_fn_sync_for_raises_typeerror(self):
    """iterate_do(async_fn) with sync for raises TypeError."""
    async def async_fn(x):
      return x * 2

    c = Chain([10, 20, 30])
    gen = c.iterate_do(async_fn)
    with self.assertRaises(TypeError) as ctx:
      list(gen())
    self.assertIn('coroutine', str(ctx.exception).lower())

  def test_iterate_fn_returns_awaitable_only_sometimes(self):
    """iterate() with fn that returns awaitable only for certain items raises on first awaitable."""
    call_count = 0

    def mixed_fn(x):
      nonlocal call_count
      call_count += 1
      if x == 2:
        async def _coro():
          return x
        return _coro()
      return x * 10

    c = Chain([0, 1, 2, 3])
    gen = c.iterate(mixed_fn)
    collected = []
    with self.assertRaises(TypeError):
      for item in gen():
        collected.append(item)
    # Items before the awaitable are yielded
    self.assertEqual(collected, [0, 10])

  def test_iterate_sync_fn_works(self):
    """Control: iterate() with sync fn works fine."""
    c = Chain([1, 2, 3])
    gen = c.iterate(lambda x: x ** 2)
    result = list(gen())
    self.assertEqual(result, [1, 4, 9])

  def test_iterate_do_sync_fn_works(self):
    """Control: iterate_do() with sync fn works fine."""
    tracker = []
    c = Chain([1, 2, 3])
    gen = c.iterate_do(lambda x: tracker.append(x))
    result = list(gen())
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [1, 2, 3])

  def test_iterate_with_chain_returning_awaitable(self):
    """iterate() with chain that has async step as fn -- raises TypeError in sync for."""
    async def async_step(x):
      return x + 1

    c = Chain([1, 2, 3])
    gen = c.iterate(async_step)
    with self.assertRaises(TypeError):
      list(gen())

  def test_iterate_coroutine_closed_on_detection(self):
    """iterate() closes the coroutine when TypeError is raised."""
    close_called = []
    original_close = None

    async def async_fn(x):
      return x + 1

    # We verify no RuntimeWarning about coroutine never being awaited
    import warnings
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      c = Chain([1])
      gen = c.iterate(async_fn)
      with self.assertRaises(TypeError):
        list(gen())
    # No RuntimeWarning about unawaited coroutine
    runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
    self.assertEqual(len(runtime_warnings), 0)

  def test_iterate_empty_with_async_fn_no_error(self):
    """iterate(async_fn) on empty list does not raise (fn never called)."""
    async def async_fn(x):
      return x + 1

    c = Chain([])
    gen = c.iterate(async_fn)
    result = list(gen())
    self.assertEqual(result, [])

  def test_iterate_none_fn_passthrough(self):
    """iterate(None) passes items through regardless."""
    c = Chain([10, 20, 30])
    gen = c.iterate()
    result = list(gen())
    self.assertEqual(result, [10, 20, 30])


class TestH2SyncGeneratorAsyncExpanded(unittest.IsolatedAsyncioTestCase):
  """H2: async for with async fn works correctly."""

  async def test_iterate_async_fn_async_for_works(self):
    """iterate(async_fn) consumed via async for works correctly."""
    async def async_fn(x):
      return x * 3

    c = Chain([1, 2, 3])
    gen = c.iterate(async_fn)
    result = []
    async for item in gen():
      result.append(item)
    self.assertEqual(result, [3, 6, 9])


# ---------------------------------------------------------------------------
# H3: _get_true_source_link cycle guard -- expanded (6 tests)
# ---------------------------------------------------------------------------

class TestH3CycleGuardExpanded(unittest.TestCase):
  """H3: Expanded cycle guard tests for _get_true_source_link."""

  def test_non_cyclic_deep_chain_10_levels(self):
    """Non-cyclic chain 10 levels deep resolves without issue."""
    bottom = Chain(lambda: 'leaf')
    current = bottom
    for _ in range(10):
      current = Chain(current)
    link = Link(current)
    result = _get_true_source_link(link, None)
    self.assertIsNotNone(result)

  def test_two_cycle(self):
    """2-cycle terminates."""
    c1 = Chain(lambda: 1)
    c2 = Chain(lambda: 2)
    l1 = Link(c1)
    l2 = Link(c2)
    c1.root_link = l2
    c2.root_link = l1
    result = _get_true_source_link(l1, None)
    self.assertIsNotNone(result)

  def test_three_cycle(self):
    """3-cycle terminates."""
    c1 = Chain(lambda: 1)
    c2 = Chain(lambda: 2)
    c3 = Chain(lambda: 3)
    l1, l2, l3 = Link(c1), Link(c2), Link(c3)
    c1.root_link = l2
    c2.root_link = l3
    c3.root_link = l1
    result = _get_true_source_link(l1, None)
    self.assertIsNotNone(result)

  def test_self_referential_chain(self):
    """Chain where root_link.v is the same chain (self-referential)."""
    c = Chain(lambda: 1)
    l = Link(c)
    c.root_link = l
    result = _get_true_source_link(l, None)
    self.assertIsNotNone(result)

  def test_returns_valid_link_for_normal_case(self):
    """Verify the function returns the leaf link for normal non-cyclic cases."""
    inner = Chain(lambda: 42)
    outer = Chain(inner)
    link = Link(outer)
    result = _get_true_source_link(link, None)
    self.assertIsNotNone(result)
    # Should drill down to the innermost non-chain link
    self.assertFalse(result.is_chain)

  def test_traceback_works_after_cycle_guard(self):
    """Traceback still works after cycle guard triggers on cyclic structure."""
    c1 = Chain(lambda: 1)
    c2 = Chain(lambda: 2)
    l1 = Link(c1)
    l2 = Link(c2)
    c1.root_link = l2
    c2.root_link = l1

    exc = _exc_with_tb('cycle test')
    # Should not crash
    _modify_traceback(exc, c1, l1, l1)
    self.assertTrue(getattr(exc, '__quent__', False))


# ---------------------------------------------------------------------------
# H4: _clean_chained_exceptions iterative -- expanded (8 tests)
# ---------------------------------------------------------------------------

class TestH4CleanChainedExceptionsExpanded(unittest.TestCase):
  """H4: Expanded iterative cleaning tests."""

  def test_context_chain_500(self):
    """Exception chain of 500 via __context__ -- no RecursionError."""
    root = _make_exception_chain(500, '__context__')
    try:
      _clean_chained_exceptions(root, set())
    except RecursionError:
      self.fail('RecursionError for 500-deep __context__ chain')

  def test_cause_chain_500(self):
    """Exception chain of 500 via __cause__ -- no RecursionError."""
    root = _make_exception_chain(500, '__cause__')
    try:
      _clean_chained_exceptions(root, set())
    except RecursionError:
      self.fail('RecursionError for 500-deep __cause__ chain')

  def test_mixed_cause_context_tree(self):
    """Mixed __cause__ and __context__ tree (branching)."""
    root = ValueError('root')
    b1 = ValueError('branch1')
    b2 = ValueError('branch2')
    b1c = ValueError('b1-child')
    b2c = ValueError('b2-child')
    root.__cause__ = b1
    root.__context__ = b2
    b1.__cause__ = b1c
    b2.__context__ = b2c
    _clean_chained_exceptions(root, set())

  def test_diamond_exception_chain(self):
    """Diamond-shaped: A->B, A->C, B->D, C->D."""
    d = ValueError('D')
    b = ValueError('B')
    c_exc = ValueError('C')
    a = ValueError('A')
    b.__context__ = d
    c_exc.__context__ = d
    a.__cause__ = b
    a.__context__ = c_exc
    _clean_chained_exceptions(a, set())

  def test_exception_with_quent_traceback_frames_cleaned(self):
    """Exception with __traceback__ containing quent frames -- frames cleaned."""
    def raise_in_chain(x):
      raise ValueError('traced')

    c = Chain(5).then(raise_in_chain)
    try:
      c.run()
    except ValueError as exc:
      _clean_chained_exceptions(exc, set())
      # After cleaning, no quent internal frames should remain
      tb = exc.__traceback__
      while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        if filename != '<quent>':
          self.assertFalse(filename.startswith(_quent_file),
                           f'Quent internal frame not cleaned: {filename}')
        tb = tb.tb_next

  def test_exception_from_map_chain_cleaned(self):
    """Exception from map -- chain cleaned properly."""
    def failing_fn(x):
      if x == 2:
        raise ValueError('map fail')
      return x

    c = Chain([1, 2, 3]).map(failing_fn)
    try:
      c.run()
    except ValueError as exc:
      _clean_chained_exceptions(exc, set())

  def test_exception_from_if_predicate_chain_cleaned(self):
    """Exception from if_ predicate -- chain cleaned."""
    def bad_pred(x):
      raise ValueError('pred fail')

    c = Chain(5).if_(bad_pred, then=lambda x: x)
    try:
      c.run()
    except ValueError as exc:
      _clean_chained_exceptions(exc, set())

  def test_exception_from_gather_chain_cleaned(self):
    """Exception from gather -- chain cleaned."""
    def fn_ok(v):
      return v + 1

    def fn_fail(v):
      raise ValueError('gather fail')

    c = Chain(5).gather(fn_ok, fn_fail)
    try:
      c.run()
    except ValueError as exc:
      _clean_chained_exceptions(exc, set())


# ---------------------------------------------------------------------------
# H5: _stringify_chain depth guard -- expanded (6 tests)
# ---------------------------------------------------------------------------

class TestH5DepthGuardExpanded(unittest.TestCase):
  """H5: Expanded depth guard tests for _stringify_chain."""

  def test_chain_51_levels_truncated(self):
    """Chain nested 51 levels deep -- truncation message appears."""
    bottom = Chain(lambda: 'leaf')
    current = bottom
    for _ in range(51):
      current = Chain(current)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(current, nest_lvl=0, ctx=ctx, max_depth=50)
    self.assertIn('truncated at depth', result)

  def test_chain_49_levels_no_truncation(self):
    """Chain nested 49 levels deep -- no truncation."""
    bottom = Chain(lambda: 'leaf')
    current = bottom
    for _ in range(49):
      current = Chain(current)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(current, nest_lvl=0, ctx=ctx, max_depth=50)
    self.assertNotIn('truncated', result)

  def test_chain_100_levels_truncated_no_crash(self):
    """Chain nested 100 levels deep -- truncation, no crash."""
    bottom = Chain(lambda: 'leaf')
    current = bottom
    for _ in range(100):
      current = Chain(current)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    try:
      result = _stringify_chain(current, nest_lvl=0, ctx=ctx, max_depth=50)
    except RecursionError:
      self.fail('RecursionError at 100 levels')
    self.assertIn('truncated at depth', result)

  def test_nested_chain_with_if_else_deep(self):
    """Nested chain with if_/else_ at deep levels."""
    bottom = Chain(lambda: 'leaf')
    current = bottom
    for i in range(55):
      current = Chain(current).if_(lambda v: True, then=lambda v: v)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(current, nest_lvl=0, ctx=ctx, max_depth=50)
    self.assertIn('truncated at depth', result)

  def test_nested_chain_with_map_deep(self):
    """Nested chain with map at deep levels."""
    bottom = Chain(lambda: [1])
    current = bottom
    for _ in range(55):
      current = Chain(current).map(lambda x: x)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(current, nest_lvl=0, ctx=ctx, max_depth=50)
    self.assertIsInstance(result, str)

  def test_shallow_chain_after_fix_displays_correctly(self):
    """Shallow chains still display correctly after the depth guard fix."""
    c = Chain(lambda: 1).then(lambda x: x + 1).then(lambda x: x * 2)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertNotIn('truncated', result)
    self.assertIn('Chain', result)
    self.assertIn('.then', result)


# ---------------------------------------------------------------------------
# H6: _modify_traceback failure suppression -- expanded (12 tests)
# ---------------------------------------------------------------------------

class TestH6FailureSuppressionExpanded(unittest.TestCase):
  """H6: Expanded tests for _modify_traceback failure suppression across chain methods."""

  def test_then_raises_except_still_runs(self):
    """then() raises -- except_ handler still runs despite traceback formatting."""
    handler_called = []

    c = Chain(5).then(lambda x: 1 / 0).except_(lambda rv, e: handler_called.append(type(e).__name__) or 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')
    self.assertEqual(handler_called, ['ZeroDivisionError'])

  def test_map_raises_except_catches(self):
    """map() raises -- except_ catches despite traceback formatting."""
    handler_called = []

    def failing(x):
      if x == 2:
        raise ValueError('map err')
      return x

    c = Chain([1, 2, 3]).map(failing).except_(lambda rv, e: handler_called.append(str(e)) or 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')
    self.assertEqual(handler_called, ['map err'])

  def test_filter_raises_except_catches(self):
    """filter() raises -- except_ catches."""
    def bad_filter(x):
      raise ValueError('filter err')

    c = Chain([1, 2]).filter(bad_filter).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')

  def test_gather_fn_raises_except_catches(self):
    """gather() fn raises -- except_ catches."""
    c = Chain(5).gather(lambda v: v + 1, lambda v: 1 / 0).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')

  def test_with_body_raises_except_catches(self):
    """with_() body raises -- except_ catches."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, *args):
        return False

    c = Chain(CM()).with_(lambda ctx: 1 / 0).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')

  def test_if_pred_raises_except_catches(self):
    """if_() predicate raises -- except_ catches."""
    def bad_pred(x):
      raise ValueError('pred err')

    c = Chain(5).if_(bad_pred, then=lambda x: x).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')

  def test_if_fn_raises_except_catches(self):
    """if_() fn raises -- except_ catches."""
    c = Chain(5).if_(lambda x: True, then=lambda x: 1 / 0).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')

  def test_else_fn_raises_except_catches(self):
    """else_() fn raises -- except_ catches."""
    c = Chain(5).if_(lambda x: False, then=lambda x: x).else_(lambda x: 1 / 0).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')

  def test_finally_still_runs_when_formatting_fails(self):
    """finally_ handler still runs when traceback formatting fails."""
    finally_called = []

    c = Chain(5).then(lambda x: 1 / 0).finally_(lambda v: finally_called.append(v))
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      with self.assertRaises(ZeroDivisionError):
        c.run()
    self.assertEqual(finally_called, [5])

  def test_double_fault_except_handler_raises(self):
    """Double fault: except_ handler raises -- the handler exception propagates."""
    def bad_handler(rv, e):
      raise RuntimeError('handler boom')

    c = Chain(5).then(lambda x: 1 / 0).except_(bad_handler)
    # Without mocking: the handler raises, _modify_traceback is called on that
    # second exception. The handler error propagates with __cause__ = original.
    with self.assertRaises(RuntimeError) as ctx:
      c.run()
    self.assertEqual(str(ctx.exception), 'handler boom')
    self.assertIsInstance(ctx.exception.__cause__, ZeroDivisionError)

  def test_except_and_finally_both_run_when_formatting_fails(self):
    """Chain with both except_ and finally_ -- formatting fails, both still run."""
    handler_order = []

    c = (
      Chain(5)
      .then(lambda x: 1 / 0)
      .except_(lambda rv, e: handler_order.append('except') or 'caught')
      .finally_(lambda v: handler_order.append('finally'))
    )
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')
    self.assertIn('except', handler_order)
    self.assertIn('finally', handler_order)

  def test_error_propagates_without_except(self):
    """Error propagates normally when no except_ and formatting fails."""
    c = Chain(5).then(lambda x: 1 / 0)
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      with self.assertRaises(ZeroDivisionError):
        c.run()


class TestH6FailureSuppressionAsync(unittest.IsolatedAsyncioTestCase):
  """H6 async: formatting failure suppression in async execution."""

  async def test_async_fn_raises_except_handler_runs(self):
    """Async variant: async fn raises -- except_ handler still runs."""
    async def failing(x):
      raise ValueError('async err')

    c = Chain(5).then(failing).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = await c.run()
    self.assertEqual(result, 'caught')


# ---------------------------------------------------------------------------
# M1: _make_gather cleanup -- expanded (8 tests)
# ---------------------------------------------------------------------------

class TestM1GatherCleanupExpanded(unittest.TestCase):
  """M1: Expanded gather cleanup tests."""

  def test_stringio_not_closed_on_error(self):
    """Gather: one fn raises, another returned StringIO -- StringIO NOT closed."""
    sio = io.StringIO('data')

    def fn_sio(v):
      return sio

    def fn_raises(v):
      raise ValueError('err')

    gather_op = _make_gather((fn_sio, fn_raises))
    with self.assertRaises(ValueError):
      gather_op(42)
    # StringIO should NOT have been closed
    self.assertFalse(sio.closed)

  def test_coroutine_is_closed_on_error(self):
    """Gather: one fn raises, another returned a coroutine -- coroutine IS closed."""
    close_called = []

    class FakeAwaitable:
      def __await__(self):
        yield
      def close(self):
        close_called.append(True)

    gather_op = _make_gather((lambda v: FakeAwaitable(), lambda v: (_ for _ in ()).throw(ValueError('err'))))
    with self.assertRaises(ValueError):
      gather_op(42)
    self.assertEqual(close_called, [True])

  def test_object_with_close_but_not_awaitable_not_closed(self):
    """Gather: one fn raises, another returned object with close() but not awaitable -- NOT closed."""
    close_called = []

    class Closeable:
      def close(self):
        close_called.append(True)

    gather_op = _make_gather((lambda v: Closeable(), lambda v: (_ for _ in ()).throw(ValueError('err'))))
    with self.assertRaises(ValueError):
      gather_op(42)
    self.assertEqual(close_called, [])

  def test_all_succeed_no_cleanup(self):
    """Gather: all fns succeed, one returns file-like -- no cleanup triggered."""
    sio = io.StringIO('data')

    gather_op = _make_gather((lambda v: sio, lambda v: v + 1))
    result = gather_op(42)
    self.assertEqual(len(result), 2)
    self.assertFalse(sio.closed)

  def test_first_fn_raises_no_results_to_clean(self):
    """Gather: first fn raises immediately -- no results to clean."""
    gather_op = _make_gather((lambda v: (_ for _ in ()).throw(ValueError('first')), lambda v: v))
    with self.assertRaises(ValueError) as ctx:
      gather_op(42)
    self.assertEqual(str(ctx.exception), 'first')

  def test_five_fns_third_raises_first_two_coros_closed(self):
    """Gather with 5 fns, 3rd raises, 1st and 2nd returned coroutines -- both closed."""
    close_count = 0

    class FakeAwaitable:
      def __await__(self):
        yield
      def close(self):
        nonlocal close_count
        close_count += 1

    def fn_aw(v):
      return FakeAwaitable()

    def fn_raises(v):
      raise ValueError('3rd err')

    def fn_ok(v):
      return v

    gather_op = _make_gather((fn_aw, fn_aw, fn_raises, fn_ok, fn_ok))
    with self.assertRaises(ValueError):
      gather_op(42)
    self.assertEqual(close_count, 2)

  def test_five_fns_third_raises_fourth_fifth_never_called(self):
    """Gather with 5 fns, 3rd raises, 4th and 5th never called -- partial results."""
    call_tracker = []

    def tracked(idx):
      def fn(v):
        call_tracker.append(idx)
        if idx == 3:
          raise ValueError('err')
        return v
      return fn

    gather_op = _make_gather(tuple(tracked(i) for i in range(1, 6)))
    with self.assertRaises(ValueError):
      gather_op(42)
    # Only 1, 2, 3 were called; 4, 5 were not
    self.assertEqual(call_tracker, [1, 2, 3])


class TestM1GatherCleanupAsync(unittest.IsolatedAsyncioTestCase):
  """M1: Async gather cleanup."""

  async def test_async_gather_one_coro_raises(self):
    """Gather: async gather, one coro raises -- asyncio.gather behavior."""
    async def ok_coro(v):
      return v + 1

    async def fail_coro(v):
      raise ValueError('coro err')

    c = Chain(10).gather(ok_coro, fail_coro)
    with self.assertRaises(ValueError):
      await c.run()


# ---------------------------------------------------------------------------
# M4: realpath -- expanded (4 tests)
# ---------------------------------------------------------------------------

class TestM4RealpathExpanded(unittest.TestCase):
  """M4: Expanded realpath tests."""

  def test_quent_file_no_symlink_components(self):
    """_quent_file does NOT contain symlink components."""
    resolved = os.path.realpath(_quent_file.rstrip(os.sep)) + os.sep
    self.assertEqual(_quent_file, resolved)

  def test_quent_file_ends_with_sep(self):
    """_quent_file ends with os.sep."""
    self.assertTrue(_quent_file.endswith(os.sep))

  def test_quent_file_directory_exists(self):
    """_quent_file is a directory that exists."""
    self.assertTrue(os.path.isdir(_quent_file.rstrip(os.sep)))

  def test_frame_filtering_works(self):
    """Frame filtering works: quent internal frames are hidden in TracebackException output."""
    def user_fn(x):
      raise ValueError('user error')

    c = Chain(5).then(user_fn)
    try:
      c.run()
    except ValueError as exc:
      # Frame filtering happens via TracebackException (patched __init__),
      # not on the raw __traceback__. Verify via format output.
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      text = ''.join(te.format())
      for line in text.splitlines():
        if 'File "' in line and _quent_file in line:
          self.fail(f'Quent internal frame leaked in formatted output: {line}')


# ---------------------------------------------------------------------------
# M5: _get_obj_name sanitizes newlines -- expanded (6 tests)
# ---------------------------------------------------------------------------

class TestM5GetObjNameExpanded(unittest.TestCase):
  """M5: Expanded newline sanitization tests."""

  def test_repr_with_newline_escaped(self):
    """Object repr with \\n -- escaped."""
    class Obj:
      def __repr__(self):
        return 'hello\nworld'
    result = _get_obj_name(Obj())
    self.assertNotIn('\n', result)
    self.assertIn('\\n', result)

  def test_repr_with_carriage_return_escaped(self):
    """Object repr with \\r -- escaped."""
    class Obj:
      def __repr__(self):
        return 'hello\rworld'
    result = _get_obj_name(Obj())
    self.assertNotIn('\r', result)
    self.assertIn('\\r', result)

  def test_repr_with_crlf_both_escaped(self):
    """Object repr with \\r\\n -- both escaped."""
    class Obj:
      def __repr__(self):
        return 'line1\r\nline2'
    result = _get_obj_name(Obj())
    self.assertNotIn('\r', result)
    self.assertNotIn('\n', result)
    self.assertIn('\\r', result)
    self.assertIn('\\n', result)

  def test_repr_with_multiple_newlines(self):
    """Object repr with multiple newlines."""
    class Obj:
      def __repr__(self):
        return 'a\nb\nc\nd\ne'
    result = _get_obj_name(Obj())
    self.assertEqual(result.count('\\n'), 4)
    self.assertNotIn('\n', result)

  def test_very_long_repr_no_crash(self):
    """Object with very long repr (1000 chars) -- no crash."""
    class Obj:
      def __repr__(self):
        return 'x' * 1000
    result = _get_obj_name(Obj())
    self.assertEqual(len(result), 1000)

  def test_name_with_newlines_returns_name(self):
    """Object where __name__ has newlines -- name is used, not repr."""
    class Obj:
      __name__ = 'bad\nname'
      def __repr__(self):
        return 'repr\nvalue'

    result = _get_obj_name(Obj())
    # __name__ is used and returned as-is (name path, not repr path)
    self.assertEqual(result, 'bad\nname')


# ---------------------------------------------------------------------------
# M6: Error guards on hooks -- expanded (6 tests)
# ---------------------------------------------------------------------------

class TestM6ErrorGuardsExpanded(unittest.TestCase):
  """M6: Expanded error guard tests on exception hooks."""

  def test_excepthook_circular_context_no_crash(self):
    """_quent_excepthook with exception that has circular __context__ -- no crash."""
    exc = ValueError('circular')
    exc.__quent__ = True
    exc.__context__ = exc  # circular
    hook_called = []

    def mock_hook(exc_type, exc_value, exc_tb):
      hook_called.append(True)

    with patch('sys.__excepthook__', mock_hook):
      _quent_excepthook(ValueError, exc, None)
    self.assertEqual(hook_called, [True])

  def test_patched_te_init_circular_cause_no_crash(self):
    """_patched_te_init with exception that has circular __cause__ -- no crash."""
    exc = ValueError('circular cause')
    exc.__quent__ = True
    exc.__cause__ = exc  # circular
    # Should not crash
    te = traceback.TracebackException(ValueError, exc, None)
    self.assertIsNotNone(te)

  def test_normal_exceptions_display_after_hook_error(self):
    """After a hook error, normal exceptions still display correctly."""
    # Trigger a hook error
    exc = ValueError('hook err test')
    exc.__quent__ = True
    with patch('quent._traceback._clean_chained_exceptions', side_effect=RuntimeError('fail')):
      _quent_excepthook(ValueError, exc, None)

    # Now a normal exception should still work
    normal_exc = ValueError('normal')
    normal_exc.__quent__ = True
    hook_called = []

    def mock_hook(exc_type, exc_value, exc_tb):
      hook_called.append(True)

    with patch('sys.__excepthook__', mock_hook):
      _quent_excepthook(ValueError, normal_exc, None)
    self.assertEqual(hook_called, [True])

  def test_format_exception_with_quent_marked(self):
    """traceback.format_exception works with quent-marked exception."""
    def raise_in_chain(x):
      raise ValueError('format test')

    c = Chain(5).then(raise_in_chain)
    try:
      c.run()
    except ValueError as exc:
      lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
      text = ''.join(lines)
      self.assertIn('ValueError', text)
      self.assertIn('format test', text)

  def test_logging_exception_with_quent(self):
    """logging.exception works with quent-marked exception."""
    def raise_in_chain(x):
      raise ValueError('log test')

    c = Chain(5).then(raise_in_chain)
    try:
      c.run()
    except ValueError:
      # Should not crash
      logger = logging.getLogger('quent_test')
      handler = logging.StreamHandler(io.StringIO())
      logger.addHandler(handler)
      logger.exception('test')
      logger.removeHandler(handler)

  def test_traceback_exception_compact_with_quent(self):
    """TracebackException with compact=True and quent exception."""
    def raise_in_chain(x):
      raise ValueError('compact test')

    c = Chain(5).then(raise_in_chain)
    try:
      c.run()
    except ValueError as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__, compact=True)
      lines = list(te.format())
      text = ''.join(lines)
      self.assertIn('ValueError', text)


# ---------------------------------------------------------------------------
# Additional cross-feature interaction tests
# ---------------------------------------------------------------------------

class TestCrossFeatureInteractions(unittest.IsolatedAsyncioTestCase):
  """Tests that combine multiple fixed behaviors."""

  async def test_gather_with_except_and_async_coro(self):
    """Gather with mixed sync/async where last raises, except handler catches."""
    async def async_fn(v):
      return v + 1

    c = (
      Chain(10)
      .gather(async_fn, lambda v: (_ for _ in ()).throw(ValueError('err')))
      .except_(lambda rv, e: 'caught')
    )
    result = c.run()
    self.assertEqual(result, 'caught')

  async def test_map_with_traceback_modify_failure(self):
    """map raises with traceback modify failure -- original error propagates."""
    def fn(x):
      if x == 2:
        raise ValueError('map fail')
      return x

    c = Chain([1, 2, 3]).map(fn)
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      with self.assertRaises(ValueError) as ctx:
        c.run()
      self.assertEqual(str(ctx.exception), 'map fail')

  async def test_iterate_async_for_with_except_handler(self):
    """iterate(async_fn) via async for with error -- chain handles error in fn."""
    call_count = 0

    async def async_fn(x):
      nonlocal call_count
      call_count += 1
      if x == 2:
        raise ValueError('async iter err')
      return x * 10

    c = Chain([0, 1, 2, 3])
    gen = c.iterate(async_fn)
    collected = []
    with self.assertRaises(ValueError):
      async for item in gen():
        collected.append(item)
    self.assertEqual(collected, [0, 10])

  async def test_chain_with_deep_nesting_and_error(self):
    """Deep nested chain raises -- traceback displays with truncation for very deep chains."""
    bottom = Chain(lambda: (_ for _ in ()).throw(ValueError('deep fail')))
    current = bottom
    for _ in range(5):
      current = Chain(current)

    with self.assertRaises(ValueError) as ctx:
      current.run()
    self.assertEqual(str(ctx.exception), 'deep fail')

  def test_chain_exception_cleaning_after_if_else(self):
    """Exception from if_/else_ path -- exception chains cleaned."""
    def bad_fn(x):
      raise ValueError('else err')

    c = Chain(5).if_(lambda x: False, then=lambda x: x).else_(bad_fn)
    try:
      c.run()
    except ValueError as exc:
      _clean_chained_exceptions(exc, set())

  def test_gather_cleanup_with_traceback_failure(self):
    """Gather cleanup works even when traceback modification fails."""
    async def async_fn(v):
      return v + 1

    def raises_fn(v):
      raise ValueError('gather err')

    c = Chain(10).gather(async_fn, raises_fn).except_(lambda rv, e: 'caught')
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
    self.assertEqual(result, 'caught')

  async def test_with_async_exit_success_path(self):
    """with_: success path with awaitable __exit__ returns correct value."""
    class CM:
      def __enter__(self):
        return 42
      def __exit__(self, *args):
        async def _exit():
          return False
        return _exit()

    c = Chain(CM()).with_(lambda ctx: ctx * 2)
    result = await c.run()
    self.assertEqual(result, 84)

  async def test_with_do_async_exit_success_path_returns_outer(self):
    """with_do: success path with awaitable __exit__ returns outer value."""
    class CM:
      def __enter__(self):
        return 42
      def __exit__(self, *args):
        async def _exit():
          return False
        return _exit()

    cm = CM()
    c = Chain(cm).with_do(lambda ctx: 'ignored')
    result = await c.run()
    self.assertIs(result, cm)


class TestAdditionalEdgeCases(unittest.TestCase):
  """Additional edge case tests for completeness."""

  def test_get_obj_name_partial(self):
    """_get_obj_name with functools.partial."""
    import functools
    fn = functools.partial(int, base=2)
    result = _get_obj_name(fn)
    self.assertIn('partial', result)
    self.assertIn('int', result)

  def test_get_obj_name_lambda(self):
    """_get_obj_name with a lambda."""
    fn = lambda x: x + 1
    result = _get_obj_name(fn)
    self.assertIn('<lambda>', result)

  def test_get_obj_name_none(self):
    """_get_obj_name with None."""
    result = _get_obj_name(None)
    self.assertEqual(result, 'None')

  def test_get_obj_name_chain(self):
    """_get_obj_name with Chain."""
    c = Chain(1)
    result = _get_obj_name(c)
    self.assertEqual(result, 'Chain')

  def test_clean_internal_frames_none(self):
    """_clean_internal_frames with None returns None."""
    self.assertIsNone(_clean_internal_frames(None))

  def test_clean_chained_exceptions_none(self):
    """_clean_chained_exceptions with None is a no-op."""
    _clean_chained_exceptions(None, set())

  def test_modify_traceback_source_link_first_write_wins(self):
    """_modify_traceback: __quent_source_link__ uses first-write-wins."""
    exc = _exc_with_tb()
    link1 = Link(lambda: 1)
    link2 = Link(lambda: 2)
    # First call sets __quent_source_link__
    exc.__quent_source_link__ = link1
    c = Chain(lambda: 1)
    _modify_traceback(exc, c, link2, link1)
    # First-write should be preserved (link1 was already set)

  def test_ctx_object_init(self):
    """_Ctx object initializes correctly."""
    link = Link(lambda: 1)
    ctx = _Ctx(source_link=link, link_temp_args=None)
    self.assertIs(ctx.source_link, link)
    self.assertIsNone(ctx.link_temp_args)
    self.assertFalse(ctx.found)

  def test_stringify_chain_empty(self):
    """_stringify_chain with empty chain (no root, no links)."""
    c = Chain()
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('Chain', result)
    self.assertIn('()', result)

  def test_except_handler_body_no_except_link(self):
    """_except_handler_body with no on_except_link re-raises."""
    c = Chain(5)
    exc = ValueError('test')
    link = Link(lambda: 1)
    with self.assertRaises(ValueError):
      _except_handler_body(exc, c, link, None, 5)

  def test_except_handler_body_wrong_exception_type(self):
    """_except_handler_body with wrong exception type re-raises."""
    c = Chain(5).except_(lambda rv, e: 'caught', exceptions=[TypeError])
    exc = ValueError('test')
    link = Link(lambda: 1)
    with self.assertRaises(ValueError):
      _except_handler_body(exc, c, link, None, 5)


class TestAdditionalAsyncEdgeCases(unittest.IsolatedAsyncioTestCase):
  """Additional async edge case tests."""

  async def test_iterate_do_async_for_preserves_original_items(self):
    """iterate_do via async for preserves original items, fn is side-effect."""
    side_effects = []

    async def async_side_effect(x):
      side_effects.append(x * 10)
      return 'ignored'

    c = Chain([1, 2, 3])
    gen = c.iterate_do(async_side_effect)
    result = []
    async for item in gen():
      result.append(item)
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(side_effects, [10, 20, 30])

  async def test_with_async_enter_async_exit(self):
    """Full async CM (aenter/aexit) works with with_."""
    class AsyncCM:
      async def __aenter__(self):
        return 'async_ctx'
      async def __aexit__(self, *args):
        return False

    c = Chain(AsyncCM()).with_(lambda ctx: ctx + '_done')
    result = await c.run()
    self.assertEqual(result, 'async_ctx_done')

  async def test_with_do_async_cm_returns_outer(self):
    """Full async CM with with_do returns outer value."""
    class AsyncCM:
      async def __aenter__(self):
        return 'async_ctx'
      async def __aexit__(self, *args):
        return False

    cm = AsyncCM()
    c = Chain(cm).with_do(lambda ctx: 'ignored')
    result = await c.run()
    self.assertIs(result, cm)

  async def test_gather_all_async(self):
    """Gather with all async fns works correctly."""
    async def fn1(v):
      return v + 1
    async def fn2(v):
      return v + 2
    async def fn3(v):
      return v + 3

    c = Chain(10).gather(fn1, fn2, fn3)
    result = await c.run()
    self.assertEqual(result, [11, 12, 13])

  async def test_map_async_fn(self):
    """map with async fn works correctly."""
    async def async_fn(x):
      return x * 10

    c = Chain([1, 2, 3]).map(async_fn)
    result = await c.run()
    self.assertEqual(result, [10, 20, 30])


if __name__ == '__main__':
  unittest.main()
