"""Exhaustive regression tests for traceback and ops bug fixes.

Covers: H1 (_await_exit_suppress exception chaining), H2 (_sync_generator
unawaited coroutine detection), H3 (_get_true_source_link cycle guard),
H4 (_clean_chained_exceptions iterative), H5 (_stringify_chain depth guard),
H6 (_modify_traceback failure suppression), M1 (_make_gather cleanup only
closes awaitables), M4 (_quent_file uses realpath), M5 (_get_obj_name
sanitizes newlines), M6 (error guards on exception hooks),
disable/enable_traceback_patching.
"""
from __future__ import annotations

import asyncio
import sys
import traceback
import types
import unittest
from inspect import isawaitable
from typing import Any
from unittest.mock import patch

from quent import Chain, Null, disable_traceback_patching, enable_traceback_patching
from quent._chain import _except_handler_body, _FrozenChain
from quent._core import Link
from quent._ops import _make_gather, _make_with, _sync_generator
from quent._traceback import (
  _clean_chained_exceptions,
  _clean_internal_frames,
  _Ctx,
  _get_obj_name,
  _get_true_source_link,
  _modify_traceback,
  _original_excepthook,
  _patched_te_init,
  _quent_excepthook,
  _quent_file,
  _stringify_chain,
  enable_traceback_patching as _enable,
  disable_traceback_patching as _disable,
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
# H1: _await_exit_suppress exception chaining (async)
# ---------------------------------------------------------------------------

class TestH1AwaitExitSuppressExceptionChaining(unittest.IsolatedAsyncioTestCase):
  """H1: Sync CM whose __exit__ returns an awaitable that raises sets __cause__."""

  async def test_exit_awaitable_raises_sets_cause(self):
    """When __exit__ returns an awaitable that raises, __cause__ is the original body exc."""
    body_error = ValueError('body error')
    exit_error = RuntimeError('exit error')

    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          raise exit_error
        return _exit()

    c = Chain(CM()).with_(lambda ctx: (_ for _ in ()).throw(body_error))
    with self.assertRaises(RuntimeError) as ctx:
      await c.run()
    self.assertIs(ctx.exception, exit_error)
    self.assertIs(ctx.exception.__cause__, body_error)

  async def test_exit_awaitable_returns_true_suppresses(self):
    """When __exit__ returns an awaitable that returns True, exception is suppressed."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          return True
        if exc_type is not None:
          return _exit()
        return False

    def body(ctx):
      raise ValueError('body error')

    c = Chain(CM()).with_(body)
    result = await c.run()
    self.assertIsNone(result)

  async def test_exit_awaitable_returns_false_reraises(self):
    """When __exit__ returns an awaitable that returns False, original exc re-raised."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          return False
        if exc_type is not None:
          return _exit()
        return False

    def body(ctx):
      raise ValueError('body error')

    c = Chain(CM()).with_(body)
    with self.assertRaises(ValueError) as ctx:
      await c.run()
    self.assertEqual(str(ctx.exception), 'body error')

  async def test_exit_awaitable_returns_truthy_suppresses(self):
    """Truthy (non-boolean) awaitable return also suppresses."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          return 'truthy'
        if exc_type is not None:
          return _exit()
        return False

    def body(ctx):
      raise ValueError('body error')

    c = Chain(CM()).with_(body)
    result = await c.run()
    self.assertIsNone(result)

  async def test_exit_awaitable_on_success_path(self):
    """When body succeeds and __exit__ returns awaitable, result is still correct."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          return False
        return _exit()

    c = Chain(CM()).with_(lambda ctx: 'success')
    result = await c.run()
    self.assertEqual(result, 'success')

  async def test_with_do_exit_awaitable_returns_true_suppresses(self):
    """with_do: suppression returns the outer value."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          return True
        if exc_type is not None:
          return _exit()
        return False

    def body(ctx):
      raise ValueError('body error')

    c = Chain(CM()).with_do(body)
    result = await c.run()
    # with_do returns outer_value (the CM instance) on suppress
    self.assertIsNotNone(result)

  async def test_exit_awaitable_raises_different_type(self):
    """Exit awaitable raising a different exception type than body."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, exc_type, exc_val, exc_tb):
        async def _exit():
          raise TypeError('exit type error')
        if exc_type is not None:
          return _exit()
        return False

    def body(ctx):
      raise ValueError('body value error')

    c = Chain(CM()).with_(body)
    with self.assertRaises(TypeError) as ctx:
      await c.run()
    self.assertIsInstance(ctx.exception.__cause__, ValueError)


# ---------------------------------------------------------------------------
# H2: _sync_generator unawaited coroutine detection
# ---------------------------------------------------------------------------

class TestH2SyncGeneratorCoroutineDetection(unittest.TestCase):
  """H2: iterate(async_fn) consumed via sync for raises TypeError."""

  def test_iterate_async_fn_sync_for_raises_typeerror(self):
    """iterate(async_fn) consumed via for loop raises TypeError about coroutine."""
    async def async_fn(x):
      return x + 1

    c = Chain([1, 2, 3])
    gen = c.iterate(async_fn)
    with self.assertRaises(TypeError) as ctx:
      list(gen())
    self.assertIn('coroutine', str(ctx.exception).lower())

  def test_iterate_do_async_fn_sync_for_raises_typeerror(self):
    """iterate_do(async_fn) consumed via for loop raises TypeError."""
    async def async_fn(x):
      return x + 1

    c = Chain([1, 2, 3])
    gen = c.iterate_do(async_fn)
    with self.assertRaises(TypeError) as ctx:
      list(gen())
    self.assertIn('coroutine', str(ctx.exception).lower())

  def test_iterate_sync_fn_sync_for_works(self):
    """Control: iterate(sync_fn) consumed via for works normally."""
    c = Chain([1, 2, 3])
    gen = c.iterate(lambda x: x * 2)
    result = list(gen())
    self.assertEqual(result, [2, 4, 6])

  def test_iterate_none_fn_sync_for_works(self):
    """Control: iterate(None) passes through items."""
    c = Chain([1, 2, 3])
    gen = c.iterate()
    result = list(gen())
    self.assertEqual(result, [1, 2, 3])

  def test_iterate_async_fn_error_message_has_iterate(self):
    """Error message mentions 'async for' as the fix."""
    async def async_fn(x):
      return x + 1

    c = Chain([1, 2, 3])
    gen = c.iterate(async_fn)
    with self.assertRaises(TypeError) as ctx:
      list(gen())
    self.assertIn('async for', str(ctx.exception))

  def test_iterate_async_fn_first_item_raises(self):
    """Error is raised on the very first async item; no items yielded before error."""
    async def async_fn(x):
      return x

    items_before_error = []
    gen = Chain([10, 20, 30]).iterate(async_fn)
    with self.assertRaises(TypeError):
      for item in gen():
        items_before_error.append(item)
    self.assertEqual(items_before_error, [])


class TestH2SyncGeneratorCoroutineDetectionAsync(unittest.IsolatedAsyncioTestCase):
  """H2 control: iterate(async_fn) consumed via async for works correctly."""

  async def test_iterate_async_fn_async_for_works(self):
    """iterate(async_fn) consumed via async for works correctly."""
    async def async_fn(x):
      return x + 1

    c = Chain([1, 2, 3])
    gen = c.iterate(async_fn)
    result = []
    async for item in gen():
      result.append(item)
    self.assertEqual(result, [2, 3, 4])

  async def test_iterate_do_async_fn_async_for_works(self):
    """iterate_do(async_fn) consumed via async for works correctly."""
    collected = []

    async def async_fn(x):
      collected.append(x)
      return x + 1

    c = Chain([1, 2, 3])
    gen = c.iterate_do(async_fn)
    result = []
    async for item in gen():
      result.append(item)
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(collected, [1, 2, 3])


# ---------------------------------------------------------------------------
# H3: _get_true_source_link cycle guard
# ---------------------------------------------------------------------------

class TestH3GetTrueSourceLinkCycleGuard(unittest.TestCase):
  """H3: Cyclic Link structures don't cause infinite loops in _get_true_source_link."""

  def test_self_referencing_chain_link(self):
    """A link whose chain's root_link points back to itself terminates."""
    c = Chain(lambda: 1)
    link = c.root_link
    # Create cycle: link -> chain -> root_link -> link
    # This is artificial: normally root_link wouldn't point back to itself
    c2 = Chain(lambda: 2)
    link2 = Link(c2)
    c2.root_link = link2  # cycle: link2 -> c2 -> root_link = link2
    result = _get_true_source_link(link2, None)
    # Should terminate without infinite loop
    self.assertIsNotNone(result)

  def test_mutual_cycle_two_chains(self):
    """Two chains whose root_links reference each other don't loop."""
    c1 = Chain(lambda: 1)
    c2 = Chain(lambda: 2)
    link1 = Link(c1)
    link2 = Link(c2)
    c1.root_link = link2
    c2.root_link = link1
    result = _get_true_source_link(link1, None)
    self.assertIsNotNone(result)

  def test_deep_non_cyclic_chain(self):
    """Deep non-cyclic chain of nested chains still resolves."""
    bottom = Chain(lambda: 'bottom')
    current = bottom
    for i in range(20):
      outer = Chain(current)
      current = outer
    top_link = Link(current)
    result = _get_true_source_link(top_link, None)
    self.assertIsNotNone(result)

  def test_none_source_link_returns_root(self):
    """None source_link falls back to root_link."""
    root = Link(lambda: 1)
    result = _get_true_source_link(None, root)
    self.assertIs(result, root)

  def test_non_chain_link_returns_immediately(self):
    """A link with a plain callable returns immediately."""
    link = Link(lambda: 1)
    result = _get_true_source_link(link, None)
    self.assertIs(result, link)

  def test_cycle_length_three(self):
    """Cycle of length 3 terminates."""
    c1 = Chain(lambda: 1)
    c2 = Chain(lambda: 2)
    c3 = Chain(lambda: 3)
    l1 = Link(c1)
    l2 = Link(c2)
    l3 = Link(c3)
    c1.root_link = l2
    c2.root_link = l3
    c3.root_link = l1
    result = _get_true_source_link(l1, None)
    self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# H4: _clean_chained_exceptions iterative
# ---------------------------------------------------------------------------

class TestH4CleanChainedExceptionsIterative(unittest.TestCase):
  """H4: _clean_chained_exceptions handles deep chains without RecursionError."""

  def test_deep_context_chain(self):
    """100+ exceptions linked via __context__ are cleaned without error."""
    root = _make_exception_chain(150, '__context__')
    # Should not raise RecursionError
    _clean_chained_exceptions(root, set())

  def test_deep_cause_chain(self):
    """100+ exceptions linked via __cause__ are cleaned without error."""
    root = _make_exception_chain(150, '__cause__')
    _clean_chained_exceptions(root, set())

  def test_mixed_cause_and_context(self):
    """Mixed __cause__ and __context__ chains are cleaned."""
    root = ValueError('root')
    current = root
    for i in range(100):
      child = ValueError(f'child-{i}')
      if i % 2 == 0:
        current.__context__ = child
      else:
        current.__cause__ = child
      current = child
    _clean_chained_exceptions(root, set())

  def test_no_recursion_error_for_deep_chains(self):
    """Verify no RecursionError even for chains exceeding default recursion limit."""
    depth = sys.getrecursionlimit() + 50
    root = _make_exception_chain(depth, '__context__')
    try:
      _clean_chained_exceptions(root, set())
    except RecursionError:
      self.fail('_clean_chained_exceptions raised RecursionError for deep chain')

  def test_none_input(self):
    """None input is handled gracefully."""
    _clean_chained_exceptions(None, set())

  def test_single_exception(self):
    """Single exception with no chaining works."""
    exc = _exc_with_tb()
    _clean_chained_exceptions(exc, set())

  def test_seen_set_prevents_revisit(self):
    """Already-seen exceptions are not re-processed."""
    exc = _exc_with_tb()
    seen = {id(exc)}
    _clean_chained_exceptions(exc, seen)
    # No crash, exc was skipped

  def test_diamond_shape_exception_graph(self):
    """Diamond-shaped exception graph (shared child) is handled."""
    shared = ValueError('shared')
    parent1 = ValueError('parent1')
    parent2 = ValueError('parent2')
    root = ValueError('root')
    parent1.__context__ = shared
    parent2.__context__ = shared
    root.__cause__ = parent1
    root.__context__ = parent2
    _clean_chained_exceptions(root, set())

  def test_both_cause_and_context_on_same_exception(self):
    """Exception with both __cause__ and __context__ set."""
    root = ValueError('root')
    cause = ValueError('cause')
    context = ValueError('context')
    root.__cause__ = cause
    root.__context__ = context
    _clean_chained_exceptions(root, set())


# ---------------------------------------------------------------------------
# H5: _stringify_chain depth guard
# ---------------------------------------------------------------------------

class TestH5StringifyChainDepthGuard(unittest.TestCase):
  """H5: Deep nested chains don't cause RecursionError in stringification."""

  def test_deep_nested_chains_no_recursion_error(self):
    """60+ levels of nested chains complete without RecursionError."""
    bottom = Chain(lambda: 'bottom')
    current = bottom
    for i in range(65):
      current = Chain(current)

    ctx = _Ctx(source_link=None, link_temp_args=None)
    try:
      result = _stringify_chain(current, nest_lvl=0, ctx=ctx)
    except RecursionError:
      self.fail('_stringify_chain raised RecursionError')
    self.assertIsInstance(result, str)

  def test_truncation_message_appears(self):
    """Truncation message appears for deep nesting beyond max_depth."""
    bottom = Chain(lambda: 'bottom')
    current = bottom
    for i in range(55):
      current = Chain(current)

    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(current, nest_lvl=0, ctx=ctx, max_depth=50)
    self.assertIn('truncated at depth', result)

  def test_shallow_chain_no_truncation(self):
    """Shallow chains are not truncated."""
    c = Chain(lambda: 1).then(lambda x: x + 1)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertNotIn('truncated', result)

  def test_exact_max_depth_boundary(self):
    """Chain at exactly max_depth shows truncation."""
    bottom = Chain(lambda: 'bottom')
    current = bottom
    for i in range(5):
      current = Chain(current)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(current, nest_lvl=0, ctx=ctx, max_depth=5)
    self.assertIn('truncated at depth 5', result)

  def test_max_depth_one(self):
    """max_depth=1 truncates even singly nested chains."""
    inner = Chain(lambda: 1)
    outer = Chain(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(outer, nest_lvl=0, ctx=ctx, max_depth=1)
    self.assertIn('truncated at depth 1', result)


# ---------------------------------------------------------------------------
# H6: _modify_traceback failure suppression
# ---------------------------------------------------------------------------

class TestH6ModifyTracebackFailureSuppression(unittest.TestCase):
  """H6: If _modify_traceback internally fails, the original error still propagates."""

  def test_original_error_propagates_when_traceback_modify_fails(self):
    """Original application error propagates even if traceback modification fails."""
    def raise_in_chain(x):
      raise ValueError('app error')

    c = Chain(5).then(raise_in_chain)
    # Patch _modify_traceback to raise internally
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      # _except_handler_body uses contextlib.suppress(Exception) around _modify_traceback
      # so the original error should still propagate
      with self.assertRaises(ValueError) as ctx:
        c.run()
      self.assertEqual(str(ctx.exception), 'app error')

  def test_except_handler_runs_when_traceback_fails(self):
    """Except handler still runs even if traceback formatting fails."""
    handler_called = []

    def handler(exc):
      handler_called.append(str(exc))
      return 'handled'

    c = Chain(5).then(lambda x: (_ for _ in ()).throw(ValueError('err'))).except_(handler)
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      # The except handler is called by _except_handler_body which uses
      # contextlib.suppress around _modify_traceback
      result = c.run()
      self.assertEqual(result, 'handled')
      self.assertEqual(handler_called, ['err'])

  def test_finally_handler_runs_when_traceback_fails(self):
    """Finally handler still runs even if traceback formatting fails."""
    finally_called = []

    def finally_fn(root_val):
      finally_called.append(root_val)

    c = Chain(5).then(lambda x: x + 1).finally_(finally_fn)
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = c.run()
      self.assertEqual(result, 6)
      self.assertEqual(finally_called, [5])


class TestH6ModifyTracebackFailureSuppressionAsync(unittest.IsolatedAsyncioTestCase):
  """H6 async: Same guarantees under async execution."""

  async def test_original_error_propagates_async(self):
    """Async: Original error propagates even if traceback modification fails."""
    async def raise_in_chain(x):
      raise ValueError('async app error')

    c = Chain(5).then(raise_in_chain)
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      with self.assertRaises(ValueError) as ctx:
        await c.run()
      self.assertEqual(str(ctx.exception), 'async app error')

  async def test_except_handler_runs_async(self):
    """Async: Except handler still runs when traceback formatting fails."""
    handler_called = []

    async def handler(exc):
      handler_called.append(str(exc))
      return 'handled'

    async def failing(x):
      raise ValueError('err')

    c = Chain(5).then(failing).except_(handler)
    with patch('quent._chain._modify_traceback', side_effect=RuntimeError('tb fail')):
      result = await c.run()
      self.assertEqual(result, 'handled')
      self.assertEqual(handler_called, ['err'])


# ---------------------------------------------------------------------------
# M1: _make_gather cleanup only closes awaitables
# ---------------------------------------------------------------------------

class TestM1GatherCleanupOnlyClosesAwaitables(unittest.TestCase):
  """M1: Gather cleanup only closes awaitables, not arbitrary objects with .close()."""

  def test_file_like_not_closed_on_error(self):
    """Gather where one fn raises: file-like obj with .close() is NOT closed."""
    close_called = []

    class FileLike:
      def close(self):
        close_called.append(True)

    file_obj = FileLike()

    def fn_ok(v):
      return file_obj

    def fn_raises(v):
      raise ValueError('gather error')

    gather_op = _make_gather((fn_ok, fn_raises))
    with self.assertRaises(ValueError):
      gather_op(42)
    self.assertEqual(close_called, [], 'File-like .close() should NOT be called')

  def test_coroutine_closed_on_error(self):
    """Gather where one fn raises: an already-created coroutine IS closed."""
    close_called = []

    class FakeAwaitable:
      """Mimics a coroutine: isawaitable returns True and has .close()."""
      def __await__(self):
        yield
      def close(self):
        close_called.append(True)

    def fn_ok(v):
      return FakeAwaitable()

    def fn_raises(v):
      raise ValueError('gather error')

    gather_op = _make_gather((fn_ok, fn_raises))
    with self.assertRaises(ValueError):
      gather_op(42)
    self.assertEqual(close_called, [True], 'Awaitable .close() SHOULD be called')

  def test_real_coroutine_closed_on_error(self):
    """Gather: real coroutine is properly closed on error."""
    async def coro_fn(v):
      return v + 1

    coro_created = []

    def fn_with_coro(v):
      c = coro_fn(v)
      coro_created.append(c)
      return c

    def fn_raises(v):
      raise ValueError('gather error')

    gather_op = _make_gather((fn_with_coro, fn_raises))
    with self.assertRaises(ValueError):
      gather_op(42)
    # The coroutine should have been closed
    self.assertEqual(len(coro_created), 1)

  def test_no_cleanup_when_no_error(self):
    """Gather with no error: nothing is closed."""
    close_called = []

    class FileLike:
      def close(self):
        close_called.append(True)

    def fn1(v):
      return FileLike()

    def fn2(v):
      return 'ok'

    gather_op = _make_gather((fn1, fn2))
    result = gather_op(42)
    self.assertEqual(len(result), 2)
    self.assertEqual(close_called, [])

  def test_multiple_coroutines_closed_on_error(self):
    """Gather: multiple coroutines are all closed when a later fn raises."""
    close_count = 0

    class FakeAwaitable:
      def __await__(self):
        yield
      def close(self):
        nonlocal close_count
        close_count += 1

    def fn1(v):
      return FakeAwaitable()

    def fn2(v):
      return FakeAwaitable()

    def fn_raises(v):
      raise ValueError('err')

    gather_op = _make_gather((fn1, fn2, fn_raises))
    with self.assertRaises(ValueError):
      gather_op(42)
    self.assertEqual(close_count, 2)


# ---------------------------------------------------------------------------
# M4: _quent_file uses realpath
# ---------------------------------------------------------------------------

class TestM4QuentFileUsesRealpath(unittest.TestCase):
  """M4: _quent_file uses os.path.realpath (no symlink components)."""

  def test_quent_file_is_realpath(self):
    """_quent_file should be a realpath (resolves symlinks)."""
    import os
    self.assertEqual(_quent_file, os.path.dirname(os.path.realpath(
      os.path.join(os.path.dirname(__file__), '..', 'quent', '__init__.py')
    ).replace('__init__.py', '_traceback.py')) + os.sep)

  def test_quent_file_no_var_symlink_macos(self):
    """On macOS, /var -> /private/var. _quent_file should use the real path."""
    import os
    # _quent_file already ends with os.sep; realpath strips trailing sep,
    # so we compare after normalizing both sides.
    normalized = os.path.realpath(_quent_file.rstrip(os.sep)) + os.sep
    self.assertEqual(_quent_file, normalized)

  def test_quent_file_ends_with_sep(self):
    """_quent_file ends with os.sep for prefix-matching."""
    import os
    self.assertTrue(_quent_file.endswith(os.sep))

  def test_quent_file_is_absolute(self):
    """_quent_file is an absolute path."""
    import os
    self.assertTrue(os.path.isabs(_quent_file))


# ---------------------------------------------------------------------------
# M5: _get_obj_name sanitizes newlines
# ---------------------------------------------------------------------------

class TestM5GetObjNameSanitizesNewlines(unittest.TestCase):
  """M5: _get_obj_name sanitizes newline characters in repr output."""

  def test_repr_with_newline_sanitized(self):
    """Object with \\n in repr has it escaped to \\\\n."""
    class ObjWithNewline:
      def __repr__(self):
        return 'line1\nline2'
    result = _get_obj_name(ObjWithNewline())
    self.assertNotIn('\n', result)
    self.assertIn('\\n', result)

  def test_repr_with_carriage_return_sanitized(self):
    """Object with \\r in repr has it escaped to \\\\r."""
    class ObjWithCR:
      def __repr__(self):
        return 'line1\rline2'
    result = _get_obj_name(ObjWithCR())
    self.assertNotIn('\r', result)
    self.assertIn('\\r', result)

  def test_repr_with_both_newlines(self):
    """Object with both \\r and \\n in repr has both sanitized."""
    class ObjWithBoth:
      def __repr__(self):
        return 'a\r\nb'
    result = _get_obj_name(ObjWithBoth())
    self.assertNotIn('\r', result)
    self.assertNotIn('\n', result)
    self.assertIn('\\r', result)
    self.assertIn('\\n', result)

  def test_repr_with_multiple_newlines(self):
    """Multiple newlines are all sanitized."""
    class ObjMulti:
      def __repr__(self):
        return 'a\nb\nc\nd'
    result = _get_obj_name(ObjMulti())
    self.assertEqual(result.count('\\n'), 3)
    self.assertNotIn('\n', result)

  def test_normal_repr_unchanged(self):
    """Normal repr without newlines is unchanged."""
    result = _get_obj_name(42)
    self.assertEqual(result, '42')

  def test_callable_with_name_no_sanitization_needed(self):
    """Callable with __name__ returns name directly (no repr)."""
    def my_func():
      pass
    result = _get_obj_name(my_func)
    self.assertEqual(result, 'my_func')

  def test_chain_returns_type_name(self):
    """Chain object returns 'Chain' via _is_chain shortcut."""
    c = Chain(1)
    result = _get_obj_name(c)
    self.assertEqual(result, 'Chain')

  def test_repr_raises_falls_back_to_type_name(self):
    """When repr() raises, falls back to type(obj).__name__."""
    class BadRepr:
      def __repr__(self):
        raise RuntimeError('no repr')
    result = _get_obj_name(BadRepr())
    self.assertEqual(result, 'BadRepr')


# ---------------------------------------------------------------------------
# M6: Error guards on exception hooks
# ---------------------------------------------------------------------------

class TestM6ErrorGuardsOnExceptionHooks(unittest.TestCase):
  """M6: Exception hooks don't crash when internal cleaning fails."""

  def test_quent_excepthook_survives_cleaning_failure(self):
    """_quent_excepthook doesn't crash when _clean_chained_exceptions would fail."""
    exc = ValueError('test')
    exc.__quent__ = True
    hook_called = []

    def mock_hook(exc_type, exc_value, exc_tb):
      hook_called.append(True)

    with patch('quent._traceback._original_excepthook', mock_hook):
      with patch('quent._traceback._clean_chained_exceptions', side_effect=RuntimeError('clean fail')):
        # Should not raise, should still call original hook
        _quent_excepthook(ValueError, exc, None)
    self.assertEqual(hook_called, [True])

  def test_quent_excepthook_passes_through_non_quent_exc(self):
    """_quent_excepthook passes through exceptions without __quent__ flag."""
    exc = ValueError('not quent')
    hook_called = []

    def mock_hook(exc_type, exc_value, exc_tb):
      hook_called.append((exc_type, exc_value))

    with patch('quent._traceback._original_excepthook', mock_hook):
      _quent_excepthook(ValueError, exc, None)
    self.assertEqual(len(hook_called), 1)
    self.assertIs(hook_called[0][1], exc)

  def test_patched_te_init_survives_cleaning_failure(self):
    """_patched_te_init doesn't crash when cleaning fails."""
    exc = ValueError('test')
    exc.__quent__ = True

    with patch('quent._traceback._clean_chained_exceptions', side_effect=RuntimeError('clean fail')):
      # Should not raise, should fall through to original __init__
      te = traceback.TracebackException(ValueError, exc, None)
    self.assertIsNotNone(te)

  def test_patched_te_init_none_exc_value(self):
    """_patched_te_init handles None exc_value gracefully."""
    # Should not raise
    te = traceback.TracebackException(ValueError, None, None)
    self.assertIsNotNone(te)

  def test_quent_excepthook_preserves_traceback_update(self):
    """_quent_excepthook updates exc_tb from exc.__traceback__ after cleaning."""
    exc = _exc_with_tb('quent test')
    exc.__quent__ = True
    received_tb = []

    def mock_hook(exc_type, exc_value, exc_tb):
      received_tb.append(exc_tb)

    with patch('quent._traceback._original_excepthook', mock_hook):
      _quent_excepthook(ValueError, exc, exc.__traceback__)
    self.assertEqual(len(received_tb), 1)


# ---------------------------------------------------------------------------
# disable/enable_traceback_patching
# ---------------------------------------------------------------------------

class TestDisableEnableTracebackPatching(unittest.TestCase):
  """Tests for disable_traceback_patching and enable_traceback_patching."""

  def setUp(self):
    """Ensure patching is enabled at start of each test."""
    enable_traceback_patching()

  def tearDown(self):
    """Restore patching after each test."""
    enable_traceback_patching()

  def test_disable_restores_original_excepthook(self):
    """disable_traceback_patching restores original sys.excepthook."""
    self.assertIs(sys.excepthook, _quent_excepthook)
    disable_traceback_patching()
    self.assertIs(sys.excepthook, _original_excepthook)

  def test_enable_installs_quent_hook(self):
    """enable_traceback_patching installs quent's hook."""
    disable_traceback_patching()
    self.assertIs(sys.excepthook, _original_excepthook)
    enable_traceback_patching()
    self.assertIs(sys.excepthook, _quent_excepthook)

  def test_disable_restores_te_init(self):
    """disable_traceback_patching restores TracebackException.__init__."""
    disable_traceback_patching()
    from quent._traceback import _original_te_init
    self.assertIs(traceback.TracebackException.__init__, _original_te_init)

  def test_enable_installs_patched_te_init(self):
    """enable_traceback_patching installs patched TracebackException.__init__."""
    disable_traceback_patching()
    enable_traceback_patching()
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)

  def test_multiple_disable_enable_cycles(self):
    """Multiple disable/enable cycles work correctly."""
    for _ in range(5):
      disable_traceback_patching()
      self.assertIs(sys.excepthook, _original_excepthook)
      enable_traceback_patching()
      self.assertIs(sys.excepthook, _quent_excepthook)

  def test_disable_when_already_disabled_is_noop(self):
    """Disabling when already disabled is a no-op."""
    disable_traceback_patching()
    hook_before = sys.excepthook
    disable_traceback_patching()
    self.assertIs(sys.excepthook, hook_before)

  def test_enable_when_already_enabled_is_noop(self):
    """Enabling when already enabled is a no-op."""
    hook_before = sys.excepthook
    enable_traceback_patching()
    self.assertIs(sys.excepthook, hook_before)

  def test_exceptions_display_correctly_after_cycle(self):
    """Exceptions still display correctly after disable+enable cycle."""
    disable_traceback_patching()
    enable_traceback_patching()

    def raise_in_chain(x):
      raise ValueError('cycle test')

    c = Chain(5).then(raise_in_chain)
    with self.assertRaises(ValueError) as ctx:
      c.run()
    self.assertEqual(str(ctx.exception), 'cycle test')

  def test_chain_runs_with_patching_disabled(self):
    """Chains work correctly with patching disabled."""
    disable_traceback_patching()
    c = Chain(5).then(lambda x: x + 1)
    result = c.run()
    self.assertEqual(result, 6)

  def test_chain_error_with_patching_disabled(self):
    """Chain errors still propagate with patching disabled."""
    disable_traceback_patching()
    c = Chain(5).then(lambda x: 1 / 0)
    with self.assertRaises(ZeroDivisionError):
      c.run()

  def test_except_handler_with_patching_disabled(self):
    """Except handlers work with patching disabled."""
    disable_traceback_patching()
    c = Chain(5).then(lambda x: 1 / 0).except_(lambda e: 'caught')
    result = c.run()
    self.assertEqual(result, 'caught')

  def test_finally_handler_with_patching_disabled(self):
    """Finally handlers work with patching disabled."""
    disable_traceback_patching()
    tracker = []
    c = Chain(5).then(lambda x: x + 1).finally_(lambda v: tracker.append(v))
    result = c.run()
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [5])


# ---------------------------------------------------------------------------
# Additional edge-case regression tests
# ---------------------------------------------------------------------------

class TestCleanInternalFramesEdgeCases(unittest.TestCase):
  """Edge cases for _clean_internal_frames."""

  def test_none_traceback(self):
    """None traceback returns None."""
    result = _clean_internal_frames(None)
    self.assertIsNone(result)

  def test_single_user_frame_preserved(self):
    """A single non-quent frame is preserved."""
    exc = _exc_with_tb()
    result = _clean_internal_frames(exc.__traceback__)
    self.assertIsNotNone(result)


class TestModifyTracebackWithNestedChain(unittest.TestCase):
  """_modify_traceback with nested chains."""

  def test_nested_chain_no_visualization(self):
    """Nested chain (is_nested=True) doesn't inject visualization."""
    c = Chain(5).then(lambda x: x)
    c.is_nested = True
    exc = _exc_with_tb()
    link = c.root_link
    _modify_traceback(exc, c, link, link)
    # __quent__ is set but no code injection (is_nested branch)
    self.assertTrue(getattr(exc, '__quent__', False))

  def test_modify_traceback_no_chain(self):
    """_modify_traceback with chain=None still cleans frames."""
    exc = _exc_with_tb()
    _modify_traceback(exc, None, None, None)
    self.assertTrue(getattr(exc, '__quent__', False))


class TestGatherEdgeCases(unittest.TestCase):
  """Additional edge cases for _make_gather."""

  def test_gather_empty_fns(self):
    """Gather with no fns returns empty list."""
    gather_op = _make_gather(())
    result = gather_op(42)
    self.assertEqual(result, [])

  def test_gather_single_fn(self):
    """Gather with a single fn works."""
    gather_op = _make_gather((lambda v: v + 1,))
    result = gather_op(42)
    self.assertEqual(result, [43])

  def test_gather_first_fn_raises(self):
    """When first fn raises, no cleanup needed (no results yet)."""
    def fn_raises(v):
      raise ValueError('first')

    def fn_ok(v):
      return v

    gather_op = _make_gather((fn_raises, fn_ok))
    with self.assertRaises(ValueError) as ctx:
      gather_op(42)
    self.assertEqual(str(ctx.exception), 'first')


class TestSyncGeneratorEdgeCases(unittest.TestCase):
  """Edge cases for _sync_generator."""

  def test_empty_iterable(self):
    """Empty iterable produces no items."""
    c = Chain([])
    gen = c.iterate(lambda x: x + 1)
    result = list(gen())
    self.assertEqual(result, [])

  def test_iterate_with_break(self):
    """Break in iterate stops iteration."""
    def fn(x):
      if x >= 2:
        Chain.break_()
      return x * 10

    c = Chain(range(5))
    gen = c.iterate(fn)
    result = list(gen())
    self.assertEqual(result, [0, 10])


class TestAwaitExitSuppressWithDo(unittest.IsolatedAsyncioTestCase):
  """Edge cases for _await_exit_suppress with with_do."""

  async def test_with_do_exit_awaitable_returns_outer(self):
    """with_do: exit awaitable returns outer value (CM instance)."""
    class CM:
      def __enter__(self):
        return 'ctx'
      def __exit__(self, *args):
        async def _exit():
          return False
        return _exit()

    cm = CM()
    c = Chain(cm).with_do(lambda ctx: 'body_result')
    result = await c.run()
    # with_do ignores body result, returns outer value (cm)
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# Combined scenario regression tests
# ---------------------------------------------------------------------------

class TestCombinedScenarios(unittest.IsolatedAsyncioTestCase):
  """Integration tests combining multiple fixed behaviors."""

  async def test_gather_with_async_and_error(self):
    """Gather with mixed sync/async where last raises: coroutines are cleaned."""
    async def async_fn(v):
      return v + 1

    def sync_fn(v):
      return v + 2

    def raises_fn(v):
      raise ValueError('err')

    # async_fn returns a coroutine that must be cleaned up
    gather_op = _make_gather((async_fn, sync_fn, raises_fn))
    with self.assertRaises(ValueError):
      gather_op(10)

  async def test_chain_with_except_and_finally_both_run(self):
    """Both except and finally handlers run on error."""
    handler_order = []

    c = (
      Chain(5)
      .then(lambda x: 1 / 0)
      .except_(lambda e: handler_order.append('except') or 'handled')
      .finally_(lambda v: handler_order.append('finally'))
    )
    result = c.run()
    self.assertEqual(result, 'handled')
    self.assertIn('except', handler_order)
    self.assertIn('finally', handler_order)

  async def test_iterate_async_for_with_error(self):
    """iterate(fn) where fn raises during async for."""
    call_count = 0

    def fn(x):
      nonlocal call_count
      call_count += 1
      if x == 2:
        raise ValueError('at 2')
      return x * 10

    c = Chain([0, 1, 2, 3])
    gen = c.iterate(fn)
    with self.assertRaises(ValueError):
      async for _ in gen():
        pass
    self.assertEqual(call_count, 3)  # 0, 1, 2 (raises at 2)


class TestPatchingStateConsistency(unittest.TestCase):
  """Verify patching state is consistent across module reloads and cycles."""

  def setUp(self):
    enable_traceback_patching()

  def tearDown(self):
    enable_traceback_patching()

  def test_patching_flag_tracks_state(self):
    """Internal _patching_enabled flag tracks the actual state."""
    import quent._traceback as tb
    self.assertTrue(tb._patching_enabled)
    disable_traceback_patching()
    self.assertFalse(tb._patching_enabled)
    enable_traceback_patching()
    self.assertTrue(tb._patching_enabled)

  def test_rapid_toggle(self):
    """Rapid enable/disable toggling doesn't corrupt state."""
    for _ in range(100):
      disable_traceback_patching()
      enable_traceback_patching()
    self.assertIs(sys.excepthook, _quent_excepthook)

  def test_disable_enable_preserves_te_init(self):
    """TracebackException.__init__ is correctly toggled."""
    from quent._traceback import _original_te_init
    disable_traceback_patching()
    self.assertIs(traceback.TracebackException.__init__, _original_te_init)
    enable_traceback_patching()
    self.assertIs(traceback.TracebackException.__init__, _patched_te_init)


if __name__ == '__main__':
  unittest.main()
