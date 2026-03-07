"""Full cross-product tests for control flow signals x operations x handlers x nesting.

Tests the complete matrix of:
  - Control flow signals: return_(value), return_(), break_(value), break_()
  - Operations: then, do, map, foreach, filter, with_, with_do, gather, iterate, nested chain
  - Handler contexts: none, except_ only, finally_ only, both
  - Nesting levels: 1, 2, 3
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from quent._core import _Return, _Break, _ControlFlowSignal
from helpers import (
  SyncCM,
  AsyncCM,
  TrackingCM,
  AsyncTrackingCM,
  AsyncRange,
  make_tracker,
  make_async_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop(x):
  return x


def _make_return_fn(has_value):
  """Return a callable that triggers Chain.return_ with or without a value."""
  if has_value:
    return lambda x: Chain.return_(42)
  return lambda x: Chain.return_()


def _make_break_fn(has_value, at_index=1):
  """Return a callable for map that breaks at the given index."""
  if has_value:
    return lambda x: Chain.break_(42) if x == at_index else x
  return lambda x: Chain.break_() if x == at_index else x


# ---------------------------------------------------------------------------
# TestReturnInEveryOperation
# ---------------------------------------------------------------------------

class TestReturnInEveryOperation(unittest.TestCase):
  """Chain.return_(42) and Chain.return_() in each operation x handler context."""

  def _build_chain_with_return(self, operation, has_value, has_except, has_finally):
    """Build a chain that triggers return_ in the given operation."""
    return_fn = _make_return_fn(has_value)
    tracker = make_tracker()
    except_tracker = make_tracker()
    finally_tracker = make_tracker()

    if operation == 'then':
      c = Chain(10).then(return_fn)
    elif operation == 'do':
      c = Chain(10).do(return_fn)
    elif operation == 'map':
      c = Chain([1, 2, 3]).map(lambda x: Chain.return_(42) if has_value and x == 2 else (Chain.return_() if not has_value and x == 2 else x))
    elif operation == 'foreach':
      c = Chain([1, 2, 3]).foreach(lambda x: Chain.return_(42) if has_value and x == 2 else (Chain.return_() if not has_value and x == 2 else x))
    elif operation == 'filter':
      c = Chain([1, 2, 3]).filter(lambda x: Chain.return_(42) if has_value and x == 2 else (Chain.return_() if not has_value and x == 2 else True))
    elif operation == 'with_':
      cm = SyncCM()
      c = Chain(cm).with_(return_fn)
    elif operation == 'with_do':
      cm = SyncCM()
      c = Chain(cm).with_do(return_fn)
    elif operation == 'gather':
      c = Chain(10).gather(return_fn)
    elif operation == 'nested':
      inner = Chain().then(return_fn)
      c = Chain(10).then(inner)
    else:
      raise ValueError(f'Unknown operation: {operation}')

    # Always add a trailing step to prove early-exit
    c = c.then(lambda x: tracker('should_not_run'))

    if has_except:
      c = c.except_(except_tracker)
    if has_finally:
      c = c.finally_(finally_tracker)

    return c, tracker, except_tracker, finally_tracker

  def test_return_matrix(self):
    operations = ['then', 'do', 'map', 'foreach', 'filter', 'with_', 'with_do', 'gather', 'nested']
    for operation in operations:
      for has_value in [True, False]:
        for has_except in [True, False]:
          for has_finally in [True, False]:
            with self.subTest(
              operation=operation, has_value=has_value,
              has_except=has_except, has_finally=has_finally
            ):
              c, tracker, except_tracker, finally_tracker = self._build_chain_with_return(
                operation, has_value, has_except, has_finally
              )
              result = c.run()

              # Verify return value
              if has_value:
                self.assertEqual(result, 42, f'Expected 42 for {operation}')
              else:
                self.assertIsNone(result, f'Expected None for {operation}')

              # Trailing step must NOT have run
              self.assertEqual(tracker.calls, [], f'Trailing step should not run for {operation}')

              # except_ must NOT have been called
              self.assertEqual(except_tracker.calls, [], f'except_ should not run for {operation} return')

              # finally_ MUST have been called if registered
              if has_finally:
                self.assertEqual(
                  len(finally_tracker.calls), 1,
                  f'finally_ should run once for {operation}'
                )


# ---------------------------------------------------------------------------
# TestReturnWithValueVariants
# ---------------------------------------------------------------------------

class TestReturnWithValueVariants(unittest.TestCase):
  """Return with different value types and contexts."""

  def test_return_literal(self):
    for context in ['then', 'nested']:
      with self.subTest(context=context, value='literal_42'):
        if context == 'then':
          result = Chain(1).then(lambda x: Chain.return_(42)).run()
        else:
          inner = Chain().then(lambda x: Chain.return_(42))
          result = Chain(1).then(inner).run()
        self.assertEqual(result, 42)

  def test_return_callable_resolved(self):
    for context in ['then', 'nested']:
      with self.subTest(context=context, value='callable'):
        if context == 'then':
          result = Chain(1).then(lambda x: Chain.return_(lambda: 42)).run()
        else:
          inner = Chain().then(lambda x: Chain.return_(lambda: 42))
          result = Chain(1).then(inner).run()
        self.assertEqual(result, 42)

  def test_return_callable_with_ellipsis(self):
    for context in ['then', 'nested']:
      with self.subTest(context=context, value='callable_ellipsis'):
        if context == 'then':
          result = Chain(1).then(lambda x: Chain.return_(lambda: 42, ...)).run()
        else:
          inner = Chain().then(lambda x: Chain.return_(lambda: 42, ...))
          result = Chain(1).then(inner).run()
        self.assertEqual(result, 42)

  def test_return_none_value(self):
    for context in ['then', 'nested']:
      with self.subTest(context=context, value='None'):
        if context == 'then':
          result = Chain(1).then(lambda x: Chain.return_(None)).run()
        else:
          inner = Chain().then(lambda x: Chain.return_(None))
          result = Chain(1).then(inner).run()
        # return_(None) passes None explicitly; _resolve_value returns None
        self.assertIsNone(result)

  def test_return_false_value(self):
    for context in ['then', 'nested']:
      with self.subTest(context=context, value='False'):
        if context == 'then':
          result = Chain(1).then(lambda x: Chain.return_(False)).run()
        else:
          inner = Chain().then(lambda x: Chain.return_(False))
          result = Chain(1).then(inner).run()
        self.assertIs(result, False)

  def test_return_zero_value(self):
    for context in ['then', 'nested']:
      with self.subTest(context=context, value='zero'):
        if context == 'then':
          result = Chain(1).then(lambda x: Chain.return_(0)).run()
        else:
          inner = Chain().then(lambda x: Chain.return_(0))
          result = Chain(1).then(inner).run()
        self.assertEqual(result, 0)

  def test_return_callable_with_args(self):
    for context in ['then', 'nested']:
      with self.subTest(context=context, value='callable_with_args'):
        if context == 'then':
          result = Chain(1).then(lambda x: Chain.return_(lambda a, b: a + b, 10, 20)).run()
        else:
          inner = Chain().then(lambda x: Chain.return_(lambda a, b: a + b, 10, 20))
          result = Chain(1).then(inner).run()
        self.assertEqual(result, 30)


# ---------------------------------------------------------------------------
# TestBreakInMap
# ---------------------------------------------------------------------------

class TestBreakInMap(unittest.TestCase):
  """Break in every position of map with and without value, with/without handlers."""

  def test_break_position_matrix(self):
    for iterable_size in [1, 3, 5, 10]:
      items = list(range(iterable_size))
      positions = [0, 1]
      if iterable_size > 2:
        positions.append(iterable_size // 2)
      if iterable_size > 1:
        positions.append(iterable_size - 1)
      # Deduplicate
      positions = sorted(set(p for p in positions if p < iterable_size))

      for break_pos in positions:
        for has_value in [True, False]:
          for has_except in [True, False]:
            for has_finally in [True, False]:
              with self.subTest(
                iterable_size=iterable_size, break_position=break_pos,
                has_value=has_value, has_except=has_except, has_finally=has_finally
              ):
                except_tracker = make_tracker()
                finally_tracker = make_tracker()

                if has_value:
                  fn = lambda x, bp=break_pos: Chain.break_(42) if x == bp else x * 2
                else:
                  fn = lambda x, bp=break_pos: Chain.break_() if x == bp else x * 2

                c = Chain(items).map(fn)
                if has_except:
                  c = c.except_(except_tracker)
                if has_finally:
                  c = c.finally_(finally_tracker)

                result = c.run()

                if has_value:
                  self.assertEqual(result, 42)
                else:
                  # Accumulated results before break
                  expected = [x * 2 for x in items[:break_pos]]
                  self.assertEqual(result, expected)

                # except_ must NOT be called
                self.assertEqual(except_tracker.calls, [])

                # finally_ must be called if registered
                if has_finally:
                  self.assertEqual(len(finally_tracker.calls), 1)

  def test_break_value_variants(self):
    """Break with different value types: 42, None, False, 0, callable."""
    variants = [
      ('literal_42', 42, 42),
      ('none', None, None),
      ('false', False, False),
      ('zero', 0, 0),
      ('callable', lambda: 'done', 'done'),
    ]
    for label, break_val, expected in variants:
      with self.subTest(break_value=label):
        result = (
          Chain([1, 2, 3])
          .map(lambda x, bv=break_val: Chain.break_(bv) if x == 2 else x)
          .run()
        )
        self.assertEqual(result, expected)


# ---------------------------------------------------------------------------
# TestBreakInForeach
# ---------------------------------------------------------------------------

class TestBreakInForeach(unittest.TestCase):
  """Break in foreach -- break behavior preserving original items."""

  def test_foreach_break_matrix(self):
    for iterable_size in [1, 3, 5]:
      items = list(range(iterable_size))
      positions = sorted(set(p for p in [0, 1, iterable_size // 2, iterable_size - 1] if p < iterable_size))

      for break_pos in positions:
        for has_value in [True, False]:
          for has_except in [True, False]:
            for has_finally in [True, False]:
              with self.subTest(
                iterable_size=iterable_size, break_position=break_pos,
                has_value=has_value, has_except=has_except, has_finally=has_finally
              ):
                except_tracker = make_tracker()
                finally_tracker = make_tracker()
                side_effects = []

                if has_value:
                  fn = lambda x, bp=break_pos, se=side_effects: (se.append(x), Chain.break_(42))[-1] if x == bp else se.append(x) or x
                else:
                  fn = lambda x, bp=break_pos, se=side_effects: (se.append(x), Chain.break_())[-1] if x == bp else se.append(x) or x

                c = Chain(items).foreach(fn)
                if has_except:
                  c = c.except_(except_tracker)
                if has_finally:
                  c = c.finally_(finally_tracker)

                result = c.run()

                if has_value:
                  self.assertEqual(result, 42)
                else:
                  # foreach preserves original items
                  expected = items[:break_pos]
                  self.assertEqual(result, expected)

                self.assertEqual(except_tracker.calls, [])

                if has_finally:
                  self.assertEqual(len(finally_tracker.calls), 1)


# ---------------------------------------------------------------------------
# TestBreakOutsideMap
# ---------------------------------------------------------------------------

class TestBreakOutsideMap(unittest.TestCase):
  """Break in invalid contexts must raise QuentException."""

  def test_break_in_then(self):
    with self.subTest(operation='then'):
      with self.assertRaises(QuentException) as ctx:
        Chain(5).then(lambda x: Chain.break_()).run()
      self.assertIn('cannot be used outside', str(ctx.exception).lower())

  def test_break_in_do(self):
    with self.subTest(operation='do'):
      with self.assertRaises(QuentException) as ctx:
        Chain(5).do(lambda x: Chain.break_()).run()
      self.assertIn('cannot be used outside', str(ctx.exception).lower())

  def test_break_in_gather(self):
    with self.subTest(operation='gather'):
      with self.assertRaises(QuentException) as ctx:
        Chain(5).gather(lambda x: Chain.break_()).run()
      self.assertIn('cannot be used outside', str(ctx.exception).lower())

  def test_break_in_with(self):
    with self.subTest(operation='with_'):
      with self.assertRaises(QuentException) as ctx:
        Chain(SyncCM()).with_(lambda ctx: Chain.break_()).run()
      self.assertIn('cannot be used outside', str(ctx.exception).lower())

  def test_break_in_run(self):
    with self.subTest(operation='run'):
      with self.assertRaises(QuentException) as ctx:
        Chain(lambda: Chain.break_()).run()
      self.assertIn('cannot be used outside', str(ctx.exception).lower())

  def test_break_in_filter(self):
    # break_() inside filter() now stops iteration early and returns partial results.
    with self.subTest(operation='filter'):
      result = Chain([1, 2, 3]).filter(lambda x: Chain.break_() if x == 2 else True).run()
      self.assertEqual(result, [1])

  def test_break_in_with_do(self):
    with self.subTest(operation='with_do'):
      with self.assertRaises(QuentException) as ctx:
        Chain(SyncCM()).with_do(lambda ctx: Chain.break_()).run()
      self.assertIn('cannot be used outside', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# TestReturnThroughNesting
# ---------------------------------------------------------------------------

class TestReturnThroughNesting(unittest.TestCase):
  """Return propagation through 1, 2, 3 nesting levels."""

  def test_return_1_level(self):
    inner = Chain().then(lambda x: Chain.return_(99))
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 99)

  def test_return_2_levels(self):
    level2 = Chain().then(lambda x: Chain.return_(99))
    level1 = Chain().then(level2)
    result = Chain(5).then(level1).run()
    self.assertEqual(result, 99)

  def test_return_3_levels(self):
    level3 = Chain().then(lambda x: Chain.return_(99))
    level2 = Chain().then(level3)
    level1 = Chain().then(level2)
    result = Chain(5).then(level1).run()
    self.assertEqual(result, 99)

  def test_return_bypasses_except_at_each_level(self):
    """return_ should bypass except_ at every nesting level."""
    for depth in [1, 2, 3]:
      with self.subTest(depth=depth):
        except_trackers = []
        innermost = Chain().then(lambda x: Chain.return_(99))

        current = innermost
        for i in range(depth - 1):
          tracker = make_tracker()
          except_trackers.append(tracker)
          current = Chain().then(current).except_(tracker)

        outer_except = make_tracker()
        except_trackers.append(outer_except)
        chain = Chain(5).then(current).except_(outer_except)

        result = chain.run()
        self.assertEqual(result, 99)
        for i, t in enumerate(except_trackers):
          self.assertEqual(t.calls, [], f'except_ at level {i} should not run')

  def test_return_triggers_finally_at_each_level(self):
    """return_ should trigger finally_ at each nesting level."""
    for depth in [1, 2, 3]:
      with self.subTest(depth=depth):
        finally_trackers = []
        innermost = Chain().then(lambda x: Chain.return_(99))

        current = innermost
        for i in range(depth - 1):
          tracker = make_tracker()
          finally_trackers.append(tracker)
          wrapper = Chain().then(current).finally_(tracker)
          current = wrapper

        outer_finally = make_tracker()
        finally_trackers.append(outer_finally)
        chain = Chain(5).then(current).finally_(outer_finally)

        result = chain.run()
        self.assertEqual(result, 99)
        # The outermost finally_ is always called
        self.assertEqual(
          len(outer_finally.calls), 1,
          'Outermost finally_ should run'
        )

  def test_return_no_value_through_nesting(self):
    """return_() (no value) propagates correctly through nesting."""
    for depth in [1, 2, 3]:
      with self.subTest(depth=depth):
        innermost = Chain().then(lambda x: Chain.return_())

        current = innermost
        for _ in range(depth - 1):
          current = Chain().then(current)

        result = Chain(5).then(current).run()
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TestBreakThroughNesting
# ---------------------------------------------------------------------------

class TestBreakThroughNesting(unittest.TestCase):
  """Break propagation in nested map contexts."""

  def test_break_in_inner_map_isolated(self):
    """Break in nested chain's map does not affect outer chain."""
    inner = Chain().map(lambda x: Chain.break_() if x == 2 else x)
    result = (
      Chain([1, 2, 3, 4])
      .then(inner)
      .then(lambda x: x + [99])
      .run()
    )
    # Inner map breaks at x==2, returns [1], then outer appends 99
    self.assertEqual(result, [1, 99])

  def test_nested_map_break_inner_only(self):
    """Nested map -- break in inner doesn't affect outer."""
    def inner_fn(items):
      # Break after processing 2 items
      count = [0]
      def fn(x):
        count[0] += 1
        if count[0] > 2:
          Chain.break_()
        return x * 10
      return Chain(items).map(fn).run()

    result = (
      Chain([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
      .map(inner_fn)
      .run()
    )
    # Each inner map processes 2 items then breaks
    self.assertEqual(result, [[10, 20], [50, 60], [90, 100]])

  def test_break_with_value_in_nested_map(self):
    """Break with value in inner map."""
    inner = Chain().map(lambda x: Chain.break_(42) if x == 2 else x)
    result = Chain([1, 2, 3]).then(inner).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# TestControlFlowWithExceptInteraction
# ---------------------------------------------------------------------------

class TestControlFlowWithExceptInteraction(unittest.TestCase):
  """Interactions between control flow signals and except_ handlers."""

  def test_return_does_not_trigger_except(self):
    handler_called = []
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .except_(lambda rv, exc: handler_called.append(exc))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(handler_called, [])

  def test_break_in_foreaches_not_trigger_except(self):
    handler_called = []
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_() if x == 2 else x)
      .except_(lambda rv, exc: handler_called.append(exc))
      .run()
    )
    self.assertEqual(result, [1])
    self.assertEqual(handler_called, [])

  def test_actual_exception_triggers_except(self):
    handler_called = []
    result = (
      Chain(5)
      .then(lambda x: 1 / 0)
      .except_(lambda rv, exc: handler_called.append(type(exc).__name__) or 'handled')
      .run()
    )
    self.assertEqual(result, 'handled')
    self.assertEqual(handler_called, ['ZeroDivisionError'])

  def test_return_in_except_handler_raises_quent_exception(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: 1 / 0).except_(lambda rv, exc: Chain.return_(99)).run()
    self.assertIn('control flow signals inside except handlers is not allowed', str(ctx.exception).lower())

  def test_break_in_except_handler_raises_quent_exception(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: 1 / 0).except_(lambda rv, exc: Chain.break_(99)).run()
    self.assertIn('control flow signals inside except handlers is not allowed', str(ctx.exception).lower())

  def test_return_with_value_and_except(self):
    """Return with value when except_ is registered -- except_ not invoked."""
    except_tracker = make_tracker()
    result = (
      Chain(10)
      .then(lambda x: x + 5)
      .then(lambda x: Chain.return_(x * 2))
      .except_(except_tracker)
      .run()
    )
    self.assertEqual(result, 30)
    self.assertEqual(except_tracker.calls, [])


# ---------------------------------------------------------------------------
# TestControlFlowWithFinallyInteraction
# ---------------------------------------------------------------------------

class TestControlFlowWithFinallyInteraction(unittest.TestCase):
  """Interactions between control flow signals and finally_ handlers."""

  def test_return_in_body_triggers_finally(self):
    finally_tracker = make_tracker()
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(len(finally_tracker.calls), 1)

  def test_break_in_map_triggers_finally(self):
    finally_tracker = make_tracker()
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_() if x == 2 else x)
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, [1])
    self.assertEqual(len(finally_tracker.calls), 1)

  def test_return_in_finally_handler_raises_quent_exception(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: x).finally_(lambda x: Chain.return_(99)).run()
    self.assertIn('control flow signals inside finally handlers is not allowed', str(ctx.exception).lower())

  def test_break_in_finally_handler_raises_quent_exception(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: x).finally_(lambda x: Chain.break_()).run()
    self.assertIn('control flow signals inside finally handlers is not allowed', str(ctx.exception).lower())

  def test_finally_receives_root_value_on_return(self):
    """finally_ receives the root_value regardless of return."""
    received_values = []
    result = (
      Chain(5)
      .then(lambda x: x + 100)
      .then(lambda x: Chain.return_(42))
      .finally_(lambda root: received_values.append(root))
      .run()
    )
    self.assertEqual(result, 42)
    # The root_value is the result of evaluating the root link (Chain(5) -> 5)
    self.assertEqual(received_values, [5])

  def test_finally_receives_root_value_on_break(self):
    """finally_ receives the root_value on break."""
    received_values = []
    items = [1, 2, 3]
    result = (
      Chain(items)
      .map(lambda x: Chain.break_() if x == 2 else x)
      .finally_(lambda root: received_values.append(root))
      .run()
    )
    self.assertEqual(result, [1])
    self.assertEqual(received_values, [items])

  def test_both_except_and_finally_with_return(self):
    """return_ -> except_ NOT called, finally_ called."""
    except_tracker = make_tracker()
    finally_tracker = make_tracker()
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .except_(except_tracker)
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(except_tracker.calls, [])
    self.assertEqual(len(finally_tracker.calls), 1)

  def test_both_except_and_finally_with_break(self):
    """break_ in map -> except_ NOT called, finally_ called."""
    except_tracker = make_tracker()
    finally_tracker = make_tracker()
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_() if x == 2 else x)
      .except_(except_tracker)
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, [1])
    self.assertEqual(except_tracker.calls, [])
    self.assertEqual(len(finally_tracker.calls), 1)


# ---------------------------------------------------------------------------
# TestControlFlowAsync
# ---------------------------------------------------------------------------

class TestControlFlowAsync(IsolatedAsyncioTestCase):
  """Async variants of control flow signal tests."""

  async def test_return_in_async_then(self):
    async def async_step(x):
      return x + 1

    result = await (
      Chain(5)
      .then(async_step)
      .then(lambda x: Chain.return_(42))
      .run()
    )
    self.assertEqual(result, 42)

  async def test_return_no_value_async(self):
    async def async_step(x):
      return x + 1

    result = await (
      Chain(5)
      .then(async_step)
      .then(lambda x: Chain.return_())
      .run()
    )
    self.assertIsNone(result)

  async def test_break_in_async_map(self):
    result = await (
      Chain(AsyncRange(6))
      .map(lambda x: Chain.break_() if x == 3 else x)
      .run()
    )
    self.assertEqual(result, [0, 1, 2])

  async def test_break_with_value_in_async_map(self):
    result = await (
      Chain(AsyncRange(6))
      .map(lambda x: Chain.break_(42) if x == 3 else x)
      .run()
    )
    self.assertEqual(result, 42)

  async def test_return_through_async_nested_chain(self):
    async def async_step(x):
      return x + 1

    inner = Chain().then(async_step).then(lambda x: Chain.return_(99))
    result = await Chain(5).then(inner).run()
    self.assertEqual(result, 99)

  async def test_return_with_async_except_not_triggered(self):
    async def async_step(x):
      return x + 1

    except_tracker = make_async_tracker()
    result = await (
      Chain(5)
      .then(async_step)
      .then(lambda x: Chain.return_(42))
      .except_(except_tracker)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(except_tracker.calls, [])

  async def test_return_with_async_finally_triggered(self):
    async def async_step(x):
      return x + 1

    finally_values = []

    async def async_finally(root):
      finally_values.append(root)

    result = await (
      Chain(5)
      .then(async_step)
      .then(lambda x: Chain.return_(42))
      .finally_(async_finally)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(len(finally_values), 1)

  async def test_break_in_async_map_with_except_not_triggered(self):
    except_tracker = make_async_tracker()
    result = await (
      Chain(AsyncRange(6))
      .map(lambda x: Chain.break_() if x == 3 else x)
      .except_(except_tracker)
      .run()
    )
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(except_tracker.calls, [])

  async def test_break_in_async_map_with_finally_triggered(self):
    finally_values = []

    async def async_finally(root):
      finally_values.append('called')

    result = await (
      Chain(AsyncRange(6))
      .map(lambda x: Chain.break_() if x == 3 else x)
      .finally_(async_finally)
      .run()
    )
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(len(finally_values), 1)

  async def test_return_in_except_handler_async_raises(self):
    async def async_step(x):
      return x + 1

    with self.assertRaises(QuentException) as ctx:
      await (
        Chain(5)
        .then(async_step)
        .then(lambda x: 1 / 0)
        .except_(lambda rv, exc: Chain.return_(99))
        .run()
      )
    self.assertIn('control flow signals inside except handlers is not allowed', str(ctx.exception).lower())

  async def test_break_in_async_finally_raises(self):
    async def async_step(x):
      return x + 1

    with self.assertRaises(QuentException) as ctx:
      await (
        Chain(5)
        .then(async_step)
        .finally_(lambda x: Chain.break_())
        .run()
      )
    self.assertIn('control flow signals inside finally handlers is not allowed', str(ctx.exception).lower())

  async def test_return_in_async_finally_raises(self):
    async def async_step(x):
      return x + 1

    with self.assertRaises(QuentException) as ctx:
      await (
        Chain(5)
        .then(async_step)
        .finally_(lambda x: Chain.return_(99))
        .run()
      )
    self.assertIn('control flow signals inside finally handlers is not allowed', str(ctx.exception).lower())

  async def test_break_outside_map_async_raises(self):
    async def async_step(x):
      return x + 1

    with self.assertRaises(QuentException) as ctx:
      await (
        Chain(5)
        .then(async_step)
        .then(lambda x: Chain.break_())
        .run()
      )
    self.assertIn('cannot be used outside', str(ctx.exception).lower())

  async def test_async_return_both_handlers(self):
    """Return in async chain with both except_ and finally_."""
    async def async_step(x):
      return x + 1

    except_tracker = make_tracker()
    finally_tracker = make_tracker()

    result = await (
      Chain(5)
      .then(async_step)
      .then(lambda x: Chain.return_(42))
      .except_(except_tracker)
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(except_tracker.calls, [])
    self.assertEqual(len(finally_tracker.calls), 1)

  async def test_async_foreach_break(self):
    """Break in foreach with async iterable."""
    side_effects = []
    result = await (
      Chain(AsyncRange(6))
      .foreach(lambda x: (side_effects.append(x), Chain.break_())[-1] if x == 3 else side_effects.append(x))
      .run()
    )
    self.assertEqual(result, [0, 1, 2])

  async def test_async_nested_return_2_levels(self):
    """Return propagation through 2 levels of async nesting."""
    async def async_step(x):
      return x + 1

    inner2 = Chain().then(async_step).then(lambda x: Chain.return_(99))
    inner1 = Chain().then(inner2)
    result = await Chain(5).then(inner1).run()
    self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# TestControlFlowWithWith
# ---------------------------------------------------------------------------

class TestControlFlowWithWith(unittest.TestCase):
  """Control flow signals inside with_ and with_do bodies -- CM cleanup."""

  def test_return_inside_with_body_cm_exit_clean(self):
    """return_ inside with_ body -> CM.__exit__ called with (None,None,None)."""
    cm = TrackingCM()
    result = Chain(cm).with_(lambda ctx: Chain.return_(42)).run()
    self.assertEqual(result, 42)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    # __exit__ should receive clean exit args (None, None, None)
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_return_inside_with_do_body_cm_exit_clean(self):
    """return_ inside with_do body -> CM.__exit__ called with (None,None,None)."""
    cm = TrackingCM()
    result = Chain(cm).with_do(lambda ctx: Chain.return_(42)).run()
    self.assertEqual(result, 42)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_break_inside_with_body_cm_exit_clean(self):
    """break_ inside with_ body -> CM.__exit__ called with (None,None,None)."""
    cm = TrackingCM()
    # break_ inside with_ but not in map -> QuentException
    with self.assertRaises(QuentException):
      Chain(cm).with_(lambda ctx: Chain.break_()).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    # _ControlFlowSignal handler calls __exit__(None, None, None)
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_return_no_value_inside_with(self):
    """return_() (no value) inside with_ body -> CM properly cleaned up."""
    cm = TrackingCM()
    result = Chain(cm).with_(lambda ctx: Chain.return_()).run()
    self.assertIsNone(result)
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  def test_return_with_finally_and_with(self):
    """return_ inside with_ body with finally_ handler."""
    cm = TrackingCM()
    finally_tracker = make_tracker()
    result = (
      Chain(cm)
      .with_(lambda ctx: Chain.return_(42))
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertTrue(cm.exited)
    self.assertEqual(len(finally_tracker.calls), 1)

  def test_return_inside_with_with_except(self):
    """return_ inside with_ body with except_ -> except_ NOT called."""
    cm = TrackingCM()
    except_tracker = make_tracker()
    result = (
      Chain(cm)
      .with_(lambda ctx: Chain.return_(42))
      .except_(except_tracker)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertTrue(cm.exited)
    self.assertEqual(except_tracker.calls, [])

  def test_multiple_return_points_only_first_exits(self):
    """Only the first return_ in the chain fires; subsequent steps are skipped."""
    cm = TrackingCM()
    tracker = []
    result = (
      Chain(cm)
      .with_(lambda ctx: Chain.return_(42))
      .then(lambda x: tracker.append('no_run'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])


# ---------------------------------------------------------------------------
# TestControlFlowWithWithAsync
# ---------------------------------------------------------------------------

class TestControlFlowWithWithAsync(IsolatedAsyncioTestCase):
  """Async variants of control flow inside with_ bodies."""

  async def test_return_inside_async_with_body(self):
    """return_ inside with_ body with async CM."""
    cm = AsyncTrackingCM()

    async def body(ctx):
      return Chain.return_(42)

    result = await Chain(cm).with_(body).run()
    self.assertEqual(result, 42)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))

  async def test_return_inside_async_with_do_body(self):
    """return_ inside with_do body with async CM."""
    cm = AsyncTrackingCM()

    async def body(ctx):
      return Chain.return_(42)

    result = await Chain(cm).with_do(body).run()
    self.assertEqual(result, 42)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))


# ---------------------------------------------------------------------------
# TestReturnFromEveryOperationDetailed
# ---------------------------------------------------------------------------

class TestReturnFromEveryOperationDetailed(unittest.TestCase):
  """Focused tests for return_ in each operation type individually."""

  def test_return_from_map_fn(self):
    """return_ in map fn exits the entire chain."""
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.return_(99) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 99)

  def test_return_from_foreach_fn(self):
    """return_ in foreach fn exits the entire chain."""
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: Chain.return_(99) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 99)

  def test_return_from_filter_predicate(self):
    """return_ in filter predicate exits the entire chain."""
    result = (
      Chain([1, 2, 3])
      .filter(lambda x: Chain.return_(99) if x == 2 else True)
      .run()
    )
    self.assertEqual(result, 99)

  def test_return_from_gather_fn(self):
    """return_ in gather fn exits the entire chain."""
    result = Chain(5).gather(lambda x: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  def test_return_from_with_body(self):
    """return_ in with_ body exits the entire chain."""
    cm = SyncCM()
    result = Chain(cm).with_(lambda ctx: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  def test_return_from_with_do_body(self):
    """return_ in with_do body exits the entire chain."""
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  def test_return_from_iterate_raises(self):
    """return_ inside iterate is not allowed."""
    with self.assertRaises(QuentException) as ctx:
      gen = Chain([1, 2, 3]).iterate(lambda x: Chain.return_(99) if x == 2 else x)
      list(gen)
    self.assertIn('return_() inside an iterator is not allowed', str(ctx.exception).lower())

  def test_return_from_do_step(self):
    """return_ in do step exits the chain."""
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: Chain.return_(42))
      .then(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])


# ---------------------------------------------------------------------------
# TestBreakValueVariantsInMap
# ---------------------------------------------------------------------------

class TestBreakValueVariantsInMap(unittest.TestCase):
  """Break with callable, None, False, 0, etc., in map."""

  def test_break_callable_resolved(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(lambda: 'resolved') if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 'resolved')

  def test_break_callable_with_ellipsis(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(lambda: 'ell', ...) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 'ell')

  def test_break_callable_with_args(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(lambda a, b: a + b, 10, 20) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 30)

  def test_break_none_value(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(None) if x == 2 else x)
      .run()
    )
    self.assertIsNone(result)

  def test_break_false_value(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(False) if x == 2 else x)
      .run()
    )
    self.assertIs(result, False)

  def test_break_zero_value(self):
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(0) if x == 2 else x)
      .run()
    )
    self.assertEqual(result, 0)


# ---------------------------------------------------------------------------
# TestMapBreakAtEveryPosition
# ---------------------------------------------------------------------------

class TestMapBreakAtEveryPosition(unittest.TestCase):
  """Break at index 0, 1, mid, last for map."""

  def test_break_at_index_0(self):
    result = Chain([10, 20, 30]).map(lambda x: Chain.break_() if x == 10 else x).run()
    self.assertEqual(result, [])

  def test_break_at_index_1(self):
    result = Chain([10, 20, 30]).map(lambda x: Chain.break_() if x == 20 else x * 2).run()
    self.assertEqual(result, [20])

  def test_break_at_mid(self):
    result = Chain([1, 2, 3, 4, 5]).map(lambda x: Chain.break_() if x == 3 else x * 10).run()
    self.assertEqual(result, [10, 20])

  def test_break_at_last(self):
    result = Chain([1, 2, 3]).map(lambda x: Chain.break_() if x == 3 else x * 10).run()
    self.assertEqual(result, [10, 20])

  def test_break_with_value_at_index_0(self):
    result = Chain([10, 20, 30]).map(lambda x: Chain.break_(42) if x == 10 else x).run()
    self.assertEqual(result, 42)

  def test_break_with_value_at_last(self):
    result = Chain([1, 2, 3]).map(lambda x: Chain.break_(42) if x == 3 else x * 10).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# TestForeachBreakAtEveryPosition
# ---------------------------------------------------------------------------

class TestForeachBreakAtEveryPosition(unittest.TestCase):
  """Break at various positions for foreach."""

  def test_foreach_break_at_index_0(self):
    side = []
    result = Chain([10, 20, 30]).foreach(lambda x: (side.append(x), Chain.break_())[-1] if x == 10 else side.append(x)).run()
    self.assertEqual(result, [])

  def test_foreach_break_at_index_1(self):
    side = []
    result = Chain([10, 20, 30]).foreach(lambda x: (side.append(x), Chain.break_())[-1] if x == 20 else side.append(x)).run()
    # foreach accumulates original items; break at index 1 means only index 0 accumulated
    self.assertEqual(result, [10])

  def test_foreach_break_with_value(self):
    side = []
    result = Chain([1, 2, 3]).foreach(lambda x: (side.append(x), Chain.break_(42))[-1] if x == 2 else side.append(x)).run()
    self.assertEqual(result, 42)

  def test_foreach_no_break(self):
    side = []
    result = Chain([1, 2, 3]).foreach(lambda x: side.append(x)).run()
    # foreach accumulates original items
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(side, [1, 2, 3])


# ---------------------------------------------------------------------------
# TestControlFlowSignalTypes
# ---------------------------------------------------------------------------

class TestControlFlowSignalTypes(unittest.TestCase):
  """Verify internal signal types and their attributes."""

  def test_return_is_control_flow_signal(self):
    exc = _Return(42, (), {})
    self.assertIsInstance(exc, _ControlFlowSignal)
    self.assertIsInstance(exc, Exception)

  def test_break_is_control_flow_signal(self):
    exc = _Break(42, (), {})
    self.assertIsInstance(exc, _ControlFlowSignal)
    self.assertIsInstance(exc, Exception)

  def test_return_stores_value(self):
    exc = _Return(42, (1, 2), {'k': 3})
    self.assertEqual(exc.value, 42)
    self.assertEqual(exc.args_, (1, 2))
    self.assertEqual(exc.kwargs_, {'k': 3})

  def test_break_stores_value(self):
    exc = _Break('hello', (), {})
    self.assertEqual(exc.value, 'hello')

  def test_return_null_value(self):
    exc = _Return(Null, (), {})
    self.assertIs(exc.value, Null)

  def test_break_null_value(self):
    exc = _Break(Null, (), {})
    self.assertIs(exc.value, Null)


# ---------------------------------------------------------------------------
# TestReturnFinallyReceivesRootValue
# ---------------------------------------------------------------------------

class TestReturnFinallyReceivesRootValue(unittest.TestCase):
  """Verify finally_ receives root_value regardless of return/break."""

  def test_return_after_multiple_then_steps(self):
    received = []
    Chain(5).then(lambda x: x + 10).then(lambda x: Chain.return_(42)).finally_(lambda root: received.append(root)).run()
    self.assertEqual(received, [5])

  def test_break_in_map_finally_gets_root(self):
    received = []
    items = [1, 2, 3]
    Chain(items).map(lambda x: Chain.break_() if x == 2 else x).finally_(lambda root: received.append(root)).run()
    self.assertEqual(received, [items])

  def test_normal_flow_finally_gets_root(self):
    received = []
    Chain(5).then(lambda x: x + 10).finally_(lambda root: received.append(root)).run()
    self.assertEqual(received, [5])


# ---------------------------------------------------------------------------
# TestReturnInNestedWithExceptFinally
# ---------------------------------------------------------------------------

class TestReturnInNestedWithExceptFinally(unittest.TestCase):
  """Return propagation with except_/finally_ at multiple nesting levels."""

  def test_nested_return_outer_finally_runs(self):
    inner = Chain().then(lambda x: Chain.return_(99))
    outer_finally = make_tracker()
    result = Chain(5).then(inner).finally_(outer_finally).run()
    self.assertEqual(result, 99)
    self.assertEqual(len(outer_finally.calls), 1)

  def test_nested_return_outer_except_not_called(self):
    inner = Chain().then(lambda x: Chain.return_(99))
    outer_except = make_tracker()
    result = Chain(5).then(inner).except_(outer_except).run()
    self.assertEqual(result, 99)
    self.assertEqual(outer_except.calls, [])

  def test_deeply_nested_return_all_finally_run(self):
    """3-level nesting: return in innermost, verify outermost finally runs."""
    inner3 = Chain().then(lambda x: Chain.return_(99))
    inner2 = Chain().then(inner3)
    outer_finally = make_tracker()
    result = Chain(5).then(inner2).finally_(outer_finally).run()
    self.assertEqual(result, 99)
    self.assertEqual(len(outer_finally.calls), 1)


# ---------------------------------------------------------------------------
# TestBreakInMapWithHandlers
# ---------------------------------------------------------------------------

class TestBreakInMapWithHandlers(unittest.TestCase):
  """Break in map with various handler configurations."""

  def test_break_no_value_with_except_only(self):
    except_tracker = make_tracker()
    result = Chain([1, 2, 3]).map(lambda x: Chain.break_() if x == 2 else x).except_(except_tracker).run()
    self.assertEqual(result, [1])
    self.assertEqual(except_tracker.calls, [])

  def test_break_no_value_with_finally_only(self):
    finally_tracker = make_tracker()
    result = Chain([1, 2, 3]).map(lambda x: Chain.break_() if x == 2 else x).finally_(finally_tracker).run()
    self.assertEqual(result, [1])
    self.assertEqual(len(finally_tracker.calls), 1)

  def test_break_no_value_with_both(self):
    except_tracker = make_tracker()
    finally_tracker = make_tracker()
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_() if x == 2 else x)
      .except_(except_tracker)
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, [1])
    self.assertEqual(except_tracker.calls, [])
    self.assertEqual(len(finally_tracker.calls), 1)

  def test_break_with_value_with_except_only(self):
    except_tracker = make_tracker()
    result = Chain([1, 2, 3]).map(lambda x: Chain.break_(42) if x == 2 else x).except_(except_tracker).run()
    self.assertEqual(result, 42)
    self.assertEqual(except_tracker.calls, [])

  def test_break_with_value_with_finally_only(self):
    finally_tracker = make_tracker()
    result = Chain([1, 2, 3]).map(lambda x: Chain.break_(42) if x == 2 else x).finally_(finally_tracker).run()
    self.assertEqual(result, 42)
    self.assertEqual(len(finally_tracker.calls), 1)

  def test_break_with_value_with_both(self):
    except_tracker = make_tracker()
    finally_tracker = make_tracker()
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.break_(42) if x == 2 else x)
      .except_(except_tracker)
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(except_tracker.calls, [])
    self.assertEqual(len(finally_tracker.calls), 1)


# ---------------------------------------------------------------------------
# TestReturnHandlerCombinations
# ---------------------------------------------------------------------------

class TestReturnHandlerCombinations(unittest.TestCase):
  """Return with all 4 handler combinations for multiple operations."""

  def _run_return_with_handlers(self, builder, has_except, has_finally):
    except_tracker = make_tracker()
    finally_tracker = make_tracker()

    c = builder()
    if has_except:
      c = c.except_(except_tracker)
    if has_finally:
      c = c.finally_(finally_tracker)

    result = c.run()

    self.assertEqual(result, 42)
    self.assertEqual(except_tracker.calls, [])
    if has_finally:
      self.assertEqual(len(finally_tracker.calls), 1)

  def test_return_in_then_all_handler_combos(self):
    for has_except in [True, False]:
      for has_finally in [True, False]:
        with self.subTest(has_except=has_except, has_finally=has_finally):
          self._run_return_with_handlers(
            lambda: Chain(5).then(lambda x: Chain.return_(42)),
            has_except, has_finally
          )

  def test_return_in_do_all_handler_combos(self):
    for has_except in [True, False]:
      for has_finally in [True, False]:
        with self.subTest(has_except=has_except, has_finally=has_finally):
          self._run_return_with_handlers(
            lambda: Chain(5).do(lambda x: Chain.return_(42)),
            has_except, has_finally
          )

  def test_return_in_with_all_handler_combos(self):
    for has_except in [True, False]:
      for has_finally in [True, False]:
        with self.subTest(has_except=has_except, has_finally=has_finally):
          self._run_return_with_handlers(
            lambda: Chain(SyncCM()).with_(lambda ctx: Chain.return_(42)),
            has_except, has_finally
          )

  def test_return_in_nested_all_handler_combos(self):
    for has_except in [True, False]:
      for has_finally in [True, False]:
        with self.subTest(has_except=has_except, has_finally=has_finally):
          inner = Chain().then(lambda x: Chain.return_(42))
          self._run_return_with_handlers(
            lambda: Chain(5).then(inner),
            has_except, has_finally
          )


# ---------------------------------------------------------------------------
# TestAsyncControlFlowWithWith
# ---------------------------------------------------------------------------

class TestAsyncControlFlowWithWith(IsolatedAsyncioTestCase):
  """Async return/break inside with_ body with async CM."""

  async def test_async_return_in_async_cm_with(self):
    cm = AsyncTrackingCM()

    async def body(ctx):
      return Chain.return_(42)

    finally_tracker = make_tracker()
    result = await Chain(cm).with_(body).finally_(finally_tracker).run()
    self.assertEqual(result, 42)
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))
    self.assertEqual(len(finally_tracker.calls), 1)

  async def test_async_break_in_async_cm_with_raises(self):
    cm = AsyncTrackingCM()

    async def body(ctx):
      Chain.break_()

    with self.assertRaises(QuentException):
      await Chain(cm).with_(body).run()
    self.assertTrue(cm.exited)
    self.assertEqual(cm.exit_args, (None, None, None))


# ---------------------------------------------------------------------------
# TestControlFlowEdgeCases
# ---------------------------------------------------------------------------

class TestControlFlowEdgeCases(unittest.TestCase):
  """Edge cases for control flow signals."""

  def test_return_in_first_then_step(self):
    result = Chain().then(lambda: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  def test_return_with_run_value(self):
    """Return in chain invoked with run(value)."""
    c = Chain().then(lambda x: Chain.return_(x * 2))
    result = c.run(21)
    self.assertEqual(result, 42)

  def test_break_with_empty_iterable(self):
    """Break fn never called on empty iterable."""
    result = Chain([]).map(lambda x: Chain.break_()).run()
    self.assertEqual(result, [])

  def test_return_from_root_link(self):
    """Return from the root link itself."""
    result = Chain(lambda: Chain.return_(42)).run()
    self.assertEqual(result, 42)

  def test_multiple_returns_only_first_fires(self):
    """Multiple return_ in the chain -- only the first fires."""
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .then(lambda x: Chain.return_(99))
      .then(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  def test_break_with_single_element_iterable(self):
    """Break with single-element iterable at first element."""
    result = Chain([1]).map(lambda x: Chain.break_() if x == 1 else x).run()
    self.assertEqual(result, [])

  def test_break_with_value_single_element(self):
    result = Chain([1]).map(lambda x: Chain.break_(42) if x == 1 else x).run()
    self.assertEqual(result, 42)

  def test_return_preserves_type(self):
    """Return preserves the exact type passed."""
    for val in [42, 'hello', [1, 2], {'a': 1}, (1,), {1, 2}, 3.14, True, b'bytes']:
      with self.subTest(val=val):
        result = Chain(5).then(lambda x, v=val: Chain.return_(v)).run()
        self.assertEqual(result, val)
        self.assertIsInstance(result, type(val))


# ---------------------------------------------------------------------------
# TestAsyncBreakInMapPositions
# ---------------------------------------------------------------------------

class TestAsyncBreakInMapPositions(IsolatedAsyncioTestCase):
  """Async map break at various positions."""

  async def test_async_break_at_0(self):
    result = await Chain(AsyncRange(5)).map(lambda x: Chain.break_() if x == 0 else x).run()
    self.assertEqual(result, [])

  async def test_async_break_at_mid(self):
    result = await Chain(AsyncRange(10)).map(lambda x: Chain.break_() if x == 5 else x * 2).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  async def test_async_break_at_last(self):
    result = await Chain(AsyncRange(3)).map(lambda x: Chain.break_() if x == 2 else x * 10).run()
    self.assertEqual(result, [0, 10])

  async def test_async_break_with_value_at_0(self):
    result = await Chain(AsyncRange(5)).map(lambda x: Chain.break_(42) if x == 0 else x).run()
    self.assertEqual(result, 42)


# ---------------------------------------------------------------------------
# TestReturnAndBreakInGather
# ---------------------------------------------------------------------------

class TestReturnAndBreakInGather(unittest.TestCase):
  """Control flow signals in gather functions."""

  def test_return_in_gather_exits_chain(self):
    result = Chain(5).gather(
      lambda x: x + 1,
      lambda x: Chain.return_(42),
    ).run()
    self.assertEqual(result, 42)

  def test_break_in_gather_raises(self):
    with self.assertRaises(QuentException):
      Chain(5).gather(lambda x: Chain.break_()).run()


# ---------------------------------------------------------------------------
# TestReturnInFilterPredicate
# ---------------------------------------------------------------------------

class TestReturnInFilterPredicate(unittest.TestCase):
  """Return inside filter predicate exits the chain."""

  def test_return_in_filter_exits(self):
    result = Chain([1, 2, 3]).filter(lambda x: Chain.return_(99) if x == 2 else True).run()
    self.assertEqual(result, 99)

  def test_return_in_filter_with_handlers(self):
    except_tracker = make_tracker()
    finally_tracker = make_tracker()
    result = (
      Chain([1, 2, 3])
      .filter(lambda x: Chain.return_(99) if x == 2 else True)
      .except_(except_tracker)
      .finally_(finally_tracker)
      .run()
    )
    self.assertEqual(result, 99)
    self.assertEqual(except_tracker.calls, [])
    self.assertEqual(len(finally_tracker.calls), 1)


# ---------------------------------------------------------------------------
# TestMapBreakFromNestedChain
# ---------------------------------------------------------------------------

class TestMapBreakFromNestedChain(unittest.TestCase):
  """Using a nested chain inside map that calls break_."""

  def test_nested_chain_break_in_map(self):
    """A plain function (not chain) in map can break."""
    result = (
      Chain([1, 2, 3, 4])
      .map(lambda x: Chain.break_(42) if x == 3 else x)
      .run()
    )
    self.assertEqual(result, 42)

  def test_map_with_nested_map_break(self):
    """Nested map with break in inner, outer continues."""
    def process_inner(items):
      # Break after processing 2 items
      count = [0]
      def fn(x):
        count[0] += 1
        if count[0] > 2:
          Chain.break_()
        return x * 10
      return Chain(items).map(fn).run()

    result = (
      Chain([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
      .map(process_inner)
      .run()
    )
    # Each inner map processes 2 items then breaks
    self.assertEqual(result, [[10, 20], [50, 60], [90, 100]])


if __name__ == '__main__':
  unittest.main()
