"""Tests for multi-operation chain composition: how operations compose in
sequence, state leakage, value corruption, handler interaction.
"""
from __future__ import annotations

import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from helpers import (
  SyncCM,
  SyncCMSuppresses,
  AsyncCM,
  AsyncCMSuppresses,
  TrackingCM,
  async_fn,
  async_identity,
  make_tracker,
  make_async_tracker,
)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def _identity(x):
  return x


def _double(x):
  return x * 2


def _add1(x):
  return x + 1


def _is_even(x):
  return x % 2 == 0


def _to_list(x):
  return list(range(x))


def _sum_list(x):
  return sum(x)


async def _async_double(x):
  return x * 2


async def _async_add1(x):
  return x + 1


async def _async_is_even(x):
  return x % 2 == 0


# ===========================================================================
# Class: TestTwoOperationPipelines
# ===========================================================================
class TestTwoOperationPipelines(unittest.TestCase):
  """All pairs of operations in sequence with value propagation verification."""

  def test_then_then(self):
    with self.subTest(op1='then', op2='then'):
      result = Chain(5).then(_double).then(_add1).run()
      self.assertEqual(result, 11)

  def test_then_do(self):
    with self.subTest(op1='then', op2='do'):
      tracker = []
      result = Chain(5).then(_double).do(lambda x: tracker.append(x)).run()
      self.assertEqual(result, 10)
      self.assertEqual(tracker, [10])

  def test_then_map(self):
    with self.subTest(op1='then', op2='map'):
      result = Chain(4).then(_to_list).map(_double).run()
      self.assertEqual(result, [0, 2, 4, 6])

  def test_then_filter(self):
    with self.subTest(op1='then', op2='filter'):
      result = Chain(6).then(_to_list).filter(_is_even).run()
      self.assertEqual(result, [0, 2, 4])

  def test_then_gather(self):
    with self.subTest(op1='then', op2='gather'):
      result = Chain(5).then(_double).gather(_add1, _double).run()
      self.assertEqual(result, [11, 20])

  def test_then_with(self):
    with self.subTest(op1='then', op2='with_'):
      result = Chain(5).then(lambda x: SyncCM()).with_(lambda ctx: ctx.upper()).run()
      self.assertEqual(result, 'CTX_VALUE')

  def test_do_then(self):
    with self.subTest(op1='do', op2='then'):
      tracker = []
      result = Chain(5).do(lambda x: tracker.append(x)).then(_double).run()
      self.assertEqual(result, 10)
      self.assertEqual(tracker, [5])

  def test_do_do(self):
    with self.subTest(op1='do', op2='do'):
      tracker1, tracker2 = [], []
      result = (
        Chain(5)
        .do(lambda x: tracker1.append(x))
        .do(lambda x: tracker2.append(x))
        .run()
      )
      self.assertEqual(result, 5)
      self.assertEqual(tracker1, [5])
      self.assertEqual(tracker2, [5])

  def test_do_map(self):
    with self.subTest(op1='do', op2='map'):
      tracker = []
      result = (
        Chain([1, 2, 3])
        .do(lambda x: tracker.append(len(x)))
        .map(_double)
        .run()
      )
      self.assertEqual(result, [2, 4, 6])
      self.assertEqual(tracker, [3])

  def test_do_filter(self):
    with self.subTest(op1='do', op2='filter'):
      tracker = []
      result = (
        Chain([1, 2, 3, 4])
        .do(lambda x: tracker.append('side'))
        .filter(_is_even)
        .run()
      )
      self.assertEqual(result, [2, 4])
      self.assertEqual(tracker, ['side'])

  def test_do_gather(self):
    with self.subTest(op1='do', op2='gather'):
      tracker = []
      result = (
        Chain(5)
        .do(lambda x: tracker.append(x))
        .gather(_add1, _double)
        .run()
      )
      self.assertEqual(result, [6, 10])
      self.assertEqual(tracker, [5])

  def test_do_with(self):
    with self.subTest(op1='do', op2='with_'):
      tracker = []
      result = (
        Chain(SyncCM())
        .do(lambda x: tracker.append('pre'))
        .with_(lambda ctx: ctx.upper())
        .run()
      )
      self.assertEqual(result, 'CTX_VALUE')
      self.assertEqual(tracker, ['pre'])

  def test_map_then(self):
    with self.subTest(op1='map', op2='then'):
      result = Chain([1, 2, 3]).map(_double).then(_sum_list).run()
      self.assertEqual(result, 12)

  def test_map_filter(self):
    with self.subTest(op1='map', op2='filter'):
      result = Chain([1, 2, 3, 4]).map(_double).filter(lambda x: x > 4).run()
      self.assertEqual(result, [6, 8])

  def test_map_map(self):
    with self.subTest(op1='map', op2='map (nested)'):
      # First map maps [1,2] -> [[0],[0,1]]; second map flattens-ish
      result = Chain([2, 3]).map(_to_list).map(_sum_list).run()
      self.assertEqual(result, [1, 3])

  def test_map_gather(self):
    with self.subTest(op1='map', op2='gather'):
      result = (
        Chain([1, 2, 3])
        .map(_double)
        .gather(_sum_list, lambda x: len(x))
        .run()
      )
      self.assertEqual(result, [12, 3])

  def test_foreach(self):
    with self.subTest(op1='map', op2='do'):
      tracker = []
      result = (
        Chain([1, 2, 3])
        .map(_double)
        .do(lambda x: tracker.append(x))
        .run()
      )
      self.assertEqual(result, [2, 4, 6])
      self.assertEqual(tracker, [[2, 4, 6]])

  def test_filter_then(self):
    with self.subTest(op1='filter', op2='then'):
      result = Chain([1, 2, 3, 4]).filter(_is_even).then(_sum_list).run()
      self.assertEqual(result, 6)

  def test_filter_map(self):
    with self.subTest(op1='filter', op2='map'):
      result = Chain([1, 2, 3, 4]).filter(_is_even).map(_double).run()
      self.assertEqual(result, [4, 8])

  def test_filter_gather(self):
    with self.subTest(op1='filter', op2='gather'):
      result = (
        Chain([1, 2, 3, 4])
        .filter(_is_even)
        .gather(_sum_list, lambda x: len(x))
        .run()
      )
      self.assertEqual(result, [6, 2])

  def test_filter_filter(self):
    with self.subTest(op1='filter', op2='filter'):
      result = (
        Chain([1, 2, 3, 4, 5, 6])
        .filter(lambda x: x > 2)
        .filter(_is_even)
        .run()
      )
      self.assertEqual(result, [4, 6])

  def test_filter_do(self):
    with self.subTest(op1='filter', op2='do'):
      tracker = []
      result = (
        Chain([1, 2, 3, 4])
        .filter(_is_even)
        .do(lambda x: tracker.append(len(x)))
        .run()
      )
      self.assertEqual(result, [2, 4])
      self.assertEqual(tracker, [2])

  def test_with_then(self):
    with self.subTest(op1='with_', op2='then'):
      result = Chain(SyncCM()).with_(lambda ctx: ctx).then(lambda x: x.upper()).run()
      self.assertEqual(result, 'CTX_VALUE')

  def test_with_map(self):
    with self.subTest(op1='with_', op2='map'):
      result = (
        Chain(SyncCM())
        .with_(lambda ctx: [1, 2, 3])
        .map(_double)
        .run()
      )
      self.assertEqual(result, [2, 4, 6])

  def test_with_filter(self):
    with self.subTest(op1='with_', op2='filter'):
      result = (
        Chain(SyncCM())
        .with_(lambda ctx: [1, 2, 3, 4])
        .filter(_is_even)
        .run()
      )
      self.assertEqual(result, [2, 4])

  def test_with_gather(self):
    with self.subTest(op1='with_', op2='gather'):
      result = (
        Chain(SyncCM())
        .with_(lambda ctx: 5)
        .gather(_add1, _double)
        .run()
      )
      self.assertEqual(result, [6, 10])

  def test_with_do(self):
    with self.subTest(op1='with_', op2='do'):
      tracker = []
      result = (
        Chain(SyncCM())
        .with_(lambda ctx: ctx)
        .do(lambda x: tracker.append(x))
        .run()
      )
      self.assertEqual(result, 'ctx_value')
      self.assertEqual(tracker, ['ctx_value'])

  def test_gather_then(self):
    with self.subTest(op1='gather', op2='then'):
      result = Chain(5).gather(_add1, _double).then(_sum_list).run()
      self.assertEqual(result, 16)

  def test_gather_map(self):
    with self.subTest(op1='gather', op2='map'):
      result = Chain(5).gather(_add1, _double).map(_double).run()
      self.assertEqual(result, [12, 20])

  def test_gather_filter(self):
    with self.subTest(op1='gather', op2='filter'):
      result = Chain(5).gather(_add1, _double).filter(_is_even).run()
      self.assertEqual(result, [6, 10])

  def test_gather_do(self):
    with self.subTest(op1='gather', op2='do'):
      tracker = []
      result = (
        Chain(5)
        .gather(_add1, _double)
        .do(lambda x: tracker.append(x))
        .run()
      )
      self.assertEqual(result, [6, 10])
      self.assertEqual(tracker, [[6, 10]])

  def test_gather_gather(self):
    with self.subTest(op1='gather', op2='gather'):
      result = (
        Chain(5)
        .gather(_add1, _double)
        .gather(_sum_list, lambda x: len(x))
        .run()
      )
      self.assertEqual(result, [16, 2])


# ===========================================================================
# Class: TestThreeOperationPipelines
# ===========================================================================
class TestThreeOperationPipelines(unittest.TestCase):
  """Common 3-operation patterns."""

  def test_then_map_then(self):
    with self.subTest(pipeline='then->map->then'):
      result = (
        Chain(4)
        .then(_to_list)
        .map(_double)
        .then(_sum_list)
        .run()
      )
      # range(4) -> [0,1,2,3], doubled -> [0,2,4,6], sum -> 12
      self.assertEqual(result, 12)

  def test_then_filter_map(self):
    with self.subTest(pipeline='then->filter->map'):
      result = (
        Chain(6)
        .then(_to_list)
        .filter(_is_even)
        .map(_double)
        .run()
      )
      # range(6) -> [0..5], filter even -> [0,2,4], double -> [0,4,8]
      self.assertEqual(result, [0, 4, 8])

  def test_map_filter_then(self):
    with self.subTest(pipeline='map->filter->then'):
      result = (
        Chain([1, 2, 3, 4, 5])
        .map(_double)
        .filter(lambda x: x > 4)
        .then(_sum_list)
        .run()
      )
      # doubled -> [2,4,6,8,10], filter >4 -> [6,8,10], sum -> 24
      self.assertEqual(result, 24)

  def test_with_then_then(self):
    with self.subTest(pipeline='with_->then->then'):
      result = (
        Chain(SyncCM())
        .with_(lambda ctx: ctx)
        .then(lambda x: x.upper())
        .then(lambda x: x + '!')
        .run()
      )
      self.assertEqual(result, 'CTX_VALUE!')

  def test_then_gather_then(self):
    with self.subTest(pipeline='then->gather->then'):
      result = (
        Chain(5)
        .then(_double)
        .gather(_add1, _double)
        .then(_sum_list)
        .run()
      )
      # 5 -> 10, gather [11, 20], sum -> 31
      self.assertEqual(result, 31)

  def test_then_then_map(self):
    with self.subTest(pipeline='then->then->map'):
      result = (
        Chain(3)
        .then(_add1)
        .then(_to_list)
        .map(_double)
        .run()
      )
      # 3 -> 4, range(4) -> [0,1,2,3], double -> [0,2,4,6]
      self.assertEqual(result, [0, 2, 4, 6])

  def test_filter_map_then(self):
    with self.subTest(pipeline='filter->map->then'):
      result = (
        Chain([1, 2, 3, 4, 5, 6])
        .filter(lambda x: x > 3)
        .map(_double)
        .then(_sum_list)
        .run()
      )
      # filter >3 -> [4,5,6], double -> [8,10,12], sum -> 30
      self.assertEqual(result, 30)

  def test_do_then_map(self):
    with self.subTest(pipeline='do->then->map'):
      tracker = []
      result = (
        Chain(3)
        .do(lambda x: tracker.append(x))
        .then(_to_list)
        .map(_double)
        .run()
      )
      self.assertEqual(result, [0, 2, 4])
      self.assertEqual(tracker, [3])

  def test_foreach_then(self):
    with self.subTest(pipeline='map->do->then'):
      tracker = []
      result = (
        Chain([1, 2, 3])
        .map(_double)
        .do(lambda x: tracker.append(list(x)))
        .then(_sum_list)
        .run()
      )
      # double -> [2,4,6], do records it, sum -> 12
      self.assertEqual(result, 12)
      self.assertEqual(tracker, [[2, 4, 6]])

  def test_gather_filter_then(self):
    with self.subTest(pipeline='gather->filter->then'):
      result = (
        Chain(5)
        .gather(_add1, _double, lambda x: x - 1)
        .filter(_is_even)
        .then(_sum_list)
        .run()
      )
      # gather [6, 10, 4], filter even -> [6, 10, 4], sum -> 20
      self.assertEqual(result, 20)

  def test_then_with_then(self):
    with self.subTest(pipeline='then->with_->then'):
      result = (
        Chain(5)
        .then(lambda x: SyncCM())
        .with_(lambda ctx: ctx)
        .then(lambda x: x.upper())
        .run()
      )
      self.assertEqual(result, 'CTX_VALUE')

  def test_with_map_then(self):
    with self.subTest(pipeline='with_->map->then'):
      result = (
        Chain(SyncCM())
        .with_(lambda ctx: [1, 2, 3])
        .map(_double)
        .then(_sum_list)
        .run()
      )
      self.assertEqual(result, 12)


# ===========================================================================
# Class: TestPipelineWithExcept
# ===========================================================================
class TestPipelineWithExcept(unittest.TestCase):
  """Multi-operation chains with except_ at different positions."""

  def test_error_in_first_op_caught(self):
    with self.subTest(error_at='first'):
      result = (
        Chain(5)
        .then(lambda x: 1 / 0)
        .then(_double)
        .except_(lambda rv, e: 'caught')
        .run()
      )
      self.assertEqual(result, 'caught')

  def test_error_in_second_op_caught(self):
    with self.subTest(error_at='second'):
      result = (
        Chain(5)
        .then(_double)
        .then(lambda x: 1 / 0)
        .then(_add1)
        .except_(lambda rv, e: 'caught')
        .run()
      )
      self.assertEqual(result, 'caught')

  def test_error_in_third_op_caught(self):
    with self.subTest(error_at='third'):
      result = (
        Chain(5)
        .then(_double)
        .then(_add1)
        .then(lambda x: 1 / 0)
        .except_(lambda rv, e: 'caught')
        .run()
      )
      self.assertEqual(result, 'caught')

  def test_error_in_map_within_pipeline(self):
    with self.subTest(error_at='map'):
      result = (
        Chain(5)
        .then(_to_list)
        .map(lambda x: 1 / 0)
        .then(_sum_list)
        .except_(lambda rv, e: 'caught')
        .run()
      )
      self.assertEqual(result, 'caught')

  def test_error_in_filter_within_pipeline(self):
    with self.subTest(error_at='filter'):
      result = (
        Chain(5)
        .then(_to_list)
        .filter(lambda x: 1 / 0)
        .then(_sum_list)
        .except_(lambda rv, e: 'caught')
        .run()
      )
      self.assertEqual(result, 'caught')

  def test_error_in_gather_within_pipeline(self):
    with self.subTest(error_at='gather'):
      result = (
        Chain(5)
        .then(_double)
        .gather(lambda x: 1 / 0, _add1)
        .except_(lambda rv, e: 'caught')
        .run()
      )
      self.assertEqual(result, 'caught')

  def test_error_in_with_within_pipeline(self):
    with self.subTest(error_at='with_'):
      result = (
        Chain(5)
        .then(lambda x: SyncCM())
        .with_(lambda ctx: 1 / 0)
        .except_(lambda rv, e: 'caught')
        .run()
      )
      self.assertEqual(result, 'caught')

  def test_accumulated_state_correct_before_error(self):
    """Verify side effects ran for steps before the error."""
    tracker = []
    result = (
      Chain(5)
      .do(lambda x: tracker.append(('step1', x)))
      .then(_double)
      .do(lambda x: tracker.append(('step2', x)))
      .then(lambda x: 1 / 0)
      .except_(lambda rv, e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, [('step1', 5), ('step2', 10)])

  def test_except_handler_receives_correct_exception_type(self):
    received = []
    def handler(rv, exc):
      received.append(type(exc))
      return 'ok'

    Chain(5).then(lambda x: 1 / 0).then(_double).except_(handler).run()
    self.assertEqual(received, [ZeroDivisionError])

  def test_no_error_except_not_called(self):
    tracker = []
    result = (
      Chain(5)
      .then(_double)
      .then(_add1)
      .except_(lambda rv, e: tracker.append('except'))
      .run()
    )
    self.assertEqual(result, 11)
    self.assertEqual(tracker, [])


# ===========================================================================
# Class: TestPipelineWithFinally
# ===========================================================================
class TestPipelineWithFinally(unittest.TestCase):
  """Multi-operation chains with finally_."""

  def test_finally_receives_root_value_on_success(self):
    tracker = []
    result = (
      Chain(5)
      .then(_double)
      .then(_add1)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 11)
    # finally_ receives root_value = 5
    self.assertEqual(tracker, [5])

  def test_finally_runs_on_error(self):
    tracker = []
    with self.assertRaises(ZeroDivisionError):
      (
        Chain(5)
        .then(lambda x: 1 / 0)
        .finally_(lambda rv: tracker.append(rv))
        .run()
      )
    # finally_ receives root_value = 5
    self.assertEqual(tracker, [5])

  def test_finally_root_not_modified_by_steps(self):
    """Verify root_value is the ORIGINAL root, not modified by intermediate steps."""
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x * 100)
      .then(lambda x: x + 999)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 1499)
    # Root value should be 5, not 500 or 1499
    self.assertEqual(tracker, [5])

  def test_finally_runs_with_except_on_success(self):
    tracker = []
    result = (
      Chain(5)
      .then(_double)
      .except_(lambda rv, e: 'caught')
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [5])

  def test_finally_runs_with_except_on_error(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: 1 / 0)
      .except_(lambda rv, e: 'caught')
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 'caught')
    self.assertEqual(tracker, [5])

  def test_finally_runs_when_except_reraises(self):
    tracker = []
    def bad_handler(rv, exc):
      raise RuntimeError('handler error') from exc
    with self.assertRaises(RuntimeError):
      (
        Chain(5)
        .then(lambda x: 1 / 0)
        .except_(bad_handler)
        .finally_(lambda rv: tracker.append(rv))
        .run()
      )
    self.assertEqual(tracker, [5])

  def test_finally_with_complex_pipeline(self):
    tracker = []
    result = (
      Chain(3)
      .then(_to_list)
      .map(_double)
      .then(_sum_list)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    # range(3) -> [0,1,2], double -> [0,2,4], sum -> 6
    self.assertEqual(result, 6)
    # Root value is range(3) == [0,1,2] (result of first evaluation of root)
    self.assertEqual(tracker, [3])

  def test_finally_with_run_value(self):
    """When run(v) is called, the root_value is v."""
    tracker = []
    chain = (
      Chain()
      .then(_double)
      .finally_(lambda rv: tracker.append(rv))
    )
    result = chain.run(7)
    self.assertEqual(result, 14)
    self.assertEqual(tracker, [7])


# ===========================================================================
# Class: TestPipelineValueFlow
# ===========================================================================
class TestPipelineValueFlow(unittest.TestCase):
  """Trace value flow through complex pipelines."""

  def test_then_do_then_value_flow(self):
    """Chain(5).then(x*2).do(tracker).then(x+1) -> 11."""
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: x * 2)
      .do(lambda x: tracker.append(x))
      .then(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, 11)
    # do receives 10 (result of x*2)
    self.assertEqual(tracker, [10])

  def test_map_filter_pipeline(self):
    """Chain([1,2,3]).map(x*2).filter(x>2) -> [4,6]."""
    result = (
      Chain([1, 2, 3])
      .map(lambda x: x * 2)
      .filter(lambda x: x > 2)
      .run()
    )
    self.assertEqual(result, [4, 6])

  def test_with_then_value_flow(self):
    """Chain(SyncCM()).with_(ctx).then(upper) -> 'CTX_VALUE'."""
    result = (
      Chain(SyncCM())
      .with_(lambda ctx: ctx)
      .then(lambda x: x.upper())
      .run()
    )
    self.assertEqual(result, 'CTX_VALUE')

  def test_gather_produces_list(self):
    result = Chain(5).gather(lambda x: x + 1, lambda x: x * 2).run()
    self.assertEqual(result, [6, 10])

  def test_long_pipeline_value_flow(self):
    """10-step pipeline tracking value at each point."""
    tracker = []
    result = (
      Chain(1)
      .then(lambda x: x + 1)               # 2
      .do(lambda x: tracker.append(('a', x)))
      .then(lambda x: x * 3)               # 6
      .do(lambda x: tracker.append(('b', x)))
      .then(lambda x: x + 4)               # 10
      .do(lambda x: tracker.append(('c', x)))
      .then(lambda x: [x, x + 1, x + 2])  # [10, 11, 12]
      .map(lambda x: x * 2)            # [20, 22, 24]
      .filter(lambda x: x > 21)            # [22, 24]
      .then(lambda x: sum(x))              # 46
      .run()
    )
    self.assertEqual(result, 46)
    self.assertEqual(tracker, [('a', 2), ('b', 6), ('c', 10)])

  def test_none_propagation(self):
    """None is a valid chain value, distinct from Null."""
    result = Chain(5).then(lambda x: None).then(lambda x: x is None).run()
    self.assertTrue(result)

  def test_empty_list_propagation(self):
    result = Chain([]).map(_double).run()
    self.assertEqual(result, [])

  def test_empty_filter_result(self):
    result = Chain([1, 3, 5]).filter(_is_even).run()
    self.assertEqual(result, [])

  def test_nested_chain_value_flow(self):
    inner = Chain().then(lambda x: x * 3)
    result = Chain(5).then(_double).then(inner).then(_add1).run()
    # 5 -> 10 -> 30 -> 31
    self.assertEqual(result, 31)

  def test_chain_with_no_root_value_and_run_value(self):
    chain = Chain().then(_double).then(_add1)
    result = chain.run(7)
    self.assertEqual(result, 15)

  def test_do_does_not_leak_value(self):
    """Multiple dos in sequence should not modify the chain value."""
    result = (
      Chain(42)
      .do(lambda x: x * 100)
      .do(lambda x: x + 999)
      .do(lambda x: 'trash')
      .then(lambda x: x)
      .run()
    )
    self.assertEqual(result, 42)

  def test_multiple_gather_value_flow(self):
    """Gather produces a list, which can be gathered again."""
    result = (
      Chain(5)
      .gather(_add1, _double)
      .gather(_sum_list, lambda x: max(x))
      .run()
    )
    # gather1: [6, 10], gather2: [16, 10]
    self.assertEqual(result, [16, 10])


# ===========================================================================
# Class: TestPipelineAsync
# ===========================================================================
class TestPipelineAsync(IsolatedAsyncioTestCase):
  """Same pipeline patterns with async operations mixed in."""

  async def test_sync_async_sync_transitions(self):
    result = await (
      Chain(5)
      .then(_double)       # sync -> 10
      .then(_async_add1)   # async -> 11
      .then(_double)       # sync -> 22
      .run()
    )
    self.assertEqual(result, 22)

  async def test_all_async_pipeline(self):
    result = await (
      Chain(5)
      .then(_async_double)
      .then(_async_add1)
      .then(_async_double)
      .run()
    )
    # 5 -> 10 -> 11 -> 22
    self.assertEqual(result, 22)

  async def test_async_map_in_pipeline(self):
    result = await (
      Chain(4)
      .then(_to_list)
      .map(_async_double)
      .then(_sum_list)
      .run()
    )
    # range(4) -> [0,1,2,3], double -> [0,2,4,6], sum -> 12
    self.assertEqual(result, 12)

  async def test_async_filter_in_pipeline(self):
    result = await (
      Chain(6)
      .then(_to_list)
      .filter(_async_is_even)
      .then(_sum_list)
      .run()
    )
    # range(6) -> [0..5], filter even -> [0,2,4], sum -> 6
    self.assertEqual(result, 6)

  async def test_mixed_sync_async_in_gather(self):
    result = await (
      Chain(5)
      .gather(_add1, _async_double, lambda x: x - 1)
      .run()
    )
    self.assertEqual(result, [6, 10, 4])

  async def test_async_with_in_pipeline(self):
    result = await (
      Chain(AsyncCM())
      .with_(lambda ctx: ctx)
      .then(lambda x: x.upper())
      .run()
    )
    self.assertEqual(result, 'CTX_VALUE')

  async def test_async_pipeline_with_except(self):
    result = await (
      Chain(5)
      .then(_async_double)
      .then(lambda x: 1 / 0)
      .except_(lambda rv, e: 'caught')
      .run()
    )
    self.assertEqual(result, 'caught')

  async def test_async_pipeline_with_finally(self):
    tracker = []
    result = await (
      Chain(5)
      .then(_async_double)
      .then(_async_add1)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 11)
    self.assertEqual(tracker, [5])

  async def test_async_pipeline_error_in_async_step(self):
    async def raiser(x):
      raise ValueError('async error')
    result = await (
      Chain(5)
      .then(_async_double)
      .then(raiser)
      .except_(lambda rv, e: f'caught:{type(e).__name__}')
      .run()
    )
    self.assertEqual(result, 'caught:ValueError')

  async def test_sync_to_async_transition_preserves_value(self):
    """Verify value is not lost during sync-to-async handoff."""
    result = await (
      Chain(5)
      .then(lambda x: x * 2)   # sync: 10
      .then(async_fn)           # async: 11
      .then(lambda x: x * 3)   # sync in async path: 33
      .run()
    )
    self.assertEqual(result, 33)

  async def test_do_in_async_pipeline(self):
    tracker = []
    result = await (
      Chain(5)
      .then(_async_double)
      .do(lambda x: tracker.append(x))
      .then(_async_add1)
      .run()
    )
    self.assertEqual(result, 11)
    self.assertEqual(tracker, [10])


# ===========================================================================
# Class: TestPipelineWithControlFlow
# ===========================================================================
class TestPipelineWithControlFlow(unittest.TestCase):
  """return_ at various positions in multi-op chain."""

  def test_return_in_first_step(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .then(lambda x: tracker.append('step2'))
      .then(lambda x: tracker.append('step3'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  def test_return_in_second_step(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: (tracker.append('step1'), x * 2)[1])
      .then(lambda x: Chain.return_(x + 1))
      .then(lambda x: tracker.append('step3'))
      .run()
    )
    self.assertEqual(result, 11)
    self.assertEqual(tracker, ['step1'])

  def test_return_in_third_step(self):
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: (tracker.append('step1'), x * 2)[1])
      .then(lambda x: (tracker.append('step2'), x + 1)[1])
      .then(lambda x: Chain.return_(x * 10))
      .then(lambda x: tracker.append('step4'))
      .run()
    )
    self.assertEqual(result, 110)
    self.assertEqual(tracker, ['step1', 'step2'])

  def test_return_in_do_step(self):
    tracker = []
    result = (
      Chain(5)
      .then(_double)
      .do(lambda x: Chain.return_(99))
      .then(lambda x: tracker.append('should_not_run'))
      .run()
    )
    self.assertEqual(result, 99)
    self.assertEqual(tracker, [])

  def test_return_with_finally_still_runs(self):
    """finally_ IS executed even with return_."""
    tracker = []
    result = (
      Chain(5)
      .then(lambda x: Chain.return_(42))
      .then(lambda x: tracker.append('step2'))
      .finally_(lambda rv: tracker.append(('finally', rv)))
      .run()
    )
    self.assertEqual(result, 42)
    # step2 should NOT be called, but finally should be
    self.assertNotIn('step2', tracker)
    self.assertIn(('finally', 5), tracker)

  def test_return_in_map_exits_chain(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .map(lambda x: Chain.return_(99) if x == 2 else x)
      .then(lambda x: tracker.append('after_map'))
      .run()
    )
    self.assertEqual(result, 99)
    self.assertEqual(tracker, [])

  def test_break_in_map_continues_chain(self):
    """break_ exits the map but the chain continues."""
    result = (
      Chain([1, 2, 3, 4, 5])
      .map(lambda x: Chain.break_() if x == 3 else x * 10)
      .then(_sum_list)
      .run()
    )
    # Items processed: 1, 2 (break at 3). Results: [10, 20], sum -> 30
    self.assertEqual(result, 30)

  def test_return_in_with_body(self):
    tracker = []
    cm = TrackingCM()
    result = (
      Chain(cm)
      .with_(lambda ctx: Chain.return_(42))
      .then(lambda x: tracker.append('after_with'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])
    # CM should have been exited cleanly
    self.assertTrue(cm.exited)

  def test_return_in_gather_exits_chain(self):
    tracker = []
    result = (
      Chain(5)
      .gather(lambda x: Chain.return_(42))
      .then(lambda x: tracker.append('after_gather'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  def test_multiple_operations_after_return_not_executed(self):
    tracker = []
    result = (
      Chain(5)
      .then(_double)
      .then(lambda x: Chain.return_(x + 1))
      .do(lambda x: tracker.append('do'))
      .then(lambda x: tracker.append('then'))
      .map(lambda x: tracker.append('map'))
      .run()
    )
    self.assertEqual(result, 11)
    self.assertEqual(tracker, [])


# ===========================================================================
# Class: TestPipelineWithControlFlowAsync
# ===========================================================================
class TestPipelineWithControlFlowAsync(IsolatedAsyncioTestCase):
  """Async versions of control flow tests."""

  async def test_return_in_async_pipeline(self):
    tracker = []
    result = await (
      Chain(5)
      .then(_async_double)
      .then(lambda x: Chain.return_(42))
      .then(lambda x: tracker.append('not_run'))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [])

  async def test_return_with_finally_in_async(self):
    tracker = []
    result = await (
      Chain(5)
      .then(_async_double)
      .then(lambda x: Chain.return_(42))
      .finally_(lambda rv: tracker.append(('finally', rv)))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertIn(('finally', 5), tracker)

  async def test_break_in_async_map(self):
    result = await (
      Chain([1, 2, 3, 4])
      .map(lambda x: Chain.break_() if x == 3 else _async_double(x))
      .run()
    )
    # Items 1, 2 processed with async_double -> awaitables
    # The results should be [2, 4]
    self.assertEqual(result, [2, 4])


# ===========================================================================
# Class: TestPipelineDoSemantics
# ===========================================================================
class TestPipelineDoSemantics(unittest.TestCase):
  """Detailed tests for do() semantics in composition."""

  def test_do_between_operations_preserves_value(self):
    """do() should pass through the value from the previous step."""
    tracker = []
    result = (
      Chain([1, 2, 3])
      .map(_double)
      .do(lambda x: tracker.append(list(x)))
      .filter(lambda x: x > 2)
      .run()
    )
    self.assertEqual(result, [4, 6])
    self.assertEqual(tracker, [[2, 4, 6]])

  def test_multiple_dos_chain_same_value(self):
    tracker1, tracker2, tracker3 = [], [], []
    result = (
      Chain(42)
      .do(lambda x: tracker1.append(x))
      .do(lambda x: tracker2.append(x))
      .do(lambda x: tracker3.append(x))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker1, [42])
    self.assertEqual(tracker2, [42])
    self.assertEqual(tracker3, [42])


# ===========================================================================
# Class: TestPipelineWithDo (with_do semantics)
# ===========================================================================
class TestPipelineWithDo(unittest.TestCase):
  """Tests for with_do() in pipelines."""

  def test_with_do_preserves_cm_value(self):
    """with_do() should preserve the CM itself as the chain value."""
    cm = SyncCM()
    result = (
      Chain(cm)
      .with_do(lambda ctx: ctx.upper())
      .run()
    )
    # with_do ignores the body result and returns the original CM
    self.assertIs(result, cm)

  def test_with_do_in_pipeline(self):
    tracker = []
    cm = SyncCM()
    result = (
      Chain(cm)
      .with_do(lambda ctx: tracker.append(ctx))
      .then(lambda x: 'done')
      .run()
    )
    self.assertEqual(result, 'done')
    self.assertEqual(tracker, ['ctx_value'])


# ===========================================================================
# Class: TestPipelineForeach
# ===========================================================================
class TestPipelineForeach(unittest.TestCase):
  """Tests for foreach() in pipelines."""

  def test_foreach_preserves_original_items(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: tracker.append(x * 10))
      .run()
    )
    # foreach keeps original items, not fn results
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [10, 20, 30])

  def test_foreach_then_pipeline(self):
    tracker = []
    result = (
      Chain([1, 2, 3])
      .foreach(lambda x: tracker.append(x))
      .then(_sum_list)
      .run()
    )
    self.assertEqual(result, 6)
    self.assertEqual(tracker, [1, 2, 3])


# ===========================================================================
# Class: TestPipelineEdgeCases
# ===========================================================================
class TestPipelineEdgeCases(unittest.TestCase):
  """Edge cases in pipeline composition."""

  def test_chain_with_no_ops(self):
    result = Chain(42).run()
    self.assertEqual(result, 42)

  def test_chain_with_no_root_no_ops(self):
    result = Chain().run()
    self.assertIsNone(result)

  def test_chain_only_do_ops(self):
    tracker = []
    result = (
      Chain(42)
      .do(lambda x: tracker.append(x))
      .run()
    )
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [42])

  def test_chain_with_then_returning_none(self):
    result = Chain(5).then(lambda x: None).run()
    self.assertIsNone(result)

  def test_chain_with_ellipsis_arg_in_pipeline(self):
    """Ellipsis convention: call with no args."""
    result = Chain(5).then(lambda: 99, ...).then(_double).run()
    self.assertEqual(result, 198)

  def test_pipeline_with_non_callable_then(self):
    """Non-callable value in then is returned as-is."""
    result = Chain(5).then(42).run()
    self.assertEqual(result, 42)

  def test_pipeline_with_class_constructor(self):
    class MyObj:
      def __init__(self, x):
        self.x = x
    result = Chain(5).then(MyObj).run()
    self.assertIsInstance(result, MyObj)
    self.assertEqual(result.x, 5)

  def test_deeply_nested_pipeline(self):
    """A pipeline with many operations."""
    chain = Chain(1)
    for _ in range(50):
      chain = chain.then(_add1)
    result = chain.run()
    self.assertEqual(result, 51)


if __name__ == '__main__':
  unittest.main()
