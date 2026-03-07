"""Tests for Chain.if_() tuple forms of the ``then=`` parameter.

Covers:
  A. 2-tuple with positional args: then=(fn, (arg1, arg2))
  B. 2-tuple with keyword args:   then=(fn, {'key': val})
  C. 3-tuple with both:           then=(fn, (arg1,), {'key': val})
  D. predicate=None (truthiness check on current value)
  E. Error cases (wrong-length tuples)
  F. Async variants (async fn in tuple, async predicate + tuple then)
  G. Separate args=/kwargs= keyword params on if_()
"""
from __future__ import annotations

import unittest

from quent import Chain, Null


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add(a, b):
  return a + b

def kw_fn(*, x=0, y=0):
  return x * y

def mixed_fn(a, *, k=0):
  return a + k

def identity(v):
  return v

def double(v):
  return v * 2

def triple(v):
  return v * 3

def negate(v):
  return -v

async def async_add(a, b):
  return a + b

async def async_kw_fn(*, x=0, y=0):
  return x * y

async def async_mixed_fn(a, *, k=0):
  return a + k

async def async_double(v):
  return v * 2

async def async_triple(v):
  return v * 3

async def async_identity(v):
  return v

async def async_truthy_pred(v):
  return bool(v)

async def async_falsy_pred(v):
  return not bool(v)

async def async_gt5(v):
  return v > 5


# ============================================================================
# A. 2-tuple with positional args: then=(fn, (arg1, arg2))
# ============================================================================

class TestTwoTupleWithArgs(unittest.TestCase):
  """then=(fn, (args,)) — fn receives explicit args, NOT current_value."""

  def test_truthy_predicate_fn_receives_explicit_args(self):
    r = Chain(10).if_(lambda v: v > 5, then=(add, (100, 200))).run()
    self.assertEqual(r, 300)

  def test_falsy_predicate_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=(add, (100, 200))).run()
    self.assertEqual(r, 3)

  def test_single_arg_tuple(self):
    r = Chain(10).if_(lambda v: v > 5, then=(double, (42,))).run()
    self.assertEqual(r, 84)

  def test_current_value_not_passed_when_args_present(self):
    """Explicit args replace the current value — fn never sees current_value."""
    tracker = []

    def capture(*args):
      tracker.extend(args)
      return 'captured'

    r = Chain(10).if_(lambda v: True, then=(capture, (1, 2, 3))).run()
    self.assertEqual(r, 'captured')
    self.assertEqual(tracker, [1, 2, 3])

  def test_with_else_truthy_takes_if(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(add, (100, 200)))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 300)

  def test_with_else_falsy_takes_else(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=(add, (100, 200)))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 9)

  def test_empty_args_tuple_passes_current_value(self):
    """then=(fn, ()) — empty tuple is falsy, so current_value flows through as first arg."""
    r = Chain(10).if_(lambda v: True, then=(double, ())).run()
    self.assertEqual(r, 20)

  def test_ellipsis_in_tuple_calls_with_no_args(self):
    """then=(fn, (...,)) — Ellipsis convention forces zero-arg call."""

    def no_args():
      return 'no_args_result'

    r = Chain(10).if_(lambda v: True, then=(no_args, (...,))).run()
    self.assertEqual(r, 'no_args_result')

  def test_chained_after_then_step(self):
    r = (
      Chain(2)
      .then(lambda v: v + 8)
      .if_(lambda v: v > 5, then=(add, (50, 60)))
      .run()
    )
    self.assertEqual(r, 110)

  def test_followed_by_then_step(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(add, (50, 60)))
      .then(lambda v: v + 1)
      .run()
    )
    # 50+60=110, then 110+1=111
    self.assertEqual(r, 111)

  def test_run_provides_value(self):
    c = Chain().if_(lambda v: v > 5, then=(add, (100, 200)))
    self.assertEqual(c.run(10), 300)
    self.assertEqual(c.run(3), 3)


# ============================================================================
# B. 2-tuple with keyword args: then=(fn, {'key': val})
# ============================================================================

class TestTwoTupleWithKwargs(unittest.TestCase):
  """then=(fn, {'key': val}) — fn receives explicit kwargs, NOT current_value."""

  def test_truthy_predicate_fn_receives_explicit_kwargs(self):
    r = Chain(10).if_(lambda v: v > 5, then=(kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 21)

  def test_falsy_predicate_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=(kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 3)

  def test_current_value_not_passed_when_kwargs_present(self):
    tracker = {}

    def capture(**kw):
      tracker.update(kw)
      return 'captured'

    r = Chain(10).if_(lambda v: True, then=(capture, {'a': 1, 'b': 2})).run()
    self.assertEqual(r, 'captured')
    self.assertEqual(tracker, {'a': 1, 'b': 2})

  def test_with_else_truthy_takes_if(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(kw_fn, {'x': 4, 'y': 5}))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 20)

  def test_with_else_falsy_takes_else(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=(kw_fn, {'x': 4, 'y': 5}))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 9)

  def test_empty_kwargs_dict_passes_current_value(self):
    """then=(fn, {}) — empty dict is falsy, so current_value flows through as first arg."""
    r = Chain(10).if_(lambda v: True, then=(double, {})).run()
    self.assertEqual(r, 20)

  def test_run_provides_value(self):
    c = Chain().if_(lambda v: v > 5, then=(kw_fn, {'x': 3, 'y': 7}))
    self.assertEqual(c.run(10), 21)
    self.assertEqual(c.run(3), 3)


# ============================================================================
# C. 3-tuple: then=(fn, (args,), {'key': val})
# ============================================================================

class TestThreeTuple(unittest.TestCase):
  """then=(fn, (args,), {'key': val}) — fn receives both args and kwargs."""

  def test_truthy_predicate_fn_receives_args_and_kwargs(self):
    r = Chain(10).if_(lambda v: v > 5, then=(mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 150)

  def test_falsy_predicate_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=(mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 3)

  def test_current_value_not_passed(self):
    tracker = {'args': None, 'kwargs': None}

    def capture(*args, **kwargs):
      tracker['args'] = args
      tracker['kwargs'] = kwargs
      return 'captured'

    r = Chain(10).if_(lambda v: True, then=(capture, (1, 2), {'k': 3})).run()
    self.assertEqual(r, 'captured')
    self.assertEqual(tracker['args'], (1, 2))
    self.assertEqual(tracker['kwargs'], {'k': 3})

  def test_with_else_truthy_takes_if(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(mixed_fn, (10,), {'k': 20}))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 30)

  def test_with_else_falsy_takes_else(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=(mixed_fn, (10,), {'k': 20}))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 9)

  def test_empty_args_with_kwargs(self):
    r = Chain(10).if_(lambda v: True, then=(kw_fn, (), {'x': 5, 'y': 6})).run()
    self.assertEqual(r, 30)

  def test_args_with_empty_kwargs(self):
    r = Chain(10).if_(lambda v: True, then=(double, (42,), {})).run()
    self.assertEqual(r, 84)

  def test_run_provides_value(self):
    c = Chain().if_(lambda v: v > 5, then=(mixed_fn, (100,), {'k': 50}))
    self.assertEqual(c.run(10), 150)
    self.assertEqual(c.run(3), 3)


# ============================================================================
# D. predicate=None (truthiness check on current value)
# ============================================================================

class TestPredicateNone(unittest.TestCase):
  """predicate=None — uses truthiness of current pipeline value."""

  def test_truthy_int_triggers_then(self):
    r = Chain(10).if_(then=double).run()
    self.assertEqual(r, 20)

  def test_falsy_zero_skips_then(self):
    r = Chain(0).if_(then=double).run()
    self.assertEqual(r, 0)

  def test_falsy_none_skips_then(self):
    r = Chain(None).if_(then=lambda v: 'replaced').run()
    self.assertIsNone(r)

  def test_falsy_empty_string_skips_then(self):
    r = Chain('').if_(then=lambda v: 'replaced').run()
    self.assertEqual(r, '')

  def test_falsy_empty_list_skips_then(self):
    r = Chain([]).if_(then=lambda v: 'replaced').run()
    self.assertEqual(r, [])

  def test_falsy_false_skips_then(self):
    r = Chain(False).if_(then=lambda v: 'replaced').run()
    self.assertIs(r, False)

  def test_truthy_string_triggers_then(self):
    r = Chain('hello').if_(then=lambda v: v.upper()).run()
    self.assertEqual(r, 'HELLO')

  def test_truthy_list_triggers_then(self):
    r = Chain([1, 2, 3]).if_(then=lambda v: sum(v)).run()
    self.assertEqual(r, 6)

  def test_truthy_true_triggers_then(self):
    r = Chain(True).if_(then=lambda v: 'yes').run()
    self.assertEqual(r, 'yes')

  def test_with_else_truthy(self):
    r = Chain(10).if_(then=lambda v: 'truthy').else_(lambda v: 'falsy').run()
    self.assertEqual(r, 'truthy')

  def test_with_else_falsy_zero(self):
    r = Chain(0).if_(then=lambda v: 'truthy').else_(lambda v: 'falsy').run()
    self.assertEqual(r, 'falsy')

  def test_with_else_falsy_none(self):
    r = Chain(None).if_(then=lambda v: 'truthy').else_(lambda v: 'falsy').run()
    self.assertEqual(r, 'falsy')

  def test_with_else_falsy_false(self):
    r = Chain(False).if_(then=lambda v: 'truthy').else_(lambda v: 'falsy').run()
    self.assertEqual(r, 'falsy')

  def test_with_else_falsy_empty_string(self):
    r = Chain('').if_(then=lambda v: 'truthy').else_(lambda v: 'falsy').run()
    self.assertEqual(r, 'falsy')

  def test_with_else_falsy_empty_list(self):
    r = Chain([]).if_(then=lambda v: 'truthy').else_(lambda v: 'falsy').run()
    self.assertEqual(r, 'falsy')


class TestPredicateNoneWithTupleThen(unittest.TestCase):
  """predicate=None combined with tuple then= forms."""

  def test_truthy_with_2tuple_args(self):
    r = Chain(10).if_(then=(add, (100, 200))).run()
    self.assertEqual(r, 300)

  def test_falsy_with_2tuple_args_passthrough(self):
    r = Chain(0).if_(then=(add, (100, 200))).run()
    self.assertEqual(r, 0)

  def test_truthy_with_2tuple_kwargs(self):
    r = Chain(10).if_(then=(kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 21)

  def test_falsy_with_2tuple_kwargs_passthrough(self):
    r = Chain(0).if_(then=(kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 0)

  def test_truthy_with_3tuple(self):
    r = Chain(10).if_(then=(mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 150)

  def test_falsy_with_3tuple_passthrough(self):
    r = Chain(0).if_(then=(mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 0)

  def test_with_else_truthy_2tuple_args(self):
    r = Chain(10).if_(then=(add, (100, 200))).else_(triple).run()
    self.assertEqual(r, 300)

  def test_with_else_falsy_2tuple_args(self):
    r = Chain(0).if_(then=(add, (100, 200))).else_(triple).run()
    self.assertEqual(r, 0)

  def test_run_provides_truthy_value(self):
    c = Chain().if_(then=(add, (100, 200)))
    self.assertEqual(c.run(10), 300)

  def test_run_provides_falsy_value(self):
    c = Chain().if_(then=(add, (100, 200)))
    self.assertEqual(c.run(0), 0)


# ============================================================================
# E. Error cases (wrong-length tuples)
# ============================================================================

class TestTupleLengthErrors(unittest.TestCase):
  """Error cases for wrong-length then= tuples."""

  def test_empty_tuple_raises_type_error(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(10).if_(lambda v: True, then=())
    self.assertIn('2 or 3 elements', str(ctx.exception))
    self.assertIn('got 0', str(ctx.exception))

  def test_one_element_tuple_raises_type_error(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(10).if_(lambda v: True, then=(double,))
    self.assertIn('2 or 3 elements', str(ctx.exception))
    self.assertIn('got 1', str(ctx.exception))

  def test_four_element_tuple_raises_type_error(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(10).if_(lambda v: True, then=(double, (1,), {}, 'extra'))
    self.assertIn('2 or 3 elements', str(ctx.exception))
    self.assertIn('got 4', str(ctx.exception))

  def test_five_element_tuple_raises_type_error(self):
    with self.assertRaises(TypeError) as ctx:
      Chain(10).if_(lambda v: True, then=(double, (1,), {}, 'a', 'b'))
    self.assertIn('2 or 3 elements', str(ctx.exception))
    self.assertIn('got 5', str(ctx.exception))

  def test_error_raised_at_build_time_not_run_time(self):
    """TypeError is raised during chain construction, not during run()."""
    with self.assertRaises(TypeError):
      Chain(10).if_(lambda v: True, then=())
    # If we get here, the error was raised at build time (during .if_() call)


# ============================================================================
# F. Async variants
# ============================================================================

class TestAsyncTupleThen(unittest.IsolatedAsyncioTestCase):
  """Async fn inside tuple then= forms."""

  async def test_async_fn_in_2tuple_args(self):
    r = await Chain(10).if_(lambda v: v > 5, then=(async_add, (100, 200))).run()
    self.assertEqual(r, 300)

  async def test_async_fn_in_2tuple_args_falsy_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=(async_add, (100, 200))).run()
    self.assertEqual(r, 3)

  async def test_async_fn_in_2tuple_kwargs(self):
    r = await Chain(10).if_(lambda v: v > 5, then=(async_kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 21)

  async def test_async_fn_in_2tuple_kwargs_falsy_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=(async_kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 3)

  async def test_async_fn_in_3tuple(self):
    r = await Chain(10).if_(lambda v: v > 5, then=(async_mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 150)

  async def test_async_fn_in_3tuple_falsy_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=(async_mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 3)

  async def test_async_fn_in_2tuple_args_with_else(self):
    r = await (
      Chain(10)
      .if_(lambda v: v > 5, then=(async_add, (100, 200)))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 300)

  async def test_async_fn_in_2tuple_args_falsy_with_else(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=(async_add, (100, 200)))
      .else_(lambda v: v * 3)
      .run()
    )
    self.assertEqual(r, 9)

  async def test_async_fn_in_2tuple_kwargs_with_else(self):
    r = await (
      Chain(10)
      .if_(lambda v: v > 5, then=(async_kw_fn, {'x': 4, 'y': 5}))
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 20)

  async def test_async_fn_in_3tuple_with_else(self):
    r = await (
      Chain(10)
      .if_(lambda v: v > 5, then=(async_mixed_fn, (10,), {'k': 20}))
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 30)


class TestAsyncPredicateWithTupleThen(unittest.IsolatedAsyncioTestCase):
  """Async predicate combined with tuple then= forms."""

  async def test_async_pred_truthy_2tuple_args(self):
    r = await Chain(10).if_(async_gt5, then=(add, (100, 200))).run()
    self.assertEqual(r, 300)

  async def test_async_pred_falsy_2tuple_args_passthrough(self):
    r = await Chain(3).if_(async_gt5, then=(add, (100, 200))).run()
    self.assertEqual(r, 3)

  async def test_async_pred_truthy_2tuple_kwargs(self):
    r = await Chain(10).if_(async_gt5, then=(kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 21)

  async def test_async_pred_falsy_2tuple_kwargs_passthrough(self):
    r = await Chain(3).if_(async_gt5, then=(kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 3)

  async def test_async_pred_truthy_3tuple(self):
    r = await Chain(10).if_(async_gt5, then=(mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 150)

  async def test_async_pred_falsy_3tuple_passthrough(self):
    r = await Chain(3).if_(async_gt5, then=(mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 3)

  async def test_async_pred_with_else_truthy(self):
    r = await (
      Chain(10)
      .if_(async_gt5, then=(add, (100, 200)))
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 300)

  async def test_async_pred_with_else_falsy(self):
    r = await (
      Chain(3)
      .if_(async_gt5, then=(add, (100, 200)))
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 9)

  async def test_async_pred_async_fn_in_2tuple_args(self):
    r = await Chain(10).if_(async_gt5, then=(async_add, (100, 200))).run()
    self.assertEqual(r, 300)

  async def test_async_pred_async_fn_in_3tuple(self):
    r = await Chain(10).if_(async_gt5, then=(async_mixed_fn, (10,), {'k': 20})).run()
    self.assertEqual(r, 30)

  async def test_async_pred_falsy_async_fn_passthrough(self):
    r = await Chain(3).if_(async_gt5, then=(async_add, (100, 200))).run()
    self.assertEqual(r, 3)


class TestPredicateNoneAsync(unittest.IsolatedAsyncioTestCase):
  """predicate=None with async fn in tuple then= forms."""

  async def test_truthy_with_async_fn_bare(self):
    r = await Chain(10).if_(then=async_double).run()
    self.assertEqual(r, 20)

  async def test_falsy_with_async_fn_bare(self):
    r = Chain(0).if_(then=async_double).run()
    self.assertEqual(r, 0)

  async def test_truthy_with_async_fn_2tuple_args(self):
    r = await Chain(10).if_(then=(async_add, (100, 200))).run()
    self.assertEqual(r, 300)

  async def test_falsy_with_async_fn_2tuple_args(self):
    r = Chain(0).if_(then=(async_add, (100, 200))).run()
    self.assertEqual(r, 0)

  async def test_truthy_with_async_fn_2tuple_kwargs(self):
    r = await Chain(10).if_(then=(async_kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 21)

  async def test_falsy_with_async_fn_2tuple_kwargs(self):
    r = Chain(0).if_(then=(async_kw_fn, {'x': 3, 'y': 7})).run()
    self.assertEqual(r, 0)

  async def test_truthy_with_async_fn_3tuple(self):
    r = await Chain(10).if_(then=(async_mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 150)

  async def test_falsy_with_async_fn_3tuple(self):
    r = Chain(0).if_(then=(async_mixed_fn, (100,), {'k': 50})).run()
    self.assertEqual(r, 0)

  async def test_truthy_async_fn_with_else(self):
    r = await Chain(10).if_(then=(async_add, (100, 200))).else_(triple).run()
    self.assertEqual(r, 300)

  async def test_falsy_async_fn_with_else(self):
    r = Chain(0).if_(then=(async_add, (100, 200))).else_(triple).run()
    self.assertEqual(r, 0)


# ============================================================================
# G. Separate args=/kwargs= keyword params on if_()
# ============================================================================

class TestSeparateArgsKwargsParams(unittest.TestCase):
  """if_(pred, then=fn, args=(...), kwargs={...}) — keyword params on if_()."""

  def test_args_param_truthy(self):
    r = Chain(10).if_(lambda v: v > 5, then=add, args=(100, 200)).run()
    self.assertEqual(r, 300)

  def test_args_param_falsy_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=add, args=(100, 200)).run()
    self.assertEqual(r, 3)

  def test_kwargs_param_truthy(self):
    r = Chain(10).if_(lambda v: v > 5, then=kw_fn, kwargs={'x': 3, 'y': 7}).run()
    self.assertEqual(r, 21)

  def test_kwargs_param_falsy_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=kw_fn, kwargs={'x': 3, 'y': 7}).run()
    self.assertEqual(r, 3)

  def test_args_and_kwargs_params_truthy(self):
    r = Chain(10).if_(lambda v: v > 5, then=mixed_fn, args=(100,), kwargs={'k': 50}).run()
    self.assertEqual(r, 150)

  def test_args_and_kwargs_params_falsy_passthrough(self):
    r = Chain(3).if_(lambda v: v > 5, then=mixed_fn, args=(100,), kwargs={'k': 50}).run()
    self.assertEqual(r, 3)

  def test_args_param_with_else_truthy(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=add, args=(100, 200))
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 300)

  def test_args_param_with_else_falsy(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=add, args=(100, 200))
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 9)

  def test_kwargs_param_with_else_truthy(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=kw_fn, kwargs={'x': 4, 'y': 5})
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 20)

  def test_kwargs_param_with_else_falsy(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=kw_fn, kwargs={'x': 4, 'y': 5})
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 9)

  def test_args_and_kwargs_params_with_else_truthy(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=mixed_fn, args=(10,), kwargs={'k': 20})
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 30)

  def test_args_and_kwargs_params_with_else_falsy(self):
    r = (
      Chain(3)
      .if_(lambda v: v > 5, then=mixed_fn, args=(10,), kwargs={'k': 20})
      .else_(triple)
      .run()
    )
    self.assertEqual(r, 9)

  def test_ellipsis_arg_calls_fn_with_no_args(self):
    """args=(...,) — Ellipsis convention forces zero-arg call."""

    def no_args():
      return 'ellipsis_result'

    r = Chain(10).if_(lambda v: True, then=no_args, args=(...,)).run()
    self.assertEqual(r, 'ellipsis_result')

  def test_predicate_none_with_args_param(self):
    r = Chain(10).if_(then=add, args=(100, 200)).run()
    self.assertEqual(r, 300)

  def test_predicate_none_with_kwargs_param(self):
    r = Chain(10).if_(then=kw_fn, kwargs={'x': 3, 'y': 7}).run()
    self.assertEqual(r, 21)

  def test_predicate_none_with_args_and_kwargs_params(self):
    r = Chain(10).if_(then=mixed_fn, args=(100,), kwargs={'k': 50}).run()
    self.assertEqual(r, 150)


class TestSeparateArgsKwargsAsync(unittest.IsolatedAsyncioTestCase):
  """Async fn with separate args=/kwargs= keyword params."""

  async def test_async_fn_with_args_param(self):
    r = await Chain(10).if_(lambda v: v > 5, then=async_add, args=(100, 200)).run()
    self.assertEqual(r, 300)

  async def test_async_fn_with_kwargs_param(self):
    r = await Chain(10).if_(lambda v: v > 5, then=async_kw_fn, kwargs={'x': 3, 'y': 7}).run()
    self.assertEqual(r, 21)

  async def test_async_fn_with_args_and_kwargs_params(self):
    r = await Chain(10).if_(lambda v: v > 5, then=async_mixed_fn, args=(100,), kwargs={'k': 50}).run()
    self.assertEqual(r, 150)

  async def test_async_pred_with_args_param(self):
    r = await Chain(10).if_(async_gt5, then=add, args=(100, 200)).run()
    self.assertEqual(r, 300)

  async def test_async_pred_with_kwargs_param(self):
    r = await Chain(10).if_(async_gt5, then=kw_fn, kwargs={'x': 3, 'y': 7}).run()
    self.assertEqual(r, 21)


# ============================================================================
# H. Interaction with other chain methods
# ============================================================================

class TestTupleThenInteractions(unittest.TestCase):
  """Tuple then= forms interacting with other chain features."""

  def test_tuple_then_with_except_handler(self):
    def bad_fn(a, b):
      raise ValueError('boom')

    r = (
      Chain(10)
      .if_(lambda v: True, then=(bad_fn, (1, 2)))
      .except_(lambda rv, exc: 'caught')
      .run()
    )
    self.assertEqual(r, 'caught')

  def test_tuple_then_with_finally_handler(self):
    tracker = []
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(add, (100, 200)))
      .finally_(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(r, 300)
    self.assertEqual(tracker, [10])

  def test_tuple_then_preceded_by_then(self):
    r = (
      Chain(2)
      .then(lambda v: v * 5)
      .if_(lambda v: v > 5, then=(add, (100, 200)))
      .run()
    )
    self.assertEqual(r, 300)

  def test_tuple_then_followed_by_then(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(add, (100, 200)))
      .then(lambda v: v + 1)
      .run()
    )
    self.assertEqual(r, 301)

  def test_tuple_then_followed_by_do(self):
    tracker = []
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(add, (100, 200)))
      .do(lambda v: tracker.append(v))
      .run()
    )
    self.assertEqual(r, 300)
    self.assertEqual(tracker, [300])

  def test_multiple_if_with_tuple_then(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(add, (100, 200)))
      .if_(lambda v: v > 250, then=(kw_fn, {'x': 5, 'y': 6}))
      .run()
    )
    # 10 > 5 -> add(100,200) = 300; 300 > 250 -> kw_fn(x=5,y=6) = 30
    self.assertEqual(r, 30)

  def test_return_inside_tuple_then(self):
    r = (
      Chain(10)
      .if_(lambda v: v > 5, then=(lambda: Chain.return_(999), (...,)))
      .then(lambda v: 'should_not_reach')
      .run()
    )
    self.assertEqual(r, 999)


if __name__ == '__main__':
  unittest.main()
