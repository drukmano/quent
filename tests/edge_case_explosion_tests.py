"""Exhaustive edge case and boundary testing across the entire quent library.

Covers: falsy values, Ellipsis convention, empty chains, type preservation,
exception propagation, nested chains, concurrent frozen chains, large data,
special callable types, and run() arg matrix.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import functools
import threading
import unittest
from typing import Any

from quent import Chain, Null, QuentException
from quent._core import Link, _evaluate_value
from helpers import (
  AsyncRange,
  CallableObj,
  BoundMethodHolder,
  SyncCM,
  AsyncCM,
  partial_fn,
  make_tracker,
)


# ---------------------------------------------------------------------------
# PART 1: Null/None/Falsy value matrix (20+ tests)
# ---------------------------------------------------------------------------

FALSY_VALUES = [None, 0, False, '', [], {}, 0.0, 0j, set(), frozenset(), b'']


class TestFalsyValueMatrix(unittest.TestCase):
  """Every chain method with every falsy value type."""

  def test_chain_then_with_none(self):
    self.assertIsNone(Chain(None).then(lambda v: v).run())

  def test_chain_then_with_zero(self):
    self.assertEqual(Chain(0).then(lambda v: v).run(), 0)

  def test_chain_then_with_false(self):
    self.assertIs(Chain(False).then(lambda v: v).run(), False)

  def test_chain_then_with_empty_string(self):
    self.assertEqual(Chain('').then(lambda v: v).run(), '')

  def test_chain_then_with_empty_list(self):
    self.assertEqual(Chain([]).then(lambda v: v).run(), [])

  def test_chain_then_with_empty_dict(self):
    self.assertEqual(Chain({}).then(lambda v: v).run(), {})

  def test_chain_then_with_zero_float(self):
    self.assertEqual(Chain(0.0).then(lambda v: v).run(), 0.0)

  def test_chain_then_with_zero_complex(self):
    self.assertEqual(Chain(0j).then(lambda v: v).run(), 0j)

  def test_chain_then_with_empty_set(self):
    self.assertEqual(Chain(set()).then(lambda v: v).run(), set())

  def test_chain_then_with_empty_frozenset(self):
    self.assertEqual(Chain(frozenset()).then(lambda v: v).run(), frozenset())

  def test_chain_then_with_empty_bytes(self):
    self.assertEqual(Chain(b'').then(lambda v: v).run(), b'')

  def test_map_on_empty_list(self):
    result = Chain([]).map(lambda v: v).run()
    self.assertEqual(result, [])

  def test_map_on_empty_string(self):
    result = Chain('').map(lambda v: v).run()
    self.assertEqual(result, [])

  def test_map_on_empty_bytes(self):
    result = Chain(b'').map(lambda v: v).run()
    self.assertEqual(result, [])

  def test_filter_on_empty_list(self):
    result = Chain([]).filter(lambda v: True).run()
    self.assertEqual(result, [])

  def test_filter_on_empty_string(self):
    result = Chain('').filter(lambda v: True).run()
    self.assertEqual(result, [])

  def test_if_with_falsy_none(self):
    # predicate receives None (current value), bool(None) is False -> passthrough
    result = Chain(None).if_(lambda v: v, lambda v: 'yes').run()
    self.assertIsNone(result)

  def test_if_with_falsy_zero(self):
    result = Chain(0).if_(lambda v: v, lambda v: 'yes').run()
    self.assertEqual(result, 0)

  def test_if_with_falsy_false(self):
    result = Chain(False).if_(lambda v: v, lambda v: 'yes').run()
    self.assertIs(result, False)

  def test_if_with_falsy_empty_list(self):
    result = Chain([]).if_(lambda v: v, lambda v: 'yes').run()
    self.assertEqual(result, [])

  def test_run_with_each_falsy_value(self):
    for val in FALSY_VALUES:
      with self.subTest(val=val):
        result = Chain().run(val)
        self.assertEqual(result, val)

  def test_chain_init_vs_run_equivalence(self):
    """Chain(falsy).run() == Chain().run(falsy) for every falsy value."""
    for val in FALSY_VALUES:
      with self.subTest(val=val):
        a = Chain(val).run()
        b = Chain().run(val)
        self.assertEqual(a, b)
        self.assertEqual(type(a), type(b))

  def test_chain_then_identity_all_falsy(self):
    for val in FALSY_VALUES:
      with self.subTest(val=val):
        result = Chain(val).then(lambda v: v).run()
        self.assertEqual(result, val)

  def test_do_preserves_falsy_value(self):
    for val in FALSY_VALUES:
      with self.subTest(val=val):
        tracker = []
        result = Chain(val).do(lambda v: tracker.append(v)).run()
        self.assertEqual(result, val)
        self.assertEqual(tracker, [val])


# ---------------------------------------------------------------------------
# PART 2: Ellipsis convention matrix (15+ tests)
# ---------------------------------------------------------------------------

class TestEllipsisConvention(unittest.TestCase):
  """Test ... (Ellipsis) with every method."""

  def test_then_ellipsis_calls_fn_no_args(self):
    result = Chain(99).then(lambda: 42, ...).run()
    self.assertEqual(result, 42)

  def test_then_ellipsis_ignores_current_value(self):
    result = Chain('hello').then(lambda: 'world', ...).run()
    self.assertEqual(result, 'world')

  def test_do_ellipsis_calls_fn_no_args(self):
    tracker = []
    result = Chain(99).do(lambda: tracker.append('called'), ...).run()
    self.assertEqual(result, 99)
    self.assertEqual(tracker, ['called'])

  def test_if_fn_with_ellipsis(self):
    result = Chain(10).if_(lambda v: True, lambda: 42, ...).run()
    self.assertEqual(result, 42)

  def test_if_fn_with_ellipsis_false_predicate(self):
    result = Chain(10).if_(lambda v: False, lambda: 42, ...).run()
    self.assertEqual(result, 10)

  def test_nested_chain_ellipsis_with_root(self):
    """Ellipsis on a nested chain with its own root value."""
    inner = Chain(5).then(lambda v: v + 1)
    result = Chain(10).then(inner, ...).run()
    # With ..., inner._run(Null, None, None) is called
    # inner has root_link=Link(5), first_link=Link(lambda v: v+1)
    # Evaluates: 5 -> lambda(5) = 6
    self.assertEqual(result, 6)

  def test_chain_constructor_with_ellipsis(self):
    """Chain(callable, ...) calls callable with no args immediately."""
    result = Chain(lambda: 100, ...).run()
    self.assertEqual(result, 100)

  def test_run_with_ellipsis_is_literal(self):
    """run(...) creates Link(v=...) -- Ellipsis is not callable, returns literal."""
    # When run(...) is called, v=Ellipsis, has_run_value=True,
    # Link(Ellipsis) is created. Ellipsis is not callable, so returns Ellipsis.
    result = Chain(42).run(...)
    self.assertIs(result, ...)

  def test_ellipsis_in_except_handler(self):
    result = (
      Chain(lambda: (_ for _ in ()).throw(ValueError('oops')))
      .except_(lambda: 'recovered', ..., exceptions=ValueError)
      .run()
    )
    self.assertEqual(result, 'recovered')

  def test_ellipsis_in_finally_handler(self):
    tracker = []
    Chain(5).finally_(lambda: tracker.append('finally'), ...).run()
    self.assertEqual(tracker, ['finally'])

  def test_ellipsis_with_map_in_chain(self):
    # Use Ellipsis in chain step before map
    result = Chain(lambda: [1, 2, 3], ...).map(lambda v: v * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_ellipsis_with_filter_in_chain(self):
    result = Chain(lambda: [1, 2, 3, 4], ...).filter(lambda v: v > 2).run()
    self.assertEqual(result, [3, 4])

  def test_ellipsis_with_gather_in_chain(self):
    result = Chain(lambda: 10, ...).gather(lambda v: v + 1, lambda v: v + 2).run()
    self.assertEqual(result, [11, 12])

  def test_ellipsis_in_chain_constructor_callable(self):
    result = Chain(lambda: 'hello', ...).run()
    self.assertEqual(result, 'hello')

  def test_ellipsis_with_args_after(self):
    # Chain(fn, ...) -> fn is called with no args (Ellipsis convention)
    result = Chain(lambda: 99, ...).run()
    self.assertEqual(result, 99)

  def test_then_with_explicit_args_and_ellipsis(self):
    # Ellipsis as first arg means "call with no args", extra args ignored? No --
    # args = (...,) and args[0] is ... -> call v()
    result = Chain(5).then(lambda: 'no_args', ...).run()
    self.assertEqual(result, 'no_args')


class TestEllipsisAsync(unittest.IsolatedAsyncioTestCase):

  async def test_ellipsis_with_async_callable(self):
    async def afn():
      return 42

    result = await Chain(99).then(afn, ...).run()
    self.assertEqual(result, 42)

  async def test_ellipsis_async_in_if(self):
    async def afn():
      return 'replaced'

    result = await Chain(10).if_(lambda v: True, afn, ...).run()
    self.assertEqual(result, 'replaced')


# ---------------------------------------------------------------------------
# PART 3: Empty chain variations (10+ tests)
# ---------------------------------------------------------------------------

class TestEmptyChainVariations(unittest.TestCase):
  """Edge cases when chain has no root value or no steps."""

  def test_empty_chain_run(self):
    result = Chain().run()
    self.assertIsNone(result)

  def test_chain_then_no_root(self):
    result = Chain().then(lambda: 42).run()
    self.assertEqual(result, 42)

  def test_chain_do_no_root(self):
    tracker = []
    result = Chain().do(lambda: tracker.append('called')).run()
    self.assertIsNone(result)
    self.assertEqual(tracker, ['called'])

  def test_chain_if_no_root_with_run_value(self):
    """If there's no root but run(v) is provided, predicate gets v."""
    result = Chain().if_(lambda v: v > 5, lambda v: 'big').else_(lambda v: 'small').run(10)
    self.assertEqual(result, 'big')

  def test_chain_if_no_root_false_with_run_value(self):
    result = Chain().if_(lambda v: v > 5, lambda v: 'big').else_(lambda v: 'small').run(3)
    self.assertEqual(result, 'small')

  def test_chain_except_no_error(self):
    result = Chain().except_(lambda e: 'handled').run()
    self.assertIsNone(result)

  def test_chain_finally_handler_called(self):
    tracker = []
    result = Chain().finally_(lambda: tracker.append('finally')).run()
    self.assertIsNone(result)
    self.assertEqual(tracker, ['finally'])

  def test_chain_freeze_run(self):
    result = Chain().freeze().run()
    self.assertIsNone(result)

  def test_chain_run_with_value(self):
    result = Chain().run(5)
    self.assertEqual(result, 5)

  def test_chain_run_with_value_then(self):
    result = Chain().then(lambda v: v * 2).run(5)
    self.assertEqual(result, 10)

  def test_empty_chain_callable(self):
    c = Chain()
    result = c()
    self.assertIsNone(result)

  def test_empty_chain_callable_with_arg(self):
    c = Chain()
    result = c(42)
    self.assertEqual(result, 42)

  def test_empty_chain_with_except_and_finally(self):
    tracker = []
    result = (
      Chain()
      .except_(lambda e: 'handled')
      .finally_(lambda: tracker.append('done'))
      .run()
    )
    self.assertIsNone(result)
    self.assertEqual(tracker, ['done'])


# ---------------------------------------------------------------------------
# PART 4: Return value type preservation (15+ tests)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Point:
  x: int
  y: int


class CustomObj:
  def __init__(self, val):
    self.val = val


class TestReturnValueTypePreservation(unittest.TestCase):
  """Chain must preserve Python types through the pipeline."""

  def test_preserves_int(self):
    self.assertIsInstance(Chain(42).run(), int)
    self.assertEqual(Chain(42).run(), 42)

  def test_preserves_float(self):
    self.assertIsInstance(Chain(3.14).run(), float)
    self.assertEqual(Chain(3.14).run(), 3.14)

  def test_preserves_str(self):
    self.assertIsInstance(Chain('hello').run(), str)
    self.assertEqual(Chain('hello').run(), 'hello')

  def test_preserves_list(self):
    val = [1, 2, 3]
    result = Chain(val).run()
    self.assertIs(result, val)

  def test_preserves_dict(self):
    val = {'a': 1}
    result = Chain(val).run()
    self.assertIs(result, val)

  def test_preserves_tuple(self):
    result = Chain((1, 2)).run()
    self.assertEqual(result, (1, 2))

  def test_preserves_set(self):
    val = {1, 2, 3}
    result = Chain(val).run()
    self.assertIs(result, val)

  def test_preserves_bytes(self):
    result = Chain(b'\x00\x01').run()
    self.assertEqual(result, b'\x00\x01')

  def test_preserves_none(self):
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_preserves_bool(self):
    self.assertIs(Chain(True).run(), True)
    self.assertIs(Chain(False).run(), False)

  def test_preserves_complex(self):
    result = Chain(1 + 2j).run()
    self.assertEqual(result, 1 + 2j)

  def test_preserves_custom_object(self):
    obj = CustomObj(99)
    result = Chain(obj).run()
    self.assertIs(result, obj)

  def test_preserves_dataclass(self):
    p = Point(1, 2)
    result = Chain(p).run()
    self.assertIs(result, p)
    self.assertEqual(result.x, 1)
    self.assertEqual(result.y, 2)

  def test_map_returns_list(self):
    result = Chain((1, 2, 3)).map(lambda v: v * 2).run()
    self.assertIsInstance(result, list)
    self.assertEqual(result, [2, 4, 6])

  def test_filter_returns_list(self):
    result = Chain((1, 2, 3, 4)).filter(lambda v: v > 2).run()
    self.assertIsInstance(result, list)
    self.assertEqual(result, [3, 4])

  def test_gather_returns_list(self):
    result = Chain(5).gather(lambda v: v + 1, lambda v: v + 2).run()
    self.assertIsInstance(result, list)
    self.assertEqual(result, [6, 7])

  def test_if_passthrough_preserves_type(self):
    p = Point(3, 4)
    result = Chain(p).if_(lambda v: False, lambda v: 'replaced').run()
    self.assertIs(result, p)

  def test_then_identity_preserves(self):
    val = {'key': [1, 2, 3]}
    result = Chain(val).then(lambda v: v).run()
    self.assertIs(result, val)

  def test_preserves_through_multiple_then(self):
    val = [1, 2, 3]
    result = Chain(val).then(lambda v: v).then(lambda v: v).run()
    self.assertIs(result, val)

  def test_preserves_bytearray(self):
    val = bytearray(b'\x01\x02')
    result = Chain(val).run()
    self.assertIs(result, val)


# ---------------------------------------------------------------------------
# PART 5: Exception propagation matrix (20+ tests)
# ---------------------------------------------------------------------------

class TestExceptionPropagation(unittest.TestCase):
  """Every method raising every exception type."""

  def test_value_error_in_root(self):
    def bad():
      raise ValueError('root')
    with self.assertRaises(ValueError):
      Chain(bad).run()

  def test_type_error_in_root(self):
    def bad():
      raise TypeError('root')
    with self.assertRaises(TypeError):
      Chain(bad).run()

  def test_runtime_error_in_root(self):
    def bad():
      raise RuntimeError('root')
    with self.assertRaises(RuntimeError):
      Chain(bad).run()

  def test_key_error_in_root(self):
    def bad():
      raise KeyError('k')
    with self.assertRaises(KeyError):
      Chain(bad).run()

  def test_index_error_in_root(self):
    def bad():
      raise IndexError('i')
    with self.assertRaises(IndexError):
      Chain(bad).run()

  def test_attribute_error_in_root(self):
    def bad():
      raise AttributeError('a')
    with self.assertRaises(AttributeError):
      Chain(bad).run()

  def test_exception_in_then(self):
    def bad(v):
      raise RuntimeError('then err')
    with self.assertRaises(RuntimeError):
      Chain(5).then(bad).run()

  def test_exception_in_do(self):
    def bad(v):
      raise RuntimeError('do err')
    with self.assertRaises(RuntimeError):
      Chain(5).do(bad).run()

  def test_exception_in_map(self):
    def bad(v):
      raise ValueError('map err')
    with self.assertRaises(ValueError):
      Chain([1, 2, 3]).map(bad).run()

  def test_exception_in_filter(self):
    def bad(v):
      raise ValueError('filter err')
    with self.assertRaises(ValueError):
      Chain([1, 2, 3]).filter(bad).run()

  def test_exception_in_gather(self):
    def bad(v):
      raise ValueError('gather err')
    with self.assertRaises(ValueError):
      Chain(5).gather(bad, lambda v: v).run()

  def test_exception_in_with_body(self):
    def bad(ctx):
      raise ValueError('body err')
    cm = SyncCM()
    with self.assertRaises(ValueError):
      Chain(cm).with_(bad).run()
    self.assertTrue(cm.exited)

  def test_exception_in_if_predicate(self):
    def bad_pred(v):
      raise RuntimeError('pred err')
    with self.assertRaises(RuntimeError):
      Chain(5).if_(bad_pred, lambda v: v).run()

  def test_exception_in_if_fn(self):
    def bad_fn(v):
      raise RuntimeError('if fn err')
    with self.assertRaises(RuntimeError):
      Chain(5).if_(lambda v: True, bad_fn).run()

  def test_exception_in_else_fn(self):
    def bad_fn(v):
      raise RuntimeError('else fn err')
    with self.assertRaises(RuntimeError):
      Chain(5).if_(lambda v: False, lambda v: v).else_(bad_fn).run()

  def test_except_handler_double_fault(self):
    def bad_handler(exc):
      raise RuntimeError('handler err')
    with self.assertRaises(RuntimeError) as ctx:
      Chain(lambda: (_ for _ in ()).throw(ValueError('orig'))).except_(bad_handler).run()
    self.assertIsInstance(ctx.exception.__cause__, ValueError)

  def test_exception_in_finally_handler(self):
    def bad_handler():
      raise RuntimeError('finally err')
    with self.assertRaises(RuntimeError):
      Chain(5).finally_(bad_handler, ...).run()

  def test_except_receives_correct_type(self):
    received = []

    def handler(exc):
      received.append(type(exc))
      return 'handled'

    result = (
      Chain(lambda: (_ for _ in ()).throw(ValueError('test')))
      .except_(handler)
      .run()
    )
    self.assertEqual(result, 'handled')
    self.assertEqual(received, [ValueError])

  def test_except_filters_exception_types(self):
    with self.assertRaises(TypeError):
      (
        Chain(lambda: (_ for _ in ()).throw(TypeError('test')))
        .except_(lambda e: 'handled', exceptions=ValueError)
        .run()
      )

  def test_context_chain_preserved(self):
    """__cause__ is set on double faults."""
    def bad_handler(exc):
      raise RuntimeError('double fault')
    try:
      Chain(lambda: (_ for _ in ()).throw(ValueError('orig'))).except_(bad_handler).run()
    except RuntimeError as exc:
      self.assertIsInstance(exc.__cause__, ValueError)

  def test_system_exit_propagates(self):
    def raise_exit():
      raise SystemExit(1)
    with self.assertRaises(SystemExit):
      Chain(raise_exit).run()

  def test_keyboard_interrupt_propagates(self):
    def raise_ki():
      raise KeyboardInterrupt()
    with self.assertRaises(KeyboardInterrupt):
      Chain(raise_ki).run()

  def test_base_exception_not_caught_by_default_except(self):
    """Default except_ only catches Exception subclasses, not BaseException."""
    def raise_base():
      raise KeyboardInterrupt()
    with self.assertRaises(KeyboardInterrupt):
      Chain(raise_base).except_(lambda e: 'handled').run()

  def test_base_exception_caught_when_specified(self):
    result = (
      Chain(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
      .except_(lambda e: 'handled', exceptions=KeyboardInterrupt)
      .run()
    )
    self.assertEqual(result, 'handled')

  def test_exception_in_then_after_multiple_steps(self):
    def bad(v):
      raise ValueError('late err')
    with self.assertRaises(ValueError):
      Chain(1).then(lambda v: v + 1).then(lambda v: v + 1).then(bad).run()

  def test_exception_preserves_message(self):
    def bad():
      raise ValueError('specific message')
    try:
      Chain(bad).run()
    except ValueError as e:
      self.assertEqual(str(e), 'specific message')


class TestExceptionPropagationAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_exception_in_then(self):
    async def bad(v):
      raise ValueError('async then err')
    with self.assertRaises(ValueError):
      await Chain(5).then(bad).run()

  async def test_async_exception_in_map(self):
    async def bad(v):
      raise ValueError('async map err')
    with self.assertRaises(ValueError):
      await Chain([1, 2]).map(bad).run()

  async def test_async_exception_in_filter(self):
    async def bad(v):
      raise ValueError('async filter err')
    with self.assertRaises(ValueError):
      await Chain([1, 2]).filter(bad).run()

  async def test_async_except_handler(self):
    async def bad(v):
      raise ValueError('async err')

    result = await (
      Chain(5)
      .then(bad)
      .except_(lambda e: 'recovered')
      .run()
    )
    self.assertEqual(result, 'recovered')


# ---------------------------------------------------------------------------
# PART 6: Nested chain depth (8+ tests)
# ---------------------------------------------------------------------------

class TestNestedChainDepth(unittest.TestCase):
  """Various nesting patterns and depths."""

  def test_2_deep(self):
    inner = Chain().then(lambda v: v * 2)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 10)

  def test_3_deep(self):
    level3 = Chain().then(lambda v: v + 1)
    level2 = Chain().then(level3)
    result = Chain(5).then(level2).run()
    self.assertEqual(result, 6)

  def test_chain_as_fn_argument(self):
    inner = Chain().then(lambda v: v + 1)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 6)

  def test_nested_chain_with_if(self):
    inner = Chain().then(lambda v: v * 10)
    result = Chain(5).if_(lambda v: v > 3, inner).run()
    self.assertEqual(result, 50)

  def test_nested_chain_with_if_false(self):
    inner = Chain().then(lambda v: v * 10)
    result = Chain(5).if_(lambda v: v > 10, inner).run()
    self.assertEqual(result, 5)

  def test_nested_chain_with_except(self):
    inner = Chain().then(lambda exc: f'caught: {exc}')
    result = (
      Chain(lambda: (_ for _ in ()).throw(ValueError('boom')))
      .except_(inner)
      .run()
    )
    self.assertEqual(result, 'caught: boom')

  def test_nested_chain_with_finally(self):
    tracker = []
    inner = Chain().then(lambda v: tracker.append('inner_finally'))
    Chain(5).finally_(inner).run()
    self.assertEqual(tracker, ['inner_finally'])

  def test_nested_chain_in_map_via_freeze(self):
    inner = Chain().then(lambda v: v * 2)
    # FrozenChain is treated as a regular callable (not sub-chain)
    result = Chain([1, 2, 3]).map(inner.freeze()).run()
    self.assertEqual(result, [2, 4, 6])

  def test_deep_nesting_5_levels(self):
    c = Chain().then(lambda v: v + 1)
    for _ in range(4):
      c = Chain().then(c)
    result = Chain(0).then(c).run()
    self.assertEqual(result, 1)

  def test_sibling_nested_chains(self):
    a = Chain().then(lambda v: v + 10)
    b = Chain().then(lambda v: v * 2)
    result = Chain(5).then(a).then(b).run()
    self.assertEqual(result, 30)

  def test_nested_chain_with_else(self):
    inner_true = Chain().then(lambda v: 'big')
    inner_false = Chain().then(lambda v: 'small')
    result = (
      Chain(3)
      .if_(lambda v: v > 5, inner_true)
      .else_(inner_false)
      .run()
    )
    self.assertEqual(result, 'small')

  def test_nested_chain_value_propagation(self):
    inner = Chain().then(lambda v: v + 100)
    result = Chain(5).then(lambda v: v * 2).then(inner).run()
    self.assertEqual(result, 110)


class TestNestedChainDepthAsync(unittest.IsolatedAsyncioTestCase):

  async def test_nested_async_chain(self):
    async def add_one(v):
      return v + 1

    inner = Chain().then(add_one)
    result = await Chain(5).then(inner).run()
    self.assertEqual(result, 6)

  async def test_3_deep_async(self):
    async def mul(v):
      return v * 2

    l3 = Chain().then(mul)
    l2 = Chain().then(l3)
    result = await Chain(3).then(l2).run()
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# PART 7: Concurrent frozen chain usage (6+ tests)
# ---------------------------------------------------------------------------

class TestConcurrentFrozenChain(unittest.TestCase):
  """Multiple threads calling frozen chain simultaneously."""

  def test_frozen_chain_concurrent_reads(self):
    # Use .then() so fn gets the run value
    fc = Chain().then(lambda v: v * 2).freeze()
    results = []
    errors = []

    def worker(val):
      try:
        results.append(fc(val))
      except Exception as e:
        errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(errors, [])
    self.assertEqual(sorted(results), [i * 2 for i in range(20)])

  def test_frozen_with_if_concurrent(self):
    fc = (
      Chain()
      .if_(lambda v: v > 5, lambda v: v * 10)
      .else_(lambda v: v * -1)
      .freeze()
    )
    results = {}
    errors = []

    def worker(val):
      try:
        results[val] = fc(val)
      except Exception as e:
        errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(errors, [])
    for i in range(20):
      expected = i * 10 if i > 5 else i * -1
      self.assertEqual(results[i], expected)

  def test_frozen_with_map_concurrent(self):
    fc = Chain().then(lambda v: list(range(v))).map(lambda x: x * 2).freeze()
    results = {}
    errors = []

    def worker(val):
      try:
        results[val] = fc(val)
      except Exception as e:
        errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 11)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(errors, [])
    for i in range(1, 11):
      self.assertEqual(results[i], [x * 2 for x in range(i)])

  def test_frozen_thread_pool_executor(self):
    fc = Chain().then(lambda v: v ** 2).freeze()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
      futures = [executor.submit(fc, i) for i in range(50)]
      results = [f.result() for f in futures]
    self.assertEqual(sorted(results), [i ** 2 for i in range(50)])

  def test_frozen_filter_concurrent(self):
    fc = Chain().then(lambda v: list(range(v))).filter(lambda x: x % 2 == 0).freeze()
    results = {}
    errors = []

    def worker(val):
      try:
        results[val] = fc(val)
      except Exception as e:
        errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 11)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertEqual(errors, [])
    for i in range(1, 11):
      self.assertEqual(results[i], [x for x in range(i) if x % 2 == 0])

  def test_frozen_gather_concurrent(self):
    fc = Chain().gather(lambda v: v + 1, lambda v: v * 2).freeze()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
      futures = {i: executor.submit(fc, i) for i in range(20)}
      for i, f in futures.items():
        self.assertEqual(f.result(), [i + 1, i * 2])


# ---------------------------------------------------------------------------
# PART 8: Large data (6+ tests)
# ---------------------------------------------------------------------------

class TestLargeData(unittest.TestCase):
  """Stress tests with large collections and deep chains."""

  def test_map_10000_elements(self):
    result = Chain(list(range(10000))).map(lambda v: v + 1).run()
    self.assertEqual(len(result), 10000)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[-1], 10000)

  def test_filter_10000_elements(self):
    result = Chain(list(range(10000))).filter(lambda v: v % 2 == 0).run()
    self.assertEqual(len(result), 5000)
    self.assertEqual(result[0], 0)
    self.assertEqual(result[-1], 9998)

  def test_gather_100_functions(self):
    fns = [lambda v, i=i: v + i for i in range(100)]
    result = Chain(0).gather(*fns).run()
    self.assertEqual(len(result), 100)
    self.assertEqual(result, list(range(100)))

  def test_deep_chain_100_steps(self):
    c = Chain(0)
    for _ in range(100):
      c = c.then(lambda v: v + 1)
    result = c.run()
    self.assertEqual(result, 100)

  def test_deep_chain_100_do_steps(self):
    tracker = []
    c = Chain(0)
    for i in range(100):
      c = c.do(lambda v, i=i: tracker.append(i))
    result = c.run()
    self.assertEqual(result, 0)
    self.assertEqual(len(tracker), 100)

  def test_large_nested_chain_10_levels(self):
    c = Chain().then(lambda v: v + 1)
    for _ in range(9):
      c = Chain().then(c)
    result = Chain(0).then(c).run()
    self.assertEqual(result, 1)

  def test_map_with_large_break(self):
    count = [0]

    def fn(v):
      count[0] += 1
      if count[0] > 500:
        Chain.break_()
      return v

    result = Chain(list(range(10000))).map(fn).run()
    # fn appends result for items 0..499 (count goes 1..500), breaks at count=501
    self.assertEqual(len(result), 500)

  def test_filter_large_all_pass(self):
    result = Chain(list(range(5000))).filter(lambda v: True).run()
    self.assertEqual(len(result), 5000)

  def test_filter_large_none_pass(self):
    result = Chain(list(range(5000))).filter(lambda v: False).run()
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# PART 9: Special callable types (10+ tests)
# ---------------------------------------------------------------------------

class TestSpecialCallableTypes(unittest.TestCase):
  """Various callable objects with Chain."""

  def test_functools_partial(self):
    result = Chain(5).then(partial_fn).run()
    self.assertEqual(result, 15)  # partial(operator.add, 10)(5) = 15

  def test_bound_method(self):
    holder = BoundMethodHolder()
    result = Chain(5).then(holder.method).run()
    self.assertEqual(result, 6)

  def test_callable_object(self):
    obj = CallableObj()
    result = Chain(5).then(obj).run()
    self.assertEqual(result, 6)

  def test_lambda(self):
    result = Chain(5).then(lambda v: v * 3).run()
    self.assertEqual(result, 15)

  def test_builtin_len(self):
    result = Chain([1, 2, 3]).then(len).run()
    self.assertEqual(result, 3)

  def test_builtin_str(self):
    result = Chain(42).then(str).run()
    self.assertEqual(result, '42')

  def test_builtin_int(self):
    result = Chain('42').then(int).run()
    self.assertEqual(result, 42)

  def test_builtin_list(self):
    result = Chain((1, 2, 3)).then(list).run()
    self.assertEqual(result, [1, 2, 3])

  def test_builtin_bool(self):
    result = Chain(1).then(bool).run()
    self.assertIs(result, True)

  def test_builtin_type(self):
    result = Chain(42).then(type).run()
    self.assertIs(result, int)

  def test_class_constructor(self):
    result = Chain(5).then(CustomObj).run()
    self.assertIsInstance(result, CustomObj)
    self.assertEqual(result.val, 5)

  def test_staticmethod_callable(self):
    class MyClass:
      @staticmethod
      def add_one(v):
        return v + 1
    result = Chain(5).then(MyClass.add_one).run()
    self.assertEqual(result, 6)

  def test_classmethod_callable(self):
    class MyClass:
      factor = 2

      @classmethod
      def multiply(cls, v):
        return v * cls.factor

    result = Chain(5).then(MyClass.multiply).run()
    self.assertEqual(result, 10)

  def test_generator_function_as_root_for_map(self):
    def gen():
      yield 1
      yield 2
      yield 3

    result = Chain(gen()).map(lambda v: v * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_functools_partial_in_map(self):
    add5 = functools.partial(lambda a, b: a + b, 5)
    result = Chain([1, 2, 3]).map(add5).run()
    self.assertEqual(result, [6, 7, 8])

  def test_callable_object_in_filter(self):
    class IsEven:
      def __call__(self, v):
        return v % 2 == 0

    result = Chain([1, 2, 3, 4]).filter(IsEven()).run()
    self.assertEqual(result, [2, 4])

  def test_builtin_abs(self):
    result = Chain(-5).then(abs).run()
    self.assertEqual(result, 5)

  def test_builtin_sorted(self):
    result = Chain([3, 1, 2]).then(sorted).run()
    self.assertEqual(result, [1, 2, 3])


class TestSpecialCallableAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_generator_as_source_for_map(self):
    async def agen():
      for i in range(5):
        yield i

    result = await Chain(agen()).map(lambda v: v * 2).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  async def test_async_callable_object(self):
    class AsyncMul:
      async def __call__(self, v):
        return v * 3

    result = await Chain(5).then(AsyncMul()).run()
    self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# PART 10: run() with args/kwargs matrix (8+ tests)
# ---------------------------------------------------------------------------

class TestRunArgsKwargsMatrix(unittest.TestCase):
  """Various argument patterns for chain.run().

  Key insight: run(v, *args, **kwargs) creates Link(v, args, kwargs) as the
  new root. The original Chain's root_link is bypassed. So Chain(fn).run(5)
  evaluates Link(5), not fn(5). To call fn with args, use Chain().then(fn).run(args).
  """

  def test_run_value_becomes_root(self):
    """run(v) makes v the root, bypassing Chain's own root_link."""
    result = Chain(42).run(99)
    self.assertEqual(result, 99)

  def test_run_callable_value_gets_called(self):
    """run(callable) calls it since callable with no current_value -> callable()."""
    result = Chain().run(lambda: 42)
    self.assertEqual(result, 42)

  def test_run_with_then_step(self):
    """run(v) feeds v into first_link's fn."""
    result = Chain().then(lambda v: v * 2).run(5)
    self.assertEqual(result, 10)

  def test_run_with_multiple_args_into_then(self):
    """run(fn, arg1, arg2) creates Link(fn, (arg1, arg2)), calling fn(arg1, arg2)."""
    result = Chain().run(lambda a, b: a + b, 3, 4)
    self.assertEqual(result, 7)

  def test_run_with_kwargs_into_then(self):
    result = Chain().run(lambda x=0, y=0: x + y, x=3, y=4)
    self.assertEqual(result, 7)

  def test_run_ellipsis_is_literal_value(self):
    """run(...) creates Link(Ellipsis). Ellipsis is not callable, returned as-is."""
    result = Chain().run(...)
    self.assertIs(result, ...)

  def test_call_is_alias_for_run(self):
    c = Chain().then(lambda v: v + 10)
    self.assertEqual(c(5), c.run(5))

  def test_call_with_callable_arg(self):
    c = Chain().then(lambda v: v + 1)
    result = c(lambda: 10)
    self.assertEqual(result, 11)

  def test_run_value_overrides_root(self):
    # Chain(100) has root_link=Link(100). run(200) bypasses it.
    result = Chain(100).then(lambda v: v + 1).run(200)
    # Link(200).next_link = first_link=Link(lambda v: v+1)
    # 200 -> lambda(200) = 201
    self.assertEqual(result, 201)

  def test_run_with_fn_and_args(self):
    result = Chain().run(lambda a, b, c: a + b + c, 1, 2, 3)
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# Additional edge cases: Chain boolean, repr, freeze, etc.
# ---------------------------------------------------------------------------

class TestChainMiscEdgeCases(unittest.TestCase):
  """Miscellaneous edge cases not fitting neatly in the other categories."""

  def test_chain_bool_always_true(self):
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(0)))
    self.assertTrue(bool(Chain(None)))
    self.assertTrue(bool(Chain(False)))

  def test_frozen_chain_bool_always_true(self):
    self.assertTrue(bool(Chain().freeze()))
    self.assertTrue(bool(Chain(0).freeze()))

  def test_chain_repr_basic(self):
    r = repr(Chain(42))
    self.assertIn('Chain(', r)

  def test_chain_repr_with_steps(self):
    c = Chain(42).then(lambda v: v)
    r = repr(c)
    self.assertIn('Chain(', r)

  def test_frozen_repr(self):
    fc = Chain(42).freeze()
    r = repr(fc)
    self.assertIn('Frozen(', r)

  def test_freeze_preserves_behavior(self):
    c = Chain(5).then(lambda v: v * 2)
    fc = c.freeze()
    self.assertEqual(fc.run(), 10)

  def test_freeze_run_with_arg(self):
    fc = Chain().then(lambda v: v * 3).freeze()
    self.assertEqual(fc(5), 15)

  def test_double_except_raises(self):
    with self.assertRaises(QuentException):
      Chain().except_(lambda e: e).except_(lambda e: e)

  def test_double_finally_raises(self):
    with self.assertRaises(QuentException):
      Chain().finally_(lambda: None).finally_(lambda: None)

  def test_else_without_if_raises(self):
    with self.assertRaises(QuentException):
      Chain().else_(lambda v: v)

  def test_else_after_then_raises(self):
    with self.assertRaises(QuentException):
      Chain().then(lambda v: v).else_(lambda v: v)

  def test_chain_with_none_root_and_then(self):
    result = Chain(None).then(lambda v: v is None).run()
    self.assertTrue(result)

  def test_do_discards_result(self):
    result = Chain(5).do(lambda v: v * 100).run()
    self.assertEqual(result, 5)

  def test_do_side_effect_executed(self):
    tracker = []
    Chain(5).do(lambda v: tracker.append(v)).run()
    self.assertEqual(tracker, [5])

  def test_multiple_do_all_executed(self):
    tracker = []
    Chain(5).do(lambda v: tracker.append('a')).do(lambda v: tracker.append('b')).run()
    self.assertEqual(tracker, ['a', 'b'])

  def test_return_in_nested_chain_propagates(self):
    inner = Chain().then(lambda v: Chain.return_(99))
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 99)

  def test_break_outside_map_raises(self):
    with self.assertRaises(QuentException):
      Chain(5).then(lambda v: Chain.break_()).run()

  def test_chain_with_generator_iterable(self):
    result = Chain(range(5)).map(lambda v: v * 2).run()
    self.assertEqual(result, [0, 2, 4, 6, 8])

  def test_except_returns_none_via_null(self):
    """When except handler returns Null, chain returns None."""
    from quent._core import Null as NullSentinel
    result = (
      Chain(lambda: (_ for _ in ()).throw(ValueError('test')))
      .except_(lambda e: NullSentinel)
      .run()
    )
    self.assertIsNone(result)

  def test_chain_then_returns_self(self):
    c = Chain()
    self.assertIs(c.then(lambda v: v), c)

  def test_chain_do_returns_self(self):
    c = Chain()
    self.assertIs(c.do(lambda v: v), c)

  def test_chain_except_returns_self(self):
    c = Chain()
    self.assertIs(c.except_(lambda e: e), c)

  def test_chain_finally_returns_self(self):
    c = Chain()
    self.assertIs(c.finally_(lambda: None), c)


# ---------------------------------------------------------------------------
# More targeted edge cases for high coverage
# ---------------------------------------------------------------------------

class TestEvaluateValueEdgeCases(unittest.TestCase):
  """Direct tests of _evaluate_value edge paths."""

  def test_link_with_non_callable_value(self):
    link = Link(42)
    result = _evaluate_value(link)
    self.assertEqual(result, 42)

  def test_link_with_callable_and_null(self):
    link = Link(lambda: 99)
    result = _evaluate_value(link)
    self.assertEqual(result, 99)

  def test_link_with_callable_and_current_value(self):
    link = Link(lambda v: v + 1)
    result = _evaluate_value(link, 5)
    self.assertEqual(result, 6)

  def test_link_with_args(self):
    link = Link(lambda a, b: a + b, (3, 4))
    result = _evaluate_value(link)
    self.assertEqual(result, 7)

  def test_link_with_kwargs(self):
    link = Link(lambda x=0, y=0: x + y, None, {'x': 3, 'y': 4})
    result = _evaluate_value(link)
    self.assertEqual(result, 7)

  def test_link_with_ellipsis_calls_no_args(self):
    link = Link(lambda: 42, (...,))
    result = _evaluate_value(link)
    self.assertEqual(result, 42)

  def test_link_non_callable_returns_value(self):
    link = Link('hello')
    result = _evaluate_value(link)
    self.assertEqual(result, 'hello')

  def test_link_non_callable_ignores_current_value(self):
    link = Link('hello')
    result = _evaluate_value(link, 999)
    self.assertEqual(result, 'hello')


class TestWithContextManagerEdgeCases(unittest.TestCase):
  """Context manager edge cases via with_."""

  def test_with_returns_body_result(self):
    cm = SyncCM()
    result = Chain(cm).with_(lambda ctx: ctx + '_modified').run()
    self.assertEqual(result, 'ctx_value_modified')

  def test_with_do_returns_original(self):
    cm = SyncCM()
    result = Chain(cm).with_do(lambda ctx: ctx + '_ignored').run()
    self.assertIs(result, cm)

  def test_with_body_exception_exits_cm(self):
    cm = SyncCM()
    with self.assertRaises(ValueError):
      Chain(cm).with_(lambda ctx: (_ for _ in ()).throw(ValueError('err'))).run()
    self.assertTrue(cm.exited)

  def test_with_enter_and_exit_called(self):
    cm = SyncCM()
    Chain(cm).with_(lambda ctx: ctx).run()
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


class TestWithContextManagerAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_cm_with_body(self):
    cm = AsyncCM()
    result = await Chain(cm).with_(lambda ctx: ctx + '_modified').run()
    self.assertEqual(result, 'ctx_value_modified')
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)


class TestIfElseEdgeCases(unittest.TestCase):
  """Edge cases for if_/else_ chains."""

  def test_if_with_none_value(self):
    result = Chain(None).if_(lambda v: v is None, lambda v: 'was_none').run()
    self.assertEqual(result, 'was_none')

  def test_if_else_chain(self):
    result = (
      Chain(3)
      .if_(lambda v: v > 5, lambda v: 'big')
      .else_(lambda v: 'small')
      .run()
    )
    self.assertEqual(result, 'small')

  def test_if_else_true_branch(self):
    result = (
      Chain(10)
      .if_(lambda v: v > 5, lambda v: 'big')
      .else_(lambda v: 'small')
      .run()
    )
    self.assertEqual(result, 'big')

  def test_chained_if_else(self):
    result = (
      Chain(5)
      .if_(lambda v: v > 10, lambda v: 'very_big')
      .else_(lambda v: v)
      .if_(lambda v: v > 3, lambda v: 'medium')
      .else_(lambda v: 'small')
      .run()
    )
    self.assertEqual(result, 'medium')

  def test_if_with_plain_value_fn(self):
    result = Chain(10).if_(lambda v: v > 5, 42).run()
    self.assertEqual(result, 42)

  def test_if_false_plain_value_passthrough(self):
    result = Chain(3).if_(lambda v: v > 5, 42).run()
    self.assertEqual(result, 3)


class TestMapEdgeCases(unittest.TestCase):
  """Edge cases for map."""

  def test_foreach_preserves_elements(self):
    tracker = []
    result = Chain([1, 2, 3]).foreach(lambda v: tracker.append(v * 10)).run()
    self.assertEqual(result, [1, 2, 3])
    self.assertEqual(tracker, [10, 20, 30])

  def test_map_on_string(self):
    result = Chain('abc').map(lambda c: c.upper()).run()
    self.assertEqual(result, ['A', 'B', 'C'])

  def test_map_on_tuple(self):
    result = Chain((1, 2, 3)).map(lambda v: v * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_map_on_set(self):
    result = Chain({3, 1, 2}).map(lambda v: v * 2).run()
    self.assertEqual(sorted(result), [2, 4, 6])

  def test_map_on_dict_iterates_keys(self):
    result = Chain({'a': 1, 'b': 2}).map(lambda k: k.upper()).run()
    self.assertEqual(sorted(result), ['A', 'B'])

  def test_map_break_returns_partial(self):
    count = [0]

    def fn(v):
      count[0] += 1
      if count[0] > 2:
        Chain.break_()
      return v * 2

    result = Chain([1, 2, 3, 4, 5]).map(fn).run()
    self.assertEqual(result, [2, 4])

  def test_map_break_with_value(self):
    def fn(v):
      if v == 3:
        Chain.break_([99])
      return v

    result = Chain([1, 2, 3, 4]).map(fn).run()
    self.assertEqual(result, [99])

  def test_map_single_element(self):
    result = Chain([42]).map(lambda v: v * 2).run()
    self.assertEqual(result, [84])

  def test_map_on_range(self):
    result = Chain(range(5)).map(lambda v: v + 10).run()
    self.assertEqual(result, [10, 11, 12, 13, 14])


class TestGatherEdgeCases(unittest.TestCase):
  """Edge cases for gather."""

  def test_gather_single_fn(self):
    result = Chain(5).gather(lambda v: v + 1).run()
    self.assertEqual(result, [6])

  def test_gather_many_fns(self):
    fns = [lambda v, i=i: v + i for i in range(10)]
    result = Chain(0).gather(*fns).run()
    self.assertEqual(result, list(range(10)))

  def test_gather_with_exception_propagates(self):
    def bad(v):
      raise ValueError('gather err')
    with self.assertRaises(ValueError):
      Chain(5).gather(lambda v: v, bad).run()

  def test_gather_all_same_fn(self):
    result = Chain(5).gather(lambda v: v, lambda v: v, lambda v: v).run()
    self.assertEqual(result, [5, 5, 5])


class TestGatherAsync(unittest.IsolatedAsyncioTestCase):

  async def test_gather_async_fns(self):
    async def add1(v):
      return v + 1

    async def mul2(v):
      return v * 2

    result = await Chain(5).gather(add1, mul2).run()
    self.assertEqual(result, [6, 10])

  async def test_gather_mixed_sync_async(self):
    async def async_add(v):
      return v + 1

    result = await Chain(5).gather(lambda v: v * 2, async_add).run()
    self.assertEqual(result, [10, 6])


class TestDecoratorEdgeCases(unittest.TestCase):
  """Chain.decorator() edge cases."""

  def test_decorator_basic(self):
    c = Chain().then(lambda v: v * 2)

    @c.decorator()
    def my_fn(x):
      return x + 1

    result = my_fn(4)
    self.assertEqual(result, 10)  # (4+1) * 2

  def test_decorator_preserves_name(self):
    c = Chain().then(lambda v: v)

    @c.decorator()
    def my_fn(x):
      return x

    self.assertEqual(my_fn.__name__, 'my_fn')

  def test_decorator_with_args(self):
    c = Chain().then(lambda v: v * 3)

    @c.decorator()
    def add(a, b):
      return a + b

    result = add(2, 3)
    self.assertEqual(result, 15)  # (2+3) * 3

  def test_decorator_with_kwargs(self):
    c = Chain().then(lambda v: v.upper())

    @c.decorator()
    def greet(name, greeting='hello'):
      return f'{greeting} {name}'

    result = greet('world')
    self.assertEqual(result, 'HELLO WORLD')


class TestIterateEdgeCases(unittest.TestCase):
  """Chain.iterate() edge cases."""

  def test_iterate_basic(self):
    items = list(Chain([1, 2, 3]).iterate())
    self.assertEqual(items, [1, 2, 3])

  def test_iterate_with_fn(self):
    items = list(Chain([1, 2, 3]).iterate(lambda v: v * 2))
    self.assertEqual(items, [2, 4, 6])

  def test_iterate_do_preserves_items(self):
    tracker = []
    items = list(Chain([1, 2, 3]).iterate_do(lambda v: tracker.append(v)))
    self.assertEqual(items, [1, 2, 3])
    self.assertEqual(tracker, [1, 2, 3])

  def test_iterate_empty(self):
    items = list(Chain([]).iterate())
    self.assertEqual(items, [])


class TestIterateAsync(unittest.IsolatedAsyncioTestCase):

  async def test_iterate_async(self):
    items = []
    async for item in Chain(AsyncRange(3)).iterate():
      items.append(item)
    self.assertEqual(items, [0, 1, 2])

  async def test_iterate_async_with_fn(self):
    items = []
    async for item in Chain([1, 2, 3]).iterate(lambda v: v * 2):
      items.append(item)
    self.assertEqual(items, [2, 4, 6])


class TestNullSentinelBehavior(unittest.TestCase):
  """Null sentinel behavior at boundaries."""

  def test_null_is_not_none(self):
    self.assertIsNot(Null, None)

  def test_null_repr(self):
    self.assertEqual(repr(Null), '<Null>')

  def test_chain_empty_returns_none_not_null(self):
    result = Chain().run()
    self.assertIsNone(result)
    self.assertIsNot(result, Null)

  def test_null_copy(self):
    import copy
    self.assertIs(copy.copy(Null), Null)

  def test_null_deepcopy(self):
    import copy
    self.assertIs(copy.deepcopy(Null), Null)

  def test_null_reduce(self):
    self.assertEqual(Null.__reduce__(), 'Null')

  def test_null_is_singleton(self):
    from quent._core import _Null
    # The module-level Null is a singleton
    self.assertIs(Null, Null)


class TestFinallyHandlerVariations(unittest.TestCase):
  """Various finally_ handler patterns."""

  def test_finally_receives_root_value(self):
    received = []
    Chain(42).finally_(lambda v: received.append(v)).run()
    self.assertEqual(received, [42])

  def test_finally_runs_on_exception(self):
    tracker = []

    def bad():
      raise ValueError('err')

    with self.assertRaises(ValueError):
      Chain(bad).finally_(lambda: tracker.append('finally'), ...).run()
    self.assertEqual(tracker, ['finally'])

  def test_finally_does_not_alter_result(self):
    result = Chain(42).finally_(lambda v: 'ignored_result').run()
    self.assertEqual(result, 42)

  def test_finally_runs_after_except(self):
    order = []
    result = (
      Chain(lambda: (_ for _ in ()).throw(ValueError('err')))
      .except_(lambda e: (order.append('except'), 'handled')[1])
      .finally_(lambda: order.append('finally'), ...)
      .run()
    )
    self.assertEqual(result, 'handled')
    self.assertEqual(order, ['except', 'finally'])


class TestExceptHandlerVariations(unittest.TestCase):
  """Various except_ handler patterns."""

  def test_except_with_multiple_exception_types(self):
    result = (
      Chain(lambda: (_ for _ in ()).throw(TypeError('err')))
      .except_(lambda e: 'handled', exceptions=[ValueError, TypeError])
      .run()
    )
    self.assertEqual(result, 'handled')

  def test_except_with_parent_exception_type(self):
    result = (
      Chain(lambda: (_ for _ in ()).throw(ValueError('err')))
      .except_(lambda e: 'handled', exceptions=Exception)
      .run()
    )
    self.assertEqual(result, 'handled')

  def test_except_handler_receives_exception_object(self):
    received = []

    def handler(exc):
      received.append(exc)
      return 'ok'

    Chain(lambda: (_ for _ in ()).throw(ValueError('msg'))).except_(handler).run()
    self.assertEqual(len(received), 1)
    self.assertIsInstance(received[0], ValueError)
    self.assertEqual(str(received[0]), 'msg')

  def test_except_with_empty_exceptions_raises(self):
    with self.assertRaises(QuentException):
      Chain().except_(lambda e: e, exceptions=[])

  def test_except_with_string_exceptions_raises(self):
    with self.assertRaises(TypeError):
      Chain().except_(lambda e: e, exceptions='ValueError')

  def test_except_with_non_exception_type_raises(self):
    with self.assertRaises(TypeError):
      Chain().except_(lambda e: e, exceptions=int)


if __name__ == '__main__':
  unittest.main()
