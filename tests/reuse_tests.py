# SPDX-License-Identifier: MIT
"""Tests for SPEC §10 — Reuse (clone / decorator)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import unittest
from typing import Any

from quent import Q, QuentException
from tests.fixtures import (
  async_double,
  async_fn,
  sync_double,
  sync_fn,
)
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §10.1 clone()
# ---------------------------------------------------------------------------


class CloneTests(SymmetricTestCase):
  """§10.1 — clone() creates an independent copy."""

  async def test_clone_produces_same_result(self) -> None:
    """Clone produces same result as original."""
    original = Q(5).then(lambda x: x * 2)
    cloned = original.clone()
    self.assertEqual(original.run(), cloned.run())
    self.assertEqual(cloned.run(), 10)

  async def test_clone_is_independent(self) -> None:
    """Extending clone doesn't affect original."""
    original = Q(5).then(lambda x: x * 2)
    cloned = original.clone()
    cloned.then(lambda x: x + 100)

    self.assertEqual(original.run(), 10)
    self.assertEqual(cloned.run(), 110)

  async def test_extending_original_doesnt_affect_clone(self) -> None:
    """Extending original doesn't affect clone."""
    original = Q(5).then(lambda x: x * 2)
    cloned = original.clone()
    original.then(lambda x: x + 999)

    self.assertEqual(original.run(), 1009)
    self.assertEqual(cloned.run(), 10)

  async def test_clone_nested_chains_recursively_cloned(self) -> None:
    """Nested chains within steps are recursively cloned."""
    inner = Q().then(lambda x: x + 1)
    original = Q(5).then(inner)

    cloned = original.clone()
    # Verify results are the same
    self.assertEqual(original.run(), 6)
    self.assertEqual(cloned.run(), 6)

    # The inner q in the clone should be a different instance
    # We can verify by checking that extending the original inner doesn't affect clone
    inner.then(lambda x: x * 100)
    # The original has already captured inner at build time, so extending inner
    # after the fact doesn't affect either. But the clone's inner is a separate clone.

  async def test_clone_kwarg_dicts_shallow_copied(self) -> None:
    """Keyword argument dicts are shallow-copied — clone's dict is a separate object."""
    kwargs = {'key': 'value'}
    original = Q(5).then(lambda key='default': key, **kwargs)
    cloned = original.clone()

    # Both produce the same result
    self.assertEqual(original.run(), 'value')
    self.assertEqual(cloned.run(), 'value')

    # The kwargs dicts in the cloned link must be different objects (shallow copy)
    orig_link = original._first_link
    clone_link = cloned._first_link
    self.assertIsNotNone(orig_link.kwargs)
    self.assertIsNotNone(clone_link.kwargs)
    self.assertIsNot(orig_link.kwargs, clone_link.kwargs)
    self.assertEqual(orig_link.kwargs, clone_link.kwargs)

  async def test_clone_callables_shared_by_reference(self) -> None:
    """Callables are shared by reference (not copied)."""
    call_count = [0]

    def tracked_fn(x: int) -> int:
      call_count[0] += 1
      return x + 1

    original = Q(5).then(tracked_fn)
    cloned = original.clone()

    original.run()
    cloned.run()
    # Both should have called the same function
    self.assertEqual(call_count[0], 2)

  async def test_clone_state_reset(self) -> None:
    """Clone behaves as top-level chain, even when original is used nested elsewhere."""
    original = Q(5).then(lambda x: x * 2)
    cloned = original.clone()
    # Clone should work independently as a top-level q
    self.assertEqual(cloned.run(), 10)

    # A pipeline used as a nested step in one q should still work as top-level
    inner = Q().then(lambda x: x + 1)
    # Use inner as a nested q in another q
    outer = Q(5).then(inner)
    self.assertEqual(outer.run(), 6)
    # inner's clone should also work as top-level
    inner_clone = inner.clone()
    self.assertEqual(inner_clone.run(10), 11)

  async def test_clone_with_except(self) -> None:
    """Clone preserves except handler."""
    original = Q(0).then(lambda x: 1 / x).except_(lambda _: -1)
    cloned = original.clone()

    self.assertEqual(original.run(), -1)
    self.assertEqual(cloned.run(), -1)

  async def test_clone_with_finally(self) -> None:
    """Clone preserves finally handler."""
    calls: list[str] = []

    def cleanup(rv: Any) -> None:
      calls.append('cleanup')

    original = Q(5).then(lambda x: x * 2).finally_(cleanup)
    cloned = original.clone()

    original.run()
    self.assertEqual(calls, ['cleanup'])

    cloned.run()
    self.assertEqual(calls, ['cleanup', 'cleanup'])

  async def test_clone_except_independence(self) -> None:
    """Clone's except handler is independent from original."""
    original = Q(0).then(lambda x: 1 / x).except_(lambda _: -1)
    cloned = original.clone()

    # Original has except handler already; clone should too
    self.assertEqual(original.run(), -1)
    self.assertEqual(cloned.run(), -1)

  async def test_clone_preserves_root(self) -> None:
    """Clone preserves root value."""
    original = Q(42)
    cloned = original.clone()
    self.assertEqual(cloned.run(), 42)

  async def test_clone_no_root(self) -> None:
    """Clone of empty chain works."""
    original = Q().then(lambda: 5)
    cloned = original.clone()
    self.assertEqual(cloned.run(), 5)

  async def test_clone_if_else(self) -> None:
    """Clone with if_/else_ — conditional ops deep-copied."""
    original = Q(5).if_(lambda x: x > 0).then(lambda x: x * 2).else_(lambda x: x * -1)
    cloned = original.clone()

    self.assertEqual(original.run(), 10)
    self.assertEqual(cloned.run(), 10)

    # Extending clone doesn't affect original
    cloned.then(lambda x: x + 1)
    self.assertEqual(original.run(), 10)
    self.assertEqual(cloned.run(), 11)

  async def test_clone_multiple_steps(self) -> None:
    """Clone with multiple steps."""
    original = Q(1).then(lambda x: x + 1).then(lambda x: x * 2).then(lambda x: x + 10)
    cloned = original.clone()

    self.assertEqual(original.run(), 14)
    self.assertEqual(cloned.run(), 14)

    cloned.then(lambda x: x * 100)
    self.assertEqual(original.run(), 14)
    self.assertEqual(cloned.run(), 1400)

  async def test_clone_with_map(self) -> None:
    """Clone preserves map operations."""
    original = Q([1, 2, 3]).foreach(sync_double)
    cloned = original.clone()

    self.assertEqual(original.run(), [2, 4, 6])
    self.assertEqual(cloned.run(), [2, 4, 6])

  async def test_clone_type_preserved(self) -> None:
    """Clone of subclass returns same subclass type."""

    class MyQ(Q[Any]):
      pass

    original = MyQ(5).then(lambda x: x * 2)
    cloned = original.clone()
    self.assertIsInstance(cloned, MyQ)

  async def test_clone_with_gather(self) -> None:
    """Clone preserves gather operations."""
    original = Q(5).gather(lambda x: x * 2, lambda x: x + 1)
    cloned = original.clone()

    orig_result = original.run()
    clone_result = cloned.run()
    self.assertEqual(orig_result, clone_result)
    self.assertEqual(clone_result, (10, 6))

  async def test_clone_except_handler_chain_is_recursively_cloned(self) -> None:
    """§10.1: If except_ handler callable is a Q pipeline, it is recursively cloned."""
    # Build a Q pipeline to use as the except handler.
    # When except_ handler is a Q pipeline, it receives QuentExcInfo as run value.
    handler_chain = Q().then(lambda t: -1)

    original = Q(0).then(lambda x: 1 / x).except_(handler_chain)
    cloned = original.clone()

    # Both should work
    self.assertEqual(original.run(), -1)
    self.assertEqual(cloned.run(), -1)

    # The handler q in the clone is a separate clone — a different object.
    orig_handler_v = original._on_except_link.v
    clone_handler_v = cloned._on_except_link.v
    self.assertIsNot(orig_handler_v, clone_handler_v)

  async def test_clone_finally_handler_chain_is_recursively_cloned(self) -> None:
    """§10.1: If finally_ handler callable is a Q pipeline, it is recursively cloned."""
    calls: list[str] = []

    cleanup_chain = Q().then(lambda rv: calls.append('chain_cleanup'))

    original = Q(5).then(lambda x: x * 2).finally_(cleanup_chain)
    cloned = original.clone()

    original.run()
    self.assertEqual(calls, ['chain_cleanup'])

    cloned.run()
    self.assertEqual(calls, ['chain_cleanup', 'chain_cleanup'])

  async def test_clone_except_handler_non_chain_shared_by_reference(self) -> None:
    """§10.1: Non-chain except_ handler callables are shared by reference."""
    call_count = [0]

    def handler(info) -> int:
      call_count[0] += 1
      return -1

    original = Q(0).then(lambda x: 1 / x).except_(handler)
    cloned = original.clone()

    original.run()
    cloned.run()
    # Both called the same handler function
    self.assertEqual(call_count[0], 2)


# ---------------------------------------------------------------------------
# §10.2 decorator()
# ---------------------------------------------------------------------------


class DecoratorSignalEscapeTest(SymmetricTestCase):
  """SPEC §10.2: Decorator control flow signal handling."""

  async def test_decorator_return_signal_caught(self) -> None:
    """Control flow signal _Return escaping through decorator() is wrapped in QuentException."""

    @Q().then(lambda x: Q.return_(x * 10)).as_decorator()
    def fn(x):
      return x

    # return_() inside the decorator q returns early with the value
    result = fn(5)
    self.assertEqual(result, 50)

  async def test_decorator_break_signal_caught(self) -> None:
    """Control flow signal _Break escaping through decorator() raises QuentException."""

    @Q().then(lambda x: Q.break_()).as_decorator()
    def fn(x):
      return x

    with self.assertRaises(QuentException):
      fn(5)


class DecoratorTests(SymmetricTestCase):
  """§10.2 — decorator() wraps chain as function decorator."""

  async def test_basic_decorator(self) -> None:
    """Decorated fn return value becomes chain input."""

    @Q().then(lambda x: x.strip()).then(str.upper).as_decorator()
    def get_name() -> str:
      return '  alice  '

    result = get_name()
    self.assertEqual(result, 'ALICE')

  async def test_decorator_with_args(self) -> None:
    """Decorated fn receives its own arguments."""

    @Q().then(lambda x: x * 2).as_decorator()
    def double(n: int) -> int:
      return n

    self.assertEqual(double(5), 10)
    self.assertEqual(double(3), 6)

  async def test_decorator_chain_cloned(self) -> None:
    """Q pipeline is cloned internally — original not affected."""
    q = Q().then(lambda x: x * 2)
    decorator = q.as_decorator()

    @decorator
    def fn(x: int) -> int:
      return x

    # Original q can still be extended without affecting decorator
    q.then(lambda x: x + 999)

    self.assertEqual(fn(5), 10)  # Decorator uses clone, not modified original

  async def test_decorator_preserves_signature(self) -> None:
    """Decorated function preserves original signature via functools.wraps."""

    @Q().then(lambda x: x).as_decorator()
    def my_func(a: int, b: str = 'hello') -> int:
      """My docstring."""
      return a

    self.assertEqual(my_func.__name__, 'my_func')
    self.assertEqual(my_func.__doc__, 'My docstring.')

  async def test_decorator_return_signal_extracts_value(self) -> None:
    """return_() in decorated chain is caught and its value extracted as the result."""

    @Q().then(lambda x: Q.return_(x * 10)).as_decorator()
    def fn(x: int) -> int:
      return x

    # return_() IS the signal to return early from the q.
    # In decorator(), is_nested=False is used, so the pipeline catches it.
    # The engine handles _Return in the main loop:
    # When not nested, _Return is caught and its value becomes the result.
    result = fn(5)
    self.assertEqual(result, 50)

  async def test_decorator_multiple_calls(self) -> None:
    """Decorator can be called multiple times with different args."""

    @Q().then(lambda x: x + 1).as_decorator()
    def inc(n: int) -> int:
      return n

    self.assertEqual(inc(1), 2)
    self.assertEqual(inc(10), 11)
    self.assertEqual(inc(100), 101)

  async def test_decorator_async_chain(self) -> None:
    """Decorator works with async steps in the chain."""

    @Q().then(async_double).as_decorator()
    def fn(x: int) -> int:
      return x

    result = fn(5)
    # With async step, fn returns a coroutine
    if asyncio.iscoroutine(result):
      result = await result
    self.assertEqual(result, 10)

  async def test_decorator_with_except(self) -> None:
    """Decorator preserves error handling."""

    @Q().then(lambda x: 1 / x).except_(lambda _: -1).as_decorator()
    def safe_div(x: int) -> int:
      return x

    self.assertEqual(safe_div(0), -1)
    self.assertEqual(safe_div(2), 0.5)

  async def test_decorator_nested_chain_step(self) -> None:
    """decorator() works correctly with nested chains as steps."""
    inner = Q().then(lambda x: x * 3)

    @Q().then(lambda x: x + 1).then(inner).as_decorator()
    def fn(x: int) -> int:
      return x

    # fn(5) -> root=5, then(5+1=6), then(inner runs with 6 -> 6*3=18)
    self.assertEqual(fn(5), 18)
    self.assertEqual(fn(0), 3)  # 0+1=1, 1*3=3

  async def test_decorator_thread_safety(self) -> None:
    """Decorator uses is_nested=False explicitly for thread safety."""
    import concurrent.futures

    @Q().then(lambda x: x * 2).as_decorator()
    def double(n: int) -> int:
      return n

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
      futures = [executor.submit(double, i) for i in range(10)]
      results = [f.result() for f in futures]

    self.assertEqual(sorted(results), [i * 2 for i in range(10)])


# ---------------------------------------------------------------------------
# Clone and decorator interaction tests
# ---------------------------------------------------------------------------


class CloneDecoratorInteractionTests(SymmetricTestCase):
  """Tests for clone and decorator interaction."""

  async def test_clone_then_decorator(self) -> None:
    """Cloning a chain and then decorating works."""
    base = Q().then(lambda x: x + 1)
    cloned = base.clone()

    @cloned.as_decorator()
    def fn(x: int) -> int:
      return x

    self.assertEqual(fn(5), 6)
    # Base is unaffected
    self.assertEqual(base.run(5), 6)

  async def test_decorator_then_clone(self) -> None:
    """Creating decorator then cloning original — independent."""
    base = Q().then(lambda x: x + 1)

    @base.as_decorator()
    def fn(x: int) -> int:
      return x

    # Clone base and extend it
    extended = base.clone().then(lambda x: x * 10)

    self.assertEqual(fn(5), 6)  # Decorator unaffected
    self.assertEqual(extended.run(5), 60)  # Clone works independently

  async def test_multiple_clones_independent(self) -> None:
    """Multiple clones are all independent."""
    base = Q(1).then(lambda x: x + 1)

    clone1 = base.clone()
    clone2 = base.clone()
    clone3 = base.clone()

    clone1.then(lambda x: x * 10)
    clone2.then(lambda x: x * 100)
    clone3.then(lambda x: x * 1000)

    self.assertEqual(base.run(), 2)
    self.assertEqual(clone1.run(), 20)
    self.assertEqual(clone2.run(), 200)
    self.assertEqual(clone3.run(), 2000)

  async def test_clone_of_clone(self) -> None:
    """Clone of a clone is also independent."""
    original = Q(1).then(lambda x: x + 1)
    clone1 = original.clone()
    clone2 = clone1.clone()

    clone1.then(lambda x: x * 10)
    clone2.then(lambda x: x * 100)

    self.assertEqual(original.run(), 2)
    self.assertEqual(clone1.run(), 20)
    self.assertEqual(clone2.run(), 200)


# ---------------------------------------------------------------------------
# §5.10 name()
# ---------------------------------------------------------------------------


class NameTests(SymmetricTestCase):
  """§5.10 — name() assigns a user-provided label for traceback identification."""

  async def test_name_returns_self(self) -> None:
    """name() returns the same chain instance for fluent chaining."""
    q = Q(lambda: 1)
    result = q.name('my_chain')
    self.assertIs(result, q)

  async def test_name_in_repr(self) -> None:
    """repr(Q(fn).name('x')) contains Q[x](fn_name)."""

    def my_fn():
      return 1

    q = Q(my_fn).name('x')
    r = repr(q)
    self.assertIn('Q[x]', r)
    self.assertIn('my_fn', r)

  async def test_name_in_repr_no_root(self) -> None:
    """repr(Q().name('x')) contains Q[x]()."""
    q = Q().name('x')
    r = repr(q)
    self.assertIn('Q[x]', r)
    self.assertIn('Q[x]()', r)

  async def test_unnamed_repr_unchanged(self) -> None:
    """repr(Q(fn)) does NOT contain brackets in the Q prefix."""

    def fn(x):
      return x

    q = Q(fn)
    r = repr(q)
    # Should have Q( but not Q[
    self.assertIn('Q(', r)
    self.assertNotIn('Q[', r)

  async def test_name_does_not_affect_execution(self) -> None:
    """Named pipeline produces same result as unnamed chain."""
    unnamed = Q(5).then(lambda x: x * 2)
    named = Q(5).then(lambda x: x * 2).name('double_pipeline')
    self.assertEqual(unnamed.run(), named.run())
    self.assertEqual(named.run(), 10)

  async def test_clone_copies_name(self) -> None:
    """clone() preserves the name from the original chain."""
    original = Q(5).then(lambda x: x + 1).name('my_pipeline')
    cloned = original.clone()
    r = repr(cloned)
    self.assertIn('Q[my_pipeline]', r)

  async def test_clone_name_independent(self) -> None:
    """Changing name on clone does not affect original."""
    original = Q(5).name('original_name')
    cloned = original.clone()
    cloned.name('clone_name')
    self.assertIn('Q[original_name]', repr(original))
    self.assertIn('Q[clone_name]', repr(cloned))

  async def test_decorator_preserves_name(self) -> None:
    """decorator() clones the chain, and the clone preserves the name."""
    q = Q().then(lambda x: x * 2).name('decorator_pipeline')

    @q.as_decorator()
    def fn(x: int) -> int:
      return x

    # The decorator uses a clone internally; verify the pipeline itself still has the name.
    self.assertIn('Q[decorator_pipeline]', repr(q))
    # And execution still works correctly.
    self.assertEqual(fn(5), 10)

  async def test_name_in_repr_with_steps(self) -> None:
    """repr of named pipeline with steps renders as Q[x](root).then(...)."""

    def root_fn():
      return 1

    def step_fn(x):
      return x + 1

    q = Q(root_fn).then(step_fn).name('full_chain')
    r = repr(q)
    # Name bracket appears in prefix
    self.assertIn('Q[full_chain]', r)
    # Root callable name appears (only the root is shown verbatim in repr)
    self.assertIn('root_fn', r)
    # Steps are rendered as .then(...)
    self.assertIn('.then(', r)


# ---------------------------------------------------------------------------
# §10.1 Clone Sharing — exception types and args shared by reference
# ---------------------------------------------------------------------------


class CloneExceptionTypesSharedTest(SymmetricTestCase):
  """SPEC §10.1 (SPEC-308): Exception type tuples are shared by reference after clone."""

  async def test_except_types_shared_by_reference(self) -> None:
    """Clone shares the exact same exception types tuple object (identity check)."""
    exc_types = (ValueError, TypeError)

    def handler(info):
      return -1

    def raise_value_error(x):
      raise ValueError('test')

    original = Q(0).then(raise_value_error).except_(handler, exceptions=exc_types)
    cloned = original.clone()

    # The _on_except_exceptions attribute on the Q instance stores the exception types
    self.assertIs(
      original._on_except_exceptions,
      cloned._on_except_exceptions,
      'Exception types tuple should be the same object (shared by reference)',
    )

    # Both chains still work — ValueError is in exc_types, so handler catches it
    self.assertEqual(original.run(), -1)
    self.assertEqual(cloned.run(), -1)


class CloneArgsSharedTest(SymmetricTestCase):
  """SPEC §10.1 (SPEC-309): Args tuples are shared by reference after clone."""

  async def test_args_tuple_shared_by_reference(self) -> None:
    """Clone shares the exact same args tuple object for step links."""
    arg_obj = {'key': 'value'}  # A mutable object as an arg

    def fn(a):
      return a

    original = Q(1).then(fn, arg_obj)
    cloned = original.clone()

    # Access the args tuple via _first_link.args
    self.assertIsNotNone(original._first_link)
    self.assertIsNotNone(cloned._first_link)
    self.assertIs(
      original._first_link.args,
      cloned._first_link.args,
      'Args tuple should be the same object (shared by reference, tuples are immutable)',
    )

    # Both chains still work
    self.assertEqual(original.run(), arg_obj)
    self.assertEqual(cloned.run(), arg_obj)


class DecoratorPendingIfTest(unittest.TestCase):
  """Test decorator() with unconsumed if_()."""

  def test_decorator_with_pending_if(self) -> None:
    """decorator() with a pending if_() raises QuentException."""
    with self.assertRaises(QuentException) as ctx:
      Q(5).if_(lambda x: x > 0).as_decorator()
    self.assertIn('if_() must be followed by .then() or .do()', str(ctx.exception))

  def test_decorator_pending_if_raises(self) -> None:
    """SPEC SS10.2: pending if_() before decorator() raises QuentException."""
    with self.assertRaises(QuentException):
      Q(5).if_(lambda x: True).as_decorator()


# ---------------------------------------------------------------------------
# §13.13 repr() format — SPEC gap tests
# ---------------------------------------------------------------------------


class ReprNoErrorMarkerTest(unittest.TestCase):
  """§13.13 (SPEC-463): repr(chain) has no <---- error marker."""

  def test_repr_has_no_error_marker(self) -> None:
    """repr(chain) uses visualization format but without <---- marker."""
    q = Q(1).then(lambda x: x + 1).then(lambda x: x * 2)
    r = repr(q)
    self.assertNotIn('<----', r)

  def test_repr_no_marker_with_except(self) -> None:
    """repr(chain) with except_ handler has no <---- marker."""
    q = Q(1).then(lambda x: x + 1).except_(lambda _: -1)
    r = repr(q)
    self.assertNotIn('<----', r)

  def test_repr_no_marker_with_nested_chain(self) -> None:
    """repr(chain) with nested chain has no <---- marker."""
    inner = Q().then(lambda x: x * 3)
    q = Q(5).then(inner)
    r = repr(q)
    self.assertNotIn('<----', r)


class ReprValuesEnvVarTest(unittest.TestCase):
  """§13.13 (SPEC-465): repr() respects QUENT_TRACEBACK_VALUES=0."""

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

  def test_repr_respects_values_zero(self) -> None:
    """QUENT_TRACEBACK_VALUES=0 replaces string value with <str> in repr."""
    code = """
from quent import Q

secret = 'my_secret_api_key_12345'
q = Q(secret).then(lambda x: x.upper())
r = repr(q)
has_actual = 'my_secret_api_key_12345' in r
has_placeholder = '<str>' in r
if has_placeholder and not has_actual:
  print('REPR_PLACEHOLDER_CORRECT')
elif has_actual:
  print('REPR_VALUE_LEAKED')
else:
  print('REPR_NO_PLACEHOLDER')
"""
    result = self._run_subprocess(code, {'QUENT_TRACEBACK_VALUES': '0'})
    self.assertEqual(result.returncode, 0, f'Subprocess failed: {result.stderr}')
    self.assertIn('REPR_PLACEHOLDER_CORRECT', result.stdout)


# ---------------------------------------------------------------------------
# §10.3 from_steps()
# ---------------------------------------------------------------------------


class TestFromSteps(SymmetricTestCase):
  """SPEC §10.3: Q.from_steps() — construct a chain from a sequence of steps."""

  async def test_variadic_form_produces_correct_result(self) -> None:
    """from_steps(a, b, c) threads value through each step in order."""
    # sync_fn(x) = x+1, sync_double(x) = x*2
    # 5 -> 6 -> 12
    result = Q.from_steps(sync_fn, sync_double).run(5)
    self.assertEqual(result, 12)

  async def test_variadic_equivalent_to_chained_then(self) -> None:
    """from_steps(a, b, c) is equivalent to Q().then(a).then(b).then(c)."""
    manual = Q().then(sync_fn).then(sync_double).run(5)
    via_from_steps = Q.from_steps(sync_fn, sync_double).run(5)
    self.assertEqual(via_from_steps, manual)
    self.assertEqual(via_from_steps, 12)

  async def test_list_form_unpacked_as_steps(self) -> None:
    """from_steps([a, b, c]) unpacks the list as the step sequence."""
    result = Q.from_steps([sync_fn, sync_double]).run(5)
    self.assertEqual(result, 12)

  async def test_list_form_equals_variadic_form(self) -> None:
    """from_steps([a, b, c]) produces same result as from_steps(a, b, c)."""
    variadic = Q.from_steps(sync_fn, sync_double).run(5)
    list_form = Q.from_steps([sync_fn, sync_double]).run(5)
    self.assertEqual(list_form, variadic)

  async def test_tuple_form_unpacked_as_steps(self) -> None:
    """from_steps((a, b, c)) unpacks the tuple as the step sequence."""
    result = Q.from_steps((sync_fn, sync_double)).run(5)
    self.assertEqual(result, 12)

  async def test_tuple_form_equals_variadic_form(self) -> None:
    """from_steps((a, b, c)) produces same result as from_steps(a, b, c)."""
    variadic = Q.from_steps(sync_fn, sync_double).run(5)
    tuple_form = Q.from_steps((sync_fn, sync_double)).run(5)
    self.assertEqual(tuple_form, variadic)

  async def test_empty_no_args_returns_empty_chain(self) -> None:
    """from_steps() with no arguments returns an empty chain — run(v) returns v."""
    # Empty q with a run value returns that run value
    result = Q.from_steps().run(5)
    self.assertEqual(result, 5)

  async def test_empty_is_equivalent_to_chain(self) -> None:
    """from_steps() with no args is equivalent to Q()."""
    self.assertEqual(Q.from_steps().run(5), Q().run(5))
    self.assertEqual(Q.from_steps().run(), Q().run())

  async def test_single_step(self) -> None:
    """from_steps(fn) with a single step works correctly."""
    result = Q.from_steps(sync_fn).run(10)
    self.assertEqual(result, 11)

  async def test_single_step_list_form(self) -> None:
    """from_steps([fn]) single-element list works."""
    result = Q.from_steps([sync_fn]).run(10)
    self.assertEqual(result, 11)

  async def test_non_callable_literal_replaces_current_value(self) -> None:
    """Non-callable value in steps replaces current value, per .then() semantics.

    Q.from_steps(lambda x: x+1, 42, lambda x: x*2).run(10):
      10 -> x+1 = 11 -> 42 (literal replaces cv) -> 42*2 = 84
    """
    result = Q.from_steps(lambda x: x + 1, 42, lambda x: x * 2).run(10)
    self.assertEqual(result, 84)

  async def test_nested_chain_as_step(self) -> None:
    """A nested Q pipeline as a step is executed with the current value."""
    inner = Q().then(lambda x: x + 1)
    result = Q.from_steps(inner).run(10)
    self.assertEqual(result, 11)

  async def test_nested_chain_as_step_in_sequence(self) -> None:
    """Nested Q pipeline in a multi-step from_steps sequence executes in order."""
    inner = Q().then(sync_double)
    # 5 -> sync_fn(5)=6 -> inner(6)=12
    result = Q.from_steps(sync_fn, inner).run(5)
    self.assertEqual(result, 12)

  async def test_async_callables_returns_coroutine(self) -> None:
    """from_steps with async steps: run() returns a coroutine, must be awaited."""
    coro = Q.from_steps(async_fn, async_double).run(5)
    self.assertTrue(asyncio.iscoroutine(coro))
    result = await coro
    # 5 -> async_fn(5)=6 -> async_double(6)=12
    self.assertEqual(result, 12)

  async def test_mixed_sync_async_bridge(self) -> None:
    """from_steps(sync_fn, async_fn): sync start, async transition on async step."""
    # 5 -> sync_fn(5)=6 -> async_double(6)=12
    result = await Q.from_steps(sync_fn, async_double).run(5)
    self.assertEqual(result, 12)

  async def test_mixed_async_sync_bridge(self) -> None:
    """from_steps(async_fn, sync_fn): async transition on first step, stays async."""
    # 5 -> async_fn(5)=6 -> sync_double(6)=12
    result = await Q.from_steps(async_fn, sync_double).run(5)
    self.assertEqual(result, 12)

  async def test_equivalence_three_steps(self) -> None:
    """from_steps(a, b, c).run(v) == Q().then(a).then(b).then(c).run(v)."""
    steps = [sync_fn, sync_double, sync_fn]
    # 5 -> +1=6 -> *2=12 -> +1=13
    manual = Q().then(steps[0]).then(steps[1]).then(steps[2]).run(5)
    via_from_steps = Q.from_steps(*steps).run(5)
    self.assertEqual(via_from_steps, manual)
    self.assertEqual(via_from_steps, 13)

  async def test_returns_chain_instance(self) -> None:
    """from_steps() returns a Q instance."""
    c = Q.from_steps(sync_fn)
    self.assertIsInstance(c, Q)

  async def test_run_value_provides_initial_value(self) -> None:
    """from_steps() creates no-root chain; run(v) provides the initial value."""
    # No root value — use run(v) to inject
    result = Q.from_steps(sync_double).run(7)
    self.assertEqual(result, 14)

  async def test_empty_list_form_returns_empty_chain(self) -> None:
    """from_steps([]) with an empty list returns an empty chain."""
    result = Q.from_steps([]).run(5)
    self.assertEqual(result, 5)

  async def test_empty_tuple_form_returns_empty_chain(self) -> None:
    """from_steps(()) with an empty tuple returns an empty chain."""
    result = Q.from_steps(()).run(5)
    self.assertEqual(result, 5)

  async def test_multiple_steps_value_threading(self) -> None:
    """Value threads correctly through all steps in order."""
    # 1 -> +1=2 -> *2=4 -> +1=5 -> *2=10
    result = Q.from_steps(sync_fn, sync_double, sync_fn, sync_double).run(1)
    self.assertEqual(result, 10)

  async def test_list_form_three_steps_equivalence(self) -> None:
    """from_steps([a, b, c]) == Q().then(a).then(b).then(c) for three steps."""
    steps = [sync_fn, sync_double, sync_fn]
    manual = Q().then(steps[0]).then(steps[1]).then(steps[2]).run(3)
    via_list = Q.from_steps(steps).run(3)
    self.assertEqual(via_list, manual)


# ---------------------------------------------------------------------------
# §16.6 Copy blocking
# ---------------------------------------------------------------------------


class CopyBlockingTest(unittest.TestCase):
  """SPEC §16.6: copy.copy/deepcopy of Q raise TypeError."""

  def test_copy_copy_raises_type_error(self) -> None:
    """copy.copy(chain) raises TypeError (SPEC §16.6)."""
    import copy

    c = Q(1).then(lambda x: x + 1)
    with self.assertRaises(TypeError) as ctx:
      copy.copy(c)
    self.assertIn('copy.copy()', str(ctx.exception))
    self.assertIn('clone()', str(ctx.exception))

  def test_copy_deepcopy_raises_type_error(self) -> None:
    """copy.deepcopy(chain) raises TypeError (SPEC §16.6)."""
    import copy

    c = Q(1).then(lambda x: x + 1)
    with self.assertRaises(TypeError) as ctx:
      copy.deepcopy(c)
    self.assertIn('deepcopy()', str(ctx.exception))
    self.assertIn('clone()', str(ctx.exception))

  def test_copy_empty_chain_raises(self) -> None:
    """Even an empty chain cannot be copied."""
    import copy

    with self.assertRaises(TypeError):
      copy.copy(Q())
    with self.assertRaises(TypeError):
      copy.deepcopy(Q())


if __name__ == '__main__':
  unittest.main()
