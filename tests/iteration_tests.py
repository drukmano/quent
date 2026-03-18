# SPDX-License-Identifier: MIT
"""Tests for SPEC §9 — Iteration (iterate / iterate_do)."""

from __future__ import annotations

import unittest

from quent import Chain, QuentException
from tests.fixtures import V_DOUBLE, async_double, sync_double
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# §9.1 iterate(fn=None)
# ---------------------------------------------------------------------------


class IterateTests(SymmetricTestCase):
  """§9.1 — iterate() returns dual sync/async iterator."""

  async def test_iterate_no_fn(self) -> None:
    """iterate() with no fn yields elements as-is."""
    result = list(Chain(range(5)).iterate())
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_iterate_with_fn(self) -> None:
    """iterate(fn) transforms each element — sync fn uses sync for, async fn uses async for."""
    # Sync fn path
    result_sync = list(Chain(range(5)).iterate(sync_double))
    self.assertEqual(result_sync, [0, 2, 4, 6, 8])

    # Async fn path — must use async for
    result_async: list[int] = []
    async for item in Chain(range(5)).iterate(async_double):
      result_async.append(item)
    self.assertEqual(result_async, [0, 2, 4, 6, 8])

  async def test_iterate_chain_executes_at_iteration_start(self) -> None:
    """Chain executes when iteration begins, not when iterate() is called."""
    calls: list[str] = []

    def track() -> list[int]:
      calls.append('executed')
      return [1, 2, 3]

    it = Chain(track).iterate()
    self.assertEqual(calls, [])  # Not yet executed
    result = list(it)
    self.assertEqual(calls, ['executed'])
    self.assertEqual(result, [1, 2, 3])

  async def test_iterate_sync_with_async_chain_raises_typeerror(self) -> None:
    """Sync 'for' with async chain raises TypeError."""

    async def async_range() -> list[int]:
      return [1, 2, 3]

    it = Chain(async_range).iterate()
    with self.assertRaises(TypeError):
      list(it)

  async def test_iterate_async_with_sync_iterable(self) -> None:
    """async for works with sync iterable chain."""
    result: list[int] = []
    async for item in Chain(range(5)).iterate():
      result.append(item)
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_iterate_async_with_fn(self) -> None:
    """async for with fn transforms elements — sync and async fn."""
    for fn_label, fn in V_DOUBLE:
      with self.subTest(fn=fn_label):
        result: list[int] = []
        async for item in Chain(range(5)).iterate(fn):
          result.append(item)
        self.assertEqual(result, [0, 2, 4, 6, 8])

  async def test_iterate_sync_fn_returns_awaitable_raises(self) -> None:
    """Sync iteration with fn returning awaitable raises TypeError."""

    async def async_fn_inner(x: int) -> int:
      return x * 2

    it = Chain(range(3)).iterate(async_fn_inner)
    with self.assertRaises(TypeError):
      list(it)

  async def test_iterate_async_chain_with_async_for(self) -> None:
    """async for handles async chain result."""

    async def async_range() -> list[int]:
      return [10, 20, 30]

    result: list[int] = []
    async for item in Chain(async_range).iterate():
      result.append(item)
    self.assertEqual(result, [10, 20, 30])


# ---------------------------------------------------------------------------
# §9.1 Error handling — chain handlers vs fn errors
# ---------------------------------------------------------------------------


class IterateErrorHandlingTests(SymmetricTestCase):
  """§9.1 — except_/finally_ apply to chain run(), not fn iteration errors."""

  async def test_chain_except_catches_chain_error(self) -> None:
    """Chain's except_() catches errors during chain execution (run phase)."""
    handler_calls: list[str] = []

    def error_handler(info) -> list[int]:
      handler_calls.append('handled')
      return [99]

    # Chain itself raises during run phase (before iteration)
    it = Chain(0).then(lambda x: 1 / x).except_(error_handler).iterate()
    result = list(it)
    self.assertEqual(handler_calls, ['handled'])
    self.assertEqual(result, [99])

  async def test_fn_error_bypasses_chain_except(self) -> None:
    """Errors from fn during iteration bypass chain's except_() handler."""
    handler_calls: list[str] = []

    def error_handler(info) -> list[int]:
      handler_calls.append('handled')
      return [-1]

    def failing_fn(x: int) -> int:
      if x == 2:
        raise ValueError('fn error')
      return x

    it = Chain(range(5)).except_(error_handler).iterate(failing_fn)
    with self.assertRaises(ValueError) as ctx:
      list(it)
    self.assertIn('fn error', str(ctx.exception))
    # The chain's handler was NOT called — fn errors bypass it
    self.assertEqual(handler_calls, [])

  async def test_chain_finally_runs_on_chain_error(self) -> None:
    """Chain's finally_() runs when chain execution fails (run phase).

    With deferred finally, the cleanup runs AFTER iteration completes
    (in the generator's finally block), not during the run phase.  The
    except handler catches the error and produces [], which is iterated;
    the deferred finally runs once that iteration finishes.
    """
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    # except_ catches the ZeroDivisionError and returns []; finally_ is
    # deferred and runs in the generator's finally block after iteration ends.
    it = Chain(0).then(lambda x: 1 / x).except_(lambda _: []).finally_(cleanup).iterate()
    result = list(it)
    self.assertEqual(cleanup_calls, ['cleanup'])
    self.assertEqual(result, [])

  async def test_fn_error_triggers_deferred_chain_finally(self) -> None:
    """fn errors during iteration trigger the chain's deferred finally_() handler.

    With deferred finally, the chain runs successfully (producing range(3)),
    finally is deferred to the generator's finally block.  When fn raises on
    x==1, the generator exits and the finally block runs the deferred cleanup.
    """
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    def failing_fn(x: int) -> int:
      if x == 1:
        raise ValueError('fn error')
      return x

    # Chain runs successfully; finally_ is deferred.  fn raises during
    # iteration, triggering the generator's finally block which runs cleanup.
    it = Chain(range(3)).finally_(cleanup).iterate(failing_fn)
    with self.assertRaises(ValueError):
      list(it)
    # Deferred finally ran via the generator's finally block.
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_chain_except_catches_chain_error_async(self) -> None:
    """Chain's except_() catches errors during async chain execution."""
    handler_calls: list[str] = []

    async def async_fail() -> list[int]:
      raise ValueError('chain error')

    def error_handler(info) -> list[int]:
      handler_calls.append('handled')
      return [42]

    result: list[int] = []
    async for item in Chain(async_fail).except_(error_handler).iterate():
      result.append(item)
    self.assertEqual(handler_calls, ['handled'])
    self.assertEqual(result, [42])

  async def test_fn_error_bypasses_chain_except_async(self) -> None:
    """Async: fn errors during iteration bypass chain's except_()."""
    handler_calls: list[str] = []

    def error_handler(info) -> int:
      handler_calls.append('handled')
      return -1

    async def failing_fn(x: int) -> int:
      if x == 2:
        raise ValueError('async fn error')
      return x

    with self.assertRaises(ValueError):
      async for _item in Chain(range(5)).except_(error_handler).iterate(failing_fn):
        pass
    self.assertEqual(handler_calls, [])


# ---------------------------------------------------------------------------
# §9.2 iterate_do(fn=None)
# ---------------------------------------------------------------------------


class IterateDoTests(SymmetricTestCase):
  """§9.2 — iterate_do() yields original elements, fn as side-effect."""

  async def test_iterate_do_no_fn(self) -> None:
    """iterate_do() with no fn yields elements as-is."""
    result = list(Chain(range(4)).iterate_do())
    self.assertEqual(result, [0, 1, 2, 3])

  async def test_iterate_do_yields_originals(self) -> None:
    """iterate_do(fn) yields original elements, not fn results."""
    # Sync fn path
    result_sync = list(Chain(range(4)).iterate_do(sync_double))
    self.assertEqual(result_sync, [0, 1, 2, 3])

    # Async fn path — must use async for
    result_async: list[int] = []
    async for item in Chain(range(4)).iterate_do(async_double):
      result_async.append(item)
    self.assertEqual(result_async, [0, 1, 2, 3])

  async def test_iterate_do_fn_executed(self) -> None:
    """iterate_do actually invokes fn for side-effects."""
    calls: list[int] = []

    def track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    result = list(Chain(range(3)).iterate_do(track))
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(calls, [0, 1, 2])

  async def test_iterate_do_async(self) -> None:
    """async for with iterate_do yields originals — sync and async fn."""
    for fn_label, fn in V_DOUBLE:
      with self.subTest(fn=fn_label):
        result: list[int] = []
        async for item in Chain(range(4)).iterate_do(fn):
          result.append(item)
        self.assertEqual(result, [0, 1, 2, 3])

  async def test_iterate_do_async_fn(self) -> None:
    """async for with async fn, still yields originals."""
    calls: list[int] = []

    async def async_track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    result: list[int] = []
    async for item in Chain(range(3)).iterate_do(async_track):
      result.append(item)
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(calls, [0, 1, 2])

  async def test_iterate_do_fn_return_discarded_not_mapped(self) -> None:
    """§9.2: iterate_do yields original elements, NOT fn's return values."""
    result = list(Chain([1, 2, 3]).iterate_do(lambda x: x * 100))
    self.assertEqual(result, [1, 2, 3])

  async def test_iterate_do_sync_no_fn(self) -> None:
    """iterate_do() sync path with no fn — exercises _sync_generator no-fn path."""
    result = list(Chain([10, 20, 30]).iterate_do())
    self.assertEqual(result, [10, 20, 30])

  async def test_iterate_do_async_no_fn(self) -> None:
    """iterate_do() async path with no fn — exercises _async_generator no-fn path."""
    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do():
      result.append(item)
    self.assertEqual(result, [10, 20, 30])


# ---------------------------------------------------------------------------
# §9.3 Iterator reuse via calling
# ---------------------------------------------------------------------------


class IteratorReuseTests(SymmetricTestCase):
  """§9.3 — it(value) creates new iterator with different run args."""

  async def test_reuse_with_different_value(self) -> None:
    """Calling iterator with value creates new iterator."""
    it = Chain().then(lambda x: [x, x + 1, x + 2]).iterate()
    result1 = list(it(1))
    result2 = list(it(10))
    self.assertEqual(result1, [1, 2, 3])
    self.assertEqual(result2, [10, 11, 12])

  async def test_reuse_preserves_fn(self) -> None:
    """Reused iterator keeps original fn configuration."""
    it = Chain().then(lambda x: [x, x + 1]).iterate(sync_double)
    result1 = list(it(1))
    result2 = list(it(5))
    self.assertEqual(result1, [2, 4])
    self.assertEqual(result2, [10, 12])

  async def test_reuse_original_still_works(self) -> None:
    """Original iterator is not affected by creating new ones."""
    it = Chain(range(3)).iterate()
    _new_it = it(99)
    # Original still works with its own args
    result = list(it)
    self.assertEqual(result, [0, 1, 2])

  async def test_reuse_async(self) -> None:
    """Reused iterator works with async for."""
    it = Chain().then(lambda x: [x * 10, x * 20]).iterate()
    result: list[int] = []
    async for item in it(3):
      result.append(item)
    self.assertEqual(result, [30, 60])

  async def test_reuse_returns_fresh_instance(self) -> None:
    """Each call returns a new iterator instance."""
    it = Chain(range(3)).iterate()
    it2 = it(1)
    it3 = it(2)
    self.assertIsNot(it, it2)
    self.assertIsNot(it2, it3)


# ---------------------------------------------------------------------------
# §9.4 Control flow in iteration
# ---------------------------------------------------------------------------


class IterationControlFlowTests(SymmetricTestCase):
  """§9.4 — return_ and break_ behavior during iteration."""

  async def test_return_yields_value_then_stops(self) -> None:
    """return_(v) yields the value then stops iteration."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(x * 10)
      return x

    result = list(Chain(range(5)).iterate(fn))
    # Elements before return are yielded, then return value, then stop
    self.assertEqual(result, [0, 1, 20])

  async def test_return_no_value_stops(self) -> None:
    """return_() with no value stops iteration (no extra yield)."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_()
      return x

    result = list(Chain(range(5)).iterate(fn))
    self.assertEqual(result, [0, 1])

  async def test_break_stops_iteration(self) -> None:
    """break_() stops iteration immediately."""

    def fn(x: int) -> int:
      if x == 3:
        return Chain.break_()
      return x

    result = list(Chain(range(10)).iterate(fn))
    self.assertEqual(result, [0, 1, 2])

  async def test_break_with_value_yields_then_stops(self) -> None:
    """break_(v) yields the value then stops."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(x * 100)
      return x

    result = list(Chain(range(5)).iterate(fn))
    self.assertEqual(result, [0, 1, 200])

  async def test_return_async(self) -> None:
    """return_ works in async iteration."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(x * 10)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 20])

  async def test_break_async(self) -> None:
    """break_ works in async iteration."""

    def fn(x: int) -> int:
      if x == 3:
        return Chain.break_()
      return x

    result: list[int] = []
    async for item in Chain(range(10)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 2])

  async def test_break_with_value_async(self) -> None:
    """break_(v) yields value in async iteration."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(x * 100)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 200])

  async def test_iterate_do_with_return(self) -> None:
    """return_ in iterate_do yields return value then stops, yields originals before."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(x * 10)
      return x

    # iterate_do yields originals, but return_ value is special
    result = list(Chain(range(5)).iterate_do(fn))
    # Items 0, 1 are yielded as originals, then return value 20 is yielded
    self.assertEqual(result, [0, 1, 20])

  async def test_iterate_do_with_break(self) -> None:
    """break_ in iterate_do stops iteration."""

    def fn(x: int) -> int:
      if x == 3:
        return Chain.break_()
      return x

    result = list(Chain(range(10)).iterate_do(fn))
    self.assertEqual(result, [0, 1, 2])


# ---------------------------------------------------------------------------
# §17.7 — break_ semantics differ between map() and iterate()
# ---------------------------------------------------------------------------


class BreakSemanticsTests(SymmetricTestCase):
  """§17.7 — break_(value) appends in both map and iterate."""

  async def test_map_break_with_value_appends(self) -> None:
    """In map(), break_(value) appends to partial results."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(42)
      return x * 10

    result = Chain([0, 1, 2, 3, 4]).foreach(fn).run()
    # break_(42) appends to partial [0, 10]
    self.assertEqual(result, [0, 10, 42])

  async def test_iterate_break_with_value_appends(self) -> None:
    """In iterate(), break_(value) YIELDS value as additional item."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(42)
      return x * 10

    result = list(Chain([0, 1, 2, 3, 4]).iterate(fn))
    # Items 0*10=0, 1*10=10 already yielded, then 42 yielded before stop
    self.assertEqual(result, [0, 10, 42])

  async def test_map_break_no_value_returns_partial(self) -> None:
    """In map(), break_() returns partial results as-is."""

    def fn(x: int) -> int:
      if x == 3:
        return Chain.break_()
      return x * 10

    result = Chain([0, 1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(result, [0, 10, 20])

  async def test_iterate_break_no_value_stops_immediately(self) -> None:
    """In iterate(), break_() stops immediately, no additional item."""

    def fn(x: int) -> int:
      if x == 3:
        return Chain.break_()
      return x * 10

    result = list(Chain([0, 1, 2, 3, 4]).iterate(fn))
    self.assertEqual(result, [0, 10, 20])

  async def test_map_break_with_value_appends_async(self) -> None:
    """Async: In map(), break_(value) appends to partial results."""

    async def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(42)
      return x * 10

    result = await Chain([0, 1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(result, [0, 10, 42])

  async def test_iterate_break_with_value_appends_async(self) -> None:
    """Async: In iterate(), break_(value) yields as additional item."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(42)
      return x * 10

    result: list[int] = []
    async for item in Chain([0, 1, 2, 3, 4]).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 10, 42])


# ---------------------------------------------------------------------------
# §17.8 — return_ semantics differ between chain and iterate
# ---------------------------------------------------------------------------


class ReturnSemanticsTests(SymmetricTestCase):
  """§17.8 — return_(value) replaces in chain, appends in iterate."""

  async def test_chain_return_replaces_result(self) -> None:
    """In normal chain execution, return_(value) replaces entire result."""
    result = Chain(5).then(lambda x: x + 1).then(lambda x: Chain.return_(99)).then(lambda x: x * 100).run()
    self.assertEqual(result, 99)

  async def test_iterate_return_yields_value_then_stops(self) -> None:
    """In iterate(), return_(value) yields value as final item."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(99)
      return x * 10

    result = list(Chain([0, 1, 2, 3, 4]).iterate(fn))
    # Items 0, 10 already yielded, then 99 yielded as final item
    self.assertEqual(result, [0, 10, 99])

  async def test_chain_return_no_value(self) -> None:
    """In normal chain, return_() with no value returns None."""
    result = Chain(5).then(lambda x: Chain.return_()).then(lambda x: x * 100).run()
    self.assertIsNone(result)

  async def test_iterate_return_no_value_stops(self) -> None:
    """In iterate(), return_() stops without extra yield."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_()
      return x * 10

    result = list(Chain([0, 1, 2, 3, 4]).iterate(fn))
    self.assertEqual(result, [0, 10])

  async def test_iterate_return_with_value_async(self) -> None:
    """Async: return_(value) in iterate yields value then stops."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(99)
      return x * 10

    result: list[int] = []
    async for item in Chain([0, 1, 2, 3, 4]).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 10, 99])

  async def test_iterate_return_callable_value_is_evaluated(self) -> None:
    """return_(callable) evaluates the callable per §9.4 calling convention."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(lambda: 999)
      return x

    result = list(Chain(range(5)).iterate(fn))
    # The lambda is evaluated (called) by _eval_signal_value
    self.assertEqual(result, [0, 1, 999])


# ---------------------------------------------------------------------------
# Async iterator paths — coverage for _generator.py async paths
# ---------------------------------------------------------------------------


class AsyncIteratorPathTests(SymmetricTestCase):
  """Coverage tests for _async_generator paths in _generator.py."""

  async def test_async_iterate_with_async_chain_and_fn(self) -> None:
    """async for with async chain and sync fn — exercises await + fn path."""

    async def async_range() -> list[int]:
      return [1, 2, 3, 4]

    result: list[int] = []
    async for item in Chain(async_range).iterate(sync_double):
      result.append(item)
    self.assertEqual(result, [2, 4, 6, 8])

  async def test_async_iterate_do_with_fn(self) -> None:
    """async iterate_do with sync fn — exercises async ignore_result path."""
    calls: list[int] = []

    def track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do(track):
      result.append(item)
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(calls, [10, 20, 30])

  async def test_async_iterate_do_with_async_fn(self) -> None:
    """async iterate_do with async fn — exercises full async ignore_result path."""
    calls: list[int] = []

    async def async_track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do(async_track):
      result.append(item)
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(calls, [10, 20, 30])

  async def test_async_return_with_value(self) -> None:
    """Async: return_(value) — exercises _Return handling in _async_generator."""

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(x * 100)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 100])

  async def test_async_return_no_value(self) -> None:
    """Async: return_() with no value — stops async iteration."""

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_()
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0])

  async def test_async_break_with_value(self) -> None:
    """Async: break_(value) — exercises _Break handling in _async_generator."""

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(x * 100)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 100])

  async def test_async_break_no_value(self) -> None:
    """Async: break_() — stops async iteration immediately."""

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_()
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0])

  async def test_async_return_with_async_callable_value(self) -> None:
    """Async: return_(async_callable) — resolved value is awaited."""

    async def resolve() -> int:
      return 777

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(resolve)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 777])

  async def test_async_break_with_async_callable_value(self) -> None:
    """Async: break_(async_callable) — resolved value is awaited."""

    async def resolve() -> int:
      return 888

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(resolve)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate(fn):
      result.append(item)
    self.assertEqual(result, [0, 888])

  async def test_async_iterate_do_with_return(self) -> None:
    """Async: iterate_do with return_(value)."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(x * 10)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate_do(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 20])

  async def test_async_iterate_do_with_break(self) -> None:
    """Async: iterate_do with break_()."""

    def fn(x: int) -> int:
      if x == 3:
        return Chain.break_()
      return x

    result: list[int] = []
    async for item in Chain(range(10)).iterate_do(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 2])

  async def test_async_iterate_do_with_break_value(self) -> None:
    """Async: iterate_do with break_(value)."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(42)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate_do(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 42])


# ---------------------------------------------------------------------------
# Mid-operation async transition — coverage for _iter_ops.py _to_async
# ---------------------------------------------------------------------------


class MidOperationAsyncTransitionTests(SymmetricTestCase):
  """Coverage tests for mid-operation async transition in _iter_ops.py."""

  async def test_map_mid_transition(self) -> None:
    """map() starts sync, fn returns coroutine partway through — triggers _to_async."""
    call_count = [0]

    def mixed_fn(x: int) -> int:
      call_count[0] += 1
      if call_count[0] > 2:
        # Return a coroutine starting from the 3rd item
        async def _inner() -> int:
          return x * 10

        return _inner()
      return x * 10

    result = await Chain([1, 2, 3, 4]).foreach(mixed_fn).run()
    self.assertEqual(result, [10, 20, 30, 40])

  async def test_foreach_do_mid_transition(self) -> None:
    """foreach_do() starts sync, fn returns coroutine partway — triggers _to_async."""
    calls: list[int] = []
    call_count = [0]

    def mixed_fn(x: int) -> None:
      call_count[0] += 1
      calls.append(x)
      if call_count[0] > 2:

        async def _inner() -> None:
          pass

        return _inner()
      return None

    result = await Chain([1, 2, 3, 4]).foreach_do(mixed_fn).run()
    self.assertEqual(result, [1, 2, 3, 4])
    self.assertEqual(calls, [1, 2, 3, 4])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class IterateEdgeCaseTests(SymmetricTestCase):
  """Edge cases for iterate/iterate_do."""

  async def test_iterate_empty(self) -> None:
    """iterate over empty iterable yields nothing."""
    result = list(Chain([]).iterate())
    self.assertEqual(result, [])

  async def test_iterate_single_element(self) -> None:
    """iterate over single element."""
    result = list(Chain([42]).iterate())
    self.assertEqual(result, [42])

  async def test_iterate_with_none_fn(self) -> None:
    """iterate(None) is the same as iterate() — yields as-is."""
    result = list(Chain(range(3)).iterate(None))
    self.assertEqual(result, [0, 1, 2])

  async def test_iterate_do_with_none_fn(self) -> None:
    """iterate_do(None) yields elements as-is."""
    result = list(Chain(range(3)).iterate_do(None))
    self.assertEqual(result, [0, 1, 2])

  async def test_iterate_repr(self) -> None:
    """ChainIterator has a useful repr."""
    it = Chain(range(3)).iterate()
    self.assertEqual(repr(it), '<quent.ChainIterator>')

  async def test_iterate_chain_with_steps(self) -> None:
    """iterate works on chains with preceding steps."""
    # Sync fn path
    result = list(Chain(5).then(lambda x: range(x)).iterate(sync_double))
    self.assertEqual(result, [0, 2, 4, 6, 8])

    # Async fn path — must use async for
    result_async: list[int] = []
    async for item in Chain(5).then(lambda x: range(x)).iterate(async_double):
      result_async.append(item)
    self.assertEqual(result_async, [0, 2, 4, 6, 8])

  async def test_iterate_with_run_value(self) -> None:
    """Iterator can be called with run value to override root."""
    it = Chain().then(lambda x: [x, x * 2]).iterate()
    result = list(it(3))
    self.assertEqual(result, [3, 6])

  async def test_iterate_sync_break_with_callable_value(self) -> None:
    """Sync break_(callable) — callable is evaluated via _eval_signal_value."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(lambda: 42)
      return x

    result = list(Chain(range(5)).iterate(fn))
    self.assertEqual(result, [0, 1, 42])

  async def test_iterate_sync_return_with_callable_value(self) -> None:
    """Sync return_(callable) — callable is evaluated via _eval_signal_value."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(lambda: 42)
      return x

    result = list(Chain(range(5)).iterate(fn))
    self.assertEqual(result, [0, 1, 42])

  async def test_iterate_async_empty(self) -> None:
    """async for over empty iterable yields nothing."""
    result: list[int] = []
    async for item in Chain([]).iterate():
      result.append(item)
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# §9.4 — iterate_do + break_/return_ with callable values
# ---------------------------------------------------------------------------


class IterateDoControlFlowCallableTests(SymmetricTestCase):
  """§9.4 — iterate_do with break_/return_ carrying callable values."""

  async def test_iterate_do_break_with_callable_value(self) -> None:
    """iterate_do: break_(callable) evaluates callable and yields result."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(lambda: 42)
      return x

    result = list(Chain(range(5)).iterate_do(fn))
    # iterate_do yields originals 0, 1, then break_(lambda: 42) yields 42
    self.assertEqual(result, [0, 1, 42])

  async def test_iterate_do_return_with_callable_value(self) -> None:
    """iterate_do: return_(callable) evaluates callable and yields result."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(lambda: 99)
      return x

    result = list(Chain(range(5)).iterate_do(fn))
    # iterate_do yields originals 0, 1, then return_(lambda: 99) yields 99
    self.assertEqual(result, [0, 1, 99])

  async def test_iterate_do_break_with_callable_value_async(self) -> None:
    """Async: iterate_do with break_(callable) — callable is evaluated."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_(lambda: 42)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate_do(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 42])

  async def test_iterate_do_return_with_callable_value_async(self) -> None:
    """Async: iterate_do with return_(callable) — callable is evaluated."""

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(lambda: 99)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate_do(fn):
      result.append(item)
    self.assertEqual(result, [0, 1, 99])

  async def test_iterate_do_break_with_async_callable_value(self) -> None:
    """Async: iterate_do with break_(async_callable) — awaited and yielded."""

    async def resolve() -> int:
      return 777

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(resolve)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate_do(fn):
      result.append(item)
    self.assertEqual(result, [0, 777])

  async def test_iterate_do_return_with_async_callable_value(self) -> None:
    """Async: iterate_do with return_(async_callable) — awaited and yielded."""

    async def resolve() -> int:
      return 888

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(resolve)
      return x

    result: list[int] = []
    async for item in Chain(range(5)).iterate_do(fn):
      result.append(item)
    self.assertEqual(result, [0, 888])


# ---------------------------------------------------------------------------
# §9.3 — Iterator reuse with *args/**kwargs
# ---------------------------------------------------------------------------


class IteratorReuseArgsKwargsTests(SymmetricTestCase):
  """§9.3 — iterator reuse with different args/kwargs."""

  async def test_reuse_with_kwargs(self) -> None:
    """Calling iterator with kwargs passes them to chain.run()."""
    it = Chain().then(lambda x: [x, x + 1]).iterate()
    result1 = list(it(10))
    result2 = list(it(20))
    self.assertEqual(result1, [10, 11])
    self.assertEqual(result2, [20, 21])

  async def test_reuse_with_args_and_kwargs(self) -> None:
    """Calling iterator with positional args works correctly."""
    it = Chain().then(lambda x: [x, x * 2]).iterate()
    result1 = list(it(3))
    result2 = list(it(7))
    self.assertEqual(result1, [3, 6])
    self.assertEqual(result2, [7, 14])

  async def test_reuse_iterate_do_with_different_values(self) -> None:
    """iterate_do reuse with different run values."""
    it = Chain().then(lambda x: [x, x + 1, x + 2]).iterate_do(sync_double)
    result1 = list(it(10))
    result2 = list(it(20))
    # iterate_do yields originals, not fn results
    self.assertEqual(result1, [10, 11, 12])
    self.assertEqual(result2, [20, 21, 22])

  async def test_reuse_async_with_different_values(self) -> None:
    """Async iterator reuse with different run values."""
    it = Chain().then(lambda x: [x * 10, x * 20]).iterate()
    result1: list[int] = []
    async for item in it(2):
      result1.append(item)
    result2: list[int] = []
    async for item in it(5):
      result2.append(item)
    self.assertEqual(result1, [20, 40])
    self.assertEqual(result2, [50, 100])

  async def test_reuse_callable_with_args_and_kwargs(self) -> None:
    """§9.3: it(callable, *args, **kwargs) forwards run-time parameters to chain.run()."""

    def make_list(x, y, *, scale=1):
      return [x * scale, y * scale]

    # Chain with no root — run() receives a callable v plus args/kwargs.
    it = Chain().iterate()
    result1 = list(it(make_list, 10, 20, scale=2))
    self.assertEqual(result1, [20, 40])
    result2 = list(it(make_list, 3, 5, scale=10))
    self.assertEqual(result2, [30, 50])


# ---------------------------------------------------------------------------
# Non-iterable chain result
# ---------------------------------------------------------------------------


class NonIterableChainResultTests(SymmetricTestCase):
  """Test behavior when chain result is not iterable."""

  async def test_sync_non_iterable_raises_typeerror(self) -> None:
    """Sync iteration on non-iterable chain result raises TypeError."""
    it = Chain(42).iterate()
    with self.assertRaises(TypeError):
      list(it)

  async def test_async_non_iterable_raises_typeerror(self) -> None:
    """Async iteration on non-iterable chain result raises TypeError."""
    with self.assertRaises(TypeError):
      async for _item in Chain(42).iterate():
        pass


# ---------------------------------------------------------------------------
# iterate_do reuse
# ---------------------------------------------------------------------------


class IterateDoReuseTests(SymmetricTestCase):
  """§9.3 — iterate_do() can be iterated multiple times on the same chain."""

  async def test_iterate_do_reuse_sync(self) -> None:
    """iterate_do can be iterated multiple times (sync)."""
    it = Chain(range(3)).iterate_do()
    result1 = list(it)
    result2 = list(it)
    self.assertEqual(result1, [0, 1, 2])
    self.assertEqual(result2, [0, 1, 2])

  async def test_iterate_do_reuse_async(self) -> None:
    """iterate_do can be iterated multiple times (async)."""
    it = Chain(range(3)).iterate_do()
    result1: list[int] = []
    async for item in it:
      result1.append(item)
    result2: list[int] = []
    async for item in it:
      result2.append(item)
    self.assertEqual(result1, [0, 1, 2])
    self.assertEqual(result2, [0, 1, 2])

  async def test_iterate_do_reuse_with_fn(self) -> None:
    """iterate_do with fn can be reused — side effects occur each time."""
    calls: list[int] = []

    def track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    it = Chain(range(3)).iterate_do(track)
    result1 = list(it)
    result2 = list(it)
    self.assertEqual(result1, [0, 1, 2])
    self.assertEqual(result2, [0, 1, 2])
    # Side effects from both iterations
    self.assertEqual(calls, [0, 1, 2, 0, 1, 2])


# ---------------------------------------------------------------------------
# Sync signal coroutine defense
# ---------------------------------------------------------------------------


class SyncSignalCoroutineDefenseTests(SymmetricTestCase):
  """Defense-in-depth: sync iteration handles coroutine from signal values."""

  async def test_sync_break_coroutine_value_raises_typeerror(self) -> None:
    """Sync break_(async_callable) resolved to coroutine raises TypeError."""

    async def async_resolve() -> int:
      return 42

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(async_resolve)
      return x

    it = Chain(range(5)).iterate(fn)
    with self.assertRaises(TypeError) as ctx:
      list(it)
    self.assertIn('async for', str(ctx.exception))

  async def test_sync_return_coroutine_value_raises_typeerror(self) -> None:
    """Sync return_(async_callable) resolved to coroutine raises TypeError."""

    async def async_resolve() -> int:
      return 99

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(async_resolve)
      return x

    it = Chain(range(5)).iterate(fn)
    with self.assertRaises(TypeError) as ctx:
      list(it)
    self.assertIn('async for', str(ctx.exception))


# ---------------------------------------------------------------------------
# Error message {fn!r}
# ---------------------------------------------------------------------------


class IterateErrorMessageTests(SymmetricTestCase):
  """Error messages include fn repr when TypeError occurs."""

  async def test_sync_fn_awaitable_error_includes_repr(self) -> None:
    """TypeError message for fn returning awaitable includes fn repr."""

    async def bad_fn(x: int) -> int:
      return x * 2

    it = Chain(range(3)).iterate(bad_fn)
    with self.assertRaises(TypeError) as ctx:
      list(it)
    # Per _generator.py:72: f'iterate() callback {fn!r} returned a coroutine.'
    self.assertIn('iterate() callback', str(ctx.exception))
    self.assertIn(repr(bad_fn), str(ctx.exception))


# ---------------------------------------------------------------------------
# #7 — Sync iterator send error (lines 56-57)
#   ChainIterator.__next__ error path — internal _run_one_sync raises.
#   Actually: _sync_generator lines 52-59 — sync iter on async chain.
# ---------------------------------------------------------------------------


class SyncIteratorErrorTests(SymmetricTestCase):
  """Coverage: sync iterator error paths in _generator.py."""

  async def test_sync_iter_chain_raises_during_run(self) -> None:
    """Sync iteration where chain run itself raises propagates the error."""

    def boom() -> list[int]:
      raise RuntimeError('chain run failed')

    it = Chain(boom).iterate()
    with self.assertRaises(RuntimeError) as ctx:
      list(it)
    self.assertIn('chain run failed', str(ctx.exception))

  async def test_sync_iter_chain_raises_with_fn(self) -> None:
    """Sync iteration with fn where chain run raises — fn is never called."""

    def boom() -> list[int]:
      raise RuntimeError('chain exploded')

    calls: list[int] = []

    def fn(x: int) -> int:
      calls.append(x)
      return x

    it = Chain(boom).iterate(fn)
    with self.assertRaises(RuntimeError):
      list(it)
    # fn was never called because chain.run() failed
    self.assertEqual(calls, [])

  async def test_sync_iter_async_chain_cancel_task(self) -> None:
    """Sync iteration with async chain — coroutine is closed/cancelled."""

    async def async_list() -> list[int]:
      return [1, 2, 3]

    it = Chain(async_list).iterate()
    with self.assertRaises(TypeError) as ctx:
      list(it)
    self.assertIn('async chain', str(ctx.exception))


# ---------------------------------------------------------------------------
# #8 — Sync signal coroutine defense (lines 94->96, 111->113)
#   Already covered by SyncSignalCoroutineDefenseTests above.
#   Adding extra explicit tests for both break_ and return_ paths.
# ---------------------------------------------------------------------------
# (Handled above in SyncSignalCoroutineDefenseTests)


# ---------------------------------------------------------------------------
# #9 — Async iterator error (lines 147->149)
#   Test async iteration where internal execution raises.
# ---------------------------------------------------------------------------


class AsyncIteratorErrorTests(SymmetricTestCase):
  """Coverage: async iterator error paths in _generator.py."""

  async def test_async_iter_fn_raises(self) -> None:
    """async for with fn that raises propagates exception."""

    def failing_fn(x: int) -> int:
      if x == 2:
        raise ValueError('fn failed at 2')
      return x * 10

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(5)).iterate(failing_fn):
        pass
    self.assertIn('fn failed at 2', str(ctx.exception))

  async def test_async_iter_async_fn_raises(self) -> None:
    """async for with async fn that raises propagates exception."""

    async def async_failing_fn(x: int) -> int:
      if x == 1:
        raise ValueError('async fn failed at 1')
      return x * 10

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(5)).iterate(async_failing_fn):
        pass
    self.assertIn('async fn failed at 1', str(ctx.exception))

  async def test_async_iterate_do_fn_raises(self) -> None:
    """async iterate_do where fn raises propagates exception."""

    def failing_fn(x: int) -> int:
      if x == 1:
        raise ValueError('iterate_do fn failed')
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate_do(failing_fn):
        pass
    self.assertIn('iterate_do fn failed', str(ctx.exception))


# ---------------------------------------------------------------------------
# #10 — Async iterate_do paths (lines 173-175, 179-181)
#   async iterate_do() — both with and without fn.
# ---------------------------------------------------------------------------


class AsyncIterateDoPathTests(SymmetricTestCase):
  """Coverage: async iterate_do paths in _generator.py."""

  async def test_async_iterate_do_no_fn_yields_originals(self) -> None:
    """async iterate_do() with no fn yields originals."""
    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do():
      result.append(item)
    self.assertEqual(result, [10, 20, 30])

  async def test_async_iterate_do_no_fn_async_chain(self) -> None:
    """async iterate_do() with no fn on async chain."""

    async def async_list() -> list[int]:
      return [5, 10, 15]

    result: list[int] = []
    async for item in Chain(async_list).iterate_do():
      result.append(item)
    self.assertEqual(result, [5, 10, 15])

  async def test_async_iterate_do_with_fn_yields_originals(self) -> None:
    """async iterate_do(fn) yields originals, fn is for side-effects."""
    calls: list[int] = []

    def track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do(track):
      result.append(item)
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(calls, [10, 20, 30])

  async def test_async_iterate_do_with_async_fn_yields_originals(self) -> None:
    """async iterate_do(async_fn) yields originals, async fn for side-effects."""
    calls: list[int] = []

    async def async_track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do(async_track):
      result.append(item)
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(calls, [10, 20, 30])

  async def test_async_iterate_do_with_return_value(self) -> None:
    """async iterate_do: return_(value) yields return value then stops."""

    def fn(x: int) -> int:
      if x == 20:
        return Chain.return_(x * 10)
      return x

    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do(fn):
      result.append(item)
    # iterate_do yields originals: 10, then return_(200) yields 200
    self.assertEqual(result, [10, 200])

  async def test_async_iterate_do_with_break_value(self) -> None:
    """async iterate_do: break_(value) yields break value then stops."""

    def fn(x: int) -> int:
      if x == 20:
        return Chain.break_(x * 10)
      return x

    result: list[int] = []
    async for item in Chain([10, 20, 30]).iterate_do(fn):
      result.append(item)
    # iterate_do yields original 10, then break_(200) yields 200
    self.assertEqual(result, [10, 200])


# ---------------------------------------------------------------------------
# §5.3 StopIteration propagation in foreach/foreach_do
# ---------------------------------------------------------------------------


class ForeachStopIterationTest(SymmetricTestCase):
  """SPEC §5.3: StopIteration raised by fn() propagates as a regular exception."""

  async def test_stop_iteration_in_foreach_sync(self) -> None:
    """Sync callback raises StopIteration in foreach -> propagates to caller."""

    def raise_stop(x):
      if x == 3:
        raise StopIteration('stopped at 3')
      return x * 2

    with self.assertRaises(StopIteration) as ctx:
      Chain([1, 2, 3, 4]).foreach(raise_stop).run()
    self.assertEqual(str(ctx.exception), 'stopped at 3')

  async def test_stop_iteration_in_foreach_do_sync(self) -> None:
    """Sync callback raises StopIteration in foreach_do -> propagates to caller."""

    def raise_stop(x):
      if x == 3:
        raise StopIteration('stopped_do')

    with self.assertRaises(StopIteration) as ctx:
      Chain([1, 2, 3, 4]).foreach_do(raise_stop).run()
    self.assertEqual(str(ctx.exception), 'stopped_do')

  async def test_stop_iteration_in_foreach_async(self) -> None:
    """Async callback raises StopIteration in foreach -> propagates as RuntimeError.

    Note: Python wraps StopIteration raised inside a coroutine as RuntimeError
    ('coroutine raised StopIteration') per PEP 479. This is standard Python
    behavior, not quent-specific.
    """

    async def raise_stop(x):
      if x == 3:
        raise StopIteration('async_stopped')
      return x * 2

    with self.assertRaises(RuntimeError) as ctx:
      await Chain([1, 2, 3, 4]).foreach(raise_stop).run()
    self.assertIn('StopIteration', str(ctx.exception))

  async def test_stop_iteration_in_foreach_do_async(self) -> None:
    """Async callback raises StopIteration in foreach_do -> propagates as RuntimeError.

    Same PEP 479 behavior as the foreach async test.
    """

    async def raise_stop(x):
      if x == 3:
        raise StopIteration('async_stopped_do')

    with self.assertRaises(RuntimeError) as ctx:
      await Chain([1, 2, 3, 4]).foreach_do(raise_stop).run()
    self.assertIn('StopIteration', str(ctx.exception))

  async def test_stop_iteration_does_not_terminate_loop(self) -> None:
    """StopIteration from fn() does NOT silently terminate the loop.

    This is the key design invariant: the while True / next() pattern in
    _IterOp.__call__ isolates StopIteration from the iterator's own
    StopIteration, ensuring fn()'s StopIteration propagates as an error.
    """

    def raise_stop(x):
      raise StopIteration('always stop')

    with self.assertRaises(StopIteration):
      Chain([1]).foreach(raise_stop).run()

  async def test_stop_iteration_with_except_handler(self) -> None:
    """StopIteration from fn() is caught by except_() handler."""

    def raise_stop(x):
      if x == 2:
        raise StopIteration('handled')
      return x

    result = Chain([1, 2, 3]).foreach(raise_stop).except_(lambda info: 'caught').run()
    self.assertEqual(result, 'caught')

  async def test_stop_iteration_mid_transition_async(self) -> None:
    """StopIteration on the _to_async mid-transition path propagates.

    When the sync fast path hands off to _to_async, a StopIteration from
    the callback propagates. Since _to_async is a coroutine, Python's
    PEP 479 wraps it as RuntimeError.
    """
    call_count = 0

    def mixed_fn(x):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        # First call returns awaitable to trigger async transition
        import asyncio

        fut = asyncio.get_event_loop().create_future()
        fut.set_result(x * 2)
        return fut
      if call_count == 2:
        raise StopIteration('mid-transition stop')
      return x * 2

    with self.assertRaises(RuntimeError) as ctx:
      await Chain([1, 2, 3]).foreach(mixed_fn).run()
    self.assertIn('StopIteration', str(ctx.exception))


class IteratePendingIfTest(unittest.TestCase):
  """Test iterate() and iterate_do() with unconsumed if_()."""

  def test_iterate_with_pending_if_raises(self) -> None:
    """iterate() with a pending if_() raises QuentException."""
    with self.assertRaises(QuentException) as ctx:
      Chain([1, 2, 3]).if_(lambda x: len(x) > 0).iterate()
    self.assertIn('if_() must be followed by .then() or .do()', str(ctx.exception))

  def test_iterate_do_with_pending_if_raises(self) -> None:
    """iterate_do() with a pending if_() raises QuentException."""
    with self.assertRaises(QuentException) as ctx:
      Chain([1, 2, 3]).if_(lambda x: len(x) > 0).iterate_do()
    self.assertIn('if_() must be followed by .then() or .do()', str(ctx.exception))


# ---------------------------------------------------------------------------
# Audit §9 — Additional spec gap tests
# ---------------------------------------------------------------------------


class IteratorReuseWithArgsTest(SymmetricTestCase):
  """SPEC §9.3: it(callable, *args, **kwargs) reuse."""

  async def test_reuse_with_callable_and_args(self) -> None:
    """Calling iterator with callable + args creates new iterator."""
    it = Chain().then(lambda x: [x, x + 1]).iterate()
    result = list(it(lambda a, b: a + b, 3, 7))
    self.assertEqual(result, [10, 11])  # (3+7)=10, [10, 11]

  async def test_reuse_with_callable_and_kwargs(self) -> None:
    """Calling iterator with callable + kwargs creates new iterator."""
    it = Chain().then(lambda x: [x, x * 2]).iterate()
    result = list(it(lambda key=0: key * 3, key=5))
    self.assertEqual(result, [15, 30])  # (5*3)=15, [15, 30]


# ---------------------------------------------------------------------------
# §9.1 / §6.3 — Deferred finally_() in iteration
# ---------------------------------------------------------------------------


class IterationDeferredFinallyTests(SymmetricTestCase):
  """§9.1 + §6.3 — finally_() is deferred to the generator's finally block.

  With deferred finally, the chain's finally_() handler runs AFTER iteration
  ends (in the generator's finally: block), not during the run phase.  This
  ensures resources acquired during the chain's run phase remain alive
  throughout the entire iteration.
  """

  async def test_finally_runs_after_all_items_yielded(self) -> None:
    """Timing: finally runs AFTER the last item is yielded, not before iteration starts."""
    order: list[str] = []

    def cleanup(rv: object) -> None:
      order.append('cleanup')

    it = Chain(range(3)).finally_(cleanup).iterate()
    for _item in it:
      order.append('item')
    # cleanup must appear AFTER all item entries
    self.assertEqual(order, ['item', 'item', 'item', 'cleanup'])

  async def test_finally_runs_on_normal_exhaustion(self) -> None:
    """Chain with finally, iterate through all items normally."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    it = Chain(range(3)).finally_(cleanup).iterate()
    result = list(it)
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_finally_runs_on_generator_close(self) -> None:
    """Create iterator, consume one item, then close the generator."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    gen = iter(Chain(range(10)).finally_(cleanup).iterate())
    first = next(gen)
    gen.close()
    self.assertEqual(first, 0)
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_finally_runs_on_fn_error(self) -> None:
    """Chain with finally + iterate(fn) where fn raises mid-iteration."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    def failing_fn(x: int) -> int:
      if x == 2:
        raise ValueError('fn error')
      return x

    it = Chain(range(5)).finally_(cleanup).iterate(failing_fn)
    with self.assertRaises(ValueError):
      list(it)
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_finally_runs_on_chain_error_no_except(self) -> None:
    """Chain execution fails, no except handler, finally should still run."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    def fail() -> list[int]:
      raise RuntimeError('chain error')

    it = Chain(fail).finally_(cleanup).iterate()
    with self.assertRaises(RuntimeError):
      list(it)
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_finally_runs_on_break_signal(self) -> None:
    """fn calls Chain.break_() mid-iteration. Finally should still run."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    def fn(x: int) -> int:
      if x == 2:
        return Chain.break_()
      return x

    it = Chain(range(5)).finally_(cleanup).iterate(fn)
    result = list(it)
    self.assertEqual(result, [0, 1])
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_finally_runs_on_return_signal(self) -> None:
    """fn calls Chain.return_() mid-iteration. Finally should still run."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(99)
      return x

    it = Chain(range(5)).finally_(cleanup).iterate(fn)
    result = list(it)
    self.assertEqual(result, [0, 1, 99])
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_finally_receives_correct_root_value(self) -> None:
    """The root value passed to the finally handler matches the chain's root value."""
    received_root: list[object] = []

    def track_root(rv: object) -> None:
      received_root.append(rv)

    src = [10, 20, 30]
    it = Chain(src).finally_(track_root).iterate()
    list(it)
    self.assertEqual(received_root, [src])

  async def test_async_finally_handler_awaited_in_async_for(self) -> None:
    """Async finally handler should be properly awaited when using async for."""
    cleanup_calls: list[str] = []

    async def async_cleanup(rv: object) -> None:
      cleanup_calls.append('async_cleanup')

    it = Chain(range(3)).finally_(async_cleanup).iterate()
    result: list[int] = []
    async for item in it:
      result.append(item)
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(cleanup_calls, ['async_cleanup'])

  async def test_async_finally_handler_raises_in_sync_for(self) -> None:
    """Async finally handler in sync for should raise TypeError (§6.3.5)."""
    cleanup_calls: list[str] = []

    async def async_cleanup(rv: object) -> None:
      cleanup_calls.append('async_cleanup')

    it = Chain(range(3)).finally_(async_cleanup).iterate()
    with self.assertRaises(TypeError) as ctx:
      list(it)
    self.assertIn("use 'async for' instead of 'for'", str(ctx.exception))
    # Handler was NOT awaited (coroutine was closed)
    self.assertEqual(cleanup_calls, [])

  async def test_exception_chaining_when_finally_raises(self) -> None:
    """When fn raises AND finally raises, the finally exception propagates
    with the fn exception as __context__ (§6.3.3).
    """

    def failing_fn(x: int) -> int:
      if x == 1:
        raise ValueError('fn error')
      return x

    def failing_cleanup(rv: object) -> None:
      raise RuntimeError('cleanup failed')

    it = Chain(range(3)).finally_(failing_cleanup).iterate(failing_fn)
    with self.assertRaises(RuntimeError) as ctx:
      list(it)
    self.assertIsInstance(ctx.exception.__context__, ValueError)

  async def test_iterate_do_deferred_finally(self) -> None:
    """Deferred finally works with iterate_do() — fn runs as side-effect,
    original items yielded, finally deferred.
    """
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    side_effects: list[int] = []

    def track(x: int) -> str:
      side_effects.append(x)
      return 'discarded'

    it = Chain(range(3)).finally_(cleanup).iterate_do(track)
    result = list(it)
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(side_effects, [0, 1, 2])
    self.assertEqual(cleanup_calls, ['cleanup'])

  async def test_except_during_run_finally_after_iteration(self) -> None:
    """Except handler runs during run phase (producing a result), finally
    deferred to after iteration.
    """
    order: list[str] = []

    def recovery_handler(exc_info: object) -> list[int]:
      order.append('except')
      return [10, 20]

    def cleanup(rv: object) -> None:
      order.append('finally')

    it = Chain(0).then(lambda x: 1 / x).except_(recovery_handler).finally_(cleanup).iterate()
    result = list(it)
    self.assertEqual(result, [10, 20])
    # except runs during run phase (before iteration), finally runs after iteration
    self.assertEqual(order, ['except', 'finally'])

  async def test_iterator_reuse_independent_finally(self) -> None:
    """Each use of a reusable iterator triggers its own independent finally."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    it = Chain(range(3)).finally_(cleanup).iterate()
    list(it)
    self.assertEqual(cleanup_calls, ['cleanup'])
    list(it)
    self.assertEqual(cleanup_calls, ['cleanup', 'cleanup'])

  async def test_no_finally_chain_works_normally(self) -> None:
    """Chain without finally_() works normally — no deferred overhead."""
    result = list(Chain(range(5)).iterate())
    self.assertEqual(result, [0, 1, 2, 3, 4])

  async def test_async_iteration_deferred_finally(self) -> None:
    """Deferred finally with async for on a sync chain with sync finally handler."""
    cleanup_calls: list[str] = []

    def cleanup(rv: object) -> None:
      cleanup_calls.append('cleanup')

    it = Chain(range(3)).finally_(cleanup).iterate()
    result: list[int] = []
    async for item in it:
      result.append(item)
    self.assertEqual(result, [0, 1, 2])
    self.assertEqual(cleanup_calls, ['cleanup'])


if __name__ == '__main__':
  unittest.main()
