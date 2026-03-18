# SPDX-License-Identifier: MIT
"""Implementation coverage tests for _iter_ops.py and _generator.py.

These tests target internal code paths for coverage.
For spec-mandated behavior tests, see iteration_tests.py.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import unittest

from quent import Chain
from tests.fixtures import async_double
from tests.symmetric import SymmetricTestCase

# ---------------------------------------------------------------------------
# #1 — _to_async error metadata (lines 199->201, 215-216)
#   Mid-transition iteration where fn raises after async transition.
#   The exception should get _set_link_temp_args metadata with item/index.
# ---------------------------------------------------------------------------


class ToAsyncErrorMetadataTests(SymmetricTestCase):
  """Coverage: _to_async exception metadata path in _iter_ops.py."""

  async def test_map_mid_transition_error(self) -> None:
    """map() mid-transition: fn raises after async handoff — exception gets metadata."""
    call_count = [0]

    def mixed_fn(x: int) -> int:
      call_count[0] += 1
      if call_count[0] <= 1:
        return x * 10  # First item sync
      if call_count[0] == 2:
        # Second item returns coroutine — triggers _to_async
        async def _inner() -> int:
          return x * 10

        return _inner()
      # Third item raises — exercises lines 215-216
      raise ValueError(f'boom at {x}')

    with self.assertRaises(ValueError) as ctx:
      await Chain([1, 2, 3]).foreach(mixed_fn).run()
    self.assertIn('boom at 3', str(ctx.exception))

  async def test_foreach_do_mid_transition_error(self) -> None:
    """foreach_do() mid-transition: fn raises after async handoff."""
    call_count = [0]

    def mixed_fn(x: int) -> None:
      call_count[0] += 1
      if call_count[0] <= 1:
        return None  # First item sync
      if call_count[0] == 2:

        async def _inner() -> None:
          pass

        return _inner()
      raise ValueError(f'foreach_do boom at {x}')

    with self.assertRaises(ValueError) as ctx:
      await Chain([1, 2, 3]).foreach_do(mixed_fn).run()
    self.assertIn('foreach_do boom at 3', str(ctx.exception))

  async def test_map_mid_transition_await_then_error(self) -> None:
    """map() mid-transition: awaitable result, then fn raises on subsequent item."""
    call_count = [0]

    def mixed_fn(x: int) -> int:
      call_count[0] += 1
      if call_count[0] == 1:
        return x * 10  # sync
      if call_count[0] == 2:
        # Return awaitable — enters _to_async
        async def _async_result() -> int:
          return x * 10

        return _async_result()
      if call_count[0] == 3:
        # Still in _to_async, returns awaitable
        async def _async_result2() -> int:
          return x * 10

        return _async_result2()
      # 4th call raises while in _to_async loop
      raise ValueError(f'late boom at {x}')

    with self.assertRaises(ValueError) as ctx:
      await Chain([1, 2, 3, 4]).foreach(mixed_fn).run()
    self.assertIn('late boom at 4', str(ctx.exception))


# ---------------------------------------------------------------------------
# #2 — _full_async exception metadata (lines 235-239)
#   Pure async iteration where fn raises.
# ---------------------------------------------------------------------------


class FullAsyncExceptionMetadataTests(SymmetricTestCase):
  """Coverage: _full_async exception metadata path in _iter_ops.py."""

  async def test_map_full_async_error(self) -> None:
    """map() with async iterable — fn raises, exception gets metadata."""

    class AsyncItems:
      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [10, 20, 30]:
          yield i

    def failing_fn(x: int) -> int:
      if x == 20:
        raise ValueError(f'async fail at {x}')
      return x * 2

    with self.assertRaises(ValueError) as ctx:
      await Chain(AsyncItems()).foreach(failing_fn).run()
    self.assertIn('async fail at 20', str(ctx.exception))

  async def test_foreach_do_full_async_error(self) -> None:
    """foreach_do() with async iterable — fn raises."""

    class AsyncItems:
      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [1, 2, 3]:
          yield i

    def failing_fn(x: int) -> None:
      if x == 2:
        raise ValueError(f'async foreach_do fail at {x}')

    with self.assertRaises(ValueError) as ctx:
      await Chain(AsyncItems()).foreach_do(failing_fn).run()
    self.assertIn('async foreach_do fail at 2', str(ctx.exception))


# ---------------------------------------------------------------------------
# #3 — Dual-protocol preference in _IterOp (lines 248-253)
#   Object with both __iter__ and __aiter__: prefer async when loop is running.
# ---------------------------------------------------------------------------


class DualProtocolPreferenceTests(SymmetricTestCase):
  """Coverage: dual-protocol iterable prefers async when event loop is running."""

  async def test_map_dual_protocol_prefers_async(self) -> None:
    """map() with dual-protocol iterable in async context uses async path."""

    class DualIterable:
      def __iter__(self):
        return iter([1, 2, 3])

      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [10, 20, 30]:
          yield i

    # In async context (event loop running), should prefer __aiter__
    result = await Chain(DualIterable()).foreach(lambda x: x * 2).run()
    # If async path is chosen, items are 10,20,30 -> doubled to 20,40,60
    self.assertEqual(result, [20, 40, 60])

  async def test_foreach_do_dual_protocol_prefers_async(self) -> None:
    """foreach_do() with dual-protocol iterable in async context."""
    calls: list[int] = []

    class DualIterable:
      def __iter__(self):
        return iter([1, 2, 3])

      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [10, 20, 30]:
          yield i

    def track(x: int) -> None:
      calls.append(x)

    result = await Chain(DualIterable()).foreach_do(track).run()
    # async path: items are 10, 20, 30
    self.assertEqual(result, [10, 20, 30])
    self.assertEqual(calls, [10, 20, 30])


# ---------------------------------------------------------------------------
# #4 — Async concurrent worker awaitable await (lines 389->391)
#   Concurrent foreach/foreach_do where fn is async (returns awaitable).
# ---------------------------------------------------------------------------


class AsyncConcurrentWorkerTests(SymmetricTestCase):
  """Coverage: async concurrent worker awaitable await in _iter_ops.py."""

  async def test_concurrent_map_async_fn(self) -> None:
    """map(async_fn, concurrency=2) — workers await the coroutine."""

    result = await Chain([1, 2, 3, 4]).foreach(async_double, concurrency=2).run()
    self.assertEqual(sorted(result), [2, 4, 6, 8])
    # Results preserve input order
    self.assertEqual(result, [2, 4, 6, 8])

  async def test_concurrent_foreach_do_async_fn(self) -> None:
    """foreach_do(async_fn, concurrency=2) — workers await the coroutine."""
    calls: list[int] = []

    async def async_track(x: int) -> str:
      calls.append(x)
      return 'discarded'

    result = await Chain([10, 20, 30]).foreach_do(async_track, concurrency=2).run()
    self.assertEqual(sorted(result), [10, 20, 30])
    self.assertEqual(sorted(calls), [10, 20, 30])


# ---------------------------------------------------------------------------
# #5 — ConcurrentIterOp dual-protocol (lines 471-476)
#   map(fn, concurrency=2) with dual-protocol iterable in async context.
# ---------------------------------------------------------------------------


class ConcurrentDualProtocolTests(SymmetricTestCase):
  """Coverage: ConcurrentIterOp dual-protocol iterable in async context."""

  async def test_concurrent_map_dual_protocol(self) -> None:
    """map(fn, concurrency=2) with dual-protocol iterable uses async path."""

    class DualIterable:
      def __iter__(self):
        return iter([1, 2, 3])

      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [10, 20, 30]:
          yield i

    result = await Chain(DualIterable()).foreach(lambda x: x * 2, concurrency=2).run()
    # async path: items 10, 20, 30 -> doubled
    self.assertEqual(result, [20, 40, 60])

  async def test_concurrent_foreach_do_dual_protocol(self) -> None:
    """foreach_do(fn, concurrency=2) with dual-protocol iterable uses async path."""
    calls: list[int] = []

    class DualIterable:
      def __iter__(self):
        return iter([1, 2, 3])

      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [100, 200, 300]:
          yield i

    def track(x: int) -> None:
      calls.append(x)

    result = await Chain(DualIterable()).foreach_do(track, concurrency=2).run()
    self.assertEqual(sorted(result), [100, 200, 300])
    self.assertEqual(sorted(calls), [100, 200, 300])


# ---------------------------------------------------------------------------
# #6 — Concurrent n==1 _Break (lines 499-501)
#   Concurrent iteration with exactly 1 item where fn raises break_().
# ---------------------------------------------------------------------------


class ConcurrentSingleItemBreakTests(SymmetricTestCase):
  """Coverage: concurrent iteration with n==1 and _Break."""

  async def test_concurrent_map_single_item_break(self) -> None:
    """map(fn, concurrency=2) with 1 item that raises break_()."""

    def fn(x: int) -> int:
      return Chain.break_(99)

    result = Chain([1]).foreach(fn, concurrency=2).run()
    # n==1, _Break with value -> appends to empty partial results
    self.assertEqual(result, [99])

  async def test_concurrent_map_single_item_break_no_value(self) -> None:
    """map(fn, concurrency=2) with 1 item that raises break_() with no value."""

    def fn(x: int) -> int:
      return Chain.break_()

    result = Chain([1]).foreach(fn, concurrency=2).run()
    # n==1, _Break with no value -> returns partial results (empty list)
    self.assertEqual(result, [])

  async def test_concurrent_foreach_do_single_item_break(self) -> None:
    """foreach_do(fn, concurrency=2) with 1 item that raises break_()."""

    def fn(x: int) -> int:
      return Chain.break_(42)

    result = Chain([1]).foreach_do(fn, concurrency=2).run()
    self.assertEqual(result, [42])


# ---------------------------------------------------------------------------
# #11 — Async iterate_do with fn (lines 188-190, 194-196)
#   async iterate_do(fn) where fn processes each item.
#   These are the _Return handler lines that await resolved coroutines.
# ---------------------------------------------------------------------------


class AsyncIterateDoReturnAwaitTests(SymmetricTestCase):
  """Coverage: async iterate_do _Return/_Break await paths in _generator.py."""

  async def test_async_iterate_do_return_with_async_value(self) -> None:
    """async iterate_do: return_(async_callable) — resolved value is awaited."""

    async def resolve() -> int:
      return 999

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(resolve)
      return x

    result: list[int] = []
    async for item in Chain(range(3)).iterate_do(fn):
      result.append(item)
    # iterate_do yields original 0, then return_ resolves to 999
    self.assertEqual(result, [0, 999])

  async def test_async_iterate_do_break_with_async_value(self) -> None:
    """async iterate_do: break_(async_callable) — resolved value is awaited."""

    async def resolve() -> int:
      return 777

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(resolve)
      return x

    result: list[int] = []
    async for item in Chain(range(3)).iterate_do(fn):
      result.append(item)
    # iterate_do yields original 0, then break_ resolves to 777
    self.assertEqual(result, [0, 777])

  async def test_async_iterate_return_await_error(self) -> None:
    """async iterate: return_(async_callable) where await raises."""

    async def failing_resolve() -> int:
      raise ValueError('resolve failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(failing_resolve)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate(fn):
        pass
    self.assertIn('resolve failed', str(ctx.exception))

  async def test_async_iterate_break_await_error(self) -> None:
    """async iterate: break_(async_callable) where await raises."""

    async def failing_resolve() -> int:
      raise ValueError('break resolve failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(failing_resolve)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate(fn):
        pass
    self.assertIn('break resolve failed', str(ctx.exception))

  async def test_async_iterate_do_return_await_error(self) -> None:
    """async iterate_do: return_(async_callable) where await raises."""

    async def failing_resolve() -> int:
      raise ValueError('iterate_do return resolve failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(failing_resolve)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate_do(fn):
        pass
    self.assertIn('iterate_do return resolve failed', str(ctx.exception))

  async def test_async_iterate_do_break_await_error(self) -> None:
    """async iterate_do: break_(async_callable) where await raises."""

    async def failing_resolve() -> int:
      raise ValueError('iterate_do break resolve failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(failing_resolve)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate_do(fn):
        pass
    self.assertIn('iterate_do break resolve failed', str(ctx.exception))


# ---------------------------------------------------------------------------
# Additional coverage: _async_generator line 147->149
# When chain_run produces result with __aiter__ (no wrapping needed).
# ---------------------------------------------------------------------------


class AsyncGeneratorNoWrapTests(SymmetricTestCase):
  """Coverage: _async_generator where chain result already has __aiter__."""

  async def test_async_iter_chain_returns_async_iterable(self) -> None:
    """async for on chain that produces an async iterable directly (no wrapping)."""

    class AsyncIterable:
      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [100, 200, 300]:
          yield i

    result: list[int] = []
    async for item in Chain(AsyncIterable()).iterate():
      result.append(item)
    self.assertEqual(result, [100, 200, 300])

  async def test_async_iter_chain_returns_async_iterable_with_fn(self) -> None:
    """async for on async iterable result with fn transformation."""

    class AsyncIterable:
      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [10, 20, 30]:
          yield i

    result: list[int] = []
    async for item in Chain(AsyncIterable()).iterate(lambda x: x * 2):
      result.append(item)
    self.assertEqual(result, [20, 40, 60])


# ---------------------------------------------------------------------------
# Additional coverage: _eval_signal_value error paths in _async_generator
# Lines 173-175 (_Break eval error) and 188-190 (_Return eval error)
# These fire when the callable passed to break_/return_ itself raises
# during evaluation (not when awaiting a coroutine).
# ---------------------------------------------------------------------------


class AsyncSignalEvalErrorTests(SymmetricTestCase):
  """Coverage: _eval_signal_value error paths in _async_generator."""

  async def test_async_break_eval_error(self) -> None:
    """async iterate: break_(callable) where callable raises during eval."""

    def failing_callable() -> int:
      raise ValueError('break eval failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(failing_callable)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate(fn):
        pass
    self.assertIn('break eval failed', str(ctx.exception))

  async def test_async_return_eval_error(self) -> None:
    """async iterate: return_(callable) where callable raises during eval."""

    def failing_callable() -> int:
      raise ValueError('return eval failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(failing_callable)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate(fn):
        pass
    self.assertIn('return eval failed', str(ctx.exception))

  async def test_async_iterate_do_break_eval_error(self) -> None:
    """async iterate_do: break_(callable) where callable raises during eval."""

    def failing_callable() -> int:
      raise ValueError('iterate_do break eval failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(failing_callable)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate_do(fn):
        pass
    self.assertIn('iterate_do break eval failed', str(ctx.exception))

  async def test_async_iterate_do_return_eval_error(self) -> None:
    """async iterate_do: return_(callable) where callable raises during eval."""

    def failing_callable() -> int:
      raise ValueError('iterate_do return eval failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(failing_callable)
      return x

    with self.assertRaises(ValueError) as ctx:
      async for _item in Chain(range(3)).iterate_do(fn):
        pass
    self.assertIn('iterate_do return eval failed', str(ctx.exception))


# ---------------------------------------------------------------------------
# Additional coverage: _sync_generator eval error paths
# Lines 90-92 (_Break eval error) and 107-109 (_Return eval error)
# These fire when the callable passed to break_/return_ raises
# during _eval_signal_value in the sync generator.
# ---------------------------------------------------------------------------


class SyncSignalEvalErrorTests(SymmetricTestCase):
  """Coverage: _eval_signal_value error paths in _sync_generator."""

  async def test_sync_break_eval_error(self) -> None:
    """Sync iterate: break_(callable) where callable raises during eval."""

    def failing_callable() -> int:
      raise ValueError('sync break eval failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(failing_callable)
      return x

    it = Chain(range(3)).iterate(fn)
    with self.assertRaises(ValueError) as ctx:
      list(it)
    self.assertIn('sync break eval failed', str(ctx.exception))

  async def test_sync_return_eval_error(self) -> None:
    """Sync iterate: return_(callable) where callable raises during eval."""

    def failing_callable() -> int:
      raise ValueError('sync return eval failed')

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(failing_callable)
      return x

    it = Chain(range(3)).iterate(fn)
    with self.assertRaises(ValueError) as ctx:
      list(it)
    self.assertIn('sync return eval failed', str(ctx.exception))


# ---------------------------------------------------------------------------
# Additional coverage: sync signal coroutine defense — no .close() path
# Lines 94->96 and 111->113 — awaitable without .close() from signal value
# ---------------------------------------------------------------------------


class SyncSignalAwaitableNoCloseTests(SymmetricTestCase):
  """Coverage: sync signal awaitable without .close() path."""

  async def test_sync_break_awaitable_no_close(self) -> None:
    """Sync iterate: break_ value resolves to awaitable without .close()."""

    class AwaitableNoClose:
      """Awaitable with __await__ but no .close() method."""

      def __await__(self):
        return iter([42])

    def make_awaitable() -> AwaitableNoClose:
      return AwaitableNoClose()

    def fn(x: int) -> int:
      if x == 1:
        return Chain.break_(make_awaitable)
      return x

    it = Chain(range(3)).iterate(fn)
    with self.assertRaises(TypeError) as ctx:
      list(it)
    self.assertIn('async for', str(ctx.exception))

  async def test_sync_return_awaitable_no_close(self) -> None:
    """Sync iterate: return_ value resolves to awaitable without .close()."""

    class AwaitableNoClose:
      """Awaitable with __await__ but no .close() method."""

      def __await__(self):
        return iter([99])

    def make_awaitable() -> AwaitableNoClose:
      return AwaitableNoClose()

    def fn(x: int) -> int:
      if x == 1:
        return Chain.return_(make_awaitable)
      return x

    it = Chain(range(3)).iterate(fn)
    with self.assertRaises(TypeError) as ctx:
      list(it)
    self.assertIn('async for', str(ctx.exception))


# ---------------------------------------------------------------------------
# Additional coverage: _full_async _ControlFlowSignal re-raise (line 236)
# This requires return_() during full async iteration path.
# ---------------------------------------------------------------------------


class FullAsyncControlFlowSignalTests(SymmetricTestCase):
  """Coverage: _ControlFlowSignal re-raise in _full_async."""

  async def test_map_full_async_return_signal(self) -> None:
    """map() with async-only iterable and return_() — signal propagates."""

    class AsyncOnlyIterable:
      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [1, 2, 3]:
          yield i

    def fn(x: int) -> int:
      if x == 2:
        return Chain.return_(99)
      return x * 10

    # return_() in map propagates as _Return signal, replacing chain result
    result = await Chain(AsyncOnlyIterable()).foreach(fn).run()
    self.assertEqual(result, 99)


# ---------------------------------------------------------------------------
# Additional coverage: dual-protocol NO event loop path (lines 250-251, 473-474)
# These need to run outside an event loop. Use a thread to avoid the running loop.
# ---------------------------------------------------------------------------


class DualProtocolSyncFallbackTests(SymmetricTestCase):
  """Coverage: dual-protocol iterable with no event loop uses sync path."""

  async def test_map_dual_protocol_sync_fallback(self) -> None:
    """map() with dual-protocol iterable in sync context uses __iter__."""

    class DualIterable:
      def __iter__(self):
        return iter([1, 2, 3])

      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [10, 20, 30]:
          yield i

    # Run in sync context (no event loop) — uses __iter__ path
    def run_sync() -> list[int]:
      return Chain(DualIterable()).foreach(lambda x: x * 2).run()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      result = await asyncio.get_event_loop().run_in_executor(executor, run_sync)
    # Sync path: __iter__ gives [1,2,3] -> doubled = [2,4,6]
    self.assertEqual(result, [2, 4, 6])

  async def test_concurrent_map_dual_protocol_sync_fallback(self) -> None:
    """map(fn, concurrency=2) with dual-protocol iterable in sync context."""

    class DualIterable:
      def __iter__(self):
        return iter([1, 2, 3])

      def __aiter__(self):
        return self._gen()

      async def _gen(self):
        for i in [10, 20, 30]:
          yield i

    def run_sync() -> list[int]:
      return Chain(DualIterable()).foreach(lambda x: x * 2, concurrency=2).run()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      result = await asyncio.get_event_loop().run_in_executor(executor, run_sync)
    # Sync path: __iter__ gives [1,2,3] -> doubled = [2,4,6]
    self.assertEqual(result, [2, 4, 6])


# ---------------------------------------------------------------------------
# Additional coverage: concurrent n==1 return_ (line 501)
# Concurrent iteration with exactly 1 item where fn raises return_().
# ---------------------------------------------------------------------------


class ConcurrentSingleItemReturnTests(SymmetricTestCase):
  """Coverage: concurrent n==1 with return_() signal (non-Break _ControlFlowSignal)."""

  async def test_concurrent_map_single_item_return(self) -> None:
    """map(fn, concurrency=2) with 1 item that raises return_()."""

    def fn(x: int) -> int:
      return Chain.return_(99)

    # return_ is a _ControlFlowSignal but not _Break — line 501 raises it
    result = Chain([1]).foreach(fn, concurrency=2).run()
    # return_ replaces the chain result
    self.assertEqual(result, 99)


if __name__ == '__main__':
  unittest.main()
