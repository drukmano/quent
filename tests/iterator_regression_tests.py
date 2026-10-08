# SPDX-License-Identifier: MIT
"""Regression tests for buffered iteration ownership and terminal boundaries."""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import threading
import warnings
from contextlib import nullcontext
from contextvars import ContextVar
from unittest import IsolatedAsyncioTestCase, TestCase

from quent import Q, QuentException, QuentIterator


class BufferRegressionTests(TestCase):
  def test_slow_consumer_receives_producer_error(self) -> None:
    code = """
import time
from quent import Q
def source():
    yield 0
    yield 1
    raise ValueError('producer failed')
g = iter(Q(source).buffer(1).iterate())
assert next(g) == 0
time.sleep(1.2)
assert next(g) == 1
try:
    next(g)
except ValueError as exc:
    assert str(exc) == 'producer failed'
else:
    raise AssertionError('producer error lost')
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=5)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_close_near_source_exhaustion_stops_thread(self) -> None:
    code = """
import threading, time, warnings
from quent import Q
before = set(threading.enumerate())
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter('always')
    g = iter(Q(range(3)).buffer(1).iterate())
    assert next(g) == 0
    time.sleep(.1)
    g.close()
assert not seen, seen
assert set(threading.enumerate()) == before
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=3)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_retained_callback_exception_does_not_retain_producer(self) -> None:
    threads = []
    closed = []

    def source():
      threads.append(threading.current_thread())
      try:
        yield from range(100)
      finally:
        closed.append(True)

    def fail(item):
      raise ValueError('callback')

    held = []
    try:
      try:
        list(Q(source).buffer(1).iterate(fail))
      except ValueError as exc:
        held.append(exc)
      self.assertEqual(len(held), 1)
      self.assertFalse(threads[0].is_alive())
      self.assertEqual(closed, [True])
    finally:
      held.clear()
      gc.collect()

  def test_sync_buffer_inherits_and_isolates_context(self) -> None:
    value = ContextVar('iterator_regression', default='missing')
    token = value.set('parent')
    Q.set('iterator_regression', 'parent')

    def source():
      yield value.get(), Q.get('iterator_regression', 'missing')
      value.set('worker')
      Q.set('iterator_regression', 'worker')

    try:
      self.assertEqual(list(Q(source).buffer(1).iterate()), [('parent', 'parent')])
      self.assertEqual(value.get(), 'parent')
      self.assertEqual(Q.get('iterator_regression'), 'parent')
    finally:
      value.reset(token)

  def test_source_close_failure_propagates_on_all_exits(self) -> None:
    class Source:
      def __init__(self):
        self.items = iter(range(10))
        self.closed = 0

      def __iter__(self):
        return self

      def __next__(self):
        return next(self.items)

      def close(self):
        self.closed += 1
        raise RuntimeError('source close failed')

    def fail(value):
      raise ValueError('callback failed')

    for mode in ('exhaust', 'callback', 'close'):
      with self.subTest(mode=mode):
        source = Source()
        gen = iter(Q(source).buffer(1).iterate(fail if mode == 'callback' else None))
        with self.assertRaisesRegex(RuntimeError, 'source close failed') as caught:
          if mode == 'close':
            next(gen)
            gen.close()
          else:
            list(gen)
        self.assertEqual(source.closed, 1)
        self.assertIsNot(caught.exception.__context__, caught.exception)
        if mode == 'callback':
          self.assertIsInstance(caught.exception.__context__, ValueError)


class AsyncBufferRegressionTests(IsolatedAsyncioTestCase):
  async def test_source_close_failure_propagates_on_all_exits(self) -> None:
    class Source:
      def __init__(self):
        self.items = iter(range(10))
        self.closed = 0

      def __aiter__(self):
        return self

      async def __anext__(self):
        try:
          return next(self.items)
        except StopIteration:
          raise StopAsyncIteration from None

      async def aclose(self):
        self.closed += 1
        await asyncio.sleep(0)
        raise RuntimeError('source close failed')

    def fail(value):
      raise ValueError('callback failed')

    for mode in ('exhaust', 'callback', 'close'):
      with self.subTest(mode=mode):
        source = Source()
        gen = Q(source).buffer(1).iterate(fail if mode == 'callback' else None).__aiter__()
        with self.assertRaisesRegex(RuntimeError, 'source close failed') as caught:
          if mode == 'close':
            await anext(gen)
            await gen.aclose()
          else:
            async for _ in gen:
              pass
        self.assertEqual(source.closed, 1)
        self.assertIsNot(caught.exception.__context__, caught.exception)
        if mode == 'callback':
          self.assertIsInstance(caught.exception.__context__, ValueError)

  async def test_retained_callback_exception_does_not_retain_producer(self) -> None:
    producers = []
    closed = []

    async def source():
      producers.append(asyncio.current_task())
      try:
        for item in range(100):
          yield item
      finally:
        closed.append(True)

    def fail(value):
      raise ValueError('callback')

    held = []
    try:
      async for _ in Q(source).buffer(1).iterate(fail):
        pass
    except ValueError as exc:
      held.append(exc)
    self.assertEqual(len(held), 1)
    self.assertTrue(producers[0].done())
    self.assertEqual(closed, [True])

  async def test_source_cancellation_reaches_consumer(self) -> None:
    async def source():
      yield 1
      raise asyncio.CancelledError('source cancelled')

    gen = Q(source).buffer(1).iterate().__aiter__()
    self.assertEqual(await anext(gen), 1)

    async def consume_error():
      # Check before wait_for's Task boundary: Python 3.10 loses cancellation
      # message arguments when a cancelled task is awaited.
      with self.assertRaisesRegex(asyncio.CancelledError, 'source cancelled'):
        await anext(gen)

    await asyncio.wait_for(consume_error(), 1)

  async def test_source_base_exceptions_reach_consumer(self) -> None:
    code = """
import asyncio
from quent import Q
async def main():
    for error_type in (GeneratorExit, SystemExit, KeyboardInterrupt):
        async def source():
            yield 1
            raise error_type('source failed')
        gen = Q(source).buffer(1).iterate().__aiter__()
        assert await anext(gen) == 1
        try:
            await anext(gen)
        except error_type as exc:
            assert str(exc) == 'source failed'
        else:
            raise AssertionError('source failure lost')
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=3)
    self.assertEqual(result.returncode, 0, result.stderr)

  async def test_aclose_stops_producer_before_context_and_finally(self) -> None:
    events = []
    blocked = asyncio.Event()

    class CM:
      async def __aenter__(self):
        return self.source()

      async def source(self):
        try:
          yield 0
          blocked.set()
          await asyncio.Event().wait()
        finally:
          events.append('producer closed')

      async def __aexit__(self, *args):
        events.append('CM exit')

    gen = (
      Q(CM())
      .with_(lambda value: value)
      .buffer(1)
      .finally_(lambda value: events.append('finally'))
      .iterate()
      .__aiter__()
    )
    self.assertEqual(await anext(gen), 0)
    await asyncio.wait_for(blocked.wait(), 1)
    await gen.aclose()
    self.assertEqual(events, ['producer closed', 'CM exit', 'finally'])

  async def test_aclose_closes_source_suspended_at_yield(self) -> None:
    closed = []

    async def source():
      try:
        for item in range(100):
          yield item
      finally:
        closed.append(True)

    retained_source = source()
    gen = Q(retained_source).buffer(1).iterate().__aiter__()
    await anext(gen)
    await gen.aclose()
    self.assertEqual(closed, [True])

  async def test_consumer_cancellation_waits_for_source_cleanup(self) -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    async def source():
      try:
        started.set()
        await asyncio.Event().wait()
        yield 0
      finally:
        await asyncio.sleep(0)
        closed.set()

    async def consume():
      async for _ in Q(source).buffer(1).iterate():
        pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await asyncio.wait_for(task, 1)
    self.assertTrue(closed.is_set())


class TerminalBoundaryRegressionTests(IsolatedAsyncioTestCase):
  async def test_sync_iteration_closes_rejected_lazy_pipeline_return(self) -> None:
    async def items():
      return [1]

    for method in ('iterate', 'iterate_do', 'flat_iterate', 'flat_iterate_do'):
      for has_finally in (False, True):
        with self.subTest(method=method, has_finally=has_finally):
          finalized = []
          q = Q(0).then(lambda value: Q.return_(items))
          if has_finally:
            q.finally_(lambda value: finalized.append(value))
          with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with self.assertRaisesRegex(TypeError, 'sync iteration on an async pipeline'):
              list(getattr(q, method)())
            gc.collect()
          self.assertFalse([warning for warning in caught if 'never awaited' in str(warning.message)])
          self.assertEqual(finalized, [0] if has_finally else [])

    self.assertEqual([item async for item in Q(0).then(lambda value: Q.return_(items)).iterate()], [1])

  async def test_custom_iterator_runner_keeps_its_calling_convention(self) -> None:
    def runner(value, args, kwargs):
      return [value, args, kwargs]

    iterator = QuentIterator(runner, None, False)(1, 2, key=3)
    self.assertEqual(list(iterator), [1, (2,), {'key': 3}])
    self.assertEqual([item async for item in iterator], [1, (2,), {'key': 3}])

  def test_sync_iteration_closes_rejected_async_finally(self) -> None:
    async def cleanup(value):
      await asyncio.sleep(0)

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      with self.assertRaises(TypeError):
        list(Q([1]).finally_(cleanup).iterate())
      gc.collect()
    self.assertFalse([warning for warning in caught if 'never awaited' in str(warning.message)])

  async def test_deferred_with_signals_sync_and_async(self) -> None:
    for signal in (Q.break_, Q.return_, Q.exit_):
      for method in ('iterate', 'iterate_do', 'flat_iterate', 'flat_iterate_do'):
        for valued in (False, True):
          with self.subTest(signal=signal.__name__, method=method, valued=valued):

            def callback(value):
              return signal(42) if valued else signal()

            expected = [42] if valued else []
            iterator = getattr(Q(nullcontext([1])).with_(callback), method)()
            self.assertEqual(list(iterator), expected)
            self.assertEqual([item async for item in iterator], expected)

  async def test_lazy_value_signals_are_misuse(self) -> None:
    for signal in (Q.break_, Q.return_, Q.exit_):
      with self.subTest(signal=signal.__name__):

        def callback(value):
          return signal(lambda: Q.break_(99))

        with self.assertRaises(QuentException):
          list(Q([1]).iterate(callback))
        with self.assertRaises(QuentException):
          [item async for item in Q([1]).iterate(callback)]

        async def lazy():
          await asyncio.sleep(0)
          Q.return_(99)

        with self.assertRaises(QuentException):
          [item async for item in Q([1]).iterate(lambda item: signal(lazy))]

  async def test_deferred_with_instrumentation(self) -> None:
    records = []
    previous = Q.on_step
    Q.on_step = lambda q, name, value, result, elapsed, exc: records.append((name, result, exc))
    try:
      q = Q(nullcontext([1])).with_(lambda value: value)
      self.assertEqual(list(q.iterate()), [1])
      self.assertEqual(records, [('root', q._root_link.v, None), ('with_', [1], None)])
      records.clear()
      self.assertEqual([item async for item in q.iterate()], [1])
      self.assertEqual([record[0] for record in records], ['root', 'with_'])

      def fail(value):
        raise ValueError('deferred body')

      for asynchronous in (False, True):
        records.clear()
        iterator = Q(nullcontext([1])).with_(fail).iterate()
        with self.assertRaises(ValueError):
          if asynchronous:
            [item async for item in iterator]
          else:
            list(iterator)
        self.assertEqual([record[0] for record in records], ['root', 'with_'])
        self.assertIsInstance(records[-1][2], ValueError)
    finally:
      Q.on_step = previous
