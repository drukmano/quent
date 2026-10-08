# SPDX-License-Identifier: MIT
"""Regressions for cancellation, executor bounds, and executor validation."""

import asyncio
import gc
import sys
import threading
import time
import warnings
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from quent import Q


class ChildCancellationTests(IsolatedAsyncioTestCase):
  async def test_independent_cancellation_outranks_regular_failure(self):
    for mode in ('foreach', 'foreach_do', 'gather'):
      for taskgroup in (True, False) if sys.version_info >= (3, 11) else (False,):
        ready = asyncio.Event()
        entered = 0
        raised = []

        async def work(i):
          nonlocal entered
          entered += 1
          if entered == 2:
            ready.set()
          await ready.wait()
          if i == 0:
            raised.append('cancellation')
            raise asyncio.CancelledError('independent cancellation')
          # Eager task creation can otherwise fail before worker 0 resumes,
          # allowing cleanup to cancel it before its independent raise occurs.
          await asyncio.sleep(0)
          raised.append('error')
          raise ValueError('regular failure')

        module = 'quent._gather_ops' if mode == 'gather' else 'quent._iter_ops'
        with self.subTest(mode=mode, taskgroup=taskgroup), patch(f'{module}._HAS_TASK_GROUP', taskgroup):
          q = (
            Q(None).gather(lambda _: work(0), lambda _: work(1))
            if mode == 'gather'
            else getattr(Q([0, 1]), mode)(work, concurrency=2)
          )
          with self.assertRaisesRegex(asyncio.CancelledError, 'independent cancellation'):
            await q.run()
          self.assertCountEqual(raised, ['cancellation', 'error'])

  async def test_cancelled_child_never_returns_partial_success(self):
    async def work(i):
      if i == 1:
        raise asyncio.CancelledError('cancelled child')
      return i

    for mode in ('foreach', 'foreach_do', 'gather'):
      for taskgroup in (True, False) if sys.version_info >= (3, 11) else (False,):
        module = 'quent._gather_ops' if mode == 'gather' else 'quent._iter_ops'
        with self.subTest(mode=mode, taskgroup=taskgroup), patch(f'{module}._HAS_TASK_GROUP', taskgroup):
          q = (
            Q(None).gather(*(lambda _, i=i: work(i) for i in range(3)))
            if mode == 'gather'
            else getattr(Q([0, 1, 2]), mode)(work, concurrency=2)
          )
          with self.assertRaises(asyncio.CancelledError):
            await q.run()

  async def test_cancelled_child_cancels_and_awaits_pending_sibling(self):
    finished = asyncio.Event()

    async def work(i):
      if i == 1:
        await asyncio.sleep(0)
        raise asyncio.CancelledError('cancelled child')
      try:
        await asyncio.Event().wait()
      finally:
        finished.set()

    with self.assertRaises(asyncio.CancelledError):
      await asyncio.wait_for(Q([0, 1]).foreach(work, concurrency=2).run(), 0.5)
    self.assertTrue(finished.is_set())

  async def test_sibling_cancellation_preserves_original_error_and_signal(self):
    async def work(i, outcome):
      if i:
        await asyncio.sleep(0)
        if outcome == 'error':
          raise ValueError('original error')
        Q.break_('stopped')
      await asyncio.Event().wait()

    with self.assertRaisesRegex(ValueError, 'original error'):
      await Q([0, 1]).foreach(lambda i: work(i, 'error'), concurrency=2).run()
    self.assertEqual(await Q([0, 1]).foreach(lambda i: work(i, 'break'), concurrency=2).run(), ['stopped'])

  async def test_self_cancellation_without_await_does_not_hide_from_taskgroup(self):
    async def work(i):
      if i:
        asyncio.current_task().cancel()
        return i
      await asyncio.Event().wait()

    with self.assertRaises(asyncio.CancelledError):
      await asyncio.wait_for(Q([0, 1]).foreach(work, concurrency=2).run(), 0.5)

  async def test_callback_may_explicitly_suppress_its_cancellation(self):
    async def work(i):
      asyncio.current_task().cancel()
      try:
        await asyncio.sleep(0)
      except asyncio.CancelledError:
        return i

    self.assertEqual(await Q([0, 1]).foreach(work, concurrency=2).run(), [0, 1])

  async def test_parent_cancellation_propagates_after_cleanup(self):
    for mode in ('foreach', 'foreach_do', 'gather'):
      for taskgroup in (True, False) if sys.version_info >= (3, 11) else (False,):
        for prior_cancellation in (False, True):
          started = asyncio.Event()
          finished = asyncio.Event()
          propagated = []

          async def work(i):
            started.set()
            try:
              await asyncio.Event().wait()
            finally:
              finished.set()

          async def parent():
            if prior_cancellation:
              asyncio.current_task().cancel('suppressed earlier request')
              try:
                await asyncio.sleep(0)
              except asyncio.CancelledError:
                pass
            q = (
              Q(None).gather(lambda _: work(0), lambda _: work(1))
              if mode == 'gather'
              else getattr(Q([0, 1]), mode)(work, concurrency=2)
            )
            try:
              return await q.run()
            except asyncio.CancelledError as exc:
              propagated.append(exc)
              raise

          module = 'quent._gather_ops' if mode == 'gather' else 'quent._iter_ops'
          with (
            self.subTest(mode=mode, taskgroup=taskgroup, prior_cancellation=prior_cancellation),
            patch(f'{module}._HAS_TASK_GROUP', taskgroup),
          ):
            task = asyncio.create_task(parent())
            await started.wait()
            task.cancel('parent reason')
            with self.assertRaises(asyncio.CancelledError) as caught:
              await task
            self.assertEqual(propagated[0].args, ('parent reason',))
            # CPython 3.10 strips cancellation arguments at a Task await boundary
            # even for a direct raise; check the library's own boundary above.
            if sys.version_info >= (3, 11):
              self.assertEqual(caught.exception.args, ('parent reason',))
              self.assertIsNone(caught.exception.__context__)
            self.assertTrue(finished.is_set())


class ExecutorBoundsTests(TestCase):
  def test_sync_gather_closes_rejected_lazy_return_coroutine(self):
    async def lazy():
      return 3

    def worker(_):
      Q.return_(lazy)

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      with self.assertRaises(TypeError):
        Q(None).gather(lambda _: 1, worker).run()
      gc.collect()
    self.assertFalse([warning for warning in caught if 'was never awaited' in str(warning.message)])
    self.assertEqual(Q(None).gather(lambda _: 1, lambda _: Q.return_()).run(), (1, None))

  def test_compatible_custom_executor_remains_supported(self):
    class ImmediateExecutor(Executor):
      def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
          future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
          future.set_exception(exc)
        return future

      def shutdown(self, *args, **kwargs):
        raise AssertionError('caller-owned executor must not be shut down')

    executor = ImmediateExecutor()
    self.assertEqual(Q([1, 2, 3]).foreach(lambda x: x + 1, concurrency=1, executor=executor).run(), [2, 3, 4])

  def test_supplied_executor_obeys_operation_limit(self):
    for mode in ('foreach', 'foreach_do', 'gather'):
      for limit in (1, 2):
        lock = threading.Lock()
        active = peak = 0

        def work(i):
          nonlocal active, peak
          if i == 0:
            return i
          with lock:
            active += 1
            peak = max(peak, active)
          time.sleep(0.01)
          with lock:
            active -= 1
          return i

        with self.subTest(mode=mode, limit=limit), ThreadPoolExecutor(max_workers=8) as pool:
          q = (
            Q(None).gather(*(lambda _, i=i: work(i) for i in range(9)), concurrency=limit, executor=pool)
            if mode == 'gather'
            else getattr(Q(range(9)), mode)(work, concurrency=limit, executor=pool)
          )
          self.assertEqual(list(q.run()), list(range(9)))
          self.assertLessEqual(peak, limit)
          self.assertEqual(pool.submit(lambda: 42).result(), 42)

  def test_serializing_executors_rejected_before_callbacks(self):
    import concurrent.futures

    families = [ProcessPoolExecutor]
    if hasattr(concurrent.futures, 'InterpreterPoolExecutor'):
      families.append(concurrent.futures.InterpreterPoolExecutor)
    for family in families:
      for mode in ('foreach', 'foreach_do', 'gather'):
        with (
          self.subTest(family=family.__name__, mode=mode),
          family(max_workers=1) as pool,
          self.assertRaisesRegex(TypeError, 'same-process'),
        ):
          getattr(Q([1]), mode)(abs, concurrency=1, executor=pool)
