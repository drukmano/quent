"""Regression coverage for execution and operator boundary contracts."""

from __future__ import annotations

import asyncio
import gc
import inspect
import sys
import unittest
import warnings

from quent import Q, QuentException
from quent._types import ExceptionGroup


async def resolve(value):
  return await value if inspect.isawaitable(value) else value


class EngineRegressionTests(unittest.IsolatedAsyncioTestCase):
  async def test_lazy_return_finishes_before_cleanup(self):
    for async_finally in (False, True):
      events = []

      async def lazy():
        events.append('lazy')
        return 5

      def cleanup(_):
        events.append('finally')

      async def async_cleanup(value):
        cleanup(value)

      result = await resolve(
        Q(1).then(lambda _: Q.return_(lazy)).finally_(async_cleanup if async_finally else cleanup).run()
      )
      self.assertEqual(result, 5)
      self.assertEqual(events, ['lazy', 'finally'])

  async def test_failing_lazy_return_awaits_cleanup(self):
    events = []

    def lazy():
      raise ValueError('lazy failed')

    async def cleanup(_):
      events.append('finally')

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      with self.assertRaisesRegex(ValueError, 'lazy failed'):
        await resolve(Q(1).then(lambda _: Q.return_(lazy)).finally_(cleanup).run())
      gc.collect()
    self.assertEqual(events, ['finally'])
    self.assertFalse([w for w in caught if 'never awaited' in str(w.message)])

  async def test_async_lazy_failure_preserves_signal_context(self):
    contexts = []

    async def lazy():
      raise ValueError('lazy')

    def step(_):
      Q.return_(lazy)

    async def async_step(_):
      Q.return_(lazy)

    for callback in (step, async_step):
      with self.assertRaises(ValueError) as caught:
        await resolve(Q(1).then(callback).run())
      contexts.append(type(caught.exception.__context__).__name__)
    self.assertEqual(contexts, ['_Return', '_Return'])

  async def test_runtime_awaitable_root_is_resolved(self):
    async def root():
      return 42

    pending = root()
    try:
      self.assertEqual(await resolve(Q().then(lambda value: value + 1).run(pending)), 43)
    finally:
      pending.close()

  async def test_exit_bypasses_base_exception_filter(self):
    calls = []
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      pipeline = (
        Q(1)
        .then(lambda _: Q.exit_('done'))
        .except_(lambda info: calls.append(info) or 'caught', exceptions=BaseException)
      )
    self.assertEqual(await resolve(pipeline.run()), 'done')
    self.assertEqual(calls, [])

  async def test_handler_signals_are_not_discarded_in_reraise_mode(self):
    for signal in (Q.return_, Q.break_):
      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with self.assertRaises(QuentException):
          await resolve(Q(1).then(lambda _: 1 / 0).except_(lambda _: signal(), reraise=True).run())

  async def test_handler_signal_exception_chains_match(self):
    chains = []

    def handler(_):
      Q.return_()

    async def async_handler(_):
      Q.return_()

    for callback in (handler, async_handler):
      with self.assertRaises(QuentException) as caught:
        await resolve(Q(1).then(lambda _: 1 / 0).except_(callback, reraise=True).run())
      exc = caught.exception
      chains.append((type(exc.__cause__), type(exc.__context__), exc.__suppress_context__))
    self.assertEqual(chains[0], chains[1])

  async def test_forbidden_handler_signal_is_a_failure_event(self):
    for async_root in (False, True):
      for async_handler in (False, True):
        events = []

        async def root():
          return 1

        def handler(_):
          Q.return_()

        async def async_callback(_):
          Q.return_()

        class InstrumentedQ(Q):
          on_step = staticmethod(lambda q, name, value, result, elapsed, exc: events.append((name, result, exc)))

        with self.assertRaises(QuentException) as caught:
          await resolve(
            InstrumentedQ(root if async_root else 1)
            .then(lambda _: 1 / 0)
            .except_(async_callback if async_handler else handler, reraise=True)
            .run()
          )
        self.assertEqual([e for e in events if e[0] == 'except_'], [('except_', None, caught.exception)])

  async def test_async_handler_factory_restores_pre_invocation_context(self):
    original = ValueError('original')

    def fail(_):
      raise original

    async def handler_failure():
      raise RuntimeError('handler')

    def handler(info):
      info.exc.__context__ = LookupError('mutated')
      info.exc.__suppress_context__ = True
      return handler_failure()

    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      with self.assertRaises(ValueError) as caught:
        await resolve(Q(1).then(fail).except_(handler, reraise=True).run())
    self.assertIs(caught.exception, original)
    self.assertIsNone(original.__context__)
    self.assertFalse(original.__suppress_context__)

  async def test_finally_instrumentation_records_resolved_completion(self):
    for async_root in (False, True):
      events = []

      async def root():
        return 1

      async def cleanup(_):
        events.append('body')
        return 77

      class InstrumentedQ(Q):
        on_step = staticmethod(lambda q, name, value, result, elapsed, exc: events.append((name, result, exc)))

      self.assertEqual(await resolve(InstrumentedQ(root if async_root else 1).finally_(cleanup).run()), 1)
      self.assertEqual(events[-2:], ['body', ('finally_', 77, None)])

  async def test_handler_failures_are_instrumented_once(self):
    for handler_kind in ('except_', 'finally_'):
      for async_root in (False, True):
        for async_handler in (False, True):
          events = []
          failure = ValueError('handler failed')

          async def root():
            return 1

          def bad(_):
            raise failure

          async def async_bad(value):
            bad(value)

          class InstrumentedQ(Q):
            on_step = staticmethod(lambda q, name, value, result, elapsed, exc: events.append((name, result, exc)))

          pipeline = InstrumentedQ(root if async_root else 1)
          if handler_kind == 'except_':
            pipeline.then(lambda _: 1 / 0)
          getattr(pipeline, handler_kind)(async_bad if async_handler else bad)
          with self.assertRaises(ValueError):
            await resolve(pipeline.run())
          self.assertEqual([e for e in events if e[0] == handler_kind], [(handler_kind, None, failure)])

  async def test_reraise_async_handler_records_its_result(self):
    events = []

    async def handler(_):
      return 77

    class InstrumentedQ(Q):
      on_step = staticmethod(lambda q, name, value, result, elapsed, exc: events.append((name, result, exc)))

    with self.assertRaises(ZeroDivisionError):
      await resolve(InstrumentedQ(1).then(lambda _: 1 / 0).except_(handler, reraise=True).run())
    self.assertEqual([e for e in events if e[0] == 'except_'], [('except_', 77, None)])

  async def test_failed_reraise_handler_records_only_failure(self):
    for async_root in (False, True):
      events = []
      failure = ValueError('handler')

      async def root():
        return 1

      async def handler(_):
        raise failure

      class InstrumentedQ(Q):
        on_step = staticmethod(lambda q, name, value, result, elapsed, exc: events.append((name, result, exc)))

      with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with self.assertRaises(ZeroDivisionError):
          await resolve(
            InstrumentedQ(root if async_root else 1).then(lambda _: 1 / 0).except_(handler, reraise=True).run()
          )
      self.assertEqual([e for e in events if e[0] == 'except_'], [('except_', None, failure)])

  async def test_reraising_original_is_still_a_handler_failure_event(self):
    events = []

    def handler(info):
      raise info.exc

    class InstrumentedQ(Q):
      on_step = staticmethod(lambda q, name, value, result, elapsed, exc: events.append((name, result, exc)))

    with self.assertRaises(ZeroDivisionError) as caught:
      InstrumentedQ(1).then(lambda _: 1 / 0).except_(handler).run()
    self.assertEqual([e for e in events if e[0] == 'except_'], [('except_', None, caught.exception)])

  async def test_failing_exception_formatting_preserves_cleanup_error(self):
    class BadTextError(Exception):
      def __str__(self):
        raise RuntimeError('broken str')

    BadTextError.__name__ = 'Bad\x1b[31mTextError'

    def body(_):
      raise BadTextError('body')

    failure = ValueError('cleanup')

    def cleanup(_):
      raise failure

    with self.assertRaises(ValueError) as caught:
      Q(1).then(body).finally_(cleanup).run()
    self.assertIs(caught.exception, failure)
    self.assertTrue(all('\x1b' not in note for note in getattr(failure, '__notes__', ())))

  async def test_discarded_handler_repr_failure_preserves_original(self):
    class BadReprError(Exception):
      def __repr__(self):
        raise RuntimeError('broken repr')

    def handler(_):
      raise BadReprError()

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      with self.assertRaises(ZeroDivisionError):
        Q(1).then(lambda _: 1 / 0).except_(handler, reraise=True).run()
    self.assertTrue(any('formatting failed' in str(w.message) for w in caught))

  async def test_lazy_return_cancellation_awaits_cleanup(self):
    entered = asyncio.Event()
    events = []

    async def lazy():
      entered.set()
      await asyncio.Event().wait()

    async def cleanup(_):
      await asyncio.sleep(0)
      events.append('closed')

    task = asyncio.create_task(Q(1).then(lambda _: Q.return_(lazy)).finally_(cleanup).run())
    await entered.wait()
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    self.assertEqual(events, ['closed'])

  async def test_drive_lazy_return_does_not_resume_generator(self):
    events = []

    def gen():
      try:
        yield 1
        events.append('resumed')
        yield 2
      finally:
        events.append('closed')

    async def compute(value):
      return value * 10

    result = await resolve(Q(gen).drive_gen(lambda value: Q.return_(compute, value)).run())
    self.assertEqual(result, 10)
    self.assertEqual(events, ['closed'])

  async def test_drive_failing_lazy_return_closes_generator(self):
    events = []

    def gen():
      try:
        yield 1
      finally:
        events.append('closed')

    def lazy():
      raise ValueError('lazy')

    generator = gen()
    try:
      with self.assertRaises(ValueError):
        Q(generator).drive_gen(lambda _: Q.return_(lazy)).run()
      self.assertEqual(events, ['closed'])
    finally:
      generator.close()

  async def test_nested_drive_worker_exit_escapes(self):
    events = []

    def gen():
      try:
        yield 1
        events.append('resumed')
      finally:
        events.append('closed')

    worker = Q().then(lambda _: Q.exit_('stop'))
    self.assertEqual(await resolve(Q(gen).drive_gen(worker).then(lambda _: 'downstream').run()), 'stop')
    self.assertEqual(events, ['closed'])

  async def test_explicit_public_drive_worker_keeps_its_boundary(self):
    def gen():
      yield 1

    worker = Q().then(lambda _: Q.exit_('inner'))
    self.assertEqual(
      Q(gen).drive_gen(lambda value: worker.run(value)).then(lambda value: 'after:' + value).run(), 'after:inner'
    )

  async def test_async_lazy_signal_misuse_is_rejected(self):
    async def lazy():
      Q.exit_('escaped')

    for signal in (Q.return_, Q.exit_):
      with self.assertRaises(QuentException):
        await resolve(Q(1).then(lambda _: signal(lazy)).run())

    with self.assertRaises(QuentException):
      await resolve(Q([1]).foreach(lambda _: Q.break_(lazy)).run())

  async def test_while_lazy_break_signal_misuse_is_rejected(self):
    with self.assertRaises(QuentException):
      await resolve(Q(1).while_(True).then(lambda _: Q.break_(lambda: Q.return_('escaped'))).run())

  async def test_async_exit_failure_preserves_context(self):
    body_error = ValueError('body')
    exit_error = RuntimeError('exit')

    class CM:
      def __enter__(self):
        return self

      def __exit__(self, *args):
        async def later():
          raise exit_error

        return later()

    def body(_):
      raise body_error

    with self.assertRaises(RuntimeError) as caught:
      await resolve(Q(CM()).with_(body).run())
    self.assertIs(caught.exception.__cause__, body_error)
    self.assertIs(caught.exception.__context__, body_error)

  async def test_awaitability_uses_type_protocol(self):
    class Data:
      def __init__(self):
        self.__await__ = lambda: iter(())

    self.assertEqual(await resolve(Q(Data()).then(lambda _: 'ok').run()), 'ok')

    class HiddenAwaitable:
      def __getattribute__(self, name):
        if name == '__await__':
          raise AttributeError(name)
        return super().__getattribute__(name)

      def __await__(self):
        yield from ()
        return 42

    self.assertEqual(await resolve(Q(HiddenAwaitable()).then(lambda value: value + 1).run()), 43)

    class AwaitableList(list):
      def __await__(self):
        yield from ()
        return 42

    # Exact built-in fast paths must leave subclass await protocols intact.
    self.assertEqual(await resolve(Q(AwaitableList()).then(lambda value: value + 1).run()), 43)
    self.assertEqual(await resolve(Q().then(lambda value: value + 1).run(AwaitableList())), 43)
    self.assertEqual(await resolve(Q([1]).foreach(lambda _: AwaitableList()).run()), [42])


@unittest.skipIf(sys.version_info >= (3, 11), 'Tests the Python 3.10 polyfill')
class ExceptionGroupRegressionTests(unittest.TestCase):
  def test_nested_filtering(self):
    group = ExceptionGroup('outer', [ExceptionGroup('inner', [ValueError('v'), TypeError('t')]), KeyError('k')])
    matched, rest = group.split(ValueError)
    self.assertIsInstance(matched.exceptions[0].exceptions[0], ValueError)
    self.assertIsInstance(rest.exceptions[0].exceptions[0], TypeError)
    self.assertIsInstance(group.subgroup(ValueError).exceptions[0].exceptions[0], ValueError)

  def test_split_partitions_with_one_predicate_evaluation(self):
    calls = []

    def condition(exc):
      if isinstance(exc, ExceptionGroup):
        return False
      calls.append(exc)
      return len(calls) == 2

    error = ValueError('v')
    matched, rest = ExceptionGroup('one', [error]).split(condition)
    self.assertIsNone(matched)
    self.assertEqual(rest.exceptions, (error,))
    self.assertEqual(calls, [error])

  def test_derive_does_not_suppress_copied_context(self):
    original = ExceptionGroup('original', [ValueError('v')])
    original.__context__ = KeyError('body')
    derived = original.derive([TypeError('t')])
    self.assertIs(derived.__context__, original.__context__)
    self.assertFalse(derived.__suppress_context__)
