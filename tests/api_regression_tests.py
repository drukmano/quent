# SPDX-License-Identifier: MIT
"""Regressions for nested ownership and public builder boundaries."""

from contextlib import nullcontext
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, Mock

from quent import Q, QuentException


class CallableDetectionTests(TestCase):
  def test_mock_callbacks_are_called_normally(self):
    for factory in (Mock, MagicMock):
      callback = factory(return_value=3)
      self.assertEqual(Q(1).then(callback).run(), 3)
      callback.assert_called_once_with(1)

  def test_pipeline_class_is_a_factory_not_an_instance(self):
    result = Q(1).then(Q).run()
    self.assertIsInstance(result, Q)
    self.assertEqual(result.run(), 1)


class CloneOwnershipTests(TestCase):
  def test_wrapped_nested_pipelines_are_cloned(self):
    for mode in ('foreach', 'gather', 'with_'):
      inner = Q().then(lambda x: x + 1)
      value = [1] if mode == 'foreach' else nullcontext(1) if mode == 'with_' else 1
      original = getattr(Q(value), mode)(inner)
      cloned = original.clone()
      inner.then(lambda x: x * 10)
      expected = [2] if mode == 'foreach' else (2,) if mode == 'gather' else 2
      with self.subTest(mode=mode):
        self.assertEqual(cloned.run(), expected)

  def test_concurrent_wrapper_and_decorator_clones_are_independent(self):
    inner = Q().then(lambda x: x + 1)
    original = Q().foreach(inner, concurrency=2)
    cloned = original.clone()
    decorated = original.as_decorator()(lambda: [1, 2])
    inner.then(lambda x: x * 10)
    self.assertEqual(cloned.run([1, 2]), [2, 3])
    self.assertEqual(decorated(), [2, 3])

  def test_drive_gen_nested_pipeline_is_cloned(self):
    inner = Q().then(lambda x: x + 1)
    cloned = Q().drive_gen(inner).clone()
    inner.then(lambda x: x * 10)
    self.assertEqual(cloned.run(iter_generator), 2)

  def test_side_effect_wrapper_nested_pipelines_are_cloned(self):
    for mode in ('foreach_do', 'with_do'):
      calls = []
      inner = Q().do(lambda x: calls.append('original'))
      original = getattr(Q([1] if mode == 'foreach_do' else nullcontext(1)), mode)(inner)
      cloned = original.clone()
      inner.do(lambda x: calls.append('added'))
      cloned.run()
      self.assertEqual(calls, ['original'])

  def test_cloned_conditional_representation_uses_cloned_body(self):
    def increment(x):
      return x + 1

    def textify(x):
      return str(x)

    inner = Q().then(increment)
    cloned = Q(1).if_(True).then(inner).clone()
    inner.then(textify)
    self.assertEqual(cloned.run(), 2)
    self.assertNotIn('textify', repr(cloned))

  def test_subclass_dictionary_and_slots_are_shallow_copied(self):
    class DictionaryQ(Q):
      pass

    class SlottedQ(Q):
      __slots__ = ('__private', 'tag')

      def set_private(self, value):
        self.__private = value

      def get_private(self):
        return self.__private

    for cls in (DictionaryQ, SlottedQ):
      q = cls(1)
      q.tag = []
      if isinstance(q, SlottedQ):
        q.set_private(q.tag)
      cloned = q.clone()
      self.assertIs(cloned.tag, q.tag)
      self.assertEqual(cloned.run(), 1)
      if isinstance(cloned, SlottedQ):
        self.assertIs(cloned.get_private(), q.tag)

  def test_subclass_state_survives_decorator_creation(self):
    class TaggedQ(Q):
      __slots__ = ('tag',)

      def _run(self, *args, **kwargs):
        self.tag.append('run')
        return super()._run(*args, **kwargs)

    q = TaggedQ()
    q.tag = []
    decorated = q.as_decorator()(lambda: 1)
    self.assertEqual(decorated(), 1)
    self.assertEqual(q.tag, ['run'])


def iter_generator():
  yield 1


class NestedCallbackTests(TestCase):
  def test_direct_pipeline_exit_reaches_outer_pipeline(self):
    worker = Q().then(lambda _: Q.exit_('stopped'))
    for mode in ('foreach', 'foreach_do', 'gather'):
      for concurrency in (None, 2):
        q = (
          Q(1).gather(worker, lambda _: 2, concurrency=concurrency or 1)
          if mode == 'gather'
          else getattr(Q([1, 2]), mode)(worker, concurrency=concurrency)
        )
        self.assertEqual(q.then(lambda _: 'unreached').run(), 'stopped')

  def test_direct_pipeline_break_reaches_foreach(self):
    worker = Q().then(lambda x: Q.break_('stopped') if x == 2 else x)
    for mode in ('foreach', 'foreach_do'):
      for concurrency in (None, 2):
        self.assertEqual(getattr(Q([1, 2, 3]), mode)(worker, concurrency=concurrency).run(), [1, 'stopped'])

  def test_explicit_lambda_wrapping_keeps_public_boundary(self):
    worker = Q().then(lambda _: Q.break_())
    with self.assertRaises(QuentException):
      Q([1]).foreach(lambda x: worker(x)).run()


class PublicTypingTests(TestCase):
  def test_transformed_output_annotations(self):
    try:
      from mypy import api
    except ImportError:
      self.skipTest('mypy is a development dependency')
    snippet = """\
from typing import Any, Coroutine
from typing_extensions import assert_type
from quent import Q

async def transform(x: int) -> str:
    return str(x)

assert_type(Q([1]).foreach(transform), Q[list[str]])
assert_type(Q([1]).foreach(Q[Any]().then(str)), Q[list[str]])
assert_type(Q([1]).foreach(None), Q[list[Any]])
assert_type(Q((1, 2)).foreach_do(print), Q[list[Any]])

@Q[Any]().then(str).as_decorator()
def decorated(x: int) -> int:
    return x

assert_type(decorated(1), str | Coroutine[Any, Any, str])
"""
    with TemporaryDirectory(prefix='quent-typing-') as cache:
      stdout, stderr, status = api.run(['--cache-dir', cache, '-c', snippet])
    self.assertEqual(status, 0, stdout + stderr)


class BuilderStateTests(TestCase):
  def test_failed_body_keeps_pending_predicate(self):
    for method in ('if_', 'while_'):
      q = getattr(Q(1), method)(False)
      with self.assertRaises(TypeError):
        q.then(5, 99)
      self.assertEqual(q.then(lambda x: x + 1).run(), 1)

  def test_failed_predicate_does_not_poison_pipeline(self):
    for method in ('if_', 'while_'):
      q = Q(1)
      with self.assertRaises(TypeError):
        getattr(q, method)(True, 99)
      self.assertEqual(q.run(), 1)

  def test_else_rejects_pending_loop(self):
    for method in ('else_', 'else_do'):
      q = Q(0).if_(False).then(100).while_(False)
      with self.assertRaises(QuentException):
        getattr(q, method)(lambda x: x)
      self.assertEqual(q.then(lambda x: x + 1).run(), 0)

  def test_nested_incomplete_pipeline_rejected(self):
    for method in ('if_', 'while_'):
      inner = getattr(Q(), method)(False)
      with self.assertRaises(QuentException):
        Q(1).then(inner).run()

  def test_buffer_cannot_bypass_run_guard_through_nesting_or_decorator(self):
    with self.assertRaises(QuentException):
      Q([1]).then(Q().buffer(1)).run()
    with self.assertRaises(QuentException):
      Q().buffer(1).as_decorator()
