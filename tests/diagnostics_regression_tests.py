# SPDX-License-Identifier: MIT
"""Regression tests for diagnostic isolation, bounded work, and rendering."""

from __future__ import annotations

import gc
import subprocess
import sys
import traceback
import weakref
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock

from quent import Q
from quent._types import ExceptionGroup


class DiagnosticsRegressionTests(TestCase):
  def test_optional_group_property_cannot_replace_original_error(self) -> None:
    class Error(ValueError):
      @property
      def exceptions(self):
        raise RuntimeError('optional metadata failed')

    original = Error('original')

    def fail():
      raise original

    with self.assertRaises(Error) as caught:
      Q(fail).run()
    self.assertIs(caught.exception, original)

  def test_mock_callable_is_not_rendered_as_a_pipeline(self) -> None:
    callback = Mock(return_value=1)
    q = Q(callback)
    self.assertEqual(q.run(), 1)
    self.assertIn('Mock', repr(q))

  def test_explicit_traceback_none_is_preserved(self) -> None:
    try:
      Q(1).then(lambda value: 1 / 0).run()
    except ZeroDivisionError as exc:
      rendered = ''.join(traceback.format_exception(type(exc), exc, None))
      self.assertNotIn('Traceback (most recent call last)', rendered)
      self.assertNotIn('<quent>', rendered)
      self.assertIn('ZeroDivisionError', rendered)

  def test_explicit_traceback_subset_is_preserved(self) -> None:
    def failing_step(value):
      raise ValueError('step failed')

    try:
      Q(1).then(failing_step).run()
    except ValueError as exc:
      tb = exc.__traceback__
      while tb.tb_next is not None:
        tb = tb.tb_next
      rendered = ''.join(traceback.format_exception(type(exc), exc, tb))
      self.assertNotIn('<quent>', rendered)
      self.assertIn('in failing_step', rendered)
      self.assertNotIn('in test_explicit_traceback_subset', rendered)

  def test_rendered_names_and_keyword_keys_are_sanitized(self) -> None:
    def fail(**kwargs):
      raise ValueError('plain')

    fail.__name__ = 'function\nFORGED\x1b[2J'
    q = Q(fail, **{'key\nFORGED\x1b[2J': 1})
    self.assertNotIn('\x1b', repr(q))
    self.assertNotIn('\n', repr(q))
    try:
      q.run()
    except ValueError as exc:
      self.assertNotIn('\x1b', ''.join(traceback.format_exception(exc)))
      if hasattr(exc, '__notes__'):
        self.assertNotIn('\n', exc.__notes__[0])

  def test_shared_and_cyclic_graphs_have_bounded_rendering(self) -> None:
    code = """
from quent import Q
inner = Q()
for _ in range(22):
    inner = Q().then(inner).then(inner)
assert len(repr(inner)) < 11000
cycle = Q()
cycle.then(cycle).then(cycle)
assert len(repr(cycle)) < 11000
def fail():
    raise ValueError('original')
try:
    Q(fail).then(inner).run()
except ValueError as exc:
    assert str(exc) == 'original'
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=3)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_runtime_root_is_rendered_and_marked(self) -> None:
    def failing_override():
      raise ValueError('override')

    try:
      Q('unused stored root').then(lambda value: value).run(failing_override)
    except ValueError as exc:
      tb = exc.__traceback__
      visualizations = []
      while tb is not None:
        if tb.tb_frame.f_code.co_filename == '<quent>':
          visualizations.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
      self.assertEqual(len(visualizations), 1)
      self.assertIn('failing_override', visualizations[0])
      self.assertNotIn('unused stored root', visualizations[0])
      self.assertEqual(visualizations[0].count('<----'), 1)
      if hasattr(exc, '__notes__'):
        self.assertIn('Q(failing_override)', exc.__notes__[0])

  def test_exception_group_metadata_does_not_retain_workers(self) -> None:
    class Owner:
      def fail(self, value):
        raise ValueError('worker')

    a, b = Owner(), Owner()
    refs = [weakref.ref(a), weakref.ref(b)]
    held = []
    Q(0).gather(lambda value: value, a.fail, b.fail).except_(lambda info: held.append(info.exc)).run()
    group = held[0]
    for exc in (group, *group.exceptions):
      self.assertFalse(set(getattr(exc, '__quent_meta__', {})) - {'quent'})
      self.assertFalse(hasattr(exc, '_quent_idx'))
      traceback.clear_frames(exc.__traceback__)
      exc.__traceback__ = None
      exc.__context__ = exc.__cause__ = None
    del a, b
    gc.collect()
    self.assertEqual([ref() for ref in refs], [None, None])


class AsyncDiagnosticsRegressionTests(IsolatedAsyncioTestCase):
  async def test_exception_group_leaves_are_cleaned(self) -> None:
    async def ok(value):
      return value

    async def fail(value):
      raise ValueError('worker')

    try:
      await Q(0).gather(ok, fail, fail).run()
    except ExceptionGroup as group:
      self.assertEqual(len(group.exceptions), 2)
      for exc in group.exceptions:
        self.assertFalse(set(getattr(exc, '__quent_meta__', {})) - {'quent'})
        self.assertFalse(hasattr(exc, '_quent_idx'))
