"""Negative and edge-case tests for ops and chain API misuse."""
from __future__ import annotations

import unittest

from quent import Chain, QuentException


class TestIterateCoroutineDetection(unittest.TestCase):
  """A. _sync_generator raises TypeError when iterate() callback returns a coroutine."""

  def test_sync_iterate_with_async_callback_raises_type_error(self):
    async def async_cb(x):
      return x * 2

    gen = Chain([1, 2, 3]).iterate(async_cb)
    with self.assertRaises(TypeError) as ctx:
      list(gen)
    self.assertIn('coroutine', str(ctx.exception))


class TestElseWithoutIf(unittest.TestCase):
  """B. else_() without preceding if_() raises QuentException."""

  def test_else_without_if_raises(self):
    with self.assertRaises(QuentException):
      Chain(5).then(lambda x: x).else_(lambda x: x + 1)

  def test_else_after_then_raises(self):
    with self.assertRaises(QuentException):
      Chain(5).then(lambda x: x).do(lambda x: None).else_(lambda x: x)

  def test_else_on_empty_chain_raises(self):
    with self.assertRaises(QuentException):
      Chain().else_(lambda x: x)


class TestRunArgCombinations(unittest.TestCase):
  """D. run() with various arg combinations."""

  def test_run_no_args_on_empty_chain(self):
    self.assertIsNone(Chain().run())

  def test_run_with_value(self):
    self.assertEqual(Chain().then(lambda x: x * 2).run(5), 10)

  def test_run_with_args_passed_to_root(self):
    # Args to the root callable are passed via Chain(fn, *args), not run(*args).
    self.assertEqual(Chain(lambda x, y: x + y, 3, 4).run(), 7)

  def test_run_with_kwargs_passed_to_root(self):
    self.assertEqual(Chain(lambda x, y=0: x + y, 3, y=7).run(), 10)


class TestExceptDuplicate(unittest.TestCase):
  """F. except_() duplicate registration raises QuentException."""

  def test_except_duplicate_raises(self):
    with self.assertRaises(QuentException):
      Chain().except_(lambda rv, e: None).except_(lambda rv, e: None)


class TestFinallyDuplicate(unittest.TestCase):
  """G. finally_() duplicate registration raises QuentException."""

  def test_finally_duplicate_raises(self):
    with self.assertRaises(QuentException):
      Chain().finally_(lambda rv: None).finally_(lambda rv: None)


class TestGeneratorRepr(unittest.TestCase):
  """J. _Generator __repr__ returns '<Quent._Generator>'."""

  def test_generator_repr(self):
    gen = Chain([1, 2]).iterate()
    self.assertEqual(repr(gen), '<Quent._Generator>')

  def test_generator_with_fn_repr(self):
    gen = Chain([1, 2]).iterate(lambda x: x * 2)
    self.assertEqual(repr(gen), '<Quent._Generator>')


if __name__ == '__main__':
  unittest.main()
