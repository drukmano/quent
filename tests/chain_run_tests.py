"""Tests for Chain.run() and Chain.__call__(): basic execution, value propagation,
falsy value preservation, and chain reuse.
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from helpers import sync_fn, sync_identity, sync_side_effect


class TestChainRunBasic(unittest.TestCase):

  def test_empty_chain_returns_none(self):
    result = Chain().run()
    self.assertIsNone(result)

  def test_single_value_chain(self):
    result = Chain(42).run()
    self.assertEqual(result, 42)

  def test_single_callable_chain(self):
    result = Chain(lambda: 10).run()
    self.assertEqual(result, 10)

  def test_run_with_value_injection(self):
    chain = Chain().then(lambda x: x + 1)
    result = chain.run(5)
    self.assertEqual(result, 6)

  def test_run_with_value_and_args(self):
    # run(v, *args) creates Link(v, args). v is callable so v(*args) is called.
    result = Chain().run(lambda x, y: x + y, 3, 4)
    self.assertEqual(result, 7)

  def test_call_is_alias_for_run(self):
    chain = Chain(42)
    self.assertEqual(chain(), chain.run())
    self.assertEqual(chain(), 42)

  def test_callable_root_called(self):
    result = Chain(lambda: 'hello').run()
    self.assertEqual(result, 'hello')

  def test_non_callable_root_as_value(self):
    result = Chain('hello').run()
    self.assertEqual(result, 'hello')


class TestValuePropagation(unittest.TestCase):

  def test_then_replaces_current_value(self):
    result = Chain(1).then(lambda x: x + 1).run()
    self.assertEqual(result, 2)

  def test_do_preserves_current_value(self):
    result = Chain(1).do(lambda x: x + 100).run()
    self.assertEqual(result, 1)

  def test_chain_of_thens_propagates(self):
    result = (
      Chain(0)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, 5)

  def test_do_in_middle_preserves_flow(self):
    result = (
      Chain(1)
      .then(lambda x: x + 1)
      .do(lambda x: x * 100)
      .then(lambda x: x + 1)
      .run()
    )
    self.assertEqual(result, 3)

  def test_first_value_becomes_root_value(self):
    # The root_value is the first evaluated result; finally_ receives it.
    tracker = []
    result = (
      Chain(1)
      .then(lambda x: x + 1)
      .then(lambda x: x + 1)
      .finally_(lambda rv: tracker.append(rv))
      .run()
    )
    self.assertEqual(result, 3)
    # root_value should be 1 (the first evaluated value from root_link).
    self.assertEqual(tracker, [1])

  def test_run_value_becomes_root(self):
    # When run(v) injects a value, the evaluated run value becomes root_value.
    tracker = []
    chain = (
      Chain()
      .then(lambda x: x * 2)
      .finally_(lambda rv: tracker.append(rv))
    )
    result = chain.run(42)
    self.assertEqual(result, 84)
    # root_value should be 42 (the injected run value, evaluated).
    self.assertEqual(tracker, [42])

  def test_none_preserved_as_value(self):
    result = Chain(None).run()
    self.assertIsNone(result)
    # Crucially, this is None-the-value, not Null. Distinguish by confirming
    # it actually ran (empty chain would also return None, but Chain(None)
    # creates a root_link with v=None).
    tracker = []
    result = (
      Chain(None)
      .then(lambda x: (tracker.append(x), x)[1])
      .run()
    )
    self.assertIsNone(result)
    self.assertEqual(tracker, [None])

  def test_false_preserved(self):
    result = Chain(False).run()
    self.assertIs(result, False)

  def test_zero_preserved(self):
    result = Chain(0).run()
    self.assertEqual(result, 0)
    self.assertIsInstance(result, int)

  def test_empty_string_preserved(self):
    result = Chain('').run()
    self.assertEqual(result, '')
    self.assertIsInstance(result, str)

  def test_empty_list_preserved(self):
    result = Chain([]).run()
    self.assertEqual(result, [])
    self.assertIsInstance(result, list)

  def test_empty_dict_preserved(self):
    result = Chain({}).run()
    self.assertEqual(result, {})
    self.assertIsInstance(result, dict)

  def test_empty_set_preserved(self):
    result = Chain(set()).run()
    self.assertEqual(result, set())
    self.assertIsInstance(result, set)

  def test_zero_float_preserved(self):
    result = Chain(0.0).run()
    self.assertEqual(result, 0.0)
    self.assertIsInstance(result, float)

  def test_complex_zero_preserved(self):
    result = Chain(0j).run()
    self.assertEqual(result, 0j)
    self.assertIsInstance(result, complex)

  def test_empty_tuple_preserved(self):
    result = Chain(()).run()
    self.assertEqual(result, ())
    self.assertIsInstance(result, tuple)

  def test_empty_bytes_preserved(self):
    result = Chain(b'').run()
    self.assertEqual(result, b'')
    self.assertIsInstance(result, bytes)


class TestChainReuse(unittest.TestCase):

  def test_run_same_chain_twice(self):
    chain = Chain(42).then(lambda x: x + 1)
    self.assertEqual(chain.run(), 43)
    self.assertEqual(chain.run(), 43)

  def test_run_with_different_values(self):
    chain = Chain().then(lambda x: x * 2)
    self.assertEqual(chain.run(1), 2)
    self.assertEqual(chain.run(2), 4)
    self.assertEqual(chain.run(10), 20)

  def test_multiple_runs_independent(self):
    chain = Chain().then(lambda x: x + 1)
    results = [chain.run(i) for i in range(10)]
    self.assertEqual(results, list(range(1, 11)))


if __name__ == '__main__':
  unittest.main()
