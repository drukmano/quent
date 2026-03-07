"""Tests for Chain construction, fluency, and link ordering."""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from quent._core import Link
from helpers import sync_fn, SyncCM


def _noop(x=None):
  return x


class TestChainConstructor(unittest.TestCase):

  def test_empty_chain(self):
    c = Chain()
    self.assertIsNone(c.root_link)
    self.assertIsNone(c.first_link)
    self.assertIsNone(c.current_link)
    self.assertIsNone(c.on_finally_link)
    self.assertIsNone(c.on_except_link)

  def test_with_callable(self):
    c = Chain(sync_fn)
    self.assertIsNotNone(c.root_link)
    self.assertIs(c.root_link.v, sync_fn)

  def test_with_non_callable(self):
    c = Chain(42)
    self.assertIsNotNone(c.root_link)
    self.assertEqual(c.root_link.v, 42)

  def test_with_callable_and_args(self):
    fn = lambda x, y: x + y
    c = Chain(fn, 1, 2)
    self.assertIs(c.root_link.v, fn)
    self.assertEqual(c.root_link.args, (1, 2))

  def test_with_callable_and_kwargs(self):
    fn = lambda x=0: x
    c = Chain(fn, x=1)
    self.assertIs(c.root_link.v, fn)
    self.assertEqual(c.root_link.kwargs, {'x': 1})

  def test_with_none(self):
    c = Chain(None)
    self.assertIsNotNone(c.root_link)
    self.assertIsNone(c.root_link.v)

  def test_with_false(self):
    c = Chain(False)
    self.assertIsNotNone(c.root_link)
    self.assertIs(c.root_link.v, False)

  def test_with_zero(self):
    c = Chain(0)
    self.assertIsNotNone(c.root_link)
    self.assertEqual(c.root_link.v, 0)

  def test_with_empty_string(self):
    c = Chain('')
    self.assertIsNotNone(c.root_link)
    self.assertEqual(c.root_link.v, '')

  def test_with_empty_list(self):
    c = Chain([])
    self.assertIsNotNone(c.root_link)
    self.assertEqual(c.root_link.v, [])

  def test_with_null_is_empty_chain(self):
    c = Chain(Null)
    self.assertIsNone(c.root_link)
    self.assertIsNone(c.first_link)

  def test_is_nested_default_false(self):
    c = Chain()
    self.assertFalse(c.is_nested)

  def test_has_is_chain_class_attr(self):
    self.assertTrue(Chain._is_chain)
    c = Chain()
    self.assertTrue(c._is_chain)


class TestChainFluency(unittest.TestCase):

  def test_then_returns_self(self):
    c = Chain()
    result = c.then(_noop)
    self.assertIs(result, c)

  def test_do_returns_self(self):
    c = Chain()
    result = c.do(_noop)
    self.assertIs(result, c)

  def test_except_returns_self(self):
    c = Chain()
    result = c.except_(_noop)
    self.assertIs(result, c)

  def test_finally_returns_self(self):
    c = Chain()
    result = c.finally_(_noop)
    self.assertIs(result, c)

  def test_map_returns_self(self):
    c = Chain()
    result = c.map(_noop)
    self.assertIs(result, c)

  def test_filter_returns_self(self):
    c = Chain()
    result = c.filter(_noop)
    self.assertIs(result, c)

  def test_gather_returns_self(self):
    c = Chain()
    result = c.gather(_noop)
    self.assertIs(result, c)

  def test_with_returns_self(self):
    c = Chain()
    c.then(SyncCM())
    result = c.with_(_noop)
    self.assertIs(result, c)

  def test_chaining_multiple_ops(self):
    c = Chain()
    result = c.then(_noop).do(_noop).then(_noop)
    self.assertIs(result, c)

  def test_long_chain_100_steps(self):
    c = Chain()
    current = c
    for _ in range(100):
      current = current.then(_noop)
    self.assertIs(current, c)


class TestChainLinkOrdering(unittest.TestCase):

  def test_first_then_sets_first_link(self):
    c = Chain()
    self.assertIsNone(c.first_link)
    c.then(_noop)
    self.assertIsNotNone(c.first_link)
    self.assertIs(c.first_link.v, _noop)

  def test_second_then_chains(self):
    fn1 = lambda x: x
    fn2 = lambda x: x + 1
    c = Chain()
    c.then(fn1)
    c.then(fn2)
    self.assertIs(c.first_link.v, fn1)
    self.assertIsNotNone(c.first_link.next_link)
    self.assertIs(c.first_link.next_link.v, fn2)

  def test_link_ordering_preserved_5_links(self):
    fns = [lambda x, i=i: x + i for i in range(5)]
    c = Chain()
    for fn in fns:
      c.then(fn)

    link = c.first_link
    for i, fn in enumerate(fns):
      self.assertIsNotNone(link, f'Link {i} should exist')
      self.assertIs(link.v, fn, f'Link {i} has wrong function')
      link = link.next_link

    self.assertIsNone(link, 'No extra links beyond the 5 appended')

  def test_root_link_next_points_to_first_link(self):
    c = Chain(42)
    c.then(_noop)
    self.assertIsNotNone(c.root_link)
    self.assertIs(c.root_link.next_link, c.first_link)

  def test_empty_chain_first_then_no_current_link(self):
    c = Chain()
    c.then(_noop)
    # After first _then, first_link is set but current_link stays None
    # (current_link is only set starting from the second _then).
    self.assertIsNotNone(c.first_link)
    self.assertIsNone(c.current_link)

  def test_second_then_sets_current_link(self):
    fn1 = lambda x: x
    fn2 = lambda x: x
    c = Chain()
    c.then(fn1)
    c.then(fn2)
    self.assertIsNotNone(c.current_link)
    self.assertIs(c.current_link.v, fn2)

  def test_current_link_advances_with_each_then(self):
    fns = [lambda x, i=i: i for i in range(5)]
    c = Chain()
    for fn in fns:
      c.then(fn)
    # current_link should point to the last appended link
    self.assertIs(c.current_link.v, fns[-1])


if __name__ == '__main__':
  unittest.main()
