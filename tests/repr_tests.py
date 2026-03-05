"""Tests for __repr__ methods across the quent library:
Chain, _FrozenChain, _Generator, and Null.
"""
from __future__ import annotations

import functools
import unittest

from quent import Chain, Null, QuentException
from quent._chain import _FrozenChain
from quent._ops import _Generator
from quent._traceback import _get_link_name, _get_obj_name
from quent._core import Link
from helpers import sync_fn, partial_fn


# ---------------------------------------------------------------------------
# _get_obj_name helper
# ---------------------------------------------------------------------------

class TestGetObjName(unittest.TestCase):

  def test_named_function(self):
    self.assertEqual(_get_obj_name(sync_fn), 'sync_fn')

  def test_lambda(self):
    name = _get_obj_name(lambda x: x)
    self.assertIn('<lambda>', name)

  def test_class(self):
    self.assertEqual(_get_obj_name(int), 'int')

  def test_plain_value(self):
    self.assertEqual(_get_obj_name(42), '42')

  def test_string_value(self):
    name = _get_obj_name('hello')
    self.assertIn('hello', name)

  def test_partial(self):
    name = _get_obj_name(partial_fn)
    self.assertIn('partial', name)
    self.assertIn('add', name)

  def test_chain_uses_type_name(self):
    self.assertEqual(_get_obj_name(Chain()), 'Chain')


# ---------------------------------------------------------------------------
# _get_link_name helper
# ---------------------------------------------------------------------------

class TestGetLinkName(unittest.TestCase):

  def test_then_link(self):
    link = Link(sync_fn)
    self.assertEqual(_get_link_name(link), 'then')

  def test_do_link(self):
    link = Link(sync_fn, ignore_result=True)
    self.assertEqual(_get_link_name(link), 'do')

  def test_foreach_link(self):
    from quent._ops import _make_foreach
    inner = Link(sync_fn)
    wrapper = _make_foreach(inner, ignore_result=False)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'foreach')

  def test_foreach_do_link(self):
    from quent._ops import _make_foreach
    inner = Link(sync_fn)
    wrapper = _make_foreach(inner, ignore_result=True)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'foreach_do')

  def test_filter_link(self):
    from quent._ops import _make_filter
    inner = Link(sync_fn)
    wrapper = _make_filter(inner)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'filter')

  def test_gather_link(self):
    from quent._ops import _make_gather
    wrapper = _make_gather((sync_fn,))
    link = Link(wrapper)
    self.assertEqual(_get_link_name(link), 'gather')

  def test_with_link(self):
    from quent._ops import _make_with
    inner = Link(sync_fn)
    wrapper = _make_with(inner, ignore_result=False)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'with_')

  def test_with_do_link(self):
    from quent._ops import _make_with
    inner = Link(sync_fn)
    wrapper = _make_with(inner, ignore_result=True)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'with_do')


# ---------------------------------------------------------------------------
# Chain.__repr__
# ---------------------------------------------------------------------------

class TestChainRepr(unittest.TestCase):

  def test_empty_chain(self):
    self.assertEqual(repr(Chain()), 'Chain()')

  def test_with_root_int(self):
    r = repr(Chain(42))
    self.assertIn('Chain(', r)
    self.assertIn('42', r)

  def test_with_root_callable(self):
    r = repr(Chain(sync_fn))
    self.assertIn('Chain(', r)
    self.assertIn('sync_fn', r)

  def test_with_then_steps(self):
    r = repr(Chain(1).then(sync_fn).then(str))
    self.assertIn('.then(...)', r)

  def test_with_do_step(self):
    r = repr(Chain(1).do(print))
    self.assertIn('.do(...)', r)

  def test_with_foreach(self):
    r = repr(Chain([1, 2]).foreach(sync_fn))
    self.assertIn('.foreach(...)', r)

  def test_with_filter(self):
    r = repr(Chain([1, 2]).filter(sync_fn))
    self.assertIn('.filter(...)', r)

  def test_with_gather(self):
    r = repr(Chain(1).gather(sync_fn, str))
    self.assertIn('.gather(...)', r)

  def test_with_with(self):
    r = repr(Chain().with_(sync_fn))
    self.assertIn('.with_(...)', r)

  def test_with_with_do(self):
    r = repr(Chain().with_do(sync_fn))
    self.assertIn('.with_do(...)', r)

  def test_except_not_in_repr(self):
    # Chain.__repr__ only walks the main link list;
    # on_except_link is stored separately and not included.
    r = repr(Chain(1).except_(sync_fn))
    self.assertNotIn('.except_', r)

  def test_finally_not_in_repr(self):
    # Chain.__repr__ only walks the main link list;
    # on_finally_link is stored separately and not included.
    r = repr(Chain(1).finally_(sync_fn))
    self.assertNotIn('.finally_', r)

  def test_multiple_steps_in_order(self):
    r = repr(
      Chain(1)
      .then(sync_fn)
      .do(print)
      .then(str)
    )
    then_pos = r.index('.then(...)')
    do_pos = r.index('.do(...)')
    last_then_pos = r.index('.then(...)', then_pos + 1)
    self.assertLess(then_pos, do_pos)
    self.assertLess(do_pos, last_then_pos)

  def test_chain_starts_with_chain_paren(self):
    r = repr(Chain(42).then(sync_fn))
    self.assertTrue(r.startswith('Chain('))


# ---------------------------------------------------------------------------
# _FrozenChain.__repr__
# ---------------------------------------------------------------------------

class TestFrozenChainRepr(unittest.TestCase):

  def test_frozen_repr_wraps_chain(self):
    frozen = Chain(42).freeze()
    r = repr(frozen)
    self.assertTrue(r.startswith('Frozen('))
    self.assertIn('Chain(', r)
    self.assertTrue(r.endswith(')'))

  def test_frozen_contains_inner_chain_repr(self):
    chain = Chain(42).then(sync_fn)
    frozen = chain.freeze()
    r = repr(frozen)
    inner_repr = repr(chain)
    self.assertEqual(r, f'Frozen({inner_repr})')

  def test_frozen_empty_chain(self):
    r = repr(Chain().freeze())
    self.assertEqual(r, 'Frozen(Chain())')


# ---------------------------------------------------------------------------
# Null.__repr__
# ---------------------------------------------------------------------------

class TestNullRepr(unittest.TestCase):

  def test_null_repr(self):
    self.assertEqual(repr(Null), '<Null>')


# ---------------------------------------------------------------------------
# _Generator.__repr__
# ---------------------------------------------------------------------------

class TestGeneratorRepr(unittest.TestCase):

  def test_generator_repr(self):
    gen = Chain([1, 2, 3]).iterate()
    self.assertEqual(repr(gen), '<Quent._Generator>')

  def test_generator_repr_with_fn(self):
    gen = Chain([1, 2, 3]).iterate(sync_fn)
    self.assertEqual(repr(gen), '<Quent._Generator>')

  def test_generator_repr_iterate_do(self):
    gen = Chain([1, 2, 3]).iterate_do(sync_fn)
    self.assertEqual(repr(gen), '<Quent._Generator>')


if __name__ == '__main__':
  unittest.main()
