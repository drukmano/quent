"""Tests for _resolve_nested_chain, _stringify_chain extra_links,
_format_call_args, and _get_true_source_link.

Covers the chain-resolution logic, argument formatting, and source link
drilling, focusing on internal helper functions used during traceback
construction.
"""
from __future__ import annotations

import functools
import unittest

from quent import Chain
from quent._core import Link, Null
from quent._ops import _make_foreach, _make_filter, _make_gather, _make_with
from quent._traceback import (
  _Ctx,
  _format_call_args,
  _get_link_name,
  _get_obj_name,
  _get_true_source_link,
  _resolve_nested_chain,
  _stringify_chain,
)
from helpers import sync_fn, raise_fn, sync_identity, partial_fn


# ---------------------------------------------------------------------------
# TestResolveNestedChain
# ---------------------------------------------------------------------------

class TestResolveNestedChain(unittest.TestCase):

  def test_no_args(self):
    """No args/kwargs -> no nested_root_link created."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, None, None, 0, ctx)
    # Result is the stringified inner chain
    self.assertIn('Chain()', result)
    self.assertIn('.then(sync_fn)', result)

  def test_with_positional_args(self):
    """args=(5, 6), first is 5 -> nested_root_link created."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, (5, 6), None, 0, ctx)
    # Should show the nested chain with the injected root value
    self.assertIn('Chain(', result)
    self.assertIn('5', result)

  def test_with_ellipsis_args(self):
    """args=(...,), Ellipsis is not Null -> nested_root_link created."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, (...,), None, 0, ctx)
    # Ellipsis triggers nested_root_link creation since ... is not Null
    # repr(...) is 'Ellipsis', so _get_obj_name returns 'Ellipsis'
    self.assertIn('Chain(', result)
    self.assertIn('Ellipsis', result)

  def test_kwargs_only(self):
    """kwargs={'k': 1}, no positional -> kwargs shown in visualization."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, None, {'k': 1}, 0, ctx)
    # kwargs-only: nested_root_link is created with Null value + kwargs
    self.assertIn('k=', result)

  def test_found_flag_propagation(self):
    """ctx.found is propagated through nested resolution."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    # Set the source_link to the inner chain's first link
    ctx = _Ctx(source_link=inner.first_link, link_temp_args=None)
    self.assertFalse(ctx.found)
    _resolve_nested_chain(link, None, None, 0, ctx)
    # After resolution, found should be True since source_link was in inner
    self.assertTrue(ctx.found)


# ---------------------------------------------------------------------------
# TestStringifyChainExtraLinks
# ---------------------------------------------------------------------------

class TestStringifyChainExtraLinks(unittest.TestCase):

  def test_extra_links_displayed(self):
    """extra_links appear in the stringified chain."""
    c = Chain(sync_fn)
    extra = Link(sync_fn)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(
      c, nest_lvl=0, ctx=ctx,
      extra_links=[(extra, 'iterate')],
    )
    self.assertIn('.iterate(sync_fn)', result)

  def test_extra_links_arrow_marking(self):
    """extra_links can be marked with the <---- arrow."""
    c = Chain(sync_fn)
    extra = Link(sync_fn)
    ctx = _Ctx(source_link=extra, link_temp_args=None)
    result = _stringify_chain(
      c, nest_lvl=0, ctx=ctx,
      extra_links=[(extra, 'iterate')],
    )
    self.assertIn('.iterate(sync_fn) <----', result)

  def test_except_and_finally_in_visualization(self):
    """except_ and finally_ appear in stringified chain."""
    c = Chain(sync_fn).except_(sync_identity).finally_(sync_identity)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.except_(sync_identity)', result)
    self.assertIn('.finally_(sync_identity)', result)

  def test_gather_fns_shown(self):
    """Gather function names appear in visualization."""
    def fn_a(x):
      return x + 1

    def fn_b(x):
      return x + 2

    c = Chain(1).gather(fn_a, fn_b)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('fn_a', result)
    self.assertIn('fn_b', result)


# ---------------------------------------------------------------------------
# TestFormatCallArgs
# ---------------------------------------------------------------------------

class TestFormatCallArgs(unittest.TestCase):

  def test_no_args_no_kwargs(self):
    self.assertEqual(_format_call_args(None, None), '')

  def test_ellipsis_arg(self):
    self.assertEqual(_format_call_args((...,), None), '...')

  def test_positional_args(self):
    result = _format_call_args((1, 'hello'), None)
    self.assertEqual(result, "1, 'hello'")

  def test_kwargs_only(self):
    result = _format_call_args(None, {'key': 'val'})
    self.assertEqual(result, "key='val'")

  def test_both(self):
    result = _format_call_args((42,), {'flag': True})
    self.assertEqual(result, '42, flag=True')

  def test_none_inputs(self):
    """Both None -> empty string."""
    self.assertEqual(_format_call_args(None, None), '')

  def test_empty_tuple_and_empty_dict(self):
    """Empty args and kwargs -> empty string (falsy containers)."""
    self.assertEqual(_format_call_args((), {}), '')

  def test_callable_in_args(self):
    """Callable argument uses _get_obj_name for display."""
    result = _format_call_args((sync_fn,), None)
    self.assertEqual(result, 'sync_fn')

  def test_multiple_kwargs(self):
    """Multiple kwargs are comma-separated."""
    result = _format_call_args(None, {'a': 1, 'b': 2})
    self.assertIn('a=1', result)
    self.assertIn('b=2', result)


# ---------------------------------------------------------------------------
# TestGetTrueSourceLink
# ---------------------------------------------------------------------------

class TestGetTrueSourceLink(unittest.TestCase):

  def test_non_chain_link(self):
    """Non-chain link is returned directly."""
    link = Link(sync_fn)
    result = _get_true_source_link(link, None)
    self.assertIs(result, link)

  def test_chain_link_drills(self):
    """Chain link follows to inner chain's root_link."""
    inner = Chain(sync_fn)
    link = Link(inner)  # link.is_chain = True
    result = _get_true_source_link(link, None)
    self.assertIs(result, inner.root_link)

  def test_none_fallback_to_root_link(self):
    """When source_link is None, returns root_link fallback."""
    fallback = Link(sync_fn)
    result = _get_true_source_link(None, fallback)
    self.assertIs(result, fallback)

  def test_nested_chain_with_no_root(self):
    """Chain with no root_link stops drilling and returns the link."""
    inner = Chain()  # root_link is None
    link = Link(inner)
    result = _get_true_source_link(link, None)
    # inner has no root_link, so loop breaks and returns the link itself
    self.assertIs(result, link)

  def test_both_none(self):
    """source_link=None, root_link=None -> returns None."""
    result = _get_true_source_link(None, None)
    self.assertIsNone(result)

  def test_drills_through_original_value_chain(self):
    """link.original_value._is_chain -> drills into that chain's root_link."""
    inner = Chain(sync_fn)
    link = Link(lambda: None, original_value=inner)
    result = _get_true_source_link(link, None)
    self.assertIs(result, inner.root_link)

  def test_multi_level_drill(self):
    """Drills through multiple levels of nested chains."""
    innermost = Chain(sync_fn)
    middle = Chain(innermost)
    outer_link = Link(middle)
    result = _get_true_source_link(outer_link, None)
    # Should drill through middle -> innermost -> root_link of innermost
    self.assertIs(result, innermost.root_link)


# ---------------------------------------------------------------------------
# BEYOND SPEC: Additional tests for thoroughness
# ---------------------------------------------------------------------------

class TestResolveNestedChainEdgeCases(unittest.TestCase):

  def test_nested_chain_with_args_and_kwargs(self):
    """Both args and kwargs are handled in nested resolution."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, (5,), {'k': 1}, 0, ctx)
    self.assertIn('Chain(', result)
    self.assertIn('5', result)

  def test_nested_chain_deep_nesting(self):
    """3 levels of nested chains resolve correctly."""
    inner3 = Chain().then(sync_fn)
    inner2 = Chain().then(inner3)
    inner1 = Chain().then(inner2)
    link = Link(inner1)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, None, None, 0, ctx)
    # Should contain multiple levels of Chain(
    chain_count = result.count('Chain(')
    self.assertGreaterEqual(chain_count, 3)

  def test_resolve_with_original_value_set(self):
    """When link.original_value is set, it is used for resolution."""
    inner = Chain().then(sync_fn)
    link = Link(lambda: None, original_value=inner)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _resolve_nested_chain(link, None, None, 0, ctx)
    self.assertIn('Chain()', result)

  def test_found_flag_not_propagated_when_not_found(self):
    """ctx.found stays False when source_link is not in the chain."""
    inner = Chain().then(sync_fn)
    link = Link(inner)
    unrelated = Link(lambda: None)
    ctx = _Ctx(source_link=unrelated, link_temp_args=None)
    _resolve_nested_chain(link, None, None, 0, ctx)
    self.assertFalse(ctx.found)


class TestStringifyChainEdgeCases(unittest.TestCase):

  def test_stringify_with_root_link_parameter(self):
    """root_link parameter provides root when chain.root_link is None."""
    c = Chain()  # no root
    root = Link(42)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, root_link=root, ctx=ctx)
    self.assertIn('Chain(42)', result)

  def test_stringify_nested_indentation_level_2(self):
    """nest_lvl=2 produces deeper indentation."""
    c = Chain(sync_fn)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=2, ctx=ctx)
    # 2 levels = 8 spaces indentation
    self.assertIn('        ', result)

  def test_stringify_extra_links_with_source_link(self):
    """extra_links containing source_link gets the arrow marker."""
    c = Chain(sync_fn)
    extra = Link(raise_fn)
    ctx = _Ctx(source_link=extra, link_temp_args=None)
    result = _stringify_chain(
      c, nest_lvl=0, ctx=ctx,
      extra_links=[(extra, 'iterate')],
    )
    self.assertIn('<----', result)
    self.assertTrue(ctx.found)

  def test_stringify_chain_with_do_link(self):
    """do links display with .do() method name."""
    c = Chain(1).do(sync_fn)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('.do(sync_fn)', result)


class TestGetLinkNameAllOps(unittest.TestCase):

  def test_then_link(self):
    link = Link(sync_fn)
    self.assertEqual(_get_link_name(link), 'then')

  def test_do_link(self):
    link = Link(sync_fn, ignore_result=True)
    self.assertEqual(_get_link_name(link), 'do')

  def test_map_link(self):
    inner = Link(sync_fn)
    wrapper = _make_foreach(inner, ignore_result=False)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'map')

  def test_foreach_link(self):
    inner = Link(sync_fn)
    wrapper = _make_foreach(inner, ignore_result=True)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'foreach')

  def test_filter_link(self):
    inner = Link(sync_fn)
    wrapper = _make_filter(inner)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'filter')

  def test_gather_link(self):
    wrapper = _make_gather((sync_fn,))
    link = Link(wrapper)
    self.assertEqual(_get_link_name(link), 'gather')

  def test_with_link(self):
    inner = Link(sync_fn)
    wrapper = _make_with(inner, ignore_result=False)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'with_')

  def test_with_do_link(self):
    inner = Link(sync_fn)
    wrapper = _make_with(inner, ignore_result=True)
    link = Link(wrapper, original_value=inner)
    self.assertEqual(_get_link_name(link), 'with_do')


class TestFormatLinkWithGatherOp(unittest.TestCase):

  def test_format_link_gather_shows_fns(self):
    """_format_link with a gather operation shows function names."""
    def alpha(x):
      return x

    def beta(x):
      return x

    c = Chain(1).gather(alpha, beta)
    ctx = _Ctx(source_link=None, link_temp_args=None)
    result = _stringify_chain(c, nest_lvl=0, ctx=ctx)
    self.assertIn('alpha', result)
    self.assertIn('beta', result)


class TestFormatCallArgsEdgeCases(unittest.TestCase):

  def test_none_value_in_args(self):
    """None as an argument displays as 'None'."""
    result = _format_call_args((None,), None)
    self.assertEqual(result, 'None')

  def test_chain_in_args(self):
    """Chain object in args displays as 'Chain'."""
    c = Chain()
    result = _format_call_args((c,), None)
    self.assertEqual(result, 'Chain')

  def test_partial_in_args(self):
    """partial() in args shows wrapped function name."""
    result = _format_call_args((partial_fn,), None)
    self.assertIn('partial(add)', result)

  def test_boolean_kwargs(self):
    """Boolean kwarg values display correctly."""
    result = _format_call_args(None, {'debug': True, 'verbose': False})
    self.assertIn('debug=True', result)
    self.assertIn('verbose=False', result)


if __name__ == '__main__':
  unittest.main()
