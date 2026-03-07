"""Tests for _set_link_temp_args lifecycle and __quent_link_temp_args__ behavior.

_set_link_temp_args attaches debug info to exceptions, keyed by link identity.
_modify_traceback reads and then deletes __quent_link_temp_args__ when building
the chain visualization for the outermost chain. Chain-level tests verify
the temp args are set (and consumed for traceback rendering) by catching the
exception directly and formatting the traceback.
"""
from __future__ import annotations

import asyncio
import traceback
import unittest
from unittest import IsolatedAsyncioTestCase

from quent import Chain
from quent._core import Link, _set_link_temp_args, Null
from helpers import SyncCM, AsyncCM


def _capture_tb(fn):
  """Call fn(), capture exception, return formatted traceback string."""
  try:
    fn()
    return None
  except BaseException as exc:
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


async def _capture_tb_async(coro):
  """Await coro, capture exception, return formatted traceback string."""
  try:
    await coro
    return None
  except BaseException as exc:
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


# ---------------------------------------------------------------------------
# Direct unit tests for _set_link_temp_args
# ---------------------------------------------------------------------------

class TestTempArgsLifecycle(unittest.TestCase):

  def test_temp_args_set_on_map_exception(self):
    """Map sets item/index; verify via traceback string."""
    def _boom(x):
      raise ValueError('map error')

    tb_str = _capture_tb(lambda: Chain([10, 20, 30]).map(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('item=10', tb_str)
    self.assertIn('index=0', tb_str)

  def test_temp_args_set_on_filter_exception(self):
    """Filter sets item/index; verify via traceback string."""
    def _boom(x):
      raise ValueError('filter error')

    tb_str = _capture_tb(lambda: Chain([10, 20]).filter(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('item=10', tb_str)
    self.assertIn('index=0', tb_str)

  def test_temp_args_set_on_with_exception(self):
    """with_ sets ctx; verify via traceback string."""
    def _boom(ctx):
      raise ValueError('with error')

    tb_str = _capture_tb(lambda: Chain(SyncCM()).with_(_boom).run())
    self.assertIsNotNone(tb_str)
    # _get_obj_name wraps string values in quotes for display.
    self.assertIn("ctx='ctx_value'", tb_str)

  def test_temp_args_set_on_chain_exception(self):
    """Chain sets current_value; verify via traceback string."""
    def _boom(x):
      raise ValueError('chain error')

    tb_str = _capture_tb(lambda: Chain(42).then(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('current_value=42', tb_str)

  def test_temp_args_keyed_by_link_id(self):
    """Different links get different entries keyed by id(link)."""
    link1 = Link(lambda: None)
    link2 = Link(lambda: None)
    exc = ValueError('test')
    _set_link_temp_args(exc, link1, a=1)
    _set_link_temp_args(exc, link2, b=2)
    self.assertIn(id(link1), exc.__quent_link_temp_args__)
    self.assertIn(id(link2), exc.__quent_link_temp_args__)
    self.assertEqual(exc.__quent_link_temp_args__[id(link1)], {'a': 1})
    self.assertEqual(exc.__quent_link_temp_args__[id(link2)], {'b': 2})

  def test_temp_args_not_cleaned_after_handling(self):
    """Except_ handler can handle the error; temp args were consumed for traceback."""
    def _handler(rv, exc):
      return 'handled'

    def _boom(x):
      raise ValueError('chain error')

    result = Chain(42).then(_boom).except_(_handler).run()
    self.assertEqual(result, 'handled')

  def test_temp_args_accumulate_through_nesting(self):
    """Nested chains add temp args visible in traceback visualization."""
    def _inner_boom(x):
      raise ValueError('nested')

    inner = Chain().then(_inner_boom)
    tb_str = _capture_tb(lambda: Chain(42).then(inner).run())
    self.assertIsNotNone(tb_str)
    # The inner chain's failing link is marked with the arrow.
    self.assertIn('_inner_boom', tb_str)
    self.assertIn('<----', tb_str)

  def test_temp_args_first_write_preserves(self):
    """Multiple links writing to the same exception preserve all entries."""
    exc = ValueError('test')
    link1 = Link(lambda: None)
    link2 = Link(lambda: None)
    _set_link_temp_args(exc, link1, first='alpha')
    _set_link_temp_args(exc, link2, second='beta')
    self.assertEqual(len(exc.__quent_link_temp_args__), 2)
    self.assertEqual(exc.__quent_link_temp_args__[id(link1)], {'first': 'alpha'})
    self.assertEqual(exc.__quent_link_temp_args__[id(link2)], {'second': 'beta'})

  def test_temp_args_hasattr_check(self):
    """First call to _set_link_temp_args creates the dict on the exception."""
    exc = ValueError('test')
    self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))
    link = Link(lambda: None)
    _set_link_temp_args(exc, link, key='value')
    self.assertTrue(hasattr(exc, '__quent_link_temp_args__'))
    self.assertIsInstance(exc.__quent_link_temp_args__, dict)

  def test_temp_args_with_empty_kwargs(self):
    """_set_link_temp_args with no extra kwargs stores an empty dict."""
    exc = ValueError('test')
    link = Link(lambda: None)
    _set_link_temp_args(exc, link)
    self.assertEqual(exc.__quent_link_temp_args__[id(link)], {})

  def test_temp_args_overwrite_same_link(self):
    """Calling _set_link_temp_args twice for the same link overwrites."""
    exc = ValueError('test')
    link = Link(lambda: None)
    _set_link_temp_args(exc, link, a=1)
    _set_link_temp_args(exc, link, b=2)
    self.assertEqual(exc.__quent_link_temp_args__[id(link)], {'b': 2})

  def test_temp_args_with_non_exception_object(self):
    """_set_link_temp_args works with any object that supports attribute assignment."""
    class FakeExc:
      pass
    obj = FakeExc()
    link = Link(lambda: None)
    _set_link_temp_args(obj, link, data='test')
    self.assertTrue(hasattr(obj, '__quent_link_temp_args__'))
    self.assertEqual(obj.__quent_link_temp_args__[id(link)], {'data': 'test'})

  def test_temp_args_with_large_values(self):
    """_set_link_temp_args can store large values without issue."""
    exc = ValueError('test')
    link = Link(lambda: None)
    large_list = list(range(10000))
    _set_link_temp_args(exc, link, big=large_list)
    self.assertIs(exc.__quent_link_temp_args__[id(link)]['big'], large_list)

  def test_temp_args_map_at_second_item(self):
    """Map exception at index 1 records correct item/index in traceback."""
    call_count = 0

    def _boom_at_1(x):
      nonlocal call_count
      if call_count == 1:
        raise ValueError('at index 1')
      call_count += 1
      return x

    tb_str = _capture_tb(lambda: Chain([10, 20, 30]).map(_boom_at_1).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('item=20', tb_str)
    self.assertIn('index=1', tb_str)

  def test_temp_args_filter_at_second_item(self):
    """Filter exception at index 1 records correct item/index in traceback."""
    call_count = 0

    def _boom_at_1(x):
      nonlocal call_count
      if call_count == 1:
        raise ValueError('at index 1')
      call_count += 1
      return True

    tb_str = _capture_tb(lambda: Chain([10, 20, 30]).filter(_boom_at_1).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('item=20', tb_str)
    self.assertIn('index=1', tb_str)

  def test_temp_args_consumed_by_modify_traceback(self):
    """After _modify_traceback processes, __quent_link_temp_args__ is deleted."""
    def _boom(x):
      raise ValueError('consumed')

    with self.assertRaises(ValueError) as ctx:
      Chain(42).then(_boom).run()
    exc = ctx.exception
    self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))

  def test_temp_args_many_links_on_same_exc(self):
    """Many links can store temp args on the same exception object."""
    exc = ValueError('multi')
    links = [Link(lambda: None) for _ in range(50)]
    for i, link in enumerate(links):
      _set_link_temp_args(exc, link, idx=i, data=f'val_{i}')
    self.assertEqual(len(exc.__quent_link_temp_args__), 50)
    for i, link in enumerate(links):
      entry = exc.__quent_link_temp_args__[id(link)]
      self.assertEqual(entry['idx'], i)
      self.assertEqual(entry['data'], f'val_{i}')

  def test_temp_args_integer_current_value(self):
    """current_value with integer appears in traceback."""
    def _boom(x):
      raise ValueError('int cv')

    tb_str = _capture_tb(lambda: Chain(99).then(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('current_value=99', tb_str)

  def test_temp_args_string_current_value(self):
    """current_value with string appears in traceback."""
    def _boom(x):
      raise ValueError('str cv')

    tb_str = _capture_tb(lambda: Chain('hello').then(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn("current_value='hello'", tb_str)


# ---------------------------------------------------------------------------
# Async temp args tests
# ---------------------------------------------------------------------------

class TestTempArgsAsync(IsolatedAsyncioTestCase):

  async def test_async_temp_args_set(self):
    """Async path sets temp args; verify via traceback string."""
    async def _boom(x):
      raise ValueError('async error')

    tb_str = await _capture_tb_async(Chain(42).then(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('current_value=42', tb_str)

  async def test_async_temp_args_map(self):
    """Async map sets item/index; verify via traceback string."""
    async def _boom(x):
      raise ValueError('async map error')

    tb_str = await _capture_tb_async(Chain([10, 20]).map(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('item=10', tb_str)
    self.assertIn('index=0', tb_str)

  async def test_async_temp_args_filter(self):
    """Async filter sets item/index; verify via traceback string."""
    async def _boom(x):
      raise ValueError('async filter error')

    tb_str = await _capture_tb_async(Chain([10, 20]).filter(_boom).run())
    self.assertIsNotNone(tb_str)
    self.assertIn('item=10', tb_str)
    self.assertIn('index=0', tb_str)

  async def test_async_temp_args_with(self):
    """Async with_ sets ctx; verify via traceback string."""
    async def _boom(ctx):
      raise ValueError('async with error')

    tb_str = await _capture_tb_async(Chain(AsyncCM()).with_(_boom).run())
    self.assertIsNotNone(tb_str)
    # _get_obj_name wraps string values in quotes for display.
    self.assertIn("ctx='ctx_value'", tb_str)

  async def test_async_temp_args_consumed(self):
    """After async _modify_traceback, __quent_link_temp_args__ is deleted."""
    async def _boom(x):
      raise ValueError('consumed async')

    try:
      await Chain(42).then(_boom).run()
    except ValueError as exc:
      self.assertFalse(hasattr(exc, '__quent_link_temp_args__'))


if __name__ == '__main__':
  unittest.main()
