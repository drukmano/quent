"""Tests for fixes M8, M9, M13, M14.

M8:  Link.__init__ no longer crashes on objects with broken __getattr__.
M9:  Link.__init__ no longer crashes on frozen objects with _is_chain=True.
M13: Stale coroutine no longer returned when async CM suppresses body exception.
M14: ExceptionGroup sub-exceptions cleaned (Python 3.11+).
"""
from __future__ import annotations

import sys
import unittest

from quent import Chain, Null, QuentException
from quent._core import Link


# ---------------------------------------------------------------------------
# M8 fixtures: objects whose __getattr__ raises non-AttributeError exceptions
# ---------------------------------------------------------------------------

class BrokenGetattr_ValueError:
  def __getattr__(self, name):
    raise ValueError(f'Cannot access {name}')


class BrokenGetattr_RuntimeError:
  def __getattr__(self, name):
    raise RuntimeError(f'Cannot access {name}')


class BrokenGetattr_TypeError:
  def __getattr__(self, name):
    raise TypeError(f'Cannot access {name}')


class BrokenGetattr_OSError:
  """OSError from __getattr__."""
  def __getattr__(self, name):
    raise OSError(f'Cannot access {name}')


# ---------------------------------------------------------------------------
# M9 fixtures: frozen-like objects with _is_chain=True but immutable
# ---------------------------------------------------------------------------

class FrozenChainLike:
  __slots__ = ()
  _is_chain = True


class ReadOnlyChainLike:
  _is_chain = True

  def __setattr__(self, name, value):
    raise AttributeError('read only')


class ReadOnlyChainLike_TypeError:
  _is_chain = True

  def __setattr__(self, name, value):
    raise TypeError('frozen object')


# ---------------------------------------------------------------------------
# M13 fixtures: async context managers that suppress body exceptions
# ---------------------------------------------------------------------------

class AsyncCMSuppresses:
  async def __aenter__(self):
    return 'ctx'

  async def __aexit__(self, *args):
    return True  # suppress all exceptions


class AsyncCMNoSuppress:
  async def __aenter__(self):
    return 'ctx'

  async def __aexit__(self, *args):
    return False


# =========================================================================
# M8: Link.__init__ tolerates broken __getattr__
# =========================================================================

class TestM8_BrokenGetattr(unittest.TestCase):

  def test_value_error_from_getattr(self):
    obj = BrokenGetattr_ValueError()
    link = Link(obj)
    self.assertFalse(link.is_chain)
    self.assertIs(link.v, obj)

  def test_runtime_error_from_getattr(self):
    obj = BrokenGetattr_RuntimeError()
    link = Link(obj)
    self.assertFalse(link.is_chain)
    self.assertIs(link.v, obj)

  def test_type_error_from_getattr(self):
    obj = BrokenGetattr_TypeError()
    link = Link(obj)
    self.assertFalse(link.is_chain)
    self.assertIs(link.v, obj)

  def test_os_error_from_getattr(self):
    """OSError from __getattr__ should be caught."""
    obj = BrokenGetattr_OSError()
    link = Link(obj)
    self.assertFalse(link.is_chain)
    self.assertIs(link.v, obj)

  def test_link_fields_initialized_correctly(self):
    """All Link fields must be properly initialized despite broken __getattr__."""
    obj = BrokenGetattr_ValueError()
    link = Link(obj, args=(1, 2), kwargs={'a': 'b'}, ignore_result=True)
    self.assertFalse(link.is_chain)
    self.assertIs(link.v, obj)
    self.assertEqual(link.args, (1, 2))
    self.assertEqual(link.kwargs, {'a': 'b'})
    self.assertTrue(link.ignore_result)
    self.assertIsNone(link.next_link)

  def test_chain_with_broken_getattr_object_as_value(self):
    """Chain should be able to use a broken-getattr object as a plain value."""
    obj = BrokenGetattr_ValueError()
    result = Chain(obj).run()
    self.assertIs(result, obj)


# =========================================================================
# M9: Link.__init__ tolerates frozen objects with _is_chain=True
# =========================================================================

class TestM9_FrozenChainLike(unittest.TestCase):

  def test_frozen_slots_object_with_is_chain(self):
    """FrozenChainLike has _is_chain=True but __slots__=() so setting
    is_nested must be silently skipped."""
    obj = FrozenChainLike()
    link = Link(obj)
    self.assertTrue(link.is_chain)
    self.assertIs(link.v, obj)

  def test_readonly_setattr_raises_attribute_error(self):
    """ReadOnlyChainLike raises AttributeError on __setattr__;
    Link.__init__ should suppress it via contextlib.suppress."""
    obj = ReadOnlyChainLike()
    link = Link(obj)
    self.assertTrue(link.is_chain)
    self.assertIs(link.v, obj)

  def test_readonly_setattr_raises_type_error(self):
    """ReadOnlyChainLike_TypeError raises TypeError on __setattr__;
    Link.__init__ should suppress it via contextlib.suppress."""
    obj = ReadOnlyChainLike_TypeError()
    link = Link(obj)
    self.assertTrue(link.is_chain)
    self.assertIs(link.v, obj)

  def test_link_fields_initialized_for_chain_like(self):
    """Verify all Link fields are properly set when v is chain-like but frozen."""
    obj = FrozenChainLike()
    link = Link(obj, args=(1,), kwargs={'k': 'v'}, ignore_result=True)
    self.assertTrue(link.is_chain)
    self.assertEqual(link.args, (1,))
    self.assertEqual(link.kwargs, {'k': 'v'})
    self.assertTrue(link.ignore_result)
    self.assertIsNone(link.next_link)


# =========================================================================
# M13: Async CM suppresses body exception -> result is None, not stale coro
# =========================================================================

class TestM13_AsyncCMSuppressesBody(unittest.IsolatedAsyncioTestCase):

  async def test_suppressed_body_exception_returns_none(self):
    """When body raises and async CM suppresses, with_ result should be None."""
    async def body_that_raises(ctx):
      raise ValueError('body error')

    result = await Chain(AsyncCMSuppresses()).with_(body_that_raises).run()
    self.assertIsNone(result)

  async def test_suppressed_body_exception_with_do_returns_cm(self):
    """When body raises and async CM suppresses, with_do should return
    the CM object (outer_value), not the body result."""
    cm = AsyncCMSuppresses()

    async def body_that_raises(ctx):
      raise ValueError('body error')

    result = await Chain(cm).with_do(body_that_raises).run()
    self.assertIs(result, cm)

  async def test_suppressed_sync_body_exception_returns_none(self):
    """Same as above but the body raises synchronously (not a coroutine)."""
    def body_that_raises_sync(ctx):
      raise ValueError('sync body error')

    result = await Chain(AsyncCMSuppresses()).with_(body_that_raises_sync).run()
    self.assertIsNone(result)

  async def test_no_suppression_propagates_exception(self):
    """When body raises and CM does NOT suppress, exception propagates normally."""
    async def body_that_raises(ctx):
      raise ValueError('body error')

    with self.assertRaises(ValueError) as cm:
      await Chain(AsyncCMNoSuppress()).with_(body_that_raises).run()
    self.assertEqual(str(cm.exception), 'body error')

  async def test_successful_body_returns_result(self):
    """Normal case: body succeeds, result is the body return value."""
    async def body_ok(ctx):
      return 'body_result'

    result = await Chain(AsyncCMSuppresses()).with_(body_ok).run()
    self.assertEqual(result, 'body_result')

  async def test_successful_body_with_do_returns_cm(self):
    """with_do: body succeeds, result is the CM (ignore_result=True)."""
    cm = AsyncCMSuppresses()

    async def body_ok(ctx):
      return 'ignored'

    result = await Chain(cm).with_do(body_ok).run()
    self.assertIs(result, cm)

  async def test_suppressed_result_not_awaitable(self):
    """Regression: the result must be a plain value (None or CM), never an awaitable."""
    import inspect

    async def body_that_raises(ctx):
      raise ValueError('body error')

    result = await Chain(AsyncCMSuppresses()).with_(body_that_raises).run()
    self.assertFalse(inspect.isawaitable(result))

  async def test_chain_continues_after_suppression(self):
    """Links after with_() should see None as current_value when suppressed."""
    async def body_that_raises(ctx):
      raise ValueError('body error')

    result = await (
      Chain(AsyncCMSuppresses())
      .with_(body_that_raises)
      .then(lambda v: f'got:{v}')
      .run()
    )
    self.assertEqual(result, 'got:None')


# =========================================================================
# M14: ExceptionGroup sub-exceptions cleaned (Python 3.11+)
# =========================================================================

@unittest.skipIf(sys.version_info < (3, 11), 'ExceptionGroup requires Python 3.11+')
class TestM14_ExceptionGroupCleaning(unittest.TestCase):

  def test_clean_chained_exceptions_traverses_exception_group(self):
    """_clean_chained_exceptions should traverse ExceptionGroup.exceptions."""
    from quent._traceback import _clean_chained_exceptions

    sub1 = ValueError('sub1')
    sub2 = TypeError('sub2')
    eg = ExceptionGroup('test', [sub1, sub2])

    seen: set[int] = set()
    _clean_chained_exceptions(eg, seen)

    # All 3 exceptions should have been visited
    self.assertIn(id(eg), seen)
    self.assertIn(id(sub1), seen)
    self.assertIn(id(sub2), seen)

  def test_nested_exception_groups(self):
    """Nested ExceptionGroups should be fully traversed."""
    from quent._traceback import _clean_chained_exceptions

    inner_sub = RuntimeError('inner')
    inner_eg = ExceptionGroup('inner', [inner_sub])
    outer_sub = ValueError('outer')
    outer_eg = ExceptionGroup('outer', [inner_eg, outer_sub])

    seen: set[int] = set()
    _clean_chained_exceptions(outer_eg, seen)

    self.assertIn(id(outer_eg), seen)
    self.assertIn(id(inner_eg), seen)
    self.assertIn(id(inner_sub), seen)
    self.assertIn(id(outer_sub), seen)

  def test_exception_group_with_chained_sub_exceptions(self):
    """Sub-exceptions with __cause__/__context__ should also be traversed."""
    from quent._traceback import _clean_chained_exceptions

    cause = OSError('cause')
    sub1 = ValueError('sub1')
    sub1.__cause__ = cause
    eg = ExceptionGroup('test', [sub1])

    seen: set[int] = set()
    _clean_chained_exceptions(eg, seen)

    self.assertIn(id(eg), seen)
    self.assertIn(id(sub1), seen)
    self.assertIn(id(cause), seen)

  def test_does_not_crash_on_empty_exception_group(self):
    """An ExceptionGroup with no sub-exceptions should be handled gracefully."""
    from quent._traceback import _clean_chained_exceptions

    # ExceptionGroup requires at least one exception, but test the
    # edge case by using a regular exception with an empty .exceptions attr
    class FakeEG(Exception):
      exceptions = ()

    fake = FakeEG('empty')
    seen: set[int] = set()
    _clean_chained_exceptions(fake, seen)
    self.assertIn(id(fake), seen)

  def test_cycle_detection_prevents_infinite_loop(self):
    """Circular __cause__ chains should be handled via the 'seen' set."""
    from quent._traceback import _clean_chained_exceptions

    exc1 = ValueError('exc1')
    exc2 = TypeError('exc2')
    exc1.__cause__ = exc2
    exc2.__cause__ = exc1  # cycle

    seen: set[int] = set()
    # Must terminate without infinite recursion/loop
    _clean_chained_exceptions(exc1, seen)
    self.assertIn(id(exc1), seen)
    self.assertIn(id(exc2), seen)


if __name__ == '__main__':
  unittest.main()
