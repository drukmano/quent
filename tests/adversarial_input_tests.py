"""Tests for adversarial and edge-case inputs to the Chain API."""
from __future__ import annotations

import asyncio
import unittest

from quent import Chain, Null, QuentException
from quent._core import Link, _evaluate_value, _Return, _Break


class FakeChainNoRun:
  """Object with _is_chain=True but no _run method."""
  _is_chain = True


class BrokenInitException(Exception):
  """Exception whose __init__ raises another exception."""
  def __init__(self, *args, **kwargs):
    raise RuntimeError('broken __init__')


class BrokenStrException(Exception):
  """Exception whose __str__ raises."""
  def __str__(self):
    raise RuntimeError('broken __str__')


class BrokenReprException(Exception):
  """Exception whose __repr__ raises."""
  def __repr__(self):
    raise RuntimeError('broken __repr__')


class AlwaysRaisesGetattr:
  """Object with __getattr__ that always raises."""
  def __getattr__(self, name):
    raise RuntimeError(f'__getattr__ always raises for {name}')


class ReentrantCM:
  """Context manager whose __enter__ creates and runs a chain."""
  def __init__(self):
    self.inner_result = None

  def __enter__(self):
    self.inner_result = Chain(10).then(lambda x: x + 5).run()
    return self.inner_result

  def __exit__(self, *args):
    return False


class AdversarialInputTests(unittest.TestCase):

  def test_object_with_is_chain_true_but_no_run(self):
    """Object with _is_chain=True but no _run method should raise AttributeError."""
    fake = FakeChainNoRun()
    c = Chain().then(fake)
    with self.assertRaises(AttributeError):
      c.run()

  def test_object_with_is_chain_true_nested_evaluation(self):
    """Verify Link correctly identifies the fake chain via duck typing."""
    fake = FakeChainNoRun()
    link = Link(fake)
    self.assertTrue(link.is_chain)

  def test_callable_mutates_chain_during_execution(self):
    """Callable that calls chain.then() during execution should not crash."""
    c = Chain(1)
    def mutator(x):
      c.then(lambda y: y + 100)
      return x + 1
    c.then(mutator)
    # Behavior is undefined for unfrozen chains, but it should not crash.
    # The mutation appends a link after current execution, which may or may
    # not be reached depending on timing.
    result = c.run()
    self.assertIsNotNone(result)

  def test_exception_with_broken_init(self):
    """Exception whose __init__ raises should propagate as RuntimeError."""
    def raiser(_):
      raise BrokenInitException('test')
    c = Chain(1).then(raiser)
    # BrokenInitException.__init__ itself raises RuntimeError,
    # so we get RuntimeError, not BrokenInitException.
    with self.assertRaises(RuntimeError) as ctx:
      c.run()
    self.assertIn('broken __init__', str(ctx.exception))

  def test_exception_with_broken_str(self):
    """Exception whose __str__ raises should still propagate."""
    exc = BrokenStrException.__new__(BrokenStrException)
    Exception.__init__(exc, 'test')
    def raiser(_):
      raise exc
    c = Chain(1).then(raiser)
    with self.assertRaises(BrokenStrException):
      c.run()

  def test_exception_with_broken_repr(self):
    """Exception whose __repr__ raises should still propagate."""
    exc = BrokenReprException.__new__(BrokenReprException)
    Exception.__init__(exc, 'test')
    def raiser(_):
      raise exc
    c = Chain(1).then(raiser)
    with self.assertRaises(BrokenReprException):
      c.run()

  def test_object_with_getattr_always_raises(self):
    """Object whose __getattr__ always raises should be treated as non-callable value."""
    obj = AlwaysRaisesGetattr()
    # Link.__init__ wraps getattr(v, '_is_chain', False) in a try/except,
    # so it should set is_chain=False without crashing.
    link = Link(obj)
    self.assertFalse(link.is_chain)

  def test_object_with_getattr_always_raises_in_chain(self):
    """Object whose __getattr__ always raises passed to then() should pass through as value."""
    obj = AlwaysRaisesGetattr()
    # callable() checks the type's __call__, not the instance, so it returns
    # False without triggering __getattr__. The object passes through as a
    # non-callable value via the `return v` branch in _evaluate_value.
    c = Chain().then(obj)
    result = c.run()
    self.assertIs(result, obj)

  def test_deeply_nested_chains_50_levels(self):
    """50 levels of chain nesting should not stack overflow."""
    inner = Chain(1).then(lambda x: x + 1)
    for _ in range(49):
      outer = Chain().then(inner)
      inner = outer
    result = inner.run(0)
    # The value 0 is passed to the outermost chain, which delegates to
    # the next inner chain, and so on all the way down. The innermost
    # chain has root_link=1, but run(0) overrides it: temp Link(0) ->
    # first_link (lambda x: x + 1) -> 0 + 1 = 1. Each outer layer just
    # delegates to its inner chain with the same value, so the result is 1.
    self.assertEqual(result, 1)

  def test_chain_with_none_as_value(self):
    """Chain().then(None).run() should pass None through as a plain value."""
    result = Chain().then(None).run(5)
    self.assertIsNone(result)

  def test_chain_with_empty_string(self):
    """Chain('').run() should return empty string."""
    result = Chain('').run()
    self.assertEqual(result, '')

  def test_chain_with_zero(self):
    """Chain(0).run() should return 0."""
    result = Chain(0).run()
    self.assertEqual(result, 0)

  def test_chain_with_false(self):
    """Chain(False).run() should return False."""
    result = Chain(False).run()
    self.assertIs(result, False)

  def test_reentrant_context_manager(self):
    """Context manager whose __enter__ creates and runs a chain."""
    cm = ReentrantCM()
    result = Chain(cm).with_(lambda ctx: ctx + 100).run()
    self.assertEqual(cm.inner_result, 15)
    self.assertEqual(result, 115)

  def test_generator_as_chain_value_with_map(self):
    """Generator object passed to a chain should be iterable via map()."""
    def gen():
      yield 1
      yield 2
      yield 3
    result = Chain(gen()).map(lambda x: x * 2).run()
    self.assertEqual(result, [2, 4, 6])

  def test_callable_that_returns_itself(self):
    """Callable that returns itself should not cause infinite loop."""
    fn = lambda x: fn  # noqa: E731
    result = Chain(1).then(fn).run()
    # fn(1) returns fn itself. Chain just stores it as the current value.
    self.assertIs(result, fn)

  def test_map_rejects_non_callable(self):
    """map(42) should raise TypeError with proper message."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1, 2]).map(42)
    self.assertEqual(str(ctx.exception), 'map() requires a callable, got int')

  def test_foreach_rejects_non_callable(self):
    """foreach("string") should raise TypeError with proper message."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1, 2]).foreach('string')
    self.assertEqual(str(ctx.exception), 'foreach() requires a callable, got str')

  def test_filter_rejects_non_callable(self):
    """filter(None) should raise TypeError with proper message."""
    with self.assertRaises(TypeError) as ctx:
      Chain([1, 2]).filter(None)
    self.assertEqual(str(ctx.exception), 'filter() requires a callable, got NoneType')

  def test_gather_rejects_non_callable_argument(self):
    """gather(lambda x: x, 42) should raise TypeError with proper message."""
    with self.assertRaises(TypeError) as ctx:
      Chain(5).gather(lambda x: x, 42)
    self.assertEqual(str(ctx.exception), 'gather() requires all arguments to be callable, got int')


class AdversarialAsyncTests(unittest.IsolatedAsyncioTestCase):

  async def test_deeply_nested_chains_50_levels_async(self):
    """50 levels of async chain nesting should work."""
    inner = Chain(1).then(lambda x: x + 1)
    for _ in range(49):
      outer = Chain().then(inner)
      inner = outer
    # Using an async callable at the root to trigger async path
    async def async_root(x):
      return x
    c = Chain().then(async_root).then(inner)
    result = await c.run(0)
    # async_root(0) returns 0, then inner(0) delegates through all 50
    # nesting levels to the innermost chain: temp Link(0) -> lambda x: x + 1
    # -> 1. Each outer layer just passes through, so the result is 1.
    self.assertEqual(result, 1)


if __name__ == '__main__':
  unittest.main()
