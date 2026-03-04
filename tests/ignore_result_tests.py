"""Tests for ignore_result=True code paths in _iteration.pxi and _control_flow.pxi.

Covers the internal ignore_result flag on _Foreach, _With, and
_Generator / sync_generator / async_generator. These paths are not reachable
through the public Chain API (which always passes ignore_result=False), so
tests construct internal objects directly.

Coverage targets (line references are to the .pxi source files):
  _iteration.pxi:
    - _Foreach.__call__         lines 57-58   (sync ignore_result branch)
    - _foreach_to_async         lines 84-85   (async continuation branch)
    - _foreach_full_async       lines 113-114 (fully async branch)

  _control_flow.pxi:
    - _With.__call__            lines 94-95   (sync ignore_result branch)
    - _with_to_async            lines 129-130 (async continuation branch)
    - _with_full_async          lines 150-151 (fully async branch)
    - sync_generator            lines 166-167 (sync generator ignore_result)
    - async_generator           lines 193-194 (async generator aiter path)
    - async_generator           lines 205-206 (async generator sync iter path)
"""
from unittest import IsolatedAsyncioTestCase
from quent.quent import (
  _Foreach, _With, _Generator,
  Link, sync_generator, async_generator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AsyncIter:
  """Async iterable that wraps a list of items."""
  def __init__(self, items):
    self._items = list(items)

  def __aiter__(self):
    self._iter = iter(self._items)
    return self

  async def __anext__(self):
    try:
      return next(self._iter)
    except StopIteration:
      raise StopAsyncIteration


class SimpleCM:
  """Sync context manager that yields a configurable value."""
  def __init__(self, value='ctx'):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return False


class AsyncCM:
  """Async context manager that yields a configurable value."""
  def __init__(self, value='actx'):
    self.value = value
    self.entered = False
    self.exited = False

  async def __aenter__(self):
    self.entered = True
    return self.value

  async def __aexit__(self, *args):
    self.exited = True
    return False


def _make_link(fn):
  """Create a Link for error context in internal objects."""
  return Link(fn, (), {})


# ---------------------------------------------------------------------------
# ForeachIgnoreResultTests
# ---------------------------------------------------------------------------
class ForeachIgnoreResultTests(IsolatedAsyncioTestCase):
  """Test _Foreach with ignore_result=True across all three execution paths."""

  async def test_sync_foreach_ignore_result_preserves_elements(self):
    """_Foreach.__call__ sync path: when ignore_result=True, appends original
    elements instead of mapped results (lines 57-58 of _iteration.pxi)."""
    fn = lambda x: x * 100
    link = _make_link(fn)
    fe = _Foreach(fn, True, link)
    result = fe([1, 2, 3])
    self.assertEqual(result, [1, 2, 3])

  async def test_sync_foreach_ignore_result_false_returns_mapped(self):
    """Baseline: ignore_result=False returns fn results."""
    fn = lambda x: x * 100
    link = _make_link(fn)
    fe = _Foreach(fn, False, link)
    result = fe([1, 2, 3])
    self.assertEqual(result, [100, 200, 300])

  async def test_async_continuation_foreach_ignore_result(self):
    """_foreach_to_async: when fn returns a coroutine mid-iteration,
    the async continuation still appends original elements when
    ignore_result=True (lines 84-85 of _iteration.pxi)."""
    async def async_fn(x):
      return x * 100

    link = _make_link(async_fn)
    fe = _Foreach(async_fn, True, link)
    result = fe([10, 20, 30])
    # fn is async so __call__ returns a coroutine
    result = await result
    self.assertEqual(result, [10, 20, 30])

  async def test_async_continuation_foreach_ignore_result_false(self):
    """Baseline: async continuation with ignore_result=False returns fn results."""
    async def async_fn(x):
      return x * 100

    link = _make_link(async_fn)
    fe = _Foreach(async_fn, False, link)
    result = await fe([10, 20, 30])
    self.assertEqual(result, [1000, 2000, 3000])

  async def test_full_async_foreach_ignore_result(self):
    """_foreach_full_async: when input is an async iterable,
    appends original elements when ignore_result=True
    (lines 113-114 of _iteration.pxi)."""
    fn = lambda x: x * 100
    link = _make_link(fn)
    fe = _Foreach(fn, True, link)
    result = await fe(AsyncIter([5, 6, 7]))
    self.assertEqual(result, [5, 6, 7])

  async def test_full_async_foreach_ignore_result_false(self):
    """Baseline: fully async foreach with ignore_result=False returns fn results."""
    fn = lambda x: x * 100
    link = _make_link(fn)
    fe = _Foreach(fn, False, link)
    result = await fe(AsyncIter([5, 6, 7]))
    self.assertEqual(result, [500, 600, 700])

  async def test_full_async_foreach_async_fn_ignore_result(self):
    """_foreach_full_async with async fn: coroutine results are awaited but
    original elements are still preserved when ignore_result=True."""
    async def async_fn(x):
      return x * 100

    link = _make_link(async_fn)
    fe = _Foreach(async_fn, True, link)
    result = await fe(AsyncIter([5, 6, 7]))
    self.assertEqual(result, [5, 6, 7])

  async def test_sync_foreach_empty_iterable(self):
    """ignore_result=True with empty iterable returns empty list."""
    fn = lambda x: x * 100
    link = _make_link(fn)
    fe = _Foreach(fn, True, link)
    result = fe([])
    self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# WithIgnoreResultTests
# ---------------------------------------------------------------------------
class WithIgnoreResultTests(IsolatedAsyncioTestCase):
  """Test _With with ignore_result=True across all three execution paths."""

  async def test_sync_with_ignore_result_returns_outer_value(self):
    """_With.__call__ sync path: when ignore_result=True, returns the original
    context manager (outer_value) instead of the body result
    (lines 94-95 of _control_flow.pxi)."""
    body_link = Link(lambda ctx: ctx.upper(), (), {})
    w = _With(body_link, True, (), {})
    cm = SimpleCM('hello')
    result = w(cm)
    self.assertIs(result, cm)

  async def test_sync_with_ignore_result_false_returns_body_result(self):
    """Baseline: ignore_result=False returns the body fn result."""
    body_link = Link(lambda ctx: ctx.upper(), (), {})
    w = _With(body_link, False, (), {})
    cm = SimpleCM('hello')
    result = w(cm)
    self.assertEqual(result, 'HELLO')

  async def test_sync_with_ignore_result_cm_lifecycle(self):
    """_With with ignore_result=True still enters and exits the CM."""
    body_link = Link(lambda ctx: ctx.upper(), (), {})
    w = _With(body_link, True, (), {})
    cm = SimpleCM('val')
    w(cm)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_async_continuation_with_ignore_result(self):
    """_with_to_async: when sync CM body returns a coroutine and
    ignore_result=True, returns outer_value after awaiting body
    (lines 129-130 of _control_flow.pxi)."""
    async def async_body(ctx):
      return ctx.upper()

    body_link = Link(async_body, (), {})
    w = _With(body_link, True, (), {})
    cm = SimpleCM('hello')
    result = await w(cm)
    self.assertIs(result, cm)

  async def test_async_continuation_with_ignore_result_false(self):
    """Baseline: async continuation with ignore_result=False returns body result."""
    async def async_body(ctx):
      return ctx.upper()

    body_link = Link(async_body, (), {})
    w = _With(body_link, False, (), {})
    cm = SimpleCM('hello')
    result = await w(cm)
    self.assertEqual(result, 'HELLO')

  async def test_async_continuation_with_ignore_result_cm_lifecycle(self):
    """Sync CM with async body and ignore_result=True still enters and exits."""
    async def async_body(ctx):
      return 'whatever'

    body_link = Link(async_body, (), {})
    w = _With(body_link, True, (), {})
    cm = SimpleCM('val')
    await w(cm)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_full_async_with_ignore_result(self):
    """_with_full_async: when CM has __aenter__, ignore_result=True returns
    the async CM (outer_value) instead of the body result
    (lines 150-151 of _control_flow.pxi)."""
    body_link = Link(lambda ctx: ctx.upper(), (), {})
    w = _With(body_link, True, (), {})
    cm = AsyncCM('hello')
    result = await w(cm)
    self.assertIs(result, cm)

  async def test_full_async_with_ignore_result_false(self):
    """Baseline: fully async with and ignore_result=False returns body result."""
    body_link = Link(lambda ctx: ctx.upper(), (), {})
    w = _With(body_link, False, (), {})
    cm = AsyncCM('hello')
    result = await w(cm)
    self.assertEqual(result, 'HELLO')

  async def test_full_async_with_ignore_result_cm_lifecycle(self):
    """Async CM with ignore_result=True still enters and exits."""
    body_link = Link(lambda ctx: 'whatever', (), {})
    w = _With(body_link, True, (), {})
    cm = AsyncCM('val')
    await w(cm)
    self.assertTrue(cm.entered)
    self.assertTrue(cm.exited)

  async def test_full_async_with_async_body_ignore_result(self):
    """_with_full_async with async body fn: coroutine is awaited but
    outer_value is returned when ignore_result=True."""
    async def async_body(ctx):
      return ctx.upper()

    body_link = Link(async_body, (), {})
    w = _With(body_link, True, (), {})
    cm = AsyncCM('hello')
    result = await w(cm)
    self.assertIs(result, cm)


# ---------------------------------------------------------------------------
# GeneratorIgnoreResultTests
# ---------------------------------------------------------------------------
class GeneratorIgnoreResultTests(IsolatedAsyncioTestCase):
  """Test sync_generator and async_generator with ignore_result=True."""

  async def test_sync_generator_ignore_result(self):
    """sync_generator: when ignore_result=True and fn is provided, yields
    original elements instead of fn results
    (lines 166-167 of _control_flow.pxi)."""
    def get_iter(*args):
      return [1, 2, 3]

    result = list(sync_generator(get_iter, (), lambda x: x * 100, True))
    self.assertEqual(result, [1, 2, 3])

  async def test_sync_generator_ignore_result_false(self):
    """Baseline: sync_generator with ignore_result=False returns fn results."""
    def get_iter(*args):
      return [1, 2, 3]

    result = list(sync_generator(get_iter, (), lambda x: x * 100, False))
    self.assertEqual(result, [100, 200, 300])

  async def test_sync_generator_ignore_result_no_fn(self):
    """sync_generator with fn=None yields elements regardless of ignore_result."""
    def get_iter(*args):
      return [10, 20, 30]

    result = list(sync_generator(get_iter, (), None, True))
    self.assertEqual(result, [10, 20, 30])

  async def test_async_generator_sync_iter_ignore_result(self):
    """async_generator with sync iterable: when ignore_result=True, yields
    original elements (lines 205-206 of _control_flow.pxi, the 'else' branch)."""
    def get_iter(*args):
      return [1, 2, 3]

    result = []
    async for item in async_generator(get_iter, (), lambda x: x * 100, True):
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_async_generator_sync_iter_ignore_result_false(self):
    """Baseline: async_generator sync iter with ignore_result=False."""
    def get_iter(*args):
      return [1, 2, 3]

    result = []
    async for item in async_generator(get_iter, (), lambda x: x * 100, False):
      result.append(item)
    self.assertEqual(result, [100, 200, 300])

  async def test_async_generator_sync_iter_async_fn_ignore_result(self):
    """async_generator with sync iter and async fn: coroutine is awaited
    but original element is yielded when ignore_result=True."""
    async def async_fn(x):
      return x * 100

    def get_iter(*args):
      return [1, 2, 3]

    result = []
    async for item in async_generator(get_iter, (), async_fn, True):
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_async_generator_aiter_ignore_result(self):
    """async_generator with async iterable: when ignore_result=True, yields
    original elements (lines 193-194 of _control_flow.pxi, the 'if is_aiter' branch)."""
    def get_aiter(*args):
      return AsyncIter([1, 2, 3])

    result = []
    async for item in async_generator(get_aiter, (), lambda x: x * 100, True):
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_async_generator_aiter_ignore_result_false(self):
    """Baseline: async_generator aiter with ignore_result=False."""
    def get_aiter(*args):
      return AsyncIter([1, 2, 3])

    result = []
    async for item in async_generator(get_aiter, (), lambda x: x * 100, False):
      result.append(item)
    self.assertEqual(result, [100, 200, 300])

  async def test_async_generator_aiter_async_fn_ignore_result(self):
    """async_generator with async iterable and async fn: coroutine is awaited
    but original element is yielded when ignore_result=True."""
    async def async_fn(x):
      return x * 100

    def get_aiter(*args):
      return AsyncIter([1, 2, 3])

    result = []
    async for item in async_generator(get_aiter, (), async_fn, True):
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_async_generator_aiter_no_fn(self):
    """async_generator with fn=None yields elements regardless of ignore_result."""
    def get_aiter(*args):
      return AsyncIter([10, 20, 30])

    result = []
    async for item in async_generator(get_aiter, (), None, True):
      result.append(item)
    self.assertEqual(result, [10, 20, 30])

  async def test_async_generator_coroutine_iterator_getter(self):
    """async_generator where iterator_getter returns a coroutine that
    resolves to an iterable, with ignore_result=True."""
    async def async_getter(*args):
      return [4, 5, 6]

    result = []
    async for item in async_generator(async_getter, (), lambda x: x * 10, True):
      result.append(item)
    self.assertEqual(result, [4, 5, 6])


# ---------------------------------------------------------------------------
# GeneratorObjectIgnoreResultTests
# ---------------------------------------------------------------------------
class GeneratorObjectIgnoreResultTests(IsolatedAsyncioTestCase):
  """Test _Generator wrapper with _ignore_result=True."""

  async def test_generator_sync_iteration_ignore_result(self):
    """_Generator.__iter__ with _ignore_result=True yields original elements."""
    def fake_run(v, args, kwargs, invoked):
      return [1, 2, 3]

    gen = _Generator(fake_run, lambda x: x * 10, _ignore_result=True)
    result = list(gen)
    self.assertEqual(result, [1, 2, 3])

  async def test_generator_async_iteration_ignore_result(self):
    """_Generator.__aiter__ with _ignore_result=True yields original elements."""
    def fake_run(v, args, kwargs, invoked):
      return [1, 2, 3]

    gen = _Generator(fake_run, lambda x: x * 10, _ignore_result=True)
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [1, 2, 3])

  async def test_generator_call_preserves_ignore_result(self):
    """Calling a _Generator returns a new instance that preserves _ignore_result."""
    def fake_run(v, args, kwargs, invoked):
      return [10, 20]

    gen = _Generator(fake_run, lambda x: x * 5, _ignore_result=True)
    nested = gen()
    result = list(nested)
    self.assertEqual(result, [10, 20])

  async def test_generator_reuse_with_ignore_result(self):
    """A _Generator with _ignore_result=True can be iterated multiple times."""
    def fake_run(v, args, kwargs, invoked):
      return [7, 8, 9]

    gen = _Generator(fake_run, lambda x: x + 1000, _ignore_result=True)
    r1 = list(gen)
    r2 = list(gen)
    self.assertEqual(r1, [7, 8, 9])
    self.assertEqual(r2, [7, 8, 9])

  async def test_generator_async_iter_source_ignore_result(self):
    """_Generator.__aiter__ with async iterable source and _ignore_result=True."""
    def fake_run(v, args, kwargs, invoked):
      return AsyncIter([100, 200, 300])

    gen = _Generator(fake_run, lambda x: x // 10, _ignore_result=True)
    result = []
    async for item in gen:
      result.append(item)
    self.assertEqual(result, [100, 200, 300])

  async def test_generator_empty_source_ignore_result(self):
    """_Generator with empty iterable and _ignore_result=True yields nothing."""
    def fake_run(v, args, kwargs, invoked):
      return []

    gen = _Generator(fake_run, lambda x: x * 10, _ignore_result=True)
    result = list(gen)
    self.assertEqual(result, [])


if __name__ == '__main__':
  import unittest
  unittest.main()
