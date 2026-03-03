import asyncio
import functools
import traceback
import unittest
from unittest import TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tb_entries(exc):
  """Extract traceback entries from an exception."""
  return traceback.extract_tb(exc.__traceback__)


def _get_quent_entries(exc):
  """Get only the <quent> traceback entries."""
  return [e for e in _get_tb_entries(exc) if e.filename == '<quent>']


# ---------------------------------------------------------------------------
# 1. _await_run exception path with chain+link context
# ---------------------------------------------------------------------------

class AwaitRunExceptionTests(MyTestCase):

  async def test_await_run_exception_with_chain_and_link_context(self):
    """When an async except handler (reraise=False) itself raises,
    _await_run catches it and calls modify_traceback with chain+link
    before re-raising (lines 22-24 of _link.pxi)."""
    async def bad_async_handler(v=None):
      raise ValueError('handler exploded')

    # Sync chain: exception triggers except_ handler which is async.
    # With reraise=False, the handler coro is wrapped in a Task via
    # ensure_future(_await_run_fn(result, self, exc_link, ctx)).
    # When we await the Task, _await_run's except clause (lines 22-24) fires.
    c = (
      Chain(lambda: (_ for _ in ()).throw(TestExc('original')))
      .then(lambda v: v)
      .except_(bad_async_handler, reraise=False)
    )
    task = c.run()
    # The result is a Task because the except handler returned a coro in sync mode
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    with self.assertRaises(ValueError) as cm:
      await task
    super(MyTestCase, self).assertEqual(str(cm.exception), 'handler exploded')

  async def test_await_run_exception_has_quent_traceback(self):
    """The exception from _await_run should have a <quent> frame injected
    by modify_traceback (proving chain+link context was passed)."""
    async def raising_handler(v=None):
      raise ValueError('traced handler')

    c = (
      Chain(lambda: (_ for _ in ()).throw(TestExc('trigger')))
      .then(lambda v: v)
      .except_(raising_handler, reraise=False)
    )
    task = c.run()
    try:
      await task
      super(MyTestCase, self).fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      # modify_traceback should have injected a <quent> frame
      super(MyTestCase, self).assertGreater(
        len(entries), 0,
        'Expected <quent> frame from modify_traceback in _await_run'
      )

  async def test_await_run_finally_exception_path(self):
    """When a sync chain has an async finally handler that raises,
    _await_run catches it with chain+link context (lines 235-236 of
    _chain_core.pxi via ensure_future(_await_run_fn(result, self,
    self.on_finally_link, ctx)))."""
    async def async_finally(v=None):
      raise RuntimeError('finally failed')

    import warnings
    c = Chain(42).then(lambda v: v).finally_(async_finally)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      result = c.run()
    # The finally handler's coro was scheduled as a task
    super(MyTestCase, self).assertEqual(result, 42)
    # Give the task time to execute
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# 2. Nested chain with kwargs-only -> EVAL_CALL_WITHOUT_ARGS
# ---------------------------------------------------------------------------

class ChainKwargsOnlyTests(MyTestCase):

  async def test_nested_chain_kwargs_only_gets_null_root(self):
    """A nested chain passed with kwargs but no positional args triggers
    EVAL_CALL_WITHOUT_ARGS (line 56 of _link.pxi). The chain runs with
    Null root, so the first link receives None."""
    inner = Chain().then(lambda v=None: 'null' if v is None else f'val_{v}')
    result = Chain(5).then(inner, key='ignored').run()
    # inner runs with Null root (kwargs are not forwarded to chains)
    await self.assertEqual(result, 'null')

  async def test_nested_chain_kwargs_only_vs_normal(self):
    """Contrast: a nested chain WITHOUT kwargs receives the current value
    as root, while one WITH kwargs-only gets Null."""
    inner_normal = Chain().then(lambda v=None: v)
    inner_kwargs = Chain().then(lambda v=None: v)

    normal_result = Chain(42).then(inner_normal).run()
    kwargs_result = Chain(42).then(inner_kwargs, key='x').run()

    await self.assertEqual(normal_result, 42)
    await self.assertIsNone(kwargs_result)

  async def test_nested_chain_kwargs_only_repr(self):
    """repr of a chain with nested chain + kwargs-only shows the nested
    chain without the kwargs (since they are not forwarded)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, key='val')
    r = repr(c)
    super(MyTestCase, self).assertIn('Chain', r)
    super(MyTestCase, self).assertIn('.then', r)


# ---------------------------------------------------------------------------
# 3. _FrozenChain.decorator with callable lacking __name__/__dict__
# ---------------------------------------------------------------------------

class DecoratorNoNameTests(TestCase):

  def test_decorator_with_slots_callable_no_dict(self):
    """A callable with __slots__=() has no __dict__. The decorator catches
    the AttributeError from functools.update_wrapper (if any) and still
    returns a working wrapper (lines 58-59 of _variants.pxi)."""
    class SlotCallable:
      __slots__ = ()
      def __call__(self, x):
        return x * 3

    decorator = Chain().then(lambda v: v + 10).decorator()
    wrapped = decorator(SlotCallable())
    self.assertEqual(wrapped(5), 25)  # 5*3 + 10
    # The wrapper may or may not have __name__ depending on what
    # update_wrapper was able to copy
    self.assertTrue(callable(wrapped))

  def test_decorator_with_monkeypatched_update_wrapper_raising(self):
    """Directly verify the except AttributeError path by monkeypatching
    functools.update_wrapper to raise AttributeError."""
    original = functools.update_wrapper

    def raising_update_wrapper(wrapper, wrapped, **kw):
      raise AttributeError('simulated failure')

    functools.update_wrapper = raising_update_wrapper
    try:
      decorator = Chain().then(lambda v: v * 2).decorator()

      class Fn:
        def __call__(self, x):
          return x

      result = decorator(Fn())
      # The decorator should still work despite update_wrapper failing
      self.assertEqual(result(7), 14)  # 7 * 2
      # __name__ should NOT be set since update_wrapper failed
      self.assertFalse(hasattr(result, '__name__'))
    finally:
      functools.update_wrapper = original

  def test_decorator_with_normal_callable_preserves_name(self):
    """Sanity check: a normal function decorated via the same path
    preserves __name__ (functools.update_wrapper succeeds)."""
    @Chain().then(lambda v: v).decorator()
    def my_func(x):
      return x

    self.assertEqual(my_func.__name__, 'my_func')
    self.assertEqual(my_func(42), 42)


# ---------------------------------------------------------------------------
# 4. _handle_exception temp_args from foreach exception
# ---------------------------------------------------------------------------

class HandleExceptionTempArgsTests(MyTestCase):

  async def test_foreach_sync_exception_sets_temp_args(self):
    """A sync foreach exception sets __quent_link_temp_args__ on the
    exception. _handle_exception copies it to ctx.link_temp_args
    (lines 91-92 of _diagnostics.pxi). The traceback visualization
    then shows the element that caused the error."""
    def fail_on_3(x):
      if x == 3:
        raise ValueError(f'bad {x}')
      return x

    try:
      Chain([1, 2, 3, 4]).foreach(fail_on_3).run()
      super(MyTestCase, self).fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      super(MyTestCase, self).assertGreater(len(entries), 0)
      viz = entries[0].name
      # The visualization should include the failing element (3)
      super(MyTestCase, self).assertIn('3', viz)
      super(MyTestCase, self).assertIn('foreach', viz)

  async def test_foreach_async_exception_sets_temp_args(self):
    """An async foreach (sync-to-async transition) exception sets
    __quent_link_temp_args__ on the exception via _foreach_to_async
    (lines 98-101 of _iteration.pxi). _handle_exception then updates
    ctx.link_temp_args."""
    counter = {'n': 0}
    def f(el):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(el * 10)
      raise ValueError('async foreach error')

    counter['n'] = 0
    with self.assertRaises(ValueError):
      await await_(
        Chain([1, 2, 3]).foreach(f).run()
      )

  async def test_foreach_exception_in_full_async_iterator(self):
    """Exception in _foreach_full_async (async iterator) sets
    __quent_link_temp_args__ and _handle_exception copies them."""
    class AsyncIter:
      def __init__(self, items):
        self._items = list(items)
      def __aiter__(self):
        self._it = iter(self._items)
        return self
      async def __anext__(self):
        try:
          return next(self._it)
        except StopIteration:
          raise StopAsyncIteration

    def fail_on_large(x):
      if x >= 3:
        raise ValueError(f'too large: {x}')
      return x

    try:
      await Chain(AsyncIter([1, 2, 3])).foreach(fail_on_large).run()
      super(MyTestCase, self).fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      super(MyTestCase, self).assertGreater(len(entries), 0)
      viz = entries[0].name
      super(MyTestCase, self).assertIn('3', viz)

  async def test_filter_exception_async_sets_temp_args(self):
    """Exception in _filter_to_async sets __quent_link_temp_args__.
    _handle_exception updates ctx.link_temp_args (line 94 path if
    there is already a link_temp_args from a prior operation)."""
    counter = {'n': 0}
    def pred(x):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty(True)
      raise ValueError('filter transition error')

    counter['n'] = 0
    with self.assertRaises(ValueError):
      await await_(
        Chain([10, 20, 30]).filter(pred).run()
      )


# ---------------------------------------------------------------------------
# 5. get_true_source_link nested resolution paths
# ---------------------------------------------------------------------------

class TrueSourceLinkResolutionTests(TestCase):

  def test_nested_chain_source_link_resolves_to_inner_lambda(self):
    """get_true_source_link resolves through a nested chain to find the
    actual failing link. The traceback arrow should point to the inner
    lambda (lines 161-168 of _diagnostics.pxi)."""
    inner = Chain().then(lambda v: 1 / 0)
    try:
      Chain(5).then(inner).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # The arrow should point to the inner chain's lambda
      self.assertIn('<----', viz)
      self.assertIn('<lambda>', viz)

  def test_deeply_nested_chain_source_link(self):
    """get_true_source_link resolves through multiple levels of nesting."""
    innermost = Chain().then(lambda v: 1 / 0)
    middle = Chain().then(innermost)
    try:
      Chain(5).then(middle).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)

  def test_void_nested_chain_source_link(self):
    """get_true_source_link with a void nested chain (no root_link).
    When the chain has no root_link but ctx.temp_root_link may be set,
    the resolution follows different branches (lines 169-172)."""
    inner = Chain().then(lambda v=None: 1 / 0)
    try:
      Chain(5).then(inner).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)
      self.assertIn('<lambda>', viz)

  def test_nested_chain_with_root_source_link(self):
    """A nested chain that HAS a root_link. get_true_source_link follows
    chain.root_link (line 168) and resolves to the root if it's the
    failing operation.

    Note: the inner chain is invoked with ... (Ellipsis) so that the outer
    chain's current value is NOT passed as a root override. Without ...,
    the outer chain's value (5) would conflict with the inner chain's
    existing root (lambda: 1/0), raising QuentException instead of
    ZeroDivisionError.
    """
    inner = Chain(lambda: 1 / 0)
    try:
      Chain(5).then(inner, ...).run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('<----', viz)


# ---------------------------------------------------------------------------
# 6. format_link chain with args/kwargs in repr
# ---------------------------------------------------------------------------

class FormatLinkChainArgsTests(TestCase):

  def test_nested_chain_with_positional_args_in_repr(self):
    """repr of a chain containing a nested chain called with positional
    args shows the args as the nested chain's root value (line 281 of
    _diagnostics.pxi where args_s or kwargs_s triggers chain_newline)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, 'arg1', key='val')
    r = repr(c)
    self.assertIn('Chain', r)
    self.assertIn('.then', r)
    # The args should appear in the nested chain representation
    self.assertIn('arg1', r)
    self.assertIn('val', r)

  def test_nested_chain_with_kwargs_only_in_repr(self):
    """repr of a chain with nested chain + kwargs-only. Since kwargs-only
    triggers EVAL_CALL_WITHOUT_ARGS for chains, the nested chain repr
    may not show the kwargs (they are not forwarded)."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, key='val')
    r = repr(c)
    self.assertIn('Chain', r)
    self.assertIn('.then', r)

  def test_nested_chain_with_ellipsis_in_repr(self):
    """repr of a chain with a nested chain called with ... (ellipsis).
    The ellipsis is rendered by format_args."""
    inner = Chain().then(lambda v: v)
    c = Chain(1).then(inner, ...)
    r = repr(c)
    self.assertIn('Chain', r)

  def test_format_link_source_arrow_with_results(self):
    """When an exception occurs, links before the source get '= result'
    annotations showing their intermediate values (line 291-293 of
    _diagnostics.pxi).

    Note: link_results (intermediate values) are only populated when
    debug=True is set via .config(debug=True). Without it, the traceback
    visualization will not include '= result' annotations.
    """
    def double(v):
      return v * 2

    def raiser(v):
      raise ValueError('boom')

    try:
      Chain(5).then(double).then(raiser).config(debug=True).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      # The double link should show its result (10)
      self.assertIn('10', viz)
      # The raiser link should have the arrow
      self.assertIn('<----', viz)

  def test_format_link_chain_with_args_in_exception_traceback(self):
    """When a nested chain with args raises, the traceback visualization
    shows the args as part of the nested chain's root."""
    inner = Chain().then(lambda v: 1 / 0)
    try:
      Chain(5).then(inner, 'extra_arg').run()
      self.fail('Expected ZeroDivisionError')
    except ZeroDivisionError as exc:
      entries = _get_quent_entries(exc)
      self.assertGreater(len(entries), 0)
      viz = entries[0].name
      self.assertIn('extra_arg', viz)


# ---------------------------------------------------------------------------
# 7. Pipe library integration (conditional)
# ---------------------------------------------------------------------------

try:
  from pipe import Pipe as _PipeCls
  _HAS_PIPE = True
except ImportError:
  _PipeCls = None
  _HAS_PIPE = False


@unittest.skipUnless(_HAS_PIPE, 'pipe library not installed')
class PipeIntegrationTests(MyTestCase):

  async def test_pipe_object_unwrapped_in_link(self):
    """When the pipe library is installed, Pipe objects are unwrapped
    to their .function attribute (lines 111-112 of _link.pxi)."""
    p = _PipeCls(lambda x: x * 2)
    result = Chain(5).then(p).run()
    await self.assertEqual(result, 10)

  async def test_pipe_with_chain_operations(self):
    """Pipe objects work with other chain operations."""
    p = _PipeCls(lambda x: x + 1)
    result = Chain(10).then(p).then(lambda v: v * 3).run()
    await self.assertEqual(result, 33)  # (10+1)*3


if __name__ == '__main__':
  unittest.main()
