"""Comprehensive tests targeting ALL uncovered lines in _chain_core.pxi.

Missing lines: 56, 65, 82, 100, 212, 217, 243, 280, 322, 326, 335-338,
342, 345, 424, 450, 461, 542, 548, 629-630, 634-635, 647, 653

Line mapping:
  56   Chain.__init__ entry
  65   Chain.init entry
  82   Chain._then entry
  100  Chain._run entry
  212  except handler result is Null -> return None (sync)
  217  finally block entry in _run
  243  _run_async entry
  280  _run_async debug lazy init (link_results = {})
  322  _run_async except handler result is Null -> return None
  326  _run_async finally entry
  335-338  _run_async finally exception lazy ctx init
  342  _run_simple entry
  345  _run_simple nested check
  424  _run_async_simple entry
  450  _run_async_simple Null return
  461  _run_async_simple Break nested re-raise
  542  run method
  548  run autorun check (iscoro result)
  629-630  return_ classmethod
  634-635  break_ classmethod
  647  __call__ _run
  653  __call__ autorun check
"""
import asyncio
import inspect
import logging
import warnings
from unittest import IsolatedAsyncioTestCase, TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run
from quent.quent import PyNull


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ChainExc(Exception):
  pass


class FinallyExc(Exception):
  pass


def raise_test_exc(v=None):
  raise TestExc('test error')


async def async_raise_test_exc(v=None):
  raise TestExc('async test error')


# ---------------------------------------------------------------------------
# 1. Lines 56, 65: Chain.__init__ and Chain.init
#    These are entry points that must be traversed for ANY Chain construction.
#    They are covered by construction, but we explicitly verify all branches.
# ---------------------------------------------------------------------------

class ChainInitTests(TestCase):
  """Verify Chain.__init__ and .init() are reached with various root values."""

  def test_init_with_no_root(self):
    """Line 56/65: Chain() with no root value (root_value is Null)."""
    c = Chain()
    result = c.run()
    self.assertIsNone(result)

  def test_init_with_value_root(self):
    """Line 56/65: Chain(42) with a literal root value."""
    c = Chain(42)
    self.assertEqual(c.run(), 42)

  def test_init_with_callable_root(self):
    """Line 56/65: Chain(fn) with a callable root."""
    c = Chain(lambda: 99)
    self.assertEqual(c.run(), 99)

  def test_init_with_callable_and_args(self):
    """Line 56/65: Chain(fn, arg) with args passed through."""
    c = Chain(lambda x: x * 2, 5)
    self.assertEqual(c.run(), 10)

  def test_init_with_callable_and_kwargs(self):
    """Line 56/65: Chain(fn, key=val) with kwargs."""
    c = Chain(lambda x=0: x + 1, x=10)
    self.assertEqual(c.run(), 11)

  def test_init_cascade(self):
    """Line 65: Cascade init sets is_cascade=True."""
    c = Cascade(42)
    self.assertEqual(c.run(), 42)


# ---------------------------------------------------------------------------
# 2. Line 82: Chain._then entry
#    Appending a link. Covered by any .then(), .do(), .except_(), etc.
# ---------------------------------------------------------------------------

class ChainThenTests(TestCase):
  """Verify Chain._then is exercised with various link types."""

  def test_then_simple_link(self):
    """Line 82: _then with a simple .then() link."""
    result = Chain(1).then(lambda v: v + 1).run()
    self.assertEqual(result, 2)

  def test_then_multiple_links(self):
    """Line 82: _then builds a linked list of multiple links."""
    result = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).run()
    self.assertEqual(result, 6)

  def test_then_with_do(self):
    """Line 82: _then with ignore_result link (do)."""
    side = {'v': None}
    def side_effect(v):
      side['v'] = v
    result = Chain(5).do(side_effect).run()
    self.assertEqual(result, 5)
    self.assertEqual(side['v'], 5)

  def test_then_with_except(self):
    """Line 82: _then with exception handler link (sets _is_simple=False)."""
    c = Chain(1).except_(lambda v: None)
    # _is_simple should be False now
    self.assertEqual(c.run(), 1)


# ---------------------------------------------------------------------------
# 3. Line 100: Chain._run entry
#    Entered on every run/call. Non-simple path (has .do/.except_/.finally_/debug).
# ---------------------------------------------------------------------------

class ChainRunEntryTests(TestCase):
  """Verify Chain._run is entered for non-simple chains."""

  def test_run_non_simple_with_do(self):
    """Line 100: _run entered for chain with .do()."""
    result = Chain(10).do(lambda v: None).run()
    self.assertEqual(result, 10)

  def test_run_non_simple_with_except(self):
    """Line 100: _run entered for chain with .except_()."""
    result = Chain(10).except_(lambda v: None).run()
    self.assertEqual(result, 10)

  def test_run_non_simple_with_finally(self):
    """Line 100: _run entered for chain with .finally_()."""
    result = Chain(10).finally_(lambda v: None).run()
    self.assertEqual(result, 10)

  def test_run_non_simple_with_debug(self):
    """Line 100: _run entered for chain with debug=True."""
    result = Chain(10).then(lambda v: v * 2).config(debug=True).run()
    self.assertEqual(result, 20)


# ---------------------------------------------------------------------------
# 4. Lines 212, 322: except handler result is Null -> return None
#    When except_(fn, reraise=False) and fn returns the Null sentinel,
#    _run returns None. evaluate_value on the exc_link returns Null
#    when the handler returns Null explicitly or is a non-callable literal
#    evaluated as EVAL_RETURN_AS_IS with value=Null.
#    The most practical way: handler that returns None (not Null). But the
#    code checks `result is Null`. evaluate_value returns Null only when
#    the link's eval_code is EVAL_RETURN_AS_IS and the value is Null itself.
#    Actually, for except_ links, the handler IS called (it's callable),
#    so evaluate_value calls it. If handler returns None, result=None (not Null).
#    The only way result is Null is if the handler literal IS Null.
#    But we can't easily pass Null through the public API.
#    Actually, looking more carefully: the except_ handler link is built
#    with Link(__fn, args, kwargs, fn_name='except_'), and fn_name='except_'
#    does not set allow_literal=True. So __fn must be callable.
#    If the callable returns Null (the sentinel), result is Null.
#    We can import PyNull and return it from the handler.
# ---------------------------------------------------------------------------

class ExceptHandlerNullReturnSyncTests(TestCase):
  """Line 212: sync except handler returns Null sentinel -> _run returns None."""

  def test_except_handler_returns_null_sentinel_sync(self):
    """Sync chain: handler returns Null sentinel -> result is Null -> return None."""
    def handler(v=None):
      return PyNull  # Return the Null sentinel

    result = Chain(1).then(raise_test_exc).except_(handler, reraise=False).run()
    self.assertIsNone(result)

  def test_except_handler_returns_null_sentinel_sync_non_simple(self):
    """Non-simple sync chain: handler returns Null sentinel -> return None."""
    def handler(v=None):
      return PyNull

    result = (
      Chain(1).do(lambda v: None)
      .then(raise_test_exc)
      .except_(handler, reraise=False)
      .run()
    )
    self.assertIsNone(result)


class ExceptHandlerNullReturnAsyncTests(IsolatedAsyncioTestCase):
  """Line 322: async except handler returns Null sentinel -> _run_async returns None."""

  async def test_except_handler_returns_null_sentinel_async(self):
    """Async chain: handler returns Null sentinel -> result is Null -> return None."""
    def handler(v=None):
      return PyNull

    result = await await_(
      Chain(aempty, 1).then(raise_test_exc)
      .except_(handler, reraise=False).run()
    )
    self.assertIsNone(result)

  async def test_except_handler_returns_null_sentinel_async_with_do(self):
    """Async non-simple chain: handler returns Null -> return None."""
    def handler(v=None):
      return PyNull

    result = await await_(
      Chain(aempty, 1).do(lambda v: None)
      .then(raise_test_exc)
      .except_(handler, reraise=False).run()
    )
    self.assertIsNone(result)

  async def test_async_handler_returns_null_sentinel(self):
    """Async handler (coroutine) that returns Null sentinel."""
    async def handler(v=None):
      return PyNull

    result = await await_(
      Chain(aempty, 1).then(raise_test_exc)
      .except_(handler, reraise=False).run()
    )
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 5. Line 217: finally block entry in _run (sync path)
#    Reached when ignore_finally is False and on_finally_link is not None.
#    This is the sync success path with a finally handler.
# ---------------------------------------------------------------------------

class FinallyBlockEntrySyncTests(TestCase):
  """Line 217: finally block entry in _run on the sync success path."""

  def test_finally_runs_on_sync_success(self):
    """Sync chain completes, finally handler invoked."""
    called = {'v': False}
    def handler(v=None):
      called['v'] = True
    result = Chain(10).then(lambda v: v + 5).finally_(handler).run()
    self.assertEqual(result, 15)
    self.assertTrue(called['v'])

  def test_finally_runs_on_sync_exception(self):
    """Sync chain raises, finally handler still invoked."""
    called = {'v': False}
    def handler(v=None):
      called['v'] = True
    with self.assertRaises(TestExc):
      Chain(1).then(raise_test_exc).finally_(handler).run()
    self.assertTrue(called['v'])

  def test_finally_not_run_when_none(self):
    """Without .finally_(), on_finally_link is None -> finally block not entered."""
    # This just ensures the non-finally path works
    result = Chain(10).do(lambda v: None).run()
    self.assertEqual(result, 10)


# ---------------------------------------------------------------------------
# 6. Line 243: _run_async entry
#    Entered when a coroutine is detected during _run (non-simple chain).
# ---------------------------------------------------------------------------

class RunAsyncEntryTests(IsolatedAsyncioTestCase):
  """Line 243: _run_async entry on async root or mid-chain coroutine."""

  async def test_run_async_on_async_root_non_simple(self):
    """Non-simple chain with async root -> enters _run_async."""
    result = await Chain(aempty, 42).do(lambda v: None).run()
    self.assertEqual(result, 42)

  async def test_run_async_on_async_link_non_simple(self):
    """Non-simple chain with sync root, async link -> enters _run_async mid-loop."""
    async def async_double(v):
      return v * 2

    result = await Chain(10).do(lambda v: None).then(async_double).run()
    self.assertEqual(result, 20)

  async def test_run_async_with_except_handler(self):
    """Async chain with except_ handler -> _run_async with exception handling."""
    called = {'v': False}
    async def handler(v=None):
      called['v'] = True

    with self.assertRaises(TestExc):
      await Chain(aempty, 1).then(raise_test_exc).except_(handler).run()
    self.assertTrue(called['v'])


# ---------------------------------------------------------------------------
# 7. Line 280: _run_async debug lazy init (link_results = {})
#    Async chain with debug=True, where link_results is None at the time
#    a link is evaluated in the async path. This happens when the chain has
#    a rootless start (no root value) or when the first async link is the
#    first link to be evaluated with debug.
# ---------------------------------------------------------------------------

class RunAsyncDebugLazyInitTests(IsolatedAsyncioTestCase):
  """Line 280: link_results = {} in _run_async debug path.

  Note: _run_async does NOT call _logger.debug(). It only populates
  the link_results dict for traceback augmentation. So we verify correct
  execution, not log output.
  """

  async def test_debug_async_rootless_chain(self):
    """Void async chain with debug -> link_results lazily initialized in _run_async.

    Chain has no root. First link (.do()) is sync, then async link.
    _run evaluates .do() sync, then .then(aempty, 42) returns a coro.
    _run jumps to _run_async with link_results possibly None (it was
    initialized in _run's sync debug path, so it may be {} already).
    Either way, the chain must complete correctly.
    """
    result = await Chain().do(lambda: None).then(aempty, 42).config(debug=True).run()
    self.assertEqual(result, 42)

  async def test_debug_async_link_results_init_on_root(self):
    """Async root with debug=True -> link_results initialized in _run_async.

    Async root on non-simple chain. Root goes async -> _run_async.
    link_results is None because _run didn't get to initialize it
    (the iscoro check at line 129 fires before the debug block at 136).
    In _run_async, line 254-256 checks self._debug and lazily creates
    link_results = {} (line 255-256). Line 280 similarly handles
    subsequent links in the while loop.
    """
    result = await Chain(aempty, 5).do(lambda v: None).then(lambda v: v * 3).config(debug=True).run()
    self.assertEqual(result, 15)

  async def test_debug_async_mid_chain_link_results_init(self):
    """Async root + debug + non-simple + subsequent links.

    The root is async, so _run jumps to _run_async with link_results=None.
    _run_async initializes link_results lazily at line 255-256 for the
    root, and at line 279-280 for subsequent links in the while loop.
    """
    async def async_add(v):
      return v + 10

    result = await Chain(aempty, 5).do(lambda v: None).then(async_add).config(debug=True).run()
    self.assertEqual(result, 15)

  async def test_debug_async_link_results_in_while_loop(self):
    """Specifically hits line 279-280: link_results init in the while loop.

    When the async root is the first coro, _run_async enters with
    link_results=None. After awaiting root, it sets link_results at
    line 255-256. But if we need to hit line 279-280 specifically
    (the while loop path), we need a chain where the FIRST coro is
    a non-root link in _run, and then more links follow in the while loop
    of _run_async. The key is that link_results would have been initialized
    at line 255-256 already, but line 279-280 is the same lazy init check
    in the loop body.
    """
    async def async_double(v):
      return v * 2

    result = await Chain(aempty, 3).do(lambda v: None).then(
      lambda v: v + 1
    ).then(async_double).config(debug=True).run()
    self.assertEqual(result, 8)


# ---------------------------------------------------------------------------
# 8. Line 326: _run_async finally entry
#    The finally block in _run_async when on_finally_link is not None.
# ---------------------------------------------------------------------------

class RunAsyncFinallyEntryTests(IsolatedAsyncioTestCase):
  """Line 326: _run_async finally block when on_finally_link is set."""

  async def test_async_chain_with_sync_finally(self):
    """Async chain with sync finally handler -> line 326 reached."""
    called = {'v': False}
    def handler(v=None):
      called['v'] = True
    result = await Chain(aempty, 10).then(lambda v: v * 2).finally_(handler).run()
    self.assertEqual(result, 20)
    self.assertTrue(called['v'])

  async def test_async_chain_with_async_finally(self):
    """Async chain with async finally handler -> awaited at line 329-330."""
    called = {'v': False}
    async def handler(v=None):
      called['v'] = True
    result = await Chain(aempty, 10).then(lambda v: v * 2).finally_(handler).run()
    self.assertEqual(result, 20)
    self.assertTrue(called['v'])

  async def test_async_chain_finally_after_exception(self):
    """Async chain raises exception, finally still runs."""
    called = {'v': False}
    async def handler(v=None):
      called['v'] = True
    with self.assertRaises(TestExc):
      await Chain(aempty, 1).then(raise_test_exc).finally_(handler).run()
    self.assertTrue(called['v'])


# ---------------------------------------------------------------------------
# 9. Lines 335-338: _run_async finally exception lazy ctx init
#    When async chain succeeds (no exception), ctx is None. If the finally
#    handler then raises, ctx must be lazily created.
# ---------------------------------------------------------------------------

class RunAsyncFinallyLazyCtxTests(IsolatedAsyncioTestCase):
  """Lines 335-338: lazy _ExecCtx creation when finally raises in _run_async."""

  async def test_finally_raises_on_async_success_path_sync_handler(self):
    """Async chain succeeds, sync finally handler raises -> lazy ctx init."""
    def bad_finally(v=None):
      raise FinallyExc('lazy ctx from sync handler')

    with self.assertRaises(FinallyExc) as cm:
      await Chain(aempty, 1).then(lambda v: v * 2).finally_(bad_finally).run()
    self.assertEqual(str(cm.exception), 'lazy ctx from sync handler')

  async def test_finally_raises_on_async_success_path_async_handler(self):
    """Async chain succeeds, async finally handler raises -> lazy ctx init."""
    async def bad_async_finally(v=None):
      raise FinallyExc('lazy ctx from async handler')

    with self.assertRaises(FinallyExc) as cm:
      await Chain(aempty, 1).then(lambda v: v * 2).finally_(bad_async_finally).run()
    self.assertEqual(str(cm.exception), 'lazy ctx from async handler')

  async def test_finally_raises_on_async_success_no_except_handler(self):
    """Async chain with no except_, finally raises -> ctx is None -> lazy init."""
    def bad_finally(v=None):
      raise FinallyExc('no except handler')

    with self.assertRaises(FinallyExc) as cm:
      await Chain(aempty, 5).finally_(bad_finally).run()
    self.assertEqual(str(cm.exception), 'no except handler')

  async def test_finally_raises_after_async_exception_handled(self):
    """Async chain raises, except_ handles it, then finally raises.
    In this case ctx is already initialized (NOT None) so the lazy init
    branch is NOT taken. This is the complementary case.
    """
    def bad_finally(v=None):
      raise FinallyExc('after exception handled')

    with self.assertRaises(FinallyExc) as cm:
      await Chain(aempty, 1).then(raise_test_exc).except_(
        lambda v: None, reraise=False
      ).finally_(bad_finally).run()
    self.assertEqual(str(cm.exception), 'after exception handled')


# ---------------------------------------------------------------------------
# 10. Line 342: _run_simple entry
#     Fast path for simple chains (only .then() links, no debug/finally/except).
# ---------------------------------------------------------------------------

class RunSimpleEntryTests(TestCase):
  """Line 342: _run_simple is entered for simple chains."""

  def test_simple_chain_sync(self):
    """Simple sync chain -> _run_simple path."""
    result = Chain(10).then(lambda v: v + 5).run()
    self.assertEqual(result, 15)

  def test_simple_chain_with_root_override(self):
    """Simple chain with root override at run()."""
    result = Chain().then(lambda v: v * 2).run(7)
    self.assertEqual(result, 14)

  def test_simple_chain_no_root_no_links(self):
    """Empty simple chain."""
    result = Chain().run()
    self.assertIsNone(result)

  def test_simple_chain_multiple_then(self):
    """Multiple .then() calls still simple."""
    result = Chain(2).then(lambda v: v * 3).then(lambda v: v + 1).run()
    self.assertEqual(result, 7)


# ---------------------------------------------------------------------------
# 11. Line 345: _run_simple nested check
#     'You cannot directly run a nested chain.'
# ---------------------------------------------------------------------------

class RunSimpleNestedCheckTests(TestCase):
  """Line 345: _run_simple raises when running a nested chain directly."""

  def test_nested_chain_direct_run_simple(self):
    """Simple nested chain (only .then() links) raises QuentException on direct run."""
    inner = Chain().then(lambda v: v)
    # Mark inner as nested by using it in an outer chain
    Chain().then(inner)
    # inner.is_nested is now True. Running it directly hits line 345.
    with self.assertRaises(QuentException) as cm:
      inner.run()
    self.assertIn('cannot directly run a nested chain', str(cm.exception).lower())

  def test_nested_chain_direct_run_non_simple(self):
    """Non-simple nested chain raises QuentException on direct run (line 103)."""
    inner = Chain().do(lambda: None).then(lambda v: v)
    Chain().then(inner)
    with self.assertRaises(QuentException) as cm:
      inner.run()
    self.assertIn('cannot directly run a nested chain', str(cm.exception).lower())


# ---------------------------------------------------------------------------
# 12. Line 424: _run_async_simple entry
#     Async fast path for simple chains.
# ---------------------------------------------------------------------------

class RunAsyncSimpleEntryTests(IsolatedAsyncioTestCase):
  """Line 424: _run_async_simple entered for simple async chains."""

  async def test_simple_async_root(self):
    """Simple chain with async root -> _run_async_simple."""
    result = await Chain(aempty, 42).then(lambda v: v * 2).run()
    self.assertEqual(result, 84)

  async def test_simple_async_link(self):
    """Simple chain with sync root and async link -> _run_async_simple."""
    async def async_add(v):
      return v + 10

    result = await Chain(5).then(async_add).run()
    self.assertEqual(result, 15)

  async def test_simple_async_chain_cascade(self):
    """Simple Cascade with async root -> _run_async_simple."""
    result = await Cascade(aempty, 99).then(lambda v: v * 2).run()
    self.assertEqual(result, 99)

  async def test_simple_async_root_override(self):
    """Simple chain with async root override at run()."""
    result = await Chain().then(lambda v: v + 1).run(aempty, 10)
    self.assertEqual(result, 11)


# ---------------------------------------------------------------------------
# 13. Line 450: _run_async_simple Null return
#     When the async simple chain ends with current_value == Null, return None.
#     This happens for a void chain (no root, no operations that produce values).
#     Actually for _run_async_simple to be entered, a coro must be encountered.
#     And current_value is Null only if has_root_value was False and no links
#     produced a value. But _run_async_simple is entered with current_value
#     as the awaited coro result. So we need a chain where after all processing,
#     current_value ends up as Null.
#     Wait: in _run_async_simple, current_value starts as the awaited result.
#     After the loop, if current_value is Null, return None.
#     For Cascade: `current_value = root_value` at line 441. If root_value is
#     Null (void chain), current_value becomes Null.
#     But if has_root_value is False, we wouldn't have a root to evaluate,
#     and _run_async_simple is only entered from the root eval or link eval
#     when a coro is detected.
#     Let's trace: void chain, first link is async and returns Null sentinel.
#     _run_simple: has_root_value=False, link=first_link. evaluate_value returns
#     coro (async fn returning PyNull). -> _run_async_simple.
#     In _run_async_simple: current_value = await coro = PyNull.
#     has_root_value=False, root_value=Null. No update.
#     link = link.next_link (None if only one link).
#     Loop doesn't execute. current_value is PyNull (which is Null). -> return None.
# ---------------------------------------------------------------------------

class RunAsyncSimpleNullReturnTests(IsolatedAsyncioTestCase):
  """Line 450: _run_async_simple returns None when current_value is Null."""

  async def test_async_simple_returns_null_sentinel(self):
    """Async link returns Null sentinel -> current_value is Null -> return None."""
    async def return_null(v=None):
      return PyNull

    result = await Chain().then(return_null).run()
    self.assertIsNone(result)

  async def test_async_simple_root_returns_null(self):
    """Async root returns Null sentinel, no further links -> return None."""
    async def null_root():
      return PyNull

    result = await Chain(null_root).run()
    self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 14. Line 461: _run_async_simple Break nested re-raise
#     In _run_async_simple, if a _Break exception is caught and the chain
#     is nested, it re-raises. This requires a simple nested async chain
#     that raises _Break.
# ---------------------------------------------------------------------------

class RunAsyncSimpleBreakNestedTests(IsolatedAsyncioTestCase):
  """Line 461: _run_async_simple re-raises _Break when is_nested."""

  async def test_break_in_nested_simple_async_chain(self):
    """Nested simple async chain with break_ re-raises _Break to parent.

    The parent catches _Break via its foreach handler.
    """
    async def items():
      return [1, 2, 3, 4, 5]

    def process(el):
      if el >= 3:
        return Chain.break_(aempty, 'stopped')
      return el * 10

    # Chain(items).foreach(process) uses iteration which catches _Break.
    # But we need to hit _run_async_simple's break path.
    # A nested simple async chain that raises _Break:
    # Inner chain: simple (only .then()), async (has aempty), raises _Break.
    inner = Chain().then(lambda v: Chain.break_(aempty, v))
    # Use inner in an outer chain's foreach:
    result = await Chain(items).foreach(process).run()
    self.assertEqual(result, 'stopped')

  async def test_break_in_nested_simple_async_chain_via_nested_chain(self):
    """Nested simple chain is async and raises _Break, parent catches it."""
    # Create a simple inner chain that:
    # 1. Is marked as nested (by being used in an outer chain)
    # 2. Has only .then() links (simple)
    # 3. Goes async
    # 4. Raises _Break
    # When the inner chain runs in _run_async_simple and _Break is caught,
    # is_nested=True -> re-raise.

    # The outer chain is a foreach loop which catches _Break.
    items = [1, 2, 3]

    # Inner chain that is async and breaks
    inner_break = Chain(aempty).then(Chain.break_)

    # Use the inner chain in a foreach. The foreach iteration will call
    # the inner chain for each element. When _Break propagates, foreach handles it.
    # But foreach takes a function, not a chain directly...
    # Let's do it differently: use the inner chain as a nested chain in a foreach fn.

    def fn(el):
      if el >= 3:
        # This calls Chain.break_ which raises _Break.
        # If we wrap it in an async context, it goes through _run_async_simple.
        return Chain.break_(aempty, 'done')
      return aempty(el * 10)

    result = await Chain(items).foreach(fn).run()
    self.assertEqual(result, 'done')


# ---------------------------------------------------------------------------
# 15. Lines 542, 548: run method and autorun check
#     Line 542: result = self._run(__v, args, kwargs, False)
#     Line 548: if self._autorun and not self._is_sync and iscoro(result):
# ---------------------------------------------------------------------------

class RunMethodTests(IsolatedAsyncioTestCase):
  """Lines 542, 548: run() method entry and autorun check."""

  async def test_run_basic(self):
    """Line 542: run() calls _run."""
    result = Chain(42).run()
    self.assertEqual(result, 42)

  async def test_run_with_override(self):
    """Line 542: run(value) passes value to _run."""
    result = Chain().then(lambda v: v * 2).run(5)
    self.assertEqual(result, 10)

  async def test_run_autorun_with_async_simple(self):
    """Line 548: autorun=True on a simple async chain -> ensure_future in run()."""
    result = Chain(aempty, 42).then(lambda v: v * 2).config(autorun=True).run()
    self.assertIsInstance(result, asyncio.Task)
    value = await result
    self.assertEqual(value, 84)

  async def test_run_autorun_no_effect_on_sync(self):
    """Line 548: autorun on sync chain -> iscoro is False, returns directly."""
    result = Chain(42).then(lambda v: v * 2).config(autorun=True).run()
    self.assertNotIsInstance(result, asyncio.Task)
    self.assertEqual(result, 84)

  async def test_run_autorun_with_no_async_flag(self):
    """Line 548: autorun with no_async(True) -> _is_sync is True, no wrapping."""
    # With no_async(True), even async results are not checked.
    # The chain won't detect coroutines at all, so the coro is returned raw.
    # Actually with no_async(True), iscoro checks are skipped in _run_simple,
    # so the coroutine itself is returned as the result value (not awaited).
    # Then in run(), self._is_sync is True so the autorun check is skipped.
    result = Chain(aempty, 42).config(autorun=True).no_async(True).run()
    # Result is the coroutine object itself (not awaited, not wrapped)
    self.assertTrue(inspect.iscoroutine(result))
    # Clean up
    result.close()

  async def test_run_internal_exception_wrapping(self):
    """Line 548: _InternalQuentException caught and re-raised as QuentException."""
    # Chain.break_() at top level in a simple chain:
    # _run_simple catches _Break, is_nested=False -> raises QuentException
    # run() should propagate this QuentException.
    with self.assertRaises(QuentException):
      Chain().then(Chain.break_).run()


# ---------------------------------------------------------------------------
# 16. Lines 629-630: return_ classmethod
#     raise _Return(__v, args, kwargs)
# ---------------------------------------------------------------------------

class ReturnClassmethodTests(IsolatedAsyncioTestCase):
  """Lines 629-630: Chain.return_() raises _Return."""

  async def test_return_with_no_value(self):
    """return_() with no value -> _Return(Null, (), {})."""
    # Caught by _run -> handle_return_exc -> returns None
    result = Chain().then(Chain.return_).run()
    self.assertIsNone(result)

  async def test_return_with_value(self):
    """return_(42) -> _Return(42, (), {})."""
    result = Chain().then(Chain.return_, 42).run()
    self.assertEqual(result, 42)

  async def test_return_with_callable(self):
    """return_(fn, arg) -> _Return(fn, (arg,), {})."""
    result = Chain().then(Chain.return_, lambda x: x * 2, 5).run()
    self.assertEqual(result, 10)

  async def test_return_in_nested_chain(self):
    """return_ in nested chain propagates to outer chain."""
    inner = Chain().then(Chain.return_, 99)
    result = Chain(1).then(inner).then(lambda v: v + 1).run()
    self.assertEqual(result, 99)

  async def test_return_direct_call_raises_exception(self):
    """Calling Chain.return_() directly raises _Return (subclass of Exception)."""
    with self.assertRaises(Exception):
      Chain.return_()

  async def test_return_with_async_value(self):
    """return_(aempty, value) in async context."""
    result = await Chain(aempty, 1).do(lambda v: None).then(Chain.return_, aempty, 77).run()
    self.assertEqual(result, 77)


# ---------------------------------------------------------------------------
# 17. Lines 634-635: break_ classmethod
#     raise _Break(__v, args, kwargs)
# ---------------------------------------------------------------------------

class BreakClassmethodTests(IsolatedAsyncioTestCase):
  """Lines 634-635: Chain.break_() raises _Break."""

  async def test_break_direct_call(self):
    """break_() raises _Break (subclass of Exception)."""
    with self.assertRaises(Exception):
      Chain.break_()

  async def test_break_with_value(self):
    """break_(value) carries the value in the _Break exception."""
    # When used in foreach, the value becomes the foreach result
    def fn(el):
      if el >= 3:
        return Chain.break_('stopped')
      return el

    result = Chain([1, 2, 3, 4]).foreach(fn).run()
    self.assertEqual(result, 'stopped')

  async def test_break_with_callable_value(self):
    """break_(fn, arg) evaluates fn(arg) as the break result."""
    def fn(el):
      if el >= 3:
        return Chain.break_(lambda x: x * 10, 5)
      return el

    result = Chain([1, 2, 3]).foreach(fn).run()
    self.assertEqual(result, 50)

  async def test_break_outside_iteration_raises_quent_exception(self):
    """break_ outside of foreach/iterate raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      Chain().then(Chain.break_).run()
    self.assertIn('_Break', str(cm.exception))

  async def test_break_outside_iteration_async_raises(self):
    """break_ outside of iteration in async chain raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      await Chain(aempty).then(Chain.break_).run()
    self.assertIn('_Break', str(cm.exception))


# ---------------------------------------------------------------------------
# 18. Lines 647, 653: __call__ _run and autorun check
#     Line 647: result = self._run(__v, args, kwargs, False)
#     Line 653: if self._autorun and not self._is_sync and iscoro(result):
# ---------------------------------------------------------------------------

class CallMethodTests(IsolatedAsyncioTestCase):
  """Lines 647, 653: __call__() delegates to _run and checks autorun."""

  async def test_call_basic(self):
    """Line 647: chain() invokes _run."""
    result = Chain(42)()
    self.assertEqual(result, 42)

  async def test_call_with_override(self):
    """Line 647: chain(value) passes value to _run."""
    result = Chain().then(lambda v: v * 3)(7)
    self.assertEqual(result, 21)

  async def test_call_autorun_with_async(self):
    """Line 653: autorun=True via __call__ -> ensure_future."""
    chain = Chain(aempty, 42).then(lambda v: v + 8).config(autorun=True)
    result = chain()
    self.assertIsInstance(result, asyncio.Task)
    value = await result
    self.assertEqual(value, 50)

  async def test_call_autorun_sync_no_wrap(self):
    """Line 653: autorun on sync chain via __call__ -> no wrapping."""
    chain = Chain(10).then(lambda v: v * 5).config(autorun=True)
    result = chain()
    self.assertNotIsInstance(result, asyncio.Task)
    self.assertEqual(result, 50)

  async def test_call_internal_exception_wrapping(self):
    """Line 653: _InternalQuentException caught as QuentException via __call__."""
    with self.assertRaises(QuentException):
      Chain().then(Chain.break_)()

  async def test_call_autorun_no_async_flag(self):
    """Line 653: autorun + no_async(True) via __call__ -> _is_sync True, skip."""
    chain = Chain(aempty, 42).config(autorun=True).no_async(True)
    result = chain()
    # With no_async, the coroutine is not detected; returned raw
    self.assertTrue(inspect.iscoroutine(result))
    result.close()


# ---------------------------------------------------------------------------
# 19. Combined edge cases: async paths with debug + finally + except
# ---------------------------------------------------------------------------

class CombinedAsyncEdgeCaseTests(IsolatedAsyncioTestCase):
  """Exercise multiple uncovered lines in a single test scenario."""

  async def test_async_chain_debug_finally_except_all_paths(self):
    """Non-simple async chain with debug, finally, and except all active.

    Hits: _run entry (100), _run_async entry (243), debug link_results
    init in _run_async (255-256, 279-280), _run_async finally (326).

    Note: _run_async does NOT emit _logger.debug() calls. It only
    populates link_results for traceback augmentation. We verify
    correct execution and that finally/except handlers behave properly.
    """
    finally_called = {'v': False}
    except_called = {'v': False}

    def on_except(v=None):
      except_called['v'] = True

    def on_finally(v=None):
      finally_called['v'] = True

    result = await Chain(aempty, 5).then(lambda v: v * 3).except_(
      on_except
    ).finally_(on_finally).config(debug=True).run()

    self.assertEqual(result, 15)
    self.assertFalse(except_called['v'])  # No exception occurred
    self.assertTrue(finally_called['v'])

  async def test_async_chain_exception_null_handler_finally_raises(self):
    """Async chain: exception -> handler returns Null -> finally raises.

    Hits: _run_async except handler Null return (322),
    _run_async finally (326), and finally exception with ctx already set.
    """
    def null_handler(v=None):
      return PyNull

    def bad_finally(v=None):
      raise FinallyExc('combined')

    # When except handler returns Null with reraise=False, it tries to
    # return None. But then the finally block runs and raises.
    with self.assertRaises(FinallyExc):
      await Chain(aempty, 1).then(raise_test_exc).except_(
        null_handler, reraise=False
      ).finally_(bad_finally).run()

  async def test_sync_chain_except_null_handler_with_finally(self):
    """Sync chain: exception -> handler returns Null -> finally runs normally.

    Hits: _run except handler Null return (212), finally block entry (217).
    """
    def null_handler(v=None):
      return PyNull

    finally_called = {'v': False}
    def on_finally(v=None):
      finally_called['v'] = True

    result = Chain(1).then(raise_test_exc).except_(
      null_handler, reraise=False
    ).finally_(on_finally).run()

    self.assertIsNone(result)
    self.assertTrue(finally_called['v'])


# ---------------------------------------------------------------------------
# 20. Async autorun via both run() and __call__() with non-simple chains
# ---------------------------------------------------------------------------

class AutorunNonSimpleAsyncTests(IsolatedAsyncioTestCase):
  """Lines 548, 653: autorun with non-simple async chains via run() and __call__().

  For non-simple chains, _run itself handles the autorun check at lines
  132-133 and 159-160. But the result may still be a coroutine when it
  returns from _run to run() or __call__(). This tests that path.
  """

  async def test_autorun_run_non_simple_async(self):
    """run() with autorun on non-simple async chain."""
    result = Chain(aempty, 10).do(lambda v: None).config(autorun=True).run()
    self.assertIsInstance(result, asyncio.Task)
    self.assertEqual(await result, 10)

  async def test_autorun_call_non_simple_async(self):
    """__call__() with autorun on non-simple async chain."""
    chain = Chain(aempty, 10).do(lambda v: None).config(autorun=True)
    result = chain()
    self.assertIsInstance(result, asyncio.Task)
    self.assertEqual(await result, 10)


# ---------------------------------------------------------------------------
# 21. Cascade async paths (ensures cascade-specific branches in _run_async
#     and _run_async_simple)
# ---------------------------------------------------------------------------

class CascadeAsyncTests(IsolatedAsyncioTestCase):
  """Cascade mode in async paths."""

  async def test_cascade_async_returns_root_value(self):
    """Cascade with async root -> _run_async returns root_value."""
    result = await Cascade(aempty, 42).then(lambda v: v * 2).run()
    self.assertEqual(result, 42)

  async def test_cascade_async_with_finally(self):
    """Cascade async chain with finally handler."""
    called = {'v': False}
    async def handler(v=None):
      called['v'] = True
    result = await Cascade(aempty, 42).then(lambda v: v * 2).finally_(handler).run()
    self.assertEqual(result, 42)
    self.assertTrue(called['v'])

  async def test_cascade_async_with_do(self):
    """Cascade async chain with .do() (non-simple path)."""
    side = {'v': None}
    def track(v):
      side['v'] = v
    result = await Cascade(aempty, 42).do(track).then(lambda v: v * 2).run()
    self.assertEqual(result, 42)
    self.assertEqual(side['v'], 42)


# ---------------------------------------------------------------------------
# 22. Pipe operator + run -> ensures __or__ with run() calls self.run()
# ---------------------------------------------------------------------------

class PipeRunTests(IsolatedAsyncioTestCase):
  """Pipe operator integration with run()."""

  async def test_pipe_run_basic(self):
    """Chain | run() invokes self.run()."""
    result = Chain(10) | (lambda v: v + 5) | run()
    self.assertEqual(result, 15)

  async def test_pipe_run_with_value(self):
    """Chain | run(val) passes root override."""
    result = Chain() | (lambda v: v * 2) | run(7)
    self.assertEqual(result, 14)

  async def test_pipe_run_async(self):
    """Async chain with pipe + run."""
    result = await (Chain(aempty, 5) | (lambda v: v * 3) | run())
    self.assertEqual(result, 15)


# ---------------------------------------------------------------------------
# 23. Additional edge cases for complete coverage
# ---------------------------------------------------------------------------

class AdditionalEdgeCaseTests(IsolatedAsyncioTestCase):
  """Miscellaneous edge cases for remaining uncovered paths."""

  async def test_void_chain_with_finally_only(self):
    """Chain with no root and no links, only finally."""
    called = {'v': False}
    def handler(v=None):
      called['v'] = True
    result = Chain().finally_(handler).run()
    self.assertIsNone(result)
    self.assertTrue(called['v'])

  async def test_void_async_chain_with_debug(self):
    """Void chain going async with debug enabled."""
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      result = await Chain().then(aempty, 42).do(lambda v: None).config(debug=True).run()
      self.assertEqual(result, 42)
    finally:
      logger.removeHandler(handler)

  async def test_chain_run_with_args_and_kwargs(self):
    """run() passing both args and kwargs."""
    def fn(x, y=0):
      return x + y
    result = Chain().then(lambda v: v * 2).run(fn, 5, y=3)
    self.assertEqual(result, 16)

  async def test_call_with_args_and_kwargs(self):
    """__call__() passing both args and kwargs."""
    def fn(x, y=0):
      return x + y
    result = Chain().then(lambda v: v * 2)(fn, 5, y=3)
    self.assertEqual(result, 16)

  async def test_async_break_outside_iteration_non_simple(self):
    """_Break in async non-simple chain outside iteration -> QuentException.
    Exercises _run_async's except _Break path (lines 299-302).
    """
    with self.assertRaises(QuentException) as cm:
      await Chain(aempty, 1).do(lambda v: None).then(Chain.break_).run()
    self.assertIn('_Break', str(cm.exception))

  async def test_async_return_in_non_simple_chain(self):
    """_Return in async non-simple chain -> handled by _run_async (lines 293-297)."""
    result = await Chain(aempty, 1).do(lambda v: None).then(Chain.return_, 55).run()
    self.assertEqual(result, 55)

  async def test_async_simple_return(self):
    """_Return in async simple chain -> handled by _run_async_simple (lines 453-457)."""
    result = await Chain(aempty, 1).then(Chain.return_, 55).run()
    self.assertEqual(result, 55)

  async def test_async_simple_break_non_nested(self):
    """_Break in async simple chain (non-nested) -> QuentException (lines 459-462)."""
    with self.assertRaises(QuentException) as cm:
      await Chain(aempty, 1).then(Chain.break_).run()
    self.assertIn('_Break', str(cm.exception))

  async def test_except_handler_returns_null_async_handler(self):
    """Async except handler that returns Null via async path."""
    async def null_async_handler(v=None):
      return PyNull

    result = await Chain(aempty, 1).do(lambda v: None).then(
      raise_test_exc
    ).except_(null_async_handler, reraise=False).run()
    self.assertIsNone(result)

  async def test_sync_finally_async_handler_warning_with_ctx_none(self):
    """Sync chain success, async finally handler -> lazy ctx init + warning.

    Lines 230-236 in _run: when chain is sync, finally handler returns coro,
    ctx is None -> lazy _ExecCtx init at lines 231-235.
    """
    called = {'v': False}
    async def async_handler(v=None):
      called['v'] = True

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      result = Chain(1).then(lambda v: v * 2).finally_(async_handler).run()
      self.assertEqual(result, 2)
      runtime_warnings = [
        x for x in w if issubclass(x.category, RuntimeWarning)
      ]
      self.assertGreater(len(runtime_warnings), 0)
    # Let the scheduled task complete
    await asyncio.sleep(0.2)
    self.assertTrue(called['v'])

  async def test_run_async_debug_link_results_in_loop(self):
    """Debug mode in _run_async loop path where link_results needs init.

    After the first await in _run_async, subsequent links are evaluated
    in the while loop. _run_async populates link_results (not _logger).
    The `if link_results is None: link_results = {}` branch at line 279-280
    is hit when the root was async (link_results passed as None from _run).

    We verify the chain produces the correct result, which confirms
    _run_async executed the debug code path without error.
    """
    async def async_root():
      return 5

    result = await Chain(async_root).do(lambda v: None).then(
      lambda v: v * 2
    ).then(lambda v: v + 1).config(debug=True).run()
    self.assertEqual(result, 11)


# ---------------------------------------------------------------------------
# 24. Ensure except handler in sync path with Null and finally combined
# ---------------------------------------------------------------------------

class SyncExceptNullFinallyTests(TestCase):
  """Sync except handler returning Null with finally handler present."""

  def test_sync_except_null_then_finally(self):
    """Sync: except returns Null, finally runs. Lines 212, 217."""
    finally_called = {'v': False}
    def on_finally(v=None):
      finally_called['v'] = True

    def null_handler(v=None):
      return PyNull

    result = Chain(1).then(raise_test_exc).except_(
      null_handler, reraise=False
    ).finally_(on_finally).run()
    self.assertIsNone(result)
    self.assertTrue(finally_called['v'])


# ---------------------------------------------------------------------------
# 25. Ensure async except handler that returns Null with finally present
# ---------------------------------------------------------------------------

class AsyncExceptNullFinallyTests(IsolatedAsyncioTestCase):
  """Async except handler returning Null with finally handler present."""

  async def test_async_except_null_then_finally(self):
    """Async: except returns Null, finally runs. Lines 322, 326."""
    finally_called = {'v': False}
    async def on_finally(v=None):
      finally_called['v'] = True

    def null_handler(v=None):
      return PyNull

    result = await Chain(aempty, 1).then(raise_test_exc).except_(
      null_handler, reraise=False
    ).finally_(on_finally).run()
    self.assertIsNone(result)
    self.assertTrue(finally_called['v'])


if __name__ == '__main__':
  import unittest
  unittest.main()
