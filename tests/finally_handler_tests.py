import asyncio
import warnings
from unittest import TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FinallyExc(Exception):
  pass


class HandlerExc(Exception):
  pass


def raise_test_exc(v=None):
  raise TestExc('chain error')


# ---------------------------------------------------------------------------
# Class 1: FinallySyncSuccessTests
# ---------------------------------------------------------------------------

class FinallySyncSuccessTests(MyTestCase):
  """Finally handler runs on the success path in _run() (lines 217-219 of _chain_core.pxi).

  When a chain completes without exception and ignore_finally is False,
  the finally handler is invoked with the root value.
  """

  async def test_finally_runs_on_success(self):
    """Chain succeeds, finally handler is called exactly once."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'count': 0}
        def handler(v=None):
          called['count'] += 1
        await await_(Chain(fn, 1).then(lambda v: v * 2).finally_(handler).run())
        super(MyTestCase, self).assertEqual(called['count'], 1)

  async def test_finally_receives_root_value_on_success(self):
    """Finally handler receives the root value, not the final computed value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}
        root_obj = object()
        def handler(v=None):
          received['value'] = v
        await await_(
          Chain(fn, root_obj).then(lambda v: 'transformed').finally_(handler).run()
        )
        super(MyTestCase, self).assertIs(received['value'], root_obj)

  async def test_finally_does_not_alter_return_value(self):
    """The finally handler's return value is discarded; chain returns the computed result."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 5).then(lambda v: v * 3).finally_(lambda v: 'ignored').run()
        )
        super(MyTestCase, self).assertEqual(result, 15)

  async def test_finally_on_void_chain(self):
    """Chain with no root value: finally handler is called with None (Null -> no arg)."""
    called = {'value': False}
    received = {'value': 'sentinel'}
    def handler(v=None):
      called['value'] = True
      received['value'] = v
    result = Chain().finally_(handler).run()
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertTrue(called['value'])
    # root_value is Null, so evaluate_value calls handler() with no positional
    # arg, meaning v defaults to None
    super(MyTestCase, self).assertIsNone(received['value'])

  async def test_finally_with_cascade_receives_root(self):
    """Cascade mode: finally handler still receives the root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'value': None}
        def handler(v=None):
          received['value'] = v
        result = await await_(
          Cascade(fn, 42).then(lambda v: v * 2).finally_(handler).run()
        )
        super(MyTestCase, self).assertEqual(result, 42)
        super(MyTestCase, self).assertEqual(received['value'], 42)


# ---------------------------------------------------------------------------
# Class 2: FinallySyncExceptionTests
# ---------------------------------------------------------------------------

class FinallySyncExceptionTests(MyTestCase):
  """Finally handler runs after an exception is caught by except_ (lines 217-219).

  The finally block executes regardless of whether the chain succeeded or failed.
  """

  async def test_finally_runs_after_except_reraise(self):
    """Exception raised, except_ catches with reraise=True, finally still runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = {'value': False}
        except_called = {'value': False}
        def on_except(v=None):
          except_called['value'] = True
        def on_finally(v=None):
          finally_called['value'] = True
        try:
          await await_(
            Chain(fn, 1).then(raise_test_exc)
            .except_(on_except, reraise=True)
            .finally_(on_finally).run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertTrue(except_called['value'])
        super(MyTestCase, self).assertTrue(finally_called['value'])

  async def test_finally_runs_after_except_noraise(self):
    """Exception suppressed by except_(reraise=False), finally still runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = {'value': False}
        except_called = {'value': False}
        def on_except(v=None):
          except_called['value'] = True
          return 'recovered'
        def on_finally(v=None):
          finally_called['value'] = True
        result = await await_(
          Chain(fn, 1).then(raise_test_exc)
          .except_(on_except, reraise=False)
          .finally_(on_finally).run()
        )
        super(MyTestCase, self).assertTrue(except_called['value'])
        super(MyTestCase, self).assertTrue(finally_called['value'])

  async def test_finally_runs_when_no_handler_matches(self):
    """Exception raised, no except_ handler matches, finally still runs before propagation."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = {'value': False}
        def on_finally(v=None):
          finally_called['value'] = True
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(raise_test_exc).finally_(on_finally).run()
          )
        super(MyTestCase, self).assertTrue(finally_called['value'])

  async def test_finally_runs_on_unhandled_exception(self):
    """No except_ registered at all; finally still runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        finally_called = {'value': False}
        def on_finally(v=None):
          finally_called['value'] = True
        with self.assertRaises(ZeroDivisionError):
          await await_(
            Chain(fn, 1).then(lambda v: 1 / 0).finally_(on_finally).run()
          )
        super(MyTestCase, self).assertTrue(finally_called['value'])


# ---------------------------------------------------------------------------
# Class 3: FinallyControlFlowSignalTests
# ---------------------------------------------------------------------------

class FinallyControlFlowSignalTests(MyTestCase):
  """Control flow signals (return_, break_) inside finally -> QuentException.

  Sync: lines 220-221 of _chain_core.pxi
  Async: lines 331-332 of _chain_core.pxi
  """

  async def test_return_in_finally_raises_quent_exception_sync(self):
    """Chain.return_ inside sync finally handler raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).finally_(Chain.return_, 42).run()
    super(MyTestCase, self).assertIn(
      'control flow signals', str(cm.exception).lower()
    )

  async def test_break_in_finally_raises_quent_exception_sync(self):
    """Chain.break_ inside sync finally handler raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      Chain(1).finally_(Chain.break_).run()
    super(MyTestCase, self).assertIn(
      'control flow signals', str(cm.exception).lower()
    )

  async def test_return_in_finally_raises_quent_exception_async(self):
    """Chain.return_ inside finally handler on async chain raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      await await_(Chain(aempty, 1).finally_(Chain.return_, 42).run())
    super(MyTestCase, self).assertIn(
      'control flow signals', str(cm.exception).lower()
    )

  async def test_break_in_finally_raises_quent_exception_async(self):
    """Chain.break_ inside finally handler on async chain raises QuentException."""
    with self.assertRaises(QuentException) as cm:
      await await_(Chain(aempty, 1).finally_(Chain.break_).run())
    super(MyTestCase, self).assertIn(
      'control flow signals', str(cm.exception).lower()
    )

  async def test_return_in_finally_after_exception_raises_quent_exception(self):
    """Chain.return_ in finally after a chain exception still raises QuentException."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(QuentException) as cm:
          await await_(
            Chain(fn, 1).then(raise_test_exc).finally_(Chain.return_, 99).run()
          )
        super(MyTestCase, self).assertIn(
          'control flow signals', str(cm.exception).lower()
        )


# ---------------------------------------------------------------------------
# Class 4: FinallyExceptionInHandlerTests
# ---------------------------------------------------------------------------

class FinallyExceptionInHandlerTests(MyTestCase):
  """Finally handler itself raises an exception, triggering lazy _ExecCtx init.

  Sync: lines 222-228 of _chain_core.pxi
  Async: lines 333-340 of _chain_core.pxi

  When the chain succeeds (no prior exception), ctx is None. The finally
  handler raising triggers the `if ctx is None` branch for lazy initialization.
  """

  async def test_finally_handler_raises_on_success_path_sync(self):
    """Sync chain succeeds, finally handler raises -> exception propagates.

    This hits the lazy ctx init path (ctx is None) at line 223.
    """
    def bad_finally(v=None):
      raise FinallyExc('finally boom')
    with self.assertRaises(FinallyExc) as cm:
      Chain(1).then(lambda v: v * 2).finally_(bad_finally).run()
    super(MyTestCase, self).assertEqual(str(cm.exception), 'finally boom')

  async def test_finally_handler_raises_on_success_path_async(self):
    """Async chain succeeds, finally handler raises -> exception propagates.

    This hits the lazy ctx init path (ctx is None) at line 334 in _run_async.
    """
    def bad_finally(v=None):
      raise FinallyExc('async finally boom')
    with self.assertRaises(FinallyExc) as cm:
      await await_(
        Chain(aempty, 1).then(lambda v: v * 2).finally_(bad_finally).run()
      )
    super(MyTestCase, self).assertEqual(str(cm.exception), 'async finally boom')

  async def test_finally_handler_raises_overrides_chain_exception(self):
    """Chain raises TestExc, finally raises FinallyExc -> FinallyExc propagates.

    In this case ctx is already initialized (not None) from exception handling,
    so the lazy init branch is NOT taken.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        def bad_finally(v=None):
          raise FinallyExc('finally override')
        with self.assertRaises(FinallyExc) as cm:
          await await_(
            Chain(fn, 1).then(raise_test_exc).finally_(bad_finally).run()
          )
        super(MyTestCase, self).assertEqual(str(cm.exception), 'finally override')
        # The original TestExc should be the __context__
        super(MyTestCase, self).assertIsInstance(cm.exception.__context__, TestExc)

  async def test_finally_handler_raises_with_except_noraise(self):
    """Exception suppressed by except_(reraise=False), then finally raises.

    ctx was initialized during exception handling, so lazy init path is NOT taken.
    The FinallyExc should propagate.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        def bad_finally(v=None):
          raise FinallyExc('finally after suppress')
        with self.assertRaises(FinallyExc) as cm:
          await await_(
            Chain(fn, 1).then(raise_test_exc)
            .except_(lambda v: 'recovered', reraise=False)
            .finally_(bad_finally).run()
          )
        super(MyTestCase, self).assertEqual(str(cm.exception), 'finally after suppress')

  async def test_finally_async_handler_raises_on_async_chain(self):
    """Async finally handler raises on async chain -> exception propagates after await."""
    async def bad_async_finally(v=None):
      raise FinallyExc('async handler boom')
    with self.assertRaises(FinallyExc) as cm:
      await await_(
        Chain(aempty, 1).then(lambda v: v * 2).finally_(bad_async_finally).run()
      )
    super(MyTestCase, self).assertEqual(str(cm.exception), 'async handler boom')

  async def test_finally_handler_raises_zero_division(self):
    """Concrete edge case: ZeroDivisionError in finally on success path."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(ZeroDivisionError):
          await await_(
            Chain(fn, 1).then(lambda v: v * 2).finally_(lambda v: 1 / 0).run()
          )


# ---------------------------------------------------------------------------
# Class 5: FinallyAsyncHandlerWarningTests
# ---------------------------------------------------------------------------

class FinallyAsyncHandlerWarningTests(MyTestCase):
  """Async finally handler on a synchronous chain -> RuntimeWarning.

  Lines 230-241 of _chain_core.pxi: when the chain completes synchronously
  but the finally handler returns a coroutine, it is scheduled as a Task
  and a RuntimeWarning is emitted.
  """

  async def test_async_finally_handler_on_sync_chain_warns(self):
    """Async handler on sync chain emits RuntimeWarning about coroutine in finally."""
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(1).then(lambda v: v * 2).finally_(async_handler).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
      super(MyTestCase, self).assertIn(
        'finally', str(runtime_warnings[0].message).lower()
      )
    await asyncio.sleep(0.1)  # let the scheduled task complete

  async def test_async_finally_handler_on_sync_chain_still_executes(self):
    """The async handler IS scheduled as a task and eventually executes."""
    called = {'value': False}
    async def async_handler(v=None):
      called['value'] = True
    with warnings.catch_warnings():
      warnings.simplefilter('ignore', RuntimeWarning)
      Chain(1).then(lambda v: v * 2).finally_(async_handler).run()
    # The task was scheduled; give the event loop time to run it
    await asyncio.sleep(0.2)
    super(MyTestCase, self).assertTrue(called['value'])

  async def test_async_finally_on_sync_chain_with_empty_root(self):
    """Async finally on a void chain (no root) still warns."""
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain().finally_(async_handler).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_async_finally_lazy_ctx_init_for_coroutine(self):
    """When async finally runs on sync success path, ctx is None -> lazy init at line 231-235."""
    called = {'value': False}
    async def async_handler(v=None):
      called['value'] = True
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      # The chain succeeds synchronously, so ctx was never initialized.
      # The async coroutine triggers the `if ctx is None` lazy init branch.
      result = Chain(empty, 5).then(lambda v: v + 1).finally_(async_handler).run()
      super(MyTestCase, self).assertEqual(result, 6)
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.2)
    super(MyTestCase, self).assertTrue(called['value'])

  async def test_sync_finally_handler_no_warning(self):
    """Sync finally handler on sync chain does NOT emit RuntimeWarning."""
    def sync_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(1).then(lambda v: v * 2).finally_(sync_handler).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertEqual(len(runtime_warnings), 0)


# ---------------------------------------------------------------------------
# Class 6: FinallyAsyncPathTests
# ---------------------------------------------------------------------------

class FinallyAsyncPathTests(MyTestCase):
  """Finally in fully async chains (_run_async, lines 325-340 of _chain_core.pxi).

  When the chain enters the async path, the finally block in _run_async
  runs after the async evaluation loop completes. Async handlers are
  directly awaited (line 330).
  """

  async def test_async_chain_sync_finally_on_success(self):
    """Async chain succeeds, sync finally handler runs."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = {'value': False}
        def handler(v=None):
          called['value'] = True
        result = await await_(
          Chain(fn, 10).then(lambda v: v * 3).finally_(handler).run()
        )
        super(MyTestCase, self).assertEqual(result, 30)
        super(MyTestCase, self).assertTrue(called['value'])

  async def test_async_chain_async_finally_on_success(self):
    """Async chain succeeds, async finally handler is awaited (line 330)."""
    called = {'value': False}
    async def async_handler(v=None):
      called['value'] = True
    result = await await_(
      Chain(aempty, 10).then(lambda v: v * 3).finally_(async_handler).run()
    )
    super(MyTestCase, self).assertEqual(result, 30)
    super(MyTestCase, self).assertTrue(called['value'])

  async def test_async_chain_finally_receives_root_value(self):
    """In async path, finally handler receives the root value."""
    received = {'value': None}
    root_obj = object()
    async def async_handler(v=None):
      received['value'] = v
    await await_(
      Chain(aempty, root_obj).then(lambda v: 'transformed').finally_(async_handler).run()
    )
    super(MyTestCase, self).assertIs(received['value'], root_obj)

  async def test_async_chain_finally_on_exception(self):
    """Async chain raises, finally handler still runs (lines 326-328)."""
    finally_called = {'value': False}
    async def async_handler(v=None):
      finally_called['value'] = True
    with self.assertRaises(TestExc):
      await await_(
        Chain(aempty, 1).then(raise_test_exc).finally_(async_handler).run()
      )
    super(MyTestCase, self).assertTrue(finally_called['value'])

  async def test_async_chain_finally_with_except_and_reraise(self):
    """Async chain: exception caught by except_(reraise=True), finally still runs."""
    except_called = {'value': False}
    finally_called = {'value': False}
    async def on_except(v=None):
      except_called['value'] = True
    async def on_finally(v=None):
      finally_called['value'] = True
    with self.assertRaises(TestExc):
      await await_(
        Chain(aempty, 1).then(raise_test_exc)
        .except_(on_except, reraise=True)
        .finally_(on_finally).run()
      )
    super(MyTestCase, self).assertTrue(except_called['value'])
    super(MyTestCase, self).assertTrue(finally_called['value'])

  async def test_async_chain_finally_does_not_alter_return(self):
    """Async path: finally handler return value discarded."""
    result = await await_(
      Chain(aempty, 5).then(lambda v: v * 4).finally_(lambda v: 'ignored').run()
    )
    super(MyTestCase, self).assertEqual(result, 20)

  async def test_async_chain_finally_handler_raises_lazy_ctx(self):
    """Async chain success, finally raises -> lazy _ExecCtx init (lines 333-340).

    When the async chain completes without exception, ctx is None. The finally
    handler raising triggers the `if ctx is None` lazy init branch at line 334.
    """
    async def bad_async_finally(v=None):
      raise FinallyExc('lazy ctx boom')
    with self.assertRaises(FinallyExc) as cm:
      await await_(
        Chain(aempty, 1).then(lambda v: v * 2).finally_(bad_async_finally).run()
      )
    super(MyTestCase, self).assertEqual(str(cm.exception), 'lazy ctx boom')

  async def test_async_chain_cascade_finally(self):
    """Cascade with async root: finally handler receives root value."""
    received = {'value': None}
    async def async_handler(v=None):
      received['value'] = v
    result = await await_(
      Cascade(aempty, 99).then(lambda v: v * 2).finally_(async_handler).run()
    )
    super(MyTestCase, self).assertEqual(result, 99)
    super(MyTestCase, self).assertEqual(received['value'], 99)


# ---------------------------------------------------------------------------
# Class 7: ExceptHandlerNullTests
# ---------------------------------------------------------------------------

class ExceptHandlerNullTests(MyTestCase):
  """Edge cases for except_ handler return values (lines 211-212 sync, 321-322 async).

  When except_(reraise=False) and the handler returns a value, that value
  becomes the chain result. If the handler returns None, the chain returns None.
  """

  async def test_except_handler_returns_none_sync(self):
    """Sync: handler returns None (implicitly), result is None."""
    def handler(v=None):
      pass  # returns None
    result = Chain(1).then(raise_test_exc).except_(handler, reraise=False).run()
    super(MyTestCase, self).assertIsNone(result)

  async def test_except_handler_returns_none_async(self):
    """Async: handler returns None (implicitly), result is None."""
    async def handler(v=None):
      pass  # returns None
    result = await await_(
      Chain(aempty, 1).then(raise_test_exc).except_(handler, reraise=False).run()
    )
    super(MyTestCase, self).assertIsNone(result)

  async def test_except_handler_returns_explicit_value(self):
    """Handler returns an explicit recovery value."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        def handler(v=None):
          return sentinel
        result = await await_(
          Chain(fn, 1).then(raise_test_exc).except_(handler, reraise=False).run()
        )
        super(MyTestCase, self).assertIs(result, sentinel)

  async def test_except_handler_returns_zero(self):
    """Handler returns 0 (falsy but not None/Null)."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(raise_test_exc)
          .except_(lambda v: 0, reraise=False).run()
        )
        super(MyTestCase, self).assertEqual(result, 0)

  async def test_except_handler_returns_empty_string(self):
    """Handler returns '' (falsy but not None/Null)."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(raise_test_exc)
          .except_(lambda v: '', reraise=False).run()
        )
        super(MyTestCase, self).assertEqual(result, '')

  async def test_except_handler_returns_false(self):
    """Handler returns False (falsy but not None/Null)."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1).then(raise_test_exc)
          .except_(lambda v: False, reraise=False).run()
        )
        super(MyTestCase, self).assertFalse(result)


# ---------------------------------------------------------------------------
# Class 8: FinallyDuplicateRegistrationTests
# ---------------------------------------------------------------------------

class FinallyDuplicateRegistrationTests(TestCase):
  """Registering a second finally_ handler raises QuentException (line 592-593)."""

  def test_duplicate_finally_raises(self):
    """Two finally_() calls on the same chain raise QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)

  def test_duplicate_finally_raises_with_cascade(self):
    """Two finally_() calls on Cascade also raise QuentException."""
    with self.assertRaises(QuentException):
      Cascade(1).finally_(lambda v: None).finally_(lambda v: None)


# ---------------------------------------------------------------------------
# Class 9: FinallyCloneTests
# ---------------------------------------------------------------------------

class FinallyCloneTests(MyTestCase):
  """Cloned chains independently execute finally handlers."""

  async def test_cloned_chain_has_independent_finally(self):
    """Cloning a chain with finally_ produces an independent finally handler."""
    for fn, ctx in self.with_fn():
      with ctx:
        call_log = {'original': 0, 'cloned': 0}
        def original_handler(v=None):
          call_log['original'] += 1
        c = Chain(fn, 1).then(lambda v: v * 2).finally_(original_handler)
        c2 = c.clone()
        await await_(c.run())
        super(MyTestCase, self).assertEqual(call_log['original'], 1)
        await await_(c2.run())
        super(MyTestCase, self).assertEqual(call_log['original'], 2)

  async def test_cloned_chain_can_add_new_finally_independently(self):
    """Clone does not share on_finally_link mutable state with original.

    Modifying the clone's finally handler does not affect the original because
    clone() creates a new Link via _clone_link.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        original_called = {'value': False}
        def original_handler(v=None):
          original_called['value'] = True
        c = Chain(fn, 1).then(lambda v: v * 2).finally_(original_handler)
        c2 = c.clone()
        await await_(c.run())
        super(MyTestCase, self).assertTrue(original_called['value'])
        original_called['value'] = False
        await await_(c2.run())
        # Both execute the same handler function (it was cloned from original)
        super(MyTestCase, self).assertTrue(original_called['value'])


# ---------------------------------------------------------------------------
# Class 10: FinallyWithNoAsyncTests
# ---------------------------------------------------------------------------

class FinallyWithNoAsyncTests(TestCase):
  """Finally handler behavior with no_async(True) (forces sync mode).

  When _is_sync is True, the `not self._is_sync` check at line 230 is False,
  so async coroutines from the finally handler are NOT scheduled as tasks
  and no RuntimeWarning is emitted. The coroutine is silently discarded.
  """

  def test_sync_finally_with_no_async(self):
    """Sync handler runs normally with no_async(True)."""
    called = {'value': False}
    def handler(v=None):
      called['value'] = True
    result = (
      Chain(1).then(lambda v: v * 2)
      .finally_(handler)
      .no_async(True)
      .run()
    )
    self.assertEqual(result, 2)
    self.assertTrue(called['value'])

  def test_async_handler_no_quent_warning_with_no_async(self):
    """With no_async(True), quent does NOT emit its own RuntimeWarning about
    scheduling the coroutine. The `not self._is_sync` guard at line 230
    prevents the coroutine scheduling and quent warning. Python itself may
    still emit 'coroutine was never awaited' warning.
    """
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(1).then(lambda v: v * 2).finally_(async_handler).no_async(True).run()
      quent_warnings = [
        x for x in w
        if issubclass(x.category, RuntimeWarning)
        and 'finally' in str(x.message).lower()
        and 'synchronous mode' in str(x.message).lower()
      ]
      # Quent's own warning about scheduling the coroutine should NOT appear
      self.assertEqual(len(quent_warnings), 0)
