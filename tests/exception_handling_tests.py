import asyncio
import sys
import traceback
import warnings
from unittest import TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, QuentException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Exc1(TestExc):
  pass


class Exc2(TestExc):
  pass


class Exc3(Exc2):
  pass


class CustomBaseExc(BaseException):
  pass


def raise_exc(exc_type=TestExc):
  raise exc_type()


# ---------------------------------------------------------------------------
# Class 1: ExceptBasicTests
# ---------------------------------------------------------------------------

class ExceptBasicTests(MyTestCase):

  async def test_except_catches_matching(self):
    """except_(handler, exceptions=ValueError) catches ValueError."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        def raiser(v=None):
          raise ValueError('boom')
        try:
          await await_(
            Chain(fn, 1).then(raiser).except_(handler, exceptions=ValueError).run()
          )
        except ValueError:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_skips_nonmatching(self):
    """except_(handler, exceptions=TypeError) does not catch ValueError."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        def raiser(v=None):
          raise ValueError('boom')
        with self.assertRaises(ValueError):
          await await_(
            Chain(fn, 1).then(raiser).except_(handler, exceptions=TypeError).run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_except_default_catches_exception(self):
    """except_(handler) catches any Exception subclass."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        def raiser(v=None):
          raise RuntimeError('generic')
        try:
          await await_(
            Chain(fn, 1).then(raiser).except_(handler).run()
          )
        except RuntimeError:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_default_does_not_catch_base_exception(self):
    """except_(handler) does NOT catch BaseException subclasses like CustomBaseExc."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        def raiser(v=None):
          raise CustomBaseExc('base')
        with self.assertRaises(CustomBaseExc):
          await await_(
            Chain(fn, 1).then(raiser).except_(handler).run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_except_with_iterable_of_exceptions(self):
    """except_(handler, exceptions=[ValueError, TypeError]) catches both."""
    for fn, ctx in self.with_fn():
      with ctx:
        for exc_type in [ValueError, TypeError]:
          called = [False]
          def handler(v=None):
            called[0] = True
          def raiser(v=None):
            raise exc_type('iterable test')
          try:
            await await_(
              Chain(fn, 1).then(raiser).except_(handler, exceptions=[ValueError, TypeError]).run()
            )
          except (ValueError, TypeError):
            pass
          super(MyTestCase, self).assertTrue(called[0])

  async def test_except_with_string_raises_type_error(self):
    """except_(handler, exceptions='ValueError') raises TypeError."""
    with self.assertRaises(TypeError):
      Chain(1).except_(lambda v: None, exceptions='ValueError')

  async def test_except_reraise_true_default(self):
    """Handler runs, then exception re-raised (reraise=True is default)."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        def raiser(v=None):
          raise TestExc('reraise default')
        with self.assertRaises(TestExc):
          await await_(
            Chain(fn, 1).then(raiser).except_(handler).run()
          )
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_reraise_false_suppresses(self):
    """Handler runs, exception suppressed (reraise=False)."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        def raiser(v=None):
          raise TestExc('suppressed')
        try:
          await await_(
            Chain(fn, 1).then(raiser).except_(handler, reraise=False).run()
          )
        except TestExc:
          super(MyTestCase, self).fail('Exception should have been suppressed')
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_return_value_with_noraise(self):
    """except_(handler, obj, reraise=False) returns obj as chain result."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()
        called = [False]
        def handler(v=None):
          called[0] = True
          return v
        def raiser(v=None):
          raise TestExc('return obj')
        result = await await_(
          Chain(fn, 1).then(raiser).except_(handler, sentinel, reraise=False).run()
        )
        super(MyTestCase, self).assertIs(result, sentinel)
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_handler_receives_root_value(self):
    """Handler is called with root value, not exception."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        root_obj = object()
        def handler(v=None):
          received[0] = v
        def raiser(v=None):
          raise TestExc('root value')
        try:
          await await_(
            Chain(fn, root_obj).then(raiser).except_(handler).run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertIs(received[0], root_obj)

  async def test_except_no_exception_handler_not_called(self):
    """Handler not invoked when chain succeeds."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        result = await await_(
          Chain(fn, 42).then(lambda v: v + 1).except_(handler).run()
        )
        super(MyTestCase, self).assertEqual(result, 43)
        super(MyTestCase, self).assertFalse(called[0])


# ---------------------------------------------------------------------------
# Class 2: ExceptMultipleHandlersTests
# ---------------------------------------------------------------------------

class ExceptMultipleHandlersTests(MyTestCase):

  async def test_multiple_handlers_first_match_wins(self):
    """Multiple except_ links, first matching one executes."""
    for fn, ctx in self.with_fn():
      with ctx:
        called1 = [False]
        called2 = [False]
        def handler1(v=None):
          called1[0] = True
        def handler2(v=None):
          called2[0] = True
        def raiser(v=None):
          raise Exc1('first match')
        try:
          await await_(
            Chain(fn, 1).then(raiser)
            .except_(handler1, exceptions=TestExc)
            .except_(handler2, exceptions=TestExc)
            .run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertTrue(called1[0])
        super(MyTestCase, self).assertFalse(called2[0])

  async def test_multiple_handlers_skip_nonmatching(self):
    """Non-matching handlers skipped, matching one executes."""
    for fn, ctx in self.with_fn():
      with ctx:
        called1 = [False]
        called2 = [False]
        called3 = [False]
        def handler1(v=None):
          called1[0] = True
        def handler2(v=None):
          called2[0] = True
        def handler3(v=None):
          called3[0] = True
        def raiser(v=None):
          raise Exc1('skip nonmatching')
        try:
          await await_(
            Chain(fn, 1).then(raiser)
            .except_(handler1, exceptions=Exc2)
            .except_(handler2, exceptions=Exc1)
            .except_(handler3, exceptions=TestExc)
            .run()
          )
        except TestExc:
          pass
        super(MyTestCase, self).assertFalse(called1[0])
        super(MyTestCase, self).assertTrue(called2[0])
        super(MyTestCase, self).assertFalse(called3[0])


# ---------------------------------------------------------------------------
# Class 3: ExceptHandlerRaisesTests
# ---------------------------------------------------------------------------

class ExceptHandlerRaisesTests(MyTestCase):

  async def test_handler_raises_with_noraise(self):
    """Handler raises TypeError, reraise=False -> TypeError propagates with original as __cause__."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise TypeError('handler error')
        def raiser(v=None):
          raise Exc1('original')
        with self.assertRaises(TypeError) as cm:
          await await_(
            Chain(fn, 1).then(raiser).except_(handler, reraise=False).run()
          )
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, Exc1)

  async def test_handler_raises_with_reraise(self):
    """Handler raises TypeError, reraise=True -> TypeError propagates with original as __cause__."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise TypeError('handler error')
        def raiser(v=None):
          raise Exc1('original')
        with self.assertRaises(TypeError) as cm:
          await await_(
            Chain(fn, 1).then(raiser).except_(handler, reraise=True).run()
          )
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, Exc1)

  async def test_async_handler_on_sync_chain_reraise_warns(self):
    """Async except handler on sync chain with reraise=True -> RuntimeWarning."""
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      try:
        Chain(raise_exc).except_(async_handler, reraise=True).run()
      except TestExc:
        pass
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Class 4: FinallyTests
# ---------------------------------------------------------------------------

class FinallyTests(MyTestCase):

  async def test_finally_runs_on_success(self):
    """Finally callback invoked after successful chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        await await_(Chain(fn, 42).finally_(handler).run())
        super(MyTestCase, self).assertTrue(called[0])

  async def test_finally_runs_on_exception(self):
    """Finally callback invoked even when exception occurs."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        def raiser(v=None):
          raise TestExc('finally on exc')
        try:
          await await_(Chain(fn, 1).then(raiser).finally_(handler).run())
        except TestExc:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_finally_receives_root_value(self):
    """Finally called with root value."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = [None]
        root_obj = object()
        def handler(v=None):
          received[0] = v
        await await_(Chain(fn, root_obj).finally_(handler).run())
        super(MyTestCase, self).assertIs(received[0], root_obj)

  async def test_finally_duplicate_raises(self):
    """Two finally_() calls raise QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: None).finally_(lambda v: None)

  async def test_finally_raises_overrides_success(self):
    """Chain succeeds, finally raises -> finally's exception propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        def finally_raises(v=None):
          raise RuntimeError('finally error')
        with self.assertRaises(RuntimeError) as cm:
          await await_(Chain(fn, 42).finally_(finally_raises).run())
        super(MyTestCase, self).assertEqual(str(cm.exception), 'finally error')

  async def test_finally_raises_overrides_chain_exception(self):
    """Chain raises, finally raises -> finally's exception propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        def finally_raises(v=None):
          raise RuntimeError('finally override')
        def raiser(v=None):
          raise Exc1('chain error')
        with self.assertRaises(RuntimeError) as cm:
          await await_(
            Chain(fn, 1).then(raiser).finally_(finally_raises).run()
          )
        super(MyTestCase, self).assertEqual(str(cm.exception), 'finally override')
        super(MyTestCase, self).assertIsInstance(cm.exception.__context__, Exc1)

  async def test_finally_with_internal_quent_exception(self):
    """Chain.return_() inside finally -> QuentException about control flow signals."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(QuentException) as cm:
          await await_(
            Chain(fn, 1).finally_(Chain.return_, 99).run()
          )
        super(MyTestCase, self).assertIn(
          'control flow signals', str(cm.exception).lower()
        )

  async def test_finally_async_on_sync_chain_warns(self):
    """Async finally on sync chain emits RuntimeWarning."""
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      Chain(empty, 1).finally_(async_handler).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_finally_on_empty_chain(self):
    """Chain().finally_(handler).run() — handler called."""
    called = [False]
    def handler(v=None):
      called[0] = True
    result = Chain().finally_(handler).run()
    super(MyTestCase, self).assertIsNone(result)
    super(MyTestCase, self).assertTrue(called[0])


# ---------------------------------------------------------------------------
# Class 5: TracebackTests
# ---------------------------------------------------------------------------

def _tb_helper_step1(v):
  return Chain(v).then(_tb_helper_step2)()


def _tb_helper_step2(v):
  return _tb_helper_raise(v)


def _tb_helper_raise(v=None):
  raise ValueError('traceback test')


class TracebackTests(TestCase):

  def test_quent_frame_in_traceback(self):
    """Exception traceback contains a <quent> frame."""
    try:
      Chain(1).then(_tb_helper_step1).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = traceback.extract_tb(exc.__traceback__)
      filenames = [e.filename for e in entries]
      self.assertIn('<quent>', filenames)

  def test_no_internal_frames_in_traceback(self):
    """No quent/ internal file frames leak through (helpers, custom)."""
    try:
      Chain(1).then(_tb_helper_step1).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = traceback.extract_tb(exc.__traceback__)
      filenames = [e.filename for e in entries]
      for fn in filenames:
        if fn == '<quent>':
          continue
        self.assertNotIn(
          'quent/helpers', fn,
          f'Internal quent helpers frame should be cleaned: {fn}'
        )
        self.assertNotIn(
          'quent/custom', fn,
          f'Internal quent custom frame should be cleaned: {fn}'
        )

  def test_user_frames_preserved(self):
    """User function names present in traceback."""
    try:
      Chain(1).then(_tb_helper_step1).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = traceback.extract_tb(exc.__traceback__)
      func_names = [e.name for e in entries]
      self.assertIn('_tb_helper_raise', func_names)
      self.assertIn('_tb_helper_step1', func_names)

  def test_quent_excepthook_quent_exception(self):
    """_quent_excepthook cleans exceptions with __quent__ attr."""
    from quent.quent import _quent_excepthook

    # Verify the hook is installed as sys.excepthook.
    self.assertTrue(callable(_quent_excepthook))
    self.assertIs(sys.excepthook, _quent_excepthook)

    # Create an exception with __quent__ attribute and a traceback
    exc = ValueError('quent exception')
    exc.__quent__ = True
    try:
      raise exc
    except ValueError:
      pass

    # The exception should have __quent__ set
    self.assertTrue(getattr(exc, '__quent__', False))

    # The excepthook recognizes __quent__ and calls _clean_chained_exceptions.
    # We verify this works without error by calling it with the
    # _original_excepthook redirected to a no-op to avoid printing to stderr.
    from quent import quent as quent_module
    saved = quent_module._original_excepthook
    hook_called_with = [None]
    def mock_original(et, ev, etb):
      hook_called_with[0] = (et, ev, etb)
    quent_module._original_excepthook = mock_original
    try:
      _quent_excepthook(ValueError, exc, exc.__traceback__)
    finally:
      quent_module._original_excepthook = saved

    # The mock should have been called
    self.assertIsNotNone(hook_called_with[0])
    self.assertIs(hook_called_with[0][0], ValueError)
    self.assertIs(hook_called_with[0][1], exc)

  def test_quent_excepthook_non_quent_exception(self):
    """Non-quent exceptions pass through unchanged by the hook."""
    from quent.quent import _quent_excepthook

    # Create a non-quent exception
    exc = ValueError('non-quent exception')
    try:
      raise exc
    except ValueError:
      pass
    tb = exc.__traceback__

    # Verify __quent__ is not present
    self.assertFalse(getattr(exc, '__quent__', False))

    # Call the hook with a non-quent exception
    from quent import quent as quent_module
    saved = quent_module._original_excepthook
    hook_called_with = [None]
    def mock_original(et, ev, etb):
      hook_called_with[0] = (et, ev, etb)
    quent_module._original_excepthook = mock_original
    try:
      _quent_excepthook(ValueError, exc, tb)
    finally:
      quent_module._original_excepthook = saved

    # The mock should have been called with the original traceback unchanged
    self.assertIsNotNone(hook_called_with[0])
    self.assertIs(hook_called_with[0][2], tb)

  def test_traceback_exception_patch(self):
    """traceback.TracebackException.__init__ is patched and cleans quent exceptions."""
    # Verify the patch is in place by testing it handles __quent__ attribute
    exc = ValueError('patch test')
    exc.__quent__ = True
    try:
      raise exc
    except ValueError:
      pass

    # TracebackException should be constructible without error when __quent__ is set
    te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
    self.assertIsNotNone(te)

    # Also verify that non-quent exceptions work normally
    exc2 = ValueError('normal')
    try:
      raise exc2
    except ValueError:
      pass
    te2 = traceback.TracebackException(type(exc2), exc2, exc2.__traceback__)
    self.assertIsNotNone(te2)

    # Verify the patch handles a chain exception correctly
    try:
      Chain(1).then(_tb_helper_raise).run()
    except ValueError as chain_exc:
      te3 = traceback.TracebackException(
        type(chain_exc), chain_exc, chain_exc.__traceback__
      )
      formatted = ''.join(te3.format())
      # Should contain user function names
      self.assertIn('_tb_helper_raise', formatted)


# ---------------------------------------------------------------------------
# Class 6: WarningTests
# ---------------------------------------------------------------------------

class WarningTests(MyTestCase):

  async def test_async_except_on_sync_chain_warns(self):
    """Async except handler on sync chain emits RuntimeWarning."""
    async def async_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      try:
        Chain(raise_exc).except_(async_handler, reraise=True).run()
      except TestExc:
        pass
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_async_finally_on_sync_chain_warns(self):
    """Async finally on sync chain emits RuntimeWarning."""
    async def async_finally(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      Chain(empty, 1).finally_(async_finally).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)
