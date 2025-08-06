import sys
import warnings
import traceback
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class InnerExc(TestExc):
  pass


class OuterExc(TestExc):
  pass


class CustomBaseExc(BaseException):
  pass


class CustomBaseExc2(BaseException):
  pass


def raise_inner(v=None):
  raise InnerExc('inner')


def raise_from(v=None):
  try:
    raise InnerExc('cause')
  except InnerExc as e:
    raise OuterExc('effect') from e


def raise_implicit_chain(v=None):
  try:
    raise InnerExc('original')
  except InnerExc:
    raise OuterExc('during handling')


def raise_suppress_context(v=None):
  try:
    raise InnerExc('suppressed')
  except InnerExc:
    raise OuterExc('no context shown') from None


def raise_system_exit(v=None):
  raise SystemExit(42)


def raise_keyboard_interrupt(v=None):
  raise KeyboardInterrupt('interrupted')


def raise_custom_base(v=None):
  raise CustomBaseExc('custom base')


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------

class MyTestCase(IsolatedAsyncioTestCase):
  def with_fn(self):
    for fn in [empty, aempty]:
      yield fn, self.subTest(fn=fn)

  async def assertTrue(self, expr, msg=None):
    return super().assertTrue(await await_(expr), msg)

  async def assertFalse(self, expr, msg=None):
    return super().assertFalse(await await_(expr), msg)

  async def assertEqual(self, first, second, msg=None):
    return super().assertEqual(await await_(first), second, msg)

  async def assertIsNone(self, obj, msg=None):
    return super().assertIsNone(await await_(obj), msg)

  async def assertIs(self, expr1, expr2, msg=None):
    return super().assertIs(await await_(expr1), expr2, msg)

  async def assertIsNot(self, expr1, expr2, msg=None):
    return super().assertIsNot(await await_(expr1), expr2, msg)


# ---------------------------------------------------------------------------
# ExceptionChainingTests
# ---------------------------------------------------------------------------

class ExceptionChainingTests(MyTestCase):

  async def test_explicit_chaining_cause_preserved(self):
    """raise X from Y preserves __cause__ through a chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(OuterExc) as cm:
          await await_(Chain(fn, 1).then(raise_from).run())
        super(MyTestCase, self).assertIsNotNone(cm.exception.__cause__)
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, InnerExc)

  async def test_explicit_chaining_cause_message(self):
    """raise X from Y preserves the cause exception message."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(OuterExc) as cm:
          await await_(Chain(fn, 1).then(raise_from).run())
        super(MyTestCase, self).assertEqual(str(cm.exception.__cause__), 'cause')

  async def test_implicit_chaining_context_preserved(self):
    """Implicit exception chaining preserves __context__ through a chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(OuterExc) as cm:
          await await_(Chain(fn, 1).then(raise_implicit_chain).run())
        super(MyTestCase, self).assertIsNotNone(cm.exception.__context__)
        super(MyTestCase, self).assertIsInstance(cm.exception.__context__, InnerExc)

  async def test_implicit_chaining_context_message(self):
    """Implicit chaining preserves the context exception message."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(OuterExc) as cm:
          await await_(Chain(fn, 1).then(raise_implicit_chain).run())
        super(MyTestCase, self).assertEqual(str(cm.exception.__context__), 'original')

  async def test_suppress_context_flag_preserved(self):
    """raise X from None sets __suppress_context__ = True, preserved through chain."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(OuterExc) as cm:
          await await_(Chain(fn, 1).then(raise_suppress_context).run())
        super(MyTestCase, self).assertTrue(cm.exception.__suppress_context__)

  async def test_suppress_context_still_has_context(self):
    """raise X from None still sets __context__ (just suppressed in display)."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(OuterExc) as cm:
          await await_(Chain(fn, 1).then(raise_suppress_context).run())
        # __context__ is set even when __suppress_context__ is True
        super(MyTestCase, self).assertIsNotNone(cm.exception.__context__)

  async def test_except_handler_raises_sets_cause(self):
    """When except_ handler raises, original exception becomes __cause__."""
    for fn, ctx in self.with_fn():
      with ctx:
        def handler(v=None):
          raise OuterExc('from handler')
        with self.assertRaises(OuterExc) as cm:
          await await_(
            Chain(fn, 1).then(raise_inner).except_(handler, reraise=False).run()
          )
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, InnerExc)
        super(MyTestCase, self).assertEqual(str(cm.exception), 'from handler')

  async def test_chained_exceptions_in_nested_chain(self):
    """Exception chaining is preserved through nested chains."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner_chain = Chain().then(raise_from)
        with self.assertRaises(OuterExc) as cm:
          await await_(Chain(fn, 1).then(inner_chain).run())
        super(MyTestCase, self).assertIsNotNone(cm.exception.__cause__)
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, InnerExc)

  async def test_cascade_preserves_exception_chaining(self):
    """Exception chaining is preserved in Cascade chains."""
    for fn, ctx in self.with_fn():
      with ctx:
        with self.assertRaises(OuterExc) as cm:
          await await_(Cascade(fn, 1).then(raise_from).run())
        super(MyTestCase, self).assertIsNotNone(cm.exception.__cause__)
        super(MyTestCase, self).assertIsInstance(cm.exception.__cause__, InnerExc)


# ---------------------------------------------------------------------------
# BaseExceptionTests
# ---------------------------------------------------------------------------

class BaseExceptionTests(MyTestCase):

  async def test_system_exit_propagates_through_chain(self):
    """SystemExit propagates through a chain without being caught by default except_."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        with self.assertRaises(SystemExit):
          await await_(
            Chain(fn, 1).then(raise_system_exit).except_(handler).run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_keyboard_interrupt_propagates_through_chain(self):
    """KeyboardInterrupt propagates through a chain without being caught by default except_."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        with self.assertRaises(KeyboardInterrupt):
          await await_(
            Chain(fn, 1).then(raise_keyboard_interrupt).except_(handler).run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_custom_base_exception_not_caught_by_default(self):
    """Custom BaseException subclass is not caught by default except_."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        with self.assertRaises(CustomBaseExc):
          await await_(
            Chain(fn, 1).then(raise_custom_base).except_(handler).run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_except_with_base_exception_catches_system_exit(self):
    """except_ with exceptions=BaseException catches SystemExit."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1)
            .then(raise_system_exit)
            .except_(handler, exceptions=BaseException)
            .run()
          )
        except SystemExit:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_with_base_exception_catches_keyboard_interrupt(self):
    """except_ with exceptions=BaseException catches KeyboardInterrupt."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1)
            .then(raise_keyboard_interrupt)
            .except_(handler, exceptions=BaseException)
            .run()
          )
        except KeyboardInterrupt:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_with_base_exception_catches_custom(self):
    """except_ with exceptions=BaseException catches custom BaseException subclasses."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1)
            .then(raise_custom_base)
            .except_(handler, exceptions=BaseException)
            .run()
          )
        except CustomBaseExc:
          pass
        super(MyTestCase, self).assertTrue(called[0])

  async def test_except_with_specific_base_exception_class(self):
    """except_ with exceptions=SystemExit catches SystemExit but not KeyboardInterrupt."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        try:
          await await_(
            Chain(fn, 1)
            .then(raise_system_exit)
            .except_(handler, exceptions=SystemExit)
            .run()
          )
        except SystemExit:
          pass
        super(MyTestCase, self).assertTrue(called[0])

        # KeyboardInterrupt should NOT be caught by SystemExit handler
        called[0] = False
        with self.assertRaises(KeyboardInterrupt):
          await await_(
            Chain(fn, 1)
            .then(raise_keyboard_interrupt)
            .except_(handler, exceptions=SystemExit)
            .run()
          )
        super(MyTestCase, self).assertFalse(called[0])

  async def test_base_exception_suppress_with_noraise(self):
    """except_ with exceptions=BaseException and reraise=False suppresses BaseException."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = await await_(
          Chain(fn, 1)
          .then(raise_custom_base)
          .except_(lambda v: None, exceptions=BaseException, reraise=False)
          .run()
        )
        super(MyTestCase, self).assertIsNone(result)

  async def test_base_exception_in_cascade(self):
    """BaseException propagates through Cascade without being caught by default except_."""
    for fn, ctx in self.with_fn():
      with ctx:
        called = [False]
        def handler(v=None):
          called[0] = True
        with self.assertRaises(CustomBaseExc):
          await await_(
            Cascade(fn, 1).then(raise_custom_base).except_(handler).run()
          )
        super(MyTestCase, self).assertFalse(called[0])


# ---------------------------------------------------------------------------
# WarningTests
# ---------------------------------------------------------------------------

class WarningTests(MyTestCase):

  async def test_async_except_on_sync_chain_emits_runtime_warning(self):
    """Async except_ handler on sync chain emits RuntimeWarning about coroutine scheduling."""
    async def async_handler(v=None):
      pass
    import asyncio
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      try:
        Chain(raise_inner).except_(async_handler, reraise=True).run()
      except InnerExc:
        pass
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_async_finally_on_sync_chain_emits_runtime_warning(self):
    """Async finally_ handler on sync chain emits RuntimeWarning about coroutine scheduling."""
    async def async_finally(v=None):
      pass
    import asyncio
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      Chain(empty, 1).finally_(async_finally).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
    await asyncio.sleep(0.1)

  async def test_warning_message_content_except(self):
    """RuntimeWarning for async except_ contains meaningful description."""
    async def async_handler(v=None):
      pass
    import asyncio
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      try:
        Chain(raise_inner).except_(async_handler, reraise=True).run()
      except InnerExc:
        pass
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
      msg = str(runtime_warnings[0].message)
      super(MyTestCase, self).assertIn('coroutine', msg.lower())
    await asyncio.sleep(0.1)

  async def test_warning_message_content_finally(self):
    """RuntimeWarning for async finally_ contains meaningful description."""
    async def async_finally(v=None):
      pass
    import asyncio
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      Chain(empty, 1).finally_(async_finally).run()
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertGreater(len(runtime_warnings), 0)
      msg = str(runtime_warnings[0].message)
      super(MyTestCase, self).assertIn('coroutine', msg.lower())
    await asyncio.sleep(0.1)

  async def test_no_warning_for_sync_except_on_sync_chain(self):
    """Sync except_ handler on sync chain does NOT emit RuntimeWarning."""
    def sync_handler(v=None):
      pass
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      try:
        Chain(raise_inner).except_(sync_handler, reraise=True).run()
      except InnerExc:
        pass
      runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
      super(MyTestCase, self).assertEqual(len(runtime_warnings), 0)


# ---------------------------------------------------------------------------
# TracebackExceptionPatchTests
# ---------------------------------------------------------------------------

class TracebackExceptionPatchTests(TestCase):

  def test_patch_is_installed(self):
    """TracebackException.__init__ is patched by quent."""
    from quent import __init__ as quent_init_module
    te_init = traceback.TracebackException.__init__
    self.assertIsNot(te_init, object.__init__)

  def test_quent_exception_format_works(self):
    """TracebackException can format a quent-raised exception without error."""
    try:
      Chain(1).then(raise_inner).run()
    except InnerExc as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      self.assertIn('InnerExc', formatted)

  def test_normal_exception_unaffected_by_patch(self):
    """Non-quent exceptions still format correctly through the patched init."""
    exc = ValueError('normal error')
    try:
      raise exc
    except ValueError:
      pass
    te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
    formatted = ''.join(te.format())
    self.assertIn('ValueError', formatted)
    self.assertIn('normal error', formatted)

  def test_chained_exception_formats_with_cause(self):
    """TracebackException formats chained exceptions (raise X from Y) correctly."""
    try:
      Chain(1).then(raise_from).run()
    except OuterExc as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      # Both the cause and the effect should appear in the formatted output
      self.assertIn('InnerExc', formatted)
      self.assertIn('OuterExc', formatted)

  def test_patch_handles_none_exc_value(self):
    """Patched TracebackException.__init__ handles None exc_value gracefully."""
    # The patch checks `if exc_value is not None` before accessing __quent__
    try:
      te = traceback.TracebackException(ValueError, None, None)
      # Should not raise
    except Exception:
      self.fail('Patched TracebackException.__init__ should handle None exc_value')

  def test_quent_frame_present_in_formatted_traceback(self):
    """Formatted traceback includes the synthetic <quent> frame."""
    try:
      Chain(1).then(raise_inner).run()
    except InnerExc as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      self.assertIn('<quent>', formatted)

  def test_implicit_chain_formats_both_exceptions(self):
    """Implicit exception chaining appears in formatted traceback output."""
    try:
      Chain(1).then(raise_implicit_chain).run()
    except OuterExc as exc:
      te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
      formatted = ''.join(te.format())
      self.assertIn('OuterExc', formatted)
      self.assertIn('InnerExc', formatted)
