import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from tests.except_tests import (
  ExceptFinallyCheckSync, ExceptFinallyCheckAsync, raise_, Exc1, Exc2, Exc3,
)
from quent import Chain


class ExceptHandlerRaisesSync(TestCase):
  """Issue 51: except_ handler that raises a different exception (sync paths)."""

  def test_handler_raises_with_noraise(self):
    """When reraise=False and the handler itself raises, the handler's exception
    should propagate with the original exception chained as __cause__."""
    def handler_raises_type_error(v=None):
      raise TypeError('handler error')

    with self.assertRaises(TypeError) as cm:
      Chain(raise_, Exc1).except_(
        handler_raises_type_error, reraise=False,
      ).run()
    self.assertIsInstance(cm.exception.__cause__, Exc1)

  def test_handler_raises_with_raise(self):
    """When reraise=True (default) and the handler itself raises, the handler's
    exception should still propagate (not the original), chained via __cause__."""
    def handler_raises_runtime_error(v=None):
      raise RuntimeError('handler failed')

    with self.assertRaises(RuntimeError) as cm:
      Chain(raise_, Exc1).except_(
        handler_raises_runtime_error, reraise=True,
      ).run()
    self.assertIsInstance(cm.exception.__cause__, Exc1)

  def test_handler_raises_preserves_cause_type(self):
    """The __cause__ of the handler exception should be the exact original exception."""
    original = ValueError('original')
    def raise_original(v=None):
      raise original
    def handler_raises(v=None):
      raise TypeError('from handler')

    with self.assertRaises(TypeError) as cm:
      Chain(raise_original).except_(
        handler_raises, reraise=False,
      ).run()
    self.assertIs(cm.exception.__cause__, original)

  def test_handler_raises_with_exception_filter(self):
    """Handler that raises, limited to specific exception types via exceptions=."""
    def handler_raises(v=None):
      raise RuntimeError('handler')

    with self.assertRaises(RuntimeError) as cm:
      Chain(raise_, Exc1).except_(
        handler_raises, exceptions=Exc1, reraise=False,
      ).run()
    self.assertIsInstance(cm.exception.__cause__, Exc1)

  def test_handler_raises_filter_no_match_skips(self):
    """When the exception filter does not match, the handler is not called,
    and the original exception propagates unchanged."""
    def handler_raises(v=None):
      raise RuntimeError('should not be called')

    with self.assertRaises(Exc1):
      Chain(raise_, Exc1).except_(
        handler_raises, exceptions=Exc2, reraise=False,
      ).run()


class ExceptHandlerRaisesAsync(IsolatedAsyncioTestCase):
  """Issue 51: except_ handler that raises a different exception (async paths)."""

  async def test_async_handler_raises_with_noraise(self):
    """Async chain: handler raises with reraise=False -> handler exception with __cause__."""
    async def async_raise_exc1(v=None):
      raise Exc1('async original')

    def handler_raises_type_error(v=None):
      raise TypeError('async handler error')

    with self.assertRaises(TypeError) as cm:
      await Chain(aempty).then(async_raise_exc1).except_(
        handler_raises_type_error, reraise=False,
      ).run()
    self.assertIsInstance(cm.exception.__cause__, Exc1)

  async def test_async_handler_raises_with_raise(self):
    """Async chain: handler raises with reraise=True -> handler exception with __cause__."""
    async def async_raise_exc1(v=None):
      raise Exc1('async original')

    def handler_raises_runtime(v=None):
      raise RuntimeError('async handler failed')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(aempty).then(async_raise_exc1).except_(
        handler_raises_runtime, reraise=True,
      ).run()
    self.assertIsInstance(cm.exception.__cause__, Exc1)

  async def test_async_handler_is_coroutine_and_raises(self):
    """Async chain: async handler that raises -> handler exception with __cause__."""
    async def async_raise_exc1(v=None):
      raise Exc1('async original')

    async def async_handler_raises(v=None):
      raise TypeError('async handler coroutine error')

    with self.assertRaises(TypeError) as cm:
      await Chain(aempty).then(async_raise_exc1).except_(
        async_handler_raises, reraise=False,
      ).run()
    self.assertIsInstance(cm.exception.__cause__, Exc1)

  async def test_async_handler_raises_preserves_cause(self):
    """The __cause__ should be the exact original exception object."""
    original = Exc2('exact original')
    async def raise_original(v=None):
      raise original

    def handler_raises(v=None):
      raise TypeError('from handler')

    with self.assertRaises(TypeError) as cm:
      await Chain(aempty).then(raise_original).except_(
        handler_raises, reraise=False,
      ).run()
    self.assertIs(cm.exception.__cause__, original)


class FinallyRaisesSync(TestCase):
  """Issue 52: finally_ callback that raises (sync paths)."""

  def test_chain_succeeds_finally_raises(self):
    """Chain runs successfully, but finally_ raises -> finally_'s exception propagates."""
    def finally_raises(v=None):
      raise RuntimeError('finally error')

    with self.assertRaises(RuntimeError) as cm:
      Chain(empty, 42).finally_(finally_raises).run()
    self.assertEqual(str(cm.exception), 'finally error')

  def test_chain_fails_finally_raises(self):
    """Chain raises an exception AND finally_ raises -> finally_'s exception propagates.
    The original chain exception may appear as __context__."""
    def finally_raises(v=None):
      raise RuntimeError('finally error')

    with self.assertRaises(RuntimeError) as cm:
      Chain(raise_, Exc1).except_(
        lambda v=None: None, reraise=True,
      ).finally_(finally_raises).run()
    self.assertEqual(str(cm.exception), 'finally error')
    # The original exception should be the implicit context
    self.assertIsInstance(cm.exception.__context__, Exc1)

  def test_chain_fails_no_handler_finally_raises(self):
    """Chain raises with no except_ handler, finally_ also raises ->
    finally_'s exception propagates (original is __context__)."""
    def finally_raises(v=None):
      raise TypeError('finally override')

    with self.assertRaises(TypeError) as cm:
      Chain(raise_, Exc1).finally_(finally_raises).run()
    self.assertEqual(str(cm.exception), 'finally override')
    self.assertIsInstance(cm.exception.__context__, Exc1)

  def test_finally_raises_overrides_successful_result(self):
    """Even when the chain returns a value, if finally_ raises, the exception wins."""
    def finally_raises(v=None):
      raise ValueError('finally overrides')

    with self.assertRaises(ValueError):
      Chain(empty, 'success').finally_(finally_raises).run()

  def test_chain_fails_except_noraise_finally_raises(self):
    """Chain raises, except_ with reraise=False suppresses it, but finally_ raises ->
    finally_'s exception propagates."""
    def finally_raises(v=None):
      raise RuntimeError('finally after suppressed')

    with self.assertRaises(RuntimeError) as cm:
      Chain(raise_, Exc1).except_(
        lambda v=None: 'recovered', reraise=False,
      ).finally_(finally_raises).run()
    self.assertEqual(str(cm.exception), 'finally after suppressed')


class FinallyRaisesAsync(IsolatedAsyncioTestCase):
  """Issue 52: finally_ callback that raises (async paths)."""

  async def test_async_chain_succeeds_finally_raises(self):
    """Async chain succeeds, finally_ raises -> finally_'s exception propagates."""
    def finally_raises(v=None):
      raise RuntimeError('async finally error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(aempty, 42).finally_(finally_raises).run()
    self.assertEqual(str(cm.exception), 'async finally error')

  async def test_async_chain_fails_finally_raises(self):
    """Async chain raises AND finally_ raises -> finally_'s exception propagates."""
    async def async_raise_exc1(v=None):
      raise Exc1('async chain error')

    def finally_raises(v=None):
      raise RuntimeError('async finally error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(aempty).then(async_raise_exc1).except_(
        lambda v=None: None, reraise=True,
      ).finally_(finally_raises).run()
    self.assertEqual(str(cm.exception), 'async finally error')
    self.assertIsInstance(cm.exception.__context__, Exc1)

  async def test_async_chain_fails_no_handler_finally_raises(self):
    """Async chain raises with no handler, finally_ raises -> finally_ exception wins."""
    async def async_raise_exc1(v=None):
      raise Exc1('async original')

    def finally_raises(v=None):
      raise TypeError('async finally override')

    with self.assertRaises(TypeError) as cm:
      await Chain(aempty).then(async_raise_exc1).finally_(finally_raises).run()
    self.assertEqual(str(cm.exception), 'async finally override')
    self.assertIsInstance(cm.exception.__context__, Exc1)

  async def test_async_finally_coroutine_raises(self):
    """Async chain with an async finally_ callback that raises."""
    async def async_finally_raises(v=None):
      raise RuntimeError('async coroutine finally error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(aempty, 42).finally_(async_finally_raises).run()
    self.assertEqual(str(cm.exception), 'async coroutine finally error')

  async def test_async_chain_fails_async_finally_raises(self):
    """Async chain raises AND async finally_ raises -> finally_'s exception propagates."""
    async def async_raise_exc1(v=None):
      raise Exc1('async chain')

    async def async_finally_raises(v=None):
      raise RuntimeError('async finally coro error')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(aempty).then(async_raise_exc1).finally_(async_finally_raises).run()
    self.assertEqual(str(cm.exception), 'async finally coro error')
    self.assertIsInstance(cm.exception.__context__, Exc1)

  async def test_async_chain_succeeds_async_finally_raises(self):
    """Async chain succeeds, async finally_ raises -> exception propagates."""
    async def async_finally_raises(v=None):
      raise ValueError('async finally overrides success')

    with self.assertRaises(ValueError):
      await Chain(aempty, 'success').finally_(async_finally_raises).run()

  async def test_async_except_noraise_finally_raises(self):
    """Async: chain raises, except_ suppresses with reraise=False, finally_ raises."""
    async def async_raise_exc1(v=None):
      raise Exc1('suppressed')

    async def async_finally_raises(v=None):
      raise RuntimeError('finally after async suppress')

    with self.assertRaises(RuntimeError) as cm:
      await Chain(aempty).then(async_raise_exc1).except_(
        lambda v=None: 'recovered', reraise=False,
      ).finally_(async_finally_raises).run()
    self.assertEqual(str(cm.exception), 'finally after async suppress')
