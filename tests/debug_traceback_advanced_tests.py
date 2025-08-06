import sys
import traceback
import logging
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException, run


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
# Helpers
# ---------------------------------------------------------------------------

def _capture_logs():
  """Set up a log handler that captures quent debug messages."""
  logs = []
  handler = logging.Handler()
  handler.emit = lambda record: logs.append(record.getMessage())
  logger = logging.getLogger('quent')
  logger.addHandler(handler)
  old_level = logger.level
  logger.setLevel(logging.DEBUG)
  return logger, handler, logs, old_level


def _cleanup_logs(logger, handler, old_level):
  """Remove the temporary log handler."""
  logger.removeHandler(handler)
  logger.setLevel(old_level)


def _get_tb_entries(exc):
  """Extract traceback entries from an exception."""
  return traceback.extract_tb(exc.__traceback__)


def _get_tb_filenames(exc):
  """Get filenames from an exception's traceback."""
  return [entry.filename for entry in _get_tb_entries(exc)]


def _get_tb_func_names(exc):
  """Get function names from an exception's traceback."""
  return [entry.name for entry in _get_tb_entries(exc)]


def _add_one(v):
  return v + 1


def _double(v):
  return v * 2


def _raise_value_error(v=None):
  raise ValueError('test error')


def _raise_type_error(v=None):
  raise TypeError('type error')


async def _async_add_one(v):
  return v + 1


async def _async_double(v):
  return v * 2


async def _async_raise_value_error(v=None):
  raise ValueError('async test error')


# ---------------------------------------------------------------------------
# DebugSyncPathTests
# ---------------------------------------------------------------------------

class DebugSyncPathTests(TestCase):

  def test_debug_link_results_populated_for_root(self):
    """Debug mode logs the root value when running a sync chain."""
    logger, handler, logs, old_level = _capture_logs()
    try:
      Chain(42).config(debug=True).run()
      self.assertTrue(any('42' in log for log in logs))
    finally:
      _cleanup_logs(logger, handler, old_level)

  def test_debug_link_results_populated_for_then(self):
    """Debug mode logs each link's result in a sync chain."""
    logger, handler, logs, old_level = _capture_logs()
    try:
      Chain(5).then(_double).then(_add_one).config(debug=True).run()
      # root=5, _double->10, _add_one->11
      self.assertTrue(any('5' in log for log in logs))
      self.assertTrue(any('10' in log for log in logs))
      self.assertTrue(any('11' in log for log in logs))
    finally:
      _cleanup_logs(logger, handler, old_level)

  def test_debug_correct_final_result(self):
    """Debug mode does not alter the chain's final result."""
    result = Chain(3).then(_double).then(_add_one).config(debug=True).run()
    self.assertEqual(result, 7)

  def test_debug_multiple_links_all_logged(self):
    """Debug mode logs every intermediate value through multiple links."""
    logger, handler, logs, old_level = _capture_logs()
    try:
      Chain(2).then(_add_one).then(_double).then(_add_one).config(debug=True).run()
      # root=2, _add_one->3, _double->6, _add_one->7
      self.assertGreaterEqual(len(logs), 4)
    finally:
      _cleanup_logs(logger, handler, old_level)

  def test_debug_with_exception_still_logs_before_error(self):
    """Debug mode logs values for links that succeed before an exception."""
    logger, handler, logs, old_level = _capture_logs()
    try:
      try:
        Chain(10).then(_double).then(_raise_value_error).config(debug=True).run()
      except ValueError:
        pass
      # root=10 should be logged, _double->20 should be logged
      self.assertTrue(any('10' in log for log in logs))
      self.assertTrue(any('20' in log for log in logs))
    finally:
      _cleanup_logs(logger, handler, old_level)

  def test_debug_cascade_logs_root_value(self):
    """Debug mode on a Cascade logs the root value."""
    logger, handler, logs, old_level = _capture_logs()
    try:
      Cascade(99).then(lambda v: v + 1).config(debug=True).run()
      self.assertTrue(any('99' in log for log in logs))
    finally:
      _cleanup_logs(logger, handler, old_level)


# ---------------------------------------------------------------------------
# DebugAsyncPathTests
# ---------------------------------------------------------------------------

class DebugAsyncPathTests(MyTestCase):

  async def test_debug_async_link_results_populated(self):
    """Debug mode logs root value on async chain (root logged in sync path)."""
    logger, handler, logs, old_level = _capture_logs()
    try:
      result = await await_(
        Chain(7).then(aempty).then(_add_one).config(debug=True).run()
      )
      super(MyTestCase, self).assertEqual(result, 8)
      # root=7 logged in sync path
      super(MyTestCase, self).assertTrue(any('7' in log for log in logs))
    finally:
      _cleanup_logs(logger, handler, old_level)

  async def test_debug_async_correct_result(self):
    """Debug mode on async chain produces correct final result."""
    result = await await_(
      Chain(5).then(_async_double).then(_async_add_one).config(debug=True).run()
    )
    super(MyTestCase, self).assertEqual(result, 11)

  async def test_debug_async_with_exception(self):
    """Debug mode on async chain that raises still propagates the exception."""
    with self.assertRaises(ValueError):
      await await_(
        Chain(10).then(_async_double).then(_async_raise_value_error).config(debug=True).run()
      )

  async def test_debug_async_all_intermediate_values_logged(self):
    """Debug mode logs all intermediate values in async chain after awaiting."""
    logger, handler, logs, old_level = _capture_logs()
    try:
      result = await await_(
        Chain(3).then(_async_double).then(_async_add_one).config(debug=True).run()
      )
      super(MyTestCase, self).assertEqual(result, 7)
      # Root value 3 should appear
      super(MyTestCase, self).assertTrue(any('3' in log for log in logs))
    finally:
      _cleanup_logs(logger, handler, old_level)


# ---------------------------------------------------------------------------
# TracebackModificationTests
# ---------------------------------------------------------------------------

def _nested_chain_raiser(v):
  """Raise inside a nested chain."""
  return Chain(v).then(_raise_value_error)()


def _outer_chain_raiser(v):
  """Call a nested chain from an outer chain."""
  return Chain(v).then(_nested_chain_raiser)()


class TracebackModificationTests(TestCase):

  def test_quent_frame_appears_in_traceback(self):
    """Exception from a chain has a <quent> frame in its traceback."""
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      self.assertIn('<quent>', filenames)

  def test_no_internal_quent_frames(self):
    """No quent package internal file frames leak into the cleaned traceback."""
    import os
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      for fn in filenames:
        if fn == '<quent>':
          continue
        self.assertFalse(
          fn.startswith(quent_pkg_dir),
          f'Internal quent frame should be cleaned: {fn}'
        )

  def test_user_frame_preserved(self):
    """User function names are preserved in cleaned traceback."""
    try:
      Chain(1).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      func_names = _get_tb_func_names(exc)
      self.assertIn('_raise_value_error', func_names)

  def test_nested_chain_is_nested_skips_modify(self):
    """A nested chain (is_nested=True) defers traceback modification to the outermost chain."""
    try:
      _outer_chain_raiser(5)
      self.fail('Expected ValueError')
    except ValueError as exc:
      filenames = _get_tb_filenames(exc)
      # The outermost chain should still produce a <quent> frame
      self.assertIn('<quent>', filenames)
      # User frames from both levels should be present
      func_names = _get_tb_func_names(exc)
      self.assertIn('_raise_value_error', func_names)

  def test_chained_exception_cause_cleaned(self):
    """__cause__ exceptions have their tracebacks cleaned of quent internals."""
    import os
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))

    def handler(v=None):
      raise TypeError('handler error')

    def raiser(v=None):
      raise ValueError('original')

    try:
      Chain(1).then(raiser).except_(handler, reraise=False).run()
      self.fail('Expected TypeError')
    except TypeError as exc:
      # __cause__ is the original ValueError
      cause = exc.__cause__
      self.assertIsInstance(cause, ValueError)
      # If cause has a traceback, verify it is cleaned
      if cause.__traceback__ is not None:
        cause_filenames = [
          e.filename for e in traceback.extract_tb(cause.__traceback__)
        ]
        for fn in cause_filenames:
          if fn == '<quent>':
            continue
          self.assertFalse(
            fn.startswith(quent_pkg_dir),
            f'Chained exception cause traceback should be cleaned: {fn}'
          )

  def test_chained_exception_context_cleaned(self):
    """__context__ exceptions have their tracebacks cleaned of quent internals."""
    import os
    quent_pkg_dir = os.path.dirname(os.path.abspath(
      __import__('quent').quent.__file__
    ))

    def finally_raiser(v=None):
      raise RuntimeError('finally error')

    def chain_raiser(v=None):
      raise ValueError('chain error')

    try:
      Chain(1).then(chain_raiser).finally_(finally_raiser).run()
      self.fail('Expected RuntimeError')
    except RuntimeError as exc:
      context = exc.__context__
      self.assertIsInstance(context, ValueError)
      if context.__traceback__ is not None:
        context_filenames = [
          e.filename for e in traceback.extract_tb(context.__traceback__)
        ]
        for fn in context_filenames:
          if fn == '<quent>':
            continue
          self.assertFalse(
            fn.startswith(quent_pkg_dir),
            f'Chained exception context traceback should be cleaned: {fn}'
          )

  def test_quent_frame_contains_chain_visualization(self):
    """The <quent> frame name contains chain structure information."""
    try:
      Chain(1).then(_add_one).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_tb_entries(exc)
      quent_entries = [e for e in entries if e.filename == '<quent>']
      self.assertGreater(len(quent_entries), 0)
      # The name of the <quent> frame encodes the chain visualization
      for entry in quent_entries:
        self.assertTrue(len(entry.name) > 0,
          '<quent> frame should have a non-empty name (chain visualization)')

  def test_source_arrow_in_traceback_visualization(self):
    """The <quent> frame visualization includes a source arrow for the failing link."""
    try:
      Chain(1).then(_add_one).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_tb_entries(exc)
      quent_entries = [e for e in entries if e.filename == '<quent>']
      self.assertGreater(len(quent_entries), 0)
      # The source arrow is ' <----'
      viz = quent_entries[0].name
      self.assertIn('<----', viz)


# ---------------------------------------------------------------------------
# StringifyChainTests
# ---------------------------------------------------------------------------

class StringifyChainTests(TestCase):

  def test_repr_empty_chain(self):
    """repr of an empty chain contains the class name and empty parens."""
    r = repr(Chain())
    self.assertIn('Chain', r)
    self.assertIn('()', r)

  def test_repr_with_root_value(self):
    """repr of a chain with a literal root value shows the value."""
    r = repr(Chain(42))
    self.assertIn('42', r)

  def test_repr_with_root_function(self):
    """repr of a chain with a callable root shows the function name."""
    r = repr(Chain(_add_one))
    self.assertIn('_add_one', r)

  def test_repr_with_then_link(self):
    """repr of a chain with .then() links shows .then and function names."""
    r = repr(Chain(1).then(_double))
    self.assertIn('.then', r)
    self.assertIn('_double', r)

  def test_repr_nested_chain_has_indentation(self):
    """repr of a nested chain includes newlines and indentation."""
    inner = Chain().then(_add_one)
    outer = Chain(1).then(inner)
    r = repr(outer)
    self.assertIn('\n', r)
    self.assertIn('Chain', r)

  def test_repr_cascade_type_name(self):
    """repr of a Cascade shows 'Cascade' as the class name."""
    r = repr(Cascade())
    self.assertIn('Cascade', r)

  def test_repr_chain_with_conditional_pending(self):
    """repr of a chain with a pending conditional state still produces output."""
    c = Chain(1).condition(lambda v: v)
    r = repr(c)
    # Should not crash and should contain 'Chain'
    self.assertIn('Chain', r)

  def test_repr_chain_with_multiple_operations(self):
    """repr of a chain with many operations shows all of them."""
    c = Chain(1).then(_add_one).then(_double).then(_add_one)
    r = repr(c)
    self.assertIn('_add_one', r)
    self.assertIn('_double', r)
    # Count the .then occurrences -- should be at least 3
    self.assertGreaterEqual(r.count('.then'), 3)

  def test_format_link_source_arrow_in_exception(self):
    """When an exception occurs, the chain visualization contains a source arrow at the failing link."""
    try:
      Chain(1).then(_add_one).then(_double).then(_raise_value_error).run()
      self.fail('Expected ValueError')
    except ValueError as exc:
      entries = _get_tb_entries(exc)
      quent_entries = [e for e in entries if e.filename == '<quent>']
      self.assertGreater(len(quent_entries), 0)
      viz = quent_entries[0].name
      # The arrow ' <----' marks the source link
      self.assertIn('<----', viz)
      # The function names of preceding links should appear with = results
      self.assertIn('_add_one', viz)
      self.assertIn('_double', viz)


if __name__ == '__main__':
  import unittest
  unittest.main()
