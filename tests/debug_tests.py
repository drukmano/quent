import unittest
import logging
from quent import Chain, Cascade


class DebugModeTests(unittest.TestCase):
  def test_debug_via_config(self):
    c = Chain(1).then(lambda v: v + 1).config(debug=True)
    self.assertEqual(c.run(), 2)

  def test_debug_logs_output(self):
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      c = Chain(5).then(lambda v: v * 2).config(debug=True)
      c.run()
      # Should have at least 2 log messages (root + then)
      self.assertGreaterEqual(len(logs), 2)
      # Check that root value was logged
      self.assertTrue(any('5' in log for log in logs))
      # Check that result was logged
      self.assertTrue(any('10' in log for log in logs))
    finally:
      logger.removeHandler(handler)

  def test_debug_disabled_no_logs(self):
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      c = Chain(5).then(lambda v: v * 2)
      c.run()
      self.assertEqual(len(logs), 0)
    finally:
      logger.removeHandler(handler)

  def test_debug_clone_preserved(self):
    c = Chain(1).config(debug=True)
    c2 = c.clone()
    self.assertEqual(c2.run(), 1)

  def test_debug_cascade(self):
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      c = Cascade(5).then(lambda v: v * 2).config(debug=True)
      c.run()
      self.assertGreater(len(logs), 0)
    finally:
      logger.removeHandler(handler)

  def test_debug_with_fn_name(self):
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      c = Chain(5).then(lambda v: v + 1).config(debug=True)
      c.run()
      # Check that 'then' appears in some log
      self.assertTrue(any('then' in log for log in logs))
    finally:
      logger.removeHandler(handler)


class DebugAsyncTests(unittest.IsolatedAsyncioTestCase):
  async def test_debug_async(self):
    logs = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())
    logger = logging.getLogger('quent')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
      async def double(v):
        return v * 2
      c = Chain(5).then(double).config(debug=True)
      result = await c.run()
      self.assertEqual(result, 10)
      # Root value should be logged at least
      self.assertGreater(len(logs), 0)
    finally:
      logger.removeHandler(handler)


if __name__ == '__main__':
  unittest.main()
