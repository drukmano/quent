from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_
from src.quent import Chain, ChainR, Cascade, CascadeR, QuentException


class MiscTest(IsolatedAsyncioTestCase):
  # TODO write tests that test async functions down the line, not as root value

  async def test_empty_root(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain().then(1).then(fn).root().then(lambda v: v+10).eq(10).run(0)))

  async def test_root_value_async(self):
    self.assertTrue(await Chain(aempty, 5).eq(5).run())
    self.assertTrue(await Chain().eq(5).run(aempty, 5))

  async def test_run_without_root(self):
    self.assertRaises(QuentException, Chain().then(empty).run)
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertIsNone(await await_(Cascade().then(fn).run()))
        self.assertIsNone(await await_(Cascade().then(fn, 1).run()))

  async def test_override_empty_value(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertTrue(await await_(Chain().then(fn).eq(1).run(1)))

  async def test_override_root_value(self):
      self.assertRaises(QuentException, Chain(1).then(empty).run, 1)

  async def test_misc_1(self):
    self.assertIsNone(Cascade().root().run())
    self.assertIsNone(Cascade().then(lambda: 5).then(lambda: 6).run())
    self.assertIn('6 links', str(Chain(5).then(lambda v: v).eq(5).else_(10).in_([10]).else_(False).not_()))
