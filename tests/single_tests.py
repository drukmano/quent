import asyncio
import contextlib
import inspect
import time
from contextlib import contextmanager, asynccontextmanager
from tests.flex_context import FlexContext
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import throw_if, empty, aempty, await_
from tests.try_except_tests import assertRaisesSync, assertRaisesAsync
from quent import Chain, ChainAttr, Cascade, CascadeAttr, QuentException, run


class SingleTest(IsolatedAsyncioTestCase):
  async def test_simple_chain(self):
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertIsNone(await await_(Cascade().run()))
        self.assertIsNone(await await_(Cascade().then(True).run()))
        self.assertIsNone(await await_(Cascade().then(fn, True).run()))

        self.assertTrue(await await_(Chain(fn, True).run()))
        self.assertTrue(await await_(Chain().run(fn, True)))
        self.assertIsNone(await await_(Chain(fn, True).then(None).run()))
        self.assertIsNone(await await_(Chain().then(None).run(fn, True)))

        self.assertTrue(await await_(Cascade(fn, True).run()))
        self.assertTrue(await await_(Cascade().run(fn, True)))
        self.assertTrue(await await_(Cascade(fn, True).then(None).run()))
        self.assertTrue(await await_(Cascade().then(None).run(fn, True)))

        self.assertTrue(await await_(Chain(empty, True).then(fn).run()))
        self.assertTrue(await await_(Chain().then(fn).run(empty, True)))

        self.assertTrue(await await_(Cascade(empty, True).then(fn).then(None).run()))
        self.assertTrue(await await_(Cascade().then(None).then(fn).run(empty, True)))
