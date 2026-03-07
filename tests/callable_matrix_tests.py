"""Tests for callable type matrix: every callable TYPE across every chain operation.

Ensures _evaluate_value handles regular functions, async functions, lambdas,
builtins, functools.partial, class constructors, callable objects, async callable
objects, bound methods, nested chains, and frozen chains correctly in then/do/
except_/finally_ and as root values. Also covers literal (non-callable) values
passed to then().
"""
from __future__ import annotations

import functools
import operator
import unittest

from quent import Chain, Null, QuentException
from helpers import (
  sync_fn, async_fn, CallableObj, AsyncCallableObj,
  BoundMethodHolder, partial_fn, Adder,
)


# ---------------------------------------------------------------------------
# then() -- callable matrix (sync)
# ---------------------------------------------------------------------------

class TestThenCallableMatrix(unittest.TestCase):

  def test_then_sync_fn(self):
    result = Chain(5).then(sync_fn).run()
    self.assertEqual(result, 6)

  def test_then_lambda(self):
    result = Chain(5).then(lambda x: x + 1).run()
    self.assertEqual(result, 6)

  def test_then_builtin_str(self):
    result = Chain(42).then(str).run()
    self.assertEqual(result, '42')

  def test_then_builtin_int(self):
    result = Chain('42').then(int).run()
    self.assertEqual(result, 42)

  def test_then_builtin_len(self):
    result = Chain([1, 2, 3]).then(len).run()
    self.assertEqual(result, 3)

  def test_then_partial(self):
    result = Chain(5).then(partial_fn).run()
    self.assertEqual(result, 15)

  def test_then_class_constructor(self):
    result = Chain(5).then(Adder).run()
    self.assertIsInstance(result, Adder)
    self.assertEqual(result.x, 5)

  def test_then_callable_obj(self):
    result = Chain(5).then(CallableObj()).run()
    self.assertEqual(result, 6)

  def test_then_bound_method(self):
    result = Chain(5).then(BoundMethodHolder().method).run()
    self.assertEqual(result, 6)

  def test_then_chain(self):
    inner = Chain().then(lambda x: x + 1)
    result = Chain(5).then(inner).run()
    self.assertEqual(result, 6)

  def test_then_frozen_chain(self):
    frozen = Chain().then(lambda x: x + 1).freeze()
    result = Chain(5).then(frozen).run()
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# then() -- callable matrix (async)
# ---------------------------------------------------------------------------

class TestThenCallableMatrixAsync(unittest.IsolatedAsyncioTestCase):

  async def test_then_async_fn(self):
    result = await Chain(5).then(async_fn).run()
    self.assertEqual(result, 6)

  async def test_then_async_callable_obj(self):
    result = await Chain(5).then(AsyncCallableObj()).run()
    self.assertEqual(result, 6)


# ---------------------------------------------------------------------------
# do() -- callable matrix (sync)
# ---------------------------------------------------------------------------

class TestDoCallableMatrix(unittest.TestCase):

  def test_do_sync_fn(self):
    result = Chain(5).do(sync_fn).run()
    self.assertEqual(result, 5)

  def test_do_lambda(self):
    result = Chain(5).do(lambda x: x * 100).run()
    self.assertEqual(result, 5)

  def test_do_callable_obj(self):
    result = Chain(5).do(CallableObj()).run()
    self.assertEqual(result, 5)

  def test_do_bound_method(self):
    result = Chain(5).do(BoundMethodHolder().method).run()
    self.assertEqual(result, 5)

  def test_do_partial(self):
    result = Chain(5).do(partial_fn).run()
    self.assertEqual(result, 5)


# ---------------------------------------------------------------------------
# do() -- callable matrix (async)
# ---------------------------------------------------------------------------

class TestDoCallableMatrixAsync(unittest.IsolatedAsyncioTestCase):

  async def test_do_async_fn(self):
    result = await Chain(5).do(async_fn).run()
    self.assertEqual(result, 5)

  async def test_do_async_callable_obj(self):
    result = await Chain(5).do(AsyncCallableObj()).run()
    self.assertEqual(result, 5)


# ---------------------------------------------------------------------------
# except_() -- callable matrix (sync)
# ---------------------------------------------------------------------------

def _raise_value_error(x=None):
  raise ValueError('boom')


class TestExceptCallableMatrix(unittest.TestCase):

  def test_except_sync_fn(self):
    def handler(rv, exc):
      return f'caught:{type(exc).__name__}'
    result = Chain(_raise_value_error).except_(handler).run()
    self.assertEqual(result, 'caught:ValueError')

  def test_except_lambda(self):
    result = Chain(_raise_value_error).except_(lambda rv, exc: 'handled').run()
    self.assertEqual(result, 'handled')

  def test_except_callable_obj(self):
    class ExcHandler:
      def __call__(self, rv, exc):
        return f'obj:{type(exc).__name__}'
    result = Chain(_raise_value_error).except_(ExcHandler()).run()
    self.assertEqual(result, 'obj:ValueError')

  def test_except_partial(self):
    def _handler(prefix, rv, exc):
      return f'{prefix}:{type(exc).__name__}'
    handler = functools.partial(_handler, 'pfx')
    result = Chain(_raise_value_error).except_(handler).run()
    self.assertEqual(result, 'pfx:ValueError')


# ---------------------------------------------------------------------------
# except_() -- callable matrix (async)
# ---------------------------------------------------------------------------

async def _async_raise_value_error(x=None):
  raise ValueError('boom')


class TestExceptCallableMatrixAsync(unittest.IsolatedAsyncioTestCase):

  async def test_except_async_fn(self):
    async def handler(rv, exc):
      return f'async_caught:{type(exc).__name__}'
    result = await Chain(_async_raise_value_error).except_(handler).run()
    self.assertEqual(result, 'async_caught:ValueError')

  async def test_except_async_callable_obj(self):
    class AsyncExcHandler:
      async def __call__(self, rv, exc):
        return f'async_obj:{type(exc).__name__}'
    result = await Chain(_async_raise_value_error).except_(AsyncExcHandler()).run()
    self.assertEqual(result, 'async_obj:ValueError')


# ---------------------------------------------------------------------------
# finally_() -- callable matrix (sync)
# ---------------------------------------------------------------------------

class TestFinallyCallableMatrix(unittest.TestCase):

  def test_finally_sync_fn(self):
    tracker = []
    def handler(root_val):
      tracker.append(root_val)
    result = Chain(42).finally_(handler).run()
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [42])

  def test_finally_lambda(self):
    tracker = []
    result = Chain(42).finally_(lambda rv: tracker.append(rv)).run()
    self.assertEqual(result, 42)
    self.assertEqual(tracker, [42])

  def test_finally_callable_obj(self):
    class FinallyHandler:
      def __init__(self):
        self.called_with = None
      def __call__(self, root_val):
        self.called_with = root_val
    handler = FinallyHandler()
    result = Chain(42).finally_(handler).run()
    self.assertEqual(result, 42)
    self.assertEqual(handler.called_with, 42)


# ---------------------------------------------------------------------------
# finally_() -- callable matrix (async)
# ---------------------------------------------------------------------------

class TestFinallyCallableMatrixAsync(unittest.IsolatedAsyncioTestCase):

  async def test_finally_async_fn(self):
    tracker = []
    async def handler(root_val):
      tracker.append(root_val)
    result = await Chain(42).then(async_fn).finally_(handler).run()
    self.assertEqual(result, 43)
    self.assertEqual(tracker, [42])

  async def test_finally_async_callable_obj(self):
    tracker = []
    class AsyncFinallyHandler:
      async def __call__(self, root_val):
        tracker.append(root_val)
    result = await Chain(42).then(async_fn).finally_(AsyncFinallyHandler()).run()
    self.assertEqual(result, 43)
    self.assertEqual(tracker, [42])


# ---------------------------------------------------------------------------
# then() -- literal (non-callable) values
# ---------------------------------------------------------------------------

class TestThenLiteralValues(unittest.TestCase):

  def test_then_int(self):
    result = Chain(5).then(42).run()
    self.assertEqual(result, 42)

  def test_then_float(self):
    result = Chain(5).then(3.14).run()
    self.assertEqual(result, 3.14)

  def test_then_string(self):
    result = Chain(5).then('hello').run()
    self.assertEqual(result, 'hello')

  def test_then_none(self):
    result = Chain(5).then(None).run()
    self.assertIsNone(result)

  def test_then_bool(self):
    result = Chain(5).then(True).run()
    self.assertIs(result, True)

  def test_then_list(self):
    result = Chain(5).then([1, 2]).run()
    self.assertEqual(result, [1, 2])

  def test_then_dict(self):
    result = Chain(5).then({'a': 1}).run()
    self.assertEqual(result, {'a': 1})

  def test_then_set(self):
    result = Chain(5).then({1, 2}).run()
    self.assertEqual(result, {1, 2})

  def test_then_tuple(self):
    result = Chain(5).then((1, 2)).run()
    self.assertEqual(result, (1, 2))

  def test_then_bytes(self):
    result = Chain(5).then(b'hello').run()
    self.assertEqual(result, b'hello')

  def test_then_frozenset(self):
    result = Chain(5).then(frozenset([1])).run()
    self.assertEqual(result, frozenset([1]))


# ---------------------------------------------------------------------------
# Root value -- callable matrix
# ---------------------------------------------------------------------------

class TestRootCallableMatrix(unittest.TestCase):

  def test_root_sync_fn(self):
    result = Chain(sync_fn, 5).run()
    self.assertEqual(result, 6)

  def test_root_lambda(self):
    result = Chain(lambda: 42).run()
    self.assertEqual(result, 42)

  def test_root_callable_obj(self):
    result = Chain(CallableObj(), 5).run()
    self.assertEqual(result, 6)

  def test_root_class_constructor(self):
    result = Chain(Adder, 5).run()
    self.assertIsInstance(result, Adder)
    self.assertEqual(result.x, 5)


if __name__ == '__main__':
  unittest.main()
