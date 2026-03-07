"""Exhaustive tests for Chain.decorator(): calling conventions, all operations,
async paths, edge cases, and beyond-spec scenarios.
"""
from __future__ import annotations

import unittest

from quent import Chain, Null, QuentException
from quent._chain import _FrozenChain
from helpers import (
  async_fn,
  async_identity,
  sync_fn,
  sync_identity,
  SyncCM,
  AsyncCM,
)


class TestDecoratorCallingConventions(unittest.TestCase):

  def test_decorated_fn_receives_args_kwargs(self):
    received = {}

    @Chain().then(lambda x: x).decorator()
    def fn(a, b, c=0):
      received['a'] = a
      received['b'] = b
      received['c'] = c
      return a + b + c

    result = fn(1, 2, c=3)
    self.assertEqual(result, 6)
    self.assertEqual(received, {'a': 1, 'b': 2, 'c': 3})

  def test_decorated_fn_return_value_is_root(self):
    # fn's return becomes the root value of the chain, and then() operates on it.
    @Chain().then(lambda x: x * 10).decorator()
    def fn(n):
      return n + 1

    self.assertEqual(fn(4), 50)  # fn(4) = 5, 5 * 10 = 50

  def test_decorated_fn_no_args(self):
    @Chain().then(lambda x: x + 100).decorator()
    def fn():
      return 42

    self.assertEqual(fn(), 142)

  def test_decorated_fn_varargs(self):
    @Chain().then(lambda x: x).decorator()
    def fn(*args, **kwargs):
      return (args, kwargs)

    result = fn(1, 2, 3, key='val')
    self.assertEqual(result, ((1, 2, 3), {'key': 'val'}))

  def test_preserves_name_and_doc(self):
    @Chain().then(lambda x: x).decorator()
    def my_function(n):
      """My docstring here."""
      return n

    self.assertEqual(my_function.__name__, 'my_function')
    self.assertEqual(my_function.__doc__, 'My docstring here.')
    self.assertIs(my_function.__wrapped__, my_function.__wrapped__)  # functools.wraps sets __wrapped__
    self.assertTrue(hasattr(my_function, '__wrapped__'))

  def test_decorator_reusable(self):
    dec = Chain().then(lambda x: x * 2).decorator()

    @dec
    def fn_a(n):
      return n

    @dec
    def fn_b(n):
      return n + 10

    @dec
    def fn_c():
      return 100

    self.assertEqual(fn_a(5), 10)
    self.assertEqual(fn_b(5), 30)
    self.assertEqual(fn_c(), 200)
    self.assertEqual(fn_a.__name__, 'fn_a')
    self.assertEqual(fn_b.__name__, 'fn_b')
    self.assertEqual(fn_c.__name__, 'fn_c')

  def test_decorator_with_empty_chain(self):
    # Chain().decorator() -- chain has no steps, so fn's return is the result.
    @Chain().decorator()
    def fn(n):
      return n * 3

    self.assertEqual(fn(7), 21)


class TestDecoratorWithAllOperations(unittest.TestCase):

  def test_decorator_with_then(self):
    @Chain().then(lambda x: x + 1).then(lambda x: x * 2).decorator()
    def fn(n):
      return n

    self.assertEqual(fn(3), 8)  # (3+1)*2

  def test_decorator_with_do(self):
    tracker = []

    @Chain().do(lambda x: tracker.append(x)).then(lambda x: x * 2).decorator()
    def fn(n):
      return n

    result = fn(5)
    self.assertEqual(result, 10)
    self.assertEqual(tracker, [5])  # do runs side-effect but does not alter chain value

  def test_decorator_with_map(self):
    @Chain().map(lambda x: x * 2).decorator()
    def fn():
      return [1, 2, 3]

    self.assertEqual(fn(), [2, 4, 6])

  def test_decorator_with_filter(self):
    @Chain().filter(lambda x: x > 2).decorator()
    def fn():
      return [1, 2, 3, 4, 5]

    self.assertEqual(fn(), [3, 4, 5])

  def test_decorator_with_gather(self):
    @Chain().gather(lambda x: x + 1, lambda x: x * 2).decorator()
    def fn(n):
      return n

    self.assertEqual(fn(5), [6, 10])

  def test_decorator_with_with(self):
    @Chain().then(lambda _: SyncCM()).with_(lambda ctx: ctx + '_used').decorator()
    def fn():
      return 'ignored_root'

    result = fn()
    self.assertEqual(result, 'ctx_value_used')

  def test_decorator_with_except(self):
    @Chain().then(lambda x: 1 / 0).except_(lambda rv, e: 'handled').decorator()
    def fn(n):
      return n

    self.assertEqual(fn(42), 'handled')

  def test_decorator_with_finally(self):
    tracker = []

    @Chain().then(lambda x: x * 2).finally_(lambda rv: tracker.append(rv)).decorator()
    def fn(n):
      return n

    result = fn(5)
    self.assertEqual(result, 10)
    # finally_ receives root_value = fn's return (5)
    self.assertEqual(tracker, [5])

  def test_decorator_with_nested_chain(self):
    inner = Chain().then(lambda x: x + 100)

    @Chain().then(inner).decorator()
    def fn(n):
      return n

    self.assertEqual(fn(5), 105)

  def test_decorator_with_return(self):
    # Chain.return_() in a chain step causes early exit with the provided value.
    @Chain().then(lambda x: Chain.return_(x * 10)).then(lambda x: x + 999).decorator()
    def fn(n):
      return n

    self.assertEqual(fn(3), 30)  # return_(30) exits early, skips +999


class TestDecoratorAsync(unittest.IsolatedAsyncioTestCase):

  async def test_async_fn_with_sync_chain(self):
    # async fn produces a coroutine; chain transitions to async mode.
    @Chain().then(lambda x: x * 2).decorator()
    async def fn(n):
      return n + 1

    result = await fn(4)
    self.assertEqual(result, 10)  # fn(4)=5, 5*2=10

  async def test_sync_fn_with_async_chain(self):
    @Chain().then(async_fn).decorator()
    def fn(n):
      return n

    result = await fn(9)
    self.assertEqual(result, 10)  # async_fn(9) = 10

  async def test_async_fn_with_async_chain(self):
    @Chain().then(async_fn).then(async_fn).decorator()
    async def fn(n):
      return n

    result = await fn(5)
    self.assertEqual(result, 7)  # fn(5)=5, +1, +1

  async def test_decorated_async_preserves_name(self):
    @Chain().then(lambda x: x).decorator()
    async def my_async_func(n):
      """Async docstring."""
      return n

    self.assertEqual(my_async_func.__name__, 'my_async_func')
    self.assertEqual(my_async_func.__doc__, 'Async docstring.')


# --- Beyond-spec tests ---

class TestDecoratorOnLambda(unittest.TestCase):

  def test_decorator_on_lambda(self):
    # You cannot use @decorator syntax on a lambda, but you can call it manually.
    dec = Chain().then(lambda x: x * 3).decorator()
    fn = dec(lambda n: n + 1)
    self.assertEqual(fn(2), 9)  # lambda(2)=3, 3*3=9
    # Lambda is the wrapped function; functools.wraps copies __name__
    self.assertEqual(fn.__name__, '<lambda>')


class TestDecoratorWithChainModifyingArgs(unittest.TestCase):

  def test_chain_step_ignores_original_args(self):
    # Chain steps receive fn's return, not the original args.
    @Chain().then(lambda x: x + '_suffix').decorator()
    def fn(a, b):
      return a + b

    result = fn('hello', '_world')
    self.assertEqual(result, 'hello_world_suffix')


class TestDecoratorWhereDecoratedFnIsCoroutine(unittest.IsolatedAsyncioTestCase):

  async def test_coroutine_function_detected(self):
    import asyncio

    @Chain().then(lambda x: x * 2).decorator()
    async def fn(n):
      await asyncio.sleep(0)
      return n

    result = await fn(7)
    self.assertEqual(result, 14)


class TestDecoratorChaining(unittest.TestCase):

  def test_decorator_of_decorator(self):
    # Decorator 1: multiply by 2
    dec1 = Chain().then(lambda x: x * 2).decorator()
    # Decorator 2: add 100
    dec2 = Chain().then(lambda x: x + 100).decorator()

    # Apply dec2 first (outer), then dec1 (inner):
    # fn(5) = 5, dec1 chain: 5*2 = 10, dec2 chain: 10+100 = 110
    @dec2
    @dec1
    def fn(n):
      return n

    self.assertEqual(fn(5), 110)


class TestDecoratorWithFrozenChain(unittest.TestCase):

  def test_decorator_then_freeze(self):
    # Chain.freeze() returns a _FrozenChain which is not the same as Chain.
    # But decorator() returns a callable, not a chain, so freeze is not relevant.
    # Instead, test: decorator of a chain that includes a frozen chain as a step.
    frozen_step = Chain().then(lambda x: x * 3).freeze()

    @Chain().then(frozen_step).decorator()
    def fn(n):
      return n

    self.assertEqual(fn(4), 12)  # frozen_step(4) -> 4*3=12

  def test_decorator_called_many_times(self):
    # Stress: same decorator, many calls, each independent.
    @Chain().then(lambda x: x + 1).decorator()
    def fn(n):
      return n

    results = [fn(i) for i in range(100)]
    self.assertEqual(results, list(range(1, 101)))


class TestDecoratorWithIterate(unittest.TestCase):

  def test_decorator_with_iterate_in_chain(self):
    # Chain with a then step that uses a sub-chain iterate is unusual,
    # but we can test that map works inside a decorator.
    @Chain().map(lambda x: x ** 2).decorator()
    def fn():
      return [1, 2, 3, 4]

    self.assertEqual(fn(), [1, 4, 9, 16])


class TestDecoratorEdgeCases(unittest.TestCase):

  def test_decorated_fn_returns_none(self):
    @Chain().then(lambda x: x is None).decorator()
    def fn():
      return None

    self.assertTrue(fn())

  def test_decorated_fn_returns_false(self):
    @Chain().then(lambda x: not x).decorator()
    def fn():
      return False

    self.assertTrue(fn())

  def test_decorated_fn_returns_empty_list(self):
    @Chain().then(lambda x: len(x)).decorator()
    def fn():
      return []

    self.assertEqual(fn(), 0)

  def test_decorated_fn_raises(self):
    @Chain().decorator()
    def fn():
      raise RuntimeError('boom')

    with self.assertRaises(RuntimeError) as ctx:
      fn()
    self.assertIn('boom', str(ctx.exception))


if __name__ == '__main__':
  unittest.main()
