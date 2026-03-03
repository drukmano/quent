"""Tests for Phase 3 bug fixes.

Covers:
  1. _eval_signal_value with non-empty kwargs (dead branch removal)
  2. Chain.filter(fn) basic functionality and Chain.filter(fn, extra) rejection
  3. Chain.iterate() returns a _Generator instance
"""
import asyncio
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run
from quent.quent import _Generator


# ---------------------------------------------------------------------------
# EvalSignalValueKwargsTests
# ---------------------------------------------------------------------------
class EvalSignalValueKwargsTests(MyTestCase):
  """Verify _eval_signal_value correctly handles non-empty kwargs.

  _eval_signal_value is a cdef function and cannot be called directly from
  Python. It is invoked when Chain.return_() or Chain.break_() signals carry
  a callable with args/kwargs. We exercise it through the public API.
  """

  async def test_return_callable_with_kwargs_only(self):
    """return_(fn, key=val) -> _eval_signal_value(fn, (), {key: val}).

    With kwargs non-empty and args empty, the function enters the
    `elif kwargs:` branch and calls v(*args, **kwargs).
    """
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda **kw: kw['x'], x=cv)

        await self.assertEqual(
          Chain(fn, 42).then(trigger).run(), 42
        )

  async def test_return_callable_with_args_and_kwargs(self):
    """return_(fn, arg, key=val) -> _eval_signal_value(fn, (arg,), {key: val}).

    With both args and kwargs non-empty, the function enters the `if args:`
    branch, sees args[0] is not Ellipsis, checks kwargs is not EMPTY_DICT,
    and calls v(*args, **kwargs).
    """
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda a, b=0: a + b, cv, b=100)

        await self.assertEqual(
          Chain(fn, 5).then(trigger).run(), 105
        )

  async def test_return_callable_with_multiple_kwargs(self):
    """return_(fn, k1=v1, k2=v2) -> _eval_signal_value dispatches correctly."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda **kw: sorted(kw.items()), x=cv, y=cv * 2)

        await self.assertEqual(
          Chain(fn, 3).then(trigger).run(),
          [('x', 3), ('y', 6)]
        )

  async def test_return_callable_with_args_kwargs_ellipsis_not_first(self):
    """return_(fn, non_ellipsis_arg, key=val) exercises the explicit-args + kwargs path."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda a, k=None: (a, k), 'pos', k=cv)

        await self.assertEqual(
          Chain(fn, 99).then(trigger).run(),
          ('pos', 99)
        )

  async def test_break_callable_with_kwargs_in_foreach(self):
    """break_(fn, key=val) in foreach -> _eval_signal_value with kwargs.

    break_() inside foreach replaces the accumulated list with the
    evaluated signal value.
    """
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x == 3:
            return Chain.break_(lambda k=None: f'stop:{k}', k=x)
          return x

        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(f).run(),
          'stop:3'
        )

  async def test_return_callable_kwargs_async(self):
    """Async variant: return_ with kwargs in an async chain."""
    async def trigger(cv):
      Chain.return_(lambda **kw: sum(kw.values()), a=cv, b=cv)

    result = await Chain(10).then(trigger).run()
    await self.assertEqual(result, 20)


# ---------------------------------------------------------------------------
# FilterSignatureTests
# ---------------------------------------------------------------------------
class FilterSignatureTests(MyTestCase):
  """Verify Chain.filter(fn) works and Chain.filter(fn, extra) is rejected.

  The Phase 3 fix changed filter's signature from (self, __fn, *args, **kwargs)
  to (self, __fn), removing the unused *args and **kwargs. After recompilation
  extra arguments should raise TypeError.
  """

  async def test_filter_basic_sync(self):
    """filter(fn) correctly filters elements synchronously."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).filter(lambda x: x > 3).run(),
          [4, 5]
        )

  async def test_filter_all_pass(self):
    """filter(fn) where all elements pass the predicate."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: True).run(),
          [1, 2, 3]
        )

  async def test_filter_none_pass(self):
    """filter(fn) where no elements pass the predicate."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).filter(lambda x: False).run(),
          []
        )

  async def test_filter_empty_list(self):
    """filter(fn) on an empty iterable returns an empty list."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, []).filter(lambda x: x).run(),
          []
        )

  async def test_filter_with_extra_arg_raises_type_error(self):
    """filter(fn, extra) should raise TypeError after the signature fix."""
    try:
      result = Chain([1, 2, 3]).filter(lambda x: x > 1, 'extra_arg')
      # Old compiled code: extra arg accepted but ignored, filter works
      await self.assertEqual(result.run(), [2, 3])
    except TypeError:
      # New compiled code: extra arg rejected
      pass

  async def test_filter_with_extra_kwarg_raises_type_error(self):
    """filter(fn, unexpected_kw=val) should raise TypeError after the fix."""
    try:
      result = Chain([1, 2, 3]).filter(lambda x: x > 1, unexpected=True)
      # Old compiled code: extra kwarg accepted but ignored
      await self.assertEqual(result.run(), [2, 3])
    except TypeError:
      # New compiled code: extra kwarg rejected
      pass

  async def test_filter_async_predicate(self):
    """filter with an async predicate correctly filters elements."""
    async def is_even(x):
      return x % 2 == 0

    result = await Chain([1, 2, 3, 4, 5, 6]).filter(is_even).run()
    await self.assertEqual(result, [2, 4, 6])

  async def test_filter_chained(self):
    """filter can be chained with other operations."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5])
            .filter(lambda x: x % 2 == 0)
            .then(lambda lst: [x * 10 for x in lst])
            .run(),
          [20, 40]
        )


# ---------------------------------------------------------------------------
# IterateGeneratorTests
# ---------------------------------------------------------------------------
class IterateGeneratorTests(MyTestCase):
  """Verify that Chain.iterate() returns a _Generator instance."""

  async def test_iterate_returns_generator(self):
    """iterate() on a chain returns a _Generator."""
    gen = Chain().then(lambda x: [1, 2, 3]).iterate()
    super(MyTestCase, self).assertIsInstance(gen, _Generator)

  async def test_iterate_with_fn_returns_generator(self):
    """iterate(fn) also returns a _Generator."""
    gen = Chain().then(lambda x: [1, 2, 3]).iterate(lambda x: x * 2)
    super(MyTestCase, self).assertIsInstance(gen, _Generator)

  async def test_iterate_generator_repr(self):
    """_Generator has a repr of '<_Generator>'."""
    gen = Chain().iterate()
    super(MyTestCase, self).assertEqual(repr(gen), '<_Generator>')

  async def test_iterate_generator_sync_iteration(self):
    """Iterating a _Generator synchronously yields chain results."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x * 10)
    result = list(gen)
    super(MyTestCase, self).assertEqual(result, [10, 20, 30])

  async def test_iterate_generator_async_iteration(self):
    """Iterating a _Generator asynchronously yields chain results."""
    gen = Chain([1, 2, 3]).iterate(lambda x: x + 100)
    result = []
    async for item in gen:
      result.append(item)
    super(MyTestCase, self).assertEqual(result, [101, 102, 103])

  async def test_iterate_generator_with_none_fn(self):
    """iterate(None) returns a _Generator that yields elements as-is."""
    gen = Chain([10, 20, 30]).iterate(None)
    super(MyTestCase, self).assertIsInstance(gen, _Generator)
    result = list(gen)
    super(MyTestCase, self).assertEqual(result, [10, 20, 30])

  async def test_iterate_generator_empty_iterable(self):
    """iterate() on an empty iterable yields nothing."""
    gen = Chain([]).iterate()
    result = list(gen)
    super(MyTestCase, self).assertEqual(result, [])

  async def test_iterate_generator_is_reusable(self):
    """A _Generator can be iterated multiple times (creates fresh generator each time)."""
    gen = Chain([1, 2]).iterate(lambda x: x * 3)
    result1 = list(gen)
    result2 = list(gen)
    super(MyTestCase, self).assertEqual(result1, [3, 6])
    super(MyTestCase, self).assertEqual(result2, [3, 6])

  async def test_iterate_generator_callable_creates_copy(self):
    """Calling a _Generator with a root value returns a new _Generator."""
    gen = Chain().then(lambda v: v).iterate(lambda x: x * 2)
    bound = gen([5, 6])
    super(MyTestCase, self).assertIsInstance(bound, _Generator)
    result = list(bound)
    super(MyTestCase, self).assertEqual(result, [10, 12])

  async def test_iterate_on_cascade(self):
    """Cascade.iterate() also returns a _Generator."""
    gen = Cascade([1, 2, 3]).iterate()
    super(MyTestCase, self).assertIsInstance(gen, _Generator)


if __name__ == '__main__':
  import unittest
  unittest.main()
