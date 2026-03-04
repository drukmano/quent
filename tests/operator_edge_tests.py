import asyncio
from itertools import product
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException, Null


# ---------------------------------------------------------------------------
# EvalSignalValueTests
# ---------------------------------------------------------------------------
class EvalSignalValueTests(IsolatedAsyncioTestCase):
  """Tests for all _eval_signal_value branches (lines 81-101 of _operators.pxi).

  _eval_signal_value is called from handle_return_exc and handle_break_exc.
  We exercise each branch via Chain.return_() and Chain.break_() within chains.
  """

  # --- return_ paths ---

  async def test_return_literal_value(self):
    """_eval_signal_value line 101: not callable, no args/kwargs -> return v."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn).then(Chain.return_, 99).run()), 99
        )

  async def test_return_callable_no_args(self):
    """_eval_signal_value line 98: callable, no args/kwargs -> v()."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn).then(Chain.return_, lambda: 77).run()), 77
        )

  async def test_return_callable_with_ellipsis(self):
    """_eval_signal_value line 84: args[0] is ... -> v()."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn).then(Chain.return_, lambda: 55, ...).run()), 55
        )

  async def test_return_callable_with_explicit_args(self):
    """_eval_signal_value line 91: args truthy, args[0] is not ...,
    kwargs is {} (not None, not EMPTY_DICT) -> v(*args, **kwargs)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def add(a, b):
          return a + b

        self.assertEqual(
          await await_(Chain(fn).then(Chain.return_, add, 10, 20).run()), 30
        )

  async def test_return_callable_with_args_and_kwargs(self):
    """_eval_signal_value line 91: args truthy + kwargs truthy -> v(*args, **kwargs)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def trigger(cv):
          Chain.return_(lambda a, key=None: a + key, 1, key=cv)

        self.assertEqual(
          await await_(Chain(fn, 42).then(trigger).run()), 43
        )

  async def test_return_callable_with_kwargs_only(self):
    """_eval_signal_value line 92-95: no args, kwargs truthy -> v(*args, **kwargs)."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def trigger(cv):
          Chain.return_(lambda **kw: kw['key'], key=cv)

        self.assertEqual(
          await await_(Chain(fn, 42).then(trigger).run()), 42
        )

  # --- break_ paths (same _eval_signal_value via handle_break_exc) ---

  async def test_break_literal_value(self):
    """_eval_signal_value line 101 via break_: non-callable literal."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(x):
          if x == 2:
            Chain.break_('stopped')
          return fn2(x)

        self.assertEqual(
          await await_(Chain(fn1, [1, 2, 3]).foreach(f).run()),
          'stopped'
        )

  async def test_break_callable_no_args(self):
    """_eval_signal_value line 98 via break_: callable, no args/kwargs -> v()."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(x):
          if x == 2:
            Chain.break_(lambda: 'break_result')
          return fn2(x)

        self.assertEqual(
          await await_(Chain(fn1, [1, 2, 3]).foreach(f).run()),
          'break_result'
        )

  async def test_break_callable_with_ellipsis(self):
    """_eval_signal_value line 84 via break_: args[0] is ... -> v()."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(x):
          if x == 2:
            Chain.break_(lambda: 'ellipsis_break', ...)
          return fn2(x)

        self.assertEqual(
          await await_(Chain(fn1, [1, 2, 3]).foreach(f).run()),
          'ellipsis_break'
        )

  async def test_break_callable_with_explicit_args(self):
    """_eval_signal_value line 91 via break_: args truthy, args[0] is not ..."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def add(a, b):
          return a + b

        def f(x):
          if x == 2:
            Chain.break_(add, 10, 20)
          return fn2(x)

        self.assertEqual(
          await await_(Chain(fn1, [1, 2, 3]).foreach(f).run()),
          30
        )

  async def test_break_callable_with_args_and_kwargs(self):
    """_eval_signal_value line 91 via break_: args + kwargs."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(x):
          if x == 2:
            Chain.break_(lambda a, key=None: a + key, 1, key=100)
          return fn2(x)

        self.assertEqual(
          await await_(Chain(fn1, [1, 2, 3]).foreach(f).run()),
          101
        )

  async def test_break_callable_with_kwargs_only(self):
    """_eval_signal_value line 92-95 via break_: kwargs only."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(x):
          if x == 2:
            Chain.break_(lambda **kw: kw['val'], val=77)
          return fn2(x)

        self.assertEqual(
          await await_(Chain(fn1, [1, 2, 3]).foreach(f).run()),
          77
        )

  async def test_break_no_value_returns_fallback(self):
    """handle_break_exc: value is Null -> returns fallback (partial list)."""
    for fn1, fn2 in product([empty, aempty], repeat=2):
      with self.subTest(fn1=fn1, fn2=fn2):
        def f(x):
          if x >= 3:
            Chain.break_()
          return fn2(x * 2)

        self.assertEqual(
          await await_(Chain(fn1, [1, 2, 3, 4]).foreach(f).run()),
          [2, 4]
        )

  async def test_return_no_value_returns_none(self):
    """handle_return_exc: value is Null -> returns None.

    Chain.return_() with no value raises _Return(Null, (), {}). Since
    value is Null, handle_return_exc returns None.
    """
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        def trigger(cv):
          Chain.return_()

        self.assertIsNone(
          await await_(Chain(fn, 10).then(trigger).run())
        )

  async def test_return_async_callable(self):
    """_eval_signal_value line 98 with async callable: v() returns a coroutine."""
    async def async_fn():
      return 88

    self.assertEqual(
      await Chain().then(Chain.return_, async_fn).run(), 88
    )

  async def test_return_falsy_literals(self):
    """_eval_signal_value line 101: falsy non-callable values returned as-is."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn).then(Chain.return_, 0).run()), 0
        )
        self.assertEqual(
          await await_(Chain(fn).then(Chain.return_, '').run()), ''
        )
        self.assertEqual(
          await await_(Chain(fn).then(Chain.return_, False).run()), False
        )


if __name__ == '__main__':
  import unittest
  unittest.main()
