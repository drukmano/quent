from unittest import IsolatedAsyncioTestCase
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
# ReturnSignalValueTests
# ---------------------------------------------------------------------------
class ReturnSignalValueTests(MyTestCase):

  async def test_return_no_args(self):
    """Chain(42).then(Chain.return_).run() — passes cv=42 to return_,
    _eval_signal_value gets (42, (), {}), 42 is not callable → returns 42."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 42).then(Chain.return_).run(), 42
        )

  async def test_return_literal_value(self):
    """Chain().then(Chain.return_, 99).run() — return_(99) raises _Return(99, (), {}),
    99 is not callable → returns 99 as-is."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, 99).run(), 99
        )

  async def test_return_callable_no_args(self):
    """Chain().then(Chain.return_, lambda: 77).run() — return_(lambda: 77) raises
    _Return(<lambda>, (), {}), lambda is callable → calls it → returns 77."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, lambda: 77).run(), 77
        )

  async def test_return_callable_with_ellipsis(self):
    """Chain().then(Chain.return_, lambda: 55, ...).run() — return_(lambda: 55, Ellipsis)
    raises _Return(<lambda>, (Ellipsis,), {}), args truthy, args[0] is ... → calls v() → 55."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, lambda: 55, ...).run(), 55
        )

  async def test_return_callable_with_explicit_args(self):
    """Chain().then(Chain.return_, fn, 10, 20).run() — return_(fn, 10, 20) raises
    _Return(fn, (10, 20), {}), args truthy, args[0] is not ... → calls fn(10, 20) → 30."""
    for fn, ctx in self.with_fn():
      with ctx:
        def add(a, b):
          return a + b
        await self.assertEqual(
          Chain(fn).then(Chain.return_, add, 10, 20).run(), 30
        )

  async def test_return_callable_with_kwargs_only(self):
    """Trigger return_ with kwargs but no positional args.
    Uses a wrapper that calls Chain.return_(fn, key=cv) directly."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda **kw: kw['key'], key=cv)

        await self.assertEqual(
          Chain(fn, 42).then(trigger).run(), 42
        )

  async def test_return_callable_with_args_and_kwargs(self):
    """Trigger return_ with both positional args and kwargs.
    Uses a wrapper that calls Chain.return_(fn, 1, key=cv) directly."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger(cv):
          Chain.return_(lambda a, key=None: a + key, 1, key=cv)

        await self.assertEqual(
          Chain(fn, 42).then(trigger).run(), 43
        )

  async def test_return_async_callable(self):
    """Chain().then(Chain.return_, async_fn).run() — async callable is called,
    returns a coroutine, which the chain awaits."""
    async def async_fn():
      return 88

    await self.assertEqual(
      Chain().then(Chain.return_, async_fn).run(), 88
    )

  async def test_return_nested_chain(self):
    """return_ exits the chain early: subsequent links are not executed.
    Chain(10).then(lambda v: Chain(v).then(lambda x: Chain.return_(x * 2)).then(lambda: 999).run()).run()
    The inner chain's return_ returns 20, skipping the .then(lambda: 999)."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 10).then(
            lambda v: Chain(v).then(lambda x: Chain.return_(x * 2)).then(lambda: 999).run()
          ).run(),
          20
        )

  async def test_return_falsy_literal(self):
    """return_ with falsy non-callable values returns them as-is."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn).then(Chain.return_, 0).run(), 0
        )
        await self.assertEqual(
          Chain(fn).then(Chain.return_, '').run(), ''
        )
        await self.assertEqual(
          Chain(fn).then(Chain.return_, False).run(), False
        )


# ---------------------------------------------------------------------------
# BreakSignalValueTests
# ---------------------------------------------------------------------------
class BreakSignalValueTests(MyTestCase):

  async def test_break_no_args(self):
    """Chain(42).while_true(lambda: Chain.break_(), ...).run() — break_() raises
    _Break(Null, (), {}), handle_break_exc sees _v is Null → returns nv=42."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 42).while_true(lambda: Chain.break_(), ...).run(), 42
        )

  async def test_break_literal_value(self):
    """Chain(1).while_true(lambda: Chain.break_(99), ...).run() — break_(99) raises
    _Break(99, (), {}), 99 is not callable → returns 99."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).while_true(lambda: Chain.break_(99), ...).run(), 99
        )

  async def test_break_callable_no_args(self):
    """Chain(1).while_true(lambda: Chain.break_(lambda: 77), ...).run() — break_(lambda: 77),
    lambda is callable → calls it → returns 77."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).while_true(lambda: Chain.break_(lambda: 77), ...).run(), 77
        )

  async def test_break_callable_with_ellipsis(self):
    """Chain(1).while_true(lambda: Chain.break_(lambda: 55, ...), ...).run() —
    args=(Ellipsis,), args[0] is ... → calls v() → 55."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).while_true(lambda: Chain.break_(lambda: 55, ...), ...).run(), 55
        )

  async def test_break_callable_with_explicit_args(self):
    """break_ with explicit positional args: break_(fn, 10, 20) → fn(10, 20)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def add(a, b):
          return a + b
        await self.assertEqual(
          Chain(fn, 1).while_true(lambda: Chain.break_(add, 10, 20), ...).run(), 30
        )

  async def test_break_callable_with_kwargs_only(self):
    """break_ with kwargs but no positional args.
    Uses a wrapper to call Chain.break_(fn, key=val) directly."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger():
          Chain.break_(lambda **kw: kw['key'], key=99)

        await self.assertEqual(
          Chain(fn, 1).while_true(trigger, ...).run(), 99
        )

  async def test_break_callable_with_args_and_kwargs(self):
    """break_ with both args and kwargs: break_(fn, 1, key=2) → fn(1, key=2)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def trigger():
          Chain.break_(lambda a, key=None: a + key, 10, key=20)

        await self.assertEqual(
          Chain(fn, 1).while_true(trigger, ...).run(), 30
        )

  async def test_break_async_callable(self):
    """break_ with an async callable: break_(aempty, 42) → awaits aempty(42) → 42."""
    await self.assertEqual(
      Chain(1).while_true(lambda: Chain.break_(aempty, 42), ...).run(), 42
    )

  async def test_break_falsy_literal(self):
    """break_(0) returns 0, not the loop's current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).while_true(lambda: Chain.break_(0), ...).run(), 0
        )

  async def test_break_none_literal(self):
    """break_(None) returns None — None is not Null and not callable."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(
          Chain(fn, 1).while_true(lambda: Chain.break_(None), ...).run()
        )

  async def test_break_in_foreach(self):
    """break_() inside foreach returns the partial list accumulated so far."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x >= 3:
            return Chain.break_()
          return x * 2

        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4, 5]).foreach(f).run(),
          [2, 4]
        )

  async def test_break_with_value_in_foreach(self):
    """break_('done') inside foreach replaces the partial list with 'done'."""
    for fn, ctx in self.with_fn():
      with ctx:
        def f(x):
          if x == 2:
            return Chain.break_('done')
          return x

        await self.assertEqual(
          Chain(fn, [1, 2, 3]).foreach(f).run(),
          'done'
        )


# ---------------------------------------------------------------------------
# InternalQuentExceptionReprTests
# ---------------------------------------------------------------------------
class InternalQuentExceptionReprTests(MyTestCase):

  async def test_return_repr(self):
    """Catching the _Return exception and verifying its repr is '<_Return>'."""
    try:
      Chain.return_(42, 1, 2, key=3)
    except Exception as exc:
      super(MyTestCase, self).assertEqual(repr(exc), '<_Return>')
    else:
      self.fail('Chain.return_ did not raise')

  async def test_break_repr(self):
    """Catching the _Break exception and verifying its repr is '<_Break>'."""
    try:
      Chain.break_(99)
    except Exception as exc:
      super(MyTestCase, self).assertEqual(repr(exc), '<_Break>')
    else:
      self.fail('Chain.break_ did not raise')

  async def test_internal_quent_exception_properties(self):
    """Verifying the exception's args tuple holds (__v, args, kwargs).
    The cdef attributes (_v, args_, kwargs_) are not accessible from Python,
    but Exception.args captures the __init__ parameters."""
    try:
      Chain.return_(42, 1, 2, key=3)
    except Exception as exc:
      # Exception.args is set to (__v, args_tuple, kwargs_dict) by Cython
      super(MyTestCase, self).assertEqual(len(exc.args), 3)
      super(MyTestCase, self).assertEqual(exc.args[0], 42)
      super(MyTestCase, self).assertEqual(exc.args[1], (1, 2))
      super(MyTestCase, self).assertEqual(exc.args[2], {'key': 3})
    else:
      self.fail('Chain.return_ did not raise')
