from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run


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
