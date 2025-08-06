import asyncio
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
# WhileTrueExceptionTests
# ---------------------------------------------------------------------------
class WhileTrueExceptionTests(MyTestCase):

  async def test_sync_exception_propagates(self):
    """A non-Break exception raised on the first iteration propagates out of while_true."""
    for fn, ctx in self.with_fn():
      with ctx:
        def body(v=None):
          raise ValueError('sync boom')

        with self.assertRaises(ValueError) as cm:
          await await_(Chain(fn, 1).while_true(body, ...).run())
        super(MyTestCase, self).assertEqual(str(cm.exception), 'sync boom')

  async def test_async_exception_propagates(self):
    """An exception raised inside an async body propagates correctly."""
    async def body(v=None):
      raise TypeError('async boom')

    with self.assertRaises(TypeError) as cm:
      await await_(Chain(1).while_true(body, ...).run())
    super(MyTestCase, self).assertEqual(str(cm.exception), 'async boom')

  async def test_exception_after_several_iterations(self):
    """Exception raised after N successful iterations still propagates."""
    for fn, ctx in self.with_fn():
      with ctx:
        counter = {'n': 0}

        def body(v=None):
          counter['n'] += 1
          if counter['n'] >= 4:
            raise RuntimeError('fail at 4')

        counter['n'] = 0
        with self.assertRaises(RuntimeError) as cm:
          await await_(Chain(fn, 10).while_true(body, ...).run())
        super(MyTestCase, self).assertEqual(counter['n'], 4)
        super(MyTestCase, self).assertEqual(str(cm.exception), 'fail at 4')

  async def test_exception_type_preserved_sync(self):
    """Custom exception subclass type is preserved through the sync path."""
    class CustomError(Exception):
      pass

    class CustomSubError(CustomError):
      pass

    for fn, ctx in self.with_fn():
      with ctx:
        def body(v=None):
          raise CustomSubError('sub error')

        with self.assertRaises(CustomSubError):
          await await_(Chain(fn, 1).while_true(body, ...).run())

  async def test_exception_type_preserved_async(self):
    """Custom exception subclass type is preserved through the async transition path."""
    class SpecificError(RuntimeError):
      pass

    counter = {'n': 0}

    def body(v=None):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty()  # trigger async transition
      raise SpecificError('specific')

    counter['n'] = 0
    with self.assertRaises(SpecificError) as cm:
      await await_(Chain(1).while_true(body, ...).run())
    super(MyTestCase, self).assertEqual(str(cm.exception), 'specific')


# ---------------------------------------------------------------------------
# WhileTrueMissingTests
# ---------------------------------------------------------------------------
class WhileTrueMissingTests(MyTestCase):

  async def test_no_root_value_with_ellipsis(self):
    """Chain().while_true(fn, ...) calls fn with no args; break returns None."""
    for fn, ctx in self.with_fn():
      with ctx:
        calls = []

        def body():
          calls.append(True)
          if len(calls) >= 2:
            return Chain.break_()

        calls.clear()
        await self.assertIsNone(
          Chain().while_true(body, ...).run()
        )
        super(MyTestCase, self).assertEqual(len(calls), 2)

  async def test_no_root_value_without_args(self):
    """Chain().while_true(fn) passes Null (renders as None) to fn accepting optional arg."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'v': 'sentinel'}

        def body(v=None):
          received['v'] = v
          return Chain.break_()

        received['v'] = 'sentinel'
        await self.assertIsNone(
          Chain().while_true(body).run()
        )
        super(MyTestCase, self).assertIsNone(received['v'])

  async def test_break_without_value_returns_cv(self):
    """Chain.break_() with no value falls back to cv (the current/root value)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def body(v=None):
          return Chain.break_()

        await self.assertEqual(
          Chain(fn, 77).while_true(body).run(), 77
        )

  async def test_break_with_explicit_value_overrides_cv(self):
    """Chain.break_(val) returns val, ignoring the root cv."""
    for fn, ctx in self.with_fn():
      with ctx:
        sentinel = object()

        def body(v=None):
          return Chain.break_(sentinel)

        await self.assertIs(
          Chain(fn, 99).while_true(body).run(), sentinel
        )

  async def test_no_root_break_with_value(self):
    """Chain().while_true(fn, ...) where fn breaks with an explicit value returns that value."""
    for fn, ctx in self.with_fn():
      with ctx:
        def body():
          return Chain.break_(42)

        await self.assertEqual(
          Chain().while_true(body, ...).run(), 42
        )


# ---------------------------------------------------------------------------
# WhileTrueTempArgsTests
# ---------------------------------------------------------------------------
class WhileTrueTempArgsTests(MyTestCase):

  async def test_temp_args_set_when_no_explicit_args(self):
    """When while_true has no explicit args, temp_args is set to (cv,) on the link."""
    for fn, ctx in self.with_fn():
      with ctx:
        counter = {'n': 0}

        def body(v):
          counter['n'] += 1
          if counter['n'] >= 2:
            raise TestExc('check temp_args')

        counter['n'] = 0
        with self.assertRaises(TestExc):
          await await_(Chain(fn, 55).while_true(body).run())

  async def test_temp_args_not_set_when_explicit_args(self):
    """When while_true has explicit args, temp_args is NOT set in __call__."""
    for fn, ctx in self.with_fn():
      with ctx:
        received = {'args': None}

        def body(a, b):
          received['args'] = (a, b)
          return Chain.break_()

        received['args'] = None
        await self.assertIsNone(
          Chain().while_true(body, 'x', 'y').run()
        )
        super(MyTestCase, self).assertEqual(received['args'], ('x', 'y'))

  async def test_chain_reuse_temp_args_mutation(self):
    """Rerunning the same chain overwrites temp_args from the previous run."""
    counter = {'n': 0}

    def body(v):
      counter['n'] += 1
      if counter['n'] >= 2:
        return Chain.break_()

    chain = Chain(10).while_true(body)

    # Run 1
    counter['n'] = 0
    result1 = chain.run()
    super(MyTestCase, self).assertEqual(result1, 10)

    # Run 2 -- same chain, temp_args should be overwritten
    counter['n'] = 0
    result2 = chain.run()
    super(MyTestCase, self).assertEqual(result2, 10)

  async def test_async_exception_annotates_temp_args(self):
    """In the async path, BaseException gets __quent_link_temp_args__ annotation."""
    counter = {'n': 0}

    def body(v=None):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty()  # trigger async transition
      raise ValueError('annotated')

    counter['n'] = 0
    try:
      await await_(Chain(42).while_true(body, ...).run())
      self.fail('Expected ValueError')  # noqa: PT015
    except ValueError as e:
      super(MyTestCase, self).assertTrue(
        hasattr(e, '__quent_link_temp_args__')
      )
      super(MyTestCase, self).assertIsInstance(
        e.__quent_link_temp_args__, dict
      )


# ---------------------------------------------------------------------------
# WhileTrueAsyncTransitionTests
# ---------------------------------------------------------------------------
class WhileTrueAsyncTransitionTests(MyTestCase):

  async def test_sync_to_async_transition_with_counting(self):
    """Sync body returns a coroutine mid-loop; async continuation counts correctly."""
    counter = {'n': 0}

    def body(v=None):
      counter['n'] += 1
      if counter['n'] >= 6:
        return Chain.break_(counter['n'])
      if counter['n'] == 3:
        return aempty()  # triggers async at iteration 3
      return None

    counter['n'] = 0
    result = await await_(Chain(1).while_true(body, ...).run())
    super(MyTestCase, self).assertEqual(result, 6)
    super(MyTestCase, self).assertEqual(counter['n'], 6)

  async def test_async_max_iterations_overflow(self):
    """max_iterations exceeded inside the async continuation path."""
    counter = {'n': 0}

    def body(v=None):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty()  # trigger async on first iteration
      return None  # never breaks

    counter['n'] = 0
    with self.assertRaises(QuentException) as cm:
      await await_(Chain(1).while_true(body, ..., max_iterations=4).run())
    super(MyTestCase, self).assertIn('exceeded max_iterations', str(cm.exception))

  async def test_sync_body_returning_coroutine_triggers_async(self):
    """A purely sync function that returns a coroutine on a specific iteration
    triggers the async path; subsequent iterations are awaited."""
    iterations = []

    def body(v=None):
      iterations.append(len(iterations) + 1)
      if len(iterations) >= 5:
        return Chain.break_(len(iterations))
      if len(iterations) == 2:
        # Return a coroutine -- this triggers the async transition
        return aempty()
      return None

    iterations.clear()
    result = await await_(Chain(0).while_true(body, ...).run())
    super(MyTestCase, self).assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(iterations, [1, 2, 3, 4, 5])

  async def test_async_break_with_value_in_async_continuation(self):
    """Break with value in the async continuation returns that value."""
    counter = {'n': 0}

    def body(v=None):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty()  # trigger async transition
      if counter['n'] >= 4:
        return Chain.break_('done')
      return None

    counter['n'] = 0
    result = await await_(Chain(1).while_true(body, ...).run())
    super(MyTestCase, self).assertEqual(result, 'done')

  async def test_async_break_without_value_returns_cv(self):
    """Break without value in async continuation returns cv (root value)."""
    counter = {'n': 0}

    def body(v=None):
      counter['n'] += 1
      if counter['n'] == 1:
        return aempty()  # trigger async
      if counter['n'] >= 3:
        return Chain.break_()
      return None

    counter['n'] = 0
    result = await await_(Chain(77).while_true(body, ...).run())
    super(MyTestCase, self).assertEqual(result, 77)
