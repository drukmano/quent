from unittest import IsolatedAsyncioTestCase
from tests.utils import TestExc, throw_if, empty, aempty, await_
from quent import Chain, Cascade, ChainAttr, CascadeAttr, QuentException, run


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


class DirectIfTests(MyTestCase):

  async def test_if_truthy_executes_branch(self):
    await self.assertEqual(Chain(1).if_(lambda v: v * 10).run(), 10)

  async def test_if_falsy_returns_original(self):
    await self.assertEqual(Chain(0).if_(lambda v: v * 10).run(), 0)

  async def test_if_none_is_falsy(self):
    await self.assertIsNone(Chain(None).if_(lambda v: 'yes').run())

  async def test_if_empty_string_is_falsy(self):
    await self.assertEqual(Chain('').if_(lambda v: 'yes').run(), '')

  async def test_if_with_literal_on_true(self):
    await self.assertEqual(Chain(True).if_(42).run(), 42)

  async def test_if_with_callable_on_true_and_ellipsis(self):
    await self.assertEqual(Chain(True).if_(lambda: 42, ...).run(), 42)

  async def test_if_with_sync_async_via_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        # truthy path
        await self.assertEqual(Chain(fn, True).if_(lambda v: 'yes').run(), 'yes')
        # falsy path
        await self.assertEqual(Chain(fn, False).if_(lambda v: 'yes').run(), False)
        # truthy with literal
        await self.assertEqual(Chain(fn, 1).if_(99).run(), 99)
        # falsy returns original
        await self.assertEqual(Chain(fn, 0).if_(99).run(), 0)


class IfElseTests(MyTestCase):

  async def test_if_else_truthy_path(self):
    await self.assertEqual(
      Chain(True).if_(lambda v: 'yes').else_(lambda v: 'no').run(), 'yes'
    )

  async def test_if_else_falsy_path(self):
    await self.assertEqual(
      Chain(False).if_(lambda v: 'yes').else_(lambda v: 'no').run(), 'no'
    )

  async def test_if_else_with_literals(self):
    await self.assertEqual(Chain(True).if_(1).else_(0).run(), 1)
    await self.assertEqual(Chain(False).if_(1).else_(0).run(), 0)

  async def test_if_else_async_on_true_value(self):
    # When if_ branch returns a coroutine (via aempty), value is properly awaited
    await self.assertEqual(
      Chain(True).if_(lambda v: aempty('async_yes')).else_(lambda v: 'no').run(),
      'async_yes'
    )
    # falsy path with async else
    await self.assertEqual(
      Chain(False).if_(lambda v: 'yes').else_(lambda v: aempty('async_no')).run(),
      'async_no'
    )

  async def test_if_else_async_condition(self):
    # Use condition() with async fn, verifying _await_cond_result path
    await self.assertEqual(
      Chain(4).condition(lambda v: aempty(v % 2 == 0)).if_(lambda v: 'even').else_(lambda v: 'odd').run(),
      'even'
    )
    await self.assertEqual(
      Chain(3).condition(lambda v: aempty(v % 2 == 0)).if_(lambda v: 'even').else_(lambda v: 'odd').run(),
      'odd'
    )

  async def test_else_without_if_raises(self):
    with self.assertRaises(QuentException):
      Chain(1).else_(2)


class IfNotTests(MyTestCase):

  async def test_if_not_falsy_executes(self):
    await self.assertEqual(Chain(False).if_not(1).run(), 1)

  async def test_if_not_truthy_skips(self):
    await self.assertEqual(Chain(True).if_not(1).run(), True)

  async def test_if_not_with_else(self):
    await self.assertEqual(Chain(True).if_not(1).else_(0).run(), 0)
    # Also verify when if_not fires and else is skipped
    await self.assertEqual(Chain(False).if_not(1).else_(0).run(), 1)

  async def test_if_not_none_executes(self):
    await self.assertEqual(Chain(None).if_not(lambda v: 'was_none').run(), 'was_none')


class ConditionMethodTests(MyTestCase):

  async def test_condition_custom_fn_truthy(self):
    await self.assertEqual(
      Chain(4).condition(lambda v: v % 2 == 0).if_(1).run(), 1
    )

  async def test_condition_custom_fn_falsy(self):
    await self.assertEqual(
      Chain(3).condition(lambda v: v % 2 == 0).if_(1).run(), 3
    )

  async def test_condition_custom_fn_with_else(self):
    await self.assertEqual(
      Chain(3).condition(lambda v: v % 2 == 0).if_(1).else_(2).run(), 2
    )

  async def test_condition_async_fn(self):
    # Tests the _await_cond_result path
    await self.assertEqual(
      Chain(4).condition(lambda v: aempty(v % 2 == 0)).if_(1).run(), 1
    )
    # Async condition that evaluates to False
    await self.assertEqual(
      Chain(3).condition(lambda v: aempty(v % 2 == 0)).if_(1).run(), 3
    )

  async def test_condition_without_if_finalizes_as_link(self):
    # condition without if_ is finalized as a regular link; returns the condition result
    await self.assertTrue(Chain(4).condition(lambda v: v % 2 == 0).run())
    await self.assertFalse(Chain(3).condition(lambda v: v % 2 == 0).run())

  async def test_condition_chained_consecutively(self):
    # Multiple condition/if sequences in one chain
    # First condition: 4 % 2 == 0 → True → returns 'even' (len 4)
    # Second condition: len('even') > 3 → True → returns 'long'
    await self.assertEqual(
      Chain(4)
      .condition(lambda v: v % 2 == 0).if_(lambda v: 'even').else_(lambda v: 'odd')
      .condition(lambda v: len(v) > 3).if_(lambda v: 'long').else_(lambda v: 'short')
      .run(),
      'long'
    )
    # First: 3 % 2 == 0 → False → returns 'odd' (len 3)
    # Second: len('odd') > 3 → False → returns 'short'
    await self.assertEqual(
      Chain(3)
      .condition(lambda v: v % 2 == 0).if_(lambda v: 'even').else_(lambda v: 'odd')
      .condition(lambda v: len(v) > 3).if_(lambda v: 'long').else_(lambda v: 'short')
      .run(),
      'short'
    )
    # First: 4 % 2 == 0 → True → returns 'even' (len 4)
    # Second: len('even') >= 4 → True → returns 'long'
    await self.assertEqual(
      Chain(4)
      .condition(lambda v: v % 2 == 0).if_(lambda v: 'even').else_(lambda v: 'odd')
      .condition(lambda v: len(v) >= 4).if_(lambda v: 'long').else_(lambda v: 'short')
      .run(),
      'long'
    )


class ComparisonOperatorTests(MyTestCase):

  async def test_eq_true(self):
    await self.assertTrue(Chain(5).eq(5).run())

  async def test_eq_false(self):
    await self.assertFalse(Chain(5).eq(6).run())

  async def test_neq_true(self):
    await self.assertTrue(Chain(5).neq(6).run())

  async def test_neq_false(self):
    await self.assertFalse(Chain(5).neq(5).run())

  async def test_is_same_object(self):
    obj = object()
    await self.assertTrue(Chain(obj).is_(obj).run())

  async def test_is_different_object(self):
    obj1 = object()
    obj2 = object()
    await self.assertFalse(Chain(obj1).is_(obj2).run())

  async def test_is_not_different(self):
    obj1 = object()
    obj2 = object()
    await self.assertTrue(Chain(obj1).is_not(obj2).run())

  async def test_is_not_same(self):
    obj = object()
    await self.assertFalse(Chain(obj).is_not(obj).run())

  async def test_in_present(self):
    await self.assertTrue(Chain(3).in_([1, 2, 3]).run())

  async def test_in_absent(self):
    await self.assertFalse(Chain(4).in_([1, 2, 3]).run())

  async def test_not_in_absent(self):
    await self.assertTrue(Chain(4).not_in([1, 2, 3]).run())

  async def test_not_in_present(self):
    await self.assertFalse(Chain(3).not_in([1, 2, 3]).run())

  async def test_eq_with_if_branch(self):
    await self.assertEqual(
      Chain(5).eq(5).if_(lambda v: 'equal').run(), 'equal'
    )

  async def test_eq_with_if_else_branch(self):
    # eq true path
    await self.assertEqual(
      Chain(5).eq(5).if_(lambda v: 'equal').else_(lambda v: 'not_equal').run(),
      'equal'
    )
    # eq false path with else
    await self.assertEqual(
      Chain(5).eq(6).if_(lambda v: 'equal').else_(lambda v: 'not_equal').run(),
      'not_equal'
    )

  async def test_comparisons_with_async_root(self):
    for fn, ctx in self.with_fn():
      with ctx:
        # eq
        await self.assertTrue(Chain(fn, 5).eq(5).run())
        await self.assertFalse(Chain(fn, 5).eq(6).run())

        # neq
        await self.assertTrue(Chain(fn, 5).neq(6).run())
        await self.assertFalse(Chain(fn, 5).neq(5).run())

        # is_ / is_not
        obj = object()
        await self.assertTrue(Chain(fn, obj).is_(obj).run())
        await self.assertFalse(Chain(fn, object()).is_(obj).run())
        await self.assertTrue(Chain(fn, object()).is_not(obj).run())
        await self.assertFalse(Chain(fn, obj).is_not(obj).run())

        # in_ / not_in
        await self.assertTrue(Chain(fn, 3).in_([1, 2, 3]).run())
        await self.assertFalse(Chain(fn, 4).in_([1, 2, 3]).run())
        await self.assertTrue(Chain(fn, 4).not_in([1, 2, 3]).run())
        await self.assertFalse(Chain(fn, 3).not_in([1, 2, 3]).run())

        # eq with if_ branch
        await self.assertEqual(
          Chain(fn, 5).eq(5).if_(lambda v: 'equal').run(), 'equal'
        )
        await self.assertEqual(
          Chain(fn, 5).eq(6).if_(lambda v: 'equal').else_(lambda v: 'not_equal').run(),
          'not_equal'
        )


class NotTests(MyTestCase):

  async def test_not_truthy(self):
    await self.assertFalse(Chain(1).not_().run())

  async def test_not_falsy(self):
    await self.assertTrue(Chain(0).not_().run())

  async def test_not_none(self):
    await self.assertTrue(Chain(None).not_().run())

  async def test_not_with_if(self):
    # not_(0) -> True (condition truthy), so if_ fires with original value 0
    await self.assertEqual(
      Chain(0).not_().if_(lambda v: 'negated_truthy').run(), 'negated_truthy'
    )
    # not_(1) -> False (condition falsy), so if_ doesn't fire, returns original value 1
    await self.assertEqual(
      Chain(1).not_().if_(lambda v: 'negated_truthy').run(), 1
    )

  async def test_not_async_root(self):
    await self.assertTrue(Chain(aempty, 0).not_().run())
    await self.assertFalse(Chain(aempty, 1).not_().run())


class IsInstanceTests(MyTestCase):

  async def test_isinstance_single_type_match(self):
    await self.assertTrue(Chain('hello').isinstance_(str).run())

  async def test_isinstance_single_type_no_match(self):
    await self.assertFalse(Chain(123).isinstance_(str).run())

  async def test_isinstance_multiple_types_match(self):
    await self.assertTrue(Chain(123).isinstance_(int, str).run())

  async def test_isinstance_multiple_types_no_match(self):
    await self.assertFalse(Chain([]).isinstance_(int, str).run())

  async def test_isinstance_with_if_branch(self):
    await self.assertEqual(
      Chain('hello').isinstance_(str).if_(lambda v: v.upper()).run(), 'HELLO'
    )
    # isinstance_ fails, if_ not taken, returns original value
    await self.assertEqual(
      Chain(123).isinstance_(str).if_(lambda v: str(v)).run(), 123
    )

  async def test_isinstance_with_if_else(self):
    await self.assertEqual(
      Chain('hello').isinstance_(str).if_(lambda v: 'str').else_(lambda v: 'not_str').run(),
      'str'
    )
    await self.assertEqual(
      Chain(123).isinstance_(str).if_(lambda v: 'str').else_(lambda v: 'not_str').run(),
      'not_str'
    )

  async def test_isinstance_subclass(self):
    # bool subclasses int
    await self.assertTrue(Chain(True).isinstance_(int).run())
    await self.assertTrue(Chain(False).isinstance_(int).run())


class IfRaiseElseRaiseTests(MyTestCase):

  async def test_if_raise_truthy_raises(self):
    with self.assertRaises(ValueError):
      await await_(Chain(True).if_raise(ValueError('x')).run())

  async def test_if_raise_falsy_passes(self):
    await self.assertEqual(Chain(False).if_raise(ValueError('x')).run(), False)

  async def test_else_raise_falsy_raises(self):
    with self.assertRaises(RuntimeError):
      await await_(Chain(False).if_(1).else_raise(RuntimeError('x')).run())

  async def test_else_raise_truthy_passes(self):
    await self.assertEqual(
      Chain(True).if_(1).else_raise(RuntimeError('x')).run(), 1
    )

  async def test_if_not_raise_falsy_raises(self):
    with self.assertRaises(ValueError):
      await await_(Chain(False).if_not_raise(ValueError('x')).run())

  async def test_if_not_raise_truthy_passes(self):
    await self.assertEqual(Chain(True).if_not_raise(ValueError('x')).run(), True)

  async def test_if_raise_with_async_root(self):
    for fn, ctx in self.with_fn():
      with ctx:
        # if_raise truthy raises
        with self.assertRaises(ValueError):
          await await_(Chain(fn, True).if_raise(ValueError('x')).run())
        # if_raise falsy passes
        await self.assertEqual(Chain(fn, False).if_raise(ValueError('x')).run(), False)

        # else_raise falsy raises
        with self.assertRaises(RuntimeError):
          await await_(Chain(fn, False).if_(1).else_raise(RuntimeError('x')).run())
        # else_raise truthy passes
        await self.assertEqual(
          Chain(fn, True).if_(1).else_raise(RuntimeError('x')).run(), 1
        )

        # if_not_raise falsy raises
        with self.assertRaises(ValueError):
          await await_(Chain(fn, False).if_not_raise(ValueError('x')).run())
        # if_not_raise truthy passes
        await self.assertEqual(
          Chain(fn, True).if_not_raise(ValueError('x')).run(), True
        )


class OrTests(MyTestCase):

  async def test_or_falsy_returns_fallback(self):
    await self.assertEqual(Chain(None).or_(42).run(), 42)

  async def test_or_zero_returns_fallback(self):
    await self.assertEqual(Chain(0).or_(42).run(), 42)

  async def test_or_empty_string_returns_fallback(self):
    await self.assertEqual(Chain('').or_('default').run(), 'default')

  async def test_or_truthy_returns_value(self):
    await self.assertEqual(Chain(10).or_(42).run(), 10)

  async def test_or_async_root(self):
    await self.assertEqual(Chain(aempty, None).or_(42).run(), 42)
    await self.assertEqual(Chain(aempty, 10).or_(42).run(), 10)

  async def test_or_chained(self):
    await self.assertEqual(Chain(None).or_(0).or_(42).run(), 42)
    # First or_ returns 0 (falsy), second or_ returns 42
    await self.assertEqual(Chain(0).or_(0).or_(42).run(), 42)
    # First or_ returns 10 (truthy), second or_ passes through
    await self.assertEqual(Chain(None).or_(10).or_(42).run(), 10)
