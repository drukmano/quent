from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_
from quent import Chain, Cascade, ChainAttr, CascadeAttr, QuentException, run
from quent.quent import PyNull


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


class NullSentinelTests(TestCase):

  def test_null_repr(self):
    result = repr(PyNull)
    self.assertEqual(result, '<Null>')

  def test_null_identity(self):
    self.assertIs(PyNull, PyNull)

  def test_null_is_not_none(self):
    self.assertIsNot(PyNull, None)

  def test_null_as_chain_value(self):
    # Chain(PyNull) behaves as a void chain (no root), so run() returns None.
    result = Chain(PyNull).run()
    self.assertIsNone(result)

  def test_chain_null_classmethod(self):
    self.assertIs(Chain.null(), PyNull)

  def test_chain_null_consistent(self):
    self.assertIs(Chain.null(), Chain.null())


class ChainConstructionTests(TestCase):

  def test_empty_chain_returns_none(self):
    result = Chain().run()
    self.assertIsNone(result)

  def test_chain_with_literal_root(self):
    result = Chain(42).run()
    self.assertEqual(result, 42)

  def test_chain_with_callable_root(self):
    result = Chain(lambda: 42).run()
    self.assertEqual(result, 42)

  def test_chain_with_callable_root_and_args(self):
    result = Chain(lambda a, b: a + b, 3, 7).run()
    self.assertEqual(result, 10)

  def test_chain_with_callable_root_and_kwargs(self):
    result = Chain(lambda a, b=5: a + b, 3, b=7).run()
    self.assertEqual(result, 10)

  def test_chain_with_kwargs_only_root(self):
    result = Chain(lambda *, x=1: x, x=5).run()
    self.assertEqual(result, 5)

  def test_chain_root_override_on_void_chain(self):
    result = Chain().run(42)
    self.assertEqual(result, 42)

  def test_chain_root_override_with_callable(self):
    result = Chain().run(lambda: 99)
    self.assertEqual(result, 99)

  def test_chain_root_override_with_args(self):
    result = Chain().run(lambda a, b: a * b, 3, 4)
    self.assertEqual(result, 12)

  def test_chain_double_root_raises(self):
    with self.assertRaises(QuentException):
      Chain(1).run(2)

  def test_chain_with_none_root(self):
    result = Chain(None).run()
    self.assertIsNone(result)

  def test_chain_with_false_root(self):
    result = Chain(False).run()
    self.assertIs(result, False)

  def test_chain_with_zero_root(self):
    result = Chain(0).run()
    self.assertEqual(result, 0)

  def test_chain_with_empty_string_root(self):
    result = Chain('').run()
    self.assertEqual(result, '')

  def test_chain_with_ellipsis_then(self):
    result = Chain(1).then(lambda: 99, ...).run()
    self.assertEqual(result, 99)

  def test_chain_bool_always_true(self):
    self.assertTrue(bool(Chain()))
    self.assertTrue(bool(Chain(0)))
    self.assertTrue(bool(Chain(None)))

  def test_cascade_bool_always_true(self):
    self.assertTrue(bool(Cascade()))

  def test_chain_returns_self_from_then(self):
    c = Chain(1)
    result = c.then(lambda v: v)
    self.assertIs(result, c)

  def test_chain_returns_self_from_do(self):
    c = Chain(1)
    result = c.do(lambda v: v)
    self.assertIs(result, c)

  def test_chain_returns_self_from_root(self):
    c = Chain(1)
    result = c.root(lambda v: v)
    self.assertIs(result, c)

  def test_chain_returns_self_from_root_do(self):
    c = Chain(1)
    result = c.root_do(lambda v: v)
    self.assertIs(result, c)


class CascadeConstructionTests(TestCase):

  def test_cascade_empty_returns_none(self):
    result = Cascade().run()
    self.assertIsNone(result)

  def test_cascade_always_returns_root(self):
    result = Cascade(10).then(lambda v: v * 2).then(lambda v: v + 5).run()
    self.assertEqual(result, 10)

  def test_cascade_with_callable_root(self):
    result = Cascade(lambda: 42).then(lambda v: v * 2).run()
    self.assertEqual(result, 42)

  def test_cascade_root_override(self):
    result = Cascade().then(lambda v: v * 2).run(10)
    self.assertEqual(result, 10)

  def test_cascade_double_root_raises(self):
    with self.assertRaises(QuentException):
      Cascade(1).run(2)

  def test_cascade_none_root(self):
    result = Cascade(None).then(lambda v: v).run()
    self.assertIsNone(result)


class _Obj:
  """Helper class with properties and methods for attribute tests."""
  def __init__(self, val=10):
    self.val = val
    self.prop = val

  @property
  def value(self):
    return self.val

  def method(self):
    return self.val * 2

  def add(self, n, *, extra=0):
    return self.val + n + extra


class EvalCodeEdgeCaseTests(TestCase):

  def test_noncallable_in_do_raises(self):
    with self.assertRaises(QuentException):
      Chain(1).do(True)

  def test_noncallable_in_then_allowed(self):
    result = Chain(1).then(True).run()
    self.assertIs(result, True)

  def test_callable_with_ellipsis_calls_without_args(self):
    result = Chain(1).then(lambda: 'no_arg', ...).run()
    self.assertEqual(result, 'no_arg')

  def test_fattr_with_no_args(self):
    obj = _Obj(7)
    result = Chain(obj).attr_fn('method').run()
    self.assertEqual(result, 14)

  def test_kwargs_only_in_link(self):
    # When only kwargs are supplied (no positional args), the link uses
    # EVAL_CALL_WITH_EXPLICIT_ARGS which calls fn(*args, **kwargs) without
    # injecting the current value.  So we use a lambda that accepts only kwargs.
    result = Chain(1).then(lambda *, extra=0: extra, extra=10).run()
    self.assertEqual(result, 10)

  def test_attr_get_attribute_eval_code(self):
    obj = _Obj(42)
    result = Chain(obj).attr('prop').run()
    self.assertEqual(result, 42)

  def test_attr_fn_with_args_and_kwargs(self):
    obj = _Obj(10)
    result = Chain(obj).attr_fn('add', 5, extra=3).run()
    self.assertEqual(result, 18)


class ValuePassingTests(MyTestCase):

  async def test_then_passes_result(self):
    for fn, ctx in self.with_fn():
      with ctx:
        # fn is used as the root callable so that the chain runs in
        # sync mode (empty) or async mode (aempty).
        result = Chain(fn, 1).then(lambda v: v + 1).then(lambda v: v * 3).run()
        await self.assertEqual(result, 6)

  async def test_do_discards_result(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, 5).do(lambda v: 999).run()
        await self.assertEqual(result, 5)

  async def test_root_passes_root_value(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, 5).then(lambda v: v * 2).root(lambda v: v + 100).run()
        await self.assertEqual(result, 105)

  async def test_root_do_passes_root_discards_result(self):
    for fn, ctx in self.with_fn():
      with ctx:
        captured = {}
        def capture_root(v):
          captured['root'] = v
          return 999  # should be discarded

        result = Chain(fn, 5).then(lambda v: v * 2).root_do(capture_root).run()
        # root_do receives root value (5) and discards its return (999);
        # current value remains v*2 = 10.
        await self.assertEqual(result, 10)
        await self.assertEqual(captured['root'], 5)

  async def test_value_none_propagates(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, None).then(lambda v: v).run()
        await self.assertIsNone(result)

  async def test_value_false_propagates(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, False).then(lambda v: v).run()
        await self.assertFalse(result)

  async def test_chain_with_class_root(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, dict(a=1)).run()
        await self.assertEqual(result, {'a': 1})
