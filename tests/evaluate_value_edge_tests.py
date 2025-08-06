import functools
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_
from quent import Chain, Cascade, QuentException, run


class _Obj:
  def __init__(self, val=10):
    self.val = val

  def method(self):
    return self.val * 2

  def add(self, n, *, extra=0):
    return self.val + n + extra


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
# EvaluateValueDispatchTests
# ---------------------------------------------------------------------------
class EvaluateValueDispatchTests(MyTestCase):

  async def test_call_with_current_value_callable(self):
    """EVAL_CALL_WITH_CURRENT_VALUE fast path: callable receives the current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 10).then(lambda v: v * 2).run(), 20
        )

  async def test_call_with_current_value_null_cv(self):
    """EVAL_CALL_WITH_CURRENT_VALUE with Null cv: void chain calls callable with no args."""
    # Sync: void chain's first link receives Null, so link.v() is called (no args).
    super(MyTestCase, self).assertEqual(
      Chain().then(lambda: 42).run(), 42
    )
    # Async: same path but through the async-capable void chain.
    await self.assertEqual(
      Chain().then(aempty).then(lambda v: 42).run(), 42
    )

  async def test_call_with_explicit_args(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS: explicit positional args replace cv."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda a, b: a + b, 10, 20).run(), 30
        )

  async def test_call_with_explicit_kwargs(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS: both positional and keyword args forwarded."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda a, b=0: a + b, 5, b=3).run(), 8
        )

  async def test_call_without_args_ellipsis(self):
    """EVAL_CALL_WITHOUT_ARGS via ellipsis: callable invoked with no arguments."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 99).then(lambda: 7, ...).run(), 7
        )

  async def test_return_as_is_literal(self):
    """EVAL_RETURN_AS_IS: non-callable integer literal returned directly."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(42).run(), 42
        )

  async def test_return_as_is_string(self):
    """EVAL_RETURN_AS_IS: non-callable string literal returned directly."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then("hello").run(), "hello"
        )

  async def test_get_attribute(self):
    """EVAL_GET_ATTRIBUTE via attr(): attribute value retrieved from current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        obj = _Obj(10)
        await self.assertEqual(
          Chain(fn, obj).attr('val').run(), 10
        )

  async def test_attr_fn_call_without_args(self):
    """EVAL_CALL_WITHOUT_ARGS via attr_fn(): method called with no extra args."""
    for fn, ctx in self.with_fn():
      with ctx:
        obj = _Obj(10)
        await self.assertEqual(
          Chain(fn, obj).attr_fn('method').run(), 20
        )

  async def test_attr_fn_call_with_explicit_args(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS via attr_fn(): method called with args and kwargs."""
    for fn, ctx in self.with_fn():
      with ctx:
        obj = _Obj(10)
        await self.assertEqual(
          Chain(fn, obj).attr_fn('add', 5, extra=3).run(), 18
        )

  async def test_attr_get_then_return(self):
    """EVAL_GET_ATTRIBUTE path: attr() retrieves the raw attribute, not calling it."""
    for fn, ctx in self.with_fn():
      with ctx:
        obj = _Obj(42)
        await self.assertEqual(
          Chain(fn, obj).attr('val').run(), 42
        )

  async def test_kwargs_only_explicit_args(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS: kwargs-only callable with keyword arguments."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda *, x=0: x, x=10).run(), 10
        )


# ---------------------------------------------------------------------------
# NestedChainEvalTests
# ---------------------------------------------------------------------------
class NestedChainEvalTests(MyTestCase):

  async def test_nested_chain_call_with_current_value(self):
    """is_chain + EVAL_CALL_WITH_CURRENT_VALUE: inner chain receives outer cv as root."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(Chain().then(lambda v: v * 3)).run(), 15
        )

  async def test_nested_chain_call_without_args(self):
    """is_chain + EVAL_CALL_WITHOUT_ARGS via ellipsis: inner chain runs with Null root."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(Chain(99), ...).run(), 99
        )

  async def test_nested_chain_call_with_explicit_args(self):
    """is_chain + EVAL_CALL_WITH_EXPLICIT_ARGS: first explicit arg becomes inner root."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(Chain().then(lambda v: v), 42).run(), 42
        )

  async def test_triple_nested_chain(self):
    """Three levels of nested chains propagating the current value through each."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 2).then(
            Chain().then(
              Chain().then(lambda v: v * 10)
            )
          ).run(),
          20
        )

  async def test_cascade_as_inner_chain(self):
    """Cascade as inner chain returns root value, discarding link results."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(Cascade().then(lambda v: v * 100)).run(), 5
        )

  async def test_nested_chain_with_explicit_args_multiple(self):
    """is_chain + EVAL_CALL_WITH_EXPLICIT_ARGS: first arg is root, rest forwarded."""
    for fn, ctx in self.with_fn():
      with ctx:
        # First explicit arg (10) becomes the inner chain's root value.
        # The inner chain's .then(lambda v: v) receives 10 and returns 10.
        await self.assertEqual(
          Chain(fn, 1).then(Chain().then(lambda v: v), 10).run(), 10
        )


# ---------------------------------------------------------------------------
# SpecialValueTests
# ---------------------------------------------------------------------------
class SpecialValueTests(MyTestCase):

  async def test_builtin_len(self):
    """Built-in len() used as a callable link receiving the current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, [1, 2, 3]).then(len).run(), 3
        )

  async def test_builtin_str(self):
    """Built-in str() used as a callable link converting int to string."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 42).then(str).run(), "42"
        )

  async def test_functools_partial(self):
    """functools.partial used as a callable link, receiving the current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 10).then(functools.partial(lambda a, b: a + b, b=5)).run(), 15
        )

  async def test_class_type_as_callable(self):
    """Class type (int) used as a callable link, acting as constructor."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, "42").then(int).run(), 42
        )

  async def test_falsy_literals_as_values(self):
    """Falsy non-callable literals returned as-is via EVAL_RETURN_AS_IS."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(0).run(), 0
        )
        await self.assertEqual(
          Chain(fn, 1).then('').run(), ''
        )
        await self.assertIs(
          Chain(fn, 1).then(False).run(), False
        )
        await self.assertIsNone(
          Chain(fn, 1).then(None).run()
        )

  async def test_generator_function_as_link(self):
    """range (a callable) used as a link, receives cv and returns a range object."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 3).then(range).run(), range(3)
        )
