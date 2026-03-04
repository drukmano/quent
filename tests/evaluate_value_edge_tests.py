import functools
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_
from quent import Chain


# ---------------------------------------------------------------------------
# EvaluateValueDispatchTests
# ---------------------------------------------------------------------------
class EvaluateValueDispatchTests(IsolatedAsyncioTestCase):

  async def test_call_with_current_value_callable(self):
    """EVAL_CALL_WITH_CURRENT_VALUE fast path: callable receives the current value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 10).then(lambda v: v * 2).run()), 20
        )

  async def test_call_with_current_value_null_cv(self):
    """EVAL_CALL_WITH_CURRENT_VALUE with Null cv: void chain calls callable with no args."""
    # Sync: void chain's first link receives Null, so link.v() is called (no args).
    self.assertEqual(
      Chain().then(lambda: 42).run(), 42
    )
    # Async: same path but through the async-capable void chain.
    self.assertEqual(
      await Chain().then(aempty).then(lambda v: 42).run(), 42
    )

  async def test_call_with_explicit_args(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS: explicit positional args replace cv."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 1).then(lambda a, b: a + b, 10, 20).run()), 30
        )

  async def test_call_with_explicit_kwargs(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS: both positional and keyword args forwarded."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 1).then(lambda a, b=0: a + b, 5, b=3).run()), 8
        )

  async def test_call_without_args_ellipsis(self):
    """EVAL_CALL_WITHOUT_ARGS via ellipsis: callable invoked with no arguments."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 99).then(lambda: 7, ...).run()), 7
        )

  async def test_return_as_is_literal(self):
    """EVAL_RETURN_AS_IS: non-callable integer literal returned directly."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 1).then(42).run()), 42
        )

  async def test_return_as_is_string(self):
    """EVAL_RETURN_AS_IS: non-callable string literal returned directly."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 1).then("hello").run()), "hello"
        )

  async def test_kwargs_only_explicit_args(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS: kwargs-only callable with keyword arguments."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 1).then(lambda *, x=0: x, x=10).run()), 10
        )


# ---------------------------------------------------------------------------
# NestedChainEvalTests
# ---------------------------------------------------------------------------
class NestedChainEvalTests(IsolatedAsyncioTestCase):

  async def test_nested_chain_call_with_current_value(self):
    """is_chain + EVAL_CALL_WITH_CURRENT_VALUE: inner chain receives outer cv as root."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 5).then(Chain().then(lambda v: v * 3)).run()), 15
        )

  async def test_nested_chain_call_without_args(self):
    """is_chain + EVAL_CALL_WITHOUT_ARGS via ellipsis: inner chain runs with Null root."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 5).then(Chain(99), ...).run()), 99
        )

  async def test_nested_chain_call_with_explicit_args(self):
    """is_chain + EVAL_CALL_WITH_EXPLICIT_ARGS: first explicit arg becomes inner root."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 5).then(Chain().then(lambda v: v), 42).run()), 42
        )

  async def test_triple_nested_chain(self):
    """Three levels of nested chains propagating the current value through each."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 2).then(
            Chain().then(
              Chain().then(lambda v: v * 10)
            )
          ).run()),
          20
        )

  async def test_nested_chain_with_explicit_args_multiple(self):
    """is_chain + EVAL_CALL_WITH_EXPLICIT_ARGS: first arg is root, rest forwarded."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        # First explicit arg (10) becomes the inner chain's root value.
        # The inner chain's .then(lambda v: v) receives 10 and returns 10.
        self.assertEqual(
          await await_(Chain(fn, 1).then(Chain().then(lambda v: v), 10).run()), 10
        )


# ---------------------------------------------------------------------------
# SpecialValueTests
# ---------------------------------------------------------------------------
class SpecialValueTests(IsolatedAsyncioTestCase):

  async def test_builtin_len(self):
    """Built-in len() used as a callable link receiving the current value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, [1, 2, 3]).then(len).run()), 3
        )

  async def test_builtin_str(self):
    """Built-in str() used as a callable link converting int to string."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 42).then(str).run()), "42"
        )

  async def test_functools_partial(self):
    """functools.partial used as a callable link, receiving the current value."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 10).then(functools.partial(lambda a, b: a + b, b=5)).run()), 15
        )

  async def test_class_type_as_callable(self):
    """Class type (int) used as a callable link, acting as constructor."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, "42").then(int).run()), 42
        )

  async def test_falsy_literals_as_values(self):
    """Falsy non-callable literals returned as-is via EVAL_RETURN_AS_IS."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 1).then(0).run()), 0
        )
        self.assertEqual(
          await await_(Chain(fn, 1).then('').run()), ''
        )
        self.assertIs(
          await await_(Chain(fn, 1).then(False).run()), False
        )
        self.assertIsNone(
          await await_(Chain(fn, 1).then(None).run())
        )

  async def test_generator_function_as_link(self):
    """range (a callable) used as a link, receives cv and returns a range object."""
    for fn in [empty, aempty]:
      with self.subTest(fn=fn):
        self.assertEqual(
          await await_(Chain(fn, 3).then(range).run()), range(3)
        )
