from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, ChainAttr, CascadeAttr, QuentException, run, FrozenChain
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


# ---------------------------------------------------------------------------
# OrOperatorDispatchTests
# ---------------------------------------------------------------------------

class OrOperatorDispatchTests(MyTestCase):

  async def test_or_with_callable_runs_fn(self):
    """Chain(5) | fn executes fn with the chain's current value."""
    result = Chain(5) | (lambda v: v * 10) | run()
    await self.assertEqual(result, 50)

  async def test_or_with_chained_pipes(self):
    """Chain(5) | fn1 | fn2 pipes through multiple callables left-to-right."""
    result = Chain(5) | (lambda v: v + 3) | (lambda v: v * 2) | run()
    await self.assertEqual(result, 16)

  async def test_or_with_run_triggers_execution(self):
    """Piping into run() triggers chain execution and returns the result."""
    result = Chain(42) | run()
    await self.assertEqual(result, 42)

  async def test_or_with_run_root_override(self):
    """Piping into run(value) supplies a root value to a void chain."""
    result = Chain() | (lambda v: v * 3) | run(7)
    await self.assertEqual(result, 21)

  async def test_or_with_literal_value(self):
    """Piping a non-callable literal replaces the current value."""
    result = Chain(5) | 99 | run()
    await self.assertEqual(result, 99)

  async def test_or_returns_self_when_not_run(self):
    """The | operator returns the same chain object when other is not run()."""
    c = Chain(5)
    result = c | (lambda v: v * 2)
    super(MyTestCase, self).assertIs(result, c)

  async def test_or_with_inner_chain(self):
    """Piping a Chain into another uses it as a nested operation."""
    inner = Chain().then(lambda v: v * 3)
    result = Chain(4) | inner | run()
    await self.assertEqual(result, 12)

  async def test_or_with_async_callable(self):
    """Piping an async callable works transparently."""
    result = Chain(10) | aempty | (lambda v: v + 5) | run()
    await self.assertEqual(result, 15)

  async def test_or_triple_pipe_with_run_args(self):
    """Chain() | fn1 | fn2 | run(root) computes fn2(fn1(root))."""
    result = Chain() | (lambda v: v + 1) | (lambda v: v * 10) | run(2)
    await self.assertEqual(result, 30)

  async def test_or_with_run_root_callable_and_kwargs(self):
    """run() can supply a callable root with kwargs via pipe."""
    result = Chain() | (lambda v: v * 2) | run(lambda *, n=1: n, n=5)
    await self.assertEqual(result, 10)


# ---------------------------------------------------------------------------
# BoolAndNullTests
# ---------------------------------------------------------------------------

class BoolAndNullTests(TestCase):

  def test_bool_empty_chain_is_true(self):
    """An empty Chain is always truthy."""
    self.assertTrue(bool(Chain()))

  def test_bool_chain_with_value_is_true(self):
    """A Chain constructed with a value is truthy."""
    self.assertTrue(bool(Chain(42)))

  def test_bool_chain_with_none_is_true(self):
    """A Chain constructed with None is still truthy."""
    self.assertTrue(bool(Chain(None)))

  def test_bool_chain_with_zero_is_true(self):
    """A Chain constructed with 0 is still truthy."""
    self.assertTrue(bool(Chain(0)))

  def test_bool_chain_with_false_is_true(self):
    """A Chain constructed with False is still truthy."""
    self.assertTrue(bool(Chain(False)))

  def test_bool_cascade_is_true(self):
    """An empty Cascade is always truthy."""
    self.assertTrue(bool(Cascade()))

  def test_bool_cascade_with_value_is_true(self):
    """A Cascade with a value is truthy."""
    self.assertTrue(bool(Cascade(0)))

  def test_bool_chain_attr_is_true(self):
    """A ChainAttr is always truthy."""
    self.assertTrue(bool(ChainAttr()))

  def test_bool_cascade_attr_is_true(self):
    """A CascadeAttr is always truthy."""
    self.assertTrue(bool(CascadeAttr()))

  def test_bool_chain_with_links_is_true(self):
    """A Chain with appended links is truthy."""
    self.assertTrue(bool(Chain(1).then(lambda v: v)))

  def test_null_returns_pynull(self):
    """Chain.null() returns the PyNull sentinel."""
    self.assertIs(Chain.null(), PyNull)

  def test_null_identity_consistent(self):
    """Multiple calls to Chain.null() return the exact same object."""
    self.assertIs(Chain.null(), Chain.null())

  def test_null_is_not_none(self):
    """The Null sentinel is not None."""
    self.assertIsNot(Chain.null(), None)

  def test_null_repr(self):
    """The Null sentinel has a meaningful repr."""
    self.assertEqual(repr(Chain.null()), '<Null>')

  def test_null_accessible_from_cascade(self):
    """Cascade.null() returns the same Null sentinel as Chain.null()."""
    self.assertIs(Cascade.null(), Chain.null())


# ---------------------------------------------------------------------------
# FluentAPIContractTests
# ---------------------------------------------------------------------------

class FluentAPIContractTests(TestCase):

  def test_chain_then_returns_self(self):
    """then() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.then(lambda v: v), c)

  def test_chain_do_returns_self(self):
    """do() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.do(lambda v: v), c)

  def test_chain_root_returns_self(self):
    """root() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.root(lambda v: v), c)

  def test_chain_root_do_returns_self(self):
    """root_do() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.root_do(lambda v: v), c)

  def test_chain_attr_returns_self(self):
    """attr() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.attr('__class__'), c)

  def test_chain_attr_fn_returns_self(self):
    """attr_fn() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.attr_fn('__str__'), c)

  def test_chain_except_returns_self(self):
    """except_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.except_(lambda v: None, reraise=False), c)

  def test_chain_suppress_returns_self(self):
    """suppress() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.suppress(Exception), c)

  def test_chain_finally_returns_self(self):
    """finally_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.finally_(lambda v: None), c)

  def test_chain_on_success_returns_self(self):
    """on_success() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.on_success(lambda v: None), c)

  def test_chain_if_returns_self(self):
    """if_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.if_(lambda v: v), c)

  def test_chain_else_returns_self(self):
    """else_() returns the same chain instance after a preceding if_."""
    c = Chain(1).if_(lambda v: v)
    self.assertIs(c.else_(lambda v: None), c)

  def test_chain_if_not_returns_self(self):
    """if_not() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.if_not(lambda v: v), c)

  def test_chain_if_raise_returns_self(self):
    """if_raise() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.if_raise(TestExc()), c)

  def test_chain_else_raise_returns_self(self):
    """else_raise() returns the same chain instance after a preceding if_."""
    c = Chain(1).if_(lambda v: v)
    self.assertIs(c.else_raise(TestExc()), c)

  def test_chain_if_not_raise_returns_self(self):
    """if_not_raise() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.if_not_raise(TestExc()), c)

  def test_chain_condition_returns_self(self):
    """condition() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.condition(lambda v: True), c)

  def test_chain_not_returns_self(self):
    """not_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.not_(), c)

  def test_chain_eq_returns_self(self):
    """eq() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.eq(1), c)

  def test_chain_neq_returns_self(self):
    """neq() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.neq(2), c)

  def test_chain_is_returns_self(self):
    """is_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.is_(1), c)

  def test_chain_is_not_returns_self(self):
    """is_not() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.is_not(2), c)

  def test_chain_in_returns_self(self):
    """in_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.in_([1, 2, 3]), c)

  def test_chain_not_in_returns_self(self):
    """not_in() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.not_in([4, 5]), c)

  def test_chain_or_returns_self(self):
    """or_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.or_(42), c)

  def test_chain_isinstance_returns_self(self):
    """isinstance_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.isinstance_(int), c)

  def test_chain_raise_returns_self(self):
    """raise_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.raise_(TestExc()), c)

  def test_chain_sleep_returns_self(self):
    """sleep() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.sleep(0), c)

  def test_chain_while_true_returns_self(self):
    """while_true() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.while_true(lambda v: v), c)

  def test_chain_foreach_returns_self(self):
    """foreach() returns the same chain instance."""
    c = Chain([1, 2])
    self.assertIs(c.foreach(lambda v: v), c)

  def test_chain_foreach_do_returns_self(self):
    """foreach_do() returns the same chain instance."""
    c = Chain([1, 2])
    self.assertIs(c.foreach_do(lambda v: v), c)

  def test_chain_filter_returns_self(self):
    """filter() returns the same chain instance."""
    c = Chain([1, 2])
    self.assertIs(c.filter(lambda v: v), c)

  def test_chain_reduce_returns_self(self):
    """reduce() returns the same chain instance."""
    c = Chain([1, 2])
    self.assertIs(c.reduce(lambda a, b: a + b), c)

  def test_chain_gather_returns_self(self):
    """gather() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.gather(lambda v: v, lambda v: v + 1), c)

  def test_chain_with_returns_self(self):
    """with_() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.with_(lambda v: v), c)

  def test_chain_with_do_returns_self(self):
    """with_do() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.with_do(lambda v: v), c)

  def test_chain_config_returns_self(self):
    """config() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.config(), c)

  def test_chain_autorun_returns_self(self):
    """autorun() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.autorun(False), c)

  def test_chain_set_async_returns_self(self):
    """set_async() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.set_async(False), c)

  def test_chain_with_context_returns_self(self):
    """with_context() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.with_context(key='val'), c)

  def test_chain_pipe_returns_self(self):
    """pipe() returns the same chain instance."""
    c = Chain(1)
    self.assertIs(c.pipe(lambda v: v), c)

  def test_cascade_fluent_api_contract(self):
    """Cascade methods also return self, preserving fluent chaining."""
    c = Cascade(1)
    self.assertIs(c.then(lambda v: v), c)
    self.assertIs(c.do(lambda v: v), c)
    self.assertIs(c.root(lambda v: v), c)
    self.assertIs(c.root_do(lambda v: v), c)
    self.assertIs(c.if_(lambda v: v), c)
    self.assertIs(c.else_(lambda v: None), c)
    self.assertIs(c.config(), c)
    self.assertIs(c.pipe(lambda v: v), c)
    self.assertIs(c.with_context(k='v'), c)
