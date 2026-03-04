"""Exhaustive tests for chain construction, value types, Cascade behavior,
cloning, freezing, decorators, pipe syntax, and the public API surface.

Covers scenarios A through M as specified — 130+ test methods exercising every
documented path through Chain, Cascade, clone, freeze, decorator, pipe, config,
no_async, __bool__, __repr__, and the public API surface.
"""
import asyncio
import math
import types
import operator
import functools
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Cascade, QuentException, run, Null
from tests.utils import empty, aempty, await_, TestExc, MyTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add(a, b):
  return a + b


def _mul(a, b):
  return a * b


async def _async_add_ten(v):
  return v + 10


class _CallableClass:
  """Object with __call__."""
  def __init__(self, factor=1):
    self.factor = factor

  def __call__(self, v=None):
    if v is None:
      return self.factor
    return v * self.factor


class _SimpleCtxMgr:
  """Minimal sync context manager for tests."""
  def __init__(self, value):
    self.value = value
    self.entered = False
    self.exited = False

  def __enter__(self):
    self.entered = True
    return self.value

  def __exit__(self, *args):
    self.exited = True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# A. Chain Construction — Root Values  (30 tests)
# ═══════════════════════════════════════════════════════════════════════════

class ChainConstructionRootValues(MyTestCase):
  """A1–A30: exercise every supported root-value type."""

  # A1
  async def test_chain_no_root_returns_none(self):
    await self.assertIsNone(Chain().run())

  # A2  — None is a *value*, not Null
  async def test_chain_none_root(self):
    result = Chain(None).run()
    await self.assertIsNone(result)
    # confirm it's actually stored (not treated as void)
    with self.assertRaises(QuentException):
      Chain(None).run(99)

  # A3
  async def test_chain_zero_root(self):
    await self.assertEqual(Chain(0).run(), 0)

  # A4
  async def test_chain_false_root(self):
    await self.assertFalse(Chain(False).run())

  # A5
  async def test_chain_empty_string_root(self):
    await self.assertEqual(Chain('').run(), '')

  # A6
  async def test_chain_empty_list_root(self):
    await self.assertEqual(Chain([]).run(), [])

  # A7
  async def test_chain_empty_dict_root(self):
    await self.assertEqual(Chain({}).run(), {})

  # A8
  async def test_chain_empty_tuple_root(self):
    await self.assertEqual(Chain(()).run(), ())

  # A9
  async def test_chain_empty_set_root(self):
    await self.assertEqual(Chain(set()).run(), set())

  # A10
  async def test_chain_nan_root(self):
    result = Chain(float('nan')).run()
    super(MyTestCase, self).assertTrue(math.isnan(result))

  # A11
  async def test_chain_inf_root(self):
    await self.assertEqual(Chain(float('inf')).run(), float('inf'))

  # A12
  async def test_chain_large_integer_root(self):
    huge = 10 ** 100
    await self.assertEqual(Chain(huge).run(), huge)

  # A13
  async def test_chain_bytes_root(self):
    await self.assertEqual(Chain(b'hello').run(), b'hello')

  # A14
  async def test_chain_complex_root(self):
    await self.assertEqual(Chain(1 + 2j).run(), 1 + 2j)

  # A15
  async def test_chain_frozenset_root(self):
    fs = frozenset([1, 2, 3])
    await self.assertEqual(Chain(fs).run(), fs)

  # A16 — callable root is invoked
  async def test_chain_callable_root_invoked(self):
    await self.assertEqual(Chain(lambda: 42).run(), 42)

  # A17 — callable root + positional args
  async def test_chain_callable_root_with_args(self):
    await self.assertEqual(Chain(int, '42').run(), 42)

  # A18 — callable root + kwargs
  async def test_chain_callable_root_with_kwargs(self):
    await self.assertEqual(Chain(dict, a=1).run(), {'a': 1})

  # A19 — callable root + args + kwargs
  async def test_chain_callable_root_with_args_and_kwargs(self):
    def fn(a, b, c=0):
      return a + b + c
    await self.assertEqual(Chain(fn, 1, 2, c=3).run(), 6)

  # A20 — Ellipsis first arg → call without args
  async def test_chain_ellipsis_calls_without_args(self):
    await self.assertEqual(Chain(lambda: 99, ...).run(), 99)

  # A21 — class as root (constructor)
  async def test_chain_class_root_constructor(self):
    await self.assertEqual(Chain(list).run(), [])

  # A22 — built-in function
  async def test_chain_builtin_function_root(self):
    await self.assertEqual(Chain(len, 'hello').run(), 5)

  # A23 — async callable root
  async def test_chain_async_callable_root(self):
    await self.assertEqual(Chain(aempty, 42).run(), 42)

  # A24 — another Chain as root
  async def test_chain_nested_chain_root(self):
    inner = Chain(42)
    await self.assertEqual(Chain(inner).run(), 42)

  # A25 — Cascade as root
  async def test_chain_cascade_as_root(self):
    inner = Cascade(99).then(lambda v: v * 2)
    await self.assertEqual(Chain(inner).run(), 99)

  # A26 — functools.partial as root
  async def test_chain_partial_root(self):
    p = functools.partial(_add, 10, 20)
    await self.assertEqual(Chain(p, ...).run(), 30)

  # A27 — callable class instance
  async def test_chain_callable_class_root(self):
    obj = _CallableClass(factor=5)
    await self.assertEqual(Chain(obj, 3).run(), 15)

  # A28 — staticmethod/classmethod (accessed from class)
  async def test_chain_staticmethod_root(self):
    class Klass:
      @staticmethod
      def s():
        return 'static'
    await self.assertEqual(Chain(Klass.s, ...).run(), 'static')

  # A29 — property getter via direct callable
  async def test_chain_property_getter_root(self):
    class Obj:
      @property
      def val(self):
        return 77
    obj = Obj()
    getter = type(obj).__dict__['val'].fget
    await self.assertEqual(Chain(getter, obj).run(), 77)

  # A30 — operator.itemgetter / attrgetter
  async def test_chain_operator_itemgetter_root(self):
    ig = operator.itemgetter(1)
    await self.assertEqual(Chain(ig, [10, 20, 30]).run(), 20)

  async def test_chain_operator_attrgetter_root(self):
    ag = operator.attrgetter('__class__')
    result = Chain(ag, 42).run()
    super(MyTestCase, self).assertIs(result, int)


# ═══════════════════════════════════════════════════════════════════════════
# B. Chain Construction — EvalCode Paths  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════

class ChainEvalCodePaths(MyTestCase):
  """B31–B38: verify every EvalCode dispatch path."""

  # B31 — EVAL_CALL_WITH_CURRENT_VALUE: callable, no args → receives previous value
  async def test_eval_call_with_current_value(self):
    await self.assertEqual(
      Chain(10).then(lambda v: v + 5).run(), 15
    )

  # B32 — EVAL_CALL_WITH_CURRENT_VALUE with Null current → called with zero args
  async def test_eval_call_with_null_current(self):
    await self.assertEqual(
      Chain().then(lambda: 'hello').run(), 'hello'
    )

  # B33 — EVAL_CALL_WITH_EXPLICIT_ARGS: callable with positional args
  async def test_eval_call_with_explicit_args(self):
    await self.assertEqual(
      Chain(1).then(_add, 2, 3).run(), 5
    )

  # B34 — EVAL_CALL_WITH_EXPLICIT_ARGS: callable with kwargs only (non-chain)
  async def test_eval_call_with_kwargs_only(self):
    await self.assertEqual(
      Chain(1).then(lambda *, extra=0: extra, extra=10).run(), 10
    )

  # B35 — EVAL_CALL_WITHOUT_ARGS: callable with ... as first arg
  async def test_eval_call_without_args(self):
    await self.assertEqual(
      Chain(99).then(lambda: 'no_arg', ...).run(), 'no_arg'
    )

  # B36 — EVAL_RETURN_AS_IS: non-callable literal in then()
  async def test_eval_return_as_is(self):
    await self.assertEqual(
      Chain(1).then(42).run(), 42
    )

  # B37 — non-callable with allow_literal=False in do() raises QuentException
  async def test_noncallable_in_do_raises(self):
    with self.assertRaises(QuentException):
      Chain(1).do(True)

  # B38 — error message for non-callable in do()
  async def test_noncallable_in_do_error_message(self):
    try:
      Chain(1).do(42)
      self.fail('Expected QuentException')  # pragma: no cover
    except QuentException as exc:
      super(MyTestCase, self).assertIn('Non-callable', str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# C. Value Propagation  (12 tests)
# ═══════════════════════════════════════════════════════════════════════════

class ValuePropagation(MyTestCase):
  """C39–C50: verify value flows through the chain correctly."""

  # C39
  async def test_single_then(self):
    await self.assertEqual(Chain(42).then(lambda v: v * 2).run(), 84)

  # C40
  async def test_two_thens(self):
    await self.assertEqual(
      Chain(42).then(lambda v: v * 2).then(lambda v: v + 1).run(), 85
    )

  # C41 — literal in then()
  async def test_literal_in_then(self):
    await self.assertEqual(Chain().then(42).run(), 42)

  # C42 — literal followed by callable
  async def test_literal_then_callable(self):
    await self.assertEqual(
      Chain().then(42).then(lambda v: v * 2).run(), 84
    )

  # C43 — void chain returns None
  async def test_void_chain_returns_none(self):
    await self.assertIsNone(Chain().run())

  # C44 — explicit None return
  async def test_explicit_none_return(self):
    await self.assertIsNone(Chain().then(lambda: None).run())

  # C45 — do() discards result, previous value flows
  async def test_do_discards_result(self):
    side = []
    await self.assertEqual(
      Chain(10).do(lambda v: side.append(v * 100)).run(), 10
    )
    super(MyTestCase, self).assertEqual(side, [1000])

  # C46 — do() on void chain
  async def test_do_on_void_chain(self):
    await self.assertIsNone(Chain().do(lambda: 42).run())

  # C47 — do() result discarded with root
  async def test_do_result_discarded_with_root(self):
    await self.assertEqual(
      Chain(10).do(lambda v: v * 2).run(), 10
    )

  # C48 — do() then then()
  async def test_do_then_then(self):
    await self.assertEqual(
      Chain(10).do(lambda v: v * 2).then(lambda v: v + 1).run(), 11
    )

  # C49 — long then chain
  async def test_long_then_chain(self):
    c = Chain(0)
    for _ in range(10):
      c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 10)

  # C50 — async value propagation
  async def test_async_value_propagation(self):
    await self.assertEqual(
      Chain(aempty, 5).then(lambda v: v * 3).run(), 15
    )


# ═══════════════════════════════════════════════════════════════════════════
# D. Cascade Comprehensive  (16 tests)
# ═══════════════════════════════════════════════════════════════════════════

class CascadeComprehensive(MyTestCase):
  """D49–D63: Cascade passes root to every link, returns root."""

  # D49
  async def test_cascade_returns_root_not_last(self):
    await self.assertEqual(
      Cascade(42).then(lambda v: v * 2).run(), 42
    )

  # D50
  async def test_cascade_returns_root_multiple_links(self):
    await self.assertEqual(
      Cascade(42).then(lambda v: v * 2).then(lambda v: v + 1).run(), 42
    )

  # D51 — every link receives the root
  async def test_cascade_passes_root_to_every_link(self):
    received = []
    Cascade(99).then(lambda v: received.append(v)).then(lambda v: received.append(v)).run()
    super(MyTestCase, self).assertEqual(received, [99, 99])

  # D52 — do() still receives root
  async def test_cascade_do_receives_root(self):
    received = []
    Cascade(77).do(lambda v: received.append(v)).run()
    super(MyTestCase, self).assertEqual(received, [77])

  # D53 — no root, override at run
  async def test_cascade_run_override(self):
    await self.assertEqual(
      Cascade().then(lambda v: v * 2).run(42), 42
    )

  # D54 — async links
  async def test_cascade_async_links(self):
    await self.assertEqual(
      Cascade(aempty, 10).then(lambda v: v * 2).run(), 10
    )

  # D55 — exception handler
  async def test_cascade_exception_handler(self):
    caught = []
    def handler(v):
      caught.append(v)
    result = Cascade(5).then(lambda v: (_ for _ in ()).throw(ValueError())).except_(
      handler, reraise=False
    ).run()
    super(MyTestCase, self).assertEqual(len(caught), 1)

  # D56 — finally handler
  async def test_cascade_finally_handler(self):
    log = []
    Cascade(5).then(lambda v: v).finally_(lambda v: log.append('fin')).run()
    super(MyTestCase, self).assertEqual(log, ['fin'])

  # D57 — simple path (no exception handlers)
  async def test_cascade_simple_path(self):
    await self.assertEqual(
      Cascade(10).then(lambda v: v + 1).then(lambda v: v + 2).run(), 10
    )

  # D58 — nested Chain inside Cascade
  async def test_cascade_nested_chain_inside(self):
    inner = Chain().then(lambda v: v * 100)
    result = Cascade(3).then(inner).run()
    await self.assertEqual(result, 3)

  # D59 — clone preserves is_cascade
  async def test_cascade_clone_preserves_flag(self):
    c = Cascade(42).then(lambda v: v * 2)
    c2 = c.clone()
    super(MyTestCase, self).assertIsInstance(c2, Cascade)
    await self.assertEqual(c2.run(), 42)

  # D60 — foreach
  async def test_cascade_foreach(self):
    result = Cascade([1, 2, 3]).foreach(lambda x: x * 2).run()
    await self.assertEqual(result, [1, 2, 3])

  # D61 — filter
  async def test_cascade_filter(self):
    result = Cascade([1, 2, 3, 4]).filter(lambda x: x % 2 == 0).run()
    await self.assertEqual(result, [1, 2, 3, 4])

  # D62 — gather
  async def test_cascade_gather(self):
    result = Cascade(5).gather(lambda v: v + 1, lambda v: v + 2).run()
    await self.assertEqual(result, 5)

  # D63 — context manager
  async def test_cascade_with_context_manager(self):
    ctx_mgr = _SimpleCtxMgr(42)
    result = Cascade(ctx_mgr).with_(lambda v: v * 2).run()
    await self.assertEqual(result, ctx_mgr)
    super(MyTestCase, self).assertTrue(ctx_mgr.entered)
    super(MyTestCase, self).assertTrue(ctx_mgr.exited)

  # D-extra: Cascade is a Chain subclass
  async def test_cascade_is_chain_subclass(self):
    super(MyTestCase, self).assertTrue(issubclass(Cascade, Chain))
    super(MyTestCase, self).assertIsInstance(Cascade(1), Chain)


# ═══════════════════════════════════════════════════════════════════════════
# E. Clone Comprehensive  (15 tests)
# ═══════════════════════════════════════════════════════════════════════════

class CloneComprehensive(MyTestCase):
  """E64–E78: clone independence, field preservation, execution isolation."""

  # E64 — clone with root_link only
  async def test_clone_root_only(self):
    c = Chain(42)
    c2 = c.clone()
    await self.assertEqual(c2.run(), 42)

  # E65 — clone with root_link + first_link
  async def test_clone_root_and_first(self):
    c = Chain(10).then(lambda v: v + 5)
    c2 = c.clone()
    await self.assertEqual(c2.run(), 15)

  # E66 — clone with root + first + current (3+ links)
  async def test_clone_root_first_current(self):
    c = Chain(1).then(lambda v: v + 1).then(lambda v: v + 1).then(lambda v: v + 1)
    c2 = c.clone()
    await self.assertEqual(c2.run(), 4)

  # E67 — clone empty chain
  async def test_clone_empty_chain(self):
    c = Chain()
    c2 = c.clone()
    await self.assertIsNone(c2.run())

  # E68 — clone no root, first_link only
  async def test_clone_no_root_first_only(self):
    c = Chain().then(lambda: 10)
    c2 = c.clone()
    await self.assertEqual(c2.run(), 10)

  # E69 — clone no root, first + current
  async def test_clone_no_root_first_and_current(self):
    c = Chain().then(lambda: 5).then(lambda v: v * 2)
    c2 = c.clone()
    await self.assertEqual(c2.run(), 10)

  # E70 — clone preserves config flags (_is_sync is the only readonly flag)
  async def test_clone_preserves_flags(self):
    c = Chain(1).no_async(True)
    c2 = c.clone()
    super(MyTestCase, self).assertTrue(c2._is_sync)
    # Also verify autorun is preserved by observing Task return
    c3 = Chain(aempty, 1).config(autorun=True)
    c4 = c3.clone()
    result = c4.run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 1)

  # E71 — clone resets is_nested
  async def test_clone_resets_is_nested(self):
    inner = Chain().then(lambda v: v + 1)
    # Nesting marks inner.is_nested = True
    outer = Chain(5).then(inner)
    outer.run()
    c2 = inner.clone()
    # clone should not be nested — can run directly
    await self.assertEqual(c2.run(10), 11)

  # E72 — clone with on_finally_link
  async def test_clone_with_finally(self):
    log = []
    c = Chain(1).then(lambda v: v).finally_(lambda v: log.append('fin'))
    c2 = c.clone()
    log.clear()
    c2.run()
    super(MyTestCase, self).assertEqual(log, ['fin'])

  # E73 — modifying clone does not affect original
  async def test_modify_clone_not_original(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c2.then(lambda v: v * 100)
    await self.assertEqual(c.run(), 2)
    await self.assertEqual(c2.run(), 200)

  # E74 — modifying original does not affect clone
  async def test_modify_original_not_clone(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c.then(lambda v: v * 100)
    await self.assertEqual(c.run(), 200)
    await self.assertEqual(c2.run(), 2)

  # E75 — executing clone does not affect original
  async def test_execute_clone_independent(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    r1 = c2.run()
    r2 = c.run()
    await self.assertEqual(r1, 2)
    await self.assertEqual(r2, 2)

  # E76 — clone of a clone
  async def test_clone_of_clone(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    c3 = c2.clone()
    await self.assertEqual(c3.run(), 2)

  # E77 — clone shares callable references (shallow copy for v)
  async def test_clone_shares_callables(self):
    fn = lambda v: v + 1
    c = Chain(1).then(fn)
    c2 = c.clone()
    # Both chains use the same lambda object — shared reference
    await self.assertEqual(c.run(), c2.run())

  # E78 — running cloned chain multiple times — no state leakage
  async def test_clone_no_state_leakage(self):
    c = Chain(1).then(lambda v: v + 1)
    c2 = c.clone()
    await self.assertEqual(c2.run(), 2)
    await self.assertEqual(c2.run(), 2)
    await self.assertEqual(c2.run(), 2)


# ═══════════════════════════════════════════════════════════════════════════
# F. FrozenChain and Decorator  (12 tests)
# ═══════════════════════════════════════════════════════════════════════════

class FrozenChainAndDecorator(MyTestCase):
  """F79–F90: freeze, decorator, _DescriptorWrapper."""

  # F79
  async def test_freeze_returns_frozen(self):
    frozen = Chain(42).freeze()
    # _FrozenChain is not directly importable, but it has .run() and __call__
    super(MyTestCase, self).assertTrue(hasattr(frozen, 'run'))
    super(MyTestCase, self).assertTrue(callable(frozen))

  # F80
  async def test_frozen_run(self):
    frozen = Chain(42).then(lambda v: v + 1).freeze()
    await self.assertEqual(frozen.run(), 43)

  # F81
  async def test_frozen_call_shorthand(self):
    frozen = Chain(42).then(lambda v: v + 1).freeze()
    await self.assertEqual(frozen(), 43)

  # F82
  async def test_frozen_run_with_override(self):
    frozen = Chain().then(lambda v: v * 3).freeze()
    await self.assertEqual(frozen.run(7), 21)

  # F83 — multiple executions, no state leakage
  async def test_frozen_reusable_no_leakage(self):
    frozen = Chain().then(lambda v: v ** 2).freeze()
    await self.assertEqual(frozen.run(3), 9)
    await self.assertEqual(frozen.run(4), 16)
    await self.assertEqual(frozen.run(5), 25)

  # F84
  async def test_frozen_decorator_returns_callable(self):
    dec = Chain().then(lambda v: v * 2).freeze().decorator()
    super(MyTestCase, self).assertTrue(callable(dec))

  # F85
  async def test_decorator_wraps_function(self):
    @Chain().then(lambda v: v * 2).decorator()
    def double(x):
      return x
    await self.assertEqual(double(5), 10)

  # F86 — decorator on sync function
  async def test_decorator_sync_function(self):
    @Chain().then(lambda v: v + 100).decorator()
    def add100(x):
      return x
    await self.assertEqual(add100(5), 105)

  # F87 — decorator on async function
  async def test_decorator_async_function(self):
    @Chain().then(lambda v: v * 5).decorator()
    async def afn(x):
      return x + 1
    await self.assertEqual(afn(2), 15)  # (2+1)*5

  # F88 — functools.update_wrapper preserves __name__
  async def test_decorator_preserves_name(self):
    @Chain().then(lambda v: v).decorator()
    def my_special_func(x):
      return x
    super(MyTestCase, self).assertEqual(my_special_func.__name__, 'my_special_func')

  # F89 — _DescriptorWrapper.__get__ returns bound method for instances
  async def test_descriptor_get_instance(self):
    class MyClass:
      @Chain().then(lambda v: v * 2).decorator()
      def double(self, x):
        return x
    obj = MyClass()
    bound = MyClass.double.__get__(obj, MyClass)
    super(MyTestCase, self).assertIsInstance(bound, types.MethodType)

  # F90 — _DescriptorWrapper.__get__ returns self for class access
  async def test_descriptor_get_class(self):
    class MyClass:
      @Chain().then(lambda v: v).decorator()
      def method(self, x):
        return x
    wrapper = MyClass.__dict__['method']
    result = wrapper.__get__(None, MyClass)
    super(MyTestCase, self).assertIs(result, wrapper)


# ═══════════════════════════════════════════════════════════════════════════
# G. Pipe Syntax  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════

class PipeSyntax(MyTestCase):
  """G91–G98: the | (or) operator for building and executing chains."""

  # G91
  async def test_pipe_single_fn(self):
    result = Chain(10) | (lambda v: v + 5) | run()
    await self.assertEqual(result, 15)

  # G92
  async def test_pipe_multiple_fns(self):
    result = Chain(2) | (lambda v: v + 3) | (lambda v: v * 10) | run()
    await self.assertEqual(result, 50)

  # G93
  async def test_pipe_fn_then_run(self):
    result = Chain(3) | (lambda v: v ** 2) | run()
    await self.assertEqual(result, 9)

  # G94
  async def test_pipe_run_with_override(self):
    result = Chain() | (lambda v: v * 4) | run(5)
    await self.assertEqual(result, 20)

  # G95 — literal value
  async def test_pipe_literal_value(self):
    result = Chain(1) | 99 | run()
    await self.assertEqual(result, 99)

  # G96 — another chain
  async def test_pipe_nested_chain(self):
    inner = Chain().then(lambda v: v * 10)
    result = Chain(3) | inner | run()
    await self.assertEqual(result, 30)

  # G97 — void chain with run override
  async def test_pipe_void_chain_with_override(self):
    result = Chain() | (lambda v: v + 1) | run(100)
    await self.assertEqual(result, 101)

  # G98 — complex pipe
  async def test_pipe_complex_chain(self):
    result = (
      Chain(1)
      | (lambda v: v + 1)
      | (lambda v: v * 2)
      | (lambda v: v + 10)
      | (lambda v: v // 2)
      | run()
    )
    await self.assertEqual(result, 7)  # ((1+1)*2+10)//2 = 7


# ═══════════════════════════════════════════════════════════════════════════
# H. config() Method  (6 tests)
# ═══════════════════════════════════════════════════════════════════════════

class ConfigMethod(MyTestCase):
  """H99–H104: config(autorun, debug).
  _autorun and _debug are cdef fields (not accessible from Python),
  so we test through observable behaviour.
  """

  # H99 — autorun=True: async chain returns a Task instead of a coroutine
  async def test_config_autorun_true(self):
    c = Chain(aempty, 1).config(autorun=True)
    result = c.run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 1)

  # H100 — autorun=False: async chain returns a coroutine (default)
  async def test_config_autorun_false(self):
    c = Chain(aempty, 1).config(autorun=True).config(autorun=False)
    result = c.run()
    # Not a Task, it's a coroutine
    import inspect
    super(MyTestCase, self).assertTrue(inspect.isawaitable(result))
    super(MyTestCase, self).assertNotIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 1)

  # H101 — debug=True: chain still produces correct result (exercises debug path)
  async def test_config_debug_true(self):
    c = Chain(10).then(lambda v: v + 5).config(debug=True)
    await self.assertEqual(c.run(), 15)

  # H102 — debug=False: chain still produces correct result
  async def test_config_debug_false(self):
    c = Chain(10).then(lambda v: v + 5).config(debug=True).config(debug=False)
    await self.assertEqual(c.run(), 15)

  # H103 — both autorun + debug
  async def test_config_both(self):
    c = Chain(aempty, 10).then(lambda v: v + 5).config(autorun=True, debug=True)
    result = c.run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 15)

  # H104 — no args, no change (autorun still active from prior config)
  async def test_config_no_args(self):
    c = Chain(aempty, 1).config(autorun=True)
    c.config()
    result = c.run()
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 1)


# ═══════════════════════════════════════════════════════════════════════════
# I. no_async() Method  (4 tests)
# ═══════════════════════════════════════════════════════════════════════════

class NoAsyncMethod(MyTestCase):
  """I105–I108: no_async() disables async detection."""

  # I105
  async def test_no_async_true(self):
    c = Chain(1).no_async(True)
    super(MyTestCase, self).assertTrue(c._is_sync)

  # I106
  async def test_no_async_false_explicit(self):
    c = Chain(1).no_async(True).no_async(False)
    super(MyTestCase, self).assertFalse(c._is_sync)

  # I107
  async def test_no_async_default(self):
    c = Chain(1).no_async()
    super(MyTestCase, self).assertTrue(c._is_sync)

  # I108 — no_async(True) returns coroutine objects as-is (not awaited)
  async def test_no_async_returns_coro_as_is(self):
    async def coro_fn(v):
      return v + 10

    c = Chain(1).then(coro_fn).no_async(True)
    result = c.run()
    # result should be a coroutine object, not 11
    import inspect
    super(MyTestCase, self).assertTrue(inspect.iscoroutine(result))
    # Clean up the coroutine
    awaited = await result
    super(MyTestCase, self).assertEqual(awaited, 11)


# ═══════════════════════════════════════════════════════════════════════════
# J. __bool__ and __repr__  (4 tests)
# ═══════════════════════════════════════════════════════════════════════════

class BoolAndRepr(MyTestCase):
  """J109–J112: truthiness and string representation."""

  # J109
  async def test_bool_empty_chain(self):
    super(MyTestCase, self).assertTrue(bool(Chain()))

  # J110
  async def test_bool_chain_with_root(self):
    super(MyTestCase, self).assertTrue(bool(Chain(42)))

  # J111
  async def test_repr_empty_chain(self):
    r = repr(Chain())
    super(MyTestCase, self).assertIsInstance(r, str)
    super(MyTestCase, self).assertIn('Chain', r)

  # J112
  async def test_repr_chain_with_links(self):
    r = repr(Chain(42).then(lambda v: v))
    super(MyTestCase, self).assertIsInstance(r, str)
    super(MyTestCase, self).assertIn('Chain', r)
    super(MyTestCase, self).assertIn('then', r)


# ═══════════════════════════════════════════════════════════════════════════
# K. Public API Surface  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════

class PublicAPISurface(MyTestCase):
  """K113–K118: verify all exports and sentinel behaviour."""

  # K113 — all exports available
  async def test_all_exports(self):
    from quent import Chain, Cascade, QuentException, run, Null, ResultOrAwaitable, __version__
    super(MyTestCase, self).assertIsNotNone(Chain)
    super(MyTestCase, self).assertIsNotNone(Cascade)
    super(MyTestCase, self).assertIsNotNone(QuentException)
    super(MyTestCase, self).assertIsNotNone(run)
    super(MyTestCase, self).assertIsNotNone(Null)
    super(MyTestCase, self).assertIsNotNone(ResultOrAwaitable)
    super(MyTestCase, self).assertIsNotNone(__version__)

  # K114 — Null is PyNull from quent.quent
  async def test_null_is_pynull(self):
    from quent.quent import PyNull
    super(MyTestCase, self).assertIs(Null, PyNull)

  # K115 — Null repr
  async def test_null_repr(self):
    super(MyTestCase, self).assertEqual(repr(Null), '<Null>')

  # K116 — QuentException is Exception subclass
  async def test_quent_exception_is_exception(self):
    super(MyTestCase, self).assertTrue(issubclass(QuentException, Exception))

  # K117 — run class stores root_value, args, kwargs
  async def test_run_stores_fields(self):
    r = run(10, 20, 30, key='val')
    super(MyTestCase, self).assertEqual(r.root_value, 10)
    super(MyTestCase, self).assertEqual(r.args, (20, 30))
    super(MyTestCase, self).assertEqual(r.kwargs, {'key': 'val'})

  # K117b — run with no args
  async def test_run_default_root_is_null(self):
    r = run()
    super(MyTestCase, self).assertIs(r.root_value, Null)

  # K118 — except_() raises TypeError for string exceptions arg
  async def test_except_string_exceptions_raises_type_error(self):
    with self.assertRaises(TypeError) as cm:
      Chain(1).except_(lambda v: v, exceptions='ValueError')
    super(MyTestCase, self).assertIn('not string', str(cm.exception))

  # K-extra — __version__ is a string
  async def test_version_is_string(self):
    from quent import __version__
    super(MyTestCase, self).assertIsInstance(__version__, str)


# ═══════════════════════════════════════════════════════════════════════════
# L. Stress and Boundary  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════

class StressAndBoundary(MyTestCase):
  """L119–L123: chains with extreme link counts and nesting depths."""

  # L119 — 100 links
  async def test_100_links(self):
    c = Chain(0)
    for _ in range(100):
      c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 100)

  # L120 — 1000 links
  async def test_1000_links(self):
    c = Chain(0)
    for _ in range(1000):
      c.then(lambda v: v + 1)
    await self.assertEqual(c.run(), 1000)

  # L121 — chain with 0 links, root only
  async def test_zero_links_root_only(self):
    await self.assertEqual(Chain(42).run(), 42)

  # L122 — chain with exactly 1 link
  async def test_one_link(self):
    await self.assertEqual(Chain(42).then(lambda v: v + 1).run(), 43)

  # L123 — deeply nested chains (10+ levels)
  async def test_deeply_nested_chains(self):
    chain = Chain().then(lambda v: v + 1)
    for _ in range(10):
      chain = Chain().then(chain)
    await self.assertEqual(chain.run(0), 1)


# ═══════════════════════════════════════════════════════════════════════════
# M. Edge Cases and Interactions  (8 tests)
# ═══════════════════════════════════════════════════════════════════════════

class EdgeCasesAndInteractions(MyTestCase):
  """M124–M131: corner cases involving Null, self-references, reuse, etc."""

  # M124 — link returns Null sentinel → chain converts to None
  async def test_link_returns_null_sentinel_becomes_none(self):
    # When the internal Null sentinel leaks as a return value, the chain
    # should convert it to None at the boundary.
    # We can't easily return the *internal* Null from user code, but we
    # can test that the PyNull exported behaves correctly.
    # Chain(Null) is treated as a void chain (Null == no root).
    await self.assertIsNone(Chain(Null).run())

  # M125 — using the same chain as a nested link marks it as nested;
  # running it directly raises QuentException
  async def test_chain_self_reference_marks_nested(self):
    c = Chain(42)
    # Passing c into c.then(c) marks c.is_nested = True
    c.then(c)
    with self.assertRaises(QuentException):
      c.run()

  # M125b — a void nested chain receives the current value from the parent
  async def test_void_chain_as_nested_receives_current(self):
    inner = Chain().then(lambda v: v + 1)
    c = Chain(42).then(inner)
    # inner receives 42 as root override from parent, returns 42+1=43
    await self.assertEqual(c.run(), 43)

  # M126 — calling run() multiple times on the same chain
  async def test_run_multiple_times_same_chain(self):
    c = Chain(10).then(lambda v: v + 1)
    await self.assertEqual(c.run(), 11)
    await self.assertEqual(c.run(), 11)
    await self.assertEqual(c.run(), 11)

  # M127 — except_ before any then links
  async def test_except_before_then(self):
    caught = []
    c = Chain(1).except_(lambda v: caught.append('exc'), reraise=False)
    # No error occurs, so handler is not called
    c.run()
    super(MyTestCase, self).assertEqual(caught, [])

  # M128 — finally_ before any then links
  async def test_finally_before_then(self):
    log = []
    c = Chain(1).finally_(lambda v: log.append('fin'))
    c.run()
    super(MyTestCase, self).assertEqual(log, ['fin'])

  # M129 — chain returns self from then/do/except_/finally_/config/no_async
  async def test_fluent_returns_self(self):
    c = Chain(1)
    super(MyTestCase, self).assertIs(c.then(lambda v: v), c)
    c2 = Chain(1)
    super(MyTestCase, self).assertIs(c2.do(lambda v: v), c2)
    c3 = Chain(1)
    super(MyTestCase, self).assertIs(c3.except_(lambda v: v), c3)
    c4 = Chain(1)
    super(MyTestCase, self).assertIs(c4.finally_(lambda v: v), c4)
    c5 = Chain(1)
    super(MyTestCase, self).assertIs(c5.config(), c5)
    c6 = Chain(1)
    super(MyTestCase, self).assertIs(c6.no_async(), c6)

  # M130 — cannot override root when chain already has one
  async def test_cannot_override_root(self):
    with self.assertRaises(QuentException):
      Chain(1).run(2)

  # M131 — cannot run nested chain directly
  async def test_cannot_run_nested_chain_directly(self):
    inner = Chain().then(lambda v: v + 1)
    outer = Chain(5).then(inner)
    # inner is now marked as nested
    with self.assertRaises(QuentException):
      inner.run(10)


# ═══════════════════════════════════════════════════════════════════════════
# N. Additional Coverage — sync/async dual-path tests  (10+ tests)
# ═══════════════════════════════════════════════════════════════════════════

class SyncAsyncDualPath(MyTestCase):
  """Additional tests running both sync and async variants via with_fn()."""

  async def test_chain_root_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(Chain(fn, 42).run(), 42)

  async def test_chain_then_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 10).then(lambda v: v + 5).run(), 15
        )

  async def test_chain_do_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        side = []
        result = Chain(fn, 10).do(lambda v: side.append(v)).run()
        await self.assertEqual(result, 10)
        super(MyTestCase, self).assertEqual(side, [10])

  async def test_cascade_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Cascade(fn, 99).then(lambda v: v * 2).run(), 99
        )

  async def test_clone_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        c = Chain(fn, 5).then(lambda v: v + 1)
        c2 = c.clone()
        await self.assertEqual(c2.run(), 6)

  async def test_freeze_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        frozen = Chain(fn, 7).then(lambda v: v * 3).freeze()
        await self.assertEqual(frozen.run(), 21)

  async def test_pipe_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(fn, 4) | (lambda v: v + 6) | run()
        await self.assertEqual(result, 10)

  async def test_long_chain_with_fn(self):
    for fn, ctx in self.with_fn():
      with ctx:
        c = Chain(fn, 0)
        for _ in range(20):
          c.then(lambda v: v + 1)
        await self.assertEqual(c.run(), 20)


# ═══════════════════════════════════════════════════════════════════════════
# O. Cascade-specific edge paths
# ═══════════════════════════════════════════════════════════════════════════

class CascadeEdgePaths(MyTestCase):
  """Cascade-specific edge cases not covered in section D."""

  async def test_cascade_void_chain_returns_none(self):
    await self.assertIsNone(Cascade().run())

  async def test_cascade_void_with_then_returns_override(self):
    result = Cascade().then(lambda v: v * 2).run(7)
    await self.assertEqual(result, 7)

  async def test_cascade_repr(self):
    r = repr(Cascade(42).then(lambda v: v))
    super(MyTestCase, self).assertIn('Cascade', r)

  async def test_cascade_bool(self):
    super(MyTestCase, self).assertTrue(bool(Cascade()))
    super(MyTestCase, self).assertTrue(bool(Cascade(42)))

  async def test_cascade_do_still_returns_root(self):
    side = []
    result = Cascade(10).do(lambda v: side.append(v * 2)).run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(side, [20])

  async def test_cascade_pipe_syntax(self):
    result = Cascade(5) | (lambda v: v * 10) | run()
    await self.assertEqual(result, 5)

  async def test_cascade_multiple_thens_all_receive_root(self):
    received = []
    (
      Cascade(42)
      .then(lambda v: received.append(('a', v)))
      .then(lambda v: received.append(('b', v)))
      .then(lambda v: received.append(('c', v)))
      .run()
    )
    super(MyTestCase, self).assertEqual(
      received, [('a', 42), ('b', 42), ('c', 42)]
    )


# ═══════════════════════════════════════════════════════════════════════════
# P. Clone advanced — except & finally interaction
# ═══════════════════════════════════════════════════════════════════════════

class CloneAdvancedExceptFinally(MyTestCase):
  """Advanced clone tests involving exception and finally handlers."""

  async def test_clone_preserves_except_handler(self):
    caught = []
    def handler(v):
      caught.append('caught')
    def raiser(v):
      raise ValueError('boom')
    c = Chain(1).then(raiser).except_(handler, reraise=False)
    c2 = c.clone()
    caught.clear()
    c2.run()
    super(MyTestCase, self).assertEqual(caught, ['caught'])

  async def test_clone_preserves_except_reraise(self):
    def raiser(v):
      raise ValueError('boom')
    c = Chain(1).then(raiser).except_(lambda v: None, reraise=True)
    c2 = c.clone()
    with self.assertRaises(ValueError):
      c2.run()

  async def test_clone_preserves_except_exceptions_filter(self):
    caught = []
    def raiser(v):
      raise TypeError('type')
    c = Chain(1).then(raiser).except_(
      lambda v: caught.append('caught'),
      exceptions=[TypeError],
      reraise=False,
    )
    c2 = c.clone()
    caught.clear()
    c2.run()
    super(MyTestCase, self).assertEqual(caught, ['caught'])

  async def test_clone_preserves_both_except_and_finally(self):
    log = []
    def raiser(v):
      raise RuntimeError('boom')
    c = (
      Chain(1)
      .then(raiser)
      .except_(lambda v: log.append('except'), reraise=False)
      .finally_(lambda v: log.append('finally'))
    )
    c2 = c.clone()
    log.clear()
    c2.run()
    super(MyTestCase, self).assertIn('except', log)
    super(MyTestCase, self).assertIn('finally', log)

  async def test_clone_finally_runs_on_success(self):
    log = []
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: log.append('fin'))
    c2 = c.clone()
    log.clear()
    result = c2.run()
    await self.assertEqual(result, 2)
    super(MyTestCase, self).assertEqual(log, ['fin'])


# ═══════════════════════════════════════════════════════════════════════════
# Q. Decorator advanced patterns
# ═══════════════════════════════════════════════════════════════════════════

class DecoratorAdvanced(MyTestCase):
  """Advanced decorator patterns."""

  async def test_chain_decorator_shorthand(self):
    # Chain.decorator() is shorthand for chain.freeze().decorator()
    @Chain().then(lambda v: v + 1).decorator()
    def inc(x):
      return x
    await self.assertEqual(inc(5), 6)

  async def test_stacked_decorators(self):
    @Chain().then(lambda v: v * 2).decorator()
    @Chain().then(lambda v: v + 1).decorator()
    def base(x):
      return x
    # Inner: base(3) -> 3, then +1 -> 4
    # Outer: 4, then *2 -> 8
    await self.assertEqual(base(3), 8)

  async def test_decorator_preserves_docstring(self):
    @Chain().then(lambda v: v).decorator()
    def documented(x):
      """My docstring."""
      return x
    super(MyTestCase, self).assertEqual(documented.__doc__, 'My docstring.')

  async def test_decorator_on_callable_without_name(self):
    class Obj:
      def __call__(self, x):
        return x * 2
    dec = Chain().then(lambda v: v + 100).decorator()
    wrapped = dec(Obj())
    await self.assertEqual(wrapped(5), 110)

  async def test_decorator_on_different_instances(self):
    class Klass:
      def __init__(self, val):
        self.val = val

      @Chain().then(lambda v: v + 1).decorator()
      def inc_val(self):
        return self.val

    a = Klass(10)
    b = Klass(20)
    await self.assertEqual(a.inc_val(), 11)
    await self.assertEqual(b.inc_val(), 21)


# ═══════════════════════════════════════════════════════════════════════════
# R. __call__ and autorun
# ═══════════════════════════════════════════════════════════════════════════

class CallAndAutorun(MyTestCase):
  """Chain.__call__ and autorun behavior."""

  async def test_call_is_run_shorthand(self):
    c = Chain(42).then(lambda v: v + 1)
    await self.assertEqual(c(), 43)

  async def test_call_with_override(self):
    c = Chain().then(lambda v: v * 2)
    await self.assertEqual(c(5), 10)

  async def test_call_override_raises_when_root_exists(self):
    c = Chain(1)
    with self.assertRaises(QuentException):
      c(2)

  async def test_autorun_with_async(self):
    c = Chain(aempty, 42).then(lambda v: v + 1).config(autorun=True)
    result = c.run()
    # autorun wraps coroutine in a Task
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 43)


# ═══════════════════════════════════════════════════════════════════════════
# S. Multiple finally_ raises
# ═══════════════════════════════════════════════════════════════════════════

class MultipleFinallyRaises(MyTestCase):
  """Only one finally_ is allowed per chain."""

  async def test_multiple_finally_raises(self):
    with self.assertRaises(QuentException):
      Chain(1).finally_(lambda v: v).finally_(lambda v: v)


# ═══════════════════════════════════════════════════════════════════════════
# T. Chain with debug mode
# ═══════════════════════════════════════════════════════════════════════════

class DebugModeTests(MyTestCase):
  """Tests that exercise the debug-enabled path."""

  async def test_debug_chain_produces_correct_result(self):
    c = Chain(10).then(lambda v: v + 5).config(debug=True)
    await self.assertEqual(c.run(), 15)

  async def test_debug_cascade_produces_correct_result(self):
    c = Cascade(10).then(lambda v: v * 2).config(debug=True)
    await self.assertEqual(c.run(), 10)

  async def test_debug_with_exception_handler(self):
    caught = []
    def raiser(v):
      raise ValueError('boom')
    c = (
      Chain(1)
      .config(debug=True)
      .then(raiser)
      .except_(lambda v: caught.append('exc'), reraise=False)
    )
    c.run()
    super(MyTestCase, self).assertEqual(caught, ['exc'])


# ═══════════════════════════════════════════════════════════════════════════
# U. to_thread
# ═══════════════════════════════════════════════════════════════════════════

class ToThreadTests(MyTestCase):
  """Basic to_thread tests."""

  async def test_to_thread_basic(self):
    result = await await_(Chain(10).to_thread(lambda v: v + 5).run())
    super(MyTestCase, self).assertEqual(result, 15)

  async def test_to_thread_returns_self(self):
    c = Chain(1)
    result = c.to_thread(lambda v: v)
    super(MyTestCase, self).assertIs(result, c)


# ═══════════════════════════════════════════════════════════════════════════
# V. Except with various exception types
# ═══════════════════════════════════════════════════════════════════════════

class ExceptVariousTypes(MyTestCase):
  """Exception handling edge cases."""

  async def test_except_catches_matching_exception(self):
    caught = []
    def raiser(v):
      raise ValueError('val')
    c = Chain(1).then(raiser).except_(
      lambda v: caught.append('caught'),
      exceptions=[ValueError],
      reraise=False,
    )
    c.run()
    super(MyTestCase, self).assertEqual(caught, ['caught'])

  async def test_except_does_not_catch_non_matching(self):
    def raiser(v):
      raise TypeError('type')
    c = Chain(1).then(raiser).except_(
      lambda v: None,
      exceptions=[ValueError],
      reraise=False,
    )
    with self.assertRaises(TypeError):
      c.run()

  async def test_except_default_catches_exception(self):
    caught = []
    def raiser(v):
      raise RuntimeError('rt')
    c = Chain(1).then(raiser).except_(
      lambda v: caught.append('caught'),
      reraise=False,
    )
    c.run()
    super(MyTestCase, self).assertEqual(caught, ['caught'])

  async def test_except_reraise_true(self):
    def raiser(v):
      raise ValueError('val')
    c = Chain(1).then(raiser).except_(
      lambda v: None,
      reraise=True,
    )
    with self.assertRaises(ValueError):
      c.run()

  async def test_except_with_iterable_exceptions(self):
    caught = []
    def raiser(v):
      raise KeyError('key')
    c = Chain(1).then(raiser).except_(
      lambda v: caught.append('caught'),
      exceptions=[ValueError, KeyError],
      reraise=False,
    )
    c.run()
    super(MyTestCase, self).assertEqual(caught, ['caught'])


# ═══════════════════════════════════════════════════════════════════════════
# W. Return value types through chain
# ═══════════════════════════════════════════════════════════════════════════

class ReturnValueTypesThrough(MyTestCase):
  """Verify diverse Python types propagate correctly through chains."""

  async def test_dict_through_chain(self):
    await self.assertEqual(
      Chain({'a': 1}).then(lambda d: {**d, 'b': 2}).run(),
      {'a': 1, 'b': 2},
    )

  async def test_list_through_chain(self):
    await self.assertEqual(
      Chain([1, 2]).then(lambda lst: lst + [3]).run(),
      [1, 2, 3],
    )

  async def test_tuple_through_chain(self):
    await self.assertEqual(
      Chain((1, 2)).then(lambda t: t + (3,)).run(),
      (1, 2, 3),
    )

  async def test_set_through_chain(self):
    await self.assertEqual(
      Chain({1, 2}).then(lambda s: s | {3}).run(),
      {1, 2, 3},
    )

  async def test_bytes_through_chain(self):
    await self.assertEqual(
      Chain(b'hello').then(lambda b: b + b' world').run(),
      b'hello world',
    )

  async def test_none_through_multiple_links(self):
    await self.assertIsNone(
      Chain(None).then(lambda v: v).then(lambda v: v).run()
    )


# ═══════════════════════════════════════════════════════════════════════════
# X. Chain with sleep
# ═══════════════════════════════════════════════════════════════════════════

class ChainSleepTests(MyTestCase):
  """Sleep link tests."""

  async def test_sleep_returns_self(self):
    c = Chain(1)
    result = c.sleep(0)
    super(MyTestCase, self).assertIs(result, c)

  async def test_sleep_preserves_value(self):
    result = await await_(Chain(42).sleep(0).run())
    super(MyTestCase, self).assertEqual(result, 42)


# ═══════════════════════════════════════════════════════════════════════════
# Y. Frozen chain with Cascade
# ═══════════════════════════════════════════════════════════════════════════

class FrozenCascadeTests(MyTestCase):
  """Freeze + Cascade combinations."""

  async def test_frozen_cascade_returns_root(self):
    frozen = Cascade(42).then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen.run(), 42)

  async def test_frozen_cascade_reusable(self):
    frozen = Cascade().then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen.run(10), 10)
    await self.assertEqual(frozen.run(20), 20)

  async def test_frozen_cascade_call_shorthand(self):
    frozen = Cascade(7).then(lambda v: v * 3).freeze()
    await self.assertEqual(frozen(), 7)
