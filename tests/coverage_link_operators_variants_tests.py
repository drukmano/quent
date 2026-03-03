"""Comprehensive tests targeting uncovered lines in _link.pxi, _operators.pxi, and _variants.pxi.

Coverage gaps addressed:
  _link.pxi:    16, 26-29, 50, 59, 112, 129, 150, 153, 164, 184, 206, 211, 221, 223
  _operators.pxi: 75, 88, 90, 94
  _variants.pxi:  19, 67, 72, 87
"""
import asyncio
from unittest import TestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null


# ============================================================================
# _link.pxi coverage
# ============================================================================


# ---------------------------------------------------------------------------
# Line 16: _await_run async function — exception path
# Lines 26-29: _await_run_fn cdef alias, _determine_eval_code entry
# ---------------------------------------------------------------------------
class AwaitRunExceptionTests(MyTestCase):
  """Trigger _await_run's exception path (line 16-24) and the _await_run_fn alias (line 26)."""

  async def test_async_chain_exception_triggers_await_run(self):
    """An async chain link that raises flows through _await_run which annotates the exception.
    This triggers line 16 (_await_run entry) and its exception handling."""
    async def raise_async(v):
      raise TestExc('async error')

    with self.assertRaises(TestExc):
      await Chain(1).then(raise_async).run()

  async def test_async_chain_exception_with_debug(self):
    """Async exception with debug mode exercises _await_run's chain/link annotation path."""
    async def raise_async(v):
      raise TestExc('debug async error')

    with self.assertRaises(TestExc):
      await Chain(1).config(debug=True).then(raise_async).run()

  async def test_await_run_fn_used_in_except_handler_coro(self):
    """When an except_ handler returns a coroutine with reraise=False,
    _await_run_fn is used to wrap it (line 26, used in _chain_core.pxi line 202)."""
    async def async_handler(v):
      return 'handled'

    result = await Chain(1).then(lambda v: (_ for _ in ()).throw(TestExc('x'))).except_(
      async_handler, reraise=False
    ).run()
    # The except handler's coroutine is passed through _await_run_fn via ensure_future
    # We just need the chain to not raise
    await self.assertEqual(result, 'handled')


# ---------------------------------------------------------------------------
# Line 50: link.kwargs = EMPTY_DICT inside elif args branch (args provided, kwargs is None)
# ---------------------------------------------------------------------------
class DetermineEvalCodeArgsNoKwargsTests(MyTestCase):
  """Line 50: When args are provided but kwargs is None at C level,
  _determine_eval_code sets link.kwargs = EMPTY_DICT."""

  async def test_then_with_positional_args_only(self):
    """Chain().then(fn, 'arg1') — args=('arg1',), kwargs={} at Python level.
    The path through `elif args:` where kwargs may be None hits line 50."""
    for fn, ctx in self.with_fn():
      with ctx:
        def add(a, b):
          return a + b
        await self.assertEqual(
          Chain(fn, 1).then(add, 10, 20).run(), 30
        )

  async def test_then_single_positional_arg(self):
    """Single positional arg triggers EVAL_CALL_WITH_EXPLICIT_ARGS with kwargs=EMPTY_DICT."""
    for fn, ctx in self.with_fn():
      with ctx:
        def identity(a):
          return a
        await self.assertEqual(
          Chain(fn, 1).then(identity, 42).run(), 42
        )


# ---------------------------------------------------------------------------
# Line 59: link.args = EMPTY_TUPLE in else branch (kwargs provided, args is None)
# ---------------------------------------------------------------------------
class DetermineEvalCodeKwargsNoArgsTests(MyTestCase):
  """Line 59: When kwargs are provided but args is None, non-chain link sets
  link.args = EMPTY_TUPLE."""

  async def test_then_with_kwargs_only(self):
    """Chain().then(fn, key='val') — kwargs provided but no positional args.
    For non-chain callables, hits the else branch line 52-60."""
    for fn, ctx in self.with_fn():
      with ctx:
        def kw_fn(*, key=None):
          return key
        await self.assertEqual(
          Chain(fn, 1).then(kw_fn, key='hello').run(), 'hello'
        )

  async def test_then_with_multiple_kwargs(self):
    """Multiple kwargs with no positional args."""
    for fn, ctx in self.with_fn():
      with ctx:
        def multi_kw(*, a=0, b=0):
          return a + b
        await self.assertEqual(
          Chain(fn, 1).then(multi_kw, a=3, b=7).run(), 10
        )


# ---------------------------------------------------------------------------
# Line 112: Pipe unwrap — v = v.function when value is a Pipe instance
# (Only hits if `pipe` library is installed)
# ---------------------------------------------------------------------------
class PipeUnwrapTests(TestCase):
  """Line 112: When a Pipe instance is used as a link value, it is unwrapped
  to v.function. Only runs if pipe is installed."""

  def test_pipe_unwrap_if_available(self):
    """If pipe is installed, Pipe instances are unwrapped to their .function attribute."""
    try:
      from pipe import Pipe
    except ImportError:
      self.skipTest('pipe library not installed')
      return
    double = Pipe(lambda x: x * 2)
    result = Chain(5).then(double).run()
    self.assertEqual(result, 10)


# ---------------------------------------------------------------------------
# Lines 129, 150, 153: _clone_link, _clone_chain_links entry and None check
# ---------------------------------------------------------------------------
class CloneLinkTests(TestCase):
  """Lines 129 (_clone_link entry), 150 (_clone_chain_links entry),
  153 (_clone_chain_links None return)."""

  def test_clone_chain_with_links(self):
    """clone() on a chain with multiple links exercises _clone_chain_links (line 150)
    and _clone_link (line 129) for each link in the chain."""
    c = Chain(1).then(lambda v: v + 1).then(lambda v: v * 2).then(lambda v: v - 1)
    c2 = c.clone()
    self.assertEqual(c2.run(), 3)  # ((1+1)*2) - 1 = 3
    # Verify independence
    c.then(lambda v: v * 100)
    self.assertEqual(c2.run(), 3)

  def test_clone_empty_chain_exercises_none_check(self):
    """clone() on empty chain — _clone_chain_links gets None src, returns None (line 153)."""
    c = Chain()
    c2 = c.clone()
    self.assertIsNone(c2.run())

  def test_clone_chain_root_only(self):
    """clone() with root but no further links — root_link cloned, first_link is None."""
    c = Chain(42)
    c2 = c.clone()
    self.assertEqual(c2.run(), 42)

  def test_clone_chain_preserves_all_link_fields(self):
    """clone() preserves is_exception_handler, exceptions, reraise fields via _clone_link."""
    def handler(v):
      return 'caught'

    c = Chain(1).then(lambda v: (_ for _ in ()).throw(ValueError('x'))).except_(
      handler, exceptions=[ValueError], reraise=False
    )
    c2 = c.clone()
    self.assertEqual(c2.run(), 'caught')

  def test_clone_chain_with_finally(self):
    """clone() clones the on_finally_link via _clone_link (line 129)."""
    log = []
    c = Chain(1).then(lambda v: v + 1).finally_(lambda v: log.append('done'))
    c2 = c.clone()
    log.clear()
    self.assertEqual(c2.run(), 2)
    self.assertEqual(log, ['done'])

  def test_clone_no_root_with_links(self):
    """clone() of chain with no root but with first_link exercises the else branch in clone()."""
    c = Chain().then(lambda: 42)
    c2 = c.clone()
    self.assertEqual(c2.run(), 42)

  def test_clone_no_root_multiple_links(self):
    """clone() of chain with no root and multiple links.
    Exercises _clone_chain_links with multiple nodes starting from first_link (no root_link)."""
    c = Chain().then(lambda: 10).then(lambda v: v * 3)
    c2 = c.clone()
    self.assertEqual(c2.run(), 30)

  def test_clone_no_root_single_link(self):
    """clone() of chain with no root and single link — current_link stays None."""
    c = Chain().then(lambda: 42)
    c2 = c.clone()
    self.assertEqual(c2.run(), 42)


# ---------------------------------------------------------------------------
# Line 164: _make_temp_link entry — triggered by .run(override_value)
# ---------------------------------------------------------------------------
class MakeTempLinkTests(MyTestCase):
  """Line 164: _make_temp_link is called when running a void chain with
  a root value override."""

  async def test_run_with_override_value(self):
    """Chain().then(fn).run(value) creates a temp link via _make_temp_link (line 164)."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain().then(fn).then(lambda v: v * 2).run(5), 10
        )

  async def test_run_with_callable_override(self):
    """run(callable) — _make_temp_link receives a callable."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain().then(fn).then(lambda v: v + 1).run(lambda: 10), 11
        )

  async def test_run_with_override_and_args(self):
    """run(fn, arg1, arg2) — _make_temp_link receives args."""
    for fn, ctx in self.with_fn():
      with ctx:
        def root_fn(a, b):
          return a + b
        await self.assertEqual(
          Chain().then(fn).then(lambda v: v * 2).run(root_fn, 3, 7), 20
        )

  async def test_run_with_override_and_kwargs(self):
    """run(fn, key=val) — _make_temp_link receives kwargs."""
    for fn, ctx in self.with_fn():
      with ctx:
        def root_fn(*, key=0):
          return key
        await self.assertEqual(
          Chain().then(fn).then(lambda v: v + 1).run(root_fn, key=99), 100
        )

  async def test_run_with_literal_override(self):
    """run(literal) — _make_temp_link with a non-callable, allow_literal=True."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain().then(fn).then(lambda v: v * 3).run(7), 21
        )

  async def test_run_override_with_ellipsis(self):
    """run(fn, ...) — _make_temp_link with Ellipsis arg -> EVAL_CALL_WITHOUT_ARGS."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain().then(fn).then(lambda v: v + 1).run(lambda: 50, ...), 51
        )


# ---------------------------------------------------------------------------
# Line 184: evaluate_value entry — central dispatch function
# ---------------------------------------------------------------------------
class EvaluateValueEntryTests(MyTestCase):
  """Line 184: evaluate_value is the central dispatch function.
  Every chain evaluation goes through it."""

  async def test_eval_call_with_current_value(self):
    """EVAL_CALL_WITH_CURRENT_VALUE: basic callable receives current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(lambda v: v * 2).run(), 10
        )

  async def test_eval_call_without_args(self):
    """EVAL_CALL_WITHOUT_ARGS via Ellipsis: callable invoked with zero args."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 99).then(lambda: 7, ...).run(), 7
        )

  async def test_eval_return_as_is(self):
    """EVAL_RETURN_AS_IS: non-callable literal returned directly."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then('hello').run(), 'hello'
        )

  async def test_eval_call_with_explicit_args(self):
    """EVAL_CALL_WITH_EXPLICIT_ARGS: explicit args replace current value."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda a, b: a + b, 3, 4).run(), 7
        )


# ---------------------------------------------------------------------------
# Line 206: Safety path for chain with empty args/None kwargs
# (<Chain>link.v)._run(Null, None, None, True) when args is falsy or kwargs is None
# ---------------------------------------------------------------------------
class NestedChainSafetyPathTests(MyTestCase):
  """Line 206: When a nested chain link has EVAL_CALL_WITH_EXPLICIT_ARGS but
  args is empty/falsy or kwargs is None, the safety path runs the chain with Null."""

  async def test_nested_chain_with_explicit_args(self):
    """Standard nested chain with explicit args — first arg becomes root, rest forwarded."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().then(lambda v: v * 10)
        await self.assertEqual(
          Chain(fn, 5).then(inner, 3).run(), 30
        )

  async def test_nested_chain_call_without_args_ellipsis(self):
    """Nested chain with Ellipsis — EVAL_CALL_WITHOUT_ARGS, runs with Null root."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain(42)
        await self.assertEqual(
          Chain(fn, 5).then(inner, ...).run(), 42
        )


# ---------------------------------------------------------------------------
# Lines 221, 223: evaluate_value non-chain paths
# Line 221: link.v() safety path when args or kwargs is None
# Line 223: link.v(*link.args) when kwargs is EMPTY_DICT
# ---------------------------------------------------------------------------
class EvaluateValueNonChainPathTests(MyTestCase):
  """Lines 221 and 223: Non-chain EVAL_CALL_WITH_EXPLICIT_ARGS paths
  where args/kwargs have specific states."""

  async def test_explicit_args_with_empty_dict_kwargs(self):
    """Line 223: link.v(*link.args) when kwargs is EMPTY_DICT.
    This is the standard path when positional args are given without kwargs."""
    for fn, ctx in self.with_fn():
      with ctx:
        def add3(a, b, c):
          return a + b + c
        await self.assertEqual(
          Chain(fn, 1).then(add3, 10, 20, 30).run(), 60
        )

  async def test_explicit_args_with_real_kwargs(self):
    """Standard path with both args and kwargs."""
    for fn, ctx in self.with_fn():
      with ctx:
        def compute(a, b, factor=1):
          return (a + b) * factor
        await self.assertEqual(
          Chain(fn, 1).then(compute, 5, 3, factor=2).run(), 16
        )

  async def test_kwargs_only_non_chain(self):
    """Line 59 + 223 combined: kwargs-only call on a non-chain callable.
    _determine_eval_code sets args=EMPTY_TUPLE, kwargs stays non-empty.
    evaluate_value calls link.v(*link.args, **link.kwargs) = link.v(**link.kwargs)."""
    for fn, ctx in self.with_fn():
      with ctx:
        def kw_only(*, x=0, y=0):
          return x * y
        await self.assertEqual(
          Chain(fn, 1).then(kw_only, x=5, y=6).run(), 30
        )


# ============================================================================
# _operators.pxi coverage
# ============================================================================


# ---------------------------------------------------------------------------
# Line 75: _eval_signal_value entry
# Lines 88, 90: kwargs=EMPTY_DICT when None, v(*args) when kwargs is EMPTY_DICT
# Line 94: v(*args, **kwargs) when kwargs is non-empty
# ---------------------------------------------------------------------------
class EvalSignalValueTests(MyTestCase):
  """Lines 75, 88, 90, 94: _eval_signal_value handles control flow signal values."""

  async def test_return_with_positional_args_no_kwargs(self):
    """Line 88: kwargs is None -> set to EMPTY_DICT.
    Line 90: return v(*args) when kwargs is EMPTY_DICT.
    Chain.return_(fn, arg1, arg2) raises _Return(fn, (arg1, arg2), {}).
    _eval_signal_value gets args=(arg1, arg2), kwargs={} at Python level."""
    for fn, ctx in self.with_fn():
      with ctx:
        def add(a, b):
          return a + b
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(add, 10, 20)).run(), 30
        )

  async def test_return_with_args_and_kwargs(self):
    """Line 94: v(*args, **kwargs) when both args and kwargs are provided.
    But this goes through the `if args:` branch, not `elif kwargs:`."""
    for fn, ctx in self.with_fn():
      with ctx:
        def compute(a, factor=1):
          return a * factor
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(compute, 5, factor=3)).run(), 15
        )

  async def test_break_with_callable_and_args(self):
    """break_(fn, arg) — _eval_signal_value called from handle_break_exc with args."""
    for fn, ctx in self.with_fn():
      with ctx:
        def double(x):
          return x * 2
        def f(el):
          if el >= 3:
            return Chain.break_(double, 100)
          return el
        await self.assertEqual(
          Chain(fn, [1, 2, 3, 4]).foreach(f).run(), 200
        )

  async def test_return_callable_no_args(self):
    """_eval_signal_value with callable v, no args, no kwargs → calls v()."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(lambda: 42)).run(), 42
        )

  async def test_return_literal_no_args(self):
    """_eval_signal_value with non-callable v, no args, no kwargs → returns v as-is."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(99)).run(), 99
        )

  async def test_return_with_ellipsis_call_without_args(self):
    """_eval_signal_value with args=(Ellipsis,) → EVAL_CALL_WITHOUT_ARGS equivalent."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(lambda: 55, ...)).run(), 55
        )

  async def test_break_with_kwargs_only(self):
    """Line 94 via elif kwargs: branch — break_(fn, key=val).
    _eval_signal_value gets v=fn, args=(), kwargs={'key': val}.
    Since args is falsy, goes to `elif kwargs:` branch (line 92-95).
    But Python sends args=() not None, so args is not None → doesn't hit line 93-94.
    We need to trigger it via the internal mechanism."""
    # The `elif kwargs:` branch at line 92 is entered when `not args` (args is empty/falsy)
    # and kwargs is truthy. When entered, if args is None, sets args=EMPTY_TUPLE.
    # Through the public API, Chain.return_/break_ always passes a tuple for args,
    # so args is () not None. Line 93 (`if args is None`) requires args to truly be None.
    # This can happen at the C level when _eval_signal_value is called from handle_break_exc
    # or handle_return_exc with an _InternalQuentException whose args_ was set to None.
    # Through public API, this is not directly triggerable since break_/return_ always
    # create _InternalQuentException with tuple args.
    # We test the reachable path: kwargs truthy, args falsy but not None.
    def f(el):
      if el >= 2:
        Chain.break_(lambda *, key=None: key, key='stopped')
      return el

    await self.assertEqual(
      Chain([1, 2, 3]).foreach(f).run(), 'stopped'
    )

  async def test_return_args_and_kwargs_combined(self):
    """Line 91: v(*args, **kwargs) in the args-truthy branch when kwargs is also truthy."""
    for fn, ctx in self.with_fn():
      with ctx:
        def combined(a, b, *, extra=0):
          return a + b + extra
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(combined, 10, 20, extra=5)).run(), 35
        )


# ============================================================================
# _variants.pxi coverage
# ============================================================================


# ---------------------------------------------------------------------------
# Line 19: Cascade.__init__ calls self.init with is_cascade=True
# ---------------------------------------------------------------------------
class CascadeInitTests(MyTestCase):
  """Line 19: Cascade.__init__ calls self.init(__v, args, kwargs, True)."""

  async def test_cascade_with_literal_root(self):
    """Cascade(5) — root value is literal 5, is_cascade=True."""
    log = []
    c = Cascade(5)
    c.do(lambda v: log.append(v))
    c.do(lambda v: log.append(v * 2))
    result = c.run()
    await self.assertEqual(result, 5)
    super(MyTestCase, self).assertEqual(log, [5, 10])

  async def test_cascade_with_callable_root(self):
    """Cascade(lambda: 10) — callable root."""
    log = []
    c = Cascade(lambda: 10)
    c.do(lambda v: log.append(v))
    result = c.run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(log, [10])

  async def test_cascade_with_root_args(self):
    """Cascade(fn, arg1, arg2) — callable root with explicit args."""
    def root_fn(a, b):
      return a + b
    c = Cascade(root_fn, 3, 7)
    log = []
    c.do(lambda v: log.append(v))
    result = c.run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(log, [10])

  async def test_cascade_with_root_kwargs(self):
    """Cascade(fn, key=val) — callable root with kwargs."""
    def root_fn(*, key=0):
      return key
    c = Cascade(root_fn, key=42)
    log = []
    c.do(lambda v: log.append(v))
    result = c.run()
    await self.assertEqual(result, 42)
    super(MyTestCase, self).assertEqual(log, [42])

  async def test_cascade_no_root(self):
    """Cascade() — void cascade, no root value."""
    c = Cascade()
    c.then(lambda: 99)
    result = c.run()
    # Cascade returns root_value which is Null -> None
    await self.assertIsNone(result)

  async def test_cascade_operations_receive_root_value(self):
    """All operations in Cascade receive the root value, not the previous result."""
    results = []
    c = Cascade(10)
    c.then(lambda v: results.append(v) or v * 2)
    c.then(lambda v: results.append(v) or v * 3)
    c.run()
    super(MyTestCase, self).assertEqual(results, [10, 10])

  async def test_cascade_async_root(self):
    """Cascade with async root callable."""
    async def async_root():
      return 42
    c = Cascade(async_root)
    log = []
    c.do(lambda v: log.append(v))
    result = await c.run()
    await self.assertEqual(result, 42)
    super(MyTestCase, self).assertEqual(log, [42])


# ---------------------------------------------------------------------------
# Line 67: _FrozenChain.run method
# ---------------------------------------------------------------------------
class FrozenChainRunTests(MyTestCase):
  """Line 67: _FrozenChain.run() executes the frozen chain."""

  async def test_frozen_chain_run_no_args(self):
    """frozen.run() — no override, uses chain's root value."""
    frozen = Chain(10).then(lambda v: v + 5).freeze()
    await self.assertEqual(frozen.run(), 15)

  async def test_frozen_chain_run_with_override(self):
    """frozen.run(5) — override root value."""
    frozen = Chain().then(lambda v: v * 3).freeze()
    await self.assertEqual(frozen.run(5), 15)

  async def test_frozen_chain_run_with_args(self):
    """frozen.run(fn, arg1) — callable override with positional args."""
    def root_fn(a, b):
      return a + b
    frozen = Chain().then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen.run(root_fn, 3, 7), 20)

  async def test_frozen_chain_run_with_kwargs(self):
    """frozen.run(fn, key=val) — callable override with kwargs."""
    def root_fn(*, key=0):
      return key
    frozen = Chain().then(lambda v: v + 1).freeze()
    await self.assertEqual(frozen.run(root_fn, key=99), 100)

  async def test_frozen_chain_run_repeated(self):
    """frozen.run() can be called multiple times with different overrides."""
    frozen = Chain().then(lambda v: v ** 2).freeze()
    await self.assertEqual(frozen.run(3), 9)
    await self.assertEqual(frozen.run(4), 16)
    await self.assertEqual(frozen.run(5), 25)

  async def test_frozen_chain_run_async(self):
    """frozen.run() with async chain."""
    async def async_double(v):
      return v * 2
    frozen = Chain().then(async_double).freeze()
    await self.assertEqual(frozen.run(5), 10)


# ---------------------------------------------------------------------------
# Line 72: _FrozenChain.__call__ method
# ---------------------------------------------------------------------------
class FrozenChainCallTests(MyTestCase):
  """Line 72: _FrozenChain.__call__() is shorthand for run()."""

  async def test_frozen_chain_call_no_args(self):
    """frozen() — no override."""
    frozen = Chain(10).then(lambda v: v + 5).freeze()
    await self.assertEqual(frozen(), 15)

  async def test_frozen_chain_call_with_override(self):
    """frozen(5) — override root value."""
    frozen = Chain().then(lambda v: v * 3).freeze()
    await self.assertEqual(frozen(5), 15)

  async def test_frozen_chain_call_with_args(self):
    """frozen(fn, arg1, arg2) — callable override with positional args."""
    def root_fn(a, b):
      return a + b
    frozen = Chain().then(lambda v: v * 2).freeze()
    await self.assertEqual(frozen(root_fn, 3, 7), 20)

  async def test_frozen_chain_call_with_kwargs(self):
    """frozen(fn, key=val) — callable override with kwargs."""
    def root_fn(*, key=0):
      return key
    frozen = Chain().then(lambda v: v + 1).freeze()
    await self.assertEqual(frozen(root_fn, key=99), 100)

  async def test_frozen_chain_call_equivalence_with_run(self):
    """frozen() and frozen.run() produce the same result."""
    frozen = Chain(10).then(lambda v: v * 2).freeze()
    run_result = frozen.run()
    call_result = frozen()
    # Both should be synchronous here since no async involved
    super(MyTestCase, self).assertEqual(run_result, call_result)
    super(MyTestCase, self).assertEqual(run_result, 20)

  async def test_frozen_chain_call_async(self):
    """frozen() with async chain."""
    async def async_add_ten(v):
      return v + 10
    frozen = Chain().then(async_add_ten).freeze()
    await self.assertEqual(frozen(5), 15)


# ---------------------------------------------------------------------------
# Line 87: run.__init__ stores root_value, args, kwargs
# ---------------------------------------------------------------------------
class RunInitTests(TestCase):
  """Line 87: run.__init__ stores root_value, args, kwargs."""

  def test_run_init_with_all_params(self):
    """run(5, 'arg1', key='val') stores all three attributes."""
    r = run(5, 'arg1', 'arg2', key='val')
    self.assertEqual(r.root_value, 5)
    self.assertEqual(r.args, ('arg1', 'arg2'))
    self.assertEqual(r.kwargs, {'key': 'val'})

  def test_run_init_no_args(self):
    """run() stores Null as root_value and empty args/kwargs."""
    r = run()
    self.assertIs(r.root_value, Null)
    self.assertEqual(r.args, ())
    self.assertEqual(r.kwargs, {})

  def test_run_init_root_only(self):
    """run(42) stores root_value only."""
    r = run(42)
    self.assertEqual(r.root_value, 42)
    self.assertEqual(r.args, ())
    self.assertEqual(r.kwargs, {})

  def test_run_init_with_kwargs_only(self):
    """run(fn, key=val) — root + kwargs, no extra positional args."""
    def fn():
      pass
    r = run(fn, key='hello')
    self.assertIs(r.root_value, fn)
    self.assertEqual(r.args, ())
    self.assertEqual(r.kwargs, {'key': 'hello'})

  def test_run_init_callable_root(self):
    """run(lambda: 42) stores the callable as root_value."""
    fn = lambda: 42
    r = run(fn)
    self.assertIs(r.root_value, fn)

  def test_run_pipe_with_args(self):
    """run(fn, arg1, arg2) used in pipe executes correctly."""
    def root_fn(a, b):
      return a + b
    result = Chain().then(lambda v: v * 2) | run(root_fn, 3, 7)
    self.assertEqual(result, 20)

  def test_run_pipe_with_kwargs(self):
    """run(fn, key=val) used in pipe executes correctly."""
    def root_fn(*, key=0):
      return key
    result = Chain().then(lambda v: v + 1) | run(root_fn, key=99)
    self.assertEqual(result, 100)


# ============================================================================
# Additional edge-case tests for deeper coverage
# ============================================================================


class NestedChainEvalCodeEdgeTests(MyTestCase):
  """Additional nested chain evaluation paths for thorough coverage."""

  async def test_nested_chain_eval_call_without_args(self):
    """Nested chain with EVAL_CALL_WITHOUT_ARGS — runs inner chain with Null root."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain(99)
        await self.assertEqual(
          Chain(fn, 5).then(inner, ...).run(), 99
        )

  async def test_nested_chain_eval_call_with_current_value(self):
    """Nested chain with EVAL_CALL_WITH_CURRENT_VALUE — inner chain receives outer cv."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().then(lambda v: v * 10)
        await self.assertEqual(
          Chain(fn, 3).then(inner).run(), 30
        )

  async def test_nested_chain_multiple_explicit_args(self):
    """Nested chain with multiple explicit args — first arg becomes the inner chain's root.
    The inner chain._run receives (args[0], args[1:], kwargs, True) from evaluate_value line 207."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain().then(lambda v: v)
        await self.assertEqual(
          Chain(fn, 1).then(inner, 42).run(), 42
        )

  async def test_deeply_nested_chains(self):
    """Multiple levels of nesting."""
    for fn, ctx in self.with_fn():
      with ctx:
        c3 = Chain().then(lambda v: v * 100)
        c2 = Chain().then(c3)
        c1 = Chain(fn, 2).then(c2)
        await self.assertEqual(c1.run(), 200)


class CloneAdvancedTests(TestCase):
  """Additional clone tests to ensure _clone_link and _clone_chain_links are exercised."""

  def test_clone_chain_with_do_operations(self):
    """clone() with ignore_result links (do operations)."""
    log = []
    c = Chain(10).do(lambda v: log.append(v)).then(lambda v: v + 5)
    c2 = c.clone()
    log.clear()
    self.assertEqual(c2.run(), 15)
    self.assertEqual(log, [10])

  def test_clone_cascade(self):
    """Clone of a Cascade preserves cascade behavior."""
    log = []
    c = Cascade(10).do(lambda v: log.append(v)).do(lambda v: log.append(v * 2))
    c2 = c.clone()
    log.clear()
    result = c2.run()
    self.assertEqual(result, 10)
    self.assertEqual(log, [10, 20])

  def test_clone_with_nested_chain(self):
    """Clone with a nested chain link."""
    inner = Chain().then(lambda v: v * 10)
    outer = Chain(2).then(inner)
    c2 = outer.clone()
    self.assertEqual(c2.run(), 20)

  def test_clone_chain_long(self):
    """Clone a chain with many links to stress _clone_chain_links loop."""
    c = Chain(1)
    for i in range(20):
      c.then(lambda v, inc=i: v + inc)
    c2 = c.clone()
    self.assertEqual(c.run(), c2.run())


class AsyncExceptionAnnotationTests(MyTestCase):
  """Test _await_run's exception annotation path with chain and link context."""

  async def test_async_exception_in_run_async(self):
    """Exception in async chain triggers _await_run annotation path."""
    async def async_fail(v):
      raise ValueError('async failure')

    # Use .do() to force non-simple path and .config(debug=True) for full annotation
    with self.assertRaises(ValueError):
      await Chain(1).config(debug=True).do(lambda v: None).then(async_fail).run()

  async def test_async_exception_in_simple_chain(self):
    """Exception in simple async chain (no debug, no do)."""
    async def async_fail(v):
      raise TestExc('simple async failure')

    with self.assertRaises(TestExc):
      await Chain(1).then(async_fail).run()

  async def test_async_exception_with_except_handler(self):
    """Async exception caught by except_ handler."""
    async def async_fail(v):
      raise TestExc('caught async')

    result = await Chain(1).then(async_fail).except_(
      lambda v: 'recovered', reraise=False
    ).run()
    await self.assertEqual(result, 'recovered')


class EvalCodeDeterminePathTests(MyTestCase):
  """Test all paths through _determine_eval_code."""

  async def test_callable_no_args_no_kwargs(self):
    """Callable with no args/kwargs → EVAL_CALL_WITH_CURRENT_VALUE."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(lambda v: v + 1).run(), 6
        )

  async def test_non_callable_no_args_allows_literal(self):
    """Non-callable with allow_literal=True → EVAL_RETURN_AS_IS."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(42).run(), 42
        )

  async def test_non_callable_no_args_disallows_literal(self):
    """Non-callable with allow_literal=False → QuentException."""
    with self.assertRaises(QuentException):
      Chain(1).do(42)

  async def test_args_with_ellipsis(self):
    """args=(Ellipsis,) → EVAL_CALL_WITHOUT_ARGS."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertEqual(
          Chain(fn, 5).then(lambda: 99, ...).run(), 99
        )

  async def test_args_without_ellipsis(self):
    """args=(val,) without Ellipsis → EVAL_CALL_WITH_EXPLICIT_ARGS, kwargs=EMPTY_DICT."""
    for fn, ctx in self.with_fn():
      with ctx:
        def take_two(a, b):
          return a + b
        await self.assertEqual(
          Chain(fn, 5).then(take_two, 10, 20).run(), 30
        )

  async def test_chain_kwargs_only(self):
    """Chain as value with kwargs only → EVAL_CALL_WITHOUT_ARGS (line 56)."""
    for fn, ctx in self.with_fn():
      with ctx:
        inner = Chain(42)
        # kwargs-only for a chain triggers line 53-56
        await self.assertEqual(
          Chain(fn, 5).then(inner, key='ignored').run(), 42
        )

  async def test_non_chain_kwargs_only(self):
    """Non-chain with kwargs only → args=EMPTY_TUPLE, EVAL_CALL_WITH_EXPLICIT_ARGS."""
    for fn, ctx in self.with_fn():
      with ctx:
        def kw_fn(*, x=0):
          return x
        await self.assertEqual(
          Chain(fn, 1).then(kw_fn, x=10).run(), 10
        )


class RunOverrideWithSimpleChainTests(MyTestCase):
  """Test _make_temp_link via .run() override on simple chains (no debug/do/finally)."""

  async def test_simple_chain_run_override(self):
    """Simple chain .run(override) triggers _make_temp_link in _run_simple."""
    for fn, ctx in self.with_fn():
      with ctx:
        c = Chain().then(fn).then(lambda v: v * 2)
        await self.assertEqual(c.run(5), 10)

  async def test_simple_chain_run_override_callable(self):
    """Simple chain .run(callable) — callable evaluated as root."""
    for fn, ctx in self.with_fn():
      with ctx:
        c = Chain().then(fn).then(lambda v: v + 1)
        await self.assertEqual(c.run(lambda: 9), 10)


class FrozenChainIntegrationTests(MyTestCase):
  """Integration tests combining freeze() with various chain features."""

  async def test_frozen_chain_with_cascade(self):
    """Cascade chain can be frozen and executed."""
    log = []
    frozen = Cascade(10).do(lambda v: log.append(v)).freeze()
    log.clear()
    result = frozen.run()
    await self.assertEqual(result, 10)
    super(MyTestCase, self).assertEqual(log, [10])

  async def test_frozen_chain_decorator_with_class(self):
    """Frozen chain decorator works as a class method decorator."""
    @Chain().then(lambda v: v * 2).freeze().decorator()
    def compute(x):
      return x + 1
    await self.assertEqual(compute(4), 10)  # (4+1)*2

  async def test_frozen_chain_run_after_call(self):
    """Can call both run() and __call__() on the same frozen chain."""
    frozen = Chain(10).then(lambda v: v + 5).freeze()
    await self.assertEqual(frozen.run(), 15)
    await self.assertEqual(frozen(), 15)


class BreakSignalEvalTests(MyTestCase):
  """Additional break_ signal tests targeting _eval_signal_value paths."""

  async def test_break_with_literal_value(self):
    """break_('literal') — _eval_signal_value with non-callable returns as-is."""
    def f(x):
      if x >= 3:
        return Chain.break_('stopped')
      return x

    await self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(f).run(), 'stopped'
    )

  async def test_break_with_callable_no_args(self):
    """break_(callable) — _eval_signal_value calls v() since callable, no args."""
    def f(x):
      if x >= 3:
        return Chain.break_(lambda: 'done')
      return x

    await self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(f).run(), 'done'
    )

  async def test_break_with_callable_and_ellipsis(self):
    """break_(callable, ...) — EVAL_CALL_WITHOUT_ARGS equivalent."""
    def f(x):
      if x >= 3:
        return Chain.break_(lambda: 'ellipsis_done', ...)
      return x

    await self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(f).run(), 'ellipsis_done'
    )

  async def test_break_with_callable_and_explicit_args(self):
    """break_(fn, arg1, arg2) — fn called with explicit args."""
    def f(x):
      if x >= 3:
        return Chain.break_(lambda a, b: a + b, 10, 20)
      return x

    await self.assertEqual(
      Chain([1, 2, 3, 4]).foreach(f).run(), 30
    )


class ReturnSignalEvalTests(MyTestCase):
  """Return signal tests targeting _eval_signal_value through handle_return_exc."""

  async def test_return_with_no_value(self):
    """return_() with no value returns None."""
    for fn, ctx in self.with_fn():
      with ctx:
        await self.assertIsNone(
          Chain(fn, 1).then(lambda v: Chain.return_()).run()
        )

  async def test_return_with_callable_args_kwargs(self):
    """return_(fn, arg, key=val) — _eval_signal_value with args and kwargs."""
    for fn, ctx in self.with_fn():
      with ctx:
        def combined(a, *, extra=0):
          return a + extra
        await self.assertEqual(
          Chain(fn, 1).then(lambda v: Chain.return_(combined, 10, extra=5)).run(), 15
        )


if __name__ == '__main__':
  import unittest
  unittest.main()
