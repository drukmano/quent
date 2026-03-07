"""Negative tests: invalid usage, wrong types, double registration, control flow in handlers."""
from __future__ import annotations

import asyncio
import unittest
from contextlib import contextmanager
from unittest import IsolatedAsyncioTestCase

from quent import Chain, Null, QuentException
from quent._chain import _FrozenChain
from quent._core import _Return, _Break, _ControlFlowSignal
from helpers import SyncCM, TrackingCM


# ---------------------------------------------------------------------------
# TestDoubleRegistration
# ---------------------------------------------------------------------------

class TestDoubleRegistration(unittest.TestCase):
  """Registering except_ or finally_ twice raises QuentException."""

  def test_double_except_raises(self):
    with self.assertRaises(QuentException):
      Chain(5).except_(lambda e: e).except_(lambda e: e)

  def test_double_finally_raises(self):
    with self.assertRaises(QuentException):
      Chain(5).finally_(lambda x: x).finally_(lambda x: x)

  def test_except_message(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).except_(lambda e: e).except_(lambda e: e)
    self.assertIn('one', str(ctx.exception).lower())

  def test_finally_message(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).finally_(lambda x: x).finally_(lambda x: x)
    self.assertIn('one', str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# TestInvalidTypes
# ---------------------------------------------------------------------------

class TestInvalidTypes(unittest.TestCase):
  """Passing invalid types to iteration/CM operations."""

  def test_map_on_non_iterable(self):
    non_iterables = [42, 3.14, True, None, object()]
    for val in non_iterables:
      with self.subTest(val=val):
        with self.assertRaises(TypeError):
          Chain(val).map(lambda x: x).run()

  def test_filter_on_non_iterable(self):
    non_iterables = [42, 3.14, True, None, object()]
    for val in non_iterables:
      with self.subTest(val=val):
        with self.assertRaises(TypeError):
          Chain(val).filter(lambda x: True).run()

  def test_with_on_non_cm(self):
    non_cms = [42, 'hello', None, [1, 2, 3]]
    for val in non_cms:
      with self.subTest(val=val):
        with self.assertRaises((AttributeError, TypeError)):
          Chain(val).with_(lambda ctx: ctx).run()

  def test_gather_with_non_callable(self):
    with self.assertRaises(TypeError):
      Chain(5).gather(42).run()

  def test_iterate_on_non_iterable(self):
    with self.assertRaises(TypeError):
      list(Chain(42).iterate())


# ---------------------------------------------------------------------------
# TestCallableEdgeCases
# ---------------------------------------------------------------------------

class TestCallableEdgeCases(unittest.TestCase):
  """Objects that appear callable but fail when called."""

  def test_object_with_call_none(self):
    """callable() returns True but calling raises TypeError."""

    class WeirdCallable:
      __call__ = None

    obj = WeirdCallable()
    self.assertTrue(callable(obj))
    with self.assertRaises(TypeError):
      Chain(5).then(obj).run()

  def test_object_with_call_not_implemented(self):
    """__call__ raises NotImplementedError."""

    class NotImplCallable:
      def __call__(self, *args, **kwargs):
        raise NotImplementedError('not implemented')

    obj = NotImplCallable()
    with self.assertRaises(NotImplementedError):
      Chain(5).then(obj).run()


# ---------------------------------------------------------------------------
# TestControlFlowInHandlers
# ---------------------------------------------------------------------------

class TestControlFlowInHandlers(unittest.TestCase):
  """Control flow signals in except/finally handlers are forbidden."""

  def test_return_in_except(self):
    c = Chain(lambda: 1 / 0).except_(lambda exc: Chain.return_(99))
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('except', str(ctx.exception).lower())

  def test_break_in_except(self):
    c = Chain(lambda: 1 / 0).except_(lambda exc: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('except', str(ctx.exception).lower())

  def test_return_in_finally(self):
    c = Chain(5).finally_(lambda x: Chain.return_(99))
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('finally', str(ctx.exception).lower())

  def test_break_in_finally(self):
    c = Chain(5).finally_(lambda x: Chain.break_())
    with self.assertRaises(QuentException) as ctx:
      c.run()
    self.assertIn('finally', str(ctx.exception).lower())

  def test_break_outside_map(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: Chain.break_()).run()
    self.assertIn('map', str(ctx.exception).lower())

  def test_break_in_then(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).then(lambda x: Chain.break_()).run()
    self.assertIn('cannot be used outside', str(ctx.exception).lower())

  def test_break_in_do(self):
    with self.assertRaises(QuentException) as ctx:
      Chain(5).do(lambda x: Chain.break_()).run()
    self.assertIn('cannot be used outside', str(ctx.exception).lower())

  def test_break_in_gather_fn(self):
    """break_() inside a gather fn: _ControlFlowSignal is not caught by gather,
    it propagates to _run where _Break is caught and raises QuentException."""
    with self.assertRaises(QuentException):
      Chain(5).gather(lambda x: Chain.break_()).run()

  def test_break_in_with_body(self):
    """break_() inside with_ body: _ControlFlowSignal propagates through with_,
    then to _run where _Break raises QuentException (not in map context)."""
    with self.assertRaises(QuentException):
      Chain(SyncCM()).with_(lambda ctx: Chain.break_()).run()


# ---------------------------------------------------------------------------
# TestEmptyChainEdgeCases
# ---------------------------------------------------------------------------

class TestEmptyChainEdgeCases(unittest.TestCase):
  """Edge cases with empty or minimal chains."""

  def test_empty_chain_run(self):
    result = Chain().run()
    self.assertIsNone(result)

  def test_empty_chain_bool(self):
    self.assertTrue(bool(Chain()))

  def test_chain_only_except(self):
    handler_called = []
    result = Chain().except_(lambda exc: handler_called.append(exc)).run()
    # No error raised, handler not called, result is None
    self.assertIsNone(result)
    self.assertEqual(handler_called, [])

  def test_chain_only_finally(self):
    handler_called = []
    # When no root value exists, finally_ is called with no args (root_value is Null,
    # so _evaluate_value calls fn() with zero args).
    result = Chain().finally_(lambda: handler_called.append('called')).run()
    self.assertIsNone(result)
    self.assertEqual(handler_called, ['called'])

  def test_chain_only_do(self):
    tracker = []
    # Chain().do(fn) -- no root value, fn is called with Null (no current_value)
    # do(fn): fn is callable, current_value is Null, so fn() called with no args
    result = Chain().do(lambda: tracker.append('called')).run()
    self.assertIsNone(result)
    self.assertEqual(tracker, ['called'])


# ---------------------------------------------------------------------------
# BEYOND SPEC: Additional negative and edge-case tests
# ---------------------------------------------------------------------------

class TestReturnOutsideChainContext(unittest.TestCase):
  """Chain.return_() outside any chain context raises _Return."""

  def test_return_raises_return_signal(self):
    """Chain.return_() raises _Return which is a _ControlFlowSignal."""
    with self.assertRaises(_Return):
      Chain.return_(42)

  def test_return_is_control_flow_signal(self):
    try:
      Chain.return_(42)
    except _ControlFlowSignal as exc:
      self.assertIsInstance(exc, _Return)
      self.assertEqual(exc.value, 42)


class TestBreakOutsideChainContext(unittest.TestCase):
  """Chain.break_() outside any chain context raises _Break."""

  def test_break_raises_break_signal(self):
    with self.assertRaises(_Break):
      Chain.break_()

  def test_break_is_control_flow_signal(self):
    try:
      Chain.break_(99)
    except _ControlFlowSignal as exc:
      self.assertIsInstance(exc, _Break)
      self.assertEqual(exc.value, 99)


class TestPassingNonCallableToDo(unittest.TestCase):
  """Passing non-callable to do(): raises TypeError eagerly."""

  def test_do_rejects_int(self):
    with self.assertRaises(TypeError) as cm:
      Chain(5).do(42)
    self.assertEqual(str(cm.exception), 'do() requires a callable, got int')

  def test_do_rejects_string(self):
    with self.assertRaises(TypeError) as cm:
      Chain(5).do('hello')
    self.assertEqual(str(cm.exception), 'do() requires a callable, got str')

  def test_do_rejects_list(self):
    with self.assertRaises(TypeError) as cm:
      Chain(5).do([1, 2, 3])
    self.assertEqual(str(cm.exception), 'do() requires a callable, got list')

  def test_do_rejects_none(self):
    with self.assertRaises(TypeError) as cm:
      Chain(5).do(None)
    self.assertEqual(str(cm.exception), 'do() requires a callable, got NoneType')


class TestMapMutatingList(unittest.TestCase):
  """map where fn modifies the list being iterated."""

  def test_map_append_during_iteration(self):
    """Modifying the list during iteration: behavior depends on iterator.
    For list iterators in Python, appending during iteration is well-defined
    but may cause infinite loops. We test with a bounded mutation."""
    data = [1, 2, 3]
    counter = [0]

    def fn(x):
      counter[0] += 1
      if counter[0] <= 3 and len(data) < 6:
        data.append(x + 10)
      return x

    result = Chain(data).map(fn).run()
    # The original list was mutated, and the iterator picked up the new items
    self.assertTrue(len(result) >= 3)


class TestFilterWithSideEffects(unittest.TestCase):
  """filter where predicate has side effects."""

  def test_filter_predicate_side_effects(self):
    tracker = []

    def predicate(x):
      tracker.append(x)
      return x > 2

    result = Chain([1, 2, 3, 4]).filter(predicate).run()
    self.assertEqual(result, [3, 4])
    self.assertEqual(tracker, [1, 2, 3, 4])


class TestWithWhereEnterReturnsChain(unittest.TestCase):
  """with_ where CM.__enter__ returns a Chain object."""

  def test_enter_returns_chain_object(self):
    """When __enter__ returns a Chain, the body fn receives it as ctx.
    The fn can use it however it wants."""

    class ChainCM:
      def __enter__(self):
        return Chain(42)
      def __exit__(self, *args):
        return False

    result = Chain(ChainCM()).with_(lambda ctx: ctx.run()).run()
    self.assertEqual(result, 42)


class TestGatherAllRaise(unittest.TestCase):
  """gather where ALL fns raise: which exception propagates?"""

  def test_gather_first_raise_propagates(self):
    """When a fn raises during setup in gather, the first exception propagates.
    gather evaluates fns sequentially, so the first raiser stops evaluation."""

    def fn1(x):
      raise ValueError('fn1 error')

    def fn2(x):
      raise TypeError('fn2 error')

    with self.assertRaises(ValueError) as ctx:
      Chain(5).gather(fn1, fn2).run()
    self.assertIn('fn1', str(ctx.exception))


class TestGatherSecondRaises(unittest.TestCase):
  """gather where first fn succeeds, second raises."""

  def test_second_fn_raises(self):
    def fn1(x):
      return x * 2

    def fn2(x):
      raise RuntimeError('fn2 failed')

    with self.assertRaises(RuntimeError):
      Chain(5).gather(fn1, fn2).run()


class TestVeryDeeplyNestedChains(unittest.TestCase):
  """Very deeply nested chains (20+ levels)."""

  def test_20_plus_levels(self):
    c = Chain().then(lambda x: x)
    for _ in range(24):
      c = Chain().then(c)
    result = c.run(42)
    self.assertEqual(result, 42)


class TestChainReuse(unittest.TestCase):
  """Running the same chain 100 times."""

  def test_reuse_consistency(self):
    c = Chain().then(lambda x: x * 3)
    results = [c.run(i) for i in range(100)]
    self.assertEqual(results, [i * 3 for i in range(100)])


class TestFrozenChainFromEmptyChain(unittest.TestCase):
  """Frozen chain from empty chain."""

  def test_frozen_empty_returns_none(self):
    frozen = Chain().freeze()
    result = frozen.run()
    self.assertIsNone(result)

  def test_frozen_empty_with_value(self):
    frozen = Chain().freeze()
    result = frozen.run(42)
    self.assertEqual(result, 42)

  def test_frozen_empty_bool(self):
    self.assertTrue(bool(Chain().freeze()))

  def test_frozen_empty_repr(self):
    frozen = Chain().freeze()
    self.assertIn('Frozen', repr(frozen))


class TestDecoratorOnDecorated(unittest.TestCase):
  """decorator() on an already-decorated function."""

  def test_decorator_preserves_wrapping(self):
    import functools

    def logging_decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
      return wrapper

    chain = Chain().then(lambda x: x + 10)

    @chain.decorator()
    @logging_decorator
    def my_fn(x):
      return x * 2

    result = my_fn(5)
    # decorator: chain._run(my_fn, (5,), {})
    # root = Link(my_fn, (5,), {}), my_fn(5) -> logging_decorator wrapper -> 5 * 2 = 10
    # then(lambda x: x + 10) -> 10 + 10 = 20
    self.assertEqual(result, 20)

  def test_decorated_preserves_name(self):
    chain = Chain().then(lambda x: x)

    @chain.decorator()
    def my_fn(x):
      return x

    self.assertEqual(my_fn.__name__, 'my_fn')


class TestChainAllStepsNone(unittest.TestCase):
  """Chain where every step returns None."""

  def test_all_then_none(self):
    result = (
      Chain(42)
      .then(lambda x: None)
      .then(lambda x: None)
      .then(lambda x: None)
      .run()
    )
    self.assertIsNone(result)

  def test_then_none_then_value(self):
    """None is a valid value; next step receives it."""
    result = (
      Chain(42)
      .then(lambda x: None)
      .then(lambda x: 'after_none' if x is None else 'not_none')
      .run()
    )
    self.assertEqual(result, 'after_none')


class TestChainAllStepsNull(unittest.TestCase):
  """Chain where steps return the Null sentinel."""

  def test_then_null(self):
    """Null is not callable, so then(Null) sets current_value to Null.
    At the end: current_value is Null -> returns None."""
    result = Chain(5).then(Null).run()
    self.assertIsNone(result)

  def test_then_returns_null(self):
    """A step that returns Null: current_value becomes Null -> returns None."""
    result = Chain(5).then(lambda x: Null).run()
    # The lambda returns the Null object. Since Null is not Null sentinel
    # in the context of current_value checks (it IS Null), the end check
    # `if current_value is Null: return None` triggers.
    self.assertIsNone(result)


class TestRunWithNullExplicit(unittest.TestCase):
  """run(Null) is equivalent to run() -- Null means 'no value'."""

  def test_run_null_equivalent_to_run(self):
    c = Chain(5)
    self.assertEqual(c.run(Null), c.run())

  def test_run_null_uses_root(self):
    result = Chain(5).run(Null)
    self.assertEqual(result, 5)


class TestExceptRegistrationEdgeCases(unittest.TestCase):
  """Edge cases in except_ registration."""

  def test_except_with_empty_list_raises(self):
    with self.assertRaises(QuentException):
      Chain().except_(lambda e: e, exceptions=[])

  def test_except_with_string_raises(self):
    with self.assertRaises(TypeError):
      Chain().except_(lambda e: e, exceptions='ValueError')

  def test_except_with_non_exception_type_raises(self):
    with self.assertRaises(TypeError):
      Chain().except_(lambda e: e, exceptions=[int])

  def test_except_with_int_as_exception_raises(self):
    with self.assertRaises(TypeError):
      Chain().except_(lambda e: e, exceptions=42)


class TestControlFlowInAsyncHandlers(IsolatedAsyncioTestCase):
  """Control flow signals in async except/finally handlers."""

  async def test_return_in_async_except(self):
    async def async_step(x):
      raise ValueError('boom')

    c = Chain(5).then(async_step).except_(lambda exc: Chain.return_(99))
    with self.assertRaises(QuentException):
      await c.run()

  async def test_break_in_async_finally(self):
    async def async_step(x):
      return x

    c = Chain(5).then(async_step).finally_(lambda x: Chain.break_())
    with self.assertRaises(QuentException):
      await c.run()


class TestWithOnNonCMDetailed(unittest.TestCase):
  """Detailed tests for with_ on non-context-managers."""

  def test_with_on_int_attribute_error(self):
    with self.assertRaises(AttributeError):
      Chain(42).with_(lambda ctx: ctx).run()

  def test_with_on_string_attribute_error(self):
    """Strings have no __aenter__ or __enter__."""
    with self.assertRaises(AttributeError):
      Chain('hello').with_(lambda ctx: ctx).run()


class TestMapOnNonIterableDetailed(unittest.TestCase):
  """map on various non-iterable types."""

  def test_map_on_object(self):
    with self.assertRaises(TypeError):
      Chain(object()).map(lambda x: x).run()

  def test_map_on_float(self):
    with self.assertRaises(TypeError):
      Chain(3.14).map(lambda x: x).run()

  def test_map_on_bool(self):
    """bool is not iterable."""
    with self.assertRaises(TypeError):
      Chain(True).map(lambda x: x).run()


class TestIterateOnNonIterable(unittest.TestCase):
  """iterate() on a chain that produces a non-iterable value."""

  def test_iterate_int_raises(self):
    with self.assertRaises(TypeError):
      list(Chain(42).iterate())

  def test_iterate_none_raises(self):
    with self.assertRaises(TypeError):
      list(Chain(None).iterate())


class TestEmptyChainWithHandlersOnly(unittest.TestCase):
  """Chains with only handlers (except/finally) and no steps."""

  def test_except_only_no_error(self):
    tracker = []
    result = Chain().except_(lambda exc: tracker.append('called')).run()
    self.assertIsNone(result)
    self.assertEqual(tracker, [])

  def test_finally_only_executes(self):
    tracker = []
    # No root value -> finally_ handler called with no args.
    result = Chain().finally_(lambda: tracker.append('cleanup')).run()
    self.assertIsNone(result)
    self.assertEqual(tracker, ['cleanup'])


if __name__ == '__main__':
  unittest.main()
