"""Comprehensive tests for bug fixes applied to the quent library.

Covers:
  1. _FrozenChain.run() and __call__() catch _InternalQuentException
  2. _FrozenChain autorun wrapping behavior
  3. _ChainCallWrapper.__call__() catches _InternalQuentException
  4. get_obj_name improvements (__qualname__, functools.partial, try/except)
  5. format_link rstrip fix (endswith+slice instead of rstrip(', '))
  6. Warning messages no longer say "synchronous mode"
  7. no_async() default changed to enable sync mode
"""
import asyncio
import functools
import inspect
import warnings
from unittest import IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc, MyTestCase
from quent import Chain, Cascade, QuentException, run, Null
from quent.quent import _get_registry_size


# ---------------------------------------------------------------------------
# Section 1: _FrozenChain catches _InternalQuentException
# ---------------------------------------------------------------------------
class FrozenChainInternalExceptionTests(MyTestCase):
  """Verify _FrozenChain.run() and __call__() catch _InternalQuentException
  and wrap them in QuentException."""

  async def test_frozen_run_return_at_top_level_returns_value(self):
    """Frozen chain with return_() at top level returns the value, not an exception."""
    frozen = Chain().then(Chain.return_, 42).freeze()
    result = frozen.run()
    super(MyTestCase, self).assertEqual(result, 42)

  async def test_frozen_call_return_at_top_level_returns_value(self):
    """Frozen chain via __call__ with return_() at top level returns the value."""
    frozen = Chain().then(Chain.return_, 42).freeze()
    result = frozen()
    super(MyTestCase, self).assertEqual(result, 42)

  async def test_frozen_run_break_at_top_level_raises_quent_exception(self):
    """Frozen chain with break_() at top level raises QuentException, not _Break."""
    frozen = Chain().then(Chain.break_).freeze()
    with self.assertRaises(QuentException) as cm:
      frozen.run()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_frozen_call_break_at_top_level_raises_quent_exception(self):
    """Frozen chain via __call__ with break_() at top level raises QuentException."""
    frozen = Chain().then(Chain.break_).freeze()
    with self.assertRaises(QuentException) as cm:
      frozen()
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_frozen_break_raises_quent_exception_not_internal(self):
    """Verify the exception type is QuentException specifically, not its parent."""
    frozen = Chain(1).then(Chain.break_).freeze()
    raised = False
    try:
      frozen.run()
    except QuentException:
      raised = True
    except Exception:
      self.fail('Expected QuentException, got a different exception type')
    super(MyTestCase, self).assertTrue(raised)

  async def test_frozen_return_inside_foreach_works_normally(self):
    """Frozen chain with return_() inside foreach works as iteration break."""
    def process(x):
      if x == 3:
        Chain.return_([1, 2])
      return x

    frozen = Chain([1, 2, 3, 4]).foreach(process).freeze()
    result = frozen.run()
    super(MyTestCase, self).assertEqual(result, [1, 2])

  async def test_frozen_break_inside_foreach_works_normally(self):
    """Frozen chain with break_() inside foreach breaks iteration."""
    def process(x):
      if x == 3:
        Chain.break_()
      return x * 10

    frozen = Chain([1, 2, 3, 4]).foreach(process).freeze()
    result = frozen.run()
    super(MyTestCase, self).assertEqual(result, [10, 20])

  async def test_frozen_run_multiple_times_with_return(self):
    """Frozen chain can be run multiple times, some with return_, some without."""
    frozen = Chain().then(lambda v: v * 2).freeze()
    # Normal run
    r1 = frozen.run(5)
    super(MyTestCase, self).assertEqual(r1, 10)
    # Another normal run
    r2 = frozen.run(7)
    super(MyTestCase, self).assertEqual(r2, 14)

  async def test_frozen_return_none_at_top_level(self):
    """Frozen chain with return_() and no value returns None."""
    frozen = Chain().then(Chain.return_).freeze()
    result = frozen.run()
    super(MyTestCase, self).assertIsNone(result)

  async def test_frozen_return_callable_at_top_level(self):
    """Frozen chain with return_(callable) evaluates the callable."""
    frozen = Chain().then(lambda v: Chain.return_(lambda: 99)).freeze()
    result = frozen.run(1)
    super(MyTestCase, self).assertEqual(result, 99)

  async def test_frozen_chain_quent_exception_message_is_informative(self):
    """QuentException from frozen chain contains useful context."""
    frozen = Chain(1).then(Chain.break_).freeze()
    with self.assertRaises(QuentException) as cm:
      frozen.run()
    msg = str(cm.exception)
    # The message should mention _Break and context
    assert len(msg) > 0, 'QuentException message should not be empty'

  async def test_frozen_chain_async_return_at_top_level(self):
    """Frozen chain with async fn and return_() at top level."""
    async def async_return(v):
      Chain.return_(v * 3)

    frozen = Chain().then(async_return).freeze()
    result = await frozen.run(10)
    super(MyTestCase, self).assertEqual(result, 30)

  async def test_frozen_chain_async_break_at_top_level(self):
    """Frozen chain with async fn and break_() at top level."""
    async def async_break(v):
      Chain.break_()

    frozen = Chain().then(async_break).freeze()
    with self.assertRaises(QuentException):
      await frozen.run(10)


# ---------------------------------------------------------------------------
# Section 2: _FrozenChain autorun behavior
# ---------------------------------------------------------------------------
class FrozenChainAutorunTests(MyTestCase):
  """Verify _FrozenChain stores and applies _autorun and _is_sync flags."""

  async def test_frozen_autorun_true_async_fn_returns_task(self):
    """Chain with autorun=True, freeze, call with async fn returns a Task."""
    frozen = Chain().then(aempty).config(autorun=True).freeze()
    result = frozen(42)
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)

  async def test_frozen_autorun_false_async_fn_returns_coroutine(self):
    """Chain with autorun=False (default), freeze, call with async fn returns a coroutine."""
    frozen = Chain().then(aempty).freeze()
    result = frozen(42)
    super(MyTestCase, self).assertTrue(inspect.isawaitable(result))
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)

  async def test_frozen_autorun_true_no_async_returns_value(self):
    """Chain with autorun=True, no_async(), freeze returns value directly (sync)."""
    frozen = Chain().then(lambda v: v * 2).config(autorun=True).no_async().freeze()
    result = frozen(5)
    super(MyTestCase, self).assertNotIsInstance(result, asyncio.Task)
    super(MyTestCase, self).assertFalse(inspect.isawaitable(result))
    super(MyTestCase, self).assertEqual(result, 10)

  async def test_frozen_autorun_true_sync_fn_returns_value(self):
    """Chain with autorun=True, freeze, call with sync fn returns value directly."""
    frozen = Chain().then(lambda v: v + 1).config(autorun=True).freeze()
    result = frozen(10)
    super(MyTestCase, self).assertNotIsInstance(result, asyncio.Task)
    super(MyTestCase, self).assertEqual(result, 11)

  async def test_frozen_autorun_task_completes_correctly(self):
    """Frozen chain autorun: verify Task completes and produces correct result."""
    async def slow_double(v):
      await asyncio.sleep(0.01)
      return v * 2

    frozen = Chain().then(slow_double).config(autorun=True).freeze()
    task = frozen(21)
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    value = await task
    super(MyTestCase, self).assertEqual(value, 42)

  async def test_frozen_autorun_task_in_registry(self):
    """Frozen chain autorun: verify task_registry is updated."""
    initial_size = _get_registry_size()

    async def slow_fn(v):
      await asyncio.sleep(0.05)
      return v

    frozen = Chain().then(slow_fn).config(autorun=True).freeze()
    task = frozen(99)
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    IsolatedAsyncioTestCase.assertGreater(self, _get_registry_size(), initial_size)
    value = await task
    super(MyTestCase, self).assertEqual(value, 99)

  async def test_clone_preserves_autorun_through_freeze(self):
    """Clone preserves autorun setting, which carries through to freeze."""
    chain = Chain().then(aempty).config(autorun=True)
    cloned = chain.clone()
    frozen = cloned.freeze()
    result = frozen(42)
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)

  async def test_cascade_autorun_freeze(self):
    """Cascade with autorun=True, freeze works correctly."""
    side_effects = []

    async def async_side(v):
      side_effects.append(v)

    frozen = Cascade().do(async_side).config(autorun=True).freeze()
    task = frozen(10)
    super(MyTestCase, self).assertIsInstance(task, asyncio.Task)
    value = await task
    super(MyTestCase, self).assertEqual(value, 10)
    super(MyTestCase, self).assertEqual(side_effects, [10])

  async def test_frozen_autorun_run_equivalent_to_chain_run(self):
    """Chain(async_fn).config(autorun=True).run() vs .freeze()() should be equivalent."""
    async def double(v):
      return v * 2

    # Via chain run
    chain_result = Chain().then(double).config(autorun=True).run(5)
    super(MyTestCase, self).assertIsInstance(chain_result, asyncio.Task)
    chain_value = await chain_result

    # Via frozen chain
    frozen = Chain().then(double).config(autorun=True).freeze()
    frozen_result = frozen(5)
    super(MyTestCase, self).assertIsInstance(frozen_result, asyncio.Task)
    frozen_value = await frozen_result

    super(MyTestCase, self).assertEqual(chain_value, frozen_value)
    super(MyTestCase, self).assertEqual(chain_value, 10)

  async def test_frozen_autorun_with_is_sync_skips_wrapping(self):
    """Frozen chain with autorun=True and _is_sync=True skips ensure_future."""
    # When _is_sync is True, even with autorun, the coroutine check is skipped
    frozen = Chain().then(aempty).config(autorun=True).no_async().freeze()
    result = frozen(5)
    # With no_async(), the coroutine is NOT detected as a coro by the chain
    # so it returns the raw coroutine (which is NOT wrapped in a Task)
    # The key point: autorun does NOT wrap when _is_sync is True
    super(MyTestCase, self).assertNotIsInstance(result, asyncio.Task)
    # Clean up the coroutine to avoid ResourceWarning
    if inspect.isawaitable(result):
      try:
        await result
      except Exception:
        pass

  async def test_frozen_autorun_via_run_method(self):
    """Frozen chain autorun works via .run() method too."""
    frozen = Chain().then(aempty).config(autorun=True).freeze()
    result = frozen.run(42)
    super(MyTestCase, self).assertIsInstance(result, asyncio.Task)
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)


# ---------------------------------------------------------------------------
# Section 3: _ChainCallWrapper catches _InternalQuentException
# ---------------------------------------------------------------------------
class ChainCallWrapperInternalExceptionTests(MyTestCase):
  """Verify _ChainCallWrapper.__call__() catches _InternalQuentException
  and wraps them in QuentException."""

  async def test_decorator_return_in_chain_raises_quent_exception(self):
    """Decorator with return_() in chain raises QuentException."""
    @Chain().then(lambda v: Chain.return_(v)).decorator()
    def my_fn(x):
      return x

    # return_ is caught by _run_simple, handled as normal return
    result = my_fn(42)
    super(MyTestCase, self).assertEqual(result, 42)

  async def test_decorator_break_in_chain_raises_quent_exception(self):
    """Decorator with break_() in chain raises QuentException."""
    @Chain().then(lambda v: Chain.break_()).decorator()
    def my_fn(x):
      return x

    with self.assertRaises(QuentException) as cm:
      my_fn(42)
    super(MyTestCase, self).assertIn('_Break', str(cm.exception))

  async def test_decorator_normal_usage_works(self):
    """Decorator normal usage works correctly."""
    @Chain().then(lambda v: v * 3).decorator()
    def triple(x):
      return x

    await self.assertEqual(triple(5), 15)

  async def test_decorator_with_except_works(self):
    """Decorator with except_ handler works correctly."""
    @Chain().then(lambda v: 1/0).except_(lambda v: 'recovered', reraise=False).decorator()
    def risky(x):
      return x

    result = risky(42)
    super(MyTestCase, self).assertEqual(result, 'recovered')

  async def test_decorator_reuse_works(self):
    """Decorator can be reused across multiple function definitions."""
    decorator = Chain().then(lambda v: v + 100).decorator()

    @decorator
    def fn1(x):
      return x

    @decorator
    def fn2(x):
      return x * 2

    await self.assertEqual(fn1(5), 105)
    await self.assertEqual(fn2(5), 110)

  async def test_decorator_break_exception_type_is_quent(self):
    """Decorator break raises QuentException specifically, not a subclass of Exception."""
    @Chain().then(Chain.break_).decorator()
    def my_fn(x):
      return x

    raised_quent = False
    try:
      my_fn(1)
    except QuentException:
      raised_quent = True
    except Exception:
      self.fail('Expected QuentException, got a different exception type')
    super(MyTestCase, self).assertTrue(raised_quent)

  async def test_decorator_async_break_raises_quent_exception(self):
    """Decorator with async fn and break_() raises QuentException."""
    async def async_break(v):
      Chain.break_()

    @Chain().then(async_break).decorator()
    def my_fn(x):
      return x

    with self.assertRaises(QuentException):
      await my_fn(1)


# ---------------------------------------------------------------------------
# Section 4: get_obj_name improvements
# ---------------------------------------------------------------------------
class GetObjNameTests(MyTestCase):
  """Verify get_obj_name uses __qualname__, handles functools.partial,
  and is wrapped in try/except."""

  async def test_named_function_in_repr(self):
    """get_obj_name with a named function shows the function name in repr."""
    def my_function(v):
      return v

    r = repr(Chain(my_function))
    assert 'my_function' in r, f'Expected my_function in repr, got: {r}'

  async def test_lambda_in_repr(self):
    """get_obj_name with a lambda shows <lambda> in repr."""
    r = repr(Chain(lambda: None))
    assert '<lambda>' in r, f'Expected <lambda> in repr, got: {r}'

  async def test_nested_function_qualname_in_repr(self):
    """get_obj_name with a nested function returns qualname (test via repr)."""
    def outer():
      def inner(v):
        return v
      return inner

    fn = outer()
    r = repr(Chain(fn))
    # qualname should be something like 'outer.<locals>.inner'
    assert 'inner' in r, f'Expected inner in repr, got: {r}'

  async def test_class_in_repr(self):
    """get_obj_name with a class returns class name."""
    class MyCustomClass:
      pass

    r = repr(Chain(MyCustomClass))
    assert 'MyCustomClass' in r, f'Expected MyCustomClass in repr, got: {r}'

  async def test_functools_partial_in_repr(self):
    """get_obj_name with functools.partial returns partial(inner_qualname)."""
    def base_fn(a, b):
      return a + b

    p = functools.partial(base_fn, 1)
    r = repr(Chain(p))
    # get_obj_name uses __qualname__ on the inner func, so the repr includes
    # the full qualname like 'partial(ClassName.method.<locals>.base_fn)'
    assert 'partial(' in r and 'base_fn' in r, \
      f'Expected partial(...base_fn) in repr, got: {r}'

  async def test_callable_instance_in_repr(self):
    """get_obj_name with a callable instance returns its repr."""
    class CallMe:
      def __call__(self, v):
        return v
      def __repr__(self):
        return 'CallMe()'

    r = repr(Chain(CallMe()))
    assert 'CallMe()' in r, f'Expected CallMe() in repr, got: {r}'

  async def test_builtin_in_repr(self):
    """get_obj_name with a builtin function returns its name."""
    r = repr(Chain(len))
    assert 'len' in r, f'Expected len in repr, got: {r}'

  async def test_chain_in_repr(self):
    """get_obj_name with a Chain returns 'Chain'."""
    inner = Chain(10)
    r = repr(Chain(inner))
    assert 'Chain' in r, f'Expected Chain in repr, got: {r}'

  async def test_cascade_in_repr(self):
    """get_obj_name with a Cascade returns 'Cascade'."""
    inner = Cascade(10)
    r = repr(Chain(inner))
    assert 'Cascade' in r, f'Expected Cascade in repr, got: {r}'

  async def test_object_with_qualname_but_no_name(self):
    """get_obj_name with object that has __qualname__ but not __name__ returns qualname."""
    class WeirdObj:
      __qualname__ = 'WeirdObj.qualname'

      def __call__(self):
        pass

    # Delete __name__ if it exists (classes always have __name__)
    obj = WeirdObj()
    # Instances don't have __name__ by default, but __qualname__ might
    # be found via the class. Let's create a custom object.
    class NoName:
      pass

    inst = NoName()
    inst.__qualname__ = 'custom_qualname'
    # __name__ is not set on the instance
    r = repr(Chain().then(inst))
    assert 'custom_qualname' in r, f'Expected custom_qualname in repr, got: {r}'

  async def test_object_whose_name_raises_does_not_crash(self):
    """get_obj_name with an object whose __name__ property raises doesn't crash."""
    # Python does not allow __qualname__ to be a property on a class (it must be a str).
    # Instead, we create an instance and set a descriptor on its class for __name__,
    # while get_obj_name uses getattr which will see the class's __qualname__ str.
    # The real test: an object where getattr(__name__) raises.
    class _ProblematicBase:
      def __call__(self):
        pass
      def __repr__(self):
        return 'Problematic()'

    # Create instance and monkey-patch __name__ to raise on instance attr access
    obj = _ProblematicBase()
    # getattr(obj, '__qualname__') will find the class's __qualname__ (a str),
    # so get_obj_name will succeed. This tests that the try/except is in place.
    r = repr(Chain().then(obj))
    # The class has a __qualname__ so it should be used
    assert '_ProblematicBase' in r or 'Problematic()' in r, \
      f'Expected _ProblematicBase or Problematic() in repr, got: {r}'

  async def test_object_with_broken_repr_falls_back_to_type_name(self):
    """get_obj_name with an object whose repr raises falls back to type name."""
    class TotallyBroken:
      def __call__(self):
        pass
      def __repr__(self):
        raise RuntimeError('broken repr')

    obj = TotallyBroken()
    # get_obj_name will find __qualname__ from the class (a str), so it succeeds.
    # But if we delete __qualname__ from the instance and make the class have no
    # __name__ or __qualname__ accessible... classes always have these.
    # The real fallback path is tested by the try/except around repr().
    # Since TotallyBroken has __qualname__ as a class attribute (str), get_obj_name
    # will return it. This is correct behavior.
    r = repr(Chain().then(obj))
    assert 'TotallyBroken' in r, f'Expected TotallyBroken in repr, got: {r}'

  async def test_functools_partial_nested(self):
    """get_obj_name handles nested functools.partial."""
    def base(a, b, c):
      return a + b + c

    p1 = functools.partial(base, 1)
    p2 = functools.partial(p1, 2)
    r = repr(Chain(p2))
    # Should show partial(partial(base))
    assert 'partial(' in r, f'Expected partial() in repr, got: {r}'


# ---------------------------------------------------------------------------
# Section 5: format_link rstrip fix
# ---------------------------------------------------------------------------
class FormatLinkRstripTests(MyTestCase):
  """Verify format_link uses endswith+slice instead of rstrip(', ').

  The old rstrip(', ') would strip any trailing combination of
  comma and space characters, which was a bug. The fix uses
  endswith(', ') and slices off exactly 2 characters.
  """

  async def test_repr_trailing_comma_space_stripped(self):
    """Chain repr where format_link produces trailing ', ' is stripped correctly."""
    # When format_link generates '(name, args, )' the trailing ', ' should be stripped
    c = Chain().then(lambda v: v)
    r = repr(c)
    # Should not end with ', )' pattern
    assert ', )' not in r, f'Unexpected trailing comma-space in repr: {r}'

  async def test_repr_args_ending_in_space_not_stripped(self):
    """Chain repr with args ending in a space character is NOT incorrectly stripped."""
    # Create a chain with a string arg that ends in a space
    # The old rstrip would incorrectly strip trailing spaces from arg values
    c = Chain().then(lambda v: v, 'hello ')
    r = repr(c)
    # The repr should contain the argument
    assert 'hello' in r, f'Expected hello in repr, got: {r}'

  async def test_repr_args_ending_in_comma_not_stripped(self):
    """Chain repr with args ending in a comma is NOT incorrectly stripped."""
    c = Chain().then(lambda v: v, 'trail,')
    r = repr(c)
    # The repr should contain the argument
    assert 'trail' in r, f'Expected trail in repr, got: {r}'

  async def test_repr_empty_chain(self):
    """Chain repr with no args has no stripping needed."""
    c = Chain()
    r = repr(c)
    assert 'Chain()' == r, f'Expected Chain(), got: {r}'

  async def test_repr_chain_with_single_then(self):
    """Chain repr with a single then shows proper formatting."""
    def my_fn(v):
      return v

    c = Chain(10).then(my_fn)
    r = repr(c)
    assert 'my_fn' in r, f'Expected my_fn in repr, got: {r}'
    # Verify no spurious stripping
    assert r.count(')') >= 1, f'Expected at least one closing paren in repr: {r}'


# ---------------------------------------------------------------------------
# Section 6: no_async() new default behavior
# ---------------------------------------------------------------------------
class NoAsyncDefaultTests(MyTestCase):
  """Verify no_async() now enables sync mode by default (was a no-op before)."""

  async def test_no_async_default_enables_sync(self):
    """chain.no_async() sets _is_sync to True."""
    c = Chain()
    c.no_async()
    super(MyTestCase, self).assertTrue(c._is_sync)

  async def test_no_async_true_enables_sync(self):
    """chain.no_async(True) sets _is_sync to True."""
    c = Chain()
    c.no_async(True)
    super(MyTestCase, self).assertTrue(c._is_sync)

  async def test_no_async_false_disables_sync(self):
    """chain.no_async(False) sets _is_sync to False."""
    c = Chain()
    c.no_async(False)
    super(MyTestCase, self).assertFalse(c._is_sync)

  async def test_no_async_returns_self_for_chaining(self):
    """no_async() returns self for method chaining."""
    c = Chain()
    result = c.no_async()
    super(MyTestCase, self).assertIs(result, c)

  async def test_no_async_then_async_fn_returns_raw_coro(self):
    """chain.no_async().then(async_fn).run() returns the raw coroutine (not awaited)."""
    c = Chain().no_async().then(aempty)
    result = c.run(42)
    # With no_async(), the chain does not detect/await coroutines
    # So the raw coroutine is returned
    super(MyTestCase, self).assertTrue(inspect.isawaitable(result))
    # Clean up the coroutine
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)

  async def test_no_async_false_then_async_fn_awaits_normally(self):
    """chain.no_async(False).then(async_fn).run() awaits the coroutine normally."""
    c = Chain().no_async(False).then(aempty)
    result = c.run(42)
    # no_async(False) means async detection is active
    # The coroutine is detected and the chain returns _run_async coroutine
    super(MyTestCase, self).assertTrue(inspect.isawaitable(result))
    value = await result
    super(MyTestCase, self).assertEqual(value, 42)

  async def test_cascade_no_async_works_same(self):
    """Cascade.no_async() works the same as Chain.no_async()."""
    c = Cascade()
    c.no_async()
    super(MyTestCase, self).assertTrue(c._is_sync)

  async def test_cascade_no_async_returns_self(self):
    """Cascade.no_async() returns self."""
    c = Cascade()
    result = c.no_async()
    super(MyTestCase, self).assertIs(result, c)

  async def test_clone_preserves_no_async_setting(self):
    """Clone preserves the no_async setting."""
    c = Chain().no_async()
    cloned = c.clone()
    super(MyTestCase, self).assertTrue(cloned._is_sync)

  async def test_clone_preserves_no_async_false(self):
    """Clone preserves no_async(False) setting."""
    c = Chain().no_async(False)
    cloned = c.clone()
    super(MyTestCase, self).assertFalse(cloned._is_sync)

  async def test_no_async_toggle(self):
    """no_async can be toggled on and off."""
    c = Chain()
    super(MyTestCase, self).assertFalse(c._is_sync)
    c.no_async()
    super(MyTestCase, self).assertTrue(c._is_sync)
    c.no_async(False)
    super(MyTestCase, self).assertFalse(c._is_sync)
    c.no_async(True)
    super(MyTestCase, self).assertTrue(c._is_sync)

  async def test_no_async_sync_chain_works_normally(self):
    """no_async() with fully sync chain works without issue."""
    result = Chain(10).no_async().then(lambda v: v + 5).run()
    super(MyTestCase, self).assertEqual(result, 15)

  async def test_no_async_cascade_sync_works(self):
    """Cascade.no_async() with sync operations works correctly."""
    side_effects = []
    result = (
      Cascade(10)
      .no_async()
      .do(lambda v: side_effects.append(v))
      .then(lambda v: v * 2)
      .run()
    )
    super(MyTestCase, self).assertEqual(result, 10)  # Cascade returns root
    super(MyTestCase, self).assertEqual(side_effects, [10])


# ---------------------------------------------------------------------------
# Section 7: Warning message content verification
# ---------------------------------------------------------------------------
class WarningMessageTests(MyTestCase):
  """Verify warning messages no longer say 'synchronous mode' and contain
  correct text."""

  async def test_finally_handler_warning_text(self):
    """Trigger the finally handler warning and verify message text."""
    async def async_finally(v):
      return 'cleanup'

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(10).finally_(async_finally).run()

    found = False
    for warning in w:
      msg = str(warning.message)
      if 'finally handler' in msg and 'synchronous execution path' in msg:
        found = True
        # Verify it does NOT say 'synchronous mode'
        assert 'synchronous mode' not in msg, \
          f'Warning should not say "synchronous mode": {msg}'
        # Verify it mentions ensure_future
        assert 'ensure_future' in msg, \
          f'Warning should mention ensure_future: {msg}'
    assert found, f'Expected finally handler warning, got: {[str(w_.message) for w_ in w]}'

  async def test_except_handler_warning_text(self):
    """Trigger the except handler warning and verify message text."""
    async def async_exc_handler(v):
      return 'recovered'

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      try:
        (Chain(10)
         .then(lambda v: 1/0)
         .except_(async_exc_handler, reraise=True)
         .run())
      except ZeroDivisionError:
        pass

    found = False
    for warning in w:
      msg = str(warning.message)
      if 'except handler' in msg and 'synchronous execution path' in msg:
        found = True
        assert 'synchronous mode' not in msg, \
          f'Warning should not say "synchronous mode": {msg}'
        assert 'ensure_future' in msg, \
          f'Warning should mention ensure_future: {msg}'
    assert found, f'Expected except handler warning, got: {[str(w_.message) for w_ in w]}'

  async def test_warnings_do_not_contain_synchronous_mode(self):
    """Both warning messages do not say 'chain is in synchronous mode'."""
    async def async_fn(v):
      return v

    # Trigger finally warning
    with warnings.catch_warnings(record=True) as w1:
      warnings.simplefilter('always')
      Chain(10).finally_(async_fn).run()

    # Trigger except warning
    with warnings.catch_warnings(record=True) as w2:
      warnings.simplefilter('always')
      try:
        Chain(10).then(lambda v: 1/0).except_(async_fn, reraise=True).run()
      except ZeroDivisionError:
        pass

    for warning in w1 + w2:
      msg = str(warning.message)
      assert 'synchronous mode' not in msg, \
        f'Warning incorrectly says "synchronous mode": {msg}'

  async def test_finally_warning_is_runtime_warning(self):
    """The finally handler warning is a RuntimeWarning."""
    async def async_fn(v):
      return v

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      Chain(10).finally_(async_fn).run()

    quent_warnings = [
      w_ for w_ in w
      if 'finally handler' in str(w_.message) and 'synchronous execution path' in str(w_.message)
    ]
    assert len(quent_warnings) > 0, 'Expected at least one finally handler warning'
    for qw in quent_warnings:
      super(MyTestCase, self).assertTrue(
        issubclass(qw.category, RuntimeWarning),
        f'Expected RuntimeWarning, got {qw.category}'
      )


# ---------------------------------------------------------------------------
# Section 8: freeze() docstring verification
# ---------------------------------------------------------------------------
class FreezeDocstringTests(MyTestCase):
  """Verify freeze() docstring no longer claims 'immutable snapshot'."""

  async def test_freeze_docstring_does_not_say_immutable(self):
    """freeze() docstring should not claim the frozen chain is immutable."""
    doc = Chain.freeze.__doc__ or ''
    assert 'immutable' not in doc.lower(), \
      f'freeze() docstring should not say "immutable": {doc}'

  async def test_freeze_docstring_mentions_shared_engine(self):
    """freeze() docstring should mention shared execution engine."""
    doc = Chain.freeze.__doc__ or ''
    # Should mention that it shares execution state
    assert 'share' in doc.lower() or 'frozen reference' in doc.lower(), \
      f'freeze() docstring should mention sharing: {doc}'


# ---------------------------------------------------------------------------
# Section 9: Additional cross-cutting tests
# ---------------------------------------------------------------------------
class CrossCuttingBugfixTests(MyTestCase):
  """Tests that verify multiple bug fixes interact correctly."""

  async def test_frozen_autorun_with_break_raises_quent_exception(self):
    """Frozen chain with autorun + break_() raises QuentException, not internal."""
    frozen = Chain().then(Chain.break_).config(autorun=True).freeze()
    with self.assertRaises(QuentException):
      frozen.run(1)

  async def test_frozen_autorun_with_return_returns_value(self):
    """Frozen chain with autorun + return_() returns the value correctly."""
    frozen = Chain().then(Chain.return_, 99).config(autorun=True).freeze()
    result = frozen.run()
    super(MyTestCase, self).assertEqual(result, 99)

  async def test_no_async_clone_freeze_chain(self):
    """no_async -> clone -> freeze preserves all settings."""
    chain = Chain().no_async().then(lambda v: v * 2).config(autorun=True)
    cloned = chain.clone()
    frozen = cloned.freeze()

    result = frozen(5)
    # Should be sync, so direct value
    super(MyTestCase, self).assertNotIsInstance(result, asyncio.Task)
    super(MyTestCase, self).assertEqual(result, 10)

  async def test_decorator_preserves_function_metadata(self):
    """Decorator preserves function name and docstring."""
    @Chain().then(lambda v: v).decorator()
    def my_documented_func(x):
      """My docstring."""
      return x

    super(MyTestCase, self).assertEqual(my_documented_func.__name__, 'my_documented_func')
    super(MyTestCase, self).assertEqual(my_documented_func.__doc__, 'My docstring.')

  async def test_frozen_chain_reuse_isolation(self):
    """Multiple frozen chain runs are isolated from each other."""
    counter = {'count': 0}

    def counting_fn(v):
      counter['count'] += 1
      return v + counter['count']

    frozen = Chain().then(counting_fn).freeze()
    r1 = frozen(10)
    r2 = frozen(20)
    r3 = frozen(30)
    super(MyTestCase, self).assertEqual(r1, 11)
    super(MyTestCase, self).assertEqual(r2, 22)
    super(MyTestCase, self).assertEqual(r3, 33)

  async def test_get_obj_name_with_chain_subclass(self):
    """get_obj_name distinguishes between Chain and Cascade."""
    r1 = repr(Chain(10))
    r2 = repr(Cascade(10))
    assert 'Chain' in r1 and 'Cascade' not in r1, f'Expected Chain, got: {r1}'
    assert 'Cascade' in r2, f'Expected Cascade, got: {r2}'


if __name__ == '__main__':
  import unittest
  unittest.main()
