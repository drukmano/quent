import types
import functools
from unittest import TestCase, IsolatedAsyncioTestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, Cascade, QuentException, run, FrozenChain


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
# DescriptorWrapperTests
# ---------------------------------------------------------------------------

class DescriptorWrapperTests(TestCase):
  def test_descriptor_class_level_access_returns_self(self):
    """Accessing a decorated method on the class (obj=None) returns the wrapper itself."""
    class MyClass:
      @Chain().then(lambda v: v * 2).decorator()
      def double(self, x):
        return x

    # Class-level access should return the _DescriptorWrapper, not a MethodType
    descriptor = MyClass.__dict__['double']
    result = descriptor.__get__(None, MyClass)
    self.assertIs(result, descriptor)

  def test_descriptor_instance_access_returns_bound_method(self):
    """Accessing a decorated method on an instance (obj!=None) returns a bound MethodType."""
    class MyClass:
      @Chain().then(lambda v: v * 2).decorator()
      def double(self, x):
        return x

    obj = MyClass()
    bound = MyClass.double.__get__(obj, MyClass)
    self.assertIsInstance(bound, types.MethodType)

  def test_descriptor_instance_binds_self_correctly(self):
    """The bound method passes the instance as the first argument."""
    class MyClass:
      def __init__(self, factor):
        self.factor = factor

      @Chain().then(lambda v: v).decorator()
      def get_factor(self):
        return self.factor

    obj = MyClass(42)
    result = obj.get_factor()
    self.assertEqual(result, 42)

  def test_descriptor_on_different_instances(self):
    """Descriptor binds correctly to different instances."""
    class MyClass:
      def __init__(self, val):
        self.val = val

      @Chain().then(lambda v: v + 1).decorator()
      def inc_val(self):
        return self.val

    a = MyClass(10)
    b = MyClass(20)
    self.assertEqual(a.inc_val(), 11)
    self.assertEqual(b.inc_val(), 21)

  def test_descriptor_wrapper_is_callable(self):
    """_DescriptorWrapper is callable and delegates to the wrapped function."""
    @Chain().then(lambda v: v * 3).decorator()
    def triple(x):
      return x

    # When used as a plain function (no class), it should be directly callable
    self.assertEqual(triple(5), 15)

  def test_descriptor_wrapper_has_dict(self):
    """_DescriptorWrapper initializes __dict__ for functools.update_wrapper compatibility."""
    @Chain().then(lambda v: v).decorator()
    def my_func(x):
      return x

    self.assertIsInstance(my_func.__dict__, dict)

  def test_descriptor_class_access_via_dotted_name(self):
    """Accessing a decorated method via MyClass.method returns the wrapper."""
    class MyClass:
      @Chain().then(lambda v: v).decorator()
      def method(self, x):
        return x

    # MyClass.method triggers __get__(None, MyClass) which returns self
    wrapper = MyClass.method
    self.assertTrue(callable(wrapper))


# ---------------------------------------------------------------------------
# FrozenChainDecoratorEdgeCaseTests
# ---------------------------------------------------------------------------

class FrozenChainDecoratorEdgeCaseTests(MyTestCase):
  async def test_decorator_with_lambda(self):
    """Decorator wrapping a lambda (no __name__ issue since lambdas have __name__)."""
    decorator = Chain().then(lambda v: v + 10).decorator()
    wrapped = decorator(lambda x: x * 2)
    await self.assertEqual(wrapped(3), 16)  # (3*2) + 10 = 16

  async def test_decorator_with_builtin(self):
    """Decorator wrapping a builtin function (builtins have __name__)."""
    decorator = Chain().then(lambda v: v).decorator()
    wrapped = decorator(abs)
    await self.assertEqual(wrapped(-5), 5)

  async def test_decorator_preserves_function_name(self):
    """functools.update_wrapper preserves __name__ of the decorated function."""
    @Chain().then(lambda v: v).decorator()
    def my_special_func(x):
      return x

    super(MyTestCase, self).assertEqual(my_special_func.__name__, 'my_special_func')

  async def test_decorator_preserves_docstring(self):
    """functools.update_wrapper preserves __doc__ of the decorated function."""
    @Chain().then(lambda v: v).decorator()
    def documented(x):
      """This is a docstring."""
      return x

    super(MyTestCase, self).assertEqual(documented.__doc__, 'This is a docstring.')

  async def test_decorator_preserves_module(self):
    """functools.update_wrapper preserves __module__ of the decorated function."""
    @Chain().then(lambda v: v).decorator()
    def modular(x):
      return x

    super(MyTestCase, self).assertEqual(modular.__module__, __name__)

  async def test_stacked_decorators(self):
    """Multiple frozen chain decorators can be stacked."""
    @Chain().then(lambda v: v * 2).decorator()
    @Chain().then(lambda v: v + 1).decorator()
    def base(x):
      return x

    # Inner: base(3) -> 3, then +1 -> 4
    # Outer: 4, then *2 -> 8
    await self.assertEqual(base(3), 8)

  async def test_decorator_on_async_function(self):
    """Decorator works with async functions."""
    @Chain().then(lambda v: v * 5).decorator()
    async def async_fn(x):
      return x + 1

    await self.assertEqual(async_fn(2), 15)  # (2+1)*5 = 15

  async def test_decorator_object_without_name(self):
    """Decorator handles objects that lack __name__ via the except AttributeError path."""
    class CallableObj:
      """A callable without __name__."""
      def __call__(self, x):
        return x * 2

    decorator = Chain().then(lambda v: v + 100).decorator()
    # CallableObj() has no __name__, so functools.update_wrapper raises AttributeError
    # which the decorator catches and ignores
    wrapped = decorator(CallableObj())
    await self.assertEqual(wrapped(5), 110)  # 5*2 + 100 = 110

  async def test_frozen_chain_run_and_call_equivalence(self):
    """FrozenChain.run() and FrozenChain.__call__() produce the same result."""
    frozen = Chain(10).then(lambda v: v + 5).freeze()
    for fn, ctx in self.with_fn():
      with ctx:
        run_result = await await_(frozen.run())
        call_result = await await_(frozen())
        super(MyTestCase, self).assertEqual(run_result, call_result)

  async def test_frozen_chain_is_reusable(self):
    """FrozenChain can be run multiple times with different root overrides."""
    frozen = Chain().then(lambda v: v ** 2).freeze()
    await self.assertEqual(frozen.run(3), 9)
    await self.assertEqual(frozen.run(4), 16)
    await self.assertEqual(frozen.run(5), 25)


# ---------------------------------------------------------------------------
# RunClassTests
# ---------------------------------------------------------------------------

class RunClassTests(MyTestCase):
  async def test_run_with_no_args(self):
    """run() with no arguments triggers chain execution with no root override."""
    result = Chain(42) | run()
    await self.assertEqual(result, 42)

  async def test_run_with_literal_root(self):
    """run(literal) passes the literal as root value to the chain."""
    result = Chain() | (lambda v: v * 3) | run(7)
    await self.assertEqual(result, 21)

  async def test_run_with_callable_root(self):
    """run(callable) passes the callable as root value, which gets called."""
    result = Chain() | (lambda v: v + 10) | run(lambda: 5)
    await self.assertEqual(result, 15)

  async def test_run_stores_root_value(self):
    """run instance stores root_value, args, and kwargs."""
    r = run(10, 20, 30, key='val')
    super(MyTestCase, self).assertEqual(r.root_value, 10)
    super(MyTestCase, self).assertEqual(r.args, (20, 30))
    super(MyTestCase, self).assertEqual(r.kwargs, {'key': 'val'})

  async def test_run_default_root_is_null(self):
    """run() with no args stores the Null sentinel as root_value."""
    from quent import Null
    r = run()
    super(MyTestCase, self).assertIs(r.root_value, Null)

  async def test_run_pipe_chain_with_operations(self):
    """run() at the end of a pipe chain executes all operations."""
    result = Chain(2) | (lambda v: v + 3) | (lambda v: v * 10) | run()
    await self.assertEqual(result, 50)  # (2+3)*10

  async def test_run_pipe_with_async(self):
    """run() works with async operations in pipe chains."""
    for fn, ctx in self.with_fn():
      with ctx:
        result = Chain(5) | fn | (lambda v: v * 4) | run()
        await self.assertEqual(result, 20)

  async def test_run_is_instance_check_in_or(self):
    """Chain.__or__ checks isinstance(other, run) to trigger execution."""
    r = run()
    super(MyTestCase, self).assertIsInstance(r, run)


# ---------------------------------------------------------------------------
# ChainCallWrapperTests
# ---------------------------------------------------------------------------

class ChainCallWrapperTests(MyTestCase):
  async def test_wrapper_delegates_to_chain_run(self):
    """_ChainCallWrapper delegates calls to the frozen chain's _run method."""
    @Chain().then(lambda v: v * 2).decorator()
    def double(x):
      return x

    await self.assertEqual(double(5), 10)

  async def test_wrapper_passes_args_to_wrapped_fn(self):
    """_ChainCallWrapper forwards positional args to the wrapped function."""
    @Chain().then(lambda v: v).decorator()
    def add(a, b):
      return a + b

    await self.assertEqual(add(3, 7), 10)

  async def test_wrapper_passes_kwargs_to_wrapped_fn(self):
    """_ChainCallWrapper forwards keyword arguments to the wrapped function."""
    @Chain().then(lambda v: v).decorator()
    def greet(name, greeting='Hello'):
      return f'{greeting}, {name}!'

    await self.assertEqual(greet('World', greeting='Hi'), 'Hi, World!')

  async def test_wrapper_chain_processes_return_value(self):
    """_ChainCallWrapper feeds the function's return value into the chain."""
    @Chain().then(lambda v: v.upper()).decorator()
    def make_str(s):
      return s

    await self.assertEqual(make_str('hello'), 'HELLO')

  async def test_wrapper_with_method_on_instance(self):
    """_ChainCallWrapper works as a method descriptor on class instances."""
    class Processor:
      def __init__(self, scale):
        self.scale = scale

      @Chain().then(lambda v: v).decorator()
      def process(self, x):
        return x * self.scale

    p = Processor(3)
    await self.assertEqual(p.process(10), 30)

  async def test_wrapper_with_async_wrapped_fn(self):
    """_ChainCallWrapper handles async wrapped functions correctly."""
    @Chain().then(lambda v: v + 100).decorator()
    async def async_compute(x):
      return x * 2

    await self.assertEqual(async_compute(5), 110)  # (5*2) + 100

  async def test_wrapper_exception_propagates(self):
    """Exceptions from the wrapped function propagate through the chain."""
    @Chain().then(lambda v: v).decorator()
    def failing():
      raise TestExc('boom')

    with self.assertRaises(TestExc):
      failing()
