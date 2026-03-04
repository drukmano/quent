import types
import functools
from unittest import IsolatedAsyncioTestCase, TestCase
from tests.utils import empty, aempty, await_, TestExc
from quent import Chain, QuentException


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
# ChainDecoratorEdgeCaseTests
# ---------------------------------------------------------------------------

class ChainDecoratorEdgeCaseTests(IsolatedAsyncioTestCase):
  async def test_decorator_with_lambda(self):
    """Decorator wrapping a lambda (no __name__ issue since lambdas have __name__)."""
    decorator = Chain().then(lambda v: v + 10).decorator()
    wrapped = decorator(lambda x: x * 2)
    self.assertEqual(wrapped(3), 16)  # (3*2) + 10 = 16

  async def test_decorator_with_builtin(self):
    """Decorator wrapping a builtin function (builtins have __name__)."""
    decorator = Chain().then(lambda v: v).decorator()
    wrapped = decorator(abs)
    self.assertEqual(wrapped(-5), 5)

  async def test_decorator_preserves_function_name(self):
    """functools.update_wrapper preserves __name__ of the decorated function."""
    @Chain().then(lambda v: v).decorator()
    def my_special_func(x):
      return x

    self.assertEqual(my_special_func.__name__, 'my_special_func')

  async def test_decorator_preserves_docstring(self):
    """functools.update_wrapper preserves __doc__ of the decorated function."""
    @Chain().then(lambda v: v).decorator()
    def documented(x):
      """This is a docstring."""
      return x

    self.assertEqual(documented.__doc__, 'This is a docstring.')

  async def test_decorator_preserves_module(self):
    """functools.update_wrapper preserves __module__ of the decorated function."""
    @Chain().then(lambda v: v).decorator()
    def modular(x):
      return x

    self.assertEqual(modular.__module__, __name__)

  async def test_stacked_decorators(self):
    """Multiple chain decorators can be stacked."""
    @Chain().then(lambda v: v * 2).decorator()
    @Chain().then(lambda v: v + 1).decorator()
    def base(x):
      return x

    # Inner: base(3) -> 3, then +1 -> 4
    # Outer: 4, then *2 -> 8
    self.assertEqual(base(3), 8)

  async def test_decorator_on_async_function(self):
    """Decorator works with async functions."""
    @Chain().then(lambda v: v * 5).decorator()
    async def async_fn(x):
      return x + 1

    self.assertEqual(await async_fn(2), 15)  # (2+1)*5 = 15

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
    self.assertEqual(wrapped(5), 110)  # 5*2 + 100 = 110

  async def test_chain_run_and_call_equivalence(self):
    """Chain.run() and Chain.__call__() produce the same result."""
    c = Chain(10).then(lambda v: v + 5)
    run_result = await await_(c.run())
    call_result = await await_(c())
    self.assertEqual(run_result, call_result)

  async def test_chain_is_reusable(self):
    """Chain can be run multiple times with different root overrides."""
    c = Chain().then(lambda v: v ** 2)
    self.assertEqual(c.run(3), 9)
    self.assertEqual(c.run(4), 16)
    self.assertEqual(c.run(5), 25)


# ---------------------------------------------------------------------------
# ChainCallWrapperTests
# ---------------------------------------------------------------------------

class ChainCallWrapperTests(IsolatedAsyncioTestCase):
  async def test_wrapper_delegates_to_chain_run(self):
    """_ChainCallWrapper delegates calls to the chain's _run method."""
    @Chain().then(lambda v: v * 2).decorator()
    def double(x):
      return x

    self.assertEqual(double(5), 10)

  async def test_wrapper_passes_args_to_wrapped_fn(self):
    """_ChainCallWrapper forwards positional args to the wrapped function."""
    @Chain().then(lambda v: v).decorator()
    def add(a, b):
      return a + b

    self.assertEqual(add(3, 7), 10)

  async def test_wrapper_passes_kwargs_to_wrapped_fn(self):
    """_ChainCallWrapper forwards keyword arguments to the wrapped function."""
    @Chain().then(lambda v: v).decorator()
    def greet(name, greeting='Hello'):
      return f'{greeting}, {name}!'

    self.assertEqual(greet('World', greeting='Hi'), 'Hi, World!')

  async def test_wrapper_chain_processes_return_value(self):
    """_ChainCallWrapper feeds the function's return value into the chain."""
    @Chain().then(lambda v: v.upper()).decorator()
    def make_str(s):
      return s

    self.assertEqual(make_str('hello'), 'HELLO')

  async def test_wrapper_with_method_on_instance(self):
    """_ChainCallWrapper works as a method descriptor on class instances."""
    class Processor:
      def __init__(self, scale):
        self.scale = scale

      @Chain().then(lambda v: v).decorator()
      def process(self, x):
        return x * self.scale

    p = Processor(3)
    self.assertEqual(p.process(10), 30)

  async def test_wrapper_with_async_wrapped_fn(self):
    """_ChainCallWrapper handles async wrapped functions correctly."""
    @Chain().then(lambda v: v + 100).decorator()
    async def async_compute(x):
      return x * 2

    self.assertEqual(await async_compute(5), 110)  # (5*2) + 100

  async def test_wrapper_exception_propagates(self):
    """Exceptions from the wrapped function propagate through the chain."""
    @Chain().then(lambda v: v).decorator()
    def failing():
      raise TestExc('boom')

    with self.assertRaises(TestExc):
      failing()
