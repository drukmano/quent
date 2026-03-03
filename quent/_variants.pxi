# _variants.pxi — Chain subclasses and supporting wrappers
#
# Defines the Chain variants (Cascade) and related utility types
# (_DescriptorWrapper, _FrozenChain, run). These are thin extensions
# of the base Chain class, each adding a specific behavioral twist.
#
# Key components:
#   - Cascade: every link receives the root value instead of the previous result
#   - _DescriptorWrapper: makes chain decorators work as instance methods
#   - _FrozenChain: immutable snapshot for safe repeated execution
#   - run: pipe syntax terminator (Chain(f1) | f2 | run())


@cython.final
cdef class Cascade(Chain):
  """Like Chain, but each operation receives the root value instead of the previous result."""

  # noinspection PyMissingConstructor
  def __init__(self, object __v = Null, *args, **kwargs):
    """Create a new Cascade with an optional root value."""
    self.init(__v, args, kwargs, True)


@cython.final
@cython.freelist(8)
cdef class _DescriptorWrapper:
  """Descriptor wrapper that makes a callable behave as an instance method when accessed on an object."""

  def __init__(self, object fn):
    """Wrap the given callable and initialize an empty __dict__ for functools.update_wrapper."""
    self._fn = fn
    self.__dict__ = {}

  def __call__(self, *args, **kwargs):
    """Delegate the call to the wrapped function."""
    __tracebackhide__ = True
    return self._fn(*args, **kwargs)

  def __get__(self, obj, objtype):
    """Return self for class access, or a bound method for instance access."""
    if obj is None:
      return self
    return types.MethodType(self, obj)


@cython.final
@cython.freelist(4)
cdef class _FrozenChain:
  """Immutable snapshot of a chain that can be executed repeatedly without mutation."""
  def decorator(self):
    """Return a decorator that wraps functions to run through this frozen chain."""
    cdef object _chain_run = self._chain_run
    def _decorator(fn):
      """Wrap fn so that calling it executes the frozen chain with fn as root value."""
      result = _DescriptorWrapper(_ChainCallWrapper(_chain_run, fn))
      try:
        functools.update_wrapper(result, fn)
      except AttributeError:
        pass
      return result
    return _decorator

  def __init__(self, object _chain_run):
    """Initialize with a reference to the chain's _run method."""
    self._chain_run = _chain_run

  def run(self, object __v = Null, *args, **kwargs):
    """Execute the frozen chain with an optional root value override."""
    __tracebackhide__ = True
    return self._chain_run(__v, args, kwargs, False)

  def __call__(self, object __v = Null, *args, **kwargs):
    """Shorthand for run(). Execute the frozen chain."""
    __tracebackhide__ = True
    return self._chain_run(__v, args, kwargs, False)


@cython.final
@cython.freelist(4)
cdef class run:
  """
    A replacement for `Chain.run()` when using pipe syntax

      Chain(f1).then(f2).run() == Chain(f1) | f2 | run()
      Chain().then(f2).run(f1) == Chain() | f2 | run(f1)
  """
  def __init__(self, object __v = Null, *args, **kwargs):
    """Store the root value and arguments to be passed when the chain is executed via pipe."""
    self.root_value = __v
    self.args = args
    self.kwargs = kwargs
