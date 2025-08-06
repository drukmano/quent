# _variants.pxi — Chain subclasses and supporting wrappers
#
# Defines the Chain variants (Cascade, ChainAttr, CascadeAttr) and related
# utility types (_DescriptorWrapper, _FrozenChain, run). These are thin
# extensions of the base Chain class, each adding a specific behavioral twist.
#
# Key components:
#   - Cascade: every link receives the root value instead of the previous result
#   - ChainAttr: supports attribute access on the current value via __getattr__
#   - CascadeAttr: combines Cascade and ChainAttr behavior
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


cdef class ChainAttr(Chain):
  """Chain variant that supports attribute access on the current value via __getattr__."""
  def __init__(self, object __v = Null, *args, **kwargs):
    """Create a new ChainAttr with an optional root value."""
    self.init(__v, args, kwargs, False)

  def __getattr__(self, str name):
    """Record an attribute access to be resolved against the current value during execution."""
    self.finalize()
    self.current_attr = name
    return self

  def __call__(self, *args, **kwargs):
    """Call the pending attribute as a method, or run the chain if no attribute is pending."""
    cdef str attr = self.current_attr
    if attr is None:
      # much slower than directly calling `._run()`, but we have no choice since
      # we wish support arbitrary __call__ invocations on attributes.
      # avoid running a chain this way. opt to use `.run()` instead.
      return self.run(*args, **kwargs)
    else:
      self.current_attr = None
      self._then(Link(attr, args, kwargs, is_attr=True, is_fattr=True))
      return self


# cannot have multiple inheritance in Cython.
@cython.final
cdef class CascadeAttr(ChainAttr):
  """Cascade variant that supports attribute access on the current value via __getattr__."""
  # noinspection PyMissingConstructor
  def __init__(self, object __v = Null, *args, **kwargs):
    """Create a new CascadeAttr with an optional root value."""
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
