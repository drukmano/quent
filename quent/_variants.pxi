# _variants.pxi — Chain supporting wrappers
#
# Defines utility types that extend or wrap Chain execution:
# (_DescriptorWrapper).
#
# Key components:
#   - _DescriptorWrapper: makes chain decorators work as instance methods


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
    return self._fn(*args, **kwargs)

  def __get__(self, obj, objtype):
    """Return self for class access, or a bound method for instance access."""
    if obj is None:
      return self
    return types.MethodType(self, obj)
