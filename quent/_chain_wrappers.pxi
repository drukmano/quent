# PERF: @cython.final enables direct C dispatch. See PERFORMANCE.md #1.
@cython.final
cdef class _ChainCallWrapper:
  def __init__(self, Chain chain, object fn):
    self._chain = chain
    self._fn = fn
  def __call__(self, *args, **kwargs):
    try:
      return self._chain._run(self._fn, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None


# PERF: @cython.final enables direct C dispatch. @cython.freelist(8) pools allocations.
# See PERFORMANCE.md #1, #2.
@cython.final
@cython.freelist(8)
cdef class _DescriptorWrapper:
  def __init__(self, object fn):
    self._fn = fn
    self.__dict__ = {}

  def __call__(self, *args, **kwargs):
    return self._fn(*args, **kwargs)

  def __get__(self, obj, objtype):
    if obj is None:
      return self
    # PERF: Cached _MethodType avoids types.MethodType attribute lookup. See PERFORMANCE.md #28.
    return _MethodType(self, obj)
