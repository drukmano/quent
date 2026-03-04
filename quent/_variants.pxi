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
    return types.MethodType(self, obj)
