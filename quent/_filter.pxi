@cython.final
@cython.freelist(8)
cdef class _Filter:
  def __init__(self, object fn, Link link):
    self.fn = fn
    self.link = link

  def __call__(self, object current_value):
    if hasattr(current_value, '__aiter__'):
      return _async_filter_fn(current_value, self.fn, self.link)
    cdef list lst = []
    cdef object el = None, result = None
    cdef object it = iter(current_value)
    cdef PyObject* _raw
    try:
      while True:
        _raw = PyIter_Next(it)
        if _raw is NULL:
          return lst
        el = <object>_raw
        Py_DECREF(el)
        result = self.fn(el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _filter_async_fn(it, self.fn, el, result, lst, self.link)
        if result:
          lst.append(el)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


cdef Link filter_(object fn):
  cdef Link link = Link(fn, EMPTY_TUPLE, EMPTY_DICT)
  return Link(_Filter(fn, link), original_value=link)


async def _filter_to_async(object current_value, object fn, object el, object result, list lst, Link link):
  cdef PyObject* _raw
  try:
    while True:
      if iscoro(result):
        result = await result
      if result:
        lst.append(el)
      _raw = PyIter_Next(current_value)
      if _raw is NULL:
        return lst
      el = <object>_raw
      Py_DECREF(el)
      result = fn(el)
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[<uintptr_t><void*>link] = (el,)
    raise


async def _filter_full_async(object current_value, object fn, Link link):
  cdef list lst = []
  cdef object el = None, result = None
  try:
    async for el in current_value:
      result = fn(el)
      if iscoro(result):
        result = await result
      if result:
        lst.append(el)
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[<uintptr_t><void*>link] = (el,)
    raise


cdef object _filter_async_fn = _filter_to_async
cdef object _async_filter_fn = _filter_full_async
