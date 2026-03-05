# PERF: @cython.final enables direct C dispatch. @cython.freelist(8) pools allocations.
# See PERFORMANCE.md #1, #2.
@cython.final
@cython.freelist(8)
cdef class _Foreach:
  def __init__(self, object fn, bint ignore_result, Link link):
    self.fn = fn
    self.ignore_result = ignore_result
    self.link = link

  def __call__(self, object current_value):
    if hasattr(current_value, '__aiter__'):
      return _async_foreach_fn(current_value, self.fn, self.ignore_result, self.link)
    cdef list lst = []
    # PERF: Initialized to None to avoid __Pyx_RaiseUnboundLocalError checks in except blocks.
    # See PERFORMANCE.md #40.
    cdef object el = None, result = None
    cdef object it = iter(current_value)
    cdef PyObject* _raw
    try:
      while True:
        # PERF: PyIter_Next (C-slot tp_iternext) — returns NULL on exhaustion instead of
        # raising StopIteration, avoiding exception creation/matching per element.
        # See PERFORMANCE.md #27.
        _raw = PyIter_Next(it)
        if _raw is NULL:
          return lst
        el = <object>_raw
        Py_DECREF(el)
        result = self.fn(el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _foreach_async_fn(it, self.fn, el, result, lst, self.ignore_result, self.link)
        if self.ignore_result:
          lst.append(el)
        else:
          lst.append(result)
    except _Break as exc:
      return handle_break_exc(exc, lst)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


cdef Link foreach(object fn, bint ignore_result):
  # PERF: EMPTY_TUPLE/EMPTY_DICT avoid per-call empty container allocation. See PERFORMANCE.md #31.
  cdef Link link = _create_link(fn, EMPTY_TUPLE, EMPTY_DICT)
  return _create_link(_Foreach(fn, ignore_result, link), None, None, False, link)


async def _foreach_to_async(object current_value, object fn, object el, object result, list lst, bint ignore_result, Link link):
  cdef PyObject* _raw
  try:
    while True:
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      _raw = PyIter_Next(current_value)
      if _raw is NULL:
        return lst
      el = <object>_raw
      Py_DECREF(el)
      result = fn(el)
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    # PERF: C-level pointer-to-int cast replaces Python id() call. See PERFORMANCE.md #32.
    exc.__quent_link_temp_args__[<uintptr_t><void*>link] = (el,)
    raise


async def _foreach_full_async(object current_value, object fn, bint ignore_result, Link link):
  cdef list lst = []
  cdef object el = None, result = None
  try:
    async for el in current_value:
      result = fn(el)
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
    return lst
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[<uintptr_t><void*>link] = (el,)
    raise


# PERF: Function aliases — cdef object references avoid module-level global dict lookups.
# See PERFORMANCE.md #19.
cdef object _async_foreach_fn = _foreach_full_async
cdef object _foreach_async_fn = _foreach_to_async
