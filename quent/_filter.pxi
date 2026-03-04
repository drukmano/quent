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
    cdef object el, result
    current_value = current_value.__iter__()
    try:
      while True:
        el = current_value.__next__()
        result = self.fn(el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _filter_async_fn(current_value, self.fn, el, result, lst, self.link)
        if result:
          lst.append(el)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


cdef Link filter_(object fn):
  cdef Link link = Link(fn, (), {})
  return Link(_Filter(fn, link), original_value=link)


async def _filter_to_async(object current_value, object fn, object el, object result, list lst, Link link):
  try:
    while True:
      if iscoro(result):
        result = await result
      if result:
        lst.append(el)
      el = current_value.__next__()
      result = fn(el)
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def _filter_full_async(object current_value, object fn, Link link):
  cdef list lst = []
  cdef object el, result
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
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


cdef object _filter_async_fn = _filter_to_async
cdef object _async_filter_fn = _filter_full_async
