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
    cdef object el, result
    current_value = current_value.__iter__()
    try:
      while True:
        el = current_value.__next__()
        result = self.fn(el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _foreach_async_fn(current_value, self.fn, el, result, lst, self.ignore_result, self.link)
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
  cdef Link link = Link(fn, (), {})
  return Link(_Foreach(fn, ignore_result, link), original_value=link)


async def _foreach_to_async(object current_value, object fn, object el, object result, list lst, bint ignore_result, Link link):
  try:
    while True:
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      el = current_value.__next__()
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
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def _foreach_full_async(object current_value, object fn, bint ignore_result, Link link):
  cdef list lst = []
  cdef object el, result
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
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


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


@cython.final
@cython.freelist(4)
cdef class _Gather:
  def __init__(self, tuple fns, Link link):
    self.fns = fns
    self.link = link

  def __call__(self, object current_value):
    cdef list results = []
    cdef object fn, result
    cdef bint has_coro = False
    for fn in self.fns:
      result = fn(current_value)
      results.append(result)
      if iscoro(result):
        has_coro = True
    if has_coro:
      return _gather_async_fn(results)
    return results


cdef Link gather_(tuple fns):
  cdef Link link = Link(None, (), {})
  return Link(_Gather(fns, link), original_value=link)


async def _gather_to_async(list results):
  cdef int i
  cdef list coros = []
  cdef list indices = []
  for i in range(len(results)):
    if iscoro(results[i]):
      coros.append(results[i])
      indices.append(i)
  cdef list resolved = await _asyncio_gather_fn(*coros)
  for i in range(len(indices)):
    results[indices[i]] = resolved[i]
  return results


cdef object _async_foreach_fn = _foreach_full_async
cdef object _foreach_async_fn = _foreach_to_async
cdef object _filter_async_fn = _filter_to_async
cdef object _async_filter_fn = _filter_full_async
cdef object _gather_async_fn = _gather_to_async
cdef object _asyncio_gather_fn = asyncio.gather
