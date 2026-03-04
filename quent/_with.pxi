@cython.final
@cython.freelist(4)
cdef class _With:
  def __init__(self, Link link, bint ignore_result, tuple args, dict kwargs):
    self.link = link
    self.ignore_result = ignore_result
    self.args = args
    self.kwargs = kwargs

  def __call__(self, object current_value):
    cdef object outer_value = current_value
    if hasattr(current_value, '__aenter__'):
      return _async_with_fn(self.link, current_value, self.ignore_result)
    cdef object ctx, result
    cdef bint entered = False
    try:
      ctx = current_value.__enter__()
      entered = True
      if not self.args and not self.kwargs:
        self.link.temp_args = (ctx,)
      result = evaluate_value(self.link, ctx)
      if iscoro(result):
        return _with_async_fn(current_value, result, self.link, entered, self.ignore_result, outer_value)
    except BaseException as exc:
      if entered:
        if not current_value.__exit__(type(exc), exc, PyException_GetTraceback(exc)):
          raise
      else:
        raise
    else:
      if entered:
        current_value.__exit__(None, None, None)
      if self.ignore_result:
        return outer_value
      return result


cdef Link with_(object fn, tuple args, dict kwargs, bint ignore_result):
  cdef Link link = Link(fn, args, kwargs)
  return Link(_With(link, ignore_result, args, kwargs), original_value=link, ignore_result=ignore_result)


async def _with_to_async(object current_value, object body_result, Link link, bint entered, bint ignore_result, object outer_value):
  try:
    body_result = await body_result
  except BaseException as exc:
    if entered:
      exit_result = current_value.__exit__(type(exc), exc, PyException_GetTraceback(exc))
      if iscoro(exit_result):
        exit_result = await exit_result
      if not exit_result:
        raise
    else:
      raise
  else:
    exit_result = current_value.__exit__(None, None, None)
    if iscoro(exit_result):
      await exit_result
    if ignore_result:
      return outer_value
    return body_result


async def _with_full_async(Link link, object current_value, bint ignore_result = False):
  cdef object ctx, result
  cdef object outer_value = current_value
  async with current_value as ctx:
    if not link.args and not link.kwargs:
      link.temp_args = (ctx,)
    try:
      result = evaluate_value(link, ctx)
      if iscoro(result):
        result = await result
    except BaseException as exc:
      if not hasattr(exc, '__quent_link_temp_args__'):
        exc.__quent_link_temp_args__ = {}
      exc.__quent_link_temp_args__[<uintptr_t><void*>link] = (ctx,)
      raise
    if ignore_result:
      return outer_value
    return result


cdef object _async_with_fn = _with_full_async
cdef object _with_async_fn = _with_to_async
