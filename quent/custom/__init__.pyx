import sys
from quent.quent cimport Link, evaluate_value, Null, iscoro, QuentException


cdef class _InternalQuentException_Custom(_InternalQuentException):
  def __init__(self, object __v, tuple args, dict kwargs):
    self._v = __v
    self.args = args
    self.kwargs = kwargs


cdef object handle_break_exc(_Break exc, object nv):
  if exc._v is Null:
    return nv
  return evaluate_value(Link.__new__(Link, exc._v, exc.args, exc.kwargs, True), Null)


cdef Link build_conditional(object conditional, bint is_custom, bint not_, Link on_true, Link on_false):
  async def if_else_async(object r, object cv):
    r = _if_else(await r, cv)
    if iscoro(r):
      r = await r
    return r

  def if_else(object cv):
    # more elegant, but slower. we suffer so others won't.
    #return Chain(conditional, cv).then(lambda r: _if_else(r, cv)).run()
    cdef object r = conditional(cv)
    if is_custom and iscoro(r):
      return if_else_async(r, cv)
    return _if_else(r, cv)

  def _if_else(object r, object cv):
    if not_:
      r = not r
    if r:
      return evaluate_value(on_true, cv)
    elif on_false is not None:
      return evaluate_value(on_false, cv)
    return cv

  return Link.__new__(Link, if_else)


cdef Link while_true(object fn, tuple args, dict kwargs):
  cdef Link link = Link.__new__(Link, fn, args, kwargs)
  def _while_true(object cv = Null):
    cdef object result, exc
    try:
      while True:
        result = evaluate_value(link, cv)
        if iscoro(result):
          return while_true_async(cv, link, result)
    except _Break as exc:
      return handle_break_exc(exc, cv)
  return Link.__new__(Link, _while_true)


async def while_true_async(object cv, Link link, object result):
  cdef object exc
  try:
    while True:
      if iscoro(result):
        await result
      result = evaluate_value(link, cv)
  except _Break as exc:
    result = handle_break_exc(exc, cv)
    if iscoro(result):
      return await result
    return result


def sync_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result, exc
  try:
    for el in iterator_getter(*run_args):
      if fn is None:
        yield el
      else:
        result = fn(el)
        # we ignore the case where `result` is awaitable - it's impossible to deal with.
        if ignore_result:
          yield el
        else:
          yield result
  except _Break as exc:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


async def async_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result, iterator, exc
  iterator = iterator_getter(*run_args)
  if iscoro(iterator):
    iterator = await iterator
  try:
    if hasattr(iterator, '__aiter__'):
      async for el in iterator:
        if fn is None:
          result = el
        else:
          result = fn(el)
          if iscoro(result):
            result = await result
        if ignore_result:
          yield el
        else:
          yield result
    else:
      for el in iterator:
        if fn is None:
          result = el
        else:
          result = fn(el)
          if iscoro(result):
            result = await result
        if ignore_result:
          yield el
        else:
          yield result
  except _Break as exc:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


cdef class _Generator:
  def __init__(self, object _chain_run, object _fn, bint _ignore_result):
    self._chain_run = _chain_run
    self._fn = _fn
    self._ignore_result = _ignore_result
    self._run_args = (Null, (), {}, False)

  def __call__(self, object __v = Null, *args, **kwargs):
    # this allows nesting of _Generator within another Chain
    self._run_args = (__v, args, kwargs, False)
    return self

  def __iter__(self):
    return sync_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __aiter__(self):
    return async_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)


cdef Link foreach(object fn, bint ignore_result):
  def _foreach(object cv):
    if hasattr(cv, '__aiter__'):
      return async_foreach(cv, fn, ignore_result)
    cdef list lst = []
    cdef object el, result, exc
    cv = cv.__iter__()
    try:
      for el in cv:
        result = fn(el)
        if iscoro(result):
          return foreach_async(cv, fn, el, result, lst, ignore_result)
        if ignore_result:
          lst.append(el)
        else:
          lst.append(result)
      return lst
    except _Break as exc:
      return handle_break_exc(exc, lst)
  return Link.__new__(Link, _foreach)


async def foreach_async(object cv, object fn, object el, object result, list lst, bint ignore_result):
  cdef object exc
  # here we manually call __next__ instead of a for-loop to
  # prevent a call to cv.__iter__() which, depending on the iterator, may
  # produce a new iterator object, instead of continuing where we left
  # off at the sync-foreach version above.
  try:
    while True:
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      el = cv.__next__()
      result = fn(el)
  except StopIteration:
    return lst
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result


async def async_foreach(object cv, object fn, bint ignore_result):
  cdef list lst = []
  cdef object el, result, exc
  try:
    async for el in cv.__aiter__():
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


cdef Link with_(Link link, bint ignore_result):
  def with_(object cv):
    if hasattr(cv, '__aenter__'):
      return async_with(link, cv)
    cdef object ctx, result
    cdef bint is_result_awaitable = False
    try:
      ctx = cv.__enter__()
      result = evaluate_value(link, ctx)
      is_result_awaitable = iscoro(result)
      if is_result_awaitable:
        return with_async(result, cv)
      return result
    finally:
      if not is_result_awaitable:
        cv.__exit__(*sys.exc_info())
  return Link.__new__(Link, with_, ignore_result=ignore_result)


async def with_async(object result, object cv):
  try:
    return await result
  finally:
    cv.__exit__(*sys.exc_info())


async def async_with(Link link, object cv):
  cdef object ctx, result
  async with cv as ctx:
    result = evaluate_value(link, ctx)
    if iscoro(result):
      return await result
    return result
