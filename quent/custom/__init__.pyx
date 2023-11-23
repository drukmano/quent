import sys

from quent.helpers cimport Null, isawaitable
from quent.link cimport Link, evaluate_value


cdef Link build_conditional(object conditional, bint is_custom, bint not_, Link on_true, Link on_false):
  async def if_else_async(object r, object cv):
    r = _if_else(await r, cv)
    if isawaitable(r):
      r = await r
    return r

  def if_else(object cv):
    # more elegant, but slower. we suffer so others won't.
    #return Chain(conditional, cv).then(lambda r: _if_else(r, cv)).run()
    cdef object r = conditional(cv)
    if is_custom and isawaitable(r):
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

  return Link(if_else)


def sync_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result
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


async def async_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result, iterator
  iterator = iterator_getter(*run_args)
  if isawaitable(iterator):
    iterator = await iterator
  if hasattr(iterator, '__aiter__'):
    async for el in iterator:
      if fn is None:
        result = el
      else:
        result = fn(el)
      if isawaitable(result):
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
      if isawaitable(result):
        result = await result
      if ignore_result:
        yield el
      else:
        yield result


cdef class _Generator:
  def __init__(self, _chain_run, _fn, _ignore_result):
    self._chain_run = _chain_run
    self._fn = _fn
    self._ignore_result = _ignore_result
    self._run_args = (Null, (), {})

  def __call__(self, __v=Null, *args, **kwargs):
    # this allows nesting of _Generator within another Chain
    self._run_args = (__v, args, kwargs)
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
    cdef object el, result
    cv = cv.__iter__()
    for el in cv:
      result = fn(el)
      if isawaitable(result):
        return foreach_async(cv, fn, el, result, lst, ignore_result)
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
    return lst
  return Link(_foreach)


async def foreach_async(object cv, object fn, object el, object result, list lst, bint ignore_result):
  result = await result
  if ignore_result:
    lst.append(el)
  else:
    lst.append(result)
  # here we manually call __next__ instead of a for-loop to
  # prevent a call to cv.__iter__() which, depending on the iterator, may
  # produce a new iterator object, instead of continuing where we left
  # off at the sync-foreach version above.
  while True:
    try:
      el = cv.__next__()
      result = fn(el)
      if isawaitable(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
    except StopIteration:
      return lst


async def async_foreach(object cv, object fn, bint ignore_result):
  cdef list lst = []
  cdef object el, result
  async for el in cv.__aiter__():
    result = fn(el)
    if isawaitable(result):
      result = await result
    if ignore_result:
      lst.append(el)
    else:
      lst.append(result)
  return lst


cdef Link with_(Link link, bint ignore_result):
  def with_(object cv):
    if hasattr(cv, '__aenter__'):
      return async_with(link, cv)
    cdef object ctx, result
    cdef bint is_result_awaitable = False
    try:
      ctx = cv.__enter__()
      result = evaluate_value(link, ctx)
      is_result_awaitable = isawaitable(result)
      if is_result_awaitable:
        return with_async(result, cv)
      return result
    finally:
      if not is_result_awaitable:
        cv.__exit__(*sys.exc_info())
  return Link(with_, ignore_result=ignore_result)


async def with_async(object result, object cv):
  try:
    return await result
  finally:
    cv.__exit__(*sys.exc_info())


async def async_with(Link link, object cv):
  cdef object ctx, result
  async with cv as ctx:
    if link.v is Null:
      return ctx
    result = evaluate_value(link, ctx)
    if isawaitable(result):
      return await result
    return result
