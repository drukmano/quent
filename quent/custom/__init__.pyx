import sys

import cython

from quent._internal import __QUENT_INTERNAL__
from quent.quent cimport Link, evaluate_value, Null, iscoro, QuentException, Chain


cdef class _InternalQuentException_Custom(_InternalQuentException):
  def __init__(self, object __v, tuple args, dict kwargs):
    self._v = __v
    self.args = args
    self.kwargs = kwargs


cdef object handle_break_exc(_Break exc, object nv):
  if exc._v is Null:
    return nv
  return evaluate_value(Link(exc._v, exc.args, exc.kwargs, True), Null)


@cython.final
@cython.freelist(8)
cdef class _Conditional:
  cdef object cv
  cdef bint result

  def __init__(self, object cv, bint result):
    self.cv = cv
    self.result = result

  def __repr__(self):
    return repr(self.result)


@cython.final
@cython.freelist(8)
cdef class _If:
  cdef _Conditional cond

  def __init__(self, _Conditional cond):
    self.cond = cond

  def __repr__(self):
    if self.cond.result:
      return repr(self.cond.cv)
    else:
      return '<None>'


async def _await_if_cond_cv(_If if_cond):
  if_cond.cond.cv = await if_cond.cond.cv
  return if_cond


async def _await_cond_result(object cv, object result):
  return _Conditional(cv, bool(await result))


cdef void build_conditional(Chain chain, Link conditional, bint is_custom, bint not_, Link on_true, Link on_false):
  def else_(_If if_cond):
    if not if_cond.cond.result:
      return evaluate_value(on_false, if_cond.cond.cv)
    else:
      return if_cond.cond.cv

  def if_(_Conditional cond):
    cdef object result
    cdef _If if_cond
    if not_:
      cond.result = not cond.result
    if cond.result:
      result = evaluate_value(on_true, cond.cv)

    if on_false is None:
      if cond.result:
        return result
      else:
        return cond.cv
    else:
      if_cond = _If(cond)
      if cond.result:
        cond.cv = result
        if iscoro(result):
          return _await_if_cond_cv(if_cond)
      return if_cond

  def direct_if(object cv):
    return if_(_Conditional(cv, bool(cv)))

  def conditional_fn(object cv):
    cdef object result = evaluate_value(conditional, cv)
    # non-custom conditionals (i.e. predefined by quent like `eq`, `is_`, etc.) are never coroutines.
    if is_custom and iscoro(result):
      return _await_cond_result(cv, result)
    return _Conditional(cv, bool(result))

  if conditional is None:
    chain._then(Link(direct_if, ogv=on_true))
  else:
    chain._then(Link(conditional_fn, ogv=conditional))
    chain._then(Link(if_, ogv=on_true))
  if on_false is not None:
    chain._then(Link(else_, ogv=on_false))


cdef Link while_true(object fn, tuple args, dict kwargs):
  cdef Link link = Link(fn, args, kwargs, fn_name='while_true')
  def _while_true(object cv = Null):
    cdef object result, exc
    try:
      if not args and not kwargs:
        link.temp_args = (cv,)
      while True:
        result = evaluate_value(link, cv)
        if iscoro(result):
          return while_true_async(cv, link, result)
    except _Break as exc:
      return handle_break_exc(exc, cv)
  return Link(_while_true, ogv=link)


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
  cdef bint is_aiter
  iterator = iterator_getter(*run_args)
  if iscoro(iterator):
    iterator = await iterator
  try:
    try:
      iterator.__aiter__
      is_aiter = True
    except AttributeError:
      is_aiter = False
    if is_aiter:
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


@cython.final
@cython.freelist(4)
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

  def __repr__(self):
    # TODO
    raise NotImplementedError


cdef Link foreach(object fn, bint ignore_result):
  cdef Link link = Link(fn, (), {}, fn_name='foreach')
  def _foreach(object cv):
    try:
      if cv.__aiter__:
        return async_foreach(cv, fn, ignore_result, link)
    except AttributeError:
      pass
    cdef list lst = []
    cdef object el, result, exc
    # we use the "raw" iteration syntax to be able to seamlessly
    # transfer flow to the async version if `fn` is a coroutine.
    cv = cv.__iter__()
    try:
      while True:
        el = cv.__next__()
        # set the current element as the function argument to be able to
        # format it with the correct element if an exception is raised.
        link.temp_args = (el,)
        result = fn(el)
        if iscoro(result):
          return foreach_async(cv, fn, el, result, lst, ignore_result, link)
        if ignore_result:
          lst.append(el)
        else:
          lst.append(result)
    except StopIteration:
      return lst
    except _Break as exc:
      return handle_break_exc(exc, lst)
  return Link(_foreach, ogv=link)


async def foreach_async(object cv, object fn, object el, object result, list lst, bint ignore_result, Link link):
  cdef object exc
  try:
    while True:
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      el = cv.__next__()
      link.temp_args = (el,)
      result = fn(el)
  except StopIteration:
    return lst
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result


async def async_foreach(object cv, object fn, bint ignore_result, Link link):
  cdef list lst = []
  cdef object el, result, exc
  try:
    async for el in cv:
      link.temp_args = (el,)
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


cdef Link with_(object fn, tuple args, dict kwargs, bint ignore_result):
  cdef Link link = Link(fn, args, kwargs, fn_name='with_')
  def with_(object cv):
    try:
      if cv.__aenter__:
        return async_with(link, cv)
    except AttributeError:
      pass
    cdef object ctx, result
    cdef bint ignore_finally = False
    try:
      ctx = cv.__enter__()
      if not args and not kwargs:
        link.temp_args = (ctx,)
      result = evaluate_value(link, ctx)
      if iscoro(result):
        ignore_finally = True
        return with_async(result, cv, link)
      return result
    finally:
      if not ignore_finally:
        cv.__exit__(*sys.exc_info())
  return Link(with_, ogv=link, ignore_result=ignore_result)


async def with_async(object result, object cv, Link link):
  try:
    return await result
  finally:
    cv.__exit__(*sys.exc_info())


async def async_with(Link link, object cv):
  cdef object ctx, result
  async with cv as ctx:
    if not link.args and not link.kwargs:
      link.temp_args = (ctx,)
    result = evaluate_value(link, ctx)
    if iscoro(result):
      return await result
    return result
