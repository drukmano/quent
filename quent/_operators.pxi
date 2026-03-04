@cython.final
@cython.freelist(16)
cdef class _ExecCtx:
  pass


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


cdef object _eval_signal_value(object v, tuple args, dict kwargs):
  if args:
    if args[0] is ...:
      # EVAL_CALL_WITHOUT_ARGS equivalent
      return v()
    else:
      # EVAL_CALL_WITH_EXPLICIT_ARGS equivalent
      if kwargs is None:
        kwargs = EMPTY_DICT
      if kwargs is EMPTY_DICT:
        return v(*args)
      return v(*args, **kwargs)
  elif kwargs:
    if args is None:
      args = EMPTY_TUPLE
    return v(*args, **kwargs)
  elif PyCallable_Check(v):
    # EVAL_CALL_WITH_CURRENT_VALUE with Null -> call without args
    return v()
  else:
    # EVAL_RETURN_AS_IS equivalent
    return v
