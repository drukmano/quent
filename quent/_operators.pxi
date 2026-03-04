# _operators.pxi — Operator wrappers and execution context
#
# Contains lightweight callable classes used as chain links for chain
# calls. Also defines the per-execution _ExecCtx and the _eval_signal_value
# dispatch function.
#
# Key components:
#   - _ExecCtx: per-execution mutable state (debug info, temp links)
#   - _ChainCallWrapper: chain call delegation (used by the decorator functionality)
#   - _eval_signal_value: evaluates values from control flow signals


@cython.final
@cython.freelist(16)
cdef class _ExecCtx:
  """Per-execution context holding mutable state.

  Created lazily only in debug/error paths so concurrent async
  executions of the same frozen chain receive isolated state.
  """
  pass


@cython.final
cdef class _ChainCallWrapper:
  """Wrapper that invokes a chain's _run with a decorated function as root value."""
  def __init__(self, Chain chain, object fn):
    """Store the chain and the wrapped function."""
    self._chain = chain
    self._fn = fn
  def __call__(self, *args, **kwargs):
    """Run the chain with the wrapped function as root value, forwarding all arguments."""
    try:
      return self._chain._run(self._fn, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None


cdef object _eval_signal_value(object v, tuple args, dict kwargs):
  """Evaluate a value from a control flow signal (_Return/_Break) without allocating a Link.

  This inlines the evaluation logic that would otherwise require creating a temporary Link.
  """
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
