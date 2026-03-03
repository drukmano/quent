# _operators.pxi — Operator wrappers and execution context
#
# Contains lightweight callable classes used as chain links for sleeping,
# thread delegation, and frozen chain calls. Also defines the per-execution
# _ExecCtx and the _eval_signal_value dispatch function.
#
# Key components:
#   - _ExecCtx: per-execution mutable state (debug info, temp links)
#   - _Sleep: sync/async sleep
#   - _ToThread: runs a function in a separate thread
#   - _ChainCallWrapper: frozen chain call delegation
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
@cython.freelist(4)
@cython.no_gc
cdef class _Sleep:
  """Callable that sleeps for a given duration, using asyncio.sleep if an event loop is running."""
  def __init__(self, float delay):
    """Store the sleep duration in seconds."""
    self.delay = delay
  def __call__(self, object current_value):
    """Sleep for the stored duration. Uses asyncio.sleep in async contexts, time.sleep otherwise."""
    if _asyncio_get_running_loop_internal() is not None:
      return _asyncio_sleep(self.delay)
    time.sleep(self.delay)


@cython.final
cdef class _ChainCallWrapper:
  """Wrapper that invokes a frozen chain's _run with a decorated function as root value."""
  def __init__(self, object _chain_run, object fn):
    """Store the chain's _run method and the wrapped function."""
    self._chain_run = _chain_run
    self._fn = fn
  def __call__(self, *args, **kwargs):
    """Run the chain with the wrapped function as root value, forwarding all arguments."""
    __tracebackhide__ = True
    return self._chain_run(self._fn, args, kwargs, False)


@cython.final
@cython.freelist(4)
cdef class _ToThread:
  """Callable that runs a function in a separate thread via asyncio.to_thread."""

  def __init__(self, object fn):
    """Store the function to run in a thread."""
    self.fn = fn

  def __call__(self, object current_value):
    """Run self.fn(current_value) in a thread pool, or call directly if no event loop."""
    if _asyncio_get_running_loop_internal() is not None:
      if current_value is Null:
        return asyncio.to_thread(self.fn)
      return asyncio.to_thread(self.fn, current_value)
    if current_value is Null:
      return self.fn()
    return self.fn(current_value)


cdef object _eval_signal_value(object v, tuple args, dict kwargs):
  """Evaluate a value from a control flow signal (_Return/_Break) without allocating a Link.

  This inlines the evaluation logic that would otherwise require creating a temporary Link.
  The value is always treated as allow_literal=True (it came from a signal).
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
  elif callable(v):
    # EVAL_CALL_WITH_CURRENT_VALUE with Null -> call without args
    return v()
  else:
    # EVAL_RETURN_AS_IS equivalent
    return v
