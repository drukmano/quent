# _operators.pxi — Operator wrappers and execution context
#
# Contains lightweight callable classes used as chain links for comparisons,
# type checks, exception raising, sleeping, and logical negation. Also defines
# the per-execution _ExecCtx and the _eval_signal_value dispatch function.
#
# Key components:
#   - _ExecCtx: per-execution mutable state (debug info, temp links)
#   - _Raiser: raises a stored exception
#   - _Comparator: binary comparison operators (==, !=, is, in, etc.)
#   - _Or: falsy-fallback operator
#   - _IsInstance: isinstance check
#   - _Sleep: sync/async sleep
#   - _ChainCallWrapper: frozen chain call delegation
#   - _Not: logical negation
#   - _eval_signal_value: evaluates values from control flow signals


@cython.final
@cython.freelist(16)
cdef class _ExecCtx:
  """Per-execution context holding mutable state.

  Created lazily only in debug/error paths so concurrent async
  executions of the same frozen chain receive isolated state.
  """
  pass


cdef:
  int _OP_EQ = 0
  int _OP_NEQ = 1
  int _OP_IS = 2
  int _OP_IS_NOT = 3
  int _OP_IN = 4
  int _OP_NOT_IN = 5


@cython.final
@cython.freelist(8)
cdef class _Raiser:
  """Callable that unconditionally raises a stored exception when invoked."""
  def __init__(self, object exc):
    """Store the exception to raise."""
    self.exc = exc
  def __call__(self, object current_value):
    """Raise the stored exception, ignoring the current value."""
    raise self.exc


@cython.final
@cython.freelist(16)
cdef class _Comparator:
  """Callable that compares the current value against a stored value using a given operator."""
  def __init__(self, object value, int op):
    """Store the comparison value and operator code."""
    self.value = value
    self.op = op
  def __call__(self, object current_value):
    """Evaluate the comparison and return a boolean result."""
    if self.op == _OP_EQ: return current_value == self.value
    elif self.op == _OP_NEQ: return current_value != self.value
    elif self.op == _OP_IS: return current_value is self.value
    elif self.op == _OP_IS_NOT: return current_value is not self.value
    elif self.op == _OP_IN: return current_value in self.value
    elif self.op == _OP_NOT_IN: return current_value not in self.value
    else: raise QuentException(f'Unknown comparator operator code: {self.op}')


@cython.final
@cython.freelist(8)
cdef class _Or:
  """Callable that returns the current value or a fallback if the current value is falsy."""
  def __init__(self, object value):
    """Store the fallback value."""
    self.value = value
  def __call__(self, object current_value):
    """Return current_value if truthy, otherwise the stored fallback value."""
    return current_value or self.value


@cython.final
@cython.freelist(8)
@cython.no_gc
cdef class _IsInstance:
  """Callable that checks whether the current value is an instance of stored types."""
  def __init__(self, tuple types):
    """Store the types to check against."""
    self.types = types
  def __call__(self, object current_value):
    """Return True if current_value is an instance of the stored types."""
    return isinstance(current_value, self.types)


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
@cython.no_gc
cdef class _Not:
  """Callable that returns the logical negation of the current value."""
  def __call__(self, object current_value):
    """Return not current_value."""
    return not current_value

cdef _Not _not_instance = _Not()


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
    if kwargs is EMPTY_DICT:
      return v(*args)
    return v(*args, **kwargs)
  elif callable(v):
    # EVAL_CALL_WITH_CURRENT_VALUE with Null -> call without args
    return v()
  else:
    # EVAL_RETURN_AS_IS equivalent
    return v
