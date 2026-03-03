# _control_flow.pxi — Control flow signals, context managers, generators
#
# Implements the control flow constructs that extend Chain beyond simple linear
# pipelines: context manager (with) blocks and lazy generators. Also defines
# the internal exception types (_Return, _Break) used as control flow signals.
#
# Key components:
#   - _InternalQuentException / _Return / _Break: exception-based control flow signals
#   - _With: context manager (sync/async) execution
#   - _Generator: lazy sync/async iteration over chain results


# --- Control flow signals ---


cdef class _InternalQuentException(Exception):
  """Base exception for internal quent control flow signals."""

  def __init__(self, object __v, tuple args, dict kwargs):
    """Initialize with a value and its call arguments."""
    self.value = __v
    self.args_ = args
    self.kwargs_ = kwargs

  def __repr__(self):
    """Return string representation."""
    return f'<{type(self).__name__}>'


cdef class _Return(_InternalQuentException):
  """Exception to exit a nested chain early, carrying a return value."""
  pass


cdef class _Break(_InternalQuentException):
  """Exception signal to break out of an iteration early, optionally carrying a value."""
  pass


cdef object handle_break_exc(_Break exc, object fallback):
  """Resolve a break exception to its final value, falling back to fallback if no value was provided."""
  if exc.value is Null:
    return fallback
  return _eval_signal_value(exc.value, exc.args_, exc.kwargs_)


cdef object handle_return_exc(_Return exc, bint propagate):
  """Resolve a return exception to its final value, or re-raise if propagating to a parent chain."""
  if propagate:
    raise exc
  if exc.value is Null:
    return None
  return _eval_signal_value(exc.value, exc.args_, exc.kwargs_)


# --- Context managers ---

@cython.final
@cython.freelist(4)
cdef class _With:
  """Callable that executes a link as the body of a context manager."""

  def __init__(self, Link link, bint ignore_result, tuple args, dict kwargs):
    """Initialize with the body link, result-ignore flag, and arguments."""
    self.link = link
    self.ignore_result = ignore_result
    self.args = args
    self.kwargs = kwargs

  def __call__(self, object current_value):
    """Enter the context manager current_value and evaluate the body link."""
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
        if not current_value.__exit__(type(exc), exc, exc.__traceback__):
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
  """Evaluate a sync context manager, delegating to _with_to_async if the body
  is a coroutine, or to _with_full_async for native async context managers.

  Implements PEP 343 semantics:
  - __exit__ is only called if __enter__ succeeded (entered flag)
  - __exit__ return value is checked for exception suppression
  - Exceptions are captured explicitly rather than via sys.exc_info()
  """
  cdef Link link = Link(fn, args, kwargs, fn_name='with_')
  return Link(_With(link, ignore_result, args, kwargs), original_value=link, ignore_result=ignore_result)


async def _with_to_async(object current_value, object body_result, Link link, bint entered, bint ignore_result, object outer_value):
  """Async continuation when a sync context manager body returns a coroutine."""
  try:
    body_result = await body_result
  except BaseException as exc:
    if entered:
      exit_result = current_value.__exit__(type(exc), exc, exc.__traceback__)
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
  """Fully async context manager handler using 'async with'."""
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
      exc.__quent_link_temp_args__[id(link)] = (ctx,)
      raise
    if ignore_result:
      return outer_value
    return result


# --- Generators ---

def sync_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  """Synchronous generator that yields chain results, applying fn to each element."""
  cdef object el, result
  try:
    for el in iterator_getter(*run_args):
      if fn is None:
        yield el
      else:
        result = fn(el)
        if ignore_result:
          yield el
        else:
          yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


async def async_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  """Asynchronous generator that yields chain results, handling both sync and async iterables."""
  cdef object el, result, iterator
  cdef bint is_aiter
  iterator = iterator_getter(*run_args)
  if iscoro(iterator):
    iterator = await iterator
  is_aiter = hasattr(iterator, '__aiter__')
  try:
    if is_aiter:
      async for el in iterator:
        if fn is None:
          yield el
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
          yield el
        else:
          result = fn(el)
          if iscoro(result):
            result = await result
          if ignore_result:
            yield el
          else:
            yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


@cython.final
@cython.freelist(4)
cdef class _Generator:
  """Lazy generator wrapper that supports both sync and async iteration over chain results."""

  def __init__(self, object _chain_run, object _fn, bint _ignore_result):
    """Initialize with the chain's run method, mapping function, and result-ignore flag."""
    self._chain_run = _chain_run
    self._fn = _fn
    self._ignore_result = _ignore_result
    self._run_args = (Null, (), {}, False)

  def __call__(self, object __v = Null, *args, **kwargs):
    """Set run arguments, allowing nesting of _Generator within another Chain."""
    cdef _Generator g = _Generator.__new__(_Generator)
    g._chain_run = self._chain_run
    g._fn = self._fn
    g._ignore_result = self._ignore_result
    g._run_args = (__v, args, kwargs, False)
    return g

  def __iter__(self):
    """Return a synchronous generator over the chain results."""
    return _sync_generator_fn(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __aiter__(self):
    """Return an asynchronous generator over the chain results."""
    return _async_generator_fn(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __repr__(self):
    """Return string representation."""
    return '<_Generator>'


# --- Function aliases ---
# Cython cdef objects used as first-class callables to avoid Python-level function
# lookup overhead in hot paths. Each alias points to the corresponding async function.

cdef object _sync_generator_fn = sync_generator
cdef object _async_generator_fn = async_generator
cdef object _async_with_fn = _with_full_async
cdef object _with_async_fn = _with_to_async
