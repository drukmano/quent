# _control_flow.pxi — Control flow signals, conditionals, loops, context managers, generators
#
# Implements the control flow constructs that extend Chain beyond simple linear
# pipelines: if/else branching, while loops, context manager (with) blocks,
# and lazy generators. Also defines the internal exception types (_Return, _Break)
# used as control flow signals.
#
# Key components:
#   - _InternalQuentException / _Return / _Break: exception-based control flow signals
#   - _Conditional / _If / _IfEvaluator / _ElseEvaluator: branching infrastructure
#   - build_conditional: assembles conditional link structures
#   - _WhileTrue: infinite loop with break support
#   - _With: context manager (sync/async) execution
#   - _Generator: lazy sync/async iteration over chain results


# --- Control flow signals ---

_MISSING = object()


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
  """Exception to break out of a while_true loop, optionally carrying a value."""
  pass


cdef object handle_break_exc(_Break exc, object fallback):
  """Resolve a break exception to its final value, falling back to fallback if no value was provided."""
  if exc.value is Null:
    return fallback
  return _eval_signal_value(exc.value, exc.args_, exc.kwargs_)


# --- Conditional evaluation ---

@cython.final
@cython.freelist(8)
cdef class _Conditional:
  """Wrapper holding a chain value paired with its boolean condition result."""

  def __init__(self, object current_value, bint result):
    """Initialize with the current value and its evaluated condition."""
    self.current_value = current_value
    self.result = result

  def __repr__(self):
    """Return string representation of the condition result."""
    return repr(self.result)


cdef inline _Conditional _make_conditional(object current_value, bint result):
  """Fast-path factory for _Conditional, bypassing __init__ via __new__."""
  cdef _Conditional cond = _Conditional.__new__(_Conditional)
  cond.current_value = current_value
  cond.result = result
  return cond


@cython.final
@cython.freelist(8)
cdef class _If:
  """Wrapper holding a _Conditional for if/else branch evaluation."""

  def __init__(self, _Conditional cond):
    """Initialize with the evaluated conditional."""
    self.cond = cond

  def __repr__(self):
    """Return the conditional value repr if true, otherwise '<None>'."""
    if self.cond.result:
      return repr(self.cond.current_value)
    else:
      return '<None>'


async def _await_if_cond_value(_If if_cond):
  """Await the conditional value inside an _If when it is a coroutine."""
  if_cond.cond.current_value = await if_cond.cond.current_value
  return if_cond


async def _await_cond_result(object current_value, object result):
  """Await a condition result coroutine and wrap it as a _Conditional."""
  return _make_conditional(current_value, <bint>(await result))


@cython.final
@cython.freelist(8)
cdef class _IfEvaluator:
  """Evaluates if/else branches based on a _Conditional's result."""

  def __init__(self, bint not_, Link on_true, bint has_on_false):
    """Initialize with negation flag, true-branch link, and false-branch presence."""
    self.not_ = not_
    self.on_true = on_true
    self.has_on_false = has_on_false

  def __call__(self, _Conditional cond):
    """Evaluate the true branch if condition holds, returning _If when else exists."""
    cdef object result
    cdef _If if_cond
    if self.not_:
      cond.result = not cond.result
    if cond.result:
      result = evaluate_value(self.on_true, cond.current_value)
    if not self.has_on_false:
      if cond.result:
        return result
      else:
        return cond.current_value
    else:
      if_cond = _If(cond)
      if cond.result:
        cond.current_value = result
        if iscoro(result):
          return _await_if_cond_value_fn(if_cond)
      return if_cond


@cython.final
@cython.freelist(8)
cdef class _DirectIf:
  """Evaluates truthiness of the current value directly as a condition."""

  def __init__(self, _IfEvaluator if_eval):
    """Initialize with the _IfEvaluator to delegate to."""
    self.if_eval = if_eval

  def __call__(self, object current_value):
    """Test current_value for truthiness and pass the resulting _Conditional to the evaluator."""
    return self.if_eval(_make_conditional(current_value, <bint>current_value))


@cython.final
@cython.freelist(8)
cdef class _ConditionalFn:
  """Evaluates a custom condition function against the current value."""

  def __init__(self, Link conditional, bint is_custom):
    """Initialize with the condition link and whether it is user-provided."""
    self.conditional = conditional
    self.is_custom = is_custom

  def __call__(self, object current_value):
    """Evaluate the condition against current_value, returning a _Conditional or async wrapper."""
    cdef object result = evaluate_value(self.conditional, current_value)
    if self.is_custom and iscoro(result):
      return _await_cond_result_fn(current_value, result)
    return _make_conditional(current_value, <bint>result)


@cython.final
@cython.freelist(8)
cdef class _ElseEvaluator:
  """Evaluates the else branch of a conditional when the condition is false."""

  def __init__(self, Link on_false):
    """Initialize with the false-branch link."""
    self.on_false = on_false

  def __call__(self, _If if_cond):
    """Evaluate the false branch if condition failed, otherwise pass through the value."""
    if not if_cond.cond.result:
      return evaluate_value(self.on_false, if_cond.cond.current_value)
    else:
      return if_cond.cond.current_value


cdef void build_conditional(Chain chain, Link conditional, bint is_custom, bint not_, Link on_true, Link on_false):
  """Build a conditional chain (if/else) by appending cdef class evaluators.

  Architecture:
  - _ConditionalFn / _DirectIf: evaluates the condition and wraps it as _Conditional
  - _IfEvaluator: evaluates on_true/on_false branches. When on_false is None, the
    else evaluator is omitted entirely (optimization: 2 links instead of 3).
  - _ElseEvaluator: only created when on_false is provided; evaluates the false branch.
  """
  cdef _IfEvaluator if_eval = _IfEvaluator(not_, on_true, on_false is not None)
  if conditional is None:
    chain._then(Link(_DirectIf(if_eval), original_value=on_true))
  else:
    chain._then(Link(_ConditionalFn(conditional, is_custom), original_value=conditional))
    chain._then(Link(if_eval, original_value=on_true))
  if on_false is not None:
    chain._then(Link(_ElseEvaluator(on_false), original_value=on_false))


# --- While loop ---

@cython.final
@cython.freelist(4)
cdef class _WhileTrue:
  """Callable that executes a link in an infinite loop until break or return."""

  def __init__(self, Link link, tuple args, dict kwargs, int max_iterations = 0):
    """Initialize with the loop body link and its arguments."""
    self.link = link
    self.args = args
    self.kwargs = kwargs
    self.max_iterations = max_iterations

  def __call__(self, object current_value = _MISSING):
    """Run the loop body repeatedly, handling break/return exceptions and async transitions.

    NOTE: temp_args is mutated on `self.link` for traceback context. Since _WhileTrue
    instances are reusable (the owning Chain may be run multiple times), this mutation
    is visible across runs. This is intentional — temp_args is only read during exception
    formatting, so the last-set value is always the relevant one.
    """
    if current_value is _MISSING:
      current_value = Null
    cdef object result
    cdef int iteration_count = 0
    if not self.args and not self.kwargs:
      self.link.temp_args = (current_value,)
    try:
      while True:
        result = evaluate_value(self.link, current_value)
        iteration_count += 1
        if self.max_iterations > 0 and iteration_count > self.max_iterations:
          raise QuentException(f'while_true exceeded max_iterations ({self.max_iterations})')
        if iscoro(result):
          return _while_true_async_fn(current_value, self.link, result, self.max_iterations, iteration_count)
    except _Break as exc:
      return handle_break_exc(exc, current_value)


cdef Link while_true(object fn, tuple args, dict kwargs, int max_iterations = 0):
  """Execute `fn` in an infinite loop until a `_Break` exception is raised.
  Transparently transitions to async if `fn` returns a coroutine.
  """
  cdef Link link = Link(fn, args, kwargs, fn_name='while_true')
  return Link(_WhileTrue(link, args, kwargs, max_iterations), original_value=link)


async def while_true_async(object current_value, Link link, object result, int max_iterations = 0, int iteration_count = 0):
  """Async continuation of while_true. Awaits the pending coroutine from the
  sync loop first, then continues iterating."""
  try:
    result = await result
    while True:
      result = evaluate_value(link, current_value)
      if iscoro(result):
        result = await result
      iteration_count += 1
      if max_iterations > 0 and iteration_count > max_iterations:
        raise QuentException(f'while_true exceeded max_iterations ({max_iterations})')
  except _Break as exc:
    result = handle_break_exc(exc, current_value)
    if iscoro(result):
      return await result
    return result
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (current_value,)
    raise


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
  """Evaluate a sync context manager, delegating to with_async if the body
  is a coroutine, or to async_with for native async context managers.

  Implements PEP 343 semantics:
  - __exit__ is only called if __enter__ succeeded (entered flag)
  - __exit__ return value is checked for exception suppression
  - Exceptions are captured explicitly rather than via sys.exc_info()
  """
  cdef Link link = Link(fn, args, kwargs, fn_name='with_')
  return Link(_With(link, ignore_result, args, kwargs), original_value=link, ignore_result=ignore_result)


async def with_async(object current_value, object body_result, Link link, bint entered, bint ignore_result, object outer_value):
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


async def async_with(Link link, object current_value, bint ignore_result = False):
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

cdef object _await_if_cond_value_fn = _await_if_cond_value
cdef object _await_cond_result_fn = _await_cond_result
cdef object _while_true_async_fn = while_true_async
cdef object _sync_generator_fn = sync_generator
cdef object _async_generator_fn = async_generator
cdef object _async_with_fn = async_with
cdef object _with_async_fn = with_async
