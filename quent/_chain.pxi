# _chain.pxi — Core Chain class
#
# Defines the Chain class, the central type in quent. A Chain is a linked list
# of operations (Links) that are evaluated sequentially, with transparent
# sync-to-async promotion. Each link receives the result of the previous one
# (or the root value in Cascade mode).
#
# Key components:
#   - Chain.__init__ / init: construction and root value registration
#   - Chain._then: appends a new Link to the chain
#   - Chain._run / _run_async: the main evaluation loop (sync with async fallback)
#   - Chain.then / then_: register callable operations
#   - Chain.if_ / else_ / not_: conditional branching
#   - Chain.foreach / filter / reduce / gather: collection operations
#   - Chain.while_true: infinite loop with break support
#   - Chain.with_: context manager execution
#   - Chain.except_ / finally_: exception handling registration
#   - Chain.return_ / break_: control flow signals
#   - Chain.freeze / as_decorator: reusable chain snapshots
#   - Chain.iterate / iterate_: lazy generator creation
#   - Chain.__or__: pipe syntax support


@cython.freelist(32)
cdef class Chain:
  """A chain of operations that are evaluated lazily.

  Each operation receives the result of the previous operation as its first argument.
  Chains automatically detect and handle coroutines, wrapping them in Tasks without
  requiring explicit ``await`` statements.

  .. warning::
    Chain instances are **not thread-safe**. Do not share a single Chain instance
    across threads. If you need to reuse a chain definition across threads, call
    ``.clone()`` to create an independent copy for each thread.
  """
  def __init__(self, object __v = Null, *args, **kwargs):
    """
    Create a new Chain
    :param v: the root value of the chain
    :param args: arguments to pass to `v`
    :param kwargs: keyword-arguments to pass to `v`
    """
    self.init(__v, args, kwargs, False)

  cdef void init(self, object root_value, tuple args, dict kwargs, bint is_cascade):
    """Initialize chain state with an optional root value and cascade mode."""
    self.is_cascade = is_cascade
    self._autorun = False
    self.uses_attr = False
    self.is_nested = False
    self._debug = False
    self._is_simple = True
    self._is_sync = False
    self._context = None
    if root_value is not Null:
      self.root_link = Link(root_value, args, kwargs, True)
    else:
      self.root_link = None
    self.first_link = None
    self.current_link = None

    self.on_true = None
    self.on_finally_link = None
    self.on_success_link = None
    self.current_conditional = None
    self.current_attr = None

  cdef void _then(self, Link link):
    """Append a Link to the end of the chain's linked list."""
    if self.current_conditional is not None or self.current_attr is not None:
      self.finalize()
    if self._is_simple and (link.ignore_result or link.is_exception_handler or link.is_with_root):
      self._is_simple = False
    if self.is_cascade:
      link.is_with_root = True
    if link.is_attr or link.is_fattr:
      self.uses_attr = True

    if self.current_link is not None:
      self.current_link.next_link = link
      self.current_link = link
    elif self.first_link is not None:
      self.first_link.next_link = link
      self.current_link = link
    else:
      self.first_link = link
      if self.root_link is not None:
        self.root_link.next_link = link

  cdef object _run(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain):
    """Core synchronous execution loop. Evaluates all links and returns the final value or a coroutine."""
    if not invoked_by_parent_chain and self.is_nested:
      raise QuentException('You cannot directly run a nested chain.')
    if self._is_simple and not self._debug and self.on_finally_link is None and self.on_success_link is None and self.current_conditional is None and self.current_attr is None:
      return self._run_simple(v, args, kwargs, invoked_by_parent_chain)
    self.finalize()
    cdef:
      Link link = self.root_link
      Link exc_link
      Link temp_root_link = None
      dict link_results = None
      _ExecCtx ctx = None
      object previous_value = Null, current_value = Null, root_value = Null, result = None, exc = None
      bint has_root_value = link is not None, is_root_value_override = v is not Null
      bint ignore_finally = False

    if is_root_value_override:
      if has_root_value:
        raise QuentException('Cannot override the root value of a Chain.')
    elif not has_root_value:
      if self.uses_attr:
        raise QuentException('Cannot use attributes without a root value.')

    try:
      # --- Root value evaluation ---
      if is_root_value_override:
        temp_root_link = _make_temp_link(v, args, kwargs)
        link = temp_root_link
        link.next_link = self.first_link
        has_root_value = True
      if has_root_value:
        root_value = evaluate_value(link, Null)
        if not self._is_sync and iscoro(root_value):
          ignore_finally = True
          result = self._run_async(temp_root_link, link_results, link, root_value, Null, Null, has_root_value)
          if self._autorun:
            return ensure_future(result)
          return result
        current_value = root_value
        if self._debug:
          _logger.debug('[%s] root -> %r', link.fn_name or 'root', root_value)
          if link_results is None:
            link_results = {}
          link_results[id(link)] = root_value
        link = link.next_link
      else:
        link = self.first_link

      # --- Link evaluation loop ---
      while link is not None:
        if link.ignore_result:
          if link.is_exception_handler:
            link = link.next_link
            continue
          previous_value = current_value
        if not link.is_with_root:
          current_value = evaluate_value(link, current_value)
        else:
          current_value = evaluate_value(link, root_value)
        if not self._is_sync and iscoro(current_value):
          ignore_finally = True
          result = self._run_async(temp_root_link, link_results, link, current_value, root_value, previous_value, has_root_value)
          if self._autorun:
            return ensure_future(result)
          return result
        if self._debug:
          _logger.debug('[%s] -> %r', link.fn_name or '?', current_value)
          if link_results is None:
            link_results = {}
          link_results[id(link)] = current_value
        if link.ignore_result:
          current_value = previous_value
        link = link.next_link

      if self.is_cascade:
        current_value = root_value

      # --- on_success handling ---
      if self.on_success_link is not None:
        result = evaluate_value(self.on_success_link, current_value)
        if not self._is_sync and iscoro(result):
          ignore_finally = True
          return self._run_async_on_success(temp_root_link, link_results, result, current_value, root_value)

      if current_value is Null:
        return None
      return current_value

    # --- Exception handling ---
    except _Return as exc:
      return handle_return_exc(exc, self.is_nested)

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      ctx = _ExecCtx.__new__(_ExecCtx)
      ctx.temp_root_link = temp_root_link
      ctx.link_results = link_results
      ctx.link_temp_args = None
      exc_link = _handle_exception(exc, self, link, ctx)
      if exc_link is None:
        raise exc
      try:
        result = evaluate_value(exc_link, root_value)
      except BaseException as exc_:
        modify_traceback(exc_, self, exc_link, ctx)
        raise exc_ from exc
      if not self._is_sync and iscoro(result):
        if not exc_link.reraise:
          return ensure_future(_await_run_fn(result, self, exc_link, ctx))
        result = ensure_future(_await_run_fn(result, self, exc_link, ctx))
        warnings.warn(
          'An \'except\' callback has returned a coroutine, but the chain is in synchronous mode. '
          'It was therefore scheduled for execution in a new Task.',
          category=RuntimeWarning
        )
      if exc_link.reraise:
        raise exc
      if result is Null:
        return None
      return result

    # --- Finally handling ---
    finally:
      if not ignore_finally and self.on_finally_link is not None:
        try:
          result = evaluate_value(self.on_finally_link, root_value)
        except _InternalQuentException:
          raise QuentException('Using control flow signals inside finally handlers is not allowed.')
        except BaseException as exc_:
          if ctx is None:
            ctx = _ExecCtx.__new__(_ExecCtx)
            ctx.temp_root_link = temp_root_link
            ctx.link_results = link_results
            ctx.link_temp_args = None
          modify_traceback(exc_, self, self.on_finally_link, ctx)
          raise exc_
        if not self._is_sync and iscoro(result):
          if ctx is None:
            ctx = _ExecCtx.__new__(_ExecCtx)
            ctx.temp_root_link = temp_root_link
            ctx.link_results = link_results
            ctx.link_temp_args = None
          ensure_future(_await_run_fn(result, self, self.on_finally_link, ctx))
          warnings.warn(
            'The \'finally\' callback has returned a coroutine, but the chain is in synchronous mode. '
            'It was therefore scheduled for execution in a new Task.',
            category=RuntimeWarning
          )

  async def _run_async(self, Link temp_root_link, dict link_results, Link link, object current_value, object root_value, object previous_value, bint has_root_value):
    """Async continuation of the execution loop, entered when a coroutine is encountered."""
    # Invariant: current_value is an awaitable when this method is entered.
    # After the first await, execution continues with the resolved value.
    cdef:
      Link exc_link
      _ExecCtx ctx = None
      object exc, result

    try:
      current_value = await current_value
      if self._debug:
        if link_results is None:
          link_results = {}
        link_results[id(link)] = current_value
      if link.ignore_result:
        current_value = previous_value
      # update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `root_value` should always be `Null`.
      if has_root_value and root_value is Null:
        root_value = current_value

      link = link.next_link
      while link is not None:
        if link.ignore_result:
          if link.is_exception_handler:
            link = link.next_link
            continue
          previous_value = current_value
        if not link.is_with_root:
          current_value = evaluate_value(link, current_value)
        else:
          current_value = evaluate_value(link, root_value)
        if iscoro(current_value):
          current_value = await current_value
        if self._debug:
          if link_results is None:
            link_results = {}
          link_results[id(link)] = current_value
        if link.ignore_result:
          current_value = previous_value
        link = link.next_link

      if self.is_cascade:
        current_value = root_value

      # --- on_success handling ---
      if self.on_success_link is not None:
        result = evaluate_value(self.on_success_link, current_value)
        if iscoro(result):
          result = await result

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      result = handle_return_exc(exc, self.is_nested)
      if iscoro(result):
        return await result
      return result

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      ctx = _ExecCtx.__new__(_ExecCtx)
      ctx.temp_root_link = temp_root_link
      ctx.link_results = link_results
      ctx.link_temp_args = None
      exc_link = _handle_exception(exc, self, link, ctx)
      if exc_link is None:
        raise exc
      try:
        result = evaluate_value(exc_link, root_value)
        if iscoro(result):
          result = await result
      except BaseException as exc_:
        modify_traceback(exc_, self, exc_link, ctx)
        raise exc_ from exc
      if exc_link.reraise:
        raise exc
      if result is Null:
        return None
      return result

    finally:
      if self.on_finally_link is not None:
        try:
          result = evaluate_value(self.on_finally_link, root_value)
          if iscoro(result):
            result = await result
        except _InternalQuentException:
          raise QuentException('Using control flow signals inside finally handlers is not allowed.')
        except BaseException as exc_:
          if ctx is None:
            ctx = _ExecCtx.__new__(_ExecCtx)
            ctx.temp_root_link = temp_root_link
            ctx.link_results = link_results
            ctx.link_temp_args = None
          modify_traceback(exc_, self, self.on_finally_link, ctx)
          raise exc_

  async def _run_async_on_success(self, Link temp_root_link, dict link_results, object result, object current_value, object root_value):
    """Async continuation for on_success handler when it returns a coroutine in sync path."""
    cdef _ExecCtx ctx = None
    try:
      await result
    except BaseException as exc:
      ctx = _ExecCtx.__new__(_ExecCtx)
      ctx.temp_root_link = temp_root_link
      ctx.link_results = link_results
      ctx.link_temp_args = None
      modify_traceback(exc, self, self.on_success_link, ctx)
      raise exc
    finally:
      if self.on_finally_link is not None:
        try:
          result = evaluate_value(self.on_finally_link, root_value)
          if iscoro(result):
            result = await result
        except _InternalQuentException:
          raise QuentException('Using control flow signals inside finally handlers is not allowed.')
        except BaseException as exc_:
          if ctx is None:
            ctx = _ExecCtx.__new__(_ExecCtx)
            ctx.temp_root_link = temp_root_link
            ctx.link_results = link_results
            ctx.link_temp_args = None
          modify_traceback(exc_, self, self.on_finally_link, ctx)
          raise exc_
    if current_value is Null:
      return None
    return current_value

  cdef object _run_simple(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain):
    """Fast execution path for simple chains (only .then() links, no debug/finally)."""
    if not invoked_by_parent_chain and self.is_nested:
      raise QuentException('You cannot directly run a nested chain.')
    cdef:
      Link link = self.root_link
      Link temp_root_link = None
      _ExecCtx ctx = None
      object current_value = Null, root_value = Null
      bint has_root_value = link is not None, is_root_value_override = v is not Null

    if is_root_value_override:
      if has_root_value:
        raise QuentException('Cannot override the root value of a Chain.')
    elif not has_root_value:
      if self.uses_attr:
        raise QuentException('Cannot use attributes without a root value.')

    try:
      # --- Root value evaluation ---
      if is_root_value_override:
        temp_root_link = _make_temp_link(v, args, kwargs)
        link = temp_root_link
        link.next_link = self.first_link
        has_root_value = True
      if has_root_value:
        root_value = evaluate_value(link, Null)
        if not self._is_sync and iscoro(root_value):
          return self._run_async_simple(temp_root_link, link, root_value, Null, has_root_value)
        current_value = root_value
        link = link.next_link
      else:
        link = self.first_link

      # --- Simple link evaluation loop ---
      # Four separate loops avoid per-iteration branching on `is_sync` x `is_cascade`.
      # Loop 1 (async, chain):   passes current_value, checks for coroutines
      # Loop 2a (sync, cascade): passes root_value, no coro check, restores root at end
      # Loop 2b (sync, chain):   passes current_value, no coro check
      # Loop 3 (async, cascade): passes root_value, checks for coroutines, restores root at end
      # IMPORTANT: logic changes must be applied to all 4 loops consistently.
      if not self._is_sync and not self.is_cascade:
        while link is not None:
          current_value = evaluate_value(link, current_value)
          if iscoro(current_value):
            return self._run_async_simple(temp_root_link, link, current_value, root_value, has_root_value)
          link = link.next_link
      elif self._is_sync:
        if self.is_cascade:
          while link is not None:
            current_value = evaluate_value(link, root_value)
            link = link.next_link
          current_value = root_value
        else:
          while link is not None:
            current_value = evaluate_value(link, current_value)
            link = link.next_link
      else:
        while link is not None:
          current_value = evaluate_value(link, root_value)
          if iscoro(current_value):
            return self._run_async_simple(temp_root_link, link, current_value, root_value, has_root_value)
          link = link.next_link
        current_value = root_value

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      return handle_return_exc(exc, self.is_nested)

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      ctx = _ExecCtx.__new__(_ExecCtx)
      ctx.temp_root_link = temp_root_link
      ctx.link_results = None
      ctx.link_temp_args = None
      modify_traceback(exc, self, link, ctx)
      raise exc

  async def _run_async_simple(self, Link temp_root_link, Link link, object current_value, object root_value, bint has_root_value):
    """Async fast path for simple chains."""
    cdef:
      _ExecCtx ctx = None

    try:
      current_value = await current_value
      if has_root_value and root_value is Null:
        root_value = current_value

      link = link.next_link
      if self.is_cascade:
        while link is not None:
          current_value = evaluate_value(link, root_value)
          if iscoro(current_value):
            current_value = await current_value
          link = link.next_link
        current_value = root_value
      else:
        while link is not None:
          current_value = evaluate_value(link, current_value)
          if iscoro(current_value):
            current_value = await current_value
          link = link.next_link

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      result = handle_return_exc(exc, self.is_nested)
      if iscoro(result):
        return await result
      return result

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      ctx = _ExecCtx.__new__(_ExecCtx)
      ctx.temp_root_link = temp_root_link
      ctx.link_results = None
      ctx.link_temp_args = None
      modify_traceback(exc, self, link, ctx)
      raise exc

  def config(self, *, object autorun = None, object debug = None, object async_ = None):
    """Configure chain options (autorun, debug, async_) in a single call."""
    if autorun is not None:
      self._autorun = bool(autorun)
    if debug is not None:
      self._debug = bool(debug)
    if async_ is not None:
      self._is_sync = not bool(async_)
    return self

  def autorun(self, bint autorun = True):
    """Enable or disable automatic scheduling of async results as Tasks."""
    self._autorun = autorun
    return self

  def set_async(self, bint enabled = True):
    self._is_sync = not enabled
    return self

  def clone(self):
    """Return a deep copy of this chain. The clone shares callables but has independent execution state."""
    self.finalize()
    cdef Chain new_chain = type(self).__new__(type(self))
    cdef Link walk
    new_chain.is_cascade = self.is_cascade
    new_chain._autorun = self._autorun
    new_chain._debug = self._debug
    new_chain._context = dict(self._context) if self._context is not None else None
    new_chain.uses_attr = self.uses_attr
    new_chain.is_nested = False
    new_chain.current_conditional = None
    new_chain.current_attr = None
    new_chain.on_true = None
    new_chain._is_simple = self._is_simple
    new_chain._is_sync = self._is_sync

    # Clone the main link chain
    if self.root_link is not None:
      new_chain.root_link = _clone_chain_links(self.root_link)
      # Walk the cloned chain to find first_link and current_link
      new_chain.first_link = new_chain.root_link.next_link
      if new_chain.first_link is not None:
        walk = new_chain.first_link
        while walk.next_link is not None:
          walk = walk.next_link
        new_chain.current_link = walk if walk is not new_chain.first_link else None
      else:
        new_chain.current_link = None
    else:
      new_chain.root_link = None
      if self.first_link is not None:
        new_chain.first_link = _clone_chain_links(self.first_link)
        walk = new_chain.first_link
        while walk.next_link is not None:
          walk = walk.next_link
        new_chain.current_link = walk if walk is not new_chain.first_link else None
      else:
        new_chain.first_link = None
        new_chain.current_link = None

    # Clone the finally link (single link, not a chain)
    if self.on_finally_link is not None:
      new_chain.on_finally_link = _clone_link(self.on_finally_link)
    else:
      new_chain.on_finally_link = None

    # Clone the on_success link
    if self.on_success_link is not None:
      new_chain.on_success_link = _clone_link(self.on_success_link)
    else:
      new_chain.on_success_link = None

    return new_chain

  def safe_run(self, object __v = Null, *args, **kwargs):
    """Thread-safe execution: clones the chain before running.

    This method creates an independent copy of the chain via .clone()
    and runs the copy, leaving the original chain unmodified. This is
    safe to call from multiple threads or async tasks concurrently.
    """
    __tracebackhide__ = True
    return self.clone().run(__v, *args, **kwargs)

  def with_context(self, **context):
    """Attach context metadata to the chain.

    Context is stored on the chain and can be retrieved via
    ``Chain.get_context()`` during execution.
    """
    if self._context is None:
      self._context = context
    else:
      self._context.update(context)
    return self

  @staticmethod
  def get_context():
    """Retrieve the current chain execution context.

    Returns the context dict set via ``with_context()``, or an empty dict.
    Only valid during chain execution.
    """
    return _current_context.get({})

  def freeze(self):
    """Return a frozen (immutable) snapshot of this chain as a ``_FrozenChain``."""
    self.finalize()
    return _FrozenChain(self._run)

  def decorator(self):
    """Shorthand for freeze().decorator(). Returns a decorator that wraps functions through this chain."""
    return self.freeze().decorator()

  def run(self, object __v = Null, *args, **kwargs):
    """Execute the chain and return the final result (or a coroutine if async)."""
    __tracebackhide__ = True
    if self._context is None:
      try:
        result = self._run(__v, args, kwargs, False)
      except _InternalQuentException as exc:
        raise QuentException(str(exc)) from None
      if self._autorun and not self._is_sync and iscoro(result):
        return ensure_future(result)
      return result
    cdef object token = _current_context.set(self._context)
    try:
      result = self._run(__v, args, kwargs, False)
    except _InternalQuentException as exc:
      _current_context.reset(token)
      raise QuentException(str(exc)) from None
    if not self._is_sync and iscoro(result):
      _current_context.reset(token)
      if self._autorun:
        return ensure_future(self._async_with_context(result))
      return self._async_with_context(result)
    _current_context.reset(token)
    return result

  async def _async_with_context(self, object coro):
    """Wraps a coroutine with context management for async chains."""
    __tracebackhide__ = True
    cdef object token = _current_context.set(self._context)
    try:
      return await coro
    finally:
      _current_context.reset(token)

  # --- Method matrix ---
  # |          | propagate result | ignore result |
  # |----------|-----------------|---------------|
  # | current  | then()          | do()          |
  # | root     | root()          | root_do()     |

  def then(self, object __v, *args, **kwargs):
    """Append an operation whose result becomes the new current value."""
    self._then(Link(__v, args, kwargs, True, 'then'))
    return self

  def do(self, object __fn, *args, **kwargs):
    """Append a side-effect operation whose result is discarded."""
    self._then(Link(__fn, args, kwargs, fn_name='do', ignore_result=True))
    return self

  def root(self, object __fn, *args, **kwargs):
    """Append an operation that receives the root value; its result becomes the new current value."""
    self._then(Link(__fn, args, kwargs, fn_name='root', is_with_root=True))
    return self

  def root_do(self, object __fn, *args, **kwargs):
    """Append a side-effect operation that receives the root value; its result is discarded."""
    self._then(Link(__fn, args, kwargs, fn_name='root_do', is_with_root=True, ignore_result=True))
    return self

  def attr(self, str name):
    """Access an attribute on the current value by name."""
    self._then(Link(name, fn_name='attr', is_attr=True))
    return self

  def attr_fn(self, str __name, *args, **kwargs):
    """Call a method on the current value by name, forwarding args and kwargs."""
    self._then(Link(__name, args, kwargs, fn_name='attr_fn', is_attr=True, is_fattr=True))
    return self

  def except_(self, object __fn, *args, object exceptions = None, bint reraise = True, **kwargs):
    """Register an exception handler. Called with the root value when a matching exception occurs."""
    cdef Link link = Link(__fn, args, kwargs, fn_name='except_')
    link.ignore_result = True
    link.is_exception_handler = True
    if exceptions is not None:
      if isinstance(exceptions, str):
        raise TypeError(f"except_() expects exception types, not string '{exceptions}'")
      if isinstance(exceptions, collections.abc.Iterable):
        link.exceptions = tuple(exceptions)
      else:
        link.exceptions = (exceptions,)
    else:
      link.exceptions = (Exception,)
    link.reraise = reraise
    self._then(link)
    return self

  def suppress(self, *exceptions):
    """Suppress specified exceptions (or all Exceptions if none specified)."""
    if not exceptions:
      exceptions = (Exception,)
    return self.except_(lambda v: None, exceptions=exceptions, reraise=False)

  def finally_(self, object __fn, *args, **kwargs):
    """Register a cleanup callback that always runs after the chain completes."""
    if self.on_finally_link is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally_link = Link(__fn, args, kwargs, fn_name='finally_')
    return self

  def on_success(self, object __fn, *args, **kwargs):
    """Register a callback that runs after successful chain completion (before finally)."""
    if self.on_success_link is not None:
      raise QuentException('You can only register one \'on_success\' callback.')
    self.on_success_link = Link(__fn, args, kwargs, fn_name='on_success')
    return self

  def while_true(self, object __fn, *args, int max_iterations = 0, **kwargs):
    """Execute fn in an infinite loop until a break_ signal is raised."""
    self._then(while_true(__fn, args, kwargs, max_iterations))
    return self

  def iterate(self, object fn = None):
    """Return a generator that lazily executes the chain per element, optionally transforming with fn."""
    return _Generator(self._run, fn, _ignore_result=False)

  def iterate_do(self, object fn = None):
    """Like iterate but ignores fn's result, yielding the original elements."""
    return _Generator(self._run, fn, _ignore_result=True)

  def foreach(self, object fn, bint with_index = False):
    """Apply fn to each element of the current iterable, collecting results."""
    if with_index:
      self._then(foreach_indexed(fn, ignore_result=False))
    else:
      self._then(foreach(fn, ignore_result=False))
    return self

  def foreach_do(self, object fn, bint with_index = False):
    """Apply fn to each element as a side effect, keeping the original elements."""
    if with_index:
      self._then(foreach_indexed(fn, ignore_result=True))
    else:
      self._then(foreach(fn, ignore_result=True))
    return self

  def filter(self, object __fn, *args, **kwargs):
    """Filter elements of the current iterable by a predicate function."""
    self._then(filter_(__fn))
    return self

  def reduce(self, object __fn, object initial = Null, *args, **kwargs):
    """Reduce the current iterable to a single value using a binary function."""
    self._then(reduce_(__fn, initial))
    return self

  def gather(self, *fns):
    """Execute multiple functions in parallel with the current value, collecting results."""
    self._then(gather_(fns))
    return self

  def pipe(self, object other):
    """Pipe the current value through another callable or chain. Semantic alias for .then()."""
    return self.then(other)

  @staticmethod
  def compose(*chains):
    """Compose multiple chains/callables into a single new Chain, executing left to right."""
    cdef Chain c = Chain()
    for chain in chains:
      c.then(chain)
    return c

  def with_(self, object __fn, *args, **kwargs):
    """Execute fn using the current value as a context manager; result becomes the new current value."""
    self._then(with_(__fn, args, kwargs, ignore_result=False))
    return self

  def with_do(self, object __fn, *args, **kwargs):
    """Execute fn using the current value as a context manager; result is discarded."""
    self._then(with_(__fn, args, kwargs, ignore_result=True))
    return self

  def if_(self, object __v, *args, **kwargs):
    """Conditionally execute fn when the current value (or condition) is truthy."""
    self._if(Link(__v, args, kwargs, True, fn_name='if_'))
    return self

  def else_(self, object __v, *args, **kwargs):
    """Execute fn when the preceding if_ or if_not condition was falsy."""
    self._else(Link(__v, args, kwargs, True, fn_name='else_'))
    return self

  def if_not(self, object __v, *args, **kwargs):
    """Conditionally execute fn when the current value (or condition) is falsy."""
    self._if(Link(__v, args, kwargs, True, fn_name='if_not'), not_=True)
    return self

  def if_raise(self, object exc):
    """Raise exc when the current value (or condition) is truthy."""
    self._if(Link(_Raiser(exc), fn_name='if_raise', original_value=exc))
    return self

  def else_raise(self, object exc):
    """Raise exc when the preceding if_ or if_not condition was falsy."""
    self._else(Link(_Raiser(exc), fn_name='else_raise', original_value=exc))
    return self

  def if_not_raise(self, object exc):
    """Raise exc when the current value (or condition) is falsy."""
    self._if(Link(_Raiser(exc), fn_name='if_not_raise', original_value=exc), not_=True)
    return self

  def condition(self, object __fn, *args, **kwargs):
    """Set a custom condition function for the next if_ or if_not call."""
    self.set_conditional(Link(__fn, args, kwargs, fn_name='condition'), custom=True)
    return self

  def not_(self):
    """Set the condition to the logical NOT of the current value."""
    self.set_conditional(Link(_not_instance, fn_name='not_'))
    return self

  def eq(self, object value):
    """Set the condition to check if the current value equals the given value."""
    self.set_conditional(Link(_Comparator(value, _OP_EQ), fn_name='eq', original_value=value))
    return self

  def neq(self, object value):
    """Set the condition to check if the current value does not equal the given value."""
    self.set_conditional(Link(_Comparator(value, _OP_NEQ), fn_name='neq', original_value=value))
    return self

  def is_(self, object value):
    """Set the condition to check identity (is) against the given value."""
    self.set_conditional(Link(_Comparator(value, _OP_IS), fn_name='is_', original_value=value))
    return self

  def is_not(self, object value):
    """Set the condition to check non-identity (is not) against the given value."""
    self.set_conditional(Link(_Comparator(value, _OP_IS_NOT), fn_name='is_not', original_value=value))
    return self

  def in_(self, object value):
    """Set the condition to check if the current value is contained in the given value."""
    self.set_conditional(Link(_Comparator(value, _OP_IN), fn_name='in_', original_value=value))
    return self

  def not_in(self, object value):
    """Set the condition to check if the current value is not contained in the given value."""
    self.set_conditional(Link(_Comparator(value, _OP_NOT_IN), fn_name='not_in', original_value=value))
    return self

  def or_(self, object value):
    """Return the current value if truthy, otherwise return the given fallback value."""
    self._then(Link(_Or(value), fn_name='or_', original_value=value))
    return self

  def isinstance_(self, *types):
    """Set the condition to check if the current value is an instance of the given types."""
    self.set_conditional(Link(_IsInstance(types), fn_name='isinstance_', original_value=types))
    return self

  def raise_(self, object exc):
    """Unconditionally raise the given exception."""
    self._then(Link(_Raiser(exc), fn_name='raise_', original_value=exc))
    return self

  def sleep(self, float delay):
    """Sleep for the given number of seconds. Uses asyncio.sleep in async contexts."""
    self._then(Link(_Sleep(delay), fn_name='sleep', original_value=delay, ignore_result=True))
    return self

  @classmethod
  def null(cls):
    """Return the Null sentinel value."""
    return Null

  @classmethod
  def return_(cls, object __v = Null, *args, **kwargs):
    """Raise a return exception to exit a nested chain early with an optional value."""
    raise _Return(__v, args, kwargs)

  @classmethod
  def break_(cls, object __v = Null, *args, **kwargs):
    """Raise a break exception to exit a while_true loop with an optional value."""
    raise _Break(__v, args, kwargs)

  cdef void _if(self, Link on_true, bint not_ = False):
    """Register a conditional true-branch link, optionally negated."""
    if self.current_conditional is None:
      self.current_conditional = (None, False)
    self.on_true = (on_true, not_)

  cdef void _else(self, Link on_false):
    """Register a conditional false-branch link. Requires a preceding _if call."""
    if self.on_true is None:
      raise QuentException(
        'You cannot use \'.else_()\' without a preceding \'.if_()\' or \'.if_not()\''
      )
    self.finalize_conditional(on_false)

  cdef void set_conditional(self, Link conditional, bint custom = False):
    """Set the current conditional link, finalizing any pending state first."""
    self.finalize()
    self.current_conditional = (conditional, custom)

  cdef void finalize(self):
    """Flush any pending conditional or attribute access into the link chain."""
    cdef str attr
    if self.current_conditional is None and self.current_attr is None:
      return
    if self.current_conditional is not None:
      self.finalize_conditional()
    # TODO separate this into a overriding function in ChainAttr / CascadeAttr
    elif self.current_attr is not None:
      attr = self.current_attr
      self.current_attr = None
      self._then(Link(attr, is_attr=True))

  cdef void finalize_conditional(self, Link on_false = None):
    """Build and append the conditional link structure (condition + true/false branches)."""
    cdef:
      Link conditional
      Link on_true
      bint is_custom, not_
    conditional, is_custom = self.current_conditional
    self.current_conditional = None
    if self.on_true:
      on_true, not_ = self.on_true
      self.on_true = None
      build_conditional(self, conditional, is_custom, not_, on_true, on_false)
    else:
      self._then(conditional)

  def __or__(self, other):
    """Pipe operator: append a value/callable, or execute the chain if other is a ``run`` instance."""
    __tracebackhide__ = True
    if isinstance(other, run):
      return self.run(other.root_value, *other.args, **other.kwargs)
    self._then(Link(other, None, None, True))
    return self

  def __call__(self, object __v = Null, *args, **kwargs):
    """Shorthand for run(). Execute the chain with an optional root value override."""
    __tracebackhide__ = True
    if self._context is not None:
      return self.run(__v, *args, **kwargs)
    try:
      result = self._run(__v, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None
    if self._autorun and not self._is_sync and iscoro(result):
      return ensure_future(result)
    return result

  def __bool__(self):
    """Always returns True so chains are truthy."""
    return True

  def __repr__(self):
    """Return a string representation of the chain's structure."""
    return stringify_chain(self, None)[0]
