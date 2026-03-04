@cython.freelist(32)
cdef class Chain:
  def __init__(self, object __v = Null, *args, **kwargs):
    self.init(__v, args, kwargs)

  cdef void init(self, object root_value, tuple args, dict kwargs):
    self.is_nested = False
    if root_value is not Null:
      self.root_link = Link(root_value, args, kwargs)
    else:
      self.root_link = None
    self.first_link = None
    self.current_link = None
    self.on_finally_link = None

  cdef void _then(self, Link link):
    """Append a Link to the end of the chain's linked list."""
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
    if not invoked_by_parent_chain and self.is_nested:
      raise QuentException('You cannot directly run a nested chain.')
    cdef:
      Link link = self.root_link
      Link exc_link
      Link temp_root_link = None
      dict link_results = None
      _ExecCtx ctx = None
      object current_value = Null, root_value = Null, result = None, exc = None
      bint has_root_value = link is not None, is_root_value_override = v is not Null
      bint ignore_finally = False

    if is_root_value_override:
      if has_root_value:
        raise QuentException('Cannot override the root value of a Chain.')

    try:
      if is_root_value_override:
        temp_root_link = _make_temp_link(v, args, kwargs)
        link = temp_root_link
        link.next_link = self.first_link
        has_root_value = True
      if has_root_value:
        result = evaluate_value(link, Null)
        if iscoro(result):
          ignore_finally = True
          return self._run_async(temp_root_link, link_results, link, result, Null, Null, has_root_value)
        root_value = result
        current_value = result
        link = link.next_link
      else:
        link = self.first_link

      while link is not None:
        if link.is_exception_handler:
          link = link.next_link
          continue
        result = evaluate_value(link, current_value)
        if iscoro(result):
          ignore_finally = True
          return self._run_async(temp_root_link, link_results, link, result, current_value, root_value, has_root_value)
        if not link.ignore_result:
          current_value = result
        link = link.next_link

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
      if iscoro(result):
        if not exc_link.reraise:
          return ensure_future(_await_run_fn(result, self, exc_link, ctx))
        result = ensure_future(_await_run_fn(result, self, exc_link, ctx))
        warnings.warn(
          'An except handler returned a coroutine from a synchronous execution path. '
          'It was scheduled as a fire-and-forget Task via ensure_future().',
          category=RuntimeWarning
        )
      if exc_link.reraise:
        raise exc
      if result is Null:
        return None
      return result

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
        if iscoro(result):
          if ctx is None:
            ctx = _ExecCtx.__new__(_ExecCtx)
            ctx.temp_root_link = temp_root_link
            ctx.link_results = link_results
            ctx.link_temp_args = None
          ensure_future(_await_run_fn(result, self, self.on_finally_link, ctx))
          warnings.warn(
            'A finally handler returned a coroutine from a synchronous execution path. '
            'It was scheduled as a fire-and-forget Task via ensure_future().',
            category=RuntimeWarning
          )

  async def _run_async(self, Link temp_root_link, dict link_results, Link link, object awaitable, object current_value, object root_value, bint has_root_value):
    cdef:
      Link exc_link
      _ExecCtx ctx = None
      object exc, result

    try:
      result = await awaitable
      if not link.ignore_result:
        current_value = result
      # Update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `root_value` should always be `Null`.
      if has_root_value and root_value is Null:
        root_value = current_value

      link = link.next_link
      while link is not None:
        if link.is_exception_handler:
          link = link.next_link
          continue
        result = evaluate_value(link, current_value)
        if iscoro(result):
          result = await result
        if not link.ignore_result:
          current_value = result
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

  def clone(self):
    cdef Chain new_chain = Chain.__new__(Chain)
    cdef Link walk
    new_chain.is_nested = False

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

    return new_chain

  def decorator(self):
    """Return a decorator that wraps functions to run through this chain."""
    cdef Chain chain = self
    def _decorator(fn):
      """Wrap fn so that calling it executes the chain with fn as root value."""
      result = _DescriptorWrapper(_ChainCallWrapper(chain, fn))
      try:
        functools.update_wrapper(result, fn)
      except AttributeError:
        pass
      return result
    return _decorator

  def run(self, object __v = Null, *args, **kwargs):
    try:
      return self._run(__v, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def then(self, object __v, *args, **kwargs):
    self._then(Link(__v, args, kwargs))
    return self

  def do(self, object __fn, *args, **kwargs):
    self._then(Link(__fn, args, kwargs, ignore_result=True))
    return self

  def except_(self, object __fn, *args, object exceptions = None, bint reraise = True, **kwargs):
    cdef Link link = Link(__fn, args, kwargs)
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

  def finally_(self, object __fn, *args, **kwargs):
    if self.on_finally_link is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally_link = Link(__fn, args, kwargs)
    return self

  def iterate(self, object fn = None):
    return _Generator(self._run, fn, _ignore_result=False)

  def foreach(self, object fn):
    self._then(foreach(fn, ignore_result=False))
    return self

  def filter(self, object __fn):
    self._then(filter_(__fn))
    return self

  def gather(self, *fns):
    self._then(gather_(fns))
    return self

  def with_(self, object __fn, *args, **kwargs):
    self._then(with_(__fn, args, kwargs, ignore_result=False))
    return self

  @classmethod
  def return_(cls, object __v = Null, *args, **kwargs):
    raise _Return(__v, args, kwargs)

  @classmethod
  def break_(cls, object __v = Null, *args, **kwargs):
    raise _Break(__v, args, kwargs)

  def __call__(self, object __v = Null, *args, **kwargs):
    try:
      return self._run(__v, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def __bool__(self):
    return True

  def __repr__(self):
    return stringify_chain(self, None)[0]
