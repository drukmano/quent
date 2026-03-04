@cython.final
@cython.freelist(32)
cdef class Chain:
  def __init__(self, object __v = Null, *args, **kwargs):
    self.init(__v, args, kwargs)

  cdef void init(self, object root_value, tuple args, dict kwargs):
    self.is_nested = False
    if root_value is not Null:
      self.root_link = _create_link(root_value, args, kwargs)
    else:
      self.root_link = None
    self.first_link = None
    self.current_link = None
    self.on_finally_link = None
    self.on_except_link = None
    self.on_except_exceptions = None

  cdef void _then(self, Link link):
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
      Link temp_root_link = None
      dict link_results = None
      dict exc_temp_args
      _ExecCtx ctx = None
      object current_value = Null, root_value = Null, result = None, exc = None
      bint has_root_value = link is not None, is_root_value_override = v is not Null
      bint ignore_finally = False

    if is_root_value_override:
      if has_root_value:
        raise QuentException('Cannot override the root value of a Chain.')

    try:
      if is_root_value_override:
        # PERF: Inline root evaluation — avoids Link allocation for the common sync case.
        # _eval_signal_value replicates evaluate_value dispatch without creating a Link.
        has_root_value = True
        result = _eval_signal_value(v, args, kwargs)
        if iscoro(result):
          # Async path: lazily create temp link for continuation + diagnostics.
          temp_root_link = _make_temp_link(v, args, kwargs)
          temp_root_link.next_link = self.first_link
          link = temp_root_link
          ignore_finally = True
          ctx = _ExecCtx.__new__(_ExecCtx)
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = temp_root_link
          ctx.current_value = Null
          ctx.root_value = Null
          ctx.has_root_value = True
          return self._run_async(ctx, result)
        root_value = result
        current_value = result
        link = self.first_link
      elif has_root_value:
        result = evaluate_value(link, Null)
        if iscoro(result):
          ignore_finally = True
          ctx = _ExecCtx.__new__(_ExecCtx)
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = link
          ctx.current_value = Null
          ctx.root_value = Null
          ctx.has_root_value = has_root_value
          return self._run_async(ctx, result)
        root_value = result
        current_value = result
        link = link.next_link
      else:
        link = self.first_link

      while link is not None:
        result = evaluate_value(link, current_value)
        if iscoro(result):
          ignore_finally = True
          # PERF: Pack async transition state into _ExecCtx instead of passing 7 args.
          ctx = _ExecCtx.__new__(_ExecCtx)
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = link
          ctx.current_value = current_value
          ctx.root_value = root_value
          ctx.has_root_value = has_root_value
          return self._run_async(ctx, result)
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
      # If root override raised before creating temp_root_link, create it now for diagnostics.
      if is_root_value_override and temp_root_link is None:
        temp_root_link = _make_temp_link(v, args, kwargs)
        temp_root_link.next_link = self.first_link
        link = temp_root_link
      ctx = _ExecCtx.__new__(_ExecCtx)
      ctx.temp_root_link = temp_root_link
      ctx.link_results = link_results
      ctx.link_temp_args = None
      exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
      if exc_temp_args is not None:
        ctx.link_temp_args = exc_temp_args
      modify_traceback(exc, self, link, ctx)
      if self.on_except_link is None or not isinstance(exc, self.on_except_exceptions):
        raise exc
      try:
        result = evaluate_value(self.on_except_link, exc)
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside except handlers is not allowed.')
      except BaseException as exc_:
        modify_traceback(exc_, self, self.on_except_link, ctx)
        raise exc_ from exc
      if iscoro(result):
        result = ensure_future(_await_run(result, self, self.on_except_link, ctx))
        warnings.warn(
          'An except handler returned a coroutine from a synchronous execution path. '
          'It was scheduled as a fire-and-forget Task via ensure_future().',
          category=RuntimeWarning
        )
        return result
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
          ensure_future(_await_run(result, self, self.on_finally_link, ctx))
          warnings.warn(
            'A finally handler returned a coroutine from a synchronous execution path. '
            'It was scheduled as a fire-and-forget Task via ensure_future().',
            category=RuntimeWarning
          )

  # PERF: Reduced from 8 parameters to 3 by packing state into _ExecCtx.
  # Eliminates Python argument parsing overhead and bint->PyBool boxing.
  async def _run_async(self, _ExecCtx ctx, object awaitable):
    cdef:
      dict exc_temp_args
      object exc, result
      # Unpack async transition state from ctx
      Link link = ctx.async_link
      object current_value = ctx.current_value
      object root_value = ctx.root_value
      bint has_root_value = ctx.has_root_value

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
      # ctx already has temp_root_link and link_results from _run — reuse it.
      ctx.link_temp_args = None
      exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
      if exc_temp_args is not None:
        ctx.link_temp_args = exc_temp_args
      modify_traceback(exc, self, link, ctx)
      if self.on_except_link is None or not isinstance(exc, self.on_except_exceptions):
        raise exc
      try:
        result = evaluate_value(self.on_except_link, exc)
        if iscoro(result):
          result = await result
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside except handlers is not allowed.')
      except BaseException as exc_:
        modify_traceback(exc_, self, self.on_except_link, ctx)
        raise exc_ from exc
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
          # ctx is always available — no need to check for None.
          modify_traceback(exc_, self, self.on_finally_link, ctx)
          raise exc_

  def decorator(self):
    cdef Chain chain = self
    def _decorator(fn):
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

  def freeze(self):
    return _FrozenChain(self)

  def then(self, object __v, *args, **kwargs):
    self._then(_create_link(__v, args, kwargs))
    return self

  def do(self, object __fn, *args, **kwargs):
    self._then(_create_link(__fn, args, kwargs, ignore_result=True))
    return self

  def except_(self, object __fn, *args, object exceptions = None, **kwargs):
    if self.on_except_link is not None:
      raise QuentException('You can only register one \'except\' callback.')
    if exceptions is not None:
      if isinstance(exceptions, str):
        raise TypeError(f"except_() expects exception types, not string '{exceptions}'")
      if isinstance(exceptions, collections.abc.Iterable):
        self.on_except_exceptions = tuple(exceptions)
      else:
        self.on_except_exceptions = (exceptions,)
    else:
      self.on_except_exceptions = (Exception,)
    self.on_except_link = _create_link(__fn, args, kwargs)
    return self

  def finally_(self, object __fn, *args, **kwargs):
    if self.on_finally_link is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally_link = _create_link(__fn, args, kwargs)
    return self

  def iterate(self, object fn = None):
    return _Generator(self._run, fn, _ignore_result=False)

  def iterate_do(self, object fn = None):
    return _Generator(self._run, fn, _ignore_result=True)

  def foreach(self, object fn):
    self._then(foreach(fn, ignore_result=False))
    return self

  def foreach_do(self, object fn):
    self._then(foreach(fn, ignore_result=True))
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

  def with_do(self, object __fn, *args, **kwargs):
    self._then(with_(__fn, args, kwargs, ignore_result=True))
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
