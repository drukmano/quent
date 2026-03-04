# PERF: _FrozenChain — pre-compiled chain evaluation using contiguous tuple instead of linked list.
# At freeze time, the linked list of Links is "compiled" into a tuple, enabling:
# - Sequential memory access via PyTuple_GET_ITEM (CPU prefetcher friendly)
# - No pointer-chasing through next_link
# - _all_simple fast path that skips evaluate_value dispatch entirely
# See PHASE-4-DATA-STRUCTURES.md for design rationale.


@cython.final
@cython.freelist(4)
cdef class _FrozenChain:
  """A frozen chain that pre-compiles the linked list into a contiguous tuple
  for faster iteration.

  Created via Chain.freeze(). Provides the same run/call interface as Chain
  but with optimized evaluation using a pre-built tuple of Links instead of
  linked-list traversal.
  """

  def __init__(self, Chain chain):
    self._chain = chain
    cdef list links = []
    cdef Link link = chain.first_link
    cdef bint all_simple = True
    while link is not None:
      links.append(link)
      if link.eval_code != EVAL_CALL_WITH_CURRENT_VALUE or link.is_chain or link.ignore_result:
        all_simple = False
      link = link.next_link
    self._links = tuple(links)
    self._n_links = len(links)
    self._all_simple = all_simple
    self._has_finally = chain.on_finally_link is not None
    self._has_except = chain.on_except_link is not None

  def run(self, object __v = Null, *args, **kwargs):
    try:
      return _frozen_run(self, __v, args, kwargs)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def __call__(self, object __v = Null, *args, **kwargs):
    try:
      return _frozen_run(self, __v, args, kwargs)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def __bool__(self):
    return True

  def __repr__(self):
    return f'<FrozenChain links={self._n_links} all_simple={self._all_simple}>'

  async def _run_async(self, _ExecCtx ctx, object awaitable, int start_idx):
    cdef:
      Chain chain = self._chain
      dict exc_temp_args
      object exc, result
      Link link = ctx.async_link
      object current_value = ctx.current_value
      object root_value = ctx.root_value
      bint has_root_value = ctx.has_root_value
      int i, n = self._n_links

    try:
      result = await awaitable
      if not link.ignore_result:
        current_value = result
      # Update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `root_value` should always be `Null`.
      if has_root_value and root_value is Null:
        root_value = current_value

      for i in range(start_idx + 1, n):
        link = <Link>self._links[i]
        result = evaluate_value(link, current_value)
        if iscoro(result):
          result = await result
        if not link.ignore_result:
          current_value = result

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      result = handle_return_exc(exc, False)
      if iscoro(result):
        return await result
      return result

    except _Break:
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      ctx.link_temp_args = None
      exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
      if exc_temp_args is not None:
        ctx.link_temp_args = exc_temp_args
      modify_traceback(exc, chain, link, ctx)
      if chain.on_except_link is None or not isinstance(exc, chain.on_except_exceptions):
        raise exc
      try:
        result = evaluate_value(chain.on_except_link, exc)
        if iscoro(result):
          result = await result
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside except handlers is not allowed.')
      except BaseException as exc_:
        modify_traceback(exc_, chain, chain.on_except_link, ctx)
        raise exc_ from exc
      if result is Null:
        return None
      return result

    finally:
      if chain.on_finally_link is not None:
        try:
          result = evaluate_value(chain.on_finally_link, root_value)
          if iscoro(result):
            result = await result
        except _InternalQuentException:
          raise QuentException('Using control flow signals inside finally handlers is not allowed.')
        except BaseException as exc_:
          modify_traceback(exc_, chain, chain.on_finally_link, ctx)
          raise exc_


cdef object _frozen_run(_FrozenChain fc, object v, tuple args, dict kwargs):
  cdef:
    Chain chain = fc._chain
    Link link = None
    Link temp_root_link = None
    dict link_results = None
    dict exc_temp_args
    _ExecCtx ctx = None
    object current_value = Null, root_value = Null, result = None
    bint has_root_value = chain.root_link is not None
    bint is_root_value_override = v is not Null
    bint ignore_finally = False
    int i, n = fc._n_links

  if is_root_value_override:
    if has_root_value:
      raise QuentException('Cannot override the root value of a Chain.')

  try:
    if is_root_value_override:
      # PERF: Inline root evaluation — avoids Link allocation for the common sync case.
      has_root_value = True
      result = _eval_signal_value(v, args, kwargs)
      if iscoro(result):
        temp_root_link = _make_temp_link(v, args, kwargs)
        ignore_finally = True
        ctx = _ExecCtx.__new__(_ExecCtx)
        ctx.temp_root_link = temp_root_link
        ctx.link_results = link_results
        ctx.link_temp_args = None
        ctx.async_link = temp_root_link
        ctx.current_value = Null
        ctx.root_value = Null
        ctx.has_root_value = True
        return fc._run_async(ctx, result, -1)
      root_value = result
      current_value = result
    elif has_root_value:
      link = chain.root_link
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
        return fc._run_async(ctx, result, -1)
      root_value = result
      current_value = result

    # PERF: Tuple-based iteration — sequential memory access via PyTuple_GET_ITEM.
    # Replaces linked-list pointer chasing (link = link.next_link).
    if fc._all_simple:
      # PERF: Ultra-fast path — all links are simple callables
      # (EVAL_CALL_WITH_CURRENT_VALUE, not is_chain, not ignore_result).
      # Skips evaluate_value dispatch entirely.
      for i in range(n):
        link = <Link>fc._links[i]
        if current_value is Null:
          result = link.v()
        else:
          result = link.v(current_value)
        if iscoro(result):
          ignore_finally = True
          ctx = _ExecCtx.__new__(_ExecCtx)
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = link
          ctx.current_value = current_value
          ctx.root_value = root_value
          ctx.has_root_value = has_root_value
          return fc._run_async(ctx, result, i)
        current_value = result
    else:
      for i in range(n):
        link = <Link>fc._links[i]
        result = evaluate_value(link, current_value)
        if iscoro(result):
          ignore_finally = True
          ctx = _ExecCtx.__new__(_ExecCtx)
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = link
          ctx.current_value = current_value
          ctx.root_value = root_value
          ctx.has_root_value = has_root_value
          return fc._run_async(ctx, result, i)
        if not link.ignore_result:
          current_value = result

    if current_value is Null:
      return None
    return current_value

  except _Return as exc:
    return handle_return_exc(exc, False)

  except _Break:
    raise QuentException('_Break cannot be used in this context.')

  except BaseException as exc:
    if is_root_value_override and temp_root_link is None:
      temp_root_link = _make_temp_link(v, args, kwargs)
      temp_root_link.next_link = chain.first_link
      link = temp_root_link
    ctx = _ExecCtx.__new__(_ExecCtx)
    ctx.temp_root_link = temp_root_link
    ctx.link_results = link_results
    ctx.link_temp_args = None
    exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
    if exc_temp_args is not None:
      ctx.link_temp_args = exc_temp_args
    modify_traceback(exc, chain, link, ctx)
    if chain.on_except_link is None or not isinstance(exc, chain.on_except_exceptions):
      raise exc
    try:
      result = evaluate_value(chain.on_except_link, exc)
    except _InternalQuentException:
      raise QuentException('Using control flow signals inside except handlers is not allowed.')
    except BaseException as exc_:
      modify_traceback(exc_, chain, chain.on_except_link, ctx)
      raise exc_ from exc
    if iscoro(result):
      result = ensure_future(_await_run(result, chain, chain.on_except_link, ctx))
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
    if not ignore_finally and chain.on_finally_link is not None:
      try:
        result = evaluate_value(chain.on_finally_link, root_value)
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside finally handlers is not allowed.')
      except BaseException as exc_:
        if ctx is None:
          ctx = _ExecCtx.__new__(_ExecCtx)
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
        modify_traceback(exc_, chain, chain.on_finally_link, ctx)
        raise exc_
      if iscoro(result):
        if ctx is None:
          ctx = _ExecCtx.__new__(_ExecCtx)
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
        ensure_future(_await_run(result, chain, chain.on_finally_link, ctx))
        warnings.warn(
          'A finally handler returned a coroutine from a synchronous execution path. '
          'It was scheduled as a fire-and-forget Task via ensure_future().',
          category=RuntimeWarning
        )
