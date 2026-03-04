# PERF: inline + noexcept — inlined at call sites, no exception-state check overhead.
# Pre-computes evaluation strategy as a C int enum at Link construction time.
# See PERFORMANCE.md #20.
cdef inline int _determine_eval_code(
  Link link, object v, tuple args, dict kwargs,
) noexcept:
  if not args and not kwargs:
    if PyCallable_Check(v):
      return EVAL_CALL_WITH_CURRENT_VALUE
    else:
      return EVAL_RETURN_AS_IS
  elif args:
    if args[0] is ...:
      return EVAL_CALL_WITHOUT_ARGS
    else:
      if kwargs is None:
        link.kwargs = EMPTY_DICT
      return EVAL_CALL_WITH_EXPLICIT_ARGS
  else:
    if link.is_chain:
      # A Chain cannot be run with custom keyword arguments but without
      # at least one positional argument.
      return EVAL_CALL_WITHOUT_ARGS
    else:
      if args is None:
        link.args = EMPTY_TUPLE
      return EVAL_CALL_WITH_EXPLICIT_ARGS


# PERF: @cython.final enables direct C dispatch. @cython.freelist(64) pools allocations.
# See PERFORMANCE.md #1, #2.
@cython.final
@cython.freelist(64)
cdef class Link:
  def __init__(
    self, object v, tuple args = None, dict kwargs = None,
    bint ignore_result = False,
    object original_value = None,
  ):
    cdef bint is_chain_type = type(v) is Chain
    if is_chain_type:
      self.is_chain = True
      (<Chain>v).is_nested = True
    else:
      self.is_chain = False
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.ignore_result = ignore_result
    self.original_value = original_value
    self.temp_args = None

    self.next_link = None
    self.eval_code = _determine_eval_code(self, v, args, kwargs)
    # PERF: Pre-split args for nested chain links at construction time.
    # Avoids tuple slice link.args[1:] (score 20) on every evaluate_value call.
    # See PERFORMANCE.md #36.
    if is_chain_type and self.eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS and self.args:
      self.temp_args = self.args[1:] if len(self.args) > 1 else EMPTY_TUPLE


cdef Link _clone_link(Link src):
  cdef Link dst = Link.__new__(Link)
  dst.v = src.v
  dst.args = src.args
  dst.kwargs = src.kwargs
  dst.eval_code = src.eval_code
  dst.ignore_result = src.ignore_result
  dst.is_chain = src.is_chain
  dst.original_value = src.original_value
  # Preserve pre-computed state (e.g., pre-split args for chain links)
  dst.temp_args = src.temp_args
  dst.next_link = None
  return dst


cdef Link _clone_chain_links(Link src):
  if src is None:
    return None
  cdef Link head = _clone_link(src)
  cdef Link prev = head
  cdef Link current = src.next_link
  while current is not None:
    prev.next_link = _clone_link(current)
    prev = prev.next_link
    current = current.next_link
  return head


cdef Link _make_temp_link(object v, tuple args, dict kwargs):
  cdef Link link = Link.__new__(Link)
  link.v = v
  link.args = args
  link.kwargs = kwargs
  link.ignore_result = False
  link.is_chain = False
  link.original_value = None
  link.temp_args = None
  link.next_link = None
  link.eval_code = _determine_eval_code(link, v, args, kwargs)
  return link


# PERF: cdef factory bypasses Link.__init__ Python argument parsing overhead (score 27).
# All internal code paths use this instead of Link(v, args, kwargs).
# Link.__init__ is kept for external API compatibility. See PERFORMANCE.md #34.
cdef Link _create_link(object v, tuple args, dict kwargs, bint ignore_result=False, object original_value=None):
  cdef Link link = Link.__new__(Link)
  cdef bint is_chain_type = type(v) is Chain
  if is_chain_type:
    link.is_chain = True
    (<Chain>v).is_nested = True
  else:
    link.is_chain = False
  link.v = v
  link.args = args
  link.kwargs = kwargs
  link.ignore_result = ignore_result
  link.original_value = original_value
  link.temp_args = None
  link.next_link = None
  link.eval_code = _determine_eval_code(link, v, args, kwargs)
  # PERF: Pre-split args for nested chain links at construction time.
  if is_chain_type and link.eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS and link.args:
    link.temp_args = link.args[1:] if len(link.args) > 1 else EMPTY_TUPLE
  return link


cdef object evaluate_value(Link link, object current_value):
  # PERF: Fast path for most common case — single branch before full dispatch.
  # See PERFORMANCE.md #20.
  if link.eval_code == EVAL_CALL_WITH_CURRENT_VALUE and not link.is_chain:
    if current_value is Null:
      return link.v()
    return link.v(current_value)

  if link.is_chain:
    # we deliberately did not set `link.v = <Chain>v` in `Link.__init__` since
    # it adds an ugly, unreadable stack frame in case of an exception.
    if link.eval_code == EVAL_CALL_WITH_CURRENT_VALUE:
      return (<Chain>link.v)._run(current_value, None, None, True)
    elif link.eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS:
      if not link.args or link.kwargs is None:
        # Safety: when args/kwargs are None (not just empty), Cython would dereference
        # a NULL PyObject* when unpacking them, causing a segfault at the C level.
        return (<Chain>link.v)._run(Null, None, None, True)
      # PERF: Use pre-split temp_args instead of link.args[1:] (avoids tuple allocation per call).
      # See PERFORMANCE.md #36.
      return (<Chain>link.v)._run(link.args[0], link.temp_args, link.kwargs, True)
    elif link.eval_code == EVAL_CALL_WITHOUT_ARGS:
      return (<Chain>link.v)._run(Null, None, None, True)
    else:
      raise QuentException(
        'Invalid evaluation code found for a nested chain. '
        'If you see this error then something has gone terribly wrong.'
      )

  else:
    if link.eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS:
      if link.args is None or link.kwargs is None:
        # Safety: when args/kwargs are None (not just empty), Cython would dereference
        # a NULL PyObject* when unpacking them, causing a segfault at the C level.
        return link.v()
      # PERF: Identity check against EMPTY_DICT sentinel skips **kwargs unpacking overhead.
      # See PERFORMANCE.md #6.
      if link.kwargs is EMPTY_DICT:
        return link.v(*link.args)
      return link.v(*link.args, **link.kwargs)
    elif link.eval_code == EVAL_CALL_WITHOUT_ARGS:
      return link.v()
    else:
      return link.v
