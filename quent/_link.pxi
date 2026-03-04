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


cdef Link _clone_link(Link src):
  cdef Link dst = Link.__new__(Link)
  dst.v = src.v
  dst.args = src.args
  dst.kwargs = src.kwargs
  dst.eval_code = src.eval_code
  dst.ignore_result = src.ignore_result
  dst.is_chain = src.is_chain
  dst.original_value = src.original_value
  # Reset execution state
  dst.temp_args = None
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


cdef object evaluate_value(Link link, object current_value):
  # Fast path for most common case: simple callable with single argument
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
      return (<Chain>link.v)._run(link.args[0], link.args[1:], link.kwargs, True)
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
      if link.kwargs is EMPTY_DICT:
        return link.v(*link.args)
      return link.v(*link.args, **link.kwargs)
    elif link.eval_code == EVAL_CALL_WITHOUT_ARGS:
      return link.v()
    else:
      return link.v
