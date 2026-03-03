# _link.pxi — Link node and value evaluation
#
# Defines the Link class (the fundamental node in a chain's linked list of
# operations), the evaluate_value dispatch function, and supporting utilities
# for cloning and creating temporary links.
#
# Key components:
#   - _await_run / _await_run_fn: async wrapper for awaiting chain results
#   - _determine_eval_code: resolves how a link should be evaluated
#   - Link: the chain node class holding a value, args, and evaluation metadata
#   - _clone_link / _clone_chain_links: deep-copy utilities for Link lists
#   - _make_temp_link: factory for temporary root-value override links
#   - evaluate_value: central dispatch — calls or returns a link's value


async def _await_run(result, chain=None, link=None, ctx=None):
  """Await a coroutine result and optionally annotate exceptions with chain traceback info."""
  __tracebackhide__ = True
  try:
    return await result
  except BaseException as exc:
    if chain is not None and link is not None:
      modify_traceback(exc, chain, link, ctx)
    raise remove_self_frames_from_traceback()

cdef object _await_run_fn = _await_run


cdef inline int _determine_eval_code(
  Link link, object v, tuple args, dict kwargs,
  bint allow_literal,
):
  """Determine the eval_code for a Link based on its value and arguments.

  Also normalizes args/kwargs on the link (fills EMPTY_DICT/EMPTY_TUPLE as needed).
  Raises QuentException if the value is a non-callable literal and allow_literal is False.
  """
  if not args and not kwargs:
    if callable(v):
      return EVAL_CALL_WITH_CURRENT_VALUE
    else:
      if not allow_literal:
        raise QuentException('Non-callable objects cannot be used with this method.')
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
  """A single node in the chain's linked list of operations.

  Each Link holds a callable (or literal value), its arguments, and metadata
  controlling how it is evaluated (eval_code) and whether its result propagates.
  Links are connected via ``next_link`` to form the evaluation sequence.
  """

  # ┌─────────────────────────────────────────────────────────────────────────┐
  # │ FIELD INVENTORY (14 fields — see quent.pxd lines 33-39)               │
  # │                                                                         │
  # │ All fields must stay in sync across three locations:                    │
  # │   • __init__           — normal construction                           │
  # │   • _clone_link()      — deep-copy for chain cloning                   │
  # │   • _make_temp_link()  — fast-path allocation (bypasses __init__)      │
  # │                                                                         │
  # │ Field            Type    Purpose                                        │
  # │ ─────            ────    ───────                                        │
  # │ v                object  The callable or literal value                  │
  # │ args             tuple   Positional args passed alongside v             │
  # │ kwargs           dict    Keyword args passed alongside v                │
  # │ is_with_root     bint    True if this link receives the root value      │
  # │ ignore_result    bint    True if v's return value is discarded          │
  # │ original_value   object  The raw user-supplied value (before wrapping)  │
  # │ fn_name          str     Display name for debugging/repr               │
  # │ temp_args        tuple   Transient args from _ExecCtx (cleared on run) │
  # │ is_exception_handler bint True if this is an except_/finally_ link     │
  # │ exceptions       object  Exception type(s) this handler catches        │
  # │ reraise          bint    True if caught exceptions should be reraised   │
  # │ next_link        Link    Pointer to the next link in the chain          │
  # │ eval_code        int     Dispatch code for evaluate_value()             │
  # │ is_chain         bint    True if v is a Chain instance                  │
  # └─────────────────────────────────────────────────────────────────────────┘

  def __init__(
    self, object v, tuple args = None, dict kwargs = None, bint allow_literal = False, str fn_name = None,
    bint is_with_root = False, bint ignore_result = False,
    object original_value = None,
  ):
    """Initialize a Link with a value, arguments, and evaluation metadata."""
    cdef bint is_chain_type = isinstance(v, Chain)
    if is_chain_type:
      self.is_chain = True
      (<Chain>v).is_nested = True
    else:
      self.is_chain = False
      if _PipeCls is not None and isinstance(v, _PipeCls):
        v = v.function
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.is_with_root = is_with_root
    self.ignore_result = ignore_result
    self.original_value = original_value
    self.fn_name = fn_name
    self.temp_args = None

    self.is_exception_handler = False
    self.exceptions = None
    self.reraise = True
    self.next_link = None
    self.eval_code = _determine_eval_code(self, v, args, kwargs, allow_literal)


cdef Link _clone_link(Link src):
  """Deep-copy a single Link, resetting execution state (result, temp_args)."""
  cdef Link dst = Link.__new__(Link)
  dst.v = src.v
  dst.args = src.args
  dst.kwargs = src.kwargs
  dst.eval_code = src.eval_code
  dst.is_with_root = src.is_with_root
  dst.ignore_result = src.ignore_result
  dst.is_chain = src.is_chain
  dst.original_value = src.original_value
  dst.fn_name = src.fn_name
  dst.is_exception_handler = src.is_exception_handler
  dst.exceptions = src.exceptions
  dst.reraise = src.reraise
  # Reset execution state
  dst.temp_args = None
  dst.next_link = None
  return dst


cdef Link _clone_chain_links(Link src):
  """Clone a linked list of Links starting at src. Returns the head of the cloned list."""
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
  """Create a temporary Link via __new__ (no __init__), used for root value overrides."""
  cdef Link link = Link.__new__(Link)
  link.v = v
  link.args = args
  link.kwargs = kwargs
  link.is_with_root = False
  link.ignore_result = False
  link.is_chain = False
  link.original_value = None
  link.fn_name = None
  link.temp_args = None
  link.is_exception_handler = False
  link.exceptions = None
  link.reraise = True
  link.next_link = None
  link.eval_code = _determine_eval_code(link, v, args, kwargs, True)
  return link


cdef object evaluate_value(Link link, object current_value):
  """Evaluate a single Link against the current pipeline value.

  Dispatches based on the link's eval_code to call the link's value with
  the appropriate arguments (current value, explicit args, or no args).
  Returns the raw result which may be a coroutine.
  """
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
