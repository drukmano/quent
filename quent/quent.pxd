from cpython.object cimport PyCallable_Check
cimport cython

# --- Core types (quent.pyx) ---

cdef class _Null:
  pass

cdef _Null Null

cdef class QuentException(Exception):
  pass

cdef:
  type _PyCoroType
  type _CyCoroType

cdef inline bint iscoro(object obj) noexcept:
  return type(obj) is _PyCoroType or type(obj) is _CyCoroType

cdef enum EvalCode:
  EVAL_CALL_WITH_EXPLICIT_ARGS = 1001
  EVAL_CALL_WITHOUT_ARGS = 1002
  EVAL_CALL_WITH_CURRENT_VALUE = 1003
  EVAL_RETURN_AS_IS = 1004

# --- Link (_link.pxi) ---

cdef class Link:
  # Hot fields (accessed every iteration)
  cdef object v
  cdef Link next_link
  cdef int eval_code
  cdef bint is_chain, ignore_result
  # Warm fields
  cdef tuple args
  cdef dict kwargs
  # Cold fields
  cdef object original_value
  cdef tuple temp_args

cdef inline int _determine_eval_code(Link link, object v, tuple args, dict kwargs) noexcept

cdef Link _clone_link(Link src)

cdef Link _clone_chain_links(Link src)

cdef Link _make_temp_link(object v, tuple args, dict kwargs)

cdef object evaluate_value(Link link, object current_value)

# --- Execution context & signal helpers (quent.pyx) ---

@cython.final
@cython.freelist(16)
cdef class _ExecCtx:
  cdef Link temp_root_link
  cdef dict link_results
  cdef dict link_temp_args

cdef object _eval_signal_value(object v, tuple args, dict kwargs)

# --- Signal exceptions (quent.pyx) ---

cdef class _InternalQuentException(Exception):
  cdef object value
  cdef tuple args_
  cdef dict kwargs_

cdef class _Return(_InternalQuentException):
  pass

cdef class _Break(_InternalQuentException):
  pass

cdef object handle_break_exc(_Break exc, object fallback)

cdef object handle_return_exc(_Return exc, bint propagate)

# --- Async task management (quent.pyx) ---

cdef set task_registry

cdef object ensure_future(object coro)

# --- With (_with.pxi) ---

@cython.final
@cython.freelist(4)
cdef class _With:
  cdef Link link
  cdef bint ignore_result
  cdef tuple args
  cdef dict kwargs

cdef Link with_(object fn, tuple args, dict kwargs, bint ignore_result)

# --- Generator (_generator.pxi) ---

@cython.final
@cython.freelist(4)
cdef class _Generator:
  cdef object _chain_run, _fn
  cdef bint _ignore_result
  cdef tuple _run_args

# --- Foreach (_foreach.pxi) ---

cdef Link foreach(object fn, bint ignore_result)

@cython.final
@cython.freelist(8)
cdef class _Foreach:
  cdef object fn
  cdef bint ignore_result
  cdef Link link

# --- Filter (_filter.pxi) ---

cdef Link filter_(object fn)

@cython.final
@cython.freelist(8)
cdef class _Filter:
  cdef object fn
  cdef Link link

# --- Gather (_gather.pxi) ---

cdef Link gather_(tuple fns)

@cython.final
@cython.freelist(4)
cdef class _Gather:
  cdef tuple fns
  cdef Link link

# --- Chain wrappers (_chain_wrappers.pxi) ---

@cython.final
cdef class _ChainCallWrapper:
  cdef Chain _chain
  cdef object _fn

@cython.final
@cython.freelist(8)
cdef class _DescriptorWrapper:
  cdef object _fn
  cdef dict __dict__

# --- Chain (_chain_core.pxi) ---

@cython.final
@cython.freelist(32)
cdef class Chain:
  cdef:
    Link root_link, first_link, on_finally_link, on_except_link
    Link current_link
    object on_except_exceptions
    bint is_nested

  cdef void init(self, object root_value, tuple args, dict kwargs)

  cdef void _then(self, Link link)

  cdef object _run(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain)

# --- Diagnostics (_diagnostics.pxi) ---

cdef void _clean_chained_exceptions(object exc, set seen)

cdef object remove_self_frames_from_traceback()

cdef void modify_traceback(object exc, Chain chain, Link link, _ExecCtx ctx)

cdef Link get_true_source_link(Link source_link, _ExecCtx ctx)

cdef str make_indent(int nest_lvl)

cdef tuple stringify_chain(Chain chain, _ExecCtx ctx, int nest_lvl = *, Link source_link = *, bint found_source_link = *)

cdef str format_args(tuple args)

cdef str format_kwargs(dict kwargs)

cdef str _get_link_name(Link link)

cdef str format_link(Link link, _ExecCtx ctx, int nest_lvl, Link source_link = *, bint found_source_link = *, str method_name = *)

cdef str get_obj_name(object obj)
