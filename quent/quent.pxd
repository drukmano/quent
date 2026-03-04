cimport cython

# --- Sentinel types ---

cdef class _Null:
  pass

cdef _Null Null

cdef class QuentException(Exception):
  pass

# --- Coroutine detection ---

cdef:
  type _PyCoroType
  type _CyCoroType

cdef inline bint iscoro(object obj) noexcept:
  return type(obj) is _PyCoroType or type(obj) is _CyCoroType

# --- Evaluation dispatch codes ---

cdef enum EvalCode:
  EVAL_CALL_WITH_EXPLICIT_ARGS = 1001
  EVAL_CALL_WITHOUT_ARGS = 1002
  EVAL_CALL_WITH_CURRENT_VALUE = 1003
  EVAL_RETURN_AS_IS = 1004

# --- Link ---

cdef class Link:
  cdef object v, original_value, exceptions
  cdef Link next_link
  cdef tuple args, temp_args
  cdef dict kwargs
  cdef bint is_with_root, ignore_result, is_chain, is_exception_handler, reraise
  cdef int eval_code
  cdef str fn_name

# --- Link utility functions ---

cdef inline int _determine_eval_code(Link link, object v, tuple args, dict kwargs, bint allow_literal)

cdef Link _clone_link(Link src)

cdef Link _clone_chain_links(Link src)

cdef Link _make_temp_link(object v, tuple args, dict kwargs)

cdef object evaluate_value(Link link, object current_value)

cdef object _eval_signal_value(object v, tuple args, dict kwargs)

# --- Operator wrappers ---

@cython.final
@cython.freelist(16)
cdef class _ExecCtx:
  cdef Link temp_root_link
  cdef dict link_results
  cdef dict link_temp_args

@cython.final
@cython.freelist(4)
@cython.no_gc
cdef class _Sleep:
  cdef float delay

@cython.final
@cython.freelist(4)
cdef class _ToThread:
  cdef object fn

@cython.final
cdef class _ChainCallWrapper:
  cdef object _chain_run, _fn

# --- Chain ---

cdef class Chain:
  cdef:
    Link root_link, first_link, on_finally_link
    Link current_link
    bint is_cascade, _autorun, is_nested, _debug
  cdef readonly:
    bint _is_simple, _is_sync

  cdef void init(self, object root_value, tuple args, dict kwargs, bint is_cascade)

  cdef void _then(self, Link link)

  cdef object _run(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain)

  cdef object _run_simple(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain)

# --- Chain variants ---

@cython.final
cdef class Cascade(Chain):
  pass

@cython.final
@cython.freelist(8)
cdef class _DescriptorWrapper:
  cdef object _fn
  cdef dict __dict__

@cython.final
@cython.freelist(4)
cdef class _FrozenChain:
  cdef object _chain_run
  cdef bint _autorun, _is_sync

@cython.final
@cython.freelist(4)
cdef class run:
  cdef public:
    object root_value
    tuple args
    dict kwargs

# --- Control flow signals ---

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

# --- Context managers and generators ---

@cython.final
@cython.freelist(4)
cdef class _Generator:
  cdef object _chain_run, _fn
  cdef bint _ignore_result
  cdef tuple _run_args

# --- Iteration operations ---

cdef Link foreach(object fn, bint ignore_result)

cdef Link filter_(object fn)

cdef Link gather_(tuple fns)

cdef Link foreach_indexed(object fn, bint ignore_result)

cdef Link with_(object fn, tuple args, dict kwargs, bint ignore_result)

@cython.final
@cython.freelist(8)
cdef class _Foreach:
  cdef object fn
  cdef bint ignore_result
  cdef Link link

@cython.final
@cython.freelist(8)
cdef class _Filter:
  cdef object fn
  cdef Link link

@cython.final
@cython.freelist(4)
cdef class _Gather:
  cdef tuple fns
  cdef Link link

@cython.final
@cython.freelist(8)
cdef class _ForeachIndexed:
  cdef object fn
  cdef bint ignore_result
  cdef Link link

@cython.final
@cython.freelist(4)
cdef class _With:
  cdef Link link
  cdef bint ignore_result
  cdef tuple args
  cdef dict kwargs

# --- Diagnostics declarations ---

cdef set task_registry

cdef object ensure_future(object coro)

cdef void _clean_chained_exceptions(object exc, set seen)

cdef object remove_self_frames_from_traceback()

cdef Link _handle_exception(object exc, Chain chain, Link link, _ExecCtx ctx)

cdef void modify_traceback(object exc, Chain chain, Link link, _ExecCtx ctx)

cdef Link get_true_source_link(Link source_link, _ExecCtx ctx)

cdef str make_indent(int nest_lvl)

cdef tuple stringify_chain(Chain chain, _ExecCtx ctx, int nest_lvl = *, Link source_link = *, bint found_source_link = *)

cdef str format_args(tuple args)

cdef str format_kwargs(dict kwargs)

cdef str format_link(Link link, _ExecCtx ctx, int nest_lvl, Link source_link = *, bint found_source_link = *)

cdef str get_obj_name(object obj)
