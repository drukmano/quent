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

cdef:
  int EVAL_CUSTOM_ARGS
  int EVAL_NO_ARGS
  int EVAL_CALLABLE
  int EVAL_LITERAL
  int EVAL_ATTR

cdef class Link:
  cdef object v
  cdef Link next_link
  cdef tuple args
  cdef dict kwargs
  cdef bint is_attr, is_fattr, is_with_root, ignore_result, is_chain, is_exception_handler
  cdef int eval_code

cdef class ExceptLink(Link):
  cdef bint raise_, return_
  cdef object exceptions

cdef object evaluate_value(Link link, object cv)

cdef class Chain:
  cdef:
    Link root_link, first_link, on_finally_link
    Link current_link
    bint is_cascade, _autorun, uses_attr, is_nested
    tuple current_conditional, on_true
    str current_attr

  cdef void init(self, object rv, tuple args, dict kwargs, bint is_cascade)

  cdef void _then(self, Link link)

  cdef object _run_nested(self, object v, tuple args, dict kwargs)

  cdef object _run(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain)

  cdef void _if(self, object on_true, tuple args = *, dict kwargs = *, bint not_ = *)

  cdef void _else(self, object on_false, tuple args = *, dict kwargs = *)

  cdef void set_conditional(self, object conditional, bint custom = *)

  cdef void finalize(self)

  cdef void finalize_conditional(self, object on_false = *, tuple args = *, dict kwargs = *)

cdef class Cascade(Chain):
  pass

cdef class ChainAttr(Chain):
  pass

cdef class CascadeAttr(ChainAttr):
  pass

cdef class _FrozenChain:
  cdef object _chain_run

cdef class run:
  cdef public:
    object root_value
    tuple args
    dict kwargs
