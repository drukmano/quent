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
  cdef object v, result, ogv, exceptions
  cdef Link next_link
  cdef tuple args, temp_args
  cdef dict kwargs
  cdef bint is_attr, is_fattr, is_with_root, ignore_result, is_chain, is_exception_handler, raise_
  cdef int eval_code
  cdef str fn_name

cdef object evaluate_value(Link link, object cv)

cdef class Chain:
  cdef:
    Link root_link, first_link, on_finally_link, temp_root_link
    Link current_link
    bint is_cascade, _autorun, uses_attr, is_nested
    tuple current_conditional, on_true
    str current_attr

  cdef void init(self, object rv, tuple args, dict kwargs, bint is_cascade)

  cdef void _then(self, Link link)

  cdef object _run(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain)

  cdef void _if(self, Link on_true, bint not_ = *)

  cdef void _else(self, Link on_false)

  cdef void set_conditional(self, Link conditional, bint custom = *)

  cdef void finalize(self)

  cdef void finalize_conditional(self, Link on_false = *)

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
