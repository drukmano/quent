cdef class Link:
  cdef object v
  cdef tuple args
  cdef dict kwargs
  cdef bint is_attr, is_fattr, is_with_root, ignore_result
  cdef int eval_code

cdef class _FrozenChain:
  cdef object _chain_run

cdef:
  int EVAL_UNKNOWN
  int EVAL_NULL
  int EVAL_CUSTOM_ARGS
  int EVAL_NO_ARGS
  int EVAL_CALLABLE
  int EVAL_LITERAL
  int EVAL_ATTR

cdef int get_eval_code(Link link)

cdef object evaluate_value(Link link, object cv)
