cdef class Link:
  cdef object v
  cdef Link next_link
  cdef tuple args
  cdef dict kwargs
  cdef bint is_attr, is_fattr, is_with_root, ignore_result
  cdef int eval_code

cdef:
  int EVAL_CUSTOM_ARGS
  int EVAL_NO_ARGS
  int EVAL_CALLABLE
  int EVAL_LITERAL
  int EVAL_ATTR

cdef object evaluate_value(Link link, object cv)
