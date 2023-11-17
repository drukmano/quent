from quent.classes cimport Link

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
