# TODO cimport Chain (need .pxd)
from quent.quent cimport Chain
from quent.helpers cimport Null
from quent.classes cimport Link


cdef object _PipeCls
try:
  from pipe import Pipe as _PipeCls
except ImportError:
  _PipeCls = None


cdef:
  int EVAL_UNKNOWN = 0
  int EVAL_NULL = 1001
  int EVAL_CUSTOM_ARGS = 1002
  int EVAL_NO_ARGS = 1003
  int EVAL_CALLABLE = 1004
  int EVAL_LITERAL = 1005
  int EVAL_ATTR = 1006


cdef int get_eval_code(Link link) except -1:
  cdef object v = link.v
  if v is Null:
    return EVAL_NULL

  if isinstance(v, Chain):
    # TODO add `autorun_explicit` where if True then do not override this value here
    #  this allows for more granular control over nested chains and autorun policies
    #  (e.g. not wanting some nested chain to auto-run but also wanting to auto-run another nested chain)
    v.autorun(False)

  elif link.is_attr:
    if not link.is_fattr:
      return EVAL_ATTR

  # Ellipsis as the first argument indicates a void method.
  if link.args and link.args[0] is ...:
    return EVAL_NO_ARGS

  # if either are specified, we assume `v` is a function.
  elif link.args or link.kwargs:
    return EVAL_CUSTOM_ARGS

  elif link.is_fattr:
    return EVAL_NO_ARGS

  elif not link.is_attr and callable(v):
    return EVAL_CALLABLE

  else:
    return EVAL_LITERAL


cdef object evaluate_value(Link link, object cv):
  cdef object v = link.v
  cdef int eval_code = link.eval_code
  if eval_code == EVAL_UNKNOWN:
    link.eval_code = eval_code = get_eval_code(link)

  if eval_code == EVAL_NULL:
    return Null

  elif link.is_attr:
    v = getattr(cv, v)
    if not link.is_fattr:
      return v

  if eval_code == EVAL_NO_ARGS:
    return v()

  elif eval_code == EVAL_CUSTOM_ARGS:
    # it is dangerous if one of those will be `None`, but it shouldn't be possible
    # as we only specify both or none.
    return v(*link.args, **link.kwargs)

  elif link.is_fattr:
    return v()

  elif eval_code == EVAL_CALLABLE:
    # `cv is Null` is for safety; in most cases, it simply means that `v` is the root value.
    return v() if cv is Null else v(cv)

  else:
    return v
