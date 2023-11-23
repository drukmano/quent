cimport cython
from quent.helpers cimport Null, QuentException


cdef object _PipeCls
try:
  from pipe import Pipe as _PipeCls
except ImportError:
  _PipeCls = None


@cython.freelist(32)
cdef class Link:
  def __init__(
    self, object v, tuple args = None, dict kwargs = None, bint allow_literal = False, bint is_with_root = False,
    bint ignore_result = False, bint is_attr = False, bint is_fattr = False
  ):
    if _PipeCls is not None and isinstance(v, _PipeCls):
      v = v.function
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.is_with_root = is_with_root
    self.ignore_result = ignore_result
    self.is_attr = is_attr or is_fattr
    self.is_fattr = is_fattr
    self.eval_code = get_eval_code(self)
    if not allow_literal and self.eval_code == EVAL_LITERAL:
      raise QuentException('Non-callable objects cannot be used with this method.')


cdef:
  int EVAL_CUSTOM_ARGS = 1001
  int EVAL_NO_ARGS = 1002
  int EVAL_CALLABLE = 1003
  int EVAL_LITERAL = 1004
  int EVAL_ATTR = 1005


cdef int get_eval_code(Link link) except -1:
  if bool(link.args):
    if link.args[0] is ...:
      return EVAL_NO_ARGS
    return EVAL_CUSTOM_ARGS

  elif bool(link.kwargs):
    return EVAL_CUSTOM_ARGS

  elif callable(link.v):
    return EVAL_CALLABLE

  elif link.is_fattr:
    return EVAL_NO_ARGS

  elif link.is_attr:
    return EVAL_ATTR

  else:
    return EVAL_LITERAL


cdef object evaluate_value(Link link, object cv):
  cdef object v = link.v
  cdef int eval_code = link.eval_code

  if eval_code == EVAL_CALLABLE:
    # `cv is Null` is for safety; in most cases, it simply means that `v` is the root value.
    return v() if cv is Null else v(cv)

  elif link.is_attr:
    v = getattr(cv, v)

  if eval_code == EVAL_CUSTOM_ARGS:
    # it is dangerous if one of those will be `None`, but it shouldn't be possible
    # as we only specify both or none.
    return v(*link.args, **link.kwargs)

  elif eval_code == EVAL_NO_ARGS:
    return v()

  elif eval_code == EVAL_ATTR:
    return v

  else:
    return v
