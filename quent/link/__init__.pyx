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
    self.is_attr = is_attr
    self.is_fattr = is_fattr
    self.next_link = None
    if bool(args):
      if args[0] is ...:
        self.eval_code = EVAL_NO_ARGS
      else:
        self.eval_code = EVAL_CUSTOM_ARGS
    elif bool(kwargs):
      self.eval_code =  EVAL_CUSTOM_ARGS
    elif callable(v):
      self.eval_code =  EVAL_CALLABLE
    elif is_fattr:
      self.eval_code = EVAL_NO_ARGS
    elif is_attr:
      self.eval_code = EVAL_ATTR
    else:
      if not allow_literal:
        raise QuentException('Non-callable objects cannot be used with this method.')
      self.eval_code = EVAL_LITERAL


cdef:
  int EVAL_CUSTOM_ARGS = 1001
  int EVAL_NO_ARGS = 1002
  int EVAL_CALLABLE = 1003
  int EVAL_LITERAL = 1004
  int EVAL_ATTR = 1005


cdef object evaluate_value(Link link, object cv):
  cdef object v
  if link.eval_code == EVAL_CALLABLE:
    # `cv is Null` is for safety; in most cases, it simply means that `v` is the root value.
    if cv is Null:
      return link.v()
    else:
      return link.v(cv)

  elif link.is_attr:
    v = getattr(cv, link.v)
    if link.eval_code == EVAL_NO_ARGS:
      return v()
    elif link.eval_code == EVAL_CUSTOM_ARGS:
      return v(*link.args, **link.kwargs)
    return v

  else:
    if link.eval_code == EVAL_CUSTOM_ARGS:
      # it is dangerous if one of those will be `None`, but it shouldn't be possible
      # as we only specify both or none.
      return link.v(*link.args, **link.kwargs)
    elif link.eval_code == EVAL_NO_ARGS:
      return link.v()
    else:
      return link.v
