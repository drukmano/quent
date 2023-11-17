cimport cython
import functools


from quent.helpers cimport Null
from quent.evaluate cimport get_eval_code, EVAL_UNKNOWN


cdef object _PipeCls
try:
  from pipe import Pipe as _PipeCls
except ImportError:
  _PipeCls = None


@cython.freelist(256)
cdef class Link:
  def __init__(
    self, v, args=None, kwargs=None, is_attr=False, is_fattr=False, is_with_root=False, ignore_result=False,
    eval_code=EVAL_UNKNOWN
  ):
    if _PipeCls is not None and isinstance(v, _PipeCls):
      v = v.function
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.is_attr = is_attr or is_fattr
    self.is_fattr = is_fattr
    self.is_with_root = is_with_root
    self.ignore_result = ignore_result
    if eval_code == EVAL_UNKNOWN:
      self.eval_code = get_eval_code(self)
    else:
      self.eval_code = eval_code


cdef class _FrozenChain:
  def decorator(self):
    cdef object _chain_run = self._chain_run
    def _decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        return _chain_run(fn, args, kwargs)
      return wrapper
    return _decorator

  def __init__(self, _chain_run):
    self._chain_run = _chain_run

  def run(self, __v=Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs)

  def __call__(self, __v=Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs)
