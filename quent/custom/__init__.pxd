from quent.quent cimport Link, Null, Chain


cdef class _InternalQuentException(Exception):
  pass

cdef class _InternalQuentException_Custom(_InternalQuentException):
  cdef object _v
  cdef tuple args
  cdef dict kwargs

cdef class _Return(_InternalQuentException_Custom):
  pass

cdef class _Break(_InternalQuentException_Custom):
  pass

cdef object handle_break_exc(_Break exc, object nv)

cdef void build_conditional(Chain chain, Link conditional, bint is_custom, bint not_, Link on_true, Link on_false)

cdef Link while_true(object fn, tuple args, dict kwargs)

cdef class _Generator:
  cdef object _chain_run, _fn
  cdef bint _ignore_result
  cdef tuple _run_args

cdef Link foreach(object fn, bint ignore_result)

cdef Link with_(object fn, tuple args, dict kwargs, bint ignore_result)
