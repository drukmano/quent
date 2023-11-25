from quent.quent cimport Link

cdef class _InternalQuentException(Exception):
  pass

cdef class _Return(_InternalQuentException):
  pass

cdef class _Break(_InternalQuentException):
  pass

cdef class _Continue(_InternalQuentException):
  pass

cdef Link build_conditional(object conditional, bint is_custom, bint not_, Link on_true, Link on_false)

cdef Link while_true(object fn, tuple args, dict kwargs)

cdef class _Generator:
  cdef object _chain_run, _fn
  cdef bint _ignore_result
  cdef tuple _run_args

cdef Link foreach(object fn, bint ignore_result)

cdef Link with_(Link link, bint ignore_result)
