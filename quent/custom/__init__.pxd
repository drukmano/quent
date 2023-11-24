from quent.quent cimport Link

cdef Link build_conditional(object conditional, bint is_custom, bint not_, Link on_true, Link on_false)

cdef class _Generator:
  cdef object _chain_run, _fn
  cdef bint _ignore_result
  cdef tuple _run_args

cdef Link foreach(object fn, bint ignore_result)

cdef Link with_(Link link, bint ignore_result)
