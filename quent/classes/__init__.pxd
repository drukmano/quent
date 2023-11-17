cdef class Link:
  cdef object v
  cdef tuple args
  cdef dict kwargs
  cdef bint is_attr, is_fattr, is_with_root, ignore_result
  cdef int eval_code

cdef class _FrozenChain:
  cdef object _chain_run
