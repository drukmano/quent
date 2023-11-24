from quent.link cimport Link

cdef class _Null:
  pass

cdef class QuentException(Exception):
  pass

cdef _Null Null

cdef:
  type _PyCoroType
  type _CyCoroType

cdef inline bint iscoro(object obj):
  return type(obj) is _PyCoroType or type(obj) is _CyCoroType

cdef set task_registry

cdef int remove_task(object task)

cdef object ensure_future(object coro)

cdef object _handle_exception(object exc, list except_links, Link link, object rv, object cv, int idx)

cdef object create_chain_link_exception(Link link, object cv, int idx)
