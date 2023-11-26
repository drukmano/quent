from quent.quent cimport Link
from quent.custom cimport _Return

cdef object handle_return_exc(_Return exc, bint propagate)

cdef set task_registry

cdef void remove_task(object task)

cdef object ensure_future(object coro)

cdef object _handle_exception(object exc, list except_links, Link link, object rv, object cv, int idx)

cdef object create_chain_link_exception(Link link, object cv, int idx)
