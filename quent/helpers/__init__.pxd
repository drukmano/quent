from quent.quent cimport Chain, Link
from quent.custom cimport _Return

cdef set task_registry

cdef object ensure_future(object coro)

cdef object handle_return_exc(_Return exc, bint propagate)

cdef object remove_self_frames_from_traceback()

cdef Link _handle_exception(object exc, Chain chain, Link link)

cdef void modify_traceback(object exc, Chain chain, Link link)

cdef Link get_true_source_link(Link source_link)

cdef str make_indent(int nest_lvl)

cdef tuple stringify_chain(Chain chain, int nest_lvl = *, Link source_link = *, bint found_source_link = *)

cdef str format_link(Link link, int nest_lvl, Link source_link = *, bint found_source_link = *)

cdef str get_obj_name(object o)
