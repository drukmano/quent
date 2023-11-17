from quent.classes cimport Link

cdef class Chain:
  cdef:
    Link root_link, on_finally
    list links, except_links
    bint is_cascade, _autorun
    tuple current_conditional, on_true
    str current_attr

  cdef int init(self, object rv, tuple args, dict kwargs, bint is_cascade) except -1

  cdef int _then(self, Link link) except -1

  cdef object _run(self, object v, tuple args, dict kwargs)

  cdef int _if(self, object on_true, tuple args=*, dict kwargs=*, bint not_=*) except -1

  cdef int _else(self, object on_false, tuple args=*, dict kwargs=*) except -1

  cdef int set_conditional(self, object conditional, bint custom=*) except -1

  cdef int finalize(self) except -1

  cdef int finalize_conditional(self, object on_false=*, tuple args=*, dict kwargs=*) except -1

cdef class Cascade(Chain):
  pass

cdef class ChainAttr(Chain):
  pass

cdef class CascadeAttr(ChainAttr):
  pass

cdef class run:
  cdef public:
    object root_value
    tuple args
    dict kwargs

cdef Chain from_list(object cls, tuple links)
