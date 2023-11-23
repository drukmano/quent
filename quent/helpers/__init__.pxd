cdef object _ensure_future

from quent.link cimport Link

cdef class _Null:
  pass

cdef class QuentException(Exception):
  pass

cdef _Null Null

cdef bint isawaitable(object obj)

# this holds a strong reference to all tasks that we create
# see: https://stackoverflow.com/a/75941086
# "... the asyncio loop avoids creating hard references (just weak) to the tasks,
# and when it is under heavy load, it may just "drop" tasks that are not referenced somewhere else."
cdef set task_registry

cdef inline int remove_task(object task) except -1:
  # this may occur when asyncio.ensure_future() is called on a Task -
  # it returns the same Task as-is. and even though we are not registering
  # the callback if the task is already in `task_registry`, a race condition is possible.
  if task in task_registry:
    try:
      task_registry.remove(task)
    except KeyError:
      pass

cdef inline object ensure_future(object coro):
  cdef object task = _ensure_future(coro)
  if task not in task_registry:
    task_registry.add(task)
    task.add_done_callback(remove_task)
  return task

cdef object _handle_exception(object exc, list except_links, Link link, object rv, object cv, int idx)

cdef object create_chain_link_exception(Link link, object cv, int idx)
