# _async_utils.pxi — Async task lifecycle management
#
# Provides the task registry and ensure_future helper that keep strong references
# to fire-and-forget asyncio tasks, preventing the event loop from dropping them
# under load. See https://stackoverflow.com/a/75941086
#
# Key components:
#   - task_registry: strong-reference set
#   - ensure_future: create and register an asyncio Task from a coroutine
#   - _get_registry_size: test-visible accessor for registry size

from asyncio import create_task as _create_task

cdef bint _HAS_EAGER_START = sys.version_info >= (3, 14)

# --- Async task management ---

# this holds a strong reference to all tasks that we create
# see: https://stackoverflow.com/a/75941086
# "... the asyncio loop avoids creating hard references (just weak) to the tasks,
# and when it is under heavy load, it may just "drop" tasks that are not referenced somewhere else."
cdef set task_registry = set()


cdef object ensure_future(object coro):
  cdef object task
  if _HAS_EAGER_START:
    task = _create_task(coro, eager_start=True)
  else:
    task = _create_task(coro)
  task_registry.add(task)
  task.add_done_callback(task_registry.discard)
  return task


def _get_registry_size():
  """Return the current size of the task registry (for testing)."""
  return len(task_registry)
