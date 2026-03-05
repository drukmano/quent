import sys
import asyncio
import logging
import types
import functools
import warnings
import collections.abc
cimport cython

from quent._internal import __QUENT_INTERNAL__


_logger = logging.getLogger('quent')

# PERF: @cython.final prevents subclassing, enabling direct C method dispatch.
# @cython.no_gc disables GC tracking for this cycle-free sentinel. See PERFORMANCE.md #1, #3.
@cython.final
@cython.no_gc
cdef class _Null:
  """Sentinel value representing the absence of a value in the chain."""
  def __repr__(self):
    """Return string representation of the Null sentinel."""
    return '<Null>'

cdef _Null Null = _Null()
PyNull = Null

# PERF: Pre-allocated empty containers reused everywhere to avoid per-call allocation.
# See PERFORMANCE.md #6.
cdef tuple EMPTY_TUPLE = ()
cdef dict EMPTY_DICT = {}
# PERF: Cached module-level reference avoids types.MethodType attribute lookup per call.
# See PERFORMANCE.md #28.
cdef object _MethodType = types.MethodType

# same impl. of types.CoroutineType but for Cython coroutines.
async def _py_coro(): pass
cdef object _cy_coro(): return _py_coro()
cdef object _coro = _cy_coro()
cdef type _c_coro_type = type(_coro)
try:
  _coro.close()  # Prevent ResourceWarning
except SystemError:
  pass  # coverage.py's ctrace tracer conflicts with coroutine close during module init

# PERF: cdef type variables enable pointer-level identity checks in iscoro().
# See PERFORMANCE.md #4.
cdef:
  type _PyCoroType = types.CoroutineType
  type _CyCoroType = _c_coro_type


# --- Async await utility ---

async def _await_run(result, chain=None, link=None, ctx=None):
  try:
    return await result
  except BaseException as exc:
    if chain is not None and link is not None:
      modify_traceback(exc, chain, link, ctx)
    raise remove_self_frames_from_traceback()

# PERF: _await_run is called directly at all call sites instead of through a cdef object alias.
# This allows Cython to generate a direct coroutine creation call within the same compilation unit,
# avoiding __Pyx_PyObject_FastCall + PyMethod_Check overhead (score 20-34 per call).
# See PERFORMANCE.md #19.


# --- Execution context ---

# PERF: @cython.freelist(16) pools freed _ExecCtx objects for instant reuse.
# Only allocated on async transitions or exception paths (lazy allocation).
# See PERFORMANCE.md #2, #18.
@cython.final
@cython.freelist(16)
cdef class _ExecCtx:
  pass


# --- Signal value evaluation ---

cdef object _eval_signal_value(object v, tuple args, dict kwargs):
  if args:
    if args[0] is ...:
      # EVAL_CALL_WITHOUT_ARGS equivalent
      return v()
    else:
      # EVAL_CALL_WITH_EXPLICIT_ARGS equivalent
      if kwargs is None:
        kwargs = EMPTY_DICT
      if kwargs is EMPTY_DICT:
        return v(*args)
      return v(*args, **kwargs)
  elif kwargs:
    if args is None:
      args = EMPTY_TUPLE
    return v(*args, **kwargs)
  elif PyCallable_Check(v):
    # EVAL_CALL_WITH_CURRENT_VALUE with Null -> call without args
    return v()
  else:
    # EVAL_RETURN_AS_IS equivalent
    return v


# --- Signal exceptions ---

cdef class _InternalQuentException(Exception):
  def __init__(self, object __v, tuple args, dict kwargs):
    self.value = __v
    self.args_ = args
    self.kwargs_ = kwargs

  def __repr__(self):
    return f'<{type(self).__name__}>'


cdef class _Return(_InternalQuentException):
  pass


cdef class _Break(_InternalQuentException):
  pass


cdef object handle_break_exc(_Break exc, object fallback):
  if exc.value is Null:
    return fallback
  return _eval_signal_value(exc.value, exc.args_, exc.kwargs_)


cdef object handle_return_exc(_Return exc, bint propagate):
  if propagate:
    raise exc
  if exc.value is Null:
    return None
  return _eval_signal_value(exc.value, exc.args_, exc.kwargs_)


# --- Async task management ---

from asyncio import create_task as _create_task

# PERF: Eager task start (Python 3.14+) avoids event loop scheduling overhead.
# _create_task_fn wraps eager_start=True in a closure to avoid per-call kwargs dict construction.
# See PERFORMANCE.md #30.
cdef bint _HAS_EAGER_START = sys.version_info >= (3, 14)

# this holds a strong reference to all tasks that we create
# see: https://stackoverflow.com/a/75941086
# "... the asyncio loop avoids creating hard references (just weak) to the tasks,
# and when it is under heavy load, it may just "drop" tasks that are not referenced somewhere else."
cdef set task_registry = set()
# PERF: Cached bound method avoids creating a new bound method object per task callback.
# See PERFORMANCE.md #29.
cdef object _registry_discard = task_registry.discard

cdef object _create_task_og = _create_task
cdef object _create_task_eager(coro):
  return _create_task_og(coro, eager_start=True)
cdef object _create_task_fn
if _HAS_EAGER_START:
  _create_task_fn = _create_task_eager
else:
  _create_task_fn = _create_task

cdef object ensure_future(object coro):
  cdef object task = _create_task_fn(coro)
  # PERF: PySet_Add — direct C-level set insertion, bypasses set.add() method dispatch.
  # See PERFORMANCE.md #39.
  PySet_Add(task_registry, task)
  task.add_done_callback(_registry_discard)
  return task

def _get_registry_size():
  """Return the current size of the task registry (for testing)."""
  return len(task_registry)


# Order matters: each file may reference symbols defined in preceding files.
# The .pxd provides forward type declarations, but function/method definitions
# must precede their first use.

include "_link.pxi"
include "_with.pxi"
include "_generator.pxi"
include "_foreach.pxi"
include "_filter.pxi"
include "_gather.pxi"
include "_chain_wrappers.pxi"
include "_diagnostics.pxi"
include "_chain_core.pxi"
include "_frozen_chain.pxi"
