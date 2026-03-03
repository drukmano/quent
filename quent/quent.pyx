# quent.pyx — Hub file and shared definitions
#
# This is the compilation entry point for the quent Cython extension. It contains
# shared imports, sentinel values, and coroutine detection setup, then includes
# the implementation files in dependency order.
#
# File organization:
#   _link.pxi          — Link node, evaluation dispatch, clone utilities
#   _operators.pxi     — Comparison, type-check, sleep, and negation operators
#   _control_flow.pxi  — Signals, context managers, generators
#   _iteration.pxi     — foreach, filter, gather collection operations
#   _async_utils.pxi   — Async task lifecycle management
#   _diagnostics.pxi   — Exception handling, traceback augmentation, chain stringification
#   _chain_core.pxi    — Core Chain class (construction, execution, API methods)
#   _variants.pxi      — Cascade, FrozenChain, run

import sys
import asyncio
import logging
import time
import types
import functools
import warnings
import collections.abc
cimport cython

from quent._internal import __QUENT_INTERNAL__


_logger = logging.getLogger('quent')

@cython.final
@cython.no_gc
cdef class _Null:
  """Sentinel value representing the absence of a value in the chain."""
  def __repr__(self):
    """Return string representation of the Null sentinel."""
    return '<Null>'

cdef _Null Null = _Null()
PyNull = Null

cdef tuple EMPTY_TUPLE = ()
cdef dict EMPTY_DICT = {}

# same impl. of types.CoroutineType but for Cython coroutines.
async def _py_coro(): pass
cdef object _cy_coro(): return _py_coro()
cdef object _coro = _cy_coro()
cdef type _c_coro_type = type(_coro)
try:
  _coro.close()  # Prevent ResourceWarning
except SystemError:
  pass  # coverage.py's ctrace tracer conflicts with coroutine close during module init

cdef:
  type _PyCoroType = types.CoroutineType
  type _CyCoroType = _c_coro_type

cdef:
  object _asyncio_sleep = asyncio.sleep
  object _asyncio_get_running_loop_internal = asyncio._get_running_loop

# EvalCode enum is declared in quent.pxd

# --- Include implementation files ---
# Order matters: each file may reference symbols defined in preceding files.
# The .pxd provides forward type declarations, but function/method definitions
# must precede their first use.

include "_link.pxi"
include "_operators.pxi"
include "_control_flow.pxi"
include "_iteration.pxi"
include "_async_utils.pxi"
include "_diagnostics.pxi"
include "_chain_core.pxi"
include "_variants.pxi"
