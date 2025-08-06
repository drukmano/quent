# quent.pyx — Hub file and shared definitions
#
# This is the compilation entry point for the quent Cython extension. It contains
# shared imports, sentinel values, and coroutine detection setup, then includes
# the implementation files in dependency order.
#
# File organization:
#   _link.pxi          — Link node, evaluation dispatch, clone utilities
#   _operators.pxi     — Comparison, type-check, sleep, and negation operators
#   _control_flow.pxi  — Signals, conditionals, loops, context managers, generators
#   _iteration.pxi     — foreach, filter, reduce, gather collection operations
#   _helpers.pxi       — Async utilities, exception handling, chain stringification
#   _chain.pxi         — Core Chain class (construction, execution, API methods)
#   _variants.pxi      — Cascade, ChainAttr, CascadeAttr, FrozenChain, run

import sys
import asyncio
import contextvars
import logging
import time
import types
import functools
import warnings
import collections.abc
cimport cython

from quent._internal import __QUENT_INTERNAL__


cdef object _PipeCls
try:
  from pipe import Pipe as _PipeCls
except ImportError:
  _PipeCls = None

_logger = logging.getLogger('quent')
cdef object _current_context = contextvars.ContextVar('quent_chain_context', default=None)

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
  object _asyncio_get_running_loop = asyncio.get_running_loop
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
include "_helpers.pxi"
include "_chain.pxi"
include "_variants.pxi"
