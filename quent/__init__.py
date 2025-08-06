"""Quent — a high-performance chain interface for Python.

Provides transparent handling of both synchronous and asynchronous operations
through a fluent API. Chains automatically detect and handle coroutines,
wrapping them in Tasks without requiring explicit ``await`` statements.

Core types:
  Chain        — sequential pipeline where each operation receives the previous result
  Cascade      — like Chain, but every operation receives the root value
  ChainAttr    — Chain with attribute access support via __getattr__
  CascadeAttr  — Cascade with attribute access support
  FrozenChain  — immutable snapshot for safe repeated execution

Control flow:
  QuentException — base exception for chain-related errors
  Null           — sentinel value indicating "no value provided"
  run            — pipe syntax terminator (Chain(f1) | f2 | run())
"""
import importlib.metadata
from typing import Awaitable
from .quent import Chain, Cascade, ChainAttr, CascadeAttr, run, QuentException
from .quent import PyNull as Null
from .quent import _FrozenChain as FrozenChain

__version__ = importlib.metadata.version("quent")

type ResultOrAwaitable[T] = T | Awaitable[T]


__all__ = [
  'Chain', 'Cascade', 'ChainAttr', 'CascadeAttr', 'QuentException', 'run', 'ResultOrAwaitable', 'Null',
  'FrozenChain', '__version__'
]

# Patch TracebackException.__init__ to clean quent internal frames at display time.
# This must be pure Python (not Cython) because Cython-compiled functions don't
# participate in the descriptor protocol needed for Class.__init__ replacement.
import traceback as _traceback_module
from quent.quent import _clean_exc_chain

_original_te_init = _traceback_module.TracebackException.__init__

def _patched_te_init(self, exc_type, exc_value=None, exc_tb=None, **kwargs):
  if exc_value is not None and getattr(exc_value, '__quent__', False):
    _clean_exc_chain(exc_value)
    exc_tb = exc_value.__traceback__
  _original_te_init(self, exc_type, exc_value, exc_tb, **kwargs)

_traceback_module.TracebackException.__init__ = _patched_te_init
