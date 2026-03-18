# SPDX-License-Identifier: MIT
"""Pipeline execution context for cross-step data sharing via contextvars."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_MISSING: Any = object()

_ctx_store: ContextVar[dict[str, Any]] = ContextVar('quent_context')


def _ctx_set(key: str, value: Any) -> None:
  """Store a value in the pipeline execution context.

  Uses copy-on-write semantics: each set() creates a new dict rather than
  mutating in place. This ensures proper isolation when contextvars are
  copied to concurrent workers via copy_context().run() -- a worker that
  calls set() gets its own dict copy without affecting the parent or
  sibling workers.
  """
  try:
    old = _ctx_store.get()
    new = {**old, key: value}
  except LookupError:
    new = {key: value}
  _ctx_store.set(new)


def _ctx_get(key: str, default: Any = _MISSING) -> Any:
  """Retrieve a value from the pipeline execution context."""
  try:
    store = _ctx_store.get()
  except LookupError:
    if default is not _MISSING:
      return default
    raise KeyError(key) from None
  try:
    return store[key]
  except KeyError:
    if default is not _MISSING:
      return default
    raise
