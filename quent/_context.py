# SPDX-License-Identifier: MIT
"""Pipeline execution context for cross-step data sharing via contextvars."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
  from ._q import Q

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


class _SetDescriptor:
  """Descriptor enabling dual-mode Q.set().

  Instance access -- ``q.set(key)`` or ``q.set(key, value)``:
    Appends a pipeline step that stores a value under *key* in the
    execution context.  With one arg, stores the current pipeline value;
    with two, stores the explicit *value*.  Current value is unchanged
    (like ``.do()``).  Returns the pipeline for fluent chaining.

  Class access -- ``Q.set(key, value)``:
    Stores an explicit *value* under *key* in the execution context.
    Not a pipeline step -- takes effect immediately. Returns ``None``.
  """

  @overload
  def __get__(self, obj: Q[Any], objtype: type | None = None) -> Callable[..., Q[Any]]: ...
  @overload
  def __get__(self, obj: None, objtype: type) -> Callable[[str, Any], None]: ...

  def __get__(self, obj: Any, objtype: Any = None) -> Any:
    if obj is not None:
      # Instance access: q.set('key') or q.set('key', value) -> pipeline step
      def instance_set(key: str, value: Any = _MISSING) -> Any:
        if value is _MISSING:

          def _store(cv: Any = None) -> None:
            _ctx_set(key, cv)

          _store.__qualname__ = _store.__name__ = f'set({key!r})'
        else:

          def _store(cv: Any = None) -> None:
            _ctx_set(key, value)

          _store.__qualname__ = _store.__name__ = f'set({key!r}, ...)'

        return obj._then(_store, (), {}, ignore_result=True)

      return instance_set
    else:
      # Class access: Q.set('key', value) -> immediate store
      def static_set(key: str, value: Any) -> None:
        _ctx_set(key, value)

      return static_set


class _GetDescriptor:
  """Descriptor enabling dual-mode Q.get().

  Instance access -- ``q.get(key)`` or ``q.get(key, default)``:
    Appends a pipeline step that retrieves the value stored under *key*
    from the execution context. The retrieved value replaces the current
    value (like ``.then()``). Returns the pipeline for fluent chaining.

  Class access -- ``Q.get(key)`` or ``Q.get(key, default)``:
    Retrieves a value from the execution context immediately.
    Not a pipeline step.
  """

  @overload
  def __get__(self, obj: Q[Any], objtype: type | None = None) -> Callable[..., Q[Any]]: ...
  @overload
  def __get__(self, obj: None, objtype: type) -> Callable[..., Any]: ...

  def __get__(self, obj: Any, objtype: Any = None) -> Any:
    if obj is not None:
      # Instance access: q.get('key') -> pipeline step
      def instance_get(key: str, default: Any = _MISSING) -> Any:
        def _retrieve(cv: Any = None) -> Any:
          return _ctx_get(key, default)

        _retrieve.__qualname__ = _retrieve.__name__ = f'get({key!r})'
        return obj._then(_retrieve, (), {})

      return instance_get
    else:
      # Class access: Q.get('key') -> immediate retrieval
      return _ctx_get
