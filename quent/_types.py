# SPDX-License-Identifier: MIT
"""Core types, sentinels, exceptions, and constants."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, NamedTuple, NoReturn, Protocol

# ---- ExceptionGroup polyfill ----

# Python 3.10 lacks ExceptionGroup; provide a minimal stand-in so that
# gather() can wrap concurrent failures without version checks elsewhere.
# On 3.11+, the builtin ExceptionGroup is re-exported unconditionally so
# that importers never need a version check.
if sys.version_info < (3, 11):  # pragma: no cover

  class _ExceptionGroup(Exception):
    """Minimal ExceptionGroup polyfill for Python 3.10."""

    def __init__(self, message: str, exceptions: list[Exception]) -> None:
      if not exceptions:
        raise ValueError('second argument (exceptions) must be a non-empty sequence')
      for exc in exceptions:
        if not isinstance(exc, Exception):
          raise TypeError(f'Item {exc!r} in second argument (exceptions) is not an Exception')
      super().__init__(message)
      self._exceptions = tuple(exceptions)

    @property
    def exceptions(self) -> tuple[Exception, ...]:
      return self._exceptions

    def __repr__(self) -> str:
      return f'ExceptionGroup({self.args[0]!r}, {list(self._exceptions)!r})'

    def __str__(self) -> str:
      count = len(self._exceptions)
      s = 's' if count != 1 else ''
      types = ', '.join(type(e).__name__ for e in self._exceptions)
      return f'{self.args[0]}: [{types}] ({count} sub-exception{s})'

    def subgroup(
      self, condition: type[BaseException] | tuple[type[BaseException], ...] | Callable[[BaseException], bool]
    ) -> _ExceptionGroup | None:
      """Return a new ExceptionGroup containing only exceptions matching the condition, or None."""
      if isinstance(condition, (type, tuple)):
        matched = [e for e in self._exceptions if isinstance(e, condition)]
      else:
        matched = [e for e in self._exceptions if condition(e)]
      if not matched:
        return None
      return self.derive(matched)

    def split(
      self, condition: type[BaseException] | tuple[type[BaseException], ...] | Callable[[BaseException], bool]
    ) -> tuple[_ExceptionGroup | None, _ExceptionGroup | None]:
      """Split into (matching, rest) ExceptionGroups. Either may be None."""
      if isinstance(condition, (type, tuple)):
        match = [e for e in self._exceptions if isinstance(e, condition)]
        rest = [e for e in self._exceptions if not isinstance(e, condition)]
      else:
        match = [e for e in self._exceptions if condition(e)]
        rest = [e for e in self._exceptions if not condition(e)]
      return (
        self.derive(match) if match else None,
        self.derive(rest) if rest else None,
      )

    def derive(self, excs: list[Exception]) -> _ExceptionGroup:
      """Create a new ExceptionGroup with the same message but different exceptions."""
      eg = _ExceptionGroup(self.args[0], excs)
      eg.__traceback__ = self.__traceback__
      eg.__cause__ = self.__cause__
      eg.__context__ = self.__context__
      if hasattr(self, '__notes__'):
        eg.__notes__ = self.__notes__[:]  # type: ignore[attr-defined]
      return eg

  # Re-export under the canonical name so importers never need a version check.
  ExceptionGroup = _ExceptionGroup

else:
  # On 3.11+, use the builtin. The 'as ExceptionGroup' form makes this a
  # re-export visible to mypy (PEP 484).
  from builtins import ExceptionGroup as ExceptionGroup  # type: ignore[no-redef]


class QuentExcInfo(NamedTuple):
  """Exception context passed to except_() handlers.

  Except handlers receive a single ``QuentExcInfo`` instance as their
  current value (per the standard 2-rule calling convention).  Access
  the caught exception via ``.exc`` and the pipeline's evaluated root
  value via ``.root_value`` (normalized to ``None`` when absent).
  """

  exc: BaseException
  root_value: Any


# ---- Sentinel ----

_null_instance_created = False


class _Null:
  """Sentinel meaning "no value was provided."

  Distinct from ``None``, which is a perfectly valid pipeline value.
  ``Q(None)`` creates a pipeline with root value ``None``; ``Q()``
  creates a pipeline with no root value at all.

  This class is a singleton — use ``quent.Null`` directly.  Attempting to
  instantiate ``NullType()`` after the singleton has been created raises
  ``TypeError``.
  """

  __slots__ = ()

  def __new__(cls) -> _Null:
    if _null_instance_created:
      raise TypeError('_Null is a singleton — use quent.Null instead of instantiating _Null()')
    return object.__new__(cls)

  def __repr__(self) -> str:
    return '<Null>'

  def __copy__(self) -> _Null:
    return self

  def __deepcopy__(self, memo: dict[int, Any]) -> _Null:
    return self

  def __reduce__(self) -> tuple[Any, ...]:
    # Return the singleton directly on unpickling.
    return (_get_null, ())


# Bypass _Null.__new__ entirely so the singleton flag check is not triggered
# during initial creation.  After Null is created, set the flag so that any
# subsequent _Null() / NullType() call raises TypeError.
Null: _Null = object.__new__(_Null)
_null_instance_created = True


def _get_null() -> _Null:
  """Return the Null singleton — used as the pickle reconstructor for _Null."""
  return Null


# ---- Concurrent operation sentinel ----

# Sentinel for unprocessed slots in concurrent result arrays (_iter_ops, _gather_ops).
# Using Null would cause a double-invocation bug if user code ever
# returned the Null sentinel (even though it is not part of the public API).
_UNPROCESSED: object = object()


# ---- Exceptions ----


class QuentException(Exception):
  """Public exception type for quent-specific errors.

  Raised for violations of quent's runtime invariants. Common causes:

  - **Control flow signal escape**: ``Q.return_()`` or ``Q.break_()``
    used outside a pipeline or in an unsupported context (e.g., inside
    ``except_()`` or ``finally_()`` handlers).
  - **Duplicate handler registration**: calling ``except_()`` or ``finally_()``
    more than once on the same pipeline.
  - **Invalid break context**: ``Q.break_()`` used outside of a
    ``foreach``/``foreach_do`` iteration.
  """

  __slots__ = ()


class _ControlFlowSignal(BaseException):
  """Base for non-local control flow within pipelines.

  ``Q.return_()`` raises ``_Return`` to exit a pipeline early.
  ``Q.break_()`` raises ``_Break`` to exit a foreach/foreach_do loop.
  Both carry an optional value (with args/kwargs) that is lazily evaluated
  when the signal is caught — this avoids unnecessary work if the signal
  propagates through multiple nested pipelines before being handled.
  """

  __slots__ = (
    'signal_args',  # Positional arguments for the signal's value evaluation
    'signal_kwargs',  # Keyword arguments for the signal's value evaluation
    'value',
  )

  signal_args: tuple[Any, ...]
  signal_kwargs: dict[str, Any] | None
  value: Any

  def __init__(self, v: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    # Intentionally skip super().__init__() — we use custom slots instead of
    # Exception.args for internal data.  BaseException.__new__ still sets
    # self.args to an empty tuple, which is fine.
    self.value = v
    self.signal_args = args
    self.signal_kwargs = kwargs or None

  def __repr__(self) -> str:
    return f'{type(self).__name__}(value={self.value!r})'

  def __str__(self) -> str:
    return type(self).__name__


class _Return(_ControlFlowSignal):
  """Signal early return from a pipeline with an optional value."""

  __slots__ = ()


class _Break(_ControlFlowSignal):
  """Signal break from foreach/foreach_do iteration with an optional value."""

  __slots__ = ()


# ---- Shared constants ----

# Pre-allocated empty tuple to avoid per-call allocations in hot paths.
_EMPTY_TUPLE: tuple[Any, ...] = ()

# ---- Operation protocol ----


class _PipelineOp(Protocol):
  """Structural protocol for pipeline operation callables.

  All operation classes (``_IfOp``, ``_IterOp``, ``_ConcurrentIterOp``,
  ``_WithOp``, ``_ConcurrentGatherOp``, ``_DriveGenOp``) set ``_link_name`` as a slot
  attribute identifying the user-facing method name (e.g. ``'if_'``,
  ``'foreach'``, ``'gather'``).

  This protocol formalizes the duck-typed contract read by:
    - ``_viz._get_link_name()``
    - ``_viz._format_link()``
    - ``_engine._record_exception_source()``
    - ``_engine._should_defer_with()``

  Optional attributes (``_fns``, ``_concurrency``, ``_else_link``) are
  present only on specific operation classes and are read defensively
  via ``getattr`` with fallback defaults — they are intentionally not
  part of this protocol.
  """

  _link_name: str

  def _clone(self) -> _PipelineOp: ...


# ---- Copy prevention ----


class _UncopyableMixin:
  """Mixin that blocks shallow and deep copying for correctness.

  Q and Link are singly-linked lists.  A shallow copy would produce a
  broken object with shared node references; a deep copy is semantically
  undefined for objects containing arbitrary callables.  ``__copy__`` and
  ``__deepcopy__`` are therefore blocked unconditionally.

  For Q objects, use ``clone()`` to produce a correct independent copy.
  """

  __slots__ = ()

  def _raise_copy_error(self) -> NoReturn:
    """Raise TypeError with a descriptive message explaining why copying is blocked."""
    msg = (
      f'{type(self).__name__} objects cannot be copied with copy.copy()/copy.deepcopy(). '
      f'Use {type(self).__name__}.clone() instead.'
      if hasattr(self, 'clone')
      else f'{type(self).__name__} objects cannot be copied with copy.copy()/copy.deepcopy().'
    )
    raise TypeError(msg)

  def __copy__(self) -> NoReturn:
    self._raise_copy_error()

  def __deepcopy__(self, memo: Any = None) -> NoReturn:
    self._raise_copy_error()
