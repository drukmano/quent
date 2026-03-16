# SPDX-License-Identifier: MIT
"""Core types, sentinels, exceptions, and constants."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, NamedTuple, NoReturn

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


class ChainExcInfo(NamedTuple):
  """Exception context passed to except_() handlers.

  Except handlers receive a single ``ChainExcInfo`` instance as their
  current value (per the standard 2-rule calling convention).  Access
  the caught exception via ``.exc`` and the chain's evaluated root
  value via ``.root_value`` (normalized to ``None`` when absent).
  """

  exc: BaseException
  root_value: Any


# ---- Sentinel ----

_null_instance_created = False


class _Null:
  """Sentinel meaning "no value was provided."

  Distinct from ``None``, which is a perfectly valid pipeline value.
  ``Chain(None)`` creates a chain with root value ``None``; ``Chain()``
  creates a chain with no root value at all.

  Pickling is blocked for consistency with Chain and Link (CWE-502).
  While Null itself contains no callables, allowing pickle round-trips
  of the sentinel could create false expectations about the serializability
  of quent objects in general.

  This class is a singleton — use ``quent.Null`` directly.  Attempting to
  instantiate ``NullType()`` after the singleton has been created raises
  ``TypeError``.
  """

  __slots__ = ()

  def __new__(cls) -> _Null:
    if _null_instance_created:
      raise TypeError('NullType is a singleton — use quent.Null instead of instantiating NullType()')
    return object.__new__(cls)

  def __repr__(self) -> str:
    return '<Null>'

  def __copy__(self) -> _Null:
    return self

  def __deepcopy__(self, memo: dict[int, Any]) -> _Null:
    return self

  def __reduce__(self, *_args: Any, **_kwargs: Any) -> NoReturn:
    raise TypeError('Null sentinel cannot be pickled. Use quent.Null directly instead of serializing it (CWE-502).')

  def __reduce_ex__(self, *_args: Any, **_kwargs: Any) -> NoReturn:
    raise TypeError('Null sentinel cannot be pickled. Use quent.Null directly instead of serializing it (CWE-502).')


# Bypass _Null.__new__ entirely so the singleton flag check is not triggered
# during initial creation.  After Null is created, set the flag so that any
# subsequent _Null() / NullType() call raises TypeError.
Null: _Null = object.__new__(_Null)
_null_instance_created = True


# ---- Exceptions ----


class QuentException(Exception):
  """Public exception type for quent-specific errors.

  Raised for violations of quent's runtime invariants. Common causes:

  - **Control flow signal escape**: ``Chain.return_()`` or ``Chain.break_()``
    used outside a chain or in an unsupported context (e.g., inside
    ``except_()`` or ``finally_()`` handlers).
  - **Duplicate handler registration**: calling ``except_()`` or ``finally_()``
    more than once on the same chain.
  - **Invalid break context**: ``Chain.break_()`` used outside of a
    ``foreach``/``foreach_do`` iteration.
  """

  __slots__ = ()


class _ControlFlowSignal(BaseException):
  """Base for non-local control flow within chains.

  ``Chain.return_()`` raises ``_Return`` to exit a chain early.
  ``Chain.break_()`` raises ``_Break`` to exit a foreach/foreach_do loop.
  Both carry an optional value (with args/kwargs) that is lazily evaluated
  when the signal is caught — this avoids unnecessary work if the signal
  propagates through multiple nested chains before being handled.
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
    self.signal_kwargs = kwargs


class _Return(_ControlFlowSignal):
  """Signal early return from a chain with an optional value."""

  __slots__ = ()


class _Break(_ControlFlowSignal):
  """Signal break from foreach/foreach_do iteration with an optional value."""

  __slots__ = ()


# ---- Unpickling prevention ----


class _UnpicklableMixin:
  """Mixin that blocks pickling for security (CWE-502).

  Both Chain and Link contain arbitrary callables whose execution during
  unpickling could lead to arbitrary code execution.  This mixin injects
  ``__reduce__`` and ``__reduce_ex__`` methods that raise ``TypeError``.
  """

  __slots__ = ()

  def _raise_pickle_error(self) -> NoReturn:
    """Raise TypeError with a descriptive message explaining why pickling is blocked."""
    msg = (
      f'{type(self).__name__} objects cannot be pickled. '
      f'{type(self).__name__}s contain arbitrary callables whose execution during '
      f'unpickling could lead to arbitrary code execution (CWE-502).'
    )
    raise TypeError(msg)

  def __reduce__(self, *_args: Any, **_kwargs: Any) -> NoReturn:
    self._raise_pickle_error()

  def __reduce_ex__(self, *_args: Any, **_kwargs: Any) -> NoReturn:
    self._raise_pickle_error()
