# SPDX-License-Identifier: MIT
"""Core types, sentinels, exceptions."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, NamedTuple, NoReturn, Protocol

# Python 3.10 lacks ExceptionGroup — minimal polyfill so gather() can wrap
# concurrent failures without version checks elsewhere.
if sys.version_info < (3, 11):  # pragma: no cover

  class _ExceptionGroup(Exception):
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
      eg = _ExceptionGroup(self.args[0], excs)
      eg.__traceback__ = self.__traceback__
      eg.__cause__ = self.__cause__
      eg.__context__ = self.__context__
      if hasattr(self, '__notes__'):
        eg.__notes__ = self.__notes__[:]  # type: ignore[attr-defined]
      return eg

  ExceptionGroup = _ExceptionGroup

else:
  from builtins import ExceptionGroup as ExceptionGroup  # type: ignore[no-redef]


class QuentExcInfo(NamedTuple):
  """Exception context passed to except_() handlers as the current value."""

  exc: BaseException
  root_value: Any


# Sentinels — each guards a different "no value" boundary; conflating any two
# introduces subtle bugs. Kept in narrowest-scope module, listed here for reference:
#   Null          (this file)         — "no value provided"; Q() vs Q(None). Never exposed to user code.
#   _UNPROCESSED  (this file)         — "slot not yet filled" in concurrent result arrays.
#   _WITH_UNSET   (_with_ops.py)      — "body result not yet available" inside _WithOp.
#   _END          (_buffer_ops.py)    — "producer finished" queue protocol marker.

_null_instance_created = False


class _Null:
  """Singleton sentinel for "no value provided". Distinct from None.

  Q(None) creates a pipeline with root value None; Q() creates one with no root.
  Use ``quent.Null`` directly — instantiating _Null() raises TypeError.
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
    return (_get_null, ())


# Bypass _Null.__new__ for initial creation; flag the class afterward.
Null: _Null = object.__new__(_Null)
_null_instance_created = True


def _get_null() -> _Null:
  return Null


_UNPROCESSED: object = object()


class QuentException(Exception):
  """Public exception for quent-specific errors.

  Raised on: control flow signal escape (return_/break_ used outside a valid
  context), duplicate except_/finally_ registration, break_() outside a loop.
  """

  __slots__ = ()


class _ControlFlowSignal(BaseException):
  """Base for non-local control flow within pipelines.

  Q.return_() raises _Return to exit a pipeline early.
  Q.break_() raises _Break to exit a loop or iteration operation.
  Value/args are evaluated lazily when the signal is caught.
  """

  __slots__ = (
    'signal_args',
    'signal_kwargs',
    'value',
  )

  signal_args: tuple[Any, ...]
  signal_kwargs: dict[str, Any] | None
  value: Any

  def __init__(self, v: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    # Skip super().__init__() — we use slots, not Exception.args.
    self.value = v
    self.signal_args = args
    self.signal_kwargs = kwargs or None

  def __repr__(self) -> str:
    return f'{type(self).__name__}(value={self.value!r})'

  def __str__(self) -> str:
    return type(self).__name__


class _Return(_ControlFlowSignal):
  """Early return from the current pipeline (like Python's ``return``).

  Each Q boundary absorbs its own _Return; from a nested Q, only that Q returns.
  """

  __slots__ = ()


class _Break(_ControlFlowSignal):
  """Break from the nearest enclosing iteration scope (like a labeled ``break``).

  Propagates outward through Q boundaries until caught by an iteration scope.
  """

  __slots__ = ()


class _Exit(_ControlFlowSignal):
  """Hard exit from the entire pipeline, regardless of nesting depth.

  Propagates through every Q boundary and every signal trap (except_/finally_/
  gather/drive_gen carve-outs). Absorbed only at the outermost run().
  Standard try/finally semantics apply during propagation.
  """

  __slots__ = ()


_EMPTY_TUPLE: tuple[Any, ...] = ()


class _PipelineOp(Protocol):
  """Structural protocol for pipeline operation callables.

  All op classes (_IfOp, _IterOp, _ConcurrentIterOp, _WithOp,
  _ConcurrentGatherOp, _DriveGenOp) set ``_link_name`` as a slot attribute
  identifying the user-facing method name. Optional attrs (_fns,
  _concurrency, _else_link) are read defensively via getattr.
  """

  _link_name: str

  def _clone(self) -> _PipelineOp: ...


class _UncopyableMixin:
  """Blocks copy.copy / copy.deepcopy for correctness.

  Q and Link are singly-linked lists; shallow copy yields shared node refs,
  deep copy is undefined over arbitrary callables. Use Q.clone() instead.
  """

  __slots__ = ()

  def _raise_copy_error(self) -> NoReturn:
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
