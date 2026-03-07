"""Core types, evaluation primitives, and async helpers."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import sys
import threading
from collections.abc import Coroutine
from typing import Any


class _Null:
  """Sentinel for 'no value provided'. Distinct from None, which is a valid chain value."""

  __slots__ = ()
  _instance: _Null | None = None

  def __new__(cls) -> _Null:
    if cls._instance is None:
      cls._instance = object.__new__(cls)
    return cls._instance

  def __repr__(self) -> str:
    return '<Null>'

  def __copy__(self) -> _Null:
    return self

  def __deepcopy__(self, memo: dict[int, Any]) -> _Null:
    return self

  def __reduce__(self) -> str:
    return 'Null'


Null = _Null()


class QuentException(Exception):
  """Public exception type for quent-specific errors."""

  __slots__ = ()


class _ControlFlowSignal(Exception):
  """Base for control-flow exceptions used internally by Chain.

  Chain.return_() raises _Return to exit a chain early with a value.
  Chain.break_() raises _Break to exit a map/foreach loop.
  Both carry an optional value (with args/kwargs) that is lazily evaluated
  when the exception is caught.
  """

  __slots__ = ('args_', 'kwargs_', 'value')

  # Intentionally skips super().__init__() — these are internal control-flow
  # signals that are never displayed to users, so the standard Exception
  # args tuple and message string are unnecessary.
  def __init__(self, v: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    self.value = v
    self.args_ = args
    self.kwargs_ = kwargs


class _Return(_ControlFlowSignal):
  """Signal early return from a chain with an optional value."""

  __slots__ = ()


class _Break(_ControlFlowSignal):
  """Signal break from a map/foreach/filter iteration with an optional value."""

  __slots__ = ()


# Convention: passing Ellipsis (...) as the first argument to a chain operation
# means "call this callable with no arguments", overriding the default behavior
# of passing the current chain value as the first argument.


def _resolve_value(v: Any, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> Any:
  """Resolve a value per calling conventions."""
  args = args or ()
  kwargs = kwargs or {}
  if args and args[0] is ...:
    return v()
  if args or kwargs:
    return v(*args, **kwargs)
  return v() if callable(v) else v


def _handle_break_exc(exc: _Break, fallback: Any) -> Any:
  """Handle a _Break exception, returning the break value or fallback."""
  if exc.value is Null:
    return fallback
  return _resolve_value(exc.value, exc.args_, exc.kwargs_)


def _handle_return_exc(exc: _Return, propagate: bool) -> Any:
  """Handle a _Return exception, re-raising if in a nested chain."""
  if propagate:
    raise exc
  if exc.value is Null:
    return None
  return _resolve_value(exc.value, exc.args_, exc.kwargs_)


def _set_link_temp_args(exc: BaseException, link: Link, /, **kwargs: Any) -> None:
  """Attach debug info to an exception for traceback display.

  Records keyword-tagged values that were available when the exception occurred,
  keyed by the link's identity. Used by _traceback._format_link to show
  actual arguments in the chain visualization (e.g. current_value=, item=, index=).
  """
  if not hasattr(exc, '__quent_link_temp_args__'):
    exc.__quent_link_temp_args__ = {}  # type: ignore[attr-defined]
  # Keyed by id(link) so _traceback._format_link can match the right args
  # to the right link when rendering the chain visualization.
  exc.__quent_link_temp_args__[id(link)] = kwargs  # type: ignore[attr-defined]


# Eager task start (Python 3.14+) avoids event loop scheduling overhead.
if sys.version_info >= (3, 14):
  _create_task_fn = functools.partial(asyncio.create_task, eager_start=True)
else:
  _create_task_fn = asyncio.create_task

# Strong reference set prevents the event loop from dropping fire-and-forget tasks.
# See: https://stackoverflow.com/a/75941086
_task_registry: set[asyncio.Task[Any]] = set()
_task_registry_lock = threading.Lock()


def _ensure_future(coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
  """Schedule a coroutine as a fire-and-forget task with strong reference tracking."""
  try:
    task = _create_task_fn(coro)
  except RuntimeError:
    coro.close()
    raise
  with _task_registry_lock:
    _task_registry.add(task)
  # Auto-removes from registry on completion to avoid unbounded growth.
  task.add_done_callback(_task_registry_discard)
  return task


def _task_registry_discard(task: asyncio.Task[Any]) -> None:
  """Thread-safe removal of a completed task from the registry."""
  with _task_registry_lock:
    _task_registry.discard(task)


class Link:
  """A single operation in a chain. Forms a linked list via next_link.

  Slots:
    v: The executable value (may differ from what the user passed after wrapping).
    next_link: Pointer to the next Link in the chain.
    ignore_result: If True, the result of evaluating this link is discarded.
    args: Positional arguments for the call.
    kwargs: Keyword arguments for the call.
    original_value: The original value before wrapping (for traceback display).
  """

  __slots__ = (
    'args',
    'ignore_result',
    'is_chain',
    'kwargs',
    'next_link',
    'original_value',
    'v',
  )

  v: Any
  next_link: Link | None
  ignore_result: bool
  args: tuple[Any, ...] | None
  kwargs: dict[str, Any] | None
  original_value: Any
  is_chain: bool

  def __init__(
    self,
    v: Any,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    ignore_result: bool = False,
    original_value: Any | None = None,
  ) -> None:
    # Duck-typing: checks for the _is_chain class attribute that only Chain
    # sets, avoiding a circular import with _chain.py.
    try:
      self.is_chain = getattr(v, '_is_chain', False)
    except Exception:
      self.is_chain = False
    if self.is_chain:
      with contextlib.suppress(AttributeError, TypeError):
        v.is_nested = True
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.ignore_result = ignore_result
    self.next_link = None
    self.original_value = original_value


def _evaluate_value(link: Link, current_value: Any = Null) -> Any:
  """Core evaluation: resolve a link's value against the current pipeline value."""
  v, args, kwargs = link.v, link.args, link.kwargs

  # Nested chains must call _run() directly, bypassing run()'s
  # _ControlFlowSignal trap, so _Return/_Break propagate to the outer chain.
  if link.is_chain:
    if args and args[0] is ...:
      return v._run(Null, None, None)
    if args or kwargs:
      return v._run(args[0] if args else current_value, args[1:] if args else None, kwargs or {})
    return v._run(current_value, None, None)

  if args and args[0] is ...:
    return v()
  if args or kwargs:
    return v(*(args or ()), **(kwargs or {}))
  if callable(v):
    return v(current_value) if current_value is not Null else v()
  return v
