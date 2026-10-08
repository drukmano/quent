# SPDX-License-Identifier: MIT
"""Evaluation dispatch and control flow handlers."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable
from types import CoroutineType, GeneratorType
from typing import Any

from ._link import Link, _is_q
from ._types import _EMPTY_TUPLE, Null, _Break, _ControlFlowSignal, _Exit, _Return

# Private C-level fn for zero-overhead loop detection. Returns None instead of
# raising RuntimeError — avoids ~1-2μs exception overhead on the sync path.
_get_running_loop = getattr(asyncio, '_get_running_loop', None)


def _has_running_loop() -> bool:
  """True if asyncio/trio/curio has a running loop.

  sys.modules check is ~50ns dict lookup when the library isn't loaded.
  """
  if _get_running_loop is not None:
    if _get_running_loop() is not None:
      return True
  else:
    try:
      asyncio.get_running_loop()
      return True
    except RuntimeError:
      pass

  _trio_lowlevel = sys.modules.get('trio.lowlevel')
  if _trio_lowlevel is not None:
    try:
      _trio_lowlevel.current_trio_token()
      return True
    except RuntimeError:
      pass

  _curio_meta = sys.modules.get('curio.meta')
  if _curio_meta is not None:
    try:
      if _curio_meta.curio_running():
        return True
    except Exception:
      pass

  return False


# CPython CO_ITERABLE_COROUTINE flag, stable since 3.5.
_CO_ITERABLE_COROUTINE = 0x100
# Exact types only: subclasses may implement the await protocol.
_SYNC_TYPES = frozenset({int, str, float, bool, list, dict, tuple, set, bytes})


def _isawaitable(value: Any) -> bool:
  """Awaitability detection with fast rejection of common synchronous results.

  Covers native coroutines, @types.coroutine generator coroutines, and
  __await__-bearing objects (Future/Task/custom).
  """
  if value is None or type(value) in _SYNC_TYPES:
    return False
  if isinstance(value, CoroutineType):
    return True
  if isinstance(value, GeneratorType):
    return bool(value.gi_code.co_flags & _CO_ITERABLE_COROUTINE)
  # Special methods belong to the type; instance attribute probing can both
  # invent awaitability for data objects and hide a valid __await__ method.
  return isinstance(value, Awaitable)


def _evaluate_value(link: Link, current_value: Any = Null) -> Any:
  """Resolve a link's value against the current pipeline state.

  Central dispatch implementing the universal calling convention — used by
  every pipeline step and by except handlers (with exc as current value).
  Two rules, first match wins:

  1. Explicit args/kwargs → v(*args, **kwargs). Current value NOT passed.
  2. Default → v(current_value) if callable and CV not Null; v() if Null;
     v as-is if not callable.

  Q instances are callable and follow the same rules; we call v._run() so
  _Return/_Break propagate to the outer pipeline instead of being trapped.

  Hot path first: ~90% of steps are simple callables with no args/kwargs.
  """
  # ~90% of calls: no explicit args/kwargs. None-slot truthiness check is ~2ns.
  if not link.args and not link.kwargs:
    if link.is_callable:
      return link.v(current_value) if current_value is not Null else link.v()
    if link.is_q:
      return link.v._run(current_value, None, None, is_nested=True)
    return link.v

  v = link.v

  if link.is_q:
    args = link.args
    run_value = args[0] if args else Null
    run_args = args[1:] if args else None
    return v._run(run_value, run_args, link.kwargs, is_nested=True)

  if not link.is_callable:
    msg = f'{v!r} is not callable but received {"arguments" if link.args else "keyword arguments"}'
    raise TypeError(msg)
  return v(*(link.args or _EMPTY_TUPLE), **link.kwargs) if link.kwargs else v(*(link.args or _EMPTY_TUPLE))


def _bind_callback(fn: Any) -> Any:
  """Resolve nested-Q dispatch once; ordinary callbacks retain their raw call path."""
  if not _is_q(fn):
    return fn

  def nested(value: Any) -> Any:
    return fn._run(value, None, None, is_nested=True)

  return nested


def _eval_signal_value(v: Any, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> Any:
  """Evaluate a control flow signal's value per the args calling convention."""
  args = args or _EMPTY_TUPLE
  if args or kwargs:
    return v(*args, **kwargs) if kwargs else v(*args)
  return v() if callable(v) else v


def _resolve_signal_value(exc: _ControlFlowSignal, *, guard_awaitable: bool = True) -> Any:
  """Resolve a lazy signal value, guarding both invocation and asynchronous work.

  Callers needing a no-value fallback must check exc.value before calling.
  Payloads are cleared even when invocation fails. A synchronous iterator that
  immediately rejects awaitables may disable wrapping so it can close the original.
  """
  from ._types import QuentException

  name, section = (
    ('return_', '7.1')
    if isinstance(exc, _Return)
    else ('break_', '7.2')
    if isinstance(exc, _Break)
    else ('exit_', '7.5')
  )
  try:
    try:
      result = None if exc.value is Null else _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
    except _ControlFlowSignal as signal:
      raise QuentException(
        f"Q.{name}()'s lazy value raised {type(signal).__name__}; "
        f'signals inside lazy values are misuse (per §{section}).'
      ) from signal
  finally:
    exc.value = Null
    exc.signal_args = _EMPTY_TUPLE
    exc.signal_kwargs = None
  if guard_awaitable and _isawaitable(result):
    return _await_signal_value(result, exc, name, section)
  return result


async def _await_signal_value(result: Any, original: _ControlFlowSignal, name: str, section: str) -> Any:
  from ._types import QuentException

  try:
    return await result
  except _ControlFlowSignal as signal:
    if signal.__context__ is None:
      signal.__context__ = original
    raise QuentException(
      f"Q.{name}()'s lazy value raised {type(signal).__name__}; signals inside lazy values are misuse (per §{section})."
    ) from signal

  except BaseException as exc:
    # A sync-origin coroutine runs after the catch frame has returned. Restore
    # the signal context Python supplies naturally when evaluation stays inline.
    if exc.__context__ is None:
      exc.__context__ = original
    raise


def _handle_break_exc(exc: _Break, fallback: Any) -> Any:
  """Resolve and append a supplied break value; preserve the no-value fallback."""
  if exc.value is Null:
    return fallback
  result = _resolve_signal_value(exc)
  if _isawaitable(result):
    return _append_break_value_async(result, fallback)
  fallback.append(result)
  return fallback


async def _append_break_value_async(result: Any, fallback: list[Any]) -> list[Any]:
  resolved = await result
  fallback.append(resolved)
  return fallback


def _should_use_async_protocol(value: Any, sync_attr: str, async_attr: str) -> bool | None:
  """For dual-protocol objects, decide sync vs async.

  Returns True=use async, False=use sync, None=neither protocol present.
  Both present + running loop → async; both present + no loop → sync.
  """
  has_sync = hasattr(value, sync_attr)
  has_async = hasattr(value, async_attr)
  if has_sync and has_async:
    return _has_running_loop()
  if has_async:
    return True
  if has_sync:
    return False
  return None


def _handle_return_exc(exc: _Return) -> Any:
  """Resolve an absorbed return signal's lazy value."""
  return _resolve_signal_value(exc)


def _handle_exit_exc(exc: _Exit) -> Any:
  """Resolve an exit signal absorbed at the outermost execution boundary."""
  return _resolve_signal_value(exc)
