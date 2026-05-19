# SPDX-License-Identifier: MIT
"""Evaluation dispatch and control flow handlers."""

from __future__ import annotations

import asyncio
import sys
from types import CoroutineType, GeneratorType
from typing import Any

from ._link import Link
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


def _isawaitable(value: Any) -> bool:
  """Fast awaitable check, replacing inspect.isawaitable() (~30ns vs ~380ns).

  Covers native coroutines, @types.coroutine generator coroutines, and
  __await__-bearing objects (Future/Task/custom).
  """
  if isinstance(value, CoroutineType):
    return True
  if isinstance(value, GeneratorType):
    return bool(value.gi_code.co_flags & _CO_ITERABLE_COROUTINE)
  try:
    return getattr(value, '__await__', None) is not None
  except Exception:
    return False


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


def _eval_signal_value(v: Any, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> Any:
  """Evaluate a control flow signal's value per the args calling convention."""
  args = args or _EMPTY_TUPLE
  if args or kwargs:
    return v(*args, **kwargs) if kwargs else v(*args)
  return v() if callable(v) else v


def _handle_break_exc(exc: _Break, fallback: Any) -> Any:
  """Append the break value to fallback if one was provided.

  Per spec §7.2: if the lazy callable raises a control-flow signal, wrap as
  QuentException (signals inside lazy values are misuse).
  """
  from ._types import QuentException as _QE

  if exc.value is Null:
    return fallback
  try:
    try:
      result = _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
    except _ControlFlowSignal as signal:
      raise _QE(
        f"Q.break_()'s lazy value raised {type(signal).__name__}; signals inside lazy values are misuse (per §7.2)."
      ) from signal
  finally:
    exc.value = Null
    exc.signal_args = _EMPTY_TUPLE
    exc.signal_kwargs = None
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
  """Extract and evaluate a _Return signal's value.

  Each Q boundary absorbs its own _Return; callers extract value via this
  helper and return it — the signal is not re-raised.

  Per spec §7.1: if the lazy callable itself raises a control-flow signal,
  wrap as QuentException (signals inside lazy values are misuse).
  """
  # Local import to avoid circular dependency at module load.
  from ._types import QuentException as _QE

  if exc.value is Null:
    return None
  try:
    try:
      result = _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
    except _ControlFlowSignal as signal:
      raise _QE(
        f"Q.return_()'s lazy value raised {type(signal).__name__}; signals inside lazy values are misuse (per §7.1)."
      ) from signal
  finally:
    # Release eagerly — value/args may hold large callables/data.
    exc.value = Null
    exc.signal_args = _EMPTY_TUPLE
    exc.signal_kwargs = None
  return result


def _handle_exit_exc(exc: _Exit) -> Any:
  """Extract and evaluate a _Exit signal's value (caught at outermost run()).

  Per spec §7.5: signal inside lazy value is misuse → QuentException.
  """
  from ._types import QuentException as _QE

  if exc.value is Null:
    return None
  try:
    try:
      result = _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
    except _ControlFlowSignal as signal:
      raise _QE(
        f"Q.exit_()'s lazy value raised {type(signal).__name__}; signals inside lazy values are misuse (per §7.5)."
      ) from signal
  finally:
    exc.value = Null
    exc.signal_args = _EMPTY_TUPLE
    exc.signal_kwargs = None
  return result
