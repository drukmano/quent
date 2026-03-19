# SPDX-License-Identifier: MIT
"""Evaluation dispatch and control flow handlers."""

from __future__ import annotations

import asyncio
import sys
from types import CoroutineType, GeneratorType
from typing import Any

from ._link import Link
from ._types import _EMPTY_TUPLE, Null, _Break, _Return

# Cache the private C-level function for zero-overhead event loop detection.
# Returns None instead of raising RuntimeError — avoids ~1-2μs exception overhead
# on the sync path. Stable across Python 3.10+ (used by uvloop, anyio, etc.).
_get_running_loop = getattr(asyncio, '_get_running_loop', None)


def _has_running_loop() -> bool:
  """Check if any async event loop is currently running.

  Detects asyncio, trio, and curio without importing them — uses
  ``sys.modules`` to check if they are already loaded, then probes
  their running-loop APIs. Zero overhead when a library is not loaded
  (~50ns dict lookup returning None).

  asyncio is checked first via the private C-level
  ``asyncio._get_running_loop()`` for performance. Falls back to
  ``asyncio.get_running_loop()`` if the private API is unavailable.
  """
  # asyncio (common case — fast path)
  if _get_running_loop is not None:
    if _get_running_loop() is not None:
      return True
  else:
    try:
      asyncio.get_running_loop()
      return True
    except RuntimeError:
      pass

  # trio — sys.modules lookup is a ~50ns dict get when trio is not loaded.
  # trio.lowlevel is imported during normal trio operation; if trio's loop
  # is running, this submodule will be present.
  _trio_lowlevel = sys.modules.get('trio.lowlevel')
  if _trio_lowlevel is not None:
    try:
      _trio_lowlevel.current_trio_token()
      return True
    except RuntimeError:
      pass

  # curio
  _curio_meta = sys.modules.get('curio.meta')
  if _curio_meta is not None:
    try:
      if _curio_meta.curio_running():
        return True
    except Exception:
      pass

  return False


# CPython CO_ITERABLE_COROUTINE flag, stable since 3.5. Verified through 3.14.
_CO_ITERABLE_COROUTINE = 0x100


# ---- Evaluation dispatch ----


def _isawaitable(value: Any) -> bool:
  """Fast awaitable check, replacing inspect.isawaitable().

  Short-circuits on the first isinstance check for the common sync case
  (~30ns vs ~380ns for inspect.isawaitable with ABC machinery).

  Handles all three awaitable types:
  1. Native coroutines (CoroutineType) — most common async case
  2. Generator-based coroutines decorated with @types.coroutine — have
     _CO_ITERABLE_COROUTINE flag but lack __await__
  3. Objects with __await__ method — Future, Task, custom awaitables

  The ``hasattr`` check is wrapped in try/except because objects with a
  ``__getattr__`` that raises non-AttributeError exceptions would otherwise
  propagate unexpectedly.
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

  This is the central dispatch that implements quent's **universal** calling
  conventions -- used by all pipeline steps (then, do, map, foreach_do,
  gather, with_, if_, finally_) and also by except handlers (where ``exc``
  is passed as the current value).  There are **2 rules**, applied in strict
  priority order (first match wins):

  1. **Explicit args/kwargs** — ``v(*args, **kwargs)``.  The current value
     is NOT implicitly passed.
  2. **Default passthrough** — ``v(current_value)`` if callable and
     current_value is not Null; ``v()`` if callable and Null; ``v`` as-is
     if not callable.

  A ``Q`` instance is callable and therefore follows these same 2 rules.
  Internally, when ``link.is_q`` is True, we call ``v._run()`` directly
  instead of ``v.run()`` so that ``_Return``/``_Break`` signals propagate to
  the outer pipeline rather than being trapped.  This is an implementation
  detail -- the user-visible calling convention is unchanged.

  The function is structured to front-load the hot common path: ~90% of
  pipeline steps are simple callables with no explicit args, no kwargs, and
  not nested pipelines.  Checking ``args``/``kwargs`` first (almost always
  None) quickly enters the no-args branch, avoiding the upfront tuple unpack
  and minimizing attribute lookups before the call.
  """
  # Hot path: no explicit args/kwargs (~90% of calls).
  # link.args and link.kwargs are None for most steps — the truthiness check
  # on a None slot is ~2ns, so this rejects the uncommon path very cheaply.
  if not link.args and not link.kwargs:
    if link.is_callable:
      return link.v(current_value) if current_value is not Null else link.v()
    if link.is_q:
      return link.v._run(current_value, None, None, is_nested=True)
    return link.v

  # Slow path: explicit args/kwargs provided.
  v = link.v

  # Nested pipeline with explicit args — dispatch to _run().
  if link.is_q:
    args = link.args
    run_value = args[0] if args else Null
    run_args = args[1:] if args else None
    return v._run(run_value, run_args, link.kwargs, is_nested=True)

  if not link.is_callable:
    msg = f'{v!r} is not callable but received {"arguments" if link.args else "keyword arguments"}'
    raise TypeError(msg)
  return v(*(link.args or _EMPTY_TUPLE), **link.kwargs) if link.kwargs else v(*(link.args or _EMPTY_TUPLE))


# ---- Control flow handlers ----


def _eval_signal_value(v: Any, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> Any:
  """Evaluate a control flow signal's value per the args calling conventions."""
  args = args or _EMPTY_TUPLE
  if args or kwargs:
    return v(*args, **kwargs) if kwargs else v(*args)
  return v() if callable(v) else v


def _handle_break_exc(exc: _Break, fallback: Any) -> Any:
  """Append the break value to fallback if one was provided, otherwise return fallback as-is."""
  if exc.value is Null:
    return fallback
  try:
    result = _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
  finally:
    exc.value = Null
    exc.signal_args = _EMPTY_TUPLE
    exc.signal_kwargs = None
  if _isawaitable(result):
    return _append_break_value_async(result, fallback)
  fallback.append(result)
  return fallback


async def _append_break_value_async(result: Any, fallback: list[Any]) -> list[Any]:
  """Await an async break value and append it to the partial results list."""
  resolved = await result
  fallback.append(resolved)
  return fallback


def _should_use_async_protocol(value: Any, sync_attr: str, async_attr: str) -> bool | None:
  """Determine whether to use the async or sync protocol for a dual-protocol object.

  Returns:
    True  — use async protocol (``async_attr`` present; use it)
    False — use sync protocol (only ``sync_attr`` present)
    None  — neither protocol found

  Logic:
  - Both protocols present: check ``_has_running_loop()``.  Running loop → True (async);
    no loop → False (sync).
  - Only async protocol present: return True.
  - Only sync protocol present: return False.
  - Neither present: return None.
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


def _handle_return_exc(exc: _Return, propagate: bool) -> Any:
  """Handle a _Return signal: re-raise if nested, otherwise extract the value."""
  if propagate:
    raise exc
  if exc.value is Null:
    return None
  try:
    result = _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
  finally:
    # Release references eagerly — the signal's value/args may hold large
    # callables or data objects that are no longer needed after evaluation.
    # Using finally ensures cleanup on both success and exception paths.
    exc.value = Null
    exc.signal_args = _EMPTY_TUPLE
    exc.signal_kwargs = None
  return result
