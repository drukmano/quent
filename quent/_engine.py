# SPDX-License-Identifier: MIT
"""Execution engine -- sync execution path, shared helpers, and re-exports.

The sync execution entry point (``_run``) and all shared helpers (step
recording, exception handling, null normalization, etc.) live here.  Async
execution paths (``_run_async``, ``_run_async_finally``, etc.) are in
``_engine_async.py`` and re-exported from this module for backward
compatibility.

Debug logging is available via the ``'quent'`` logger
(``logging.getLogger('quent')``).  Set to DEBUG to see step-by-step pipeline
execution traces including link results and async transitions.
"""

from __future__ import annotations

import itertools
import logging
import sys
import time
import warnings
from types import CoroutineType
from typing import TYPE_CHECKING, Any, NamedTuple

from ._eval import (
  _evaluate_value,
  _handle_return_exc,
  _isawaitable,
)
from ._exc_meta import META_SOURCE_LINK, _clean_exc_meta, _clean_quent_idx, _get_exc_meta, _set_link_temp_args
from ._link import Link
from ._traceback import _modify_traceback, _user_stacklevel
from ._types import (
  Null,
  QuentException,
  QuentExcInfo,
  _Break,
  _ControlFlowSignal,
  _Return,
)
from ._viz import _MAX_REPR_LEN, _get_link_name, _sanitize_repr, _show_traceback_values

if TYPE_CHECKING:
  from ._q import Q
  from ._types import _PipelineOp

_log = logging.getLogger('quent')

# Common sync return types — O(1) frozenset lookup (~30ns) avoids the full
# _isawaitable call (~350ns) for the ~90% of steps that return basic types.
_SYNC_TYPES = frozenset({int, str, float, bool, list, dict, tuple, set, bytes})
_perf_counter_ns = time.perf_counter_ns
_DEBUG_LEVEL = logging.DEBUG
_exec_counter = itertools.count()

# Free-threaded Python (PEP 703) fallback: itertools.count() is atomic under
# the GIL but not under free-threaded builds.  Detect via sys._is_gil_enabled
# (available in Python 3.13+).  When the GIL is disabled, fall back to a
# threading.Lock for correctness (the counter is only used for debug log
# correlation, so the overhead is acceptable).
_gil_enabled: bool = True
try:
  _gil_enabled = sys._is_gil_enabled()  # type: ignore[attr-defined]
except AttributeError:
  _gil_enabled = True  # Pre-3.13: GIL is always enabled

if not _gil_enabled:
  import threading

  _exec_counter_lock = threading.Lock()
  _exec_counter_value = 0


def _next_exec_id() -> int:
  """Return a unique execution ID, safe under free-threaded Python (PEP 703).

  Masked to 6 hex digits (0x000000-0xFFFFFF) per spec 14.5, wrapping after
  16,777,215 executions to maintain the fixed-width ``[exec:XXXXXX]`` format.

  Uses ``itertools.count()`` (C-level atomic under the GIL) for zero-overhead
  ID generation.  Falls back to a ``threading.Lock`` when the GIL is disabled
  (free-threaded Python) for correctness.
  """
  if _gil_enabled:
    return next(_exec_counter) & 0xFFFFFF
  global _exec_counter_value
  with _exec_counter_lock:
    val = _exec_counter_value
    _exec_counter_value = (val + 1) & 0xFFFFFF
  return val


_ON_STEP_NOT_RESOLVED = object()


def _timing_ctx(q: Q[Any], on_step: Any = _ON_STEP_NOT_RESOLVED) -> tuple[Any, bool, bool]:
  """Return (on_step, debug, needs_timing) for the given pipeline.

  ``on_step`` may be passed in from ``_run``'s call to ``_run_async`` (which
  already resolved it) to avoid a redundant ``type(q).on_step`` lookup.
  """
  _on_step = type(q).on_step if on_step is _ON_STEP_NOT_RESOLVED else on_step
  _debug = _log.isEnabledFor(_DEBUG_LEVEL)
  return _on_step, _debug, _on_step is not None or _debug


def _signal_in_handler_msg(signal: _ControlFlowSignal, handler: str) -> str:
  """Build the error message for _ControlFlowSignal caught inside except/finally handlers."""
  return f'Using {type(signal).__name__} inside {handler} handlers is not allowed.'


# ---- Null normalization ----


def _null_to_none(value: Any) -> Any:
  """Normalize Null to None for external consumption."""
  return None if value is Null else value


def _warn_except_handler_failed(original_exc: BaseException, handler_exc: BaseException) -> None:
  """Log and warn when an except handler (reraise=True) fails; attach note if supported."""
  sanitized = _sanitize_repr(repr(handler_exc))
  _log.warning('except handler (reraise=True) failed: %s', sanitized)
  warnings.warn(
    f'quent: except handler (reraise=True) failed and its error was discarded: {sanitized}',
    RuntimeWarning,
    stacklevel=_user_stacklevel(),
  )
  if hasattr(original_exc, 'add_note'):
    original_exc.add_note(f'quent: except handler (reraise=True) also failed: {sanitized}')


def _except_handler_body(
  exc: BaseException,
  q: Q[Any],
  link: Link | None,
  root_link: Link | None,
  root_value: Any = Null,
  is_nested: bool = False,
) -> Any:
  """Evaluate the pipeline's except handler and return its result.

  Uses the unified calling convention via ``_evaluate_value`` with a
  ``QuentExcInfo`` as the current value.  The standard 2-rule dispatch
  applies naturally:

  1. **Explicit args/kwargs** — ``handler(*args, **kwargs)``; ``exc_info``
     is NOT passed (same as standard convention).
  2. **Default** — ``handler(exc_info)``; ``exc_info`` is the current value.
     For nested pipelines: ``handler_pipeline._run(exc_info, None, None)``; with
     explicit args: ``handler_pipeline._run(args[0], args[1:], kwargs)``.
  """
  __tracebackhide__ = True
  try:
    _modify_traceback(exc, q, link, root_link, is_nested=is_nested)
  except Exception as e:
    _clean_exc_meta(exc)
    warnings.warn(f'quent: traceback enhancement failed: {e!r}', RuntimeWarning, stacklevel=_user_stacklevel())
  if q._on_except_link is None:
    raise exc
  if q._on_except_exceptions is None:  # invariant: set when _on_except_link is set
    raise QuentException('_on_except_exceptions must be set when _on_except_link is set')
  if not isinstance(exc, q._on_except_exceptions):
    raise exc
  except_link = q._on_except_link
  try:
    # Unified convention: QuentExcInfo is the "current value" for the handler.
    # _evaluate_value handles nested pipelines, explicit args, and default dispatch.
    exc_info = QuentExcInfo(exc=exc, root_value=_null_to_none(root_value))
    result = _evaluate_value(except_link, exc_info)
  except _ControlFlowSignal as signal:
    if signal.__context__ is None:
      signal.__context__ = exc
    raise QuentException(_signal_in_handler_msg(signal, 'except')) from exc
  except BaseException as exc_:
    _set_link_temp_args(exc_, q._on_except_link, exc=exc)
    _modify_traceback(exc_, q, q._on_except_link, root_link, is_nested=is_nested)
    raise exc_ from exc
  return result


def _handle_finally_exc(
  exc: BaseException,
  q: Q[Any],
  finally_link: Link,
  root_link: Link | None,
  root_value: Any,
  is_nested: bool,
) -> None:
  """Handle an exception from a finally handler -- attach metadata and re-raise.

  Handles both ``_ControlFlowSignal`` (converted to ``QuentException``) and
  general ``BaseException`` (stamps exception source, modifies traceback).
  Shared by ``_finally_handler_body`` and ``_await_finally_result``.
  """
  if isinstance(exc, _ControlFlowSignal):
    raise QuentException(_signal_in_handler_msg(exc, 'finally')) from None
  if root_value is not Null:
    _set_link_temp_args(exc, finally_link, root_value=root_value)
  try:
    _modify_traceback(exc, q, finally_link, root_link, is_nested=is_nested)
  except Exception as e:
    warnings.warn(f'quent: traceback enhancement failed: {e!r}', RuntimeWarning, stacklevel=_user_stacklevel())
  raise exc


def _finally_handler_body(q: Q[Any], root_value: Any, root_link: Link | None, is_nested: bool = False) -> Any:
  """Evaluate the pipeline's finally handler and return its result."""
  __tracebackhide__ = True
  assert q._on_finally_link is not None  # guaranteed by caller
  _finally_link = q._on_finally_link
  try:
    return _evaluate_value(_finally_link, _null_to_none(root_value))
  except BaseException as exc_:
    _handle_finally_exc(exc_, q, _finally_link, root_link, root_value, is_nested)


async def _await_finally_result(
  finally_result: Any,
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  is_nested: bool = False,
) -> None:
  """Await an async finally handler result with proper error handling.

  Extracted from _run_async's finally block to reduce nesting. Handles:
  - Awaiting the finally handler result
  - Catching _ControlFlowSignal (raises QuentException)
  - Catching BaseException (stamps exception source, modifies traceback)
  """
  __tracebackhide__ = True
  assert q._on_finally_link is not None  # guaranteed by caller
  _finally_link = q._on_finally_link
  try:
    await finally_result
  except BaseException as exc_:
    _handle_finally_exc(exc_, q, _finally_link, root_link, root_value, is_nested)


def _run_sync_finally(
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  active_exc: BaseException | None,
  is_nested: bool = False,
  exec_id: int = 0,
) -> Any:
  """Execute the finally handler in the sync execution path.

  Returns the finally handler's awaitable result if it returned a coroutine
  (for the caller to wrap in ``_async_finally_transition``), or ``None`` if
  the handler completed synchronously.
  """
  __tracebackhide__ = True
  _on_step, _debug, _needs_timing = _timing_ctx(q)
  try:
    if _needs_timing:
      _t0_fin = _perf_counter_ns()
    result = _finally_handler_body(q, root_value, root_link, is_nested=is_nested)
    if _needs_timing:
      _record_finally_step(
        q,
        root_value,
        root_link,
        result,
        _t0_fin,
        _on_step,
        _debug,
        exec_id=exec_id,
      )
    if _isawaitable(result):
      return result  # Return the awaitable for the caller to wrap
    return None
  except BaseException as finally_exc:
    if active_exc is not None:
      _chain_finally_exc(finally_exc, active_exc)
    raise


def _record_exception_source(exc: BaseException, link: Link | None, current_value: Any) -> None:
  """Stamp the failing link onto the exception for traceback display.

  First-write-wins: only the innermost failing link is recorded. Temp args
  are attached when the link has no explicit args and isn't a pipeline or
  operation, so the traceback can show what value was flowing through.

  Also cleans the ad-hoc ``_quent_idx`` attribute that concurrent workers
  may have attached — this is defense-in-depth to ensure it never leaks
  onto exceptions visible to user code, even for BaseException subclasses
  that bypass ``_modify_traceback`` / ``_clean_exc_meta``.
  """
  __tracebackhide__ = True
  _clean_quent_idx(exc)
  meta = _get_exc_meta(exc)
  if meta.get(META_SOURCE_LINK) is None and link is not None:
    meta[META_SOURCE_LINK] = link
  if link is None:
    return
  # Attach current_value as temp args only for plain user callables (then/do steps).
  # Operations (conforming to _PipelineOp, identifiable by _link_name) stamp their
  # own specialized temp args.
  op: _PipelineOp | Any = link.v
  if (
    current_value is not Null
    and not link.args
    and not link.kwargs
    and not link.is_q
    and not getattr(op, '_link_name', None)
  ):
    _set_link_temp_args(exc, link, current_value=current_value)


def _handle_base_exception(exc: BaseException, link: Link | None, current_value: Any) -> None:
  """Handle an exception from pipeline execution: record source or clean up.

  For regular exceptions, records the failing link via
  ``_record_exception_source`` for traceback display.  For
  ``KeyboardInterrupt`` / ``SystemExit``, skips traceback modification
  (these should propagate with their original traceback intact) and only
  cleans the ad-hoc ``_quent_idx`` attribute that concurrent workers may
  have attached.
  """
  if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
    _record_exception_source(exc, link, current_value)
  else:
    # Clean _quent_idx if present — concurrent workers attach it directly
    # to exceptions, and since we skip _modify_traceback / _clean_exc_meta
    # for KeyboardInterrupt/SystemExit, it would otherwise leak.
    _clean_quent_idx(exc)


def _debug_repr(v: Any, max_len: int = _MAX_REPR_LEN) -> str:
  """Truncate repr() output to *max_len* characters for debug logging.

  Respects ``QUENT_TRACEBACK_VALUES=0`` — when value display is suppressed,
  returns only the type name to prevent sensitive data from leaking into logs.
  """
  if not _show_traceback_values:  # pragma: no cover  # tested via subprocess in traceback_tests
    return f'<{type(v).__name__}>'
  try:
    r = _sanitize_repr(repr(v))
  except Exception:
    return f'<repr failed: {type(v).__name__}>'
  if len(r) > max_len:
    return r[:max_len] + '...<truncated>'
  return r


def _record_step(
  q: Q[Any],
  link: Link,
  root_link: Link | None,
  input_value: Any,
  result: Any,
  t0: int,
  on_step: Any,
  debug: bool,
  step_name: str | None = None,
  exec_id: int = 0,
  exception: BaseException | None = None,
) -> None:
  """Record on_step callback and debug log for a completed link evaluation."""
  if step_name is None:
    step_name = 'root' if link is root_link else _get_link_name(link)
  if on_step is not None:
    try:
      elapsed = _perf_counter_ns() - t0
      on_step(q, step_name, input_value, result, elapsed, exception)
    except Exception as cb_exc:
      _log.warning('quent: on_step callback raised: %r', cb_exc)
      warnings.warn(f'quent: on_step callback raised: {cb_exc!r}', RuntimeWarning, stacklevel=_user_stacklevel())
  if debug:
    _log.debug('[exec:%06x] pipeline %r: %s -> %s', exec_id, q, step_name, _debug_repr(result))


# ---- Shared exception-handling helpers ----


def _chain_finally_exc(finally_exc: BaseException, active_exc: BaseException) -> None:
  """Chain a finally handler exception to the original active exception.

  Sets ``__context__`` so Python displays the original exception as context,
  and attaches a note (Python 3.11+) so the original is visible even if
  downstream handlers don't inspect ``__context__``.

  Shared by ``_run_sync_finally``, ``_run_async_finally``, and
  ``_async_finally_transition``.
  """
  if finally_exc.__context__ is None:
    finally_exc.__context__ = active_exc
  if hasattr(finally_exc, 'add_note'):
    finally_exc.add_note(
      f'quent: This finally handler error replaced the original pipeline exception: '
      f'{type(active_exc).__name__}: {_sanitize_repr(str(active_exc))}'
    )


def _log_exc_debug(
  q: Q[Any],
  link: Link | None,
  root_link: Link | None,
  exc: BaseException,
  exec_id: int = 0,
) -> None:
  """Log a debug message when an exception is caught in the link-walk loop.

  Shared by both ``_run`` and ``_run_async`` to avoid duplicating the
  step-name resolution and formatting logic.
  """
  _step = _get_link_name(link) if link is not None and link is not root_link else 'root'
  _log.debug('[exec:%06x] pipeline %r: failed at %s: %s', exec_id, q, _step, _debug_repr(exc))


def _record_finally_step(
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  result: Any,
  t0: int,
  on_step: Any,
  debug: bool,
  exec_id: int = 0,
) -> None:
  """Record the on_step callback and debug log for a finally handler evaluation.

  Shared by ``_run_sync_finally`` and ``_run_async_finally``. The caller
  must verify ``_needs_timing`` before calling.
  """
  assert q._on_finally_link is not None  # guaranteed by caller
  _record_step(
    q,
    q._on_finally_link,
    root_link,
    _null_to_none(root_value),
    result,
    t0,
    on_step,
    debug,
    step_name='finally_',
    exec_id=exec_id,
  )


def _record_except_step(
  q: Q[Any],
  exc: BaseException,
  root_value: Any,
  root_link: Link | None,
  result: Any,
  t0: int,
  on_step: Any,
  debug: bool,
  exec_id: int = 0,
) -> None:
  """Record the on_step callback and debug log for an except handler evaluation.

  Shared by ``_run``, ``_run_async``, and ``_async_except_handler``. The
  caller must verify ``_needs_timing and q._on_except_link is not None``
  before calling.
  """
  assert q._on_except_link is not None  # guaranteed by caller
  _except_input = QuentExcInfo(exc=exc, root_value=_null_to_none(root_value))
  _record_step(
    q,
    q._on_except_link,
    root_link,
    _except_input,
    result,
    t0,
    on_step,
    debug,
    step_name='except_',
    exec_id=exec_id,
  )


# ---- Shared except-handler post-processing ----


def _except_handler_failed(
  exc: BaseException,
  handler_exc: BaseException,
  reraise: bool,
  orig_context: BaseException | None,
  orig_suppress: bool,
) -> bool:
  """Post-process a failed except handler invocation.

  Handles the shared logic when the except handler itself raises an exception:

  1. **reraise=True + Exception handler failure:** Emit RuntimeWarning, clean
     metadata, restore ``__context__``/``__suppress_context__`` to pre-handler
     values (preventing the handler's exception from permanently mutating the
     original exception's context chain).
  2. **reraise=True + non-Exception (BaseException):** Clean metadata only
     (system signals like KeyboardInterrupt propagate without warning logic).
  3. **reraise=False:** Clean metadata only (the handler's exception propagates
     with the original set as ``__cause__``).

  Returns ``True`` if the handler failure was absorbed (reraise=True with
  Exception — caller should propagate the original ``exc``).  Returns
  ``False`` if the handler's exception should propagate instead.

  Shared by ``_run_sync_except_handler``, ``_async_except_handler``, and
  ``_run_async_except_dispatch``.
  """
  if reraise and isinstance(handler_exc, Exception):
    _warn_except_handler_failed(exc, handler_exc)
    _clean_exc_meta(exc)
    exc.__context__ = orig_context
    exc.__suppress_context__ = orig_suppress
    return True  # Absorbed: caller re-raises original exc
  # Non-absorbed: clean metadata and let the handler's exception propagate.
  _clean_exc_meta(exc)
  return False


def _except_handler_succeeded(
  exc: BaseException,
  result: Any,
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  t0_exc: int,
  needs_timing: bool,
  on_step: Any,
  debug: bool,
  exec_id: int,
) -> None:
  """Post-process a successful except handler invocation.

  Handles the shared logic after the except handler completes without error:

  1. Record timing/on_step callback (if timing is active).
  2. Clean exception metadata.

  The caller is responsible for checking ``q._on_except_reraise`` and either
  re-raising the original exception or returning the handler result.

  Shared by ``_run_sync_except_handler``, ``_async_except_handler``, and
  ``_run_async_except_dispatch``.
  """
  if needs_timing and q._on_except_link is not None:
    _record_except_step(
      q,
      exc,
      root_value,
      root_link,
      result,
      t0_exc,
      on_step,
      debug,
      exec_id=exec_id,
    )
  _clean_exc_meta(exc)


# ---- Sync except-handler dispatch ----


class _SyncExceptResult(NamedTuple):
  """State returned by ``_run_sync_except_handler`` to its caller.

  Bundles the updated execution state so ``_run`` can apply it without
  relying on mutable closures.  ``async_coro`` is non-None only when the
  handler returned an awaitable and the caller should ``return`` it.
  """

  ignore_finally: bool
  active_exc: BaseException | None
  exc_to_propagate: BaseException | None
  sync_result: Any
  async_coro: Any


def _run_sync_except_handler(
  exc: BaseException,
  q: Q[Any],
  link: Link | None,
  root_link: Link | None,
  root_value: Any,
  *,
  is_nested: bool,
  _needs_timing: bool,
  _on_step: Any,
  _debug: bool,
  _exec_id: int,
  deferred_finally: list[Any] | None = None,
) -> _SyncExceptResult:
  """Dispatch the except handler in the sync execution path.

  Evaluates the pipeline's except handler, handles reraise logic, and
  detects async transitions (handler returning an awaitable).

  Returns a ``_SyncExceptResult`` with updated execution state for the
  caller to apply.
  """
  __tracebackhide__ = True
  _active_exc: BaseException | None = exc
  _exc_to_propagate: BaseException | None = None
  _sync_result: Any = Null
  _async_coro: Any = None
  _t0_exc = 0
  # Save the original context chain so we can restore it if the handler fails
  # with reraise=True (prevents the handler's exception from permanently
  # mutating the original exception's __context__/__suppress_context__).
  _orig_context = exc.__context__
  _orig_suppress = exc.__suppress_context__
  try:
    if _needs_timing:
      _t0_exc = _perf_counter_ns()
    result = _except_handler_body(exc, q, link, root_link, root_value, is_nested=is_nested)
  except BaseException as propagating_exc:
    if propagating_exc is exc:
      # Handler was never invoked — exception didn't match the filter or no
      # handler was registered.  _except_handler_body re-raised the original
      # exception unchanged.  Just propagate normally.
      _active_exc = exc
      _exc_to_propagate = exc
    elif _except_handler_failed(exc, propagating_exc, q._on_except_reraise, _orig_context, _orig_suppress):
      # Absorbed: reraise=True handler failed with Exception — propagate original.
      # _active_exc stays as exc (already set above), which is correct: the
      # finally handler sees _active_exc = exc, matching the exception that propagates.
      _exc_to_propagate = exc
    else:
      # Non-absorbed: handler's exception propagates (non-Exception BaseExceptions
      # like KeyboardInterrupt/SystemExit, or reraise=False handler failures).
      # _active_exc is updated so the finally handler sees the most recent active exception.
      _active_exc = propagating_exc
      _exc_to_propagate = propagating_exc
  else:
    # Sync path: if handler returns a coroutine, we can't await it inline.
    # Transition to async via _async_except_handler (which also handles finally).
    _reraise = q._on_except_reraise
    if _isawaitable(result):
      # _async_except_handler handles finally internally (see its finally block).
      _async_coro = _async_except_handler(
        result,
        exc,
        q,
        root_value,
        root_link,
        reraise=_reraise,
        sync_t0=_t0_exc if _needs_timing else 0,
        is_nested=is_nested,
        exec_id=_exec_id,
        deferred_finally=deferred_finally,
      )
      return _SyncExceptResult(
        ignore_finally=True,
        active_exc=_active_exc,
        exc_to_propagate=None,
        sync_result=Null,
        async_coro=_async_coro,
      )
    _except_handler_succeeded(
      exc,
      result,
      q,
      root_value,
      root_link,
      _t0_exc,
      _needs_timing,
      _on_step,
      _debug,
      _exec_id,
    )
    if _reraise:
      _exc_to_propagate = (
        exc  # Use stored raise (not bare raise) so the modified __traceback__ is respected on Python <3.11.
      )
    else:
      # Exception consumed by handler: reset so the finally handler
      # sees a success-path context (active_exc=None).
      _active_exc = None
      _sync_result = _null_to_none(result)
  return _SyncExceptResult(
    ignore_finally=False,
    active_exc=_active_exc,
    exc_to_propagate=_exc_to_propagate,
    sync_result=_sync_result,
    async_coro=None,
  )


# ---- Sync finally-transition dispatch ----


def _run_sync_finally_dispatch(
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  _active_exc: BaseException | None,
  _sync_result: Any,
  _exc_to_propagate: BaseException | None = None,
  *,
  is_nested: bool,
  exec_id: int,
) -> Any:
  """Execute the finally handler in ``_run`` and compute the async-transition override.

  Calls ``_run_sync_finally`` and, when the handler returns a coroutine,
  wraps it in ``_async_finally_transition`` so the caller can return it
  directly.  Returns ``None`` when no async transition is needed.
  """
  __tracebackhide__ = True
  _fin_coro = _run_sync_finally(q, root_value, root_link, _active_exc, is_nested=is_nested, exec_id=exec_id)
  if _fin_coro is not None:
    if _active_exc is not None:
      # Failure path: wrap the finally coroutine with the active exception.
      # _async_finally_transition will re-raise it after awaiting the finally handler.
      return _async_finally_transition(_fin_coro, Null, _active_exc, q, root_value, root_link, is_nested=is_nested)
    elif _sync_result is not Null:
      # Success path: wrap the finally coroutine with the pipeline result.
      return _async_finally_transition(_fin_coro, _sync_result, None, q, root_value, root_link, is_nested=is_nested)
    elif _exc_to_propagate is not None:
      # Control flow signal propagating (e.g., nested _Break/_Return) — still need
      # to await the finally handler to honor the "always runs" contract (spec §6.3.2).
      return _async_finally_transition(
        _fin_coro, Null, _exc_to_propagate, q, root_value, root_link, is_nested=is_nested
      )
    else:
      # Defensive: no active exception, no result, no propagating exception.
      # Close the coro to avoid ResourceWarning.
      if hasattr(_fin_coro, 'close'):
        _fin_coro.close()
  return None


# ---- Link resolution ----


def _resolve_root_link(
  q: Q[Any],
  v: Any,
  args: tuple[Any, ...] | None,
  kwargs: dict[str, Any] | None,
  root_link: Link | None,
  has_run_value: bool,
) -> tuple[Link | None, Link | None]:
  """Resolve the starting link and root_link for pipeline execution.

  Returns ``(link, root_link)`` after handling:
  - Run-value dispatch: wraps ``v`` in a synthetic root link.
  - kwargs-only dispatch: replaces root args/kwargs with caller's args/kwargs.
  - Passthrough: uses ``q._root_link`` or ``q._first_link``.
  - Validates the root_link invariant (no ``ignore_result``).
  """
  link: Link | None
  if has_run_value:
    link = Link(v, args, kwargs)
    link.next_link = q._first_link
    root_link = link
  elif (args or kwargs) and root_link is not None:
    # kwargs-only dispatch (v is Null): caller's args/kwargs replace
    # the root link's build-time args/kwargs entirely.
    # Supports nested pipeline invocation: .then(inner_pipeline, key=val)
    link = Link(root_link.v, args or None, kwargs or None)
    link.next_link = root_link.next_link
    root_link = link
  elif root_link is not None:
    link = root_link
  else:
    link = q._first_link

  # Invariant: root_link never has ignore_result=True.  The root value
  # capture in _run_async's one-shot section depends on this -- if violated,
  # root_value would be set to a side-effect result instead of the root's value.
  if root_link is not None and root_link.ignore_result:
    raise QuentException('root_link must not have ignore_result=True')

  return link, root_link


def _should_defer_with(link: Link, deferred_with: bool) -> bool:
  """Return True if this is the terminal with_/with_do link that should be deferred."""
  if deferred_with and link.next_link is None:
    op: _PipelineOp | Any = link.v
    _lname = getattr(op, '_link_name', None)
    if _lname == 'with_' or _lname == 'with_do':
      return True
  return False


# ---- Execution engine ----


def _run(
  q: Q[Any],
  v: Any,
  args: tuple[Any, ...] | None,
  kwargs: dict[str, Any] | None,
  is_nested: bool = False,
  *,
  deferred_finally: list[Any] | None = None,
  deferred_with: bool = False,
) -> Any:
  """Synchronous execution engine.

  This is the heart of Q. It walks the linked list of Links, evaluating
  each one and threading the current value through the pipeline.

  **Two-tier execution model:** Execution starts here, synchronously.
  After each link evaluation, the result is checked with ``_isawaitable()``.
  On the first awaitable result, we immediately delegate to
  _run_async(), passing the pending awaitable and all accumulated
  state. ``_run_async`` picks up exactly where we left off -- it awaits the
  result and continues the link walk in async mode.

  The finally-handler logic runs as normal code after the try/except (not
  in a ``finally`` block) to avoid ``return`` inside ``finally``.  When
  delegating to ``_run_async``, the early return exits before reaching
  the finally section (``ignore_finally`` guard), because ``_run_async``
  has its own finally handling.

  Returns:
    The final pipeline value, or a coroutine from ``_run_async`` if an
    async transition occurred.
  """
  __tracebackhide__ = True
  root_link: Link | None = q._root_link
  current_value: Any = Null
  root_value: Any = Null
  has_run_value = v is not Null
  has_root_value = has_run_value or root_link is not None
  ignore_finally = False  # True when _run_async handles cleanup
  _on_step = type(q).on_step
  if _on_step is not None or _log.isEnabledFor(_DEBUG_LEVEL):
    _on_step, _debug, _needs_timing = _timing_ctx(q, _on_step)
    _exec_id = _next_exec_id()
  else:
    _debug = False
    _needs_timing = False
    _exec_id = 0
  _active_exc: BaseException | None = None
  _sync_result: Any = Null  # Null = no result (exception propagating)
  _exc_to_propagate: BaseException | None = None
  _fin_override: Any = None  # Async finally transition coroutine
  # First-link guard: captures root_value and initializes current_value
  # on the first link only.  The async path handles this in its one-shot section.
  first_link_processed = False
  link: Link | None = None

  if _debug:
    _log.debug('[exec:%06x] pipeline %r: run started', _exec_id, q)

  try:
    # Fast path: non-callable run(v) with no args — skip Link allocation entirely.
    # _resolve_root_link would create a throwaway Link that _evaluate_value returns as-is.
    # Disabled when on_step/debug is active (_needs_timing) to preserve callback behavior.
    if has_run_value and not args and not kwargs and not callable(v) and not _needs_timing:
      current_value = v
      root_value = v
      first_link_processed = True
      link = q._first_link
    else:
      link, root_link = _resolve_root_link(q, v, args, kwargs, root_link, has_run_value)

      # Invariant: root_link never has ignore_result (validated by _resolve_root_link,
      # but asserted here as a safety net per JPL Rule 5).
      if root_link is not None and root_link.ignore_result:
        raise QuentException('root_link must not have ignore_result=True')

    # Link-walk loop (sync path).  The async counterpart lives in
    # _run_async().  Shared logic: _record_step().  Differences:
    # sync checks _isawaitable and delegates to _run_async on transition;
    # async awaits inline.  First-link initialization (first_link_processed)
    # exists only in the sync path — see the comment at its declaration.
    _t0 = 0
    _input_value = None
    while link is not None:
      if _should_defer_with(link, deferred_with):
        break
      if _needs_timing:
        if not first_link_processed and has_run_value:
          _input_value = _null_to_none(v)  # Root step: input is the run value.
        else:
          _input_value = _null_to_none(current_value)
        _t0 = _perf_counter_ns()
      result = _evaluate_value(link, current_value)
      # Fast path: reject common sync return types without calling _isawaitable.
      # CoroutineType is checked first (exact type match) for the async-transition
      # case; the frozenset lookup rejects ints/strings/etc in ~30ns vs ~350ns.
      if type(result) is CoroutineType or (
        result is not None and type(result) not in _SYNC_TYPES and _isawaitable(result)
      ):
        # Async transition: hand off to _run_async with all current state.
        if _debug:
          _log.debug(
            '[exec:%06x] pipeline %r: async transition at %s',
            _exec_id,
            q,
            _get_link_name(link) if link is not root_link else 'root',
          )
        ignore_finally = True
        return _run_async(
          q,
          result,
          link,
          current_value,
          root_value,
          has_root_value,
          root_link,
          is_nested,
          sync_t0=_t0 if _needs_timing else 0,
          sync_input_value=_input_value if _needs_timing else None,
          on_step=_on_step,
          exec_id=_exec_id,
          deferred_finally=deferred_finally,
          deferred_with=deferred_with,
        )
      if _needs_timing:
        _record_step(
          q,
          link,
          root_link,
          _input_value,
          result,
          _t0,
          _on_step,
          _debug,
          exec_id=_exec_id,
        )
      if not first_link_processed:
        first_link_processed = True
        # (a) Capture root_value: the first link's result becomes the root
        # value passed to except_/finally_ handlers.  Guarded by
        # `root_value is Null` so it's a one-shot assignment.
        if has_root_value and root_value is Null:
          root_value = result
        # (b) Initialize current_value: if the first link is a .do()
        # (ignore_result=True), current_value stays Null until a non-ignored
        # link produces a result.  This explicit initialization handles the
        # case where _evaluate_value needs a non-Null current_value for the
        # next link's calling convention.
        if current_value is Null and not link.ignore_result:
          current_value = result
      if not link.ignore_result:
        current_value = result
      link = link.next_link

    # Invariant: loop walked to end of list (no early exit except via return/raise),
    # OR deferred_with broke out at the last _WithOp link.
    if link is not None and not deferred_with:
      raise QuentException('link-walk loop exited with link still set')

    if _debug:
      _log.debug('[exec:%06x] pipeline %r: completed -> %s', _exec_id, q, _debug_repr(_null_to_none(current_value)))
    # If all steps were .do() (ignore_result=True) with no root value,
    # current_value stays Null throughout and _null_to_none returns None.
    _sync_result = _null_to_none(current_value)

  except _Return as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: early return', _exec_id, q)
    if is_nested:
      _exc_to_propagate = exc
    else:
      _sync_result = _handle_return_exc(exc, False)

  except _Break as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: break signal', _exec_id, q)
    if is_nested:
      _exc_to_propagate = exc
    else:
      msg = (
        'Q.break_() cannot be used outside of a loop or iteration context'
        ' (foreach, foreach_do, iterate, iterate_do, flat_iterate, flat_iterate_do, while_).'
      )
      _q_exc = QuentException(msg)
      _q_exc.__suppress_context__ = True
      _exc_to_propagate = _q_exc

  except BaseException as exc:
    if _debug:
      _log_exc_debug(q, link, root_link, exc, exec_id=_exec_id)
    # Fire on_step for the failing step with the exception.
    if _needs_timing and link is not None:
      _record_step(
        q,
        link,
        root_link,
        _input_value,
        None,
        _t0,
        _on_step,
        False,
        exec_id=_exec_id,
        exception=exc,
      )
    _active_exc = exc
    _handle_base_exception(exc, link, current_value)
    _seh = _run_sync_except_handler(
      exc,
      q,
      link,
      root_link,
      root_value,
      is_nested=is_nested,
      _needs_timing=_needs_timing,
      _on_step=_on_step,
      _debug=_debug,
      _exec_id=_exec_id,
      deferred_finally=deferred_finally,
    )
    ignore_finally = _seh.ignore_finally
    _active_exc = _seh.active_exc
    _exc_to_propagate = _seh.exc_to_propagate
    if _seh.sync_result is not Null:
      _sync_result = _seh.sync_result
    if _seh.async_coro is not None:
      return _seh.async_coro

  finally:
    if deferred_finally is not None and not ignore_finally:
      deferred_finally[0] = root_value
      deferred_finally[1] = root_link
      deferred_finally[2] = _exec_id
    elif not ignore_finally and q._on_finally_link is not None:
      _fin_override = _run_sync_finally_dispatch(
        q,
        root_value,
        root_link,
        _active_exc,
        _sync_result,
        _exc_to_propagate,
        is_nested=is_nested,
        exec_id=_exec_id,
      )

  # Post-clause: return the async finally transition, re-raise stored
  # exceptions, or return the sync result.  The ignore_finally=True early
  # returns in try/except exit the function before reaching here (the finally
  # block still runs for those, but ignore_finally=True makes it a no-op).
  if _fin_override is not None:
    return _fin_override
  if _exc_to_propagate is not None:
    raise _exc_to_propagate
  return _sync_result


# Async execution functions are in _engine_async.py.
# Re-exported here for backward compatibility with existing imports.
from ._engine_async import (  # noqa: E402
  _async_except_handler,
  _async_finally_transition,
  _run_async,
  _run_async_finally,  # noqa: F401 — re-exported for _generator.py
)
