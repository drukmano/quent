# SPDX-License-Identifier: MIT
"""Execution engine -- sync/async link walking and error handling.

Debug logging is available via the ``'quent'`` logger
(``logging.getLogger('quent')``).  Set to DEBUG to see step-by-step pipeline
execution traces including link results and async transitions.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
import warnings
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
_perf_counter_ns = time.perf_counter_ns
_DEBUG_LEVEL = logging.DEBUG
_exec_counter_lock = threading.Lock()
_exec_counter_value = 0


def _next_exec_id() -> int:
  """Return a unique execution ID, safe under free-threaded Python (PEP 703).

  Masked to 6 hex digits (0x000000-0xFFFFFF) per spec 14.5, wrapping after
  16,777,215 executions to maintain the fixed-width ``[exec:XXXXXX]`` format.
  """
  global _exec_counter_value
  with _exec_counter_lock:
    val = _exec_counter_value
    _exec_counter_value = (val + 1) & 0xFFFFFF
  return val


_ON_STEP_NOT_RESOLVED = object()
_ON_STEP_ARITY_ATTR = '_quent_on_step_arity'


def _detect_on_step_arity(callback: Any) -> int:
  """Detect whether an on_step callback accepts 5 or 6 positional args.

  Returns the number of positional parameters the callback accepts (5 or 6).
  The result is cached on the callback object via ``_quent_on_step_arity``
  to avoid repeated ``inspect.signature`` introspection.
  """
  cached: int | None = getattr(callback, _ON_STEP_ARITY_ATTR, None)
  if cached is not None:
    return cached
  arity = 6  # Default to 6 (new signature)
  try:
    sig = inspect.signature(callback)
    positional_kinds = (
      inspect.Parameter.POSITIONAL_ONLY,
      inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    if has_var_positional:
      arity = 6
    else:
      n_positional = sum(1 for p in sig.parameters.values() if p.kind in positional_kinds)
      arity = 6 if n_positional >= 6 else 5
  except (ValueError, TypeError):
    arity = 6  # If inspection fails, assume new-style (6-arg)
  try:
    object.__setattr__(callback, _ON_STEP_ARITY_ATTR, arity)
  except (AttributeError, TypeError):
    pass  # Built-ins / frozen objects — detection runs each time, no big deal
  return arity


def _timing_ctx(q: Q[Any], on_step: Any = _ON_STEP_NOT_RESOLVED) -> tuple[Any, bool, bool, int]:
  """Return (on_step, debug, needs_timing, on_step_arity) for the given pipeline.

  ``on_step`` may be passed in from ``_run``'s call to ``_run_async`` (which
  already resolved it) to avoid a redundant ``type(q).on_step`` lookup.

  ``on_step_arity`` is 5 or 6 depending on the callback's signature.
  """
  _on_step = type(q).on_step if on_step is _ON_STEP_NOT_RESOLVED else on_step
  _debug = _log.isEnabledFor(_DEBUG_LEVEL)
  _arity = _detect_on_step_arity(_on_step) if _on_step is not None else 6
  return _on_step, _debug, _on_step is not None or _debug, _arity


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


async def _async_finally_transition(
  finally_result: Any,
  pipeline_result: Any,
  active_exc: BaseException | None,
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  is_nested: bool = False,
) -> Any:
  """Async transition for sync pipeline's finally handler that returned a coroutine.

  Awaits the finally handler, then returns the pipeline result or re-raises
  the active exception. This preserves the pipeline's value through the
  async transition -- the caller gets the real result when they await.
  """
  __tracebackhide__ = True
  try:
    await _await_finally_result(finally_result, q, root_value, root_link, is_nested=is_nested)
  except BaseException as finally_exc:
    if active_exc is not None:
      _chain_finally_exc(finally_exc, active_exc)
    raise
  if active_exc is not None:
    raise active_exc
  # Return the pipeline result, awaiting if it's itself a coroutine
  # (e.g., Q.return_(async_fn) where the return value is an awaitable)
  if _isawaitable(pipeline_result):
    return await pipeline_result
  return pipeline_result


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
  _on_step, _debug, _needs_timing, _on_step_arity = _timing_ctx(q)
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
        on_step_arity=_on_step_arity,
      )
    if _isawaitable(result):
      return result  # Return the awaitable for the caller to wrap
    return None
  except BaseException as finally_exc:
    if active_exc is not None:
      _chain_finally_exc(finally_exc, active_exc)
    raise


async def _run_async_finally(
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  active_exc: BaseException | None,
  is_nested: bool = False,
  exec_id: int = 0,
) -> None:
  """Execute the finally handler in the async execution path."""
  __tracebackhide__ = True
  _on_step, _debug, _needs_timing, _on_step_arity = _timing_ctx(q)
  try:
    if _needs_timing:
      _t0_fin = _perf_counter_ns()
    finally_result = _finally_handler_body(q, root_value, root_link, is_nested=is_nested)
    if _isawaitable(finally_result):
      await _await_finally_result(finally_result, q, root_value, root_link, is_nested=is_nested)
    if _needs_timing:
      _record_finally_step(
        q,
        root_value,
        root_link,
        finally_result,
        _t0_fin,
        _on_step,
        _debug,
        exec_id=exec_id,
        on_step_arity=_on_step_arity,
      )
  except BaseException as finally_exc:
    if active_exc is not None:
      _chain_finally_exc(finally_exc, active_exc)
    raise


async def _async_except_handler(
  handler_coro: Any,
  exc: BaseException,
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  *,
  reraise: bool,
  sync_t0: int = 0,
  is_nested: bool = False,
  exec_id: int = 0,
  deferred_finally: list[Any] | None = None,
) -> Any:
  """Async transition: await except handler, then re-raise (reraise=True) or return result (reraise=False)."""
  __tracebackhide__ = True
  _active_exc: BaseException | None = exc
  _on_step, _debug, _needs_timing, _on_step_arity = _timing_ctx(q)
  result: Any = None
  try:
    try:
      if reraise:
        await handler_coro
      else:
        result = await handler_coro
    except _ControlFlowSignal as signal:
      if signal.__context__ is None:
        signal.__context__ = exc
      msg = _signal_in_handler_msg(signal, 'except')
      # Use `from exc` consistently: the original exception that triggered the
      # handler is the most relevant context for debugging, regardless of reraise.
      # No _clean_exc_meta here: code after raise is unreachable, and exc metadata
      # was already cleaned by _modify_traceback in _except_handler_body (called
      # by the sync path before handing off to this async handler).
      raise QuentException(msg) from exc
    except BaseException as handler_exc:
      if reraise:
        if not isinstance(handler_exc, Exception):
          _clean_exc_meta(exc)
          _active_exc = handler_exc  # Ensure finally sees the true active exception
          raise  # BaseException (KeyboardInterrupt, SystemExit) propagates naturally
        _warn_except_handler_failed(exc, handler_exc)
        exc.__suppress_context__ = True
      else:
        _clean_exc_meta(exc)
        _active_exc = handler_exc
        raise handler_exc from exc
    if _needs_timing and q._on_except_link is not None:
      _t0 = sync_t0 if sync_t0 else _perf_counter_ns()
      _record_except_step(
        q,
        exc,
        root_value,
        root_link,
        result,
        _t0,
        _on_step,
        _debug,
        exec_id=exec_id,
        on_step_arity=_on_step_arity,
      )
    _clean_exc_meta(exc)
    if reraise:
      raise exc
    # Exception consumed by handler: reset so the finally handler
    # sees a success-path context (active_exc=None).
    _active_exc = None
    return _null_to_none(result)
  finally:
    if deferred_finally is not None:
      deferred_finally[0] = root_value
      deferred_finally[1] = root_link
      deferred_finally[2] = exec_id
    elif q._on_finally_link is not None:
      await _run_async_finally(q, root_value, root_link, _active_exc, is_nested=is_nested, exec_id=exec_id)


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
  on_step_arity: int = 6,
  exception: BaseException | None = None,
) -> None:
  """Record on_step callback and debug log for a completed link evaluation."""
  if step_name is None:
    step_name = 'root' if link is root_link else _get_link_name(link)
  if on_step is not None:
    # 5-arg callbacks don't receive the exception parameter and are not
    # called on step failure — only 6-arg callbacks receive failure events.
    if on_step_arity < 6 and exception is not None:
      pass  # Skip: legacy 5-arg callback, failure event
    else:
      try:
        elapsed = _perf_counter_ns() - t0
        if on_step_arity >= 6:
          on_step(q, step_name, input_value, result, elapsed, exception)
        else:
          on_step(q, step_name, input_value, result, elapsed)
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
  on_step_arity: int = 6,
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
    on_step_arity=on_step_arity,
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
  on_step_arity: int = 6,
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
    on_step_arity=on_step_arity,
  )


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
  _on_step_arity: int = 6,
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
    elif q._on_except_reraise and isinstance(propagating_exc, Exception):
      # reraise=True: handler ran for side-effects only; re-raise original.
      # Log the handler failure, attach a note to the original exception,
      # and re-raise the original so users get the exception they expect.
      # Note: _active_exc is NOT updated here because we re-raise the
      # original `exc` (which _active_exc already references from the assignment above).
      # This is correct — the finally handler sees _active_exc = exc, which
      # matches the exception that actually propagates.  Compare with the
      # BaseException branch below, where _active_exc IS updated because a
      # different exception (propagating_exc) propagates.
      _warn_except_handler_failed(exc, propagating_exc)
      _clean_exc_meta(exc)
      exc.__suppress_context__ = (
        True  # Prevent circular/misleading context: handler failure already logged via _warn_except_handler_failed
      )
      _exc_to_propagate = exc
    else:
      # Non-Exception BaseExceptions (KeyboardInterrupt, SystemExit) propagate
      # without _warn_except_handler_failed — this is intentional (system
      # signals should not be delayed by warning logic).  Note: _active_exc
      # is updated so the finally handler sees the most recent active exception.
      _clean_exc_meta(exc)
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
    if _needs_timing and q._on_except_link is not None:
      _record_except_step(
        q,
        exc,
        root_value,
        root_link,
        result,
        _t0_exc,
        _on_step,
        _debug,
        exec_id=_exec_id,
        on_step_arity=_on_step_arity,
      )
    _clean_exc_meta(exc)
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


# ---- Async except-handler dispatch ----


async def _run_async_except_dispatch(
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
  exec_id: int,
  _on_step_arity: int = 6,
) -> tuple[BaseException | None, Any]:
  """Dispatch the except handler in the async execution path.

  Awaits the handler result inline (unlike the sync path which must
  delegate to ``_async_except_handler``).

  Returns ``(active_exc, result)`` where:
  - ``active_exc`` is ``None`` when the handler consumed the exception
    (the caller should return the result).
  - ``active_exc`` is non-None only if re-raised via an outer handler
    (not reached here — those paths raise directly).

  All exception paths raise directly from this function so the caller's
  ``finally`` block sees the correct ``_active_exc`` via the raised
  exception propagating out.
  """
  __tracebackhide__ = True
  _active_exc: BaseException | None = exc
  _t0_exc = 0
  try:
    # Exception propagation paths:
    # 1. _except_handler_body or its awaited result raises _ControlFlowSignal
    #    -> caught by `except _ControlFlowSignal` below, wrapped in QuentException
    # 2. _except_handler_body or its awaited result raises other BaseException
    #    -> propagates out of this try/except (not caught by _ControlFlowSignal handler)
    #    -> Python sets __context__ to `exc` automatically
    # 3. on_except_reraise=True: `raise exc` re-raises the original exception
    #    -> propagates out normally (not a _ControlFlowSignal, so not caught below)
    try:
      if _needs_timing:
        _t0_exc = _perf_counter_ns()
      result = _except_handler_body(exc, q, link, root_link, root_value, is_nested=is_nested)
    except BaseException as propagating_exc:
      _active_exc = propagating_exc
      raise
    if _isawaitable(result):
      try:
        result = await result
      except _ControlFlowSignal as signal:
        if signal.__context__ is None:
          signal.__context__ = exc
        raise QuentException(_signal_in_handler_msg(signal, 'except')) from exc
      except BaseException as handler_exc:
        if q._on_except_reraise:
          if not isinstance(handler_exc, Exception):
            _clean_exc_meta(exc)
            _active_exc = handler_exc  # Ensure finally sees the true active exception
            raise  # BaseException (KeyboardInterrupt, SystemExit) propagates naturally
          # reraise=True: handler ran for side-effects only; re-raise original.
          # Log the handler failure so users can debug broken error handlers.
          # Note: _active_exc is NOT updated here because we re-raise the
          # original `exc` (which _active_exc already references from the assignment above).
          # This matches the sync path in _run() — see comment there.
          _warn_except_handler_failed(exc, handler_exc)
          _clean_exc_meta(exc)
          exc.__suppress_context__ = (
            True  # Prevent circular/misleading context: handler failure already logged via _warn_except_handler_failed
          )
          raise exc  # noqa: B904
        _clean_exc_meta(exc)
        _active_exc = handler_exc
        raise handler_exc from exc
    if _needs_timing and q._on_except_link is not None:
      _record_except_step(
        q,
        exc,
        root_value,
        root_link,
        result,
        _t0_exc,
        _on_step,
        _debug,
        exec_id=exec_id,
        on_step_arity=_on_step_arity,
      )
    if q._on_except_reraise:
      _clean_exc_meta(exc)
      raise exc
  except _ControlFlowSignal as signal:
    if signal.__context__ is None:
      signal.__context__ = exc
    qe = QuentException(_signal_in_handler_msg(signal, 'except'))
    _active_exc = qe
    raise qe from exc
  # Exception consumed by handler (reraise=False) — drop heavy metadata
  # references so stored exception objects don't hold pipeline internals alive.
  _clean_exc_meta(exc)
  return None, result


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
    _on_step, _debug, _needs_timing, _on_step_arity = _timing_ctx(q, _on_step)
    _exec_id = _next_exec_id()
  else:
    _debug = False
    _needs_timing = False
    _on_step_arity = 6
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
      # Operations (map, foreach_do, gather, with_, if_) are invoked here as regular
      # callables — their __call__ methods perform the operation logic (see _*Op classes).
      result = _evaluate_value(link, current_value)
      if _isawaitable(result):
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
          on_step_arity=_on_step_arity,
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
    _sync_result = _handle_return_exc(exc, is_nested)

  except _Break as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: break signal', _exec_id, q)
    if is_nested:
      _exc_to_propagate = exc
    else:
      msg = (
        'Q.break_() cannot be used outside of an iteration context'
        ' (foreach, foreach_do, iterate, iterate_do, flat_iterate, flat_iterate_do).'
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
        on_step_arity=_on_step_arity,
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
      _on_step_arity=_on_step_arity,
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


async def _run_async(
  q: Q[Any],
  awaitable: Any,
  link: Link,
  current_value: Any = Null,
  root_value: Any = Null,
  has_root_value: bool = False,
  root_link: Link | None = None,
  is_nested: bool = False,
  sync_t0: int = 0,
  sync_input_value: Any = None,
  on_step: Any = None,
  exec_id: int = 0,
  deferred_finally: list[Any] | None = None,
  deferred_with: bool = False,
) -> Any:
  """Async continuation of the execution engine.

  Called by _run() when the first awaitable result is encountered.
  Receives the pending awaitable and all accumulated state from the sync
  path, then continues the link walk in async mode (awaiting any further
  awaitables).
  """
  __tracebackhide__ = True
  # Initialized to None so the except block can reference it even if assignment never completed.
  _active_exc: BaseException | None = None
  if on_step is not None or _log.isEnabledFor(_DEBUG_LEVEL):
    _on_step, _debug, _needs_timing, _on_step_arity = _timing_ctx(q, on_step)
  else:
    _on_step = None
    _debug = False
    _needs_timing = False
    _on_step_arity = 6

  if _debug:
    _log.debug('[exec:%06x] pipeline %r: async continuation started', exec_id, q)

  # Invariant carried from _run(): root_link never has ignore_result.
  if root_link is not None and root_link.ignore_result:
    raise QuentException('root_link must not have ignore_result=True')

  try:
    # Complete the in-progress step handed off from _run().
    # This is a one-shot section (not a loop), so it handles the same two
    # first-link concerns that _run() guards with `first_link_processed`:
    # (a) capture root_value, (b) initialize current_value.
    # These may already be resolved if _run() processed links before the
    # async transition; the `is Null` guards make them no-ops in that case.
    # Use sync_t0 from _run() for accurate end-to-end timing that includes
    # the sync _evaluate_value call that produced the awaitable.
    if _needs_timing:
      _t0 = sync_t0 if sync_t0 else _perf_counter_ns()
      _input_value = sync_input_value if sync_t0 else _null_to_none(current_value)
    result = await awaitable
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
        exec_id=exec_id,
        on_step_arity=_on_step_arity,
      )
    if has_root_value and root_value is Null:
      root_value = result
    if current_value is Null and not link.ignore_result:
      current_value = result
    if not link.ignore_result:
      current_value = result
    next_link: Link | None = link.next_link

    # Link-walk loop (async path).  The sync counterpart lives in _run().
    # Shared logic: _record_step().  Differences: async awaits inline;
    # sync delegates to _run_async on first awaitable.
    _t0 = 0
    _input_value = None
    while next_link is not None:
      link = next_link
      if _should_defer_with(link, deferred_with):
        break
      if _needs_timing:
        _input_value = _null_to_none(current_value)
        _t0 = _perf_counter_ns()
      # Operations (map, foreach_do, gather, with_, if_) are invoked here as regular
      # callables — their __call__ methods perform the operation logic (see _*Op classes).
      result = _evaluate_value(link, current_value)
      if _isawaitable(result):
        result = await result
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
          exec_id=exec_id,
          on_step_arity=_on_step_arity,
        )
      if not link.ignore_result:
        current_value = result
      next_link = link.next_link

    # Invariant: loop walked to end of list (no early exit except via return/raise),
    # OR deferred_with broke out at the last _WithOp link.
    if next_link is not None and not deferred_with:
      raise QuentException('link-walk loop exited with next_link still set')

    if _debug:
      _log.debug('[exec:%06x] pipeline %r: completed -> %s', exec_id, q, _debug_repr(_null_to_none(current_value)))
    return _null_to_none(current_value)

  except _Return as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: early return', exec_id, q)
    result = _handle_return_exc(exc, is_nested)
    if _isawaitable(result):
      return await result
    return result

  except _Break:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: break signal', exec_id, q)
    if is_nested:
      raise
    msg = (
      'Q.break_() cannot be used outside of an iteration context'
      ' (foreach, foreach_do, iterate, iterate_do, flat_iterate, flat_iterate_do).'
    )
    raise QuentException(msg) from None

  except BaseException as exc:
    if _debug:
      _log_exc_debug(q, link, root_link, exc, exec_id=exec_id)
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
        exec_id=exec_id,
        on_step_arity=_on_step_arity,
        exception=exc,
      )
    _active_exc = exc
    _handle_base_exception(exc, link, current_value)
    try:
      _active_exc, result = await _run_async_except_dispatch(
        exc,
        q,
        link,
        root_link,
        root_value,
        is_nested=is_nested,
        _needs_timing=_needs_timing,
        _on_step=_on_step,
        _debug=_debug,
        exec_id=exec_id,
        _on_step_arity=_on_step_arity,
      )
      # Handler consumed the exception (active_exc=None); return result.
      return _null_to_none(result)
    except BaseException as dispatch_exc:
      _active_exc = dispatch_exc
      raise

  finally:
    if deferred_finally is not None:
      deferred_finally[0] = root_value
      deferred_finally[1] = root_link
      deferred_finally[2] = exec_id
    elif q._on_finally_link is not None:
      await _run_async_finally(q, root_value, root_link, _active_exc, is_nested=is_nested, exec_id=exec_id)
