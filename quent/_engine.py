# SPDX-License-Identifier: MIT
"""Execution engine — sync path, shared helpers, re-exports.

Sync entry (``_run``) and all shared helpers (step recording, exception
handling, null normalization) live here. Async paths (``_run_async`` etc.)
live in ``_engine_async.py``, re-exported from this module.

Debug logging: ``logging.getLogger('quent')`` at DEBUG level.
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
  _Exit,
  _Return,
)
from ._viz import _MAX_REPR_LEN, _get_link_name, _sanitize_repr, _show_traceback_values

if TYPE_CHECKING:
  from ._q import Q
  from ._types import _PipelineOp

_log = logging.getLogger('quent')

# Common sync types — O(1) frozenset (~30ns) avoids _isawaitable (~350ns)
# for the ~90% of steps that return basic types.
_SYNC_TYPES = frozenset({int, str, float, bool, list, dict, tuple, set, bytes})
_perf_counter_ns = time.perf_counter_ns
_DEBUG_LEVEL = logging.DEBUG
_exec_counter = itertools.count()

# Free-threaded Python (PEP 703): itertools.count() is atomic under the GIL
# but not under no-GIL. Detect via sys._is_gil_enabled (3.13+); fall back to
# a threading.Lock when the GIL is disabled.
_gil_enabled: bool = True
try:
  _gil_enabled = sys._is_gil_enabled()  # type: ignore[attr-defined]
except AttributeError:
  _gil_enabled = True

if not _gil_enabled:
  import threading

  _exec_counter_lock = threading.Lock()
  _exec_counter_value = 0


def _next_exec_id() -> int:
  """Unique exec ID, masked to 6 hex digits. Safe under PEP 703."""
  if _gil_enabled:
    return next(_exec_counter) & 0xFFFFFF
  global _exec_counter_value
  with _exec_counter_lock:
    val = _exec_counter_value
    _exec_counter_value = (val + 1) & 0xFFFFFF
  return val


_ON_STEP_NOT_RESOLVED = object()


def _timing_ctx(q: Q[Any], on_step: Any = _ON_STEP_NOT_RESOLVED) -> tuple[Any, bool, bool]:
  """Return (on_step, debug, needs_timing). on_step may be passed pre-resolved."""
  _on_step = type(q).on_step if on_step is _ON_STEP_NOT_RESOLVED else on_step
  _debug = _log.isEnabledFor(_DEBUG_LEVEL)
  return _on_step, _debug, _on_step is not None or _debug


def _signal_in_handler_msg(signal: _ControlFlowSignal, handler: str) -> str:
  return f'Using {type(signal).__name__} inside {handler} handlers is not allowed.'


def _null_to_none(value: Any) -> Any:
  return None if value is Null else value


def _warn_except_handler_failed(original_exc: BaseException, handler_exc: BaseException) -> None:
  """Log+warn when an except handler (reraise=True) fails; attach a note."""
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
  """Evaluate the except handler. QuentExcInfo is the current value (unified convention)."""
  __tracebackhide__ = True
  try:
    _modify_traceback(exc, q, link, root_link, is_nested=is_nested)
  except Exception as e:
    _clean_exc_meta(exc)
    warnings.warn(f'quent: traceback enhancement failed: {e!r}', RuntimeWarning, stacklevel=_user_stacklevel())
  if q._on_except_link is None:
    raise exc
  if q._on_except_exceptions is None:
    raise QuentException('_on_except_exceptions must be set when _on_except_link is set')
  if not isinstance(exc, q._on_except_exceptions):
    raise exc
  except_link = q._on_except_link
  try:
    exc_info = QuentExcInfo(exc=exc, root_value=_null_to_none(root_value))
    result = _evaluate_value(except_link, exc_info)
  except _Exit:
    # Q.exit_() bypasses except trap — propagate to outermost run(). finally_ still runs.
    raise
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
  """Attach metadata and re-raise an exception from a finally handler."""
  if isinstance(exc, _Exit):
    raise
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
  """Evaluate the finally handler."""
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
  """Await an async finally handler with proper error handling."""
  __tracebackhide__ = True
  assert q._on_finally_link is not None
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
  """Sync finally execution. Returns the awaitable result if coroutine, else None."""
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
      return result
    return None
  except BaseException as finally_exc:
    if active_exc is not None:
      _chain_finally_exc(finally_exc, active_exc)
    raise


def _record_exception_source(exc: BaseException, link: Link | None, current_value: Any) -> None:
  """Stamp the failing link onto the exception for traceback display.

  First-write-wins. Also cleans _quent_idx (defense-in-depth for BaseException
  subclasses that bypass _modify_traceback).
  """
  __tracebackhide__ = True
  _clean_quent_idx(exc)
  meta = _get_exc_meta(exc)
  if meta.get(META_SOURCE_LINK) is None and link is not None:
    meta[META_SOURCE_LINK] = link
  if link is None:
    return
  # Plain user callables (then/do): stamp current_value as temp args.
  # Operations (with _link_name) stamp their own specialized temp args.
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
  """Record source for regular exceptions; for KI/SystemExit, just clean _quent_idx."""
  if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
    _record_exception_source(exc, link, current_value)
  else:
    # KI/SystemExit must propagate with original traceback intact, but we still
    # clean _quent_idx since it would otherwise leak (we skip _modify_traceback).
    _clean_quent_idx(exc)


def _debug_repr(v: Any, max_len: int = _MAX_REPR_LEN) -> str:
  """Truncated repr for debug logs. Respects QUENT_TRACEBACK_VALUES=0."""
  if not _show_traceback_values:  # pragma: no cover
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


def _chain_finally_exc(finally_exc: BaseException, active_exc: BaseException) -> None:
  """Chain a finally handler exception to the original active exception.

  Sets __context__ + add_note (3.11+) so the original is visible even to
  handlers that don't inspect __context__.
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
  """Record on_step + debug log for a finally handler. Caller must check _needs_timing."""
  assert q._on_finally_link is not None
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
  """Record on_step + debug log for an except handler. Caller must verify timing+link."""
  assert q._on_except_link is not None
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


def _except_handler_failed(
  exc: BaseException,
  handler_exc: BaseException,
  reraise: bool,
  orig_context: BaseException | None,
  orig_suppress: bool,
) -> bool:
  """Post-process a failed except-handler invocation.

  - reraise=True + Exception handler-failure: warn, clean meta, restore
    __context__/__suppress_context__ (don't let the handler's exc permanently
    mutate the original). Returns True (absorbed — caller propagates exc).
  - reraise=True + non-Exception BaseException (KI/SystemExit): clean meta only.
    Returns False (handler's exc propagates).
  - reraise=False: clean meta only. Returns False (handler's exc propagates
    with original as __cause__).
  """
  if reraise and isinstance(handler_exc, Exception):
    _warn_except_handler_failed(exc, handler_exc)
    _clean_exc_meta(exc)
    exc.__context__ = orig_context
    exc.__suppress_context__ = orig_suppress
    return True
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
  """Record timing/on_step (if active), then clean exception metadata."""
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


class _SyncExceptResult(NamedTuple):
  """State returned by _run_sync_except_handler. async_coro is non-None when caller should return it."""

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
  """Dispatch the except handler on the sync path; detect async transitions."""
  __tracebackhide__ = True
  _active_exc: BaseException | None = exc
  _exc_to_propagate: BaseException | None = None
  _sync_result: Any = Null
  _async_coro: Any = None
  _t0_exc = 0
  # Save original context chain so we can restore it if reraise=True handler fails.
  _orig_context = exc.__context__
  _orig_suppress = exc.__suppress_context__
  try:
    if _needs_timing:
      _t0_exc = _perf_counter_ns()
    result = _except_handler_body(exc, q, link, root_link, root_value, is_nested=is_nested)
  except BaseException as propagating_exc:
    if propagating_exc is exc:
      # Filter mismatch or no handler — _except_handler_body re-raised unchanged.
      _active_exc = exc
      _exc_to_propagate = exc
    elif _except_handler_failed(exc, propagating_exc, q._on_except_reraise, _orig_context, _orig_suppress):
      # Absorbed: reraise=True+Exception failed → propagate original.
      _exc_to_propagate = exc
    else:
      # Non-absorbed: handler's exception propagates (KI/SystemExit reraise=True, or reraise=False).
      _active_exc = propagating_exc
      _exc_to_propagate = propagating_exc
  else:
    # Sync path: if handler returned a coroutine, transition via _async_except_handler.
    _reraise = q._on_except_reraise
    if _isawaitable(result):
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
      # Stored raise (not bare) so modified __traceback__ is respected on Python <3.11.
      _exc_to_propagate = exc
    else:
      # Consumed — finally sees success-path context.
      _active_exc = None
      _sync_result = _null_to_none(result)
  return _SyncExceptResult(
    ignore_finally=False,
    active_exc=_active_exc,
    exc_to_propagate=_exc_to_propagate,
    sync_result=_sync_result,
    async_coro=None,
  )


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
  """Execute finally; on coroutine result, wrap in _async_finally_transition.

  Returns the wrapped coroutine, or None when no async transition is needed.
  """
  __tracebackhide__ = True
  _fin_coro = _run_sync_finally(q, root_value, root_link, _active_exc, is_nested=is_nested, exec_id=exec_id)
  if _fin_coro is not None:
    # Priority for the async-finally transition:
    # 1. _exc_to_propagate set (error/signal propagating outward) → re-raise after finally.
    # 2. _sync_result set (absorbed _Return or normal completion) → return value;
    #    pass _active_exc only for chain-if-finally-raises (per §6.2 context preservation).
    # 3. Otherwise _active_exc set (error in flight, no result) → re-raise after finally.
    if _exc_to_propagate is not None:
      return _async_finally_transition(
        _fin_coro, Null, _exc_to_propagate, q, root_value, root_link, is_nested=is_nested
      )
    elif _sync_result is not Null:
      # Absorbed _Return (or normal pipeline success).  Pass _active_exc — the
      # transition uses it only for finally-chain (pipeline_result is set, so no re-raise).
      return _async_finally_transition(
        _fin_coro, _sync_result, _active_exc, q, root_value, root_link, is_nested=is_nested
      )
    elif _active_exc is not None:
      return _async_finally_transition(_fin_coro, Null, _active_exc, q, root_value, root_link, is_nested=is_nested)
    else:
      # Defensive: nothing to propagate. Close to avoid ResourceWarning.
      if hasattr(_fin_coro, 'close'):
        _fin_coro.close()
  return None


def _resolve_root_link(
  q: Q[Any],
  v: Any,
  args: tuple[Any, ...] | None,
  kwargs: dict[str, Any] | None,
  root_link: Link | None,
  has_run_value: bool,
) -> tuple[Link | None, Link | None]:
  """Resolve the starting link and root_link for execution.

  Handles run-value dispatch, kwargs-only dispatch (caller args replace
  root's build-time args), and passthrough. Validates root_link invariant
  (no ignore_result).
  """
  link: Link | None
  if has_run_value:
    link = Link(v, args, kwargs)
    link.next_link = q._first_link
    root_link = link
  elif (args or kwargs) and root_link is not None:
    # kwargs-only dispatch (v is Null): caller's args replace root's build-time args.
    link = Link(root_link.v, args or None, kwargs or None)
    link.next_link = root_link.next_link
    root_link = link
  elif root_link is not None:
    link = root_link
  else:
    link = q._first_link

  # root_link must never have ignore_result — _run_async's one-shot root capture
  # depends on this; otherwise root_value would capture a side-effect result.
  if root_link is not None and root_link.ignore_result:
    raise QuentException('root_link must not have ignore_result=True')

  return link, root_link


def _should_defer_with(link: Link, deferred_with: bool) -> bool:
  """True if this is the terminal with_/with_do link that should be deferred."""
  if deferred_with and link.next_link is None:
    op: _PipelineOp | Any = link.v
    _lname = getattr(op, '_link_name', None)
    if _lname == 'with_' or _lname == 'with_do':
      return True
  return False


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
  """Sync execution engine — walks the Link list, threading current value.

  Two-tier model: execution starts here, sync. On the first awaitable result,
  delegates to _run_async with all accumulated state. _run_async picks up
  exactly where we left off.

  finally-handler logic runs as normal code after try/except (not in finally:)
  to avoid `return` inside finally. ignore_finally=True guard prevents the
  finally block from re-running when delegating to _run_async (which has its
  own finally handling).
  """
  __tracebackhide__ = True
  root_link: Link | None = q._root_link
  current_value: Any = Null
  root_value: Any = Null
  has_run_value = v is not Null
  has_root_value = has_run_value or root_link is not None
  ignore_finally = False
  _on_step = type(q).on_step
  if _on_step is not None or _log.isEnabledFor(_DEBUG_LEVEL):
    _on_step, _debug, _needs_timing = _timing_ctx(q, _on_step)
    _exec_id = _next_exec_id()
  else:
    _debug = False
    _needs_timing = False
    _exec_id = 0
  _active_exc: BaseException | None = None
  _sync_result: Any = Null
  _exc_to_propagate: BaseException | None = None
  _fin_override: Any = None
  # First-link guard captures root_value and initializes current_value on the
  # first link only. Async path handles this in its one-shot section.
  first_link_processed = False
  link: Link | None = None

  if _debug:
    _log.debug('[exec:%06x] pipeline %r: run started', _exec_id, q)

  try:
    # Fast path: non-callable run(v) with no args — skip Link allocation entirely.
    # Disabled with on_step/debug active to preserve callback behavior.
    if has_run_value and not args and not kwargs and not callable(v) and not _needs_timing:
      current_value = v
      root_value = v
      first_link_processed = True
      link = q._first_link
    else:
      link, root_link = _resolve_root_link(q, v, args, kwargs, root_link, has_run_value)

      # Safety net (also validated by _resolve_root_link).
      if root_link is not None and root_link.ignore_result:
        raise QuentException('root_link must not have ignore_result=True')

    # Link-walk loop (sync path). Async counterpart in _run_async.
    # Shared: _record_step. Differences: sync checks _isawaitable + delegates;
    # async awaits inline. first_link_processed is sync-only.
    _t0 = 0
    _input_value = None
    while link is not None:
      if _should_defer_with(link, deferred_with):
        break
      if _needs_timing:
        if not first_link_processed and has_run_value:
          _input_value = _null_to_none(v)
        else:
          _input_value = _null_to_none(current_value)
        _t0 = _perf_counter_ns()
      result = _evaluate_value(link, current_value)
      # Fast path: reject common sync return types without _isawaitable.
      # CoroutineType is exact-type checked first; frozenset rejects ints/strings
      # in ~30ns vs ~350ns.
      if type(result) is CoroutineType or (
        result is not None and type(result) not in _SYNC_TYPES and _isawaitable(result)
      ):
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
        # (a) root_value: first link's result becomes the root for except_/finally_.
        if has_root_value and root_value is Null:
          root_value = result
        # (b) current_value: stays Null after a .do() first link; init from result
        # otherwise so next link's calling convention sees a non-Null CV.
        if current_value is Null and not link.ignore_result:
          current_value = result
      if not link.ignore_result:
        current_value = result
      link = link.next_link

    # Loop walked to end of list, or deferred_with broke out at the last _WithOp.
    if link is not None and not deferred_with:
      raise QuentException('link-walk loop exited with link still set')

    if _debug:
      _log.debug('[exec:%06x] pipeline %r: completed -> %s', _exec_id, q, _debug_repr(_null_to_none(current_value)))
    # All-.do() pipelines with no root: current_value stays Null → None.
    _sync_result = _null_to_none(current_value)

  except _Return as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: early return', _exec_id, q)
    # Each Q boundary absorbs its own _Return. Set _active_exc so a finally
    # handler raising during signal propagation sees the signal as __context__.
    _active_exc = exc
    _sync_result = _handle_return_exc(exc)

  except _Break as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: break signal', _exec_id, q)
    if is_nested:
      _active_exc = exc
      _exc_to_propagate = exc
    else:
      msg = (
        'Q.break_() cannot be used outside of a loop or iteration context'
        ' (foreach, foreach_do, iterate, iterate_do, flat_iterate, flat_iterate_do, while_).'
      )
      _q_exc = QuentException(msg)
      _q_exc.__suppress_context__ = True
      # Use the user-visible QuentException for finally-chaining; raw _Break is internal.
      _active_exc = _q_exc
      _exc_to_propagate = _q_exc

  except BaseException as exc:
    if _debug:
      _log_exc_debug(q, link, root_link, exc, exec_id=_exec_id)
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

  # Post-clause: async finally transition, stored exceptions, or sync result.
  # ignore_finally=True early returns in try/except exit before reaching here.
  if _fin_override is not None:
    return _fin_override
  if _exc_to_propagate is not None:
    raise _exc_to_propagate
  return _sync_result


# Async execution lives in _engine_async.py; re-exported for compat.
from ._engine_async import (  # noqa: E402
  _async_except_handler,
  _async_finally_transition,
  _run_async,
  _run_async_finally,  # noqa: F401 — re-exported for _generator.py
)
