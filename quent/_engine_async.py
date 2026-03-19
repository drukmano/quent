# SPDX-License-Identifier: MIT
"""Async execution paths for the quent pipeline engine.

This module contains the async continuation, async except-handler dispatch,
async finally handling, and async transition functions.  The sync execution
path and all shared helpers live in ``_engine.py``.

Separated from ``_engine.py`` to keep the sync and async code paths in
focused, manageable modules.
"""

from __future__ import annotations

import logging
from types import CoroutineType
from typing import TYPE_CHECKING, Any

from ._engine import (
  _DEBUG_LEVEL,
  _SYNC_TYPES,
  _await_finally_result,
  _chain_finally_exc,
  _debug_repr,
  _except_handler_body,
  _except_handler_failed,
  _except_handler_succeeded,
  _finally_handler_body,
  _handle_base_exception,
  _log_exc_debug,
  _null_to_none,
  _perf_counter_ns,
  _record_finally_step,
  _record_step,
  _should_defer_with,
  _signal_in_handler_msg,
  _timing_ctx,
)
from ._eval import (
  _evaluate_value,
  _handle_return_exc,
  _isawaitable,
)
from ._link import Link
from ._types import (
  Null,
  QuentException,
  _Break,
  _ControlFlowSignal,
  _Return,
)

if TYPE_CHECKING:
  from ._q import Q

_log = logging.getLogger('quent')


# ---- Async finally handling ----


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
  _on_step, _debug, _needs_timing = _timing_ctx(q)
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
      )
  except BaseException as finally_exc:
    if active_exc is not None:
      _chain_finally_exc(finally_exc, active_exc)
    raise


# ---- Async except-handler dispatch ----


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
  _on_step, _debug, _needs_timing = _timing_ctx(q)
  result: Any = None
  # Save the original context chain so we can restore it if the handler fails
  # with reraise=True (prevents the handler's exception from permanently
  # mutating the original exception's __context__/__suppress_context__).
  _orig_context = exc.__context__
  _orig_suppress = exc.__suppress_context__
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
      if _except_handler_failed(exc, handler_exc, reraise, _orig_context, _orig_suppress):
        pass  # Absorbed: reraise=True handler failed with Exception — falls through to reraise below
      elif reraise:
        # Non-Exception BaseException (KeyboardInterrupt, SystemExit) with reraise=True.
        # _except_handler_failed already cleaned exc metadata.
        _active_exc = handler_exc  # Ensure finally sees the true active exception
        raise  # BaseException propagates naturally
      else:
        # reraise=False: handler's exception propagates with original as __cause__.
        # _except_handler_failed already cleaned exc metadata.
        _active_exc = handler_exc
        raise handler_exc from exc
    _t0 = sync_t0 if sync_t0 else _perf_counter_ns()
    _except_handler_succeeded(
      exc,
      result,
      q,
      root_value,
      root_link,
      _t0,
      _needs_timing,
      _on_step,
      _debug,
      exec_id,
    )
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
  # Save the original context chain so we can restore it if the handler fails
  # with reraise=True (prevents the handler's exception from permanently
  # mutating the original exception's __context__/__suppress_context__).
  _orig_context = exc.__context__
  _orig_suppress = exc.__suppress_context__
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
        if _except_handler_failed(exc, handler_exc, q._on_except_reraise, _orig_context, _orig_suppress):
          # Absorbed: reraise=True handler failed with Exception — re-raise original.
          raise exc  # noqa: B904
        elif q._on_except_reraise:
          # Non-Exception BaseException (KeyboardInterrupt, SystemExit) with reraise=True.
          # _except_handler_failed already cleaned exc metadata.
          _active_exc = handler_exc  # Ensure finally sees the true active exception
          raise  # BaseException propagates naturally
        else:
          # reraise=False: handler's exception propagates with original as __cause__.
          # _except_handler_failed already cleaned exc metadata.
          _active_exc = handler_exc
          raise handler_exc from exc
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
      exec_id,
    )
    if q._on_except_reraise:
      raise exc
  except _ControlFlowSignal as signal:
    if signal.__context__ is None:
      signal.__context__ = exc
    qe = QuentException(_signal_in_handler_msg(signal, 'except'))
    _active_exc = qe
    raise qe from exc
  # Exception consumed by handler (reraise=False) — _except_handler_succeeded
  # already cleaned metadata above.
  return None, result


# ---- Async execution engine ----


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
    _on_step, _debug, _needs_timing = _timing_ctx(q, on_step)
  else:
    _on_step = None
    _debug = False
    _needs_timing = False

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
      result = _evaluate_value(link, current_value)
      # Fast path: reject common sync return types without calling _isawaitable.
      if type(result) is CoroutineType or (
        result is not None and type(result) not in _SYNC_TYPES and _isawaitable(result)
      ):
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
