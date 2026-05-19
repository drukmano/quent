# SPDX-License-Identifier: MIT
"""Async execution paths — continuation, except/finally dispatch, transition.

The sync path and shared helpers live in ``_engine.py``.
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
  _Exit,
  _Return,
)

if TYPE_CHECKING:
  from ._q import Q

_log = logging.getLogger('quent')


async def _async_finally_transition(
  finally_result: Any,
  pipeline_result: Any,
  active_exc: BaseException | None,
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  is_nested: bool = False,
) -> Any:
  """Async transition for a sync pipeline whose finally handler returned a coroutine.

  Awaits the finally, then returns the pipeline result or re-raises active_exc.
  Preserves the pipeline's value through the transition.

  ``active_exc`` semantics:
  - If ``pipeline_result is Null`` → ``active_exc`` is the exception to re-raise after finally.
  - If ``pipeline_result is not Null`` → ``active_exc`` is a chain hint only:
    if finally raises, chain it as ``__context__``; otherwise return ``pipeline_result``
    (this is the §6.2 absorbed-signal-with-context-preservation case).
  """
  __tracebackhide__ = True
  try:
    await _await_finally_result(finally_result, q, root_value, root_link, is_nested=is_nested)
  except BaseException as finally_exc:
    if active_exc is not None:
      _chain_finally_exc(finally_exc, active_exc)
    raise
  # Re-raise active_exc only when there's no pipeline_result — i.e., the active
  # exception IS what propagates (error path, not absorbed signal).
  if active_exc is not None and pipeline_result is Null:
    raise active_exc
  # Await if the pipeline result is itself a coroutine (e.g. Q.return_(async_fn)).
  if _isawaitable(pipeline_result):
    return await pipeline_result
  return None if pipeline_result is Null else pipeline_result


async def _run_async_finally(
  q: Q[Any],
  root_value: Any,
  root_link: Link | None,
  active_exc: BaseException | None,
  is_nested: bool = False,
  exec_id: int = 0,
) -> None:
  """Execute the finally handler in the async path."""
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
  """Async transition: await except handler, re-raise or return per reraise."""
  __tracebackhide__ = True
  _active_exc: BaseException | None = exc
  _on_step, _debug, _needs_timing = _timing_ctx(q)
  result: Any = None
  # Save original context chain so we can restore it if reraise=True handler fails.
  _orig_context = exc.__context__
  _orig_suppress = exc.__suppress_context__
  try:
    try:
      if reraise:
        await handler_coro
      else:
        result = await handler_coro
    except _Exit:
      raise
    except _ControlFlowSignal as signal:
      if signal.__context__ is None:
        signal.__context__ = exc
      msg = _signal_in_handler_msg(signal, 'except')
      # `from exc`: original exc is the most relevant context regardless of reraise.
      # No _clean_exc_meta — code after raise is unreachable, and exc meta was
      # already cleaned by _modify_traceback in _except_handler_body.
      raise QuentException(msg) from exc
    except BaseException as handler_exc:
      if _except_handler_failed(exc, handler_exc, reraise, _orig_context, _orig_suppress):
        pass  # Absorbed — falls through to reraise below.
      elif reraise:
        # Non-Exception BaseException (KI/SystemExit) with reraise=True.
        _active_exc = handler_exc
        raise
      else:
        # reraise=False: handler exc propagates with original as __cause__.
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
    # Consumed — finally sees success-path context.
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
  """Dispatch the except handler on the async path; awaits inline.

  All exception paths raise directly so the caller's finally sees the correct
  _active_exc via the propagating exception.
  """
  __tracebackhide__ = True
  _active_exc: BaseException | None = exc
  _t0_exc = 0
  _orig_context = exc.__context__
  _orig_suppress = exc.__suppress_context__
  # Handler result accounting. We raise OUTSIDE any active `except` block so
  # Python's auto-chaining of __context__ doesn't undo restoration that
  # _except_handler_failed performed.
  result: Any = Null
  _propagate_original = False
  _propagate_handler_exc: BaseException | None = None
  _chain_from_exc = False
  try:
    try:
      if _needs_timing:
        _t0_exc = _perf_counter_ns()
      result = _except_handler_body(exc, q, link, root_link, root_value, is_nested=is_nested)
    except BaseException as propagating_exc:
      if propagating_exc is exc:
        # Filter mismatch — handler not invoked. Propagate as-is.
        _active_exc = exc
        raise
      if _except_handler_failed(exc, propagating_exc, q._on_except_reraise, _orig_context, _orig_suppress):
        # Absorbed — raise original outside except block (raising here would auto-chain
        # __context__ to propagating_exc).
        _active_exc = exc
        _propagate_original = True
      else:
        _active_exc = propagating_exc
        raise
    else:
      if _isawaitable(result):
        try:
          result = await result
        except _Exit:
          raise
        except _ControlFlowSignal as signal:
          if signal.__context__ is None:
            signal.__context__ = exc
          raise QuentException(_signal_in_handler_msg(signal, 'except')) from exc
        except BaseException as handler_exc:
          if _except_handler_failed(exc, handler_exc, q._on_except_reraise, _orig_context, _orig_suppress):
            _active_exc = exc
            _propagate_original = True
          elif q._on_except_reraise:
            # KI/SystemExit with reraise=True.
            _active_exc = handler_exc
            _propagate_handler_exc = handler_exc
          else:
            # reraise=False: handler exc propagates with original as __cause__.
            _active_exc = handler_exc
            _propagate_handler_exc = handler_exc
            _chain_from_exc = True
      if not _propagate_original and _propagate_handler_exc is None:
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
          _active_exc = exc
          _propagate_original = True
  except _Exit:
    raise
  except _ControlFlowSignal as signal:
    if signal.__context__ is None:
      signal.__context__ = exc
    qe = QuentException(_signal_in_handler_msg(signal, 'except'))
    _active_exc = qe
    raise qe from exc
  # Outside the except block — raise without auto-chain interference.
  if _propagate_original:
    raise exc
  if _propagate_handler_exc is not None:
    if _chain_from_exc:
      raise _propagate_handler_exc from exc
    raise _propagate_handler_exc
  # Consumed (reraise=False success).
  return None, result


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
  """Async continuation. Called by _run() on first awaitable; receives the pending
  awaitable and all accumulated state, then continues the link walk in async mode.
  """
  __tracebackhide__ = True
  _active_exc: BaseException | None = None
  if on_step is not None or _log.isEnabledFor(_DEBUG_LEVEL):
    _on_step, _debug, _needs_timing = _timing_ctx(q, on_step)
  else:
    _on_step = None
    _debug = False
    _needs_timing = False

  if _debug:
    _log.debug('[exec:%06x] pipeline %r: async continuation started', exec_id, q)

  # Carried invariant from _run(): root_link never has ignore_result.
  if root_link is not None and root_link.ignore_result:
    raise QuentException('root_link must not have ignore_result=True')

  try:
    # Complete the in-progress step handed off from _run(). One-shot section,
    # not a loop. Handles the same first-link concerns as _run()'s
    # first_link_processed guard: capture root_value, init current_value.
    # Use sync_t0 for accurate end-to-end timing including the sync evaluate.
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

    # Link-walk loop (async path). Sync counterpart in _run().
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

    if next_link is not None and not deferred_with:
      raise QuentException('link-walk loop exited with next_link still set')

    if _debug:
      _log.debug('[exec:%06x] pipeline %r: completed -> %s', exec_id, q, _debug_repr(_null_to_none(current_value)))
    return _null_to_none(current_value)

  except _Return as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: early return', exec_id, q)
    # Each Q absorbs its own _Return. Set _active_exc so finally sees the signal
    # as in-flight exception.
    _active_exc = exc
    result = _handle_return_exc(exc)
    if _isawaitable(result):
      return await result
    return result

  except _Break as exc:
    if _debug:
      _log.debug('[exec:%06x] pipeline %r: break signal', exec_id, q)
    if is_nested:
      _active_exc = exc
      raise
    msg = (
      'Q.break_() cannot be used outside of a loop or iteration context'
      ' (foreach, foreach_do, iterate, iterate_do, flat_iterate, flat_iterate_do, while_).'
    )
    _q_exc = QuentException(msg)
    _q_exc.__suppress_context__ = True
    _active_exc = _q_exc
    raise _q_exc from None

  except BaseException as exc:
    if _debug:
      _log_exc_debug(q, link, root_link, exc, exec_id=exec_id)
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
