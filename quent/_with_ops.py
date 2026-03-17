# SPDX-License-Identifier: MIT
"""Context manager operations (with_/with_do)."""

from __future__ import annotations

from typing import Any

from ._eval import _evaluate_value, _isawaitable, _should_use_async_protocol
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import _ControlFlowSignal

# Private sentinel for _full_async's uninitialized result state.
# Using Null would conflate "no result yet" with a body callable that
# explicitly returns the Null sentinel.
_WITH_UNSET: Any = object()


class _WithOp:
  """Context manager operation: enter value as CM, call fn with context.

  Enters the current pipeline value as a context manager, calls the link's
  callable with the context value, and handles exit properly.  When
  ``ignore_result`` is True, the original value passes through (side-effect mode).

  Execution scenarios and method map::

    Scenario                          Entry point    Helpers used
    --------------------------------  -------------  ----------------------------------
    Sync CM, sync body                __call__       → _sync_cm, _suppressed_result
    Sync CM, async body transition    __call__       → _sync_cm → _to_async, _suppressed_result
    Sync CM, async __exit__           __call__       → _sync_cm → _await_exit_suppress /
                                                       _await_exit_success / _await_exit_signal
    Async CM (or dual-protocol CM     __call__       → _full_async, _suppressed_result
      with running event loop)
  """

  __slots__ = ('_ignore_result', '_link', '_link_name')

  _ignore_result: bool
  _link: Link
  _link_name: str

  def __init__(self, link: Link, ignore_result: bool) -> None:
    self._link = link
    self._ignore_result = ignore_result
    self._link_name = 'with_do' if ignore_result else 'with_'

  def _suppressed_result(self, outer_value: Any) -> Any:
    """Return the appropriate value when an exception is suppressed by __exit__."""
    return outer_value if self._ignore_result else None

  async def _to_async(self, current_value: Any, body_result: Any, outer_value: Any, ctx: Any) -> Any:
    """Await the body result and handle sync __exit__ that may return awaitables."""
    __tracebackhide__ = True
    try:
      body_result = await body_result
    except _ControlFlowSignal as signal:
      try:
        exit_result = current_value.__exit__(None, None, None)
        if _isawaitable(exit_result):
          await exit_result
      except BaseException as exit_exc:
        raise exit_exc from signal
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, ctx=ctx)
      try:
        suppress = current_value.__exit__(type(exc), exc, exc.__traceback__)
        if _isawaitable(suppress):
          suppress = await suppress
      except BaseException as exit_exc:
        raise exit_exc from exc
      if not suppress:
        raise
      return self._suppressed_result(outer_value)
    else:
      exit_result = current_value.__exit__(None, None, None)
      if _isawaitable(exit_result):
        await exit_result
      if self._ignore_result:
        return outer_value
      return body_result

  async def _full_async(self, current_value: Any) -> Any:
    """Handle a native async context manager (has __aenter__/__aexit__)."""
    __tracebackhide__ = True
    outer_value = current_value
    result = _WITH_UNSET
    try:
      ctx = await current_value.__aenter__()
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, ctx='<aenter failed>')
      raise
    try:
      result = _evaluate_value(self._link, ctx)
      if _isawaitable(result):
        result = await result
    except _ControlFlowSignal as signal:
      try:
        await current_value.__aexit__(None, None, None)
      except BaseException as exit_exc:
        raise exit_exc from signal
      raise
    except BaseException as exc:
      result = _WITH_UNSET
      _set_link_temp_args(exc, self._link, ctx=ctx)
      try:
        suppress = await current_value.__aexit__(type(exc), exc, exc.__traceback__)
      except BaseException as exit_exc:
        raise exit_exc from exc
      if not suppress:
        raise
      return self._suppressed_result(outer_value)
    else:
      await current_value.__aexit__(None, None, None)
    if result is _WITH_UNSET:
      return self._suppressed_result(outer_value)
    if self._ignore_result:
      return outer_value
    return result

  async def _await_exit_suppress(self, suppress: Any, exc: BaseException, outer_value: Any) -> Any:
    """Await an async __exit__ that may suppress the exception."""
    __tracebackhide__ = True
    try:
      if await suppress:
        return self._suppressed_result(outer_value)
    except BaseException as exit_exc:
      raise exit_exc from exc
    raise exc

  async def _await_exit_success(self, exit_result: Any, outer_value: Any, result: Any) -> Any:
    """Await an async __exit__ on the success path."""
    __tracebackhide__ = True
    await exit_result
    if self._ignore_result:
      return outer_value
    return result

  async def _await_exit_signal(self, exit_result: Any, signal: _ControlFlowSignal) -> Any:
    """Await an async __exit__ on the control flow signal path, then re-raise."""
    __tracebackhide__ = True
    try:
      await exit_result
    except BaseException as exit_exc:
      raise exit_exc from signal
    raise signal

  def _sync_cm(self, cm: Any, outer_value: Any) -> Any:
    """Execute the sync context manager lifecycle: __enter__ -> body -> __exit__.

    Handles all 6 exit scenarios:
      a. ControlFlowSignal + __exit__(None, None, None)
      b. Exception + __exit__ suppresses
      c. Exception + __exit__ does not suppress
      d. Exception + async __exit__
      e. Success + sync __exit__
      f. Success + async __exit__
    """
    __tracebackhide__ = True
    try:
      ctx = cm.__enter__()
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, ctx='<enter failed>')
      raise
    try:
      result = _evaluate_value(self._link, ctx)
      if _isawaitable(result):
        return self._to_async(cm, result, outer_value, ctx)
    except _ControlFlowSignal as signal:
      try:
        exit_result = cm.__exit__(None, None, None)
        if _isawaitable(exit_result):
          return self._await_exit_signal(exit_result, signal)
      except BaseException as exit_exc:
        raise exit_exc from signal
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, ctx=ctx)
      try:
        suppress = cm.__exit__(type(exc), exc, exc.__traceback__)
      except BaseException as exit_exc:
        raise exit_exc from exc
      if _isawaitable(suppress):
        return self._await_exit_suppress(suppress, exc, outer_value)
      if not suppress:
        raise
      return self._suppressed_result(outer_value)
    else:
      exit_result = cm.__exit__(None, None, None)
      if _isawaitable(exit_result):
        return self._await_exit_success(exit_result, outer_value, result)
      if self._ignore_result:
        return outer_value
      return result

  def __call__(self, current_value: Any) -> Any:
    """Enter current_value as a context manager, call fn with context value.

    Detects the CM protocol and dispatches:
    - Async (or dual-protocol with running loop) -> _full_async
    - Sync -> _sync_cm
    - Neither protocol -> TypeError
    """
    __tracebackhide__ = True
    _use_async = _should_use_async_protocol(current_value, '__enter__', '__aenter__')
    if _use_async is True:
      return self._full_async(current_value)
    if _use_async is None:
      msg = (
        f'{type(current_value).__name__} object does not support the context manager protocol '
        f'(__enter__/__exit__ or __aenter__/__aexit__). '
        f'Ensure the pipeline value at this step is a context manager.'
      )
      raise TypeError(msg)
    return self._sync_cm(current_value, current_value)
