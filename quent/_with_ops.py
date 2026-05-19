# SPDX-License-Identifier: MIT
"""Context manager operations (with_/with_do)."""

from __future__ import annotations

from typing import Any

from ._eval import _evaluate_value, _isawaitable, _should_use_async_protocol
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import _ControlFlowSignal

# "No result yet" — distinct from Null because a body callable could legitimately
# return Null. See _types.py for the sentinel landscape.
_WITH_UNSET: Any = object()


async def _async_cm_exit(cm: Any, is_async_cm: bool, *args: Any) -> Any:
  """Call __aexit__ (async CM) or __exit__ (sync CM, awaiting if awaitable)."""
  if is_async_cm:
    return await cm.__aexit__(*args)
  result = cm.__exit__(*args)
  if _isawaitable(result):
    return await result
  return result


class _WithOp:
  """Enter pipeline value as a CM, call fn with the context, handle exit.

  When ignore_result is True, the original value passes through (side-effect mode).
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
    return outer_value if self._ignore_result else None

  async def _to_async(self, current_value: Any, body_result: Any, outer_value: Any, ctx: Any) -> Any:
    """Await body result; handle sync __exit__ that may return awaitables."""
    __tracebackhide__ = True
    try:
      body_result = await body_result
    except _ControlFlowSignal as signal:
      try:
        await _async_cm_exit(current_value, False, None, None, None)
      except BaseException as exit_exc:
        raise exit_exc from signal
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._link, ctx=ctx)
      try:
        suppress = await _async_cm_exit(current_value, False, type(exc), exc, exc.__traceback__)
      except BaseException as exit_exc:
        raise exit_exc from exc
      if not suppress:
        raise
      return self._suppressed_result(outer_value)
    else:
      await _async_cm_exit(current_value, False, None, None, None)
      if self._ignore_result:
        return outer_value
      return body_result

  async def _full_async(self, current_value: Any) -> Any:
    """Native async CM (has __aenter__/__aexit__)."""
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
        await _async_cm_exit(current_value, True, None, None, None)
      except BaseException as exit_exc:
        raise exit_exc from signal
      raise
    except BaseException as exc:
      result = _WITH_UNSET
      _set_link_temp_args(exc, self._link, ctx=ctx)
      try:
        suppress = await _async_cm_exit(current_value, True, type(exc), exc, exc.__traceback__)
      except BaseException as exit_exc:
        raise exit_exc from exc
      if not suppress:
        raise
      return self._suppressed_result(outer_value)
    else:
      await _async_cm_exit(current_value, True, None, None, None)
    if result is _WITH_UNSET:
      return self._suppressed_result(outer_value)
    if self._ignore_result:
      return outer_value
    return result

  async def _await_exit_suppress(self, suppress: Any, exc: BaseException, outer_value: Any) -> Any:
    __tracebackhide__ = True
    try:
      if await suppress:
        return self._suppressed_result(outer_value)
    except BaseException as exit_exc:
      raise exit_exc from exc
    raise exc

  async def _await_exit_success(self, exit_result: Any, outer_value: Any, result: Any) -> Any:
    __tracebackhide__ = True
    await exit_result
    if self._ignore_result:
      return outer_value
    return result

  async def _await_exit_signal(self, exit_result: Any, signal: _ControlFlowSignal) -> Any:
    __tracebackhide__ = True
    try:
      await exit_result
    except BaseException as exit_exc:
      raise exit_exc from signal
    raise signal

  def _sync_cm(self, cm: Any, outer_value: Any) -> Any:
    """Sync CM lifecycle: __enter__ → body → __exit__. Handles 6 exit paths."""
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
