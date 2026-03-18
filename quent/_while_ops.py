# SPDX-License-Identifier: MIT
"""While-loop operations (while_)."""

from __future__ import annotations

from typing import Any

from ._eval import _eval_signal_value, _evaluate_value, _isawaitable
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import _EMPTY_TUPLE, Null, _Break, _ControlFlowSignal


class _WhileOp:
  """While-loop operation: repeatedly evaluate a body while a predicate is truthy."""

  __slots__ = ('_body_link', '_ignore_result', '_link_name', '_predicate_link')

  _predicate_link: Link | None
  _body_link: Link
  _ignore_result: bool
  _link_name: str

  def __init__(self, predicate_link: Link | None, body_link: Link, ignore_result: bool) -> None:
    self._predicate_link = predicate_link
    self._body_link = body_link
    self._ignore_result = ignore_result
    self._link_name = 'while_'

  def _handle_break(self, exc: _Break, current_value: Any) -> Any:
    """Handle a _Break signal from the loop body or predicate."""
    try:
      if exc.value is Null:
        return current_value
      result = _eval_signal_value(exc.value, exc.signal_args, exc.signal_kwargs)
      if _isawaitable(result):
        return _await_break_value(result)
      return result
    finally:
      exc.value = Null
      exc.signal_args = _EMPTY_TUPLE
      exc.signal_kwargs = None

  def __call__(self, current_value: Any = Null) -> Any:
    """Evaluate the while loop: sync fast path with async transition on first awaitable."""
    __tracebackhide__ = True
    # Step 1: Evaluate predicate
    if self._predicate_link is not None:
      try:
        pred = _evaluate_value(self._predicate_link, current_value)
      except _Break as exc:
        return self._handle_break(exc, current_value)
      except _ControlFlowSignal:
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, self._predicate_link, current_value=current_value)
        raise
      if _isawaitable(pred):
        return self._async_from_pred(pred, current_value)
    else:
      pred = False if current_value is Null else current_value

    # Step 2: Loop while predicate is truthy
    while pred:
      # Evaluate body
      try:
        result = _evaluate_value(self._body_link, current_value)
      except _Break as exc:
        return self._handle_break(exc, current_value)
      except _ControlFlowSignal:
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, self._body_link, current_value=current_value)
        raise
      if _isawaitable(result):
        return self._async_from_body(result, current_value)
      if not self._ignore_result:
        current_value = result

      # Re-evaluate predicate
      if self._predicate_link is not None:
        try:
          pred = _evaluate_value(self._predicate_link, current_value)
        except _Break as exc:
          return self._handle_break(exc, current_value)
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _set_link_temp_args(exc, self._predicate_link, current_value=current_value)
          raise
        if _isawaitable(pred):
          return self._async_from_pred(pred, current_value)
      else:
        pred = False if current_value is Null else current_value

    return current_value

  async def _async_from_pred(self, pred_awaitable: Any, current_value: Any) -> Any:
    """Transition to async after an awaitable predicate result."""
    __tracebackhide__ = True
    try:
      pred = await pred_awaitable
    except _Break as exc:
      return self._handle_break(exc, current_value)
    if not pred:
      return current_value
    return await self._full_async_body_first(current_value)

  async def _async_from_body(self, result_awaitable: Any, current_value: Any) -> Any:
    """Transition to async after an awaitable body result."""
    __tracebackhide__ = True
    try:
      result = await result_awaitable
    except _Break as exc:
      return self._handle_break(exc, current_value)
    if not self._ignore_result:
      current_value = result
    return await self._full_async(current_value)

  async def _full_async_body_first(self, current_value: Any) -> Any:
    """Evaluate body first (predicate was already truthy), then enter full async loop."""
    __tracebackhide__ = True
    try:
      result = _evaluate_value(self._body_link, current_value)
    except _Break as exc:
      return self._handle_break(exc, current_value)
    except _ControlFlowSignal:
      raise
    except BaseException as exc:
      _set_link_temp_args(exc, self._body_link, current_value=current_value)
      raise
    if _isawaitable(result):
      try:
        result = await result
      except _Break as exc:
        return self._handle_break(exc, current_value)
    if not self._ignore_result:
      current_value = result
    return await self._full_async(current_value)

  async def _full_async(self, current_value: Any) -> Any:
    """Full async while loop: predicate and body both awaited."""
    __tracebackhide__ = True
    while True:
      # Evaluate predicate
      if self._predicate_link is not None:
        try:
          pred = _evaluate_value(self._predicate_link, current_value)
        except _Break as exc:
          return self._handle_break(exc, current_value)
        except _ControlFlowSignal:
          raise
        except BaseException as exc:
          _set_link_temp_args(exc, self._predicate_link, current_value=current_value)
          raise
        if _isawaitable(pred):
          try:
            pred = await pred
          except _Break as exc:
            return self._handle_break(exc, current_value)
      else:
        pred = False if current_value is Null else current_value

      if not pred:
        return current_value

      # Evaluate body
      try:
        result = _evaluate_value(self._body_link, current_value)
      except _Break as exc:
        return self._handle_break(exc, current_value)
      except _ControlFlowSignal:
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, self._body_link, current_value=current_value)
        raise
      if _isawaitable(result):
        try:
          result = await result
        except _Break as exc:
          return self._handle_break(exc, current_value)
      if not self._ignore_result:
        current_value = result

  def _clone(self) -> _WhileOp:
    from ._link import _clone_link

    new_op = _WhileOp.__new__(_WhileOp)
    new_op._predicate_link = _clone_link(self._predicate_link) if self._predicate_link is not None else None
    new_op._body_link = _clone_link(self._body_link)
    new_op._ignore_result = self._ignore_result
    new_op._link_name = self._link_name
    return new_op


async def _await_break_value(result: Any) -> Any:
  """Await an async break value."""
  return await result
