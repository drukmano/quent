# SPDX-License-Identifier: MIT
"""Conditional operations (if_/else_)."""

from __future__ import annotations

from typing import Any

from ._eval import _evaluate_value, _isawaitable
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import Null, QuentException, _ControlFlowSignal


class _IfOp:
  """Conditional op: evaluate a branch based on a predicate."""

  __slots__ = ('_else_link', '_link_name', '_predicate_link', '_v_link')

  _else_link: Link | None
  _link_name: str
  _predicate_link: Link | None
  _v_link: Link

  def __init__(self, predicate_link: Link | None, v_link: Link) -> None:
    self._predicate_link = predicate_link
    self._v_link = v_link
    self._else_link: Link | None = None
    self._link_name = 'if_'

  def _eval_branch(self, pred_result: Any, current_value: Any) -> Any:
    """Evaluate truthy branch, falsy branch, or pass through when no else.

    Respects ignore_result — .do() branches discard their result.
    """
    __tracebackhide__ = True
    if pred_result:
      link = self._v_link
    elif self._else_link is not None:
      link = self._else_link
    else:
      return current_value
    result = _evaluate_value(link, current_value)
    if link.ignore_result:
      if _isawaitable(result):
        return self._await_and_discard(result, current_value)
      return current_value
    return result

  async def _await_and_discard(self, result: Any, current_value: Any) -> Any:
    __tracebackhide__ = True
    await result
    return current_value

  async def _to_async_pred(self, pred_result: Any, current_value: Any) -> Any:
    """Await async predicate, then evaluate the branch.

    if_() does NOT trap break_/return_ — they propagate outward.
    """
    __tracebackhide__ = True
    pred_result = await pred_result
    result = self._eval_branch(pred_result, current_value)
    if _isawaitable(result):
      return await result
    return result

  def __call__(self, current_value: Any = Null) -> Any:
    __tracebackhide__ = True
    if self._predicate_link is not None:
      try:
        pred_result = _evaluate_value(self._predicate_link, current_value)
      except _ControlFlowSignal:
        # Signals propagate through if_(); _IfOp holds no resources.
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, self._predicate_link, current_value=current_value)
        raise
      if _isawaitable(pred_result):
        return self._to_async_pred(pred_result, current_value)
    else:
      # None predicate → use current_value's truthiness. Null is always falsy.
      pred_result = False if current_value is Null else current_value
    return self._eval_branch(pred_result, current_value)

  def _clone(self) -> _IfOp:
    from ._link import _clone_link

    new_op = _IfOp.__new__(_IfOp)
    new_op._predicate_link = _clone_link(self._predicate_link) if self._predicate_link is not None else None
    new_op._v_link = _clone_link(self._v_link)
    new_op._else_link = _clone_link(self._else_link) if self._else_link is not None else None
    new_op._link_name = self._link_name
    return new_op

  def set_else(self, link: Link) -> None:
    if self._else_link is not None:
      raise QuentException(
        'else_() has already been registered for this if_() — '
        'only one else branch is allowed per if_() (consistent with except_/finally_).'
      )
    self._else_link = link
