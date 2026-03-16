# SPDX-License-Identifier: MIT
"""Conditional operations (if_/else_)."""

from __future__ import annotations

from typing import Any

from ._eval import _evaluate_value, _isawaitable
from ._exc_meta import _set_link_temp_args
from ._link import Link
from ._types import Null, QuentException, _Break, _ControlFlowSignal


class _IfOp:
  """Conditional operation: evaluate a branch based on a predicate."""

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
    """Evaluate the truthy branch (``_v_link``), the falsy branch (``_else_link``),
    or pass ``current_value`` through unchanged when no else is registered.

    Respects ``ignore_result`` on branch links — when the branch is a
    ``.do()`` side-effect, the result is discarded and ``current_value``
    passes through.
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
    """Await a side-effect branch result, then return ``current_value``."""
    __tracebackhide__ = True
    await result
    return current_value

  async def _to_async_pred(self, pred_result: Any, current_value: Any) -> Any:
    """Await an async predicate, then evaluate the appropriate branch."""
    __tracebackhide__ = True
    try:
      pred_result = await pred_result
    except _Break as exc:
      raise QuentException('break_() cannot be used inside an if_() predicate.') from exc
    result = self._eval_branch(pred_result, current_value)
    if _isawaitable(result):
      return await result
    return result

  def __call__(self, current_value: Any = Null) -> Any:
    """Evaluate predicate and dispatch to the truthy or falsy branch.

    The predicate is evaluated via ``_evaluate_value`` using the standard
    2-rule calling convention:

    - **Explicit Args**: ``predicate(*args, **kwargs)`` -- current value NOT passed.
    - **Default**: ``predicate(current_value)`` if callable, or literal truthiness.

    When the predicate is a non-callable literal, ``_evaluate_value``
    returns it as-is and its truthiness is tested.
    """
    __tracebackhide__ = True
    if self._predicate_link is not None:
      try:
        pred_result = _evaluate_value(self._predicate_link, current_value)
      except _Break as exc:
        raise QuentException('break_() cannot be used inside an if_() predicate.') from exc
      except _ControlFlowSignal:
        # _Return propagates to the outer chain — early exit from a predicate
        # is valid (SPEC §5.8).  No cleanup needed (_IfOp holds no resources).
        raise
      except BaseException as exc:
        _set_link_temp_args(exc, self._predicate_link, current_value=current_value)
        raise
      if _isawaitable(pred_result):
        return self._to_async_pred(pred_result, current_value)
    else:
      # When predicate is None, use current_value as the predicate result.
      # Null sentinel is always falsy for predicate purposes (SPEC §5.9).
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
