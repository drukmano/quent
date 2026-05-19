# SPDX-License-Identifier: MIT
"""Link — atomic operation node in a pipeline's singly-linked list."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._types import QuentException, _UncopyableMixin

if TYPE_CHECKING:
  from ._if_ops import _IfOp
  from ._while_ops import _WhileOp


class Link(_UncopyableMixin):
  """Atomic operation node in a pipeline's singly-linked list.

  Holds a callable (or raw value, or nested Q) plus the args it should be
  called with. Appended via the pipeline's tail pointer for O(1) insertion.
  """

  __slots__ = (
    'args',
    'ignore_result',
    'is_callable',
    'is_q',
    'kwargs',
    'next_link',
    'original_value',
    'v',
  )

  v: Any
  next_link: Link | None
  ignore_result: bool
  is_callable: bool
  args: tuple[Any, ...] | None
  kwargs: dict[str, Any] | None
  original_value: Any
  is_q: bool

  def __init__(
    self,
    v: Any,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    ignore_result: bool = False,
    original_value: Any | None = None,
  ) -> None:
    # Duck-typing: Q sets `_quent_is_q = True` and is always callable.
    # getattr() avoids the circular import.
    self.v = v
    _is_callable = callable(v)
    _is_q = _is_callable and getattr(v, '_quent_is_q', False)
    self.is_q = _is_q
    self.is_callable = _is_callable and not _is_q
    # Non-callable values cannot receive args/kwargs (pipelines do — Rule 1).
    if (args or kwargs) and not self.is_callable and not self.is_q:
      msg = f'Arguments were provided but the value is not callable (got {type(v).__name__})'
      raise TypeError(msg)
    self.args = args or None
    self.kwargs = kwargs or None
    self.ignore_result = ignore_result
    self.next_link = None
    self.original_value = original_value


# Sentinel for clone()'s slot-completeness check — uninitialized __slots__
# raise AttributeError on access; `getattr(new, s, _CLONE_SENTINEL)` detects them.
_CLONE_SENTINEL = object()


_IfOp_cls: type[_IfOp] | None = None
_WhileOp_cls: type[_WhileOp] | None = None


def _clone_link(link: Link) -> Link:
  """Shallow copy of a Link (next_link is None).

  Nested pipelines and stateful ops (_IfOp, _WhileOp) are deep-cloned so
  each clone has independent state. Any new stateful op must be added here.
  """
  global _IfOp_cls, _WhileOp_cls
  if _IfOp_cls is None:
    # Python's import lock serializes the import; redundant assignment is benign.
    from ._if_ops import _IfOp

    _IfOp_cls = _IfOp
  if _WhileOp_cls is None:
    from ._while_ops import _WhileOp

    _WhileOp_cls = _WhileOp

  new = Link.__new__(Link)
  new.v = link.v
  new.is_q = link.is_q
  new.is_callable = link.is_callable
  new.args = link.args
  # kwargs is mutable; args is a tuple so reference copy is safe.
  new.kwargs = dict(link.kwargs) if link.kwargs is not None else None
  new.ignore_result = link.ignore_result
  new.original_value = link.original_value
  new.next_link = None
  if link.is_q:
    new.v = link.v.clone()
  elif isinstance(link.v, (_IfOp_cls, _WhileOp_cls)):
    new.v = link.v._clone()
  if __debug__:
    for _s in Link.__slots__:
      if getattr(new, _s, _CLONE_SENTINEL) is _CLONE_SENTINEL:
        raise QuentException(f'_clone_link: slot {_s!r} not initialized in cloned Link')
  return new
