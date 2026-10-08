# SPDX-License-Identifier: MIT
"""Link — atomic operation node in a pipeline's singly-linked list."""

from __future__ import annotations

from typing import Any

from ._types import QuentException, _UncopyableMixin


def _is_q(value: Any) -> bool:
  """Recognize pipeline instances without querying dynamic instance attributes."""
  return callable(value) and not isinstance(value, type) and getattr(type(value), '_quent_is_q', False) is True


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
    self.v = v
    _is_callable = callable(v)
    self.is_q = _is_q(v)
    self.is_callable = _is_callable and not self.is_q
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


_CLONE_OP_TYPES: tuple[type[Any], ...] | None = None


def _clone_link(link: Link) -> Link:
  """Shallow copy of a Link (next_link is None).

  Operation wrappers own links too: clone them even when the wrapper itself
  has no mutable execution state, so its nested pipelines remain independent.
  """
  global _CLONE_OP_TYPES
  if _CLONE_OP_TYPES is None:
    # Python's import lock serializes the import; redundant assignment is benign.
    from ._drive_gen_ops import _DriveGenOp
    from ._gather_ops import _ConcurrentGatherOp
    from ._if_ops import _IfOp
    from ._iter_ops import _ConcurrentIterOp, _IterOp
    from ._while_ops import _WhileOp
    from ._with_ops import _WithOp

    _CLONE_OP_TYPES = (_IfOp, _WhileOp, _IterOp, _ConcurrentIterOp, _ConcurrentGatherOp, _WithOp, _DriveGenOp)

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
  elif isinstance(link.v, _CLONE_OP_TYPES):
    new.v = link.v._clone()
    # Visualization must follow the same cloned inner link as execution.
    if isinstance(link.original_value, Link):
      for attr in ('_link', '_v_link', '_body_link'):
        if link.original_value is getattr(link.v, attr, None):
          new.original_value = getattr(new.v, attr)
          break
  if __debug__:
    for _s in Link.__slots__:
      if getattr(new, _s, _CLONE_SENTINEL) is _CLONE_SENTINEL:
        raise QuentException(f'_clone_link: slot {_s!r} not initialized in cloned Link')
  return new
