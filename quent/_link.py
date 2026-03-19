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

  Each Link holds a callable (or raw value, or nested Q) plus the
  arguments it should be called with.  Links are appended via the pipeline's
  tail pointer for O(1) insertion and evaluated sequentially during
  ``_run()``.
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
    """Create a new Link node.

    Args:
      v: The callable, value, or nested Q to evaluate.
      args: Optional positional arguments passed when ``v`` is invoked.
      kwargs: Optional keyword arguments passed when ``v`` is invoked.
      ignore_result: If True, the result of evaluating this link is discarded
        (the previous pipeline value passes through unchanged).
      original_value: The original user-provided value, stored for traceback
        display when ``v`` has been wrapped by an operation factory.
    """
    # Duck-typing pipeline detection: Q sets `_quent_is_q = True` and is always
    # callable (has __call__).  Using getattr() avoids a circular import.
    # The callable() check is reused for is_callable to avoid a redundant call.
    self.v = v
    _is_callable = callable(v)
    _is_q = _is_callable and getattr(v, '_quent_is_q', False)
    self.is_q = _is_q
    self.is_callable = _is_callable and not _is_q
    # Build-time enforcement: non-callable values cannot receive args/kwargs.
    # Pipelines (is_q) are excluded — they have their own args dispatch (Rule 1).
    if (args or kwargs) and not self.is_callable and not self.is_q:
      msg = f'Arguments were provided but the value is not callable (got {type(v).__name__})'
      raise TypeError(msg)
    # Normalize empty tuple to None (no args is semantically None, not ()).
    self.args = args or None
    self.kwargs = kwargs or None
    self.ignore_result = ignore_result
    self.next_link = None
    self.original_value = original_value


# Sentinel used in clone()'s slot-completeness check.
# Uninitialized __slots__ attributes raise AttributeError on access; using
# getattr(new, s, _CLONE_SENTINEL) safely detects them without try/except.
_CLONE_SENTINEL = object()


_IfOp_cls: type[_IfOp] | None = None
_WhileOp_cls: type[_WhileOp] | None = None


def _clone_link(link: Link) -> Link:
  """Create a shallow copy of a Link node (next_link is always None).

  Nested pipelines (``link.is_q is True``) are deep-cloned via
  ``link.v.clone()`` to prevent cross-clone state sharing (e.g.
  concurrent execution of shared inner pipelines).

  Operations with internal Link objects (``_IfOp``, ``_WhileOp``) get a
  fresh copy so each clone has independent nested chains.

  **Invariant:** Any operation factory that attaches mutable state to its
  operation must be handled here.  Currently ``if_`` (via ``_IfOp``) and
  ``while_`` (via ``_WhileOp``) carry internal Link state.  If a new
  stateful operation is added, update the cloning logic below accordingly.
  """
  global _IfOp_cls, _WhileOp_cls
  if _IfOp_cls is None:
    # Thread-safety note: this double-checked locking pattern is safe because
    # Python's import lock (retained under PEP 703 / free-threaded builds)
    # serializes the actual import. Concurrent threads may both enter this
    # branch, but both will import the same class object and the redundant
    # assignment is benign (same value, no torn read risk).
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
  # Shallow-copy kwargs (mutable dict) so original and clone don't share
  # the same dict object.  args is a tuple (immutable) so reference copy is safe.
  new.kwargs = dict(link.kwargs) if link.kwargs is not None else None
  new.ignore_result = link.ignore_result
  new.original_value = link.original_value
  new.next_link = None
  # Deep-clone nested pipelines to prevent cross-clone state sharing.
  # Without this, two clones of a pipeline containing a nested pipeline would
  # share that inner pipeline's mutable state.
  if link.is_q:
    new.v = link.v.clone()
  # _IfOp carries mutable state (_else_link) and inner Link objects
  # (_predicate_link, _v_link, _else_link) that must be deep-copied.
  # Other operation classes are immutable and don't need cloning.
  elif isinstance(link.v, (_IfOp_cls, _WhileOp_cls)):
    new.v = link.v._clone()
  # Verify all slots are initialized — catches new slots added to Link
  # without corresponding updates to _clone_link.
  if __debug__:
    for _s in Link.__slots__:
      if getattr(new, _s, _CLONE_SENTINEL) is _CLONE_SENTINEL:
        raise QuentException(f'_clone_link: slot {_s!r} not initialized in cloned Link')
  return new
