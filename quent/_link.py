# SPDX-License-Identifier: MIT
"""Link — atomic operation node in a chain's singly-linked list."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._types import _UncopyableMixin

if TYPE_CHECKING:
  from ._if_ops import _IfOp


class Link(_UncopyableMixin):
  """Atomic operation node in a chain's singly-linked list.

  Each Link holds a callable (or raw value, or nested Chain) plus the
  arguments it should be called with.  Links are appended via the chain's
  tail pointer for O(1) insertion and evaluated sequentially during
  ``_run()``.
  """

  __slots__ = (
    'args',
    'ignore_result',
    'is_callable',
    'is_chain',
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
  is_chain: bool

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
      v: The callable, value, or nested Chain to evaluate.
      args: Optional positional arguments passed when ``v`` is invoked.
      kwargs: Optional keyword arguments passed when ``v`` is invoked.
      ignore_result: If True, the result of evaluating this link is discarded
        (the previous pipeline value passes through unchanged).
      original_value: The original user-provided value, stored for traceback
        display when ``v`` has been wrapped by an operation factory.
    """
    # Duck-typing chain detection: Chain sets a class attribute `_quent_is_chain = True`.
    # Using getattr() instead of isinstance() avoids a circular import between
    # _link and _chain.  The except handles exotic descriptors that raise.
    try:
      is_chain = getattr(v, '_quent_is_chain', False)
      if is_chain and not callable(getattr(v, '_run', None)):
        is_chain = False
      self.is_chain = is_chain
    except Exception:
      self.is_chain = False
    self.v = v
    self.is_callable = callable(v)  # cached: avoids repeated callable() in hot path
    # Build-time enforcement: non-callable values cannot receive args/kwargs.
    # Chains (is_chain) are excluded — they have their own args dispatch (Rule 1).
    if (args or kwargs) and not self.is_callable and not self.is_chain:
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


def _clone_link(link: Link) -> Link:
  """Create a shallow copy of a Link node (next_link is always None).

  Nested chains (``link.is_chain is True``) are deep-cloned via
  ``link.v.clone()`` to prevent cross-clone state sharing (e.g.
  concurrent execution of shared inner chains).

  Operations with mutable ``_else_link`` state (``_IfOp`` instances) get a
  fresh copy so each clone is independently modifiable.

  **Invariant:** Any operation factory that attaches mutable state to its
  operation must be handled here.  Currently only ``if_`` (via ``_IfOp``)
  carries mutable ``_else_link`` state.  If a new stateful operation is
  added, update the cloning logic below accordingly.
  """
  global _IfOp_cls
  if _IfOp_cls is None:
    # Thread-safety note: this double-checked locking pattern is safe because
    # Python's import lock (retained under PEP 703 / free-threaded builds)
    # serializes the actual import. Concurrent threads may both enter this
    # branch, but both will import the same class object and the redundant
    # assignment is benign (same value, no torn read risk).
    from ._if_ops import _IfOp

    _IfOp_cls = _IfOp

  new = Link.__new__(Link)
  new.v = link.v
  new.is_chain = link.is_chain
  new.is_callable = link.is_callable
  new.args = link.args
  # Shallow-copy kwargs (mutable dict) so original and clone don't share
  # the same dict object.  args is a tuple (immutable) so reference copy is safe.
  new.kwargs = dict(link.kwargs) if link.kwargs is not None else None
  new.ignore_result = link.ignore_result
  new.original_value = link.original_value
  new.next_link = None
  # Deep-clone nested chains to prevent cross-clone state sharing.
  # Without this, two clones of a chain containing a nested chain would
  # share that inner chain's mutable state.
  if link.is_chain:
    new.v = link.v.clone()
  # _IfOp carries mutable state (_else_link) and inner Link objects
  # (_predicate_link, _v_link, _else_link) that must be deep-copied.
  # Other operation classes are immutable and don't need cloning.
  elif isinstance(link.v, _IfOp_cls):
    new.v = link.v._clone()
  if __debug__:
    # Verify all slots are initialized — catches new slots added to Link
    # without corresponding updates to _clone_link.
    for _s in Link.__slots__:
      assert getattr(new, _s, _CLONE_SENTINEL) is not _CLONE_SENTINEL, (
        f'_clone_link: slot {_s!r} not initialized in cloned Link'
      )
  return new
