# SPDX-License-Identifier: MIT
"""Exception metadata helpers for attaching and cleaning quent runtime state on exceptions.

Metadata contract
-----------------
All per-exception quent state is stored in a single ``__quent_meta__`` dict
attached to the exception object.  The canonical keys are exposed as
module-level constants so every writer/reader uses the same name.

Keys (constant → dict key):

- ``META_SOURCE_LINK`` (``'source_link'``) — ``Link | None``.
  Writers: ``_engine._record_exception_source``, ``_traceback._modify_traceback``.
  Reader: ``_traceback``. First-write-wins; popped by ``_modify_traceback``;
  cleaned by ``_clean_exc_meta``.

- ``META_LINK_TEMP_ARGS`` (``'link_temp_args'``) — ``dict[int, dict]``.
  Writer: ``_set_link_temp_args``. Reader: ``_traceback``.
  Keyed by ``id(link)``; popped by ``_modify_traceback``;
  cleaned by ``_clean_exc_meta``.

- ``META_GATHER_INDEX`` (``'gather_index'``) — ``int``.
  Writer: ``_set_gather_meta``. Reader: ``_traceback``.
  -1 for ExceptionGroup wrappers; >=0 for individual failures;
  first-write-wins; cleaned by ``_clean_exc_meta``.

- ``META_GATHER_FN`` (``'gather_fn'``) — ``Callable | None``.
  Writer: ``_set_gather_meta``. Reader: ``_traceback``.
  Parallel to ``gather_index``; first-write-wins;
  cleaned by ``_clean_exc_meta``.

- ``META_QUENT`` (``'quent'``) — ``bool``.
  Writer: ``_traceback._modify_traceback``.
  Reader: ``_traceback`` (excepthook / TracebackException patch).
  Set to ``True`` after visualization injection; never cleaned.

Exception to consolidation contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``_quent_idx`` is an ``int`` attribute attached *directly* to exception objects
(not inside ``__quent_meta__``) by concurrent workers in ``_iter_ops`` and
``_gather_ops``.  It records the 0-based index of the item/function being
processed.  This bypasses ``__quent_meta__`` for performance in hot concurrent
paths and is consumed exclusively by the triage functions in the same scope.
Cleaned by ``_clean_exc_meta``; see ``_clean_exc_meta`` body for full rationale.
"""

from __future__ import annotations

from typing import Any

from ._link import Link

# ---- Metadata key constants ----

META_SOURCE_LINK = 'source_link'
META_LINK_TEMP_ARGS = 'link_temp_args'
META_GATHER_INDEX = 'gather_index'
META_GATHER_FN = 'gather_fn'
META_QUENT = 'quent'


def _get_exc_meta(exc: BaseException) -> dict[str, Any]:
  """Get or create the ``__quent_meta__`` dict on an exception.

  All per-exception quent metadata is consolidated into one dict rather
  than scattering multiple ``__quent_*`` attributes across the object.
  Uses ``setdefault`` on the instance dict for atomicity when possible.

  Thread safety: under free-threaded Python (PEP 703), concurrent access
  to the same exception's metadata is a potential race.  This is accepted
  as a known limitation — exception metadata is best-effort for traceback
  display and must never break exception propagation.
  """
  try:
    return vars(exc).setdefault('__quent_meta__', {})  # type: ignore[no-any-return]  # setdefault returns Any from dict.__setdefault__
  except TypeError:  # pragma: no cover — defensive: all BaseException subclasses have __dict__
    # Slotted exception without __dict__ — fall back to getattr.
    meta = getattr(exc, '__quent_meta__', None)
    if meta is None:
      meta = {}
      try:
        exc.__quent_meta__ = meta  # type: ignore[attr-defined]  # dynamically attaching metadata dict to exception
      except (AttributeError, TypeError):
        pass
    return meta


def _set_link_temp_args(exc: BaseException, link: Link, /, **kwargs: Any) -> None:
  """Attach runtime context to an exception for traceback display.

  Records the values that were live when the exception occurred (e.g.
  ``current_value=``, ``item=``, ``index=``), keyed by the link's
  identity.  The traceback formatter reads these to show actual arguments
  in the chain visualization.
  """
  meta = _get_exc_meta(exc)
  link_temp_args = meta.get(META_LINK_TEMP_ARGS)
  if link_temp_args is None:
    link_temp_args = {}
    meta[META_LINK_TEMP_ARGS] = link_temp_args
  link_temp_args[id(link)] = kwargs


def _set_gather_meta(exc: BaseException, index: int, fn: Any = None) -> None:
  """Attach gather-specific metadata to an exception for traceback display.

  First-write-wins: only records metadata if ``gather_index`` is not
  already present, preserving the innermost failure context.
  """
  meta = _get_exc_meta(exc)
  if META_GATHER_INDEX not in meta:
    meta[META_GATHER_INDEX] = index
    meta[META_GATHER_FN] = fn


def _clean_quent_idx(exc: BaseException) -> None:
  """Remove the ad-hoc ``_quent_idx`` attribute from an exception if present.

  ``_quent_idx`` is an integer attached *directly* to exception objects
  (not inside ``__quent_meta__``) by concurrent worker tasks in
  ``_iter_ops._ConcurrentIterOp`` and ``_gather_ops._ConcurrentGatherOp``.
  Each worker records the 0-based index of the item/function it was
  processing when the exception occurred.  The triage functions
  (``_triage_iter_exceptions``, ``_triage_gather_exceptions``) use this
  value as a sort key so the chain reports the exception from the
  *earliest* failing item/function by original input order, regardless of
  wall-clock completion order.

  Why directly on the exception rather than in ``__quent_meta__``?
  It is set inside a hot concurrent worker coroutine/thread immediately
  after the exception is caught.  Bypassing ``_get_exc_meta`` avoids its
  dict-creation overhead and is safe because ``_quent_idx`` is consumed
  exclusively by the triage function in the same concurrent scope — it is
  never needed by the traceback formatter.  This helper ensures the
  attribute does not leak onto exceptions that propagate to user code.
  """
  try:
    del exc._quent_idx  # type: ignore[attr-defined]  # dynamically attached index for triage ordering
  except AttributeError:
    pass


def _pop_heavy_meta_keys(meta: dict[str, Any]) -> None:
  """Remove heavy chain-internal reference keys from a metadata dict.

  Shared by ``_clean_exc_meta`` (after-except cleanup) and
  ``_traceback._cleanup_outermost_meta`` (defense-in-depth at the
  outermost chain boundary).  The lightweight ``quent`` flag is
  intentionally preserved.
  """
  meta.pop(META_SOURCE_LINK, None)
  meta.pop(META_LINK_TEMP_ARGS, None)
  meta.pop(META_GATHER_INDEX, None)
  meta.pop(META_GATHER_FN, None)


def _clean_exc_meta(exc: BaseException) -> None:
  """Remove heavy chain-internal references from exception metadata.

  Called after an exception has been consumed by an except handler
  (``reraise=False``) to prevent holding chain internals alive via the
  exception object.  The lightweight ``quent`` flag is preserved.
  """
  meta = getattr(exc, '__quent_meta__', None)
  if meta is None:
    return
  _pop_heavy_meta_keys(meta)
  _clean_quent_idx(exc)
