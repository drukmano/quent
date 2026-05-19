# SPDX-License-Identifier: MIT
"""Exception metadata attached to exceptions for traceback display.

All per-exception state lives in a single ``__quent_meta__`` dict under the
constants below. Exception: ``_quent_idx`` is attached directly to exception
objects (not via the meta dict) by concurrent workers — hot path, consumed
only by triage in the same scope.

Keys:
  META_SOURCE_LINK    — Link | None; failing link. First-write-wins; popped by traceback.
  META_LINK_TEMP_ARGS — dict[id(link), dict]; runtime context for viz. Popped by traceback.
  META_GATHER_INDEX   — int; -1 for ExceptionGroup, >=0 for individual. First-write-wins.
  META_GATHER_FN      — Callable | None; parallel to gather_index.
  META_QUENT          — bool; set True after viz injection; never cleaned.
"""

from __future__ import annotations

from typing import Any

from ._link import Link

META_SOURCE_LINK = 'source_link'
META_LINK_TEMP_ARGS = 'link_temp_args'
META_GATHER_INDEX = 'gather_index'
META_GATHER_FN = 'gather_fn'
META_QUENT = 'quent'


def _get_exc_meta(exc: BaseException) -> dict[str, Any]:
  """Get or create the __quent_meta__ dict on an exception.

  Best-effort under free-threaded Python — must never break exception propagation.
  """
  try:
    return vars(exc).setdefault('__quent_meta__', {})  # type: ignore[no-any-return]  # setdefault returns Any
  except TypeError:  # pragma: no cover — defensive
    # Slotted exception without __dict__ — fall back.
    meta = getattr(exc, '__quent_meta__', None)
    if meta is None:
      meta = {}
      try:
        exc.__quent_meta__ = meta  # type: ignore[attr-defined]
      except (AttributeError, TypeError):
        pass
    return meta


def _set_link_temp_args(exc: BaseException, link: Link, /, **kwargs: Any) -> None:
  """Record live values (current_value, item, index) keyed by link identity for viz."""
  meta = _get_exc_meta(exc)
  link_temp_args = meta.get(META_LINK_TEMP_ARGS)
  if link_temp_args is None:
    link_temp_args = {}
    meta[META_LINK_TEMP_ARGS] = link_temp_args
  link_temp_args[id(link)] = kwargs


def _set_gather_meta(exc: BaseException, index: int, fn: Any = None) -> None:
  """Attach gather-specific metadata. First-write-wins preserves innermost failure."""
  meta = _get_exc_meta(exc)
  if META_GATHER_INDEX not in meta:
    meta[META_GATHER_INDEX] = index
    meta[META_GATHER_FN] = fn


def _clean_quent_idx(exc: BaseException) -> None:
  """Remove the ad-hoc _quent_idx attribute (set by concurrent workers for triage ordering).

  Lives directly on the exception (not in meta dict) for hot-path performance —
  only consumed by triage in the same concurrent scope. Cleaned here so it
  never leaks to user code.
  """
  try:
    del exc._quent_idx  # type: ignore[attr-defined]
  except AttributeError:
    pass


def _pop_heavy_meta_keys(meta: dict[str, Any]) -> None:
  """Remove heavy pipeline-internal refs. Preserves the lightweight 'quent' flag."""
  meta.pop(META_SOURCE_LINK, None)
  meta.pop(META_LINK_TEMP_ARGS, None)
  meta.pop(META_GATHER_INDEX, None)
  meta.pop(META_GATHER_FN, None)


def _clean_exc_meta(exc: BaseException) -> None:
  """Remove heavy refs after an exception is consumed (reraise=False)."""
  meta = getattr(exc, '__quent_meta__', None)
  if meta is None:
    return
  _pop_heavy_meta_keys(meta)
  _clean_quent_idx(exc)
