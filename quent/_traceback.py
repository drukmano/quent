# SPDX-License-Identifier: MIT
"""Traceback enhancement — inject pipeline visualizations into exceptions.

Installs two global patches at import: ``sys.excepthook`` and
``traceback.TracebackException.__init__``. Both skipped when
``QUENT_NO_TRACEBACK=1`` is set before import.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
import types
import warnings
from typing import TYPE_CHECKING, Any

from ._exc_meta import (
  META_GATHER_FN,
  META_GATHER_INDEX,
  META_LINK_TEMP_ARGS,
  META_QUENT,
  META_SOURCE_LINK,
  _clean_quent_idx,
  _get_exc_meta,
  _pop_heavy_meta_keys,
)
from ._link import Link
from ._viz import (
  _get_link_name,
  _get_obj_name,
  _get_true_source_link,
  _make_indent,
  _sanitize_repr,
  _stringify_q,
  _VizContext,
)

if TYPE_CHECKING:
  from ._q import Q


def _cleanup_outermost_meta(meta: dict[str, Any]) -> None:
  """Defense-in-depth — heavy refs removed even if visualization failed."""
  _pop_heavy_meta_keys(meta)


_log = logging.getLogger('quent')

_quent_dir: str = os.path.dirname(os.path.realpath(__file__)) + os.sep


def _user_stacklevel() -> int:
  """Stacklevel for warnings.warn() that points at the first user frame."""
  frame = sys._getframe(1)
  level = 1
  while frame is not None:
    if not frame.f_code.co_filename.startswith(_quent_dir):
      return level
    frame = frame.f_back  # type: ignore[assignment]
    level += 1
  return level


# Pre-compiled template; co_name gets replaced with the pipeline viz string,
# making the viz appear as the "function name" in Python's traceback output.
_RAISE_CODE: types.CodeType = compile('raise __exc__', '<quent>', 'exec')

# 3.11+ formatters read co_qualname instead of co_name — set both.
_HAS_QUALNAME: bool = sys.version_info >= (3, 11)

_TracebackType: type[types.TracebackType] = types.TracebackType

_traceback_enabled: bool = os.environ.get('QUENT_NO_TRACEBACK', '').strip().lower() not in ('1', 'true', 'yes')
if not _traceback_enabled:
  _log.info('quent traceback enhancement disabled via QUENT_NO_TRACEBACK')


def _clean_internal_frames(tb: types.TracebackType | None) -> types.TracebackType | None:
  """Strip quent-internal frames, keep user and synthetic frames."""
  stack = []
  tb_next = None

  frame_tb = tb
  while frame_tb is not None:
    filename = frame_tb.tb_frame.f_code.co_filename
    # Keep <quent> synthetic frames and user frames.
    if filename == '<quent>' or not filename.startswith(_quent_dir):
      stack.append(frame_tb)
    frame_tb = frame_tb.tb_next

  for entry in reversed(stack):
    new_tb = _TracebackType(tb_next, entry.tb_frame, entry.tb_lasti, entry.tb_lineno)
    tb_next = new_tb

  return tb_next


_MAX_CHAINED_EXCEPTION_DEPTH: int = 1000


def _clean_chained_exceptions(exc: BaseException | None, seen: set[int]) -> None:
  """Iteratively clean frames from chained exceptions (incl. ExceptionGroup)."""
  stack = [exc]
  depth = 0
  while stack:
    if depth >= _MAX_CHAINED_EXCEPTION_DEPTH:
      break
    exc = stack.pop()
    if exc is None or id(exc) in seen:
      continue
    seen.add(id(exc))
    depth += 1
    if exc.__traceback__ is not None:
      exc.__traceback__ = _clean_internal_frames(exc.__traceback__)
    stack.append(exc.__cause__)
    stack.append(exc.__context__)
    if hasattr(exc, 'exceptions'):
      stack.extend(exc.exceptions)


def _inject_visualization(
  exc: BaseException,
  q: Q[Any],
  root_link: Link | None,
  source_link: Link | None,
  meta: dict[str, Any],
  extra_links: list[tuple[Link, str]] | None,
) -> None:
  """Build viz string, inject via the code-object hack. Falls back on failure."""
  globals_: dict[str, Any] | None = None
  try:
    ctx = _VizContext(
      source_link=_get_true_source_link(source_link, root_link),
      link_temp_args=meta.pop(META_LINK_TEMP_ARGS, None),
    )
    viz_source = _stringify_q(q, nest_lvl=0, root_link=root_link, ctx=ctx, extra_links=extra_links)
    # Indent so viz nests under the <quent> frame header.
    viz_source = _make_indent(1).join(['', *viz_source.splitlines()])

    # SECURITY INVARIANT: exec() arg MUST remain pre-compiled _RAISE_CODE.
    # User-controlled data (callable names, reprs) flows ONLY into
    # co_name/co_qualname metadata via code.replace() — never into executed
    # code or globals_. globals_ holds only the exception under a fixed key.
    filename = '<quent>'
    exc_value = exc
    globals_ = {'__name__': filename, '__file__': filename, '__exc__': exc_value}
    if _HAS_QUALNAME:
      code = _RAISE_CODE.replace(co_name=viz_source, co_qualname=viz_source)  # type: ignore[call-arg]  # co_qualname added in 3.11
    else:
      code = _RAISE_CODE.replace(co_name=viz_source)
    if code.co_code != _RAISE_CODE.co_code:
      raise RuntimeError('SECURITY: exec() must only use _RAISE_CODE')
    try:
      exec(code, globals_, {})  # nosec B102 — code object is pre-compiled from constant 'raise __exc__'
    except BaseException as caught_exc:
      # Must be BaseException — exc may be any subclass. But KI/SystemExit
      # arriving during exec() must never be absorbed.
      if caught_exc is not exc_value and isinstance(caught_exc, (KeyboardInterrupt, SystemExit)):
        raise
      new_tb = sys.exc_info()[1].__traceback__  # type: ignore[union-attr]
      exc.__traceback__ = _clean_internal_frames(new_tb)
    finally:
      # Break the cycle: exc → __traceback__ → frame → f_globals → globals_ → exc.
      if globals_ is not None:
        globals_.clear()
        globals_ = None
  except Exception as viz_exc:
    if globals_ is not None:
      globals_.clear()
      globals_ = None
    # Viz failures must never break exception handling — fall back to plain cleaning.
    _log.debug('pipeline visualization failed: %r', viz_exc)
    warnings.warn(
      f'quent: pipeline visualization failed: {viz_exc!r}',
      RuntimeWarning,
      stacklevel=_user_stacklevel(),
    )
    exc.__traceback__ = _clean_internal_frames(exc.__traceback__)


def _attach_exception_note(exc: BaseException, q: Q[Any], source_link: Link | None) -> None:
  """One-line exception note identifying the failing step (3.11+; survives reformatting)."""
  if not hasattr(exc, 'add_note'):
    return
  existing_notes = getattr(exc, '__notes__', [])
  if any(n.startswith('quent: exception at') for n in existing_notes):
    return
  try:
    if source_link is not None:
      obj_name = _get_obj_name(source_link.original_value if source_link.original_value is not None else source_link.v)
      step_name = f'.{_get_link_name(source_link)}({obj_name})'
    else:
      step_name = '?'
    root_name = _get_obj_name(q._root_link.v) if q._root_link is not None else ''
    q_label = f'Q[{_sanitize_repr(q._name)}]' if q._name is not None else 'Q'
    exc.add_note(f'quent: exception at {step_name} in {q_label}({root_name})')
  except Exception as note_exc:
    _log.debug('exception note attachment failed: %r', note_exc)  # never let notes break exception handling


def _modify_traceback(
  exc: BaseException,
  q: Q[Any] | None = None,
  link: Link | None = None,
  root_link: Link | None = None,
  extra_links: list[tuple[Link, str]] | None = None,
  is_nested: bool = False,
) -> BaseException:
  """Inject viz into traceback, or just strip internal frames.

  Returns the exception for use in ``raise`` expressions.

  is_nested=True: only frame cleaning, no viz (reserved for outermost).

  Not thread-safe; concurrent threads sharing __cause__/__context__ may race.
  Accepted limitation — traceback enhancement is best-effort and must never
  suppress the underlying exception.
  """
  if not _traceback_enabled:
    # Still clean heavy meta so Link/callables/values don't leak via
    # __quent_meta__ — only at outermost boundary, same as the enabled path.
    if q is not None and link is not None and not is_nested:
      _meta = getattr(exc, '__quent_meta__', None)
      if _meta is not None:
        _cleanup_outermost_meta(_meta)
      _clean_quent_idx(exc)
    return exc.with_traceback(exc.__traceback__)

  meta = _get_exc_meta(exc)
  if meta.get(META_SOURCE_LINK) is None:
    meta[META_SOURCE_LINK] = link

  if q is not None and link is not None and not is_nested:
    meta[META_QUENT] = True
    source_link = meta.pop(META_SOURCE_LINK, None)
    meta.pop(META_GATHER_INDEX, None)
    meta.pop(META_GATHER_FN, None)

    _inject_visualization(exc, q, root_link, source_link, meta, extra_links)
    _attach_exception_note(exc, q, source_link)
  else:
    meta[META_QUENT] = True
    exc.__traceback__ = _clean_internal_frames(exc.__traceback__)

  # Defense-in-depth — redundant with _inject_visualization's internal pops,
  # covers the fallback path if viz failed.
  if q is not None and link is not None and not is_nested:
    _cleanup_outermost_meta(meta)
  _clean_quent_idx(exc)

  seen: set[int] = set()
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)
  return exc.with_traceback(exc.__traceback__)


def _try_clean_quent_exc(exc_value: BaseException | None) -> tuple[bool, types.TracebackType | None]:
  """Check the __quent_meta__ flag; clean frames if present."""
  try:
    meta = getattr(exc_value, '__quent_meta__', None) if exc_value is not None else None
    if meta is not None and meta.get(META_QUENT, False):
      _clean_chained_exceptions(exc_value, set())
      return True, exc_value.__traceback__  # type: ignore[union-attr]  # exc_info()[1] is guaranteed non-None inside except block
  except Exception as e:
    _log.debug('_try_clean_quent_exc failed: %r', e)
  return False, None


def _quent_excepthook(
  exc_type: type[BaseException], exc_value: BaseException, exc_tb: types.TracebackType | None
) -> None:
  cleaned, tb = _try_clean_quent_exc(exc_value)
  if cleaned:
    exc_tb = tb
  _prev_excepthook(exc_type, exc_value, exc_tb)


# Capture originals before patching; _hooks_captured guard ensures the true
# originals are saved exactly once — importlib.reload() must not re-capture
# already-patched hooks.
if not globals().get('_hooks_captured', False):
  _prev_excepthook = sys.excepthook
  _original_te_init = traceback.TracebackException.__init__
  _hooks_captured = True


def _patched_te_init(
  self: traceback.TracebackException,
  exc_type: type[BaseException],
  exc_value: BaseException | None = None,
  exc_traceback: types.TracebackType | None = None,
  **kwargs: Any,
) -> None:
  cleaned, tb = _try_clean_quent_exc(exc_value)
  if cleaned:
    exc_traceback = tb
  _original_te_init(self, exc_type, exc_value, exc_traceback, **kwargs)  # type: ignore[arg-type]


# Module-level patches serialized by Python's import lock. importlib.reload()
# is not supported and may race under concurrent imports.
if _traceback_enabled:
  # Verify TE.__init__ signature — a future Python version changing param
  # order would make our patch silently misbehave.
  import inspect as _inspect

  _te_signature_ok = True
  try:
    _te_params = list(_inspect.signature(traceback.TracebackException.__init__).parameters.keys())
    if _te_params[:4] != ['self', 'exc_type', 'exc_value', 'exc_traceback']:
      _te_signature_ok = False
      warnings.warn(
        'quent: TracebackException.__init__ has an unexpected signature; '
        'skipping TracebackException patch to avoid incorrect argument forwarding.',
        RuntimeWarning,
        stacklevel=1,
      )
  except (ValueError, TypeError):
    pass  # inspect.signature can fail on builtins/C extensions
  del _inspect

  # Idempotency — prevents reload stacking patches (would cause infinite recursion:
  # new hook → old hook → _prev_excepthook (== old hook)).
  if sys.excepthook is not _quent_excepthook:
    sys.excepthook = _quent_excepthook
  if _te_signature_ok and traceback.TracebackException.__init__ is not _patched_te_init:
    traceback.TracebackException.__init__ = _patched_te_init  # type: ignore[method-assign]
  del _te_signature_ok
