# SPDX-License-Identifier: MIT
"""Traceback enhancement — inject chain visualizations into exception output.

Importing this module installs two global patches:
  - ``sys.excepthook`` is replaced with ``_quent_excepthook`` to clean
    quent-internal frames before the default exception display.
  - ``traceback.TracebackException.__init__`` is patched so that all
    rendering paths (logging, ``traceback.format_exception``, etc.) also
    receive cleaned tracebacks.

Both patches are skipped when the environment variable
``QUENT_NO_TRACEBACK=1`` is set before the module is imported.
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
)
from ._link import Link
from ._viz import (
  _get_link_name,
  _get_obj_name,
  _get_true_source_link,
  _make_indent,
  _sanitize_repr,
  _stringify_chain,
  _VizContext,
)

if TYPE_CHECKING:
  from ._chain import Chain


# ---- Module-level constants ----

_log = logging.getLogger('quent')

_quent_dir: str = os.path.dirname(os.path.realpath(__file__)) + os.sep


def _user_stacklevel() -> int:
  """Compute the stacklevel needed for warnings.warn() to point at the first user frame outside quent/.

  Walks the call stack upward from the caller of this function until it finds a frame whose
  filename is not inside the quent package directory.  Returns the level count suitable
  for passing directly to ``warnings.warn(..., stacklevel=_user_stacklevel())``.
  """
  frame = sys._getframe(1)  # caller of _user_stacklevel
  level = 1
  while frame is not None:
    if not frame.f_code.co_filename.startswith(_quent_dir):
      return level
    frame = frame.f_back  # type: ignore[assignment]
    level += 1
  return level


# Pre-compiled code object used as a template for the traceback injection hack.
# Its co_name is replaced with the chain visualization string at runtime, so
# the visualization appears as the "function name" in Python's traceback output.
_RAISE_CODE: types.CodeType = compile('raise __exc__', '<quent>', 'exec')

# Python 3.11+ supports code.replace(co_qualname=...), which newer traceback
# formatters read instead of co_name.  We need to set both for full coverage.
_HAS_QUALNAME: bool = sys.version_info >= (3, 11)

# PEP 579: TracebackType() can construct traceback objects directly.
_TracebackType: type[types.TracebackType] = types.TracebackType

# QUENT_NO_TRACEBACK=1 disables all traceback modifications (visualization
# injection, frame cleaning, hook patching).
_traceback_enabled: bool = os.environ.get('QUENT_NO_TRACEBACK', '').strip().lower() not in ('1', 'true', 'yes')
if not _traceback_enabled:
  _log.info('quent traceback enhancement disabled via QUENT_NO_TRACEBACK')


# ---- Frame cleaning ----


def _clean_internal_frames(tb: types.TracebackType | None) -> types.TracebackType | None:
  """Remove quent-internal frames from a traceback, keeping user and synthetic frames."""
  stack = []
  tb_next = None

  frame_tb = tb
  while frame_tb is not None:
    filename = frame_tb.tb_frame.f_code.co_filename
    # Keep <quent> synthetic frames and frames outside the quent package.
    if filename == '<quent>' or not filename.startswith(_quent_dir):
      stack.append(frame_tb)
    frame_tb = frame_tb.tb_next

  for entry in reversed(stack):
    new_tb = _TracebackType(tb_next, entry.tb_frame, entry.tb_lasti, entry.tb_lineno)
    tb_next = new_tb

  return tb_next


_MAX_CHAINED_EXCEPTION_DEPTH: int = 1000


def _clean_chained_exceptions(exc: BaseException | None, seen: set[int]) -> None:
  """Iteratively clean internal frames from chained exceptions."""
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
    # Python 3.11+: ExceptionGroup wraps sub-exceptions.
    if hasattr(exc, 'exceptions'):
      stack.extend(exc.exceptions)


# ---- Traceback injection ----


def _inject_visualization(
  exc: BaseException,
  chain: Chain[Any],
  root_link: Link | None,
  source_link: Link | None,
  meta: dict[str, Any],
  extra_links: list[tuple[Link, str]] | None,
) -> None:
  """Build chain visualization string and inject it into the traceback via code object hack.

  Creates a synthetic <quent> frame whose co_name is the chain visualization,
  so it appears as the "function name" in Python's traceback output.
  Falls back to plain frame cleaning on visualization failure.
  """
  globals_: dict[str, Any] | None = None
  try:
    ctx = _VizContext(
      source_link=_get_true_source_link(source_link, root_link),
      link_temp_args=meta.pop(META_LINK_TEMP_ARGS, None),
    )
    chain_source = _stringify_chain(chain, nest_lvl=0, root_link=root_link, ctx=ctx, extra_links=extra_links)
    # Indent the chain visualization so it appears nested under the <quent> frame header.
    chain_source = _make_indent(1).join(['', *chain_source.splitlines()])

    # HACK — code object injection trick
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We exec() a ``raise`` statement using a code object whose co_name has been
    # replaced with the chain visualization string.  Python's traceback machinery
    # reads co_name as the "function name", so the chain structure appears inline
    # in the traceback as if it were a function name.  The exec() creates a real
    # traceback frame that we then graft onto the exception.
    # SECURITY INVARIANT: The code argument to exec() MUST remain the pre-compiled
    # constant ``_RAISE_CODE`` (``compile('raise __exc__', ...)``, line 49).
    # User-controlled data (callable names, repr output) MUST only flow into
    # ``co_name``/``co_qualname`` metadata via ``code.replace()``, never into
    # executed code or the ``globals_`` dict.  The ``globals_`` dict contains
    # only the exception object under a fixed key.
    filename = '<quent>'
    exc_value = exc
    globals_ = {'__name__': filename, '__file__': filename, '__exc__': exc_value}
    if _HAS_QUALNAME:
      # Python 3.11+: newer traceback formatters prefer co_qualname over co_name,
      # so we must set both to ensure the visualization renders everywhere.
      code = _RAISE_CODE.replace(co_name=chain_source, co_qualname=chain_source)  # type: ignore[call-arg]  # co_qualname added in Python 3.11; mypy doesn't know about it
    else:
      code = _RAISE_CODE.replace(co_name=chain_source)
    if code.co_code != _RAISE_CODE.co_code:
      raise RuntimeError('SECURITY: exec() must only use _RAISE_CODE')
    try:
      exec(code, globals_, {})  # nosec B102 — code object is pre-compiled from constant 'raise __exc__'; user input only reaches co_name metadata, never executed code
    except BaseException:
      # Must be BaseException (not Exception) to catch the deliberately re-raised
      # exc, which may be any BaseException subclass.  Theoretical race: a
      # KeyboardInterrupt arriving during exec() would be caught here, but the
      # window is negligibly small (~nanoseconds).
      new_tb = sys.exc_info()[1].__traceback__  # type: ignore[union-attr]  # exc_info()[1] is guaranteed non-None inside except block
      exc.__traceback__ = _clean_internal_frames(new_tb)
    finally:
      # Clear the globals dict to break the reference cycle:
      # exc -> __traceback__ -> frame -> f_globals -> globals_ -> exc
      if globals_ is not None:
        globals_.clear()
        globals_ = None  # Release the dict object itself
  except Exception as viz_exc:
    if globals_ is not None:
      globals_.clear()
      globals_ = None
    # Graceful degradation: visualization failures must never break exception
    # handling.  Emit a warning and fall back to plain frame cleaning.
    _log.debug('chain visualization failed: %r', viz_exc)
    warnings.warn(
      f'quent: chain visualization failed: {viz_exc!r}',
      RuntimeWarning,
      stacklevel=_user_stacklevel(),
    )
    exc.__traceback__ = _clean_internal_frames(exc.__traceback__)


def _attach_exception_note(exc: BaseException, chain: Chain[Any], source_link: Link | None) -> None:
  """Attach a concise one-line note identifying the failing step (Python 3.11+).

  Exception notes survive traceback reformatting/stripping.
  """
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
    root_name = _get_obj_name(chain.root_link.v) if chain.root_link is not None else ''
    chain_label = f'Chain[{_sanitize_repr(chain._name)}]' if chain._name is not None else 'Chain'
    exc.add_note(f'quent: exception at {step_name} in {chain_label}({root_name})')
  except Exception as note_exc:
    _log.debug('exception note attachment failed: %r', note_exc)  # never let note generation break exception handling


def _modify_traceback(
  exc: BaseException,
  chain: Chain[Any] | None = None,
  link: Link | None = None,
  root_link: Link | None = None,
  extra_links: list[tuple[Link, str]] | None = None,
  is_nested: bool = False,
) -> BaseException:
  """Inject chain visualization into the traceback, or just strip internal frames.

  Always returns the exception for use in ``raise`` expressions.

  *is_nested* indicates whether the chain is executing as a nested step
  inside another chain.  When True, only frame cleaning is performed
  (no visualization injection) — visualization is reserved for the
  outermost chain.

  Not thread-safe.  If concurrent threads process exceptions that share
  ``__cause__`` or ``__context__`` objects, traceback mutations may race.
  This is accepted as a known limitation — traceback enhancement is
  best-effort and must never suppress the underlying exception.
  """
  if not _traceback_enabled:
    # Clean heavy metadata even when traceback enhancement is disabled,
    # to prevent Link objects, callables, and runtime values from leaking
    # via __quent_meta__ to user code.  Only at the outermost chain boundary
    # (same condition as the enabled path) and only if metadata is present.
    if chain is not None and link is not None and not is_nested:
      _meta = getattr(exc, '__quent_meta__', None)
      if _meta is not None:
        _meta.pop(META_SOURCE_LINK, None)
        _meta.pop(META_LINK_TEMP_ARGS, None)
        _meta.pop(META_GATHER_INDEX, None)
        _meta.pop(META_GATHER_FN, None)
      _clean_quent_idx(exc)
    # Return unmodified exception for consistent caller interface.
    return exc.with_traceback(exc.__traceback__)

  meta = _get_exc_meta(exc)
  if meta.get(META_SOURCE_LINK) is None:
    meta[META_SOURCE_LINK] = link

  if chain is not None and link is not None and not is_nested:
    meta[META_QUENT] = True
    source_link = meta.pop(META_SOURCE_LINK, None)
    meta.pop(META_GATHER_INDEX, None)
    meta.pop(META_GATHER_FN, None)

    _inject_visualization(exc, chain, root_link, source_link, meta, extra_links)
    _attach_exception_note(exc, chain, source_link)
  else:
    meta[META_QUENT] = True
    exc.__traceback__ = _clean_internal_frames(exc.__traceback__)

  # Defense-in-depth cleanup for the outermost chain only: remove heavy
  # chain-internal references from __quent_meta__ for exceptions that
  # propagate without being consumed by except_().  The primary cleanup
  # point is _clean_exc_meta (called when except_() consumes an exception).
  # This secondary cleanup prevents internal Link objects, callables, and
  # their arguments from leaking to user code — especially relevant in
  # concurrent operations where link_temp_args may contain user data
  # (item=, current_value=).
  # Only run at the outermost chain: nested chains need source_link and
  # link_temp_args intact for the outer chain's visualization pass.
  if chain is not None and link is not None and not is_nested:
    # These pops are intentionally redundant with the pops at lines 245-247 and
    # _inject_visualization's internal pop.  They act as defense-in-depth: if
    # _inject_visualization fails and falls back to plain frame cleaning, these
    # ensure META_SOURCE_LINK and META_LINK_TEMP_ARGS are still removed from
    # the exception metadata before it reaches user code.
    meta.pop(META_SOURCE_LINK, None)
    meta.pop(META_LINK_TEMP_ARGS, None)
  # Clean _quent_idx: ad-hoc attribute attached by concurrent workers
  # (see _exc_meta._clean_quent_idx docstring).
  _clean_quent_idx(exc)

  seen: set[int] = set()
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)
  return exc.with_traceback(exc.__traceback__)


# ---- Hook installation ----


def _try_clean_quent_exc(exc_value: BaseException | None) -> tuple[bool, types.TracebackType | None]:
  """Check for ``__quent_meta__`` flag and clean frames if present."""
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
  """Custom ``sys.excepthook`` that cleans quent-internal frames before display."""
  cleaned, tb = _try_clean_quent_exc(exc_value)
  if cleaned:
    exc_tb = tb
  _prev_excepthook(exc_type, exc_value, exc_tb)


# Unconditionally capture the current hooks *before* any patching, so
# ``_traceback_enabled`` toggle always has valid restore targets — even if
# mutated by external code.
# Guard with _hooks_captured to ensure the true originals are captured
# exactly once — importlib.reload() must not re-capture already-patched hooks.
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
  """Patched ``TracebackException.__init__`` that cleans quent frames."""
  cleaned, tb = _try_clean_quent_exc(exc_value)
  if cleaned:
    exc_traceback = tb
  _original_te_init(self, exc_type, exc_value, exc_traceback, **kwargs)  # type: ignore[arg-type]  # signature matches CPython's TracebackException.__init__


# ---- Hook installation (runs once at import time) ----

# These module-level patches are serialized by Python's import lock on first
# import.  importlib.reload(quent) is not supported and may produce race
# conditions if called concurrently from multiple threads.
if _traceback_enabled:
  # Verify TracebackException.__init__ has the expected positional signature.
  # If a future Python version changes the parameter order, our patch would
  # silently misbehave.  Detect this early and disable patching.
  import inspect as _inspect

  try:
    # Check the current TE.__init__ (not _original_te_init) to detect
    # if a future Python version changed the parameter order.
    _te_params = list(_inspect.signature(traceback.TracebackException.__init__).parameters.keys())
    if _te_params[:4] != ['self', 'exc_type', 'exc_value', 'exc_traceback']:
      # stacklevel=1: module-level code during import; no user frame above.
      warnings.warn(
        'quent: TracebackException.__init__ has an unexpected signature; '
        'traceback enhancements may not work correctly.',
        RuntimeWarning,
        stacklevel=1,
      )
  except (ValueError, TypeError):
    pass  # inspect.signature can fail on builtins/C extensions
  del _inspect

  # Idempotency guard: prevent stacking patches on importlib.reload(quent).
  # Without this, reload would save the already-patched hook as _prev_excepthook,
  # creating an infinite recursion: new hook -> old hook -> _prev_excepthook (== old hook).
  # Note: _prev_excepthook and _original_te_init are NOT re-captured here —
  # the true originals were captured once by the _hooks_captured guard above.
  if sys.excepthook is not _quent_excepthook:
    sys.excepthook = _quent_excepthook
  if traceback.TracebackException.__init__ is not _patched_te_init:
    traceback.TracebackException.__init__ = _patched_te_init  # type: ignore[method-assign]  # monkeypatching __init__ for traceback enhancement
