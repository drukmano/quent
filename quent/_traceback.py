"""Traceback rewriting: injects readable chain visualizations into exception tracebacks."""

from __future__ import annotations

import os
import sys
import traceback
import types
from typing import TYPE_CHECKING, Any

from ._core import Link, Null

if TYPE_CHECKING:
  from ._chain import Chain


_quent_file: str = os.path.dirname(os.path.realpath(__file__)) + os.sep
# Pre-compiled code object for the traceback injection hack (see _modify_traceback).
# The code simply raises an exception. Its co_name/co_qualname will be replaced with
# the chain visualization string, so when Python builds the traceback frame, the
# "function name" shown IS the chain visualization.
_RAISE_CODE: types.CodeType = compile('raise __exc__', '<quent>', 'exec')
# Python 3.11+ supports code.replace(co_qualname=...), needed for the traceback hack.
_HAS_QUALNAME: bool = sys.version_info >= (3, 11)
# Local alias used as a constructor in _clean_internal_frames. TracebackType()
# can construct traceback objects since Python 3.7 (PEP 579).
_TracebackType: type[types.TracebackType] = types.TracebackType


class _Ctx:
  """Shared context threaded through chain stringification."""

  __slots__ = ('found', 'link_temp_args', 'source_link')

  def __init__(self, source_link: Link | None, link_temp_args: dict[int, dict[str, Any]] | None) -> None:
    self.source_link = source_link
    self.link_temp_args = link_temp_args
    self.found = False


def _clean_internal_frames(tb: types.TracebackType | None) -> types.TracebackType | None:
  """Remove quent-internal frames from a traceback, keeping only user frames
  and the synthetic <quent> frame.
  """
  stack = []
  tb_next = None

  while tb is not None:
    filename = tb.tb_frame.f_code.co_filename
    # Keep <quent> synthetic frames and frames outside the quent package.
    if filename == '<quent>' or not filename.startswith(_quent_file):
      stack.append(tb)
    tb = tb.tb_next

  for tb in reversed(stack):
    new_tb = _TracebackType(tb_next, tb.tb_frame, tb.tb_lasti, tb.tb_lineno)
    tb_next = new_tb

  return tb_next


def _clean_chained_exceptions(exc: BaseException | None, seen: set[int]) -> None:
  """Iteratively clean internal frames from chained exceptions."""
  stack = [exc]
  while stack:
    exc = stack.pop()
    if exc is None or id(exc) in seen:
      continue
    seen.add(id(exc))
    if exc.__traceback__ is not None:
      exc.__traceback__ = _clean_internal_frames(exc.__traceback__)
    stack.append(exc.__cause__)
    stack.append(exc.__context__)
    # Python 3.11+: ExceptionGroup wraps sub-exceptions
    if hasattr(exc, 'exceptions'):
      stack.extend(exc.exceptions)


def _modify_traceback(
  exc: BaseException,
  chain: Chain | None = None,
  link: Link | None = None,
  root_link: Link | None = None,
  extra_links: list[tuple[Link, str]] | None = None,
) -> BaseException:
  """Inject chain visualization or just strip internal frames.
  Always returns the exception for use in `raise` expressions.
  """
  if getattr(exc, '__quent_source_link__', None) is None:
    exc.__quent_source_link__ = link  # type: ignore[attr-defined]

  if chain is not None and link is not None and not chain.is_nested:
    exc.__quent__ = True  # type: ignore[attr-defined]
    source_link = exc.__quent_source_link__  # type: ignore[attr-defined]
    del exc.__quent_source_link__  # type: ignore[attr-defined]

    try:
      ctx = _Ctx(
        source_link=_get_true_source_link(source_link, root_link),
        link_temp_args=getattr(exc, '__quent_link_temp_args__', None),
      )
      if hasattr(exc, '__quent_link_temp_args__'):
        del exc.__quent_link_temp_args__
      chain_source = _stringify_chain(chain, nest_lvl=0, root_link=root_link, ctx=ctx, extra_links=extra_links)
      # Indent the chain visualization so it appears nested under the <quent> frame header.
      chain_source = _make_indent(1).join(['', *chain_source.splitlines()])

      # HACK: Inject chain visualization into the traceback by exec'ing a `raise` statement
      # with a code object whose co_name has been replaced with the chain visualization string.
      # Python's traceback machinery reads co_name as the "function name", so the chain
      # structure appears as if it were a function name in the traceback.
      # The exec creates a real traceback frame that we then graft onto the exception.
      filename = '<quent>'
      exc_value = exc
      globals_ = {'__name__': filename, '__file__': filename, '__exc__': exc_value}
      if _HAS_QUALNAME:
        code = _RAISE_CODE.replace(co_name=chain_source, co_qualname=chain_source)  # type: ignore[call-arg]
      else:
        code = _RAISE_CODE.replace(co_name=chain_source)
      try:
        exec(code, globals_, {})
      except BaseException:
        new_tb = sys.exc_info()[1].__traceback__  # type: ignore[union-attr]
        exc.__traceback__ = _clean_internal_frames(new_tb)
    except Exception:
      # Visualization failed — fall back to just cleaning frames.
      exc.__traceback__ = _clean_internal_frames(exc.__traceback__)
  else:
    exc.__quent__ = True  # type: ignore[attr-defined]
    exc.__traceback__ = _clean_internal_frames(exc.__traceback__)

  seen: set[int] = set()
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)
  return exc.with_traceback(exc.__traceback__)


def _get_true_source_link(source_link: Link | None, root_link: Link | None) -> Link | None:
  """Retrieve the first non-chain link in a chain of nested chains.

  Drills through nested Chain root links to find the actual user-provided
  callable or value that caused the exception.
  """
  seen = set()
  while source_link is not None and id(source_link) not in seen:
    seen.add(id(source_link))
    if source_link.is_chain:
      chain = source_link.v
    elif getattr(source_link.original_value, '_is_chain', False):
      chain = source_link.original_value
    else:
      break
    if chain.root_link is not None:
      source_link = chain.root_link
    else:
      break
  if source_link is None:
    source_link = root_link
  return source_link


def _make_indent(nest_lvl: int) -> str:
  """Create a newline followed by indentation for the given nesting level."""
  return '\n' + ' ' * 4 * nest_lvl


def _get_link_name(link: Link) -> str:
  """Reconstruct a display name for a link from its type and flags (cold path only)."""
  v = link.v
  op = getattr(v, '_quent_op', None)
  if op == 'with':
    return 'with_do' if v._ignore_result else 'with_'
  if op == 'map':
    return 'foreach' if v._ignore_result else 'map'
  if op == 'filter':
    return 'filter'
  if op == 'gather':
    return 'gather'
  if op == 'if':
    return 'if_'
  if link.ignore_result:
    return 'do'
  return 'then'


def _get_obj_name(obj: Any) -> str:
  """Get a human-readable name for an object, for traceback display."""
  if getattr(obj, '_is_chain', False):
    return type(obj).__name__
  try:
    name = getattr(obj, '__name__', None)
    if name is None:
      name = getattr(obj, '__qualname__', None)
    if name is not None:
      return str(name)
  except Exception:
    pass
  if hasattr(obj, 'func'):  # functools.partial
    return f'partial({_get_obj_name(obj.func)})'
  try:
    return repr(obj).replace('\n', '\\n').replace('\r', '\\r')
  except Exception:
    return type(obj).__name__


def _format_call_args(args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> str:
  """Format positional and keyword arguments for chain visualization."""
  parts = []
  if args:
    if args[0] is ...:
      parts.append('...')
    else:
      parts.extend(_get_obj_name(a) for a in args)
  if kwargs:
    parts.extend(f'{k}={_get_obj_name(v)}' for k, v in kwargs.items())
  return ', '.join(parts)


def _resolve_nested_chain(
  link: Link, args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None, nest_lvl: int, ctx: _Ctx, max_depth: int = 50
) -> str:
  """Resolve a nested chain link into its string representation."""
  original_value = link.original_value if link.original_value is not None else link.v
  nested_root_link = None
  _temp_args = args or ()
  _temp_kwargs = kwargs or {}
  if _temp_args or _temp_kwargs:
    _temp_v = _temp_args[0] if _temp_args else Null
    if _temp_v is not Null or _temp_kwargs:
      nested_root_link = Link(
        _temp_v,
        args=_temp_args[1:] if len(_temp_args) > 1 else None,
        kwargs=_temp_kwargs or None,
      )
  nested_ctx = _Ctx(source_link=ctx.source_link, link_temp_args=None)
  nested_ctx.found = ctx.found
  result = _stringify_chain(
    original_value,
    nest_lvl=nest_lvl + 1,
    root_link=nested_root_link,
    ctx=nested_ctx,
    max_depth=max_depth,
  )
  ctx.found = nested_ctx.found
  return result


def _stringify_chain(
  chain: Chain,
  nest_lvl: int = 0,
  root_link: Link | None = None,
  *,
  ctx: _Ctx,
  extra_links: list[tuple[Link, str]] | None = None,
  max_depth: int = 50,
) -> str:
  """Build a string visualization of a chain for traceback display.

  Returns the output string.
  """
  if nest_lvl >= max_depth:
    return f'{_make_indent(nest_lvl)}Chain(...<truncated at depth {max_depth}>...)'
  output = ''
  root = chain.root_link
  if root is None and root_link is not None:
    root = root_link

  if nest_lvl > 0:
    output += _make_indent(nest_lvl)
  output += _get_obj_name(chain)

  if root is None:
    output += '()'
  else:
    output += _format_link(root, nest_lvl=nest_lvl, ctx=ctx, max_depth=max_depth)
    if not ctx.found and root is ctx.source_link:
      ctx.found = True

  links: list[tuple[Link, str]] = []
  link = chain.first_link
  while link is not None:
    links.append((link, _get_link_name(link)))
    # If this is an 'if' operation, check for an else branch.
    op = getattr(link.v, '_quent_op', None)
    if op == 'if' and getattr(link.v, '_else_link', None) is not None:
      links.append((link.v._else_link, 'else_'))
    link = link.next_link
  if chain.on_except_link is not None:
    links.append((chain.on_except_link, 'except_'))
  if chain.on_finally_link is not None:
    links.append((chain.on_finally_link, 'finally_'))

  for link, method_name in links:
    output += _make_indent(nest_lvl)
    output += _format_link(link, nest_lvl=nest_lvl, ctx=ctx, method_name=method_name, max_depth=max_depth)
    if not ctx.found and link is ctx.source_link:
      ctx.found = True

  if extra_links:
    for link, method_name in extra_links:
      output += _make_indent(nest_lvl)
      output += _format_link(link, nest_lvl=nest_lvl, ctx=ctx, method_name=method_name, max_depth=max_depth)
      if not ctx.found and link is ctx.source_link:
        ctx.found = True

  return output


def _format_link(link: Link, nest_lvl: int, ctx: _Ctx, method_name: str | None = None, max_depth: int = 50) -> str:
  """Format a single link for chain visualization.

  Handles nested chains, argument display, source link marking, and
  drills into original_value if it is a Link (for wrapped operations
  like _Foreach, _Filter).
  """
  outer_link = link
  if isinstance(link.original_value, Link):
    link = link.original_value

  original_value = link.original_value
  args = link.args
  kwargs = link.kwargs
  link_v = ''
  output = ''
  is_chain = False

  if not ctx.found and ctx.link_temp_args is not None and id(link) in ctx.link_temp_args:
    temp_kwargs = ctx.link_temp_args[id(link)]
    if temp_kwargs:
      kwargs = {**(kwargs or {}), **temp_kwargs}

  if original_value is None:
    original_value = link.v
  if link.is_chain or getattr(original_value, '_is_chain', False):
    link_v = _resolve_nested_chain(link, args, kwargs, nest_lvl, ctx, max_depth=max_depth)
    args = kwargs = None
    is_chain = True
  else:
    op = getattr(outer_link.v, '_quent_op', None)
    if op == 'gather' and hasattr(outer_link.v, '_fns'):
      link_v = ', '.join(_get_obj_name(f) for f in outer_link.v._fns)
    else:
      link_v = _get_obj_name(original_value)

  if method_name is not None:
    output += f'.{method_name}'

  # Determine if this link represents a callable invocation.
  if callable(link.v) or link.args or link.kwargs or link.is_chain:
    if is_chain:
      call_args = _format_call_args(args, kwargs)
      chain_newline = _make_indent(nest_lvl + 1) if call_args else ''
      call_prefix = f', {call_args}' if call_args else ''
      output += f'({link_v}{chain_newline}{call_prefix}{_make_indent(nest_lvl)})'
    else:
      call_args = _format_call_args(args, kwargs)
      call_prefix = f', {call_args}' if call_args else ''
      output += f'({link_v}{call_prefix})'
  else:
    output += f'({link_v})'

  if not ctx.found and outer_link is ctx.source_link:
    output += ' <' + '-' * 4

  return output


# Override the default exception display to clean quent-internal frames.
# Without this, uncaught exceptions would show quent's internal call stack.
_original_excepthook: Any = sys.excepthook


def _quent_excepthook(
  exc_type: type[BaseException], exc_value: BaseException, exc_tb: types.TracebackType | None
) -> None:
  """Custom excepthook that cleans quent internal frames before display."""
  try:
    if getattr(exc_value, '__quent__', False):
      _clean_chained_exceptions(exc_value, set())
      exc_tb = exc_value.__traceback__
  except Exception:
    pass
  _original_excepthook(exc_type, exc_value, exc_tb)


# Also patch TracebackException (used by logging, traceback.format_exception, etc.)
# so that quent-internal frames are cleaned in ALL exception rendering paths,
# not just the default sys.excepthook.
_original_te_init = traceback.TracebackException.__init__


def _patched_te_init(
  self: traceback.TracebackException,
  exc_type: type[BaseException],
  exc_value: BaseException | None = None,
  exc_traceback: types.TracebackType | None = None,
  **kwargs: Any,
) -> None:
  """Patched TracebackException.__init__ that cleans quent frames."""
  try:
    if exc_value is not None and getattr(exc_value, '__quent__', False):
      _clean_chained_exceptions(exc_value, set())
      exc_traceback = exc_value.__traceback__
  except Exception:
    pass
  _original_te_init(self, exc_type, exc_value, exc_traceback, **kwargs)  # type: ignore[arg-type]


_patching_enabled = False


def disable_traceback_patching() -> None:
  """Restore the original sys.excepthook and TracebackException.__init__."""
  global _patching_enabled
  if not _patching_enabled:
    return
  _patching_enabled = False
  sys.excepthook = _original_excepthook
  traceback.TracebackException.__init__ = _original_te_init  # type: ignore[method-assign]


def enable_traceback_patching() -> None:
  """Install quent's custom excepthook and TracebackException patch."""
  global _patching_enabled
  if _patching_enabled:
    return
  _patching_enabled = True
  sys.excepthook = _quent_excepthook
  traceback.TracebackException.__init__ = _patched_te_init  # type: ignore[method-assign]


enable_traceback_patching()
