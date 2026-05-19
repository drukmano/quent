# SPDX-License-Identifier: MIT
"""Pipeline visualization for traceback display and ``__repr__``."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

from ._link import Link
from ._types import Null

if TYPE_CHECKING:
  from ._q import Q
  from ._types import _PipelineOp


_ERROR_MARKER = ' <----'
_MAX_REPR_LEN = 200
_MAX_CALL_ARGS_LEN = 500
_VIZ_MAX_LINKS_PER_LEVEL = 100
_VIZ_MAX_LENGTH = 10_000
_VIZ_MAX_NESTING_DEPTH = 50
_VIZ_MAX_TOTAL_CALLS = 500
_VIZ_INDENT_WIDTH = 4

# Regex for stripping ANSI escape sequences (CSI, OSC, and simple ESC sequences).
_ANSI_ESCAPE_RE = re.compile(
  r'\x1b\[[0-9;?!>]*[A-Za-z]'  # CSI sequences
  r'|\x1b][^\x07\x1b]*\x07'  # OSC terminated by BEL
  r'|\x1b][^\x07\x1b]*\x1b\\'  # OSC terminated by ST (ESC + backslash)
  r'|\x1bP[^\x1b]*\x1b\\'  # DCS (Device Control String) terminated by ST
  r'|\x1b[^[\]()]'  # Simple ESC sequences
)

# Unicode control characters to strip (C0/C1 controls except tab, newline, carriage return).
_CONTROL_CHAR_RE = re.compile(
  r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufff9-\ufffb'
  r'\U000e0001-\U000e007f\U000e0100-\U000e01ef]'
)

# QUENT_TRACEBACK_VALUES=0 suppresses argument values in pipeline visualizations
# while keeping step names and pipeline structure intact. Useful in production to
# prevent sensitive pipeline values from leaking into tracebacks and logs.
_show_traceback_values: bool = os.environ.get('QUENT_TRACEBACK_VALUES', '').strip().lower() not in ('0', 'false', 'no')


def _sanitize_repr(s: str) -> str:
  """Strip ANSI escapes + Unicode control chars (CWE-117 defense)."""
  s = s.replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
  s = _ANSI_ESCAPE_RE.sub('', s)
  return _CONTROL_CHAR_RE.sub('', s)


# Cold path only (tracebacks, __repr__). Clarity > performance throughout.


class _VizContext:
  """Shared context threaded through pipeline stringification."""

  __slots__ = ('found', 'link_temp_args', 'source_link', 'total_calls')

  def __init__(self, source_link: Link | None, link_temp_args: dict[int, dict[str, Any]] | None) -> None:
    self.source_link = source_link
    self.link_temp_args = link_temp_args
    self.found = False
    self.total_calls = 0

  def mark_found(self, link: Link) -> None:
    if not self.found and link is self.source_link:
      self.found = True

  def increment_and_check(self) -> bool:
    self.total_calls += 1
    return self.total_calls > _VIZ_MAX_TOTAL_CALLS


def _make_indent(nest_lvl: int) -> str:
  return '\n' + ' ' * _VIZ_INDENT_WIDTH * nest_lvl


def _get_link_name(link: Link) -> str:
  """User-facing method name from a link.

  Op classes expose _link_name as a slot attr; fallback is 'do'/'then' based
  on ignore_result. Co-read by _engine._record_exception_source.
  """
  op: _PipelineOp | Any = link.v
  link_name: str | None = getattr(op, '_link_name', None)
  if link_name is not None:
    return link_name
  return 'do' if link.ignore_result else 'then'


def _get_obj_name(obj: Any, _depth: int = 0) -> str:
  if getattr(obj, '_quent_is_q', False):
    base = type(obj).__name__
    q_name = getattr(obj, '_name', None)
    return f'{base}[{_sanitize_repr(q_name)}]' if q_name is not None else base
  try:
    name = getattr(obj, '__name__', None)
    if name is None:
      name = getattr(obj, '__qualname__', None)
    if name is not None:
      return str(name)
  except Exception:
    pass
  if hasattr(obj, 'func'):
    if _depth >= 10:  # pragma: no cover
      return 'partial(<...>)'
    return f'partial({_get_obj_name(obj.func, _depth + 1)})'
  if not _show_traceback_values:  # pragma: no cover  # tested via subprocess
    return f'<{type(obj).__name__}>'
  try:
    r = _sanitize_repr(repr(obj))
    return r[:_MAX_REPR_LEN] + '...' if len(r) > _MAX_REPR_LEN else r
  except Exception:
    return type(obj).__name__


def _format_call_args(args: tuple[Any, ...] | None, kwargs: dict[str, Any] | None) -> str:
  parts: list[str] = []
  if args:
    parts.extend(_get_obj_name(a) for a in args)
  if kwargs:
    parts.extend(f'{k}={_get_obj_name(v)}' for k, v in kwargs.items())
  result = ', '.join(parts)
  return result[:_MAX_CALL_ARGS_LEN] + '...' if len(result) > _MAX_CALL_ARGS_LEN else result


def _get_true_source_link(
  source_link: Link | None, root_link: Link | None, max_depth: int = _VIZ_MAX_NESTING_DEPTH
) -> Link | None:
  """Drill through nested pipelines to find the actual failing callable."""
  # `seen` guards against infinite loops if pipelines reference each other.
  seen = set()
  depth = 0
  while source_link is not None and id(source_link) not in seen and depth < max_depth:
    seen.add(id(source_link))
    depth += 1
    if source_link.is_q:
      nested = source_link.v
    elif getattr(source_link.original_value, '_quent_is_q', False):
      nested = source_link.original_value
    else:
      break
    if nested._root_link is not None:
      source_link = nested._root_link
    elif nested._first_link is not None:
      source_link = nested._first_link
    else:
      break
  if source_link is None:
    source_link = root_link
  return source_link


def _resolve_nested_chain(
  link: Link,
  args: tuple[Any, ...] | None,
  kwargs: dict[str, Any] | None,
  nest_lvl: int,
  ctx: _VizContext,
  max_depth: int = _VIZ_MAX_NESTING_DEPTH,
) -> str:
  """Resolve a nested pipeline link to its indented string repr.

  Fresh _VizContext per level — total_calls is isolated so one deep pipeline
  can't exhaust the budget and truncate siblings. The `found` flag is synced
  bidirectionally since it affects error-marker placement globally.
  """
  original_value = link.original_value if link.original_value is not None else link.v
  nested_root_link = None
  nested_args = args or ()
  nested_kwargs = kwargs or {}
  if nested_args or nested_kwargs:
    nested_v = nested_args[0] if nested_args else Null
    if nested_v is not Null or nested_kwargs:
      nested_root_link = Link(
        nested_v,
        args=nested_args[1:] if len(nested_args) > 1 else None,
        kwargs=nested_kwargs or None,
      )
  nested_ctx = _VizContext(source_link=ctx.source_link, link_temp_args=ctx.link_temp_args)
  nested_ctx.found = ctx.found
  result = _stringify_q(
    original_value,
    nest_lvl=nest_lvl + 1,
    root_link=nested_root_link,
    ctx=nested_ctx,
    max_depth=max_depth,
  )
  ctx.found = nested_ctx.found
  return result


def _stringify_q(
  q: Q[Any],
  nest_lvl: int = 0,
  root_link: Link | None = None,
  *,
  ctx: _VizContext,
  extra_links: list[tuple[Link, str]] | None = None,
  max_depth: int = _VIZ_MAX_NESTING_DEPTH,
) -> str:
  """Full string viz of a pipeline; ``<----`` marks the failing link."""
  # max_depth guards against infinite recursion in pathological nesting.
  if nest_lvl >= max_depth or ctx.increment_and_check():
    return f'{_make_indent(nest_lvl)}Q(...<truncated at depth {max_depth}>...)'
  output = ''
  truncated_count = 0
  root = q._root_link
  if root is None and root_link is not None:
    root = root_link

  if nest_lvl > 0:
    output += _make_indent(nest_lvl)
  output += _get_obj_name(q)

  if root is None:
    output += '()'
  else:
    output += _format_link(root, nest_lvl=nest_lvl, ctx=ctx, max_depth=max_depth)
    ctx.mark_found(root)

  links: list[tuple[Link, str]] = []
  link = q._first_link
  while link is not None:
    links.append((link, _get_link_name(link)))
    if getattr(link.v, '_else_link', None) is not None:
      links.append((link.v._else_link, 'else_do' if link.v._else_link.ignore_result else 'else_'))
    link = link.next_link
  if q._on_except_link is not None:
    links.append((q._on_except_link, 'except_'))
  if q._on_finally_link is not None:
    links.append((q._on_finally_link, 'finally_'))

  if len(links) > _VIZ_MAX_LINKS_PER_LEVEL:
    truncated_count = len(links) - _VIZ_MAX_LINKS_PER_LEVEL
    links = links[:_VIZ_MAX_LINKS_PER_LEVEL]

  for link, method_name in links:
    output += _make_indent(nest_lvl)
    output += _format_link(link, nest_lvl=nest_lvl, ctx=ctx, method_name=method_name, max_depth=max_depth)
    ctx.mark_found(link)

  if truncated_count > 0:
    output += _make_indent(nest_lvl)
    output += f'... and {truncated_count} more steps'

  if extra_links:
    for link, method_name in extra_links:
      output += _make_indent(nest_lvl)
      output += _format_link(link, nest_lvl=nest_lvl, ctx=ctx, method_name=method_name, max_depth=max_depth)
      ctx.mark_found(link)

  if len(output) > _VIZ_MAX_LENGTH:
    output = output[:_VIZ_MAX_LENGTH] + '\n... <truncated>'

  return output


def _format_link(
  link: Link, nest_lvl: int, ctx: _VizContext, method_name: str | None = None, max_depth: int = _VIZ_MAX_NESTING_DEPTH
) -> str:
  """Format a single link (including nested pipelines and op-specific rendering).

  op_link: the Link as it appears in the list — its v may be an op class
  exposing _link_name. user_link: the user callable's link (drilled through
  original_value when an op wraps the user's callable in a new Link).
  """
  op_link = link
  op: _PipelineOp | Any = op_link.v
  user_link = link
  if isinstance(user_link.original_value, Link):
    user_link = user_link.original_value

  original_value = user_link.original_value
  args = user_link.args
  kwargs = user_link.kwargs
  output = ''
  is_q = False

  if not ctx.found and ctx.link_temp_args is not None and id(user_link) in ctx.link_temp_args:
    temp_kwargs = ctx.link_temp_args[id(user_link)]
    if temp_kwargs:
      kwargs = {**(kwargs or {}), **temp_kwargs}

  if original_value is None:
    original_value = user_link.v
  if user_link.is_q or getattr(original_value, '_quent_is_q', False):
    link_v = _resolve_nested_chain(user_link, args, kwargs, nest_lvl, ctx, max_depth=max_depth)
    args = kwargs = None
    is_q = True
  else:
    _fns = getattr(op, '_fns', None)
    if _fns is not None:
      link_v = ', '.join(_get_obj_name(f) for f in _fns)
    else:
      link_v = _get_obj_name(original_value)

  if method_name is not None:
    output += f'.{method_name}'

  if user_link.is_callable or user_link.args or user_link.kwargs or user_link.is_q:
    if is_q:
      call_args = _format_call_args(args, kwargs)
      chain_newline = _make_indent(nest_lvl + 1) if call_args else ''
      call_prefix = f', {call_args}' if call_args else ''
      output += f'({link_v}{chain_newline}{call_prefix}{_make_indent(nest_lvl)})'
    else:
      call_args = _format_call_args(args, kwargs)
      call_prefix = f', {call_args}' if call_args else ''
      _concurrency = getattr(op, '_concurrency', None)
      if _concurrency is not None:
        c_str = f'concurrency={_concurrency}'
        call_prefix = f'{call_prefix}, {c_str}' if call_prefix else f', {c_str}'
      output += f'({link_v}{call_prefix})'
  else:
    output += f'({link_v})'

  if not ctx.found and op_link is ctx.source_link:
    output += _ERROR_MARKER

  return output
