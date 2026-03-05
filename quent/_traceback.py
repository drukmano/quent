"""Traceback rewriting: injects readable chain visualizations into exception tracebacks."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import sys
import os
import types
import traceback

from ._core import Null, Link

if TYPE_CHECKING:
  from ._chain import Chain


_quent_file: str = os.path.dirname(os.path.abspath(__file__)) + os.sep
_RAISE_CODE: types.CodeType = compile('raise __exc__', '<quent>', 'exec')
_HAS_QUALNAME: bool = sys.version_info >= (3, 11)
_TracebackType: type[types.TracebackType] = types.TracebackType


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
  """Recursively clean internal frames from chained exceptions."""
  if exc is None or id(exc) in seen:
    return
  seen.add(id(exc))
  if exc.__traceback__ is not None:
    exc.__traceback__ = _clean_internal_frames(exc.__traceback__)
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)


def _remove_self_frames() -> BaseException:
  """Clean internal frames from the current exception and return it
  with the cleaned traceback.
  """
  exc_value = sys.exc_info()[1]
  exc_value.__quent__ = True
  tb = exc_value.__traceback__

  cleaned_tb = _clean_internal_frames(tb)

  seen = set()
  _clean_chained_exceptions(exc_value.__cause__, seen)
  _clean_chained_exceptions(exc_value.__context__, seen)

  return exc_value.with_traceback(cleaned_tb)


def _modify_traceback(exc: BaseException, chain: Chain, link: Link, root_link: Link | None = None) -> None:
  """Inject a readable chain visualization into the exception's traceback.

  Creates a synthetic <quent> frame whose 'function name' is the
  chain visualization string, so the traceback naturally displays
  the chain structure.
  """
  if getattr(exc, '__quent_source_link__', None) is None:
    exc.__quent_source_link__ = link

  if chain.is_nested:
    return

  exc.__quent__ = True
  source_link = exc.__quent_source_link__
  del exc.__quent_source_link__
  link_temp_args = getattr(exc, '__quent_link_temp_args__', None)

  filename = '<quent>'
  chain_source, _ = _stringify_chain(
    chain, nest_lvl=0,
    root_link=root_link,
    link_temp_args=link_temp_args,
    source_link=_get_true_source_link(source_link, root_link),
    found_source_link=False,
  )
  chain_source = _make_indent(1).join([''] + chain_source.splitlines())

  exc_value = sys.exc_info()[1]
  globals_ = {
    '__name__': filename,
    '__file__': filename,
    '__exc__': exc_value,
  }
  if _HAS_QUALNAME:
    code = _RAISE_CODE.replace(co_name=chain_source, co_qualname=chain_source)
  else:
    code = _RAISE_CODE.replace(co_name=chain_source)
  try:
    exec(code, globals_, {})
  except BaseException:
    new_tb = sys.exc_info()[1].__traceback__
    cleaned_tb = _clean_internal_frames(new_tb)
    exc.__traceback__ = cleaned_tb

  seen = set()
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)


def _get_true_source_link(source_link: Link | None, root_link: Link | None) -> Link | None:
  """Retrieve the first non-chain link in a chain of nested chains.

  Drills through nested Chain root links to find the actual user-provided
  callable or value that caused the exception.
  """
  while source_link is not None:
    if source_link.is_chain:
      chain = source_link.v
    elif getattr(source_link.original_value, '_is_chain', False):
      chain = source_link.original_value
    else:
      break
    if root_link is not None:
      source_link = root_link
    else:
      break
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
  if op == 'foreach':
    return 'foreach_do' if v._ignore_result else 'foreach'
  if op == 'filter':
    return 'filter'
  if op == 'gather':
    return 'gather'
  if link.ignore_result:
    return 'do'
  return 'then'


def _get_obj_name(obj: Any) -> str:
  """Get a human-readable name for an object, for traceback display."""
  if getattr(obj, '_is_chain', False):
    return type(obj).__name__
  try:
    name = getattr(obj, '__name__', None) or getattr(obj, '__qualname__', None)
    if name is not None:
      return str(name)
  except Exception:
    pass
  if hasattr(obj, 'func'):  # functools.partial
    return f'partial({_get_obj_name(obj.func)})'
  try:
    return repr(obj)
  except Exception:
    return type(obj).__name__


def _format_args(args: tuple[Any, ...] | None) -> str:
  """Format positional arguments for chain visualization."""
  if not args:
    return ''
  if args[0] is ...:
    return ', ...'
  return ', ' + ', '.join([_get_obj_name(a) for a in args])


def _format_kwargs(kwargs: dict[str, Any] | None) -> str:
  """Format keyword arguments for chain visualization."""
  if not kwargs:
    return ''
  return ', ' + ', '.join([f'{k}={_get_obj_name(v)}' for k, v in kwargs.items()])


def _stringify_chain(chain: Chain, nest_lvl: int = 0, temp_root_link: Link | None = None, link_temp_args: dict[int, tuple[Any, ...]] | None = None,
                     source_link: Link | None = None, found_source_link: bool = False) -> tuple[str, bool]:
  """Build a string visualization of a chain for traceback display.

  Returns a tuple of (output_string, found_source_link_bool).
  """
  output = ''
  root = chain.root_link
  if root is None and temp_root_link is not None:
    root = temp_root_link

  if nest_lvl > 0:
    output += _make_indent(nest_lvl)
  output += _get_obj_name(chain)

  if root is None:
    output += '()'
  else:
    output += _format_link(
      root, nest_lvl=nest_lvl, link_temp_args=link_temp_args,
      source_link=source_link, found_source_link=found_source_link,
    )
    if not found_source_link and root is source_link:
      found_source_link = True

  links = []
  link = chain.first_link
  while link is not None:
    links.append((link, _get_link_name(link)))
    link = link.next_link
  if chain.on_except_link is not None:
    links.append((chain.on_except_link, 'except_'))
  if chain.on_finally_link is not None:
    links.append((chain.on_finally_link, 'finally_'))

  for link, method_name in links:
    output += _make_indent(nest_lvl)
    output += _format_link(
      link, nest_lvl=nest_lvl, link_temp_args=link_temp_args,
      source_link=source_link, found_source_link=found_source_link,
      method_name=method_name,
    )
    if not found_source_link and link is source_link:
      found_source_link = True

  return output, found_source_link


def _format_link(link: Link, nest_lvl: int, link_temp_args: dict[int, tuple[Any, ...]] | None = None, source_link: Link | None = None, found_source_link: bool = False, method_name: str | None = None) -> str:
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

  if not found_source_link:
    if link_temp_args is not None and id(link) in link_temp_args:
      args = link_temp_args[id(link)]
      kwargs = {}

  if original_value is None:
    original_value = link.v
  if link.is_chain or getattr(original_value, '_is_chain', False):
    nested_temp_root_link = None
    _temp_args = args or ()
    _temp_kwargs = kwargs or {}
    if _temp_args or _temp_kwargs:
      _temp_v = _temp_args[0] if _temp_args else Null
      if _temp_v is not Null:
        nested_temp_root_link = Link(
          _temp_v,
          args=_temp_args[1:] if len(_temp_args) > 1 else None,
          kwargs=_temp_kwargs or None,
        )
    args = kwargs = None
    link_v, found_source_link = _stringify_chain(
      original_value, nest_lvl=nest_lvl + 1,
      temp_root_link=nested_temp_root_link, link_temp_args=None,
      source_link=source_link, found_source_link=found_source_link,
    )
    is_chain = True
  else:
    link_v = _get_obj_name(original_value)

  if method_name is not None:
    output += f'.{method_name}'

  # Determine if this link represents a callable invocation.
  if callable(link.v) or link.args or link.kwargs or link.is_chain:
    if is_chain:
      args_s = _format_args(args)
      kwargs_s = _format_kwargs(kwargs)
      chain_newline = ''
      if args_s or kwargs_s:
        chain_newline = _make_indent(nest_lvl + 1)
      link_v = f'({link_v}{chain_newline}{args_s}{kwargs_s}'
      if link_v.endswith(', '):
        link_v = link_v[:-2]
      output += link_v + f'{_make_indent(nest_lvl)})'
    else:
      link_v = f'({link_v}{_format_args(args)}{_format_kwargs(kwargs)}'
      if link_v.endswith(', '):
        link_v = link_v[:-2]
      output += link_v + ')'
  else:
    output += f'({link_v})'

  if not found_source_link:
    if outer_link is source_link:
      output += ' <' + '-' * 4

  return output


_original_excepthook: Any = sys.excepthook


def _quent_excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: types.TracebackType | None) -> None:
  """Custom excepthook that cleans quent internal frames before display."""
  if getattr(exc_value, '__quent__', False):
    _clean_chained_exceptions(exc_value, set())
    exc_tb = exc_value.__traceback__
  _original_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _quent_excepthook


_original_te_init = traceback.TracebackException.__init__


def _patched_te_init(self: traceback.TracebackException, exc_type: type[BaseException], exc_value: BaseException | None = None, exc_tb: types.TracebackType | None = None, **kwargs: Any) -> None:
  """Patched TracebackException.__init__ that cleans quent frames."""
  if exc_value is not None and getattr(exc_value, '__quent__', False):
    _clean_chained_exceptions(exc_value, set())
    exc_tb = exc_value.__traceback__
  _original_te_init(self, exc_type, exc_value, exc_tb, **kwargs)


traceback.TracebackException.__init__ = _patched_te_init
