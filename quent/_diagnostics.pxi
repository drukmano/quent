import os as _os
import inspect

_quent_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
cdef type _TracebackType = types.TracebackType
cdef object _isroutine = inspect.isroutine
cdef object _isclass = inspect.isclass
_RAISE_CODE = compile('raise __exc__', '<quent>', 'exec')
cdef bint _HAS_QUALNAME = sys.version_info >= (3, 11)


cdef object clean_internal_frames(object tb):
  cdef list stack = []
  cdef object tb_next = None
  cdef object new_tb
  cdef str filename

  while tb is not None:
    filename = tb.tb_frame.f_code.co_filename
    # Keep only frames that are NOT from quent internal files or are the special <quent> frame
    if filename == '<quent>':
      # Always keep the special <quent> frames showing the chain
      stack.append(tb)
    elif not filename.startswith(_quent_pkg_dir):
      # Also check if it's not marked as internal
      if tb.tb_frame.f_globals.get('__QUENT_INTERNAL__') is not __QUENT_INTERNAL__:
        # Keep user code frames
        stack.append(tb)
    tb = tb.tb_next

  # Build the chain in reverse using types.TracebackType constructor
  for tb in reversed(stack):
    new_tb = _TracebackType(tb_next, tb.tb_frame, tb.tb_lasti, tb.tb_lineno)
    tb_next = new_tb

  return tb_next


cdef void _clean_chained_exceptions(object exc, set seen):
  if exc is None or id(exc) in seen:
    return
  seen.add(id(exc))
  if exc.__traceback__ is not None:
    exc.__traceback__ = clean_internal_frames(exc.__traceback__)
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)


cdef object remove_self_frames_from_traceback():
  cdef object exc_value, tb, cleaned_tb
  exc_value = sys.exc_info()[1]
  setattr(exc_value, '__quent__', True)
  tb = exc_value.__traceback__

  # Use the same cleaning logic as clean_internal_frames
  cleaned_tb = clean_internal_frames(tb)

  # Recursively clean chained exceptions
  cdef set seen = set()
  _clean_chained_exceptions(exc_value.__cause__, seen)
  _clean_chained_exceptions(exc_value.__context__, seen)

  return exc_value.with_traceback(cleaned_tb)

# --- Traceback augmentation ---

cdef void modify_traceback(object exc, Chain chain, Link link, _ExecCtx ctx):
  cdef Link source_link
  cdef str filename
  cdef str chain_source
  cdef object exc_value, tb, new_tb, cleaned_tb
  cdef object code
  cdef dict globals
  # save the link that the exception was thrown at.
  if getattr(exc, '__quent_source_link__', None) is None:
    setattr(exc, '__quent_source_link__', link)

  if chain.is_nested:
    return
  source_link = getattr(exc, '__quent_source_link__')
  delattr(exc, '__quent_source_link__')
  filename = '<quent>'
  chain_source, _ = stringify_chain(
    chain, ctx, nest_lvl=0, source_link=get_true_source_link(source_link, ctx), found_source_link=False
  )
  chain_source = make_indent(1).join([''] + chain_source.splitlines())
  exc_value = sys.exc_info()[1]
  globals = {
    '__name__': filename,
    '__file__': filename,
    '__exc__': exc_value,
  }
  if _HAS_QUALNAME:
    code = _RAISE_CODE.replace(co_name=chain_source, co_qualname=chain_source)
  else:
    code = _RAISE_CODE.replace(co_name=chain_source)
  try:
    exec(code, globals, {})
  except BaseException:
    # Get the traceback from the exec'd code which includes our <quent> frame
    new_tb = sys.exc_info()[1].__traceback__
    # Clean the traceback to remove internal quent frames while keeping <quent> frames
    cleaned_tb = clean_internal_frames(new_tb)
    exc.__traceback__ = cleaned_tb

  # Recursively clean chained exception tracebacks
  cdef set seen = set()
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)

cdef Link get_true_source_link(Link source_link, _ExecCtx ctx):
  """ retrieves the first non-chain link """
  cdef Chain chain
  while source_link is not None:
    if source_link.is_chain:
      chain = source_link.v
    elif isinstance(source_link.original_value, Chain):
      chain = source_link.original_value
    else:
      break
    if chain.root_link is not None:
      source_link = chain.root_link
    elif ctx is not None and ctx.temp_root_link is not None:
      source_link = ctx.temp_root_link
    else:
      break
  return source_link

cdef str make_indent(int nest_lvl):
  return '\n' + ' ' * 4 * nest_lvl


cdef str _get_link_name(Link link):
  """Reconstruct a display name for a link from its type and flags (cold path only)."""
  cdef type vt = type(link.v)
  if vt is _Foreach:
    return 'foreach'
  if vt is _Filter:
    return 'filter'
  if vt is _Gather:
    return 'gather'
  if vt is _With:
    return 'with_'
  if link.ignore_result:
    return 'do'
  return 'then'


cdef tuple stringify_chain(Chain chain, _ExecCtx ctx, int nest_lvl = 0, Link source_link = None, bint found_source_link = False):
  cdef str output = ''
  cdef Link link = chain.root_link
  if link is None and ctx is not None:
    link = ctx.temp_root_link

  if nest_lvl > 0:
    output += make_indent(nest_lvl)
  output += get_obj_name(chain)
  if link is None:
    output += '()'
  else:
    output += format_link(link, ctx, nest_lvl=nest_lvl, source_link=source_link, found_source_link=found_source_link)
    if not found_source_link and link is source_link:
      found_source_link = True

  link = chain.first_link
  while link is not None:
    output += make_indent(nest_lvl)
    output += format_link(link, ctx, nest_lvl=nest_lvl, source_link=source_link, found_source_link=found_source_link, method_name=_get_link_name(link))
    if not found_source_link and link is source_link:
      found_source_link = True
    link = link.next_link

  link = chain.on_except_link
  if link is not None:
    output += make_indent(nest_lvl)
    output += format_link(link, ctx, nest_lvl=nest_lvl, source_link=source_link, found_source_link=found_source_link, method_name='except_')
    if not found_source_link and link is source_link:
      found_source_link = True

  link = chain.on_finally_link
  if link is not None:
    output += make_indent(nest_lvl)
    output += format_link(link, ctx, nest_lvl=nest_lvl, source_link=source_link, found_source_link=found_source_link, method_name='finally_')
    if not found_source_link and link is source_link:
      found_source_link = True

  return output, found_source_link


cdef str format_args(tuple args):
  if not args:
    return ''
  if args[0] is ...:
    return ', ...'
  return ', ' + ', '.join([get_obj_name(a) for a in args])


cdef str format_kwargs(dict kwargs):
  if not kwargs:
    return ''
  return ', ' + ', '.join([f'{k}={get_obj_name(v)}' for k, v in kwargs.items()])


cdef str format_link(Link link, _ExecCtx ctx, int nest_lvl, Link source_link = None, bint found_source_link = False, str method_name = None):
  cdef Link outer_link = link
  if isinstance(link.original_value, Link):
    link = link.original_value

  cdef object original_value = link.original_value
  cdef tuple args = link.args
  cdef dict kwargs = link.kwargs
  cdef str link_v = ''
  cdef str output = ''
  cdef bint is_chain = False
  cdef tuple _temp_args
  cdef dict _temp_kwargs
  cdef object _temp_v
  cdef _ExecCtx nested_ctx
  cdef object _result

  if not found_source_link:
    if ctx is not None and ctx.link_temp_args is not None and id(link) in ctx.link_temp_args:
      args = ctx.link_temp_args[id(link)]
      kwargs = {}
    elif link.temp_args:
      args = link.temp_args
      kwargs = {}

  if original_value is None:
    original_value = link.v
  if link.is_chain or isinstance(original_value, Chain):
    nested_ctx = _ExecCtx.__new__(_ExecCtx)
    nested_ctx.link_temp_args = None
    _temp_args = args or ()
    _temp_kwargs = kwargs or {}
    if _temp_args or _temp_kwargs:
      _temp_v = _temp_args[0] if _temp_args else Null
      if _temp_v is not Null:
        nested_ctx.temp_root_link = Link(_temp_v, _temp_args[1:], _temp_kwargs)
    args = kwargs = None
    link_v, found_source_link = stringify_chain(
      original_value, nested_ctx, nest_lvl=nest_lvl + 1, source_link=source_link, found_source_link=found_source_link
    )
    is_chain = True
  else:
    link_v = get_obj_name(original_value)

  if method_name is not None:
    output += f'.{method_name}'
  if link.eval_code in {EVAL_CALL_WITHOUT_ARGS, EVAL_CALL_WITH_EXPLICIT_ARGS, EVAL_CALL_WITH_CURRENT_VALUE}:
    if is_chain:
      args_s = format_args(args)
      kwargs_s = format_kwargs(kwargs)
      chain_newline = ''
      if args_s or kwargs_s:
        chain_newline = make_indent(nest_lvl + 1)
      link_v = f'({link_v}{chain_newline}{args_s}{kwargs_s}'
      if link_v.endswith(', '):
        link_v = link_v[:len(link_v) - 2]
      output += link_v + f'{make_indent(nest_lvl)})'
    else:
      link_v = f'({link_v}{format_args(args)}{format_kwargs(kwargs)}'
      if link_v.endswith(', '):
        link_v = link_v[:len(link_v) - 2]
      output += link_v + ')'
  else:
    output += f'({link_v})'
  if not found_source_link:
    if outer_link is source_link:
      output += ' <' + '-' * 4
    elif ctx is not None and ctx.link_results is not None:
      _result = ctx.link_results.get(id(outer_link), Null)
      if _result is not Null:
        output += ' = ' + repr(_result)[:100]
  return output


cdef str get_obj_name(object obj):
  if isinstance(obj, Chain):
    return type(obj).__name__
  cdef object name
  try:
    name = getattr(obj, '__qualname__', None) or getattr(obj, '__name__', None)
    if name is not None:
      return str(name)
  except Exception:
    pass
  if hasattr(obj, 'func'):  # functools.partial
    return f'partial({get_obj_name(obj.func)})'
  try:
    return repr(obj)
  except Exception:
    return type(obj).__name__


_original_excepthook = sys.excepthook

def _quent_excepthook(exc_type, exc_value, exc_tb):
  if getattr(exc_value, '__quent__', False):
    _clean_chained_exceptions(exc_value, set())
    exc_tb = exc_value.__traceback__
  _original_excepthook(exc_type, exc_value, exc_tb)

sys.excepthook = _quent_excepthook

def _clean_exc_chain(exc):
  """Python-visible wrapper for _clean_chained_exceptions."""
  _clean_chained_exceptions(exc, set())
