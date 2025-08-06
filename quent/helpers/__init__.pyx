import sys
import inspect
import collections.abc
from asyncio import ensure_future as _ensure_future

from quent._internal import __QUENT_INTERNAL__
from quent.quent cimport (
  Chain, Cascade, Link, evaluate_value, Null, QuentException, EVAL_NO_ARGS, EVAL_CUSTOM_ARGS, EVAL_CALLABLE
)
from quent.custom cimport _Return


# this holds a strong reference to all tasks that we create
# see: https://stackoverflow.com/a/75941086
# "... the asyncio loop avoids creating hard references (just weak) to the tasks,
# and when it is under heavy load, it may just "drop" tasks that are not referenced somewhere else."
cdef set task_registry = set()


cdef object ensure_future(object coro):
  cdef object task = _ensure_future(coro)
  task_registry.add(task)
  task.add_done_callback(task_registry.discard)
  return task


cdef object handle_return_exc(_Return exc, bint propagate):
  if propagate:
    raise exc
  if exc._v is Null:
    return None
  return evaluate_value(Link(exc._v, exc.args, exc.kwargs, True), Null)


cdef object clean_internal_frames(object tb):
  """Clean internal frames from a traceback while preserving <quent> frames"""
  cdef list stack = []
  cdef object tb_next = None
  cdef str filename
  
  while tb is not None:
    filename = tb.tb_frame.f_code.co_filename
    # Keep only frames that are NOT from quent internal files or are the special <quent> frame
    if filename == '<quent>':
      # Always keep the special <quent> frames showing the chain
      stack.append(tb)
    elif filename not in ('quent/quent.pyx', 'quent/helpers/__init__.pyx', 'quent/custom/__init__.pyx'):
      # Also check if it's not marked as internal
      if not tb.tb_frame.f_globals.get('__QUENT_INTERNAL__') is __QUENT_INTERNAL__:
        # Keep user code frames
        stack.append(tb)
    tb = tb.tb_next
  
  # Assign tb_next in reverse to avoid circular references
  for tb in reversed(stack):
    tb.tb_next = tb_next
    tb_next = tb
  
  return tb_next


cdef object remove_self_frames_from_traceback():
  # TODO also handle exc.__context__ and __cause__
  cdef object _, exc_value, tb
  _, exc_value, tb = sys.exc_info()
  
  # Use the same cleaning logic as clean_internal_frames
  cleaned_tb = clean_internal_frames(tb)
  
  return exc_value.with_traceback(cleaned_tb)


cdef Link _handle_exception(object exc, Chain chain, Link link):
  cdef Link exc_link
  modify_traceback(exc, chain, link)

  while link is not None:
    if not link.is_exception_handler:
      link = link.next_link
      continue
    exc_link = link
    link = link.next_link
    if exc_link.exceptions is not None:
      exceptions = exc_link.exceptions
      if not isinstance(exceptions, collections.abc.Iterable):
        exceptions = (exceptions,)
      else:
        exceptions = tuple(exceptions)
      if not issubclass(type(exc), exceptions):
        continue
    return exc_link
  return None


cdef void modify_traceback(object exc, Chain chain, Link link):
  cdef Link source_link

  # save the link that the exception was thrown at.
  if getattr(exc, '__quent_source_link__', None) is None:
    setattr(exc, '__quent_source_link__', link)

  if chain.is_nested:
    return
  source_link = getattr(exc, '__quent_source_link__')
  delattr(exc, '__quent_source_link__')
  filename = '<quent>'
  chain_source, _ = stringify_chain(
    chain, nest_lvl=0, source_link=get_true_source_link(source_link), found_source_link=False
  )
  chain_source = make_indent(1).join([''] + chain_source.splitlines())
  _, exc_value, tb = sys.exc_info()
  globals = {
    '__name__': filename,
    '__file__': filename,
    '__exc__': exc_value,
  }
  # TODO `'\n' * (link-global-idx) + 'raise __exc__'` to include the
  #  line number as the link index (globally; include parent chains).
  code = compile('raise __exc__', filename, 'exec')
  # TODO add support for older versions; test the lowest python version quent works for.
  #if sys.version_info >= (3, 8):
  code = code.replace(co_name=chain_source)
  try:
    exec(code, globals, {})
  except BaseException:
    # Get the traceback from the exec'd code which includes our <quent> frame
    _, _, new_tb = sys.exc_info()
    # Clean the traceback to remove internal quent frames while keeping <quent> frames
    cleaned_tb = clean_internal_frames(new_tb)
    exc.__traceback__ = cleaned_tb


cdef Link get_true_source_link(Link source_link):
  """ retrieves the first non-chain link """
  cdef Chain chain
  while source_link is not None:
    if source_link.is_chain:
      chain = source_link.v
    elif isinstance(source_link.ogv, Chain):
      chain = source_link.ogv
    else:
      break
    if chain.root_link is not None:
      source_link = chain.root_link
    elif chain.temp_root_link is not None:
      source_link = chain.temp_root_link
    else:
      break
  return source_link


cdef str make_indent(int nest_lvl):
  return '\n' + ' ' * 4 * nest_lvl


cdef tuple stringify_chain(Chain chain, int nest_lvl = 0, Link source_link = None, bint found_source_link = False):
  cdef str s = ''
  cdef Link link = chain.root_link
  if link is None:
    link = chain.temp_root_link

  if nest_lvl > 0:
    s += make_indent(nest_lvl)
  s += get_obj_name(chain)
  if link is None:
    s += '()'
  else:
    s += format_link(link, nest_lvl=nest_lvl, source_link=source_link, found_source_link=found_source_link)
    if not found_source_link and link is source_link:
      found_source_link = True

  link = chain.first_link
  while link is not None:
    s += make_indent(nest_lvl)
    s += format_link(link, nest_lvl=nest_lvl, source_link=source_link, found_source_link=found_source_link)
    if not found_source_link and link is source_link:
      found_source_link = True
    link = link.next_link

  link = chain.on_finally_link
  if link is not None:
    s += make_indent(nest_lvl)
    s += format_link(link, nest_lvl=nest_lvl, source_link=source_link, found_source_link=found_source_link)
    if not found_source_link and link is source_link:
      found_source_link = True

  return s, found_source_link


cdef str format_link(Link link, int nest_lvl, Link source_link = None, bint found_source_link = False):
  cdef Link oglink = link
  if isinstance(link.ogv, Link):
    link = link.ogv

  cdef object ogv = link.ogv
  cdef tuple args = link.args
  cdef dict kwargs = link.kwargs
  cdef str link_v = ''
  cdef str s = ''
  cdef bint is_chain = False

  def format_args():
    if args and args[0] is ...:
      return ', ...'
    return ', '.join([''] + [get_obj_name(a) for a in (args or ())])

  def format_kwargs():
    return ', '.join([''] + [f'{k}={get_obj_name(v)}' for k, v in (kwargs or {}).items()])

  def set_temp_root_link(Chain chain, v = Null, *args, **kwargs):
    if v is not Null:
      chain.temp_root_link = Link(v, args, kwargs, True)

  if not found_source_link and link.temp_args:
    args = link.temp_args
    kwargs = {}

  if ogv is None:
    ogv = link.v
  if link.is_chain or isinstance(ogv, Chain):
    if (<Chain>ogv).temp_root_link is None:
      set_temp_root_link(ogv, *(args or ()), **(kwargs or {}))
    args = kwargs = None
    link_v, found_source_link = stringify_chain(
      ogv, nest_lvl=nest_lvl + 1, source_link=source_link, found_source_link=found_source_link
    )
    is_chain = True
  else:
    link_v = get_obj_name(ogv)

  if link.fn_name is not None:
    s += f'.{link.fn_name}'
  if link.eval_code in {EVAL_NO_ARGS, EVAL_CUSTOM_ARGS, EVAL_CALLABLE}:
    if is_chain:
      args_s = format_args()
      kwargs_s = format_kwargs()
      chain_newline = ''
      # this moves the arguments one line down while keeping the "," on the same line.
      # comment this condition and uncomment the following two to see.
      if args_s or kwargs_s:
        chain_newline = make_indent(nest_lvl + 1)
      #if args_s:
      #  args_s = args_s[:1] + '\n' + args_s[2:]
      #elif kwargs_s:
      #  kwargs_s = kwargs_s[:1] + '\n' + kwargs_s[2:]
      s += f'({link_v}{chain_newline}{args_s}{kwargs_s}'.rstrip(', ') + f'{make_indent(nest_lvl)})'
    else:
      s += f'({link_v}{format_args()}{format_kwargs()}'.rstrip(', ') + ')'
  else:
    s += f'({link_v})'
  if not found_source_link:
    if oglink is source_link:
      s += ' <' + '-' * 4
    elif oglink.result is not Null:
      s += ' = ' + repr(oglink.result)[:100]
  return s


cdef str get_obj_name(object o):
  if inspect.isroutine(o) or inspect.isclass(o):
    return o.__name__
  if isinstance(o, Chain):
    return type(o).__name__
  return repr(o)
