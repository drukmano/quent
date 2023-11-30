import collections.abc
from asyncio import ensure_future as _ensure_future

from quent.quent cimport Chain, Cascade, Link, evaluate_value, Null, QuentException, ExceptLink
from quent.custom cimport _Return


cdef object handle_return_exc(_Return exc, bint propagate):
  if propagate:
    raise exc
  if exc._v is Null:
    return None
  return evaluate_value(Link(exc._v, exc.args, exc.kwargs, True), Null)


# this holds a strong reference to all tasks that we create
# see: https://stackoverflow.com/a/75941086
# "... the asyncio loop avoids creating hard references (just weak) to the tasks,
# and when it is under heavy load, it may just "drop" tasks that are not referenced somewhere else."
cdef set task_registry = set()


cdef void remove_task(object task):
  # this may occur when asyncio.ensure_future() is called on a Task -
  # it returns the same Task as-is. and even though we are not registering
  # the callback if the task is already in `task_registry`, a race condition is possible.
  if task in task_registry:
    try:
      task_registry.remove(task)
    except KeyError:
      pass


cdef object ensure_future(object coro):
  cdef object task = _ensure_future(coro)
  if task not in task_registry:
    task_registry.add(task)
    task.add_done_callback(remove_task)
  return task


cdef ExceptLink _handle_exception(object exc, Link link, object cv, int idx):
  cdef ExceptLink exc_link
  cdef object quent_exc = create_chain_link_exception(link, cv, idx), exceptions

  if exc.__cause__ is not None:
    if quent_exc.__cause__ is None:
      quent_exc.__cause__ = exc.__cause__
    else:
      quent_exc.__cause__.__cause__ = exc.__cause__
  exc.__cause__ = quent_exc

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
      try:
        raise exc
      except exceptions:
        return exc_link
      except type(exc):
        continue
    return exc_link
  return None


cdef object create_chain_link_exception(Link link, object cv, int idx):
  # TODO rewrite this, format the entire chain links
  #  each link in a line, show where the exception occurred
  #  along with the result of the previous chains, and any necessary information
  """
  Create a string representation of the evaluation of 'v' based on the same rules
  used in `evaluate_value`
  """

  def get_object_name(o, literal: bool) -> str:
    if literal:
      return str(o)
    try:
      return o.__name__
    except AttributeError:
      return type(o).__name__

  def format_exception_details(literal: bool):
    v, args, kwargs, is_fattr = link.v, link.args, link.kwargs, link.is_fattr
    if v is Null:
      return str(v), 'Null'

    elif link.is_attr:
      if not is_fattr:
        s = get_object_name(cv, literal)
        return f'Attribute \'{v}\' of \'{cv}\'', f'{s}.{v}'

    if args and args[0] is ...:
      if is_fattr:
        s = get_object_name(cv, literal)
        return f'Method attribute \'{v}\' of \'{cv}\'', f'{s}.{v}()'
      s = get_object_name(v, literal)
      return f'{v}', f'{s}()'

    elif args or kwargs:
      kwargs_ = [f'{k}={v_}' for k, v_ in kwargs.items()]
      args_ = ', '.join(str(arg) for arg in list(args) + kwargs_)
      if is_fattr:
        s = get_object_name(cv, literal)
        return f'Method attribute \'{v}\' of \'{cv}\'', f'{s}.{v}({args_})'
      s = get_object_name(v, literal)
      return f'{v}', f'{s}({args_})'

    elif is_fattr:
      s = get_object_name(cv, literal)
      return f'Method attribute \'{v}\' of \'{cv}\'', f'{s}.{v}()'

    elif not link.is_attr and callable(v):
      s = get_object_name(v, literal)
      return f'{v}', f'{s}()' if cv is Null else f'{s}({cv})'

    else:
      return str(v), str(v)

  try:
    try:
      object_str, readable_str = format_exception_details(False)
    except AttributeError as e:
      # this should not happen, but just in case.
      object_str, readable_str = format_exception_details(True)

    if idx == -1:
      s = 'The chain root link has raised an exception:'
    else:
      s = 'A chain link has raised an exception:'
      s += f'\n\tLink position (not including the root link): {idx}'
    return QuentException(
      s
      + f'\n\t{object_str}'
      + f'\n\t`{readable_str}`'
    )

  except Exception as e:
    exc = QuentException('Unable to format exception details.')
    exc.__cause__ = e
    return exc
