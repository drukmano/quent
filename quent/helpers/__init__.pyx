import types
import collections.abc
from asyncio import ensure_future as _ensure_future

from quent.quent import Cascade
from quent.link cimport Link, evaluate_value


cdef set task_registry = set()


cdef class _Null:
  def __repr__(self):
    return '<Null>'

cdef _Null Null = _Null()


cdef class QuentException(Exception):
  pass


# a modified version of `asyncio.iscoroutine`.
cdef:
  # `abc.Awaitable` might catch most if not all awaitable objects, but checking
  # for `types.CoroutineType` is about x2 faster and also catches most coroutines.
  tuple _AWAITABLE_TYPES = (types.CoroutineType, collections.abc.Awaitable)
  set _isawaitable_typecache = set()  # TODO = _AWAITABLE_TYPES ?
  # TODO add more primitives
  set _nonawaitable_typecache = {int, str, bool, float, set, tuple, dict, list}
  int _isawaitable_cache_count = 0
  int _nonawaitable_cache_count = len(_nonawaitable_typecache)

cdef bint isawaitable(object obj):
  global _isawaitable_cache_count, _nonawaitable_cache_count

  cdef type obj_t = type(obj)
  if obj_t in _nonawaitable_typecache:
    return False
  elif obj_t in _isawaitable_typecache:
    return True

  if isinstance(obj, _AWAITABLE_TYPES):
    if _isawaitable_cache_count < 1000:
      _isawaitable_cache_count += 1
      _isawaitable_typecache.add(obj_t)
    return True

  else:
    if _nonawaitable_cache_count < 1000:
      _nonawaitable_cache_count += 1
      _nonawaitable_typecache.add(obj_t)
    return False


cdef object _handle_exception(object exc, list except_links, Link link, object rv, object cv, int idx):
  cdef object quent_exc = create_chain_link_exception(link, cv, idx), exceptions
  cdef bint reraise = True, raise_, exc_match
  cdef object chain
  if exc.__cause__ is not None:
    if quent_exc.__cause__ is None:
      quent_exc.__cause__ = exc.__cause__
    else:
      quent_exc.__cause__.__cause__ = exc.__cause__
  exc.__cause__ = quent_exc
  if except_links is None:
    return None, reraise
  chain = Cascade()
  for link, exceptions, raise_ in except_links:
    if exceptions is None:
      exc_match = True
    else:
      if not isinstance(exceptions, collections.abc.Iterable):
        exceptions = (exceptions,)
      else:
        exceptions = tuple(exceptions)
      exc_match = False
      try:
        raise exc
      except exceptions:
        exc_match = True
      except Exception:
        pass
    if exc_match:
      reraise = raise_
      chain.then(evaluate_value, link, rv)
  return chain.run(), reraise


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
      object_str, readable_str = format_exception_details(literal=False)
    except AttributeError as e:
      # this should not happen, but just in case.
      object_str, readable_str = format_exception_details(literal=True)

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
