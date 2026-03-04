import sys
import os
import asyncio
import logging
import types
import functools
import warnings
import collections.abc
import inspect
import traceback as _traceback_module


_logger = logging.getLogger('quent')


class _Null:
  __slots__ = ()

  def __repr__(self):
    return '<Null>'


Null = _Null()
_iscoro = inspect.isawaitable


class QuentException(Exception):
  __slots__ = ()


class _InternalQuentException(Exception):
  __slots__ = ('value', 'args_', 'kwargs_')

  def __init__(self, v, args, kwargs):
    self.value = v
    self.args_ = args
    self.kwargs_ = kwargs

  def __repr__(self):
    return f'<{type(self).__name__}>'


class _Return(_InternalQuentException):
  __slots__ = ()


class _Break(_InternalQuentException):
  __slots__ = ()


# TODO convert to Enum
EVAL_CALL_WITH_EXPLICIT_ARGS = 1001
EVAL_CALL_WITHOUT_ARGS = 1002
EVAL_CALL_WITH_CURRENT_VALUE = 1003
EVAL_RETURN_AS_IS = 1004


def _eval_signal_value(v, args, kwargs):
  if args and args[0] is ...:
    return v()
  if args or kwargs:
    return v(*args, **kwargs)
  return v() if callable(v) else v


def _handle_break_exc(exc, fallback):
  if exc.value is Null:
    return fallback
  return _eval_signal_value(exc.value, exc.args_, exc.kwargs_)


def _handle_return_exc(exc, propagate):
  if propagate:
    raise exc
  if exc.value is Null:
    return None
  return _eval_signal_value(exc.value, exc.args_, exc.kwargs_)


if sys.version_info >= (3, 14):
  _create_task_fn = functools.partial(asyncio.create_task, eager_start=True)
else:
  _create_task_fn = asyncio.create_task

_task_registry = set()
def _ensure_future(coro):
  task = _create_task_fn(coro)
  _task_registry.add(task)
  task.add_done_callback(_task_registry.discard)
  return task


class _ExecCtx:
  __slots__ = (
    'temp_root_link', 'link_results', 'link_temp_args',
    'async_link', 'current_value', 'root_value', 'has_root_value',
  )

  def __init__(self):
    # TODO why do we need the temp root link? link results? temp?
    #  these are artifacts of performance optimization; we no longer need this.
    #  code should be as simple, elegant, and beautifully crafted as possible. readability
    #  wins. simplicity wins.
    self.temp_root_link = None
    self.link_results = None
    self.link_temp_args = None
    self.async_link = None
    self.current_value = Null
    self.root_value = Null
    self.has_root_value = False


async def _await_run(result, chain=None, link=None, ctx=None):
  try:
    return await result
  except BaseException as exc:
    if chain is not None and link is not None:
      _modify_traceback(exc, chain, link, ctx)
    raise _remove_self_frames()


def _determine_eval_code(link, v, args, kwargs):
  if not args and not kwargs:
    if callable(v):
      return EVAL_CALL_WITH_CURRENT_VALUE
    else:
      return EVAL_RETURN_AS_IS
  elif args:
    if args[0] is ...:
      return EVAL_CALL_WITHOUT_ARGS
    else:
      if kwargs is None:
        link.kwargs = {}
      return EVAL_CALL_WITH_EXPLICIT_ARGS
  else:
    if link.is_chain:
      return EVAL_CALL_WITHOUT_ARGS
    else:
      if args is None:
        link.args = ()
      return EVAL_CALL_WITH_EXPLICIT_ARGS


class Link:
  __slots__ = (
    'v', 'next_link', 'eval_code', 'is_chain', 'ignore_result',
    'args', 'kwargs', 'original_value', 'temp_args',
  )

  def __init__(self, v, args=None, kwargs=None, ignore_result=False, original_value=None):
    is_chain_type = type(v) is Chain
    if is_chain_type:
      self.is_chain = True
      v.is_nested = True
    else:
      self.is_chain = False
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.ignore_result = ignore_result
    self.original_value = original_value
    self.temp_args = None
    self.next_link = None
    # TODO why early eval?
    #  these are artifacts of performance optimization; we no longer need this.
    #  code should be as simple, elegant, and beautifully crafted as possible. readability
    #  wins. simplicity wins.
    self.eval_code = _determine_eval_code(self, v, args, kwargs)
    if is_chain_type and self.eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS and self.args:
      # TODO why temp args?
      #  these are artifacts of performance optimization; we no longer need this.
      #  code should be as simple, elegant, and beautifully crafted as possible. readability
      #  wins. simplicity wins.
      self.temp_args = self.args[1:] if len(self.args) > 1 else ()


# TODO remove; use directly
def _make_temp_link(v, args, kwargs):
  return Link(v, args=args, kwargs=kwargs, ignore_result=False, original_value=v)


# TODO remove; use directly
def _create_link(v, args, kwargs, ignore_result=False, original_value=None):
  return Link(v, args=args, kwargs=kwargs, ignore_result=ignore_result, original_value=original_value)


def _evaluate_value(link, current_value):
  # Fast path for most common case: simple callable with single argument
  # TODO no need for optimization here. do a regular hot path (most common to least common) order but thats it
  if link.eval_code == EVAL_CALL_WITH_CURRENT_VALUE and not link.is_chain:
    if current_value is Null:
      return link.v()
    return link.v(current_value)

  if link.is_chain:
    if link.eval_code == EVAL_CALL_WITH_CURRENT_VALUE:
      return link.v._run(current_value, None, None, True)
    elif link.eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS:
      if not link.args or link.kwargs is None:
        return link.v._run(Null, None, None, True)
      return link.v._run(link.args[0], link.temp_args, link.kwargs, True)
    elif link.eval_code == EVAL_CALL_WITHOUT_ARGS:
      return link.v._run(Null, None, None, True)
    else:
      raise QuentException(
        'Invalid evaluation code found for a nested chain. '
        'If you see this error then something has gone terribly wrong.'
      )
  else:
    if link.eval_code == EVAL_CALL_WITH_EXPLICIT_ARGS:
      if link.args is None or link.kwargs is None:
        return link.v()
      if link.kwargs is _EMPTY_DICT:
        return link.v(*link.args)
      return link.v(*link.args, **link.kwargs)
    elif link.eval_code == EVAL_CALL_WITHOUT_ARGS:
      return link.v()
    else:
      return link.v


# ─── With (Context Manager Support) ─────────────────────────────────────────

# TODO would this be or not be simpler with a function than with a class? again - readability counts. simplicity.
#  these are artifacts of performance optimization; we no longer need this.
#  code should be as simple, elegant, and beautifully crafted as possible. readability
#  wins. simplicity wins.
class _With:
  __slots__ = ('link', 'ignore_result', 'args', 'kwargs')

  def __init__(self, link, ignore_result, args, kwargs):
    self.link = link
    self.ignore_result = ignore_result
    self.args = args
    self.kwargs = kwargs

  def __call__(self, current_value):
    outer_value = current_value
    if hasattr(current_value, '__aenter__'):
      return _with_full_async(self.link, current_value, self.ignore_result)
    entered = False
    try:
      ctx = current_value.__enter__()
      entered = True
      if not self.args and not self.kwargs:
        self.link.temp_args = (ctx,)
      result = _evaluate_value(self.link, ctx)
      if _iscoro(result):
        return _with_to_async(current_value, result, self.link, entered, self.ignore_result, outer_value)
    except BaseException as exc:
      if entered:
        if not current_value.__exit__(type(exc), exc, exc.__traceback__):
          raise
      else:
        raise
    else:
      if entered:
        current_value.__exit__(None, None, None)
      if self.ignore_result:
        return outer_value
      return result


def _make_with_link(fn, args, kwargs, ignore_result):
  link = _create_link(fn, args, kwargs)
  return _create_link(_With(link, ignore_result, args, kwargs), None, None, ignore_result, link)


async def _with_to_async(current_value, body_result, link, entered, ignore_result, outer_value):
  try:
    body_result = await body_result
  except BaseException as exc:
    if entered:
      exit_result = current_value.__exit__(type(exc), exc, exc.__traceback__)
      if _iscoro(exit_result):
        exit_result = await exit_result
      if not exit_result:
        raise
    else:
      raise
  else:
    exit_result = current_value.__exit__(None, None, None)
    if _iscoro(exit_result):
      await exit_result
    if ignore_result:
      return outer_value
    return body_result


async def _with_full_async(link, current_value, ignore_result=False):
  outer_value = current_value
  async with current_value as ctx:
    if not link.args and not link.kwargs:
      link.temp_args = (ctx,)
    try:
      result = _evaluate_value(link, ctx)
      if _iscoro(result):
        result = await result
    except BaseException as exc:
      if not hasattr(exc, '__quent_link_temp_args__'):
        exc.__quent_link_temp_args__ = {}
      exc.__quent_link_temp_args__[id(link)] = (ctx,)
      raise
    if ignore_result:
      return outer_value
    return result


# ─── Generator ───────────────────────────────────────────────────────────────

_DEFAULT_RUN_ARGS = (Null, _EMPTY_TUPLE, _EMPTY_DICT, False)


def _sync_generator(iterator_getter, run_args, fn, ignore_result):
  try:
    for el in iterator_getter(*run_args):
      if fn is None:
        yield el
      else:
        result = fn(el)
        if ignore_result:
          yield el
        else:
          yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


async def _async_generator(iterator_getter, run_args, fn, ignore_result):
  iterator = iterator_getter(*run_args)
  if _iscoro(iterator):
    iterator = await iterator
  is_aiter = hasattr(iterator, '__aiter__')
  try:
    if is_aiter:
      async for el in iterator:
        if fn is None:
          yield el
        else:
          result = fn(el)
          if _iscoro(result):
            result = await result
          if ignore_result:
            yield el
          else:
            yield result
    else:
      for el in iterator:
        if fn is None:
          yield el
        else:
          result = fn(el)
          if _iscoro(result):
            result = await result
          if ignore_result:
            yield el
          else:
            yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


class _Generator:
  __slots__ = ('_chain_run', '_fn', '_ignore_result', '_run_args')

  def __init__(self, chain_run, fn, _ignore_result):
    self._chain_run = chain_run
    self._fn = fn
    self._ignore_result = _ignore_result
    self._run_args = _DEFAULT_RUN_ARGS

  def __call__(self, v=Null, *args, **kwargs):
    # TODO
    #  these are artifacts of performance optimization; we no longer need this.
    #  code should be as simple, elegant, and beautifully crafted as possible. readability
    #  wins. simplicity wins.
    g = _Generator.__new__(_Generator)
    g._chain_run = self._chain_run
    g._fn = self._fn
    g._ignore_result = self._ignore_result
    g._run_args = (v, args, kwargs, False)
    return g

  def __iter__(self):
    return _sync_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __aiter__(self):
    return _async_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __repr__(self):
    return '<_Generator>'


# ─── Foreach ─────────────────────────────────────────────────────────────────

class _Foreach:
  __slots__ = ('fn', 'ignore_result', 'link')

  def __init__(self, fn, ignore_result, link):
    self.fn = fn
    self.ignore_result = ignore_result
    self.link = link

  def __call__(self, current_value):
    if hasattr(current_value, '__aiter__'):
      return _foreach_full_async(current_value, self.fn, self.ignore_result, self.link)
    lst = []
    el = None
    result = None
    it = iter(current_value)
    try:
      while True:
        el = next(it)
        result = self.fn(el)
        if _iscoro(result):
          self.link.temp_args = (el,)
          return _foreach_to_async(it, self.fn, el, result, lst, self.ignore_result, self.link)
        if self.ignore_result:
          lst.append(el)
        else:
          lst.append(result)
    except _Break as exc:
      return _handle_break_exc(exc, lst)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


def _make_foreach_link(fn, ignore_result):
  link = _create_link(fn, _EMPTY_TUPLE, _EMPTY_DICT)
  return _create_link(_Foreach(fn, ignore_result, link), None, None, False, link)


async def _foreach_to_async(current_value, fn, el, result, lst, ignore_result, link):
  try:
    while True:
      if _iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      el = next(current_value)
      result = fn(el)
  except _Break as exc:
    result = _handle_break_exc(exc, lst)
    if _iscoro(result):
      return await result
    return result
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def _foreach_full_async(current_value, fn, ignore_result, link):
  lst = []
  el = None
  result = None
  try:
    async for el in current_value:
      result = fn(el)
      if _iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
    return lst
  except _Break as exc:
    result = _handle_break_exc(exc, lst)
    if _iscoro(result):
      return await result
    return result
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


# ─── Filter ──────────────────────────────────────────────────────────────────

class _Filter:
  __slots__ = ('fn', 'link')

  def __init__(self, fn, link):
    self.fn = fn
    self.link = link

  def __call__(self, current_value):
    if hasattr(current_value, '__aiter__'):
      return _filter_full_async(current_value, self.fn, self.link)
    lst = []
    el = None
    result = None
    it = iter(current_value)
    try:
      while True:
        el = next(it)
        result = self.fn(el)
        if _iscoro(result):
          self.link.temp_args = (el,)
          return _filter_to_async(it, self.fn, el, result, lst, self.link)
        if result:
          lst.append(el)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


def _make_filter_link(fn):
  link = _create_link(fn, _EMPTY_TUPLE, _EMPTY_DICT)
  return _create_link(_Filter(fn, link), None, None, False, link)


async def _filter_to_async(current_value, fn, el, result, lst, link):
  try:
    while True:
      if _iscoro(result):
        result = await result
      if result:
        lst.append(el)
      el = next(current_value)
      result = fn(el)
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def _filter_full_async(current_value, fn, link):
  lst = []
  el = None
  result = None
  try:
    async for el in current_value:
      result = fn(el)
      if _iscoro(result):
        result = await result
      if result:
        lst.append(el)
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


# ─── Gather ──────────────────────────────────────────────────────────────────

class _Gather:
  __slots__ = ('fns', 'link')

  def __init__(self, fns, link):
    self.fns = fns
    self.link = link

  def __call__(self, current_value):
    results = []
    has_coro = False
    for fn in self.fns:
      result = fn(current_value)
      results.append(result)
      if _iscoro(result):
        has_coro = True
    if has_coro:
      return _gather_to_async(results)
    return results


def _make_gather_link(fns):
  link = _create_link(None, _EMPTY_TUPLE, _EMPTY_DICT)
  return _create_link(_Gather(fns, link), None, None, False, link)


async def _gather_to_async(results):
  coros = []
  indices = []
  for i in range(len(results)):
    if _iscoro(results[i]):
      coros.append(results[i])
      indices.append(i)
  resolved = await asyncio.gather(*coros)
  for i in range(len(indices)):
    results[indices[i]] = resolved[i]
  return results


# ─── Chain Wrappers ──────────────────────────────────────────────────────────

class _ChainCallWrapper:
  __slots__ = ('_chain', '_fn')

  def __init__(self, chain, fn):
    self._chain = chain
    self._fn = fn

  def __call__(self, *args, **kwargs):
    try:
      return self._chain._run(self._fn, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None


class _DescriptorWrapper:
  __slots__ = ('_fn', '__dict__')

  def __init__(self, fn):
    self._fn = fn
    self.__dict__ = {}

  def __call__(self, *args, **kwargs):
    return self._fn(*args, **kwargs)

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self
    import types
    # TODO what is this?
    return types.MethodType(self, obj)


# ─── Diagnostics ─────────────────────────────────────────────────────────────

# TODO
#  these are artifacts of performance optimization; we no longer need this.
#  code should be as simple, elegant, and beautifully crafted as possible. readability
#  wins. simplicity wins.
_quent_file = os.path.abspath(__file__)
_TracebackType = types.TracebackType
_isroutine = inspect.isroutine
_isclass = inspect.isclass
_RAISE_CODE = compile('raise __exc__', '<quent>', 'exec')
_HAS_QUALNAME = sys.version_info >= (3, 11)


def _clean_internal_frames(tb):
  stack = []
  tb_next = None

  while tb is not None:
    filename = tb.tb_frame.f_code.co_filename
    if filename == '<quent>':
      stack.append(tb)
    elif filename != _quent_file:
      stack.append(tb)
    tb = tb.tb_next

  for tb in reversed(stack):
    new_tb = _TracebackType(tb_next, tb.tb_frame, tb.tb_lasti, tb.tb_lineno)
    tb_next = new_tb

  return tb_next


def _clean_chained_exceptions(exc, seen):
  if exc is None or id(exc) in seen:
    return
  seen.add(id(exc))
  if exc.__traceback__ is not None:
    exc.__traceback__ = _clean_internal_frames(exc.__traceback__)
  _clean_chained_exceptions(exc.__cause__, seen)
  _clean_chained_exceptions(exc.__context__, seen)


def _remove_self_frames():
  exc_value = sys.exc_info()[1]
  setattr(exc_value, '__quent__', True)
  tb = exc_value.__traceback__

  cleaned_tb = _clean_internal_frames(tb)

  seen = set()
  _clean_chained_exceptions(exc_value.__cause__, seen)
  _clean_chained_exceptions(exc_value.__context__, seen)

  return exc_value.with_traceback(cleaned_tb)


def _modify_traceback(exc, chain, link, ctx):
  if getattr(exc, '__quent_source_link__', None) is None:
    setattr(exc, '__quent_source_link__', link)

  if chain.is_nested:
    return
  source_link = getattr(exc, '__quent_source_link__')
  delattr(exc, '__quent_source_link__')
  filename = '<quent>'
  chain_source, _ = _stringify_chain(
    chain, ctx, nest_lvl=0,
    source_link=_get_true_source_link(source_link, ctx),
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


def _get_true_source_link(source_link, ctx):
  """Retrieves the first non-chain link."""
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


def _make_indent(nest_lvl):
  return '\n' + ' ' * 4 * nest_lvl


def _get_link_name(link):
  """Reconstruct a display name for a link from its type and flags."""
  vt = type(link.v)
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


def _stringify_chain(chain, ctx, nest_lvl=0, source_link=None, found_source_link=False):
  output = ''
  link = chain.root_link
  if link is None and ctx is not None:
    link = ctx.temp_root_link

  if nest_lvl > 0:
    output += _make_indent(nest_lvl)
  output += _get_obj_name(chain)
  if link is None:
    output += '()'
  else:
    output += _format_link(
      link, ctx, nest_lvl=nest_lvl,
      source_link=source_link, found_source_link=found_source_link,
    )
    if not found_source_link and link is source_link:
      found_source_link = True

  link = chain.first_link
  while link is not None:
    output += _make_indent(nest_lvl)
    output += _format_link(
      link, ctx, nest_lvl=nest_lvl,
      source_link=source_link, found_source_link=found_source_link,
      method_name=_get_link_name(link),
    )
    if not found_source_link and link is source_link:
      found_source_link = True
    link = link.next_link

  link = chain.on_except_link
  if link is not None:
    output += _make_indent(nest_lvl)
    output += _format_link(
      link, ctx, nest_lvl=nest_lvl,
      source_link=source_link, found_source_link=found_source_link,
      method_name='except_',
    )
    if not found_source_link and link is source_link:
      found_source_link = True

  link = chain.on_finally_link
  if link is not None:
    output += _make_indent(nest_lvl)
    output += _format_link(
      link, ctx, nest_lvl=nest_lvl,
      source_link=source_link, found_source_link=found_source_link,
      method_name='finally_',
    )
    if not found_source_link and link is source_link:
      found_source_link = True

  return output, found_source_link


def _format_args(args):
  if not args:
    return ''
  if args[0] is ...:
    return ', ...'
  return ', ' + ', '.join([_get_obj_name(a) for a in args])


def _format_kwargs(kwargs):
  if not kwargs:
    return ''
  return ', ' + ', '.join([f'{k}={_get_obj_name(v)}' for k, v in kwargs.items()])


def _format_link(link, ctx, nest_lvl, source_link=None, found_source_link=False, method_name=None):
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
    if ctx is not None and ctx.link_temp_args is not None and id(link) in ctx.link_temp_args:
      args = ctx.link_temp_args[id(link)]
      kwargs = {}
    elif link.temp_args:
      args = link.temp_args
      kwargs = {}

  if original_value is None:
    original_value = link.v
  if link.is_chain or isinstance(original_value, Chain):
    nested_ctx = _ExecCtx()
    nested_ctx.link_temp_args = None
    _temp_args = args or ()
    _temp_kwargs = kwargs or {}
    if _temp_args or _temp_kwargs:
      _temp_v = _temp_args[0] if _temp_args else Null
      if _temp_v is not Null:
        nested_ctx.temp_root_link = _create_link(_temp_v, _temp_args[1:], _temp_kwargs)
    args = kwargs = None
    link_v, found_source_link = _stringify_chain(
      original_value, nested_ctx, nest_lvl=nest_lvl + 1,
      source_link=source_link, found_source_link=found_source_link,
    )
    is_chain = True
  else:
    link_v = _get_obj_name(original_value)

  if method_name is not None:
    output += f'.{method_name}'
  if link.eval_code in {EVAL_CALL_WITHOUT_ARGS, EVAL_CALL_WITH_EXPLICIT_ARGS, EVAL_CALL_WITH_CURRENT_VALUE}:
    if is_chain:
      args_s = _format_args(args)
      kwargs_s = _format_kwargs(kwargs)
      chain_newline = ''
      if args_s or kwargs_s:
        chain_newline = _make_indent(nest_lvl + 1)
      link_v = f'({link_v}{chain_newline}{args_s}{kwargs_s}'
      if link_v.endswith(', '):
        link_v = link_v[:len(link_v) - 2]
      output += link_v + f'{_make_indent(nest_lvl)})'
    else:
      link_v = f'({link_v}{_format_args(args)}{_format_kwargs(kwargs)}'
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


def _get_obj_name(obj):
  if isinstance(obj, Chain):
    return type(obj).__name__
  try:
    name = getattr(obj, '__qualname__', None) or getattr(obj, '__name__', None)
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


# ─── Chain ───────────────────────────────────────────────────────────────────

class Chain:
  """Sequential pipeline where each operation receives the result of the previous one."""

  __slots__ = (
    'root_link', 'first_link', 'on_finally_link', 'on_except_link',
    'current_link', 'on_except_exceptions', 'is_nested',
  )

  def __init__(self, v=Null, *args, **kwargs):
    self._init(v, args, kwargs)

  def _init(self, root_value, args, kwargs):
    self.is_nested = False
    if root_value is not Null:
      self.root_link = _create_link(root_value, args, kwargs)
    else:
      self.root_link = None
    self.first_link = None
    self.current_link = None
    self.on_finally_link = None
    self.on_except_link = None
    self.on_except_exceptions = None

  def _then(self, link):
    if self.current_link is not None:
      self.current_link.next_link = link
      self.current_link = link
    elif self.first_link is not None:
      self.first_link.next_link = link
      self.current_link = link
    else:
      self.first_link = link
      if self.root_link is not None:
        self.root_link.next_link = link

  def _run(self, v, args, kwargs, invoked_by_parent_chain):
    if not invoked_by_parent_chain and self.is_nested:
      raise QuentException('You cannot directly run a nested chain.')

    link = self.root_link
    temp_root_link = None
    link_results = None
    ctx = None
    current_value = Null
    root_value = Null
    result = None
    has_root_value = link is not None
    is_root_value_override = v is not Null
    ignore_finally = False

    if is_root_value_override:
      if has_root_value:
        raise QuentException('Cannot override the root value of a Chain.')

    try:
      if is_root_value_override:
        has_root_value = True
        result = _eval_signal_value(v, args, kwargs)
        if _iscoro(result):
          temp_root_link = _make_temp_link(v, args, kwargs)
          temp_root_link.next_link = self.first_link
          link = temp_root_link
          ignore_finally = True
          ctx = _ExecCtx()
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = temp_root_link
          ctx.current_value = Null
          ctx.root_value = Null
          ctx.has_root_value = True
          return self._run_async(ctx, result)
        root_value = result
        current_value = result
        link = self.first_link
      elif has_root_value:
        result = _evaluate_value(link, Null)
        if _iscoro(result):
          ignore_finally = True
          ctx = _ExecCtx()
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = link
          ctx.current_value = Null
          ctx.root_value = Null
          ctx.has_root_value = has_root_value
          return self._run_async(ctx, result)
        root_value = result
        current_value = result
        link = link.next_link
      else:
        link = self.first_link

      while link is not None:
        result = _evaluate_value(link, current_value)
        if _iscoro(result):
          ignore_finally = True
          ctx = _ExecCtx()
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
          ctx.async_link = link
          ctx.current_value = current_value
          ctx.root_value = root_value
          ctx.has_root_value = has_root_value
          return self._run_async(ctx, result)
        if not link.ignore_result:
          current_value = result
        link = link.next_link

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      return _handle_return_exc(exc, self.is_nested)

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      if is_root_value_override and temp_root_link is None:
        temp_root_link = _make_temp_link(v, args, kwargs)
        temp_root_link.next_link = self.first_link
        link = temp_root_link
      ctx = _ExecCtx()
      ctx.temp_root_link = temp_root_link
      ctx.link_results = link_results
      ctx.link_temp_args = None
      exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
      if exc_temp_args is not None:
        ctx.link_temp_args = exc_temp_args
      _modify_traceback(exc, self, link, ctx)
      if self.on_except_link is None or not isinstance(exc, self.on_except_exceptions):
        raise exc
      try:
        result = _evaluate_value(self.on_except_link, exc)
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside except handlers is not allowed.')
      except BaseException as exc_:
        _modify_traceback(exc_, self, self.on_except_link, ctx)
        raise exc_ from exc
      if _iscoro(result):
        result = _ensure_future(_await_run(result, self, self.on_except_link, ctx))
        warnings.warn(
          'An except handler returned a coroutine from a synchronous execution path. '
          'It was scheduled as a fire-and-forget Task via ensure_future().',
          category=RuntimeWarning,
        )
        return result
      if result is Null:
        return None
      return result

    finally:
      if not ignore_finally and self.on_finally_link is not None:
        try:
          result = _evaluate_value(self.on_finally_link, root_value)
        except _InternalQuentException:
          raise QuentException('Using control flow signals inside finally handlers is not allowed.')
        except BaseException as exc_:
          if ctx is None:
            ctx = _ExecCtx()
            ctx.temp_root_link = temp_root_link
            ctx.link_results = link_results
            ctx.link_temp_args = None
          _modify_traceback(exc_, self, self.on_finally_link, ctx)
          raise exc_
        if _iscoro(result):
          if ctx is None:
            ctx = _ExecCtx()
            ctx.temp_root_link = temp_root_link
            ctx.link_results = link_results
            ctx.link_temp_args = None
          _ensure_future(_await_run(result, self, self.on_finally_link, ctx))
          warnings.warn(
            'A finally handler returned a coroutine from a synchronous execution path. '
            'It was scheduled as a fire-and-forget Task via ensure_future().',
            category=RuntimeWarning,
          )

  async def _run_async(self, ctx, awaitable):
    link = ctx.async_link
    current_value = ctx.current_value
    root_value = ctx.root_value
    has_root_value = ctx.has_root_value

    try:
      result = await awaitable
      if not link.ignore_result:
        current_value = result
      if has_root_value and root_value is Null:
        root_value = current_value

      link = link.next_link
      while link is not None:
        result = _evaluate_value(link, current_value)
        if _iscoro(result):
          result = await result
        if not link.ignore_result:
          current_value = result
        link = link.next_link

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      result = _handle_return_exc(exc, self.is_nested)
      if _iscoro(result):
        return await result
      return result

    except _Break:
      if self.is_nested:
        raise
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      ctx.link_temp_args = None
      exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
      if exc_temp_args is not None:
        ctx.link_temp_args = exc_temp_args
      _modify_traceback(exc, self, link, ctx)
      if self.on_except_link is None or not isinstance(exc, self.on_except_exceptions):
        raise exc
      try:
        result = _evaluate_value(self.on_except_link, exc)
        if _iscoro(result):
          result = await result
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside except handlers is not allowed.')
      except BaseException as exc_:
        _modify_traceback(exc_, self, self.on_except_link, ctx)
        raise exc_ from exc
      if result is Null:
        return None
      return result

    finally:
      if self.on_finally_link is not None:
        try:
          result = _evaluate_value(self.on_finally_link, root_value)
          if _iscoro(result):
            result = await result
        except _InternalQuentException:
          raise QuentException('Using control flow signals inside finally handlers is not allowed.')
        except BaseException as exc_:
          _modify_traceback(exc_, self, self.on_finally_link, ctx)
          raise exc_

  def decorator(self):
    chain = self
    def _decorator(fn):
      result = _DescriptorWrapper(_ChainCallWrapper(chain, fn))
      try:
        functools.update_wrapper(result, fn)
      except AttributeError:
        pass
      return result
    return _decorator

  def run(self, v=Null, *args, **kwargs):
    try:
      return self._run(v, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def freeze(self):
    return _FrozenChain(self)

  def then(self, v, *args, **kwargs):
    self._then(_create_link(v, args, kwargs))
    return self

  def do(self, fn, *args, **kwargs):
    self._then(_create_link(fn, args, kwargs, ignore_result=True))
    return self

  def except_(self, fn, *args, exceptions=None, **kwargs):
    if self.on_except_link is not None:
      raise QuentException('You can only register one \'except\' callback.')
    if exceptions is not None:
      if isinstance(exceptions, str):
        raise TypeError(f"except_() expects exception types, not string '{exceptions}'")
      if isinstance(exceptions, collections.abc.Iterable):
        self.on_except_exceptions = tuple(exceptions)
      else:
        self.on_except_exceptions = (exceptions,)
    else:
      self.on_except_exceptions = (Exception,)
    self.on_except_link = _create_link(fn, args, kwargs)
    return self

  def finally_(self, fn, *args, **kwargs):
    if self.on_finally_link is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally_link = _create_link(fn, args, kwargs)
    return self

  def iterate(self, fn=None):
    return _Generator(self._run, fn, _ignore_result=False)

  def iterate_do(self, fn=None):
    return _Generator(self._run, fn, _ignore_result=True)

  def foreach(self, fn):
    self._then(_make_foreach_link(fn, ignore_result=False))
    return self

  def foreach_do(self, fn):
    self._then(_make_foreach_link(fn, ignore_result=True))
    return self

  def filter(self, fn):
    self._then(_make_filter_link(fn))
    return self

  def gather(self, *fns):
    self._then(_make_gather_link(fns))
    return self

  def with_(self, fn, *args, **kwargs):
    self._then(_make_with_link(fn, args, kwargs, ignore_result=False))
    return self

  def with_do(self, fn, *args, **kwargs):
    self._then(_make_with_link(fn, args, kwargs, ignore_result=True))
    return self

  @classmethod
  def return_(cls, v=Null, *args, **kwargs):
    raise _Return(v, args, kwargs)

  @classmethod
  def break_(cls, v=Null, *args, **kwargs):
    raise _Break(v, args, kwargs)

  def __call__(self, v=Null, *args, **kwargs):
    try:
      return self._run(v, args, kwargs, False)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def __bool__(self):
    return True

  def __repr__(self):
    return _stringify_chain(self, None)[0]


# ─── FrozenChain ─────────────────────────────────────────────────────────────

class _FrozenChain:
  """Frozen chain with pre-compiled link tuple for optimized evaluation."""

  __slots__ = ('_chain', '_links', '_n_links', '_all_simple', '_has_finally', '_has_except')

  def __init__(self, chain):
    self._chain = chain
    links = []
    link = chain.first_link
    all_simple = True
    while link is not None:
      links.append(link)
      if link.eval_code != EVAL_CALL_WITH_CURRENT_VALUE or link.is_chain or link.ignore_result:
        all_simple = False
      link = link.next_link
    self._links = tuple(links)
    self._n_links = len(links)
    self._all_simple = all_simple
    self._has_finally = chain.on_finally_link is not None
    self._has_except = chain.on_except_link is not None

  def run(self, v=Null, *args, **kwargs):
    try:
      return _frozen_run(self, v, args, kwargs)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def __call__(self, v=Null, *args, **kwargs):
    try:
      return _frozen_run(self, v, args, kwargs)
    except _InternalQuentException as exc:
      raise QuentException(str(exc)) from None

  def __bool__(self):
    return True

  def __repr__(self):
    return f'<FrozenChain links={self._n_links} all_simple={self._all_simple}>'

  async def _run_async(self, ctx, awaitable, start_idx):
    chain = self._chain
    link = ctx.async_link
    current_value = ctx.current_value
    root_value = ctx.root_value
    has_root_value = ctx.has_root_value
    n = self._n_links

    try:
      result = await awaitable
      if not link.ignore_result:
        current_value = result
      if has_root_value and root_value is Null:
        root_value = current_value

      for i in range(start_idx + 1, n):
        link = self._links[i]
        result = _evaluate_value(link, current_value)
        if _iscoro(result):
          result = await result
        if not link.ignore_result:
          current_value = result

      if current_value is Null:
        return None
      return current_value

    except _Return as exc:
      result = _handle_return_exc(exc, False)
      if _iscoro(result):
        return await result
      return result

    except _Break:
      raise QuentException('_Break cannot be used in this context.')

    except BaseException as exc:
      ctx.link_temp_args = None
      exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
      if exc_temp_args is not None:
        ctx.link_temp_args = exc_temp_args
      _modify_traceback(exc, chain, link, ctx)
      if chain.on_except_link is None or not isinstance(exc, chain.on_except_exceptions):
        raise exc
      try:
        result = _evaluate_value(chain.on_except_link, exc)
        if _iscoro(result):
          result = await result
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside except handlers is not allowed.')
      except BaseException as exc_:
        _modify_traceback(exc_, chain, chain.on_except_link, ctx)
        raise exc_ from exc
      if result is Null:
        return None
      return result

    finally:
      if chain.on_finally_link is not None:
        try:
          result = _evaluate_value(chain.on_finally_link, root_value)
          if _iscoro(result):
            result = await result
        except _InternalQuentException:
          raise QuentException('Using control flow signals inside finally handlers is not allowed.')
        except BaseException as exc_:
          _modify_traceback(exc_, chain, chain.on_finally_link, ctx)
          raise exc_


def _frozen_run(fc, v, args, kwargs):
  chain = fc._chain
  link = None
  temp_root_link = None
  link_results = None
  ctx = None
  current_value = Null
  root_value = Null
  result = None
  has_root_value = chain.root_link is not None
  is_root_value_override = v is not Null
  ignore_finally = False
  n = fc._n_links

  if is_root_value_override:
    if has_root_value:
      raise QuentException('Cannot override the root value of a Chain.')

  try:
    if is_root_value_override:
      has_root_value = True
      result = _eval_signal_value(v, args, kwargs)
      if _iscoro(result):
        temp_root_link = _make_temp_link(v, args, kwargs)
        ignore_finally = True
        ctx = _ExecCtx()
        ctx.temp_root_link = temp_root_link
        ctx.link_results = link_results
        ctx.link_temp_args = None
        ctx.async_link = temp_root_link
        ctx.current_value = Null
        ctx.root_value = Null
        ctx.has_root_value = True
        return fc._run_async(ctx, result, -1)
      root_value = result
      current_value = result
    elif has_root_value:
      link = chain.root_link
      result = _evaluate_value(link, Null)
      if _iscoro(result):
        ignore_finally = True
        ctx = _ExecCtx()
        ctx.temp_root_link = temp_root_link
        ctx.link_results = link_results
        ctx.link_temp_args = None
        ctx.async_link = link
        ctx.current_value = Null
        ctx.root_value = Null
        ctx.has_root_value = has_root_value
        return fc._run_async(ctx, result, -1)
      root_value = result
      current_value = result

    for i in range(n):
      link = fc._links[i]
      result = _evaluate_value(link, current_value)
      if _iscoro(result):
        ignore_finally = True
        ctx = _ExecCtx()
        ctx.temp_root_link = temp_root_link
        ctx.link_results = link_results
        ctx.link_temp_args = None
        ctx.async_link = link
        ctx.current_value = current_value
        ctx.root_value = root_value
        ctx.has_root_value = has_root_value
        return fc._run_async(ctx, result, i)
      if not link.ignore_result:
        current_value = result

    if current_value is Null:
      return None
    return current_value

  except _Return as exc:
    return _handle_return_exc(exc, False)

  except _Break:
    raise QuentException('_Break cannot be used in this context.')

  except BaseException as exc:
    if is_root_value_override and temp_root_link is None:
      temp_root_link = _make_temp_link(v, args, kwargs)
      temp_root_link.next_link = chain.first_link
      link = temp_root_link
    ctx = _ExecCtx()
    ctx.temp_root_link = temp_root_link
    ctx.link_results = link_results
    ctx.link_temp_args = None
    exc_temp_args = getattr(exc, '__quent_link_temp_args__', None)
    if exc_temp_args is not None:
      ctx.link_temp_args = exc_temp_args
    _modify_traceback(exc, chain, link, ctx)
    if chain.on_except_link is None or not isinstance(exc, chain.on_except_exceptions):
      raise exc
    try:
      result = _evaluate_value(chain.on_except_link, exc)
    except _InternalQuentException:
      raise QuentException('Using control flow signals inside except handlers is not allowed.')
    except BaseException as exc_:
      _modify_traceback(exc_, chain, chain.on_except_link, ctx)
      raise exc_ from exc
    if _iscoro(result):
      result = _ensure_future(_await_run(result, chain, chain.on_except_link, ctx))
      warnings.warn(
        'An except handler returned a coroutine from a synchronous execution path. '
        'It was scheduled as a fire-and-forget Task via ensure_future().',
        category=RuntimeWarning,
      )
      return result
    if result is Null:
      return None
    return result

  finally:
    if not ignore_finally and chain.on_finally_link is not None:
      try:
        result = _evaluate_value(chain.on_finally_link, root_value)
      except _InternalQuentException:
        raise QuentException('Using control flow signals inside finally handlers is not allowed.')
      except BaseException as exc_:
        if ctx is None:
          ctx = _ExecCtx()
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
        _modify_traceback(exc_, chain, chain.on_finally_link, ctx)
        raise exc_
      if _iscoro(result):
        if ctx is None:
          ctx = _ExecCtx()
          ctx.temp_root_link = temp_root_link
          ctx.link_results = link_results
          ctx.link_temp_args = None
        _ensure_future(_await_run(result, chain, chain.on_finally_link, ctx))
        warnings.warn(
          'A finally handler returned a coroutine from a synchronous execution path. '
          'It was scheduled as a fire-and-forget Task via ensure_future().',
          category=RuntimeWarning,
        )


# ─── Excepthook ─────────────────────────────────────────────────────────────

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


# ─── TracebackException Patch ────────────────────────────────────────────────

_original_te_init = _traceback_module.TracebackException.__init__

def _patched_te_init(self, exc_type, exc_value=None, exc_tb=None, **kwargs):
  if exc_value is not None and getattr(exc_value, '__quent__', False):
    _clean_exc_chain(exc_value)
    exc_tb = exc_value.__traceback__
  _original_te_init(self, exc_type, exc_value, exc_tb, **kwargs)

_traceback_module.TracebackException.__init__ = _patched_te_init
