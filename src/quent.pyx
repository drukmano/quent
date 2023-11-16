# cython: binding=False, boundscheck=False, wraparound=False, language_level=3, embedsignature=True
## cython: profile=True, linetrace=True, warn.undeclared=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

cimport cython
import collections.abc
import functools
import sys
import types
import warnings


cdef object _ensure_future
from asyncio import ensure_future as _ensure_future


cdef object _PipeCls
try:
  from pipe import Pipe as _PipeCls
except ImportError:
  _PipeCls = None


cdef class _Null:
  def __repr__(self):
    return '<Null>'

cdef _Null Null = _Null()


cdef class QuentException(Exception):
  pass


cdef class Chain:
  cdef:
    Link root_link, on_finally
    list links, except_links
    bint is_cascade, _autorun
    tuple current_conditional, on_true
    str current_attr

  @classmethod
  def from_(cls, *args) -> Chain:
    return from_list(cls, args)

  def _populate_chain(self, root_link, is_cascade, _autorun, links, except_links, on_finally) -> Chain:
    # TODO find how to iterate the class attributes (like __slots__ / __dict__, but Cython classes does not implement
    #  those)
    self.root_link = root_link
    self.is_cascade = is_cascade
    self._autorun = _autorun
    self.links = links.copy()
    self.except_links = except_links.copy()
    self.on_finally = on_finally
    return self

  def __init__(self, __v=Null, *args, **kwargs):
    """
    Create a new Chain
    :param v: the root value of the chain
    :param args: arguments to pass to `v`
    :param kwargs: keyword-arguments to pass to `v`
    """
    self.init(__v, args, kwargs, is_cascade=False)

  cdef int init(self, object rv, tuple args, dict kwargs, bint is_cascade) except -1:
    self.is_cascade = is_cascade
    self._autorun = False
    self.links = []
    if rv is not Null:
      self.root_link = Link(rv, args=args, kwargs=kwargs)
    else:
      self.root_link = None

    self.except_links = []
    self.on_true = None
    self.on_finally = None
    self.current_conditional = None
    self.current_attr = None

  cdef int _then(self, Link link) except -1:
    self.finalize()
    if self.is_cascade:
      link.is_with_root = True
    self.links.append(link)

  cdef object _run(self, object v, tuple args, dict kwargs):
    self.finalize()
    cdef:
      # current chain value, root value
      object cv = Null, rv = Null, result, exc
      bint is_void = self.root_link is None, ignore_finally = False, is_null = v is Null
      list links = self.links
      int idx = -1
      Link root_link = self.root_link, link

    if not is_void and not is_null:
      raise QuentException('Cannot override the root value of a Chain.')
    elif is_void and is_null and not self.is_cascade:
      raise QuentException('Cannot run a Chain without a root value. Use Cascade for that.')

    try:
      # this condition is False only for a void Cascade.
      if not (is_void and is_null):
        if root_link is None:
          root_link = Link(v, args=args, kwargs=kwargs)
        rv = cv = evaluate_value(root_link, cv=Null)
        is_void = False
        if isawaitable(rv):
          ignore_finally = True
          result = self._run_async(root_link, result=rv, rv=Null, cv=Null, idx=idx)
          if self._autorun:
            return ensure_future(result)
          return result

      for link in links:
        idx += 1
        if link.is_attr and is_void:
          raise QuentException('Cannot use attributes without a root value.')

        # do not change the current value if .root_ignore() is called.
        if link.is_with_root and not link.ignore_result:
          cv = rv
        # `v is Null` is only possible when an empty `.root()` call has been made.
        if link.v is not Null:
          result = evaluate_value(link, cv=rv if link.is_with_root else cv)
          if isawaitable(result):
            ignore_finally = True
            result = self._run_async(link, result=result, rv=rv, cv=cv, idx=idx)
            if self._autorun:
              return ensure_future(result)
            return result
          if not link.ignore_result and result is not Null:
            cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as exc:
      _handle_exception(exc, self.except_links, link, rv, cv, idx)

    finally:
      if not ignore_finally and self.on_finally is not None:
        result = evaluate_value(self.on_finally, cv=rv)
        if isawaitable(result):
          ensure_future(result)
          warnings.warn(
            'The \'finally\' callback has returned a coroutine, but the chain is in synchronous mode. '
            'It was therefore scheduled for execution in a new Task.',
            category=RuntimeWarning
          )

  async def _run_async(self, Link link, object result, object rv, object cv, int idx):
    cdef:
      object exc
      bint is_void = self.root_link is None
      list links = self.links

    try:
      result = await result
      if not link.ignore_result and result is not Null:
        cv = result
      # update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `rv` should always be `Null`.
      if not is_void and rv is Null:
        rv = cv

      for idx in range(idx+1, len(links)):
        link = links[idx]
        if link.is_attr and is_void:
          raise QuentException('Cannot use attributes without a root value.')

        if link.is_with_root and not link.ignore_result:
          cv = rv
        if link.v is not Null:
          result = evaluate_value(link, cv=rv if link.is_with_root else cv)
          if isawaitable(result):
            result = await result
          if not link.ignore_result and result is not Null:
            cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as exc:
      await _handle_exception(exc, self.except_links, link, rv, cv, idx, async_=True)

    finally:
      if self.on_finally is not None:
        result = evaluate_value(self.on_finally, cv=rv)
        if isawaitable(result):
          await result

  def config(self, *, autorun=None) -> Chain:
    if autorun is not None:
      self._autorun = bool(autorun)
    return self

  def autorun(self, autorun=True) -> Chain:
    self._autorun = bool(autorun)
    return self

  def clone(self) -> Chain:
    self.finalize()
    return self.__class__()._populate_chain(
      self.root_link, self.is_cascade, self._autorun, self.links, self.except_links, self.on_finally
    )

  def freeze(self) -> FrozenChain:
    self.finalize()
    return FrozenChain(self._run)

  def decorator(self):
    return self.freeze().decorator()

  def run(self, __v=Null, *args, **kwargs):
    return self._run(__v, args, kwargs)

  def then(self, __v, *args, **kwargs) -> Chain:
    self._then(Link(__v, args=args, kwargs=kwargs))
    return self

  def do(self, __v, *args, **kwargs) -> Chain:
    # register a value to be evaluated but will not propagate its result forwards.
    self._then(Link(__v, args=args, kwargs=kwargs, ignore_result=True))
    return self

  def root(self, __v=Null, *args, **kwargs) -> Chain:
    self._then(Link(__v, args=args, kwargs=kwargs, is_with_root=True))
    return self

  def root_do(self, __v, *args, **kwargs) -> Chain:
    self._then(Link(__v, args=args, kwargs=kwargs, is_with_root=True, ignore_result=True))
    return self

  def attr(self, __v) -> Chain:
    self._then(Link(__v, is_attr=True))
    return self

  def attr_fn(self, __v, *args, **kwargs) -> Chain:
    self._then(Link(__v, args=args, kwargs=kwargs, is_fattr=True))
    return self

  def except_(self, __v, *args, exceptions=None, raise_=True, **kwargs) -> Chain:
    self.except_links.append((Link(__v, args=args, kwargs=kwargs), exceptions, raise_))
    return self

  def finally_(self, __v, *args, **kwargs) -> Chain:
    if self.on_finally is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally = Link(__v, args=args, kwargs=kwargs)
    return self

  def iterate(self, fn=None):
    return _Generator(self._run, fn, _ignore_result=False)

  def iterate_do(self, fn=None):
    return _Generator(self._run, fn, _ignore_result=True)

  def foreach(self, fn) -> Chain:
    self._then(foreach(fn, ignore_result=False))
    return self

  def foreach_do(self, fn) -> Chain:
    self._then(foreach(fn, ignore_result=True))
    return self

  def with_(self, __v=Null, *args, **kwargs) -> Chain:
    self._then(with_(Link(__v, args=args, kwargs=kwargs), ignore_result=False))
    return self

  def with_do(self, __v, *args, **kwargs) -> Chain:
    self._then(with_(Link(__v, args=args, kwargs=kwargs), ignore_result=True))
    return self

  def if_(self, on_true, *args, **kwargs) -> Chain:
    self._if(on_true, args, kwargs)
    return self

  def else_(self, on_false, *args, **kwargs) -> Chain:
    self._else(on_false, args, kwargs)
    return self

  def if_not(self, on_true, *args, **kwargs) -> Chain:
    self._if(on_true, args, kwargs, not_=True)
    return self

  def if_raise(self, exc) -> Chain:
    def if_raise(object cv): raise exc
    self._if(if_raise)
    return self

  def else_raise(self, exc) -> Chain:
    def else_raise(object cv): raise exc
    self._else(else_raise)
    return self

  def if_not_raise(self, exc) -> Chain:
    def if_not_raise(object cv): raise exc
    self._if(if_not_raise, None, None, not_=True)
    return self

  def condition(self, __v, *args, **kwargs) -> Chain:
    cdef Link link = Link(__v, args=args, kwargs=kwargs)
    def condition(object cv):
      return evaluate_value(link, cv=cv)
    self.set_conditional(condition, custom=True)
    return self

  def not_(self) -> Chain:
    # use named functions (instead of a lambda) to have more details in the exception stacktrace
    def not_(object cv) -> bool: return not cv
    self.set_conditional(not_)
    return self

  def eq(self, value) -> Chain:
    def equals(object cv) -> bool: return cv == value
    self.set_conditional(equals)
    return self

  def neq(self, value) -> Chain:
    def not_equals(object cv) -> bool: return cv != value
    self.set_conditional(not_equals)
    return self

  def is_(self, value) -> Chain:
    def is_(object cv) -> bool: return cv is value
    self.set_conditional(is_)
    return self

  def is_not(self, value) -> Chain:
    def is_not(object cv) -> bool: return cv is not value
    self.set_conditional(is_not)
    return self

  def in_(self, value) -> Chain:
    def in_(object cv) -> bool: return cv in value
    self.set_conditional(in_)
    return self

  def not_in(self, value) -> Chain:
    def not_in(object cv) -> bool: return cv not in value
    self.set_conditional(not_in)
    return self

  def or_(self, value) -> Chain:
    def or_(object cv): return cv or value
    self._then(Link(or_, eval_code=EVAL_CALLABLE))
    return self

  def raise_(self, exc) -> Chain:
    def raise_(object cv): raise exc
    self._then(Link(raise_, eval_code=EVAL_CALLABLE))
    return self

  cdef int _if(self, object on_true, tuple args = None, dict kwargs = None, bint not_ = False) except -1:
    if self.current_conditional is None:
      self.current_conditional = (bool, False)
    self.on_true = (Link(on_true, args=args, kwargs=kwargs), not_)

  cdef int _else(self, object on_false, tuple args = None, dict kwargs = None) except -1:
    if self.on_true is None:
      raise QuentException(
        'You cannot use \'.else_()\' without a preceding \'.if_()\' or \'.if_not()\''
      )
    self.finalize_conditional(on_false, args, kwargs)

  cdef int set_conditional(self, object conditional, bint custom = False) except -1:
    self.finalize()
    self.current_conditional = (conditional, custom)

  cdef int finalize(self) except -1:
    cdef str attr = self.current_attr
    if attr is not None:
      self.current_attr = None
      self._then(Link(attr, is_attr=True, eval_code=EVAL_ATTR))
    if self.current_conditional is not None:
      self.finalize_conditional()

  cdef int finalize_conditional(self, object on_false = Null, tuple args = None, dict kwargs = None) except -1:
    cdef:
      object conditional
      Link on_true_link, on_false_link = None
      bint is_custom, not_
    conditional, is_custom = self.current_conditional
    self.current_conditional = None
    if self.on_true:
      on_true_link, not_ = self.on_true
      if on_false is not Null:
        on_false_link = Link(on_false, args=args, kwargs=kwargs)
      self.on_true = None
      self._then(build_conditional(conditional, is_custom, not_, on_true_link, on_false_link))
    else:
      self._then(Link(conditional, eval_code=EVAL_CALLABLE))

  def __or__(self, other) -> Chain:
    if isinstance(other, run):
      return self._run(other.root_value, other.args, other.kwargs)
    self._then(Link(other))
    return self

  def __call__(self, __v=Null, *args, **kwargs):
    return self._run(__v, args, kwargs)

  # while this may be nice to have, I fear that it will cause troubles as
  # people might forget to call `.run()` when dealing with non-async code (or
  # code that could be both but is not known to the one creating the chain).
  #def __await__(self):
  #  return self._run(Null, None, None).__await__()

  def __bool__(self):
    return True

  def __repr__(self):
    cdef:
      Link root_link = self.root_link
      object s = f'<{self.__class__.__name__}'

    if root_link is not None:
      s += f'({root_link.v}, {root_link.args}, {root_link.kwargs})'
    else:
      s += '()'
    s += f'({len(self.links)} links)>'
    return s


cdef class Cascade(Chain):
  # TODO mark all cascade items as `ignore_result=True` to (marginally) increase performance.

  # noinspection PyMissingConstructor
  def __init__(self, __v=Null, *args, **kwargs):
    self.init(__v, args, kwargs, is_cascade=True)


cdef class ChainAttr(Chain):
  def __getattr__(self, attr) -> ChainAttr:
    self.finalize()
    self.current_attr = attr
    return self

  def __call__(self, *args, **kwargs) -> ChainAttr:
    cdef str attr = self.current_attr
    if attr is None:
      # much slower than directly calling `._run()`, but we have no choice since
      # we wish support arbitrary __call__ invocations on attributes.
      # avoid running a chain this way. opt to use `.run()` instead.
      return self.run(*args, **kwargs)
    else:
      self.current_attr = None
      self._then(Link(attr, args=args, kwargs=kwargs, is_fattr=True))
      return self


# cannot have multiple inheritance in Cython.
cdef class CascadeAttr(ChainAttr):
  # noinspection PyMissingConstructor
  def __init__(self, __v=Null, *args, **kwargs):
    self.init(__v, args, kwargs, is_cascade=True)


###############
### HELPERS ###
###############


### CLASSES ###
@cython.freelist(256)
cdef class Link:
  cdef object v
  cdef tuple args
  cdef dict kwargs
  cdef bint is_attr, is_fattr, is_with_root, ignore_result
  cdef int eval_code

  def __init__(
    self, v, args=None, kwargs=None, is_attr=False, is_fattr=False, is_with_root=False, ignore_result=False,
    eval_code=EVAL_UNKNOWN
  ):
    if _PipeCls is not None and isinstance(v, _PipeCls):
      v = v.function
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.is_attr = is_attr or is_fattr
    self.is_fattr = is_fattr
    self.is_with_root = is_with_root
    self.ignore_result = ignore_result
    if eval_code == EVAL_UNKNOWN:
      self.eval_code = get_eval_code(self)
    else:
      self.eval_code = eval_code


cdef class FrozenChain:
  cdef object _chain_run

  def decorator(self):
    cdef object _chain_run = self._chain_run
    def _decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        return _chain_run(fn, args, kwargs)
      return wrapper
    return _decorator

  def __init__(self, _chain_run):
    self._chain_run = _chain_run

  def run(self, __v=Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs)

  def __call__(self, __v=Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs)


### EVALUATION ###
cdef int EVAL_UNKNOWN = 0
cdef int EVAL_NULL = 1001
cdef int EVAL_CUSTOM_ARGS = 1002
cdef int EVAL_NO_ARGS = 1003
cdef int EVAL_CALLABLE = 1004
cdef int EVAL_LITERAL = 1005
cdef int EVAL_ATTR = 1006


cdef int get_eval_code(Link link) except -1:
  cdef object v = link.v
  if v is Null:
    return EVAL_NULL

  if isinstance(v, Chain):
    # TODO add `autorun_explicit` where if True then do not override this value here
    #  this allows for more granular control over nested chains and autorun policies
    #  (e.g. not wanting some nested chain to auto-run but also wanting to auto-run another nested chain)
    v.autorun(False)

  elif link.is_attr:
    if not link.is_fattr:
      return EVAL_ATTR

  # Ellipsis as the first argument indicates a void method.
  if link.args and link.args[0] is ...:
    return EVAL_NO_ARGS

  # if either are specified, we assume `v` is a function.
  elif link.args or link.kwargs:
    return EVAL_CUSTOM_ARGS

  elif link.is_fattr:
    return EVAL_NO_ARGS

  elif not link.is_attr and callable(v):
    return EVAL_CALLABLE

  else:
    return EVAL_LITERAL


cdef object evaluate_value(Link link, object cv):
  cdef object v = link.v
  cdef int eval_code = link.eval_code
  if eval_code == EVAL_UNKNOWN:
    link.eval_code = eval_code = get_eval_code(link)

  if eval_code == EVAL_NULL:
    return Null

  elif link.is_attr:
    v = getattr(cv, v)
    if not link.is_fattr:
      return v

  if eval_code == EVAL_NO_ARGS:
    return v()

  elif eval_code == EVAL_CUSTOM_ARGS:
    # it is dangerous if one of those will be `None`, but it shouldn't be possible
    # as we only specify both or none.
    return v(*link.args, **link.kwargs)

  elif link.is_fattr:
    return v()

  elif eval_code == EVAL_CALLABLE:
    # `cv is Null` is for safety; in most cases, it simply means that `v` is the root value.
    return v() if cv is Null else v(cv)

  else:
    return v


### CONDITIONALS ###
cdef Link build_conditional(object conditional, bint is_custom, bint not_, Link on_true, Link on_false):
  async def if_else_async(object r, object cv):
    r = _if_else(await r, cv)
    if isawaitable(r):
      r = await r
    return r

  def if_else(object cv):
    cdef object r = conditional(cv)
    if is_custom and isawaitable(r):
      return if_else_async(r, cv)
    return _if_else(r, cv)

  def _if_else(object r, object cv):
    if not_:
      r = not r
    if r:
      return evaluate_value(on_true, cv=cv)
    elif on_false is not None:
      return evaluate_value(on_false, cv=cv)
    return cv

  return Link(if_else, eval_code=EVAL_CALLABLE)


### ITERATORS ###
def sync_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result
  for el in iterator_getter(*run_args):
    if fn is None:
      yield el
    else:
      result = fn(el)
      # we ignore the case where `result` is awaitable - it's impossible to deal with.
      yield el if ignore_result else result


async def async_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result, iterator
  iterator = iterator_getter(*run_args)
  if isawaitable(iterator):
    iterator = await iterator
  if hasattr(iterator, '__aiter__'):
    async for el in iterator:
      if fn is None:
        result = el
      else:
        result = fn(el)
      if isawaitable(result):
        result = await result
      yield el if ignore_result else result
  else:
    for el in iterator:
      if fn is None:
        result = el
      else:
        result = fn(el)
      if isawaitable(result):
        result = await result
      yield el if ignore_result else result


cdef class _Generator:
  cdef object _chain_run, _fn
  cdef bint _ignore_result
  cdef tuple _run_args

  def __init__(self, _chain_run, _fn, _ignore_result):
    self._chain_run = _chain_run
    self._fn = _fn
    self._ignore_result = _ignore_result
    self._run_args = (Null, (), {})

  def __call__(self, __v=Null, *args, **kwargs):
    # this allows nesting of _Generator within another Chain
    self._run_args = (__v, args, kwargs)
    return self

  def __iter__(self):
    return sync_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __aiter__(self):
    return async_generator(self._chain_run, self._run_args, self._fn, self._ignore_result)


cdef Link foreach(object fn, bint ignore_result):
  def _foreach(object cv):
    if hasattr(cv, '__aiter__'):
      return async_gen_foreach(cv, fn, ignore_result)
    cdef list lst = []
    cdef object el, result
    cv = cv.__iter__()
    for el in cv:
      result = fn(el)
      if isawaitable(result):
        return async_foreach(cv, fn, el, result, lst, ignore_result)
      lst.append(el if ignore_result else result)
    return lst
  return Link(_foreach, eval_code=EVAL_CALLABLE)


async def async_foreach(object cv, object fn, object el, object result, list lst, bint ignore_result):
  result = await result
  lst.append(el if ignore_result else result)
  for el in cv:
    result = fn(el)
    if isawaitable(result):
      result = await result
    lst.append(el if ignore_result else result)
  return lst


async def async_gen_foreach(object cv, object fn, bint ignore_result):
  cdef list lst = []
  cdef object el, result
  async for el in cv.__aiter__():
    result = fn(el)
    if isawaitable(result):
      result = await result
    lst.append(el if ignore_result else result)
  return lst


### CONTEXT MANAGERS ###
cdef Link with_(Link link, bint ignore_result):
  async def with_async(object result, object cv):
    try:
      return await result
    finally:
      cv.__exit__(*sys.exc_info())
  def with_(object cv):
    if hasattr(cv, '__aenter__'):
      return async_with(link, cv=cv)
    cdef object ctx, result = None
    try:
      ctx = cv.__enter__()
      if link.v is Null:
        result = ctx
      else:
        result = evaluate_value(link, cv=ctx)
    finally:
      if isawaitable(result):
        return with_async(result, cv)
      cv.__exit__(*sys.exc_info())
      return result
  return Link(with_, ignore_result=ignore_result, eval_code=EVAL_CALLABLE)


async def async_with(Link link, object cv):
  cdef object ctx, result
  async with cv as ctx:
    if link.v is Null:
      return ctx
    result = evaluate_value(link, cv=ctx)
    if isawaitable(result):
      return await result
    return result


### ASYNCIO ###
# this holds a strong reference to all tasks that we create
# see: https://stackoverflow.com/a/75941086
# "... the asyncio loop avoids creating hard references (just weak) to the tasks,
# and when it is under heavy load, it may just "drop" tasks that are not referenced somewhere else."
cdef set task_registry = set()

cdef int remove_task(object task) except -1:
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


# a modified version of `asyncio.iscoroutine`.
cdef:
  # `abc.Awaitable` might catch most if not all awaitable objects, but checking
  # for `types.CoroutineType` is about x2 faster and also catches most coroutines.
  tuple _AWAITABLE_TYPES = (types.CoroutineType, collections.abc.Awaitable)
  set _isawaitable_typecache = set()
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


### MISC. ###
cdef object _handle_exception(
  object exc, list except_links, Link link, object rv, object cv, int idx, bint async_ = False
):
  cdef object quent_exc = create_chain_link_exception(link, cv, idx), exceptions, e, result
  cdef bint reraise = True, raise_, exc_match
  if exc.__cause__ is not None:
    if quent_exc.__cause__ is None:
      quent_exc.__cause__ = exc.__cause__
    else:
      quent_exc.__cause__.__cause__ = exc.__cause__
  exc.__cause__ = quent_exc
  if async_:
    return _handle_exception_async(exc, except_links, rv, quent_exc)
  try:
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
        result = evaluate_value(link, cv=rv)
        if isawaitable(result):
          ensure_future(result)
          warnings.warn(
            'An \'except\' callback has returned a coroutine, but the chain is in synchronous mode. '
            'It was therefore scheduled for execution in a new Task.',
            category=RuntimeWarning
          )
  except Exception as e:
    e.__cause__ = exc
    raise e
  finally:
    if reraise:
      raise exc


async def _handle_exception_async(object exc, list except_links, object rv, object quent_exc):
  cdef bint reraise = True, raise_, exc_match
  cdef object exceptions, e, link, result
  try:
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
        result = evaluate_value(link, cv=rv)
        if isawaitable(result):
          await result
  except Exception as e:
    e.__cause__ = exc
    raise e
  finally:
    if reraise:
      raise exc


cdef class run:
  """
    A replacement for `Chain.run()` when using pipe syntax

      Chain(f1).then(f2).run() == Chain(f1) | f2 | run()
      Chain().then(f2).run(f1) == Chain() | f2 | run(f1)
  """
  cdef public:
    object root_value
    tuple args
    dict kwargs

  def __init__(self, __v=Null, *args, **kwargs):
    self.root_value = __v
    self.args = args
    self.kwargs = kwargs


# TODO how to declare `Type[Chain] cls` ?
cdef Chain from_list(object cls, tuple links):
  cdef object el
  cdef Chain seq = cls()
  for el in links:
    seq._then(el)
  return seq


def create_chain_link_exception(Link link, object cv, int idx):
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
