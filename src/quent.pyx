# cython: binding=False, boundscheck=False, wraparound=False, language_level=3, embedsignature=True
## cython: profile=True, linetrace=True, warn.undeclared=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

#cimport cython
import collections.abc
import functools
import logging
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


cdef object evaluate_value(object v, object cv, bint is_attr, bint is_fattr, tuple args, dict kwargs):
  """ The main value evaluation function
  
  Given a value `v`, and a value `cv`, evaluates `v`.
  :param v: a value.
  :param cv: the current chain value.
  :param is_attr: whether `v` is an attribute of `cv`.
  :param is_fattr: whether `v` is a method attribute of `cv`.
  :param args: the arguments to pass to `v`.
  :param kwargs: the keyword-arguments to pass to `v`.
  :return: `Null`, `v`, or the evaluation of `v`.
  """
  if v is Null:
    return Null

  if isinstance(v, Chain):
    # TODO add `autorun_explicit` where if True then do not override this value here
    #  this allows for more granular control over nested chains and autorun policies
    #  (e.g. not wanting some nested chain to auto-run but also wanting to auto-run another nested chain)
    v.autorun(False)

  elif is_attr:
    v = getattr(cv, v)
    if not is_fattr:
      return v

  # Ellipsis as the first argument indicates a void method.
  if args and args[0] is ...:
    return v()

  # if either are specified, we assume `v` is a function.
  elif args or kwargs:
    # it is dangerous if one of those will be `None`, but it shouldn't be possible
    # as we only specify both or none.
    return v(*args, **kwargs)

  elif is_fattr:
    return v()

  elif not is_attr and callable(v):
    # `cv is Null` is for safety; in most cases, it simply means that `v` is the root value.
    return v() if cv is Null else v(cv)

  else:
    return v


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
    self.init(__v, args, kwargs)

  cdef int init(self, object v, tuple args, dict kwargs) except -1:
    self.root_value = v
    self.args = args
    self.kwargs = kwargs


# TODO how to declare `Type[Chain] cls` ?
cdef Chain from_list(object cls, tuple links):
  cdef object el
  cdef Chain seq = cls()
  for el in links:
    seq._then(el)
  return seq


cdef class Chain:
  cdef:
    tuple root_link
    bint is_cascade, _autorun, raise_on_exception
    list links
    tuple on_except, on_finally, current_conditional, current_on_true
    str current_attr

  @classmethod
  def from_(cls, *args) -> Chain:
    return from_list(cls, args)

  def _populate_chain(self, root_link, is_cascade, _autorun, raise_on_exception, links, on_except, on_finally) -> Chain:
    # TODO find how to iterate the class attributes (like __slots__ / __dict__, but Cython classes does not implement
    #  those)
    self.root_link = root_link
    self.is_cascade = is_cascade
    self._autorun = _autorun
    self.raise_on_exception = raise_on_exception
    self.links = links.copy()
    self.on_except = on_except
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

  cdef int init(self, object root_value, tuple args, dict kwargs, bint is_cascade) except -1:
    self.is_cascade = is_cascade
    self._autorun = False
    self.raise_on_exception = True

    # `links` is a list which contains tuples of the following structure:
    # (
    #   object, # the link value, 'v'
    #   bool,   # whether 'v' is attribute
    #   bool,   # whether 'v' is method attribute
    #   bool,   # whether to evaluate 'v' with the [evaluated] root value
    #   bool,   # whether to ignore this link evaluation result
    #   tuple,  # the arguments to evaluate 'v' with
    #   dict    # the keyword-arguments to evaluate 'v' with
    # )
    self.links = []

    if root_value is not Null:
      self.root_link = (root_value, False, False, False, False, args, kwargs)
    else:
      self.root_link = None

    self.on_except = None
    self.on_finally = None
    self.current_conditional = None
    self.current_on_true = None
    self.current_attr = None

  # https://github.com/cython/cython/issues/1630
  cdef int _then(
    self, object v, bint is_attr = False, bint is_fattr = False, bint is_with_root = False, bint ignore_result = False,
    tuple args = None, dict kwargs = None
  ) except -1:
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_conditional is not None:
      self.finalize_conditional()

    is_with_root = is_with_root or self.is_cascade
    if _PipeCls is not None and isinstance(v, _PipeCls):
      v = v.function
    self.links.append((v, is_attr, is_fattr, is_with_root, ignore_result, args, kwargs))

  cdef object _run(self, object v, tuple args, dict kwargs):
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_conditional is not None:
      self.finalize_conditional()

    cdef:
      # current chain value, root value
      object cv = Null, rv = Null, result, exc, quent_exc
      bint is_attr = False, is_fattr = False, is_with_root = False, ignore_result = False
      bint is_void = self.root_link is None, ignore_finally = False, is_null = v is Null
      bint raise_on_exception = self.raise_on_exception
      list links = self.links
      tuple link, on_except = self.on_except, on_finally = self.on_finally, root_link = self.root_link
      tuple exceptions = (Exception, )
      int idx = -1

    if on_except is not None:
      exceptions = tuple(on_except[3] or []) or (Exception, )

    if not is_void and not is_null:
      raise QuentException('Cannot override the root value of a Chain.')
    elif is_void and is_null and not self.is_cascade:
      raise QuentException('Cannot run a Chain without a root value. Use Cascade for that.')

    try:
      # this condition is False only for a void Cascade.
      if not (is_void and is_null):
        if root_link is not None:
          v, is_attr, is_fattr, is_with_root, ignore_result, args, kwargs = root_link
        rv = cv = evaluate_value(
          v=v, cv=Null, is_attr=False, is_fattr=False, args=args, kwargs=kwargs
        )
        is_void = False
        if isawaitable(rv):
          ignore_finally = True
          result = self._run_async(rv, Null, idx, is_void, exceptions, v, Null, False, False, False, args, kwargs)
          if self._autorun:
            return ensure_future(result)
          return result

      for idx, link in enumerate(links):
        # TODO optimize this line
        v, is_attr, is_fattr, is_with_root, ignore_result, args, kwargs = link
        if is_attr and is_void:
          raise QuentException('Cannot use attributes without a root value.')

        # do not change the current value if .root_ignore() is called.
        if is_with_root and not ignore_result:
          cv = rv
        # `v is Null` is only possible when an empty `.root()` call has been made.
        if v is not Null:
          result = evaluate_value(v, rv if is_with_root else cv, is_attr, is_fattr, args, kwargs)
          if isawaitable(result):
            ignore_finally = True
            result = self._run_async(result, rv, idx, is_void, exceptions, v, cv, is_attr, is_fattr, ignore_result, args, kwargs)
            if self._autorun:
              return ensure_future(result)
            return result
          if not ignore_result and result is not Null:
            cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as exc:
      quent_exc = QuentException
      try:
        quent_exc = create_chain_link_exception(v, cv, is_attr, is_fattr, args, kwargs, idx)
        if not raise_on_exception:
          # TODO in this case, how can we pass `exc` to on_except callback?
          logging.exception(str(quent_exc))
      finally:
        try:
          if on_except is not None:
            raise exc
        except exceptions:
          on_except = tuple(list(on_except)[:3])
          result = run_callback(on_except, rv)
          if isawaitable(result):
            ensure_future(result)
            warnings.warn(
              'The \'except\' callback has returned a coroutine, but the chain is in synchronous mode. '
              'It was therefore scheduled for execution in a new Task.',
              category=RuntimeWarning
            )
        except Exception:
          if not raise_on_exception:
            raise exc from quent_exc
        finally:
          if raise_on_exception:
            raise exc from quent_exc

    finally:
      if not ignore_finally and on_finally is not None:
        result = run_callback(on_finally, rv)
        if isawaitable(result):
          ensure_future(result)
          warnings.warn(
            'The \'finally\' callback has returned a coroutine, but the chain is in synchronous mode. '
            'It was therefore scheduled for execution in a new Task.',
            category=RuntimeWarning
          )

  async def _run_async(
    self, object result, object rv, int idx, bint is_void, tuple exceptions,
    object v, object cv, bint is_attr, bint is_fattr, bint ignore_result, tuple args, dict kwargs
  ):
    # we pass the full current link data to be able to format an appropriate
    # exception message in case one is raised from awaiting `result`
    cdef:
      object exc, quent_exc
      bint is_with_root = False, is_exc_raised_on_await = False
      bint raise_on_exception = self.raise_on_exception
      tuple on_except = self.on_except, on_finally = self.on_finally
      list links = self.links

    try:
      try:
        result = await result
      except:
        is_exc_raised_on_await = True
        raise
      if not ignore_result and result is not Null:
        cv = result
      # update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `rv` should always be `Null`.
      if not is_void and rv is Null:
        rv = cv

      for idx in range(idx+1, len(links)):
        v, is_attr, is_fattr, is_with_root, ignore_result, args, kwargs = links[idx]
        if is_attr and is_void:
          raise QuentException('Cannot use attributes without a root value.')

        if is_with_root and not ignore_result:
          cv = rv
        if v is not Null:
          result = evaluate_value(v, rv if is_with_root else cv, is_attr, is_fattr, args, kwargs)
          if isawaitable(result):
            try:
              result = await result
            except:
              is_exc_raised_on_await = True
              raise
          if not ignore_result and result is not Null:
            cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as exc:
      quent_exc = QuentException
      try:
        quent_exc = create_chain_link_exception(v, cv, is_attr, is_fattr, args, kwargs, idx, is_exc_raised_on_await)
        if not raise_on_exception:
          logging.exception(str(quent_exc))
      finally:
        # even though it seems that a coroutine that has raised an exception still
        # returns True for `isawaitable(result)`, I don't want to use this method to check
        # whether an exception was raised from awaiting `result` as I couldn't find if
        # this is an intended behavior or not.
        try:
          if on_except is not None:
            raise exc
        except exceptions:
          on_except = tuple(list(on_except)[:3])
          result = run_callback(on_except, rv)
          if isawaitable(result):
            await result
        except Exception:
          if not raise_on_exception:
            raise exc from quent_exc
        finally:
          if raise_on_exception:
            raise exc from quent_exc

    finally:
      if on_finally is not None:
        result = run_callback(on_finally, rv)
        if isawaitable(result):
          await result

  cdef int set_except(self, object fn_or_attr, tuple args, dict kwargs, object exceptions, bint raise_ = True):
    if self.on_except is not None:
      raise QuentException('You can only register one \'except\' callback.')
    self.on_except = fn_or_attr, args, kwargs, exceptions
    self.raise_on_exception = raise_

  cdef int set_finally(self, object fn_or_attr, tuple args, dict kwargs):
    if self.on_finally is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally = fn_or_attr, args, kwargs

  cdef int _if(
    self, object on_true_v, tuple true_args = None, dict true_kwargs = None, bint not_ = False
  ) except -1:
    if self.current_conditional is None:
      self.current_conditional = (bool, False)
    self.current_on_true = (on_true_v, true_args, true_kwargs, not_)

  cdef int _else(self, object on_false_v, tuple false_args = None, dict false_kwargs = None) except -1:
    if self.current_on_true is None:
      raise QuentException(
        'You cannot use \'.else_()\' without a preceding \'.if_()\' or \'.if_not()\''
      )
    self.finalize_conditional(on_false_v, false_args, false_kwargs)

  def config(self, *, autorun=None) -> Chain:
    if autorun is not None:
      self._autorun = bool(autorun)
    return self

  def autorun(self, autorun=True) -> Chain:
    self._autorun = bool(autorun)
    return self

  def clone(self) -> Chain:
    return self.__class__()._populate_chain(
      self.root_link, self.is_cascade, self._autorun, self.raise_on_exception, self.links, self.on_except,
      self.on_finally
    )

  def freeze(self) -> FrozenChain:
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_conditional is not None:
      self.finalize_conditional()
    return FrozenChain(self._run)

  @property
  def decorator(self):
    return self.freeze().decorator

  def run(self, __v=Null, *args, **kwargs):
    return self._run(__v, args, kwargs)

  # cannot use cpdef with *args **kwargs.
  def then(self, __v, *args, **kwargs) -> Chain:
    self._then(__v, is_attr=False, is_fattr=False, is_with_root=False, ignore_result=False, args=args, kwargs=kwargs)
    return self

  def do(self, __v, *args, **kwargs) -> Chain:
    # register a value to be evaluated but will not propagate its result forwards.
    self._then(__v, is_attr=False, is_fattr=False, is_with_root=False, ignore_result=True, args=args, kwargs=kwargs)
    return self

  def root(self, __v=Null, *args, **kwargs) -> Chain:
    self._then(__v, is_attr=False, is_fattr=False, is_with_root=True, ignore_result=False, args=args, kwargs=kwargs)
    return self

  def root_do(self, __v, *args, **kwargs) -> Chain:
    self._then(__v, is_attr=False, is_fattr=False, is_with_root=True, ignore_result=True, args=args, kwargs=kwargs)
    return self

  def attr(self, name) -> Chain:
    self._then(name, is_attr=True)
    return self

  def attr_fn(self, name, *args, **kwargs) -> Chain:
    self._then(name, is_attr=True, is_fattr=True, is_with_root=False, ignore_result=False, args=args, kwargs=kwargs)
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
    self._then(with_(__v, args, kwargs))
    return self

  def with_do(self, __v, *args, **kwargs) -> Chain:
    self._then(
      with_(__v, args, kwargs), is_attr=False, is_fattr=False, is_with_root=False, ignore_result=True, args=None,
      kwargs=None
    )
    return self

  def except_(self, fn_or_attr, *args, exceptions=None, **kwargs) -> Chain:
    self.set_except(fn_or_attr, args, kwargs, exceptions)
    return self

  def except_do(self, fn_or_attr, *args, exceptions=None, **kwargs) -> Chain:
    self.set_except(fn_or_attr, args, kwargs, exceptions, raise_=False)
    return self

  def finally_(self, fn_or_attr, *args, **kwargs) -> Chain:
    self.set_finally(fn_or_attr, args, kwargs)
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
    def condition(object cv):
      return evaluate_value(v=__v, cv=cv, is_attr=False, is_fattr=False, args=args, kwargs=kwargs)
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
    self._then(or_)
    return self

  def raise_(self, exc) -> Chain:
    def raise_(object cv): raise exc
    self._then(raise_)
    return self

  cdef int finalize_attr(self) except -1:
    cdef str attr = self.current_attr
    self.current_attr = None
    self._then(attr, is_attr=True)

  cdef int finalize_conditional(
    self, object on_false_v = Null, tuple false_args = None, dict false_kwargs = None
  ) except -1:
    cdef:
      object conditional, on_true_v
      tuple true_args
      dict true_kwargs
      bint is_custom, not_
    conditional, is_custom = self.current_conditional
    self.current_conditional = None
    if self.current_on_true:
      on_true_v, true_args, true_kwargs, not_ = self.current_on_true
      self.current_on_true = None
      self._then(
        build_conditional(
          conditional, is_custom, not_, on_true_v, true_args, true_kwargs, on_false_v, false_args, false_kwargs
        )
      )
    else:
      self._then(conditional)

  cdef int set_conditional(self, object conditional, bint custom = False) except -1:
    if self.current_conditional is not None:
      self.finalize_conditional()
    self.current_conditional = (conditional, custom)

  def __or__(self, other) -> Chain:
    if isinstance(other, run):
      return self._run(other.root_value, other.args, other.kwargs)
    self._then(other)
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
      tuple root_link = self.root_link
      object s = f'<{self.__class__.__name__}'

    if root_link is not None:
      s += f'({root_link[0]}, {root_link[5]}, {root_link[6]})'
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
  cdef int on_attr(self, str attr) except -1:
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_conditional is not None:
      self.finalize_conditional()
    self.current_attr = attr

  cdef int finalize_fattr(self, tuple args, dict kwargs) except -1:
    cdef str attr = self.current_attr
    self.current_attr = None
    self._then(attr, is_attr=True, is_fattr=True, is_with_root=False, ignore_result=False, args=args, kwargs=kwargs)

  def __getattr__(self, attr) -> ChainAttr:
    self.on_attr(attr)
    return self

  def __call__(self, *args, **kwargs) -> ChainAttr:
    if self.current_attr is None:
      # much slower than directly calling `._run()`, but we have no choice since
      # we wish support arbitrary __call__ invocations on attributes.
      # avoid running a chain this way. opt to use `.run()` instead.
      return self.run(*args, **kwargs)
    else:
      self.finalize_fattr(args, kwargs)
      return self


# cannot have multiple inheritance in Cython.
cdef class CascadeAttr(ChainAttr):
  # noinspection PyMissingConstructor
  def __init__(self, __v=Null, *args, **kwargs):
    self.init(__v, args, kwargs, is_cascade=True)


cdef class FrozenChain:
  cdef object _chain_run

  @property
  def decorator(self):
    _chain_run = self._chain_run
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


cdef object build_conditional(
  object conditional, bint is_custom, bint not_, object on_true_v, tuple true_args, dict true_kwargs, object on_false_v,
  tuple false_args, dict false_kwargs
):
  """ A helper method to create conditionals
  
  Creates a function which conditionally invokes a callback.
  :param on_true_v: a callback for a truthy value.
  :param true_args: the arguments for the truthy callback.
  :param true_kwargs: the keyword-arguments for the truthy callback.
  :param on_false_v: a callback for a falsy value.
  :param false_args: the arguments for the falsy callback. 
  :param false_kwargs: the keyword-arguments for the falsy callback.
  :return: A function which accepts a boolean value and returns either the same value, or the evaluation of the
  corresponding callback.
  """
  # a similar implementation using a class was tested and found to be marginally slower than this
  # so there isn't really a way to further optimize it until Cython adds support for nested `cdef`s.
  async def if_else_async(object r, object cv):
    r = _if_else(await r, cv)
    if isawaitable(r):
      r = await r
    return r

  def if_else(object cv):
    r = conditional(cv)
    if is_custom and isawaitable(r):
      return if_else_async(r, cv)
    return _if_else(r, cv)

  def _if_else(object r, object cv):
    if not_:
      r = not r
    if r:
      return evaluate_value(
        v=on_true_v, cv=cv, is_attr=False, is_fattr=False, args=true_args, kwargs=true_kwargs
      )
    elif on_false_v is not Null:
      return evaluate_value(
        v=on_false_v, cv=cv, is_attr=False, is_fattr=False, args=false_args, kwargs=false_kwargs
      )
    return cv

  return if_else


cdef object run_callback(tuple callback_meta, object rv):
  """
  A helper method to run the callbacks of on-except/on-finally
  :param callback_meta: A tuple of the form `(fn_or_attr, args, kwargs)`.
  :param rv: the root value of the Chain.
  :return: the result of the callback.
  """
  cdef:
    object fn_or_attr
    bint is_attr
    tuple args
    dict kwargs

  fn_or_attr, args, kwargs = callback_meta
  is_attr = isinstance(fn_or_attr, str)
  # if the target is an attribute, we also assume that it is a method attribute, since
  # it doesn't really make sense to access an attribute of an object on except/finally
  # (unless it is a @property function, which is not common, and a bad design to have a
  # @property function perform actions).
  return evaluate_value(
    v=fn_or_attr, cv=rv, is_attr=is_attr, is_fattr=is_attr, args=args, kwargs=kwargs
  )


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


cdef object foreach(object fn, bint ignore_result):
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
  return _foreach


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


cdef object with_(object v, tuple args, dict kwargs):
  async def with_async(object result, object cv):
    try:
      return await result
    finally:
      cv.__exit__(*sys.exc_info())
  def with_(object cv):
    if hasattr(cv, '__aenter__'):
      return async_with(v, cv, args, kwargs)
    cdef object ctx, result = None
    try:
      ctx = cv.__enter__()
      if v is Null:
        result = ctx
      else:
        result = evaluate_value(
          v=v, cv=ctx, is_attr=False, is_fattr=False, args=args, kwargs=kwargs
        )
    finally:
      if isawaitable(result):
        return with_async(result, cv)
      cv.__exit__(*sys.exc_info())
      return result
  return with_


async def async_with(object v, object cv, tuple args, dict kwargs):
  cdef object ctx, result
  async with cv as ctx:
    if v is Null:
      return ctx
    result = evaluate_value(
      v=v, cv=ctx, is_attr=False, is_fattr=False, args=args, kwargs=kwargs
    )
    if isawaitable(result):
      return await result
    return result


def create_chain_link_exception(
  v, cv, is_attr: bool, is_fattr: bool, args: tuple | None, kwargs: dict | None, idx: int,
  is_exc_raised_on_await: bool = False
) -> QuentException:
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

  def append_await(s: str) -> str:
    if is_exc_raised_on_await:
      s = f'await {s}'
    return s

  def format_exception_details(literal: bool):
    if v is Null:
      return str(v), 'Null'

    elif is_attr:
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

    elif not is_attr and callable(v):
      s = get_object_name(v, literal)
      return f'{v}', f'{s}()' if cv is Null else f'{s}({cv})'

    else:
      return str(v), str(v)

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
  if is_exc_raised_on_await:
    s += '\n\tThe exception was raised as the result of awaiting a coroutine'
  return QuentException(
    s
    + f'\n\t{object_str}'
    + f'\n\t`{append_await(readable_str)}`'
  )
