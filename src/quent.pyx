# cython: binding=False, boundscheck=False, wraparound=False, language_level=3, embedsignature=True
## cython: profile=True, linetrace=True, warn.undeclared=True
## distutils: define_macros=CYTHON_TRACE_NOGIL=1

#cimport cython
import collections.abc
import types
from typing import Any, Callable, Coroutine


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

  def __init__(self, v: Any | Callable = Null, *args, **kwargs):
    self.init(v, args, kwargs)

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
    bint is_cascade
    list links
    tuple on_except, on_finally, current_on_true
    str current_attr

  @classmethod
  def from_(cls, *args) -> Chain:
    return from_list(cls, args)

  def __init__(self, root_value: Any | Callable = Null, *args, **kwargs):
    """
    Create a new Chain
    :param v: the root value of the chain
    :param args: arguments to pass to `v`
    :param kwargs: keyword-arguments to pass to `v`
    """
    self.init(root_value, args, kwargs)

  cdef int init(self, object root_value, args: tuple, kwargs: dict, bint is_cascade = False) except -1:
    self.is_cascade = is_cascade

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
    self.current_on_true = None
    self.current_attr = None

  # https://github.com/cython/cython/issues/1630
  cdef int _then(
    self, object v, bint is_attr = False, bint is_fattr = False, bint is_with_root = False, bint ignore_result = False,
    tuple args = None, dict kwargs = None
  ) except -1:
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_on_true is not None:
      self.finalize_conditional()

    is_with_root = is_with_root or self.is_cascade
    if _PipeCls is not None and isinstance(v, _PipeCls):
      v = v.function
    self.links.append((v, is_attr, is_fattr, is_with_root, ignore_result, args, kwargs))

  cdef object _run(self, object v, tuple args, dict kwargs):
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_on_true is not None:
      self.finalize_conditional()

    cdef:
      # current chain value, root value
      object cv = Null, rv = Null, result
      bint is_attr = False, is_fattr = False, is_with_root = False, ignore_result = False,
      bint is_void = self.root_link is None, ignore_try = False, is_null = v is Null
      list links = self.links
      tuple link, on_except = self.on_except, on_finally = self.on_finally, root_link = self.root_link
      int idx = -1
      Exception e

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
          ignore_try = True
          return self._run_async(rv, Null, idx, is_void, v, Null, False, False, False, args, kwargs)

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
            ignore_try = True
            return self._run_async(result, rv, idx, is_void, v, cv, is_attr, is_fattr, ignore_result, args, kwargs)
          if not ignore_result:
            cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as e:
      if not ignore_try and on_except is not None:
        run_callback(on_except, rv)
      raise e from create_chain_link_exception(v, cv, is_attr, is_fattr, args, kwargs, idx)

    finally:
      if not ignore_try and on_finally is not None:
        run_callback(on_finally, rv)

  async def _run_async(
    self, object result, object rv, int idx, bint is_void,
    object v, object cv, bint is_attr, bint is_fattr, bint ignore_result, tuple args, dict kwargs
  ):
    # we pass the full current link data to be able to format an appropriate
    # exception message in case one is raised from awaiting `result`
    cdef:
      bint is_with_root = False, is_exc_raised_on_await = False
      tuple on_except = self.on_except, on_finally = self.on_finally
      list links = self.links
      Exception e

    try:
      try:
        result = await result
        if not ignore_result:
          cv = result
      except:
        is_exc_raised_on_await = True
        raise
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
              if not ignore_result:
                cv = result
            except:
              is_exc_raised_on_await = True
              raise
          elif not ignore_result:
            cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as e:
      # even though it seems that a coroutine that has raised an exception still
      # returns True for `isawaitable(result)`, I don't want to use this method to check
      # whether an exception was raised from awaiting `result` as I couldn't find if
      # this is an intended behavior or not.
      if on_except is not None:
        result = run_callback(on_except, rv)
        if isawaitable(result):
          await result
      raise e from create_chain_link_exception(
        v, cv, is_attr, is_fattr, args, kwargs, idx, is_exc_raised_on_await
      )

    finally:
      if on_finally is not None:
        result = run_callback(on_finally, rv)
        if isawaitable(result):
          await result

  cdef int _except(self, object fn_or_attr, tuple args, dict kwargs):
    if self.on_except is not None:
      raise QuentException('You can only register one \'except\' callback.')
    self.on_except = fn_or_attr, args, kwargs

  cdef int _finally(self, object fn_or_attr, tuple args, dict kwargs):
    if self.on_finally is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally = fn_or_attr, args, kwargs

  cdef int _if(self, object predicate, object on_true_v = Null, tuple true_args = None, dict true_kwargs = None) except -1:
    self._then(predicate)
    self.current_on_true = (on_true_v, true_args, true_kwargs)

  cdef int _else(self, object on_false_v, tuple false_args, dict false_kwargs) except -1:
    if self.current_on_true is None:
      raise QuentException(
        'You cannot use \'.else_()\' without a preceding conditional (\'.if_()\', \'.eq()\', etc.)'
      )
    self.finalize_conditional(on_false_v, false_args, false_kwargs)

  cdef int finalize_attr(self) except -1:
    cdef str attr = self.current_attr
    self.current_attr = None
    self._then(attr, is_attr=True)

  cdef int finalize_conditional(
    self, object on_false_v = Null, tuple false_args = None, dict false_kwargs = None
  ) except -1:
    cdef:
      object on_true_v
      tuple true_args
      dict true_kwargs
    on_true_v, true_args, true_kwargs = self.current_on_true
    self.current_on_true = None
    if on_true_v is not Null or on_false_v is not Null:
      self._then(build_conditional(on_true_v, true_args, true_kwargs, on_false_v, false_args, false_kwargs))

  def run(self, v: Any | Callable = Null, *args, **kwargs) -> Any:
    return self._run(v, args, kwargs)

  # cannot use cpdef with *args **kwargs.
  def then(self, v: Any | Callable, *args, **kwargs) -> Chain:
    self._then(v, is_attr=False, is_fattr=False, is_with_root=False, ignore_result=False, args=args, kwargs=kwargs)
    return self

  def root(self, v: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._then(v, is_attr=False, is_fattr=False, is_with_root=True, ignore_result=False, args=args, kwargs=kwargs)
    return self

  def ignore(self, v: Any | Callable, *args, **kwargs) -> Chain:
    # register a value to be evaluated but will not propagate its result forwards.
    self._then(v, is_attr=False, is_fattr=False, is_with_root=False, ignore_result=True, args=args, kwargs=kwargs)
    return self

  def root_ignore(self, v: Any | Callable, *args, **kwargs) -> Chain:
    self._then(v, is_attr=False, is_fattr=False, is_with_root=True, ignore_result=True, args=args, kwargs=kwargs)
    return self

  def attr(self, name: str) -> Chain:
    self._then(name, is_attr=True)
    return self

  def attr_fn(self, name: str, *args, **kwargs) -> Chain:
    self._then(name, is_attr=True, is_fattr=True, is_with_root=False, ignore_result=False, args=args, kwargs=kwargs)
    return self

  def foreach(self, fn: Callable) -> Chain:
    self._then(foreach(fn, True))
    return self

  def foreach_do(self, fn: Callable) -> Chain:
    self._then(
      foreach(fn, False), is_attr=False, is_fattr=False, is_with_root=False, ignore_result=True, args=None, kwargs=None
    )
    return self

  def with_(self, v: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._then(with_(v, args, kwargs))
    return self

  def with_do(self, v: Any | Callable, *args, **kwargs) -> Chain:
    self._then(
      with_(v, args, kwargs), is_attr=False, is_fattr=False, is_with_root=False, ignore_result=True, args=None,
      kwargs=None
    )
    return self

  def except_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Chain:
    self._except(fn_or_attr, args, kwargs)
    return self

  def finally_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Chain:
    self._finally(fn_or_attr, args, kwargs)
    return self

  def if_(self, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(bool, on_true, args, kwargs)
    return self

  def else_(self, on_false: Any | Callable, *args, **kwargs) -> Chain:
    self._else(on_false, args, kwargs)
    return self

  def not_(self) -> Chain:
    # use a named function (instead of a lambda) to have more details in the exception stacktrace
    def not_(object cv) -> bool: return not cv
    self._if(not_)
    return self

  def eq(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    def equals(object cv) -> bool: return cv == v
    self._if(equals, on_true, args, kwargs)
    return self

  def neq(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    def not_equals(object cv) -> bool: return cv != v
    self._if(not_equals, on_true, args, kwargs)
    return self

  def is_(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    def is_(object cv) -> bool: return cv is v
    self._if(is_, on_true, args, kwargs)
    return self

  def is_not(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    def is_not(object cv) -> bool: return cv is not v
    self._if(is_not, on_true, args, kwargs)
    return self

  def in_(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    def in_(object cv) -> bool: return cv in v
    self._if(in_, on_true, args, kwargs)
    return self

  def not_in(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    def not_in(object cv) -> bool: return cv not in v
    self._if(not_in, on_true, args, kwargs)
    return self

  def __or__(self, other: Any) -> Chain | Coroutine:
    if isinstance(other, run):
      return self._run(other.root_value, other.args, other.kwargs)
    self._then(other)
    return self

  def __call__(self, v: Any | Callable = Null, *args, **kwargs) -> Any:
    return self._run(v, args, kwargs)

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
  def __init__(self, v: Any | Callable = Null, *args, **kwargs):
    self.init(v, args, kwargs, is_cascade=True)


cdef class ChainAttr(Chain):
  cdef int on_attr(self, str attr) except -1:
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_on_true is not None:
      self.finalize_conditional()
    self.current_attr = attr

  cdef int finalize_fattr(self, tuple args, dict kwargs) except -1:
    cdef str attr = self.current_attr
    self.current_attr = None
    self._then(attr, is_attr=True, is_fattr=True, is_with_root=False, ignore_result=False, args=args, kwargs=kwargs)

  def __getattr__(self, attr: str) -> ChainAttr:
    self.on_attr(attr)
    return self

  def __call__(self, *args, **kwargs) -> Any | ChainAttr:
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
  def __init__(self, v: Any | Callable = Null, *args, **kwargs):
    self.init(v, args, kwargs, is_cascade=True)


cdef object build_conditional(
  object on_true_v, tuple true_args, dict true_kwargs, object on_false_v, tuple false_args, dict false_kwargs
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
  def if_else(bint v):
    if v:
      if on_true_v is not Null:
        return evaluate_value(
          v=on_true_v, cv=Null, is_attr=False, is_fattr=False, args=true_args, kwargs=true_kwargs
        )
    elif on_false_v is not Null:
      return evaluate_value(
        v=on_false_v, cv=Null, is_attr=False, is_fattr=False, args=false_args, kwargs=false_kwargs
      )
    return v
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


cdef object foreach(object fn, bint with_lst):
  """ A helper method for `.foreach()` """
  def foreach(object cv):
    if hasattr(cv, '__aiter__'):
      return async_gen_foreach(cv, fn, with_lst)
    cdef list lst = []
    cdef object el, result
    cv = iter(cv)
    for el in cv:
      result = fn(el)
      if isawaitable(result):
        return async_foreach(cv, fn, result, lst, with_lst)
      if with_lst:
        lst.append(result)
    if with_lst:
      return lst
  return foreach


async def async_foreach(object cv, object fn, object result, list lst, bint with_lst):
  """ A helper method for `.foreach()` """
  cdef object el
  if with_lst:
    lst.append(await result)
  for el in cv:
    result = fn(el)
    if isawaitable(result):
      result = await result
    if with_lst:
      lst.append(result)
  if with_lst:
    return lst


async def async_gen_foreach(object cv, object fn, bint with_lst):
  """ A helper method for `.foreach()` """
  cdef list lst
  if with_lst:
    lst = []
  cdef object el, result
  async for el in cv:
    result = fn(el)
    if isawaitable(result):
      result = await result
    if with_lst:
      lst.append(result)
  if with_lst:
    return lst


cdef object with_(object v, tuple args, dict kwargs):
  """ A helper method for `.with_()` """
  def with_(object cv):
    if hasattr(cv, '__aenter__'):
      return async_with(v, cv, args, kwargs)
    cdef object ctx
    with cv as ctx:
      if v is Null:
        return ctx
      return evaluate_value(
        v=v, cv=ctx, is_attr=False, is_fattr=False, args=args, kwargs=kwargs
      )
  return with_


async def async_with(object v, object cv, tuple args, dict kwargs):
  """ A helper method for `.with_()` """
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
  v: Any, cv: Any, is_attr: bool, is_fattr: bool, args: tuple | None, kwargs: dict | None, idx: int,
  is_exc_raised_on_await: bool = False
) -> QuentException:
  """
  Create a string representation of the evaluation of 'v' based on the same rules
  used in `evaluate_value`
  """

  def get_object_name(o: Any, literal: bool) -> str:
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
