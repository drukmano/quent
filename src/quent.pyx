# cython: binding=False, boundscheck=False, wraparound=False, language_level=3str
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


cdef object evaluate_value(object v, object pv, bint is_attr, bint is_fattr, bint is_void, tuple args, dict kwargs):
  """
  Given a value `v`, and a previous value `pv`, evaluates `v`.
  :param v: the current value.
  :param pv: the previous value.
  :param is_attr: whether `v` is an attribute of `pv`.
  :param is_fattr: whether `v` is a method attribute of `pv`.
  :param is_void: whether `v` should be applied with no arguments.
  :param args: the arguments to pass to `v`.
  :param kwargs: the keyword-arguments to pass to `v`.
  :return: `Null`, `v`, or the evaluation of `v`.
  """
  if v is Null:
    return Null

  elif is_attr:
    v = getattr(pv, v)
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
    # `pv is Null` is for safety; in most cases, it simply means that `v` is the root value.
    return v() if is_void or pv is Null else v(pv)

  else:
    return v


cdef object run_callback(tuple callback_meta, object rv, bint is_void):
  """
  A helper method to run the callbacks of on-except/on-finally
  :param callback_meta: A tuple of the form `(fn_or_attr, args, kwargs)`.
  :param rv: the root value of the Chain.
  :param is_void: whether `fn` should be applied with no parameters.
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
    v=fn_or_attr, pv=rv, is_attr=is_attr, is_fattr=is_attr, is_void=is_void, args=args, kwargs=kwargs
  )


cdef object build_conditional(
  object on_true_v, tuple true_args, dict true_kwargs, object on_false_v, tuple false_args, dict false_kwargs
):
  """
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
  def fn(bint v):
    if v:
      if on_true_v is not Null:
        return evaluate_value(
          v=on_true_v, pv=Null, is_attr=False, is_fattr=False, is_void=True, args=true_args, kwargs=true_kwargs
        )
    elif on_false_v is not Null:
      return evaluate_value(
        v=on_false_v, pv=Null, is_attr=False, is_fattr=False, is_void=True, args=false_args, kwargs=false_kwargs
      )
    return v
  return fn


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
  if not links:
    return cls()
  cdef Chain seq = cls(links[0])
  for el in links[1:]:
    seq._then(el)
  return seq


cdef class Chain:
  cdef:
    tuple root_link
    bint is_cascade, is_void
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
    # whether all links will be evaluated with no arguments (except links with explicit arguments).
    # note that if `is_void=True`, calling `.run()` without a root value is only possible
    # with Cascade, e.g. for running a sequence of operations that are independent of each other.
    self.is_void = root_value is Null

    # `links` is a list which contains tuples of the following structure:
    # (
    #   object, # the link value, 'v'
    #   bool,   # whether 'v' is attribute
    #   bool,   # whether 'v' is method attribute
    #   bool,   # whether to evaluate 'v' with the [evaluated] root value
    #   tuple,  # the arguments to evaluate 'v' with
    #   dict    # the keyword-arguments to evaluate 'v' with
    # )
    self.links = []

    if not self.is_void:
      self.root_link = (root_value, False, False, False, args, kwargs)
    else:
      self.root_link = None

    self.on_except = None
    self.on_finally = None
    self.current_on_true = None
    self.current_attr = None

  # https://github.com/cython/cython/issues/1630
  cdef int _then(self, object v, bint is_attr = False, bint is_fattr = False, bint is_with_root = False, tuple args = None, dict kwargs = None) except -1:
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_on_true is not None:
      self.finalize_conditional()

    is_with_root = is_with_root or self.is_cascade
    if _PipeCls is not None and isinstance(v, _PipeCls):
      v = v.function
    self.links.append((v, is_attr, is_fattr, is_with_root, args, kwargs))

  cdef object _run(self, object v, tuple args, dict kwargs):
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_on_true is not None:
      self.finalize_conditional()

    cdef:
      # previous value, root value
      object pv, rv = Null
      bint is_attr, is_fattr, is_with_root, is_void = self.is_void, ignore_try = False, is_null = v is Null
      list links = self.links
      tuple link, on_except = self.on_except, on_finally = self.on_finally, root_link = self.root_link
      int idx

    if not is_void and not is_null:
      raise QuentException('Cannot override the root value of a Chain.')
    elif is_void and is_null and not self.is_cascade:
      raise QuentException('Unable to run a Chain without a root value. Use Cascade for that.')

    try:
      if not (is_void and is_null):
        if root_link is not None:
          v, is_attr, is_fattr, is_with_root, args, kwargs = root_link
        rv = pv = evaluate_value(
          v=v, pv=Null, is_attr=False, is_fattr=False, is_void=True, args=args, kwargs=kwargs
        )
        is_void = False
        if isawaitable(rv):
          ignore_try = True
          return self._run_async(rv, Null, 0, is_void)

      for idx, link in enumerate(links):
        # TODO optimize this line
        v, is_attr, is_fattr, is_with_root, args, kwargs = link
        if is_attr and is_void:
          raise QuentException('Cannot use attributes without a root value.')

        if is_with_root:
          pv = rv
        # `v is Null` is only possible when an empty `.root()` call has been made.
        if v is not Null:
          pv = evaluate_value(v, pv, is_attr, is_fattr, is_void, args, kwargs)
          if isawaitable(pv):
            ignore_try = True
            return self._run_async(pv, rv, idx+1, is_void)

      if self.is_cascade:
        pv = rv
      return pv if not is_void and pv is not Null else None

    except:
      if not ignore_try and on_except is not None:
        run_callback(on_except, rv, is_void)
      raise

    finally:
      if not ignore_try and on_finally is not None:
        run_callback(on_finally, rv, is_void)

  async def _run_async(self, object pv, object rv, int i, bint is_void):
    cdef:
      object v, callback_v
      bint is_attr, is_fattr, is_with_root, ignore_try = False
      tuple args, link, on_except = self.on_except, on_finally = self.on_finally
      list links = self.links
      dict kwargs

    try:
      pv = await pv
      # update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `rv` should always be `Null`.
      if not is_void and rv is Null:
        rv = pv

      for link in links[i:]:
        v, is_attr, is_fattr, is_with_root, args, kwargs = link
        if is_attr and is_void:
          raise QuentException('Cannot use attributes without a root value.')

        if is_with_root:
          pv = rv
        if v is not Null:
          pv = evaluate_value(v, pv, is_attr, is_fattr, is_void, args, kwargs)
          if isawaitable(pv):
            pv = await pv

      if self.is_cascade:
        pv = rv
      return pv if not is_void and pv is not Null else None

    except:
      if not ignore_try and on_except is not None:
        callback_v = run_callback(on_except, rv, is_void)
        if isawaitable(callback_v):
          await callback_v
      raise

    finally:
      if not ignore_try and on_finally is not None:
        callback_v = run_callback(on_finally, rv, is_void)
        if isawaitable(callback_v):
          await callback_v

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

  # cannot use cpdef with *args **kwargs.
  def then(self, v: Any | Callable, *args, **kwargs) -> Chain:
    self._then(v, is_attr=False, is_fattr=False, is_with_root=False, args=args, kwargs=kwargs)
    return self

  def run(self, v: Any | Callable = Null, *args, **kwargs) -> Any:
    return self._run(v, args, kwargs)

  def attr(self, attr: str) -> Chain:
    self._then(attr, is_attr=True)
    return self

  def call(self, attr: str, *args, **kwargs) -> Chain:
    self._then(attr, is_attr=True, is_fattr=True, is_with_root=False, args=args, kwargs=kwargs)
    return self

  def root(self, v: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._then(v, is_attr=False, is_fattr=False, is_with_root=True, args=args, kwargs=kwargs)
    return self

  def root_attr(self, attr: str) -> Chain:
    self._then(attr, is_attr=True, is_fattr=False, is_with_root=True)
    return self

  def root_call(self, attr: str, *args, **kwargs) -> Chain:
    self._then(attr, is_attr=True, is_fattr=True, is_with_root=True, args=args, kwargs=kwargs)
    return self

  def except_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Chain:
    self._except(fn_or_attr, args, kwargs)
    return self

  def finally_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Chain:
    self._finally(fn_or_attr, args, kwargs)
    return self

  def if_(self, v: Any | Callable | Ellipsis, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    if v is Ellipsis:
      v = bool
    self._if(v, on_true, args, kwargs)
    return self

  def else_(self, on_false: Any | Callable, *args, **kwargs) -> Chain:
    self._else(on_false, args, kwargs)
    return self

  def truthy(self) -> Chain:
    self._if(bool)
    return self

  def not_(self, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(lambda pv: not pv)
    return self

  def eq(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(lambda pv: pv == v, on_true, args, kwargs)
    return self

  def neq(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(lambda pv: pv != v, on_true, args, kwargs)
    return self

  def is_(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(lambda pv: pv is v, on_true, args, kwargs)
    return self

  def is_not(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(lambda pv: pv is not v, on_true, args, kwargs)
    return self

  def in_(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(lambda pv: pv in v, on_true, args, kwargs)
    return self

  def not_in(self, v: Any, on_true: Any | Callable = Null, *args, **kwargs) -> Chain:
    self._if(lambda pv: pv not in v, on_true, args, kwargs)
    return self

  def __or__(self, other: Any) -> Chain | Coroutine:
    if isinstance(other, run):
      return self._run(other.root_value, other.args, other.kwargs)
    self._then(other)
    return self

  def __call__(self, v: Any | Callable = Null, *args, **kwargs) -> Any:
    return self._run(v, args, kwargs)

  # while this is nice to have, I fear it will lead to unexpected behavior as
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
      s += f'({root_link[0]}, {root_link[4]}, {root_link[5]})'
    else:
      s += f'()'
    s += f'({len(self.links)} links)>'
    return s


cdef class ChainR(Chain):
  cdef dict __dict__

  cdef int on_attr(self, str attr) except -1:
    if self.current_attr is not None:
      self.finalize_attr()
    if self.current_on_true is not None:
      self.finalize_conditional()
    self.current_attr = attr

  cdef int finalize_fattr(self, tuple args, dict kwargs) except -1:
    cdef str attr = self.current_attr
    self.current_attr = None
    self._then(attr, is_attr=True, is_fattr=True, is_with_root=False, args=args, kwargs=kwargs)

  def __getattr__(self, attr: str) -> ChainR:
    self.on_attr(attr)
    return self

  def __call__(self, *args, **kwargs) -> Any | ChainR:
    if self.current_attr is None:
      # much slower than directly calling `._run()`, but we have no choice since
      # we wish support arbitrary __call__ invocations on attributes.
      # avoid running a chain this way. opt to use `.run()` instead.
      return self.run(*args, **kwargs)
    else:
      self.finalize_fattr(args, kwargs)
      return self


cdef class Cascade(Chain):
  # noinspection PyMissingConstructor
  def __init__(self, v: Any | Callable = Null, *args, **kwargs):
    self.init(v, args, kwargs, is_cascade=True)


# cannot have multiple inheritance in Cython.
cdef class CascadeR(ChainR):
  # noinspection PyMissingConstructor
  def __init__(self, v: Any | Callable = Null, *args, **kwargs):
    self.init(v, args, kwargs, is_cascade=True)
