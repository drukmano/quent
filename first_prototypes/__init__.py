from __future__ import annotations

import asyncio
from inspect import iscoroutine, iscoroutinefunction

from typing import Any, Callable, Coroutine

from pydruk import Null


try:
  from pipe import Pipe as _PipeObj
except ImportError:
  _PipeObj = None


class SequentException(Exception):
  pass


def process_next_value(
  v: Any | Callable, pv: Any, is_attr: bool, is_fattr: bool, is_void: bool, args: tuple, kwargs: dict
) -> Any:
  """
  Given a value 'v', and a previous value 'pv', determine how to apply 'v'
  :param v: current value
  :param pv: previous value
  :param is_attr: whether v is an attribute of pv
  :param is_fattr: whether v is a method of pv
  :param is_void: whether v should be applied with no parameters
  :param args: arguments to pass to v
  :param kwargs: kw-arguments to pass to v
  :return: Null, v, or the result of v
  """
  if v is Null or v is ...:
    return Null
  elif is_attr:
    v = getattr(pv, v)
  if args and args[0] is ...:
    return v()
  elif args or kwargs:
    return v(*args, **kwargs)
  elif is_fattr:
    return v()
  elif not is_attr and callable(v):
    return v() if is_void or pv is Null else v(pv)
  else:
    return v


def parse_and_run_meta_func(fn_meta: tuple[Callable | str, tuple, dict], v: Any) -> Any:
  fn, args, kwargs = fn_meta
  is_attr = isinstance(fn, str)
  if is_attr:
    fn = getattr(v, fn)
  # using Sequent might be more elegant and reduce a few lines of code, but it adds more unnecessary overhead
  #return ~Sequent(process_next_value(fn, v, is_attr, is_attr, v is Null, args, kwargs)).eq(v, f1).else_(f2)
  if args and args[0] is ...:
    return fn()
  elif args or kwargs:
    return fn(*args, **kwargs)
  elif not is_attr:
    return fn() if v is Null else fn(v)
  else:
    raise SequentException('An invalid value was passed to except/finally.')


async def await_callback(v: Coroutine, callback: Callable):
  v = callback(await v)
  return await v if iscoroutine(v) else v


async def await_(v: Coroutine):
  return await v


def await_callback_if_coro(v: Any, callback: Callable):
  return await_callback(v, callback) if iscoroutine(v) else callback(v)


def await_if_coro(v: Any):
  return await_(v) if iscoroutine(v) else v


class _SequentIfElse:
  __slots__ = '_comparator', '_sequent', '_on_true', '_on_false', '_true_args', '_true_kwargs', '_false_args', '_false_kwargs'

  def __init__(self, sequent: Sequent, comparator: Callable, on_true: Callable = None, *args, **kwargs):
    self._comparator = comparator
    self._sequent: Sequent = sequent
    # this variable may (later) hold 'on_false' instead if the result is false-y
    self._on_true: Callable | None = on_true
    self._true_args: tuple = args
    self._true_kwargs: dict = kwargs
    self._on_false: Callable | None = None
    self._false_args: tuple | None = None
    self._false_kwargs: dict | None = None

  def else_(self, on_false: Callable, *args, **kwargs) -> Sequent:
    self._on_false = on_false
    self._false_args = args
    self._false_kwargs = kwargs
    return self._finalize()

  def attr(self, attr: str, *args, **kwargs) -> Sequent:
    return self._finalize().attr(attr, *args, **kwargs)

  def call(self, attr: str, *args, **kwargs) -> Sequent:
    return self._finalize().call(attr, *args, **kwargs)

  def root(self, v: Any | Callable = ...) -> Sequent:
    return self._finalize().root(v)

  def if_(self, v: Callable | Ellipsis, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return self._finalize().if_(v, on_true, *args, **kwargs)

  def eq(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return self._finalize().eq(v, on_true, *args, **kwargs)

  def neq(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return self._finalize().neq(v, on_true, *args, **kwargs)

  def is_(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return self._finalize().is_(v, on_true, *args, **kwargs)

  def is_not(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return self._finalize().is_not(v, on_true, *args, **kwargs)

  def in_(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return self._finalize().in_(v, on_true, *args, **kwargs)

  def not_in(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return self._finalize().not_in(v, on_true, *args, **kwargs)

  def then(self, v: Any | Callable = Null, *args, **kwargs) -> Sequent:
    return self._finalize().then(v, *args, **kwargs)

  def except_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Sequent:
    return self._finalize().except_(fn_or_attr, *args, **kwargs)

  def finally_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Sequent:
    return self._finalize().finally_(fn_or_attr, *args, **kwargs)

  def resolve(self, v: Any | Callable = Null, *args, **kwargs) -> Any:
    return self._finalize().resolve(v, *args, **kwargs)

  def _finalize(self) -> Sequent:
    return self._sequent.then(self._compare)

  def _compare(self, v: Any):
    v = self._comparator(v)
    return await_callback(v, self._resolve_sync) if iscoroutine(v) else self._resolve_sync(v)

  def _resolve_sync(self, v: bool):
    if v:
      if self._on_true:
        return self._on_true(*self._true_args, **self._true_kwargs)
    elif self._on_false:
      return self._on_false(*self._false_args, **self._false_kwargs)
    return v

  def __call__(self, v: Any | Callable = Null, *args, **kwargs):
    return self.resolve(v, *args, **kwargs)

  def __invert__(self):
    return self.resolve()

  def __bool__(self):
    return True


class _SequentMeta(type):
  def __or__(cls, other: Any) -> Sequent:
    return cls(other)

  def __ror__(cls, other: Any) -> Sequent:
    return cls(other)

  def __rrshift__(cls, other: Any) -> Sequent:
    return cls(other)

  def __lshift__(cls, other: Any) -> Sequent:
    return cls(other)


class Sequent(metaclass=_SequentMeta):
  """ Sequentially and declaratively execute synchronous and asynchronous code

      Example:
          `await Sequent(fetch_data).then(process_data).then(send_data).then(print)`
          where 'fetch_data' and 'send_data' is asynchronous and 'process_data' (and 'print') is synchronous
      The equivalent statement in "pure" Python would be:
          `print(await send_data(process_data(await fetch_data())))`
      Which I find quite ugly and unreadable.

      A big reason I developed this project is to be able to work with objects which could be either async or sync.
      Take the last example, we can do this:
          def some_method(url):
              return Sequent(fetch_data(url)).then(process_data).then(send_data).resolve()
      Here, 'fetch_data' may be async or sync, which the code which called 'some_method' will know, but we don't.

      Without using Sequent, this is impossible to achieve unless we split the function into async and sync versions:
          def some_method_sync(url):
              return send_data(process_data(fetch_data(url)))

          async def some_method_async(url):
              return await send_data(process_data(await fetch_data(url)))
      Which is a duplicate code, and thus double the maintenance, double the headache.

      Sequent also allows for transparent-like proxies on arbitrary objects:
          await Sequent(get_cls_instance()).attr
          await Sequent(get_cls_instance()).func()

      Provide any args or kwargs to .then(fn) in order to run fn() without the previous value as an argument. If you
      wish to ignore the previous value but the function does not receive any arguments, you may pass it an
      Ellipsis (...) as an arg, like so `.then(fn, ...)`
  """

  __slots__ = (
    'v', '_nv', '_is_try', '_void', '_is_root', '_is_attr', '_is_fattr', '_args', '_kwargs', '_root', '_with_root',
    '_subsequent', '_attr', '_on_except', '_on_finally', '_can_eager', '_is_invoked', '_frozen'
  )

  @classmethod
  def lazy(cls, v: Any | Callable = Null, *args, **kwargs):
    return cls.try_(v, *args, **kwargs)

  @classmethod
  def try_(cls, v: Any | Callable = Null, *args, **kwargs):
    seq: Sequent = cls(v, *args, __seq_iseager__=False, **kwargs)
    seq._is_try = True
    return seq

  @classmethod
  def from_list(cls, lst: list | tuple) -> Sequent:
    if not lst:
      return cls()
    seq = cls(lst[0])
    for el in lst[1:]:
      seq = seq._consequent(el)
    return seq.resolve()

  @classmethod
  def from_(cls, *args) -> Sequent:
    return cls.from_list(args)

  def __init__(self, v: Any | Callable = Null, *args, **kwargs):
    self._root: Sequent = self
    self._subsequent: Sequent | None = None
    self._attr: str | None = None
    self._on_except: tuple | None = None
    self._on_finally: tuple | None = None
    self._is_root: bool = kwargs.pop('__seq_isroot__', True)
    self._is_attr: bool = False
    self._is_fattr: bool = False
    self._with_root: bool = False
    self._is_try: bool = False

    self.v: Any
    self._args: tuple
    self._kwargs: dict
    self._nv: Any
    self._can_eager: bool
    self._is_invoked: bool
    self._init_root(v, args, kwargs)

  def _init_root(self, v: Any | Callable, args: tuple, kwargs: dict):
    if self._is_root:
      self._frozen: bool = v is not Null
      if v is ...:
        v = Null
        self._args = args = ...,
        self._void = True
      else:
        self._void: bool = not self._frozen

    # is eager
    if kwargs.pop('__seq_iseager__', True) and not self._void:
      v = process_next_value(v, Null, False, False, True, args, kwargs)
      self._can_eager = not iscoroutine(v)
      self._is_invoked = True
    else:
      self._args = args
      self._kwargs = kwargs
      self._can_eager = False
      self._is_invoked = False
    self.v = self._nv = v

  def attr(self, attr: str, *args, **kwargs) -> Sequent:
    return self.then(attr, *args, __seq_attr__=True, **kwargs)

  def call(self, attr: str, *args, **kwargs) -> Sequent:
    return self.then(attr, *args, __seq_fattr__=True, **kwargs)

  def root(self, v: Any | Callable = ...) -> Sequent:
    # the reason for the Ellipsis default (and not Null) is to force 'then()' to create a Sequent (even if it is
    # empty) since if v == Null it ignores it.
    return self.then(v, __seq_with_root__=True)

  def if_(self, v: Callable | Ellipsis, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    if v is Ellipsis:
      v = bool
    return _SequentIfElse(self, v, on_true, *args, **kwargs)

  def eq(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return _SequentIfElse(self, lambda pv: pv == v, on_true, *args, **kwargs)

  def neq(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return _SequentIfElse(self, lambda pv: pv != v, on_true, *args, **kwargs)

  def is_(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return _SequentIfElse(self, lambda pv: pv is v, on_true, *args, **kwargs)

  def is_not(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return _SequentIfElse(self, lambda pv: pv is not v, on_true, *args, **kwargs)

  def in_(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return _SequentIfElse(self, lambda pv: pv in v, on_true, *args, **kwargs)

  def not_in(self, v: Any, on_true: Callable = None, *args, **kwargs) -> _SequentIfElse:
    return _SequentIfElse(self, lambda pv: pv not in v, on_true, *args, **kwargs)

  def then(self, v: Any | Callable = Null, *args, **kwargs) -> Sequent:
    if v is Null:
      return self
    if _PipeObj and isinstance(v, _PipeObj):
      v = v.function
    # this is primarily to improve performance since we are not creating more Sequents
    if self._can_eager:
      pv = self._nv
      is_fattr = kwargs.pop('__seq_fattr__', False)
      is_attr = kwargs.pop('__seq_attr__', False) or is_fattr  # intentional order, for .pop()
      is_with_root = kwargs.pop('__seq_with_root__', False)
      if is_with_root:
        pv = self.v
      v = process_next_value(v, pv, is_attr, is_fattr, self._void, args, kwargs)
      if v is Null:
        if is_with_root:
          self._nv = pv
        return self
      if not iscoroutine(v):
        self._nv = v
        return self
      args, kwargs = (), {}
      self._can_eager = False
    if self._attr is not None:
      return self._finalize_attr()._consequent(v, *args, **kwargs)
    return self._consequent(v, *args, **kwargs)

  def except_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Sequent:
    if not self._root._is_try:
      raise SequentException('Use with \'Sequent.try_()\' to use try-catch callbacks.')
    self._root._on_except = fn_or_attr, args, kwargs
    return self

  def finally_(self, fn_or_attr: Callable | str, *args, **kwargs) -> Sequent:
    if not self._root._is_try:
      raise SequentException('Use with \'Sequent.try_()\' to use try-catch callbacks.')
    self._root._on_finally = fn_or_attr, args, kwargs
    return self

  def _consequent(
    self, v: Any | Callable, *args,
    __seq_attr__: bool = False, __seq_fattr__: bool = False, __seq_with_root__: bool = False, **kwargs
  ):
    self._subsequent = subsequent = self.__class__(v, *args, __seq_iseager__=False, __seq_isroot__=False, **kwargs)
    subsequent._root = self._root
    #subsequent._is_root = False
    subsequent._is_attr = __seq_attr__ or __seq_fattr__
    subsequent._is_fattr = __seq_fattr__
    subsequent._with_root = __seq_with_root__
    del subsequent._on_except, subsequent._on_finally
    if not self._with_root:
      del self._root
    return subsequent

  def resolve(self, v: Any | Callable = Null, *args, **kwargs) -> Any:
    if self._attr is not None:
      return self._finalize_attr().resolve()
    root = self._root
    if not self._with_root:
      del self._root
    if v is not Null and not root._frozen:
      kwargs['__seq_iseager__'] = False
      root._init_root(v, args, kwargs)
    return root._resolve()

  def _resolve(self):
    if self._can_eager:
      # if self.with_root then Insequent, since self is root
      v = self._nv if not self._with_root else self.v
      return v if v is not Null else None
    ignore_finally = False
    try:
      if not self._void:
        if not self._is_invoked:
          self.v = process_next_value(self.v, Null, False, False, True, self._args, self._kwargs)
          del self._args, self._kwargs
          self._is_invoked = True
        if iscoroutine(self.v):
          ignore_finally = True
          return self._resolve_async(True)
      # iscoroutinefunction is quite expensive, so we check for it only in case 'v' is not a coroutine
      if (
        (self._on_finally and iscoroutinefunction(self._on_finally[0]))
        or (self._on_except and iscoroutinefunction(self._on_except[0]))
      ):
        ignore_finally = True
        return self._resolve_async(False)
      v = self._next_subsequent(self.v)
      return v if v is not Null else None
    except:
      if self._on_except:
        parse_and_run_meta_func(self._on_except, self.v)
      raise
    finally:
      if not ignore_finally and self._on_finally:
        parse_and_run_meta_func(self._on_finally, self.v)

  async def _resolve_async(self, is_v_async: bool):
    try:
      if is_v_async:
        self.v = await self.v
      v = self._next_subsequent(self.v)
      if v is Null:
        return None
      elif iscoroutine(v):
        v = await v
      return v
    except:
      if self._on_except:
        v = parse_and_run_meta_func(self._on_except, self.v)
        if iscoroutine(v):
          await v
      raise
    finally:
      if self._on_finally:
        v = parse_and_run_meta_func(self._on_finally, self.v)
        if iscoroutine(v):
          await v

  def _next_subsequent(self, pv: Any):
    sub: Sequent = self._subsequent
    del self._subsequent
    if sub is None:
      # if root.with_root then Insequent
      if self._with_root and (self.v is Null or self._root._with_root):
        pv = self._root.v
      return pv
    sub._void = is_void = self._void
    if sub._with_root:
      pv = sub._root.v
    v = process_next_value(sub.v, pv, sub._is_attr, sub._is_fattr, is_void, sub._args, sub._kwargs)
    if v is Null:
      v = pv
    del sub._args, sub._kwargs,  # sub.v  # used above in 'if sub is None' - 'self.v'
    return await_callback(v, sub._next_subsequent) if iscoroutine(v) else sub._next_subsequent(v)

  def _finalize_attr(self, *args, **kwargs) -> Sequent:
    attr = self._attr
    self._attr = None
    return self.attr(attr, *args, **kwargs)

  def __call__(self, v: Any | Callable = Null, *args, **kwargs):
    return self.resolve(v, *args, **kwargs)

  def __invert__(self):
    return self.resolve()

  # this can introduce confusions and unexpected behavior
  #def __await__(self):
  #    return self.resolve().__await__()

  def __bool__(self):
    return True

  def __or__(self, other: Any):
    return self.then(other)

  def __rshift__(self, other: Any):
    return self.then(other)


class Sequentr(Sequent):
  """ Like Sequent, but with dynamic attribute access
      This acts as a shortcut for seq.attr('a')
          seq.a   seq.a()

      Since declaring __getattr__ impacts memory and performance (due to __getattr__ and __slots__),
      we make this syntactic sugar opt-in
  """
  __slots__ = ()

  def __getattr__(self, attr: str):
    if self._attr is not None:
      return self._finalize_attr().__getattr__(attr)
    self._attr = attr
    return self

  def __call__(self, *args, **kwargs):
    if self._attr is None:
      return self.resolve()
    else:
      if not args:
        args = ...,
      return self._finalize_attr(*args, **kwargs)


class Insequent(Sequent):
  """ Like Sequent, but instead of chaining results,
      This uses the root value (the one passed to Insequent obstructor) as the previous
      argument for all sub Sequents.
      For example,
          Insequentr(obj).a1().a2().a3.a4()
      is equivalent to
          obj.a1()
          obj.a2()
          obj.a3
          obj.a4()
          return obj

      And,
          Insequent(obj).then(b1).then(b2).then(b3, something).then(b4)
      is equivalent to
          b1(obj)
          b2(obj)
          b3(something)
          b4(obj)
          return obj
  """
  __slots__ = ()

  def __init__(self, v: Any | Callable = Null, *args, **kwargs):
    super().__init__(v, *args, **kwargs)
    self._with_root = True

  def then(self, v: Any | Callable = Null, *args, **kwargs) -> Sequent:
    kwargs['__seq_with_root__'] = True
    return super().then(v, *args, **kwargs)


class Insequentr(Insequent, Sequentr):
  __slots__ = ()
