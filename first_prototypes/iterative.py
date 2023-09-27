"""
This implementation of Sequent is iterative (instead of the recursive one) which in some cases may a bit be faster,
  but is much less readable and elegant. But, in most cases when using Sequent with asynchronous objects, the recursive
  one is a bit faster.

NOTE This IS NOT optimized the way we optimized the recursive Sequent, so it might be even better (memory usage also
  wasn't tested yet). But code needs a redesign, it is not clear.
"""

from __future__ import annotations

import collections
import inspect

from typing import Any, Callable, Awaitable

from pydruk import Null


class SequentException(Exception):
  pass


def is_async(obj: Any, async_: bool = None):
  return async_ is True or (async_ is None and inspect.isawaitable(obj))


class Sequent:
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

      This also allows for
          await Sequent(get_cls_instance()).cls_attr
          await Sequent(get_cls_instance()).cls_func()
  """

  def __new__(cls, obj: Any | Callable | Awaitable = Null, *args, **kwargs) -> SequentBase:
    return (SequentAsync if inspect.isawaitable(obj) else SequentSync)(obj, *args, **kwargs)

  @staticmethod
  def func(obj: Any | Callable = Null, *args, **kwargs):
    return SequentSync(obj, *args, **kwargs)

  @staticmethod
  def sync(obj: Any | Callable = Null, *args, **kwargs):
    return SequentSync(obj, *args, **kwargs)

  @staticmethod
  def async_(obj: Any | Callable | Awaitable = Null, *args, **kwargs):
    return SequentAsync(obj, *args, **kwargs)


class SequentBase:
  __slots__ = (
    'obj', 'args', 'kwargs', '_original_obj', '_chain', '_catch_chain', '_finally_chain', '_resolved', '_cattr',
    '_is_awaitable'
  )

  def __init__(self, obj: Any | Callable = Null, *args, **kwargs):
    self.obj = obj
    self.args = args
    self.kwargs = kwargs
    self._original_obj: Any = None

    # (fn_or_obj, is_async, None, False (is_attribute))
    # (attribute, args, kwargs, True (is_attribute))
    self._chain: list[
      tuple[Any | Callable, bool | None, None, bool] |
      tuple[str, tuple | None, dict | None, bool]
      ] = []
    self._catch_chain: list[tuple[Callable, bool]] = []
    self._finally_chain: list[tuple[Callable, bool, bool]] = []
    self._resolved: bool = False
    self._cattr: str | None = None
    self._is_awaitable: bool = False

  def then(self, obj: Any | Callable, *, a: bool = None) -> SequentBase:
    """
    Process 'obj' and chain the result forward
      if 'obj' is a function, then it will receive the previous result as an argument
      else, simply pass it forward
    """
    self._chain.append((obj, a, None, False))
    return self

  def catch(self, f: Callable, *, a: bool = None) -> SequentBase:
    """ Register a function which runs in the 'except' clause and receives (exc) """
    self._catch_chain.append((f, a))
    return self

  def finally_(self, f: Callable, *, a: bool = None, with_obj: bool = False) -> SequentBase:
    """ Register a function which runs in the 'finally' clause and optionally receives (obj) """
    self._finally_chain.append((f, a, with_obj))
    return self

  def _process_resolve(self, obj: Any):
    if self._cattr is not None:
      self._chain.append((self._cattr, None, None, True))
      self._cattr = None
    if obj is not Null:
      self.obj = obj
    elif self.obj is Null:
      raise SequentException('Attempted to resolve an empty Sequent.')

  def _process_chain(self):
    for link in self._chain:
      if link[3]:  # is an attribute
        attr, args, kwargs, _ = link
        self.obj = getattr(self.obj, attr)
        if args is not None:
          self.obj = self.obj(*args, **kwargs)
        yield None
      else:
        f_or_v, a_, _, _ = link
        if callable(f_or_v):
          self.obj = f_or_v(self.obj)
        else:
          self.obj = f_or_v
        yield a_

  def resolve(self, obj: Any | Callable = Null) -> Any:
    """ Resolve the Sequent """
    raise NotImplementedError

  def _run_try(self):
    raise NotImplementedError

  def _run_catch(self, e: Exception):
    raise NotImplementedError

  def _run_finally(self):
    raise NotImplementedError

  def reset(self):
    self.obj = None
    self.args = ()
    self.kwargs = {}
    self._original_obj = None
    self._chain = []
    self._catch_chain = []
    self._finally_chain = []
    self._cattr = None

  def then_sync(self, obj: Any | Callable) -> SequentBase:
    return self.then(obj, a=False)

  def then_async(self, obj: Any | Callable | Awaitable) -> SequentBase:
    return self.then(obj, a=True)

  def catch_sync(self, f: Callable) -> SequentBase:
    return self.catch(f, a=False)

  def catch_async(self, f: Callable) -> SequentBase:
    return self.catch(f, a=True)

  def finally_sync(self, f: Callable, *, with_obj: bool = False) -> SequentBase:
    return self.finally_(f, a=False, with_obj=with_obj)

  def finally_async(self, f: Callable, *, with_obj: bool = False) -> SequentBase:
    return self.finally_(f, a=True, with_obj=with_obj)

  #def __getattr__(self, attr: str):
  #    if self._cattr is None:
  #        self._cattr = attr
  #    else:
  #        self._chain.append((self._cattr, None, None, True))
  #        self._cattr = attr
  #    return self

  def __call__(self, *args, **kwargs):
    if self._cattr is None:
      return self.resolve(*args)
    else:
      self._chain.append((self._cattr, args, kwargs, True))
      self._cattr = None
    return self

  def __bool__(self):
    return self._resolved

  def __or__(self, other: Any):
    return self.then(other)

  @classmethod
  def _from_sequent(cls, sequent: SequentBase) -> SequentBase:
    new_sequent = SequentAsync(sequent.obj, *sequent.args, **sequent.kwargs)
    new_sequent._chain = sequent._chain
    new_sequent._catch_chain = sequent._catch_chain
    new_sequent._finally_chain = sequent._finally_chain
    new_sequent._is_awaitable = True
    sequent.reset()
    return new_sequent


class SequentSync(SequentBase):
  __slots__ = ()

  def resolve(self, obj: Any | Callable = Null) -> Any:
    """ Resolve the Sequent """
    self._process_resolve(obj)
    try:
      if self._resolved:
        raise SequentException('This Sequent is already resolved and cannot be used again.')
      return self._run_try()
    except Exception as e:
      self._run_catch(e)
      raise e
    finally:
      self._run_finally()
      self._resolved = True
      self.reset()

  def _run_try(self):
    if callable(self.obj):
      self.obj = self.obj(*self.args, **self.kwargs)
    if inspect.isawaitable(self.obj):
      # in some cases, it may be useful to pass a sync function to Sequent which later resolves into an
      # awaitable, usually due to the need to get the object inside the try-finally clause
      return SequentAsync._from_sequent(self).resolve()
    self._original_obj = self.obj
    collections.deque(self._process_chain(), maxlen=0)
    return self.obj

  def _run_catch(self, e: Exception):
    for (f, a) in self._catch_chain:
      f(e)

  def _run_finally(self):
    for (f, a, with_obj) in self._finally_chain:
      if with_obj:
        f(self._original_obj)
      else:
        f()


class SequentAsync(SequentBase):
  __slots__ = ()

  async def resolve(self, obj: Any | Callable = Null) -> Any:
    self._process_resolve(obj)
    try:
      if self._resolved:
        raise SequentException('This Sequent is already resolved and cannot be used again.')
      return await self._run_try()
    except Exception as e:
      await self._run_catch(e)
      raise e
    finally:
      await self._run_finally()
      self._resolved = True
      self.reset()

  async def _run_try(self) -> Any:
    if self._is_awaitable:
      self.obj = await self.obj
    else:
      if callable(self.obj):
        self.obj = self.obj(*self.args, **self.kwargs)
      if inspect.isawaitable(self.obj):
        self.obj = await self.obj
    self._original_obj = self.obj
    for a in self._process_chain():
      if is_async(self.obj, a):
        self.obj = await self.obj
    return self.obj

  async def _run_catch(self, e: Exception):
    for (f, a) in self._catch_chain:
      r = f(e)
      if is_async(r, a):
        await r

  async def _run_finally(self):
    for (f, a, with_obj) in self._finally_chain:
      if with_obj:
        r = f(self._original_obj)
      else:
        r = f()
      if is_async(r, a):
        await r

  def __await__(self):
    return self.resolve().__await__()
