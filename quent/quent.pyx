import sys
import asyncio
import inspect
import time
import types
import functools
import warnings
cimport cython
#from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from quent._internal import __QUENT_INTERNAL__
from quent.helpers cimport (
  _handle_exception, ensure_future, handle_return_exc, remove_self_frames_from_traceback, stringify_chain,
  modify_traceback
)
from quent.custom cimport (
  _Generator, foreach, with_, build_conditional, while_true, _Return, _Break, _InternalQuentException
)


cdef object _PipeCls
try:
  from pipe import Pipe as _PipeCls
except ImportError:
  _PipeCls = None

@cython.final
cdef class _Null:
  def __repr__(self):
    return '<Null>'

cdef _Null Null = _Null()
PyNull = Null

cdef tuple EMPTY_TUPLE = ()
cdef dict EMPTY_DICT = {}

# same impl. of types.CoroutineType but for Cython coroutines.
async def _py_coro(): pass
cdef object _cy_coro(): return _py_coro()
cdef object _coro = _cy_coro()
cdef type _c_coro_type = type(_coro)
_coro.close()  # Prevent ResourceWarning

cdef:
  type _PyCoroType = types.CoroutineType
  type _CyCoroType = _c_coro_type

cdef:
  int EVAL_CUSTOM_ARGS = 1001
  int EVAL_NO_ARGS = 1002
  int EVAL_CALLABLE = 1003
  int EVAL_LITERAL = 1004
  int EVAL_ATTR = 1005


async def _await_run(result, chain=None, link=None):
  try:
    return await result
  except BaseException as exc:
    if chain is not None and link is not None:
      modify_traceback(exc, chain, link)
    raise remove_self_frames_from_traceback()


async def _await_modify_traceback(result, chain, link):
  try:
    return await result
  except BaseException as exc:
    modify_traceback(exc, chain, link)
    raise remove_self_frames_from_traceback()


@cython.final
@cython.freelist(64)
cdef class Link:
  def __init__(
    self, object v, tuple args = None, dict kwargs = None, bint allow_literal = False, str fn_name = None,
    bint is_with_root = False, bint ignore_result = False, bint is_attr = False, bint is_fattr = False,
    object ogv = None,
  ):
    # TODO if we ever add `is_nested` support for _FrozenChain, _Generator, we can
    #  create `class _Quent` which they (and Chain) inherit from and just check
    #  `isinstance(v, _Quent)`; _Quent class will also have _Quent.chain property
    #  so we could mark `_Quent.chain.is_nested = True`.
    cdef bint is_chain_type = type(v) is Chain or type(v) is Cascade or type(v) is ChainAttr or type(v) is CascadeAttr
    if is_chain_type:
      self.is_chain = True
      (<Chain>v).is_nested = True
    else:
      self.is_chain = False
      if _PipeCls is not None and isinstance(v, _PipeCls):
        v = v.function
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.is_with_root = is_with_root
    self.ignore_result = ignore_result
    self.is_attr = is_attr
    self.is_fattr = is_fattr
    self.ogv = ogv
    self.fn_name = fn_name
    self.result = Null
    self.temp_args = None

    self.is_exception_handler = False
    self.exceptions = None
    self.raise_ = True
    self.next_link = None
    if args:
      if args[0] is ...:
        self.eval_code = EVAL_NO_ARGS
      else:
        if kwargs is None:
          self.kwargs = EMPTY_DICT
        self.eval_code = EVAL_CUSTOM_ARGS
    elif kwargs:
      if self.is_chain:
        # A Chain cannot be run with custom keyword arguments but without
        # at least one positional argument.
        self.eval_code = EVAL_NO_ARGS
      else:
        if args is None:
          self.args = EMPTY_TUPLE
        self.eval_code =  EVAL_CUSTOM_ARGS
    elif callable(v):
      self.eval_code =  EVAL_CALLABLE
    elif is_fattr:
      self.eval_code = EVAL_NO_ARGS
    elif is_attr:
      self.eval_code = EVAL_ATTR
    else:
      if not allow_literal:
        raise QuentException('Non-callable objects cannot be used with this method.')
      self.eval_code = EVAL_LITERAL


cdef object evaluate_value(Link link, object cv):
  cdef object v

  # Fast path for most common case: simple callable with single argument
  if link.eval_code == EVAL_CALLABLE and not link.is_chain and not link.is_attr:
    if cv is Null:
      return link.v()
    return link.v(cv)

  if link.is_chain:
    # we deliberately did not set `link.v = <Chain>v` in `Link.__init__` since
    # it adds an ugly, unreadable stack frame in case of an exception.
    if link.eval_code == EVAL_CALLABLE:
      return (<Chain>link.v)._run(cv, None, None, True)
    elif link.eval_code == EVAL_CUSTOM_ARGS:
      if not link.args or link.kwargs is None:
        # This is for extra safety to avoid segfault.
        return (<Chain>link.v)._run(Null, None, None, True)
      return (<Chain>link.v)._run(link.args[0], link.args[1:], link.kwargs, True)
    elif link.eval_code == EVAL_NO_ARGS:
      return (<Chain>link.v)._run(Null, None, None, True)
    else:
      raise QuentException(
        'Invalid evaluation code found for a nested chain.'
        'If you see this error then something has gone terribly wrong.'
      )

  elif link.eval_code == EVAL_CALLABLE:
    if cv is Null:
      return link.v()
    return link.v(cv)

  elif not link.is_attr:
    if link.eval_code == EVAL_CUSTOM_ARGS:
      if link.args is None or link.kwargs is None:
        # This is for extra safety to avoid segfault.
        return link.v()
      return link.v(*link.args, **link.kwargs)
    elif link.eval_code == EVAL_NO_ARGS:
      return link.v()
    else:
      return link.v

  else:
    v = getattr(cv, link.v)
    if link.eval_code == EVAL_NO_ARGS:
      return v()
    elif link.eval_code == EVAL_CUSTOM_ARGS:
      if link.args is None or link.kwargs is None:
        # This is for extra safety to avoid segfault.
        return link.v()
      return v(*link.args, **link.kwargs)
    else:
      return v


@cython.freelist(32)
cdef class Chain:
  def __init__(self, object __v = Null, *args, **kwargs):
    """
    Create a new Chain
    :param v: the root value of the chain
    :param args: arguments to pass to `v`
    :param kwargs: keyword-arguments to pass to `v`
    """
    self.init(__v, args, kwargs, False)

  cdef void init(self, object rv, tuple args, dict kwargs, bint is_cascade):
    self.is_cascade = is_cascade
    self._autorun = False
    self.uses_attr = False
    self.is_nested = False
    if rv is not Null:
      self.root_link = Link(rv, args, kwargs, True)
    else:
      self.root_link = None
    self.first_link = None
    self.current_link = None
    self.temp_root_link = None

    self.on_true = None
    self.on_finally_link = None
    self.current_conditional = None
    self.current_attr = None

  cdef void _then(self, Link link):
    self.finalize()
    if self.is_cascade:
      link.is_with_root = True
    if link.is_attr or link.is_fattr:
      self.uses_attr = True

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

  cdef object _run(self, object v, tuple args, dict kwargs, bint invoked_by_parent_chain):
    if not invoked_by_parent_chain and self.is_nested:
      raise QuentException('You cannot directly run a nested chain.')
    self.finalize()
    cdef:
      Link link = self.root_link
      Link exc_link
      object pv = Null, cv = Null, rv = Null, result = None, exc = None
      bint has_root_value = link is not None, is_root_value_override = v is not Null
      bint ignore_finally = False

    if self.temp_root_link is not None:
      self.temp_root_link = None

    if is_root_value_override:
      if has_root_value:
        raise QuentException('Cannot override the root value of a Chain.')
    elif not has_root_value:
      if self.uses_attr:
        raise QuentException('Cannot use attributes without a root value.')

    try:
      if is_root_value_override:
        self.temp_root_link = Link(v, args, kwargs, True)
        link = self.temp_root_link
        link.next_link = self.first_link
        has_root_value = True
      if has_root_value:
        rv = evaluate_value(link, Null)
        if iscoro(rv):
          ignore_finally = True
          result = self._run_async(link, cv=rv, rv=Null, pv=Null, has_root_value=has_root_value)
          if self._autorun:
            return ensure_future(result)
          return result
        cv = rv
        link.result = rv
        link = link.next_link
      else:
        link = self.first_link

      while link is not None:
        if link.ignore_result:
          if link.is_exception_handler:
            link = link.next_link
            continue
          pv = cv
        if link.is_with_root:
          cv = evaluate_value(link, rv)
        else:
          cv = evaluate_value(link, cv)
        if iscoro(cv):
          ignore_finally = True
          result = self._run_async(link, cv=cv, rv=rv, pv=pv, has_root_value=has_root_value)
          if self._autorun:
            return ensure_future(result)
          return result
        link.result = cv
        if link.ignore_result:
          cv = pv
        link = link.next_link

      if self.is_cascade:
        cv = rv
      if cv is Null:
        return None
      return cv

    except _Return as exc:
      # TODO we can probably improve performance (quite a bit) when using .break_() and .return_()
      #  by not throwing an exception and then catching it but just return a value _Break / _Return (like Null)
      return handle_return_exc(exc, self.is_nested)

    except BaseException as exc:
      if issubclass(type(exc), _InternalQuentException):
        if self.is_nested:
          raise
        try:
          raise QuentException(f'{type(exc)} cannot be used in this context.')
        except QuentException as exc_:
          exc = exc_
      exc_link = _handle_exception(exc, self, link)
      if exc_link is None:
        raise exc
      try:
        result = evaluate_value(exc_link, rv)
      except BaseException as exc_:
        modify_traceback(exc_, self, exc_link)
        raise exc_ from exc
      if iscoro(result):
        result = ensure_future(_await_modify_traceback(result, self, exc_link))
        warnings.warn(
          'An \'except\' callback has returned a coroutine, but the chain is in synchronous mode. '
          'It was therefore scheduled for execution in a new Task.',
          category=RuntimeWarning
        )
      if exc_link.raise_:
        raise exc
      if result is Null:
        return None
      return result

    finally:
      if not ignore_finally and self.on_finally_link is not None:
        try:
          result = evaluate_value(self.on_finally_link, rv)
        except BaseException as exc_:
          modify_traceback(exc_, self, self.on_finally_link)
          raise exc_
        if iscoro(result):
          ensure_future(_await_modify_traceback(result, self, self.on_finally_link))
          warnings.warn(
            'The \'finally\' callback has returned a coroutine, but the chain is in synchronous mode. '
            'It was therefore scheduled for execution in a new Task.',
            category=RuntimeWarning
          )

  async def _run_async(self, Link link, object cv, object rv, object pv, bint has_root_value):
    cdef:
      Link exc_link
      object exc, result

    try:
      cv = await cv
      link.result = cv
      if link.ignore_result:
        cv = pv
      # update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `rv` should always be `Null`.
      if has_root_value and rv is Null:
        rv = cv

      link = link.next_link
      while link is not None:
        if link.ignore_result:
          if link.is_exception_handler:
            link = link.next_link
            continue
          pv = cv
        if link.is_with_root:
          cv = evaluate_value(link, rv)
        else:
          cv = evaluate_value(link, cv)
        if iscoro(cv):
          cv = await cv
        link.result = cv
        if link.ignore_result:
          cv = pv
        link = link.next_link

      if self.is_cascade:
        cv = rv
      if cv is Null:
        return None
      return cv

    except _Return as exc:
      result = handle_return_exc(exc, self.is_nested)
      if iscoro(result):
        return await result
      return result

    except BaseException as exc:
      if issubclass(type(exc), _InternalQuentException):
        if self.is_nested:
          raise
        try:
          raise QuentException(f'{type(exc)} cannot be used in this context.')
        except QuentException as exc_:
          exc = exc_
      exc_link = _handle_exception(exc, self, link)
      if exc_link is None:
        raise exc
      try:
        result = evaluate_value(exc_link, rv)
        if iscoro(result):
          result = await result
      except BaseException as exc_:
        modify_traceback(exc_, self, exc_link)
        raise exc_ from exc
      if exc_link.raise_:
        raise exc
      if result is Null:
        return None
      return result

    finally:
      if self.on_finally_link is not None:
        try:
          result = evaluate_value(self.on_finally_link, rv)
          if iscoro(result):
            await result
        except BaseException as exc_:
          modify_traceback(exc_, self, self.on_finally_link)
          raise exc_

  def config(self, *, object autorun = None, object debug = None):
    if autorun is not None:
      self._autorun = bool(autorun)
    # TODO
    #if debug is not None:
    #  self._debug = bool(debug)
    return self

  def autorun(self, bint autorun = True):
    self._autorun = bool(autorun)
    return self

  #def clone(self):
  #  self.finalize()
  #  # TODO a proper clone requires cloning all Links as well
  #  return self.__class__()._clone(
  #    self.root_link, self.first_link, self.current_link, self.is_cascade, self._autorun, self.except_links,
  #    self.on_finally_link
  #  )

  def freeze(self):
    # shallow freeze; does not actually prevent modification of the chain itself.
    # we need `_FrozenChain(self.clone()._run)` for that.
    self.finalize()
    return _FrozenChain(self._run)

  def decorator(self):
    return self.freeze().decorator()

  def run(self, object __v = Null, *args, **kwargs):
    try:
      result = self._run(__v, args, kwargs, False)
      if iscoro(result):
        return _await_run(result)
      return result
    except BaseException:
      raise remove_self_frames_from_traceback()

  def then(self, object __v, *args, **kwargs):
    self._then(Link(__v, args, kwargs, True, 'then'))
    return self

  def do(self, object __fn, *args, **kwargs):
    # register a value to be evaluated but will not propagate its result forwards.
    self._then(Link(__fn, args, kwargs, fn_name='do', ignore_result=True))
    return self

  def root(self, object __fn, *args, **kwargs):
    self._then(Link(__fn, args, kwargs, fn_name='root', is_with_root=True))
    return self

  def root_do(self, object __fn, *args, **kwargs):
    self._then(Link(__fn, args, kwargs, fn_name='root_do', is_with_root=True, ignore_result=True))
    return self

  def attr(self, str name):
    self._then(Link(name, fn_name='attr', is_attr=True))
    return self

  def attr_fn(self, str __name, *args, **kwargs):
    self._then(Link(__name, args, kwargs, fn_name='attr_fn', is_attr=True, is_fattr=True))
    return self

  def except_(self, object __fn, *args, object exceptions = None, bint raise_ = True, **kwargs):
    cdef Link link = Link(__fn, args, kwargs, fn_name='except_')
    link.ignore_result = True
    link.is_exception_handler = True
    link.exceptions = exceptions
    link.raise_ = raise_
    self._then(link)
    return self

  def finally_(self, object __fn, *args, **kwargs):
    if self.on_finally_link is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally_link = Link(__fn, args, kwargs, fn_name='finally_')
    return self

  def while_true(self, object __fn, *args, **kwargs):
    self._then(while_true(__fn, args, kwargs))
    return self

  def iterate(self, object fn = None):
    return _Generator(self._run, fn, _ignore_result=False)

  def iterate_do(self, object fn = None):
    return _Generator(self._run, fn, _ignore_result=True)

  def foreach(self, object fn):
    self._then(foreach(fn, ignore_result=False))
    return self

  def foreach_do(self, object fn):
    self._then(foreach(fn, ignore_result=True))
    return self

  def with_(self, object __fn, *args, **kwargs):
    self._then(with_(__fn, args, kwargs, ignore_result=False))
    return self

  def with_do(self, object __fn, *args, **kwargs):
    self._then(with_(__fn, args, kwargs, ignore_result=True))
    return self

  def if_(self, object __v, *args, **kwargs):
    self._if(Link(__v, args, kwargs, True, fn_name='if_'))
    return self

  def else_(self, object __v, *args, **kwargs):
    self._else(Link(__v, args, kwargs, True, fn_name='else_'))
    return self

  def if_not(self, object __v, *args, **kwargs):
    self._if(Link(__v, args, kwargs, True, fn_name='if_not'), not_=True)
    return self

  def if_raise(self, object exc):
    def if_raise(object cv): raise exc
    self._if(Link(if_raise, fn_name='if_raise', ogv=exc))
    return self

  def else_raise(self, object exc):
    def else_raise(object cv): raise exc
    self._else(Link(else_raise, fn_name='else_raise', ogv=exc))
    return self

  def if_not_raise(self, object exc):
    def if_not_raise(object cv): raise exc
    self._if(Link(if_not_raise, fn_name='if_not_raise', ogv=exc), not_=True)
    return self

  def condition(self, object __fn, *args, **kwargs):
    self.set_conditional(Link(__fn, args, kwargs, fn_name='condition'), custom=True)
    return self

  def not_(self):
    def not_(object cv): return not cv
    self.set_conditional(Link(not_, fn_name='not_'))
    return self

  def eq(self, object value):
    def eq(object cv): return cv == value
    self.set_conditional(Link(eq, fn_name='eq', ogv=value))
    return self

  def neq(self, object value):
    def neq(object cv): return cv != value
    self.set_conditional(Link(neq, fn_name='neq', ogv=value))
    return self

  def is_(self, object value):
    def is_(object cv): return cv is value
    self.set_conditional(Link(is_, fn_name='is_', ogv=value))
    return self

  def is_not(self, object value):
    def is_not(object cv): return cv is not value
    self.set_conditional(Link(is_not, fn_name='is_not', ogv=value))
    return self

  def in_(self, object value):
    def in_(object cv): return cv in value
    self.set_conditional(Link(in_, fn_name='in_', ogv=value))
    return self

  def not_in(self, object value):
    def not_in(object cv): return cv not in value
    self.set_conditional(Link(not_in, fn_name='not_in', ogv=value))
    return self

  def or_(self, object value):
    def or_(object cv): return cv or value
    self._then(Link(or_, fn_name='or_', ogv=value))
    return self

  def raise_(self, object exc):
    def raise_(object cv): raise exc
    self._then(Link(raise_, fn_name='raise_', ogv=exc))
    return self

  def sleep(self, float delay):
    def sleep(object cv): return asyncio.sleep(delay) if asyncio._get_running_loop() else time.sleep(delay)
    self._then(Link(sleep, fn_name='sleep', ogv=delay, ignore_result=True))
    return self

  @classmethod
  def null(cls):
    return Null

  @classmethod
  def return_(cls, object __v = Null, *args, **kwargs):
    raise _Return(__v, args, kwargs)

  @classmethod
  def break_(cls, object __v = Null, *args, **kwargs):
    raise _Break(__v, args, kwargs)

  cdef void _if(self, Link on_true, bint not_ = False):
    if self.current_conditional is None:
      self.current_conditional = (None, False)
    self.on_true = (on_true, not_)

  cdef void _else(self, Link on_false):
    if self.on_true is None:
      raise QuentException(
        'You cannot use \'.else_()\' without a preceding \'.if_()\' or \'.if_not()\''
      )
    self.finalize_conditional(on_false)

  cdef void set_conditional(self, Link conditional, bint custom = False):
    self.finalize()
    self.current_conditional = (conditional, custom)

  cdef void finalize(self):
    cdef str attr
    if self.current_conditional is not None:
      self.finalize_conditional()
    # TODO separate this into a overriding function in ChainAttr / CascadeAttr
    elif self.current_attr is not None:
      attr = self.current_attr
      self.current_attr = None
      self._then(Link(attr, is_attr=True))

  cdef void finalize_conditional(self, Link on_false = None):
    cdef:
      Link conditional
      Link on_true
      bint is_custom, not_
    conditional, is_custom = self.current_conditional
    self.current_conditional = None
    if self.on_true:
      on_true, not_ = self.on_true
      self.on_true = None
      build_conditional(self, conditional, is_custom, not_, on_true, on_false)
    else:
      self._then(conditional)

  #def _clone(
  #  self, Link root_link, Link first_link, Link current_link, bint is_cascade, bint _autorun, list except_links,
  #  Link on_finally_link
  #):
  #  # TODO find how to iterate the class attributes (like __slots__ / __dict__, but Cython classes does not implement
  #  #  those)
  #  self.root_link = root_link
  #  self.first_link = first_link
  #  self.current_link = current_link
  #  self.is_cascade = is_cascade
  #  self._autorun = _autorun
  #  self.except_links = None if except_links is None else except_links.copy()
  #  self.on_finally_link = on_finally_link
  #  return self

  def __or__(self, other):
    if isinstance(other, run):
      return self._run(other.root_value, other.args, other.kwargs, False)
    self._then(Link(other, None, None, True))
    return self

  def __call__(self, object __v = Null, *args, **kwargs):
    return self._run(__v, args, kwargs, False)

  # while this may be nice to have, I fear that it will cause troubles as
  # people might forget to call `.run()` when dealing with non-async code (or
  # code that could be both but is not known to the one creating the chain).
  #def __await__(self):
  #  return self._run(Null, None, None).__await__()

  def __bool__(self):
    return True

  def __repr__(self):
    self.finalize()
    return stringify_chain(self)[0]


cdef class Cascade(Chain):
  # TODO mark all cascade items as `ignore_result=True` to (marginally) increase performance.

  # noinspection PyMissingConstructor
  def __init__(self, object __v = Null, *args, **kwargs):
    self.init(__v, args, kwargs, True)


cdef class ChainAttr(Chain):
  def __getattr__(self, str name):
    self.finalize()
    self.current_attr = name
    return self

  def __call__(self, *args, **kwargs):
    cdef str attr = self.current_attr
    if attr is None:
      # much slower than directly calling `._run()`, but we have no choice since
      # we wish support arbitrary __call__ invocations on attributes.
      # avoid running a chain this way. opt to use `.run()` instead.
      return self.run(*args, **kwargs)
    else:
      self.current_attr = None
      self._then(Link(attr, args, kwargs, is_attr=True, is_fattr=True))
      return self


# cannot have multiple inheritance in Cython.
cdef class CascadeAttr(ChainAttr):
  # noinspection PyMissingConstructor
  def __init__(self, object __v = Null, *args, **kwargs):
    self.init(__v, args, kwargs, True)


@cython.final
@cython.freelist(4)
cdef class _FrozenChain:
  def decorator(self):
    cdef object _chain_run = self._chain_run
    def _decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        return _chain_run(fn, args, kwargs, False)
      return wrapper
    return _decorator

  def __init__(self, object _chain_run):
    self._chain_run = _chain_run

  def run(self, object __v = Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs, False)

  def __call__(self, object __v = Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs, False)


@cython.final
@cython.freelist(4)
cdef class run:
  """
    A replacement for `Chain.run()` when using pipe syntax

      Chain(f1).then(f2).run() == Chain(f1) | f2 | run()
      Chain().then(f2).run(f1) == Chain() | f2 | run(f1)
  """
  def __init__(self, object __v = Null, *args, **kwargs):
    self.root_value = __v
    self.args = args
    self.kwargs = kwargs
