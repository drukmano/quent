import types
import functools
import warnings
cimport cython
#from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from quent.helpers cimport _handle_exception, ensure_future
from quent.custom cimport (
  _Generator, foreach, with_, build_conditional, while_true, _Return, _Break, _Continue, _InternalQuentException
)


cdef object _PipeCls
try:
  from pipe import Pipe as _PipeCls
except ImportError:
  _PipeCls = None

cdef class _Null:
  def __repr__(self):
    return '<Null>'

cdef _Null Null = _Null()
PyNull = Null

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


@cython.freelist(32)
cdef class Link:
  def __cinit__(
    self, object v, tuple args = None, dict kwargs = None, bint allow_literal = False, bint is_with_root = False,
    bint ignore_result = False, bint is_attr = False, bint is_fattr = False
  ):
    if _PipeCls is not None and isinstance(v, _PipeCls):
      v = v.function
    self.v = v
    self.args = args
    self.kwargs = kwargs
    self.is_with_root = is_with_root
    self.ignore_result = ignore_result
    self.is_attr = is_attr
    self.is_fattr = is_fattr
    self.next_link = None
    if bool(args):
      if args[0] is ...:
        self.eval_code = EVAL_NO_ARGS
      else:
        self.eval_code = EVAL_CUSTOM_ARGS
    elif bool(kwargs):
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
  if link.eval_code == EVAL_CALLABLE:
    if cv is Null:
      return link.v()
    else:
      return link.v(cv)

  elif link.is_attr:
    v = getattr(cv, link.v)
    if link.eval_code == EVAL_NO_ARGS:
      return v()
    elif link.eval_code == EVAL_CUSTOM_ARGS:
      return v(*link.args, **link.kwargs)
    return v

  else:
    if link.eval_code == EVAL_CUSTOM_ARGS:
      # it is dangerous if one of those will be `None`, but it shouldn't be possible
      # as we only specify both or none.
      return link.v(*link.args, **link.kwargs)
    elif link.eval_code == EVAL_NO_ARGS:
      return link.v()
    else:
      return link.v


@cython.freelist(8)
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
    if rv is not Null:
      self.root_link = Link.__new__(Link, rv, args, kwargs, True)
    else:
      self.root_link = None
    self.first_link = None
    self.current_link = None

    self.except_links = None
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
      if self.root_link is not None and self.root_link.next_link is None:
        self.root_link.next_link = link

  cdef object _run(self, object v, tuple args, dict kwargs):
    self.finalize()
    cdef:
      Link link = self.root_link
      object pv = Null, cv = Null, rv = Null, result = None, exc = None
      bint has_root_value = link is not None, is_root_value_override = v is not Null
      bint reraise, return_except_result, ignore_finally = False
      int idx = -1
    cdef object v_, cv_
    cdef int eval_code

    if is_root_value_override:
      if has_root_value:
        raise QuentException('Cannot override the root value of a Chain.')
    elif not has_root_value:
      if self.uses_attr:
        raise QuentException('Cannot use attributes without a root value.')

    try:
      if is_root_value_override:
        link = Link.__new__(Link, v, args, kwargs, True)
        link.next_link = self.first_link
        has_root_value = True
      if has_root_value:
        rv = evaluate_value(link, Null)
        if iscoro(rv):
          ignore_finally = True
          result = self._run_async(link, cv=rv, rv=Null, pv=Null, idx=idx, has_root_value=has_root_value)
          if self._autorun:
            return ensure_future(result)
          return result
        cv = rv
        link = link.next_link
      else:
        link = self.first_link

      if link is not None:
        while True:
          idx += 1
          if link.ignore_result:
            pv = cv
          if link.is_with_root:
            cv = evaluate_value(link, rv)
          else:
            cv = evaluate_value(link, cv)
          if iscoro(cv):
            ignore_finally = True
            result = self._run_async(link, cv=cv, rv=rv, pv=pv, idx=idx, has_root_value=has_root_value)
            if self._autorun:
              return ensure_future(result)
            return result
          if link.ignore_result:
            cv = pv
          if link.next_link is None:
            break
          link = link.next_link

      if self.is_cascade:
        cv = rv
      if cv is Null:
        return None
      return cv

    except _InternalQuentException:
      raise  # passthrough

    except Exception as exc:
      result, reraise, return_except_result = _handle_exception(exc, self.except_links, link, rv, cv, idx)
      if iscoro(result):
        result = ensure_future(result)
        warnings.warn(
          'An \'except\' callback has returned a coroutine, but the chain is in synchronous mode. '
          'It was therefore scheduled for execution in a new Task.',
          category=RuntimeWarning
        )
        # we cannot check if result is Null here.
        if return_except_result:
          return result
      elif return_except_result and result is not Null:
        return result
      if reraise:
        raise exc

    finally:
      if not ignore_finally and self.on_finally_link is not None:
        result = evaluate_value(self.on_finally_link, rv)
        if iscoro(result):
          ensure_future(result)
          warnings.warn(
            'The \'finally\' callback has returned a coroutine, but the chain is in synchronous mode. '
            'It was therefore scheduled for execution in a new Task.',
            category=RuntimeWarning
          )

  async def _run_async(self, Link link, object cv, object rv, object pv, int idx, bint has_root_value):
    cdef:
      object exc, result
      bint reraise, return_except_result

    try:
      cv = await cv
      if link.ignore_result:
        cv = pv
      # update the root value only if this is not a void chain, since otherwise
      # if this is a void chain, `rv` should always be `Null`.
      if has_root_value and rv is Null:
        rv = cv

      link = link.next_link
      if link is not None:
        while True:
          idx += 1
          if link.ignore_result:
            pv = cv
          if link.is_with_root:
            cv = evaluate_value(link, rv)
          else:
            cv = evaluate_value(link, cv)
          if iscoro(cv):
            cv = await cv
          if link.ignore_result:
            cv = pv
          if link.next_link is None:
            break
          link = link.next_link

      if self.is_cascade:
        cv = rv
      if cv is Null:
        return None
      return cv

    except _InternalQuentException:
      raise  # passthrough

    except Exception as exc:
      result, reraise, return_except_result = _handle_exception(exc, self.except_links, link, rv, cv, idx)
      if iscoro(result):
        result = await result
      if return_except_result and result is not Null:
        return result
      if reraise:
        raise exc

    finally:
      if self.on_finally_link is not None:
        result = evaluate_value(self.on_finally_link, rv)
        if iscoro(result):
          await result

  def config(self, *, object autorun = None):
    if autorun is not None:
      self._autorun = bool(autorun)
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
    self.finalize()
    return _FrozenChain(self._run)

  def decorator(self):
    return self.freeze().decorator()

  def run(self, object __v = Null, *args, **kwargs):
    return self._run(__v, args, kwargs)

  def then(self, object __v, *args, **kwargs):
    self._then(Link.__new__(Link, __v, args, kwargs, True))
    return self

  def do(self, object __fn, *args, **kwargs):
    # register a value to be evaluated but will not propagate its result forwards.
    self._then(Link.__new__(Link, __fn, args, kwargs, ignore_result=True))
    return self

  def root(self, object __fn, *args, **kwargs):
    self._then(Link.__new__(Link, __fn, args, kwargs, is_with_root=True))
    return self

  def root_do(self, object __fn, *args, **kwargs):
    self._then(Link.__new__(Link, __fn, args, kwargs, is_with_root=True, ignore_result=True))
    return self

  def attr(self, str name):
    self._then(Link.__new__(Link, name, is_attr=True))
    return self

  def attr_fn(self, str __name, *args, **kwargs):
    self._then(Link.__new__(Link, __name, args, kwargs, is_attr=True, is_fattr=True))
    return self

  def except_(self, object __fn, *args, object exceptions = None, bint raise_ = True, bint return_ = False, **kwargs):
    if self.except_links is None:
      self.except_links = []
    self.except_links.append((Link.__new__(Link, __fn, args, kwargs), exceptions, raise_, return_))
    return self

  def finally_(self, object __fn, *args, **kwargs):
    if self.on_finally_link is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally_link = Link.__new__(Link, __fn, args, kwargs)
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
    self._then(with_(Link.__new__(Link, __fn, args, kwargs), ignore_result=False))
    return self

  def with_do(self, object __fn, *args, **kwargs):
    self._then(with_(Link.__new__(Link, __fn, args, kwargs), ignore_result=True))
    return self

  def if_(self, object __v, *args, **kwargs):
    self._if(__v, args, kwargs)
    return self

  def else_(self, object __v, *args, **kwargs):
    self._else(__v, args, kwargs)
    return self

  def if_not(self, object __v, *args, **kwargs):
    self._if(__v, args, kwargs, not_=True)
    return self

  def if_raise(self, object exc):
    def if_raise(object cv): raise exc
    self._if(if_raise)
    return self

  def else_raise(self, object exc):
    def else_raise(object cv): raise exc
    self._else(else_raise)
    return self

  def if_not_raise(self, object exc):
    def if_not_raise(object cv): raise exc
    self._if(if_not_raise, None, None, not_=True)
    return self

  def condition(self, object __fn, *args, **kwargs):
    cdef Link link = Link.__new__(Link, __fn, args, kwargs)
    def condition(object cv):
      return evaluate_value(link, cv)
    self.set_conditional(condition, custom=True)
    return self

  def not_(self):
    # use named functions (instead of a lambda) to have more details in the exception stacktrace
    def not_(object cv) -> bool: return not cv
    self.set_conditional(not_)
    return self

  def eq(self, object value):
    def equals(object cv) -> bool: return cv == value
    self.set_conditional(equals)
    return self

  def neq(self, object value):
    def not_equals(object cv) -> bool: return cv != value
    self.set_conditional(not_equals)
    return self

  def is_(self, object value):
    def is_(object cv) -> bool: return cv is value
    self.set_conditional(is_)
    return self

  def is_not(self, object value):
    def is_not(object cv) -> bool: return cv is not value
    self.set_conditional(is_not)
    return self

  def in_(self, object value):
    def in_(object cv) -> bool: return cv in value
    self.set_conditional(in_)
    return self

  def not_in(self, object value):
    def not_in(object cv) -> bool: return cv not in value
    self.set_conditional(not_in)
    return self

  def or_(self, object value):
    def or_(object cv): return cv or value
    self._then(Link.__new__(Link, or_))
    return self

  def raise_(self, object exc):
    def raise_(object cv): raise exc
    self._then(Link.__new__(Link, raise_))
    return self

  @classmethod
  def null(cls):
    return Null

  @classmethod
  def return_(cls, object value = Null):
    raise _Return(value)

  @classmethod
  def break_(cls):
    raise _Break

  @classmethod
  def continue_(cls):
    raise _Continue

  cdef void _if(self, object on_true, tuple args = None, dict kwargs = None, bint not_ = False):
    if self.current_conditional is None:
      self.current_conditional = (bool, False)
    self.on_true = (Link.__new__(Link, on_true, args, kwargs, True), not_)

  cdef void _else(self, object on_false, tuple args = None, dict kwargs = None):
    if self.on_true is None:
      raise QuentException(
        'You cannot use \'.else_()\' without a preceding \'.if_()\' or \'.if_not()\''
      )
    self.finalize_conditional(on_false, args, kwargs)

  cdef void set_conditional(self, object conditional, bint custom = False):
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
      self._then(Link.__new__(Link, attr, is_attr=True))

  cdef void finalize_conditional(self, object on_false = Null, tuple args = None, dict kwargs = None):
    cdef:
      object conditional
      Link on_true_link, on_false_link = None
      bint is_custom, not_
    conditional, is_custom = self.current_conditional
    self.current_conditional = None
    if self.on_true:
      on_true_link, not_ = self.on_true
      if on_false is not Null:
        on_false_link = Link.__new__(Link, on_false, args, kwargs, True)
      self.on_true = None
      self._then(build_conditional(conditional, is_custom, not_, on_true_link, on_false_link))
    else:
      self._then(Link.__new__(Link, conditional))

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
      return self._run(other.root_value, other.args, other.kwargs)
    self._then(Link.__new__(Link, other, None, None, True))
    return self

  def __call__(self, object __v = Null, *args, **kwargs):
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

    # TODO format the chain with proper chain actions, function names, and arguments
    if root_link is not None:
      s += f'({root_link.v}, {root_link.args}, {root_link.kwargs})'
    else:
      s += '()'
    #s += f'({len(self.links)} links)>'
    return s


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
      self._then(Link.__new__(Link, attr, args, kwargs, is_attr=True, is_fattr=True))
      return self


# cannot have multiple inheritance in Cython.
cdef class CascadeAttr(ChainAttr):
  # noinspection PyMissingConstructor
  def __init__(self, object __v = Null, *args, **kwargs):
    self.init(__v, args, kwargs, True)


cdef class _FrozenChain:
  def decorator(self):
    cdef object _chain_run = self._chain_run
    def _decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        return _chain_run(fn, args, kwargs)
      return wrapper
    return _decorator

  def __init__(self, object _chain_run):
    self._chain_run = _chain_run

  def run(self, object __v = Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs)

  def __call__(self, object __v = Null, *args, **kwargs):
    return self._chain_run(__v, args, kwargs)


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
