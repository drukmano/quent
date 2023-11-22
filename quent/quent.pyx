import functools
import warnings

from quent.helpers cimport isawaitable, Null, QuentException, _handle_exception, ensure_future
from quent.custom cimport _Generator, foreach, with_, build_conditional
from quent.link cimport Link, evaluate_value


cdef class Chain:
  cdef:
    Link root_link, on_finally
    list links, except_links
    bint is_cascade, _autorun
    tuple current_conditional, on_true
    str current_attr

  def __init__(self, object __v = Null, *args, **kwargs):
    """
    Create a new Chain
    :param v: the root value of the chain
    :param args: arguments to pass to `v`
    :param kwargs: keyword-arguments to pass to `v`
    """
    self.init(__v, args, kwargs, False)

  cdef int init(self, object rv, tuple args, dict kwargs, bint is_cascade) except -1:
    self.is_cascade = is_cascade
    self._autorun = False
    self.links = []
    if rv is not Null:
      self.root_link = Link(rv, args, kwargs, True)
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
      Link root_link = self.root_link, link = root_link
      list links = self.links
      object cv = Null, rv = Null, result, exc
      bint is_void = root_link is None, ignore_finally = False, is_null = v is Null, reraise
      int idx = -1

    if not is_void and not is_null:
      raise QuentException('Cannot override the root value of a Chain.')
    elif is_void and is_null and not self.is_cascade:
      raise QuentException('Cannot run a Chain without a root value. Use Cascade for that.')

    try:
      # this condition is False only for a void Cascade.
      if not (is_void and is_null):
        if root_link is None:
          root_link = link = Link(v, args, kwargs, True)
          is_void = False
        rv = cv = evaluate_value(root_link, Null)
        if isawaitable(rv):
          ignore_finally = True
          result = self._run_async(root_link, result=rv, rv=Null, cv=Null, idx=idx, is_void=is_void)
          if self._autorun:
            return ensure_future(result)
          return result

      for link in links:
        idx += 1
        if link.is_attr and is_void:
          raise QuentException('Cannot use attributes without a root value.')

        result = evaluate_value(link, rv if link.is_with_root else cv)
        if isawaitable(result):
          ignore_finally = True
          result = self._run_async(link, result=result, rv=rv, cv=cv, idx=idx, is_void=is_void)
          if self._autorun:
            return ensure_future(result)
          return result
        if not link.ignore_result: # TODO remove; and result is not Null:
          cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as exc:
      result, reraise = _handle_exception(exc, self.except_links, link, rv, cv, idx)
      if isawaitable(result):
        ensure_future(result)
        warnings.warn(
          'An \'except\' callback has returned a coroutine, but the chain is in synchronous mode. '
          'It was therefore scheduled for execution in a new Task.',
          category=RuntimeWarning
        )
      if reraise:
        raise exc

    finally:
      if not ignore_finally and self.on_finally is not None:
        result = evaluate_value(self.on_finally, rv)
        if isawaitable(result):
          ensure_future(result)
          warnings.warn(
            'The \'finally\' callback has returned a coroutine, but the chain is in synchronous mode. '
            'It was therefore scheduled for execution in a new Task.',
            category=RuntimeWarning
          )

  async def _run_async(self, Link link, object result, object rv, object cv, int idx, bint is_void):
    cdef:
      object exc
      list links = self.links
      bint reraise

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

        result = evaluate_value(link, rv if link.is_with_root else cv)
        if isawaitable(result):
          result = await result
        if not link.ignore_result: # TODO remove; and result is not Null:
          cv = result

      if self.is_cascade:
        cv = rv
      return cv if cv is not Null else None

    except Exception as exc:
      result, reraise = _handle_exception(exc, self.except_links, link, rv, cv, idx)
      if isawaitable(result):
        await result
      if reraise:
        raise exc

    finally:
      if self.on_finally is not None:
        result = evaluate_value(self.on_finally, rv)
        if isawaitable(result):
          await result

  def config(self, *, object autorun = None):
    if autorun is not None:
      self._autorun = bool(autorun)
    return self

  def autorun(self, bint autorun = True):
    self._autorun = bool(autorun)
    return self

  def clone(self):
    self.finalize()
    return self.__class__()._clone(
      self.root_link, self.is_cascade, self._autorun, self.links, self.except_links, self.on_finally
    )

  def freeze(self):
    self.finalize()
    return _FrozenChain(self._run)

  def decorator(self):
    return self.freeze().decorator()

  def run(self, object __v = Null, *args, **kwargs):
    return self._run(__v, args, kwargs)

  def then(self, object __v, *args, **kwargs):
    self._then(Link(__v, args, kwargs, True))
    return self

  def do(self, object __fn, *args, **kwargs):
    # register a value to be evaluated but will not propagate its result forwards.
    self._then(Link(__fn, args, kwargs, ignore_result=True))
    return self

  def root(self, object __fn, *args, **kwargs):
    self._then(Link(__fn, args, kwargs, is_with_root=True))
    return self

  def root_do(self, object __fn, *args, **kwargs):
    self._then(Link(__fn, args, kwargs, is_with_root=True, ignore_result=True))
    return self

  def attr(self, str name):
    self._then(Link(name, is_attr=True))
    return self

  def attr_fn(self, str __name, *args, **kwargs):
    self._then(Link(__name, args, kwargs, is_fattr=True))
    return self

  def except_(self, object __fn, *args, object exceptions = None, bint raise_ = True, **kwargs):
    self.except_links.append((Link(__fn, args, kwargs), exceptions, raise_))
    return self

  def finally_(self, object __fn, *args, **kwargs):
    if self.on_finally is not None:
      raise QuentException('You can only register one \'finally\' callback.')
    self.on_finally = Link(__fn, args, kwargs)
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
    self._then(with_(Link(__fn, args, kwargs), ignore_result=False))
    return self

  def with_do(self, object __fn, *args, **kwargs):
    self._then(with_(Link(__fn, args, kwargs), ignore_result=True))
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
    cdef Link link = Link(__fn, args, kwargs)
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
    self._then(Link(or_))
    return self

  def raise_(self, object exc):
    def raise_(object cv): raise exc
    self._then(Link(raise_))
    return self

  cdef int _if(self, object on_true, tuple args = None, dict kwargs = None, bint not_ = False) except -1:
    if self.current_conditional is None:
      self.current_conditional = (bool, False)
    self.on_true = (Link(on_true, args, kwargs, True), not_)

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
      self._then(Link(attr, is_attr=True))
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
        on_false_link = Link(on_false, args, kwargs, True)
      self.on_true = None
      self._then(build_conditional(conditional, is_custom, not_, on_true_link, on_false_link))
    else:
      self._then(Link(conditional))

  def _clone(
    self, Link root_link, bint is_cascade, bint _autorun, list links, list except_links, Link on_finally
  ):
    # TODO find how to iterate the class attributes (like __slots__ / __dict__, but Cython classes does not implement
    #  those)
    self.root_link = root_link
    self.is_cascade = is_cascade
    self._autorun = _autorun
    self.links = links.copy()
    self.except_links = except_links.copy()
    self.on_finally = on_finally
    return self

  def __or__(self, other):
    if isinstance(other, run):
      return self._run(other.root_value, other.args, other.kwargs)
    self._then(Link(other, None, None, True))
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

    if root_link is not None:
      s += f'({root_link.v}, {root_link.args}, {root_link.kwargs})'
    else:
      s += '()'
    s += f'({len(self.links)} links)>'
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
      self._then(Link(attr, args, kwargs, is_fattr=True))
      return self


# cannot have multiple inheritance in Cython.
cdef class CascadeAttr(ChainAttr):
  # noinspection PyMissingConstructor
  def __init__(self, object __v = Null, *args, **kwargs):
    self.init(__v, args, kwargs, True)


cdef class _FrozenChain:
  cdef object _chain_run

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
  cdef public:
    object root_value
    tuple args
    dict kwargs

  """
    A replacement for `Chain.run()` when using pipe syntax

      Chain(f1).then(f2).run() == Chain(f1) | f2 | run()
      Chain().then(f2).run(f1) == Chain() | f2 | run(f1)
  """
  def __init__(self, object __v = Null, *args, **kwargs):
    self.root_value = __v
    self.args = args
    self.kwargs = kwargs
