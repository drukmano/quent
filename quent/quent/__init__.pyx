import warnings

from quent.helpers cimport isawaitable, Null, QuentException, _handle_exception, ensure_future
from quent.classes cimport Link, _FrozenChain
from quent.evaluate cimport EVAL_CALLABLE, EVAL_ATTR, evaluate_value
from quent.custom cimport _Generator, foreach, with_, build_conditional


cdef class Chain:
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
      bint is_void = self.root_link is None, ignore_finally = False, is_null = v is Null, reraise
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
          result = self._run_async(root_link, result=rv, rv=Null, cv=Null, idx=idx, is_void=is_void)
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
            result = self._run_async(link, result=result, rv=rv, cv=cv, idx=idx, is_void=is_void)
            if self._autorun:
              return ensure_future(result)
            return result
          if not link.ignore_result and result is not Null:
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
        result = evaluate_value(self.on_finally, cv=rv)
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
      result, reraise = _handle_exception(exc, self.except_links, link, rv, cv, idx)
      if isawaitable(result):
        await result
      if reraise:
        raise exc

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

  def freeze(self) -> _FrozenChain:
    self.finalize()
    return _FrozenChain(self._run)

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


cdef class run:
  """
    A replacement for `Chain.run()` when using pipe syntax

      Chain(f1).then(f2).run() == Chain(f1) | f2 | run()
      Chain().then(f2).run(f1) == Chain() | f2 | run(f1)
  """
  def __init__(self, __v=Null, *args, **kwargs):
    self.root_value = __v
    self.args = args
    self.kwargs = kwargs


# TODO how to declare `Type[Chain] cls` ?
cdef Chain from_list(object cls, tuple links):
  cdef object el
  cdef Chain seq = cls()
  for el in links:
    seq._then(Link(el))
  return seq
