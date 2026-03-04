# PERF: Pre-built default args tuple avoids per-call allocation. See PERFORMANCE.md #6.
cdef tuple _DEFAULT_RUN_ARGS = (Null, EMPTY_TUPLE, EMPTY_DICT, False)

def sync_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result
  try:
    for el in iterator_getter(*run_args):
      if fn is None:
        yield el
      else:
        result = fn(el)
        if ignore_result:
          yield el
        else:
          yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


async def async_generator(object iterator_getter, tuple run_args, object fn, bint ignore_result):
  cdef object el, result, iterator
  cdef bint is_aiter
  iterator = iterator_getter(*run_args)
  if iscoro(iterator):
    iterator = await iterator
  is_aiter = hasattr(iterator, '__aiter__')
  try:
    if is_aiter:
      async for el in iterator:
        if fn is None:
          yield el
        else:
          result = fn(el)
          if iscoro(result):
            result = await result
          if ignore_result:
            yield el
          else:
            yield result
    else:
      for el in iterator:
        if fn is None:
          yield el
        else:
          result = fn(el)
          if iscoro(result):
            result = await result
          if ignore_result:
            yield el
          else:
            yield result
  except _Break:
    return
  except _Return:
    raise QuentException('Using `.return_()` inside an iterator is not allowed.')


# PERF: @cython.final enables direct C dispatch. @cython.freelist(4) pools allocations.
# See PERFORMANCE.md #1, #2.
@cython.final
@cython.freelist(4)
cdef class _Generator:
  def __init__(self, object _chain_run, object _fn, bint _ignore_result):
    self._chain_run = _chain_run
    self._fn = _fn
    self._ignore_result = _ignore_result
    self._run_args = _DEFAULT_RUN_ARGS

  def __call__(self, object __v = Null, *args, **kwargs):
    cdef _Generator g = _Generator.__new__(_Generator)
    g._chain_run = self._chain_run
    g._fn = self._fn
    g._ignore_result = self._ignore_result
    g._run_args = (__v, args, kwargs, False)
    return g

  def __iter__(self):
    return _sync_generator_fn(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __aiter__(self):
    return _async_generator_fn(self._chain_run, self._run_args, self._fn, self._ignore_result)

  def __repr__(self):
    """Return string representation."""
    return '<_Generator>'


# PERF: Function aliases — cdef object references avoid module-level global dict lookups.
# See PERFORMANCE.md #19.
cdef object _sync_generator_fn = sync_generator
cdef object _async_generator_fn = async_generator
