# _iteration.pxi — Collection operations (foreach, filter, reduce, gather)
#
# Implements the iterable transformation operations that apply functions to
# each element of a collection. Each operation follows a 3-function architecture:
#   1. Sync class (__call__): iterates synchronously, hands off to async on coroutine
#   2. Async continuation: continues iteration from where sync left off
#   3. Fully async variant: used when the input itself is an async iterator
#
# Key components:
#   - _Foreach / foreach: apply fn to each element, collecting results
#   - _ForeachIndexed / foreach_indexed: like foreach but fn receives (index, element)
#   - _Filter / filter_: keep elements where fn returns truthy
#   - _Reduce / reduce_: fold elements into a single value with fn(acc, el)
#   - _Gather / gather_: execute multiple functions in parallel, collect results


# --- Foreach ---

# foreach uses a 3-function architecture to handle sync-to-async transitions:
# 1. foreach (sync): iterates synchronously, calling fn(el) for each element.
#    If fn returns a coroutine, hands off to foreach_async mid-iteration.
# 2. foreach_async (transition): continues iteration from where sync left off,
#    awaiting coroutine results. Handles the sync-to-async boundary.
# 3. async_foreach (fully async): used when the input itself is an async
#    iterator (__aiter__). All iteration and fn calls are fully async.

@cython.final
@cython.freelist(8)
cdef class _Foreach:
  """Callable that applies a function to each element of an iterable."""

  def __init__(self, object fn, bint ignore_result, Link link):
    """Initialize with the mapping function, result-ignore flag, and error context link."""
    self.fn = fn
    self.ignore_result = ignore_result
    self.link = link

  def __call__(self, object current_value):
    """Iterate over current_value, applying fn to each element and collecting results.

    NOTE: temp_args is mutated on `self.link` for traceback context (set to the current
    element on async transition or exception). Since _Foreach instances are reusable,
    this mutation persists across runs. This is intentional — temp_args is only read
    during exception formatting, so the last-set value is always the relevant one.
    """
    if hasattr(current_value, '__aiter__'):
      return _async_foreach_fn(current_value, self.fn, self.ignore_result, self.link)
    cdef list lst = []
    cdef object el, result
    current_value = current_value.__iter__()
    try:
      while True:
        el = current_value.__next__()
        result = self.fn(el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _foreach_async_fn(current_value, self.fn, el, result, lst, self.ignore_result, self.link)
        if self.ignore_result:
          lst.append(el)
        else:
          lst.append(result)
    except _Break as exc:
      return handle_break_exc(exc, lst)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


cdef Link foreach(object fn, bint ignore_result):
  """Apply `fn` to each element of an iterable, collecting results into a list.
  Automatically detects async iterables and coroutine-returning functions.
  """
  cdef Link link = Link(fn, (), {}, fn_name='foreach')
  return Link(_Foreach(fn, ignore_result, link), original_value=link)


async def foreach_async(object current_value, object fn, object el, object result, list lst, bint ignore_result, Link link):
  """Async continuation of foreach when a coroutine is encountered mid-iteration."""
  try:
    while True:
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      el = current_value.__next__()
      result = fn(el)
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def async_foreach(object current_value, object fn, bint ignore_result, Link link):
  """Fully async foreach for async iterables, awaiting each fn result."""
  cdef list lst = []
  cdef object el, result
  try:
    async for el in current_value:
      result = fn(el)
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
    return lst
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


# --- Filter ---

# filter uses the same 3-function architecture as foreach:
# 1. filter_ (sync): iterates synchronously, calling fn(el) for each element.
#    If fn returns a coroutine, hands off to filter_async mid-iteration.
# 2. filter_async (transition): continues iteration from where sync left off,
#    awaiting coroutine results. Handles the sync-to-async boundary.
# 3. async_filter (fully async): used when the input itself is an async
#    iterator (__aiter__). All iteration and fn calls are fully async.

@cython.final
@cython.freelist(8)
cdef class _Filter:
  """Callable that filters elements of an iterable based on a predicate function."""

  def __init__(self, object fn, Link link):
    """Initialize with the predicate function and error context link."""
    self.fn = fn
    self.link = link

  def __call__(self, object current_value):
    """Iterate over current_value, applying fn to each element and keeping those where fn returns truthy.

    NOTE: temp_args is mutated on `self.link` for traceback context (set to the current
    element on async transition or exception). Since _Filter instances are reusable,
    this mutation persists across runs. This is intentional — temp_args is only read
    during exception formatting, so the last-set value is always the relevant one.
    """
    if hasattr(current_value, '__aiter__'):
      return _async_filter_fn(current_value, self.fn, self.link)
    cdef list lst = []
    cdef object el, result
    current_value = current_value.__iter__()
    try:
      while True:
        el = current_value.__next__()
        result = self.fn(el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _filter_async_fn(current_value, self.fn, el, result, lst, self.link)
        if result:
          lst.append(el)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


cdef Link filter_(object fn):
  """Filter elements of an iterable by a predicate function."""
  cdef Link link = Link(fn, (), {}, fn_name='filter')
  return Link(_Filter(fn, link), original_value=link)


async def filter_async(object current_value, object fn, object el, object result, list lst, Link link):
  """Async continuation of filter when a coroutine is encountered mid-iteration."""
  try:
    while True:
      if iscoro(result):
        result = await result
      if result:
        lst.append(el)
      el = current_value.__next__()
      result = fn(el)
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def async_filter(object current_value, object fn, Link link):
  """Fully async filter for async iterables, awaiting each fn result."""
  cdef list lst = []
  cdef object el, result
  try:
    async for el in current_value:
      result = fn(el)
      if iscoro(result):
        result = await result
      if result:
        lst.append(el)
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


# --- Reduce ---

# reduce uses the same 3-function architecture:
# 1. reduce_ (sync): iterates synchronously, calling fn(acc, el) for each element.
#    If fn returns a coroutine, hands off to reduce_async mid-iteration.
# 2. reduce_async (transition): continues iteration from where sync left off.
# 3. async_reduce (fully async): used when the input is an async iterator.

@cython.final
@cython.freelist(8)
cdef class _Reduce:
  """Callable that reduces an iterable to a single value using a binary function."""

  def __init__(self, object fn, object initial, Link link):
    """Initialize with the reducer function, initial accumulator, and error context link."""
    self.fn = fn
    self.initial = initial
    self.link = link

  def __call__(self, object current_value):
    """Iterate over current_value, applying fn(accumulator, element) to reduce to a single value.

    NOTE: temp_args is mutated on `self.link` for traceback context.
    """
    if hasattr(current_value, '__aiter__'):
      return _async_reduce_fn(current_value, self.fn, self.initial, self.link)
    cdef object acc, el, result
    current_value = current_value.__iter__()
    if self.initial is Null:
      try:
        acc = current_value.__next__()
      except StopIteration:
        raise TypeError('reduce() of empty iterable with no initial value')
    else:
      acc = self.initial
    try:
      while True:
        el = current_value.__next__()
        result = self.fn(acc, el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _reduce_async_fn(current_value, self.fn, el, result, acc, self.link)
        acc = result
    except StopIteration:
      return acc
    except BaseException:
      self.link.temp_args = (el,)
      raise


cdef Link reduce_(object fn, object initial):
  """Reduce an iterable to a single value using a binary function."""
  cdef Link link = Link(fn, (), {}, fn_name='reduce')
  return Link(_Reduce(fn, initial, link), original_value=link)


async def reduce_async(object current_value, object fn, object el, object result, object acc, Link link):
  """Async continuation of reduce when a coroutine is encountered mid-iteration."""
  try:
    while True:
      if iscoro(result):
        result = await result
      acc = result
      el = current_value.__next__()
      result = fn(acc, el)
  except StopIteration:
    return acc
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def async_reduce(object current_value, object fn, object initial, Link link):
  """Fully async reduce for async iterables."""
  cdef object acc, el, result
  cdef bint first = True
  if initial is not Null:
    acc = initial
    first = False
  try:
    async for el in current_value:
      if first:
        acc = el
        first = False
        continue
      result = fn(acc, el)
      if iscoro(result):
        result = await result
      acc = result
    if first:
      raise TypeError('reduce() of empty iterable with no initial value')
    return acc
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


# --- Gather ---

@cython.final
@cython.freelist(4)
cdef class _Gather:
  """Callable that executes multiple functions in parallel and collects results."""

  def __init__(self, tuple fns, Link link):
    """Initialize with the tuple of callables and error context link."""
    self.fns = fns
    self.link = link

  def __call__(self, object current_value):
    """Call each function with current_value, collecting results. If any returns a coroutine,
    use asyncio.gather on all results."""
    cdef list results = []
    cdef object fn, result
    cdef bint has_coro = False
    for fn in self.fns:
      result = fn(current_value)
      results.append(result)
      if iscoro(result):
        has_coro = True
    if has_coro:
      return _gather_async_fn(results)
    return results


cdef Link gather_(tuple fns):
  """Execute multiple functions in parallel and collect results."""
  cdef Link link = Link(None, (), {}, True, fn_name='gather')
  return Link(_Gather(fns, link), original_value=link)


async def gather_async(list results):
  """Await all coroutines in results using asyncio.gather."""
  cdef int i
  cdef list coros = []
  cdef list indices = []
  for i in range(len(results)):
    if iscoro(results[i]):
      coros.append(results[i])
      indices.append(i)
  cdef list resolved = await _asyncio_gather_fn(*coros)
  for i in range(len(indices)):
    results[indices[i]] = resolved[i]
  return results


# --- Foreach indexed ---

# foreach_indexed uses the same 3-function architecture as foreach:
# 1. foreach_indexed (sync): iterates synchronously, calling fn(idx, el).
#    If fn returns a coroutine, hands off to foreach_indexed_async mid-iteration.
# 2. foreach_indexed_async (transition): continues iteration from where sync left off.
# 3. async_foreach_indexed (fully async): used when the input is an async iterator.

@cython.final
@cython.freelist(8)
cdef class _ForeachIndexed:
  """Callable that applies a function to each element with its index."""

  def __init__(self, object fn, bint ignore_result, Link link):
    """Initialize with the mapping function, result-ignore flag, and error context link."""
    self.fn = fn
    self.ignore_result = ignore_result
    self.link = link

  def __call__(self, object current_value):
    """Iterate over current_value, calling fn(index, element) for each element.

    NOTE: temp_args is mutated on `self.link` for traceback context.
    """
    if hasattr(current_value, '__aiter__'):
      return _async_foreach_indexed_fn(current_value, self.fn, self.ignore_result, self.link)
    cdef list lst = []
    cdef object el, result
    cdef int idx = 0
    current_value = current_value.__iter__()
    try:
      while True:
        el = current_value.__next__()
        result = self.fn(idx, el)
        if iscoro(result):
          self.link.temp_args = (el,)
          return _foreach_indexed_async_fn(current_value, self.fn, idx, el, result, lst, self.ignore_result, self.link)
        if self.ignore_result:
          lst.append(el)
        else:
          lst.append(result)
        idx += 1
    except _Break as exc:
      return handle_break_exc(exc, lst)
    except StopIteration:
      return lst
    except BaseException:
      self.link.temp_args = (el,)
      raise


cdef Link foreach_indexed(object fn, bint ignore_result):
  """Apply fn(index, element) to each element of an iterable."""
  cdef Link link = Link(fn, (), {}, fn_name='foreach')
  return Link(_ForeachIndexed(fn, ignore_result, link), original_value=link)


async def foreach_indexed_async(object current_value, object fn, int idx, object el, object result, list lst, bint ignore_result, Link link):
  """Async continuation of foreach_indexed when a coroutine is encountered mid-iteration."""
  try:
    while True:
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      idx += 1
      el = current_value.__next__()
      result = fn(idx, el)
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result
  except StopIteration:
    return lst
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


async def async_foreach_indexed(object current_value, object fn, bint ignore_result, Link link):
  """Fully async foreach_indexed for async iterables."""
  cdef list lst = []
  cdef object el, result
  cdef int idx = 0
  try:
    async for el in current_value:
      result = fn(idx, el)
      if iscoro(result):
        result = await result
      if ignore_result:
        lst.append(el)
      else:
        lst.append(result)
      idx += 1
    return lst
  except _Break as exc:
    result = handle_break_exc(exc, lst)
    if iscoro(result):
      return await result
    return result
  except BaseException as exc:
    if not hasattr(exc, '__quent_link_temp_args__'):
      exc.__quent_link_temp_args__ = {}
    exc.__quent_link_temp_args__[id(link)] = (el,)
    raise


# --- Function aliases ---

cdef object _async_foreach_fn = async_foreach
cdef object _foreach_async_fn = foreach_async
cdef object _filter_async_fn = filter_async
cdef object _async_filter_fn = async_filter
cdef object _reduce_async_fn = reduce_async
cdef object _async_reduce_fn = async_reduce
cdef object _gather_async_fn = gather_async
cdef object _asyncio_gather_fn = asyncio.gather
cdef object _foreach_indexed_async_fn = foreach_indexed_async
cdef object _async_foreach_indexed_fn = async_foreach_indexed
