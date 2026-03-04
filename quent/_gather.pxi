# PERF: @cython.final enables direct C dispatch. @cython.freelist(4) pools allocations.
# See PERFORMANCE.md #1, #2.
@cython.final
@cython.freelist(4)
cdef class _Gather:
  def __init__(self, tuple fns, Link link):
    self.fns = fns
    self.link = link

  def __call__(self, object current_value):
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
  # PERF: EMPTY_TUPLE/EMPTY_DICT avoid per-call empty container allocation. See PERFORMANCE.md #31.
  cdef Link link = _create_link(None, EMPTY_TUPLE, EMPTY_DICT)
  return _create_link(_Gather(fns, link), None, None, False, link)


async def _gather_to_async(list results):
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


# PERF: Function aliases — cdef object references avoid module-level global dict lookups.
# _asyncio_gather_fn caches asyncio.gather to avoid module attribute lookup per call.
# See PERFORMANCE.md #19.
cdef object _gather_async_fn = _gather_to_async
cdef object _asyncio_gather_fn = asyncio.gather
