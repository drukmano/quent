from time import process_time


def time_func(loops: int, fn, /, *args, **kwargs):
  start = process_time()
  for _ in range(loops):
    fn(*args, **kwargs)
  return process_time() - start


async def async_time_func(loops: int, fn, /, *args, **kwargs):
  start = process_time()
  for _ in range(loops):
    await fn(*args, **kwargs)
  return process_time() - start
