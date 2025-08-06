import inspect


class TestExc(Exception):
  pass


def throw_if(v):
  if v:
    raise TestExc
  return v


def empty(v=None):
  return v


async def aempty(v=None):
  return v


async def await_(v):
  if inspect.isawaitable(v):
    return await v
  return v
