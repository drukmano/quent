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


class DummySync:
  @property
  def a1(self):
    return self

  def b1(self):
    return self


class DummyAsync:
  @property
  def a1(self):
    return self

  def b1(self):
    return self
