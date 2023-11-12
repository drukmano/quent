import pyximport; pyximport.install()
from src.quent import Chain, Cascade
from scripts.time_func import time_func
from tests.utils import throw_if, aempty, empty, await_

import inspect
import asyncio

async def test_exception_format():
  raise Exception
class A:
  def __eq__(self, other):
    raise Exception


class gen:
  def __init__(self):
    self.rng = range(10)

  def __iter__(self):
    return iter(self.rng)


class agen:
  def __init__(self):
    self.rng = range(10)

  def __aiter__(self):
    async def f():
      for i in self.rng:
        yield i
    return f()


async def main():
  import time
  #await await_(Chain(empty).then(FlexContext, value=1).with_(Chain().then(aempty).then(lambda ctx: ctx.get(value=0)).do(print)).then(print).then(FlexContext.get, value=0).then(print).run())
  # TODO is this possible? when calling .generator(), we should return a class that implements both __iter__
  #  and __aiter__
  for i in Chain(gen).then(Chain().then(empty).iterate_do(lambda i: time.sleep(0.1))).iterate(lambda v: v*10):
    print('result: ', i)
  async for i in Chain(agen).then(Chain().then(aempty).iterate_do(lambda i: asyncio.sleep(0.1))).iterate(lambda v: v*10):
    print('result: ', i)
  #async for i in Chain(empty, 1).then(agen, ...).then(empty).iterate_do(lambda i: time.sleep(0.1)):
  #  print(i)
  #async for i in Chain(gen()).iterate_do(lambda i: asyncio.sleep(0.5)):
  #  print(i)
  #async for i in Chain(agen()).iterate_do(lambda i: time.sleep(0.5)):
  #  print(i)
  #async for i in Chain(agen()).iterate_do(lambda i: asyncio.sleep(0.5)):
  #  print(i)

  #for i in Chain(range(10)).foreach_yield_do(lambda i: time.sleep(0.5)).generator():
  #  print(i)
  #print(time_func(100000, lambda: Chain(A()).eq(1)))
  #print(time_func(100000, lambda: Chain(A()).neq(1)))
  pass

asyncio.run(main())
#Chain(True).then(throw_if).run()
#Chain(throw_if, True).run()

#class A:
#  def f1(self, *args, **kwargs):
#    raise Exception
#
#Cascade(A()).then(lambda: 5, ...).call('f1', 1,2,3, some_ok=5, no=6).run()
