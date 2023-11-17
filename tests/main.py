import pyximport; pyximport.install()
from scripts.time_func import time_func
from quent import Chain, Cascade
from tests.utils import throw_if


import inspect
import asyncio

async def test_exception_format():
  raise Exception
class A:
  def __eq__(self, other):
    raise Exception
async def main():
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
