import pyximport; pyximport.install()
import asyncio
from scripts.time_func import async_time_func
from src.quent import Chain


async def f1():
  return 1


async def f2(v):
  return v*10


async def direct_call():
  return await f2(await f1())


def with_quent():
  return Chain(f1).then(f2).run()


seq: Chain = Chain().then(f2)


def with_quent_recipe():
  return seq.run(f1)


async def main():
  loops = 100000
  iterations = 10
  direct_call_total = 0
  with_quent_total = 0
  with_quent_recipe_total = 0

  for _ in range(iterations):
    direct_call_total += await async_time_func(loops, direct_call)
    with_quent_total += await async_time_func(loops, with_quent)
    with_quent_recipe_total += await async_time_func(loops, with_quent_recipe)

  print(f'average of {iterations} iterations of {loops} loops each:')
  print('direct_call:', direct_call_total / iterations)
  print('with_quent:', with_quent_total / iterations)
  print('with_quent_recipe:', with_quent_recipe_total / iterations)


asyncio.run(main())


"""
average of 10 iterations of 100000 loops each:
direct_call: 1.1850
with_quent: 1.2045
with_quent_recipe: 1.0563
"""
