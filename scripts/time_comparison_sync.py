from scripts.time_func import time_func
from quent.quent import Chain


def f1():
  return 1


def f2(v):
  return v*10


def direct_call():
  return f2(f1())


def with_quent():
  return Chain(f1).then(f2).run()


seq: Chain = Chain().then(f2)


def with_quent_frozen():
  return seq.run(f1)


def main():
  loops = 100000
  iterations = 10
  direct_call_total = 0
  with_quent_total = 0
  with_quent_frozen_total = 0

  for _ in range(iterations):
    direct_call_total += time_func(loops, direct_call)
    with_quent_total += time_func(loops, with_quent)
    with_quent_frozen_total += time_func(loops, with_quent_frozen)

  print(f'average of {iterations} iterations of {loops} loops each:')
  print('direct_call:', direct_call_total / iterations)
  print('with_quent:', with_quent_total / iterations)
  print('with_quent_frozen:', with_quent_frozen_total / iterations)


main()


"""
average of 10 iterations of 100000 loops each:
direct_call: 1.1850
with_quent: 1.2045
with_quent_frozen: 1.0563
"""
