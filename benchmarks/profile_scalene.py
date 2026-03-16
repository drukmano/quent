# SPDX-License-Identifier: MIT
"""Standalone workload script designed to be run under scalene.

Scalene profiles CPU, memory, and GPU usage with line-level granularity
and minimal overhead. This script provides a rich but focused workload
that exercises the critical paths in quent.

Run with:
    scalene --cpu-only benchmarks/profile_scalene.py
    scalene benchmarks/profile_scalene.py              # CPU + memory
    scalene --html --outfile benchmarks/results/scalene.html benchmarks/profile_scalene.py
"""

from __future__ import annotations

from benchmarks._helpers import (
  DummyCM,
  add_one,
  identity,
  make_sync_chain,
  noop,
)
from quent import Chain

ITERATIONS = 20000


def main() -> None:
  print(f'Running scalene workload ({ITERATIONS} iterations)...')

  for _ in range(ITERATIONS):
    # Basic chain execution at various sizes.
    for n in (1, 5, 10, 50):
      chain = make_sync_chain(n)
      chain.run(0)

    # Operations.
    data = list(range(20))
    Chain(data).foreach(add_one).run()
    Chain(data).foreach_do(noop).run()
    Chain().gather(identity, identity, identity).run(42)
    Chain(DummyCM(99)).with_(identity).run()
    Chain().if_(predicate_true).then(identity).run(42)
    Chain().if_(predicate_true).then(identity).else_(add_one).run(42)

    # Nested chain.
    inner = Chain().then(add_one)
    Chain().then(inner).run(10)

  print(f'Completed {ITERATIONS} iterations.')
  print('Summary:')
  print('  Chain sizes tested: 1, 5, 10, 50')
  print('  Operations tested: foreach, foreach_do, gather, with_, if/else, nested')
  print(f'  Total chain executions: ~{ITERATIONS * (4 + 7 + 1):,}')


if __name__ == '__main__':
  main()
