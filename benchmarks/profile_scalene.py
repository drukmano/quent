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
  make_sync_pipeline,
  noop,
  predicate_true,
)
from quent import Q

ITERATIONS = 20000


def main() -> None:
  print(f'Running scalene workload ({ITERATIONS} iterations)...')

  for _ in range(ITERATIONS):
    # Basic pipeline execution at various sizes.
    for n in (1, 5, 10, 50):
      q = make_sync_pipeline(n)
      q.run(0)

    # Operations.
    data = list(range(20))
    Q(data).foreach(add_one).run()
    Q(data).foreach_do(noop).run()
    Q().gather(identity, identity, identity).run(42)
    Q(DummyCM(99)).with_(identity).run()
    Q().if_(predicate_true).then(identity).run(42)
    Q().if_(predicate_true).then(identity).else_(add_one).run(42)

    # Nested pipeline.
    inner = Q().then(add_one)
    Q().then(inner).run(10)

  print(f'Completed {ITERATIONS} iterations.')
  print('Summary:')
  print('  Pipeline sizes tested: 1, 5, 10, 50')
  print('  Operations tested: foreach, foreach_do, gather, with_, if/else, nested')
  print(f'  Total pipeline executions: ~{ITERATIONS * (4 + 7 + 1):,}')


if __name__ == '__main__':
  main()
