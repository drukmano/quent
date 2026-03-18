# SPDX-License-Identifier: MIT
"""cProfile wrapper for quent: runs a representative workload and prints stats.

Also saves a .prof file for use with pstats or snakeviz.

Run with:
    python benchmarks/profile_cprofile.py               # 10 000 iterations
    python benchmarks/profile_cprofile.py 50000         # custom iteration count
    snakeviz benchmarks/results/quent.prof              # visualize interactively
"""

from __future__ import annotations

import cProfile
import os
import pstats
import sys

from benchmarks._helpers import (
  DummyCM,
  add_one,
  identity,
  make_sync_pipeline,
  noop,
  predicate_true,
)
from quent import Q

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def workload(iterations: int) -> None:
  """Run a representative mix of pipeline operations."""
  for _ in range(iterations):
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


def main() -> None:
  iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
  print(f'Running cProfile workload ({iterations} iterations)...')

  prof = cProfile.Profile()
  prof.enable()
  workload(iterations)
  prof.disable()

  stats = pstats.Stats(prof)
  stats.strip_dirs()

  print('\n' + '=' * 80)
  print('Top 20 by cumulative time')
  print('=' * 80)
  stats.sort_stats('cumulative')
  stats.print_stats(20)

  print('=' * 80)
  print('Top 20 by self time')
  print('=' * 80)
  stats.sort_stats('tottime')
  stats.print_stats(20)

  # Save .prof file for offline analysis.
  os.makedirs(_RESULTS_DIR, exist_ok=True)
  prof_path = os.path.join(_RESULTS_DIR, 'quent.prof')
  prof.dump_stats(prof_path)
  print(f'\nProfile saved to {prof_path}')
  print('Visualize with: snakeviz ' + prof_path)


if __name__ == '__main__':
  main()
