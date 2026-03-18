# SPDX-License-Identifier: MIT
"""line_profiler integration for quent hot-path functions.

Profiles _evaluate_value, _run, and _run_async line-by-line to identify
the most expensive lines in the execution hot path.

Run with:
    # Using the LineProfiler API directly (no kernprof needed):
    python benchmarks/profile_line.py

    # Or using kernprof for annotated source output:
    kernprof -l -v benchmarks/profile_line.py
"""

from __future__ import annotations

import sys

from benchmarks._helpers import (
  DummyCM,
  add_one,
  identity,
  make_sync_pipeline,
  noop,
)
from quent import Q
from quent._engine import _run, _run_async
from quent._eval import _evaluate_value


def workload() -> None:
  """Run a representative workload for line profiling."""
  for _ in range(5000):
    # Pipeline execution at various sizes.
    for n in (1, 5, 10):
      q = make_sync_pipeline(n)
      q.run(0)

    # Operations.
    data = list(range(20))
    Q(data).foreach(add_one).run()
    Q(data).foreach_do(noop).run()
    Q().gather(identity, identity, identity).run(42)
    Q(DummyCM(99)).with_(identity).run()


def main() -> None:
  try:
    from line_profiler import LineProfiler
  except ImportError:
    print('line_profiler is not installed. Install it with:')
    print('  uv sync --group bench')
    print('  # or: pip install line-profiler')
    sys.exit(1)

  profiler = LineProfiler()

  # Profile the hot-path functions.
  profiler.add_function(_evaluate_value)
  profiler.add_function(_run)
  profiler.add_function(_run_async)

  print('Running line profiler workload (5000 iterations)...')
  profiler.runcall(workload)

  print('\n' + '=' * 80)
  print('Line-by-line profile')
  print('=' * 80)
  profiler.print_stats()


if __name__ == '__main__':
  main()
