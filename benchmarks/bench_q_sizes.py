# SPDX-License-Identifier: MIT
"""pyperf scaling benchmarks: execution time vs. pipeline length.

Measures how pipeline execution time scales from 1 to 1000 links,
useful for detecting per-link overhead growth.

Run with:
    python benchmarks/bench_q_sizes.py
    python benchmarks/bench_q_sizes.py --fast   # quick smoke-test
"""

from __future__ import annotations

import pyperf

from benchmarks._helpers import STANDARD_SIZES, make_sync_pipeline

# ---- Benchmark factory ----


def _make_bench(q):  # type: ignore[no-untyped-def]
  """Return a bench_time_func-compatible function for *q*."""

  def bench(loops: int) -> float:
    t0 = pyperf.perf_counter()
    for _ in range(loops):
      q.run(0)
    return pyperf.perf_counter() - t0

  return bench


# Extended size range including 1000 for scaling analysis.
_SCALING_SIZES: tuple[int, ...] = (*STANDARD_SIZES, 500, 1000)


if __name__ == '__main__':
  runner = pyperf.Runner()
  for n in _SCALING_SIZES:
    q = make_sync_pipeline(n)
    runner.bench_time_func(f'pipeline_{n}_links', _make_bench(q))
