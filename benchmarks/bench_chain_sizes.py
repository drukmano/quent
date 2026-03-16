# SPDX-License-Identifier: MIT
"""pyperf scaling benchmarks: execution time vs. chain length.

Measures how chain execution time scales from 1 to 1000 links,
useful for detecting per-link overhead growth.

Run with:
    python benchmarks/bench_chain_sizes.py
    python benchmarks/bench_chain_sizes.py --fast   # quick smoke-test
"""

from __future__ import annotations

import pyperf

from benchmarks._helpers import STANDARD_SIZES, make_sync_chain

# ---- Benchmark factory ----


def _make_bench(chain):  # type: ignore[no-untyped-def]
  """Return a bench_time_func-compatible function for *chain*."""

  def bench(loops: int) -> float:
    t0 = pyperf.perf_counter()
    for _ in range(loops):
      chain.run(0)
    return pyperf.perf_counter() - t0

  return bench


# Extended size range including 1000 for scaling analysis.
_SCALING_SIZES: tuple[int, ...] = (*STANDARD_SIZES, 500, 1000)


if __name__ == '__main__':
  runner = pyperf.Runner()
  for n in _SCALING_SIZES:
    chain = make_sync_chain(n)
    runner.bench_time_func(f'chain_{n}_links', _make_bench(chain))
