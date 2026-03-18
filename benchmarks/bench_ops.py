# SPDX-License-Identifier: MIT
"""pyperf benchmarks for operation factories: foreach, foreach_do, gather, with_, if_.

Run with:
    python benchmarks/bench_ops.py
    python benchmarks/bench_ops.py --fast   # quick smoke-test
"""

from __future__ import annotations

import pyperf

from benchmarks._helpers import (
  DummyCM,
  add_one,
  identity,
  noop,
  predicate_false,
  predicate_true,
)
from quent import Q

# ---- Benchmark functions ----


def bench_map_10(loops: int) -> float:
  """.foreach(add_one) over a 10-element list."""
  data = list(range(10))
  q = Q(data).foreach(add_one)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run()
  return pyperf.perf_counter() - t0


def bench_map_100(loops: int) -> float:
  """.foreach(add_one) over a 100-element list."""
  data = list(range(100))
  q = Q(data).foreach(add_one)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run()
  return pyperf.perf_counter() - t0


def bench_map_1000(loops: int) -> float:
  """.foreach(add_one) over a 1000-element list."""
  data = list(range(1000))
  q = Q(data).foreach(add_one)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run()
  return pyperf.perf_counter() - t0


def bench_foreach_do_10(loops: int) -> float:
  """.foreach_do(noop) over a 10-element list."""
  data = list(range(10))
  q = Q(data).foreach_do(noop)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run()
  return pyperf.perf_counter() - t0


def bench_foreach_do_100(loops: int) -> float:
  """.foreach_do(noop) over a 100-element list."""
  data = list(range(100))
  q = Q(data).foreach_do(noop)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run()
  return pyperf.perf_counter() - t0


def bench_gather_2(loops: int) -> float:
  """.gather(identity, identity) — 2 concurrent branches."""
  q = Q().gather(identity, identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run(42)
  return pyperf.perf_counter() - t0


def bench_gather_5(loops: int) -> float:
  """.gather(identity * 5) — 5 concurrent branches."""
  q = Q().gather(identity, identity, identity, identity, identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run(42)
  return pyperf.perf_counter() - t0


def bench_gather_10(loops: int) -> float:
  """.gather(identity * 10) — 10 concurrent branches."""
  q = Q().gather(*([identity] * 10))
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run(42)
  return pyperf.perf_counter() - t0


def bench_with_cm(loops: int) -> float:
  """.with_(identity) using DummyCM context manager."""
  q = Q(DummyCM(99)).with_(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run()
  return pyperf.perf_counter() - t0


def bench_if_true(loops: int) -> float:
  """.if_(predicate_true).then(identity) — branch taken."""
  q = Q().if_(predicate_true).then(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run(42)
  return pyperf.perf_counter() - t0


def bench_if_false_else(loops: int) -> float:
  """.if_(predicate_false).then(identity).else_(add_one) — else branch taken."""
  q = Q().if_(predicate_false).then(identity).else_(add_one)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    q.run(42)
  return pyperf.perf_counter() - t0


if __name__ == '__main__':
  runner = pyperf.Runner()
  runner.bench_time_func('map_10', bench_map_10)
  runner.bench_time_func('map_100', bench_map_100)
  runner.bench_time_func('map_1000', bench_map_1000)
  runner.bench_time_func('foreach_do_10', bench_foreach_do_10)
  runner.bench_time_func('foreach_do_100', bench_foreach_do_100)
  runner.bench_time_func('gather_2', bench_gather_2)
  runner.bench_time_func('gather_5', bench_gather_5)
  runner.bench_time_func('gather_10', bench_gather_10)
  runner.bench_time_func('with_cm', bench_with_cm)
  runner.bench_time_func('if_true', bench_if_true)
  runner.bench_time_func('if_false_else', bench_if_false_else)
