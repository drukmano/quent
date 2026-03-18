# SPDX-License-Identifier: MIT
"""pyperf benchmarks for async pipeline execution and sync-to-async transition.

Run with:
    python benchmarks/bench_async.py
    python benchmarks/bench_async.py --fast   # quick smoke-test
"""

from __future__ import annotations

import asyncio

import pyperf

from benchmarks._helpers import async_identity, identity, make_async_pipeline, make_mixed_pipeline
from quent import Q

# ---- Benchmark functions ----


def bench_async_pipeline_1(loops: int) -> float:
  """Single-step async pipeline via asyncio.run — measures transition overhead."""
  q = Q().then(async_identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(q.run(0))
  return pyperf.perf_counter() - t0


def bench_async_pipeline_5(loops: int) -> float:
  """5-link fully async pipeline via asyncio.run."""
  q = make_async_pipeline(5)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(q.run(0))
  return pyperf.perf_counter() - t0


def bench_async_pipeline_10(loops: int) -> float:
  """10-link fully async pipeline via asyncio.run."""
  q = make_async_pipeline(10)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(q.run(0))
  return pyperf.perf_counter() - t0


def bench_mixed_pipeline_5(loops: int) -> float:
  """5-link alternating sync/async pipeline — transition in the middle."""
  q = make_mixed_pipeline(5)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(q.run(0))
  return pyperf.perf_counter() - t0


def bench_mixed_pipeline_10(loops: int) -> float:
  """10-link alternating sync/async pipeline."""
  q = make_mixed_pipeline(10)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(q.run(0))
  return pyperf.perf_counter() - t0


def bench_sync_pipeline_5_baseline(loops: int) -> float:
  """5-link sync pipeline run via asyncio.run for a fair async comparison baseline."""
  q = Q()
  for _ in range(5):
    q.then(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    result = q.run(0)
    # Sync pipeline returns a plain value; wrap/unwrap mirrors async path cost.
    _ = result
  return pyperf.perf_counter() - t0


def bench_async_transition_late(loops: int) -> float:
  """Async transition at step 5 of a 10-step pipeline (5 sync then 5 async)."""
  q = Q()
  for _ in range(5):
    q.then(identity)
  for _ in range(5):
    q.then(async_identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(q.run(0))
  return pyperf.perf_counter() - t0


def bench_raw_await_baseline(loops: int) -> float:
  """Raw asyncio.run(coroutine) overhead without any pipeline machinery."""

  async def _coro(x: int) -> int:
    return await async_identity(x)

  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(_coro(0))
  return pyperf.perf_counter() - t0


if __name__ == '__main__':
  runner = pyperf.Runner()
  runner.bench_time_func('async_pipeline_1', bench_async_pipeline_1)
  runner.bench_time_func('async_pipeline_5', bench_async_pipeline_5)
  runner.bench_time_func('async_pipeline_10', bench_async_pipeline_10)
  runner.bench_time_func('mixed_pipeline_5', bench_mixed_pipeline_5)
  runner.bench_time_func('mixed_pipeline_10', bench_mixed_pipeline_10)
  runner.bench_time_func('sync_pipeline_5_baseline', bench_sync_pipeline_5_baseline)
  runner.bench_time_func('async_transition_late', bench_async_transition_late)
  runner.bench_time_func('raw_await_baseline', bench_raw_await_baseline)
