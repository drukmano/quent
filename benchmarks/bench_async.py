# SPDX-License-Identifier: MIT
"""pyperf benchmarks for async chain execution and sync-to-async transition.

Run with:
    python benchmarks/bench_async.py
    python benchmarks/bench_async.py --fast   # quick smoke-test
"""

from __future__ import annotations

import asyncio

import pyperf

from benchmarks._helpers import async_identity, identity, make_async_chain, make_mixed_chain
from quent import Chain

# ---- Benchmark functions ----


def bench_async_chain_1(loops: int) -> float:
  """Single-step async chain via asyncio.run — measures transition overhead."""
  chain = Chain().then(async_identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(chain.run(0))
  return pyperf.perf_counter() - t0


def bench_async_chain_5(loops: int) -> float:
  """5-link fully async chain via asyncio.run."""
  chain = make_async_chain(5)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(chain.run(0))
  return pyperf.perf_counter() - t0


def bench_async_chain_10(loops: int) -> float:
  """10-link fully async chain via asyncio.run."""
  chain = make_async_chain(10)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(chain.run(0))
  return pyperf.perf_counter() - t0


def bench_mixed_chain_5(loops: int) -> float:
  """5-link alternating sync/async chain — transition in the middle."""
  chain = make_mixed_chain(5)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(chain.run(0))
  return pyperf.perf_counter() - t0


def bench_mixed_chain_10(loops: int) -> float:
  """10-link alternating sync/async chain."""
  chain = make_mixed_chain(10)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(chain.run(0))
  return pyperf.perf_counter() - t0


def bench_sync_chain_5_baseline(loops: int) -> float:
  """5-link sync chain run via asyncio.run for a fair async comparison baseline."""
  chain = Chain()
  for _ in range(5):
    chain.then(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    result = chain.run(0)
    # Sync chain returns a plain value; wrap/unwrap mirrors async path cost.
    _ = result
  return pyperf.perf_counter() - t0


def bench_async_transition_late(loops: int) -> float:
  """Async transition at step 5 of a 10-step chain (5 sync then 5 async)."""
  chain = Chain()
  for _ in range(5):
    chain.then(identity)
  for _ in range(5):
    chain.then(async_identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(chain.run(0))
  return pyperf.perf_counter() - t0


def bench_raw_await_baseline(loops: int) -> float:
  """Raw asyncio.run(coroutine) overhead without any chain machinery."""

  async def _coro(x: int) -> int:
    return await async_identity(x)

  t0 = pyperf.perf_counter()
  for _ in range(loops):
    asyncio.run(_coro(0))
  return pyperf.perf_counter() - t0


if __name__ == '__main__':
  runner = pyperf.Runner()
  runner.bench_time_func('async_chain_1', bench_async_chain_1)
  runner.bench_time_func('async_chain_5', bench_async_chain_5)
  runner.bench_time_func('async_chain_10', bench_async_chain_10)
  runner.bench_time_func('mixed_chain_5', bench_mixed_chain_5)
  runner.bench_time_func('mixed_chain_10', bench_mixed_chain_10)
  runner.bench_time_func('sync_chain_5_baseline', bench_sync_chain_5_baseline)
  runner.bench_time_func('async_transition_late', bench_async_transition_late)
  runner.bench_time_func('raw_await_baseline', bench_raw_await_baseline)
