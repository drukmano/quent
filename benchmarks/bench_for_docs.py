# SPDX-License-Identifier: MIT
"""Quick benchmark script to generate numbers for README / docs.

Run with:
    .venv/bin/python3 benchmarks/bench_for_docs.py
"""

from __future__ import annotations

import asyncio
import time
import timeit

from quent import Q

# ---- Callables ----


def identity(x):
  return x


async def async_identity(x):
  return x


def add_one(x):
  return x + 1


async def async_add_one(x):
  return x + 1


async def simulated_io(x):
  """Simulate a fast I/O call (~1ms)."""
  await asyncio.sleep(0.001)
  return x


async def simulated_db_query(x):
  """Simulate a DB query (~5ms)."""
  await asyncio.sleep(0.005)
  return x


async def simulated_http(x):
  """Simulate an HTTP call (~50ms)."""
  await asyncio.sleep(0.050)
  return x


# ---- Helpers ----


def bench_sync(label, stmt, setup='', number=500_000, repeat=5):
  """Run timeit and print results in ns per operation."""
  times = timeit.repeat(stmt, setup=setup, number=number, repeat=repeat, globals=globals())
  best = min(times) / number * 1e9
  mean = sum(times) / len(times) / number * 1e9
  return label, best, mean


def bench_async_runner(label, coro_factory, number=10_000, repeat=5):
  """Benchmark an async operation via asyncio.run."""
  times = []
  for _ in range(repeat):
    t0 = time.perf_counter()
    for _ in range(number):
      asyncio.run(coro_factory())
    times.append(time.perf_counter() - t0)
  best = min(times) / number * 1e9
  mean = sum(times) / len(times) / number * 1e9
  return label, best, mean


def bench_async_in_loop(label, coro_factory, number=50_000, repeat=5):
  """Benchmark an async operation inside a running loop (no asyncio.run overhead)."""

  async def runner():
    times = []
    for _ in range(repeat):
      t0 = time.perf_counter()
      for _ in range(number):
        await coro_factory()
      times.append(time.perf_counter() - t0)
    return times

  times = asyncio.run(runner())
  best = min(times) / number * 1e9
  mean = sum(times) / len(times) / number * 1e9
  return label, best, mean


def print_section(title, results):
  print(f'\n{"=" * 70}')
  print(f'  {title}')
  print(f'{"=" * 70}')
  name_width = max(len(r[0]) for r in results) + 2
  print(f'  {"Operation":<{name_width}} {"Best (ns)":>12} {"Mean (ns)":>12}')
  print(f'  {"-" * name_width} {"-" * 12} {"-" * 12}')
  for label, best, mean in results:
    print(f'  {label:<{name_width}} {best:>12.1f} {mean:>12.1f}')


# ---- Baselines ----


def run_baselines():
  results = []
  results.append(bench_sync('Raw function call', 'identity(42)'))
  results.append(bench_sync('5x raw calls (chain)', 'identity(identity(identity(identity(identity(42)))))'))
  results.append(
    bench_sync(
      '10x raw calls (chain)',
      """
x = 42
x = identity(x); x = identity(x); x = identity(x); x = identity(x); x = identity(x)
x = identity(x); x = identity(x); x = identity(x); x = identity(x); x = identity(x)
""",
    )
  )
  return results


# ---- Sync pipeline benchmarks ----

q_empty = Q()
q_1 = Q().then(identity)
q_5 = Q()
for _ in range(5):
  q_5.then(identity)
q_10 = Q()
for _ in range(10):
  q_10.then(identity)


def run_sync_pipeline():
  results = []
  results.append(bench_sync('Q().run() [empty]', 'q_empty.run()'))
  results.append(bench_sync('Q.then(f).run(v) [1 step]', 'q_1.run(42)'))
  results.append(bench_sync('Q.then(f)x5.run(v) [5 steps]', 'q_5.run(42)'))
  results.append(bench_sync('Q.then(f)x10.run(v) [10 steps]', 'q_10.run(42)'))
  return results


# ---- Async pipeline benchmarks (inside running loop) ----

q_async_1 = Q().then(async_identity)
q_async_5 = Q()
for _ in range(5):
  q_async_5.then(async_identity)
q_async_10 = Q()
for _ in range(10):
  q_async_10.then(async_identity)
q_mixed_5 = Q()
for i in range(5):
  q_mixed_5.then(identity if i % 2 == 0 else async_identity)
q_mixed_10 = Q()
for i in range(10):
  q_mixed_10.then(identity if i % 2 == 0 else async_identity)


def run_async_pipeline():
  results = []
  results.append(bench_async_in_loop('Raw: await async_identity(42)', lambda: async_identity(42)))
  results.append(bench_async_in_loop('Q.then(async_f).run(v) [1 step]', lambda: q_async_1.run(42)))
  results.append(bench_async_in_loop('Q.then(async_f)x5.run(v) [5 steps]', lambda: q_async_5.run(42)))
  results.append(bench_async_in_loop('Q.then(async_f)x10.run(v) [10 steps]', lambda: q_async_10.run(42)))
  results.append(bench_async_in_loop('Mixed sync/async x5', lambda: q_mixed_5.run(42)))
  results.append(bench_async_in_loop('Mixed sync/async x10', lambda: q_mixed_10.run(42)))
  return results


# ---- I/O-bound benchmarks ----


def run_io_benchmarks():
  results = []

  # 1ms simulated I/O -- raw vs pipeline
  async def raw_io_1ms():
    return await simulated_io(42)

  q_io_1ms = Q().then(simulated_io)

  async def pipeline_io_1ms():
    return await q_io_1ms.run(42)

  # 3-step pipeline with 1ms I/O in the middle
  q_io_3step = Q().then(identity).then(simulated_io).then(identity)

  async def pipeline_io_3step():
    return await q_io_3step.run(42)

  # 5ms DB query -- raw vs pipeline
  async def raw_db_5ms():
    return await simulated_db_query(42)

  q_db_5ms = Q().then(identity).then(simulated_db_query).then(identity)

  async def pipeline_db_5ms():
    return await q_db_5ms.run(42)

  # 50ms HTTP -- raw vs pipeline
  async def raw_http_50ms():
    return await simulated_http(42)

  q_http_50ms = Q().then(identity).then(simulated_http).then(identity)

  async def pipeline_http_50ms():
    return await q_http_50ms.run(42)

  results.append(bench_async_in_loop('Raw: await io_1ms()', raw_io_1ms, number=500, repeat=3))
  results.append(bench_async_in_loop('Pipeline: io_1ms [3 steps]', pipeline_io_3step, number=500, repeat=3))
  results.append(bench_async_in_loop('Raw: await db_5ms()', raw_db_5ms, number=200, repeat=3))
  results.append(bench_async_in_loop('Pipeline: db_5ms [3 steps]', pipeline_db_5ms, number=200, repeat=3))
  results.append(bench_async_in_loop('Raw: await http_50ms()', raw_http_50ms, number=50, repeat=3))
  results.append(bench_async_in_loop('Pipeline: http_50ms [3 steps]', pipeline_http_50ms, number=50, repeat=3))
  return results


# ---- Construction benchmarks ----


def run_construction():
  results = []
  results.append(bench_sync('Q() construction', 'Q()'))
  results.append(
    bench_sync(
      'Q() + 5x .then()',
      """
q = Q()
q.then(identity); q.then(identity); q.then(identity); q.then(identity); q.then(identity)
""",
      number=200_000,
    )
  )
  results.append(
    bench_sync(
      'Q() + 10x .then()',
      """
q = Q()
q.then(identity); q.then(identity); q.then(identity); q.then(identity); q.then(identity)
q.then(identity); q.then(identity); q.then(identity); q.then(identity); q.then(identity)
""",
      number=200_000,
    )
  )
  return results


# ---- Per-step overhead ----


def compute_per_step():
  """Compute per-step overhead by comparing pipeline sizes."""
  # Sync
  sync_results = {}
  for n in [1, 5, 10]:
    q = Q()
    for _ in range(n):
      q.then(identity)
    times = timeit.repeat('q.run(42)', number=500_000, repeat=5, globals={'q': q})
    sync_results[n] = min(times) / 500_000 * 1e9

  # Raw call baseline
  raw_times = timeit.repeat('identity(42)', number=500_000, repeat=5, globals={'identity': identity})
  raw_ns = min(raw_times) / 500_000 * 1e9

  sync_per_step_5_10 = (sync_results[10] - sync_results[5]) / 5
  sync_per_step_1_5 = (sync_results[5] - sync_results[1]) / 4

  # Async (inside running loop)
  async def measure_async(n, number=50_000, repeat=5):
    q = Q()
    for _ in range(n):
      q.then(async_identity)
    times = []
    for _ in range(repeat):
      t0 = time.perf_counter()
      for _ in range(number):
        await q.run(42)
      times.append(time.perf_counter() - t0)
    return min(times) / number * 1e9

  async def run_all():
    r = {}
    for n in [1, 5, 10]:
      r[n] = await measure_async(n)
    return r

  async_results = asyncio.run(run_all())
  async_per_step_5_10 = (async_results[10] - async_results[5]) / 5
  async_per_step_1_5 = (async_results[5] - async_results[1]) / 4

  return {
    'raw_call': raw_ns,
    'sync': sync_results,
    'async': async_results,
    'sync_per_step': (sync_per_step_1_5 + sync_per_step_5_10) / 2,
    'async_per_step': (async_per_step_1_5 + async_per_step_5_10) / 2,
  }


if __name__ == '__main__':
  import platform

  print(f'Platform: {platform.platform()}')
  print(f'Python: {platform.python_version()}')
  print(f'CPU: {platform.processor()}')

  print_section('BASELINES — Raw Function Calls', run_baselines())
  print_section('SYNC PIPELINE EXECUTION', run_sync_pipeline())
  print_section('PIPELINE CONSTRUCTION', run_construction())
  print_section('ASYNC PIPELINE EXECUTION (inside running loop)', run_async_pipeline())
  print_section('I/O-BOUND: Pipeline Overhead vs Real I/O', run_io_benchmarks())

  print(f'\n{"=" * 70}')
  print('  PER-STEP OVERHEAD (derived)')
  print(f'{"=" * 70}')
  ps = compute_per_step()
  print(f'  Raw function call:          {ps["raw_call"]:.1f} ns')
  print(f'  Sync per-step overhead:     {ps["sync_per_step"]:.1f} ns')
  print(f'  Async per-step overhead:    {ps["async_per_step"]:.1f} ns')
  print()
  print('  Sync pipeline totals:')
  for n, v in sorted(ps['sync'].items()):
    print(f'    {n:>3} steps: {v:.1f} ns')
  print('  Async pipeline totals:')
  for n, v in sorted(ps['async'].items()):
    print(f'    {n:>3} steps: {v:.1f} ns')
