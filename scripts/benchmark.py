import asyncio
from dataclasses import dataclass

from scripts.time_func import time_func, async_time_func
from quent.quent import Chain, run

LOOPS = 100_000
ITERATIONS = 10
FOREACH_LOOPS = 10_000


@dataclass
class BenchResult:
  name: str
  plain_time: float
  quent_time: float
  reuse_time: float | None = None

  @property
  def overhead_pct(self) -> float:
    if self.plain_time == 0:
      return float('inf')
    return ((self.quent_time - self.plain_time) / self.plain_time) * 100

  @property
  def reuse_overhead_pct(self) -> float | None:
    if self.reuse_time is None:
      return None
    if self.plain_time == 0:
      return float('inf')
    return ((self.reuse_time - self.plain_time) / self.plain_time) * 100


@dataclass
class ComparisonResult:
  name: str
  baseline_time: float
  optimized_time: float
  baseline_label: str = 'baseline'
  optimized_label: str = 'optimized'

  @property
  def speedup_pct(self) -> float:
    if self.optimized_time == 0:
      return float('inf')
    return ((self.baseline_time - self.optimized_time) / self.optimized_time) * 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def f_identity(v):
  return v

def f_mul_10(v):
  return v * 10

def f_add_5(v):
  return v + 5

def f_sub_3(v):
  return v - 3

def f_neg(v):
  return -v

def f_double(v):
  return v * 2

def f_square(v):
  return v ** 2

def f_mod_7(v):
  return v % 7

def f_abs(v):
  return abs(v)

def f_plus_1(v):
  return v + 1

def f_noop(_v):
  pass


# Async equivalents

async def af_mul_10(v):
  return v * 10

async def af_add_5(v):
  return v + 5

async def af_sub_3(v):
  return v - 3

async def af_double(v):
  return v * 2


# ---------------------------------------------------------------------------
# Timing runners
# ---------------------------------------------------------------------------

def print_result(r: BenchResult):
  print(f'  {r.name}:')
  print(f'    plain:   {r.plain_time:.6f}s')
  print(f'    quent:   {r.quent_time:.6f}s  ({r.overhead_pct:+.1f}%)')
  if r.reuse_time is not None:
    print(f'    reuse:   {r.reuse_time:.6f}s  ({r.reuse_overhead_pct:+.1f}%)')


def run_bench(name, plain_fn, quent_fn, reuse_fn=None, loops=LOOPS):
  plain_total = 0.0
  quent_total = 0.0
  reuse_total = 0.0
  for _ in range(ITERATIONS):
    plain_total += time_func(loops, plain_fn)
    quent_total += time_func(loops, quent_fn)
    if reuse_fn is not None:
      reuse_total += time_func(loops, reuse_fn)
  result = BenchResult(
    name=name,
    plain_time=plain_total / ITERATIONS,
    quent_time=quent_total / ITERATIONS,
    reuse_time=reuse_total / ITERATIONS if reuse_fn is not None else None,
  )
  print_result(result)
  return result


async def async_run_bench(name, plain_fn, quent_fn, reuse_fn=None, loops=LOOPS):
  plain_total = 0.0
  quent_total = 0.0
  reuse_total = 0.0
  for _ in range(ITERATIONS):
    plain_total += await async_time_func(loops, plain_fn)
    quent_total += await async_time_func(loops, quent_fn)
    if reuse_fn is not None:
      reuse_total += await async_time_func(loops, reuse_fn)
  result = BenchResult(
    name=name,
    plain_time=plain_total / ITERATIONS,
    quent_time=quent_total / ITERATIONS,
    reuse_time=reuse_total / ITERATIONS if reuse_fn is not None else None,
  )
  print_result(result)
  return result


def print_comparison_result(r: ComparisonResult):
  print(f'  {r.name}:')
  print(f'    {r.baseline_label}:   {r.baseline_time:.6f}s')
  print(f'    {r.optimized_label}:   {r.optimized_time:.6f}s  ({r.speedup_pct:+.1f}% speedup)')


def run_comparison_bench(name, baseline_fn, optimized_fn, baseline_label='baseline', optimized_label='optimized', loops=LOOPS):
  baseline_total = 0.0
  optimized_total = 0.0
  for _ in range(ITERATIONS):
    baseline_total += time_func(loops, baseline_fn)
    optimized_total += time_func(loops, optimized_fn)
  result = ComparisonResult(
    name=name,
    baseline_time=baseline_total / ITERATIONS,
    optimized_time=optimized_total / ITERATIONS,
    baseline_label=baseline_label,
    optimized_label=optimized_label,
  )
  print_comparison_result(result)
  return result


async def async_run_comparison_bench(name, baseline_fn, optimized_fn, baseline_label='baseline', optimized_label='optimized', loops=LOOPS):
  baseline_total = 0.0
  optimized_total = 0.0
  for _ in range(ITERATIONS):
    baseline_total += await async_time_func(loops, baseline_fn)
    optimized_total += await async_time_func(loops, optimized_fn)
  result = ComparisonResult(
    name=name,
    baseline_time=baseline_total / ITERATIONS,
    optimized_time=optimized_total / ITERATIONS,
    baseline_label=baseline_label,
    optimized_label=optimized_label,
  )
  print_comparison_result(result)
  return result


# ---------------------------------------------------------------------------
# Sync benchmarks
# ---------------------------------------------------------------------------

def bench_simple_chain():
  def plain():
    return f_mul_10(1)
  def quent():
    return Chain(1).then(f_mul_10).run()
  c = Chain(1).then(f_mul_10)
  return run_bench('simple_chain (2 ops)', plain, quent, c.run)


def bench_medium_chain():
  def plain():
    v = 1
    v = f_mul_10(v)
    v = f_add_5(v)
    v = f_sub_3(v)
    v = f_double(v)
    return v
  def quent():
    return Chain(1).then(f_mul_10).then(f_add_5).then(f_sub_3).then(f_double).run()
  c = Chain(1).then(f_mul_10).then(f_add_5).then(f_sub_3).then(f_double)
  return run_bench('medium_chain (5 ops)', plain, quent, c.run)


def bench_long_chain():
  def plain():
    v = 1
    v = f_mul_10(v)
    v = f_add_5(v)
    v = f_sub_3(v)
    v = f_double(v)
    v = f_square(v)
    v = f_mod_7(v)
    v = f_abs(v)
    v = f_plus_1(v)
    v = f_neg(v)
    v = f_abs(v)
    v = f_double(v)
    return v
  def quent():
    return Chain(1).then(f_mul_10).then(f_add_5).then(f_sub_3).then(f_double).then(f_square).then(f_mod_7).then(f_abs).then(f_plus_1).then(f_neg).then(f_abs).then(f_double).run()
  c = Chain(1).then(f_mul_10).then(f_add_5).then(f_sub_3).then(f_double).then(f_square).then(f_mod_7).then(f_abs).then(f_plus_1).then(f_neg).then(f_abs).then(f_double)
  return run_bench('long_chain (12 ops)', plain, quent, c.run)


def bench_do_side_effects():
  side_effects = []
  def plain():
    v = 10
    side_effects.append(v)
    return f_mul_10(v)
  def quent():
    return Chain(10).do(side_effects.append).then(f_mul_10).run()
  c = Chain(10).do(side_effects.append).then(f_mul_10)
  return run_bench('do_side_effects', plain, quent, c.run)


def bench_foreach():
  items = list(range(100))
  def plain():
    return [f_mul_10(x) for x in items]
  def quent():
    return Chain(items).foreach(f_mul_10).run()
  c = Chain(items).foreach(f_mul_10)
  return run_bench('foreach (100 items)', plain, quent, c.run, loops=FOREACH_LOOPS)


def bench_except_finally():
  def on_exc(v):
    pass
  def on_finally(v):
    pass
  def plain():
    try:
      v = f_mul_10(1)
    except Exception:
      on_exc(1)
    finally:
      on_finally(1)
    return v
  def quent():
    return Chain(1).then(f_mul_10).except_(on_exc).finally_(on_finally).run()
  c = Chain(1).then(f_mul_10).except_(on_exc).finally_(on_finally)
  return run_bench('except_finally (happy path)', plain, quent, c.run)


def bench_pipe_syntax():
  def plain():
    return f_add_5(f_mul_10(1))
  def quent():
    return Chain(1) | f_mul_10 | f_add_5 | run()
  return run_bench('pipe_syntax', plain, quent)


def bench_nested_chains():
  def plain():
    v = 1
    v = f_mul_10(v)
    inner = f_add_5(v)
    inner = f_double(inner)
    return inner
  def quent():
    return Chain(1).then(f_mul_10).then(Chain().then(f_add_5).then(f_double)).run()
  return run_bench('nested_chains', plain, quent)


# ---------------------------------------------------------------------------
# Async benchmarks
# ---------------------------------------------------------------------------

async def bench_async_simple():
  async def plain():
    return await af_mul_10(1)
  def quent():
    return Chain(1).then(af_mul_10).run()
  return await async_run_bench('async_simple (2 ops)', plain, quent)


async def bench_async_medium():
  async def plain():
    v = 1
    v = await af_mul_10(v)
    v = await af_add_5(v)
    v = await af_sub_3(v)
    v = await af_double(v)
    return v
  def quent():
    return Chain(1).then(af_mul_10).then(af_add_5).then(af_sub_3).then(af_double).run()
  return await async_run_bench('async_medium (4 ops)', plain, quent)


async def bench_async_reuse():
  async def plain():
    v = 1
    v = await af_mul_10(v)
    v = await af_add_5(v)
    return v
  def quent():
    return Chain(1).then(af_mul_10).then(af_add_5).run()
  c = Chain(1).then(af_mul_10).then(af_add_5)
  return await async_run_bench('async_reuse (2 ops)', plain, quent, c.run)


async def bench_async_foreach():
  items = list(range(50))
  async def plain():
    return [await af_mul_10(x) for x in items]
  def quent():
    return Chain(items).foreach(af_mul_10).run()
  return await async_run_bench('async_foreach (50 items)', plain, quent, loops=FOREACH_LOOPS)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
  print('\n' + '=' * 80)
  print('SUMMARY')
  print('=' * 80)
  print(f'{"Benchmark":<30} {"Plain":>10} {"Quent":>10} {"Overhead":>10} {"Reuse":>10} {"R.Overhead":>10}')
  print('-' * 80)
  for r in results:
    reuse_str = f'{r.reuse_time:.6f}' if r.reuse_time is not None else '-'
    reuse_ovh = f'{r.reuse_overhead_pct:+.1f}%' if r.reuse_overhead_pct is not None else '-'
    print(f'{r.name:<30} {r.plain_time:>10.6f} {r.quent_time:>10.6f} {r.overhead_pct:>+9.1f}% {reuse_str:>10} {reuse_ovh:>10}')
  print('=' * 80)


def print_comparison_summary(results):
  if not results:
    return
  label = results[0].baseline_label
  opt_label = results[0].optimized_label
  print('\n' + '=' * 70)
  print(f'COMPARISON SUMMARY ({label} vs {opt_label})')
  print('=' * 70)
  print(f'{"Benchmark":<30} {label:>12} {opt_label:>12} {"Speedup":>10}')
  print('-' * 70)
  for r in results:
    print(f'{r.name:<30} {r.baseline_time:>12.6f} {r.optimized_time:>12.6f} {r.speedup_pct:>+9.1f}%')
  print('=' * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_async_benchmarks():
  results = []
  results.append(await bench_async_simple())
  results.append(await bench_async_medium())
  results.append(await bench_async_reuse())
  results.append(await bench_async_foreach())
  return results


def main():
  print(f'Quent Performance Benchmark')
  print(f'LOOPS={LOOPS}, ITERATIONS={ITERATIONS}, FOREACH_LOOPS={FOREACH_LOOPS}')
  print()

  results = []

  print('Sync Benchmarks:')
  results.append(bench_simple_chain())
  results.append(bench_medium_chain())
  results.append(bench_long_chain())
  results.append(bench_do_side_effects())
  results.append(bench_foreach())
  results.append(bench_except_finally())
  results.append(bench_pipe_syntax())
  results.append(bench_nested_chains())

  print()
  print('Async Benchmarks:')
  async_results = asyncio.run(run_async_benchmarks())
  results.extend(async_results)

  print_summary(results)


if __name__ == '__main__':
  main()
