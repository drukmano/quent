# SPDX-License-Identifier: MIT
"""pyperf benchmarks for core operations: _evaluate_value, Link, Chain.run.

Run with:
    python benchmarks/bench_core.py
    python benchmarks/bench_core.py --fast    # quick smoke-test
    python benchmarks/bench_core.py -o results.json  # save results
"""

from __future__ import annotations

import pyperf

from benchmarks._helpers import identity, noop
from quent import Chain
from quent._eval import _evaluate_value
from quent._link import Link

# ---- Benchmark functions ----
# Each function follows the pyperf bench_time_func protocol: the first argument
# is the number of loops (injected by pyperf), and the body runs that many
# iterations of the operation under test, returning elapsed time in seconds.


def bench_raw_call_overhead(loops: int) -> float:
  """Baseline: raw Python function call with no chain overhead."""
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    identity(42)
  return pyperf.perf_counter() - t0


def bench_evaluate_value_passthrough(loops: int) -> float:
  """Hot path: callable link with current_value present."""
  link = Link(identity)
  current_value = 42
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    _evaluate_value(link, current_value)
  return pyperf.perf_counter() - t0


def bench_evaluate_value_no_args(loops: int) -> float:
  """Callable link, no current_value (Null sentinel)."""
  link = Link(noop)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    _evaluate_value(link)
  return pyperf.perf_counter() - t0


def bench_evaluate_value_with_args(loops: int) -> float:
  """Callable link with explicit args (current value NOT passed)."""
  link = Link(identity, args=(99,))
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    _evaluate_value(link, 42)
  return pyperf.perf_counter() - t0


def bench_evaluate_value_non_callable(loops: int) -> float:
  """Non-callable link: raw value passthrough (no call)."""
  link = Link(42)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    _evaluate_value(link)
  return pyperf.perf_counter() - t0


def bench_link_init(loops: int) -> float:
  """Link construction with a callable."""
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    Link(identity)
  return pyperf.perf_counter() - t0


def bench_link_init_with_args(loops: int) -> float:
  """Link construction with args and kwargs."""
  args = (1, 2)
  kwargs = {'key': 'value'}
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    Link(identity, args=args, kwargs=kwargs)
  return pyperf.perf_counter() - t0


def bench_chain_construction_empty(loops: int) -> float:
  """Empty Chain() construction — build time only."""
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    Chain()
  return pyperf.perf_counter() - t0


def bench_chain_construction_5(loops: int) -> float:
  """Chain construction: 5 .then() calls, no execution."""
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    c = Chain()
    c.then(identity)
    c.then(identity)
    c.then(identity)
    c.then(identity)
    c.then(identity)
  return pyperf.perf_counter() - t0


def bench_chain_run_empty(loops: int) -> float:
  """Chain().run() with no links."""
  chain = Chain()
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run()
  return pyperf.perf_counter() - t0


def bench_chain_run_1(loops: int) -> float:
  """Chain with 1 .then(identity) link."""
  chain = Chain().then(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run(0)
  return pyperf.perf_counter() - t0


def bench_chain_run_5(loops: int) -> float:
  """Chain with 5 .then(identity) links."""
  chain = Chain()
  for _ in range(5):
    chain.then(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run(0)
  return pyperf.perf_counter() - t0


def bench_chain_run_10(loops: int) -> float:
  """Chain with 10 .then(identity) links."""
  chain = Chain()
  for _ in range(10):
    chain.then(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run(0)
  return pyperf.perf_counter() - t0


def bench_chain_run_50(loops: int) -> float:
  """Chain with 50 .then(identity) links."""
  chain = Chain()
  for _ in range(50):
    chain.then(identity)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run(0)
  return pyperf.perf_counter() - t0


def bench_then_vs_do(loops: int) -> float:
  """then() vs do(): measure do() (side-effect, value not propagated)."""
  chain = Chain()
  for _ in range(5):
    chain.do(noop)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run(0)
  return pyperf.perf_counter() - t0


def bench_lambda_callable(loops: int) -> float:
  """Chain using lambda callables instead of named functions."""
  chain = Chain()
  for _ in range(5):
    chain.then(lambda x: x)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run(0)
  return pyperf.perf_counter() - t0


class _Incrementer:
  def __call__(self, x: int) -> int:
    return x + 1


def bench_class_callable(loops: int) -> float:
  """Chain using __call__ instances (method dispatch overhead)."""
  fn = _Incrementer()
  chain = Chain()
  for _ in range(5):
    chain.then(fn)
  t0 = pyperf.perf_counter()
  for _ in range(loops):
    chain.run(0)
  return pyperf.perf_counter() - t0


if __name__ == '__main__':
  runner = pyperf.Runner()
  runner.bench_time_func('raw_call_overhead', bench_raw_call_overhead)
  runner.bench_time_func('evaluate_value_passthrough', bench_evaluate_value_passthrough)
  runner.bench_time_func('evaluate_value_no_args', bench_evaluate_value_no_args)
  runner.bench_time_func('evaluate_value_with_args', bench_evaluate_value_with_args)
  runner.bench_time_func('evaluate_value_non_callable', bench_evaluate_value_non_callable)
  runner.bench_time_func('link_init', bench_link_init)
  runner.bench_time_func('link_init_with_args', bench_link_init_with_args)
  runner.bench_time_func('chain_construction_empty', bench_chain_construction_empty)
  runner.bench_time_func('chain_construction_5', bench_chain_construction_5)
  runner.bench_time_func('chain_run_empty', bench_chain_run_empty)
  runner.bench_time_func('chain_run_1', bench_chain_run_1)
  runner.bench_time_func('chain_run_5', bench_chain_run_5)
  runner.bench_time_func('chain_run_10', bench_chain_run_10)
  runner.bench_time_func('chain_run_50', bench_chain_run_50)
  runner.bench_time_func('then_vs_do', bench_then_vs_do)
  runner.bench_time_func('lambda_callable', bench_lambda_callable)
  runner.bench_time_func('class_callable', bench_class_callable)
