---
title: "Performance & Benchmarks"
description: "Benchmark results showing quent's overhead on sync and async pipelines, and how it compares to raw function calls and real I/O operations."
tags:
  - performance
  - benchmarks
  - overhead
search:
  boost: 5
---

# Performance & Benchmarks

quent adds dispatch overhead to each pipeline step. This page quantifies that overhead and puts it in context. The takeaway: for I/O-bound workloads -- quent's primary use case -- the overhead is a rounding error.

All benchmarks: Python 3.14.3, Apple M-series (ARM64), macOS. Measured with `timeit` (5 repeats, best-of-5). Scripts in [`benchmarks/`](https://github.com/drukmano/quent/tree/master/benchmarks).

---

## At a Glance

| Pipeline | 1 Step | 5 Steps | 10 Steps | Per Step |
|:---------|:------:|:-------:|:--------:|:--------:|
| **Sync** | 0.8 us | 1.7 us | 2.7 us | ~210 ns |
| **Async** | 17.5 us | 33 us | 53 us | ~4 us |
| **Mixed** (sync+async) | -- | 22 us | 34 us | -- |

Baselines: raw function call ~26 ns. Bare `await coroutine()` ~2.8 us.

---

## Sync Pipeline

Fully synchronous pipelines execute in a tight `while` loop. No event loop, no coroutines, no async machinery.

| Benchmark | Time |
|:----------|-----:|
| Raw `fn(x)` call | 26 ns |
| Raw 5-call chain `fn(fn(fn(fn(fn(x)))))` | 118 ns |
| Raw 10-call chain | 229 ns |
| `Q().run()` (empty pipeline) | 822 ns |
| 1-step pipeline | 817 ns |
| 5-step pipeline | 1,675 ns |
| 10-step pipeline | 2,734 ns |

### Per-step breakdown

The fixed pipeline overhead (empty `Q().run()`) is **~820 ns**. Each additional step adds **~210 ns**:

- ~26 ns -- the function call itself
- ~30 ns -- frozenset awaitable type check (`_SYNC_TYPES` lookup)
- ~154 ns -- linked-list traversal, calling convention dispatch, state management

The per-step cost scales linearly. A 100-step sync pipeline takes approximately 820 + (100 x 210) = **21.8 us**.

---

## Async Pipeline

Async benchmarks run inside a running event loop (no `asyncio.run()` startup cost per iteration). This measures quent's actual dispatch overhead, not event loop bootstrapping.

| Benchmark | Time |
|:----------|-----:|
| Bare `await async_fn(x)` | 2,846 ns |
| 1-step async pipeline | 17,499 ns |
| 5-step async pipeline | 33,425 ns |
| 10-step async pipeline | 52,890 ns |
| 5-step mixed (3 sync + 2 async) | 22,137 ns |
| 10-step mixed (5 sync + 5 async) | 34,388 ns |

### Per-step breakdown

The first async step pays a one-time transition cost of **~14.7 us** (17.5 us total - 2.8 us baseline). Each subsequent async step adds **~4 us**.

Mixed pipelines are faster because sync steps within the async continuation still only pay ~210 ns, not the full ~4 us async per-step cost.

!!! tip "Why mixed pipelines matter"
    Real-world pipelines are rarely all-async. A typical 5-step pipeline might have 2 I/O steps (async) and 3 computation/validation steps (sync). The mixed pipeline benchmark (**22 us** for 5 steps) is the most realistic number for most applications.

---

## I/O-Bound Workloads

quent's primary use case is I/O-bound pipelines -- database queries, HTTP calls, cache lookups. Pipeline overhead is a constant ~22 us for a typical 5-step mixed pipeline. I/O latency is orders of magnitude larger.

### Calculated overhead

| I/O Operation | Typical Latency | Pipeline Overhead | % of Total |
|:--------------|----------------:|:-----------------:|:----------:|
| Slow cache (network Redis) | 500 us | 22 us | 4.2% |
| Fast database query | 1 ms | 22 us | 2.2% |
| Typical database query | 5 ms | 22 us | 0.4% |
| HTTP API (same region) | 10 ms | 22 us | 0.2% |
| HTTP API (cross-region) | 50 ms | 22 us | 0.04% |
| External API | 200 ms | 22 us | 0.01% |

!!! note "Sync-only pipelines with sync I/O"
    If your pipeline is fully synchronous (e.g. using synchronous Redis or file I/O), the overhead drops to **~1.7 us** for 5 steps instead of 22 us. Sync-over-sync pipelines have 10x lower overhead than mixed async pipelines.

### Measured with simulated I/O

These benchmarks use `asyncio.sleep()` to simulate real I/O. The pipeline has 3 steps: sync &rarr; async I/O &rarr; sync.

| Scenario | Raw | Pipeline (3 steps) | Measured Delta |
|:---------|----:|-------------------:|---------------:|
| 1 ms I/O | 1.60 ms | 1.81 ms | ~0.2 ms |
| 5 ms I/O | 7.50 ms | 7.75 ms | ~0.3 ms |
| 50 ms I/O | 53.9 ms | 53.1 ms | noise |

!!! note
    The measured deltas for 1 ms and 5 ms I/O are larger than the calculated ~22 us pipeline overhead because `asyncio.sleep()` has significant timer jitter at the millisecond scale. The 50 ms measurement confirms that pipeline overhead is lost in the noise for realistic network I/O.

---

## Pipeline Construction

Pipeline construction is a build-time cost, not a runtime cost. Pipelines are typically built once and executed many times.

| Benchmark | Time |
|:----------|-----:|
| `Q()` (empty) | 515 ns |
| `Q()` + 5x `.then()` | 7.4 us |
| `Q()` + 10x `.then()` | 14.6 us |

Each `.then()` call creates a `Link` node and appends it to the linked list. If construction cost matters for your use case, build once and call `.run()` repeatedly -- or use `.clone()` to fork a pre-built base pipeline.

---

## What Makes It Fast

quent's engine is optimized for the common case at every decision point:

### Fast awaitable check (~30 ns)

A `frozenset` of common sync return types (`int`, `str`, `float`, `bool`, `list`, `dict`, `tuple`, `set`, `bytes`) rejects non-awaitables in a single O(1) lookup. This is **~10x faster** than `inspect.isawaitable()` (~380 ns), which goes through the ABC machinery.

```python
# _engine.py — hot path
_SYNC_TYPES = frozenset({int, str, float, bool, list, dict, tuple, set, bytes})

# After each step:
if type(result) is CoroutineType or (
  result is not None and type(result) not in _SYNC_TYPES and _isawaitable(result)
):
  # async transition
```

### Hot-path calling convention

~90% of pipeline steps use the default calling convention (no explicit args). The dispatch front-loads this path:

```python
# _eval.py — _evaluate_value hot path
if not link.args and not link.kwargs:  # ~2ns truthiness check on None
  if link.is_callable:
    return link.v(current_value) if current_value is not Null else link.v()
  return link.v
```

### Zero instrumentation overhead

When `on_step` is `None` (the default), all timing and callback logic is completely bypassed -- not a no-op callback, but a short-circuited code path. Zero cost when you don't use it.

### One-way async transition

The sync-to-async transition happens at most **once** per pipeline execution. Once in async mode, the engine never checks whether to go back to sync. This means N async steps pay one transition cost plus N marginal costs -- not N transition costs.

### No async imports on sync path

Sync pipelines never touch `asyncio` at evaluation time. No event loop interaction, no coroutine creation, no async frame allocation.

See [Sync/Async Bridging -- Performance](async.md#performance-zero-async-overhead-for-sync-pipelines) for more on the zero-overhead sync path.

---

## Running Benchmarks

The [`benchmarks/`](https://github.com/drukmano/quent/tree/master/benchmarks) directory contains reproducible scripts:

```bash
# Quick overview (timeit-based, ~2 minutes)
python benchmarks/bench_for_docs.py

# Rigorous microbenchmarks (pyperf, ~15 minutes each)
python benchmarks/bench_core.py          # core operations
python benchmarks/bench_async.py         # async execution
python benchmarks/bench_ops.py           # foreach, gather, with_, if_
python benchmarks/bench_q_sizes.py       # scaling from 1 to 1000 steps

# Profiling
python benchmarks/profile_cprofile.py    # cProfile
bash benchmarks/flamegraph_pyspy.sh      # py-spy flamegraph
```

!!! tip
    For the most stable results, close other applications and use `pyperf` with `--rigorous`. The `bench_for_docs.py` script uses `timeit` for quick, reproducible results.

---

## Further Reading

- **[Sync/Async Bridging](async.md)** -- the two-tier execution model and zero-overhead sync path
- **[Pipelines & Methods](pipelines.md)** -- all pipeline-building operations
