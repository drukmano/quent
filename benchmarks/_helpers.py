# SPDX-License-Identifier: MIT
"""Shared pipeline factories and workloads for benchmark scripts."""

from __future__ import annotations

import asyncio
import statistics
from contextlib import contextmanager
from typing import Any

from quent import Q
from quent._link import Link

# ---- Constants ----

DEFAULT_ITERATIONS: int = 1000
STANDARD_SIZES: tuple[int, ...] = (1, 5, 10, 50, 100)

# ---- Atomic callables ----


def identity(x: Any) -> Any:
  return x


async def async_identity(x: Any) -> Any:
  return x


def add_one(x: Any) -> Any:
  return x + 1


async def async_add_one(x: Any) -> Any:
  return x + 1


def noop(*args: Any, **kwargs: Any) -> None:
  pass


def predicate_true(x: Any) -> bool:
  return True


def predicate_false(x: Any) -> bool:
  return False


# ---- Context manager ----


class DummyCM:
  """Simple sync context manager that yields a value."""

  __slots__ = ('_value',)

  def __init__(self, value: Any = 42) -> None:
    self._value = value

  def __enter__(self) -> Any:
    return self._value

  def __exit__(self, *args: Any) -> None:
    pass


@contextmanager
def dummy_cm(value: Any = 42):  # type: ignore[no-untyped-def]
  yield value


# ---- Pipeline factories ----


def make_sync_pipeline(n_links: int) -> Q:
  """Create a pipeline with *n_links* ``.then(identity)`` steps."""
  q = Q()
  for _ in range(n_links):
    q.then(identity)
  return q


def make_async_pipeline(n_links: int) -> Q:
  """Create a pipeline with *n_links* ``.then(async_identity)`` steps."""
  q = Q()
  for _ in range(n_links):
    q.then(async_identity)
  return q


def make_mixed_pipeline(n_links: int) -> Q:
  """Create a pipeline alternating sync and async identity steps."""
  q = Q()
  for i in range(n_links):
    if i % 2 == 0:
      q.then(identity)
    else:
      q.then(async_identity)
  return q


# ---- Execution helpers ----


def run_sync(q: Q, value: Any = 0) -> Any:
  """Execute a pipeline synchronously."""
  return q.run(value)


def run_async(q: Q, value: Any = 0) -> Any:
  """Execute a pipeline via asyncio.run."""
  return asyncio.run(q.run(value))


# ---- Link factory ----


def make_link(fn: Any = identity, current_value: Any = 0) -> tuple[Link, Any]:
  """Create a Link and a current_value for microbenchmarks.

  Returns (link, current_value) so callers can pass them to _evaluate_value.
  """
  return Link(fn), current_value


# ---- Statistics helpers ----


def summarize(samples: list[float], label: str) -> None:
  """Print mean, stdev, min/max and percentiles for *samples* (in seconds)."""
  if not samples:
    print(f'{label}: no data')
    return
  mean = statistics.mean(samples)
  stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
  lo = min(samples)
  hi = max(samples)
  sorted_s = sorted(samples)
  p50 = sorted_s[len(sorted_s) // 2]
  p95 = sorted_s[int(len(sorted_s) * 0.95)]
  print(
    f'{label:<40} mean={mean * 1e6:8.2f}µs  stdev={stdev * 1e6:7.2f}µs'
    f'  p50={p50 * 1e6:8.2f}µs  p95={p95 * 1e6:8.2f}µs'
    f'  [{lo * 1e6:.2f}µs - {hi * 1e6:.2f}µs]'
  )


def print_table(rows: list[tuple[str, float]], title: str = '') -> None:
  """Print a simple two-column table of (label, time_in_seconds)."""
  if title:
    print(f'\n{title}')
    print('-' * len(title))
  col = max(len(r[0]) for r in rows) + 2
  for label, t in rows:
    print(f'  {label:<{col}} {t * 1e6:10.3f} µs')
