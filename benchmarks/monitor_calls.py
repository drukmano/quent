# SPDX-License-Identifier: MIT
"""Python 3.12+ sys.monitoring call counter for quent hot-path functions.

Counts calls to _evaluate_value, _run, _run_async, and Link.__init__
during a standard workload using the low-overhead sys.monitoring API.
Falls back to a simpler manual counter on Python < 3.12.

Usage:
    python benchmarks/monitor_calls.py
"""

from __future__ import annotations

import sys

from benchmarks._helpers import (
  DummyCM,
  add_one,
  identity,
  make_sync_pipeline,
  noop,
)
from quent import Q
from quent._engine import _run, _run_async
from quent._eval import _evaluate_value
from quent._link import Link


def workload() -> None:
  """Run a standard workload to count function calls."""
  for _ in range(1000):
    for n in (1, 5, 10):
      q = make_sync_pipeline(n)
      q.run(0)

    data = list(range(20))
    Q(data).foreach(add_one).run()
    Q(data).foreach_do(noop).run()
    Q().gather(identity, identity, identity).run(42)
    Q(DummyCM(99)).with_(identity).run()


# ---- sys.monitoring path (Python 3.12+) ----


def _run_with_monitoring() -> dict[str, int]:
  """Use sys.monitoring to count calls to hot-path functions."""
  TOOL_ID = sys.monitoring.DEBUGGER_ID
  call_counts: dict[str, int] = {
    '_evaluate_value': 0,
    '_run': 0,
    '_run_async': 0,
    'Link.__init__': 0,
  }
  tracked_codes: dict[int, str] = {}

  targets = [
    (_evaluate_value, '_evaluate_value'),
    (_run, '_run'),
    (_run_async, '_run_async'),
    (Link.__init__, 'Link.__init__'),
  ]
  for fn, name in targets:
    code = fn.__code__
    tracked_codes[id(code)] = name

  def _call_handler(code: object, instruction_offset: int, callable: object, arg0: object) -> object:
    if hasattr(callable, '__code__'):
      name = tracked_codes.get(id(callable.__code__))  # type: ignore[union-attr]
      if name is not None:
        call_counts[name] += 1
    return sys.monitoring.DISABLE  # type: ignore[attr-defined, return-value]

  sys.monitoring.use_tool_id(TOOL_ID, 'quent_bench')
  sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.CALL)
  sys.monitoring.register_callback(TOOL_ID, sys.monitoring.events.CALL, _call_handler)

  try:
    workload()
  finally:
    sys.monitoring.set_events(TOOL_ID, 0)
    sys.monitoring.free_tool_id(TOOL_ID)

  return call_counts


# ---- Manual wrapping path (Python < 3.12) ----


def _run_with_wrapping() -> dict[str, int]:
  """Count calls by wrapping target functions (Python < 3.12 fallback)."""
  import functools

  call_counts: dict[str, int] = {
    '_evaluate_value': 0,
    '_run': 0,
    '_run_async': 0,
    'Link.__init__': 0,
  }

  import quent._engine as _engine_mod
  import quent._eval as _eval_mod
  import quent._link as _link_mod

  original_evaluate = _eval_mod._evaluate_value
  original_run = _engine_mod._run
  original_run_async = _engine_mod._run_async
  original_link_init = Link.__init__

  @functools.wraps(original_evaluate)
  def _counted_evaluate(*args, **kwargs):  # type: ignore[no-untyped-def]
    call_counts['_evaluate_value'] += 1
    return original_evaluate(*args, **kwargs)

  @functools.wraps(original_run)
  def _counted_run(*args, **kwargs):  # type: ignore[no-untyped-def]
    call_counts['_run'] += 1
    return original_run(*args, **kwargs)

  @functools.wraps(original_run_async)
  async def _counted_run_async(*args, **kwargs):  # type: ignore[no-untyped-def]
    call_counts['_run_async'] += 1
    return await original_run_async(*args, **kwargs)

  @functools.wraps(original_link_init)
  def _counted_link_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    call_counts['Link.__init__'] += 1
    original_link_init(self, *args, **kwargs)

  _eval_mod._evaluate_value = _counted_evaluate  # type: ignore[assignment]
  _engine_mod._run = _counted_run  # type: ignore[assignment]
  _engine_mod._run_async = _counted_run_async  # type: ignore[assignment]
  _link_mod.Link.__init__ = _counted_link_init  # type: ignore[assignment]

  try:
    workload()
  finally:
    _eval_mod._evaluate_value = original_evaluate  # type: ignore[assignment]
    _engine_mod._run = original_run  # type: ignore[assignment]
    _engine_mod._run_async = original_run_async  # type: ignore[assignment]
    _link_mod.Link.__init__ = original_link_init  # type: ignore[assignment]

  return call_counts


def main() -> None:
  print(f'Python {sys.version}')

  if sys.version_info >= (3, 12):
    print('Using sys.monitoring for call counting...')
    counts = _run_with_monitoring()
  else:
    print('Python < 3.12: using function wrapping for call counting...')
    counts = _run_with_wrapping()

  print(f'\n{"=" * 60}')
  print('Call counts (1000 iterations)')
  print(f'{"=" * 60}')
  for name, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {name:<30} {count:>10,}')
  print(f'{"=" * 60}')


if __name__ == '__main__':
  main()
