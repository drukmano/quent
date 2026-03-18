# SPDX-License-Identifier: MIT
"""Python 3.11+ adaptive specialization viewer for quent hot-path functions.

Uses ``dis.dis(fn, adaptive=True)`` to show specialized bytecodes after
the adaptive interpreter has optimized them. Requires Python 3.11+ for
the ``adaptive`` parameter; requires Python 3.12+ for ``sys.monitoring``
specialization stats.

Usage:
    python benchmarks/analyze_specialization.py
"""

from __future__ import annotations

import dis
import sys


def _warm_up() -> None:
  """Execute pipelines to trigger adaptive specialization."""
  from benchmarks._helpers import make_sync_pipeline

  for _ in range(200):
    for n in (1, 5, 10):
      q = make_sync_pipeline(n)
      q.run(0)


def main() -> None:
  if sys.version_info < (3, 11):
    print(f'Python {sys.version}')
    print('Adaptive specialization requires Python 3.11+. Exiting.')
    sys.exit(0)

  print(f'Python {sys.version}')
  print('Warming up to trigger adaptive specialization...')
  _warm_up()

  from quent._engine import _run
  from quent._eval import _evaluate_value

  targets = [
    (_evaluate_value, '_evaluate_value'),
    (_run, '_run'),
  ]

  for fn, name in targets:
    print(f'\n{"=" * 80}')
    print(f'Specialized bytecode: {name}')
    print(f'{"=" * 80}')
    dis.dis(fn, adaptive=True)  # type: ignore[call-arg]

  # Python 3.12+ only: show specialization failure stats if available.
  if sys.version_info >= (3, 12) and hasattr(sys, '_stats_on'):
    print(f'\n{"=" * 80}')
    print('Specialization stats (Python 3.12+ only)')
    print(f'{"=" * 80}')
    print('Hint: build CPython with --enable-pystats and run with -X pystats')
    print('      to get detailed specialization counters.')

  print('\nDone. Inspect the specialized opcodes above for LOAD_ATTR_MODULE,')
  print('CALL_PY_EXACT_ARGS, and similar specialized forms indicating hot-path')
  print('optimization by the adaptive interpreter.')


if __name__ == '__main__':
  main()
