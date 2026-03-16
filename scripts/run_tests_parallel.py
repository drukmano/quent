#!/usr/bin/env python3
"""Run test modules in parallel with per-process coverage."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Slow modules are excluded by default. Include with QUENT_SLOW=1.
SLOW_MODULES = {'tests.bridge_tests', 'tests.property_tests', 'tests.repr_fuzz_tests'}


def main() -> int:
  test_dir = Path('tests')
  modules = sorted(f'tests.{f.stem}' for f in test_dir.glob('*_tests.py'))

  if not os.environ.get('QUENT_SLOW'):
    skipped = [m for m in modules if m in SLOW_MODULES]
    modules = [m for m in modules if m not in SLOW_MODULES]
    if skipped:
      print(f'Skipping slow modules (set QUENT_SLOW=1 to include): {", ".join(skipped)}')

  if not modules:
    print('No test modules found')
    return 1

  print(f'Running {len(modules)} test modules in parallel\n')
  t0 = time.monotonic()

  # Launch all test processes simultaneously
  procs: dict[str, subprocess.Popen[str]] = {}
  for mod in modules:
    # Stream stdout for slow modules so progress logs are visible in real-time
    stream_stdout = mod in SLOW_MODULES
    procs[mod] = subprocess.Popen(
      [sys.executable, '-m', 'coverage', 'run', '--parallel-mode', '-m', 'unittest', mod],
      stdout=None if stream_stdout else subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )

  # Collect results in module order
  failed: list[tuple[str, str]] = []
  for mod in modules:
    p = procs[mod]
    _, stderr = p.communicate()

    ok = p.returncode == 0
    summary = ''
    for line in stderr.strip().splitlines():
      if line.startswith('Ran '):
        summary = line
        break
    print(f'  {"OK" if ok else "FAIL":4s}  {mod}  {summary}')
    if not ok:
      failed.append((mod, stderr))

  # Combine per-process coverage data
  subprocess.run(
    [sys.executable, '-m', 'coverage', 'combine'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
  )

  elapsed = time.monotonic() - t0

  if failed:
    print(f'\n{"=" * 70}')
    for mod, stderr in failed:
      print(f'\n--- {mod} ---')
      print(stderr)
    print(f'\n{len(failed)}/{len(modules)} module(s) FAILED in {elapsed:.1f}s')
    return 1

  print(f'\nAll {len(modules)} modules passed in {elapsed:.1f}s')
  return 0


if __name__ == '__main__':
  sys.exit(main())
