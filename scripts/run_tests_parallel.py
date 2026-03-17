#!/usr/bin/env python3
"""Run test modules in parallel.

Discovers test modules from tests/*_tests.py and launches one subprocess per
module.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
  test_dir = Path('tests')
  modules = sorted(f'tests.{f.stem}' for f in test_dir.glob('*_tests.py'))

  if not modules:
    print('No test modules found')
    return 1

  print(f'Running {len(modules)} test modules in parallel\n')
  t0 = time.monotonic()

  # Launch all test processes simultaneously
  procs: dict[str, subprocess.Popen[str]] = {}
  for mod in modules:
    procs[mod] = subprocess.Popen(
      [sys.executable, '-m', 'unittest', mod],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )

  def _kill_all() -> None:
    """Terminate all child processes, then kill any that don't stop."""
    for p in procs.values():
      if p.poll() is None:
        p.send_signal(signal.SIGTERM)
    for p in procs.values():
      try:
        p.wait(timeout=2)
      except subprocess.TimeoutExpired:
        p.kill()
    for p in procs.values():
      p.wait()

  # Collect results in launch order
  try:
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
  except KeyboardInterrupt:
    print('\nInterrupted — killing child processes...')
    _kill_all()
    return 130

  elapsed = time.monotonic() - t0

  if failed:
    print(f'\n{"=" * 70}')
    for name, stderr in failed:
      print(f'\n--- {name} ---')
      print(stderr)
    print(f'\n{len(failed)}/{len(modules)} module(s) FAILED in {elapsed:.1f}s')
    return 1

  print(f'\nAll {len(modules)} modules passed in {elapsed:.1f}s')
  return 0


if __name__ == '__main__':
  sys.exit(main())
