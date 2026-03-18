#!/usr/bin/env python3
"""Run test modules in parallel with optional coverage.

Discovers test modules from tests/*_tests.py and launches one subprocess per
module, limited to os.cpu_count() concurrent processes.

When --coverage is passed (or the QUENT_COVERAGE env var is set), each module
runs under `coverage run --parallel-mode` and a combined report + XML output
are generated after all modules finish.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _use_coverage() -> bool:
  """Return True if coverage instrumentation is requested."""
  if '--coverage' in sys.argv:
    return True
  return os.environ.get('QUENT_COVERAGE', '').strip() not in ('', '0')


def _run_module(mod: str, *, coverage: bool) -> tuple[str, int, str]:
  """Run a single test module and return (mod, returncode, stderr)."""
  if coverage:
    cmd = [
      sys.executable,
      '-m',
      'coverage',
      'run',
      '--parallel-mode',
      '-m',
      'unittest',
      mod,
    ]
  else:
    cmd = [sys.executable, '-m', 'unittest', mod]

  p = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
  )
  return mod, p.returncode, p.stderr


def _coverage_finalize() -> int:
  """Combine parallel coverage data and produce report + XML."""
  print('\n==> Combining coverage data')
  rc = subprocess.run([sys.executable, '-m', 'coverage', 'combine']).returncode
  if rc != 0:
    print('coverage combine failed')
    return rc

  print('\n==> Coverage report')
  rc = subprocess.run([sys.executable, '-m', 'coverage', 'report']).returncode
  if rc != 0:
    # fail_under threshold not met — still produce XML for upload
    print('(coverage report exited non-zero — threshold may not be met)')

  print('\n==> Generating coverage.xml')
  subprocess.run([sys.executable, '-m', 'coverage', 'xml'])
  return 0


def main() -> int:
  coverage = _use_coverage()
  test_dir = Path('tests')
  modules = sorted(f'tests.{f.stem}' for f in test_dir.glob('*_tests.py'))

  if not modules:
    print('No test modules found')
    return 1

  # Clean stale coverage data before starting
  if coverage:
    for p in Path('.').glob('.coverage*'):
      if p.is_file():
        p.unlink()
    cov_xml = Path('coverage.xml')
    if cov_xml.exists():
      cov_xml.unlink()

  max_workers = os.cpu_count() or 4
  mode = ' (with coverage)' if coverage else ''
  print(f'Running {len(modules)} test modules in parallel{mode}\n')
  t0 = time.monotonic()

  results: dict[str, tuple[int, str]] = {}

  try:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {executor.submit(_run_module, mod, coverage=coverage): mod for mod in modules}
      try:
        for future in as_completed(futures):
          mod, returncode, stderr = future.result()
          results[mod] = (returncode, stderr)
      except KeyboardInterrupt:
        print('\nInterrupted — killing child processes...')
        executor.shutdown(wait=False, cancel_futures=True)
        return 130
  except KeyboardInterrupt:
    print('\nInterrupted — killing child processes...')
    return 130

  # Print results in launch order
  failed: list[tuple[str, str]] = []
  for mod in modules:
    returncode, stderr = results[mod]
    ok = returncode == 0
    summary = ''
    for line in stderr.strip().splitlines():
      if line.startswith('Ran '):
        summary = line
        break
    print(f'  {"OK" if ok else "FAIL":4s}  {mod}  {summary}')
    if not ok:
      failed.append((mod, stderr))

  elapsed = time.monotonic() - t0

  if failed:
    print(f'\n{"=" * 70}')
    for name, stderr in failed:
      print(f'\n--- {name} ---')
      print(stderr)
    print(f'\n{len(failed)}/{len(modules)} module(s) FAILED in {elapsed:.1f}s')
    return 1

  print(f'\nAll {len(modules)} modules passed in {elapsed:.1f}s')

  if coverage:
    rc = _coverage_finalize()
    if rc != 0:
      return rc

  return 0


if __name__ == '__main__':
  sys.exit(main())
