"""Bind release approval to successful push CI for one trusted commit.

The workflow invokes this from its trusted checkout before checking out the
release candidate. This script only reads GitHub state; it never publishes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def _trusted_success(run: dict[str, Any], repository: str) -> bool:
  return (
    run.get('event') == 'push'
    and run.get('head_branch') == 'master'
    and (run.get('head_repository') or {}).get('full_name') == repository
    and run.get('status') == 'completed'
    and run.get('conclusion') == 'success'
  )


def select_sha(event_name: str, event: dict[str, Any], repository: str, ref: str, sha: str) -> str:
  if event_name == 'workflow_run':
    run = event.get('workflow_run', {})
    if not _trusted_success(run, repository):
      raise ValueError('Automatic releases require successful push CI from this repository on master')
    sha = run.get('head_sha', '')
  elif event_name != 'workflow_dispatch' or ref != 'refs/heads/master':
    raise ValueError('Manual releases must run from master')
  if not re.fullmatch(r'[0-9a-f]{40}', sha):
    raise ValueError('Release candidate must be a full commit SHA')
  return sha


def verify_ci(repository: str, sha: str) -> None:
  # The workflow endpoint, unlike commit statuses, identifies our actual CI.
  response = subprocess.check_output(
    [
      'gh',
      'api',
      '--method',
      'GET',
      f'repos/{repository}/actions/workflows/ci.yml/runs',
      '-f',
      f'head_sha={sha}',
      '-f',
      'event=push',
      '-f',
      'branch=master',
      '-f',
      'status=success',
      '-f',
      'per_page=100',
    ],
    text=True,
  )
  runs = json.loads(response).get('workflow_runs', [])
  if not any(run.get('head_sha') == sha and _trusted_success(run, repository) for run in runs):
    raise ValueError(f'No successful trusted CI run for {sha}')


def main() -> None:
  env = os.environ
  event = json.loads(Path(env['GITHUB_EVENT_PATH']).read_text())
  sha = select_sha(env['GITHUB_EVENT_NAME'], event, env['GITHUB_REPOSITORY'], env['GITHUB_REF'], env['GITHUB_SHA'])
  verify_ci(env['GITHUB_REPOSITORY'], sha)
  with Path(env['GITHUB_OUTPUT']).open('a') as output:
    output.write(f'sha={sha}\n')


if __name__ == '__main__':
  main()
