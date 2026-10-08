"""Regression tests for local QA and release safety gates."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from scripts import run_tests_parallel

ROOT = Path(__file__).resolve().parents[1]


class CoverageGateTests(TestCase):
  def test_all_command_failures_reach_the_caller(self):
    for statuses, expected in [((0, 0, 0), 0), ((0, 2, 0), 2), ((0, 0, 1), 1), ((0, 2, 1), 2), ((1,), 1)]:
      with self.subTest(statuses=statuses):
        responses = [SimpleNamespace(returncode=code) for code in statuses]
        with patch.object(run_tests_parallel.subprocess, 'run', side_effect=responses) as run:
          self.assertEqual(run_tests_parallel._coverage_finalize(), expected)
          self.assertEqual(run.call_count, len(statuses))


class LocalRunnerTests(TestCase):
  def test_default_environment_and_arguments(self):
    bash = shutil.which('bash')
    if os.name == 'nt' or bash is None:
      self.skipTest('POSIX shell entry point')
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      script = root / 'run_tests.sh'
      script.write_text((ROOT / 'run_tests.sh').read_text())
      binaries = root / '.venv' / 'bin'
      binaries.mkdir(parents=True)
      for name in ('ruff', 'mypy', 'python'):
        executable = binaries / name
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
        executable.chmod(0o755)
      env = dict(os.environ)
      env.pop('VIRTUAL_ENV', None)
      env['PATH'] = '/usr/bin:/bin'
      result = subprocess.run(
        [bash, str(script), '--slow'], cwd=root, env=env, capture_output=True, text=True, timeout=10
      )
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn('scripts/run_tests_parallel.py --slow', result.stdout)


class ReleaseGateTests(TestCase):
  def setUp(self):
    path = ROOT / 'scripts' / 'verify_release.py'
    self.assertTrue(path.exists(), 'Release approval must validate the designated CI workflow, not legacy statuses')
    spec = importlib.util.spec_from_file_location('verify_release', path)
    self.policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(self.policy)
    self.sha = 'a' * 40
    self.repo = 'owner/project'
    self.run = {
      'head_sha': self.sha,
      'event': 'push',
      'head_branch': 'master',
      'head_repository': {'full_name': self.repo},
      'status': 'completed',
      'conclusion': 'success',
    }

  def test_automatic_release_selects_ci_sha_not_default_branch_tip(self):
    selected = self.policy.select_sha(
      'workflow_run', {'workflow_run': self.run}, self.repo, 'refs/heads/master', 'b' * 40
    )
    self.assertEqual(selected, self.sha)

  def test_untrusted_or_failed_triggers_are_rejected(self):
    changes = [
      {'event': 'pull_request'},
      {'head_repository': {'full_name': 'fork/project'}},
      {'head_branch': 'feature'},
      {'status': 'in_progress'},
      {'conclusion': 'failure'},
      {'head_sha': 'not-a-sha'},
    ]
    for change in changes:
      with self.subTest(change=change), self.assertRaises(ValueError):
        self.policy.select_sha('workflow_run', {'workflow_run': self.run | change}, self.repo, '', '')

  def test_manual_release_is_restricted_to_master(self):
    self.assertEqual(
      self.policy.select_sha('workflow_dispatch', {}, self.repo, 'refs/heads/master', self.sha), self.sha
    )
    with self.assertRaises(ValueError):
      self.policy.select_sha('workflow_dispatch', {}, self.repo, 'refs/heads/feature', self.sha)
    with self.assertRaises(ValueError):
      self.policy.select_sha('pull_request', {}, self.repo, 'refs/heads/master', self.sha)

  def test_ci_verification_requires_matching_successful_trusted_push(self):
    bad_runs = [
      [],
      [self.run | {'head_sha': 'b' * 40}],
      [self.run | {'conclusion': 'failure'}],
      [self.run | {'event': 'pull_request'}],
      [self.run | {'head_repository': {'full_name': 'fork/project'}}],
    ]
    for runs in bad_runs:
      with (
        self.subTest(runs=runs),
        patch.object(self.policy.subprocess, 'check_output', return_value=json.dumps({'workflow_runs': runs})),
        self.assertRaises(ValueError),
      ):
        self.policy.verify_ci(self.repo, self.sha)
    with patch.object(
      self.policy.subprocess, 'check_output', return_value=json.dumps({'workflow_runs': [self.run]})
    ) as query:
      self.policy.verify_ci(self.repo, self.sha)
      args = query.call_args.args[0]
      self.assertIn(f'repos/{self.repo}/actions/workflows/ci.yml/runs', args)
      self.assertIn(f'head_sha={self.sha}', args)

  def test_workflow_builds_the_verified_sha_and_allows_same_tag_retries(self):
    workflow = (ROOT / '.github/workflows/release.yml').read_text()
    self.assertIn('ref: ${{ steps.verify.outputs.sha }}', workflow)
    self.assertIn('ref: ${{ needs.release.outputs.sha }}', workflow)
    self.assertIn('--target "$RELEASE_SHA"', workflow)
    self.assertIn('git rev-parse "$TAG^{commit}"', workflow)
    self.assertIn('skip-existing: true', workflow)
    self.assertNotIn('commits/$SHA/status', workflow)

  def test_release_tag_decisions_without_remote_side_effects(self):
    bash = shutil.which('bash')
    if os.name == 'nt' or bash is None:
      self.skipTest('GitHub release job uses a POSIX shell')
    workflow = (ROOT / '.github/workflows/release.yml').read_text()
    step = workflow.split('- name: Prepare verified release tag\n', 1)[1]
    script = textwrap.dedent(step.split('run: |\n', 1)[1].split('\n        env:', 1)[0])
    cases = [
      # An absent tag creates a release; an existing matching tag retries it.
      (2, self.sha, 0, 'true', True),
      (0, self.sha, 0, 'true', False),
      (0, 'b' * 40, 0, 'false', False),
      (128, self.sha, 128, None, False),
      # Re-check after creation: never build if the remote tag resolves elsewhere.
      (2, 'b' * 40, 1, None, True),
    ]
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      binaries = root / 'bin'
      binaries.mkdir()
      (binaries / 'python3').symlink_to(sys.executable)
      for name, body in {
        'git': (
          'case "$1" in\n'
          '  ls-remote) exit "$REMOTE_STATUS" ;;\n'
          '  fetch) exit 0 ;;\n'
          '  rev-parse) printf "%s\\n" "$TAG_SHA" ;;\n'
          '  *) exit 99 ;;\n'
          'esac\n'
        ),
        'gh': 'printf "%s\\n" "$*" >> "$GH_CALLS"\n',
      }.items():
        executable = binaries / name
        executable.write_text('#!/bin/sh\n' + body)
        executable.chmod(0o755)
      # Keep this shell test compatible with the library's Python 3.10 floor.
      # The hosted release runner supplies Python 3.11+ and tomllib.
      (root / 'tomllib.py').write_text("def load(file): return {'project': {'version': '1.2.3'}}\n")
      (root / 'pyproject.toml').write_text('[project]\nversion = "1.2.3"\n')
      for remote_status, tag_sha, exit_code, publish, creates in cases:
        with self.subTest(remote_status=remote_status, tag_sha=tag_sha):
          output = root / 'output'
          calls = root / 'calls'
          output.write_text('')
          calls.write_text('')
          env = dict(
            os.environ,
            PATH=f'{binaries}{os.pathsep}/usr/bin:/bin',
            REMOTE_STATUS=str(remote_status),
            TAG_SHA=tag_sha,
            RELEASE_SHA=self.sha,
            GITHUB_OUTPUT=str(output),
            GH_CALLS=str(calls),
            PYTHONPATH=str(root),
          )
          result = subprocess.run(
            [bash, '--noprofile', '--norc', '-eo', 'pipefail', '-c', script],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
          )
          self.assertEqual(result.returncode, exit_code, result.stderr)
          self.assertEqual('publish=' in output.read_text(), publish is not None)
          if publish is not None:
            self.assertIn(f'publish={publish}\n', output.read_text())
          if creates:
            self.assertIn(f'release create v1.2.3 --target {self.sha}', calls.read_text())
          else:
            self.assertEqual(calls.read_text(), '')
