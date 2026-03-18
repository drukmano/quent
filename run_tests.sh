#!/usr/bin/env bash
set -euo pipefail

# Activate venv if not already active.
VENV_DIR="${VIRTUAL_ENV:-.venv@3.tmp}"
export PATH="$VENV_DIR/bin:$PATH"

# Optimal order: cheapest/fastest checks first, fail fast before expensive operations.

echo "==> Format check (ruff format)"
ruff format --check quent/ tests/

echo "==> Lint check (ruff check)"
ruff check quent/ tests/

echo "==> Type check (mypy)"
mypy quent/

echo "==> Tests (with coverage)"
python scripts/run_tests_parallel.py --coverage

echo "==> All checks passed."
