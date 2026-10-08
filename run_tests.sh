#!/usr/bin/env bash
set -euo pipefail

# Resolve the project environment even when launched outside the checkout.
cd "$(dirname "$0")"
VENV_DIR="${VIRTUAL_ENV:-.venv}"
export PATH="$VENV_DIR/bin:$PATH"

# Optimal order: cheapest/fastest checks first, fail fast before expensive operations.

echo "==> Format check (ruff format)"
ruff format --check quent/ tests/

echo "==> Lint check (ruff check)"
ruff check quent/ tests/

echo "==> Type check (mypy)"
mypy quent/

echo "==> Tests"
python scripts/run_tests_parallel.py "$@"

echo "==> All checks passed."
