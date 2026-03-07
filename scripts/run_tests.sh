#!/usr/bin/env bash
set -e

# Optimal order: cheapest/fastest checks first, fail fast before expensive operations.

echo "==> Formatting (ruff format)"
ruff format --check quent/

echo "==> Lint fix (ruff check)"
ruff check quent/

echo "==> Type check (mypy)"
mypy quent/

echo "==> Tests + coverage"
coverage run -m unittest discover -s tests -p '*_tests.py'
coverage report -m
coverage html

echo "==> All checks passed."
