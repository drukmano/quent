#!/usr/bin/env bash
set -euo pipefail

echo "==> Format (ruff format)"
ruff format quent/ tests/

echo "==> Lint fix (ruff check --fix)"
ruff check --fix quent/ tests/

echo "==> Done."
