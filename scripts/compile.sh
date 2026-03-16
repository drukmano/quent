#!/usr/bin/env bash
set -euo pipefail

# Byte-compiles all .py files, catching syntax errors early.

echo "==> Cleaning stale bytecode"
find quent/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find quent/ -name '*.pyc' -delete 2>/dev/null || true

echo "==> Byte-compiling quent/"
python3 -m compileall -q quent/

echo "==> Compile check passed."
