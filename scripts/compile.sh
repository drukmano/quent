#!/usr/bin/env bash
set -e

# Pure Python equivalent of Cython's compile step:
# byte-compiles all .py files, catching syntax errors early
# and pre-generating .pyc files for faster import.

echo "==> Cleaning stale bytecode"
find quent/ -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find quent/ -name '*.pyc' -delete 2>/dev/null || true

echo "==> Byte-compiling quent/"
python3 -m compileall -q quent/

echo "==> Compile check passed."
