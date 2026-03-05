#!/usr/bin/env bash
set -e

# Build and upload to PyPI.
# Mirrors scripts_cython/distribute.sh but for pure Python distribution.

./scripts/build.sh

if [ ! -d "dist" ]; then
  echo "Error: 'dist' directory does not exist. Build may have failed."
  exit 1
fi

echo "==> Uploading to PyPI"
python3 -m twine upload dist/*

echo "==> Distribution complete."
