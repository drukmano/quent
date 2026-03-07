#!/usr/bin/env bash
set -e

# Build and upload to PyPI.

./scripts/build.sh

if [ ! -d "dist" ]; then
  echo "Error: 'dist' directory does not exist. Build may have failed."
  exit 1
fi

echo "==> Checking distribution"
python3 -m twine check dist/*

echo "==> Uploading to PyPI"
python3 -m twine upload dist/*

echo "==> Distribution complete."
