#!/usr/bin/env bash
set -e

echo "==> Cleaning previous build artifacts"
rm -rf dist build *.egg-info

echo "==> Building sdist + wheel"
python3 -m build

echo "==> Build complete. Artifacts in dist/"
ls -lh dist/
