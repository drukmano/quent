#!/usr/bin/env bash
set -euo pipefail

echo "==> Cleaning previous build artifacts"
rm -rf dist build quent.egg-info

echo "==> Building sdist + wheel"
uv build

echo "==> Build complete. Artifacts in dist/"
ls -lh dist/
