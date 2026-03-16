#!/usr/bin/env bash
set -euo pipefail

# Build and upload to PyPI.
# NOTE: The CI publish workflow (.github/workflows/release.yml) is the
# preferred release path. Use this script only for manual releases.

# Quality gate: run the full test suite before building.
./scripts/run_tests.sh

./scripts/build.sh

if [ ! -d "dist" ]; then
  echo "Error: 'dist' directory does not exist. Build may have failed."
  exit 1
fi

echo "==> Checking distribution"
uv run --group publish python -m twine check dist/*

echo ""
echo "WARNING: You are about to upload directly to PyPI, bypassing CI and trusted publisher (OIDC) flow."
echo "The preferred release path is the CI publish workflow (.github/workflows/release.yml)."
read -p "Are you sure you want to continue? [y/N] " confirm
if [[ "$confirm" != [yY] ]]; then
  echo "Aborted."
  exit 1
fi

echo "==> Uploading to PyPI"
# Enforce API token authentication (not username/password).
# To store your token securely, use: keyring set https://upload.pypi.org/legacy/ __token__
uv run --group publish python -m twine upload --username __token__ dist/*

echo "==> Distribution complete."
