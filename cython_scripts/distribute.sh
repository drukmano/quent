#!/usr/bin/env bash
set -e
./scripts/build.sh
if [ ! -d "dist" ]; then
  echo "Error: build directory 'dist' does not exist. Build may have failed."
  exit 1
fi
rm dist/quent-*.whl
python3 -m twine upload dist/quent-*.tar.gz
