#!/usr/bin/env bash
set -e
rm -rf dist
rm -rf build
rm -rf quent.egg-info
python3 -m build
