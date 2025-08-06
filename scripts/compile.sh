#!/usr/bin/env bash
set -e

# Always clean test artifacts
rm -f .coverage
rm -rf htmlcov

clean_build_artifacts() {
  rm -f quent/*.cpython*
  rm -f quent/*.c
  rm -f quent/*.html
  rm -f quent/*/*.cpython*
  rm -f quent/*/*.c
  rm -f quent/*/*.html
}

# Check if any .pyx/.pxi source is newer than the newest .so file.
# Returns 0 (true) if recompilation is needed, 1 (false) if up to date.
sources_changed() {
  # If no .so files exist, recompilation is needed
  local newest_so
  newest_so=$(find quent -name '*.so' -print -quit 2>/dev/null)
  if [ -z "$newest_so" ]; then
    return 0
  fi

  # Find the newest .so file
  local so_files
  so_files=$(find quent -name '*.so')
  local newest_so_file
  newest_so_file=$(ls -t $so_files 2>/dev/null | head -1)

  # Check if any .pyx or .pxi file is newer than the newest .so
  for src in quent/*.pyx quent/*.pxi quent/*/*.pyx quent/*/*.pxi; do
    [ -f "$src" ] || continue
    if [ "$src" -nt "$newest_so_file" ]; then
      return 0
    fi
  done

  return 1
}

# Parse arguments
force=false
mode=""
extra_args=""
for arg in "$@"; do
  case "$arg" in
    force|--force)
      force=true
      ;;
    pgo)
      mode="pgo"
      ;;
    *)
      extra_args="$extra_args $arg"
      ;;
  esac
done

if [ "$mode" = "pgo" ]; then
  # PGO always does a full recompilation
  clean_build_artifacts

  # Pass 1: compile with profile generation
  QUENT_PGO=generate python cython_setup.py build_ext --inplace

  # Run test suite to generate profile data
  python -m unittest discover -s tests -p '*_tests.py'

  # Clean .so files but keep .gcda profile data
  rm -f quent/*.cpython*
  rm -f quent/*/*.cpython*

  # Pass 2: recompile using profile data
  QUENT_PGO=use python cython_setup.py build_ext --inplace

  # Clean profile data files
  find . -name '*.gcda' -delete
  find . -name '*.gcno' -delete
else
  if [ "$force" = true ] || sources_changed; then
    [ "$force" = true ] && echo "Forcing full recompilation..."
    clean_build_artifacts
    python cython_setup.py $extra_args build_ext --inplace
  else
    echo "All .so files are up to date — skipping compilation."
  fi
fi
