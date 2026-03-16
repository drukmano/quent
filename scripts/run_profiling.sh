#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

RESULTS_DIR="benchmarks/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Defaults
RUN_ANALYSIS=true
RUN_PROFILERS=true
RUN_BENCHMARKS=true
RUN_FLAMEGRAPH=true
CPROFILE_ITERATIONS=5000

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run all profiling and benchmarking tools for quent.

Options:
  --quick          Fast profilers only (skip pyperf benchmarks and flame graph)
  --bench-only     Run only pyperf benchmarks
  --profile-only   Run only profilers (cProfile, line_profiler)
  --help           Show this help message
EOF
  exit 0
}

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --quick)
      RUN_BENCHMARKS=false
      RUN_FLAMEGRAPH=false
      CPROFILE_ITERATIONS=1000
      ;;
    --bench-only)
      RUN_ANALYSIS=false
      RUN_PROFILERS=false
      RUN_FLAMEGRAPH=false
      ;;
    --profile-only)
      RUN_ANALYSIS=false
      RUN_BENCHMARKS=false
      RUN_FLAMEGRAPH=false
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      ;;
  esac
done

# Summary tracking
declare -a RAN=()
declare -a SKIPPED=()

mkdir -p "$RESULTS_DIR"

PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")

# ---- (a) Static analysis ----

if $RUN_ANALYSIS; then
  echo ""
  echo "==> Bytecode analysis"
  python benchmarks/analyze_bytecode.py
  RAN+=("bytecode analysis")

  echo ""
  echo "==> Specialization analysis"
  if [[ "$PY_MINOR" -ge 11 ]]; then
    python benchmarks/analyze_specialization.py
    RAN+=("specialization analysis")
  else
    echo "    Skipped: requires Python 3.11+ (running 3.${PY_MINOR})"
    SKIPPED+=("specialization analysis (Python 3.11+ required)")
  fi
fi

# ---- (b) Call monitoring ----

if $RUN_ANALYSIS; then
  echo ""
  echo "==> Call monitoring"
  if [[ "$PY_MINOR" -ge 12 ]]; then
    python benchmarks/monitor_calls.py
    RAN+=("call monitoring")
  else
    echo "    Skipped: requires Python 3.12+ (running 3.${PY_MINOR})"
    SKIPPED+=("call monitoring (Python 3.12+ required)")
  fi
fi

# ---- (c) cProfile ----

if $RUN_PROFILERS; then
  echo ""
  echo "==> cProfile ($CPROFILE_ITERATIONS iterations)"
  python benchmarks/profile_cprofile.py "$CPROFILE_ITERATIONS"
  RAN+=("cProfile (${CPROFILE_ITERATIONS} iterations)")
fi

# ---- (d) Line profiler ----

if $RUN_PROFILERS; then
  echo ""
  echo "==> Line profiler"
  set +e
  python -c "import line_profiler" 2>/dev/null
  LP_AVAILABLE=$?
  set -e
  if [[ $LP_AVAILABLE -eq 0 ]]; then
    python benchmarks/profile_line.py
    RAN+=("line profiler")
  else
    echo "    Skipped: line-profiler not installed"
    echo "    Install with: pip install line-profiler"
    SKIPPED+=("line profiler (pip install line-profiler)")
  fi
fi

# ---- (e) pyperf benchmarks ----

if $RUN_BENCHMARKS; then
  echo ""
  echo "==> pyperf: core benchmarks"
  python benchmarks/bench_core.py -o "$RESULTS_DIR/bench_core_${TIMESTAMP}.json"
  RAN+=("pyperf bench_core")

  echo ""
  echo "==> pyperf: ops benchmarks"
  python benchmarks/bench_ops.py -o "$RESULTS_DIR/bench_ops_${TIMESTAMP}.json"
  RAN+=("pyperf bench_ops")

  echo ""
  echo "==> pyperf: chain size benchmarks"
  python benchmarks/bench_chain_sizes.py -o "$RESULTS_DIR/bench_chain_sizes_${TIMESTAMP}.json"
  RAN+=("pyperf bench_chain_sizes")

  echo ""
  echo "==> pyperf: async benchmarks"
  python benchmarks/bench_async.py -o "$RESULTS_DIR/bench_async_${TIMESTAMP}.json"
  RAN+=("pyperf bench_async")
fi

# ---- (f) Flame graph ----

if $RUN_FLAMEGRAPH; then
  echo ""
  echo "==> Flame graph (py-spy)"
  if command -v py-spy &>/dev/null; then
    bash benchmarks/flamegraph_pyspy.sh "$RESULTS_DIR/flamegraph_${TIMESTAMP}.svg"
    RAN+=("flame graph")
  else
    echo "    Skipped: py-spy not installed"
    echo "    Install with: pip install py-spy"
    SKIPPED+=("flame graph (pip install py-spy)")
  fi
fi

# ---- Summary ----

echo ""
echo "=========================================="
echo "  Profiling summary"
echo "=========================================="

if [[ ${#RAN[@]} -gt 0 ]]; then
  echo ""
  echo "  Ran:"
  for item in "${RAN[@]}"; do
    echo "    - $item"
  done
fi

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo ""
  echo "  Skipped:"
  for item in "${SKIPPED[@]}"; do
    echo "    - $item"
  done
fi

if $RUN_BENCHMARKS; then
  echo ""
  echo "  pyperf results saved to $RESULTS_DIR/"
fi

echo ""
echo "==> Done."
