#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Generate a CPU flame graph for quent using py-spy.
#
# Usage:
#   ./benchmarks/flamegraph_pyspy.sh [output.svg]
#
# Requirements:
#   pip install py-spy
#   # macOS: may need sudo (SIP restriction)
#   # Linux: may need --sudo or appropriate capabilities
#
# Examples:
#   ./benchmarks/flamegraph_pyspy.sh
#   ./benchmarks/flamegraph_pyspy.sh benchmarks/results/my_flamegraph.svg

set -euo pipefail

OUTPUT="${1:-benchmarks/results/flamegraph.svg}"
mkdir -p "$(dirname "$OUTPUT")"

WORKLOAD="benchmarks/profile_scalene.py"

echo "Generating flame graph -> $OUTPUT"
echo "NOTE: On macOS, py-spy may require sudo (SIP restriction)."

if command -v sudo &>/dev/null && [[ "$(uname)" == "Darwin" ]]; then
  sudo py-spy record -o "$OUTPUT" --format flamegraph -- python "$WORKLOAD"
else
  py-spy record -o "$OUTPUT" --format flamegraph -- python "$WORKLOAD"
fi

echo "Flame graph saved to $OUTPUT"
echo "Open in a browser: open $OUTPUT"
