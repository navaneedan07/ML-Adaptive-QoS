#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-$SCRIPT_DIR/results/qos_udp.json}"
python3 "$SCRIPT_DIR/measure_qos.py" --iperf-output "$OUTPUT"