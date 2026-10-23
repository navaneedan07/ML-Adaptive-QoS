#!/usr/bin/env bash
set -euo pipefail

# Fair QoS benchmark on Linux/WSL. Same 10 Mbit bottleneck, same load; only the
# queueing differs:
#   baseline = one FIFO queue          qos = priority classes + fq_codel
# Load: 4 bulk TCP streams (port 5003, default/bulk class) running while a
# voice-like probe (1 Mbit UDP, 200-byte packets, port 5001) and ping are measured.
#
# Usage: ./run_benchmark.sh [output_dir]      (DURATION=30 by default)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$SCRIPT_DIR/results/benchmark}"
DURATION="${DURATION:-30}"
NS="${NAMESPACE:-ns1}"
SERVER_NS="${SERVER_NS:-ns2}"
HOST="10.0.0.2"
mkdir -p "$OUT"

cleanup() {
  sudo ip netns exec "$SERVER_NS" pkill iperf3 2>/dev/null || true
  "$SCRIPT_DIR/remove_qos.sh" || true
}
trap cleanup EXIT

"$SCRIPT_DIR/setup_network.sh"
for port in 5001 5002 5003; do
  sudo ip netns exec "$SERVER_NS" iperf3 -s -p "$port" -D
done
sleep 1

run_scenario() {
  local name="$1" setup="$2"
  echo "== scenario: $name"
  "$setup"
  sudo ip netns exec "$NS" iperf3 -c "$HOST" -p 5003 -t $((DURATION + 3)) -P 4 > "$OUT/${name}_bulk.txt" &
  local bulk=$!
  sleep 2   # let the bulk transfers fill the queue first
  sudo ip netns exec "$NS" ping -c "$DURATION" -i 1 "$HOST" > "$OUT/${name}_ping.txt" &
  local pinger=$!
  sudo ip netns exec "$NS" iperf3 -c "$HOST" -p 5001 -u -b 1M -l 200 -t "$DURATION" -J > "$OUT/${name}_voice.json"
  wait "$pinger" || true
  wait "$bulk" || true
  "$SCRIPT_DIR/remove_qos.sh"
}

run_scenario baseline "$SCRIPT_DIR/setup_baseline.sh"
run_scenario qos "$SCRIPT_DIR/setup_qos.sh"

python3 "$SCRIPT_DIR/analyze_benchmark.py" "$OUT" "$OUT/summary.csv"
