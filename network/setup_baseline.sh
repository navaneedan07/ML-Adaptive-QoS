#!/usr/bin/env bash
set -euo pipefail

# Fair "best effort" baseline: the SAME 10 Mbit bottleneck as setup_qos.sh, but one
# plain FIFO queue and no classes. Without a bottleneck nothing ever queues, so
# comparing QoS against an unshaped link would say nothing about prioritisation.
DEVICE="${DEVICE:-veth1}"
NAMESPACE="${NAMESPACE:-ns1}"

sudo ip netns exec "$NAMESPACE" tc qdisc replace dev "$DEVICE" root tbf rate 10mbit burst 32kbit latency 400ms
