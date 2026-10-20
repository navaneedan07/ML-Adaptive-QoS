#!/usr/bin/env bash
set -euo pipefail

sudo ip netns exec "${NAMESPACE:-ns1}" tc qdisc del dev "${DEVICE:-veth1}" root 2>/dev/null || true
