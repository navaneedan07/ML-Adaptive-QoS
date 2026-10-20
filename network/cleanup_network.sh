#!/usr/bin/env bash
set -euo pipefail

sudo ip netns del ns1 2>/dev/null || true
sudo ip netns del ns2 2>/dev/null || true
echo "Network namespaces removed."