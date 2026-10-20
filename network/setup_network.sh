#!/usr/bin/env bash
set -euo pipefail

# Create an isolated two-node network namespace topology:
# ns1 (client) <-> veth pair <-> ns2 (iperf server).
sudo ip netns add ns1 2>/dev/null || true
sudo ip netns add ns2 2>/dev/null || true
sudo ip link add veth1 type veth peer name veth2 2>/dev/null || true
sudo ip link set veth1 netns ns1 2>/dev/null || true
sudo ip link set veth2 netns ns2 2>/dev/null || true
sudo ip -n ns1 addr add 10.0.0.1/24 dev veth1 2>/dev/null || true
sudo ip -n ns2 addr add 10.0.0.2/24 dev veth2 2>/dev/null || true
sudo ip -n ns1 link set lo up
sudo ip -n ns2 link set lo up
sudo ip -n ns1 link set veth1 up
sudo ip -n ns2 link set veth2 up
echo "Network namespaces ns1 (10.0.0.1) and ns2 (10.0.0.2) are ready."