#!/usr/bin/env bash
set -euo pipefail

# HTB plus fq_codel prevents the bufferbloat observed with HTB alone.
DEVICE="${DEVICE:-veth1}"
NAMESPACE="${NAMESPACE:-ns1}"

sudo ip netns exec "$NAMESPACE" tc qdisc replace dev "$DEVICE" root handle 1: htb default 30
sudo ip netns exec "$NAMESPACE" tc class replace dev "$DEVICE" parent 1: classid 1:1 htb rate 10mbit ceil 10mbit

for class in 10 20 30; do
  sudo ip netns exec "$NAMESPACE" tc class replace dev "$DEVICE" parent 1:1 classid "1:$class" htb rate 1mbit ceil 10mbit
  sudo ip netns exec "$NAMESPACE" tc qdisc replace dev "$DEVICE" parent "1:$class" handle "$class:1" fq_codel
done

sudo ip netns exec "$NAMESPACE" tc filter replace dev "$DEVICE" protocol ip parent 1: prio 1 flower ip_proto udp dst_port 5001 flowid 1:10
sudo ip netns exec "$NAMESPACE" tc filter replace dev "$DEVICE" protocol ip parent 1: prio 2 flower ip_proto tcp dst_port 5002 flowid 1:20
