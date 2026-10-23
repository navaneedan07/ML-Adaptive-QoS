#!/usr/bin/env bash
set -euo pipefail

# HTB plus fq_codel prevents the bufferbloat observed with HTB alone.
# Classes are real priorities: 1:10 (latency-sensitive: voip/chat) is served
# first, then 1:20 (streaming/email), then 1:30 (bulk/default). Each class is
# guaranteed a share of the 10 Mbit bottleneck and may borrow up to the ceiling.
DEVICE="${DEVICE:-veth1}"
NAMESPACE="${NAMESPACE:-ns1}"

sudo ip netns exec "$NAMESPACE" tc qdisc replace dev "$DEVICE" root handle 1: htb default 30
sudo ip netns exec "$NAMESPACE" tc class replace dev "$DEVICE" parent 1: classid 1:1 htb rate 10mbit ceil 10mbit

sudo ip netns exec "$NAMESPACE" tc class replace dev "$DEVICE" parent 1:1 classid 1:10 htb rate 3mbit ceil 10mbit prio 0
sudo ip netns exec "$NAMESPACE" tc class replace dev "$DEVICE" parent 1:1 classid 1:20 htb rate 4mbit ceil 10mbit prio 1
sudo ip netns exec "$NAMESPACE" tc class replace dev "$DEVICE" parent 1:1 classid 1:30 htb rate 3mbit ceil 10mbit prio 2
for class in 10 20 30; do
  sudo ip netns exec "$NAMESPACE" tc qdisc replace dev "$DEVICE" parent "1:$class" handle "$class:1" fq_codel
done

sudo ip netns exec "$NAMESPACE" tc filter replace dev "$DEVICE" protocol ip parent 1: prio 1 flower ip_proto udp dst_port 5001 flowid 1:10
sudo ip netns exec "$NAMESPACE" tc filter replace dev "$DEVICE" protocol ip parent 1: prio 2 flower ip_proto tcp dst_port 5002 flowid 1:20
# Ping is a latency probe: treat it like the latency-sensitive class.
sudo ip netns exec "$NAMESPACE" tc filter replace dev "$DEVICE" protocol ip parent 1: prio 3 flower ip_proto icmp flowid 1:10
