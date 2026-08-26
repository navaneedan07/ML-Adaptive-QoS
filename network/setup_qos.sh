#!/bin/bash

echo "Setting up low-latency QoS..."

# Remove old configuration
sudo ip netns exec ns1 tc qdisc del dev veth1 root 2>/dev/null || true

# Root HTB
sudo ip netns exec ns1 tc qdisc add dev veth1 root handle 1: htb default 20

# Total bandwidth
sudo ip netns exec ns1 tc class add dev veth1 \
    parent 1: classid 1:1 \
    htb rate 10mbit ceil 10mbit

# High-priority traffic
sudo ip netns exec ns1 tc class add dev veth1 \
    parent 1:1 classid 1:10 \
    htb rate 7mbit ceil 10mbit prio 0

# Low-priority traffic
sudo ip netns exec ns1 tc class add dev veth1 \
    parent 1:1 classid 1:20 \
    htb rate 3mbit ceil 10mbit prio 1

# Low-latency queues
sudo ip netns exec ns1 tc qdisc add dev veth1 \
    parent 1:10 handle 10: fq_codel

sudo ip netns exec ns1 tc qdisc add dev veth1 \
    parent 1:20 handle 20: fq_codel

# UDP → high priority
sudo ip netns exec ns1 tc filter add dev veth1 \
    protocol ip parent 1: \
    prio 1 flower ip_proto udp \
    flowid 1:10

# TCP → low priority
sudo ip netns exec ns1 tc filter add dev veth1 \
    protocol ip parent 1: \
    prio 2 flower ip_proto tcp \
    flowid 1:20

echo "Low-latency QoS configured."
