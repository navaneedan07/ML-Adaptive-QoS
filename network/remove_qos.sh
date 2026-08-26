#!/bin/bash
#!/bin/bash

echo "Removing QoS configuration..."

sudo ip netns exec ns1 tc qdisc del dev veth1 root 2>/dev/null || true

echo "QoS removed."
