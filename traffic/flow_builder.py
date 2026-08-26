"""
flow_builder.py

Step B of the pipeline: packets -> flows

Groups packets from capture.py into bidirectional flows using the
5-tuple (src_ip, dst_ip, src_port, dst_port, proto), treating A->B and
B->A packets as the same flow. A new flow starts if there's a gap of
more than FLOW_TIMEOUT seconds since the last packet in that 5-tuple
(handles reused ports across separate sessions).

Usage (as a module):
    from capture import parse_pcap
    from flow_builder import build_flows

    packets = parse_pcap("chat_skype.pcap")
    flows = build_flows(packets)
"""

FLOW_TIMEOUT = 120.0  # seconds of inactivity before starting a new flow


def _flow_key(pkt):
    """
    Canonical, direction-independent key for a packet's 5-tuple, so that
    A->B and B->A packets land in the same flow. Also returns whether
    this packet is in the "forward" direction relative to that key,
    which flow_builder needs to compute per-direction stats later.
    """
    a = (pkt["src_ip"], pkt["src_port"])
    b = (pkt["dst_ip"], pkt["dst_port"])
    if a <= b:
        key = (a[0], a[1], b[0], b[1], pkt["proto"])
        forward = True
    else:
        key = (b[0], b[1], a[0], a[1], pkt["proto"])
        forward = False
    return key, forward


def build_flows(packets):
    """
    Group packets into flows.

    Returns a list of flow dicts, each with:
        src_ip, src_port, dst_ip, dst_port, proto   (from the first packet)
        packets       list of packet dicts belonging to this flow, in order
        source_file   pcap filename the flow came from (for labeling)
    """
    # active_flows[key] -> flow dict currently being built
    active_flows = {}
    finished_flows = []

    # Process in timestamp order so the timeout logic is meaningful
    for pkt in sorted(packets, key=lambda p: p["timestamp"]):
        key, forward = _flow_key(pkt)

        flow = active_flows.get(key)
        if flow is not None:
            last_ts = flow["packets"][-1]["timestamp"]
            if pkt["timestamp"] - last_ts > FLOW_TIMEOUT:
                # Too much time has passed -> close old flow, start a new one
                finished_flows.append(flow)
                flow = None

        if flow is None:
            flow = {
                "src_ip": key[0],
                "src_port": key[1],
                "dst_ip": key[2],
                "dst_port": key[3],
                "proto": key[4],
                "packets": [],
                "source_file": pkt["source_file"],
            }
            active_flows[key] = flow

        flow["packets"].append(pkt)

    # Close out any flows still open at the end
    finished_flows.extend(active_flows.values())

    return finished_flows


if __name__ == "__main__":
    import sys
    from capture import parse_pcap, parse_folder
    import os

    if len(sys.argv) != 2:
        print("Usage: python flow_builder.py <pcap_file_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    pkts = parse_folder(target) if os.path.isdir(target) else parse_pcap(target)
    flows = build_flows(pkts)
    print(f"Built {len(flows)} flows from {len(pkts)} packets")
