"""

# NOTE: file follows project structure naming ("capture.py"), but this
# module actually reads existing PCAP files (offline), not a live capture.
# For real live traffic capture, add a separate live-capture function here
# using tcpdump/pyshark sniff() if the team needs that later.
capture.py

Step A of the pipeline: PCAP -> packets

Reads one or more .pcap files and extracts a flat list of packet-level
records (timestamp, size, protocol, src/dst ip:port). This is the raw
material that flow_builder.py groups into flows.

Usage (as a script):
    python capture.py path/to/file.pcap
    python capture.py path/to/pcap_folder/

Usage (as a module):
    from capture import parse_pcap
    packets = parse_pcap("chat_skype.pcap")
"""

import os
import sys
from scapy.all import PcapReader, IP, TCP, UDP


def parse_pcap(pcap_path):
    """
    Parse a single .pcap file into a list of packet dicts.

    Each dict has:
        timestamp   float, seconds (epoch time from the capture)
        size        int, total packet length in bytes (wire length)
        proto       "TCP" / "UDP" / "OTHER"
        src_ip, dst_ip     str
        src_port, dst_port int (0 if not TCP/UDP)
        source_file str, the pcap filename (used later for labeling)
    """
    packets = []
    fname = os.path.basename(pcap_path)

    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            if IP not in pkt:
                # Skip non-IP packets (ARP, etc.) - not useful for flow features
                continue

            ip_layer = pkt[IP]
            proto = "OTHER"
            src_port = 0
            dst_port = 0

            if TCP in pkt:
                proto = "TCP"
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
            elif UDP in pkt:
                proto = "UDP"
                src_port = int(pkt[UDP].sport)
                dst_port = int(pkt[UDP].dport)

            packets.append({
                "timestamp": float(pkt.time),
                "size": len(pkt),
                "proto": proto,
                "src_ip": ip_layer.src,
                "dst_ip": ip_layer.dst,
                "src_port": src_port,
                "dst_port": dst_port,
                "source_file": fname,
            })

    return packets


def parse_folder(folder_path):
    """
    Parse every .pcap file in a folder.
    Returns a single flat list of packet dicts across all files.
    """
    all_packets = []
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(folder_path)
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".pcap", ".pcapng")):
            full_path = os.path.join(folder_path, fname)
            print(f"Parsing {fname} ...")
            pkts = parse_pcap(full_path)
            print(f"  -> {len(pkts)} IP packets")
            all_packets.extend(pkts)
    return all_packets


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python capture.py <pcap_file_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        pkts = parse_folder(target)
    else:
        pkts = parse_pcap(target)

    print(f"Total packets parsed: {len(pkts)}")
