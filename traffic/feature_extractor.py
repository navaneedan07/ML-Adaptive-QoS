"""
feature_extractor.py

Step C of the pipeline: flows -> features -> features.csv

Computes flow-level features (per the plan: flow duration, packet count,
total bytes, mean/std packet size, mean/std inter-arrival time, burst
statistics) for each flow from flow_builder.py, labels each flow based
on its source pcap filename, and writes everything to features.csv.

Usage:
    python feature_extractor.py <pcap_file_or_folder> <output_csv>

Example:
    python feature_extractor.py ../dataset/ ../dataset/features.csv
"""

import os
import sys
import csv
import statistics

# Keep these in sync with flow_builder.build_flows() output
BURST_GAP_THRESHOLD = 1.0  # seconds; gap larger than this starts a new "burst"

# ISCX VPN-nonVPN files are named by application/activity, e.g.
# "skype_chat1a.pcap", "facebook_video2b.pcap", "vpn_email2a.pcap".
# Adjust/extend these keyword -> class mappings once you've looked at
# your actual downloaded filenames.
LABEL_KEYWORDS = {
    "chat": ["chat", "aim", "icq", "skype_chat", "hangout_chat", "facebook_chat"],
    "email": ["email", "gmail"],
    "streaming": ["youtube", "netflix", "vimeo", "spotify", "video"],
    "voip": ["voip", "voice", "hangouts_audio", "skype_audio", "facebook_audio"],
    "p2p": ["p2p", "torrent", "bittorrent"],
    "file_transfer": ["ftps", "sftp", "filetransfer", "file_transfer"],
    "browsing": ["browsing", "firefox", "chrome"],
}


def label_from_filename(filename):
    """
    Best-effort label from an ISCX-style pcap filename.
    Returns (traffic_class, is_vpn).
    Falls back to "unknown" if no keyword matches -- check these rows
    manually and refine LABEL_KEYWORDS above.
    """
    lower = filename.lower()
    is_vpn = lower.startswith("vpn") or "vpn_" in lower

    for label, keywords in LABEL_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return label, is_vpn

    return "unknown", is_vpn


def _burst_count(timestamps):
    """
    Count "bursts" in a flow: consecutive packets closer together than
    BURST_GAP_THRESHOLD form one burst; a gap larger than the threshold
    starts a new burst. Returns (num_bursts, mean_burst_size).
    """
    if len(timestamps) <= 1:
        return 1, float(len(timestamps))

    bursts = [1]
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > BURST_GAP_THRESHOLD:
            bursts.append(1)
        else:
            bursts[-1] += 1

    return len(bursts), statistics.mean(bursts)


def extract_features(flow):
    """
    Compute the feature dict for a single flow (as produced by
    flow_builder.build_flows).
    """
    pkts = sorted(flow["packets"], key=lambda p: p["timestamp"])
    sizes = [p["size"] for p in pkts]
    timestamps = [p["timestamp"] for p in pkts]

    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    iats = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    num_bursts, mean_burst_size = _burst_count(timestamps)

    label, is_vpn = label_from_filename(flow["source_file"])

    return {
        "src_ip": flow["src_ip"],
        "src_port": flow["src_port"],
        "dst_ip": flow["dst_ip"],
        "dst_port": flow["dst_port"],
        "proto": flow["proto"],
        "flow_duration": round(duration, 6),
        "packet_count": len(pkts),
        "total_bytes": sum(sizes),
        "mean_packet_size": round(statistics.mean(sizes), 4),
        "std_packet_size": round(statistics.pstdev(sizes), 4) if len(sizes) > 1 else 0.0,
        "mean_iat": round(statistics.mean(iats), 6) if iats else 0.0,
        "std_iat": round(statistics.pstdev(iats), 6) if len(iats) > 1 else 0.0,
        "num_bursts": num_bursts,
        "mean_burst_size": round(mean_burst_size, 4),
        "is_vpn": int(is_vpn),
        "source_file": flow["source_file"],
        "label": label,
    }


FIELDNAMES = [
    "src_ip", "src_port", "dst_ip", "dst_port", "proto",
    "flow_duration", "packet_count", "total_bytes",
    "mean_packet_size", "std_packet_size",
    "mean_iat", "std_iat",
    "num_bursts", "mean_burst_size",
    "is_vpn", "source_file", "label",
]


def write_features_csv(flows, output_path):
    rows = [extract_features(f) for f in flows]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python feature_extractor.py <pcap_file_or_folder> <output_csv>")
        sys.exit(1)

    pcap_target, output_csv = sys.argv[1], sys.argv[2]

    from capture import parse_pcap, parse_folder
    # Support both `python traffic/feature_extractor.py ...` from the
    # repository root and `python feature_extractor.py ...` from this folder.
    try:
        from flow_builder import build_flows
    except ImportError:
        from traffic.flow_builder import build_flows

    pkts = parse_folder(pcap_target) if os.path.isdir(pcap_target) else parse_pcap(pcap_target)
    flows = build_flows(pkts)
    rows = write_features_csv(flows, output_csv)

    print(f"Wrote {len(rows)} flows to {output_csv}")
    unknown = sum(1 for r in rows if r["label"] == "unknown")
    if unknown:
        print(f"WARNING: {unknown} flows labeled 'unknown' -- check LABEL_KEYWORDS "
              f"against your actual pcap filenames.")
