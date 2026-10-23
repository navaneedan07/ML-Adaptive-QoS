"""Synthetic traffic for testing the pipeline and for demos. NOT real data.

Each class is simulated from the rough shape of that kind of traffic (steady
small UDP packets for voice, sporadic small TCP messages for chat, large
chunked TCP downloads for streaming, and so on). Every "recording" draws its
own parameters, so cross-recording splits are still meaningful, but the classes
are separable by construction.

Use it to prove the code works end to end and to demonstrate drift and
retraining. Never quote accuracy measured on it as a real-world result.
Files are named ``synthetic_<class>_<n>.pcap`` so they can never be mistaken
for the real dataset.

Usage:
    python -m traffic.synthetic data/synthetic/features.csv
    python -m traffic.synthetic out.csv --pcap-dir data/synthetic/pcaps   # needs scapy
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Callable

from traffic.feature_extractor import FIELDNAMES, extract_features

CLASSES = ("voip", "chat", "streaming", "email", "p2p", "file_transfer")
FILE_STEM = {"streaming": "video"}  # the labeller recognises "video" as streaming


def recording_name(label: str, index: int) -> str:
    return f"synthetic_{FILE_STEM.get(label, label)}_{index}.pcap"


def _packet(flow: dict, t: float, size: int, forward: bool) -> dict:
    a, b = ("src", "dst") if forward else ("dst", "src")
    return {
        "timestamp": t,
        "size": max(54, int(size)),
        "proto": flow["proto"],
        "src_ip": flow[f"{a}_ip"], "dst_ip": flow[f"{b}_ip"],
        "src_port": flow[f"{a}_port"], "dst_port": flow[f"{b}_port"],
        "source_file": flow["source_file"],
    }


def _voip(rng, rec, flow, n):
    gap = rec["gap"] * rng.uniform(0.9, 1.1)  # ~20 ms codec frames
    t = 0.0
    for i in range(n):
        yield _packet(flow, t, rng.gauss(rec["size"], 12), i % 2 == 0)
        t += max(0.002, rng.gauss(gap, gap * 0.15))


def _chat(rng, rec, flow, n):
    t, i = 0.0, 0
    while i < n:
        for _ in range(rng.randint(1, 3)):  # a short message exchange
            yield _packet(flow, t, rng.uniform(60, rec["size"]), True)
            t += rng.uniform(0.01, 0.15)
            yield _packet(flow, t, rng.uniform(54, 90), False)
            t += rng.uniform(0.01, 0.1)
            i += 2
        t += rng.expovariate(1 / rec["gap"])  # idle until the next message


def _streaming(rng, rec, flow, n):
    t, i = 0.0, 0
    while i < n:
        for _ in range(rng.randint(10, 40)):  # a media chunk
            yield _packet(flow, t, rng.gauss(rec["size"], 60), False)
            t += rng.uniform(0.0005, 0.004)
            i += 1
        t += rng.uniform(1.0, 1.0 + rec["gap"])  # player buffer is full


def _email(rng, rec, flow, n):
    t = 0.0
    for i in range(n):
        phase = i < 8  # protocol handshake: small packets
        size = rng.uniform(60, 200) if phase else rng.choice([66, 66, rec["size"], rng.uniform(300, 1460)])
        yield _packet(flow, t, size, i % 3 != 0)
        t += rng.expovariate(1 / (rec["gap"] * (4 if phase else 1)))


def _p2p(rng, rec, flow, n):
    t = 0.0
    for i in range(n):
        size = rng.choice([68, 100, 400, 800, rec["size"], 1400])
        yield _packet(flow, t, size, rng.random() < 0.5)
        t += rng.expovariate(1 / rec["gap"])


def _file_transfer(rng, rec, flow, n):
    t = 0.0
    for i in range(n):
        data = i % 3 != 2
        yield _packet(flow, t, rng.gauss(rec["size"], 30) if data else 66, data)
        t += rng.uniform(0.0003, rec["gap"])


# class -> (generator, port choices, proto, packet size range, gap range, packet count range)
_SPECS: dict[str, tuple[Callable, list[int], str, tuple[float, float], tuple[float, float], tuple[int, int]]] = {
    "voip": (_voip, [10000, 16384, 3478], "UDP", (110, 230), (0.018, 0.035), (60, 400)),
    "chat": (_chat, [443, 5222, 5223], "TCP", (150, 420), (2.0, 9.0), (14, 80)),
    "streaming": (_streaming, [443, 80], "TCP", (1250, 1480), (2.0, 8.0), (120, 600)),
    "email": (_email, [993, 587, 143, 25], "TCP", (300, 900), (0.02, 0.2), (20, 150)),
    "p2p": (_p2p, [6881, 6889, 51413], "UDP", (500, 1300), (0.005, 0.08), (40, 400)),
    "file_transfer": (_file_transfer, [22, 21, 990], "TCP", (1380, 1500), (0.001, 0.006), (200, 900)),
}


# Drift moves a class's size/timing toward another class's, like a voice app
# switching to a video-style codec or a chat app starting to send media.
DRIFT_TOWARD = {
    "voip": "streaming", "chat": "email", "streaming": "voip",
    "email": "chat", "p2p": "file_transfer", "file_transfer": "p2p",
}


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def generate_flows(
    recordings_per_class: int = 6,
    flows_per_recording: int = 40,
    *,
    seed: int = 0,
    drift: float = 0.0,
    classes: tuple[str, ...] = CLASSES,
) -> list[dict]:
    """Return flows in the format ``flow_builder.build_flows`` produces.

    ``drift`` (0-1) moves each class's packet sizes and timing toward another
    class's (see ``DRIFT_TOWARD``), to simulate traffic changing over time.
    """
    rng = random.Random(seed)
    flows = []
    for label in classes:
        generator, ports, proto, size_range, gap_range, count_range = _SPECS[label]
        target = _SPECS[DRIFT_TOWARD[label]]
        size_range = _lerp(size_range, target[3], drift)
        gap_range = _lerp(gap_range, target[4], drift)
        for r in range(recordings_per_class):
            name = recording_name(label, r + 1)
            rec = {
                "size": rng.uniform(*size_range),
                "gap": rng.uniform(*gap_range),
            }
            client_ip, server_ip = f"10.1.{r}.2", f"10.9.{r}.7"
            for f in range(flows_per_recording):
                flow = {
                    "src_ip": client_ip, "src_port": rng.randint(32768, 60999),
                    "dst_ip": server_ip, "dst_port": rng.choice(ports),
                    "proto": proto, "source_file": name, "packets": [],
                }
                n = rng.randint(*count_range)
                start = f * 400.0
                packets = [dict(p, timestamp=start + p["timestamp"]) for p in generator(rng, rec, flow, n)][:n]
                flow["packets"] = packets
                flows.append(flow)
    return flows


def write_features_csv(flows: list[dict], output_csv: str | Path, max_packets: int | None = None) -> int:
    rows = [extract_features(flow, max_packets) for flow in flows]
    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_pcaps(flows: list[dict], pcap_dir: str | Path) -> int:
    """Write one PCAP per recording (needs scapy) so the real parser can be tested."""
    from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap

    by_file: dict[str, list] = {}
    for flow in flows:
        for p in flow["packets"]:
            transport = (UDP if p["proto"] == "UDP" else TCP)(sport=p["src_port"], dport=p["dst_port"])
            overhead = 14 + 20 + (8 if p["proto"] == "UDP" else 20)
            pkt = Ether() / IP(src=p["src_ip"], dst=p["dst_ip"]) / transport / Raw(b"x" * max(0, p["size"] - overhead))
            pkt.time = p["timestamp"]
            by_file.setdefault(p["source_file"], []).append(pkt)
    os.makedirs(pcap_dir, exist_ok=True)
    for name, packets in by_file.items():
        wrpcap(os.path.join(pcap_dir, name), packets)
    return len(by_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--recordings", type=int, default=6, help="recordings per class")
    parser.add_argument("--flows", type=int, default=40, help="flows per recording")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--max-packets", type=int, default=None)
    parser.add_argument("--pcap-dir", type=Path)
    args = parser.parse_args()
    flows = generate_flows(args.recordings, args.flows, seed=args.seed, drift=args.drift)
    print(f"Wrote {write_features_csv(flows, args.output_csv, args.max_packets)} synthetic flows to {args.output_csv}")
    if args.pcap_dir:
        print(f"Wrote {write_pcaps(flows, args.pcap_dir)} synthetic PCAPs to {args.pcap_dir}")
    print("NOTE: synthetic data - for pipeline testing and demos, not real-world accuracy.")


if __name__ == "__main__":
    main()
