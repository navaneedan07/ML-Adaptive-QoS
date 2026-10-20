"""Live packet capture and idle-flow assembly.

Scapy is imported lazily so the controller's API and dry-run mode work on
Windows without raw-socket privileges.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from traffic.feature_extractor import extract_features


@dataclass
class FlowTracker:
    idle_timeout: float = 5.0
    source_file: str = "live"
    _flows: dict[tuple[object, ...], dict] = field(default_factory=dict)

    def _key(self, packet: dict) -> tuple[object, ...]:
        left = (packet["src_ip"], packet["src_port"])
        right = (packet["dst_ip"], packet["dst_port"])
        endpoints = tuple(sorted((left, right)))
        return (*endpoints, packet["proto"])

    def add(self, packet: dict, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        key = self._key(packet)
        flow = self._flows.get(key)
        if flow is None:
            flow = {
                "src_ip": key[0][0], "src_port": key[0][1],
                "dst_ip": key[1][0], "dst_port": key[1][1],
                "proto": key[2], "packets": [], "source_file": self.source_file,
            }
            self._flows[key] = flow
        flow["packets"].append(packet)
        return self.flush(now)

    def flush(self, now: float | None = None, force: bool = False) -> list[dict]:
        now = time.time() if now is None else now
        completed = []
        for key, flow in list(self._flows.items()):
            last = float(flow["packets"][-1]["timestamp"])
            if force or now - last >= self.idle_timeout:
                completed.append(extract_features(flow))
                del self._flows[key]
        return completed


def packet_to_record(packet: object, source_file: str = "live") -> dict:
    """Convert a Scapy IP packet into the project's packet record shape."""
    from scapy.layers.inet import IP, TCP, UDP

    if IP not in packet:
        raise ValueError("only IPv4 packets are supported")
    ip = packet[IP]
    transport = packet.getlayer(TCP) or packet.getlayer(UDP)
    proto = "TCP" if TCP in packet else "UDP" if UDP in packet else "OTHER"
    return {
        "timestamp": float(packet.time), "size": len(packet), "proto": proto,
        "src_ip": ip.src, "dst_ip": ip.dst,
        "src_port": int(transport.sport) if transport else 0,
        "dst_port": int(transport.dport) if transport else 0,
        "source_file": source_file,
    }


def sniff_forever(
    on_flow: Callable[[dict], None],
    *,
    interface: str | None = None,
    idle_timeout: float = 5.0,
    stop_filter: Callable[[object], bool] | None = None,
) -> None:
    """Capture packets until interrupted and emit completed feature rows."""
    from scapy.sendrecv import sniff

    tracker = FlowTracker(idle_timeout=idle_timeout)

    def handle(packet: object) -> None:
        try:
            for flow in tracker.add(packet_to_record(packet)):
                on_flow(flow)
        except ValueError:
            return

    sniff(iface=interface, prn=handle, store=False, stop_filter=stop_filter)
    for flow in tracker.flush(force=True):
        on_flow(flow)

