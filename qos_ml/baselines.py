"""Baselines the ML classifier is compared against.

* ``classify_by_port``: the classic port-based method. It names the class from
  well-known ports. It is weak by design: nearly everything is on 443 now,
  which is the problem the project sets out to solve.
* ``classify_best_effort``: no classification at all (every flow is best-effort).
"""

from __future__ import annotations

from typing import Any

EMAIL_PORTS = {25, 110, 143, 465, 587, 993, 995}
CHAT_PORTS = {1863, 4244, 5190, 5222, 5223, 6667}
STREAMING_PORTS = {80, 443, 554, 1935, 8080, 8443}
VOIP_PORTS = {3478, 3479, 5004, 5005, 5060, 5061}
RTP_RANGE = range(16384, 32768)  # UDP media ports


def classify_by_port(flow: dict[str, Any]) -> str:
    """Return a class from the flow's ports, or ``unknown`` if no rule matches."""
    ports = []
    for key in ("src_port", "dst_port"):
        try:
            ports.append(int(float(flow.get(key) or 0)))
        except (TypeError, ValueError):
            continue
    udp = str(flow.get("proto", "")).upper() == "UDP"
    for port in ports:
        if port in EMAIL_PORTS:
            return "email"
        if port in CHAT_PORTS:
            return "chat"
        if port in VOIP_PORTS or (udp and port in RTP_RANGE):
            return "voip"
    for port in ports:
        if port in STREAMING_PORTS:
            return "streaming"
    return "unknown"


def classify_best_effort(flow: dict[str, Any]) -> str:
    return "unknown"
