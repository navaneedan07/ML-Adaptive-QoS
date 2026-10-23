"""Shared feature definitions for training and online prediction."""

from __future__ import annotations

from typing import Any

FEATURE_COLUMNS = [
    "proto",
    "flow_duration",
    "packet_count",
    "total_bytes",
    "mean_packet_size",
    "std_packet_size",
    "mean_iat",
    "std_iat",
    "num_bursts",
    "mean_burst_size",
    "is_vpn",
]

# Background service ports (DNS, DHCP, NTP, NetBIOS, SSDP, mDNS, LLMNR). Flows on
# these ports are name-lookup/discovery chatter that inherits its capture's label
# from the filename, so they are excluded from training and left on the default
# best-effort queue at runtime.
NOISE_PORTS = frozenset({53, 67, 68, 123, 137, 138, 139, 1900, 5353, 5355})

CATEGORICAL_COLUMNS = ["proto"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def rows_to_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only model features and normalize values read from CSV files."""
    return [
        {
            column: (row[column] if column == "proto" else float(row[column]))
            for column in FEATURE_COLUMNS
        }
        for row in rows
    ]



def is_noise_flow(flow: dict[str, Any]) -> bool:
    """True if either endpoint uses a background service port."""
    for key in ("src_port", "dst_port"):
        try:
            if int(float(flow.get(key) or 0)) in NOISE_PORTS:
                return True
        except (TypeError, ValueError):
            continue
    return False
