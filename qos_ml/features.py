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

