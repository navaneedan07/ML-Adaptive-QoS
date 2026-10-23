"""Summarise ping and iperf3 voice-probe results of run_benchmark.sh into a CSV.

Also parses any ``*_ping.txt`` file, so older ping logs can be summarised too.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    from network.analyze_results import read_iperf_result
except ImportError:  # run as a script from the network/ folder
    from analyze_results import read_iperf_result

FIELDS = [
    "Scenario", "Ping_Avg_ms", "Ping_Max_ms", "Ping_Loss_Percent",
    "Voice_Jitter_ms", "Voice_Loss_Percent",
]


def parse_ping(text: str) -> dict[str, float | None]:
    """Extract loss and RTT statistics from the summary of a ``ping`` run."""
    loss = re.search(r"(\d+(?:\.\d+)?)% packet loss", text)
    rtt = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", text)
    if not loss:
        raise ValueError("not a ping summary: no packet-loss line found")
    return {
        "loss_percent": float(loss.group(1)),
        "min_ms": float(rtt.group(1)) if rtt else None,
        "avg_ms": float(rtt.group(2)) if rtt else None,
        "max_ms": float(rtt.group(3)) if rtt else None,
    }


def summarize(results_dir: str | Path, output_csv: str | Path) -> list[dict]:
    rows = []
    for ping_file in sorted(Path(results_dir).glob("*_ping.txt")):
        name = ping_file.name[: -len("_ping.txt")]
        ping = parse_ping(ping_file.read_text(encoding="utf-8"))
        voice_file = ping_file.with_name(f"{name}_voice.json")
        voice = read_iperf_result(voice_file) if voice_file.exists() else {}
        rows.append(
            {
                "Scenario": name,
                "Ping_Avg_ms": ping["avg_ms"],
                "Ping_Max_ms": ping["max_ms"],
                "Ping_Loss_Percent": ping["loss_percent"],
                "Voice_Jitter_ms": voice.get("jitter_ms"),
                "Voice_Loss_Percent": voice.get("packet_loss_percent"),
            }
        )
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    for row in summarize(args.results_dir, args.output_csv):
        print(row)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
