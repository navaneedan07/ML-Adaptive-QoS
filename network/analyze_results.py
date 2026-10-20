"""Compare two iperf3 JSON result files and write a CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_iperf_result(filename: str | Path) -> dict[str, float | int]:
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    try:
        udp = data["end"]["sum"]
        return {
            "throughput_mbps": udp["bits_per_second"] / 1_000_000,
            "jitter_ms": udp.get("jitter_ms", 0.0),
            "lost_packets": udp.get("lost_packets", 0),
            "packets": udp.get("packets", 0),
            "packet_loss_percent": udp.get("lost_percent", 0.0),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid iperf3 JSON result: {filename}") from exc


def write_comparison(best_effort: str | Path, qos: str | Path, output: str | Path) -> None:
    fields = ["Scenario", "Throughput_Mbps", "Jitter_ms", "Lost_Packets", "Packets", "Packet_Loss_Percent"]
    rows = [("Best Effort", read_iperf_result(best_effort)), ("QoS", read_iperf_result(qos))]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for label, metrics in rows:
            writer.writerow([
                label,
                metrics["throughput_mbps"],
                metrics["jitter_ms"],
                metrics["lost_packets"],
                metrics["packets"],
                metrics["packet_loss_percent"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("best_effort", type=Path)
    parser.add_argument("qos", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_comparison(args.best_effort, args.qos, args.output)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
