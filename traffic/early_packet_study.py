"""How accurate is classification after only the first N packets of a flow?

Builds flows from raw PCAPs once, then for each N extracts features from the
first N packets, trains with a per-recording split, and writes one row per N.
Only flows with at least N packets are used, because a shorter flow can never
be classified early (it stays on the default best-effort queue).

Usage:
    python -m traffic.early_packet_study data/raw results/early_packet_accuracy.csv
    python -m traffic.early_packet_study data/raw out.csv --packets 3 5 10 20 50
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

from qos_ml.train import MODEL_NAMES, train_models
from traffic.feature_extractor import FIELDNAMES, extract_features

DEFAULT_PACKETS = (3, 5, 10, 20, 50)


def run_study(
    flows: list[dict],
    packet_counts: tuple[int, ...] | list[int] = DEFAULT_PACKETS,
    *,
    model_names: tuple[str, ...] = ("random_forest", "hist_gb"),
    dataset_dir: str | Path | None = None,
    seed: int = 42,
) -> list[dict]:
    """Return one result row per (packet count, model)."""
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for n in packet_counts:
            usable = [flow for flow in flows if len(flow["packets"]) >= n]
            rows = [extract_features(flow, n) for flow in usable]
            rows = [row for row in rows if row["label"] != "unknown"]
            if len({row["label"] for row in rows}) < 2:
                continue
            csv_dir = Path(dataset_dir) if dataset_dir else Path(scratch)
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"features_first{n}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
            summary = train_models(
                csv_path, Path(scratch) / f"models_{n}", seed=seed, model_names=list(model_names),
                min_packets=0,  # flows were already filtered to >= n packets
            )
            for name, metrics in summary["models"].items():
                results.append(
                    {
                        "packets": n,
                        "model": name,
                        "flows": len(rows),
                        "split": summary["split"]["method"],
                        "accuracy": round(metrics["accuracy"], 4),
                        "macro_f1": round(metrics["macro_f1"], 4),
                    }
                )
    return results


def write_results(results: list[dict], output_csv: str | Path) -> None:
    os.makedirs(os.path.dirname(str(output_csv)) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["packets", "model", "flows", "split", "accuracy", "macro_f1"]
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap_target", help="PCAP file or folder")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--packets", type=int, nargs="+", default=list(DEFAULT_PACKETS))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["random_forest", "hist_gb"])
    parser.add_argument("--dataset-dir", type=Path, help="also keep the per-N feature CSVs here")
    args = parser.parse_args()

    from traffic.capture import parse_folder, parse_pcap
    from traffic.flow_builder import build_flows

    target = args.pcap_target
    packets = parse_folder(target) if os.path.isdir(target) else parse_pcap(target)
    flows = build_flows(packets)
    results = run_study(
        flows, args.packets, model_names=tuple(args.models), dataset_dir=args.dataset_dir
    )
    write_results(results, args.output_csv)
    print(f"{'packets':>8} {'model':<14} {'flows':>7} {'macro_f1':>9} {'accuracy':>9}")
    for row in results:
        print(
            f"{row['packets']:>8} {row['model']:<14} {row['flows']:>7} "
            f"{row['macro_f1']:>9} {row['accuracy']:>9}"
        )


if __name__ == "__main__":
    main()
