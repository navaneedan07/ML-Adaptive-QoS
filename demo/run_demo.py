"""Drive a recordable end-to-end dashboard demonstration.

This uses a real trained model and real feature rows from the bundled dataset.
It exercises classification, QoS policy selection, monitoring display, drift
feedback, and automatic retraining without requiring Linux privileges.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/features_balanced.csv"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--feedback-count", type=int, default=100)
    args = parser.parse_args()

    with args.dataset.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    shown = set()
    for row in rows:
        if row["label"] not in shown:
            result = post(f"{args.base_url}/api/classify", row)
            print(
                f"{row['label']:12} -> predicted={result['label']:12} "
                f"DSCP={result['enforcement']['dscp']} "
                f"queue={result['enforcement']['classid']}"
            )
            shown.add(row["label"])
        if len(shown) >= 4:
            break

    for _ in range(args.feedback_count):
        post(f"{args.base_url}/api/feedback", {"predicted": "voip", "actual": "chat"})
    print(f"Sent {args.feedback_count} feedback events; drift/retraining should now be visible.")
    print(f"Open {args.base_url}/dashboard and refresh to show the final state.")
    time.sleep(2)


if __name__ == "__main__":
    main()
