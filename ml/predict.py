"""Predict traffic classes from feature CSV rows using a saved model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import pandas as pd

from ml.features import rows_to_features


def predict_file(model_path: str | Path, input_csv: str | Path, output_csv: str | Path) -> int:
    model = joblib.load(model_path)
    with Path(input_csv).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Input CSV is empty: {input_csv}")
    predictions = model.predict(pd.DataFrame(rows_to_features(rows)))
    fieldnames = list(rows[0]) + ["predicted_label"]
    with Path(output_csv).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({**row, "predicted_label": prediction})
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    print(f"Wrote predictions for {predict_file(args.model, args.input_csv, args.output_csv)} flows")


if __name__ == "__main__":
    main()
