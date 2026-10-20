"""Train and evaluate Random Forest and SVM traffic classifiers.

The model intentionally excludes addresses, ports, and source filenames. Those
fields identify the capture rather than the traffic behavior and cause leakage
when a model is used on a new network.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from qos_ml.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, rows_to_features


def _load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in rows[0]]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    rows = [row for row in rows if row["label"] != "unknown"]
    if len({row["label"] for row in rows}) < 2:
        raise ValueError("At least two known classes are required for training")
    return rows_to_features(rows), [row["label"] for row in rows]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def _models(seed: int) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    SVC(
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def train_models(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train both baselines, write models/reports, and return the summary."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rows, labels = _load_dataset(Path(dataset_path))
    x_train, x_test, y_train, y_test = train_test_split(
        rows,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"dataset": str(dataset_path), "samples": len(rows), "models": {}}

    for name, model in _models(seed).items():
        model.fit(pd.DataFrame(x_train), y_train)
        predictions = model.predict(pd.DataFrame(x_test))
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "weighted_f1": f1_score(y_test, predictions, average="weighted"),
            "classification_report": report,
        }
        joblib.dump(model, output / f"{name}.joblib")
        (output / f"{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary["models"][name] = metrics

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = train_models(args.dataset, args.output_dir, test_size=args.test_size, seed=args.seed)
    for name, metrics in summary["models"].items():
        print(f"{name}: accuracy={metrics['accuracy']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
