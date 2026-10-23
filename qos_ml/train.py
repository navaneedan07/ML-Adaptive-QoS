"""Train and evaluate traffic classifiers with a per-recording test split.

The model intentionally excludes addresses, ports, and source filenames. Those
fields identify the capture rather than the traffic behavior and cause leakage
when a model is used on a new network.

Evaluation is also split by recording (``source_file``): every flow from one
capture is either in the training set or in the test set, never both. A random
split over flows lets the model memorise a capture and inflates the scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from qos_ml.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    is_noise_flow,
    rows_to_features,
)
from qos_ml.model_selection import measure_inference_ms
from qos_ml.qos_policy import policy_for

MODEL_NAMES = ("random_forest", "svm", "hist_gb", "decision_tree")
_SPLIT_ATTEMPTS = 20
DEFAULT_MIN_PACKETS = 10


def queue_of(label: str) -> str:
    """The tc queue a class is mapped to (what QoS actually acts on)."""
    return str(policy_for(label)["classid"])


def _load_rows(
    path: str | Path, min_packets: int = DEFAULT_MIN_PACKETS, drop_noise_ports: bool = True
) -> list[dict[str, Any]]:
    """Read a feature CSV, dropping ``unknown`` rows and flows under ``min_packets``.
    Flows on background service ports (see ``NOISE_PORTS``) are dropped unless
    ``drop_noise_ports`` is False.

    Most flows in the raw captures are 2-3 packet name-resolution chatter
    (DNS/LLMNR/mDNS) that inherits the capture's label from its filename. They
    carry no traffic-type signal, so training on them teaches noise.
    """
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    missing = [column for column in FEATURE_COLUMNS + ["label"] if column not in rows[0]]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    rows = [
        row
        for row in rows
        if row["label"] != "unknown"
        and float(row["packet_count"]) >= min_packets
        and not (drop_noise_ports and is_noise_flow(row))
    ]
    if len({row["label"] for row in rows}) < 2:
        raise ValueError("At least two known classes are required for training")
    return rows


def _groups(rows: list[dict[str, Any]]) -> list[str]:
    return [row.get("source_file") or "" for row in rows]


def split_rows(
    rows: list[dict[str, Any]], test_size: float = 0.2, seed: int = 42
) -> tuple[list[int], list[int], str]:
    """Return (train_idx, test_idx, method), keeping each recording on one side.

    Tries several shuffles until every test class also appears in training. Falls
    back to a stratified random split when there are too few recordings.
    """
    labels = [row["label"] for row in rows]
    groups = _groups(rows)
    folds = max(2, round(1 / test_size))
    if len(set(groups)) >= folds and "" not in set(groups):
        for attempt in range(_SPLIT_ATTEMPTS):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + attempt)
            try:
                train_idx, test_idx = next(splitter.split(np.zeros(len(rows)), labels, groups))
            except ValueError:
                break
            train_labels = {labels[i] for i in train_idx}
            test_labels = {labels[i] for i in test_idx}
            if test_labels <= train_labels and len(train_labels) >= 2:
                return list(train_idx), list(test_idx), "by_recording"
    train_idx, test_idx = train_test_split(
        list(range(len(rows))), test_size=test_size, random_state=seed, stratify=labels
    )
    return train_idx, test_idx, "random_flows"


def load_holdout(
    dataset_path: str | Path,
    test_size: float = 0.2,
    seed: int = 42,
    min_packets: int = DEFAULT_MIN_PACKETS,
    drop_noise_ports: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the same held-out test rows that ``train_models`` would use."""
    rows = _load_rows(dataset_path, min_packets, drop_noise_ports)
    _, test_idx, _ = split_rows(rows, test_size, seed)
    return (
        pd.DataFrame(rows_to_features([rows[i] for i in test_idx])),
        [rows[i]["label"] for i in test_idx],
    )


def load_holdout_flags(
    dataset_path: str | Path,
    test_size: float = 0.2,
    seed: int = 42,
    min_packets: int = DEFAULT_MIN_PACKETS,
    drop_noise_ports: bool = True,
) -> tuple[pd.DataFrame, list[str], list[bool]]:
    """Like ``load_holdout`` plus a flag per row: True if it is live feedback data."""
    rows = _load_rows(dataset_path, min_packets, drop_noise_ports)
    _, test_idx, _ = split_rows(rows, test_size, seed)
    test_rows = [rows[i] for i in test_idx]
    return (
        pd.DataFrame(rows_to_features(test_rows)),
        [row["label"] for row in test_rows],
        [(row.get("source_file") or "").startswith("live_") for row in test_rows],
    )


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ],
        sparse_threshold=0,
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
        # Compact tree: the lightweight option for a tight inference budget.
        "decision_tree": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "hist_gb": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=100,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def _score(y_true: list[str], predictions: Iterable[str]) -> dict[str, Any]:
    predictions = list(predictions)
    true_queues = [queue_of(label) for label in y_true]
    predicted_queues = [queue_of(label) for label in predictions]
    return {
        "queue": {
            "accuracy": accuracy_score(true_queues, predicted_queues),
            "macro_f1": f1_score(true_queues, predicted_queues, average="macro", zero_division=0),
            "classification_report": classification_report(
                true_queues, predicted_queues, output_dict=True, zero_division=0
            ),
        },
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }


def _group_cv(
    name: str, rows: list[dict[str, Any]], folds: int, seed: int
) -> dict[str, float] | None:
    """Macro-F1 across several recording-wise folds (a steadier estimate)."""
    labels = [row["label"] for row in rows]
    groups = _groups(rows)
    if len(set(groups)) < folds or "" in set(groups):
        return None
    frame = pd.DataFrame(rows_to_features(rows))
    scores = []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(frame, labels, groups):
        train_labels = [labels[i] for i in train_idx]
        if len(set(train_labels)) < 2:
            continue
        model = _models(seed)[name]
        model.fit(frame.iloc[train_idx], train_labels)
        predicted = model.predict(frame.iloc[test_idx])
        scores.append(f1_score([labels[i] for i in test_idx], predicted, average="macro", zero_division=0))
    if not scores:
        return None
    return {"folds": len(scores), "macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores))}


def train_models(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    eval_dataset_path: str | Path | None = None,
    cv_folds: int = 0,
    model_names: Iterable[str] | None = None,
    min_packets: int = DEFAULT_MIN_PACKETS,
    drop_noise_ports: bool = True,
) -> dict[str, Any]:
    """Train models, write them with reports, and return the summary.

    ``eval_dataset_path`` is an optional natural-distribution CSV (for example the
    unbalanced features file). Its rows from recordings used in training are
    dropped, and the rest become a second, realistic test set.
    ``cv_folds`` >= 2 adds a cross-validated per-recording macro-F1.
    ``min_packets`` drops flows shorter than that from training and testing, and
    ``drop_noise_ports`` drops background-service flows (DNS, LLMNR, mDNS, ...).
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    names = list(model_names or MODEL_NAMES)
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    rows = _load_rows(dataset_path, min_packets, drop_noise_ports)
    train_idx, test_idx, method = split_rows(rows, test_size, seed)
    x_train = pd.DataFrame(rows_to_features([rows[i] for i in train_idx]))
    y_train = [rows[i]["label"] for i in train_idx]
    x_test = pd.DataFrame(rows_to_features([rows[i] for i in test_idx]))
    y_test = [rows[i]["label"] for i in test_idx]
    train_recordings = {_groups(rows)[i] for i in train_idx}

    natural_x = natural_y = None
    if eval_dataset_path:
        natural_rows = [
            row
            for row in _load_rows(eval_dataset_path, min_packets, drop_noise_ports)
            if (row.get("source_file") or "") not in train_recordings
        ]
        if natural_rows:
            natural_x = pd.DataFrame(rows_to_features(natural_rows))
            natural_y = [row["label"] for row in natural_rows]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": str(dataset_path),
        "samples": len(rows),
        "min_packets": min_packets,
        "drop_noise_ports": drop_noise_ports,
        "split": {
            "method": method,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "test_recordings": sorted({_groups(rows)[i] for i in test_idx}),
            "test_class_counts": {label: y_test.count(label) for label in sorted(set(y_test))},
        },
        "models": {},
    }

    for name in names:
        model = _models(seed)[name]
        model.fit(x_train, y_train)
        if name == "random_forest":
            # Parallel workers only add overhead when classifying one flow at a time.
            model.set_params(classifier__n_jobs=1)
        metrics = _score(y_test, model.predict(x_test))
        metrics["inference_ms_per_flow"] = measure_inference_ms(
            model, rows_to_features([rows[i] for i in test_idx])
        )
        if natural_x is not None:
            metrics["natural_test"] = {
                "rows": len(natural_y),
                **_score(natural_y, model.predict(natural_x)),
            }
        if cv_folds >= 2:
            metrics["group_cv"] = _group_cv(name, rows, cv_folds, seed)
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
    parser.add_argument("--eval-dataset", type=Path, help="natural-distribution CSV for a second test")
    parser.add_argument("--cv-folds", type=int, default=0, help="per-recording cross-validation folds")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="train only these models")
    parser.add_argument(
        "--min-packets",
        type=int,
        default=DEFAULT_MIN_PACKETS,
        help="ignore flows shorter than this (they are mostly DNS/LLMNR noise); 0 keeps all",
    )
    parser.add_argument(
        "--keep-noise-ports", action="store_true",
        help="keep DNS/LLMNR/mDNS/NetBIOS/... flows in the training data",
    )
    args = parser.parse_args()
    summary = train_models(
        args.dataset,
        args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        eval_dataset_path=args.eval_dataset,
        cv_folds=args.cv_folds,
        model_names=args.models,
        min_packets=args.min_packets,
        drop_noise_ports=not args.keep_noise_ports,
    )
    split = summary["split"]
    print(f"split: {split['method']} ({split['train_rows']} train / {split['test_rows']} test rows)")
    if split["method"] == "by_recording":
        print(f"test recordings: {', '.join(split['test_recordings'])}")
    else:
        print("WARNING: too few recordings for a per-recording split; scores are optimistic")
    for name, metrics in summary["models"].items():
        line = (
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}, "
            f"queue_acc={metrics['queue']['accuracy']:.4f}, "
            f"inference={metrics['inference_ms_per_flow']:.2f} ms/flow"
        )
        if "natural_test" in metrics:
            line += f" | natural macro_f1={metrics['natural_test']['macro_f1']:.4f}"
        if metrics.get("group_cv"):
            cv = metrics["group_cv"]
            line += f" | cv macro_f1={cv['macro_f1_mean']:.4f}+/-{cv['macro_f1_std']:.4f}"
        print(line)


if __name__ == "__main__":
    main()
