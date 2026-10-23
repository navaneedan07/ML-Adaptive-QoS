"""Honest evaluation report for slides: per-recording cross-validation.

Every flow is predicted by a model that never saw its recording. Writes class,
queue and confidence tables (CSV), a confusion matrix and a confidence chart
(PNG, if matplotlib is installed) and prints a recommended ``--min-confidence``.

Usage:
    python -m qos_ml.report data/processed/features.csv --output-dir results/honest
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from qos_ml.baselines import classify_best_effort, classify_by_port
from qos_ml.features import rows_to_features
from qos_ml.model_selection import measure_inference_ms, select_model
from qos_ml.train import DEFAULT_MIN_PACKETS, MODEL_NAMES, _load_rows, _models, queue_of

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def cross_validated_predictions(
    rows: list[dict[str, Any]], model_name: str, *, folds: int = 4, seeds: int = 5
) -> pd.DataFrame:
    """Out-of-recording predictions: columns true, predicted, confidence."""
    labels = np.array([row["label"] for row in rows])
    groups = np.array([row.get("source_file") or "" for row in rows])
    if len(set(groups)) < folds or "" in set(groups):
        raise ValueError(f"need at least {folds} distinct recordings (source_file) for cross-validation")
    frame = pd.DataFrame(rows_to_features(rows))
    records = []
    for seed in range(seeds):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            splits = list(splitter.split(frame, labels, groups))
        for train_idx, test_idx in splits:
            if len(set(labels[train_idx])) < 2:
                continue
            model = _models(seed)[model_name]
            model.fit(frame.iloc[train_idx], labels[train_idx])
            probabilities = model.predict_proba(frame.iloc[test_idx])
            best = probabilities.argmax(axis=1)
            for truth, index, row_probabilities in zip(labels[test_idx], best, probabilities):
                records.append((truth, str(model.classes_[index]), float(row_probabilities[index])))
    return pd.DataFrame(records, columns=["true", "predicted", "confidence"])


def confidence_table(predictions: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Share of flows answered, and accuracy of those answers, per threshold."""
    queue_true = predictions["true"].map(queue_of)
    queue_pred = predictions["predicted"].map(queue_of)
    rows = []
    for threshold in thresholds:
        answered = predictions["confidence"] >= threshold
        rows.append(
            {
                "min_confidence": threshold,
                "flows_answered": round(float(answered.mean()), 3),
                "class_accuracy": round(float((predictions["true"] == predictions["predicted"])[answered].mean()), 3)
                if answered.any() else float("nan"),
                "queue_accuracy": round(float((queue_true == queue_pred)[answered].mean()), 3)
                if answered.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def recommend_threshold(table: pd.DataFrame, target_queue_accuracy: float = 0.75) -> float | None:
    """Lowest threshold whose answered flows reach the target queue accuracy."""
    ok = table[table["queue_accuracy"] >= target_queue_accuracy]
    return float(ok["min_confidence"].min()) if len(ok) else None


def _summary_metrics(true: list[str], predicted: list[str]) -> dict[str, float]:
    return {
        "class_macro_f1": round(float(f1_score(true, predicted, average="macro", zero_division=0)), 3),
        "class_accuracy": round(float(accuracy_score(true, predicted)), 3),
        "queue_accuracy": round(float(accuracy_score([queue_of(t) for t in true], [queue_of(p) for p in predicted])), 3),
    }


def compare_methods(
    rows: list[dict[str, Any]],
    *,
    folds: int = 4,
    seeds: int = 5,
    latency_budget_ms: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Baselines vs every ML model, and cost-based model selection.

    Returns (baseline_comparison, model_selection, chosen_model). ML numbers are
    per-recording cross-validated; the baselines need no training.
    """
    truth = [row["label"] for row in rows]
    comparison = [
        {"method": "no classification (all best-effort)", **_summary_metrics(truth, [classify_best_effort(r) for r in rows])},
        {"method": "port-based", **_summary_metrics(truth, [classify_by_port(r) for r in rows])},
    ]
    candidates = []
    frame_rows = rows_to_features(rows)
    for name in MODEL_NAMES:
        predictions = cross_validated_predictions(rows, name, folds=folds, seeds=seeds)
        metrics = _summary_metrics(list(predictions["true"]), list(predictions["predicted"]))
        comparison.append({"method": f"ML: {name}", **metrics})
        model = _models(0)[name]
        model.fit(pd.DataFrame(frame_rows), truth)
        if name == "random_forest":
            model.set_params(classifier__n_jobs=1)
        candidates.append(
            {"name": name, "macro_f1": metrics["class_macro_f1"], "inference_ms": measure_inference_ms(model, rows)}
        )
    chosen, table = select_model(candidates, budget_ms=latency_budget_ms, alpha=alpha, beta=beta)
    selection = pd.DataFrame(table).round({"macro_f1": 3, "inference_ms": 3, "cost": 4})
    return pd.DataFrame(comparison), selection, chosen


def build_report(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "hist_gb",
    min_packets: int = DEFAULT_MIN_PACKETS,
    folds: int = 4,
    seeds: int = 5,
    target_queue_accuracy: float = 0.75,
    latency_budget_ms: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.02,
) -> dict[str, Any]:
    rows = _load_rows(dataset, min_packets)
    predictions = cross_validated_predictions(rows, model_name, folds=folds, seeds=seeds)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    class_report = pd.DataFrame(
        classification_report(predictions["true"], predictions["predicted"], output_dict=True, zero_division=0)
    ).T
    queue_report = pd.DataFrame(
        classification_report(
            predictions["true"].map(queue_of), predictions["predicted"].map(queue_of),
            output_dict=True, zero_division=0,
        )
    ).T
    table = confidence_table(predictions)
    class_report.round(3).to_csv(out / "per_class.csv", index_label="class")
    queue_report.round(3).to_csv(out / "per_queue.csv", index_label="queue")
    table.to_csv(out / "confidence_curve.csv", index=False)

    classes = sorted(set(predictions["true"]) | set(predictions["predicted"]))
    matrix = confusion_matrix(predictions["true"], predictions["predicted"], labels=classes)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv", index_label="true")
    _plots(out, classes, matrix, table, model_name)

    baselines, selection, chosen = compare_methods(
        rows, folds=folds, seeds=seeds, latency_budget_ms=latency_budget_ms, alpha=alpha, beta=beta
    )
    baselines.to_csv(out / "baseline_comparison.csv", index=False)
    selection.to_csv(out / "model_selection.csv", index=False)

    summary = {
        "selected_model": chosen,
        "model": model_name,
        "flows": len(rows),
        "recordings": len({row["source_file"] for row in rows}),
        "class_macro_f1": float(class_report.loc["macro avg", "f1-score"]),
        "class_accuracy": float(class_report.loc["accuracy", "precision"]),
        "queue_accuracy": float(queue_report.loc["accuracy", "precision"]),
        "recommended_min_confidence": recommend_threshold(table, target_queue_accuracy),
    }
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(summary.items())
    return summary


def _plots(out: Path, classes: list[str], matrix: np.ndarray, table: pd.DataFrame, model_name: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{model_name}: unseen recordings")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(table["min_confidence"], table["queue_accuracy"], marker="o", label="queue accuracy")
    ax.plot(table["min_confidence"], table["class_accuracy"], marker="o", label="class accuracy")
    ax.plot(table["min_confidence"], table["flows_answered"], marker="o", linestyle="--", label="share of flows answered")
    ax.set_xlabel("minimum confidence")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Higher threshold: more accurate, fewer flows prioritised")
    fig.tight_layout()
    fig.savefig(out / "confidence_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/honest"))
    parser.add_argument("--model", choices=MODEL_NAMES, default="hist_gb")
    parser.add_argument("--min-packets", type=int, default=DEFAULT_MIN_PACKETS)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--target-queue-accuracy", type=float, default=0.75)
    parser.add_argument("--latency-budget-ms", type=float, default=None, help="inference budget for model selection")
    parser.add_argument("--alpha", type=float, default=1.0, help="cost weight on (1 - macro F1)")
    parser.add_argument("--beta", type=float, default=0.02, help="cost weight on inference ms per flow")
    args = parser.parse_args()
    summary = build_report(
        args.dataset, args.output_dir, model_name=args.model, min_packets=args.min_packets,
        folds=args.folds, seeds=args.seeds, target_queue_accuracy=args.target_queue_accuracy,
        latency_budget_ms=args.latency_budget_ms, alpha=args.alpha, beta=args.beta,
    )
    for key, value in summary.items():
        print(f"{key}: {round(value, 3) if isinstance(value, float) else value}")


if __name__ == "__main__":
    main()
