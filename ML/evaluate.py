"""
Evaluate RF + SVM baselines on the held-out test set.

Produces exactly what slide D asks for: accuracy, precision, recall, F1,
confusion matrix, inference time — for both models, so they can be compared.

Input : data/processed/test.csv, models/random_forest.joblib, models/svm.joblib
Output: results/classification_report.txt
        results/confusion_matrix_rf.png, results/confusion_matrix_svm.png
        results/metrics_summary.csv   (one row per model — accuracy/F1/inference time)

Usage:
    python evaluate.py
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def time_inference(predict_fn, X, n_repeats=5):
    """Mean per-flow inference time in milliseconds, averaged over n_repeats passes."""
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        predict_fn(X)
        elapsed = time.perf_counter() - start
        times.append(elapsed / len(X) * 1000)  # ms per flow
    return sum(times) / len(times)


def evaluate_model(name, y_true, y_pred, inference_ms, out_dir: Path):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision (wtd): {prec:.4f}")
    print(f"Recall (wtd):    {rec:.4f}")
    print(f"F1 (wtd):        {f1:.4f}")
    print(f"Inference time:  {inference_ms:.4f} ms/flow")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    fig_path = out_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {fig_path}")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "inference_ms_per_flow": inference_ms,
    }, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate RF + SVM baselines")
    parser.add_argument("--test", default="../data/processed/test.csv")
    parser.add_argument("--models", default="../models")
    parser.add_argument("--outdir", default="../results")
    args = parser.parse_args()

    models_dir = Path(args.models)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "feature_columns.json") as f:
        schema = json.load(f)
    feature_cols, label_col = schema["feature_cols"], schema["label_col"]

    df = pd.read_csv(args.test)
    X, y_true = df[feature_cols], df[label_col]

    all_metrics = []
    reports = []

    # --- Random Forest ---
    rf = joblib.load(models_dir / "random_forest.joblib")
    rf_ms = time_inference(rf.predict, X)
    y_pred_rf = rf.predict(X)
    metrics_rf, report_rf = evaluate_model("Random Forest", y_true, y_pred_rf, rf_ms, out_dir)
    all_metrics.append(metrics_rf)
    reports.append(f"=== Random Forest ===\n{report_rf}")

    # --- SVM ---
    svm_bundle = joblib.load(models_dir / "svm.joblib")
    svm, scaler = svm_bundle["model"], svm_bundle["scaler"]
    X_scaled = scaler.transform(X)
    svm_ms = time_inference(svm.predict, X_scaled)
    y_pred_svm = svm.predict(X_scaled)
    metrics_svm, report_svm = evaluate_model("SVM", y_true, y_pred_svm, svm_ms, out_dir)
    all_metrics.append(metrics_svm)
    reports.append(f"=== SVM ===\n{report_svm}")

    pd.DataFrame(all_metrics).to_csv(out_dir / "metrics_summary.csv", index=False)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write("\n\n".join(reports))

    print(f"\nSaved summary -> {out_dir / 'metrics_summary.csv'}")
    print(f"Saved full reports -> {out_dir / 'classification_report.txt'}")
    print("\nDone. Next: python predict.py")


if __name__ == "__main__":
    main()