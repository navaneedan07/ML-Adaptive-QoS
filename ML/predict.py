"""
Produce prediction.csv — the handoff file for Saravana (slide E).

Format:
    flow_id,predicted_class,confidence
    1,video,0.94
    2,bulk,0.88
    3,voip,0.91

By default uses the Random Forest model (usually faster + competitive with SVM
on tabular flow features — swap --model svm if your evaluate.py results say
otherwise).

Input : data/processed/features.csv (or any file with the same feature schema)
Output: results/prediction.csv

Usage:
    python predict.py
    python predict.py --model svm
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

# Sriram's features.csv has no flow_id column — we generate one from row number.
FLOW_ID_CANDIDATES = ["flow_id", "id"]


def find_flow_id_column(df: pd.DataFrame):
    for c in FLOW_ID_CANDIDATES:
        if c in df.columns:
            return c
    return None  # fall back to a generated 1-based flow_id


def main():
    parser = argparse.ArgumentParser(description="Generate prediction.csv for Saravana")
    parser.add_argument("--input", default="../data/processed/features.csv",
                         help="Flows to predict on (needs the same feature columns as training)")
    parser.add_argument("--models", default="../models")
    parser.add_argument("--model", choices=["rf", "svm"], default="rf")
    parser.add_argument("--outdir", default="../results")
    args = parser.parse_args()

    models_dir = Path(args.models)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "feature_columns.json") as f:
        schema = json.load(f)
    feature_cols = schema["feature_cols"]

    df = pd.read_csv(args.input)
    flow_id_col = find_flow_id_column(df)
    flow_ids = df[flow_id_col] if flow_id_col else (df.index + 1)  # 1-based generated id

    X = df[feature_cols]

    if args.model == "rf":
        model = joblib.load(models_dir / "random_forest.joblib")
        probs = model.predict_proba(X)
        classes = model.classes_
    else:
        bundle = joblib.load(models_dir / "svm.joblib")
        model, scaler = bundle["model"], bundle["scaler"]
        X = scaler.transform(X)
        probs = model.predict_proba(X)
        classes = model.classes_

    pred_idx = probs.argmax(axis=1)
    predicted_class = [classes[i] for i in pred_idx]
    confidence = probs.max(axis=1)

    out_df = pd.DataFrame({
        "flow_id": flow_ids,
        "predicted_class": predicted_class,
        "confidence": confidence.round(2),
    })

    out_path = out_dir / "prediction.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df):,} predictions -> {out_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()