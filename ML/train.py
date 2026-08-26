"""
Train baseline models for ML-Adaptive-QoS (Random Forest + SVM).

Deliberately NOT starting with CNN/Transformer, per the 4-week plan.

Input : data/processed/train.csv (from preprocess.py)
Output: models/random_forest.joblib, models/svm.joblib
        models/feature_columns.json (so predict.py/evaluate.py know the schema)

Usage:
    python train.py
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABEL_CANDIDATES = ["label", "class", "traffic_class", "category", "target"]


def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No label column found among {LABEL_CANDIDATES}")


def main():
    parser = argparse.ArgumentParser(description="Train RF + SVM baselines")
    parser.add_argument("--train", default="../data/processed/train.csv")
    parser.add_argument("--outdir", default="../models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    label_col = find_label_column(df)
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols]
    y = df[label_col]
    print(f"Training on {len(df):,} rows, {len(feature_cols)} features, "
          f"{y.nunique()} classes: {sorted(y.unique())}")
    counts = y.value_counts()
    print(f"Class counts before SMOTE:\n{counts}")

    # SMOTE creates synthetic examples of minority classes by interpolating
    # between real neighboring samples of the same class. We deliberately do
    # NOT oversample all the way up to the majority class's size (voip,
    # ~43k) — that would balloon total rows and make SVM training
    # impractically slow. Instead, bring every minority class up to the size
    # of the second-largest class only.
    sorted_counts = counts.sort_values(ascending=False)
    target_size = int(sorted_counts.iloc[1]) if len(sorted_counts) > 1 else int(sorted_counts.iloc[0])
    sampling_strategy = {cls: target_size for cls, cnt in counts.items() if cnt < target_size}

    if sampling_strategy:
        smallest_class_size = min(counts[cls] for cls in sampling_strategy)
        k_neighbors = max(1, min(5, smallest_class_size - 1))
        smote = SMOTE(random_state=args.seed, k_neighbors=k_neighbors,
                       sampling_strategy=sampling_strategy)
        X, y = smote.fit_resample(X, y)
        print(f"\nClass counts after SMOTE (minorities raised to {target_size:,}, "
              f"majority untouched):\n{y.value_counts()}")
    else:
        print("\nClasses already balanced enough — skipping SMOTE")

    # --- Random Forest (no scaling needed) ---
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    joblib.dump(rf, out_dir / "random_forest.joblib")
    print(f"Saved -> {out_dir / 'random_forest.joblib'}")

    # --- SVM (needs scaled features; pack scaler + model together) ---
    print("\nTraining SVM...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel="rbf", probability=True, random_state=args.seed)
    svm.fit(X_scaled, y)
    joblib.dump({"model": svm, "scaler": scaler}, out_dir / "svm.joblib")
    print(f"Saved -> {out_dir / 'svm.joblib'}")

    # Save schema so evaluate.py / predict.py stay in sync
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "label_col": label_col}, f, indent=2)
    print(f"Saved -> {out_dir / 'feature_columns.json'}")

    print("\nDone. Next: python evaluate.py")


if __name__ == "__main__":
    main()