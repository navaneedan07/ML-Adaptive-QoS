"""
Preprocessing pipeline for ML-Adaptive-QoS.

Input : data/processed/features.csv   (Sriram's flow features, ~60,480 flows)
Output: data/processed/train.csv, data/processed/test.csv
        (cleaned, feature-selected, split — ready for train.py)

Usage:
    python preprocess.py
    python preprocess.py --input ../data/processed/features.csv --test-size 0.2
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Columns that are identifiers/metadata, never features for the model.
# Matches the actual header of Sriram's features.csv:
# src_ip,src_port,dst_ip,dst_port,proto,flow_duration,packet_count,total_bytes,
# mean_packet_size,std_packet_size,mean_iat,std_iat,num_bursts,mean_burst_size,
# is_vpn,source_file,label
NON_FEATURE_COLS = {
    "src_ip", "dst_ip", "src_port", "dst_port", "source_file",
}

# "proto" (tcp/udp) is categorical text, not numeric — select_features() drops it
# automatically. If you want protocol as a feature, one-hot encode it before
# calling select_features(), e.g. pd.get_dummies(df, columns=["proto"]).

# Candidate names for the target column — first match wins.
LABEL_CANDIDATES = ["label", "class", "traffic_class", "category", "target"]


def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a label column. Looked for {LABEL_CANDIDATES}. "
        f"Actual columns: {list(df.columns)}. "
        f"Edit LABEL_CANDIDATES in preprocess.py to match Sriram's features.csv."
    )


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns from {path}")

    before = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} rows with missing/duplicate data "
          f"({len(df):,} remain)")

    return df


def select_features(df: pd.DataFrame, label_col: str):
    drop_cols = (NON_FEATURE_COLS | {label_col}) & set(df.columns)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Keep only numeric features — RF/SVM need numeric input.
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    dropped_non_numeric = set(feature_cols) - set(numeric_cols)
    if dropped_non_numeric:
        print(f"Dropping non-numeric feature columns: {sorted(dropped_non_numeric)} "
              f"(encode these separately if they matter, e.g. protocol one-hot)")

    print(f"Selected {len(numeric_cols)} feature columns: {numeric_cols}")
    return numeric_cols


def main():
    parser = argparse.ArgumentParser(description="Clean, select features, and split features.csv")
    parser.add_argument("--input", default="../data/processed/features.csv",
                         help="Path to Sriram's features.csv")
    parser.add_argument("--outdir", default="../data/processed",
                         help="Where to write train.csv / test.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(in_path)
    label_col = find_label_column(df)
    feature_cols = select_features(df, label_col)

    keep_cols = feature_cols + [label_col]
    df = df[keep_cols]

    print(f"\nClass distribution ({label_col}):")
    print(df[label_col].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[label_col],
    )

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nWrote {len(train_df):,} rows -> {train_path}")
    print(f"Wrote {len(test_df):,} rows  -> {test_path}")


if __name__ == "__main__":
    main()