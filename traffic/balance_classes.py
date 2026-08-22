"""
balance_classes.py

Fixes class imbalance in features.csv (reported by Hitesh: voip flows
dominate the dataset, ~92% of all rows). Undersamples the majority
class(es) down to a target count so the ML model doesn't just learn
to predict "voip" every time.

Usage:
    python balance_classes.py ../data/processed/features.csv ../data/processed/features_balanced.csv

By default, caps every class at the count of the second-largest class
(so we don't throw away as much data as capping to the smallest class
would). Adjust --cap to set an exact number instead.
"""

import csv
import sys
import random
import argparse
import collections


def balance(input_csv, output_csv, cap=None, seed=42):
    random.seed(seed)

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    by_label = collections.defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    counts = {label: len(items) for label, items in by_label.items()}
    print("Original class counts:")
    for label, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    if cap is None:
        # Default: cap at the second-largest class so we don't discard
        # more data than necessary, but still knock down the dominant class.
        sorted_counts = sorted(counts.values(), reverse=True)
        cap = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]

    print(f"\nCapping every class at {cap} rows...")

    balanced_rows = []
    for label, items in by_label.items():
        if len(items) > cap:
            balanced_rows.extend(random.sample(items, cap))
        else:
            balanced_rows.extend(items)

    random.shuffle(balanced_rows)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(balanced_rows)

    new_counts = collections.Counter(r["label"] for r in balanced_rows)
    print("\nBalanced class counts:")
    for label, count in new_counts.most_common():
        print(f"  {label}: {count}")
    print(f"\nWrote {len(balanced_rows)} rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--cap", type=int, default=None,
                         help="Max rows per class. Default: size of the "
                              "second-largest class.")
    args = parser.parse_args()

    balance(args.input_csv, args.output_csv, cap=args.cap)
