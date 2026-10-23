# Results folders

- `honest/` - **use these.** Evaluated on recordings the model never saw
  (per-recording cross-validation), after removing background-service flows and
  flows under 10 packets. Regenerate with
  `python -m qos_ml.report data/processed/features.csv --output-dir results/honest`.
- Files directly in `results/` (`metrics_summary.csv`, `classification_report.txt`,
  `confusion_matrix_*.png`, `prediction.csv`) are **legacy**. They came from the old
  `ML/` scripts with a random split over flows, so recordings leaked between train
  and test and the scores (about 96% accuracy) are optimistic. Do not quote them.
