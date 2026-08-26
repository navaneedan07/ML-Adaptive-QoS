# ML-Adaptive-QoS

Traffic classification project for adaptive QoS experiments.

## Project layout

- `traffic/` — PCAP parsing, flow building, and feature extraction
- `ML/` — preprocessing, training, evaluation, and prediction scripts
- `data/processed/` — feature tables and train/test splits
- `models/` — trained model artifacts and schema files
- `results/` — evaluation outputs and prediction exports

## Workflow

1. Build `features.csv` from PCAP files using `traffic/feature_extractor.py`
2. Split and clean the data with `ML/preprocess.py`
3. Train baseline models with `ML/train.py`
4. Evaluate them with `ML/evaluate.py`
5. Generate the handoff file with `ML/predict.py`

## Requirements

Install the Python packages listed in `requirements.txt`.

## Notes

- Raw PCAP files are not included in the repo.
- The dataset currently uses flow-level features extracted from ISCX VPN-nonVPN captures.
- Generated artifacts in `models/` and `results/` are kept here as the current project snapshot.
