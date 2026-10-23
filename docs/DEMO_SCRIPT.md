# Demo script

Two terminals, from the repository root. Python 3.11+ and `pip install -r requirements.txt`.

## 0. One-time setup
```bash
python -m qos_ml.train data/processed/features.csv --output-dir models --cv-folds 4
```
(The committed `models/` already hold models trained this way.)

## 1. Dashboard + classification (about 2 min)
Terminal 1:
```bash
python -m api.server models/random_forest.joblib --dataset data/processed/features.csv --output-dir models --port 8080
```
Open http://127.0.0.1:8080/dashboard.

Terminal 2:
```bash
python -m demo.run_demo
```
Say: each line is one real flow. It prints the predicted class, how confident the
model is, and the DSCP tag + queue the policy picks. Point out that a flow below
70% confidence is **not** prioritised ("low confidence -> best-effort"): a wrong
priority is worse than none. Then the 100 wrong-label feedback events push the
error rate up on the dashboard and trigger drift detection and a retrain.

## 2. Adapting to changed traffic (about 2 min, synthetic)
```bash
python -m demo.drift_demo --drift 0.8
```
Say: synthetic traffic, so it shows the mechanism, not real accuracy. Traffic
shifts (accuracy drops 100% to about 58%), feedback triggers drift detection,
the model retrains and is checked on old and recent traffic before it is swapped
in, and accuracy recovers (to about 84%). Try `--drift 0.5` for a milder shift.

## 3. Honest results (about 1 min)
Show `results/honest/confusion_matrix.png` and `results/honest/confidence_curve.png`.
Regenerate with:
```bash
python -m qos_ml.report data/processed/features.csv --output-dir results/honest
```

## 4. Linux proof (optional, pre-record it)
On WSL/Linux, after `network/setup_network.sh`:
```bash
cd network && DURATION=30 ./run_benchmark.sh results/benchmark   # baseline vs QoS summary.csv
```
These scripts were written on macOS and syntax-checked only. Run them once before
presenting and record the output; do not demo them cold. For the live controller:
`sudo ip netns exec ns1 python -m qos_ml.controller models/random_forest.joblib --dataset data/processed/features.csv --live --interface veth1 --namespace ns1 --in-namespace`
(dry-run unless `--enforce`).

## If something fails
- Port busy: the server picks the next free port and prints it; pass it to `--base-url`.
- `ModuleNotFoundError: scapy`: `pip install -r requirements.txt` (only needed for PCAP/live capture).
- Dashboard empty: run `python -m demo.run_demo` first so there are events.
