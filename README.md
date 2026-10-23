# ML Adaptive QoS

An encryption-safe traffic classification and QoS benchmark pipeline.

## Components

1. `traffic/` parses PCAP/PCAPNG files, builds bidirectional flows, extracts
   packet-size/IAT/burst features, and balances classes.
2. `qos_ml/` trains reproducible Random Forest and SVM baselines, predicts
   encrypted-traffic-safe flow requirements, and exposes class-aware QoS
   policy mappings.
3. `network/` creates an optional Linux namespace testbed, applies HTB plus
   `fq_codel`, measures latency/iperf3 results, and compares benchmark runs.
4. `api/` exposes a local health/status dashboard and feedback endpoint.

The runtime is one closed-loop system rather than separate scripts:

```text
packet capture
    -> flow features
    -> requirement/classification model
    -> QoS policy (queue + DSCP)
    -> tc/iptables enforcement
    -> RTT and queue metrics
    -> parent adaptive drift model
    -> retraining and policy adaptation
```

For example, a user generating sustained video traffic is classified as
`streaming`, mapped to DSCP 26 and queue `1:20`, and monitored continuously.
The parent drift model combines classification errors with RTT spikes and new
queue drops, so adaptation can begin even when ground-truth labels are not
available.

## Quick start

```bash
python -m pip install -r requirements.txt
# for the exact versions behind the paper's numbers: python -m pip install -r requirements-lock.txt
python traffic/feature_extractor.py data/raw data/processed/features.csv
python traffic/balance_classes.py data/processed/features.csv data/processed/features_balanced.csv   # optional: no longer needed, training handles imbalance itself
python -m qos_ml.train data/processed/features.csv --output-dir models --eval-dataset data/processed/features.csv --cv-folds 4
python -m qos_ml.predict models/random_forest.joblib data/processed/features.csv data/processed/predictions.csv
python -m pytest
```

## Honest evaluation

`qos_ml.train` splits by recording (`source_file`), so a capture is never in both
train and test, and ignores flows under 10 packets (`--min-packets`) and flows on
background-service ports such as DNS/LLMNR/mDNS (`--keep-noise-ports` to keep
them): that chatter only inherits its label from the filename.
Judge models by **macro F1**, not accuracy. `--eval-dataset` adds a test on the
natural class mix, `--cv-folds` a cross-validated per-recording score, and
`--models random_forest svm hist_gb` picks the models.

Slide-ready tables and charts (per-class, per-queue, confusion matrix, confidence
curve and a recommended `--min-confidence`) come from per-recording
cross-validation:

```bash
python -m qos_ml.report data/processed/features.csv --output-dir results/honest
```

To measure how accuracy grows with the number of packets seen (needs raw PCAPs):

```bash
python -m traffic.early_packet_study data/raw results/early_packet_accuracy.csv
```

At runtime, `--min-confidence` (default 0.7) sends low-confidence predictions to
the default best-effort queue, `--early-packets N` classifies a flow after N
packets (use a model trained on the same N), feedback posted with the flow's
features (`{"predicted":..,"actual":..,"flow":{..}}`) is added to the training
data, and a retrained model replaces the active one only if it is not worse on
held-out data (the old one is kept as `<model>.prev.joblib`).

## Baselines and latency-budgeted model selection

`python -m qos_ml.report` also compares the ML models with two baselines (no
classification, and classic port-based classification) and scores each model with
`cost = alpha * (1 - macro_F1) + beta * inference_ms`, written to
`results/honest/baseline_comparison.csv` and `model_selection.csv`. Models:
`random_forest`, `svm`, `hist_gb` and a compact `decision_tree` (the lightweight
option). At runtime the controller takes `--latency-budget-ms` and
`--fallback-model models/decision_tree.joblib`: if classifying one flow exceeds the
budget it switches to the lightweight model. Drift detection uses the paper's
misclassification rate lambda = 1 - F1 over a sliding window of feedback.
A 1D-CNN is not implemented.

## Synthetic traffic (testing and demos only)

`traffic/synthetic.py` simulates the six classes so the pipeline can be tested
without the raw dataset. It is **not real data**; accuracy measured on it says
nothing about real traffic.

```bash
python -m traffic.synthetic data/synthetic/features.csv --pcap-dir data/synthetic/pcaps
python -m traffic.early_packet_study data/synthetic/pcaps results/synthetic_early.csv
python -m demo.drift_demo --drift 0.8
```

`demo.drift_demo` shows the adaptive loop: traffic shifts, accuracy drops,
feedback triggers drift detection and retraining, and accuracy recovers. A
retrained model is judged on old and recent traffic equally, and recent feedback
rows are repeated so they outweigh the older dataset.

## Live closed-loop mode

Train a model once, then start the controller and its local dashboard:

```bash
python -m qos_ml.train data/processed/features.csv --output-dir models
python -m api.server models/random_forest.joblib --dataset data/processed/features.csv --output-dir models --port 8080 --monitor-interval 10
```

If port 8080 is already in use, the server automatically tries the next
available ports and prints the actual URL. Use that printed URL for the
dashboard and demo. The API provides `/health`,
`/api/status`, `/api/classify`, and `/dashboard`. The controller records predictions,
classifies flows, and automatically starts retraining when feedback reaches
the configured drift window:

```bash
curl -X POST http://127.0.0.1:8080/api/feedback ^
  -H "Content-Type: application/json" ^
  -d "{\"predicted\":\"voip\",\"actual\":\"chat\"}"
```

For live packet capture on Linux/WSL, run the controller inside the client
namespace. `--in-namespace` prevents nested `ip netns exec` calls:

```bash
sudo ip netns exec ns1 python -m qos_ml.controller models/random_forest.joblib \
  --dataset data/processed/features.csv \
  --live --interface veth1 --namespace ns1 --in-namespace \
  --monitor-interval 10
```

This defaults to **dry-run enforcement**. Add `--enforce` only on a
privileged Linux host after `setup_network.sh` and `setup_qos.sh`; predictions
then drive both `tc` class assignment and idempotent iptables DSCP marking.
The monitor samples RTT and `tc` dropped-packet counters in the background
when `--monitor-interval` is enabled. The parent drift model feeds those
measurements back into the controller and can trigger retraining. The complete path has been validated in
WSL2 with real namespaces, ping, iperf3, HTB/fq_codel, `tc` counters, iptables
DSCP marking, live Scapy packet capture, prediction, and drift-triggered
retraining. Native Windows supports the API, dry-run controller, offline
prediction, and tests; raw capture and Linux enforcement require WSL or Linux.

The bundled processed CSV files allow model training without downloading PCAPs.
Rows labeled `unknown` are excluded from training and reported by the feature
pipeline; extend `LABEL_KEYWORDS` when adding a new capture source.

The Linux network benchmark is documented in
[`network/README.md`](network/README.md).

## Recordable project demonstration

For a simple recording on Windows, use two terminals. In terminal 1, start the
API with a trained model and a dataset so retraining is enabled:

```powershell
python -m qos_ml.train data\processed\features.csv --output-dir models
python -m api.server models\random_forest.joblib --dataset data\processed\features.csv --output-dir models --port 8080
```

Open `http://127.0.0.1:8080/dashboard` in the browser. In terminal 2, run:

```powershell
python -m demo.run_demo
```

The terminal prints each detected category, predicted category, DSCP value,
and selected queue. The dashboard shows the event count, error rate, latest
prediction, and retraining state. Capture both the terminal output and
dashboard. For the final Linux proof, separately record the WSL command from
the live closed-loop section together with `ping`, `iperf3`, `tc -s qdisc`,
and the DSCP rule output.
