# ML Adaptive QoS

An encryption-safe traffic classification and QoS benchmark pipeline.

## Components

1. `traffic/` parses PCAP/PCAPNG files, builds bidirectional flows, extracts
   packet-size/IAT/burst features, and balances classes.
2. `ml/` trains reproducible Random Forest and SVM baselines, evaluates them,
   predicts new flows, detects error-rate drift, and exposes class-aware QoS
   policy mappings.
3. `network/` creates an optional Linux namespace testbed, applies HTB plus
   `fq_codel`, measures latency/iperf3 results, and compares benchmark runs.
4. `api/` exposes a local health/status dashboard and feedback endpoint.

## Quick start

```bash
python -m pip install -r requirements.txt
python traffic/feature_extractor.py data/raw data/processed/features.csv
python traffic/balance_classes.py data/processed/features.csv data/processed/features_balanced.csv
python -m qos_ml.train data/processed/features_balanced.csv --output-dir models
python -m qos_ml.predict models/random_forest.joblib data/processed/features.csv data/processed/predictions.csv
python -m pytest
```

## Live closed-loop mode

Train a model once, then start the controller and its local dashboard:

```bash
python -m qos_ml.train data/processed/features_balanced.csv --output-dir models
python -m api.server models/random_forest.joblib --dataset data/processed/features_balanced.csv --output-dir models --port 8080 --monitor-interval 10
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
  --dataset data/processed/features_balanced.csv \
  --live --interface veth1 --namespace ns1 --in-namespace \
  --monitor-interval 10
```

This defaults to **dry-run enforcement**. Add `--enforce` only on a
privileged Linux host after `setup_network.sh` and `setup_qos.sh`; predictions
then drive both `tc` class assignment and idempotent iptables DSCP marking.
The monitor samples RTT and `tc` dropped-packet counters in the background
when `--monitor-interval` is enabled. The complete path has been validated in
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
python -m qos_ml.train data\processed\features_balanced.csv --output-dir models
python -m api.server models\random_forest.joblib --dataset data\processed\features_balanced.csv --output-dir models --port 8080
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
