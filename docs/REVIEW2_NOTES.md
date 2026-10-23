# Notes for the next review: what changed since Review 1

## 1. The headline numbers in Review 1 do not survive an honest evaluation
Review 1 reports Random Forest 0.96 accuracy / macro F1 0.66 on 11,852 test flows.
That evaluation split randomly over **flows**, so flows from the same recording were
in both train and test, and about 97% of the "voip" flows were 2-3 packet
DNS/LLMNR/mDNS chatter labelled from the filename. On recordings the model had never
seen, accuracy was about 3%.

Honest evaluation (per-recording cross-validation, flows of at least 10 packets,
background-service ports excluded; 836 flows, 23 recordings, 4 classes):

| Method | Class macro F1 | Class accuracy | Queue accuracy |
|---|---|---|---|
| No classification (all best-effort) | 0.00 | 0.00 | 0.00 |
| Port-based | 0.29 | 0.41 | 0.42 |
| Random Forest | 0.54 | 0.58 | 0.65 |
| SVM | 0.42 | 0.51 | 0.70 |
| Decision tree (compact) | 0.48 | 0.49 | 0.68 |
| Gradient boosting | **0.56** | 0.58 | **0.71** |

(`results/honest/`, regenerate with `python -m qos_ml.report data/processed/features.csv`.)
Say plainly in the review that the earlier figure was inflated by a split that leaked
recordings, and that the corrected, lower figure is the one to trust. The conclusion
"classification is limited by class imbalance" becomes "limited by label quality and by
the small number of recordings per class".

## 2. Table III (network results) is not backed by the files in this repository
Recomputed from the committed iperf and ping files, they differ from Table III (see
`network/README.md`). The HTB + fq_codel run has no iperf files committed at all.
Before submitting: commit the raw files behind Table III, or run
`network/run_benchmark.sh` on WSL and regenerate the table.

## 3. Status of the Review 1 contributions

| Contribution / plan item | Status |
|---|---|
| Closed-loop enforcement (classifier -> DSCP + tc/HTB) | Implemented (`qos_ml/controller.py`, `network/enforcer.py`); dry-run by default. The README says it was validated in WSL; not re-verified here. |
| Joint accuracy + QoS evaluation | Classifier side done. Network side: benchmark script written (`run_benchmark.sh`), **never run**; see item 2. |
| AQM-aware analysis (HTB vs HTB + fq_codel) | Scripts exist; raw files for the fq_codel run missing. |
| Latency-budgeted model selection | Implemented: per-flow inference time, cost function, controller fallback to a compact decision tree. **1D-CNN not implemented.** |
| Drift-aware retraining (lambda = 1 - F1) | Implemented, plus RTT and queue-drop signals. Recovery shown on **synthetic** traffic only. |
| Four setups: no QoS / port-DPI / classical ML / closed loop | Classifier-level comparison done (table above). Network-level comparison for the port/DPI setup not done. |
| Early (first-N-packets) classification | Implemented and unit-tested; needs raw PCAPs to train and a Linux run. |
| Raw dataset (ISCXVPN2016) | Download not completed. Only 4 of the 6 policy classes have data (no p2p, file transfer). |

## 4. Not done / open
- Raw PCAPs: more recordings per class is the biggest lever for accuracy.
- Running the Linux scripts and recording output.
- 1D-CNN comparison (needs a deep-learning dependency).
- Review 1's "Code and Reproducibility" section lists `classifiers/ enforcement/
  benchmark/` folders; the actual layout is `qos_ml/ traffic/ network/ api/ demo/`.
  Update the paper or rename.
