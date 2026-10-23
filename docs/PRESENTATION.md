# Presentation outline (about 10 minutes)

Honest numbers are used throughout. Do not quote the legacy 96% accuracy in
`results/` (random split, leaked recordings).

## 1. Problem (1 min)
- On a shared link a big download ruins a video call ("best effort" treats all packets alike).
- Traffic is encrypted, so the network cannot read the payload to tell them apart.

## 2. Idea (1 min)
- Classify by **behaviour**: packet sizes, timing, bursts, protocol.
- Map the class to a priority (DSCP tag + queue), enforce it with Linux `tc`, and close the loop: measure, detect drift, retrain.

## 3. Architecture (2 min)
`packets -> flows -> features -> model -> policy (DSCP + queue) -> tc/iptables -> RTT + drop monitor -> drift detector -> retrain -> back to the model`
- voip/chat -> queue 1:10 (fast); streaming/email -> 1:20; p2p/file transfer/unknown -> 1:30 (default).
- Stack: Python, scikit-learn (Random Forest, SVM, gradient boosting), Scapy, Linux HTB + fq_codel, iptables DSCP, small HTTP API + dashboard.

## 4. Demo (3 min) - see `docs/DEMO_SCRIPT.md`
Live classification with confidence, the dashboard, then the drift-and-recovery demo.

## 5. Results - what we measured honestly (2 min)
- Data: 836 usable flows from 23 recordings, 4 classes (voip, chat, streaming, email). p2p and file transfer have no data yet.
- Per-recording cross-validation (every flow predicted by a model that never saw its recording): **class macro F1 about 0.56, queue accuracy about 71%** (`results/honest/`).
- Versus baselines: port-based classification gets macro F1 0.29 / queue accuracy 42%; best ML model 0.56 / 71%.
- Latency budget: the compact decision tree classifies a flow in about 0.8 ms versus about 8 ms for boosting (this laptop; hardware-dependent) at macro F1 0.48 vs 0.56. The controller can fall back to it automatically.
- Confidence is meaningful: at confidence >= 0.9 the queue is right about 96% of the time on 25% of flows; at >= 0.7, about 77% on 57%. We prioritise only what the model is fairly sure about (default threshold 0.7).
- Email is the strongest class (F1 about 0.83); voip vs streaming is the main confusion.
- What we found and fixed: the original 96% was a leak (about 3% on unseen recordings), because most "voip" flows were 2-3 packet DNS/LLMNR chatter labelled from the filename. We now drop flows under 10 packets and flows on background-service ports.
- Things we tried that did not help (data is the limit, not the model): derived features, recording weights, model voting, a two-stage queue model.
- Adaptation (synthetic traffic, mechanism only): accuracy 58% after a traffic shift, back to 84% after feedback-driven retraining.

## 6. Limits and next steps (1 min)
- Review 1 reported 0.96 accuracy; that came from a split that leaked recordings. Say so up front (see `docs/REVIEW2_NOTES.md`).
- Review 1's Table III raw network files are not in the repo; re-run the benchmark or commit them.
- Small, partly noisy dataset; labels come from filenames. More raw captures is the biggest lever (ISCXVPN2016 download pending).
- Early classification (decide after the first N packets) is built and unit-tested but needs a model trained on raw-PCAP early features and a run on Linux.
- The fair baseline-vs-QoS benchmark (`network/run_benchmark.sh`) is written but not yet run; earlier ping logs were not like-for-like (see `network/README.md`).
- p2p and file transfer classes need data.

## Likely questions
- *Why not 96%?* That number came from a split that leaked recordings; we replaced it with an honest evaluation.
- *Why does low confidence go best-effort?* A wrong priority hurts real calls; no priority is harmless.
- *Does it work on encrypted traffic?* Yes, it never reads payloads, only sizes and timing.
- *How does it adapt?* Feedback with the flow's features is stored, recent feedback outweighs old data, and the retrained model replaces the current one only if it is not worse on old and recent traffic.
