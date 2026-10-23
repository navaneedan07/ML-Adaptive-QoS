"""Show the closed loop adapting when traffic changes (synthetic traffic).

1. Train on "regime A" traffic and measure accuracy on fresh regime A flows.
2. Traffic shifts to "regime B" (different packet sizes and timing): accuracy drops.
3. Labelled feedback with each flow's features is sent to the controller. The
   drift detector fires, the controller retrains on dataset + feedback, checks
   the new model on held-out data and swaps it in.
4. Accuracy on fresh regime B flows recovers.

Synthetic data only demonstrates the mechanism; it says nothing about real-world
accuracy. Usage: python -m demo.drift_demo [--drift 0.8] [--feedback 150]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from qos_ml.controller import LiveController
from qos_ml.drift import AdaptiveDriftDetector
from qos_ml.train import train_models
from traffic.feature_extractor import extract_features
from traffic.synthetic import generate_flows, write_features_csv


def accuracy(controller: LiveController, rows: list[dict]) -> float:
    hits = sum(controller.classify(row)["predicted_label"] == row["label"] for row in rows)
    return hits / len(rows)


def run(drift: float = 0.8, feedback: int = 150, workdir: Path | None = None) -> dict:
    work = Path(workdir or tempfile.mkdtemp(prefix="drift_demo_"))
    regime_a = generate_flows(5, 30, seed=0)
    dataset = work / "regime_a.csv"
    write_features_csv(regime_a, dataset)
    train_models(dataset, work, model_names=["random_forest"])

    controller = LiveController(
        work / "random_forest.joblib",
        dataset_path=dataset,
        retrain_output=work,
        min_confidence=0.0,  # measure raw accuracy, not the confidence fallback
        drift=AdaptiveDriftDetector(window_size=100, threshold=0.2),
    )
    fresh_a = [extract_features(f) for f in generate_flows(2, 15, seed=1)]
    drifted_train = [extract_features(f) for f in generate_flows(3, 20, seed=2, drift=drift)]
    fresh_b = [extract_features(f) for f in generate_flows(2, 15, seed=3, drift=drift)]

    result = {"regime_a": accuracy(controller, fresh_a), "regime_b_before": accuracy(controller, fresh_b)}
    drift_detected = False
    for row in (drifted_train * 3)[:feedback]:
        predicted = controller.classify(row)["predicted_label"]
        drift_detected |= controller.feedback(predicted, row["label"], row)
    for _ in range(600):  # retraining runs in a background thread
        if controller.last_retrain is not None and not controller.retraining:
            break
        time.sleep(0.1)
    result.update(
        drift_detected=drift_detected,
        last_retrain=controller.last_retrain,
        regime_b_after=accuracy(controller, fresh_b),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift", type=float, default=0.8)
    parser.add_argument("--feedback", type=int, default=150)
    args = parser.parse_args()
    result = run(args.drift, args.feedback)
    print(f"accuracy on original traffic (regime A):    {result['regime_a']:.0%}")
    print(f"accuracy after traffic shifts (regime B):   {result['regime_b_before']:.0%}")
    print(f"drift detected from feedback:               {result['drift_detected']}")
    retrain = result["last_retrain"] or {}
    print(f"retrained model promoted:                   {retrain.get('promoted')}")
    print(f"accuracy on regime B after retraining:      {result['regime_b_after']:.0%}")
    print("(synthetic traffic: demonstrates the adaptation mechanism, not real-world accuracy)")


if __name__ == "__main__":
    main()
