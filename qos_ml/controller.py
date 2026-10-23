"""Closed-loop live classification, enforcement, monitoring, and retraining."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import f1_score

from qos_ml.drift import AdaptiveDriftDetector, ErrorRateDriftDetector
from qos_ml.features import FEATURE_COLUMNS, is_noise_flow, rows_to_features
from qos_ml.qos_policy import CLASS_POLICY
from qos_ml.train import MODEL_NAMES, load_holdout_flags, train_models
from network.enforcer import TcEnforcer
from network.monitor import QoSMonitor


FEEDBACK_COLUMNS = [*FEATURE_COLUMNS, "src_ip", "src_port", "dst_ip", "dst_port", "source_file", "label"]


class LiveController:
    """Coordinate prediction, QoS enforcement, feedback, and retraining.

    Predictions below ``min_confidence`` are treated as ``unknown`` and get the
    default best-effort policy instead of a possibly wrong priority.
    Feedback that includes the flow's features is saved and added to the
    training data on the next retrain. A retrained model replaces the active
    one only if it is not noticeably worse on held-out data.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        dataset_path: str | Path | None = None,
        retrain_output: str | Path | None = None,
        enforcer: TcEnforcer | None = None,
        monitor: QoSMonitor | None = None,
        drift: ErrorRateDriftDetector | None = None,
        min_confidence: float = 0.7,
        max_regression: float = 0.02,
        feedback_repeat: int = 5,
        latency_budget_ms: float | None = None,
        fallback_model: str | Path | None = None,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.retrain_output = Path(retrain_output or self.model_path.parent)
        self.enforcer = enforcer or TcEnforcer()
        self.monitor = monitor or QoSMonitor()
        self.drift = drift or AdaptiveDriftDetector(metric="f1")
        self.min_confidence = min_confidence
        self.max_regression = max_regression
        self.feedback_repeat = max(1, feedback_repeat)
        self._feedback_count = 0
        self.feedback_path = self.retrain_output / "feedback.csv"
        if self.feedback_path.exists():
            with self.feedback_path.open(encoding="utf-8") as file:
                self._feedback_count = max(0, sum(1 for _ in file) - 1)
        self.model = joblib.load(self.model_path)
        self.latency_budget_ms = latency_budget_ms
        self.fallback = joblib.load(fallback_model) if fallback_model else None
        self.inference_ms: float | None = None  # moving average of the active model
        self.using_fallback = False
        self.last_retrain: dict[str, Any] | None = None
        self.lock = threading.RLock()
        self.events: list[dict[str, Any]] = []
        self.retraining = False
        self.retrain_error: str | None = None

    @staticmethod
    def _predict(model: Any, frame: pd.DataFrame) -> tuple[str, float | None]:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)[0]
            index = int(probabilities.argmax())
            return str(model.classes_[index]), float(probabilities[index])
        return str(model.predict(frame)[0]), None

    def classify(self, features: dict[str, Any]) -> dict[str, Any]:
        if is_noise_flow(features):
            # DNS/DHCP/mDNS/... chatter: not classifiable, stay best-effort.
            predicted, confidence, label = "unknown", None, "unknown"
        else:
            row = rows_to_features([features])[0]
            with self.lock:
                model = self.fallback if self.using_fallback else self.model
            started = time.perf_counter()
            predicted, confidence = self._predict(model, pd.DataFrame([row]))
            self._track_latency((time.perf_counter() - started) * 1000)
            label = predicted
            if confidence is not None and confidence < self.min_confidence:
                label = "unknown"
        enforcement = self.enforcer.apply(label, features)
        event = {
            "timestamp": time.time(),
            "label": label,
            "predicted_label": predicted,
            "confidence": confidence,
            "flow": features,
            "enforcement": enforcement,
        }
        with self.lock:
            self.events.append(event)
            del self.events[:-1000]
        return event

    def _track_latency(self, elapsed_ms: float) -> None:
        """Switch to the lightweight model if inference exceeds the latency budget."""
        with self.lock:
            self.inference_ms = elapsed_ms if self.inference_ms is None else 0.8 * self.inference_ms + 0.2 * elapsed_ms
            if (
                self.latency_budget_ms is not None
                and self.fallback is not None
                and not self.using_fallback
                and self.inference_ms > self.latency_budget_ms
            ):
                self.using_fallback = True
                self.inference_ms = None  # re-measure on the lightweight model

    def _known_labels(self) -> set[str]:
        with self.lock:
            return set(CLASS_POLICY) | {str(label) for label in getattr(self.model, "classes_", [])}

    def feedback(self, predicted: str, actual: str, flow: dict[str, Any] | None = None) -> bool:
        """Record a labelled outcome. ``flow`` (its features) lets it become training data."""
        if flow is not None and self.dataset_path and actual in self._known_labels():
            self._save_feedback(flow, actual)
        drifted = self.drift.update(predicted, actual)
        if drifted and self.dataset_path and not self.retraining:
            threading.Thread(target=self.retrain, daemon=True).start()
        return drifted

    def _save_feedback(self, flow: dict[str, Any], actual: str) -> None:
        try:
            rows_to_features([flow])
        except (KeyError, TypeError, ValueError):
            return  # not a usable feature row; ignore rather than poison the data
        record = {column: flow.get(column, "") for column in FEEDBACK_COLUMNS}
        record["label"] = actual
        with self.lock:
            # Buckets of 20 rows act as separate "recordings" for the by-recording split.
            record["source_file"] = f"live_{self._feedback_count // 20}"
            self._feedback_count += 1
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.feedback_path.exists()
            with self.feedback_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FEEDBACK_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(record)

    def _training_file(self) -> Path:
        """Dataset plus saved feedback rows, written to a combined CSV."""
        assert self.dataset_path is not None
        if not self.feedback_path.exists():
            return self.dataset_path
        with self.dataset_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or FEEDBACK_COLUMNS
            rows = list(reader)
        with self.feedback_path.open(newline="", encoding="utf-8") as file:
            # Recent feedback is repeated so it outweighs the older bulk dataset.
            rows.extend(list(csv.DictReader(file)) * self.feedback_repeat)
        combined = self.retrain_output / "_training_combined.csv"
        self.retrain_output.mkdir(parents=True, exist_ok=True)
        with combined.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return combined

    def retrain(self) -> dict[str, Any] | None:
        if not self.dataset_path:
            return None
        with self.lock:
            if self.retraining:
                return None
            self.retraining = True
            self.retrain_error = None
        try:
            name = self.model_path.stem if self.model_path.stem in MODEL_NAMES else "random_forest"
            training_file = self._training_file()
            candidate_dir = self.retrain_output / "candidate"
            summary = train_models(training_file, candidate_dir, model_names=[name])
            candidate = joblib.load(candidate_dir / f"{name}.joblib")
            x_test, y_test, recent = load_holdout_flags(training_file)
            with self.lock:
                current = self.model
            candidate_f1 = self._holdout_score(candidate, x_test, y_test, recent)
            current_f1 = self._holdout_score(current, x_test, y_test, recent)
            # The current model may have seen some held-out rows, so allow a small margin.
            promoted = candidate_f1 + self.max_regression >= current_f1
            if promoted:
                active = self.retrain_output / f"{name}.joblib"
                if active.exists():
                    shutil.copy2(active, self.retrain_output / f"{name}.prev.joblib")
                shutil.copy2(candidate_dir / f"{name}.joblib", active)
            with self.lock:
                if promoted:
                    self.model = candidate
                self.last_retrain = {
                    "promoted": promoted,
                    "candidate_macro_f1": candidate_f1,
                    "current_macro_f1": current_f1,
                    "feedback_rows_used": training_file != self.dataset_path,
                }
                self.drift.reset()
            return summary
        except (OSError, RuntimeError, ValueError) as exc:
            with self.lock:
                self.retrain_error = str(exc)
            return None
        finally:
            with self.lock:
                self.retraining = False

    @staticmethod
    def _holdout_score(model: Any, x_test: pd.DataFrame, y_test: list[str], recent: list[bool]) -> float:
        """Macro-F1 where old and recent (feedback) traffic count equally.

        Judging only on a mostly-old test set would reject a model that has
        adapted to new traffic, which is the whole point of retraining.
        """
        def score(mask: list[bool]) -> float | None:
            if sum(mask) < 5:
                return None
            idx = [i for i, keep in enumerate(mask) if keep]
            predicted = model.predict(x_test.iloc[idx])
            return f1_score([y_test[i] for i in idx], predicted, average="macro", zero_division=0)

        parts = [value for value in (score([not r for r in recent]), score(recent)) if value is not None]
        return sum(parts) / len(parts) if parts else 0.0

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model": str(self.model_path),
                "events": len(self.events),
                "error_rate": self.drift.error_rate,
                "retraining": self.retraining,
                "retrain_error": self.retrain_error,
                "last_retrain": self.last_retrain,
                "min_confidence": self.min_confidence,
                "inference_ms": self.inference_ms,
                "latency_budget_ms": self.latency_budget_ms,
                "using_fallback_model": self.using_fallback,
                "drift_reason": getattr(self.drift, "last_reason", None),
                "last_event": self.events[-1] if self.events else None,
                "metrics": self.monitor.history[-1] if self.monitor.history else None,
            }

    def run_monitor(self, interval: float = 10.0, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            metrics = self.monitor.sample()
            if hasattr(self.drift, "update_metrics"):
                metrics_drifted = self.drift.update_metrics(metrics)
                if metrics_drifted and self.dataset_path and not self.retraining:
                    threading.Thread(target=self.retrain, daemon=True).start()
            stop.wait(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interface", default="veth1")
    parser.add_argument("--namespace", default="ns1")
    parser.add_argument(
        "--in-namespace",
        action="store_true",
        help="run commands directly because the controller is already inside the namespace",
    )
    parser.add_argument("--live", action="store_true", help="capture packets with Scapy")
    parser.add_argument("--enforce", action="store_true", help="execute Linux tc/iptables commands")
    parser.add_argument("--monitor-interval", type=float, default=10.0)
    parser.add_argument(
        "--min-confidence", type=float, default=0.7,
        help="predictions below this confidence get the default best-effort policy",
    )
    parser.add_argument(
        "--early-packets", type=int, default=None,
        help="classify each flow after this many packets (use a model trained with the same value)",
    )
    parser.add_argument(
        "--latency-budget-ms", type=float, default=None,
        help="switch to --fallback-model if classifying one flow takes longer than this",
    )
    parser.add_argument("--fallback-model", type=Path, help="lightweight model, for example decision_tree.joblib")
    args = parser.parse_args()
    controller = LiveController(
        args.model,
        dataset_path=args.dataset,
        retrain_output=args.output_dir,
        min_confidence=args.min_confidence,
        latency_budget_ms=args.latency_budget_ms,
        fallback_model=args.fallback_model,
        enforcer=TcEnforcer(
            None if args.in_namespace else args.namespace,
            args.interface,
            dry_run=not args.enforce,
        ),
        monitor=QoSMonitor(None if args.in_namespace else args.namespace, args.interface),
    )
    print(json.dumps(controller.status(), indent=2))
    if args.live:
        monitor_stop = threading.Event()
        threading.Thread(
            target=controller.run_monitor,
            kwargs={"interval": args.monitor_interval, "stop": monitor_stop},
            daemon=True,
        ).start()
        from qos_ml.live_capture import sniff_forever
        try:
            sniff_forever(
                controller.classify, interface=args.interface, early_packets=args.early_packets
            )
        finally:
            monitor_stop.set()


if __name__ == "__main__":
    main()
