"""Closed-loop live classification, enforcement, monitoring, and retraining."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from qos_ml.drift import ErrorRateDriftDetector
from qos_ml.features import rows_to_features
from qos_ml.train import train_models
from network.enforcer import TcEnforcer
from network.monitor import QoSMonitor


class LiveController:
    """Coordinate prediction, QoS enforcement, feedback, and retraining."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        dataset_path: str | Path | None = None,
        retrain_output: str | Path | None = None,
        enforcer: TcEnforcer | None = None,
        monitor: QoSMonitor | None = None,
        drift: ErrorRateDriftDetector | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.retrain_output = Path(retrain_output or self.model_path.parent)
        self.enforcer = enforcer or TcEnforcer()
        self.monitor = monitor or QoSMonitor()
        self.drift = drift or ErrorRateDriftDetector()
        self.model = joblib.load(self.model_path)
        self.lock = threading.RLock()
        self.events: list[dict[str, Any]] = []
        self.retraining = False
        self.retrain_error: str | None = None

    def classify(self, features: dict[str, Any]) -> dict[str, Any]:
        row = rows_to_features([features])[0]
        label = str(self.model.predict(pd.DataFrame([row]))[0])
        enforcement = self.enforcer.apply(label, features)
        event = {
            "timestamp": time.time(),
            "label": label,
            "flow": features,
            "enforcement": enforcement,
        }
        with self.lock:
            self.events.append(event)
            del self.events[:-1000]
        return event

    def feedback(self, predicted: str, actual: str) -> bool:
        drifted = self.drift.update(predicted, actual)
        if drifted and self.dataset_path and not self.retraining:
            threading.Thread(target=self.retrain, daemon=True).start()
        return drifted

    def retrain(self) -> dict[str, Any] | None:
        if not self.dataset_path:
            return None
        with self.lock:
            if self.retraining:
                return None
            self.retraining = True
            self.retrain_error = None
        try:
            summary = train_models(self.dataset_path, self.retrain_output)
            candidate = self.retrain_output / "random_forest.joblib"
            new_model = joblib.load(candidate)
            with self.lock:
                self.model = new_model
                self.drift.reset()
            return summary
        except (OSError, RuntimeError, ValueError) as exc:
            with self.lock:
                self.retrain_error = str(exc)
            return None
        finally:
            with self.lock:
                self.retraining = False

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model": str(self.model_path),
                "events": len(self.events),
                "error_rate": self.drift.error_rate,
                "retraining": self.retraining,
                "retrain_error": self.retrain_error,
                "last_event": self.events[-1] if self.events else None,
                "metrics": self.monitor.history[-1] if self.monitor.history else None,
            }

    def run_monitor(self, interval: float = 10.0, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            self.monitor.sample()
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
    args = parser.parse_args()
    controller = LiveController(
        args.model,
        dataset_path=args.dataset,
        retrain_output=args.output_dir,
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
            sniff_forever(controller.classify, interface=args.interface)
        finally:
            monitor_stop.set()


if __name__ == "__main__":
    main()
