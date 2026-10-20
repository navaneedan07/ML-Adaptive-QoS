"""Small, dependency-free drift detector for the closed-loop controller."""

from __future__ import annotations

from collections import deque
from typing import Any


class ErrorRateDriftDetector:
    """Trigger retraining when the recent error rate exceeds a threshold."""

    def __init__(self, window_size: int = 100, threshold: float = 0.2) -> None:
        if window_size < 1 or not 0 <= threshold <= 1:
            raise ValueError("window_size must be positive and threshold must be in [0, 1]")
        self.window_size = window_size
        self.threshold = threshold
        self._errors: deque[bool] = deque(maxlen=window_size)

    def update(self, predicted: str, actual: str) -> bool:
        self._errors.append(predicted != actual)
        return len(self._errors) == self.window_size and self.error_rate >= self.threshold

    @property
    def error_rate(self) -> float:
        return sum(self._errors) / len(self._errors) if self._errors else 0.0

    def reset(self) -> None:
        self._errors.clear()


class AdaptiveDriftDetector(ErrorRateDriftDetector):
    """Parent drift model combining label and network-quality signals."""

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.2,
        *,
        rtt_multiplier: float = 2.0,
        drop_delta: int = 1,
    ) -> None:
        super().__init__(window_size, threshold)
        if rtt_multiplier <= 1 or drop_delta < 1:
            raise ValueError("rtt_multiplier must exceed 1 and drop_delta must be positive")
        self.rtt_multiplier = rtt_multiplier
        self.drop_delta = drop_delta
        self.baseline_rtt: float | None = None
        self.last_dropped_packets = 0
        self.last_reason: str | None = None

    def update_metrics(self, metrics: dict[str, Any]) -> bool:
        """Learn a healthy baseline, then detect RTT spikes or new drops."""
        rtt = metrics.get("rtt_ms")
        dropped = metrics.get("dropped_packets")
        reasons: list[str] = []
        if isinstance(rtt, (int, float)) and rtt > 0:
            if self.baseline_rtt is None:
                self.baseline_rtt = float(rtt)
            elif float(rtt) >= self.baseline_rtt * self.rtt_multiplier:
                reasons.append("rtt_spike")
        if isinstance(dropped, (int, float)):
            dropped_value = int(dropped)
            if dropped_value - self.last_dropped_packets >= self.drop_delta:
                reasons.append("queue_drops")
            self.last_dropped_packets = max(self.last_dropped_packets, dropped_value)
        self.last_reason = ",".join(reasons) if reasons else None
        return bool(reasons)

    def reset(self) -> None:
        super().reset()
        self.baseline_rtt = None
        self.last_dropped_packets = 0
        self.last_reason = None
