"""Small, dependency-free drift detector for the closed-loop controller."""

from __future__ import annotations

from collections import deque


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

