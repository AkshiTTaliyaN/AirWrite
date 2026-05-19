"""Adaptive smoothing for cursor and landmark stability."""

from __future__ import annotations


class ExponentialSmoother:
    """Exponential moving average for 2D coordinates."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x = self.alpha * x + (1 - self.alpha) * self._x
            self._y = self.alpha * y + (1 - self.alpha) * self._y
        return self._x, self._y

    def reset(self) -> None:
        self._x = None
        self._y = None


class PredictionStabilizer:
    """Accept predictions only after N consistent frames."""

    def __init__(self, required_frames: int = 5):
        self.required_frames = required_frames
        self._candidate = ""
        self._count = 0
        self.stable = ""

    def update(self, prediction: str) -> str:
        if prediction == self._candidate:
            self._count += 1
        else:
            self._candidate = prediction
            self._count = 1

        if self._count >= self.required_frames:
            self.stable = prediction
        return self.stable

    def reset(self) -> None:
        self._candidate = ""
        self._count = 0
        self.stable = ""
