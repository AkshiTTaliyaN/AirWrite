"""Shared helper functions."""

from __future__ import annotations

import time


def landmark_to_pixel(lm, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Convert normalized MediaPipe landmark to pixel coordinates."""
    return int(lm.x * frame_w), int(lm.y * frame_h)


def distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def now_ms() -> float:
    return time.time() * 1000.0


def mirror_x(x: int, frame_w: int) -> int:
    """Mirror x for natural webcam interaction."""
    return frame_w - x
