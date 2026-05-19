"""Heuristic segmentation for continuous air-written words."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from core.stroke_engine import StrokePoint


@dataclass
class CharacterSegment:
    points: list[StrokePoint]
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2 on canvas


class SegmentationEngine:
    """
    Estimate character boundaries using:
    velocity drops, pen lifts, stroke gaps, x-axis clustering, tiny pauses.
    """

    def __init__(self, canvas_size: tuple[int, int] = (400, 400)):
        self.canvas_size = canvas_size

    def segment(self, points: list[StrokePoint], pen_down: bool) -> list[CharacterSegment]:
        if len(points) < 3:
            if points:
                bbox = self._bbox(points)
                return [CharacterSegment(points=list(points), bbox=bbox)]
            return []

        boundaries = {0, len(points)}
        boundaries |= self._boundaries_from_pauses(points)
        boundaries |= self._boundaries_from_velocity(points)
        boundaries |= self._boundaries_from_gaps(points)
        boundaries |= self._boundaries_from_x_clusters(points)

        if not pen_down and len(points) > 1:
            boundaries.add(len(points))

        sorted_bounds = sorted(boundaries)
        segments: list[CharacterSegment] = []
        for i in range(len(sorted_bounds) - 1):
            start, end = sorted_bounds[i], sorted_bounds[i + 1]
            chunk = points[start:end]
            if len(chunk) < 2:
                continue
            segments.append(CharacterSegment(points=chunk, bbox=self._bbox(chunk)))
        return segments if segments else [CharacterSegment(points=list(points), bbox=self._bbox(points))]

    def _boundaries_from_pauses(self, points: list[StrokePoint]) -> set[int]:
        bounds: set[int] = set()
        for i in range(1, len(points)):
            dt = points[i].t - points[i - 1].t
            if dt > config.PAUSE_SEGMENT_MS:
                bounds.add(i)
        return bounds

    def _boundaries_from_velocity(self, points: list[StrokePoint]) -> set[int]:
        bounds: set[int] = set()
        velocities = []
        for i in range(1, len(points)):
            dt = max(1.0, points[i].t - points[i - 1].t)
            dist = ((points[i].x - points[i - 1].x) ** 2 + (points[i].y - points[i - 1].y) ** 2) ** 0.5
            velocities.append(dist / dt)
        if not velocities:
            return bounds
        avg_v = sum(velocities) / len(velocities)
        threshold = avg_v * config.VELOCITY_DROP_RATIO
        for i, v in enumerate(velocities, start=1):
            if v < threshold:
                bounds.add(i)
        return bounds

    def _boundaries_from_gaps(self, points: list[StrokePoint]) -> set[int]:
        bounds: set[int] = set()
        for i in range(1, len(points)):
            dist = ((points[i].x - points[i - 1].x) ** 2 + (points[i].y - points[i - 1].y) ** 2) ** 0.5
            if dist > config.STROKE_GAP_PX:
                bounds.add(i)
        return bounds

    def _boundaries_from_x_clusters(self, points: list[StrokePoint]) -> set[int]:
        """Split when x jumps between character clusters."""
        bounds: set[int] = set()
        for i in range(1, len(points)):
            if abs(points[i].x - points[i - 1].x) > config.X_CLUSTER_MIN_GAP:
                bounds.add(i)
        return bounds

    def _bbox(self, points: list[StrokePoint]) -> tuple[int, int, int, int]:
        w, h = self.canvas_size
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        pad = 10
        return (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(w, max(xs) + pad),
            min(h, max(ys) + pad),
        )

    def crop_character(self, canvas: np.ndarray, segment: CharacterSegment) -> np.ndarray:
        """Extract 28x28 normalized character image from canvas."""
        x1, y1, x2, y2 = segment.bbox
        crop = canvas[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((28, 28), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE))
        return crop
