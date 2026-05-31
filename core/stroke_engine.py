"""Continuous stroke buffering and virtual canvas rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from utils.helpers import now_ms


@dataclass
class StrokePoint:
    x: int
    y: int
    t: float


@dataclass
class StrokeEngine:
    """Store temporal stroke history and render on canvas."""

    canvas_size: tuple[int, int] = (400, 400)
    stroke_width: int = 14
    points: list[StrokePoint] = field(default_factory=list)
    pen_down: bool = False
    _canvas: np.ndarray | None = None
    _frame_w: int = 640
    _frame_h: int = 480

    def reset_buffer(self) -> None:
        self.points.clear()
        self.pen_down = False
        self._init_canvas()

    def _init_canvas(self) -> None:
        w, h = self.canvas_size
        self._canvas = np.zeros((h, w), dtype=np.uint8)

    def set_canvas_mapping(self, frame_w: int, frame_h: int) -> None:
        self._frame_w = frame_w
        self._frame_h = frame_h

    def map_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        """Map frame pixel coordinates to canvas coordinates."""
        w, h = self.canvas_size
        cx = int(x / self._frame_w * w)
        cy = int(y / self._frame_h * h)
        return min(w - 1, max(0, cx)), min(h - 1, max(0, cy))

    def add_point_mapped(self, x: int, y: int) -> None:
        """Add a point (in frame space) and draw onto canvas."""
        if not self.pen_down:
            return
        self.points.append(StrokePoint(x, y, now_ms()))
        if self._canvas is None:
            self._init_canvas()
        cx, cy = self.map_to_canvas(x, y)
        if len(self.points) >= 2:
            prev = self.points[-2]
            px, py = self.map_to_canvas(prev.x, prev.y)
            cv2.line(self._canvas, (px, py), (cx, cy), 255, self.stroke_width)
        else:
            cv2.circle(self._canvas, (cx, cy), self.stroke_width // 2, 255, -1)

    def end_stroke(self) -> None:
        self.pen_down = False

    def get_canvas_image(self) -> np.ndarray:
        if self._canvas is None:
            self._init_canvas()
        return self._canvas.copy()

    def render_overlay(self, frame: np.ndarray, x: int, y: int) -> None:
        """Draw live fingertip dot and trail on webcam frame."""
        color = (0, 255, 255) if self.pen_down else (100, 100, 100)
        cv2.circle(frame, (x, y), 8, color, 2)
        if len(self.points) >= 2:
            start = max(1, len(self.points) - 29)  # last 30 points only
            for i in range(start, len(self.points)):
                p0 = (self.points[i - 1].x, self.points[i - 1].y)
                p1 = (self.points[i].x, self.points[i].y)
                cv2.line(frame, p0, p1, (255, 200, 0), 3)

    def render_canvas_preview(self, frame: np.ndarray, origin: tuple[int, int] = (20, 140)) -> None:
        """Embed small canvas preview on HUD area."""
        if self._canvas is None:
            return
        preview = cv2.cvtColor(self._canvas, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(preview, (180, 180))
        ox, oy = origin
        h, w = preview.shape[:2]
        fh, fw = frame.shape[:2]
        if oy + h < fh and ox + w < fw:
            frame[oy: oy + h, ox: ox + w] = preview