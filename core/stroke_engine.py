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

    def reset_buffer(self) -> None:
        self.points.clear()
        self.pen_down = False
        self._init_canvas()

    def _init_canvas(self) -> None:
        w, h = self.canvas_size
        self._canvas = np.zeros((h, w), dtype=np.uint8)

    def start_stroke(self, x: int, y: int) -> None:
        self.pen_down = True
        self.points.append(StrokePoint(x, y, now_ms()))

    def add_point(self, x: int, y: int) -> None:
        if not self.pen_down:
            return
        self.points.append(StrokePoint(x, y, now_ms()))
        if self._canvas is None:
            self._init_canvas()
        if len(self.points) >= 2:
            p1 = self.points[-2]
            p2 = self.points[-1]
            cv2.line(
                self._canvas,
                self._to_canvas(p1.x, p1.y),
                self._to_canvas(p2.x, p2.y),
                255,
                self.stroke_width,
            )

    def end_stroke(self) -> None:
        self.pen_down = False

    def _to_canvas(self, x: int, y: int) -> tuple[int, int]:
        """Map frame coordinates into fixed canvas."""
        w, h = self.canvas_size
        # Normalized mapping assumes points already in frame space; scale to canvas
        cx = int(x * w / max(1, w))
        cy = int(y * h / max(1, h))
        return min(w - 1, max(0, cx)), min(h - 1, max(0, cy))

    def set_canvas_mapping(self, frame_w: int, frame_h: int) -> None:
        self._frame_w = frame_w
        self._frame_h = frame_h

    def map_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        w, h = self.canvas_size
        fw = getattr(self, "_frame_w", w)
        fh = getattr(self, "_frame_h", h)
        cx = int(x / fw * w)
        cy = int(y / fh * h)
        return min(w - 1, max(0, cx)), min(h - 1, max(0, cy))

    def render_point(self, x: int, y: int) -> None:
        if self._canvas is None:
            self._init_canvas()
        cx, cy = self.map_to_canvas(x, y)
        if len(self.points) >= 2:
            prev = self.points[-2]
            px, py = self.map_to_canvas(prev.x, prev.y)
            cv2.line(self._canvas, (px, py), (cx, cy), 255, self.stroke_width)
        else:
            cv2.circle(self._canvas, (cx, cy), self.stroke_width // 2, 255, -1)

    def add_point_mapped(self, x: int, y: int) -> None:
        if not self.pen_down:
            return
        self.points.append(StrokePoint(x, y, now_ms()))
        self.render_point(x, y)

    def get_canvas_image(self) -> np.ndarray:
        if self._canvas is None:
            self._init_canvas()
        return self._canvas.copy()

    def get_bounding_regions(self) -> list[tuple[int, int, int, int]]:
        """Return list of (x1,y1,x2,y2) in canvas space for each stroke segment."""
        if not self.points:
            return []
        coords = [self.map_to_canvas(p.x, p.y) for p in self.points]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        pad = 8
        w, h = self.canvas_size
        return [
            (
                max(0, min(xs) - pad),
                max(0, min(ys) - pad),
                min(w, max(xs) + pad),
                min(h, max(ys) + pad),
            )
        ]

    def render_overlay(self, frame: np.ndarray, x: int, y: int) -> None:
        """Draw live fingertip trail on webcam frame."""
        color = (0, 255, 255) if self.pen_down else (100, 100, 100)
        cv2.circle(frame, (x, y), 8, color, 2)
        if len(self.points) >= 2:
            for i in range(1, len(self.points)):
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
            frame[oy : oy + h, ox : ox + w] = preview
