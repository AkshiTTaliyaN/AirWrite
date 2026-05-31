"""HUD overlay with per-mode colour coding and mode-switch banner."""

from __future__ import annotations

import time

import cv2
import numpy as np

import config

# Per-mode colour scheme  (BGR)
_MODE_COLORS: dict[str, dict] = {
    config.MODE_CURSOR: {
        "panel":  (15, 25, 15),        # very dark green-tinted bg
        "text":   (0, 255, 160),       # bright teal
        "border": (0, 200, 120),
        "banner": (0, 255, 160),
    },
    config.MODE_WRITE: {
        "panel":  (15, 15, 35),        # very dark blue-tinted bg
        "text":   (60, 170, 255),      # warm amber-orange
        "border": (40, 130, 220),
        "banner": (60, 170, 255),
    },
}

BANNER_DURATION = 1.5   # seconds the big centered banner stays on screen


class HUD:
    """Draw mode, FPS, prediction preview, and status indicators."""

    def __init__(self):
        self.fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

    def tick_fps(self) -> None:
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

    # ── Main draw call ─────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        *,
        mode: str,
        prediction: str,
        stable_prediction: str,
        pen_down: bool,
        gesture_label: str = "",
        mode_switched_at: float = 0.0,   # timestamp of last mode switch (0 = never)
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        scheme = _MODE_COLORS.get(mode, _MODE_COLORS[config.MODE_CURSOR])

        # ── Semi-transparent top panel ────────────────────────────────────────
        overlay = frame.copy()
        panel_h = 115
        cv2.rectangle(overlay, (0, 0), (w, panel_h), scheme["panel"], -1)
        cv2.addWeighted(overlay, config.HUD_BG_ALPHA, frame, 1 - config.HUD_BG_ALPHA, 0, frame)

        # Coloured bottom border on the panel — makes mode instantly obvious
        cv2.rectangle(frame, (0, panel_h - 2), (w, panel_h), scheme["border"], -1)

        # ── HUD text lines ────────────────────────────────────────────────────
        lines = [
            f"Mode: {mode.upper()}",
            f"FPS:  {self.fps:.1f}",
            f"Gesture: {gesture_label or '-'}",
            f"Preview: {prediction or '-'}",
            f"Stable:  {stable_prediction or '-'}",
            f"Pen: {'DOWN' if pen_down else 'UP'}",
        ]
        y = 22
        for line in lines:
            cv2.putText(
                frame, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.HUD_FONT_SCALE,
                scheme["text"], 1, cv2.LINE_AA,
            )
            y += 18

        # ── Bottom hint bar ───────────────────────────────────────────────────
        cv2.putText(
            frame,
            "Open palm hold: switch mode  |  Tab: instant switch  |  Q: quit",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        # ── Mode-switch banner (shown for BANNER_DURATION after every switch) ─
        if mode_switched_at > 0 and (time.time() - mode_switched_at) < BANNER_DURATION:
            self._draw_mode_banner(frame, mode, scheme, w, h)

        return frame

    # ── Banner helper ──────────────────────────────────────────────────────────

    def _draw_mode_banner(
        self,
        frame: np.ndarray,
        mode: str,
        scheme: dict,
        w: int,
        h: int,
    ) -> None:
        label = f"  {mode.upper()} MODE  "
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.6
        thickness  = 3

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tx = (w - tw) // 2
        ty = h // 2 + th // 2

        pad = 18
        # Dark semi-transparent background box
        box_overlay = frame.copy()
        cv2.rectangle(
            box_overlay,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + baseline + pad),
            (10, 10, 10), -1,
        )
        cv2.addWeighted(box_overlay, 0.75, frame, 0.25, 0, frame)

        # Coloured border around box
        cv2.rectangle(
            frame,
            (tx - pad, ty - th - pad),
            (tx + tw + pad, ty + baseline + pad),
            scheme["border"], 2,
        )

        # Text
        cv2.putText(
            frame, label, (tx, ty),
            font, font_scale, scheme["banner"], thickness, cv2.LINE_AA,
        )
