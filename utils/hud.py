"""Semi-polished functional HUD overlay."""

from __future__ import annotations

import cv2
import numpy as np

import config


class HUD:
    """Draw mode, FPS, prediction preview, and status indicators."""

    def __init__(self):
        self.fps = 0.0
        self._frame_count = 0
        self._fps_timer = 0.0

    def tick_fps(self) -> None:
        import time

        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

    def draw(
        self,
        frame: np.ndarray,
        *,
        mode: str,
        prediction: str,
        stable_prediction: str,
        pen_down: bool,
        gesture_label: str = "",
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        panel_h = 110
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, config.HUD_BG_ALPHA, frame, 1 - config.HUD_BG_ALPHA, 0, frame)

        lines = [
            f"Mode: {mode.upper()}",
            f"FPS: {self.fps:.1f}",
            f"Gesture: {gesture_label or '-'}",
            f"Preview: {prediction or '-'}",
            f"Stable: {stable_prediction or '-'}",
            f"Pen: {'DOWN' if pen_down else 'UP'}",
        ]
        y = 22
        for line in lines:
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.HUD_FONT_SCALE,
                config.HUD_COLOR,
                1,
                cv2.LINE_AA,
            )
            y += 18

        # Fingertip indicator hint
        cv2.putText(
            frame,
            "Open palm hold: switch mode | Q: quit",
            (12, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return frame
