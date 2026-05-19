"""OS-level cursor control via pyautogui."""

from __future__ import annotations

import pyautogui

import config
from utils.smoothing import ExponentialSmoother

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class CursorEngine:
    """Map hand landmarks to screen cursor with smoothing.

    NOTE: expects coordinates already in display space (frame already
    flipped with cv2.flip in the main loop). No mirroring is done here.
    """

    def __init__(self):
        self.smoother = ExponentialSmoother(alpha=config.CURSOR_SMOOTHING)
        self.screen_w, self.screen_h = pyautogui.size()
        self._scroll_anchor_y: int | None = None
        self._clicked_this_pinch = False

    def move(self, x: int, y: int, frame_w: int, frame_h: int) -> None:
        """Move cursor. x/y are already in display space (no mirroring)."""
        sx = int(x / frame_w * self.screen_w)
        sy = int(y / frame_h * self.screen_h)
        sx = max(config.SCREEN_MARGIN, min(self.screen_w - config.SCREEN_MARGIN, sx))
        sy = max(config.SCREEN_MARGIN, min(self.screen_h - config.SCREEN_MARGIN, sy))
        sx, sy = self.smoother.update(sx, sy)
        pyautogui.moveTo(int(sx), int(sy), _pause=False)

    def left_click(self) -> None:
        pyautogui.click(_pause=False)
        self._clicked_this_pinch = True

    def right_click(self) -> None:
        pyautogui.rightClick(_pause=False)
        self._clicked_this_pinch = True

    def scroll(self, dy: int) -> None:
        pyautogui.scroll(dy, _pause=False)

    def handle_scroll_drag(self, finger_y: int, active: bool) -> None:
        if not active:
            self._scroll_anchor_y = None
            return
        if self._scroll_anchor_y is None:
            self._scroll_anchor_y = finger_y
            return
        delta = self._scroll_anchor_y - finger_y
        if abs(delta) > 15:
            self.scroll(3 if delta > 0 else -3)
            self._scroll_anchor_y = finger_y

    def reset_pinch_click(self) -> None:
        self._clicked_this_pinch = False

    def can_click(self) -> bool:
        return not self._clicked_this_pinch