"""Gesture detection and mode switching."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

import config
from utils.helpers import distance_2d, landmark_to_pixel


class GestureType(Enum):
    NONE = auto()
    INDEX_EXTENDED = auto()
    PINCH = auto()
    TWO_FINGER_PINCH = auto()
    OPEN_PALM = auto()
    WIPE = auto()


@dataclass
class HandState:
    landmarks: list
    frame_w: int
    frame_h: int

    def _lm(self, idx: int):
        return self.landmarks[idx]

    def pinch_distance(self) -> float:
        """Thumb (4) to index (8) distance, normalized."""
        t = self._lm(4)
        i = self._lm(8)
        return distance_2d((t.x, t.y), (i.x, i.y))

    def middle_index_distance(self) -> float:
        m = self._lm(12)
        i = self._lm(8)
        return distance_2d((m.x, m.y), (i.x, i.y))

    def is_pinch(self) -> bool:
        return self.pinch_distance() < config.PINCH_THRESHOLD

    def is_two_finger_pinch(self) -> bool:
        return (
            self.pinch_distance() < config.SCROLL_PINCH_THRESHOLD
            and self.middle_index_distance() < config.SCROLL_PINCH_THRESHOLD
        )

    def is_open_palm(self) -> bool:
        """Fingers extended and spread."""
        tips = [8, 12, 16, 20]
        wrist = self._lm(0)
        extended = 0
        for tip_idx in tips:
            tip = self._lm(tip_idx)
            pip = self._lm(tip_idx - 2)
            if tip.y < pip.y:
                extended += 1
        if extended < 4:
            return False
        spreads = []
        for i in range(len(tips) - 1):
            a = self._lm(tips[i])
            b = self._lm(tips[i + 1])
            spreads.append(distance_2d((a.x, a.y), (b.x, b.y)))
        return min(spreads) > config.OPEN_PALM_FINGER_SPREAD and not self.is_pinch()

    def index_tip_pixel(self) -> tuple[int, int]:
        return landmark_to_pixel(self._lm(8), self.frame_w, self.frame_h)

    def palm_center_pixel(self) -> tuple[int, int]:
        xs = [self._lm(i).x for i in (0, 5, 9, 13, 17)]
        ys = [self._lm(i).y for i in (0, 5, 9, 13, 17)]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        return int(cx * self.frame_w), int(cy * self.frame_h)


class GestureEngine:
    """Detect gestures and manage cursor/write mode."""

    def __init__(self, initial_mode: str = config.MODE_CURSOR):
        self.mode = initial_mode
        self._pinch_start: float | None = None
        self._open_palm_start: float | None = None
        self._wipe_cooldown_until = 0.0
        self._wipe_prev_x: float | None = None
        self._wipe_distance = 0.0

    def detect_primary(self, hand: HandState) -> GestureType:
        if hand.is_open_palm():
            return GestureType.OPEN_PALM
        if hand.is_two_finger_pinch():
            return GestureType.TWO_FINGER_PINCH
        if hand.is_pinch():
            return GestureType.PINCH
        tip = hand._lm(8)
        pip = hand._lm(6)
        if tip.y < pip.y:
            return GestureType.INDEX_EXTENDED
        return GestureType.NONE

    def update_mode_switch(self, gesture: GestureType) -> bool:
        """Return True if mode was toggled."""
        now = time.time()
        if gesture == GestureType.OPEN_PALM:
            if self._open_palm_start is None:
                self._open_palm_start = now
            elif now - self._open_palm_start >= config.MODE_SWITCH_HOLD:
                self.mode = (
                    config.MODE_WRITE
                    if self.mode == config.MODE_CURSOR
                    else config.MODE_CURSOR
                )
                self._open_palm_start = None
                return True
        else:
            self._open_palm_start = None
        return False

    def check_right_click_hold(self, gesture: GestureType) -> bool:
        """Hold pinch for 1s triggers right click."""
        now = time.time()
        if gesture == GestureType.PINCH:
            if self._pinch_start is None:
                self._pinch_start = now
            return (now - self._pinch_start) >= config.RIGHT_CLICK_HOLD
        self._pinch_start = None
        return False

    def reset_pinch_timer(self) -> None:
        self._pinch_start = None

    def detect_wipe(self, hand: HandState) -> bool:
        """Horizontal full-hand swipe for delete/clear."""
        now = time.time()
        if now < self._wipe_cooldown_until:
            return False
        if not hand.is_open_palm():
            self._wipe_prev_x = None
            self._wipe_distance = 0.0
            return False

        cx, _ = hand.palm_center_pixel()
        nx = cx / hand.frame_w
        if self._wipe_prev_x is not None:
            dx = abs(nx - self._wipe_prev_x)
            self._wipe_distance += dx
            velocity = dx
            if (
                self._wipe_distance >= config.WIPE_MIN_DISTANCE
                and velocity >= config.WIPE_MIN_VELOCITY
            ):
                self._wipe_cooldown_until = now + config.WIPE_COOLDOWN
                self._wipe_prev_x = None
                self._wipe_distance = 0.0
                return True
        self._wipe_prev_x = nx
        return False

    def gesture_label(self, gesture: GestureType) -> str:
        return gesture.name.replace("_", " ").title()
