"""
AirWrite — unified touchless input pipeline.

Webcam → MediaPipe → Gesture → Mode → Stroke Buffer → Segmentation → CNN → pyautogui
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pynput import keyboard

sys.path.insert(0, str(Path(__file__).parent))

import config
from core.cursor_engine import CursorEngine
from core.gesture_engine import GestureEngine, GestureType, HandState
from core.segmentation_engine import SegmentationEngine
from core.stroke_engine import StrokeEngine
from ml.inference.predictor import CharacterPredictor
from utils.hud import HUD
from utils.smoothing import PredictionStabilizer


class AirWriteApp:
    def __init__(self):
        self.gestures = GestureEngine()
        self.cursor = CursorEngine()
        self.strokes = StrokeEngine()
        self.segmentation = SegmentationEngine(self.strokes.canvas_size)
        self.predictor = CharacterPredictor()
        self.stabilizer = PredictionStabilizer(config.PREDICTION_STABLE_FRAMES)
        self.hud = HUD()

        self._last_pen_up_time: float = 0.0
        self._last_committed_word = ""
        self._raw_prediction = ""
        self._running = True
        self._backspace_pressed = False
        self._wipe_active = False  # tracks if wipe is in progress, suppresses mode switch
        self._mode_switched_at: float = 0.0  # timestamp of last mode switch for banner

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        pyautogui.FAILSAFE = True

    # ── Keyboard listener ─────────────────────────────────────────────────────

    def _on_key(self, key) -> None:
        try:
            if key == keyboard.Key.esc:
                self._running = False
            elif key == keyboard.Key.backspace:
                self._backspace_pressed = True
            elif key == keyboard.Key.tab:
                # Instant mode toggle — reliable alternative to open-palm gesture
                self.gestures.mode = (
                    config.MODE_WRITE
                    if self.gestures.mode == config.MODE_CURSOR
                    else config.MODE_CURSOR
                )
                self.strokes.reset_buffer()
                self.stabilizer.reset()
                self._raw_prediction = ""
                self._mode_switched_at = time.time()
            elif hasattr(key, "char") and key.char in ("q", "Q"):
                self._running = False
        except AttributeError:
            pass

    def _start_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self._on_key)
        listener.daemon = True
        listener.start()

    # ── Recognition ───────────────────────────────────────────────────────────

    def _update_recognition(self) -> None:
        """Run segmentation + CNN on current stroke buffer. Call on pen-up only."""
        if not self.strokes.points:
            return
        canvas = self.strokes.get_canvas_image()
        segments = self.segmentation.segment(self.strokes.points, self.strokes.pen_down)
        crops = [self.segmentation.crop_character(canvas, seg) for seg in segments]
        self._raw_prediction = self.predictor.predict_word(crops)
        self.stabilizer.update(self._raw_prediction)

    # ── Word lifecycle ────────────────────────────────────────────────────────

    def _commit_word(self) -> None:
        word = self.stabilizer.stable or self._raw_prediction
        if word and word != "?":
            pyautogui.write(word + " ", interval=0.02)
            self._last_committed_word = word
        self.strokes.reset_buffer()
        self.stabilizer.reset()
        self._raw_prediction = ""

    def _clear_current_word(self) -> None:
        self.strokes.reset_buffer()
        self.stabilizer.reset()
        self._raw_prediction = ""

    def _delete_previous_word(self) -> None:
        if self._last_committed_word:
            # Delete word + the trailing space we injected
            for _ in range(len(self._last_committed_word) + 1):
                pyautogui.press("backspace")
            self._last_committed_word = ""

    def _check_word_pause(self) -> None:
        """Commit word if pen has been up long enough."""
        if self.strokes.pen_down or not self.strokes.points:
            return
        if time.time() - self._last_pen_up_time >= config.WORD_PAUSE_THRESHOLD:
            self._commit_word()

    # ── Wipe gesture ──────────────────────────────────────────────────────────

    def _handle_wipe(self) -> None:
        """
        Wipe before word commit → clear current buffer.
        Wipe after word commit  → delete previously injected word.
        """
        if self.strokes.points:
            self._clear_current_word()
        elif self._last_committed_word:
            self._delete_previous_word()

    # ── Mode handlers ─────────────────────────────────────────────────────────

    def _process_cursor_mode(self, hand: HandState, gesture: GestureType) -> None:
        # Frame is already cv2.flip()'d — use raw landmark pixel directly.
        x, y = hand.index_tip_pixel()

        if gesture == GestureType.INDEX_EXTENDED:
            self.cursor.move(x, y, hand.frame_w, hand.frame_h)

        if gesture == GestureType.TWO_FINGER_PINCH:
            self.cursor.handle_scroll_drag(y, active=True)
        else:
            self.cursor.handle_scroll_drag(y, active=False)

        if self.gestures.check_right_click_hold(gesture):
            self.cursor.right_click()
            self.gestures.reset_pinch_timer()
            self.cursor.reset_pinch_click()
        elif gesture == GestureType.PINCH and self.cursor.can_click():
            self.cursor.left_click()

        if gesture != GestureType.PINCH:
            self.cursor.reset_pinch_click()
            self.gestures.reset_pinch_timer()

    def _process_write_mode(self, hand: HandState, gesture: GestureType) -> None:
        x, y = hand.index_tip_pixel()

        if gesture == GestureType.PINCH:
            if not self.strokes.pen_down:
                self.strokes.set_canvas_mapping(hand.frame_w, hand.frame_h)
                self.strokes.pen_down = True
            self.strokes.add_point_mapped(x, y)

        elif self.strokes.pen_down:
            # Pen just lifted — run recognition once here, not every frame
            self.strokes.end_stroke()
            self._last_pen_up_time = time.time()
            self._update_recognition()

        self._check_word_pause()

        if self._backspace_pressed:
            pyautogui.press("backspace")
            self._backspace_pressed = False

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._start_keyboard_listener()

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        if not self.predictor.is_ready:
            print(
                f"Warning: No trained model found at {config.DEFAULT_MODEL_PATH}\n"
                "Train first: python -m ml.training.train_cnn --epochs 5\n"
                "Cursor mode will still work."
            )

        print("AirWrite started.")
        print("  Open palm hold (~0.8s) → switch mode")
        print("  Q / Esc                → quit")

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

            # Single flip here — everything downstream uses these coordinates
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            gesture = GestureType.NONE
            hand_state = None

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                hand_state = HandState(lm.landmark, w, h)
                gesture = self.gestures.detect_primary(hand_state)

                # Wipe detection — if active, skip mode switch this frame
                wipe_fired = self.gestures.detect_wipe(hand_state)
                if wipe_fired:
                    self._handle_wipe()

                # Mode switch: allow unless a wipe fired this frame OR hand is
                # actively swiping fast (per-frame velocity, not cumulative drift).
                if not wipe_fired and self.gestures._last_wipe_dx < config.WIPE_MIN_VELOCITY:
                    if self.gestures.update_mode_switch(gesture):
                        self.strokes.reset_buffer()
                        self.stabilizer.reset()
                        self._raw_prediction = ""
                        self._mode_switched_at = time.time()

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )

                if self.gestures.mode == config.MODE_CURSOR:
                    self._process_cursor_mode(hand_state, gesture)
                else:
                    self._process_write_mode(hand_state, gesture)
                    x, y = hand_state.index_tip_pixel()
                    self.strokes.render_overlay(frame, x, y)

            self.hud.tick_fps()
            self.hud.draw(
                frame,
                mode=self.gestures.mode,
                prediction=self._raw_prediction,
                stable_prediction=self.stabilizer.stable,
                pen_down=self.strokes.pen_down,
                gesture_label=self.gestures.gesture_label(gesture) if hand_state else "",
                mode_switched_at=self._mode_switched_at,
            )

            if self.gestures.mode == config.MODE_WRITE:
                self.strokes.render_canvas_preview(frame)

            cv2.imshow("AirWrite", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("AirWrite stopped.")


def main():
    AirWriteApp().run()


if __name__ == "__main__":
    main()