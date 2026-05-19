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

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

import config
from core.cursor_engine import CursorEngine
from core.gesture_engine import GestureEngine, GestureType, HandState
from core.segmentation_engine import SegmentationEngine
from core.stroke_engine import StrokeEngine
from ml.inference.predictor import CharacterPredictor
from utils.helpers import mirror_x
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

        self._last_activity = time.time()
        self._last_committed_word = ""
        self._raw_prediction = ""
        self._running = True
        self._backspace_pressed = False

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        pyautogui.FAILSAFE = True

    def _on_key(self, key) -> None:
        try:
            if key == keyboard.Key.esc or (hasattr(key, "char") and key.char in ("q", "Q")):
                self._running = False
            if hasattr(key, "char") and key.char == "\x08":
                self._backspace_pressed = True
        except AttributeError:
            pass

    def _start_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self._on_key)
        listener.daemon = True
        listener.start()

    def _update_recognition(self) -> None:
        canvas = self.strokes.get_canvas_image()
        segments = self.segmentation.segment(self.strokes.points, self.strokes.pen_down)
        crops = [self.segmentation.crop_character(canvas, seg) for seg in segments]
        self._raw_prediction = self.predictor.predict_word(crops)
        self.stabilizer.update(self._raw_prediction)

    def _commit_word(self) -> None:
        word = self.stabilizer.stable or self._raw_prediction
        if word and word != "?":
            pyautogui.write(word, interval=0.02)
            self._last_committed_word = word
        self.strokes.reset_buffer()
        self.stabilizer.reset()
        self._raw_prediction = ""
        self._last_activity = time.time()

    def _clear_current_word(self) -> None:
        self.strokes.reset_buffer()
        self.stabilizer.reset()
        self._raw_prediction = ""

    def _delete_previous_word(self) -> None:
        if self._last_committed_word:
            for _ in range(len(self._last_committed_word)):
                pyautogui.press("backspace")
            self._last_committed_word = ""

    def _handle_wipe(self, committed: bool) -> None:
        if committed:
            self._delete_previous_word()
        else:
            self._clear_current_word()

    def _check_word_pause(self) -> None:
        if self.strokes.pen_down:
            return
        if not self.strokes.points:
            return
        if time.time() - self._last_activity >= config.WORD_PAUSE_THRESHOLD:
            self._commit_word()

    def _process_cursor_mode(self, hand: HandState, gesture: GestureType) -> None:
        x, y = hand.index_tip_pixel()
        mx = mirror_x(x, hand.frame_w)

        if gesture == GestureType.INDEX_EXTENDED:
            self.cursor.move(mx, y, hand.frame_w, hand.frame_h)

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
            self.cursor.reset_pinch_click()

        if gesture != GestureType.PINCH:
            self.cursor.reset_pinch_click()
            self.gestures.reset_pinch_timer()

    def _process_write_mode(self, hand: HandState, gesture: GestureType) -> None:
        x, y = hand.index_tip_pixel()
        mx = mirror_x(x, hand.frame_w)

        if gesture == GestureType.PINCH:
            if not self.strokes.pen_down:
                self.strokes.set_canvas_mapping(hand.frame_w, hand.frame_h)
                self.strokes.pen_down = True
                self.strokes.add_point_mapped(mx, y)
            else:
                self.strokes.add_point_mapped(mx, y)
            self._last_activity = time.time()
        elif self.strokes.pen_down:
            self.strokes.end_stroke()
            self._last_activity = time.time()

        if self.strokes.points:
            self._update_recognition()

        self._check_word_pause()

        if self._backspace_pressed:
            pyautogui.press("backspace")
            self._backspace_pressed = False

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
                "Warning: No trained model found at",
                config.DEFAULT_MODEL_PATH,
                "\nTrain with: python -m ml.training.train_cnn --epochs 5",
            )

        print("AirWrite started. Open palm hold to switch mode. Press Q or Esc to quit.")

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

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

                if self.gestures.detect_wipe(hand_state):
                    committed = not self.strokes.points and bool(self._last_committed_word)
                    self._handle_wipe(committed=committed)

                if self.gestures.update_mode_switch(gesture):
                    self.strokes.reset_buffer()
                    self.stabilizer.reset()

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
                    if hand_state:
                        x, y = hand_state.index_tip_pixel()
                        mx = mirror_x(x, w)
                        self.strokes.render_overlay(frame, mx, y)

            self.hud.tick_fps()
            self.hud.draw(
                frame,
                mode=self.gestures.mode,
                prediction=self._raw_prediction,
                stable_prediction=self.stabilizer.stable,
                pen_down=self.strokes.pen_down,
                gesture_label=self.gestures.gesture_label(gesture) if hand_state else "",
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
