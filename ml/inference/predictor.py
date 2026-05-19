"""Runtime CNN inference for segmented characters."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from ml.models.cnn import EMNIST_BALANCED_LABELS, EMNISTCNN, NUM_CLASSES


class CharacterPredictor:
    """Load trained weights and predict characters from stroke crops."""

    def __init__(self, model_path: Path | None = None, device: str | None = None):
        self.model_path = Path(model_path or config.DEFAULT_MODEL_PATH)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = EMNISTCNN(NUM_CLASSES).to(self.device)
        self._ready = False
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def preprocess(self, gray: np.ndarray) -> torch.Tensor:
        """Normalize 28x28 grayscale crop for CNN."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE))
        # EMNIST images are white on black; strokes are white on our canvas
        img = 255 - img if np.mean(img) > 127 else img
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict_char(self, gray: np.ndarray) -> tuple[str, float]:
        if not self._ready:
            return "?", 0.0
        with torch.no_grad():
            x = self.preprocess(gray)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            label = EMNIST_BALANCED_LABELS[int(idx.item())]
            return label, float(conf.item())

    def predict_word(self, crops: list[np.ndarray]) -> str:
        chars = []
        for crop in crops:
            ch, conf = self.predict_char(crop)
            if conf > 0.15:
                chars.append(ch)
        return "".join(chars)
