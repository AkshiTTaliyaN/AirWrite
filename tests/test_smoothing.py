"""Unit tests for prediction stabilizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.smoothing import ExponentialSmoother, PredictionStabilizer


def test_smoother():
    s = ExponentialSmoother(alpha=0.5)
    x, y = s.update(100, 100)
    assert x == 100 and y == 100
    x2, y2 = s.update(200, 200)
    assert x2 == 150 and y2 == 150


def test_stabilizer():
    st = PredictionStabilizer(required_frames=3)
    assert st.update("H") == ""
    assert st.update("H") == ""
    assert st.update("H") == "H"
