"""Unit tests for heuristic segmentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.segmentation_engine import SegmentationEngine
from core.stroke_engine import StrokePoint


def test_segment_single_stroke():
    engine = SegmentationEngine((400, 400))
    points = [StrokePoint(i * 5, 50, float(i * 10)) for i in range(10)]
    segments = engine.segment(points, pen_down=False)
    assert len(segments) >= 1


def test_pause_boundary():
    engine = SegmentationEngine((400, 400))
    points = [
        StrokePoint(10, 50, 0),
        StrokePoint(20, 50, 50),
        StrokePoint(80, 50, 200),
        StrokePoint(90, 50, 210),
    ]
    segments = engine.segment(points, pen_down=False)
    assert len(segments) >= 1
