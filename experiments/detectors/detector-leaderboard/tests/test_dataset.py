"""Tests for the YOLO->COCO bbox conversion used in D-FINE training."""

import pytest

from detector_leaderboard.dataset import yolo_to_coco_bbox


def test_centered_box():
    # Full-image centered box on a 100x200 image.
    x, y, w, h = yolo_to_coco_bbox(0.5, 0.5, 1.0, 1.0, 100, 200)
    assert (x, y, w, h) == pytest.approx([0.0, 0.0, 100.0, 200.0])


def test_quarter_box_top_left():
    # cx,cy=0.25; w,h=0.5 -> spans [0,0.5] in normalized -> absolute on 200x100.
    x, y, w, h = yolo_to_coco_bbox(0.25, 0.25, 0.5, 0.5, 200, 100)
    assert (x, y, w, h) == pytest.approx([0.0, 0.0, 100.0, 50.0])


def test_clips_to_image_bounds():
    # A box partly outside the image is clipped so x+w <= W and y+h <= H.
    x, y, w, h = yolo_to_coco_bbox(0.95, 0.95, 0.2, 0.2, 100, 100)
    assert x >= 0.0 and y >= 0.0
    assert x + w == pytest.approx(100.0, abs=1e-6)
    assert y + h == pytest.approx(100.0, abs=1e-6)
