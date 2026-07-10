"""Tests for the frame-level object-detection metrics."""

from pathlib import Path

import pytest

from detector_leaderboard.metrics import (
    compute_detection_metrics,
    iou_xywhn,
    match_frame,
    select_best_threshold,
)
from detector_leaderboard.serialization import save_predictions
from detector_leaderboard.types import Detection, FrameResult


def _det(cx, cy, w, h, conf=1.0):
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=conf)


# --- iou_xywhn ---------------------------------------------------------------


def test_iou_identical_boxes():
    box = _det(0.5, 0.5, 0.2, 0.2)
    assert iou_xywhn(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    a = _det(0.1, 0.1, 0.1, 0.1)
    b = _det(0.9, 0.9, 0.1, 0.1)
    assert iou_xywhn(a, b) == 0.0


def test_iou_partial_overlap():
    a = _det(0.25, 0.25, 0.5, 0.5)  # spans [0.0, 0.5] x [0.0, 0.5]
    b = _det(0.50, 0.50, 0.5, 0.5)  # spans [0.25, 0.75] x [0.25, 0.75]
    # intersection = 0.25*0.25 = 0.0625; union = 0.25 + 0.25 - 0.0625 = 0.4375
    assert iou_xywhn(a, b) == pytest.approx(0.0625 / 0.4375)


# --- match_frame -------------------------------------------------------------


def test_match_all_true_positives():
    gts = [_det(0.5, 0.5, 0.2, 0.2), _det(0.2, 0.2, 0.1, 0.1)]
    preds = [_det(0.5, 0.5, 0.2, 0.2, 0.9), _det(0.2, 0.2, 0.1, 0.1, 0.8)]
    assert match_frame(preds, gts, iou_threshold=0.5) == (2, 0, 0)


def test_match_with_false_positive_and_negative():
    gts = [_det(0.5, 0.5, 0.2, 0.2)]
    preds = [
        _det(0.5, 0.5, 0.2, 0.2, 0.9),  # matches the GT
        _det(0.9, 0.9, 0.1, 0.1, 0.5),  # spurious -> FP
    ]
    assert match_frame(preds, gts, iou_threshold=0.5) == (1, 1, 0)
    # An unmatched GT becomes a FN when there is no prediction near it.
    assert match_frame([], gts, iou_threshold=0.5) == (0, 0, 1)


def test_match_one_prediction_cannot_claim_two_gts():
    gts = [_det(0.5, 0.5, 0.2, 0.2), _det(0.5, 0.5, 0.2, 0.2)]
    preds = [_det(0.5, 0.5, 0.2, 0.2, 0.9)]
    assert match_frame(preds, gts, iou_threshold=0.5) == (1, 0, 1)


# --- compute_detection_metrics ----------------------------------------------


def _build_split(tmp_path: Path, frames: list[dict]) -> tuple[Path, Path]:
    """Create a flat YOLO split + a predictions.json from frame specs.

    Each frame spec is a dict with ``stem``, ``gt`` (list of Detection) and
    ``preds`` (list of Detection). Returns ``(data_dir, predictions_path)``.
    """
    data_dir = tmp_path / "test"
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[FrameResult] = []
    for f in frames:
        (images_dir / f"{f['stem']}.jpg").write_bytes(b"")  # content unused
        lines = [f"0 {b.cx} {b.cy} {b.w} {b.h}" for b in f["gt"]]
        (labels_dir / f"{f['stem']}.txt").write_text("\n".join(lines))
        predictions.append(FrameResult(frame_id=f["stem"], detections=f["preds"]))

    predictions_path = tmp_path / "predictions.json"
    save_predictions(predictions, predictions_path)
    return data_dir, predictions_path


def test_compute_metrics_end_to_end(tmp_path):
    gt_box = _det(0.5, 0.5, 0.2, 0.2)
    data_dir, predictions_path = _build_split(
        tmp_path,
        frames=[
            # GT frame, correct detection -> TP.
            {"stem": "f0", "gt": [gt_box], "preds": [_det(0.5, 0.5, 0.2, 0.2, 0.9)]},
            # GT frame, misses GT + spurious box -> FP + FN.
            {"stem": "f1", "gt": [gt_box], "preds": [_det(0.9, 0.9, 0.1, 0.1, 0.8)]},
            # Background frame that fires -> image FP.
            {"stem": "bg0", "gt": [], "preds": [_det(0.3, 0.3, 0.1, 0.1, 0.7)]},
            # Background frame that stays quiet.
            {"stem": "bg1", "gt": [], "preds": []},
        ],
    )

    m = compute_detection_metrics(
        model_name="dummy",
        data_dir=data_dir,
        predictions_path=predictions_path,
        conf_threshold=0.2,
        iou_threshold=0.5,
    )

    assert m.num_frames_with_gt == 2
    assert (m.box_tp, m.box_fp, m.box_fn) == (1, 1, 1)
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.f1 == 0.5
    assert m.num_background_frames == 2
    assert m.background_frames_fired == 1
    assert m.image_fpr == 0.5
    assert m.mean_fp_per_background_frame == 0.5


def test_conf_threshold_filters_low_confidence_predictions(tmp_path):
    gt_box = _det(0.5, 0.5, 0.2, 0.2)
    data_dir, predictions_path = _build_split(
        tmp_path,
        frames=[
            # Only prediction is below the operating threshold -> dropped.
            {"stem": "f0", "gt": [gt_box], "preds": [_det(0.5, 0.5, 0.2, 0.2, 0.1)]},
        ],
    )

    m = compute_detection_metrics(
        model_name="dummy",
        data_dir=data_dir,
        predictions_path=predictions_path,
        conf_threshold=0.2,
        iou_threshold=0.5,
    )
    # No prediction survives -> the GT box is a false negative.
    assert (m.box_tp, m.box_fp, m.box_fn) == (0, 0, 1)
    assert m.recall == 0.0


def test_select_best_threshold_prefers_high_conf_correct_box(tmp_path):
    gt = _det(0.5, 0.5, 0.2, 0.2)
    # One smoke frame: a correct high-conf box (0.6) and a spurious low-conf
    # box (0.05). A threshold above 0.05 drops the FP and yields F1=1.0.
    data_dir, predictions_path = _build_split(
        tmp_path,
        frames=[
            {
                "stem": "f0",
                "gt": [gt],
                "preds": [
                    _det(0.5, 0.5, 0.2, 0.2, 0.6),
                    _det(0.1, 0.1, 0.05, 0.05, 0.05),
                ],
            }
        ],
    )
    grid = [0.01, 0.1, 0.3, 0.5, 0.7]
    best_conf, best_f1 = select_best_threshold(
        "dummy", data_dir, predictions_path, iou_threshold=0.5, conf_grid=grid
    )
    assert best_f1 == 1.0
    # Lowest grid value that already achieves F1=1.0 (drops the 0.05 FP).
    assert best_conf == 0.1


def test_background_only_split_has_zero_division_guards(tmp_path):
    data_dir, predictions_path = _build_split(
        tmp_path,
        frames=[{"stem": "bg0", "gt": [], "preds": []}],
    )
    m = compute_detection_metrics(
        model_name="dummy",
        data_dir=data_dir,
        predictions_path=predictions_path,
        conf_threshold=0.2,
        iou_threshold=0.5,
    )
    assert m.num_frames_with_gt == 0
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)
    assert m.num_background_frames == 1
    assert m.image_fpr == 0.0
