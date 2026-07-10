"""Frame-level object-detection metrics.

Computes Pyronear-style detection metrics for a detector's cached predictions
over a flat YOLO split:

* **Box-level precision / recall / F1** on frames that carry ground-truth boxes.
  Predicted boxes are greedily matched to GT boxes by descending confidence, a
  match requiring ``IoU >= iou_threshold``.
* **Image-level false-positive rate** on background frames (empty/no GT label).
  A background frame "fires" if it has any detection at ``conf >= conf_threshold``.

All bounding boxes are normalized center-based (``cx, cy, w, h`` in [0, 1]).
"""

from collections.abc import Sequence
from pathlib import Path

from .data import iter_frames
from .serialization import load_predictions
from .types import Detection, DetectionMetrics


def iou_xywhn(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two normalized center-based boxes.

    Returns 0.0 when the boxes do not overlap or either has zero area.
    """
    ax1, ay1 = a.cx - a.w / 2, a.cy - a.h / 2
    ax2, ay2 = a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1 = b.cx - b.w / 2, b.cy - b.h / 2
    bx2, by2 = b.cx + b.w / 2, b.cy + b.h / 2

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    union = a.w * a.h + b.w * b.h - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def match_frame(
    preds: Sequence[Detection],
    gts: Sequence[Detection],
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedily match predictions to ground-truth boxes for a single frame.

    Predictions are considered in descending confidence order. Each prediction
    claims the highest-IoU unmatched GT box whose ``IoU >= iou_threshold``.
    Matching is class-agnostic.

    Returns:
        ``(tp, fp, fn)`` where ``tp`` = matched predictions, ``fp`` = unmatched
        predictions, and ``fn`` = unmatched GT boxes.
    """
    matched_gt: set[int] = set()
    tp = 0
    for pred in sorted(preds, key=lambda d: d.confidence, reverse=True):
        best_iou = iou_threshold
        best_gt = -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = iou_xywhn(pred, gt)
            if iou >= best_iou:
                best_iou = iou
                best_gt = gi
        if best_gt >= 0:
            matched_gt.add(best_gt)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def _filter_by_conf(detections: Sequence[Detection], conf: float) -> list[Detection]:
    return [d for d in detections if d.confidence >= conf]


def compute_detection_metrics(
    model_name: str,
    data_dir: Path,
    predictions_path: Path,
    conf_threshold: float,
    iou_threshold: float,
) -> DetectionMetrics:
    """Compute detection metrics for one detector over a flat YOLO split.

    Iterates over every frame in *data_dir* (via :func:`iter_frames`), pairs it
    with the cached predictions from *predictions_path* (matched by frame stem),
    and accumulates box-level TP/FP/FN on GT-bearing frames and firing counts on
    background frames.

    Args:
        model_name: Identifier recorded in the returned metrics.
        data_dir: Flat split directory containing ``images/`` and ``labels/``.
        predictions_path: JSON file written by ``scripts/infer.py``.
        conf_threshold: Operating confidence threshold.
        iou_threshold: IoU threshold for matching predictions to GT.

    Returns:
        A :class:`DetectionMetrics` summarising the detector on this split.
    """
    preds_by_frame = {
        fr.frame_id: fr.detections for fr in load_predictions(predictions_path)
    }

    box_tp = box_fp = box_fn = 0
    num_frames_with_gt = 0
    num_background_frames = 0
    background_frames_fired = 0
    background_detections_total = 0

    for frame in iter_frames(data_dir):
        preds = _filter_by_conf(preds_by_frame.get(frame.stem, []), conf_threshold)

        if frame.gt_boxes:
            num_frames_with_gt += 1
            tp, fp, fn = match_frame(preds, frame.gt_boxes, iou_threshold)
            box_tp += tp
            box_fp += fp
            box_fn += fn
        else:
            num_background_frames += 1
            background_detections_total += len(preds)
            if preds:
                background_frames_fired += 1

    precision = box_tp / (box_tp + box_fp) if (box_tp + box_fp) > 0 else 0.0
    recall = box_tp / (box_tp + box_fn) if (box_tp + box_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    image_fpr = (
        background_frames_fired / num_background_frames
        if num_background_frames > 0
        else 0.0
    )
    mean_fp_per_background_frame = (
        background_detections_total / num_background_frames
        if num_background_frames > 0
        else 0.0
    )

    return DetectionMetrics(
        model_name=model_name,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        num_frames_with_gt=num_frames_with_gt,
        num_background_frames=num_background_frames,
        box_tp=box_tp,
        box_fp=box_fp,
        box_fn=box_fn,
        precision=precision,
        recall=recall,
        f1=f1,
        background_frames_fired=background_frames_fired,
        image_fpr=image_fpr,
        mean_fp_per_background_frame=mean_fp_per_background_frame,
    )


def select_best_threshold(
    model_name: str,
    data_dir: Path,
    predictions_path: Path,
    iou_threshold: float,
    conf_grid: list[float],
) -> tuple[float, float]:
    """Pick the confidence threshold that maximizes F1 over *conf_grid*.

    Intended to be run on the validation split so the chosen operating point is
    selected without touching the test set. Ties keep the lower threshold (the
    grid is scanned in ascending order and a strictly-greater F1 is required to
    update).

    Returns:
        ``(best_conf, best_f1)``.
    """
    best_conf = conf_grid[0]
    best_f1 = -1.0
    for conf in conf_grid:
        m = compute_detection_metrics(
            model_name, data_dir, predictions_path, conf, iou_threshold
        )
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_conf = conf
    return best_conf, best_f1


__all__ = [
    "iou_xywhn",
    "match_frame",
    "compute_detection_metrics",
    "select_best_threshold",
]
