"""YOLO inference wrapper.

Provides functions to load a YOLO model and run it on individual frames. JSON
(de)serialization lives in :mod:`serialization` (re-exported here) so callers
that only read cached predictions need not import the ultralytics stack.
"""

from pathlib import Path

from ultralytics import YOLO

from .serialization import load_predictions, save_predictions
from .types import Detection, FrameResult

__all__ = [
    "load_model",
    "run_inference_on_image",
    "save_predictions",
    "load_predictions",
]


def load_model(model_path: Path) -> YOLO:
    """Load a YOLO model from a .pt file."""
    return YOLO(str(model_path))


def run_inference_on_image(
    model: YOLO,
    image_path: Path,
    conf: float,
    iou_nms: float,
    img_size: int,
) -> FrameResult:
    """Run YOLO on a single image.

    Args:
        model: Loaded YOLO model instance.
        image_path: Path to the image file.
        conf: Minimum confidence threshold for YOLO predictions.
        iou_nms: IoU threshold used by Non-Maximum Suppression.
        img_size: Input image size (pixels) passed to YOLO.

    Returns:
        A :class:`FrameResult` (``frame_id`` = filename stem) with normalized
        center-based detections (xywhn).
    """
    preds = model.predict(
        str(image_path),
        conf=conf,
        iou=iou_nms,
        imgsz=img_size,
        verbose=False,
    )

    detections = []
    for pred in preds:
        boxes = pred.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            xywhn = boxes.xywhn[i].tolist()
            detections.append(
                Detection(
                    class_id=int(boxes.cls[i].item()),
                    cx=xywhn[0],
                    cy=xywhn[1],
                    w=xywhn[2],
                    h=xywhn[3],
                    confidence=float(boxes.conf[i].item()),
                )
            )

    return FrameResult(frame_id=image_path.stem, detections=detections)
