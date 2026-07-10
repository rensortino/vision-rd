"""(De)serialization of per-frame detection results to/from JSON.

Kept separate from :mod:`detector` so that evaluation and metrics code can read
cached predictions without importing the heavy YOLO/ultralytics stack.
"""

import json
from pathlib import Path

from .types import Detection, FrameResult


def save_predictions(results: list[FrameResult], output_path: Path) -> None:
    """Save per-frame detection results as a single JSON file.

    Args:
        results: Frame results to serialize.
        output_path: Destination ``.json`` path (parent dirs are created).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "frame_id": frame.frame_id,
            "detections": [
                {
                    "class_id": d.class_id,
                    "cx": d.cx,
                    "cy": d.cy,
                    "w": d.w,
                    "h": d.h,
                    "confidence": d.confidence,
                }
                for d in frame.detections
            ],
        }
        for frame in results
    ]
    output_path.write_text(json.dumps(data, indent=2))


def load_predictions(input_path: Path) -> list[FrameResult]:
    """Load cached predictions written by :func:`save_predictions`."""
    data = json.loads(input_path.read_text())
    results = []
    for frame_data in data:
        detections = [
            Detection(
                class_id=d["class_id"],
                cx=d["cx"],
                cy=d["cy"],
                w=d["w"],
                h=d["h"],
                confidence=d["confidence"],
            )
            for d in frame_data["detections"]
        ]
        results.append(
            FrameResult(frame_id=frame_data["frame_id"], detections=detections)
        )
    return results
