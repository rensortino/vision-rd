"""Run RF-DETR inference over a flat YOLO split and cache predictions.

Writes the same ``predictions.json`` schema the backend-agnostic evaluator
reads -- a list of ``{frame_id, detections:[{class_id,cx,cy,w,h,confidence}]}``
with boxes in normalized center format. Inference runs over the canonical YOLO
``--data-dir/images`` (not the RF-DETR COCO symlink tree) so frame ids (image
stems) line up with the ground-truth label files the evaluator loads.

A low confidence floor (0.01) caches every box; the operating point is selected
later on val by the evaluator. Every class id is folded to 0 ("smoke").

Run with the isolated interpreter::

    .rfdetr-venv/bin/python scripts/infer_rfdetr.py \
        --data-dir data/01_raw/datasets/test \
        --checkpoint data/06_models/rfdetr-nano/checkpoint_best_total.pth \
        --output-file data/02_intermediate/rfdetr-nano/test_predictions.json \
        --resolution 1024 --confidence-threshold 0.01
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image
from rfdetr import RFDETRNano
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR inference -> predictions")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    args = parser.parse_args()

    images_dir = args.data_dir / "images"
    image_paths = sorted(images_dir.glob("*.jpg"))
    logger.info("Found %d images in %s", len(image_paths), images_dir)

    model = RFDETRNano.from_checkpoint(
        str(args.checkpoint), resolution=args.resolution, num_classes=1
    )
    model.optimize_for_inference()

    results = []
    for path in tqdm(image_paths, desc="Inference (rfdetr)"):
        image = Image.open(path).convert("RGB")
        w, h = image.size
        det = model.predict(image, threshold=args.confidence_threshold)
        detections = []
        for (x1, y1, x2, y2), score in zip(det.xyxy, det.confidence, strict=True):
            detections.append(
                {
                    "class_id": 0,  # single class: smoke
                    "cx": (float(x1) + float(x2)) / 2 / w,
                    "cy": (float(y1) + float(y2)) / 2 / h,
                    "w": (float(x2) - float(x1)) / w,
                    "h": (float(y2) - float(y1)) / h,
                    "confidence": float(score),
                }
            )
        results.append({"frame_id": path.stem, "detections": detections})

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results, indent=2))
    logger.info("Saved %d frame predictions to %s", len(results), args.output_file)


if __name__ == "__main__":
    main()
