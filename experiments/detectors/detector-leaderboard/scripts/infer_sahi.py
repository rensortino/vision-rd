"""Sliced (SAHI) inference over a flat YOLO split -> cached predictions.

Runs Slicing-Aided Hyper Inference: each frame is cut into overlapping tiles, the
detector runs on every tile (plus the full frame), and boxes are merged back to
full-image coordinates. Tiny smoke occupies more pixels within a tile than in the
downscaled full frame, so small-object recall can improve.

Writes the same ``predictions.json`` schema the backend-agnostic evaluator reads
(normalized center boxes, class folded to 0), so a SAHI run is scored and ranked
exactly like a plain run — only the ``model_name`` differs (``<model>-sahi``).

Backends:
- ``ultralytics``: SAHI's native wrapper (correct YOLO post-processing).
- ``hf_detr``: our :class:`HFDetrSahiModel` (SAHI's built-in HF wrapper mis-scores
  D-FINE/LW-DETR/RT-DETR; ours reuses the model's real post-processing).
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_model(
    backend: str, model_path: str, conf: float, slice_size: int, device: str
):
    if backend == "ultralytics":
        from sahi import AutoDetectionModel  # noqa: PLC0415

        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=conf,
            image_size=slice_size,
            device=device,
        )
    if backend == "hf_detr":
        from transformers import (  # noqa: PLC0415
            AutoImageProcessor,
            AutoModelForObjectDetection,
        )

        from detector_leaderboard.sahi_hf import HFDetrSahiModel  # noqa: PLC0415

        model = AutoModelForObjectDetection.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        return HFDetrSahiModel(
            model=model, processor=processor, confidence_threshold=conf, device=device
        )
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAHI sliced inference -> predictions")
    parser.add_argument("--backend", choices=["ultralytics", "hf_detr"], required=True)
    parser.add_argument(
        "--model-path", type=str, required=True, help="YOLO .pt file or HF model dir"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    images = sorted((args.data_dir / "images").glob("*.jpg"))
    logger.info("Found %d images in %s/images", len(images), args.data_dir)

    det = _build_model(
        args.backend,
        args.model_path,
        args.confidence_threshold,
        args.slice_size,
        args.device,
    )

    results = []
    for image_path in tqdm(images, desc=f"SAHI ({args.backend})"):
        width, height = Image.open(image_path).size
        prediction = get_sliced_prediction(
            str(image_path),
            det,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            verbose=0,
        )
        detections = []
        for obj in prediction.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            detections.append(
                {
                    "class_id": 0,  # single class: smoke (matching is class-agnostic)
                    "cx": (x1 + x2) / 2 / width,
                    "cy": (y1 + y2) / 2 / height,
                    "w": (x2 - x1) / width,
                    "h": (y2 - y1) / height,
                    "confidence": float(obj.score.value),
                }
            )
        results.append({"frame_id": image_path.stem, "detections": detections})

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results, indent=2))
    logger.info("Saved %d frame predictions to %s", len(results), args.output_file)


if __name__ == "__main__":
    main()
