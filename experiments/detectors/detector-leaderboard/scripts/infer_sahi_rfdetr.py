"""Sliced (SAHI) inference for RF-DETR, in its isolated venv.

Mirrors ``infer_sahi.py`` for the other backends but wraps ``RFDETRNano.predict``
(which returns ``supervision`` detections). SAHI owns the slicing/merging; the
custom model below just runs RF-DETR per tile and returns calibrated scores.
Emits the standard ``predictions.json`` (normalized center boxes, class 0), so
the main-env evaluator scores it identically.

    .rfdetr-venv/bin/python scripts/infer_sahi_rfdetr.py \
        --checkpoint data/06_models/rfdetr-nano/checkpoint_best_total.pth \
        --data-dir data/01_raw/datasets/test \
        --output-file data/02_intermediate/rfdetr-nano-sahi/test_predictions.json \
        --resolution 1024 --slice-size 640 --overlap 0.2 --confidence-threshold 0.01
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image
from rfdetr import RFDETRNano
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RFDetrSahiModel(DetectionModel):
    """SAHI wrapper around a loaded RF-DETR model's ``predict``."""

    def __init__(self, rf_model, confidence_threshold: float, device: str) -> None:
        self._rf = rf_model
        super().__init__(
            model_path=None,
            model=rf_model,
            confidence_threshold=confidence_threshold,
            device=device,
            category_mapping={"0": "smoke"},
            load_at_init=False,
        )
        self.model = rf_model

    def check_dependencies(self) -> None:
        pass

    def load_model(self) -> None:
        pass

    def set_model(self, model) -> None:
        self.model = model

    def perform_inference(self, image) -> None:
        pil = Image.fromarray(image)
        self._original_predictions = self._rf.predict(
            pil, threshold=self.confidence_threshold
        )

    def _create_object_prediction_list_from_original_predictions(
        self, shift_amount_list=None, full_shape_list=None
    ) -> None:
        shift_amount_list = fix_shift_amount_list(shift_amount_list or [[0, 0]])
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]
        det = self._original_predictions

        objects = []
        for (x1, y1, x2, y2), score in zip(det.xyxy, det.confidence, strict=True):
            bbox = [max(0.0, float(x1)), max(0.0, float(y1)), float(x2), float(y2)]
            if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                continue
            objects.append(
                ObjectPrediction(
                    bbox=bbox,
                    category_id=0,
                    category_name="smoke",
                    score=float(score),
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
            )
        self._object_prediction_list_per_image = [objects]


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR SAHI inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    args = parser.parse_args()

    images = sorted((args.data_dir / "images").glob("*.jpg"))
    logger.info("Found %d images in %s/images", len(images), args.data_dir)

    rf = RFDETRNano.from_checkpoint(
        str(args.checkpoint), resolution=args.resolution, num_classes=1
    )
    rf.optimize_for_inference()
    det = RFDetrSahiModel(
        rf, confidence_threshold=args.confidence_threshold, device="cuda"
    )

    results = []
    for image_path in tqdm(images, desc="SAHI (rfdetr)"):
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
                    "class_id": 0,
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
