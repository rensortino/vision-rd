"""Sliced (SAHI) inference for DEIMv2 at native 640 — tiles fed with NO resize.

DEIMv2-S is a 640px model; running it on a full ~1280px frame means downscaling
to 640 (losing half the resolution). Here we slice into **640x640 tiles** and feed
each tile to the model *at native scale* — no resize, so the detector sees the
scene at full resolution across the tiles. SAHI keeps every tile full-size (it
shifts edge windows back), so tiles are exactly 640x640; any short edge tile is
zero-padded top-left (padding, not resizing, preserves pixel scale). Emits the
standard predictions.json; runs in the isolated ``.deimv2-venv``.

    .deimv2-venv/bin/python scripts/infer_sahi_deimv2.py \
        --config deimv2_repo/configs/deimv2/deimv2_s_smoke.yml \
        --checkpoint deimv2_repo/outputs/deimv2_s_smoke/best_stg2.pth \
        --data-dir data/01_raw/datasets/test \
        --output-file data/02_intermediate/deimv2-s-sahi/test_predictions.json \
        --tile 640 --overlap 0.2 --confidence-threshold 0.01
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent / "deimv2_repo"
sys.path.insert(0, str(REPO))
from engine.core import YAMLConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DeimSahiModel(DetectionModel):
    """SAHI model that runs DEIMv2 on each tile at native 640 (no resize)."""

    def __init__(self, config, checkpoint, tile, confidence_threshold, device):
        self._config = config
        self._checkpoint = checkpoint
        self._tile = tile
        self._norm = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(
            model_path=checkpoint,
            confidence_threshold=confidence_threshold,
            device=device,
            category_mapping={"0": "smoke"},
            load_at_init=True,
        )

    def check_dependencies(self):
        pass

    def load_model(self):
        cfg = YAMLConfig(self._config, resume=self._checkpoint)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        ckpt = torch.load(self._checkpoint, map_location="cpu")
        state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
        cfg.model.load_state_dict(state)
        self._net = cfg.model.deploy().to(self.device).eval()
        self._post = cfg.postprocessor.deploy()
        self.model = self._net

    def set_model(self, model):
        self.model = model

    @torch.no_grad()
    def perform_inference(self, image: np.ndarray):
        h, w = image.shape[:2]
        t = self._tile
        # Pad (not resize) to tile x tile, content top-left — preserves pixel scale.
        canvas = np.zeros((t, t, 3), dtype=image.dtype)
        canvas[: min(h, t), : min(w, t)] = image[: min(h, t), : min(w, t)]
        x = self._norm(Image.fromarray(canvas)).unsqueeze(0).to(self.device)
        orig = torch.tensor([[t, t]]).to(self.device)
        outputs = self._net(x)
        labels, boxes, scores = self._post(outputs, orig)
        self._original_predictions = (labels[0], boxes[0], scores[0])

    def _create_object_prediction_list_from_original_predictions(
        self, shift_amount_list=None, full_shape_list=None
    ):
        shift_amount_list = fix_shift_amount_list(shift_amount_list or [[0, 0]])
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]
        _labels, boxes, scores = self._original_predictions

        objects = []
        for (x1, y1, x2, y2), score in zip(
            boxes.tolist(), scores.tolist(), strict=True
        ):
            if score < self.confidence_threshold:
                continue
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="DEIMv2 SAHI (640 tiles, no resize)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    det = DeimSahiModel(
        args.config, args.checkpoint, args.tile, args.confidence_threshold, args.device
    )
    images = sorted((args.data_dir / "images").glob("*.jpg"))
    logger.info("Found %d images in %s/images", len(images), args.data_dir)

    results = []
    for image_path in tqdm(images, desc="SAHI-deimv2 (640 tiles)"):
        width, height = Image.open(image_path).size
        prediction = get_sliced_prediction(
            str(image_path),
            det,
            slice_height=args.tile,
            slice_width=args.tile,
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
