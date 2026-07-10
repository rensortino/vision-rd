"""DEIMv2 inference over a flat YOLO split -> cached predictions.

Runs in the isolated ``.deimv2-venv`` against the cloned ``deimv2_repo`` (added to
``sys.path``). Builds the deploy-mode model + postprocessor from the training
config, runs each frame at the model's ``eval_spatial_size`` (640), and writes the
standard ``predictions.json`` (normalized center boxes, class folded to 0) that the
main-env evaluator scores. Mirrors ``deimv2_repo/tools/inference/torch_inf.py``.

    .deimv2-venv/bin/python scripts/infer_deimv2.py \
        --config deimv2_repo/configs/deimv2/deimv2_s_smoke.yml \
        --checkpoint deimv2_repo/outputs/deimv2_s_smoke/best_stg2.pth \
        --data-dir data/01_raw/datasets/test \
        --output-file data/02_intermediate/deimv2-s/test_predictions.json \
        --confidence-threshold 0.01
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent / "deimv2_repo"
sys.path.insert(0, str(REPO))
from engine.core import YAMLConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_model(config: str, checkpoint: str, device: str):
    cfg = YAMLConfig(config, resume=checkpoint)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = _Model().to(device).eval()
    size = cfg.yaml_cfg["eval_spatial_size"]  # [H, W]
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    return model, size, vit_backbone


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="DEIMv2 inference -> predictions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, size, vit_backbone = _load_model(args.config, args.checkpoint, args.device)
    tf = T.Compose(
        [
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if vit_backbone
            else T.Lambda(lambda x: x),
        ]
    )

    images = sorted((args.data_dir / "images").glob("*.jpg"))
    logger.info("Found %d images in %s/images", len(images), args.data_dir)

    results = []
    for image_path in tqdm(images, desc="DEIMv2"):
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        data = tf(im).unsqueeze(0).to(args.device)
        orig = torch.tensor([[w, h]]).to(args.device)
        labels, boxes, scores = model(data, orig)
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        detections = []
        for (x1, y1, x2, y2), score in zip(
            boxes.tolist(), scores.tolist(), strict=True
        ):
            if score < args.confidence_threshold:
                continue
            detections.append(
                {
                    "class_id": 0,  # single class: smoke (matching is class-agnostic)
                    "cx": (x1 + x2) / 2 / w,
                    "cy": (y1 + y2) / 2 / h,
                    "w": (x2 - x1) / w,
                    "h": (y2 - y1) / h,
                    "confidence": float(score),
                }
            )
        results.append({"frame_id": image_path.stem, "detections": detections})

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results, indent=2))
    logger.info("Saved %d frame predictions to %s", len(results), args.output_file)


if __name__ == "__main__":
    main()
