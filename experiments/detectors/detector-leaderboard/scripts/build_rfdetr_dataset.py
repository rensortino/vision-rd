"""Convert our flat YOLO splits to the Roboflow-COCO layout RF-DETR expects.

RF-DETR trains on ``dataset_dir/{train,valid,test}/_annotations.coco.json`` with
the images alongside. We map train->train, val->valid, test->test, symlink the
images, and emit a single-class ("smoke") COCO annotation file per split (every
YOLO box folded to category 1, matching ``nc: 1``).

Usage:
    uv run python scripts/build_rfdetr_dataset.py \
        --train-dir data/01_raw/datasets/train \
        --val-dir data/01_raw/datasets/val \
        --test-dir data/01_raw/datasets/test \
        --output-dir data/05_model_input/rfdetr_coco
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from detector_leaderboard.data import list_frame_images, parse_yolo_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = [{"id": 1, "name": "smoke", "supercategory": "none"}]


def _build_split(src_split: Path, dst_split: Path) -> tuple[int, int]:
    """Symlink images and write _annotations.coco.json. Returns (n_images, n_boxes)."""
    dst_split.mkdir(parents=True, exist_ok=True)
    labels_dir = src_split / "labels"

    images, annotations = [], []
    ann_id = 1
    image_paths = list_frame_images(src_split)
    for img_id, img in enumerate(image_paths, start=1):
        link = dst_split / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        with Image.open(img) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img.name, "width": w, "height": h})
        for b in parse_yolo_label(labels_dir / f"{img.stem}.txt"):
            bw, bh = b.w * w, b.h * h
            x, y = (b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,  # single class: smoke
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    (dst_split / "_annotations.coco.json").write_text(json.dumps(coco))
    return len(images), len(annotations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RF-DETR COCO dataset.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for src, name in [
        (args.train_dir, "train"),
        (args.val_dir, "valid"),
        (args.test_dir, "test"),
    ]:
        n_img, n_box = _build_split(src, args.output_dir / name)
        logger.info("%s -> %s: %d images, %d boxes", src, name, n_img, n_box)


if __name__ == "__main__":
    main()
