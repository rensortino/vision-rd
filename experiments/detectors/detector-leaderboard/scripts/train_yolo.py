"""Train a YOLO detector from scratch/COCO-init on the smoke training split.

A control experiment: train a YOLO of the *same architecture* as the production
baseline (yolo11s) on our imported train/val splits, then evaluate it through
the same pipeline. If it reproduces the pretrained model's accuracy, the
training data + pipeline are sound (and D-FINE's weakness is D-FINE-specific);
if it also underperforms, the training data is suspect.

Builds a temporary single-class YOLO dataset (images symlinked, every label's
class id forced to 0 = smoke, matching the dataset's ``nc: 1``), trains with
ultralytics, and copies the best checkpoint to ``--output-dir/best.pt``.

Usage:
    uv run python scripts/train_yolo.py \
        --arch yolo11s.pt \
        --train-dir data/01_raw/datasets/train \
        --val-dir data/01_raw/datasets/val \
        --output-dir data/06_models/yolo11s-smoke-repro \
        --imgsz 1024 --epochs 100 --batch 16 --patience 20 --seed 42
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_single_class_split(src_split: Path, dst_split: Path) -> int:
    """Symlink images and write single-class (id 0) labels into *dst_split*."""
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)
    src_labels = src_split / "labels"
    n = 0
    for img in sorted((src_split / "images").glob("*.jpg")):
        (dst_split / "images" / img.name).symlink_to(img.resolve())
        src_lbl = src_labels / f"{img.stem}.txt"
        lines = []
        if src_lbl.is_file():
            for raw in src_lbl.read_text().splitlines():
                parts = raw.split()
                if len(parts) >= 5:
                    # Force class id -> 0 (smoke); keep the box coords.
                    lines.append("0 " + " ".join(parts[1:5]))
        (dst_split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO smoke detector.")
    parser.add_argument("--arch", type=str, required=True, help="e.g. yolo11s.pt")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers. Lower (e.g. 2) avoids stochastic worker "
        "segfaults seen with YOLO26 on this CUDA stack; does not affect the "
        "trained weights (same seed/data/augmentation).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="yolo_train_"))
    try:
        data_root = scratch / "dataset"
        n_train = _build_single_class_split(args.train_dir, data_root / "train")
        n_val = _build_single_class_split(args.val_dir, data_root / "val")
        logger.info("Built single-class dataset: %d train / %d val", n_train, n_val)

        data_yaml = data_root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(data_root.resolve()),
                    "train": "train/images",
                    "val": "val/images",
                    "nc": 1,
                    "names": ["smoke"],
                }
            )
        )

        model = YOLO(args.arch)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            seed=args.seed,
            single_cls=True,
            workers=args.workers,
            project=str(scratch / "runs"),
            name="train",
            exist_ok=True,
            verbose=True,
        )

        best = scratch / "runs" / "train" / "weights" / "best.pt"
        shutil.copy(best, args.output_dir / "best.pt")
        logger.info("Saved trained model to %s", args.output_dir / "best.pt")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
