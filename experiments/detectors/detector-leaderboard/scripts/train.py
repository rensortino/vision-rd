"""Finetune a D-FINE detector on the smoke training split.

Single-class (smoke) finetuning from a COCO-pretrained checkpoint, with early
stopping on validation loss. Saves the best model + image processor to the
output directory (the experiment's model location).

Usage:
    uv run python scripts/train.py \
        --checkpoint ustc-community/dfine-small-coco \
        --train-dir data/01_raw/datasets/train \
        --val-dir data/01_raw/datasets/val \
        --output-dir data/06_models/dfine-small \
        --image-size 640 --epochs 60 --batch-size 8 \
        --learning-rate 1e-4 --weight-decay 1e-4 \
        --early-stop-patience 8 --seed 42
"""

import argparse
import logging
from pathlib import Path

from detector_leaderboard.training import TrainConfig, finetune

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune D-FINE on smoke data.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--early-stop-patience", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    # D-FINE paper recipe (opt-in): discriminative backbone LR + EMA.
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--no-wd-on-norm-bias", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--ema-warmup-steps", type=int, default=2000)
    # Weights & Biases logging.
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-images", action="store_true")
    parser.add_argument("--log-n-images", type=int, default=12)
    args = parser.parse_args()

    config = TrainConfig(
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        num_workers=args.num_workers,
        backbone_lr=args.backbone_lr,
        no_wd_on_norm_bias=args.no_wd_on_norm_bias,
        ema_decay=args.ema_decay,
        ema_warmup_steps=args.ema_warmup_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_images=args.log_images,
        log_n_images=args.log_n_images,
    )
    finetune(
        checkpoint=args.checkpoint,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
