"""Finetune RF-DETR (Nano) on the smoke dataset.

RF-DETR lives in its own ecosystem (the ``rfdetr`` package) with a private,
pinned dependency set, so it runs in an isolated virtualenv (``.rfdetr-venv``)
and is invoked as a standalone script rather than through the main package.

It trains on the Roboflow-COCO layout produced by ``build_rfdetr_dataset.py``
(``dataset_dir/{train,valid,test}/_annotations.coco.json``). Checkpoint
selection is RF-DETR's own validation **mAP** tracker (``val/mAP_50_95``,
mode=max), which writes ``checkpoint_best_total.pth`` to ``--output-dir`` -- a
proper detection metric, unlike the val-*loss* selection that crippled the HF
DETRs here.

Run with the isolated interpreter, e.g.::

    .rfdetr-venv/bin/python scripts/train_rfdetr.py \
        --dataset-dir data/05_model_input/rfdetr_coco \
        --output-dir data/06_models/rfdetr-nano \
        --resolution 1024 --epochs 100 --batch-size 2 --grad-accum-steps 8
"""

import argparse
import logging
from pathlib import Path

from rfdetr import RFDETRNano

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune RF-DETR Nano on smoke.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Single-class detection: the COCO classification head is re-initialized to
    # one category ("smoke"). resolution must be divisible by 32 (backbone
    # stride); 1024 matches the YOLO/D-FINE baselines for a like-for-like input.
    model = RFDETRNano(resolution=args.resolution, num_classes=1)

    logger.info(
        "Training RF-DETR Nano: res=%d epochs=%d batch=%d x accum=%d (eff=%d)",
        args.resolution,
        args.epochs,
        args.batch_size,
        args.grad_accum_steps,
        args.batch_size * args.grad_accum_steps,
    )

    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        resolution=args.resolution,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stopping=True,
        early_stopping_patience=args.early_stop_patience,
        # The pretrained model's epoch-0 mAP can otherwise dominate selection
        # before training adapts to the single-class smoke task.
        skip_best_epochs=3,
        tensorboard=False,
        wandb=False,
    )

    best = args.output_dir / "checkpoint_best_total.pth"
    logger.info("Training complete; best=%s exists=%s", best, best.exists())


if __name__ == "__main__":
    main()
