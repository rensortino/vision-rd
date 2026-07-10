"""Evaluate a detector at its own best operating point.

The confidence threshold is selected on the **validation** split (the value
that maximizes box F1), then the detector is scored on the **test** split at
that threshold. This makes the comparison fair across architectures whose raw
scores are calibrated differently (e.g. YOLO vs. DETR-style D-FINE), without
tuning on the test set. Writes a ``metrics.json`` consumed by the leaderboard.

Usage:
    uv run python scripts/evaluate.py \
        --model-name <model> \
        --val-predictions data/02_intermediate/<model>/val_predictions.json \
        --val-dir data/01_raw/datasets/val \
        --test-predictions data/02_intermediate/<model>/test_predictions.json \
        --test-dir data/01_raw/datasets/test \
        --output-dir data/07_model_output/<model> \
        --iou-threshold 0.1 --conf-min 0.01 --conf-max 0.95 --conf-step 0.01
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from detector_leaderboard.metrics import (
    compute_detection_metrics,
    select_best_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_grid(lo: float, hi: float, step: float) -> list[float]:
    grid = []
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        grid.append(round(lo + i * step, 6))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector at best op point.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, required=True)
    parser.add_argument("--conf-min", type=float, default=0.01)
    parser.add_argument("--conf-max", type=float, default=0.95)
    parser.add_argument("--conf-step", type=float, default=0.01)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid = _build_grid(args.conf_min, args.conf_max, args.conf_step)

    # Select the operating confidence on validation.
    best_conf, val_f1 = select_best_threshold(
        args.model_name,
        args.val_dir,
        args.val_predictions,
        args.iou_threshold,
        grid,
    )
    logger.info(
        "%s: selected conf=%.3f on val (val F1=%.4f)",
        args.model_name,
        best_conf,
        val_f1,
    )

    # Score on test at the val-selected threshold.
    metrics = compute_detection_metrics(
        model_name=args.model_name,
        data_dir=args.test_dir,
        predictions_path=args.test_predictions,
        conf_threshold=best_conf,
        iou_threshold=args.iou_threshold,
    )

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(dataclasses.asdict(metrics), indent=2))
    logger.info("Saved metrics to %s", metrics_path)
    logger.info(
        "%s @conf=%.3f: P=%.4f R=%.4f F1=%.4f | image_fpr=%.4f",
        metrics.model_name,
        metrics.conf_threshold,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.image_fpr,
    )


if __name__ == "__main__":
    main()
