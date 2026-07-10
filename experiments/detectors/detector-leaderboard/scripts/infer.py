"""Run detector inference over a flat YOLO split (ultralytics or D-FINE).

Runs the detector on every image under ``<data-dir>/images/`` and writes a
single JSON file of per-frame detections. Inference is run at a low confidence
so all candidate boxes are cached; the operating confidence is applied later,
at evaluation time, without re-running inference.

Usage (ultralytics):
    uv run python scripts/infer.py --backend ultralytics \
        --data-dir data/01_raw/datasets/test \
        --model-path data/01_raw/models/<model>/best.pt \
        --output-file data/02_intermediate/<model>/predictions.json \
        --confidence-threshold 0.01 --iou-nms 0.5 --image-size 1024

Usage (dfine):
    uv run python scripts/infer.py --backend dfine \
        --data-dir data/01_raw/datasets/test \
        --model-dir data/06_models/<model> \
        --output-file data/02_intermediate/<model>/predictions.json \
        --confidence-threshold 0.01
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from detector_leaderboard.data import list_frame_images
from detector_leaderboard.detector import save_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _infer_ultralytics(args, images):
    from detector_leaderboard.detector import (  # noqa: PLC0415
        load_model,
        run_inference_on_image,
    )

    logger.info("Loading ultralytics model from %s", args.model_path)
    model = load_model(args.model_path)
    return [
        run_inference_on_image(
            model=model,
            image_path=img,
            conf=args.confidence_threshold,
            iou_nms=args.iou_nms,
            img_size=args.image_size,
        )
        for img in tqdm(images, desc="Inference (ultralytics)")
    ]


def _infer_dfine(args, images):
    import torch  # noqa: PLC0415

    from detector_leaderboard.dfine import (  # noqa: PLC0415
        load_dfine,
        run_inference_on_image,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading D-FINE model from %s (device=%s)", args.model_dir, device)
    model, processor = load_dfine(args.model_dir, device=device)
    return [
        run_inference_on_image(
            model=model,
            processor=processor,
            image_path=img,
            conf=args.confidence_threshold,
            device=device,
        )
        for img in tqdm(images, desc="Inference (dfine)")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector inference.")
    parser.add_argument(
        "--backend", choices=["ultralytics", "dfine", "hf_detr"], required=True
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    # ultralytics-only
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=1024)
    # dfine-only
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()

    images = list_frame_images(args.data_dir)
    logger.info("Found %d images in %s", len(images), args.data_dir / "images")

    if args.backend == "ultralytics":
        if args.model_path is None:
            parser.error("--model-path is required for backend=ultralytics")
        results = _infer_ultralytics(args, images)
    else:
        if args.model_dir is None:
            parser.error("--model-dir is required for backend=dfine")
        results = _infer_dfine(args, images)

    save_predictions(results, args.output_file)
    logger.info("Inference complete. %d frames -> %s", len(results), args.output_file)


if __name__ == "__main__":
    main()
