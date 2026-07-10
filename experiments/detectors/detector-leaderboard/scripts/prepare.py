"""Download YOLO model weights from HuggingFace Hub.

Fetches a single model file from a HuggingFace repository and saves it
to the local data directory. Skips the download if the file already exists.
The test dataset images are imported separately via ``dvc import``.

Usage:
    uv run python scripts/prepare.py \
        --model-repo pyronear/yolo11s_nimble-narwhal_v6.0.0 \
        --model-filename best.pt \
        --output-model-dir data/01_raw/models/yolo11s-nimble-narwhal-v6.0.0
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO model.")
    parser.add_argument(
        "--model-repo",
        type=str,
        required=True,
        help="HuggingFace model repo ID.",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Model filename in the HuggingFace repo.",
    )
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        required=True,
        help="Output directory for model weights.",
    )
    args = parser.parse_args()

    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    dst_model = args.output_model_dir / args.model_filename
    if dst_model.exists():
        logger.info("Model already exists at %s, skipping.", dst_model)
    else:
        logger.info("Downloading %s from %s...", args.model_filename, args.model_repo)
        hf_hub_download(
            repo_id=args.model_repo,
            filename=args.model_filename,
            local_dir=args.output_model_dir,
        )
        logger.info("Model saved to %s", dst_model)


if __name__ == "__main__":
    main()
