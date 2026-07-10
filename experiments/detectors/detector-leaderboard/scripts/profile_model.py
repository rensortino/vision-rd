"""Profile a detector's inference efficiency.

Measures parameter count, GFLOPs, forward-pass latency per image (batch 1), and
peak GPU memory at the backend's operating input size. Writes ``profile.json``
consumed by the leaderboard stage.

Usage (ultralytics):
    uv run python scripts/profile_model.py --backend ultralytics \
        --model-name <model> --model-path data/01_raw/models/<model>/best.pt \
        --image-size 1024 --output-dir data/07_model_output/<model>

Usage (dfine):
    uv run python scripts/profile_model.py --backend dfine \
        --model-name <model> --model-dir data/06_models/<model> \
        --image-size 640 --output-dir data/07_model_output/<model>
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile detector inference cost.")
    parser.add_argument(
        "--backend", choices=["ultralytics", "dfine", "hf_detr"], required=True
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--model-path", type=Path)  # ultralytics
    parser.add_argument("--model-dir", type=Path)  # dfine
    args = parser.parse_args()

    import torch  # noqa: PLC0415

    from detector_leaderboard.profiling import (  # noqa: PLC0415
        profile_dfine,
        profile_ultralytics,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.backend == "ultralytics":
        if args.model_path is None:
            parser.error("--model-path is required for backend=ultralytics")
        metrics = profile_ultralytics(
            model_name=args.model_name,
            model_path=args.model_path,
            image_size=args.image_size,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
    else:
        if args.model_dir is None:
            parser.error("--model-dir is required for backend=dfine")
        metrics = profile_dfine(
            model_name=args.model_name,
            model_dir=args.model_dir,
            image_size=args.image_size,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "profile.json"
    profile_path.write_text(json.dumps(dataclasses.asdict(metrics), indent=2))
    logger.info("Saved profile to %s", profile_path)
    logger.info(
        "%s [%s]: params=%.2fM gflops=%s latency=%.2fms peak_gpu=%.0fMB @ %dpx",
        metrics.model_name,
        metrics.backend,
        metrics.num_params_m,
        f"{metrics.gflops:.2f}" if metrics.gflops is not None else "n/a",
        metrics.latency_ms,
        metrics.peak_gpu_mem_mb,
        metrics.image_size,
    )


if __name__ == "__main__":
    main()
