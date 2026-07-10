"""Profile DEIMv2: params, GFLOPs, latency, peak GPU memory (isolated venv).

Same methodology as the other profilers (batch-1 forward at the model's operating
resolution, GFLOPs via FlopCounterMode, peak CUDA memory) so numbers are
comparable, writing the standard ``profile.json`` the leaderboard reads. DEIMv2-S
runs at its native 640 (the other models are 1024 — noted in the leaderboard).

    .deimv2-venv/bin/python scripts/profile_deimv2.py \
        --config deimv2_repo/configs/deimv2/deimv2_s_smoke.yml \
        --checkpoint deimv2_repo/outputs/deimv2_s_smoke/best_stg2.pth \
        --model-name deimv2-s \
        --output-file data/07_model_output/deimv2-s/profile.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent / "deimv2_repo"
sys.path.insert(0, str(REPO))
from engine.core import YAMLConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _measure_gflops(forward) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode  # noqa: PLC0415

        counter = FlopCounterMode(display=False)
        with torch.enable_grad(), counter:
            forward()
        return counter.get_total_flops() / 1e9
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.warning("GFLOPs measurement failed: %s", exc)
        return None


def _benchmark(forward, device, warmup, iters) -> tuple[float, float]:
    for _ in range(warmup):
        forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        forward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 1e6


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DEIMv2 -> profile.json")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = "cuda"
    cfg = YAMLConfig(args.config, resume=args.checkpoint)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    model = cfg.model.deploy().to(device).eval()

    h, w = cfg.yaml_cfg["eval_spatial_size"]
    dummy = torch.zeros(1, 3, h, w, device=device)

    def forward():
        return model(dummy)

    latency_ms, peak_mb = _benchmark(forward, device, args.warmup, args.iters)
    profile = {
        "model_name": args.model_name,
        "backend": "deimv2",
        "image_size": int(w),
        "num_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "gflops": _measure_gflops(forward),
        "latency_ms": latency_ms,
        "peak_gpu_mem_mb": peak_mb,
        "device": device,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(profile, indent=2))
    logger.info(
        "%s: %.2fM params, %s GFLOPs, %.2f ms, %.0f MB",
        args.model_name,
        profile["num_params_m"],
        f"{profile['gflops']:.1f}" if profile["gflops"] else "N/A",
        latency_ms,
        peak_mb,
    )


if __name__ == "__main__":
    main()
