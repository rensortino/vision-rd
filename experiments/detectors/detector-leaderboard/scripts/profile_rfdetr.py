"""Profile RF-DETR: params, GFLOPs, latency, peak GPU memory.

Mirrors the methodology of ``detector_leaderboard.profiling`` (batch-1 forward
on a dummy tensor of the operating resolution, GFLOPs via ``FlopCounterMode``,
peak CUDA memory) so the numbers are comparable to the other detectors, and
writes the same ``profile.json`` schema the leaderboard reads. Runs in the
isolated ``.rfdetr-venv``.

    .rfdetr-venv/bin/python scripts/profile_rfdetr.py \
        --model-name rfdetr-nano \
        --checkpoint data/06_models/rfdetr-nano/checkpoint_best_total.pth \
        --output-file data/07_model_output/rfdetr-nano/profile.json \
        --resolution 1024 --warmup 10 --iters 50
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from rfdetr import RFDETRNano
from rfdetr.utilities.tensors import NestedTensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _count_params_m(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


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


def _benchmark(forward, device: str, warmup: int, iters: int) -> tuple[float, float]:
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        forward()
    if is_cuda:
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
    t0 = time.perf_counter()
    for _ in range(iters):
        forward()
    return (time.perf_counter() - t0) / iters * 1000, 0.0


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Profile RF-DETR -> profile.json")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFDETRNano.from_checkpoint(
        str(args.checkpoint), resolution=args.resolution, num_classes=1
    )
    net = model.model.model.to(device).eval()

    res = args.resolution
    dummy = torch.zeros(1, 3, res, res, device=device)
    mask = torch.zeros(1, res, res, dtype=torch.bool, device=device)

    def forward():
        return net(NestedTensor(dummy, mask))

    latency_ms, peak_mb = _benchmark(forward, device, args.warmup, args.iters)
    profile = {
        "model_name": args.model_name,
        "backend": "rfdetr",
        "image_size": res,
        "num_params_m": _count_params_m(net),
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
