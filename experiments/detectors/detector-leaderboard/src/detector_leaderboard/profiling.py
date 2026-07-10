"""Inference-efficiency profiling: params, GFLOPs, latency, peak GPU memory.

Forward-pass latency is measured at batch size 1 on a dummy tensor of the
backend's operating input size (so it isolates model compute, excluding image
decoding, pre-processing and NMS/post-processing). GFLOPs are measured with
PyTorch's ``FlopCounterMode`` (consistent across both backends); custom ops that
do not decompose to counted aten ops may be under-counted, so values are
approximate and reported as ``None`` if measurement fails.
"""

import logging
import time
from pathlib import Path

import torch

from .types import ProfileMetrics

logger = logging.getLogger(__name__)


def _count_params_m(module: torch.nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def _measure_gflops(forward) -> float | None:
    """Count GFLOPs for one execution of *forward* via ``FlopCounterMode``."""
    try:
        from torch.utils.flop_counter import FlopCounterMode  # noqa: PLC0415

        counter = FlopCounterMode(display=False)
        # FlopCounterMode needs the autograd graph; force grad even if the
        # caller is under torch.no_grad() (some models assert a grad fn).
        with torch.enable_grad(), counter:
            forward()
        return counter.get_total_flops() / 1e9
    except Exception as exc:  # noqa: BLE001 - FLOP counting is best-effort
        logger.warning("GFLOPs measurement failed: %s", exc)
        return None


def _benchmark(forward, device: str, warmup: int, iters: int) -> tuple[float, float]:
    """Time *forward* (a zero-arg callable) and capture peak GPU memory.

    Returns ``(latency_ms_per_iter, peak_gpu_mem_mb)``.
    """
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        forward()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    if is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            forward()
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / iters
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            forward()
        latency_ms = (time.perf_counter() - t0) / iters * 1000
        peak_mb = 0.0
    return latency_ms, peak_mb


@torch.no_grad()
def profile_ultralytics(
    model_name: str,
    model_path: Path,
    image_size: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> ProfileMetrics:
    """Profile an ultralytics YOLO ``.pt`` model."""
    from ultralytics import YOLO  # noqa: PLC0415

    yolo = YOLO(str(model_path))
    module = yolo.model.to(device).eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)

    latency_ms, peak_mb = _benchmark(lambda: module(dummy), device, warmup, iters)
    return ProfileMetrics(
        model_name=model_name,
        backend="ultralytics",
        image_size=image_size,
        num_params_m=_count_params_m(module),
        gflops=_measure_gflops(lambda: module(dummy)),
        latency_ms=latency_ms,
        peak_gpu_mem_mb=peak_mb,
        device=device,
    )


@torch.no_grad()
def profile_dfine(
    model_name: str,
    model_dir: Path,
    image_size: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> ProfileMetrics:
    """Profile a finetuned D-FINE model."""
    from .dfine import load_dfine  # noqa: PLC0415

    model, _ = load_dfine(model_dir, device=device)
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)

    latency_ms, peak_mb = _benchmark(
        lambda: model(pixel_values=dummy), device, warmup, iters
    )
    return ProfileMetrics(
        model_name=model_name,
        backend="dfine",
        image_size=image_size,
        num_params_m=_count_params_m(model),
        gflops=_measure_gflops(lambda: model(pixel_values=dummy)),
        latency_ms=latency_ms,
        peak_gpu_mem_mb=peak_mb,
        device=device,
    )
