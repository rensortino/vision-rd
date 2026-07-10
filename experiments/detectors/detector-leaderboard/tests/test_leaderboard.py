"""Tests for leaderboard sorting and formatting."""

import json

from detector_leaderboard.leaderboard import format_table, sort_entries, to_json
from detector_leaderboard.types import (
    DetectionMetrics,
    LeaderboardEntry,
    ProfileMetrics,
)


def _entry(name, precision, recall, f1, image_fpr, profile=None):
    return LeaderboardEntry(
        metrics=DetectionMetrics(
            model_name=name,
            conf_threshold=0.2,
            iou_threshold=0.1,
            num_frames_with_gt=100,
            num_background_frames=100,
            box_tp=0,
            box_fp=0,
            box_fn=0,
            precision=precision,
            recall=recall,
            f1=f1,
            background_frames_fired=int(image_fpr * 100),
            image_fpr=image_fpr,
            mean_fp_per_background_frame=image_fpr,
        ),
        profile=profile,
    )


def _profile(name, backend, params_m, gflops, latency, mem, size):
    return ProfileMetrics(
        model_name=name,
        backend=backend,
        image_size=size,
        num_params_m=params_m,
        gflops=gflops,
        latency_ms=latency,
        peak_gpu_mem_mb=mem,
        device="cuda",
    )


def test_sort_by_f1_descending():
    entries = [
        _entry("low", 0.5, 0.5, 0.50, 0.2),
        _entry("high", 0.9, 0.9, 0.90, 0.1),
        _entry("mid", 0.7, 0.7, 0.70, 0.15),
    ]
    ranked = sort_entries(entries, "f1")
    assert [e.metrics.model_name for e in ranked] == ["high", "mid", "low"]


def test_sort_by_image_fpr_ascending():
    entries = [
        _entry("a", 0.5, 0.5, 0.5, 0.30),
        _entry("b", 0.5, 0.5, 0.5, 0.05),
        _entry("c", 0.5, 0.5, 0.5, 0.15),
    ]
    ranked = sort_entries(entries, "image_fpr")
    assert [e.metrics.model_name for e in ranked] == ["b", "c", "a"]


def test_format_table_has_header_and_rows():
    entries = sort_entries([_entry("model-x", 0.9, 0.8, 0.85, 0.1)], "f1")
    table = format_table(entries)
    lines = table.splitlines()
    assert lines[0].startswith("Rank")
    assert "Image FPR" in lines[0]
    assert "model-x" in lines[-1]
    assert "0.8500" in lines[-1]


def test_to_json_round_trips_all_fields():
    entries = sort_entries([_entry("model-x", 0.9, 0.8, 0.85, 0.1)], "f1")
    data = json.loads(to_json(entries))
    assert len(data) == 1
    row = data[0]
    assert row["model_name"] == "model-x"
    assert row["f1"] == 0.85
    assert row["image_fpr"] == 0.1
    assert "profile" not in row  # no profile attached
    # Reconstruct the dataclass from the serialized accuracy fields.
    DetectionMetrics(**row)


def test_profile_columns_and_merge():
    prof = _profile("yolo", "ultralytics", 9.4, 21.5, 3.2, 512.0, 1024)
    entries = sort_entries(
        [
            _entry("yolo", 0.9, 0.8, 0.85, 0.1, profile=prof),
            _entry("no-prof", 0.7, 0.7, 0.70, 0.2),  # missing profile -> "-"
        ],
        "f1",
    )

    table = format_table(entries)
    header = table.splitlines()[0]
    for col in ("Params(M)", "GFLOPs", "Latency(ms)", "GPU(MB)", "Input"):
        assert col in header

    data = json.loads(to_json(entries))
    yolo_row = next(r for r in data if r["model_name"] == "yolo")
    assert yolo_row["profile"]["num_params_m"] == 9.4
    assert yolo_row["profile"]["gflops"] == 21.5
    assert "model_name" not in yolo_row["profile"]  # de-duplicated
    noprof_row = next(r for r in data if r["model_name"] == "no-prof")
    assert "profile" not in noprof_row
