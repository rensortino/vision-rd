"""Multi-detector comparison and leaderboard formatting.

Each entry carries accuracy metrics (:class:`DetectionMetrics`) and, when
available, inference-efficiency metrics (:class:`ProfileMetrics`). The table and
JSON merge both so detectors can be compared on quality and cost at once.
"""

import dataclasses
import json

from .types import LeaderboardEntry

# Accuracy metrics where lower is better (sorted ascending).
_LOWER_IS_BETTER = {"image_fpr", "mean_fp_per_background_frame"}


def sort_entries(
    entries: list[LeaderboardEntry],
    primary_metric: str = "f1",
) -> list[LeaderboardEntry]:
    """Sort leaderboard entries by *primary_metric* (a ``DetectionMetrics`` field).

    Rate metrics (precision, recall, f1) are sorted descending (higher is
    better); the false-positive metrics are sorted ascending (lower is better).
    """
    reverse = primary_metric not in _LOWER_IS_BETTER

    def key(entry: LeaderboardEntry) -> float:
        val = getattr(entry.metrics, primary_metric)
        if val is None:
            return float("inf") if not reverse else float("-inf")
        return val

    return sorted(entries, key=key, reverse=reverse)


def format_table(entries: list[LeaderboardEntry]) -> str:
    """Format entries as an aligned plain-text table.

    Accuracy columns are always shown; efficiency columns (Params, GFLOPs,
    Latency, GPU mem, input size) are shown when at least one entry has a
    profile, with ``-`` for entries that lack one.
    """
    show_profile = any(e.profile is not None for e in entries)

    headers = [
        "Rank",
        "Model",
        "Precision",
        "Recall",
        "F1",
        "Image FPR",
        "Mean FP/frame",
    ]
    if show_profile:
        headers += ["Params(M)", "GFLOPs", "Latency(ms)", "GPU(MB)", "Input"]

    rows: list[list[str]] = []
    for i, entry in enumerate(entries, start=1):
        m = entry.metrics
        row = [
            str(i),
            m.model_name,
            f"{m.precision:.4f}",
            f"{m.recall:.4f}",
            f"{m.f1:.4f}",
            f"{m.image_fpr:.4f}",
            f"{m.mean_fp_per_background_frame:.2f}",
        ]
        if show_profile:
            p = entry.profile
            if p is None:
                row += ["-", "-", "-", "-", "-"]
            else:
                row += [
                    f"{p.num_params_m:.2f}",
                    f"{p.gflops:.1f}" if p.gflops is not None else "-",
                    f"{p.latency_ms:.2f}",
                    f"{p.peak_gpu_mem_mb:.0f}",
                    str(p.image_size),
                ]
        rows.append(row)

    col_widths = [
        max(len(h), *(len(row[j]) for row in rows)) for j, h in enumerate(headers)
    ]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=True))

    lines = [fmt_row(headers), "  ".join("-" * w for w in col_widths)]
    lines += [fmt_row(row) for row in rows]
    return "\n".join(lines)


def to_json(entries: list[LeaderboardEntry]) -> str:
    """Serialize entries to a JSON array, merging accuracy + efficiency fields."""
    data = []
    for entry in entries:
        row = dataclasses.asdict(entry.metrics)
        if entry.profile is not None:
            prof = dataclasses.asdict(entry.profile)
            prof.pop("model_name", None)  # already present from metrics
            row["profile"] = prof
        data.append(row)
    return json.dumps(data, indent=2)
