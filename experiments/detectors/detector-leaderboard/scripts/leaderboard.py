"""Aggregate per-detector metrics into a ranked leaderboard.

Scans a results directory for ``<model>/metrics.json`` (accuracy) and the
optional ``<model>/profile.json`` (efficiency), ranks the detectors, and writes
``leaderboard.json`` (DVC metric) and a human-readable ``leaderboard.txt`` table.

Usage:
    uv run python scripts/leaderboard.py \
        --results-dir data/07_model_output \
        --output-dir data/08_reporting
"""

import argparse
import json
import logging
from pathlib import Path

from detector_leaderboard.leaderboard import format_table, sort_entries, to_json
from detector_leaderboard.types import (
    DetectionMetrics,
    LeaderboardEntry,
    ProfileMetrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_entry(model_dir: Path) -> LeaderboardEntry | None:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = DetectionMetrics(**json.loads(metrics_path.read_text()))

    profile = None
    profile_path = model_dir / "profile.json"
    if profile_path.is_file():
        profile = ProfileMetrics(**json.loads(profile_path.read_text()))

    return LeaderboardEntry(metrics=metrics, profile=profile)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce a leaderboard from detector evaluation results."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", type=str, default="f1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[LeaderboardEntry] = []
    for model_dir in sorted(args.results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        entry = _load_entry(model_dir)
        if entry is None:
            continue
        entries.append(entry)
        logger.info(
            "Loaded %s (profile: %s)",
            entry.metrics.model_name,
            "yes" if entry.profile else "no",
        )

    if not entries:
        logger.warning("No detector results found in %s", args.results_dir)
        return

    sorted_entries = sort_entries(entries, args.primary_metric)

    table = format_table(sorted_entries)
    (args.output_dir / "leaderboard.txt").write_text(table + "\n")
    (args.output_dir / "leaderboard.json").write_text(to_json(sorted_entries) + "\n")
    print(table)


if __name__ == "__main__":
    main()
