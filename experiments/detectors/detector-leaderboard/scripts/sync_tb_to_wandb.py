"""Live-mirror a TensorBoard event dir into a Weights & Biases run.

DEIMv2 (external repo) logs training to TensorBoard, not W&B. This tails its
event files and streams every scalar into a W&B run in the same project, so DEIM
shows up alongside the HF-Trainer runs without touching (or restarting) the
training process. Scalars are replayed in ascending global-step order so W&B's
step axis stays monotonic.

    uv run python scripts/sync_tb_to_wandb.py \
        --logdir deimv2_repo/outputs/deimv2_s_smoke/summary \
        --project detector-leaderboard --run-name deimv2-s \
        --done-marker /tmp/deimv2_chain.log --poll-seconds 60
"""

import argparse
import logging
import re
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import wandb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _collect(logdir: str) -> dict[int, dict[str, float]]:
    """Return {step: {tag: value}} for every scalar in the event dir."""
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    by_step: dict[int, dict[str, float]] = {}
    for tag in ea.Tags().get("scalars", []):
        for event in ea.Scalars(tag):
            by_step.setdefault(event.step, {})[tag] = event.value
    return by_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror TensorBoard scalars to W&B")
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--done-marker",
        type=str,
        default=None,
        help="File to grep for DONE/FAILED; sync finishes one poll after it appears.",
    )
    parser.add_argument("--done-regex", type=str, default="DEIMV2 DONE|TRAIN FAILED")
    args = parser.parse_args()

    wandb.init(project=args.project, name=args.run_name, resume="never")
    last_step = -1
    finishing = False
    while True:
        by_step = _collect(args.logdir)
        for step in sorted(s for s in by_step if s > last_step):
            wandb.log(by_step[step], step=step)
            last_step = step
        logger.info("synced up to step %d (%d steps seen)", last_step, len(by_step))
        if finishing:
            break
        marker = Path(args.done_marker) if args.done_marker else None
        done = False
        if marker and marker.is_file():
            done = bool(re.search(args.done_regex, marker.read_text()))
        if done:
            logger.info("done-marker matched; one final sync then finish")
            finishing = True  # loop once more to catch trailing events
        time.sleep(args.poll_seconds)
    wandb.finish()
    logger.info("wandb sync complete (final step %d)", last_step)


if __name__ == "__main__":
    main()
