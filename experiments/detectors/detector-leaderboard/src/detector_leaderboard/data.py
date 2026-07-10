"""Frame discovery and YOLO label parsing for a flat YOLO split.

Walks a split directory with the layout::

    <split>/
      images/*.jpg
      labels/*.txt

A frame's ground truth is read directly from its label file; an empty or
missing label file means a **background** frame (no smoke). Matching is
class-agnostic, so the parsed ``class_id`` is retained but not used to gate
matches.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .types import Detection


@dataclass
class FrameRef:
    """One image plus its ground-truth boxes.

    Attributes:
        stem: Filename without extension (matches the inference ``frame_id``).
        image_path: Path to the ``.jpg`` file.
        label_path: Path to the matching ``.txt`` file (may not exist).
        gt_boxes: Parsed GT boxes; empty for a background frame.
    """

    stem: str
    image_path: Path
    label_path: Path
    gt_boxes: list[Detection] = field(default_factory=list)


def parse_yolo_label(label_path: Path) -> list[Detection]:
    """Parse a YOLO ``.txt`` label file into ground-truth :class:`Detection`s.

    Returns an empty list if the file is missing or empty. Each non-blank line
    is ``class cx cy w h`` (extra columns are ignored). Ground-truth boxes are
    assigned ``confidence = 1.0``.
    """
    if not label_path.is_file():
        return []
    boxes: list[Detection] = []
    for raw_line in label_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        # class_id may be encoded as a float (e.g. "0.0"); accept both.
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:5])
        boxes.append(
            Detection(class_id=class_id, cx=cx, cy=cy, w=w, h=h, confidence=1.0)
        )
    return boxes


def list_frame_images(split_dir: Path) -> list[Path]:
    """Return ``.jpg`` paths under ``split_dir/images/``, sorted by name."""
    images_dir = Path(split_dir) / "images"
    if not images_dir.is_dir():
        return []
    return sorted(images_dir.glob("*.jpg"))


def iter_frames(split_dir: Path) -> Iterator[FrameRef]:
    """Yield one :class:`FrameRef` per image, in filename-sorted order.

    A missing ``labels/`` directory is tolerated (all frames are treated as
    background).
    """
    labels_dir = Path(split_dir) / "labels"
    for image_path in list_frame_images(split_dir):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        yield FrameRef(
            stem=stem,
            image_path=image_path,
            label_path=label_path,
            gt_boxes=parse_yolo_label(label_path),
        )
