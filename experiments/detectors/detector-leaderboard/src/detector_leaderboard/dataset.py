"""Training dataset: a flat YOLO split adapted to D-FINE / HF detection format.

Each item yields a PIL image plus COCO-format annotations, which the HF image
processor turns into ``pixel_values`` + ``labels``. All ground-truth boxes are
folded to a single class ``smoke`` (category id 0), per the dataset's
``data.yaml`` (``nc: 1``); out-of-spec class ids in the labels are relabelled to
smoke rather than dropped.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .data import list_frame_images, parse_yolo_label

SMOKE_CATEGORY_ID = 0


def yolo_to_coco_bbox(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> list[float]:
    """Convert a normalized YOLO box to an absolute COCO ``[x, y, w, h]`` box.

    Coordinates are clipped to the image bounds.
    """
    bw = w * img_w
    bh = h * img_h
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    x = max(0.0, min(x, img_w))
    y = max(0.0, min(y, img_h))
    bw = max(0.0, min(bw, img_w - x))
    bh = max(0.0, min(bh, img_h - y))
    return [x, y, bw, bh]


class SmokeDetectionDataset(Dataset):
    """Flat YOLO split as (PIL image, COCO annotations) pairs for HF detection.

    Args:
        split_dir: Directory containing ``images/`` and ``labels/``.
    """

    def __init__(self, split_dir: Path) -> None:
        self.split_dir = Path(split_dir)
        self.labels_dir = self.split_dir / "labels"
        self.image_paths = list_frame_images(self.split_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        boxes = parse_yolo_label(self.labels_dir / f"{image_path.stem}.txt")
        annotations = []
        for ann_id, b in enumerate(boxes):
            bbox = yolo_to_coco_bbox(b.cx, b.cy, b.w, b.h, img_w, img_h)
            annotations.append(
                {
                    "image_id": idx,
                    "id": ann_id,
                    "category_id": SMOKE_CATEGORY_ID,  # fold every box to smoke
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )

        return {
            "image": image,
            "image_id": idx,
            "annotations": annotations,
        }
