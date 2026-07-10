"""HF Transformers DETR-family loading and inference.

Generic over any ``AutoModelForObjectDetection`` checkpoint (D-FINE, RT-DETR,
LW-DETR, …) so it produces the same :class:`~detector_leaderboard.types.FrameResult`
as the ultralytics path. Boxes are converted from absolute xyxy (the processor's
post-processing output) back to normalized center form (``cx, cy, w, h``).
"""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from .types import Detection, FrameResult


def load_dfine(model_dir: Path, device: str = "cuda"):
    """Load a finetuned HF detection model + image processor in eval mode."""
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def run_inference_on_image(
    model,
    processor,
    image_path: Path,
    conf: float,
    device: str = "cuda",
) -> FrameResult:
    """Run an HF detection model on a single image and return normalized detections.

    Args:
        model: Loaded :class:`DFineForObjectDetection`.
        processor: Matching image processor.
        image_path: Path to the image file.
        conf: Confidence threshold passed to post-processing.
        device: Torch device for inference.

    Returns:
        A :class:`FrameResult` (``frame_id`` = filename stem) with normalized
        center-based detections.
    """
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs,
        target_sizes=[(img_h, img_w)],
        threshold=conf,
    )[0]

    detections = []
    for score, label_id, box in zip(
        result["scores"], result["labels"], result["boxes"], strict=True
    ):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        detections.append(
            Detection(
                class_id=int(label_id.item()),
                cx=cx,
                cy=cy,
                w=bw,
                h=bh,
                confidence=float(score.item()),
            )
        )

    return FrameResult(frame_id=image_path.stem, detections=detections)
