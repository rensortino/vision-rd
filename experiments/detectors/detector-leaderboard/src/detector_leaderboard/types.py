"""Core data types for the detector-leaderboard pipeline.

Evaluation is **frame-level**: the detector is run on each image of a flat YOLO
split independently. All bounding-box coordinates use a normalized center-based
format (``cx, cy, w, h`` in [0, 1]).
"""

from dataclasses import dataclass


@dataclass
class Detection:
    """A single detection (predicted or ground-truth) in a frame.

    Attributes:
        class_id: Integer class label.
        cx: Normalized center-x of the bounding box.
        cy: Normalized center-y of the bounding box.
        w: Normalized width of the bounding box.
        h: Normalized height of the bounding box.
        confidence: Detection confidence score in [0, 1] (1.0 for ground truth).
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float


@dataclass
class FrameResult:
    """All detections produced by a detector for a single image.

    Attributes:
        frame_id: Unique identifier — the image filename stem.
        detections: Detections found in this frame (may be empty).
    """

    frame_id: str
    detections: list[Detection]


@dataclass
class DetectionMetrics:
    """Aggregated frame-level object-detection metrics for a single detector.

    Box-level precision/recall/F1 are computed on frames that carry ground-truth
    boxes. The false-positive rate is computed on background frames (empty/no GT
    label), where any detection counts as a false positive.

    Attributes:
        model_name: Human-readable detector identifier (registry key).
        conf_threshold: Operating confidence threshold applied at evaluation.
        iou_threshold: IoU threshold for matching a prediction to a GT box.
        num_frames_with_gt: Frames with at least one GT box (used for P/R/F1).
        num_background_frames: Frames with no GT box (used for the FP-rate).
        box_tp: Predicted boxes matched to a GT box (GT frames).
        box_fp: Predicted boxes with no GT match (GT frames).
        box_fn: GT boxes with no matched prediction (GT frames).
        precision: ``box_tp / (box_tp + box_fp)``.
        recall: ``box_tp / (box_tp + box_fn)``.
        f1: Harmonic mean of precision and recall.
        background_frames_fired: Background frames with at least one detection.
        image_fpr: ``background_frames_fired / num_background_frames`` (lower is
            better).
        mean_fp_per_background_frame: Mean detections per background frame.
    """

    model_name: str
    conf_threshold: float
    iou_threshold: float
    num_frames_with_gt: int
    num_background_frames: int
    box_tp: int
    box_fp: int
    box_fn: int
    precision: float
    recall: float
    f1: float
    background_frames_fired: int
    image_fpr: float
    mean_fp_per_background_frame: float


@dataclass
class ProfileMetrics:
    """Inference-efficiency metrics for a single detector.

    Attributes:
        model_name: Detector identifier (registry key).
        backend: ``"ultralytics"`` or ``"dfine"``.
        image_size: Square input size used for profiling.
        num_params_m: Parameter count in millions.
        gflops: Estimated GFLOPs at ``image_size`` (``None`` if estimation failed).
        latency_ms: Forward-pass latency per image (batch 1), in milliseconds.
        peak_gpu_mem_mb: Peak GPU memory during a forward pass, in MB.
        device: Device the profile was measured on.
    """

    model_name: str
    backend: str
    image_size: int
    num_params_m: float
    gflops: float | None
    latency_ms: float
    peak_gpu_mem_mb: float
    device: str


@dataclass
class LeaderboardEntry:
    """One row of the leaderboard: a detector with its accuracy + efficiency."""

    metrics: DetectionMetrics
    profile: ProfileMetrics | None = None
