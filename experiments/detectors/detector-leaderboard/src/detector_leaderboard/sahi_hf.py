"""A SAHI ``DetectionModel`` for HF DETR-family detectors.

SAHI's built-in ``huggingface`` wrapper mis-scores D-FINE/LW-DETR/RT-DETR — every
box comes back at score 1.0 (it does not apply the model's score transform), so
threshold selection becomes meaningless. This wrapper instead reuses the *same*
``AutoImageProcessor.post_process_object_detection`` path our non-sliced
inference uses, so per-slice scores are calibrated identically. SAHI still owns
the slicing, coordinate shifting and box merging (NMM).
"""

import numpy as np
import torch
from PIL import Image
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


class HFDetrSahiModel(DetectionModel):
    """Wrap a finetuned ``AutoModelForObjectDetection`` + processor for SAHI."""

    def __init__(
        self,
        model,
        processor,
        confidence_threshold: float = 0.05,
        device: str = "cuda",
        category_mapping: dict | None = None,
    ) -> None:
        self._processor = processor
        super().__init__(
            model_path=None,
            model=model,
            confidence_threshold=confidence_threshold,
            device=device,
            category_mapping=category_mapping or {"0": "smoke"},
            load_at_init=False,
        )
        self.set_model(model)

    def check_dependencies(self) -> None:  # deps already imported
        pass

    def load_model(self) -> None:  # model is injected, nothing to load
        pass

    def set_model(self, model) -> None:
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def perform_inference(self, image: np.ndarray) -> None:
        """Run the detector on one (slice) image; cache post-processed output."""
        pil = Image.fromarray(image)
        h, w = image.shape[:2]
        inputs = self._processor(images=[pil], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        self._original_predictions = self._processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=self.confidence_threshold
        )[0]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ) -> None:
        shift_amount_list = fix_shift_amount_list(shift_amount_list or [[0, 0]])
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]
        post = self._original_predictions

        object_predictions = []
        boxes = post["boxes"].cpu().numpy()
        scores = post["scores"].cpu().numpy()
        labels = post["labels"].cpu().numpy()
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels, strict=True):
            if float(score) < self.confidence_threshold:
                continue
            bbox = [max(0.0, float(x1)), max(0.0, float(y1)), float(x2), float(y2)]
            if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                continue
            object_predictions.append(
                ObjectPrediction(
                    bbox=bbox,
                    category_id=int(label),
                    category_name=self.category_mapping[str(int(label))],
                    score=float(score),
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
            )
        self._object_prediction_list_per_image = [object_predictions]
