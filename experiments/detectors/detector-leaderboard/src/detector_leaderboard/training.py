"""Finetune an HF DETR-family detector on the smoke dataset via the HF Trainer.

Single-class (``smoke``) detection: the COCO-pretrained classification head is
re-initialized to one label.

Checkpoint selection: for DETR detectors the validation **loss** bottoms out
within a couple of epochs and then rises while detection quality keeps
improving, so selecting on ``eval_loss`` keeps a barely-trained model. Instead a
callback evaluates the model on val each epoch with our own box-level F1
(IoU-matched, confidence swept — the same metric the leaderboard reports), saves
the best-F1 checkpoint to ``output_dir``, and early-stops on F1.
"""

import copy
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .data import iter_frames
from .dataset import SmokeDetectionDataset
from .metrics import select_best_threshold
from .serialization import save_predictions
from .types import Detection, FrameResult

logger = logging.getLogger(__name__)

ID2LABEL = {0: "smoke"}
LABEL2ID = {"smoke": 0}


@dataclass
class TrainConfig:
    """Hyperparameters for a DETR finetuning run."""

    image_size: int = 640
    epochs: int = 60
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 8
    seed: int = 42
    num_workers: int = 8
    # Per-epoch F1-selection settings.
    select_iou: float = 0.1
    select_max_val_images: int = 500
    select_conf_grid_step: float = 0.05
    # --- D-FINE paper recipe (all opt-in; None/0 -> current flat behavior) ---
    # Discriminative LR: backbone params train at ``backbone_lr`` while the rest
    # use ``learning_rate`` (D-FINE-S: 1e-4 backbone / 2e-4 base). Norm/bias
    # params get no weight decay. EMA (decay ~0.9999) tracks a moving average of
    # weights and is what D-FINE evaluates/reports.
    backbone_lr: float | None = None
    no_wd_on_norm_bias: bool = False
    ema_decay: float | None = None
    ema_warmup_steps: int = 2000
    # --- Weights & Biases logging (opt-in) ---
    # When wandb_project is set, the Trainer logs loss/LR to W&B; with log_images
    # we also log a panel of training input images (with GT boxes) once and, each
    # epoch, a val panel with GT + predicted boxes overlaid at the selected conf.
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    log_images: bool = False
    log_n_images: int = 12


def _make_collate_fn(processor: AutoImageProcessor):
    def collate_fn(batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        annotations = [
            {"image_id": b["image_id"], "annotations": b["annotations"]} for b in batch
        ]
        enc = processor(images=images, annotations=annotations, return_tensors="pt")
        out = {"pixel_values": enc["pixel_values"], "labels": enc["labels"]}
        if "pixel_mask" in enc:
            out["pixel_mask"] = enc["pixel_mask"]
        return out

    return collate_fn


@torch.no_grad()
def _predict_val(model, processor, image_paths: list[Path], device: str, batch: int):
    """Batched inference over val images -> list[FrameResult] (normalized boxes)."""
    results: list[FrameResult] = []
    for i in range(0, len(image_paths), batch):
        chunk = image_paths[i : i + batch]
        images = [Image.open(p).convert("RGB") for p in chunk]
        sizes = [(im.height, im.width) for im in images]
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        post = processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=0.01
        )
        for path, (h, w), res in zip(chunk, sizes, post, strict=True):
            dets = []
            for score, label_id, box in zip(
                res["scores"], res["labels"], res["boxes"], strict=True
            ):
                x1, y1, x2, y2 = (float(v) for v in box.tolist())
                dets.append(
                    Detection(
                        class_id=int(label_id.item()),
                        cx=(x1 + x2) / 2 / w,
                        cy=(y1 + y2) / 2 / h,
                        w=(x2 - x1) / w,
                        h=(y2 - y1) / h,
                        confidence=float(score.item()),
                    )
                )
            results.append(FrameResult(frame_id=path.stem, detections=dets))
    return results


class ModelEMA:
    """Exponential moving average of model weights (D-FINE-style, warmup ramp)."""

    def __init__(self, model, decay: float, warmup_steps: int) -> None:
        self.decay = decay
        self.warmup_steps = max(1, warmup_steps)
        self.updates = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        # Ramp the decay in so early (noisy) steps are not over-weighted.
        d = self.decay * (1 - math.exp(-self.updates / self.warmup_steps))
        msd = model.state_dict()
        for k, shadow_v in self.shadow.items():
            shadow_v.mul_(d).add_(msd[k].detach().float(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        msd = model.state_dict()
        for k, shadow_v in self.shadow.items():
            msd[k].copy_(shadow_v)


def _build_optimizer(model, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with D-FINE param groups: backbone LR + no WD on norm/bias."""
    base_lr = config.learning_rate
    backbone_lr = config.backbone_lr if config.backbone_lr is not None else base_lr
    wd = config.weight_decay
    groups: dict[str, dict] = {
        "base_decay": {"params": [], "lr": base_lr, "weight_decay": wd},
        "base_nodecay": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        "bb_decay": {"params": [], "lr": backbone_lr, "weight_decay": wd},
        "bb_nodecay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "backbone" in name
        # ndim<=1 catches norm weights + biases (the standard no-decay heuristic).
        no_decay = config.no_wd_on_norm_bias and param.ndim <= 1
        key = f"{'bb' if is_backbone else 'base'}_{'nodecay' if no_decay else 'decay'}"
        groups[key]["params"].append(param)
    param_groups = [g for g in groups.values() if g["params"]]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def _wandb_image_with_boxes(image_path, gt_boxes, pred_dets, conf_threshold):
    """Build a ``wandb.Image`` with GT (blue) and predicted (per-conf) box overlays.

    Positions use the unambiguous pixel domain. Predictions are filtered to
    ``conf_threshold`` so the overlay shows the operating point, not the 0.01 floor.
    """
    import wandb  # noqa: PLC0415

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    def _box(det, is_pred: bool) -> dict:
        entry = {
            "position": {
                "minX": (det.cx - det.w / 2) * w,
                "maxX": (det.cx + det.w / 2) * w,
                "minY": (det.cy - det.h / 2) * h,
                "maxY": (det.cy + det.h / 2) * h,
            },
            "domain": "pixel",
            "class_id": 0,
            "box_caption": f"pred {det.confidence:.2f}" if is_pred else "gt",
        }
        if is_pred:
            entry["scores"] = {"confidence": float(det.confidence)}
        return entry

    labels = {0: "smoke"}
    boxes = {
        "ground_truth": {
            "box_data": [_box(d, False) for d in gt_boxes],
            "class_labels": labels,
        },
        "predictions": {
            "box_data": [
                _box(d, True) for d in pred_dets if d.confidence >= conf_threshold
            ],
            "class_labels": labels,
        },
    }
    return wandb.Image(image, boxes=boxes)


class BestF1Callback(TrainerCallback):
    """Evaluate val box-F1 each epoch, save the best checkpoint, early-stop on F1."""

    def __init__(
        self,
        processor: AutoImageProcessor,
        val_dir: Path,
        output_dir: Path,
        config: TrainConfig,
        device: str,
        train_dir: Path | None = None,
    ) -> None:
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.config = config
        self.device = device
        self.train_dir = train_dir
        # Fixed val subset (deterministic stride keeps the wildfire/background mix).
        frames = list(iter_frames(val_dir))
        stride = max(1, len(frames) // config.select_max_val_images)
        self.val_dir = val_dir
        self.val_frames = frames[::stride]
        self.val_images = [f.image_path for f in self.val_frames]
        self.conf_grid = [
            round(0.01 + i * config.select_conf_grid_step, 4)
            for i in range(int((0.95 - 0.01) / config.select_conf_grid_step) + 1)
        ]
        self.best_f1 = -1.0
        self.epochs_no_improve = 0
        self.ema: ModelEMA | None = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.config.ema_decay is not None:
            self.ema = ModelEMA(
                model, self.config.ema_decay, self.config.ema_warmup_steps
            )
            logger.info("[ema] tracking EMA (decay=%.4f)", self.config.ema_decay)
        # Log a panel of training inputs (image + GT boxes) once, so the data fed
        # to the model can be eyeballed. wandb is already initialized by the
        # Trainer's own WandbCallback (it runs on_train_begin before this one).
        if self.config.log_images and self.train_dir is not None:
            self._log_images(
                "train/inputs",
                list(iter_frames(self.train_dir))[: self.config.log_n_images],
                preds=None,
                conf=0.0,
                step=0,
            )
        return control

    def _log_images(self, key, frames, preds, conf, step):
        try:
            import wandb  # noqa: PLC0415

            if wandb.run is None:
                return
            images = []
            for i, frame in enumerate(frames):
                dets = preds[i].detections if preds is not None else []
                images.append(
                    _wandb_image_with_boxes(
                        frame.image_path, frame.gt_boxes, dets, conf
                    )
                )
            wandb.log({key: images}, step=step)
        except Exception as exc:  # noqa: BLE001 - logging must never kill training
            logger.warning("wandb image logging failed: %s", exc)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.ema is not None:
            self.ema.update(model)
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()
        # D-FINE reports the EMA weights: evaluate/save those, then restore the
        # raw weights so training continues from the live model.
        raw_state = None
        if self.ema is not None:
            raw_state = copy.deepcopy(model.state_dict())
            self.ema.copy_to(model)
        preds = _predict_val(
            model, self.processor, self.val_images, self.device, self.config.batch_size
        )
        with tempfile.TemporaryDirectory() as td:
            pj = Path(td) / "val_pred.json"
            save_predictions(preds, pj)
            best_conf, f1 = select_best_threshold(
                "val", self.val_dir, pj, self.config.select_iou, self.conf_grid
            )
        logger.info(
            "[select] epoch %.0f: val F1=%.4f @conf=%.2f (best=%.4f)",
            state.epoch,
            f1,
            best_conf,
            max(self.best_f1, 0.0),
        )
        if self.config.log_images:
            self._log_images(
                "val/predictions",
                self.val_frames[: self.config.log_n_images],
                preds,
                best_conf,
                int(state.global_step),
            )
        if self.config.wandb_project is not None:
            try:
                import wandb  # noqa: PLC0415

                if wandb.run is not None:
                    wandb.log(
                        {"val/box_f1": f1, "val/best_conf": best_conf},
                        step=int(state.global_step),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb metric logging failed: %s", exc)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.epochs_no_improve = 0
            model.save_pretrained(self.output_dir)
            self.processor.save_pretrained(self.output_dir)
            logger.info("[select] new best F1=%.4f -> saved to %s", f1, self.output_dir)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.config.early_stop_patience:
                logger.info(
                    "[select] no F1 improvement for %d epochs; stopping",
                    self.config.early_stop_patience,
                )
                control.should_training_stop = True
        if raw_state is not None:
            model.load_state_dict(raw_state)
        model.train()
        return control


def finetune(
    checkpoint: str,
    train_dir: Path,
    val_dir: Path,
    output_dir: Path,
    config: TrainConfig,
) -> None:
    """Finetune *checkpoint* on the smoke dataset and save the best-F1 model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    size = {"height": config.image_size, "width": config.image_size}
    processor = AutoImageProcessor.from_pretrained(checkpoint, size=size, use_fast=True)

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds = SmokeDetectionDataset(train_dir)
    logger.info("train frames: %d", len(train_ds))

    trainer_dir = output_dir / "_trainer"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if config.wandb_project is not None:
        import os  # noqa: PLC0415

        os.environ["WANDB_PROJECT"] = config.wandb_project
    report_to = "wandb" if config.wandb_project is not None else "none"
    args = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # Checkpoint selection is handled by BestF1Callback on val box-F1, not
        # the Trainer (val loss is a poor proxy for DETR detection quality).
        eval_strategy="no",
        save_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,  # keep PIL images for the collate
        seed=config.seed,
        logging_steps=50,
        report_to=report_to,
        run_name=config.wandb_run_name,
    )

    # Opt into the D-FINE param-group optimizer (discriminative backbone LR +
    # no WD on norm/bias) when requested; otherwise let the Trainer build the
    # default flat-LR AdamW. The cosine scheduler is created by the Trainer for
    # whichever optimizer it is given, preserving per-group LRs.
    optimizers = (None, None)
    if config.backbone_lr is not None or config.no_wd_on_norm_bias:
        optimizers = (_build_optimizer(model, config), None)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=_make_collate_fn(processor),
        callbacks=[
            BestF1Callback(
                processor, val_dir, output_dir, config, device, train_dir=train_dir
            )
        ],
        optimizers=optimizers,
    )

    trainer.train()
    shutil.rmtree(trainer_dir, ignore_errors=True)
    logger.info("Finetuning complete; best-F1 model saved to %s", output_dir)
