# 🎯 Detector Leaderboard

Frame-level **object-detection** evaluation and ranking of smoke detectors on pyro-dataset's flat
YOLO test split. Unlike the
[temporal-model-leaderboard](../../temporal-models/temporal-model-leaderboard/), this measures the
**raw detector** — how well a checkpoint localizes smoke boxes per frame — with **no temporal
logic**. It pits the production YOLO11s against a YOLO26 control and finetuned transformer detectors
(D-FINE, LW-DETR, RT-DETRv2, RF-DETR, DEIMv2) on identical metrics.

## Objective

Quantify a detector's per-frame localization quality before any sequence-level verification stage,
so detector checkpoints can be compared head-to-head on the same frame-level benchmark.

## Approach

Several detector families are compared on the same frame-level benchmark. Each is a registry in
`params.yaml`; all run at **1024px** (matching the production YOLO's input):

- **ultralytics** (`ultralytics_detectors`): pretrained YOLO `.pt` weights from HuggingFace Hub (the
  production smoke detector).
- **yolo_trained** (`yolo_trained_detectors`): YOLO detectors trained here on our train split with a
  single recipe (1024px, 100 epochs, batch 16, patience 20, seed 42). Includes a **YOLO11s** control
  (same arch as production — checks the data/pipeline reproduce the baseline) and **YOLO26s / YOLO26n**
  (newer arch; YOLO26s at 9.9M is the closest param match to the 9.4M production model). Each entry
  carries a `workers` count — YOLO26 uses `workers: 2` (see note below).
- **dfine** / **hf_detr** (`dfine_detectors`, `hf_detr_detectors`): HF Transformers DETR-family
  detectors — [D-FINE](https://huggingface.co/docs/transformers/model_doc/d_fine) nano/small,
  [LW-DETR](https://huggingface.co/docs/transformers/model_doc/lw_detr)-tiny, and
  [RT-DETRv2](https://huggingface.co/docs/transformers/model_doc/rt_detr_v2)-r18 — each finetuned
  from a COCO-pretrained checkpoint to single-class smoke via the shared `AutoModelForObjectDetection`
  path.
- **rfdetr** (`scripts/*_rfdetr.py`): [RF-DETR](https://github.com/roboflow/rf-detr)-Nano. RF-DETR
  ships a pinned dependency set, so it runs in an **isolated virtualenv** (`.rfdetr-venv`) as
  standalone scripts rather than through the DVC pipeline; it emits the same prediction/profile JSON
  the evaluator and leaderboard read.
- **deimv2** (`scripts/*_deimv2.py`): [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)-S (the
  9.7M variant, closest to the 9.4M baseline). Its own repo/training loop, run in an isolated
  virtualenv (`.deimv2-venv`) on the cloned `deimv2_repo`, reusing the `rfdetr_coco` COCO conversion
  (category_id 1 → `num_classes=2`). Trained with its native recipe at 640px; emits the same JSON.

Pipeline stages (DVC `foreach` over each registry):

1. **prepare** (ultralytics) — download weights. **train** / **train_hf** / **train_yolo** —
   finetune each detector on the train split.
2. **infer** — run each detector on the **test** and **val** frames at a low confidence, caching all
   candidate boxes (the operating confidence is applied at evaluation, so it can be selected without
   re-running inference).
3. **evaluate** — select each detector's confidence threshold on **val** (max F1), then score on
   **test** at it. This keeps the comparison fair across architectures whose raw scores are
   calibrated differently (YOLO scores high and peaky; the DETRs score low and diffuse).
4. **profile** — measure params, GFLOPs, latency/img, peak GPU memory (see Efficiency).
5. **leaderboard** — merge accuracy + efficiency and rank.

### DETR checkpoint selection (F1, not val loss)

For the DETR-family detectors, the validation **loss** bottoms out within a couple of epochs and
then drifts up *while detection quality keeps improving*, so selecting the checkpoint on `eval_loss`
keeps a barely-trained model. Finetuning here instead evaluates **box-level F1 on val each epoch**
(the same metric the leaderboard reports) and keeps the best-F1 checkpoint, early-stopping on F1.
RF-DETR's own trainer already selects on validation **mAP**, which is likewise a detection metric.
This selection is what makes the DETRs competitive (see Findings).

Beyond selection, the HF finetuner also supports D-FINE's **optimizer recipe** (opt-in flags on
`train.py`): a discriminative backbone learning rate (`--backbone-lr`), no weight decay on
norm/bias (`--no-wd-on-norm-bias`), and weight EMA (`--ema-decay`) — needed to get D-FINE-small to
its potential (see Findings). Training runs optionally log to **Weights & Biases** (`--wandb-project`,
`--log-images`): loss/LR, per-epoch val box-F1, and image panels of training inputs and val
predictions with GT + predicted boxes overlaid. DEIMv2 logs to TensorBoard (its framework's default);
`scripts/sync_tb_to_wandb.py` mirrors those scalars into the same W&B project.

## Metrics

Ground truth is read per frame from its YOLO label file. A frame with ≥1 GT box is a **smoke**
frame; a frame with an empty or missing label is a **background** frame (no smoke). Metrics split
accordingly:

- **Precision / Recall / F1** — box-level, on smoke frames. At the operating confidence, predicted
  boxes are greedily matched to GT boxes by descending confidence, a match requiring
  `IoU ≥ iou_threshold` (matching is class-agnostic).
- **Image FPR** — image-level false-positive rate on background frames: the fraction that have at
  least one detection at the operating confidence (lower is better).
- **Mean FP/frame** — mean number of detections per background frame.

The operating confidence is **selected per model on the val split** (max F1), so each detector is
scored at its own best point. The leaderboard ranks by **F1** descending by default.

### Efficiency

Profiled per model: parameter count, GFLOPs, forward-pass latency per image (batch 1, at the input
size), and peak GPU memory. GFLOPs use PyTorch `FlopCounterMode` (consistent across backends;
custom ops may be slightly under-counted). Latency is forward-only (excludes pre/post-processing
and NMS) on an RTX 4090.

## Data

Imported via DVC from [pyro-dataset](https://github.com/pyronear/pyro-dataset)
**v4.0.0-corrected** — the flat YOLO splits (`data/processed/yolo_{train_val,test}`):

- **test**: 2640 frames (1320 with GT smoke boxes + 1320 background)
- **train / val** (D-FINE finetuning): 14767 / 1426 frames
- Layout: `data/01_raw/datasets/{train,val,test}/{images/*.jpg, labels/*.txt}`
- Single class **smoke** (`nc: 1`); out-of-spec class ids are folded to smoke. GT read per frame
  (empty/missing label = background)

## Results

Accuracy at per-model val-selected confidence, `iou_threshold=0.1`, on the 2640-frame test set.
Detectors run at 1024px **except DEIMv2-S, whose sweet spot is its native 640px** (a 1024 retrain is
included and is slightly *worse* — see note). Efficiency measured on an RTX 4090 (latency = batch-1
forward only; GPU-clock sensitive, treat as indicative — params/GFLOPs are the reliable metrics).

| Rank | Model | conf | Precision | Recall | F1 | Image FPR | Params(M) | GFLOPs | Latency(ms) | GPU(MB) | Input |
|------|-------|------|-----------|--------|----|-----------|-----------|--------|-------------|---------|-------|
| 1 | rfdetr-nano (finetuned) | 0.40 | 0.923 | 0.846 | **0.883** | **0.039** | 31.5 | 328.7 | 15.1 | 300 | 1024 |
| 2 | yolo11s-nimble-narwhal-v6.0.0 (prod) | 0.21 | 0.853 | 0.891 | **0.872** | 0.125 | 9.4 | 55.3 | 3.0 | 219 | 1024 |
| 3 | yolo11s-smoke-repro (ours, control) | 0.22 | 0.890 | 0.852 | **0.870** | 0.131 | 9.4 | 55.3 | 3.0 | 219 | 1024 |
| 4 | lwdetr-tiny (finetuned) | 0.37 | 0.910 | 0.830 | **0.868** | 0.102 | 11.7 | 78.0 | 11.2 | 138 | 1024 |
| 5 | deimv2-s (finetuned, fair-param) | 0.40 | 0.863 | 0.864 | **0.864** | 0.065 | 9.7 | 49.3 | 8.4 | **100** | 640 |
| 6 | deimv2-s-1024 (same, retrained @1024) | 0.40 | 0.839 | 0.868 | 0.853 | 0.086 | 9.7 | 218.0 | 45.5 | 185 | 1024 |
| 7 | dfine-small-paper (finetuned, paper recipe) | 0.40 | 0.893 | 0.803 | **0.846** | 0.080 | 10.2 | 66.4 | 6.8 | 297 | 1024 |
| 8 | yolo26s-smoke (ours, fair-param match) | 0.25 | 0.832 | 0.835 | 0.834 | 0.061 | 9.9 | 58.4 | 3.9 | 221 | 1024 |
| 9 | yolo26n-smoke (ours) | 0.22 | 0.802 | 0.847 | 0.823 | 0.106 | 2.5 | 15.2 | 3.6 | 124 | 1024 |
| 10 | dfine-nano (finetuned) | 0.01 | 0.837 | 0.720 | 0.774 | 0.152 | 3.7 | 17.9 | 5.9 | 142 | 1024 |
| 11 | rtdetrv2-r18 (finetuned) | 0.01 | 0.711 | 0.780 | 0.744 | 0.170 | 20.1 | 153.4 | 7.1 | 342 | 1024 |
| 12 | dfine-small (flawed recipe, superseded) | 0.01 | 0.472 | 0.659 | 0.550 | 0.201 | 10.2 | 66.4 | 6.9 | 297 | 1024 |

Row 12 (`dfine-small`) is the flat-LR/no-EMA run kept only for the before/after in Findings; row 7
(`dfine-small-paper`) is the same model trained with D-FINE's own optimizer recipe. Row 6
(`deimv2-s-1024`) is DEIMv2-S retrained at 1024 — *worse* than its native 640 (row 5) at 4.4× the
GFLOPs, so 640 is kept as its canonical row.

### Findings

- **Data + pipeline are sound (no bug).** Our from-scratch YOLO11s control (`yolo11s-smoke-repro`,
  trained on our train split) reproduces the production model almost exactly (F1 0.870 vs 0.872).
  Since it uses ultralytics' own data loading, this clears both the training data and our pipeline.
- **The DETRs' early weakness was a checkpoint-selection bug, not the models or the data.** With
  `eval_loss`-based selection the DETRs looked broken (RT-DETRv2 F1 0.14, D-FINE-nano 0.24,
  D-FINE-small 0.01). Switching to **per-epoch val-F1 selection** (see Approach) fixed every one of
  them: RT-DETRv2 0.14 → **0.74**, D-FINE-nano 0.24 → **0.77**, D-FINE-small 0.01 → **0.55**,
  LW-DETR 0.82 → **0.87**. The architectures and COCO annotations were fine all along — DETR val
  loss simply isn't a usable proxy for detection quality and bottoms out long before F1 peaks.
- **D-FINE-small needed its own optimizer recipe, not just the selection fix.** Even after F1
  selection, D-FINE-small lagged its smaller sibling (0.55 vs nano's 0.77) — suspicious for a bigger
  model. The cause was our *flat* recipe (uniform LR 1e-4, no EMA) applied to a model D-FINE trains
  with a **discriminative LR** (head/encoder/decoder 2e-4, backbone 1e-4) plus **EMA (0.9999)**.
  Retraining with [D-FINE's own hyperparameters](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_s_coco.yml)
  (`dfine-small-paper`) lifted it **0.55 → 0.846** and cut its FPR **0.20 → 0.08** — again confirming
  the data/format were never at fault. Selection and optimizer recipe are *both* load-bearing for
  DETR finetuning. (Opt-in via `--backbone-lr / --ema-decay / --no-wd-on-norm-bias` in `train.py`.)
- **DEIMv2-S is the best fair-param detector here — at 640px.** At **9.7M params** (the closest
  match to production's 9.4M) DEIMv2-S reaches **F1 0.864** with a low **0.065** FPR at the **lowest
  compute and GPU memory** on the board (49 GFLOPs, 100 MB), running at its native 640px. Trained
  with its native recipe (Mosaic/mixup aug, EMA, mAP selection) via its own repo.
- **Bigger input is not better for DEIMv2-S — 640 is its sweet spot.** Retrained at 1024 for a
  matched-resolution comparison, it got *slightly worse* (F1 0.864 → **0.853**, FPR 0.065 → 0.086:
  recall edged up but precision fell) while GFLOPs rose **4.4×** (49 → 218) and latency **5.4×**
  (8.4 → 45.5 ms — its ViT-Tiny attention is quadratic in tokens). So unlike the tiling experiment,
  this isn't a resolution-starved model: the small ViT backbone can't turn extra pixels into
  accuracy here, and 640 is both faster and better. Its 640 row is the one to compare.
- **A finetuned transformer beats production YOLO on this benchmark.** **RF-DETR-Nano tops the board
  at F1 0.883** and, more importantly for a smoke detector, fires on only **3.9% of background
  frames** vs YOLO's 12.5% — a ~3× lower false-alarm rate at higher recall-adjusted precision. The
  cost is compute: 31.5M params / 329 GFLOPs / 15 ms, the heaviest model here.
- **LW-DETR-tiny is the most attractive accuracy/FP trade.** It matches YOLO on F1 (0.868 vs 0.872)
  at 11.7M params while cutting the false-positive rate to 0.102 (vs 0.125) — and uses the least GPU
  memory on the board (138 MB).
- **D-FINE-nano is the efficiency standout.** F1 0.774 at just **3.7M params / 17.9 GFLOPs** (the
  smallest, lowest-compute model) — ~2.5× fewer params than YOLO. A strong candidate where compute
  budget dominates.
- **At a fair param budget (~9.4–11.7M), the field is tight on F1 but the transformers win on
  false alarms.** Ranked at like-for-like size: YOLO11s control **0.870**, LW-DETR **0.868**,
  DEIMv2-S **0.864**, D-FINE-small-paper **0.846**, YOLO26s **0.834** — all within ~4 F1 points. The
  real separator for a smoke detector is the **false-positive rate**, where the newer detectors
  crush the YOLO11s control (0.131): DEIMv2-S **0.065**, D-FINE-small-paper **0.080**, YOLO26s
  **0.061**, LW-DETR **0.102**. So at equal parameters you don't gain much raw F1 over production
  YOLO, but you can roughly **halve the false-alarm rate** — the metric that matters operationally.
  RF-DETR's higher F1 (0.883) comes at 3.3× the parameters, so it isn't a like-for-like comparison.
- **YOLO26-nano punches above its weight.** At **2.5M params / 15.2 GFLOPs** (the lightest model on
  the board) it reaches F1 0.823 — beating D-FINE-nano (0.774) with fewer params, and the joint-best
  latency/memory profile. The strongest pick under a tight compute budget.
- **Efficiency / latency:** measured back-to-back in one quiet GPU window. YOLO11s is fastest
  (~3 ms). The DETRs cluster at 6–15 ms; note RT-DETRv2's 153 GFLOPs still runs in ~7 ms (it
  parallelizes well), while RF-DETR's 329 GFLOPs at 1024px dominate its 15 ms. Latency is a batch-1
  forward pass (no pre/post-processing or NMS) on an RTX 4090 and is clock-sensitive — treat
  params/GFLOPs as the reliable size/compute metrics.
- **COCO annotation format verified:** the HF DETRs consume standard COCO-detection annotations; a
  GT box round-trips through our converter + the image processor exactly, so the labels fed to every
  DETR are correct.

### Does SAHI (tiled inference) help small-object recall?

[SAHI](https://github.com/obss/sahi) slices each frame into overlapping 640px tiles, runs the
detector on every tile (plus the full frame), and merges the boxes — the standard trick for tiny
objects. We re-ran **every** model with it (`-sahi` rows in `leaderboard.txt`; scripts
`infer_sahi.py`, `infer_sahi_rfdetr.py`, wrapper `sahi_hf.py`), selecting each variant's confidence
on val exactly as for the plain runs.

**Verdict: it does not help this benchmark — 7 of 9 models got worse, and the false-positive rate
rose for 7 of 9.** Test-set F1 (Δ vs plain) and image-FPR:

| Model | plain F1 | SAHI F1 | ΔF1 | plain FPR | SAHI FPR |
|-------|---------:|--------:|----:|----------:|---------:|
| rfdetr-nano | 0.883 | 0.847 | −0.035 | 0.039 | 0.059 |
| yolo11s-nimble-narwhal (prod) | 0.872 | 0.834 | −0.038 | 0.125 | 0.152 |
| yolo11s-smoke-repro | 0.870 | 0.839 | −0.032 | 0.131 | 0.181 |
| lwdetr-tiny | 0.868 | 0.724 | **−0.144** | 0.102 | 0.107 |
| yolo26s-smoke | 0.834 | 0.844 | **+0.011** | 0.061 | 0.104 |
| yolo26n-smoke | 0.823 | 0.825 | +0.002 | 0.106 | 0.167 |
| dfine-nano | 0.774 | 0.691 | −0.082 | 0.152 | 0.183 |
| rtdetrv2-r18 | 0.744 | 0.500 | **−0.244** | 0.170 | 0.140 |
| dfine-small | 0.550 | 0.472 | −0.078 | 0.201 | 0.153 |
| deimv2-s (640 tiles, no resize) | 0.864 | 0.806 | −0.058 | 0.065 | **0.327** |

The last row is a deliberate best-case test: DEIMv2-S is native-640, so tiling a ~1280px frame into
640×640 tiles feeds it at **full resolution with no downscale and no resize** (SAHI keeps tiles
full-size). It *still* lost 0.06 F1 — and the mechanism is unmistakable: **recall barely moved
(0.864→0.846, so native-res tiling did preserve the smoke) while precision collapsed and image-FPR
went 5× (0.065→0.327)**. That pins the SAHI failure on **false-positive multiplication / lost global
context**, not on resolution: even the ideal resolution scenario can't overcome ~6 background tiles
per frame each firing on context-free cloud/haze. Input resolution was never the bottleneck here.

Why SAHI backfires here:
- **The objects aren't small *enough*.** Native frames are ~1280px with a median GT box max-side of
  ~52px (only ~26% are <32px). The detectors already run at 1024px, so smoke is resolved fine
  without tiling; SAHI's usual win (tiny objects in multi-MP images) doesn't apply.
- **Tiling destroys global context.** Smoke is large and diffuse — a 640px tile often contains an
  ambiguous grey smudge the model can only read against the full horizon. The **DETRs suffer most**
  (global attention, fixed-resolution training): RT-DETRv2 −0.24, LW-DETR −0.14. The scale-augmented
  YOLOs are the most robust (yolo26s even nudges +0.01), but not enough to matter.
- **More tiles → more false alarms.** Each background tile is another chance to fire, so image-FPR
  climbed for almost every model — the opposite of what a smoke detector wants.

Net: for this data, **input resolution beats tiling**. If small-object recall needs a further push,
raising train/inference resolution (1280/1536) is the lever to try before tiling; SAHI would only
pay off on genuinely small objects in higher-resolution imagery. (SAHI's per-frame cost is also
~7× a single forward — the `-sahi` rows omit efficiency columns for that reason.)

> Note: SAHI's built-in `huggingface` model returns every D-FINE/LW-DETR/RT-DETR box at score 1.0
> (it skips the model's score transform), which would make threshold selection meaningless. The
> `sahi_hf.py` wrapper reuses the model's real `post_process_object_detection`; verified to match the
> non-sliced inference exactly when run unsliced.

> Caveat: the in-training val-F1 selector uses a strided 500-image val subset on a coarse confidence
> grid for speed, so its reported F1 is a noisy *ranking* proxy and reads much lower than the final
> full-val number (e.g. ~0.45 vs 0.85 for the same LW-DETR checkpoint). It ranked checkpoints well
> enough to land results matching production YOLO; a finer in-training eval might recover a slightly
> better epoch.

> RF-DETR-Nano (31.5M) is well above the ~9.4M YOLO baseline — it is the **smallest** RF-DETR
> variant offered, so it is the closest available, not a param-matched comparison. **YOLO26s (9.9M)
> and DEIMv2-S (9.7M) are the fair, like-for-like param matches** to the 9.4M production model.

> DEIMv2-S is scored at its **native 640px**. We also retrained it at 1024 for a matched-resolution
> comparison (`deimv2-s-1024`): it came out **slightly worse** (0.853 vs 0.864) at 4.4× the compute,
> so 640 is not a handicap — it's the model's sweet spot, and the canonical row. Config
> `deimv2_s_smoke_1024.yml` is kept for reproducibility.

> YOLO26 training segfaults (SIGSEGV, exit 139, no traceback) stochastically at a random epoch with
> the ultralytics default of 8 dataloader workers on this CUDA stack (torch 2.12+cu130). `workers: 2`
> makes it rarer but not impossible (it still crashed once at epoch 26), so `train_yolo` runs are
> wrapped in a retry. Worker count does not affect the trained weights, so the comparison stays fair.
> YOLO11s is unaffected.

## How to Reproduce

```bash
cd experiments/detectors/detector-leaderboard
make install

# One-time: import the flat YOLO splits (images + labels). test comes from
# yolo_test; train/val from yolo_train_val. Needs pyronear dataset S3 creds.
REPO=https://github.com/pyronear/pyro-dataset
REV=v4.0.0-corrected
for mod in images labels; do
  AWS_PROFILE=pyro uv run dvc import $REPO data/processed/yolo_test/$mod/test \
    -o data/01_raw/datasets/test/$mod --rev $REV
  AWS_PROFILE=pyro uv run dvc import $REPO data/processed/yolo_train_val/$mod/train \
    -o data/01_raw/datasets/train/$mod --rev $REV
  AWS_PROFILE=pyro uv run dvc import $REPO data/processed/yolo_train_val/$mod/val \
    -o data/01_raw/datasets/val/$mod --rev $REV
done
# Collaborators with the .dvc files committed can instead just pull:
AWS_PROFILE=pyro uv run dvc pull

uv run dvc repro            # trains the DETR + YOLO-control models (GPU, multi-hour)
uv run dvc metrics show
cat data/08_reporting/leaderboard.txt
```

RF-DETR runs outside DVC, in its isolated venv (after the splits are imported):

```bash
uv venv .rfdetr-venv && uv pip install --python .rfdetr-venv/bin/python "rfdetr[train,loggers]"
uv run python scripts/build_rfdetr_dataset.py \
  --train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val \
  --test-dir data/01_raw/datasets/test --output-dir data/05_model_input/rfdetr_coco
PY=.rfdetr-venv/bin/python
$PY scripts/train_rfdetr.py --dataset-dir data/05_model_input/rfdetr_coco \
  --output-dir data/06_models/rfdetr-nano --resolution 1024 --epochs 100 \
  --batch-size 2 --grad-accum-steps 8 --early-stop-patience 20
for split in val test; do
  $PY scripts/infer_rfdetr.py --data-dir data/01_raw/datasets/$split \
    --checkpoint data/06_models/rfdetr-nano/checkpoint_best_total.pth \
    --output-file data/02_intermediate/rfdetr-nano/${split}_predictions.json --resolution 1024
done
$PY scripts/profile_rfdetr.py --model-name rfdetr-nano \
  --checkpoint data/06_models/rfdetr-nano/checkpoint_best_total.pth \
  --output-file data/07_model_output/rfdetr-nano/profile.json --resolution 1024
# evaluate (main venv, backend-agnostic) then rebuild the leaderboard:
uv run python scripts/evaluate.py --model-name rfdetr-nano \
  --val-predictions data/02_intermediate/rfdetr-nano/val_predictions.json --val-dir data/01_raw/datasets/val \
  --test-predictions data/02_intermediate/rfdetr-nano/test_predictions.json --test-dir data/01_raw/datasets/test \
  --output-dir data/07_model_output/rfdetr-nano --iou-threshold 0.1
uv run python scripts/leaderboard.py --results-dir data/07_model_output --output-dir data/08_reporting
```

D-FINE-small with the paper optimizer recipe (`dfine-small-paper`), with W&B logging:

```bash
uv run python scripts/train.py --checkpoint ustc-community/dfine-small-coco \
  --train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val \
  --output-dir data/06_models/dfine-small-paper \
  --image-size 1024 --epochs 132 --batch-size 8 \
  --learning-rate 2e-4 --backbone-lr 1e-4 --weight-decay 1e-4 --no-wd-on-norm-bias \
  --ema-decay 0.9999 --early-stop-patience 20 --seed 42 \
  --wandb-project detector-leaderboard --wandb-run-name dfine-small-paper --log-images
# then infer (backend dfine) -> evaluate -> profile -> leaderboard, as for the DVC dfine rows.
```

DEIMv2-S runs outside DVC in its own venv on the cloned repo (reuses `rfdetr_coco`):

```bash
git clone https://github.com/Intellindust-AI-Lab/DEIMv2 deimv2_repo
uv venv .deimv2-venv --python 3.11
uv pip install --python .deimv2-venv/bin/python -r deimv2_repo/requirements.txt gdown
# ViT-Tiny distilled backbone -> ckpts/ (config deimv2_s_smoke.yml expects it there):
.deimv2-venv/bin/gdown 1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs -O deimv2_repo/ckpts/vitt_distill.pt
# config configs/deimv2/deimv2_s_smoke.yml: num_classes=2, remap off, rfdetr_coco paths.
cd deimv2_repo && ../.deimv2-venv/bin/torchrun --nproc_per_node=1 train.py \
  -c configs/deimv2/deimv2_s_smoke.yml --use-amp --seed=0 && cd ..
# infer (val+test) + profile in the venv, then evaluate + leaderboard in the main env:
.deimv2-venv/bin/python scripts/infer_deimv2.py --config deimv2_repo/configs/deimv2/deimv2_s_smoke.yml \
  --checkpoint deimv2_repo/outputs/deimv2_s_smoke/best_stg2.pth \
  --data-dir data/01_raw/datasets/test \
  --output-file data/02_intermediate/deimv2-s/test_predictions.json  # + val
.deimv2-venv/bin/python scripts/profile_deimv2.py --config <cfg> --checkpoint <ckpt> \
  --model-name deimv2-s --output-file data/07_model_output/deimv2-s/profile.json
uv run python scripts/evaluate.py --model-name deimv2-s ...  # then leaderboard.py
# mirror DEIMv2's TensorBoard logs into W&B:
uv run python scripts/sync_tb_to_wandb.py --logdir deimv2_repo/outputs/deimv2_s_smoke/summary \
  --project detector-leaderboard --run-name deimv2-s
```

Optional — sliced (SAHI) inference for any model (see the SAHI finding above). Emits `<model>-sahi`
rows scored by the same evaluator:

```bash
uv pip install sahi   # main venv; and: uv pip install --python .rfdetr-venv/bin/python sahi
# YOLO (ultralytics) or HF DETR (hf_detr):
uv run python scripts/infer_sahi.py --backend ultralytics \
  --model-path data/06_models/yolo26s-smoke/best.pt \
  --data-dir data/01_raw/datasets/test \
  --output-file data/02_intermediate/yolo26s-smoke-sahi/test_predictions.json \
  --slice-size 640 --overlap 0.2 --confidence-threshold 0.01
# RF-DETR (isolated venv): scripts/infer_sahi_rfdetr.py, same flags + --checkpoint/--resolution.
# Then evaluate (val+test) with --model-name <model>-sahi and rebuild the leaderboard as above.
```

## Adding a Detector

- **YOLO (ultralytics):** add an entry under `ultralytics_detectors:` in `params.yaml`
  (`model_repo` + `model_filename` on HuggingFace Hub). For a YOLO trained here, add under
  `yolo_trained_detectors:` (an `arch` init `.pt`).
- **HF DETR (D-FINE / LW-DETR / RT-DETR / …):** add an entry under `dfine_detectors:` or
  `hf_detr_detectors:` (a COCO-pretrained `checkpoint`); it is finetuned with val-F1 selection by
  the `train` / `train_hf` stage. Any `AutoModelForObjectDetection`-compatible checkpoint works.
- **RF-DETR / DEIMv2:** run via their isolated-venv scripts above (not DVC); they emit the same
  `predictions.json` / `profile.json`, so the evaluator and leaderboard treat them like any other row.

For the DVC-backed registries, the key becomes the output dir name and the leaderboard `model_name`;
run `uv run dvc repro` — DVC's `foreach` picks up the new detector automatically. The leaderboard
auto-discovers any `data/07_model_output/<model>/metrics.json` (+ optional `profile.json`).
