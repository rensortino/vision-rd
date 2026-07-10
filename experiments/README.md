# 🧪 Experiments

This directory contains R&D experiments for Pyronear wildfire smoke detection. Each experiment is a self-contained Python project with its own dependencies, data, and results.

Experiments are organized by category:

```
experiments/
├── template/                          # Starter template (copy to create new experiments)
├── temporal-models/                   # Temporal smoke detection models
│   ├── bbox-tube-temporal/            # Bbox-tube temporal classifier (timm backbone + temporal head)
│   ├── mtb-change-detection/          # YOLO + pixel-wise change detection (SlowFastMTB)
│   ├── pyro-detector-baseline/        # Production pyro-predictor wrapped as a TemporalModel baseline
│   ├── tracking-fsm-baseline/         # YOLO + IoU FSM tracker baseline
│   └── temporal-model-leaderboard/    # Evaluation & ranking of temporal models on pyro-dataset v3.0.0
├── detectors/                         # Frame-level object detector evaluation
│   └── detector-leaderboard/          # Object-detection metrics & ranking of detectors (no temporal stage)
└── data-quality/                      # Label-error audits on the pyro-dataset
    ├── sequential/                    # Sequence-level audit (TemporalModel oracle)
    └── frame-level/                   # Frame-level audit (YOLO oracle + FiftyOne)
```

### Categories

- **`temporal-models/`** — new candidate models for the server-side temporal
  verification stage (bbox tubes, FSM tracking, change detection, …).
- **`detectors/`** — evaluate the frame-level object detector on its own
  (per-frame box localization, no temporal logic) and rank detectors on a
  leaderboard.
- **`data-quality/`** — use an existing model as an *oracle* against the
  pyro-dataset ground truth and surface disagreements as FiftyOne review
  sets. `sequential/` audits per-sequence labels (`wildfire/` vs `fp/`);
  `frame-level/` audits per-frame bboxes in the flat YOLO split.

## 🚀 Starting a New Experiment

```bash
cp -r template/ <category>/<kebab-case-name>/
cd <category>/<kebab-case-name>/
```

Then:

1. Rename `src/project_name/` to `src/<snake_case_name>/`
2. Update `pyproject.toml`: name, description, dependencies, `packages = ["src/<snake_case_name>"]`, and `[tool.uv.sources]` paths (adjust `../` depth based on nesting)
3. Run `make install`

## 📁 Experiment Template Structure

```
experiments/<category>/<experiment-name>/
├── README.md              # Objective, approach, results, how to reproduce
├── pyproject.toml         # Experiment deps (uv) + hatchling build + ruff config
├── uv.lock
├── .python-version        # Pinned Python version
├── .dvc/                  # Experiment-specific DVC config
├── .dvcignore
├── Makefile               # install, lint, format, test
├── src/<package_name>/    # Source code (unique importable package name)
├── notebooks/             # Exploration notebooks
├── configs/               # Experiment configs (YAML)
├── scripts/               # Training/eval scripts
└── data/                  # Kedro-style layered data engineering
    ├── 01_raw/            # Immutable raw data as received
    ├── 02_intermediate/   # Cleaned, parsed, typed data
    ├── 03_primary/        # Domain-specific datasets ready for analysis
    ├── 04_feature/        # Engineered features for modeling
    ├── 05_model_input/    # Final train/val/test splits
    ├── 06_models/         # Trained model artifacts (weights, checkpoints)
    ├── 07_model_output/   # Predictions, scores, inference outputs
    └── 08_reporting/      # Plots, metrics, reports
```

## 🗂️ Kedro Data Layers

The `data/` directory follows [Kedro's data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention) with numbered layers that reflect the data processing pipeline:

| Layer | Purpose | Example |
|-------|---------|---------|
| `01_raw` | Immutable input data | Downloaded datasets, raw images |
| `02_intermediate` | Cleaned and parsed data | Filtered frames, parsed annotations |
| `03_primary` | Domain-specific datasets | Smoke/no-smoke labeled crops |
| `04_feature` | Engineered features | Extracted embeddings, optical flow |
| `05_model_input` | Train/val/test splits | Final tensors ready for training |
| `06_models` | Trained artifacts | Checkpoints, exported weights |
| `07_model_output` | Predictions and scores | Inference results, confidence maps |
| `08_reporting` | Communication artifacts | Plots, metrics tables, reports |

Data flows from lower to higher numbers. Raw data is never modified — all transformations produce new files in subsequent layers.

## 📏 Guidelines

See [GUIDELINES.md](GUIDELINES.md) for standards on tooling, reproducibility, and benchmarking.
