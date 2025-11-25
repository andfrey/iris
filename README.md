# Cell Cycle Phase Prediction Pipeline

Deep learning pipeline for predicting cell cycle phase from FUCCI microscopy data using PyTorch Lightning.

## Quick Setup
```bash
conda create -n iris_env python=3.10
conda activate iris_env
poetry install --with dev
```

## Overview

**Pipeline:** HDF5 → Quality Filters → Transforms → Phase Projection → Model → Evaluation

**Models:** CNet (CNN), MLP, ConvNeXtV2, XGBoost  
**Outputs:** 1D phase [0,1] with geodesic metrics OR 2D intensities [488, 561]  
**Evaluation:** Standard metrics (MAE, MSE, RMSE, R²) + Geodesic metrics for phase

---

## Training

```bash
# Train model using lightning (CNN)
python src/train/train.py lightning \
    --config configs/lightning/train_cnet_modular.yaml

# Train XGBoost
python src/train/train.py xgboost \
    --config configs/xgboost/xgboost_config.yaml
```

---

## Evaluation

```bash
# Comprehensive evaluation with metrics and plots
python src/evaluation/checkpoint_eval.py \
    --ckpt path/to/checkpoint.ckpt \
    --split test \
    --out logs/eval_results
```

**Outputs:**
- `metrics.json` - All metrics (standard + geodesic)
- `scatter.png`, `residual.png` - Standard plots
- `phase_scatter.png`, `geodesic_residual.png` - Phase analysis

---

## Project Structure

```
iris/
├── configs/lightning/          # Training configs
│   ├── data_config.yaml
│   └── train_cnet_modular.yaml
├── src/
│   ├── data_pipeline/          # Data loading & preprocessing
│   │   ├── data_sources.py     # H5 loading, filtering
│   │   ├── data_transforms.py  # Image transforms
│   │   ├── curve_projector.py  # 2D→1D phase projection
│   │   └── dataset.py          # PyTorch datasets
│   ├── models/                 # Model architectures
│   │   ├── base_model.py       # Base class with training logic
│   │   ├── cnet.py             # CNN
│   │   ├── mlp.py              # MLP
│   │   └── convnextv2.py       # ConvNeXt
│   ├── train/                  # Training scripts
│   │   └── train_with_cli.py   # Lightning CLI
│   └── evaluation/             # Evaluation tools
│       ├── checkpoint_eval.py  # Metrics & plots
│       ├── model_interpretation.py  # GradCAM
│       ├── evaluator.py        # Evaluation orchestrator
│       ├── metrics.py          # Metric computation
│       └── plots.py            # Plot generation
├── data/                       # HDF5 data files
└── logs/                       # Training logs & checkpoints
```

---

## Data Pipeline

**1. Load HDF5** → `H5DataSource` reads channels (405, 488, 561, bf) and segmentation masks

**2. Filter** → Remove invalid cells (multiple objects, failed segmentation, etc.)

**3. Transform** → Normalize, crop, select z-planes, remove background noise

**4. Project (optional)** → `FucciCurveProjector` maps 2D intensities → 1D phase [0,1] on manifold

**5. Train** → Models inherit from `BaseRegressionModel`, use geodesic loss for phase prediction

**6. Evaluate** → Unified evaluator computes metrics in both intensity and phase space


---

## Contributing

```bash
poetry install --with dev
pytest tests/
black src/ tests/
```