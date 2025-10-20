# Data Pipeline & Training Workflow

Comprehensive guide for the modular data pipeline and CLI-based training workflow for cell cycle analysis.

## Table of Contents

1. [Overview](#overview)
2. [Data Pipeline Architecture](#data-pipeline-architecture)
3. [Quick Start](#quick-start)
4. [Data Pipeline Workflow](#data-pipeline-workflow)
5. [Training with CLI](#training-with-cli)

---

## Overview

The project provides a modular, composable pipeline for:
- Loading cell microscopy data from HDF5 files
- Applying quality filters to remove invalid cells
- Preprocessing images with transformations
- Training deep learning models with PyTorch Lightning CLI


---

## Data Pipeline Architecture

The pipeline consists of 5 modular components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline Flow                        │
└─────────────────────────────────────────────────────────────┘

1. Data Source          → H5DataSource loads from .h5 file
   ├── Channels: 405, 488, 561, bf
   ├── Segmentation: seg, nuclei_seg
   └── Metadata

2. Quality Filters      → FilteredDataSource applies filters
   ├── PlaneCountFilter (3 planes expected)
   ├── EmptySegmentationFilter (min 10 pixels)
   ├── MultipleObjectsFilter (max 1 object)
   └── CellNucleiOverlappingFilter (max ratio 1.2)

3. Transform Pipeline   → Preprocessing transformations
   ├── SelectPlanesTransform (middle/first/last/all)
   ├── NormalizeTransform (minmax/standardize)
   ├── CropTransform(padding/dimension)

4. PyTorch Dataset      → ModularCellDataset
   ├── Returns: (images, labels)
   ├── Images: Tensor (C, H, W)
   └── Labels: FUCCI intensities [488, 561]

5. DataLoader(train/validate/test)          → PyTorch DataLoader for training (wrapped in the LightningDataModule)
   └── Batched, shuffled, multi-worker, test
```

### Module Files

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| [`data_sources.py`](src/data_pipeline/data_sources.py) | Storage abstraction | `H5DataSource`, `CellData` |
| [`data_filters.py`](src/data_pipeline/data_filters.py) | Quality control | `PlaneCountFilter`, `EmptySegmentationFilter` |
| [`data_sources.py`](src/data_pipeline/data_sources.py) | Caching wrapper | `FilteredDataSource` |
| [`data_transforms.py`](src/data_pipeline/data_transforms.py) | Preprocessing | `NormalizeTransform`, `SelectPlanesTransform` |
| [`dataset.py`](src/data_pipeline/dataset.py) | PyTorch interface | `ModularCellDataset`, `ModularCellDataModule` |

---

## Quick Start

### 1. Prepare Your Data

Ensure your H5 file has the following structure:

```
file.h5
├── cell_001/
│   ├── 405/          # DAPI nuclear stain
│   ├── 488/          # FUCCI green
│   ├── 561/          # FUCCI red
│   ├── bf/           # Brightfield
│   ├── seg/          # Cell segmentation mask
│   └── nuclei_seg/   # Nuclei segmentation mask
├── cell_002/
│   └── ...
```

### 2. Basic Training

```bash
cd /myhome/iris

# Quick training with default settings
python src/train/train_with_cli.py fit \
    --config configs/train_cnet_modular.yaml

```

### 3. Monitor Training

```bash
# Launch mlflow
mlflow server

```

---

## Data Pipeline Workflow

### Step 1: Load Data Source

```python
from src.data_pipeline.data_sources import H5DataSource

# Load H5 file
data_source = H5DataSource(
    path="datapath.h5",
    channel_keys=["405", "488", "561", "bf"],
    seg_key="seg",
    nuclei_seg_key="nuclei_seg",
    plane_selection="all"  # Load all planes initially
)

print(f"Total cells: {len(data_source.get_cell_ids())}")
```

### Step 2: Apply Quality Filters

```python
from src.data_pipeline.data_filters import (
    PlaneCountFilter,
    EmptySegmentationFilter,
    MultipleObjectsFilter,
    CellNucleiOverlappingFilter,
)
from src.data_pipeline.data_sources import FilteredDataSource

# Define quality filters
filters = [
    PlaneCountFilter(expected_planes=3),           # Expect 3 z-planes
    EmptySegmentationFilter(min_pixels=10),        # At least 10 pixels
    MultipleObjectsFilter(max_objects=1),          # Single cell only
    CellNucleiOverlappingFilter(max_ratio=0.2),    # max 0.2 of nuclei mask outside of cell mask relative to whole cell mask area
]

# Apply filters with caching
filtered_source = FilteredDataSource(
    data_source=data_source,
    filters=filters,
    cache_results=True,      # Cache results in a json file
    force_refilter=False     # Use cache if available
)

# View statistics
filtered_source.statistics.print_summary()
```

**Expected Output:**
```
FILTERING STATISTICS
============================================================
Total cells:                22,677
Valid cells:                17,989
Invalid cells:              4,688

Rejection reasons:
  - nuclei_too_large              : 2,540
  - failed_segmentation           : 1,005
  - multiple_cells                : 966
  - invalid_plane_count           : 177

Retention rate:             79.3%
```

### Step 3: Create Transform Pipeline

```python
from src.data_pipeline.data_transforms import (
    NormalizeTransform,
    TransformPipeline,
)

# Define preprocessing steps
transforms = [
    NormalizeTransform(
        method="standardize",      # or "minmax"
        channel_keys=["405", "bf"]
    ),
    CropTransform(padding=20, dimension=250)
]

# Create pipeline
transform_pipeline = TransformPipeline(transforms)

# Test on a sample cell
sample_cell = filtered_source.load_cell(filtered_source.get_cell_ids()[0])
transformed = transform_pipeline(sample_cell)
print(f"Transformed channels: {list(transformed.channels.keys())}")
```

### Step 4: Create LightningDataModule and get dataloader

```python
from src.data_pipeline.dataset import ModularCellDataModule, show_images

# Create datamodule
data_module = ModularCellDataModule(
    "configs/data_config.yaml" 
)

data_module.setup()
print(f"Train dataset size: {len(data_module.train_dataloader())}")

# Get a sample
images, labels = next(iter(data_module.train_dataloader()))
print(f"Images shape: {images.shape}")  # (batch_size, 6, H, W) (3xBF images, 3x405 images)
print(f"Labels shape: {labels.shape}")  # (2,) - [488_intensity, 561_intensity]
```

---

## Training with CLI

### Configuration File Structure

**Example: `configs/train_cnet_modular.yaml`**

### Training Commands

#### Basic Training

```bash
# Train with config file
python src/train/train_with_cli.py fit \
    --config configs/train_cnet_modular.yaml
```

#### Override Parameters

```bash
# Change epochs and batch size
python src/train/train_with_cli.py fit \
    --config configs/train_cnet_modular.yaml \
    --trainer.max_epochs 150 \
    --data.init_args.batch_size 128

# Change learning rate
python src/train/train_with_cli.py fit \
    --config configs/train_cnet_modular.yaml \
    --model.init_args.learning_rate 0.0005

# Disable quality filters
python src/train/train_with_cli.py fit \
    --config configs/train_cnet_modular.yaml \
    --data.init_args.use_quality_filters false
```

#### Resume Training

```bash
# Resume from last checkpoint
python src/train/train_with_cli.py fit \
    --config configs/train_cnet_modular.yaml \
    --ckpt_path logs/cnet_modular_training/version_0/checkpoints/last.ckpt
```

#### Test Model

```bash
# Test with best checkpoint
python src/train/train_with_cli.py test \
    --config configs/train_cnet_modular.yaml \
    --ckpt_path logs/cnet_modular_training/version_0/checkpoints/best.ckpt
```

#### Make Predictions

```bash
# Predict on full dataset
python src/train/train_with_cli.py predict \
    --config configs/train_cnet_modular.yaml \
    --ckpt_path logs/cnet_modular_training/version_0/checkpoints/best.ckpt
```
