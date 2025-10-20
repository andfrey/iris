#!/usr/bin/env python

import sys
from pathlib import Path
from typing import Optional, Literal
import os

import torch
import lightning as L
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader, random_split

# Set matmul precision for Tensor Cores (A100, RTX GPUs)
# Can be controlled via environment variable: MATMUL_PRECISION=high|medium|highest
# 'high' = TF32 format, ~3x faster, minimal accuracy loss (recommended)
# 'medium' = even faster but may affect accuracy
# 'highest' = full float32 precision (slowest)
matmul_precision = os.getenv("MATMUL_PRECISION", "high")
torch.set_float32_matmul_precision(matmul_precision)
print(f"✓ Set float32 matmul precision to: {matmul_precision}")

# Add parent directory to path to import from models and src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnet import CNet

# from models.convnextv2 import ConvNeXtV2
from src.data_pipeline.dataset import ModularCellDataModule


def cli_main():
    """
    Main entry point for the CLI.
    """
    # Create CLI with custom configuration
    cli = LightningCLI(
        datamodule_class=ModularCellDataModule,
        seed_everything_default=42,
        save_config_callback=None,  # Disable automatic config saving
        trainer_defaults={
            "max_epochs": 100,
            "accelerator": "auto",
            "devices": 1,
            "log_every_n_steps": 10,
            "precision": "16-mixed",  # Use mixed precision training
        },
        subclass_mode_model=True,  # Allow subclasses of CNet (CNetDeep, CNetLite)
    )


if __name__ == "__main__":
    cli_main()
