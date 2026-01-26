#!/usr/bin/env python
"""
Unified training script for cell cycle prediction using PyTorch Lightning or XGBoost.

This script provides full control over training, validation, and hyperparameter tuning for both Lightning and XGBoost backends, with explicit callback and logger initialization. XGBoost training and sweep logic is unified to avoid code duplication.

Usage:
    # Train with PyTorch Lightning (e.g., CNet or MLP)
    python src/train/train.py lightning \
        --config configs/lightning_config.yaml

    # To use the MLP model, set model.model: "MLP" and ensure your data config has add_features: true

    # Train with XGBoost
    python src/train/train.py xgboost \
        --config configs/xgboost/xgboost_config.yaml

    # XGBoost hyperparameter tuning (W&B sweep)
    python src/train/train.py xgboost --tune \
        --config configs/xgboost/xgboost_config.yaml \
        --sweep-config configs/xgboost/xgboost_sweep.yaml

Key Functions:
    - run_lightning: Standard Lightning training, validation, and test.
    - run_xgboost: Standard XGBoost training, validation, and test using unified logic.
    - run_xgboost_tune: W&B sweep for XGBoost, using unified logic for each trial.
    - train_and_evaluate_xgboost: Unified logic for XGBoost training, validation, test, and model saving. Used by both normal and sweep modes.
"""

import importlib
import sys
import os
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load .env file from current directory
    print("✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠ Could not load .env file: {e}")

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    ModelSummary,
    DeviceStatsMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from sklearn.linear_model import Lasso, Ridge
import wandb


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnet import CNet, ABSResiduals
from src.models.mlp import MLP
from src.evaluation.evaluator import Evaluator
from src.data_pipeline.dataset import (
    ModularCellDataModule,
    ModularCellFeaturesDataset,
    ModularCellImageFeatureDataset,
)

# Set matmul precision for Tensor Cores
matmul_precision = os.getenv("MATMUL_PRECISION", "high")
torch.set_float32_matmul_precision(matmul_precision)
print(f"✓ Set float32 matmul precision to: {matmul_precision}")

FILE_DIR_PATH = Path(__file__).resolve().parent

MODELS = {
    "CNet": CNet,
    "MLP": MLP,
}


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train cell cycle prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common parser for both backends to hold config file args
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the train config YAML file",
    )
    common_parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name",
    )
    common_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name",
    )

    common_parser.add_argument(
        "--holdout-exp-id",
        type=str,
        default=None,
        help="Experiment ID to hold out for testing (leave-one-experiment-out)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        abs_path = Path(config_path).resolve().parent.parent
        if abs_path.exists():
            config_path = str(abs_path / Path(config_path).name)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


##################################################################
######                  Lightning part                      ######
##################################################################


def resolve_import_name(name: str) -> Any:
    """Dynamically import a module attribute from a string name."""
    module_name, attr_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def create_model(model_config: Optional[Dict[str, Any]]):
    """
    Create and configure the model.

    Args:
        model_config: Optional dictionary with model configuration.
                     If None, uses default parameters.

    Returns:
        Initialized model
    """

    params = model_config["init_args"]
    model = model_config["model"]

    model = MODELS[model](**params)
    # model = torch.compile(model)
    return model


def run_lightning(
    config: Dict,
    experiment_name: Optional[str] = None,
    project_name: Optional[str] = None,
    holdout_exp_id: Optional[str] = None,
):
    """
    Run Lightning training with automatic validation and test evaluation.

    Args:
        config: Dictionary containing 'trainer', 'model', and 'data' config sections.
        experiment_name: Optional experiment name override
        project_name: Optional W&B project name override
        holdout_exp_id: Optional experiment ID to hold out for testing (LOEO)
    """
    # Update experiment name if doing LOEO
    if holdout_exp_id:
        experiment_name = f"holdout_{holdout_exp_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        experiment_name = experiment_name or config.get("experiment_name", "experiment")
        experiment_name = experiment_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # load configs
    trainer_config = config.get("trainer", {})
    model_config = config.get("model", {})
    data_config_path = config.get("data", {}).get("data_config_path", "")

    print("=" * 80 + "\n")

    seed = trainer_config.get("seed", 42)
    project_name = project_name or config.get("project_name", None)
    # Set seed for reproducibility
    L.seed_everything(seed, workers=True)

    # Initialize callbacks
    callbacks = [
        resolve_import_name(cb["class_path"])(**cb.get("init_args", {}))
        for cb in trainer_config.pop("callbacks", [])
    ]

    # Initialize logger
    logger = WandbLogger(
        project=project_name,
        log_model="all",
        name=experiment_name,
        save_dir=f"lightning_logs/{experiment_name}",
    )

    datamodule = ModularCellDataModule(
        data_config_path=data_config_path, hold_out_exp_id=holdout_exp_id
    )
    datamodule.setup()

    model = create_model(model_config)

    # If training MLP, ensure the datamodule is configured to provide features
    if isinstance(model, MLP):
        # Setup minimally to know dataset type without building loaders twice
        datamodule.setup()
        if not isinstance(datamodule.full_dataset, ModularCellImageFeatureDataset):
            print(
                "⚠ MLP expects feature inputs. Set `add_features: true` in your data config so the datamodule returns (images, features)."
            )

    # Configure trainer
    trainer = L.Trainer(
        **trainer_config,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
    )

    # optionally load from checkpoint if provided in trainer config
    ckpt_path = trainer_config.get("ckpt_path")
    if ckpt_path:
        try:
            model_cls = model.__class__
            model = model_cls.load_from_checkpoint(ckpt_path)
            print(f"Loaded model weights from checkpoint: {ckpt_path}")
        except Exception:
            print(f"Could not load checkpoint from {ckpt_path}; continuing with fresh model.")

    # Train model
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    logger.watch(model, log_freq=100)
    trainer.fit(model, datamodule=datamodule)

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    trainer.validate(model, datamodule=datamodule)

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST")
    print("=" * 80)
    trainer.test(model, datamodule=datamodule)

    return trainer


def main():
    """
    Entry point for the script.
    Parses command line arguments and dispatches to the selected backend.
    """
    args = parse_args()

    config = load_config(args.config)

    holdout_exp_id = getattr(args, "holdout_exp_id", None)
    run_lightning(
        config,
        project_name=project_name,
        experiment_name=experiment_name,
        holdout_exp_id=holdout_exp_id,
    )


if __name__ == "__main__":
    main()
