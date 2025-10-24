#!/usr/bin/env python
"""
Manual training script with explicit callback initialization.
This script provides full control over training setup without using Lightning CLI.

Usage:
    # Train with PyTorch Lightning
    python src/train/train.py lightning \\
        --data-config configs/data_config.yaml \\
        --trainer-config configs/trainer_config.yaml \\
        --model-config configs/model_config.yaml
    
    # Train with XGBoost
    python src/train/train.py xgboost \\
        --data-config configs/data_config.yaml \\
        --model-config configs/model_config.yaml
    
    # XGBoost hyperparameter tuning
    python src/train/train.py xgboost --tune \\
        --data-config configs/data_config.yaml \\
        --model-config configs/model_config.yaml \\
        --sweep-config configs/xgboost_sweep.yaml
"""

import importlib
import sys
import os
import argparse
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

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
import wandb


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnet import CNet
from src.models.xgboost_trainer import XGBoostCellCycleTrainer
from src.data_pipeline.dataset import ModularCellDataModule, ModularCellFeaturesDataset

# Set matmul precision for Tensor Cores
matmul_precision = os.getenv("MATMUL_PRECISION", "high")
torch.set_float32_matmul_precision(matmul_precision)
print(f"✓ Set float32 matmul precision to: {matmul_precision}")

FILE_DIR_PATH = Path(__file__).resolve().parent

MODELS = {
    "CNet": CNet,
}


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train/evaluate cell cycle prediction models with selectable backends",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Top-level subparsers for backend selection
    subparsers = parser.add_subparsers(dest="backend", required=True)

    # Common parser for both backends to hold config file args
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the train config YAML file",
    )

    # Lightning backend
    lightning_parser = subparsers.add_parser(
        "lightning",
        parents=[common_parser],
        help="Train with PyTorch Lightning and evaluate on validation/test sets",
    )

    # XGBoost backend
    xgb_parser = subparsers.add_parser(
        "xgboost",
        parents=[common_parser],
        help="Train with XGBoost and evaluate on validation/test sets",
    )

    # Add tune flag for XGBoost hyperparameter sweep
    xgb_parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter sweep with W&B instead of single training run",
    )
    xgb_parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help="Path to W&B sweep configuration YAML (used with --tune)",
    )
    xgb_parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of sweep trials to run (used with --tune, default: run until stopped)",
    )
    xgb_parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name",
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

    if "model" in model_config and "init_args" in model_config:
        params = {**model_config["init_args"]}
        model = model_config["model"]
    else:
        params = {}
        model = "CNet"

    model = MODELS[model](**params)
    return model


def run_lightning(config: Dict):
    """Run Lightning training with automatic validation and test evaluation.

    Args:
        data_config: Path to data configuration YAML
        trainer_config_path: Path to trainer configuration YAML
        model_config_path: Path to model configuration YAML
    """
    # load configs
    trainer_config = config.get("trainer", {})
    model_config = config.get("model", {})
    data_config_path = config.get("data", {}).get("data_config_path", "")

    print("=" * 80 + "\n")

    seed = trainer_config.get("seed", 42)
    experiment_name = trainer_config.get("experiment_name", "default_experiment")

    # Set seed for reproducibility
    L.seed_everything(seed, workers=True)

    # Initialize callbacks
    callbacks = [
        resolve_import_name(cb["class_path"])(**cb.get("init_args", {}))
        for cb in trainer_config.pop("callbacks", [])
    ]

    # Initialize logger
    logger = WandbLogger(
        project=experiment_name,
        log_model="all",
    )

    # Create model and datamodule
    model = create_model(model_config)

    datamodule = ModularCellDataModule(
        data_config_path=data_config_path,
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


###################################################################
######                  XGBoost part                         ######
###################################################################


def xgboost_training_setup(
    data_config: str,
    wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
):
    """Initialize trainer with data configuration.

    Args:
        data_config_path: Path to data configuration YAML
        use_wandb: Whether to log to Weights & Biases
    """

    dataset = ModularCellFeaturesDataset(data_config=data_config)
    train_df, test_df = dataset.split_train_test_set()
    fucci_scalers = dataset.fucci_scaler
    # Allow override
    train_val_split_ratio = data_config.get("train_val_split_ratio", 0.8)

    trainer = XGBoostCellCycleTrainer(
        training_data=train_df,
        wandb_run=wandb_run,
        train_val_split_ratio=train_val_split_ratio,
        fucci_scalers=fucci_scalers,
    )
    return trainer, train_df, test_df


def train_and_evaluate_xgboost(
    params: Optional[Dict[str, Any]],
    trainer: XGBoostCellCycleTrainer,
    wandb_run: wandb.sdk.wandb_run.Run,
    model_suffix: str = "",
    save_model: bool = True,
    test_df=None,
):
    """
    Unified XGBoost training and evaluation logic for both normal and sweep modes.
    """

    if wandb_run:
        trainer.set_wandb_run(wandb_run)
    print("\nTraining XGBoost model...")
    model, metrics = trainer.train(params=params)

    model_path = None
    if save_model:
        model_path = f"checkpoints/xgboost/xgboost{model_suffix}_val_r2_{metrics['val_r2']:.4f}_val_mae_{metrics['val_mae']:.4f}.json"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model_path)

    test_X, test_y = None, None
    if test_df is not None:
        test_X, test_y = ModularCellFeaturesDataset.split_X_y(test_df)
        test_metrics = trainer.evaluate(test_X, test_y, prefix="test")
        print(f"\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        test_metrics = None

    if model_path:
        print(f"\n✓ Model saved to {model_path}")
    # Log model as artifact
    artifact = wandb.Artifact(
        name=f"xgboost-model-{wandb_run.id}",
        type="model",
        description=f"XGBoost model from sweep trial {wandb_run.name}",
    )
    artifact.add_file(str(model_path))
    wandb_run.log_artifact(artifact)
    return trainer, metrics, test_metrics, model_path


def run_xgboost(config: dict, project_name: str):
    """Run XGBoost training/validation/testing.

    Args:
        data_config: Data configuration dictionary
        model_config: Model/hyperparameter configuration dictionary
    """

    data_config = config.get("data_config", {})
    model_config = config.get("model_config", {})
    run = wandb.init(project=project_name, config=config)
    trainer, train_df, test_df = xgboost_training_setup(
        data_config=data_config,
    )
    train_and_evaluate_xgboost(
        params=model_config,
        trainer=trainer,
        test_df=test_df,
        wandb_run=run,
        save_model=True,
        model_suffix=f"_sweep_{run.id}",
    )
    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)


def run_xgboost_tune(
    config: dict,
    sweep_config: dict,
    project_name: str,
    count: Optional[int] = None,
):
    """Run W&B hyperparameter sweep for XGBoost.

    This function runs the sweep trials directly in-process, making it a single
    command to run hyperparameter tuning.

    Args:
        data_config: Data configuration dictionary
        sweep_config: W&B sweep configuration dictionary
        default_model_config: Default model configuration dictionary
    """

    # Extract project name and count from sweep config
    project_name = project_name or sweep_config.get("project", "xgboost-cell-cycle-sweep")
    count = count or sweep_config.get("count", 10)

    print("\n" + "=" * 80)
    print("INITIALIZING W&B SWEEP")
    print("=" * 80)
    print(f"Project: {project_name}")
    print(f"Trial count: {count if count else 'unlimited'}")
    print("=" * 80 + "\n")

    # Load data once (reuse for all trials)
    print("Loading data (will be reused across trials)...")
    data_config = config.get("data", {})

    trainer, train_df, test_df = xgboost_training_setup(
        data_config=data_config,
    )
    print(f"✓ Loaded {len(train_df)} samples with {len(train_df.columns)} features")

    def train_trial():
        with wandb.init(project=project_name, config=config) as run:
            params = run.config
            # Use unified function
            train_and_evaluate_xgboost(
                params=params,
                trainer=trainer,
                test_df=test_df,
                wandb_run=run,
                save_model=True,
                model_suffix=f"_sweep_{run.id}",
            )

    # Initialize sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)

    print(f"\nSweep initialized: {sweep_id}")
    print("\nStarting sweep trials...\n")

    # Run sweep agent with the training function
    wandb.agent(sweep_id, function=train_trial, count=count or 10, project=project_name)

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


def main():
    """
    Entry point for the script.
    Parse command line arguments and start training.
    """
    args = parse_args()

    # Dispatch to appropriate backend
    print("\n" + "=" * 80)
    print("SELECTED MODE")
    print("=" * 80)
    print(f"Backend: {args.backend}")
    if args.backend == "xgboost" and args.tune:
        print(f"Mode: Hyperparameter Tuning")
        print(f"Sweep config:   {args.sweep_config}")
    print("=" * 80 + "\n")

    config = load_config(args.config)
    if args.backend == "lightning":
        run_lightning(config)
    elif args.backend == "xgboost":
        if args.tune:
            # Run hyperparameter sweep
            sweep_config = load_config(args.sweep_config)
            run_xgboost_tune(
                config,
                sweep_config=sweep_config,
                project_name=args.project,
                count=args.count,
            )
        else:
            # Run fit (which includes validation and test evaluation)
            project_name = args.project or "xgboost-cell-cycle"
            run_xgboost(config, project_name)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
