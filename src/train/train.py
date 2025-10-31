#!/usr/bin/env python
"""
Unified training script for cell cycle prediction using PyTorch Lightning or XGBoost.

This script provides full control over training, validation, and hyperparameter tuning for both Lightning and XGBoost backends, with explicit callback and logger initialization. XGBoost training and sweep logic is unified to avoid code duplication.

Usage:
    # Train with PyTorch Lightning
    python src/train/train.py lightning \
        --config configs/lightning_config.yaml

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

from src.models.cnet import CNet
from src.train.utils import evaluate_regression, train_val_split
from src.models.xgboost_trainer import XGBoostCellCycleTrainer
from src.data_pipeline.dataset import ModularCellDataModule, ModularCellFeaturesDataset

# Set matmul precision for Tensor Cores
matmul_precision = os.getenv("MATMUL_PRECISION", "high")
torch.set_float32_matmul_precision(matmul_precision)
print(f"✓ Set float32 matmul precision to: {matmul_precision}")

FILE_DIR_PATH = Path(__file__).resolve().parent

MODELS = {
    "CNet": CNet,
    "Ridge": Ridge,
    "Lasso": Lasso,
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

    # Linear regression backend
    linear_parser = subparsers.add_parser(
        "linear",
        parents=[common_parser],
        help="Train with Linear Regression (sklearn) and evaluate on validation/test sets",
    )

    # Experiment backend
    dataset_size_experiment = subparsers.add_parser(
        "dataset_size_experiment",
        parents=[common_parser],
        help="Run dataset size experiment",
    )
    dataset_size_experiment.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Step size for increasing dataset size in the experiment",
    )

    linear_parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter sweep with W&B instead of single training run",
    )
    linear_parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help="Path to W&B sweep configuration YAML (used with --tune)",
    )
    linear_parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of sweep trials to run (used with --tune, default: run until stopped)",
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
    return model


def run_dataset_size_experiment(config, steps):
    """
    Run experiments to evaluate the effect of dataset size on model performance.

    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment for logging
    """
    data_config = config.get("data")
    trainer_config = config.get("trainer")
    datamodule = ModularCellDataModule(
        data_config_path=data_config.get("data_config_path"),
    )
    datamodule.setup()
    len_dataset = len(datamodule.train_dataset)
    nr_dataset_batches = len(datamodule.train_dataloader())
    for dataset_size in range(steps, len_dataset, steps):
        print("\n" + "=" * 80)
        print(f"Running experiment with dataset size: {dataset_size}")
        print("=" * 80 + "\n")
        # Here you would implement the logic to train and evaluate the model
        # using only 'dataset_size' number of samples from the training set.
        # This function can be expanded based on specific experiment requirements.
        nr_batches = dataset_size // datamodule.batch_size
        dataset_batch_fraction = nr_batches / nr_dataset_batches
        print(f"Using {nr_batches} batches ({dataset_batch_fraction:.2%} of full dataset)")
        # Example: You might want to modify the datamodule or dataloader
        data_config.update({"train_dataset_size": dataset_size})
        trainer_config.update({"overfit_batches": dataset_batch_fraction})
        run_lightning(
            config,
            experiment_name=f"dataset_size_{dataset_size}",
        )

    # This function can be implemented to run experiments varying dataset sizes


def run_lightning(
    config: Dict, experiment_name: Optional[str] = None, project_name: Optional[str] = None
):
    """
    Run Lightning training with automatic validation and test evaluation.

    Args:
        config: Dictionary containing 'trainer', 'model', and 'data' config sections.
    """
    # load configs
    trainer_config = config.get("trainer", {})
    model_config = config.get("model", {})
    data_config_path = config.get("data", {}).get("data_config_path", "")

    print("=" * 80 + "\n")

    seed = trainer_config.get("seed", 42)
    project_name = project_name or config.get("project_name", None)
    experiment_name = experiment_name or config.get("experiment_name", "experiment")
    experiment_name = experiment_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
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
        data_config_path=data_config_path,
    )

    model = create_model(model_config)

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
    """
    Initialize XGBoostCellCycleTrainer and split data for XGBoost workflow.

    Args:
        data_config: Data configuration dictionary
        wandb_run: Optional W&B run object

    Returns:
        trainer: XGBoostCellCycleTrainer instance
        train_df: Training DataFrame
        test_df: Test DataFrame
    """

    dataset = ModularCellFeaturesDataset(data_config=data_config)
    train_df, test_df = dataset.split_train_test_set()
    # Allow override
    train_val_split_ratio = data_config.get("train_val_split_ratio", 0.8)

    trainer = XGBoostCellCycleTrainer(
        training_data=train_df,
        wandb_run=wandb_run,
        train_val_split_ratio=train_val_split_ratio,
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

    Args:
        params: XGBoost hyperparameters (dict)
        trainer: XGBoostCellCycleTrainer instance
        wandb_run: W&B run object for logging
        model_suffix: Suffix to append to model filename (e.g., for sweeps)
        save_model: Whether to save the trained model to disk
        test_df: Optional test DataFrame for evaluation

    Returns:
        trainer: The fitted XGBoostCellCycleTrainer instance
        metrics: Validation metrics dictionary
        test_metrics: Test metrics dictionary (if test_df provided)
        model_path: Path to saved model (if saved)
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
        test_metrics = evaluate_regression(
            test_y,
            model.predict(test_X),
            prefix="test",
            plot=False,
            wandb_run=wandb_run,
        )
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
    """
    Run XGBoost training/validation/testing using unified logic.

    Args:
        config: Dictionary containing 'data_config' and 'model_config' sections
        project_name: W&B project name
    """

    data_config = config.get("data", {})
    model_config = config.get("model", {})
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
    """
    Run W&B hyperparameter sweep for XGBoost using unified logic for each trial.

    Args:
        config: Dictionary containing 'data' section
        sweep_config: W&B sweep configuration dictionary
        project_name: W&B project name
        count: Number of sweep trials to run
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
        with wandb.init(project=project_name, config=data_config) as run:
            # Use unified function
            train_and_evaluate_xgboost(
                params=run.config,
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


################################################################
######               Linear Regression part                ######
################################################################


def run_linear_regression(
    config: dict,
    sweep: bool = False,
    sweep_config: dict = None,
    project_name: str = None,
    count: int = None,
):
    """
    Train and evaluate a lasso regression model using sklearn, with optional W&B sweep.
    """
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from src.data_pipeline.dataset import ModularCellFeaturesDataset

    # Load training data
    data_config = config["data"]
    model_config = config.get("model", {})
    model_name = config.get("model_name")
    model = MODELS[model_name]

    count = count or sweep_config.get("count", 10) if sweep and sweep_config else None
    dataset = ModularCellFeaturesDataset(data_config=data_config)
    train_df, test_df = dataset.split_train_test_set()
    feature_names = [
        column for column in train_df.columns.tolist() if not column.startswith("label_")
    ]
    X, y = dataset.split_X_y(train_df)
    X_train, y_train, X_val, y_val = train_val_split(
        X, y, split_ratio=config.get("train_val_split_ratio", 0.8), random_state=42
    )

    def train_trial(trial_config, wandb_run, model=model):
        # Use config from sweep or default
        params = trial_config
        alpha = params.get("alpha")
        model = model(alpha=alpha)
        model.fit(X_train, y_train)
        print("Model:", model)
        val_preds = model.predict(X_val)
        train_preds = model.predict(X_train)
        metrics = {}
        train_metrics = evaluate_regression(
            y_train,
            train_preds,
            prefix="train",
            plot=False,
            wandb_run=wandb_run,
        )
        metrics.update(train_metrics)
        val_metrics = evaluate_regression(
            y_val,
            val_preds,
            prefix="val",
            plot=True,
            wandb_run=wandb_run,
        )
        metrics.update(val_metrics)
        importance = np.abs(model.coef_)
        importance_df = pd.DataFrame(
            data=importance,
            columns=feature_names,
            index=["488", "561"],
        )
        wandb.log({"feature_importance": wandb.Table(dataframe=importance_df)})
        print(f"{model_name} Regression Validation Results:")
        print(f"MSE: {metrics['val_mse']:.4f}")
        print(f"MAE: {metrics['val_mae']:.4f}")
        print(f"R2: {metrics['val_r2']:.4f}")

    if sweep and sweep_config is not None:
        # W&B sweep
        project_name = sweep_config.get("project") or project_name
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        print(f"Sweep initialized: {sweep_id}")
        print("Starting sweep trials...")

        def wandb_train():
            with wandb.init(project=project_name, config=data_config) as run:
                train_trial(run.config, wandb_run=run, model=model)

        wandb.agent(sweep_id, function=wandb_train, count=count or 10, project=project_name)
        print("SWEEP COMPLETE")
    else:
        wandb.init(project=project_name, config=config)
        train_trial(trial_config=model_config, wandb_run=wandb.run, model=model)


def main():
    """
    Entry point for the script.
    Parses command line arguments and dispatches to the selected backend.
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

    project_name = args.project
    experiment_name = args.experiment_name
    if args.backend == "lightning":
        run_lightning(config, project_name=project_name, experiment_name=experiment_name)
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
            run_xgboost(config, project_name)
    elif args.backend == "linear":
        if getattr(args, "tune", False):
            sweep_config = load_config(args.sweep_config)
            run_linear_regression(
                config,
                sweep=True,
                sweep_config=sweep_config,
                project_name=args.project or "linear-cell-cycle",
                count=args.count,
            )
        else:
            run_linear_regression(config)
    elif args.backend == "dataset_size_experiment":
        run_dataset_size_experiment(config, steps=args.steps)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
