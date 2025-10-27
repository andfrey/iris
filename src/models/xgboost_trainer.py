"""XGBoost trainer for cell cycle prediction.

This module provides XGBoost model training/evaluation with W&B integration
using the ModularCellFeaturesDataset for feature extraction.
"""

from pathlib import Path
import sys
import numpy as np
from xgboost import XGBRegressor
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yaml

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_pipeline.dataset import ModularCellFeaturesDataset
from src.train.utils import evaluate_regression, train_val_split


class XGBoostCellCycleTrainer:
    """Trainer for XGBoost-based cell cycle prediction.

    Uses ModularCellFeaturesDataset for feature extraction.
    """

    def __init__(
        self,
        training_data: pd.DataFrame,
        train_val_split_ratio: float = 0.8,
        wandb_run: Optional[wandb.Run] = None,
        fucci_scalers: np.ndarray = None,
    ):
        """Initialize trainer with data configuration.

        Args:
            data_config_path: Path to data configuration YAML
            use_wandb: Whether to log to Weights & Biases
        """
        self.training_data = training_data
        self.features_columns = [
            col for col in training_data.columns if not col.startswith("label_")
        ]
        self.wandb_run = wandb_run
        self.fucci_scalers = fucci_scalers
        if self.fucci_scalers is None:
            self.fucci_scalers = np.array(1.0, 1.0)
        self.train_val_split_ratio = train_val_split_ratio
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.prepare_dataset()

    def set_wandb_run(self, wandb_run: wandb.Run):
        """Set the W&B run for logging.

        Args:
            wandb_run: W&B run object
        """
        self.wandb_run = wandb_run

    def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        X_train, y_train = ModularCellFeaturesDataset.split_X_y(self.training_data)
        X_train, y_train, X_val, y_val = train_val_split(
            X_train, y_train, self.train_val_split_ratio, random_state=42
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        print(
            f"Prepared dataset: {X_train.shape} training samples, {X_val.shape} validation samples"
        )

    def train(
        self,
        params: Optional[Dict[str, Any]] = None,
    ) -> XGBRegressor:
        """Train XGBoost model.

        Args:
            params: XGBoost hyperparameters (optional)

        Returns:
            Trained XGBoost Booster
        """

        # Train
        self.model = XGBRegressor(
            **params,
        )
        self.model.fit(
            self.X_train,
            self.y_train,
        )
        # Evaluate and log metrics
        metrics = self.evaluate()
        self.log_feature_importance()
        return self.model, metrics

    def evaluate(self) -> Dict[str, float]:
        metrics = {}

        train_metrics = evaluate_regression(
            self.y_train * self.fucci_scalers,
            self.model.predict(self.X_train) * self.fucci_scalers,
            prefix="train",
            wandb_run=self.wandb_run,
            plot=False,
        )
        metrics.update(train_metrics)
        val_metrics = evaluate_regression(
            self.y_val * self.fucci_scalers,
            self.model.predict(self.X_val) * self.fucci_scalers,
            prefix="val",
            wandb_run=self.wandb_run,
            plot=True,
        )
        metrics.update(val_metrics)
        return metrics

    def log_feature_importance(self):
        """Log feature importances using model's booster."""
        if self.model is None:
            print("Model not trained yet. Cannot log feature importances.")
            return
        importances = self.model.feature_importances_

        feature_names = self.features_columns

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        print("Feature importances:")
        print(importance_df)
        if self.wandb_run is not None:
            self.wandb_run.log({"feature_importance": wandb.Table(dataframe=importance_df)})

    def save_model(self, path: str):
        """Save trained model to disk.

        Args:
            path: File path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        self.model = XGBRegressor()
        self.model.load_model(path)
        print(f"Model loaded from {path}")
