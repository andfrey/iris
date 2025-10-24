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
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.dataset import ModularCellFeaturesDataset


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
        train_size = int(len(self.training_data) * self.train_val_split_ratio)
        train_data = self.training_data.iloc[:train_size]
        val_data = self.training_data.iloc[train_size:]

        X_train, y_train = ModularCellFeaturesDataset.split_X_y(train_data)
        X_val, y_val = ModularCellFeaturesDataset.split_X_y(val_data)

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
        # Log feature importances
        self.log_feature_importance()
        # log error scatter plot
        self.plot_error_scatter()
        return self.model, metrics

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

    def plot_error_scatter(self, X=None, y=None, title="Error Scatter Plot", x_axis=0, y_axis=1):
        """
        Visualize squared error on a 2D scatter plot.
        Args:
            X: Features (default: validation set)
            y: True labels (default: validation set)
            title: Plot title
            x_axis: Feature index for x-axis
            y_axis: Feature index for y-axis
        """
        import matplotlib.pyplot as plt

        if X is None:
            X = self.X_val
        if y is None:
            y = self.y_val
        X = X
        y = y * self.fucci_scalers
        preds = self.model.predict(X) * self.fucci_scalers
        errors = np.linalg.norm(preds - y, axis=1)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(np.log(y[:, 0]), np.log(y[:, 1]), c=errors, cmap="Blues", alpha=0.7)
        plt.colorbar(sc, label="Norm Error")
        plt.xlabel(f"Log 488 Intensity")
        plt.ylabel(f"Log 561 Intensity")
        plt.title(title)
        plt.tight_layout()
        plt.show()
        if self.wandb_run is not None:
            self.wandb_run.log({"error_scatter": wandb.Image(plt)})

    def evaluate(
        self, X: np.ndarray = None, y: np.ndarray = None, prefix: str = "train"
    ) -> Dict[str, float]:
        """Evaluate model on a dataset.

        Args:
            X: Features
            y: Labels
            prefix: Metric prefix (e.g., 'test', 'val')

        Returns:
            Dictionary of metrics
        """
        if prefix != "train" and (X is None or y is None):
            raise ValueError("X and y must be provided for non-training evaluation")

        if self.model is None:
            raise ValueError("Model not trained yet")

        if prefix == "train":
            val_preds = self.model.predict(self.X_val) * self.fucci_scalers
            train_preds = self.model.predict(self.X_train) * self.fucci_scalers
            val_true = self.y_val * self.fucci_scalers
            train_true = self.y_train * self.fucci_scalers
            val_mse = mean_squared_error(val_true, val_preds)
            train_mse = mean_squared_error(train_true, train_preds)
            val_mae = mean_absolute_error(val_true, val_preds)
            train_mae = mean_absolute_error(train_true, train_preds)
            val_r2 = r2_score(val_true, val_preds)
            train_r2 = r2_score(train_true, train_preds)

            metrics = {
                f"val_mse": val_mse,
                f"train_mse": train_mse,
                f"val_mae": val_mae,
                f"train_mae": train_mae,
                f"val_r2": val_r2,
                f"train_r2": train_r2,
            }
        else:
            y = y * self.fucci_scalers
            preds = self.model.predict(X) * self.fucci_scalers
            mse = mean_squared_error(y, preds)
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)

            metrics = {
                f"{prefix}_mse": mse,
                f"{prefix}_mae": mae,
                f"{prefix}_r2": r2,
            }
        if self.wandb_run is not None:
            self.wandb_run.log(metrics)

        return metrics

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
