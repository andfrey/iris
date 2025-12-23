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

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_pipeline.dataset import ModularCellFeaturesDataset
from src.evaluation.evaluation import evaluate_regression
from src.train.utils import log_regression_plots
from src.evaluation.evaluator import Evaluator


class XGBoostCellCycleTrainer:
    """Trainer for XGBoost-based cell cycle prediction.

    Uses ModularCellFeaturesDataset for feature extraction.
    """

    def __init__(
        self,
        training_data: pd.DataFrame,
        val_data: pd.DataFrame,
        evaluator: Evaluator,
        wandb_run: Optional[wandb.Run] = None,
        dataset: ModularCellFeaturesDataset = None,
    ):
        """Initialize trainer with data configuration.

        Args:
            training_data: Training data as a pandas DataFrame
            val_data: Validation data as a pandas DataFrame
            wandb_run: Optional W&B run object for logging
            dataset: ModularCellFeaturesDataset for feature target splitting
        """
        self.training_data = training_data
        self.val_data = val_data
        self.dataset = dataset
        self.features_columns = [
            col for col in training_data.columns if not col.startswith("label_") or col != "phase"
        ]
        self.wandb_run = wandb_run
        self.evaluator = evaluator

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
        X_train, y_train = self.dataset.split_X_y(self.training_data)
        X_val, y_val = self.dataset.split_X_y(self.val_data)

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
        # self.log_feature_importance()
        return self.model, metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def evaluate(self) -> Dict[str, float]:
        metrics = {}

        train_evaluation = self.evaluator.evaluate(
            self.y_train.squeeze(),
            self.model.predict(self.X_train),
            prefix="train",
        )
        train_evaluation.log_to_wandb(wandb_run=self.wandb_run, prefix="train")
        metrics.update(train_evaluation.metrics.to_dict(prefix="train"))
        val_evaluation = self.evaluator.evaluate(
            self.y_val.squeeze(),
            self.model.predict(self.X_val),
            prefix="val",
        )
        val_evaluation.log_to_wandb(wandb_run=self.wandb_run, prefix="val")
        metrics.update(val_evaluation.metrics.to_dict(prefix="val"))
        for plot_name, fig in val_evaluation.plots.items():
            self.wandb_run.log({f"{plot_name}": wandb.Image(fig)})
            fig.clf()
        return metrics

    # def log_feature_importance(self):
    #     """Log feature importances using model's booster."""
    #     if self.model is None:
    #         print("Model not trained yet. Cannot log feature importances.")
    #         return
    #     importances = self.model.feature_importances_

    #     feature_names = self.features_columns

    #     importance_df = pd.DataFrame(
    #         {"feature": feature_names, "importance": importances}
    #     ).sort_values("importance", ascending=False)
    #     print("Feature importances:")
    #     print(importance_df)
    #     if self.wandb_run is not None:
    #         self.wandb_run.log({"feature_importance": wandb.Table(dataframe=importance_df)})

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
