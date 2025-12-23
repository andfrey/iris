"""
Centralized metrics computation for regression evaluation.

Provides unified metric computation for both standard and geodesic metrics.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from scipy.stats import circmean


@dataclass
class MetricResult:
    geodesic_mae: float
    geodesic_mse: float
    geodesic_rmse: float
    geodesic_r2: float
    mae: float = None  # Optional standard MAE
    mse: float = None
    rmse: float = None
    r2: float = None

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """
        Convert to dictionary with optional prefix.

        Args:
            prefix: Prefix for metric names (e.g., "train/", "val/")

        Returns:
            Dictionary of metrics
        """
        # Ensure prefix ends with / if not empty
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"

        metrics = {
            f"{prefix}geodesic_mae": self.geodesic_mae,
            f"{prefix}geodesic_mse": self.geodesic_mse,
            f"{prefix}geodesic_rmse": self.geodesic_rmse,
            f"{prefix}geodesic_r2": self.geodesic_r2,
            f"{prefix}mae": self.mae,
            f"{prefix}mse": self.mse,
            f"{prefix}rmse": self.rmse,
            f"{prefix}r2": self.r2,
        }
        return metrics


class MetricComputer:
    """Unified metric computation for both training and evaluation"""

    def __init__(self, projector=None):
        """
        Initialize metric computer.

        Args:
            projector: FucciCurveProjector for geodesic distance computation
        """
        self.projector = projector
        self.frechet_mean = None

    def compute(
        self, y_true: np.ndarray, y_pred: np.ndarray, space: str = None
    ) -> Union[MetricResult]:
        """
        Compute all relevant metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            space: Space of the values ("intensity" or "phase")

        Returns:
            MetricResult with all computed metrics
        """

        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must match")

        if space is None:
            if y_true.ndim == 1 or (y_true.ndim > 1 and y_true.shape[1] == 1):
                space = "phase"
            else:
                space = "intensity"

        if space == "intensity":
            y_pred_intensity = y_pred
            y_true_intensity = y_true
            y_pred = [self.projector.project(y_p) for y_p in y_pred_intensity]
            y_true = [self.projector.project(y_t) for y_t in y_true_intensity]
            y_pred_intensity = np.array(y_pred_intensity)
            y_true_intensity = np.array(y_true_intensity)
        else:
            y_pred = y_true + self._geodesic_distance(y_true, y_pred)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Standard metrics
        geodesic_mae = mean_absolute_error(y_true, y_pred)
        geodesic_mse = mean_squared_error(y_true, y_pred)
        geodesic_rmse = np.sqrt(geodesic_mse)
        geodesic_r2 = r2_score(y_true, y_pred)
        if space == "intensity":
            mae = mean_absolute_error(y_true_intensity, y_pred_intensity)
            mse = mean_squared_error(y_true_intensity, y_pred_intensity)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_intensity, y_pred_intensity)
            return MetricResult(
                mae=mae,
                mse=mse,
                rmse=rmse,
                r2=r2,
            )

        if space == "phase":
            return MetricResult(
                geodesic_mae=geodesic_mae,
                geodesic_mse=geodesic_mse,
                geodesic_rmse=geodesic_rmse,
                geodesic_r2=geodesic_r2,
            )

        return None

    def _geodesic_distance(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute geodesic distance between predictions and targets.

        Args:
            y_true: Ground truth phases
            y_pred: Predicted phases

        Returns:
            Array of geodesic distances
        """
        return np.asarray(
            [self.projector.geodesic_distance(y_p, y_t) for y_t, y_p in zip(y_true, y_pred)]
        )

    def geodesic_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute R² score based on geodesic distances.

        Args:
            y_true: Ground truth phases
            y_pred: Predicted phases

        Returns:
            R² score
        """
        geodesic_distances = self._geodesic_distance(y_true, y_pred)
        ss_res = np.sum(geodesic_distances**2)
        mean_true = self.get_frechet_mean(y_true)
        ss_tot = np.sum(self._geodesic_distance(y_true, mean_true) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        return r2

    def get_frechet_mean(self, phases: np.ndarray) -> float:
        """
        Compute the Fréchet mean of a set of phases on the Fucci curve.

        Args:
            phases: Array of phase values

        Returns:
            Fréchet mean phase
        """
        if self.frechet_mean is not None:
            return self.frechet_mean
        phases = phases.flatten()

        def objective(t):
            return sum(self.projector.geodesic_distance(ti, t) ** 2 for ti in phases)

        result = minimize(
            objective, x0=circmean(phases, high=1.0, low=0.0), method="L-BFGS-B", bounds=[(0, 1)]
        )
        # result = minimize(objective, x0=np.median(phases), method="L-BFGS-B", bounds=[(0, 1)])
        self.frechet_mean = result.x[0]
        return self.frechet_mean
