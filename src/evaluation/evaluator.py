"""
Main evaluator orchestrating metrics computation and plot generation.

Provides unified evaluation interface for both training and checkpoint evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.metrics import MetricResult, MetricComputer
from src.evaluation.plots import PlotGenerator, PlotConfig
from src.data_pipeline.curve_projector import FucciCurveProjector


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    is_phase: bool = False
    plot_config: PlotConfig = field(default_factory=PlotConfig)
    plots_to_generate: List[str] = field(
        default_factory=lambda: [
            "scatter",
            "residual",
        ]
    )

    def __post_init__(self):
        """Validate configuration"""
        valid_plots = {
            "scatter",
            "residual",
            "error_heatmap",
            "true_residuals",
            "fucci_color_comparison",
            "geodesic_residual",
        }

        for plot_type in self.plots_to_generate:
            if plot_type not in valid_plots:
                raise ValueError(f"Unknown plot type: {plot_type}. Valid: {valid_plots}")


@dataclass
class EvaluationResult:
    """Complete evaluation result with metrics and plots"""

    metrics: MetricResult
    plots: Dict[str, plt.Figure]

    def log_to_wandb(self, wandb_run, prefix: str = ""):
        """
        Log metrics and plots to Weights & Biases.

        Args:
            wandb_run: W&B run object (from trainer.logger.experiment)
            prefix: Prefix for metric/plot names (e.g., "train/", "val/")
        """
        import wandb

        # Ensure prefix ends with / if not empty
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"

        # Log metrics
        wandb_run.log(self.metrics.to_dict(prefix=prefix))

        # Log plots
        for plot_name, fig in self.plots.items():
            wandb_run.log({f"{prefix}{plot_name}": wandb.Image(fig)})
            plt.close(fig)

    def save_plots(self, save_dir, prefix: str = ""):
        """
        Save all plots to directory.

        Args:
            save_dir: Path to directory for saving plots
            prefix: Prefix for filenames
        """
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for plot_name, fig in self.plots.items():
            filename = f"{prefix}{plot_name}.png" if prefix else f"{plot_name}.png"
            fig.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)


class Evaluator:
    """Main evaluator for regression models"""

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        projector: Optional[FucciCurveProjector] = None,
    ):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
            projector: FucciCurveProjector for geodesic computations (required if is_phase=True)
        """
        self.config = config or EvaluationConfig()
        self.projector = projector

        # Initialize metric computer
        self.metric_computer = MetricComputer(projector=projector)

        # Initialize plot generator
        self.plot_generator = PlotGenerator(self.config.plot_config)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "",
    ) -> EvaluationResult:
        """
        Perform complete evaluation with metrics and plots.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            prefix: Prefix for metric/plot names (e.g., "train", "val", "test")

        Returns:
            EvaluationResult with metrics and plots
        """

        # Compute metrics
        if y_true.shape == y_pred.shape:
            metrics = self.metric_computer.compute(y_true, y_pred)

        # Filter out negative values for plotting if 2D intensities
        y_true, y_pred = self._filter_data(y_true, y_pred)

        # Generate plots
        plots = self._generate_plots(y_true, y_pred, prefix)

        return EvaluationResult(metrics=metrics, plots=plots)

    def _filter_data(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """Filter out invalid data points (negative intensities for 2D targets)"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Only filter for 2D targets (FUCCI intensities)
        if y_true.ndim > 1 and y_true.shape[1] == 2:
            mask = (y_true[:, 0] >= 0) & (y_true[:, 1] >= 0)
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        return y_true, y_pred

    def _generate_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str,
        residuals: Optional[np.ndarray] = None,
    ) -> Dict[str, plt.Figure]:
        """Generate all configured plots"""
        plots = {}

        # Determine data dimensionality
        y_true_arr = np.asarray(y_true)
        is_2d = y_true_arr.ndim > 1 and y_true_arr.shape[1] == 2

        if residuals is None:
            residuals = [
                self.projector.geodesic_distance(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)
            ]

        for plot_type in self.config.plots_to_generate:
            if plot_type == "scatter":
                plots[f"scatter"] = self.plot_generator.scatter_plot(
                    y_true, y_pred, residuals=residuals, title=f"{prefix} Predictions vs True"
                )

            elif plot_type == "residual":
                plots[f"residual"] = self.plot_generator.residual_plot(
                    y_true, y_pred, title=f"{prefix} Residual Plot"
                )

            elif plot_type == "error_heatmap" and is_2d:
                plots[f"error_heatmap"] = self.plot_generator.error_heatmap(
                    y_true, y_pred, title=f"{prefix} Error Heatmap"
                )

            elif plot_type == "true_residuals" and is_2d:
                plots[f"true_residuals"] = self.plot_generator.true_residuals_plot(
                    y_true=y_true,
                    residuals=residuals,
                    y_pred=y_pred,
                    title=f"{prefix} True Residuals Diagnostics",
                )

            elif plot_type == "fucci_color_comparison" and is_2d:
                plots[f"fucci_color_comparison"] = self.plot_generator.fucci_color_comparison(
                    y_true, y_pred, title=f"{prefix} FUCCI Color Distribution"
                )

            elif plot_type == "geodesic_residual" and self.config.is_phase:
                plots[f"geodesic_residual"] = self.plot_generator.geodesic_residual_plot(
                    y_true,
                    y_pred,
                    residuals=residuals,
                    title=f"{prefix} Geodesic Residual Plot",
                )

        return plots
