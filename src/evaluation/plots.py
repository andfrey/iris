"""
Plot generation for regression evaluation.

Provides all visualization functions for metrics, residuals, and comparisons.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


@dataclass
class PlotConfig:
    """Configuration for plots"""

    figsize: tuple = (10, 8)
    dpi: int = 100
    title_fontsize: int = 14
    label_fontsize: int = 12
    alpha: float = 0.6


class PlotGenerator:
    """Generate all types of plots for regression evaluation"""

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plot generator.

        Args:
            config: Plot configuration (uses defaults if None)
        """
        self.config = config or PlotConfig()

    def scatter_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        title: str = "Predictions vs True",
        **kwargs,
    ) -> plt.Figure:
        """
        Create scatter plot of predictions vs true values.

        Handles both 1D and 2D targets automatically.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            title: Plot title
            **kwargs: Additional plot arguments
            residuals: Optional residuals for coloring or sizing points

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Determine if 2D or 1D
        is_2d = y_true.ndim > 1 and y_true.shape[1] == 2

        if is_2d:
            # 2D scatter (e.g., FUCCI intensities)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.config.dpi)

            # Channel 0
            ax1.scatter(
                y_true[:, 0],
                y_pred[:, 0],
                alpha=self.config.alpha,
                s=20,
                label="Channel 0",
            )
            ax1.plot(
                [y_true[:, 0].min(), y_true[:, 0].max()],
                [y_true[:, 0].min(), y_true[:, 0].max()],
                "r--",
                lw=2,
                label="Perfect prediction",
            )
            ax1.set_xlabel("True Values", fontsize=self.config.label_fontsize)
            ax1.set_ylabel("Predicted Values", fontsize=self.config.label_fontsize)
            ax1.set_title(f"{title} - Channel 0", fontsize=self.config.title_fontsize)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Channel 1
            ax2.scatter(
                y_true[:, 1],
                y_pred[:, 1],
                alpha=self.config.alpha,
                s=20,
                label="Channel 1",
                color="green",
            )
            ax2.plot(
                [y_true[:, 1].min(), y_true[:, 1].max()],
                [y_true[:, 1].min(), y_true[:, 1].max()],
                "r--",
                lw=2,
                label="Perfect prediction",
            )
            ax2.set_xlabel("True Values", fontsize=self.config.label_fontsize)
            ax2.set_ylabel("Predicted Values", fontsize=self.config.label_fontsize)
            ax2.set_title(f"{title} - Channel 1", fontsize=self.config.title_fontsize)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

        else:
            # 1D scatter
            if y_true.ndim > 1:
                y_true = y_true.flatten()
                y_pred = y_pred.flatten()

            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            scatter = ax.scatter(
                y_true,
                y_pred,
                alpha=self.config.alpha,
                s=20,
                c=np.abs(residuals),
                cmap="Blues",
            )
            ax.plot(
                [y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                "r--",
                lw=2,
                label="Perfect prediction",
            )
            plt.colorbar(scatter, label="ABS Residuals", ax=ax)
            ax.set_xlabel("True Values", fontsize=self.config.label_fontsize)
            ax.set_ylabel("Predicted Values", fontsize=self.config.label_fontsize)
            ax.set_title(title, fontsize=self.config.title_fontsize)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

        return fig

    def residual_plot(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residual Plot", **kwargs
    ) -> plt.Figure:
        """
        Create residual plot (residuals vs predictions).

        Handles both 1D and 2D targets.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            title: Plot title
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.config.dpi)

        # Channel 0
        axes[0, 0].scatter(y_pred[:, 0], residuals[:, 0], alpha=self.config.alpha, s=20)
        axes[0, 0].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0, 0].set_xlabel("Predicted Values", fontsize=self.config.label_fontsize)
        axes[0, 0].set_ylabel("Residuals", fontsize=self.config.label_fontsize)
        axes[0, 0].set_title(f"{title} - Channel 0", fontsize=self.config.title_fontsize)
        axes[0, 0].grid(True, alpha=0.3)

        # Channel 1
        axes[0, 1].scatter(
            y_pred[:, 1],
            residuals[:, 1],
            alpha=self.config.alpha,
            s=20,
            color="green",
        )
        axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[0, 1].set_xlabel("Predicted Values", fontsize=self.config.label_fontsize)
        axes[0, 1].set_ylabel("Residuals", fontsize=self.config.label_fontsize)
        axes[0, 1].set_title(f"{title} - Channel 1", fontsize=self.config.title_fontsize)
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram - Channel 0
        axes[1, 0].hist(residuals[:, 0], bins=50, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[1, 0].set_xlabel("Residuals", fontsize=self.config.label_fontsize)
        axes[1, 0].set_ylabel("Frequency", fontsize=self.config.label_fontsize)
        axes[1, 0].set_title("Residual Distribution - Channel 0")
        axes[1, 0].grid(True, alpha=0.3)

        # Histogram - Channel 1
        axes[1, 1].hist(residuals[:, 1], bins=50, alpha=0.7, color="green", edgecolor="black")
        axes[1, 1].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[1, 1].set_xlabel("Residuals", fontsize=self.config.label_fontsize)
        axes[1, 1].set_ylabel("Frequency", fontsize=self.config.label_fontsize)
        axes[1, 1].set_title("Residual Distribution - Channel 1")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    def error_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: int = 50,
        title: str = "Error Heatmap",
        **kwargs,
    ) -> plt.Figure:
        """
        Create 2D error heatmap (for 2D targets like FUCCI intensities).

        Args:
            y_true: Ground truth values (Nx2)
            y_pred: Predicted values (Nx2)
            bins: Number of bins for heatmap
            title: Plot title
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape[1] != 2:
            raise ValueError("error_heatmap requires 2D targets")

        errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        # Create 2D histogram
        h = ax.hist2d(
            y_true[:, 0],
            y_true[:, 1],
            bins=bins,
            weights=errors,
            cmap="YlOrRd",
        )

        plt.colorbar(h[3], ax=ax, label="Mean Error")
        ax.set_xlabel("True Channel 0", fontsize=self.config.label_fontsize)
        ax.set_ylabel("True Channel 1", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)

        plt.tight_layout()
        return fig

    def true_residuals_plot(
        self,
        y_true: np.ndarray,
        residuals: np.ndarray,
        y_pred: np.ndarray = None,
        title: str = "True Residuals Diagnostics",
        **kwargs,
    ) -> plt.Figure:
        """
        Create diagnostic plot of residuals vs true values (for 2D targets).

        Args:
            y_true: Ground truth values (Nx2)
            residuals: Residuals (Nx2)
            y_pred: Predicted values (Nx2), optional
            title: Plot title
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred) if y_pred is not None else None
        residuals = np.asarray(residuals)
        if y_true.shape[1] != 2:
            raise ValueError("true_residuals_plot requires 2D targets")
        if residuals.ndim > 1 and residuals.shape[1] == 2:
            residuals = np.linalg.norm(residuals, axis=1)

        fig, axes = plt.subplots(
            1, 2 if y_pred is not None else 1, figsize=(14, 12), dpi=self.config.dpi
        )

        # Residuals vs True - Channel 0
        ax = axes[0] if y_pred is not None else axes
        scatter = ax.scatter(
            y_true[:, 0], y_true[:, 1], c=residuals, alpha=self.config.alpha, s=20, cmap="Blues"
        )
        ax.set_xlabel("FUCCI 488nm", fontsize=self.config.label_fontsize)
        ax.set_ylabel("FUCCI 561nm", fontsize=self.config.label_fontsize)
        ax.set_title("Errors on true FUCCI intensities")
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, label="Residual Magnitude")
        if y_pred is not None:
            # Residuals vs True - Channel 1
            axes[1].scatter(
                y_pred[:, 0],
                y_pred[:, 1],
                c=residuals,
                alpha=self.config.alpha,
                s=20,
                cmap="Blues",
            )
            axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
            axes[1].set_xlabel("FUCCI 561nm", fontsize=self.config.label_fontsize)
            axes[1].set_ylabel("Residuals", fontsize=self.config.label_fontsize)
            axes[1].set_title("Errors on predicted FUCCI intensities")
            axes[1].grid(True, alpha=0.3)
            axes[1].colorbar(label="Residual Magnitude")

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2)
        plt.tight_layout()
        return fig

    def fucci_color_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "FUCCI Color Distribution",
        **kwargs,
    ) -> plt.Figure:
        """
        Visualize FUCCI intensities as colors (for 2D FUCCI targets).

        Args:
            y_true: Ground truth FUCCI intensities (Nx2)
            y_pred: Predicted FUCCI intensities (Nx2)
            title: Plot title
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape[1] != 2:
            raise ValueError("fucci_color_comparison requires 2D targets")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.config.dpi)

        # Normalize for color mapping
        norm_true = Normalize(vmin=0, vmax=max(y_true.max(), y_pred.max()))
        norm_pred = Normalize(vmin=0, vmax=max(y_true.max(), y_pred.max()))

        # True values
        colors_true = plt.cm.RdYlGn(norm_true(y_true[:, 0]))
        ax1.scatter(
            y_true[:, 0],
            y_true[:, 1],
            c=colors_true,
            alpha=self.config.alpha,
            s=30,
        )
        ax1.set_xlabel("Red Intensity", fontsize=self.config.label_fontsize)
        ax1.set_ylabel("Green Intensity", fontsize=self.config.label_fontsize)
        ax1.set_title(f"{title} - True", fontsize=self.config.title_fontsize)
        ax1.grid(True, alpha=0.3)

        # Predicted values
        colors_pred = plt.cm.RdYlGn(norm_pred(y_pred[:, 0]))
        ax2.scatter(
            y_pred[:, 0],
            y_pred[:, 1],
            c=colors_pred,
            alpha=self.config.alpha,
            s=30,
        )
        ax2.set_xlabel("Red Intensity", fontsize=self.config.label_fontsize)
        ax2.set_ylabel("Green Intensity", fontsize=self.config.label_fontsize)
        ax2.set_title(f"{title} - Predicted", fontsize=self.config.title_fontsize)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def geodesic_residual_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        title: str = "Geodesic Residual Plot",
        **kwargs,
    ) -> plt.Figure:
        """
        Create residual plot using geodesic distance (for circular/phase data).

        Args:
            y_true: Ground truth phase values
            y_pred: Predicted phase values
            geodesic_distance_fn: Function to compute geodesic distance
            title: Plot title
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib figure
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.config.dpi)

        # Residuals vs predictions
        ax1.scatter(y_pred, residuals, alpha=self.config.alpha, s=20)
        ax1.axhline(y=0, color="r", linestyle="--", lw=2)
        ax1.set_xlabel("Predicted Phase", fontsize=self.config.label_fontsize)
        ax1.set_ylabel("Geodesic Distance", fontsize=self.config.label_fontsize)
        ax1.set_title(f"{title} - vs Predictions", fontsize=self.config.title_fontsize)
        ax1.grid(True, alpha=0.3)

        # Histogram of residuals
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(x=0, color="r", linestyle="--", lw=2)
        ax2.set_xlabel("Geodesic Distance", fontsize=self.config.label_fontsize)
        ax2.set_ylabel("Frequency", fontsize=self.config.label_fontsize)
        ax2.set_title(f"{title} - Distribution", fontsize=self.config.title_fontsize)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
